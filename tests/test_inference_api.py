from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pbg.core.checkpointing import CHECKPOINT_SCHEMA_VERSION
from pbg.core.storage import TRAINING_STORAGE_SCHEMA_VERSION
from pbg.games.poker.adapter import PokerModelBundle
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet
from pbg.games.poker.rules import PokerConfig
from pbg.inference import PokerInferenceSession, parse_card
from pbg.inference_artifact import (
    INFERENCE_ARTIFACT_SCHEMA_VERSION,
    export_inference_artifact,
    load_inference_source,
)


def _scientific(n_players: int = 2) -> dict:
    return {
        "game": {"n_players": n_players, "stack_bb": 8.0, "rank_count": 4},
        "model": {
            "d_model": 16,
            "n_heads": 4,
            "n_layers_pre": 1,
            "n_layers_post": 1,
            "m_raise": 2,
            "k_act": 5,
            "max_players": n_players,
            "class_embedding_dim": 4,
            "ffn_hidden": 24,
            "folder_beta_rank": 0,
            "use_branching_heads": True,
            "max_history_tokens": 18,
        },
    }


def _artifact(path: Path, n_players: int = 2) -> Path:
    scientific = _scientific(n_players)
    torch.manual_seed(401)
    cfg = PokerModelConfig.from_dict(scientific["model"])
    geometry = PokerConfig(n_players=n_players, stack_bb=8.0, rank_count=4).geometry
    model = PokerModelBundle(PreflopNet(cfg, geometry), PostflopNet(cfg, geometry))
    torch.save(
        {
            "inference_artifact_schema_version": INFERENCE_ARTIFACT_SCHEMA_VERSION,
            "source_checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "source_checkpoint_sha256": "1" * 64,
            "iteration": 7,
            "config": scientific,
            "model": model.state_dict(),
        },
        path,
    )
    return path


def _current_checkpoint(model: torch.nn.Module, config: dict, iteration: int) -> dict:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "storage_schema_version": TRAINING_STORAGE_SCHEMA_VERSION,
        "controller_schema_version": None,
        "branching_controller_schema_version": None,
        "cost_safeguard_controller_schema_version": None,
        "model": model.state_dict(),
        "optimizer": {},
        "scaler": {},
        "iteration": iteration,
        "config": config,
        "hardware_profile": None,
        "torch_generator": torch.Generator().get_state(),
        "numpy_generator": {},
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": None,
        "run_id": None,
        "rank_rng_states": None,
        "reference_model": None,
        "reference_iteration": None,
        "reference_effective_hands_rollout": 0.0,
        "adaptive_work_state": None,
        "branching_controller_state": None,
        "cost_safeguard_controller_state": None,
        "probe_target": None,
        "logical_work": None,
        "initialization": None,
    }


def test_inference_artifact_export_is_model_only(tmp_path: Path) -> None:
    scientific = _scientific()
    cfg = PokerModelConfig.from_dict(scientific["model"])
    geometry = PokerConfig(n_players=2, stack_bb=8.0, rank_count=4).geometry
    model = PokerModelBundle(PreflopNet(cfg, geometry), PostflopNet(cfg, geometry))
    checkpoint = tmp_path / "checkpoint.pt"
    full_config = {
        **scientific,
        "training": {"iters": 11},
        "loss": {"name": "rb_ppo"},
    }
    payload = _current_checkpoint(model, full_config, 11)
    payload["optimizer"] = {"large": torch.ones(32)}
    torch.save(payload, checkpoint)
    output = tmp_path / "policy.pt"
    report = export_inference_artifact(checkpoint, output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert report["schema_version"] == INFERENCE_ARTIFACT_SCHEMA_VERSION
    assert payload["iteration"] == 11
    assert "optimizer" not in payload
    assert set(payload) == {
        "inference_artifact_schema_version",
        "source_checkpoint_schema_version",
        "source_checkpoint_sha256",
        "iteration",
        "config",
        "model",
    }
    normalized = load_inference_source(output)
    assert normalized["iteration"] == 11
    assert normalized["config"] == scientific


def test_inference_artifact_rejects_nonhexadecimal_source_hash(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "policy.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["source_checkpoint_sha256"] = "z" * 64
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="hash is invalid"):
        load_inference_source(path)


def test_card_parser_uses_global_compact_deck_ids() -> None:
    assert parse_card("2c") == 0
    assert parse_card("As") == 51
    assert parse_card(51) == 51
    with pytest.raises(ValueError, match="active deck"):
        parse_card("Tc", first_card=36)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stateful_singleton_and_batch_inference_contract(tmp_path: Path) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt"), device="cuda"
    )
    first, second = session.new_hands([0, 1])
    singleton = session.predict(first, ["As", "Kd"], action_mode="argmax")
    assert singleton.selected_action in singleton.actions
    assert singleton.probabilities.shape == (len(singleton.actions),)
    torch.testing.assert_close(singleton.probabilities.sum(), torch.tensor(1.0))

    batch = session.predict_many([first, second], [["As", "Kd"], ["Qh", "Js"]])
    torch.testing.assert_close(batch[0].probabilities, singleton.probabilities)
    assert [result.hand for result in batch] == [first, second]

    session.advance(second, "call")
    mixed = session.predict_many([second, first], [["Qh", "Js"], ["As", "Kd"]])
    assert [result.hand for result in mixed] == [second, first]
    torch.testing.assert_close(
        mixed[0].probabilities,
        session.predict(second, ["Qh", "Js"]).probabilities,
    )

    full = session.full_strategy([first, second])
    combo = singleton.private_hand_id
    columns = full.legal_mask[0].nonzero(as_tuple=False).squeeze(1)
    torch.testing.assert_close(
        full.probabilities[0, combo, columns], singleton.probabilities
    )
    assert full.probabilities.shape == (2, 120, session.backend.e_max)
    assert full.values_bb.shape == (2, 120)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stateful_actions_ranges_and_postflop_neutral_summary(tmp_path: Path) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt"), device="cuda"
    )
    hand = session.new_hand()
    before = session._state(hand).log_reach.clone()
    session.advance(hand, "call")
    assert not torch.equal(before, session._state(hand).log_reach)
    session.advance(hand, "check")
    with pytest.raises(ValueError, match="chance node"):
        session.predict(hand, ["As", "Kd"])
    chance_ref = session._state(hand).ref
    chance_reach = session._state(hand).log_reach.clone()
    with pytest.raises(ValueError, match="unique"):
        session.deal(hand, ["Jh", "Jh", "Kc"])
    assert session._state(hand).ref == chance_ref
    assert torch.equal(session._state(hand).log_reach, chance_reach)
    session.deal(hand, ["Jh", "Qd", "Kc"])
    result = session.predict(hand, ["As", "Ad"])
    torch.testing.assert_close(result.probabilities.sum(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="conflicts"):
        session.predict(hand, ["Jh", "Ac"])


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_mps_model_and_outputs_remain_on_metal(tmp_path: Path) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt"), device="mps", precision="fp32"
    )
    assert all(
        parameter.device.type == "mps" for parameter in session.model.parameters()
    )
    hand = session.new_hand()
    probability, value, seat_value, _, _ = session._evaluate_full_device([hand])
    assert probability.device.type == value.device.type == "mps"
    assert seat_value is None or seat_value.device.type == "mps"
    actual = session.predict(hand, ["As", "Kd"])
    assert torch.isfinite(actual.probabilities).all()
    torch.testing.assert_close(actual.probabilities.sum(), torch.tensor(1.0))


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
@pytest.mark.parametrize(
    ("precision", "rtol", "atol"),
    [("fp32", 3e-4, 3e-4), ("fp16", 3e-3, 3e-3)],
)
def test_mps_postflop_and_batched_parity(
    tmp_path: Path, precision: str, rtol: float, atol: float
) -> None:
    artifact = _artifact(tmp_path / "policy.pt")
    metal = PokerInferenceSession.from_checkpoint(
        artifact, device="mps", precision=precision, mode="throughput"
    )
    metal_hands = metal.new_hands([0, 1])
    metal.advance_many(metal_hands, ["call", "call"])
    metal.advance_many(metal_hands, ["check", "check"])
    metal.deal_many(metal_hands, [["Jh", "Qd", "Kc"], ["Jd", "Qh", "Ks"]])
    private = [["As", "Ad"], ["Ac", "Kh"]]
    actual = metal.predict_many(metal_hands, private)
    for left in actual:
        assert torch.isfinite(left.probabilities).all()
        torch.testing.assert_close(
            left.probabilities.sum(), torch.tensor(1.0), rtol=rtol, atol=atol
        )
