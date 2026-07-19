from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pbg.core.checkpointing import CHECKPOINT_SCHEMA_VERSION
from pbg.core.storage import TRAINING_STORAGE_SCHEMA_VERSION
from pbg.games.poker.bundle import build_model_from_config
from pbg.games.poker.folded import PartitionSummaryBatch
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
            "use_moment_heads": True,
        },
    }


def _artifact(path: Path, n_players: int = 2) -> Path:
    scientific = _scientific(n_players)
    torch.manual_seed(401)
    model = build_model_from_config(scientific, torch.device("cpu"))
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


def test_inference_artifact_export_is_model_only(tmp_path: Path) -> None:
    scientific = _scientific()
    model = build_model_from_config(scientific, torch.device("cpu"))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "storage_schema_version": TRAINING_STORAGE_SCHEMA_VERSION,
            "controller_schema_version": None,
            "model": model.state_dict(),
            "optimizer": {"large": torch.ones(32)},
            "iteration": 11,
            "config": scientific,
        },
        checkpoint,
    )
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


def test_card_parser_uses_global_compact_deck_ids() -> None:
    assert parse_card("2c") == 0
    assert parse_card("As") == 51
    assert parse_card(51) == 51
    with pytest.raises(ValueError, match="active deck"):
        parse_card("Tc", first_card=36)


def test_stateful_singleton_and_batch_inference_contract(tmp_path: Path) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt"), device="cpu"
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
    assert full.probabilities.shape == (2, 120, session.engine.e_max)
    assert full.values_bb.shape == (2, 120)


def test_stateful_actions_ranges_and_postflop_neutral_summary(tmp_path: Path) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt"), device="cpu"
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


def test_multiway_fold_summary_stays_on_cpu_before_model_transfer(
    tmp_path: Path,
) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt", n_players=3), device="cpu"
    )

    class FoldedCPU:
        called = 0

        def summarize(self, *, board_cards, live_range_weights, folded):
            self.called += 1
            assert board_cards.device.type == "cpu"
            assert live_range_weights.device.type == "cpu"
            assert folded.reach_weights.device.type == "cpu"
            batch, folders = folded.folder_mask.shape
            cards = session.game_config.geometry.n_cards
            fields = {
                "base_log_z": torch.zeros(batch),
                "single_card_log_ratio": torch.zeros(batch, cards),
                "pair_card_log_ratio": torch.zeros(batch, cards, cards),
                "beta_eigenvalue": torch.empty(batch, 0),
                "beta_eigenvector": torch.empty(batch, cards, 0),
                "folder_log_mass": torch.zeros(batch, folders),
                "folder_single_card_log_ratio": torch.zeros(batch, folders, cards),
                "folder_card_marginal": torch.zeros(batch, folders, cards),
                "folder_delta": torch.zeros(batch, folders, cards),
                "tight_cluster_scalar": torch.zeros(batch),
                "folder_output_valid": folded.folder_mask.clone(),
                "output_valid": torch.ones(batch, dtype=torch.bool),
                "folder_mask": folded.folder_mask.clone(),
                "folder_line_meta": folded.line_meta.clone(),
                "folder_seat": folded.seat_ids.clone(),
            }
            return PartitionSummaryBatch(fields, {"backend": "cpu_test"})

    folded_cpu = FoldedCPU()
    session.partition_adapter = folded_cpu
    hand = session.new_hand()
    session.advance(hand, "call")
    session.advance(hand, "fold")
    session.advance(hand, "check")
    session.deal(hand, ["Jh", "Qd", "Kc"])
    result = session.predict(hand, ["As", "Ad"])
    assert folded_cpu.called == 1
    assert torch.isfinite(result.probabilities).all()
    # Same board/fold dependency must use the cached CPU summary.
    session.predict(hand, ["As", "Ad"])
    assert folded_cpu.called == 1


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
    cpu = PokerInferenceSession.from_checkpoint(
        tmp_path / "policy.pt", device="cpu", precision="fp32"
    )
    cpu_hand = cpu.new_hand()
    expected = cpu.predict(cpu_hand, ["As", "Kd"])
    actual = session.predict(hand, ["As", "Kd"])
    torch.testing.assert_close(
        actual.probabilities, expected.probabilities, rtol=3e-4, atol=3e-4
    )


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
    cpu = PokerInferenceSession.from_checkpoint(artifact, device="cpu")
    metal = PokerInferenceSession.from_checkpoint(
        artifact, device="mps", precision=precision, mode="throughput"
    )
    cpu_hands = cpu.new_hands([0, 1])
    metal_hands = metal.new_hands([0, 1])
    for session, hands in ((cpu, cpu_hands), (metal, metal_hands)):
        session.advance_many(hands, ["call", "call"])
        session.advance_many(hands, ["check", "check"])
        session.deal_many(hands, [["Jh", "Qd", "Kc"], ["Jd", "Qh", "Ks"]])
    private = [["As", "Ad"], ["Ac", "Kh"]]
    expected = cpu.predict_many(cpu_hands, private)
    actual = metal.predict_many(metal_hands, private)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            left.probabilities, right.probabilities, rtol=rtol, atol=atol
        )
        assert torch.isfinite(left.probabilities).all()
        torch.testing.assert_close(
            left.probabilities.sum(), torch.tensor(1.0), rtol=rtol, atol=atol
        )
