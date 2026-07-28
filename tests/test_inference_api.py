from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pbg.config import load_config
from pbg.games.poker.queued import build_model_from_config
from pbg.inference import PokerInferenceSession, parse_card
from pbg.inference_artifact import (
    INFERENCE_ARTIFACT_SCHEMA_VERSION,
    load_inference_source,
)
from pbg.core.persistence import CHECKPOINT_SCHEMA, model_state_sha256

CONFIG = Path("configs/train/nlhe_hu_8bb_normal_smoke.yaml")


def _artifact(path: Path, device: str) -> Path:
    config = load_config(CONFIG)
    model = build_model_from_config(config.scientific_mapping(), device)
    model_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    artifact = {
        "artifact_schema": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "source_checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_checkpoint_sha256": "1" * 64,
        "iteration": 7,
        "scientific_config": config.scientific_mapping(),
        "model": model_state,
        "model_sha256": model_state_sha256(model_state),
    }
    identity = {
        name: artifact[name]
        for name in (
            "artifact_schema",
            "source_checkpoint_schema",
            "source_checkpoint_sha256",
            "iteration",
            "scientific_config",
            "model_sha256",
        )
    }
    artifact["payload_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    torch.save(artifact, path)
    return path


def test_inference_artifact_rejects_legacy_and_invalid_hash(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {"inference_artifact_schema_version": "pbg-inference-v1"}, legacy
    )
    with pytest.raises(RuntimeError, match="strict queued"):
        load_inference_source(legacy)
    invalid = {
        "artifact_schema": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "source_checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_checkpoint_sha256": "z" * 64,
        "iteration": 0,
        "scientific_config": {},
        "model": {},
    }
    path = tmp_path / "invalid.pt"
    torch.save(invalid, path)
    with pytest.raises(RuntimeError):
        load_inference_source(path)


def test_card_parser_uses_global_compact_deck_ids() -> None:
    assert parse_card("2c") == 0
    assert parse_card("As") == 51
    with pytest.raises(ValueError, match="active deck"):
        parse_card("Tc", first_card=44)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stateful_singleton_batch_actions_and_export_identity(tmp_path: Path) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt", "cuda"), device="cuda"
    )
    first, second = session.new_hands((0, 1))
    singleton = session.predict(first, ("As", "Kd"), action_mode="argmax")
    assert singleton.selected_action in singleton.actions
    torch.testing.assert_close(singleton.probabilities.sum(), torch.tensor(1.0))
    batch = session.predict_many(
        (first, second), (("As", "Kd"), ("Qh", "Js"))
    )
    torch.testing.assert_close(batch[0].probabilities, singleton.probabilities)
    full = session.full_strategy((first, second))
    assert full.probabilities.shape[:2] == (2, 120)
    assert full.values_bb.shape == (2, 120)
    session.advance(second, "call")
    assert session.source_checkpoint_sha256 == "1" * 64


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_mps_preflop_stays_on_metal_and_postflop_fails_early(
    tmp_path: Path,
) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt", "mps"), device="mps"
    )
    assert all(value.device.type == "mps" for value in session.model.parameters())
    hand = session.new_hand()
    decision = session.predict(hand, ("As", "Kd"))
    torch.testing.assert_close(decision.probabilities.sum(), torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="unsupported before CPU staging"):
        session.deal(hand, ("Jh", "Qd", "Kc"))
