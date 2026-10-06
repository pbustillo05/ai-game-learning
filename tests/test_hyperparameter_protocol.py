from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from experiments.hyperparameter_search.protocol import (
    ExposureClock,
    apply_overrides,
    atomic_json,
    canonical_sha256,
    entropy_coefficient,
    resolve_spec,
    sha256_file,
    validate_control,
)


ROOT = Path(__file__).parents[1]
PROFILE_PATH = ROOT / "configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml"


def profile() -> dict[str, object]:
    value = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_apply_overrides_deep_copies_and_allows_mapping_replacement() -> None:
    source = profile()
    original = copy.deepcopy(source)
    result = apply_overrides(
        source,
        {
            "scientific.optimizer.learning_rate": 2.5e-5,
            "scientific.objective.value": {
                "kind": "mse",
                "coefficient": 0.016,
            },
            "control.reference.interval_decisions": None,
        },
    )
    assert source == original
    assert result["training"]["scientific"]["optimizer"]["learning_rate"] == 2.5e-5
    assert result["training"]["scientific"]["objective"]["value"]["kind"] == "mse"
    assert result["control"]["reference"]["interval_decisions"] is None


def test_apply_overrides_null_reference_removes_regularizer() -> None:
    result = apply_overrides(
        profile(), {"scientific.objective.reference_kl": None}
    )
    assert "reference_kl" not in result["training"]["scientific"]["objective"]
    with pytest.raises(ValueError, match="unknown override path"):
        apply_overrides(profile(), {"scientific.objective.nope": 1})


def test_resolve_self_play_arm_sets_runtime_identity_and_worker_clock() -> None:
    config, control = resolve_spec(
        {
            "kind": "self_play",
            "profile": profile(),
            "arm": "kl_weak",
            "private_estimator": "sampled",
            "seed": 123,
            "output_dir": "runs/experiments/protocol_test",
        },
        rank=2,
        world=3,
    )
    assert config["kind"] == "nlhe_run_v1"
    assert config["scientific"]["mode"]["training_regime"] == "self_play"
    assert config["scientific"]["mode"]["private_estimator"] == "sampled"
    assert "completion" not in config["scientific"]
    assert config["run"]["seed"] == 123
    assert config["run"]["iterations"] == 100_000_000
    assert config["run"]["checkpoint_every"] == 20
    assert config["run"]["execution"]["device"] == "cuda"
    assert config["run"]["execution"]["distributed"] is True
    assert config["scientific"]["objective"]["entropy"] == {
        "kind": "constant",
        "coefficient": control["entropy"]["initial_coefficient"],
    }
    assert config["scientific"]["objective"]["reference_kl"]["refresh"] == {
        "iterations": 1_000_000_000
    }


def test_smoke_resolves_small_worker_budgets() -> None:
    config, control = resolve_spec(
        {
            "kind": "self_play",
            "profile": profile(),
            "arm": "control",
            "private_estimator": "rao_blackwell",
            "smoke": True,
        },
        rank=0,
        world=1,
    )
    assert config["run"]["workload"]["roots"] == 64
    assert config["run"]["recovery"]["minimum_roots"] == 64
    assert config["scientific"]["schedule"] == {"batch_size": 16, "steps": 2}
    assert config["scientific"]["completion"]["samples"] == 32
    assert config["run"]["execution"]["microbatch"] == {
        "rollout_rows": 16,
        "completion_rows": 32,
        "learner_rows": 8,
    }
    assert control["entropy"]["zero_after_decisions"] == 128
    assert control["reference"]["interval_decisions"] == 128
    assert control["checkpoint_seconds"] == 1
    assert control["snapshot_seconds"] == 1


def test_smoke_best_response_keeps_reference_disabled_and_uses_base_training(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.pt"
    target.write_bytes(b"target checkpoint")
    source = profile()
    config, control = resolve_spec(
        {
            "kind": "best_response",
            "overrides": {},
            "profile": source,
            "arm": "learning_rate_fast",
            "initialization": "target",
            "target_checkpoint": str(target),
            "smoke": True,
        },
        rank=0,
        world=1,
    )
    assert config["scientific"]["mode"] == {
        "training_regime": "best_response",
        "rollout_mode": "sequential",
        "private_estimator": "rao_blackwell",
    }
    assert config["scientific"]["optimizer"]["learning_rate"] == 1e-4
    assert "reference_kl" not in config["scientific"]["objective"]
    assert config["run"]["execution"]["reference_cache"] == "off"
    assert control["reference"]["interval_decisions"] is None
    assert config["run"]["frozen"] == {
        "path": str(target),
        "sha256": sha256_file(target),
    }
    assert config["scientific"]["response_initialization"] == "target"
    assert "probe" not in config["scientific"]


def test_best_response_rejects_candidate_overrides() -> None:
    with pytest.raises(ValueError, match="candidate overrides"):
        resolve_spec(
            {
                "kind": "best_response",
                "profile": profile(),
                "overrides": {"scientific.optimizer.learning_rate": 1e-6},
                "initialization": "random",
                "target_checkpoint": __file__,
            },
            rank=0,
            world=1,
        )


def test_exposure_clock_retains_overshoot_and_is_checkpointable() -> None:
    clock = ExposureClock()
    assert clock.observe(250, 30, 2, 100) is True
    assert clock.to_dict() == {
        "global_decisions": 250,
        "global_presentations": 30,
        "optimizer_steps": 2,
        "global_effective_hands": 250.0,
        "next_reference_decisions": 300.0,
        "refresh_count": 2,
    }
    assert clock.observe(0, 0, 0, 100) is False
    assert clock.observe(51, 1, 1, 100) is True
    assert clock.refresh_count == 3
    restored = ExposureClock.from_dict(clock.to_dict())
    assert restored.to_dict() == clock.to_dict()
    with pytest.raises(ValueError, match="unknown or missing"):
        ExposureClock.from_dict({"global_decisions": 0})


def test_entropy_clock_reaches_exact_zero() -> None:
    control = profile()["control"]
    assert entropy_coefficient(control, 0) == pytest.approx(0.0015625)
    assert entropy_coefficient(control, 125_000) == pytest.approx(0.00078125)
    assert entropy_coefficient(control, 250_000) == 0.0
    assert entropy_coefficient(control, 400_000) == 0.0
    clock = ExposureClock(global_decisions=250_000)
    assert clock.entropy_coefficient(control) == 0.0


def test_posterior_clock_uses_exact_exposure_and_discount_exponent() -> None:
    control = copy.deepcopy(profile()["control"])
    control["clock"] = "posterior_effective_hands"
    control["posterior_clock"] = {"exponent": 0.5}
    control["entropy"]["zero_after_decisions"] = 100.0
    clock = ExposureClock()
    clock.observe(10, 20, 2, 50.0, effective_hands=36.0, schedule_clock="posterior_effective_hands")
    assert clock.global_decisions == 10
    assert clock.global_effective_hands == 36.0
    # The clock stores the measured effective exposure; exponent application is
    # performed by the worker before observation and is never hidden here.
    assert clock.schedule_exposure(control) == 36.0


def test_standalone_training_profile_accepts_control_arm() -> None:
    standalone = yaml.safe_load((ROOT / "configs/profiles/nlhe_hu_32bb_rb_normalized_control_v1.yaml").read_text())
    config, control = resolve_spec({"kind": "self_play", "profile": standalone, "arm": "control"}, 0, 1)
    assert config["kind"] == "nlhe_run_v1"
    assert control["clock"] == "global_player_decisions"


def test_control_validation_rejects_wrong_protocol_semantics() -> None:
    control = profile()["control"]
    validate_control(control)
    bad = copy.deepcopy(control)
    bad["gradient_reduction"] = "sum"
    with pytest.raises(ValueError, match="global_mean"):
        validate_control(bad)
    bad = copy.deepcopy(control)
    bad["reference"]["interval_decisions"] = 0
    with pytest.raises(ValueError, match="positive or null"):
        validate_control(bad)


def test_hash_and_atomic_json(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"abc")
    assert sha256_file(source) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    payload = {"b": 2, "a": [1, 3]}
    destination = tmp_path / "nested" / "manifest.json"
    atomic_json(destination, payload)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert canonical_sha256(payload) == canonical_sha256({"a": [1, 3], "b": 2})
