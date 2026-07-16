from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from pbg.benchmark import benchmark_modes, compare_benchmark
from pbg.config import load_config


def _result(speed: float, seconds: float, target_bytes: float) -> dict:
    return {
        "schema": "pbg-performance-v1",
        "environment": {
            "device_type": "cpu",
            "device_name": "test-cpu",
            "compute_capability": None,
            "torch_threads": 1,
        },
        "workload": {
            "game": "poker",
            "sequential_starts": 64,
            "node_budget": 128,
            "completion_samples": 32,
            "completion_backend": "cpu",
            "minibatch_size": 32,
            "gd_steps_per_iter": 1,
            "logical_rng": "counter_hash_v2",
            "training_seed": 7,
            "model_initialization_protocol": "torch_seed_before_bundle_v1",
            "model_initialization_seed": 7,
            "repeat_seeds": [104_736, 209_465],
            "profile_iteration": 1,
            "family_batch_rows": None,
            "max_active_trajectories": None,
            "arena_chunk_nodes": None,
        },
        "modes": [
            {
                "mode": "normal",
                "metrics": {
                    "decisions_per_second": speed,
                    "profiled_iteration_seconds": seconds,
                    "target_bytes_per_decision": target_bytes,
                },
            }
        ],
    }


def test_performance_gate_accepts_tolerance_and_reports_regressions() -> None:
    baseline = _result(100.0, 1.0, 50.0)
    assert compare_benchmark(_result(81.0, 1.19, 59.0), baseline) == []
    failures = compare_benchmark(_result(79.0, 1.21, 61.0), baseline)
    assert {failure.split(":", 1)[0] for failure in failures} == {
        "normal.decisions_per_second",
        "normal.profiled_iteration_seconds",
        "normal.target_bytes_per_decision",
    }


def test_performance_gate_rejects_cross_environment_comparisons() -> None:
    baseline = _result(100.0, 1.0, 50.0)
    current = copy.deepcopy(baseline)
    current["environment"]["torch_threads"] = 8
    with pytest.raises(ValueError, match="environment"):
        compare_benchmark(current, baseline)


def test_performance_gate_rejects_completion_backend_mismatch() -> None:
    baseline = _result(100.0, 1.0, 50.0)
    current = copy.deepcopy(baseline)
    current["workload"]["completion_backend"] = "cuda"
    with pytest.raises(ValueError, match="workload"):
        compare_benchmark(current, baseline)


def test_performance_gate_rejects_logical_rng_version_mismatch() -> None:
    baseline = _result(100.0, 1.0, 50.0)
    current = copy.deepcopy(baseline)
    baseline["workload"]["logical_rng"] = "counter_hash_v1"
    current["workload"]["logical_rng"] = "counter_hash_v2"
    with pytest.raises(ValueError, match="workload"):
        compare_benchmark(current, baseline)



def test_benchmark_seeds_model_construction_before_build(monkeypatch) -> None:
    cfg = load_config("configs/train/nlhe6.yaml")
    cfg.loss.c_kl = 0.0
    observed_initializations: list[torch.Tensor] = []

    def build(_cfg, device):
        model = torch.nn.Linear(2, 2)
        observed_initializations.append(model.weight.detach().clone())
        return SimpleNamespace(
            model=model,
            rollout_backend=SimpleNamespace(device=torch.device(device)),
        )

    def iteration(*_args, profile=False, **_kwargs):
        return (
            {
                "profiled_iteration_seconds": 1.0,
                "phase_rollout_seconds": 0.5,
                "decisions": 1.0,
            }
            if profile
            else {}
        )

    monkeypatch.setattr("pbg.benchmark.build_game_bundle", build)
    monkeypatch.setattr("pbg.benchmark._one_iteration", iteration)
    torch.manual_seed(1)
    benchmark_modes(cfg, ("normal",), warmup=0, repeats=1, device="cpu")
    torch.manual_seed(999_999)
    benchmark_modes(cfg, ("normal",), warmup=0, repeats=1, device="cpu")
    assert len(observed_initializations) == 2
    assert torch.equal(observed_initializations[0], observed_initializations[1])


def test_benchmark_loads_and_records_read_only_model_checkpoint(
    monkeypatch, tmp_path
) -> None:
    cfg = load_config("configs/train/nlhe6.yaml")
    cfg.loss.c_kl = 0.0
    expected = torch.nn.Linear(2, 2).state_dict()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": expected, "iteration": 17, "run_id": "trained-policy"},
        checkpoint,
    )
    observed: list[dict[str, torch.Tensor]] = []

    def build(_cfg, device):
        return SimpleNamespace(
            model=torch.nn.Linear(2, 2),
            rollout_backend=SimpleNamespace(device=torch.device(device)),
        )

    def iteration(_cfg, bundle, *_args, profile=False, **_kwargs):
        if profile:
            observed.append(copy.deepcopy(bundle.model.state_dict()))
            return {
                "profiled_iteration_seconds": 1.0,
                "phase_rollout_seconds": 0.5,
                "decisions": 1.0,
            }
        return {}

    monkeypatch.setattr("pbg.benchmark.build_game_bundle", build)
    monkeypatch.setattr("pbg.benchmark._one_iteration", iteration)
    result = benchmark_modes(
        cfg,
        ("normal",),
        warmup=0,
        repeats=1,
        device="cpu",
        checkpoint=checkpoint,
    )

    assert len(observed) == 1
    assert all(torch.equal(observed[0][key], value) for key, value in expected.items())
    assert result["workload"]["model_initialization_protocol"] == "checkpoint_model_state_v1"
    assert result["workload"]["model_checkpoint"]["iteration"] == 17
    assert result["workload"]["model_checkpoint"]["run_id"] == "trained-policy"
