from __future__ import annotations

import copy
import pytest

from pbg.benchmark import compare_benchmark


def _result(speed: float, seconds: float, target_bytes: float) -> dict:
    return {
        "schema": "pbg-performance-v1",
        "environment": {
            "device_type": "cuda",
            "device_name": "test-gpu",
            "compute_capability": None,
            "torch_threads": 1,
        },
        "workload": {
            "game": "poker",
            "roots_per_gpu": 64,
            "storage_page_rows": 128,
            "completion_samples": 32,
            "execution_device": "cuda:0",
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
            "storage_page_rows": None,
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


def test_performance_gate_rejects_execution_device_mismatch() -> None:
    baseline = _result(100.0, 1.0, 50.0)
    current = copy.deepcopy(baseline)
    current["workload"]["execution_device"] = "cuda:1"
    with pytest.raises(ValueError, match="workload"):
        compare_benchmark(current, baseline)


def test_performance_gate_rejects_logical_rng_version_mismatch() -> None:
    baseline = _result(100.0, 1.0, 50.0)
    current = copy.deepcopy(baseline)
    baseline["workload"]["logical_rng"] = "counter_hash_v1"
    current["workload"]["logical_rng"] = "counter_hash_v2"
    with pytest.raises(ValueError, match="workload"):
        compare_benchmark(current, baseline)
