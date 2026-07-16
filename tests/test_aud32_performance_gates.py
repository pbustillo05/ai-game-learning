from __future__ import annotations

import copy

import pytest

from pbg.benchmark import compare_benchmark


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
            "n_trees": 64,
            "node_budget": 128,
            "completion_samples": 32,
            "minibatch_size": 32,
            "gd_steps_per_iter": 1,
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
