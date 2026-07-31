from __future__ import annotations

import copy
import json

from pbg.benchmark import compare_benchmark
from pbg.config import load_config


def _result(rate: float) -> dict:
    cfg = load_config("configs/train/nlhe_hu_8bb_normal_smoke.yaml")
    required = [
        "seconds",
        "iteration",
        "player_nodes",
        "player_nodes_per_second",
        "incremental_allocated_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "logical_work_before",
        "logical_work_after",
        "structure",
        "telemetry",
    ]
    return {
        "schema": "pbg-queued-benchmark-v1",
        "mode": dict(cfg.mode),
        "scientific_config": cfg.scientific_mapping(),
        "execution": cfg.execution.mapping(),
        "identity": {"required_sample_metrics": required},
        "environment": {"device": "cuda:0", "device_name": "test-gpu"},
        "median": {"player_nodes_per_second": rate},
        "samples": [
            {
                "seconds": 1.0,
                "iteration": 0,
                "player_nodes": 5,
                "player_nodes_per_second": rate,
                "incremental_allocated_bytes": 0,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
                "logical_work_before": [
                    {
                        "trajectories": 0,
                        "completion_samples": 0,
                        "minibatches": 0,
                        "optimizer_steps": 0,
                    }
                ],
                "logical_work_after": [
                    {
                        "trajectories": 1,
                        "completion_samples": 0,
                        "minibatches": 1,
                        "optimizer_steps": 1,
                    }
                ],
                "structure": [{}],
                "telemetry": [None],
            }
        ],
    }


def test_performance_gate_accepts_tolerance_and_reports_regression() -> None:
    baseline = _result(100.0)
    assert compare_benchmark(_result(90.0), baseline) == []
    assert compare_benchmark(_result(89.0), baseline) == [
        "player-node throughput regressed"
    ]
    assert compare_benchmark(
        _result(100.0),
        json.loads(json.dumps(baseline)),
    ) == []
    mismatched = _result(100.0)
    mismatched["samples"][0]["logical_work_after"][0][
        "trajectories"
    ] = 100
    assert compare_benchmark(mismatched, baseline) == [
        "logical work differs"
    ]


def test_performance_gate_rejects_identity_and_environment_mismatch() -> None:
    baseline = _result(100.0)
    current = copy.deepcopy(baseline)
    current["environment"]["device"] = "cuda:1"
    assert "environment identity differs" in compare_benchmark(current, baseline)
    assert compare_benchmark(
        current, baseline, allow_environment_mismatch=True
    ) == []
    current = copy.deepcopy(baseline)
    current["scientific_config"]["run"]["seed"] += 1
    assert "scientific_config identity differs" in compare_benchmark(
        current, baseline
    )


def test_benchmark_never_converts_modes_or_capacity_manifests() -> None:
    config = load_config("configs/train/nlhe_hu_8bb_normal_smoke.yaml")
    result = _result(100.0)
    assert result["mode"] == dict(config.mode)
    assert result["execution"]["manifest"] == config.execution.mapping()["manifest"]
