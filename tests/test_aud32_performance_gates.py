from __future__ import annotations

import copy

from pbg.benchmark import compare_benchmark
from pbg.config import load_config


def _result(rate: float) -> dict:
    cfg = load_config("configs/train/nlhe_hu_8bb_normal_queued_smoke.yaml")
    return {
        "schema": "pbg-queued-benchmark-v1",
        "mode": dict(cfg.mode),
        "scientific_config": cfg.scientific_mapping(),
        "execution": cfg.execution.mapping(),
        "environment": {"device": "cuda:0", "device_name": "test-gpu"},
        "median": {"player_nodes_per_second": rate},
    }


def test_performance_gate_accepts_tolerance_and_reports_regression() -> None:
    baseline = _result(100.0)
    assert compare_benchmark(_result(90.0), baseline) == []
    assert compare_benchmark(_result(89.0), baseline) == [
        "player-node throughput regressed"
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
    current["scientific_config"]["seed"] += 1
    assert "scientific_config identity differs" in compare_benchmark(
        current, baseline
    )


def test_benchmark_never_converts_modes_or_capacity_manifests() -> None:
    config = load_config("configs/train/nlhe_hu_8bb_normal_queued_smoke.yaml")
    result = _result(100.0)
    assert result["mode"] == dict(config.mode)
    assert result["execution"]["manifest"] == config.execution.mapping()["manifest"]
