from __future__ import annotations

import json

from pbg.cli.runpod_replica_selection import (
    _select,
    _summarize_end_to_end,
)


def test_end_to_end_summary_excludes_warmup_and_kernel_probe(tmp_path) -> None:
    records = []
    for iteration in range(5):
        records.append(
            {
                "iter": iteration,
                "physical_gpu_seconds": float(iteration + 1),
                "fresh_decisions": 10 * (iteration + 1),
                "cuda_kernels": 120 if iteration == 4 else None,
                "cuda_kernels_per_decision": 2.4 if iteration == 4 else None,
                "peak_allocated_bytes": 100 + iteration,
                "peak_reserved_bytes": 200 + iteration,
                "phase_seconds": {
                    "snapshot": 0.1,
                    "rollout": 0.2,
                    "backward": 0.3,
                    "seal": 0.4,
                },
                "optimization_seconds": 0.5,
                "objective_mass": [1.0, 1.0],
                "optimizer_steps_applied": [1.0, 1.0],
            }
        )
    (tmp_path / "aggregate_metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    for lane in range(2):
        lane_dir = tmp_path / f"replica_{lane:03d}"
        lane_dir.mkdir()
        (lane_dir / "metrics.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "iter": iteration,
                        "statistical_rows": 8.0,
                        "exact_objective_mass": 1.0,
                    }
                )
                + "\n"
                for iteration in range(5)
            ),
            encoding="utf-8",
        )

    summary = _summarize_end_to_end(
        tmp_path,
        measured_iterations=(1, 2, 3),
        kernel_iteration=4,
    )

    assert summary["physical_gpu_seconds"] == 9.0
    assert summary["fresh_decisions"] == 90
    assert summary["cuda_kernels_per_decision"] == 2.4
    assert summary["peak_allocated_bytes"] == 104
    assert summary["finite_and_consistent"]


def test_selection_requires_incremental_throughput_and_kernel_reduction() -> None:
    def candidate(k: int, throughput: float, kernels: float):
        return {
            "replicas": k,
            "status": "ok",
            "learner": {
                "finite": [True] * k,
                "measured_steps": 50,
                "objective_mass": [50.0] * k,
            },
            "summary": {
                "fresh_decisions_per_second": throughput,
                "cuda_kernels_per_decision": kernels,
                "peak_allocated_bytes": 10,
                "finite_and_consistent": True,
            },
        }

    selection = _select(
        [
            candidate(1, 100.0, 10.0),
            candidate(2, 115.0, 7.0),
            candidate(4, 125.0, 5.0),
            candidate(8, 150.0, 5.5),
        ],
        minimum_gain=0.10,
        peak_limit_bytes=1_000,
    )

    assert selection["replicas"] == 2
    assert selection["gates"][2]["reasons"] == ["throughput_gain"]
    assert selection["gates"][3]["reasons"] == ["kernel_reduction"]
