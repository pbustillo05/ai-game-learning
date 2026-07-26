#!/usr/bin/env python3
"""Reduce the 2026-07-24 rollout diagnosis without discarding raw samples."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("runs/experiments/nlhe_hu_32bb_rollout_diagnosis_20260724")
ORDERS = {
    0: ("sequential_rb", "branched_baseline", "branched_reuse_private_profile"),
    1: ("branched_reuse_private_profile", "branched_baseline", "sequential_rb"),
    2: ("branched_baseline", "branched_reuse_private_profile", "sequential_rb"),
}


def median(records: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def load_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["modes"][0]["samples"][0]
    variant = path.stem
    decisions = float(sample["decisions"])
    pre_rows = float(sample.get("execution_rollout_inference_pre_timed_model_rows", 0.0))
    post_rows = float(
        sample.get("execution_rollout_inference_post_2_timed_model_rows", 0.0)
    )
    pre_seconds = float(
        sample.get("execution_rollout_inference_pre_model_seconds", 0.0)
    )
    post_seconds = float(
        sample.get("execution_rollout_inference_post_2_model_seconds", 0.0)
    )
    sampler_seconds = float(
        sample.get("execution_rollout_rejection_sampler_cuda_work_seconds", 0.0)
    )
    internal_seal = float(sample["phase_internal_seal_cuda_seconds"])
    rollout_seconds = float(sample["phase_rollout_seconds"])
    other_rollout = (
        rollout_seconds
        - pre_seconds
        - post_seconds
        - sampler_seconds
        - internal_seal
    )
    return {
        "gpu": int(path.parent.name.split("_", 1)[0].removeprefix("gpu")),
        "seed": int(path.parent.name.rsplit("seed", 1)[1]),
        "variant": variant,
        "nodes": int(sample["nodes"]),
        "decisions": int(decisions),
        "leaves": int(sample["leaves"]),
        "rollout_seconds": rollout_seconds,
        "rollout_ms_per_decision": 1_000.0 * rollout_seconds / decisions,
        "pre_rows": int(pre_rows),
        "pre_seconds": pre_seconds,
        "pre_ms_per_node": 0.0 if not pre_rows else 1_000.0 * pre_seconds / pre_rows,
        "post_rows": int(post_rows),
        "post_seconds": post_seconds,
        "post_ms_per_node": 0.0 if not post_rows else 1_000.0 * post_seconds / post_rows,
        "sampler_rows": int(
            sample.get("execution_rollout_rejection_sampler_cuda_work_rows", 0.0)
        ),
        "sampler_calls": int(
            sample.get("execution_rollout_rejection_sampler_cuda_work_calls", 0.0)
        ),
        "sampler_seconds": sampler_seconds,
        "sampler_ms_per_decision": 1_000.0 * sampler_seconds / decisions,
        "event_sampling_seconds": float(
            sample.get("execution_rollout_event_sampling_cuda_work_seconds", 0.0)
        ),
        "state_transition_seconds": float(
            sample.get(
                "execution_rollout_state_transition_cuda_work_seconds", 0.0
            )
        ),
        "reach_update_seconds": float(
            sample.get("execution_rollout_reach_update_cuda_work_seconds", 0.0)
        ),
        "child_reservation_seconds": float(
            sample.get(
                "execution_rollout_child_reservation_cuda_work_seconds", 0.0
            )
        ),
        "child_commit_seconds": float(
            sample.get("execution_rollout_child_commit_cuda_work_seconds", 0.0)
        ),
        "chance_closure_seconds": float(
            sample.get("execution_rollout_chance_closure_cuda_work_seconds", 0.0)
        ),
        "internal_leaf_seal_seconds": float(
            sample.get(
                "execution_rollout_internal_leaf_seal_cuda_work_seconds", 0.0
            )
        ),
        "internal_decision_seal_seconds": float(
            sample.get(
                "execution_rollout_internal_decision_seal_cuda_work_seconds", 0.0
            )
        ),
        "internal_seal_seconds": internal_seal,
        "final_seal_seconds": float(sample["phase_seal_seconds"]),
        "all_seal_seconds": float(sample["phase_all_seal_seconds"]),
        "completion_seconds": float(sample["phase_completion_seconds"]),
        "target_propagation_seconds": float(
            sample["phase_target_propagation_cuda_seconds"]
        ),
        "target_total_seconds": float(sample["phase_target_total_cuda_seconds"]),
        "other_rollout_seconds": other_rollout,
        "other_rollout_ms_per_decision": 1_000.0 * other_rollout / decisions,
        "sample_closure_target_seconds": (
            rollout_seconds
            + float(sample["phase_seal_seconds"])
            + float(sample["phase_backward_seconds"])
        ),
        "host_sync_calls": int(sample["execution_host_synchronizations"]),
        "host_sync_seconds_overlap": float(
            sample["execution_host_synchronization_seconds"]
        ),
        "action_reservation_sync_calls": int(
            sample.get(
                "execution_host_sync_poker_action_reservation_total_host_synchronizations",
                0.0,
            )
        ),
        "action_reservation_sync_seconds_overlap": float(
            sample.get(
                "execution_host_sync_poker_action_reservation_total_host_synchronization_seconds",
                0.0,
            )
        ),
        "d2h_transfers": int(sample["execution_d2h_transfers"]),
        "d2h_bytes": int(sample["execution_d2h_bytes"]),
        "h2d_transfers": int(sample["execution_h2d_transfers"]),
        "h2d_bytes": int(sample["execution_h2d_bytes"]),
        "storage_routing_bytes": int(sample["execution_storage_routing_bytes"]),
        "storage_rows_moved": int(sample["execution_storage_rows_moved"]),
        "storage_candidate_rows": int(sample["execution_storage_candidate_rows"]),
        "storage_routing_tensor_operations": int(
            sample["execution_storage_routing_tensor_operations"]
        ),
    }


def thermal_segments(path: Path, gpu: int) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            points.append(
                {
                    "time": datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                    "temperature_c": float(row[2]),
                    "power_w": float(row[3]),
                    "sm_clock_mhz": float(row[4]),
                    "utilization_percent": float(row[6]),
                    "memory_mib": float(row[7]),
                }
            )
    active = [point for point in points if point["utilization_percent"] >= 10.0]
    groups: list[list[dict[str, Any]]] = []
    for point in active:
        if not groups or (point["time"] - groups[-1][-1]["time"]).total_seconds() > 15:
            groups.append([])
        groups[-1].append(point)
    if len(groups) != len(ORDERS[gpu]):
        raise RuntimeError(
            f"GPU {gpu} telemetry produced {len(groups)} active segments, "
            f"expected {len(ORDERS[gpu])}"
        )
    result = []
    for variant, group in zip(ORDERS[gpu], groups, strict=True):
        clocks = [point["sm_clock_mhz"] for point in group]
        result.append(
            {
                "gpu": gpu,
                "variant": variant,
                "active_seconds": (
                    group[-1]["time"] - group[0]["time"]
                ).total_seconds(),
                "max_temperature_c": max(
                    point["temperature_c"] for point in group
                ),
                "median_sm_clock_mhz": float(statistics.median(clocks)),
                "min_sm_clock_mhz": min(clocks),
                "median_power_w": float(
                    statistics.median(point["power_w"] for point in group)
                ),
            }
        )
    return result


def main() -> None:
    records = sorted(
        (
            load_record(path)
            for path in ROOT.glob("gpu*_seed*/*.json")
        ),
        key=lambda record: (record["seed"], record["variant"]),
    )
    by_variant = {
        variant: [record for record in records if record["variant"] == variant]
        for variant in sorted({record["variant"] for record in records})
    }
    median_keys = (
        "nodes",
        "decisions",
        "leaves",
        "rollout_seconds",
        "rollout_ms_per_decision",
        "pre_rows",
        "pre_seconds",
        "pre_ms_per_node",
        "post_rows",
        "post_seconds",
        "post_ms_per_node",
        "sampler_rows",
        "sampler_calls",
        "sampler_seconds",
        "sampler_ms_per_decision",
        "event_sampling_seconds",
        "state_transition_seconds",
        "reach_update_seconds",
        "child_reservation_seconds",
        "child_commit_seconds",
        "chance_closure_seconds",
        "internal_seal_seconds",
        "final_seal_seconds",
        "completion_seconds",
        "target_propagation_seconds",
        "other_rollout_seconds",
        "other_rollout_ms_per_decision",
        "sample_closure_target_seconds",
        "host_sync_calls",
        "host_sync_seconds_overlap",
        "action_reservation_sync_calls",
        "action_reservation_sync_seconds_overlap",
        "d2h_transfers",
        "d2h_bytes",
        "storage_routing_bytes",
        "storage_rows_moved",
        "storage_candidate_rows",
        "storage_routing_tensor_operations",
    )
    medians = {
        variant: {key: median(group, key) for key in median_keys}
        for variant, group in by_variant.items()
    }
    legal_attribution = []
    ablation_pairs = []
    by_seed_variant = {
        (record["seed"], record["variant"]): record for record in records
    }
    for seed in sorted({record["seed"] for record in records}):
        sequential = by_seed_variant[(seed, "sequential_rb")]
        branched = by_seed_variant[(seed, "branched_baseline")]
        ablated = by_seed_variant[(seed, "branched_reuse_private_profile")]
        rollout_gap = (
            branched["rollout_ms_per_decision"]
            - sequential["rollout_ms_per_decision"]
        )
        contributions = {
            "model_mix_ms_per_decision": 1_000.0
            * (
                (branched["pre_seconds"] + branched["post_seconds"])
                / branched["decisions"]
                - (sequential["pre_seconds"] + sequential["post_seconds"])
                / sequential["decisions"]
            ),
            "sampler_ms_per_decision": (
                branched["sampler_ms_per_decision"]
                - sequential["sampler_ms_per_decision"]
            ),
            "internal_seal_ms_per_decision": 1_000.0
            * (
                branched["internal_seal_seconds"] / branched["decisions"]
                - sequential["internal_seal_seconds"] / sequential["decisions"]
            ),
            "other_rollout_ms_per_decision": (
                branched["other_rollout_ms_per_decision"]
                - sequential["other_rollout_ms_per_decision"]
            ),
        }
        legal_attribution.append(
            {
                "seed": seed,
                "rollout_gap_ms_per_decision": rollout_gap,
                **contributions,
                **{
                    key.removesuffix("_ms_per_decision") + "_share": value / rollout_gap
                    for key, value in contributions.items()
                },
            }
        )
        ablation_pairs.append(
            {
                "seed": seed,
                "baseline_nodes": branched["nodes"],
                "ablated_nodes": ablated["nodes"],
                "baseline_rollout_ms_per_decision": branched[
                    "rollout_ms_per_decision"
                ],
                "ablated_rollout_ms_per_decision": ablated[
                    "rollout_ms_per_decision"
                ],
                "rollout_ms_per_decision_reduction_fraction": 1.0
                - ablated["rollout_ms_per_decision"]
                / branched["rollout_ms_per_decision"],
                "sampler_seconds_saved": (
                    branched["sampler_seconds"] - ablated["sampler_seconds"]
                ),
                "sampler_rows_removed": (
                    branched["sampler_rows"] - ablated["sampler_rows"]
                ),
            }
        )
    thermal = []
    for gpu_dir in sorted(ROOT.glob("gpu*_seed*")):
        gpu = int(gpu_dir.name.split("_", 1)[0].removeprefix("gpu"))
        thermal.extend(thermal_segments(gpu_dir / "gpu_telemetry.csv", gpu))
    output = {
        "schema": "nlhe-hu32-rollout-diagnosis-summary-v1",
        "records": records,
        "medians": medians,
        "legal_rollout_attribution": legal_attribution,
        "legal_rollout_attribution_median": {
            key: median(legal_attribution, key)
            for key in legal_attribution[0]
            if key != "seed"
        },
        "legal_rollout_attribution_mean": {
            key: float(
                statistics.mean(float(record[key]) for record in legal_attribution)
            )
            for key in legal_attribution[0]
            if key != "seed"
        },
        "private_profile_ablation": ablation_pairs,
        "private_profile_ablation_median": {
            key: median(ablation_pairs, key)
            for key in ablation_pairs[0]
            if key != "seed"
        },
        "thermal_segments": thermal,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
