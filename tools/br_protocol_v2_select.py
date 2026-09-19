#!/usr/bin/env python3
"""Promote and freeze protocol-v2 response candidates without reading test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROTOCOL = "fixed_target_paired_response_gain_v2"
REPORT_SUFFIX = "v2"
SCHEMA_VERSION = "v2"
OUTPUT_ROOT = Path(
    "runs/experiments/nlhe_hu_32bb_rb_br_comparison_protocol_v2_20260916"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def context(
    plan_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[int, Any]]]:
    plan = load(plan_path)
    manifest = load(manifest_path)
    if plan["protocol_version"] != PROTOCOL:
        raise RuntimeError("plan protocol differs from selector")
    targets = {target["id"]: target for target in plan["targets"]}
    inventory = {
        target["id"]: {
            int(candidate["response_iteration"]): candidate
            for candidate in target["candidates"]
        }
        for target in manifest["priority_targets"]
    }
    if targets.keys() != inventory.keys():
        raise RuntimeError("plan and inventory target sets differ")
    return plan, targets, inventory


def checked_report(
    path: Path,
    *,
    split: dict[str, Any],
    target: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    report = load(path)
    expected = {
        "protocol_version": PROTOCOL,
        "split_namespace": split["namespace"],
        "evaluation_seed": split["seed"],
        "trajectory_count": split[
            "trajectories_per_trained_candidate"
        ],
        "target_sha256": target["target_sha256"],
        "response_sha256": candidate["checkpoint_sha256"],
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise RuntimeError(f"{path}: {name} differs from plan/inventory")
    if len(report.get("observations", ())) != expected["trajectory_count"]:
        raise RuntimeError(f"{path}: raw observation count differs from report")
    return {
        "response_iteration": candidate["response_iteration"],
        "response_checkpoint_sha256": candidate["checkpoint_sha256"],
        "response_artifact_sha256": candidate["inference_artifact_sha256"],
        "report": str(path),
        "report_sha256": sha256_file(path),
        "screening_or_validation_mean_gain_bb_per_hand": report[
            "mean_gain_bb_per_hand"
        ],
        "per_seat_gain_bb_per_hand": report["per_seat_gain_bb_per_hand"],
        "standard_error_bb_per_hand": report["standard_error_bb_per_hand"],
        "confidence_interval_bb_per_hand": report[
            "confidence_interval_bb_per_hand"
        ],
        "elapsed_seconds": report["elapsed_seconds"],
    }


def requested_target_ids(
    requested: list[str] | None, available: dict[str, Any]
) -> list[str]:
    values = list(available) if requested is None else requested
    if not values or len(values) != len(set(values)):
        raise ValueError("target filter must contain unique target ids")
    unknown = sorted(set(values) - set(available))
    if unknown:
        raise ValueError(f"unknown target ids: {unknown}")
    return values


def promote(args: argparse.Namespace) -> None:
    plan, targets, inventory = context(args.plan, args.manifest)
    target_ids = requested_target_ids(args.target_id, targets)
    split = plan["splits"]["screening"]
    result: dict[str, Any] = {
        "schema": f"pbg-fixed-target-br-validation-promotion-{SCHEMA_VERSION}",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_version": PROTOCOL,
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "run_manifest": str(args.manifest),
        "run_manifest_sha256": sha256_file(args.manifest),
        "rule": plan["selection"],
        "targets": [],
    }
    for target_id in target_ids:
        target = targets[target_id]
        rows = []
        for iteration in plan["candidate_grid"]["trained_response_iterations"]:
            candidate = inventory[target_id][iteration]
            path = args.results / (
                f"screening_{target_id}_ckpt{iteration:04d}_"
                f"{split['trajectories_per_trained_candidate']}traj_{REPORT_SUFFIX}.json"
            )
            rows.append(
                checked_report(
                    path,
                    split=split,
                    target=target,
                    candidate=candidate,
                )
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["screening_or_validation_mean_gain_bb_per_hand"],
                row["response_iteration"],
            ),
        )
        result["targets"].append(
            {
                "target_id": target_id,
                "target_sha256": target["target_sha256"],
                "all_screening_candidates": rows,
                "promoted_response_iterations": [
                    row["response_iteration"] for row in ranked[:3]
                ],
            }
        )
    write_exclusive(args.output, result)


def select(args: argparse.Namespace) -> None:
    plan, targets, inventory = context(args.plan, args.manifest)
    promotion = load(args.promotion)
    split = plan["splits"]["validation"]
    promoted = {
        target["target_id"]: target["promoted_response_iterations"]
        for target in promotion["targets"]
    }
    if any(target_id not in targets for target_id in promoted):
        raise RuntimeError("promotion manifest contains an unknown target")
    target_ids = requested_target_ids(
        args.target_id,
        {target_id: targets[target_id] for target_id in promoted},
    )
    result: dict[str, Any] = {
        "schema": f"pbg-fixed-target-br-selection-manifest-{SCHEMA_VERSION}",
        "selection_frozen_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "protocol_version": PROTOCOL,
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "run_manifest": str(args.manifest),
        "run_manifest_sha256": sha256_file(args.manifest),
        "promotion_manifest": str(args.promotion),
        "promotion_manifest_sha256": sha256_file(args.promotion),
        "test_results_inspected": False,
        "rule": plan["selection"],
        "targets": [],
    }
    for target_id in target_ids:
        target = targets[target_id]
        rows = []
        for iteration in promoted[target_id]:
            candidate = inventory[target_id][iteration]
            path = args.results / (
                f"validation_{target_id}_ckpt{iteration:04d}_"
                f"{split['trajectories_per_trained_candidate']}traj_{REPORT_SUFFIX}.json"
            )
            rows.append(
                checked_report(
                    path,
                    split=split,
                    target=target,
                    candidate=candidate,
                )
            )
        candidates = [
            {
                "selection_kind": "trained_response",
                "validation_mean_gain_bb_per_hand": row[
                    "screening_or_validation_mean_gain_bb_per_hand"
                ],
                **row,
            }
            for row in rows
        ]
        candidates.append(
            {
                "selection_kind": "zero_gain_control",
                "response_iteration": None,
                "response_checkpoint_sha256": target["target_sha256"],
                "validation_mean_gain_bb_per_hand": 0.0,
                "per_seat_gain_bb_per_hand": [0.0, 0.0],
                "standard_error_bb_per_hand": 0.0,
                "confidence_interval_bb_per_hand": [0.0, 0.0],
            }
        )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -row["validation_mean_gain_bb_per_hand"],
                0 if row["selection_kind"] == "zero_gain_control" else 1,
                -1 if row["response_iteration"] is None else row["response_iteration"],
            ),
        )
        chosen = ranked[0]
        result["targets"].append(
            {
                "target_id": target_id,
                "target_sha256": target["target_sha256"],
                "validation_candidates": candidates,
                "selected": chosen,
            }
        )
    write_exclusive(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("promote", "select"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--target-id", action="append")
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "promote":
        promote(args)
    else:
        if args.promotion is None:
            parser.error("select requires --promotion")
        select(args)


if __name__ == "__main__":
    main()
