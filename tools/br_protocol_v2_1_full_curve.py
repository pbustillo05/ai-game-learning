#!/usr/bin/env python3
"""Precommit the original full-curve v2.1 target set and export saved checkpoints.

All exports are model-only checkpoint I/O; evaluation remains accelerator-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pbg.inference_artifact import export_inference_artifact

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs/experiments/nlhe_hu_32bb_rb_br_comparison_protocol_v2_20260916"
EXPORT_ROOT = RESULTS / "full_curve_exports_v2_1"
REQUIRED = (50, 100, 150, 200, 250, 300)
PATTERN = re.compile(
    r"nlhe_hu_32bb_rb_brprobe_(normal|rb)_i(\d+)_300iter_\d+\.yaml"
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def sources() -> list[tuple[str, int, Path, dict[str, Any]]]:
    result = []
    for path in sorted((ROOT / "configs/experiments").glob(
        "nlhe_hu_32bb_rb_brprobe_*_300iter_*.yaml"
    )):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            continue  # Excludes the distinct 12-step RB 4,400 diagnostic.
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        result.append((match[1], int(match[2]), path, raw))
    result.sort(key=lambda row: (row[0] != "normal", row[1]))
    if len(result) != 18:
        raise RuntimeError(f"expected 18 original targets, found {len(result)}")
    return result


def make_plan(priority_plan: Path, output: Path) -> None:
    base = load(priority_plan)
    if base["protocol_version"] != "fixed_target_paired_response_gain_v2_1":
        raise RuntimeError("source plan is not v2.1")
    targets = []
    gaps = []
    for method, iteration, config_path, raw in sources():
        run_config = raw["run"]
        scientific = raw["scientific"]
        if (
            run_config["seed"] != base["candidate_grid"]["response_training_seed"]
            or run_config["iterations"] != base["candidate_grid"]["response_training_iterations"]
            or scientific["schedule"]["steps"] != base["candidate_grid"]["optimizer_steps_per_iteration"]
            or scientific["response_initialization"] != "target"
        ):
            raise RuntimeError(f"response-training budget mismatch: {config_path}")
        run = ROOT / run_config["output_dir"]
        target_checkpoint = Path(run_config["frozen"]["path"])
        if not target_checkpoint.is_file():
            raise RuntimeError(f"missing target checkpoint: {target_checkpoint}")
        target_sha = digest(target_checkpoint)
        if target_sha != run_config["frozen"]["sha256"]:
            raise RuntimeError(f"target checkpoint hash mismatch: {config_path}")
        target_id = f"{method}_i{config_path.stem.split('_i', 1)[1].split('_', 1)[0]}"
        present = [
            value for value in REQUIRED
            if (run / f"checkpoint_{value:08d}.pt").is_file()
        ]
        missing = [value for value in REQUIRED if value not in present]
        exported = [
            value for value in REQUIRED
            if (run / f"probe_inference_ckpt{value:04d}.pt").is_file()
            and (run / f"probe_inference_ckpt{value:04d}_report.json").is_file()
        ]
        legacy = [
            relative(path)
            for pattern in ("br_eval_20000_seed1729.json",
                            "br_test_selected_ckpt*_20000_seed1729.json")
            for path in sorted(run.glob(pattern))
        ]
        row = {
            "id": target_id,
            "method": "Normal" if method == "normal" else "RB",
            "target_iteration": iteration,
            "target_checkpoint": relative(target_checkpoint),
            "target_sha256": target_sha,
            "response_run": relative(run),
            "response_config": relative(config_path),
            "legacy_test_reports": legacy,
        }
        if missing:
            gaps.append({
                **row,
                "required_response_iterations_present": present,
                "required_response_iterations_missing": missing,
                "reason": "six-checkpoint saved-response grid is unavailable",
            })
            continue
        if len(exported) != len(REQUIRED):
            row["response_artifact_dir"] = relative(EXPORT_ROOT / target_id)
            row["artifact_export_required"] = True
        targets.append(row)
    result = {
        **base,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "precommitted_full_original_curve_extension_before_new_target_tests",
        "source_priority_plan": relative(priority_plan),
        "source_priority_plan_sha256": digest(priority_plan),
        "original_target_count": len(targets) + len(gaps),
        "unavailable_original_targets": gaps,
        "targets": targets,
        "planned_comparisons": [
            "All original Normal/RB target iterations with a complete six-saved-response grid",
            "Preserve unavailable original targets as visible missing observations",
            "Reuse finished v2.1 endpoint and spike tests without changing their frozen selection",
        ],
        "full_curve_notes": [
            "Every included target uses six saved responses, 1000 screening, top-three 5000 validation, and one frozen 20000 fresh test.",
            "Early hollow-dot final-response reports are legacy measurements, not v2.1 observations.",
            "The distinct RB 4400 12-step response run is excluded from the common 60-step search budget.",
            "Saved response weights were trained under adaptive_mixed_profile_range_v1; evaluation uses corrected fixed-target semantics.",
        ],
    }
    exclusive_json(output, result)
    print(output)


def export_missing(plan_path: Path, target_ids: set[str] | None) -> None:
    plan = load(plan_path)
    if plan["protocol_version"] != "fixed_target_paired_response_gain_v2_1":
        raise RuntimeError("export plan is not v2.1")
    for target in plan["targets"]:
        if "response_artifact_dir" not in target:
            continue
        if target_ids is not None and target["id"] not in target_ids:
            continue
        run = ROOT / target["response_run"]
        output = ROOT / target["response_artifact_dir"]
        output.mkdir(parents=True, exist_ok=True)
        pairs = [(ROOT / target["target_checkpoint"], output / "target_inference.pt")]
        pairs.extend(
            (run / f"checkpoint_{value:08d}.pt",
             output / f"probe_inference_ckpt{value:04d}.pt")
            for value in REQUIRED
        )
        for source, artifact in pairs:
            report_path = artifact.with_name(artifact.stem + "_report.json")
            if artifact.exists() or report_path.exists():
                if not artifact.is_file() or not report_path.is_file():
                    raise RuntimeError(f"incomplete existing export: {artifact}")
                report = load(report_path)
                if (
                    report["source_checkpoint_sha256"] != digest(source)
                    or report["artifact_sha256"] != digest(artifact)
                ):
                    raise RuntimeError(f"existing export hash mismatch: {artifact}")
                continue
            report = export_inference_artifact(source, artifact)
            if report["source_checkpoint_sha256"] != digest(source):
                raise RuntimeError(f"new export source hash mismatch: {artifact}")
            exclusive_json(report_path, report)
            print(report_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--priority-plan", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    export = subparsers.add_parser("export")
    export.add_argument("--plan", required=True, type=Path)
    export.add_argument("--target-id", action="append")
    args = parser.parse_args()
    if args.operation == "plan":
        make_plan(args.priority_plan.resolve(), args.output.resolve())
    else:
        export_missing(args.plan.resolve(), None if args.target_id is None else set(args.target_id))


if __name__ == "__main__":
    main()
