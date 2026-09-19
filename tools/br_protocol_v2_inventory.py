#!/usr/bin/env python3
"""Generate a traceable inventory for a versioned fixed-target BR plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REQUIRED = (50, 100, 150, 200, 250, 300)
TASK_CODE = (
    "pbg/inference.py",
    "pbg/games/poker/queued/hooks.py",
    "pbg/games/poker/queued/response.py",
    "pbg/eval/br_probe.py",
    "pbg/cli/br_eval.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative(root: Path, value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.relative_to(root))
    return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_iterations(run: Path) -> list[int]:
    values = []
    for path in run.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            values.append(int(match.group(1)))
    return sorted(values)


def priority_inventory(root: Path, target: dict[str, Any]) -> dict[str, Any]:
    run = root / target["response_run"]
    artifact_dir = root / target.get("response_artifact_dir", target["response_run"])
    config = root / "configs/experiments" / f"{run.name}.yaml"
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    status_path = run / "status.json"
    target_checkpoint = root / target["target_checkpoint"]
    actual_target_hash = sha256_file(target_checkpoint)
    if actual_target_hash != target["target_sha256"]:
        raise RuntimeError(f"target hash mismatch: {target['id']}")
    candidates = []
    for iteration in REQUIRED:
        checkpoint = run / f"checkpoint_{iteration:08d}.pt"
        artifact = artifact_dir / f"probe_inference_ckpt{iteration:04d}.pt"
        export_report_path = artifact_dir / f"probe_inference_ckpt{iteration:04d}_report.json"
        if not checkpoint.is_file() or not artifact.is_file() or not export_report_path.is_file():
            raise RuntimeError(
                f"{target['id']} lacks checkpoint/artifact/report at {iteration}"
            )
        export_report = load_json(export_report_path)
        checkpoint_hash = sha256_file(checkpoint)
        artifact_hash = sha256_file(artifact)
        if checkpoint_hash != export_report["source_checkpoint_sha256"]:
            raise RuntimeError(
                f"{target['id']} checkpoint/export mismatch at {iteration}"
            )
        if artifact_hash != export_report["artifact_sha256"]:
            raise RuntimeError(
                f"{target['id']} artifact/export mismatch at {iteration}"
            )
        candidates.append(
            {
                "response_iteration": iteration,
                "checkpoint": relative(root, checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "inference_artifact": relative(root, artifact),
                "inference_artifact_sha256": artifact_hash,
                "model_sha256": export_report["model_sha256"],
                "export_report": relative(root, export_report_path),
                "export_report_sha256": sha256_file(export_report_path),
            }
        )
    target_artifact = artifact_dir / "target_inference.pt"
    target_export_path = artifact_dir / "target_inference_report.json"
    target_export = load_json(target_export_path)
    target_artifact_hash = sha256_file(target_artifact)
    if target_artifact_hash != target_export["artifact_sha256"]:
        raise RuntimeError(f"{target['id']} target artifact/export mismatch")
    if actual_target_hash != target_export["source_checkpoint_sha256"]:
        raise RuntimeError(f"{target['id']} target checkpoint/export mismatch")
    legacy_reports = []
    for report_path in sorted(run.glob("br_*.json")):
        report = load_json(report_path)
        legacy_reports.append(
            {
                "path": relative(root, report_path),
                "sha256": sha256_file(report_path),
                "legacy_semantics": report.get("semantics"),
                "mean_gain_bb_per_hand": report.get("mean_gain_bb_per_hand"),
                "trajectory_count": report.get("trajectory_count"),
            }
        )
    scientific = raw["scientific"]
    run_config = raw["run"]
    return {
        **target,
        "response_config": relative(root, config),
        "response_config_sha256": sha256_file(config),
        "response_run_status": load_json(status_path) if status_path.is_file() else None,
        "response_seed": run_config["seed"],
        "response_iterations": run_config["iterations"],
        "checkpoint_every": run_config["checkpoint_every"],
        "optimizer_steps_per_iteration": scientific["schedule"]["steps"],
        "all_checkpoint_iterations": checkpoint_iterations(run),
        "common_grid_complete": all(
            (run / f"checkpoint_{iteration:08d}.pt").is_file()
            for iteration in REQUIRED
        ),
        "target_checkpoint_actual_sha256": actual_target_hash,
        "target_inference_artifact": relative(root, target_artifact),
        "target_inference_artifact_sha256": target_artifact_hash,
        "target_model_sha256": target_export["model_sha256"],
        "target_export_report": relative(root, target_export_path),
        "target_export_report_sha256": sha256_file(target_export_path),
        "candidates": candidates,
        "legacy_reports": legacy_reports,
    }


def all_run_summary(root: Path) -> list[dict[str, Any]]:
    result = []
    configs = sorted(
        (root / "configs/experiments").glob(
            "nlhe_hu_32bb_rb_brprobe_*_300iter_*.yaml"
        )
    )
    for config in configs:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        output = Path(raw["run"]["output_dir"])
        if not output.is_absolute():
            output = root / output
        present = checkpoint_iterations(output) if output.is_dir() else []
        result.append(
            {
                "config": relative(root, config),
                "config_sha256": sha256_file(config),
                "run": relative(root, output),
                "run_exists": output.is_dir(),
                "state": (
                    load_json(output / "status.json").get("state")
                    if (output / "status.json").is_file()
                    else None
                ),
                "checkpoint_iterations": present,
                "required_present": [value for value in REQUIRED if value in present],
                "required_missing": [value for value in REQUIRED if value not in present],
            }
        )
    return result


def git_output(root: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output = args.output if args.output.is_absolute() else root / args.output
    plan = load_json(plan_path)
    tracked_diff = git_output(root, "diff", "--binary", "HEAD", "--")
    status = git_output(root, "status", "--porcelain=v1", "-z")
    result = {
        "schema": "pbg-fixed-target-br-run-manifest-v2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_version": plan["protocol_version"],
        "plan": relative(root, plan_path),
        "plan_sha256": sha256_file(plan_path),
        "git_revision": git_output(root, "rev-parse", "HEAD").decode().strip(),
        "tracked_worktree_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "worktree_status_sha256": hashlib.sha256(status).hexdigest(),
        "evaluation_code_sha256": {
            path: sha256_file(root / path) for path in TASK_CODE
        },
        "required_response_iterations": list(REQUIRED),
        "priority_targets": [
            priority_inventory(root, target) for target in plan["targets"]
        ],
        "all_response_run_summary": all_run_summary(root),
        "notes": [
            "Legacy reports are inventoried but are not observations under this protocol version.",
            "Priority checkpoint and artifact hashes were recomputed and checked against export reports.",
            "Response training used adaptive_mixed_profile_range_v1 semantics.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
