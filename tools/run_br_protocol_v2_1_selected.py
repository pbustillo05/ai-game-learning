#!/usr/bin/env python3
"""Run exactly the frozen protocol-v2.1 test selection on one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROTOCOL = "fixed_target_paired_response_gain_v2_1"
CUDA_HOME = Path("/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()

    plan = load(args.plan)
    manifest = load(args.manifest)
    selection = load(args.selection)
    if any(
        document.get("protocol_version") != PROTOCOL
        for document in (plan, manifest, selection)
    ):
        raise RuntimeError("plan, manifest, and selection must all use protocol v2.1")
    if selection.get("test_results_inspected") is not False:
        raise RuntimeError("selection was not frozen before test inspection")
    if len(selection["targets"]) != 1:
        raise RuntimeError("selected-test runner requires one target per manifest")

    selected_target = selection["targets"][0]
    target_id = selected_target["target_id"]
    targets = {target["id"]: target for target in plan["targets"]}
    inventory = {
        target["id"]: target for target in manifest["priority_targets"]
    }
    if target_id not in targets or target_id not in inventory:
        raise RuntimeError("selected target is absent from plan or run manifest")
    target = targets[target_id]
    target_inventory = inventory[target_id]
    if selected_target["target_sha256"] != target["target_sha256"]:
        raise RuntimeError("selection target hash differs from plan")

    split = plan["splits"]["test"]
    selected = selected_target["selected"]
    if selected["selection_kind"] == "zero_gain_control":
        output = args.results / f"test_{target_id}_zero_control_0traj_v2_1.json"
        write_exclusive(
            output,
            {
                "schema": "pbg-fixed-target-zero-control-test-v2.1",
                "protocol_version": PROTOCOL,
                "split_namespace": split["namespace"],
                "evaluation_seed": split["seed"],
                "planned_trajectory_count": split[
                    "trajectories_for_frozen_selection"
                ],
                "actual_trajectory_count": 0,
                "selection_kind": "zero_gain_control",
                "target_id": target_id,
                "target_sha256": target["target_sha256"],
                "mean_gain_bb_per_hand": 0.0,
                "standard_error_bb_per_hand": 0.0,
                "confidence_interval_bb_per_hand": [0.0, 0.0],
                "reason": "unchanged target is deterministically identical in both paired arms",
            },
        )
        print(output)
        return

    iteration = selected["response_iteration"]
    candidates = {
        candidate["response_iteration"]: candidate
        for candidate in target_inventory["candidates"]
    }
    candidate = candidates[iteration]
    if candidate["checkpoint_sha256"] != selected["response_checkpoint_sha256"]:
        raise RuntimeError("frozen response hash differs from run manifest")
    count = split["trajectories_for_frozen_selection"]
    output = args.results / (
        f"test_{target_id}_ckpt{iteration:04d}_{count}traj_v2_1.json"
    )
    environment = os.environ.copy()
    venv_bin = Path("/home/pablob/.venvs/poker/bin")
    environment["CUDA_HOME"] = str(CUDA_HOME)
    environment["PATH"] = (
        f"{venv_bin}:{CUDA_HOME / 'bin'}:{environment.get('PATH', '')}"
    )
    environment["LD_LIBRARY_PATH"] = (
        f"{CUDA_HOME / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    )
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pbg.cli.br_eval",
            "--probe",
            candidate["inference_artifact"],
            "--target",
            target_inventory["target_inference_artifact"],
            "--hands",
            str(count),
            "--split-namespace",
            split["namespace"],
            "--seed",
            str(split["seed"]),
            "--logical-deal-start",
            str(split["logical_deal_start"]),
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
