#!/usr/bin/env python3
"""Finish in-flight RB 6,600 validation and frozen v2.1 test exactly once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from br_protocol_v2_1_full_curve_queue import (
    RESULTS,
    ROOT,
    await_gpu,
    checked_existing,
    emit,
    load,
    publish_plot,
    test_file,
)

PRIORITY_PLAN = RESULTS / "evaluation_plan_v2_1.json"
PRIORITY_MANIFEST = RESULTS / "run_manifest_v2_1_range_conditioned.json"
FULL_PLAN = RESULTS / "evaluation_plan_v2_1_full_curve.json"
TARGET = "rb_i6600"
PROMOTED = (50, 250, 150)


def wait_validation() -> None:
    deadline = time.monotonic() + 2 * 3600
    while time.monotonic() < deadline:
        complete = True
        for iteration in PROMOTED:
            path = RESULTS / (
                f"validation_{TARGET}_ckpt{iteration:04d}_5000traj_v2_1.json"
            )
            if not path.is_file():
                complete = False
                break
            try:
                report = load(path)
            except (json.JSONDecodeError, OSError):
                complete = False
                break
            if (
                report.get("protocol_version")
                != "fixed_target_paired_response_gain_v2_1"
                or report.get("split_namespace") != "main-validation-v2_1"
                or report.get("trajectory_count") != 5000
            ):
                raise RuntimeError(f"invalid priority validation report: {path}")
        if complete:
            return
        time.sleep(30)
    raise TimeoutError("RB 6,600 validation did not complete within two hours")


def finish() -> None:
    wait_validation()
    selection_path = RESULTS / f"selection_{TARGET}_v2_1.json"
    if not selection_path.exists():
        command = [
            sys.executable, "tools/br_protocol_v2_1_select.py", "select",
            "--plan", str(PRIORITY_PLAN),
            "--manifest", str(PRIORITY_MANIFEST),
            "--results", str(RESULTS),
            "--target-id", TARGET,
            "--promotion", str(RESULTS / f"promotion_{TARGET}_v2_1.json"),
            "--output", str(selection_path),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
    selection = load(selection_path)
    if selection.get("test_results_inspected") is not False:
        raise RuntimeError("priority RB 6,600 selection was not frozen")
    test = test_file(TARGET, selection, load(PRIORITY_PLAN))
    if not checked_existing(test):
        await_gpu(0)
        emit(f"gpu=0 start frozen test {TARGET}")
        subprocess.run(
            [
                sys.executable, "tools/run_br_protocol_v2_1_selected.py",
                "--physical-gpu", "0",
                "--plan", str(PRIORITY_PLAN),
                "--manifest", str(PRIORITY_MANIFEST),
                "--selection", str(selection_path),
                "--results", str(RESULTS),
            ],
            check=True,
            cwd=ROOT,
        )
        checked_existing(test)
        await_gpu(0)
    emit(f"completed priority target {TARGET}: {test}")
    publish_plot(FULL_PLAN, load(FULL_PLAN), TARGET)


def detach(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as output:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "run"],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    print(json.dumps({"pid": child.pid, "log": str(log_path)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("run", "detach"))
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.operation == "run":
        finish()
    elif args.log is None:
        parser.error("detach requires --log")
    else:
        detach(args.log.resolve())


if __name__ == "__main__":
    main()
