#!/usr/bin/env python3
"""Bounded, resumable v2.1 full-curve evaluation queue on idle physical GPUs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs/experiments/nlhe_hu_32bb_rb_br_comparison_protocol_v2_20260916"
CUDA_HOME = Path("/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13")
ORDER = (
    "normal_i0700", "rb_i1100", "normal_i1400", "rb_i2250",
    "normal_i2100", "rb_i3350", "normal_i2800", "rb_i4400",
    "normal_i4200", "rb_i5500", "normal_i5600", "normal_i8400",
)
PROTOCOL = "fixed_target_paired_response_gain_v2_1"
PRINT_LOCK = threading.Lock()
PLOT_LOCK = threading.Lock()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def emit(message: str) -> None:
    with PRINT_LOCK:
        print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def file_for(target_id: str, stage: str, iteration: int, count: int) -> Path:
    return RESULTS / f"{stage}_{target_id}_ckpt{iteration:04d}_{count}traj_v2_1.json"


def test_file(target_id: str, selection: dict[str, Any], plan: dict[str, Any]) -> Path:
    chosen = selection["targets"][0]["selected"]
    if chosen["selection_kind"] == "zero_gain_control":
        return RESULTS / f"test_{target_id}_zero_control_0traj_v2_1.json"
    count = plan["splits"]["test"]["trajectories_for_frozen_selection"]
    return file_for(target_id, "test", chosen["response_iteration"], count)


def gpu_idle(gpu: int) -> bool:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    ).splitlines()
    indexed = {}
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        indexed[int(parts[0])] = (parts[1], int(parts[2]), int(parts[3]))
    if gpu not in indexed:
        raise RuntimeError(f"physical GPU {gpu} is unavailable")
    uuid, memory, utilization = indexed[gpu]
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader"], text=True,
    )
    occupied = any(
        row.split(",", 1)[0].strip() == uuid
        for row in processes.splitlines() if row.strip()
    )
    return not occupied and memory <= 64 and utilization <= 5


def await_gpu(gpu: int) -> None:
    last_notice = 0.0
    while not gpu_idle(gpu):
        if time.monotonic() - last_notice > 120:
            emit(f"gpu={gpu} waiting for clean idle state; respecting current owner")
            last_notice = time.monotonic()
        time.sleep(20)


def checked_existing(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = load(path)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"incomplete existing report: {path}") from exc
    if value.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"existing report has wrong protocol: {path}")
    return True


def run_eval(
    gpu: int,
    target_id: str,
    target: dict[str, Any],
    candidate: dict[str, Any],
    stage: str,
    plan: dict[str, Any],
) -> None:
    split = plan["splits"][stage]
    count = split.get("trajectories_per_trained_candidate")
    if count is None:
        raise RuntimeError(f"unsupported candidate stage: {stage}")
    iteration = candidate["response_iteration"]
    output = file_for(target_id, stage, iteration, count)
    if checked_existing(output):
        emit(f"gpu={gpu} reuse {output.name}")
        return
    await_gpu(gpu)
    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(CUDA_HOME)
    environment["PATH"] = (
        f"/home/pablob/.venvs/poker/bin:{CUDA_HOME / 'bin'}:"
        f"{environment.get('PATH', '')}"
    )
    environment["LD_LIBRARY_PATH"] = (
        f"{CUDA_HOME / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    )
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        sys.executable, "-m", "pbg.cli.br_eval",
        "--probe", candidate["inference_artifact"],
        "--target", target["target_inference_artifact"],
        "--hands", str(count),
        "--split-namespace", split["namespace"],
        "--seed", str(split["seed"]),
        "--logical-deal-start", str(split["logical_deal_start"]),
        "--device", "cuda:0",
        "--output", str(output),
    ]
    emit(f"gpu={gpu} start {stage} {target_id} response={iteration} n={count}")
    subprocess.run(command, check=True, cwd=ROOT, env=environment)
    checked_existing(output)
    await_gpu(gpu)
    emit(f"gpu={gpu} complete {stage} {target_id} response={iteration}")


def selector(
    operation: str, target_id: str, plan_path: Path, manifest_path: Path
) -> Path:
    output = RESULTS / f"{'promotion' if operation == 'promote' else 'selection'}_{target_id}_v2_1.json"
    if output.exists():
        value = load(output)
        if value.get("protocol_version") != PROTOCOL:
            raise RuntimeError(f"existing selector manifest has wrong protocol: {output}")
        return output
    command = [
        sys.executable, "tools/br_protocol_v2_1_select.py", operation,
        "--plan", str(plan_path), "--manifest", str(manifest_path),
        "--results", str(RESULTS), "--target-id", target_id,
        "--output", str(output),
    ]
    if operation == "select":
        command += ["--promotion", str(RESULTS / f"promotion_{target_id}_v2_1.json")]
    subprocess.run(command, check=True, cwd=ROOT)
    emit(f"frozen {operation} {target_id}: {output.name}")
    return output


def completed_ids(plan: dict[str, Any]) -> list[str]:
    result = []
    for target in plan["targets"]:
        target_id = target["id"]
        selection_path = RESULTS / f"selection_{target_id}_v2_1.json"
        promotion_path = RESULTS / f"promotion_{target_id}_v2_1.json"
        if not selection_path.is_file() or not promotion_path.is_file():
            continue
        selection = load(selection_path)
        if not test_file(target_id, selection, plan).is_file():
            continue
        result.append(target_id)
    return result


def publish_plot(plan_path: Path, plan: dict[str, Any], target_id: str) -> None:
    with PLOT_LOCK:
        done = completed_ids(plan)
        if target_id not in done:
            raise RuntimeError(f"cannot plot incomplete target {target_id}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = f"full_curve_{len(done):02d}_{target_id}_{stamp}"
        command = [
            sys.executable, "tools/br_protocol_v2_1_plot.py",
            "--plan", str(plan_path),
            "--results", str(RESULTS),
            "--output-dir", str(RESULTS / "figures_full_curve_v2_1"),
            "--prefix", prefix,
        ]
        for item in done:
            command += [
                "--promotion", str(RESULTS / f"promotion_{item}_v2_1.json"),
                "--selection", str(RESULTS / f"selection_{item}_v2_1.json"),
            ]
        subprocess.run(command, check=True, cwd=ROOT)
        emit(f"plot updated for {len(done)} targets: {prefix}")


def run_target(
    gpu: int,
    target_id: str,
    plan_path: Path,
    manifest_path: Path,
    plan: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    target = inventory[target_id]
    candidates = {
        item["response_iteration"]: item for item in target["candidates"]
    }
    if len(candidates) != 6:
        raise RuntimeError(f"{target_id}: incomplete six-response inventory")
    selection_path = RESULTS / f"selection_{target_id}_v2_1.json"
    if selection_path.is_file() and test_file(target_id, load(selection_path), plan).is_file():
        emit(f"gpu={gpu} reuse completed target {target_id}")
        return
    for iteration in plan["candidate_grid"]["trained_response_iterations"]:
        run_eval(gpu, target_id, target, candidates[iteration], "screening", plan)
    promotion_path = selector("promote", target_id, plan_path, manifest_path)
    promotion = load(promotion_path)
    promoted = promotion["targets"][0]["promoted_response_iterations"]
    if len(promoted) != 3:
        raise RuntimeError(f"{target_id}: promotion did not select exactly three")
    for iteration in promoted:
        run_eval(gpu, target_id, target, candidates[iteration], "validation", plan)
    selection_path = selector("select", target_id, plan_path, manifest_path)
    selection = load(selection_path)
    if selection.get("test_results_inspected") is not False:
        raise RuntimeError(f"{target_id}: selection was not frozen before test")
    output = test_file(target_id, selection, plan)
    if not checked_existing(output):
        await_gpu(gpu)
        command = [
            sys.executable, "tools/run_br_protocol_v2_1_selected.py",
            "--physical-gpu", str(gpu),
            "--plan", str(plan_path),
            "--manifest", str(manifest_path),
            "--selection", str(selection_path),
            "--results", str(RESULTS),
        ]
        emit(f"gpu={gpu} start frozen test {target_id}")
        subprocess.run(command, check=True, cwd=ROOT)
        checked_existing(output)
        await_gpu(gpu)
    emit(f"gpu={gpu} completed target {target_id}")
    publish_plot(plan_path, plan, target_id)


def create_queue(plan_path: Path, manifest_path: Path, output: Path) -> None:
    plan = load(plan_path)
    manifest = load(manifest_path)
    if any(value["protocol_version"] != PROTOCOL for value in (plan, manifest)):
        raise RuntimeError("queue requires matching v2.1 plan and manifest")
    available = {item["id"] for item in plan["targets"]}
    if not set(ORDER) <= available or len(set(ORDER)) != len(ORDER):
        raise RuntimeError("queue order differs from available original targets")
    document = {
        "schema": "pbg-fixed-target-br-full-curve-queue-v2.1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_version": PROTOCOL,
        "plan": str(plan_path),
        "plan_sha256": digest(plan_path),
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "queued_target_ids": list(ORDER),
        "in_flight_priority_target": "rb_i6600",
        "reused_completed_target_ids": [
            "normal_i10000", "rb_i10000", "rb_i7700",
        ],
        "unavailable_original_target_ids": [
            item["id"] for item in plan["unavailable_original_targets"]
        ],
        "physical_gpus": [0, 1, 2],
        "gpu_0_reserved_until": "rb_i6600 frozen v2.1 test report exists and GPU 0 is idle",
        "evaluation_budgets": plan["splits"],
        "selection_rule": plan["selection"],
        "response_training_semantics": "adaptive_mixed_profile_range_v1",
        "target_evaluation_semantics": "fixed_target_reference_range_board_conditioned_v2",
        "estimated_available_curve_wall_hours": [22, 30],
        "estimate_scope": "16 measurable original targets on three idle GPUs; excludes missing-response targets and interference from other users",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(document, destination, indent=2, sort_keys=True)
        destination.write("\n")
    emit(f"queue committed: {output}")


def run_queue(queue_path: Path) -> None:
    queued = load(queue_path)
    if queued["protocol_version"] != PROTOCOL:
        raise RuntimeError("queue protocol mismatch")
    plan_path = Path(queued["plan"]).resolve()
    manifest_path = Path(queued["manifest"]).resolve()
    if digest(plan_path) != queued["plan_sha256"]:
        raise RuntimeError("queued plan hash changed")
    if digest(manifest_path) != queued["manifest_sha256"]:
        raise RuntimeError("queued run-manifest hash changed")
    plan, manifest = load(plan_path), load(manifest_path)
    for path, expected in manifest["evaluation_code_sha256"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"evaluator source changed: {path}")
    inventory = {item["id"]: item for item in manifest["priority_targets"]}
    pending: queue.Queue[str] = queue.Queue()
    for item in queued["queued_target_ids"]:
        if item not in inventory:
            raise RuntimeError(f"queued target missing from manifest: {item}")
        pending.put(item)
    lock_path = RESULTS / "full_curve_queue_v2_1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = []
        def worker(gpu: int) -> None:
            if gpu == 0:
                priority_selection_path = RESULTS / "selection_rb_i6600_v2_1.json"
                while True:
                    if priority_selection_path.is_file():
                        priority_test = test_file(
                            "rb_i6600", load(priority_selection_path), plan
                        )
                        if priority_test.is_file():
                            break
                    time.sleep(30)
                emit("gpu=0 priority RB 6.6k test complete; joining full-curve queue")
            while True:
                try:
                    target_id = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    run_target(
                        gpu, target_id, plan_path, manifest_path, plan, inventory
                    )
                except (
                    RuntimeError,
                    subprocess.CalledProcessError,
                    OSError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ) as exc:
                    emit(f"FAILED gpu={gpu} target={target_id}: {exc!r}")
                    with PRINT_LOCK:
                        failures.append(target_id)
                finally:
                    pending.task_done()
        emit(f"queue start: {list(queued['queued_target_ids'])}")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, gpu) for gpu in (1, 2, 0)]
            for future in futures:
                future.result()
        emit(f"queue finished; failed={failures}")
        if failures:
            raise RuntimeError(f"full-curve targets failed: {failures}")


def detach(queue_path: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as output:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "run",
             "--queue", str(queue_path)],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    print(json.dumps({"pid": child.pid, "log": str(log_path)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--plan", required=True, type=Path)
    create.add_argument("--manifest", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--queue", required=True, type=Path)
    start = subparsers.add_parser("detach")
    start.add_argument("--queue", required=True, type=Path)
    start.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "create":
        create_queue(args.plan.resolve(), args.manifest.resolve(), args.output.resolve())
    elif args.operation == "run":
        run_queue(args.queue.resolve())
    else:
        detach(args.queue.resolve(), args.log.resolve())


if __name__ == "__main__":
    main()
