"""Turnkey H100 selection experiment for independent Normal/RB replicas."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import torch

from pbg.config import load_config
from pbg.core.checkpointing import config_fingerprint


GIB = 2**30


def _command_output(arguments: Sequence[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            list(arguments), cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        values = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(value, dict) for value in values):
        raise TypeError(f"{path} contained a non-object JSONL record")
    return values


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return False


class ExperimentDeadline:
    def __init__(self, minutes: float) -> None:
        if not math.isfinite(minutes) or minutes <= 0.0:
            raise ValueError("deadline minutes must be finite and positive")
        self.started = time.monotonic()
        self.deadline = self.started + minutes * 60.0

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self) -> float:
        return max(self.deadline - time.monotonic(), 0.0)


def _run_stage(
    *,
    name: str,
    command: list[str],
    output_dir: Path,
    deadline: ExperimentDeadline,
    root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[runpod-selection] starting {name}", flush=True)
    command_path = output_dir / "command.json"
    command_path.write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    remaining = deadline.remaining_seconds
    if remaining <= 1.0:
        print(
            f"[runpod-selection] skipped {name}: deadline exhausted",
            flush=True,
        )
        return {
            "name": name,
            "status": "deadline_exhausted",
            "seconds": 0.0,
        }
    started = time.monotonic()
    try:
        with (output_dir / "stdout.log").open(
            "w", encoding="utf-8"
        ) as stdout, (output_dir / "stderr.log").open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=stdout,
                stderr=stderr,
                timeout=remaining,
                check=False,
                env=os.environ.copy(),
            )
        status = "ok" if completed.returncode == 0 else "failed"
        print(
            f"[runpod-selection] finished {name}: {status} "
            f"({time.monotonic() - started:.1f}s)",
            flush=True,
        )
        return {
            "name": name,
            "status": status,
            "returncode": completed.returncode,
            "seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired:
        print(
            f"[runpod-selection] stopped {name}: deadline exhausted",
            flush=True,
        )
        return {
            "name": name,
            "status": "deadline_exhausted",
            "seconds": time.monotonic() - started,
        }


def _base_command(
    *,
    config: Path,
    hardware_profile: Path,
    replicas: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pbg.cli.train_replicas",
        "--config",
        str(config),
        "--hardware-profile",
        str(hardware_profile),
        "--replicas",
        str(replicas),
        "--out-root",
        str(output_dir),
        "--no-checkpoints",
    ]


def _summarize_end_to_end(
    output_dir: Path,
    *,
    measured_iterations: tuple[int, ...],
    kernel_iteration: int,
) -> dict[str, Any]:
    records = _read_jsonl(output_dir / "aggregate_metrics.jsonl")
    by_iteration = {int(record["iter"]): record for record in records}
    missing = set((*measured_iterations, kernel_iteration)).difference(by_iteration)
    if missing:
        raise RuntimeError(f"end-to-end output omitted iterations {sorted(missing)}")
    measured = [by_iteration[index] for index in measured_iterations]
    kernel = by_iteration[kernel_iteration]
    seconds = sum(float(record["physical_gpu_seconds"]) for record in measured)
    decisions = sum(int(record["fresh_decisions"]) for record in measured)
    per_lane_records: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(output_dir.glob("replica_*/metrics.jsonl")):
        lane = int(path.parent.name.removeprefix("replica_"))
        lane_values = {
            int(record["iter"]): record for record in _read_jsonl(path)
        }
        per_lane_records[lane] = [
            lane_values[index] for index in measured_iterations
        ]
    objective_mass = [
        record["objective_mass"] for record in measured
    ]
    steps = [record["optimizer_steps_applied"] for record in measured]
    finite = (
        _finite_tree(measured)
        and _finite_tree(per_lane_records)
        and all(
            float(value) > 0.0
            for iteration_mass in objective_mass
            for value in iteration_mass
        )
        and all(
            float(value) > 0.0
            for iteration_steps in steps
            for value in iteration_steps
        )
        and all(
            math.isclose(
                float(iteration_mass[lane]),
                float(steps[record_index][lane]),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            for record_index, iteration_mass in enumerate(objective_mass)
            for lane in range(len(iteration_mass))
        )
    )
    peak_allocated = max(
        int(record["peak_allocated_bytes"]) for record in records
    )
    peak_reserved = max(
        int(record["peak_reserved_bytes"]) for record in records
    )
    return {
        "measured_iterations": list(measured_iterations),
        "kernel_probe_iteration": kernel_iteration,
        "physical_gpu_seconds": seconds,
        "fresh_decisions": decisions,
        "fresh_decisions_per_second": decisions / max(seconds, 1.0e-12),
        "fresh_decisions_per_h100_hour": decisions
        / max(seconds, 1.0e-12)
        * 3600.0,
        "cuda_kernels": kernel["cuda_kernels"],
        "kernel_probe_decisions": kernel["fresh_decisions"],
        "cuda_kernels_per_decision": kernel[
            "cuda_kernels_per_decision"
        ],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_memory_bytes": max(peak_allocated, peak_reserved),
        "phase_seconds": {
            phase: sum(
                float(record["phase_seconds"][phase]) for record in measured
            )
            for phase in ("snapshot", "rollout", "backward", "seal")
        }
        | {
            "optimization": sum(
                float(record["optimization_seconds"]) for record in measured
            )
        },
        "objective_mass": objective_mass,
        "optimizer_steps_applied": steps,
        "per_replica": per_lane_records,
        "finite_and_consistent": finite,
    }


def _select(
    candidates: list[dict[str, Any]],
    *,
    minimum_gain: float,
    peak_limit_bytes: int,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    selected: int | None = None
    for index, candidate in enumerate(candidates):
        reasons: list[str] = []
        summary = candidate.get("summary")
        learner = candidate.get("learner")
        passed = candidate.get("status") == "ok"
        if not passed or not isinstance(summary, dict):
            reasons.append("execution_failure")
        else:
            peak_memory = int(
                summary.get(
                    "peak_memory_bytes",
                    summary.get(
                        "peak_reserved_bytes",
                        summary["peak_allocated_bytes"],
                    ),
                )
            )
            if peak_memory >= peak_limit_bytes:
                reasons.append("peak_memory")
            if not summary["finite_and_consistent"]:
                reasons.append("counter_objective_or_finite")
            if (
                summary["cuda_kernels_per_decision"] is None
                or float(summary["cuda_kernels_per_decision"]) <= 0.0
            ):
                reasons.append("kernel_measurement")
            if not isinstance(learner, dict) or not all(learner["finite"]):
                reasons.append("learner_finite")
            elif any(
                not math.isclose(
                    float(value),
                    float(learner["measured_steps"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
                for value in learner["objective_mass"]
            ):
                reasons.append("learner_objective_mass")
            if index:
                previous = candidates[index - 1]
                previous_summary = previous.get("summary")
                if previous.get("status") != "ok" or not isinstance(
                    previous_summary, dict
                ):
                    reasons.append("previous_candidate_unavailable")
                    previous_summary = None
                if previous_summary is not None:
                    gain = (
                        summary["fresh_decisions_per_second"]
                        / previous_summary["fresh_decisions_per_second"]
                        - 1.0
                    )
                    if gain < minimum_gain:
                        reasons.append("throughput_gain")
                    if (
                        summary["cuda_kernels_per_decision"] is None
                        or previous_summary["cuda_kernels_per_decision"] is None
                        or summary["cuda_kernels_per_decision"]
                        >= previous_summary["cuda_kernels_per_decision"]
                    ):
                        reasons.append("kernel_reduction")
        passed = not reasons
        gates.append(
            {
                "replicas": candidate["replicas"],
                "passed": passed,
                "reasons": reasons,
            }
        )
        if passed:
            selected = int(candidate["replicas"])
    return {
        "replicas": selected,
        "minimum_throughput_gain": minimum_gain,
        "peak_limit_bytes": peak_limit_bytes,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete <=90-minute H100 replica selection matrix"
    )
    parser.add_argument("--normal-config", required=True, type=Path)
    parser.add_argument("--rb-config", required=True, type=Path)
    parser.add_argument("--hardware-profile", required=True, type=Path)
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=os.environ.get("PBG_EVALUATOR_ROOT"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--deadline-minutes", type=float, default=90.0)
    parser.add_argument("--minimum-throughput-gain", type=float, default=0.10)
    parser.add_argument("--peak-limit-gib", type=float, default=70.0)
    parser.add_argument("--learner-warmup", type=int, default=10)
    parser.add_argument("--learner-steps", type=int, default=50)
    parser.add_argument("--allow-non-h100", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    configs = {
        "normal": args.normal_config.resolve(),
        "rb": args.rb_config.resolve(),
    }
    profile = args.hardware_profile.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        parser.error("output-root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    if (
        not args.replicas
        or any(value <= 0 for value in args.replicas)
        or list(args.replicas) != sorted(set(args.replicas))
    ):
        parser.error("replica candidates must be unique positive ascending integers")
    if args.learner_warmup < 0 or args.learner_steps <= 0:
        parser.error("learner warmup/steps are invalid")
    if (
        not math.isfinite(args.deadline_minutes)
        or not 0.0 < args.deadline_minutes <= 90.0
    ):
        parser.error("RunPod selection deadline must lie in (0,90] minutes")
    if not 0.0 <= args.minimum_throughput_gain < 1.0:
        parser.error("minimum throughput gain must lie in [0,1)")
    if not 0.0 < args.peak_limit_gib <= 70.0:
        parser.error("peak limit must lie in (0,70] GiB")
    deadline = ExperimentDeadline(args.deadline_minutes)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("CUDA_VISIBLE_DEVICES must select the single RunPod H100")

    gpu_query = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,uuid,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        cwd=root,
    )
    if gpu_query is None or len(gpu_query.splitlines()) != 1:
        parser.error("exactly one visible GPU is required")
    gpu_fields = [field.strip() for field in gpu_query.split(",")]
    if not args.allow_non_h100 and (
        "H100" not in gpu_fields[0] or gpu_fields[-1] != "9.0"
    ):
        parser.error(f"selection requires an H100 SM90, observed {gpu_query}")
    driver_major = int(gpu_fields[2].split(".", 1)[0])
    cuda_version = tuple(
        int(value) for value in (torch.version.cuda or "0.0").split(".")[:2]
    )
    if cuda_version >= (13, 0) and driver_major < 580:
        parser.error("CUDA 13 requires an R580-or-newer RunPod driver")
    if (12, 8) <= cuda_version < (13, 0) and driver_major < 570:
        parser.error("CUDA 12.8 requires an R570-or-newer RunPod driver")
    evaluator_root = (
        None
        if args.evaluator_root is None
        else args.evaluator_root.resolve()
    )
    if evaluator_root is not None and not evaluator_root.is_dir():
        parser.error("evaluator-root is not a directory")

    loaded = {
        mode: load_config(path, hardware_profile=profile)
        for mode, path in configs.items()
    }
    if loaded["normal"].training.mode != "normal":
        parser.error("normal config does not select Normal")
    if loaded["rb"].training.mode != "rao_blackwell":
        parser.error("RB config does not select sequential Rao-Blackwell")
    if asdict(loaded["normal"].game) != asdict(loaded["rb"].game):
        parser.error("Normal/RB configs use different games")
    if loaded["normal"].model != loaded["rb"].model:
        parser.error("Normal/RB configs use different models")
    if loaded["normal"].training.seed != loaded["rb"].training.seed:
        parser.error("Normal/RB configs must use matched training seeds")
    matched_training_fields = (
        "gd_steps_per_iter",
        "minibatch_size",
        "lr",
        "adam_betas",
        "optimizer",
        "grad_clip",
        "amp",
        "ppo_epochs",
        "reproducibility",
    )
    if any(
        getattr(loaded["normal"].training, name)
        != getattr(loaded["rb"].training, name)
        for name in matched_training_fields
    ):
        parser.error("Normal/RB optimizer and learner hyperparameters differ")
    matched_loss_fields = (
        "lam",
        "clip_eps",
        "clip_eps_v",
        "c_v",
        "c_h",
        "c_kl",
        "entropy_schedule",
        "entropy_decay_iters",
        "utility_normalization",
        "value_loss",
    )
    if any(
        getattr(loaded["normal"].loss, name)
        != getattr(loaded["rb"].loss, name)
        for name in matched_loss_fields
    ):
        parser.error("Normal/RB shared loss hyperparameters differ")

    report: dict[str, Any] = {
        "schema": "pbg-runpod-h100-replica-selection-v1",
        "status": "running",
        "source_revision": _command_output(
            ["git", "rev-parse", "HEAD"], cwd=root
        ),
        "source_dirty": bool(
            _command_output(["git", "status", "--porcelain"], cwd=root)
        ),
        "evaluator_root": (
            None if evaluator_root is None else str(evaluator_root)
        ),
        "evaluator_revision": (
            None
            if evaluator_root is None
            else _command_output(
                ["git", "rev-parse", "HEAD"], cwd=evaluator_root
            )
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu_query,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "deadline_minutes": args.deadline_minutes,
        "configs": {
            mode: {
                "path": str(configs[mode]),
                "fingerprint": config_fingerprint(loaded[mode]),
                "resolved": asdict(loaded[mode]),
            }
            for mode in configs
        },
        "parity": {},
        "learner": {},
        "end_to_end": {},
        "stages": [],
    }
    report_path = output_root / "selection_report.json"

    try:
        for mode in ("normal", "rb"):
            stage_dir = output_root / "parity" / mode
            command = _base_command(
                config=configs[mode],
                hardware_profile=profile,
                replicas=1,
                output_dir=stage_dir,
            ) + ["--max-iterations", "1", "--verify-legacy-k1"]
            stage = _run_stage(
                name=f"parity_{mode}",
                command=command,
                output_dir=stage_dir,
                deadline=deadline,
                root=root,
            )
            report["stages"].append(stage)
            if stage["status"] == "ok":
                report["parity"][mode] = _read_json(
                    stage_dir / "legacy_k1_parity.json"
                )
            else:
                report["parity"][mode] = stage

        failed_parity = [
            mode
            for mode in ("normal", "rb")
            if report["parity"].get(mode, {}).get("passed") is not True
        ]
        if failed_parity:
            raise RuntimeError(
                "K=1 parity failed for "
                + ", ".join(failed_parity)
                + "; refusing to spend GPU time on selection benchmarks"
            )

        for mode in ("normal", "rb"):
            report["learner"][mode] = []
            for replicas in args.replicas:
                stage_dir = (
                    output_root / "learner" / mode / f"k{replicas}"
                )
                command = _base_command(
                    config=configs[mode],
                    hardware_profile=profile,
                    replicas=replicas,
                    output_dir=stage_dir,
                ) + [
                    "--max-iterations",
                    "1",
                    "--learner-benchmark-warmup",
                    str(args.learner_warmup),
                    "--learner-benchmark-steps",
                    str(args.learner_steps),
                ]
                stage = _run_stage(
                    name=f"learner_{mode}_k{replicas}",
                    command=command,
                    output_dir=stage_dir,
                    deadline=deadline,
                    root=root,
                )
                report["stages"].append(stage)
                candidate = {"replicas": replicas, "status": stage["status"]}
                if stage["status"] == "ok":
                    candidate.update(
                        _read_json(stage_dir / "learner_benchmark.json")
                    )
                else:
                    candidate["stage"] = stage
                report["learner"][mode].append(candidate)

        for mode in ("normal", "rb"):
            report["end_to_end"][mode] = []
            learners = {
                item["replicas"]: item for item in report["learner"][mode]
            }
            for replicas in args.replicas:
                stage_dir = (
                    output_root / "end_to_end" / mode / f"k{replicas}"
                )
                command = _base_command(
                    config=configs[mode],
                    hardware_profile=profile,
                    replicas=replicas,
                    output_dir=stage_dir,
                ) + [
                    "--max-iterations",
                    "5",
                    "--profile-kernel-iterations",
                    "4",
                ]
                stage = _run_stage(
                    name=f"end_to_end_{mode}_k{replicas}",
                    command=command,
                    output_dir=stage_dir,
                    deadline=deadline,
                    root=root,
                )
                report["stages"].append(stage)
                candidate = {
                    "replicas": replicas,
                    "status": stage["status"],
                    "learner": learners[replicas]
                    if learners[replicas].get("status") == "ok"
                    else None,
                }
                if stage["status"] == "ok":
                    candidate["summary"] = _summarize_end_to_end(
                        stage_dir,
                        measured_iterations=(1, 2, 3),
                        kernel_iteration=4,
                    )
                else:
                    candidate["stage"] = stage
                report["end_to_end"][mode].append(candidate)

        parity_ok = all(
            report["parity"].get(mode, {}).get("passed") is True
            for mode in ("normal", "rb")
        )
        selections = {
            mode: _select(
                report["end_to_end"][mode],
                minimum_gain=args.minimum_throughput_gain,
                peak_limit_bytes=int(args.peak_limit_gib * GIB),
            )
            for mode in ("normal", "rb")
        }
        report["selection"] = selections
        report["parity_passed"] = parity_ok
        report["status"] = (
            "complete"
            if parity_ok
            and all(value["replicas"] is not None for value in selections.values())
            else "incomplete"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = deadline.elapsed_seconds
        report["remaining_seconds"] = deadline.remaining_seconds
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report["selection"], indent=2, sort_keys=True))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
