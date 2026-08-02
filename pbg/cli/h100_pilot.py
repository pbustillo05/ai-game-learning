"""Prepare, smoke-test, and run the deadline-bounded one-H100 pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from pbg.benchmark import benchmark_training
from pbg.config import load_config
from pbg.h100_pilot import (
    H100_MEMORY_BUDGET_BYTES,
    RB_SAMPLE_COUNTS,
    Arm,
    PilotError,
    base_plan,
    benchmark_summary,
    export_model_initializer,
    geometry_roots,
    h100_preflight,
    load_model_initializer,
    select_replica_count,
    write_arm_config,
    write_json,
)

DEFAULT_NORMAL_CHECKPOINT = Path(
    "runs/nlhe_hu_32bb_24h_normal_20260722/ckpt_0000700.pt"
)
DEFAULT_RB_CHECKPOINT = Path(
    "runs/nlhe_hu_32bb_24h_rb_20260722/ckpt_0000700.pt"
)


def _arm(value: dict[str, object]) -> Arm:
    return Arm(**value)  # type: ignore[arg-type]


def _prepare(args: argparse.Namespace) -> None:
    destination = Path(args.directory)
    configs = Path(args.configs_directory)
    artifacts = destination / "artifacts"
    reports = destination / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    initializers = {}
    for mode, source in (
        ("normal", Path(args.normal_checkpoint)),
        ("rb", Path(args.rb_checkpoint)),
    ):
        target = artifacts / f"nlhe_hu_32bb_{mode}_initializer.pt"
        initializers[mode] = export_model_initializer(source, target)
    plan = base_plan()
    plan["directory"] = str(destination)
    plan["configs_directory"] = str(configs)
    plan["reports_directory"] = str(reports)
    plan["initializers"] = initializers
    write_json(plan, destination / "plan.json")
    for value in plan["arms"]:
        arm = _arm(value)
        write_arm_config(
            arm,
            configs / f"{arm.name}.yaml",
            output_dir=destination / "runtime" / arm.name,
        )
    print(destination / "plan.json")


def _run_arm(args: argparse.Namespace) -> None:
    if not args.allow_non_h100:
        hardware = h100_preflight()
    else:
        hardware = {"preflight": "explicitly bypassed for local smoke"}
    arm = Arm(
        name=args.name,
        stage=args.stage,
        mode=args.mode,
        replicas=args.replicas,
        roots=args.roots,
        completion_samples=args.completion_samples,
        rollout_pack=args.rollout_pack,
        completion_pack=args.completion_pack,
        learner_pack=args.learner_pack,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    config = load_config(args.config)
    if config.execution.manifest.n_replica != arm.replicas:
        raise PilotError("arm replica count differs from strict config")
    if config.scientific.common.root_count != arm.roots:
        raise PilotError("arm root count differs from strict config")
    state, initializer = load_model_initializer(args.initializer)
    benchmark = benchmark_training(
        config,
        warmup=arm.warmup,
        repeats=arm.repeats,
        initial_model_state=state,
        progress_path=Path(args.report).with_suffix(".progress.json"),
    )
    report = {
        "summary": benchmark_summary(arm, benchmark),
        "arm": asdict(arm),
        "hardware_preflight": hardware,
        "initializer": initializer,
        "benchmark": benchmark,
    }
    write_json(report, args.report)


def _invoke_arm(
    arm: Arm,
    *,
    directory: Path,
    configs_directory: Path,
    initializer: Path,
    hardware: str,
    timeout_seconds: float | None,
) -> dict[str, object]:
    config = configs_directory / f"{arm.name}.yaml"
    report = directory / "reports" / f"{arm.name}.json"
    write_arm_config(
        arm,
        config,
        output_dir=directory / "runtime" / arm.name,
        hardware=hardware,
    )
    command = [
        sys.executable,
        "-m",
        "pbg.cli.h100_pilot",
        "arm",
        "--name",
        arm.name,
        "--stage",
        arm.stage,
        "--mode",
        arm.mode,
        "--replicas",
        str(arm.replicas),
        "--roots",
        str(arm.roots),
        "--rollout-pack",
        str(arm.rollout_pack),
        "--completion-pack",
        str(arm.completion_pack),
        "--learner-pack",
        str(arm.learner_pack),
        "--warmup",
        str(arm.warmup),
        "--repeats",
        str(arm.repeats),
        "--config",
        str(config),
        "--initializer",
        str(initializer),
        "--report",
        str(report),
    ]
    if arm.completion_samples is not None:
        command.extend(
            ["--completion-samples", str(arm.completion_samples)]
        )
    if hardware == "local":
        command.append("--allow-non-h100")
    subprocess.run(
        command,
        check=True,
        timeout=timeout_seconds,
    )
    value = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotError(f"{arm.name} emitted a non-object report")
    return value


def _record_failure(
    arm: Arm, error: BaseException
) -> dict[str, object]:
    return {
        "arm": asdict(arm),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _run(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "pbg-h100-pilot-plan-v1":
        raise PilotError("unsupported pilot plan")
    hardware = h100_preflight()
    directory = Path(plan["directory"])
    configs_directory = Path(plan["configs_directory"])
    initializer_paths = {
        mode: Path(plan["initializers"][mode]["path"])
        for mode in ("normal", "rb")
    }
    deadline_minutes = float(
        args.deadline_minutes
        if args.deadline_minutes is not None
        else plan["deadline_minutes"]
    )
    reserve_seconds = float(plan["shutdown_reserve_minutes"]) * 60.0
    deadline = time.monotonic() + deadline_minutes * 60.0
    state: dict[str, object] = {
        "schema": "pbg-h100-pilot-run-v1",
        "plan": str(plan_path),
        "hardware": hardware,
        "deadline_minutes": deadline_minutes,
        "completed": [],
        "failures": [],
        "selected_replicas": {},
        "selected_roots": {},
        "deadline_reached": False,
    }
    state_path = directory / "run_state.json"

    def execute(arm: Arm, *, required: bool = False) -> dict[str, object] | None:
        remaining = deadline - time.monotonic() - reserve_seconds
        if remaining <= 0:
            state["deadline_reached"] = True
            write_json(state, state_path)
            return None
        try:
            report = _invoke_arm(
                arm,
                directory=directory,
                configs_directory=configs_directory,
                initializer=initializer_paths[arm.mode],
                hardware="h100",
                timeout_seconds=remaining,
            )
        except (OSError, subprocess.SubprocessError, PilotError) as error:
            state["failures"].append(_record_failure(arm, error))
            write_json(state, state_path)
            if required:
                raise
            return None
        state["completed"].append(report["summary"])
        write_json(state, state_path)
        return report

    arms = [_arm(value) for value in plan["arms"]]
    for arm in (value for value in arms if value.stage == "smoke"):
        execute(arm, required=True)
    scaling: dict[str, list[dict[str, object]]] = {
        "normal": [],
        "rb": [],
    }
    for arm in (value for value in arms if value.stage == "replica_scaling"):
        report = execute(arm)
        if report is not None:
            scaling[arm.mode].append(report["summary"])
    for mode in ("normal", "rb"):
        selected = select_replica_count(
            scaling[mode],
            minimum_gain=float(plan["replica_gain_threshold"]),
            memory_ceiling_bytes=int(plan["memory_ceiling_bytes"]),
        )
        state["selected_replicas"][mode] = selected
        geometry_reports = []
        for roots in geometry_roots(selected):
            arm = Arm(
                f"nlhe_hu_32bb_{mode}_h100_k{selected}_roots{roots}",
                "geometry",
                mode,
                selected,
                roots,
                1024 if mode == "rb" else None,
            )
            report = execute(arm)
            if report is not None:
                geometry_reports.append(report["summary"])
        passing = [
            report
            for report in geometry_reports
            if int(report["peak_reserved_bytes"]) <= H100_MEMORY_BUDGET_BYTES
        ]
        if not passing:
            raise PilotError(f"{mode} geometry stage has no passing arm")
        winner = max(
            passing,
            key=lambda report: float(report["player_nodes_per_second"]),
        )
        state["selected_roots"][mode] = int(winner["roots_per_replica"])
        write_json(state, state_path)
    rb_k = int(state["selected_replicas"]["rb"])
    rb_roots = int(state["selected_roots"]["rb"])
    for samples in RB_SAMPLE_COUNTS:
        execute(
            Arm(
                f"nlhe_hu_32bb_rb_h100_k{rb_k}_roots{rb_roots}_samples{samples}",
                "rb_samples",
                "rb",
                rb_k,
                rb_roots,
                samples,
            )
        )
    state["elapsed_seconds"] = deadline_minutes * 60.0 - max(
        deadline - time.monotonic(), 0.0
    )
    write_json(state, state_path)
    (directory / "PILOT_COMPLETE").touch()
    print(state_path)


def _local_smoke(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise PilotError("local smoke requires CUDA")
    device = torch.device("cuda:0")
    directory = Path(args.directory)
    configs_directory = Path(args.configs_directory)
    initializer_paths = {
        "normal": Path(args.normal_initializer),
        "rb": Path(args.rb_initializer),
    }
    reports = []
    arms = (
        Arm("nlhe_hu_32bb_normal_local_smoke_k1", "local_smoke", "normal", 1, 1, None, rollout_pack=1, completion_pack=1, learner_pack=1, warmup=0, repeats=1),
        Arm("nlhe_hu_32bb_normal_local_smoke_k2", "local_smoke", "normal", 2, 1, None, rollout_pack=1, completion_pack=1, learner_pack=1, warmup=0, repeats=1),
        Arm("nlhe_hu_32bb_rb_local_smoke_k1", "local_smoke", "rb", 1, 1, 4, rollout_pack=1, completion_pack=1, learner_pack=1, warmup=0, repeats=1),
        Arm("nlhe_hu_32bb_rb_local_smoke_k2", "local_smoke", "rb", 2, 1, 4, rollout_pack=1, completion_pack=1, learner_pack=1, warmup=0, repeats=1),
    )
    for arm in arms:
        try:
            reports.append(
                _invoke_arm(
                    arm,
                    directory=directory,
                    configs_directory=configs_directory,
                    initializer=initializer_paths[arm.mode],
                    hardware="local",
                    timeout_seconds=args.timeout_minutes * 60.0,
                )["summary"]
            )
        except (OSError, subprocess.SubprocessError, PilotError) as error:
            reports.append(_record_failure(arm, error))
            if args.fail_fast:
                raise
    destination = directory / "local_smoke.json"
    write_json(
        {
            "schema": "pbg-h100-pilot-local-smoke-v1",
            "device_name": torch.cuda.get_device_properties(device).name,
            "reports": reports,
        },
        destination,
    )
    print(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--directory",
        default="runs/experiments/nlhe_hu_32bb_h100_pilot",
    )
    prepare.add_argument(
        "--configs-directory",
        default="configs/experiments/nlhe_hu_32bb_h100_pilot",
    )
    prepare.add_argument(
        "--normal-checkpoint", default=str(DEFAULT_NORMAL_CHECKPOINT)
    )
    prepare.add_argument("--rb-checkpoint", default=str(DEFAULT_RB_CHECKPOINT))
    prepare.set_defaults(function=_prepare)

    arm = commands.add_parser("arm")
    arm.add_argument("--name", required=True)
    arm.add_argument("--stage", required=True)
    arm.add_argument("--mode", choices=("normal", "rb"), required=True)
    arm.add_argument("--replicas", type=int, required=True)
    arm.add_argument("--roots", type=int, required=True)
    arm.add_argument("--completion-samples", type=int)
    arm.add_argument("--rollout-pack", type=int, default=256)
    arm.add_argument("--completion-pack", type=int, default=512)
    arm.add_argument("--learner-pack", type=int, default=256)
    arm.add_argument("--warmup", type=int, default=1)
    arm.add_argument("--repeats", type=int, default=3)
    arm.add_argument("--config", required=True)
    arm.add_argument("--initializer", required=True)
    arm.add_argument("--report", required=True)
    arm.add_argument("--allow-non-h100", action="store_true")
    arm.set_defaults(function=_run_arm)

    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--deadline-minutes", type=float)
    run.set_defaults(function=_run)

    smoke = commands.add_parser("local-smoke")
    smoke.add_argument(
        "--directory",
        default="runs/experiments/nlhe_hu_32bb_h100_pilot_local",
    )
    smoke.add_argument(
        "--configs-directory",
        default="configs/experiments/nlhe_hu_32bb_h100_pilot_local",
    )
    smoke.add_argument("--normal-initializer", required=True)
    smoke.add_argument("--rb-initializer", required=True)
    smoke.add_argument("--timeout-minutes", type=float, default=20.0)
    smoke.add_argument("--fail-fast", action="store_true")
    smoke.set_defaults(function=_local_smoke)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "deadline_minutes", 1.0) is not None and getattr(
        args, "deadline_minutes", 1.0
    ) <= 0:
        _parser().error("--deadline-minutes must be positive")
    if getattr(args, "timeout_minutes", 1.0) <= 0:
        _parser().error("--timeout-minutes must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
