"""Prepare, smoke-test, and run the deadline-bounded one-H100 pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import shutil
from dataclasses import asdict, replace
from pathlib import Path


from pbg.benchmark import benchmark_training
from pbg.config import load_config
from pbg.core.persistence import sha256_file
from pbg.identity import code_identity
from pbg.h100_pilot import (
    PLAN_SCHEMA,
    Arm,
    PilotError,
    base_plan,
    benchmark_summary,
    export_model_initializer,
    h100_preflight,
    load_model_initializer,
    select_replica_count,
    selected_gpu_snapshot,
    write_arm_config,
    write_json,
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
    for mode, source in (("normal", args.normal_checkpoint), ("rb", args.rb_checkpoint)):
        target = artifacts / f"nlhe_hu_32bb_{mode}_initializer.pt"
        initializers[mode] = ({"kind": "seeded", "seed": 104830} if source is None else
                              export_model_initializer(Path(source), target))
    plan = base_plan()
    plan["directory"] = str(destination)
    plan["configs_directory"] = str(configs)
    plan["reports_directory"] = str(reports)
    plan["initializers"] = initializers
    plan["code"] = code_identity()
    write_json(plan, destination / "plan.json")
    for value in plan["arms"]:
        arm = _arm(value)
        write_arm_config(
            arm,
            configs / f"{arm.name}.yaml",
            output_dir=destination / "runtime" / arm.name,
            resolve=False,
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
        seed=args.seed,
    )
    config = load_config(args.config)
    if config.execution.manifest.n_replica != arm.replicas:
        raise PilotError("arm replica count differs from strict config")
    if config.scientific.common.root_count != arm.roots:
        raise PilotError("arm root count differs from strict config")
    state, initializer = (load_model_initializer(args.initializer) if args.initializer else
                          (None, {"kind": "seeded", "seed": config.scientific.common.seed}))
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
    initializer: Path | None,
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
        resolve=False,
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
        "--seed",
        str(arm.seed),
        "--report",
        str(report),
    ]
    if initializer is not None:
        command.extend(["--initializer", str(initializer)])
    if arm.completion_samples is not None:
        command.extend(
            ["--completion-samples", str(arm.completion_samples)]
        )
    if hardware == "local":
        command.append("--allow-non-h100")
    idle_before = selected_gpu_snapshot()
    try:
        subprocess.run(command, check=True, timeout=timeout_seconds)
    except BaseException:
        # Only this arm's generated configuration/runtime scratch is disposable.
        shutil.rmtree(directory / "runtime" / arm.name, ignore_errors=True)
        config.unlink(missing_ok=True)
        raise
    finally:
        idle_after = selected_gpu_snapshot()
        if idle_after["used_memory_mib"] > idle_before["used_memory_mib"] + 64:
            raise PilotError("selected GPU memory did not return to its idle baseline after child exit")
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


def _preflight(args: argparse.Namespace) -> None:
    report = h100_preflight()
    write_json(report, args.report)
    print(args.report)


def _run(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise PilotError("unsupported pilot plan; regenerate the shared-model rental plan")
    directory, configs_directory = Path(plan["directory"]), Path(plan["configs_directory"])
    identity = {"plan_sha256": sha256_file(plan_path), "code": code_identity()}
    if plan.get("code") != identity["code"]:
        raise PilotError("pilot plan code identity differs; prepare a new plan after code changes")
    state_path = directory / "run_state.json"
    if state_path.exists():
        if not args.resume:
            raise PilotError("pilot progress already exists; pass --resume or use a new directory")
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise PilotError("cannot reuse benchmark evidence from a different plan or code revision")
    else:
        state = {"schema": "pbg-h100-replica-run-v2", "identity": identity,
                 "completed": [], "artifacts": {}, "failures": [], "selected_replicas": {}}
    preflight = directory / "reports" / "preflight.json"
    # Keep the orchestration process free of CUDA contexts so each child leaves
    # the selected GPU idle and another process is never mistaken for our work.
    subprocess.run([sys.executable, "-m", "pbg.cli.h100_pilot", "preflight", "--report", str(preflight)], check=True)
    hardware = json.loads(preflight.read_text())
    state["hardware"] = hardware
    deadline_minutes = args.deadline_minutes or plan["deadline_minutes"]
    deadline = time.monotonic() + deadline_minutes * 60 - plan["shutdown_reserve_minutes"] * 60
    initializers = {mode: (Path(plan["initializers"][mode]["path"])
                          if "path" in plan["initializers"][mode] else None)
                    for mode in ("normal", "rb")}
    write_json(state, state_path)
    log = directory / "reports" / "correctness.log"
    if state.get("correctness_passed", False) and (
        not log.exists() or sha256_file(log) != state.get("correctness_log_sha256")
    ):
        raise PilotError("completed correctness evidence changed")
    if not state.get("correctness_passed", False):
        log = directory / "reports" / "correctness.log"
        with log.open("w") as output:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                                    stdout=output, stderr=subprocess.STDOUT,
                                    timeout=max(1, deadline - time.monotonic()))
        if result.returncode:
            state["failures"].append({"stage": "correctness", "log": str(log)})
            write_json(state, state_path)
            raise PilotError(f"correctness gates failed; inspect {log} before benchmarking")
        state["correctness_passed"] = True
        state["correctness_log_sha256"] = sha256_file(log)
        write_json(state, state_path)

    def execute(arm, *, required=False):
        report_path = directory / "reports" / f"{arm.name}.json"
        artifact = state["artifacts"].get(arm.name)
        if artifact is not None:
            if not report_path.exists() or sha256_file(report_path) != artifact:
                raise PilotError(f"completed evidence changed: {report_path}")
            return json.loads(report_path.read_text())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            state["deadline_reached"] = True
            write_json(state, state_path)
            return None
        try:
            report = _invoke_arm(arm, directory=directory, configs_directory=configs_directory,
                                 initializer=initializers[arm.mode], hardware="h100", timeout_seconds=remaining)
        except (OSError, subprocess.SubprocessError, PilotError, ValueError) as error:
            state["failures"].append(_record_failure(arm, error))
            write_json(state, state_path)
            if required:
                raise
            return None
        state["completed"].append(report["summary"])
        state["artifacts"][arm.name] = sha256_file(report_path)
        write_json(state, state_path)
        return report

    arms = [_arm(value) for value in plan["arms"]]
    for arm in (value for value in arms if value.stage == "smoke"):
        if execute(arm, required=True) is None:
            return
    scaling = {mode: [] for mode in ("normal", "rb")}
    for arm in (value for value in arms if value.stage == "replica_scaling"):
        report = execute(arm)
        if report is not None:
            scaling[arm.mode].append(report["summary"])
    for mode in ("normal", "rb"):
        if not any(row["replicas"] >= 16 for row in scaling[mode]):
            fallback = next(arm for arm in arms if arm.mode == mode and arm.stage == "fallback")
            report = execute(fallback)
            if report is not None:
                scaling[mode].append(report["summary"])
        if not any(row["replicas"] == 1 for row in scaling[mode]):
            write_json(state, state_path)
            return
        selected = select_replica_count(scaling[mode], minimum_gain=plan["replica_gain_threshold"],
                                        memory_ceiling_bytes=hardware["memory_budget_bytes"])
        state["selected_replicas"][mode] = selected
        baseline = next(row for row in scaling[mode] if row["replicas"] == 1)
        state.setdefault("matched_scaling", {})[mode] = [
            {"replicas": row["replicas"],
             "aggregate_rate_ratio": row["player_nodes_per_second"] / baseline["player_nodes_per_second"],
             "K_T1_over_TK": row["replicas"] * baseline["seconds"] / row["seconds"],
             "per_run_latency_seconds": row["seconds"]} for row in scaling[mode]]
        selected_arm = next(arm for arm in arms if arm.mode == mode and arm.replicas == selected
                            and arm.stage != "smoke")
        for seed in plan["validation_seeds"]:
            arm = replace(selected_arm, name=f"nlhe_hu_32bb_{mode}_h100_k{selected}_seed{seed}",
                          stage="seed_validation", seed=seed, repeats=3)
            if execute(arm, required=True) is None:
                return
    state["complete"] = True
    write_json(state, state_path)
    (directory / "PILOT_COMPLETE").touch()
    print(state_path)


def _local_smoke(args: argparse.Namespace) -> None:
    hardware = selected_gpu_snapshot()
    directory = Path(args.directory)
    configs_directory = Path(args.configs_directory)
    initializer_paths = {
        "normal": None if args.normal_initializer is None else Path(args.normal_initializer),
        "rb": None if args.rb_initializer is None else Path(args.rb_initializer),
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
            "device_name": hardware["name"],
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
        "--normal-checkpoint", default=None
    )
    prepare.add_argument("--rb-checkpoint", default=None)
    prepare.set_defaults(function=_prepare)

    arm = commands.add_parser("arm")
    arm.add_argument("--name", required=True)
    arm.add_argument("--stage", required=True)
    arm.add_argument("--mode", choices=("normal", "rb"), required=True)
    arm.add_argument("--replicas", type=int, required=True)
    arm.add_argument("--roots", type=int, required=True)
    arm.add_argument("--completion-samples", type=int)
    arm.add_argument("--rollout-pack", type=int, default=8)
    arm.add_argument("--completion-pack", type=int, default=64)
    arm.add_argument("--learner-pack", type=int, default=32)
    arm.add_argument("--warmup", type=int, default=1)
    arm.add_argument("--repeats", type=int, default=3)
    arm.add_argument("--config", required=True)
    arm.add_argument("--initializer")
    arm.add_argument("--seed", type=int, default=104830)
    arm.add_argument("--report", required=True)
    arm.add_argument("--allow-non-h100", action="store_true")
    arm.set_defaults(function=_run_arm)

    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--deadline-minutes", type=float)
    run.add_argument("--resume", action="store_true")
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
    smoke.add_argument("--normal-initializer")
    smoke.add_argument("--rb-initializer")
    smoke.add_argument("--timeout-minutes", type=float, default=20.0)
    smoke.add_argument("--fail-fast", action="store_true")
    smoke.set_defaults(function=_local_smoke)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--report", required=True)
    preflight.set_defaults(function=_preflight)
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
