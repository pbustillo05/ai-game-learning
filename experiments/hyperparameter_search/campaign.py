"""Resumable, GPU-aware orchestration of a bounded hyperparameter campaign.

Only workers own training state and accumulated training time. This launcher
never infers completion from process exit alone and never deletes partial runs.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/pablob/.venvs/poker/bin/python")
CUDA_HOME = Path("/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13")
SCHEMA = "nlhe_hyperparameter_campaign_v1"
GAIN = "mean_candidate_minus_baseline_bb_per_hand"
CI = "confidence_interval_candidate_minus_baseline_bb_per_hand"
ESTIMATORS = ("sampled", "rao_blackwell")
SHUTDOWN_SECONDS = 300


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def source_digest() -> str:
    """Hash live production and experiment source, including untracked modules."""
    sha = hashlib.sha256()
    for directory in (ROOT / "pbg", ROOT / "experiments"):
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h")
            ):
                relative = str(path.relative_to(ROOT)).encode()
                sha.update(len(relative).to_bytes(8, "big"))
                sha.update(relative)
                content = path.read_bytes()
                sha.update(len(content).to_bytes(8, "big"))
                sha.update(content)
    return sha.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        if path.read_bytes() != canonical(value):
            raise ValueError(f"Immutable campaign artifact differs: {path}")
    else:
        atomic_json(path, value)


@contextmanager
def campaign_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Campaign already running: {output}") from exc
        yield


def load_profile(path: Path) -> dict:
    profile = yaml.safe_load(path.read_text())
    if not isinstance(profile, dict) or profile.get("kind") != SCHEMA:
        raise ValueError(f"Expected profile kind {SCHEMA}")
    arms = profile.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("Profile needs nonempty arms")
    names = [arm.get("name") for arm in arms]
    if any(
        not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name)
        for name in names
    ):
        raise ValueError(
            "Arm names must contain lowercase letters, digits, underscores"
        )
    if len(set(names)) != len(names) or "control" not in names:
        raise ValueError("Arm names must be unique and include control")
    if any(not isinstance(arm.get("overrides", {}), dict) for arm in arms):
        raise ValueError("Arm overrides must be mappings")
    if any(
        "scientific.mode.private_estimator" in arm.get("overrides", {}) for arm in arms
    ):
        raise ValueError("Estimator is a paired condition, never a selection arm")
    settings = profile["campaign"]
    for stage in ("screen", "confirm", "cold"):
        for key in (f"{stage}_seconds", f"{stage}_br_seconds"):
            if (
                not isinstance(settings[key], (int, float))
                or not math.isfinite(settings[key])
                or settings[key] <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        if (
            not isinstance(settings[f"{stage}_hands"], int)
            or settings[f"{stage}_hands"] <= 0
        ):
            raise ValueError(f"{stage}_hands must be positive")
    if len(settings["br_seeds"]) != 3 or len(set(settings["br_seeds"])) != 3:
        raise ValueError("Exactly three distinct BR seeds are required")
    if len(settings["confirmation_seeds"]) < 2 or len(
        set(settings["confirmation_seeds"])
    ) != len(settings["confirmation_seeds"]):
        raise ValueError("At least two distinct confirmation seeds are required")
    if settings["screen_seed"] in settings["confirmation_seeds"]:
        raise ValueError("Confirmation must use new seeds")
    if not 1 <= settings["shortlist"] <= len(arms):
        raise ValueError("shortlist must fit the arm list")
    if not 1 <= settings["cold_finalists"] <= settings["shortlist"]:
        raise ValueError("cold_finalists must fit the shortlist")
    if not isinstance(profile.get("training"), dict):
        raise TypeError("Profile needs a complete training config")
    canonical(profile)
    return profile


def child_environment(gpus: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    venv = str(PYTHON.parent.parent)
    env.update(
        VIRTUAL_ENV=venv,
        CUDA_HOME=str(CUDA_HOME),
        CUDA_VISIBLE_DEVICES=",".join(gpus),
        OMP_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )
    env["PATH"] = f"{PYTHON.parent}:{CUDA_HOME / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{CUDA_HOME / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    # A launcher invoked from another torchrun must not inherit its rendezvous.
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        env.pop(key, None)
    return env


def _smi(query: str, domain: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{domain}={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        [part.strip() for part in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def resolve_gpus(requested: list[str]) -> list[str]:
    inventory = _smi("index,uuid", "gpu")
    lookup = {key: uuid for index, uuid in inventory for key in (index, uuid)}
    try:
        selected = [lookup[item] for item in requested]
    except KeyError as exc:
        raise RuntimeError(f"Requested GPU is not available: {exc.args[0]}") from exc
    if len(set(selected)) != len(selected):
        raise ValueError("GPU list contains duplicate devices")
    return selected


def check_idle(gpus: list[str]) -> None:
    occupied = [
        row
        for row in _smi("gpu_uuid,pid,process_name", "compute-apps")
        if row[0] in gpus
    ]
    if occupied:
        details = "; ".join(", ".join(row) for row in occupied)
        raise RuntimeError(
            f"Selected GPU has a compute process; refusing launch: {details}"
        )


def budget_plan(profile: dict, gpu_count: int, scale: float, smoke: bool) -> dict:
    settings = profile["campaign"]
    counts = {
        "screen": 1 if smoke else len(profile["arms"]),
        "confirm": 0
        if smoke
        else settings["shortlist"] * len(settings["confirmation_seeds"]),
        "cold": 0 if smoke else settings["cold_finalists"],
    }
    stages = {}
    for stage, count in counts.items():
        train = settings[f"{stage}_seconds"] * scale
        br = settings[f"{stage}_br_seconds"] * scale
        stages[stage] = {
            "paired_settings": count,
            "trials": 2 * count,
            "estimators": list(ESTIMATORS),
            "self_play_seconds_per_trial": train,
            "br_seconds_per_restart": br,
            "br_restarts": 3,
            "br_waves_per_trial": math.ceil(3 / gpu_count),
            "nominal_training_wall_seconds": 2
            * count
            * (train + math.ceil(3 / gpu_count) * br),
            "nominal_gpu_seconds": 2 * count * (gpu_count * train + 3 * br),
        }
    return {
        "stages": stages,
        "nominal_training_wall_seconds": sum(
            s["nominal_training_wall_seconds"] for s in stages.values()
        ),
        "nominal_gpu_seconds": sum(s["nominal_gpu_seconds"] for s in stages.values()),
        "excludes": ["evaluation", "initialization", "interrupted job recovery"],
        "smoke_iteration_limit": 1 if smoke else None,
        "note": "Worker accumulated training seconds govern budgets; smoke ends after one iteration. Wall estimates exclude overhead.",
    }


def set_dotted(mapping: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    target = mapping
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        if not isinstance(target[part], dict):
            raise TypeError(f"Override traverses a nonmapping: {key}")
        target = target[part]
    target[parts[-1]] = copy.deepcopy(value)


def resolved_arm(profile: dict, arm: dict) -> dict:
    resolved = copy.deepcopy(profile)
    for key, value in arm.get("overrides", {}).items():
        if key.startswith("control."):
            set_dotted(
                resolved.setdefault("control", {}), key.removeprefix("control."), value
            )
        elif key.startswith(("scientific.", "run.")):
            set_dotted(resolved["training"], key, value)
        else:
            raise ValueError(f"Unsupported override root: {key}")
    return resolved


def score_evaluations(evaluations: list[dict]) -> dict:
    if len(evaluations) != 3:
        raise ValueError("Scoring requires all three response evaluations")
    if any(value.get("raw_match") is True for value in evaluations):
        raise ValueError("Best-response scoring does not accept raw-match evaluations")
    for value in evaluations:
        validate_evaluation(value)
    means = [float(value[GAIN]) for value in evaluations]
    return {
        "worst_response_gain_bb_per_hand": max(means),
        "worst_response_interval": [
            max(float(value[CI][0]) for value in evaluations),
            max(float(value[CI][1]) for value in evaluations),
        ],
        "restart_mean_gains": means,
        "note": "Maximum observed response gain; not true exploitability. Intervals are descriptive, not multiplicity adjusted.",
    }


def validate_evaluation(value: dict) -> None:
    if (
        value.get("status") != "complete"
        or not isinstance(value.get("requested_hands"), int)
        or value["requested_hands"] <= 0
        or value.get("completed_hands") != value["requested_hands"]
    ):
        raise ValueError("Evaluation is not explicitly complete")
    if value.get("raw_match") is True:
        statistics = value.get("candidate_payoff_vs_fixed_opponent")
        if not isinstance(statistics, dict):
            raise ValueError("Raw-match evaluation lacks candidate payoff statistics")
        interval = statistics.get("confidence_interval_bb_per_hand")
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not all(
                isinstance(item, (float, int)) and math.isfinite(item)
                for item in (statistics.get("mean_bb_per_hand"), *interval)
            )
        ):
            raise ValueError("Raw-match evaluation payoff is nonfinite or missing")
        if interval[0] > interval[1]:
            raise ValueError("Raw-match evaluation confidence interval is reversed")
        if value.get("baseline_evaluated") is not False:
            raise ValueError("Raw-match evaluation baseline status is invalid")
        return
    interval = value.get(CI)
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("Evaluation lacks its confidence interval")
    if not all(
        isinstance(x, (float, int)) and math.isfinite(x)
        for x in (value.get(GAIN), *interval)
    ):
        raise ValueError("Evaluation score is nonfinite or missing")
    if interval[0] > interval[1]:
        raise ValueError("Evaluation confidence interval is reversed")


def pool_estimators(scores: dict[str, dict]) -> dict:
    if set(scores) != set(ESTIMATORS):
        raise ValueError("Every candidate must have both estimator conditions")
    return {
        "worst_response_gain_bb_per_hand": sum(
            row["worst_response_gain_bb_per_hand"] for row in scores.values()
        )
        / 2,
        "worst_response_interval": [
            sum(row["worst_response_interval"][i] for row in scores.values()) / 2
            for i in (0, 1)
        ],
        "per_mode": scores,
        "sampled_minus_rb_gain": scores["sampled"]["worst_response_gain_bb_per_hand"]
        - scores["rao_blackwell"]["worst_response_gain_bb_per_hand"],
        "estimator_weights": {name: 0.5 for name in ESTIMATORS},
        "note": "Equal mean of each mode's worst response gain selects SHARED parameters; no estimator is selected. Averaged interval endpoints are descriptive, not a pooled confidence interval.",
    }


def promote(
    scores: dict[str, dict], count: int, *, include_control: bool = False
) -> dict:
    ordered = sorted(
        scores, key=lambda name: (scores[name]["worst_response_gain_bb_per_hand"], name)
    )
    if not ordered or not 1 <= count <= len(ordered):
        raise ValueError("Invalid promotion population/count")
    selected = ordered[:count]
    if include_control and "control" not in selected:
        if "control" not in scores:
            raise ValueError("Control is missing from promotion scores")
        selected = [
            "control",
            *[name for name in ordered if name != "control"][: count - 1],
        ]
    leader_interval = scores[ordered[0]]["worst_response_interval"]
    uncertain = [
        name
        for name in ordered[1:]
        if scores[name]["worst_response_interval"][0] <= leader_interval[1]
    ]
    return {
        "provisional": True,
        "selected": selected,
        "ordered": ordered,
        "scores": scores,
        "uncertainty_overlap_with_leader": uncertain,
        "confidence_status": "unresolved_requires_heldout_validation",
        "criterion": "Lowest equally pooled sampled/RB worst response gain selects shared parameters; uncertainty does not establish superiority. No estimator selection.",
    }


def combination_candidate(arms: dict[str, dict], ordered: list[str]) -> dict | None:
    """Pre-registered interaction check: strongest two nonconflicting changes."""
    candidates = [name for name in ordered if name != "control"]
    if len(candidates) < 2:
        return None
    first = candidates[0]
    left = arms[first].get("overrides", {})
    for second in candidates[1:]:
        right = arms[second].get("overrides", {})
        conflicting = any(
            a == b or a.startswith(b + ".") or b.startswith(a + ".")
            for a in left
            for b in right
        )
        if not conflicting:
            return {
                "name": "combined_screen_winners",
                "overrides": {**left, **right},
                "components": [first, second],
            }
    return None


def aggregate_confirmation(
    trials: dict[str, dict[int, dict]], seeds: list[int]
) -> dict[str, dict]:
    result = {}
    for arm, by_seed in trials.items():
        if set(by_seed) != set(seeds):
            raise ValueError("Confirmation scores must contain identical paired seeds")
        rows = [by_seed[seed] for seed in seeds]
        result[arm] = {
            "worst_response_gain_bb_per_hand": sum(
                row["worst_response_gain_bb_per_hand"] for row in rows
            )
            / len(rows),
            "worst_response_interval": [
                sum(row["worst_response_interval"][i] for row in rows) / len(rows)
                for i in (0, 1)
            ],
            "seeds": seeds,
            "per_seed": {str(seed): by_seed[seed] for seed in seeds},
            "interval_note": "Average of per-seed interval endpoints, a descriptive uncertainty envelope, not a new confidence interval.",
        }
        if "control" in trials:
            result[arm]["paired_gain_difference_vs_control"] = {
                str(seed): by_seed[seed]["worst_response_gain_bb_per_hand"]
                - trials["control"][seed]["worst_response_gain_bb_per_hand"]
                for seed in seeds
            }
        if all("per_mode" in row for row in rows):
            result[arm]["per_mode"] = {
                mode: {
                    "worst_response_gain_bb_per_hand": sum(
                        row["per_mode"][mode]["worst_response_gain_bb_per_hand"]
                        for row in rows
                    )
                    / len(rows),
                    "worst_response_interval": [
                        sum(
                            row["per_mode"][mode]["worst_response_interval"][i]
                            for row in rows
                        )
                        / len(rows)
                        for i in (0, 1)
                    ],
                }
                for mode in ESTIMATORS
            }
            result[arm]["sampled_minus_rb_gain"] = sum(
                row["sampled_minus_rb_gain"] for row in rows
            ) / len(rows)
    return result


@dataclass
class Job:
    name: str
    directory: Path
    command: list[str]
    gpus: list[str]
    result_path: Path
    spec_hash: str
    evaluation: bool = False


class Campaign:
    def __init__(
        self,
        profile: dict,
        output: Path,
        requested_gpus: list[str],
        *,
        scale: float = 1.0,
        smoke: bool = False,
    ):
        self.profile, self.output = profile, output.resolve()
        self.requested_gpus, self.scale, self.smoke = requested_gpus, scale, smoke
        self.gpus: list[str] = []
        self.children: dict[int, subprocess.Popen] = {}
        self.interrupted = False
        self.arms = {arm["name"]: arm for arm in profile["arms"]}
        self._old_handlers: dict[int, Any] = {}
        self.expected_source_digest: str | None = None

    def prepare(self, *, resume: bool) -> None:
        manifest = {
            "kind": SCHEMA,
            "profile": self.profile,
            "profile_sha256": digest(self.profile),
            "requested_gpus": self.requested_gpus,
            "duration_scale": self.scale,
            "smoke": self.smoke,
            "launcher_schema": 2,
            "paired_estimators": list(ESTIMATORS),
            "source_sha256": source_digest(),
        }
        path = self.output / "manifest.json"
        if path.exists() and not resume:
            raise ValueError(
                "Output already has a manifest; use --resume to verify and continue"
            )
        immutable_json(path, manifest)
        self.expected_source_digest = manifest["source_sha256"]
        immutable_json(
            self.output / "budget_plan.json",
            budget_plan(self.profile, len(self.requested_gpus), self.scale, self.smoke),
        )
        names = ["control"] if self.smoke else list(self.arms)
        for name in names:
            for mode in ESTIMATORS:
                self.worker_spec(
                    "screen",
                    name,
                    self.profile["campaign"]["screen_seed"],
                    private_estimator=mode,
                )

    def worker_spec(
        self,
        stage: str,
        arm: str,
        seed: int,
        *,
        private_estimator: str,
        restart: int | None = None,
        target: str | None = None,
    ) -> tuple[Path, dict]:
        if private_estimator not in ESTIMATORS:
            raise ValueError("Unknown private estimator condition")
        directory = self.output / stage / arm / f"seed_{seed}" / private_estimator
        if restart is not None:
            directory = directory / f"response_{restart}"
        settings = self.profile["campaign"]
        spec = {
            "kind": "self_play" if restart is None else "best_response",
            "profile": self.profile,
            "arm": arm,
            "overrides": self.arms[arm].get("overrides", {}) if restart is None else {},
            "seed": seed if restart is None else settings["br_seeds"][restart],
            "output_dir": str(directory),
            "budget_seconds": settings[
                f"{stage}{'_br' if restart is not None else ''}_seconds"
            ]
            * self.scale,
            "initialization": ("reference_model" if stage == "cold" else "model")
            if restart is None
            else ("target", "target", "random")[restart],
            "smoke": self.smoke,
            "private_estimator": private_estimator,
        }
        if self.smoke:
            spec["max_iterations"] = 1
        if target is not None:
            spec["target_checkpoint"] = target
        path = directory / "job.json"
        immutable_json(path, spec)
        return path, spec

    def _worker_job(self, spec_path: Path, spec: dict, gpus: list[str]) -> Job:
        command = [str(PYTHON)]
        if spec["kind"] == "self_play":
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={len(gpus)}",
                "-m",
            ]
        else:
            command += ["-m"]
        command += [
            "experiments.hyperparameter_search.worker",
            "--spec",
            str(spec_path),
        ]
        return Job(
            str(spec_path.relative_to(self.output)),
            spec_path.parent,
            command,
            gpus,
            spec_path.parent / "result.json",
            digest(spec),
        )

    def _read_result(self, job: Job) -> dict:
        value = json.loads(job.result_path.read_text())
        if job.evaluation:
            validate_evaluation(value)
        else:
            if value.get("complete") is not True:
                raise ValueError(
                    f"Worker has not reported completion: {job.result_path}"
                )
            elapsed = value.get("training_seconds")
            if (
                not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("Worker training_seconds is invalid")
            for name in ("snapshot", "checkpoint"):
                path = Path(value.get(name, ""))
                if not path.is_absolute() or not path.is_file():
                    raise ValueError(
                        f"Worker {name} is not an existing absolute file: {path}"
                    )
        return value

    def _status(self, job: Job, state: str, **extra: Any) -> None:
        atomic_json(
            job.directory
            / ("evaluation_status.json" if job.evaluation else "status.json"),
            {
                "state": state,
                "job": job.name,
                "spec_sha256": job.spec_hash,
                "command": job.command,
                "gpu_uuids": job.gpus,
                "updated_at_unix": time.time(),
                **extra,
            },
        )

    def _completed(self, job: Job) -> dict | None:
        status_path = job.directory / (
            "evaluation_status.json" if job.evaluation else "status.json"
        )
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("source_invalidated"):
            raise RuntimeError(
                f"Job source changed during execution; refusing its result: {status_path}"
            )
        if status and status.get("spec_sha256") != job.spec_hash:
            raise ValueError(f"Job status has incompatible spec hash: {status_path}")
        if not job.result_path.exists():
            if status.get("state") == "complete":
                raise ValueError(f"Completed job result disappeared: {job.result_path}")
            return None
        try:
            value = self._read_result(job)
        except (ValueError, OSError, json.JSONDecodeError):
            if status.get("state") == "complete":
                raise
            return None
        result_hash = digest(value)
        if (
            status.get("state") == "complete"
            and status.get("result_sha256") != result_hash
        ):
            raise ValueError(f"Completed result changed: {job.result_path}")
        self._status(
            job, "complete", result_sha256=result_hash, recovered_completion=True
        )
        return value

    def _signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = True
        self._terminate_owned(signum)

    def _terminate_owned(self, signum: int = signal.SIGTERM) -> None:
        for process in list(self.children.values()):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

    def run_jobs(self, jobs: list[Job]) -> list[dict]:
        """Run one GPU-disjoint wave, allowing workers to checkpoint on failure."""
        results: dict[str, dict] = {}
        pending = []
        for job in jobs:
            value = self._completed(job)
            if value is None:
                pending.append(job)
            else:
                results[job.name] = value
        occupied = [gpu for job in pending for gpu in job.gpus]
        if len(set(occupied)) != len(occupied):
            raise ValueError("Concurrent jobs overlap GPU assignments")
        if self.interrupted:
            raise InterruptedError("Campaign interrupted")
        if pending:
            if (
                self.expected_source_digest is not None
                and source_digest() != self.expected_source_digest
            ):
                raise RuntimeError(
                    "Production/experiment source changed since campaign preparation; refusing mixed-code run"
                )
            check_idle(occupied)
        running: list[tuple[Job, subprocess.Popen, Any]] = []
        failure: Exception | None = None
        stopping_since: float | None = None
        try:
            for job in pending:
                if self.interrupted:
                    raise InterruptedError("Campaign interrupted before launch")
                job.directory.mkdir(parents=True, exist_ok=True)
                log = (
                    job.directory
                    / ("evaluation_process.log" if job.evaluation else "process.log")
                ).open("ab")
                record = {
                    "job": job.name,
                    "command": job.command,
                    "cwd": str(ROOT),
                    "gpu_uuids": job.gpus,
                    "spec_sha256": job.spec_hash,
                    "time_unix": time.time(),
                }
                with (self.output / "commands.jsonl").open("ab") as stream:
                    stream.write(json.dumps(record, sort_keys=True).encode() + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    process = subprocess.Popen(
                        job.command,
                        cwd=ROOT,
                        env=child_environment(job.gpus),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                self.children[process.pid] = process
                running.append((job, process, log))
                self._status(job, "running", pid=process.pid)
            while running:
                if self.interrupted or failure is not None:
                    stopping_since = (
                        time.monotonic() if stopping_since is None else stopping_since
                    )
                    if time.monotonic() - stopping_since > SHUTDOWN_SECONDS:
                        self._terminate_owned(signal.SIGKILL)
                for item in list(running):
                    job, process, log = item
                    code = process.poll()
                    if code is None:
                        continue
                    running.remove(item)
                    self.children.pop(process.pid, None)
                    log.close()
                    source_changed = (
                        self.expected_source_digest is not None
                        and source_digest() != self.expected_source_digest
                    )
                    try:
                        if source_changed:
                            raise RuntimeError(
                                "Production/experiment source changed while job was running"
                            )
                        if code != 0 or self.interrupted:
                            raise RuntimeError(f"Job exited {code}: {job.name}")
                        value = self._read_result(job)
                        results[job.name] = value
                        self._status(
                            job,
                            "complete",
                            result_sha256=digest(value),
                            returncode=code,
                        )
                    except Exception as exc:  # noqa: BLE001 - persist failure at the process-supervision boundary
                        self._status(
                            job,
                            "interrupted" if self.interrupted else "failed",
                            returncode=code,
                            error=str(exc),
                            source_invalidated=source_changed,
                        )
                        failure = failure or exc
                        self._terminate_owned()
                if running:
                    time.sleep(0.25)
            if self.interrupted:
                raise InterruptedError("Campaign interrupted; rerun with --resume")
            if failure is not None:
                raise failure
            if pending:
                # CUDA process teardown may briefly trail process exit.
                for attempt in range(20):
                    try:
                        check_idle(occupied)
                        break
                    except RuntimeError:
                        if attempt == 19:
                            raise
                        time.sleep(0.5)
        except BaseException:
            self._terminate_owned()
            for job, process, log in running:
                try:
                    process.wait(timeout=SHUTDOWN_SECONDS)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                self.children.pop(process.pid, None)
                log.close()
                self._status(
                    job,
                    "interrupted",
                    returncode=process.returncode,
                    error="Wave aborted; preserve worker checkpoint for resume",
                )
            raise
        return [results[job.name] for job in jobs]

    def trial(self, stage: str, arm: str, seed: int, private_estimator: str) -> dict:
        path, spec = self.worker_spec(
            stage, arm, seed, private_estimator=private_estimator
        )
        trained = self.run_jobs([self._worker_job(path, spec, self.gpus)])[0]
        responses = []
        for first in range(0, 3, len(self.gpus)):
            wave = []
            for restart in range(first, min(3, first + len(self.gpus))):
                br_path, br_spec = self.worker_spec(
                    stage,
                    arm,
                    seed,
                    private_estimator=private_estimator,
                    restart=restart,
                    target=trained["checkpoint"],
                )
                wave.append(
                    self._worker_job(br_path, br_spec, [self.gpus[restart - first]])
                )
            responses.extend(self.run_jobs(wave))
        evaluations = []
        for first in range(0, 3, len(self.gpus)):
            wave = []
            for restart in range(first, min(3, first + len(self.gpus))):
                directory = path.parent / f"evaluation_{restart}"
                result_path = directory / "result.json"
                evaluation_spec = {
                    "candidate": responses[restart]["snapshot"],
                    "opponent": trained["snapshot"],
                    "baseline": trained["snapshot"],
                    "hands": 4
                    if self.smoke
                    else self.profile["campaign"][f"{stage}_hands"],
                    "seed": seed,
                    "split": f"{stage}_heldout",
                    "device": "cuda:0",
                }
                immutable_json(directory / "job.json", evaluation_spec)
                command = [
                    str(PYTHON),
                    "-m",
                    "experiments.hyperparameter_search.evaluation",
                ]
                for key, value in evaluation_spec.items():
                    command += [f"--{key}", str(value)]
                command += ["--output", str(result_path)]
                wave.append(
                    Job(
                        str(directory.relative_to(self.output)),
                        directory,
                        command,
                        [self.gpus[restart - first]],
                        result_path,
                        digest(evaluation_spec),
                        evaluation=True,
                    )
                )
            evaluations.extend(self.run_jobs(wave))
        score = score_evaluations(evaluations)
        atomic_json(path.parent / "score.json", score)
        return score

    def paired_trial(self, stage: str, arm: str, seed: int) -> dict:
        modes = list(ESTIMATORS)
        if (list(self.arms).index(arm) + seed) % 2:
            modes.reverse()
        scores = {mode: self.trial(stage, arm, seed, mode) for mode in modes}
        pooled = pool_estimators(scores)
        atomic_json(
            self.output / stage / arm / f"seed_{seed}" / "paired_score.json", pooled
        )
        return pooled

    def run(self) -> None:
        self.gpus = resolve_gpus(self.requested_gpus)
        immutable_json(
            self.output / "gpu_assignment.json",
            {"requested": self.requested_gpus, "uuids": self.gpus},
        )
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[sig] = signal.signal(sig, self._signal)
        try:
            settings = self.profile["campaign"]
            screen_order = [name for name in self.arms if name != "control"]
            random.Random(settings["screen_seed"]).shuffle(screen_order)
            screen_order = ["control"] if self.smoke else ["control", *screen_order]
            atomic_json(
                self.output / "execution_order.json",
                {
                    "screen_arms": screen_order,
                    "mode_order": "alternating by registered arm index plus seed parity",
                },
            )
            screen = {
                name: self.paired_trial("screen", name, settings["screen_seed"])
                for name in screen_order
            }
            if self.smoke:
                atomic_json(
                    self.output / "report.json",
                    {
                        "complete": True,
                        "smoke": True,
                        "provisional": True,
                        "screen": screen,
                        "recommendation": None,
                    },
                )
                return
            promotion = {
                "screen": promote(screen, settings["shortlist"], include_control=True)
            }
            combined = combination_candidate(self.arms, promotion["screen"]["ordered"])
            if combined is not None and settings["shortlist"] >= 3:
                self.arms[combined["name"]] = combined
                promotion["screen"]["selected"] = [
                    "control",
                    combined["components"][0],
                    combined["name"],
                ]
                promotion["screen"]["interaction_candidate"] = combined
                immutable_json(self.output / "combination_candidate.json", combined)
            atomic_json(self.output / "promotion.json", promotion)
            confirmations = {
                name: {
                    seed: self.paired_trial("confirm", name, seed)
                    for seed in settings["confirmation_seeds"]
                }
                for name in promotion["screen"]["selected"]
            }
            confirmed = aggregate_confirmation(
                confirmations, settings["confirmation_seeds"]
            )
            promotion["confirmation"] = promote(confirmed, settings["cold_finalists"])
            atomic_json(self.output / "promotion.json", promotion)
            cold = {
                name: self.paired_trial("cold", name, settings["cold_seed"])
                for name in promotion["confirmation"]["selected"]
            }
            promotion["cold"] = promote(cold, 1)
            atomic_json(self.output / "promotion.json", promotion)
            from experiments.hyperparameter_search.cross_panel import run_cross_panel

            cross_panel = run_cross_panel(self, cold)
            winner = cross_panel["recommended_arm"]
            promotion["cross_panel"] = cross_panel
            atomic_json(self.output / "promotion.json", promotion)
            selected = resolved_arm(self.profile, self.arms[winner])
            selected["recommendation"] = {
                "provisional": True,
                "shared_arm": winner,
                "paired_estimators": list(ESTIMATORS),
                "estimator_selected": False,
            }
            atomic_json(self.output / "recommended_profile.json", selected)
            mode_profiles = {}
            for mode in ESTIMATORS:
                training = copy.deepcopy(selected["training"])
                set_dotted(training, "scientific.mode.private_estimator", mode)
                if mode == "sampled":
                    training["scientific"].pop("completion", None)
                mode_path = self.output / f"recommended_training_{mode}.json"
                atomic_json(
                    mode_path,
                    {
                        "kind": "nlhe_training_profile_v1",
                        "status": "provisional; requires held-out validation",
                        "initialization": selected["initialization"],
                        "training": training,
                        "control": selected["control"],
                    },
                )
                mode_profiles[mode] = str(mode_path)
            recommendation = {
                "provisional": True,
                "arm": winner,
                "profile": str(self.output / "recommended_profile.json"),
                "training": selected["training"],
                "control": selected.get("control", {}),
                "paired_training_configs": mode_profiles,
                "estimator_selected": False,
                "caveat": "Observed response gains do not establish Nash convergence or true exploitability; review confidence overlap.",
            }
            atomic_json(
                self.output / "report.json",
                {
                    "complete": True,
                    "smoke": False,
                    "promotion": promotion,
                    "recommendation": recommendation,
                },
            )
        finally:
            for sig, previous in self._old_handlers.items():
                signal.signal(sig, previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not 1 <= len(gpus) <= 3 or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain one to three distinct GPU indices or UUIDs")
    if not math.isfinite(args.duration_scale) or args.duration_scale <= 0:
        parser.error("--duration-scale must be finite and positive")
    try:
        profile = load_profile(args.profile)
        with campaign_lock(args.output):
            campaign = Campaign(
                profile, args.output, gpus, scale=args.duration_scale, smoke=args.smoke
            )
            campaign.prepare(resume=args.resume)
            if not args.prepare_only:
                campaign.run()
        return 0
    except InterruptedError as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - persist failure at the process-supervision boundary
        print(f"Campaign stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
