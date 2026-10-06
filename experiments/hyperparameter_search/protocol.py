"""Pure orchestration helpers for the registered campaign protocol.

This module deliberately has no CUDA or model imports.  Workers use it to
resolve an immutable training mapping, maintain the campaign's global exposure
clock, and write small crash-safe JSON manifests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROFILE_KIND = "nlhe_hyperparameter_campaign_v1"
PROFILE_KINDS = {PROFILE_KIND, "nlhe_training_profile_v1"}
TRAINING_KIND = "nlhe_run_v1"
RUN_ITERATIONS = 100_000_000
NATIVE_REFERENCE_ITERATIONS = 1_000_000_000


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using the campaign canonical encoding."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | os.PathLike[str], value: object) -> None:
    """Write JSON atomically, including a best-effort directory sync."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # Directory fsync is not available on every supported filesystem.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _set_override(profile: dict[str, Any], path: str, value: object) -> None:
    pieces = tuple(path.split("."))
    if not pieces or any(not piece for piece in pieces):
        raise ValueError(f"unknown override path: {path!r}")
    if pieces[0] in ("scientific", "run"):
        root = profile.get("training")
        tail = pieces
    elif pieces[0] == "control":
        root = profile.get("control")
        tail = pieces[1:]
    else:
        raise ValueError(f"unknown override path: {path}")
    if not isinstance(root, dict):
        raise TypeError(f"unknown override path: {path}")
    if not tail:
        raise ValueError(f"override path names a container: {path}")
    current: dict[str, Any] = root
    for piece in tail[:-1]:
        child = current.get(piece)
        if not isinstance(child, dict):
            raise TypeError(f"unknown override path: {path}")
        current = child
    leaf = tail[-1]
    if leaf not in current:
        raise ValueError(f"unknown override path: {path}")
    # A null reference_kl means that the entire regularizer is absent.  Other
    # nullable controls (notably interval_decisions) retain their field.
    if value is None and path == "scientific.objective.reference_kl":
        del current[leaf]
    else:
        current[leaf] = copy.deepcopy(value)


def apply_overrides(
    profile: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Deep-copy ``profile`` and apply validated dotted campaign overrides."""

    result = _copy_mapping(profile)
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")
    for path, value in overrides.items():
        if not isinstance(path, str):
            raise TypeError("override paths must be strings")
        _set_override(result, path, value)
    return result


def _arm_overrides(profile: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arms = profile.get("arms")
    if arms is None and arm == "control":
        return {}
    if not isinstance(arms, list):
        raise TypeError("profile.arms must be a list")
    for candidate in arms:
        if isinstance(candidate, Mapping) and candidate.get("name") == arm:
            raw = candidate.get("overrides", {})
            if not isinstance(raw, Mapping):
                raise ValueError(f"profile.arms[{arm!r}].overrides must be a mapping")
            return dict(raw)
    raise ValueError(f"unknown campaign arm: {arm}")


def _spec_mode(spec: Mapping[str, Any]) -> str:
    mode = spec.get("kind", spec.get("mode", spec.get("training_regime", "self_play")))
    if isinstance(mode, Mapping):
        mode = mode.get("training_regime")
    if mode not in ("self_play", "best_response"):
        raise ValueError("spec mode must be self_play or best_response")
    return str(mode)


def _control(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = profile.get("control")
    if not isinstance(value, Mapping):
        raise TypeError("profile.control must be a mapping")
    return copy.deepcopy(dict(value))


def validate_control(control: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy the closed v1 campaign control mapping."""

    value = _mapping(control, "control")
    required = {
        "clock",
        "gradient_reduction",
        "entropy",
        "reference",
        "checkpoint_seconds",
        "snapshot_seconds",
        "initial_loss_scale",
        "thermal_policy",
        "max_nonfinite_steps",
    }
    optional = {"posterior_clock", "entropy_clock", "reference_clock"}
    if set(value) - required - optional or not required <= set(value):
        raise ValueError("control has unknown or missing v1 fields")
    if value["clock"] not in ("global_player_decisions", "posterior_effective_hands"):
        raise ValueError("control.clock has an unsupported exposure clock")
    for schedule in ("entropy", "reference"):
        _clock_name(value, schedule)
    if value["gradient_reduction"] != "global_mean":
        raise ValueError("control.gradient_reduction must be global_mean")
    if value["thermal_policy"] != "telemetry_only":
        raise ValueError("control.thermal_policy must be telemetry_only")
    if "posterior_clock" in value:
        posterior = _mapping(value["posterior_clock"], "control.posterior_clock")
        if set(posterior) != {"exponent"}:
            raise ValueError("control.posterior_clock has unknown or missing fields")
        exponent = posterior["exponent"]
        if (
            isinstance(exponent, bool)
            or not isinstance(exponent, (int, float))
            or not math.isfinite(float(exponent))
            or float(exponent) not in (1.0, 0.5)
        ):
            raise ValueError("control.posterior_clock.exponent must be 1 or0.5")
    entropy = _mapping(value["entropy"], "control.entropy")
    if set(entropy) != {"initial_coefficient", "zero_after_decisions"}:
        raise ValueError("control.entropy has unknown or missing fields")
    coefficient = entropy["initial_coefficient"]
    horizon = entropy["zero_after_decisions"]
    if (
        isinstance(coefficient, bool)
        or not isinstance(coefficient, (int, float))
        or not math.isfinite(float(coefficient))
        or float(coefficient) < 0
    ):
        raise ValueError(
            "control.entropy.initial_coefficient must be finite and nonnegative"
        )
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not math.isfinite(float(horizon))
        or float(horizon) <= 0
    ):
        raise ValueError(
            "control.entropy.zero_after_decisions must be finite and positive"
        )
    reference = _mapping(value["reference"], "control.reference")
    if set(reference) != {"interval_decisions"}:
        raise ValueError("control.reference has unknown or missing fields")
    interval = reference["interval_decisions"]
    if interval is not None and (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0
    ):
        raise ValueError(
            "control.reference.interval_decisions must be finite and positive or null"
        )
    for name in ("checkpoint_seconds", "snapshot_seconds"):
        seconds = value[name]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0
        ):
            raise ValueError(f"control.{name} must be finite and positive")
    loss_scale = value["initial_loss_scale"]
    if (
        isinstance(loss_scale, bool)
        or not isinstance(loss_scale, (int, float))
        or not math.isfinite(float(loss_scale))
        or float(loss_scale) <= 0
    ):
        raise ValueError("control.initial_loss_scale must be finite and positive")
    nonfinite = value["max_nonfinite_steps"]
    if isinstance(nonfinite, bool) or not isinstance(nonfinite, int) or nonfinite < 0:
        raise ValueError("control.max_nonfinite_steps must be nonnegative")
    return copy.deepcopy(value)


def _training(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = profile.get("training")
    if not isinstance(value, Mapping) or value.get("kind") != TRAINING_KIND:
        raise ValueError("profile.training must be a complete nlhe_run_v1 mapping")
    return copy.deepcopy(dict(value))


def resolve_spec(
    spec: Mapping[str, Any], rank: int, world: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one campaign spec into an ordinary config and worker controls."""

    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a mapping")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if isinstance(world, bool) or not isinstance(world, int) or world <= 0:
        raise ValueError("world must be a positive integer")
    if rank >= world:
        raise ValueError("rank must be less than world")
    profile = spec.get("profile")
    if not isinstance(profile, Mapping) or profile.get("kind") not in PROFILE_KINDS:
        raise ValueError("spec.profile must be a complete campaign profile")
    if "training" not in profile or "control" not in profile:
        raise ValueError("spec.profile is incomplete")
    mode = _spec_mode(spec)

    if mode == "best_response":
        if spec.get("overrides", {}) != {}:
            raise ValueError("best response cannot use candidate overrides")
        response = profile.get("best_response")
        if not isinstance(response, Mapping):
            raise ValueError("profile.best_response must be a mapping")
        overrides = response.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("profile.best_response.overrides must be a mapping")
        resolved_profile = apply_overrides(profile, overrides)
    else:
        if "overrides" in spec:
            overrides = spec["overrides"]
            if not isinstance(overrides, Mapping):
                raise ValueError("spec.overrides must be a mapping")
        elif spec.get("arm") is not None:
            overrides = _arm_overrides(profile, str(spec["arm"]))
        else:
            overrides = {}
        resolved_profile = apply_overrides(profile, overrides)

    config = _training(resolved_profile)
    control = validate_control(_control(resolved_profile))
    scientific = config["scientific"]
    run = config["run"]
    if not isinstance(scientific, dict) or not isinstance(run, dict):
        raise TypeError("profile.training is incomplete")
    mode_mapping = scientific.get("mode")
    if not isinstance(mode_mapping, dict):
        raise TypeError("training.scientific.mode is incomplete")
    mode_mapping["training_regime"] = mode
    estimator = spec.get("private_estimator")
    if mode == "best_response":
        estimator = "rao_blackwell"
    elif estimator is not None and estimator not in ("sampled", "rao_blackwell"):
        raise ValueError("spec.private_estimator must be sampled or rao_blackwell")
    if estimator is not None:
        mode_mapping["private_estimator"] = estimator

    if mode == "best_response":
        initialization = spec.get("initialization")
        if initialization not in ("target", "random"):
            raise ValueError("best response initialization must be target or random")
        checkpoint = spec.get("target_checkpoint")
        if not isinstance(checkpoint, (str, os.PathLike)) or not str(checkpoint):
            raise ValueError("best response requires target_checkpoint")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        scientific["response_initialization"] = initialization
        run["frozen"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint_path),
        }
        # The parent worker adds probe.initial_model_sha256 after it has
        # materialized the response initializer.  Keep this resolver probe-free.
    else:
        scientific.pop("response_initialization", None)
        run.pop("frozen", None)

    if spec.get("seed") is not None:
        seed = spec["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("spec.seed must be an integer")
        run["seed"] = seed
    if spec.get("output_dir") is not None:
        output = spec["output_dir"]
        if not isinstance(output, (str, os.PathLike)) or not str(output):
            raise ValueError("spec.output_dir must be a nonempty path")
        run["output_dir"] = str(output)

    execution = run.get("execution")
    if not isinstance(execution, dict):
        raise TypeError("training.run.execution is incomplete")
    execution["device"] = "cuda" if world > 1 else f"cuda:{rank}"
    if world > 1:
        execution["distributed"] = True
    else:
        execution.pop("distributed", None)
    run["iterations"] = RUN_ITERATIONS
    run["checkpoint_every"] = 20

    # Native iteration clocks are intentionally inert.  The worker owns the
    # global decision clock and copies references at its checkpoint boundaries.
    entropy = scientific.get("objective", {}).get("entropy")
    if not isinstance(entropy, dict):
        raise TypeError("training.scientific.objective.entropy is incomplete")
    entropy.clear()
    entropy.update(
        {"kind": "constant", "coefficient": control["entropy"]["initial_coefficient"]}
    )
    objective = scientific.get("objective")
    if not isinstance(objective, dict):
        raise TypeError("training.scientific.objective is incomplete")
    reference = objective.get("reference_kl")
    if reference is None:
        execution["reference_cache"] = "off"
    elif isinstance(reference, dict):
        reference["refresh"] = {"iterations": NATIVE_REFERENCE_ITERATIONS}
    else:
        raise ValueError("training.scientific.objective.reference_kl is invalid")

    if scientific["mode"]["private_estimator"] == "sampled":
        scientific.pop("completion", None)

    if spec.get("smoke", False):
        workload = run.get("workload")
        schedule = scientific.get("schedule")
        microbatch = execution.get("microbatch")
        if not isinstance(workload, dict) or not isinstance(schedule, dict):
            raise ValueError("training workload/schedule is incomplete")
        if not isinstance(microbatch, dict):
            raise ValueError("training execution.microbatch is incomplete")
        workload["roots"] = 64
        recovery = run.setdefault("recovery", {})
        if not isinstance(recovery, dict):
            raise ValueError("training.run.recovery is invalid")
        recovery["minimum_roots"] = 64
        schedule.update({"batch_size": 16, "steps": 2})
        if scientific["mode"]["private_estimator"] == "rao_blackwell":
            completion = scientific.get("completion")
            if not isinstance(completion, dict):
                raise ValueError("RB smoke requires completion settings")
            completion["samples"] = 32
        microbatch.update(
            {"rollout_rows": 16, "completion_rows": 32, "learner_rows": 8}
        )
        control["entropy"]["zero_after_decisions"] = 128
        if control["reference"]["interval_decisions"] is not None:
            control["reference"]["interval_decisions"] = 128
        control["checkpoint_seconds"] = 1
        control["snapshot_seconds"] = 1

    return config, validate_control(control)


def _entropy_settings(control: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate: object = control.get("entropy") if isinstance(control, Mapping) else None
    if candidate is None and isinstance(control, Mapping):
        candidate = control
    if not isinstance(candidate, Mapping):
        raise TypeError("entropy control is missing")
    return candidate


def _clock_name(control: Mapping[str, Any], schedule: str | None = None) -> str:
    name = control.get(
        f"{schedule}_clock", control.get("clock", "global_player_decisions")
    )
    if name not in ("global_player_decisions", "posterior_effective_hands"):
        raise ValueError("control.clock has an unsupported exposure clock")
    return str(name)


def _posterior_exponent(control: Mapping[str, Any]) -> float:
    value = control.get("posterior_clock", {})
    if not isinstance(value, Mapping):
        raise TypeError("control.posterior_clock is invalid")
    exponent = value.get("exponent", 1.0)
    if (
        isinstance(exponent, bool)
        or not isinstance(exponent, (int, float))
        or not math.isfinite(float(exponent))
        or float(exponent) <= 0
    ):
        raise ValueError("control.posterior_clock.exponent must be finite and positive")
    return float(exponent)


def entropy_coefficient(control: Mapping[str, Any], decisions: float = 0) -> float:
    """Return the exact-zero linear entropy coefficient at selected exposure."""

    if (
        isinstance(decisions, bool)
        or not isinstance(decisions, (int, float))
        or not math.isfinite(float(decisions))
        or float(decisions) < 0
    ):
        raise ValueError("exposure must be finite and nonnegative")
    entropy = _entropy_settings(control)
    initial = entropy.get("initial_coefficient")
    horizon = entropy.get("zero_after_decisions")
    if (
        isinstance(initial, bool)
        or not isinstance(initial, (int, float))
        or not math.isfinite(float(initial))
        or float(initial) < 0
        or isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not math.isfinite(float(horizon))
        or float(horizon) <= 0
    ):
        raise ValueError("invalid entropy control")
    return float(initial) * max(0.0, 1.0 - float(decisions) / float(horizon))


@dataclass
class ExposureClock:
    """Checkpointable global counters shared by all campaign workers."""

    global_decisions: int = 0
    global_presentations: int = 0
    optimizer_steps: int = 0
    global_effective_hands: float = 0.0
    next_reference_decisions: float | None = None
    refresh_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "global_decisions",
            "global_presentations",
            "optimizer_steps",
            "refresh_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            isinstance(self.global_effective_hands, bool)
            or not isinstance(self.global_effective_hands, (int, float))
            or not math.isfinite(float(self.global_effective_hands))
            or float(self.global_effective_hands) < 0
        ):
            raise ValueError("global_effective_hands must be finite and nonnegative")
        if self.next_reference_decisions is not None and (
            isinstance(self.next_reference_decisions, bool)
            or not isinstance(self.next_reference_decisions, (int, float))
            or not math.isfinite(float(self.next_reference_decisions))
            or float(self.next_reference_decisions) <= 0
        ):
            raise ValueError(
                "next_reference_decisions must be finite and positive or null"
            )

    def schedule_exposure(
        self, control: Mapping[str, Any], schedule: str | None = None
    ) -> float:
        if _clock_name(control, schedule) == "global_player_decisions":
            return float(self.global_decisions)
        return float(self.global_effective_hands)

    def entropy_coefficient(self, control: Mapping[str, Any]) -> float:
        return entropy_coefficient(control, self.schedule_exposure(control, "entropy"))

    def observe(
        self,
        decisions: int,
        presentations: int,
        steps: int,
        reference_interval: float | None,
        effective_hands: float | None = None,
        schedule_clock: str = "global_player_decisions",
    ) -> bool:
        """Add one completed boundary and report whether reference copies are due."""

        for name, value in (
            ("decisions", decisions),
            ("presentations", presentations),
            ("steps", steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if effective_hands is None:
            effective_hands = float(decisions)
        if (
            isinstance(effective_hands, bool)
            or not isinstance(effective_hands, (int, float))
            or not math.isfinite(float(effective_hands))
            or float(effective_hands) < 0
        ):
            raise ValueError("effective_hands must be finite and nonnegative")
        if schedule_clock not in (
            "global_player_decisions",
            "posterior_effective_hands",
        ):
            raise ValueError("unsupported schedule_clock")
        if reference_interval is not None and (
            isinstance(reference_interval, bool)
            or not isinstance(reference_interval, (int, float))
            or not math.isfinite(float(reference_interval))
            or float(reference_interval) <= 0
        ):
            raise ValueError("reference_interval must be finite and positive or null")
        self.global_decisions += decisions
        self.global_presentations += presentations
        self.optimizer_steps += steps
        self.global_effective_hands += float(effective_hands)
        if reference_interval is None:
            self.next_reference_decisions = None
            return False
        if self.next_reference_decisions is None:
            self.next_reference_decisions = float(reference_interval)
        exposure = float(
            self.global_decisions
            if schedule_clock == "global_player_decisions"
            else self.global_effective_hands
        )
        due = False
        while exposure >= float(self.next_reference_decisions):
            self.refresh_count += 1
            due = True
            self.next_reference_decisions = float(
                self.next_reference_decisions
            ) + float(reference_interval)
        return due

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExposureClock:
        if not isinstance(value, Mapping):
            raise TypeError("exposure clock must be a mapping")
        expected = {
            "global_decisions",
            "global_presentations",
            "optimizer_steps",
            "global_effective_hands",
            "next_reference_decisions",
            "refresh_count",
        }
        if set(value) != expected:
            raise ValueError("exposure clock has unknown or missing fields")
        return cls(**dict(value))


__all__ = [
    "NATIVE_REFERENCE_ITERATIONS",
    "PROFILE_KIND",
    "PROFILE_KINDS",
    "RUN_ITERATIONS",
    "ExposureClock",
    "apply_overrides",
    "atomic_json",
    "canonical_sha256",
    "entropy_coefficient",
    "resolve_spec",
    "sha256_file",
    "validate_control",
]
