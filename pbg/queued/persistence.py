"""Strict queued checkpoints and the one-time legacy converter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from pbg.queued.learner import ReferenceState
from pbg.queued.optimization import DenseAdam, LossScaler


CHECKPOINT_SCHEMA = "pbg-queued-checkpoint-v1"
CONVERSION_SCHEMA = "pbg-queued-conversion-v1"
LEGACY_SCHEMA = "pbg-checkpoint-v8"
LEGACY_SCHEMA_V7 = "pbg-checkpoint-v7"


class PersistenceError(RuntimeError):
    """Checkpoint identity, schema, or resume state is invalid."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clone_cpu(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return value


def _atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class RunState:
    run_id: str
    iteration: int = 0
    root_count: int = 1
    retry_epoch: int = 0
    attempt_logical_base: int = 0
    logical_work: dict[str, int] | None = None
    recovered_overflow: bool = False
    overflow_recoveries: list[dict[str, object]] | None = None
    metrics_prefix: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or self.iteration < 0
            or self.root_count <= 0
            or self.retry_epoch < 0
            or self.attempt_logical_base < 0
        ):
            raise ValueError("run identity and iteration are invalid")
        if self.logical_work is None:
            self.logical_work = {
                "trajectories": 0,
                "completion_samples": 0,
                "minibatches": 0,
                "optimizer_steps": 0,
            }
        if self.overflow_recoveries is None:
            self.overflow_recoveries = []
        if self.metrics_prefix is None:
            self.metrics_prefix = []
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.logical_work.values()
        ):
            raise ValueError("logical work counters must be nonnegative integers")
        iterations = [int(item["iteration"]) for item in self.metrics_prefix]
        if iterations and iterations != list(range(iterations[0], iterations[-1] + 1)):
            raise ValueError("metrics prefix must be gap-free")

    def state_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "root_count": self.root_count,
            "retry_epoch": self.retry_epoch,
            "attempt_logical_base": self.attempt_logical_base,
            "logical_work": dict(self.logical_work),
            "recovered_overflow": self.recovered_overflow,
            "overflow_recoveries": list(self.overflow_recoveries),
            "metrics_prefix": list(self.metrics_prefix),
        }


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: DenseAdam,
    scaler: LossScaler,
    reference_model: nn.Module,
    reference_state: ReferenceState,
    run_state: RunState,
    scientific_config: Mapping[str, object],
    execution_config: Mapping[str, object],
    generator: torch.Generator,
) -> str:
    """Atomically persist the exact completed optimizer boundary."""

    state = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "scientific_config": dict(scientific_config),
        "scientific_sha256": _canonical_hash(scientific_config),
        "execution_config": dict(execution_config),
        "model": _clone_cpu(model.state_dict()),
        "optimizer": _clone_cpu(optimizer.state_dict()),
        "scaler": scaler.state_dict(),
        "reference_model": _clone_cpu(reference_model.state_dict()),
        "reference_state": reference_state.state_dict(),
        "run_state": run_state.state_dict(),
        "explicit_rng": generator.get_state().cpu(),
        "ambient_cpu_rng": torch.get_rng_state(),
        "ambient_cuda_rng": tuple(value.cpu() for value in torch.cuda.get_rng_state_all()),
    }
    destination = Path(path)
    _atomic_torch_save(state, destination)
    return sha256_file(destination)


_CHECKPOINT_FIELDS = {
    "checkpoint_schema",
    "scientific_config",
    "scientific_sha256",
    "execution_config",
    "model",
    "optimizer",
    "scaler",
    "reference_model",
    "reference_state",
    "run_state",
    "explicit_rng",
    "ambient_cpu_rng",
    "ambient_cuda_rng",
}


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: DenseAdam,
    reference_model: nn.Module,
    expected_scientific_config: Mapping[str, object],
    generator: torch.Generator,
    restore_ambient_rng: bool = True,
) -> tuple[LossScaler, ReferenceState, RunState, Mapping[str, object]]:
    """Strictly resume queued state; legacy schemas are never loaded here."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != _CHECKPOINT_FIELDS:
        raise PersistenceError("checkpoint has unknown or missing top-level fields")
    if state["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise PersistenceError(
            "runtime accepts only queued checkpoints; use the offline converter"
        )
    expected_hash = _canonical_hash(expected_scientific_config)
    if (
        state["scientific_sha256"] != expected_hash
        or state["scientific_config"] != dict(expected_scientific_config)
    ):
        raise PersistenceError("resume rejected scientific configuration drift")
    model.load_state_dict(state["model"], strict=True)
    reference_model.load_state_dict(state["reference_model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler = LossScaler.from_state_dict(state["scaler"])
    optimizer.scaler = scaler
    reference = ReferenceState.from_state_dict(state["reference_state"])
    run_data = state["run_state"]
    expected_run = {
        "run_id",
        "iteration",
        "root_count",
        "retry_epoch",
        "attempt_logical_base",
        "logical_work",
        "recovered_overflow",
        "overflow_recoveries",
        "metrics_prefix",
    }
    if not isinstance(run_data, dict) or set(run_data) != expected_run:
        raise PersistenceError("run state has unknown or missing fields")
    run = RunState(**run_data)
    generator.set_state(state["explicit_rng"])
    if restore_ambient_rng:
        torch.set_rng_state(state["ambient_cpu_rng"])
        cuda_states = state["ambient_cuda_rng"]
        if len(cuda_states) != torch.cuda.device_count():
            raise PersistenceError("checkpoint CUDA RNG rank/device count differs")
        torch.cuda.set_rng_state_all(list(cuda_states))
    return scaler, reference, run, state["execution_config"]


_LEGACY_FIELDS_V7 = {
    "adaptive_work_state",
    "branching_controller_schema_version",
    "branching_controller_state",
    "checkpoint_schema_version",
    "config",
    "controller_schema_version",
    "hardware_profile",
    "initialization",
    "iteration",
    "logical_work",
    "model",
    "numpy_generator",
    "optimizer",
    "probe_target",
    "rank_rng_states",
    "reference_effective_hands_rollout",
    "reference_iteration",
    "reference_model",
    "run_id",
    "scaler",
    "storage_schema_version",
    "torch_cpu_rng",
    "torch_cuda_rng",
    "torch_generator",
}
_LEGACY_FIELDS_V8 = _LEGACY_FIELDS_V7 | {
    "cost_safeguard_controller_schema_version",
    "cost_safeguard_controller_state",
}


def _convert_optimizer(
    legacy: Mapping[str, object], training: Mapping[str, object]
) -> dict[str, object]:
    if set(legacy) != {"state", "param_groups"}:
        raise PersistenceError("legacy optimizer has unknown fields")
    groups = legacy["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise PersistenceError("Phase 5 converter requires one optimizer group")
    group = groups[0]
    parameter_ids = group["params"]
    state = legacy["state"]
    if set(state) != set(parameter_ids):
        raise PersistenceError("legacy optimizer parameter state is incomplete")
    exp_avg = torch.cat(tuple(state[key]["exp_avg"].reshape(-1) for key in parameter_ids))
    exp_avg_sq = torch.cat(
        tuple(state[key]["exp_avg_sq"].reshape(-1) for key in parameter_ids)
    )
    steps = {
        int(float(state[key]["step"]))
        for key in parameter_ids
    }
    if len(steps) != 1:
        raise PersistenceError("legacy optimizer parameters have divergent steps")
    betas = tuple(training["adam_betas"])
    optimizer_name = str(training["optimizer"])
    config = {
        "name": optimizer_name,
        "learning_rate": float(group["lr"]),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "epsilon": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "grad_clip": float(training["grad_clip"]),
    }
    return {
        "schema": "queued-dense-adam-v1",
        "config": config,
        "step": steps.pop(),
        "exp_avg": exp_avg.float(),
        "exp_avg_sq": exp_avg_sq.float(),
    }


def _convert_scaler(legacy: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(legacy) != expected:
        raise PersistenceError("legacy scaler has unknown or missing fields")
    scale = legacy["scale"]
    tracker = legacy["_growth_tracker"]
    return {
        "enabled": True,
        "scale": float(scale),
        "growth_factor": float(legacy["growth_factor"]),
        "backoff_factor": float(legacy["backoff_factor"]),
        "growth_interval": int(legacy["growth_interval"]),
        "growth_tracker": int(tracker),
    }


def convert_legacy_sequential_sampled(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Convert one selected v8 normal checkpoint exactly once, offline."""

    source = Path(source_path)
    destination = Path(destination_path)
    if destination.exists():
        raise PersistenceError("conversion destination already exists")
    source_hash = sha256_file(source)
    legacy = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(legacy, dict):
        raise PersistenceError("legacy checkpoint must be a mapping")
    source_schema = legacy.get("checkpoint_schema_version")
    expected_fields = {
        LEGACY_SCHEMA_V7: _LEGACY_FIELDS_V7,
        LEGACY_SCHEMA: _LEGACY_FIELDS_V8,
    }.get(source_schema)
    if expected_fields is None or set(legacy) != expected_fields:
        raise PersistenceError(
            "legacy checkpoint schema has unknown or missing fields"
        )
    if source_schema == LEGACY_SCHEMA and (
        legacy["cost_safeguard_controller_schema_version"] is not None
        or legacy["cost_safeguard_controller_state"] is not None
    ):
        raise PersistenceError(
            "sequential sampled artifact cannot carry a cost safeguard"
        )
    config = legacy["config"]
    if (
        config["training"]["mode"] != "nor" + "mal"
        or config["rollout"]["branching_mode"] is not None
        or config["completion"]["n_samples"] is not None
    ):
        raise PersistenceError(
            "Phase 5 converter accepts only sequential sampled artifacts"
        )
    scientific = {
        "mode": {
            "training_regime": "self_play",
            "rollout_mode": "sequential",
            "private_estimator": "sampled",
        },
        "game": dict(config["game"]),
        "model": {
            key: value
            for key, value in config["model"].items()
            if key not in ("k_act", "use_branching_heads", "max_history_tokens")
        },
        "loss": dict(config["loss"]),
        "training": {
            key: value
            for key, value in config["training"].items()
            if key not in ("out_dir",)
        },
        "root_count": int(
            legacy["adaptive_work_state"]["current_roots_per_gpu"]
        ),
        "root_count_minimum": int(
            legacy["adaptive_work_state"]["minimum_roots_per_gpu"]
        ),
    }
    execution = {
        "hardware_profile": dict(legacy["hardware_profile"]),
        "checkpoint_cadence": config["training"].get("checkpoint_every"),
    }
    training = config["training"]
    scaler = (
        _convert_scaler(legacy["scaler"])
        if bool(training["amp"])
        else LossScaler(enabled=False).state_dict()
    )
    run_state = RunState(
        run_id=legacy["run_id"],
        iteration=int(legacy["iteration"]),
        root_count=int(legacy["adaptive_work_state"]["current_roots_per_gpu"]),
        logical_work=dict(legacy["logical_work"]),
    )
    converted = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "scientific_config": scientific,
        "scientific_sha256": _canonical_hash(scientific),
        "execution_config": execution,
        "model": _clone_cpu(legacy["model"]),
        "optimizer": _convert_optimizer(legacy["optimizer"], training),
        "scaler": scaler,
        "reference_model": _clone_cpu(legacy["reference_model"]),
        "reference_state": {
            "reference_iteration": int(legacy["reference_iteration"]),
            "effective_hands_rollout": float(
                legacy["reference_effective_hands_rollout"]
            ),
            "interval_iterations": int(training["kl_ref_interval"]),
            "interval_effective_hands": None,
        },
        "run_state": run_state.state_dict(),
        "explicit_rng": legacy["torch_generator"].clone(),
        "ambient_cpu_rng": legacy["torch_cpu_rng"].clone(),
        "ambient_cuda_rng": tuple(value.clone() for value in legacy["torch_cuda_rng"]),
        "conversion": {
            "schema": CONVERSION_SCHEMA,
            "source_sha256": source_hash,
            "source_schema": source_schema,
            "mode": "self_play/sequential/sampled",
        },
    }
    # Conversion provenance is recorded beside the strict runtime payload,
    # never accepted as a runtime field.
    runtime = dict(converted)
    provenance = runtime.pop("conversion")
    _atomic_torch_save(runtime, destination)
    result = {
        **provenance,
        "result_sha256": sha256_file(destination),
        "destination": str(destination),
    }
    metadata = destination.with_suffix(destination.suffix + ".conversion.json")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata.name}.", suffix=".tmp", dir=metadata.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, metadata)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return result


def trim_metrics_after_checkpoint(
    metrics: Sequence[Mapping[str, object]], iteration: int
) -> list[dict[str, object]]:
    """Back up externally if desired, then retain the exact completed prefix."""

    retained = [
        dict(item) for item in metrics if int(item["iteration"]) <= iteration
    ]
    values = [int(item["iteration"]) for item in retained]
    if values and values != list(range(values[0], values[-1] + 1)):
        raise PersistenceError("metrics before the checkpoint are not gap-free")
    return retained


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONVERSION_SCHEMA",
    "LEGACY_SCHEMA",
    "LEGACY_SCHEMA_V7",
    "PersistenceError",
    "RunState",
    "convert_legacy_sequential_sampled",
    "load_checkpoint",
    "save_checkpoint",
    "sha256_file",
    "trim_metrics_after_checkpoint",
]
