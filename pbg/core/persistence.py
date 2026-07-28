"""Strict checkpoints for the final queued core."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from pbg.core.learner import ReferenceState
from pbg.core.optimization import DenseAdam, LossScaler

CHECKPOINT_SCHEMA = "pbg-queued-checkpoint-v2"
REPLICA_CHECKPOINT_SCHEMA = "pbg-queued-replica-checkpoint-v2"
DISTRIBUTED_CHECKPOINT_SCHEMA = "pbg-queued-distributed-checkpoint-v1"


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


def model_state_sha256(model: nn.Module | Mapping[str, Tensor]) -> str:
    """Hash one model state independent of device and checkpoint container."""

    state = model.state_dict() if isinstance(model, nn.Module) else model
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, Tensor):
            raise TypeError("model state hash requires tensor values")
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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


LOGICAL_WORK_KEYS = (
    "trajectories",
    "completion_samples",
    "minibatches",
    "optimizer_steps",
)


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
    branch_controller_state: dict[str, object] | None = None
    cost_safeguard_state: dict[str, object] | None = None

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
            self.logical_work = {name: 0 for name in LOGICAL_WORK_KEYS}
        if self.overflow_recoveries is None:
            self.overflow_recoveries = []
        if self.metrics_prefix is None:
            self.metrics_prefix = []
        if set(self.logical_work) != set(LOGICAL_WORK_KEYS) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.logical_work.values()
        ):
            raise ValueError(
                "logical work requires the exact nonnegative integer counters"
            )
        try:
            iterations = [int(item["iteration"]) for item in self.metrics_prefix]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("metrics prefix entries require an iteration") from exc
        if iterations != list(range(self.iteration)):
            raise ValueError(
                "metrics prefix must be the exact completed-iteration prefix"
            )

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
            "branch_controller_state": (
                None
                if self.branch_controller_state is None
                else dict(self.branch_controller_state)
            ),
            "cost_safeguard_state": (
                None
                if self.cost_safeguard_state is None
                else dict(self.cost_safeguard_state)
            ),
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
        "ambient_cuda_rng": tuple(
            value.cpu() for value in torch.cuda.get_rng_state_all()
        ),
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

_RUN_STATE_FIELDS = {
    "run_id",
    "iteration",
    "root_count",
    "retry_epoch",
    "attempt_logical_base",
    "logical_work",
    "recovered_overflow",
    "overflow_recoveries",
    "metrics_prefix",
    "branch_controller_state",
    "cost_safeguard_state",
}


def _validate_run_state_mapping(value: object) -> RunState:
    if not isinstance(value, dict) or set(value) != _RUN_STATE_FIELDS:
        raise PersistenceError("run state has unknown or missing fields")
    try:
        return RunState(**value)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"run state is invalid: {exc}") from exc


def validate_checkpoint_payload(state: object) -> dict[str, object]:
    """Validate one loaded queued checkpoint without mutating runtime owners."""

    if not isinstance(state, dict):
        raise PersistenceError("checkpoint payload must be a mapping")
    schema = state.get("checkpoint_schema")
    if schema == CHECKPOINT_SCHEMA:
        if set(state) != _CHECKPOINT_FIELDS:
            raise PersistenceError("checkpoint has unknown or missing top-level fields")
        configs = (state["scientific_config"],)
        hashes = (state["scientific_sha256"],)
        runs = (state["run_state"],)
    elif schema == REPLICA_CHECKPOINT_SCHEMA:
        if set(state) != _REPLICA_CHECKPOINT_FIELDS:
            raise PersistenceError(
                "replica checkpoint has unknown or missing top-level fields"
            )
        count = state.get("n_replica")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PersistenceError("replica checkpoint lane count is invalid")
        configs = state["scientific_config"]
        hashes = state["scientific_sha256"]
        runs = state["run_state"]
        if any(
            not isinstance(value, (tuple, list)) or len(value) != count
            for value in (configs, hashes, runs, state["explicit_rng"])
        ):
            raise PersistenceError("replica checkpoint lanes are not aligned")
    else:
        raise PersistenceError(
            "unsupported checkpoint schema"
        )
    if any(
        not isinstance(config, Mapping)
        or digest != _canonical_hash(config)
        for config, digest in zip(configs, hashes, strict=True)
    ):
        raise PersistenceError("checkpoint scientific configuration hash is invalid")
    for run in runs:
        _validate_run_state_mapping(run)
    return state


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
    """Strictly resume current state; retired schemas are never loaded here."""

    state = validate_checkpoint_payload(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    if state["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise PersistenceError(
            "unsupported checkpoint schema"
        )
    expected_hash = _canonical_hash(expected_scientific_config)
    if state["scientific_sha256"] != expected_hash or state[
        "scientific_config"
    ] != dict(expected_scientific_config):
        raise PersistenceError("resume rejected scientific configuration drift")
    model.load_state_dict(state["model"], strict=True)
    reference_model.load_state_dict(state["reference_model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler = LossScaler.from_state_dict(state["scaler"])
    optimizer.scaler = scaler
    reference = ReferenceState.from_state_dict(state["reference_state"])
    run_data = state["run_state"]
    run = _validate_run_state_mapping(run_data)
    generator.set_state(state["explicit_rng"])
    if restore_ambient_rng:
        torch.set_rng_state(state["ambient_cpu_rng"])
        cuda_states = state["ambient_cuda_rng"]
        if len(cuda_states) != torch.cuda.device_count():
            raise PersistenceError("checkpoint CUDA RNG rank/device count differs")
        torch.cuda.set_rng_state_all(list(cuda_states))
    return scaler, reference, run, state["execution_config"]


def save_replica_checkpoint(
    path: str | os.PathLike[str],
    *,
    modules: object,
    optimizers: object,
    run_states: Sequence[RunState],
    scientific_configs: Sequence[Mapping[str, object]],
    execution_config: Mapping[str, object],
    generators: Sequence[torch.Generator],
    reference_modules: object | None = None,
    reference_states: Sequence[ReferenceState] | None = None,
) -> str:
    """Atomically preserve every independent lane at one optimizer boundary."""

    from pbg.core.replica import ReplicaModuleOwner, ReplicaOptimizerGroup

    if not isinstance(modules, ReplicaModuleOwner) or not isinstance(
        optimizers, ReplicaOptimizerGroup
    ):
        raise TypeError("replica checkpoint requires queued replica owners")
    count = modules.n_replica
    if (
        optimizers.modules is not modules
        or len(run_states) != count
        or len(scientific_configs) != count
        or len(generators) != count
        or (reference_states is not None and len(reference_states) != count)
        or (
            reference_modules is not None
            and (
                not isinstance(reference_modules, ReplicaModuleOwner)
                or reference_modules.n_replica != count
            )
        )
    ):
        raise ValueError("replica checkpoint lane owners are not aligned")
    if (reference_modules is None) != (reference_states is None):
        raise ValueError(
            "replica reference models and cadence states must be owned together"
        )
    if any(
        generator.device.type != modules.device.type
        or (
            generator.device.index is not None
            and modules.device.index is not None
            and generator.device.index != modules.device.index
        )
        for generator in generators
    ):
        raise ValueError("replica checkpoint RNG must be lane-local on one device")
    state = {
        "checkpoint_schema": REPLICA_CHECKPOINT_SCHEMA,
        "n_replica": count,
        "scientific_config": tuple(dict(value) for value in scientific_configs),
        "scientific_sha256": tuple(
            _canonical_hash(value) for value in scientific_configs
        ),
        "execution_config": dict(execution_config),
        "model": _clone_cpu(modules.state_dict()),
        "optimizer": _clone_cpu(optimizers.state_dict()),
        "reference_model": (
            None
            if reference_modules is None
            else _clone_cpu(reference_modules.state_dict())
        ),
        "reference_state": (
            None
            if reference_states is None
            else tuple(value.state_dict() for value in reference_states)
        ),
        "run_state": tuple(value.state_dict() for value in run_states),
        "explicit_rng": tuple(generator.get_state().cpu() for generator in generators),
        "ambient_cpu_rng": torch.get_rng_state(),
        "ambient_cuda_rng": tuple(
            value.cpu() for value in torch.cuda.get_rng_state_all()
        ),
    }
    destination = Path(path)
    _atomic_torch_save(state, destination)
    return sha256_file(destination)


_REPLICA_CHECKPOINT_FIELDS = {
    "checkpoint_schema",
    "n_replica",
    "scientific_config",
    "scientific_sha256",
    "execution_config",
    "model",
    "optimizer",
    "reference_model",
    "reference_state",
    "run_state",
    "explicit_rng",
    "ambient_cpu_rng",
    "ambient_cuda_rng",
}


def load_replica_checkpoint(
    path: str | os.PathLike[str],
    *,
    modules: object,
    optimizers: object,
    expected_scientific_configs: Sequence[Mapping[str, object]],
    generators: Sequence[torch.Generator],
    reference_modules: object | None = None,
    restore_ambient_rng: bool = True,
) -> tuple[
    tuple[ReferenceState, ...] | None,
    tuple[RunState, ...],
    Mapping[str, object],
]:
    """Strictly restore a queued replica checkpoint without compatibility."""

    from pbg.core.replica import ReplicaModuleOwner, ReplicaOptimizerGroup

    if not isinstance(modules, ReplicaModuleOwner) or not isinstance(
        optimizers, ReplicaOptimizerGroup
    ):
        raise TypeError("replica checkpoint requires queued replica owners")
    state = validate_checkpoint_payload(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    if state["checkpoint_schema"] != REPLICA_CHECKPOINT_SCHEMA:
        raise PersistenceError("runtime accepts only queued replica checkpoints")
    count = modules.n_replica
    if (
        state["n_replica"] != count
        or optimizers.modules is not modules
        or len(expected_scientific_configs) != count
        or len(generators) != count
    ):
        raise PersistenceError("replica checkpoint lane count differs")
    expected = tuple(dict(value) for value in expected_scientific_configs)
    if state["scientific_config"] != expected or state["scientific_sha256"] != tuple(
        _canonical_hash(value) for value in expected
    ):
        raise PersistenceError("replica resume rejected scientific drift")
    modules.load_state_dict(state["model"])
    optimizers.load_state_dict(state["optimizer"])
    stored_reference = state["reference_model"]
    stored_reference_states = state["reference_state"]
    if (
        (stored_reference is None) != (reference_modules is None)
        or (stored_reference_states is None) != (reference_modules is None)
    ):
        raise PersistenceError("replica reference ownership differs")
    references: tuple[ReferenceState, ...] | None = None
    if reference_modules is not None:
        if (
            not isinstance(reference_modules, ReplicaModuleOwner)
            or reference_modules.n_replica != count
        ):
            raise PersistenceError("replica reference lanes are invalid")
        reference_modules.load_state_dict(stored_reference)
        if (
            not isinstance(stored_reference_states, (tuple, list))
            or len(stored_reference_states) != count
        ):
            raise PersistenceError("replica reference-state lanes are invalid")
        try:
            references = tuple(
                ReferenceState.from_state_dict(value)
                for value in stored_reference_states
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceError("replica reference-state lanes are invalid") from exc
    run_data = state["run_state"]
    runs = tuple(_validate_run_state_mapping(value) for value in run_data)
    rng = state["explicit_rng"]
    if not isinstance(rng, (tuple, list)) or len(rng) != count:
        raise PersistenceError("replica RNG lanes are invalid")
    for generator, value in zip(generators, rng, strict=True):
        generator.set_state(value)
    if restore_ambient_rng:
        torch.set_rng_state(state["ambient_cpu_rng"])
        cuda_states = state["ambient_cuda_rng"]
        if len(cuda_states) != torch.cuda.device_count():
            raise PersistenceError("checkpoint CUDA rank/device count differs")
        torch.cuda.set_rng_state_all(list(cuda_states))
    return references, runs, state["execution_config"]


def save_distributed_checkpoint_manifest(
    path: str | os.PathLike[str],
    *,
    rank_files: Sequence[str],
    rank_sha256: Sequence[str],
    scientific_config: Mapping[str, object],
    execution_config: Mapping[str, object],
) -> str:
    """Atomically publish one strict manifest after every rank file exists."""

    if (
        not rank_files
        or len(rank_files) != len(rank_sha256)
        or len(set(rank_files)) != len(rank_files)
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in rank_sha256
        )
    ):
        raise PersistenceError("distributed checkpoint rank inventory is invalid")
    state = {
        "checkpoint_schema": DISTRIBUTED_CHECKPOINT_SCHEMA,
        "world_size": len(rank_files),
        "scientific_config": dict(scientific_config),
        "scientific_sha256": _canonical_hash(scientific_config),
        "execution_config": dict(execution_config),
        "rank_files": tuple(rank_files),
        "rank_sha256": tuple(rank_sha256),
    }
    destination = Path(path)
    _atomic_torch_save(state, destination)
    return sha256_file(destination)


def load_distributed_checkpoint_manifest(
    path: str | os.PathLike[str],
    *,
    expected_scientific_config: Mapping[str, object],
    world_size: int,
) -> tuple[Path, ...]:
    """Validate a distributed manifest and every content-addressed rank file."""

    source = Path(path)
    state = torch.load(source, map_location="cpu", weights_only=False)
    fields = {
        "checkpoint_schema",
        "world_size",
        "scientific_config",
        "scientific_sha256",
        "execution_config",
        "rank_files",
        "rank_sha256",
    }
    if (
        not isinstance(state, dict)
        or set(state) != fields
        or state["checkpoint_schema"] != DISTRIBUTED_CHECKPOINT_SCHEMA
        or state["world_size"] != world_size
    ):
        raise PersistenceError("distributed checkpoint manifest is invalid")
    expected = dict(expected_scientific_config)
    if (
        state["scientific_config"] != expected
        or state["scientific_sha256"] != _canonical_hash(expected)
    ):
        raise PersistenceError("distributed resume rejected scientific drift")
    files = state["rank_files"]
    hashes = state["rank_sha256"]
    if (
        not isinstance(files, (tuple, list))
        or not isinstance(hashes, (tuple, list))
        or len(files) != world_size
        or len(hashes) != world_size
    ):
        raise PersistenceError("distributed checkpoint rank inventory is invalid")
    paths = tuple(source.parent / str(value) for value in files)
    if any(
        not rank_path.is_file() or sha256_file(rank_path) != expected_hash
        for rank_path, expected_hash in zip(paths, hashes, strict=True)
    ):
        raise PersistenceError("distributed checkpoint rank hash differs")
    return paths



def trim_metrics_after_checkpoint(
    metrics: Sequence[Mapping[str, object]], iteration: int
) -> list[dict[str, object]]:
    """Back up externally if desired, then retain the exact completed prefix."""

    retained = [dict(item) for item in metrics if int(item["iteration"]) < iteration]
    values = [int(item["iteration"]) for item in retained]
    if values != list(range(iteration)):
        raise PersistenceError(
            "metrics before the checkpoint are not the exact completed prefix"
        )
    return retained


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DISTRIBUTED_CHECKPOINT_SCHEMA",
    "REPLICA_CHECKPOINT_SCHEMA",
    "PersistenceError",
    "RunState",
    "load_checkpoint",
    "load_distributed_checkpoint_manifest",
    "load_replica_checkpoint",
    "model_state_sha256",
    "save_checkpoint",
    "save_distributed_checkpoint_manifest",
    "save_replica_checkpoint",
    "sha256_file",
    "trim_metrics_after_checkpoint",
    "validate_checkpoint_payload",
]
