"""Atomic training checkpoints and resume-identity validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pbg.core.config import RootConfig
from pbg.core.adaptive_branching import BRANCHING_CONTROLLER_VERSION
from pbg.core.memory_control import SEQUENTIAL_MEMORY_CONTROLLER_VERSION
from pbg.core.storage import TRAINING_STORAGE_SCHEMA_VERSION


CHECKPOINT_SCHEMA_VERSION = "pbg-checkpoint-v5"
PREVIOUS_CHECKPOINT_SCHEMA_VERSION = "pbg-checkpoint-v4"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "pbg-checkpoint-v3"
ANCIENT_CHECKPOINT_SCHEMA_VERSION = "pbg-checkpoint-v2"
_RETIRED_ROLLOUT_CAPACITY_FIELD = "node" + "_budget"


def _migrate_rollout_config(migrated: dict[str, Any]) -> None:
    """Remove the retired allocation capacity without reinterpreting its value."""

    config = migrated.get("config")
    rollout = config.get("rollout") if isinstance(config, dict) else None
    if isinstance(rollout, dict) and _RETIRED_ROLLOUT_CAPACITY_FIELD in rollout:
        rollout = dict(rollout)
        rollout.pop(_RETIRED_ROLLOUT_CAPACITY_FIELD)
        config = dict(config)
        config["rollout"] = rollout
        migrated["config"] = config
        transitions = list(migrated.get("configuration_schema_transitions", ()))
        transitions.append("removed retired total-row allocation field; no replacement")
        migrated["configuration_schema_transitions"] = transitions


def _migrate_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata and deterministically migrate the recent v2 payload.

    Migration changes only metadata. It never reconstructs a legacy runtime
    tree, reach cache, target store, or learner allocation.
    """

    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")
    version = payload.get("checkpoint_schema_version")
    # Migration mutates metadata keys only. A shallow copy avoids duplicating
    # model/optimizer tensors, especially after map_location places them on a
    # GPU during the actual restore.
    migrated = dict(payload)
    if version is None:
        required = {
            "model", "optimizer", "scaler", "iteration", "config",
            "torch_generator", "numpy_generator", "torch_cpu_rng",
        }
        missing = required.difference(migrated)
        if missing:
            raise RuntimeError(
                "legacy checkpoint is partial; missing required fields "
                f"{sorted(missing)}"
            )
        migrated["migrated_from_checkpoint_schema"] = ANCIENT_CHECKPOINT_SCHEMA_VERSION
        migrated["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
        migrated["storage_schema_version"] = TRAINING_STORAGE_SCHEMA_VERSION
        state = migrated.get("adaptive_work_state")
        if state is not None and state.get("version") != SEQUENTIAL_MEMORY_CONTROLLER_VERSION:
            raise RuntimeError("legacy checkpoint has incompatible controller state")
        migrated["controller_schema_version"] = (
            None if state is None else SEQUENTIAL_MEMORY_CONTROLLER_VERSION
        )
    elif version in {PREVIOUS_CHECKPOINT_SCHEMA_VERSION, LEGACY_CHECKPOINT_SCHEMA_VERSION}:
        migrated["migrated_from_checkpoint_schema"] = version
        migrated["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
        if version == LEGACY_CHECKPOINT_SCHEMA_VERSION:
            migrated["branching_controller_schema_version"] = None
            migrated["branching_controller_state"] = None
    elif version != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported checkpoint schema version {version!r}")
    migrated.setdefault("branching_controller_schema_version", None)
    migrated.setdefault("branching_controller_state", None)
    migrated.setdefault("initialization", None)
    if migrated.get("storage_schema_version") == "pbg-training-storage-v1":
        migrated["storage_schema_version"] = TRAINING_STORAGE_SCHEMA_VERSION
    if migrated.get("storage_schema_version") != TRAINING_STORAGE_SCHEMA_VERSION:
        raise RuntimeError("checkpoint storage schema is incompatible")
    _migrate_rollout_config(migrated)
    controller_version = migrated.get("controller_schema_version")
    state = migrated.get("adaptive_work_state")
    if (state is None) != (controller_version is None):
        raise RuntimeError("checkpoint contains partial controller metadata")
    if state is not None and (
        controller_version != SEQUENTIAL_MEMORY_CONTROLLER_VERSION
        or state.get("version") != SEQUENTIAL_MEMORY_CONTROLLER_VERSION
    ):
        raise RuntimeError("checkpoint controller schema is incompatible")
    branching_state = migrated.get("branching_controller_state")
    branching_version = migrated.get("branching_controller_schema_version")
    if (branching_state is None) != (branching_version is None):
        raise RuntimeError("checkpoint contains partial adaptive KKT controller metadata")
    if branching_state is not None and (
        branching_version != BRANCHING_CONTROLLER_VERSION
        or branching_state.get("version") != BRANCHING_CONTROLLER_VERSION
    ):
        raise RuntimeError("checkpoint adaptive KKT controller schema is incompatible")
    rank_states = migrated.get("rank_rng_states")
    if rank_states is not None and any(
        not isinstance(item, dict)
        or not {"torch_generator", "numpy_generator", "torch_cpu_rng", "torch_cuda_rng"}.issubset(item)
        for item in rank_states
    ):
        raise RuntimeError("checkpoint contains a partial per-rank RNG state")
    return migrated


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Validate checkpoint compatibility on CPU before any GPU allocation."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    migrated = _migrate_checkpoint_payload(payload)
    return {
        "checkpoint_schema_version": migrated["checkpoint_schema_version"],
        "migrated_from_checkpoint_schema": migrated.get(
            "migrated_from_checkpoint_schema"
        ),
        "storage_schema_version": migrated["storage_schema_version"],
        "controller_schema_version": migrated["controller_schema_version"],
        "branching_controller_schema_version": migrated[
            "branching_controller_schema_version"
        ],
        "iteration": int(migrated["iteration"]),
        "config": copy.deepcopy(migrated["config"]),
        "configuration_schema_transitions": tuple(
            migrated.get("configuration_schema_transitions", ())
        ),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    iteration: int,
    cfg: RootConfig,
    torch_gen: torch.Generator,
    np_gen: np.random.Generator,
    *,
    run_id: str | None = None,
    rank_rng_states: list[dict[str, Any]] | None = None,
    reference_model: nn.Module | None = None,
    reference_iteration: int | None = None,
    adaptive_work_state: dict[str, Any] | None = None,
    branching_controller_state: dict[str, Any] | None = None,
    probe_target: dict[str, Any] | None = None,
    logical_work: dict[str, int] | None = None,
    initialization: dict[str, Any] | None = None,
) -> None:
    """Atomically store model, optimizer, scaler, config, and explicit RNG state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    stored_config = scientific_config(cfg)
    hardware_profile = asdict(cfg).get("hardware")
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "storage_schema_version": TRAINING_STORAGE_SCHEMA_VERSION,
        "controller_schema_version": (
            None
            if adaptive_work_state is None
            else SEQUENTIAL_MEMORY_CONTROLLER_VERSION
        ),
        "branching_controller_schema_version": (
            None if branching_controller_state is None else BRANCHING_CONTROLLER_VERSION
        ),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": int(iteration),
        "config": stored_config,
        "hardware_profile": hardware_profile,
        "torch_generator": torch_gen.get_state(),
        "numpy_generator": np_gen.bit_generator.state,
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "run_id": run_id,
        "rank_rng_states": rank_rng_states,
        "reference_model": (
            None if reference_model is None else reference_model.state_dict()
        ),
        "reference_iteration": reference_iteration,
        "adaptive_work_state": adaptive_work_state,
        "branching_controller_state": branching_controller_state,
        "probe_target": copy.deepcopy(probe_target),
        "logical_work": copy.deepcopy(logical_work),
        "initialization": copy.deepcopy(initialization),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    torch_gen: torch.Generator | None = None,
    np_gen: np.random.Generator | None = None,
    *,
    map_location: str | torch.device = "cpu",
    rank: int = 0,
    reference_model: nn.Module | None = None,
) -> dict[str, Any]:
    """Restore checkpoint state and return iteration/config metadata."""

    payload = _migrate_checkpoint_payload(
        torch.load(path, map_location=map_location, weights_only=False)
    )
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload["model"])
    reference_state = payload.get("reference_model")
    if reference_model is not None:
        if reference_state is None:
            raise RuntimeError("checkpoint has no KL reference-policy state")
        reference_model.load_state_dict(reference_state)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    rank_states = payload.get("rank_rng_states")
    rng = None
    if rank_states is not None:
        if not 0 <= rank < len(rank_states):
            raise RuntimeError("checkpoint has no RNG state for this DDP rank")
        rng = rank_states[rank]
    if torch_gen is not None:
        state = payload["torch_generator"] if rng is None else rng["torch_generator"]
        torch_gen.set_state(state.cpu())
    if np_gen is not None:
        np_gen.bit_generator.state = (
            payload["numpy_generator"] if rng is None else rng["numpy_generator"]
        )
    torch.set_rng_state(
        (payload["torch_cpu_rng"] if rng is None else rng["torch_cpu_rng"]).cpu()
    )
    cuda_rng = payload["torch_cuda_rng"] if rng is None else rng["torch_cuda_rng"]
    if torch.cuda.is_available() and cuda_rng is not None:
        # CUDA's RNG setter specifically requires CPU ByteTensor states even
        # when the checkpoint's model/optimizer tensors map directly to CUDA.
        torch.cuda.set_rng_state_all(
            [state.detach().to(device="cpu", dtype=torch.uint8) for state in cuda_rng]
        )
    return {
        "iter": int(payload["iteration"]),
        "cfg_dict": payload["config"],
        "run_id": payload.get("run_id"),
        "reference_iteration": payload.get("reference_iteration"),
        "adaptive_work_state": payload.get("adaptive_work_state"),
        "branching_controller_state": payload.get("branching_controller_state"),
        "probe_target": copy.deepcopy(payload.get("probe_target")),
        "logical_work": copy.deepcopy(payload.get("logical_work")),
        "initialization": copy.deepcopy(payload.get("initialization")),
        "checkpoint_schema_version": payload["checkpoint_schema_version"],
        "migrated_from_checkpoint_schema": payload.get(
            "migrated_from_checkpoint_schema"
        ),
        "storage_schema_version": payload["storage_schema_version"],
        "controller_schema_version": payload["controller_schema_version"],
        "branching_controller_schema_version": payload[
            "branching_controller_schema_version"
        ],
    }


def capture_rng_state(
    torch_gen: torch.Generator, np_gen: np.random.Generator
) -> dict[str, Any]:
    """Capture complete process and explicit-generator RNG state."""

    return {
        "torch_generator": torch_gen.get_state(),
        "numpy_generator": copy.deepcopy(np_gen.bit_generator.state),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def scientific_config(cfg: RootConfig) -> dict[str, Any]:
    """Return scientific configuration, excluding hardware and log cadence."""

    result = asdict(cfg)
    result.pop("hardware", None)
    # Checkpoint frequency changes persistence policy, not the estimator,
    # model, optimizer, or RNG trajectory. It must not invalidate resume.
    result["training"].pop("checkpoint_every", None)
    if result.get("probe") is not None:
        # The content hash is scientific identity; the filesystem location is not.
        result["probe"].pop("target_checkpoint", None)
    return result


def compatible_scientific_configs(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    """Compare current/legacy checkpoint configs ignoring log cadence."""

    def normalized(value: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(value)
        result.pop("hardware", None)
        # Checkpoints written before probe training existed omit the key;
        # absence is equivalent to the current schema's explicit null.
        result.setdefault("probe", None)
        result.get("training", {}).pop("checkpoint_every", None)
        if result.get("probe") is not None:
            result["probe"].pop("target_checkpoint", None)
        rollout = result.get("rollout", {})
        rollout.pop(_RETIRED_ROLLOUT_CAPACITY_FIELD, None)
        # Width bounds and calibration are execution controls.  They do not
        # change the estimator represented by a scientific checkpoint.
        for key in (
            "sequential_roots_per_gpu_minimum",
            "sequential_roots_per_gpu_maximum",
            "memory_calibration_window",
            "memory_growth_fraction",
            "memory_growth_delay",
            "memory_prediction_z",
        ):
            rollout.pop(key, None)
        return result

    return normalized(left) == normalized(right)


def legacy_config_fingerprint(cfg: RootConfig) -> str:
    """Return the pre-cadence-separation fingerprint for old run manifests."""

    result = asdict(cfg)
    result.pop("hardware", None)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def config_dict_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint a stored legacy scientific-config dictionary."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def matches_legacy_run_fingerprint(
    recorded: str | None, checkpoint_config: dict[str, Any]
) -> bool:
    """Match an old run manifest while allowing only checkpoint cadence drift.

    Old manifests hashed ``training.checkpoint_every`` even though it controls
    persistence rather than the scientific trajectory.  A later checkpoint
    may therefore be scientifically compatible but carry a different cadence.
    Recover the original manifest hash by exhaustively trying the finite set of
    positive cadences within the configured training horizon.  No other field
    is relaxed.
    """

    if not isinstance(recorded, str) or len(recorded) != 64:
        return False
    if config_dict_fingerprint(checkpoint_config) == recorded:
        return True
    training = checkpoint_config.get("training")
    if not isinstance(training, dict):
        return False
    iterations = training.get("iters")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        return False
    candidate = copy.deepcopy(checkpoint_config)
    candidate_training = candidate["training"]
    for cadence in range(1, iterations + 1):
        candidate_training["checkpoint_every"] = cadence
        if config_dict_fingerprint(candidate) == recorded:
            return True
    return False


def config_fingerprint(cfg: RootConfig) -> str:
    """Return the stable SHA-256 identity of the scientific configuration."""

    encoded = json.dumps(
        scientific_config(cfg), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_metrics_prefix(path: Path, completed: int, run_id: str) -> None:
    """Validate and trim metrics newer than the selected resume checkpoint."""

    if not path.exists():
        raise RuntimeError("resume requires the existing metrics.jsonl")
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in raw_lines]
    iterations = [record.get("iter") for record in records]
    if len(records) < completed or iterations[:completed] != list(range(completed)):
        raise RuntimeError(
            "metrics.jsonl is not the exact gap-free prefix required by the checkpoint"
        )
    if any(record.get("run_id") != run_id for record in records[:completed]):
        raise RuntimeError("metrics.jsonl belongs to a different run identity")
    if len(records) > completed:
        backup = path.with_name(f"{path.name}.before_resume_{completed:07d}.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text("\n".join(raw_lines[:completed]) + ("\n" if completed else ""), encoding="utf-8")


# Compatibility aliases retained for existing internal imports while callers
# migrate to the cohesive checkpointing module.
_capture_rng_state = capture_rng_state
_scientific_config = scientific_config
_config_fingerprint = config_fingerprint
_validate_metrics_prefix = validate_metrics_prefix
