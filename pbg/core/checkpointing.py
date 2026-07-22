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


CHECKPOINT_SCHEMA_VERSION = "pbg-checkpoint-v6"
_REQUIRED_CHECKPOINT_FIELDS = frozenset(
    {
        "checkpoint_schema_version",
        "storage_schema_version",
        "controller_schema_version",
        "branching_controller_schema_version",
        "model",
        "optimizer",
        "scaler",
        "iteration",
        "config",
        "hardware_profile",
        "torch_generator",
        "numpy_generator",
        "torch_cpu_rng",
        "torch_cuda_rng",
        "run_id",
        "rank_rng_states",
        "reference_model",
        "reference_iteration",
        "adaptive_work_state",
        "branching_controller_state",
        "probe_target",
        "logical_work",
        "initialization",
    }
)


def validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Require the sole current checkpoint schema without runtime migration."""

    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")
    version = payload.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported checkpoint schema version {version!r}")
    missing = _REQUIRED_CHECKPOINT_FIELDS.difference(payload)
    extra = set(payload).difference(_REQUIRED_CHECKPOINT_FIELDS)
    if missing or extra:
        raise RuntimeError(
            f"checkpoint {CHECKPOINT_SCHEMA_VERSION} fields do not match the current "
            f"schema; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if payload.get("storage_schema_version") != TRAINING_STORAGE_SCHEMA_VERSION:
        raise RuntimeError("checkpoint storage schema is incompatible")
    if not isinstance(payload["model"], dict) or not isinstance(payload["config"], dict):
        raise RuntimeError("checkpoint model and config must be mappings")
    controller_version = payload["controller_schema_version"]
    state = payload["adaptive_work_state"]
    if (state is None) != (controller_version is None):
        raise RuntimeError("checkpoint contains partial controller metadata")
    if state is not None and (
        controller_version != SEQUENTIAL_MEMORY_CONTROLLER_VERSION
        or state.get("version") != SEQUENTIAL_MEMORY_CONTROLLER_VERSION
    ):
        raise RuntimeError("checkpoint controller schema is incompatible")
    branching_state = payload["branching_controller_state"]
    branching_version = payload["branching_controller_schema_version"]
    if (branching_state is None) != (branching_version is None):
        raise RuntimeError("checkpoint contains partial adaptive KKT controller metadata")
    if branching_state is not None and (
        branching_version != BRANCHING_CONTROLLER_VERSION
        or branching_state.get("version") != BRANCHING_CONTROLLER_VERSION
    ):
        raise RuntimeError("checkpoint adaptive KKT controller schema is incompatible")
    rank_states = payload["rank_rng_states"]
    if rank_states is not None and any(
        not isinstance(item, dict)
        or not {"torch_generator", "numpy_generator", "torch_cpu_rng", "torch_cuda_rng"}.issubset(item)
        for item in rank_states
    ):
        raise RuntimeError("checkpoint contains a partial per-rank RNG state")
    return payload


def load_checkpoint_payload(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load and validate one current-schema project checkpoint."""

    return validate_checkpoint_payload(
        torch.load(path, map_location=map_location, weights_only=False)
    )


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Validate the current checkpoint schema on CPU before GPU allocation."""

    payload = load_checkpoint_payload(path)
    return {
        "checkpoint_schema_version": payload["checkpoint_schema_version"],
        "storage_schema_version": payload["storage_schema_version"],
        "controller_schema_version": payload["controller_schema_version"],
        "branching_controller_schema_version": payload[
            "branching_controller_schema_version"
        ],
        "iteration": int(payload["iteration"]),
        "config": copy.deepcopy(payload["config"]),
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

    payload = load_checkpoint_payload(path, map_location=map_location)
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
