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
    probe_target: dict[str, Any] | None = None,
    logical_work: dict[str, int] | None = None,
) -> None:
    """Atomically store model, optimizer, scaler, config, and explicit RNG state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    stored_config = scientific_config(cfg)
    hardware_profile = asdict(cfg).get("hardware")
    payload = {
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
        "probe_target": copy.deepcopy(probe_target),
        "logical_work": copy.deepcopy(logical_work),
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

    payload = torch.load(path, map_location=map_location, weights_only=False)
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
        "probe_target": copy.deepcopy(payload.get("probe_target")),
        "logical_work": copy.deepcopy(payload.get("logical_work")),
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
        # Memory forecasting is execution control, and these defaults were
        # introduced after existing scientific checkpoints were written.
        rollout.setdefault("memory_forecast_window", 128)
        rollout.setdefault("memory_forecast_half_life", 32.0)
        rollout.setdefault("memory_forecast_min_observations", 40)
        rollout.setdefault("memory_soft_exceedance_rate", 0.025)
        rollout.setdefault("memory_max_start_increase_fraction", 0.10)
        rollout.setdefault("memory_retry_backoff_fraction", 0.20)
        rollout.setdefault("memory_max_rollout_retries", 2)
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
