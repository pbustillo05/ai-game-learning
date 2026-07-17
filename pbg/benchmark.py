"""Repeatable comparative performance profiling for the three training modes."""

from __future__ import annotations

import copy
import gc
import hashlib
import os
import platform
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import torch

from pbg.core.config import RootConfig, seed_all
from pbg.core.execution import LOGICAL_RNG_VERSION
from pbg.core.training import (
    _one_iteration,
    _precision,
    _scaler,
)
from pbg.runtime import build_game_bundle


MODES = ("normal", "rao_blackwell", "branched_rao_blackwell")
LOWER_IS_BETTER = (
    "profiled_iteration_seconds",
    "target_bytes_per_decision",
    "retained_target_bytes",
)
HIGHER_IS_BETTER = (
    "decisions_per_second",
    "terminal_samples_per_second",
    "completion_effective_samples_per_second",
)

MODEL_INITIALIZATION_PROTOCOL = "torch_seed_before_bundle_v1"
CHECKPOINT_INITIALIZATION_PROTOCOL = "checkpoint_model_state_v1"


def _benchmark_checkpoint(
    path: str | Path | None,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    """Load one read-only model state and stable identity for benchmarking."""

    if path is None:
        return None, None
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError(f"checkpoint {checkpoint} has no model state")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "path": str(checkpoint),
        "sha256": digest.hexdigest(),
        "iteration": int(payload.get("iteration", -1)),
        "run_id": payload.get("run_id"),
    }
    return dict(payload["model"]), metadata


def _mode_config(base: RootConfig, mode: str) -> RootConfig:
    if mode not in MODES:
        raise ValueError(f"unknown benchmark mode {mode!r}")
    cfg = copy.deepcopy(base)
    cfg.training.mode = mode
    if mode == "normal":
        if cfg.loss.name == "rb_ppo":
            cfg.loss.name = "ppo"
        cfg.rollout.beta = None
        cfg.completion.n_samples = None
    elif mode == "rao_blackwell":
        if cfg.loss.name == "ppo":
            cfg.loss.name = "rb_ppo"
        cfg.rollout.beta = None
        cfg.completion.n_samples = cfg.completion.n_samples or 64
    else:
        if cfg.loss.name == "ppo":
            cfg.loss.name = "rb_ppo"
        # Branched rollout owns one labeled public tree per rank; its realized
        # decision count, rather than root count, is the cross-mode normalizer.
        cfg.rollout.roots_per_gpu = 1
        cfg.rollout.beta = cfg.rollout.beta or {"default": 1.5}
        cfg.completion.n_samples = cfg.completion.n_samples or 64
    return cfg


def _environment(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        device_name = props.name
        capability: list[int] | None = [props.major, props.minor]
    else:
        device_name = platform.processor() or platform.uname().machine
        capability = None
    return {
        "device_type": device.type,
        "device_name": device_name,
        "compute_capability": capability,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
    }


def _median_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(sample) for sample in samples))
    return {
        key: float(statistics.median(sample[key] for sample in samples))
        for key in sorted(keys)
        if all(isinstance(sample[key], (int, float)) for sample in samples)
    }


def _cpu_snapshot(value: Any) -> Any:
    """Clone nested benchmark reset state without retaining device storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def benchmark_modes(
    cfg: RootConfig,
    modes: Iterable[str] = MODES,
    *,
    warmup: int = 1,
    repeats: int = 5,
    device: torch.device | str | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Profile modes and return median phase/cost metrics plus raw samples."""

    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    checkpoint_state, checkpoint_metadata = _benchmark_checkpoint(checkpoint)
    initialization_protocol = (
        CHECKPOINT_INITIALIZATION_PROTOCOL
        if checkpoint_state is not None
        else MODEL_INITIALIZATION_PROTOCOL
    )
    results: list[dict[str, Any]] = []
    environment: dict[str, Any] | None = None
    for mode in modes:
        local = _mode_config(cfg, mode)
        requested = torch.device(
            device
            if device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
        if requested.type == "cuda":
            torch.cuda.set_device(requested)
            torch.cuda.empty_cache()
            setup_allocated = torch.cuda.memory_allocated(requested)
            setup_allowance = 0 if not results else 128 * 1024**2
            if setup_allocated > setup_allowance:
                raise RuntimeError(
                    "benchmark requires a clean CUDA allocator before setup; "
                    f"found {setup_allocated} bytes"
                )
            free_before, _ = torch.cuda.mem_get_info(requested)
        else:
            free_before = 0
        # Model construction consumes the process-global torch RNG.  Seed it
        # before building the bundle so separately launched benchmark artifacts
        # start from the same frozen policy, then reset again below so bundle
        # setup does not perturb rollout/training draws.
        seed_all(local.training.seed, 0, requested)
        bundle = build_game_bundle(local, requested)
        device = torch.device(bundle.rollout_backend.device)
        bundle.model.to(device)
        if checkpoint_state is not None:
            bundle.model.load_state_dict(checkpoint_state, strict=True)
        torch_gen, _ = seed_all(local.training.seed, 0, device)
        optimizer_cls = (
            torch.optim.AdamW
            if local.training.optimizer == "adamw"
            else torch.optim.Adam
        )
        optimizer = optimizer_cls(
            bundle.model.parameters(),
            lr=local.training.lr,
            betas=local.training.adam_betas,
        )
        source = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
        reference_model = None
        if local.loss.c_kl > 0.0:
            reference_model = copy.deepcopy(source).eval()
            reference_model.requires_grad_(False)
        amp_enabled, amp_dtype = _precision(local, device)
        scaler = _scaler(bool(amp_enabled and amp_dtype == torch.float16))
        environment = _environment(device)
        for iteration in range(warmup):
            _one_iteration(
                local,
                bundle,
                optimizer,
                scaler,
                torch_gen,
                iteration,
                profile=False,
                reference_model=reference_model,
            )
        model_state = _cpu_snapshot(bundle.model.state_dict())
        optimizer_state = _cpu_snapshot(optimizer.state_dict())
        scaler_state = _cpu_snapshot(scaler.state_dict())
        repeat_seeds = [
            int(local.training.seed) + 104_729 * (repeat + 1)
            for repeat in range(repeats)
        ]
        profile_iteration = warmup
        samples: list[dict[str, float]] = []
        for repeat_seed in repeat_seeds:
            bundle.model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            scaler.load_state_dict(scaler_state)
            torch_gen, _ = seed_all(repeat_seed, 0, device)
            local.rollout.logical_seed = repeat_seed
            sample = _one_iteration(
                local,
                bundle,
                optimizer,
                scaler,
                torch_gen,
                profile_iteration,
                profile=True,
                reference_model=reference_model,
            )
            decisions = float(sample["decisions"])
            if decisions <= 0:
                raise RuntimeError("performance samples require realized decisions")
            sample["phase_rollout_seconds_per_decision"] = (
                sample["phase_rollout_seconds"] / decisions
            )
            sample["profiled_iteration_seconds_per_decision"] = (
                sample["profiled_iteration_seconds"] / decisions
            )
            samples.append(sample)
        results.append(
            {
                "mode": mode,
                "workload": {
                    "roots_per_gpu": local.rollout.roots_per_gpu,
                    "beta": local.rollout.beta,
                    "completion_samples": local.completion.n_samples,
                    "completion_backend": local.completion.backend,
                    "logical_rng": LOGICAL_RNG_VERSION,
                    "training_seed": local.training.seed,
                    "model_initialization_protocol": initialization_protocol,
                    "model_initialization_seed": local.training.seed,
                    "model_checkpoint": checkpoint_metadata,
                    "repeat_seeds": repeat_seeds,
                    "profile_iteration": profile_iteration,
                    "family_batch_rows": local.rollout.family_batch_rows,
                    "max_active_trajectories": local.rollout.max_active_trajectories,
                    "arena_chunk_nodes": local.rollout.arena_chunk_nodes,
                },
                "metrics": _median_metrics(samples),
                "samples": samples,
            }
        )
        del scaler, optimizer, reference_model, source, bundle
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            # CUDA 13 plus the evaluator JIT extension retains process-global
            # module/context allocations. Live tensors retain the measured
            # 128 MiB ceiling. Large compiled calls can leave non-releasable
            # allocator segments after empty_cache(), so cached reservation has
            # a separate ceiling; the driver free-memory delta below remains an
            # independent process-level cleanup gate.
            allocated_allowance = 128 * 1024**2
            reserved_allowance = 256 * 1024**2
            if allocated > allocated_allowance or reserved > reserved_allowance:
                raise RuntimeError(
                    "benchmark left CUDA allocations above the framework-context allowance: "
                    f"allocated={allocated}, reserved={reserved}"
                )
            # The evaluator JIT plus the larger sequential kernel set
            # retains about 0.5 GiB even at 512 starts and grows with compiled
            # shapes. Keep this below the hardware profile's 2 GiB reserve and
            # continue recording the exact delta in every emitted artifact.
            driver_context_allowance = 1024 * 1024**2
            if free_after + driver_context_allowance < free_before:
                raise RuntimeError("GPU did not return to its pre-benchmark free-memory state")
            assert environment is not None
            environment["teardown_allocated_bytes"] = allocated
            environment["teardown_reserved_bytes"] = reserved
            environment["teardown_free_memory_delta_bytes"] = free_after - free_before
    return {
        "schema": "pbg-performance-v1",
        "environment": environment,
        "workload": {
            "game": "nlhe",
            "roots_per_gpu": cfg.rollout.roots_per_gpu,
            "node_budget": cfg.rollout.node_budget,
            "completion_samples": cfg.completion.n_samples,
            "completion_backend": cfg.completion.backend,
            "minibatch_size": cfg.training.minibatch_size,
            "gd_steps_per_iter": cfg.training.gd_steps_per_iter,
            "logical_rng": LOGICAL_RNG_VERSION,
            "training_seed": cfg.training.seed,
            "model_initialization_protocol": initialization_protocol,
            "model_initialization_seed": cfg.training.seed,
            "model_checkpoint": checkpoint_metadata,
            "repeat_seeds": [
                int(cfg.training.seed) + 104_729 * (repeat + 1)
                for repeat in range(repeats)
            ],
            "profile_iteration": warmup,
            "family_batch_rows": cfg.rollout.family_batch_rows,
            "max_active_trajectories": cfg.rollout.max_active_trajectories,
            "arena_chunk_nodes": cfg.rollout.arena_chunk_nodes,
            "warmup": warmup,
            "repeats": repeats,
        },
        "modes": results,
    }


def compare_benchmark(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_regression: float = 0.20,
    allow_environment_mismatch: bool = False,
) -> list[str]:
    """Return human-readable gate failures against a same-workload baseline."""

    if not 0.0 <= max_regression < 1.0:
        raise ValueError("max_regression must be in [0,1)")
    if current.get("schema") != "pbg-performance-v1" or baseline.get("schema") != current.get("schema"):
        raise ValueError("benchmark schema mismatch")
    environment_keys = (
        "device_type",
        "device_name",
        "compute_capability",
        "torch",
        "cuda",
        "torch_threads",
    )
    if not allow_environment_mismatch and any(
        current["environment"].get(key) != baseline["environment"].get(key)
        for key in environment_keys
    ):
        raise ValueError("benchmark environment does not match the baseline")
    workload_keys = (
        "game",
        "roots_per_gpu",
        "node_budget",
        "completion_samples",
        "completion_backend",
        "minibatch_size",
        "gd_steps_per_iter",
        "logical_rng",
        "training_seed",
        "model_initialization_protocol",
        "model_initialization_seed",
        "model_checkpoint",
        "repeat_seeds",
        "profile_iteration",
        "family_batch_rows",
        "max_active_trajectories",
        "arena_chunk_nodes",
    )
    if any(current["workload"].get(key) != baseline["workload"].get(key) for key in workload_keys):
        raise ValueError("benchmark workload does not match the baseline")
    old_by_mode = {record["mode"]: record for record in baseline["modes"]}
    failures: list[str] = []
    for record in current["modes"]:
        mode = record["mode"]
        if mode not in old_by_mode:
            failures.append(f"{mode}: absent from baseline")
            continue
        now = record["metrics"]
        old = old_by_mode[mode]["metrics"]
        if record.get("workload") != old_by_mode[mode].get("workload"):
            raise ValueError(f"{mode} benchmark workload does not match the baseline")
        for metric in HIGHER_IS_BETTER:
            if metric in now and metric in old and now[metric] < old[metric] * (1.0 - max_regression):
                failures.append(
                    f"{mode}.{metric}: {now[metric]:.6g} < allowed "
                    f"{old[metric] * (1.0 - max_regression):.6g}"
                )
        for metric in LOWER_IS_BETTER:
            if metric in now and metric in old and now[metric] > old[metric] * (1.0 + max_regression):
                failures.append(
                    f"{mode}.{metric}: {now[metric]:.6g} > allowed "
                    f"{old[metric] * (1.0 + max_regression):.6g}"
                )
        peak_keys = [
            key for key in now if key.startswith("phase_") and key.endswith("_peak_reserved_bytes")
        ]
        for metric in peak_keys:
            if metric in old and now[metric] > old[metric] * (1.0 + max_regression):
                failures.append(
                    f"{mode}.{metric}: {now[metric]:.6g} > allowed "
                    f"{old[metric] * (1.0 + max_regression):.6g}"
                )
    return failures
