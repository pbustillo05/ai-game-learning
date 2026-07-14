"""Repeatable comparative performance profiling for the three training modes."""

from __future__ import annotations

import copy
import gc
import os
import platform
import statistics
from typing import Any, Iterable

import torch

from pbg.core.config import RootConfig, seed_all
from pbg.core.training import (
    _one_iteration,
    _precision,
    _scaler,
    build_game_bundle,
)


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
        cfg.rollout.n_trees = 1
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
) -> dict[str, Any]:
    """Profile modes and return median phase/cost metrics plus raw samples."""

    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    results: list[dict[str, Any]] = []
    environment: dict[str, Any] | None = None
    for mode in modes:
        local = _mode_config(cfg, mode)
        requested = torch.device(
            "cuda:0"
            if local.game.name == "poker" and local.completion.backend == "cuda"
            else "cpu"
        )
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
        if requested.type == "cuda":
            torch.cuda.set_device(requested)
            torch.cuda.empty_cache()
            setup_allocated = torch.cuda.memory_allocated(requested)
            setup_allowance = 0 if not results else 64 * 1024**2
            if setup_allocated > setup_allowance:
                raise RuntimeError(
                    "benchmark requires a clean CUDA allocator before setup; "
                    f"found {setup_allocated} bytes"
                )
            free_before, _ = torch.cuda.mem_get_info(requested)
        else:
            free_before = 0
        bundle = build_game_bundle(local, requested)
        device = torch.device(bundle.rollout_backend.device)
        bundle.model.to(device)
        torch_gen, _ = seed_all(local.training.seed, 0, device)
        optimizer = torch.optim.Adam(
            bundle.model.parameters(),
            lr=local.training.lr,
            betas=local.training.adam_betas,
        )
        amp_enabled, amp_dtype = _precision(local, device)
        scaler = _scaler(bool(amp_enabled and amp_dtype == torch.float16))
        environment = _environment(device)
        for iteration in range(warmup):
            _one_iteration(
                local, bundle, optimizer, scaler, torch_gen, iteration, profile=False
            )
        model_state = _cpu_snapshot(bundle.model.state_dict())
        optimizer_state = _cpu_snapshot(optimizer.state_dict())
        scaler_state = _cpu_snapshot(scaler.state_dict())
        generator_state = torch_gen.get_state()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        samples: list[dict[str, float]] = []
        for repeat in range(repeats):
            bundle.model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            scaler.load_state_dict(scaler_state)
            torch_gen.set_state(generator_state)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            samples.append(_one_iteration(
                local,
                bundle,
                optimizer,
                scaler,
                torch_gen,
                warmup + repeat,
                profile=True,
            ))
        results.append(
            {
                "mode": mode,
                "workload": {
                    "n_trees": local.rollout.n_trees,
                    "beta": local.rollout.beta,
                    "completion_samples": local.completion.n_samples,
                },
                "metrics": _median_metrics(samples),
                "samples": samples,
            }
        )
        del scaler, optimizer, bundle
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            context_allowance = 64 * 1024**2
            if allocated > context_allowance or reserved > context_allowance:
                raise RuntimeError(
                    "benchmark left CUDA allocations above the framework-context allowance: "
                    f"allocated={allocated}, reserved={reserved}"
                )
            driver_context_allowance = 512 * 1024**2
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
            "game": cfg.game.name,
            "n_trees": cfg.rollout.n_trees,
            "node_budget": cfg.rollout.node_budget,
            "completion_samples": cfg.completion.n_samples,
            "minibatch_size": cfg.training.minibatch_size,
            "gd_steps_per_iter": cfg.training.gd_steps_per_iter,
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
        "n_trees",
        "node_budget",
        "completion_samples",
        "minibatch_size",
        "gd_steps_per_iter",
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
