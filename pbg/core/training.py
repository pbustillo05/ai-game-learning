"""Minimal rollout/target/optimization training loop with DDP checkpoints."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import platform
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from pbg.core.config import RootConfig, seed_all
from pbg.core.checkpointing import (
    _capture_rng_state,
    config_dict_fingerprint,
    _config_fingerprint,
    compatible_scientific_configs,
    legacy_config_fingerprint,
    matches_legacy_run_fingerprint,
    _scientific_config,
    _validate_metrics_prefix,
    load_checkpoint,
    save_checkpoint,
)
from pbg.core.distributed import stratum_objective_scale
from pbg.core.game import GameBundle
from pbg.core.losses import COMPOSITES, LossSpec, _selected_logp, finalize_metric_stats
from pbg.core.memory_control import (
    MEMORY_PHASES,
    MemoryAdmissionStop,
    MemoryLedger,
    PhaseMemoryForecaster,
    PhaseMemoryModel,
    PhaseMemoryObservation,
    SequentialMemoryController,
)
from pbg.core.minibatch import (
    gather_minibatch_rows,
    index_minibatch,
    sample_minibatch,
)
from pbg.core.normal import (
    build_normal_dataset_from_sampled,
    gather_normal_minibatch,
    index_normal_minibatch,
    sample_normal_minibatch,
)
from pbg.core.optimization import (
    FlatGradientBuffer,
    finite_gated_optimizer_step as _finite_gated_optimizer_step,
    sync_scaled_gradients as _sync_scaled_gradients,
)
from pbg.core.rollout import (
    ProbeRollout,
    sample_trees,
    sample_trees_sequential,
    sequential_leaf_logical_rng_keys,
)
from pbg.core.execution import LOGICAL_RNG_VERSION, DrawPurpose
from pbg.core.tree import DecisionReachCache
from pbg.core.values import HeadEvals, streamed_branched_value


def _configure_reproducibility(profile: str) -> None:
    """Apply one explicit backend reproducibility/performance profile."""

    strict = profile == "strict"
    throughput = profile == "throughput"
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = throughput
    torch.backends.cudnn.allow_tf32 = throughput
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = throughput
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if throughput else "highest")


def _scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler(
            "cuda", enabled=enabled, init_scale=1024.0, growth_interval=2000
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled, init_scale=1024.0, growth_interval=2000
        )


def _loss_spec(cfg: RootConfig) -> LossSpec:
    try:
        return COMPOSITES[cfg.loss.name]
    except KeyError as exc:
        raise ValueError(f"unknown composite loss {cfg.loss.name!r}") from exc


def _precision(cfg: RootConfig, device: torch.device) -> tuple[bool, torch.dtype]:
    name = "fp16" if cfg.hardware is None else cfg.hardware.precision
    if device.type != "cuda" or name == "fp32" or not cfg.training.amp:
        return False, torch.float32
    return True, torch.bfloat16 if name == "bf16" else torch.float16


def _hardware_metadata(
    cfg: RootConfig, device: torch.device, world: int
) -> dict[str, Any]:
    """Validate the selected device and return reproducibility metadata."""

    profile = cfg.hardware
    metadata: dict[str, Any] = {
        "hardware_profile": None if profile is None else profile.name,
        "device": str(device),
        "world_size": world,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_toolkit": os.environ.get("CUDA_HOME"),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "scheduler_time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "reproducibility": cfg.training.reproducibility,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "logical_rng": LOGICAL_RNG_VERSION,
        "scheduler_control_sync_policy": "one_fixed_control_block_per_decision_v1",
        "completion_backend": cfg.completion.backend,
        "execution_calibration": {
            "batch_rows": cfg.rollout.family_batch_rows,
            "max_active_trajectories": cfg.rollout.max_active_trajectories,
            "arena_chunk_nodes": cfg.rollout.arena_chunk_nodes,
        },
        "memory_limits": {
            "execution_memory_bytes": cfg.rollout.execution_memory_bytes,
            "node_budget": cfg.rollout.node_budget,
            "reserve_vram_gib": None if cfg.hardware is None else cfg.hardware.reserve_vram_gib,
            "max_memory_fraction": None if cfg.hardware is None else cfg.hardware.max_memory_fraction,
        },
    }
    try:
        evaluator = importlib.import_module("poker_gpu_evaluator.pbg")
        metadata["evaluator_api_version"] = evaluator.API_VERSION
        metadata["evaluator_module"] = str(Path(evaluator.__file__).resolve())
    except (ImportError, AttributeError):
        metadata["evaluator_api_version"] = None
        metadata["evaluator_module"] = None
    if cfg.training.mode == "branched_rao_blackwell":
        metadata["rank_start_quotas"] = [1] * world
        metadata["sequential_roots_per_gpu"] = None
    else:
        metadata["rank_start_quotas"] = [cfg.rollout.roots_per_gpu] * world
        metadata["sequential_roots_per_gpu"] = cfg.rollout.roots_per_gpu
    if profile is None:
        return metadata
    if world > profile.max_devices:
        raise RuntimeError(
            f"hardware profile {profile.name!r} allows at most {profile.max_devices} devices"
        )
    if device.type != "cuda":
        metadata["attention_backend"] = "cpu_reference"
        return metadata
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = 1024**3
    capability = (props.major, props.minor)
    if (
        profile.expected_compute_capability is not None
        and capability != profile.expected_compute_capability
    ):
        raise RuntimeError(
            f"hardware profile {profile.name!r} expects compute capability "
            f"{profile.expected_compute_capability}, found {capability}"
        )
    total_gib = total_bytes / gib
    if profile.min_vram_gib is not None and total_gib < profile.min_vram_gib:
        raise RuntimeError("visible GPU has less VRAM than the hardware profile requires")
    if profile.max_vram_gib is not None and total_gib > profile.max_vram_gib:
        raise RuntimeError("visible GPU has more VRAM than the selected validated tier")
    reserve = int(profile.reserve_vram_gib * gib)
    usable = min(free_bytes - reserve, int(total_bytes * profile.max_memory_fraction))
    if usable <= 0:
        raise MemoryError(
            f"hardware profile {profile.name!r} cannot preserve its "
            f"{profile.reserve_vram_gib:.1f} GiB reserve on {device}"
        )
    metadata.update(
        {
            "gpu_name": props.name,
            "compute_capability": capability,
            "total_vram_bytes": total_bytes,
            "free_vram_bytes": free_bytes,
            "usable_vram_bytes": usable,
            "attention_backend": profile.attention_backend,
            "query_chunk_size": profile.query_chunk_size,
            "training_max_tokens": profile.training_max_tokens,
            "training_max_token_pairs": profile.training_max_token_pairs,
            "inference_max_tokens": profile.inference_max_tokens,
            "inference_max_token_pairs": profile.inference_max_token_pairs,
        }
    )
    return metadata


def _rank_local_config(cfg: RootConfig, rank: int, world: int) -> RootConfig:
    """Return the common per-GPU width with a collision-free rank address."""

    local = copy.deepcopy(cfg)
    if cfg.training.mode == "branched_rao_blackwell":
        local.rollout.roots_per_gpu = 1
        local.rollout.trajectory_rank_counts = tuple(1 for _ in range(world))
        return local
    quotas = tuple(cfg.rollout.roots_per_gpu for _ in range(world))
    local.rollout.roots_per_gpu = cfg.rollout.roots_per_gpu
    # These attributes are execution metadata, not accepted YAML keys.
    local.rollout.trajectory_rank_counts = quotas
    return local


def _alias_table(
    probability: Tensor, *, metrics: Any | None = None
) -> tuple[Tensor, Tensor]:
    """Construct a device-resident Walker alias table for one stratum law."""

    if probability.ndim != 1 or probability.numel() == 0:
        raise ValueError("alias probability must be a nonempty vector")
    if not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()):
        raise ValueError("alias probabilities must be finite and nonnegative")
    probability = probability / probability.sum()
    count = probability.numel()
    scaled_tensor = probability * count
    scaled = scaled_tensor.tolist()
    if metrics is not None and probability.device.type == "cuda":
        metrics.record_transfer("d2h", nbytes=scaled_tensor.numel() * 8)
    threshold_values = [1.0] * count
    alias_values = list(range(count))
    small = [index for index, value in enumerate(scaled) if value < 1.0]
    large = [index for index, value in enumerate(scaled) if value >= 1.0]
    while small and large:
        low = small.pop()
        high = large.pop()
        threshold_values[low] = scaled[low]
        alias_values[low] = high
        scaled[high] -= 1.0 - scaled[low]
        (small if scaled[high] < 1.0 else large).append(high)
    threshold = torch.tensor(
        threshold_values, dtype=torch.float64, device=probability.device
    )
    alias = torch.tensor(alias_values, dtype=torch.int64, device=probability.device)
    if metrics is not None and probability.device.type == "cuda":
        metrics.record_transfer(
            "h2d",
            nbytes=(
                threshold.numel() * threshold.element_size()
                + alias.numel() * alias.element_size()
            ),
        )
    return threshold, alias


def _sample_alias(
    threshold: Tensor,
    alias: Tensor,
    count: int,
    gen: torch.Generator,
) -> Tensor:
    """Draw iid alias-table indices without host-dependent scheduling state."""

    columns = torch.randint(
        threshold.numel(), (count,), device=threshold.device, generator=gen
    )
    uniforms = torch.rand(count, dtype=torch.float64, device=threshold.device, generator=gen)
    return torch.where(uniforms < threshold[columns], columns, alias[columns])


def _decision_mass(data: _IterationData, device: torch.device) -> Tensor:
    """Return local fixed-rollout decision mass before any normalization."""

    if data.normal:
        return torch.tensor(float(len(data.dataset)), dtype=torch.float64, device=device)
    if data.decision_weight is None:
        raise ValueError("RB iteration data is missing decision weights")
    return data.decision_weight.to(device=device, dtype=torch.float64).sum()


def _global_decision_mass(
    local_mass: Tensor,
    *,
    metrics: Any | None = None,
) -> Tensor:
    """SUM decision mass across ranks, preserving the global empirical law."""

    if local_mass.numel() != 1 or local_mass.dtype != torch.float64:
        raise ValueError("local decision mass must be one float64 scalar")
    total = local_mass.clone()
    if dist.is_available() and dist.is_initialized():
        started = time.perf_counter()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if metrics is not None:
            metrics.record_collective(
                nbytes=total.numel() * total.element_size(),
                seconds=time.perf_counter() - started,
            )
    if not bool(total > 0):
        raise ValueError("global decision mass must be positive")
    return total


def _begin_global_decision_mass(
    local_mass: Tensor,
) -> tuple[Tensor, Any, float | None]:
    """Launch the global Z SUM so target construction can overlap it."""

    if local_mass.numel() != 1 or local_mass.dtype != torch.float64:
        raise ValueError("local decision mass must be one float64 scalar")
    total = local_mass.clone()
    if dist.is_available() and dist.is_initialized():
        started = time.perf_counter()
        work = dist.all_reduce(total, op=dist.ReduceOp.SUM, async_op=True)
        return total, work, started
    return total, None, None


def _await_global_decision_mass(
    data: _IterationData, *, metrics: Any | None = None
) -> tuple[Tensor, Tensor]:
    """Finish an overlapped Z SUM and return ``(local_Z, global_Z)``."""

    local = data.local_decision_mass
    total = data.global_mass_buffer
    if local is None or total is None:
        raise ValueError("iteration data is missing its decision-mass reduction")
    if data.global_mass_work is not None:
        wait_started = time.perf_counter()
        data.global_mass_work.wait()
        finished = time.perf_counter()
        if metrics is not None:
            metrics.record_collective(
                nbytes=total.numel() * total.element_size(),
                seconds=finished - float(data.global_mass_started),
                exposed_seconds=finished - wait_started,
            )
    if not bool(total > 0):
        raise ValueError("global decision mass must be positive")
    return local, total


def _stratified_decision_rows(
    data: _IterationData,
    bundle: GameBundle,
    model: nn.Module,
    logical_budget: int,
    global_mass: Tensor,
    gen: torch.Generator,
    variance_ema: dict[str, float] | None = None,
) -> list[tuple[str, Tensor, Tensor]]:
    """Sample rank/family strata and return ``(family, rows, alpha)``."""

    if isinstance(logical_budget, bool) or logical_budget <= 0:
        raise ValueError("logical learner budget must be positive")
    if data.normal:
        state_refs = data.dataset.state_refs
        weights = torch.ones(
            state_refs.numel(), dtype=torch.float64, device=state_refs.device
        )
    else:
        state_refs = data.decision_reach.state_refs
        if data.decision_weight is None:
            raise ValueError("RB iteration data is missing decision weights")
        weights = data.decision_weight.to(
            device=state_refs.device, dtype=torch.float64
        )
    if state_refs.numel() == 0:
        return []
    backend = bundle.rollout_backend
    family_ids = backend.rollout_family_ids(
        state_refs.to(backend.device)
    ).to(state_refs.device)
    names = tuple(getattr(backend, "execution_families", ("default",)))
    values = tuple(getattr(backend, "execution_family_values", range(len(names))))
    strata: list[tuple[str, Tensor, Tensor, float]] = []
    for name, value in zip(names, values, strict=True):
        rows = (family_ids == int(value)).nonzero(as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            continue
        mass = weights[rows].sum()
        if not bool(mass > 0):
            continue
        maximum_width = getattr(backend, "_family_maximum_width", None)
        cost = (
            float(maximum_width(model, name)) ** 2
            if callable(maximum_width)
            else 1.0
        )
        # The lagged EMA is a cheap backend-independent trace-gradient-energy
        # proxy. It affects logical work allocation only, never the objective.
        variance = 1.0 if variance_ema is None else variance_ema.get(name, 1.0)
        sigma = math.sqrt(max(variance, 1.0e-12))
        score = float(mass) * sigma / math.sqrt(max(cost, 1.0))
        strata.append((name, rows, mass, score))
    if not strata:
        return []
    if logical_budget < len(strata):
        raise ValueError(
            "training.minibatch_size must cover every nonempty execution family"
        )
    remaining = logical_budget - len(strata)
    total_score = sum(item[3] for item in strata)
    raw = [remaining * item[3] / total_score for item in strata]
    extra = [math.floor(value) for value in raw]
    leftover = remaining - sum(extra)
    order = sorted(
        range(len(strata)),
        key=lambda index: (raw[index] - extra[index], -index),
        reverse=True,
    )
    for index in order[:leftover]:
        extra[index] += 1
    result: list[tuple[str, Tensor, Tensor]] = []
    for (name, rows, mass, _), addition in zip(strata, extra, strict=True):
        count = 1 + addition
        probability = weights[rows] / mass
        threshold, alias = _alias_table(
            probability, metrics=getattr(backend, "execution_metrics", None)
        )
        sampled = rows[_sample_alias(threshold, alias, count, gen)]
        alpha = stratum_objective_scale(
            mass, global_mass.to(mass.device)
        ).to(torch.float32)
        result.append((name, sampled, alpha))
    return result


def _add_distributed_metric_summaries(
    metrics: dict[str, float], device: torch.device, world: int
) -> None:
    """Attach cross-rank means and maxima for launch-critical diagnostics."""

    if world <= 1:
        return
    names = (
        "nodes",
        "leaves",
        "decisions",
        "decision_nodes_preflop",
        "decision_nodes_flop",
        "decision_nodes_turn",
        "decision_nodes_river",
        "loss",
        "policy",
        "value",
        "entropy",
        "reference_kl",
        "grad_norm_pre_clip",
        "grad_clip_scale",
        "completion_mean_estimator_stddev",
        "completion_max_estimator_stddev",
        "trajectories",
        "completion_samples",
        "minibatches",
        "optimizer_steps",
    )
    present = torch.tensor(
        [name in metrics for name in names], dtype=torch.float64, device=device
    )
    local = torch.tensor(
        [metrics.get(name, 0.0) for name in names],
        dtype=torch.float64,
        device=device,
    )
    summed = torch.cat((local, present))
    maximum = torch.where(
        present.bool(), local, torch.full_like(local, -torch.inf)
    )
    summary_sync_start = time.perf_counter()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    metrics["diagnostic_metric_summary_sync_seconds"] = (
        time.perf_counter() - summary_sync_start
    )
    metrics["diagnostic_metric_summary_sync_bytes"] = float(
        (summed.numel() + maximum.numel()) * summed.element_size()
    )
    total, count = summed.chunk(2)
    for index, name in enumerate(names):
        if not bool(count[index]):
            continue
        metrics[f"{name}_rank_mean"] = float(total[index] / count[index])
        metrics[f"{name}_rank_max"] = float(maximum[index])
        metrics[f"{name}_rank_sum"] = float(total[index])


def _synchronize_memory_control_metrics(
    metrics: dict[str, Any], device: torch.device, world: int, n_players: int
) -> None:
    """Replace controller inputs with rank maxima before online fitting."""

    if world <= 1:
        return
    keys = [
        "nodes",
        "leaves",
        "decisions",
        "retained_target_bytes",
        "iteration_peak_allocated_bytes",
        "iteration_peak_reserved_bytes",
        "iteration_end_allocated_bytes",
        "iteration_end_reserved_bytes",
        "memory_iteration_start_allocated_bytes",
        "memory_iteration_start_reserved_bytes",
        "memory_iteration_start_peak_allocated_bytes",
        "memory_iteration_start_peak_reserved_bytes",
        "memory_preoptimization_allocated_bytes",
        "memory_preoptimization_reserved_bytes",
        "memory_preoptimization_peak_allocated_bytes",
        "memory_preoptimization_peak_reserved_bytes",
        "execution_padded_tokens",
        "execution_padded_token_pairs",
        "execution_padded_tokens_max",
        "execution_padded_token_pairs_max",
        "execution_queue_depth_max",
        "execution_queue_oldest_age_max",
        "execution_forced_drain_rows",
        "execution_forced_drains",
        "execution_model_calls",
        "execution_completion_batches",
        "execution_completion_rows",
        "trajectory_length_mean",
        "trajectory_length_max",
        "trajectory_length_p50",
        "trajectory_length_p90",
        "trajectory_length_p99",
        "trajectory_length_le_4",
        "trajectory_length_le_8",
        "trajectory_length_le_16",
        "trajectory_length_le_32",
        "trajectory_length_le_64",
    ]
    for phase in ("snapshot", "rollout", "completion", "backward", "compaction", "optimization"):
        keys.extend(
            (
                f"phase_{phase}_peak_allocated_bytes",
                f"phase_{phase}_peak_reserved_bytes",
            )
        )
    for phase in ("inference", "training"):
        for family in ("pre", *(f"post_{live}" for live in range(2, n_players + 1))):
            base = f"execution_{phase}_{family}"
            keys.extend(
                (
                    f"{base}_model_calls",
                    f"{base}_model_rows",
                    f"{base}_model_rows_max",
                    f"{base}_padded_tokens",
                    f"{base}_padded_tokens_max",
                    f"{base}_padded_token_pairs",
                    f"{base}_padded_token_pairs_max",
                )
            )
    values = torch.tensor(
        [float(metrics.get(key, 0.0)) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    for key, value in zip(keys, values.tolist(), strict=True):
        if key in metrics or value != 0.0:
            metrics[f"{key}_rank_max"] = float(value)
            # Online fitting must be byte-identical on every rank.
            metrics[key] = float(value)


@dataclass
class _IterationData:
    """Objects intentionally retained across rollout and optimization phases."""

    normal: bool
    completion_quality: dict[str, float]
    dataset: Any = None
    tree: Any = None
    targets: Any = None
    policy_cache: Any = None
    decision_reach: Any = None
    decision_weight: Tensor | None = None
    local_decision_mass: Tensor | None = None
    global_mass_buffer: Tensor | None = None
    global_mass_work: Any = None
    global_mass_started: float | None = None
    profile_metrics: dict[str, float] | None = None


def _value_workspace_budget(cfg: RootConfig, device: torch.device) -> int:
    """Return an explicit byte budget for completion and backward reduction."""

    gib = 1024**3
    if device.type != "cuda":
        return gib
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    profile = cfg.hardware
    if profile is None:
        budget = free_bytes // 2
    else:
        reserve = int(profile.reserve_vram_gib * gib)
        budget = min(
            free_bytes - reserve,
            int(total_bytes * profile.max_memory_fraction),
        )
    if budget <= 0:
        raise MemoryError("no device memory remains for completion/backward workspace")
    return int(budget)


def _completion_batch_rows(
    bundle: GameBundle, leaf_count: int, workspace_bytes: int
) -> int:
    """Choose a conservative completion microbatch from a byte budget."""

    if leaf_count <= 0:
        return 1
    p_max = int(bundle.rollout_backend.n_players)
    x_max = max(int(size) for size in bundle.rollout_backend.x_sizes)
    per_leaf = p_max * x_max * (3 * 4 + 1)
    byte_limited = max(1, (workspace_bytes // 8) // max(per_leaf, 1))
    return min(leaf_count, 256, int(byte_limited))


def _merge_completion_quality(
    parts: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """Merge scalar quality summaries emitted by completion microbatches."""

    if not parts:
        return {}
    result: dict[str, float] = {}
    keys = {key for _, values in parts for key in values}
    for key in keys:
        present = [(rows, values[key]) for rows, values in parts if key in values]
        if key.startswith("completion_min_"):
            result[key] = min(value for _, value in present)
        elif key.startswith("completion_max_"):
            result[key] = max(value for _, value in present)
        elif key == "completion_invalid_posterior_mass":
            result[key] = sum(value for _, value in present)
        else:
            denominator = sum(rows for rows, _ in present)
            result[key] = sum(rows * value for rows, value in present) / max(
                denominator, 1
            )
    return result


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Return unique tensor storage bytes reachable from a compact data object."""

    if seen is None:
        seen = set()
    if isinstance(value, Tensor):
        storage = value.untyped_storage()
        identity = storage.data_ptr()
        if identity in seen:
            return 0
        seen.add(identity)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item, seen) for item in value)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return sum(_tensor_bytes(getattr(value, name), seen) for name in fields)
    return 0


def _analytical_memory_plan(
    cfg: RootConfig, bundle: GameBundle, *, has_reference: bool
) -> dict[str, int]:
    """Return a mode-aware bound for allocations that can coexist by phase.

    Sequential trajectories retain exactly one realized child at each player
    node. Charging the backend's entire legal event vocabulary to every node
    produces a bound for tensors this execution mode never allocates. Capacity
    remains the conservative decision-row count, but sequential policy and
    target widths are one. The hardware reserve is intentionally left outside
    this structural plan for attention/autograd workspaces and allocator slack.
    """

    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    parameter = sum(
        value.numel() * value.element_size() for value in model.parameters()
    )
    opponent_parameter = (
        0
        if bundle.opponent_model is None
        else sum(
            value.numel() * value.element_size()
            for value in bundle.opponent_model.parameters()
        )
    )
    backend = bundle.rollout_backend
    players = int(backend.n_players)
    types = max(int(value) for value in backend.x_sizes)
    events = int(getattr(backend, "e_max", 1))
    resident_trees = int(cfg.rollout.roots_per_gpu)
    nodes = resident_trees * int(cfg.rollout.node_budget)
    fp32 = 4
    int32 = 4
    sequential = cfg.training.mode != "branched_rao_blackwell"
    retained_events = 1 if sequential else events
    active_rows = min(
        resident_trees,
        max(
            int(cfg.rollout.family_batch_rows or 1),
            int(cfg.rollout.max_active_trajectories or 1),
        ),
    )
    topology = nodes * (
        int32 * (6 + players) + fp32 * 2
    )
    dense_reach = nodes * players * types * fp32
    reach_row = players * types * fp32
    if sequential:
        # Root reach and the overwriteable per-trajectory working reach coexist.
        rollout_reach = 2 * resident_trees * reach_row
        # Current/child/gather/scatter waves are bounded by scheduler admission,
        # not by the total node capacity accumulated across all trajectories.
        rollout_transient_reach = 4 * active_rows * reach_row
    else:
        rollout_reach = dense_reach
        rollout_transient_reach = 4 * dense_reach

    # Sequential poker currently requests full [P,H] decision reaches. Include
    # the node/state/owner indices retained by DecisionReachCache as well.
    decision_reach = dense_reach + nodes * 3 * int32
    policy_cache = nodes * (
        (retained_events * types + types) * fp32 + int32
    )
    rb_targets = (
        0
        if cfg.training.mode == "normal"
        else nodes
        * (
            (2 + 2 * retained_events) * types * fp32
            + int32  # node id
            + retained_events * (int32 + 1)  # child event and mask
            + int32  # child count
        )
    )
    # Once single-child target validation completes, child_p_hat aliases
    # parent_p_hat. Target construction still needs the larger rb_targets term.
    compact_rb_targets = (
        0
        if cfg.training.mode == "normal"
        else nodes
        * (
            (2 + retained_events) * types * fp32
            + int32
            + retained_events * (int32 + 1)
            + int32
        )
    )
    normal_targets = (
        nodes * (5 * fp32 + int32)
        if cfg.training.mode == "normal"
        else 0
    )
    optimizer_and_grad = 3 * parameter  # grad + two fp32 Adam moments
    gradient_sync = parameter  # explicit flat SUM buffer; this code does not use DDP buckets
    # Rollout is synchronous with optimization and reads the live model under
    # no_grad/eval; there is no second full behavior-policy allocation.
    rollout_snapshot = 0
    reference = parameter if has_reference else 0
    alias = nodes * (8 + 8 + int32) if sequential else 0
    resident_seen: set[int] = set()
    backend_resident = _tensor_bytes(vars(backend), resident_seen)
    for owned in (getattr(backend, "states", None),):
        if owned is not None:
            backend_resident += _tensor_bytes(vars(owned), resident_seen)
    partition_bound_fn = getattr(backend, "partition_cache_memory_bound", None)
    if callable(partition_bound_fn):
        registered_cache_bytes = 0
        for service in getattr(backend, "_partition_services", {}).values():
            cache = service[1]
            registered_cache_bytes += _tensor_bytes(vars(cache), resident_seen)
        backend_resident += registered_cache_bytes
        backend_resident += max(
            int(partition_bound_fn()) - registered_cache_bytes,
            0,
        )
    else:
        cache = getattr(backend, "partition_summary_cache", None)
        if cache is not None:
            backend_resident += _tensor_bytes(vars(cache), resident_seen)
    completion_workspace = (
        8 * resident_trees * reach_row
        + min(resident_trees, 256) * players * types * (3 * fp32 + 1)
        if cfg.training.mode != "normal"
        else 0
    )
    rollout_fixed = (
        parameter
        + opponent_parameter
        + rollout_snapshot
        + reference
        + backend_resident
    )
    rollout_variable = (
        topology
        + rollout_reach
        + rollout_transient_reach
        + policy_cache
    )
    rollout_phase = rollout_fixed + rollout_variable
    target_fixed = (
        parameter
        + opponent_parameter
        + optimizer_and_grad
        + reference
        + backend_resident
    )
    target_variable = (
        topology
        + rollout_reach
        + policy_cache
        + rb_targets
        + normal_targets
        + completion_workspace
    )
    target_phase = target_fixed + target_variable
    optimization_fixed = (
        parameter
        + opponent_parameter
        + optimizer_and_grad
        + gradient_sync
        + reference
        + backend_resident
    )
    optimization_variable = (
        decision_reach
        + policy_cache
        + compact_rb_targets
        + normal_targets
        + alias
    )
    optimization_phase = optimization_fixed + optimization_variable
    seal_fixed = max(rollout_fixed, target_fixed)
    seal_variable = max(rollout_variable, target_variable)
    return {
        "memory_plan_parameter_bytes": parameter,
        "memory_plan_opponent_parameter_bytes": opponent_parameter,
        "memory_plan_backend_resident_bytes": backend_resident,
        "memory_plan_topology_bytes": topology,
        "memory_plan_rollout_reach_bytes": rollout_reach,
        "memory_plan_transient_reach_bytes": rollout_transient_reach,
        "memory_plan_decision_reach_bytes": decision_reach,
        "memory_plan_policy_cache_bytes": policy_cache,
        "memory_plan_target_bytes": rb_targets + normal_targets,
        "memory_plan_compact_target_bytes": compact_rb_targets + normal_targets,
        "memory_plan_alias_bytes": alias,
        "memory_plan_completion_workspace_bytes": completion_workspace,
        "memory_plan_optimizer_and_grad_bytes": optimizer_and_grad,
        "memory_plan_gradient_sync_bytes": gradient_sync,
        "memory_plan_snapshot_bytes": rollout_snapshot,
        "memory_plan_reference_bytes": reference,
        "memory_plan_rollout_phase_bytes": rollout_phase,
        "memory_plan_target_phase_bytes": target_phase,
        "memory_plan_optimization_phase_bytes": optimization_phase,
        "memory_plan_rollout_fixed_bytes": rollout_fixed,
        "memory_plan_rollout_variable_bytes": rollout_variable,
        "memory_plan_seal_fixed_bytes": seal_fixed,
        "memory_plan_seal_variable_bytes": seal_variable,
        "memory_plan_target_fixed_bytes": target_fixed,
        "memory_plan_target_variable_bytes": target_variable,
        "memory_plan_optimization_fixed_bytes": optimization_fixed,
        "memory_plan_optimization_variable_bytes": optimization_variable,
        "memory_plan_structural_peak_bytes": max(
            rollout_phase, target_phase, optimization_phase
        ),
    }


def _memory_limit(
    cfg: RootConfig, device: torch.device, *, account_current_free: bool = True
) -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if cfg.hardware is None:
        absolute = int(total_bytes * 0.85)
        return min(int(free_bytes * 0.85), absolute) if account_current_free else absolute
    reserve = int(cfg.hardware.reserve_vram_gib * 1024**3)
    absolute = min(
        total_bytes - reserve, int(total_bytes * cfg.hardware.max_memory_fraction)
    )
    return min(free_bytes - reserve, absolute) if account_current_free else absolute


def _memory_operating_ceiling(
    cfg: RootConfig, device: torch.device, current_reserved_bytes: int
) -> int:
    """Return the absolute PyTorch-reservation ceiling under current driver use.

    The configured reserve is the bootstrap safety artifact.  Unlike a static
    percentage check, this ceiling contracts immediately when another context
    consumes driver memory: only current PyTorch reservation plus genuinely
    free bytes above the bootstrap guard are admissible.
    """

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    absolute = _memory_limit(cfg, device, account_current_free=False)
    driver_guard = max(0, total_bytes - absolute)
    externally_feasible = current_reserved_bytes + max(
        0, free_bytes - driver_guard
    )
    return min(absolute, externally_feasible)


def _validate_memory_preflight(
    cfg: RootConfig, device: torch.device, plan: dict[str, int]
) -> None:
    """Reject a structurally impossible CUDA plan before rollout allocation."""

    if device.type != "cuda":
        return
    # ``plan`` describes the complete phase peak, including model/backend
    # tensors already allocated when this check runs. Comparing it with current
    # free memory would double-count those residents.
    limit = _memory_limit(cfg, device, account_current_free=False)
    structural = plan["memory_plan_structural_peak_bytes"]
    if limit <= 0 or structural > limit:
        raise MemoryError(
            "analytical CUDA memory plan exceeds the configured safe limit: "
            f"structural_peak={structural}, safe_limit={limit}"
        )


def _validate_memory_calibration(
    cfg: RootConfig, device: torch.device, metrics: dict[str, float]
) -> None:
    """Gate later iterations using the measured byte-complete CUDA peak."""

    if device.type != "cuda":
        return
    peaks = [
        value
        for key, value in metrics.items()
        if (
            key == "iteration_peak_reserved_bytes"
            or (key.startswith("phase_") and key.endswith("_peak_reserved_bytes"))
        )
    ]
    if not peaks:
        raise RuntimeError("first CUDA iteration did not emit memory calibration peaks")
    measured = int(max(peaks))
    metrics["calibrated_peak_reserved_bytes"] = float(measured)
    safe_limit = _memory_limit(cfg, device, account_current_free=False)
    metrics["calibrated_safe_limit_bytes"] = float(safe_limit)
    headroom = safe_limit - measured
    metrics["calibrated_actual_headroom_bytes"] = float(headroom)
    metrics["memory_hard_limit_exceeded"] = float(headroom < 0)


def _profiled_call(
    metrics: dict[str, float],
    name: str,
    device: torch.device,
    enabled: bool,
    function: Any,
) -> Any:
    """Run one benchmark phase with accurate optional CUDA boundaries."""

    if not enabled:
        return function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics[f"phase_{name}_seconds"] = time.perf_counter() - start
    if device.type == "cuda":
        metrics[f"phase_{name}_peak_allocated_bytes"] = float(
            torch.cuda.max_memory_allocated(device)
        )
        metrics[f"phase_{name}_peak_reserved_bytes"] = float(
            torch.cuda.max_memory_reserved(device)
        )
    return result


def _decision_street_metrics(
    profile_metrics: dict[str, float],
    backend: Any,
    tree: Any,
    node_ids: Tensor,
) -> None:
    """Record decision-node counts by poker street for one sampled rollout.

    Street labels live in the backend's device-resident state store. This is
    called only on the diagnostic path, so the small count reduction is copied
    to the host once per iteration rather than synchronizing every decision.
    """

    states = getattr(backend, "states", None)
    street = getattr(states, "street", None)
    if street is None or node_ids.numel() == 0:
        return
    refs = tree.state_ref[node_ids.to(torch.int64)].to(torch.int64)
    labels = street[refs].to(torch.int64).clamp(0, 3)
    counts = torch.bincount(labels, minlength=4).detach().cpu().tolist()
    names = ("preflop", "flop", "turn", "river")
    total = max(sum(int(value) for value in counts), 1)
    for name, count in zip(names, counts, strict=True):
        profile_metrics[f"decision_nodes_{name}"] = float(count)
        profile_metrics[f"decision_fraction_{name}"] = float(count) / total


def _trajectory_length_metrics(
    profile_metrics: dict[str, float], tree: Any, starts: int
) -> None:
    """Record policy-conditioned realized trajectory-length diagnostics."""

    if starts <= 0 or tree.tree_id.numel() == 0:
        return
    lengths = torch.bincount(tree.tree_id.to(torch.int64), minlength=starts).float()
    ordered = lengths.sort().values
    count = ordered.numel()
    profile_metrics["trajectory_length_mean"] = float(lengths.mean())
    profile_metrics["trajectory_length_max"] = float(lengths.max())
    for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        index = min(count - 1, max(0, math.ceil(count * quantile) - 1))
        profile_metrics[f"trajectory_length_{name}"] = float(ordered[index])
    for upper in (4, 8, 16, 32, 64):
        profile_metrics[f"trajectory_length_le_{upper}"] = float(
            (lengths <= upper).sum()
        )


def _build_iteration_data(
    cfg: RootConfig,
    bundle: GameBundle,
    torch_gen: torch.Generator,
    spec: LossSpec,
    *,
    profile: bool = False,
    frozen_model: nn.Module | None = None,
    iteration: int = 0,
) -> _IterationData:
    """Build compact training data in a frame that owns all frozen transients.

    Rollout and target construction are synchronous, so the live model is put
    in evaluation mode and remains immutable until this function returns.  A
    second full policy copy is deliberately not allocated.
    """

    source = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = next(source.parameters()).device
    amp_enabled, amp_dtype = _precision(cfg, device)
    execution_metrics = getattr(bundle.rollout_backend, "execution_metrics", None)
    profile_metrics: dict[str, float] = {}
    if frozen_model is None:
        frozen_model = _profiled_call(
            profile_metrics,
            "snapshot",
            device,
            profile,
            source.eval,
        )
    frozen_bundle = GameBundle(
        model=frozen_model,
        rollout_backend=bundle.rollout_backend,
        policy_adapter=bundle.policy_adapter,
        completion_backend=bundle.completion_backend,
        opponent_model=bundle.opponent_model,
        probe_target=bundle.probe_target,
    )
    normal = cfg.training.mode == "normal"

    def sample_normal_rollout() -> Any:
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            return sample_trees_sequential(
                frozen_bundle,
                cfg,
                torch_gen,
                return_profiles=True,
                iteration=iteration,
            )

    def sample_rb_rollout() -> Any:
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            return sample_trees(
                frozen_bundle, cfg, torch_gen, iteration=iteration
            )

    with torch.no_grad():
        if normal:
            sampled = _profiled_call(
                profile_metrics,
                "rollout",
                device,
                profile,
                sample_normal_rollout,
            )
            sampled_tree, _, sampled_policy, _ = sampled
            local_mass = torch.tensor(
                float(sampled_policy.node_ids.numel()),
                dtype=torch.float64,
                device=device,
            )
            global_mass, global_work, global_started = _begin_global_decision_mass(
                local_mass
            )
            dataset = _profiled_call(
                profile_metrics,
                "backward",
                device,
                profile,
                lambda: build_normal_dataset_from_sampled(
                    frozen_bundle,
                    cfg,
                    torch_gen,
                    sampled,
                    iteration=iteration,
                ),
            )
            if execution_metrics is not None:
                execution_metrics.record_target_bytes(_tensor_bytes(dataset))
            if len(dataset) != int(sampled_policy.node_ids.numel()):
                raise ValueError("normal target table lost or duplicated decision rows")
            if profile:
                tree, policy_cache = sampled_tree, sampled_policy
                leaves = int((tree.child_ptr[1:] == tree.child_ptr[:-1]).sum())
                decisions = len(dataset)
                offspring = tree.child_ptr[1:] - tree.child_ptr[:-1]
                profile_metrics.update(
                    {
                        "trajectories": float(cfg.rollout.roots_per_gpu),
                        "nodes": float(tree.owner.numel()),
                        "leaves": float(leaves),
                        "decisions": float(decisions),
                        "terminal_samples": float(leaves),
                        "retained_target_bytes": float(_tensor_bytes(dataset)),
                        "policy_decisions": float(policy_cache.node_ids.numel()),
                        "offspring_mean": float(offspring.float().mean()),
                        "offspring_max": float(offspring.max()),
                    }
                )
                _decision_street_metrics(
                    profile_metrics,
                    bundle.rollout_backend,
                    tree,
                    policy_cache.node_ids,
                )
                _trajectory_length_metrics(
                    profile_metrics, tree, int(cfg.rollout.roots_per_gpu)
                )
            return _IterationData(
                normal=True,
                completion_quality={},
                dataset=dataset,
                local_decision_mass=local_mass,
                global_mass_buffer=global_mass,
                global_mass_work=global_work,
                global_mass_started=global_started,
                profile_metrics=profile_metrics,
            )

        rb_sample = _profiled_call(
            profile_metrics,
            "rollout",
            device,
            profile,
            sample_rb_rollout,
        )
        probe_rollout = rb_sample if isinstance(rb_sample, ProbeRollout) else None
        if probe_rollout is None:
            tree, reach_cache, policy_cache = rb_sample
            probe_seats = None
            prebuilt_decision_reach = None
        else:
            tree = probe_rollout.tree
            reach_cache = probe_rollout.leaf_reach
            policy_cache = probe_rollout.policy
            probe_seats = probe_rollout.seats
            prebuilt_decision_reach = probe_rollout.decision_reach
        decision_weight = tree.path_weight[
            policy_cache.node_ids.to(tree.path_weight.device, dtype=torch.int64)
        ].detach().to(torch.float64)
        local_mass = decision_weight.sum()
        global_mass, global_work, global_started = _begin_global_decision_mass(
            local_mass
        )
        # The rollout view has no consumer in completion or target reduction.
        del frozen_bundle
        leaf_ids = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero(
            as_tuple=False
        ).squeeze(1)
        if bundle.completion_backend is None or cfg.completion.n_samples is None:
            raise RuntimeError("RB modes require a completion backend and sample budget")
        heads = HeadEvals(node_ids=policy_cache.node_ids, v=policy_cache.old_v)
        workspace_bytes = _value_workspace_budget(cfg, device)
        completion_rows = _completion_batch_rows(
            bundle, int(leaf_ids.numel()), workspace_bytes
        )
        keyed_completion = getattr(bundle.completion_backend, "complete_keyed", None)
        selected_keyed_completion = getattr(
            bundle.completion_backend, "complete_selected_keyed", None
        )
        completion_keys = None
        if callable(keyed_completion) and cfg.training.mode == "rao_blackwell":
            completion_keys = sequential_leaf_logical_rng_keys(
                tree,
                cfg,
                torch_gen,
                leaf_ids,
                iteration=iteration,
                purpose=DrawPurpose.COMPLETION,
            )
        quality_parts: list[tuple[int, dict[str, float]]] = []

        def completion_batches():
            for first in range(0, int(leaf_ids.numel()), completion_rows):
                selected = leaf_ids[first : first + completion_rows].to(torch.int32)
                leaf_reach = reach_cache.rows_for(
                    tree,
                    selected,
                    bundle.rollout_backend,
                    policy_cache,
                )

                def complete_selected():
                    if completion_keys is not None:
                        if probe_seats is not None:
                            if not callable(selected_keyed_completion):
                                raise TypeError(
                                    "probe completion backend lacks complete_selected_keyed"
                                )
                            selected_trees = tree.tree_id[
                                selected.to(tree.tree_id.device, dtype=torch.int64)
                            ].to(torch.int64)
                            return selected_keyed_completion(
                                bundle.rollout_backend,
                                tree,
                                selected,
                                leaf_reach,
                                cfg.completion.n_samples,
                                completion_keys[first : first + selected.numel()],
                                probe_seats[selected_trees],
                            )
                        return keyed_completion(
                            bundle.rollout_backend,
                            tree,
                            selected,
                            leaf_reach,
                            cfg.completion.n_samples,
                            completion_keys[first : first + selected.numel()],
                        )
                    return bundle.completion_backend.complete(
                        bundle.rollout_backend,
                        tree,
                        selected,
                        leaf_reach,
                        cfg.completion.n_samples,
                        torch_gen,
                    )

                batch_profile: dict[str, float] = {}
                completion_result = _profiled_call(
                    batch_profile,
                    "completion",
                    device,
                    profile,
                    complete_selected,
                )
                if execution_metrics is not None:
                    execution_metrics.record_completion(rows=int(selected.numel()))
                for key, value in batch_profile.items():
                    if key.endswith("_seconds"):
                        profile_metrics[key] = profile_metrics.get(key, 0.0) + value
                    else:
                        profile_metrics[key] = max(
                            profile_metrics.get(key, 0.0), value
                        )
                completion = completion_result.for_learning()
                quality_parts.append(
                    (
                        int(selected.numel()),
                        {
                            key: float(value)
                            for key, value in completion.metadata.items()
                            if key.startswith("completion_")
                            and isinstance(value, (int, float))
                        },
                    )
                )
                yield selected, completion
                del completion, completion_result, leaf_reach

        targets = _profiled_call(
            profile_metrics,
            "backward",
            device,
            profile,
            lambda: streamed_branched_value(
                tree,
                leaf_ids.to(torch.int32),
                completion_batches(),
                reach_cache,
                heads,
                cfg.loss.lam,
                bundle.rollout_backend,
                configured_node_budget=cfg.rollout.node_budget,
                completion_batch_rows=completion_rows,
                emit_q=False,
                max_workspace_bytes=workspace_bytes,
                decision_players_by_tree=probe_seats,
            ),
        )
        if execution_metrics is not None:
            execution_metrics.record_target_bytes(_tensor_bytes(targets))
        completion_quality = _merge_completion_quality(quality_parts)
        if not torch.equal(policy_cache.node_ids, targets.node_ids):
            raise ValueError("PolicyCache and NodeTargets node_ids must match exactly")
        unavailable = {
            name for name in spec.required_targets
            if not hasattr(targets, name) or getattr(targets, name) is None
        }
        if unavailable:
            raise ValueError(
                f"loss {cfg.loss.name!r} requires unavailable targets {sorted(unavailable)}"
            )
        decision_reach = _profiled_call(
            profile_metrics,
            "compaction",
            device,
            profile,
            lambda: (
                prebuilt_decision_reach
                if prebuilt_decision_reach is not None
                else policy_cache.decision_reach
            ),
        )
        if decision_reach is None:
            raise RuntimeError("rollout did not retain natural decision inputs")
        decisions = int(policy_cache.node_ids.numel())
        offspring = tree.child_ptr[1:] - tree.child_ptr[:-1]
        profile_metrics.update(
            {
                (
                    "sampled_trees"
                    if cfg.training.mode == "branched_rao_blackwell"
                    else "trajectories"
                ): float(cfg.rollout.roots_per_gpu),
                "nodes": float(tree.owner.numel()),
                "leaves": float(leaf_ids.numel()),
                "decisions": float(decisions),
                "completion_samples": float(
                    leaf_ids.numel() * int(cfg.completion.n_samples)
                ),
                "offspring_mean": float(offspring.float().mean()),
                "offspring_max": float(offspring.max()),
            }
        )
        _decision_street_metrics(
            profile_metrics,
            bundle.rollout_backend,
            tree,
            policy_cache.node_ids,
        )
        _trajectory_length_metrics(
            profile_metrics, tree, int(cfg.rollout.roots_per_gpu)
        )
        if profile:
            completion_effective = (
                completion_quality.get("completion_mean_n_eff", 0.0)
                * float(leaf_ids.numel())
            )
            retained = _tensor_bytes((targets, policy_cache, decision_reach, decision_weight))
            profile_metrics.update(
                {
                    "completion_effective_samples": completion_effective,
                    "retained_target_bytes": float(retained),
                }
            )
        return _IterationData(
            normal=False,
            completion_quality=completion_quality,
            targets=targets,
            policy_cache=policy_cache,
            decision_reach=decision_reach,
            decision_weight=decision_weight,
            local_decision_mass=local_mass,
            global_mass_buffer=global_mass,
            global_mass_work=global_work,
            global_mass_started=global_started,
            profile_metrics=profile_metrics,
        )


def _validate_old_policy_identity(
    cfg: RootConfig,
    bundle: GameBundle,
    data: _IterationData,
) -> dict[str, float]:
    """Gate the first update on exact rollout/optimization model inputs.

    The rollout snapshot and the still-unmodified trainable model must agree on
    cached selected-action log probabilities and values.  Every retained
    decision is checked in the backend's bounded physical batches.
    """

    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = next(model.parameters()).device
    amp_enabled, amp_dtype = _precision(cfg, device)
    was_training = model.training
    model.eval()
    if data.normal:
        rows = torch.arange(len(data.dataset), device=device)
        minibatch = gather_normal_minibatch(data.dataset, bundle, rows)
    else:
        rows = torch.arange(data.targets.node_ids.numel(), device=device)
        minibatch = gather_minibatch_rows(
            data.tree,
            data.targets,
            data.policy_cache,
            data.decision_reach,
            bundle,
            rows,
        )
    planner = getattr(bundle.rollout_backend, "plan_policy_batches", None)
    plans = (
        planner(model, minibatch.state_refs, "inference")
        if planner is not None
        else [rows]
    )
    maximum_logp_error = torch.zeros((), device=device)
    maximum_value_error = torch.zeros((), device=device)
    maximum_value_scale = torch.zeros((), device=device)
    ratio_min = torch.full((), torch.inf, device=device)
    ratio_max = torch.zeros((), device=device)
    with torch.no_grad():
        for positions in plans:
            physical = (
                index_normal_minibatch(minibatch, positions)
                if data.normal
                else index_minibatch(minibatch, positions)
            )
            with torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                output = bundle.rollout_backend.policy_forward(
                    model, physical.state_refs, physical.reach
                )
            if data.normal:
                actor = physical.actor_type.to(device=device, dtype=torch.int64)
                batch_rows = torch.arange(actor.numel(), device=device)
                selected = bundle.policy_adapter.log_prob(
                    output,
                    physical.event.to(device)[:, None],
                    physical.legal_event_ids.to(device),
                    physical.legal_event_mask.to(device),
                )[:, 0]
                current_logp = selected[batch_rows, actor].float()
                current_value = output.v[batch_rows, actor].float()
                cached_logp = physical.old_logp.to(current_logp)
                cached_value = physical.old_v.to(current_value)
            else:
                selected = _selected_logp(
                    output, physical, bundle.policy_adapter
                ).float()
                active = (
                    physical.targets.child_mask.to(device)[:, :, None]
                    & physical.valid_type_mask.to(device)[:, None, :]
                )
                current_logp = selected[active]
                cached_logp = physical.old_logp_child.to(selected)[active]
                valid = physical.valid_type_mask.to(device)
                current_value = output.v.float()[valid]
                cached_value = physical.old_v.to(output.v).float()[valid]
            log_ratio = current_logp - cached_logp
            ratios = torch.exp(log_ratio)
            maximum_logp_error = torch.maximum(
                maximum_logp_error, log_ratio.abs().max()
            )
            maximum_value_error = torch.maximum(
                maximum_value_error, (current_value - cached_value).abs().max()
            )
            maximum_value_scale = torch.maximum(
                maximum_value_scale,
                torch.maximum(
                    current_value.abs().max(), cached_value.abs().max()
                ),
            )
            ratio_min = torch.minimum(ratio_min, ratios.min())
            ratio_max = torch.maximum(ratio_max, ratios.max())
    if was_training:
        model.train()
    if dist.is_initialized():
        dist.all_reduce(maximum_logp_error, op=dist.ReduceOp.MAX)
        dist.all_reduce(maximum_value_error, op=dist.ReduceOp.MAX)
        dist.all_reduce(maximum_value_scale, op=dist.ReduceOp.MAX)
        dist.all_reduce(ratio_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(ratio_max, op=dist.ReduceOp.MAX)
    # Rollout queues and this exhaustive validation can partition identical
    # inputs into different physical batch shapes. FP16 attention/reductions
    # are shape-dependent; require identity to within a 2% policy ratio rather
    # than expecting bitwise equality. This remains one tenth of the configured
    # 20% PPO clipping interval for the production run. Full precision remains
    # tight enough to expose input/cache misalignment directly.
    logp_tolerance = 2.0e-2 if amp_enabled else 2.0e-5
    value_tolerance = 1.0e-2 if amp_enabled else 2.0e-5
    value_ok = maximum_value_error <= value_tolerance * (
        1.0 + maximum_value_scale
    )
    logp_ok = maximum_logp_error <= logp_tolerance
    unclipped = (ratio_min >= 1.0 - float(cfg.loss.clip_eps)) & (
        ratio_max <= 1.0 + float(cfg.loss.clip_eps)
    )
    ok = value_ok & logp_ok & unclipped
    if device.type == "cuda":
        # This launch/resume gate is an explicit synchronization boundary: its
        # four diagnostics are returned to the host immediately below.  Raise a
        # normal exception with the measured disagreement instead of poisoning
        # the CUDA context with a device-side assertion.
        torch.cuda.synchronize(device)
    diagnostics = {
        "identity_max_abs_logp_error": float(maximum_logp_error),
        "identity_max_abs_value_error": float(maximum_value_error),
        "identity_ratio_min": float(ratio_min),
        "identity_ratio_max": float(ratio_max),
        "identity_logp_tolerance": float(logp_tolerance),
        "identity_value_tolerance": float(value_tolerance),
    }
    if not bool(ok):
        raise RuntimeError(
            "old/current-policy identity launch gate failed: "
            f"max_abs_logp_error={diagnostics['identity_max_abs_logp_error']:.8g}, "
            f"max_abs_value_error={diagnostics['identity_max_abs_value_error']:.8g}, "
            f"ratio_min={diagnostics['identity_ratio_min']:.8g}, "
            f"ratio_max={diagnostics['identity_ratio_max']:.8g}, "
            f"logp_tolerance={logp_tolerance:.8g}, "
            f"value_tolerance={value_tolerance:.8g}"
        )
    return diagnostics


def _one_iteration(
    cfg: RootConfig,
    bundle: GameBundle,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    torch_gen: torch.Generator,
    iteration: int,
    *,
    profile: bool = False,
    reference_model: nn.Module | None = None,
    validate_identity: bool = False,
) -> dict[str, Any]:
    spec = _loss_spec(cfg)
    execution_metrics = getattr(bundle.rollout_backend, "execution_metrics", None)
    if execution_metrics is not None:
        execution_metrics.clear()
    model = bundle.model
    source = model.module if hasattr(model, "module") else model
    device = next(model.parameters()).device
    memory_ledger = MemoryLedger(device)
    memory_metrics: dict[str, float] = {}
    if device.type == "cuda":
        # Calibration must cover allocator growth and new cache bins on every
        # iteration, not only the initially profiled one.
        torch.cuda.reset_peak_memory_stats(device)
        memory_metrics.update(
            memory_ledger.snapshot(
                "iteration_start",
                owned={
                    "model": source,
                    "optimizer": optimizer.state,
                    "reference": reference_model,
                    "backend": bundle.rollout_backend,
                },
            )
        )
    amp_enabled, amp_dtype = _precision(cfg, device)
    normal = cfg.training.mode == "normal"
    total_trajectories = int(cfg.rollout.roots_per_gpu)
    profile_parts: list[dict[str, float]] = []
    quality_parts: list[tuple[int, dict[str, float]]] = []
    optimization_records: list[tuple[int, int, dict[str, float]]] = []
    optimization_host_seconds: dict[str, float] = {
        "minibatch_sampling": 0.0,
        "forward_loss": 0.0,
        "backward_pass": 0.0,
        "gradient_sync": 0.0,
        "gradient_update": 0.0,
    }
    identity_metrics: dict[str, float] = {}
    decision_total = 0
    data = _build_iteration_data(
        cfg,
        bundle,
        torch_gen,
        spec,
        profile=profile,
        iteration=iteration,
    )
    if profile:
        memory_metrics.update(
            memory_ledger.snapshot(
                "preoptimization",
                owned={
                    "model": source,
                    "optimizer": optimizer.state,
                    "reference": reference_model,
                    "backend": bundle.rollout_backend,
                    "iteration": data,
                },
            )
        )
    if validate_identity:
        if execution_metrics is None:
            identity_metrics = _validate_old_policy_identity(cfg, bundle, data)
        else:
            # The identity gate is launch/resume validation, not scientific
            # rollout or learner work.  Excluding it keeps exact execution
            # counters invariant under checkpoint continuation.
            with execution_metrics.suspended():
                identity_metrics = _validate_old_policy_identity(
                    cfg, bundle, data
                )

    decisions = (
        len(data.dataset)
        if data.normal
        else int(data.targets.node_ids.numel())
    )
    decision_total += decisions
    leaves = int((data.profile_metrics or {}).get("leaves", total_trajectories))
    quality_parts.append((leaves, data.completion_quality))
    profile_parts.append(data.profile_metrics or {})

    optimizer_steps_this_iteration = int(cfg.training.gd_steps_per_iter)

    local_mass, global_mass = _await_global_decision_mass(
        data, metrics=execution_metrics
    )
    rank_objective_scale = stratum_objective_scale(
        local_mass, global_mass
    ).to(torch.float32)
    if data.profile_metrics is not None:
        data.profile_metrics["local_decision_mass"] = float(local_mass)
        data.profile_metrics["global_decision_mass"] = float(global_mass)
        data.profile_metrics["rank_objective_scale"] = float(
            rank_objective_scale
        )

    model.train()
    flat_gradient = FlatGradientBuffer(source)
    source._pbg_direct_flat_gradient_buffer = flat_gradient
    optimization_start = None
    minibatch_count = 0
    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        optimization_start = time.perf_counter()

    if cfg.training.ppo_epochs is None:
        batches: list[tuple[int, Tensor | None]] = [
            (-1, None) for _ in range(optimizer_steps_this_iteration)
        ]
    else:
        batches = []
        for epoch in range(cfg.training.ppo_epochs):
            permutation = torch.randperm(decisions, generator=torch_gen, device=device)
            for first in range(0, decisions, cfg.training.minibatch_size):
                batches.append(
                    (epoch, permutation[first : first + cfg.training.minibatch_size])
                )

    for epoch, selected_rows in batches:
        flat_gradient.zero_()
        metric_parts: list[dict[str, Tensor]] = []
        variance_ema = getattr(source, "_pbg_family_gradient_variance_ema", None)
        if variance_ema is None:
            variance_ema = {}
            source._pbg_family_gradient_variance_ema = variance_ema
        sampling_start = time.perf_counter()
        if selected_rows is None:
            sample_specs = _stratified_decision_rows(
                data,
                bundle,
                source,
                cfg.training.minibatch_size,
                global_mass,
                torch_gen,
                variance_ema,
            )
        else:
            sample_specs = [
                ("", selected_rows, rank_objective_scale)
            ]
        if not sample_specs:
            raise ValueError("cannot optimize an empty fixed rollout")
        minibatch_count += len(sample_specs)
        optimization_host_seconds["minibatch_sampling"] += (
            time.perf_counter() - sampling_start
        )
        effective_total = 0
        previous_gradient_energy = 0.0
        trainable_count = max(
            1, sum(parameter.numel() for parameter in source.parameters() if parameter.requires_grad)
        )
        for family, logical_rows, objective_scale in sample_specs:
            sampling_start = time.perf_counter()
            minibatch = (
                gather_normal_minibatch(data.dataset, bundle, logical_rows)
                if data.normal
                else gather_minibatch_rows(
                    data.tree,
                    data.targets,
                    data.policy_cache,
                    data.decision_reach,
                    bundle,
                    logical_rows,
                )
            )
            effective_rows = int(minibatch.state_refs.numel())
            if effective_rows == 0:
                raise ValueError("cannot optimize an empty family stratum")
            effective_total += effective_rows
            optimization_host_seconds["minibatch_sampling"] += (
                time.perf_counter() - sampling_start
            )
            planner = getattr(bundle.rollout_backend, "plan_policy_batches", None)
            physical_plans = (
                planner(source, minibatch.state_refs, "training")
                if callable(planner)
                else [
                    torch.arange(
                        effective_rows,
                        dtype=torch.int64,
                        device=minibatch.state_refs.device,
                    )
                ]
            )
            for positions in physical_plans:
                physical = (
                    index_normal_minibatch(minibatch, positions)
                    if data.normal
                    else index_minibatch(minibatch, positions)
                )
                phase_start = time.perf_counter()
                with torch.autocast(
                    device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
                ):
                    family_forward = getattr(
                        bundle.rollout_backend, "policy_forward_family", None
                    )
                    output = (
                        family_forward(
                            source, family, physical.state_refs, physical.reach
                        )
                        if family and callable(family_forward)
                        else bundle.rollout_backend.policy_forward(
                            source, physical.state_refs, physical.reach
                        )
                    )
                    reference_output = None
                    if float(cfg.loss.c_kl) > 0.0:
                        if reference_model is None:
                            raise ValueError(
                                "reference_model is required when loss.c_kl is positive"
                            )
                        with torch.no_grad():
                            reference_output = (
                                family_forward(
                                    reference_model,
                                    family,
                                    physical.state_refs,
                                    physical.reach,
                                )
                                if family and callable(family_forward)
                                else bundle.rollout_backend.policy_forward(
                                    reference_model,
                                    physical.state_refs,
                                    physical.reach,
                                )
                            )
                    loss_sum, stats = spec.fn(
                        output,
                        physical,
                        physical,
                        bundle.policy_adapter,
                        cfg,
                        iteration=iteration,
                        reduction="sum",
                        reference=reference_output,
                    )
                    # Every physical split contributes to the original logical
                    # family mean.  Dividing by the split size would overweight
                    # families that need more physical model calls.
                    loss = (
                        loss_sum
                        / effective_rows
                        * objective_scale.to(loss_sum.device)
                    )
                optimization_host_seconds["forward_loss"] += (
                    time.perf_counter() - phase_start
                )
                phase_start = time.perf_counter()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite training loss")
                scaler.scale(loss).backward()
                optimization_host_seconds["backward_pass"] += (
                    time.perf_counter() - phase_start
                )
                if family:
                    scaled_energy = sum(
                        parameter.grad.detach().float().square().sum()
                        for parameter in source.parameters()
                        if parameter.grad is not None
                    )
                    scale = (
                        float(scaler.get_scale()) if scaler.is_enabled() else 1.0
                    )
                    current_energy = float(scaled_energy) / (scale * scale)
                    incremental_proxy = max(
                        abs(current_energy - previous_gradient_energy)
                        / trainable_count,
                        1.0e-12,
                    )
                    previous_gradient_energy = current_energy
                    old_proxy = variance_ema.get(family, incremental_proxy)
                    variance_ema[family] = (
                        0.95 * old_proxy + 0.05 * incremental_proxy
                    )
                metric_parts.append(stats)
                phase_start = time.perf_counter()
        sync_start = time.perf_counter()
        gradients_finite = _sync_scaled_gradients(
            model,
            int(os.environ.get("WORLD_SIZE", "1")),
            metrics=execution_metrics,
            flat_buffer=flat_gradient,
        )
        optimization_host_seconds["gradient_sync"] += time.perf_counter() - sync_start
        update_start = time.perf_counter()
        step_result = _finite_gated_optimizer_step(
            model,
            optimizer,
            scaler,
            gradients_finite=gradients_finite,
            grad_clip=cfg.training.grad_clip,
        )
        optimization_host_seconds["gradient_update"] += time.perf_counter() - update_start
        step_metrics = finalize_metric_stats(metric_parts)
        step_metrics.update(step_result.metrics())
        optimization_records.append((epoch, effective_total, step_metrics))

    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        assert optimization_start is not None
        iteration_profile = profile_parts[-1]
        iteration_profile["phase_optimization_seconds"] = (
            time.perf_counter() - optimization_start
        )
        if device.type == "cuda":
            iteration_profile["phase_optimization_peak_allocated_bytes"] = float(
                torch.cuda.max_memory_allocated(device)
            )
            iteration_profile["phase_optimization_peak_reserved_bytes"] = float(
                torch.cuda.max_memory_reserved(device)
            )

    del data
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        metrics_peak_allocated = float(torch.cuda.max_memory_allocated(device))
        metrics_peak_reserved = float(torch.cuda.max_memory_reserved(device))
    else:
        metrics_peak_allocated = 0.0
        metrics_peak_reserved = 0.0
    if profile:
        memory_metrics.update(
            memory_ledger.snapshot(
                "iteration_end",
                owned={
                    "model": source,
                    "optimizer": optimizer.state,
                    "reference": reference_model,
                    "backend": bundle.rollout_backend,
                },
            )
        )

    def aggregate_records(
        records: list[tuple[int, int, dict[str, float]]]
    ) -> dict[str, float]:
        if not records:
            return {}
        result: dict[str, float] = {}
        keys = {key for _, _, values in records for key in values}
        maximum_keys = {"ratio_max", "grad_norm_pre_clip"}
        minimum_keys = {"grad_clip_scale"}
        for key in keys:
            present = [
                (rows, values[key])
                for _, rows, values in records
                if key in values
            ]
            if key in maximum_keys:
                result[key] = max(value for _, value in present)
            elif key in minimum_keys:
                result[key] = min(value for _, value in present)
            else:
                denominator = sum(rows for rows, _ in present)
                result[key] = sum(rows * value for rows, value in present) / max(
                    denominator, 1
                )
        return result

    metrics = aggregate_records(optimization_records)
    if device.type == "cuda":
        metrics["iteration_peak_allocated_bytes"] = metrics_peak_allocated
        metrics["iteration_peak_reserved_bytes"] = metrics_peak_reserved
        metrics["iteration_end_allocated_bytes"] = float(
            torch.cuda.memory_allocated(device)
        )
        metrics["iteration_end_reserved_bytes"] = float(
            torch.cuda.memory_reserved(device)
        )
        metrics["iteration_end_allocator_slack_bytes"] = (
            metrics["iteration_end_reserved_bytes"]
            - metrics["iteration_end_allocated_bytes"]
        )
    metrics.update(
        {
            "trajectories": float(total_trajectories),
            "decisions": float(decision_total),
            "optimizer_steps": sum(
                values["optimizer_step_applied"]
                for _, _, values in optimization_records
            ),
            "minibatches": float(minibatch_count),
            "optimized_decision_rows": float(
                sum(rows for _, rows, _ in optimization_records)
            ),
        }
    )
    metrics.update(
        {
            f"phase_{name}_host_seconds": float(seconds)
            for name, seconds in optimization_host_seconds.items()
        }
    )
    metrics["phase_optimization_host_seconds"] = float(
        sum(optimization_host_seconds.values())
    )
    if cfg.training.ppo_epochs is not None:
        metrics["ppo_epochs"] = float(cfg.training.ppo_epochs)
        metrics["decision_epoch_coverage"] = (
            sum(rows for _, rows, _ in optimization_records)
            / max(decision_total * cfg.training.ppo_epochs, 1)
        )
        for epoch in range(cfg.training.ppo_epochs):
            epoch_records = [
                record for record in optimization_records if record[0] == epoch
            ]
            for key, value in aggregate_records(epoch_records).items():
                metrics[f"ppo_epoch_{epoch}_{key}"] = value

    profile_metrics: dict[str, float] = {}
    sum_keys = {
        "trajectories",
        "nodes",
        "leaves",
        "decisions",
        "terminal_samples",
        "completion_samples",
        "retained_target_bytes",
        "completion_effective_samples",
    }
    for part in profile_parts:
        for key, value in part.items():
            if key in sum_keys or key.endswith("_seconds"):
                profile_metrics[key] = profile_metrics.get(key, 0.0) + value
            elif "peak_" in key or key == "offspring_max":
                profile_metrics[key] = max(profile_metrics.get(key, 0.0), value)
            elif key not in profile_metrics:
                profile_metrics[key] = value
    profile_metrics["trajectories"] = float(total_trajectories)
    profile_metrics["decisions"] = float(decision_total)
    metrics.update(profile_metrics)
    metrics.update(memory_metrics)

    if profile:
        # Completion is streamed inside the RB backward-estimator call. Keep
        # its own timing for attribution, but do not double-count it in the
        # iteration wall-time estimate.
        top_level_phases = {
            "snapshot",
            "rollout",
            "backward",
            "compaction",
            "optimization",
        }
        phase_seconds = sum(
            metrics.get(f"phase_{name}_seconds", 0.0)
            for name in top_level_phases
        )
        metrics["profiled_iteration_seconds"] = phase_seconds
        metrics["phase_backward_estimator_seconds_excluding_completion"] = max(
            metrics.get("phase_backward_seconds", 0.0)
            - metrics.get("phase_completion_seconds", 0.0),
            0.0,
        )
        metrics["decisions_per_second"] = decision_total / max(phase_seconds, 1.0e-12)
        metrics["target_bytes_per_decision"] = metrics.get(
            "retained_target_bytes", 0.0
        ) / max(decision_total, 1)
        completion_seconds = metrics.get("phase_completion_seconds", 0.0)
        if completion_seconds:
            metrics["completion_effective_samples_per_second"] = metrics.get(
                "completion_effective_samples", 0.0
            ) / completion_seconds
        rollout_seconds = metrics.get("phase_rollout_seconds", 0.0)
        if rollout_seconds and normal:
            metrics["terminal_samples_per_second"] = total_trajectories / rollout_seconds

    metrics.update(_merge_completion_quality(quality_parts))
    metrics.update(identity_metrics)
    if execution_metrics is not None:
        execution_snapshot = execution_metrics.snapshot()
        metrics.update(execution_snapshot)
        totals = execution_metrics.totals()
        metrics["policy_physical_batches"] = float(totals.get("model_calls", 0))
        metrics["policy_planned_batches"] = float(
            totals.get("policy_pack_calls", 0)
        )
        metrics["policy_total_padded_tokens"] = float(
            totals.get("padded_tokens", 0)
        )
        metrics["policy_total_padded_token_pairs"] = float(
            totals.get("padded_token_pairs", 0)
        )
        metrics["policy_max_padded_tokens"] = execution_snapshot[
            "execution_padded_tokens_max"
        ]
        metrics["policy_max_padded_token_pairs"] = execution_snapshot[
            "execution_padded_token_pairs_max"
        ]
    return metrics


def run_training(
    cfg: RootConfig,
    *,
    bundle_builder: Callable[[RootConfig, torch.device], GameBundle],
    target_model_builder: Callable[[dict[str, Any], torch.device, Any | None], nn.Module]
    | None = None,
    resume_from: str | Path | None = None,
) -> None:
    """Train from scratch or resume an identity-checked complete run."""

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if cfg.completion.backend == "cpu":
        requested = torch.device("cpu")
    elif cfg.completion.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("poker completion backend 'cuda' requires an available CUDA device")
        requested = torch.device(f"cuda:{local_rank}")
    else:
        requested = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if requested.type == "cuda" and cfg.hardware is None:
        raise RuntimeError(
            "CUDA training requires an explicit hardware profile with token, "
            "memory, and device limits"
        )
    target_payload: dict[str, Any] | None = None
    probe_target: dict[str, Any] | None = None
    if cfg.probe is not None:
        if target_model_builder is None:
            raise RuntimeError("probe training requires a target-model builder")
        target_path = Path(cfg.probe.target_checkpoint)
        digest = hashlib.sha256()
        with target_path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if (
            cfg.probe.target_sha256 is not None
            and cfg.probe.target_sha256 != actual_sha256
        ):
            raise RuntimeError("probe target checkpoint SHA-256 does not match configuration")
        cfg.probe.target_sha256 = actual_sha256
        target_payload = torch.load(target_path, map_location="cpu", weights_only=False)
        if not isinstance(target_payload, dict) or not isinstance(
            target_payload.get("config"), dict
        ) or "model" not in target_payload:
            raise RuntimeError("probe target is not a PBG model checkpoint")
        target_game = target_payload["config"].get("game")
        expected_game = {
            "n_players": cfg.game.n_players,
            "stack_bb": cfg.game.stack_bb,
            "rank_count": cfg.game.rank_count,
        }
        if target_game != expected_game:
            raise RuntimeError("probe and target checkpoints must use identical (P,S,R)")
        probe_target = {
            "sha256": actual_sha256,
            "path": str(target_path),
            "run_id": target_payload.get("run_id"),
            "iteration": int(target_payload.get("iteration", -1)),
        }
    _configure_reproducibility(cfg.training.reproducibility)
    local_cfg = _rank_local_config(cfg, rank, world)
    allocation_cfg = local_cfg
    sequential_memory_control = cfg.training.mode != "branched_rao_blackwell"
    if sequential_memory_control:
        initial_per_gpu = int(cfg.rollout.sequential_roots_per_gpu_initial)
        maximum = int(cfg.rollout.sequential_roots_per_gpu_maximum)
        local_cfg.rollout.roots_per_gpu = initial_per_gpu
        local_cfg.rollout.trajectory_rank_counts = (initial_per_gpu,) * world
        allocation_cfg = copy.deepcopy(local_cfg)
        allocation_cfg.rollout.roots_per_gpu = maximum
        allocation_cfg.rollout.trajectory_rank_counts = (maximum,) * world
    _hardware_metadata(cfg, requested, world)
    torch_gen, np_gen = seed_all(cfg.training.seed, rank, requested)
    bundle = bundle_builder(allocation_cfg, requested)
    if not isinstance(bundle, GameBundle):
        raise TypeError("bundle builder must return GameBundle")
    device = torch.device(bundle.rollout_backend.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if torch_gen.device != device:
        torch_gen, np_gen = seed_all(cfg.training.seed, rank, device)
    bundle.model.to(device)
    if target_payload is not None:
        assert target_model_builder is not None and cfg.probe is not None
        opponent = target_model_builder(
            target_payload["config"], device, cfg.hardware
        )
        opponent.load_state_dict(target_payload["model"])
        opponent.eval().requires_grad_(False)
        bundle.opponent_model = opponent
        register_policy_model = getattr(
            bundle.rollout_backend, "register_policy_model", None
        )
        if callable(register_policy_model):
            register_policy_model(opponent)
            register_policy_model(bundle.model)
        bundle.probe_target = probe_target
        if cfg.probe.initialization == "target":
            try:
                bundle.model.load_state_dict(target_payload["model"])
            except RuntimeError as exc:
                raise RuntimeError(
                    "probe initialization 'target' requires identical architectures"
                ) from exc
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        # Each rank owns a plain model replica. Gradients are accumulated under
        # the exact local objective scale and reduced once through the explicit
        # flat SUM path; DDP's equal-rank averaging is intentionally not used.
        # Rank-addressed rollout RNG must not imply rank-addressed model
        # initialization: establish one identical learner before any rollout.
        for value in bundle.model.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer_cls = (
        torch.optim.AdamW
        if cfg.training.optimizer == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_cls(
        bundle.model.parameters(), lr=cfg.training.lr, betas=cfg.training.adam_betas
    )
    source_model = (
        bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    )
    reference_model: nn.Module | None = None
    reference_iteration: int | None = None
    if cfg.loss.c_kl > 0.0:
        reference_model = copy.deepcopy(source_model).eval()
        reference_model.requires_grad_(False)
    amp_enabled, amp_dtype = _precision(cfg, device)
    scaler = _scaler(bool(amp_enabled and amp_dtype == torch.float16))
    out_dir = Path(cfg.training.out_dir)
    start_iteration = 0
    run_id: str | None = None
    restored_memory_state: Mapping[str, Any] | None = None
    resume_path = None if resume_from is None else Path(resume_from)
    logical_work = {
        "trajectories": 0,
        "completion_samples": 0,
        "minibatches": 0,
        "optimizer_steps": 0,
    }

    if resume_path is not None:
        if resume_path.parent.resolve() != out_dir.resolve():
            raise RuntimeError("resume checkpoint must belong to training.out_dir")
        checkpoint_metadata = load_checkpoint(
            resume_path,
            bundle.model,
            optimizer,
            scaler,
            torch_gen,
            np_gen,
            map_location=device,
            rank=rank,
            reference_model=reference_model,
        )
        stored_logical_work = checkpoint_metadata.get("logical_work")
        if stored_logical_work is not None:
            if set(stored_logical_work) != set(logical_work):
                raise RuntimeError("checkpoint has malformed logical-work counters")
            logical_work = {
                key: int(stored_logical_work[key]) for key in logical_work
            }
        if cfg.probe is not None:
            stored_target = checkpoint_metadata.get("probe_target")
            if stored_target is None or stored_target.get("sha256") != cfg.probe.target_sha256:
                raise RuntimeError("resume probe target hash does not match checkpoint")
        if not compatible_scientific_configs(
            checkpoint_metadata["cfg_dict"], _scientific_config(cfg)
        ):
            raise RuntimeError("resume configuration does not match the checkpoint")
        start_iteration = int(checkpoint_metadata["iter"])
        if not 0 <= start_iteration <= cfg.training.iters:
            raise RuntimeError("checkpoint iteration is outside the configured schedule")
        run_id = checkpoint_metadata["run_id"]
        reference_iteration = checkpoint_metadata["reference_iteration"]
        if sequential_memory_control:
            restored_memory_state = checkpoint_metadata["adaptive_work_state"]
            if restored_memory_state is None:
                raise RuntimeError("sequential resume checkpoint has no memory controller state")
        if not run_id:
            raise RuntimeError("checkpoint predates run identities and cannot be resumed safely")
        if device.type == "cuda":
            # torch.load(map_location=device) temporarily materializes checkpoint
            # tensors before load_state_dict copies them into the live model and
            # optimizer. Release those now-dead cache blocks so resume does not
            # carry an artificial allocator peak into rollout calibration.
            torch.cuda.empty_cache()

    memory_plan = _analytical_memory_plan(
        allocation_cfg, bundle, has_reference=reference_model is not None
    )
    _validate_memory_preflight(allocation_cfg, device, memory_plan)
    memory_controller: SequentialMemoryController | None = None
    if sequential_memory_control:
        maximum = int(cfg.rollout.sequential_roots_per_gpu_maximum)
        phase_parts = {
            "rollout": (
                memory_plan["memory_plan_rollout_fixed_bytes"],
                memory_plan["memory_plan_rollout_variable_bytes"],
            ),
            "seal": (
                memory_plan["memory_plan_seal_fixed_bytes"],
                memory_plan["memory_plan_seal_variable_bytes"],
            ),
            "targets": (
                memory_plan["memory_plan_target_fixed_bytes"],
                memory_plan["memory_plan_target_variable_bytes"],
            ),
            "optimization": (
                memory_plan["memory_plan_optimization_fixed_bytes"],
                memory_plan["memory_plan_optimization_variable_bytes"],
            ),
        }
        phase_totals = {
            phase: fixed + variable
            for phase, (fixed, variable) in phase_parts.items()
        }
        analytic = {
            phase: PhaseMemoryModel(
                fixed_bytes=fixed,
                upper_bytes_per_root={
                    "analytic_schema_and_workspace": variable / maximum
                },
            )
            for phase, (fixed, variable) in phase_parts.items()
        }
        if device.type == "cuda":
            _, device_bytes = torch.cuda.mem_get_info(device)
            hard = _memory_limit(cfg, device, account_current_free=False)
            guard_bytes = int(device_bytes) - int(hard)
        else:
            # CPU runs exercise controller state/geometry without a device
            # allocator ceiling.  Leave ample synthetic headroom while still
            # using the same four phase observations and hysteresis logic.
            device_bytes = 8 * max(phase_totals.values()) + 1024
            guard_bytes = 0
        memory_controller = (
            SequentialMemoryController.from_state_dict(restored_memory_state)
            if restored_memory_state is not None
            else SequentialMemoryController(
                minimum_roots_per_gpu=int(
                    cfg.rollout.sequential_roots_per_gpu_minimum
                ),
                maximum_roots_per_gpu=maximum,
                current_roots_per_gpu=int(
                    cfg.rollout.sequential_roots_per_gpu_initial
                ),
                device_bytes=int(device_bytes),
                guard_bytes=guard_bytes,
                forecaster=PhaseMemoryForecaster(
                    analytic,
                    window=int(cfg.rollout.memory_calibration_window),
                    prediction_z=float(cfg.rollout.memory_prediction_z),
                ),
                growth_fraction=float(cfg.rollout.memory_growth_fraction),
                growth_delay=int(cfg.rollout.memory_growth_delay),
            )
        )
        local_cfg.rollout.roots_per_gpu = memory_controller.current_roots_per_gpu
        local_cfg.rollout.trajectory_rank_counts = (
            memory_controller.current_roots_per_gpu,
        ) * world

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_path = out_dir / "run.json"
        if resume_path is None:
            occupied = any(
                path.exists()
                for path in (run_path, out_dir / "metrics.jsonl")
            ) or any(out_dir.glob("ckpt_*.pt"))
            if occupied:
                raise RuntimeError(
                    "output directory already contains a run; pass resume_from explicitly"
                )
            run_id = uuid.uuid4().hex
            run_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "config_sha256": _config_fingerprint(cfg),
                        "world_size": world,
                        "probe_target": probe_target,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            if not run_path.exists():
                raise RuntimeError("resume requires the existing run.json")
            run_record = json.loads(run_path.read_text(encoding="utf-8"))
            if run_record.get("run_id") != run_id:
                raise RuntimeError("checkpoint belongs to a different run identity")
            recorded_fingerprint = run_record.get("config_sha256")
            if recorded_fingerprint not in {
                _config_fingerprint(cfg),
                legacy_config_fingerprint(cfg),
                config_dict_fingerprint(checkpoint_metadata["cfg_dict"]),
            } and not matches_legacy_run_fingerprint(
                recorded_fingerprint, checkpoint_metadata["cfg_dict"]
            ):
                raise RuntimeError("run configuration fingerprint does not match")
            if run_record.get("world_size") != world:
                raise RuntimeError("resume requires the original DDP world size")
            _validate_metrics_prefix(
                out_dir / "metrics.jsonl", start_iteration, run_id
            )
            for memory_rank in range(world):
                memory_path = out_dir / f"memory_rank_{memory_rank}.jsonl"
                if memory_path.exists():
                    _validate_metrics_prefix(memory_path, start_iteration, run_id)
        metadata = _hardware_metadata(cfg, device, world)
        metadata.update(memory_plan)
        hardware_name = (
            "hardware.json"
            if resume_path is None
            else f"hardware_resume_{start_iteration:07d}.json"
        )
        with (out_dir / hardware_name).open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
    if world > 1:
        identity = [run_id]
        dist.broadcast_object_list(identity, src=0)
        run_id = identity[0]
        dist.barrier()
    assert run_id is not None

    for iteration in range(start_iteration, cfg.training.iters):
        if memory_controller is not None:
            try:
                current = memory_controller.choose_root_width()
            except MemoryAdmissionStop as exc:
                # No current-iteration draw has occurred. Persist the unchanged
                # RNG/model state and the conservative controller evidence,
                # then stop without narrowing and rerolling the iteration.
                local_rng_state = _capture_rng_state(torch_gen, np_gen)
                if world > 1:
                    gathered_stop: list[dict[str, Any] | None] | None = (
                        [None] * world if rank == 0 else None
                    )
                    dist.gather_object(local_rng_state, gathered_stop, dst=0)
                    rank_rng_states = (
                        None
                        if rank != 0
                        else [item for item in gathered_stop or [] if item is not None]
                    )
                else:
                    rank_rng_states = [local_rng_state]
                if rank == 0:
                    save_checkpoint(
                        out_dir / f"ckpt_memory_stop_{iteration:07d}.pt",
                        bundle.model,
                        optimizer,
                        scaler,
                        iteration,
                        cfg,
                        torch_gen,
                        np_gen,
                        run_id=run_id,
                        rank_rng_states=rank_rng_states,
                        reference_model=reference_model,
                        reference_iteration=reference_iteration,
                        adaptive_work_state=memory_controller.state_dict(),
                        probe_target=probe_target,
                        logical_work=logical_work,
                    )
                    (out_dir / "memory_stop.json").write_text(
                        json.dumps(
                            {
                                "iter": iteration,
                                "run_id": run_id,
                                "reason": str(exc),
                                "roots_per_gpu": memory_controller.current_roots_per_gpu,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                break
            local_cfg.rollout.roots_per_gpu = current
            local_cfg.rollout.trajectory_rank_counts = (current,) * world
        if reference_model is not None and iteration % cfg.training.kl_ref_interval == 0:
            reference_model.load_state_dict(source_model.state_dict())
            reference_model.eval()
            reference_iteration = iteration
        metrics = _one_iteration(
            local_cfg,
            bundle,
            optimizer,
            scaler,
            torch_gen,
            iteration,
            profile=True,
            reference_model=reference_model,
            validate_identity=iteration == start_iteration,
        )
        if resume_path is not None and iteration == start_iteration:
            for key in tuple(metrics):
                if key.startswith("identity_"):
                    metrics.pop(key)
        if iteration == 0:
            metrics.update({key: float(value) for key, value in memory_plan.items()})
        diagnostic = {
            key: value
            for key, value in metrics.items()
            if (
                key.startswith("memory_")
                or key.startswith("phase_") and "peak_" in key
                or key.startswith("execution_")
                and any(
                    marker in key
                    for marker in (
                        "model_calls",
                        "model_rows",
                        "padded_token",
                        "queue_",
                        "forced_drain",
                        "completion_",
                    )
                )
                or key.startswith("trajectory_length_")
                or key in {"nodes", "leaves", "decisions", "retained_target_bytes"}
            )
        }
        with (out_dir / f"memory_rank_{rank}.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        "iter": iteration,
                        "run_id": run_id,
                        "rank": rank,
                        "optimizer_step": logical_work["optimizer_steps"],
                        "entered_learning": True,
                        **diagnostic,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _add_distributed_metric_summaries(metrics, device, world)
        _synchronize_memory_control_metrics(
            metrics, device, world, int(bundle.rollout_backend.n_players)
        )
        _validate_memory_calibration(cfg, device, metrics)
        if memory_controller is not None:
            predicted = memory_controller.last_predicted_peaks
            fallback = int(
                metrics.get(
                    "iteration_peak_reserved_bytes",
                    max(predicted.values()),
                )
            )
            actual_keys = {
                "rollout": "phase_rollout_peak_reserved_bytes",
                "seal": "memory_preoptimization_peak_reserved_bytes",
                "targets": "phase_backward_peak_reserved_bytes",
                "optimization": "phase_optimization_peak_reserved_bytes",
            }
            phase_observations = {}
            for phase in MEMORY_PHASES:
                actual = int(metrics.get(actual_keys[phase], fallback))
                expected = int(predicted[phase])
                analytic_fixed = memory_controller.forecaster.analytic_models[
                    phase
                ].fixed_bytes
                observed_fixed = min(analytic_fixed, actual)
                phase_observations[phase] = PhaseMemoryObservation(
                    roots_per_gpu=memory_controller.current_roots_per_gpu,
                    fixed_bytes=observed_fixed,
                    variable_bytes={
                        "analytic_schema_and_workspace": actual - observed_fixed
                    },
                    predicted_peak_bytes=expected,
                    actual_peak_bytes=actual,
                )
            metrics.update(memory_controller.observe_success(phase_observations))
        if reference_iteration is not None:
            metrics["reference_iteration"] = float(reference_iteration)
        for name in logical_work:
            distributed_key = (
                f"{name}_rank_max"
                if name == "optimizer_steps"
                else f"{name}_rank_sum"
            )
            value = metrics.get(
                distributed_key if world > 1 else name,
                0.0,
            )
            logical_work[name] += int(round(value))
            metrics[f"cumulative_{name}"] = float(logical_work[name])
        if "decisions_rank_sum" in metrics:
            global_decisions = max(metrics["decisions_rank_sum"], 1.0)
            for street in ("preflop", "flop", "turn", "river"):
                key = f"decision_nodes_{street}_rank_sum"
                if key in metrics:
                    metrics[f"decision_fraction_{street}_global"] = (
                        metrics[key] / global_decisions
                    )
        checkpoint_every = (
            cfg.training.checkpoint_every
            if cfg.hardware is None or cfg.hardware.checkpoint_every is None
            else cfg.hardware.checkpoint_every
        )
        checkpoint_due = (iteration + 1) % checkpoint_every == 0
        if rank == 0:
            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"iter": iteration, "run_id": run_id, **metrics},
                        sort_keys=True,
                    )
                    + "\n"
                )
        rank_rng_states: list[dict[str, Any]] | None = None
        if checkpoint_due:
            local_rng_state = _capture_rng_state(torch_gen, np_gen)
            if world > 1:
                gathered: list[dict[str, Any] | None] | None = (
                    [None] * world if rank == 0 else None
                )
                dist.gather_object(local_rng_state, gathered, dst=0)
                if rank == 0:
                    assert gathered is not None and all(item is not None for item in gathered)
                    rank_rng_states = [item for item in gathered if item is not None]
            else:
                rank_rng_states = [local_rng_state]
        if rank == 0 and checkpoint_due:
            save_checkpoint(
                out_dir / f"ckpt_{iteration + 1:07d}.pt",
                bundle.model,
                optimizer,
                scaler,
                iteration + 1,
                cfg,
                torch_gen,
                np_gen,
                run_id=run_id,
                rank_rng_states=rank_rng_states,
                reference_model=reference_model,
                reference_iteration=reference_iteration,
                adaptive_work_state=(
                    None
                    if memory_controller is None
                    else memory_controller.state_dict()
                ),
                probe_target=probe_target,
                logical_work=logical_work,
            )
    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
