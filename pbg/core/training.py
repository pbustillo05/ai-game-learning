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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from pbg.core.config import RootConfig, seed_all
from pbg.core.game import GameBundle
from pbg.core.losses import COMPOSITES, LossSpec, _selected_logp, finalize_metric_stats
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
from pbg.core.rollout import (
    sample_trees,
    sample_trees_sequential,
    sequential_leaf_logical_rng_keys,
)
from pbg.core.queued import LOGICAL_RNG_VERSION, DrawPurpose
from pbg.core.tree import DecisionReachCache, build_decision_alias
from pbg.core.values import HeadEvals, streamed_branched_value


GAME_FACTORIES = {
    "poker": "pbg.games.poker.bundle:build_bundle",
}


def build_game_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Resolve and call the selected game factory without static game imports."""

    try:
        location = GAME_FACTORIES[cfg.game.name]
    except KeyError as exc:
        raise NotImplementedError(f"no game factory is registered for {cfg.game.name!r}") from exc
    module_name, separator, function_name = location.partition(":")
    if not separator:
        raise RuntimeError(f"invalid game factory location {location!r}")
    factory = getattr(importlib.import_module(module_name), function_name)
    bundle = factory(cfg, device)
    if not isinstance(bundle, GameBundle):
        raise TypeError("game factory must return GameBundle")
    return bundle


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
) -> None:
    """Atomically store model, optimizer, scaler, config, and explicit RNG state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    scientific_config = asdict(cfg)
    hardware_profile = scientific_config.pop("hardware", None)
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": int(iteration),
        "config": scientific_config,
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
        # ``map_location=device`` correctly restores model/optimizer tensors,
        # but CUDA's RNG setter specifically requires CPU ByteTensor states.
        # Normalize both legacy top-level and per-rank checkpoint payloads at
        # this boundary rather than coupling checkpoint loading to one device.
        torch.cuda.set_rng_state_all(
            [state.detach().to(device="cpu", dtype=torch.uint8) for state in cuda_rng]
        )
    return {
        "iter": int(payload["iteration"]),
        "cfg_dict": payload["config"],
        "run_id": payload.get("run_id"),
        "reference_iteration": payload.get("reference_iteration"),
    }


def _capture_rng_state(
    torch_gen: torch.Generator, np_gen: np.random.Generator
) -> dict[str, Any]:
    """Capture the complete process and explicit-generator RNG state."""

    return {
        "torch_generator": torch_gen.get_state(),
        "numpy_generator": copy.deepcopy(np_gen.bit_generator.state),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _scientific_config(cfg: RootConfig) -> dict[str, Any]:
    """Return the checkpoint-compatible configuration, excluding hardware."""

    result = asdict(cfg)
    result.pop("hardware", None)
    return result


def _config_fingerprint(cfg: RootConfig) -> str:
    encoded = json.dumps(
        _scientific_config(cfg), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _validate_metrics_prefix(path: Path, completed: int, run_id: str) -> None:
    """Require a gap-free metrics prefix before appending a resumed run."""

    if not path.exists():
        raise RuntimeError("resume requires the existing metrics.jsonl")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    iterations = [record.get("iter") for record in records]
    if iterations != list(range(completed)):
        raise RuntimeError(
            "metrics.jsonl is not the exact gap-free prefix required by the checkpoint"
        )
    if any(record.get("run_id") != run_id for record in records):
        raise RuntimeError("metrics.jsonl belongs to a different run identity")


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
    if device.type != "cuda" or name == "fp32" or not cfg.train.amp:
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
        "reproducibility": cfg.train.reproducibility,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "logical_rng": LOGICAL_RNG_VERSION,
        "scheduler_control_sync_policy": "one_fixed_control_block_per_decision_v1",
        "completion_backend": cfg.completion.backend,
        "queue_calibration": {
            "batch_rows": cfg.rollout.queue_batch_rows,
            "packing": cfg.rollout.queue_packing,
            "admission_size": cfg.rollout.queue_admission_size,
            "arena_chunk_nodes": cfg.rollout.arena_chunk_nodes,
        },
        "memory_limits": {
            "queue_memory_bytes": cfg.rollout.queue_memory_bytes,
            "node_budget": cfg.rollout.node_budget,
            "reserve_vram_gib": None if cfg.hardware is None else cfg.hardware.reserve_vram_gib,
            "max_memory_fraction": None if cfg.hardware is None else cfg.hardware.max_memory_fraction,
        },
    }
    if cfg.game.name == "poker":
        try:
            evaluator = importlib.import_module("poker_gpu_evaluator.pbg")
            metadata["evaluator_api_version"] = evaluator.API_VERSION
            metadata["evaluator_module"] = str(Path(evaluator.__file__).resolve())
        except (ImportError, AttributeError):
            metadata["evaluator_api_version"] = None
            metadata["evaluator_module"] = None
    if cfg.training.mode == "branched_rao_blackwell":
        metadata["rank_start_quotas"] = [1] * world
        metadata["global_sequential_starts"] = None
    else:
        metadata["rank_start_quotas"] = list(
            _sequential_start_quotas(cfg.rollout.sequential_starts, world)
        )
        metadata["global_sequential_starts"] = cfg.rollout.sequential_starts
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


def _sequential_start_quotas(total: int, world: int) -> tuple[int, ...]:
    """Partition a global start budget into deterministic contiguous prefixes."""

    if isinstance(total, bool) or total <= 0 or world <= 0:
        raise ValueError("sequential start and world sizes must be positive")
    if total < world:
        raise ValueError(
            "rollout.sequential_starts must provide at least one start per rank"
        )
    base, extra = divmod(total, world)
    return tuple(base + (rank < extra) for rank in range(world))


def _rank_local_config(cfg: RootConfig, rank: int, world: int) -> RootConfig:
    """Return one rank's rollout quota with a collision-free global prefix."""

    local = copy.deepcopy(cfg)
    if cfg.training.mode == "branched_rao_blackwell":
        local.rollout.sequential_starts = 1
        local.rollout.trajectory_rank_counts = tuple(1 for _ in range(world))
        return local
    quotas = _sequential_start_quotas(cfg.rollout.sequential_starts, world)
    local.rollout.sequential_starts = quotas[rank]
    # These attributes are execution metadata, not accepted YAML keys.
    local.rollout.trajectory_rank_counts = quotas
    return local


def _sync_scaled_gradients(
    model: nn.Module, world: int, *, metrics: Any | None = None
) -> bool:
    """SUM one reusable flat fp32 buffer and return the common finite decision."""

    source = model.module if hasattr(model, "module") else model
    parameters = [parameter for parameter in source.parameters() if parameter.requires_grad]
    total = sum(parameter.numel() for parameter in parameters)
    if total == 0:
        return True
    cache = getattr(source, "_pbg_flat_gradient_buffers", None)
    if cache is None:
        cache = {}
        source._pbg_flat_gradient_buffers = cache
    key = (str(parameters[0].device), total)
    flat = cache.get(key)
    if flat is None or flat.device != parameters[0].device:
        flat = torch.empty(total, dtype=torch.float32, device=parameters[0].device)
        cache[key] = flat
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        if parameter.grad is None:
            flat[offset : offset + count].zero_()
        else:
            flat[offset : offset + count].copy_(
                parameter.grad.detach().reshape(-1).to(torch.float32)
            )
        offset += count
    if world > 1:
        started = time.perf_counter()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if metrics is not None:
            metrics.record_collective(
                nbytes=flat.numel() * flat.element_size(),
                seconds=time.perf_counter() - started,
            )
    finite = bool(torch.isfinite(flat).all())
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.copy_(
            flat[offset : offset + count].view_as(parameter).to(parameter.grad.dtype)
        )
        offset += count
    return finite


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
        alpha = (mass / global_mass.to(mass.device)).to(torch.float32)
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
        "loss",
        "policy",
        "value",
        "entropy",
        "reference_kl",
        "grad_norm_pre_clip",
        "grad_clip_scale",
        "completion_mean_estimator_stddev",
        "completion_max_estimator_stddev",
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
    dist.all_reduce(summed, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    total, count = summed.chunk(2)
    for index, name in enumerate(names):
        if not bool(count[index]):
            continue
        metrics[f"{name}_rank_mean"] = float(total[index] / count[index])
        metrics[f"{name}_rank_max"] = float(maximum[index])


@dataclass
class _IterationData:
    """Objects intentionally retained across rollout and optimization phases."""

    normal: bool
    completion_quality: dict[str, float]
    dataset: Any = None
    tree: Any = None
    targets: Any = None
    alias: Any = None
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
    """Return a conservative byte plan for preallocated training structures."""

    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    parameter = sum(
        value.numel() * value.element_size() for value in model.parameters()
    )
    backend = bundle.rollout_backend
    players = int(backend.n_players)
    types = max(int(value) for value in backend.x_sizes)
    events = int(getattr(backend, "e_max", 1))
    resident_trees = int(
        cfg.rollout.n_trees
    )
    nodes = resident_trees * int(cfg.rollout.node_budget)
    fp32 = 4
    int32 = 4
    topology = nodes * (
        int32 * (6 + players) + fp32 * 2
    )
    dense_reach = nodes * players * types * fp32
    if cfg.rollout.reach_storage == "recompute":
        rollout_reach = resident_trees * players * types * fp32
    else:
        rollout_reach = dense_reach
    # Four simultaneous wave-sized rows cover active, repeated-child,
    # successor, and gather/output reach tensors before the old wave dies.
    rollout_transient_reach = 4 * dense_reach
    decision_reach = dense_reach
    policy_cache = nodes * (events * types + types) * fp32
    rb_targets = (
        0
        if cfg.training.mode == "normal"
        else nodes * ((2 + 2 * events) * types * fp32 + events * (int32 + 1))
    )
    normal_targets = (
        nodes * (5 * fp32 + int32)
        if cfg.training.mode == "normal"
        else 0
    )
    optimizer_and_grad = 3 * parameter  # grad + two fp32 Adam moments
    rollout_snapshot = parameter
    reference = parameter if has_reference else 0
    rollout_phase = (
        parameter
        + rollout_snapshot
        + reference
        + topology
        + rollout_reach
        + rollout_transient_reach
        + policy_cache
    )
    optimization_phase = (
        parameter
        + optimizer_and_grad
        + reference
        + decision_reach
        + policy_cache
        + rb_targets
        + normal_targets
    )
    return {
        "memory_plan_parameter_bytes": parameter,
        "memory_plan_topology_bytes": topology,
        "memory_plan_rollout_reach_bytes": rollout_reach,
        "memory_plan_transient_reach_bytes": rollout_transient_reach,
        "memory_plan_decision_reach_bytes": decision_reach,
        "memory_plan_policy_cache_bytes": policy_cache,
        "memory_plan_target_bytes": rb_targets + normal_targets,
        "memory_plan_optimizer_and_grad_bytes": optimizer_and_grad,
        "memory_plan_snapshot_bytes": rollout_snapshot,
        "memory_plan_reference_bytes": reference,
        "memory_plan_rollout_phase_bytes": rollout_phase,
        "memory_plan_optimization_phase_bytes": optimization_phase,
        "memory_plan_structural_peak_bytes": max(rollout_phase, optimization_phase),
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


def _validate_memory_preflight(
    cfg: RootConfig, device: torch.device, plan: dict[str, int]
) -> None:
    """Reject a structurally impossible CUDA plan before rollout allocation."""

    if device.type != "cuda":
        return
    limit = _memory_limit(cfg, device)
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
        if key.startswith("phase_") and key.endswith("_peak_reserved_bytes")
    ]
    if not peaks:
        raise RuntimeError("first CUDA iteration did not emit memory calibration peaks")
    measured = int(max(peaks))
    metrics["calibrated_peak_reserved_bytes"] = float(measured)
    metrics["calibrated_peak_with_safety_bytes"] = float(int(measured * 1.25))
    safe_limit = _memory_limit(cfg, device, account_current_free=False)
    metrics["calibrated_safe_limit_bytes"] = float(safe_limit)
    if int(measured * 1.25) > safe_limit:
        raise MemoryError(
            "measured CUDA peak plus 25% topology/allocator safety margin exceeds "
            f"the configured limit: measured={measured}, safe_limit={safe_limit}"
        )


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

    Returning from this function is the lifetime boundary: the frozen rollout
    model, completion diagnostics, leaf reaches, and full reach cache are no
    longer referenced when optimization starts.
    """

    source = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = next(source.parameters()).device
    execution_metrics = getattr(bundle.rollout_backend, "execution_metrics", None)
    profile_metrics: dict[str, float] = {}
    owns_frozen_model = frozen_model is None
    if frozen_model is None:
        frozen_model = _profiled_call(
            profile_metrics,
            "snapshot",
            device,
            profile,
            lambda: copy.deepcopy(source).eval(),
        )
    frozen_bundle = GameBundle(
        model=frozen_model,
        rollout_backend=bundle.rollout_backend,
        policy_adapter=bundle.policy_adapter,
        completion_backend=bundle.completion_backend,
    )
    normal = cfg.training.mode == "normal"
    with torch.no_grad():
        if normal:
            sampled = _profiled_call(
                profile_metrics,
                "rollout",
                device,
                profile,
                lambda: sample_trees_sequential(
                    frozen_bundle,
                    cfg,
                    torch_gen,
                    return_profiles=True,
                    iteration=iteration,
                ),
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
                        "trajectories": float(cfg.rollout.n_trees),
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

        tree, reach_cache, policy_cache = _profiled_call(
            profile_metrics,
            "rollout",
            device,
            profile,
            lambda: sample_trees(
                frozen_bundle, cfg, torch_gen, iteration=iteration
            ),
        )
        decision_weight = tree.path_weight[
            policy_cache.node_ids.to(tree.path_weight.device, dtype=torch.int64)
        ].detach().to(torch.float64)
        local_mass = decision_weight.sum()
        global_mass, global_work, global_started = _begin_global_decision_mass(
            local_mass
        )
        # The rollout snapshot has no consumer in completion or target
        # reduction; end its lifetime before either phase allocates workspace.
        del frozen_bundle
        if owns_frozen_model:
            del frozen_model
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
                emit_q="q_hat" in spec.required_targets,
                max_workspace_bytes=workspace_bytes,
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
        alias, decision_reach = _profiled_call(
            profile_metrics,
            "compaction",
            device,
            profile,
            lambda: (
                build_decision_alias(tree),
                DecisionReachCache.from_reach_cache(
                    tree, reach_cache, policy_cache, bundle.rollout_backend
                ),
            ),
        )
        decisions = int(policy_cache.node_ids.numel())
        offspring = tree.child_ptr[1:] - tree.child_ptr[:-1]
        profile_metrics.update(
            {
                (
                    "sampled_trees"
                    if cfg.training.mode == "branched_rao_blackwell"
                    else "trajectories"
                ): float(cfg.rollout.n_trees),
                "nodes": float(tree.owner.numel()),
                "leaves": float(leaf_ids.numel()),
                "decisions": float(decisions),
                "offspring_mean": float(offspring.float().mean()),
                "offspring_max": float(offspring.max()),
            }
        )
        if profile:
            completion_effective = (
                completion_quality.get("completion_mean_n_eff", 0.0)
                * float(leaf_ids.numel())
            )
            retained = _tensor_bytes((targets, alias, policy_cache, decision_reach))
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
            alias=alias,
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
            ratio_min = torch.minimum(ratio_min, ratios.min())
            ratio_max = torch.maximum(ratio_max, ratios.max())
    if was_training:
        model.train()
    tolerance = 2.0e-3 if amp_enabled else 2.0e-5
    value_ok = maximum_value_error <= tolerance * (
        1.0 + torch.maximum(current_value.abs().max(), cached_value.abs().max())
    )
    logp_ok = maximum_logp_error <= tolerance
    unclipped = (ratio_min >= 1.0 - float(cfg.loss.clip_eps)) & (
        ratio_max <= 1.0 + float(cfg.loss.clip_eps)
    )
    ok = value_ok & logp_ok & unclipped
    if device.type == "cuda":
        torch._assert_async(ok, "old/current-policy identity launch gate failed")
        torch.cuda.synchronize(device)
    elif not bool(ok):
        raise RuntimeError("old/current-policy identity launch gate failed")
    return {
        "identity_max_abs_logp_error": float(maximum_logp_error),
        "identity_max_abs_value_error": float(maximum_value_error),
        "identity_ratio_min": float(ratio_min),
        "identity_ratio_max": float(ratio_max),
    }


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
) -> dict[str, float]:
    spec = _loss_spec(cfg)
    execution_metrics = getattr(bundle.rollout_backend, "execution_metrics", None)
    if execution_metrics is not None:
        execution_metrics.clear()
    model = bundle.model
    source = model.module if hasattr(model, "module") else model
    device = next(model.parameters()).device
    amp_enabled, amp_dtype = _precision(cfg, device)
    normal = cfg.training.mode == "normal"
    total_trajectories = int(cfg.rollout.n_trees)
    profile_parts: list[dict[str, float]] = []
    quality_parts: list[tuple[int, dict[str, float]]] = []
    optimization_records: list[tuple[int, int, dict[str, float]]] = []
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
    if validate_identity:
        if execution_metrics is None:
            identity_metrics = _validate_old_policy_identity(cfg, bundle, data)
        else:
            # The identity gate is launch/resume validation, not scientific
            # rollout or learner work.  Excluding it keeps exact executor
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

    local_mass, global_mass = _await_global_decision_mass(
        data, metrics=execution_metrics
    )
    rank_objective_scale = (local_mass / global_mass).to(torch.float32)
    if data.profile_metrics is not None:
        data.profile_metrics["local_decision_mass"] = float(local_mass)
        data.profile_metrics["global_decision_mass"] = float(global_mass)
        data.profile_metrics["rank_objective_scale"] = float(
            rank_objective_scale
        )

    model.train()
    optimization_start = None
    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        optimization_start = time.perf_counter()

    if cfg.train.ppo_epochs is None:
        batches: list[tuple[int, Tensor | None]] = [
            (-1, None) for _ in range(cfg.train.gd_steps_per_iter)
        ]
    else:
        batches = []
        for epoch in range(cfg.train.ppo_epochs):
            permutation = torch.randperm(decisions, generator=torch_gen, device=device)
            for first in range(0, decisions, cfg.train.minibatch_size):
                batches.append(
                    (epoch, permutation[first : first + cfg.train.minibatch_size])
                )

    for epoch, selected_rows in batches:
        optimizer.zero_grad(set_to_none=True)
        metric_parts: list[dict[str, Tensor]] = []
        variance_ema = getattr(source, "_pbg_family_gradient_variance_ema", None)
        if variance_ema is None:
            variance_ema = {}
            source._pbg_family_gradient_variance_ema = variance_ema
        if selected_rows is None:
            sample_specs = _stratified_decision_rows(
                data,
                bundle,
                source,
                cfg.train.minibatch_size,
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
        effective_total = 0
        previous_gradient_energy = 0.0
        trainable_count = max(
            1, sum(parameter.numel() for parameter in source.parameters() if parameter.requires_grad)
        )
        for family, logical_rows, objective_scale in sample_specs:
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
            with torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                family_forward = getattr(
                    bundle.rollout_backend, "policy_forward_family", None
                )
                output = (
                    family_forward(
                        source, family, minibatch.state_refs, minibatch.reach
                    )
                    if family and callable(family_forward)
                    else bundle.rollout_backend.policy_forward(
                        source, minibatch.state_refs, minibatch.reach
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
                                minibatch.state_refs,
                                minibatch.reach,
                            )
                            if family and callable(family_forward)
                            else bundle.rollout_backend.policy_forward(
                                reference_model,
                                minibatch.state_refs,
                                minibatch.reach,
                            )
                        )
                loss_sum, stats = spec.fn(
                    output,
                    minibatch,
                    minibatch,
                    bundle.policy_adapter,
                    cfg,
                    iteration=iteration,
                    reduction="sum",
                    reference=reference_output,
                )
                loss = (
                    loss_sum
                    / effective_rows
                    * objective_scale.to(loss_sum.device)
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            if family:
                scaled_energy = sum(
                    parameter.grad.detach().float().square().sum()
                    for parameter in source.parameters()
                    if parameter.grad is not None
                )
                scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
                current_energy = float(scaled_energy) / (scale * scale)
                incremental_proxy = max(
                    abs(current_energy - previous_gradient_energy) / trainable_count,
                    1.0e-12,
                )
                previous_gradient_energy = current_energy
                old_proxy = variance_ema.get(family, incremental_proxy)
                variance_ema[family] = 0.95 * old_proxy + 0.05 * incremental_proxy
            metric_parts.append(stats)
        gradients_finite = _sync_scaled_gradients(
            model,
            int(os.environ.get("WORLD_SIZE", "1")),
            metrics=execution_metrics,
        )
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.train.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        step_metrics = finalize_metric_stats(metric_parts)
        step_metrics["grad_norm_pre_clip"] = float(grad_norm)
        step_metrics["grad_clip_scale"] = min(
            1.0, float(cfg.train.grad_clip) / max(float(grad_norm), 1.0e-12)
        )
        step_metrics["reduced_gradients_finite"] = float(gradients_finite)
        optimization_records.append((epoch, effective_total, step_metrics))

    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        assert optimization_start is not None
        chunk_profile = profile_parts[-1]
        chunk_profile["phase_optimization_seconds"] = (
            time.perf_counter() - optimization_start
        )
        if device.type == "cuda":
            chunk_profile["phase_optimization_peak_allocated_bytes"] = float(
                torch.cuda.max_memory_allocated(device)
            )
            chunk_profile["phase_optimization_peak_reserved_bytes"] = float(
                torch.cuda.max_memory_reserved(device)
            )

    del data

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
    metrics.update(
        {
            "trajectories": float(total_trajectories),
            "decisions": float(decision_total),
            "optimizer_steps": float(len(optimization_records)),
            "optimized_decision_rows": float(
                sum(rows for _, rows, _ in optimization_records)
            ),
        }
    )
    if cfg.train.ppo_epochs is not None:
        metrics["ppo_epochs"] = float(cfg.train.ppo_epochs)
        metrics["decision_epoch_coverage"] = (
            sum(rows for _, rows, _ in optimization_records)
            / max(decision_total * cfg.train.ppo_epochs, 1)
        )
        for epoch in range(cfg.train.ppo_epochs):
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

    if profile:
        phase_seconds = sum(
            value
            for key, value in metrics.items()
            if key.startswith("phase_") and key.endswith("_seconds")
        )
        metrics["profiled_iteration_seconds"] = phase_seconds
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


def train(
    cfg: RootConfig, *, resume_from: str | Path | None = None
) -> None:
    """Train from scratch or resume an identity-checked complete run."""

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if cfg.game.name == "poker" and cfg.completion.backend == "cpu":
        requested = torch.device("cpu")
    elif cfg.game.name == "poker" and cfg.completion.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("poker completion backend 'cuda' requires an available CUDA device")
        requested = torch.device(f"cuda:{local_rank}")
    else:
        requested = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _configure_reproducibility(cfg.train.reproducibility)
    local_cfg = _rank_local_config(cfg, rank, world)
    _hardware_metadata(cfg, requested, world)
    torch_gen, np_gen = seed_all(cfg.train.seed, rank, requested)
    bundle = build_game_bundle(local_cfg, requested)
    device = torch.device(bundle.rollout_backend.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if torch_gen.device != device:
        torch_gen, np_gen = seed_all(cfg.train.seed, rank, device)
    bundle.model.to(device)
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        # Each rank owns a plain model replica. Gradients are accumulated under
        # the exact local objective scale and reduced once through the explicit
        # flat SUM path; DDP's equal-rank averaging is intentionally not used.
    optimizer_cls = (
        torch.optim.AdamW
        if cfg.train.optimizer == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_cls(
        bundle.model.parameters(), lr=cfg.train.lr, betas=cfg.train.adam_betas
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
    out_dir = Path(cfg.train.out_dir)
    start_iteration = 0
    run_id: str | None = None
    resume_path = None if resume_from is None else Path(resume_from)

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
        if checkpoint_metadata["cfg_dict"] != _scientific_config(cfg):
            raise RuntimeError("resume configuration does not match the checkpoint")
        start_iteration = int(checkpoint_metadata["iter"])
        if not 0 <= start_iteration <= cfg.train.iters:
            raise RuntimeError("checkpoint iteration is outside the configured schedule")
        run_id = checkpoint_metadata["run_id"]
        reference_iteration = checkpoint_metadata["reference_iteration"]
        if not run_id:
            raise RuntimeError("checkpoint predates run identities and cannot be resumed safely")

    memory_plan = _analytical_memory_plan(
        local_cfg, bundle, has_reference=reference_model is not None
    )
    _validate_memory_preflight(local_cfg, device, memory_plan)

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
            if run_record.get("config_sha256") != _config_fingerprint(cfg):
                raise RuntimeError("run configuration fingerprint does not match")
            if run_record.get("world_size") != world:
                raise RuntimeError("resume requires the original DDP world size")
            _validate_metrics_prefix(
                out_dir / "metrics.jsonl", start_iteration, run_id
            )
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

    for iteration in range(start_iteration, cfg.train.iters):
        if reference_model is not None and iteration % cfg.train.kl_ref_interval == 0:
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
            profile=iteration == 0 and device.type == "cuda",
            reference_model=reference_model,
            validate_identity=iteration == start_iteration,
        )
        if resume_path is not None and iteration == start_iteration:
            for key in tuple(metrics):
                if key.startswith("identity_"):
                    metrics.pop(key)
        if iteration == 0:
            metrics.update({key: float(value) for key, value in memory_plan.items()})
            _validate_memory_calibration(cfg, device, metrics)
        if reference_iteration is not None:
            metrics["reference_iteration"] = float(reference_iteration)
        _add_distributed_metric_summaries(metrics, device, world)
        checkpoint_every = (
            cfg.train.checkpoint_every
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
            )
    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
