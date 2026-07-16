"""Minimal rollout/target/optimization training loop with DDP checkpoints."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
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
from pbg.core.losses import COMPOSITES, LossSpec, finalize_metric_stats
from pbg.core.minibatch import index_minibatch, sample_minibatch
from pbg.core.normal import (
    build_normal_dataset_from_sampled,
    index_normal_minibatch,
    sample_normal_minibatch,
)
from pbg.core.rollout import _sample_trees_sequential, sample_trees
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
) -> dict[str, Any]:
    """Restore checkpoint state and return iteration/config metadata."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload["model"])
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
        torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "iter": int(payload["iteration"]),
        "cfg_dict": payload["config"],
        "run_id": payload.get("run_id"),
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
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


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
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "scheduler_time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "reproducibility": cfg.train.reproducibility,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
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


def _sync_scaled_gradients(model: nn.Module, world: int) -> None:
    """Perform one deterministic DDP reduction after local accumulation."""

    if world <= 1:
        return
    source = model.module if hasattr(model, "module") else model
    for parameter in source.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


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
) -> _IterationData:
    """Build compact training data in a frame that owns all frozen transients.

    Returning from this function is the lifetime boundary: the frozen rollout
    model, completion diagnostics, leaf reaches, and full reach cache are no
    longer referenced when optimization starts.
    """

    source = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = next(source.parameters()).device
    profile_metrics: dict[str, float] = {}
    frozen_model = _profiled_call(
        profile_metrics, "snapshot", device, profile, lambda: copy.deepcopy(source).eval()
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
                lambda: _sample_trees_sequential(
                    frozen_bundle, cfg.rollout, torch_gen, return_profiles=True
                ),
            )
            dataset = _profiled_call(
                profile_metrics,
                "backward",
                device,
                profile,
                lambda: build_normal_dataset_from_sampled(
                    frozen_bundle, cfg, torch_gen, sampled
                ),
            )
            if profile:
                tree, _, policy_cache, _ = sampled
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
                profile_metrics=profile_metrics,
            )

        tree, reach_cache, policy_cache = _profiled_call(
            profile_metrics,
            "rollout",
            device,
            profile,
            lambda: sample_trees(frozen_bundle, cfg, torch_gen),
        )
        # The rollout snapshot has no consumer in completion or target
        # reduction; end its lifetime before either phase allocates workspace.
        del frozen_bundle, frozen_model
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
        if profile:
            decisions = int(policy_cache.node_ids.numel())
            completion_effective = (
                completion_quality.get("completion_mean_n_eff", 0.0)
                * float(leaf_ids.numel())
            )
            retained = _tensor_bytes((targets, alias, policy_cache, decision_reach))
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
                    "completion_effective_samples": completion_effective,
                    "retained_target_bytes": float(retained),
                    "offspring_mean": float(offspring.float().mean()),
                    "offspring_max": float(offspring.max()),
                }
            )
        return _IterationData(
            normal=False,
            completion_quality=completion_quality,
            targets=targets,
            alias=alias,
            policy_cache=policy_cache,
            decision_reach=decision_reach,
            profile_metrics=profile_metrics,
        )


def _one_iteration(
    cfg: RootConfig,
    bundle: GameBundle,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    torch_gen: torch.Generator,
    iteration: int,
    *,
    profile: bool = False,
) -> dict[str, float]:
    spec = _loss_spec(cfg)
    batch_metrics = getattr(bundle.rollout_backend, "policy_batch_metrics", None)
    if batch_metrics is not None:
        batch_metrics.clear()
    model = bundle.model
    source = model.module if hasattr(model, "module") else model
    data = _build_iteration_data(cfg, bundle, torch_gen, spec, profile=profile)
    normal = data.normal
    model.train()
    metrics: dict[str, float] = {}
    device = next(model.parameters()).device
    amp_enabled, amp_dtype = _precision(cfg, device)
    optimization_start = None
    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        optimization_start = time.perf_counter()
    for _ in range(cfg.train.gd_steps_per_iter):
        if normal:
            minibatch = sample_normal_minibatch(
                data.dataset, bundle, cfg.train.minibatch_size, torch_gen
            )
        else:
            minibatch = sample_minibatch(
                data.tree,
                data.targets,
                data.alias,
                data.policy_cache,
                data.decision_reach,
                bundle,
                cfg.train.minibatch_size,
                torch_gen,
            )
        optimizer.zero_grad(set_to_none=True)
        planner = getattr(bundle.rollout_backend, "plan_policy_batches", None)
        plans = (
            planner(source, minibatch.state_refs, "training")
            if planner is not None
            else [torch.arange(minibatch.state_refs.numel(), device=device)]
        )
        metric_parts: list[dict[str, Tensor]] = []
        effective_rows = int(minibatch.state_refs.numel())
        if effective_rows == 0:
            raise ValueError("cannot optimize an empty effective minibatch")
        for rows in plans:
            physical = (
                index_normal_minibatch(minibatch, rows)
                if normal
                else index_minibatch(minibatch, rows)
            )
            with torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                output = bundle.rollout_backend.policy_forward(
                    source, physical.state_refs, physical.reach
                )
                loss_sum, stats = spec.fn(
                    output,
                    physical,
                    physical,
                    bundle.policy_adapter,
                    cfg,
                    iteration=iteration,
                    reduction="sum",
                )
                loss = loss_sum / effective_rows
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            metric_parts.append(stats)
        _sync_scaled_gradients(model, int(os.environ.get("WORLD_SIZE", "1")))
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        metrics = finalize_metric_stats(metric_parts)
    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        assert optimization_start is not None
        metrics.update(data.profile_metrics or {})
        metrics["phase_optimization_seconds"] = time.perf_counter() - optimization_start
        if device.type == "cuda":
            metrics["phase_optimization_peak_allocated_bytes"] = float(
                torch.cuda.max_memory_allocated(device)
            )
            metrics["phase_optimization_peak_reserved_bytes"] = float(
                torch.cuda.max_memory_reserved(device)
            )
        phase_seconds = sum(
            value
            for key, value in metrics.items()
            if key.startswith("phase_") and key.endswith("_seconds")
        )
        decisions = metrics.get("decisions", 0.0)
        metrics["profiled_iteration_seconds"] = phase_seconds
        metrics["decisions_per_second"] = decisions / max(phase_seconds, 1.0e-12)
        metrics["target_bytes_per_decision"] = metrics.get(
            "retained_target_bytes", 0.0
        ) / max(decisions, 1.0)
        completion_seconds = metrics.get("phase_completion_seconds", 0.0)
        if completion_seconds:
            metrics["completion_effective_samples_per_second"] = metrics.get(
                "completion_effective_samples", 0.0
            ) / completion_seconds
        rollout_seconds = metrics.get("phase_rollout_seconds", 0.0)
        if rollout_seconds and data.normal:
            metrics["terminal_samples_per_second"] = metrics.get(
                "terminal_samples", 0.0
            ) / rollout_seconds
    metrics.update(data.completion_quality)
    if batch_metrics:
        records = list(batch_metrics)
        metrics.update(
            {
                "policy_physical_batches": float(len(records)),
                "policy_max_padded_tokens": float(
                    max(int(record["padded_tokens"]) for record in records)
                ),
                "policy_max_padded_token_pairs": float(
                    max(int(record["padded_token_pairs"]) for record in records)
                ),
            }
        )
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
    _hardware_metadata(cfg, requested, world)
    torch_gen, np_gen = seed_all(cfg.train.seed, rank, requested)
    bundle = build_game_bundle(cfg, requested)
    device = torch.device(bundle.rollout_backend.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if torch_gen.device != device:
        torch_gen, np_gen = seed_all(cfg.train.seed, rank, device)
    bundle.model.to(device)
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        bundle.model = nn.parallel.DistributedDataParallel(
            bundle.model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    local_cfg = copy.deepcopy(cfg)
    optimizer = torch.optim.Adam(
        bundle.model.parameters(), lr=cfg.train.lr, betas=cfg.train.adam_betas
    )
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
        )
        if checkpoint_metadata["cfg_dict"] != _scientific_config(cfg):
            raise RuntimeError("resume configuration does not match the checkpoint")
        start_iteration = int(checkpoint_metadata["iter"])
        if not 0 <= start_iteration <= cfg.train.iters:
            raise RuntimeError("checkpoint iteration is outside the configured schedule")
        run_id = checkpoint_metadata["run_id"]
        if not run_id:
            raise RuntimeError("checkpoint predates run identities and cannot be resumed safely")

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
        metrics = _one_iteration(
            local_cfg, bundle, optimizer, scaler, torch_gen, iteration
        )
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
            )
    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
