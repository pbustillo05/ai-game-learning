"""Minimal rollout/target/optimization training loop with DDP checkpoints."""

from __future__ import annotations

import copy
import importlib
import json
import os
import platform
from dataclasses import asdict
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
    build_normal_dataset,
    index_normal_minibatch,
    sample_normal_minibatch,
)
from pbg.core.rollout import sample_trees
from pbg.core.tree import build_decision_alias
from pbg.core.values import HeadEvals, branched_value


GAME_FACTORIES = {
    "synthetic": "pbg.games.synthetic.bundle:build_bundle",
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
) -> dict[str, Any]:
    """Restore checkpoint state and return iteration/config metadata."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    if torch_gen is not None:
        torch_gen.set_state(payload["torch_generator"].cpu())
    if np_gen is not None:
        np_gen.bit_generator.state = payload["numpy_generator"]
    torch.set_rng_state(payload["torch_cpu_rng"].cpu())
    if torch.cuda.is_available() and payload["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
    return {"iter": int(payload["iteration"]), "cfg_dict": payload["config"]}


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


def _one_iteration(
    cfg: RootConfig,
    bundle: GameBundle,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    torch_gen: torch.Generator,
    iteration: int,
) -> dict[str, float]:
    spec = _loss_spec(cfg)
    batch_metrics = getattr(bundle.rollout_backend, "policy_batch_metrics", None)
    if batch_metrics is not None:
        batch_metrics.clear()
    model = bundle.model
    source = model.module if hasattr(model, "module") else model
    frozen_model = copy.deepcopy(source).eval()
    frozen_bundle = GameBundle(
        model=frozen_model,
        rollout_backend=bundle.rollout_backend,
        policy_adapter=bundle.policy_adapter,
        completion_backend=bundle.completion_backend,
    )
    normal = cfg.training.mode == "normal"
    with torch.no_grad():
        if normal:
            dataset = build_normal_dataset(frozen_bundle, cfg, torch_gen)
        else:
            tree, reach_cache, policy_cache = sample_trees(frozen_bundle, cfg, torch_gen)
            leaf_ids = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero(as_tuple=False).squeeze(1)
            leaf_reach = reach_cache.rows_for(
                tree, leaf_ids.to(torch.int32), bundle.rollout_backend, policy_cache
            )
            if bundle.completion_backend is None or cfg.completion.n_samples is None:
                raise RuntimeError("RB modes require a completion backend and sample budget")
            completion = bundle.completion_backend.complete(
                bundle.rollout_backend,
                tree,
                leaf_ids.to(torch.int32),
                leaf_reach,
                cfg.completion.n_samples,
                torch_gen,
            )
            heads = HeadEvals(node_ids=policy_cache.node_ids, v=policy_cache.old_v)
            targets = branched_value(
                tree,
                completion,
                reach_cache,
                heads,
                cfg.loss.lam,
                bundle.rollout_backend,
                configured_node_budget=cfg.rollout.node_budget,
                emit_q="q_hat" in spec.required_targets,
            )
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
            alias = build_decision_alias(tree)
    model.train()
    metrics: dict[str, float] = {}
    device = next(model.parameters()).device
    amp_enabled, amp_dtype = _precision(cfg, device)
    for _ in range(cfg.train.gd_steps_per_iter):
        if normal:
            minibatch = sample_normal_minibatch(
                dataset, bundle, cfg.train.minibatch_size, torch_gen
            )
        else:
            minibatch = sample_minibatch(
                tree,
                targets,
                alias,
                policy_cache,
                reach_cache,
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


def train(cfg: RootConfig) -> None:
    """Run independent rank-local rollouts with gradient-only DDP synchronization."""

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
    _hardware_metadata(cfg, requested, world)
    bundle = build_game_bundle(cfg, requested)
    device = torch.device(bundle.rollout_backend.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
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
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata = _hardware_metadata(cfg, device, world)
        with (out_dir / "hardware.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
    for iteration in range(cfg.train.iters):
        metrics = _one_iteration(
            local_cfg, bundle, optimizer, scaler, torch_gen, iteration
        )
        if rank == 0:
            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"iter": iteration, **metrics}, sort_keys=True) + "\n")
            checkpoint_every = (
                cfg.train.checkpoint_every
                if cfg.hardware is None or cfg.hardware.checkpoint_every is None
                else cfg.hardware.checkpoint_every
            )
            if (iteration + 1) % checkpoint_every == 0:
                save_checkpoint(
                    out_dir / f"ckpt_{iteration + 1:07d}.pt",
                    bundle.model,
                    optimizer,
                    scaler,
                    iteration + 1,
                    cfg,
                    torch_gen,
                    np_gen,
                )
    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
