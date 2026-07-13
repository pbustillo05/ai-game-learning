"""Minimal rollout/target/optimization training loop with DDP checkpoints."""

from __future__ import annotations

import copy
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from pbg.core.config import RootConfig, seed_all
from pbg.core.game import GameBundle
from pbg.core.losses import COMPOSITES, LossSpec
from pbg.core.minibatch import sample_minibatch
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
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": int(iteration),
        "config": asdict(cfg),
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


def _one_iteration(
    cfg: RootConfig,
    bundle: GameBundle,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    torch_gen: torch.Generator,
    iteration: int,
) -> dict[str, float]:
    spec = _loss_spec(cfg)
    model = bundle.model
    source = model.module if hasattr(model, "module") else model
    frozen_model = copy.deepcopy(source).eval()
    frozen_bundle = GameBundle(
        model=frozen_model,
        rollout_backend=bundle.rollout_backend,
        policy_adapter=bundle.policy_adapter,
        completion_backend=bundle.completion_backend,
    )
    with torch.no_grad():
        tree, reach_cache, policy_cache = sample_trees(frozen_bundle, cfg, torch_gen)
        leaf_ids = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero(as_tuple=False).squeeze(1)
        leaf_reach = reach_cache.rows_for(
            tree, leaf_ids.to(torch.int32), bundle.rollout_backend, policy_cache
        )
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
        name for name in spec.required_targets if not hasattr(targets, name) or getattr(targets, name) is None
    }
    if unavailable:
        raise ValueError(f"loss {cfg.loss.name!r} requires unavailable targets {sorted(unavailable)}")

    alias = build_decision_alias(tree)
    model.train()
    metrics: dict[str, float] = {}
    device = next(model.parameters()).device
    amp_enabled = bool(cfg.train.amp and device.type == "cuda")
    for _ in range(cfg.train.gd_steps_per_iter):
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
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
            output = bundle.rollout_backend.policy_forward(
                model, minibatch.state_refs, minibatch.reach
            )
            loss, metrics = spec.fn(
                output, minibatch, minibatch, bundle.policy_adapter, cfg, iteration=iteration
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
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
    scaler = _scaler(bool(cfg.train.amp and device.type == "cuda"))
    out_dir = Path(cfg.train.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(cfg.train.iters):
        metrics = _one_iteration(
            local_cfg, bundle, optimizer, scaler, torch_gen, iteration
        )
        if rank == 0:
            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"iter": iteration, **metrics}, sort_keys=True) + "\n")
            if (iteration + 1) % cfg.train.checkpoint_every == 0:
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
