"""Training, checkpointing, DDP, and game wiring.

``train`` enables deterministic Torch algorithms and disables cuDNN benchmark
selection.  This makes equal-seed single-rank runs exactly reproducible, at the
cost of slower kernels (and an explicit error if Torch has no deterministic
implementation for an operation).
"""

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
from torch import Tensor, nn

from pbg.core.branching import IntensityController
from pbg.core.completion import BatchedCompletionStats
from pbg.core.config import RootConfig, seed_all
from pbg.core.losses import COMPOSITES
from pbg.core.minibatch import MinibatchSample, sample_minibatch
from pbg.core.rollout import sample_trees
from pbg.core.tree import build_decision_alias
from pbg.core.values import HeadEvals, branched_value


def build_game_bundle(cfg: RootConfig) -> tuple[Any, nn.Module, Any, Any, Any]:
    """Construct the game-owned pieces without importing poker into core."""

    if cfg.game.name != "synthetic":
        raise NotImplementedError("poker training wiring is implemented by P13")
    # Dynamic loading preserves core's static dependency direction while the
    # game factory remains the single P11/P13 extension point.
    family = importlib.import_module("pbg.games.synthetic.family")
    model_module = importlib.import_module("pbg.games.synthetic.model")

    values = dict(cfg.game.synthetic or {})
    values["board_per_round"] = tuple(values["board_per_round"])
    values["bet_size"] = tuple(values["bet_size"])
    game = family.SyntheticPBG(family.SyntheticDialConfig(**values))
    model = model_module.SyntheticNet(max(game.x_sizes), game.n_player_events, cfg.model)
    adapter = model_module.SyntheticPolicyAdapter()
    return game, model, adapter, model, IntensityController.synthetic_bucket


def _legal_head_mask(game: Any, refs: Tensor, width: int, device: torch.device) -> Tensor:
    event_ids, event_mask = game.legal_events(refs.cpu())
    result = torch.zeros((refs.numel(), width), dtype=torch.bool, device=device)
    rows = torch.arange(refs.numel())[:, None].expand_as(event_ids)
    selected = event_ids[event_mask].long()
    if selected.numel() and bool((selected >= width).any()):
        raise ValueError("model event head is smaller than a legal event id")
    result[rows[event_mask].to(device), selected.to(device)] = True
    return result


def _head_evals(game: Any, model: nn.Module, tree: Any, use_moments: bool) -> HeadEvals:
    nodes = ((tree.owner >= 0) & (tree.owner < game.n_players)).nonzero(
        as_tuple=False
    ).squeeze(1)
    device = next(model.parameters()).device
    refs = tree.state_ref[nodes].to(device)
    raw_model = model.module if hasattr(model, "module") else model
    mask = _legal_head_mask(game, refs, raw_model.e_max, device)
    with torch.no_grad():
        out = model(game.encode_public(refs), mask)
    return HeadEvals(
        node_ids=nodes.to(torch.int32), v=out.v.float().cpu(),
        vvec=out.vvec.float().cpu() if use_moments and out.vvec is not None else None,
        u=out.u.float().cpu() if use_moments and out.u is not None else None,
    )


def _complete_terminal_leaves(
    game: Any,
    tree: Any,
    reach: Any,
    policy_cache: Any,
    n_samples: int,
    gen: torch.Generator,
) -> BatchedCompletionStats:
    """Synthetic collision-conditioned MC completion for stored terminal leaves."""

    leaf_ids = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero(
        as_tuple=False
    ).squeeze(1)
    if n_samples <= 0:
        raise ValueError("completion.n_samples must be positive")
    if not bool(game.is_terminal(tree.state_ref[leaf_ids]).all()):
        raise NotImplementedError(
            "synthetic PRETERM completion is not needed by the shipped Kuhn smoke config"
        )
    rows = reach.rows_for(tree, leaf_ids, game, policy_cache)
    profiles = game.sample_profiles(tree.state_ref[leaf_ids], rows, n_samples, gen)
    leaves = int(leaf_ids.numel())
    players, x_max = game.n_players, max(game.x_sizes)
    p_hat = torch.zeros((leaves, players, x_max), dtype=torch.float32)
    v_hat = torch.zeros_like(p_hat)
    vvec = torch.zeros((leaves, players), dtype=torch.float32)
    u = torch.zeros(leaves, dtype=torch.float32)
    for row in range(leaves):
        refs = tree.state_ref[leaf_ids[row]].repeat(n_samples)
        rewards = game.reward(profiles[row], refs).float()
        vvec[row] = rewards.mean(0)
        u[row] = rewards.square().sum(-1).mean()
        for player, width in enumerate(game.x_sizes):
            types = profiles[row, :, player].long()
            counts = torch.bincount(types, minlength=width).float()
            p_hat[row, player, :width] = counts / n_samples
            for xtype in (counts > 0).nonzero(as_tuple=False).squeeze(1).tolist():
                chosen = types == xtype
                v_hat[row, player, xtype] = rewards[chosen, player].mean()
    return BatchedCompletionStats(
        p_hat=p_hat, v_hat=v_hat, vvec_hat=vvec, u_hat=u,
        n_eff=torch.full((leaves,), float(n_samples)),
        metadata={"node_ids": leaf_ids.to(torch.int32)},
    )


def _forward_sample(game: Any, model: nn.Module, sample: MinibatchSample) -> Any:
    device = next(model.parameters()).device
    refs = sample.state_refs.to(device)
    raw_model = model.module if hasattr(model, "module") else model
    mask = _legal_head_mask(game, refs, raw_model.e_max, device)
    return model(game.encode_public(refs), mask)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: Any,
    controller: IntensityController,
    iteration: int,
    cfg: RootConfig,
    torch_gen: torch.Generator | None = None,
    np_gen: np.random.Generator | None = None,
) -> None:
    """Atomically store optimizer, controller, and all reproducibility state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": raw_model.state_dict(), "opt": opt.state_dict(),
        "scaler": scaler.state_dict(), "controller": controller.state_dict(),
        "iter": int(iteration), "cfg_dict": asdict(cfg),
        "torch_rng": torch.get_rng_state(),
        "torch_gen_rng": torch_gen.get_state() if torch_gen is not None else None,
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "np_rng_state": np_gen.bit_generator.state if np_gen is not None else None,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    opt: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    controller: IntensityController | None = None,
    torch_gen: torch.Generator | None = None,
    np_gen: np.random.Generator | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint and return its iteration/config metadata."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload["model"])
    if opt is not None:
        opt.load_state_dict(payload["opt"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    if controller is not None:
        controller.load_state_dict(payload["controller"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch_gen is not None and payload.get("torch_gen_rng") is not None:
        torch_gen.set_state(payload["torch_gen_rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if np_gen is not None and payload.get("np_rng_state") is not None:
        np_gen.bit_generator.state = payload["np_rng_state"]
    return {"iter": int(payload["iter"]), "cfg_dict": payload["cfg_dict"]}


def _scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _one_iteration(
    cfg: RootConfig, game: Any, model: nn.Module, adapter: Any,
    controller: IntensityController, opt: torch.optim.Optimizer, scaler: Any,
    torch_gen: torch.Generator, np_gen: np.random.Generator, iteration: int,
) -> dict[str, float]:
    model.eval()
    source = model.module if hasattr(model, "module") else model
    frozen = copy.deepcopy(source).eval()
    rollout_controller = None if cfg.rollout.beta_mode == "sequential" else controller
    with torch.no_grad():
        tree, reach, cache = sample_trees(
            game, frozen, adapter, cfg.rollout, None, rollout_controller, torch_gen, np_gen
        )
        completion = _complete_terminal_leaves(
            game, tree, reach, cache, cfg.completion.n_samples, torch_gen
        )
        heads = _head_evals(game, frozen, tree, cfg.loss.use_moment_heads)
        targets = branched_value(tree, completion, reach, heads, cfg.loss.lam, game)
    alias = build_decision_alias(tree)
    model.train()
    metrics: dict[str, float] = {}
    device = next(model.parameters()).device
    amp_enabled = bool(cfg.train.amp and device.type == "cuda")
    for _ in range(cfg.train.gd_steps_per_iter):
        sample = sample_minibatch(
            tree, targets, alias, cache, cfg.train.minibatch_size, torch_gen
        )
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = _forward_sample(game, model, sample)
            loss, metrics = COMPOSITES[cfg.loss.name](
                output, sample, sample, adapter, cfg, iteration=iteration
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(opt)
        scaler.update()
    controller.end_iteration({})
    return metrics


def train(cfg: RootConfig) -> None:
    """Run phase-alternating rollout/training under torchrun or one process."""

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if cfg.rollout.n_trees % world:
        raise ValueError("rollout.n_trees must be divisible by WORLD_SIZE")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch_gen, np_gen = seed_all(cfg.train.seed, rank)
    game, model, adapter, _, bucket_fn = build_game_bundle(cfg)
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            model = model.cuda(local_rank)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )
    elif torch.cuda.is_available():
        model = model.cuda()
    local_cfg = copy.deepcopy(cfg)
    local_cfg.rollout.n_trees //= world
    controller = IntensityController(
        local_cfg.rollout, bucket_fn, use_moment_heads=cfg.loss.use_moment_heads
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, betas=cfg.train.adam_betas)
    scaler = _scaler(bool(cfg.train.amp and torch.cuda.is_available()))
    out_dir = Path(cfg.train.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.jsonl"
    for iteration in range(cfg.train.iters):
        metrics = _one_iteration(
            local_cfg, game, model, adapter, controller, opt, scaler,
            torch_gen, np_gen, iteration,
        )
        if rank == 0:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"iter": iteration, **metrics}, sort_keys=True) + "\n")
            if (iteration + 1) % cfg.train.checkpoint_every == 0:
                save_checkpoint(
                    out_dir / f"ckpt_{iteration + 1:07d}.pt", model, opt, scaler,
                    controller, iteration + 1, cfg, torch_gen, np_gen,
                )
