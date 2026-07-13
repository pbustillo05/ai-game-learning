"""Fixed-intensity duplicate-preserving labeled branching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor

from pbg.core.config import RolloutConfig
from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch, GameBundle
from pbg.core.tree import (
    DeviceTreeBuffer,
    PolicyCache,
    ReachCache,
    TreeBatch,
    TreeBuilder,
    _replay_reach_update,
)


def _combine_event_rows(
    size: int,
    parts: list[tuple[Tensor, EventBatch]],
    device: torch.device,
) -> EventBatch:
    """Scatter partitioned event batches into abstract-child order ``[C]``."""

    public_id = torch.empty(size, dtype=torch.int32, device=device)  # [C]
    payload: dict[str, Tensor] = {}
    for positions, events in parts:
        public_id[positions] = events.public_id
        for key, value in events.payload.items():
            if key not in payload:
                payload[key] = torch.zeros(
                    (size, *value.shape[1:]), dtype=value.dtype, device=device
                )  # [C,...]
            payload[key][positions] = value
    return EventBatch(public_id=public_id, payload=payload)


def _validate_beta(beta: Tensor, count: int, device: torch.device) -> Tensor:
    """Validate and return fixed intensities as fp32 ``[B]``."""

    if beta.shape != (count,) or beta.dtype != torch.float32 or beta.device != device:
        raise ValueError("beta_fn must return device-local fp32 [B]")
    if not bool(torch.isfinite(beta).all()) or bool((beta < 1.0).any()):
        raise ValueError("rollout beta must be finite and at least one")
    return beta


def _offspring_counts(beta: Tensor, gen: torch.Generator) -> Tensor:
    """Randomized-round fp32 intensities to int64 offspring counts ``[B]``."""

    base = torch.floor(beta)
    fraction = beta - base
    # Integer intensities are deterministic and consume no unnecessary random draw.
    fractional = fraction > 0
    extra = torch.zeros_like(base)
    if bool(fractional.any()):
        extra[fractional] = torch.bernoulli(fraction[fractional], generator=gen)
    return (base + extra).to(torch.int64)  # [B]


def _sample_branched_labeled(
    bundle: GameBundle,
    cfg: RolloutConfig,
    gen: torch.Generator,
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Sample fixed-intensity labeled public trees on the backend device.

    The returned tree contains ``N`` nodes and ``C`` duplicate-preserving edges;
    reach rows are ``[N_stored,P,X_max]`` and cached decision rows are
    ``[N_d,K_max,X_max]``.
    """

    backend = bundle.rollout_backend
    rollout_cfg = getattr(cfg, "rollout", cfg)
    device = torch.device(backend.device)
    if torch.device(gen.device) != device:
        raise ValueError("rollout generator must be created on the backend device")
    beta_mode = getattr(rollout_cfg, "beta_mode", None)
    if beta_mode not in {"sequential", "fixed"}:
        raise ValueError("beta_mode must be 'sequential' or 'fixed'")
    if rollout_cfg.reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ValueError("unknown reach_storage mode")
    n_trees = int(rollout_cfg.n_trees)
    node_budget = int(rollout_cfg.node_budget)
    if n_trees < 0 or node_budget <= 0:
        raise ValueError("n_trees must be nonnegative and node_budget positive")

    p_max = int(backend.n_players)
    x_sizes = tuple(int(value) for value in backend.x_sizes)
    x_max = max(x_sizes)
    roots = backend.root_states(n_trees)
    if roots.shape != (n_trees,) or roots.dtype != torch.int32 or roots.device != device:
        raise ValueError("root_states must return device-local int32 [n_trees]")
    root_owner = backend.classify(roots).to(torch.int32)  # [G]
    buffer = DeviceTreeBuffer(n_trees, node_budget, p_max, device)
    root_nodes = buffer.add_roots(roots, root_owner)  # [G]

    root_reach = torch.zeros(
        (n_trees, p_max, x_max), dtype=torch.float32, device=device
    )  # [G,P,X]
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    reach_rows = torch.empty(
        (buffer.node_capacity, p_max, x_max), dtype=torch.float32, device=device
    )  # [capacity,P,X]
    reach_rows[:n_trees] = root_reach
    edge_logp = torch.full(
        (buffer.edge_capacity, x_max), -torch.inf, dtype=torch.float32, device=device
    )  # [edge_capacity,X]
    node_value = torch.zeros(
        (buffer.node_capacity, x_max), dtype=torch.float32, device=device
    )  # [node_capacity,X]

    frontier_nodes = root_nodes
    frontier_states = roots
    frontier_owner = root_owner
    frontier_reach = root_reach
    for _ in range(int(backend.max_rollout_steps)):
        live = (frontier_owner != TERMINAL) & (frontier_owner != PRETERM)
        if not bool(live.any()):
            break
        active_nodes = frontier_nodes[live]  # [A]
        active_states = frontier_states[live]  # [A]
        active_owner = frontier_owner[live]  # [A]
        active_reach = frontier_reach[live]  # [A,P,X]
        active_count = active_nodes.numel()

        if beta_mode == "sequential":
            beta = torch.ones(active_count, dtype=torch.float32, device=device)  # [A]
        else:
            beta = _validate_beta(backend.beta_fn(active_states), active_count, device)
        offspring = _offspring_counts(beta, gen)  # [A]
        parent_positions = torch.repeat_interleave(
            torch.arange(active_count, device=device), offspring
        )  # [C_new]
        parents = active_nodes[parent_positions]  # [C_new]
        states = active_states[parent_positions]  # [C_new]
        reaches = active_reach[parent_positions]  # [C_new,P,X]
        owners = active_owner[parent_positions]  # [C_new]
        child_count = parents.numel()
        profiles = backend.sample_profiles(states, reaches, gen)  # [C_new,P]

        player_positions = ((owners >= 0) & (owners < p_max)).nonzero().squeeze(1)
        chance_positions = (owners == CHANCE).nonzero().squeeze(1)
        event_parts: list[tuple[Tensor, EventBatch]] = []
        next_reach = torch.empty_like(reaches)  # [C_new,P,X]
        player_logp: Tensor | None = None
        player_output: Any | None = None

        if player_positions.numel() != 0:
            player_states = states[player_positions]  # [Cp]
            player_reach = reaches[player_positions]  # [Cp,P,X]
            with torch.no_grad():
                player_output = backend.policy_forward(
                    bundle.model, player_states, player_reach
                )
                player_events = backend.sample_events(
                    player_states,
                    profiles[player_positions],
                    player_output,
                    bundle.policy_adapter,
                    gen,
                )
                next_reach[player_positions] = backend.reach_update(
                    player_reach, player_states, player_events, player_output
                )
                legal_ids, legal_mask = backend.legal_events(player_states)
                from pbg.core.rollout import _selected_policy_logp

                player_logp = _selected_policy_logp(
                    bundle.policy_adapter,
                    player_output,
                    player_events.public_id[:, None],
                    legal_ids,
                    legal_mask,
                ).detach()[:, 0]  # [Cp,X]
            event_parts.append((player_positions, player_events))

        if chance_positions.numel() != 0:
            chance_states = states[chance_positions]  # [Cc]
            chance_events = backend.sample_events(
                chance_states,
                profiles[chance_positions],
                None,
                bundle.policy_adapter,
                gen,
            )
            next_reach[chance_positions] = backend.reach_update(
                reaches[chance_positions], chance_states, chance_events, None
            )
            event_parts.append((chance_positions, chance_events))

        if player_positions.numel() + chance_positions.numel() != child_count:
            raise RuntimeError("backend classified a live state with an invalid owner")
        events = _combine_event_rows(child_count, event_parts, device)
        next_states = backend.step(states, events)  # [C_new]
        next_owner = backend.classify(next_states).to(torch.int32)  # [C_new]
        weights = offspring[parent_positions].reciprocal().to(torch.float32)  # [C_new]
        first_edge = buffer.edge_count
        child_nodes = buffer.append_children(
            parents, next_states, next_owner, events.public_id, weights
        )  # [C_new]
        reach_rows[child_nodes.to(torch.int64)] = next_reach

        if player_positions.numel() != 0:
            assert player_logp is not None and player_output is not None
            edge_rows = torch.arange(
                first_edge, first_edge + child_count, dtype=torch.int64, device=device
            )  # [C_new]
            edge_logp[edge_rows[player_positions]] = player_logp
            decision_parent_positions = (
                (active_owner >= 0) & (active_owner < p_max)
            ).nonzero().squeeze(1)
            player_counts = offspring[decision_parent_positions]  # [Nd_wave]
            value_rows = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(player_counts, dim=0)[:-1],
                )
            )  # [Nd_wave]
            node_value[active_nodes[decision_parent_positions].to(torch.int64)] = (
                player_output.v[value_rows].detach().float()
            )

        frontier_nodes = child_nodes
        frontier_states = next_states
        frontier_owner = next_owner
        frontier_reach = next_reach

    live_frontier = (frontier_owner != TERMINAL) & (frontier_owner != PRETERM)
    if bool(live_frontier.any()):
        raise RuntimeError(
            "rollout_backend.max_rollout_steps is too small; live frontier remains"
        )

    tree = buffer.finalize()
    all_nodes = torch.arange(buffer.node_count, dtype=torch.int32, device=device)  # [N]
    if rollout_cfg.reach_storage == "full":
        reach_nodes = all_nodes
    elif rollout_cfg.reach_storage == "decisions_only":
        stored = ((tree.owner >= 0) & (tree.owner < p_max)) | (
            (tree.owner == TERMINAL) | (tree.owner == PRETERM)
        )
        reach_nodes = all_nodes[stored]
    else:
        reach_nodes = root_nodes
    reach_cache = ReachCache(
        mode=rollout_cfg.reach_storage,
        log_r=reach_rows[reach_nodes.to(torch.int64)],
        node_index=reach_nodes,
    )

    decision_nodes = ((tree.owner >= 0) & (tree.owner < p_max)).nonzero().squeeze(1)
    if decision_nodes.numel() != 0:
        counts = tree.child_ptr[decision_nodes + 1] - tree.child_ptr[decision_nodes]  # [Nd]
        width = int(counts.max())
        logp_child = torch.full(
            (decision_nodes.numel(), width, x_max),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )  # [Nd,K_max,X]
        edge_parent = buffer.edge_parent[: buffer.edge_count].to(torch.int64)  # [C]
        player_edge = (tree.owner[edge_parent] >= 0) & (tree.owner[edge_parent] < p_max)
        edge_ids = torch.arange(buffer.edge_count, device=device)[player_edge]  # [Cd]
        parent_ids = edge_parent[player_edge]  # [Cd]
        node_to_row = torch.full(
            (buffer.node_count,), -1, dtype=torch.int64, device=device
        )  # [N]
        node_to_row[decision_nodes] = torch.arange(decision_nodes.numel(), device=device)
        columns = edge_ids - tree.child_ptr[parent_ids].to(torch.int64)  # [Cd]
        logp_child[node_to_row[parent_ids], columns] = edge_logp[edge_ids]
        v_old = node_value[decision_nodes]  # [Nd,X]
    else:
        logp_child = torch.empty((0, 0, x_max), dtype=torch.float32, device=device)
        v_old = torch.empty((0, x_max), dtype=torch.float32, device=device)
    return tree, reach_cache, PolicyCache(
        node_ids=decision_nodes.to(torch.int32),
        old_logp_child=logp_child,
        old_v=v_old,
    )
