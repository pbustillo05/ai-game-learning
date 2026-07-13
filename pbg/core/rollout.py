"""Device-resident sequential rollout and unified tree-sampling entry point."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch, GameBundle
from pbg.core.tree import DeviceTreeBuffer, PolicyCache, ReachCache, TreeBatch


def _selected_policy_logp(
    adapter: Any,
    output: Any,
    selected: Tensor,
    legal_ids: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    """Return selected all-type log probabilities as fp32 ``[B,S,X]``."""

    result = adapter.log_prob(output, selected, legal_ids, legal_mask)
    if result.ndim != 3 or result.shape[:2] != selected.shape:
        raise ValueError("adapter.log_prob must return selected rows [B,S,X]")
    return result.float()  # [B,S,X]


def _combine_events(
    size: int,
    parts: list[tuple[Tensor, EventBatch]],
    device: torch.device,
) -> EventBatch:
    """Scatter partitioned opaque event rows back to frontier order ``[B]``."""

    public_id = torch.empty(size, dtype=torch.int32, device=device)  # [B]
    payload: dict[str, Tensor] = {}
    for positions, events in parts:
        public_id[positions] = events.public_id
        for key, value in events.payload.items():
            if key not in payload:
                payload[key] = torch.zeros(
                    (size, *value.shape[1:]), dtype=value.dtype, device=device
                )  # [B,...]
            payload[key][positions] = value
    return EventBatch(public_id=public_id, payload=payload)


def _sample_trees_sequential(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    return_profiles: bool = False,
) -> tuple[TreeBatch, ReachCache, PolicyCache] | tuple[
    TreeBatch, ReachCache, PolicyCache, Tensor
]:
    """Sample sequential public trees with one private profile per root."""

    backend = bundle.rollout_backend
    rollout_cfg = getattr(cfg, "rollout", cfg)
    device = torch.device(backend.device)
    if torch.device(gen.device) != device:
        raise ValueError("rollout generator must be created on the backend device")
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
    if n_trees:
        with torch.no_grad():
            root_profiles = backend.sample_profiles(roots, root_reach, gen)  # [G,P]
    else:
        root_profiles = torch.empty(
            (0, p_max), dtype=torch.int32, device=device
        )  # [0,P]
    if (
        root_profiles.shape != (n_trees, p_max)
        or root_profiles.dtype != torch.int32
        or root_profiles.device != device
    ):
        raise ValueError("sample_profiles must return device-local int32 [B,P]")
    reach_rows: Tensor | None = None
    if rollout_cfg.reach_storage != "recompute":
        reach_rows = torch.empty(
            (buffer.node_capacity, p_max, x_max), dtype=torch.float32, device=device
        )  # [capacity,P,X]
        reach_rows[:n_trees] = root_reach

    frontier_nodes, frontier_states = root_nodes, roots
    frontier_owner, frontier_reach = root_owner, root_reach
    frontier_profiles = root_profiles
    policy_nodes: list[Tensor] = []
    policy_logp: list[Tensor] = []
    policy_values: list[Tensor] = []
    for _ in range(int(backend.max_rollout_steps)):
        live = (frontier_owner != TERMINAL) & (frontier_owner != PRETERM)
        active_nodes = frontier_nodes[live]  # [A]
        active_states = frontier_states[live]  # [A]
        active_owner = frontier_owner[live]  # [A]
        active_reach = frontier_reach[live]  # [A,P,X]
        active_profiles = frontier_profiles[live]  # [A,P]
        active_count = active_nodes.numel()
        if active_count == 0:
            frontier_nodes, frontier_states = active_nodes, active_states
            frontier_owner, frontier_reach = active_owner, active_reach
            frontier_profiles = active_profiles
            break

        players = ((active_owner >= 0) & (active_owner < p_max)).nonzero().squeeze(1)
        chances = (active_owner == CHANCE).nonzero().squeeze(1)
        next_states = torch.empty_like(active_states)  # [A]
        next_reach = torch.empty_like(active_reach)  # [A,P,X]
        event_parts: list[tuple[Tensor, EventBatch]] = []
        if players.numel():
            states, reaches = active_states[players], active_reach[players]
            legal_ids, legal_mask = backend.legal_events(states)
            with torch.no_grad():
                output = backend.policy_forward(bundle.model, states, reaches)
                events = backend.sample_events(
                    states, active_profiles[players], output, bundle.policy_adapter, gen
                )
                updated = backend.reach_update(reaches, states, events, output)
            next_states[players] = backend.step(states, events)
            next_reach[players] = updated
            event_parts.append((players, events))
            policy_nodes.append(active_nodes[players])
            policy_logp.append(
                _selected_policy_logp(
                    bundle.policy_adapter,
                    output,
                    events.public_id[:, None],
                    legal_ids,
                    legal_mask,
                ).detach()
            )
            policy_values.append(output.v.detach().float())
        if chances.numel():
            states, reaches = active_states[chances], active_reach[chances]
            events = backend.sample_events(
                states,
                active_profiles[chances],
                None,
                bundle.policy_adapter,
                gen,
            )
            next_states[chances] = backend.step(states, events)
            next_reach[chances] = backend.reach_update(reaches, states, events, None)
            event_parts.append((chances, events))
        if players.numel() + chances.numel() != active_count:
            raise RuntimeError("backend classified a live state with an invalid owner")
        events = _combine_events(active_count, event_parts, device)
        next_owner = backend.classify(next_states).to(torch.int32)  # [A]
        children = buffer.append_children(
            active_nodes,
            next_states,
            next_owner,
            events.public_id,
            torch.ones(active_count, dtype=torch.float32, device=device),
        )  # [A]
        if reach_rows is not None:
            reach_rows[children.to(torch.int64)] = next_reach
        frontier_nodes, frontier_states = children, next_states
        frontier_owner, frontier_reach = next_owner, next_reach
        frontier_profiles = active_profiles

    if bool(((frontier_owner != TERMINAL) & (frontier_owner != PRETERM)).any()):
        raise RuntimeError("rollout_backend.max_rollout_steps is too small; live frontier remains")
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
        rollout_cfg.reach_storage,
        root_reach if reach_rows is None else reach_rows[reach_nodes.to(torch.int64)],
        root_nodes if reach_rows is None else reach_nodes,
    )
    if policy_nodes:
        node_ids = torch.cat(policy_nodes).to(torch.int32)  # [Nd]
        old_logp = torch.cat(policy_logp).float()  # [Nd,1,X]
        old_v = torch.cat(policy_values).float()  # [Nd,X]
        order = torch.argsort(node_ids.to(torch.int64))
        node_ids, old_logp, old_v = node_ids[order], old_logp[order], old_v[order]
    else:
        node_ids = torch.empty(0, dtype=torch.int32, device=device)
        old_logp = torch.empty((0, 1, x_max), dtype=torch.float32, device=device)
        old_v = torch.empty((0, x_max), dtype=torch.float32, device=device)
    result = (tree, reach_cache, PolicyCache(node_ids, old_logp, old_v))
    if return_profiles:
        return (*result, root_profiles)
    return result


def sample_trees(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Sample sequential or fixed-intensity labeled trees."""

    rollout_cfg = getattr(cfg, "rollout", cfg)
    training = getattr(cfg, "training", None)
    mode = getattr(training, "mode", getattr(rollout_cfg, "beta_mode", None))
    if mode in {"rao_blackwell", "sequential"}:
        return _sample_trees_sequential(bundle, rollout_cfg, gen)
    if mode in {"branched_rao_blackwell", "fixed"}:
        from pbg.core.branching import _sample_branched_labeled

        return _sample_branched_labeled(bundle, rollout_cfg, gen)
    if training is None:
        raise ValueError("legacy beta_mode must be 'sequential' or 'fixed'")
    raise ValueError("sample_trees requires an RB training mode")
