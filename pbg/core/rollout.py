"""Frontier-based sequential public-tree rollout and cache construction."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch, GameBundle
from pbg.core.tree import (
    DeviceTreeBuffer,
    PolicyCache,
    ReachCache,
    TreeBatch,
    TreeBuilder,
    _replay_reach_update,
)


def _selected_policy_logp(
    adapter: Any,
    output: Any,
    selected: Tensor,
    legal_ids: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    """Return selected all-type log probabilities as fp32 ``[B,S,X]``."""

    try:
        result = adapter.log_prob(output, selected, legal_ids, legal_mask)
    except TypeError:
        try:
            result = adapter.log_prob(output, selected, legal_mask)
        except TypeError:
            result = adapter.log_prob(output, selected)
    if result.ndim != 3:
        raise ValueError("adapter.log_prob must return [B,S,X]")
    if result.shape[1] != selected.shape[1] and result.shape[2] == selected.shape[1]:
        result = result.transpose(1, 2)
    if result.shape[:2] != selected.shape:
        raise ValueError("adapter.log_prob is not aligned with selected events")
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
                )
            payload[key][positions] = value
    return EventBatch(public_id=public_id, payload=payload)


def sample_trees_device(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Sample sequential public trees entirely on the backend device.

    The returned topology has ``N`` nodes, reach rows have shape
    ``[N_stored,P,X_max]``, and cached decision rows are ``[N_d,1,X_max]``.
    """

    backend = bundle.rollout_backend
    rollout_cfg = getattr(cfg, "rollout", cfg)
    device = torch.device(backend.device)
    if torch.device(gen.device) != device:
        raise ValueError("rollout generator must be created on the backend device")
    if getattr(rollout_cfg, "beta_mode", "sequential") != "sequential":
        raise ValueError("sample_trees_device implements sequential rollout only")
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

    frontier_nodes = root_nodes
    frontier_states = roots
    frontier_owner = root_owner
    frontier_reach = root_reach
    policy_nodes: list[Tensor] = []
    policy_logp: list[Tensor] = []
    policy_values: list[Tensor] = []

    for _ in range(int(backend.max_rollout_steps)):
        live = (frontier_owner != TERMINAL) & (frontier_owner != PRETERM)  # [F]
        active_nodes = frontier_nodes[live]  # [A]
        active_states = frontier_states[live]  # [A]
        active_owner = frontier_owner[live]  # [A]
        active_reach = frontier_reach[live]  # [A,P,X]
        active_count = active_nodes.numel()
        if active_count == 0:
            frontier_nodes = active_nodes
            frontier_states = active_states
            frontier_owner = active_owner
            frontier_reach = active_reach
            continue

        player_positions = (
            (active_owner >= 0) & (active_owner < p_max)
        ).nonzero(as_tuple=False).squeeze(1)
        chance_positions = (active_owner == CHANCE).nonzero(as_tuple=False).squeeze(1)
        next_states = torch.empty_like(active_states)  # [A]
        next_reach = torch.empty_like(active_reach)  # [A,P,X]
        event_parts: list[tuple[Tensor, EventBatch]] = []

        if player_positions.numel() != 0:
            states = active_states[player_positions]  # [Bp]
            reaches = active_reach[player_positions]  # [Bp,P,X]
            legal_ids, legal_mask = backend.legal_events(states)
            with torch.no_grad():
                output = backend.policy_forward(bundle.model, states, reaches)
                profiles = backend.sample_profiles(states, reaches, gen)
                events = backend.sample_events(
                    states, profiles, output, bundle.policy_adapter, gen
                )
                updated = backend.reach_update(reaches, states, events, output)
            next_states[player_positions] = backend.step(states, events)
            next_reach[player_positions] = updated
            event_parts.append((player_positions, events))
            selected = events.public_id[:, None]  # [Bp,1]
            policy_nodes.append(active_nodes[player_positions])
            policy_logp.append(
                _selected_policy_logp(
                    bundle.policy_adapter, output, selected, legal_ids, legal_mask
                ).detach()
            )
            policy_values.append(output.v.detach().float())

        if chance_positions.numel() != 0:
            states = active_states[chance_positions]  # [Bc]
            reaches = active_reach[chance_positions]  # [Bc,P,X]
            profiles = backend.sample_profiles(states, reaches, gen)
            events = backend.sample_events(
                states, profiles, None, bundle.policy_adapter, gen
            )
            next_states[chance_positions] = backend.step(states, events)
            next_reach[chance_positions] = backend.reach_update(
                reaches, states, events, None
            )
            event_parts.append((chance_positions, events))

        if player_positions.numel() + chance_positions.numel() != active_count:
            raise RuntimeError("backend classified a live state with an invalid owner")
        events = _combine_events(active_count, event_parts, device)
        next_owner = backend.classify(next_states).to(torch.int32)  # [A]
        weights = torch.ones(active_count, dtype=torch.float32, device=device)  # [A]
        child_nodes = buffer.append_children(
            active_nodes, next_states, next_owner, events.public_id, weights
        )  # [A]
        reach_rows[child_nodes.to(torch.int64)] = next_reach
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

    if policy_nodes:
        node_index = torch.cat(policy_nodes).to(torch.int32)  # [Nd]
        logp_child = torch.cat(policy_logp, dim=0).float()  # [Nd,1,X]
        v_old = torch.cat(policy_values, dim=0).float()  # [Nd,X]
        order = torch.argsort(node_index.to(torch.int64))
        node_index = node_index[order]
        logp_child = logp_child[order]
        v_old = v_old[order]
    else:
        node_index = torch.empty(0, dtype=torch.int32, device=device)
        logp_child = torch.empty((0, 1, x_max), dtype=torch.float32, device=device)
        v_old = torch.empty((0, x_max), dtype=torch.float32, device=device)
    return tree, reach_cache, PolicyCache(logp_child, v_old, node_index)


def _classify(game: Any, state_refs: Tensor) -> Tensor:
    """Return owner codes with terminal/pre-terminal leaves taking precedence."""

    owner = game.owner_of(state_refs).to(torch.int32)
    terminal = game.is_terminal(state_refs)
    preterminal = game.is_pre_terminal(state_refs)
    owner = torch.where(terminal, torch.full_like(owner, TERMINAL), owner)
    owner = torch.where(preterminal & ~terminal, torch.full_like(owner, PRETERM), owner)
    return owner


def _model_device(policy_fn: Callable[..., Any], fallback: torch.device) -> torch.device:
    candidate = getattr(policy_fn, "parameters", None)
    if candidate is None:
        return fallback
    try:
        return next(candidate()).device
    except (StopIteration, TypeError):
        return fallback


def _call_policy(
    game: Any,
    policy_fn: Callable[..., Any],
    state_refs: Tensor,
    legal_ids: Tensor,
    legal_mask: Tensor,
) -> Any:
    """Support both a raw SyntheticNet and a game-owned one-argument wrapper."""

    device = _model_device(policy_fn, state_refs.device)
    refs = state_refs.to(device)
    encoded = game.encode_public(refs)
    head_width = int(getattr(policy_fn, "e_max", getattr(game, "n_player_events", 0)))
    if head_width <= 0:
        head_width = int(legal_ids[legal_mask].max()) + 1
    head_mask = torch.zeros((refs.numel(), head_width), dtype=torch.bool, device=device)
    ids = legal_ids.to(device)
    mask = legal_mask.to(device)
    valid_ids = ids[mask]
    if valid_ids.numel() and bool((valid_ids >= head_width).any()):
        raise ValueError("policy event head is too small for a legal player event")
    rows = torch.arange(refs.numel(), device=device)[:, None].expand_as(ids)
    head_mask[rows[mask], ids[mask].long()] = True
    try:
        return policy_fn(encoded, head_mask)
    except TypeError as two_argument_error:
        try:
            return policy_fn(refs)
        except TypeError:
            raise two_argument_error


def _adapter_log_prob(
    adapter: Any,
    output: Any,
    legal_ids: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    """Return padded legal-column log-probabilities as ``[B,E,X]``."""

    output_device = output.v.device
    ids = legal_ids.to(output_device).to(torch.int64)
    mask = legal_mask.to(output_device)
    try:
        result = adapter.log_prob(output, ids)
    except TypeError:
        result = adapter.log_prob(output, ids, mask)
    if result.ndim != 3:
        raise ValueError("adapter.log_prob over legal events must return a rank-3 tensor")
    # Synthetic is [B,E,X]; poker's complete legal-row API is [B,X,E].
    if result.shape[1] != ids.shape[1] and result.shape[2] == ids.shape[1]:
        result = result.transpose(1, 2)
    if result.shape[:2] != ids.shape:
        raise ValueError("adapter.log_prob is not aligned with legal event columns")
    return result.float().masked_fill(~mask[..., None], -torch.inf)


def _draw_children(
    node_kind: int,
    log_probability: Tensor,
    legal_event_ids: Tensor,
    legal_mask: Tensor,
    torch_gen: torch.Generator,
) -> Tensor:
    """Sequential child hook: draw exactly one event for every input node.

    P09 replaces this hook's child-count/event rule while retaining the frontier
    growth loop below.
    """

    del node_kind
    probability = torch.exp(log_probability).masked_fill(~legal_mask, 0.0)
    normalizer = probability.sum(dim=1, keepdim=True)
    if bool((normalizer <= 0).any()):
        raise RuntimeError("a rollout event distribution has zero mass")
    # The positional P08/P09 compatibility path may pair a CUDA policy with its
    # historical CPU generator.  Keep that generator's exact stream; the C08/C09
    # production entry points instead require a device-matched generator.
    sample_device = torch.device(torch_gen.device)
    sample_probability = (probability / normalizer).to(sample_device)
    columns = torch.multinomial(sample_probability, 1, generator=torch_gen)
    return legal_event_ids.gather(1, columns.to(legal_event_ids.device)).squeeze(1).to(torch.int32)


def _hook_reach(
    hook: Callable[..., Any] | None,
    game: Any,
    state_refs: Tensor,
    rows: Tensor,
) -> Tensor:
    if hook is None:
        return rows
    try:
        value = hook(game, state_refs, rows)
    except TypeError as three_argument_error:
        try:
            value = hook(state_refs, rows)
        except TypeError:
            try:
                value = hook(rows)
            except TypeError:
                raise three_argument_error
    if value is None:
        return rows
    value = getattr(value, "log_r", value)
    result = torch.as_tensor(value, dtype=torch.float32, device=rows.device)
    if result.shape != rows.shape:
        raise ValueError("completion_reach_hook returned an invalid reach shape")
    return result


def sample_trees(
    game: Any,
    policy_fn: Callable[..., Any],
    adapter: Any,
    cfg_rollout: Any,
    completion_reach_hook: Callable[..., Any] | None,
    intensity: Any,
    torch_gen: torch.Generator,
    np_gen: np.random.Generator,
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Deprecated P08 positional sampler retained until the C11 switch."""

    if bool(getattr(cfg_rollout, "debug_asserts", False)):
        warnings.warn(
            "the positional sample_trees API is deprecated; use sample_trees_device",
            DeprecationWarning,
            stacklevel=2,
        )

    if cfg_rollout.beta_mode != "sequential":
        # Keep the P08 path below unchanged: its random-number consumption is a
        # reproducibility contract.  The import is local to avoid a module
        # cycle while allowing branching.py to reuse the rollout helpers.
        from pbg.core.branching import _legacy_sample_branched_trees

        return _legacy_sample_branched_trees(
            game,
            policy_fn,
            adapter,
            cfg_rollout,
            completion_reach_hook,
            intensity,
            torch_gen,
            np_gen,
        )
    del np_gen  # Reserved for P09 Poisson child counts.
    if intensity is not None:
        raise AssertionError("intensity must be None in sequential rollout mode")
    if cfg_rollout.reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ValueError("unknown reach_storage mode")
    n_trees = int(cfg_rollout.n_trees)
    budget = int(cfg_rollout.node_budget)
    if n_trees < 0:
        raise ValueError("n_trees must be nonnegative")
    if budget <= 0 and n_trees:
        raise ValueError("rollout node_budget must be positive")

    p_max = int(game.n_players)
    x_sizes = tuple(int(value) for value in game.x_sizes)
    x_max = max(x_sizes)
    roots = game.root_states(n_trees).to(torch.int32)
    root_owner = _classify(game, roots)
    builder = TreeBuilder(p_max)
    frontier: list[int] = []
    for tree_id, (state_ref, owner) in enumerate(zip(roots.tolist(), root_owner.tolist(), strict=True)):
        frontier.append(builder.add_root(state_ref, owner, tree_id))

    root_reach = torch.zeros((n_trees, p_max, x_max), dtype=torch.float32, device=roots.device)
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    root_reach = _hook_reach(completion_reach_hook, game, roots, root_reach)
    reach_by_node: list[Tensor] = [row.clone() for row in root_reach]
    tree_counts = [1] * n_trees
    policy_records: dict[int, tuple[Tensor, Tensor]] = {}

    while frontier:
        active = [
            node for node in frontier if builder.owners[node] not in (TERMINAL, PRETERM)
        ]
        frontier = []
        if not active:
            break
        refs = torch.tensor([builder.state_refs[node] for node in active], dtype=torch.int32)
        owners = torch.tensor([builder.owners[node] for node in active], dtype=torch.int32)
        player_positions = ((owners >= 0) & (owners < p_max)).nonzero(as_tuple=False).squeeze(1)
        chance_positions = (owners == CHANCE).nonzero(as_tuple=False).squeeze(1)
        drawn_event: dict[int, int] = {}
        next_ref: dict[int, int] = {}
        taken_logp: dict[int, Tensor] = {}

        if player_positions.numel():
            player_nodes = [active[int(position)] for position in player_positions.tolist()]
            player_refs = refs[player_positions]
            player_owners = owners[player_positions].to(torch.long)
            legal_ids, legal_mask = game.legal_events(player_refs)
            with torch.no_grad():
                output = _call_policy(game, policy_fn, player_refs, legal_ids, legal_mask)
                all_logp = _adapter_log_prob(adapter, output, legal_ids, legal_mask)
            reach_rows = torch.stack([reach_by_node[node] for node in player_nodes]).to(player_refs.device)
            profiles = game.sample_profiles(player_refs, reach_rows, 1, torch_gen)[:, 0]
            actor_type = profiles.gather(1, player_owners[:, None]).squeeze(1).long()
            batch = torch.arange(player_refs.numel(), device=all_logp.device)
            mixture_logp = all_logp[
                batch,
                :,
                actor_type.to(all_logp.device),
            ]
            events = _draw_children(
                0,
                mixture_logp,
                legal_ids.to(mixture_logp.device),
                legal_mask.to(mixture_logp.device),
                torch_gen,
            ).to(player_refs.device)
            successors = game.step(player_refs, events)
            selected_columns = (
                (legal_ids.to(events.device) == events[:, None]) & legal_mask.to(events.device)
            ).to(torch.int64).argmax(1)
            selected_all_types = all_logp[
                torch.arange(player_refs.numel(), device=all_logp.device),
                selected_columns.to(all_logp.device),
            ]
            values = output.v.detach().float()
            for row, node in enumerate(player_nodes):
                drawn_event[node] = int(events[row])
                next_ref[node] = int(successors[row])
                taken_logp[node] = selected_all_types[row].detach().cpu()
                policy_records[node] = (
                    selected_all_types[row].detach().cpu(),
                    values[row].detach().cpu(),
                )

        if chance_positions.numel():
            chance_nodes = [active[int(position)] for position in chance_positions.tolist()]
            chance_refs = refs[chance_positions]
            chance_reach = torch.stack([reach_by_node[node] for node in chance_nodes]).to(chance_refs.device)
            profiles = game.sample_profiles(chance_refs, chance_reach, 1, torch_gen)[:, 0]
            if hasattr(game, "draw_chance"):
                draw = game.draw_chance(chance_refs, profiles, torch_gen)
                events = torch.full((chance_refs.numel(),), -1, dtype=torch.int32, device=chance_refs.device)
                successors = draw.state_refs
            else:
                legal_ids, legal_mask = game.legal_events(chance_refs)
                probability = game.chance_probs(chance_refs, profiles)
                chance_logp = torch.where(
                    probability > 0,
                    torch.log(probability),
                    torch.full_like(probability, -torch.inf),
                )
                events = _draw_children(
                    CHANCE, chance_logp, legal_ids, legal_mask, torch_gen
                )
                successors = game.step(chance_refs, events)
            for row, node in enumerate(chance_nodes):
                drawn_event[node] = int(events[row])
                next_ref[node] = int(successors[row])

        for node in active:
            tree_id = builder.tree_ids[node]
            if tree_counts[tree_id] >= budget:
                raise RuntimeError(
                    f"rollout tree {tree_id} exceeded node_budget={budget}"
                )
            event = drawn_event[node]
            successor = next_ref[node]
            successor_ref = torch.tensor([successor], dtype=torch.int32)
            child_owner = int(_classify(game, successor_ref)[0])
            parent_reach = reach_by_node[node]
            child_reach = _replay_reach_update(
                game,
                parent_reach,
                torch.tensor([builder.state_refs[node]], dtype=torch.int32),
                torch.tensor([event], dtype=torch.int32),
                builder.owners[node],
                taken_logp.get(node),
            )
            child_reach = _hook_reach(
                completion_reach_hook, game, successor_ref, child_reach[None]
            )[0]
            child = builder.add_child(node, successor, event, 1.0, child_owner)
            reach_by_node.append(child_reach)
            tree_counts[tree_id] += 1
            frontier.append(child)

    tree = builder.finalize()
    node_count = len(builder.parents)
    if len(reach_by_node) != node_count:
        raise AssertionError("internal reach rows lost alignment with tree nodes")

    if cfg_rollout.reach_storage == "full":
        reach_nodes = list(range(node_count))
    elif cfg_rollout.reach_storage == "decisions_only":
        reach_nodes = [
            node
            for node, owner in enumerate(builder.owners)
            if 0 <= owner < p_max or owner in (TERMINAL, PRETERM)
        ]
    else:
        reach_nodes = [node for node, parent in enumerate(builder.parents) if parent == -1]
    reach_cache = ReachCache(
        mode=cfg_rollout.reach_storage,
        log_r=torch.stack([reach_by_node[node] for node in reach_nodes]).float()
        if reach_nodes
        else torch.empty((0, p_max, x_max), dtype=torch.float32),
        node_index=torch.tensor(reach_nodes, dtype=torch.int32),
    )

    decision_nodes = sorted(policy_records)
    e_max = max(
        (int(tree.child_ptr[node + 1] - tree.child_ptr[node]) for node in decision_nodes),
        default=0,
    )
    logp_child = torch.full(
        (len(decision_nodes), e_max, x_max), -torch.inf, dtype=torch.float32
    )
    v_old = torch.zeros((len(decision_nodes), x_max), dtype=torch.float32)
    for row, node in enumerate(decision_nodes):
        logp, value = policy_records[node]
        width = min(x_max, logp.numel())
        logp_child[row, 0, :width] = logp[:width]
        v_old[row, : min(x_max, value.numel())] = value[:x_max]
    policy_cache = PolicyCache(
        logp_child=logp_child,
        v_old=v_old,
        node_index=torch.tensor(decision_nodes, dtype=torch.int32),
    )
    return tree, reach_cache, policy_cache
