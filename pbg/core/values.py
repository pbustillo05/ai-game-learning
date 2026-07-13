"""Backward value propagation on stored public trees.

This module is deliberately game neutral.  Games supply completed leaf
statistics and already-evaluated frozen heads; the code below only applies the
stored-tree measure and the capture-before-mix lambda-return recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.tree import TreeBatch


@dataclass
class HeadEvals:
    """Frozen value-head rows, keyed by player decision node id."""

    node_ids: Tensor  # int32 [Nd]
    v: Tensor  # [Nd,X_max]
    vvec: Tensor | None = None  # [Nd,P_max]
    u: Tensor | None = None  # [Nd]


@dataclass
class NodeTargets:
    """Training targets aligned with ``node_ids`` (one row per player node)."""

    node_ids: Tensor  # int32 [Nd]
    p_hat: Tensor  # [Nd,X_max]
    q_hat: Tensor  # [Nd,E_max,X_max]
    v_hat: Tensor  # [Nd,X_max]
    vvec_hat: Tensor  # [Nd,P_max]
    u_hat: Tensor  # [Nd]
    nu_row: Tensor  # [Nd,E_max]
    child_event_row: Tensor  # int32 [Nd,E_max]


def _floating_dtype(leaf_stats: BatchedCompletionStats, heads: HeadEvals) -> torch.dtype:
    """Use fp32 in production while retaining the prompt's fp64 oracle mode."""

    values = (
        leaf_stats.p_hat,
        leaf_stats.v_hat,
        leaf_stats.vvec_hat,
        leaf_stats.u_hat,
        heads.v,
    )
    if heads.vvec is not None:
        values += (heads.vvec,)
    if heads.u is not None:
        values += (heads.u,)
    if any(value.dtype in (torch.float16, torch.bfloat16) for value in values):
        raise TypeError("branched_value does not accept fp16/bf16 accumulators")
    return torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32


def _leaf_node_ids(tree: TreeBatch, leaf_stats: BatchedCompletionStats) -> Tensor:
    """Resolve leaf rows, accepting explicit metadata but defaulting to node order."""

    counts = tree.child_ptr[1:].to(torch.int64) - tree.child_ptr[:-1].to(torch.int64)
    default = (counts == 0).nonzero(as_tuple=False).squeeze(1)
    explicit = leaf_stats.metadata.get("node_ids", leaf_stats.metadata.get("leaf_node_ids"))
    if explicit is None:
        result = default
    else:
        result = torch.as_tensor(explicit, dtype=torch.int64, device=tree.parent.device)
    if result.ndim != 1 or result.numel() != leaf_stats.p_hat.shape[0]:
        raise ValueError("leaf statistics must contain exactly one row per stored leaf")
    if result.numel():
        if bool((result < 0).any()) or bool((result >= tree.parent.numel()).any()):
            raise ValueError("leaf statistics contain an invalid node id")
        if result.unique().numel() != result.numel() or not bool((counts[result] == 0).all()):
            raise ValueError("leaf statistics node ids must be unique leaves")
        if result.numel() != default.numel():
            raise ValueError("leaf statistics do not cover every stored leaf")
    return result


@dataclass
class _Layout:
    device: torch.device
    dtype: torch.dtype
    n: int
    p_max: int
    x_max: int
    decision_nodes: Tensor
    decision_row: Tensor
    head_row: Tensor
    leaf_nodes: Tensor
    leaf_row: Tensor
    counts: Tensor
    e_max: int
    type_mask: Tensor
    seat_mask: Tensor
    moments: bool


def _validate_and_layout(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    reach: Any,
    heads: HeadEvals,
    lam: float,
    game: Any,
    dtype: torch.dtype,
) -> _Layout:
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lam must be in [0,1]")
    tree.validate()
    device = leaf_stats.p_hat.device
    n = int(tree.parent.numel())
    if leaf_stats.p_hat.ndim != 3:
        raise ValueError("leaf p_hat must have shape [L,P_max,X_max]")
    leaves, p_max, x_max = map(int, leaf_stats.p_hat.shape)
    expected_leaf_shapes = {
        "v_hat": (leaves, p_max, x_max),
        "vvec_hat": (leaves, p_max),
        "u_hat": (leaves,),
        "n_eff": (leaves,),
    }
    for name, shape in expected_leaf_shapes.items():
        if tuple(getattr(leaf_stats, name).shape) != shape:
            raise ValueError(f"leaf {name} must have shape {shape}")
    n_players = int(game.n_players)
    x_sizes = tuple(int(size) for size in game.x_sizes)
    if len(x_sizes) != n_players or not 0 < n_players <= p_max:
        raise ValueError("game player dimensions do not fit completion statistics")
    if any(size <= 0 or size > x_max for size in x_sizes):
        raise ValueError("game private-type dimensions do not fit completion statistics")

    owner = tree.owner.to(device)
    decision_nodes = ((owner >= 0) & (owner < n_players)).nonzero(as_tuple=False).squeeze(1)
    nd = int(decision_nodes.numel())
    if heads.node_ids.ndim != 1 or heads.node_ids.numel() != nd:
        raise ValueError("heads must contain exactly one row per player node")
    if tuple(heads.v.shape) != (nd, x_max):
        raise ValueError(f"heads.v must have shape {(nd, x_max)}")
    moments = heads.vvec is not None or heads.u is not None
    if (heads.vvec is None) != (heads.u is None):
        raise ValueError("vvec and u heads must either both be present or both be absent")
    if moments and (tuple(heads.vvec.shape) != (nd, p_max) or tuple(heads.u.shape) != (nd,)):
        raise ValueError("moment heads have incompatible shapes")

    head_ids = heads.node_ids.to(device=device, dtype=torch.int64)
    if head_ids.numel() and (
        bool((head_ids < 0).any())
        or bool((head_ids >= n).any())
        or head_ids.unique().numel() != head_ids.numel()
    ):
        raise ValueError("head node ids must be unique valid node ids")
    head_row = torch.full((n,), -1, dtype=torch.int64, device=device)
    if nd:
        head_row[head_ids] = torch.arange(nd, device=device)
        if not bool((head_row[decision_nodes] >= 0).all()):
            raise ValueError("heads do not cover the tree's player nodes")
    decision_row = torch.full((n,), -1, dtype=torch.int64, device=device)
    decision_row[decision_nodes] = torch.arange(nd, device=device)

    leaf_nodes_tree = _leaf_node_ids(tree, leaf_stats)
    leaf_nodes = leaf_nodes_tree.to(device=device, dtype=torch.int64)
    leaf_row = torch.full((n,), -1, dtype=torch.int64, device=device)
    if leaf_nodes.numel():
        leaf_row[leaf_nodes] = torch.arange(leaf_nodes.numel(), device=device)

    # Reach is intentionally not consumed by the recurrence.  Assert only the
    # dimensional contract when a cache/tensor is supplied; sparse cache modes
    # need not contain one row per tree node.
    reach_rows = getattr(reach, "log_r", reach if isinstance(reach, Tensor) else None)
    if reach_rows is not None and (reach_rows.ndim != 3 or tuple(reach_rows.shape[1:]) != (p_max, x_max)):
        raise ValueError("reach rows must have trailing shape [P_max,X_max]")

    counts = (tree.child_ptr[1:].to(device=device, dtype=torch.int64) - tree.child_ptr[:-1].to(device=device, dtype=torch.int64))
    e_max = int(counts[decision_nodes].max()) if nd else 0
    type_mask = torch.zeros((p_max, x_max), dtype=torch.bool, device=device)
    for player, size in enumerate(x_sizes):
        type_mask[player, :size] = True
    seat_mask = torch.arange(p_max, device=device) < n_players
    return _Layout(
        device, dtype, n, p_max, x_max, decision_nodes, decision_row, head_row,
        leaf_nodes, leaf_row, counts, e_max, type_mask, seat_mask, moments,
    )


def _empty_targets(layout: _Layout) -> NodeTargets:
    nd, e_max, x_max, p_max = (
        layout.decision_nodes.numel(), layout.e_max, layout.x_max, layout.p_max
    )
    zeros = lambda *shape: torch.zeros(shape, dtype=layout.dtype, device=layout.device)
    return NodeTargets(
        node_ids=layout.decision_nodes.to(torch.int32),
        p_hat=zeros(nd, x_max),
        q_hat=zeros(nd, e_max, x_max),
        v_hat=zeros(nd, x_max),
        vvec_hat=zeros(nd, p_max),
        u_hat=zeros(nd),
        nu_row=zeros(nd, e_max),
        child_event_row=torch.zeros((nd, e_max), dtype=torch.int32, device=layout.device),
    )


def _fill_edge_rows(result: NodeTargets, layout: _Layout, tree: TreeBatch) -> None:
    """Copy CSR edge metadata into padded decision rows."""

    ptr = tree.child_ptr.to(device=layout.device, dtype=torch.int64)
    nu = tree.nu.detach().to(device=layout.device, dtype=layout.dtype)
    events = tree.child_event.to(device=layout.device, dtype=torch.int32)
    for node in layout.decision_nodes.tolist():
        row = int(layout.decision_row[node])
        start, stop = int(ptr[node]), int(ptr[node + 1])
        width = stop - start
        result.nu_row[row, :width] = nu[start:stop]
        result.child_event_row[row, :width] = events[start:stop]


def _tree_groups(tree: TreeBatch, p_max: int, max_resident_rows: int) -> list[Tensor]:
    """Greedily group complete trees without splitting a dependency chain."""

    if max_resident_rows <= 0:
        raise ValueError("max_resident_rows must be positive")
    ids = tree.tree_id.detach().cpu().tolist()
    order: list[int] = []
    by_tree: dict[int, list[int]] = {}
    for node, tree_id in enumerate(ids):
        if tree_id not in by_tree:
            order.append(tree_id)
            by_tree[tree_id] = []
        by_tree[tree_id].append(node)
    groups: list[Tensor] = []
    pending: list[int] = []
    pending_rows = 0
    for tree_id in order:
        nodes = by_tree[tree_id]
        rows = len(nodes) * p_max
        if pending and pending_rows + rows > max_resident_rows:
            groups.append(torch.tensor(pending, dtype=torch.int64))
            pending, pending_rows = [], 0
        pending.extend(nodes)
        pending_rows += rows
    if pending:
        groups.append(torch.tensor(pending, dtype=torch.int64))
    return groups


def _topological_depths(tree: TreeBatch, device: torch.device) -> Tensor:
    """Compute graph depths without assuming a particular node-id ordering."""

    parents = [int(value) for value in tree.parent.detach().cpu().tolist()]
    depths = [-1] * len(parents)
    visiting: set[int] = set()

    def depth(node: int) -> int:
        if depths[node] >= 0:
            return depths[node]
        if node in visiting:
            raise ValueError("tree parent relation contains a cycle")
        visiting.add(node)
        parent = parents[node]
        value = 0 if parent < 0 else depth(parent) + 1
        visiting.remove(node)
        depths[node] = value
        return value

    for node in range(len(parents)):
        depth(node)
    return torch.tensor(depths, dtype=torch.int64, device=device)


def branched_value(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    reach: Any,
    heads: HeadEvals,
    lam: float,
    game: Any,
    max_resident_rows: int = 2_000_000,
) -> NodeTargets:
    """Propagate completed leaf values and emit capture-before-mix targets."""

    dtype = _floating_dtype(leaf_stats, heads)
    layout = _validate_and_layout(tree, leaf_stats, reach, heads, lam, game, dtype)
    result = _empty_targets(layout)
    _fill_edge_rows(result, layout, tree)
    if layout.n == 0:
        return result

    device = layout.device
    ptr = tree.child_ptr.to(device=device, dtype=torch.int64)
    child_global = tree.child_node.to(device=device, dtype=torch.int64)
    owner_global = tree.owner.to(device=device, dtype=torch.int64)
    nu = tree.nu.detach().to(device=device, dtype=dtype)
    leaf_p = leaf_stats.p_hat.detach().to(device=device, dtype=dtype)
    leaf_v = leaf_stats.v_hat.detach().to(device=device, dtype=dtype)
    leaf_vvec = leaf_stats.vvec_hat.detach().to(device=device, dtype=dtype)
    leaf_u = leaf_stats.u_hat.detach().to(device=device, dtype=dtype)
    head_v = heads.v.detach().to(device=device, dtype=dtype)
    head_vvec = heads.vvec.detach().to(device=device, dtype=dtype) if layout.moments else None
    head_u = heads.u.detach().to(device=device, dtype=dtype) if layout.moments else None
    type_mask = layout.type_mask
    seat_mask = layout.seat_mask
    graph_depth = _topological_depths(tree, device)

    for group_cpu in _tree_groups(tree, layout.p_max, max_resident_rows):
        nodes = group_cpu.to(device)
        size = int(nodes.numel())
        local_of = torch.full((layout.n,), -1, dtype=torch.int64, device=device)
        local_of[nodes] = torch.arange(size, device=device)
        p_hat = torch.zeros((size, layout.p_max, layout.x_max), dtype=dtype, device=device)
        v_all = torch.zeros_like(p_hat)
        vvec = torch.zeros((size, layout.p_max), dtype=dtype, device=device)
        u = torch.zeros(size, dtype=dtype, device=device)

        group_leaf_mask = layout.leaf_row[nodes] >= 0
        if bool(group_leaf_mask.any()):
            local_leaf = group_leaf_mask.nonzero(as_tuple=False).squeeze(1)
            rows = layout.leaf_row[nodes[local_leaf]]
            p_hat[local_leaf] = leaf_p[rows].masked_fill(~type_mask[None], 0.0)
            v_all[local_leaf] = leaf_v[rows].masked_fill(~type_mask[None], 0.0)
            vvec[local_leaf] = leaf_vvec[rows].masked_fill(~seat_mask[None], 0.0)
            u[local_leaf] = leaf_u[rows]

        # Actual graph depth is distinct from depth_player because chance
        # edges do not increment the latter.
        depths = graph_depth[nodes]

        for depth in range(int(depths.max()), -1, -1):
            current_local = ((depths == depth) & (layout.counts[nodes] > 0)).nonzero(as_tuple=False).squeeze(1)
            if current_local.numel() == 0:
                continue
            current = nodes[current_local]
            counts = layout.counts[current]
            total = int(counts.sum())
            repeated_local = torch.repeat_interleave(current_local, counts)
            cumulative = counts.cumsum(0) - counts
            slots = torch.arange(total, device=device) - torch.repeat_interleave(cumulative, counts)
            starts = ptr[current]
            edge_ids = torch.repeat_interleave(starts, counts) + slots
            children = child_global[edge_ids]
            child_local = local_of[children]
            weights = nu[edge_ids]
            p_hat.index_add_(0, repeated_local, p_hat[child_local] * weights[:, None, None])
            v_all.index_add_(0, repeated_local, v_all[child_local] * weights[:, None, None])
            vvec.index_add_(0, repeated_local, vvec[child_local] * weights[:, None])
            u.index_add_(0, repeated_local, u[child_local] * weights)

            edge_owner = owner_global[current][torch.repeat_interleave(torch.arange(current.numel(), device=device), counts)]
            edge_rows = layout.decision_row[current][torch.repeat_interleave(torch.arange(current.numel(), device=device), counts)]
            decision_edges = edge_rows >= 0
            if bool(decision_edges.any()):
                result.q_hat[edge_rows[decision_edges], slots[decision_edges]] = v_all[
                    child_local[decision_edges], edge_owner[decision_edges]
                ]

            decision_mask = layout.decision_row[current] >= 0
            if bool(decision_mask.any()):
                dec_nodes = current[decision_mask]
                dec_local = current_local[decision_mask]
                rows = layout.decision_row[dec_nodes]
                players = owner_global[dec_nodes]
                result.p_hat[rows] = p_hat[dec_local, players]
                result.v_hat[rows] = v_all[dec_local, players]
                result.vvec_hat[rows] = vvec[dec_local]
                result.u_hat[rows] = u[dec_local]

                hrows = layout.head_row[dec_nodes]
                mixed_owner = float(lam) * v_all[dec_local, players] + (1.0 - float(lam)) * head_v[hrows]
                mixed_owner = mixed_owner.masked_fill(~type_mask[players], 0.0)
                v_all[dec_local, players] = mixed_owner
                if layout.moments:
                    vvec[dec_local] = (
                        float(lam) * vvec[dec_local] + (1.0 - float(lam)) * head_vvec[hrows]
                    ).masked_fill(~seat_mask[None], 0.0)
                    u[dec_local] = float(lam) * u[dec_local] + (1.0 - float(lam)) * head_u[hrows]
    return result


def branched_value_reference(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    reach: Any,
    heads: HeadEvals,
    lam: float,
    game: Any,
) -> NodeTargets:
    """Independent fp64 recursive oracle for the stored-path semantics."""

    layout = _validate_and_layout(tree, leaf_stats, reach, heads, lam, game, torch.float64)
    result = _empty_targets(layout)
    _fill_edge_rows(result, layout, tree)
    device = layout.device
    ptr = tree.child_ptr.to(device=device, dtype=torch.int64)
    child = tree.child_node.to(device=device, dtype=torch.int64)
    owner = tree.owner.to(device=device, dtype=torch.int64)
    nu = tree.nu.detach().to(device=device, dtype=torch.float64)
    leaf_p = leaf_stats.p_hat.detach().to(device=device, dtype=torch.float64)
    leaf_v = leaf_stats.v_hat.detach().to(device=device, dtype=torch.float64)
    leaf_vvec = leaf_stats.vvec_hat.detach().to(device=device, dtype=torch.float64)
    leaf_u = leaf_stats.u_hat.detach().to(device=device, dtype=torch.float64)
    head_v = heads.v.detach().to(device=device, dtype=torch.float64)
    head_vvec = heads.vvec.detach().to(device=device, dtype=torch.float64) if layout.moments else None
    head_u = heads.u.detach().to(device=device, dtype=torch.float64) if layout.moments else None
    memo: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}

    def descend(node: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return post-node-mix statistics by recursively enumerating children."""

        if node in memo:
            return memo[node]
        start, stop = int(ptr[node]), int(ptr[node + 1])
        if start == stop:
            row = int(layout.leaf_row[node])
            value = (
                leaf_p[row].masked_fill(~layout.type_mask, 0.0).clone(),
                leaf_v[row].masked_fill(~layout.type_mask, 0.0).clone(),
                leaf_vvec[row].masked_fill(~layout.seat_mask, 0.0).clone(),
                leaf_u[row].clone(),
            )
            memo[node] = value
            return value
        children = [descend(int(child[edge])) for edge in range(start, stop)]
        p_value = sum((nu[edge] * children[edge - start][0] for edge in range(start, stop)), torch.zeros((layout.p_max, layout.x_max), dtype=torch.float64, device=device))
        v_value = sum((nu[edge] * children[edge - start][1] for edge in range(start, stop)), torch.zeros((layout.p_max, layout.x_max), dtype=torch.float64, device=device))
        vvec_value = sum((nu[edge] * children[edge - start][2] for edge in range(start, stop)), torch.zeros(layout.p_max, dtype=torch.float64, device=device))
        u_value = sum((nu[edge] * children[edge - start][3] for edge in range(start, stop)), torch.zeros((), dtype=torch.float64, device=device))
        player = int(owner[node])
        if 0 <= player < int(game.n_players):
            row = int(layout.decision_row[node])
            result.p_hat[row] = p_value[player]
            result.v_hat[row] = v_value[player]
            result.vvec_hat[row] = vvec_value
            result.u_hat[row] = u_value
            for slot, child_values in enumerate(children):
                result.q_hat[row, slot] = child_values[1][player]
            hrow = int(layout.head_row[node])
            v_value[player] = (
                float(lam) * v_value[player] + (1.0 - float(lam)) * head_v[hrow]
            ).masked_fill(~layout.type_mask[player], 0.0)
            if layout.moments:
                vvec_value = (
                    float(lam) * vvec_value + (1.0 - float(lam)) * head_vvec[hrow]
                ).masked_fill(~layout.seat_mask, 0.0)
                u_value = float(lam) * u_value + (1.0 - float(lam)) * head_u[hrow]
        value = (p_value, v_value, vvec_value, u_value)
        memo[node] = value
        return value

    for root in (tree.parent == -1).nonzero(as_tuple=False).squeeze(1).tolist():
        descend(root)
    return result
