"""Posterior-mass backward propagation on labeled sampled trees.

This is the corrected value engine staged alongside :mod:`pbg.core.values`.
It propagates posterior bucket mass ``P`` and posterior-weighted return mass
``M``; conditional values are formed only when decision targets are emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.tree import TreeBatch


@dataclass
class HeadEvals:
    """Frozen value heads in stable decision-row order.

    ``node_ids`` is int32 ``[Nd]``, ``v`` is fp32 ``[Nd,X]``, and each entry
    of ``aux`` has leading dimension ``Nd``.
    """

    node_ids: Tensor  # int32 [Nd]
    v: Tensor  # float32 [Nd,X]
    aux: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class NodeTargets:
    """Posterior targets aligned with stored player-decision nodes.

    The padded dimensions are ``[Nd,E,X]``.  ``q_hat`` is absent unless the
    caller requests it, and auxiliary captures retain their channel-specific
    trailing dimensions after a leading ``Nd`` dimension.
    """

    node_ids: Tensor  # int32 [Nd]
    parent_p_hat: Tensor  # float32 [Nd,X]
    child_p_hat: Tensor  # float32 [Nd,E,X]
    a_hat: Tensor  # float32 [Nd,E,X]
    v_hat: Tensor  # float32 [Nd,X]
    child_event_row: Tensor  # int32 [Nd,E]
    child_mask: Tensor  # bool [Nd,E]
    child_count: Tensor  # int32 [Nd]
    q_hat: Tensor | None = None  # float32 [Nd,E,X]
    aux: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class AuxiliaryChannelSpec:
    """Describe one optional dense backward-propagated tensor channel."""

    name: str
    leaf_field: str
    head_field: str | None = None
    bootstrap: Literal["none", "every_player_node"] = "none"
    emit_field: str | None = None


@dataclass
class _Layout:
    device: torch.device
    n: int
    p_max: int
    x_max: int
    n_players: int
    decision_nodes: Tensor
    decision_row: Tensor
    head_row: Tensor
    leaf_nodes: Tensor
    leaf_row: Tensor
    child_count_all: Tensor
    e_max: int
    type_mask: Tensor


@dataclass
class _AuxiliaryChannel:
    spec: AuxiliaryChannelSpec
    leaf: Tensor  # [L,...]
    head: Tensor | None  # [Nd,...]


def conditionalize(moment: Tensor, mass: Tensor) -> Tensor:
    """Return ``moment / mass`` with exact zeros for invalid mass ``[...]``."""

    valid = mass > 0
    result = torch.zeros_like(moment)
    result[valid] = moment[valid] / mass[valid]
    return result


def aggregate(
    values: Tensor,
    child_local: Tensor,
    parent_local: Tensor,
    parent_count: Tensor,
) -> None:
    """Add child ``values [C,...]`` into uniform parent means in-place."""

    values.index_add_(0, parent_local, values[child_local])
    ready = parent_local.unique()
    view = (ready.numel(),) + (1,) * (values.ndim - 1)
    values[ready] /= parent_count[ready].reshape(view).to(values.dtype)


def bootstrap(moment: Tensor, mass: Tensor, old_v: Tensor, lam: float) -> Tensor:
    """Mix owner moment rows ``[B,X]`` with posterior-weighted head values."""

    return float(lam) * moment + (1.0 - float(lam)) * mass * old_v


def emit(
    result: NodeTargets,
    rows: Tensor,
    players: Tensor,
    parent_p: Tensor,
    parent_m: Tensor,
    child_p: Tensor,
    child_m: Tensor,
    child_slots: Tensor,
    old_v: Tensor,
    lam: float,
) -> None:
    """Emit decision targets from captured parent/child ``P,M`` tensors."""

    edge_players = players.repeat_interleave(result.child_count[rows].to(torch.int64))
    selected_child_p = child_p[torch.arange(child_p.shape[0], device=child_p.device), edge_players]
    selected_child_m = child_m[torch.arange(child_m.shape[0], device=child_m.device), edge_players]
    q_value = conditionalize(selected_child_m, selected_child_p)  # [C,X]

    selected_parent_p = parent_p[torch.arange(rows.numel(), device=rows.device), players]
    selected_parent_m = parent_m[torch.arange(rows.numel(), device=rows.device), players]
    result.parent_p_hat[rows] = selected_parent_p
    result.v_hat[rows] = conditionalize(selected_parent_m, selected_parent_p)
    edge_rows = rows.repeat_interleave(result.child_count[rows].to(torch.int64))
    result.child_p_hat[edge_rows, child_slots] = selected_child_p

    counts = result.child_count[rows].to(torch.int64)
    repeated_parent_p = selected_parent_p.repeat_interleave(counts, dim=0)
    repeated_parent_m = selected_parent_m.repeat_interleave(counts, dim=0)
    repeated_old_v = old_v.repeat_interleave(counts, dim=0)
    multi = counts.repeat_interleave(counts) > 1
    sibling_p = repeated_parent_p * counts.repeat_interleave(counts)[:, None] - selected_child_p
    sibling_m = repeated_parent_m * counts.repeat_interleave(counts)[:, None] - selected_child_m
    sibling = conditionalize(sibling_m, sibling_p)
    baseline = repeated_old_v.clone()
    baseline[multi] = float(lam) * sibling[multi] + (1.0 - float(lam)) * repeated_old_v[multi]
    result.a_hat[edge_rows, child_slots] = q_value - baseline
    if result.q_hat is not None:
        result.q_hat[edge_rows, child_slots] = q_value


def _reject_low_precision(name: str, value: Tensor) -> None:
    if value.dtype in (torch.float16, torch.bfloat16):
        raise TypeError(f"{name} cannot use fp16/bf16 accumulators")


def _leaf_node_ids(tree: TreeBatch, leaf_stats: BatchedCompletionStats) -> Tensor:
    counts = tree.child_ptr[1:].to(torch.int64) - tree.child_ptr[:-1].to(torch.int64)
    default = (counts == 0).nonzero(as_tuple=False).squeeze(1)
    explicit = leaf_stats.metadata.get("node_ids", leaf_stats.metadata.get("leaf_node_ids"))
    result = default if explicit is None else torch.as_tensor(explicit, device=tree.parent.device)
    result = result.to(torch.int64)
    if result.ndim != 1 or result.numel() != leaf_stats.p_hat.shape[0]:
        raise ValueError("leaf statistics must contain exactly one row per stored leaf")
    if result.numel() != default.numel() or result.unique().numel() != result.numel():
        raise ValueError("leaf statistics must uniquely cover every stored leaf")
    if result.numel() and (
        bool((result < 0).any())
        or bool((result >= counts.numel()).any())
        or not bool((counts[result] == 0).all())
    ):
        raise ValueError("leaf statistics contain a non-leaf or invalid node id")
    return result


def _layout(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    reach: Any,
    heads: HeadEvals,
    lam: float,
    game: Any,
) -> _Layout:
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lam must be in [0,1]")
    tree.validate()
    device = leaf_stats.p_hat.device
    n = int(tree.parent.numel())
    if leaf_stats.p_hat.ndim != 3:
        raise ValueError("leaf p_hat must have shape [L,P_max,X_max]")
    leaves, p_max, x_max = map(int, leaf_stats.p_hat.shape)
    if tuple(leaf_stats.v_hat.shape) != (leaves, p_max, x_max):
        raise ValueError("leaf v_hat must match leaf p_hat")
    _reject_low_precision("leaf p_hat", leaf_stats.p_hat)
    _reject_low_precision("leaf v_hat", leaf_stats.v_hat)
    if not bool(torch.isfinite(leaf_stats.p_hat).all()) or bool((leaf_stats.p_hat < 0).any()):
        raise ValueError("leaf posterior mass must be finite and nonnegative")
    if not bool(torch.isfinite(leaf_stats.v_hat).all()):
        raise ValueError("leaf values must be finite")

    n_players = int(game.n_players)
    x_sizes = tuple(int(size) for size in game.x_sizes)
    if len(x_sizes) != n_players or not 0 < n_players <= p_max:
        raise ValueError("game player dimensions do not fit completion statistics")
    if any(size <= 0 or size > x_max for size in x_sizes):
        raise ValueError("game type dimensions do not fit completion statistics")
    for name in ("parent", "child_ptr", "child_node", "child_event", "tree_id", "owner"):
        if getattr(tree, name).device != device:
            raise ValueError("tree and completion tensors must share a device")

    owner = tree.owner.to(torch.int64)
    decision_nodes = ((owner >= 0) & (owner < n_players)).nonzero(as_tuple=False).squeeze(1)
    nd = int(decision_nodes.numel())
    if (
        heads.node_ids.ndim != 1
        or heads.node_ids.numel() != nd
        or tuple(heads.v.shape) != (nd, x_max)
    ):
        raise ValueError("heads must have one [X_max] row per player node")
    _reject_low_precision("heads.v", heads.v)
    if heads.node_ids.device != device or heads.v.device != device:
        raise ValueError("heads and completion tensors must share a device")
    head_ids = heads.node_ids.to(torch.int64)
    if head_ids.numel() and (
        head_ids.unique().numel() != nd
        or bool((head_ids < 0).any())
        or bool((head_ids >= n).any())
    ):
        raise ValueError("head node ids must be unique valid ids")
    head_row = torch.full((n,), -1, dtype=torch.int64, device=device)
    head_row[head_ids] = torch.arange(nd, device=device)
    if nd and not bool((head_row[decision_nodes] >= 0).all()):
        raise ValueError("heads do not cover every player node")
    decision_row = torch.full((n,), -1, dtype=torch.int64, device=device)
    decision_row[decision_nodes] = torch.arange(nd, device=device)

    leaf_nodes = _leaf_node_ids(tree, leaf_stats)
    leaf_row = torch.full((n,), -1, dtype=torch.int64, device=device)
    leaf_row[leaf_nodes] = torch.arange(leaves, device=device)
    counts = tree.child_ptr[1:].to(torch.int64) - tree.child_ptr[:-1].to(torch.int64)
    e_max = int(counts[decision_nodes].max()) if nd else 0
    type_mask = torch.zeros((p_max, x_max), dtype=torch.bool, device=device)
    for player, size in enumerate(x_sizes):
        type_mask[player, :size] = True

    reach_rows = getattr(reach, "log_r", reach if isinstance(reach, Tensor) else None)
    if reach_rows is not None and (
        reach_rows.ndim != 3
        or tuple(reach_rows.shape[1:]) != (p_max, x_max)
    ):
        raise ValueError("reach rows must have trailing shape [P_max,X_max]")
    return _Layout(
        device, n, p_max, x_max, n_players, decision_nodes, decision_row,
        head_row, leaf_nodes, leaf_row, counts, e_max, type_mask,
    )


def _channels(
    specs: Sequence[AuxiliaryChannelSpec],
    leaf_stats: BatchedCompletionStats,
    heads: HeadEvals,
    layout: _Layout,
) -> list[_AuxiliaryChannel]:
    channels: list[_AuxiliaryChannel] = []
    names: set[str] = set()
    emits: set[str] = set()
    for spec in specs:
        if spec.name in names:
            raise ValueError(f"duplicate auxiliary channel {spec.name!r}")
        names.add(spec.name)
        if spec.bootstrap not in ("none", "every_player_node"):
            raise ValueError(f"invalid bootstrap mode {spec.bootstrap!r}")
        leaf = getattr(leaf_stats, spec.leaf_field, None)
        if not isinstance(leaf, Tensor) or leaf.ndim < 1 or leaf.shape[0] != layout.leaf_nodes.numel():
            raise ValueError(f"auxiliary leaf field {spec.leaf_field!r} has the wrong leading dimension")
        _reject_low_precision(f"auxiliary channel {spec.name}", leaf)
        head = None
        if spec.bootstrap == "every_player_node":
            if spec.head_field is None or spec.head_field not in heads.aux:
                raise ValueError(f"auxiliary channel {spec.name!r} requires its head field")
            head = heads.aux[spec.head_field]
            if tuple(head.shape) != (layout.decision_nodes.numel(), *leaf.shape[1:]):
                raise ValueError(f"auxiliary head field {spec.head_field!r} has the wrong shape")
            _reject_low_precision(f"auxiliary head {spec.head_field}", head)
        elif spec.head_field is not None:
            raise ValueError("head_field is only valid for a bootstrapped auxiliary channel")
        if leaf.device != layout.device or (head is not None and head.device != layout.device):
            raise ValueError("auxiliary tensors must share the tree device")
        if spec.emit_field is not None:
            if spec.emit_field in emits:
                raise ValueError(f"duplicate auxiliary emit field {spec.emit_field!r}")
            emits.add(spec.emit_field)
        channels.append(
            _AuxiliaryChannel(
                spec,
                leaf.detach().to(torch.float32),
                None if head is None else head.detach().to(torch.float32),
            )
        )
    return channels


def _empty_targets(layout: _Layout, emit_q: bool, channels: Sequence[_AuxiliaryChannel]) -> NodeTargets:
    nd, e_max, x_max = int(layout.decision_nodes.numel()), layout.e_max, layout.x_max
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=layout.device)
    aux = {
        channel.spec.emit_field: zeros(nd, *channel.leaf.shape[1:])
        for channel in channels
        if channel.spec.emit_field is not None
    }
    return NodeTargets(
        node_ids=layout.decision_nodes.to(torch.int32),
        parent_p_hat=zeros(nd, x_max),
        child_p_hat=zeros(nd, e_max, x_max),
        a_hat=zeros(nd, e_max, x_max),
        v_hat=zeros(nd, x_max),
        child_event_row=torch.full((nd, e_max), -1, dtype=torch.int32, device=layout.device),
        child_mask=torch.zeros((nd, e_max), dtype=torch.bool, device=layout.device),
        child_count=layout.child_count_all[layout.decision_nodes].to(torch.int32),
        q_hat=zeros(nd, e_max, x_max) if emit_q else None,
        aux=aux,
    )


def _edge_metadata(result: NodeTargets, tree: TreeBatch, layout: _Layout) -> None:
    counts = result.child_count.to(torch.int64)
    if not counts.numel() or int(counts.sum()) == 0:
        return
    ptr = tree.child_ptr.to(torch.int64)
    cumulative = torch.cumsum(counts, 0) - counts
    slots = torch.arange(int(counts.sum()), device=layout.device) - torch.repeat_interleave(cumulative, counts)
    edge_ids = torch.repeat_interleave(ptr[layout.decision_nodes], counts) + slots
    rows = torch.repeat_interleave(torch.arange(counts.numel(), device=layout.device), counts)
    result.child_event_row[rows, slots] = tree.child_event[edge_ids]
    result.child_mask[rows, slots] = True


def _process_chunk(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    heads: HeadEvals,
    lam: float,
    layout: _Layout,
    result: NodeTargets,
    channels: Sequence[_AuxiliaryChannel],
    nodes: Tensor,
) -> None:
    size = int(nodes.numel())
    local_of = torch.full((layout.n,), -1, dtype=torch.int64, device=layout.device)
    local_of[nodes] = torch.arange(size, device=layout.device)
    parent = tree.parent[nodes].to(torch.int64)
    counts = layout.child_count_all[nodes]
    p_mass = torch.zeros((size, layout.p_max, layout.x_max), dtype=torch.float32, device=layout.device)
    m_mass = torch.zeros_like(p_mass)
    aux_values = [
        torch.zeros(
            (size, *channel.leaf.shape[1:]),
            dtype=torch.float32,
            device=layout.device,
        )
        for channel in channels
    ]

    leaf_rows = layout.leaf_row[nodes]
    leaf_local = (leaf_rows >= 0).nonzero(as_tuple=False).squeeze(1)
    source_rows = leaf_rows[leaf_local]
    leaf_p = (
        leaf_stats.p_hat[source_rows]
        .detach()
        .to(torch.float32)
        .masked_fill(~layout.type_mask[None], 0.0)
    )
    leaf_v = (
        leaf_stats.v_hat[source_rows]
        .detach()
        .to(torch.float32)
        .masked_fill(~layout.type_mask[None], 0.0)
    )
    leaf_v = torch.where(leaf_p > 0, leaf_v, torch.zeros_like(leaf_v))
    p_mass[leaf_local] = leaf_p
    m_mass[leaf_local] = leaf_p * leaf_v
    for values, channel in zip(aux_values, channels):
        values[leaf_local] = channel.leaf[source_rows]

    processed = torch.zeros(size, dtype=torch.int64, device=layout.device)
    current = leaf_local
    while current.numel():
        global_parent = parent[current]
        has_parent = global_parent >= 0
        child_local = current[has_parent]
        if not bool(has_parent.any()):
            break
        parent_local = local_of[global_parent[has_parent]]
        if bool((parent_local < 0).any()):
            raise ValueError("a memory chunk split a tree dependency chain")

        # Accumulate completed child rows; division happens exactly once when
        # all abstract children of a parent have arrived.
        p_mass.index_add_(0, parent_local, p_mass[child_local])
        m_mass.index_add_(0, parent_local, m_mass[child_local])
        for values in aux_values:
            values.index_add_(0, parent_local, values[child_local])
        processed.index_add_(0, parent_local, torch.ones_like(parent_local))
        candidates = parent_local.unique()
        ready = candidates[processed[candidates] == counts[candidates]]
        divisor = counts[ready].to(torch.float32)
        p_mass[ready] /= divisor[:, None, None]
        m_mass[ready] /= divisor[:, None, None]
        for values in aux_values:
            values[ready] /= divisor.reshape((ready.numel(),) + (1,) * (values.ndim - 1))

        global_ready = nodes[ready]
        decision = layout.decision_row[global_ready] >= 0
        if bool(decision.any()):
            dec_local = ready[decision]
            dec_nodes = global_ready[decision]
            rows = layout.decision_row[dec_nodes]
            players = tree.owner[dec_nodes].to(torch.int64)
            hrows = layout.head_row[dec_nodes]
            dec_counts = counts[dec_local]
            starts = tree.child_ptr[dec_nodes].to(torch.int64)
            cumulative = torch.cumsum(dec_counts, 0) - dec_counts
            slots = torch.arange(
                int(dec_counts.sum()), device=layout.device
            ) - torch.repeat_interleave(cumulative, dec_counts)
            edges = torch.repeat_interleave(starts, dec_counts) + slots
            child_nodes = tree.child_node[edges].to(torch.int64)
            child_rows = local_of[child_nodes]
            emit(
                result, rows, players, p_mass[dec_local], m_mass[dec_local],
                p_mass[child_rows], m_mass[child_rows], slots,
                heads.v[hrows].detach().to(torch.float32), lam,
            )
            for values, channel in zip(aux_values, channels):
                if channel.spec.emit_field is not None:
                    result.aux[channel.spec.emit_field][rows] = values[dec_local]
                if channel.spec.bootstrap == "every_player_node":
                    values[dec_local] = float(lam) * values[dec_local] + (1.0 - float(lam)) * channel.head[hrows]
            owner_m = m_mass[dec_local, players]
            owner_p = p_mass[dec_local, players]
            old_v = heads.v[hrows].detach().to(torch.float32).masked_fill(~layout.type_mask[players], 0.0)
            m_mass[dec_local, players] = bootstrap(owner_m, owner_p, old_v, lam)
        current = ready


def branched_value(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    reach: Any,
    heads: HeadEvals,
    lam: float,
    game: Any,
    *,
    configured_node_budget: int,
    emit_q: bool = False,
    auxiliary_reducers: Sequence[AuxiliaryChannelSpec] = (),
    max_resident_rows: int = 2_000_000,
) -> NodeTargets:
    """Propagate posterior moments and emit decision targets.

    Leaf tensors are ``[L,P,X]`` and head rows are ``[Nd,X]``.  Returned
    parent fields are ``[Nd,X]`` and child fields are padded ``[Nd,E,X]``.
    Complete trees are selected in fixed contiguous tree-id ranges and a
    dependency chain is never split across chunks.
    """

    if configured_node_budget <= 0:
        raise ValueError("configured_node_budget must be positive")
    if max_resident_rows <= 0:
        raise ValueError("max_resident_rows must be positive")
    layout = _layout(tree, leaf_stats, reach, heads, lam, game)
    if configured_node_budget * layout.p_max > max_resident_rows:
        raise ValueError("configured tree can exceed max_resident_rows")
    if layout.n:
        _, actual_counts = tree.tree_id.to(torch.int64).unique(return_counts=True)
        if bool((actual_counts > configured_node_budget).any()):
            raise ValueError("stored tree exceeds configured_node_budget")
        if bool((tree.tree_id < 0).any()):
            raise ValueError("tree ids must be nonnegative")

    channels = _channels(tuple(auxiliary_reducers), leaf_stats, heads, layout)
    result = _empty_targets(layout, bool(emit_q), channels)
    _edge_metadata(result, tree, layout)
    if layout.n == 0:
        return result

    trees_per_chunk = max(1, max_resident_rows // (configured_node_budget * layout.p_max))
    tree_ids = tree.tree_id.to(torch.int64)
    max_tree_id = int(tree_ids.max())
    for first in range(0, max_tree_id + 1, trees_per_chunk):
        selected = ((tree_ids >= first) & (tree_ids < first + trees_per_chunk)).nonzero(as_tuple=False).squeeze(1)
        if selected.numel():
            _process_chunk(tree, leaf_stats, heads, lam, layout, result, channels, selected)
    return result
