"""Posterior-mass backward propagation on labeled sampled trees.

This is the corrected value engine staged alongside :mod:`pbg.core.values`.
It propagates posterior bucket mass ``P`` and posterior-weighted return mass
``M``; conditional values are formed only when decision targets are emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

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


@dataclass(frozen=True)
class BackwardFrontierMemoryPlan:
    """Conservative bytes resident while reducing completed tree regions.

    ``frontier_bytes`` counts the persistent ``P/M`` boundary.  ``scratch_bytes``
    conservatively allows three additional copies of that boundary while a set
    of ready siblings is gathered, reduced, and compacted.  Unlike the former
    abstract row cap, every term includes private-type width and dtype size.
    """

    leaf_input_bytes: int
    target_bytes: int
    frontier_bytes: int
    scratch_bytes: int
    structure_bytes: int
    max_frontier_rows: int

    @property
    def peak_bytes(self) -> int:
        return (
            self.leaf_input_bytes
            + self.target_bytes
            + self.frontier_bytes
            + self.scratch_bytes
            + self.structure_bytes
        )


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
    device = leaf_stats.p_hat.device
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

    leaf_nodes = _leaf_node_ids(tree, leaf_stats)
    return _layout_from_shape(
        tree,
        leaf_nodes,
        p_max,
        x_max,
        device,
        reach,
        heads,
        lam,
        game,
    )


def _layout_from_shape(
    tree: TreeBatch,
    leaf_nodes: Tensor,
    p_max: int,
    x_max: int,
    device: torch.device,
    reach: Any,
    heads: HeadEvals,
    lam: float,
    game: Any,
) -> _Layout:
    """Validate topology and construct layout without allocating leaf values."""

    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lam must be in [0,1]")
    tree.validate()
    n = int(tree.parent.numel())
    leaf_nodes = leaf_nodes.to(device=device, dtype=torch.int64)
    counts = tree.child_ptr[1:].to(torch.int64) - tree.child_ptr[:-1].to(torch.int64)
    default_leaves = (counts == 0).nonzero(as_tuple=False).squeeze(1)
    if (
        leaf_nodes.ndim != 1
        or leaf_nodes.numel() != default_leaves.numel()
        or leaf_nodes.unique().numel() != leaf_nodes.numel()
        or not torch.equal(leaf_nodes.sort().values, default_leaves.sort().values)
    ):
        raise ValueError("leaf node ids must uniquely cover every stored leaf")

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

    leaf_row = torch.full((n,), -1, dtype=torch.int64, device=device)
    leaf_row[leaf_nodes] = torch.arange(leaf_nodes.numel(), device=device)
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


def _compact_single_child_posterior(result: NodeTargets) -> None:
    """Share the identical parent/child posterior storage on trajectories."""

    if (
        result.child_p_hat.shape[1] == 1
        and result.child_count.numel()
        and bool((result.child_count == 1).all())
    ):
        if not torch.equal(result.child_p_hat[:, 0], result.parent_p_hat):
            raise RuntimeError("single-child posterior targets must be identical")
        result.child_p_hat = result.parent_p_hat[:, None]


def _tensor_bytes(value: Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def _frontier_memory_plan(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    layout: _Layout,
    channels: Sequence[_AuxiliaryChannel],
    emit_q: bool,
) -> BackwardFrontierMemoryPlan:
    """Plan the value phase in bytes before allocating its dense frontier."""

    return _frontier_memory_plan_from_shape(
        tree,
        layout,
        channels,
        emit_q,
        _tensor_bytes(leaf_stats.p_hat) + _tensor_bytes(leaf_stats.v_hat),
    )


def _frontier_memory_plan_from_shape(
    tree: TreeBatch,
    layout: _Layout,
    channels: Sequence[_AuxiliaryChannel],
    emit_q: bool,
    leaf_input_bytes: int,
) -> BackwardFrontierMemoryPlan:
    """Plan a full or incrementally supplied completion/value phase."""

    leaves = int(layout.leaf_nodes.numel())
    moment_row_bytes = layout.p_max * layout.x_max * torch.tensor([], dtype=torch.float32).element_size()
    frontier_bytes = 2 * leaves * moment_row_bytes
    auxiliary_frontier = 0
    auxiliary_targets = 0
    for channel in channels:
        trailing = 1
        for size in channel.leaf.shape[1:]:
            trailing *= int(size)
        auxiliary_frontier += leaves * trailing * 4
        if channel.spec.emit_field is not None:
            auxiliary_targets += int(layout.decision_nodes.numel()) * trailing * 4
    frontier_bytes += auxiliary_frontier
    nd, e_max, x_max = (
        int(layout.decision_nodes.numel()),
        layout.e_max,
        layout.x_max,
    )
    target_float_values = nd * x_max * (2 + 2 * e_max)
    if emit_q:
        target_float_values += nd * e_max * x_max
    target_bytes = (
        target_float_values * 4
        + nd * 4  # node ids
        + nd * e_max * 4  # child event ids
        + nd * e_max  # child mask
        + nd * 4  # child counts
        + auxiliary_targets
    )
    # Parent/child gathers, newly reduced rows, and the compacted next frontier
    # can overlap with the old frontier in PyTorch's caching allocator.
    scratch_bytes = 3 * frontier_bytes
    structure_bytes = layout.n * (8 * 5 + 1) + int(tree.child_node.numel()) * 8
    return BackwardFrontierMemoryPlan(
        leaf_input_bytes=leaf_input_bytes,
        target_bytes=target_bytes,
        frontier_bytes=frontier_bytes,
        scratch_bytes=scratch_bytes,
        structure_bytes=structure_bytes,
        max_frontier_rows=leaves,
    )


def _reduce_frontier(
    tree: TreeBatch,
    leaf_stats: BatchedCompletionStats,
    heads: HeadEvals,
    lam: float,
    layout: _Layout,
    result: NodeTargets,
    channels: Sequence[_AuxiliaryChannel],
) -> int:
    """Contract completed siblings into parent ``P/M`` boundary summaries.

    The frontier starts at the pre-terminal leaves.  Whenever all children of
    a parent are present, their targets are emitted and those child rows are
    replaced by one parent row.  Consequently no ``[N,P,X]`` moment arena is
    allocated; the persistent moment state never exceeds the initial leaf cut.
    """

    frontier_nodes = layout.leaf_nodes
    frontier_p = (
        leaf_stats.p_hat.detach().to(torch.float32)
        .masked_fill(~layout.type_mask[None], 0.0)
    )
    leaf_v = (
        leaf_stats.v_hat.detach().to(torch.float32)
        .masked_fill(~layout.type_mask[None], 0.0)
    )
    leaf_v = torch.where(frontier_p > 0, leaf_v, torch.zeros_like(leaf_v))
    frontier_m = frontier_p * leaf_v
    frontier_aux = [channel.leaf.detach().to(torch.float32) for channel in channels]
    peak_rows = int(frontier_nodes.numel())

    while frontier_nodes.numel():
        frontier_parent = tree.parent[frontier_nodes].to(torch.int64)
        has_parent = frontier_parent >= 0
        if not bool(has_parent.any()):
            break
        boundary_count = torch.bincount(
            frontier_parent[has_parent], minlength=layout.n
        ).to(torch.int64)
        ready = (
            (layout.child_count_all > 0)
            & (boundary_count == layout.child_count_all)
        ).nonzero(as_tuple=False).squeeze(1)
        if ready.numel() == 0:
            raise RuntimeError("backward frontier stalled before reaching every root")

        counts = layout.child_count_all[ready].to(torch.int64)
        starts = tree.child_ptr[ready].to(torch.int64)
        cumulative = torch.cumsum(counts, 0) - counts
        slots = torch.arange(
            int(counts.sum()), device=layout.device
        ) - torch.repeat_interleave(cumulative, counts)
        edges = torch.repeat_interleave(starts, counts) + slots
        child_nodes = tree.child_node[edges].to(torch.int64)
        frontier_row = torch.full(
            (layout.n,), -1, dtype=torch.int64, device=layout.device
        )
        frontier_row[frontier_nodes] = torch.arange(
            frontier_nodes.numel(), device=layout.device
        )
        child_rows = frontier_row[child_nodes]
        if bool((child_rows < 0).any()):
            raise RuntimeError("a ready parent is missing a child boundary summary")

        parent_rows = torch.repeat_interleave(
            torch.arange(ready.numel(), device=layout.device), counts
        )
        parent_p = torch.zeros(
            (ready.numel(), layout.p_max, layout.x_max),
            dtype=torch.float32,
            device=layout.device,
        )
        parent_m = torch.zeros_like(parent_p)
        parent_p.index_add_(0, parent_rows, frontier_p[child_rows])
        parent_m.index_add_(0, parent_rows, frontier_m[child_rows])
        divisor = counts.to(torch.float32)[:, None, None]
        parent_p /= divisor
        parent_m /= divisor

        parent_aux: list[Tensor] = []
        for values in frontier_aux:
            reduced = torch.zeros(
                (ready.numel(), *values.shape[1:]),
                dtype=torch.float32,
                device=layout.device,
            )
            reduced.index_add_(0, parent_rows, values[child_rows])
            reduced /= counts.to(torch.float32).reshape(
                (ready.numel(),) + (1,) * (values.ndim - 1)
            )
            parent_aux.append(reduced)

        decision = layout.decision_row[ready] >= 0
        if bool(decision.any()):
            dec_local = decision.nonzero(as_tuple=False).squeeze(1)
            dec_nodes = ready[dec_local]
            rows = layout.decision_row[dec_nodes]
            players = tree.owner[dec_nodes].to(torch.int64)
            hrows = layout.head_row[dec_nodes]
            dec_counts = counts[dec_local]
            dec_edge_mask = torch.repeat_interleave(decision, counts)
            emit(
                result,
                rows,
                players,
                parent_p[dec_local],
                parent_m[dec_local],
                frontier_p[child_rows[dec_edge_mask]],
                frontier_m[child_rows[dec_edge_mask]],
                slots[dec_edge_mask],
                heads.v[hrows].detach().to(torch.float32),
                lam,
            )
            for values, channel in zip(parent_aux, channels):
                if channel.spec.emit_field is not None:
                    result.aux[channel.spec.emit_field][rows] = values[dec_local]
                if channel.spec.bootstrap == "every_player_node":
                    values[dec_local] = (
                        float(lam) * values[dec_local]
                        + (1.0 - float(lam)) * channel.head[hrows]
                    )
            owner_m = parent_m[dec_local, players]
            owner_p = parent_p[dec_local, players]
            old_v = (
                heads.v[hrows].detach().to(torch.float32)
                .masked_fill(~layout.type_mask[players], 0.0)
            )
            parent_m[dec_local, players] = bootstrap(owner_m, owner_p, old_v, lam)

        consumed = torch.zeros(
            frontier_nodes.numel(), dtype=torch.bool, device=layout.device
        )
        consumed[child_rows] = True
        frontier_nodes = torch.cat((frontier_nodes[~consumed], ready))
        frontier_p = torch.cat((frontier_p[~consumed], parent_p))
        frontier_m = torch.cat((frontier_m[~consumed], parent_m))
        frontier_aux = [
            torch.cat((old[~consumed], new))
            for old, new in zip(frontier_aux, parent_aux)
        ]
        peak_rows = max(peak_rows, int(frontier_nodes.numel()))

    roots = (tree.parent < 0).nonzero(as_tuple=False).squeeze(1).to(torch.int64)
    if frontier_nodes.numel() != roots.numel() or not torch.equal(
        frontier_nodes.sort().values, roots.sort().values
    ):
        raise RuntimeError("backward frontier did not reduce to exactly the tree roots")
    return peak_rows


def _reduce_streamed_leaf_batches(
    tree: TreeBatch,
    batches: Iterable[tuple[Tensor, BatchedCompletionStats]],
    heads: HeadEvals,
    lam: float,
    layout: _Layout,
    result: NodeTargets,
) -> int:
    """Incrementally complete leaves and reduce every newly ready subtree."""

    frontier_nodes = torch.empty(0, dtype=torch.int64, device=layout.device)
    frontier_p = torch.empty(
        (0, layout.p_max, layout.x_max), dtype=torch.float32, device=layout.device
    )
    frontier_m = torch.empty_like(frontier_p)
    seen = torch.zeros(layout.n, dtype=torch.bool, device=layout.device)
    peak_rows = 0

    for node_ids, stats in batches:
        node_ids = node_ids.to(device=layout.device, dtype=torch.int64)
        if node_ids.numel() == 0:
            raise ValueError("streamed completion batches cannot be empty")
        if stats.p_hat.shape != (node_ids.numel(), layout.p_max, layout.x_max):
            raise ValueError("streamed leaf p_hat has the wrong shape")
        if stats.v_hat.shape != stats.p_hat.shape:
            raise ValueError("streamed leaf v_hat must match p_hat")
        _reject_low_precision("streamed leaf p_hat", stats.p_hat)
        _reject_low_precision("streamed leaf v_hat", stats.v_hat)
        if node_ids.unique().numel() != node_ids.numel():
            raise ValueError("a streamed completion batch contains duplicate leaves")
        valid = (node_ids >= 0) & (node_ids < layout.n)
        if bool(valid.all()):
            valid &= layout.leaf_row[node_ids] >= 0
            valid &= ~seen[node_ids]
        if not bool(valid.all()):
            raise ValueError("streamed completion leaves must be unique unconsumed leaves")
        if not bool(torch.isfinite(stats.p_hat).all()) or bool((stats.p_hat < 0).any()):
            raise ValueError("streamed posterior mass must be finite and nonnegative")
        if not bool(torch.isfinite(stats.v_hat).all()):
            raise ValueError("streamed leaf values must be finite")
        seen[node_ids] = True

        added_p = (
            stats.p_hat.detach().to(torch.float32)
            .masked_fill(~layout.type_mask[None], 0.0)
        )
        added_v = (
            stats.v_hat.detach().to(torch.float32)
            .masked_fill(~layout.type_mask[None], 0.0)
        )
        added_v = torch.where(added_p > 0, added_v, torch.zeros_like(added_v))
        frontier_nodes = torch.cat((frontier_nodes, node_ids))
        frontier_p = torch.cat((frontier_p, added_p))
        frontier_m = torch.cat((frontier_m, added_p * added_v))
        peak_rows = max(peak_rows, int(frontier_nodes.numel()))

        contracted = False
        while frontier_nodes.numel():
            frontier_parent = tree.parent[frontier_nodes].to(torch.int64)
            has_parent = frontier_parent >= 0
            if not bool(has_parent.any()):
                break
            boundary_count = torch.bincount(
                frontier_parent[has_parent], minlength=layout.n
            ).to(torch.int64)
            ready = (
                (layout.child_count_all > 0)
                & (boundary_count == layout.child_count_all)
            ).nonzero(as_tuple=False).squeeze(1)
            if ready.numel() == 0:
                break
            contracted = True

            counts = layout.child_count_all[ready].to(torch.int64)
            starts = tree.child_ptr[ready].to(torch.int64)
            cumulative = torch.cumsum(counts, 0) - counts
            slots = torch.arange(
                int(counts.sum()), device=layout.device
            ) - torch.repeat_interleave(cumulative, counts)
            edges = torch.repeat_interleave(starts, counts) + slots
            child_nodes = tree.child_node[edges].to(torch.int64)
            frontier_row = torch.full(
                (layout.n,), -1, dtype=torch.int64, device=layout.device
            )
            frontier_row[frontier_nodes] = torch.arange(
                frontier_nodes.numel(), device=layout.device
            )
            child_rows = frontier_row[child_nodes]
            if bool((child_rows < 0).any()):
                raise RuntimeError("a ready parent is missing a child boundary summary")

            parent_rows = torch.repeat_interleave(
                torch.arange(ready.numel(), device=layout.device), counts
            )
            parent_p = torch.zeros(
                (ready.numel(), layout.p_max, layout.x_max),
                dtype=torch.float32,
                device=layout.device,
            )
            parent_m = torch.zeros_like(parent_p)
            parent_p.index_add_(0, parent_rows, frontier_p[child_rows])
            parent_m.index_add_(0, parent_rows, frontier_m[child_rows])
            divisor = counts.to(torch.float32)[:, None, None]
            parent_p /= divisor
            parent_m /= divisor

            decision = layout.decision_row[ready] >= 0
            if bool(decision.any()):
                dec_local = decision.nonzero(as_tuple=False).squeeze(1)
                dec_nodes = ready[dec_local]
                rows = layout.decision_row[dec_nodes]
                players = tree.owner[dec_nodes].to(torch.int64)
                hrows = layout.head_row[dec_nodes]
                dec_edge_mask = torch.repeat_interleave(decision, counts)
                emit(
                    result,
                    rows,
                    players,
                    parent_p[dec_local],
                    parent_m[dec_local],
                    frontier_p[child_rows[dec_edge_mask]],
                    frontier_m[child_rows[dec_edge_mask]],
                    slots[dec_edge_mask],
                    heads.v[hrows].detach().to(torch.float32),
                    lam,
                )
                owner_m = parent_m[dec_local, players]
                owner_p = parent_p[dec_local, players]
                old_v = (
                    heads.v[hrows].detach().to(torch.float32)
                    .masked_fill(~layout.type_mask[players], 0.0)
                )
                parent_m[dec_local, players] = bootstrap(
                    owner_m, owner_p, old_v, lam
                )

            consumed = torch.zeros(
                frontier_nodes.numel(), dtype=torch.bool, device=layout.device
            )
            consumed[child_rows] = True
            frontier_nodes = torch.cat((frontier_nodes[~consumed], ready))
            frontier_p = torch.cat((frontier_p[~consumed], parent_p))
            frontier_m = torch.cat((frontier_m[~consumed], parent_m))
            peak_rows = max(peak_rows, int(frontier_nodes.numel()))

        del stats, added_p, added_v, frontier_parent, has_parent
        if contracted:
            del parent_p, parent_m, child_rows, parent_rows, divisor, consumed

    if not bool(seen[layout.leaf_nodes].all()):
        raise ValueError("streamed completion did not cover every stored leaf")
    roots = (tree.parent < 0).nonzero(as_tuple=False).squeeze(1).to(torch.int64)
    if frontier_nodes.numel() != roots.numel() or not torch.equal(
        frontier_nodes.sort().values, roots.sort().values
    ):
        raise RuntimeError("streamed backward frontier did not reduce to every root")
    return peak_rows


def streamed_branched_value(
    tree: TreeBatch,
    leaf_nodes: Tensor,
    completion_batches: Iterable[tuple[Tensor, BatchedCompletionStats]],
    reach: Any,
    heads: HeadEvals,
    lam: float,
    game: Any,
    *,
    configured_node_budget: int,
    completion_batch_rows: int,
    emit_q: bool = False,
    max_workspace_bytes: int = 1 << 30,
) -> NodeTargets:
    """Construct targets while completion batches feed a ``P/M`` frontier."""

    if configured_node_budget <= 0:
        raise ValueError("configured_node_budget must be positive")
    if completion_batch_rows <= 0:
        raise ValueError("completion_batch_rows must be positive")
    if isinstance(max_workspace_bytes, bool) or max_workspace_bytes <= 0:
        raise ValueError("max_workspace_bytes must be a positive byte count")
    device = tree.parent.device
    layout = _layout_from_shape(
        tree,
        leaf_nodes,
        int(game.n_players),
        max(int(size) for size in game.x_sizes),
        device,
        reach,
        heads,
        lam,
        game,
    )
    if layout.n:
        _, actual_counts = tree.tree_id.to(torch.int64).unique(return_counts=True)
        if bool((actual_counts > configured_node_budget).any()):
            raise ValueError("stored tree exceeds configured_node_budget")
    # p_hat, v_hat, estimator_stddev and output_valid conservatively represent
    # the dense learning/quality fields returned by active completion backends.
    completion_bytes = (
        min(completion_batch_rows, int(leaf_nodes.numel()))
        * layout.p_max
        * layout.x_max
        * (3 * 4 + 1)
    )
    plan = _frontier_memory_plan_from_shape(
        tree, layout, (), bool(emit_q), completion_bytes
    )
    if plan.peak_bytes > max_workspace_bytes:
        gib = 1024**3
        raise MemoryError(
            "streamed completion/backward plan requires "
            f"{plan.peak_bytes / gib:.3f} GiB, exceeding the configured "
            f"{max_workspace_bytes / gib:.3f} GiB byte budget"
        )
    result = _empty_targets(layout, bool(emit_q), ())
    _edge_metadata(result, tree, layout)
    if layout.n:
        _reduce_streamed_leaf_batches(
            tree, completion_batches, heads, lam, layout, result
        )
    _compact_single_child_posterior(result)
    return result


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
    max_workspace_bytes: int = 1 << 30,
) -> NodeTargets:
    """Propagate posterior moments and emit decision targets.

    Leaf tensors are ``[L,P,X]`` and head rows are ``[Nd,X]``.  Returned
    parent fields are ``[Nd,X]`` and child fields are padded ``[Nd,E,X]``.
    Completed siblings are reduced through a ``P/M`` boundary frontier.  The
    persistent moment workspace is therefore bounded by the number of leaves,
    not the total number of stored nodes.
    """

    if configured_node_budget <= 0:
        raise ValueError("configured_node_budget must be positive")
    if isinstance(max_workspace_bytes, bool) or max_workspace_bytes <= 0:
        raise ValueError("max_workspace_bytes must be a positive byte count")
    layout = _layout(tree, leaf_stats, reach, heads, lam, game)
    if layout.n:
        _, actual_counts = tree.tree_id.to(torch.int64).unique(return_counts=True)
        if bool((actual_counts > configured_node_budget).any()):
            raise ValueError("stored tree exceeds configured_node_budget")
        if bool((tree.tree_id < 0).any()):
            raise ValueError("tree ids must be nonnegative")

    channels = _channels(tuple(auxiliary_reducers), leaf_stats, heads, layout)
    plan = _frontier_memory_plan(
        tree, leaf_stats, layout, channels, bool(emit_q)
    )
    if plan.peak_bytes > max_workspace_bytes:
        gib = 1024**3
        raise MemoryError(
            "backward frontier plan requires "
            f"{plan.peak_bytes / gib:.3f} GiB, exceeding the configured "
            f"{max_workspace_bytes / gib:.3f} GiB byte budget"
        )
    result = _empty_targets(layout, bool(emit_q), channels)
    _edge_metadata(result, tree, layout)
    if layout.n == 0:
        return result
    _reduce_frontier(tree, leaf_stats, heads, lam, layout, result, channels)
    _compact_single_child_posterior(result)
    return result
