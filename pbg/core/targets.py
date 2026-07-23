"""Streaming target builders over sealed rollout storage.

The builders retain only v1 loss arguments.  In particular they never expose
``qHat``, ``W``, ``S``, a duplicated sequential advantage/value target, or a
branched parent-posterior array.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor

from pbg.core.adaptive_branching import gamma_deviance
from pbg.core.partition import CompletionSpec, PartitionSet
from pbg.core.storage import (
    RolloutStorage,
    SamplingMetadata,
    SequentialPosteriorPool,
    TrainingPartition,
    TrainingStorage,
)
from pbg.core.topology import BranchedTreeSpan, SequentialDecisionCSR


def conditionalize(moment: Tensor, mass: Tensor) -> Tensor:
    """Return ``moment / mass`` with exact zeros where posterior mass is zero."""

    valid = mass > 0
    return torch.where(valid, moment / torch.where(valid, mass, torch.ones_like(mass)), 0)


def _column(block: Any, *names: str) -> Tensor:
    for name in names:
        if name in block.columns:
            return block.columns[name]
    raise KeyError(f"none of the required columns exist: {names}")


def _completion_field(result: Any, name: str) -> Tensor:
    if isinstance(result, Mapping):
        value = result[name]
    else:
        value = getattr(result, name)
    if not isinstance(value, Tensor):
        raise TypeError(f"completion output {name!r} must be a tensor")
    return value


def _validate_completion_boundary(
    completion: CompletionSpec, rollout: RolloutStorage, batch_rows: int
) -> None:
    declared = set(completion.leaf_fields) | set(completion.key_fields)
    missing = declared.difference(rollout.leaves.columns)
    if missing:
        raise ValueError(f"completion leaf store is missing fields {sorted(missing)}")
    if not {"p_hat", "v_hat"}.issubset(completion.output_fields):
        raise ValueError("completion must declare p_hat and v_hat output rows")
    completion.workspace_bound(batch_rows)


def _validate_lambda(lam: float) -> float:
    value = float(lam)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("lambda must be finite and lie in [0,1]")
    return value


def _phi_exact(old_value: Tensor, mapping: tuple[int, ...]) -> Tensor:
    index = torch.tensor(mapping, dtype=torch.int64, device=old_value.device)
    return old_value[..., index]


def _walker_alias(weights: Tensor) -> tuple[Tensor, Tensor]:
    """Build fp64 Walker thresholds and int32 aliases without retaining weights."""

    values = weights.detach().to(dtype=torch.float64, device="cpu").flatten()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("alias weights must be finite, nonnegative, and nonempty")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("alias weights must have positive mass")
    count = values.numel()
    scaled = (values * count / total).tolist()
    small = [index for index, value in enumerate(scaled) if value < 1.0]
    large = [index for index, value in enumerate(scaled) if value >= 1.0]
    probability = torch.ones(count, dtype=torch.float64)
    alias = torch.arange(count, dtype=torch.int32)
    while small and large:
        low = small.pop()
        high = large.pop()
        probability[low] = scaled[low]
        alias[low] = high
        scaled[high] -= 1.0 - scaled[low]
        (small if scaled[high] < 1.0 else large).append(high)
    return probability.to(weights.device), alias.to(weights.device)


def _sampling_metadata(
    rollout: RolloutStorage,
    *,
    branched: bool,
) -> SamplingMetadata:
    log_z: dict[int, Tensor] = {}
    probabilities: dict[int, Tensor] = {}
    aliases: dict[int, Tensor] = {}
    for partition_id, block in rollout.partitions.items():
        rows = block.decisions.rows
        if rows == 0:
            log_z[partition_id] = torch.tensor(
                -torch.inf,
                dtype=torch.float64,
                device=next(iter(block.decisions.columns.values())).device,
            )
            continue
        if branched:
            log_weight = _column(block.decisions, "log_weight", "logWeight").double()
            log_z[partition_id] = torch.logsumexp(log_weight, 0)
            probability, alias = _walker_alias(torch.exp(log_weight - log_z[partition_id]))
            probabilities[partition_id] = probability
            aliases[partition_id] = alias
        else:
            log_z[partition_id] = torch.tensor(
                math.log(rows), dtype=torch.float64, device=block.decisions.columns[next(iter(block.decisions.columns))].device
            )
    return SamplingMetadata(log_z, probabilities, aliases)


def _base_columns(block: Any) -> dict[str, Tensor]:
    """Detach rollout decision fields without cloning their storage."""

    return {name: value.detach() for name, value in block.decisions.columns.items()}


def _host_csr_pointer(pointer: Tensor) -> Tensor:
    """Copy one compact CSR pointer at a declared structural control boundary.

    Target payloads and recurrences remain on the accelerator.  This single
    bulk transfer supplies Python only with bounded trajectory-batch extents;
    it replaces per-trajectory/per-decision CUDA scalar extraction.
    """

    return pointer.detach().to(device="cpu", dtype=torch.int64)


def _flat_partition_layout(
    rollout: RolloutStorage,
    *,
    device: torch.device,
) -> tuple[dict[int, int], Tensor, int]:
    """Return immutable partition offsets and a device lookup table."""

    offsets: dict[int, int] = {}
    total = 0
    for partition_id, block in rollout.partitions.items():
        offsets[partition_id] = total
        total += block.decisions.rows
    if not offsets:
        raise ValueError("target construction requires at least one partition")
    lookup = torch.full(
        (max(offsets) + 1,), -1, dtype=torch.int64, device=device
    )
    for partition_id, offset in offsets.items():
        lookup[partition_id] = offset
    return offsets, lookup, total


def _batch_addresses(
    topology: SequentialDecisionCSR,
    pointer_host: Tensor,
    first: int,
    stop: int,
) -> tuple[int, int, Tensor, Tensor, Tensor]:
    """Return one bounded trajectory batch's device-local CSR addresses."""

    begin_ordinal = int(pointer_host[first])
    stop_ordinal = int(pointer_host[stop])
    count = stop_ordinal - begin_ordinal
    counts = (
        topology.trajectory_ptr[first + 1 : stop + 1]
        - topology.trajectory_ptr[first:stop]
    ).to(torch.int64)
    trajectory = torch.repeat_interleave(
        torch.arange(first, stop, dtype=torch.int64, device=counts.device),
        counts,
        output_size=count,
    )
    return (
        begin_ordinal,
        stop_ordinal,
        trajectory,
        topology.decision_partition[begin_ordinal:stop_ordinal].to(torch.int64),
        topology.decision_row[begin_ordinal:stop_ordinal].to(torch.int64),
    )


def _gather_batch_owner(
    rollout: RolloutStorage,
    partition: Tensor,
    row: Tensor,
) -> Tensor:
    """Gather partitioned owners without data-dependent CUDA output shapes."""

    owner = torch.zeros(row.numel(), dtype=torch.int64, device=row.device)
    covered = torch.zeros(row.numel(), dtype=torch.bool, device=row.device)
    for partition_id, block in rollout.partitions.items():
        if block.decisions.rows == 0:
            continue
        matches = partition == partition_id
        safe_row = torch.where(matches, row, torch.zeros_like(row))
        candidate = _column(block.decisions, "owner")[safe_row].to(torch.int64)
        owner = torch.where(matches, candidate, owner)
        covered |= matches
    torch._assert_async(covered.all(), "sequential target address has no partition")
    return owner


def _gather_batch_old_value(
    rollout: RolloutStorage,
    partitions: PartitionSet,
    partition: Tensor,
    row: Tensor,
    *,
    exact_width: int | None,
) -> Tensor:
    """Gather scalar or exact-expanded old values in ordinal order."""

    shape = (row.numel(),) if exact_width is None else (row.numel(), exact_width)
    old_value = torch.zeros(shape, dtype=torch.float32, device=row.device)
    covered = torch.zeros(row.numel(), dtype=torch.bool, device=row.device)
    for partition_id, block in rollout.partitions.items():
        if block.decisions.rows == 0:
            continue
        matches = partition == partition_id
        safe_row = torch.where(matches, row, torch.zeros_like(row))
        candidate = _column(block.decisions, "old_value", "oldV")[safe_row].float()
        if exact_width is not None:
            partitions[partition_id].assert_model_width(candidate)
            candidate = _phi_exact(candidate, partitions[partition_id].exact_to_model)
        mask = matches.reshape((matches.numel(),) + (1,) * (candidate.ndim - 1))
        old_value = torch.where(mask, candidate, old_value)
        covered |= matches
    torch._assert_async(covered.all(), "sequential target value has no partition")
    return old_value


def _player_clock_segmented_targets(
    old_value: Tensor,
    terminal_value: Tensor,
    trajectory: Tensor,
    owner: Tensor,
    lam: float,
) -> Tensor:
    """Evaluate the player-clock recurrence with a device segmented scan.

    Binary pointer jumping composes the affine updates belonging to later
    decisions of the same ``(trajectory, owner)``.  It uses logarithmically
    many device operations and never extracts an accelerator scalar.
    """

    count = owner.numel()
    if count == 0:
        return old_value.clone()
    n_players = terminal_value.shape[1]
    key = trajectory * n_players + owner
    order = torch.argsort(key, stable=True)
    ordered_key = key[order]
    successor = torch.full((count,), -1, dtype=torch.int64, device=owner.device)
    if count > 1:
        same = ordered_key[:-1] == ordered_key[1:]
        successor[order[:-1]] = torch.where(
            same, order[1:], torch.full_like(order[:-1], -1)
        )

    linked = successor >= 0
    safe_successor = successor.clamp_min(0)
    coefficient = torch.where(
        linked,
        torch.full((count,), lam, dtype=old_value.dtype, device=old_value.device),
        torch.ones(count, dtype=old_value.dtype, device=old_value.device),
    )
    linked_shape = (count,) + (1,) * (old_value.ndim - 1)
    intercept = torch.where(
        linked.reshape(linked_shape),
        (1.0 - lam) * old_value[safe_successor],
        torch.zeros_like(old_value),
    )
    link = successor
    for _ in range(max(1, count.bit_length())):
        has_link = link >= 0
        safe_link = link.clamp_min(0)
        next_coefficient = coefficient[safe_link]
        next_intercept = intercept[safe_link]
        combined_intercept = intercept + coefficient.reshape(linked_shape) * next_intercept
        coefficient = torch.where(
            has_link, coefficient * next_coefficient, coefficient
        )
        intercept = torch.where(
            has_link.reshape(linked_shape), combined_intercept, intercept
        )
        link = torch.where(has_link, link[safe_link], link)

    terminal = terminal_value[trajectory, owner]
    return coefficient.reshape(linked_shape) * terminal + intercept


def _flat_destinations(
    partition: Tensor,
    row: Tensor,
    offset_lookup: Tensor,
) -> Tensor:
    offset = offset_lookup[partition]
    torch._assert_async((offset >= 0).all(), "unknown sequential target partition")
    return offset + row


def build_normal_targets(
    rollout: RolloutStorage,
    partitions: PartitionSet,
    lam: float,
    *,
    trajectory_batch_rows: int = 256,
) -> TrainingStorage:
    """Build scalar terminal-trajectory player-clock lambda returns."""

    lam = _validate_lambda(lam)
    if rollout.mode != "normal" or not isinstance(rollout.topology, SequentialDecisionCSR):
        raise ValueError("normal targets require normal sequential rollout storage")
    if trajectory_batch_rows <= 0:
        raise ValueError("trajectory_batch_rows must be positive")
    topology = rollout.topology
    partition_rows = {
        partition_id: block.decisions.rows for partition_id, block in rollout.partitions.items()
    }
    topology.validate(partition_rows, rollout.leaves.rows)
    reward = _column(rollout.leaves, "reward", "terminal_reward").float()
    if reward.ndim != 2:
        raise ValueError("normal leaf reward must be fp32 [L,P]")
    offsets, offset_lookup, decision_count = _flat_partition_layout(
        rollout, device=reward.device
    )
    flat_return = torch.empty(decision_count, dtype=torch.float32, device=reward.device)
    valid = torch.zeros(decision_count, dtype=torch.bool, device=reward.device)
    pointer_host = _host_csr_pointer(topology.trajectory_ptr)
    for first in range(0, topology.roots, trajectory_batch_rows):
        batch_stop = min(first + trajectory_batch_rows, topology.roots)
        _, _, trajectory, partition, row = _batch_addresses(
            topology, pointer_host, first, batch_stop
        )
        owner = _gather_batch_owner(rollout, partition, row)
        old_value = _gather_batch_old_value(
            rollout, partitions, partition, row, exact_width=None
        )
        terminal = reward[topology.leaf_row[first:batch_stop].to(torch.int64)]
        target = _player_clock_segmented_targets(
            old_value, terminal, trajectory - first, owner, lam
        )
        destination = _flat_destinations(partition, row, offset_lookup)
        flat_return[destination] = target
        valid[destination] = True
    torch._assert_async(valid.all(), "normal target builder did not write every decision once")
    training: dict[int, TrainingPartition] = {}
    for partition_id, block in rollout.partitions.items():
        columns = _base_columns(block)
        begin = offsets[partition_id]
        columns["lambda_return"] = flat_return[begin : begin + block.decisions.rows]
        # Algebraic duplicates are explicitly absent.
        columns.pop("advantage", None)
        columns.pop("v_target", None)
        training[partition_id] = TrainingPartition(
            partition_id,
            block.inputs,
            columns,
            input_pools=block.input_pools,
        )
    return TrainingStorage(
        "normal", training, _sampling_metadata(rollout, branched=False)
    )


def build_sequential_rb_targets(
    rollout: RolloutStorage,
    partitions: PartitionSet,
    completion: CompletionSpec,
    lam: float,
    *,
    leaf_microbatch_rows: int = 64,
) -> TrainingStorage:
    """Build deduplicated sequential Rao--Blackwell exact-hand targets."""

    lam = _validate_lambda(lam)
    if rollout.mode != "sequential_rb" or not isinstance(
        rollout.topology, SequentialDecisionCSR
    ):
        raise ValueError("sequential RB targets require sequential RB rollout storage")
    if leaf_microbatch_rows <= 0:
        raise ValueError("leaf_microbatch_rows must be positive")
    _validate_completion_boundary(completion, rollout, leaf_microbatch_rows)
    topology = rollout.topology
    partition_rows = {key: value.decisions.rows for key, value in rollout.partitions.items()}
    topology.validate(partition_rows, rollout.leaves.rows)
    widths = {partitions[key].exact_width for key in rollout.partitions}
    if len(widths) != 1:
        raise NotImplementedError(
            "the v1 sequential posterior pool requires one canonical exact width per run"
        )
    exact_width = widths.pop()
    first_block = next(iter(rollout.partitions.values()))
    device = next(iter(first_block.decisions.columns.values())).device
    offsets, offset_lookup, decision_count = _flat_partition_layout(
        rollout, device=device
    )
    flat_return = torch.empty(
        (decision_count, exact_width), dtype=torch.float32, device=device
    )
    flat_ref = torch.full((decision_count,), -1, dtype=torch.int32, device=device)
    valid = torch.zeros(decision_count, dtype=torch.bool, device=device)
    pointer_host = _host_csr_pointer(topology.trajectory_ptr)

    # Mark the exact set of required (trajectory, player) posterior rows using
    # bounded device batches.  P is a run constant (2..6), so this structural
    # bitmap remains independent of the exact-hand width.
    if decision_count == 0:
        raise ValueError("sequential RB target construction requires a decision")
    # Owners are immutable scalar address metadata.  One bulk copy determines
    # the compact player bitmap width without inspecting CUDA values in a loop.
    owner_structure = torch.empty(
        decision_count, dtype=torch.int64, device=device
    )
    for partition_id, block in rollout.partitions.items():
        begin = offsets[partition_id]
        owner_structure[begin : begin + block.decisions.rows] = _column(
            block.decisions, "owner"
        )
    owner_host = owner_structure.to(device="cpu")
    del owner_structure
    selected_owner_field = completion.selected_owner_field
    selected_owner = None
    if selected_owner_field is not None:
        selected_owner = _column(rollout.leaves, selected_owner_field).to(torch.int64)
        if selected_owner.shape != (rollout.leaves.rows,):
            raise ValueError("selected completion owners must be leaf-aligned [L]")
        selected_owner_host = selected_owner.detach().to(device="cpu")
        n_players = max(int(owner_host.max()), int(selected_owner_host.max())) + 1
    else:
        n_players = int(owner_host.max()) + 1
    required = torch.zeros(
        (topology.roots, n_players), dtype=torch.bool, device=device
    )
    for first in range(0, topology.roots, leaf_microbatch_rows):
        batch_stop = min(first + leaf_microbatch_rows, topology.roots)
        _, _, trajectory, partition, row = _batch_addresses(
            topology, pointer_host, first, batch_stop
        )
        owner = _gather_batch_owner(rollout, partition, row)
        required[trajectory, owner] = True
    if selected_owner is not None:
        trajectory_selected = selected_owner[
            topology.leaf_row.to(torch.int64)
        ]
        torch._assert_async(
            ((trajectory_selected >= 0) & (trajectory_selected < n_players)).all(),
            "selected completion owner is out of range",
        )
        selected_required = required.gather(
            1, trajectory_selected[:, None]
        ).squeeze(1)
        retained_count = required.sum(1)
        torch._assert_async(
            (
                (retained_count <= 1)
                & ((retained_count == 0) | selected_required)
            ).all(),
            "probe trajectories must retain decisions for only their selected owner",
        )
    pool_counts = required.sum(1, dtype=torch.int64)
    pool_ptr = torch.empty(topology.roots + 1, dtype=torch.int64, device=device)
    pool_ptr[0] = 0
    pool_ptr[1:] = pool_counts.cumsum(0)
    # Exact allocation requires one scalar total.  Transfer the complete small
    # pointer once so this is an explicit bounded allocation boundary, not a
    # synchronization in a trajectory or decision loop.
    pool_ptr_host = _host_csr_pointer(pool_ptr)
    pool_rows = int(pool_ptr_host[-1])
    required_flat = required.flatten()
    required_order = torch.argsort((~required_flat).to(torch.int8), stable=True)[
        :pool_rows
    ]
    pool_trajectory = torch.div(
        required_order, n_players, rounding_mode="floor"
    ).to(torch.int32)
    pool_player = torch.remainder(required_order, n_players).to(torch.int32)
    pair_ref = torch.full_like(required_flat, -1, dtype=torch.int32)
    pair_ref[required_order] = torch.arange(
        pool_rows, dtype=torch.int32, device=device
    )
    pair_ref = pair_ref.reshape(topology.roots, n_players)
    posterior_tensor = torch.empty(
        (pool_rows, exact_width), dtype=torch.float32, device=device
    )
    pool_valid = torch.zeros(pool_rows, dtype=torch.bool, device=device)

    for first in range(0, topology.roots, leaf_microbatch_rows):
        trajectories = torch.arange(
            first, min(first + leaf_microbatch_rows, topology.roots),
            dtype=torch.int64, device=device,
        )
        leaf_rows = topology.leaf_row[trajectories].to(torch.int64)
        completed = completion.complete(rollout.leaves.columns, rows=leaf_rows)
        posterior = _completion_field(completed, "p_hat").float()
        value = _completion_field(completed, "v_hat").float()
        if posterior.shape != value.shape or posterior.ndim != 3:
            raise ValueError("completion p_hat/v_hat must be matching [B,P,X]")
        if posterior.shape[0] != trajectories.numel() or posterior.shape[2] != exact_width:
            raise ValueError("completion output violates the canonical exact domain")
        if selected_owner is not None and posterior.shape[1] != 1:
            raise ValueError("selected completion must return exactly one player row")
        pool_begin = int(pool_ptr_host[first])
        pool_stop = int(pool_ptr_host[min(first + leaf_microbatch_rows, topology.roots)])
        batch_pool_player = pool_player[pool_begin:pool_stop].to(torch.int64)
        batch_pool_trajectory = (
            pool_trajectory[pool_begin:pool_stop].to(torch.int64) - first
        )
        output_player = (
            torch.zeros_like(batch_pool_player)
            if selected_owner is not None
            else batch_pool_player
        )
        posterior_tensor[pool_begin:pool_stop] = posterior[
            batch_pool_trajectory, output_player
        ]
        pool_valid[pool_begin:pool_stop] = True

        batch_stop = min(first + leaf_microbatch_rows, topology.roots)
        _, _, trajectory, partition, row = _batch_addresses(
            topology, pointer_host, first, batch_stop
        )
        local_trajectory = trajectory - first
        owner = _gather_batch_owner(rollout, partition, row)
        old_value = _gather_batch_old_value(
            rollout, partitions, partition, row, exact_width=exact_width
        )
        target_owner = torch.zeros_like(owner) if selected_owner is not None else owner
        support = posterior[local_trajectory, target_owner] > 0
        old_value = torch.where(support, old_value, torch.zeros_like(old_value))
        terminal = torch.where(
            posterior > 0, value, torch.zeros_like(value)
        )
        target = _player_clock_segmented_targets(
            old_value, terminal, local_trajectory, target_owner, lam
        )
        destination = _flat_destinations(partition, row, offset_lookup)
        flat_return[destination] = target
        flat_ref[destination] = pair_ref[trajectory, owner]
        valid[destination] = True
        del completed, posterior, value
    torch._assert_async(pool_valid.all(), "sequential posterior pool has unwritten rows")
    torch._assert_async(valid.all(), "sequential RB target builder missed a decision")
    pool = SequentialPosteriorPool(
        posterior_tensor,
        pool_trajectory,
        pool_player,
    )
    training: dict[int, TrainingPartition] = {}
    for partition_id, block in rollout.partitions.items():
        begin = offsets[partition_id]
        stop = begin + block.decisions.rows
        columns = _base_columns(block)
        columns["posterior_ref"] = flat_ref[begin:stop]
        columns["lambda_return"] = flat_return[begin:stop]
        for forbidden in ("advantage", "v_target", "q_hat", "parent_posterior"):
            columns.pop(forbidden, None)
        training[partition_id] = TrainingPartition(
            partition_id,
            block.inputs,
            columns,
            input_pools=block.input_pools,
        )
    return TrainingStorage(
        "sequential_rb",
        training,
        _sampling_metadata(rollout, branched=False),
        posterior_pool=pool,
    )


@dataclass(frozen=True)
class BranchedFrontierPlan:
    leaf_schedule: Tensor
    peak_slots: int
    leaf_microbatch_rows: int


def plan_branched_frontier(
    topology: BranchedTreeSpan, *, leaf_microbatch_rows: int
) -> BranchedFrontierPlan:
    """Simulate structural ready counters before allocating posterior slots."""

    if leaf_microbatch_rows <= 0:
        raise ValueError("leaf_microbatch_rows must be positive")
    topology.validate()
    # Structural planning has one declared device-to-host boundary.  The
    # subsequent exact simulation reads only CPU scalar metadata and therefore
    # never performs a CUDA synchronization per node or edge.
    structure = torch.stack((topology.parent, topology.degree), 1).detach().to("cpu")
    parent_host = structure[:, 0].to(torch.int64)
    degree_host = structure[:, 1].to(torch.int64)
    leaves_host = (degree_host == 0).nonzero(as_tuple=False).squeeze(1)
    ready = [0] * topology.nodes
    live = 0
    peak = 0
    for first in range(0, leaves_host.numel(), leaf_microbatch_rows):
        batch = [int(value) for value in leaves_host[first : first + leaf_microbatch_rows]]
        live += len(batch)
        peak = max(peak, live)
        for leaf in batch:
            node = leaf
            while int(parent_host[node]) >= 0:
                parent = int(parent_host[node])
                ready[parent] += 1
                if ready[parent] != int(degree_host[parent]):
                    break
                # n child slots contract to one parent slot.
                live -= int(degree_host[parent]) - 1
                node = parent
            peak = max(peak, live)
    if live != 1:
        raise RuntimeError("branched frontier simulation did not contract to one root")
    return BranchedFrontierPlan(
        leaves_host.to(device=topology.parent.device, dtype=torch.int32),
        max(peak, 1),
        leaf_microbatch_rows,
    )


def build_branched_rb_targets(
    rollout: RolloutStorage,
    partitions: PartitionSet,
    completion: CompletionSpec,
    lam: float,
    *,
    leaf_microbatch_rows: int = 32,
    calibration_metrics: dict[str, float] | None = None,
) -> TrainingStorage:
    """Build fixed/dynamic branched targets with a prebounded live frontier."""

    lam = _validate_lambda(lam)
    if rollout.mode != "branched_rb" or not isinstance(rollout.topology, BranchedTreeSpan):
        raise ValueError("branched targets require branched rollout storage")
    topology = rollout.topology
    _validate_completion_boundary(completion, rollout, leaf_microbatch_rows)
    partition_rows = {key: value.decisions.rows for key, value in rollout.partitions.items()}
    action_ptr = {
        key: value.action_ptr for key, value in rollout.partitions.items() if value.action_ptr is not None
    }
    topology.validate(
        partition_rows=partition_rows,
        action_ptr=action_ptr,
        leaf_rows=rollout.leaves.rows,
    )
    widths = {partitions[key].exact_width for key in rollout.partitions}
    if len(widths) != 1:
        raise NotImplementedError("branched v1 requires one canonical exact width per run")
    exact_width = widths.pop()
    first_block = next(iter(rollout.partitions.values()))
    device = next(iter(first_block.decisions.columns.values())).device
    leaf_reach = next(
        (
            rollout.leaves.columns[name]
            for name in ("range_weights", "log_reach", "reach")
            if name in rollout.leaves.columns
        ),
        None,
    )
    if leaf_reach is not None and leaf_reach.ndim == 3:
        n_players = int(leaf_reach.shape[1])
    else:
        owner_rows = [
            _column(block.decisions, "owner")
            for block in rollout.partitions.values()
            if block.decisions.rows
        ]
        n_players = 1 + max(
            (int(rows.max()) for rows in owner_rows if rows.numel()),
            default=-1,
        )
    if n_players <= 0:
        raise ValueError("branched rollout contains no player decisions")
    plan = plan_branched_frontier(topology, leaf_microbatch_rows=leaf_microbatch_rows)
    topology_host = torch.stack(
        (
            topology.parent,
            topology.child_begin,
            topology.degree,
            topology.decision_partition,
            topology.decision_row,
            topology.leaf_row,
        ),
        1,
    ).detach().to("cpu", dtype=torch.int64)
    parent_host = topology_host[:, 0]
    child_begin_host = topology_host[:, 1]
    degree_host = topology_host[:, 2]
    decision_partition_host = topology_host[:, 3]
    decision_row_host = topology_host[:, 4]
    owner_host = {
        key: _column(block.decisions, "owner").detach().to("cpu", dtype=torch.int64)
        for key, block in rollout.partitions.items()
    }
    action_ptr_host = {
        key: block.action_ptr.detach().to("cpu", dtype=torch.int64)
        for key, block in rollout.partitions.items()
        if block.action_ptr is not None
    }
    nominal_decision_host = {
        key: _column(block.decisions, "nominal_child_count")
        .detach()
        .to("cpu", dtype=torch.int64)
        for key, block in rollout.partitions.items()
        if "nominal_child_count" in block.decisions.columns
    }
    mass_slots = torch.empty(
        (plan.peak_slots, n_players, exact_width), dtype=torch.float32, device=device
    )
    moment_slots = torch.empty_like(mass_slots)
    value_vector_slots = torch.empty(
        (plan.peak_slots, n_players), dtype=torch.float32, device=device
    )
    cost_slots = torch.empty(plan.peak_slots, dtype=torch.float32, device=device)
    delta_slots = torch.empty_like(cost_slots)
    node_slot = [-1] * topology.nodes
    ready = [0] * topology.nodes
    free_slots = list(range(plan.peak_slots - 1, -1, -1))
    value_hat = {
        key: torch.empty((value.decisions.rows, exact_width), dtype=torch.float32, device=device)
        for key, value in rollout.partitions.items()
    }
    child_posterior = {
        key: torch.empty((value.actions.rows, exact_width), dtype=torch.float32, device=device)
        for key, value in rollout.partitions.items()
        if value.actions is not None
    }
    advantage = {key: torch.empty_like(value) for key, value in child_posterior.items()}
    decision_valid = {
        key: torch.zeros(value.decisions.rows, dtype=torch.bool, device=device)
        for key, value in rollout.partitions.items()
    }
    action_valid = {
        key: torch.zeros(value.shape[0], dtype=torch.bool, device=device)
        for key, value in child_posterior.items()
    }
    dynamic_heads = any(
        "frozen_trace_variance" in block.decisions.columns
        or "frozen_cost" in block.decisions.columns
        for block in rollout.partitions.values()
    )
    required_dynamic = {
        "frozen_trace_variance",
        "frozen_cost",
        "nominal_child_count",
        "executed_child_count",
    }
    if dynamic_heads and any(
        not required_dynamic.issubset(block.decisions.columns)
        for block in rollout.partitions.values()
    ):
        raise ValueError("dynamic branched rollout has incomplete controller columns")
    trace_target = {
        key: torch.zeros(value.decisions.rows, dtype=torch.float32, device=device)
        for key, value in rollout.partitions.items()
    }
    trace_mask = {
        key: torch.zeros(value.decisions.rows, dtype=torch.bool, device=device)
        for key, value in rollout.partitions.items()
    }
    cost_target = {
        key: torch.zeros(value.decisions.rows, dtype=torch.float32, device=device)
        for key, value in rollout.partitions.items()
    }

    def allocate_slot() -> int:
        if not free_slots:
            raise RuntimeError("precomputed branched frontier bound was underestimated")
        return free_slots.pop()

    def reduce_ready(start_node: int) -> None:
        node = start_node
        while int(parent_host[node]) >= 0:
            parent = int(parent_host[node])
            ready[parent] += 1
            if ready[parent] != int(degree_host[parent]):
                return
            begin = int(child_begin_host[parent])
            count = int(degree_host[parent])
            children = torch.arange(begin, begin + count, device=device)
            slot_ids = node_slot[begin : begin + count]
            if any(slot < 0 for slot in slot_ids):
                raise RuntimeError("ready parent has a missing child frontier slot")
            slots = torch.tensor(slot_ids, dtype=torch.int64, device=device)
            target_count = topology.protected_root_degree if parent == 0 else count
            target_slots = slots[:target_count]
            target_children = children[:target_count]
            summed_mass = mass_slots[target_slots].sum(0)
            summed_moment = moment_slots[target_slots].sum(0)
            parent_mass = summed_mass / target_count
            parent_moment = summed_moment / target_count
            parent_value_vector = value_vector_slots[target_slots].mean(0)
            nominal_degree = (
                topology.degree
                if topology.nominal_degree is None
                else topology.nominal_degree
            )
            child_nominal = nominal_degree[target_children].to(torch.float32)
            parent_cost = (
                1.0 + child_nominal * cost_slots[target_slots]
            ).mean()
            parent_delta = delta_slots[target_slots].mean()
            partition_id = int(decision_partition_host[parent])
            if partition_id >= 0:
                row = int(decision_row_host[parent])
                block = rollout.partitions[partition_id]
                spec = partitions[partition_id]
                owner = int(owner_host[partition_id][row])
                old_value = _column(block.decisions, "old_value", "oldV")[row].float()
                spec.assert_model_width(old_value)
                exact_old = _phi_exact(old_value, spec.exact_to_model)
                value_hat[partition_id][row] = conditionalize(
                    summed_moment[owner], summed_mass[owner]
                )
                assert block.action_ptr is not None
                action_begin = int(action_ptr_host[partition_id][row])
                for ordinal, slot in enumerate(target_slots):
                    child_mass = mass_slots[slot, owner]
                    child_moment = moment_slots[slot, owner]
                    continuation = conditionalize(child_moment, child_mass)
                    if target_count == 1:
                        baseline = exact_old
                    else:
                        sibling_mass = summed_mass[owner] - child_mass
                        sibling_moment = summed_moment[owner] - child_moment
                        baseline = lam * conditionalize(
                            sibling_moment, sibling_mass
                        ) + (1.0 - lam) * exact_old
                    action_row = action_begin + ordinal
                    child_posterior[partition_id][action_row] = child_mass
                    advantage[partition_id][action_row] = continuation - baseline
                    action_valid[partition_id][action_row] = True
                if dynamic_heads:
                    frozen_t = _column(
                        block.decisions, "frozen_trace_variance"
                    )[row].float()
                    frozen_c = _column(block.decisions, "frozen_cost")[row].float()
                    nominal = float(nominal_decision_host[partition_id][row])
                    if target_count >= 2:
                        centered = (
                            value_vector_slots[target_slots] - parent_value_vector
                        )
                        executed_trace = centered.square().sum(1).sum() / (
                            target_count - 1
                        )
                        corrected = (executed_trace - parent_delta).clamp_min(0.0)
                        trace_target[partition_id][row] = corrected.detach()
                        trace_mask[partition_id][row] = True
                        t_star = corrected
                    else:
                        t_star = frozen_t
                    cost_target[partition_id][row] = parent_cost.detach()
                    parent_delta = (
                        (t_star + parent_delta) / float(target_count)
                        - t_star / nominal
                    )
                    parent_cost = lam * parent_cost + (1.0 - lam) * frozen_c
                parent_moment[owner] = (
                    lam * parent_moment[owner]
                    + (1.0 - lam) * parent_mass[owner] * exact_old
                )
                decision_valid[partition_id][row] = True
            elif dynamic_heads:
                # Keep the current chance-node counterfactual correction.
                parent_delta = parent_delta / float(target_count)
            destination = slot_ids[0]
            mass_slots[destination] = parent_mass
            moment_slots[destination] = parent_moment
            value_vector_slots[destination] = parent_value_vector
            cost_slots[destination] = parent_cost
            delta_slots[destination] = parent_delta
            node_slot[parent] = destination
            for slot in slot_ids[1:]:
                free_slots.append(slot)
            for child in range(begin, begin + count):
                node_slot[child] = -1
            node_slot[parent] = destination
            node = parent

    for first in range(0, plan.leaf_schedule.numel(), leaf_microbatch_rows):
        leaf_nodes = plan.leaf_schedule[first : first + leaf_microbatch_rows]
        leaf_rows = topology.leaf_row[leaf_nodes.to(torch.int64)].to(torch.int64)
        completed = completion.complete(rollout.leaves.columns, rows=leaf_rows)
        posterior = _completion_field(completed, "p_hat").float()
        value = _completion_field(completed, "v_hat").float()
        if posterior.shape != value.shape or posterior.shape != (
            leaf_nodes.numel(), n_players, exact_width
        ):
            raise ValueError("branched completion outputs violate [B,P,X]")
        # Reserve the whole declared microbatch before reductions, matching the
        # frontier-plan simulation and completion workspace lifetime.
        slots = [allocate_slot() for _ in range(leaf_nodes.numel())]
        leaf_host = leaf_nodes.detach().to("cpu", dtype=torch.int64).tolist()
        for local, (leaf, slot) in enumerate(zip(leaf_host, slots, strict=True)):
            mass_slots[slot] = posterior[local]
            moment_slots[slot] = posterior[local] * value[local]
            value_vector_slots[slot] = (
                posterior[local] * value[local]
            ).sum(-1)
            cost_slots[slot] = 0.0
            delta_slots[slot] = 0.0
            node_slot[leaf] = slot
        for leaf in leaf_host:
            reduce_ready(int(leaf))
        del completed, posterior, value
    if node_slot[0] < 0 or len(free_slots) != plan.peak_slots - 1:
        raise RuntimeError("branched frontier did not finish with only the root slot")
    if dynamic_heads and calibration_metrics is not None:
        calibration_partitions = [
            (partition_id, block)
            for partition_id, block in rollout.partitions.items()
            if block.decisions.rows
        ]
        if not calibration_partitions:
            raise RuntimeError("cost calibration requires at least one decision")
        calibration_rows = sum(
            block.decisions.rows for _, block in calibration_partitions
        )
        underprediction = torch.empty(
            calibration_rows, dtype=torch.float32, device=device
        )
        deviance_sum = torch.zeros((), dtype=torch.float32, device=device)
        cursor = 0
        for partition_id, block in calibration_partitions:
            end = cursor + block.decisions.rows
            prediction = _column(
                block.decisions, "frozen_cost"
            ).detach().float()
            target = cost_target[partition_id].detach().float()
            underprediction[cursor:end].copy_(
                target / prediction.clamp_min(torch.finfo(prediction.dtype).tiny)
            )
            deviance_sum.add_(gamma_deviance(prediction, target).sum())
            cursor = end
        calibration_metrics.update(
            {
                "cost_calibration_rows": float(calibration_rows),
                "cost_calibration_gamma_deviance_mean": float(
                    deviance_sum / calibration_rows
                ),
                "cost_calibration_underprediction_p95": float(
                    torch.quantile(underprediction, 0.95)
                ),
            }
        )
    training: dict[int, TrainingPartition] = {}
    for partition_id, block in rollout.partitions.items():
        if not bool(decision_valid[partition_id].all()) or not bool(
            action_valid[partition_id].all()
        ):
            raise RuntimeError("branched target builder missed a decision or action")
        columns = _base_columns(block)
        columns["value_hat"] = value_hat[partition_id]
        if dynamic_heads:
            columns["trace_variance_hat"] = trace_target[partition_id]
            columns["trace_variance_mask"] = trace_mask[partition_id]
            columns["cost_hat"] = cost_target[partition_id]
        for forbidden in (
            "q_hat",
            "parent_posterior",
            "W",
            "S",
            "frozen_trace_variance",
            "frozen_cost",
        ):
            columns.pop(forbidden, None)
        assert block.actions is not None and block.action_ptr is not None
        action_columns = {
            name: value.detach() for name, value in block.actions.columns.items()
        }
        action_columns["child_posterior"] = child_posterior[partition_id]
        action_columns["advantage_hat"] = advantage[partition_id]
        action_columns["action_ptr"] = block.action_ptr
        pointer = action_ptr_host[partition_id]
        max_actions_per_decision = (
            int((pointer[1:] - pointer[:-1]).max()) if block.decisions.rows else 0
        )
        training[partition_id] = TrainingPartition(
            partition_id,
            block.inputs,
            columns,
            action_columns,
            max_actions_per_decision,
            input_pools=block.input_pools,
        )
    return TrainingStorage(
        "branched_rb", training, _sampling_metadata(rollout, branched=True)
    )


def finalize_training_storage(storage: TrainingStorage) -> TrainingStorage:
    """Validate that only immutable training-owned fields survived targets."""

    forbidden = {
        "reach",
        "log_r",
        "state_ref",
        "parent",
        "child_begin",
        "leaf_row",
        "valid",
        "q_hat",
        "W",
        "S",
    }
    for partition in storage.partitions.values():
        overlap = forbidden.intersection(partition.columns)
        if overlap:
            raise ValueError(f"training storage retained transient fields {sorted(overlap)}")
        if partition.action_columns is not None:
            overlap = forbidden.intersection(partition.action_columns)
            if overlap:
                raise ValueError(
                    f"training action storage retained transient fields {sorted(overlap)}"
                )
    return storage
