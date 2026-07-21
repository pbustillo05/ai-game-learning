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

from pbg.core.partition import CompletionSpec, PartitionSet
from pbg.core.storage import (
    RolloutStorage,
    SamplingMetadata,
    SequentialPosteriorPool,
    TrainingPartition,
    TrainingStorage,
)
from pbg.core.topology import BranchedTreeSpan, SequentialDecisionCSR
from pbg.core.values import conditionalize


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
    return old_value[index]


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
    returns = {
        partition_id: torch.empty(
            block.decisions.rows, dtype=torch.float32, device=reward.device
        )
        for partition_id, block in rollout.partitions.items()
    }
    valid = {
        partition_id: torch.zeros(block.decisions.rows, dtype=torch.bool, device=reward.device)
        for partition_id, block in rollout.partitions.items()
    }
    for first in range(0, topology.roots, trajectory_batch_rows):
        for trajectory in range(first, min(first + trajectory_batch_rows, topology.roots)):
            carry = reward[int(topology.leaf_row[trajectory])].clone()
            begin = int(topology.trajectory_ptr[trajectory])
            stop = int(topology.trajectory_ptr[trajectory + 1])
            for ordinal in range(stop - 1, begin - 1, -1):
                partition_id = int(topology.decision_partition[ordinal])
                row = int(topology.decision_row[ordinal])
                block = rollout.partitions[partition_id]
                owner = int(_column(block.decisions, "owner")[row])
                old_value = float(_column(block.decisions, "old_value", "oldV")[row])
                returns[partition_id][row] = carry[owner]
                valid[partition_id][row] = True
                carry[owner] = lam * carry[owner] + (1.0 - lam) * old_value
    training: dict[int, TrainingPartition] = {}
    for partition_id, block in rollout.partitions.items():
        if not bool(valid[partition_id].all()):
            raise RuntimeError("normal target builder did not write every decision once")
        columns = _base_columns(block)
        columns["lambda_return"] = returns[partition_id]
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
    returns = {
        key: torch.empty((value.decisions.rows, exact_width), dtype=torch.float32, device=device)
        for key, value in rollout.partitions.items()
    }
    refs = {
        key: torch.full((value.decisions.rows,), -1, dtype=torch.int32, device=device)
        for key, value in rollout.partitions.items()
    }
    valid = {
        key: torch.zeros(value.decisions.rows, dtype=torch.bool, device=device)
        for key, value in rollout.partitions.items()
    }
    pool_trajectory: list[int] = []
    pool_player: list[int] = []
    pair_ref: dict[tuple[int, int], int] = {}

    # Topology and owner columns are already sealed, so determine the exact
    # posterior-pool extent before completion allocates any exact-hand rows.
    # Python retains only scalar address metadata, never tensor payload.
    for trajectory in range(topology.roots):
        begin = int(topology.trajectory_ptr[trajectory])
        stop = int(topology.trajectory_ptr[trajectory + 1])
        for ordinal in range(begin, stop):
            partition_id = int(topology.decision_partition[ordinal])
            row = int(topology.decision_row[ordinal])
            owner = int(_column(rollout.partitions[partition_id].decisions, "owner")[row])
            pair = (trajectory, owner)
            if pair not in pair_ref:
                pair_ref[pair] = len(pool_trajectory)
                pool_trajectory.append(trajectory)
                pool_player.append(owner)
    posterior_tensor = torch.empty(
        (len(pair_ref), exact_width), dtype=torch.float32, device=device
    )
    pool_valid = torch.zeros(len(pair_ref), dtype=torch.bool, device=device)

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
        for local, trajectory_tensor in enumerate(trajectories):
            trajectory = int(trajectory_tensor)
            begin = int(topology.trajectory_ptr[trajectory])
            stop = int(topology.trajectory_ptr[trajectory + 1])
            required_players = []
            for ordinal in range(begin, stop):
                partition_id = int(topology.decision_partition[ordinal])
                row = int(topology.decision_row[ordinal])
                owner = int(_column(rollout.partitions[partition_id].decisions, "owner")[row])
                if owner not in required_players:
                    required_players.append(owner)
            moment = posterior[local] * value[local]
            for owner in required_players:
                pool_row = pair_ref[(trajectory, owner)]
                if bool(pool_valid[pool_row]):
                    raise RuntimeError("trajectory/player posterior was written more than once")
                posterior_tensor[pool_row] = posterior[local, owner]
                pool_valid[pool_row] = True
            for ordinal in range(stop - 1, begin - 1, -1):
                partition_id = int(topology.decision_partition[ordinal])
                row = int(topology.decision_row[ordinal])
                spec = partitions[partition_id]
                block = rollout.partitions[partition_id]
                owner = int(_column(block.decisions, "owner")[row])
                mass = posterior[local, owner]
                returns[partition_id][row] = conditionalize(moment[owner], mass)
                refs[partition_id][row] = pair_ref[(trajectory, owner)]
                valid[partition_id][row] = True
                old_value = _column(block.decisions, "old_value", "oldV")[row].float()
                spec.assert_model_width(old_value)
                exact_old = _phi_exact(old_value, spec.exact_to_model)
                moment[owner] = lam * moment[owner] + (1.0 - lam) * mass * exact_old
        del completed, posterior, value
    if not bool(pool_valid.all()):
        raise RuntimeError("sequential posterior pool has unwritten rows")
    pool = SequentialPosteriorPool(
        posterior_tensor,
        torch.tensor(pool_trajectory, dtype=torch.int32, device=device),
        torch.tensor(pool_player, dtype=torch.int32, device=device),
    )
    training: dict[int, TrainingPartition] = {}
    for partition_id, block in rollout.partitions.items():
        if not bool(valid[partition_id].all()):
            raise RuntimeError("sequential RB target builder missed a decision")
        columns = _base_columns(block)
        columns["posterior_ref"] = refs[partition_id]
        columns["lambda_return"] = returns[partition_id]
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
            for name in ("log_reach", "reach")
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
        n_players = 0 if not owner_rows else int(torch.cat(owner_rows).max()) + 1
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
        training[partition_id] = TrainingPartition(
            partition_id,
            block.inputs,
            columns,
            action_columns,
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


# Algorithm-style public aliases used by design documents and callers.
BuildNormalTargets = build_normal_targets
BuildSequentialRBTargets = build_sequential_rb_targets
BuildBranchedRBTargets = build_branched_rb_targets
FinalizeTrainingStorage = finalize_training_storage
