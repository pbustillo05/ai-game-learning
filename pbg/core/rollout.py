"""Device-resident sequential rollout and unified tree-sampling entry point."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, PROBE_FOLD, TERMINAL, EventBatch, GameBundle
from pbg.core.execution import (
    DONE,
    ChunkedOccurrenceArena,
    DeviceWorkQueue,
    DrawPurpose,
    TrajectoryAdmissionPlan,
    arena_fields,
    child_path_keys,
    logical_rng_keys,
    stable_partition_masked,
)
from pbg.core.tree import DecisionReachCache, DeviceTreeBuffer, PolicyCache, ReachCache, TreeBatch


@dataclass
class ProbeRollout:
    """Sequential mixed-policy data retained only for the exploiter."""

    tree: TreeBatch
    leaf_reach: ReachCache
    policy: PolicyCache
    decision_reach: DecisionReachCache
    seats: Tensor
    root_profiles: Tensor


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



def _family_spec(
    backend: Any,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Return stable execution-family names and their device routing values."""

    names = tuple(getattr(backend, "execution_families", ("default",)))
    values = tuple(getattr(backend, "execution_family_values", range(len(names))))
    priorities = tuple(getattr(backend, "execution_family_priorities", (0,) * len(names)))
    if (
        not names
        or len(values) != len(names)
        or len(priorities) != len(names)
        or len(set(values)) != len(values)
    ):
        raise ValueError("rollout execution families require unique names and values")
    return names, values, priorities


def _family_ids(backend: Any, states: Tensor) -> Tensor:
    route = getattr(backend, "rollout_family_ids", None)
    if callable(route):
        result = route(states)
    else:
        result = torch.zeros(states.shape, dtype=torch.int64, device=states.device)
    if result.shape != states.shape or result.device != states.device:
        raise ValueError("rollout_family_ids must return device-local [B]")
    return result.to(torch.int64)


def _logical_seed(cfg: Any, gen: torch.Generator) -> int:
    rollout_cfg = getattr(cfg, "rollout", cfg)
    training_cfg = getattr(cfg, "training", None)
    configured = getattr(rollout_cfg, "logical_seed", None)
    if configured is None and training_cfg is not None:
        configured = getattr(training_cfg, "seed", None)
    return int(gen.initial_seed() if configured is None else configured)


def sequential_leaf_logical_rng_keys(
    tree: TreeBatch,
    cfg: Any,
    gen: torch.Generator,
    leaf_ids: Tensor,
    *,
    iteration: int,
    purpose: DrawPurpose,
    draw: int = 0,
) -> Tensor:
    """Address one stochastic operation at each sequential stopped leaf.

    Sequential records contain exactly one chain and one stopped leaf per
    global trajectory.  Their abstract path is therefore determined by the
    chain's edge count.  Building the small depth-to-path table on CPU avoids
    a device scalar read or one CUDA launch per stored node while producing the
    same recursive path address used by the family-scheduled scheduler.
    """

    rollout_cfg = getattr(cfg, "rollout", cfg)
    start_count = int(rollout_cfg.roots_per_gpu)
    node_budget = int(
        getattr(rollout_cfg, "node_budget", max(int(tree.parent.numel()), 1))
    )
    leaves = leaf_ids.to(device=tree.parent.device, dtype=torch.int64)
    if leaves.ndim != 1:
        raise ValueError("sequential leaf ids must be one-dimensional")
    if start_count <= 0 or node_budget <= 0:
        raise ValueError("sequential rollout dimensions are invalid")
    if leaves.numel() != start_count:
        raise ValueError("sequential rollout must contain one stopped leaf per trajectory")

    tree_ids = tree.tree_id.to(torch.int64)
    counts = torch.bincount(tree_ids, minlength=start_count)
    selected_tree = tree_ids[leaves]
    valid = (
        (selected_tree >= 0)
        & (selected_tree < start_count)
        & (counts[selected_tree.clamp(0, max(start_count - 1, 0))] <= node_budget)
    )
    unique = torch.bincount(selected_tree.clamp_min(0), minlength=start_count)[:start_count] == 1
    if tree.parent.device.type == "cuda":
        torch._assert_async(valid.all(), "invalid sequential stopped-leaf address")
        torch._assert_async(unique.all(), "sequential trajectories require one stopped leaf")
    elif not bool(valid.all()) or not bool(unique.all()):
        raise ValueError("invalid sequential stopped-leaf topology")

    depth_paths = torch.empty(node_budget, dtype=torch.int64)
    depth_paths[0] = 0
    for depth in range(1, node_budget):
        depth_paths[depth] = child_path_keys(depth_paths[depth - 1 : depth], 1)[0]
    leaf_depth = counts[selected_tree] - 1
    path = depth_paths.to(tree.parent.device)[leaf_depth]
    start = _global_trajectory_start(cfg, start_count)
    return logical_rng_keys(
        seed=_logical_seed(cfg, gen),
        iteration=iteration,
        trajectory=selected_tree + start,
        path=path,
        draw=draw,
        purpose=purpose,
    )


def _global_trajectory_start(cfg: Any, start_count: int) -> int:
    rollout_cfg = getattr(cfg, "rollout", cfg)
    configured = getattr(rollout_cfg, "trajectory_start_index", None)
    if configured is not None:
        return int(configured)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    rank_counts = getattr(rollout_cfg, "trajectory_rank_counts", None)
    if rank_counts is not None:
        counts = tuple(int(value) for value in rank_counts)
        if rank >= len(counts):
            raise ValueError(
                "trajectory_rank_counts has no quota for the current distributed rank"
            )
        if counts[rank] != start_count:
            raise ValueError(
                "local rollout.start_count does not match trajectory_rank_counts for "
                f"rank {rank}: local={start_count}, quota={counts[rank]}"
            )
        return sum(counts[:rank])
    return rank * start_count


def _admission_plan(
    *,
    backend: Any,
    rollout_cfg: Any,
    buffer: DeviceTreeBuffer,
    arena: ChunkedOccurrenceArena,
    queues: dict[str, DeviceWorkQueue],
    root_reach: Tensor,
    root_profiles: Tensor,
    working_reach: Tensor,
    batch_rows: int,
    desired_resident_starts: int,
) -> TrajectoryAdmissionPlan:
    """Build the pre-sampling retained plus maximum-in-flight byte transaction."""

    calibrated = getattr(backend, "admission_byte_terms", None)
    if callable(calibrated):
        retained, resident_per_start, max_inflight = calibrated(
            rollout_cfg, batch_rows
        )
        retained = int(retained)
        resident_per_start = int(resident_per_start)
        max_inflight = int(max_inflight)
    else:
        seen: set[int] = set()

        def unique_tensor_bytes(value: Tensor) -> int:
            storage = value.untyped_storage()
            identity = storage.data_ptr()
            if identity in seen:
                return 0
            seen.add(identity)
            return int(storage.nbytes())

        retained = sum(
            unique_tensor_bytes(value)
            for value in vars(buffer).values()
            if isinstance(value, Tensor)
        )
        # The occurrence arena is chunked, but SafeAdmit reserves its complete
        # retained structural bound rather than relying on future allocation.
        retained += arena.capacity * arena.row_bytes
        retained += sum(queue.reserved_bytes for queue in queues.values())
        retained += sum(
            unique_tensor_bytes(value)
            for value in (root_reach, root_profiles, working_reach)
        )
        p_max = int(backend.n_players)
        x_max = max(int(value) for value in backend.x_sizes)
        # One selected old log-probability row and one old value row per
        # possible player occurrence are retained by the current cache contract.
        retained += buffer.node_capacity * 2 * x_max * 4
        reach_row = p_max * x_max * 4
        profile_row = p_max * 4
        resident_per_start = reach_row + profile_row + 64
        # Current/next reaches, curried old outputs, profile/event metadata, and
        # expansion scratch form the generic core-owned in-flight transaction.
        # A game backend may replace this conservative term with calibration.
        inflight_per_row = 2 * reach_row + 2 * x_max * 4 + profile_row + 128
        max_inflight = batch_rows * inflight_per_row

    configured_limit = getattr(rollout_cfg, "execution_memory_bytes", None)
    if configured_limit is None:
        resident_capacity = max(1, int(desired_resident_starts))
        configured_limit = (
            retained + max_inflight + resident_capacity * resident_per_start
        )
    plan = TrajectoryAdmissionPlan(
        limit_bytes=int(configured_limit),
        retained_bytes=retained,
        max_inflight_bytes=max_inflight,
        resident_bytes_per_start=resident_per_start,
    )
    plan.require_progress_capacity(int(rollout_cfg.roots_per_gpu))
    return plan



def _sample_profiles_logical(
    backend: Any,
    states: Tensor,
    reach: Tensor,
    keys: Tensor,
    gen: torch.Generator,
) -> Tensor:
    keyed = getattr(backend, "sample_profiles_keyed", None)
    if callable(keyed):
        return keyed(states, reach, keys)
    return backend.sample_profiles(states, reach, gen)


def _sample_events_logical(
    backend: Any,
    states: Tensor,
    profiles: Tensor,
    output: Any | None,
    adapter: Any,
    keys: Tensor,
    gen: torch.Generator,
) -> EventBatch:
    keyed = getattr(backend, "sample_events_keyed", None)
    if callable(keyed):
        return keyed(states, profiles, output, adapter, keys)
    return backend.sample_events(states, profiles, output, adapter, gen)


def _sample_trees_sequential(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    return_profiles: bool = False,
    iteration: int = 0,
) -> tuple[TreeBatch, ReachCache, PolicyCache] | tuple[
    TreeBatch, ReachCache, PolicyCache, Tensor
] | ProbeRollout:
    """Sample sequential paths with persistent family work queue ready queues.

    The model scheduler may change batch boundaries without changing private,
    action, or chance draws: every stochastic boundary is addressed by the
    global trajectory id, abstract path, iteration, and draw purpose.
    """

    backend = bundle.rollout_backend
    rollout_cfg = getattr(cfg, "rollout", cfg)
    device = torch.device(backend.device)
    if torch.device(gen.device) != device:
        raise ValueError("rollout generator must be created on the backend device")
    start_count = int(rollout_cfg.roots_per_gpu)
    node_budget = int(rollout_cfg.node_budget)
    if start_count < 0 or node_budget <= 0:
        raise ValueError("start_count must be nonnegative and node_budget positive")
    batch_rows = int(
        getattr(rollout_cfg, "family_batch_rows", None) or min(start_count, 256)
    )
    admission_size = int(
        getattr(rollout_cfg, "max_active_trajectories", None) or batch_rows
    )
    if batch_rows <= 0 or admission_size <= 0:
        raise ValueError("family-scheduled batch and admission sizes must be positive")
    chunk_nodes = int(
        getattr(rollout_cfg, "arena_chunk_nodes", None)
        or min(start_count * node_budget, max(batch_rows, 256))
    )

    p_max = int(backend.n_players)
    x_sizes = tuple(int(value) for value in backend.x_sizes)
    x_max = max(x_sizes)
    roots = backend.sequential_start_states(start_count)
    if roots.shape != (start_count,) or roots.dtype != torch.int32 or roots.device != device:
        raise ValueError("sequential_start_states must return device-local int32 [count]")
    root_owner = backend.classify(roots).to(torch.int32)
    metrics = getattr(backend, "execution_metrics", None)
    buffer = DeviceTreeBuffer(start_count, node_budget, p_max, device)
    root_nodes = buffer.add_roots(roots, root_owner)
    arena = ChunkedOccurrenceArena(
        buffer.node_capacity,
        chunk_nodes,
        device,
        fields=arena_fields(("trajectory_id", "path_key")),
        label="sequential_occurrences",
    )
    root_handles = arena.allocate(start_count)
    expected_roots = root_nodes.to(torch.int64)
    if device.type == "cuda":
        torch._assert_async(
            (root_handles == expected_roots).all(),
            "family-scheduled occurrence and tree root handles diverged",
        )
    elif not torch.equal(root_handles, expected_roots):
        raise RuntimeError("family-scheduled occurrence and tree root handles diverged")

    trajectory_start = _global_trajectory_start(cfg, start_count)
    trajectory_ids = torch.arange(
        trajectory_start,
        trajectory_start + start_count,
        dtype=torch.int64,
        device=device,
    )
    probe_enabled = bundle.opponent_model is not None
    probe_seats = None
    if probe_enabled:
        configured_seats = getattr(rollout_cfg, "probe_seats", None)
        if configured_seats is not None:
            probe_seats = torch.as_tensor(
                configured_seats, dtype=torch.int32, device=device
            )
            if probe_seats.shape != (start_count,) or bool(
                ((probe_seats < 0) | (probe_seats >= p_max)).any()
            ):
                raise ValueError("rollout.probe_seats must select one valid seat per start")
        else:
            seat_keys = logical_rng_keys(
                seed=_logical_seed(cfg, gen),
                iteration=iteration,
                trajectory=trajectory_ids,
                path=torch.zeros_like(trajectory_ids),
                draw=0,
                purpose=DrawPurpose.PROBE_SEAT,
            )
            probe_seats = torch.remainder(seat_keys, p_max).to(torch.int32)
    arena.scatter_many(
        root_handles,
        {
            "trajectory_id": trajectory_ids,
            "path_key": torch.zeros(start_count, dtype=torch.int64, device=device),
        },
    )

    root_reach = torch.zeros(
        (start_count, p_max, x_max), dtype=torch.float32, device=device
    )
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    # A sequential trajectory has exactly one live occurrence.  Its active
    # reach row is overwritten as the path advances; immutable natural decision
    # input and stopped-leaf reach rows are appended at their last consumer.
    working_reach_rows = start_count
    working_reach = torch.empty(
        (working_reach_rows, p_max, x_max), dtype=torch.float32, device=device
    )
    working_reach[:start_count] = root_reach
    root_profiles = torch.empty(
        (start_count, p_max), dtype=torch.int32, device=device
    )

    family_names, family_values, family_priorities = _family_spec(backend)
    queues = {
        name: DeviceWorkQueue(start_count, device) for name in family_names
    }
    actual_cost_forward = getattr(
        backend, "policy_forward_family_actual_cost", None
    )
    if not callable(actual_cost_forward):
        raise ValueError("production requires an actual-cost family forward")
    admission_plan = _admission_plan(
        backend=backend,
        rollout_cfg=rollout_cfg,
        buffer=buffer,
        arena=arena,
        queues=queues,
        root_reach=root_reach,
        root_profiles=root_profiles,
        working_reach=working_reach,
        batch_rows=batch_rows,
        desired_resident_starts=min(start_count, admission_size),
    )
    logical_seed = _logical_seed(cfg, gen)
    policy_nodes: list[Tensor] = []
    policy_logp: list[Tensor] = []
    policy_values: list[Tensor] = []
    decision_reach_nodes: list[Tensor] = []
    decision_reach_rows: list[Tensor] = []
    leaf_reach_nodes: list[Tensor] = []
    leaf_reach_rows: list[Tensor] = []
    scheduler_steps = 0
    active_starts = 0
    if metrics is not None:
        metrics.maximum("reserved_bytes_max", admission_plan.base_bytes)

    def reach_rows_for(nodes: Tensor) -> Tensor:
        rows = buffer.tree_id[nodes.to(torch.int64)].to(torch.int64)
        return working_reach[rows]

    def read_control_counts(counts: Tensor) -> tuple[int, ...]:
        """Read one fixed-size scheduler control block at a declared boundary."""

        if counts.ndim != 1 or counts.dtype != torch.int64:
            raise ValueError("scheduler control counts must be int64 [K]")
        if device.type == "cuda":
            started = time.perf_counter()
            host = counts.to(device="cpu")
            elapsed = time.perf_counter() - started
            if metrics is not None:
                metrics.record_scheduler_control_synchronization(
                    seconds=elapsed,
                    nbytes=counts.numel() * counts.element_size(),
                )
        else:
            host = counts
        return tuple(int(value) for value in host.tolist())

    def planned_family_pack(name: str) -> tuple[int, float, bool]:
        """Return logical rows, batch fill, and full-threshold status."""

        queue = queues[name]
        if queue.host_count == 0:
            return 0, 0.0, False
        selected = min(queue.host_count, batch_rows)
        fill = selected / float(batch_rows)
        full_threshold = selected == batch_rows
        return selected, fill, full_threshold

    def record_queues() -> None:
        if metrics is None:
            return
        for name, queue in queues.items():
            metrics.record_queue(
                name,
                depth=queue.host_count,
                oldest_age=float(queue.oldest_age(scheduler_steps)),
            )

    def append_children(
        parents: Tensor,
        next_states: Tensor,
        next_owner: Tensor,
        events: EventBatch,
        next_reach: Tensor,
    ) -> Tensor:
        count = int(parents.numel())
        buffer.require_child_capacity(count)
        child_handles = arena.allocate(count)
        children = buffer.append_children(
            parents.to(torch.int32),
            next_states,
            next_owner,
            events.public_id,
            torch.ones(count, dtype=torch.float32, device=device),
        )
        if device.type == "cuda":
            torch._assert_async(
                (child_handles == children.to(torch.int64)).all(),
                "family-scheduled occurrence and tree child handles diverged",
            )
        elif not torch.equal(child_handles, children.to(torch.int64)):
            raise RuntimeError("family-scheduled occurrence and tree child handles diverged")
        parent_rows = parents.to(torch.int64)
        parent_address = arena.gather_many(
            ("trajectory_id", "path_key"), parent_rows
        )
        arena.scatter_many(
            child_handles,
            {
                "trajectory_id": parent_address["trajectory_id"],
                "path_key": child_path_keys(parent_address["path_key"], 1),
            },
        )
        reach_rows = buffer.tree_id[children.to(torch.int64)].to(torch.int64)
        working_reach[reach_rows] = next_reach
        return children

    def chance_close(nodes: Tensor) -> None:
        nonlocal active_starts
        frontier = nodes.to(torch.int32)
        chance_steps = 0
        while frontier.numel():
            rows = frontier.to(torch.int64)
            owners = buffer.owner[rows]
            states = buffer.state_ref[rows]
            family_ids = _family_ids(backend, states)
            chance_route = -1
            leaf_route = -2
            invalid_route = -3
            player = (owners >= 0) & (owners < p_max)
            route = torch.where(
                owners == CHANCE,
                torch.full_like(family_ids, chance_route),
                torch.where(
                    (owners == TERMINAL) | (owners == PRETERM) | (owners == PROBE_FOLD),
                    torch.full_like(family_ids, leaf_route),
                    torch.where(
                        player,
                        family_ids,
                        torch.full_like(family_ids, invalid_route),
                    ),
                ),
            )
            routed = stable_partition_masked(
                frontier.to(torch.int64),
                route,
                (*family_values, chance_route, leaf_route),
            )
            counts = read_control_counts(routed.selected_count)
            family_counts = counts[: len(family_names)]
            chance_count, stopped = counts[-2:]
            if stopped:
                stopped_nodes = routed.candidate_handles[-1, :stopped].to(torch.int32)
                leaf_reach_nodes.append(stopped_nodes)
                leaf_reach_rows.append(reach_rows_for(stopped_nodes).detach().clone())
            if stopped > active_starts:
                raise RuntimeError("family-scheduled active-trajectory accounting underflow")
            active_starts -= stopped
            if metrics is not None:
                metrics.maximum(
                    "reserved_bytes_max",
                    admission_plan.committed_bytes(active_starts),
                )
            for index, (name, count) in enumerate(
                zip(family_names, family_counts, strict=True)
            ):
                if count:
                    queues[name].enqueue(
                        routed.candidate_handles[index, :count],
                        arena=arena,
                        logical_time=scheduler_steps,
                    )
            record_queues()
            if chance_count == 0:
                break
            chance_steps += 1
            if chance_steps > int(backend.max_rollout_steps):
                raise RuntimeError("chance-only segment exceeded max_rollout_steps")
            chance_nodes = routed.candidate_handles[-2, :chance_count].to(torch.int32)
            chance_rows = chance_nodes.to(torch.int64)
            states = buffer.state_ref[chance_rows]
            reaches = reach_rows_for(chance_nodes)
            tree_ids = buffer.tree_id[chance_rows].to(torch.int64)
            profiles = root_profiles[tree_ids]
            address = arena.gather_many(
                ("trajectory_id", "path_key"), chance_rows
            )
            keys = logical_rng_keys(
                seed=logical_seed,
                iteration=iteration,
                trajectory=address["trajectory_id"],
                path=address["path_key"],
                draw=1,
                purpose=DrawPurpose.CHANCE,
            )
            events = _sample_events_logical(
                backend, states, profiles, None, bundle.policy_adapter, keys, gen
            )
            step_chance = getattr(backend, "step_chance_batch", backend.step)
            next_states = step_chance(states, events)
            next_reach = backend.reach_update(reaches, states, events, None)
            next_owner = backend.classify(next_states).to(torch.int32)
            frontier = append_children(
                chance_nodes, next_states, next_owner, events, next_reach
            )

    admitted = 0
    while admitted < start_count or any(queue.host_count for queue in queues.values()):
        scheduler_steps += 1
        if scheduler_steps > start_count * (int(backend.max_rollout_steps) + 2):
            raise RuntimeError("family-scheduled rollout scheduler exceeded its progress bound")

        family_plans = {
            name: planned_family_pack(name)
            for name in family_names
            if queues[name].host_count
        }
        full = [name for name, plan in family_plans.items() if plan[2]]
        selected_family: str | None = None
        if full:
            # family_names is a topological priority order; within the first
            # priority layer the fullest work queue wins with stable ties.
            upstream = min(
                family_priorities[family_names.index(name)] for name in full
            )
            eligible = [
                name
                for name in full
                if family_priorities[family_names.index(name)] == upstream
            ]
            selected_family = max(
                eligible,
                key=lambda name: (
                    queues[name].host_count / max(family_plans[name][0], 1),
                    queues[name].oldest_age(scheduler_steps),
                    -family_names.index(name),
                ),
            )
        elif admitted < start_count and (
            safe_count := admission_plan.safe_admit(
                active_starts=active_starts,
                unstarted=start_count - admitted,
                requested=admission_size,
            )
        ) > 0:
            stop = admitted + safe_count
            roots_batch = root_nodes[admitted:stop]
            root_rows = roots_batch.to(torch.int64)
            address = arena.gather_many(
                ("trajectory_id", "path_key"), root_rows
            )
            profile_keys = logical_rng_keys(
                seed=logical_seed,
                iteration=iteration,
                trajectory=address["trajectory_id"],
                path=address["path_key"],
                draw=0,
                purpose=DrawPurpose.PRIVATE,
            )
            root_profiles[admitted:stop] = _sample_profiles_logical(
                backend,
                roots[admitted:stop],
                root_reach[admitted:stop],
                profile_keys,
                gen,
            )
            active_starts += safe_count
            if metrics is not None:
                metrics.maximum(
                    "reserved_bytes_max",
                    admission_plan.committed_bytes(active_starts),
                )
            chance_close(roots_batch)
            admitted = stop
            continue
        else:
            nonempty = [name for name in family_names if queues[name].host_count]
            if not nonempty:
                break
            selected_family = max(
                nonempty,
                key=lambda name: (
                    family_plans[name][1],
                    queues[name].oldest_age(scheduler_steps),
                    -family_priorities[family_names.index(name)],
                    -family_names.index(name),
                ),
            )
            forced_drain = True

        if full:
            forced_drain = False

        queue = queues[selected_family]
        count = family_plans[selected_family][0]
        batch_nodes = queue.dequeue(count, arena=arena).to(torch.int32)
        if metrics is not None and forced_drain:
            metrics.record_forced_drain(selected_family, rows=count)
        record_queues()
        batch_rows_index = batch_nodes.to(torch.int64)
        states = buffer.state_ref[batch_rows_index]
        reaches = reach_rows_for(batch_nodes)
        tree_ids = buffer.tree_id[batch_rows_index].to(torch.int64)
        profiles = root_profiles[tree_ids]
        address = arena.gather_many(
            ("trajectory_id", "path_key"), batch_rows_index
        )
        keys = logical_rng_keys(
            seed=logical_seed,
            iteration=iteration,
            trajectory=address["trajectory_id"],
            path=address["path_key"],
            draw=1,
            purpose=DrawPurpose.ACTION,
        )
        if probe_enabled:
            assert probe_seats is not None
            owners = buffer.owner[batch_rows_index]
            is_probe = owners == probe_seats[tree_ids]
            role_parts = (
                (is_probe.nonzero(as_tuple=False).squeeze(1), bundle.model, True),
                ((~is_probe).nonzero(as_tuple=False).squeeze(1), bundle.opponent_model, False),
            )
        else:
            role_parts = (
                (torch.arange(count, device=device), bundle.model, True),
            )
        next_states = torch.empty_like(states)
        next_reach = torch.empty_like(reaches)
        event_ids = torch.empty(count, dtype=torch.int32, device=device)
        event_payload: dict[str, Tensor] = {}
        with torch.no_grad():
            for positions, role_model, retain in role_parts:
                if positions.numel() == 0:
                    continue
                if role_model is None:
                    raise RuntimeError("mixed-policy rollout is missing its opponent model")
                output = actual_cost_forward(
                    role_model,
                    selected_family,
                    states[positions],
                    reaches[positions],
                )
                role_events = _sample_events_logical(
                    backend,
                    states[positions],
                    profiles[positions],
                    output,
                    bundle.policy_adapter,
                    keys[positions],
                    gen,
                )
                step_player = getattr(backend, "step_player_batch", backend.step)
                next_states[positions] = step_player(states[positions], role_events)
                next_reach[positions] = backend.reach_update(
                    reaches[positions], states[positions], role_events, output
                )
                event_ids[positions] = role_events.public_id
                for name, value in role_events.payload.items():
                    destination = event_payload.get(name)
                    if destination is None:
                        destination = torch.empty(
                            (count, *value.shape[1:]),
                            dtype=value.dtype,
                            device=value.device,
                        )
                        event_payload[name] = destination
                    destination[positions] = value
                if retain:
                    legal_ids, legal_mask = backend.legal_events(states[positions])
                    policy_nodes.append(batch_nodes[positions])
                    policy_logp.append(
                        _selected_policy_logp(
                            bundle.policy_adapter,
                            output,
                            role_events.public_id[:, None],
                            legal_ids,
                            legal_mask,
                        ).detach()
                    )
                    policy_values.append(output.v.detach().float())
                    decision_reach_nodes.append(batch_nodes[positions])
                    decision_reach_rows.append(reaches[positions].detach().clone())
        events = EventBatch(public_id=event_ids, payload=event_payload)
        next_owner = backend.classify(next_states).to(torch.int32)
        if probe_enabled:
            cutoff_fn = getattr(backend, "probe_fold_mask", None)
            if not callable(cutoff_fn):
                raise TypeError("probe rollout backend must implement probe_fold_mask")
            assert probe_seats is not None
            cutoff = cutoff_fn(next_states, probe_seats[tree_ids]) & is_probe
            next_owner = torch.where(
                cutoff, torch.full_like(next_owner, PROBE_FOLD), next_owner
            )
        children = append_children(
            batch_nodes, next_states, next_owner, events, next_reach
        )
        DeviceWorkQueue.mark_done(arena, batch_nodes.to(torch.int64))
        chance_close(children)

    tree = buffer.finalize()
    if admitted != start_count or active_starts != 0:
        raise RuntimeError(
            "family-scheduled rollout terminated with incomplete trajectory admission/accounting"
        )
    if leaf_reach_nodes:
        reach_nodes = torch.cat(leaf_reach_nodes).to(torch.int32)
        retained_reach = torch.cat(leaf_reach_rows).float()
        leaf_order = torch.argsort(reach_nodes.to(torch.int64))
        reach_nodes = reach_nodes[leaf_order]
        retained_reach = retained_reach[leaf_order]
    else:
        reach_nodes = torch.empty(0, dtype=torch.int32, device=device)
        retained_reach = torch.empty((0, p_max, x_max), dtype=torch.float32, device=device)
    reach_cache = ReachCache("sealed", retained_reach, reach_nodes)
    if policy_nodes:
        node_ids = torch.cat(policy_nodes).to(torch.int32)
        old_logp = torch.cat(policy_logp).float()
        old_v = torch.cat(policy_values).float()
        order = torch.argsort(node_ids.to(torch.int64))
        node_ids, old_logp, old_v = node_ids[order], old_logp[order], old_v[order]
    else:
        node_ids = torch.empty(0, dtype=torch.int32, device=device)
        old_logp = torch.empty((0, 1, x_max), dtype=torch.float32, device=device)
        old_v = torch.empty((0, x_max), dtype=torch.float32, device=device)
    if decision_reach_nodes:
        decision_nodes = torch.cat(decision_reach_nodes).to(torch.int32)
        decision_rows = torch.cat(decision_reach_rows).float()
        decision_order = torch.argsort(decision_nodes.to(torch.int64))
        decision_nodes = decision_nodes[decision_order]
        decision_rows = decision_rows[decision_order]
    else:
        decision_nodes = torch.empty(0, dtype=torch.int32, device=device)
        decision_rows = torch.empty(
            (0, p_max, x_max), dtype=torch.float32, device=device
        )
    if not torch.equal(decision_nodes, node_ids):
        raise RuntimeError("policy and natural decision-input rows are misaligned")
    state_refs = tree.state_ref[decision_nodes.to(torch.int64)].to(torch.int32)
    owners = tree.owner[decision_nodes.to(torch.int64)].to(torch.int32)
    decision_reach = DecisionReachCache.from_rows(
        decision_nodes,
        state_refs,
        decision_rows,
        backend,
        owners,
    )
    policy = PolicyCache(node_ids, old_logp, old_v, decision_reach)
    if probe_enabled:
        assert probe_seats is not None
        return ProbeRollout(
            tree, reach_cache, policy, decision_reach, probe_seats, root_profiles
        )
    result = (tree, reach_cache, policy)
    if return_profiles:
        return (*result, root_profiles)
    return result


def sample_trees_sequential(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    return_profiles: bool = False,
    iteration: int = 0,
) -> tuple[TreeBatch, ReachCache, PolicyCache] | tuple[
    TreeBatch, ReachCache, PolicyCache, Tensor
] | ProbeRollout:
    """Sample family-scheduled sequential trajectories."""

    return _sample_trees_sequential(
        bundle,
        cfg,
        gen,
        return_profiles=return_profiles,
        iteration=iteration,
    )


def sample_trees(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    iteration: int = 0,
) -> tuple[TreeBatch, ReachCache, PolicyCache] | ProbeRollout:
    """Sample trajectories for sequential RB or one tree for branched RB."""

    training = getattr(cfg, "training", None)
    mode = getattr(training, "mode", None)
    if mode == "rao_blackwell":
        return sample_trees_sequential(bundle, cfg, gen, iteration=iteration)
    if mode == "branched_rao_blackwell":
        from pbg.core.branching import sample_branched_tree

        return sample_branched_tree(bundle, cfg, gen, iteration=iteration)
    raise ValueError("sample_trees requires an RB training mode")
