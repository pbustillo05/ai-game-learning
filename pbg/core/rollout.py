"""Device-resident sequential rollout and unified tree-sampling entry point."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from pbg.core.br import training_probe_seats
from pbg.core.game import CHANCE, PRETERM, PROBE_FOLD, TERMINAL, EventBatch, GameBundle
from pbg.core.partition import PartitionSet
from pbg.core.execution import (
    DONE,
    ChunkedOccurrenceArena,
    DeviceWorkQueue,
    DrawPurpose,
    TrajectoryAdmissionPlan,
    arena_fields,
    child_path_keys,
    logical_rng_keys,
    select_work_queue,
    stable_partition_masked,
)
from pbg.core.tree import PagedRolloutTopology
from pbg.core.storage import PagedTensorArena, SealedRolloutSample, stopped_leaf_source_schema
from pbg.core.topology import SequentialDecisionCSR


@dataclass
class ProbeRollout:
    """Sequential mixed-policy data retained only for the exploiter."""

    sample: SealedRolloutSample
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



def _rollout_partitions(backend: Any) -> PartitionSet:
    factory = getattr(backend, "training_partition_specs", None)
    if not callable(factory):
        raise TypeError("rollout backend must expose its immutable PartitionSet")
    result = factory()
    if not isinstance(result, PartitionSet):
        raise TypeError("training_partition_specs must return PartitionSet")
    return result


def _partition_ids(partitions: PartitionSet, states: Tensor) -> Tensor:
    result = partitions.route(states)
    if not isinstance(result, Tensor):
        result = torch.full_like(states, int(result), dtype=torch.int64)
    if result.shape != states.shape or result.device != states.device:
        raise ValueError("PartitionSet.route must return device-local [B]")
    return result.to(torch.int64)


def _logical_seed(cfg: Any, gen: torch.Generator) -> int:
    rollout_cfg = getattr(cfg, "rollout", cfg)
    training_cfg = getattr(cfg, "training", None)
    configured = getattr(rollout_cfg, "logical_seed", None)
    if configured is None and training_cfg is not None:
        configured = getattr(training_cfg, "seed", None)
    return int(gen.initial_seed() if configured is None else configured)


def sequential_leaf_logical_rng_keys(
    root_id: Tensor,
    path_key: Tensor,
    cfg: Any,
    gen: torch.Generator,
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
    leaves = root_id.to(dtype=torch.int64)
    paths = path_key.to(device=leaves.device, dtype=torch.int64)
    if leaves.ndim != 1 or paths.shape != leaves.shape:
        raise ValueError("sequential leaf root/path addresses must be matching vectors")
    if start_count <= 0:
        raise ValueError("sequential rollout dimensions are invalid")
    if leaves.numel() != start_count:
        raise ValueError("sequential rollout must contain one stopped leaf per trajectory")

    selected_tree = leaves
    valid = (
        (selected_tree >= 0)
        & (selected_tree < start_count)
    )
    unique = torch.bincount(selected_tree.clamp_min(0), minlength=start_count)[:start_count] == 1
    if leaves.device.type == "cuda":
        torch._assert_async(valid.all(), "invalid sequential stopped-leaf address")
        torch._assert_async(unique.all(), "sequential trajectories require one stopped leaf")
    elif not bool(valid.all()) or not bool(unique.all()):
        raise ValueError("invalid sequential stopped-leaf topology")

    start = _global_trajectory_start(cfg, start_count)
    return logical_rng_keys(
        seed=_logical_seed(cfg, gen),
        iteration=iteration,
        trajectory=selected_tree + start,
        path=paths,
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
    buffer: PagedRolloutTopology,
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

        page_count = math.ceil(arena.capacity / buffer.nodes.page_rows)
        retained = buffer.nodes.schema.capacity_bytes(
            arena.capacity, chunks=page_count
        ) + buffer.edges.schema.capacity_bytes(
            max(0, arena.capacity - int(rollout_cfg.roots_per_gpu)),
            chunks=max(0, page_count),
        )
        # The sequential exact path bound is a feasibility reservation, not an
        # eager tensor allocation. Realized topology still grows by pages.
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
        retained += arena.capacity * 2 * x_max * 4
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
) -> SealedRolloutSample | ProbeRollout:
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
    if start_count < 0:
        raise ValueError("start_count must be nonnegative")
    batch_rows = int(
        getattr(rollout_cfg, "family_batch_rows", None) or min(start_count, 256)
    )
    admission_size = int(
        getattr(rollout_cfg, "max_active_trajectories", None) or batch_rows
    )
    if batch_rows <= 0 or admission_size <= 0:
        raise ValueError("family-scheduled batch and admission sizes must be positive")
    max_rows = max(1, start_count * (int(backend.max_rollout_steps) + 1))
    chunk_nodes = int(
        getattr(backend, "storage_page_rows", None)
        or getattr(rollout_cfg, "storage_page_rows", None)
        or min(max_rows, max(batch_rows, 256))
    )

    p_max = int(backend.n_players)
    x_sizes = tuple(int(value) for value in backend.x_sizes)
    x_max = max(x_sizes)
    roots = backend.sequential_start_states(start_count)
    if roots.shape != (start_count,) or roots.dtype != torch.int32 or roots.device != device:
        raise ValueError("sequential_start_states must return device-local int32 [count]")
    root_owner = backend.classify(roots).to(torch.int32)
    metrics = getattr(backend, "execution_metrics", None)
    buffer = PagedRolloutTopology(start_count, p_max, device, page_rows=chunk_nodes)
    root_nodes = buffer.add_roots(roots, root_owner)
    arena = ChunkedOccurrenceArena(
        max_rows,
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
    capture_inputs = getattr(backend, "capture_decision_inputs", None)
    begin_captures = getattr(backend, "begin_decision_input_captures", None)
    reserve_capture = getattr(backend, "reserve_decision_input_captures", None)
    write_capture = getattr(backend, "write_decision_input_capture", None)
    seal_captures = getattr(backend, "seal_decision_input_captures", None)
    if not all(
        callable(value)
        for value in (capture_inputs, begin_captures, reserve_capture, write_capture, seal_captures)
    ):
        raise TypeError(
            "sequential rollout backends must capture immutable natural decision inputs"
        )
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
            probe_seats = training_probe_seats(seat_keys, p_max)
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

    partitions = _rollout_partitions(backend)
    partition_ids = tuple(spec.partition_id for spec in partitions.specs)
    partition_preorder = tuple(spec.preorder for spec in partitions.specs)
    queues = {
        partition_id: DeviceWorkQueue(
            start_count,
            device,
            staging_rows=min(start_count, batch_rows),
            label=f"sequential_queue_partition_{partition_id}",
        )
        for partition_id in partition_ids
    }
    actual_cost_forward = getattr(
        backend, "policy_forward_partition_actual_cost", None
    )
    if not callable(actual_cost_forward):
        raise ValueError("production requires an actual-cost partition forward")
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
    training_mode = getattr(getattr(cfg, "training", None), "mode", None)
    source_mode = (
        "normal"
        if (training_mode == "normal" or return_profiles) and not probe_enabled
        else "sequential_rb"
    )
    decision_input_store = begin_captures(mode=source_mode)
    leaf_schema_factory = getattr(backend, "rollout_leaf_source_schema", None)
    leaf_schema = (
        leaf_schema_factory("probe" if probe_enabled else source_mode)
        if callable(leaf_schema_factory)
        else stopped_leaf_source_schema(
            "generic", players=p_max, exact_width=x_max
        )
    )
    leaf_rows = PagedTensorArena(
        leaf_schema,
        chunk_nodes,
        device,
        label="sequential_stopped_leaf_rows",
    )
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

    def record_queues() -> None:
        if metrics is None:
            return
        for partition_id, queue in queues.items():
            metrics.record_queue(
                partitions[partition_id].name,
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
            routed_partition_ids = _partition_ids(partitions, states)
            chance_route = -1
            leaf_route = -2
            invalid_route = -3
            player = (owners >= 0) & (owners < p_max)
            route = torch.where(
                owners == CHANCE,
                torch.full_like(routed_partition_ids, chance_route),
                torch.where(
                    (owners == TERMINAL) | (owners == PRETERM) | (owners == PROBE_FOLD),
                    torch.full_like(routed_partition_ids, leaf_route),
                    torch.where(
                        player,
                        routed_partition_ids,
                        torch.full_like(routed_partition_ids, invalid_route),
                    ),
                ),
            )
            routed = stable_partition_masked(
                frontier.to(torch.int64),
                route,
                (*partition_ids, chance_route, leaf_route),
            )
            counts = read_control_counts(routed.selected_count)
            partition_counts = counts[: len(partition_ids)]
            chance_count, stopped = counts[-2:]
            if stopped:
                stopped_nodes = routed.candidate_handles[-1, :stopped].to(torch.int32)
                stopped_rows = stopped_nodes.to(torch.int64)
                stopped_states = buffer.state_ref[stopped_rows].to(torch.int32)
                stopped_owners = buffer.owner[stopped_rows].to(torch.int32)
                stopped_reach = reach_rows_for(stopped_nodes).detach().clone()
                leaf_address = arena.gather_many(
                    ("trajectory_id", "path_key"), stopped_rows
                )
                capture_leaf = None
                mode = getattr(getattr(cfg, "training", None), "mode", None)
                if probe_enabled or mode == "normal" or return_profiles:
                    capture_name = (
                        "capture_probe_leaf" if probe_enabled else "capture_normal_leaf"
                    )
                    capture = getattr(backend, capture_name, None)
                    if callable(capture):
                        tree_ids = buffer.tree_id[stopped_rows].to(torch.int64)
                        if probe_enabled:
                            assert probe_seats is not None
                            capture_leaf = capture(
                                stopped_states,
                                stopped_reach,
                                stopped_owners,
                                probe_seats[tree_ids],
                            )
                        else:
                            address = arena.gather_many(
                                ("trajectory_id", "path_key"), stopped_rows
                            )
                            keys = logical_rng_keys(
                                seed=logical_seed,
                                iteration=iteration,
                                trajectory=address["trajectory_id"],
                                path=address["path_key"],
                                draw=1,
                                purpose=DrawPurpose.COMPLETION,
                            )
                            capture_leaf = capture(
                                stopped_states, root_profiles[tree_ids], keys
                            )
                else:
                    capture = getattr(backend, "capture_rb_leaf", None)
                    if callable(capture):
                        capture_leaf = capture(stopped_states, stopped_reach)
                leaf_rows.append(
                    {
                        "node_id": stopped_nodes,
                        "root_id": buffer.tree_id[stopped_rows].to(torch.int32),
                        "path_key": leaf_address["path_key"].to(torch.int64),
                        "log_reach": stopped_reach,
                        **({} if capture_leaf is None else capture_leaf),
                    }
                )
                if capture_leaf is not None:
                    release = getattr(backend, "release_state_rows", None)
                    if callable(release):
                        release(stopped_states)
                        buffer.state_ref[stopped_rows] = -1
            if stopped > active_starts:
                raise RuntimeError("family-scheduled active-trajectory accounting underflow")
            active_starts -= stopped
            if metrics is not None:
                metrics.maximum(
                    "reserved_bytes_max",
                    admission_plan.committed_bytes(active_starts),
                )
            for index, (partition_id, count) in enumerate(
                zip(partition_ids, partition_counts, strict=True)
            ):
                if count:
                    queues[partition_id].enqueue(
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
            release = getattr(backend, "release_state_rows", None)
            if callable(release):
                release(states)
                buffer.state_ref[chance_rows] = -1

    admitted = 0
    while admitted < start_count or any(queue.host_count for queue in queues.values()):
        scheduler_steps += 1
        if scheduler_steps > start_count * (int(backend.max_rollout_steps) + 2):
            raise RuntimeError("family-scheduled rollout scheduler exceeded its progress bound")

        selection = select_work_queue(
            partition_ids,
            partition_preorder,
            queues,
            batch_rows,
            allow_partial=False,
            logical_time=scheduler_steps,
        )
        if selection is None and admitted < start_count and (
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
            if selection is None:
                selection = select_work_queue(
                    partition_ids,
                    partition_preorder,
                    queues,
                    batch_rows,
                    allow_partial=True,
                    logical_time=scheduler_steps,
                )
            if selection is None:
                break
        partition_id = selection.partition_id
        forced_drain = selection.forced_partial
        queue = queues[partition_id]
        count = selection.rows
        batch_nodes = queue.dequeue(count, arena=arena).to(torch.int32)
        if metrics is not None and forced_drain:
            metrics.record_forced_drain(partitions[partition_id].name, rows=count)
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
        physical_pack = partitions[partition_id].byte_work.realize(
            states, rows=count
        )
        reserved_by_role: dict[int, Tensor] = {}
        for role_index, (positions, _, retain) in enumerate(role_parts):
            if retain and positions.numel():
                reserved_by_role[role_index] = reserve_capture(
                    decision_input_store, partition_id, positions.numel()
                )
        next_states = torch.empty_like(states)
        next_reach = torch.empty_like(reaches)
        event_ids = torch.empty(count, dtype=torch.int32, device=device)
        event_payload: dict[str, Tensor] = {}
        with torch.no_grad():
            for role_index, (positions, role_model, retain) in enumerate(role_parts):
                if positions.numel() == 0:
                    continue
                if role_model is None:
                    raise RuntimeError("mixed-policy rollout is missing its opponent model")
                role_states = states[positions]
                role_reaches = reaches[positions]
                role_profiles = profiles[positions]
                role_keys = keys[positions]
                role_nodes = batch_nodes[positions]
                role_rows = batch_rows_index[positions]
                output = actual_cost_forward(
                    role_model,
                    partition_id,
                    role_states,
                    role_reaches,
                    physical_pack=(
                        physical_pack
                        if positions.numel() == count
                        else partitions[partition_id].byte_work.realize(
                            role_states, rows=positions.numel()
                        )
                    ),
                )
                role_events = _sample_events_logical(
                    backend,
                    role_states,
                    role_profiles,
                    output,
                    bundle.policy_adapter,
                    role_keys,
                    gen,
                )
                selected_logp_fn = getattr(
                    backend, "selected_policy_logp", None
                )
                role_selected_logp = (
                    selected_logp_fn(
                        role_states,
                        role_events,
                        output,
                        bundle.policy_adapter,
                    )
                    if callable(selected_logp_fn)
                    else None
                )
                step_player = getattr(backend, "step_player_batch", backend.step)
                next_states[positions] = step_player(role_states, role_events)
                if role_selected_logp is None:
                    next_reach[positions] = backend.reach_update(
                        role_reaches, role_states, role_events, output
                    )
                else:
                    next_reach[positions] = backend.reach_update(
                        role_reaches,
                        role_states,
                        role_events,
                        output,
                        selected_logp=role_selected_logp,
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
                    if role_selected_logp is None:
                        legal_ids, legal_mask = backend.legal_events(
                            role_states
                        )
                        selected_logp = _selected_policy_logp(
                            bundle.policy_adapter,
                            output,
                            role_events.public_id[:, None],
                            legal_ids,
                            legal_mask,
                        ).detach()[:, 0]
                    else:
                        selected_logp = role_selected_logp
                    write_capture(
                        decision_input_store,
                        reserved_by_role[role_index],
                        role_nodes,
                        capture_inputs(
                            partition_id,
                            role_states,
                            role_reaches,
                        ),
                        {
                            "node_id": role_nodes.to(torch.int32),
                            "root_id": buffer.tree_id[role_rows].to(torch.int32),
                            "decision_ordinal": (
                                buffer.depth_player[role_rows] - 1
                            ).to(torch.int32),
                            "owner": buffer.owner[role_rows].to(torch.int32),
                            "event": role_events.public_id.to(torch.int32),
                            "old_logp": selected_logp.to(torch.float32),
                            "old_v": output.v.detach().to(torch.float32),
                        },
                    )
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
        release = getattr(backend, "release_state_rows", None)
        if callable(release):
            release(states)
            buffer.state_ref[batch_nodes.to(torch.int64)] = -1
        DeviceWorkQueue.mark_done(arena, batch_nodes.to(torch.int64))
        chance_close(children)

    if admitted != start_count or active_starts != 0:
        raise RuntimeError(
            "family-scheduled rollout terminated with incomplete trajectory admission/accounting"
        )
    for queue in queues.values():
        queue.release()
    if not leaf_rows.size:
        raise RuntimeError("sequential rollout produced no stopped-leaf rows")
    leaf_root = leaf_rows.gather(
        "root_id", torch.arange(leaf_rows.size, device=device)
    )
    leaf_order = torch.argsort(leaf_root.to(torch.int64), stable=True)
    leaf_block, leaf_metrics = leaf_rows.seal(
        scratch_rows=chunk_nodes, row_order=leaf_order, release_source=True
    )
    verify_recycling = getattr(backend, "verify_state_recycling_boundary", None)
    if callable(verify_recycling):
        verify_recycling()
    decision_inputs = seal_captures(decision_input_store)
    if not decision_inputs.node_ids.numel():
        raise RuntimeError("sequential rollout produced no decision rows")
    source_root = torch.empty_like(decision_inputs.node_ids)
    source_ordinal = torch.empty_like(decision_inputs.node_ids)
    for partition_id, source in decision_inputs.decision_sources.items():
        selected = decision_inputs.partition_id == partition_id
        rows = decision_inputs.partition_row[selected].to(torch.int64)
        source_root[selected] = source["root_id"][rows]
        source_ordinal[selected] = source["decision_ordinal"][rows]
    topology = SequentialDecisionCSR.build(
        root_id=source_root,
        decision_ordinal=source_ordinal,
        decision_partition=decision_inputs.partition_id,
        decision_row=decision_inputs.partition_row,
        leaf_root_id=leaf_block["root_id"],
        leaf_row=torch.arange(leaf_block.rows, dtype=torch.int32, device=device),
        root_count=start_count,
    )
    all_nodes = torch.arange(buffer.node_count, device=device)
    trajectory_node_count = torch.bincount(
        buffer.tree_id[all_nodes].to(torch.int64), minlength=start_count
    ).to(torch.int32)
    node_count = buffer.node_count
    arena.release()
    buffer.release()
    mode = (
        "probe"
        if probe_enabled
        else "sequential_rb"
        if training_mode == "rao_blackwell"
        else "normal"
    )
    sample = SealedRolloutSample(
        mode=mode,
        topology=topology,
        leaf_source=leaf_block,
        decision_inputs=decision_inputs,
        node_count=node_count,
        root_profiles=root_profiles if (return_profiles or probe_enabled) else None,
        probe_seats=probe_seats,
        trajectory_node_count=trajectory_node_count,
        seal_metrics={
            "leaf_rows": leaf_metrics,
            **decision_inputs.seal_metrics,
        },
    )
    if probe_enabled:
        assert probe_seats is not None
        return ProbeRollout(sample, probe_seats, root_profiles)
    return sample


def sample_trees_sequential(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    return_profiles: bool = False,
    iteration: int = 0,
) -> SealedRolloutSample | ProbeRollout:
    """Sample family-scheduled sequential trajectories."""

    try:
        return _sample_trees_sequential(
            bundle,
            cfg,
            gen,
            return_profiles=return_profiles,
            iteration=iteration,
        )
    finally:
        release = getattr(bundle.rollout_backend, "release_rollout_owners", None)
        if callable(release):
            release()


def sample_trees(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    iteration: int = 0,
    branching_controller: Any | None = None,
    safeguard_controller: Any | None = None,
    branching_memory_controller: Any | None = None,
    branching_attempt: int = 0,
    branching_diagnostic_limits: Any | None = None,
) -> SealedRolloutSample | ProbeRollout:
    """Sample trajectories for sequential RB or one tree for branched RB."""

    training = getattr(cfg, "training", None)
    mode = getattr(training, "mode", None)
    if mode == "rao_blackwell":
        return sample_trees_sequential(bundle, cfg, gen, iteration=iteration)
    if mode == "branched_rao_blackwell":
        from pbg.core.branching import sample_branched_tree

        return sample_branched_tree(
            bundle, cfg, gen, iteration=iteration,
            controller=branching_controller,
            safeguard_controller=safeguard_controller,
            memory_controller=branching_memory_controller,
            attempt=branching_attempt,
            diagnostic_limits=branching_diagnostic_limits,
        )
    raise ValueError("sample_trees requires an RB training mode")
