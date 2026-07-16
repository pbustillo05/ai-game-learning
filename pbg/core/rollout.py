"""Device-resident sequential rollout and unified tree-sampling entry point."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch, GameBundle
from pbg.core.queued import (
    DONE,
    ChunkedOccurrenceArena,
    DeviceFifoRing,
    DrawPurpose,
    FamilyPacker,
    FamilyPackingLimits,
    SequentialAdmissionPlan,
    arena_fields,
    child_path_keys,
    logical_rng_keys,
    stable_partition_masked,
)
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



def _queued_family_spec(
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


def _queued_family_ids(backend: Any, states: Tensor) -> Tensor:
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
    same recursive path address used by the queued scheduler.
    """

    rollout_cfg = getattr(cfg, "rollout", cfg)
    n_trees = int(rollout_cfg.n_trees)
    node_budget = int(
        getattr(rollout_cfg, "node_budget", max(int(tree.parent.numel()), 1))
    )
    leaves = leaf_ids.to(device=tree.parent.device, dtype=torch.int64)
    if leaves.ndim != 1:
        raise ValueError("sequential leaf ids must be one-dimensional")
    if n_trees <= 0 or node_budget <= 0:
        raise ValueError("sequential rollout dimensions are invalid")
    if leaves.numel() != n_trees:
        raise ValueError("sequential rollout must contain one stopped leaf per trajectory")

    tree_ids = tree.tree_id.to(torch.int64)
    counts = torch.bincount(tree_ids, minlength=n_trees)
    selected_tree = tree_ids[leaves]
    valid = (
        (selected_tree >= 0)
        & (selected_tree < n_trees)
        & (counts[selected_tree.clamp(0, max(n_trees - 1, 0))] <= node_budget)
    )
    unique = torch.bincount(selected_tree.clamp_min(0), minlength=n_trees)[:n_trees] == 1
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
    start = _global_trajectory_start(cfg, n_trees)
    return logical_rng_keys(
        seed=_logical_seed(cfg, gen),
        iteration=iteration,
        trajectory=selected_tree + start,
        path=path,
        draw=draw,
        purpose=purpose,
    )


def _global_trajectory_start(cfg: Any, n_trees: int) -> int:
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
        if counts[rank] != n_trees:
            raise ValueError(
                "local rollout.n_trees does not match trajectory_rank_counts for "
                f"rank {rank}: local={n_trees}, quota={counts[rank]}"
            )
        return sum(counts[:rank])
    return rank * n_trees


def _queued_admission_plan(
    *,
    backend: Any,
    rollout_cfg: Any,
    buffer: DeviceTreeBuffer,
    arena: ChunkedOccurrenceArena,
    queues: dict[str, DeviceFifoRing],
    root_reach: Tensor,
    root_profiles: Tensor,
    working_reach: Tensor,
    batch_rows: int,
    desired_resident_starts: int,
) -> SequentialAdmissionPlan:
    """Build the pre-sampling retained plus maximum-in-flight byte transaction."""

    calibrated = getattr(backend, "queued_admission_byte_terms", None)
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

    configured_limit = getattr(rollout_cfg, "queue_memory_bytes", None)
    if configured_limit is None:
        resident_capacity = max(1, int(desired_resident_starts))
        configured_limit = (
            retained + max_inflight + resident_capacity * resident_per_start
        )
    plan = SequentialAdmissionPlan(
        limit_bytes=int(configured_limit),
        retained_bytes=retained,
        max_inflight_bytes=max_inflight,
        resident_bytes_per_start=resident_per_start,
    )
    plan.require_progress_capacity(int(rollout_cfg.n_trees))
    return plan


def _queued_family_packers(
    backend: Any,
    cfg: Any,
    family_names: tuple[str, ...],
    batch_rows: int,
) -> dict[str, FamilyPacker]:
    """Construct family-only calibrated limits without introducing stage keys."""

    hook = getattr(backend, "queued_family_packing_limits", None)
    profile = getattr(backend, "execution_profile", None)
    result: dict[str, FamilyPacker] = {}
    for family in family_names:
        if callable(hook):
            limits = hook(family, "inference", batch_rows)
            if not isinstance(limits, FamilyPackingLimits):
                raise TypeError(
                    "queued_family_packing_limits must return FamilyPackingLimits"
                )
        elif profile is not None:
            limits = FamilyPackingLimits(
                max_rows=batch_rows,
                max_padded_tokens=int(profile.inference_max_tokens),
                max_padded_token_pairs=int(profile.inference_max_token_pairs),
                max_workspace_bytes=int(
                    getattr(cfg, "queue_workspace_bytes", None) or (1 << 62)
                ),
            )
        else:
            limits = FamilyPackingLimits(
                max_rows=batch_rows,
                max_padded_tokens=1 << 62,
                max_padded_token_pairs=1 << 62,
                max_workspace_bytes=1 << 62,
            )
        result[family] = FamilyPacker(family, limits)
    return result


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
]:
    """Sample sequential paths with persistent family FIFO ready queues.

    The model scheduler may change batch boundaries without changing private,
    action, or chance draws: every stochastic boundary is addressed by the
    global trajectory id, abstract path, iteration, and draw purpose.
    """

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
    batch_rows = int(
        getattr(rollout_cfg, "queue_batch_rows", None) or min(n_trees, 256)
    )
    admission_size = int(
        getattr(rollout_cfg, "queue_admission_size", None) or batch_rows
    )
    if batch_rows <= 0 or admission_size <= 0:
        raise ValueError("queued batch and admission sizes must be positive")
    chunk_nodes = int(
        getattr(rollout_cfg, "arena_chunk_nodes", None)
        or min(n_trees * node_budget, max(batch_rows, 256))
    )

    p_max = int(backend.n_players)
    x_sizes = tuple(int(value) for value in backend.x_sizes)
    x_max = max(x_sizes)
    roots = backend.sequential_start_states(n_trees)
    if roots.shape != (n_trees,) or roots.dtype != torch.int32 or roots.device != device:
        raise ValueError("sequential_start_states must return device-local int32 [count]")
    root_owner = backend.classify(roots).to(torch.int32)
    metrics = getattr(backend, "execution_metrics", None)
    buffer = DeviceTreeBuffer(n_trees, node_budget, p_max, device)
    root_nodes = buffer.add_roots(roots, root_owner)
    arena = ChunkedOccurrenceArena(
        buffer.node_capacity,
        chunk_nodes,
        device,
        fields=arena_fields(("trajectory_id", "path_key")),
        label="sequential_occurrences",
    )
    root_handles = arena.allocate(n_trees)
    expected_roots = root_nodes.to(torch.int64)
    if device.type == "cuda":
        torch._assert_async(
            (root_handles == expected_roots).all(),
            "queued occurrence and tree root handles diverged",
        )
    elif not torch.equal(root_handles, expected_roots):
        raise RuntimeError("queued occurrence and tree root handles diverged")

    trajectory_start = _global_trajectory_start(cfg, n_trees)
    trajectory_ids = torch.arange(
        trajectory_start,
        trajectory_start + n_trees,
        dtype=torch.int64,
        device=device,
    )
    arena.scatter_many(
        root_handles,
        {
            "trajectory_id": trajectory_ids,
            "path_key": torch.zeros(n_trees, dtype=torch.int64, device=device),
        },
    )

    root_reach = torch.zeros(
        (n_trees, p_max, x_max), dtype=torch.float32, device=device
    )
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    # A sequential trajectory has exactly one live occurrence.  Replay mode
    # therefore needs one transient row per trajectory, not one row per
    # possible node; the row is overwritten as that trajectory advances.
    working_reach_rows = (
        n_trees
        if rollout_cfg.reach_storage == "recompute"
        else buffer.node_capacity
    )
    working_reach = torch.empty(
        (working_reach_rows, p_max, x_max), dtype=torch.float32, device=device
    )
    working_reach[:n_trees] = root_reach
    root_profiles = torch.empty(
        (n_trees, p_max), dtype=torch.int32, device=device
    )

    family_names, family_values, family_priorities = _queued_family_spec(backend)
    queues = {
        name: DeviceFifoRing(n_trees, device) for name in family_names
    }
    packers = _queued_family_packers(
        backend, rollout_cfg, family_names, batch_rows
    )
    queue_packing = getattr(
        rollout_cfg, "queue_packing", "fixed_family_quantum"
    )
    if queue_packing == "actual_cost_greedy" and not callable(
        getattr(backend, "policy_forward_family_actual_cost", None)
    ):
        raise ValueError(
            "actual_cost_greedy requires a backend actual-cost family forward"
        )
    admission_plan = _queued_admission_plan(
        backend=backend,
        rollout_cfg=rollout_cfg,
        buffer=buffer,
        arena=arena,
        queues=queues,
        root_reach=root_reach,
        root_profiles=root_profiles,
        working_reach=working_reach,
        batch_rows=batch_rows,
        desired_resident_starts=min(n_trees, admission_size),
    )
    logical_seed = _logical_seed(cfg, gen)
    policy_nodes: list[Tensor] = []
    policy_logp: list[Tensor] = []
    policy_values: list[Tensor] = []
    scheduler_steps = 0
    active_starts = 0
    if metrics is not None:
        metrics.maximum("reserved_bytes_max", admission_plan.base_bytes)

    def reach_rows_for(nodes: Tensor) -> Tensor:
        rows = nodes.to(torch.int64)
        if rollout_cfg.reach_storage == "recompute":
            rows = buffer.tree_id[rows].to(torch.int64)
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
        """Return logical rows, calibrated fill, and full-threshold status."""

        queue = queues[name]
        if queue.host_count == 0:
            return 0, 0.0, False
        packer = packers[name]
        candidate_limit = packer.limits.max_rows
        static_capacity = getattr(backend, "queued_static_pack_capacity", None)
        actual_cost_greedy = queue_packing == "actual_cost_greedy"
        if device.type == "cuda" and callable(static_capacity) and not actual_cost_greedy:
            candidate_limit = min(
                candidate_limit,
                int(static_capacity(bundle.model, name, "inference", packer.limits)),
            )
            if candidate_limit <= 0:
                raise MemoryError(
                    f"execution family {name!r} has no statically safe CUDA row"
                )
        candidate_count = min(queue.host_count, candidate_limit)
        candidates = queue.peek(candidate_count)
        states = buffer.state_ref[candidates]
        row_cost = getattr(backend, "rollout_row_cost", None)
        if callable(row_cost):
            padded_width, workspace = row_cost(bundle.model, states)
        else:
            padded_width = torch.ones_like(candidates, dtype=torch.int64)
            workspace = torch.zeros_like(candidates, dtype=torch.int64)
        if device.type == "cpu":
            packed = packer.pack(candidates, padded_width, workspace)
            selected = int(packed.handles.numel())
            fill = float(packed.fill)
        elif actual_cost_greedy:
            # This is a logical dequeue block. The backend transfers its fixed-
            # shape cost vector once and emits several actual-cost physical
            # calls, so the block itself need not fit one worst-case launch.
            selected = candidate_count
            fill = selected / float(packer.limits.max_rows)
        elif bool(getattr(backend, "queued_static_pack_safe", False)) or callable(
            static_capacity
        ):
            # Backend calibration proves this fixed candidate quantum is
            # feasible.  Consume the masked device representation and assert
            # that proof on-device; its computed prefix length never becomes a
            # host scalar.
            masked = packer.pack_masked(candidates, padded_width, workspace)
            torch._assert_async(
                masked.selected_mask.all(),
                "calibrated CUDA family quantum exceeds masked packing limits",
            )
            selected = candidate_count
            fill = selected / float(candidate_limit)
        else:
            # Until a backend provides a static/masked execution kernel, retain
            # its existing maximum-feasible physical splitter on CUDA.
            selected = candidate_count
            fill = selected / float(packer.limits.max_rows)
        full_threshold = (
            selected == packer.limits.max_rows or selected < queue.host_count
        )
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
                "queued occurrence and tree child handles diverged",
            )
        elif not torch.equal(child_handles, children.to(torch.int64)):
            raise RuntimeError("queued occurrence and tree child handles diverged")
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
        reach_rows = children.to(torch.int64)
        if rollout_cfg.reach_storage == "recompute":
            reach_rows = buffer.tree_id[reach_rows].to(torch.int64)
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
            family_ids = _queued_family_ids(backend, states)
            chance_route = -1
            leaf_route = -2
            invalid_route = -3
            player = (owners >= 0) & (owners < p_max)
            route = torch.where(
                owners == CHANCE,
                torch.full_like(family_ids, chance_route),
                torch.where(
                    (owners == TERMINAL) | (owners == PRETERM),
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
            if stopped > active_starts:
                raise RuntimeError("queued active-trajectory accounting underflow")
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
    while admitted < n_trees or any(queue.host_count for queue in queues.values()):
        scheduler_steps += 1
        if scheduler_steps > n_trees * (int(backend.max_rollout_steps) + 2):
            raise RuntimeError("queued rollout scheduler exceeded its progress bound")

        family_plans = {
            name: planned_family_pack(name)
            for name in family_names
            if queues[name].host_count
        }
        full = [name for name, plan in family_plans.items() if plan[2]]
        selected_family: str | None = None
        if full:
            # family_names is a topological priority order; within the first
            # priority layer the fullest FIFO wins with stable ties.
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
        elif admitted < n_trees and (
            safe_count := admission_plan.safe_admit(
                active_starts=active_starts,
                unstarted=n_trees - admitted,
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
        with torch.no_grad():
            actual_cost_forward = getattr(
                backend, "policy_forward_family_actual_cost", None
            )
            family_forward = getattr(backend, "policy_forward_family", None)
            output = (
                actual_cost_forward(bundle.model, selected_family, states, reaches)
                if queue_packing == "actual_cost_greedy"
                and callable(actual_cost_forward)
                else family_forward(bundle.model, selected_family, states, reaches)
                if callable(family_forward)
                else backend.policy_forward(bundle.model, states, reaches)
            )
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
            events = _sample_events_logical(
                backend,
                states,
                profiles,
                output,
                bundle.policy_adapter,
                keys,
                gen,
            )
            step_player = getattr(backend, "step_player_batch", backend.step)
            next_states = step_player(states, events)
            next_reach = backend.reach_update(reaches, states, events, output)
        next_owner = backend.classify(next_states).to(torch.int32)
        children = append_children(
            batch_nodes, next_states, next_owner, events, next_reach
        )
        legal_ids, legal_mask = backend.legal_events(states)
        policy_nodes.append(batch_nodes)
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
        DeviceFifoRing.mark_done(arena, batch_nodes.to(torch.int64))
        chance_close(children)

    tree = buffer.finalize()
    if admitted != n_trees or active_starts != 0:
        raise RuntimeError(
            "queued rollout terminated with incomplete trajectory admission/accounting"
        )
    all_nodes = torch.arange(buffer.node_count, dtype=torch.int32, device=device)
    if rollout_cfg.reach_storage == "full":
        reach_nodes = all_nodes
        retained_reach = working_reach[: buffer.node_count]
    elif rollout_cfg.reach_storage == "decisions_only":
        stored = ((tree.owner >= 0) & (tree.owner < p_max)) | (
            (tree.owner == TERMINAL) | (tree.owner == PRETERM)
        )
        reach_nodes = all_nodes[stored]
        retained_reach = working_reach[reach_nodes.to(torch.int64)]
    else:
        reach_nodes = root_nodes
        retained_reach = root_reach
    reach_cache = ReachCache(rollout_cfg.reach_storage, retained_reach, reach_nodes)
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
    result = (tree, reach_cache, PolicyCache(node_ids, old_logp, old_v))
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
]:
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
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Sample trajectories for sequential RB or one tree for branched RB."""

    rollout_cfg = getattr(cfg, "rollout", cfg)
    training = getattr(cfg, "training", None)
    mode = getattr(training, "mode", None)
    if mode == "rao_blackwell":
        return sample_trees_sequential(bundle, cfg, gen, iteration=iteration)
    if mode == "branched_rao_blackwell":
        from pbg.core.branching import sample_branched_tree

        return sample_branched_tree(bundle, cfg, gen, iteration=iteration)
    raise ValueError("sample_trees requires an RB training mode")
