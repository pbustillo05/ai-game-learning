"""Fixed or post-warmup dynamic duplicate-preserving labeled branching."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL, GameBundle
from pbg.core.adaptive_branching import (
    child_path_factors,
    log_kkt_intensity,
    randomized_round,
)
from pbg.core.tree import PagedRolloutTopology
from pbg.core.storage import SealedRolloutSample
from pbg.core.topology import BranchedTreeSpan


class BranchingCutoffReached(RuntimeError):
    """A geometry-only rollout stopped with a live, unexpanded frontier."""

    def __init__(
        self,
        *,
        node_count: int,
        queued_nodes: int,
        frontier_nodes: int,
        family_queue_nodes: Mapping[str, int],
        completed_leaves: int,
        completed_decisions: int,
    ) -> None:
        self.node_count = node_count
        self.queued_nodes = queued_nodes
        self.frontier_nodes = frontier_nodes
        self.pending_nodes = queued_nodes + frontier_nodes
        self.family_queue_nodes = dict(family_queue_nodes)
        self.completed_leaves = completed_leaves
        self.completed_decisions = completed_decisions
        super().__init__(
            f"branched rollout stopped at {node_count} nodes with "
            f"{self.pending_nodes} live nodes pending"
        )


@dataclass(frozen=True)
class BranchingDiagnosticLimits:
    """Evaluation-only node stops; never part of the training configuration."""

    upper_nodes: int | None = None
    stop_nodes: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("upper_nodes", self.upper_nodes), ("stop_nodes", self.stop_nodes)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 1
            ):
                raise ValueError(f"diagnostic {name} must be an integer greater than one")


def sample_branched_tree(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    iteration: int = 0,
    diagnostics: dict[str, int] | None = None,
    controller: Any | None = None,
    safeguard_controller: Any | None = None,
    memory_controller: Any | None = None,
    attempt: int = 0,
    diagnostic_limits: BranchingDiagnosticLimits | None = None,
) -> SealedRolloutSample:
    """Sample one tree and release mutable backend owners on every exit."""

    if isinstance(attempt, bool) or attempt < 0:
        raise ValueError("branched rollout attempt must be nonnegative")
    try:
        return _sample_branched_tree_impl(
            bundle,
            cfg,
            gen,
            iteration=iteration,
            diagnostics=diagnostics,
            controller=controller,
            safeguard_controller=safeguard_controller,
            memory_controller=memory_controller,
            attempt=attempt,
            diagnostic_limits=diagnostic_limits,
        )
    except MemoryError as exc:
        from pbg.core.memory_control import BranchedMemoryRejection

        if memory_controller is None or isinstance(exc, BranchedMemoryRejection):
            raise
        device = torch.device(bundle.rollout_backend.device)
        allocated_peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        reserved_peak = (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        )
        raise BranchedMemoryRejection(
            phase=str(getattr(memory_controller, "last_phase", None) or "rollout"),
            reason=f"{type(exc).__name__}: {exc}",
            predicted_bytes=max(
                int(getattr(memory_controller, "last_predicted_bytes", 0)),
                allocated_peak,
                reserved_peak,
            ),
            hard_ceiling_bytes=int(memory_controller.hard_ceiling_bytes),
            allocated_peak_bytes=allocated_peak,
            reserved_peak_bytes=reserved_peak,
        ) from exc
    finally:
        release = getattr(bundle.rollout_backend, "release_rollout_owners", None)
        if callable(release):
            release()


def _sample_branched_tree_impl(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    iteration: int = 0,
    diagnostics: dict[str, int] | None = None,
    controller: Any | None = None,
    safeguard_controller: Any | None = None,
    memory_controller: Any | None = None,
    attempt: int = 0,
    diagnostic_limits: BranchingDiagnosticLimits | None = None,
) -> SealedRolloutSample:
    """Sample one duplicate-preserving branched tree through execution families."""

    if diagnostics is None:
        diagnostics = {}

    # Imports are local because rollout dispatch imports this module lazily.
    import time
    import torch.distributed as dist

    from pbg.core.execution import (
        ArenaField,
        ByteAccountant,
        DONE,
        ChunkedOccurrenceArena,
        DeviceWorkQueue,
        DrawPurpose,
        arena_fields,
        child_path_keys,
        logical_rng_keys,
        logical_uniform,
        select_work_queue,
        stable_partition_masked,
    )
    from pbg.core.storage import (
        RecyclableTensorPool,
        PagedTensorArena,
        PagedTensorColumn,
        SealedTensorBlock,
        TensorFieldSchema,
        TensorSchema,
        stopped_leaf_source_schema,
    )
    from pbg.core.rollout import (
        _logical_seed,
        _family_ids,
        _family_spec,
        _sample_events_logical,
        _sample_profiles_logical,
        _selected_policy_logp,
    )

    rollout_cfg = getattr(cfg, "rollout", cfg)
    backend = bundle.rollout_backend
    device = torch.device(backend.device)
    if torch.device(gen.device) != device:
        raise ValueError("rollout generator must be created on the backend device")
    if int(rollout_cfg.roots_per_gpu) != 1:
        raise ValueError("family-scheduled branched rollout requires one root")

    p_max = int(backend.n_players)
    x_sizes = tuple(int(value) for value in backend.x_sizes)
    x_max = max(x_sizes)
    page_rows = int(getattr(rollout_cfg, "storage_page_rows", None) or 256)
    child_pack_rows = int(
        getattr(rollout_cfg, "family_batch_rows", None) or page_rows
    )
    if child_pack_rows <= 0:
        raise ValueError("branched physical child rows must be positive")
    execution_limit = int(
        getattr(rollout_cfg, "execution_memory_bytes", None) or (1 << 30)
    )
    accountant = ByteAccountant(execution_limit)
    bind_accountant = getattr(backend, "bind_rollout_accountant", None)
    if callable(bind_accountant):
        bind_accountant(accountant)
    limits = diagnostic_limits or BranchingDiagnosticLimits()
    upper_cutoff = limits.upper_nodes
    stop_cutoff = limits.stop_nodes
    minimum_nodes = getattr(rollout_cfg, "branching_min_nodes", None)
    if minimum_nodes is not None and (
        isinstance(minimum_nodes, bool)
        or not isinstance(minimum_nodes, int)
        or minimum_nodes <= 1
    ):
        raise ValueError(
            "branching_min_nodes must be an integer greater than one"
        )
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "upper_cutoff_batches": 0,
                "upper_cutoff_parents": 0,
                "upper_cutoff_avoided_children": 0,
                "upper_cutoff_first_node_count": -1,
            }
        )
    roots = backend.branched_root_state()
    if roots.shape != (1,) or roots.dtype != torch.int32 or roots.device != device:
        raise ValueError("branched_root_state must return device-local int32 [1]")
    root_owner = backend.classify(roots).to(torch.int32)
    buffer = PagedRolloutTopology(
        1,
        p_max,
        device,
        page_rows=page_rows,
        accountant=accountant,
    )
    root_nodes = buffer.add_roots(roots, root_owner)
    arena = ChunkedOccurrenceArena(
        None,
        page_rows,
        device,
        fields=arena_fields(
            ("path_key",),
            extra={"reach_handle": ArenaField(torch.int64, fill=-1)},
        ),
        accountant=accountant,
        label="branched_occurrences",
    )
    root_handles = arena.allocate(1)
    if device.type == "cuda":
        torch._assert_async(
            (root_handles == root_nodes.to(torch.int64)).all(),
            "branched occurrence and root handles diverged",
        )
    elif not torch.equal(root_handles, root_nodes.to(torch.int64)):
        raise RuntimeError("branched occurrence and root handles diverged")
    root_reach = torch.zeros((1, p_max, x_max), dtype=torch.float32, device=device)
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    reach_pool = RecyclableTensorPool(
        TensorSchema(
            "branched_live_reach",
            (
                TensorFieldSchema(
                    "log_reach",
                    torch.float32,
                    (p_max, x_max),
                    alignment=16,
                ),
            ),
            chunk_header_bytes=64,
            allocator_slack_bytes=64,
        ),
        page_rows,
        device,
        label="branched_live_reach",
        accountant=accountant,
    )
    root_reach_handles = reach_pool.allocate(1)
    reach_pool.write(root_reach_handles, {"log_reach": root_reach})
    arena.scatter_many(
        root_handles,
        {
            "path_key": torch.zeros(1, dtype=torch.int64, device=device),
            "reach_handle": root_reach_handles,
        },
    )

    def reaches_for(occurrences: Tensor) -> Tensor:
        handles = arena.gather_many(
            ("reach_handle",), occurrences.to(torch.int64)
        )["reach_handle"]
        return reach_pool.gather("log_reach", handles)

    def release_reaches(occurrences: Tensor) -> None:
        handles = arena.gather_many(
            ("reach_handle",), occurrences.to(torch.int64)
        )["reach_handle"]
        reach_pool.release_rows(handles)
    family_names, family_values, family_priorities = _family_spec(backend)
    queue_limit = getattr(rollout_cfg, "queue_capacity_rows", None)
    if (
        isinstance(queue_limit, bool)
        or not isinstance(queue_limit, int)
        or queue_limit <= 0
    ):
        raise ValueError("branched rollout requires positive queue_capacity_rows")
    if child_pack_rows > queue_limit:
        raise ValueError("family_batch_rows cannot exceed queue_capacity_rows")
    queues = {
        name: DeviceWorkQueue(
            queue_limit,
            device,
            staging_rows=child_pack_rows,
            accountant=accountant,
            label=f"branched_queue_{name}",
        )
        for name in family_names
    }
    metrics = getattr(backend, "execution_metrics", None)
    actual_cost_forward = getattr(
        backend, "policy_forward_family_actual_cost", None
    )
    if not callable(actual_cost_forward):
        raise ValueError("production requires an actual-cost family forward")
    logical_seed = _logical_seed(cfg, gen)
    tree_address = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    rng_tree_address = tree_address + int(attempt) * world_size
    scheduler_step = 0
    completed_decision_rows = 0
    capture_inputs = getattr(backend, "capture_decision_inputs", None)
    begin_captures = getattr(backend, "begin_decision_input_captures", None)
    reserve_capture = getattr(backend, "reserve_decision_input_captures", None)
    write_capture = getattr(backend, "write_decision_input_capture", None)
    reserve_actions = getattr(backend, "reserve_action_captures", None)
    write_actions = getattr(backend, "write_action_captures", None)
    seal_captures = getattr(backend, "seal_decision_input_captures", None)
    if not all(
        callable(value)
        for value in (
            capture_inputs,
            begin_captures,
            reserve_capture,
            write_capture,
            reserve_actions,
            write_actions,
            seal_captures,
        )
    ):
        raise TypeError(
            "branched rollout backends must capture immutable natural decision inputs"
        )
    leaf_schema_factory = getattr(backend, "rollout_leaf_source_schema", None)
    leaf_schema = (
        leaf_schema_factory("branched_rb")
        if callable(leaf_schema_factory)
        else stopped_leaf_source_schema(
            "generic", players=p_max, exact_width=x_max
        )
    )
    leaf_rows = PagedTensorArena(
        leaf_schema,
        page_rows,
        device,
        accountant=accountant,
        label="branched_stopped_leaf_rows",
    )
    cutoff_active = False
    branching_mode = getattr(rollout_cfg, "branching_mode", None)
    if branching_mode not in {"fixed", "dynamic"}:
        raise ValueError("branched rollout requires branching_mode 'fixed' or 'dynamic'")
    adaptive_cfg = getattr(rollout_cfg, "adaptive_kkt", None)
    dynamic_configured = branching_mode == "dynamic"
    if dynamic_configured and adaptive_cfg is None:
        raise ValueError("dynamic branching requires adaptive_kkt configuration")
    if not dynamic_configured and adaptive_cfg is not None:
        raise ValueError("adaptive_kkt configuration is only valid for dynamic branching")
    dynamic_active = bool(
        dynamic_configured and iteration >= int(adaptive_cfg.warmup_iters)
    )
    if dynamic_active and controller is None:
        raise ValueError("dynamic branching after warmup requires controller state")
    fixed_scale = (
        float(controller.scale)
        if branching_mode == "fixed" and hasattr(controller, "scale")
        else 1.0
    )
    source_model = (
        bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    )
    preflop_model = getattr(source_model, "preflop", None)
    retains_branching_heads = (
        getattr(preflop_model, "branching_heads", None) is not None
    )
    decision_input_store = begin_captures(
        mode="branched_rb",
        dynamic_heads=retains_branching_heads,
        accountant=accountant,
    )
    safe_tree_size = getattr(rollout_cfg, "safe_tree_size", None)
    if safe_tree_size is not None and (
        isinstance(safe_tree_size, bool)
        or not math.isfinite(float(safe_tree_size))
        or float(safe_tree_size) < 1.0
    ):
        raise ValueError("safe_tree_size must be finite and at least one")
    if safe_tree_size is not None and safeguard_controller is None:
        raise ValueError("safe_tree_size requires cost-safeguard controller state")
    if safe_tree_size is None and safeguard_controller is not None:
        raise ValueError(
            "cost-safeguard controller state requires safe_tree_size configuration"
        )
    queue_pressure_active = bool(
        safe_tree_size is not None and safeguard_controller.active
    )
    if safe_tree_size is not None:
        diagnostics.update(
            {
                "safe_tree_configured": 1,
                "safe_tree_active": int(queue_pressure_active),
            }
        )
    control_arena = PagedTensorArena(
        TensorSchema(
            "branched_controller_rows",
            (
                TensorFieldSchema("nominal_count", torch.int32, alignment=4),
                TensorFieldSchema("beta", torch.float32, alignment=4),
                TensorFieldSchema(
                    "log_player_sum", torch.float32, fill=-torch.inf, alignment=4
                ),
                TensorFieldSchema("log_ancestor_product", torch.float32, alignment=4),
                TensorFieldSchema("queue_cost", torch.float32, alignment=4),
            ),
            chunk_header_bytes=64,
            allocator_slack_bytes=64,
        ),
        page_rows,
        device,
        accountant=accountant,
        label="branched_controller_rows",
    )
    nominal_count = PagedTensorColumn(control_arena, "nominal_count")
    beta_rows = PagedTensorColumn(control_arena, "beta")
    log_player_sum = PagedTensorColumn(control_arena, "log_player_sum")
    log_ancestor_product = PagedTensorColumn(
        control_arena, "log_ancestor_product"
    )
    queue_cost = PagedTensorColumn(control_arena, "queue_cost")
    root_is_player = (root_owner >= 0) & (root_owner < p_max)
    control_arena.append(
        {
            "nominal_count": torch.zeros(1, dtype=torch.int32, device=device),
            "beta": torch.zeros(1, dtype=torch.float32, device=device),
            "log_player_sum": torch.where(
                root_is_player,
                torch.zeros(1, dtype=torch.float32, device=device),
                torch.full((1,), -torch.inf, dtype=torch.float32, device=device),
            ),
            "log_ancestor_product": torch.zeros(1, device=device),
            "queue_cost": torch.zeros(1, device=device),
        }
    )
    queue_cost_total = 0.0

    def admit_child_wave(count: int) -> None:
        """Admit every mutable child owner before draws or state mutation."""

        checkpoints = (
            (buffer.nodes, buffer.nodes.page_count),
            (buffer.edges, buffer.edges.page_count),
            (control_arena, control_arena.page_count),
            (reach_pool.arena, reach_pool.arena.page_count),
        )
        occurrence_chunks = arena.chunk_count
        state_arenas = (
            getattr(getattr(backend, "states", None), "arena", None),
            getattr(getattr(backend, "states", None), "history_arena", None),
        )
        state_checkpoints = tuple(
            (owner, owner.page_count) for owner in state_arenas if owner is not None
        )
        try:
            buffer.require_child_capacity(count)
            arena.require_capacity(count)
            control_arena.require_rows(count)
            reach_pool.require_capacity(count)
            require_states = getattr(backend, "require_successor_capacity", None)
            if callable(require_states):
                require_states(count)
        except BaseException:
            for owner, pages in reversed((*checkpoints, *state_checkpoints)):
                owner.rollback_page_growth(pages)
            arena.rollback_chunk_growth(occurrence_chunks)
            raise

    def stop_if_cutoff_reached(frontier_nodes: int = 0) -> None:
        if stop_cutoff is None or buffer.node_count < stop_cutoff:
            return
        family_queue_nodes = {
            name: int(queue.host_count) for name, queue in queues.items()
        }
        raise BranchingCutoffReached(
            node_count=buffer.node_count,
            queued_nodes=sum(family_queue_nodes.values()),
            frontier_nodes=frontier_nodes,
            family_queue_nodes=family_queue_nodes,
            completed_leaves=leaf_rows.size,
            completed_decisions=completed_decision_rows,
        )

    def read_control(value: Tensor) -> tuple[int, ...]:
        if value.ndim != 1 or value.dtype != torch.int64:
            raise ValueError("branched scheduler control must be int64 [K]")
        if device.type == "cuda":
            started = time.perf_counter()
            host = value.to("cpu")
            elapsed = time.perf_counter() - started
            if metrics is not None:
                metrics.record_scheduler_control_synchronization(
                    seconds=elapsed,
                    nbytes=value.numel() * value.element_size(),
                )
        else:
            host = value
        return tuple(int(item) for item in host.tolist())

    def read_float_control(value: Tensor) -> float:
        if value.numel() != 1 or not value.is_floating_point():
            raise ValueError("branched scheduler float control must be one scalar")
        if device.type == "cuda":
            started = time.perf_counter()
            host = value.detach().to("cpu")
            elapsed = time.perf_counter() - started
            if metrics is not None:
                metrics.record_scheduler_control_synchronization(
                    seconds=elapsed, nbytes=value.element_size()
                )
        else:
            host = value.detach()
        return float(host)

    def record_queues() -> None:
        if metrics is None:
            return
        for name, queue in queues.items():
            metrics.record_queue(
                name,
                depth=queue.host_count,
                oldest_age=float(queue.oldest_age(scheduler_step)),
            )

    def forecast_workspace(family: str, rows: int) -> int:
        predictor = getattr(backend, "training_transient_bytes", None)
        if not callable(predictor):
            return 0
        source = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
        amp = bool(getattr(getattr(cfg, "training", None), "amp", False))
        return int(predictor(source, family, max(1, int(rows)), amp=amp))

    def admit(
        phase: str,
        *,
        additional_nodes: int = 0,
        workspace_bytes: int = 0,
        reason: str,
    ) -> None:
        if memory_controller is None:
            return
        predicted = memory_controller.admit(
            phase,
            buffer.node_count,
            additional_units=additional_nodes,
            workspace_bytes=workspace_bytes,
            device=device,
            reason=reason,
        )
        diagnostics["memory_predicted_bytes"] = int(predicted)

    def offspring_for(
        nodes: Tensor,
        family: str | None,
        output: Any | None,
        frontier_cost_forecast: float,
    ) -> tuple[Tensor, Tensor, int]:
        nonlocal cutoff_active
        rows = nodes.to(torch.int64)
        states = buffer.state_ref[rows]
        fixed_beta = _validate_beta(backend.beta_fn(states), nodes.numel(), device)
        if branching_mode == "fixed":
            fixed_beta = (fixed_beta * fixed_scale).clamp_min(1.0)
        if dynamic_active and family is not None:
            if output is None or output.t is None or output.c is None:
                raise RuntimeError("adaptive KKT rollout model omitted T/C heads")
            beta = log_kkt_intensity(
                output.t,
                output.c,
                controller.log_kappa,
                log_player_sum[rows],
                log_ancestor_product[rows],
                float(adaptive_cfg.beta_max),
            )
        else:
            beta = fixed_beta
        if cutoff_active:
            beta = torch.ones_like(beta)
        paths = arena.gather_many(("path_key",), rows)["path_key"]
        keys = logical_rng_keys(
            seed=logical_seed,
            iteration=iteration,
            trajectory=torch.full_like(paths, rng_tree_address),
            path=paths,
            draw=0,
            purpose=DrawPurpose.ROUND,
        )
        nominal = randomized_round(beta, logical_uniform(keys))
        offspring = nominal
        safeguard = bool(
            family is not None
            and queue_pressure_active
            and buffer.node_count + frontier_cost_forecast
            > float(safe_tree_size)
        )
        if safeguard:
            offspring = torch.ones_like(nominal)
        total = read_control(offspring.sum().reshape(1))[0]
        if upper_cutoff is not None and (
            cutoff_active or buffer.node_count + total >= upper_cutoff
        ):
            cutoff_active = True
            proposed_total = total
            offspring = torch.ones_like(offspring)
            total = nodes.numel()
            if diagnostics is not None:
                diagnostics["upper_cutoff_batches"] += 1
                diagnostics["upper_cutoff_parents"] += int(nodes.numel())
                diagnostics["upper_cutoff_avoided_children"] += max(
                    0, proposed_total - total
                )
                if diagnostics["upper_cutoff_first_node_count"] < 0:
                    diagnostics["upper_cutoff_first_node_count"] = buffer.node_count
        if total <= 0:
            raise RuntimeError("branched intensity produced no child")
        admit(
            "rollout",
            additional_nodes=total,
            reason="child-wave owners",
        )
        admit_child_wave(total)
        nominal_count[rows] = nominal.to(torch.int32)
        beta_rows[rows] = beta.detach().to(torch.float32)
        if queue_pressure_active and diagnostics is not None:
            diagnostics["safeguard_batches"] = diagnostics.get("safeguard_batches", 0) + int(safeguard)
            diagnostics["safeguard_suppressed_children"] = diagnostics.get(
                "safeguard_suppressed_children", 0
            ) + (int((nominal - offspring).sum()) if safeguard else 0)
        return offspring, nominal, total

    def expand(
        nodes: Tensor,
        family: str | None,
        frontier_cost_forecast: float = 0.0,
        *,
        forced_offspring: Tensor | None = None,
        forced_edge_weight: float | None = None,
        record_policy_node: bool = True,
    ) -> Tensor:
        nonlocal completed_decision_rows
        rows = nodes.to(torch.int64)
        states = buffer.state_ref[rows]
        reaches = reaches_for(nodes)
        partition_for_family = getattr(backend, "rollout_partition_id", None)
        if family is None:
            partition_id = None
        elif callable(partition_for_family):
            partition_id = int(partition_for_family(family))
        else:
            partition_id = family_names.index(family)
        decision_rows = None
        if partition_id is not None and record_policy_node:
            decision_rows = reserve_capture(
                decision_input_store, partition_id, nodes.numel()
            )
            completed_decision_rows += nodes.numel()
        output = None
        if dynamic_active and family is not None:
            admit(
                "rollout",
                workspace_bytes=forecast_workspace(family, nodes.numel()),
                reason=f"{family} adaptive policy forward",
            )
            output = actual_cost_forward(bundle.model, family, states, reaches)
        if forced_offspring is None:
            offspring, _, total = offspring_for(
                nodes, family, output, frontier_cost_forecast
            )
        else:
            if (
                forced_offspring.shape != nodes.shape
                or forced_offspring.device != device
                or forced_offspring.dtype != torch.int64
                or bool((forced_offspring <= 0).any())
            ):
                raise ValueError(
                    "forced offspring must be positive device-local int64 [B]"
                )
            offspring = forced_offspring
            total = read_control(offspring.sum().reshape(1))[0]
            admit(
                "rollout",
                additional_nodes=total,
                reason="root-refill child wave",
            )
            admit_child_wave(total)
        if family is not None and output is None:
            admit(
                "rollout",
                workspace_bytes=forecast_workspace(family, nodes.numel()),
                reason=f"{family} policy forward",
            )
            output = actual_cost_forward(bundle.model, family, states, reaches)
        action_rows = None
        if partition_id is not None and record_policy_node:
            assert output is not None
            assert decision_rows is not None
            action_rows = reserve_actions(
                decision_input_store, partition_id, decision_rows, offspring
            )
            output_t = getattr(output, "t", None)
            output_c = getattr(output, "c", None)
            write_capture(
                decision_input_store,
                decision_rows,
                nodes.to(torch.int32),
                capture_inputs(family, states, reaches),
                {
                    "node_id": nodes.to(torch.int32),
                    "old_v": output.v.detach().to(torch.float32),
                    **(
                        {}
                        if output_t is None or output_c is None
                        else {
                            "old_t": output_t.detach().to(torch.float32),
                            "old_c": output_c.detach().to(torch.float32),
                        }
                    ),
                },
            )
        prefix = torch.empty_like(offspring)
        prefix[0] = 0
        prefix[1:] = offspring.cumsum(0)[:-1]
        child_ends = prefix + offspring
        prior_children = buffer.child_count[rows].to(torch.int64)
        parent_paths = arena.gather_many(("path_key",), rows)["path_key"]
        if (
            forced_edge_weight is not None
            and not 0.0 < float(forced_edge_weight) <= 1.0
        ):
            raise ValueError("forced edge weight must lie in (0,1]")

        def logical_child_subspan(start: int, end: int) -> tuple[Tensor, Tensor, Tensor]:
            logical = torch.arange(start, end, dtype=torch.int64, device=device)
            parent_positions = torch.searchsorted(child_ends, logical, right=True)
            ordinal = (
                logical
                - prefix[parent_positions]
                + 1
                + prior_children[parent_positions]
            )
            return logical, parent_positions, ordinal

        # Fix every topology, occurrence, reach, controller, and RNG address
        # before the first posterior/event draw.  Only scalar address vectors are
        # live here; dense child state/reach/profile work is staged below.
        first_child = buffer.node_count
        first_edge = buffer.edge_count
        for start in range(0, total, child_pack_rows):
            end = min(start + child_pack_rows, total)
            _, parent_positions, ordinal = logical_child_subspan(start, end)
            parents = nodes[parent_positions].to(torch.int32)
            if forced_edge_weight is None:
                weights = offspring[parent_positions].reciprocal().to(torch.float32)
            else:
                weights = torch.full(
                    (end - start,),
                    float(forced_edge_weight),
                    dtype=torch.float32,
                    device=device,
                )
            children, edge_rows = buffer.reserve_children(parents, weights)
            expected_children = torch.arange(
                first_child + start,
                first_child + end,
                dtype=torch.int32,
                device=device,
            )
            expected_edges = torch.arange(
                first_edge + start,
                first_edge + end,
                dtype=torch.int32,
                device=device,
            )
            if not torch.equal(children, expected_children) or not torch.equal(
                edge_rows, expected_edges
            ):
                raise RuntimeError("branched logical child reservation lost stable order")
            control_rows = control_arena.append(
                {
                    "nominal_count": torch.zeros(
                        end - start, dtype=torch.int32, device=device
                    ),
                    "beta": torch.zeros(
                        end - start, dtype=torch.float32, device=device
                    ),
                    "log_player_sum": torch.full(
                        (end - start,), -torch.inf, dtype=torch.float32, device=device
                    ),
                    "log_ancestor_product": torch.zeros(end - start, device=device),
                    "queue_cost": torch.zeros(end - start, device=device),
                }
            )
            if not torch.equal(control_rows.to(torch.int32), children):
                raise RuntimeError("branched controller rows lost topology alignment")
            child_handles = arena.allocate(end - start)
            if device.type == "cuda":
                torch._assert_async(
                    (child_handles == children.to(torch.int64)).all(),
                    "branched occurrence and child handles diverged",
                )
            elif not torch.equal(child_handles, children.to(torch.int64)):
                raise RuntimeError("branched occurrence and child handles diverged")
            child_reach_handles = reach_pool.allocate(end - start)
            expanded_paths = parent_paths[parent_positions]
            arena.scatter_many(
                child_handles,
                {
                    "path_key": child_path_keys(expanded_paths, ordinal),
                    "reach_handle": child_reach_handles,
                },
            )

        for start in range(0, total, child_pack_rows):
            end = min(start + child_pack_rows, total)
            _, parent_positions, ordinal = logical_child_subspan(start, end)
            children = torch.arange(
                first_child + start,
                first_child + end,
                dtype=torch.int32,
                device=device,
            )
            edge_rows = torch.arange(
                first_edge + start,
                first_edge + end,
                dtype=torch.int32,
                device=device,
            )
            child_states = states[parent_positions]
            child_reach = reaches[parent_positions]
            expanded_paths = parent_paths[parent_positions]
            posterior_keys = logical_rng_keys(
                seed=logical_seed,
                iteration=iteration,
                trajectory=torch.full_like(expanded_paths, rng_tree_address),
                path=expanded_paths,
                draw=ordinal,
                purpose=DrawPurpose.POSTERIOR,
            )
            profiles = _sample_profiles_logical(
                backend, child_states, child_reach, posterior_keys, gen
            )
            child_output = None
            if family is not None:
                assert output is not None
                child_output = _index_policy_output(output, parent_positions)
            event_keys = logical_rng_keys(
                seed=logical_seed,
                iteration=iteration,
                trajectory=torch.full_like(expanded_paths, rng_tree_address),
                path=expanded_paths,
                draw=ordinal,
                purpose=(
                    DrawPurpose.ACTION if family is not None else DrawPurpose.CHANCE
                ),
            )
            events = _sample_events_logical(
                backend,
                child_states,
                profiles,
                child_output,
                bundle.policy_adapter,
                event_keys,
                gen,
            )
            if family is None:
                step_batch = getattr(backend, "step_chance_batch", backend.step)
            else:
                step_batch = getattr(backend, "step_player_batch", backend.step)
            next_states = step_batch(child_states, events)
            next_reach = backend.reach_update(
                child_reach, child_states, events, child_output
            )
            next_owner = backend.classify(next_states).to(torch.int32)
            buffer.commit_children(
                children, edge_rows, next_states, next_owner, events.public_id
            )
            child_reach_handles = arena.gather_many(
                ("reach_handle",), children.to(torch.int64)
            )["reach_handle"]
            reach_pool.write(child_reach_handles, {"log_reach": next_reach})
            child_log_sum, child_log_product = child_path_factors(
                log_player_sum[rows][parent_positions],
                log_ancestor_product[rows][parent_positions],
                beta_rows[rows][parent_positions],
                (next_owner >= 0) & (next_owner < p_max),
            )
            log_player_sum[children.to(torch.int64)] = child_log_sum
            log_ancestor_product[children.to(torch.int64)] = child_log_product
            if family is not None and record_policy_node:
                assert child_output is not None
                legal_ids, legal_mask = backend.legal_events(child_states)
                edge_logp = _selected_policy_logp(
                    bundle.policy_adapter,
                    child_output,
                    events.public_id[:, None],
                    legal_ids,
                    legal_mask,
                ).detach()[:, 0]
                assert partition_id is not None and action_rows is not None
                write_actions(
                    decision_input_store,
                    partition_id,
                    action_rows[start:end],
                    {
                        "event": events.public_id.to(torch.int32),
                        "old_logp": edge_logp.to(torch.float32),
                    },
                )
        children = torch.arange(
            first_child,
            first_child + total,
            dtype=torch.int32,
            device=device,
        )
        if family is not None and record_policy_node:
            DeviceWorkQueue.mark_done(arena, rows)
        releasable = nodes[nodes != 0]
        if releasable.numel():
            release_reaches(releasable)
            release_states = getattr(backend, "release_state_rows", None)
            if callable(release_states):
                released_rows = releasable.to(torch.int64)
                release_states(buffer.state_ref[released_rows].to(torch.int32))
                buffer.state_ref[released_rows] = -1
        return children

    def chance_close(frontier: Tensor) -> None:
        nonlocal scheduler_step, queue_cost_total
        frontier = frontier.to(torch.int32)
        chance_steps = 0
        while frontier.numel():
            stop_if_cutoff_reached(int(frontier.numel()))
            scheduler_step += 1
            rows = frontier.to(torch.int64)
            owners = buffer.owner[rows]
            states = buffer.state_ref[rows]
            family_ids = _family_ids(backend, states)
            chance_route, leaf_route, invalid_route = -1, -2, -3
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
            partition = stable_partition_masked(
                frontier.to(torch.int64),
                route,
                (*family_values, chance_route, leaf_route),
            )
            counts = read_control(partition.selected_count)
            for index, (name, count) in enumerate(
                zip(family_names, counts[: len(family_names)], strict=True)
            ):
                if count:
                    queued = partition.candidate_handles[index, :count]
                    queues[name].enqueue(
                        queued,
                        arena=arena,
                        logical_time=scheduler_step,
                    )
                    if queue_pressure_active:
                        qrows = queued.to(torch.int64)
                        qstates = buffer.state_ref[qrows]
                        admit(
                            "rollout",
                            workspace_bytes=forecast_workspace(name, count),
                            reason=f"{name} C-head queue forecast",
                        )
                        qout = actual_cost_forward(
                            bundle.model, name, qstates, reaches_for(queued)
                        )
                        if qout.c is None:
                            raise RuntimeError("queue-pressure accounting requires C head")
                        queue_cost[qrows] = qout.c.detach().float()
                        queue_cost_total += read_float_control(
                            qout.c.detach().float().sum()
                        )
            record_queues()
            stopped = counts[-1]
            if stopped:
                stopped_nodes = partition.candidate_handles[-1, :stopped].to(torch.int32)
                stopped_rows = stopped_nodes.to(torch.int64)
                stopped_states = buffer.state_ref[stopped_rows].to(torch.int32)
                stopped_reach = reaches_for(stopped_nodes).detach().clone()
                capture = getattr(backend, "capture_rb_leaf", None)
                payload = None
                if callable(capture):
                    payload = capture(stopped_states, stopped_reach)
                    release_states = getattr(backend, "release_state_rows", None)
                    if callable(release_states):
                        release_states(stopped_states)
                        buffer.state_ref[stopped_rows] = -1
                leaf_rows.append(
                    {
                        "node_id": stopped_nodes,
                        "root_id": torch.full(
                            (stopped,), int(tree_address), dtype=torch.int32, device=device
                        ),
                        "path_key": arena.gather_many(
                            ("path_key",), stopped_rows
                        )["path_key"].to(torch.int64),
                        "log_reach": stopped_reach,
                        **({} if payload is None else payload),
                    }
                )
                release_reaches(stopped_nodes)
            chance_count = counts[-2]
            if chance_count == 0:
                return
            chance_steps += 1
            if chance_steps > int(backend.max_rollout_steps):
                raise RuntimeError("chance-only segment exceeded max_rollout_steps")
            chance_nodes = partition.candidate_handles[-2, :chance_count].to(torch.int32)
            frontier = expand(chance_nodes, None)

    def drain_player_queues() -> None:
        nonlocal scheduler_step, queue_cost_total
        while any(queue.host_count for queue in queues.values()):
            stop_if_cutoff_reached()
            scheduler_step += 1
            selection = select_work_queue(
                family_names,
                family_priorities,
                queues,
                child_pack_rows,
                allow_partial=True,
                logical_time=scheduler_step,
            )
            if selection is None:
                raise RuntimeError("branched queue scheduler lost nonempty work")
            family = selection.family
            queue = queues[family]
            count = selection.rows
            frontier_cost_forecast = queue_cost_total
            nodes = queue.dequeue(count, arena=arena).to(torch.int32)
            if queue_pressure_active:
                # This synchronization is co-located with the scheduler's existing
                # host control boundary and keeps R_F incremental.
                removed = read_float_control(queue_cost[nodes.to(torch.int64)].sum())
                queue_cost_total -= removed
            record_queues()
            chance_close(expand(nodes, family, frontier_cost_forecast))

    chance_close(root_nodes)
    drain_player_queues()

    pre_refill_node_count: int | None = None
    root_pre_refill_child_count: int | None = None
    if minimum_nodes is not None:
        pre_refill_node_count = buffer.node_count
        root_pre_refill_child_count = int(buffer.child_count[0])
        if root_pre_refill_child_count <= 0:
            raise RuntimeError("a refillable branched root must have an ordinary child")
        root_owner_value = int(root_owner[0])
        if root_owner_value == CHANCE:
            root_family = None
        elif 0 <= root_owner_value < p_max:
            root_family_value = read_control(
                _family_ids(backend, roots).reshape(1)
            )[0]
            try:
                root_family = family_names[family_values.index(root_family_value)]
            except ValueError as error:
                raise RuntimeError("branched root has an unknown execution family") from error
        else:
            raise RuntimeError("branched root must be expandable for root refill")

        refill_count = 0
        while buffer.node_count < minimum_nodes:
            refill_count += 1
            extra = expand(
                root_nodes,
                root_family,
                forced_offspring=torch.ones(1, dtype=torch.int64, device=device),
                forced_edge_weight=1.0 / root_pre_refill_child_count,
                record_policy_node=False,
            )
            chance_close(extra)
            drain_player_queues()
        diagnostics.update(
            {
                "root_refill_count": refill_count,
                "pre_refill_nodes": pre_refill_node_count,
                "root_refill_added_nodes": buffer.node_count - pre_refill_node_count,
                "root_pre_refill_child_count": root_pre_refill_child_count,
                "root_final_child_count": int(buffer.child_count[0]),
            }
        )

    for queue in queues.values():
        queue.release()

    release_reaches(root_nodes)
    release_states = getattr(backend, "release_state_rows", None)
    if callable(release_states):
        release_states(buffer.state_ref[root_nodes.to(torch.int64)].to(torch.int32))
        buffer.state_ref[root_nodes.to(torch.int64)] = -1
    if reach_pool.live_rows != 0:
        raise RuntimeError("branched live reach slots survived their last consumer")
    reach_pool.release()
    node_count = buffer.node_count
    admit("seal", reason="exact branched rollout sealing")
    all_nodes = torch.arange(node_count, device=device)
    old_parent = buffer.parent[all_nodes].to(torch.int64)
    old_order = torch.empty(node_count, dtype=torch.int64, device=device)
    old_order[0] = 0
    old_order[1:] = torch.argsort(old_parent[1:], stable=True) + 1
    old_to_new = torch.empty(node_count, dtype=torch.int64, device=device)
    old_to_new[old_order] = torch.arange(node_count, device=device)
    parent = old_parent[old_order]
    parent = torch.where(parent < 0, parent, old_to_new[parent]).to(torch.int32)
    degree_old = buffer.child_count[all_nodes].to(torch.int32)
    degree = degree_old[old_order]
    child_begin = torch.full((node_count,), -1, dtype=torch.int32, device=device)
    if node_count > 1:
        child_ids = torch.arange(1, node_count, dtype=torch.int32, device=device)
        first = torch.full((node_count,), node_count, dtype=torch.int32, device=device)
        first.scatter_reduce_(
            0, parent[1:].to(torch.int64), child_ids, reduce="amin", include_self=True
        )
        child_begin[degree > 0] = first[degree > 0]

    if not leaf_rows.size:
        raise RuntimeError("branched rollout produced no stopped-leaf rows")
    source_nodes = leaf_rows.gather(
        "node_id", torch.arange(leaf_rows.size, device=device)
    )
    leaf_order = torch.argsort(source_nodes.to(torch.int64), stable=True)
    leaf_block, leaf_metrics = leaf_rows.seal(
        scratch_rows=page_rows, row_order=leaf_order, release_source=True
    )
    verify_recycling = getattr(backend, "verify_state_recycling_boundary", None)
    if callable(verify_recycling):
        verify_recycling()
    decision_inputs = seal_captures(decision_input_store)
    decision_nodes = decision_inputs.node_ids.to(torch.int64)
    decision_new_nodes = old_to_new[decision_nodes]
    decision_partition = torch.full(
        (node_count,), -1, dtype=torch.int32, device=device
    )
    decision_row = torch.full_like(decision_partition, -1)
    decision_partition[decision_new_nodes] = decision_inputs.partition_id
    decision_row[decision_new_nodes] = decision_inputs.partition_row
    leaf_old = leaf_block["node_id"].to(torch.int64)
    leaf_new = old_to_new[leaf_old]
    leaf_row = torch.full((node_count,), -1, dtype=torch.int32, device=device)
    leaf_row[leaf_new] = torch.arange(
        leaf_block.rows, dtype=torch.int32, device=device
    )
    nominal_old = nominal_count[:node_count].detach().to(torch.int32)
    topology = BranchedTreeSpan(
        parent=parent,
        child_begin=child_begin,
        degree=degree,
        decision_partition=decision_partition,
        decision_row=decision_row,
        leaf_row=leaf_row,
        root_target_degree=root_pre_refill_child_count,
        pre_refill_node_count=pre_refill_node_count,
        nominal_degree=nominal_old[old_order],
    )
    topology.validate(
        partition_rows={
            int(partition): block.rows
            for partition, block in decision_inputs.inputs.items()
        },
        leaf_rows=leaf_block.rows,
    )

    if not decision_nodes.numel():
        raise RuntimeError("branched rollout produced no player decisions")
    target_counts = degree_old[decision_nodes].to(torch.int64)
    if root_pre_refill_child_count is not None:
        root_row = (decision_nodes == 0).nonzero(as_tuple=False).squeeze(1)
        if root_row.numel() != 1:
            raise RuntimeError("protected root is not a retained decision")
        target_counts[root_row] = int(root_pre_refill_child_count)
    for partition_id, ptr in decision_inputs.action_ptrs.items():
        selected = decision_inputs.partition_id == partition_id
        if not torch.equal(
            (ptr[1:] - ptr[:-1]).to(torch.int64), target_counts[selected]
        ):
            raise RuntimeError("reserved natural action rows disagree with topology")
    path_weight = buffer.path_weight[decision_nodes].detach().to(torch.float64)
    if root_pre_refill_child_count is not None:
        root_final_child_count = int(degree_old[0])
        path_weight.mul_(root_pre_refill_child_count / root_final_child_count)
        path_weight[decision_nodes == 0] = 1.0
    decision_aux = {
        "owner": buffer.owner[decision_nodes].detach().to(torch.int32),
        "log_weight": torch.log(path_weight),
        "nominal_child_count": nominal_old[decision_nodes],
        "executed_child_count": target_counts.to(torch.int32),
    }
    decision_beta = beta_rows[decision_nodes].detach().to(torch.float32)
    sealed_diagnostics = {
        **diagnostics,
        "branching_attempt": int(attempt),
        "fixed_beta_scale": fixed_scale,
        "execution_accountant_limit_bytes": accountant.limit_bytes,
        "execution_accountant_high_water_bytes": accountant.high_water_bytes,
    }
    arena.release()
    buffer.release()
    control_arena.release()
    return SealedRolloutSample(
        mode="branched_rb",
        topology=topology,
        leaf_source=leaf_block,
        decision_inputs=decision_inputs,
        node_count=node_count,
        decision_node=decision_new_nodes.to(torch.int32),
        decision_aux=decision_aux,
        branching_beta=decision_beta,
        branching_diagnostics=sealed_diagnostics,
        seal_metrics={
            "leaf_rows": leaf_metrics,
            **decision_inputs.seal_metrics,
        },
    )



def _validate_beta(beta: Tensor, count: int, device: torch.device) -> Tensor:
    """Validate and return fixed intensities as fp32 ``[B]``."""

    if beta.shape != (count,) or beta.dtype != torch.float32 or beta.device != device:
        raise ValueError("beta_fn must return device-local fp32 [B]")
    if not bool(torch.isfinite(beta).all()) or bool((beta < 1.0).any()):
        raise ValueError("rollout beta must be finite and at least one")
    return beta



def _index_policy_output(output: Any, rows: Tensor) -> Any:
    """Gather batch rows from a lightweight, tensor-structured policy output."""

    if isinstance(output, Tensor):
        return output[rows]
    if is_dataclass(output) and not isinstance(output, type):
        return type(output)(
            **{
                field.name: _index_policy_output(getattr(output, field.name), rows)
                for field in fields(output)
            }
        )
    if isinstance(output, Mapping):
        return type(output)(
            (key, _index_policy_output(value, rows)) for key, value in output.items()
        )
    if isinstance(output, tuple):
        return type(output)(_index_policy_output(value, rows) for value in output)
    if isinstance(output, list):
        return [_index_policy_output(value, rows) for value in output]
    # Immutable metadata may be shared by all gathered rows.
    return output
