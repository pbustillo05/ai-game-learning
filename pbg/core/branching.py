"""Fixed or post-warmup dynamic duplicate-preserving labeled branching."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL, GameBundle
from pbg.core.adaptive_branching import (
    child_path_factors,
    log_kkt_intensity,
    randomized_round,
)
from pbg.core.tree import (
    DecisionReachCache,
    DeviceTreeBuffer,
    PolicyCache,
    ReachCache,
    TreeBatch,
)


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
        state_refs: Tensor,
        owners: Tensor,
    ) -> None:
        self.node_count = node_count
        self.queued_nodes = queued_nodes
        self.frontier_nodes = frontier_nodes
        self.pending_nodes = queued_nodes + frontier_nodes
        self.family_queue_nodes = dict(family_queue_nodes)
        self.completed_leaves = completed_leaves
        self.completed_decisions = completed_decisions
        self.state_refs = state_refs
        self.owners = owners
        super().__init__(
            f"branched rollout stopped at {node_count} nodes with "
            f"{self.pending_nodes} live nodes pending"
        )


def sample_branched_tree(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    *,
    iteration: int = 0,
    diagnostics: dict[str, int] | None = None,
    controller: Any | None = None,
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Sample one duplicate-preserving branched tree through execution families."""

    if diagnostics is None:
        diagnostics = {}

    # Imports are local because rollout dispatch imports this module lazily.
    import time
    import torch.distributed as dist

    from pbg.core.execution import (
        DONE,
        ChunkedOccurrenceArena,
        DeviceWorkQueue,
        DrawPurpose,
        arena_fields,
        child_path_keys,
        logical_rng_keys,
        logical_uniform,
        stable_partition_masked,
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
    if int(rollout_cfg.roots_per_gpu) != 1 or int(rollout_cfg.node_budget) <= 0:
        raise ValueError("family-scheduled branched rollout requires one root and a positive budget")

    p_max = int(backend.n_players)
    x_sizes = tuple(int(value) for value in backend.x_sizes)
    x_max = max(x_sizes)
    node_budget = int(rollout_cfg.node_budget)
    upper_cutoff = getattr(rollout_cfg, "branching_upper_cutoff_nodes", None)
    if upper_cutoff is not None:
        if (
            isinstance(upper_cutoff, bool)
            or not isinstance(upper_cutoff, int)
            or not 1 < upper_cutoff < node_budget
        ):
            raise ValueError(
                "branching_upper_cutoff_nodes must be an integer strictly between "
                "one and node_budget"
            )
    stop_cutoff = getattr(rollout_cfg, "branching_stop_cutoff_nodes", None)
    if stop_cutoff is not None:
        if (
            isinstance(stop_cutoff, bool)
            or not isinstance(stop_cutoff, int)
            or not 1 < stop_cutoff < node_budget
        ):
            raise ValueError(
                "branching_stop_cutoff_nodes must be an integer strictly between "
                "one and node_budget"
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
    buffer = DeviceTreeBuffer(1, node_budget, p_max, device)
    root_nodes = buffer.add_roots(roots, root_owner)
    arena = ChunkedOccurrenceArena(
        node_budget,
        int(getattr(rollout_cfg, "arena_chunk_nodes", None) or min(node_budget, 256)),
        device,
        fields=arena_fields(("path_key",)),
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
    arena.scatter_many(
        root_handles,
        {"path_key": torch.zeros(1, dtype=torch.int64, device=device)},
    )

    root_reach = torch.zeros((1, p_max, x_max), dtype=torch.float32, device=device)
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    reach_rows = torch.empty(
        (node_budget, p_max, x_max), dtype=torch.float32, device=device
    )
    reach_rows[0] = root_reach[0]
    family_names, family_values, family_priorities = _family_spec(backend)
    queues = {name: DeviceWorkQueue(node_budget, device) for name in family_names}
    metrics = getattr(backend, "execution_metrics", None)
    actual_cost_forward = getattr(
        backend, "policy_forward_family_actual_cost", None
    )
    if not callable(actual_cost_forward):
        raise ValueError("production requires an actual-cost family forward")
    logical_seed = _logical_seed(cfg, gen)
    tree_address = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    scheduler_step = 0
    policy_edge_ids: list[Tensor] = []
    policy_edge_logp: list[Tensor] = []
    policy_node_ids: list[Tensor] = []
    policy_node_values: list[Tensor] = []
    policy_node_t: list[Tensor] = []
    policy_node_c: list[Tensor] = []
    policy_node_reaches: list[Tensor] = []
    leaf_node_ids: list[Tensor] = []
    leaf_reaches: list[Tensor] = []
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
    nominal_count = torch.zeros(node_budget, dtype=torch.int32, device=device)
    beta_rows = torch.zeros(node_budget, dtype=torch.float32, device=device)
    log_player_sum = torch.full(
        (node_budget,), -torch.inf, dtype=torch.float32, device=device
    )
    log_ancestor_product = torch.zeros(
        node_budget, dtype=torch.float32, device=device
    )
    root_is_player = (root_owner >= 0) & (root_owner < p_max)
    log_player_sum[0] = torch.where(
        root_is_player[0], torch.tensor(0.0, device=device),
        torch.tensor(-torch.inf, device=device),
    )
    queue_cost = torch.zeros(node_budget, dtype=torch.float32, device=device)
    queue_cost_total = 0.0

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
            completed_leaves=sum(int(nodes.numel()) for nodes in leaf_node_ids),
            completed_decisions=sum(int(nodes.numel()) for nodes in policy_node_ids),
            state_refs=buffer.state_ref[: buffer.node_count].detach().clone(),
            owners=buffer.owner[: buffer.node_count].detach().clone(),
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
            trajectory=torch.full_like(paths, tree_address),
            path=paths,
            draw=0,
            purpose=DrawPurpose.ROUND,
        )
        nominal = randomized_round(beta, logical_uniform(keys))
        offspring = nominal
        safeguard = bool(
            dynamic_active
            and family is not None
            and buffer.node_count + frontier_cost_forecast
            > float(adaptive_cfg.safe_tree_size)
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
        buffer.require_child_capacity(total)
        arena.require_capacity(total)
        nominal_count[rows] = nominal.to(torch.int32)
        beta_rows[rows] = beta.detach().to(torch.float32)
        if dynamic_active and diagnostics is not None:
            diagnostics["safeguard_batches"] = diagnostics.get("safeguard_batches", 0) + int(safeguard)
            diagnostics["safeguard_suppressed_children"] = diagnostics.get(
                "safeguard_suppressed_children", 0
            ) + (int((nominal - offspring).sum()) if safeguard else 0)
        return offspring, nominal, total

    def expand(
        nodes: Tensor, family: str | None, frontier_cost_forecast: float = 0.0
    ) -> Tensor:
        rows = nodes.to(torch.int64)
        states = buffer.state_ref[rows]
        reaches = reach_rows[rows]
        output = None
        if dynamic_active and family is not None:
            output = actual_cost_forward(bundle.model, family, states, reaches)
        offspring, _, total = offspring_for(
            nodes, family, output, frontier_cost_forecast
        )
        if family is not None and output is None:
            output = actual_cost_forward(bundle.model, family, states, reaches)
        parent_positions = torch.repeat_interleave(
            torch.arange(nodes.numel(), device=device),
            offspring,
            output_size=total,
        )
        parents = nodes[parent_positions].to(torch.int32)
        child_states = states[parent_positions]
        child_reach = reaches[parent_positions]
        prefix = torch.cat(
            (torch.zeros(1, dtype=torch.int64, device=device), offspring.cumsum(0)[:-1])
        )
        ordinal = torch.arange(total, device=device) - prefix[parent_positions] + 1
        parent_paths = arena.gather_many(("path_key",), rows)["path_key"]
        expanded_paths = parent_paths[parent_positions]
        posterior_keys = logical_rng_keys(
            seed=logical_seed,
            iteration=iteration,
            trajectory=torch.full_like(expanded_paths, tree_address),
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
            policy_node_reaches.append(reaches.detach().clone())
        event_keys = logical_rng_keys(
            seed=logical_seed,
            iteration=iteration,
            trajectory=torch.full_like(expanded_paths, tree_address),
            path=expanded_paths,
            draw=ordinal,
            purpose=(DrawPurpose.ACTION if family is not None else DrawPurpose.CHANCE),
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
        first_edge = buffer.edge_count
        weights = offspring[parent_positions].reciprocal().to(torch.float32)
        children = buffer.append_children(
            parents, next_states, next_owner, events.public_id, weights
        )
        child_handles = arena.allocate(total)
        if device.type == "cuda":
            torch._assert_async(
                (child_handles == children.to(torch.int64)).all(),
                "branched occurrence and child handles diverged",
            )
        elif not torch.equal(child_handles, children.to(torch.int64)):
            raise RuntimeError("branched occurrence and child handles diverged")
        arena.scatter_many(
            child_handles,
            {"path_key": child_path_keys(expanded_paths, ordinal)},
        )
        reach_rows[children.to(torch.int64)] = next_reach
        child_log_sum, child_log_product = child_path_factors(
            log_player_sum[rows][parent_positions],
            log_ancestor_product[rows][parent_positions],
            beta_rows[rows][parent_positions],
            (next_owner >= 0) & (next_owner < p_max),
        )
        log_player_sum[children.to(torch.int64)] = child_log_sum
        log_ancestor_product[children.to(torch.int64)] = child_log_product
        if family is not None:
            assert output is not None and child_output is not None
            legal_ids, legal_mask = backend.legal_events(child_states)
            edge_logp = _selected_policy_logp(
                bundle.policy_adapter,
                child_output,
                events.public_id[:, None],
                legal_ids,
                legal_mask,
            ).detach()[:, 0]
            policy_edge_ids.append(
                torch.arange(
                    first_edge,
                    first_edge + total,
                    dtype=torch.int64,
                    device=device,
                )
            )
            policy_edge_logp.append(edge_logp)
            policy_node_ids.append(nodes.to(torch.int32))
            policy_node_values.append(output.v.detach().float())
            output_t = getattr(output, "t", None)
            output_c = getattr(output, "c", None)
            if output_t is not None and output_c is not None:
                policy_node_t.append(output_t.detach().float())
                policy_node_c.append(output_c.detach().float())
            DeviceWorkQueue.mark_done(arena, rows)
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
                    if dynamic_active:
                        qrows = queued.to(torch.int64)
                        qstates = buffer.state_ref[qrows]
                        qout = actual_cost_forward(
                            bundle.model, name, qstates, reach_rows[qrows]
                        )
                        if qout.c is None:
                            raise RuntimeError("adaptive KKT queue accounting requires C head")
                        queue_cost[qrows] = qout.c.detach().float()
                        queue_cost_total += read_float_control(
                            qout.c.detach().float().sum()
                        )
            record_queues()
            stopped = counts[-1]
            if stopped:
                stopped_nodes = partition.candidate_handles[-1, :stopped].to(torch.int32)
                leaf_node_ids.append(stopped_nodes)
                leaf_reaches.append(
                    reach_rows[stopped_nodes.to(torch.int64)].detach().clone()
                )
            chance_count = counts[-2]
            if chance_count == 0:
                return
            chance_steps += 1
            if chance_steps > int(backend.max_rollout_steps):
                raise RuntimeError("chance-only segment exceeded max_rollout_steps")
            chance_nodes = partition.candidate_handles[-2, :chance_count].to(torch.int32)
            frontier = expand(chance_nodes, None)

    chance_close(root_nodes)
    progress_limit = node_budget * (int(backend.max_rollout_steps) + 2)
    while any(queue.host_count for queue in queues.values()):
        stop_if_cutoff_reached()
        scheduler_step += 1
        if scheduler_step > progress_limit:
            raise RuntimeError("family-scheduled branched scheduler exceeded its progress bound")
        nonempty = [name for name in family_names if queues[name].host_count]
        upstream = min(family_priorities[family_names.index(name)] for name in nonempty)
        eligible = [
            name
            for name in nonempty
            if family_priorities[family_names.index(name)] == upstream
        ]
        family = max(
            eligible,
            key=lambda name: (
                queues[name].host_count,
                queues[name].oldest_age(scheduler_step),
                -family_names.index(name),
            ),
        )
        queue = queues[family]
        configured_rows = int(
            getattr(rollout_cfg, "family_batch_rows", None) or queue.host_count
        )
        count = min(queue.host_count, configured_rows)
        frontier_cost_forecast = queue_cost_total
        nodes = queue.dequeue(count, arena=arena).to(torch.int32)
        if dynamic_active:
            # This synchronization is co-located with the scheduler's existing
            # host control boundary and keeps R_F incremental.
            removed = read_float_control(
                queue_cost[nodes.to(torch.int64)].sum()
            )
            queue_cost_total -= removed
        record_queues()
        chance_close(expand(nodes, family, frontier_cost_forecast))

    tree = buffer.finalize()
    tree.nominal_child_count = nominal_count[: buffer.node_count].detach()
    tree.branching_beta = beta_rows[: buffer.node_count].detach()
    tree.branching_diagnostics = dict(diagnostics)
    tree.validate()
    if leaf_node_ids:
        reach_nodes = torch.cat(leaf_node_ids).to(torch.int32)
        retained_leaf_reach = torch.cat(leaf_reaches).float()
        leaf_order = torch.argsort(reach_nodes.to(torch.int64))
        reach_nodes = reach_nodes[leaf_order]
        retained_leaf_reach = retained_leaf_reach[leaf_order]
    else:
        reach_nodes = torch.empty(0, dtype=torch.int32, device=device)
        retained_leaf_reach = torch.empty(
            (0, p_max, x_max), dtype=torch.float32, device=device
        )
    reach_cache = ReachCache("sealed", retained_leaf_reach, reach_nodes)
    decision_nodes = ((tree.owner >= 0) & (tree.owner < p_max)).nonzero().squeeze(1)
    if decision_nodes.numel():
        counts = tree.child_ptr[decision_nodes + 1] - tree.child_ptr[decision_nodes]
        width = int(counts.max())
        logp_child = torch.full(
            (decision_nodes.numel(), width, x_max),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        edge_ids = torch.cat(policy_edge_ids).to(torch.int64)
        edge_logp = torch.cat(policy_edge_logp).to(torch.float32)
        parents = buffer.edge_parent[edge_ids].to(torch.int64)
        node_to_row = torch.full(
            (buffer.node_count,), -1, dtype=torch.int64, device=device
        )
        node_to_row[decision_nodes] = torch.arange(decision_nodes.numel(), device=device)
        # Append-order edge ids are not CSR positions under work queue scheduling:
        # another family may expand a later node first. DeviceTreeBuffer stores
        # each edge's stable parent-local ordinal explicitly for this mapping.
        columns = buffer.edge_ordinal[edge_ids].to(torch.int64)
        logp_child[node_to_row[parents], columns] = edge_logp
        cached_nodes = torch.cat(policy_node_ids).to(torch.int32)
        cached_values = torch.cat(policy_node_values).to(torch.float32)
        cached_reaches = torch.cat(policy_node_reaches).to(torch.float32)
        order = torch.argsort(cached_nodes.to(torch.int64))
        if not torch.equal(cached_nodes[order], decision_nodes):
            raise RuntimeError("family-scheduled branched rollout omitted a decision cache row")
        old_v = cached_values[order]
        old_t = torch.cat(policy_node_t).to(torch.float32)[order] if policy_node_t else None
        old_c = torch.cat(policy_node_c).to(torch.float32)[order] if policy_node_c else None
        decision_rows = cached_reaches[order]
    else:
        logp_child = torch.empty((0, 0, x_max), dtype=torch.float32, device=device)
        old_v = torch.empty((0, x_max), dtype=torch.float32, device=device)
        old_t = old_c = None
        decision_rows = torch.empty(
            (0, p_max, x_max), dtype=torch.float32, device=device
        )
    state_refs = tree.state_ref[decision_nodes].to(torch.int32)
    owners = tree.owner[decision_nodes].to(torch.int32)
    decision_reach = DecisionReachCache.from_rows(
        decision_nodes.to(torch.int32),
        state_refs,
        decision_rows,
        backend,
        owners,
    )
    return tree, reach_cache, PolicyCache(
        decision_nodes.to(torch.int32), logp_child, old_v, decision_reach,
        old_t, old_c,
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
