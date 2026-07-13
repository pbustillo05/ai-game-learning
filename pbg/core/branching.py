"""Fixed-intensity labeled branching and private legacy P09 support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor

from pbg.core.config import RolloutConfig
from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch, GameBundle
from pbg.core.tree import (
    DeviceTreeBuffer,
    PolicyCache,
    ReachCache,
    TreeBatch,
    TreeBuilder,
    _replay_reach_update,
)


def _combine_event_rows(
    size: int,
    parts: list[tuple[Tensor, EventBatch]],
    device: torch.device,
) -> EventBatch:
    """Scatter partitioned event batches into abstract-child order ``[C]``."""

    public_id = torch.empty(size, dtype=torch.int32, device=device)  # [C]
    payload: dict[str, Tensor] = {}
    for positions, events in parts:
        public_id[positions] = events.public_id
        for key, value in events.payload.items():
            if key not in payload:
                payload[key] = torch.zeros(
                    (size, *value.shape[1:]), dtype=value.dtype, device=device
                )  # [C,...]
            payload[key][positions] = value
    return EventBatch(public_id=public_id, payload=payload)


def _validate_beta(beta: Tensor, count: int, device: torch.device) -> Tensor:
    """Validate and return fixed intensities as fp32 ``[B]``."""

    if beta.shape != (count,) or beta.dtype != torch.float32 or beta.device != device:
        raise ValueError("beta_fn must return device-local fp32 [B]")
    if not bool(torch.isfinite(beta).all()) or bool((beta < 1.0).any()):
        raise ValueError("rollout beta must be finite and at least one")
    return beta


def _offspring_counts(beta: Tensor, gen: torch.Generator) -> Tensor:
    """Randomized-round fp32 intensities to int64 offspring counts ``[B]``."""

    base = torch.floor(beta)
    fraction = beta - base
    # Integer intensities are deterministic and consume no unnecessary random draw.
    fractional = fraction > 0
    extra = torch.zeros_like(base)
    if bool(fractional.any()):
        extra[fractional] = torch.bernoulli(fraction[fractional], generator=gen)
    return (base + extra).to(torch.int64)  # [B]


def sample_branched_labeled(
    bundle: GameBundle,
    cfg: RolloutConfig,
    gen: torch.Generator,
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Sample fixed-intensity labeled public trees on the backend device.

    The returned tree contains ``N`` nodes and ``C`` duplicate-preserving edges;
    reach rows are ``[N_stored,P,X_max]`` and cached decision rows are
    ``[N_d,K_max,X_max]``.
    """

    backend = bundle.rollout_backend
    rollout_cfg = getattr(cfg, "rollout", cfg)
    device = torch.device(backend.device)
    if torch.device(gen.device) != device:
        raise ValueError("rollout generator must be created on the backend device")
    beta_mode = getattr(rollout_cfg, "beta_mode", None)
    if beta_mode not in {"sequential", "fixed"}:
        raise ValueError("beta_mode must be 'sequential' or 'fixed'")
    if rollout_cfg.reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ValueError("unknown reach_storage mode")
    n_trees = int(rollout_cfg.n_trees)
    node_budget = int(rollout_cfg.node_budget)
    if n_trees < 0 or node_budget <= 0:
        raise ValueError("n_trees must be nonnegative and node_budget positive")

    p_max = int(backend.n_players)
    x_sizes = tuple(int(value) for value in backend.x_sizes)
    x_max = max(x_sizes)
    roots = backend.root_states(n_trees)
    if roots.shape != (n_trees,) or roots.dtype != torch.int32 or roots.device != device:
        raise ValueError("root_states must return device-local int32 [n_trees]")
    root_owner = backend.classify(roots).to(torch.int32)  # [G]
    buffer = DeviceTreeBuffer(n_trees, node_budget, p_max, device)
    root_nodes = buffer.add_roots(roots, root_owner)  # [G]

    root_reach = torch.zeros(
        (n_trees, p_max, x_max), dtype=torch.float32, device=device
    )  # [G,P,X]
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    reach_rows = torch.empty(
        (buffer.node_capacity, p_max, x_max), dtype=torch.float32, device=device
    )  # [capacity,P,X]
    reach_rows[:n_trees] = root_reach
    edge_logp = torch.full(
        (buffer.edge_capacity, x_max), -torch.inf, dtype=torch.float32, device=device
    )  # [edge_capacity,X]
    node_value = torch.zeros(
        (buffer.node_capacity, x_max), dtype=torch.float32, device=device
    )  # [node_capacity,X]

    frontier_nodes = root_nodes
    frontier_states = roots
    frontier_owner = root_owner
    frontier_reach = root_reach
    for _ in range(int(backend.max_rollout_steps)):
        live = (frontier_owner != TERMINAL) & (frontier_owner != PRETERM)
        if not bool(live.any()):
            break
        active_nodes = frontier_nodes[live]  # [A]
        active_states = frontier_states[live]  # [A]
        active_owner = frontier_owner[live]  # [A]
        active_reach = frontier_reach[live]  # [A,P,X]
        active_count = active_nodes.numel()

        if beta_mode == "sequential":
            beta = torch.ones(active_count, dtype=torch.float32, device=device)  # [A]
        else:
            beta = _validate_beta(backend.beta_fn(active_states), active_count, device)
        offspring = _offspring_counts(beta, gen)  # [A]
        parent_positions = torch.repeat_interleave(
            torch.arange(active_count, device=device), offspring
        )  # [C_new]
        parents = active_nodes[parent_positions]  # [C_new]
        states = active_states[parent_positions]  # [C_new]
        reaches = active_reach[parent_positions]  # [C_new,P,X]
        owners = active_owner[parent_positions]  # [C_new]
        child_count = parents.numel()
        profiles = backend.sample_profiles(states, reaches, gen)  # [C_new,P]

        player_positions = ((owners >= 0) & (owners < p_max)).nonzero().squeeze(1)
        chance_positions = (owners == CHANCE).nonzero().squeeze(1)
        event_parts: list[tuple[Tensor, EventBatch]] = []
        next_reach = torch.empty_like(reaches)  # [C_new,P,X]
        player_logp: Tensor | None = None
        player_output: Any | None = None

        if player_positions.numel() != 0:
            player_states = states[player_positions]  # [Cp]
            player_reach = reaches[player_positions]  # [Cp,P,X]
            with torch.no_grad():
                player_output = backend.policy_forward(
                    bundle.model, player_states, player_reach
                )
                player_events = backend.sample_events(
                    player_states,
                    profiles[player_positions],
                    player_output,
                    bundle.policy_adapter,
                    gen,
                )
                next_reach[player_positions] = backend.reach_update(
                    player_reach, player_states, player_events, player_output
                )
                legal_ids, legal_mask = backend.legal_events(player_states)
                from pbg.core.rollout import _selected_policy_logp

                player_logp = _selected_policy_logp(
                    bundle.policy_adapter,
                    player_output,
                    player_events.public_id[:, None],
                    legal_ids,
                    legal_mask,
                ).detach()[:, 0]  # [Cp,X]
            event_parts.append((player_positions, player_events))

        if chance_positions.numel() != 0:
            chance_states = states[chance_positions]  # [Cc]
            chance_events = backend.sample_events(
                chance_states,
                profiles[chance_positions],
                None,
                bundle.policy_adapter,
                gen,
            )
            next_reach[chance_positions] = backend.reach_update(
                reaches[chance_positions], chance_states, chance_events, None
            )
            event_parts.append((chance_positions, chance_events))

        if player_positions.numel() + chance_positions.numel() != child_count:
            raise RuntimeError("backend classified a live state with an invalid owner")
        events = _combine_event_rows(child_count, event_parts, device)
        next_states = backend.step(states, events)  # [C_new]
        next_owner = backend.classify(next_states).to(torch.int32)  # [C_new]
        weights = offspring[parent_positions].reciprocal().to(torch.float32)  # [C_new]
        first_edge = buffer.edge_count
        child_nodes = buffer.append_children(
            parents, next_states, next_owner, events.public_id, weights
        )  # [C_new]
        reach_rows[child_nodes.to(torch.int64)] = next_reach

        if player_positions.numel() != 0:
            assert player_logp is not None and player_output is not None
            edge_rows = torch.arange(
                first_edge, first_edge + child_count, dtype=torch.int64, device=device
            )  # [C_new]
            edge_logp[edge_rows[player_positions]] = player_logp
            decision_parent_positions = (
                (active_owner >= 0) & (active_owner < p_max)
            ).nonzero().squeeze(1)
            player_counts = offspring[decision_parent_positions]  # [Nd_wave]
            value_rows = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(player_counts, dim=0)[:-1],
                )
            )  # [Nd_wave]
            node_value[active_nodes[decision_parent_positions].to(torch.int64)] = (
                player_output.v[value_rows].detach().float()
            )

        frontier_nodes = child_nodes
        frontier_states = next_states
        frontier_owner = next_owner
        frontier_reach = next_reach

    live_frontier = (frontier_owner != TERMINAL) & (frontier_owner != PRETERM)
    if bool(live_frontier.any()):
        raise RuntimeError(
            "rollout_backend.max_rollout_steps is too small; live frontier remains"
        )

    tree = buffer.finalize()
    all_nodes = torch.arange(buffer.node_count, dtype=torch.int32, device=device)  # [N]
    if rollout_cfg.reach_storage == "full":
        reach_nodes = all_nodes
    elif rollout_cfg.reach_storage == "decisions_only":
        stored = ((tree.owner >= 0) & (tree.owner < p_max)) | (
            (tree.owner == TERMINAL) | (tree.owner == PRETERM)
        )
        reach_nodes = all_nodes[stored]
    else:
        reach_nodes = root_nodes
    reach_cache = ReachCache(
        mode=rollout_cfg.reach_storage,
        log_r=reach_rows[reach_nodes.to(torch.int64)],
        node_index=reach_nodes,
    )

    decision_nodes = ((tree.owner >= 0) & (tree.owner < p_max)).nonzero().squeeze(1)
    if decision_nodes.numel() != 0:
        counts = tree.child_ptr[decision_nodes + 1] - tree.child_ptr[decision_nodes]  # [Nd]
        width = int(counts.max())
        logp_child = torch.full(
            (decision_nodes.numel(), width, x_max),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )  # [Nd,K_max,X]
        edge_parent = buffer.edge_parent[: buffer.edge_count].to(torch.int64)  # [C]
        player_edge = (tree.owner[edge_parent] >= 0) & (tree.owner[edge_parent] < p_max)
        edge_ids = torch.arange(buffer.edge_count, device=device)[player_edge]  # [Cd]
        parent_ids = edge_parent[player_edge]  # [Cd]
        node_to_row = torch.full(
            (buffer.node_count,), -1, dtype=torch.int64, device=device
        )  # [N]
        node_to_row[decision_nodes] = torch.arange(decision_nodes.numel(), device=device)
        columns = edge_ids - tree.child_ptr[parent_ids].to(torch.int64)  # [Cd]
        logp_child[node_to_row[parent_ids], columns] = edge_logp[edge_ids]
        v_old = node_value[decision_nodes]  # [Nd,X]
    else:
        logp_child = torch.empty((0, 0, x_max), dtype=torch.float32, device=device)
        v_old = torch.empty((0, x_max), dtype=torch.float32, device=device)
    return tree, reach_cache, PolicyCache(
        logp_child=logp_child,
        v_old=v_old,
        node_index=decision_nodes.to(torch.int32),
    )


@dataclass(frozen=True)
class NodeMeta:
    """Metadata exposed to an intensity bucket function for one node."""

    state_ref: Any
    owner: int
    depth: int
    depth_player: int
    tree_id: int
    game: Any | None = None
    seat_mask: Tensor | None = None


class IntensityController:
    """Bucketed fixed/warmup and moment-head Neyman intensity controller."""

    def __init__(
        self,
        cfg_rollout: Any,
        bucket_fn: Callable[[NodeMeta], str],
        use_moment_heads: bool,
    ) -> None:
        if not callable(bucket_fn):
            raise TypeError("bucket_fn must be callable")
        self.cfg = cfg_rollout
        self.bucket_fn = bucket_fn
        self.use_moment_heads = bool(use_moment_heads)
        self.iteration = 0
        self.bucket_means: dict[str, float] = {}
        self._validate_tables()

    def _validate_tables(self) -> None:
        for name in ("beta_fix", "beta_avg", "beta_max"):
            table = getattr(self.cfg, name, None)
            if not isinstance(table, Mapping):
                raise TypeError(f"cfg_rollout.{name} must be a mapping")
            if name == "beta_fix" and "default" not in table:
                raise ValueError("cfg_rollout.beta_fix['default'] is required")
            for bucket, value in table.items():
                if not np.isfinite(float(value)) or float(value) < 1.0:
                    raise ValueError(f"{name}[{bucket!r}] must be finite and at least one")

    @staticmethod
    def synthetic_bucket(node: NodeMeta) -> str:
        """Default finite-game bucket: pairs of public-tree depths."""

        depth = node["depth"] if isinstance(node, Mapping) else node.depth
        return f"d{int(depth) // 2}"

    def _setting(self, name: str, bucket: str) -> float:
        table = getattr(self.cfg, name)
        if bucket in table:
            return float(table[bucket])
        if "default" in table:
            return float(table["default"])
        # beta_fix/default is the specified missing-bucket fallback even when
        # the optional target/ceiling tables do not declare a default.
        return float(self.cfg.beta_fix["default"])

    @staticmethod
    def _metas(node_meta: Any) -> list[Any]:
        """Accept row objects/lists and conventional batched metadata mappings."""

        if isinstance(node_meta, NodeMeta):
            return [node_meta]
        if isinstance(node_meta, Mapping):
            def batched(value: Any) -> bool:
                if isinstance(value, (list, tuple)):
                    return True
                if isinstance(value, (Tensor, np.ndarray)):
                    return value.ndim > 0
                return False

            lengths = [
                len(value)
                for value in node_meta.values()
                if batched(value)
            ]
            if not lengths:
                return [node_meta]
            if len(set(lengths)) != 1:
                raise ValueError("batched node metadata fields must have equal lengths")
            return [
                {
                    key: (
                        value[row]
                        if batched(value)
                        else value
                    )
                    for key, value in node_meta.items()
                }
                for row in range(lengths[0])
            ]
        if isinstance(node_meta, (list, tuple)):
            return list(node_meta)
        return [node_meta]

    @staticmethod
    def _meta_value(meta: Any, name: str, default: Any = None) -> Any:
        if isinstance(meta, Mapping):
            return meta.get(name, default)
        return getattr(meta, name, default)

    @staticmethod
    def moment_s(
        moment_outputs: Any, count: int, metas: list[Any] | None = None
    ) -> Tensor | None:
        if moment_outputs is None:
            return None
        vvec = getattr(moment_outputs, "vvec", None)
        u = getattr(moment_outputs, "u", None)
        if vvec is None or u is None:
            if isinstance(moment_outputs, Mapping):
                vvec, u = moment_outputs.get("vvec"), moment_outputs.get("u")
        if vvec is None or u is None:
            return None
        vector = torch.as_tensor(vvec, dtype=torch.float32).clone()
        squared = torch.as_tensor(u, dtype=torch.float32).reshape(-1)
        if vector.ndim != 2 or vector.shape[0] != count or squared.shape != (count,):
            raise ValueError("moment outputs must be vvec [B,P] and u [B]")
        if metas is not None:
            for row, meta in enumerate(metas):
                seat_mask = IntensityController._meta_value(meta, "seat_mask")
                if seat_mask is None:
                    continue
                mask = torch.as_tensor(seat_mask, dtype=torch.bool, device=vector.device)
                if mask.shape != (vector.shape[1],):
                    raise ValueError("node seat_mask must have shape [P]")
                vector[row] = vector[row].masked_fill(~mask, 0.0)
        return torch.sqrt(torch.relu(squared - vector.square().sum(dim=-1)))

    def beta(self, node_meta: Any, moment_outputs: Any) -> Tensor:
        """Return fp32 intensities, one for each metadata row."""

        metas = self._metas(node_meta)
        s_value = (
            self.moment_s(moment_outputs, len(metas), metas)
            if self.use_moment_heads else None
        )
        warm = self.iteration < int(self.cfg.warmup_iters) or s_value is None
        result: list[float] = []
        for row, meta in enumerate(metas):
            bucket = str(self.bucket_fn(meta))
            fixed = self._setting("beta_fix", bucket)
            if warm or self._meta_value(meta, "owner") == CHANCE:
                result.append(fixed)
                continue
            previous = self.bucket_means.get(bucket, 0.0)
            if previous <= 0.0 or not np.isfinite(previous):
                result.append(fixed)
                continue
            raw = self._setting("beta_avg", bucket) * float(s_value[row]) / previous
            result.append(min(max(raw, 1.0), self._setting("beta_max", bucket)))
        return torch.tensor(result, dtype=torch.float32)

    def end_iteration(self, collected_s_by_bucket: Mapping[str, Any]) -> None:
        """Replace bucket means with this iteration's finite nonnegative means."""

        means: dict[str, float] = {}
        for bucket, values in collected_s_by_bucket.items():
            tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
            if tensor.numel() == 0:
                continue
            if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
                raise ValueError("collected s values must be finite and nonnegative")
            means[str(bucket)] = float(tensor.mean())
        self.bucket_means = means
        self.iteration += 1

    # Older contract spelling retained for callers written against CONTRACTS.md.
    update_bucket_means = end_iteration

    def state_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "bucket_means": dict(self.bucket_means),
            "use_moment_heads": self.use_moment_heads,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        iteration = int(state["iteration"])
        means = {str(key): float(value) for key, value in state["bucket_means"].items()}
        if iteration < 0 or any(not np.isfinite(value) or value < 0 for value in means.values()):
            raise ValueError("invalid intensity-controller state")
        self.iteration = iteration
        self.bucket_means = means
        if "use_moment_heads" in state:
            self.use_moment_heads = bool(state["use_moment_heads"])


def _default_controller(game: Any, cfg: Any) -> IntensityController:
    bucket_fn = getattr(game, "intensity_bucket", IntensityController.synthetic_bucket)
    return IntensityController(cfg, bucket_fn, use_moment_heads=False)


def _metadata(
    nodes: list[int], builder: TreeBuilder, depths: list[int], player_depths: list[int], game: Any
) -> list[NodeMeta]:
    return [
        NodeMeta(
            state_ref=builder.state_refs[node],
            owner=builder.owners[node],
            depth=depths[node],
            depth_player=player_depths[node],
            tree_id=builder.tree_ids[node],
            game=game,
        )
        for node in nodes
    ]


def _sample_columns(probability: Tensor, gen: torch.Generator) -> Tensor:
    normalizer = probability.sum(dim=1, keepdim=True)
    if bool((normalizer <= 0).any()):
        raise RuntimeError("a rollout event distribution has zero mass")
    return torch.multinomial(probability / normalizer, 1, generator=gen).squeeze(1)


def _finalize_caches(
    game: Any,
    cfg: Any,
    builder: TreeBuilder,
    reach_by_node: list[Tensor],
    policy_records: dict[int, tuple[list[Tensor], Tensor]],
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    tree = builder.finalize()
    node_count = len(builder.parents)
    p_max = int(game.n_players)
    x_max = max(int(value) for value in game.x_sizes)
    if cfg.reach_storage == "full":
        reach_nodes = list(range(node_count))
    elif cfg.reach_storage == "decisions_only":
        reach_nodes = [
            node for node, owner in enumerate(builder.owners)
            if 0 <= owner < p_max or owner in (TERMINAL, PRETERM)
        ]
    else:
        reach_nodes = [node for node, parent in enumerate(builder.parents) if parent == -1]
    reach_cache = ReachCache(
        mode=cfg.reach_storage,
        log_r=torch.stack([reach_by_node[node] for node in reach_nodes]).float()
        if reach_nodes else torch.empty((0, p_max, x_max), dtype=torch.float32),
        node_index=torch.tensor(reach_nodes, dtype=torch.int32),
    )
    decision_nodes = sorted(policy_records)
    e_max = max((len(policy_records[node][0]) for node in decision_nodes), default=0)
    logp_child = torch.full((len(decision_nodes), e_max, x_max), -torch.inf, dtype=torch.float32)
    v_old = torch.zeros((len(decision_nodes), x_max), dtype=torch.float32)
    for row, node in enumerate(decision_nodes):
        logps, value = policy_records[node]
        for column, logp in enumerate(logps):
            width = min(x_max, logp.numel())
            logp_child[row, column, :width] = logp[:width]
        v_old[row, : min(x_max, value.numel())] = value[:x_max]
    return tree, reach_cache, PolicyCache(
        logp_child=logp_child,
        v_old=v_old,
        node_index=torch.tensor(decision_nodes, dtype=torch.int32),
    )


def _legacy_sample_branched_trees(
    game: Any,
    policy_fn: Callable[..., Any],
    adapter: Any,
    cfg_rollout: Any,
    completion_reach_hook: Callable[..., Any] | None,
    intensity: IntensityController | None,
    torch_gen: torch.Generator,
    np_gen: np.random.Generator,
) -> tuple[TreeBatch, ReachCache, PolicyCache]:
    """Sample historical P09 quotient trees for positional callers only."""

    # Imports stay here so rollout.py can dispatch without a module cycle.
    from pbg.core.rollout import _adapter_log_prob, _call_policy, _classify, _hook_reach

    if cfg_rollout.beta_mode not in {"fixed", "neyman"}:
        raise ValueError("branched rollout requires beta_mode 'fixed' or 'neyman'")
    if cfg_rollout.reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ValueError("unknown reach_storage mode")
    n_trees, budget = int(cfg_rollout.n_trees), int(cfg_rollout.node_budget)
    if n_trees < 0 or (n_trees and budget <= 0):
        raise ValueError("n_trees must be nonnegative and node_budget positive")
    controller = intensity or _default_controller(game, cfg_rollout)
    p_max = int(game.n_players)
    x_sizes = tuple(int(value) for value in game.x_sizes)
    x_max = max(x_sizes)
    roots = game.root_states(n_trees).to(torch.int32)
    owners = _classify(game, roots)
    builder = TreeBuilder(p_max)
    frontier: list[int] = []
    for tree_id, (state_ref, owner) in enumerate(zip(roots.tolist(), owners.tolist(), strict=True)):
        frontier.append(builder.add_root(state_ref, owner, tree_id))
    root_reach = torch.zeros((n_trees, p_max, x_max), dtype=torch.float32, device=roots.device)
    for player, size in enumerate(x_sizes):
        root_reach[:, player, size:] = -torch.inf
    root_reach = _hook_reach(completion_reach_hook, game, roots, root_reach)
    reach_by_node = [row.clone() for row in root_reach]
    depths = [0] * n_trees
    player_depths = [0] * n_trees
    tree_counts = [1] * n_trees
    clamped = [False] * n_trees
    policy_records: dict[int, tuple[list[Tensor], Tensor]] = {}

    while frontier:
        active = [node for node in frontier if builder.owners[node] not in (TERMINAL, PRETERM)]
        frontier = []
        if not active:
            break
        refs = torch.tensor([builder.state_refs[node] for node in active], dtype=torch.int32)
        active_owners = torch.tensor([builder.owners[node] for node in active], dtype=torch.int32)
        player_pos = ((active_owners >= 0) & (active_owners < p_max)).nonzero().squeeze(1)
        chance_pos = (active_owners == CHANCE).nonzero().squeeze(1)
        children: dict[int, list[tuple[int, int, Tensor | None]]] = {}
        multiplicities: dict[int, list[int]] = {}
        reserved_counts = tree_counts.copy()

        if player_pos.numel():
            nodes = [active[int(pos)] for pos in player_pos.tolist()]
            node_refs = refs[player_pos]
            node_owners = active_owners[player_pos].long()
            legal_ids, legal_mask = game.legal_events(node_refs)
            with torch.no_grad():
                output = _call_policy(game, policy_fn, node_refs, legal_ids, legal_mask)
                all_logp = _adapter_log_prob(adapter, output, legal_ids, legal_mask)
            moments = output if cfg_rollout.beta_mode == "neyman" else None
            betas = controller.beta(_metadata(nodes, builder, depths, player_depths, game), moments)
            for row, node in enumerate(nodes):
                tree_id = builder.tree_ids[node]
                k = (
                    1
                    if clamped[tree_id]
                    else 1 + int(np_gen.poisson(max(float(betas[row]) - 1.0, 0.0)))
                )
                reach = reach_by_node[node][None].to(node_refs.device)
                profiles = game.sample_profiles(node_refs[row : row + 1], reach, k, torch_gen)[0]
                actor_types = profiles[:, int(node_owners[row])].long().to(all_logp.device)
                probability = torch.exp(all_logp[row, :, actor_types].transpose(0, 1))
                probability = probability.masked_fill(~legal_mask[row].to(probability.device), 0.0)
                columns = _sample_columns(probability, torch_gen)
                events = legal_ids[row].to(columns.device)[columns].to(torch.int32)
                if (
                    not clamped[tree_id]
                    and reserved_counts[tree_id] + len(set(events.tolist())) > budget
                ):
                    events = events[:1]
                    k, clamped[tree_id] = 1, True
                unique: dict[int, tuple[int, int]] = {}
                for event in events.tolist():
                    if event in unique:
                        first, count = unique[event]
                        unique[event] = (first, count + 1)
                    else:
                        unique[event] = (len(unique), 1)
                ordered = sorted(unique.items(), key=lambda item: item[1][0])
                reserved_counts[tree_id] += len(ordered)
                event_tensor = torch.tensor([event for event, _ in ordered], dtype=torch.int32)
                successors = game.step(node_refs[row].repeat(len(ordered)), event_tensor)
                rows: list[tuple[int, int, Tensor | None]] = []
                cached: list[Tensor] = []
                for column, ((event, (_, count)), successor) in enumerate(zip(ordered, successors.tolist(), strict=True)):
                    legal_column = int(((legal_ids[row] == event) & legal_mask[row]).nonzero()[0])
                    selected = all_logp[row, legal_column].detach().cpu()
                    rows.append((event, int(successor), selected))
                    cached.append(selected)
                children[node] = rows
                policy_records[node] = (cached, output.v[row].detach().float().cpu())
                multiplicities[node] = [count for _, (_, count) in ordered]

        if chance_pos.numel():
            nodes = [active[int(pos)] for pos in chance_pos.tolist()]
            node_refs = refs[chance_pos]
            betas = controller.beta(_metadata(nodes, builder, depths, player_depths, game), None)
            for row, node in enumerate(nodes):
                tree_id = builder.tree_ids[node]
                k = (
                    1
                    if clamped[tree_id]
                    else 1 + int(np_gen.poisson(max(float(betas[row]) - 1.0, 0.0)))
                )
                reach = reach_by_node[node][None].to(node_refs.device)
                profiles = game.sample_profiles(node_refs[row : row + 1], reach, k, torch_gen)[0]
                if hasattr(game, "draw_chance"):
                    draw = game.draw_chance(node_refs[row].repeat(k), profiles, torch_gen)
                    events = torch.full((k,), -1, dtype=torch.int32)
                    successors = draw.state_refs
                    # Poker's engine intentionally appends dealt states, so
                    # state refs cannot identify duplicate public deals.
                    keys = [
                        tuple(sorted(int(card) for card in cards if int(card) >= 0))
                        for cards in draw.cards.detach().cpu().tolist()
                    ]
                else:
                    legal_ids, legal_mask = game.legal_events(node_refs[row : row + 1])
                    probability = game.chance_probs(node_refs[row].repeat(k), profiles)
                    columns = _sample_columns(probability, torch_gen)
                    events = legal_ids[0].to(columns.device)[columns].to(torch.int32)
                    successors = game.step(node_refs[row].repeat(k), events)
                    keys = [int(value) for value in events.tolist()]
                if (
                    not clamped[tree_id]
                    and reserved_counts[tree_id] + len(set(keys)) > budget
                ):
                    events = events[:1]
                    successors = successors[:1]
                    keys = keys[:1]
                    k, clamped[tree_id] = 1, True
                unique: dict[Any, tuple[int, int, int, int]] = {}
                for event, successor, key in zip(events.tolist(), successors.tolist(), keys, strict=True):
                    if key in unique:
                        first, count, old_event, old_successor = unique[key]
                        unique[key] = (first, count + 1, old_event, old_successor)
                    else:
                        unique[key] = (len(unique), 1, int(event), int(successor))
                ordered = sorted(unique.values(), key=lambda value: value[0])
                reserved_counts[tree_id] += len(ordered)
                children[node] = [(event, successor, None) for _, _, event, successor in ordered]
                multiplicities[node] = [count for _, count, _, _ in ordered]

        # Append only after every active node has been sampled, preserving the
        # frontier batching and deterministic sibling order.
        for node in active:
            rows = children[node]
            counts = multiplicities[node]
            total = sum(counts)
            for (event, successor, taken_logp), count in zip(rows, counts, strict=True):
                successor_ref = torch.tensor([successor], dtype=torch.int32)
                child_owner = int(_classify(game, successor_ref)[0])
                child_reach = _replay_reach_update(
                    game,
                    reach_by_node[node],
                    torch.tensor([builder.state_refs[node]], dtype=torch.int32),
                    torch.tensor([event], dtype=torch.int32),
                    builder.owners[node],
                    taken_logp,
                )
                child_reach = _hook_reach(completion_reach_hook, game, successor_ref, child_reach[None])[0]
                child = builder.add_child(node, successor, event, count / total, child_owner)
                reach_by_node.append(child_reach)
                depths.append(depths[node] + 1)
                player_depths.append(player_depths[node] + int(builder.owners[node] >= 0))
                tree_counts[builder.tree_ids[node]] += 1
                frontier.append(child)

    tree, reach_cache, policy_cache = _finalize_caches(
        game, cfg_rollout, builder, reach_by_node, policy_records
    )
    # The sampler's public return tuple is frozen, so expose the rollout metric
    # on the tree object and as an aggregate mapping without changing arity.
    tree.budget_clamped = bool(any(clamped))
    tree.rollout_metrics = {"budget_clamped": tree.budget_clamped}
    return tree, reach_cache, policy_cache
