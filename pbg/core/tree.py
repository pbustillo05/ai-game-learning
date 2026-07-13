"""Stored rollout trees, reach/policy caches, and decision alias tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch


@dataclass
class TreeBatch:
    """A batch of public rollout trees in structure-of-arrays/CSR form."""

    parent: Tensor
    child_ptr: Tensor
    child_node: Tensor
    child_event: Tensor
    nu: Tensor
    tree_id: Tensor
    owner: Tensor
    depth_p: Tensor
    depth_player: Tensor
    state_ref: Tensor
    path_weight: Tensor

    def validate(self) -> None:
        """Raise ``ValueError`` when topology or rollout-law invariants fail."""

        n = int(self.parent.numel())
        p_max = int(self.depth_p.shape[1]) if self.depth_p.ndim == 2 else -1
        one_dimensional = {
            "parent": self.parent,
            "tree_id": self.tree_id,
            "owner": self.owner,
            "depth_player": self.depth_player,
            "state_ref": self.state_ref,
            "path_weight": self.path_weight,
        }
        for name, value in one_dimensional.items():
            if value.ndim != 1 or value.numel() != n:
                raise ValueError(f"{name} must have shape [N]")
        if self.depth_p.ndim != 2 or self.depth_p.shape[0] != n:
            raise ValueError("depth_p must have shape [N,P_max]")
        if self.child_ptr.ndim != 1 or self.child_ptr.numel() != n + 1:
            raise ValueError("child_ptr must have shape [N+1]")
        c = int(self.child_node.numel())
        for name, value in {
            "child_node": self.child_node,
            "child_event": self.child_event,
            "nu": self.nu,
        }.items():
            if value.ndim != 1 or value.numel() != c:
                raise ValueError(f"{name} must have shape [C]")
        if n == 0:
            if c or int(self.child_ptr[0]) != 0:
                raise ValueError("an empty tree batch cannot contain edges")
            return
        ptr = self.child_ptr.to(torch.int64)
        if int(ptr[0]) != 0 or int(ptr[-1]) != c or bool((ptr[1:] < ptr[:-1]).any()):
            raise ValueError("child_ptr is not a valid CSR pointer")
        if not bool(torch.isfinite(self.path_weight).all()) or bool((self.path_weight < 0).any()):
            raise ValueError("path_weight must be finite and nonnegative")
        if not bool(torch.isfinite(self.nu).all()) or bool((self.nu < 0).any()):
            raise ValueError("nu must be finite and nonnegative")

        incoming = torch.zeros(n, dtype=torch.int64, device=self.parent.device)
        for node in range(n):
            start, stop = int(ptr[node]), int(ptr[node + 1])
            leaf = start == stop
            node_owner = int(self.owner[node])
            if leaf:
                if node_owner not in (TERMINAL, PRETERM):
                    raise ValueError("every leaf must be TERMINAL or PRETERM")
            else:
                if node_owner in (TERMINAL, PRETERM):
                    raise ValueError("TERMINAL/PRETERM nodes cannot have children")
                if not torch.isclose(
                    self.nu[start:stop].sum(),
                    torch.ones((), dtype=self.nu.dtype, device=self.nu.device),
                    atol=1e-6,
                    rtol=0.0,
                ):
                    raise ValueError("nu rows must sum to one")
            for edge in range(start, stop):
                child = int(self.child_node[edge])
                if not 0 <= child < n:
                    raise ValueError("child_node contains an invalid node id")
                incoming[child] += 1
                if int(self.parent[child]) != node:
                    raise ValueError("child and parent arrays disagree")
                if int(self.tree_id[child]) != int(self.tree_id[node]):
                    raise ValueError("an edge cannot cross tree ids")
                expected_depth = self.depth_p[node].clone()
                if 0 <= node_owner < p_max:
                    expected_depth[node_owner] += 1
                if not torch.equal(self.depth_p[child], expected_depth):
                    raise ValueError("depth_p is inconsistent with the parent chain")
                expected_player_depth = int(self.depth_player[node]) + int(
                    0 <= node_owner < p_max
                )
                if int(self.depth_player[child]) != expected_player_depth:
                    raise ValueError("depth_player is inconsistent with the parent chain")
                expected_weight = self.path_weight[node] * self.nu[edge]
                if not torch.isclose(
                    self.path_weight[child], expected_weight, atol=1e-6, rtol=1e-6
                ):
                    raise ValueError("path_weight is inconsistent with edge nu")

        roots = self.parent == -1
        if not bool(roots.any()) or bool(((self.parent < 0) & ~roots).any()):
            raise ValueError("parents must be -1 only at roots")
        if bool((incoming[roots] != 0).any()) or bool((incoming[~roots] != 1).any()):
            raise ValueError("every non-root must have exactly one incoming edge")
        if not torch.equal(self.depth_p[roots], torch.zeros_like(self.depth_p[roots])):
            raise ValueError("root depth_p must be zero")
        if not torch.equal(
            self.depth_player[roots], torch.zeros_like(self.depth_player[roots])
        ):
            raise ValueError("root depth_player must be zero")
        if not torch.allclose(
            self.path_weight[roots], torch.ones_like(self.path_weight[roots]), atol=0, rtol=0
        ):
            raise ValueError("root path_weight must be one")
        root_ids = self.tree_id[roots].to(torch.int64)
        if root_ids.unique().numel() != root_ids.numel():
            raise ValueError("each tree id must have exactly one root")


class TreeBuilder:
    """Append-only builder used by frontier rollout construction."""

    def __init__(self, n_players: int):
        if n_players <= 0:
            raise ValueError("n_players must be positive")
        self.n_players = int(n_players)
        self.parents: list[int] = []
        self.tree_ids: list[int] = []
        self.owners: list[int] = []
        self.state_refs: list[int] = []
        self.children: list[list[tuple[int, int, float]]] = []

    def add_root(self, state_ref: int, owner: int, tree_id: int | None = None) -> int:
        """Append one root and return its node id."""

        if tree_id is None:
            tree_id = sum(parent == -1 for parent in self.parents)
        return self.add_node(-1, state_ref, owner, tree_id=tree_id)

    append_root = add_root

    def add_node(
        self,
        parent: int,
        state_ref: int,
        owner: int,
        *,
        tree_id: int | None = None,
        event: int | None = None,
        nu: float = 1.0,
    ) -> int:
        """Append a root or child; child calls also append the parent edge."""

        node = len(self.parents)
        if parent == -1:
            if tree_id is None:
                tree_id = sum(value == -1 for value in self.parents)
            if event is not None:
                raise ValueError("a root cannot have a parent event")
        else:
            if not 0 <= parent < node:
                raise ValueError("parent must refer to an earlier node")
            if event is None:
                raise ValueError("a child requires its parent event")
            inherited = self.tree_ids[parent]
            if tree_id is None:
                tree_id = inherited
            elif int(tree_id) != inherited:
                raise ValueError("a child must inherit its parent's tree id")
        self.parents.append(int(parent))
        self.tree_ids.append(int(tree_id))
        self.owners.append(int(owner))
        self.state_refs.append(int(state_ref))
        self.children.append([])
        if parent >= 0:
            self.children[parent].append((node, int(event), float(nu)))
        return node

    def add_child(
        self, parent: int, state_ref: int, event: int, nu: float, owner: int
    ) -> int:
        """Append one child node and its incoming edge."""

        return self.add_node(parent, state_ref, owner, event=event, nu=nu)

    append_child = add_child

    def finalize(self) -> TreeBatch:
        """Materialize contiguous tensors and compute depths/path weights."""

        n = len(self.parents)
        child_ptr = [0]
        child_node: list[int] = []
        child_event: list[int] = []
        edge_nu: list[float] = []
        for edges in self.children:
            for child, event, weight in edges:
                child_node.append(child)
                child_event.append(event)
                edge_nu.append(weight)
            child_ptr.append(len(child_node))

        depth_p = torch.zeros((n, self.n_players), dtype=torch.int32)
        depth_player = torch.zeros(n, dtype=torch.int32)
        path_weight = torch.ones(n, dtype=torch.float32)
        incoming_nu = {child: weight for edges in self.children for child, _, weight in edges}
        for node, parent in enumerate(self.parents):
            if parent < 0:
                continue
            depth_p[node] = depth_p[parent]
            parent_owner = self.owners[parent]
            if 0 <= parent_owner < self.n_players:
                depth_p[node, parent_owner] += 1
                depth_player[node] = depth_player[parent] + 1
            else:
                depth_player[node] = depth_player[parent]
            path_weight[node] = path_weight[parent] * incoming_nu[node]

        result = TreeBatch(
            parent=torch.tensor(self.parents, dtype=torch.int32),
            child_ptr=torch.tensor(child_ptr, dtype=torch.int32),
            child_node=torch.tensor(child_node, dtype=torch.int32),
            child_event=torch.tensor(child_event, dtype=torch.int32),
            nu=torch.tensor(edge_nu, dtype=torch.float32),
            tree_id=torch.tensor(self.tree_ids, dtype=torch.int32),
            owner=torch.tensor(self.owners, dtype=torch.int32),
            depth_p=depth_p,
            depth_player=depth_player,
            state_ref=torch.tensor(self.state_refs, dtype=torch.int32),
            path_weight=path_weight,
        )
        result.validate()
        return result


class DeviceTreeBuffer:
    """Preallocated device-resident storage for a batch of rollout trees.

    Node capacity is ``n_trees * node_budget`` and edge capacity is that value
    minus ``n_trees``.  Roots always occupy the first ``n_trees`` rows.
    """

    def __init__(
        self,
        n_trees: int,
        node_budget: int,
        n_players: int,
        device: torch.device | str,
    ) -> None:
        if n_trees < 0 or node_budget <= 0 or n_players <= 0:
            raise ValueError("n_trees must be nonnegative and budgets must be positive")
        self.n_trees = int(n_trees)
        self.node_budget = int(node_budget)
        self.n_players = int(n_players)
        self.device = torch.device(device)
        self.node_capacity = self.n_trees * self.node_budget
        self.edge_capacity = self.node_capacity - self.n_trees
        self.node_count = 0
        self.edge_count = 0

        self.parent = torch.full(
            (self.node_capacity,), -1, dtype=torch.int32, device=self.device
        )
        self.tree_id = torch.empty(
            self.node_capacity, dtype=torch.int32, device=self.device
        )
        self.owner = torch.empty(
            self.node_capacity, dtype=torch.int32, device=self.device
        )
        self.state_ref = torch.empty(
            self.node_capacity, dtype=torch.int32, device=self.device
        )
        self.depth_p = torch.zeros(
            (self.node_capacity, self.n_players), dtype=torch.int32, device=self.device
        )
        self.depth_player = torch.zeros(
            self.node_capacity, dtype=torch.int32, device=self.device
        )
        self.path_weight = torch.empty(
            self.node_capacity, dtype=torch.float32, device=self.device
        )
        self.edge_parent = torch.empty(
            self.edge_capacity, dtype=torch.int32, device=self.device
        )
        self.child_node = torch.empty(
            self.edge_capacity, dtype=torch.int32, device=self.device
        )
        self.child_event = torch.empty(
            self.edge_capacity, dtype=torch.int32, device=self.device
        )
        self.edge_weight = torch.empty(
            self.edge_capacity, dtype=torch.float32, device=self.device
        )

    def add_roots(self, states: Tensor, owners: Tensor) -> Tensor:
        """Store int32 root ``states``/``owners`` ``[G]`` and return node ids."""

        if states.shape != (self.n_trees,) or owners.shape != (self.n_trees,):
            raise ValueError("root tensors must have shape [n_trees]")
        if states.device != self.device or owners.device != self.device:
            raise ValueError("root tensors must be on the buffer device")
        ids = torch.arange(self.n_trees, dtype=torch.int32, device=self.device)  # [G]
        self.state_ref[: self.n_trees] = states.to(torch.int32)
        self.owner[: self.n_trees] = owners.to(torch.int32)
        self.tree_id[: self.n_trees] = ids
        self.path_weight[: self.n_trees] = 1.0
        self.node_count = self.n_trees
        return ids

    append_roots = add_roots

    def append_children(
        self,
        parents: Tensor,
        states: Tensor,
        owners: Tensor,
        events: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Append flattened parent-major child rows and return int32 node ids.

        All arguments are one-dimensional ``[C_new]`` tensors.  Device prefix
        sums assign contiguous child and edge rows without per-node mutation.
        """

        count = parents.numel()
        if any(value.shape != (count,) for value in (states, owners, events, weights)):
            raise ValueError("child append tensors must all have shape [C_new]")
        if any(value.device != self.device for value in (parents, states, owners, events, weights)):
            raise ValueError("child append tensors must be on the buffer device")
        requested_nodes = self.node_count + count
        requested_edges = self.edge_count + count
        if requested_nodes > self.node_capacity or requested_edges > self.edge_capacity:
            raise RuntimeError(
                "DeviceTreeBuffer overflow: "
                f"requested nodes={requested_nodes}, edges={requested_edges}; "
                f"capacity nodes={self.node_capacity}, edges={self.edge_capacity}"
            )
        if count == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        offsets = torch.cumsum(
            torch.ones(count, dtype=torch.int32, device=self.device), dim=0
        ).to(torch.int32) - 1  # [C_new]
        nodes = offsets + self.node_count  # [C_new]
        edges = offsets.to(torch.int64) + self.edge_count  # [C_new]
        node_rows = nodes.to(torch.int64)
        parent_rows = parents.to(torch.int64)
        parent_owner = self.owner[parent_rows]  # [C_new]

        self.parent[node_rows] = parents.to(torch.int32)
        self.state_ref[node_rows] = states.to(torch.int32)
        self.owner[node_rows] = owners.to(torch.int32)
        self.tree_id[node_rows] = self.tree_id[parent_rows]
        self.depth_p[node_rows] = self.depth_p[parent_rows]
        player_mask = (parent_owner >= 0) & (parent_owner < self.n_players)  # [C_new]
        increments = torch.zeros(
            (count, self.n_players), dtype=torch.int32, device=self.device
        )  # [C_new,P]
        safe_owner = parent_owner.clamp(0, self.n_players - 1).to(torch.int64)
        increments.scatter_(1, safe_owner[:, None], player_mask.to(torch.int32)[:, None])
        self.depth_p[node_rows] += increments
        self.depth_player[node_rows] = (
            self.depth_player[parent_rows] + player_mask.to(torch.int32)
        )
        self.path_weight[node_rows] = self.path_weight[parent_rows] * weights.float()

        self.edge_parent[edges] = parents.to(torch.int32)
        self.child_node[edges] = nodes
        self.child_event[edges] = events.to(torch.int32)
        self.edge_weight[edges] = weights.float()
        self.node_count = requested_nodes
        self.edge_count = requested_edges
        return nodes

    def finalize(self) -> TreeBatch:
        """Return device views truncated to the realized node/edge counts."""

        n = self.node_count
        c = self.edge_count
        edge_parent = self.edge_parent[:c].to(torch.int64)  # [C]
        counts = torch.bincount(edge_parent, minlength=n).to(torch.int32)  # [N]
        child_ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(counts, dim=0).to(torch.int32),
            )
        )  # [N+1]
        return TreeBatch(
            parent=self.parent[:n],
            child_ptr=child_ptr,
            child_node=self.child_node[:c],
            child_event=self.child_event[:c],
            nu=self.edge_weight[:c],
            tree_id=self.tree_id[:n],
            owner=self.owner[:n],
            depth_p=self.depth_p[:n],
            depth_player=self.depth_player[:n],
            state_ref=self.state_ref[:n],
            path_weight=self.path_weight[:n],
        )


@dataclass
class AliasTable:
    """Walker alias table with optional external outcomes."""

    probability: Tensor
    alias: Tensor
    outcomes: Tensor

    @classmethod
    def build(cls, weights: Tensor | np.ndarray | Iterable[float]) -> "AliasTable":
        source_device = weights.device if isinstance(weights, Tensor) else torch.device("cpu")
        values = torch.as_tensor(weights, dtype=torch.float64).flatten().cpu()
        if values.numel() == 0:
            raise ValueError("alias weights cannot be empty")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("alias weights must be finite and nonnegative")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("at least one alias weight must be positive")
        count = values.numel()
        scaled = (values * count / total).tolist()
        probability = [0.0] * count
        alias = list(range(count))
        small = [index for index, value in enumerate(scaled) if value < 1.0]
        large = [index for index, value in enumerate(scaled) if value >= 1.0]
        while small and large:
            low = small.pop()
            high = large.pop()
            probability[low] = scaled[low]
            alias[low] = high
            scaled[high] = scaled[high] - (1.0 - scaled[low])
            (small if scaled[high] < 1.0 else large).append(high)
        for index in (*small, *large):
            probability[index] = 1.0
            alias[index] = index
        return cls(
            probability=torch.tensor(probability, dtype=torch.float64, device=source_device),
            alias=torch.tensor(alias, dtype=torch.int64, device=source_device),
            outcomes=torch.arange(count, dtype=torch.int32, device=source_device),
        )

    @property
    def prob(self) -> Tensor:
        """Conventional short alias for the acceptance probabilities."""

        return self.probability

    def sample(
        self, n: int, gen: torch.Generator | np.random.Generator
    ) -> Tensor:
        """Sample ``n`` outcomes with replacement as int32."""

        if n < 0:
            raise ValueError("n must be nonnegative")
        size = int(self.probability.numel())
        if isinstance(gen, np.random.Generator):
            if self.probability.device.type != "cpu":
                raise ValueError("NumPy generators can only sample CPU alias tables")
            columns = torch.from_numpy(gen.integers(size, size=n, dtype=np.int64))
            uniforms = torch.from_numpy(gen.random(n))
        else:
            columns = torch.randint(
                size, (n,), generator=gen, dtype=torch.int64, device=self.probability.device
            )
            uniforms = torch.rand(
                n, generator=gen, dtype=torch.float64, device=self.probability.device
            )
        picked = torch.where(
            uniforms < self.probability[columns], columns, self.alias[columns]
        )
        return self.outcomes[picked].to(torch.int32)


def build_decision_alias(tree: TreeBatch) -> AliasTable:
    """Build the length-biased decision law over decision-row indices."""

    p_max = tree.depth_p.shape[1]
    node_ids = ((tree.owner >= 0) & (tree.owner < p_max)).nonzero(as_tuple=False).squeeze(1)
    if node_ids.numel() == 0:
        raise ValueError("tree batch contains no player decision nodes")
    return AliasTable.build(tree.path_weight[node_ids])


@dataclass
class PolicyCache:
    """Frozen rollout outputs for the stored children of player nodes."""

    node_ids: Tensor  # int32 [Nd]
    old_logp_child: Tensor  # float32 [Nd,E,X]
    old_v: Tensor  # float32 [Nd,X]

    def __post_init__(self) -> None:
        if self.node_ids.ndim != 1 or self.node_ids.dtype != torch.int32:
            raise ValueError("PolicyCache.node_ids must be int32 [Nd]")
        if self.old_logp_child.ndim != 3 or self.old_logp_child.shape[0] != self.node_ids.numel():
            raise ValueError("PolicyCache.old_logp_child must have shape [Nd,E,X]")
        if self.old_v.ndim != 2 or self.old_v.shape[0] != self.node_ids.numel():
            raise ValueError("PolicyCache.old_v must have shape [Nd,X]")
        if self.old_logp_child.dtype != torch.float32 or self.old_v.dtype != torch.float32:
            raise ValueError("PolicyCache floating tensors must be fp32")


def _replay_reach_update(
    game: Any,
    row: Tensor,
    state_ref: Tensor,
    child_state_ref: Tensor,
    event: Tensor,
    owner: int,
    taken_logp: Tensor | None,
) -> Tensor:
    """Replay one public edge from cached selected-action likelihoods."""

    if owner >= 0:
        if taken_logp is None:
            raise ValueError("player reach replay requires cached log-probabilities")
        result = row.clone()
        width = min(result.shape[-1], taken_logp.numel())
        result[owner, :width] += taken_logp[:width].to(result.device, result.dtype)
        return result

    # A game may reconstruct chance information from its successor state.  Poker
    # uses this to recover dealt board cards without retaining poker-specific
    # edge payloads in TreeBatch.
    replay_chance = getattr(game, "replay_chance_reach", None)
    if callable(replay_chance):
        updated = replay_chance(row[None], state_ref, child_state_ref)
        return updated[0]

    # Generic rollout backends already expose the typed EventBatch boundary.
    # In particular, do not manufacture a policy tensor indexed by public_id:
    # chance labels need not be compact action indices.
    updated = game.reach_update(
        row[None], state_ref, EventBatch(public_id=event), None
    )
    return updated[0]


@dataclass
class ReachCache:
    """Mode-dependent log-reach storage with transparent replay."""

    mode: str
    log_r: Tensor
    node_index: Tensor

    def rows_for(
        self,
        tree: TreeBatch,
        node_ids: Tensor,
        game: Any,
        policy_cache: PolicyCache,
    ) -> Tensor:
        """Return log-reach rows for arbitrary stored-tree nodes."""

        if self.mode not in {"full", "decisions_only", "recompute"}:
            raise ValueError(f"unknown reach-cache mode {self.mode!r}")
        requested = node_ids.to(device=self.node_index.device, dtype=torch.int64)
        stored_ids = self.node_index.to(torch.int64)
        positions = torch.searchsorted(stored_ids, requested)
        present = positions < stored_ids.numel()
        if stored_ids.numel():
            present &= stored_ids[positions.clamp_max(stored_ids.numel() - 1)] == requested
        if self.mode != "recompute":
            if not bool(present.all()):
                raise KeyError("one or more requested reach rows are not stored")
            return self.log_r[positions.to(self.log_r.device)]

        n = int(tree.parent.numel())
        memo = torch.empty((n, *self.log_r.shape[1:]), dtype=self.log_r.dtype, device=self.log_r.device)
        known = torch.zeros(n, dtype=torch.bool, device=self.log_r.device)
        memo[stored_ids.to(self.log_r.device)] = self.log_r
        known[stored_ids.to(self.log_r.device)] = True
        policy_ids = policy_cache.node_ids.to(torch.int64)

        def compute(node: int) -> Tensor:
            if bool(known[node]):
                return memo[node]
            parent = int(tree.parent[node])
            if parent < 0:
                raise KeyError(f"root node {node} is absent from recompute cache")
            edge_start, edge_stop = int(tree.child_ptr[parent]), int(tree.child_ptr[parent + 1])
            matches = [
                edge for edge in range(edge_start, edge_stop) if int(tree.child_node[edge]) == node
            ]
            if len(matches) != 1:
                raise ValueError("tree CSR does not contain exactly one parent edge")
            edge = matches[0]
            taken = None
            owner = int(tree.owner[parent])
            if owner >= 0:
                cache_row = int(torch.searchsorted(policy_ids, torch.tensor(parent, device=policy_ids.device)))
                if cache_row >= policy_ids.numel() or int(policy_ids[cache_row]) != parent:
                    raise KeyError(f"player node {parent} is absent from PolicyCache")
                child_slot = edge - edge_start
                taken = policy_cache.old_logp_child[cache_row, child_slot]
            value = _replay_reach_update(
                game,
                compute(parent),
                tree.state_ref[parent : parent + 1],
                tree.state_ref[node : node + 1],
                tree.child_event[edge : edge + 1],
                owner,
                taken,
            )
            memo[node] = value
            known[node] = True
            return value

        return torch.stack([compute(int(node)) for node in requested.tolist()], dim=0)
