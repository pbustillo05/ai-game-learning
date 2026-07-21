"""Legacy rollout trees, sealed leaf rows, and decision compatibility views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, PROBE_FOLD, TERMINAL


@dataclass
class TreeBatch:
    """CSR rollout topology: trajectories or one sampled branching tree.

    ``depth_p[node, p]`` counts the ``p``-owned nodes on the path through
    ``node``, including ``node`` itself when ``owner[node] == p``.
    ``depth_player`` is the corresponding count over all players.
    """

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
        if p_max <= 0:
            raise ValueError("depth_p must have a positive player dimension")
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

        # Validate all CSR rows and edges in a bounded number of tensor
        # operations.  The former scalar loop read several CUDA values for
        # every node and edge, which synchronized the host thousands of times
        # for an otherwise device-resident rollout.
        counts = ptr[1:] - ptr[:-1]
        leaves = counts == 0
        leaf_coded = (
            (self.owner == TERMINAL)
            | (self.owner == PRETERM)
            | (self.owner == PROBE_FOLD)
        )
        if bool((leaves & ~leaf_coded).any()):
            raise ValueError("every leaf must be TERMINAL, PRETERM, or PROBE_FOLD")
        if bool((~leaves & leaf_coded).any()):
            raise ValueError("leaf-coded nodes cannot have children")

        edge_parent = torch.repeat_interleave(
            torch.arange(n, dtype=torch.int64, device=self.parent.device), counts
        )
        row_nu = torch.zeros(n, dtype=self.nu.dtype, device=self.nu.device)
        row_nu.index_add_(0, edge_parent, self.nu)
        if not torch.allclose(
            row_nu[~leaves],
            torch.ones_like(row_nu[~leaves]),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("nu rows must sum to one")

        child = self.child_node.to(torch.int64)
        if bool(((child < 0) | (child >= n)).any()):
            raise ValueError("child_node contains an invalid node id")
        if not torch.equal(self.parent[child].to(torch.int64), edge_parent):
            raise ValueError("child and parent arrays disagree")
        if not torch.equal(self.tree_id[child], self.tree_id[edge_parent]):
            raise ValueError("an edge cannot cross tree ids")

        child_owner = self.owner[child].to(torch.int64)
        child_is_player = (child_owner >= 0) & (child_owner < p_max)
        expected_depth = self.depth_p[edge_parent].clone()
        expected_depth.scatter_add_(
            1,
            child_owner.clamp(0, p_max - 1)[:, None],
            child_is_player.to(expected_depth.dtype)[:, None],
        )
        if not torch.equal(self.depth_p[child], expected_depth):
            raise ValueError("depth_p is inconsistent with the parent chain")
        expected_player_depth = (
            self.depth_player[edge_parent]
            + child_is_player.to(self.depth_player.dtype)
        )
        if not torch.equal(self.depth_player[child], expected_player_depth):
            raise ValueError("depth_player is inconsistent with the parent chain")
        expected_weight = self.path_weight[edge_parent] * self.nu
        if not torch.allclose(
            self.path_weight[child], expected_weight, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("path_weight is inconsistent with edge nu")

        incoming = torch.bincount(child, minlength=n)

        roots = self.parent == -1
        if not bool(roots.any()) or bool(((self.parent < 0) & ~roots).any()):
            raise ValueError("parents must be -1 only at roots")
        if bool((incoming[roots] != 0).any()) or bool((incoming[~roots] != 1).any()):
            raise ValueError("every non-root must have exactly one incoming edge")
        root_rows = roots.nonzero(as_tuple=False).squeeze(1)
        expected_root_depth = torch.zeros_like(self.depth_p[root_rows])
        root_owner = self.owner[root_rows].to(torch.int64)
        root_player = (root_owner >= 0) & (root_owner < p_max)
        safe_root_owner = root_owner.clamp(0, p_max - 1)
        expected_root_depth.scatter_(
            1, safe_root_owner[:, None], root_player.to(torch.int32)[:, None]
        )
        if not torch.equal(self.depth_p[root_rows], expected_root_depth):
            raise ValueError("root depth_p must count the root owner")
        expected_root_player_depth = root_player.to(self.depth_player.dtype)
        if not torch.equal(
            self.depth_player[root_rows], expected_root_player_depth
        ):
            raise ValueError("root depth_player must count a player-owned root")
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
            if parent >= 0:
                depth_p[node] = depth_p[parent]
                depth_player[node] = depth_player[parent]
                path_weight[node] = path_weight[parent] * incoming_nu[node]
            owner = self.owners[node]
            if 0 <= owner < self.n_players:
                depth_p[node, owner] += 1
                depth_player[node] += 1

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

    Node capacity is ``root_count * node_budget`` and edge capacity is that value
    minus ``root_count``.  Roots always occupy the first ``root_count`` rows.
    """

    def __init__(
        self,
        root_count: int,
        node_budget: int,
        n_players: int,
        device: torch.device | str,
    ) -> None:
        if root_count < 0 or node_budget <= 0 or n_players <= 0:
            raise ValueError("root_count must be nonnegative and budgets must be positive")
        self.root_count = int(root_count)
        self.node_budget = int(node_budget)
        self.n_players = int(n_players)
        self.device = torch.device(device)
        self.node_capacity = self.root_count * self.node_budget
        self.edge_capacity = self.node_capacity - self.root_count
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
        self.child_count = torch.zeros(
            self.node_capacity, dtype=torch.int32, device=self.device
        )
        self.edge_ordinal = torch.empty(
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

        if states.shape != (self.root_count,) or owners.shape != (self.root_count,):
            raise ValueError("root tensors must have shape [root_count]")
        if states.device != self.device or owners.device != self.device:
            raise ValueError("root tensors must be on the buffer device")
        ids = torch.arange(self.root_count, dtype=torch.int32, device=self.device)  # [G]
        self.state_ref[: self.root_count] = states.to(torch.int32)
        self.owner[: self.root_count] = owners.to(torch.int32)
        self.tree_id[: self.root_count] = ids
        root_owners = owners.to(torch.int64)
        player_mask = (root_owners >= 0) & (root_owners < self.n_players)
        self.depth_p[: self.root_count].zero_()
        safe_owner = root_owners.clamp(0, self.n_players - 1)
        self.depth_p[ids.to(torch.int64), safe_owner] = player_mask.to(torch.int32)
        self.depth_player[: self.root_count] = player_mask.to(torch.int32)
        self.path_weight[: self.root_count] = 1.0
        self.node_count = self.root_count
        return ids

    append_roots = add_roots

    def require_child_capacity(self, count: int | Tensor) -> None:
        """Fail before a wave allocates expanded child/state/reach tensors."""

        remaining = min(
            self.node_capacity - self.node_count,
            self.edge_capacity - self.edge_count,
        )
        if isinstance(count, Tensor):
            if count.numel() != 1:
                raise ValueError("child count must be scalar")
            condition = count <= remaining
            if count.device.type == "cuda":
                torch._assert_async(
                    condition,
                    "DeviceTreeBuffer overflow before child expansion",
                )
                return
            count = int(count)
        if int(count) > remaining:
            raise RuntimeError(
                "DeviceTreeBuffer overflow before child expansion: "
                f"requested nodes={self.node_count + int(count)}, "
                f"edges={self.edge_count + int(count)}; capacity "
                f"nodes={self.node_capacity}, edges={self.edge_capacity}"
            )

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
        self.parent[node_rows] = parents.to(torch.int32)
        self.state_ref[node_rows] = states.to(torch.int32)
        self.owner[node_rows] = owners.to(torch.int32)
        self.tree_id[node_rows] = self.tree_id[parent_rows]
        self.depth_p[node_rows] = self.depth_p[parent_rows]
        child_owner = owners.to(torch.int64)
        player_mask = (child_owner >= 0) & (child_owner < self.n_players)  # [C_new]
        increments = torch.zeros(
            (count, self.n_players), dtype=torch.int32, device=self.device
        )  # [C_new,P]
        safe_owner = child_owner.clamp(0, self.n_players - 1)
        increments.scatter_(1, safe_owner[:, None], player_mask.to(torch.int32)[:, None])
        self.depth_p[node_rows] += increments
        self.depth_player[node_rows] = (
            self.depth_player[parent_rows] + player_mask.to(torch.int32)
        )
        self.path_weight[node_rows] = self.path_weight[parent_rows] * weights.float()

        # ``parents`` is parent-major by contract.  Retain the stable ordinal of
        # every abstract child so finalization can scatter edges directly into
        # CSR order even when family-scheduled execution expands parents across depths.
        local_rows = torch.arange(count, dtype=torch.int32, device=self.device)
        group_start = torch.zeros(count, dtype=torch.int32, device=self.device)
        group_start[0] = 0
        if count > 1:
            starts = parents[1:] != parents[:-1]
            group_start[1:] = torch.where(starts, local_rows[1:], 0)
        group_start = torch.cummax(group_start, dim=0).values
        local_ordinal = local_rows - group_start
        child_base = self.child_count[parent_rows]

        self.edge_parent[edges] = parents.to(torch.int32)
        self.edge_ordinal[edges] = child_base + local_ordinal
        self.child_node[edges] = nodes
        self.child_event[edges] = events.to(torch.int32)
        self.edge_weight[edges] = weights.float()
        self.child_count.index_add_(
            0,
            parent_rows,
            torch.ones(count, dtype=torch.int32, device=self.device),
        )
        self.node_count = requested_nodes
        self.edge_count = requested_edges
        return nodes

    def finalize(self) -> TreeBatch:
        """Return device views truncated to the realized node/edge counts."""

        n = self.node_count
        c = self.edge_count
        edge_parent = self.edge_parent[:c].to(torch.int64)  # [C]
        counts = self.child_count[:n]  # [N]
        child_ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(counts, dim=0).to(torch.int32),
            )
        )  # [N+1]
        # Queued execution can finish a deeper occurrence before an older
        # occurrence in another family.  Stable ordinals recorded at expansion
        # make this a direct O(C) scatter, with no label sort or host transfer.
        edge_destination = (
            child_ptr[edge_parent] + self.edge_ordinal[:c]
        ).to(torch.int64)
        child_node = torch.empty_like(self.child_node[:c])
        child_event = torch.empty_like(self.child_event[:c])
        edge_weight = torch.empty_like(self.edge_weight[:c])
        child_node[edge_destination] = self.child_node[:c]
        child_event[edge_destination] = self.child_event[:c]
        edge_weight[edge_destination] = self.edge_weight[:c]
        return TreeBatch(
            parent=self.parent[:n],
            child_ptr=child_ptr,
            child_node=child_node,
            child_event=child_event,
            nu=edge_weight,
            tree_id=self.tree_id[:n],
            owner=self.owner[:n],
            depth_p=self.depth_p[:n],
            depth_player=self.depth_player[:n],
            state_ref=self.state_ref[:n],
            path_weight=self.path_weight[:n],
        )


@dataclass
class AliasTable:
    """Device-resident categorical decision table with external outcomes.

    The historical name is retained for the public API. Construction and
    sampling stay on the source tensor's device; sampling delegates to
    ``torch.multinomial`` instead of building a Walker table through CPU
    lists and scalar transfers.
    """

    probability: Tensor
    alias: Tensor
    outcomes: Tensor

    @classmethod
    def build(cls, weights: Tensor | np.ndarray | Iterable[float]) -> "AliasTable":
        source_device = weights.device if isinstance(weights, Tensor) else torch.device("cpu")
        values = torch.as_tensor(weights, dtype=torch.float64, device=source_device).flatten()
        if values.numel() == 0:
            raise ValueError("alias weights cannot be empty")
        finite_nonnegative = torch.isfinite(values).all() & (values >= 0).all()
        positive_total = values.sum() > 0
        if values.device.type == "cuda":
            torch._assert_async(
                finite_nonnegative,
                "alias weights must be finite and nonnegative",
            )
            torch._assert_async(
                positive_total,
                "at least one alias weight must be positive",
            )
        else:
            if not bool(finite_nonnegative):
                raise ValueError("alias weights must be finite and nonnegative")
            if not bool(positive_total):
                raise ValueError("at least one alias weight must be positive")
        count = values.numel()
        probability = values / values.sum()
        identity = torch.arange(count, dtype=torch.int64, device=source_device)
        return cls(
            probability=probability,
            alias=identity,
            outcomes=identity.to(torch.int32),
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
        if isinstance(gen, np.random.Generator):
            if self.probability.device.type != "cpu":
                raise ValueError("NumPy generators can only sample CPU alias tables")
            columns = torch.from_numpy(
                gen.choice(
                    self.probability.numel(),
                    size=n,
                    replace=True,
                    p=self.probability.numpy(),
                ).astype(np.int64, copy=False)
            )
        else:
            columns = torch.multinomial(
                self.probability,
                n,
                replacement=True,
                generator=gen,
            )
        return self.outcomes[columns].to(torch.int32)


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
    decision_reach: Any | None = None  # zero-copy-compatible natural input owner

    def __post_init__(self) -> None:
        if self.node_ids.ndim != 1 or self.node_ids.dtype != torch.int32:
            raise ValueError("PolicyCache.node_ids must be int32 [Nd]")
        if self.old_logp_child.ndim != 3 or self.old_logp_child.shape[0] != self.node_ids.numel():
            raise ValueError("PolicyCache.old_logp_child must have shape [Nd,E,X]")
        if self.old_v.ndim != 2 or self.old_v.shape[0] != self.node_ids.numel():
            raise ValueError("PolicyCache.old_v must have shape [Nd,X]")
        if self.old_logp_child.dtype != torch.float32 or self.old_v.dtype != torch.float32:
            raise ValueError("PolicyCache floating tensors must be fp32")


@dataclass
class ReachCache:
    """Exact last-consumer leaf reach rows; no discarded path can be replayed."""

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
        """Return directly stored rows for arbitrary sealed leaf ids."""

        if self.mode not in {"sealed", "decisions_only"}:
            raise ValueError(f"unknown sealed reach mode {self.mode!r}")
        requested = node_ids.to(device=self.node_index.device, dtype=torch.int64)
        stored_ids = self.node_index.to(torch.int64)
        positions = torch.searchsorted(stored_ids, requested)
        present = positions < stored_ids.numel()
        if stored_ids.numel():
            present &= stored_ids[positions.clamp_max(stored_ids.numel() - 1)] == requested
        if not bool(present.all()):
            raise KeyError("one or more requested leaf reach rows are not sealed")
        return self.log_r[positions.to(self.log_r.device)]


@dataclass
class DecisionReachCache:
    """Model-input reaches retained after rollout/completion has finished.

    Games may replace dense ``[D,P,X]`` rows with an opaque compact payload.
    The payload is expanded only for sampled optimization rows.
    """

    node_ids: Tensor  # int32 [D], aligned with decision/target rows
    state_refs: Tensor  # int32 [D]
    payload: Any
    owners: Tensor | None = None  # int32 [D], present for compact RB minibatches

    @classmethod
    def from_rows(
        cls,
        node_ids: Tensor,
        state_refs: Tensor,
        rows: Tensor,
        game: Any,
        owners: Tensor | None = None,
    ) -> "DecisionReachCache":
        """Compact dense decision rows through the optional game boundary."""

        if rows.ndim != 3 or rows.shape[0] != node_ids.numel():
            raise ValueError("decision reach rows must have shape [D,P,X]")
        if state_refs.shape != node_ids.shape:
            raise ValueError("decision state refs must align with node ids")
        compact = getattr(game, "compact_decision_reach", None)
        if bool(getattr(game, "requires_full_decision_reach", False)):
            compact = None
        payload = compact(state_refs, rows) if callable(compact) else rows
        return cls(
            node_ids=node_ids.to(torch.int32),
            state_refs=state_refs.to(torch.int32),
            payload=payload,
            owners=None if owners is None else owners.to(torch.int32),
        )

    def gather(self, rows: Tensor, game: Any) -> Tensor:
        """Expand sampled decision rows to the backend's dense policy boundary."""

        rows = rows.to(device=self.node_ids.device, dtype=torch.int64)
        if isinstance(self.payload, Tensor):
            return self.payload[rows]
        expand = getattr(game, "expand_decision_reach", None)
        if not callable(expand):
            raise TypeError("game compacted decision reaches but cannot expand them")
        return expand(self.payload, rows)
