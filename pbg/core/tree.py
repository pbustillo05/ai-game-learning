"""Scheduler topology plus exact sealed leaf and decision rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, PROBE_FOLD, TERMINAL
from pbg.core.storage import (
    ByteAccountant,
    PagedTensorArena,
    PagedTensorColumn,
    SealMetrics,
    TensorFieldSchema,
    TensorSchema,
    admit_arena_growth,
)


@dataclass
class SampledTree:
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
    # Nominal randomized-rounded offspring and frozen rollout controller
    # quantities. They are absent on sequential or manually built test trees.
    nominal_child_count: Tensor | None = None
    branching_beta: Tensor | None = None
    # Root-refill metadata. The protected prefix of root children and nodes
    # belongs to the ordinary beta law; later rows are size-triggered samples.
    root_pre_refill_child_count: int | None = None
    pre_refill_node_count: int | None = None

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
        if self.nominal_child_count is not None and (
            self.nominal_child_count.shape != (n,)
            or self.nominal_child_count.dtype != torch.int32
            or self.nominal_child_count.device != self.parent.device
            or bool((self.nominal_child_count < 0).any())
        ):
            raise ValueError("nominal_child_count must be device-local nonnegative int32 [N]")
        if self.branching_beta is not None and (
            self.branching_beta.shape != (n,)
            or self.branching_beta.dtype != torch.float32
            or self.branching_beta.device != self.parent.device
            or bool((~torch.isfinite(self.branching_beta)).any())
            or bool((self.branching_beta < 0).any())
        ):
            raise ValueError("branching_beta must be device-local finite fp32 [N]")
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
        if (self.root_pre_refill_child_count is None) != (
            self.pre_refill_node_count is None
        ):
            raise ValueError("root refill metadata must be present as a complete pair")
        if self.root_pre_refill_child_count is not None:
            if root_rows.numel() != 1:
                raise ValueError("root refill metadata requires exactly one root")
            root = root_rows[0]
            root_degree = int(counts[root])
            protected = self.root_pre_refill_child_count
            if (
                isinstance(protected, bool)
                or not isinstance(protected, int)
                or not 1 <= protected <= root_degree
            ):
                raise ValueError(
                    "root_pre_refill_child_count must be between one and the root degree"
                )
        if self.pre_refill_node_count is not None:
            protected_nodes = self.pre_refill_node_count
            if (
                isinstance(protected_nodes, bool)
                or not isinstance(protected_nodes, int)
                or not 1 <= protected_nodes <= n
            ):
                raise ValueError("pre_refill_node_count must lie within the stored tree")
            assert self.root_pre_refill_child_count is not None
            root = root_rows[0]
            start = int(self.child_ptr[root])
            stop = int(self.child_ptr[root + 1])
            root_children = self.child_node[start:stop].to(torch.int64)
            protected = self.root_pre_refill_child_count
            if protected_nodes < 1 + protected:
                raise ValueError("the pre-refill tree cannot contain its protected root cohort")
            if bool((root_children[:protected] >= protected_nodes).any()) or bool(
                (root_children[protected:] < protected_nodes).any()
            ):
                raise ValueError(
                    "root children must preserve the ordinary prefix before refill children"
                )


class DiagnosticTreeBuilder:
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

    def finalize(self) -> SampledTree:
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

        result = SampledTree(
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



class PagedTreeSource:
    """Append-only page-backed scheduler topology and public-state handles."""

    @staticmethod
    def node_schema(n_players: int) -> TensorSchema:
        return TensorSchema(
            "rollout_topology_nodes",
            (
                TensorFieldSchema("parent", torch.int32, fill=-1, alignment=4),
                TensorFieldSchema("tree_id", torch.int32, alignment=4),
                TensorFieldSchema("owner", torch.int32, alignment=4),
                TensorFieldSchema("state_ref", torch.int32, fill=-1, alignment=4),
                TensorFieldSchema("depth_p", torch.int32, (n_players,), alignment=16),
                TensorFieldSchema("depth_player", torch.int32, alignment=4),
                TensorFieldSchema("path_weight", torch.float32, alignment=4),
                TensorFieldSchema("child_count", torch.int32, alignment=4),
            ),
            chunk_header_bytes=64,
            allocator_slack_bytes=64,
        )

    @staticmethod
    def edge_schema() -> TensorSchema:
        return TensorSchema(
            "rollout_topology_edges",
            (
                TensorFieldSchema("edge_parent", torch.int32, alignment=4),
                TensorFieldSchema("edge_ordinal", torch.int32, alignment=4),
                TensorFieldSchema("child_node", torch.int32, alignment=4),
                TensorFieldSchema("child_event", torch.int32, alignment=4),
                TensorFieldSchema("edge_weight", torch.float32, alignment=4),
            ),
            chunk_header_bytes=64,
            allocator_slack_bytes=64,
        )

    def __init__(
        self,
        root_count: int,
        n_players: int,
        device: torch.device | str,
        *,
        page_rows: int = 256,
        accountant: ByteAccountant | None = None,
    ) -> None:
        if root_count < 0 or n_players <= 0 or page_rows <= 0:
            raise ValueError("paged tree dimensions must be nonnegative/positive")
        self.root_count = int(root_count)
        self.n_players = int(n_players)
        self.device = torch.device(device)
        self.nodes = PagedTensorArena(
            self.node_schema(n_players),
            page_rows,
            self.device,
            accountant=accountant,
            label="rollout_topology_nodes",
        )
        self.edges = PagedTensorArena(
            self.edge_schema(),
            page_rows,
            self.device,
            accountant=accountant,
            label="rollout_topology_edges",
        )
        for name in self.nodes.schema.by_name:
            setattr(self, name, PagedTensorColumn(self.nodes, name))
        for name in self.edges.schema.by_name:
            setattr(self, name, PagedTensorColumn(self.edges, name))

    @property
    def node_count(self) -> int:
        return self.nodes.size

    @property
    def edge_count(self) -> int:
        return self.edges.size

    @property
    def allocated_capacity_bytes(self) -> int:
        return self.nodes.allocated_capacity_bytes + self.edges.allocated_capacity_bytes

    def add_roots(self, states: Tensor, owners: Tensor) -> Tensor:
        if states.shape != (self.root_count,) or owners.shape != (self.root_count,):
            raise ValueError("root tensors must have shape [root_count]")
        if states.device != self.device or owners.device != self.device:
            raise ValueError("root tensors must be on the topology device")
        ids = torch.arange(self.root_count, dtype=torch.int32, device=self.device)
        owner64 = owners.to(torch.int64)
        player = (owner64 >= 0) & (owner64 < self.n_players)
        depth = torch.zeros(
            (self.root_count, self.n_players), dtype=torch.int32, device=self.device
        )
        if self.root_count:
            depth.scatter_(
                1,
                owner64.clamp(0, self.n_players - 1)[:, None],
                player.to(torch.int32)[:, None],
            )
        rows = self.nodes.append(
            {
                "parent": torch.full_like(ids, -1),
                "tree_id": ids,
                "owner": owners.to(torch.int32),
                "state_ref": states.to(torch.int32),
                "depth_p": depth,
                "depth_player": player.to(torch.int32),
                "path_weight": torch.ones(self.root_count, device=self.device),
                "child_count": torch.zeros(
                    self.root_count, dtype=torch.int32, device=self.device
                ),
            }
        )
        return rows.to(torch.int32)

    append_roots = add_roots

    def require_child_capacity(self, count: int | Tensor) -> None:
        """Validate a wave count; append admits its realized page extents."""

        if isinstance(count, Tensor):
            if count.numel() != 1:
                raise ValueError("child count must be scalar")
            torch._assert_async(count >= 0, "child count must be nonnegative")
        elif isinstance(count, bool) or int(count) < 0:
            raise ValueError("child count must be nonnegative")

    def append_children(
        self,
        parents: Tensor,
        states: Tensor,
        owners: Tensor,
        events: Tensor,
        weights: Tensor,
    ) -> Tensor:
        count = int(parents.numel())
        if any(value.shape != (count,) for value in (states, owners, events, weights)):
            raise ValueError("child append tensors must all have shape [C_new]")
        if any(value.device != self.device for value in (parents, states, owners, events, weights)):
            raise ValueError("child append tensors must be on the topology device")
        if count == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        admit_arena_growth({self.nodes: count, self.edges: count})
        parent_rows = parents.to(torch.int64)
        parent_tree = self.tree_id[parent_rows]
        parent_depth = self.depth_p[parent_rows]
        parent_player_depth = self.depth_player[parent_rows]
        parent_weight = self.path_weight[parent_rows]
        current_count = self.child_count[parent_rows]
        local_rows = torch.arange(count, dtype=torch.int32, device=self.device)
        group_start = torch.zeros(count, dtype=torch.int32, device=self.device)
        if count > 1:
            group_start[1:] = torch.where(
                parents[1:] != parents[:-1], local_rows[1:], 0
            )
        group_start = torch.cummax(group_start, dim=0).values
        local_ordinal = local_rows - group_start
        child_owner = owners.to(torch.int64)
        player = (child_owner >= 0) & (child_owner < self.n_players)
        increments = torch.zeros_like(parent_depth)
        increments.scatter_(
            1,
            child_owner.clamp(0, self.n_players - 1)[:, None],
            player.to(torch.int32)[:, None],
        )
        expected_nodes = torch.arange(
            self.node_count,
            self.node_count + count,
            dtype=torch.int32,
            device=self.device,
        )
        nodes = self.nodes.append(
            {
                "parent": parents.to(torch.int32),
                "tree_id": parent_tree.to(torch.int32),
                "owner": owners.to(torch.int32),
                "state_ref": states.to(torch.int32),
                "depth_p": parent_depth + increments,
                "depth_player": parent_player_depth + player.to(torch.int32),
                "path_weight": parent_weight * weights.float(),
                "child_count": torch.zeros(count, dtype=torch.int32, device=self.device),
            }
        ).to(torch.int32)
        if not torch.equal(nodes, expected_nodes):
            raise RuntimeError("paged topology node ids lost monotone order")
        self.edges.append(
            {
                "edge_parent": parents.to(torch.int32),
                "edge_ordinal": current_count + local_ordinal,
                "child_node": nodes,
                "child_event": events.to(torch.int32),
                "edge_weight": weights.float(),
            }
        )
        unique_parents, run_counts = torch.unique_consecutive(
            parent_rows, return_counts=True
        )
        updated = self.child_count[unique_parents] + run_counts.to(torch.int32)
        self.nodes.write(unique_parents, {"child_count": updated})
        return nodes

    def finalize(self) -> SampledTree:
        node_rows = torch.arange(self.node_count, device=self.device)
        edge_rows = torch.arange(self.edge_count, device=self.device)
        node = {name: self.nodes.gather(name, node_rows) for name in self.nodes.schema.by_name}
        edge = {name: self.edges.gather(name, edge_rows) for name in self.edges.schema.by_name}
        child_ptr = torch.empty(
            self.node_count + 1, dtype=torch.int32, device=self.device
        )
        child_ptr[0] = 0
        child_ptr[1:] = node["child_count"].cumsum(0).to(torch.int32)
        destination = (
            child_ptr[edge["edge_parent"].to(torch.int64)]
            + edge["edge_ordinal"]
        ).to(torch.int64)
        child_node = torch.empty_like(edge["child_node"])
        child_event = torch.empty_like(edge["child_event"])
        edge_weight = torch.empty_like(edge["edge_weight"])
        child_node[destination] = edge["child_node"]
        child_event[destination] = edge["child_event"]
        edge_weight[destination] = edge["edge_weight"]
        return SampledTree(
            parent=node["parent"],
            child_ptr=child_ptr,
            child_node=child_node,
            child_event=child_event,
            nu=edge_weight,
            tree_id=node["tree_id"],
            owner=node["owner"],
            depth_p=node["depth_p"],
            depth_player=node["depth_player"],
            state_ref=node["state_ref"],
            path_weight=node["path_weight"],
        )

    def release(self) -> None:
        self.nodes.release()
        self.edges.release()


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


def build_decision_alias(tree: SampledTree) -> AliasTable:
    """Build the length-biased decision law over decision-row indices."""

    p_max = tree.depth_p.shape[1]
    node_ids = ((tree.owner >= 0) & (tree.owner < p_max)).nonzero(as_tuple=False).squeeze(1)
    if node_ids.numel() == 0:
        raise ValueError("tree batch contains no player decision nodes")
    return AliasTable.build(tree.path_weight[node_ids])


@dataclass
class RolloutDecisionRows:
    """Frozen decision outputs with one exact ragged row per retained action."""

    node_ids: Tensor  # int32 [D]
    action_ptr: Tensor  # int32 [D+1]
    action_event: Tensor  # int32 [A]
    old_logp_action: Tensor  # float32 [A,X]
    old_v: Tensor  # float32 [D,X]
    decision_inputs: Any | None = None  # zero-copy-compatible natural input owner
    old_t: Tensor | None = None  # float32 [Nd]
    old_c: Tensor | None = None  # float32 [Nd]
    seal_metrics: Mapping[str, SealMetrics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_ids.ndim != 1 or self.node_ids.dtype != torch.int32:
            raise ValueError("RolloutDecisionRows.node_ids must be int32 [D]")
        if (
            self.action_ptr.shape != (self.node_ids.numel() + 1,)
            or self.action_ptr.dtype != torch.int32
            or int(self.action_ptr[0]) != 0
            or bool((self.action_ptr[1:] < self.action_ptr[:-1]).any())
        ):
            raise ValueError("RolloutDecisionRows.action_ptr must be CSR int32 [D+1]")
        actions = int(self.action_ptr[-1])
        if self.action_event.shape != (actions,) or self.action_event.dtype != torch.int32:
            raise ValueError("RolloutDecisionRows.action_event must be int32 [A]")
        if self.old_logp_action.ndim != 2 or self.old_logp_action.shape[0] != actions:
            raise ValueError("RolloutDecisionRows.old_logp_action must have shape [A,X]")
        if self.old_v.ndim != 2 or self.old_v.shape[0] != self.node_ids.numel():
            raise ValueError("RolloutDecisionRows.old_v must have shape [D,X]")
        if self.old_logp_action.dtype != torch.float32 or self.old_v.dtype != torch.float32:
            raise ValueError("RolloutDecisionRows floating tensors must be fp32")
        for name, value in (("old_t", self.old_t), ("old_c", self.old_c)):
            if value is not None and (
                value.shape != (self.node_ids.numel(),)
                or value.dtype != torch.float32
                or value.device != self.old_v.device
                or bool((~torch.isfinite(value)).any())
                or (name == "old_c" and bool((value <= 0).any()))
            ):
                raise ValueError(f"RolloutDecisionRows.{name} must be an aligned finite fp32 vector")


@dataclass
class StoppedLeafRows:
    """Exact last-consumer leaf reach rows; no discarded path can be replayed."""

    mode: str
    log_r: Tensor
    node_index: Tensor
    payload: dict[str, Tensor] | None = None
    seal_metrics: Mapping[str, SealMetrics] = field(default_factory=dict)

    def rows_for(
        self,
        tree: SampledTree,
        node_ids: Tensor,
        game: Any,
        rollout_decisions: RolloutDecisionRows,
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

    def payload_for(self, node_ids: Tensor) -> dict[str, Tensor]:
        """Return immutable stopped-leaf fields aligned to requested node ids."""

        if self.payload is None:
            raise RuntimeError("stopped leaves do not own immutable payload fields")
        requested = node_ids.to(device=self.node_index.device, dtype=torch.int64)
        stored = self.node_index.to(torch.int64)
        positions = torch.searchsorted(stored, requested)
        if stored.numel() == 0 or bool(
            (positions >= stored.numel()).any()
        ) or not torch.equal(stored[positions], requested):
            raise KeyError("one or more requested leaf payload rows are not sealed")
        return {name: value[positions.to(value.device)] for name, value in self.payload.items()}
