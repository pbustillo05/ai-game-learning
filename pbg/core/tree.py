"""Paged rollout topology used only until direct compact sealing."""

from __future__ import annotations

from typing import Mapping
import torch
from torch import Tensor

from pbg.core.storage import (
    ByteAccountant,
    PagedTensorArena,
    PagedTensorColumn,
    TensorFieldSchema,
    TensorSchema,
    admit_arena_growth,
)


class PagedRolloutTopology:
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
        """Transactionally admit node and edge pages for a logical child wave."""

        if isinstance(count, Tensor):
            if count.numel() != 1:
                raise ValueError("child count must be scalar")
            torch._assert_async(count >= 0, "child count must be nonnegative")
            count = int(count)
        elif isinstance(count, bool) or int(count) < 0:
            raise ValueError("child count must be nonnegative")
        admit_arena_growth({self.nodes: int(count), self.edges: int(count)})

    def append_children(
        self,
        parents: Tensor,
        states: Tensor,
        owners: Tensor,
        events: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Reserve and immediately commit one child batch.

        Rollout paths which split logical child work across physical subspans use
        :meth:`reserve_children` and :meth:`commit_children` separately.  This
        convenience boundary remains useful for callers whose logical and
        physical batches are identical.
        """

        count = int(parents.numel())
        if any(value.shape != (count,) for value in (states, owners, events, weights)):
            raise ValueError("child append tensors must all have shape [C_new]")
        if any(
            value.device != self.device
            for value in (parents, states, owners, events, weights)
        ):
            raise ValueError("child append tensors must be on the topology device")
        node_start = self.node_count
        edge_start = self.edge_count
        nodes, edges = self.reserve_children(parents, weights)
        self.commit_children(
            nodes,
            edges,
            states,
            owners,
            events,
            _node_start=node_start,
            _edge_start=edge_start,
        )
        return nodes

    def reserve_children(
        self,
        parents: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Publish stable child/edge addresses without materializing children."""

        count = int(parents.numel())
        if weights.shape != (count,):
            raise ValueError("child reservation tensors must all have shape [C_new]")
        if parents.device != self.device or weights.device != self.device:
            raise ValueError("child reservation tensors must be on the topology device")
        if count == 0:
            empty = torch.empty(0, dtype=torch.int32, device=self.device)
            return empty, empty
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
                "owner": torch.full(
                    (count,), -1, dtype=torch.int32, device=self.device
                ),
                "state_ref": torch.full(
                    (count,), -1, dtype=torch.int32, device=self.device
                ),
                "depth_p": parent_depth,
                "depth_player": parent_player_depth,
                "path_weight": parent_weight * weights.float(),
                "child_count": torch.zeros(count, dtype=torch.int32, device=self.device),
            }
        ).to(torch.int32)
        if not torch.equal(nodes, expected_nodes):
            raise RuntimeError("paged topology node ids lost monotone order")
        edges = self.edges.append(
            {
                "edge_parent": parents.to(torch.int32),
                "edge_ordinal": current_count + local_ordinal,
                "child_node": nodes,
                "child_event": torch.full(
                    (count,), -1, dtype=torch.int32, device=self.device
                ),
                "edge_weight": weights.float(),
            }
        ).to(torch.int32)
        unique_parents, run_counts = torch.unique_consecutive(
            parent_rows, return_counts=True
        )
        updated = self.child_count[unique_parents] + run_counts.to(torch.int32)
        self.nodes.write(unique_parents, {"child_count": updated})
        return nodes, edges

    def commit_children(
        self,
        nodes: Tensor,
        edges: Tensor,
        states: Tensor,
        owners: Tensor,
        events: Tensor,
        *,
        _node_start: int | None = None,
        _edge_start: int | None = None,
    ) -> None:
        """Fill one physical subspan of previously reserved child rows."""

        count = int(nodes.numel())
        if any(value.shape != (count,) for value in (edges, states, owners, events)):
            raise ValueError("child commit tensors must all have shape [C_subspan]")
        if any(
            value.device != self.device
            for value in (nodes, edges, states, owners, events)
        ):
            raise ValueError("child commit tensors must be on the topology device")
        if count == 0:
            return
        node_rows = nodes.to(torch.int64)
        parent_rows = self.parent[node_rows].to(torch.int64)
        parent_depth = self.depth_p[parent_rows]
        child_owner = owners.to(torch.int64)
        player = (child_owner >= 0) & (child_owner < self.n_players)
        increments = torch.zeros_like(parent_depth)
        increments.scatter_(
            1,
            child_owner.clamp(0, self.n_players - 1)[:, None],
            player.to(torch.int32)[:, None],
        )
        node_values = {
            "owner": owners.to(torch.int32),
            "state_ref": states.to(torch.int32),
            "depth_p": parent_depth + increments,
            "depth_player": self.depth_player[parent_rows]
            + player.to(torch.int32),
        }
        edge_values = {"child_event": events.to(torch.int32)}
        if (_node_start is None) != (_edge_start is None):
            raise ValueError("contiguous child commit requires both host cursors")
        if _node_start is None:
            self.nodes.write(node_rows, node_values)
            self.edges.write(edges.to(torch.int64), edge_values)
        else:
            self.nodes.write_contiguous(_node_start, node_values)
            self.edges.write_contiguous(_edge_start, edge_values)

    def release(self) -> None:
        self.nodes.release()
        self.edges.release()
