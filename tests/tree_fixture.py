"""Small CPU tree builder for tests that need hand-authored topology."""

from __future__ import annotations

import torch

from pbg.core.tree import SampledTree


class DiagnosticTreeBuilder:
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
        if tree_id is None:
            tree_id = sum(parent == -1 for parent in self.parents)
        return self.add_node(-1, state_ref, owner, tree_id=tree_id)

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
        return self.add_node(parent, state_ref, owner, event=event, nu=nu)

    def finalize(self) -> SampledTree:
        count = len(self.parents)
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

        depth_p = torch.zeros((count, self.n_players), dtype=torch.int32)
        depth_player = torch.zeros(count, dtype=torch.int32)
        path_weight = torch.ones(count, dtype=torch.float32)
        incoming_nu = {
            child: weight
            for edges in self.children
            for child, _, weight in edges
        }
        for node, parent in enumerate(self.parents):
            if parent >= 0:
                depth_p[node] = depth_p[parent]
                depth_player[node] = depth_player[parent]
                path_weight[node] = path_weight[parent] * incoming_nu[node]
            owner = self.owners[node]
            if 0 <= owner < self.n_players:
                depth_p[node, owner] += 1
                depth_player[node] += 1

        tree = SampledTree(
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
        tree.validate()
        return tree
