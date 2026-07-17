"""Compact decision-only sequential and contiguous-span branched topology."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _require_int32_vector(name: str, value: Tensor, length: int) -> None:
    if value.dtype != torch.int32 or value.ndim != 1 or value.numel() != length:
        raise ValueError(f"{name} must be int32 [{length}]")


@dataclass(frozen=True)
class SequentialDecisionCSR:
    """Chronological player decisions in fixed root-trajectory order."""

    trajectory_ptr: Tensor  # int32 [G+1]
    decision_partition: Tensor  # int32 [D]
    decision_row: Tensor  # int32 [D]
    leaf_row: Tensor  # int32 [G]

    @property
    def trajectoryPtr(self) -> Tensor:
        return self.trajectory_ptr

    @property
    def decisionPartition(self) -> Tensor:
        return self.decision_partition

    @property
    def decisionRow(self) -> Tensor:
        return self.decision_row

    @property
    def leafRow(self) -> Tensor:
        return self.leaf_row

    @property
    def roots(self) -> int:
        return int(self.leaf_row.numel())

    @property
    def decisions(self) -> int:
        return int(self.decision_row.numel())

    @property
    def nbytes(self) -> int:
        return 4 * (2 * self.decisions + 2 * self.roots + 1)

    def validate(self, partition_rows: dict[int, int] | None = None, leaf_rows: int | None = None) -> None:
        g, d = self.roots, self.decisions
        _require_int32_vector("trajectory_ptr", self.trajectory_ptr, g + 1)
        _require_int32_vector("decision_partition", self.decision_partition, d)
        _require_int32_vector("decision_row", self.decision_row, d)
        _require_int32_vector("leaf_row", self.leaf_row, g)
        devices = {
            self.trajectory_ptr.device,
            self.decision_partition.device,
            self.decision_row.device,
            self.leaf_row.device,
        }
        if len(devices) != 1:
            raise ValueError("sequential topology columns must share one device")
        ptr = self.trajectory_ptr.to(torch.int64)
        if int(ptr[0]) != 0 or int(ptr[-1]) != d or bool((ptr[1:] < ptr[:-1]).any()):
            raise ValueError("trajectory_ptr is not a valid decision CSR pointer")
        if partition_rows is not None:
            covered: dict[int, Tensor] = {}
            for partition, count in partition_rows.items():
                mask = self.decision_partition == int(partition)
                rows = self.decision_row[mask].to(torch.int64)
                if rows.numel() != int(count):
                    raise ValueError("sequential topology does not cover each partition row")
                if rows.numel() and not torch.equal(
                    rows.sort().values,
                    torch.arange(count, device=rows.device),
                ):
                    raise ValueError("sequential topology partition rows are not a permutation")
                covered[int(partition)] = rows
            known = torch.isin(
                self.decision_partition,
                torch.tensor(
                    tuple(partition_rows),
                    dtype=torch.int32,
                    device=self.decision_partition.device,
                ),
            )
            if bool((~known).any()):
                raise ValueError("sequential topology contains an unknown partition")
        if leaf_rows is not None:
            if g and (
                bool(((self.leaf_row < 0) | (self.leaf_row >= leaf_rows)).any())
                or self.leaf_row.unique().numel() != g
            ):
                raise ValueError("leaf_row must uniquely address one leaf per trajectory")

    @classmethod
    def build(
        cls,
        *,
        root_id: Tensor,
        decision_ordinal: Tensor,
        decision_partition: Tensor,
        decision_row: Tensor,
        leaf_root_id: Tensor,
        leaf_row: Tensor,
        root_count: int,
    ) -> "SequentialDecisionCSR":
        """Seal occurrence addresses into stable chronological CSR order."""

        d = int(root_id.numel())
        if any(value.shape != (d,) for value in (decision_ordinal, decision_partition, decision_row)):
            raise ValueError("decision occurrence columns must be matching vectors")
        if root_count < 0 or leaf_root_id.shape != (root_count,) or leaf_row.shape != (root_count,):
            raise ValueError("leaf occurrence columns must contain one row per root")
        device = root_id.device
        if any(value.device != device for value in (decision_ordinal, decision_partition, decision_row, leaf_root_id, leaf_row)):
            raise ValueError("topology source columns must share one device")
        if d and bool(
            ((root_id < 0) | (root_id >= root_count) | (decision_ordinal < 0)).any()
        ):
            raise ValueError("decision root/ordinal address is out of range")
        # Lexicographic root/ordinal key; ordinals are bounded by D, and the
        # original row order is the stable final tiebreak (duplicates reject).
        key = root_id.to(torch.int64) * (d + 1) + decision_ordinal.to(torch.int64)
        order = torch.argsort(key, stable=True)
        ordered_key = key[order]
        if d > 1 and bool((ordered_key[1:] == ordered_key[:-1]).any()):
            raise ValueError("duplicate within-trajectory decision ordinal")
        counts = torch.bincount(root_id.to(torch.int64), minlength=root_count)
        ptr = torch.empty(root_count + 1, dtype=torch.int32, device=device)
        ptr[0] = 0
        ptr[1:] = counts.cumsum(0).to(torch.int32)
        leaf_order = torch.argsort(leaf_root_id.to(torch.int64), stable=True)
        if root_count and not torch.equal(
            leaf_root_id[leaf_order].to(torch.int64), torch.arange(root_count, device=device)
        ):
            raise ValueError("leaf roots must uniquely cover the fixed root set")
        result = cls(
            trajectory_ptr=ptr,
            decision_partition=decision_partition[order].to(torch.int32),
            decision_row=decision_row[order].to(torch.int32),
            leaf_row=leaf_row[leaf_order].to(torch.int32),
        )
        result.validate(leaf_rows=root_count)
        return result


@dataclass(frozen=True)
class BranchedTreeSpan:
    """One rooted tree with implicit contiguous child edges."""

    parent: Tensor
    child_begin: Tensor
    degree: Tensor
    decision_partition: Tensor
    decision_row: Tensor
    leaf_row: Tensor

    @property
    def nodes(self) -> int:
        return int(self.parent.numel())

    @property
    def nbytes(self) -> int:
        return 6 * 4 * self.nodes

    def children(self, node: int) -> slice:
        start = int(self.child_begin[node])
        return slice(start, start + int(self.degree[node]))

    def validate(
        self,
        *,
        partition_rows: dict[int, int] | None = None,
        action_ptr: dict[int, Tensor] | None = None,
        leaf_rows: int | None = None,
    ) -> None:
        n = self.nodes
        for name in self.__dataclass_fields__:
            _require_int32_vector(name, getattr(self, name), n)
        if len({getattr(self, name).device for name in self.__dataclass_fields__}) != 1:
            raise ValueError("branched topology columns must share one device")
        if n == 0:
            raise ValueError("a branched rollout must contain its distinguished root")
        if int(self.parent[0]) != -1 or bool((self.parent[1:] < 0).any()):
            raise ValueError("branched root must be row zero and the only root")
        incoming = torch.zeros(n, dtype=torch.int32, device=self.parent.device)
        for node in range(n):
            degree = int(self.degree[node])
            begin = int(self.child_begin[node])
            if degree < 0:
                raise ValueError("branched node degree cannot be negative")
            if degree == 0:
                if begin != -1:
                    raise ValueError("leaf child_begin must use sentinel -1")
                if int(self.leaf_row[node]) < 0:
                    raise ValueError("every topology leaf must own a leaf row")
            else:
                if begin <= node or begin + degree > n:
                    raise ValueError("branched child span is not a valid forward contiguous span")
                children = torch.arange(begin, begin + degree, device=self.parent.device)
                if not bool((self.parent[children] == node).all()):
                    raise ValueError("branched parent and child span columns disagree")
                incoming[children] += 1
                if int(self.leaf_row[node]) != -1:
                    raise ValueError("nonleaf node cannot own a leaf row")
            partition = int(self.decision_partition[node])
            row = int(self.decision_row[node])
            if (partition == -1) != (row == -1):
                raise ValueError("decision partition/row sentinels must agree")
            if partition >= 0 and degree == 0:
                raise ValueError("a player decision must have sampled children")
        if int(incoming[0]) != 0 or bool((incoming[1:] != 1).any()):
            raise ValueError("every nonroot branched node must have one incoming edge")
        if leaf_rows is not None:
            selected = self.leaf_row[self.leaf_row >= 0].to(torch.int64)
            if selected.numel() != leaf_rows or (
                selected.numel()
                and not torch.equal(selected.sort().values, torch.arange(leaf_rows, device=selected.device))
            ):
                raise ValueError("branched leaf rows must be covered exactly once")
        if partition_rows is not None:
            known = torch.isin(
                self.decision_partition[self.decision_partition >= 0],
                torch.tensor(
                    tuple(partition_rows),
                    dtype=torch.int32,
                    device=self.decision_partition.device,
                ),
            )
            if bool((~known).any()):
                raise ValueError("branched topology contains an unknown partition")
            for partition, count in partition_rows.items():
                rows = self.decision_row[
                    self.decision_partition == int(partition)
                ].to(torch.int64)
                if rows.numel() != count or (
                    count and not torch.equal(rows.sort().values, torch.arange(count, device=rows.device))
                ):
                    raise ValueError("branched decision rows are not covered exactly once")
                if action_ptr is not None:
                    ptr = action_ptr[int(partition)].to(torch.int64)
                    if ptr.shape != (count + 1,):
                        raise ValueError("branched action pointer has wrong extent")
                    node_ids = (self.decision_partition == int(partition)).nonzero(as_tuple=False).squeeze(1)
                    decision_rows = self.decision_row[node_ids].to(torch.int64)
                    degrees = self.degree[node_ids].to(torch.int64)
                    if not torch.equal(ptr[decision_rows + 1] - ptr[decision_rows], degrees):
                        raise ValueError("player action and child span degrees differ")
