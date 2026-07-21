"""Paged recyclable public states with separate immutable history rows."""

from __future__ import annotations

import torch
from torch import Tensor

from pbg.core.storage import (
    PagedTensorColumn,
    RecyclableTensorPool,
    TensorFieldSchema,
    TensorSchema,
)
from pbg.core.execution import ByteAccountant


def _history_jump_levels(max_rollout_steps: int) -> int:
    """Return binary-lift columns needed for every legal ancestor distance."""

    return max(1, int(max_rollout_steps).bit_length())


def poker_live_state_schema(n_players: int) -> TensorSchema:
    per_player = (n_players,)
    return TensorSchema(
        "poker_live_state",
        (
            *(
                TensorFieldSchema(name, torch.float32, per_player, alignment=16)
                for name in ("stacks", "committed_street", "committed_total")
            ),
            TensorFieldSchema("status", torch.int8, per_player),
            TensorFieldSchema("acted", torch.bool, per_player),
            TensorFieldSchema("reopen_mask", torch.bool, per_player),
            *(
                TensorFieldSchema(name, torch.int32, alignment=4)
                for name in ("actor", "street", "button", "n_raises", "phase")
            ),
            TensorFieldSchema("board", torch.int16, (5,), fill=-1, alignment=2),
            TensorFieldSchema("last_full_raise", torch.float32, alignment=4),
            TensorFieldSchema("current_bet", torch.float32, alignment=4),
            TensorFieldSchema("history_ref", torch.int32, fill=-1, alignment=4),
            TensorFieldSchema("history_length", torch.int32, alignment=4),
            TensorFieldSchema("fold_payoff", torch.float32, per_player, alignment=16),
            TensorFieldSchema(
                "board_at_fold", torch.uint8, (n_players, 5), fill=255
            ),
            TensorFieldSchema(
                "fold_line_meta", torch.float32, (n_players, 8), alignment=16
            ),
            TensorFieldSchema(
                "partition_cache_key", torch.int32, fill=-1, alignment=4
            ),
        ),
        chunk_header_bytes=64,
        allocator_slack_bytes=64,
    )


def poker_public_history_schema(max_rollout_steps: int) -> TensorSchema:
    return TensorSchema(
        "poker_public_history",
        (
            TensorFieldSchema("history_refcount", torch.int32, alignment=4),
            TensorFieldSchema(
                "history_jump",
                torch.int32,
                (_history_jump_levels(max_rollout_steps),),
                fill=-1,
                alignment=4,
            ),
            TensorFieldSchema("history_event", torch.int32, fill=-1, alignment=4),
            TensorFieldSchema("history_actor", torch.int8, fill=-1),
            TensorFieldSchema("history_street", torch.int8, fill=-1),
            TensorFieldSchema("history_action", torch.int8, fill=-1),
            *(
                TensorFieldSchema(name, torch.float32, alignment=4)
                for name in (
                    "history_amount",
                    "history_pot_before",
                    "history_call_amount",
                    "history_current_bet",
                    "history_min_raise_to",
                    "history_max_raise_to",
                )
            ),
        ),
        chunk_header_bytes=64,
        allocator_slack_bytes=64,
    )



class PokerTensorStateStore:
    """Store realized public poker states in byte-measurable pages.

    Args:
        capacity: Maximum number of state rows.
        max_rollout_steps: Maximum public action count and ancestor-gather bound.
        device: Owning Torch device.
        n_players: Number of table seats ``P``.

    State handles are int32 ``[B]`` row indices. During the compatibility
    rollout allocation is monotone; later rollout phases consume this same
    schema through recyclable live-slot ownership. There is deliberately no
    Python state table.
    """

    def __init__(
        self,
        max_rollout_steps: int,
        device: torch.device | str,
        n_players: int = 6,
        page_rows: int = 256,
    ) -> None:
        if (
            isinstance(max_rollout_steps, bool)
            or not isinstance(max_rollout_steps, int)
            or max_rollout_steps <= 0
        ):
            raise ValueError("max_rollout_steps must be a positive integer")
        if isinstance(n_players, bool) or not isinstance(n_players, int) or not 2 <= n_players <= 9:
            raise ValueError("n_players must be an integer in [2,9]")
        self.max_rollout_steps = max_rollout_steps
        self.n_players = n_players
        self.device = torch.device(device)
        self.schema = poker_live_state_schema(n_players)
        self.pool = RecyclableTensorPool(
            self.schema,
            page_rows,
            self.device,
            label="poker_live_state",
        )
        self.arena = self.pool.arena
        for field in self.schema.fields:
            setattr(self, field.name, PagedTensorColumn(self.arena, field.name))
        self.history_schema = poker_public_history_schema(max_rollout_steps)
        self.history_pool = RecyclableTensorPool(
            self.history_schema,
            page_rows,
            self.device,
            label="poker_public_history",
        )
        self.history_arena = self.history_pool.arena
        for field in self.history_schema.fields:
            setattr(
                self,
                field.name,
                PagedTensorColumn(self.history_arena, field.name),
            )

    @property
    def capacity(self) -> int:
        return self.arena.capacity_rows

    @property
    def allocated_capacity_bytes(self) -> int:
        return self.arena.allocated_capacity_bytes

    @property
    def page_slack_bytes(self) -> int:
        return self.arena.page_slack_bytes

    @property
    def size(self) -> Tensor:
        return torch.tensor(self.arena.size, dtype=torch.int32, device=self.device)

    @property
    def history_parent(self) -> Tensor:
        return self.history_jump[:, 0]

    def reset(self, *, release_pages: bool = False) -> None:
        """Reset the device allocation cursor without reallocating storage."""

        self.pool.reset(release_pages=release_pages)
        self.history_pool.reset(release_pages=release_pages)
        if not release_pages:
            # CUDA history packing uses a fixed configured width.  Masked
            # padding therefore still needs one valid gather row even when a
            # root has no public history.  Row zero is the immutable sentinel;
            # real history records start at one.
            sentinel = self.history_pool.allocate(1)
            self.history_refcount[sentinel] = 0

    def bind_accountant(self, accountant: ByteAccountant | None) -> None:
        """Bind empty state/history page owners to one rollout byte ledger."""

        for arena in (self.arena, self.history_arena):
            if arena.page_count or arena.size:
                raise RuntimeError("rollout accountant must bind before state reset")
            arena.accountant = accountant

    def require_successor_capacity(self, count: int) -> None:
        """Admit recyclable state and history slots for one complete child wave."""

        self.pool.require_capacity(count)
        self.history_pool.require_capacity(count)

    def allocate(self, requested: Tensor) -> Tensor:
        """Allocate requested rows and return int32 handles ``[B]``.

        ``requested`` is bool ``[B]``.  Unrequested positions receive ``-1``;
        requested positions are assigned in input order by a device prefix sum.
        """

        if requested.ndim != 1 or requested.dtype != torch.bool:
            raise TypeError("requested must be bool [B]")
        if requested.device != self.device:
            raise ValueError("requested must be on the state-store device")
        offsets = torch.cumsum(requested.to(torch.int32), dim=0) - 1  # [B]
        count = int(requested.sum())
        if count == 0:
            return torch.full_like(offsets, -1, dtype=torch.int32)
        allocated = self.pool.allocate(count).to(torch.int32)
        safe = offsets.clamp_min(0).to(torch.int64)
        handles = torch.where(requested, allocated[safe], -torch.ones_like(offsets))
        return handles.to(torch.int32)  # [B]

    def allocate_all(self, count: int) -> Tensor:
        """Allocate ``count`` rows and return int32 handles ``[B]``."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        return self.pool.allocate(count).to(torch.int32)

    def release_rows(self, handles: Tensor) -> None:
        """Release physical rows after their final state/history consumer."""

        rows = handles.to(torch.int64)
        history = self.history_ref[rows].detach().clone()
        self.pool.release_rows(rows)
        self._release_history_rows(history)

    def _retain_history_rows(self, handles: Tensor) -> None:
        valid = handles[handles >= 0].to(torch.int64)
        if not valid.numel():
            return
        unique, counts = torch.unique(valid, sorted=False, return_counts=True)
        self.history_refcount[unique] += counts.to(torch.int32)

    def _release_history_rows(self, handles: Tensor) -> None:
        """Drop live-state references and recursively reclaim dead ancestors."""

        frontier = handles[handles >= 0].to(torch.int64)
        for _ in range(self.max_rollout_steps + 1):
            if not frontier.numel():
                return
            unique, counts = torch.unique(frontier, sorted=False, return_counts=True)
            remaining = self.history_refcount[unique] - counts.to(torch.int32)
            torch._assert_async(
                (remaining >= 0).all(), "history reference accounting underflow"
            )
            self.history_refcount[unique] = remaining
            dead = unique[remaining == 0]
            if not dead.numel():
                return
            parents = self.history_jump[dead, 0].detach().clone()
            self.history_pool.release_rows(dead)
            frontier = parents[parents >= 0].to(torch.int64)
        if frontier.numel():
            raise RuntimeError("history release exceeded the configured path bound")

    def copy_rows(self, source: Tensor, destination: Tensor) -> None:
        """Copy state rows from ``source`` to ``destination`` handles ``[B]``."""

        if source.shape != destination.shape or source.ndim != 1:
            raise ValueError("source and destination must have shape [B]")
        if source.device != self.device or destination.device != self.device:
            raise ValueError("handles must be on the state-store device")
        src = source.to(torch.int64)
        dst = destination.to(torch.int64)
        for name in (
            "stacks",
            "committed_street",
            "committed_total",
            "status",
            "acted",
            "reopen_mask",
            "actor",
            "street",
            "button",
            "n_raises",
            "phase",
            "board",
            "last_full_raise",
            "current_bet",
            "history_ref",
            "history_length",
            "fold_payoff",
            "board_at_fold",
            "fold_line_meta",
            "partition_cache_key",
        ):
            value = getattr(self, name)
            value[dst] = value[src]

    def append_copies(self, source: Tensor) -> Tensor:
        """Allocate and copy one successor row per source handle ``[B]``."""

        destination = self.allocate_all(source.shape[0])  # [B]
        self.copy_rows(source, destination)
        src = source.to(torch.int64)
        dst = destination.to(torch.int64)
        history = self.history_pool.allocate(source.numel()).to(torch.int32)
        self.history_ref[dst] = history
        history_rows = history.to(torch.int64)
        parent_history = self.history_ref[src].to(torch.int32)
        self.history_refcount[history_rows] = 1
        self._retain_history_rows(parent_history)
        self.history_jump[history_rows, 0] = parent_history
        for level in range(1, self.history_jump.shape[1]):
            previous = self.history_jump[history_rows, level - 1]
            safe = previous.clamp_min(0).to(torch.int64)
            ancestor = self.history_jump[safe, level - 1]
            self.history_jump[history_rows, level] = torch.where(
                previous >= 0, ancestor, -torch.ones_like(ancestor)
            )
        self.history_event[history_rows] = -1
        self.history_actor[history_rows] = -1
        self.history_street[history_rows] = -1
        self.history_action[history_rows] = -1
        return destination

    @staticmethod
    def storage_nbytes(
        capacity: int, n_players: int, max_rollout_steps: int = 1
    ) -> int:
        """Return transitional forecast bytes for ``capacity`` realized rows."""

        if capacity < 0 or n_players <= 0:
            raise ValueError("capacity must be nonnegative and n_players positive")
        if max_rollout_steps <= 0:
            raise ValueError("max_rollout_steps must be positive")
        chunks = int(capacity > 0)
        return poker_live_state_schema(n_players).capacity_bytes(
            capacity, chunks=chunks
        ) + poker_public_history_schema(max_rollout_steps).capacity_bytes(
            capacity, chunks=chunks
        )
