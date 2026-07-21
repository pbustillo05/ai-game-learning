"""Preallocated device-resident structure-of-arrays poker state storage."""

from __future__ import annotations

import torch
from torch import Tensor

from pbg.core.storage import (
    PagedTensorArena,
    RecyclableTensorPool,
    TensorFieldSchema,
    TensorSchema,
)


def _history_jump_levels(max_rollout_steps: int) -> int:
    """Return binary-lift columns needed for every legal ancestor distance."""

    return max(1, int(max_rollout_steps).bit_length())


class _PagedColumn:
    """Tensor-indexing facade over one page-backed structure-of-arrays field."""

    def __init__(self, arena: PagedTensorArena, name: str) -> None:
        self.arena = arena
        self.name = name
        self.spec = arena.schema.by_name[name]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.arena.size, *self.spec.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.dtype

    @property
    def device(self) -> torch.device:
        return self.arena.device

    def numel(self) -> int:
        return self.arena.size * self.spec.elements_per_row

    def _rows(self, selector) -> tuple[Tensor, tuple[int, ...], bool]:
        if isinstance(selector, slice):
            start, stop, step = selector.indices(self.arena.size)
            rows = torch.arange(start, stop, step, device=self.device)
            return rows, (rows.numel(),), False
        if isinstance(selector, int):
            row = selector if selector >= 0 else self.arena.size + selector
            return torch.tensor([row], dtype=torch.int64, device=self.device), (), True
        rows = torch.as_tensor(selector, device=self.device)
        if rows.dtype == torch.bool:
            if rows.ndim != 1 or rows.numel() != self.arena.size:
                raise IndexError("paged boolean row selector has the wrong extent")
            rows = rows.nonzero(as_tuple=False).squeeze(1)
        else:
            rows = rows.to(torch.int64)
        return rows.reshape(-1), tuple(rows.shape), False

    @staticmethod
    def _select(full: Tensor, row_shape: tuple[int, ...], rest: tuple) -> Tensor:
        if not rest:
            return full
        first = rest[0]
        if isinstance(first, Tensor) and tuple(first.shape) == row_shape:
            flat = full.reshape(-1, *full.shape[len(row_shape) :])
            selected = flat[
                torch.arange(flat.shape[0], device=flat.device),
                first.to(device=flat.device, dtype=torch.int64).reshape(-1),
            ]
            selected = selected.reshape(*row_shape, *selected.shape[1:])
            return selected[(slice(None),) * len(row_shape) + rest[1:]]
        return full[(slice(None),) * len(row_shape) + rest]

    @staticmethod
    def _assign(
        full: Tensor, row_shape: tuple[int, ...], rest: tuple, value
    ) -> None:
        if not rest:
            full[...] = value
            return
        first = rest[0]
        if isinstance(first, Tensor) and tuple(first.shape) == row_shape:
            flat = full.reshape(-1, *full.shape[len(row_shape) :])
            positions = torch.arange(flat.shape[0], device=flat.device)
            columns = first.to(device=flat.device, dtype=torch.int64).reshape(-1)
            if len(rest) == 1:
                flat[positions, columns] = value
            else:
                selected = flat[positions, columns]
                selected[(slice(None),) * len(row_shape) + rest[1:]] = value
                flat[positions, columns] = selected
            return
        full[(slice(None),) * len(row_shape) + rest] = value

    def __getitem__(self, key):
        parts = key if isinstance(key, tuple) else (key,)
        rows, row_shape, scalar = self._rows(parts[0])
        full = self.arena.gather(self.name, rows)
        if scalar:
            full = full[0]
            row_shape = ()
        else:
            full = full.reshape(*row_shape, *self.spec.shape)
        return self._select(full, row_shape, parts[1:])

    def __setitem__(self, key, value) -> None:
        parts = key if isinstance(key, tuple) else (key,)
        rows, row_shape, scalar = self._rows(parts[0])
        full = self.arena.gather(self.name, rows)
        if scalar:
            shaped = full[0]
            self._assign(shaped, (), parts[1:], value)
            full = shaped.reshape(1, *self.spec.shape)
        else:
            shaped = full.reshape(*row_shape, *self.spec.shape)
            self._assign(shaped, row_shape, parts[1:], value)
            full = shaped.reshape(rows.numel(), *self.spec.shape)
        self.arena.write(rows, {self.name: full})


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
        capacity: int,
        max_rollout_steps: int,
        device: torch.device | str,
        n_players: int = 6,
        page_rows: int = 256,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if (
            isinstance(max_rollout_steps, bool)
            or not isinstance(max_rollout_steps, int)
            or max_rollout_steps <= 0
        ):
            raise ValueError("max_rollout_steps must be a positive integer")
        if isinstance(n_players, bool) or not isinstance(n_players, int) or not 2 <= n_players <= 9:
            raise ValueError("n_players must be an integer in [2,9]")
        self.maximum_rows = capacity
        self.max_rollout_steps = max_rollout_steps
        self.n_players = n_players
        self.device = torch.device(device)
        per_player = (n_players,)
        fields = [
            *(TensorFieldSchema(name, torch.float32, per_player, alignment=16) for name in ("stacks", "committed_street", "committed_total")),
            TensorFieldSchema("status", torch.int8, per_player),
            TensorFieldSchema("acted", torch.bool, per_player),
            TensorFieldSchema("reopen_mask", torch.bool, per_player),
            *(TensorFieldSchema(name, torch.int32, alignment=4) for name in ("actor", "street", "button", "n_raises", "phase")),
            TensorFieldSchema("board", torch.int16, (5,), fill=-1, alignment=2),
            TensorFieldSchema("last_full_raise", torch.float32, alignment=4),
            TensorFieldSchema("current_bet", torch.float32, alignment=4),
            TensorFieldSchema("history_jump", torch.int32, (_history_jump_levels(max_rollout_steps),), fill=-1, alignment=4),
            TensorFieldSchema("history_event", torch.int32, fill=-1, alignment=4),
            TensorFieldSchema("history_actor", torch.int8, fill=-1),
            TensorFieldSchema("history_street", torch.int8, fill=-1),
            TensorFieldSchema("history_action", torch.int8, fill=-1),
            *(TensorFieldSchema(name, torch.float32, alignment=4) for name in ("history_amount", "history_pot_before", "history_call_amount", "history_current_bet", "history_min_raise_to", "history_max_raise_to")),
            TensorFieldSchema("history_length", torch.int32, alignment=4),
            TensorFieldSchema("fold_payoff", torch.float32, per_player, alignment=16),
            TensorFieldSchema("board_at_fold", torch.uint8, (n_players, 5), fill=255),
            TensorFieldSchema("fold_line_meta", torch.float32, (n_players, 8), alignment=16),
            TensorFieldSchema("partition_cache_key", torch.int32, fill=-1, alignment=4),
        ]
        self.schema = TensorSchema(
            "poker_live_state",
            tuple(fields),
            chunk_header_bytes=64,
            allocator_slack_bytes=64,
        )
        self.pool = RecyclableTensorPool(
            self.schema,
            page_rows,
            self.device,
            label="poker_live_state",
            maximum_rows=capacity,
        )
        self.arena = self.pool.arena
        for field in self.schema.fields:
            setattr(self, field.name, _PagedColumn(self.arena, field.name))

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

        self.pool.release_rows(handles.to(torch.int64))

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
        dst = destination.to(torch.int64)
        self.history_jump[dst, 0] = source.to(torch.int32)
        for level in range(1, self.history_jump.shape[1]):
            previous = self.history_jump[dst, level - 1]
            safe = previous.clamp_min(0).to(torch.int64)
            ancestor = self.history_jump[safe, level - 1]
            self.history_jump[dst, level] = torch.where(
                previous >= 0, ancestor, -torch.ones_like(ancestor)
            )
        self.history_event[dst] = -1
        self.history_actor[dst] = -1
        self.history_street[dst] = -1
        self.history_action[dst] = -1
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
        # Per-player fields: three fp32 commitments/stacks, one fp32 fold
        # payoff, one int8 status, and two bool flags.
        per_row = n_players * (4 * 4 + 1 + 1 + 1 + 5 + 8 * 4)
        # Seven int32 state/history/cache fields, board, two public arithmetic
        # floats, the event id, a logarithmic ancestor table, and payload.
        per_row += 7 * 4 + 5 * 2 + 2 * 4 + 4
        per_row += _history_jump_levels(max_rollout_steps) * 4 + 3 + 6 * 4
        return capacity * per_row + 4  # allocation cursor
