"""Preallocated device-resident structure-of-arrays poker state storage."""

from __future__ import annotations

import torch
from torch import Tensor


class PokerTensorStateStore:
    """Store public poker states in preallocated tensors.

    Args:
        capacity: Maximum number of state rows.
        max_rollout_steps: Width of ``action_history``.
        device: Owning Torch device.
        n_players: Number of table seats ``P``.

    State handles are int32 ``[B]`` row indices.  Allocation is monotone and
    device-side: a scalar tensor cursor plus an inclusive prefix sum assigns
    each requested row.  There is deliberately no Python state table or free
    list.
    """

    def __init__(
        self,
        capacity: int,
        max_rollout_steps: int,
        device: torch.device | str,
        n_players: int = 6,
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
        self.capacity = capacity
        self.max_rollout_steps = max_rollout_steps
        self.n_players = n_players
        self.device = torch.device(device)
        shape = (capacity, n_players)
        self.stacks = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.committed_street = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.committed_total = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.status = torch.empty(shape, dtype=torch.int8, device=self.device)
        self.acted = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.reopen_mask = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.actor = torch.empty(capacity, dtype=torch.int32, device=self.device)
        self.street = torch.empty(capacity, dtype=torch.int32, device=self.device)
        self.button = torch.empty(capacity, dtype=torch.int32, device=self.device)
        self.n_raises = torch.empty(capacity, dtype=torch.int32, device=self.device)
        self.phase = torch.empty(capacity, dtype=torch.int32, device=self.device)
        self.board = torch.full((capacity, 5), -1, dtype=torch.int16, device=self.device)
        self.last_full_raise = torch.empty(capacity, dtype=torch.float32, device=self.device)
        self.current_bet = torch.empty(capacity, dtype=torch.float32, device=self.device)
        self.action_history = torch.full(
            (capacity, max_rollout_steps), -1, dtype=torch.int32, device=self.device
        )
        self.history_length = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        self.fold_payoff = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.size = torch.zeros((), dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        """Reset the device allocation cursor without reallocating storage."""

        self.size.zero_()

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
        count = requested.to(torch.int32).sum()
        end = self.size + count
        torch._assert(end <= self.capacity, "poker tensor state capacity exceeded")
        handles = torch.where(requested, self.size + offsets, -torch.ones_like(offsets))
        self.size.add_(count)
        return handles.to(torch.int32)  # [B]

    def allocate_all(self, count: int) -> Tensor:
        """Allocate ``count`` rows and return int32 handles ``[B]``."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        requested = torch.ones(count, dtype=torch.bool, device=self.device)  # [B]
        return self.allocate(requested)

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
            "action_history",
            "history_length",
            "fold_payoff",
        ):
            value = getattr(self, name)
            value[dst] = value[src]

    def append_copies(self, source: Tensor) -> Tensor:
        """Allocate and copy one successor row per source handle ``[B]``."""

        destination = self.allocate_all(source.shape[0])  # [B]
        self.copy_rows(source, destination)
        return destination

