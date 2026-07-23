"""Common-order terminal agreement for rank-local mandatory failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import Tensor


T = TypeVar("T")


class DistributedPhaseFailure(RuntimeError):
    """A mandatory local phase failed, so every rank must stop this iteration."""

    def __init__(self, phase: str, failing_rank: int, local_detail: str | None = None):
        super().__init__(
            f"distributed iteration stopped after rank {failing_rank} failed "
            f"during {phase}"
        )
        self.phase = phase
        self.failing_rank = failing_rank
        self.local_detail = local_detail


class DistributedFailureAgreement:
    """A reusable, preallocated NCCL control agreement.

    Every rank calls :meth:`run` for the same phases in the same order.  Local
    work is allowed to fail independently, but no rank proceeds past the phase
    until an all-reduced first-failing rank proves that every peer succeeded.
    The control tensor is allocated before iteration work so an OOM path does
    not need another device allocation merely to tell peers to stop.
    """

    def __init__(self, device: torch.device, *, group: object | None = None) -> None:
        self.device = torch.device(device)
        self.group = group
        self._first_failure = torch.empty((), dtype=torch.int64, device=self.device)

    @property
    def distributed(self) -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(self.group) > 1
        )

    def _agree(self, phase: str, local_detail: str | None) -> None:
        if not self.distributed:
            if local_detail is not None:
                raise DistributedPhaseFailure(phase, 0, local_detail)
            return
        rank = dist.get_rank(self.group)
        world = dist.get_world_size(self.group)
        self._first_failure.fill_(rank if local_detail is not None else world)
        dist.all_reduce(self._first_failure, op=dist.ReduceOp.MIN, group=self.group)
        failing_rank = int(self._first_failure)
        if failing_rank < world:
            raise DistributedPhaseFailure(
                phase,
                failing_rank,
                local_detail if rank == failing_rank else None,
            )

    def run(self, phase: str, operation: Callable[[], T]) -> T:
        """Run local work, then make its terminal disposition rank-common."""

        if not self.distributed:
            return operation()
        result: T | None = None
        local_detail: str | None = None
        try:
            result = operation()
        except DistributedPhaseFailure:
            # A nested phase already made this disposition rank-common.  Do
            # not perform a second agreement or replace the original rank.
            raise
        except Exception as exc:  # noqa: BLE001 - mandatory failures are terminal
            local_detail = f"{type(exc).__name__}: {exc}"
        self._agree(phase, local_detail)
        return result  # type: ignore[return-value]

    def check(self, phase: str, local_detail: str | None = None) -> None:
        """Agree on an already-computed local terminal status."""

        self._agree(phase, local_detail)

    def run_retryable(
        self,
        phase: str,
        operation: Callable[[], T],
        *,
        retry_types: tuple[type[BaseException], ...],
    ) -> tuple[T | None, BaseException | None, int | None]:
        """Run one common attempt, distinguishing retryable from terminal failure.

        The single preallocated control scalar encodes terminal ranks before
        retry ranks, so every rank makes the same disposition without trying
        to allocate communication storage on an OOM path.
        """

        if not self.distributed:
            try:
                return operation(), None, None
            except retry_types as exc:
                return None, exc, 0

        rank = dist.get_rank(self.group)
        world = dist.get_world_size(self.group)
        result: T | None = None
        retry: BaseException | None = None
        terminal_detail: str | None = None
        try:
            result = operation()
        except retry_types as exc:
            retry = exc
        except DistributedPhaseFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - common terminal disposition
            terminal_detail = f"{type(exc).__name__}: {exc}"

        code = (
            rank
            if terminal_detail is not None
            else world + rank
            if retry is not None
            else 2 * world
        )
        self._first_failure.fill_(code)
        dist.all_reduce(self._first_failure, op=dist.ReduceOp.MIN, group=self.group)
        common = int(self._first_failure)
        if common < world:
            raise DistributedPhaseFailure(
                phase,
                common,
                terminal_detail if rank == common else None,
            )
        if common < 2 * world:
            return None, retry if rank == common else None, common - world
        return result, None, None
