"""Signal-safe advance notice with synchronized iteration-boundary decisions."""

from __future__ import annotations

from dataclasses import dataclass
import signal
import threading
from types import FrameType
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PreemptionNotice:
    requested: bool
    signal_number: int | None
    signal_name: str | None


class PreemptionController:
    """Record signals in handlers and act only at audited safe boundaries."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._signal_number: int | None = None
        self._previous: dict[int, Any] = {}

    def _handler(
        self, signal_number: int, _frame: FrameType | None
    ) -> None:
        if self._signal_number is None:
            self._signal_number = int(signal_number)
        self._event.set()

    def install(self) -> None:
        """Install advance/termination handlers once on the main thread."""

        if self._previous:
            raise RuntimeError("preemption handlers are already installed")
        signals = [signal.SIGTERM]
        advance = getattr(signal, "SIGUSR1", None)
        if advance is not None:
            signals.insert(0, advance)
        for selected in signals:
            self._previous[int(selected)] = signal.getsignal(selected)
            signal.signal(selected, self._handler)

    def restore(self) -> None:
        for signal_number, previous in self._previous.items():
            signal.signal(signal_number, previous)
        self._previous.clear()

    def notice(self) -> PreemptionNotice:
        number = self._signal_number
        try:
            name = None if number is None else signal.Signals(number).name
        except ValueError:
            name = None if number is None else str(number)
        return PreemptionNotice(self._event.is_set(), number, name)

    def synchronized_notice(
        self,
        device: torch.device,
        *,
        group: Any | None = None,
    ) -> PreemptionNotice:
        """Make one common checkpoint/stop decision across distributed ranks."""

        local = self.notice()
        if not (dist.is_available() and dist.is_initialized()):
            return local
        requested = torch.tensor(
            int(local.requested), dtype=torch.int32, device=device
        )
        dist.all_reduce(requested, op=dist.ReduceOp.MAX, group=group)
        if not bool(requested):
            return local
        numbers: list[int | None] = [local.signal_number]
        gathered: list[list[int | None]] = [
            [None] for _ in range(dist.get_world_size(group))
        ]
        dist.all_gather_object(gathered, numbers, group=group)
        first = next(
            (
                item[0]
                for item in gathered
                if item and item[0] is not None
            ),
            None,
        )
        try:
            name = None if first is None else signal.Signals(first).name
        except ValueError:
            name = None if first is None else str(first)
        return PreemptionNotice(True, first, name)


__all__ = ["PreemptionController", "PreemptionNotice"]
