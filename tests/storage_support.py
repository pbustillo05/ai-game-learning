from __future__ import annotations

from torch import Tensor

from pbg.core.storage import MonotoneSlab


def published_rows(slab: MonotoneSlab, lane: int, field: str) -> Tensor:
    """Return published rows without adding a test-only production method."""

    return slab.columns[field][lane, : slab.tail(lane)]
