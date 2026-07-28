"""Process-lifetime ownership of contract-v2 poker evaluator bindings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=10)
def bound_evaluator(rank_count: int) -> Any:
    """Bind one evaluator per supported deck geometry and reuse its workspace."""

    if (
        isinstance(rank_count, bool)
        or not isinstance(rank_count, int)
        or not 4 <= rank_count <= 13
    ):
        raise ValueError("rank_count must be an integer in [4,13]")
    try:
        import poker_gpu_evaluator as poker
    except ImportError as exc:
        raise RuntimeError("poker-gpu-evaluator is required") from exc
    if str(getattr(poker, "CONTRACT_VERSION", "")) != "2":
        raise RuntimeError("poker evaluator contract v2 is required")
    evaluator = poker.bind(rank_count=rank_count)
    info = evaluator.info
    if (
        str(info.contract_version) != "2"
        or int(info.rank_count) != rank_count
    ):
        raise RuntimeError("bound evaluator identity differs from deck geometry")
    return evaluator


__all__ = ["bound_evaluator"]
