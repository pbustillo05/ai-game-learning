"""Explicit training-utility units shared by target builders and migration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def utility_scale(cfg: Any) -> float:
    """Return the multiplier from raw game payoff to configured training units."""

    loss = _field(cfg, "loss", cfg)
    convention = _field(loss, "utility_normalization", "none")
    if convention == "none":
        return 1.0
    if convention != "stack":
        raise ValueError(f"unknown utility normalization {convention!r}")
    game = _field(cfg, "game", None)
    stack = None if game is None else _field(game, "stack_bb", None)
    if isinstance(stack, bool) or not isinstance(stack, (int, float)) or stack <= 0:
        raise ValueError("stack utility normalization requires a positive game.stack_bb")
    return 1.0 / float(stack)


def utility_conversion(source_cfg: Any, target_cfg: Any) -> float:
    """Return the value-output multiplier between two utility conventions."""

    return utility_scale(target_cfg) / utility_scale(source_cfg)
