"""Loss configuration shared by validation and the streamed learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _cfg(cfg: Any, name: str, default: Any = None) -> Any:
    source = getattr(cfg, "loss", cfg)
    return getattr(source, name, default)


def _entropy_coefficient(cfg: Any, iteration: int) -> float:
    """Return the entropy coefficient for a zero-based training iteration."""

    value = float(_cfg(cfg, "c_h"))
    if _cfg(cfg, "entropy_schedule", "const") == "linear_decay":
        schedule = getattr(cfg, "training", cfg)
        iters = _cfg(cfg, "entropy_decay_iters", None)
        if iters is None:
            iters = getattr(schedule, "iters", None)
        if isinstance(iters, bool) or not isinstance(iters, int) or iters <= 0:
            raise ValueError(
                "linear entropy decay requires a positive training.iters horizon"
            )
        value *= max(0.0, 1.0 - float(iteration) / float(iters))
    return value


@dataclass(frozen=True)
class LossSpec:
    """Configuration-time description of the modes a loss name selects."""

    modes: frozenset[str]


COMPOSITES: dict[str, LossSpec] = {
    "ppo": LossSpec(frozenset({"normal"})),
    "rb_ppo": LossSpec(
        frozenset({"rao_blackwell", "branched_rao_blackwell"})
    ),
}
