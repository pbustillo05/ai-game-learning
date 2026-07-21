"""Adaptive KKT public branching primitives.

All controller inputs are rollout-time tensors or detached scalars.  This
module deliberately has no game/private-state dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor


BRANCHING_CONTROLLER_VERSION = "adaptive-kkt-v1"


def gamma_deviance(
    prediction: Tensor,
    target: Tensor,
    *,
    shift: float = 0.0,
) -> Tensor:
    """Return elementwise Gamma deviance with an optional common shift.

    ``target/prediction + log(prediction/target) - 1`` is evaluated as
    ``ratio - log(ratio) - 1``.  The latter avoids a second division and is
    accurate near the optimum through ``log1p``.
    """

    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Gamma deviance inputs must be floating point")
    if not math.isfinite(float(shift)) or shift < 0.0:
        raise ValueError("Gamma deviance shift must be finite and nonnegative")
    work = torch.promote_types(prediction.dtype, target.dtype)
    if work in (torch.float16, torch.bfloat16):
        work = torch.float32
    pred = prediction.to(work) + float(shift)
    wanted = target.detach().to(work) + float(shift)
    tiny = torch.finfo(work).tiny
    if bool((wanted <= 0).any()):
        raise ValueError("Gamma deviance target plus shift must be positive")
    pred = pred.clamp_min(tiny)
    ratio = wanted / pred
    # ratio - 1 - log(ratio), stable when ratio is close to one.
    delta = ratio - 1.0
    return delta - torch.log1p(delta)


def randomized_round(beta: Tensor, uniform: Tensor) -> Tensor:
    """Ordinarily randomized-round positive continuous intensities."""

    if beta.shape != uniform.shape or beta.device != uniform.device:
        raise ValueError("beta and uniform must have identical device-local shapes")
    if not beta.is_floating_point() or not uniform.is_floating_point():
        raise TypeError("randomized rounding inputs must be floating point")
    if bool((~torch.isfinite(beta)).any()) or bool((beta < 1.0).any()):
        raise ValueError("beta must be finite and at least one")
    base = torch.floor(beta)
    return (base + (uniform < beta - base).to(beta.dtype)).to(torch.int64)


def log_kkt_intensity(
    t: Tensor,
    cost: Tensor,
    log_kappa: float | Tensor,
    log_player_ancestor_sum: Tensor,
    log_ancestor_product: Tensor,
    beta_max: float,
) -> Tensor:
    """Evaluate the clipped KKT fixed-point update in the log domain."""

    if t.shape != cost.shape or t.shape != log_player_ancestor_sum.shape:
        raise ValueError("KKT head/path tensors must have identical shapes")
    if log_ancestor_product.shape != t.shape:
        raise ValueError("KKT ancestor-product tensor has the wrong shape")
    if not math.isfinite(float(beta_max)) or beta_max < 1.0:
        raise ValueError("beta_max must be finite and at least one")
    work_t = t.detach().float().clamp_min(torch.finfo(torch.float32).tiny)
    work_c = cost.detach().float().clamp_min(torch.finfo(torch.float32).tiny)
    log_value = 0.5 * (
        torch.as_tensor(log_kappa, device=t.device, dtype=torch.float32)
        + torch.log(work_t)
        - torch.log(work_c)
        + log_player_ancestor_sum.detach().float()
        - log_ancestor_product.detach().float()
    )
    return torch.exp(log_value.clamp(min=0.0, max=math.log(float(beta_max))))


def child_path_factors(
    log_player_ancestor_sum: Tensor,
    log_ancestor_product: Tensor,
    parent_beta: Tensor,
    child_is_player: Tensor,
) -> tuple[Tensor, Tensor]:
    """Advance the two KKT ancestor caches from parent to child."""

    log_beta = torch.log(parent_beta.detach().float())
    inherited = log_player_ancestor_sum.detach().float() - log_beta
    local = torch.where(
        child_is_player,
        torch.zeros_like(inherited),
        torch.full_like(inherited, -torch.inf),
    )
    return torch.logaddexp(inherited, local), log_ancestor_product.detach().float() + log_beta


@dataclass
class AdaptiveKKTController:
    """Damped log-``kappa`` controller driven by counterfactual tree size."""

    target_tree_size: float
    gamma_kappa: float
    eta_kappa: float
    kappa_min: float
    kappa_max: float
    log_kappa: float
    ema_tree_size: float | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        values = (
            self.target_tree_size, self.gamma_kappa, self.eta_kappa,
            self.kappa_min, self.kappa_max,
        )
        if any(not math.isfinite(float(value)) or value <= 0.0 for value in values):
            raise ValueError("adaptive KKT controller values must be finite and positive")
        if self.gamma_kappa > 1.0 or self.eta_kappa > 1.0:
            raise ValueError("gamma_kappa and eta_kappa must be at most one")
        if self.kappa_min > self.kappa_max:
            raise ValueError("kappa_min cannot exceed kappa_max")
        lower, upper = math.log(self.kappa_min), math.log(self.kappa_max)
        if not lower <= self.log_kappa <= upper:
            raise ValueError("log_kappa is outside configured bounds")
        if self.ema_tree_size is not None and (
            not math.isfinite(self.ema_tree_size) or self.ema_tree_size <= 0.0
        ):
            raise ValueError("ema_tree_size must be finite and positive")

    @classmethod
    def from_config(cls, cfg: Any) -> "AdaptiveKKTController":
        return cls(
            target_tree_size=float(cfg.target_tree_size),
            gamma_kappa=float(cfg.gamma_kappa),
            eta_kappa=float(cfg.eta_kappa),
            kappa_min=float(cfg.kappa_min),
            kappa_max=float(cfg.kappa_max),
            log_kappa=math.log(float(cfg.kappa_initial)),
        )

    def observe(self, counterfactual_tree_size: float) -> Mapping[str, float]:
        size = float(counterfactual_tree_size)
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("counterfactual tree size must be finite and positive")
        previous = size if self.ema_tree_size is None else self.ema_tree_size
        self.ema_tree_size = (
            (1.0 - self.gamma_kappa) * previous + self.gamma_kappa * size
        )
        proposed = self.log_kappa + self.eta_kappa * math.log(
            self.target_tree_size / self.ema_tree_size
        )
        self.log_kappa = min(
            math.log(self.kappa_max), max(math.log(self.kappa_min), proposed)
        )
        self.updates += 1
        return {
            "counterfactual_tree_size": size,
            "kkt_ema_tree_size": self.ema_tree_size,
            "log_kappa": self.log_kappa,
            "kappa_saturated_min": float(self.log_kappa == math.log(self.kappa_min)),
            "kappa_saturated_max": float(self.log_kappa == math.log(self.kappa_max)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": BRANCHING_CONTROLLER_VERSION,
            "target_tree_size": self.target_tree_size,
            "gamma_kappa": self.gamma_kappa,
            "eta_kappa": self.eta_kappa,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "log_kappa": self.log_kappa,
            "ema_tree_size": self.ema_tree_size,
            "updates": self.updates,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AdaptiveKKTController":
        if state.get("version") != BRANCHING_CONTROLLER_VERSION:
            raise RuntimeError("checkpoint adaptive KKT controller schema is incompatible")
        controller = cls(
            target_tree_size=float(state["target_tree_size"]),
            gamma_kappa=float(state["gamma_kappa"]),
            eta_kappa=float(state["eta_kappa"]),
            kappa_min=float(state["kappa_min"]),
            kappa_max=float(state["kappa_max"]),
            log_kappa=float(state["log_kappa"]),
            ema_tree_size=(
                None if state.get("ema_tree_size") is None
                else float(state["ema_tree_size"])
            ),
            updates=int(state.get("updates", 0)),
        )
        if controller.updates < 0:
            raise RuntimeError("checkpoint adaptive KKT update count is invalid")
        return controller
