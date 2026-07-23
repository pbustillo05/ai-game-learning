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


BRANCHING_CONTROLLER_VERSION = "adaptive-kkt-v2"
FIXED_BETA_CONTROLLER_VERSION = "fixed-beta-v1"
COST_SAFEGUARD_CONTROLLER_VERSION = "cost-safeguard-v1"


def branched_attempt_address(
    attempt: int,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> int:
    """Return the disjoint rank-local logical tree address for one attempt."""

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("branched rollout attempt must be a nonnegative integer")
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if rank is None:
        rank = torch.distributed.get_rank() if distributed else 0
    if world_size is None:
        world_size = torch.distributed.get_world_size() if distributed else 1
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or not 0 <= rank < world_size
    ):
        raise ValueError("branched rollout rank must lie in [0, world_size)")
    return rank + attempt * world_size


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
    emergency_backoffs: int = 0
    pending_emergency: bool = False

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

    def _damped_size(self, tree_size: float) -> tuple[float, float]:
        size = float(tree_size)
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("counterfactual tree size must be finite and positive")
        previous = size if self.ema_tree_size is None else self.ema_tree_size
        self.ema_tree_size = (
            (1.0 - self.gamma_kappa) * previous + self.gamma_kappa * size
        )
        return size, math.log(self.target_tree_size / self.ema_tree_size)

    def _apply_feedback(self, feedback: float) -> tuple[float, float]:
        before = self.log_kappa
        proposed = before + self.eta_kappa * float(feedback)
        updated = min(
            math.log(self.kappa_max), max(math.log(self.kappa_min), proposed)
        )
        # An OOM attempt is memory-pressure evidence, not permission for the
        # following successful tree to immediately undo its emergency backoff.
        self.log_kappa = min(self.log_kappa, updated) if self.pending_emergency else updated
        self.pending_emergency = False
        self.updates += 1
        return before, self.log_kappa

    def observe(self, counterfactual_tree_size: float) -> Mapping[str, float]:
        size, feedback = self._damped_size(counterfactual_tree_size)
        self._apply_feedback(feedback)
        return {
            "counterfactual_tree_size": size,
            "kkt_ema_tree_size": self.ema_tree_size,
            "log_kappa": self.log_kappa,
            "kappa_saturated_min": float(self.log_kappa == math.log(self.kappa_min)),
            "kappa_saturated_max": float(self.log_kappa == math.log(self.kappa_max)),
        }

    def emergency_backoff(self, factor: float) -> Mapping[str, float]:
        if not math.isfinite(float(factor)) or not 0.0 < factor < 1.0:
            raise ValueError("KKT emergency backoff must lie in (0,1)")
        before = self.log_kappa
        self.log_kappa = max(math.log(self.kappa_min), before + math.log(factor))
        self.emergency_backoffs += 1
        self.pending_emergency = True
        return {
            "kkt_emergency_backoffs": float(self.emergency_backoffs),
            "kkt_emergency_log_kappa_before": before,
            "log_kappa": self.log_kappa,
            "kappa_saturated_min": float(self.log_kappa == math.log(self.kappa_min)),
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
            "emergency_backoffs": self.emergency_backoffs,
            "pending_emergency": self.pending_emergency,
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
            emergency_backoffs=int(state.get("emergency_backoffs", 0)),
            pending_emergency=bool(state.get("pending_emergency", False)),
        )
        if controller.updates < 0 or controller.emergency_backoffs < 0:
            raise RuntimeError("checkpoint adaptive KKT update count is invalid")
        return controller


@dataclass
class FixedBetaController:
    """Fixed-beta facade over the same damped log controller used by KKT."""

    scalar: AdaptiveKKTController
    memory_pressure_target: float
    ema_memory_pressure: float | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.memory_pressure_target)
            or not 0.0 < self.memory_pressure_target <= 1.0
        ):
            raise ValueError("fixed-beta memory pressure target must lie in (0,1]")
        if self.ema_memory_pressure is not None and (
            not math.isfinite(self.ema_memory_pressure)
            or self.ema_memory_pressure <= 0.0
        ):
            raise ValueError("fixed-beta pressure EMA must be finite and positive")

    @classmethod
    def from_config(cls, cfg: Any) -> "FixedBetaController":
        if cfg.branching_ideal_nodes is None:
            raise ValueError("fixed branching requires branching_ideal_nodes")
        return cls(
            scalar=AdaptiveKKTController(
                target_tree_size=float(cfg.branching_ideal_nodes),
                gamma_kappa=float(cfg.fixed_beta_ema),
                eta_kappa=float(cfg.fixed_beta_step),
                kappa_min=float(cfg.fixed_beta_scale_min),
                kappa_max=float(cfg.fixed_beta_scale_max),
                log_kappa=0.0,
            ),
            memory_pressure_target=float(cfg.branching_memory_pressure_target),
        )

    @property
    def scale(self) -> float:
        return math.exp(self.scalar.log_kappa)

    def effective_schedule(
        self, configured: Mapping[str, float]
    ) -> dict[str, float]:
        return {
            str(name): max(1.0, float(value) * self.scale)
            for name, value in configured.items()
        }

    def observe(
        self, tree_size: float, memory_pressure: float
    ) -> Mapping[str, float]:
        pressure = float(memory_pressure)
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ValueError("fixed-beta memory pressure must be finite and positive")
        size, target_error = self.scalar._damped_size(tree_size)
        old_pressure = (
            pressure if self.ema_memory_pressure is None else self.ema_memory_pressure
        )
        self.ema_memory_pressure = (
            (1.0 - self.scalar.gamma_kappa) * old_pressure
            + self.scalar.gamma_kappa * pressure
        )
        pressure_error = min(
            0.0,
            math.log(self.memory_pressure_target / self.ema_memory_pressure),
        )
        # Unsafe memory pressure is a veto: even a small tree may not increase
        # beta until measured headroom is safe again.
        feedback = (
            pressure_error
            if self.ema_memory_pressure > self.memory_pressure_target
            else target_error
        )
        self.scalar._apply_feedback(feedback)
        lower, upper = math.log(self.scalar.kappa_min), math.log(self.scalar.kappa_max)
        return {
            "fixed_beta_tree_size": size,
            "fixed_beta_ema_tree_size": self.scalar.ema_tree_size,
            "fixed_beta_target_error": target_error,
            "fixed_beta_memory_pressure": pressure,
            "fixed_beta_ema_memory_pressure": self.ema_memory_pressure,
            "fixed_beta_scale": self.scale,
            "fixed_beta_scale_saturated_min": float(self.scalar.log_kappa == lower),
            "fixed_beta_scale_saturated_max": float(self.scalar.log_kappa == upper),
        }

    def emergency_backoff(self, factor: float) -> Mapping[str, float]:
        before = self.scale
        self.scalar.emergency_backoff(factor)
        return {
            "fixed_beta_emergency_backoffs": float(self.scalar.emergency_backoffs),
            "fixed_beta_scale_before_backoff": before,
            "fixed_beta_scale": self.scale,
            "fixed_beta_scale_saturated_min": float(
                self.scale == self.scalar.kappa_min
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": FIXED_BETA_CONTROLLER_VERSION,
            "scalar": self.scalar.state_dict(),
            "memory_pressure_target": self.memory_pressure_target,
            "ema_memory_pressure": self.ema_memory_pressure,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FixedBetaController":
        if state.get("version") != FIXED_BETA_CONTROLLER_VERSION:
            raise RuntimeError("checkpoint fixed-beta controller schema is incompatible")
        return cls(
            scalar=AdaptiveKKTController.from_state_dict(state["scalar"]),
            memory_pressure_target=float(state["memory_pressure_target"]),
            ema_memory_pressure=(
                None if state.get("ema_memory_pressure") is None
                else float(state["ema_memory_pressure"])
            ),
        )


@dataclass
class CostSafeguardController:
    """Gate cost-head tree-size safeguards on sustained calibration."""

    gamma_deviance_max: float
    p95_underprediction_max: float
    consecutive_iterations_required: int
    consecutive_successes: int = 0
    active: bool = False
    observations: int = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.gamma_deviance_max))
            or self.gamma_deviance_max < 0.0
        ):
            raise ValueError("cost safeguard Gamma-deviance threshold is invalid")
        if (
            not math.isfinite(float(self.p95_underprediction_max))
            or self.p95_underprediction_max < 1.0
        ):
            raise ValueError("cost safeguard underprediction threshold is invalid")
        if (
            isinstance(self.consecutive_iterations_required, bool)
            or self.consecutive_iterations_required < 1
        ):
            raise ValueError("cost safeguard consecutive-iteration count is invalid")
        if (
            isinstance(self.consecutive_successes, bool)
            or not 0 <= self.consecutive_successes
            <= self.consecutive_iterations_required
        ):
            raise ValueError("cost safeguard success streak is invalid")
        if isinstance(self.observations, bool) or self.observations < 0:
            raise ValueError("cost safeguard observation count is invalid")
        expected_active = (
            self.consecutive_successes >= self.consecutive_iterations_required
        )
        if bool(self.active) != expected_active:
            raise ValueError("cost safeguard active state is inconsistent with its streak")

    @classmethod
    def from_config(cls, cfg: Any) -> "CostSafeguardController":
        return cls(
            gamma_deviance_max=float(
                cfg.safe_tree_activation_gamma_deviance_max
            ),
            p95_underprediction_max=float(
                cfg.safe_tree_activation_p95_underprediction_max
            ),
            consecutive_iterations_required=int(
                cfg.safe_tree_activation_consecutive_iters
            ),
        )

    def observe(
        self,
        gamma_deviance_mean: float,
        p95_underprediction: float,
        calibration_rows: int,
    ) -> Mapping[str, float]:
        deviance = float(gamma_deviance_mean)
        underprediction = float(p95_underprediction)
        rows = int(calibration_rows)
        if not math.isfinite(deviance) or deviance < 0.0:
            raise ValueError("cost calibration Gamma deviance is invalid")
        if not math.isfinite(underprediction) or underprediction < 0.0:
            raise ValueError("cost calibration underprediction factor is invalid")
        if rows < 1:
            raise ValueError("cost calibration requires at least one row")

        passed = (
            deviance < self.gamma_deviance_max
            and underprediction < self.p95_underprediction_max
        )
        if passed:
            self.consecutive_successes = min(
                self.consecutive_iterations_required,
                self.consecutive_successes + 1,
            )
        else:
            self.consecutive_successes = 0
        self.active = (
            self.consecutive_successes >= self.consecutive_iterations_required
        )
        self.observations += 1
        return {
            "cost_calibration_gamma_deviance_mean": deviance,
            "cost_calibration_underprediction_p95": underprediction,
            "cost_calibration_rows": float(rows),
            "safe_tree_calibration_passed": float(passed),
            "safe_tree_calibration_streak": float(self.consecutive_successes),
            "safe_tree_calibration_required": float(
                self.consecutive_iterations_required
            ),
            "safe_tree_active_next_iteration": float(self.active),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": COST_SAFEGUARD_CONTROLLER_VERSION,
            "gamma_deviance_max": self.gamma_deviance_max,
            "p95_underprediction_max": self.p95_underprediction_max,
            "consecutive_iterations_required": self.consecutive_iterations_required,
            "consecutive_successes": self.consecutive_successes,
            "active": self.active,
            "observations": self.observations,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CostSafeguardController":
        if state.get("version") != COST_SAFEGUARD_CONTROLLER_VERSION:
            raise RuntimeError("checkpoint cost-safeguard controller schema is incompatible")
        return cls(
            gamma_deviance_max=float(state["gamma_deviance_max"]),
            p95_underprediction_max=float(state["p95_underprediction_max"]),
            consecutive_iterations_required=int(
                state["consecutive_iterations_required"]
            ),
            consecutive_successes=int(state.get("consecutive_successes", 0)),
            active=bool(state.get("active", False)),
            observations=int(state.get("observations", 0)),
        )
