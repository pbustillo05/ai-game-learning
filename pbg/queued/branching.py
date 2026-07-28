"""Branching controls and whole-attempt recovery for the queued engine.

This module is game-opaque.  It owns the one fixed/KKT controller contract,
ordinary randomized rounding, queue-pressure correction, lower-size root
refill metadata, and the only recoverable transition for a branched attempt.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar

import torch
from torch import Tensor

from pbg.queued.contracts import ModeSchema, RolloutMode
from pbg.queued.persistence import RunState
from pbg.queued.rollout import DrawPurpose, logical_rng_keys, logical_uniform
from pbg.queued.training import AttemptAddress, RecoverableOverflow, TrainingError


class BranchingError(RuntimeError):
    """A branching control or recovery invariant was violated."""


BRANCH_CONTROLLER_VERSION = "queued-branch-controller-v1"
COST_SAFEGUARD_VERSION = "queued-cost-safeguard-v1"


def randomized_round(beta: Tensor, uniform: Tensor) -> Tensor:
    """Ordinarily randomized-round finite intensities in ``[1, beta_max]``."""

    if (
        beta.shape != uniform.shape
        or beta.device != uniform.device
        or not beta.is_floating_point()
        or not uniform.is_floating_point()
    ):
        raise ValueError("beta and uniform must be aligned floating tensors")
    torch._assert_async(
        (torch.isfinite(beta) & beta.ge(1.0)).all(),
        "branching intensities must be finite and at least one",
    )
    base = torch.floor(beta)
    return (base + uniform.lt(beta - base)).to(torch.int32)


def log_kkt_intensity(
    trace: Tensor,
    cost: Tensor,
    *,
    log_kappa: float,
    log_player_ancestor_sum: Tensor,
    log_ancestor_product: Tensor,
    beta_max: float,
) -> Tensor:
    """Evaluate the authoritative continuous KKT update in FP32 log space."""

    if not (
        trace.shape
        == cost.shape
        == log_player_ancestor_sum.shape
        == log_ancestor_product.shape
    ):
        raise ValueError("KKT head and path tensors must be aligned")
    if not math.isfinite(beta_max) or beta_max < 1.0:
        raise ValueError("beta_max must be finite and at least one")
    tiny = torch.finfo(torch.float32).tiny
    log_beta = 0.5 * (
        float(log_kappa)
        + trace.detach().float().clamp_min(tiny).log()
        - cost.detach().float().clamp_min(tiny).log()
        + log_player_ancestor_sum.detach().float()
        - log_ancestor_product.detach().float()
    )
    return log_beta.clamp_(0.0, math.log(beta_max)).exp_()


def child_path_factors(
    log_player_ancestor_sum: Tensor,
    log_ancestor_product: Tensor,
    parent_beta: Tensor,
    *,
    child_is_player: Tensor,
) -> tuple[Tensor, Tensor]:
    """Advance the two KKT ancestor factors from one parent to its children."""

    if not (
        log_player_ancestor_sum.shape
        == log_ancestor_product.shape
        == parent_beta.shape
        == child_is_player.shape
    ):
        raise ValueError("KKT child path factors must be aligned")
    log_beta = parent_beta.detach().float().log()
    inherited = log_player_ancestor_sum.detach().float() - log_beta
    local = torch.where(
        child_is_player,
        torch.zeros_like(inherited),
        torch.full_like(inherited, -torch.inf),
    )
    return (
        torch.logaddexp(inherited, local),
        log_ancestor_product.detach().float() + log_beta,
    )


def decision_path_mass(parent: Tensor, executed_degree: Tensor) -> Tensor:
    """Recompute final decision masses after protected root augmentation."""

    if (
        parent.device.type != "cpu"
        or executed_degree.device.type != "cpu"
        or parent.dtype not in (torch.int32, torch.int64)
        or executed_degree.dtype not in (torch.int32, torch.int64)
        or parent.ndim != 1
        or parent.shape != executed_degree.shape
    ):
        raise ValueError("topology mass inputs must be aligned CPU integers")
    mass = torch.ones(parent.numel(), dtype=torch.float64)
    for node in range(parent.numel()):
        ancestor = int(parent[node])
        if ancestor >= node:
            raise BranchingError("topology mass order must place every parent first")
        if ancestor >= 0:
            degree = int(executed_degree[ancestor])
            if degree <= 0:
                raise BranchingError("topology mass encountered zero degree")
            mass[node] = mass[ancestor] / degree
    return mass


@dataclass(frozen=True)
class BranchDecision:
    """Controller result for one device-local node pack."""

    beta: Tensor
    nominal_degree: Tensor
    executed_degree: Tensor
    target_degree: Tensor

    def __post_init__(self) -> None:
        shape = self.beta.shape
        if self.beta.dtype != torch.float32 or self.beta.ndim != 1:
            raise ValueError("branch beta must be FP32 [nodes]")
        for name in ("nominal_degree", "executed_degree", "target_degree"):
            value = getattr(self, name)
            if (
                value.device != self.beta.device
                or value.dtype != torch.int32
                or value.shape != shape
            ):
                raise ValueError(f"{name} must be aligned int32 [nodes]")
        torch._assert_async(
            (
                self.nominal_degree.ge(1)
                & self.executed_degree.ge(1)
                & self.target_degree.ge(1)
                & self.target_degree.le(self.executed_degree)
            ).all(),
            "branch degrees violate 1 <= N_tgt <= N",
        )


class BranchController(Protocol):
    """The only rollout-facing fixed/KKT controller interface."""

    mode: RolloutMode
    beta_max: float

    def decide(
        self,
        *,
        schema: ModeSchema,
        root: Tensor,
        path: Tensor,
        owner_is_player: Tensor,
        iteration: int,
        seed: int,
        retry_epoch: Tensor | int,
        trace: Tensor | None,
        cost: Tensor | None,
        log_player_ancestor_sum: Tensor,
        log_ancestor_product: Tensor,
        frontier_cost: Tensor | None = None,
    ) -> BranchDecision: ...

    def backoff(self) -> Mapping[str, float]: ...

    @property
    def at_minimum(self) -> bool: ...

    def state_dict(self) -> dict[str, object]: ...


@dataclass
class FixedBranchController:
    """Stage-fixed intensities using the shared controller operation."""

    fixed_beta: Mapping[str, float]
    beta_max: float
    scale: float = 1.0
    minimum_scale: float = 0.25
    backoff_factor: float = 0.5
    safe_tree_size: float | None = None
    mode: RolloutMode = RolloutMode.BRANCHED_FIXED
    emergency_backoffs: int = 0

    def __post_init__(self) -> None:
        values = dict(self.fixed_beta)
        if not values or any(
            not isinstance(name, str)
            or not name
            or not math.isfinite(float(value))
            or not 1.0 <= float(value) <= self.beta_max
            for name, value in values.items()
        ):
            raise ValueError("fixed_beta must contain declared intensities")
        self.fixed_beta = MappingProxyType(
            {name: float(value) for name, value in values.items()}
        )
        if (
            not math.isfinite(self.beta_max)
            or self.beta_max < 1.0
            or not math.isfinite(self.scale)
            or not math.isfinite(self.minimum_scale)
            or not 0 < self.minimum_scale <= self.scale
            or not 0 < self.backoff_factor < 1
        ):
            raise ValueError("fixed branching scale bounds are invalid")
        if self.safe_tree_size is not None and (
            not math.isfinite(self.safe_tree_size) or self.safe_tree_size < 1
        ):
            raise ValueError("safe_tree_size must be finite and at least one")

    def decide(
        self,
        *,
        schema: ModeSchema,
        root: Tensor,
        path: Tensor,
        owner_is_player: Tensor,
        iteration: int,
        seed: int,
        retry_epoch: Tensor | int,
        trace: Tensor | None,
        cost: Tensor | None,
        log_player_ancestor_sum: Tensor,
        log_ancestor_product: Tensor,
        frontier_cost: Tensor | None = None,
    ) -> BranchDecision:
        del trace, cost, log_player_ancestor_sum, log_ancestor_product
        if schema.rollout_mode is not self.mode:
            raise BranchingError("fixed controller received a different mode")
        try:
            player_beta = self.fixed_beta["player"]
            chance_beta = self.fixed_beta["chance"]
        except KeyError as exc:
            raise BranchingError("fixed beta omits the requested node family") from exc
        beta = torch.where(
            owner_is_player,
            torch.full_like(
                root,
                min(self.beta_max, max(1.0, player_beta * self.scale)),
                dtype=torch.float32,
            ),
            torch.full_like(
                root,
                min(self.beta_max, max(1.0, chance_beta * self.scale)),
                dtype=torch.float32,
            ),
        )
        uniform = logical_uniform(
            logical_rng_keys(
                seed=seed,
                iteration=iteration,
                root=root,
                path=path,
                draw=0,
                purpose=DrawPurpose.ROUND,
                retry_epoch=retry_epoch,
            )
        )
        nominal = randomized_round(beta, uniform)
        executed = nominal.clone()
        if self.safe_tree_size is not None and frontier_cost is not None:
            torch._assert_async(
                (torch.isfinite(frontier_cost) & frontier_cost.ge(0)).all(),
                "frontier cost must be finite and nonnegative",
            )
            suppressed = owner_is_player & frontier_cost.gt(self.safe_tree_size)
            executed = torch.where(suppressed, torch.ones_like(executed), executed)
        return BranchDecision(beta, nominal, executed, executed.clone())

    @property
    def at_minimum(self) -> bool:
        return self.scale <= self.minimum_scale

    def backoff(self) -> Mapping[str, float]:
        before = self.scale
        self.scale = max(self.minimum_scale, self.scale * self.backoff_factor)
        self.emergency_backoffs += 1
        return MappingProxyType(
            {
                "fixed_scale_before": before,
                "fixed_scale_after": self.scale,
                "emergency_backoffs": float(self.emergency_backoffs),
            }
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": BRANCH_CONTROLLER_VERSION,
            "mode": self.mode.value,
            "fixed_beta": dict(self.fixed_beta),
            "beta_max": self.beta_max,
            "scale": self.scale,
            "minimum_scale": self.minimum_scale,
            "backoff_factor": self.backoff_factor,
            "safe_tree_size": self.safe_tree_size,
            "emergency_backoffs": self.emergency_backoffs,
        }


@dataclass
class KKTBranchController:
    """Warmup-fixed then trace/cost KKT branching through the same interface."""

    fixed_beta: Mapping[str, float]
    beta_max: float
    warmup_iterations: int
    target_tree_size: float
    gamma_kappa: float
    eta_kappa: float
    kappa_min: float
    kappa_max: float
    log_kappa: float
    backoff_factor: float = 0.5
    safe_tree_size: float | None = None
    ema_tree_size: float | None = None
    updates: int = 0
    emergency_backoffs: int = 0
    mode: RolloutMode = RolloutMode.BRANCHED_KKT

    def __post_init__(self) -> None:
        self.fixed_beta = MappingProxyType(
            {name: float(value) for name, value in dict(self.fixed_beta).items()}
        )
        if set(self.fixed_beta) != {"player", "chance"} or any(
            not math.isfinite(value) or not 1 <= value <= self.beta_max
            for value in self.fixed_beta.values()
        ):
            raise ValueError("KKT warmup beta must cover player and chance")
        positive = (
            self.beta_max,
            self.target_tree_size,
            self.gamma_kappa,
            self.eta_kappa,
            self.kappa_min,
            self.kappa_max,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("KKT controller values must be finite and positive")
        if (
            self.beta_max < 1
            or self.gamma_kappa > 1
            or self.eta_kappa > 1
            or self.kappa_min > self.kappa_max
            or not math.log(self.kappa_min)
            <= self.log_kappa
            <= math.log(self.kappa_max)
            or not 0 < self.backoff_factor < 1
            or self.warmup_iterations < 0
        ):
            raise ValueError("KKT controller bounds are invalid")

    def decide(
        self,
        *,
        schema: ModeSchema,
        root: Tensor,
        path: Tensor,
        owner_is_player: Tensor,
        iteration: int,
        seed: int,
        retry_epoch: Tensor | int,
        trace: Tensor | None,
        cost: Tensor | None,
        log_player_ancestor_sum: Tensor,
        log_ancestor_product: Tensor,
        frontier_cost: Tensor | None = None,
    ) -> BranchDecision:
        if schema.rollout_mode is not self.mode:
            raise BranchingError("KKT controller received a different mode")
        if iteration < self.warmup_iterations:
            player_beta = torch.full_like(
                root, self.fixed_beta["player"], dtype=torch.float32
            )
        else:
            has_player = bool(owner_is_player.any())
            if has_player and (trace is None or cost is None):
                raise BranchingError("active KKT branching requires T/C heads")
            player_beta = (
                log_kkt_intensity(
                    trace,
                    cost,
                    log_kappa=self.log_kappa,
                    log_player_ancestor_sum=log_player_ancestor_sum,
                    log_ancestor_product=log_ancestor_product,
                    beta_max=self.beta_max,
                )
                if has_player
                else torch.ones_like(root, dtype=torch.float32)
            )
        chance = torch.full_like(
            player_beta,
            min(self.beta_max, self.fixed_beta["chance"]),
        )
        beta = torch.where(owner_is_player, player_beta, chance)
        uniform = logical_uniform(
            logical_rng_keys(
                seed=seed,
                iteration=iteration,
                root=root,
                path=path,
                draw=0,
                purpose=DrawPurpose.ROUND,
                retry_epoch=retry_epoch,
            )
        )
        nominal = randomized_round(beta, uniform)
        executed = nominal.clone()
        if self.safe_tree_size is not None and frontier_cost is not None:
            suppressed = owner_is_player & frontier_cost.gt(self.safe_tree_size)
            executed = torch.where(suppressed, torch.ones_like(executed), executed)
        return BranchDecision(beta, nominal, executed, executed.clone())

    def observe(self, counterfactual_tree_size: float) -> Mapping[str, float]:
        size = float(counterfactual_tree_size)
        if not math.isfinite(size) or size <= 0:
            raise ValueError("counterfactual tree size must be positive")
        previous = size if self.ema_tree_size is None else self.ema_tree_size
        self.ema_tree_size = (1 - self.gamma_kappa) * previous + self.gamma_kappa * size
        self.log_kappa = min(
            math.log(self.kappa_max),
            max(
                math.log(self.kappa_min),
                self.log_kappa
                + self.eta_kappa * math.log(self.target_tree_size / self.ema_tree_size),
            ),
        )
        self.updates += 1
        return MappingProxyType(
            {
                "counterfactual_tree_size": size,
                "ema_tree_size": self.ema_tree_size,
                "log_kappa": self.log_kappa,
            }
        )

    @property
    def at_minimum(self) -> bool:
        return self.log_kappa <= math.log(self.kappa_min)

    def backoff(self) -> Mapping[str, float]:
        before = self.log_kappa
        self.log_kappa = max(
            math.log(self.kappa_min),
            self.log_kappa + math.log(self.backoff_factor),
        )
        self.emergency_backoffs += 1
        return MappingProxyType(
            {
                "log_kappa_before": before,
                "log_kappa_after": self.log_kappa,
                "emergency_backoffs": float(self.emergency_backoffs),
            }
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": BRANCH_CONTROLLER_VERSION,
            "mode": self.mode.value,
            "fixed_beta": dict(self.fixed_beta),
            "beta_max": self.beta_max,
            "warmup_iterations": self.warmup_iterations,
            "target_tree_size": self.target_tree_size,
            "gamma_kappa": self.gamma_kappa,
            "eta_kappa": self.eta_kappa,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "log_kappa": self.log_kappa,
            "backoff_factor": self.backoff_factor,
            "safe_tree_size": self.safe_tree_size,
            "ema_tree_size": self.ema_tree_size,
            "updates": self.updates,
            "emergency_backoffs": self.emergency_backoffs,
        }


def controller_from_state_dict(
    state: Mapping[str, object],
) -> FixedBranchController | KKTBranchController:
    if state.get("version") != BRANCH_CONTROLLER_VERSION:
        raise BranchingError("branch controller checkpoint version is incompatible")
    mode = state.get("mode")
    values = dict(state)
    values.pop("version")
    values.pop("mode")
    if mode == RolloutMode.BRANCHED_FIXED.value:
        return FixedBranchController(**values)
    if mode == RolloutMode.BRANCHED_KKT.value:
        return KKTBranchController(**values)
    raise BranchingError("branch controller checkpoint has an invalid mode")


@dataclass
class CostSafeguardController:
    """Enable queue pressure only after sustained strict calibration."""

    gamma_deviance_max: float
    p95_underprediction_max: float
    consecutive_iterations: int
    consecutive_successes: int = 0
    active: bool = False
    observations: int = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.gamma_deviance_max)
            or self.gamma_deviance_max < 0
            or not math.isfinite(self.p95_underprediction_max)
            or self.p95_underprediction_max < 1
            or isinstance(self.consecutive_iterations, bool)
            or self.consecutive_iterations <= 0
            or not 0 <= self.consecutive_successes <= self.consecutive_iterations
            or self.active
            != (self.consecutive_successes == self.consecutive_iterations)
            or self.observations < 0
        ):
            raise ValueError("cost safeguard state is invalid")

    def observe(
        self,
        gamma_deviance_mean: float,
        p95_underprediction: float,
        rows: int,
    ) -> Mapping[str, float]:
        if (
            not math.isfinite(gamma_deviance_mean)
            or gamma_deviance_mean < 0
            or not math.isfinite(p95_underprediction)
            or p95_underprediction < 0
            or rows <= 0
        ):
            raise ValueError("cost safeguard observation is invalid")
        passed = (
            gamma_deviance_mean < self.gamma_deviance_max
            and p95_underprediction < self.p95_underprediction_max
        )
        self.consecutive_successes = (
            min(self.consecutive_iterations, self.consecutive_successes + 1)
            if passed
            else 0
        )
        self.active = self.consecutive_successes == self.consecutive_iterations
        self.observations += 1
        return MappingProxyType(
            {
                "gamma_deviance_mean": float(gamma_deviance_mean),
                "p95_underprediction": float(p95_underprediction),
                "calibration_rows": float(rows),
                "passed": float(passed),
                "active": float(self.active),
            }
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": COST_SAFEGUARD_VERSION,
            "gamma_deviance_max": self.gamma_deviance_max,
            "p95_underprediction_max": self.p95_underprediction_max,
            "consecutive_iterations": self.consecutive_iterations,
            "consecutive_successes": self.consecutive_successes,
            "active": self.active,
            "observations": self.observations,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> CostSafeguardController:
        if state.get("version") != COST_SAFEGUARD_VERSION:
            raise BranchingError("cost safeguard checkpoint is incompatible")
        values = dict(state)
        values.pop("version")
        return cls(**values)


AttemptResult = TypeVar("AttemptResult")


@dataclass(frozen=True)
class BranchedRecoveryRecord:
    severity: str
    iteration: int
    replica: int
    failed_phase: str
    failed_address: str
    replacement_address: str
    old_control: Mapping[str, object]
    new_control: Mapping[str, object]
    capacity: int
    observed_demand: int

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "iteration": self.iteration,
            "replica": self.replica,
            "failed_phase": self.failed_phase,
            "failed_address": self.failed_address,
            "replacement_address": self.replacement_address,
            "old_control": dict(self.old_control),
            "new_control": dict(self.new_control),
            "capacity": self.capacity,
            "observed_demand": self.observed_demand,
        }


class BranchedOverflowRecovery(Generic[AttemptResult]):
    """Discard a failed tree and retry at a disjoint address with less work."""

    def __init__(
        self,
        *,
        iteration: int,
        replica: int,
        controller: BranchController,
        run_state: RunState,
        logger: logging.Logger | None = None,
        rank_common: Callable[[RecoverableOverflow], RecoverableOverflow] | None = None,
    ) -> None:
        if iteration < 0 or replica < 0:
            raise ValueError("iteration and replica must be nonnegative")
        self.iteration = iteration
        self.replica = replica
        self.controller = controller
        self.run_state = run_state
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.rank_common = rank_common

    def run(
        self,
        attempt: Callable[[AttemptAddress], AttemptResult],
        discard: Callable[[RecoverableOverflow], None],
    ) -> AttemptResult:
        epoch = self.run_state.retry_epoch
        logical_base = self.run_state.attempt_logical_base
        while True:
            address = AttemptAddress(epoch, logical_base)
            try:
                return attempt(address)
            except RecoverableOverflow as local:
                failure = local if self.rank_common is None else self.rank_common(local)
                discard(failure)
                if self.controller.at_minimum:
                    raise TrainingError(
                        "minimum branched control overflowed fixed capacity"
                    ) from failure
                old = self.controller.state_dict()
                self.controller.backoff()
                new = self.controller.state_dict()
                replacement = AttemptAddress(
                    epoch + 1,
                    logical_base + self.run_state.root_count,
                )
                record = BranchedRecoveryRecord(
                    severity="critical",
                    iteration=self.iteration,
                    replica=self.replica,
                    failed_phase=failure.failure.phase,
                    failed_address=AttemptAddress(
                        epoch, failure.failure.logical_address
                    ).as_text(),
                    replacement_address=replacement.as_text(),
                    old_control=old,
                    new_control=new,
                    capacity=failure.capacity,
                    observed_demand=failure.observed_demand,
                )
                self.run_state.recovered_overflow = True
                self.run_state.retry_epoch = replacement.retry_epoch
                self.run_state.attempt_logical_base = replacement.logical_base
                self.run_state.branch_controller_state = new
                self.run_state.overflow_recoveries.append(record.as_dict())
                payload = json.dumps(record.as_dict(), sort_keys=True)
                self.logger.critical("RECOVERED_BRANCHED_OVERFLOW %s", payload)
                warnings.warn(
                    f"RECOVERED_BRANCHED_OVERFLOW {payload}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                epoch = replacement.retry_epoch
                logical_base = replacement.logical_base


__all__ = [
    "BRANCH_CONTROLLER_VERSION",
    "COST_SAFEGUARD_VERSION",
    "BranchController",
    "BranchDecision",
    "BranchedOverflowRecovery",
    "BranchedRecoveryRecord",
    "BranchingError",
    "CostSafeguardController",
    "FixedBranchController",
    "KKTBranchController",
    "child_path_factors",
    "controller_from_state_dict",
    "decision_path_mass",
    "log_kkt_intensity",
    "randomized_round",
]
