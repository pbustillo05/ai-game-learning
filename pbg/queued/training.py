"""Phase 5 iteration composition and whole-attempt overflow recovery."""

from __future__ import annotations

import json
import logging
import math
import signal
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

import torch
from torch import Tensor, nn

from pbg.queued.learner import (
    BranchedBatchSampler,
    LearnerOperation,
    LossResult,
    ReferenceOutputCache,
    SequentialBatchSampler,
    SequentialLearnerBatch,
)
from pbg.queued.optimization import DenseAdam, OptimizerStep
from pbg.queued.persistence import RunState
from pbg.queued.storage import DeviceFailure


class TrainingError(RuntimeError):
    """Iteration execution cannot continue safely."""


class RecoverableOverflow(RuntimeError):
    """A complete unpublished rollout attempt exceeded a fixed capacity."""

    def __init__(
        self,
        failure: DeviceFailure,
        *,
        capacity: int,
        observed_demand: int,
    ) -> None:
        if failure.failure_code != "capacity_overflow":
            raise ValueError("only capacity overflow is recoverable")
        self.failure = failure
        self.capacity = int(capacity)
        self.observed_demand = int(observed_demand)
        super().__init__(
            f"{failure.phase} overflow at logical address {failure.logical_address}"
        )


@dataclass(frozen=True)
class AttemptAddress:
    retry_epoch: int
    logical_base: int

    def __post_init__(self) -> None:
        if self.retry_epoch < 0 or self.logical_base < 0:
            raise ValueError("attempt addresses must be nonnegative")

    def as_text(self) -> str:
        return f"{self.retry_epoch}:{self.logical_base}"


@dataclass(frozen=True)
class OverflowRecoveryRecord:
    severity: str
    iteration: int
    replica: int
    failed_phase: str
    failed_address: str
    replacement_address: str
    old_root_count: int
    new_root_count: int
    capacity: int
    observed_demand: int

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


AttemptResult = TypeVar("AttemptResult")


class SequentialOverflowRecovery(Generic[AttemptResult]):
    """Retry complete attempts at disjoint addresses and lower root counts."""

    def __init__(
        self,
        *,
        iteration: int,
        replica: int,
        root_count: int,
        minimum_root_count: int,
        run_state: RunState,
        logger: logging.Logger | None = None,
        rank_common: Callable[[RecoverableOverflow], RecoverableOverflow] | None = None,
    ) -> None:
        if iteration < 0 or replica < 0:
            raise ValueError("iteration and replica must be nonnegative")
        if not 1 <= minimum_root_count <= root_count:
            raise ValueError("root_count must cover its positive minimum")
        self.iteration = iteration
        self.replica = replica
        self.root_count = root_count
        self.minimum_root_count = minimum_root_count
        self.run_state = run_state
        self.run_state.root_count = root_count
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.rank_common = rank_common

    def _lower(self, value: int) -> int:
        replacement = max(self.minimum_root_count, value // 2)
        if replacement == value and value > self.minimum_root_count:
            replacement = value - 1
        return replacement

    def run(
        self,
        attempt: Callable[[int, AttemptAddress], AttemptResult],
        discard: Callable[[RecoverableOverflow], None],
    ) -> AttemptResult:
        epoch = self.run_state.retry_epoch
        logical_base = self.run_state.attempt_logical_base
        while True:
            address = AttemptAddress(epoch, logical_base)
            try:
                return attempt(self.root_count, address)
            except RecoverableOverflow as local:
                failure = local if self.rank_common is None else self.rank_common(local)
                # The callback must reset every live attempt-owned owner before
                # another address is exposed. Completed replica lanes are not
                # passed to this callback.
                discard(failure)
                if self.root_count == self.minimum_root_count:
                    raise TrainingError(
                        "minimum sequential root_count overflowed fixed capacity"
                    ) from failure
                replacement = self._lower(self.root_count)
                replacement_address = AttemptAddress(
                    epoch + 1,
                    logical_base + self.root_count,
                )
                record = OverflowRecoveryRecord(
                    severity="critical",
                    iteration=self.iteration,
                    replica=self.replica,
                    failed_phase=failure.failure.phase,
                    failed_address=AttemptAddress(
                        epoch, failure.failure.logical_address
                    ).as_text(),
                    replacement_address=replacement_address.as_text(),
                    old_root_count=self.root_count,
                    new_root_count=replacement,
                    capacity=failure.capacity,
                    observed_demand=failure.observed_demand,
                )
                self.run_state.recovered_overflow = True
                self.run_state.root_count = replacement
                self.run_state.retry_epoch = replacement_address.retry_epoch
                self.run_state.attempt_logical_base = replacement_address.logical_base
                self.run_state.overflow_recoveries.append(record.as_dict())
                payload = json.dumps(record.as_dict(), sort_keys=True)
                self.logger.critical("RECOVERED_SEQUENTIAL_OVERFLOW %s", payload)
                warnings.warn(
                    f"RECOVERED_SEQUENTIAL_OVERFLOW {payload}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.root_count = replacement
                epoch = replacement_address.retry_epoch
                logical_base = replacement_address.logical_base


@dataclass(frozen=True)
class IterationResult:
    losses: tuple[dict[str, float], ...]
    optimizer_steps: tuple[OptimizerStep, ...]
    statistical_presentations: int


class SequentialSampledIteration:
    """Compose aligned sampling, one learner op, and dense updates."""

    def __init__(
        self,
        *,
        model: nn.Module,
        learner: LearnerOperation,
        optimizer: DenseAdam,
        run_state: RunState | None = None,
        physical_batch_cap: int | Mapping[str, int] | None = None,
    ) -> None:
        self.model = model
        self.learner = learner
        self.optimizer = optimizer
        self.run_state = run_state
        if isinstance(physical_batch_cap, Mapping):
            caps = dict(physical_batch_cap)
            if not caps or any(
                not isinstance(name, str)
                or not name
                or isinstance(cap, bool)
                or not isinstance(cap, int)
                or cap <= 0
                for name, cap in caps.items()
            ):
                raise ValueError(
                    "physical_batch_cap mapping must contain positive integer caps"
                )
            self.physical_batch_cap: int | Mapping[str, int] | None = MappingProxyType(
                caps
            )
        else:
            if physical_batch_cap is not None and (
                isinstance(physical_batch_cap, bool)
                or not isinstance(physical_batch_cap, int)
                or physical_batch_cap <= 0
            ):
                raise ValueError("physical_batch_cap must be positive")
            self.physical_batch_cap = physical_batch_cap

    def optimize(
        self,
        sampler: SequentialBatchSampler,
        *,
        iteration: int,
        reference_operation=None,
        reference_caches: Mapping[str, ReferenceOutputCache] | None = None,
    ) -> IterationResult:
        metric_names = (
            "total",
            "policy",
            "value",
            "entropy",
            "reference_kl",
            "approximate_kl",
            "clip_fraction",
        )
        loss_tensors: list[dict[str, Tensor]] = []
        steps: list[OptimizerStep] = []
        presentations = 0
        active_logical_step: int | None = None
        weighted_metrics: dict[str, Tensor] = {}

        def finish_logical_step() -> None:
            if active_logical_step is None:
                return
            steps.append(self.optimizer.step())
            loss_tensors.append(dict(weighted_metrics))

        for batch in sampler.batches():
            if batch.logical_step != active_logical_step:
                finish_logical_step()
                self.optimizer.zero_grad()
                active_logical_step = batch.logical_step
                weighted_metrics = {}
            configured_cap = self.physical_batch_cap
            if isinstance(configured_cap, Mapping):
                if batch.input_type not in configured_cap:
                    raise ValueError(
                        f"physical_batch_cap has no entry for {batch.input_type!r}"
                    )
                configured_cap = configured_cap[batch.input_type]
            cap = (
                batch.rows
                if configured_cap is None
                else min(batch.rows, configured_cap)
            )
            for begin in range(0, batch.rows, cap):
                end = min(begin + cap, batch.rows)
                physical = SequentialLearnerBatch(
                    batch.input_type,
                    batch.indices[begin:end],
                    MappingProxyType(
                        {name: value[begin:end] for name, value in batch.fields.items()}
                    ),
                )
                weight = physical.rows / batch.objective_rows
                result: LossResult = self.learner.forward_loss_backward(
                    physical,
                    iteration=iteration,
                    reference_operation=(
                        reference_operation if reference_caches is None else None
                    ),
                    reference_cache=(
                        None
                        if reference_caches is None
                        else reference_caches.get(batch.input_type)
                    ),
                    loss_scale=self.optimizer.scaler.scale,
                    objective_weight=weight,
                )
                for name in metric_names:
                    value = getattr(result, name).detach()
                    weighted_metrics[name] = (
                        weighted_metrics.get(name, 0.0) + weight * value
                    )
            presentations += batch.rows
        finish_logical_step()
        if loss_tensors:
            host_metrics = (
                torch.stack(
                    tuple(
                        torch.stack(tuple(values[name] for name in metric_names))
                        for values in loss_tensors
                    )
                )
                .to(device="cpu")
                .tolist()
            )
            losses = tuple(
                {name: float(row[index]) for index, name in enumerate(metric_names)}
                for row in host_metrics
            )
        else:
            losses = ()
        if self.run_state is not None:
            self.run_state.logical_work["minibatches"] += len(steps)
            self.run_state.logical_work["optimizer_steps"] += sum(
                int(step.applied) for step in steps
            )
            self.run_state.iteration = iteration + 1
            self.run_state.metrics_prefix.append(
                {
                    "iteration": iteration,
                    "statistical_presentations": presentations,
                    "minibatches": len(steps),
                    "optimizer_steps": sum(int(step.applied) for step in steps),
                }
            )
        return IterationResult(losses, tuple(steps), presentations)


class BranchedSampledIteration:
    """Compose ragged decision sampling with the shared learner and optimizer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        learner: LearnerOperation,
        optimizer: DenseAdam,
        sigma_pack_cap: int,
        action_pack_cap: int,
        run_state: RunState | None = None,
    ) -> None:
        if not learner.mode.rollout_mode.is_branched:
            raise ValueError("branched iteration requires a branched learner")
        if min(sigma_pack_cap, action_pack_cap) <= 0:
            raise ValueError("branched physical pack caps must be positive")
        self.model = model
        self.learner = learner
        self.optimizer = optimizer
        self.sigma_pack_cap = int(sigma_pack_cap)
        self.action_pack_cap = int(action_pack_cap)
        self.run_state = run_state

    def optimize(
        self,
        sampler: BranchedBatchSampler,
        *,
        iteration: int,
        reference_operation=None,
    ) -> IterationResult:
        metric_names = (
            "total",
            "policy",
            "value",
            "entropy",
            "reference_kl",
            "approximate_kl",
            "clip_fraction",
        )
        metric_tensors: list[Tensor] = []
        steps: list[OptimizerStep] = []
        presentations = 0
        active_logical_step: int | None = None
        weighted: dict[str, Tensor] = {}

        def finish_step() -> None:
            if active_logical_step is None:
                return
            steps.append(self.optimizer.step())
            metric_tensors.append(
                torch.stack(tuple(weighted[name] for name in metric_names))
            )

        for batch in sampler.batches():
            if batch.logical_step != active_logical_step:
                finish_step()
                self.optimizer.zero_grad()
                active_logical_step = batch.logical_step
                weighted = {}
            if (
                batch.rows > self.sigma_pack_cap
                or batch.action_rows > self.action_pack_cap
            ):
                raise TrainingError(
                    "ragged learner batch exceeds an admitted physical pack"
                )
            weight = batch.rows / batch.objective_rows
            result = self.learner.forward_loss_backward(
                batch,
                iteration=iteration,
                reference_operation=reference_operation,
                loss_scale=self.optimizer.scaler.scale,
                objective_weight=weight,
            )
            for name in metric_names:
                weighted[name] = (
                    weighted.get(name, 0.0) + weight * getattr(result, name).detach()
                )
            presentations += batch.rows
        finish_step()
        losses = (
            tuple(
                {name: float(row[index]) for index, name in enumerate(metric_names)}
                for row in torch.stack(metric_tensors).cpu().tolist()
            )
            if metric_tensors
            else ()
        )
        if self.run_state is not None:
            self.run_state.logical_work["minibatches"] += len(steps)
            self.run_state.logical_work["optimizer_steps"] += sum(
                int(step.applied) for step in steps
            )
            self.run_state.iteration = iteration + 1
            self.run_state.metrics_prefix.append(
                {
                    "iteration": iteration,
                    "statistical_presentations": presentations,
                    "minibatches": len(steps),
                    "optimizer_steps": sum(int(step.applied) for step in steps),
                }
            )
        return IterationResult(losses, tuple(steps), presentations)


class PreemptionNotice:
    """Signal handlers record a notice; boundaries own checkpoint decisions."""

    def __init__(self) -> None:
        self.requested_signal: int | None = None
        self._previous: dict[int, object] = {}

    def _handler(self, signum: int, _frame: object) -> None:
        self.requested_signal = signum

    def install(self) -> None:
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)

    def restore(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()

    @property
    def requested(self) -> bool:
        return self.requested_signal is not None


def convert_value_head_units(
    state: Mapping[str, Tensor],
    *,
    stack_bb: float,
    source: str,
    destination: str,
) -> dict[str, Tensor]:
    """Convert only final pre/post value affine weights and biases."""

    if (
        not math.isfinite(stack_bb)
        or stack_bb <= 0
        or source not in ("raw_bb", "stack_normalized")
        or destination not in ("raw_bb", "stack_normalized")
    ):
        raise ValueError("utility conversion arguments are invalid")
    result = {name: value.detach().clone() for name, value in state.items()}
    if source == destination:
        return result
    scale = stack_bb if source == "stack_normalized" else 1.0 / stack_bb
    selected = tuple(
        name
        for name in result
        if (name.startswith("preflop.") or name.startswith("postflop."))
        and (
            name.endswith("value_head.affine.weight")
            or name.endswith("value_head.affine.bias")
        )
    )
    if len(selected) != 4:
        raise ValueError(
            "utility conversion requires exactly two final value affine layers"
        )
    for name in selected:
        result[name].mul_(scale)
    return result


__all__ = [
    "AttemptAddress",
    "BranchedSampledIteration",
    "IterationResult",
    "OverflowRecoveryRecord",
    "PreemptionNotice",
    "RecoverableOverflow",
    "SequentialOverflowRecovery",
    "SequentialSampledIteration",
    "TrainingError",
    "convert_value_head_units",
]
