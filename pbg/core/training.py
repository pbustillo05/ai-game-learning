"""Phase 5 iteration composition and whole-attempt overflow recovery."""

from __future__ import annotations

import signal
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn

from pbg.core.learner import (
    BranchedBatchSampler,
    LearnerOperation,
    LossResult,
    ReferenceOutputCache,
    SequentialBatchSampler,
    SequentialLearnerBatch,
)
from pbg.core.optimization import DenseAdam, OptimizerStep
from pbg.core.persistence import RunState
from pbg.core.storage import DeviceFailure


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


@dataclass(frozen=True)
class IterationResult:
    losses: tuple[dict[str, float], ...]
    optimizer_steps: tuple[OptimizerStep, ...]
    statistical_presentations: int


class LearnerIteration:
    """One optimizer state machine for aligned and ragged decision batches."""

    _METRICS = (
        "total",
        "policy",
        "value",
        "entropy",
        "reference_kl",
        "approximate_kl",
        "clip_fraction",
    )

    def __init__(
        self,
        *,
        model: nn.Module,
        learner: LearnerOperation,
        optimizer: DenseAdam,
        run_state: RunState | None = None,
        physical_batch_cap: int | Mapping[str, int] | None = None,
        sigma_pack_cap: int | None = None,
        action_pack_cap: int | None = None,
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
                    "physical batch caps must be positive integer rows"
                )
            self.physical_batch_cap: int | Mapping[str, int] | None = (
                MappingProxyType(caps)
            )
        else:
            if physical_batch_cap is not None and (
                isinstance(physical_batch_cap, bool)
                or not isinstance(physical_batch_cap, int)
                or physical_batch_cap <= 0
            ):
                raise ValueError("physical batch cap must be positive")
            self.physical_batch_cap = physical_batch_cap
        if (sigma_pack_cap is None) != (action_pack_cap is None):
            raise ValueError("ragged sigma/action pack caps must be paired")
        if sigma_pack_cap is not None and min(
            sigma_pack_cap, action_pack_cap  # type: ignore[arg-type]
        ) <= 0:
            raise ValueError("ragged pack caps must be positive")
        self.sigma_pack_cap = sigma_pack_cap
        self.action_pack_cap = action_pack_cap

    def _physical_batches(self, batch):
        if isinstance(batch, SequentialLearnerBatch):
            configured = self.physical_batch_cap
            if isinstance(configured, Mapping):
                if batch.input_type not in configured:
                    raise ValueError(
                        "physical batch caps omit "
                        f"{batch.input_type!r}"
                    )
                configured = configured[batch.input_type]
            cap = (
                batch.rows
                if configured is None
                else min(batch.rows, configured)
            )
            for begin in range(0, batch.rows, cap):
                end = min(begin + cap, batch.rows)
                physical = SequentialLearnerBatch(
                    batch.input_type,
                    batch.indices[begin:end],
                    MappingProxyType(
                        {
                            name: value[begin:end]
                            for name, value in batch.fields.items()
                        }
                    ),
                )
                yield physical, physical.rows / batch.objective_rows
            return
        if self.sigma_pack_cap is None or self.action_pack_cap is None:
            raise TrainingError("ragged batch reached an aligned-only learner")
        if (
            batch.rows > self.sigma_pack_cap
            or batch.action_rows > self.action_pack_cap
        ):
            raise TrainingError(
                "ragged learner batch exceeds an admitted physical pack"
            )
        yield batch, batch.objective_weight

    def optimize(
        self,
        sampler: SequentialBatchSampler | BranchedBatchSampler,
        *,
        iteration: int,
        reference_operation=None,
        reference_caches: Mapping[str, ReferenceOutputCache] | None = None,
    ) -> IterationResult:
        metric_rows: list[Tensor] = []
        steps: list[OptimizerStep] = []
        presentations = 0
        active_step: int | None = None
        weighted: dict[str, Tensor] = {}
        step_error: Exception | None = None
        agree_failure = getattr(self.optimizer, "agree_failure", None)
        common_step_count = getattr(self.optimizer, "common_step_count", None)
        expected = sampler.optimizer_steps
        if common_step_count is not None:
            expected = common_step_count(expected)

        def finish_step() -> None:
            nonlocal step_error
            if active_step is None:
                return
            if agree_failure is not None and agree_failure(
                step_error is not None
            ):
                self.optimizer.zero_grad()
                error = step_error
                step_error = None
                if error is None:
                    raise TrainingError(
                        "learner step failed on another distributed rank"
                    )
                if getattr(self.optimizer, "distributed", None) is None:
                    raise TrainingError(
                        "learner step failed before optimizer transition"
                    ) from error
                raise TrainingError(
                    "learner step failed before distributed SUM"
                ) from error
            steps.append(self.optimizer.step())
            metric_rows.append(
                torch.stack(tuple(weighted[name] for name in self._METRICS))
            )
            step_error = None

        for batch in sampler.batches():
            if batch.logical_step != active_step:
                finish_step()
                self.optimizer.zero_grad()
                active_step = batch.logical_step
                weighted = {}
            if step_error is None:
                try:
                    for physical, weight in self._physical_batches(batch):
                        result: LossResult = (
                            self.learner.forward_loss_backward(
                                physical,
                                iteration=iteration,
                                reference_operation=(
                                    reference_operation
                                    if reference_caches is None
                                    else None
                                ),
                                reference_cache=(
                                    None
                                    if reference_caches is None
                                    else reference_caches.get(
                                        batch.input_type
                                    )
                                ),
                                loss_scale=self.optimizer.scaler.scale,
                                objective_weight=weight,
                            )
                        )
                        for name in self._METRICS:
                            weighted[name] = (
                                weighted.get(name, 0.0)
                                + weight * getattr(result, name).detach()
                            )
                except Exception as exc:
                    if agree_failure is None:
                        raise
                    step_error = exc
            presentations += batch.rows
        finish_step()
        while len(steps) < expected:
            self.optimizer.zero_grad()
            if agree_failure is not None and agree_failure(False):
                raise TrainingError(
                    "learner step failed on another distributed rank"
                )
            steps.append(self.optimizer.step())
        if len(steps) != expected:
            raise TrainingError(
                "learner sampler exceeded its rank-common optimizer budget"
            )
        losses = (
            tuple(
                {
                    name: float(row[index])
                    for index, name in enumerate(self._METRICS)
                }
                for row in torch.stack(metric_rows).cpu().tolist()
            )
            if metric_rows
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
                    "optimizer_steps": sum(
                        int(step.applied) for step in steps
                    ),
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


__all__ = [
    "AttemptAddress",
    "IterationResult",
    "LearnerIteration",
    "OverflowRecoveryRecord",
    "PreemptionNotice",
    "RecoverableOverflow",
    "TrainingError",
]
