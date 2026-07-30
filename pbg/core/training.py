"""Phase 5 iteration composition and whole-attempt overflow recovery."""

from __future__ import annotations

import copy
import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor

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
        learner: LearnerOperation,
        optimizer: DenseAdam,
        physical_batch_cap: int | Mapping[str, int] | None = None,
        sigma_pack_cap: int | None = None,
        action_pack_cap: int | None = None,
    ) -> None:
        self.learner = learner
        self.optimizer = optimizer
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
        return IterationResult(losses, tuple(steps), presentations)

    @staticmethod
    def commit_run_state(
        run_state: RunState,
        result: IterationResult,
        *,
        iteration: int,
    ) -> None:
        """Publish counters only after the complete learner transaction succeeds."""

        if run_state.iteration != iteration:
            raise TrainingError("learner run state is outside the commit iteration")
        applied = sum(int(step.applied) for step in result.optimizer_steps)
        run_state.logical_work["minibatches"] += len(result.optimizer_steps)
        run_state.logical_work["optimizer_steps"] += applied
        run_state.iteration = iteration + 1
        run_state.metrics_prefix.append(
            {
                "iteration": iteration,
                "statistical_presentations": result.statistical_presentations,
                "minibatches": len(result.optimizer_steps),
                "optimizer_steps": applied,
            }
        )


class TransactionalOptimizer(Protocol):
    """Mutable optimizer owner with one accelerator-resident rollback point."""

    def begin_transaction(self) -> None: ...

    def rollback_transaction(self) -> None: ...

    def commit_transaction(self) -> None: ...


@dataclass(frozen=True)
class PreparedLearner:
    """One presentation-specific sampler plugged into the common learner."""

    iteration: LearnerIteration
    sampler: SequentialBatchSampler | BranchedBatchSampler
    reference_operation: object | None = None
    reference_caches: Mapping[str, ReferenceOutputCache] | None = None
    release: Callable[[], None] = lambda: None

    def optimize(self, *, iteration: int) -> IterationResult:
        return self.iteration.optimize(
            self.sampler,
            iteration=iteration,
            reference_operation=self.reference_operation,
            reference_caches=self.reference_caches,
        )


class LearnerTransaction:
    """One real learner pass per lane with all-or-nothing publication."""

    def __init__(
        self,
        *,
        optimizer: TransactionalOptimizer,
        generators: Sequence[torch.Generator],
        run_states: Sequence[RunState | None],
        distributed: object | None = None,
        device: torch.device,
        error_type: type[Exception] = TrainingError,
    ) -> None:
        self.optimizer = optimizer
        self.generators = tuple(generators)
        self.run_states = tuple(run_states)
        self.distributed = distributed
        self.device = device
        self.error_type = error_type
        if (
            not self.generators
            or len(self.generators) != len(self.run_states)
            or any(generator.device != device for generator in self.generators)
        ):
            raise ValueError(
                "learner transaction requires aligned device-local lanes"
            )

    def _common_failure(self, lane: int, failed: bool) -> bool:
        if self.distributed is None:
            return failed
        local = torch.zeros(
            len(self.generators),
            dtype=torch.bool,
            device=self.device,
        )
        local[lane] = failed
        return bool(self.distributed.agree_failures(local).any())

    @staticmethod
    def _restore_run_state(
        destination: RunState,
        state: Mapping[str, object],
    ) -> None:
        restored = RunState(**copy.deepcopy(dict(state)))
        for name in destination.__dataclass_fields__:
            setattr(destination, name, getattr(restored, name))

    def optimize(
        self,
        prepare_lane: Callable[[int], PreparedLearner],
        *,
        iteration: int,
    ) -> tuple[IterationResult, ...]:
        """Execute each lane once and commit only after every lane/rank succeeds."""

        generator_states = tuple(
            generator.get_state() for generator in self.generators
        )
        run_state_snapshots = tuple(
            None if state is None else copy.deepcopy(state.state_dict())
            for state in self.run_states
        )
        prepared: list[PreparedLearner | None] = [
            None for _ in self.generators
        ]
        results: list[IterationResult] = []
        transaction_open = False
        failure: Exception | None = None
        try:
            self.optimizer.begin_transaction()
            transaction_open = True
            for lane in range(len(prepared)):
                local_failure: Exception | None = None
                try:
                    prepared[lane] = prepare_lane(lane)
                except Exception as exc:  # noqa: BLE001 - lane transaction
                    local_failure = exc
                if self._common_failure(lane, local_failure is not None):
                    failure = (
                        local_failure
                        if local_failure is not None
                        else TrainingError(
                            "learner preparation failed on another rank"
                        )
                    )
                    break
            if failure is None:
                for lane, work in enumerate(prepared):
                    if work is None:
                        raise TrainingError(
                            "learner transaction omitted a prepared lane"
                        )
                    local_failure = None
                    result: IterationResult | None = None
                    try:
                        result = work.optimize(iteration=iteration)
                    except Exception as exc:  # noqa: BLE001 - lane transaction
                        local_failure = exc
                    if self._common_failure(lane, local_failure is not None):
                        failure = (
                            local_failure
                            if local_failure is not None
                            else TrainingError(
                                "learner execution failed on another rank"
                            )
                        )
                        break
                    if result is None:
                        raise TrainingError(
                            "successful learner lane omitted its result"
                        )
                    results.append(result)
            if failure is not None:
                raise failure
            for state, result in zip(
                self.run_states, results, strict=True
            ):
                if state is not None:
                    LearnerIteration.commit_run_state(
                        state,
                        result,
                        iteration=iteration,
                    )
            self.optimizer.commit_transaction()
            transaction_open = False
            return tuple(results)
        except Exception as exc:
            if transaction_open:
                self.optimizer.rollback_transaction()
            for generator, state in zip(
                self.generators, generator_states, strict=True
            ):
                generator.set_state(state)
            for run_state, snapshot in zip(
                self.run_states, run_state_snapshots, strict=True
            ):
                if run_state is not None and snapshot is not None:
                    self._restore_run_state(run_state, snapshot)
            raise self.error_type(
                "learner transaction failed; every transition was rolled back"
            ) from exc
        finally:
            for work in prepared:
                if work is not None:
                    work.release()


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
    "LearnerTransaction",
    "OverflowRecoveryRecord",
    "PreemptionNotice",
    "PreparedLearner",
    "RecoverableOverflow",
    "TrainingError",
]
