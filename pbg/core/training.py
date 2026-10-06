"""Phase 5 iteration composition and whole-attempt overflow recovery."""

from __future__ import annotations

import copy
import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch

from pbg.core.learner import (
    BranchedBatchSampler,
    LearnerOperation,
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
        "negative_advantage_cap_fraction",
    )
    _MAX_METRICS = (
        "negative_advantage_log_ratio_max",
        "negative_advantage_influence_max",
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
                yield (
                    physical,
                    batch.objective_weight * physical.rows / batch.rows,
                )
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
        work = PreparedLearner(self, sampler, reference_operation, reference_caches)
        return optimize_prepared((work,), iteration=iteration, optimizer=self.optimizer,
                                 distributed=getattr(self.optimizer, "distributed", None))[0]

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


def _optimizer_binding(value):
    """Resolve scheduling views to the sole dense device optimizer."""
    if isinstance(value, DenseAdam):
        return value, None
    group = getattr(value, "group", value)
    owner = getattr(group, "optimizer", None)
    if isinstance(owner, DenseAdam):
        return owner, group
    owner = getattr(value, "owner", None)
    if isinstance(owner, DenseAdam):
        return owner, None
    raise TrainingError("learner requires a dense tensor optimizer owner")


def _operation_binding(operation):
    module = operation if isinstance(operation, torch.nn.Module) else getattr(operation, "__self__", None)
    reference = getattr(module, "_replica_owner_ref", None)
    owner = None if reference is None else reference()
    if owner is None:
        return None
    return owner, module._replica_lane, ("__call__" if operation is module else operation.__name__)


def _cohort_operation(operations):
    first = operations[0]
    binding = _operation_binding(first)
    if binding is None:
        if len(operations) != 1:
            raise TrainingError("unbound model operation cannot form a replica cohort")
        return first
    owner, lane, method = binding
    if any(_operation_binding(op) != (owner, lane + offset, method)
           for offset, op in enumerate(operations)):
        raise TrainingError("model cohort lost contiguous parameter ownership")
    return getattr(owner.cohort_module(lane, len(operations)), method)


def _merge_learner_batches(batches):
    from pbg.core.learner import BranchedLearnerBatch

    first = batches[0]
    if len(batches) == 1:
        return first

    def fields(name):
        mappings = [getattr(batch, name) for batch in batches]
        if any(tuple(mapping) != tuple(mappings[0]) for mapping in mappings):
            raise TrainingError("cohort learner fields differ")
        return MappingProxyType({key: torch.cat([mapping[key] for mapping in mappings])
                                 for key in mappings[0]})

    indices = torch.cat([batch.sigma_index for batch in batches])
    if isinstance(first, SequentialLearnerBatch):
        return SequentialLearnerBatch(first.input_type, indices, fields("fields"))
    pointers = torch.cat([batch.action_ptr[:-1] + lane * first.action_rows
                          for lane, batch in enumerate(batches)] +
                         [first.action_ptr.new_tensor([len(batches) * first.action_rows])])
    return BranchedLearnerBatch(first.input_type, indices, pointers,
                               torch.cat([batch.action_index for batch in batches]),
                               fields("sigma_fields"), fields("action_fields"))


def _learner_cohort_key(work, batch):
    learner = work.iteration.learner
    binding = _operation_binding(learner.model_operation)
    model = (id(learner.model_operation),) if binding is None else (id(binding[0]), binding[2])
    policy = getattr(learner.policy_operation, "cohort_key", id(learner.policy_operation))
    reference = _operation_binding(work.reference_operation)
    reference_key = ((id(work.reference_operation),) if reference is None
                     else (id(reference[0]), reference[2]))
    schema = tuple((name, tuple(value.shape[1:]), value.dtype)
                   for mapping in (batch.sigma_fields, batch.action_fields)
                   for name, value in mapping.items())
    return (type(batch), batch.input_type, batch.rows, batch.action_rows,
            learner.mode.key, learner.precision, policy, model, reference_key,
            work.reference_caches is not None, learner.loss.branching is not None, schema)


def _run_learner_cohort(works, packets, *, iteration, owner, lanes):
    from pbg.core.contracts import PrivateEstimator
    from pbg.core.learner import _select_sampled_private

    batch = _merge_learner_batches([packet[1] for packet in packets])
    learners = [work.iteration.learner for work in works]
    first = learners[0]
    sampled = first.mode.private_estimator is PrivateEstimator.SAMPLED
    sigma = batch.sigma_fields
    private = sigma["sampled_private_state"] if sampled else sigma["owner"][:, None]
    operation = _cohort_operation([learner.model_operation for learner in learners])
    controls = first.controls(tuple(learner.loss for learner in learners), iteration, owner.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[first.precision]
    reference_raw = None
    with torch.autocast("cuda", dtype=dtype, enabled=first.precision != "fp32"):
        with torch.no_grad():
            if works[0].reference_caches is not None:
                references = []
                for work, packet in zip(works, packets, strict=True):
                    physical = packet[1]
                    cached = work.reference_caches[physical.input_type].gather(physical.sigma_index)
                    if sampled:
                        cached = {name: _select_sampled_private(value,
                                  physical.sigma_fields["sampled_private_state"],
                                  physical.sigma_fields["owner"]) for name, value in cached.items()}
                    references.append(cached)
                reference_raw = {name: torch.cat([ref[name] for ref in references]) for name in references[0]}
            elif works[0].reference_operation is not None:
                reference = _cohort_operation([work.reference_operation for work in works])
                raw = reference(batch.input_type, sigma["model_input"], private, sigma["owner"]).policy_raw
                reference_raw = ({name: _select_sampled_private(value, private, sigma["owner"])
                                  for name, value in raw.items()} if sampled else raw)
    weights = torch.stack([torch.as_tensor(packet[2], dtype=torch.float32, device=owner.device)
                           for packet in packets])
    torch._assert_async((torch.isfinite(weights) & weights.gt(0) & weights.le(1)).all(),
                        "physical objective weights must lie in (0,1]")
    lane_index = torch.tensor(lanes, dtype=torch.int64, device=owner.device)
    scales = owner.controls.index_select(0, lane_index)[:, 6]
    sequential = isinstance(batch, SequentialLearnerBatch)
    sigma_names = tuple(batch.sigma_fields)
    action_names = () if sequential else tuple(batch.action_fields)
    reference_names = () if reference_raw is None else tuple(reference_raw)
    input_type, count = batch.input_type, len(works)
    names = (*LearnerIteration._METRICS, *LearnerIteration._MAX_METRICS)

    def program(*values):
        from pbg.core.learner import BranchedLearnerBatch

        sigma_index, action_ptr, action_index = values[:3]
        offset = 3
        sigma = dict(zip(sigma_names, values[offset:offset + len(sigma_names)], strict=True))
        offset += len(sigma_names)
        action = dict(zip(action_names, values[offset:offset + len(action_names)], strict=True))
        offset += len(action_names)
        raw = dict(zip(reference_names, values[offset:offset + len(reference_names)], strict=True))
        offset += len(reference_names)
        loss_controls, objective_weights, loss_scales = values[offset:]
        physical = (SequentialLearnerBatch(input_type, sigma_index, sigma) if sequential else
                    BranchedLearnerBatch(input_type, sigma_index, action_ptr, action_index, sigma, action))
        private = sigma["sampled_private_state"] if sampled else sigma["owner"][:, None]
        with torch.autocast("cuda", dtype=dtype, enabled=first.precision != "fp32"):
            evaluation = operation(input_type, sigma["model_input"], private, sigma["owner"])
            result = first.forward_loss(physical, iteration=0, lane_count=count,
                                        loss_controls=loss_controls, evaluation=evaluation,
                                        reference_raw=raw if reference_names else None)
        metrics = torch.stack([getattr(result, name).detach() for name in names], dim=1)
        metrics = torch.cat((metrics[:, :8] * objective_weights[:, None], metrics[:, 8:]), dim=1)
        return result.total.detach(), metrics, (result.total * objective_weights * loss_scales).sum()

    values = (batch.sigma_index, batch.action_ptr, batch.action_index,
              *batch.sigma_fields.values(),
              *(() if sequential else batch.action_fields.values()),
              *(() if reference_raw is None else reference_raw.values()), controls, weights, scales)
    key = (_learner_cohort_key(works[0], packets[0][1]), tuple(lanes), reference_names, 2)
    total, metrics = owner.program_runner.run(key, program, values,
        mutable=(owner.gradient_rows, *(buffer for module in owner.modules for buffer in module.buffers())))
    work = owner.program_runner.work_counts.setdefault(input_type,
        {"sigma_presentations": 0, "action_presentations": 0, "cohort_calls": 0,
         "reference_forward_sigma_presentations": 0, "reference_cache_sigma_presentations": 0})
    work["sigma_presentations"] += batch.rows
    work["action_presentations"] += batch.action_rows
    work["cohort_calls"] += 1
    if reference_raw is not None:
        reference_kind = "cache" if works[0].reference_caches is not None else "forward"
        work[f"reference_{reference_kind}_sigma_presentations"] += batch.rows
    owner.fatal[lane_index] |= ~torch.isfinite(total)
    for gradient_owner in owner.gradient_owners:
        gradient_owner.assert_bound()
    return metrics


def _ready_learner_cohort(ready, *, lanes, keys, family_ranks, batched):
    """Flush earlier FIFO families and batch the largest exact contiguous run.

    A narrow tail must not let the lowest lane consume all of its next family
    while other lanes are still finishing the previous one. These are bounded
    host scheduling metadata; no rows, draws, or per-lane FIFO order change.
    """
    priority = min(family_ranks[index] for index in ready)
    groups = []
    for index in ready:
        if family_ranks[index] != priority:
            continue
        if (groups and lanes[index] == lanes[groups[-1][-1]] + 1
                and keys[index] == keys[groups[-1][0]]
                and batched[index] and batched[groups[-1][0]]):
            groups[-1].append(index)
        else:
            groups.append([index])
    return max(groups, key=lambda indices: (len(indices), -indices[0]))


def optimize_prepared(works, *, iteration, optimizer, distributed=None):
    """Logical-step FIFO scheduling of exact family cohorts, including K=1."""
    owner, group = _optimizer_binding(optimizer)
    lanes = tuple(getattr(work.iteration.optimizer, "lane", index) for index, work in enumerate(works))
    if len(set(lanes)) != len(lanes) or any(lane < 0 or lane >= owner.count for lane in lanes):
        raise TrainingError("prepared learner lanes do not match optimizer ownership")
    budgets = [work.sampler.optimizer_steps for work in works]
    if distributed is not None:
        budgets = [distributed.maximum_steps(count, device=owner.device) for count in budgets]
    metrics = [[] for _ in works]
    transitions = [[] for _ in works]
    presentations = [0 for _ in works]

    def packets(work):
        for batch in work.sampler.batches():
            for physical, weight in work.iteration._physical_batches(batch):
                yield batch.logical_step, physical, weight

    streams = [iter(packets(work)) for work in works]
    pending = [None for _ in works]
    ended = [False for _ in works]
    # Production samplers expose the canonical store/FIFO order, including
    # empty families. Diagnostic samplers discover their order on first use.
    family_orders = [{name: rank for rank, name in enumerate(getattr(work.sampler, "stores", ()))}
                     for work in works]
    batched = {index: _operation_binding(work.iteration.learner.model_operation) is not None
               for index, work in enumerate(works)}
    for logical_step in range(max(budgets, default=0)):
        active = [index for index, budget in enumerate(budgets) if logical_step < budget]
        mask = torch.zeros(owner.count, dtype=torch.bool, device=owner.device)
        mask[[lanes[index] for index in active]] = True
        owner.gradient_rows.masked_fill_(mask[:, None], 0)
        weighted = torch.zeros((len(works), 10), dtype=torch.float32, device=owner.device)
        seen = set()
        failure = None
        try:
            while True:
                for index in active:
                    if pending[index] is None and not ended[index]:
                        pending[index] = next(streams[index], None)
                        ended[index] = pending[index] is None
                    if pending[index] is not None and pending[index][0] < logical_step:
                        raise TrainingError("learner FIFO logical step moved backwards")
                ready = [index for index in active if pending[index] is not None
                         and pending[index][0] == logical_step]
                if not ready:
                    break
                keys, family_ranks = {}, {}
                for index in ready:
                    physical = pending[index][1]
                    keys[index] = _learner_cohort_key(works[index], physical)
                    order = family_orders[index]
                    family_ranks[index] = order.setdefault(physical.input_type, len(order))
                cohort = _ready_learner_cohort(ready, lanes=lanes, keys=keys,
                                              family_ranks=family_ranks, batched=batched)
                result = _run_learner_cohort([works[index] for index in cohort],
                    [pending[index] for index in cohort], iteration=iteration, owner=owner,
                    lanes=[lanes[index] for index in cohort])
                for row, index in enumerate(cohort):
                    weighted[index, :8].add_(result[row, :8])
                    weighted[index, 8:] = (torch.maximum(weighted[index, 8:], result[row, 8:])
                                           if index in seen else result[row, 8:])
                    seen.add(index)
                    presentations[index] += pending[index][1].rows
                    pending[index] = None
        except Exception as exc:
            failure = exc
        failed = owner.fatal.clone()
        if failure is not None:
            failed.logical_or_(mask)
        if distributed is not None:
            failed = distributed.agree_failures(failed)
        # This is the rank-common failure boundary before dense gradient SUM.
        if failure is not None or (distributed is not None and bool(failed.any())):
            raise TrainingError("learner failed before optimizer transition or distributed SUM") from failure
        if distributed is not None:
            distributed.sum_gradients(owner.gradient_rows, mask)
        transition = None
        try:
            transition = owner.step_device(mask)
            if group is not None and hasattr(group, "versions"):
                group.versions.add_(transition[0].to(group.versions.dtype))
            for index in active:
                transitions[index].append(tuple(value[lanes[index]].clone() for value in transition))
                if index in seen:
                    metrics[index].append(weighted[index].clone())
        except Exception as exc:
            failure = exc
        failed = owner.fatal.clone()
        if failure is not None:
            failed.logical_or_(mask)
        if distributed is not None:
            failed = distributed.agree_failures(failed)
        if failure is not None or (distributed is not None and bool(failed.any())):
            raise TrainingError("learner optimizer failed before rank-common publication") from failure
    failed = owner.fatal.clone()
    if any(packet is not None for packet in pending):
        failed.fill_(True)
    if distributed is not None:
        failed = distributed.agree_failures(failed)
    if bool(failed.any()):
        raise TrainingError("learner failed or exceeded its optimizer budget before publication")
    names = (*LearnerIteration._METRICS, *LearnerIteration._MAX_METRICS)
    results = []
    for index in range(len(works)):
        losses = tuple(dict(zip(names, row, strict=True))
                       for row in (torch.stack(metrics[index]).cpu().tolist() if metrics[index] else ()))
        steps = (DenseAdam.materialize_steps(tuple(torch.stack([row[field] for row in transitions[index]])
                 for field in range(7))) if transitions[index] else ())
        results.append(IterationResult(losses, steps, presentations[index]))
    return tuple(results)


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
                try:
                    if any(work is None for work in prepared):
                        raise TrainingError("learner transaction omitted a prepared lane")
                    results = list(optimize_prepared(
                        prepared, iteration=iteration, optimizer=self.optimizer,
                        distributed=self.distributed,
                    ))
                    for state, result in zip(self.run_states, results, strict=True):
                        if state is not None:
                            LearnerIteration.commit_run_state(state, result, iteration=iteration)
                except Exception as exc:
                    failure = exc
                # Reporting and counter publication can fail after the final
                # device update. Every rank must retain its rollback point.
                if self._common_failure(0, failure is not None):
                    if failure is None:
                        failure = TrainingError("learner publication failed on another rank")
            if failure is not None:
                raise failure
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
