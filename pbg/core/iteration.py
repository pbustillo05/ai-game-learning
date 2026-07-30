"""One production iteration lifecycle for every mode and execution dimension."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from pbg.core.branching import (
    BranchController,
    BranchedRecoveryRecord,
    RankCommonFailureAgreement,
)
from pbg.core.capacity import ExecutionPhase, GeneratedOwners, StoragePlan
from pbg.core.contracts import ModeSchema, validate_replica_group
from pbg.core.optimization import OptimizerStep
from pbg.core.persistence import RunState
from pbg.core.rollout import (
    AsyncTopologyPublisher,
    CompletionFIFO,
    CompletionHook,
    FamilyQueueState,
    FamilyScheduler,
    RolloutCommit,
    RolloutHooks,
    RolloutOwnerRegistry,
    RolloutTransaction,
    ScheduledFamily,
    SemanticFamilyFIFO,
)
from pbg.core.storage import DeviceFailure
from pbg.core.topology import (
    LayerPlan,
    Topology,
    TopologyBindings,
    build_layer_plan,
)
from pbg.core.training import (
    AttemptAddress,
    OverflowRecoveryRecord,
    RecoverableOverflow,
    TrainingError,
)


class IterationError(RuntimeError):
    """The common iteration lifecycle could not complete transactionally."""


@dataclass(frozen=True)
class LaneContext:
    """Immutable execution and recovery ownership for one replica lane."""

    mode: ModeSchema
    run_state: RunState
    generator: torch.Generator
    optimizer: object
    model_operation: object
    loss: object | None = None
    reference_operation: object | None = None
    branch_controller: BranchController | None = None
    minimum_root_count: int = 1

    def __post_init__(self) -> None:
        if self.minimum_root_count <= 0:
            raise ValueError("minimum_root_count must be positive")
        if self.mode.rollout_mode.is_branched != (
            self.branch_controller is not None
        ):
            raise ValueError(
                "branched lanes require exactly one branch controller"
            )


@dataclass(frozen=True)
class IterationConfig:
    """Closed common configuration for one multi-lane iteration."""

    plan: StoragePlan
    lanes: tuple[LaneContext, ...]

    def __post_init__(self) -> None:
        if not self.lanes:
            raise ValueError("an iteration requires at least one lane")
        if len(self.lanes) != self.plan.manifest.n_replica:
            raise ValueError("lane contexts differ from the storage plan")
        validate_replica_group(tuple(lane.mode for lane in self.lanes))
        if any(
            not lane.mode.physically_compatible_with(self.plan.mode)
            for lane in self.lanes
        ):
            raise ValueError("lane mode is physically incompatible with the plan")
        if any(
            lane.mode.rollout_mode.is_branched
            and lane.run_state.root_count != self.plan.manifest.root_count
            for lane in self.lanes
        ):
            raise ValueError("branched root count differs from the storage plan")


@dataclass(frozen=True)
class RolloutLoop:
    """One schema-selected rollout submitted to the common scheduler."""

    scheduler: FamilyScheduler
    states: Callable[[int], Sequence[FamilyQueueState]]
    has_work: Callable[[], bool]
    queues: Mapping[str, SemanticFamilyFIFO]
    registry: RolloutOwnerRegistry
    schemas: tuple[ModeSchema, ...]
    seed: int
    models: tuple[Callable[[object, torch.Tensor], Mapping[str, torch.Tensor]], ...]
    hooks: tuple[RolloutHooks, ...]
    accept: Callable[[tuple[AcceptedRollout, ...]], None]
    topology: AsyncTopologyPublisher
    failure: Callable[[int], Exception]
    inflight_cap: int
    logical_time: int = 0
    logical_limit: int | None = None


@dataclass(frozen=True)
class RolloutSubmission:
    """One lane-local transaction submitted by a game rollout hook."""

    scheduled: ScheduledFamily
    commit: RolloutCommit


@dataclass(frozen=True)
class AcceptedRollout:
    """Commit-visible queue counts accepted by the core topology authority."""

    scheduled: ScheduledFamily
    commit: RolloutCommit
    produced: tuple[int, ...]


@dataclass(frozen=True)
class SealedTopology:
    """Core-owned compact forests and their exact common layer plan."""

    topologies: tuple[Topology, ...]
    bindings: tuple[TopologyBindings, ...]
    leaf_counts: tuple[int, ...]
    plan: LayerPlan
    sampled_nodes: tuple[int, ...]
    record_counts: tuple[int, ...]
    counterfactual_corrections: tuple[float, ...]


@dataclass(frozen=True)
class CompletionLoop:
    """One evaluator FIFO submitted to the common completion scheduler."""

    completion: CompletionFIFO
    remaining: Sequence[int]
    hook: CompletionHook
    wait_for: torch.cuda.Event
    failure_message: str


class IterationAttempt(Protocol):
    """Game hooks at the semantic boundaries owned by ``IterationEngine``."""

    def prepare_rollout(self, *, iteration: int) -> RolloutLoop: ...

    def finish_rollout(
        self, *, logical_time: int, topology: SealedTopology
    ) -> None: ...

    def finish_queue_drain(self) -> None: ...

    def prepare_completion(self) -> CompletionLoop: ...

    def finish_completion(self) -> None: ...

    def align_and_seal(self) -> None: ...

    def execute_targets(self) -> None: ...

    def execute_learner(self, *, iteration: int) -> None: ...

    def finish_backward(self) -> object: ...


class IterationHooks(Protocol):
    """The game boundary used to construct one unpublished attempt."""

    def build_attempt(self, owners: GeneratedOwners) -> IterationAttempt: ...


_PHASES = tuple(ExecutionPhase)


class LearnerPreflightOptimizer:
    """Exercise learner batches without permitting a parameter transition."""

    def __init__(self, optimizer: object) -> None:
        self.optimizer = optimizer
        self.scaler = optimizer.scaler

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> OptimizerStep:
        self.optimizer.zero_grad()
        scale = self.scaler.scale
        return OptimizerStep(False, True, 0.0, 1.0, scale, scale, 0)


class IterationEngine:
    """Own phase progression, release, recovery, and close for all iterations."""

    def __init__(
        self,
        *,
        config: IterationConfig,
        owner_factory: Callable[[], GeneratedOwners],
        hooks: IterationHooks,
        rank_agreement: RankCommonFailureAgreement | None = None,
    ) -> None:
        self.config = config
        self.owner_factory = owner_factory
        self.hooks = hooks
        self.rank_agreement = (
            RankCommonFailureAgreement.from_runtime()
            if rank_agreement is None
            else rank_agreement
        )

    @staticmethod
    def schedule_rollout(
        *,
        scheduler: FamilyScheduler,
        states: Callable[[int], Sequence[FamilyQueueState]],
        has_work: Callable[[], bool],
        queues: Mapping[str, SemanticFamilyFIFO],
        registry: RolloutOwnerRegistry,
        schemas: Sequence[ModeSchema],
        seed: int,
        iteration: int,
        models: Sequence[
            Callable[[object, torch.Tensor], Mapping[str, torch.Tensor]]
        ],
        hooks: Sequence[RolloutHooks],
        accept: Callable[[tuple[AcceptedRollout, ...]], None],
        topology: AsyncTopologyPublisher,
        failure: Callable[[int], Exception],
        inflight_cap: int,
        logical_time: int = 0,
        logical_limit: int | None = None,
    ) -> int:
        """Run the sole stable-prefix scheduling loop used by poker attempts."""

        if inflight_cap <= 0 or logical_time < 0:
            raise ValueError("rollout scheduling bounds must be positive")
        if not has_work():
            return logical_time
        lane_schemas = tuple(schemas)
        lane_models = tuple(models)
        lane_hooks = tuple(hooks)
        if not (
            len(lane_schemas)
            == len(lane_models)
            == len(lane_hooks)
            == registry.n_replica
        ):
            raise IterationError("rollout lane operations are not aligned")
        transactions = tuple(
            RolloutTransaction(
                lane_schemas,
                seed=seed,
                iteration=iteration,
                topology=topology,
                owners=registry,
            )
            for _lane in range(registry.n_replica)
        )
        while has_work():
            pending: list[RolloutSubmission] = []
            while len(pending) < inflight_cap and has_work():
                scheduled = scheduler.select(
                    states(logical_time), allow_partial=True
                )
                if scheduled is None:
                    raise IterationError("scheduler lost published queue work")
                if logical_limit is not None and logical_time >= logical_limit:
                    raise IterationError("rollout exceeded its logical bound")
                for lane in scheduled.lanes:
                    local = ScheduledFamily(
                        input_type=scheduled.input_type,
                        lanes=(lane,),
                        lane_schemas=(lane_schemas[lane],),
                        rows=scheduled.rows,
                        forced_partial=scheduled.forced_partial,
                    )
                    commit = transactions[lane].execute(
                        queues[scheduled.input_type],
                        local,
                        lane_models[lane],
                        lane_hooks[lane],
                    )
                    pending.append(RolloutSubmission(local, commit))
                logical_time += 1
            accepted: list[AcceptedRollout] = []
            for submission in pending:
                produced: tuple[int, ...] | None = None
                for publication in submission.commit.topology:
                    failed, current = topology.accept_expansion(publication)
                    if failed:
                        raise failure(publication.lane)
                    if produced is not None:
                        raise IterationError(
                            "one rollout submission published two topology packets"
                        )
                    produced = current
                if produced is None:
                    raise IterationError(
                        "rollout commit omitted its topology expansion"
                    )
                accepted.append(
                    AcceptedRollout(
                        submission.scheduled,
                        submission.commit,
                        produced,
                    )
                )
            accept(tuple(accepted))
        return logical_time

    def seal_topology(
        self, publisher: AsyncTopologyPublisher
    ) -> SealedTopology:
        """Seal every lane through the sole assembler and layer planner."""

        plan = publisher.owners.plan
        lane_count = plan.manifest.n_replica
        sealed = tuple(
            publisher.seal_lane(lane) for lane in range(lane_count)
        )
        sampled_nodes = tuple(
            publisher.sampled_tree_size(lane) for lane in range(lane_count)
        )
        record_counts = tuple(
            publisher.record_count(lane) for lane in range(lane_count)
        )
        corrections = tuple(
            publisher.counterfactual_correction(
                lane, record_counts[lane]
            )
            for lane in range(lane_count)
        )
        publisher.assert_drained()
        publisher.release_expansion_storage()
        topologies = tuple(item[0] for item in sealed)
        bindings = tuple(item[1] for item in sealed)
        leaf_counts = tuple(item[2] for item in sealed)
        mode = plan.mode
        p_train = (
            plan.manifest.geometry.actor_count
            if mode.training_regime.value == "self_play"
            else 1
        )
        plan = build_layer_plan(
            mode,
            plan.partitions,
            topologies,
            bindings,
            p_train=p_train,
            manifest=plan.manifest,
        )
        return SealedTopology(
            topologies,
            bindings,
            leaf_counts,
            plan,
            sampled_nodes,
            record_counts,
            corrections,
        )

    @staticmethod
    def drain_completion(
        *,
        completion: CompletionFIFO,
        remaining: Sequence[int],
        hook: CompletionHook,
        wait_for: torch.cuda.Event,
        failure_message: str,
    ) -> None:
        """Drain every compatible lane through the sole completion loop."""

        outstanding = [int(value) for value in remaining]
        if any(value < 0 for value in outstanding):
            raise ValueError("completion depths must be nonnegative")
        while any(outstanding):
            active_lanes = tuple(
                lane
                for lane, value in enumerate(outstanding)
                if value > 0
            )
            rows = min(
                completion.completion_pack_cap,
                *(outstanding[lane] for lane in active_lanes),
            )
            success = completion.complete(
                active_lanes,
                rows,
                hook,
                wait_for=wait_for,
            )
            lane_index = torch.tensor(
                active_lanes,
                dtype=torch.int64,
                device=completion.device,
            )
            torch._assert_async(
                success.index_select(0, lane_index).all(),
                failure_message,
            )
            for lane in active_lanes:
                outstanding[lane] -= rows
        completion.release()

    @staticmethod
    def preflight_learners(
        *,
        generators: Sequence[torch.Generator],
        validate_lane: Callable[[int], None],
        clear_gradients: Callable[[], None],
        distributed: object | None,
        device: torch.device,
        error_type: Callable[[str], Exception],
    ) -> None:
        """Validate all lanes transactionally before any optimizer transition."""

        generator_states = tuple(
            generator.get_state() for generator in generators
        )
        preflight_error: Exception | None = None
        try:
            for lane in range(len(generators)):
                validate_lane(lane)
        except Exception as exc:  # noqa: BLE001 - transaction-wide abort
            preflight_error = exc
        finally:
            for generator, state in zip(
                generators, generator_states, strict=True
            ):
                generator.set_state(state)
            clear_gradients()
        if distributed is not None:
            local = torch.zeros(
                len(generators),
                dtype=torch.bool,
                device=device,
            )
            if preflight_error is not None:
                local.fill_(True)
            common = distributed.agree_failures(local)
            if bool(common.any()) and preflight_error is None:
                preflight_error = error_type(
                    "replica learner preflight failed on another rank"
                )
        if preflight_error is not None:
            raise error_type(
                "replica learner preflight failed before optimizer transition"
            ) from preflight_error

    @staticmethod
    def _failure_from_device(
        failure: DeviceFailure,
    ) -> RecoverableOverflow:
        return RecoverableOverflow(
            failure,
            capacity=failure.available_extent,
            observed_demand=failure.requested_extent,
        )

    @staticmethod
    def _finish_owners(
        owners: GeneratedOwners,
        *,
        entered_index: int,
        failure_lane: int | None = None,
    ) -> None:
        failure_lanes = (
            range(owners.plan.manifest.n_replica)
            if failure_lane is None
            else (failure_lane,)
        )
        for lane in failure_lanes:
            if owners.failure_state.snapshot(lane) is not None:
                owners.abort_failed_lane(lane)
        for index in range(entered_index, len(_PHASES)):
            phase = _PHASES[index]
            if index != entered_index:
                owners.enter_phase(phase)
            owners.release_after(phase)
        owners.close()

    def _run_attempt(
        self,
        owners: GeneratedOwners,
        *,
        iteration: int,
    ) -> object:
        attempt = self.hooks.build_attempt(owners)
        entered_index = 0
        owners.enter_phase(_PHASES[0])
        phase_open = True
        try:
            rollout = attempt.prepare_rollout(iteration=iteration)
            logical_time = self.schedule_rollout(
                scheduler=rollout.scheduler,
                states=rollout.states,
                has_work=rollout.has_work,
                queues=rollout.queues,
                registry=rollout.registry,
                schemas=rollout.schemas,
                seed=rollout.seed,
                iteration=iteration,
                models=rollout.models,
                hooks=rollout.hooks,
                accept=rollout.accept,
                topology=rollout.topology,
                failure=rollout.failure,
                inflight_cap=rollout.inflight_cap,
                logical_time=rollout.logical_time,
                logical_limit=rollout.logical_limit,
            )
            attempt.finish_rollout(
                logical_time=logical_time,
                topology=self.seal_topology(rollout.topology),
            )
            remote = self.rank_agreement.agree(None)
            if remote is not None:
                agreed = self._failure_from_device(remote.failure)
                agreed._rank_agreed = True  # type: ignore[attr-defined]
                raise agreed
            owners.release_after(ExecutionPhase.ROLLOUT)
            phase_open = False

            phase_operations = (
                (ExecutionPhase.QUEUE_DRAIN, attempt.finish_queue_drain),
                (
                    ExecutionPhase.COMPLETION,
                    lambda: self._execute_completion(attempt),
                ),
                (ExecutionPhase.ALIGN_AND_SEAL, attempt.align_and_seal),
                (ExecutionPhase.TARGETS, attempt.execute_targets),
                (
                    ExecutionPhase.LEARNER,
                    lambda: attempt.execute_learner(iteration=iteration),
                ),
                (ExecutionPhase.FINAL_BACKWARD, attempt.finish_backward),
            )
            result: object | None = None
            for index, (phase, operation) in enumerate(
                phase_operations, start=1
            ):
                entered_index = index
                owners.enter_phase(phase)
                phase_open = True
                value = operation()
                if phase is ExecutionPhase.FINAL_BACKWARD:
                    result = value
                owners.release_after(phase)
                phase_open = False
            owners.close()
            return result
        except BaseException as exc:
            failure = getattr(exc, "failure", None)
            self._finish_owners(
                owners,
                entered_index=entered_index + int(not phase_open),
                failure_lane=(
                    int(failure.lane)
                    if isinstance(failure, DeviceFailure)
                    else None
                ),
            )
            raise

    def _execute_completion(self, attempt: IterationAttempt) -> None:
        completion = attempt.prepare_completion()
        self.drain_completion(
            completion=completion.completion,
            remaining=completion.remaining,
            hook=completion.hook,
            wait_for=completion.wait_for,
            failure_message=completion.failure_message,
        )
        attempt.finish_completion()

    def _agree_failure(
        self, failure: RecoverableOverflow
    ) -> RecoverableOverflow:
        if bool(getattr(failure, "_rank_agreed", False)):
            return failure
        agreed = self.rank_agreement.agree(failure)
        if agreed is None:
            raise IterationError("rank agreement lost a recoverable failure")
        return agreed

    def _recover(
        self,
        *,
        iteration: int,
        failure: RecoverableOverflow,
    ) -> None:
        lane_index = int(failure.failure.lane)
        if not 0 <= lane_index < len(self.config.lanes):
            raise IterationError("recoverable failure names an invalid lane")
        lane = self.config.lanes[lane_index]
        state = lane.run_state
        replacement = AttemptAddress(
            state.retry_epoch + 1,
            state.attempt_logical_base + state.root_count,
        )
        if lane.mode.rollout_mode.is_branched:
            controller = lane.branch_controller
            assert controller is not None
            if controller.at_minimum:
                raise TrainingError(
                    "minimum branched control overflowed fixed capacity"
                ) from failure
            old_control = controller.state_dict()
            controller.backoff()
            new_control = controller.state_dict()
            record = BranchedRecoveryRecord(
                severity="critical",
                iteration=iteration,
                replica=lane_index,
                failed_phase=failure.failure.phase,
                failed_address=AttemptAddress(
                    state.retry_epoch,
                    failure.failure.logical_address,
                ).as_text(),
                replacement_address=replacement.as_text(),
                old_control=old_control,
                new_control=new_control,
                capacity=failure.capacity,
                observed_demand=failure.observed_demand,
            )
            state.branch_controller_state = new_control
        else:
            if state.root_count == lane.minimum_root_count:
                raise TrainingError(
                    "minimum sequential root_count overflowed fixed capacity"
                ) from failure
            old_root_count = state.root_count
            replacement_count = max(
                lane.minimum_root_count, old_root_count // 2
            )
            if replacement_count == old_root_count:
                replacement_count -= 1
            state.root_count = replacement_count
            record = OverflowRecoveryRecord(
                severity="critical",
                iteration=iteration,
                replica=lane_index,
                failed_phase=failure.failure.phase,
                failed_address=AttemptAddress(
                    state.retry_epoch,
                    failure.failure.logical_address,
                ).as_text(),
                replacement_address=replacement.as_text(),
                old_root_count=old_root_count,
                new_root_count=replacement_count,
                capacity=failure.capacity,
                observed_demand=failure.observed_demand,
            )
        state.recovered_overflow = True
        state.retry_epoch = replacement.retry_epoch
        state.attempt_logical_base = replacement.logical_base
        state.overflow_recoveries.append(record.as_dict())
        tag = (
            "RECOVERED_BRANCHED_OVERFLOW"
            if lane.mode.rollout_mode.is_branched
            else "RECOVERED_SEQUENTIAL_OVERFLOW"
        )
        payload = json.dumps(record.as_dict(), sort_keys=True)
        logging.getLogger(__name__).critical("%s %s", tag, payload)
        warnings.warn(
            f"{tag} {payload}",
            RuntimeWarning,
            stacklevel=2,
        )

    def run(self, *, iteration: int) -> object:
        """Retry unpublished attempts and return one committed iteration."""

        while True:
            owners = self.owner_factory()
            if owners.plan != self.config.plan:
                owners.failure_state.release()
                raise IterationError(
                    "owner factory changed the immutable storage plan"
                )
            try:
                return self._run_attempt(owners, iteration=iteration)
            except RecoverableOverflow as caught:
                failure = self._agree_failure(caught)
                self._recover(iteration=iteration, failure=failure)


__all__ = [
    "AcceptedRollout",
    "CompletionLoop",
    "IterationAttempt",
    "IterationConfig",
    "IterationEngine",
    "IterationError",
    "IterationHooks",
    "LaneContext",
    "LearnerPreflightOptimizer",
    "RolloutLoop",
    "RolloutSubmission",
    "SealedTopology",
]
