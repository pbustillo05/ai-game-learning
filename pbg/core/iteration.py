"""One production iteration lifecycle for every game and execution dimension."""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor

from pbg.core.branching import (
    BranchController,
    BranchedRecoveryRecord,
    CostSafeguardController,
    RankCommonFailureAgreement,
    decision_log_path_mass,
)
from pbg.core.capacity import ExecutionPhase, GeneratedOwners, StoragePlan
from pbg.core.contracts import ModeSchema, validate_replica_group
from pbg.core.learner import (
    AlignedSequentialStore,
    BranchedBatchSampler,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    ModelOperation,
    PolicyOperation,
    RaggedBranchedStore,
    ReferenceOutputCache,
    SealedBranchedInputs,
    SealedSequentialInputs,
    SequentialBatchSampler,
    allocate_statistical_counts,
    build_reference_output_cache,
)
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
from pbg.core.storage import (
    DeviceDestinationRequest,
    DeviceFailure,
    MonotoneSlab,
    PublicationTransaction,
)
from pbg.core.targets import LayeredTargetExecutor, TargetAlgebra
from pbg.core.topology import (
    LayerPlan,
    Topology,
    TopologyBindings,
    TopologyPacketStager,
    build_layer_plan,
)
from pbg.core.training import (
    AttemptAddress,
    IterationResult,
    LearnerIteration,
    LearnerTransaction,
    OverflowRecoveryRecord,
    PreparedLearner,
    RecoverableOverflow,
    TrainingError,
)


class IterationError(RuntimeError):
    """The common iteration lifecycle could not complete transactionally."""


@dataclass(frozen=True)
class LaneContext:
    """Immutable execution, learner, controller, and recovery lane ownership."""

    mode: ModeSchema
    run_state: RunState | None
    generator: torch.Generator
    optimizer: object
    model_operation: ModelOperation
    loss: LossConfig
    model: object | None = None
    reference_operation: ModelOperation | None = None
    branch_controller: BranchController | None = None
    cost_safeguard: CostSafeguardController | None = None
    allocation_inputs: tuple[Tensor, Tensor, Tensor, int] | None = None
    minimum_root_count: int = 1
    minimum_tree_size: int = 1

    def __post_init__(self) -> None:
        if self.minimum_root_count <= 0 or self.minimum_tree_size <= 0:
            raise ValueError("minimum root and tree sizes must be positive")
        if self.mode.rollout_mode.is_branched != (
            self.branch_controller is not None
        ):
            raise ValueError(
                "branched lanes require exactly one branch controller"
            )
        if self.cost_safeguard is not None and self.branch_controller is None:
            raise ValueError("cost safeguards require a branched controller")


@dataclass(frozen=True)
class IterationConfig:
    """Closed generic configuration for one multi-lane iteration."""

    plan: StoragePlan
    lanes: tuple[LaneContext, ...]
    schedule: LearnerSchedule
    optimizer_owner: object
    lambda_: float
    precision: str = "fp32"
    cache_reference_outputs: bool = False
    reproducibility: str = "strict"
    distributed: object | None = None
    benchmark_telemetry: bool = False
    learner_family_rows: Mapping[str, int] | None = None

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
            and lane.run_state is not None
            and lane.run_state.root_count != self.plan.manifest.root_count
            for lane in self.lanes
        ):
            raise ValueError("branched root count differs from the storage plan")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("iteration precision is invalid")
        if self.reproducibility not in ("strict", "seeded", "throughput"):
            raise ValueError("iteration reproducibility is invalid")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("iteration lambda must lie in [0,1]")
        if type(self.cache_reference_outputs) is not bool:
            raise TypeError("cache_reference_outputs must be bool")
        if type(self.benchmark_telemetry) is not bool:
            raise TypeError("benchmark_telemetry must be bool")


@dataclass(frozen=True)
class RootDestination:
    """Game-declared input partition that receives every root row."""

    input_type: str

    def __post_init__(self) -> None:
        if not self.input_type or not self.input_type.isidentifier():
            raise ValueError("root destination must name one input partition")


@dataclass(frozen=True)
class RootRequest:
    """Lane-local logical root request presented to a game operation."""

    plan: StoragePlan
    destination: RootDestination
    lane: int
    rows: int
    iteration: int
    retry_epoch: int
    logical_base: int
    safeguard: CostSafeguardController | None


@dataclass(frozen=True)
class RootPayload:
    """Game-computed root columns; core reserves and publishes them."""

    values: Mapping[str, Tensor]
    frontier_cost: Tensor | None = None


@dataclass(frozen=True)
class RolloutOperationRequest:
    """Core-owned rollout objects supplied to semantic game hooks."""

    owners: GeneratedOwners
    registry: RolloutOwnerRegistry
    queues: Mapping[str, SemanticFamilyFIFO]
    stagers: tuple[TopologyPacketStager, ...]
    iteration: int
    roots: tuple[int, ...]
    logical_bases: tuple[int, ...]
    controllers: tuple[BranchController | None, ...]
    safeguards: tuple[CostSafeguardController | None, ...]


@dataclass(frozen=True)
class GameRollout:
    """Opaque game operations plugged into the common rollout transaction."""

    hooks: tuple[RolloutHooks, ...]
    models: tuple[
        Callable[[object, Tensor], Mapping[str, Tensor]], ...
    ]
    max_prefix: tuple[int | None, ...]
    begin_refill: Callable[[int, int], None] | None = None
    routing_traffic: Callable[[int], int] = lambda _lane: 0


@dataclass(frozen=True)
class CompletionContext:
    """Core completion storage exposed to the game evaluator binding."""

    owners: GeneratedOwners
    algebra: TargetAlgebra


@dataclass(frozen=True)
class GameCompletion:
    """Game evaluator operation and its pure post-call observations."""

    hook: CompletionHook
    observations: Callable[[], Mapping[str, object]] = (
        lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class CompletedLane:
    """Storage-free context for pure game observability formatting."""

    lane: int
    topology: Topology
    bindings: TopologyBindings
    partitions: object
    model_timing: Mapping[str, object]


class GameOperations(Protocol):
    """Small semantic boundary consumed by the generic iteration engine."""

    seed: int
    error_type: type[Exception]

    def root_destination(self, plan: StoragePlan) -> RootDestination: ...

    def root_payload(self, request: RootRequest) -> RootPayload: ...

    def rollout(self, request: RolloutOperationRequest) -> GameRollout: ...

    def rollout_failure(
        self, lane: int, failure: DeviceFailure | None
    ) -> Exception: ...

    def completion(self, context: CompletionContext) -> GameCompletion: ...

    def target_algebra(self, plan: StoragePlan) -> TargetAlgebra: ...

    def learner_policy(self, lane: int) -> PolicyOperation: ...

    def model_costs(self, lane: int) -> Mapping[str, int] | None: ...

    def logical_rollout_limit(self, plan: StoragePlan) -> int: ...

    def observability(self, lane: CompletedLane) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class AcceptedRollout:
    """Commit-visible queue counts accepted by the core topology authority."""

    scheduled: ScheduledFamily
    commit: RolloutCommit
    produced: tuple[int, ...]


@dataclass(frozen=True)
class RolloutSubmission:
    """One lane-local transaction submitted to the common scheduler."""

    scheduled: ScheduledFamily
    commit: RolloutCommit


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
class IterationTelemetry:
    """Generic matched-work measurements around the common phases."""

    rollout_seconds: float
    routing_traffic_bytes: int
    host_synchronizations: int
    child_reservation_commit_seconds: float
    target_propagation_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    model_inference_seconds: float
    phase_measurements: dict[str, dict[str, float | int]] = field(default_factory=dict)


@dataclass(frozen=True)
class LaneIterationResult:
    """Generic committed result for one lane."""

    learner: IterationResult
    player_nodes: int
    leaves: int
    target_layers: int
    executed_edges: int
    root_executed_degree: tuple[int, ...]
    root_target_degree: tuple[int, ...]
    duplicate_child_labels: int
    sampled_tree_nodes: int
    posterior_effective_hands: float | None = None
    telemetry: IterationTelemetry | None = None
    observability: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class IterationBatchResult:
    """Lane results from one shared-owner transaction."""

    results: tuple[LaneIterationResult | None, ...]
    failures: Mapping[int, str]
    versions: tuple[int, ...]


@dataclass
class _ModelTiming:
    events: dict[
        tuple[int, str],
        list[tuple[torch.cuda.Event, torch.cuda.Event, int]],
    ] = field(default_factory=dict)

    def wrap(
        self,
        cohort: tuple[int, ...],
        input_type: str,
        rows: int,
        operation: Callable[[object, Tensor], Mapping[str, Tensor]],
    ) -> Callable[[object, Tensor], Mapping[str, Tensor]]:
        def timed(value: object, lanes: Tensor) -> Mapping[str, Tensor]:
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            result = operation(value, lanes)
            finished.record()
            for lane in cohort:
                self.events.setdefault((lane, input_type), []).append(
                    (started, finished, rows)
                )
            return result

        return timed

    def lane(self, lane: int) -> Mapping[str, object]:
        result: dict[str, object] = {}
        for (event_lane, family), values in self.events.items():
            if event_lane != lane:
                continue
            if values:
                values[-1][1].synchronize()
            result[family] = {
                "batches": len(values),
                "rows": sum(value[2] for value in values),
                "seconds": sum(
                    value[0].elapsed_time(value[1]) for value in values
                )
                / 1000.0,
            }
        return MappingProxyType(result)


@dataclass
class _IterationState:
    input_names: tuple[str, ...]
    topology: SealedTopology
    last_event: torch.cuda.Event
    roots: tuple[int, ...]
    counterfactual_sizes: tuple[float, ...]
    routing_traffic: tuple[int, ...]
    model_timing: _ModelTiming
    benchmark_started: float
    rollout_seconds: float = 0.0
    target_seconds: float = 0.0
    phase_measurements: dict[str, dict[str, float | int]] = field(default_factory=dict)
    host_synchronizations: int = 0
    completion_observations: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    sealed: Mapping[
        tuple[int, str], SealedSequentialInputs | SealedBranchedInputs
    ] = field(default_factory=dict)
    learner_results: tuple[IterationResult | None, ...] = ()


_PHASES = tuple(ExecutionPhase)


class IterationEngine:
    """Own root publication through final close for every supported mode."""

    def __init__(
        self,
        *,
        config: IterationConfig,
        owner_factory: Callable[[], GeneratedOwners],
        operations: GameOperations,
        rank_agreement: RankCommonFailureAgreement | None = None,
    ) -> None:
        self.config = config
        self.owner_factory = owner_factory
        self.operations = operations
        self.rank_agreement = (
            RankCommonFailureAgreement.from_runtime()
            if rank_agreement is None
            else rank_agreement
        )

    @staticmethod
    def _defaults(slab: MonotoneSlab, rows: int) -> dict[str, Tensor]:
        return {
            field.name: torch.full(
                (slab.n_replica, rows, *field.shape),
                field.fill,
                dtype=field.dtype,
                device=slab.device,
            )
            for field in slab.schema.fields
        }

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
            Callable[[object, Tensor], Mapping[str, Tensor]]
        ],
        hooks: Sequence[RolloutHooks],
        accept: Callable[[tuple[AcceptedRollout, ...]], None],
        topology: AsyncTopologyPublisher,
        failure: Callable[[int], Exception],
        inflight_cap: int,
        timing: _ModelTiming,
        logical_time: int = 0,
        logical_limit: int | None = None,
    ) -> int:
        """Run the sole stable-prefix scheduler and rollout transaction loop."""

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
                share_cohort = (
                    all(
                        lane_hooks[lane] is lane_hooks[scheduled.lanes[0]]
                        for lane in scheduled.lanes
                    )
                    and all(
                        lane_models[lane] is lane_models[scheduled.lanes[0]]
                        for lane in scheduled.lanes
                    )
                )
                cohorts = (
                    (scheduled,)
                    if share_cohort
                    else tuple(
                        ScheduledFamily(
                            input_type=scheduled.input_type,
                            lanes=(lane,),
                            lane_schemas=(lane_schemas[lane],),
                            rows=scheduled.rows,
                            forced_partial=scheduled.forced_partial,
                        )
                        for lane in scheduled.lanes
                    )
                )
                for cohort in cohorts:
                    first = cohort.lanes[0]
                    model = timing.wrap(
                        cohort.lanes,
                        scheduled.input_type,
                        scheduled.rows,
                        lane_models[first],
                    )
                    commit = transactions[first].execute(
                        queues[scheduled.input_type],
                        cohort,
                        model,
                        lane_hooks[first],
                    )
                    pending.append(RolloutSubmission(cohort, commit))
                logical_time += 1
            accepted: list[AcceptedRollout] = []
            for submission in pending:
                produced: dict[int, tuple[int, ...]] = {}
                for publication in submission.commit.topology:
                    failed, current = topology.accept(publication)
                    if failed:
                        raise failure(publication.lane)
                    if publication.lane in produced:
                        raise IterationError(
                            "one rollout lane published two topology packets"
                        )
                    produced[publication.lane] = current
                if set(produced) != set(submission.scheduled.lanes):
                    raise IterationError(
                        "rollout commit topology lanes differ from its cohort"
                    )
                for lane in submission.scheduled.lanes:
                    accepted.append(
                        AcceptedRollout(
                            ScheduledFamily(
                                input_type=submission.scheduled.input_type,
                                lanes=(lane,),
                                lane_schemas=(lane_schemas[lane],),
                                rows=submission.scheduled.rows,
                                forced_partial=(
                                    submission.scheduled.forced_partial
                                ),
                            ),
                            submission.commit,
                            produced[lane],
                        )
                    )
            accept(tuple(accepted))
        return logical_time

    @staticmethod
    def seal_topology(
        publisher: AsyncTopologyPublisher,
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
        publisher.release_storage()
        topologies = tuple(item[0] for item in sealed)
        bindings = tuple(item[1] for item in sealed)
        leaf_counts = tuple(item[2] for item in sealed)
        p_train = (
            plan.manifest.geometry.actor_count
            if plan.mode.training_regime.value == "self_play"
            else 1
        )
        layer_plan = build_layer_plan(
            plan.mode,
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
            layer_plan,
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
                "game completion failed",
            )
            for lane in active_lanes:
                outstanding[lane] -= rows
        completion.release()

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
    def _recoverable_overflow(
        owners: GeneratedOwners, lane: int
    ) -> RecoverableOverflow:
        failure = owners.failure_state.snapshot(lane)
        if failure is None or failure.failure_code != "capacity_overflow":
            raise IterationError(
                "rollout failed without a recoverable capacity record"
            )
        candidates = [
            slab
            for slab in owners.live_slabs
            if slab.capacity - slab.tail(lane) == failure.available_extent
            and slab.schema.name.split(".", 1)[0]
            in {"FamilyWork", "SigmaHead", "ActionHead", "Leaf"}
            and (
                slab.schema.name == "Leaf"
                or slab.schema.name.endswith(f".{failure.input_type}")
            )
        ]
        if not candidates:
            candidates = [
                slab
                for slab in owners.live_slabs
                if slab.capacity - slab.tail(lane)
                == failure.available_extent
            ]
        if candidates:
            slab = min(candidates, key=lambda value: value.capacity)
            capacity = slab.capacity
            observed = slab.tail(lane) + failure.requested_extent
        else:
            capacity = failure.available_extent
            observed = failure.requested_extent
        return RecoverableOverflow(
            failure,
            capacity=capacity,
            observed_demand=observed,
        )

    @staticmethod
    def _finish_owners(
        owners: GeneratedOwners,
        *,
        entered_index: int,
        failure_lane: int | None = None,
    ) -> None:
        if (
            isinstance(owners, GeneratedOwners)
            and not owners.layer_work_is_bound
        ):
            owners.bind_empty_layer_work_for_abort()
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

    def _root_counts(self) -> tuple[int, ...]:
        maximum = self.config.plan.manifest.root_count
        result = tuple(
            maximum
            if lane.run_state is None
            else lane.run_state.root_count
            for lane in self.config.lanes
        )
        if any(not 1 <= value <= maximum for value in result):
            raise IterationError(
                "runtime root count exceeds its immutable admitted maximum"
            )
        return result

    def _publish_roots(
        self,
        owners: GeneratedOwners,
        queue: SemanticFamilyFIFO,
        *,
        destination: RootDestination,
        iteration: int,
        roots: tuple[int, ...],
        logical_bases: tuple[int, ...],
    ) -> None:
        maximum = max(roots)
        values = self._defaults(queue.family, maximum)
        frontier: list[Tensor | None] = []
        for lane, rows in enumerate(roots):
            state = self.config.lanes[lane].run_state
            payload = self.operations.root_payload(
                RootRequest(
                    owners.plan,
                    destination,
                    lane,
                    rows,
                    iteration,
                    0 if state is None else state.retry_epoch,
                    logical_bases[lane],
                    self.config.lanes[lane].cost_safeguard,
                )
            )
            if set(payload.values) - set(values):
                raise IterationError("root payload contains undeclared columns")
            for name, source in payload.values.items():
                column_destination = values[name][lane, :rows]
                if source.shape != column_destination.shape:
                    raise IterationError(
                        f"root column {name!r} has the wrong shape"
                    )
                column_destination.copy_(source)
            if payload.frontier_cost is not None and payload.frontier_cost.shape != (
                rows,
            ):
                raise IterationError("root frontier cost must be lane-local [rows]")
            frontier.append(payload.frontier_cost)
        device = owners.plan.device
        counts = torch.tensor(roots, dtype=torch.int64, device=device)
        transaction = PublicationTransaction.begin_device(
            queue.family.destination_requests(
                counts,
                values,
                queue.input_type,
                torch.tensor(
                    logical_bases, dtype=torch.int64, device=device
                ),
            ),
            phase="root_commit",
            active=torch.ones(
                len(roots), dtype=torch.bool, device=device
            ),
        )
        transaction.fill_all()
        success = transaction.publish()
        if not bool(success.all()):
            failed = int(torch.nonzero(~success).flatten()[0])
            raise self._recoverable_overflow(owners, failed)
        queue.publish_family_tail(
            torch.ones(len(roots), dtype=torch.bool, device=device)
        )
        for lane, cost in enumerate(frontier):
            if cost is not None:
                queue.frontier_cost[lane].copy_(cost.sum())

    def _publish_refill_root(
        self,
        queue: SemanticFamilyFIFO,
        *,
        lane: int,
        original_row: int,
        logical_root: int,
    ) -> None:
        values = {
            name: column[:, original_row : original_row + 1].clone()
            for name, column in queue.family.columns.items()
        }
        count = torch.zeros(
            queue.family.n_replica,
            dtype=torch.int64,
            device=queue.family.device,
        )
        count[lane] = 1
        address = torch.full_like(count, torch.iinfo(torch.int64).max)
        address[lane] = logical_root
        active = torch.arange(
            count.numel(), device=queue.family.device
        ).eq(lane)
        transaction = PublicationTransaction.begin_device(
            queue.family.destination_requests(
                count,
                values,
                queue.input_type,
                address,
            ),
            phase="root_refill",
            active=active,
        )
        transaction.fill_all()
        success = transaction.publish()
        torch._assert_async(
            success[lane],
            "root refill exceeded fixed capacity",
        )
        queue.publish_family_tail(active)
        if "frontier_cost_contribution" in values:
            queue.frontier_cost[lane].add_(
                values["frontier_cost_contribution"][lane].sum()
            )

    def _execute_rollout(
        self,
        owners: GeneratedOwners,
        *,
        iteration: int,
    ) -> _IterationState:
        benchmark_started = 0.0
        host_synchronizations = 0
        if self.config.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            torch.cuda.reset_peak_memory_stats(owners.plan.device)
            benchmark_started = time.perf_counter()
            host_synchronizations = 1
        count = owners.plan.manifest.n_replica
        registry = RolloutOwnerRegistry.from_generated(owners)
        input_names = tuple(
            partition.name for partition in owners.plan.partitions.inputs
        )
        root_destination = self.operations.root_destination(owners.plan)
        if not isinstance(root_destination, RootDestination):
            raise IterationError(
                "game root destination has the wrong contract"
            )
        root_input_type = root_destination.input_type
        if root_input_type not in input_names:
            raise IterationError(
                "game root destination is not a declared input partition"
            )
        queues = {
            name: SemanticFamilyFIFO(
                name,
                owners.plan.partitions,
                owners.slab("FamilyWork", name),
                owners.slab("FamilyRetained", name),
                owners.slab("QueueMetadata", name),
            )
            for name in input_names
        }
        for queue in queues.values():
            queue.initialize()
        roots = self._root_counts()
        logical_bases = tuple(
            0 if lane.run_state is None else lane.run_state.attempt_logical_base
            for lane in self.config.lanes
        )
        self._publish_roots(
            owners,
            queues[root_input_type],
            destination=root_destination,
            iteration=iteration,
            roots=roots,
            logical_bases=logical_bases,
        )
        output_names = tuple(
            output.name for output in owners.plan.partitions.outputs
        )
        input_to_output = tuple(
            output_names.index(
                owners.plan.partitions.output_for(name).name
            )
            for name in input_names
        )
        publisher = AsyncTopologyPublisher(
            registry,
            p_train=(
                owners.plan.manifest.geometry.actor_count
                if owners.plan.mode.training_regime.value == "self_play"
                else 1
            ),
            input_names=input_names,
            input_to_output=input_to_output,
        )
        rollout = self.operations.rollout(
            RolloutOperationRequest(
                owners,
                registry,
                MappingProxyType(queues),
                publisher.stagers,
                iteration,
                roots,
                logical_bases,
                tuple(lane.branch_controller for lane in self.config.lanes),
                tuple(lane.cost_safeguard for lane in self.config.lanes),
            )
        )
        if not (
            len(rollout.hooks)
            == len(rollout.models)
            == len(rollout.max_prefix)
            == count
        ):
            raise IterationError("game rollout lanes are not aligned")
        depths = [
            {
                name: (roots[lane] if name == root_input_type else 0)
                for name in input_names
            }
            for lane in range(count)
        ]
        last_event: torch.cuda.Event | None = None
        pre_refill_nodes: list[int | None] = [None] * count
        pre_refill_correction: list[float | None] = [None] * count

        def queue_states(logical_time: int) -> tuple[FamilyQueueState, ...]:
            return tuple(
                FamilyQueueState(
                    lane,
                    name,
                    index,
                    0,
                    depths[lane][name],
                    logical_time,
                    0,
                    self.config.lanes[lane].mode,
                    max_prefix=rollout.max_prefix[lane],
                )
                for lane in range(count)
                for index, name in enumerate(input_names)
            )

        def failure(lane: int) -> Exception:
            record = owners.failure_state.snapshot(lane)
            if record is not None and record.failure_code == "capacity_overflow":
                return self._recoverable_overflow(owners, lane)
            return self.operations.rollout_failure(lane, record)

        def accept(pending: tuple[AcceptedRollout, ...]) -> None:
            nonlocal last_event
            for accepted in pending:
                lane = int(accepted.scheduled.lanes[0])
                depths[lane][accepted.scheduled.input_type] -= (
                    accepted.scheduled.rows
                )
                for name, produced in zip(
                    input_names, accepted.produced, strict=True
                ):
                    depths[lane][name] += produced
                last_event = accepted.commit.event

        def has_work() -> bool:
            if any(any(value.values()) for value in depths):
                return True
            if not owners.plan.mode.rollout_mode.is_branched:
                return False
            if rollout.begin_refill is None:
                raise IterationError("branched rollout omitted refill operation")
            for lane, stager in enumerate(publisher.stagers):
                logical_base = logical_bases[lane]
                deficient = next(
                    (
                        root
                        for root in range(
                            logical_base, logical_base + roots[lane]
                        )
                        if stager.tree_size(root)
                        < self.config.lanes[lane].minimum_tree_size
                    ),
                    None,
                )
                if deficient is None:
                    continue
                if pre_refill_nodes[lane] is None:
                    pre_refill_nodes[lane] = stager.sampled_tree_size()
                    pre_refill_correction[lane] = (
                        stager.counterfactual_correction(stager.record_count)
                    )
                rollout.begin_refill(
                    lane, stager.root_degree(deficient) + 1
                )
                self._publish_refill_root(
                    queues[root_input_type],
                    lane=lane,
                    original_row=deficient - logical_base,
                    logical_root=deficient,
                )
                depths[lane][root_input_type] = 1
                return True
            return False

        timing = _ModelTiming()
        self.schedule_rollout(
            scheduler=FamilyScheduler(
                input_names, owners.plan.manifest.family_pack_cap
            ),
            states=queue_states,
            has_work=has_work,
            queues=queues,
            registry=registry,
            schemas=tuple(lane.mode for lane in self.config.lanes),
            seed=self.operations.seed,
            iteration=iteration,
            models=rollout.models,
            hooks=rollout.hooks,
            accept=accept,
            topology=publisher,
            failure=failure,
            inflight_cap=1,
            timing=timing,
            logical_limit=self.operations.logical_rollout_limit(owners.plan),
        )
        if last_event is None:
            raise IterationError("rollout produced no commit event")
        topology = self.seal_topology(publisher)
        owners.bind_layer_work(
            topology.plan.output_names,
            topology.plan.arena_rows,
        )
        counterfactual: list[float] = []
        for lane, sampled in enumerate(topology.sampled_nodes):
            if pre_refill_nodes[lane] is None:
                pre_refill_nodes[lane] = sampled
                pre_refill_correction[lane] = (
                    topology.counterfactual_corrections[lane]
                )
            correction = pre_refill_correction[lane]
            assert correction is not None
            counterfactual.append(
                (float(pre_refill_nodes[lane]) + correction) / roots[lane]
            )
        routing = tuple(
            rollout.routing_traffic(lane) for lane in range(count)
        )
        rollout_seconds = 0.0
        if self.config.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            host_synchronizations += 1
            rollout_seconds = time.perf_counter() - benchmark_started
        # These objects own rollout hooks, queues, registry references, and
        # game state.  Completion is serialized after their exact last use.
        del registry
        return _IterationState(
            input_names,
            topology,
            last_event,
            roots,
            tuple(counterfactual),
            routing,
            timing,
            benchmark_started,
            rollout_seconds=rollout_seconds,
            host_synchronizations=host_synchronizations,
        )

    def _execute_completion(
        self,
        owners: GeneratedOwners,
        state: _IterationState,
    ) -> None:
        completion = CompletionFIFO(
            owners.slab("Leaf"),
            RolloutOwnerRegistry.from_generated(owners),
            head=torch.zeros(
                owners.plan.manifest.n_replica,
                dtype=torch.int64,
                device=owners.plan.device,
            ),
            observed_epoch=torch.zeros(
                owners.plan.manifest.n_replica,
                dtype=torch.int64,
                device=owners.plan.device,
            ),
        )
        algebra = self.operations.target_algebra(owners.plan)
        game_completion = self.operations.completion(
            CompletionContext(owners, algebra)
        )
        self.drain_completion(
            completion=completion,
            remaining=state.topology.leaf_counts,
            hook=game_completion.hook,
            wait_for=state.last_event,
        )
        state.completion_observations = MappingProxyType(
            dict(game_completion.observations())
        )
        del game_completion
        plan = state.topology.plan
        for output, name in enumerate(plan.output_names):
            work = owners.slab("LayerWork", name)
            for lane in range(owners.plan.manifest.n_replica):
                missing = plan.arena_rows[lane][output] - work.tail(lane)
                if missing < 0:
                    raise IterationError(
                        "completion exceeded the target arena plan"
                    )
                if not missing:
                    continue
                values = self._defaults(work, missing)
                counts = torch.zeros(
                    owners.plan.manifest.n_replica,
                    dtype=torch.int64,
                    device=owners.plan.device,
                )
                counts[lane] = missing
                address = torch.full_like(
                    counts, torch.iinfo(torch.int64).max
                )
                address[lane] = state.topology.leaf_counts[lane]
                active = torch.arange(
                    counts.numel(), device=owners.plan.device
                ).eq(lane)
                padding = PublicationTransaction.begin_device(
                    (
                        DeviceDestinationRequest(
                            work, counts, values, name, address
                        ),
                    ),
                    phase="target_arena_padding",
                    active=active,
                )
                padding.fill_all()
                torch._assert_async(
                    padding.publish()[lane],
                    "target arena padding failed",
                )
        if self.config.distributed is not None:
            self.config.distributed.barrier()

    def _seal_learner_inputs(
        self,
        owners: GeneratedOwners,
        state: _IterationState,
    ) -> None:
        sealed: dict[
            tuple[int, str], SealedSequentialInputs | SealedBranchedInputs
        ] = {}
        if not owners.plan.mode.rollout_mode.is_branched:
            sealed.update(
                {
                    (lane, name): SealedSequentialInputs.seal(
                        owners.slab("FamilyWork", name),
                        owners.slab("FamilyRetained", name),
                        (
                            owners.slab("LearnerRetained", name)
                            if owners.plan.mode.training_regime.value
                            == "best_response"
                            else None
                        ),
                        input_type=name,
                        lane=lane,
                    )
                    for lane in range(owners.plan.manifest.n_replica)
                    for name in state.input_names
                }
            )
            state.sealed = sealed
            return
        for lane, (topology, bindings) in enumerate(
            zip(
                state.topology.topologies,
                state.topology.bindings,
                strict=True,
            )
        ):
            log_mass = decision_log_path_mass(
                topology.parent,
                topology.executed_degree,
                device=owners.plan.device,
            )
            for input_index, name in enumerate(state.input_names):
                selected = (
                    (
                        bindings.input_type.eq(input_index)
                        & topology.owner.ge(0)
                    )
                    .nonzero()
                    .flatten()
                )
                selected = selected.index_select(
                    0,
                    torch.argsort(
                        topology.sigma_head_row[selected],
                        stable=True,
                    ),
                )
                device_selected = selected.to(
                    device=owners.plan.device,
                    dtype=torch.int64,
                )
                sealed[(lane, name)] = SealedBranchedInputs.seal(
                    owners.slab("FamilyRetained", name),
                    owners.slab("LearnerIndex", name),
                    (
                        owners.slab("LearnerRetained", name)
                        if self.config.lanes[lane].mode.training_regime.value
                        == "best_response"
                        else None
                    ),
                    input_type=name,
                    family_row=topology.familywork_row[selected].to(
                        owners.plan.device
                    ),
                    action_begin=topology.action_begin[selected].to(
                        owners.plan.device
                    ),
                    action_count=topology.target_degree[selected].to(
                        owners.plan.device
                    ),
                    target_degree=topology.target_degree[selected].to(
                        owners.plan.device
                    ),
                    log_sampling_weight=log_mass.index_select(
                        0, device_selected
                    ),
                    lane=lane,
                )
        state.sealed = sealed

    def _execute_targets(
        self,
        owners: GeneratedOwners,
        state: _IterationState,
    ) -> None:
        started = time.perf_counter() if self.config.benchmark_telemetry else 0.0
        result = LayeredTargetExecutor(
            owners.plan.mode,
            self.operations.target_algebra(owners.plan),
            lambda_=self.config.lambda_,
        ).execute(owners, state.topology.plan)
        if result.failed_lanes:
            raise IterationError("layered targets failed")
        for lane, context in enumerate(self.config.lanes):
            safeguard = context.cost_safeguard
            if safeguard is None:
                continue
            predicted_cost: list[Tensor] = []
            target_cost: list[Tensor] = []
            for name in state.input_names:
                head = owners.slab("SigmaHead", name)
                target = owners.slab("SigmaTarget", name)
                rows = head.tail(lane)
                if rows:
                    predicted_cost.append(
                        head.columns["old_cost"][lane, :rows]
                    )
                    target_cost.append(
                        target.columns["cost_target"][lane, :rows]
                    )
            if predicted_cost:
                predicted = torch.cat(predicted_cost).float().clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                target = torch.cat(target_cost).float()
                ratio = target / predicted
                deviance = ratio - ratio.log() - 1
                safeguard.observe(
                    float(deviance.mean()),
                    float(torch.quantile(ratio, 0.95)),
                    int(ratio.numel()),
                )
        if self.config.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            state.host_synchronizations += 1
            state.target_seconds = time.perf_counter() - started

    def _reference_caches(
        self,
        owners: GeneratedOwners,
        lane: int,
        stores: Mapping[str, AlignedSequentialStore | RaggedBranchedStore],
        *,
        sampler: SequentialBatchSampler | BranchedBatchSampler,
    ) -> dict[str, ReferenceOutputCache] | None:
        reference = self.config.lanes[lane].reference_operation
        if not self.config.cache_reference_outputs or reference is None:
            return None
        admitted: dict[str, ReferenceOutputCache] = {}
        resident = 0
        total_budget = owners.plan.layout.reference_cache_bytes
        share, remainder = divmod(total_budget, len(self.config.lanes))
        budget = share + int(lane < remainder)
        if budget == 0:
            return None
        counts = sampler.statistical_counts
        total = sampler.statistical_presentations
        population = sum(store.rows for store in stores.values())
        for name, store in stores.items():
            columns = (
                store.sealed_inputs.aligned_payload_columns
                if isinstance(store, RaggedBranchedStore)
                else store.sealed_inputs.columns
            )
            # Estimate row reuse over the entire learning schedule, rather
            # than the number of chunks in one traversal of fresh rollout rows.
            # Counts are metadata; estimating them must not consume sampler RNG.
            presentations = (total * store.rows / population if counts is None
                             else total * counts[name] / sum(counts.values()))
            passes = presentations / store.rows
            candidate = build_reference_output_cache(
                reference,
                input_type=name,
                model_input=columns["model_input"],
                sampled_private_state=columns.get(
                    "sampled_private_state", columns["owner"][:, None]
                ),
                owner=columns["owner"],
                pack_cap=min(owners.plan.manifest.learner_sigma_pack_cap,
                    (self.config.learner_family_rows or {}).get(
                        name, owners.plan.manifest.learner_sigma_pack_cap)),
                precision=self.config.precision,
                byte_budget=budget,
                resident_bytes=resident,
                predicted_saved_seconds=max(0.0, passes - 1.0),
                predicted_build_seconds=0.5,
            )
            if candidate is None:
                for cache in admitted.values():
                    cache.release()
                return None
            admitted[name] = candidate
            resident += candidate.allocated_bytes
        return admitted or None

    def _lane_optimizer(self, lane: int) -> object:
        lane_stepper = getattr(self.config.optimizer_owner, "lane_stepper", None)
        if lane_stepper is None:
            return self.config.lanes[lane].optimizer
        return lane_stepper(
            lane, distributed=self.config.distributed
        )

    def _prepare_learner(
        self,
        owners: GeneratedOwners,
        state: _IterationState,
        lane: int,
    ) -> PreparedLearner:
        context = self.config.lanes[lane]
        branched = owners.plan.mode.rollout_mode.is_branched
        if branched:
            stores: Mapping[str, AlignedSequentialStore | RaggedBranchedStore] = {
                name: RaggedBranchedStore(
                    owners,
                    name,
                    sealed_inputs=state.sealed[(lane, name)],
                    lane=lane,
                )
                for name in state.input_names
                if state.sealed[(lane, name)].rows
            }
            allocation = context.allocation_inputs
            statistical_counts = None
            if allocation is not None:
                counts = allocate_statistical_counts(*allocation)
                if counts.numel() != len(stores):
                    raise IterationError(
                        "statistical allocation differs from active strata"
                    )
                statistical_counts = MappingProxyType(
                    {
                        name: int(counts[index])
                        for index, name in enumerate(stores)
                    }
                )
            sampler = BranchedBatchSampler(
                stores,
                self.config.schedule,
                generator=context.generator,
                sigma_pack_cap=owners.plan.manifest.learner_sigma_pack_cap,
                action_pack_cap=owners.plan.manifest.learner_action_pack_cap,
                statistical_counts=statistical_counts,
                family_sigma_caps=self.config.learner_family_rows,
            )
        else:
            stores = {
                name: AlignedSequentialStore(
                    owners,
                    name,
                    sealed_inputs=state.sealed[(lane, name)],
                    lane=lane,
                )
                for name in state.input_names
                if state.sealed[(lane, name)].rows
            }
            allocation = context.allocation_inputs
            sampler = (
                SequentialBatchSampler(
                    stores,
                    self.config.schedule,
                    generator=context.generator,
                    reproducibility=self.config.reproducibility,
                )
                if allocation is None
                else SequentialBatchSampler.from_costs(
                    stores,
                    self.config.schedule,
                    generator=context.generator,
                    mass=allocation[0],
                    variance=allocation[1],
                    seconds_per_draw=allocation[2],
                    total_count=allocation[3],
                    reproducibility=self.config.reproducibility,
                )
            )
        reference_caches: dict[str, ReferenceOutputCache] | None = None
        try:
            reference_caches = self._reference_caches(
                owners, lane, stores, sampler=sampler
            )
            learner = LearnerOperation(
                context.mode,
                context.loss,
                context.model_operation,
                self.operations.learner_policy(lane),
                precision=self.config.precision,
            )
            if branched:
                iteration = LearnerIteration(
                    learner=learner,
                    optimizer=self._lane_optimizer(lane),  # type: ignore[arg-type]
                    sigma_pack_cap=(
                        owners.plan.manifest.learner_sigma_pack_cap
                    ),
                    action_pack_cap=(
                        owners.plan.manifest.learner_action_pack_cap
                    ),
                )
            else:
                global_cap = min(
                    owners.plan.manifest.learner_sigma_pack_cap,
                    owners.plan.manifest.learner_action_pack_cap,
                )
                costs = self.operations.model_costs(lane)
                physical_caps: int | Mapping[str, int] = global_cap
                if costs is not None:
                    declared = owners.plan.manifest.geometry
                    if (
                        max(costs.values()) > declared.model_token_bound
                        or max(value * value for value in costs.values())
                        > declared.model_token_pair_bound
                    ):
                        raise IterationError(
                            "partition model cost exceeds immutable geometry"
                        )
                    physical_caps = {
                        name: min(
                            owners.plan.manifest.learner_sigma_pack_cap,
                            owners.plan.manifest.learner_action_pack_cap,
                        )
                        for name, tokens in costs.items()
                    }
                if self.config.learner_family_rows is not None:
                    physical_caps = {name: min(global_cap, cap)
                                     for name, cap in self.config.learner_family_rows.items()}
                iteration = LearnerIteration(
                    learner=learner,
                    optimizer=self._lane_optimizer(lane),  # type: ignore[arg-type]
                    physical_batch_cap=physical_caps,
                )

            def release() -> None:
                if reference_caches is not None:
                    for cache in reference_caches.values():
                        cache.release()
                if isinstance(sampler, BranchedBatchSampler):
                    sampler.release()

            return PreparedLearner(
                iteration,
                sampler,
                reference_operation=context.reference_operation,
                reference_caches=reference_caches,
                release=release,
            )
        except Exception:
            if reference_caches is not None:
                for cache in reference_caches.values():
                    cache.release()
            if isinstance(sampler, BranchedBatchSampler):
                sampler.release()
            raise

    def _execute_learner(
        self,
        owners: GeneratedOwners,
        state: _IterationState,
        *,
        iteration: int,
    ) -> None:
        try:
            learner_results = LearnerTransaction(
                optimizer=self.config.optimizer_owner,  # type: ignore[arg-type]
                generators=tuple(
                    lane.generator for lane in self.config.lanes
                ),
                run_states=tuple(
                    lane.run_state for lane in self.config.lanes
                ),
                distributed=self.config.distributed,
                device=owners.plan.device,
                error_type=self.operations.error_type,
            ).optimize(
                lambda lane: self._prepare_learner(
                    owners, state, lane
                ),
                iteration=iteration,
            )
        finally:
            for sealed in state.sealed.values():
                sealed.release()
        for lane, result in enumerate(learner_results):
            if result is None:
                continue
            context = self.config.lanes[lane]
            if context.branch_controller is not None:
                context.branch_controller.observe(
                    state.counterfactual_sizes[lane],
                    iteration=iteration,
                )
            run_state = context.run_state
            if run_state is not None:
                run_state.logical_work["trajectories"] += run_state.root_count
                if context.branch_controller is not None:
                    run_state.branch_controller_state = (
                        context.branch_controller.state_dict()
                    )
                    run_state.cost_safeguard_state = (
                        None
                        if context.cost_safeguard is None
                        else context.cost_safeguard.state_dict()
                    )
        state.learner_results = tuple(learner_results)

    def _versions(self) -> tuple[int, ...]:
        versions = getattr(self.config.optimizer_owner, "versions", None)
        if isinstance(versions, Tensor):
            return tuple(int(value) for value in versions.tolist())
        return tuple(
            int(getattr(lane.optimizer, "step_count", 0))
            for lane in self.config.lanes
        )

    def _build_result(
        self,
        owners: GeneratedOwners,
        state: _IterationState,
    ) -> IterationBatchResult:
        effective = state.completion_observations.get(
            "posterior_effective_hands"
        )
        effective_by_lane = (
            None if effective is None else tuple(float(value) for value in effective)
        )
        results: list[LaneIterationResult | None] = []
        for lane, topology in enumerate(state.topology.topologies):
            learner = state.learner_results[lane]
            if learner is None:
                results.append(None)
                continue
            duplicate_labels = 0
            for node in range(topology.node_count):
                begin = int(topology.child_begin[node])
                degree = int(topology.executed_degree[node])
                labels = topology.label[begin : begin + degree]
                duplicate_labels += degree - int(labels.unique().numel())
            model_timing = state.model_timing.lane(lane)
            model_seconds = sum(
                float(value["seconds"])
                for value in model_timing.values()
            )
            telemetry = None
            if self.config.benchmark_telemetry:
                peak_allocated = torch.cuda.max_memory_allocated(
                    owners.plan.device
                )
                peak_reserved = torch.cuda.max_memory_reserved(
                    owners.plan.device
                )
                telemetry = IterationTelemetry(
                    rollout_seconds=state.rollout_seconds,
                    routing_traffic_bytes=state.routing_traffic[lane],
                    host_synchronizations=state.host_synchronizations,
                    child_reservation_commit_seconds=max(
                        state.rollout_seconds - model_seconds,
                        torch.finfo(torch.float64).eps,
                    ),
                    target_propagation_seconds=max(
                        state.target_seconds,
                        torch.finfo(torch.float64).eps,
                    ),
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                    model_inference_seconds=model_seconds,
                    phase_measurements=dict(state.phase_measurements),
                )
            observability = self.operations.observability(
                CompletedLane(
                    lane,
                    topology,
                    state.topology.bindings[lane],
                    owners.plan.partitions,
                    model_timing,
                )
            )
            root = topology.parent.eq(-1)
            results.append(
                LaneIterationResult(
                    learner,
                    player_nodes=int(topology.owner.ge(0).sum()),
                    leaves=state.topology.leaf_counts[lane],
                    target_layers=state.topology.plan.layer_count,
                    executed_edges=topology.edge_count,
                    root_executed_degree=tuple(
                        int(value)
                        for value in topology.executed_degree[root].tolist()
                    ),
                    root_target_degree=tuple(
                        int(value)
                        for value in topology.target_degree[root].tolist()
                    ),
                    duplicate_child_labels=duplicate_labels,
                    sampled_tree_nodes=state.topology.sampled_nodes[lane],
                    posterior_effective_hands=(
                        None
                        if effective_by_lane is None
                        else effective_by_lane[lane]
                    ),
                    telemetry=telemetry,
                    observability=MappingProxyType(dict(observability)),
                )
            )
        return IterationBatchResult(
            tuple(results),
            MappingProxyType({}),
            self._versions(),
        )

    def _run_once(
        self,
        owners: GeneratedOwners,
        *,
        iteration: int,
    ) -> IterationBatchResult:
        entered_index = 0
        rollout_started = time.perf_counter()
        owners.enter_phase(_PHASES[0])
        phase_open = True
        state: _IterationState | None = None
        try:
            state = self._execute_rollout(owners, iteration=iteration)
            remote = self.rank_agreement.agree(None)
            if remote is not None:
                agreed = self._failure_from_device(remote.failure)
                agreed._rank_agreed = True  # type: ignore[attr-defined]
                raise agreed
            if self.config.benchmark_telemetry:
                torch.cuda.synchronize(owners.plan.device)
                state.phase_measurements["ROLLOUT"] = {
                    "synchronized_wall_seconds": time.perf_counter() - rollout_started,
                    "allocated_bytes_at_boundary": torch.cuda.memory_allocated(owners.plan.device),
                    "cumulative_peak_allocated_bytes": torch.cuda.max_memory_allocated(owners.plan.device),
                    "named_live_owner_bytes": owners.memory_report().allocated_owner_bytes,
                }
            owners.release_after(ExecutionPhase.ROLLOUT)
            phase_open = False

            operations: tuple[
                tuple[ExecutionPhase, Callable[[], None]], ...
            ] = (
                (ExecutionPhase.QUEUE_DRAIN, lambda: None),
                (
                    ExecutionPhase.COMPLETION,
                    lambda: self._execute_completion(owners, state),
                ),
                (
                    ExecutionPhase.ALIGN_AND_SEAL,
                    lambda: self._seal_learner_inputs(owners, state),
                ),
                (
                    ExecutionPhase.TARGETS,
                    lambda: self._execute_targets(owners, state),
                ),
                (
                    ExecutionPhase.LEARNER,
                    lambda: self._execute_learner(
                        owners, state, iteration=iteration
                    ),
                ),
                (ExecutionPhase.FINAL_BACKWARD, lambda: None),
            )
            for index, (phase, operation) in enumerate(
                operations, start=1
            ):
                entered_index = index
                if self.config.benchmark_telemetry:
                    torch.cuda.synchronize(owners.plan.device)
                phase_started = time.perf_counter()
                owners.enter_phase(phase)
                phase_open = True
                operation()
                if self.config.benchmark_telemetry:
                    torch.cuda.synchronize(owners.plan.device)
                    state.phase_measurements[phase.name] = {
                        "synchronized_wall_seconds": time.perf_counter() - phase_started,
                        "allocated_bytes_at_boundary": torch.cuda.memory_allocated(owners.plan.device),
                        "cumulative_peak_allocated_bytes": torch.cuda.max_memory_allocated(owners.plan.device),
                        "named_live_owner_bytes": owners.memory_report().allocated_owner_bytes,
                    }
                owners.release_after(phase)
                phase_open = False
            result = self._build_result(owners, state)
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
        if state is None:
            raise TrainingError(
                "recoverable execution requires a mutable run state"
            ) from failure
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

    def run(self, *, iteration: int) -> IterationBatchResult:
        """Retry unpublished attempts and return one committed iteration."""

        while True:
            owners = self.owner_factory()
            if owners.plan != self.config.plan:
                owners.failure_state.release()
                raise IterationError(
                    "owner factory changed the immutable storage plan"
                )
            try:
                return self._run_once(owners, iteration=iteration)
            except RecoverableOverflow as caught:
                failure = self._agree_failure(caught)
                self._recover(iteration=iteration, failure=failure)


__all__ = [
    "AcceptedRollout",
    "CompletedLane",
    "CompletionContext",
    "GameCompletion",
    "GameOperations",
    "GameRollout",
    "IterationBatchResult",
    "IterationConfig",
    "IterationEngine",
    "IterationError",
    "IterationTelemetry",
    "LaneContext",
    "LaneIterationResult",
    "RolloutOperationRequest",
    "RolloutSubmission",
    "RootDestination",
    "RootPayload",
    "RootRequest",
    "SealedTopology",
]
