"""End-to-end Phase 5 NLHE iteration composition."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import torch
from torch import Tensor, nn

from pbg.games.poker.queued.hooks import (
    NLHEGeometry,
    PokerForwardInput,
    PokerLearnerPolicy,
    PokerStateMachine,
    PokerTransitionTrace,
    SampledShowdownBinding,
    SequentialSampledPokerHooks,
)
from pbg.games.poker.queued.policy import PokerPolicyOperator
from pbg.queued.capacity import (
    ExecutionPhase,
    GeneratedOwners,
    StoragePlan,
)
from pbg.queued.contracts import ModeSchema
from pbg.queued.learner import (
    AlignedSequentialStore,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    ModelOperation,
    SealedSequentialInputs,
    SequentialBatchSampler,
)
from pbg.queued.optimization import DenseAdam
from pbg.queued.persistence import RunState
from pbg.queued.rollout import (
    AsyncTopologyPublisher,
    CompletionFIFO,
    FamilyQueueState,
    FamilyScheduler,
    RolloutOwnerRegistry,
    RolloutTransaction,
    SemanticFamilyFIFO,
)
from pbg.queued.storage import (
    DestinationRequest,
    DeviceDestinationRequest,
    PublicationTransaction,
)
from pbg.queued.targets import LayeredTargetExecutor, TargetAlgebra
from pbg.queued.topology import (
    TargetKind,
    Topology,
    TopologyBindings,
    build_layer_plan,
)
from pbg.queued.training import (
    AttemptAddress,
    IterationResult,
    RecoverableOverflow,
    SequentialOverflowRecovery,
    SequentialSampledIteration,
)


class PokerIterationError(RuntimeError):
    """The end-to-end poker iteration violated its closed composition."""


@dataclass(frozen=True)
class PokerIterationResult:
    learner: IterationResult
    player_nodes: int
    leaves: int
    target_layers: int


def _i32(values: list[int]) -> Tensor:
    return torch.tensor(values, dtype=torch.int32)


class _TopologyRecorder:
    """Attempt-local trace owner; discarded wholesale on a failed attempt."""

    def __init__(
        self,
        input_names: tuple[str, ...],
        *,
        pack_cap: int,
        topology_cap: int,
        device: torch.device,
    ) -> None:
        self.input_index = {
            name: index for index, name in enumerate(input_names)
        }
        self.parent: list[int] = []
        self.owner: list[int] = []
        self.familywork_row: list[int] = []
        self.sigma_head_row: list[int] = []
        self.action_begin: list[int] = []
        self.label: list[int] = []
        self.target_kind: list[int] = []
        self.target_key: list[tuple[int, int] | None] = []
        self.target_row: list[int] = []
        self.input_type: list[int] = []
        self.logical_to_node: dict[tuple[int, int], int] = {}
        self.pending_parent: dict[tuple[int, int], int] = {}
        self.leaves = 0
        self._pack_cap = pack_cap
        self._host_trace = torch.empty(
            (topology_cap, 10), dtype=torch.int64, pin_memory=True
        )
        self._trace_rows = 0
        self._device_feedback = torch.empty(
            len(input_names) + 1, dtype=torch.int64, device=device
        )
        self._feedback = torch.empty(
            len(input_names) + 1, dtype=torch.int64, pin_memory=True
        )
        self._feedback_event = torch.cuda.Event(blocking=False)
        self._trace_events: list[torch.cuda.Event] = []
        self._trace_stream = torch.cuda.Stream(device=device)
        self._feedback_stream = torch.cuda.Stream(device=device)
        self._device = device

    def append(
        self,
        trace: PokerTransitionTrace,
        *,
        failed: Tensor,
        ready: torch.cuda.Event,
    ) -> tuple[bool, tuple[int, ...]]:
        lane_count, rows = trace.terminal.shape
        if lane_count != 1:
            raise PokerIterationError("Phase 5 topology expects one ordinary lane")
        if rows > self._pack_cap:
            raise PokerIterationError("topology trace exceeds the family pack cap")
        end = self._trace_rows + rows
        if end > self._host_trace.shape[0]:
            raise PokerIterationError("topology trace exceeds its admitted cap")
        stream = torch.cuda.current_stream(self._device)
        stream.wait_event(ready)
        packed = torch.cat(
            (
                trace.logical_rng_key.to(torch.int64),
                trace.next_logical_rng_key.to(torch.int64),
                trace.next_input_type.to(torch.int64)[..., None],
                trace.terminal.to(torch.int64)[..., None],
                trace.owner.to(torch.int64)[..., None],
                trace.action.to(torch.int64)[..., None],
                trace.familywork_row.to(torch.int64)[..., None],
                torch.full(
                    (1, rows, 1),
                    self.input_index[trace.input_type],
                    dtype=torch.int64,
                    device=self._device,
                ),
            ),
            -1,
        )[0]
        staged = torch.cuda.Event(blocking=False)
        staged.record(stream)
        with torch.cuda.stream(self._trace_stream):
            self._trace_stream.wait_event(staged)
            self._host_trace[self._trace_rows : end].copy_(
                packed, non_blocking=True
            )
            trace_ready = torch.cuda.Event(blocking=False)
            trace_ready.record(self._trace_stream)
            self._trace_events.append(trace_ready)
        with torch.cuda.stream(self._feedback_stream):
            self._feedback_stream.wait_event(staged)
            self._device_feedback[0].copy_(failed[0].to(torch.int64))
            live = ~packed[:, 5].bool()
            for index in range(len(self.input_index)):
                self._device_feedback[index + 1].copy_(
                    packed[:, 4].eq(index).logical_and(live).sum()
                )
            self._feedback.copy_(
                self._device_feedback, non_blocking=True
            )
            self._feedback_event.record(self._feedback_stream)
        packed.record_stream(self._trace_stream)
        packed.record_stream(self._feedback_stream)
        while not self._feedback_event.query():
            pass
        if bool(self._feedback[0]):
            return True, tuple(0 for _ in self.input_index)
        counts = tuple(int(value) for value in self._feedback[1:])
        self._trace_rows = end
        return False, counts

    def _accept_trace(self) -> None:
        if self._trace_events:
            with torch.cuda.stream(self._trace_stream):
                for event in self._trace_events:
                    self._trace_stream.wait_event(event)
                all_ready = torch.cuda.Event(blocking=False)
                all_ready.record(self._trace_stream)
            all_ready.synchronize()
            self._trace_events.clear()
        feedback = self._host_trace[: self._trace_rows]
        for row in range(self._trace_rows):
            logical = tuple(
                int(value) for value in feedback[row, 0:2].tolist()
            )
            if logical in self.logical_to_node:
                raise PokerIterationError("a logical player node was processed twice")
            node = len(self.parent)
            parent = self.pending_parent.pop(logical, -1)
            self.logical_to_node[logical] = node
            self.parent.append(parent)
            self.owner.append(int(feedback[row, 6]))
            self.familywork_row.append(
                int(feedback[row, 8])
            )
            self.sigma_head_row.append(
                int(feedback[row, 8])
            )
            self.action_begin.append(
                int(feedback[row, 8])
            )
            self.label.append(int(feedback[row, 7]))
            self.input_type.append(int(feedback[row, 9]))
            if bool(feedback[row, 5]):
                self.target_kind.append(int(TargetKind.LEAF))
                self.target_key.append(None)
                self.target_row.append(self.leaves)
                self.leaves += 1
            else:
                child = tuple(
                    int(value)
                    for value in feedback[row, 2:4].tolist()
                )
                if child in self.pending_parent:
                    raise PokerIterationError(
                        "two player nodes produced one logical child"
                    )
                self.pending_parent[child] = node
                self.target_kind.append(int(TargetKind.TOPOLOGY))
                self.target_key.append(child)
                self.target_row.append(-1)

    def seal(self, *, players: int) -> tuple[Topology, TopologyBindings]:
        self._accept_trace()
        if not self.parent or self.pending_parent:
            raise PokerIterationError(
                "rollout topology is empty or has unresolved player children"
            )
        for edge, key in enumerate(self.target_key):
            if key is not None:
                try:
                    self.target_row[edge] = self.logical_to_node[key]
                except KeyError as exc:
                    raise PokerIterationError(
                        "rollout topology child was never processed"
                    ) from exc
        nodes = len(self.parent)
        topology = Topology(
            parent=_i32(self.parent),
            child_begin=torch.arange(nodes, dtype=torch.int32),
            executed_degree=torch.ones(nodes, dtype=torch.int32),
            nominal_degree=torch.ones(nodes, dtype=torch.int32),
            target_degree=torch.ones(nodes, dtype=torch.int32),
            owner=_i32(self.owner),
            familywork_row=_i32(self.familywork_row),
            sigma_head_row=_i32(self.sigma_head_row),
            action_begin=_i32(self.action_begin),
            label=_i32(self.label),
            path_length=torch.ones(nodes, dtype=torch.int32),
            target_kind=_i32(self.target_kind),
            target_row=_i32(self.target_row),
        )
        bindings = TopologyBindings(
            input_type=_i32(self.input_type),
            output_type=torch.zeros(nodes, dtype=torch.int32),
            sigma_target_row=_i32(self.sigma_head_row),
            child_output_type=torch.zeros(nodes, dtype=torch.int32),
        )
        topology.validate(bindings, p_train=players)
        self._device_feedback = torch.empty(
            0, dtype=torch.int64, device=self._device
        )
        self._host_trace = torch.empty(0, dtype=torch.int64)
        self._feedback = torch.empty(0, dtype=torch.int64)
        return topology, bindings


class SequentialSampledPokerIteration:
    """Run roots through rollout, evaluator, targets, learner, and optimizer."""

    def __init__(
        self,
        *,
        geometry: NLHEGeometry,
        owners: GeneratedOwners,
        mode: ModeSchema,
        model: nn.Module,
        model_operation: ModelOperation,
        optimizer: DenseAdam,
        loss: LossConfig,
        schedule: LearnerSchedule,
        generator: torch.Generator,
        seed: int,
        lambda_: float,
        run_state: RunState | None = None,
        reference_operation: ModelOperation | None = None,
    ) -> None:
        if owners.plan.mode != mode:
            raise ValueError("poker iteration mode differs from generated owners")
        if owners.plan.manifest.n_replica != 1:
            raise ValueError("ordinary Phase 5 iteration requires one replica")
        if owners.plan.manifest.geometry.players != geometry.players:
            raise ValueError("poker iteration geometry differs from its manifest")
        if generator.device != owners.plan.device:
            raise ValueError("poker learner RNG must use the selected CUDA device")
        self.geometry = geometry
        self.owners = owners
        self.mode = mode
        self.model = model
        self.model_operation = model_operation
        self.reference_operation = reference_operation
        self.optimizer = optimizer
        self.loss = loss
        self.schedule = schedule
        self.generator = generator
        self.seed = int(seed)
        self.lambda_ = float(lambda_)
        self.run_state = run_state
        self.policy = PokerPolicyOperator()

    @staticmethod
    def _defaults(slab, rows: int) -> dict[str, Tensor]:
        return {
            field.name: torch.full(
                (slab.n_replica, rows, *field.shape),
                field.fill,
                dtype=field.dtype,
                device=slab.device,
            )
            for field in slab.schema.fields
        }

    def _publish_roots(
        self,
        machine: PokerStateMachine,
        queue: SemanticFamilyFIFO,
        *,
        iteration: int,
        logical_base: int,
        retry_epoch: int,
    ) -> None:
        roots = (
            self.owners.plan.manifest.root_count
            if self.run_state is None
            else self.run_state.root_count
        )
        if not 1 <= roots <= self.owners.plan.manifest.root_count:
            raise PokerIterationError(
                "runtime root_count exceeds its immutable admitted maximum"
            )
        state = machine.initial_roots(
            roots,
            seed=self.seed,
            iteration=iteration,
            retry_epoch=retry_epoch,
            device=self.owners.plan.device,
        )
        values = self._defaults(queue.family, roots)
        logical_root = (
            torch.arange(roots, dtype=torch.int64, device=state.device)
            + logical_base
        )
        values["logical_rng_key"][0, :, 0] = logical_root
        values["logical_rng_key"][0, :, 1] = 0
        values["true_profile"][0] = state.combo_ids
        values["model_input"][0] = machine.direct_model_input(state)
        values["owner"][0] = state.to_act
        values["sampled_private_state"][0] = state.combo_ids
        counts = torch.tensor([roots], dtype=torch.int64, device=state.device)
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    queue.family,
                    counts,
                    values,
                    "pre",
                    torch.tensor(
                        [logical_base], dtype=torch.int64, device=state.device
                    ),
                ),
            ),
            phase="poker_root_commit",
            active=torch.ones(1, dtype=torch.bool, device=state.device),
        )
        transaction.fill_all()
        success = transaction.publish()
        if not bool(success.all()):
            raise self._recoverable_overflow()
        queue.publish_family_tail(
            torch.ones(1, dtype=torch.bool, device=state.device)
        )

    def _recoverable_overflow(self) -> RecoverableOverflow:
        failure = self.owners.failure_state.snapshot(0)
        if failure is None or failure.failure_code != "capacity_overflow":
            raise PokerIterationError(
                "rollout failed without a recoverable capacity record"
            )
        candidates = [
            slab
            for slab in self.owners.live_slabs
            if slab.capacity - slab.tail(0) == failure.available_extent
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
                for slab in self.owners.live_slabs
                if slab.capacity - slab.tail(0) == failure.available_extent
            ]
        if candidates:
            slab = min(candidates, key=lambda value: value.capacity)
            capacity = slab.capacity
            observed = slab.tail(0) + failure.requested_extent
        else:
            capacity = failure.available_extent
            observed = failure.requested_extent
        return RecoverableOverflow(
            failure,
            capacity=capacity,
            observed_demand=observed,
        )

    def _model(
        self, forward: PokerForwardInput, _lanes: Tensor
    ) -> Mapping[str, Tensor]:
        with torch.no_grad():
            evaluation = self.model_operation(
                forward.input_type,
                forward.model_input,
                forward.sampled_private_state,
                forward.owner,
            )
        if not isinstance(evaluation, ModelEvaluation):
            raise PokerIterationError("rollout model returned the wrong contract")
        return MappingProxyType(
            {**dict(evaluation.policy_raw), "value": evaluation.value}
        )

    def run(self, *, iteration: int) -> PokerIterationResult:
        owners = self.owners
        owners.enter_phase(ExecutionPhase.ROLLOUT)
        registry = RolloutOwnerRegistry.from_generated(owners)
        input_names = tuple(
            partition.name for partition in owners.plan.partitions.inputs
        )
        queues = {
            name: SemanticFamilyFIFO(
                name,
                owners.plan.partitions,
                owners.slab("FamilyWork", name),
                owners.slab("QueueMetadata", name),
            )
            for name in input_names
        }
        for queue in queues.values():
            queue.initialize()
        retry_epoch = 0 if self.run_state is None else self.run_state.retry_epoch
        logical_base = (
            0 if self.run_state is None else self.run_state.attempt_logical_base
        )
        machine = PokerStateMachine(self.geometry)
        self._publish_roots(
            machine,
            queues["pre"],
            iteration=iteration,
            logical_base=logical_base,
            retry_epoch=retry_epoch,
        )
        hooks = SequentialSampledPokerHooks(
            self.geometry,
            registry,
            queues,
            seed=self.seed,
            iteration=iteration,
            policy=self.policy,
        )
        topology_publisher = AsyncTopologyPublisher(
            registry, p_train=self.geometry.players
        )
        transaction = RolloutTransaction(
            self.mode,
            seed=self.seed,
            iteration=iteration,
            topology=topology_publisher,
            owners=registry,
        )
        scheduler = FamilyScheduler(
            input_names, owners.plan.manifest.family_pack_cap
        )
        recorder = _TopologyRecorder(
            input_names,
            pack_cap=owners.plan.manifest.family_pack_cap,
            topology_cap=owners.plan.manifest.topology_cap,
            device=owners.plan.device,
        )
        depths = {name: 0 for name in input_names}
        depths["pre"] = (
            owners.plan.manifest.root_count
            if self.run_state is None
            else self.run_state.root_count
        )
        logical_time = 0
        last_event: torch.cuda.Event | None = None
        while True:
            states = tuple(
                FamilyQueueState(
                    0,
                    name,
                    index,
                    0,
                    depths[name],
                    logical_time,
                    0,
                    self.mode,
                )
                for index, (name, queue) in enumerate(queues.items())
            )
            scheduled = scheduler.select(states, allow_partial=True)
            if scheduled is None:
                break
            commit = transaction.execute(
                queues[scheduled.input_type],
                scheduled,
                self._model,
                hooks,
            )
            if hooks.last_trace is None:
                raise PokerIterationError("poker transition omitted its trace")
            failed, produced = recorder.append(
                hooks.last_trace,
                failed=commit.failed,
                ready=commit.event,
            )
            hooks.last_trace = None
            if failed:
                raise self._recoverable_overflow()
            depths[scheduled.input_type] -= scheduled.rows
            for name, count in zip(input_names, produced, strict=True):
                depths[name] += count
            last_event = commit.event
            logical_time += 1
            if logical_time > (
                owners.plan.manifest.root_count
                * self.geometry.player_node_bound
            ):
                raise PokerIterationError("poker rollout exceeded its geometry bound")
        if last_event is None:
            raise PokerIterationError("poker rollout produced no player nodes")
        topology_publisher.assert_drained()
        topology, bindings = recorder.seal(players=self.geometry.players)
        plan = build_layer_plan(
            self.mode,
            owners.plan.partitions,
            (topology,),
            (bindings,),
            p_train=self.geometry.players,
            manifest=owners.plan.manifest,
        )
        owners.release_after(ExecutionPhase.ROLLOUT)
        del hooks, transaction, topology_publisher, registry, queues

        owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
        owners.release_after(ExecutionPhase.QUEUE_DRAIN)

        owners.enter_phase(ExecutionPhase.COMPLETION)
        completion = CompletionFIFO(
            owners.slab("Leaf"),
            RolloutOwnerRegistry.from_generated(owners),
            head=torch.zeros(1, dtype=torch.int64, device=owners.plan.device),
            observed_epoch=torch.zeros(
                1, dtype=torch.int64, device=owners.plan.device
            ),
        )
        binding = SampledShowdownBinding(
            self.geometry,
            owners.slab("LayerWork", "all"),
            n_replica=1,
            pack_cap=owners.plan.manifest.completion_pack_cap,
        )
        remaining_leaves = recorder.leaves
        while remaining_leaves:
            rows = min(
                remaining_leaves,
                owners.plan.manifest.completion_pack_cap,
            )
            success = completion.complete(
                (0,),
                rows,
                binding.leaf_to_layer_work,
                wait_for=last_event,
            )
            torch._assert_async(success.all(), "poker completion failed")
            remaining_leaves -= rows
        completion.release()
        work = owners.slab("LayerWork", "all")
        missing = plan.arena_rows[0][0] - work.tail(0)
        if missing < 0:
            raise PokerIterationError("completion exceeded the target arena plan")
        if missing:
            padding = PublicationTransaction.begin(
                (
                    DestinationRequest(
                        work, 0, missing, "all", recorder.leaves
                    ),
                ),
                phase="poker_target_arena_padding",
            )
            padding.fill_defaults(0)
            padding.publish()
        owners.release_after(ExecutionPhase.COMPLETION)

        owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
        sealed = {
            name: SealedSequentialInputs.seal(
                owners.slab("FamilyWork", name), input_type=name
            )
            for name in input_names
        }
        owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)

        owners.enter_phase(ExecutionPhase.TARGETS)
        target_result = LayeredTargetExecutor(
            self.mode,
            TargetAlgebra(owners.plan.partitions, ()),
            lambda_=self.lambda_,
        ).execute(owners, plan)
        if target_result.failed_lanes:
            raise PokerIterationError("layered poker targets failed")
        owners.release_after(ExecutionPhase.TARGETS)

        owners.enter_phase(ExecutionPhase.LEARNER)
        stores = {
            name: AlignedSequentialStore(
                owners, name, sealed_inputs=sealed[name]
            )
            for name in input_names
            if sealed[name].rows
        }
        sampler = SequentialBatchSampler(
            stores, self.schedule, generator=self.generator
        )
        learner = LearnerOperation(
            self.mode,
            self.loss,
            self.model_operation,
            PokerLearnerPolicy(self.geometry, self.policy),
        )
        result = SequentialSampledIteration(
            model=self.model,
            learner=learner,
            optimizer=self.optimizer,
            run_state=self.run_state,
        ).optimize(
            sampler,
            iteration=iteration,
            reference_operation=self.reference_operation,
        )
        if self.run_state is not None:
            self.run_state.logical_work["trajectories"] += (
                self.run_state.root_count
            )
        for value in sealed.values():
            value.release()
        owners.release_after(ExecutionPhase.LEARNER)
        owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
        owners.release_after(ExecutionPhase.FINAL_BACKWARD)
        owners.close()
        return PokerIterationResult(
            result,
            player_nodes=len(recorder.parent),
            leaves=recorder.leaves,
            target_layers=plan.layer_count,
        )


def _discard_rollout_attempt(owners: GeneratedOwners) -> None:
    """Reset and close an unpublished failed attempt before replacement."""

    owners.abort_failed_lane(0)
    owners.release_after(ExecutionPhase.ROLLOUT)
    for phase in (
        ExecutionPhase.QUEUE_DRAIN,
        ExecutionPhase.COMPLETION,
        ExecutionPhase.ALIGN_AND_SEAL,
        ExecutionPhase.TARGETS,
        ExecutionPhase.LEARNER,
        ExecutionPhase.FINAL_BACKWARD,
    ):
        owners.enter_phase(phase)
        owners.release_after(phase)
    owners.close()


class RecoveringSequentialSampledPokerIteration:
    """Own attempt allocation and automatic lower-root disjoint retries."""

    def __init__(
        self,
        *,
        owner_factory: Callable[[], GeneratedOwners],
        plan: StoragePlan,
        minimum_root_count: int,
        geometry: NLHEGeometry,
        mode: ModeSchema,
        model: nn.Module,
        model_operation: ModelOperation,
        optimizer: DenseAdam,
        loss: LossConfig,
        schedule: LearnerSchedule,
        generator: torch.Generator,
        seed: int,
        lambda_: float,
        run_state: RunState,
        reference_operation: ModelOperation | None = None,
    ) -> None:
        self.owner_factory = owner_factory
        self.plan = plan
        self.minimum_root_count = int(minimum_root_count)
        self.arguments = {
            "geometry": geometry,
            "mode": mode,
            "model": model,
            "model_operation": model_operation,
            "optimizer": optimizer,
            "loss": loss,
            "schedule": schedule,
            "generator": generator,
            "seed": seed,
            "lambda_": lambda_,
            "run_state": run_state,
            "reference_operation": reference_operation,
        }
        self.run_state = run_state

    def run(self, *, iteration: int) -> PokerIterationResult:
        active: list[GeneratedOwners] = []
        controller = SequentialOverflowRecovery[PokerIterationResult](
            iteration=iteration,
            replica=0,
            root_count=self.run_state.root_count,
            minimum_root_count=self.minimum_root_count,
            run_state=self.run_state,
        )

        def attempt(
            root_count: int, address: AttemptAddress
        ) -> PokerIterationResult:
            if (
                self.run_state.root_count != root_count
                or self.run_state.retry_epoch != address.retry_epoch
                or self.run_state.attempt_logical_base != address.logical_base
            ):
                raise PokerIterationError(
                    "overflow controller and run state addresses diverged"
                )
            owners = self.owner_factory()
            if owners.plan != self.plan:
                owners.failure_state.release()
                raise PokerIterationError(
                    "overflow retry changed the immutable storage plan"
                )
            active[:] = [owners]
            return SequentialSampledPokerIteration(
                owners=owners, **self.arguments
            ).run(iteration=iteration)

        def discard(_failure: RecoverableOverflow) -> None:
            if len(active) != 1:
                raise PokerIterationError("failed attempt owner is unavailable")
            _discard_rollout_attempt(active.pop())

        return controller.run(attempt, discard)


__all__ = [
    "PokerIterationError",
    "PokerIterationResult",
    "RecoveringSequentialSampledPokerIteration",
    "SequentialSampledPokerIteration",
]
