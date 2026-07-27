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
from pbg.queued.capacity import ExecutionPhase, GeneratedOwners
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
    IterationResult,
    RecoverableOverflow,
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

    def __init__(self, input_names: tuple[str, ...]) -> None:
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

    def append(self, trace: PokerTransitionTrace) -> None:
        tensors = {
            "logical": trace.logical_rng_key.to("cpu"),
            "next_logical": trace.next_logical_rng_key.to("cpu"),
            "terminal": trace.terminal.to("cpu"),
            "owner": trace.owner.to("cpu"),
            "action": trace.action.to("cpu"),
            "familywork_row": trace.familywork_row.to("cpu"),
        }
        lane_count, rows = tensors["terminal"].shape
        if lane_count != 1:
            raise PokerIterationError("Phase 5 topology expects one ordinary lane")
        for row in range(rows):
            logical = tuple(
                int(value) for value in tensors["logical"][0, row].tolist()
            )
            if logical in self.logical_to_node:
                raise PokerIterationError("a logical player node was processed twice")
            node = len(self.parent)
            parent = self.pending_parent.pop(logical, -1)
            self.logical_to_node[logical] = node
            self.parent.append(parent)
            self.owner.append(int(tensors["owner"][0, row]))
            self.familywork_row.append(
                int(tensors["familywork_row"][0, row])
            )
            self.sigma_head_row.append(
                int(tensors["familywork_row"][0, row])
            )
            self.action_begin.append(
                int(tensors["familywork_row"][0, row])
            )
            self.label.append(int(tensors["action"][0, row]))
            self.input_type.append(self.input_index[trace.input_type])
            if bool(tensors["terminal"][0, row]):
                self.target_kind.append(int(TargetKind.LEAF))
                self.target_key.append(None)
                self.target_row.append(self.leaves)
                self.leaves += 1
            else:
                child = tuple(
                    int(value)
                    for value in tensors["next_logical"][0, row].tolist()
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
        torch._assert_async(success.all(), "poker root publication failed")
        queue.publish_family_tail(
            torch.ones(1, dtype=torch.bool, device=state.device)
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
        recorder = _TopologyRecorder(input_names)
        logical_time = 0
        last_event: torch.cuda.Event | None = None
        while True:
            states = tuple(
                FamilyQueueState(
                    0,
                    name,
                    index,
                    int(queue.head[0]),
                    int(queue.ready_tail[0]),
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
            commit.event.synchronize()
            if bool(commit.failed.any()):
                failure = owners.failure_state.snapshot(0)
                if failure is None or failure.failure_code != "capacity_overflow":
                    raise PokerIterationError(
                        "rollout failed without a recoverable capacity record"
                    )
                matching_capacities = [
                    slab.capacity
                    for slab in owners.live_slabs
                    if slab.capacity - slab.tail(0) == failure.available_extent
                ]
                capacity = (
                    min(matching_capacities)
                    if matching_capacities
                    else failure.available_extent
                )
                raise RecoverableOverflow(
                    failure,
                    capacity=capacity,
                    observed_demand=failure.requested_extent,
                )
            if hooks.last_trace is None:
                raise PokerIterationError("poker transition omitted its trace")
            recorder.append(hooks.last_trace)
            hooks.last_trace = None
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
        while int(completion.depth[0]):
            rows = min(
                int(completion.depth[0]),
                owners.plan.manifest.completion_pack_cap,
            )
            success = completion.complete(
                (0,),
                rows,
                binding.leaf_to_layer_work,
                wait_for=last_event,
            )
            torch._assert_async(success.all(), "poker completion failed")
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


__all__ = [
    "PokerIterationError",
    "PokerIterationResult",
    "SequentialSampledPokerIteration",
]
