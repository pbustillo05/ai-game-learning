"""End-to-end Phase 5 NLHE iteration composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import torch
from torch import Tensor, nn

from pbg.core.capacity import (
    GeneratedOwners,
)
from pbg.core.contracts import (
    ModeSchema,
    PrivateEstimator,
    RolloutMode,
    TrainingRegime,
)
from pbg.core.iteration import (
    CompletionLoop,
    IterationEngine,
    LearnerPreflightOptimizer,
    RolloutLoop,
)
from pbg.core.learner import (
    AlignedSequentialStore,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    ModelOperation,
    ReferenceOutputCache,
    SealedSequentialInputs,
    SequentialBatchSampler,
    build_reference_output_cache,
    compact_best_response_evaluation,
)
from pbg.core.optimization import DenseAdam
from pbg.core.persistence import (
    RunState,
)
from pbg.core.replica import (
    DistributedSUM,
    ReplicaOptimizerGroup,
)
from pbg.core.rollout import (
    AsyncTopologyPublisher,
    CompletionFIFO,
    FamilyQueueState,
    FamilyScheduler,
    RolloutOwnerRegistry,
    RolloutTransaction,
    SemanticFamilyFIFO,
)
from pbg.core.storage import (
    DestinationRequest,
    DeviceDestinationRequest,
    PublicationTransaction,
)
from pbg.core.targets import ConversionMap, LayeredTargetExecutor, TargetAlgebra
from pbg.core.topology import (
    build_layer_plan,
)
from pbg.core.training import (
    IterationResult,
    LearnerIteration,
    RecoverableOverflow,
)
from pbg.games.poker.model.capacity import (
    max_policy_partition_sequence_tokens,
)
from pbg.games.poker.queued.hooks import (
    NLHEGeometry,
    PokerForwardInput,
    PokerLearnerPolicy,
    PokerStateMachine,
    RBCompletionBinding,
    SampledShowdownBinding,
    SequentialPokerHooks,
)
from pbg.games.poker.queued.observability import topology_diagnostics
from pbg.games.poker.queued.policy import PokerPolicyOperator
from pbg.games.poker.queued.replica import PokerReplicaInference


class PokerIterationError(RuntimeError):
    """The end-to-end poker iteration violated its closed composition."""


@dataclass(frozen=True)
class PokerIterationResult:
    learner: IterationResult | None
    player_nodes: int
    leaves: int
    target_layers: int
    posterior_effective_hands: float | None = None
    observability: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class ReplicaPokerIterationResult:
    """Lane-owned results from one shared-owner poker transaction."""

    results: tuple[PokerIterationResult | None, ...]
    failures: Mapping[int, str]
    versions: tuple[int, ...]


class PokerSequentialProgram:
    """Supply sequential poker work to the common iteration engine."""

    def __init__(
        self,
        *,
        geometry: NLHEGeometry,
        owners: GeneratedOwners,
        mode: ModeSchema,
        model: nn.Module,
        model_operation: ModelOperation,
        rollout_operation: ModelOperation | None = None,
        optimizer: DenseAdam,
        loss: LossConfig,
        schedule: LearnerSchedule,
        generator: torch.Generator,
        seed: int,
        lambda_: float,
        precision: str = "fp32",
        utility: str = "raw_bb",
        cache_reference_outputs: bool = False,
        allocation_inputs: tuple[Tensor, Tensor, Tensor, int] | None = None,
        reproducibility: str = "strict",
        run_state: RunState | None = None,
        reference_operation: ModelOperation | None = None,
        frozen_operation: ModelOperation | None = None,
        completion_samples: int = 0,
        response_cutoff: bool = False,
        replica_inference: PokerReplicaInference | None = None,
        replica_optimizers: ReplicaOptimizerGroup | None = None,
        replica_losses: Sequence[LossConfig] | None = None,
        replica_generators: Sequence[torch.Generator] | None = None,
        replica_run_states: Sequence[RunState | None] | None = None,
        replica_model_operations: Sequence[ModelOperation] | None = None,
        replica_reference_operations: Sequence[ModelOperation | None]
        | None = None,
        replica_frozen_inference: PokerReplicaInference | None = None,
        distributed: DistributedSUM | None = None,
    ) -> None:
        if owners.plan.mode != mode:
            raise ValueError("poker iteration mode differs from generated owners")
        n_replica = owners.plan.manifest.n_replica
        if owners.plan.manifest.geometry.actor_count != geometry.players:
            raise ValueError("poker iteration geometry differs from its manifest")
        generators = (
            (generator,)
            if replica_generators is None
            else tuple(replica_generators)
        )
        if len(generators) != n_replica or any(
            value.device != owners.plan.device for value in generators
        ):
            raise ValueError("one device-local poker learner RNG is required per lane")
        if mode.rollout_mode is not RolloutMode.SEQUENTIAL:
            raise ValueError("sequential poker iteration requires sequential mode")
        if any(
            owners.plan.layout.input_shapes[partition.name]
            != (geometry.model_input_width(partition.name),)
            for partition in owners.plan.partitions.inputs
        ):
            raise ValueError(
                "family model-input layout differs from its natural partition"
            )
        postflop = getattr(model, "postflop", None)
        postflop_config = getattr(postflop, "cfg", None)
        m_raise = int(getattr(postflop_config, "m_raise", 1))
        from pbg.games.poker.queued.folded import (
            sequential_family_workspace_row_bytes,
        )

        required_family_workspace = sequential_family_workspace_row_bytes(
            geometry, m_raise
        )
        if (
            owners.plan.layout.family_workspace_row_bytes
            < required_family_workspace
        ):
            raise ValueError(
                "family workspace is below the concurrent sequential bound"
            )
        if mode.private_estimator is PrivateEstimator.RAO_BLACKWELL:
            from pbg.core.targets import (
                rb_conversion_library_workspace_bytes,
                rb_conversion_workspace_row_bytes,
            )

            required_conversion_workspace = (
                rb_conversion_workspace_row_bytes(
                    geometry.private_hands,
                    geometry.private_classes,
                )
            )
            if (
                owners.plan.layout.completion_workspace_row_bytes
                < required_conversion_workspace
            ):
                raise ValueError(
                    "completion workspace is below the H-to-K conversion bound"
                )
            required_library_bytes = rb_conversion_library_workspace_bytes(
                geometry.private_hands,
                geometry.private_classes,
            )
            if (
                owners.plan.manifest.library_workspace_bytes
                < required_library_bytes
            ):
                raise ValueError(
                    "library workspace omits H-to-K mask/GEMM bounds"
                )
        if (
            mode.training_regime is TrainingRegime.BEST_RESPONSE
            and frozen_operation is None
            and replica_frozen_inference is None
        ):
            raise ValueError("best response requires a frozen profile operation")
        if (
            mode.private_estimator is PrivateEstimator.RAO_BLACKWELL
            and completion_samples <= 0
        ):
            raise ValueError("RB poker requires positive completion_samples")
        self.geometry = geometry
        self.owners = owners
        self.mode = mode
        self.n_replica = n_replica
        if n_replica == 1:
            models = (model,)
            model_operations = (model_operation,)
            optimizers = (optimizer,)
        else:
            if replica_inference is None or replica_optimizers is None:
                raise ValueError(
                    "multi-lane poker requires shared replica inference and optimizer owners"
                )
            if (
                replica_inference.modules is not replica_optimizers.modules
                or replica_inference.modules.n_replica != n_replica
            ):
                raise ValueError("replica model and optimizer owners are not aligned")
            models = tuple(
                replica_inference.modules.lane_module(lane)
                for lane in range(n_replica)
            )
            model_operations = (
                tuple(replica_model_operations)
                if replica_model_operations is not None
                else tuple(value.learner_operation for value in models)
            )
            optimizers = replica_optimizers.optimizers
            if model is not models[0] or optimizer is not optimizers[0]:
                raise ValueError("lane-zero compatibility bindings lost ownership")
        if len(model_operations) != n_replica:
            raise ValueError("one learner model operation is required per lane")
        if (
            n_replica > 1
            and replica_reference_operations is None
            and reference_operation is not None
        ):
            raise ValueError(
                "multi-lane reference policy requires one operation per lane"
            )
        reference_operations = (
            (reference_operation,) * n_replica
            if replica_reference_operations is None
            else tuple(replica_reference_operations)
        )
        if len(reference_operations) != n_replica:
            raise ValueError("one reference model operation is required per lane")
        losses = (
            tuple(loss for _ in range(n_replica))
            if replica_losses is None
            else tuple(replica_losses)
        )
        if len(losses) != n_replica:
            raise ValueError("one loss control is required per lane")
        run_states = (
            (run_state,)
            if replica_run_states is None
            else tuple(replica_run_states)
        )
        if len(run_states) != n_replica:
            raise ValueError("one run state is required per lane")
        self.models = models
        self.model_operations = model_operations
        self.reference_operations = reference_operations
        self.optimizers = optimizers
        self.losses = losses
        self.generators = generators
        self.run_states = run_states
        self.replica_inference = replica_inference
        self.replica_frozen_inference = replica_frozen_inference
        self.replica_optimizers = replica_optimizers
        self.distributed = distributed
        self.model = models[0]
        self.model_operation = model_operations[0]
        self.rollout_operation = (
            (
                replica_inference
                if n_replica > 1
                else getattr(model, "rollout_operation", model_operation)
            )
            if rollout_operation is None
            else rollout_operation
        )
        self.reference_operation = reference_operations[0]
        self.frozen_operation = frozen_operation
        self.completion_samples = int(completion_samples)
        if type(response_cutoff) is not bool:
            raise TypeError("response_cutoff must be bool")
        if response_cutoff and (
            mode.training_regime is not TrainingRegime.BEST_RESPONSE
            or mode.private_estimator is not PrivateEstimator.RAO_BLACKWELL
        ):
            raise ValueError("probe cutoff requires sequential BR/RB")
        self.response_cutoff = response_cutoff
        self.optimizer = optimizers[0]
        self.loss = losses[0]
        self.schedule = schedule
        self.generator = generators[0]
        self.seed = int(seed)
        self.lambda_ = float(lambda_)
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("poker iteration precision is invalid")
        if utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("poker iteration utility is invalid")
        self.precision = precision
        self.utility = utility
        if type(cache_reference_outputs) is not bool:
            raise TypeError("cache_reference_outputs must be bool")
        self.cache_reference_outputs = cache_reference_outputs
        self.allocation_inputs = allocation_inputs
        if reproducibility not in ("strict", "seeded", "throughput"):
            raise ValueError("poker iteration reproducibility is invalid")
        self.reproducibility = reproducibility
        self.run_state = run_states[0]
        self.policy = PokerPolicyOperator()
        self._model_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event, int]]
        ] = {}
        if (
            n_replica > 1
            and mode.training_regime is TrainingRegime.BEST_RESPONSE
            and (
                replica_frozen_inference is None
                or replica_frozen_inference.modules.n_replica != n_replica
                or replica_frozen_inference.modules.device != owners.plan.device
            )
        ):
            raise ValueError(
                "replica best response requires an aligned frozen inference owner"
            )

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
        roots_by_lane = tuple(
            self.owners.plan.manifest.root_count
            if state is None
            else state.root_count
            for state in self.run_states
        )
        if any(
            not 1 <= roots <= self.owners.plan.manifest.root_count
            for roots in roots_by_lane
        ):
            raise PokerIterationError(
                "runtime root_count exceeds its immutable admitted maximum"
            )
        maximum = max(roots_by_lane)
        values = self._defaults(queue.family, maximum)
        logical_bases = tuple(
            logical_base if state is None else state.attempt_logical_base
            for state in self.run_states
        )
        retry_epochs = tuple(
            retry_epoch if state is None else state.retry_epoch
            for state in self.run_states
        )
        for lane, roots in enumerate(roots_by_lane):
            state = machine.initial_roots(
                roots,
                seed=self.seed,
                iteration=iteration,
                retry_epoch=retry_epochs[lane],
                logical_base=logical_bases[lane],
                lane=lane,
                device=self.owners.plan.device,
            )
            logical_root = (
                torch.arange(roots, dtype=torch.int64, device=state.device)
                + logical_bases[lane]
            )
            values["logical_rng_key"][lane, :roots, 0] = logical_root
            values["logical_rng_key"][lane, :roots, 1] = 0
            values["true_profile"][lane, :roots] = state.combo_ids
            values["model_input"][lane, :roots] = machine.direct_model_input(
                "pre", state
            )
            values["owner"][lane, :roots] = state.to_act
            if "sampled_private_state" in values:
                values["sampled_private_state"][lane, :roots] = state.combo_ids
            if "role_selector" in values:
                values["role_selector"][lane, :roots] = (
                    logical_root.remainder(self.geometry.players).to(torch.uint8)
                )
        device = self.owners.plan.device
        counts = torch.tensor(
            roots_by_lane, dtype=torch.int64, device=device
        )
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    queue.family,
                    counts,
                    values,
                    "pre",
                    torch.tensor(
                        logical_bases, dtype=torch.int64, device=device
                    ),
                ),
            ),
            phase="poker_root_commit",
            active=torch.ones(self.n_replica, dtype=torch.bool, device=device),
        )
        transaction.fill_all()
        success = transaction.publish()
        if not bool(success.all()):
            raise self._recoverable_overflow()
        queue.publish_family_tail(
            torch.ones(self.n_replica, dtype=torch.bool, device=device)
        )

    def _recoverable_overflow(self, lane: int = 0) -> RecoverableOverflow:
        failure = self.owners.failure_state.snapshot(lane)
        if failure is None or failure.failure_code != "capacity_overflow":
            raise PokerIterationError(
                "rollout failed without a recoverable capacity record"
            )
        candidates = [
            slab
            for slab in self.owners.live_slabs
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
                for slab in self.owners.live_slabs
                if slab.capacity - slab.tail(lane) == failure.available_extent
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

    def _model(
        self, forward: PokerForwardInput, lanes: Tensor
    ) -> Mapping[str, Tensor]:
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        n_replica = getattr(self, "n_replica", 1)
        autocast_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            if self.mode.training_regime is TrainingRegime.SELF_PLAY:
                if n_replica == 1:
                    evaluation = self.rollout_operation(
                        forward.input_type,
                        forward.model_input,
                        forward.sampled_private_state,
                        forward.owner,
                    )
                else:
                    rows = forward.owner.numel() // lanes.numel()
                    evaluation = self.rollout_operation(
                        forward.input_type,
                        forward.model_input.reshape(
                            lanes.numel(), rows, -1
                        ),
                        forward.sampled_private_state.reshape(
                            lanes.numel(), rows, -1
                        ),
                        forward.owner.reshape(lanes.numel(), rows),
                        active_lanes=lanes,
                    )
            else:
                if forward.role_selector is None:
                    raise PokerIterationError(
                        "best-response forward omitted selected roles"
                    )
                if n_replica == 1:
                    assert self.frozen_operation is not None
                    evaluation = compact_best_response_evaluation(
                        self.rollout_operation,
                        self.frozen_operation,
                        input_type=forward.input_type,
                        model_input=forward.model_input,
                        sampled_private_state=forward.sampled_private_state,
                        owner=forward.owner,
                        role_selector=forward.role_selector,
                    )
                else:
                    assert self.replica_inference is not None
                    assert self.replica_frozen_inference is not None
                    trained = forward.owner.eq(
                        forward.role_selector
                    ).reshape(lanes.numel(), -1)
                    learned = self._compact_replica_inference(
                        self.replica_inference,
                        forward,
                        lanes,
                        trained,
                    )
                    frozen = self._compact_replica_inference(
                        self.replica_frozen_inference,
                        forward,
                        lanes,
                        ~trained,
                    )
                    prototype = learned if learned is not None else frozen
                    if prototype is None:
                        raise PokerIterationError(
                            "best-response replica batch is empty"
                        )

                    def merge(
                        left: Tensor | None, right: Tensor | None
                    ) -> Tensor:
                        source = left if left is not None else right
                        assert source is not None
                        mask = trained
                        for _ in range(source.ndim - mask.ndim):
                            mask = mask.unsqueeze(-1)
                        return torch.where(
                            mask,
                            left
                            if left is not None
                            else torch.zeros_like(source),
                            right
                            if right is not None
                            else torch.zeros_like(source),
                        )

                    evaluation = ModelEvaluation(
                        MappingProxyType(
                            {
                                name: merge(
                                    None
                                    if learned is None
                                    else learned.policy_raw[name],
                                    None
                                    if frozen is None
                                    else frozen.policy_raw[name],
                                )
                                for name in prototype.policy_raw
                            }
                        ),
                        merge(
                            None if learned is None else learned.value,
                            None if frozen is None else frozen.value,
                        ),
                        MappingProxyType(
                            {
                                name: merge(
                                    None
                                    if learned is None
                                    else learned.public_heads[name],
                                    None
                                    if frozen is None
                                    else frozen.public_heads[name],
                                )
                                for name in prototype.public_heads
                            }
                        ),
                    )
        if not isinstance(evaluation, ModelEvaluation):
            raise PokerIterationError("rollout model returned the wrong contract")
        finished.record()
        events = getattr(self, "_model_events", None)
        if events is None:
            events = {}
            self._model_events = events
        events.setdefault(forward.input_type, []).append(
            (started, finished, int(forward.owner.numel()))
        )
        return MappingProxyType(
            {
                **{
                    name: value.reshape(
                        -1, *value.shape[2:]
                    ).float()
                    if n_replica > 1
                    else value.float()
                    for name, value in evaluation.policy_raw.items()
                },
                "value": (
                    evaluation.value.reshape(
                        -1, *evaluation.value.shape[2:]
                    ).float()
                    if n_replica > 1
                    else evaluation.value.float()
                ),
            }
        )

    def _model_timing(self) -> Mapping[str, object]:
        result: dict[str, object] = {}
        for family, values in getattr(self, "_model_events", {}).items():
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

    def _compact_replica_inference(
        self,
        operation: PokerReplicaInference,
        forward: PokerForwardInput,
        lanes: Tensor,
        selected: Tensor,
    ) -> ModelEvaluation | None:
        """Group equal ragged row counts and invoke no padded replica vmaps."""

        active = int(lanes.numel())
        rows = forward.owner.numel() // active
        mask = selected.reshape(active, rows)
        counts = tuple(int(value) for value in mask.sum(1).tolist())
        groups: dict[int, list[int]] = {}
        for local_lane, count in enumerate(counts):
            if count:
                groups.setdefault(count, []).append(local_lane)
        if not groups:
            return None
        policy: dict[str, Tensor] | None = None
        public: dict[str, Tensor] | None = None
        natural_value: Tensor | None = None
        model_input = forward.model_input.reshape(active, rows, -1)
        private = forward.sampled_private_state.reshape(active, rows, -1)
        owner = forward.owner.reshape(active, rows)
        for count, local_lanes in groups.items():
            local = torch.tensor(
                local_lanes, dtype=torch.int64, device=lanes.device
            )
            row = torch.stack(
                tuple(
                    torch.nonzero(mask[index], as_tuple=False).flatten()
                    for index in local_lanes
                )
            )
            current = operation(
                forward.input_type,
                model_input[local[:, None], row],
                private[local[:, None], row],
                owner[local[:, None], row],
                active_lanes=lanes.index_select(0, local),
            )
            if policy is None:
                policy = {
                    name: value.new_empty((active, rows, *value.shape[2:]))
                    for name, value in current.policy_raw.items()
                }
                public = {
                    name: value.new_empty((active, rows, *value.shape[2:]))
                    for name, value in current.public_heads.items()
                }
                natural_value = current.value.new_empty(
                    (active, rows, *current.value.shape[2:])
                )
            assert public is not None and natural_value is not None
            for name, value in current.policy_raw.items():
                policy[name][local[:, None], row] = value
            for name, value in current.public_heads.items():
                public[name][local[:, None], row] = value
            natural_value[local[:, None], row] = current.value
        assert policy is not None and public is not None and natural_value is not None
        return ModelEvaluation(
            MappingProxyType(policy),
            natural_value,
            MappingProxyType(public),
        )

    def prepare_rollout(self, *, iteration: int) -> RolloutLoop:
        owners = self.owners
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
        hooks = SequentialPokerHooks(
            self.geometry,
            registry,
            queues,
            seed=self.seed,
            iteration=iteration,
            mode=self.mode,
            policy=self.policy,
            response_cutoff=self.response_cutoff,
        )
        topology_publisher = AsyncTopologyPublisher(
            registry,
            p_train=(
                self.geometry.players
                if self.mode.training_regime is TrainingRegime.SELF_PLAY
                else 1
            ),
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
        depths = {
            lane: {name: 0 for name in input_names}
            for lane in range(self.n_replica)
        }
        for lane, state in enumerate(self.run_states):
            depths[lane]["pre"] = (
                owners.plan.manifest.root_count
                if state is None
                else state.root_count
            )
        self._iteration_last_event: torch.cuda.Event | None = None

        def queue_states(time: int) -> tuple[FamilyQueueState, ...]:
            return tuple(
                FamilyQueueState(
                    lane,
                    name,
                    index,
                    0,
                    depths[lane][name],
                    time,
                    0,
                    self.mode,
                )
                for lane in range(self.n_replica)
                for index, name in enumerate(queues)
            )

        def execute_scheduled(scheduled, _time: int):
            commit = transaction.execute(
                queues[scheduled.input_type],
                scheduled,
                self._model,
                hooks,
            )
            if not hooks.last_topology:
                raise PokerIterationError(
                    "poker transition omitted atomic topology"
                )
            publications = hooks.last_topology
            hooks.last_topology = ()
            return scheduled, commit, publications

        def accept_commits(pending: tuple[object, ...]) -> None:
            for scheduled, commit, publications in pending:
                for publication in publications:
                    failed, produced = topology_publisher.wait_deferred_feedback(
                        publication,
                        input_count=len(input_names),
                    )
                    if failed:
                        raise self._recoverable_overflow(publication.lane)
                    depths[publication.lane][scheduled.input_type] -= scheduled.rows
                    for name, count in zip(input_names, produced, strict=True):
                        depths[publication.lane][name] += count
                self._iteration_last_event = commit.event

        self._iteration_registry = registry
        self._iteration_input_names = input_names
        self._iteration_queues = queues
        self._iteration_hooks = hooks
        self._iteration_topology_publisher = topology_publisher
        self._iteration_transaction = transaction
        self._iteration_depths = depths
        self._iteration_iteration = iteration
        return RolloutLoop(
            scheduler=scheduler,
            states=queue_states,
            has_work=lambda: any(
                any(lane_depths.values()) for lane_depths in depths.values()
            ),
            execute=execute_scheduled,
            accept=accept_commits,
            inflight_cap=1,
            logical_limit=(
                self.n_replica
                * owners.plan.manifest.root_count
                * self.geometry.player_node_bound
            ),
        )

    def finish_rollout(self, *, logical_time: int) -> None:
        del logical_time
        owners = self.owners
        input_names = self._iteration_input_names
        topology_publisher = self._iteration_topology_publisher
        if self._iteration_last_event is None:
            raise PokerIterationError("poker rollout produced no player nodes")
        sealed_topologies = tuple(
            topology_publisher.seal_expansions(
                lane=lane,
                input_count=len(input_names),
                input_output=tuple(
                    tuple(
                        item.name for item in owners.plan.partitions.outputs
                    ).index(
                        owners.plan.partitions.output_for(name).name
                    )
                    for name in input_names
                ),
            )
            for lane in range(self.n_replica)
        )
        self._iteration_topologies = tuple(
            item[0] for item in sealed_topologies
        )
        self._iteration_bindings = tuple(
            item[1] for item in sealed_topologies
        )
        self._iteration_leaf_counts = tuple(
            item[2] for item in sealed_topologies
        )
        topology_publisher.assert_drained()
        self._iteration_plan = build_layer_plan(
            self.mode,
            owners.plan.partitions,
            self._iteration_topologies,
            self._iteration_bindings,
            p_train=(
                self.geometry.players
                if self.mode.training_regime is TrainingRegime.SELF_PLAY
                else 1
            ),
            manifest=owners.plan.manifest,
        )
        self._iteration_hooks = None
        self._iteration_transaction = None
        self._iteration_topology_publisher = None
        self._iteration_registry = None
        self._iteration_queues = None

    def finish_queue_drain(self) -> None:
        return None

    def prepare_completion(self) -> CompletionLoop:
        owners = self.owners
        # Completion admits native evaluator workspace against driver-visible
        # free memory. Return inactive rollout blocks held by PyTorch's cache.
        torch.cuda.synchronize(owners.plan.device)
        torch.cuda.empty_cache()
        completion = CompletionFIFO(
            owners.slab("Leaf"),
            RolloutOwnerRegistry.from_generated(owners),
            head=torch.zeros(
                self.n_replica,
                dtype=torch.int64,
                device=owners.plan.device,
            ),
            observed_epoch=torch.zeros(
                self.n_replica,
                dtype=torch.int64,
                device=owners.plan.device,
            ),
        )
        self._iteration_algebra = self._target_algebra()
        self._iteration_completion_binding = (
            SampledShowdownBinding(
                self.geometry,
                owners.slab("LayerWork", "all"),
                n_replica=self.n_replica,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                utility=self.utility,
            )
            if self.mode.private_estimator is PrivateEstimator.SAMPLED
            else RBCompletionBinding(
                self.geometry,
                {
                    name: owners.slab("LayerWork", name)
                    for name in ("pre", "post")
                },
                self._iteration_algebra,
                mode=self.mode,
                n_replica=self.n_replica,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                n_samples=self.completion_samples,
                utility=self.utility,
            )
        )
        assert self._iteration_last_event is not None
        return CompletionLoop(
            completion=completion,
            remaining=self._iteration_leaf_counts,
            hook=self._iteration_completion_binding.leaf_to_layer_work,
            wait_for=self._iteration_last_event,
            failure_message="poker completion failed",
        )

    def finish_completion(self) -> None:
        owners = self.owners
        binding = self._iteration_completion_binding
        plan = self._iteration_plan
        leaf_counts = self._iteration_leaf_counts
        posterior_effective_hands = (
            None
            if not isinstance(binding, RBCompletionBinding)
            else tuple(
                float(value)
                for value in binding.effective_hands.detach().cpu().tolist()
            )
        )
        self._iteration_posterior_effective_hands = posterior_effective_hands
        # RBCompletionBinding owns the resident H-to-K CUDA mask. Completion
        # is its exact last consumer; do not carry library workspace into
        # align/targets/learner or the zero-workspace final-backward phase.
        del binding
        for output, name in enumerate(plan.output_names):
            work = owners.slab("LayerWork", name)
            for lane in range(self.n_replica):
                missing = plan.arena_rows[lane][output] - work.tail(lane)
                if missing < 0:
                    raise PokerIterationError(
                        "completion exceeded the target arena plan"
                    )
                if missing:
                    padding = PublicationTransaction.begin(
                        (
                            DestinationRequest(
                                work,
                                lane,
                                missing,
                                name,
                                leaf_counts[lane],
                            ),
                        ),
                        phase="poker_target_arena_padding",
                    )
                    padding.fill_defaults(0)
                    padding.publish()
        if self.distributed is not None:
            self.distributed.barrier()

    def align_and_seal(self) -> None:
        owners = self.owners
        self._iteration_sealed = {
            (lane, name): SealedSequentialInputs.seal(
                owners.slab("FamilyWork", name),
                owners.slab("LearnerIndex", name),
                input_type=name,
                lane=lane,
            )
            for lane in range(self.n_replica)
            for name in self._iteration_input_names
        }

    def execute_targets(self) -> None:
        target_result = LayeredTargetExecutor(
            self.mode,
            self._iteration_algebra,
            lambda_=self.lambda_,
        ).execute(self.owners, self._iteration_plan)
        if target_result.failed_lanes:
            raise PokerIterationError("layered poker targets failed")

    def execute_learner(self, *, iteration: int) -> None:
        owners = self.owners
        input_names = self._iteration_input_names
        sealed = self._iteration_sealed
        learner_results: list[IterationResult | None] = []
        if self.n_replica > 1 or self.distributed is not None:
            assert self.replica_optimizers is not None
            try:
                IterationEngine.preflight_learners(
                    generators=self.generators,
                    validate_lane=lambda lane: self._learn_lane(
                        lane,
                        input_names=input_names,
                        sealed=sealed,
                        iteration=iteration,
                        validate_only=True,
                    ),
                    clear_gradients=lambda: self.replica_optimizers.zero_grad(
                        range(self.n_replica)
                    ),
                    distributed=self.distributed,
                    device=owners.plan.device,
                    error_type=PokerIterationError,
                )
            except Exception:
                for value in sealed.values():
                    value.release()
                raise
        for lane in range(self.n_replica):
            learner_results.append(
                self._learn_lane(
                    lane,
                    input_names=input_names,
                    sealed=sealed,
                    iteration=iteration,
                )
            )
        for value in sealed.values():
            value.release()
        self._iteration_learner_results = learner_results

    def finish_backward(
        self,
    ) -> PokerIterationResult | ReplicaPokerIterationResult:
        owners = self.owners
        topologies = self._iteration_topologies
        bindings = self._iteration_bindings
        leaf_counts = self._iteration_leaf_counts
        plan = self._iteration_plan
        learner_results = self._iteration_learner_results
        posterior_effective_hands = (
            self._iteration_posterior_effective_hands
        )
        results = tuple(
            None
            if learner_results[lane] is None
            else PokerIterationResult(
                learner_results[lane],
                player_nodes=topologies[lane].node_count,
                leaves=leaf_counts[lane],
                target_layers=plan.layer_count,
                posterior_effective_hands=(
                    None
                    if posterior_effective_hands is None
                    else posterior_effective_hands[lane]
                ),
                observability=MappingProxyType(
                    {
                        **topology_diagnostics(
                            topologies[lane],
                            bindings[lane],
                            owners.plan.partitions,
                        ),
                        "model_timing_by_family": dict(
                            self._model_timing()
                        ),
                    }
                ),
            )
            for lane in range(self.n_replica)
        )
        if self.n_replica == 1:
            assert results[0] is not None
            return results[0]
        assert self.replica_optimizers is not None
        return ReplicaPokerIterationResult(
            results,
            MappingProxyType({}),
            tuple(
                int(value)
                for value in self.replica_optimizers.versions.tolist()
            ),
        )

    def _learn_lane(
        self,
        lane: int,
        *,
        input_names: tuple[str, ...],
        sealed: Mapping[tuple[int, str], SealedSequentialInputs],
        iteration: int,
        validate_only: bool = False,
    ) -> IterationResult:
        owners = self.owners
        model = self.models[lane]
        model_operation = self.model_operations[lane]
        generator = self.generators[lane]
        run_state = self.run_states[lane]
        reference_operation = self.reference_operations[lane]
        stores = {
            name: AlignedSequentialStore(
                owners,
                name,
                sealed_inputs=sealed[(lane, name)],
                lane=lane,
            )
            for name in input_names
            if sealed[(lane, name)].rows
        }
        allocation_inputs = self.allocation_inputs
        if allocation_inputs is None and self.schedule.gradient_steps is not None:
            names = tuple(stores)
            preflop = getattr(model, "preflop", None)
            model_config = getattr(preflop, "cfg", None)
            if model_config is not None:
                token_bounds = max_policy_partition_sequence_tokens(
                    n_players=self.geometry.players,
                    stack_bb=self.geometry.stack_bb,
                    rank_count=self.geometry.rank_count,
                    k_act=model_config.k_act,
                )
                allocation_inputs = (
                    torch.tensor(
                        [float(stores[name].rows) for name in names],
                        dtype=torch.float64,
                    ),
                    torch.ones(len(names), dtype=torch.float64),
                    torch.tensor(
                        [
                            float(token_bounds[name] * token_bounds[name])
                            for name in names
                        ],
                        dtype=torch.float64,
                    ),
                    self.schedule.batch_size,
                )
        sampler = (
            SequentialBatchSampler(
                stores,
                self.schedule,
                generator=generator,
                reproducibility=self.reproducibility,
            )
            if allocation_inputs is None
            else SequentialBatchSampler.from_costs(
                stores,
                self.schedule,
                generator=generator,
                mass=allocation_inputs[0],
                variance=allocation_inputs[1],
                seconds_per_draw=allocation_inputs[2],
                total_count=allocation_inputs[3],
                reproducibility=self.reproducibility,
            )
        )
        reference_caches: dict[str, ReferenceOutputCache] | None = None
        if self.cache_reference_outputs and reference_operation is not None:
            admitted: dict[str, ReferenceOutputCache] = {}
            resident = 0
            budget = owners.plan.layout.reference_cache_bytes
            for name, store in stores.items():
                columns = store.sealed_inputs.columns
                batch_calls = max(
                    1,
                    (store.rows + self.schedule.batch_size - 1)
                    // self.schedule.batch_size,
                )
                candidate = build_reference_output_cache(
                    reference_operation,
                    input_type=name,
                    model_input=columns["model_input"],
                    sampled_private_state=columns.get(
                        "sampled_private_state", columns["owner"][:, None]
                    ),
                    owner=columns["owner"],
                    pack_cap=owners.plan.manifest.learner_sigma_pack_cap,
                    precision=self.precision,
                    byte_budget=budget,
                    resident_bytes=resident,
                    predicted_saved_seconds=float(max(0, batch_calls - 1)),
                    predicted_build_seconds=0.5,
                )
                if candidate is None:
                    for cache in admitted.values():
                        cache.release()
                    admitted = {}
                    break
                admitted[name] = candidate
                resident += candidate.allocated_bytes
            if admitted:
                reference_caches = admitted
        learner = LearnerOperation(
            self.mode,
            self.losses[lane],
            model_operation,
            PokerLearnerPolicy(self.geometry, self.policy),
            precision=self.precision,
        )
        global_cap = min(
            owners.plan.manifest.learner_sigma_pack_cap,
            owners.plan.manifest.learner_action_pack_cap,
            owners.plan.manifest.hardware.training_max_tokens
            // owners.plan.manifest.geometry.model_token_bound,
            owners.plan.manifest.hardware.training_max_token_pairs
            // owners.plan.manifest.geometry.model_token_pair_bound,
        )
        physical_caps: int | dict[str, int] = global_cap
        preflop = getattr(model, "preflop", None)
        model_config = getattr(preflop, "cfg", None)
        if model_config is not None:
            token_bounds = max_policy_partition_sequence_tokens(
                n_players=self.geometry.players,
                stack_bb=self.geometry.stack_bb,
                rank_count=self.geometry.rank_count,
                k_act=model_config.k_act,
            )
            declared = owners.plan.manifest.geometry
            if (
                max(token_bounds.values()) > declared.model_token_bound
                or max(value * value for value in token_bounds.values())
                > declared.model_token_pair_bound
            ):
                raise PokerIterationError(
                    "partition model cost exceeds the immutable geometry bound"
                )
            physical_caps = {
                name: min(
                    owners.plan.manifest.learner_sigma_pack_cap,
                    owners.plan.manifest.learner_action_pack_cap,
                    owners.plan.manifest.hardware.training_max_tokens // tokens,
                    owners.plan.manifest.hardware.training_max_token_pairs
                    // (tokens * tokens),
                )
                for name, tokens in token_bounds.items()
            }
        optimizer = (
            self.optimizers[lane]
            if self.replica_optimizers is None
            else self.replica_optimizers.lane_stepper(
                lane,
                distributed=self.distributed,
            )
        )
        if validate_only:
            optimizer = LearnerPreflightOptimizer(self.optimizers[lane])
        result = LearnerIteration(
            model=model,
            learner=learner,
            optimizer=optimizer,  # type: ignore[arg-type]
            run_state=None if validate_only else run_state,
            physical_batch_cap=physical_caps,
        ).optimize(
            sampler,
            iteration=iteration,
            reference_operation=reference_operation,
            reference_caches=reference_caches,
        )
        if reference_caches is not None:
            for cache in reference_caches.values():
                cache.release()
        if run_state is not None and not validate_only:
            run_state.logical_work["trajectories"] += run_state.root_count
        return result

    def _target_algebra(self) -> TargetAlgebra:
        if self.mode.private_estimator is PrivateEstimator.SAMPLED:
            return TargetAlgebra(self.owners.plan.partitions, ())
        cards = PokerStateMachine(self.geometry).combo_cards(
            torch.arange(self.geometry.private_hands, dtype=torch.int32)
        )
        ranks = self.geometry.rank_count
        rank = cards.long() // 4 - (13 - ranks)
        high, low = rank.max(1).values, rank.min(1).values
        hp, lp = ranks - 1 - high, ranks - 1 - low
        suited_or_pair = rank[:, 0].eq(rank[:, 1]) | cards[:, 0].remainder(4).eq(
            cards[:, 1].remainder(4)
        )
        classes = torch.where(
            suited_or_pair, hp * ranks + lp, lp * ranks + hp
        )
        return TargetAlgebra(
            self.owners.plan.partitions,
            (ConversionMap("post", "pre", tuple(int(x) for x in classes.tolist())),),
        )

__all__ = [
    "PokerIterationError",
    "PokerIterationResult",
    "PokerSequentialProgram",
]
