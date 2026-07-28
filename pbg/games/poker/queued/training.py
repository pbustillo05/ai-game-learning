"""End-to-end Phase 5 NLHE iteration composition."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from torch import Tensor, nn

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
from pbg.games.poker.queued.policy import PokerPolicyOperator
from pbg.games.poker.queued.replica import PokerReplicaInference
from pbg.games.poker.queued.response import (
    PairedProbeStatistics,
    SequentialSharedResponseEvaluator,
    paired_probe_statistics,
    probe_evaluation_design,
)
from pbg.queued.capacity import (
    ExecutionPhase,
    GeneratedOwners,
    StoragePlan,
)
from pbg.queued.config import Phase5Config
from pbg.queued.contracts import (
    ModeSchema,
    PrivateEstimator,
    RolloutMode,
    TrainingRegime,
)
from pbg.queued.learner import (
    AlignedSequentialStore,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    ModelOperation,
    ReferenceOutputCache,
    ReferenceState,
    SealedSequentialInputs,
    SequentialBatchSampler,
    build_reference_output_cache,
    compact_best_response_evaluation,
)
from pbg.queued.optimization import DenseAdam, LossScaler
from pbg.queued.persistence import (
    RunState,
    load_checkpoint,
    save_checkpoint,
    trim_metrics_after_checkpoint,
)
from pbg.queued.replica import (
    DistributedSUM,
    ReplicaOptimizerGroup,
)
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
from pbg.queued.targets import ConversionMap, LayeredTargetExecutor, TargetAlgebra
from pbg.queued.topology import (
    build_layer_plan,
)
from pbg.queued.training import (
    AttemptAddress,
    IterationResult,
    PreemptionNotice,
    RecoverableOverflow,
    SequentialOverflowRecovery,
    SequentialSampledIteration,
)


class PokerIterationError(RuntimeError):
    """The end-to-end poker iteration violated its closed composition."""


@dataclass(frozen=True)
class PokerIterationResult:
    learner: IterationResult | None
    player_nodes: int
    leaves: int
    target_layers: int
    telemetry: SequentialBenchmarkTelemetry | None = None


@dataclass(frozen=True)
class ReplicaPokerIterationResult:
    """Lane-owned results from one shared-owner poker transaction."""

    results: tuple[PokerIterationResult | None, ...]
    failures: Mapping[int, str]
    versions: tuple[int, ...]


@dataclass(frozen=True)
class SequentialBenchmarkTelemetry:
    """Rollout-only timing emitted solely by an explicit benchmark run."""

    rollout_seconds: float


class SequentialPokerIteration:
    """Run any sequential schema through the sole queued engine."""

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
        probe_cutoff: bool = False,
        benchmark_rollout_only: bool = False,
        replica_inference: PokerReplicaInference | None = None,
        replica_optimizers: ReplicaOptimizerGroup | None = None,
        replica_losses: Sequence[LossConfig] | None = None,
        replica_generators: Sequence[torch.Generator] | None = None,
        replica_run_states: Sequence[RunState | None] | None = None,
        replica_model_operations: Sequence[ModelOperation] | None = None,
        replica_frozen_inference: PokerReplicaInference | None = None,
        distributed: DistributedSUM | None = None,
    ) -> None:
        if owners.plan.mode != mode:
            raise ValueError("poker iteration mode differs from generated owners")
        n_replica = owners.plan.manifest.n_replica
        if owners.plan.manifest.geometry.players != geometry.players:
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
        expected_width = geometry.direct_model_input_width
        if any(
            owners.plan.layout.input_shapes[partition.name]
            != (expected_width,)
            for partition in owners.plan.partitions.inputs
        ):
            raise ValueError(
                "family model-input layout omits retained folded statistics"
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
            from pbg.queued.targets import (
                rb_conversion_library_workspace_bytes,
                rb_conversion_workspace_row_bytes,
            )

            required_conversion_workspace = (
                rb_conversion_workspace_row_bytes(
                    geometry.private_hands,
                    geometry.rank_count**2,
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
                geometry.rank_count**2,
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
                else model_operation
            )
            if rollout_operation is None
            else rollout_operation
        )
        self.reference_operation = reference_operation
        self.frozen_operation = frozen_operation
        self.completion_samples = int(completion_samples)
        if type(probe_cutoff) is not bool:
            raise TypeError("probe_cutoff must be bool")
        if probe_cutoff and (
            mode.training_regime is not TrainingRegime.BEST_RESPONSE
            or mode.private_estimator is not PrivateEstimator.RAO_BLACKWELL
        ):
            raise ValueError("probe cutoff requires sequential BR/RB")
        self.probe_cutoff = probe_cutoff
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
        if type(benchmark_rollout_only) is not bool:
            raise TypeError("benchmark_rollout_only must be bool")
        self.benchmark_rollout_only = benchmark_rollout_only
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
            values["model_input"][lane, :roots] = machine.direct_model_input(state)
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
                if self.n_replica == 1:
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
                if self.n_replica == 1:
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
        return MappingProxyType(
            {
                **{
                    name: value.reshape(
                        -1, *value.shape[2:]
                    ).float()
                    if self.n_replica > 1
                    else value.float()
                    for name, value in evaluation.policy_raw.items()
                },
                "value": (
                    evaluation.value.reshape(
                        -1, *evaluation.value.shape[2:]
                    ).float()
                    if self.n_replica > 1
                    else evaluation.value.float()
                ),
            }
        )

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

    def run(
        self, *, iteration: int
    ) -> PokerIterationResult | ReplicaPokerIterationResult:
        owners = self.owners
        if self.benchmark_rollout_only:
            torch.cuda.synchronize(owners.plan.device)
            rollout_started = time.perf_counter()
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
        hooks = SequentialPokerHooks(
            self.geometry,
            registry,
            queues,
            seed=self.seed,
            iteration=iteration,
            mode=self.mode,
            policy=self.policy,
            probe_cutoff=self.probe_cutoff,
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
        logical_time = 0
        last_event: torch.cuda.Event | None = None
        while True:
            states = tuple(
                FamilyQueueState(
                    lane,
                    name,
                    index,
                    0,
                    depths[lane][name],
                    logical_time,
                    0,
                    self.mode,
                )
                for lane in range(self.n_replica)
                for index, name in enumerate(queues)
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
            if not hooks.last_topology:
                raise PokerIterationError(
                    "poker transition omitted atomic topology"
                )
            publications = hooks.last_topology
            hooks.last_topology = ()
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
            last_event = commit.event
            logical_time += 1
            if logical_time > (
                self.n_replica
                * owners.plan.manifest.root_count
                * self.geometry.player_node_bound
            ):
                raise PokerIterationError("poker rollout exceeded its geometry bound")
        if last_event is None:
            raise PokerIterationError("poker rollout produced no player nodes")
        sealed_topologies = tuple(
            topology_publisher.seal_deferred_unary(
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
        topologies = tuple(item[0] for item in sealed_topologies)
        bindings = tuple(item[1] for item in sealed_topologies)
        leaf_counts = tuple(item[2] for item in sealed_topologies)
        topology_publisher.assert_drained()
        plan = build_layer_plan(
            self.mode,
            owners.plan.partitions,
            topologies,
            bindings,
            p_train=(
                self.geometry.players
                if self.mode.training_regime is TrainingRegime.SELF_PLAY
                else 1
            ),
            manifest=owners.plan.manifest,
        )
        if self.benchmark_rollout_only:
            if self.n_replica != 1:
                raise PokerIterationError(
                    "rollout benchmark requires the ordinary one-lane composition"
                )
            torch.cuda.synchronize(owners.plan.device)
            rollout_seconds = time.perf_counter() - rollout_started
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
            return PokerIterationResult(
                None,
                player_nodes=int(topologies[0].owner.ge(0).sum()),
                leaves=leaf_counts[0],
                target_layers=plan.layer_count,
                telemetry=SequentialBenchmarkTelemetry(rollout_seconds),
            )
        owners.release_after(ExecutionPhase.ROLLOUT)
        del hooks, transaction, topology_publisher, registry, queues

        owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
        owners.release_after(ExecutionPhase.QUEUE_DRAIN)

        owners.enter_phase(ExecutionPhase.COMPLETION)
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
        algebra = self._target_algebra()
        binding = (
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
                algebra,
                mode=self.mode,
                n_replica=self.n_replica,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                n_samples=self.completion_samples,
                utility=self.utility,
            )
        )
        remaining_leaves = list(leaf_counts)
        while any(remaining_leaves):
            active_lanes = tuple(
                lane
                for lane, remaining in enumerate(remaining_leaves)
                if remaining > 0
            )
            rows = min(
                owners.plan.manifest.completion_pack_cap,
                *(remaining_leaves[lane] for lane in active_lanes),
            )
            success = completion.complete(
                active_lanes,
                rows,
                binding.leaf_to_layer_work,
                wait_for=last_event,
            )
            torch._assert_async(success.all(), "poker completion failed")
            for lane in active_lanes:
                remaining_leaves[lane] -= rows
        completion.release()
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
        owners.release_after(ExecutionPhase.COMPLETION)

        owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
        sealed = {
            (lane, name): SealedSequentialInputs.seal(
                owners.slab("FamilyWork", name),
                owners.slab("LearnerIndex", name),
                input_type=name,
                lane=lane,
            )
            for lane in range(self.n_replica)
            for name in input_names
        }
        owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)

        owners.enter_phase(ExecutionPhase.TARGETS)
        target_result = LayeredTargetExecutor(
            self.mode,
            algebra,
            lambda_=self.lambda_,
        ).execute(owners, plan)
        if target_result.failed_lanes:
            raise PokerIterationError("layered poker targets failed")
        owners.release_after(ExecutionPhase.TARGETS)

        owners.enter_phase(ExecutionPhase.LEARNER)
        learner_results: list[IterationResult | None] = []
        learner_failures: dict[int, str] = {}
        for lane in range(self.n_replica):
            try:
                learner_results.append(
                    self._learn_lane(
                        lane,
                        input_names=input_names,
                        sealed=sealed,
                        iteration=iteration,
                    )
                )
            except Exception as exc:
                learner_results.append(None)
                learner_failures[lane] = f"{type(exc).__name__}: {exc}"
                if self.replica_optimizers is not None:
                    self.replica_optimizers.gradients[lane].zero_()
                else:
                    raise
        for value in sealed.values():
            value.release()
        owners.release_after(ExecutionPhase.LEARNER)
        owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
        owners.release_after(ExecutionPhase.FINAL_BACKWARD)
        owners.close()
        results = tuple(
            None
            if learner_results[lane] is None
            else PokerIterationResult(
                learner_results[lane],
                player_nodes=topologies[lane].node_count,
                leaves=leaf_counts[lane],
                target_layers=plan.layer_count,
            )
            for lane in range(self.n_replica)
        )
        if self.n_replica == 1:
            assert results[0] is not None
            return results[0]
        assert self.replica_optimizers is not None
        return ReplicaPokerIterationResult(
            results,
            MappingProxyType(learner_failures),
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
    ) -> IterationResult:
        owners = self.owners
        model = self.models[lane]
        model_operation = self.model_operations[lane]
        generator = self.generators[lane]
        run_state = self.run_states[lane]
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
        if self.cache_reference_outputs and self.reference_operation is not None:
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
                    self.reference_operation,
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
        result = SequentialSampledIteration(
            model=model,
            learner=learner,
            optimizer=optimizer,  # type: ignore[arg-type]
            run_state=run_state,
            physical_batch_cap=physical_caps,
        ).optimize(
            sampler,
            iteration=iteration,
            reference_operation=self.reference_operation,
            reference_caches=reference_caches,
        )
        if reference_caches is not None:
            for cache in reference_caches.values():
                cache.release()
        if run_state is not None:
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


# Phase 5 public imports continue to name the shared implementation.
SequentialSampledPokerIteration = SequentialPokerIteration


@dataclass(frozen=True)
class SequentialSharedResponseProbe:
    """Shared-seat probe consumer using the ordinary sequential BR/RB path."""

    iteration: SequentialPokerIteration
    target_sha256: str
    initial_model_sha256: str
    initialization: str = "random"

    def __post_init__(self) -> None:
        mode = self.iteration.mode
        if (
            mode.training_regime is not TrainingRegime.BEST_RESPONSE
            or mode.rollout_mode is not RolloutMode.SEQUENTIAL
            or mode.private_estimator is not PrivateEstimator.RAO_BLACKWELL
        ):
            raise ValueError("shared-response probe requires sequential BR/RB")
        if not self.iteration.probe_cutoff:
            raise ValueError(
                "shared-response probe requires selected-seat fold cutoff"
            )
        if (
            len(self.target_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.target_sha256
            )
        ):
            raise ValueError("probe target_sha256 must be lowercase SHA-256")
        if (
            len(self.initial_model_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.initial_model_sha256
            )
        ):
            raise ValueError(
                "probe initial_model_sha256 must be lowercase SHA-256"
            )
        if self.initialization not in ("random", "target"):
            raise ValueError("probe initialization must be random or target")

    @property
    def mode(self) -> ModeSchema:
        return self.iteration.mode

    def run(self, *, iteration: int) -> PokerIterationResult:
        return self.iteration.run(iteration=iteration)

    def summarize_paired_gain(
        self, differences: Tensor, trajectory_ids: Tensor
    ) -> PairedProbeStatistics:
        seats = probe_evaluation_design(
            trajectory_ids, self.iteration.geometry.players
        )
        return paired_probe_statistics(
            differences, seats, self.iteration.geometry.players
        )

    def evaluator(self) -> SequentialSharedResponseEvaluator:
        return SequentialSharedResponseEvaluator(
            self.iteration.geometry.players,
            self.target_sha256,
            self.initial_model_sha256,
            self.initialization,
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
            "rollout_operation": (
                model_operation
                if rollout_operation is None
                else rollout_operation
            ),
            "optimizer": optimizer,
            "loss": loss,
            "schedule": schedule,
            "generator": generator,
            "seed": seed,
            "lambda_": lambda_,
            "precision": precision,
            "utility": utility,
            "cache_reference_outputs": cache_reference_outputs,
            "allocation_inputs": allocation_inputs,
            "reproducibility": reproducibility,
            "run_state": run_state,
            "reference_operation": reference_operation,
        }
        self.mode = mode
        self.model = model
        self.model_operation = model_operation
        self.rollout_operation = self.arguments["rollout_operation"]
        self.optimizer = optimizer
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


class Phase5TrainingSession:
    """Public config-driven Phase 5 composition and strict resume boundary."""

    def __init__(
        self,
        *,
        config: Phase5Config,
        plan: StoragePlan,
        owner_factory: Callable[[], GeneratedOwners],
        model: nn.Module,
        reference_model: nn.Module,
        model_operation: ModelOperation,
        reference_operation: ModelOperation | None,
        optimizer: DenseAdam,
        generator: torch.Generator,
        run_state: RunState,
        reference_state: ReferenceState,
        execution_config: Mapping[str, object] | None = None,
        preemption: PreemptionNotice | None = None,
        preemption_checkpoint: str | Path | None = None,
        allocation_inputs: tuple[Tensor, Tensor, Tensor, int] | None = None,
    ) -> None:
        geometry = NLHEGeometry(
            config.game.players,
            config.game.stack_bb,
            config.game.rank_count,
        )
        expected_mode = "self_play/sequential/sampled"
        if plan.mode.key != expected_mode:
            raise ValueError("Phase 5 session plan has the wrong mode")
        if (
            plan.manifest.geometry.players != geometry.players
            or plan.manifest.geometry.stack_bb != geometry.stack_bb
            or plan.manifest.geometry.rank_count != geometry.rank_count
            or plan.manifest.root_count != config.root_count
        ):
            raise ValueError("Phase 5 public config differs from the storage plan")
        if not config.root_count_minimum <= run_state.root_count <= config.root_count:
            raise ValueError("run root_count is outside the public config")
        if generator.device != plan.device:
            raise ValueError("session RNG differs from the admitted CUDA device")
        if optimizer.config != config.optimizer:
            raise ValueError("session optimizer differs from the public config")
        if optimizer.scaler.enabled != (config.precision.name == "fp16"):
            raise ValueError("loss scaler enablement differs from precision")
        if (
            reference_state.interval_iterations
            != config.reference.interval_iterations
            or reference_state.interval_effective_hands
            != config.reference.interval_effective_hands
        ):
            raise ValueError("reference cadence differs from the public config")
        from pbg.games.poker.queued.folded import (
            sequential_family_workspace_row_bytes,
        )

        required_model_workspace = sequential_family_workspace_row_bytes(
            geometry, config.model.m_raise
        )
        if plan.layout.family_workspace_row_bytes < required_model_workspace:
            raise ValueError(
                "family model workspace is below the derived Phase 5 geometry"
            )
        if plan.layout.topology_packet_row_bytes < 11 * 8:
            raise ValueError(
                "topology packet workspace is below the atomic trace width"
            )
        self.config = config
        self.geometry = geometry
        self.plan = plan
        self.owner_factory = owner_factory
        self.model = model
        self.reference_model = reference_model
        self.model_operation = model_operation
        self.reference_operation = reference_operation
        self.optimizer = optimizer
        self.generator = generator
        self.run_state = run_state
        self.reference_state = reference_state
        self.execution_config = MappingProxyType(
            {} if execution_config is None else dict(execution_config)
        )
        configurable = tuple(
            configure
            for configure in (
                getattr(model, "configure_execution", None),
                getattr(reference_model, "configure_execution", None),
            )
            if callable(configure)
        )
        if configurable:
            hardware_profile = self.execution_config.get("hardware_profile")
            if not isinstance(hardware_profile, Mapping):
                raise ValueError(
                    "queued poker model requires an execution hardware profile"
                )
            try:
                for configure_execution in configurable:
                    configure_execution(
                        str(hardware_profile["attention_backend"]),
                        int(hardware_profile["query_chunk_size"]),
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "execution hardware profile has no valid attention plan"
                ) from exc
        self.preemption = preemption
        self.preemption_checkpoint = (
            None
            if preemption_checkpoint is None
            else Path(preemption_checkpoint)
        )
        self.allocation_inputs = allocation_inputs

    @classmethod
    def fresh(
        cls,
        *,
        config: Phase5Config,
        plan: StoragePlan,
        owner_factory: Callable[[], GeneratedOwners],
        model: nn.Module,
        reference_model: nn.Module,
        model_operation: ModelOperation,
        reference_operation: ModelOperation | None,
        generator: torch.Generator,
        run_id: str,
        execution_config: Mapping[str, object] | None = None,
        preemption: PreemptionNotice | None = None,
        preemption_checkpoint: str | Path | None = None,
        allocation_inputs: tuple[Tensor, Tensor, Tensor, int] | None = None,
    ) -> Phase5TrainingSession:
        reference_model.load_state_dict(model.state_dict(), strict=True)
        scaler = LossScaler(enabled=config.precision.name == "fp16")
        return cls(
            config=config,
            plan=plan,
            owner_factory=owner_factory,
            model=model,
            reference_model=reference_model,
            model_operation=model_operation,
            reference_operation=reference_operation,
            optimizer=DenseAdam(model, config.optimizer, scaler=scaler),
            generator=generator,
            run_state=RunState(
                run_id,
                root_count=config.root_count,
            ),
            reference_state=ReferenceState(
                interval_iterations=config.reference.interval_iterations,
                interval_effective_hands=(
                    config.reference.interval_effective_hands
                ),
            ),
            execution_config=execution_config,
            preemption=preemption,
            preemption_checkpoint=preemption_checkpoint,
            allocation_inputs=allocation_inputs,
        )

    @classmethod
    def resume(
        cls,
        checkpoint: str | Path,
        *,
        config: Phase5Config,
        plan: StoragePlan,
        owner_factory: Callable[[], GeneratedOwners],
        model: nn.Module,
        reference_model: nn.Module,
        model_operation: ModelOperation,
        reference_operation: ModelOperation | None,
        generator: torch.Generator,
        external_metrics: tuple[Mapping[str, object], ...] = (),
        preemption: PreemptionNotice | None = None,
        preemption_checkpoint: str | Path | None = None,
        allocation_inputs: tuple[Tensor, Tensor, Tensor, int] | None = None,
    ) -> Phase5TrainingSession:
        optimizer = DenseAdam(
            model,
            config.optimizer,
            scaler=LossScaler(enabled=config.precision.name == "fp16"),
        )
        scaler, reference_state, run_state, execution = load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            reference_model=reference_model,
            expected_scientific_config=config.scientific_mapping(),
            generator=generator,
        )
        optimizer.scaler = scaler
        if external_metrics:
            run_state.metrics_prefix = trim_metrics_after_checkpoint(
                external_metrics, run_state.iteration - 1
            )
        return cls(
            config=config,
            plan=plan,
            owner_factory=owner_factory,
            model=model,
            reference_model=reference_model,
            model_operation=model_operation,
            reference_operation=reference_operation,
            optimizer=optimizer,
            generator=generator,
            run_state=run_state,
            reference_state=reference_state,
            execution_config=execution,
            preemption=preemption,
            preemption_checkpoint=preemption_checkpoint,
            allocation_inputs=allocation_inputs,
        )

    def checkpoint(self, path: str | Path) -> str:
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.optimizer.scaler,
            reference_model=self.reference_model,
            reference_state=self.reference_state,
            run_state=self.run_state,
            scientific_config=self.config.scientific_mapping(),
            execution_config=self.execution_config,
            generator=self.generator,
        )

    def run(self, *, iteration: int | None = None) -> PokerIterationResult:
        selected_iteration = (
            self.run_state.iteration if iteration is None else int(iteration)
        )
        if selected_iteration != self.run_state.iteration:
            raise ValueError("session may run only the next checkpointed iteration")
        result = RecoveringSequentialSampledPokerIteration(
            owner_factory=self.owner_factory,
            plan=self.plan,
            minimum_root_count=self.config.root_count_minimum,
            geometry=self.geometry,
            mode=self.plan.mode,
            model=self.model,
            model_operation=self.model_operation,
            optimizer=self.optimizer,
            loss=self.config.loss,
            schedule=self.config.schedule,
            generator=self.generator,
            seed=self.config.seed,
            lambda_=self.config.lambda_,
            precision=self.config.precision.name,
            utility=self.config.utility,
            cache_reference_outputs=self.config.reference.cache_outputs,
            allocation_inputs=self.allocation_inputs,
            reproducibility=self.config.reproducibility,
            run_state=self.run_state,
            reference_operation=self.reference_operation,
        ).run(iteration=selected_iteration)
        if self.reference_state.observe(
            iteration=self.run_state.iteration,
            effective_hands=float(result.player_nodes),
        ):
            self.reference_model.load_state_dict(
                self.model.state_dict(), strict=True
            )
        if self.preemption is not None and self.preemption.requested:
            if self.preemption_checkpoint is None:
                raise PokerIterationError(
                    "preemption requested without a checkpoint destination"
                )
            self.checkpoint(self.preemption_checkpoint)
        return result


__all__ = [
    "Phase5TrainingSession",
    "PokerIterationError",
    "PokerIterationResult",
    "RecoveringSequentialSampledPokerIteration",
    "SequentialPokerIteration",
    "SequentialSampledPokerIteration",
    "SequentialSharedResponseProbe",
]
