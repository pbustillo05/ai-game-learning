"""End-to-end Phase 5 NLHE iteration composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

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
    SequentialSampledPokerHooks,
)
from pbg.games.poker.queued.policy import PokerPolicyOperator
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
)
from pbg.queued.optimization import DenseAdam, LossScaler
from pbg.queued.persistence import (
    RunState,
    load_checkpoint,
    save_checkpoint,
    trim_metrics_after_checkpoint,
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
    learner: IterationResult
    player_nodes: int
    leaves: int
    target_layers: int


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
    ) -> None:
        if owners.plan.mode != mode:
            raise ValueError("poker iteration mode differs from generated owners")
        if owners.plan.manifest.n_replica != 1:
            raise ValueError("ordinary Phase 5 iteration requires one replica")
        if owners.plan.manifest.geometry.players != geometry.players:
            raise ValueError("poker iteration geometry differs from its manifest")
        if generator.device != owners.plan.device:
            raise ValueError("poker learner RNG must use the selected CUDA device")
        if mode.rollout_mode is not RolloutMode.SEQUENTIAL:
            raise ValueError("sequential poker iteration requires sequential mode")
        if (
            mode.training_regime is TrainingRegime.BEST_RESPONSE
            and frozen_operation is None
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
        self.model = model
        self.model_operation = model_operation
        self.rollout_operation = (
            model_operation if rollout_operation is None else rollout_operation
        )
        self.reference_operation = reference_operation
        self.frozen_operation = frozen_operation
        self.completion_samples = int(completion_samples)
        self.optimizer = optimizer
        self.loss = loss
        self.schedule = schedule
        self.generator = generator
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
            logical_base=logical_base,
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
        if "sampled_private_state" in values:
            values["sampled_private_state"][0] = state.combo_ids
        if "role_selector" in values:
            # Root identity is the counter-keyed seat draw.  Consecutive roots
            # form the preferred cyclic balance whenever the count permits it.
            values["role_selector"][0] = (
                logical_root.remainder(self.geometry.players).to(torch.uint8)
            )
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
                evaluation = self.rollout_operation(
                    forward.input_type,
                    forward.model_input,
                    forward.sampled_private_state,
                    forward.owner,
                )
            else:
                assert self.frozen_operation is not None
                frozen = self.frozen_operation(
                    forward.input_type,
                    forward.model_input,
                    forward.sampled_private_state,
                    forward.owner,
                )
                if forward.role_selector is None:
                    raise PokerIterationError(
                        "best-response forward omitted selected roles"
                    )
                trained = forward.owner.to(torch.int64).eq(
                    forward.role_selector.to(torch.int64)
                )
                learned = self.rollout_operation(
                    forward.input_type,
                    forward.model_input,
                    forward.sampled_private_state,
                    forward.owner,
                )
                shape = (trained.shape[0],) + (1,) * (learned.value.ndim - 1)
                evaluation = ModelEvaluation(
                    MappingProxyType(
                        {
                            name: torch.where(
                                trained.reshape(
                                    (trained.shape[0],)
                                    + (1,) * (value.ndim - 1)
                                ),
                                value,
                                frozen.policy_raw[name],
                            )
                            for name, value in learned.policy_raw.items()
                        }
                    ),
                    torch.where(
                        trained.reshape(shape), learned.value, frozen.value
                    ),
                )
        if not isinstance(evaluation, ModelEvaluation):
            raise PokerIterationError("rollout model returned the wrong contract")
        return MappingProxyType(
            {
                **{
                    name: value.float()
                    for name, value in evaluation.policy_raw.items()
                },
                "value": evaluation.value.float(),
            }
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
        hooks = SequentialPokerHooks(
            self.geometry,
            registry,
            queues,
            seed=self.seed,
            iteration=iteration,
            mode=self.mode,
            policy=self.policy,
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
            if hooks.last_topology is None:
                raise PokerIterationError(
                    "poker transition omitted atomic topology"
                )
            failed, produced = topology_publisher.wait_deferred_feedback(
                hooks.last_topology,
                input_count=len(input_names),
            )
            hooks.last_topology = None
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
        topology, bindings, leaf_count = (
            topology_publisher.seal_deferred_unary(
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
        )
        topology_publisher.assert_drained()
        plan = build_layer_plan(
            self.mode,
            owners.plan.partitions,
            (topology,),
            (bindings,),
            p_train=(
                self.geometry.players
                if self.mode.training_regime is TrainingRegime.SELF_PLAY
                else 1
            ),
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
        algebra = self._target_algebra()
        binding = (
            SampledShowdownBinding(
                self.geometry,
                owners.slab("LayerWork", "all"),
                n_replica=1,
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
                n_replica=1,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                n_samples=self.completion_samples,
                utility=self.utility,
            )
        )
        remaining_leaves = leaf_count
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
        for output, name in enumerate(plan.output_names):
            work = owners.slab("LayerWork", name)
            missing = plan.arena_rows[0][output] - work.tail(0)
            if missing < 0:
                raise PokerIterationError("completion exceeded the target arena plan")
            if missing:
                padding = PublicationTransaction.begin(
                    (DestinationRequest(work, 0, missing, name, leaf_count),),
                    phase="poker_target_arena_padding",
                )
                padding.fill_defaults(0)
                padding.publish()
        owners.release_after(ExecutionPhase.COMPLETION)

        owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
        sealed = {
            name: SealedSequentialInputs.seal(
                owners.slab("FamilyWork", name),
                owners.slab("LearnerIndex", name),
                input_type=name,
            )
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
        stores = {
            name: AlignedSequentialStore(
                owners, name, sealed_inputs=sealed[name]
            )
            for name in input_names
            if sealed[name].rows
        }
        allocation_inputs = self.allocation_inputs
        if allocation_inputs is None and self.schedule.gradient_steps is not None:
            names = tuple(stores)
            preflop = getattr(self.model, "preflop", None)
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
                generator=self.generator,
                reproducibility=self.reproducibility,
            )
            if allocation_inputs is None
            else SequentialBatchSampler.from_costs(
                stores,
                self.schedule,
                generator=self.generator,
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
            budget = owners.plan.admission(
                ExecutionPhase.LEARNER
            ).bounded_workspace_bytes
            for name, store in stores.items():
                columns = store.sealed_inputs.columns
                with torch.no_grad():
                    evaluation = self.reference_operation(
                        name,
                        columns["model_input"],
                        columns.get(
                            "sampled_private_state", columns["owner"][:, None]
                        ),
                        columns["owner"],
                    )
                batch_calls = max(
                    1,
                    (store.rows + self.schedule.batch_size - 1)
                    // self.schedule.batch_size,
                )
                build_bytes = sum(
                    value.numel() * value.element_size()
                    for value in evaluation.policy_raw.values()
                )
                candidate = ReferenceOutputCache.admit(
                    evaluation.policy_raw,
                    byte_budget=budget,
                    resident_bytes=resident,
                    build_bytes=build_bytes,
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
            self.loss,
            self.model_operation,
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
        preflop = getattr(self.model, "preflop", None)
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
        result = SequentialSampledIteration(
            model=self.model,
            learner=learner,
            optimizer=self.optimizer,
            run_state=self.run_state,
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
            player_nodes=topology.node_count,
            leaves=leaf_count,
            target_layers=plan.layer_count,
        )

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
        required_model_workspace = 4 * (
            geometry.direct_model_input_width
            + geometry.private_hands * (5 + 3 * config.model.m_raise)
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
    ) -> "Phase5TrainingSession":
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
    ) -> "Phase5TrainingSession":
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
    "SequentialSampledPokerIteration",
    "SequentialPokerIteration",
]
