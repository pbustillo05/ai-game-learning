"""Semantic NLHE operations consumed by the generic iteration engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch
from torch import Tensor, nn

from pbg.core.capacity import StoragePlan
from pbg.core.contracts import (
    ModeSchema,
    PrivateEstimator,
    TrainingRegime,
    validate_replica_group,
)
from pbg.core.iteration import (
    CompletedLane,
    CompletionContext,
    GameCompletion,
    GameRollout,
    RolloutOperationRequest,
    RootDestination,
    RootPayload,
    RootRequest,
)
from pbg.core.learner import (
    ModelEvaluation,
    ModelOperation,
    PolicyOperation,
    compact_best_response_evaluation,
)
from pbg.core.rollout import DrawPurpose, logical_rng_keys
from pbg.core.storage import DeviceFailure
from pbg.core.targets import ConversionMap, TargetAlgebra
from pbg.games.poker.model.capacity import (
    max_policy_partition_sequence_tokens,
)
from pbg.games.poker.queued.branching import (
    BranchedPosteriorExhaustion,
    BranchedSampledPokerHooks,
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
from pbg.games.poker.queued.rb_sampling import (
    CollisionConditionedProfileSampler,
)
from pbg.games.poker.queued.replica import PokerReplicaInference


class PokerOperationsError(RuntimeError):
    """A semantic poker operation violated its closed contract."""


def balanced_response_roles(
    logical_roots: Tensor,
    *,
    players: int,
    seed: int,
    iteration: int,
    retry_epoch: int,
    lane: int,
) -> Tensor:
    """Assign maximally even, logically randomized BR seats.

    A logical-root interval receives a single RNG-selected seat permutation.
    Cycling through that permutation gives every seat ``floor(N/P)`` roots and
    gives the remainder to distinct seats without replacement. Role identity
    follows logical roots and is invariant to physical row order.
    """

    if logical_roots.dtype != torch.int64 or logical_roots.ndim != 1:
        raise TypeError("logical roots must be a one-dimensional int64 tensor")
    if logical_roots.numel() == 0:
        raise ValueError("logical roots must be nonempty")
    if isinstance(players, bool) or not isinstance(players, int) or players < 2:
        raise ValueError("players must be an integer of at least two")
    base = logical_roots.amin()
    ordinal = logical_roots - base
    if int(ordinal.amax()) + 1 != logical_roots.numel() or int(
        torch.unique(logical_roots).numel()
    ) != logical_roots.numel():
        raise ValueError("logical roots must form one contiguous interval")
    seats = torch.arange(players, dtype=torch.int64, device=logical_roots.device)
    keys = logical_rng_keys(
        seed=seed,
        iteration=iteration,
        root=base.expand(players),
        path=seats,
        draw=0,
        purpose=DrawPurpose.ROLE_SELECTION,
        lane=lane,
        retry_epoch=retry_epoch,
    )
    permutation = torch.argsort(keys, stable=True)
    return permutation[ordinal.remainder(players)].to(torch.uint8)


class PokerOperations:
    """One immutable-style NLHE bundle for all rollout and estimator modes."""

    error_type = PokerOperationsError

    def __init__(
        self,
        *,
        geometry: NLHEGeometry,
        plan: StoragePlan,
        modes: Sequence[ModeSchema],
        models: Sequence[nn.Module],
        rollout_operations: Sequence[ModelOperation],
        frozen_operations: Sequence[ModelOperation | None],
        seed: int,
        precision: str,
        utility: str,
        completion_samples: int,
        response_cutoff: bool = False,
        posterior_max_attempts: int = 256,
        inference: PokerReplicaInference | None = None,
        frozen_inference: PokerReplicaInference | None = None,
    ) -> None:
        count = plan.manifest.n_replica
        self.geometry = geometry
        self.plan = plan
        self.modes = tuple(modes)
        self.models = tuple(models)
        self.rollout_operations = tuple(rollout_operations)
        self.frozen_operations = tuple(frozen_operations)
        self.seed = int(seed)
        self.precision = precision
        self.utility = utility
        self.completion_samples = int(completion_samples)
        self.response_cutoff = response_cutoff
        self.posterior_max_attempts = int(posterior_max_attempts)
        self.inference = inference
        self.frozen_inference = frozen_inference
        self.policy = PokerPolicyOperator()
        if any(
            len(values) != count
            for values in (
                self.modes,
                self.models,
                self.rollout_operations,
                self.frozen_operations,
            )
        ):
            raise ValueError("poker operation lanes are not aligned")
        validate_replica_group(self.modes)
        if plan.mode != self.modes[0] or any(
            not mode.physically_compatible_with(plan.mode)
            for mode in self.modes
        ):
            raise ValueError("poker operation modes differ from storage")
        if plan.manifest.geometry.actor_count != geometry.players:
            raise ValueError("poker geometry differs from its manifest")
        if any(
            plan.layout.input_shapes[partition.name]
            != (geometry.model_input_width(partition.name),)
            for partition in plan.partitions.inputs
        ):
            raise ValueError(
                "family model-input layout differs from poker geometry"
            )
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("poker precision is invalid")
        if utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("poker utility is invalid")
        if type(response_cutoff) is not bool:
            raise TypeError("response_cutoff must be bool")
        if response_cutoff and (
            plan.mode.training_regime is not TrainingRegime.BEST_RESPONSE
            or plan.mode.private_estimator
            is not PrivateEstimator.RAO_BLACKWELL
            or plan.mode.rollout_mode.is_branched
        ):
            raise ValueError("response cutoff requires sequential BR/RB")
        if (
            plan.mode.private_estimator is PrivateEstimator.RAO_BLACKWELL
            and completion_samples <= 0
        ):
            raise ValueError("RB poker requires positive completion samples")
        if posterior_max_attempts <= 0:
            raise ValueError("posterior_max_attempts must be positive")
        if (
            any(
                mode.training_regime is TrainingRegime.BEST_RESPONSE
                for mode in self.modes
            )
            and frozen_inference is None
            and any(value is None for value in self.frozen_operations)
        ):
            raise ValueError("best response requires frozen model operations")
        if inference is not None and (
            inference.modules.n_replica != count
            or inference.modules.device != plan.device
        ):
            raise ValueError("replica inference differs from the storage plan")
        if frozen_inference is not None and (
            frozen_inference.modules.n_replica != count
            or frozen_inference.modules.device != plan.device
        ):
            raise ValueError(
                "frozen replica inference differs from the storage plan"
            )

    def _direct_evaluation(
        self,
        lane: int,
        forward: PokerForwardInput,
    ) -> ModelEvaluation:
        learned = self.rollout_operations[lane]
        mode = self.modes[lane]
        if mode.training_regime is TrainingRegime.SELF_PLAY:
            result = learned(
                forward.input_type,
                forward.model_input,
                forward.sampled_private_state,
                forward.owner,
            )
        else:
            frozen = self.frozen_operations[lane]
            if frozen is None or forward.role_selector is None:
                raise PokerOperationsError(
                    "best-response forward omitted its frozen role"
                )
            result = compact_best_response_evaluation(
                learned,
                frozen,
                input_type=forward.input_type,
                model_input=forward.model_input,
                sampled_private_state=forward.sampled_private_state,
                owner=forward.owner,
                role_selector=forward.role_selector,
            )
        if not isinstance(result, ModelEvaluation):
            raise PokerOperationsError(
                "rollout model returned the wrong contract"
            )
        return result

    def _replica_evaluation(
        self,
        lane: int,
        forward: PokerForwardInput,
        lanes: Tensor,
    ) -> ModelEvaluation:
        if self.inference is None:
            raise PokerOperationsError("replica inference is unavailable")
        lane_count = int(lanes.numel())
        if (
            lanes.dtype != torch.int64
            or lanes.device != self.plan.device
            or lane_count <= 0
            or forward.owner.numel() % lane_count
        ):
            raise PokerOperationsError("replica model lanes are invalid")
        rows = forward.owner.numel() // lane_count
        model_input = forward.model_input.reshape(
            lane_count, rows, *forward.model_input.shape[1:]
        )
        private = forward.sampled_private_state.reshape(
            lane_count, rows, *forward.sampled_private_state.shape[1:]
        )
        owner = forward.owner.reshape(lane_count, rows)
        role = (
            None
            if forward.role_selector is None
            else forward.role_selector.reshape(lane_count, rows)
        )

        def flatten(value: Tensor) -> Tensor:
            return value.flatten(0, 1)

        def evaluate(
            operation: PokerReplicaInference,
            selected: Tensor,
        ) -> tuple[ModelEvaluation, Tensor, Tensor] | None:
            count = selected.sum(1, dtype=torch.int64)
            width = int(count.max().item())
            if width == 0:
                return None
            ordinal = torch.arange(
                rows, dtype=torch.int64, device=self.plan.device
            )[None].expand(lane_count, -1)
            packed_row = torch.argsort(
                torch.where(selected, ordinal, rows + ordinal),
                dim=1,
                stable=True,
            )[:, :width]
            valid = (
                torch.arange(
                    width, dtype=torch.int64, device=self.plan.device
                )[None]
                < count[:, None]
            )
            packed_row = torch.where(
                valid, packed_row, torch.zeros_like(packed_row)
            )
            lane_row = torch.arange(
                lane_count, dtype=torch.int64, device=self.plan.device
            )[:, None]
            result = operation(
                forward.input_type,
                model_input[lane_row, packed_row],
                private[lane_row, packed_row],
                owner[lane_row, packed_row],
                active_lanes=lanes,
            )
            return result, packed_row, valid

        mode = self.modes[lane]
        if mode.training_regime is TrainingRegime.SELF_PLAY:
            result = self.inference(
                forward.input_type,
                model_input,
                private,
                owner,
                active_lanes=lanes,
            )
            return ModelEvaluation(
                MappingProxyType(
                    {
                        name: flatten(value)
                        for name, value in result.policy_raw.items()
                    }
                ),
                flatten(result.value),
                MappingProxyType(
                    {
                        name: flatten(value)
                        for name, value in result.public_heads.items()
                    }
                ),
            )
        if self.frozen_inference is None or role is None:
            raise PokerOperationsError(
                "replica best response omitted its frozen role"
            )
        trained = owner.eq(role)
        learned = evaluate(self.inference, trained)
        frozen = evaluate(self.frozen_inference, ~trained)
        prototype = learned[0] if learned is not None else frozen[0]
        if prototype is None:
            raise PokerOperationsError("best-response model batch is empty")

        def restore(
            learned_value: Tensor | None,
            frozen_value: Tensor | None,
        ) -> Tensor:
            source = learned_value if learned_value is not None else frozen_value
            assert source is not None
            output = source.new_empty(
                (lane_count, rows, *source.shape[2:])
            )
            for value, packed in (
                (learned_value, learned),
                (frozen_value, frozen),
            ):
                if value is None or packed is None:
                    continue
                packed_row, valid = packed[1:]
                lane_row = torch.arange(
                    lane_count,
                    dtype=torch.int64,
                    device=self.plan.device,
                )[:, None].expand_as(packed_row)
                output[lane_row[valid], packed_row[valid]] = value[valid]
            return flatten(output)

        return ModelEvaluation(
            MappingProxyType(
                {
                    name: restore(
                        None
                        if learned is None
                        else learned[0].policy_raw[name],
                        None
                        if frozen is None
                        else frozen[0].policy_raw[name],
                    )
                    for name in prototype.policy_raw
                }
            ),
            restore(
                None if learned is None else learned[0].value,
                None if frozen is None else frozen[0].value,
            ),
            MappingProxyType(
                {
                    name: restore(
                        None
                        if learned is None
                        else learned[0].public_heads[name],
                        None
                        if frozen is None
                        else frozen[0].public_heads[name],
                    )
                    for name in prototype.public_heads
                }
            ),
        )

    @torch.no_grad()
    def _model(
        self,
        lane: int,
        forward: PokerForwardInput,
        _lanes: Tensor,
    ) -> Mapping[str, Tensor]:
        autocast_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise PokerOperationsError(
                "selected CUDA device does not support BF16"
            )
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            evaluation = (
                self._direct_evaluation(lane, forward)
                if self.inference is None
                else self._replica_evaluation(lane, forward, _lanes)
            )
        return MappingProxyType(
            {
                **{
                    name: value.float()
                    for name, value in evaluation.policy_raw.items()
                },
                "value": evaluation.value.float(),
                **{
                    name: value.float()
                    for name, value in evaluation.public_heads.items()
                },
            }
        )

    def root_destination(self, plan: StoragePlan) -> RootDestination:
        if plan != self.plan:
            raise PokerOperationsError(
                "root destination requested for a different storage plan"
            )
        return RootDestination("pre")

    def root_payload(self, request: RootRequest) -> RootPayload:
        machine = PokerStateMachine(self.geometry)
        state = machine.initial_roots(
            request.rows,
            seed=self.seed,
            iteration=request.iteration,
            retry_epoch=request.retry_epoch,
            logical_base=request.logical_base,
            lane=request.lane,
            device=request.plan.device,
        )
        logical = torch.arange(
            request.logical_base,
            request.logical_base + request.rows,
            dtype=torch.int64,
            device=request.plan.device,
        )
        values: dict[str, Tensor] = {
            "logical_rng_key": torch.stack(
                (logical, torch.zeros_like(logical)), dim=1
            ),
            "true_profile": state.combo_ids,
            "model_input": machine.direct_model_input(
                request.destination.input_type, state
            ),
            "owner": state.to_act,
        }
        if (
            request.plan.mode.private_estimator
            is PrivateEstimator.SAMPLED
        ):
            values["sampled_private_state"] = state.combo_ids
        if (
            request.plan.mode.training_regime
            is TrainingRegime.BEST_RESPONSE
        ):
            values["role_selector"] = balanced_response_roles(
                logical,
                players=self.geometry.players,
                seed=self.seed,
                iteration=request.iteration,
                retry_epoch=request.retry_epoch,
                lane=request.lane,
            )
        frontier_cost = None
        if request.plan.mode.rollout_mode.is_branched:
            values.update(
                {
                    "path_mass": torch.ones(
                        request.rows,
                        dtype=torch.float32,
                        device=request.plan.device,
                    ),
                    "log_player_ancestor_sum": torch.zeros(
                        request.rows,
                        dtype=torch.float32,
                        device=request.plan.device,
                    ),
                    "log_ancestor_product": torch.zeros(
                        request.rows,
                        dtype=torch.float32,
                        device=request.plan.device,
                    ),
                    "branch_beta": torch.ones(
                        request.rows,
                        dtype=torch.float32,
                        device=request.plan.device,
                    ),
                }
            )
            if request.safeguard is not None:
                pack = request.plan.manifest.family_pack_cap
                role = values.get("role_selector")
                frontier_cost = torch.cat(
                    tuple(
                        self._model(
                            request.lane,
                            PokerForwardInput(
                                "pre",
                                values["model_input"][begin:end],
                                state.combo_ids[begin:end],
                                state.to_act[begin:end],
                                None if role is None else role[begin:end],
                            ),
                            torch.tensor(
                                [request.lane],
                                dtype=torch.int64,
                                device=request.plan.device,
                            ),
                        )["cost"]
                        for begin in range(0, request.rows, pack)
                        for end in (min(begin + pack, request.rows),)
                    )
                )
                values["frontier_cost_contribution"] = frontier_cost
        return RootPayload(MappingProxyType(values), frontier_cost)

    def rollout(self, request: RolloutOperationRequest) -> GameRollout:
        count = request.owners.plan.manifest.n_replica
        if not request.owners.plan.mode.rollout_mode.is_branched:
            hooks = SequentialPokerHooks(
                self.geometry,
                request.registry,
                request.queues,
                seed=self.seed,
                iteration=request.iteration,
                mode=self.modes[0],
                policy=self.policy,
                response_cutoff=self.response_cutoff,
                topology_stagers=request.stagers,
            )
            if self.inference is None:
                models = tuple(
                    (
                        lambda forward, lanes, selected=lane: self._model(
                            selected, forward, lanes
                        )
                    )
                    for lane in range(count)
                )
            else:
                cohort_model = (
                    lambda forward, lanes: self._model(
                        0, forward, lanes
                    )
                )
                models = tuple(cohort_model for _lane in range(count))
            return GameRollout(
                tuple(hooks for _lane in range(count)),
                models,
                tuple(None for _lane in range(count)),
            )
        hooks_by_lane: list[BranchedSampledPokerHooks] = []
        max_prefix: list[int] = []
        for lane in range(count):
            controller = request.controllers[lane]
            if controller is None:
                raise PokerOperationsError(
                    "branched rollout omitted its controller"
                )
            mode = self.modes[lane]
            posterior = (
                None
                if mode.private_estimator is PrivateEstimator.SAMPLED
                else CollisionConditionedProfileSampler(
                    self.geometry,
                    max_attempts=self.posterior_max_attempts,
                )
            )
            hooks = BranchedSampledPokerHooks(
                self.geometry,
                request.registry,
                request.queues,
                seed=self.seed,
                iteration=request.iteration,
                mode=mode,
                controller=controller,
                cost_safeguard=request.safeguards[lane],
                child_model=(
                    lambda forward, lanes, selected=lane: self._model(
                        selected, forward, lanes
                    )
                ),
                topology_stager=request.stagers[lane],
                policy=self.policy,
                posterior_sampler=posterior,
            )
            hooks.realized_tree_size.fill_(request.roots[lane])
            player_degree = controller.max_degree(owner_is_player=True)
            chance_degree = controller.max_degree(owner_is_player=False)
            topology_rows_per_root = player_degree * sum(
                chance_degree**generation for generation in range(4)
            )
            if (
                topology_rows_per_root
                > request.owners.plan.manifest.topology_packet_cap
            ):
                raise ValueError(
                    "topology packet cannot cover one maximal poker root"
                )
            hooks_by_lane.append(hooks)
            max_prefix.append(
                min(
                    request.owners.plan.manifest.family_pack_cap,
                    request.owners.plan.manifest.topology_packet_cap
                    // topology_rows_per_root,
                )
            )

        def begin_refill(lane: int, child_ordinal: int) -> None:
            hooks_by_lane[lane].begin_root_refill(child_ordinal)

        return GameRollout(
            tuple(hooks_by_lane),
            tuple(
                (
                    lambda forward, lanes, selected=lane: self._model(
                        selected, forward, lanes
                    )
                )
                for lane in range(count)
            ),
            tuple(max_prefix),
            begin_refill=begin_refill,
            routing_traffic=(
                lambda lane: hooks_by_lane[lane].routing_traffic_bytes
            ),
        )

    def rollout_failure(
        self, lane: int, failure: DeviceFailure | None
    ) -> Exception:
        del lane
        if failure is not None and failure.failure_code == "rejection_exhausted":
            return BranchedPosteriorExhaustion(failure)
        return PokerOperationsError("poker rollout failed terminally")

    def target_algebra(self, plan: StoragePlan) -> TargetAlgebra:
        if plan.mode.private_estimator is PrivateEstimator.SAMPLED:
            return TargetAlgebra(plan.partitions, ())
        cards = PokerStateMachine(self.geometry).combo_cards(
            torch.arange(self.geometry.private_hands, dtype=torch.int32)
        )
        ranks = self.geometry.rank_count
        rank = cards.long() // 4 - (13 - ranks)
        high, low = rank.max(1).values, rank.min(1).values
        hp, lp = ranks - 1 - high, ranks - 1 - low
        suited_or_pair = rank[:, 0].eq(rank[:, 1]) | cards[
            :, 0
        ].remainder(4).eq(cards[:, 1].remainder(4))
        classes = torch.where(
            suited_or_pair, hp * ranks + lp, lp * ranks + hp
        )
        return TargetAlgebra(
            plan.partitions,
            (
                ConversionMap(
                    "post",
                    "pre",
                    tuple(int(value) for value in classes.tolist()),
                ),
            ),
        )

    def completion(self, context: CompletionContext) -> GameCompletion:
        owners = context.owners
        count = owners.plan.manifest.n_replica
        if owners.plan.mode.private_estimator is PrivateEstimator.SAMPLED:
            binding = SampledShowdownBinding(
                self.geometry,
                owners.slab("LayerWork", "all"),
                n_replica=count,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                utility=self.utility,
            )
            return GameCompletion(binding.leaf_to_layer_work)
        binding = RBCompletionBinding(
            self.geometry,
            {
                name: owners.slab("LayerWork", name)
                for name in ("pre", "post")
            },
            context.algebra,
            mode=owners.plan.mode,
            n_replica=count,
            pack_cap=owners.plan.manifest.completion_pack_cap,
            n_samples=self.completion_samples,
            utility=self.utility,
        )

        def observations() -> Mapping[str, object]:
            return MappingProxyType(
                {
                    "posterior_effective_hands": tuple(
                        float(value)
                        for value in binding.effective_hands.detach()
                        .cpu()
                        .tolist()
                    )
                }
            )

        return GameCompletion(binding.leaf_to_layer_work, observations)

    def learner_policy(self, lane: int) -> PolicyOperation:
        del lane
        return PokerLearnerPolicy(self.geometry, self.policy)

    def model_costs(self, lane: int) -> Mapping[str, int] | None:
        preflop = getattr(self.models[lane], "preflop", None)
        model_config = getattr(preflop, "cfg", None)
        if model_config is None:
            return None
        return MappingProxyType(
            max_policy_partition_sequence_tokens(
                n_players=self.geometry.players,
                stack_bb=self.geometry.stack_bb,
                rank_count=self.geometry.rank_count,
                k_act=model_config.k_act,
            )
        )

    def logical_rollout_limit(self, plan: StoragePlan) -> int:
        if plan.mode.rollout_mode.is_branched:
            return plan.manifest.n_replica * plan.manifest.topology_cap
        return (
            plan.manifest.n_replica
            * plan.manifest.root_count
            * self.geometry.player_node_bound
        )

    def observability(self, lane: CompletedLane) -> Mapping[str, object]:
        return MappingProxyType(
            {
                **topology_diagnostics(
                    lane.topology,
                    lane.bindings,
                    lane.partitions,
                ),
                "model_timing_by_family": dict(lane.model_timing),
            }
        )


__all__ = ["PokerOperations", "PokerOperationsError"]
