"""Sampled and Rao--Blackwell branched NLHE through one queued engine."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch
from torch import Tensor, nn

from pbg.core.branching import (
    BranchController,
    BranchDecision,
    CostSafeguardController,
    child_path_factors,
    decision_log_path_mass,
)
from pbg.core.capacity import (
    BRANCHED_TOPOLOGY_PACKET_ROW_BYTES,
    GeneratedOwners,
)
from pbg.core.contracts import (
    ModeSchema,
    PrivateEstimator,
    TrainingRegime,
    validate_replica_group,
)
from pbg.core.iteration import (
    AcceptedRollout,
    CompletionLoop,
    IterationEngine,
    LearnerPreflightOptimizer,
    RolloutLoop,
    SealedTopology,
)
from pbg.core.learner import (
    BranchedBatchSampler,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    ModelOperation,
    RaggedBranchedStore,
    ReferenceOutputCache,
    SealedBranchedInputs,
    allocate_statistical_counts,
    build_reference_output_cache,
    compact_best_response_evaluation,
)
from pbg.core.persistence import RunState
from pbg.core.replica import (
    DistributedSUM,
    ReplicaOptimizerGroup,
)
from pbg.core.rollout import (
    AsyncTopologyPublisher,
    BoundedChanceClosure,
    CommitCoverage,
    CommitPlan,
    CompletionFIFO,
    DeviceCommitFailure,
    DrawPurpose,
    FamilyQueueState,
    FamilyScheduler,
    RolloutOwnerRegistry,
    SampleRequest,
    SampleWorkspace,
    SemanticFamilyFIFO,
    TopologyPublication,
    TransitionRequest,
    child_path_keys,
    logical_rng_keys,
    stable_route,
)
from pbg.core.storage import (
    DeviceDestinationRequest,
    DeviceFailure,
    MonotoneSlab,
    PublicationTransaction,
)
from pbg.core.targets import LayeredTargetExecutor, TargetAlgebra
from pbg.core.topology import (
    TopologyExpansionRecord,
    TopologyPacketStager,
)
from pbg.core.training import (
    LearnerIteration,
    RecoverableOverflow,
)
from pbg.games.poker.queued.hooks import (
    FOLD,
    PAD_CARD,
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


class BranchedPokerError(RuntimeError):
    """The sampled branched poker composition violated its contract."""


class BranchedPosteriorExhaustion(BranchedPokerError):
    """A bounded posterior sampler exhausted before any publication."""

    def __init__(self, failure: DeviceFailure) -> None:
        super().__init__(
            "branched RB posterior rejection exhausted at logical "
            f"address {failure.logical_address} after "
            f"{failure.sampler_attempt_count} attempts"
        )
        self.failure = failure


def sampled_branched_chance_frontier_row_bytes(
    geometry: NLHEGeometry,
) -> int:
    """Conservative concurrent device bytes for one chance-frontier row."""

    from pbg.games.poker.queued.folded import (
        folded_statistics_workspace_row_bytes,
    )

    players = geometry.players
    hands = geometry.private_hands
    history = (
        geometry.postflop_base_model_input_width
        - 8
        - 10 * players
        - players * hands
    ) // 10
    return (
        512
        + 8 * geometry.postflop_model_input_width
        + 16 * players * hands
        + 128 * history
        + 64 * geometry.deck_cards
        + 128 * players
        + 3 * BRANCHED_TOPOLOGY_PACKET_ROW_BYTES
        + folded_statistics_workspace_row_bytes(geometry)
    )


def branched_rb_chance_frontier_row_bytes(
    geometry: NLHEGeometry,
    *,
    max_attempts: int = 256,
) -> int:
    """Concurrent RB frontier bytes including bounded rejection telemetry."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    return (
        sampled_branched_chance_frontier_row_bytes(geometry)
        + 4 * geometry.players * geometry.private_hands
        + max_attempts
        * (
            4 * geometry.players
            + 2 * geometry.deck_cards
            + geometry.players
        )
        + 32
    )


def sampled_branched_topology_packet_row_bytes(
    _geometry: NLHEGeometry,
) -> int:
    """Device bytes for one compact topology node/edge packet row."""

    return BRANCHED_TOPOLOGY_PACKET_ROW_BYTES


def action_conditioned_log_reach(
    log_reach: Tensor,
    selected_log_policy: Tensor,
    actor: Tensor,
) -> Tensor:
    """Condition the acting seat's exact live/folded range on one action."""

    if (
        log_reach.ndim != 3
        or selected_log_policy.shape
        != (log_reach.shape[0], log_reach.shape[2])
        or actor.shape != log_reach.shape[:1]
        or actor.dtype not in (torch.int32, torch.int64)
        or selected_log_policy.device != log_reach.device
        or actor.device != log_reach.device
    ):
        raise BranchedPokerError(
            "action conditioning requires [B,P,H], [B,H], and [B]"
        )
    actor64 = actor.to(torch.int64)
    torch._assert_async(
        actor64.ge(0).logical_and(
            actor64.lt(log_reach.shape[1])
        ).all(),
        "action-conditioning actor is outside the seat axis",
    )
    selected = log_reach.gather(
        1,
        actor64[:, None, None].expand(
            -1, 1, log_reach.shape[2]
        ),
    ).squeeze(1)
    return log_reach.scatter(
        1,
        actor64[:, None, None].expand(
            -1, 1, log_reach.shape[2]
        ),
        (selected + selected_log_policy.float())[:, None],
    )


@dataclass(frozen=True)
class BranchedPokerResult:
    learner: object
    player_nodes: int
    leaves: int
    target_layers: int
    executed_edges: int
    root_executed_degree: tuple[int, ...]
    root_target_degree: tuple[int, ...]
    duplicate_child_labels: int
    sampled_tree_nodes: int
    pre_refill_nodes: int | None = None
    refill_count: int = 0
    posterior_effective_hands: float | None = None
    telemetry: BranchedBenchmarkTelemetry | None = None
    observability: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class BranchedBenchmarkTelemetry:
    """Matched-work counters emitted only for explicit benchmark runs."""

    rollout_seconds: float
    routing_traffic_bytes: int
    host_synchronizations: int
    child_reservation_commit_seconds: float
    target_propagation_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    model_inference_seconds: float = 0.0
    posterior_and_action_sampling_seconds: float = 0.0
    transition_seconds: float = 0.0
    target_kernel_cuda_seconds: float = 0.0
    rollout_peak_allocated_bytes: int = 0
    completion_peak_allocated_bytes: int = 0
    align_peak_allocated_bytes: int = 0
    targets_peak_allocated_bytes: int = 0
    model_stage_peak_allocated_bytes: int = 0
    sampling_stage_peak_allocated_bytes: int = 0
    transition_stage_peak_allocated_bytes: int = 0
    target_setup_seconds: float = 0.0


class BranchedSampledPokerHooks(SequentialPokerHooks):
    """Sample duplicate-preserving children under the schema's private law."""

    def __init__(
        self,
        geometry: NLHEGeometry,
        owners: RolloutOwnerRegistry,
        queues: Mapping[str, SemanticFamilyFIFO],
        *,
        seed: int,
        iteration: int,
        mode: ModeSchema,
        controller: BranchController,
        cost_safeguard: CostSafeguardController | None,
        child_model: Callable[[PokerForwardInput, Tensor], Mapping[str, Tensor]],
        topology_stager: TopologyPacketStager,
        policy: PokerPolicyOperator,
        posterior_sampler: CollisionConditionedProfileSampler | None = None,
    ) -> None:
        if not mode.rollout_mode.is_branched:
            raise ValueError("branched hooks require branched mode")
        # Initialize the shared helper vocabulary without its sequential guard.
        self.mode = mode
        self.geometry = geometry
        self.machine = PokerStateMachine(geometry)
        self.owners = owners
        self.queues = MappingProxyType(dict(queues))
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.policy = policy
        self.posterior_sampler = posterior_sampler
        if (
            mode.private_estimator is PrivateEstimator.RAO_BLACKWELL
            and posterior_sampler is None
        ):
            raise ValueError("branched RB hooks require the posterior sampler")
        self.response_cutoff = False
        self.input_index = {name: index for index, name in enumerate(self.queues)}
        self._route_family_values = (-1, *range(len(self.queues)))
        self._route_families = torch.tensor(
            self._route_family_values,
            dtype=torch.int32,
            device=owners.device,
        )
        self.controller = controller
        if (
            owners.family_pack_cap
            * controller.max_degree(
                owner_is_player=True,
            )
            > owners.chance_frontier_cap
        ):
            raise ValueError(
                "chance frontier does not cover one maximal player pack"
            )
        self.cost_safeguard = cost_safeguard
        self.child_model = child_model
        self.topology_stager = topology_stager
        self.realized_tree_size = torch.zeros(
            (), dtype=torch.float32, device=owners.device
        )
        self._frontier_cost: Tensor | None = None
        self._safe_tree_triggered = False
        self.last_topology: TopologyPublication | None = None
        self._topology_sequence = 0
        self._refill = False
        self._refill_child_ordinal = 0
        self.routing_traffic_bytes = 0
        self._posterior_failed_slots = torch.zeros(
            topology_stager.inflight_cap,
            dtype=torch.bool,
            device=owners.device,
        )
        self._posterior_address_slots = torch.full(
            (topology_stager.inflight_cap,),
            torch.iinfo(torch.int64).max,
            dtype=torch.int64,
            device=owners.device,
        )
        self._posterior_attempt_slots = torch.zeros(
            topology_stager.inflight_cap,
            dtype=torch.int64,
            device=owners.device,
        )
        self._posterior_failed = self._posterior_failed_slots[0]
        self._posterior_address = self._posterior_address_slots[0]
        self._posterior_attempts = self._posterior_attempt_slots[0]

    def _observe_posterior(self) -> None:
        assert self.posterior_sampler is not None
        observation = self.posterior_sampler.last_observation
        assert observation is not None
        failed = observation.exhausted.any()
        self._posterior_failed |= failed
        self._posterior_address.copy_(
            torch.minimum(
                self._posterior_address,
                observation.minimum_failing_address,
            )
        )
        self._posterior_attempts.copy_(
            torch.maximum(
                self._posterior_attempts,
                observation.attempt_count.masked_fill(
                    ~observation.exhausted, 0
                ).amax(),
            )
        )

    def begin_root_refill(self, child_ordinal: int) -> None:
        if self._refill or child_ordinal <= 0:
            raise BranchedPokerError("invalid or overlapping root refill")
        self._refill = True
        self._refill_child_ordinal = int(child_ordinal)

    def sample(self, request: SampleRequest) -> SampleWorkspace:
        batch = request.batch
        # Each in-flight commit owns one preallocated sampler-status slot.
        # The slot is not reused until the four-pack publication group drains.
        slot = self._topology_sequence % self.topology_stager.inflight_cap
        self._posterior_failed = self._posterior_failed_slots[slot]
        self._posterior_address = self._posterior_address_slots[slot]
        self._posterior_attempts = self._posterior_attempt_slots[slot]
        self._posterior_failed.zero_()
        self._posterior_address.fill_(torch.iinfo(torch.int64).max)
        self._posterior_attempts.zero_()
        if batch.lane_count != 1:
            raise BranchedPokerError(
                "branched sampling requires one active lane"
            )
        lane = int(batch.lanes[0])
        rows = batch.rows
        root = batch.fields["logical_rng_key"][0, :, 0]
        path = batch.fields["logical_rng_key"][0, :, 1]
        values = request.model.values
        if "trace" not in values or "cost" not in values:
            raise BranchedPokerError("branched rollout omitted T/C heads")
        missing_frontier_cost = values["cost"].detach().float().masked_fill(
            batch.fields["frontier_cost_contribution"][0].gt(0), 0
        ).sum()
        frontier_cost = (
            self.realized_tree_size.to(values["cost"].dtype).expand_as(
                values["cost"]
            )
            + (
                torch.stack(
                    tuple(
                        queue.frontier_cost[lane]
                        for queue in self.queues.values()
                    )
                ).sum()
                + missing_frontier_cost
                if self.cost_safeguard is not None
                and self.cost_safeguard.active
                else 0
            )
            if self.cost_safeguard is not None
            else None
        )
        self._frontier_cost = frontier_cost
        self._safe_tree_triggered = (
            frontier_cost is not None
            and self.controller.safe_tree_size is not None
            and bool(
                frontier_cost.gt(self.controller.safe_tree_size).all()
            )
        )
        decision = self.controller.decide(
            schema=self.mode,
            root=root,
            path=path,
            owner_is_player=torch.ones(rows, dtype=torch.bool, device=root.device),
            iteration=self.iteration,
            seed=self.seed,
            retry_epoch=self.owners.retry_epoch()[lane],
            trace=values["trace"],
            cost=values["cost"],
            log_player_ancestor_sum=batch.fields["log_player_ancestor_sum"][0],
            log_ancestor_product=batch.fields["log_ancestor_product"][0],
            frontier_cost=frontier_cost,
            player_family=(
                "pre" if batch.input_type == "pre" else "post"
            ),
        )
        if self._refill:
            if rows != 1:
                raise BranchedPokerError("refill must replay exactly one root")
            torch._assert_async(path.eq(0).all(), "refill path must be the root")
            ones = torch.ones_like(decision.executed_degree)
            decision = BranchDecision(
                decision.beta,
                ones,
                ones,
                ones,
            )
        degree = decision.executed_degree.to(torch.int64)
        total_tensor = degree.sum()
        player_family = "pre" if batch.input_type == "pre" else "post"
        beta_max = self.controller.max_degree(
            owner_is_player=True,
            player_family=player_family,
        )
        source = torch.arange(
            rows, dtype=torch.int64, device=root.device
        ).repeat_interleave(beta_max)
        ordinal = torch.arange(
            beta_max, dtype=torch.int64, device=root.device
        ).repeat(rows)
        active = ordinal < degree.index_select(0, source)
        torch._assert_async(
            active.sum(dtype=torch.int64).eq(total_tensor),
            "fixed player expansion envelope lost a child",
        )
        child_ordinal = (
            torch.full_like(ordinal, self._refill_child_ordinal)
            if self._refill
            else ordinal + 1
        )
        child_path = child_path_keys(path.index_select(0, source), child_ordinal)
        expanded_input = self._forward_model_input(batch).index_select(0, source)
        parent_profile = batch.fields["true_profile"][0]
        private = parent_profile.index_select(0, source)
        if (
            self.mode.private_estimator is PrivateEstimator.RAO_BLACKWELL
            and not (self._safe_tree_triggered and not self._refill)
        ):
            log_reach, _history = self._resolved_model_context(batch)
            finite = torch.isfinite(log_reach)
            gauge = log_reach.masked_fill(~finite, -torch.inf).amax(
                -1, keepdim=True
            )
            range_weights = torch.exp(
                log_reach
                - gauge.masked_fill(~torch.isfinite(gauge), 0.0)
            ).masked_fill(~finite, 0.0)
            parent_state = self.machine.state_from_model_input(
                batch.input_type,
                self._forward_model_input(batch),
                parent_profile,
            )
            assert self.posterior_sampler is not None
            private = self.posterior_sampler.sample_children(
                range_weights=range_weights,
                board_cards=parent_state.board_cards,
                inherited_profile=parent_profile,
                source=source,
                # Posterior ordinals are zero-based across the complete
                # logical root degree, including later refill packets.
                # Only the original child zero inherits the parent profile.
                ordinal=child_ordinal - 1,
                active=active,
                root=root.index_select(0, source),
                path=child_path,
                seed=self.seed,
                iteration=self.iteration,
                retry_epoch=self.owners.retry_epoch()[lane],
            )
            self._observe_posterior()
        owner = batch.fields["owner"][0].index_select(0, source).to(torch.int64)
        state = self.machine.state_from_model_input(
            batch.input_type, expanded_input, private
        )
        raw = {
            name: value.index_select(0, source)
            for name, value in values.items()
            if name not in ("value", "trace", "cost")
        }
        actor_hand = self._private_index(
            batch.input_type,
            private.gather(1, owner[:, None]).squeeze(1),
        )
        action_key = logical_rng_keys(
            seed=self.seed,
            iteration=self.iteration,
            root=root.index_select(0, source),
            path=child_path,
            draw=0,
            purpose=DrawPurpose.ACTION,
            lane=lane,
            retry_epoch=self.owners.retry_epoch()[lane],
            active=active,
        )
        sampled = self.policy.sample_actor(
            raw,
            self.machine.legality(state),
            actor_hand,
            action_key,
        )
        return SampleWorkspace(
            MappingProxyType(
                {
                    "action": sampled.action,
                    "selected_log_policy": sampled.selected_log_policy,
                    "source": source,
                    "ordinal": ordinal,
                    "active": active,
                    "child_path": child_path,
                    "beta": decision.beta,
                    "nominal_degree": decision.nominal_degree,
                    "executed_degree": decision.executed_degree,
                    "target_degree": decision.target_degree,
                    "profile": private,
                }
            ),
            total_tensor.reshape(1),
        )

    def transition(self, request: TransitionRequest) -> CommitPlan:
        batch = request.batch
        if batch.lane_count != 1:
            raise BranchedPokerError("branched transition requires one active lane")
        lane = int(batch.lanes[0])
        rows = batch.rows
        device = self.owners.device
        sample = request.sample.values
        source = sample["source"].to(torch.int64)
        action_active = sample["active"].to(torch.bool)
        action_source = source
        action_total = int(action_source.numel())
        flat_input = self._forward_model_input(batch)
        parent_private = batch.fields["true_profile"][0]
        parent_state = self.machine.state_from_model_input(
            batch.input_type, flat_input, parent_private
        )
        state = replace(
            parent_state,
            **{
                name: value.index_select(0, source)
                for name, value in parent_state.__dict__.items()
                if isinstance(value, Tensor)
            },
        )
        private = sample["profile"]
        state = replace(state, combo_ids=private)
        action = sample["action"]
        resolved_reach, resolved_history = self._resolved_model_context(batch)
        log_reach = resolved_reach.index_select(0, source)
        history = replace(
            resolved_history,
            **{
                name: value.index_select(0, source)
                for name, value in resolved_history.__dict__.items()
            },
        )
        raw = MappingProxyType(
            {
                name: value.index_select(0, source)
                for name, value in request.model.values.items()
                if name not in ("value", "trace", "cost")
            }
        )
        selected_log = (
            self.policy.selected_log_pi(raw, self.machine.legality(state), action)
            .detach()
            .float()
        )
        selected_log_exact = self._exact_rows(selected_log, batch.input_type)
        actor = state.to_act.to(torch.int64)
        log_reach = action_conditioned_log_reach(
            log_reach,
            selected_log_exact,
            actor,
        )
        fold_log_reach = log_reach
        fold_parent_street = state.street
        history = self.machine.append_player_history(state, action, history)
        next_state = self.machine.step_player(state, action)
        root = batch.fields["logical_rng_key"][0, :, 0].index_select(0, action_source)
        child_path = sample["child_path"]
        retry = self.owners.retry_epoch()[lane]
        first_child_key = torch.stack((root, child_path), -1)
        initial_endpoint_source = action_source
        initial_endpoint_mass = (
            batch.fields["path_mass"][0].index_select(0, action_source)
            / sample["executed_degree"].index_select(0, action_source).float()
        )
        initial_log_sum, initial_log_product = child_path_factors(
            batch.fields["log_player_ancestor_sum"][0].index_select(0, action_source),
            batch.fields["log_ancestor_product"][0].index_select(0, action_source),
            sample["beta"].index_select(0, action_source),
            child_is_player=torch.zeros(action_total, dtype=torch.bool, device=device),
        )
        chance_capacity = self.owners.chance_frontier_cap
        if action_total > chance_capacity:
            raise BranchedPokerError(
                "player expansion exceeds the admitted chance frontier"
            )
        action_row = torch.arange(
            action_total, dtype=torch.int64, device=device
        )
        current_active = action_active
        current_key = first_child_key
        current_state = replace(
            next_state,
            **{
                name: value
                for name, value in next_state.__dict__.items()
                if isinstance(value, Tensor)
            },
        )
        current_private = private
        current_history = replace(
            history,
            **{
                name: value
                for name, value in history.__dict__.items()
            },
        )
        current_log_reach = log_reach
        current_frontier_cost = (
            None
            if self._frontier_cost is None
            else self._frontier_cost.index_select(0, action_source)
        )
        endpoint_source = initial_endpoint_source
        endpoint_action = action_row
        endpoint_mass = initial_endpoint_mass
        endpoint_log_sum = initial_log_sum
        endpoint_log_product = initial_log_product
        chance_records: list[TopologyExpansionRecord] = []
        current_beta = torch.ones(
            action_total, dtype=torch.float32, device=device
        )
        sampled_nodes = action_active.sum(dtype=torch.int64)
        frontier_demand = sampled_nodes.clone()
        frontier_overflow = torch.zeros(
            (), dtype=torch.bool, device=device
        )
        endpoint_record = torch.zeros(
            action_total, dtype=torch.int64, device=device
        )
        endpoint_record_edge = action_row
        for generation in BoundedChanceClosure(
            max_generations=3,
            generation_bound=self.owners.chance_generation_cap,
        ):
            current_rows = current_active.numel()
            generation_capacity = min(
                chance_capacity,
                current_rows
                * self.controller.max_degree(owner_is_player=False),
            )
            frontier_row = torch.arange(
                generation_capacity,
                dtype=torch.int64,
                device=device,
            )
            chance = self.machine.needs_chance(current_state) & current_active
            chance_decision = self.controller.decide(
                schema=self.mode,
                root=current_key[:, 0],
                path=current_key[:, 1],
                owner_is_player=torch.zeros_like(chance),
                iteration=self.iteration,
                seed=self.seed,
                retry_epoch=retry,
                trace=None,
                cost=None,
                log_player_ancestor_sum=endpoint_log_sum,
                log_ancestor_product=endpoint_log_product,
                frontier_cost=current_frontier_cost,
                active=chance,
            )
            ones = torch.ones_like(chance_decision.executed_degree)
            chance_decision = BranchDecision(
                torch.where(
                    chance,
                    chance_decision.beta,
                    torch.ones_like(chance_decision.beta),
                ),
                torch.where(
                    chance,
                    chance_decision.nominal_degree,
                    ones,
                ),
                torch.where(
                    chance,
                    chance_decision.executed_degree,
                    ones,
                ),
                torch.where(
                    chance,
                    chance_decision.target_degree,
                    ones,
                ),
            )
            degree = torch.where(
                current_active,
                chance_decision.executed_degree,
                torch.zeros_like(chance_decision.executed_degree),
            ).to(torch.int64)
            sampled_nodes += degree.masked_fill(~chance, 0).sum()
            cumulative = degree.cumsum(0)
            chance_total = cumulative[-1]
            frontier_demand = torch.maximum(frontier_demand, chance_total)
            frontier_overflow |= chance_total.gt(generation_capacity)
            next_active = frontier_row < chance_total
            chance_source = torch.searchsorted(
                cumulative,
                frontier_row,
                right=True,
            ).clamp_max(current_rows - 1)
            prefix = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    cumulative[:-1],
                )
            )
            ordinal = frontier_row - prefix.index_select(0, chance_source)
            source_path = current_key[:, 1].index_select(0, chance_source)
            source_is_chance = chance.index_select(0, chance_source)
            branched_path = child_path_keys(
                source_path,
                ordinal + 1,
            )
            next_path = torch.where(
                source_is_chance,
                branched_path,
                source_path,
            )
            next_key = torch.stack(
                (
                    current_key[:, 0].index_select(0, chance_source),
                    next_path,
                ),
                -1,
            )
            expanded_state = replace(
                current_state,
                **{
                    name: value.index_select(0, chance_source)
                    for name, value in current_state.__dict__.items()
                    if isinstance(value, Tensor)
                },
            )
            expanded_private = current_private.index_select(0, chance_source)
            if (
                self.mode.private_estimator is PrivateEstimator.RAO_BLACKWELL
                and not self._safe_tree_triggered
            ):
                finite = torch.isfinite(current_log_reach)
                gauge = current_log_reach.masked_fill(
                    ~finite, -torch.inf
                ).amax(-1, keepdim=True)
                range_weights = torch.exp(
                    current_log_reach
                    - gauge.masked_fill(~torch.isfinite(gauge), 0.0)
                ).masked_fill(~finite, 0.0)
                assert self.posterior_sampler is not None
                expanded_private = self.posterior_sampler.sample_children(
                    range_weights=range_weights,
                    board_cards=current_state.board_cards,
                    inherited_profile=current_private,
                    source=chance_source,
                    ordinal=ordinal,
                    active=next_active & source_is_chance,
                    root=next_key[:, 0],
                    path=next_key[:, 1],
                    seed=self.seed,
                    iteration=self.iteration,
                    retry_epoch=retry,
                )
                self._observe_posterior()
            expanded_state = replace(expanded_state, combo_ids=expanded_private)
            appended_history = self.machine.append_chance_history(
                current_state, chance, current_history
            )
            chance_history = replace(
                appended_history,
                **{
                    name: value.index_select(0, chance_source)
                    for name, value in appended_history.__dict__.items()
                },
            )
            dealt_state = self.machine.deal_next_street_keyed(
                expanded_state,
                seed=self.seed,
                iteration=self.iteration,
                root=next_key[:, 0],
                path=next_key[:, 1],
                draw=generation,
                retry_epoch=retry,
                active=next_active,
            )
            final_generation = generation == 2
            terminal_child = (
                dealt_state.terminal
                if final_generation
                else torch.zeros_like(dealt_state.terminal)
            )
            dealt_street = expanded_state.street
            flop_label = (
                dealt_state.board_cards[:, 0].to(torch.int32) * (52 * 52)
                + dealt_state.board_cards[:, 1].to(torch.int32) * 52
                + dealt_state.board_cards[:, 2].to(torch.int32)
            )
            turn_or_river = dealt_state.board_cards.gather(
                1,
                dealt_street.clamp(1, 2).to(torch.int64)[:, None] + 2,
            ).squeeze(1).to(torch.int32)
            chance_label = torch.where(
                dealt_street.eq(0), flop_label, turn_or_river
            )
            next_endpoint_record = endpoint_record.index_select(
                0, chance_source
            )
            next_endpoint_record_edge = endpoint_record_edge.index_select(
                0, chance_source
            )
            chance_edge_active = (
                next_active & chance.index_select(0, chance_source)
            )
            record_identifier = 1 + len(chance_records)
            next_endpoint_record = torch.where(
                chance_edge_active,
                torch.full_like(next_endpoint_record, record_identifier),
                next_endpoint_record,
            )
            next_endpoint_record_edge = torch.where(
                chance_edge_active,
                frontier_row,
                next_endpoint_record_edge,
            )
            chance_records.append(
                TopologyExpansionRecord(
                    node_active=chance,
                    edge_active=chance_edge_active,
                    parent_key=current_key,
                    child_key=next_key,
                    source_ordinal=chance_source,
                    terminal=terminal_child,
                    child_input=self.machine.input_type(dealt_state),
                    child_leaf_row=torch.full(
                        (generation_capacity,),
                        -1,
                        dtype=torch.int64,
                        device=device,
                    ),
                    label=chance_label,
                    owner=torch.full(
                        (current_rows,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    input_type=torch.full(
                        (current_rows,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    familywork_row=torch.full(
                        (current_rows,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    sigma_head_row=torch.full(
                        (current_rows,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    action_begin=torch.full(
                        (current_rows,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    executed_degree=chance_decision.executed_degree,
                    nominal_degree=chance_decision.nominal_degree,
                    target_degree=chance_decision.target_degree,
                    trained=torch.zeros(
                        current_rows,
                        dtype=torch.bool,
                        device=device,
                    ),
                    refill=torch.zeros(
                        current_rows,
                        dtype=torch.bool,
                        device=device,
                    ),
                    path_length=torch.ones(
                        generation_capacity,
                        dtype=torch.int32,
                        device=device,
                    ),
                )
            )
            endpoint_source = endpoint_source.index_select(0, chance_source)
            endpoint_action = endpoint_action.index_select(0, chance_source)
            endpoint_mass = (
                endpoint_mass.index_select(0, chance_source)
                / degree.index_select(0, chance_source).clamp_min(1).float()
            )
            endpoint_log_sum, endpoint_log_product = child_path_factors(
                endpoint_log_sum.index_select(0, chance_source),
                endpoint_log_product.index_select(0, chance_source),
                chance_decision.beta.index_select(0, chance_source),
                child_is_player=(
                    ~dealt_state.terminal & ~self.machine.needs_chance(dealt_state)
                    if final_generation
                    else torch.zeros(
                        generation_capacity,
                        dtype=torch.bool,
                        device=device,
                    )
                ),
            )
            current_key = next_key
            current_state = dealt_state
            current_private = expanded_private
            current_history = chance_history
            current_log_reach = current_log_reach.index_select(0, chance_source)
            current_frontier_cost = (
                None
                if current_frontier_cost is None
                else current_frontier_cost.index_select(0, chance_source)
            )
            current_beta = chance_decision.beta.index_select(0, chance_source)
            current_active = next_active
            endpoint_record = next_endpoint_record
            endpoint_record_edge = next_endpoint_record_edge
        next_state = current_state
        private = current_private
        history = current_history
        log_reach = current_log_reach
        root = current_key[:, 0]
        child_path = current_key[:, 1]
        total = current_active.numel()
        board = next_state.board_cards
        cards = self.machine.combo_cards(
            torch.arange(
                self.geometry.private_hands,
                dtype=torch.int32,
                device=device,
            )
        )
        conflict = (
            (
                board.ne(PAD_CARD)[..., None, None]
                & board.long()[..., None, None].eq(cards[None])
            )
            .any(1)
            .any(-1)
        )
        log_reach = log_reach.masked_fill(conflict[:, None, :], -torch.inf)
        terminal = next_state.terminal
        next_family = self.machine.input_type(next_state)
        # A terminal output domain is selected by street, not by the number
        # of surviving contenders (postflop folds may leave one contender).
        next_family = torch.where(
            terminal,
            next_state.street.ne(0).to(next_family.dtype),
            next_family,
        )
        route_family = torch.where(
            terminal,
            torch.full_like(next_family, -1),
            next_family,
        )[None]
        route = stable_route(
            route_family,
            current_active[None],
            self._route_family_values,
            declared_values=self._route_families,
        )
        local_address = batch.fields["logical_rng_key"][:, :, 0].amin(1)
        address = torch.full(
            (self.owners.n_replica,),
            torch.iinfo(torch.int64).max,
            dtype=torch.int64,
            device=device,
        )
        address[lane] = local_address[0]

        def lane_count(value: Tensor) -> Tensor:
            result = torch.zeros(
                self.owners.n_replica,
                dtype=torch.int64,
                device=device,
            )
            result[lane] = value.to(torch.int64)
            return result

        destinations: list[DeviceDestinationRequest] = []
        owner = batch.fields["owner"][0].to(torch.int64)
        selected_player = (
            batch.fields["role_selector"][0].to(torch.int64)
            if self.mode.training_regime is TrainingRegime.BEST_RESPONSE
            else owner
        )
        leaf = next(slab for slab in self.owners.slabs if slab.schema.name == "Leaf")
        leaf_old_tail = leaf.reserved_tails_device()[lane]
        leaf_values = self._defaults(leaf, total)
        terminal_state = replace(
            next_state,
            terminal=torch.ones_like(next_state.terminal),
        )
        sampled_mode = self.mode.private_estimator is PrivateEstimator.SAMPLED
        leaf_source = (
            self.machine.sampled_leaf(terminal_state)
            if sampled_mode
            else self.machine.rb_leaf(
                terminal_state,
                log_reach,
                seed=logical_rng_keys(
                    seed=self.seed,
                    iteration=self.iteration,
                    root=root,
                    path=child_path,
                    draw=0,
                    purpose=DrawPurpose.COMPLETION,
                    lane=lane,
                    retry_epoch=retry,
                    active=current_active,
                ),
                selected_player=(
                    selected_player.index_select(0, endpoint_source)
                    if self.mode.training_regime
                    is TrainingRegime.BEST_RESPONSE
                    else None
                ),
            )
        )
        if (
            sampled_mode
            and self.mode.training_regime is TrainingRegime.BEST_RESPONSE
        ):
            leaf_source = MappingProxyType(
                {
                    **dict(leaf_source),
                    "trained_actor": selected_player.index_select(
                        0, endpoint_source
                    ).to(torch.int32),
                }
        )
        terminal_rows = route.source_row[0, 0]
        for name, value in leaf_source.items():
            leaf_values[name][lane].copy_(value.index_select(0, terminal_rows))
        leaf_count = lane_count(route.counts[0, 0])
        destinations.append(
            DeviceDestinationRequest(
                leaf, leaf_count, leaf_values, batch.input_type, address
            )
        )

        trained = (
            owner.eq(selected_player)
            if self.mode.training_regime is TrainingRegime.BEST_RESPONSE
            else torch.ones(rows, dtype=torch.bool, device=device)
        )
        if self._refill:
            trained = torch.zeros_like(trained)
        trained_row = torch.argsort(~trained, stable=True)
        trained_valid = trained.index_select(0, trained_row)
        trained_count = lane_count(trained.sum(dtype=torch.int64))
        sigma = next(
            slab
            for slab in self.owners.slabs
            if slab.schema.name == f"SigmaHead.{batch.input_type}"
        )
        sigma_old_tail = sigma.reserved_tails_device()[lane]
        sigma_values = self._defaults(sigma, rows)
        model_value = request.model.values["value"]
        parent_index = self._private_index(
            batch.input_type,
            parent_private.gather(1, owner[:, None]).squeeze(1),
        )
        selected_value = (
            model_value.gather(
                1,
                parent_index[:, None],
            ).float()
            if sampled_mode
            else self._natural_rows(model_value, batch.input_type)
        )
        sigma_values["old_value"][lane].copy_(
            selected_value.index_select(0, trained_row)
        )
        sigma_values["old_trace"][lane].copy_(
            request.model.values["trace"].index_select(0, trained_row)
        )
        sigma_values["old_cost"][lane].copy_(
            request.model.values["cost"].index_select(0, trained_row)
        )
        destinations.append(
            DeviceDestinationRequest(
                sigma, trained_count, sigma_values, batch.input_type, address
            )
        )

        output_type = (
            "all"
            if sampled_mode
            else ("pre" if batch.input_type == "pre" else "post")
        )
        action_head = next(
            slab
            for slab in self.owners.slabs
            if slab.schema.name == f"ActionHead.{output_type}"
        )
        action_old_tail = action_head.reserved_tails_device()[lane]
        trained_child = trained.index_select(0, action_source) & action_active
        trained_action_row = torch.argsort(~trained_child, stable=True)
        trained_action_valid = trained_child.index_select(
            0, trained_action_row
        )
        trained_action_count = lane_count(
            trained_child.sum(dtype=torch.int64)
        )
        action_values = self._defaults(action_head, action_total)
        action_values["action"][lane].copy_(
            action.index_select(0, trained_action_row)
        )
        retained_log = (
            sample["selected_log_policy"].reshape(action_total, -1)[:, :1]
            if sampled_mode
            else self._natural_rows(selected_log, batch.input_type)
        )
        action_values["old_log_policy"][lane].copy_(
            retained_log.index_select(0, trained_action_row)
        )
        destinations.append(
            DeviceDestinationRequest(
                action_head,
                trained_action_count,
                action_values,
                output_type,
                address,
            )
        )

        fold = action.eq(FOLD) & action_active
        endpoint_nonterminal = current_active & ~terminal
        fold_consumed = torch.zeros(
            action_total, dtype=torch.int32, device=device
        )
        fold_consumed.scatter_reduce_(
            0,
            endpoint_action,
            endpoint_nonterminal.to(torch.int32),
            reduce="amax",
            include_self=True,
        )
        stored_fold = fold & fold_consumed.gt(0)
        finite_fold = torch.isfinite(fold_log_reach)
        fold_gauge = fold_log_reach.masked_fill(
            ~finite_fold, -torch.inf
        ).amax(-1, keepdim=True)
        fold_weight = torch.exp(
            fold_log_reach
            - fold_gauge.masked_fill(~torch.isfinite(fold_gauge), 0.0)
        ).masked_fill(~finite_fold, 0.0)
        actor_weight = fold_weight.gather(
            1,
            actor[:, None, None].expand(
                -1, 1, self.geometry.private_hands
            ),
        ).squeeze(1)
        actor_weight /= actor_weight.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        folded_partition = "pre" if batch.input_type == "pre" else "post"
        folded_slab = self._owner_slab("FoldedRanges", folded_partition)
        folded_rows = actor_weight
        if folded_partition == "pre":
            classes = self._combo_class()
            folded_rows = torch.zeros(
                (action_total, self.geometry.private_classes),
                dtype=torch.float32,
                device=device,
            )
            folded_rows.scatter_add_(
                1,
                classes[None].expand_as(actor_weight),
                actor_weight,
            )
        if folded_slab.columns["range"].shape[-1] == 1:
            folded_rows = folded_rows.sum(-1, keepdim=True)
        fold_order = torch.argsort(~stored_fold, stable=True)
        folded_values = self._defaults(folded_slab, action_total)
        folded_values["range"][lane].copy_(
            folded_rows.index_select(0, fold_order)
        )
        actor_log_mass = torch.logsumexp(
            fold_log_reach.gather(
                1,
                actor[:, None, None].expand(
                    -1, 1, self.geometry.private_hands
                ),
            ).squeeze(1),
            -1,
        )
        folded_values["log_mass"][lane].copy_(
            actor_log_mass.index_select(0, fold_order)
        )
        fold_count = lane_count(stored_fold.sum(dtype=torch.int64))
        fold_base = folded_slab.reserved_tails_device()[lane]
        fold_ordinal = stored_fold.cumsum(0, dtype=torch.int64) - 1
        encoded_reference = (
            fold_base + fold_ordinal
            if folded_partition == "pre"
            else -(fold_base + fold_ordinal + 2)
        )
        action_folded_reference = batch.fields["folded_range_row"][
            0
        ].index_select(0, action_source).to(torch.int64)
        prior_actor_reference = action_folded_reference.gather(
            1, actor[:, None]
        ).squeeze(1)
        action_folded_reference = action_folded_reference.scatter(
            1,
            actor[:, None],
            torch.where(
                stored_fold,
                encoded_reference,
                prior_actor_reference,
            )[:, None],
        )
        folded_reference = action_folded_reference.index_select(
            0, endpoint_action
        )
        post_summary = log_reach.new_zeros(
            (total, self.geometry.folded_summary_width)
        )
        if self.geometry.folded_summary_width:
            if batch.input_type != "pre":
                post_summary.copy_(
                    flat_input.index_select(0, endpoint_source)[
                        :, self.geometry.postflop_base_model_input_width :
                    ].float()
                )
            endpoint_fold = stored_fold.index_select(0, endpoint_action)
            refresh = (
                endpoint_fold
                | next_state.street.ne(
                    fold_parent_street.index_select(0, endpoint_action)
                )
            ) & endpoint_nonterminal & next_state.street.ne(0)
            refresh_rows = torch.nonzero(
                refresh, as_tuple=False
            ).flatten()
            from pbg.games.poker.queued.folded import (
                folded_statistics,
                pack_folded_statistics,
            )

            refreshed_summary = folded_statistics(
                self.geometry,
                board_cards=next_state.board_cards.index_select(
                    0, refresh_rows
                ),
                range_weights=torch.exp(
                    log_reach.index_select(0, refresh_rows).float()
                ).masked_fill(
                    ~torch.isfinite(
                        log_reach.index_select(0, refresh_rows)
                    ),
                    0.0,
                ),
                status=next_state.status.index_select(0, refresh_rows),
                combo_cards=cards,
                beta_rank=self.geometry.deck_cards,
                line_dim=0,
                tight_mask=self._tight_folder_mask(
                    history, next_state.status
                ).index_select(0, refresh_rows),
            )
            post_summary.index_copy_(
                0,
                refresh_rows,
                pack_folded_statistics(
                    self.geometry, refreshed_summary
                ),
            )
        next_logical = torch.stack((root, child_path), -1)
        family_source: dict[str, Tensor] = {
            "logical_rng_key": next_logical,
            "true_profile": private,
            "folded_range_row": folded_reference.to(torch.int32),
            "owner": next_state.to_act,
            "path_mass": endpoint_mass,
            "log_player_ancestor_sum": endpoint_log_sum,
            "log_ancestor_product": endpoint_log_product,
            "branch_beta": current_beta,
            "nominal_degree": torch.zeros(total, dtype=torch.int32, device=device),
            "executed_degree": torch.zeros(total, dtype=torch.int32, device=device),
            "target_degree": torch.zeros(total, dtype=torch.int32, device=device),
        }
        if sampled_mode:
            family_source["sampled_private_state"] = private
        if "role_selector" in batch.fields:
            family_source["role_selector"] = batch.fields["role_selector"][
                0
            ].index_select(0, endpoint_source)
        published: list[SemanticFamilyFIFO] = []
        queue_cost_additions: list[Tensor] = []
        for family_index, (name, queue) in enumerate(self.queues.items()):
            selected_rows = route.source_row[0, family_index + 1]
            selected_valid = route.valid[0, family_index + 1]
            values = self._defaults(queue.family, total)
            next_input = self.machine.direct_model_input(
                name, next_state, log_reach, history
            )
            if name != "pre" and self.geometry.folded_summary_width:
                next_input[
                    :, self.geometry.postflop_base_model_input_width :
                ].copy_(post_summary.to(next_input.dtype))
            queue_source = {**family_source, "model_input": next_input}
            for field_name, value in queue_source.items():
                if field_name in values:
                    values[field_name][lane].copy_(
                        value.index_select(0, selected_rows)
                    )
            cost_addition = torch.zeros(
                self.owners.n_replica,
                dtype=torch.float32,
                device=device,
            )
            if (
                self.cost_safeguard is not None
                and self.cost_safeguard.active
            ):
                active_position = torch.nonzero(
                    selected_valid, as_tuple=False
                ).flatten()
                active_rows = selected_rows.index_select(
                    0, active_position
                )
                selected_role = (
                    None
                    if "role_selector" not in family_source
                    else family_source["role_selector"].index_select(
                        0, active_rows
                    )
                )
                child_cost = torch.zeros(
                    selected_valid.shape,
                    dtype=torch.float32,
                    device=device,
                )
                if active_rows.numel():
                    with torch.no_grad():
                        child_evaluation = self.child_model(
                            PokerForwardInput(
                                name,
                                next_input.index_select(0, active_rows),
                                private.index_select(0, active_rows),
                                next_state.to_act.index_select(
                                    0, active_rows
                                ),
                                selected_role,
                            ),
                            torch.tensor(
                                [lane],
                                dtype=torch.int64,
                                device=device,
                            ),
                        )
                    child_cost.index_copy_(
                        0,
                        active_position,
                        child_evaluation["cost"].detach().float(),
                    )
                values["frontier_cost_contribution"][lane].copy_(child_cost)
                cost_addition[lane] = child_cost.sum()
            count = lane_count(route.counts[0, family_index + 1])
            destinations.append(
                DeviceDestinationRequest(queue.family, count, values, name, address)
            )
            published.append(queue)
            queue_cost_additions.append(cost_addition)

        folded_destination = len(destinations)
        destinations.append(
            DeviceDestinationRequest(
                folded_slab,
                fold_count,
                folded_values,
                batch.input_type,
                address,
            )
        )
        descriptor_source: list[Tensor] = []
        descriptor_destination: list[Tensor] = []
        descriptor_ordinal: list[Tensor] = []
        descriptor_valid: list[Tensor] = []
        compact = torch.arange(rows, dtype=torch.int64, device=device)
        descriptor_source.append(trained_row)
        descriptor_destination.append(torch.full_like(compact, 1))
        descriptor_ordinal.append(compact)
        descriptor_valid.append(trained_valid)
        action_compact = torch.arange(
            action_total, dtype=torch.int64, device=device
        )
        descriptor_source.append(
            action_source.index_select(0, trained_action_row)
        )
        descriptor_destination.append(torch.full_like(action_compact, 2))
        descriptor_ordinal.append(action_compact)
        descriptor_valid.append(trained_action_valid)
        descriptor_source.append(
            action_source.index_select(0, fold_order)
        )
        descriptor_destination.append(
            torch.full_like(action_compact, folded_destination)
        )
        descriptor_ordinal.append(action_compact)
        descriptor_valid.append(
            stored_fold.index_select(0, fold_order)
        )
        endpoint_destination = torch.zeros(
            total, dtype=torch.int64, device=device
        )
        endpoint_ordinal = torch.zeros_like(endpoint_destination)
        for family_value in self._route_family_values:
            selected = route_family[0].eq(family_value)
            request_index = 0 if family_value == -1 else 3 + family_value
            endpoint_destination = torch.where(
                selected,
                torch.full_like(endpoint_destination, request_index),
                endpoint_destination,
            )
            endpoint_ordinal = torch.where(
                selected,
                selected.cumsum(0, dtype=torch.int64) - 1,
                endpoint_ordinal,
            )
        descriptor_source.append(endpoint_source)
        descriptor_destination.append(endpoint_destination)
        descriptor_ordinal.append(endpoint_ordinal)
        descriptor_valid.append(current_active)
        source_descriptor = torch.cat(descriptor_source)
        destination_descriptor = torch.cat(descriptor_destination)
        ordinal_descriptor = torch.cat(descriptor_ordinal)
        valid_descriptor = torch.cat(descriptor_valid)
        extent = int(source_descriptor.numel())
        coverage_valid = torch.zeros(
            (self.owners.n_replica, extent),
            dtype=torch.bool,
            device=device,
        )
        coverage_source = torch.zeros(
            (self.owners.n_replica, extent),
            dtype=torch.int64,
            device=device,
        )
        coverage_destination = torch.zeros_like(coverage_source)
        coverage_ordinal = torch.zeros_like(coverage_source)
        coverage_valid[lane].copy_(valid_descriptor)
        coverage_source[lane].copy_(source_descriptor)
        coverage_destination[lane].copy_(destination_descriptor)
        coverage_ordinal[lane].copy_(ordinal_descriptor)
        coverage = CommitCoverage(
            valid=coverage_valid,
            source_ordinal=coverage_source,
            destination=coverage_destination,
            destination_ordinal=coverage_ordinal,
        )
        self.routing_traffic_bytes += sum(
            value.numel() * value.element_size()
            for value in (
                route.source_row,
                route.valid,
                route.counts,
                route.offsets,
                route.family_values,
                route.declared,
                coverage.valid,
                coverage.source_ordinal,
                coverage.destination,
                coverage.destination_ordinal,
            )
        )

        trained_prefix = trained.to(torch.int64).cumsum(0) - 1
        degree_prefix = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                sample["executed_degree"].to(torch.int64).cumsum(0)[:-1],
            )
        )
        trained_degree = sample["executed_degree"].to(torch.int64) * trained
        action_prefix = trained_degree.cumsum(0) - trained_degree
        leaf_row = torch.where(
            current_active & terminal,
            leaf_old_tail + endpoint_ordinal,
            torch.full_like(endpoint_ordinal, -1),
        )
        player_record = TopologyExpansionRecord(
            node_active=torch.ones(rows, dtype=torch.bool, device=device),
            edge_active=action_active,
            parent_key=batch.fields["logical_rng_key"][0],
            child_key=first_child_key,
            source_ordinal=action_source,
            terminal=torch.zeros(action_total, dtype=torch.bool, device=device),
            child_input=torch.full(
                (action_total,), -1, dtype=torch.int32, device=device
            ),
            child_leaf_row=torch.full(
                (action_total,), -1, dtype=torch.int64, device=device
            ),
            label=action.to(torch.int32),
            owner=(
                torch.zeros_like(owner, dtype=torch.int32)
                if self.mode.training_regime is TrainingRegime.BEST_RESPONSE
                else owner.to(torch.int32)
            ),
            input_type=torch.full(
                (rows,),
                self.input_index[batch.input_type],
                dtype=torch.int32,
                device=device,
            ),
            familywork_row=batch.row[0].to(torch.int32),
            sigma_head_row=torch.where(
                trained,
                sigma_old_tail + trained_prefix,
                torch.full_like(trained_prefix, -1),
            ).to(torch.int32),
            action_begin=torch.where(
                trained,
                action_old_tail + action_prefix,
                torch.full_like(action_prefix, -1),
            ).to(torch.int32),
            executed_degree=sample["executed_degree"].to(torch.int32),
            nominal_degree=sample["nominal_degree"].to(torch.int32),
            target_degree=sample["target_degree"].to(torch.int32),
            trained=trained,
            refill=torch.full((rows,), self._refill, dtype=torch.bool, device=device),
            path_length=torch.ones(action_total, dtype=torch.int32, device=device),
        )
        device_records = (player_record, *chance_records)
        final_key = torch.stack((root, child_path), -1)
        for record_identifier, record in enumerate(device_records):
            edge_count = record.child_key.shape[0]
            matching = current_active & endpoint_record.eq(record_identifier)
            candidate = torch.where(
                matching,
                frontier_row,
                torch.full_like(frontier_row, total),
            )
            edge = endpoint_record_edge.clamp(0, edge_count - 1)
            inverse = torch.full(
                (edge_count,), total, dtype=torch.int64, device=device
            )
            inverse.scatter_reduce_(
                0, edge, candidate, reduce="amin", include_self=True
            )
            valid_edge = inverse.lt(total)
            final_row = inverse.clamp_max(total - 1)
            record.child_key.copy_(
                torch.where(
                    valid_edge[:, None],
                    final_key.index_select(0, final_row),
                    record.child_key,
                )
            )
            record.terminal.copy_(
                torch.where(
                    valid_edge,
                    terminal.index_select(0, final_row),
                    record.terminal,
                )
            )
            record.child_input.copy_(
                torch.where(
                    valid_edge,
                    next_family.index_select(0, final_row),
                    record.child_input,
                )
            )
            record.child_leaf_row.copy_(
                torch.where(
                    valid_edge,
                    leaf_row.index_select(0, final_row),
                    record.child_leaf_row,
                )
            )
        correction = (
            (
                sample["nominal_degree"].float()
                - sample["executed_degree"].float()
            )
            * request.model.values["cost"].detach().float()
        ).sum()
        staged_payload = self.topology_stager.stage(
            device_records,
            slot=(
                self._topology_sequence
                % self.topology_stager.inflight_cap
            ),
            produced=route.counts[0, 1:],
            sampled_nodes=sampled_nodes,
            counterfactual_correction=correction,
            frontier_overflow=frontier_overflow,
            frontier_demand=frontier_demand,
        )
        self.realized_tree_size.add_(sampled_nodes.to(torch.float32))
        topology_publication = TopologyPublication(
            lane=lane,
            sequence=self._topology_sequence,
            declared_nodes=self.topology_stager.topology_packet_cap,
            declared_edges=0,
            payload=staged_payload,
        )
        self._topology_sequence += 1
        self.last_topology = topology_publication
        self._refill = False
        self._refill_child_ordinal = 0
        del degree_prefix
        return CommitPlan(
            tuple(destinations),
            coverage,
            published_queues=tuple(published),
            topology=(topology_publication,),
            queue_cost_additions=tuple(queue_cost_additions),
            device_failure=(
                DeviceCommitFailure(
                    failed=torch.zeros(
                        self.owners.n_replica,
                        dtype=torch.bool,
                        device=device,
                    ).index_copy(
                        0, batch.lanes, self._posterior_failed.reshape(1)
                    ),
                    logical_address=torch.full(
                        (self.owners.n_replica,),
                        torch.iinfo(torch.int64).max,
                        dtype=torch.int64,
                        device=device,
                    ).index_copy(
                        0, batch.lanes, self._posterior_address.reshape(1)
                    ),
                    requested_extent=torch.zeros(
                        self.owners.n_replica,
                        dtype=torch.int64,
                        device=device,
                    ).index_copy(
                        0,
                        batch.lanes,
                        (self._posterior_attempts + 1).reshape(1),
                    ),
                    available_extent=torch.zeros(
                        self.owners.n_replica,
                        dtype=torch.int64,
                        device=device,
                    ).index_fill_(
                        0, batch.lanes, self.posterior_sampler.max_attempts
                    ),
                    sampler_attempt_count=torch.zeros(
                        self.owners.n_replica,
                        dtype=torch.int64,
                        device=device,
                    ).index_copy(
                        0, batch.lanes, self._posterior_attempts.reshape(1)
                    ),
                    phase="posterior_sample",
                    failure_code="rejection_exhausted",
                    input_type=batch.input_type,
                )
                if self.posterior_sampler is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ReplicaBranchedPokerResult:
    """Lane-owned results from one shared branched storage transaction."""

    results: tuple[BranchedPokerResult | None, ...]
    failures: Mapping[int, str]
    versions: tuple[int, ...]


class PokerBranchedProgram:
    """Supply every branched lane to the common iteration engine."""

    def __init__(
        self,
        *,
        geometry: NLHEGeometry,
        owners: GeneratedOwners,
        modes: Sequence[ModeSchema],
        controllers: Sequence[BranchController],
        safeguards: Sequence[CostSafeguardController | None],
        inference: PokerReplicaInference | None,
        reference_operations: Sequence[ModelOperation | None],
        optimizers: ReplicaOptimizerGroup | object,
        losses: Sequence[LossConfig],
        schedule: LearnerSchedule,
        generators: Sequence[torch.Generator],
        run_states: Sequence[RunState | None],
        minimum_tree_sizes: Sequence[int],
        seed: int,
        lambda_: float,
        precision: str = "fp32",
        utility: str = "raw_bb",
        completion_samples: int = 0,
        posterior_max_attempts: int = 256,
        cache_reference_outputs: bool = False,
        frozen_inference: PokerReplicaInference | None = None,
        models: Sequence[nn.Module] | None = None,
        rollout_operations: Sequence[ModelOperation] | None = None,
        learner_operations: Sequence[ModelOperation] | None = None,
        frozen_operations: Sequence[ModelOperation | None] | None = None,
        allocation_inputs: Sequence[
            tuple[Tensor, Tensor, Tensor, int] | None
        ] | None = None,
        benchmark_telemetry: bool = False,
        distributed: DistributedSUM | None = None,
    ) -> None:
        count = owners.plan.manifest.n_replica
        self.modes = tuple(modes)
        self.controllers = tuple(controllers)
        self.safeguards = tuple(safeguards)
        self.reference_operations = tuple(reference_operations)
        self.losses = tuple(losses)
        self.generators = tuple(generators)
        self.run_states = tuple(run_states)
        self.minimum_tree_sizes = tuple(int(value) for value in minimum_tree_sizes)
        if any(
            len(values) != count
            for values in (
                self.modes,
                self.controllers,
                self.safeguards,
                self.reference_operations,
                self.losses,
                self.generators,
                self.run_states,
                self.minimum_tree_sizes,
            )
        ):
            raise ValueError("branched replica lane owners are not aligned")
        if any(value <= 0 for value in self.minimum_tree_sizes):
            raise ValueError("minimum tree sizes must be positive")
        validate_replica_group(self.modes)
        if not all(
            mode.rollout_mode.is_branched for mode in self.modes
        ) or owners.plan.mode != self.modes[0]:
            raise ValueError("branched replica physical owners are incompatible")
        if inference is not None:
            if (
                inference.modules.n_replica != count
                or getattr(optimizers, "modules", None)
                is not inference.modules
            ):
                raise ValueError(
                    "branched replica model and optimizer owners differ"
                )
            resolved_models = tuple(
                inference.modules.lane_module(lane)
                for lane in range(count)
            )
        else:
            resolved_models = () if models is None else tuple(models)
            if len(resolved_models) != count:
                raise ValueError(
                    "direct branched execution requires one model per lane"
                )
        self.models = resolved_models
        self.rollout_operations = (
            None
            if rollout_operations is None
            else tuple(rollout_operations)
        )
        if inference is None and (
            self.rollout_operations is None
            or len(self.rollout_operations) != count
        ):
            raise ValueError(
                "direct branched execution requires rollout operations"
            )
        self.learner_operations = (
            None
            if learner_operations is None
            else tuple(learner_operations)
        )
        if self.learner_operations is not None and (
            len(self.learner_operations) != count
        ):
            raise ValueError("learner operation lanes are not aligned")
        self.frozen_operations = (
            tuple(None for _ in range(count))
            if frozen_operations is None
            else tuple(frozen_operations)
        )
        if len(self.frozen_operations) != count:
            raise ValueError("frozen operation lanes are not aligned")
        self.allocation_inputs = (
            tuple(None for _ in range(count))
            if allocation_inputs is None
            else tuple(allocation_inputs)
        )
        if len(self.allocation_inputs) != count:
            raise ValueError("statistical allocation lanes are not aligned")
        if type(benchmark_telemetry) is not bool:
            raise TypeError("benchmark_telemetry must be bool")
        self.benchmark_telemetry = benchmark_telemetry
        if any(
            controller.mode is not mode.rollout_mode
            or controller.beta_max != owners.plan.manifest.beta_max
            for mode, controller in zip(
                self.modes, self.controllers, strict=True
            )
        ):
            raise ValueError("branched replica controllers differ from their lanes")
        if any(
            generator.device != owners.plan.device
            for generator in self.generators
        ):
            raise ValueError("branched replica RNGs must use the selected CUDA device")
        if any(
            state is not None
            and state.root_count != owners.plan.manifest.root_count
            for state in self.run_states
        ):
            raise ValueError("branched replica root counts differ from admission")
        if (
            any(
                mode.training_regime is TrainingRegime.BEST_RESPONSE
                for mode in self.modes
            )
            and frozen_inference is None
            and any(value is None for value in self.frozen_operations)
        ):
            raise ValueError(
                "branched replica best response requires frozen lane ownership"
            )
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("branched replica precision is invalid")
        if utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("branched replica utility is invalid")
        if any(
            mode.private_estimator is PrivateEstimator.RAO_BLACKWELL
            for mode in self.modes
        ) and completion_samples <= 0:
            raise ValueError("branched replica RB requires completion samples")
        self.geometry = geometry
        self.owners = owners
        self.inference = inference
        self.frozen_inference = frozen_inference
        self.optimizers = optimizers
        self.schedule = schedule
        self.seed = int(seed)
        self.lambda_ = float(lambda_)
        self.precision = precision
        self.utility = utility
        self.completion_samples = int(completion_samples)
        self.posterior_max_attempts = int(posterior_max_attempts)
        if type(cache_reference_outputs) is not bool:
            raise TypeError("cache_reference_outputs must be bool")
        self.cache_reference_outputs = cache_reference_outputs
        self.distributed = distributed
        self.policy = PokerPolicyOperator()
        self._model_events: dict[
            tuple[int, str],
            list[tuple[torch.cuda.Event, torch.cuda.Event, int]],
        ] = {}

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

    def _publish_root(
        self,
        lane: int,
        machine: PokerStateMachine,
        queue: SemanticFamilyFIFO,
        *,
        iteration: int,
    ) -> None:
        state_owner = self.run_states[lane]
        roots = (
            self.owners.plan.manifest.root_count
            if state_owner is None
            else state_owner.root_count
        )
        epoch = 0 if state_owner is None else state_owner.retry_epoch
        logical_base = (
            0 if state_owner is None else state_owner.attempt_logical_base
        )
        state = machine.initial_roots(
            roots,
            seed=self.seed,
            iteration=iteration,
            retry_epoch=epoch,
            logical_base=logical_base,
            lane=lane,
            device=self.owners.plan.device,
        )
        values = self._defaults(queue.family, roots)
        logical = torch.arange(
            logical_base,
            logical_base + roots,
            dtype=torch.int64,
            device=state.device,
        )
        values["logical_rng_key"][lane, :, 0] = logical
        values["logical_rng_key"][lane, :, 1] = 0
        values["true_profile"][lane] = state.combo_ids
        if "sampled_private_state" in values:
            values["sampled_private_state"][lane] = state.combo_ids
        values["model_input"][lane] = machine.direct_model_input("pre", state)
        values["owner"][lane] = state.to_act
        values["path_mass"][lane].fill_(1)
        values["log_player_ancestor_sum"][lane].zero_()
        values["log_ancestor_product"][lane].zero_()
        values["branch_beta"][lane].fill_(1)
        if "role_selector" in values:
            values["role_selector"][lane] = logical.remainder(
                self.geometry.players
            ).to(torch.uint8)
        root_cost = torch.zeros(
            roots, dtype=torch.float32, device=state.device
        )
        if self.safeguards[lane] is not None:
            pack = self.owners.plan.manifest.family_pack_cap
            with torch.no_grad():
                root_cost = torch.cat(
                    tuple(
                        self._inference(
                            lane,
                            PokerForwardInput(
                                "pre",
                                values["model_input"][lane, begin:end],
                                state.combo_ids[begin:end],
                                state.to_act[begin:end],
                                (
                                    None
                                    if "role_selector" not in values
                                    else values["role_selector"][
                                        lane, begin:end
                                    ]
                                ),
                            ),
                        )["cost"]
                        for begin in range(0, roots, pack)
                        for end in (min(begin + pack, roots),)
                    )
                )
            values["frontier_cost_contribution"][lane].copy_(root_cost)
        counts = torch.zeros(
            self.owners.plan.manifest.n_replica,
            dtype=torch.int64,
            device=state.device,
        )
        counts[lane] = roots
        address = torch.full_like(counts, torch.iinfo(torch.int64).max)
        address[lane] = logical_base
        active = torch.arange(
            counts.numel(), dtype=torch.int64, device=state.device
        ).eq(lane)
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    queue.family,
                    counts,
                    values,
                    "pre",
                    address,
                ),
            ),
            phase="replica_branched_root_commit",
            active=active,
        )
        transaction.fill_all()
        torch._assert_async(
            transaction.publish()[lane],
            "branched replica root publication failed",
        )
        queue.publish_family_tail(active)
        queue.frontier_cost[lane].copy_(root_cost.sum())

    def _publish_refill_root(
        self,
        lane: int,
        queue: SemanticFamilyFIFO,
        *,
        original_row: int,
        logical_root: int,
    ) -> None:
        values = {
            name: column[:, original_row : original_row + 1].clone()
            for name, column in queue.family.columns.items()
        }
        count = queue.family.columns[
            next(iter(queue.family.columns))
        ].new_zeros(
            queue.family.n_replica,
            dtype=torch.int64,
        )
        count[lane] = 1
        address = torch.full_like(count, torch.iinfo(torch.int64).max)
        address[lane] = logical_root
        active = torch.arange(
            count.numel(), device=queue.family.device
        ).eq(lane)
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    queue.family,
                    count,
                    values,
                    "pre",
                    address,
                ),
            ),
            phase="branched_root_refill",
            active=active,
        )
        transaction.fill_all()
        success = transaction.publish()
        torch._assert_async(
            success[lane],
            "branched root refill exceeded fixed capacity",
        )
        queue.publish_family_tail(active)
        queue.frontier_cost[lane].add_(
            values["frontier_cost_contribution"][lane].sum()
        )

    @torch.no_grad()
    def _inference(
        self,
        lane: int,
        forward: PokerForwardInput,
    ) -> Mapping[str, Tensor]:
        autocast_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise BranchedPokerError(
                "selected CUDA device does not support BF16"
            )
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            return self._inference_impl(lane, forward)

    def _inference_impl(
        self,
        lane: int,
        forward: PokerForwardInput,
    ) -> Mapping[str, Tensor]:
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        mode = self.modes[lane]
        if self.inference is None:
            learned_operation = self.rollout_operations[lane]
            assert learned_operation is not None
            if mode.training_regime is TrainingRegime.SELF_PLAY:
                evaluation = learned_operation(
                    forward.input_type,
                    forward.model_input,
                    forward.sampled_private_state,
                    forward.owner,
                )
            else:
                frozen_operation = self.frozen_operations[lane]
                assert frozen_operation is not None
                assert forward.role_selector is not None
                evaluation = compact_best_response_evaluation(
                    learned_operation,
                    frozen_operation,
                    input_type=forward.input_type,
                    model_input=forward.model_input,
                    sampled_private_state=forward.sampled_private_state,
                    owner=forward.owner,
                    role_selector=forward.role_selector,
                )
        else:
            lanes = torch.tensor(
                [lane], dtype=torch.int64, device=self.owners.plan.device
            )

            def evaluate(
                operation: PokerReplicaInference,
                selected: Tensor | None = None,
            ) -> ModelEvaluation | None:
                if selected is None:
                    selected = torch.arange(
                        forward.owner.numel(),
                        dtype=torch.int64,
                        device=forward.owner.device,
                    )
                if selected.numel() == 0:
                    return None
                return operation(
                    forward.input_type,
                    forward.model_input.index_select(0, selected)[None],
                    forward.sampled_private_state.index_select(0, selected)[None],
                    forward.owner.index_select(0, selected)[None],
                    active_lanes=lanes,
                )

            if mode.training_regime is TrainingRegime.SELF_PLAY:
                result = evaluate(self.inference)
                assert result is not None
                evaluation = result
            else:
                assert self.frozen_inference is not None
                assert forward.role_selector is not None
                trained = forward.owner.eq(forward.role_selector)
                learned_row = torch.nonzero(
                    trained, as_tuple=False
                ).flatten()
                frozen_row = torch.nonzero(
                    ~trained, as_tuple=False
                ).flatten()
                learned = evaluate(self.inference, learned_row)
                frozen = evaluate(self.frozen_inference, frozen_row)
                prototype = learned if learned is not None else frozen
                assert prototype is not None

                def restore(
                    learned_value: Tensor | None,
                    frozen_value: Tensor | None,
                ) -> Tensor:
                    source = (
                        learned_value
                        if learned_value is not None
                        else frozen_value
                    )
                    assert source is not None
                    output = source.new_empty(
                        (1, forward.owner.numel(), *source.shape[2:])
                    )
                    if learned_value is not None:
                        output[0, learned_row] = learned_value[0]
                    if frozen_value is not None:
                        output[0, frozen_row] = frozen_value[0]
                    return output

                evaluation = ModelEvaluation(
                    MappingProxyType(
                        {
                            name: restore(
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
                    restore(
                        None if learned is None else learned.value,
                        None if frozen is None else frozen.value,
                    ),
                    MappingProxyType(
                        {
                            name: restore(
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
            evaluation = ModelEvaluation(
                MappingProxyType(
                    {
                        name: value.squeeze(0)
                        for name, value in evaluation.policy_raw.items()
                    }
                ),
                evaluation.value.squeeze(0),
                MappingProxyType(
                    {
                        name: value.squeeze(0)
                        for name, value in evaluation.public_heads.items()
                    }
                ),
            )
        if set(evaluation.public_heads) != {"trace", "cost"}:
            raise BranchedPokerError("branched replica model omitted T/C heads")
        result = MappingProxyType(
            {
                **{
                    name: value.detach().float()
                    for name, value in evaluation.policy_raw.items()
                },
                "value": evaluation.value.detach().float(),
                "trace": evaluation.public_heads["trace"].detach().float(),
                "cost": evaluation.public_heads["cost"].detach().float(),
            }
        )
        finished.record()
        self._model_events.setdefault((lane, forward.input_type), []).append(
            (started, finished, int(forward.owner.numel()))
        )
        return result

    def _model_timing(self, lane: int) -> Mapping[str, object]:
        result: dict[str, object] = {}
        for (event_lane, family), values in self._model_events.items():
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

    def _target_algebra(self) -> TargetAlgebra:
        if self.modes[0].private_estimator is PrivateEstimator.SAMPLED:
            return TargetAlgebra(self.owners.plan.partitions, ())
        from pbg.core.targets import ConversionMap

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
            self.owners.plan.partitions,
            (
                ConversionMap(
                    "post",
                    "pre",
                    tuple(int(value) for value in classes.tolist()),
                ),
            ),
        )

    def prepare_rollout(self, *, iteration: int) -> RolloutLoop:
        owners = self.owners
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            torch.cuda.reset_peak_memory_stats(owners.plan.device)
            self._iteration_benchmark_started = time.perf_counter()
            self._iteration_host_synchronizations = 1
        count = owners.plan.manifest.n_replica
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
        machine = PokerStateMachine(self.geometry)
        input_to_output = tuple(
            tuple(
                output.name for output in owners.plan.partitions.outputs
            ).index(owners.plan.partitions.output_for(name).name)
            for name in input_names
        )
        publisher = AsyncTopologyPublisher(
            registry,
            p_train=(
                self.geometry.players
                if self.modes[0].training_regime
                is TrainingRegime.SELF_PLAY
                else 1
            ),
            input_names=input_names,
            input_to_output=input_to_output,
        )
        recorders: list[TopologyPacketStager] = []
        hooks_by_lane: list[BranchedSampledPokerHooks] = []
        depths_by_lane: list[dict[str, int]] = []
        rollout_caps: list[int] = []
        roots_by_lane: list[int] = []
        logical_bases: list[int] = []
        for lane in range(count):
            self._publish_root(
                lane, machine, queues["pre"], iteration=iteration
            )
            mode = self.modes[lane]
            recorder = publisher.stagers[lane]
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
                registry,
                queues,
                seed=self.seed,
                iteration=iteration,
                mode=mode,
                controller=self.controllers[lane],
                cost_safeguard=self.safeguards[lane],
                child_model=lambda forward, _lanes, selected=lane: (
                    self._inference(selected, forward)
                ),
                topology_stager=recorder,
                policy=self.policy,
                posterior_sampler=posterior,
            )
            roots = (
                owners.plan.manifest.root_count
                if self.run_states[lane] is None
                else self.run_states[lane].root_count
            )
            logical_base = (
                0
                if self.run_states[lane] is None
                else self.run_states[lane].attempt_logical_base
            )
            depths = {name: 0 for name in input_names}
            depths["pre"] = roots
            hooks.realized_tree_size.fill_(roots)
            player_beta_max = self.controllers[lane].max_degree(
                owner_is_player=True
            )
            chance_beta_max = self.controllers[lane].max_degree(
                owner_is_player=False
            )
            topology_rows_per_root = player_beta_max * sum(
                chance_beta_max**generation for generation in range(4)
            )
            if topology_rows_per_root > owners.plan.manifest.topology_packet_cap:
                raise ValueError(
                    "topology packet cannot cover one maximal branched root"
                )
            recorders.append(recorder)
            hooks_by_lane.append(hooks)
            depths_by_lane.append(depths)
            roots_by_lane.append(roots)
            logical_bases.append(logical_base)
            rollout_caps.append(
                min(
                    owners.plan.manifest.family_pack_cap,
                    owners.plan.manifest.topology_packet_cap
                    // topology_rows_per_root,
                )
            )

        scheduler = FamilyScheduler(
            input_names, owners.plan.manifest.family_pack_cap
        )
        last_events: list[torch.cuda.Event | None] = [None] * count
        self._iteration_last_commit_event: torch.cuda.Event | None = None
        self._iteration_pre_refill_nodes: list[int | None] = [
            None
        ] * count
        self._iteration_pre_refill_records: list[int | None] = [
            None
        ] * count
        self._iteration_pre_refill_correction: list[float | None] = [
            None
        ] * count
        self._iteration_refill_counts = [0] * count

        def queue_states(time: int) -> tuple[FamilyQueueState, ...]:
            return tuple(
                FamilyQueueState(
                    lane,
                    name,
                    index,
                    0,
                    depths_by_lane[lane][name],
                    time,
                    0,
                    self.modes[lane],
                    max_prefix=rollout_caps[lane],
                )
                for lane in range(count)
                for index, name in enumerate(input_names)
            )

        def rollout_failure(lane: int) -> Exception:
            failure = owners.failure_state.snapshot(lane)
            if (
                failure is not None
                and failure.failure_code == "rejection_exhausted"
            ):
                return BranchedPosteriorExhaustion(failure)
            if (
                failure is not None
                and failure.failure_code == "capacity_overflow"
            ):
                return RecoverableOverflow(
                    failure,
                    capacity=failure.available_extent,
                    observed_demand=failure.requested_extent,
                )
            return BranchedPokerError("branched rollout failed terminally")

        def accept_commits(pending: tuple[AcceptedRollout, ...]) -> None:
            for accepted in pending:
                lane = int(accepted.scheduled.lanes[0])
                depths = depths_by_lane[lane]
                depths[accepted.scheduled.input_type] -= (
                    accepted.scheduled.rows
                )
                for name, produced in zip(
                    input_names, accepted.produced, strict=True
                ):
                    depths[name] += produced
                last_events[lane] = accepted.commit.event
                self._iteration_last_commit_event = accepted.commit.event

        def has_work() -> bool:
            if any(any(depths.values()) for depths in depths_by_lane):
                return True
            for lane, recorder in enumerate(recorders):
                logical_base = logical_bases[lane]
                deficient = next(
                    (
                        root
                        for root in range(
                            logical_base,
                            logical_base + roots_by_lane[lane],
                        )
                        if recorder.tree_size(root)
                        < self.minimum_tree_sizes[lane]
                    ),
                    None,
                )
                if deficient is None:
                    continue
                if self._iteration_pre_refill_nodes[lane] is None:
                    self._iteration_pre_refill_nodes[lane] = (
                        recorder.sampled_tree_size()
                    )
                    self._iteration_pre_refill_records[lane] = (
                        recorder.record_count
                    )
                    self._iteration_pre_refill_correction[lane] = (
                        recorder.counterfactual_correction(
                            recorder.record_count
                        )
                    )
                hooks_by_lane[lane].begin_root_refill(
                    recorder.root_degree(deficient) + 1
                )
                self._publish_refill_root(
                    lane,
                    queues["pre"],
                    original_row=deficient - logical_base,
                    logical_root=deficient,
                )
                self._iteration_refill_counts[lane] += 1
                depths_by_lane[lane]["pre"] = 1
                return True
            return False

        self._iteration_iteration = iteration
        self._iteration_registry = registry
        self._iteration_input_names = input_names
        self._iteration_queues = queues
        self._iteration_publisher = publisher
        self._iteration_recorders = recorders
        self._iteration_hooks_by_lane = hooks_by_lane
        self._iteration_roots_by_lane = roots_by_lane
        self._iteration_logical_bases = logical_bases
        self._iteration_last_events = last_events
        self._iteration_failures: dict[int, str] = {}
        return RolloutLoop(
            scheduler=scheduler,
            states=queue_states,
            has_work=has_work,
            queues=queues,
            registry=registry,
            schemas=tuple(self.modes),
            seed=self.seed,
            models=tuple(
                (
                    lambda forward, _lanes, selected=lane: self._inference(
                        selected, forward
                    )
                )
                for lane in range(count)
            ),
            hooks=tuple(hooks_by_lane),
            accept=accept_commits,
            topology=publisher,
            failure=rollout_failure,
            # Produced depths become host-visible with the accepted topology
            # packet.  Select the next stable prefix only after that accept.
            inflight_cap=1,
            logical_limit=count * owners.plan.manifest.topology_cap,
        )

    def finish_rollout(
        self, *, logical_time: int, topology: SealedTopology
    ) -> None:
        del logical_time
        owners = self.owners
        topologies = topology.topologies
        bindings = topology.bindings
        leaf_counts = topology.leaf_counts
        sampled_nodes = topology.sampled_nodes
        counterfactual_sizes: list[float] = []
        for lane, _record in enumerate(topologies):
            sampled = sampled_nodes[lane]
            if self._iteration_pre_refill_nodes[lane] is None:
                self._iteration_pre_refill_nodes[lane] = sampled
                self._iteration_pre_refill_records[lane] = (
                    topology.record_counts[lane]
                )
                self._iteration_pre_refill_correction[lane] = (
                    topology.counterfactual_corrections[lane]
                )
            pre_refill_nodes = self._iteration_pre_refill_nodes[lane]
            pre_refill_records = self._iteration_pre_refill_records[lane]
            assert (
                pre_refill_nodes is not None
                and pre_refill_records is not None
            )
            correction = self._iteration_pre_refill_correction[lane]
            assert correction is not None
            counterfactual_sizes.append(
                (
                    float(pre_refill_nodes)
                    + correction
                )
                / self._iteration_roots_by_lane[lane]
            )
        self._iteration_plan = topology.plan
        self._iteration_topologies = topologies
        self._iteration_bindings = bindings
        self._iteration_leaf_counts = leaf_counts
        self._iteration_sampled_nodes = sampled_nodes
        self._iteration_counterfactual_sizes = tuple(
            counterfactual_sizes
        )
        self._iteration_registry = None
        self._iteration_queues = None
        self._iteration_publisher = None
        if self.benchmark_telemetry:
            self._iteration_routing_traffic = tuple(
                hook.routing_traffic_bytes
                for hook in self._iteration_hooks_by_lane
            )
        self._iteration_hooks_by_lane = None
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            self._iteration_host_synchronizations += 1
            self._iteration_rollout_seconds = (
                time.perf_counter() - self._iteration_benchmark_started
            )

    def finish_queue_drain(self) -> None:
        return None

    def prepare_completion(self) -> CompletionLoop:
        owners = self.owners
        count = owners.plan.manifest.n_replica
        # Completion admits native evaluator workspace against driver-visible
        # free memory. Return inactive rollout blocks held by PyTorch's cache.
        torch.cuda.synchronize(owners.plan.device)
        torch.cuda.empty_cache()
        completion = CompletionFIFO(
            owners.slab("Leaf"),
            RolloutOwnerRegistry.from_generated(owners),
            head=torch.zeros(
                count, dtype=torch.int64, device=owners.plan.device
            ),
            observed_epoch=torch.zeros(
                count, dtype=torch.int64, device=owners.plan.device
            ),
        )
        self._iteration_algebra = self._target_algebra()
        self._iteration_completion_binding = (
            SampledShowdownBinding(
                self.geometry,
                owners.slab("LayerWork", "all"),
                n_replica=count,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                utility=self.utility,
            )
            if self.modes[0].private_estimator
            is PrivateEstimator.SAMPLED
            else RBCompletionBinding(
                self.geometry,
                {
                    name: owners.slab("LayerWork", name)
                    for name in ("pre", "post")
                },
                self._iteration_algebra,
                mode=self.modes[0],
                n_replica=count,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                n_samples=self.completion_samples,
                utility=self.utility,
            )
        )
        if self._iteration_last_commit_event is None:
            raise BranchedPokerError("branched rollout produced no commit event")
        return CompletionLoop(
            completion=completion,
            remaining=self._iteration_leaf_counts,
            hook=self._iteration_completion_binding.leaf_to_layer_work,
            wait_for=self._iteration_last_commit_event,
            failure_message="branched replica completion failed",
        )

    def finish_completion(self) -> None:
        owners = self.owners
        count = owners.plan.manifest.n_replica
        plan = self._iteration_plan
        leaf_counts = self._iteration_leaf_counts
        completion_binding = self._iteration_completion_binding
        posterior_effective_hands = (
            None
            if not isinstance(completion_binding, RBCompletionBinding)
            else tuple(
                float(value)
                for value in completion_binding.effective_hands.detach().cpu().tolist()
            )
        )
        self._iteration_posterior_effective_hands = (
            posterior_effective_hands
        )
        del completion_binding
        for output, name in enumerate(plan.output_names):
            work = owners.slab("LayerWork", name)
            for lane in range(count):
                missing = plan.arena_rows[lane][output] - work.tail(lane)
                if missing < 0:
                    raise BranchedPokerError(
                        "branched replica completion exceeded its arena"
                    )
                if missing:
                    values = self._defaults(work, missing)
                    counts = torch.zeros(
                        count,
                        dtype=torch.int64,
                        device=owners.plan.device,
                    )
                    counts[lane] = missing
                    address = torch.full_like(
                        counts, torch.iinfo(torch.int64).max
                    )
                    address[lane] = leaf_counts[lane]
                    active = torch.arange(
                        count, device=owners.plan.device
                    ).eq(lane)
                    padding = PublicationTransaction.begin_device(
                        (
                            DeviceDestinationRequest(
                                work, counts, values, name, address
                            ),
                        ),
                        phase="branched_replica_target_padding",
                        active=active,
                    )
                    padding.fill_all()
                    torch._assert_async(
                        padding.publish()[lane],
                        "branched replica target padding failed",
                    )
        if self.distributed is not None:
            self.distributed.barrier()

    def align_and_seal(self) -> None:
        owners = self.owners
        input_names = self._iteration_input_names
        topologies = self._iteration_topologies
        bindings = self._iteration_bindings
        sealed: dict[tuple[int, str], SealedBranchedInputs] = {}
        for lane, (topology, binding) in enumerate(
            zip(topologies, bindings, strict=True)
        ):
            log_mass = decision_log_path_mass(
                topology.parent,
                topology.executed_degree,
                device=owners.plan.device,
            )
            for input_index, name in enumerate(input_names):
                selected = (
                    (
                        binding.input_type.eq(input_index)
                        & topology.owner.ge(0)
                    )
                    .nonzero()
                    .flatten()
                )
                sealed[(lane, name)] = SealedBranchedInputs.seal(
                    owners.slab("FamilyWork", name),
                    owners.slab("LearnerIndex", name),
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
                        0,
                        selected.to(
                            device=owners.plan.device,
                            dtype=torch.int64,
                        ),
                    ),
                    lane=lane,
                )
        self._iteration_sealed = sealed

    def execute_targets(self) -> None:
        owners = self.owners
        input_names = self._iteration_input_names
        target_started = (
            time.perf_counter() if self.benchmark_telemetry else 0.0
        )
        target_result = LayeredTargetExecutor(
            self.modes[0],
            self._iteration_algebra,
            lambda_=self.lambda_,
        ).execute(owners, self._iteration_plan)
        if target_result.failed_lanes:
            raise BranchedPokerError("branched replica targets failed")
        for lane, safeguard in enumerate(self.safeguards):
            if safeguard is None:
                continue
            predicted_cost: list[Tensor] = []
            target_cost: list[Tensor] = []
            for name in input_names:
                head = owners.slab("SigmaHead", name)
                target_slab = owners.slab("SigmaTarget", name)
                rows = head.tail(lane)
                if rows:
                    predicted_cost.append(
                        head.columns["old_cost"][lane, :rows]
                    )
                    target_cost.append(
                        target_slab.columns["cost_target"][lane, :rows]
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
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            self._iteration_host_synchronizations += 1
            self._iteration_target_seconds = (
                time.perf_counter() - target_started
            )

    def execute_learner(self, *, iteration: int) -> None:
        owners = self.owners
        count = owners.plan.manifest.n_replica
        input_names = self._iteration_input_names
        sealed = self._iteration_sealed

        def statistical_counts(
            lane: int, stores: Mapping[str, RaggedBranchedStore]
        ) -> Mapping[str, int] | None:
            allocation = self.allocation_inputs[lane]
            if allocation is None:
                return None
            counts = allocate_statistical_counts(*allocation)
            if counts.numel() != len(stores):
                raise BranchedPokerError(
                    "statistical allocation differs from active strata"
                )
            return MappingProxyType(
                {
                    name: int(counts[index])
                    for index, name in enumerate(stores)
                }
            )

        def validate_lane(lane: int) -> None:
            stores = {
                name: RaggedBranchedStore(
                    owners,
                    name,
                    sealed_inputs=sealed[(lane, name)],
                    lane=lane,
                )
                for name in input_names
                if sealed[(lane, name)].rows
            }
            sampler = BranchedBatchSampler(
                stores,
                self.schedule,
                generator=self.generators[lane],
                sigma_pack_cap=owners.plan.manifest.learner_sigma_pack_cap,
                action_pack_cap=owners.plan.manifest.learner_action_pack_cap,
                statistical_counts=statistical_counts(lane, stores),
            )
            module = self.models[lane]
            operation = (
                self.learner_operations[lane]
                if self.learner_operations is not None
                else (
                    module.learner_operation
                    if self.modes[lane].private_estimator
                    is PrivateEstimator.SAMPLED
                    else module.rb_learner_operation
                )
            )
            try:
                LearnerIteration(
                    model=module,
                    learner=LearnerOperation(
                        self.modes[lane],
                        self.losses[lane],
                        operation,
                        PokerLearnerPolicy(self.geometry, self.policy),
                        precision=self.precision,
                    ),
                    optimizer=LearnerPreflightOptimizer(
                        self.optimizers.optimizers[lane]
                    ),  # type: ignore[arg-type]
                    sigma_pack_cap=(
                        owners.plan.manifest.learner_sigma_pack_cap
                    ),
                    action_pack_cap=(
                        owners.plan.manifest.learner_action_pack_cap
                    ),
                ).optimize(
                    sampler,
                    iteration=iteration,
                    reference_operation=self.reference_operations[lane],
                )
            finally:
                sampler.release()

        try:
            IterationEngine.preflight_learners(
                generators=self.generators,
                validate_lane=validate_lane,
                clear_gradients=lambda: self.optimizers.zero_grad(range(count)),
                distributed=self.distributed,
                device=owners.plan.device,
                error_type=BranchedPokerError,
            )
        except Exception:
            for value in sealed.values():
                value.release()
            raise
        learner_results: list[object | None] = []
        for lane in range(count):
            try:
                stores = {
                    name: RaggedBranchedStore(
                        owners,
                        name,
                        sealed_inputs=sealed[(lane, name)],
                        lane=lane,
                    )
                    for name in input_names
                    if sealed[(lane, name)].rows
                }
                sampler = BranchedBatchSampler(
                    stores,
                    self.schedule,
                    generator=self.generators[lane],
                    sigma_pack_cap=(
                        owners.plan.manifest.learner_sigma_pack_cap
                    ),
                    action_pack_cap=(
                        owners.plan.manifest.learner_action_pack_cap
                    ),
                    statistical_counts=statistical_counts(lane, stores),
                )
                module = self.models[lane]
                model_operation = (
                    self.learner_operations[lane]
                    if self.learner_operations is not None
                    else (
                        module.learner_operation
                        if self.modes[lane].private_estimator
                        is PrivateEstimator.SAMPLED
                        else module.rb_learner_operation
                    )
                )
                learner = LearnerOperation(
                    self.modes[lane],
                    self.losses[lane],
                    model_operation,
                    PokerLearnerPolicy(self.geometry, self.policy),
                    precision=self.precision,
                )
                reference_operation = self.reference_operations[lane]
                reference_caches: dict[str, ReferenceOutputCache] | None = None
                if (
                    self.cache_reference_outputs
                    and reference_operation is not None
                ):
                    admitted: dict[str, ReferenceOutputCache] = {}
                    resident = 0
                    budget = owners.plan.layout.reference_cache_bytes
                    for name, store in stores.items():
                        columns = store.sealed_inputs.columns
                        candidate = build_reference_output_cache(
                            reference_operation,
                            input_type=name,
                            model_input=columns["model_input"],
                            sampled_private_state=columns[
                                "sampled_private_state"
                            ],
                            owner=columns["owner"],
                            pack_cap=(
                                owners.plan.manifest.learner_sigma_pack_cap
                            ),
                            precision=self.precision,
                            byte_budget=budget,
                            resident_bytes=resident,
                            predicted_saved_seconds=1.0,
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
                try:
                    optimized = LearnerIteration(
                        model=module,
                        learner=learner,
                        optimizer=self.optimizers.lane_stepper(
                            lane, distributed=self.distributed
                        ),  # type: ignore[arg-type]
                        sigma_pack_cap=(
                            owners.plan.manifest.learner_sigma_pack_cap
                        ),
                        action_pack_cap=(
                            owners.plan.manifest.learner_action_pack_cap
                        ),
                        run_state=self.run_states[lane],
                    ).optimize(
                        sampler,
                        iteration=iteration,
                        reference_operation=reference_operation,
                        reference_caches=reference_caches,
                    )
                finally:
                    if reference_caches is not None:
                        for cache in reference_caches.values():
                            cache.release()
                    sampler.release()
            except Exception as exc:
                raise BranchedPokerError(
                    "replica learner failed after successful preflight"
                ) from exc
            learner_results.append(optimized)
        for value in sealed.values():
            value.release()
        for lane, optimized in enumerate(learner_results):
            if optimized is None:
                continue
            self.controllers[lane].observe(
                self._iteration_counterfactual_sizes[lane],
                iteration=iteration,
            )
            state = self.run_states[lane]
            if state is not None:
                state.logical_work["trajectories"] += state.root_count
                state.branch_controller_state = self.controllers[
                    lane
                ].state_dict()
                state.cost_safeguard_state = (
                    None
                    if self.safeguards[lane] is None
                    else self.safeguards[lane].state_dict()
                )
        self._iteration_learner_results = learner_results

    def finish_backward(self) -> ReplicaBranchedPokerResult:
        owners = self.owners
        topologies = self._iteration_topologies
        bindings = self._iteration_bindings
        leaf_counts = self._iteration_leaf_counts
        sampled_nodes = self._iteration_sampled_nodes
        posterior_effective_hands = (
            self._iteration_posterior_effective_hands
        )
        plan = self._iteration_plan
        learner_results = self._iteration_learner_results
        results: list[BranchedPokerResult | None] = []
        for lane, topology in enumerate(topologies):
            optimized = learner_results[lane]
            if optimized is None:
                results.append(None)
                continue
            duplicate_labels = 0
            for node in range(topology.node_count):
                begin = int(topology.child_begin[node])
                degree = int(topology.executed_degree[node])
                labels = topology.label[begin : begin + degree]
                duplicate_labels += degree - int(labels.unique().numel())
            telemetry = None
            if self.benchmark_telemetry:
                model_seconds = sum(
                    float(value["seconds"])
                    for value in self._model_timing(lane).values()
                )
                rollout_seconds = self._iteration_rollout_seconds
                target_seconds = self._iteration_target_seconds
                peak_allocated = torch.cuda.max_memory_allocated(
                    owners.plan.device
                )
                peak_reserved = torch.cuda.max_memory_reserved(
                    owners.plan.device
                )
                telemetry = BranchedBenchmarkTelemetry(
                    rollout_seconds=rollout_seconds,
                    routing_traffic_bytes=self._iteration_routing_traffic[lane],
                    host_synchronizations=(
                        self._iteration_host_synchronizations
                    ),
                    child_reservation_commit_seconds=max(
                        rollout_seconds - model_seconds,
                        torch.finfo(torch.float64).eps,
                    ),
                    target_propagation_seconds=max(
                        target_seconds, torch.finfo(torch.float64).eps
                    ),
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                    model_inference_seconds=model_seconds,
                    target_kernel_cuda_seconds=target_seconds,
                    rollout_peak_allocated_bytes=peak_allocated,
                    completion_peak_allocated_bytes=peak_allocated,
                    align_peak_allocated_bytes=peak_allocated,
                    targets_peak_allocated_bytes=peak_allocated,
                    model_stage_peak_allocated_bytes=peak_allocated,
                    sampling_stage_peak_allocated_bytes=peak_allocated,
                    transition_stage_peak_allocated_bytes=peak_allocated,
                )
            results.append(
                BranchedPokerResult(
                    optimized,
                    player_nodes=int(topology.owner.ge(0).sum()),
                    leaves=leaf_counts[lane],
                    target_layers=plan.layer_count,
                    executed_edges=topology.edge_count,
                    root_executed_degree=tuple(
                        int(value)
                        for value in topology.executed_degree[
                            topology.parent.eq(-1)
                        ].tolist()
                    ),
                    root_target_degree=tuple(
                        int(value)
                        for value in topology.target_degree[
                            topology.parent.eq(-1)
                        ].tolist()
                    ),
                    duplicate_child_labels=duplicate_labels,
                    sampled_tree_nodes=sampled_nodes[lane],
                    posterior_effective_hands=(
                        None
                        if posterior_effective_hands is None
                        else posterior_effective_hands[lane]
                    ),
                    telemetry=telemetry,
                    observability=MappingProxyType(
                        {
                            **topology_diagnostics(
                                topology,
                                bindings[lane],
                                owners.plan.partitions,
                            ),
                            "model_timing_by_family": dict(
                                self._model_timing(lane)
                            ),
                        }
                    ),
                )
            )
        return ReplicaBranchedPokerResult(
            tuple(results),
            MappingProxyType(self._iteration_failures),
            tuple(
                int(value) for value in self.optimizers.versions.tolist()
            ),
        )

__all__ = [
    "BranchedBenchmarkTelemetry",
    "BranchedPokerError",
    "BranchedPokerResult",
    "BranchedPosteriorExhaustion",
    "BranchedSampledPokerHooks",
    "PokerBranchedProgram",
    "ReplicaBranchedPokerResult",
    "action_conditioned_log_reach",
    "branched_rb_chance_frontier_row_bytes",
    "sampled_branched_chance_frontier_row_bytes",
    "sampled_branched_topology_packet_row_bytes",
]
