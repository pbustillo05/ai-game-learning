"""Sampled and Rao--Blackwell branched NLHE through one queued engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType

import torch
from torch import Tensor

from pbg.core.branching import (
    BranchController,
    BranchDecision,
    CostSafeguardController,
    child_path_factors,
)
from pbg.core.capacity import (
    BRANCHED_TOPOLOGY_PACKET_ROW_BYTES,
)
from pbg.core.contracts import (
    ModeSchema,
    PrivateEstimator,
    TrainingRegime,
)
from pbg.core.rollout import (
    BoundedChanceClosure,
    CommitCoverage,
    CommitPlan,
    DeviceCommitFailure,
    DrawPurpose,
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
)
from pbg.core.topology import (
    TopologyExpansionRecord,
    TopologyPacketStager,
)
from pbg.games.poker.queued.hooks import (
    FOLD,
    PAD_CARD,
    NLHEGeometry,
    PokerForwardInput,
    PokerStateMachine,
    SequentialPokerHooks,
)
from pbg.games.poker.queued.policy import PokerPolicyOperator
from pbg.games.poker.queued.rb_sampling import (
    CollisionConditionedProfileSampler,
)


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


__all__ = [
    "BranchedPokerError",
    "BranchedPosteriorExhaustion",
    "BranchedSampledPokerHooks",
    "action_conditioned_log_reach",
    "branched_rb_chance_frontier_row_bytes",
    "sampled_branched_chance_frontier_row_bytes",
    "sampled_branched_topology_packet_row_bytes",
]
