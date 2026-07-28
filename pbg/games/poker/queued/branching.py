"""Sampled-private branched NLHE through the queued transaction engine."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch
from torch import Tensor, nn

from pbg.games.poker.queued.hooks import (
    PAD_CARD,
    NLHEGeometry,
    PokerForwardInput,
    PokerLearnerPolicy,
    PokerStateMachine,
    SampledShowdownBinding,
    SequentialPokerHooks,
)
from pbg.games.poker.queued.policy import PokerPolicyOperator
from pbg.queued.branching import (
    BranchController,
    BranchDecision,
    BranchedOverflowRecovery,
    CostSafeguardController,
    RankCommonFailureAgreement,
    child_path_factors,
    controller_from_state_dict,
    decision_path_mass,
)
from pbg.queued.capacity import ExecutionPhase, GeneratedOwners, StoragePlan
from pbg.queued.contracts import (
    ModeSchema,
    PrivateEstimator,
    TrainingRegime,
)
from pbg.queued.learner import (
    BranchedBatchSampler,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    ModelOperation,
    RaggedBranchedStore,
    SealedBranchedInputs,
)
from pbg.queued.optimization import DenseAdam
from pbg.queued.persistence import RunState
from pbg.queued.rollout import (
    AsyncTopologyPublisher,
    CommitCoverage,
    CommitPlan,
    CompletionFIFO,
    DeferredBranchedTopologyPublication,
    DrawPurpose,
    FamilyQueueState,
    FamilyScheduler,
    RolloutOwnerRegistry,
    RolloutTransaction,
    SampleRequest,
    SampleWorkspace,
    SemanticFamilyFIFO,
    TransitionRequest,
    child_path_keys,
    logical_rng_keys,
    stable_route,
)
from pbg.queued.storage import (
    DeviceDestinationRequest,
    MonotoneSlab,
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
    BranchedSampledIteration,
    RecoverableOverflow,
)


class BranchedPokerError(RuntimeError):
    """The sampled branched poker composition violated its contract."""


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


@dataclass(frozen=True)
class _ExpansionRecord:
    parent_key: Tensor
    child_key: Tensor
    source_ordinal: Tensor
    terminal: Tensor
    child_input: Tensor
    child_leaf_row: Tensor
    label: Tensor
    owner: Tensor
    input_type: Tensor
    familywork_row: Tensor
    sigma_head_row: Tensor
    action_begin: Tensor
    executed_degree: Tensor
    nominal_degree: Tensor
    target_degree: Tensor
    trained: Tensor
    refill: Tensor
    path_length: Tensor


@dataclass(frozen=True)
class _StagedBranchedPayload:
    records: tuple[_ExpansionRecord, ...]
    produced: tuple[int, ...]
    sampled_nodes: Tensor
    counterfactual_correction: Tensor


class BranchedTopologyRecorder:
    """Commit-visible logical expansions assembled into compact CPU topology."""

    def __init__(
        self,
        *,
        input_names: tuple[str, ...],
        p_train: int,
        topology_cap: int,
        topology_edge_cap: int,
    ) -> None:
        self.input_names = input_names
        self.p_train = p_train
        self.topology_cap = topology_cap
        self.topology_edge_cap = topology_edge_cap
        self._records: list[_ExpansionRecord] = []
        self._record_corrections: list[float] = []
        self._node_capacity = topology_cap + topology_edge_cap
        self._node_tail = 0
        self._edge_tail = 0
        self._feedback_tail = 0
        self._node_buffers = {
            "parent_key": torch.empty(
                (self._node_capacity, 2),
                dtype=torch.int64,
                pin_memory=True,
            ),
            **{
                name: torch.empty(
                    self._node_capacity,
                    dtype=dtype,
                    pin_memory=True,
                )
                for name, dtype in (
                    ("owner", torch.int32),
                    ("input_type", torch.int32),
                    ("familywork_row", torch.int32),
                    ("sigma_head_row", torch.int32),
                    ("action_begin", torch.int32),
                    ("executed_degree", torch.int32),
                    ("nominal_degree", torch.int32),
                    ("target_degree", torch.int32),
                    ("trained", torch.bool),
                    ("refill", torch.bool),
                )
            },
        }
        self._edge_buffers = {
            "child_key": torch.empty(
                (topology_edge_cap, 2),
                dtype=torch.int64,
                pin_memory=True,
            ),
            **{
                name: torch.empty(
                    topology_edge_cap,
                    dtype=dtype,
                    pin_memory=True,
                )
                for name, dtype in (
                    ("source_ordinal", torch.int64),
                    ("terminal", torch.bool),
                    ("child_input", torch.int32),
                    ("child_leaf_row", torch.int64),
                    ("label", torch.int32),
                    ("path_length", torch.int32),
                )
            },
        }
        self._sampled_nodes = torch.empty(
            self._node_capacity, dtype=torch.int64, pin_memory=True
        )
        self._correction = torch.empty(
            self._node_capacity, dtype=torch.float32, pin_memory=True
        )

    def accept_payload(self, payload: _StagedBranchedPayload) -> None:
        if not payload.records:
            raise BranchedPokerError("branched topology payload is empty")
        self._records.extend(payload.records)
        self._record_corrections.extend(
            (
                float(payload.counterfactual_correction),
                *(0.0 for _ in payload.records[1:]),
            )
        )

    def stage_payload(
        self,
        records: tuple[_ExpansionRecord, ...],
        *,
        produced: tuple[int, ...],
        sampled_nodes: Tensor,
        counterfactual_correction: Tensor,
    ) -> _StagedBranchedPayload:
        node_rows = sum(record.parent_key.shape[0] for record in records)
        edge_rows = sum(record.child_key.shape[0] for record in records)
        if (
            not records
            or self._node_tail + node_rows > self._node_capacity
            or self._edge_tail + edge_rows > self.topology_edge_cap
            or self._feedback_tail >= self._node_capacity
        ):
            raise BranchedPokerError(
                "branched topology exceeds its fixed pinned arena"
            )
        staged: list[_ExpansionRecord] = []
        for record in records:
            node_count = record.parent_key.shape[0]
            edge_count = record.child_key.shape[0]
            node_slice = slice(self._node_tail, self._node_tail + node_count)
            edge_slice = slice(self._edge_tail, self._edge_tail + edge_count)
            values: dict[str, Tensor] = {}
            for name, buffer in self._node_buffers.items():
                destination = buffer[node_slice]
                destination.copy_(getattr(record, name), non_blocking=True)
                values[name] = destination
            for name, buffer in self._edge_buffers.items():
                destination = buffer[edge_slice]
                destination.copy_(getattr(record, name), non_blocking=True)
                values[name] = destination
            staged.append(_ExpansionRecord(**values))
            self._node_tail += node_count
            self._edge_tail += edge_count
        feedback = self._feedback_tail
        self._sampled_nodes[feedback].copy_(
            sampled_nodes.detach(), non_blocking=True
        )
        self._correction[feedback].copy_(
            counterfactual_correction.detach(), non_blocking=True
        )
        self._feedback_tail += 1
        return _StagedBranchedPayload(
            tuple(staged),
            produced,
            self._sampled_nodes[feedback],
            self._correction[feedback],
        )

    def materialize_cpu(self) -> None:
        if any(
            getattr(record, name).device.type != "cpu"
            for record in self._records
            for name in record.__dataclass_fields__
        ):
            raise BranchedPokerError(
                "committed topology was not staged in CPU storage"
            )

    def release(self) -> None:
        """Release the fixed pinned topology arena at its rollout consumer."""

        self._records.clear()
        self._record_corrections.clear()
        self._node_buffers.clear()
        self._edge_buffers.clear()
        self._sampled_nodes = torch.empty(0, dtype=torch.int64)
        self._correction = torch.empty(0, dtype=torch.float32)

    def seal(self) -> tuple[Topology, TopologyBindings, int]:
        if not self._records:
            raise BranchedPokerError("branched topology is empty")
        keys: list[tuple[int, int]] = []
        record_for_key: dict[tuple[int, int], list[tuple[_ExpansionRecord, int]]] = {}
        for record in self._records:
            for row, values in enumerate(record.parent_key.tolist()):
                key = (int(values[0]), int(values[1]))
                if key in record_for_key:
                    if not bool(record.refill[row]):
                        raise BranchedPokerError(
                            "a branched logical node was expanded twice"
                        )
                    record_for_key[key].append((record, row))
                else:
                    if bool(record.refill[row]):
                        raise BranchedPokerError(
                            "root refill preceded its ordinary expansion"
                        )
                    record_for_key[key] = [(record, row)]
                    keys.append(key)
        ordered_keys: list[tuple[int, int]] = []
        visited: set[tuple[int, int]] = set()

        def visit(key: tuple[int, int]) -> None:
            if key in visited:
                return
            visited.add(key)
            ordered_keys.append(key)
            for expansion, row in record_for_key[key]:
                edges = expansion.source_ordinal.eq(row).nonzero().flatten()
                for edge in edges.tolist():
                    if not bool(expansion.terminal[edge]):
                        child = expansion.child_key[edge].tolist()
                        visit((int(child[0]), int(child[1])))

        for root_key in sorted(key for key in keys if key[1] == 0):
            visit(root_key)
        if len(visited) != len(record_for_key):
            raise BranchedPokerError(
                "branched topology contains an unreachable logical node"
            )
        keys = ordered_keys
        node_for_key = {key: node for node, key in enumerate(keys)}
        parent = [-1] * len(keys)
        child_begin: list[int] = []
        executed: list[int] = []
        nominal: list[int] = []
        target: list[int] = []
        owner: list[int] = []
        family: list[int] = []
        sigma: list[int] = []
        action_begin: list[int] = []
        input_type: list[int] = []
        output_type: list[int] = []
        sigma_target: list[int] = []
        labels: list[int] = []
        path_length: list[int] = []
        target_kind: list[int] = []
        target_row: list[int] = []
        child_output: list[int] = []
        leaves: set[int] = set()
        for node, key in enumerate(keys):
            expansions = record_for_key[key]
            record, row = expansions[0]
            edge_groups = [
                item.source_ordinal.eq(item_row).nonzero().flatten()
                for item, item_row in expansions
            ]
            degrees = [
                int(item.executed_degree[item_row]) for item, item_row in expansions
            ]
            if any(
                edge_rows.numel() != degree
                for edge_rows, degree in zip(edge_groups, degrees, strict=True)
            ):
                raise BranchedPokerError("recorded branch degree is incomplete")
            degree = sum(degrees)
            child_begin.append(len(labels))
            executed.append(degree)
            nominal.append(int(record.nominal_degree[row]))
            target.append(int(record.target_degree[row]))
            trained = bool(record.trained[row])
            owner.append(int(record.owner[row]) if trained else -1)
            family.append(int(record.familywork_row[row]) if trained else -1)
            sigma.append(int(record.sigma_head_row[row]) if trained else -1)
            action_begin.append(int(record.action_begin[row]) if trained else -1)
            input_index = int(record.input_type[row])
            input_type.append(input_index if trained else -1)
            output_type.append(0)
            sigma_target.append(int(record.sigma_head_row[row]) if trained else -1)
            for expansion, edge_rows in zip(
                (item for item, _ in expansions),
                edge_groups,
                strict=True,
            ):
                for edge_row in edge_rows.tolist():
                    labels.append(int(expansion.label[edge_row]))
                    path_length.append(int(expansion.path_length[edge_row]))
                    child_output.append(0)
                    if bool(expansion.terminal[edge_row]):
                        leaf = int(expansion.child_leaf_row[edge_row])
                        if leaf < 0:
                            raise BranchedPokerError(
                                "terminal branch has no committed leaf row"
                            )
                        target_kind.append(int(TargetKind.LEAF))
                        target_row.append(leaf)
                        leaves.add(leaf)
                    else:
                        child_values = expansion.child_key[edge_row].tolist()
                        child_key = (
                            int(child_values[0]),
                            int(child_values[1]),
                        )
                        try:
                            child = node_for_key[child_key]
                        except KeyError as exc:
                            raise BranchedPokerError(
                                "branched child was never expanded"
                            ) from exc
                        if parent[child] != -1:
                            raise BranchedPokerError(
                                "branched child has two structural parents"
                            )
                        parent[child] = node
                        target_kind.append(int(TargetKind.TOPOLOGY))
                        target_row.append(child)

        def i32(values: list[int]) -> Tensor:
            return torch.tensor(values, dtype=torch.int32)

        topology = Topology(
            parent=i32(parent),
            child_begin=i32(child_begin),
            executed_degree=i32(executed),
            nominal_degree=i32(nominal),
            target_degree=i32(target),
            owner=i32(owner),
            familywork_row=i32(family),
            sigma_head_row=i32(sigma),
            action_begin=i32(action_begin),
            label=i32(labels),
            path_length=i32(path_length),
            target_kind=i32(target_kind),
            target_row=i32(target_row),
        )
        bindings = TopologyBindings(
            input_type=i32(input_type),
            output_type=i32(output_type),
            sigma_target_row=i32(sigma_target),
            child_output_type=i32(child_output),
        )
        topology.validate(bindings, p_train=self.p_train, _allow_zero_path=True)
        topology, bindings = topology.compact(
            bindings, p_train=self.p_train, _allow_zero_path=True
        )
        if topology.node_count > self.topology_cap:
            raise BranchedPokerError("realized topology exceeds its fixed cap")
        if topology.edge_count > self.topology_edge_cap:
            raise BranchedPokerError("realized edges exceed their fixed cap")
        return topology, bindings, len(leaves)

    def tree_size(self, logical_root: int) -> int:
        nodes = 1
        for record in self._records:
            parent_selected = record.parent_key[:, 0].eq(logical_root)
            edge_selected = parent_selected.index_select(
                0, record.source_ordinal.to(torch.int64)
            )
            nodes += int(record.path_length[edge_selected].sum())
        return nodes

    def sampled_tree_size(self, record_limit: int | None = None) -> int:
        records = (
            self._records
            if record_limit is None
            else self._records[:record_limit]
        )
        roots = {
            int(record.parent_key[row, 0])
            for record in records
            for row in range(record.parent_key.shape[0])
            if int(record.parent_key[row, 1]) == 0
        }
        return len(roots) + sum(
            int(record.path_length.sum()) for record in records
        )

    def root_degree(self, logical_root: int) -> int:
        degree = 0
        for record in self._records:
            selected = record.parent_key[:, 0].eq(logical_root) & record.parent_key[
                :, 1
            ].eq(0)
            if bool(selected.any()):
                degree += int(record.executed_degree[selected].sum())
        if degree <= 0:
            raise BranchedPokerError("logical root has no committed expansion")
        return degree

    @property
    def record_count(self) -> int:
        return len(self._records)

    def counterfactual_correction(self, record_limit: int) -> float:
        if not 0 <= record_limit <= len(self._records):
            raise ValueError("counterfactual record limit is invalid")
        return sum(self._record_corrections[:record_limit])


class BranchedSampledPokerHooks(SequentialPokerHooks):
    """Sample ragged duplicate-preserving children from one profile per tree."""

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
        topology_recorder: BranchedTopologyRecorder,
        policy: PokerPolicyOperator,
    ) -> None:
        if not mode.rollout_mode.is_branched:
            raise ValueError("branched hooks require branched mode")
        if mode.private_estimator is not PrivateEstimator.SAMPLED:
            raise ValueError("Phase 7 hooks require sampled private state")
        # Initialize the shared helper vocabulary without its sequential guard.
        self.mode = mode
        self.geometry = geometry
        self.machine = PokerStateMachine(geometry)
        self.owners = owners
        self.queues = MappingProxyType(dict(queues))
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.policy = policy
        self.probe_cutoff = False
        self.input_index = {name: index for index, name in enumerate(self.queues)}
        self._route_family_values = (-1, *range(len(self.queues)))
        self._route_families = torch.tensor(
            self._route_family_values,
            dtype=torch.int32,
            device=owners.device,
        )
        self.controller = controller
        self.cost_safeguard = cost_safeguard
        self.child_model = child_model
        self.topology_recorder = topology_recorder
        self.frontier_cost_forecast = torch.zeros(
            (), dtype=torch.float32, device=owners.device
        )
        self.realized_tree_size = 0
        self.last_topology: DeferredBranchedTopologyPublication | None = None
        self._topology_sequence = 0
        self._count_feedback = torch.empty(
            (), dtype=torch.int64, pin_memory=True
        )
        self._refill = False
        self._refill_child_ordinal = 0

    def begin_root_refill(self, child_ordinal: int) -> None:
        if self._refill or child_ordinal <= 0:
            raise BranchedPokerError("invalid or overlapping root refill")
        self._refill = True
        self._refill_child_ordinal = int(child_ordinal)

    def _pinned_count(self, value: Tensor) -> int:
        if value.device.type != "cuda" or value.numel() != 1:
            raise BranchedPokerError("dynamic count must be one CUDA scalar")
        self._count_feedback.copy_(
            value.to(torch.int64), non_blocking=True
        )
        ready = torch.cuda.Event(blocking=False)
        ready.record(torch.cuda.current_stream(value.device))
        while not ready.query():
            time.sleep(0)
        return int(self._count_feedback)

    def sample(self, request: SampleRequest) -> SampleWorkspace:
        batch = request.batch
        if batch.lane_count != 1:
            raise BranchedPokerError("ordinary Phase 7 hooks require one lane")
        rows = batch.rows
        root = batch.fields["logical_rng_key"][0, :, 0]
        path = batch.fields["logical_rng_key"][0, :, 1]
        values = request.model.values
        if "trace" not in values or "cost" not in values:
            raise BranchedPokerError("branched rollout omitted T/C heads")
        missing_frontier_cost = values["cost"].detach().float().masked_fill(
            batch.fields["frontier_cost_contribution"][0].gt(0), 0
        ).sum()
        decision = self.controller.decide(
            schema=self.mode,
            root=root,
            path=path,
            owner_is_player=torch.ones(rows, dtype=torch.bool, device=root.device),
            iteration=self.iteration,
            seed=self.seed,
            retry_epoch=self.owners.retry_epoch()[0],
            trace=values["trace"],
            cost=values["cost"],
            log_player_ancestor_sum=batch.fields["log_player_ancestor_sum"][0],
            log_ancestor_product=batch.fields["log_ancestor_product"][0],
            frontier_cost=(
                torch.full_like(values["cost"], self.realized_tree_size)
                + self.frontier_cost_forecast
                + missing_frontier_cost
                if (self.cost_safeguard is not None and self.cost_safeguard.active)
                else None
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
        total = self._pinned_count(total_tensor)
        source = torch.repeat_interleave(
            torch.arange(rows, device=root.device),
            degree,
            output_size=total,
        )
        prefix = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=root.device),
                degree.cumsum(0)[:-1],
            )
        )
        ordinal = torch.arange(total, device=root.device) - torch.repeat_interleave(
            prefix, degree
        )
        child_ordinal = (
            torch.full_like(ordinal, self._refill_child_ordinal)
            if self._refill
            else ordinal + 1
        )
        child_path = child_path_keys(path.index_select(0, source), child_ordinal)
        expanded_input = self._forward_model_input(batch).index_select(0, source)
        private = batch.fields["sampled_private_state"][0].index_select(0, source)
        owner = batch.fields["owner"][0].index_select(0, source).to(torch.int64)
        state = self.machine.state_from_model_input(expanded_input, private)
        raw = {
            name: value.index_select(0, source)
            for name, value in values.items()
            if name not in ("value", "trace", "cost")
        }
        actor_hand = private.gather(1, owner[:, None]).squeeze(1)
        action_key = logical_rng_keys(
            seed=self.seed,
            iteration=self.iteration,
            root=root.index_select(0, source),
            path=child_path,
            draw=0,
            purpose=DrawPurpose.ACTION,
            retry_epoch=self.owners.retry_epoch()[0],
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
                    "child_path": child_path,
                    "beta": decision.beta,
                    "nominal_degree": decision.nominal_degree,
                    "executed_degree": decision.executed_degree,
                    "target_degree": decision.target_degree,
                }
            ),
            total_tensor.reshape(1),
        )

    def transition(self, request: TransitionRequest) -> CommitPlan:
        batch = request.batch
        rows = batch.rows
        device = self.owners.device
        sample = request.sample.values
        source = sample["source"].to(torch.int64)
        action_source = source
        action_total = int(action_source.numel())
        flat_input = self._forward_model_input(batch)
        parent_private = batch.fields["sampled_private_state"][0]
        parent_state = self.machine.state_from_model_input(flat_input, parent_private)
        state = replace(
            parent_state,
            **{
                name: value.index_select(0, source)
                for name, value in parent_state.__dict__.items()
                if isinstance(value, Tensor)
            },
        )
        private = parent_private.index_select(0, source)
        state = replace(state, combo_ids=private)
        action = sample["action"]
        log_reach, history = self.machine.model_context(
            flat_input.index_select(0, source)
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
        actor = state.to_act.to(torch.int64)
        actor_reach = log_reach.gather(
            1,
            actor[:, None, None].expand(-1, 1, self.geometry.private_hands),
        ).squeeze(1)
        log_reach = log_reach.scatter(
            1,
            actor[:, None, None].expand(-1, 1, self.geometry.private_hands),
            (actor_reach + selected_log)[:, None],
        )
        history = self.machine.append_player_history(state, action, history)
        next_state = self.machine.step_player(state, action)
        root = batch.fields["logical_rng_key"][0, :, 0].index_select(0, action_source)
        child_path = sample["child_path"]
        retry = self.owners.retry_epoch()[0]
        first_child_key = torch.stack((root, child_path), -1)
        endpoint_source = action_source
        endpoint_mass = (
            batch.fields["path_mass"][0].index_select(0, action_source)
            / sample["executed_degree"].index_select(0, action_source).float()
        )
        endpoint_log_sum, endpoint_log_product = child_path_factors(
            batch.fields["log_player_ancestor_sum"][0].index_select(0, action_source),
            batch.fields["log_ancestor_product"][0].index_select(0, action_source),
            sample["beta"].index_select(0, action_source),
            child_is_player=torch.zeros(action_total, dtype=torch.bool, device=device),
        )
        chance_records: list[_ExpansionRecord] = []
        current_key = first_child_key
        current_state = next_state
        current_private = private
        current_history = history
        current_log_reach = log_reach
        current_beta = torch.ones(action_total, dtype=torch.float32, device=device)
        sampled_nodes = torch.tensor(
            action_total, dtype=torch.int64, device=device
        )
        endpoint_record = torch.zeros(
            action_total, dtype=torch.int64, device=device
        )
        endpoint_record_edge = torch.arange(
            action_total, dtype=torch.int64, device=device
        )
        for generation in range(3):
            chance = self.machine.needs_chance(current_state)
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
            )
            ones = torch.ones_like(chance_decision.executed_degree)
            chance_decision = BranchDecision(
                torch.where(
                    chance,
                    chance_decision.beta,
                    torch.ones_like(chance_decision.beta),
                ),
                torch.where(chance, chance_decision.nominal_degree, ones),
                torch.where(chance, chance_decision.executed_degree, ones),
                torch.where(chance, chance_decision.target_degree, ones),
            )
            degree = chance_decision.executed_degree.to(torch.int64)
            sampled_nodes += degree.masked_select(chance).sum()
            chance_total = self._pinned_count(degree.sum())
            chance_source = torch.repeat_interleave(
                torch.arange(
                    current_key.shape[0],
                    dtype=torch.int64,
                    device=device,
                ),
                degree,
                output_size=chance_total,
            )
            prefix = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    degree.cumsum(0)[:-1],
                )
            )
            ordinal = torch.arange(
                chance_total, dtype=torch.int64, device=device
            ) - torch.repeat_interleave(prefix, degree)
            next_path = child_path_keys(
                current_key[:, 1].index_select(0, chance_source),
                ordinal + 1,
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
            chance_parent_rows = chance.nonzero().flatten()
            chance_edge_rows = chance.index_select(
                0, chance_source
            ).nonzero().flatten()
            if chance_parent_rows.numel():
                compact_parent = chance.to(torch.int64).cumsum(0) - 1
                record_source = compact_parent.index_select(
                    0,
                    chance_source.index_select(0, chance_edge_rows),
                )
                parent_count = chance_parent_rows.numel()
                edge_count = chance_edge_rows.numel()
                record_identifier = 1 + len(chance_records)
                next_endpoint_record[chance_edge_rows] = record_identifier
                next_endpoint_record_edge[chance_edge_rows] = torch.arange(
                    edge_count, dtype=torch.int64, device=device
                )
                chance_records.append(
                    _ExpansionRecord(
                        parent_key=current_key.index_select(
                            0, chance_parent_rows
                        ),
                        child_key=next_key.index_select(0, chance_edge_rows),
                        source_ordinal=record_source,
                        terminal=terminal_child.index_select(
                            0, chance_edge_rows
                        ),
                        child_input=self.machine.input_type(
                            dealt_state
                        ).index_select(0, chance_edge_rows),
                        child_leaf_row=torch.full(
                            (edge_count,),
                            -1,
                            dtype=torch.int64,
                            device=device,
                        ),
                        label=chance_label.index_select(0, chance_edge_rows),
                        owner=torch.full(
                            (parent_count,),
                            -1,
                            dtype=torch.int32,
                            device=device,
                        ),
                        input_type=torch.full(
                            (parent_count,),
                            -1,
                            dtype=torch.int32,
                            device=device,
                        ),
                        familywork_row=torch.full(
                            (parent_count,),
                            -1,
                            dtype=torch.int32,
                            device=device,
                        ),
                        sigma_head_row=torch.full(
                            (parent_count,),
                            -1,
                            dtype=torch.int32,
                            device=device,
                        ),
                        action_begin=torch.full(
                            (parent_count,),
                            -1,
                            dtype=torch.int32,
                            device=device,
                        ),
                        executed_degree=chance_decision.executed_degree.index_select(
                            0, chance_parent_rows
                        ),
                        nominal_degree=chance_decision.nominal_degree.index_select(
                            0, chance_parent_rows
                        ),
                        target_degree=chance_decision.target_degree.index_select(
                            0, chance_parent_rows
                        ),
                        trained=torch.zeros(
                            parent_count,
                            dtype=torch.bool,
                            device=device,
                        ),
                        refill=torch.zeros(
                            parent_count,
                            dtype=torch.bool,
                            device=device,
                        ),
                        path_length=torch.ones(
                            edge_count,
                            dtype=torch.int32,
                            device=device,
                        ),
                    )
                )
            endpoint_source = endpoint_source.index_select(0, chance_source)
            endpoint_mass = (
                endpoint_mass.index_select(0, chance_source)
                / degree.index_select(0, chance_source).float()
            )
            endpoint_log_sum, endpoint_log_product = child_path_factors(
                endpoint_log_sum.index_select(0, chance_source),
                endpoint_log_product.index_select(0, chance_source),
                chance_decision.beta.index_select(0, chance_source),
                child_is_player=(
                    ~dealt_state.terminal & ~self.machine.needs_chance(dealt_state)
                    if final_generation
                    else torch.zeros(chance_total, dtype=torch.bool, device=device)
                ),
            )
            current_key = next_key
            current_state = dealt_state
            current_private = expanded_private
            current_history = chance_history
            current_log_reach = current_log_reach.index_select(0, chance_source)
            current_beta = chance_decision.beta.index_select(0, chance_source)
            endpoint_record = next_endpoint_record
            endpoint_record_edge = next_endpoint_record_edge
        next_state = current_state
        private = current_private
        history = current_history
        log_reach = current_log_reach
        root = current_key[:, 0]
        child_path = current_key[:, 1]
        total = int(root.numel())
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
        route_family = torch.where(
            terminal,
            torch.full_like(next_family, -1),
            next_family,
        )[None]
        route = stable_route(
            route_family,
            torch.ones_like(route_family, dtype=torch.bool),
            self._route_family_values,
            declared_values=self._route_families,
        )
        address = batch.fields["logical_rng_key"][:, :, 0].amin(1)
        destinations: list[DeviceDestinationRequest] = []
        owner = batch.fields["owner"][0].to(torch.int64)
        selected_player = (
            batch.fields["role_selector"][0].to(torch.int64)
            if self.mode.training_regime is TrainingRegime.BEST_RESPONSE
            else owner
        )
        leaf = next(slab for slab in self.owners.slabs if slab.schema.name == "Leaf")
        leaf_old_tail = leaf.reserved_tails_device()[0]
        leaf_values = self._defaults(leaf, total)
        leaf_source = self.machine.sampled_leaf(
            replace(
                next_state,
                terminal=torch.ones_like(next_state.terminal),
            )
        )
        if self.mode.training_regime is TrainingRegime.BEST_RESPONSE:
            leaf_source = MappingProxyType(
                {
                    **dict(leaf_source),
                    "selected_player": selected_player.index_select(
                        0, endpoint_source
                    ).to(torch.int32),
                }
            )
        terminal_rows = route.source_row[0, 0]
        valid_terminal = route.valid[0, 0]
        selected_terminal = terminal_rows[valid_terminal]
        for name, value in leaf_source.items():
            if selected_terminal.numel():
                leaf_values[name][0, : selected_terminal.numel()].copy_(
                    value.index_select(0, selected_terminal)
                )
        leaf_count = route.counts[0, 0].reshape(1)
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
        trained_row = torch.nonzero(trained, as_tuple=False).flatten()
        trained_count = trained.sum(dtype=torch.int64).reshape(1)
        sigma = next(
            slab
            for slab in self.owners.slabs
            if slab.schema.name == f"SigmaHead.{batch.input_type}"
        )
        sigma_old_tail = sigma.reserved_tails_device()[0]
        sigma_values = self._defaults(sigma, rows)
        model_value = request.model.values["value"]
        actor_combo = parent_private.gather(1, owner[:, None]).to(torch.int64)
        selected_value = model_value.gather(1, actor_combo).float()
        sigma_values["old_value"][0, : trained_row.numel()].copy_(
            selected_value.index_select(0, trained_row)
        )
        sigma_values["old_trace"][0, : trained_row.numel()].copy_(
            request.model.values["trace"].index_select(0, trained_row)
        )
        sigma_values["old_cost"][0, : trained_row.numel()].copy_(
            request.model.values["cost"].index_select(0, trained_row)
        )
        destinations.append(
            DeviceDestinationRequest(
                sigma, trained_count, sigma_values, batch.input_type, address
            )
        )

        action_head = next(
            slab for slab in self.owners.slabs if slab.schema.name == "ActionHead.all"
        )
        action_old_tail = action_head.reserved_tails_device()[0]
        trained_child = trained.index_select(0, action_source)
        trained_action_row = torch.nonzero(trained_child, as_tuple=False).flatten()
        trained_action_count = trained_child.sum(dtype=torch.int64).reshape(1)
        action_values = self._defaults(action_head, action_total)
        action_values["action"][0, : trained_action_row.numel()].copy_(
            action.index_select(0, trained_action_row)
        )
        retained_log = sample["selected_log_policy"].reshape(action_total, -1)[:, :1]
        action_values["old_log_policy"][0, : trained_action_row.numel()].copy_(
            retained_log.index_select(0, trained_action_row)
        )
        destinations.append(
            DeviceDestinationRequest(
                action_head,
                trained_action_count,
                action_values,
                "all",
                address,
            )
        )

        next_input = self.machine.direct_model_input(next_state, log_reach, history)
        next_logical = torch.stack((root, child_path), -1)
        family_source: dict[str, Tensor] = {
            "logical_rng_key": next_logical,
            "true_profile": private,
            "folded_range_row": batch.fields["folded_range_row"][0].index_select(
                0, endpoint_source
            ),
            "model_input": next_input,
            "owner": next_state.to_act,
            "sampled_private_state": private,
            "path_mass": endpoint_mass,
            "log_player_ancestor_sum": endpoint_log_sum,
            "log_ancestor_product": endpoint_log_product,
            "branch_beta": current_beta,
            "nominal_degree": torch.zeros(total, dtype=torch.int32, device=device),
            "executed_degree": torch.zeros(total, dtype=torch.int32, device=device),
            "target_degree": torch.zeros(total, dtype=torch.int32, device=device),
        }
        if "role_selector" in batch.fields:
            family_source["role_selector"] = batch.fields["role_selector"][
                0
            ].index_select(0, endpoint_source)
        published: list[SemanticFamilyFIFO] = []
        produced: list[int] = []
        queue_cost_additions: list[Tensor] = []
        for family_index, (name, queue) in enumerate(self.queues.items()):
            selected_rows = route.source_row[0, family_index + 1][
                route.valid[0, family_index + 1]
            ]
            values = self._defaults(queue.family, total)
            for field, value in family_source.items():
                if field in values and selected_rows.numel():
                    values[field][0, : selected_rows.numel()].copy_(
                        value.index_select(0, selected_rows)
                    )
            cost_addition = torch.zeros(
                1, dtype=torch.float32, device=device
            )
            if self.cost_safeguard is not None and selected_rows.numel():
                selected_role = (
                    None
                    if "role_selector" not in family_source
                    else family_source["role_selector"].index_select(
                        0, selected_rows
                    )
                )
                with torch.no_grad():
                    child_evaluation = self.child_model(
                        PokerForwardInput(
                            name,
                            next_input.index_select(0, selected_rows),
                            private.index_select(0, selected_rows),
                            next_state.to_act.index_select(0, selected_rows),
                            selected_role,
                        ),
                        torch.zeros(
                            1, dtype=torch.int64, device=device
                        ),
                    )
                child_cost = child_evaluation["cost"].detach().float()
                values["frontier_cost_contribution"][
                    0, : selected_rows.numel()
                ].copy_(child_cost)
                cost_addition[0] = child_cost.sum()
            count = route.counts[0, family_index + 1].reshape(1)
            destinations.append(
                DeviceDestinationRequest(queue.family, count, values, name, address)
            )
            published.append(queue)
            queue_cost_additions.append(cost_addition)
            produced.append(selected_rows.numel())

        descriptor_source: list[Tensor] = []
        descriptor_destination: list[Tensor] = []
        descriptor_ordinal: list[Tensor] = []
        if trained_row.numel():
            compact = torch.arange(
                trained_row.numel(), dtype=torch.int64, device=device
            )
            descriptor_source.append(trained_row)
            descriptor_destination.append(torch.full_like(compact, 1))
            descriptor_ordinal.append(compact)
        if trained_action_row.numel():
            descriptor_source.append(action_source.index_select(0, trained_action_row))
            descriptor_destination.append(
                torch.full(
                    (trained_action_row.numel(),),
                    2,
                    dtype=torch.int64,
                    device=device,
                )
            )
            descriptor_ordinal.append(
                torch.arange(
                    trained_action_row.numel(),
                    dtype=torch.int64,
                    device=device,
                )
            )
        endpoint_destination = torch.empty(total, dtype=torch.int64, device=device)
        endpoint_ordinal = torch.empty_like(endpoint_destination)
        for family_value in self._route_family_values:
            selected = route_family[0].eq(family_value)
            request_index = 0 if family_value == -1 else 3 + family_value
            endpoint_destination[selected] = request_index
            endpoint_ordinal[selected] = (
                selected.cumsum(0, dtype=torch.int64)[selected] - 1
            )
        descriptor_source.append(endpoint_source)
        descriptor_destination.append(endpoint_destination)
        descriptor_ordinal.append(endpoint_ordinal)
        source_descriptor = torch.cat(descriptor_source)
        destination_descriptor = torch.cat(descriptor_destination)
        ordinal_descriptor = torch.cat(descriptor_ordinal)
        extent = int(source_descriptor.numel())
        coverage = CommitCoverage(
            valid=torch.ones((1, extent), dtype=torch.bool, device=device),
            source_ordinal=source_descriptor[None],
            destination=destination_descriptor[None],
            destination_ordinal=ordinal_descriptor[None],
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
        leaf_row = torch.full((total,), -1, dtype=torch.int64, device=device)
        leaf_row[terminal] = leaf_old_tail + endpoint_ordinal[terminal]
        player_record = _ExpansionRecord(
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
            selected = endpoint_record.eq(record_identifier).nonzero().flatten()
            if not selected.numel():
                continue
            edge = endpoint_record_edge.index_select(0, selected)
            record.child_key.index_copy_(
                0, edge, final_key.index_select(0, selected)
            )
            record.terminal.index_copy_(
                0, edge, terminal.index_select(0, selected)
            )
            record.child_input.index_copy_(
                0, edge, next_family.index_select(0, selected)
            )
            record.child_leaf_row.index_copy_(
                0, edge, leaf_row.index_select(0, selected)
            )
        correction = (
            (
                sample["nominal_degree"].float()
                - sample["executed_degree"].float()
            )
            * request.model.values["cost"].detach().float()
        ).sum()
        staged_payload = self.topology_recorder.stage_payload(
            device_records,
            produced=tuple(produced),
            sampled_nodes=sampled_nodes,
            counterfactual_correction=correction,
        )
        topology_publication = DeferredBranchedTopologyPublication(
            lane=0,
            sequence=self._topology_sequence,
            declared_nodes=sum(
                record.parent_key.shape[0] for record in device_records
            ),
            declared_edges=sum(
                record.child_key.shape[0] for record in device_records
            ),
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
            deferred_branched_topology=(topology_publication,),
            queue_cost_additions=tuple(queue_cost_additions),
        )


class BranchedSampledPokerIteration:
    """Run fixed or KKT sampled-private trees through one queued composition."""

    def __init__(
        self,
        *,
        geometry: NLHEGeometry,
        owners: GeneratedOwners,
        mode: ModeSchema,
        controller: BranchController,
        model: nn.Module,
        model_operation: ModelOperation,
        optimizer: DenseAdam,
        loss: LossConfig,
        schedule: LearnerSchedule,
        generator: torch.Generator,
        seed: int,
        lambda_: float,
        minimum_tree_size: int = 1,
        cost_safeguard: CostSafeguardController | None = None,
        precision: str = "fp32",
        utility: str = "raw_bb",
        run_state: RunState | None = None,
        frozen_operation: ModelOperation | None = None,
        reference_operation: ModelOperation | None = None,
    ) -> None:
        if (
            not mode.rollout_mode.is_branched
            or mode.private_estimator is not PrivateEstimator.SAMPLED
        ):
            raise ValueError("Phase 7 iteration requires a sampled branched mode")
        if controller.mode is not mode.rollout_mode:
            raise ValueError("branch controller mode differs from the schema")
        if owners.plan.mode != mode or owners.plan.manifest.n_replica != 1:
            raise ValueError("branched owners differ from the ordinary schema")
        if (
            mode.training_regime is TrainingRegime.BEST_RESPONSE
            and frozen_operation is None
        ):
            raise ValueError("branched best response requires a frozen operation")
        self.geometry = geometry
        self.owners = owners
        self.mode = mode
        self.controller = controller
        self.model = model
        self.model_operation = model_operation
        self.optimizer = optimizer
        self.loss = loss
        self.schedule = schedule
        self.generator = generator
        self.seed = int(seed)
        self.lambda_ = float(lambda_)
        if (
            isinstance(minimum_tree_size, bool)
            or not isinstance(minimum_tree_size, int)
            or minimum_tree_size <= 0
        ):
            raise ValueError("minimum_tree_size must be a positive integer")
        self.minimum_tree_size = minimum_tree_size
        self.cost_safeguard = cost_safeguard
        self.precision = precision
        self.utility = utility
        self.run_state = run_state
        if (
            run_state is not None
            and run_state.branch_controller_state is not None
            and controller_from_state_dict(
                run_state.branch_controller_state
            ).state_dict()
            != controller.state_dict()
        ):
            raise ValueError("branched controller differs from checkpointed run state")
        if (
            run_state is not None
            and run_state.cost_safeguard_state is not None
            and (
                cost_safeguard is None
                or CostSafeguardController.from_state_dict(
                    run_state.cost_safeguard_state
                ).state_dict()
                != cost_safeguard.state_dict()
            )
        ):
            raise ValueError("cost safeguard differs from checkpointed run state")
        self.frozen_operation = frozen_operation
        self.reference_operation = reference_operation
        self.policy = PokerPolicyOperator()

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

    def _publish_roots(
        self, machine: PokerStateMachine, queue: SemanticFamilyFIFO, iteration: int
    ) -> None:
        roots = (
            self.owners.plan.manifest.root_count
            if self.run_state is None
            else self.run_state.root_count
        )
        epoch = 0 if self.run_state is None else self.run_state.retry_epoch
        logical_base = (
            0 if self.run_state is None else self.run_state.attempt_logical_base
        )
        state = machine.initial_roots(
            roots,
            seed=self.seed,
            iteration=iteration,
            retry_epoch=epoch,
            logical_base=logical_base,
            device=self.owners.plan.device,
        )
        values = self._defaults(queue.family, roots)
        logical_root = torch.arange(
            logical_base,
            logical_base + roots,
            dtype=torch.int64,
            device=state.device,
        )
        values["logical_rng_key"][0, :, 0] = logical_root
        values["logical_rng_key"][0, :, 1] = 0
        values["true_profile"][0] = state.combo_ids
        values["sampled_private_state"][0] = state.combo_ids
        values["model_input"][0] = machine.direct_model_input(state)
        values["owner"][0] = state.to_act
        values["path_mass"][0].fill_(1)
        values["log_player_ancestor_sum"][0].fill_(0)
        values["log_ancestor_product"][0].fill_(0)
        values["branch_beta"][0].fill_(1)
        if "role_selector" in values:
            values["role_selector"][0] = logical_root.remainder(
                self.geometry.players
            ).to(torch.uint8)
        root_cost = torch.zeros(
            roots, dtype=torch.float32, device=state.device
        )
        if self.cost_safeguard is not None:
            with torch.no_grad():
                root_cost = self._model(
                    PokerForwardInput(
                        "pre",
                        values["model_input"][0],
                        state.combo_ids,
                        state.to_act,
                        (
                            None
                            if "role_selector" not in values
                            else values["role_selector"][0]
                        ),
                    ),
                    torch.zeros(1, dtype=torch.int64, device=state.device),
                )["cost"].detach().float()
            values["frontier_cost_contribution"][0].copy_(root_cost)
        count = torch.tensor([roots], dtype=torch.int64, device=state.device)
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    queue.family,
                    count,
                    values,
                    "pre",
                    torch.tensor(
                        [logical_base], dtype=torch.int64, device=state.device
                    ),
                ),
            ),
            phase="branched_root_commit",
            active=torch.ones(1, dtype=torch.bool, device=state.device),
        )
        transaction.fill_all()
        if not bool(transaction.publish().all()):
            raise BranchedPokerError("branched roots exceeded fixed capacity")
        queue.publish_family_tail(torch.ones(1, dtype=torch.bool, device=state.device))
        queue.frontier_cost.copy_(root_cost.sum().reshape(1))

    def _publish_refill_root(
        self,
        queue: SemanticFamilyFIFO,
        *,
        original_row: int,
        logical_root: int,
    ) -> None:
        values = {
            name: column[:, original_row : original_row + 1].clone()
            for name, column in queue.family.columns.items()
        }
        count = torch.ones(1, dtype=torch.int64, device=queue.family.device)
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    queue.family,
                    count,
                    values,
                    "pre",
                    torch.tensor(
                        [logical_root],
                        dtype=torch.int64,
                        device=queue.family.device,
                    ),
                ),
            ),
            phase="branched_root_refill",
            active=torch.ones(1, dtype=torch.bool, device=queue.family.device),
        )
        transaction.fill_all()
        if not bool(transaction.publish().all()):
            failure = self.owners.failure_state.snapshot(0)
            if failure is None:
                raise BranchedPokerError(
                    "root refill failed without a device failure record"
                )
            raise RecoverableOverflow(
                failure,
                capacity=queue.family.capacity,
                observed_demand=int(queue.family.reserved_tails_device()[0]),
            )
        queue.publish_family_tail(
            torch.ones(1, dtype=torch.bool, device=queue.family.device)
        )
        queue.frontier_cost.add_(
            values["frontier_cost_contribution"].sum(1)
        )

    def _model(
        self, forward: PokerForwardInput, _lanes: Tensor
    ) -> Mapping[str, Tensor]:
        if self.mode.training_regime is TrainingRegime.SELF_PLAY:
            evaluation = self.model_operation(
                forward.input_type,
                forward.model_input,
                forward.sampled_private_state,
                forward.owner,
            )
        else:
            assert self.frozen_operation is not None
            if forward.role_selector is None:
                raise BranchedPokerError("BR rollout omitted selected roles")
            trained = forward.owner.to(torch.int64).eq(
                forward.role_selector.to(torch.int64)
            )
            groups: list[tuple[Tensor, ModelEvaluation]] = []
            for mask, operation in (
                (trained, self.model_operation),
                (~trained, self.frozen_operation),
            ):
                row = torch.nonzero(mask, as_tuple=False).flatten()
                if row.numel():
                    groups.append(
                        (
                            row,
                            operation(
                                forward.input_type,
                                forward.model_input.index_select(0, row),
                                forward.sampled_private_state.index_select(0, row),
                                forward.owner.index_select(0, row),
                            ),
                        )
                    )
            prototype = groups[0][1]

            def merge(name: str, *, public: bool = False) -> Tensor:
                source = (
                    prototype.public_heads[name]
                    if public
                    else (
                        prototype.value
                        if name == "value"
                        else prototype.policy_raw[name]
                    )
                )
                result = source.new_empty((forward.owner.numel(), *source.shape[1:]))
                for row, item in groups:
                    value = (
                        item.public_heads[name]
                        if public
                        else (item.value if name == "value" else item.policy_raw[name])
                    )
                    result.index_copy_(0, row, value)
                return result

            evaluation = ModelEvaluation(
                MappingProxyType({name: merge(name) for name in prototype.policy_raw}),
                merge("value"),
                MappingProxyType(
                    {name: merge(name, public=True) for name in prototype.public_heads}
                ),
            )
        if set(evaluation.public_heads) != {"trace", "cost"}:
            raise BranchedPokerError("branched model omitted public heads")
        return MappingProxyType(
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

    def run(self, *, iteration: int) -> BranchedPokerResult:
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
        machine = PokerStateMachine(self.geometry)
        self._publish_roots(machine, queues["pre"], iteration)
        publisher = AsyncTopologyPublisher(
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
            topology=publisher,
            owners=registry,
        )
        recorder = BranchedTopologyRecorder(
            input_names=input_names,
            p_train=(
                self.geometry.players
                if self.mode.training_regime is TrainingRegime.SELF_PLAY
                else 1
            ),
            topology_cap=owners.plan.manifest.topology_cap,
            topology_edge_cap=owners.plan.manifest.topology_edge_cap,
        )
        hooks = BranchedSampledPokerHooks(
            self.geometry,
            registry,
            queues,
            seed=self.seed,
            iteration=iteration,
            mode=self.mode,
            controller=self.controller,
            cost_safeguard=self.cost_safeguard,
            child_model=self._model,
            topology_recorder=recorder,
            policy=self.policy,
        )
        scheduler = FamilyScheduler(input_names, owners.plan.manifest.family_pack_cap)
        roots = (
            owners.plan.manifest.root_count
            if self.run_state is None
            else self.run_state.root_count
        )
        depths = {name: 0 for name in input_names}
        depths["pre"] = roots
        hooks.realized_tree_size = roots
        logical_time = 0
        last_event: torch.cuda.Event | None = None
        pre_refill_size: int | None = None
        pre_refill_record_count: int | None = None
        logical_base = (
            0 if self.run_state is None else self.run_state.attempt_logical_base
        )
        while True:
            while any(depths.values()):
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
                    for index, name in enumerate(input_names)
                )
                scheduled = scheduler.select(states, allow_partial=True)
                if scheduled is None:
                    raise BranchedPokerError("branched scheduler lost queued work")
                hooks.frontier_cost_forecast = torch.stack(
                    tuple(queue.frontier_cost[0] for queue in queues.values())
                ).sum()
                commit = transaction.execute(
                    queues[scheduled.input_type],
                    scheduled,
                    self._model,
                    hooks,
                )
                publication = hooks.last_topology
                if publication is None:
                    raise BranchedPokerError("branched commit omitted topology")
                failed, feedback = publisher.wait_branched_feedback(publication)
                hooks.last_topology = None
                if failed:
                    failure = owners.failure_state.snapshot(0)
                    if failure is None or failure.failure_code != "capacity_overflow":
                        raise BranchedPokerError("branched rollout failed terminally")
                    raise RecoverableOverflow(
                        failure,
                        capacity=failure.available_extent,
                        observed_demand=failure.requested_extent,
                    )
                if not isinstance(feedback, _StagedBranchedPayload):
                    raise BranchedPokerError(
                        "branched commit returned invalid pinned feedback"
                    )
                hooks.realized_tree_size += int(feedback.sampled_nodes)
                depths[scheduled.input_type] -= scheduled.rows
                for name, count in zip(
                    input_names, feedback.produced, strict=True
                ):
                    depths[name] += count
                logical_time += 1
                last_event = commit.event
                if logical_time > owners.plan.manifest.topology_cap:
                    raise BranchedPokerError("branched rollout exceeded topology cap")
            for payload in publisher.drain_branched(block=True):
                if not isinstance(payload, _StagedBranchedPayload):
                    raise BranchedPokerError(
                        "branched publisher returned an invalid topology payload"
                    )
                recorder.accept_payload(payload)
            deficient = next(
                (
                    root
                    for root in range(logical_base, logical_base + roots)
                    if recorder.tree_size(root) < self.minimum_tree_size
                ),
                None,
            )
            if deficient is None:
                break
            if pre_refill_size is None:
                pre_refill_size = recorder.sampled_tree_size()
                pre_refill_record_count = recorder.record_count
            hooks.begin_root_refill(recorder.root_degree(deficient) + 1)
            self._publish_refill_root(
                queues["pre"],
                original_row=deficient - logical_base,
                logical_root=deficient,
            )
            depths["pre"] = 1
        assert last_event is not None
        recorder.materialize_cpu()
        topology, bindings, leaf_count = recorder.seal()
        if pre_refill_size is None:
            pre_refill_size = recorder.sampled_tree_size()
            pre_refill_record_count = recorder.record_count
        assert pre_refill_record_count is not None
        sampled_tree_nodes = recorder.sampled_tree_size()
        counterfactual_tree_size = (
            float(pre_refill_size)
            + recorder.counterfactual_correction(pre_refill_record_count)
        ) / roots
        recorder.release()
        publisher.assert_drained()
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
        del transaction, publisher, registry, queues, hooks
        owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
        owners.release_after(ExecutionPhase.QUEUE_DRAIN)

        owners.enter_phase(ExecutionPhase.COMPLETION)
        completion = CompletionFIFO(
            owners.slab("Leaf"),
            RolloutOwnerRegistry.from_generated(owners),
            head=torch.zeros(1, dtype=torch.int64, device=owners.plan.device),
            observed_epoch=torch.zeros(1, dtype=torch.int64, device=owners.plan.device),
        )
        binding = SampledShowdownBinding(
            self.geometry,
            owners.slab("LayerWork", "all"),
            n_replica=1,
            pack_cap=owners.plan.manifest.completion_pack_cap,
            utility=self.utility,
        )
        remaining = leaf_count
        assert last_event is not None
        while remaining:
            count = min(remaining, owners.plan.manifest.completion_pack_cap)
            success = completion.complete(
                (0,),
                count,
                binding.leaf_to_layer_work,
                wait_for=last_event,
            )
            torch._assert_async(success.all(), "branched completion failed")
            remaining -= count
        completion.release()
        owners.release_after(ExecutionPhase.COMPLETION)

        owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
        decision_mass = decision_path_mass(topology.parent, topology.executed_degree)
        sealed: dict[str, SealedBranchedInputs] = {}
        for input_index, name in enumerate(input_names):
            selected = (
                (bindings.input_type.eq(input_index) & topology.owner.ge(0))
                .nonzero()
                .flatten()
            )
            device = owners.plan.device
            sealed[name] = SealedBranchedInputs.seal(
                owners.slab("FamilyWork", name),
                owners.slab("LearnerIndex", name),
                input_type=name,
                family_row=topology.familywork_row[selected].to(device),
                action_begin=topology.action_begin[selected].to(device),
                action_count=topology.target_degree[selected].to(device),
                target_degree=topology.target_degree[selected].to(device),
                log_sampling_weight=decision_mass[selected]
                .log()
                .to(device=device, dtype=torch.float32),
            )
        owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)

        owners.enter_phase(ExecutionPhase.TARGETS)
        result = LayeredTargetExecutor(
            self.mode,
            TargetAlgebra(owners.plan.partitions, ()),
            lambda_=self.lambda_,
        ).execute(owners, plan)
        if result.failed_lanes:
            raise BranchedPokerError("branched layered targets failed")
        if self.cost_safeguard is not None:
            predicted_cost: list[Tensor] = []
            target_cost: list[Tensor] = []
            for name in input_names:
                head = owners.slab("SigmaHead", name)
                target_slab = owners.slab("SigmaTarget", name)
                rows = head.tail(0)
                if rows:
                    predicted_cost.append(head.columns["old_cost"][0, :rows])
                    target_cost.append(
                        target_slab.columns["cost_target"][0, :rows]
                    )
            if predicted_cost:
                predicted = torch.cat(predicted_cost).float().clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                target_cost_value = torch.cat(target_cost).float()
                ratio = target_cost_value / predicted
                deviance = ratio - ratio.log() - 1
                self.cost_safeguard.observe(
                    float(deviance.mean()),
                    float(torch.quantile(ratio, 0.95)),
                    int(ratio.numel()),
                )
        owners.release_after(ExecutionPhase.TARGETS)

        owners.enter_phase(ExecutionPhase.LEARNER)
        stores = {
            name: RaggedBranchedStore(owners, name, sealed_inputs=value)
            for name, value in sealed.items()
            if value.rows
        }
        sampler = BranchedBatchSampler(
            stores,
            self.schedule,
            generator=self.generator,
            sigma_pack_cap=owners.plan.manifest.learner_sigma_pack_cap,
            action_pack_cap=owners.plan.manifest.learner_action_pack_cap,
        )
        learner = LearnerOperation(
            self.mode,
            self.loss,
            self.model_operation,
            PokerLearnerPolicy(self.geometry, self.policy),
            precision=self.precision,
        )
        optimized = BranchedSampledIteration(
            model=self.model,
            learner=learner,
            optimizer=self.optimizer,
            sigma_pack_cap=owners.plan.manifest.learner_sigma_pack_cap,
            action_pack_cap=owners.plan.manifest.learner_action_pack_cap,
            run_state=self.run_state,
        ).optimize(
            sampler,
            iteration=iteration,
            reference_operation=self.reference_operation,
        )
        owners.release_after(ExecutionPhase.LEARNER)
        owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
        owners.release_after(ExecutionPhase.FINAL_BACKWARD)
        self.controller.observe(
            counterfactual_tree_size,
            iteration=iteration,
        )
        if self.run_state is not None:
            self.run_state.branch_controller_state = self.controller.state_dict()
            self.run_state.cost_safeguard_state = (
                None
                if self.cost_safeguard is None
                else self.cost_safeguard.state_dict()
            )
        owners.close()
        duplicate_labels = 0
        for node in range(topology.node_count):
            begin = int(topology.child_begin[node])
            count = int(topology.executed_degree[node])
            labels = topology.label[begin : begin + count]
            duplicate_labels += count - int(labels.unique().numel())
        return BranchedPokerResult(
            optimized,
            player_nodes=topology.node_count,
            leaves=leaf_count,
            target_layers=plan.layer_count,
            executed_edges=topology.edge_count,
            root_executed_degree=tuple(
                int(value)
                for value in topology.executed_degree[topology.parent.eq(-1)].tolist()
            ),
            root_target_degree=tuple(
                int(value)
                for value in topology.target_degree[topology.parent.eq(-1)].tolist()
            ),
            duplicate_child_labels=duplicate_labels,
            sampled_tree_nodes=sampled_tree_nodes,
        )


def _discard_attempt(owners: GeneratedOwners) -> None:
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


class RecoveringBranchedSampledPokerIteration:
    """Allocate and discard whole attempts around the branched controller."""

    def __init__(
        self,
        *,
        owner_factory: Callable[[], GeneratedOwners],
        plan: StoragePlan,
        controller: BranchController,
        run_state: RunState,
        arguments: Mapping[str, object],
        rank_agreement: RankCommonFailureAgreement | None = None,
    ) -> None:
        self.owner_factory = owner_factory
        self.plan = plan
        self.controller = controller
        self.run_state = run_state
        self.arguments = MappingProxyType(dict(arguments))
        self.rank_agreement = (
            RankCommonFailureAgreement()
            if rank_agreement is None
            else rank_agreement
        )

    def run(self, *, iteration: int) -> BranchedPokerResult:
        active: list[GeneratedOwners] = []
        recovery = BranchedOverflowRecovery[BranchedPokerResult](
            iteration=iteration,
            replica=0,
            controller=self.controller,
            run_state=self.run_state,
            rank_agreement=self.rank_agreement,
        )

        def attempt(address: AttemptAddress) -> BranchedPokerResult:
            if (
                address.retry_epoch != self.run_state.retry_epoch
                or address.logical_base != self.run_state.attempt_logical_base
            ):
                raise BranchedPokerError("branched retry address diverged")
            owners = self.owner_factory()
            if owners.plan != self.plan:
                owners.failure_state.release()
                raise BranchedPokerError("branched retry changed storage plan")
            active[:] = [owners]
            return BranchedSampledPokerIteration(
                owners=owners,
                controller=self.controller,
                run_state=self.run_state,
                **self.arguments,
            ).run(iteration=iteration)

        def discard(_failure: RecoverableOverflow) -> None:
            if len(active) != 1:
                raise BranchedPokerError("failed attempt owner is unavailable")
            _discard_attempt(active.pop())

        return recovery.run(attempt, discard)


__all__ = [
    "BranchedPokerError",
    "BranchedPokerResult",
    "BranchedSampledPokerHooks",
    "BranchedSampledPokerIteration",
    "BranchedTopologyRecorder",
    "RecoveringBranchedSampledPokerIteration",
]
