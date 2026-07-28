"""Sampled-private branched NLHE through the queued transaction engine."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch
from torch import Tensor, nn

from pbg.games.poker.model.capacity import (
    max_policy_partition_sequence_tokens,
)
from pbg.games.poker.queued.hooks import (
    FOLD,
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
    decision_log_path_mass,
)
from pbg.queued.capacity import (
    BRANCHED_TOPOLOGY_PACKET_ROW_BYTES,
    ExecutionPhase,
    GeneratedOwners,
    StoragePlan,
)
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
    ReferenceOutputCache,
    SealedBranchedInputs,
    allocate_statistical_counts,
    build_reference_output_cache,
    compact_best_response_evaluation,
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
    DeviceFailure,
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


class BranchedTopologyArenaOverflow(BranchedPokerError):
    """A compact rollout packet or cumulative topology exceeded admission."""

    def __init__(self, *, capacity: int, observed_demand: int) -> None:
        self.capacity = int(capacity)
        self.observed_demand = int(observed_demand)
        super().__init__(
            "branched compact topology exceeded its fixed admitted arena"
        )


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
        geometry.base_model_input_width
        - 8
        - 10 * players
        - players * hands
    ) // 10
    return (
        512
        + 8 * geometry.direct_model_input_width
        + 16 * players * hands
        + 128 * history
        + 64 * geometry.deck_cards
        + 128 * players
        + 3 * BRANCHED_TOPOLOGY_PACKET_ROW_BYTES
        + folded_statistics_workspace_row_bytes(geometry)
    )


def sampled_branched_topology_packet_row_bytes(
    _geometry: NLHEGeometry,
) -> int:
    """Device bytes for one compact topology node/edge packet row."""

    return BRANCHED_TOPOLOGY_PACKET_ROW_BYTES


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
    node_active: Tensor
    edge_active: Tensor
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
    produced: Tensor
    sampled_nodes: Tensor
    counterfactual_correction: Tensor
    node_count: Tensor
    edge_count: Tensor
    overflow: Tensor
    observed_demand: Tensor


@dataclass
class _CompactEdge:
    label: int
    path_length: int
    terminal: bool
    child_key: tuple[int, int]
    child_leaf_row: int


@dataclass
class _CompactNode:
    key: tuple[int, int]
    executed_degree: int
    nominal_degree: int
    target_degree: int
    owner: int
    input_type: int
    familywork_row: int
    sigma_head_row: int
    action_begin: int
    edges: list[_CompactEdge]


class BranchedTopologyRecorder:
    """Commit-visible logical expansions assembled into compact CPU topology."""

    def __init__(
        self,
        *,
        input_names: tuple[str, ...],
        p_train: int,
        topology_cap: int,
        topology_edge_cap: int,
        topology_packet_cap: int,
        device: torch.device,
    ) -> None:
        self.input_names = input_names
        self.p_train = p_train
        self.topology_cap = topology_cap
        self.topology_edge_cap = topology_edge_cap
        self.topology_packet_cap = topology_packet_cap
        self.device = device
        self._nodes: dict[tuple[int, int], _CompactNode] = {}
        self._compressed_roots: dict[tuple[int, int], _CompactNode] = {}
        self._pending: dict[tuple[int, int], list[_CompactEdge]] = {}
        self._tree_edges: dict[int, int] = {}
        self._root_degrees: dict[int, int] = {}
        self._sampled_roots: set[int] = set()
        self._sampled_prefix: list[int] = []
        self._record_corrections: list[float] = []
        packet_rows = topology_packet_cap + 1
        self._device_node_buffers = {
            "parent_key": torch.empty(
                (packet_rows, 2),
                dtype=torch.int64,
                device=device,
            ),
            **{
                name: torch.empty(
                    packet_rows,
                    dtype=dtype,
                    device=device,
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
        self._device_edge_buffers = {
            "child_key": torch.empty(
                (packet_rows, 2),
                dtype=torch.int64,
                device=device,
            ),
            **{
                name: torch.empty(
                    packet_rows,
                    dtype=dtype,
                    device=device,
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
        self._host_node_buffers = {
            name: torch.empty_like(
                value[:topology_packet_cap],
                device="cpu",
                pin_memory=True,
            )
            for name, value in self._device_node_buffers.items()
        }
        self._host_edge_buffers = {
            name: torch.empty_like(
                value[:topology_packet_cap],
                device="cpu",
                pin_memory=True,
            )
            for name, value in self._device_edge_buffers.items()
        }
        self._host_node_active = torch.ones(
            topology_packet_cap, dtype=torch.bool, pin_memory=True
        )
        self._host_edge_active = torch.ones(
            topology_packet_cap, dtype=torch.bool, pin_memory=True
        )
        self._produced = torch.empty(
            len(input_names), dtype=torch.int64, pin_memory=True
        )
        self._sampled_nodes = torch.empty((), dtype=torch.int64, pin_memory=True)
        self._correction = torch.empty((), dtype=torch.float32, pin_memory=True)
        self._node_count = torch.empty((), dtype=torch.int64, pin_memory=True)
        self._edge_count = torch.empty((), dtype=torch.int64, pin_memory=True)
        self._overflow = torch.empty((), dtype=torch.bool, pin_memory=True)
        self._observed_demand = torch.empty(
            (), dtype=torch.int64, pin_memory=True
        )

    def accept_payload(self, payload: _StagedBranchedPayload) -> None:
        if not payload.records:
            raise BranchedPokerError("branched topology payload is empty")
        node_count = int(payload.node_count)
        edge_count = int(payload.edge_count)
        if bool(payload.overflow) or (
            node_count > self.topology_packet_cap
            or edge_count > self.topology_packet_cap
        ):
            raise BranchedTopologyArenaOverflow(
                capacity=self.topology_packet_cap,
                observed_demand=int(payload.observed_demand),
            )
        record = payload.records[0]
        roots_before = len(self._sampled_roots)
        edges_by_source: list[list[int]] = [[] for _ in range(node_count)]
        for edge in range(edge_count):
            edges_by_source[int(record.source_ordinal[edge])].append(edge)
        for row in range(node_count):
            key_values = record.parent_key[row]
            key = (int(key_values[0]), int(key_values[1]))
            edge_rows = edges_by_source[row]
            degree = int(record.executed_degree[row])
            if len(edge_rows) != degree:
                raise BranchedPokerError(
                    "compact packet lost an executed child"
                )
            outgoing = [
                _CompactEdge(
                    label=int(record.label[edge]),
                    path_length=int(record.path_length[edge]),
                    terminal=bool(record.terminal[edge]),
                    child_key=(
                        int(record.child_key[edge, 0]),
                        int(record.child_key[edge, 1]),
                    ),
                    child_leaf_row=int(record.child_leaf_row[edge]),
                )
                for edge in edge_rows
            ]
            logical_root = key[0]
            self._tree_edges[logical_root] = (
                self._tree_edges.get(logical_root, 0) + len(outgoing)
            )
            if key[1] == 0:
                self._sampled_roots.add(logical_root)
                self._root_degrees[logical_root] = (
                    self._root_degrees.get(logical_root, 0) + degree
                )
            compressed_root = self._compressed_roots.pop(key, None)
            if compressed_root is not None:
                if not bool(record.refill[row]):
                    raise BranchedPokerError(
                        "a compressed branched root was expanded twice"
                    )
                self._nodes[key] = compressed_root
            incoming = self._pending.pop(key, [])
            trained = bool(record.trained[row])
            retain = (
                key in self._nodes
                or trained
                or degree >= 2
                or int(record.nominal_degree[row]) >= 2
            )
            if retain:
                node = self._nodes.get(key)
                if node is None:
                    node = _CompactNode(
                        key=key,
                        executed_degree=degree,
                        nominal_degree=int(record.nominal_degree[row]),
                        target_degree=int(record.target_degree[row]),
                        owner=(
                            int(record.owner[row]) if trained else -1
                        ),
                        input_type=(
                            int(record.input_type[row]) if trained else -1
                        ),
                        familywork_row=(
                            int(record.familywork_row[row]) if trained else -1
                        ),
                        sigma_head_row=(
                            int(record.sigma_head_row[row]) if trained else -1
                        ),
                        action_begin=(
                            int(record.action_begin[row]) if trained else -1
                        ),
                        edges=[],
                    )
                    self._nodes[key] = node
                elif not bool(record.refill[row]):
                    raise BranchedPokerError(
                        "a branched logical node was expanded twice"
                    )
                else:
                    node.executed_degree += degree
                node.edges.extend(outgoing)
                for edge in outgoing:
                    if not edge.terminal:
                        self._pending.setdefault(edge.child_key, []).append(edge)
                del incoming
            else:
                if degree != 1:
                    raise BranchedPokerError(
                        "only ordinary unary rows may be compressed"
                    )
                child = outgoing[0]
                if key[1] == 0 and not incoming:
                    compressed = _CompactNode(
                        key=key,
                        executed_degree=degree,
                        nominal_degree=int(record.nominal_degree[row]),
                        target_degree=int(record.target_degree[row]),
                        owner=-1,
                        input_type=-1,
                        familywork_row=-1,
                        sigma_head_row=-1,
                        action_begin=-1,
                        edges=outgoing,
                    )
                    self._compressed_roots[key] = compressed
                    if not child.terminal:
                        self._pending.setdefault(child.child_key, []).append(child)
                for edge in incoming:
                    edge.path_length += child.path_length
                    edge.terminal = child.terminal
                    edge.child_key = child.child_key
                    edge.child_leaf_row = child.child_leaf_row
                    if not edge.terminal:
                        self._pending.setdefault(edge.child_key, []).append(edge)
        compact_node_count = len(self._nodes) + len(self._compressed_roots)
        if compact_node_count > self.topology_cap:
            raise BranchedTopologyArenaOverflow(
                capacity=self.topology_cap,
                observed_demand=compact_node_count,
            )
        compact_edges = sum(
            len(node.edges)
            for nodes in (self._nodes.values(), self._compressed_roots.values())
            for node in nodes
        )
        if compact_edges > self.topology_edge_cap:
            raise BranchedTopologyArenaOverflow(
                capacity=self.topology_edge_cap,
                observed_demand=compact_edges,
            )
        sampled = int(payload.sampled_nodes)
        self._sampled_prefix.append(
            sampled
            + len(self._sampled_roots)
            - roots_before
            + (self._sampled_prefix[-1] if self._sampled_prefix else 0)
        )
        self._record_corrections.append(
            float(payload.counterfactual_correction)
        )

    def stage_payload(
        self,
        records: tuple[_ExpansionRecord, ...],
        *,
        produced: Tensor,
        sampled_nodes: Tensor,
        counterfactual_correction: Tensor,
        frontier_overflow: Tensor,
        frontier_demand: Tensor,
    ) -> _StagedBranchedPayload:
        if (
            not records
            or produced.device != self.device
            or produced.shape != (len(self.input_names),)
        ):
            raise BranchedPokerError("branched topology payload is invalid")
        node_offset = torch.zeros((), dtype=torch.int64, device=self.device)
        edge_offset = torch.zeros((), dtype=torch.int64, device=self.device)
        dummy = self.topology_packet_cap
        for record in records:
            node_rank = record.node_active.cumsum(0, dtype=torch.int64) - 1
            edge_rank = record.edge_active.cumsum(0, dtype=torch.int64) - 1
            node_destination = node_offset + node_rank
            edge_destination = edge_offset + edge_rank
            valid_node = record.node_active & node_destination.lt(dummy)
            valid_edge = record.edge_active & edge_destination.lt(dummy)
            node_destination = torch.where(
                valid_node,
                node_destination,
                torch.full_like(node_destination, dummy),
            )
            edge_destination = torch.where(
                valid_edge,
                edge_destination,
                torch.full_like(edge_destination, dummy),
            )
            for name, buffer in self._device_node_buffers.items():
                buffer.index_copy_(
                    0,
                    node_destination,
                    getattr(record, name),
                )
            for name, buffer in self._device_edge_buffers.items():
                value = getattr(record, name)
                if name == "source_ordinal":
                    value = node_destination.index_select(
                        0, value.to(torch.int64)
                    )
                buffer.index_copy_(0, edge_destination, value)
            node_offset += record.node_active.sum(dtype=torch.int64)
            edge_offset += record.edge_active.sum(dtype=torch.int64)
        for name, host in self._host_node_buffers.items():
            host.copy_(
                self._device_node_buffers[name][:-1],
                non_blocking=True,
            )
        for name, host in self._host_edge_buffers.items():
            host.copy_(
                self._device_edge_buffers[name][:-1],
                non_blocking=True,
            )
        self._produced.copy_(produced, non_blocking=True)
        self._sampled_nodes.copy_(sampled_nodes.detach(), non_blocking=True)
        self._correction.copy_(
            counterfactual_correction.detach(), non_blocking=True
        )
        self._node_count.copy_(node_offset, non_blocking=True)
        self._edge_count.copy_(edge_offset, non_blocking=True)
        self._overflow.copy_(
            node_offset.gt(self.topology_packet_cap)
            | edge_offset.gt(self.topology_packet_cap)
            | frontier_overflow,
            non_blocking=True,
        )
        self._observed_demand.copy_(
            torch.maximum(
                torch.maximum(node_offset, edge_offset),
                frontier_demand,
            ),
            non_blocking=True,
        )
        staged = _ExpansionRecord(
            node_active=self._host_node_active,
            edge_active=self._host_edge_active,
            **self._host_node_buffers,
            **self._host_edge_buffers,
        )
        return _StagedBranchedPayload(
            (staged,),
            self._produced,
            self._sampled_nodes,
            self._correction,
            self._node_count,
            self._edge_count,
            self._overflow,
            self._observed_demand,
        )

    def materialize_cpu(self) -> None:
        return None

    def release(self) -> None:
        """Release the fixed pinned topology arena at its rollout consumer."""

        self._nodes.clear()
        self._compressed_roots.clear()
        self._pending.clear()
        self._tree_edges.clear()
        self._root_degrees.clear()
        self._sampled_roots.clear()
        self._sampled_prefix.clear()
        self._record_corrections.clear()
        self._device_node_buffers.clear()
        self._device_edge_buffers.clear()
        self._host_node_buffers.clear()
        self._host_edge_buffers.clear()
        self._host_node_active = torch.empty(0, dtype=torch.bool)
        self._host_edge_active = torch.empty(0, dtype=torch.bool)
        self._produced = torch.empty(0, dtype=torch.int64)
        self._sampled_nodes = torch.empty(0, dtype=torch.int64)
        self._correction = torch.empty(0, dtype=torch.float32)
        self._node_count = torch.empty(0, dtype=torch.int64)
        self._edge_count = torch.empty(0, dtype=torch.int64)
        self._overflow = torch.empty(0, dtype=torch.bool)
        self._observed_demand = torch.empty(0, dtype=torch.int64)

    def seal(self) -> tuple[Topology, TopologyBindings, int]:
        if not self._nodes:
            raise BranchedPokerError("branched topology is empty")
        if self._pending:
            raise BranchedPokerError(
                "branched topology retains unresolved compact endpoints"
            )
        keys = list(self._nodes)
        targeted = {
            edge.child_key
            for node in self._nodes.values()
            for edge in node.edges
            if not edge.terminal
        }
        ordered_keys: list[tuple[int, int]] = []
        visited: set[tuple[int, int]] = set()

        def visit(key: tuple[int, int]) -> None:
            if key in visited:
                return
            visited.add(key)
            ordered_keys.append(key)
            for edge in self._nodes[key].edges:
                if not edge.terminal:
                    visit(edge.child_key)

        for root_key in sorted(key for key in keys if key not in targeted):
            visit(root_key)
        if len(visited) != len(self._nodes):
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
            record = self._nodes[key]
            degree = record.executed_degree
            child_begin.append(len(labels))
            executed.append(degree)
            nominal.append(record.nominal_degree)
            target.append(record.target_degree)
            owner.append(record.owner)
            family.append(record.familywork_row)
            sigma.append(record.sigma_head_row)
            action_begin.append(record.action_begin)
            input_type.append(record.input_type)
            output_type.append(0)
            sigma_target.append(record.sigma_head_row)
            for edge in record.edges:
                labels.append(edge.label)
                path_length.append(edge.path_length)
                child_output.append(0)
                if edge.terminal:
                    leaf = edge.child_leaf_row
                    if leaf < 0:
                        raise BranchedPokerError(
                            "terminal branch has no committed leaf row"
                        )
                    target_kind.append(int(TargetKind.LEAF))
                    target_row.append(leaf)
                    leaves.add(leaf)
                else:
                    try:
                        child = node_for_key[edge.child_key]
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
        topology.validate(
            bindings,
            p_train=self.p_train,
            compact=True,
            _allow_zero_path=True,
        )
        if topology.node_count > self.topology_cap:
            raise BranchedPokerError("realized topology exceeds its fixed cap")
        if topology.edge_count > self.topology_edge_cap:
            raise BranchedPokerError("realized edges exceed their fixed cap")
        return topology, bindings, len(leaves)

    def tree_size(self, logical_root: int) -> int:
        return 1 + self._tree_edges.get(logical_root, 0)

    def sampled_tree_size(self, record_limit: int | None = None) -> int:
        if not self._sampled_prefix:
            return 0
        if record_limit is None:
            return self._sampled_prefix[-1]
        if not 0 <= record_limit <= len(self._sampled_prefix):
            raise ValueError("sampled record limit is invalid")
        return 0 if record_limit == 0 else self._sampled_prefix[record_limit - 1]

    def root_degree(self, logical_root: int) -> int:
        degree = self._root_degrees.get(logical_root, 0)
        if degree <= 0:
            raise BranchedPokerError("logical root has no committed expansion")
        return degree

    @property
    def record_count(self) -> int:
        return len(self._sampled_prefix)

    def counterfactual_correction(self, record_limit: int) -> float:
        if not 0 <= record_limit <= len(self._record_corrections):
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
        if (
            owners.family_pack_cap * controller.beta_max
            > owners.chance_frontier_cap
        ):
            raise ValueError(
                "chance frontier does not cover one maximal player pack"
            )
        self.cost_safeguard = cost_safeguard
        self.child_model = child_model
        self.topology_recorder = topology_recorder
        self.frontier_cost_forecast = torch.zeros(
            (), dtype=torch.float32, device=owners.device
        )
        self.realized_tree_size = 0
        self.last_topology: DeferredBranchedTopologyPublication | None = None
        self._topology_sequence = 0
        self._refill = False
        self._refill_child_ordinal = 0

    def begin_root_refill(self, child_ordinal: int) -> None:
        if self._refill or child_ordinal <= 0:
            raise BranchedPokerError("invalid or overlapping root refill")
        self._refill = True
        self._refill_child_ordinal = int(child_ordinal)

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
        beta_max = self.controller.beta_max
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
        action_active = sample["active"].to(torch.bool)
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
        resolved_input = self._resolved_model_input(batch)
        log_reach, history = self.machine.model_context(
            resolved_input.index_select(0, source)
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
        fold_log_reach = log_reach
        fold_parent_street = state.street
        history = self.machine.append_player_history(state, action, history)
        next_state = self.machine.step_player(state, action)
        root = batch.fields["logical_rng_key"][0, :, 0].index_select(0, action_source)
        child_path = sample["child_path"]
        retry = self.owners.retry_epoch()[0]
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
        frontier_row = torch.arange(
            chance_capacity, dtype=torch.int64, device=device
        )
        action_row = frontier_row.clamp_max(action_total - 1)
        current_active = frontier_row.lt(action_total) & action_active.index_select(
            0, action_row
        )
        current_key = first_child_key.index_select(0, action_row)
        current_state = replace(
            next_state,
            **{
                name: value.index_select(0, action_row)
                for name, value in next_state.__dict__.items()
                if isinstance(value, Tensor)
            },
        )
        current_private = private.index_select(0, action_row)
        current_history = replace(
            history,
            **{
                name: value.index_select(0, action_row)
                for name, value in history.__dict__.items()
            },
        )
        current_log_reach = log_reach.index_select(0, action_row)
        endpoint_source = initial_endpoint_source.index_select(0, action_row)
        endpoint_action = action_row
        endpoint_mass = initial_endpoint_mass.index_select(0, action_row)
        endpoint_log_sum = initial_log_sum.index_select(0, action_row)
        endpoint_log_product = initial_log_product.index_select(0, action_row)
        chance_records: list[_ExpansionRecord] = []
        current_beta = torch.ones(
            chance_capacity, dtype=torch.float32, device=device
        )
        sampled_nodes = action_active.sum(dtype=torch.int64)
        frontier_demand = sampled_nodes.clone()
        frontier_overflow = torch.zeros(
            (), dtype=torch.bool, device=device
        )
        endpoint_record = torch.zeros(
            chance_capacity, dtype=torch.int64, device=device
        )
        endpoint_record_edge = action_row
        for generation in range(3):
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
            frontier_overflow |= chance_total.gt(chance_capacity)
            next_active = frontier_row < chance_total
            chance_source = torch.searchsorted(
                cumulative,
                frontier_row,
                right=True,
            ).clamp_max(chance_capacity - 1)
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
                _ExpansionRecord(
                    node_active=chance,
                    edge_active=chance_edge_active,
                    parent_key=current_key,
                    child_key=next_key,
                    source_ordinal=chance_source,
                    terminal=terminal_child,
                    child_input=self.machine.input_type(dealt_state),
                    child_leaf_row=torch.full(
                        (chance_capacity,),
                        -1,
                        dtype=torch.int64,
                        device=device,
                    ),
                    label=chance_label,
                    owner=torch.full(
                        (chance_capacity,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    input_type=torch.full(
                        (chance_capacity,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    familywork_row=torch.full(
                        (chance_capacity,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    sigma_head_row=torch.full(
                        (chance_capacity,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    action_begin=torch.full(
                        (chance_capacity,),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    executed_degree=chance_decision.executed_degree,
                    nominal_degree=chance_decision.nominal_degree,
                    target_degree=chance_decision.target_degree,
                    trained=torch.zeros(
                        chance_capacity,
                        dtype=torch.bool,
                        device=device,
                    ),
                    refill=torch.zeros(
                        chance_capacity,
                        dtype=torch.bool,
                        device=device,
                    ),
                    path_length=torch.ones(
                        chance_capacity,
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
                        chance_capacity, dtype=torch.bool, device=device
                    )
                ),
            )
            current_key = next_key
            current_state = dealt_state
            current_private = expanded_private
            current_history = chance_history
            current_log_reach = current_log_reach.index_select(0, chance_source)
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
        total = chance_capacity
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
            current_active[None],
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
        for name, value in leaf_source.items():
            leaf_values[name][0].copy_(value.index_select(0, terminal_rows))
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
        trained_row = torch.argsort(~trained, stable=True)
        trained_valid = trained.index_select(0, trained_row)
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
        sigma_values["old_value"][0].copy_(
            selected_value.index_select(0, trained_row)
        )
        sigma_values["old_trace"][0].copy_(
            request.model.values["trace"].index_select(0, trained_row)
        )
        sigma_values["old_cost"][0].copy_(
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
        trained_child = trained.index_select(0, action_source) & action_active
        trained_action_row = torch.argsort(~trained_child, stable=True)
        trained_action_valid = trained_child.index_select(
            0, trained_action_row
        )
        trained_action_count = trained_child.sum(dtype=torch.int64).reshape(1)
        action_values = self._defaults(action_head, action_total)
        action_values["action"][0].copy_(
            action.index_select(0, trained_action_row)
        )
        retained_log = sample["selected_log_policy"].reshape(action_total, -1)[:, :1]
        action_values["old_log_policy"][0].copy_(
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
                (action_total, self.geometry.rank_count**2),
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
        folded_values["range"][0].copy_(
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
        folded_values["log_mass"][0].copy_(
            actor_log_mass.index_select(0, fold_order)
        )
        fold_count = stored_fold.sum(dtype=torch.int64).reshape(1)
        fold_base = folded_slab.reserved_tails_device()[0]
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
        next_input = self.machine.direct_model_input(next_state, log_reach, history)
        if self.geometry.folded_summary_width:
            summary_slice = slice(self.geometry.base_model_input_width, None)
            next_input[:, summary_slice].copy_(
                resolved_input.index_select(0, endpoint_source)[:, summary_slice]
            )
            endpoint_fold = stored_fold.index_select(0, endpoint_action)
            refresh = (
                endpoint_fold
                | next_state.street.ne(
                    fold_parent_street.index_select(0, endpoint_action)
                )
            ) & endpoint_nonterminal
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
            next_input[:, summary_slice].index_copy_(
                0,
                refresh_rows,
                pack_folded_statistics(
                    self.geometry, refreshed_summary
                ).to(next_input.dtype),
            )
        next_logical = torch.stack((root, child_path), -1)
        family_source: dict[str, Tensor] = {
            "logical_rng_key": next_logical,
            "true_profile": private,
            "folded_range_row": folded_reference.to(torch.int32),
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
        queue_cost_additions: list[Tensor] = []
        for family_index, (name, queue) in enumerate(self.queues.items()):
            selected_rows = route.source_row[0, family_index + 1]
            selected_valid = route.valid[0, family_index + 1]
            values = self._defaults(queue.family, total)
            for field, value in family_source.items():
                if field in values:
                    values[field][0].copy_(
                        value.index_select(0, selected_rows)
                    )
            cost_addition = torch.zeros(
                1, dtype=torch.float32, device=device
            )
            if self.cost_safeguard is not None:
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
                child_cost = child_cost.masked_fill(~selected_valid, 0)
                values["frontier_cost_contribution"][0].copy_(child_cost)
                cost_addition[0] = child_cost.sum()
            count = route.counts[0, family_index + 1].reshape(1)
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
        coverage = CommitCoverage(
            valid=valid_descriptor.reshape(1, extent),
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
        leaf_row = torch.where(
            current_active & terminal,
            leaf_old_tail + endpoint_ordinal,
            torch.full_like(endpoint_ordinal, -1),
        )
        player_record = _ExpansionRecord(
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
        staged_payload = self.topology_recorder.stage_payload(
            device_records,
            produced=route.counts[0, 1:],
            sampled_nodes=sampled_nodes,
            counterfactual_correction=correction,
            frontier_overflow=frontier_overflow,
            frontier_demand=frontier_demand,
        )
        topology_publication = DeferredBranchedTopologyPublication(
            lane=0,
            sequence=self._topology_sequence,
            declared_nodes=self.topology_recorder.topology_packet_cap,
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
            deferred_branched_topology=(topology_publication,),
            queue_cost_additions=tuple(queue_cost_additions),
        )


class _BranchedSampledPokerAttempt:
    """One internal fixed-capacity sampled branched attempt."""

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
        cache_reference_outputs: bool = False,
        allocation_inputs: tuple[Tensor, Tensor, Tensor, int] | None = None,
        run_state: RunState | None = None,
        frozen_operation: ModelOperation | None = None,
        reference_operation: ModelOperation | None = None,
        rank_boundary: Callable[[], None] | None = None,
    ) -> None:
        if (
            not mode.rollout_mode.is_branched
            or mode.private_estimator is not PrivateEstimator.SAMPLED
        ):
            raise ValueError("Phase 7 iteration requires a sampled branched mode")
        if controller.mode is not mode.rollout_mode:
            raise ValueError("branch controller mode differs from the schema")
        if controller.beta_max != owners.plan.manifest.beta_max:
            raise ValueError(
                "branch controller beta_max differs from the storage manifest"
            )
        if owners.plan.mode != mode or owners.plan.manifest.n_replica != 1:
            raise ValueError("branched owners differ from the ordinary schema")
        if (
            run_state is not None
            and run_state.root_count != owners.plan.manifest.root_count
        ):
            raise ValueError("branched root count differs from the storage plan")
        postflop = getattr(model, "postflop", None)
        postflop_config = getattr(postflop, "cfg", None)
        m_raise = int(getattr(postflop_config, "m_raise", 1))
        from pbg.games.poker.queued.folded import (
            sequential_family_workspace_row_bytes,
        )

        if (
            owners.plan.layout.family_workspace_row_bytes
            < sequential_family_workspace_row_bytes(geometry, m_raise)
        ):
            raise ValueError(
                "family workspace is below the concurrent branched bound"
            )
        if (
            owners.plan.layout.chance_frontier_row_bytes
            < sampled_branched_chance_frontier_row_bytes(geometry)
        ):
            raise ValueError(
                "chance-frontier workspace is below the sampled poker bound"
            )
        if (
            owners.plan.layout.topology_packet_row_bytes
            < sampled_branched_topology_packet_row_bytes(geometry)
        ):
            raise ValueError(
                "topology packet workspace is below the branched packet bound"
            )
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
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("branched poker precision is invalid")
        if utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("branched poker utility is invalid")
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
        if type(cache_reference_outputs) is not bool:
            raise TypeError("cache_reference_outputs must be bool")
        self.cache_reference_outputs = cache_reference_outputs
        self.allocation_inputs = allocation_inputs
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
        self.rank_boundary = rank_boundary
        self._attempt_success = torch.empty(
            (), dtype=torch.bool, pin_memory=True
        )
        self._attempt_success_event = torch.cuda.Event(blocking=False)
        self.policy = PokerPolicyOperator()

    def _publication_succeeded(self, success: Tensor) -> bool:
        """Poll the fixed attempt-boundary feedback slot."""

        if success.device != self.owners.plan.device or success.numel() != 1:
            raise BranchedPokerError(
                "attempt publication success has an invalid shape"
            )
        self._attempt_success.copy_(success.reshape(()), non_blocking=True)
        self._attempt_success_event.record(
            torch.cuda.current_stream(success.device)
        )
        while not self._attempt_success_event.query():
            time.sleep(0)
        return bool(self._attempt_success)

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
                root_cost = self._bounded_mixed_model(
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
        if not self._publication_succeeded(transaction.publish().all()):
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
        if not self._publication_succeeded(transaction.publish().all()):
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
        self, forward: PokerForwardInput, lanes: Tensor
    ) -> Mapping[str, Tensor]:
        autocast_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise BranchedPokerError("selected CUDA device does not support BF16")
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            return self._model_inference(forward, lanes)

    @staticmethod
    def _model_output(evaluation: ModelEvaluation) -> Mapping[str, Tensor]:
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

    def _model_inference(
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
            evaluation = compact_best_response_evaluation(
                self.model_operation,
                self.frozen_operation,
                input_type=forward.input_type,
                model_input=forward.model_input,
                sampled_private_state=forward.sampled_private_state,
                owner=forward.owner,
                role_selector=forward.role_selector,
            )
        return self._model_output(evaluation)

    def _bounded_mixed_model(
        self, forward: PokerForwardInput, lanes: Tensor
    ) -> Mapping[str, Tensor]:
        """Run auxiliary T/C forwards in admitted family-pack subspans."""

        rows = int(forward.owner.numel())
        cap = self.owners.plan.manifest.family_pack_cap
        if rows <= cap:
            return self._model(forward, lanes)
        pieces: dict[str, list[Tensor]] = {}
        for begin in range(0, rows, cap):
            end = min(begin + cap, rows)
            item = self._model(
                PokerForwardInput(
                    forward.input_type,
                    forward.model_input[begin:end],
                    forward.sampled_private_state[begin:end],
                    forward.owner[begin:end],
                    (
                        None
                        if forward.role_selector is None
                        else forward.role_selector[begin:end]
                    ),
                ),
                lanes,
            )
            for name, value in item.items():
                pieces.setdefault(name, []).append(value)
        return MappingProxyType(
            {name: torch.cat(values) for name, values in pieces.items()}
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
            topology_packet_cap=owners.plan.manifest.topology_packet_cap,
            device=owners.plan.device,
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
            child_model=self._bounded_mixed_model,
            topology_recorder=recorder,
            policy=self.policy,
        )
        beta_max = owners.plan.manifest.beta_max
        topology_rows_per_root = sum(
            beta_max**generation for generation in range(1, 5)
        )
        if topology_rows_per_root > owners.plan.manifest.topology_packet_cap:
            raise ValueError(
                "topology packet cannot cover one maximal branched root"
            )
        rollout_pack_cap = min(
            owners.plan.manifest.family_pack_cap,
            owners.plan.manifest.topology_packet_cap
            // topology_rows_per_root,
        )
        scheduler = FamilyScheduler(input_names, rollout_pack_cap)
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
                if logical_time >= owners.plan.manifest.topology_cap:
                    recorder.release()
                    raise RecoverableOverflow(
                        DeviceFailure(
                            lane=0,
                            status=1,
                            phase="branched_topology_sequence",
                            failure_code="capacity_overflow",
                            input_type=scheduled.input_type,
                            logical_address=logical_base,
                            requested_extent=logical_time + 1,
                            available_extent=owners.plan.manifest.topology_cap,
                        ),
                        capacity=owners.plan.manifest.topology_cap,
                        observed_demand=logical_time + 1,
                    )
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
                    recorder.release()
                    del feedback, publication
                    raise RecoverableOverflow(
                        failure,
                        capacity=failure.available_extent,
                        observed_demand=failure.requested_extent,
                    )
                if not isinstance(feedback, _StagedBranchedPayload):
                    raise BranchedPokerError(
                        "branched commit returned invalid pinned feedback"
                    )
                try:
                    recorder.accept_payload(feedback)
                except BranchedTopologyArenaOverflow as overflow:
                    failure = DeviceFailure(
                        lane=0,
                        status=1,
                        phase="branched_topology",
                        failure_code="capacity_overflow",
                        input_type=scheduled.input_type,
                        logical_address=logical_base,
                        requested_extent=overflow.observed_demand,
                        available_extent=overflow.capacity,
                    )
                    recorder.release()
                    hooks.last_topology = None
                    del feedback, publication
                    raise RecoverableOverflow(
                        failure,
                        capacity=overflow.capacity,
                        observed_demand=overflow.observed_demand,
                    ) from None
                hooks.realized_tree_size += int(feedback.sampled_nodes)
                depths[scheduled.input_type] -= scheduled.rows
                produced = tuple(int(value) for value in feedback.produced)
                for name, count in zip(input_names, produced, strict=True):
                    depths[name] += count
                logical_time += 1
                last_event = commit.event
                del feedback, publication
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
            try:
                self._publish_refill_root(
                    queues["pre"],
                    original_row=deficient - logical_base,
                    logical_root=deficient,
                )
            except RecoverableOverflow:
                recorder.release()
                raise
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
        del transaction, publisher, registry, queues, hooks
        if self.rank_boundary is not None:
            self.rank_boundary()
        owners.release_after(ExecutionPhase.ROLLOUT)
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
        decision_log_mass = decision_log_path_mass(
            topology.parent,
            topology.executed_degree,
            device=owners.plan.device,
        )
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
                log_sampling_weight=decision_log_mass.index_select(
                    0,
                    selected.to(device=device, dtype=torch.int64),
                ),
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
        statistical_counts = None
        if allocation_inputs is not None:
            counts = allocate_statistical_counts(
                allocation_inputs[0],
                allocation_inputs[1],
                allocation_inputs[2],
                allocation_inputs[3],
            )
            statistical_counts = {
                name: int(counts[index])
                for index, name in enumerate(stores)
            }
        sampler = BranchedBatchSampler(
            stores,
            self.schedule,
            generator=self.generator,
            sigma_pack_cap=owners.plan.manifest.learner_sigma_pack_cap,
            action_pack_cap=owners.plan.manifest.learner_action_pack_cap,
            statistical_counts=statistical_counts,
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
                    (
                        sampler.statistical_counts[name]
                        + self.schedule.batch_size
                        - 1
                    )
                    // self.schedule.batch_size,
                )
                candidate = build_reference_output_cache(
                    self.reference_operation,
                    input_type=name,
                    model_input=columns["model_input"],
                    sampled_private_state=columns["sampled_private_state"],
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
            reference_operation=(
                None
                if reference_caches is not None
                else self.reference_operation
            ),
            reference_caches=reference_caches,
        )
        sampler.release()
        if reference_caches is not None:
            for cache in reference_caches.values():
                cache.release()
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
    if owners.failure_state.snapshot(0) is not None:
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


class BranchedSampledPokerIteration:
    """Sole public sampled-branched iteration with whole-attempt recovery."""

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
        if controller.beta_max != plan.manifest.beta_max:
            raise ValueError(
                "branch controller beta_max differs from the storage manifest"
            )
        if run_state.root_count != plan.manifest.root_count:
            raise ValueError("branched root count differs from the storage plan")
        if "rank_boundary" in arguments:
            raise ValueError("rank agreement is owned by the iteration")
        self.arguments = MappingProxyType(dict(arguments))
        self.rank_agreement = (
            RankCommonFailureAgreement.from_runtime()
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

        def rank_boundary() -> None:
            failure = self.rank_agreement.agree(None)
            if failure is None:
                return
            agreed = RecoverableOverflow(
                failure.failure,
                capacity=failure.capacity,
                observed_demand=failure.observed_demand,
            )
            agreed._rank_agreed = True  # type: ignore[attr-defined]
            raise agreed

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
            return _BranchedSampledPokerAttempt(
                owners=owners,
                controller=self.controller,
                run_state=self.run_state,
                rank_boundary=rank_boundary,
                **self.arguments,
            ).run(iteration=iteration)

        def discard(_failure: RecoverableOverflow | None) -> None:
            if len(active) != 1:
                raise BranchedPokerError("failed attempt owner is unavailable")
            _discard_attempt(active.pop())

        return recovery.run(
            attempt,
            discard,
            agreement_in_attempt=True,
        )


__all__ = [
    "BranchedPokerError",
    "BranchedPokerResult",
    "BranchedSampledPokerHooks",
    "BranchedSampledPokerIteration",
    "BranchedTopologyRecorder",
    "sampled_branched_chance_frontier_row_bytes",
    "sampled_branched_topology_packet_row_bytes",
]
