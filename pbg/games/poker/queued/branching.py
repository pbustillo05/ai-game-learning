"""Sampled and Rao--Blackwell branched NLHE through one queued engine."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from pbg.games.poker.queued.training import LearnerPreflightOptimizer
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
    validate_replica_group,
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
from pbg.queued.replica import (
    DistributedSUM,
    ReplicaOptimizerGroup,
)
from pbg.queued.rollout import (
    AsyncTopologyPublisher,
    CommitCoverage,
    CommitPlan,
    CompletionFIFO,
    DeferredBranchedTopologyPublication,
    DeviceCommitFailure,
    DrawPurpose,
    FamilyQueueState,
    FamilyScheduler,
    RolloutCommit,
    RolloutOwnerRegistry,
    RolloutTransaction,
    SampleRequest,
    SampleWorkspace,
    ScheduledFamily,
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


class BranchedPosteriorExhaustion(BranchedPokerError):
    """A bounded posterior sampler exhausted before any publication."""

    def __init__(self, failure: DeviceFailure) -> None:
        super().__init__(
            "branched RB posterior rejection exhausted at logical "
            f"address {failure.logical_address} after "
            f"{failure.sampler_attempt_count} attempts"
        )
        self.failure = failure


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
    child_input: int


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

    inflight_cap = 4

    def __init__(
        self,
        *,
        input_names: tuple[str, ...],
        p_train: int,
        topology_cap: int,
        topology_edge_cap: int,
        topology_packet_cap: int,
        device: torch.device,
        input_to_output: tuple[int, ...] | None = None,
    ) -> None:
        self.input_names = input_names
        self.p_train = p_train
        self.topology_cap = topology_cap
        self.topology_edge_cap = topology_edge_cap
        self.topology_packet_cap = topology_packet_cap
        self.device = device
        self.input_to_output = (
            tuple(0 for _ in input_names)
            if input_to_output is None
            else tuple(input_to_output)
        )
        if (
            len(self.input_to_output) != len(input_names)
            or any(value < 0 for value in self.input_to_output)
        ):
            raise ValueError("input-to-output topology map is invalid")
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
                (self.inflight_cap, packet_rows, 2),
                dtype=torch.int64,
                device=device,
            ),
            **{
                name: torch.empty(
                    (self.inflight_cap, packet_rows),
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
                (self.inflight_cap, packet_rows, 2),
                dtype=torch.int64,
                device=device,
            ),
            **{
                name: torch.empty(
                    (self.inflight_cap, packet_rows),
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
                value[:, :topology_packet_cap],
                device="cpu",
                pin_memory=True,
            )
            for name, value in self._device_node_buffers.items()
        }
        self._host_edge_buffers = {
            name: torch.empty_like(
                value[:, :topology_packet_cap],
                device="cpu",
                pin_memory=True,
            )
            for name, value in self._device_edge_buffers.items()
        }
        self._host_node_active = torch.ones(
            (self.inflight_cap, topology_packet_cap),
            dtype=torch.bool,
            pin_memory=True,
        )
        self._host_edge_active = torch.ones(
            (self.inflight_cap, topology_packet_cap),
            dtype=torch.bool,
            pin_memory=True,
        )
        self._produced = torch.empty(
            (self.inflight_cap, len(input_names)),
            dtype=torch.int64,
            pin_memory=True,
        )
        self._sampled_nodes = torch.empty(
            self.inflight_cap, dtype=torch.int64, pin_memory=True
        )
        self._correction = torch.empty(
            self.inflight_cap, dtype=torch.float32, pin_memory=True
        )
        self._node_count = torch.empty(
            self.inflight_cap, dtype=torch.int64, pin_memory=True
        )
        self._edge_count = torch.empty(
            self.inflight_cap, dtype=torch.int64, pin_memory=True
        )
        self._overflow = torch.empty(
            self.inflight_cap, dtype=torch.bool, pin_memory=True
        )
        self._observed_demand = torch.empty(
            self.inflight_cap, dtype=torch.int64, pin_memory=True
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
                    child_input=int(record.child_input[edge]),
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
                    edge.child_input = child.child_input
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
        slot: int,
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
            or not 0 <= slot < self.inflight_cap
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
            for name, buffers in self._device_node_buffers.items():
                buffer = buffers[slot]
                buffer.index_copy_(
                    0,
                    node_destination,
                    getattr(record, name),
                )
            for name, buffers in self._device_edge_buffers.items():
                buffer = buffers[slot]
                value = getattr(record, name)
                if name == "source_ordinal":
                    value = node_destination.index_select(
                        0, value.to(torch.int64)
                    )
                buffer.index_copy_(0, edge_destination, value)
            node_offset += record.node_active.sum(dtype=torch.int64)
            edge_offset += record.edge_active.sum(dtype=torch.int64)
        for name, hosts in self._host_node_buffers.items():
            host = hosts[slot]
            host.copy_(
                self._device_node_buffers[name][slot, :-1],
                non_blocking=True,
            )
        for name, hosts in self._host_edge_buffers.items():
            host = hosts[slot]
            host.copy_(
                self._device_edge_buffers[name][slot, :-1],
                non_blocking=True,
            )
        self._produced[slot].copy_(produced, non_blocking=True)
        self._sampled_nodes[slot].copy_(
            sampled_nodes.detach(), non_blocking=True
        )
        self._correction[slot].copy_(
            counterfactual_correction.detach(), non_blocking=True
        )
        self._node_count[slot].copy_(node_offset, non_blocking=True)
        self._edge_count[slot].copy_(edge_offset, non_blocking=True)
        self._overflow[slot].copy_(
            node_offset.gt(self.topology_packet_cap)
            | edge_offset.gt(self.topology_packet_cap)
            | frontier_overflow,
            non_blocking=True,
        )
        self._observed_demand[slot].copy_(
            torch.maximum(
                torch.maximum(node_offset, edge_offset),
                frontier_demand,
            ),
            non_blocking=True,
        )
        staged = _ExpansionRecord(
            node_active=self._host_node_active[slot],
            edge_active=self._host_edge_active[slot],
            **{
                name: values[slot]
                for name, values in self._host_node_buffers.items()
            },
            **{
                name: values[slot]
                for name, values in self._host_edge_buffers.items()
            },
        )
        return _StagedBranchedPayload(
            (staged,),
            self._produced[slot],
            self._sampled_nodes[slot],
            self._correction[slot],
            self._node_count[slot],
            self._edge_count[slot],
            self._overflow[slot],
            self._observed_demand[slot],
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
        resolved_output: dict[tuple[int, int], int] = {}

        def resolve_output(key: tuple[int, int]) -> int:
            cached = resolved_output.get(key)
            if cached is not None:
                return cached
            record = self._nodes[key]
            if record.input_type >= 0:
                result = self.input_to_output[record.input_type]
            else:
                child_domains: list[int] = []
                for edge in record.edges:
                    if edge.terminal:
                        if not 0 <= edge.child_input < len(
                            self.input_to_output
                        ):
                            raise BranchedPokerError(
                                "terminal branched child output partition "
                                "is unresolved"
                            )
                        child_domains.append(
                            self.input_to_output[edge.child_input]
                        )
                    else:
                        child_domains.append(
                            resolve_output(edge.child_key)
                        )
                if not child_domains or any(
                    domain != child_domains[0]
                    for domain in child_domains[1:]
                ):
                    raise BranchedPokerError(
                        "structural branched node has no single output domain"
                    )
                result = child_domains[0]
            resolved_output[key] = result
            return result
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
            output_type.append(resolve_output(key))
            sigma_target.append(record.sigma_head_row)
            for edge in record.edges:
                labels.append(edge.label)
                path_length.append(edge.path_length)
                if edge.terminal:
                    if not 0 <= edge.child_input < len(
                        self.input_to_output
                    ):
                        raise BranchedPokerError(
                            "terminal branched child output partition "
                            "is unresolved"
                        )
                    child_output.append(
                        self.input_to_output[edge.child_input]
                    )
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
                    child_output.append(
                        resolve_output(edge.child_key)
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
        topology_recorder: BranchedTopologyRecorder,
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
        self.realized_tree_size = torch.zeros(
            (), dtype=torch.float32, device=owners.device
        )
        self.last_topology: DeferredBranchedTopologyPublication | None = None
        self._topology_sequence = 0
        self._refill = False
        self._refill_child_ordinal = 0
        self.routing_traffic_bytes = 0
        self._posterior_failed_slots = torch.zeros(
            topology_recorder.inflight_cap,
            dtype=torch.bool,
            device=owners.device,
        )
        self._posterior_address_slots = torch.full(
            (topology_recorder.inflight_cap,),
            torch.iinfo(torch.int64).max,
            dtype=torch.int64,
            device=owners.device,
        )
        self._posterior_attempt_slots = torch.zeros(
            topology_recorder.inflight_cap,
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
        slot = self._topology_sequence % self.topology_recorder.inflight_cap
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
            frontier_cost=(
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
        parent_profile = batch.fields["true_profile"][0]
        private = parent_profile.index_select(0, source)
        if self.mode.private_estimator is PrivateEstimator.RAO_BLACKWELL:
            resolved = self._resolved_model_input(batch)
            log_reach, _history = self.machine.model_context(resolved)
            finite = torch.isfinite(log_reach)
            gauge = log_reach.masked_fill(~finite, -torch.inf).amax(
                -1, keepdim=True
            )
            range_weights = torch.exp(
                log_reach
                - gauge.masked_fill(~torch.isfinite(gauge), 0.0)
            ).masked_fill(~finite, 0.0)
            parent_state = self.machine.state_from_model_input(
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
        parent_state = self.machine.state_from_model_input(flat_input, parent_private)
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
        log_reach = action_conditioned_log_reach(
            log_reach,
            selected_log,
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
        endpoint_source = initial_endpoint_source
        endpoint_action = action_row
        endpoint_mass = initial_endpoint_mass
        endpoint_log_sum = initial_log_sum
        endpoint_log_product = initial_log_product
        chance_records: list[_ExpansionRecord] = []
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
        for generation in range(3):
            current_rows = current_active.numel()
            generation_capacity = min(
                chance_capacity,
                current_rows * self.controller.beta_max,
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
            if self.mode.private_estimator is PrivateEstimator.RAO_BLACKWELL:
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
                _ExpansionRecord(
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
                    "selected_player": selected_player.index_select(
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
        selected_value = (
            model_value.gather(
                1,
                parent_private.gather(1, owner[:, None]).to(torch.int64),
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
            for field, value in family_source.items():
                if field in values:
                    values[field][lane].copy_(
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
                        torch.tensor(
                            [lane], dtype=torch.int64, device=device
                        ),
                    )
                child_cost = child_evaluation["cost"].detach().float()
                child_cost = child_cost.masked_fill(~selected_valid, 0)
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
            slot=(
                self._topology_sequence
                % self.topology_recorder.inflight_cap
            ),
            produced=route.counts[0, 1:],
            sampled_nodes=sampled_nodes,
            counterfactual_correction=correction,
            frontier_overflow=frontier_overflow,
            frontier_demand=frontier_demand,
        )
        self.realized_tree_size.add_(sampled_nodes.to(torch.float32))
        topology_publication = DeferredBranchedTopologyPublication(
            lane=lane,
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


class _BranchedSampledPokerAttempt:
    """One internal fixed-capacity branched poker attempt."""

    def __init__(
        self,
        *,
        geometry: NLHEGeometry,
        owners: GeneratedOwners,
        mode: ModeSchema,
        controller: BranchController,
        model: nn.Module,
        model_operation: ModelOperation,
        rollout_operation: ModelOperation | None = None,
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
        completion_samples: int = 0,
        posterior_max_attempts: int = 256,
        benchmark_telemetry: bool = False,
        benchmark_identity: Mapping[str, str] | None = None,
        benchmark_rollout_only: bool = False,
    ) -> None:
        if not mode.rollout_mode.is_branched:
            raise ValueError("branched poker requires a branched mode")
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
        required_frontier = (
            sampled_branched_chance_frontier_row_bytes(geometry)
            if mode.private_estimator is PrivateEstimator.SAMPLED
            else branched_rb_chance_frontier_row_bytes(
                geometry, max_attempts=posterior_max_attempts
            )
        )
        if owners.plan.layout.chance_frontier_row_bytes < required_frontier:
            raise ValueError(
                "chance-frontier workspace is below the branched poker bound"
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
        if mode.private_estimator is PrivateEstimator.RAO_BLACKWELL:
            from pbg.queued.targets import (
                rb_conversion_library_workspace_bytes,
                rb_conversion_workspace_row_bytes,
            )

            if (
                owners.plan.layout.completion_workspace_row_bytes
                < rb_conversion_workspace_row_bytes(
                    geometry.private_hands,
                    geometry.rank_count**2,
                )
            ):
                raise ValueError(
                    "completion workspace is below the RB conversion bound"
                )
            if (
                owners.plan.manifest.library_workspace_bytes
                < rb_conversion_library_workspace_bytes(
                    geometry.private_hands,
                    geometry.rank_count**2,
                )
            ):
                raise ValueError(
                    "library workspace omits the RB conversion bound"
                )
        self.geometry = geometry
        self.owners = owners
        self.mode = mode
        self.controller = controller
        self.model = model
        self.model_operation = model_operation
        self.rollout_operation = (
            model_operation
            if rollout_operation is None
            else rollout_operation
        )
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
        self._model_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event, int]]
        ] = {}
        if (
            mode.private_estimator is PrivateEstimator.RAO_BLACKWELL
            and (
                isinstance(completion_samples, bool)
                or not isinstance(completion_samples, int)
                or completion_samples <= 0
            )
        ):
            raise ValueError("branched RB requires positive completion_samples")
        self.completion_samples = int(completion_samples)
        self.posterior_sampler = (
            None
            if mode.private_estimator is PrivateEstimator.SAMPLED
            else CollisionConditionedProfileSampler(
                geometry, max_attempts=posterior_max_attempts
            )
        )
        self.benchmark_telemetry = bool(benchmark_telemetry)
        self.benchmark_rollout_only = bool(benchmark_rollout_only)
        if self.benchmark_rollout_only and not self.benchmark_telemetry:
            raise ValueError(
                "rollout-only execution is reserved for benchmark telemetry"
            )
        self.benchmark_identity = (
            None
            if benchmark_identity is None
            else MappingProxyType(dict(benchmark_identity))
        )
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
        if "sampled_private_state" in values:
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
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
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
            result = self._model_inference(forward, lanes)
        finished.record()
        events = getattr(self, "_model_events", None)
        if events is None:
            events = {}
            self._model_events = events
        events.setdefault(forward.input_type, []).append(
            (started, finished, int(forward.owner.numel()))
        )
        return result

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
            evaluation = self.rollout_operation(
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
                self.rollout_operation,
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
        """Run independent inference rows in bounded CUDA subspans."""

        rows = int(forward.owner.numel())
        cap = min(self.owners.plan.manifest.family_pack_cap, 64)
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
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            torch.cuda.reset_peak_memory_stats(owners.plan.device)
            rollout_started = time.perf_counter()
        else:
            rollout_started = 0.0
        child_reservation_commit_seconds = 0.0
        model_inference_seconds = 0.0
        posterior_and_action_sampling_seconds = 0.0
        transition_seconds = 0.0
        model_stage_peak_allocated_bytes = 0
        sampling_stage_peak_allocated_bytes = 0
        transition_stage_peak_allocated_bytes = 0
        host_synchronizations = 0
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
        host_synchronizations += 1
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
            input_to_output=tuple(
                tuple(
                    output.name
                    for output in owners.plan.partitions.outputs
                ).index(
                    owners.plan.partitions.output_for(name).name
                )
                for name in input_names
            ),
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
            posterior_sampler=self.posterior_sampler,
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
        hooks.realized_tree_size.fill_(roots)
        logical_time = 0
        last_event: torch.cuda.Event | None = None
        pre_refill_size: int | None = None
        pre_refill_record_count: int | None = None
        logical_base = (
            0 if self.run_state is None else self.run_state.attempt_logical_base
        )
        while True:
            while any(depths.values()):
                inflight: list[
                    tuple[
                        ScheduledFamily,
                        RolloutCommit,
                        DeferredBranchedTopologyPublication,
                    ]
                ] = []
                while (
                    len(inflight) < recorder.inflight_cap
                    and any(depths.values())
                ):
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
                        raise BranchedPokerError(
                            "branched scheduler lost queued work"
                        )
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
                                available_extent=(
                                    owners.plan.manifest.topology_cap
                                ),
                            ),
                            capacity=owners.plan.manifest.topology_cap,
                            observed_demand=logical_time + 1,
                        )
                    commit = transaction.execute(
                        queues[scheduled.input_type],
                        scheduled,
                        self._bounded_mixed_model,
                        hooks,
                        measure_commit=self.benchmark_telemetry,
                    )
                    publication = hooks.last_topology
                    if publication is None:
                        raise BranchedPokerError(
                            "branched commit omitted topology"
                        )
                    hooks.last_topology = None
                    inflight.append((scheduled, commit, publication))
                    depths[scheduled.input_type] -= scheduled.rows
                    logical_time += 1
                last_publication = inflight[-1][2]
                last_feedback = publisher.wait_branched_feedback(
                    last_publication
                )
                host_synchronizations += 1
                feedback_rows: list[
                    tuple[
                        ScheduledFamily,
                        RolloutCommit,
                        bool,
                        object | None,
                    ]
                ] = []
                for scheduled, commit, publication in inflight:
                    result = (
                        last_feedback
                        if publication is last_publication
                        else publisher.branched_feedback(publication)
                    )
                    if result is None:
                        raise BranchedPokerError(
                            "ordered branched commit feedback was not visible"
                        )
                    failed, feedback = result
                    feedback_rows.append(
                        (scheduled, commit, failed, feedback)
                    )
                failed_rows = tuple(
                    row for row in feedback_rows if row[2]
                )
                if failed_rows:
                    failure = owners.failure_state.snapshot(0)
                    if (
                        failure is not None
                        and failure.failure_code == "rejection_exhausted"
                    ):
                        recorder.release()
                        del feedback, publication
                        _discard_attempt(owners)
                        raise BranchedPosteriorExhaustion(failure)
                    if (
                        failure is None
                        or failure.failure_code != "capacity_overflow"
                    ):
                        raise BranchedPokerError(
                            "branched rollout failed terminally"
                        )
                    recorder.release()
                    del feedback, publication
                    raise RecoverableOverflow(
                        failure,
                        capacity=failure.available_extent,
                        observed_demand=failure.requested_extent,
                    )
                for scheduled, commit, _failed, feedback in feedback_rows:
                    if not isinstance(feedback, _StagedBranchedPayload):
                        raise BranchedPokerError(
                            "branched commit returned invalid pinned feedback"
                        )
                    if self.benchmark_telemetry:
                        assert (
                            commit.model_started is not None
                            and commit.model_finished is not None
                            and commit.sample_finished is not None
                            and commit.commit_started is not None
                        )
                        model_inference_seconds += (
                            commit.model_started.elapsed_time(
                                commit.model_finished
                            )
                            / 1000.0
                        )
                        posterior_and_action_sampling_seconds += (
                            commit.model_finished.elapsed_time(
                                commit.sample_finished
                            )
                            / 1000.0
                        )
                        transition_seconds += (
                            commit.sample_finished.elapsed_time(
                                commit.commit_started
                            )
                            / 1000.0
                        )
                        if (
                            commit.peak_after_model
                            > commit.peak_before_model
                        ):
                            model_stage_peak_allocated_bytes = max(
                                model_stage_peak_allocated_bytes,
                                commit.peak_after_model,
                            )
                        if (
                            commit.peak_after_sample
                            > commit.peak_after_model
                        ):
                            sampling_stage_peak_allocated_bytes = max(
                                sampling_stage_peak_allocated_bytes,
                                commit.peak_after_sample,
                            )
                        if (
                            commit.peak_after_transition
                            > commit.peak_after_sample
                        ):
                            transition_stage_peak_allocated_bytes = max(
                                transition_stage_peak_allocated_bytes,
                                commit.peak_after_transition,
                            )
                        child_reservation_commit_seconds += (
                            commit.commit_started.elapsed_time(commit.event)
                            / 1000.0
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
                        raise RecoverableOverflow(
                            failure,
                            capacity=overflow.capacity,
                            observed_demand=overflow.observed_demand,
                        ) from None
                    produced = tuple(
                        int(value) for value in feedback.produced
                    )
                    for name, count in zip(
                        input_names, produced, strict=True
                    ):
                        depths[name] += count
                    last_event = commit.event
                del feedback_rows, inflight
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
                host_synchronizations += 1
            except RecoverableOverflow:
                recorder.release()
                raise
            depths["pre"] = 1
        assert last_event is not None
        recorder.materialize_cpu()
        topology, bindings, leaf_count = recorder.seal()
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            rollout_seconds = time.perf_counter() - rollout_started
            rollout_peak_allocated_bytes = int(
                torch.cuda.max_memory_allocated(owners.plan.device)
            )
            rollout_peak_reserved_bytes = int(
                torch.cuda.max_memory_reserved(owners.plan.device)
            )
            torch.cuda.reset_peak_memory_stats(owners.plan.device)
        else:
            rollout_seconds = 0.0
            rollout_peak_allocated_bytes = 0
            rollout_peak_reserved_bytes = 0
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
        hooks_routing_traffic_bytes = hooks.routing_traffic_bytes
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
        posterior_effective_hands = (
            None
            if not isinstance(binding, RBCompletionBinding)
            else float(binding.effective_hands[0].detach().cpu())
        )
        del binding
        for output, name in enumerate(plan.output_names):
            work = owners.slab("LayerWork", name)
            missing = plan.arena_rows[0][output] - work.tail(0)
            if missing < 0:
                raise BranchedPokerError(
                    "completion exceeded the target arena plan"
                )
            if missing:
                values = self._defaults(work, missing)
                counts = torch.tensor(
                    [missing], dtype=torch.int64, device=owners.plan.device
                )
                padding = PublicationTransaction.begin_device(
                    (
                        DeviceDestinationRequest(
                            work,
                            counts,
                            values,
                            name,
                            torch.tensor(
                                [leaf_count],
                                dtype=torch.int64,
                                device=owners.plan.device,
                            ),
                        ),
                    ),
                    phase="branched_target_arena_padding",
                    active=torch.ones(
                        1, dtype=torch.bool, device=owners.plan.device
                    ),
                )
                padding.fill_all()
                torch._assert_async(
                    padding.publish().all(),
                    "branched target arena padding failed",
                )
        owners.release_after(ExecutionPhase.COMPLETION)
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            completion_peak_allocated_bytes = int(
                torch.cuda.max_memory_allocated(owners.plan.device)
            )
            completion_peak_reserved_bytes = int(
                torch.cuda.max_memory_reserved(owners.plan.device)
            )
            torch.cuda.reset_peak_memory_stats(owners.plan.device)
        else:
            completion_peak_allocated_bytes = 0
            completion_peak_reserved_bytes = 0

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
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            align_peak_allocated_bytes = int(
                torch.cuda.max_memory_allocated(owners.plan.device)
            )
            align_peak_reserved_bytes = int(
                torch.cuda.max_memory_reserved(owners.plan.device)
            )
            torch.cuda.reset_peak_memory_stats(owners.plan.device)
        else:
            align_peak_allocated_bytes = 0
            align_peak_reserved_bytes = 0

        owners.enter_phase(ExecutionPhase.TARGETS)
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            target_started = time.perf_counter()
        result = LayeredTargetExecutor(
            self.mode,
            algebra,
            lambda_=self.lambda_,
        ).execute(owners, plan)
        if self.benchmark_telemetry:
            torch.cuda.synchronize(owners.plan.device)
            target_propagation_seconds = (
                time.perf_counter() - target_started
            )
        else:
            target_propagation_seconds = 0.0
        target_setup_seconds = target_propagation_seconds
        target_kernel_cuda_seconds = result.counters.cuda_seconds
        target_propagation_seconds = target_kernel_cuda_seconds
        if self.benchmark_telemetry:
            targets_peak_allocated_bytes = int(
                torch.cuda.max_memory_allocated(owners.plan.device)
            )
            targets_peak_reserved_bytes = int(
                torch.cuda.max_memory_reserved(owners.plan.device)
            )
        else:
            targets_peak_allocated_bytes = 0
            targets_peak_reserved_bytes = 0
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

        def finish(learner_result: object) -> BranchedPokerResult:
            self.controller.observe(
                counterfactual_tree_size,
                iteration=iteration,
            )
            if self.run_state is not None:
                self.run_state.logical_work["trajectories"] += (
                    self.run_state.root_count
                )
                self.run_state.branch_controller_state = (
                    self.controller.state_dict()
                )
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
                duplicate_labels += count - int(
                    labels.unique().numel()
                )
            telemetry = (
                BranchedBenchmarkTelemetry(
                    rollout_seconds=rollout_seconds,
                    routing_traffic_bytes=hooks_routing_traffic_bytes,
                    host_synchronizations=host_synchronizations,
                    child_reservation_commit_seconds=(
                        child_reservation_commit_seconds
                    ),
                    target_propagation_seconds=(
                        target_propagation_seconds
                    ),
                    peak_allocated_bytes=int(
                        max(
                            rollout_peak_allocated_bytes,
                            completion_peak_allocated_bytes,
                            align_peak_allocated_bytes,
                            targets_peak_allocated_bytes,
                        )
                    ),
                    peak_reserved_bytes=int(
                        max(
                            rollout_peak_reserved_bytes,
                            completion_peak_reserved_bytes,
                            align_peak_reserved_bytes,
                            targets_peak_reserved_bytes,
                        )
                    ),
                    model_inference_seconds=model_inference_seconds,
                    posterior_and_action_sampling_seconds=(
                        posterior_and_action_sampling_seconds
                    ),
                    transition_seconds=transition_seconds,
                    target_kernel_cuda_seconds=(
                        target_kernel_cuda_seconds
                    ),
                    rollout_peak_allocated_bytes=(
                        rollout_peak_allocated_bytes
                    ),
                    completion_peak_allocated_bytes=(
                        completion_peak_allocated_bytes
                    ),
                    align_peak_allocated_bytes=(
                        align_peak_allocated_bytes
                    ),
                    targets_peak_allocated_bytes=(
                        targets_peak_allocated_bytes
                    ),
                    model_stage_peak_allocated_bytes=(
                        model_stage_peak_allocated_bytes
                    ),
                    sampling_stage_peak_allocated_bytes=(
                        sampling_stage_peak_allocated_bytes
                    ),
                    transition_stage_peak_allocated_bytes=(
                        transition_stage_peak_allocated_bytes
                    ),
                    target_setup_seconds=target_setup_seconds,
                )
                if self.benchmark_telemetry
                else None
            )
            return BranchedPokerResult(
                learner_result,
                player_nodes=int(topology.owner.ge(0).sum()),
                leaves=leaf_count,
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
                sampled_tree_nodes=sampled_tree_nodes,
                posterior_effective_hands=posterior_effective_hands,
                telemetry=telemetry,
                observability=MappingProxyType(
                    {
                        **topology_diagnostics(
                            topology, bindings, owners.plan.partitions
                        ),
                        "model_timing_by_family": dict(
                            self._model_timing()
                        ),
                    }
                ),
            )

        if self.benchmark_rollout_only:
            owners.enter_phase(ExecutionPhase.LEARNER)
            owners.release_after(ExecutionPhase.LEARNER)
            owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
            owners.release_after(ExecutionPhase.FINAL_BACKWARD)
            return finish(None)

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
        return finish(optimized)

    def _target_algebra(self) -> TargetAlgebra:
        if self.mode.private_estimator is PrivateEstimator.SAMPLED:
            return TargetAlgebra(self.owners.plan.partitions, ())
        from pbg.queued.targets import ConversionMap

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


@dataclass(frozen=True)
class ReplicaBranchedPokerResult:
    """Lane-owned results from one shared branched storage transaction."""

    results: tuple[BranchedPokerResult | None, ...]
    failures: Mapping[int, str]
    versions: tuple[int, ...]


class ReplicaBranchedPokerIteration:
    """Schedule compatible fixed/KKT lanes over one generated owner set."""

    def __init__(
        self,
        *,
        geometry: NLHEGeometry,
        owners: GeneratedOwners,
        modes: Sequence[ModeSchema],
        controllers: Sequence[BranchController],
        safeguards: Sequence[CostSafeguardController],
        inference: PokerReplicaInference,
        reference_operations: Sequence[ModelOperation | None],
        optimizers: ReplicaOptimizerGroup,
        losses: Sequence[LossConfig],
        schedule: LearnerSchedule,
        generators: Sequence[torch.Generator],
        run_states: Sequence[RunState | None],
        seed: int,
        lambda_: float,
        precision: str = "fp32",
        utility: str = "raw_bb",
        completion_samples: int = 0,
        posterior_max_attempts: int = 256,
        cache_reference_outputs: bool = False,
        frozen_inference: PokerReplicaInference | None = None,
        distributed: DistributedSUM | None = None,
        rank_boundary: Callable[[], None] | None = None,
    ) -> None:
        count = owners.plan.manifest.n_replica
        self.modes = tuple(modes)
        self.controllers = tuple(controllers)
        self.safeguards = tuple(safeguards)
        self.reference_operations = tuple(reference_operations)
        self.losses = tuple(losses)
        self.generators = tuple(generators)
        self.run_states = tuple(run_states)
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
            )
        ):
            raise ValueError("branched replica lane owners are not aligned")
        validate_replica_group(self.modes)
        if (
            not all(mode.rollout_mode.is_branched for mode in self.modes)
            or owners.plan.mode != self.modes[0]
            or inference.modules.n_replica != count
            or optimizers.modules is not inference.modules
        ):
            raise ValueError("branched replica physical owners are incompatible")
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
            and (
                frozen_inference is None
                or frozen_inference.modules.n_replica != count
            )
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
        self.rank_boundary = rank_boundary
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
        values["model_input"][lane] = machine.direct_model_input(state)
        values["owner"][lane] = state.to_act
        values["path_mass"][lane].fill_(1)
        values["log_player_ancestor_sum"][lane].zero_()
        values["log_ancestor_product"][lane].zero_()
        values["branch_beta"][lane].fill_(1)
        if "role_selector" in values:
            values["role_selector"][lane] = logical.remainder(
                self.geometry.players
            ).to(torch.uint8)
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

    def _inference(
        self,
        lane: int,
        forward: PokerForwardInput,
    ) -> Mapping[str, Tensor]:
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
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

        mode = self.modes[lane]
        if mode.training_regime is TrainingRegime.SELF_PLAY:
            result = evaluate(self.inference)
            assert result is not None
            evaluation = result
        else:
            assert self.frozen_inference is not None
            assert forward.role_selector is not None
            trained = forward.owner.eq(forward.role_selector)
            learned_row = torch.nonzero(trained, as_tuple=False).flatten()
            frozen_row = torch.nonzero(~trained, as_tuple=False).flatten()
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
        if set(evaluation.public_heads) != {"trace", "cost"}:
            raise BranchedPokerError("branched replica model omitted T/C heads")
        result = MappingProxyType(
            {
                **{
                    name: value.squeeze(0).detach().float()
                    for name, value in evaluation.policy_raw.items()
                },
                "value": evaluation.value.squeeze(0).detach().float(),
                "trace": evaluation.public_heads["trace"]
                .squeeze(0)
                .detach()
                .float(),
                "cost": evaluation.public_heads["cost"]
                .squeeze(0)
                .detach()
                .float(),
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
        from pbg.queued.targets import ConversionMap

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

    def run(self, *, iteration: int) -> ReplicaBranchedPokerResult:
        owners = self.owners
        count = owners.plan.manifest.n_replica
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
        topologies: list[Topology] = []
        bindings: list[TopologyBindings] = []
        leaf_counts: list[int] = []
        sampled_nodes: list[int] = []
        counterfactual_sizes: list[float] = []
        last_events: list[torch.cuda.Event] = []
        failures: dict[int, str] = {}
        input_to_output = tuple(
            tuple(
                output.name for output in owners.plan.partitions.outputs
            ).index(owners.plan.partitions.output_for(name).name)
            for name in input_names
        )
        lane_schemas = self.modes
        for lane in range(count):
            try:
                self._publish_root(
                    lane, machine, queues["pre"], iteration=iteration
                )
                mode = self.modes[lane]
                publisher = AsyncTopologyPublisher(
                    registry,
                    p_train=(
                        self.geometry.players
                        if mode.training_regime is TrainingRegime.SELF_PLAY
                        else 1
                    ),
                )
                transaction = RolloutTransaction(
                    lane_schemas,
                    seed=self.seed,
                    iteration=iteration,
                    topology=publisher,
                    owners=registry,
                )
                recorder = BranchedTopologyRecorder(
                    input_names=input_names,
                    input_to_output=input_to_output,
                    p_train=(
                        self.geometry.players
                        if mode.training_regime is TrainingRegime.SELF_PLAY
                        else 1
                    ),
                    topology_cap=owners.plan.manifest.topology_cap,
                    topology_edge_cap=owners.plan.manifest.topology_edge_cap,
                    topology_packet_cap=(
                        owners.plan.manifest.topology_packet_cap
                    ),
                    device=owners.plan.device,
                )
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
                    topology_recorder=recorder,
                    policy=self.policy,
                    posterior_sampler=posterior,
                )
                roots = (
                    owners.plan.manifest.root_count
                    if self.run_states[lane] is None
                    else self.run_states[lane].root_count
                )
                depths = {name: 0 for name in input_names}
                depths["pre"] = roots
                hooks.realized_tree_size.fill_(roots)
                topology_rows_per_root = sum(
                    owners.plan.manifest.beta_max**generation
                    for generation in range(1, 5)
                )
                if (
                    topology_rows_per_root
                    > owners.plan.manifest.topology_packet_cap
                ):
                    raise ValueError(
                        "topology packet cannot cover one maximal branched root"
                    )
                scheduler = FamilyScheduler(
                    input_names,
                    min(
                        owners.plan.manifest.family_pack_cap,
                        owners.plan.manifest.topology_packet_cap
                        // topology_rows_per_root,
                    ),
                )
                logical_time = 0
                while any(depths.values()):
                    states = tuple(
                        FamilyQueueState(
                            lane,
                            name,
                            index,
                            0,
                            depths[name],
                            logical_time,
                            0,
                            mode,
                        )
                        for index, name in enumerate(input_names)
                    )
                    scheduled = scheduler.select(states, allow_partial=True)
                    if scheduled is None:
                        raise BranchedPokerError(
                            "branched replica scheduler lost work"
                        )
                    commit = transaction.execute(
                        queues[scheduled.input_type],
                        scheduled,
                        lambda forward, _lanes, selected=lane: self._inference(
                            selected, forward
                        ),
                        hooks,
                    )
                    publication = hooks.last_topology
                    hooks.last_topology = None
                    if publication is None:
                        raise BranchedPokerError(
                            "branched replica commit omitted topology"
                        )
                    failed, feedback = publisher.wait_branched_feedback(
                        publication
                    )
                    if failed or not isinstance(
                        feedback, _StagedBranchedPayload
                    ):
                        raise BranchedPokerError(
                            "branched replica commit failed"
                        )
                    recorder.accept_payload(feedback)
                    depths[scheduled.input_type] -= scheduled.rows
                    for name, produced in zip(
                        input_names, feedback.produced, strict=True
                    ):
                        depths[name] += int(produced)
                    last_events.append(commit.event)
                    logical_time += 1
                recorder.materialize_cpu()
                topology, binding, leaves = recorder.seal()
                sampled = recorder.sampled_tree_size()
                sampled_nodes.append(sampled)
                counterfactual_sizes.append(
                    (
                        float(sampled)
                        + recorder.counterfactual_correction(
                            recorder.record_count
                        )
                    )
                    / roots
                )
                recorder.release()
                publisher.assert_drained()
                topologies.append(topology)
                bindings.append(binding)
                leaf_counts.append(leaves)
            except Exception as exc:
                failures[lane] = f"{type(exc).__name__}: {exc}"
                raise
        if self.rank_boundary is not None:
            self.rank_boundary()
        plan = build_layer_plan(
            self.modes[0],
            owners.plan.partitions,
            tuple(topologies),
            tuple(bindings),
            p_train=(
                self.geometry.players
                if self.modes[0].training_regime
                is TrainingRegime.SELF_PLAY
                else 1
            ),
            manifest=owners.plan.manifest,
        )
        owners.release_after(ExecutionPhase.ROLLOUT)
        del registry, queues
        owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
        owners.release_after(ExecutionPhase.QUEUE_DRAIN)

        owners.enter_phase(ExecutionPhase.COMPLETION)
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
        algebra = self._target_algebra()
        completion_binding = (
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
                algebra,
                mode=self.modes[0],
                n_replica=count,
                pack_cap=owners.plan.manifest.completion_pack_cap,
                n_samples=self.completion_samples,
                utility=self.utility,
            )
        )
        remaining = list(leaf_counts)
        while any(remaining):
            active = tuple(
                lane for lane, value in enumerate(remaining) if value
            )
            rows = min(
                owners.plan.manifest.completion_pack_cap,
                *(remaining[lane] for lane in active),
            )
            success = completion.complete(
                active,
                rows,
                completion_binding.leaf_to_layer_work,
                wait_for=last_events[-1],
            )
            lane_index = torch.tensor(
                active,
                dtype=torch.int64,
                device=owners.plan.device,
            )
            torch._assert_async(
                success.index_select(0, lane_index).all(),
                "branched replica completion failed",
            )
            for lane in active:
                remaining[lane] -= rows
        completion.release()
        posterior_effective_hands = (
            None
            if not isinstance(completion_binding, RBCompletionBinding)
            else tuple(
                float(value)
                for value in completion_binding.effective_hands.detach().cpu().tolist()
            )
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
        owners.release_after(ExecutionPhase.COMPLETION)
        if self.distributed is not None:
            self.distributed.barrier()

        owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
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
        owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)
        owners.enter_phase(ExecutionPhase.TARGETS)
        target_result = LayeredTargetExecutor(
            self.modes[0], algebra, lambda_=self.lambda_
        ).execute(owners, plan)
        if target_result.failed_lanes:
            raise BranchedPokerError("branched replica targets failed")
        for lane, safeguard in enumerate(self.safeguards):
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
        owners.release_after(ExecutionPhase.TARGETS)

        owners.enter_phase(ExecutionPhase.LEARNER)
        generator_states = tuple(
            generator.get_state() for generator in self.generators
        )
        preflight_error: Exception | None = None
        try:
            for lane in range(count):
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
                )
                module = self.inference.modules.lane_module(lane)
                operation = (
                    module.learner_operation
                    if self.modes[lane].private_estimator
                    is PrivateEstimator.SAMPLED
                    else module.rb_learner_operation
                )
                try:
                    BranchedSampledIteration(
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
        except Exception as exc:  # noqa: BLE001 - transaction-wide abort
            preflight_error = exc
        finally:
            for generator, state in zip(
                self.generators, generator_states, strict=True
            ):
                generator.set_state(state)
            self.optimizers.zero_grad(range(count))
        if self.distributed is not None:
            local = torch.zeros(
                count,
                dtype=torch.bool,
                device=owners.plan.device,
            )
            if preflight_error is not None:
                local.fill_(True)
            common = self.distributed.agree_failures(local)
            if bool(common.any()) and preflight_error is None:
                preflight_error = BranchedPokerError(
                    "replica learner preflight failed on another rank"
                )
        if preflight_error is not None:
            for value in sealed.values():
                value.release()
            owners.release_after(ExecutionPhase.LEARNER)
            owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
            owners.release_after(ExecutionPhase.FINAL_BACKWARD)
            owners.close()
            raise BranchedPokerError(
                "replica learner preflight failed before optimizer transition"
            ) from preflight_error
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
                )
                module = self.inference.modules.lane_module(lane)
                model_operation = (
                    module.learner_operation
                    if self.modes[lane].private_estimator
                    is PrivateEstimator.SAMPLED
                    else module.rb_learner_operation
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
                    optimized = BranchedSampledIteration(
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
        owners.release_after(ExecutionPhase.LEARNER)
        for lane, optimized in enumerate(learner_results):
            if optimized is None:
                continue
            self.controllers[lane].observe(
                counterfactual_sizes[lane],
                iteration=iteration,
            )
            state = self.run_states[lane]
            if state is not None:
                state.logical_work["trajectories"] += state.root_count
                state.branch_controller_state = self.controllers[
                    lane
                ].state_dict()
                state.cost_safeguard_state = self.safeguards[
                    lane
                ].state_dict()
        owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
        owners.release_after(ExecutionPhase.FINAL_BACKWARD)
        owners.close()
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
            MappingProxyType(failures),
            tuple(
                int(value) for value in self.optimizers.versions.tolist()
            ),
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
    """Sole public branched iteration with whole-attempt recovery."""

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
        required = {"mode", "model", "model_operation", "optimizer"}
        if not required.issubset(self.arguments):
            raise ValueError(
                "branched iteration arguments omit replica ownership fields"
            )
        self.mode = self.arguments["mode"]
        self.model = self.arguments["model"]
        self.model_operation = self.arguments["model_operation"]
        self.rollout_operation = self.arguments.get(
            "rollout_operation", self.model_operation
        )
        self.optimizer = self.arguments["optimizer"]
        if (
            not isinstance(self.mode, ModeSchema)
            or not isinstance(self.model, nn.Module)
            or not isinstance(self.optimizer, DenseAdam)
        ):
            raise TypeError("branched iteration ownership fields are invalid")
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


BranchedPokerHooks = BranchedSampledPokerHooks
BranchedPokerIteration = BranchedSampledPokerIteration


__all__ = [
    "BranchedBenchmarkTelemetry",
    "BranchedPokerError",
    "BranchedPokerHooks",
    "BranchedPokerIteration",
    "BranchedPokerResult",
    "BranchedPosteriorExhaustion",
    "BranchedSampledPokerHooks",
    "BranchedSampledPokerIteration",
    "BranchedTopologyRecorder",
    "ReplicaBranchedPokerIteration",
    "ReplicaBranchedPokerResult",
    "action_conditioned_log_reach",
    "branched_rb_chance_frontier_row_bytes",
    "sampled_branched_chance_frontier_row_bytes",
    "sampled_branched_topology_packet_row_bytes",
]
