"""Semantic queues and the sole queued rollout transaction.

The module is game-opaque.  It schedules stable ``FamilyWork`` prefixes and
executes one ``model -> sample -> transition -> commit`` operation for both
sequential and branched schemas.  Retained writes are possible only through
the device-counted publication transaction owned by :mod:`pbg.queued.storage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from pbg.queued.contracts import (
    ModeSchema,
    PartitionContract,
    validate_replica_group,
)
from pbg.queued.storage import (
    DeviceDestinationRequest,
    DevicePublicationTransaction,
    MonotoneSlab,
    StorageError,
)
from pbg.queued.topology import (
    Topology,
    TopologyBindings,
    TopologyError,
    TopologyPacket,
)


class RolloutError(RuntimeError):
    """A queue, hook, or rollout-state contract was violated."""


class DrawPurpose(IntEnum):
    """Closed counter namespaces for every queued stochastic operation."""

    PRIVATE = 1
    ROUND = 2
    POSTERIOR = 3
    ACTION = 4
    CHANCE = 5
    COMPLETION = 6
    LEARNER = 7
    PROBE_SEAT = 8


LOGICAL_RNG_VERSION = "counter_hash_v3"
_MASK63 = (1 << 63) - 1


def _mix63(value: Tensor) -> Tensor:
    value = torch.bitwise_xor(value, value >> 30)
    value = (value * 0x3C79AC492BA7B653) & _MASK63
    value = torch.bitwise_xor(value, value >> 27)
    value = (value * 0x1C69B3F74AC4AE35) & _MASK63
    return torch.bitwise_xor(value, value >> 31) & _MASK63


def _mix63_scalar(value: int) -> int:
    value &= _MASK63
    value ^= value >> 30
    value = (value * 0x3C79AC492BA7B653) & _MASK63
    value ^= value >> 27
    value = (value * 0x1C69B3F74AC4AE35) & _MASK63
    return (value ^ (value >> 31)) & _MASK63


def logical_rng_keys(
    *,
    seed: int,
    iteration: int,
    root: Tensor | None = None,
    trajectory: Tensor | None = None,
    path: Tensor,
    draw: Tensor | int,
    purpose: DrawPurpose | int,
    lane: Tensor | int = 0,
    retry_epoch: Tensor | int = 0,
    active: Tensor | None = None,
) -> Tensor:
    """Return placement-, subdivision-, and schedule-independent CUDA keys.

    ``trajectory`` is retained as an explicit alias for callers whose logical
    root vocabulary predates the queued package.  It cannot be supplied with
    ``root``.  Inactive addresses receive the sentinel zero and therefore do
    not consume a logical draw.
    """

    if (root is None) == (trajectory is None):
        raise ValueError("provide exactly one of root or trajectory")
    root_address = trajectory if root is None else root
    assert root_address is not None
    if root_address.device != path.device:
        raise ValueError("root and path addresses must share a device")
    root_address, path = torch.broadcast_tensors(
        root_address.to(torch.int64), path.to(torch.int64)
    )
    draw_tensor = torch.as_tensor(
        draw, dtype=torch.int64, device=root_address.device
    )
    lane_tensor = torch.as_tensor(
        lane, dtype=torch.int64, device=root_address.device
    )
    epoch_tensor = torch.as_tensor(
        retry_epoch, dtype=torch.int64, device=root_address.device
    )
    root_address, path, draw_tensor, lane_tensor, epoch_tensor = (
        torch.broadcast_tensors(
            root_address,
            path,
            draw_tensor,
            lane_tensor,
            epoch_tensor,
        )
    )
    purpose_value = int(purpose)
    if purpose_value <= 0:
        raise ValueError("draw purpose must be a positive namespace identifier")
    namespace = _mix63_scalar(
        (int(seed) & _MASK63)
        ^ ((int(iteration) + 0x6A09E667F3BCC909) & _MASK63)
    )
    namespace = _mix63_scalar(
        namespace ^ ((purpose_value + 0x5BE0CD19137E2179) & _MASK63)
    )
    counter = (
        root_address * 0x1C69B3F74AC4AE35
        + path * 0x3C79AC492BA7B653
        + draw_tensor * 0x1F83D9ABFB41BD6B
        + lane_tensor * 0x4A7484AA6EA6E483
        + epoch_tensor * 0x2E1B21385C26C926
        + namespace
    ) & _MASK63
    keys = _mix63(counter)
    if active is not None:
        if active.device != keys.device:
            raise ValueError("active RNG mask must share the key device")
        active = torch.broadcast_to(active.to(torch.bool), keys.shape)
        keys = torch.where(active, keys, torch.zeros_like(keys))
    return keys


def child_path_keys(parent_path: Tensor, child_ordinal: Tensor | int) -> Tensor:
    """Extend a logical path by its positive one-based child ordinal."""

    ordinal = torch.as_tensor(
        child_ordinal, dtype=torch.int64, device=parent_path.device
    )
    parent_path, ordinal = torch.broadcast_tensors(
        parent_path.to(torch.int64), ordinal
    )
    if parent_path.device.type == "cuda":
        torch._assert_async(
            (ordinal > 0).all(), "child ordinals must be positive"
        )
    elif not bool((ordinal > 0).all()):
        raise ValueError("child ordinals must be positive")
    return _mix63(
        torch.bitwise_xor(
            parent_path,
            (ordinal + 0x243F6A8885A308D3) & _MASK63,
        )
    )


def logical_uniform(
    keys: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    active: Tensor | None = None,
) -> Tensor:
    """Map logical keys to reproducible open-interval uniforms."""

    if keys.dtype != torch.int64:
        raise ValueError("logical RNG keys must be int64")
    mantissa = torch.bitwise_and(keys >> 39, (1 << 24) - 1).to(
        torch.float32
    )
    uniform = ((mantissa + 0.5) * (1.0 / (1 << 24))).to(dtype)
    if active is not None:
        if active.device != keys.device:
            raise ValueError("active uniform mask must share the key device")
        uniform = torch.where(
            torch.broadcast_to(active.to(torch.bool), uniform.shape),
            uniform,
            torch.zeros_like(uniform),
        )
    return uniform


def logical_categorical(
    probability: Tensor,
    keys: Tensor,
    *,
    mass: Tensor | None = None,
    active: Tensor | None = None,
) -> Tensor:
    """Sample validated categorical rows at declared logical addresses."""

    if probability.ndim != 2 or keys.shape != probability.shape[:1]:
        raise ValueError("probability must be [B,E] and keys [B]")
    if probability.device != keys.device or keys.dtype != torch.int64:
        raise ValueError("probability and int64 keys must share a device")
    if mass is None:
        mass = probability.sum(1)
    elif mass.shape != keys.shape or mass.device != probability.device:
        raise ValueError("categorical mass must be device-local [B]")
    cdf = probability.cumsum(1)
    threshold = (logical_uniform(keys) * mass).unsqueeze(1).contiguous()
    selected = torch.searchsorted(cdf, threshold, right=False).squeeze(1)
    selected = selected.clamp_max(probability.shape[1] - 1)
    if active is not None:
        if active.shape != keys.shape or active.device != keys.device:
            raise ValueError("categorical active mask must match keys")
        selected = torch.where(
            active, selected, torch.full_like(selected, -1)
        )
    return selected


@dataclass(frozen=True)
class DeviceStableRoute:
    """Fixed-shape stable destination routing with device-resident counts."""

    source_row: Tensor
    valid: Tensor
    counts: Tensor
    offsets: Tensor
    family_values: Tensor
    declared: Tensor


def device_exclusive_scan(counts: Tensor) -> tuple[Tensor, Tensor]:
    """Return device-only exclusive offsets and totals along the last axis."""

    if counts.dtype != torch.int64 or counts.ndim < 1:
        raise ValueError("scan counts must be an int64 tensor with an axis")
    if counts.device.type != "cuda":
        raise ValueError("queued scans require CUDA")
    torch._assert_async((counts >= 0).all(), "scan counts must be nonnegative")
    inclusive = counts.cumsum(-1)
    offsets = torch.zeros_like(counts)
    offsets[..., 1:] = inclusive[..., :-1]
    return offsets, inclusive[..., -1]


def stable_route(
    family_id: Tensor,
    active: Tensor,
    family_values: Sequence[int],
) -> DeviceStableRoute:
    """Route every active row once while preserving source order per family."""

    values = tuple(int(value) for value in family_values)
    if not values or len(set(values)) != len(values):
        raise ValueError("routing families must be unique and nonempty")
    if (
        family_id.device.type != "cuda"
        or family_id.ndim != 2
        or active.shape != family_id.shape
        or active.dtype != torch.bool
        or active.device != family_id.device
        or family_id.dtype
        not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    ):
        raise ValueError(
            "routing requires CUDA integer family ids and bool active [R,B]"
        )
    declared = torch.tensor(
        values, dtype=family_id.dtype, device=family_id.device
    )
    membership = family_id[..., None] == declared[None, None, :]
    declared_rows = ~active | membership.sum(-1, dtype=torch.int64).eq(1)
    replicas, rows = family_id.shape
    source = torch.arange(
        rows, dtype=torch.int64, device=family_id.device
    ).expand(replicas, rows)
    routed_rows: list[Tensor] = []
    routed_valid: list[Tensor] = []
    for family in range(len(values)):
        selected = active & membership[..., family]
        order = torch.argsort((~selected).to(torch.uint8), dim=1, stable=True)
        routed_rows.append(source.gather(1, order))
        routed_valid.append(selected.gather(1, order))
    source_row = torch.stack(routed_rows, dim=1)
    valid = torch.stack(routed_valid, dim=1)
    counts = valid.sum(-1, dtype=torch.int64)
    offsets, _ = device_exclusive_scan(counts)
    return DeviceStableRoute(
        source_row=source_row,
        valid=valid,
        counts=counts,
        offsets=offsets,
        family_values=declared,
        declared=declared_rows,
    )


@dataclass(frozen=True)
class FamilyQueueState:
    """One scheduler-visible semantic queue description."""

    lane: int
    input_type: str
    preorder: int
    head: int
    ready_tail: int
    logical_time: int
    oldest_time: int
    schema: ModeSchema
    active: bool = True

    def __post_init__(self) -> None:
        if self.lane < 0 or not self.input_type or self.preorder < 0:
            raise ValueError("queue lane, input type, and preorder are invalid")
        if not 0 <= self.head <= self.ready_tail:
            raise ValueError("queue state violates head <= ready_tail")
        if not 0 <= self.oldest_time <= self.logical_time:
            raise ValueError("queue age times are invalid")

    @property
    def depth(self) -> int:
        return self.ready_tail - self.head

    @property
    def age(self) -> int:
        return 0 if not self.depth else self.logical_time - self.oldest_time


@dataclass(frozen=True)
class ScheduledFamily:
    """A compatible, unpadded replica cohort and one common FIFO prefix."""

    input_type: str
    lanes: tuple[int, ...]
    rows: int
    forced_partial: bool

    def __post_init__(self) -> None:
        if (
            not self.input_type
            or not self.lanes
            or len(set(self.lanes)) != len(self.lanes)
            or self.rows <= 0
        ):
            raise ValueError("invalid scheduled family cohort")


class FamilyScheduler:
    """Stable full-first, preorder, load, and age scheduler."""

    def __init__(
        self,
        input_preorder: Sequence[str],
        family_pack_cap: int,
    ) -> None:
        names = tuple(input_preorder)
        if (
            not names
            or len(set(names)) != len(names)
            or family_pack_cap <= 0
        ):
            raise ValueError(
                "scheduler needs unique families and a positive pack cap"
            )
        self.input_preorder = names
        self.family_pack_cap = int(family_pack_cap)
        self._positions = {name: index for index, name in enumerate(names)}

    def select(
        self,
        states: Sequence[FamilyQueueState],
        *,
        allow_partial: bool,
    ) -> ScheduledFamily | None:
        candidates = tuple(
            state
            for state in states
            if state.active and state.depth > 0
        )
        if not candidates:
            return None
        undeclared = sorted(
            {
                state.input_type
                for state in candidates
                if state.input_type not in self._positions
            }
        )
        if undeclared:
            raise RolloutError(f"undeclared queue families: {undeclared}")
        full = tuple(
            state
            for state in candidates
            if state.depth >= self.family_pack_cap
        )
        if full:
            eligible = full
            rows = self.family_pack_cap
            forced_partial = False
        else:
            if not allow_partial:
                return None
            eligible = candidates
            forced_partial = True
            first_preorder = min(
                self._positions[state.input_type] for state in eligible
            )
            layer = tuple(
                state
                for state in eligible
                if self._positions[state.input_type] == first_preorder
            )
            rows = max(state.depth for state in layer)
        first = min(
            eligible,
            key=lambda state: (
                self._positions[state.input_type],
                -state.depth / self.family_pack_cap,
                -state.age,
                state.lane,
            ),
        )
        same_family = tuple(
            state
            for state in candidates
            if state.input_type == first.input_type and state.depth >= rows
        )
        validate_replica_group(tuple(state.schema for state in same_family))
        lanes = tuple(sorted(state.lane for state in same_family))
        return ScheduledFamily(
            input_type=first.input_type,
            lanes=lanes,
            rows=rows,
            forced_partial=forced_partial,
        )


@dataclass(frozen=True)
class FamilyBatch:
    """One stable direct model batch gathered from monotone FamilyWork."""

    input_type: str
    lanes: Tensor
    row: Tensor
    fields: Mapping[str, Tensor]

    @property
    def lane_count(self) -> int:
        return int(self.lanes.numel())

    @property
    def rows(self) -> int:
        return int(self.row.shape[1])


class SemanticFamilyFIFO:
    """FIFO view over one declared ``FamilyWork[input_type]`` slab."""

    def __init__(
        self,
        input_type: str,
        partitions: PartitionContract,
        family: MonotoneSlab,
        metadata: MonotoneSlab,
    ) -> None:
        declared = tuple(item.name for item in partitions.inputs)
        if input_type not in declared:
            raise RolloutError(f"undeclared queue family {input_type!r}")
        if family.schema.name != f"FamilyWork.{input_type}":
            raise RolloutError("family FIFO is bound to the wrong owner")
        if metadata.schema.name != f"QueueMetadata.{input_type}":
            raise RolloutError("family FIFO metadata is bound to the wrong owner")
        if family.n_replica != metadata.n_replica or family.device != metadata.device:
            raise RolloutError("family FIFO owners disagree on replica/device")
        required = {"head", "ready_tail", "reserved_tail"}
        if not required.issubset(metadata.columns):
            raise RolloutError("queue metadata is missing its closed tail fields")
        self.input_type = input_type
        self.partitions = partitions
        self.family = family
        self.metadata = metadata
        self.device = family.device
        self.n_replica = family.n_replica
        self._ready_event = torch.cuda.Event(blocking=False)
        self._has_ready_event = False

    def _record_ready(
        self, stream: torch.cuda.Stream | None = None
    ) -> None:
        self._ready_event.record(
            torch.cuda.current_stream(self.device)
            if stream is None
            else stream
        )
        self._has_ready_event = True

    def wait_ready(
        self, stream: torch.cuda.Stream | None = None
    ) -> None:
        if self._has_ready_event:
            (
                torch.cuda.current_stream(self.device)
                if stream is None
                else stream
            ).wait_event(self._ready_event)

    def initialize(self) -> None:
        """Publish the single mutable metadata row for every replica."""

        values = {
            field.name: torch.full(
                (self.n_replica, 1, *field.shape),
                field.fill,
                dtype=field.dtype,
                device=self.device,
            )
            for field in self.metadata.schema.fields
        }
        counts = torch.ones(
            self.n_replica, dtype=torch.int64, device=self.device
        )
        transaction = DevicePublicationTransaction.begin(
            (
                DeviceDestinationRequest(
                    slab=self.metadata,
                    counts=counts,
                    values=values,
                    input_type=self.input_type,
                    logical_address=torch.zeros_like(counts),
                ),
            ),
            phase="queue_initialization",
            active=torch.ones(
                self.n_replica, dtype=torch.bool, device=self.device
            ),
        )
        transaction.fill_all()
        transaction.publish()
        self._record_ready()

    @property
    def head(self) -> Tensor:
        return self.metadata.columns["head"][:, 0]

    @property
    def ready_tail(self) -> Tensor:
        return self.metadata.columns["ready_tail"][:, 0]

    @property
    def reserved_tail(self) -> Tensor:
        return self.metadata.columns["reserved_tail"][:, 0]

    @property
    def depth(self) -> Tensor:
        return self.ready_tail - self.head

    def publish_family_tail(self, active: Tensor) -> None:
        """Expose newly committed family rows without reading a host scalar."""

        if (
            active.shape != (self.n_replica,)
            or active.dtype != torch.bool
            or active.device != self.device
        ):
            raise ValueError("queue publication mask must be bool [n_replica]")
        published = self.family._state[:, MonotoneSlab._PUBLISHED].to(
            torch.int32
        )
        reserved = self.family._state[:, MonotoneSlab._RESERVED].to(
            torch.int32
        )
        self.ready_tail.copy_(
            torch.where(active, published, self.ready_tail)
        )
        self.reserved_tail.copy_(
            torch.where(active, reserved, self.reserved_tail)
        )
        torch._assert_async(
            (
                (self.head >= 0)
                & (self.head <= self.ready_tail)
                & (self.ready_tail <= self.reserved_tail)
            ).all(),
            "semantic queue tail order is invalid",
        )
        self._record_ready()

    def stable_prefix(self, scheduled: ScheduledFamily) -> FamilyBatch:
        if scheduled.input_type != self.input_type:
            raise RolloutError("scheduled family does not match this FIFO")
        self.wait_ready()
        lanes = torch.tensor(
            scheduled.lanes, dtype=torch.int64, device=self.device
        )
        width = scheduled.rows
        torch._assert_async(
            (self.depth[lanes] >= width).all(),
            "scheduled family prefix is not ready",
        )
        ordinal = torch.arange(width, dtype=torch.int64, device=self.device)
        row = self.head[lanes].to(torch.int64)[:, None] + ordinal[None, :]
        fields = MappingProxyType(
            {
                name: value[lanes[:, None], row]
                for name, value in self.family.columns.items()
            }
        )
        return FamilyBatch(self.input_type, lanes, row, fields)

    def consume(self, lanes: Tensor, rows: int, success: Tensor) -> None:
        if rows <= 0:
            raise ValueError("queue consumption rows must be positive")
        if (
            lanes.ndim != 1
            or lanes.dtype != torch.int64
            or lanes.device != self.device
            or success.shape != (self.n_replica,)
            or success.dtype != torch.bool
            or success.device != self.device
        ):
            raise ValueError("queue consumption tensors have invalid contracts")
        selected = success[lanes]
        old = self.head[lanes]
        new = old + rows
        torch._assert_async(
            (~selected | (new <= self.ready_tail[lanes])).all(),
            "queue consumption exceeds its stable prefix",
        )
        self.head[lanes] = torch.where(selected, new, old)
        self._record_ready()


@dataclass(frozen=True)
class ModelWorkspace:
    """Raw model outputs owned only until transition materialization."""

    values: Mapping[str, Tensor]


@dataclass(frozen=True)
class SampleWorkspace:
    """Labels and descriptors produced by sampling, never retained storage."""

    values: Mapping[str, Tensor]
    draw_count: Tensor


@dataclass(frozen=True)
class SampleRequest:
    schema: ModeSchema
    batch: FamilyBatch
    model: ModelWorkspace
    rng_key: Tensor


@dataclass(frozen=True)
class TransitionRequest:
    schema: ModeSchema
    batch: FamilyBatch
    model: ModelWorkspace
    sample: SampleWorkspace


@dataclass(frozen=True)
class TopologyDraftPacket:
    """A sampled compact packet whose listed endpoints await commit."""

    sequence: int
    node_begin: int
    edge_begin: int
    topology: Topology
    bindings: TopologyBindings
    unresolved_edges: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.sequence, self.node_begin, self.edge_begin) < 0:
            raise TopologyError("draft packet addresses must be nonnegative")
        if len(set(self.unresolved_edges)) != len(self.unresolved_edges):
            raise TopologyError("draft packet repeats an unresolved endpoint")
        if any(
            edge < 0 or edge >= self.topology.edge_count
            for edge in self.unresolved_edges
        ):
            raise TopologyError("draft endpoint is outside its edge table")
        self.bindings.validate(
            self.topology.node_count, self.topology.edge_count
        )


@dataclass(frozen=True)
class EndpointPatch:
    """Commit-time endpoint rows for one sampled topology packet."""

    sequence: int
    edge: Tensor
    target_kind: Tensor
    target_row: Tensor
    child_output_type: Tensor
    commit_sequence: int

    def __post_init__(self) -> None:
        count = int(self.edge.numel())
        for name in (
            "edge",
            "target_kind",
            "target_row",
            "child_output_type",
        ):
            value = getattr(self, name)
            if (
                value.device.type != "cpu"
                or value.dtype != torch.int32
                or value.shape != (count,)
                or not value.is_contiguous()
            ):
                raise TopologyError(
                    f"endpoint patch {name} must be contiguous CPU int32 [{count}]"
                )
        if self.sequence < 0 or self.commit_sequence != self.sequence:
            raise TopologyError("endpoint patch lacks its commit marker")


@dataclass(frozen=True)
class TopologyPublication:
    """One replica-local draft/patch pair tied to a device commit lane."""

    lane: int
    draft: TopologyDraftPacket
    patch: EndpointPatch

    def __post_init__(self) -> None:
        if self.lane < 0:
            raise TopologyError("topology publication lane is negative")
        if self.draft.sequence != self.patch.sequence:
            raise TopologyError("topology draft and patch sequences differ")


@dataclass(frozen=True)
class _PendingTopologyPublication:
    publication: TopologyPublication
    marker: Tensor
    event: torch.cuda.Event


class AsyncTopologyPublisher:
    """Accept topology descriptors only after endpoint commit visibility."""

    def __init__(
        self, topology_packet_cap: int, *, max_inflight_packets: int = 2
    ) -> None:
        if topology_packet_cap <= 0 or max_inflight_packets <= 0:
            raise ValueError("topology publication bounds must be positive")
        self.topology_packet_cap = int(topology_packet_cap)
        self.max_inflight_packets = int(max_inflight_packets)
        self._drafts: dict[int, TopologyDraftPacket] = {}
        self._packets: list[TopologyPacket] = []
        self._pending: list[_PendingTopologyPublication] = []

    def stage(self, draft: TopologyDraftPacket) -> None:
        if draft.sequence in self._drafts or any(
            packet.sequence == draft.sequence for packet in self._packets
        ):
            raise TopologyError("duplicate topology transaction sequence")
        if (
            draft.topology.node_count + draft.topology.edge_count
            > self.topology_packet_cap
        ):
            raise TopologyError("topology draft exceeds its fixed packet cap")
        inflight = set(self._drafts).union(
            item.publication.draft.sequence for item in self._pending
        )
        if len(inflight) >= self.max_inflight_packets:
            raise TopologyError("topology publication exceeds its in-flight cap")
        self._drafts[draft.sequence] = draft

    @staticmethod
    def _materialize(
        draft: TopologyDraftPacket, patch: EndpointPatch
    ) -> TopologyPacket:
        expected = draft.unresolved_edges
        actual = tuple(int(value) for value in patch.edge.tolist())
        if actual != expected:
            raise TopologyError(
                "endpoint patch must cover unresolved edges in stable order"
            )
        topology_columns = {
            name: getattr(draft.topology, name).clone()
            for name in (
                "parent",
                "child_begin",
                "executed_degree",
                "nominal_degree",
                "target_degree",
                "owner",
                "familywork_row",
                "sigma_head_row",
                "action_begin",
                "label",
                "path_length",
                "target_kind",
                "target_row",
            )
        }
        binding_columns = {
            "input_type": draft.bindings.input_type.clone(),
            "output_type": draft.bindings.output_type.clone(),
            "sigma_target_row": draft.bindings.sigma_target_row.clone(),
            "child_output_type": draft.bindings.child_output_type.clone(),
        }
        if actual:
            edge = patch.edge.to(torch.int64)
            topology_columns["target_kind"][edge] = patch.target_kind
            topology_columns["target_row"][edge] = patch.target_row
            binding_columns["child_output_type"][
                edge
            ] = patch.child_output_type
        topology = Topology(**topology_columns)
        bindings = TopologyBindings(**binding_columns)
        packet = TopologyPacket(
            sequence=draft.sequence,
            node_begin=draft.node_begin,
            edge_begin=draft.edge_begin,
            commit_sequence=patch.commit_sequence,
            topology=topology,
            bindings=bindings,
        )
        return packet

    def preflight(
        self, draft: TopologyDraftPacket, patch: EndpointPatch
    ) -> None:
        """Prove endpoint closure before any retained destination reserves."""

        if draft.sequence != patch.sequence:
            raise TopologyError("topology draft and patch sequences differ")
        self._materialize(draft, patch)

    def commit(self, patch: EndpointPatch) -> TopologyPacket:
        try:
            draft = self._drafts[patch.sequence]
        except KeyError as exc:
            raise TopologyError("endpoint patch has no staged packet") from exc
        packet = self._materialize(draft, patch)
        del self._drafts[patch.sequence]
        self._packets.append(packet)
        return packet

    def defer(
        self,
        publication: TopologyPublication,
        marker: Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Queue CPU acceptance behind the CUDA transaction commit event."""

        if (
            marker.device.type != "cuda"
            or marker.dtype != torch.int64
            or marker.ndim != 1
            or publication.lane >= marker.numel()
        ):
            raise TopologyError(
                "topology commit marker must be CUDA int64 [n_replica]"
            )
        if publication.draft.sequence not in self._drafts:
            raise TopologyError("deferred topology draft was not staged")
        event = torch.cuda.Event(blocking=False)
        event.record(
            torch.cuda.current_stream(marker.device)
            if stream is None
            else stream
        )
        self._pending.append(
            _PendingTopologyPublication(publication, marker, event)
        )

    def drain(self) -> tuple[TopologyPacket, ...]:
        """Wait once per transfer and accept only visible commit markers."""

        accepted: list[TopologyPacket] = []
        pending, self._pending = self._pending, []
        for item in pending:
            item.event.synchronize()
            publication = item.publication
            marker = int(item.marker[publication.lane].item())
            if marker == publication.draft.sequence:
                accepted.append(self.commit(publication.patch))
            else:
                self.abort(publication.draft.sequence)
        return tuple(accepted)

    def abort(self, sequence: int) -> None:
        if sequence not in self._drafts:
            raise TopologyError("cannot abort an unknown topology sequence")
        del self._drafts[sequence]

    @property
    def committed_packets(self) -> tuple[TopologyPacket, ...]:
        return tuple(sorted(self._packets, key=lambda packet: packet.sequence))

    def assert_drained(self) -> None:
        if self._drafts or self._pending:
            raise TopologyError("uncommitted topology packets remain staged")


@dataclass(frozen=True)
class CommitPlan:
    """Every retained destination and structural effect of one transition."""

    destinations: tuple[DeviceDestinationRequest, ...]
    published_queues: tuple[SemanticFamilyFIFO, ...] = ()
    topology: tuple[TopologyPublication, ...] = ()

    def __post_init__(self) -> None:
        if not self.destinations:
            raise ValueError("a commit plan requires retained destinations")
        sequences = tuple(item.draft.sequence for item in self.topology)
        if len(set(sequences)) != len(sequences):
            raise ValueError("a commit plan repeats a topology sequence")
        lanes = tuple(item.lane for item in self.topology)
        if len(set(lanes)) != len(lanes):
            raise ValueError(
                "a family commit may stage only one topology packet per lane"
            )


class RolloutHooks(Protocol):
    """Opaque synthetic/game boundary used by the rollout transaction."""

    def model_input(self, batch: FamilyBatch) -> object: ...

    def sample(self, request: SampleRequest) -> SampleWorkspace: ...

    def transition(self, request: TransitionRequest) -> CommitPlan: ...


class _MachineState(Enum):
    READY = "ready"
    MODELED = "modeled"
    SAMPLED = "sampled"
    COMMITTED = "committed"
    FAILED = "failed"


@dataclass(frozen=True)
class RolloutCommit:
    success: Tensor
    failed: Tensor
    sequence: int
    draw_count: Tensor
    event: torch.cuda.Event


class RolloutTransaction:
    """The sole generic ``model -> sample -> commit`` state machine."""

    def __init__(
        self,
        schema: ModeSchema,
        *,
        seed: int,
        iteration: int,
        topology: AsyncTopologyPublisher,
        retained_slabs: Sequence[MonotoneSlab],
        model_stream: torch.cuda.Stream | None = None,
        commit_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.schema = schema
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.topology = topology
        self.retained_slabs = tuple(retained_slabs)
        if not self.retained_slabs:
            raise ValueError("rollout transaction needs its retained owner set")
        self.device = self.retained_slabs[0].device
        self.n_replica = self.retained_slabs[0].n_replica
        if any(
            slab.device != self.device or slab.n_replica != self.n_replica
            for slab in self.retained_slabs
        ):
            raise ValueError("retained rollout owners disagree on device/replicas")
        self.model_stream = (
            torch.cuda.Stream(device=self.device)
            if model_stream is None
            else model_stream
        )
        self.commit_stream = (
            torch.cuda.Stream(device=self.device)
            if commit_stream is None
            else commit_stream
        )
        if self.model_stream == self.commit_stream:
            raise ValueError("model/sample and commit streams must be distinct")
        self._state = _MachineState.READY

    def execute(
        self,
        source: SemanticFamilyFIFO,
        scheduled: ScheduledFamily,
        model: Callable[[object, Tensor], Mapping[str, Tensor]],
        hooks: RolloutHooks,
        *,
        purpose: DrawPurpose = DrawPurpose.ACTION,
        retry_epoch: Tensor | int = 0,
        sampler_attempt_count: Tensor | None = None,
        failure_injector: Callable[[], None] | None = None,
    ) -> RolloutCommit:
        if self._state not in (_MachineState.READY, _MachineState.COMMITTED):
            raise RolloutError(
                f"rollout transaction cannot start from {self._state.value}"
            )
        self._state = _MachineState.READY
        with torch.cuda.stream(self.model_stream):
            batch = source.stable_prefix(scheduled)
            before = tuple(
                (
                    slab._state._version,
                    tuple(
                        (name, value._version)
                        for name, value in slab.columns.items()
                    ),
                )
                for slab in self.retained_slabs
            )
            model_input = hooks.model_input(batch)
            model_workspace = ModelWorkspace(
                MappingProxyType(dict(model(model_input, batch.lanes)))
            )
            self._state = _MachineState.MODELED
            logical = batch.fields["logical_rng_key"]
            if logical.shape[-1] != 2:
                raise RolloutError(
                    "FamilyWork.logical_rng_key must end in [root,path]"
                )
            lane = batch.lanes[:, None].expand_as(logical[..., 0])
            active = torch.ones_like(logical[..., 0], dtype=torch.bool)
            rng_key = logical_rng_keys(
                seed=self.seed,
                iteration=self.iteration,
                root=logical[..., 0],
                path=logical[..., 1],
                draw=0,
                purpose=purpose,
                lane=lane,
                retry_epoch=retry_epoch,
                active=active,
            )
            sampled = hooks.sample(
                SampleRequest(self.schema, batch, model_workspace, rng_key)
            )
            if (
                sampled.draw_count.shape != (batch.lane_count,)
                or sampled.draw_count.dtype != torch.int64
                or sampled.draw_count.device != self.device
            ):
                raise RolloutError(
                    "sample draw_count must be CUDA int64 [active_lanes]"
                )
            full_draw_count = torch.zeros(
                self.n_replica, dtype=torch.int64, device=self.device
            )
            full_draw_count[batch.lanes] = sampled.draw_count
            for slab, (state_version, column_versions) in zip(
                self.retained_slabs, before
            ):
                current_columns = tuple(
                    (name, value._version)
                    for name, value in slab.columns.items()
                )
                if (
                    slab._state._version != state_version
                    or current_columns != column_versions
                ):
                    self._state = _MachineState.FAILED
                    raise RolloutError(
                        "sample wrote or published a retained destination"
                    )
            self._state = _MachineState.SAMPLED
            plan = hooks.transition(
                TransitionRequest(
                    self.schema, batch, model_workspace, sampled
                )
            )
            active_lanes = torch.zeros(
                self.n_replica, dtype=torch.bool, device=self.device
            )
            active_lanes[batch.lanes] = True
            marker = torch.full(
                (self.n_replica,),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            sample_ready = torch.cuda.Event(blocking=False)
            sample_ready.record(self.model_stream)
        for publication in plan.topology:
            if publication.lane >= self.n_replica:
                raise RolloutError("topology publication lane is outside replicas")
            self.topology.preflight(publication.draft, publication.patch)
        staged: list[int] = []
        transaction: DevicePublicationTransaction | None = None
        try:
            with torch.cuda.stream(self.commit_stream):
                self.commit_stream.wait_event(sample_ready)
                transaction = DevicePublicationTransaction.begin(
                    plan.destinations,
                    phase="rollout_commit",
                    active=active_lanes,
                    sampler_attempt_count=sampler_attempt_count,
                )
                for publication in plan.topology:
                    self.topology.stage(publication.draft)
                    staged.append(publication.draft.sequence)
                transaction.fill_all()
                success = transaction.publish(
                    commit_marker=marker,
                    sequence=0,
                    failure_injector=failure_injector,
                )
                for queue in plan.published_queues:
                    queue.publish_family_tail(success)
                source.consume(batch.lanes, scheduled.rows, success)
                for publication in plan.topology:
                    marker[publication.lane].copy_(
                        torch.where(
                            success[publication.lane],
                            torch.as_tensor(
                                publication.draft.sequence,
                                dtype=torch.int64,
                                device=self.device,
                            ),
                            marker[publication.lane],
                        )
                    )
                    self.topology.defer(
                        publication, marker, stream=self.commit_stream
                    )
                commit_ready = torch.cuda.Event(blocking=False)
                commit_ready.record(self.commit_stream)
        except BaseException:
            if transaction is not None:
                transaction.abort()
            for sequence in staged:
                self.topology.abort(sequence)
            self._state = _MachineState.FAILED
            raise
        sequence = 0 if not plan.topology else plan.topology[-1].draft.sequence
        self._state = _MachineState.COMMITTED
        return RolloutCommit(
            success=success,
            failed=active_lanes & ~success,
            sequence=sequence,
            draw_count=full_draw_count,
            event=commit_ready,
        )


@dataclass(frozen=True)
class ChanceFrontier:
    """One fixed-capacity chance frontier with device-resident validity."""

    active: Tensor
    root: Tensor
    path: Tensor
    payload: Mapping[str, Tensor]

    def validate(self, capacity: int) -> None:
        if (
            self.active.device.type != "cuda"
            or self.active.dtype != torch.bool
            or self.active.ndim != 2
            or self.active.shape[1] != capacity
        ):
            raise RolloutError(
                "chance active mask must be CUDA bool [n_replica, capacity]"
            )
        if (
            self.root.shape != self.active.shape
            or self.path.shape != self.active.shape
            or self.root.dtype != torch.int64
            or self.path.dtype != torch.int64
            or self.root.device != self.active.device
            or self.path.device != self.active.device
        ):
            raise RolloutError("chance logical addresses must match its frontier")
        for name, value in self.payload.items():
            if (
                not name
                or value.shape[:2] != self.active.shape
                or value.device != self.active.device
            ):
                raise RolloutError("chance payload violates its fixed frontier")


class ChanceSampler(Protocol):
    def __call__(
        self,
        frontier: ChanceFrontier,
        rng_key: Tensor,
        generation: int,
    ) -> object: ...


class ChanceCommitter(Protocol):
    def __call__(
        self,
        frontier: ChanceFrontier,
        sampled: object,
        generation: int,
    ) -> ChanceFrontier: ...


@dataclass(frozen=True)
class ChanceClosureResult:
    frontier: ChanceFrontier
    generation_draws: Tensor


class BoundedChanceClosure:
    """Run a fixed launch envelope without synchronizing between generations."""

    def __init__(
        self,
        *,
        seed: int,
        iteration: int,
        capacity: int,
        max_generations: int,
    ) -> None:
        if capacity <= 0 or max_generations <= 0:
            raise ValueError("chance closure bounds must be positive")
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.capacity = int(capacity)
        self.max_generations = int(max_generations)

    def run(
        self,
        frontier: ChanceFrontier,
        sample: ChanceSampler,
        commit: ChanceCommitter,
    ) -> ChanceClosureResult:
        frontier.validate(self.capacity)
        draws: list[Tensor] = []
        lanes = torch.arange(
            frontier.active.shape[0],
            dtype=torch.int64,
            device=frontier.active.device,
        )[:, None]
        for generation in range(self.max_generations):
            key = logical_rng_keys(
                seed=self.seed,
                iteration=self.iteration,
                root=frontier.root,
                path=frontier.path,
                draw=generation,
                purpose=DrawPurpose.CHANCE,
                lane=lanes,
                active=frontier.active,
            )
            sampled = sample(frontier, key, generation)
            draws.append(frontier.active.sum(dtype=torch.int64))
            frontier = commit(frontier, sampled, generation)
            frontier.validate(self.capacity)
        return ChanceClosureResult(
            frontier=frontier,
            generation_draws=torch.stack(draws),
        )


@dataclass(frozen=True)
class CompletionBatch:
    lanes: Tensor
    row: Tensor
    fields: Mapping[str, Tensor]


class CompletionHook(Protocol):
    def __call__(
        self, batch: CompletionBatch
    ) -> tuple[DeviceDestinationRequest, ...]: ...


class CompletionFIFO:
    """Stable, bounded ``Leaf`` prefixes feeding the completion hook."""

    def __init__(self, leaf: MonotoneSlab, completion_pack_cap: int) -> None:
        if leaf.schema.name != "Leaf" or completion_pack_cap <= 0:
            raise ValueError("completion FIFO requires Leaf and a positive pack")
        if completion_pack_cap > leaf.capacity:
            raise ValueError("completion pack exceeds Leaf capacity")
        self.leaf = leaf
        self.device = leaf.device
        self.n_replica = leaf.n_replica
        self.completion_pack_cap = int(completion_pack_cap)
        self.head = torch.zeros(
            self.n_replica, dtype=torch.int64, device=self.device
        )

    @property
    def ready_tail(self) -> Tensor:
        return self.leaf._state[:, MonotoneSlab._PUBLISHED]

    @property
    def depth(self) -> Tensor:
        return self.ready_tail - self.head

    def stable_prefix(
        self, lanes: Sequence[int], rows: int
    ) -> CompletionBatch:
        if rows <= 0 or rows > self.completion_pack_cap:
            raise ValueError("completion rows are outside its pack bound")
        lane_values = tuple(int(lane) for lane in lanes)
        lane = torch.tensor(lanes, dtype=torch.int64, device=self.device)
        if not lane_values or len(set(lane_values)) != len(lane_values):
            raise ValueError("completion lanes must be unique and nonempty")
        torch._assert_async(
            (self.depth[lane] >= rows).all(),
            "completion prefix is not ready",
        )
        ordinal = torch.arange(rows, dtype=torch.int64, device=self.device)
        row = self.head[lane, None] + ordinal[None, :]
        fields = MappingProxyType(
            {
                name: value[lane[:, None], row]
                for name, value in self.leaf.columns.items()
            }
        )
        return CompletionBatch(lane, row, fields)

    def complete(
        self,
        lanes: Sequence[int],
        rows: int,
        hook: CompletionHook,
        *,
        sampler_attempt_count: Tensor | None = None,
        wait_for: torch.cuda.Event | None = None,
    ) -> Tensor:
        if wait_for is not None:
            torch.cuda.current_stream(self.device).wait_event(wait_for)
        batch = self.stable_prefix(lanes, rows)
        requests = hook(batch)
        active = torch.zeros(
            self.n_replica, dtype=torch.bool, device=self.device
        )
        active[batch.lanes] = True
        transaction = DevicePublicationTransaction.begin(
            requests,
            phase="completion",
            active=active,
            sampler_attempt_count=sampler_attempt_count,
        )
        transaction.fill_all()
        success = transaction.publish()
        old = self.head[batch.lanes]
        self.head[batch.lanes] = torch.where(
            success[batch.lanes], old + rows, old
        )
        return success

    def reset_failed_lane(self, lane: int) -> None:
        """Reset completion consumption after the owner attempt is discarded."""

        if lane < 0 or lane >= self.n_replica:
            raise ValueError("completion retry lane is outside replicas")
        self.head[lane].zero_()


__all__ = [
    "AsyncTopologyPublisher",
    "BoundedChanceClosure",
    "ChanceClosureResult",
    "ChanceFrontier",
    "CommitPlan",
    "CompletionBatch",
    "CompletionFIFO",
    "DeviceStableRoute",
    "DrawPurpose",
    "EndpointPatch",
    "FamilyBatch",
    "FamilyQueueState",
    "FamilyScheduler",
    "LOGICAL_RNG_VERSION",
    "ModelWorkspace",
    "RolloutCommit",
    "RolloutError",
    "RolloutHooks",
    "RolloutTransaction",
    "SampleRequest",
    "SampleWorkspace",
    "ScheduledFamily",
    "SemanticFamilyFIFO",
    "TopologyDraftPacket",
    "TopologyPublication",
    "TransitionRequest",
    "child_path_keys",
    "device_exclusive_scan",
    "logical_categorical",
    "logical_rng_keys",
    "logical_uniform",
    "stable_route",
]
