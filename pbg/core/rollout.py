"""Semantic queues and the sole queued rollout transaction.

The module is game-opaque.  It schedules stable ``FamilyWork`` prefixes and
executes one ``model -> sample -> transition -> commit`` operation for both
sequential and branched schemas.  Retained writes are possible only through
the device-counted publication transaction owned by :mod:`pbg.core.storage`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cache
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor

from pbg.core.capacity import ExecutionPhase
from pbg.core.contracts import (
    ModeSchema,
    PartitionContract,
    validate_replica_group,
)
from pbg.core.storage import (
    DeviceDestinationRequest,
    MonotoneSlab,
    PublicationTransaction,
    SoASchema,
)
from pbg.core.topology import (
    StagedTopologyPacket,
    Topology,
    TopologyBindings,
    TopologyError,
    TopologyPacketStager,
)


@cache
def _device_streams(
    device_index: int,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream, torch.cuda.Stream]:
    """Own one process-lifetime rollout stream set per CUDA device.

    cuBLAS retains an allocator-visible workspace for every stream handle it
    observes. Recreating these streams per iteration would therefore retain a
    new workspace after the Python rollout owners are released.
    """

    device = torch.device("cuda", device_index)
    return (
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
    )


def _runtime_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream, torch.cuda.Stream]:
    index = torch.cuda.current_device() if device.index is None else device.index
    return _device_streams(index)


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
    ROLE_SELECTION = 8


LOGICAL_RNG_VERSION = "counter_hash_v4"
_MASK63 = (1 << 63) - 1
_OWNER_REGISTRY_TOKEN = object()


def _int64_like(reference: Tensor, value: Tensor | int) -> Tensor:
    """Materialize scalar counters without a host-to-device scalar copy."""

    if isinstance(value, Tensor):
        return value.to(dtype=torch.int64, device=reference.device)
    return torch.full_like(reference, int(value), dtype=torch.int64)


def _mix63(value: Tensor) -> Tensor:
    first = torch.bitwise_xor(value, torch.bitwise_right_shift(value, 30))
    first = torch.bitwise_and(first * 0x3C79AC492BA7B653, _MASK63)
    second = torch.bitwise_xor(first, torch.bitwise_right_shift(first, 27))
    second = torch.bitwise_and(second * 0x1C69B3F74AC4AE35, _MASK63)
    return torch.bitwise_and(
        torch.bitwise_xor(second, torch.bitwise_right_shift(second, 31)),
        _MASK63,
    )


def _scalar_counter_hash(value: int) -> int:
    """Scalar form of the versioned counter hash used for namespaces."""

    value &= _MASK63
    for shift, multiplier in (
        (30, 0x3C79AC492BA7B653),
        (27, 0x1C69B3F74AC4AE35),
    ):
        value = ((value ^ (value >> shift)) * multiplier) & _MASK63
    return (value ^ (value >> 31)) & _MASK63


def logical_rng_keys(
    *,
    seed: int,
    iteration: int,
    root: Tensor,
    path: Tensor,
    draw: Tensor | int,
    purpose: DrawPurpose | int,
    lane: Tensor | int = 0,
    retry_epoch: Tensor | int = 0,
    active: Tensor | None = None,
) -> Tensor:
    """Return placement-, subdivision-, and schedule-independent CUDA keys.

    Inactive addresses receive the sentinel zero and therefore do not consume
    a logical draw.
    """

    if root.device != path.device:
        raise ValueError("root and path addresses must share a device")
    root_address, path = torch.broadcast_tensors(
        root.to(torch.int64), path.to(torch.int64)
    )
    draw_tensor = _int64_like(root_address, draw)
    lane_tensor = _int64_like(root_address, lane)
    epoch_tensor = _int64_like(root_address, retry_epoch)
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
    namespace = _scalar_counter_hash(
        (int(seed) & _MASK63)
        ^ ((int(iteration) + 0x6A09E667F3BCC909) & _MASK63)
    )
    namespace = _scalar_counter_hash(
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

    ordinal = _int64_like(parent_path, child_ordinal)
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
    uniform = torch.minimum(
        uniform,
        torch.nextafter(
            torch.ones((), dtype=dtype, device=keys.device),
            torch.zeros((), dtype=dtype, device=keys.device),
        ),
    )
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
    threshold = torch.minimum(
        logical_uniform(keys) * mass,
        torch.nextafter(mass, torch.zeros_like(mass)),
    ).unsqueeze(1).contiguous()
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
    *,
    declared_values: Tensor | None = None,
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
    if declared_values is None:
        declared = torch.tensor(
            values, dtype=family_id.dtype, device=family_id.device
        )
    else:
        declared = declared_values
        if (
            declared.device != family_id.device
            or declared.dtype != family_id.dtype
            or declared.shape != (len(values),)
        ):
            raise ValueError(
                "preallocated routing families disagree with the route contract"
            )
    membership = family_id[..., None] == declared[None, None, :]
    declared_rows = ~active | membership.sum(-1, dtype=torch.int64).eq(1)
    torch._assert_async(
        declared_rows.all(),
        "every active row must have exactly one declared queue family",
    )
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
    max_prefix: int | None = None

    def __post_init__(self) -> None:
        if self.lane < 0 or not self.input_type or self.preorder < 0:
            raise ValueError("queue lane, input type, and preorder are invalid")
        if not 0 <= self.head <= self.ready_tail:
            raise ValueError("queue state violates head <= ready_tail")
        if not 0 <= self.oldest_time <= self.logical_time:
            raise ValueError("queue age times are invalid")
        if self.max_prefix is not None and self.max_prefix <= 0:
            raise ValueError("queue maximum prefix must be positive")

    @property
    def depth(self) -> int:
        return self.ready_tail - self.head

    @property
    def age(self) -> int:
        return 0 if not self.depth else self.logical_time - self.oldest_time

    def prefix_cap(self, global_cap: int) -> int:
        return (
            global_cap
            if self.max_prefix is None
            else min(global_cap, self.max_prefix)
        )


@dataclass(frozen=True)
class ScheduledFamily:
    """A compatible, unpadded replica cohort and one common FIFO prefix."""

    input_type: str
    lanes: tuple[int, ...]
    lane_schemas: tuple[ModeSchema, ...]
    rows: int
    forced_partial: bool

    def __post_init__(self) -> None:
        if (
            not self.input_type
            or not self.lanes
            or len(set(self.lanes)) != len(self.lanes)
            or len(self.lane_schemas) != len(self.lanes)
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
            if state.depth >= state.prefix_cap(self.family_pack_cap)
        )
        if full:
            eligible = full
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
            rows = max(
                min(state.depth, state.prefix_cap(self.family_pack_cap))
                for state in layer
            )
        first = min(
            eligible,
            key=lambda state: (
                self._positions[state.input_type],
                -state.depth / state.prefix_cap(self.family_pack_cap),
                -state.age,
                state.lane,
            ),
        )
        if full:
            rows = first.prefix_cap(self.family_pack_cap)
        same_family = tuple(
            state
            for state in candidates
            if state.input_type == first.input_type
            and state.depth >= rows
            and state.prefix_cap(self.family_pack_cap) >= rows
        )
        validate_replica_group(tuple(state.schema for state in same_family))
        lanes = tuple(sorted(state.lane for state in same_family))
        schema_by_lane = {state.lane: state.schema for state in same_family}
        return ScheduledFamily(
            input_type=first.input_type,
            lanes=lanes,
            lane_schemas=tuple(schema_by_lane[lane] for lane in lanes),
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
    lane_ids: tuple[int, ...] | None = None

    @property
    def lane_count(self) -> int:
        return int(self.lanes.numel())

    @property
    def rows(self) -> int:
        return int(self.row.shape[1])


class FamilyStorage:
    """One logical family row split across hot and retained lifetimes."""

    def __init__(self, hot: MonotoneSlab, retained: MonotoneSlab) -> None:
        if not hot.schema.name.startswith("FamilyWork."):
            raise RolloutError("family hot storage has the wrong owner")
        input_type = hot.schema.name.split(".", 1)[1]
        if retained.schema.name != f"FamilyRetained.{input_type}":
            raise RolloutError("family retained storage has the wrong owner")
        if (
            hot.capacity != retained.capacity
            or hot.n_replica != retained.n_replica
            or hot.device != retained.device
            or hot.failure_state is not retained.failure_state
        ):
            raise RolloutError("family lifetime owners are physically incompatible")
        names = tuple(field.name for field in (*hot.schema.fields, *retained.schema.fields))
        if len(set(names)) != len(names):
            raise RolloutError("family lifetime owners overlap fields")
        self.hot = hot
        self.retained = retained
        self.schema = SoASchema(
            name=hot.schema.name,
            fields=(*hot.schema.fields, *retained.schema.fields),
            lifetime="root_commit_to_final_family_consumer",
        )
        self.capacity = hot.capacity
        self.n_replica = hot.n_replica
        self.device = hot.device
        self.failure_state = hot.failure_state
        self.label = hot.label

    @property
    def columns(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {**dict(self.hot.columns), **dict(self.retained.columns)}
        )

    @property
    def _state(self) -> Tensor:
        return self.hot._state

    def _assert_aligned(self) -> None:
        torch._assert_async(
            (
                self.hot._state[:, :2]
                == self.retained._state[:, :2]
            ).all(),
            "family hot and retained tails diverged",
        )

    def tail(self, lane: int) -> int:
        hot_tail = self.hot.tail(lane)
        if self.retained.tail(lane) != hot_tail:
            raise RolloutError("family hot and retained tails diverged")
        return hot_tail

    def reserved_tails_device(self) -> Tensor:
        self._assert_aligned()
        return self.hot.reserved_tails_device()

    def destination_requests(
        self,
        counts: Tensor,
        values: Mapping[str, Tensor],
        input_type: str,
        logical_address: Tensor,
    ) -> tuple[DeviceDestinationRequest, DeviceDestinationRequest]:
        self._assert_aligned()
        if tuple(values) != tuple(field.name for field in self.schema.fields):
            raise RolloutError("family publication fields differ from its schema")
        return (
            DeviceDestinationRequest(
                self.hot,
                counts,
                MappingProxyType(
                    {
                        field.name: values[field.name]
                        for field in self.hot.schema.fields
                    }
                ),
                input_type,
                logical_address,
            ),
            DeviceDestinationRequest(
                self.retained,
                counts,
                MappingProxyType(
                    {
                        field.name: values[field.name]
                        for field in self.retained.schema.fields
                    }
                ),
                input_type,
                logical_address,
            ),
        )


class SemanticFamilyFIFO:
    """FIFO view over one declared ``FamilyWork[input_type]`` slab."""

    def __init__(
        self,
        input_type: str,
        partitions: PartitionContract,
        family: MonotoneSlab,
        retained: MonotoneSlab,
        metadata: MonotoneSlab,
    ) -> None:
        declared = tuple(item.name for item in partitions.inputs)
        if input_type not in declared:
            raise RolloutError(f"undeclared queue family {input_type!r}")
        family_storage = FamilyStorage(family, retained)
        if family_storage.schema.name != f"FamilyWork.{input_type}":
            raise RolloutError("family FIFO is bound to the wrong owner")
        if metadata.schema.name != f"QueueMetadata.{input_type}":
            raise RolloutError("family FIFO metadata is bound to the wrong owner")
        if (
            family_storage.n_replica != metadata.n_replica
            or family_storage.device != metadata.device
        ):
            raise RolloutError("family FIFO owners disagree on replica/device")
        required = {"head", "ready_tail", "reserved_tail"}
        if not required.issubset(metadata.columns):
            raise RolloutError("queue metadata is missing its closed tail fields")
        self.input_type = input_type
        self.partitions = partitions
        self.family = family_storage
        self.metadata = metadata
        self.device = family_storage.device
        self.n_replica = family_storage.n_replica
        self._all_lanes = torch.arange(
            self.n_replica, dtype=torch.int64, device=self.device
        )
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
        transaction = PublicationTransaction.begin_device(
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

    @property
    def frontier_cost(self) -> Tensor:
        if "frontier_cost" not in self.metadata.columns:
            return torch.zeros(
                self.n_replica, dtype=torch.float32, device=self.device
            )
        return self.metadata.columns["frontier_cost"][:, 0]

    def publish_family_tail(self, active: Tensor) -> None:
        """Expose newly committed family rows without reading a host scalar."""

        if (
            active.shape != (self.n_replica,)
            or active.dtype != torch.bool
            or active.device != self.device
        ):
            raise ValueError("queue publication mask must be bool [n_replica]")
        self.family._assert_aligned()
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
        lanes = (
            self._all_lanes
            if scheduled.lanes == tuple(range(self.n_replica))
            else torch.tensor(
                scheduled.lanes, dtype=torch.int64, device=self.device
            )
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
        return FamilyBatch(self.input_type, lanes, row, fields, scheduled.lanes)

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
    lane_schemas: tuple[ModeSchema, ...]
    batch: FamilyBatch
    model: ModelWorkspace
    rng_key: Tensor


@dataclass(frozen=True)
class TransitionRequest:
    lane_schemas: tuple[ModeSchema, ...]
    batch: FamilyBatch
    model: ModelWorkspace
    sample: SampleWorkspace


@dataclass(frozen=True)
class CommitCoverage:
    """Fixed-shape proof connecting sources to every committed row.

    Every valid descriptor names one scheduled source ordinal and one
    destination request/ordinal. A source may have several descendants, but
    every scheduled source must occur and every realized destination row must
    occur exactly once.
    """

    valid: Tensor
    source_ordinal: Tensor
    destination: Tensor
    destination_ordinal: Tensor

    def validate(
        self,
        batch: FamilyBatch,
        requests: Sequence[DeviceDestinationRequest],
        active: Tensor,
        *,
        descriptor_cap: int,
        workspace_byte_cap: int,
    ) -> tuple[Tensor, ...]:
        shape = self.valid.shape
        if (
            self.valid.device != batch.lanes.device
            or self.valid.dtype != torch.bool
            or self.valid.ndim != 2
            or shape[0] != active.numel()
        ):
            raise RolloutError(
                "commit coverage valid must be CUDA bool [n_replica, extent]"
            )
        if shape[1] > descriptor_cap:
            raise RolloutError(
                "commit coverage exceeds its generated descriptor bound"
            )
        temporary_bytes = (
            self.valid.numel() * (1 + 3 * 8)
            + active.numel() * batch.rows * 8
            + sum(
                active.numel() * request.max_rows * 16
                for request in requests
            )
            + active.numel() * len(requests) * 8
        )
        if temporary_bytes > workspace_byte_cap:
            raise RolloutError(
                "commit coverage exceeds admitted rollout workspace bytes"
            )
        for name in (
            "source_ordinal",
            "destination",
            "destination_ordinal",
        ):
            value = getattr(self, name)
            if (
                value.shape != shape
                or value.device != self.valid.device
                or value.dtype != torch.int64
            ):
                raise RolloutError(
                    f"commit coverage {name} must be CUDA int64 {shape}"
                )
        lane_active = torch.zeros_like(active)
        lane_active.index_fill_(0, batch.lanes, True)
        torch._assert_async(
            (~self.valid | lane_active[:, None]).all(),
            "commit coverage contains an inactive lane",
        )
        source_in_range = (
            (self.source_ordinal >= 0)
            & (self.source_ordinal < batch.rows)
        )
        torch._assert_async(
            (~self.valid | source_in_range).all(),
            "commit coverage source ordinal is outside the scheduled prefix",
        )
        source_hit = torch.zeros(
            (active.numel(), batch.rows),
            dtype=torch.int64,
            device=self.valid.device,
        )
        source_hit.scatter_add_(
            1,
            self.source_ordinal.clamp(0, batch.rows - 1),
            (self.valid & source_in_range).to(torch.int64),
        )
        torch._assert_async(
            (
                torch.where(
                    lane_active[:, None],
                    source_hit > 0,
                    source_hit == 0,
                )
            ).all(),
            "commit coverage skipped a scheduled source row",
        )
        destination_declared = torch.zeros_like(self.valid)
        realized_counts: list[Tensor] = []
        for index, request in enumerate(requests):
            selected = self.valid & self.destination.eq(index)
            realized = selected.sum(1, dtype=torch.int64)
            realized_counts.append(realized)
            in_range = (
                (self.destination_ordinal >= 0)
                & (self.destination_ordinal < request.max_rows)
            )
            torch._assert_async(
                (~selected | in_range).all(),
                "commit coverage destination ordinal is outside its pack",
            )
            hit = torch.zeros(
                (active.numel(), request.max_rows),
                dtype=torch.int64,
                device=self.valid.device,
            )
            hit.scatter_add_(
                1,
                self.destination_ordinal.clamp(0, request.max_rows - 1),
                (selected & in_range).to(torch.int64),
            )
            ordinal = torch.arange(
                request.max_rows,
                dtype=torch.int64,
                device=self.valid.device,
            )
            expected = (
                active[:, None] & (ordinal[None, :] < realized[:, None])
            )
            torch._assert_async(
                torch.where(expected, hit == 1, hit == 0).all(),
                "commit coverage must name every destination row exactly once",
            )
            source_at_destination = torch.zeros_like(hit)
            source_at_destination.scatter_add_(
                1,
                self.destination_ordinal.clamp(0, request.max_rows - 1),
                torch.where(
                    selected & in_range,
                    self.source_ordinal + 1,
                    torch.zeros_like(self.source_ordinal),
                ),
            )
            if request.max_rows > 1:
                adjacent = source_at_destination[:, 1:] >= (
                    source_at_destination[:, :-1]
                )
                valid_adjacent = (
                    ordinal[None, 1:] < realized[:, None]
                )
                torch._assert_async(
                    (~valid_adjacent | adjacent).all(),
                    "commit coverage reordered a destination FIFO",
                )
            destination_declared |= selected
        torch._assert_async(
            (~self.valid | destination_declared).all(),
            "commit coverage names an undeclared destination",
        )
        stacked = torch.stack(realized_counts, dim=1)
        device_exclusive_scan(stacked)
        return tuple(realized_counts)


class RolloutOwnerRegistry:
    """Closed retained-owner set and immutable manifest bounds for rollout."""

    def __init__(
        self,
        slabs: Sequence[MonotoneSlab],
        *,
        family_pack_cap: int,
        chance_frontier_cap: int,
        completion_pack_cap: int,
        topology_packet_cap: int,
        topology_cap: int,
        topology_edge_cap: int,
        chance_generation_cap: int,
        descriptor_cap: int,
        workspace_byte_cap: int,
        completion_workspace_byte_cap: int,
        _token: object | None = None,
    ) -> None:
        if _token is not _OWNER_REGISTRY_TOKEN:
            raise ValueError("rollout owners must come from a generated plan")
        self.slabs = tuple(slabs)
        if not self.slabs or len({id(slab) for slab in self.slabs}) != len(
            self.slabs
        ):
            raise ValueError("rollout owner registry must be unique and nonempty")
        self.device = self.slabs[0].device
        self.n_replica = self.slabs[0].n_replica
        self.failure_state = self.slabs[0].failure_state
        if any(
            slab.device != self.device
            or slab.n_replica != self.n_replica
            or slab.failure_state is not self.failure_state
            for slab in self.slabs
        ):
            raise ValueError("rollout owners disagree on device/replica/failure")
        bounds = (
            family_pack_cap,
            chance_frontier_cap,
            completion_pack_cap,
            topology_packet_cap,
            topology_cap,
            topology_edge_cap,
            chance_generation_cap,
            descriptor_cap,
            workspace_byte_cap,
            completion_workspace_byte_cap,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in bounds
        ):
            raise ValueError("rollout registry bounds must be positive integers")
        (
            self.family_pack_cap,
            self.chance_frontier_cap,
            self.completion_pack_cap,
            self.topology_packet_cap,
            self.topology_cap,
            self.topology_edge_cap,
            self.chance_generation_cap,
            self.descriptor_cap,
            self.workspace_byte_cap,
            self.completion_workspace_byte_cap,
        ) = bounds
        self._ids = frozenset(id(slab) for slab in self.slabs)

    @classmethod
    def from_generated(cls, owners: object) -> RolloutOwnerRegistry:
        """Bind to the complete live set and its immutable CapacityManifest."""

        plan = owners.plan
        manifest = plan.manifest
        slabs = owners.live_slabs
        result = cls(
            slabs,
            family_pack_cap=manifest.family_pack_cap,
            chance_frontier_cap=manifest.chance_frontier_cap,
            completion_pack_cap=manifest.completion_pack_cap,
            topology_packet_cap=manifest.topology_packet_cap,
            topology_cap=manifest.topology_cap,
            topology_edge_cap=manifest.topology_edge_cap,
            chance_generation_cap=manifest.geometry.occurrence_bound,
            descriptor_cap=sum(slab.capacity for slab in slabs),
            workspace_byte_cap=plan.admission(
                ExecutionPhase.ROLLOUT
            ).bounded_workspace_bytes,
            completion_workspace_byte_cap=plan.admission(
                ExecutionPhase.COMPLETION
            ).bounded_workspace_bytes,
            _token=_OWNER_REGISTRY_TOKEN,
        )
        result.plan = plan
        return result

    def require(self, slab: MonotoneSlab | FamilyStorage) -> None:
        if isinstance(slab, FamilyStorage):
            if id(slab.hot) not in self._ids or id(slab.retained) not in self._ids:
                raise RolloutError(
                    f"{slab.schema.name} is outside the generated rollout owner set"
                )
            return
        if id(slab) not in self._ids:
            raise RolloutError(
                f"{slab.schema.name} is outside the generated rollout owner set"
            )

    def retry_epoch(self) -> Tensor:
        epochs = torch.stack(
            tuple(
                slab._state[:, MonotoneSlab._EPOCH] for slab in self.slabs
            )
        )
        torch._assert_async(
            (epochs == epochs[:1]).all(),
            "rollout owner retry epochs diverged",
        )
        return epochs[0]

    def guard(
        self,
    ) -> tuple[
        int,
        tuple[tuple[int, tuple[tuple[str, int], ...]], ...],
    ]:
        return self.failure_state._records._version, tuple(
            (
                slab._state._version,
                tuple(
                    (name, value._version)
                    for name, value in slab.columns.items()
                ),
            )
            for slab in self.slabs
        )

    def assert_unchanged(
        self,
        before: tuple[
            int,
            tuple[tuple[int, tuple[tuple[str, int], ...]], ...],
        ],
        phase: str,
    ) -> None:
        failure_version, slab_versions = before
        if self.failure_state._records._version != failure_version:
            raise RolloutError(f"{phase} mutated device failure records")
        for slab, (state_version, column_versions) in zip(
            self.slabs, slab_versions
        ):
            if slab._state._version != state_version or tuple(
                (name, value._version)
                for name, value in slab.columns.items()
            ) != column_versions:
                raise RolloutError(
                    f"{phase} wrote or published retained owner {slab.label}"
                )


@dataclass(frozen=True)
class TopologyPublication:
    """One mode-independent topology packet awaiting commit visibility."""

    lane: int
    sequence: int
    declared_nodes: int
    declared_edges: int
    payload: StagedTopologyPacket

    def __post_init__(self) -> None:
        if (
            self.lane < 0
            or self.sequence < 0
            or self.declared_nodes <= 0
            or self.declared_edges < 0
        ):
            raise TopologyError("topology publication is invalid")


@dataclass
class _PendingTopology:
    publication: TopologyPublication
    success: Tensor
    event: torch.cuda.Event


class AsyncTopologyPublisher:
    """Publish every mode's topology packets after commit visibility."""

    def __init__(
        self,
        owners: RolloutOwnerRegistry,
        *,
        p_train: int,
        input_names: Sequence[str],
        input_to_output: Sequence[int],
    ) -> None:
        if p_train <= 0:
            raise ValueError("topology publication bounds must be positive")
        self.owners = owners
        self.p_train = int(p_train)
        self._stagers = tuple(
            TopologyPacketStager(
                input_names=input_names,
                input_to_output=input_to_output,
                p_train=self.p_train,
                topology_cap=owners.topology_cap,
                topology_edge_cap=owners.topology_edge_cap,
                topology_packet_cap=owners.topology_packet_cap,
                device=owners.device,
            )
            for _lane in range(owners.n_replica)
        )
        self._accept_stream = _runtime_streams(owners.device)[2]
        self._staged: dict[tuple[int, int], TopologyPublication] = {}
        self._pending: list[_PendingTopology] = []
        self._accepted: dict[tuple[int, int], StagedTopologyPacket] = {}
        self._success_buffer = torch.empty(
            owners.n_replica * owners.topology_cap,
            dtype=torch.bool,
            pin_memory=True,
        )

    def stage(self, publication: TopologyPublication) -> None:
        key = (publication.lane, publication.sequence)
        if key in self._staged:
            raise TopologyError("duplicate topology packet sequence")
        if (
            publication.declared_nodes + publication.declared_edges
            > self.owners.topology_packet_cap
        ):
            raise TopologyError("topology packet exceeds its fixed packet cap")
        if publication.sequence >= self.owners.topology_cap:
            raise TopologyError("topology packet exceeds its sequence arena")
        self._staged[key] = publication

    def finalize(
        self,
        publication: TopologyPublication,
        success: Tensor,
        *,
        stream: torch.cuda.Stream,
    ) -> None:
        key = (publication.lane, publication.sequence)
        if self._staged.get(key) is not publication:
            raise TopologyError("topology packet was not staged")
        host_success = self._success_buffer[
            publication.lane * self.owners.topology_cap
            + publication.sequence
        ]
        host_success.copy_(success[publication.lane], non_blocking=True)
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
        self._pending.append(
            _PendingTopology(publication, host_success, event)
        )

    def cancel(
        self,
        publications: Sequence[TopologyPublication],
    ) -> None:
        keys = {(item.lane, item.sequence) for item in publications}
        self._pending = [
            pending
            for pending in self._pending
            if (
                pending.publication.lane,
                pending.publication.sequence,
            )
            not in keys
        ]
        for key in keys:
            self._staged.pop(key, None)
            self._accepted.pop(key, None)

    def _accept_ready(self, *, block: bool = False) -> None:
        ready: list[_PendingTopology] = []
        waiting: list[_PendingTopology] = []
        if block and self._pending:
            with torch.cuda.stream(self._accept_stream):
                for pending in self._pending:
                    self._accept_stream.wait_event(pending.event)
                all_ready = torch.cuda.Event(blocking=False)
                all_ready.record(self._accept_stream)
            all_ready.synchronize()
            ready = self._pending
        else:
            for pending in self._pending:
                (ready if pending.event.query() else waiting).append(pending)
        self._pending = waiting
        for pending in ready:
            publication = pending.publication
            key = (publication.lane, publication.sequence)
            if bool(pending.success):
                self._accepted[key] = publication.payload
            self._staged.pop(key, None)

    def feedback(
        self,
        publication: TopologyPublication,
    ) -> tuple[bool, StagedTopologyPacket | None] | None:
        """Poll one pinned commit result without reading a device scalar."""

        self._accept_ready()
        key = (publication.lane, publication.sequence)
        if key in self._accepted:
            return False, self._accepted.pop(key)
        if key not in self._staged:
            return True, None
        return None

    def wait_feedback(
        self,
        publication: TopologyPublication,
    ) -> tuple[bool, StagedTopologyPacket | None]:
        while True:
            result = self.feedback(publication)
            if result is not None:
                return result
            time.sleep(0)

    @property
    def stagers(self) -> tuple[TopologyPacketStager, ...]:
        return self._stagers

    def accept(
        self,
        publication: TopologyPublication,
    ) -> tuple[bool, tuple[int, ...]]:
        failed, packet = self.wait_feedback(publication)
        if failed or packet is None:
            return True, ()
        stager = self.stagers[publication.lane]
        stager.accept(packet)
        return False, tuple(int(value) for value in packet.produced)

    def seal_lane(self, lane: int) -> tuple[Topology, TopologyBindings, int]:
        return self.stagers[lane].seal()

    def tree_size(self, lane: int, logical_root: int) -> int:
        return self.stagers[lane].tree_size(logical_root)

    def sampled_tree_size(
        self, lane: int, record_limit: int | None = None
    ) -> int:
        return self.stagers[lane].sampled_tree_size(record_limit)

    def root_degree(self, lane: int, logical_root: int) -> int:
        return self.stagers[lane].root_degree(logical_root)

    def record_count(self, lane: int) -> int:
        return self.stagers[lane].record_count

    def counterfactual_correction(
        self, lane: int, record_limit: int
    ) -> float:
        return self.stagers[lane].counterfactual_correction(
            record_limit
        )

    def release_storage(self) -> None:
        for stager in self._stagers:
            stager.release()
        self._stagers = ()

    def assert_drained(self) -> None:
        if self._staged or self._pending or self._accepted:
            raise TopologyError("uncommitted topology packets remain staged")


@dataclass(frozen=True)
class CommitPlan:
    """Every retained destination and structural effect of one transition."""

    destinations: tuple[DeviceDestinationRequest, ...]
    coverage: CommitCoverage
    published_queues: tuple[SemanticFamilyFIFO, ...] = ()
    topology: tuple[TopologyPublication, ...] = ()
    queue_cost_additions: tuple[Tensor, ...] = ()
    device_failure: DeviceCommitFailure | None = None

    def __post_init__(self) -> None:
        if not self.destinations:
            raise ValueError("a commit plan requires retained destinations")
        keys = tuple((item.lane, item.sequence) for item in self.topology)
        if len(set(keys)) != len(keys):
            raise ValueError("a commit plan repeats a topology sequence")
        lanes = tuple(item.lane for item in self.topology)
        if len(set(lanes)) != len(lanes):
            raise ValueError(
                "a family commit may stage only one topology packet per lane"
            )
        queue_ids = tuple(id(queue) for queue in self.published_queues)
        if len(set(queue_ids)) != len(queue_ids):
            raise ValueError("a commit plan repeats a published queue")
        if self.queue_cost_additions and (
            len(self.queue_cost_additions) != len(self.published_queues)
            or any(
                value.dtype != torch.float32 or value.ndim != 1
                for value in self.queue_cost_additions
            )
        ):
            raise ValueError(
                "queue cost additions must align with published queues"
            )


@dataclass(frozen=True)
class DeviceCommitFailure:
    """Device-resident terminal failure that suppresses lane publication."""

    failed: Tensor
    logical_address: Tensor
    requested_extent: Tensor
    available_extent: Tensor
    sampler_attempt_count: Tensor
    phase: str
    failure_code: str
    input_type: str

    def validate(self, owners: RolloutOwnerRegistry) -> None:
        expected = (owners.n_replica,)
        for name, value, dtype in (
            ("failed", self.failed, torch.bool),
            ("logical_address", self.logical_address, torch.int64),
            ("requested_extent", self.requested_extent, torch.int64),
            ("available_extent", self.available_extent, torch.int64),
            ("sampler_attempt_count", self.sampler_attempt_count, torch.int64),
        ):
            if (
                value.shape != expected
                or value.dtype != dtype
                or value.device != owners.device
            ):
                raise RolloutError(
                    f"commit failure {name} must be {dtype} {expected} "
                    f"on {owners.device}"
                )
        if not self.phase or not self.failure_code or not self.input_type:
            raise RolloutError("commit failure labels must be nonempty")


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
    model_started: torch.cuda.Event | None
    model_finished: torch.cuda.Event | None
    sample_finished: torch.cuda.Event | None
    peak_before_model: int
    peak_after_model: int
    peak_after_sample: int
    peak_after_transition: int
    commit_started: torch.cuda.Event | None
    event: torch.cuda.Event
    topology: tuple[TopologyPublication, ...] = ()


class RolloutTransaction:
    """The sole generic ``model -> sample -> commit`` state machine."""

    def __init__(
        self,
        schemas: ModeSchema | Sequence[ModeSchema],
        *,
        seed: int,
        iteration: int,
        topology: AsyncTopologyPublisher,
        owners: RolloutOwnerRegistry,
        model_stream: torch.cuda.Stream | None = None,
        commit_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.topology = topology
        self.owners = owners
        if topology.owners is not owners:
            raise ValueError("rollout and topology must share one owner registry")
        self.device = owners.device
        self.n_replica = owners.n_replica
        self.lane_schemas = (
            (schemas,) * self.n_replica
            if isinstance(schemas, ModeSchema)
            else tuple(schemas)
        )
        if len(self.lane_schemas) != self.n_replica:
            raise ValueError("rollout needs one semantic schema per replica lane")
        self.model_stream = (
            _runtime_streams(self.device)[0]
            if model_stream is None
            else model_stream
        )
        self.commit_stream = (
            _runtime_streams(self.device)[1]
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
        sampler_attempt_count: Tensor | None = None,
        measure_commit: bool = False,
        failure_injector: Callable[[], None] | None = None,
        post_publish_failure_injector: Callable[[], None] | None = None,
    ) -> RolloutCommit:
        if self._state not in (_MachineState.READY, _MachineState.COMMITTED):
            raise RolloutError(
                f"rollout transaction cannot start from {self._state.value}"
            )
        if type(measure_commit) is not bool:
            raise TypeError("measure_commit must be bool")
        self._state = _MachineState.READY
        if scheduled.rows > self.owners.family_pack_cap:
            raise RolloutError("scheduled prefix exceeds manifest family_pack_cap")
        expected_schemas = tuple(
            self.lane_schemas[lane] for lane in scheduled.lanes
        )
        if scheduled.lane_schemas != expected_schemas:
            raise RolloutError(
                "scheduled cohort lost its lane-local semantic mode identity"
            )
        self.owners.require(source.family)
        self.owners.require(source.metadata)
        peak_before_model = (
            int(torch.cuda.max_memory_allocated(self.device))
            if measure_commit
            else 0
        )
        with torch.cuda.stream(self.model_stream):
            batch = source.stable_prefix(scheduled)
            before = self.owners.guard()
            model_input = hooks.model_input(batch)
            model_started = (
                torch.cuda.Event(enable_timing=True, blocking=False)
                if measure_commit
                else None
            )
            model_finished = (
                torch.cuda.Event(enable_timing=True, blocking=False)
                if measure_commit
                else None
            )
            sample_finished = (
                torch.cuda.Event(enable_timing=True, blocking=False)
                if measure_commit
                else None
            )
            if model_started is not None:
                model_started.record(self.model_stream)
            model_workspace = ModelWorkspace(
                MappingProxyType(dict(model(model_input, batch.lanes)))
            )
            if model_finished is not None:
                model_finished.record(self.model_stream)
            peak_after_model = (
                int(torch.cuda.max_memory_allocated(self.device))
                if measure_commit
                else 0
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
                retry_epoch=self.owners.retry_epoch()[batch.lanes, None],
                active=active,
            )
            sampled = hooks.sample(
                SampleRequest(
                    scheduled.lane_schemas,
                    batch,
                    model_workspace,
                    rng_key,
                )
            )
            if sample_finished is not None:
                sample_finished.record(self.model_stream)
            peak_after_sample = (
                int(torch.cuda.max_memory_allocated(self.device))
                if measure_commit
                else 0
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
            self.owners.assert_unchanged(before, "sample")
            self._state = _MachineState.SAMPLED
            plan = hooks.transition(
                TransitionRequest(
                    scheduled.lane_schemas,
                    batch,
                    model_workspace,
                    sampled,
                )
            )
            self.owners.assert_unchanged(before, "transition")
            active_lanes = torch.zeros(
                self.n_replica, dtype=torch.bool, device=self.device
            )
            active_lanes.index_fill_(0, batch.lanes, True)
            declared_inputs = frozenset(
                item.name for item in source.partitions.inputs
            )
            declared_destinations = declared_inputs | frozenset(
                item.name for item in source.partitions.outputs
            )
            for request in plan.destinations:
                self.owners.require(request.slab)
                if request.input_type not in declared_destinations:
                    raise RolloutError(
                        "destination names an undeclared partition"
                    )
                if (
                    request.slab.schema.name.startswith("FamilyWork.")
                    and request.input_type not in declared_inputs
                ):
                    raise RolloutError(
                        "family destination names a non-input partition"
                    )
            for queue in plan.published_queues:
                self.owners.require(queue.family)
                self.owners.require(queue.metadata)
                if (
                    queue.input_type not in declared_inputs
                    or queue.partitions != source.partitions
                ):
                    raise RolloutError(
                        "published queue is outside the source partition contract"
                    )
                matching = tuple(
                    request
                    for request in plan.destinations
                    if request.slab is queue.family
                )
                if len(matching) != 1 or (
                    matching[0].input_type != queue.input_type
                ):
                    raise RolloutError(
                        "published queue lacks its declared family destination"
                    )
            published_family_ids = frozenset(
                id(queue.family) for queue in plan.published_queues
            )
            if any(
                request.slab.schema.name.startswith("FamilyWork.")
                and id(request.slab) not in published_family_ids
                for request in plan.destinations
            ):
                raise RolloutError(
                    "FamilyWork destination lacks queue publication"
                )
            realized_counts = plan.coverage.validate(
                batch,
                plan.destinations,
                active_lanes,
                descriptor_cap=self.owners.descriptor_cap,
                workspace_byte_cap=self.owners.workspace_byte_cap,
            )
            destinations_list: list[DeviceDestinationRequest] = []
            for request, count in zip(plan.destinations, realized_counts):
                if isinstance(request.slab, FamilyStorage):
                    destinations_list.extend(
                        request.slab.destination_requests(
                            count,
                            request.values,
                            request.input_type,
                            request.logical_address,
                        )
                    )
                else:
                    destinations_list.append(
                        DeviceDestinationRequest(
                            slab=request.slab,
                            counts=count,
                            values=request.values,
                            input_type=request.input_type,
                            logical_address=request.logical_address,
                        )
                    )
            destinations = tuple(destinations_list)
            sample_ready = torch.cuda.Event(blocking=False)
            sample_ready.record(self.model_stream)
            device_failure = plan.device_failure
            if device_failure is not None:
                device_failure.validate(self.owners)
            peak_after_transition = (
                int(torch.cuda.max_memory_allocated(self.device))
                if measure_commit
                else 0
            )
        for publication in plan.topology:
            if publication.lane >= self.n_replica:
                raise RolloutError("topology publication lane is outside replicas")
            if publication.lane not in scheduled.lanes:
                raise RolloutError("topology publication lane was not scheduled")
        transaction = None
        published = False
        source_head_before: Tensor | None = None
        queue_before: list[tuple[SemanticFamilyFIFO, Tensor, Tensor]] = []
        frontier_before: list[tuple[SemanticFamilyFIFO, Tensor]] = []
        try:
            with torch.cuda.stream(self.commit_stream):
                self.commit_stream.wait_event(sample_ready)
                commit_started = (
                    torch.cuda.Event(enable_timing=True, blocking=False)
                    if measure_commit
                    else None
                )
                if commit_started is not None:
                    commit_started.record(self.commit_stream)
                for publication in plan.topology:
                    self.topology.stage(publication)
                transaction = PublicationTransaction.begin_device(
                    destinations,
                    phase="rollout_commit",
                    active=active_lanes,
                    sampler_attempt_count=(
                        device_failure.sampler_attempt_count
                        if device_failure is not None
                        else sampler_attempt_count
                    ),
                    blocked=(
                        None
                        if device_failure is None
                        else device_failure.failed
                    ),
                )
                if device_failure is not None:
                    self.owners.failure_state.record_device(
                        failed=device_failure.failed,
                        logical_address=device_failure.logical_address,
                        requested_extent=device_failure.requested_extent,
                        available_extent=device_failure.available_extent,
                        sampler_attempt_count=(
                            device_failure.sampler_attempt_count
                        ),
                        phase=device_failure.phase,
                        failure_code=device_failure.failure_code,
                        input_type=device_failure.input_type,
                    )
                transaction.fill_all()
                if failure_injector is not None:
                    failure_injector()
                success = transaction.success
                source_head_before = source.head.clone()
                for queue in plan.published_queues:
                    queue_before.append(
                        (
                            queue,
                            queue.ready_tail.clone(),
                            queue.reserved_tail.clone(),
                        )
                    )
                    request_index = next(
                        index
                        for index, request in enumerate(destinations)
                        if request.slab is queue.family.hot
                    )
                    end = transaction.ends[request_index].to(torch.int32)
                    queue.ready_tail.copy_(
                        torch.where(success, end, queue.ready_tail)
                    )
                    queue.reserved_tail.copy_(
                        torch.where(success, end, queue.reserved_tail)
                    )
                selected = success[batch.lanes]
                old_source_head = source.head[batch.lanes]
                new_source_head = old_source_head + scheduled.rows
                torch._assert_async(
                    (
                        ~selected
                        | (new_source_head <= source.ready_tail[batch.lanes])
                    ).all(),
                    "queue consumption exceeds its stable prefix",
                )
                source.head[batch.lanes] = torch.where(
                    selected, new_source_head, old_source_head
                )
                success = transaction.publish()
                published = True
                if "frontier_cost_contribution" in batch.fields:
                    frontier_before.append(
                        (source, source.frontier_cost.clone())
                    )
                    removed = batch.fields[
                        "frontier_cost_contribution"
                    ].sum(1)
                    source.frontier_cost.copy_(
                        torch.where(
                            success,
                            (source.frontier_cost - removed).clamp_min(0),
                            source.frontier_cost,
                        )
                    )
                if plan.queue_cost_additions:
                    for queue, addition in zip(
                        plan.published_queues,
                        plan.queue_cost_additions,
                        strict=True,
                    ):
                        if (
                            addition.device != self.device
                            or addition.shape != (self.n_replica,)
                        ):
                            raise RolloutError(
                                "queue cost addition has an invalid device contract"
                            )
                        frontier_before.append(
                            (queue, queue.frontier_cost.clone())
                        )
                        queue.frontier_cost.copy_(
                            torch.where(
                                success,
                                queue.frontier_cost + addition,
                                queue.frontier_cost,
                            )
                        )
                if post_publish_failure_injector is not None:
                    post_publish_failure_injector()
                source._record_ready(self.commit_stream)
                for queue in plan.published_queues:
                    queue._record_ready(self.commit_stream)
                commit_ready = torch.cuda.Event(
                    enable_timing=measure_commit, blocking=False
                )
                commit_ready.record(self.commit_stream)
                for publication in plan.topology:
                    self.topology.finalize(
                        publication,
                        success,
                        stream=self.commit_stream,
                    )
        except BaseException:
            if transaction is not None:
                with torch.cuda.stream(self.commit_stream):
                    if published:
                        transaction.rollback_published()
                    else:
                        transaction.abort()
                    if source_head_before is not None:
                        source.head.copy_(source_head_before)
                    for queue, ready, reserved in queue_before:
                        queue.ready_tail.copy_(ready)
                        queue.reserved_tail.copy_(reserved)
                    for queue, frontier in reversed(frontier_before):
                        queue.frontier_cost.copy_(frontier)
            self.topology.cancel(plan.topology)
            self._state = _MachineState.FAILED
            raise
        sequence = 0 if not plan.topology else plan.topology[-1].sequence
        self._state = _MachineState.COMMITTED
        return RolloutCommit(
            success=success,
            failed=active_lanes & ~success,
            sequence=sequence,
            draw_count=full_draw_count,
            model_started=model_started,
            model_finished=model_finished,
            sample_finished=sample_finished,
            peak_before_model=peak_before_model,
            peak_after_model=peak_after_model,
            peak_after_sample=peak_after_sample,
            peak_after_transition=peak_after_transition,
            commit_started=commit_started,
            event=commit_ready,
            topology=plan.topology,
        )


class BoundedChanceClosure:
    """Own the immutable generation envelope for all public chance closure."""

    def __init__(
        self,
        *,
        max_generations: int,
        generation_bound: int,
    ) -> None:
        if (
            max_generations <= 0
            or generation_bound <= 0
            or max_generations > generation_bound
        ):
            raise ValueError(
                "chance generations must fit the geometry-derived bound"
            )
        self.max_generations = int(max_generations)

    def __iter__(self):
        return iter(range(self.max_generations))


@dataclass(frozen=True)
class CompletionBatch:
    lanes: Tensor
    row: Tensor
    fields: Mapping[str, Tensor]

    @property
    def rows(self) -> int:
        return int(self.row.shape[1])


class CompletionHook(Protocol):
    def __call__(
        self, batch: CompletionBatch
    ) -> CompletionPlan: ...


@dataclass(frozen=True)
class CompletionPlan:
    destinations: tuple[DeviceDestinationRequest, ...]
    coverage: CommitCoverage

    def __post_init__(self) -> None:
        if not self.destinations:
            raise ValueError("completion needs retained destinations")


class CompletionFIFO:
    """Stable, bounded ``Leaf`` prefixes feeding the completion hook."""

    def __init__(
        self,
        leaf: MonotoneSlab,
        owners: RolloutOwnerRegistry,
        *,
        head: Tensor,
        observed_epoch: Tensor,
    ) -> None:
        if leaf.schema.name != "Leaf":
            raise ValueError("completion FIFO requires Leaf")
        owners.require(leaf)
        if owners.completion_pack_cap > leaf.capacity:
            raise ValueError("completion pack exceeds Leaf capacity")
        for name, value in (
            ("head", head),
            ("observed_epoch", observed_epoch),
        ):
            if (
                value.shape != (owners.n_replica,)
                or value.dtype != torch.int64
                or value.device != owners.device
            ):
                raise ValueError(
                    f"completion {name} must be owner workspace int64 [n_replica]"
                )
        if (
            2 * owners.n_replica * 8
            > owners.completion_workspace_byte_cap
        ):
            raise ValueError(
                "completion cursor exceeds admitted workspace bytes"
            )
        self.leaf = leaf
        self.owners = owners
        self.device = leaf.device
        self.n_replica = leaf.n_replica
        self.completion_pack_cap = owners.completion_pack_cap
        self.head = head
        self.observed_epoch = observed_epoch
        self._released = False

    def _check_live(self) -> None:
        if self._released:
            raise RolloutError("completion FIFO workspace has been released")

    def _observe_retry_epoch(self) -> None:
        self._check_live()
        epoch = self.owners.retry_epoch()
        changed = self.observed_epoch != epoch
        self.head.copy_(torch.where(changed, torch.zeros_like(self.head), self.head))
        self.observed_epoch.copy_(epoch)

    @property
    def ready_tail(self) -> Tensor:
        self._check_live()
        return self.leaf._state[:, MonotoneSlab._PUBLISHED]

    @property
    def depth(self) -> Tensor:
        return self.ready_tail - self.head

    def stable_prefix(
        self, lanes: Sequence[int], rows: int
    ) -> CompletionBatch:
        self._observe_retry_epoch()
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
        wait_for: torch.cuda.Event,
        failure_injector: Callable[[], None] | None = None,
    ) -> Tensor:
        torch.cuda.current_stream(self.device).wait_event(wait_for)
        batch = self.stable_prefix(lanes, rows)
        before = self.owners.guard()
        workspace_versions = (self.head._version, self.observed_epoch._version)
        plan = hook(batch)
        self.owners.assert_unchanged(before, "completion hook")
        if workspace_versions != (
            self.head._version,
            self.observed_epoch._version,
        ):
            raise RolloutError("completion hook mutated FIFO workspace")
        active = torch.zeros(
            self.n_replica, dtype=torch.bool, device=self.device
        )
        active[batch.lanes] = True
        for request in plan.destinations:
            self.owners.require(request.slab)
            if request.slab.schema.name != f"LayerWork.{request.input_type}":
                raise RolloutError(
                    "completion destinations must be matching LayerWork partitions"
                )
        realized_counts = plan.coverage.validate(
            batch,
            plan.destinations,
            active,
            descriptor_cap=self.owners.descriptor_cap,
            workspace_byte_cap=self.owners.completion_workspace_byte_cap,
        )
        destinations = tuple(
            DeviceDestinationRequest(
                slab=request.slab,
                counts=count,
                values=request.values,
                input_type=request.input_type,
                logical_address=request.logical_address,
            )
            for request, count in zip(plan.destinations, realized_counts)
            if request.slab.capacity > 0
        )
        if not destinations:
            raise RolloutError("completion batch has no realized LayerWork output")
        transaction = PublicationTransaction.begin_device(
            destinations,
            phase="completion",
            active=active,
            sampler_attempt_count=sampler_attempt_count,
        )
        try:
            transaction.fill_all()
            if failure_injector is not None:
                failure_injector()
            success = transaction.publish()
            old = self.head[batch.lanes]
            self.head[batch.lanes] = torch.where(
                success[batch.lanes], old + rows, old
            )
        except BaseException:
            transaction.abort()
            raise
        return success

    def release(self) -> None:
        """Drop the bounded completion-workspace views exactly once."""

        self._check_live()
        self.head = torch.empty(0, dtype=torch.int64, device=self.device)
        self.observed_epoch = torch.empty(
            0, dtype=torch.int64, device=self.device
        )
        self._released = True


__all__ = [
    "LOGICAL_RNG_VERSION",
    "AsyncTopologyPublisher",
    "BoundedChanceClosure",
    "CommitCoverage",
    "CommitPlan",
    "CompletionBatch",
    "CompletionFIFO",
    "CompletionPlan",
    "DeviceCommitFailure",
    "DeviceStableRoute",
    "DrawPurpose",
    "FamilyBatch",
    "FamilyQueueState",
    "FamilyScheduler",
    "FamilyStorage",
    "ModelWorkspace",
    "RolloutCommit",
    "RolloutError",
    "RolloutHooks",
    "RolloutOwnerRegistry",
    "RolloutTransaction",
    "SampleRequest",
    "SampleWorkspace",
    "ScheduledFamily",
    "SemanticFamilyFIFO",
    "TopologyPublication",
    "TransitionRequest",
    "child_path_keys",
    "device_exclusive_scan",
    "logical_categorical",
    "logical_rng_keys",
    "logical_uniform",
    "stable_route",
]
