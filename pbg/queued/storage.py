"""Fixed-capacity storage and lifetime ownership for the queued engine.

The public allocation mechanisms in this module are deliberately small:

``SoASchema``
    The immutable physical description and byte authority for one owner.
``MonotoneSlab``
    One fixed CUDA allocation with lane-local monotone tails.
``ExactBlock``
    One fixed CUDA allocation whose realized row extent is immutable.

There is no growth, paging, recycling, wrapping, schema inference, or legacy
storage conversion in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor


class StorageError(RuntimeError):
    """Base error for queued storage contract violations."""


class StorageOverflow(MemoryError):
    """A fixed-capacity reservation failed before any tail advanced."""

    def __init__(self, failure: "DeviceFailure") -> None:
        self.failure = failure
        super().__init__(
            f"{failure.phase} overflow at logical address "
            f"{failure.logical_address}: requested={failure.requested_extent}, "
            f"available={failure.available_extent}"
        )


class TransactionError(StorageError):
    """A multi-destination transaction could not publish."""


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dtype_size(dtype: torch.dtype) -> int:
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"unsupported storage dtype {dtype}") from exc


@dataclass(frozen=True, order=True)
class SoAField:
    """One contiguous field in an aligned structure-of-arrays allocation."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...] = ()
    alignment: int = 64
    fill: int | float | bool = 0

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid field name {self.name!r}")
        shape = tuple(self.shape)
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in shape
        ):
            raise ValueError(f"{self.name} has an invalid trailing shape")
        object.__setattr__(self, "shape", shape)
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError(f"{self.name} alignment must be a positive power of two")
        element_size = _dtype_size(self.dtype)
        if self.alignment < element_size or self.alignment % element_size:
            raise ValueError(
                f"{self.name} alignment must be a multiple of its dtype size"
            )

    @property
    def elements_per_row(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def row_bytes(self) -> int:
        return self.elements_per_row * _dtype_size(self.dtype)


@dataclass(frozen=True, order=True)
class FieldExtent:
    """One field's exact byte extent within a backing allocation."""

    name: str
    offset_bytes: int
    size_bytes: int


@dataclass(frozen=True)
class SoASchema:
    """Immutable fields, dtype, shape, alignment, and lifetime authority."""

    name: str
    fields: tuple[SoAField, ...]
    lifetime: str
    alignment: int = 256

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("schema name must be nonempty")
        fields = tuple(self.fields)
        if not fields or len({field.name for field in fields}) != len(fields):
            raise ValueError("schema fields must be nonempty and uniquely named")
        object.__setattr__(self, "fields", fields)
        if not self.lifetime:
            raise ValueError("schema lifetime must be nonempty")
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("schema alignment must be a positive power of two")
        if any(field.alignment > self.alignment for field in fields):
            raise ValueError("field alignment exceeds schema alignment")

    @property
    def by_name(self) -> Mapping[str, SoAField]:
        return MappingProxyType({field.name: field for field in self.fields})

    def field_extents(self, rows: int, n_replica: int = 1) -> tuple[FieldExtent, ...]:
        rows = _nonnegative_integer(rows, "rows")
        n_replica = _positive_integer(n_replica, "n_replica")
        cursor = 0
        extents: list[FieldExtent] = []
        for field in self.fields:
            cursor = _align(cursor, field.alignment)
            size = n_replica * rows * field.row_bytes
            extents.append(FieldExtent(field.name, cursor, size))
            cursor += size
        return tuple(extents)

    def allocation_bytes(self, rows: int, n_replica: int = 1) -> int:
        extents = self.field_extents(rows, n_replica)
        if not extents:
            return 0
        final = extents[-1].offset_bytes + extents[-1].size_bytes
        return _align(final, self.alignment) if final else 0

    def payload_bytes(self, rows: int, n_replica: int = 1) -> int:
        rows = _nonnegative_integer(rows, "rows")
        n_replica = _positive_integer(n_replica, "n_replica")
        return n_replica * rows * sum(field.row_bytes for field in self.fields)

    def allocate(
        self,
        rows: int,
        n_replica: int,
        device: torch.device | str,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        """Allocate one backing arena and typed contiguous field views."""

        rows = _nonnegative_integer(rows, "rows")
        n_replica = _positive_integer(n_replica, "n_replica")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise StorageError("queued rollout storage requires CUDA")
        if not torch.cuda.is_available():
            raise StorageError("CUDA is unavailable")
        total = self.allocation_bytes(rows, n_replica)
        backing = torch.empty(total, dtype=torch.uint8, device=resolved)
        if total and backing.data_ptr() % self.alignment:
            raise StorageError(
                f"allocator did not satisfy {self.alignment}-byte schema alignment"
            )
        columns: dict[str, Tensor] = {}
        extents = {extent.name: extent for extent in self.field_extents(rows, n_replica)}
        for field in self.fields:
            extent = extents[field.name]
            if extent.size_bytes == 0:
                typed = torch.empty(
                    (n_replica, rows, *field.shape),
                    dtype=field.dtype,
                    device=resolved,
                )
            else:
                byte_view = backing.narrow(
                    0, extent.offset_bytes, extent.size_bytes
                )
                typed = byte_view.view(field.dtype).view(
                    n_replica, rows, *field.shape
                )
                if typed.data_ptr() % field.alignment:
                    raise StorageError(
                        f"{self.name}.{field.name} is not aligned to "
                        f"{field.alignment} bytes"
                    )
            typed.fill_(field.fill)
            columns[field.name] = typed
        return backing, MappingProxyType(columns)

    def validate_columns(
        self,
        columns: Mapping[str, Tensor],
        rows: int,
        n_replica: int,
        device: torch.device | str,
    ) -> None:
        if tuple(columns) != tuple(field.name for field in self.fields):
            raise StorageError(f"{self.name} columns do not match its schema")
        resolved = torch.device(device)
        for field in self.fields:
            value = columns[field.name]
            expected = (n_replica, rows, *field.shape)
            if (
                tuple(value.shape) != expected
                or value.dtype != field.dtype
                or value.device != resolved
                or not value.is_contiguous()
            ):
                raise StorageError(
                    f"{self.name}.{field.name} must be contiguous {expected} "
                    f"{field.dtype} on {resolved}"
                )


@dataclass(frozen=True, order=True)
class DeviceFailure:
    """One lane-local failure selected by minimum logical address."""

    lane: int
    status: int
    phase: str
    failure_code: str
    input_type: str
    logical_address: int
    requested_extent: int
    available_extent: int
    sampler_attempt_count: int = 0

    def __post_init__(self) -> None:
        _nonnegative_integer(self.lane, "failure lane")
        _positive_integer(self.status, "failure status")
        if not self.phase or not self.failure_code:
            raise ValueError("failure phase and code must be nonempty")
        _nonnegative_integer(self.logical_address, "logical address")
        _nonnegative_integer(self.requested_extent, "requested extent")
        _nonnegative_integer(self.available_extent, "available extent")
        _nonnegative_integer(self.sampler_attempt_count, "sampler attempt count")


class DeviceFailureState:
    """One fixed CUDA failure record per replica lane.

    The tensor is authoritative. Python dictionaries only decode diagnostic
    string IDs when a quiescent snapshot is requested.
    """

    _WIDTH = 8
    _STATUS = 0
    _PHASE = 1
    _CODE = 2
    _INPUT = 3
    _ADDRESS = 4
    _REQUESTED = 5
    _AVAILABLE = 6
    _ATTEMPTS = 7
    _IMMUTABLE_ATTRIBUTES = frozenset(("n_replica", "device"))

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._IMMUTABLE_ATTRIBUTES and hasattr(self, name):
            raise AttributeError(f"{name} is immutable")
        super().__setattr__(name, value)

    def __init__(self, n_replica: int, device: torch.device | str) -> None:
        self.n_replica = _positive_integer(n_replica, "n_replica")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise StorageError("device failures require CUDA")
        self._records = torch.zeros(
            (self.n_replica, self._WIDTH), dtype=torch.int64, device=self.device
        )
        self._records[:, self._ADDRESS].fill_(torch.iinfo(torch.int64).max)
        self._phase_ids: dict[str, int] = {}
        self._code_ids: dict[str, int] = {}
        self._input_ids: dict[str, int] = {}
        self._released = False

    @property
    def allocated_bytes(self) -> int:
        self._check_live()
        return int(self._records.untyped_storage().nbytes())

    @staticmethod
    def analytical_bytes(n_replica: int) -> int:
        return _positive_integer(n_replica, "n_replica") * DeviceFailureState._WIDTH * 8

    @staticmethod
    def _id(table: dict[str, int], value: str) -> int:
        if not value:
            raise ValueError("failure labels must be nonempty")
        if value not in table:
            table[value] = len(table) + 1
        return table[value]

    @staticmethod
    def _decode(table: Mapping[str, int], value: int) -> str:
        for label, identifier in table.items():
            if identifier == value:
                return label
        raise StorageError(f"unknown device failure label id {value}")

    def _check_live(self) -> None:
        if self._released:
            raise StorageError("device failure state has been released")

    def record(self, failure: DeviceFailure) -> None:
        self.record_many((failure,))

    def record_many(self, failures: Sequence[DeviceFailure]) -> None:
        """Merge failures deterministically, retaining each lane's minimum."""

        self._check_live()
        selected: dict[int, DeviceFailure] = {}
        for failure in failures:
            if failure.lane >= self.n_replica:
                raise ValueError("failure lane is outside the replica axis")
            current = selected.get(failure.lane)
            key = (
                failure.logical_address,
                failure.phase,
                failure.failure_code,
                failure.input_type,
            )
            if current is None or key < (
                current.logical_address,
                current.phase,
                current.failure_code,
                current.input_type,
            ):
                selected[failure.lane] = failure
        for lane, failure in selected.items():
            current_address = int(self._records[lane, self._ADDRESS].item())
            if failure.logical_address > current_address:
                continue
            if failure.logical_address == current_address:
                current = self.snapshot(lane)
                if current is not None and (
                    failure.phase,
                    failure.failure_code,
                    failure.input_type,
                ) >= (current.phase, current.failure_code, current.input_type):
                    continue
            row = self._records[lane]
            row[self._STATUS].fill_(failure.status)
            row[self._PHASE].fill_(self._id(self._phase_ids, failure.phase))
            row[self._CODE].fill_(self._id(self._code_ids, failure.failure_code))
            row[self._INPUT].fill_(self._id(self._input_ids, failure.input_type))
            row[self._ADDRESS].fill_(failure.logical_address)
            row[self._REQUESTED].fill_(failure.requested_extent)
            row[self._AVAILABLE].fill_(failure.available_extent)
            row[self._ATTEMPTS].fill_(failure.sampler_attempt_count)

    def snapshot(self, lane: int) -> DeviceFailure | None:
        self._check_live()
        lane = _nonnegative_integer(lane, "lane")
        if lane >= self.n_replica:
            raise ValueError("lane is outside the replica axis")
        row = self._records[lane].cpu().tolist()
        if not row[self._STATUS]:
            return None
        return DeviceFailure(
            lane=lane,
            status=int(row[self._STATUS]),
            phase=self._decode(self._phase_ids, int(row[self._PHASE])),
            failure_code=self._decode(self._code_ids, int(row[self._CODE])),
            input_type=self._decode(self._input_ids, int(row[self._INPUT])),
            logical_address=int(row[self._ADDRESS]),
            requested_extent=int(row[self._REQUESTED]),
            available_extent=int(row[self._AVAILABLE]),
            sampler_attempt_count=int(row[self._ATTEMPTS]),
        )

    def clear_lane(self, lane: int) -> None:
        self._check_live()
        lane = _nonnegative_integer(lane, "lane")
        if lane >= self.n_replica:
            raise ValueError("lane is outside the replica axis")
        self._records[lane].zero_()
        self._records[lane, self._ADDRESS].fill_(torch.iinfo(torch.int64).max)

    def release(self) -> None:
        self._check_live()
        self._records = torch.empty(0, dtype=torch.int64, device=self.device)
        self._released = True


@dataclass(frozen=True, order=True)
class ReservationAddress:
    """A row span whose epoch makes retry addresses disjoint."""

    epoch: int
    begin: int
    end: int


class SlabReservation:
    """One unpublished fixed span in a :class:`MonotoneSlab`."""

    def __init__(
        self,
        slab: "MonotoneSlab",
        lane: int,
        address: ReservationAddress,
        token: int,
    ) -> None:
        self._slab = slab
        self.lane = lane
        self.address = address
        self._token = token
        self._filled = False
        self._closed = False

    @property
    def rows(self) -> int:
        return self.address.end - self.address.begin

    @property
    def filled(self) -> bool:
        return self._filled

    def fill(self, values: Mapping[str, Tensor]) -> None:
        if self._closed:
            raise StorageError("reservation is closed")
        self._slab._fill(self, values)
        self._filled = True

    def view(self, field: str) -> Tensor:
        if self._closed:
            raise StorageError("reservation is closed")
        return self._slab._reservation_view(self, field)


class MonotoneSlab:
    """Fixed-capacity ``reserve -> fill -> publish`` CUDA storage.

    Tails and failures are lane-local. A failed transaction poisons only its
    lane until the complete attempt is discarded with :meth:`reset_failed_lane`.
    Published row addresses never move, wrap, or repeat within an attempt; an
    incremented epoch makes retry addresses disjoint.
    """

    _PUBLISHED = 0
    _RESERVED = 1
    _ATTEMPT_HIGH_WATER = 2
    _LIFETIME_HIGH_WATER = 3
    _EPOCH = 4
    _OPEN_BEGIN = 5
    _OPEN_END = 6
    _POISONED = 7
    _STATE_WIDTH = 8
    _IMMUTABLE_ATTRIBUTES = frozenset(
        ("schema", "capacity", "n_replica", "device", "failure_state", "label")
    )

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._IMMUTABLE_ATTRIBUTES and hasattr(self, name):
            raise AttributeError(f"{name} is immutable")
        super().__setattr__(name, value)

    def __init__(
        self,
        schema: SoASchema,
        capacity: int,
        n_replica: int,
        device: torch.device | str,
        *,
        failure_state: DeviceFailureState,
        label: str | None = None,
    ) -> None:
        self.schema = schema
        self.capacity = _positive_integer(capacity, "slab capacity")
        self.n_replica = _positive_integer(n_replica, "n_replica")
        self.device = torch.device(device)
        if failure_state.n_replica != self.n_replica:
            raise ValueError("slab and failure state replica axes differ")
        if failure_state.device != self.device:
            raise ValueError("slab and failure state devices differ")
        self.failure_state = failure_state
        self.label = schema.name if label is None else str(label)
        if not self.label:
            raise ValueError("slab label must be nonempty")
        self._backing, self._columns = schema.allocate(
            self.capacity, self.n_replica, self.device
        )
        self._state = torch.zeros(
            (self.n_replica, self._STATE_WIDTH),
            dtype=torch.int64,
            device=self.device,
        )
        self._state[:, self._OPEN_BEGIN : self._OPEN_END + 1].fill_(-1)
        self._next_token = 1
        self._tokens: dict[int, SlabReservation] = {}
        self._released = False

    @staticmethod
    def state_bytes(n_replica: int) -> int:
        return _positive_integer(n_replica, "n_replica") * MonotoneSlab._STATE_WIDTH * 8

    @classmethod
    def analytical_bytes(
        cls, schema: SoASchema, capacity: int, n_replica: int
    ) -> int:
        return schema.allocation_bytes(capacity, n_replica) + cls.state_bytes(n_replica)

    @property
    def allocated_bytes(self) -> int:
        self._check_live()
        return int(self._backing.untyped_storage().nbytes()) + int(
            self._state.untyped_storage().nbytes()
        )

    @property
    def columns(self) -> Mapping[str, Tensor]:
        self._check_live()
        return self._columns

    def _check_live(self) -> None:
        if self._released:
            raise StorageError(f"{self.label} has been released")

    def _lane_state(self, lane: int) -> list[int]:
        self._check_live()
        lane = _nonnegative_integer(lane, "lane")
        if lane >= self.n_replica:
            raise ValueError("lane is outside the replica axis")
        return [int(value) for value in self._state[lane].cpu().tolist()]

    def tail(self, lane: int) -> int:
        return self._lane_state(lane)[self._PUBLISHED]

    def reserved_tail(self, lane: int) -> int:
        return self._lane_state(lane)[self._RESERVED]

    def high_water(self, lane: int, *, lifetime: bool = False) -> int:
        index = self._LIFETIME_HIGH_WATER if lifetime else self._ATTEMPT_HIGH_WATER
        return self._lane_state(lane)[index]

    def epoch(self, lane: int) -> int:
        return self._lane_state(lane)[self._EPOCH]

    def preflight(self, lane: int, count: int) -> tuple[int, int]:
        count = _positive_integer(count, "reservation count")
        state = self._lane_state(lane)
        if state[self._POISONED]:
            raise StorageError(f"{self.label} lane {lane} is poisoned")
        if state[self._OPEN_BEGIN] != -1:
            raise StorageError(f"{self.label} lane {lane} already has a reservation")
        begin = state[self._RESERVED]
        return begin, self.capacity - begin

    def reserve(
        self,
        lane: int,
        count: int,
        *,
        phase: str,
        input_type: str,
        logical_address: int,
        sampler_attempt_count: int = 0,
    ) -> SlabReservation:
        count = _positive_integer(count, "reservation count")
        logical_address = _nonnegative_integer(
            logical_address, "logical address"
        )
        sampler_attempt_count = _nonnegative_integer(
            sampler_attempt_count, "sampler attempt count"
        )
        if not phase or not input_type:
            raise ValueError("reservation phase and input type must be nonempty")
        begin, available = self.preflight(lane, count)
        if count > available:
            failure = DeviceFailure(
                lane=lane,
                status=1,
                phase=phase,
                failure_code="capacity_overflow",
                input_type=input_type,
                logical_address=logical_address,
                requested_extent=count,
                available_extent=available,
                sampler_attempt_count=sampler_attempt_count,
            )
            self.failure_state.record(failure)
            raise StorageOverflow(self.failure_state.snapshot(lane) or failure)
        end = begin + count
        state = self._state[lane]
        state[self._RESERVED].fill_(end)
        state[self._ATTEMPT_HIGH_WATER].fill_(
            max(int(state[self._ATTEMPT_HIGH_WATER].item()), end)
        )
        state[self._LIFETIME_HIGH_WATER].fill_(
            max(int(state[self._LIFETIME_HIGH_WATER].item()), end)
        )
        state[self._OPEN_BEGIN].fill_(begin)
        state[self._OPEN_END].fill_(end)
        token = self._next_token
        self._next_token += 1
        reservation = SlabReservation(
            self,
            lane,
            ReservationAddress(self.epoch(lane), begin, end),
            token,
        )
        self._tokens[token] = reservation
        return reservation

    def _require_reservation(self, reservation: SlabReservation) -> None:
        self._check_live()
        if (
            reservation._slab is not self
            or reservation._closed
            or self._tokens.get(reservation._token) is not reservation
        ):
            raise StorageError("reservation does not belong to this live slab")
        state = self._lane_state(reservation.lane)
        if (
            state[self._OPEN_BEGIN] != reservation.address.begin
            or state[self._OPEN_END] != reservation.address.end
            or state[self._EPOCH] != reservation.address.epoch
        ):
            raise StorageError("reservation address no longer matches slab state")

    def _reservation_view(self, reservation: SlabReservation, field: str) -> Tensor:
        self._require_reservation(reservation)
        if field not in self._columns:
            raise KeyError(f"{self.schema.name} has no field {field!r}")
        return self._columns[field][
            reservation.lane, reservation.address.begin : reservation.address.end
        ]

    def _fill(self, reservation: SlabReservation, values: Mapping[str, Tensor]) -> None:
        self._require_reservation(reservation)
        expected_names = tuple(field.name for field in self.schema.fields)
        if tuple(values) != expected_names:
            raise StorageError(
                f"{self.label} fill fields must exactly match {expected_names}"
            )
        for field in self.schema.fields:
            value = values[field.name]
            expected = (reservation.rows, *field.shape)
            if (
                tuple(value.shape) != expected
                or value.dtype != field.dtype
                or value.device != self.device
            ):
                raise StorageError(
                    f"{self.label}.{field.name} fill must be {expected} "
                    f"{field.dtype} on {self.device}"
                )
        for field in self.schema.fields:
            self._reservation_view(reservation, field.name).copy_(values[field.name])

    def _prepare_publish(self, reservation: SlabReservation) -> None:
        self._require_reservation(reservation)
        if not reservation.filled:
            raise TransactionError(f"{self.label} reservation is not filled")
        state = self._lane_state(reservation.lane)
        if state[self._PUBLISHED] != reservation.address.begin:
            raise TransactionError("publication would create a row-address gap")

    def _publish(self, reservation: SlabReservation) -> None:
        self._state[reservation.lane, self._PUBLISHED].fill_(
            reservation.address.end
        )
        self._state[
            reservation.lane, self._OPEN_BEGIN : self._OPEN_END + 1
        ].fill_(-1)
        reservation._closed = True
        del self._tokens[reservation._token]

    def _rollback_publication(self, reservation: SlabReservation) -> None:
        self._state[reservation.lane, self._PUBLISHED].fill_(
            reservation.address.begin
        )
        self._state[reservation.lane, self._POISONED].fill_(1)

    def _abort(self, reservation: SlabReservation) -> None:
        if reservation._closed:
            return
        self._require_reservation(reservation)
        self._state[reservation.lane, self._POISONED].fill_(1)
        self._state[
            reservation.lane, self._OPEN_BEGIN : self._OPEN_END + 1
        ].fill_(-1)
        reservation._closed = True
        del self._tokens[reservation._token]

    def published(self, lane: int, field: str) -> Tensor:
        tail = self.tail(lane)
        if field not in self._columns:
            raise KeyError(f"{self.schema.name} has no field {field!r}")
        return self._columns[field][lane, :tail]

    def _prepare_reset_failed_lane(self, lane: int) -> list[int]:
        state = self._lane_state(lane)
        if not state[self._POISONED] and self.failure_state.snapshot(lane) is None:
            raise StorageError("only a failed lane may start a retry epoch")
        open_tokens = [
            token
            for token, reservation in self._tokens.items()
            if reservation.lane == lane
        ]
        if open_tokens:
            raise StorageError("failed lane still owns an open reservation")
        return state

    def reset_failed_lane(self, lane: int) -> int:
        """Discard one complete failed attempt without touching other lanes."""

        state = self._prepare_reset_failed_lane(lane)
        for field in self.schema.fields:
            self._columns[field.name][lane].fill_(field.fill)
        lifetime_high_water = state[self._LIFETIME_HIGH_WATER]
        new_epoch = state[self._EPOCH] + 1
        self._state[lane].zero_()
        self._state[lane, self._LIFETIME_HIGH_WATER].fill_(lifetime_high_water)
        self._state[lane, self._EPOCH].fill_(new_epoch)
        self._state[lane, self._OPEN_BEGIN : self._OPEN_END + 1].fill_(-1)
        return new_epoch

    def release(self) -> None:
        self._check_live()
        if self._tokens:
            raise StorageError(f"{self.label} has open reservations")
        self._columns = MappingProxyType({})
        self._backing = torch.empty(0, dtype=torch.uint8, device=self.device)
        self._state = torch.empty(0, dtype=torch.int64, device=self.device)
        self._released = True


@dataclass(frozen=True)
class DestinationRequest:
    """One destination extent in an atomic publication transaction."""

    slab: MonotoneSlab
    lane: int
    count: int
    input_type: str
    logical_address: int


class PublicationTransaction:
    """Atomic multi-destination reservation, fill, and publication."""

    def __init__(self, reservations: tuple[SlabReservation, ...]) -> None:
        if not reservations:
            raise ValueError("a publication transaction requires destinations")
        lanes = {reservation.lane for reservation in reservations}
        failures = {
            id(reservation._slab.failure_state) for reservation in reservations
        }
        if len(lanes) != 1 or len(failures) != 1:
            raise ValueError(
                "transaction destinations must share one lane and failure state"
            )
        identities = tuple(
            (id(reservation._slab), reservation.lane)
            for reservation in reservations
        )
        if len(set(identities)) != len(identities):
            raise ValueError("a transaction may publish a slab lane only once")
        self.reservations = reservations
        self._closed = False

    @classmethod
    def begin(
        cls,
        requests: Sequence[DestinationRequest],
        *,
        phase: str,
        sampler_attempt_count: int = 0,
    ) -> "PublicationTransaction":
        if not requests:
            raise ValueError("a publication transaction requires destinations")
        if not phase:
            raise ValueError("transaction phase must be nonempty")
        sampler_attempt_count = _nonnegative_integer(
            sampler_attempt_count, "sampler attempt count"
        )
        identities = tuple((id(item.slab), item.lane) for item in requests)
        if len(set(identities)) != len(identities):
            raise ValueError("a transaction may reserve a slab lane only once")
        if len({item.lane for item in requests}) != 1 or len(
            {id(item.slab.failure_state) for item in requests}
        ) != 1:
            raise ValueError(
                "transaction destinations must share one lane and failure state"
            )

        failures: list[DeviceFailure] = []
        for request in requests:
            count = _positive_integer(request.count, "reservation count")
            logical_address = _nonnegative_integer(
                request.logical_address, "logical address"
            )
            if not request.input_type:
                raise ValueError("destination input type must be nonempty")
            begin, available = request.slab.preflight(request.lane, count)
            del begin
            if count > available:
                failures.append(
                    DeviceFailure(
                        lane=request.lane,
                        status=1,
                        phase=phase,
                        failure_code="capacity_overflow",
                        input_type=request.input_type,
                        logical_address=logical_address,
                        requested_extent=count,
                        available_extent=available,
                        sampler_attempt_count=sampler_attempt_count,
                    )
                )
        if failures:
            registry = requests[0].slab.failure_state
            registry.record_many(failures)
            minimum = min(
                failures,
                key=lambda item: (
                    item.logical_address,
                    item.lane,
                    item.input_type,
                ),
            )
            recorded = registry.snapshot(minimum.lane)
            raise StorageOverflow(recorded or minimum)

        reservations: list[SlabReservation] = []
        try:
            for request in requests:
                reservations.append(
                    request.slab.reserve(
                        request.lane,
                        request.count,
                        phase=phase,
                        input_type=request.input_type,
                        logical_address=request.logical_address,
                        sampler_attempt_count=sampler_attempt_count,
                    )
                )
        except BaseException:
            for reservation in reservations:
                reservation._slab._abort(reservation)
            raise
        return cls(tuple(reservations))

    def fill(self, index: int, values: Mapping[str, Tensor]) -> None:
        if self._closed:
            raise TransactionError("transaction is closed")
        try:
            self.reservations[index].fill(values)
        except BaseException:
            self.abort()
            raise

    def publish(
        self,
        *,
        failure_injector: Callable[[], None] | None = None,
    ) -> None:
        if self._closed:
            raise TransactionError("transaction is closed")
        try:
            for reservation in self.reservations:
                reservation._slab._prepare_publish(reservation)
            if failure_injector is not None:
                failure_injector()
        except BaseException:
            self.abort()
            raise

        published: list[SlabReservation] = []
        try:
            for reservation in self.reservations:
                reservation._slab._publish(reservation)
                published.append(reservation)
        except BaseException as exc:
            for reservation in published:
                reservation._slab._rollback_publication(reservation)
            for reservation in self.reservations:
                if not reservation._closed:
                    reservation._slab._abort(reservation)
            self._closed = True
            raise TransactionError("atomic publication failed") from exc
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        for reservation in self.reservations:
            if not reservation._closed:
                reservation._slab._abort(reservation)
        self._closed = True


class ExactBlock:
    """One explicitly released CUDA block with an immutable realized extent."""

    _IMMUTABLE_ATTRIBUTES = frozenset(("schema", "rows", "device"))

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._IMMUTABLE_ATTRIBUTES and hasattr(self, name):
            raise AttributeError(f"{name} is immutable")
        super().__setattr__(name, value)

    def __init__(
        self,
        schema: SoASchema,
        rows: int,
        device: torch.device | str,
    ) -> None:
        self.schema = schema
        self.rows = _nonnegative_integer(rows, "exact rows")
        self.device = torch.device(device)
        backing, columns = schema.allocate(self.rows, 1, self.device)
        self._backing = backing
        self._columns = MappingProxyType(
            {name: value[0] for name, value in columns.items()}
        )
        self._released = False

    @property
    def columns(self) -> Mapping[str, Tensor]:
        self._check_live()
        return self._columns

    @property
    def allocated_bytes(self) -> int:
        self._check_live()
        return int(self._backing.untyped_storage().nbytes())

    @property
    def payload_bytes(self) -> int:
        return self.schema.payload_bytes(self.rows)

    def _check_live(self) -> None:
        if self._released:
            raise StorageError(f"{self.schema.name} exact block has been released")

    def release(self) -> None:
        self._check_live()
        self._columns = MappingProxyType({})
        self._backing = torch.empty(0, dtype=torch.uint8, device=self.device)
        self._released = True


__all__ = [
    "DestinationRequest",
    "DeviceFailure",
    "DeviceFailureState",
    "ExactBlock",
    "FieldExtent",
    "MonotoneSlab",
    "PublicationTransaction",
    "ReservationAddress",
    "SlabReservation",
    "SoAField",
    "SoASchema",
    "StorageError",
    "StorageOverflow",
    "TransactionError",
]
