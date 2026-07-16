"""Generic device-local primitives for family-scheduled rollout execution.

This module contains no game-specific routing. Logical work queues, occurrence
arenas, admission accounting, stable routing, and schedule-independent random
addresses are shared by all three production training modes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable, Mapping

import torch
from torch import Tensor


NEW = 0
QUEUED = 1
EVALUATING = 2
DONE = 3


def _device_assert(condition: Tensor, message: str) -> None:
    """Assert a scalar tensor without a CUDA device-to-host transfer."""

    if condition.numel() != 1:
        condition = condition.all()
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


class ByteAccountant:
    """Exact labeled byte reservations under one hard limit."""

    def __init__(self, limit_bytes: int) -> None:
        if isinstance(limit_bytes, bool) or limit_bytes < 0:
            raise ValueError("limit_bytes must be a nonnegative integer")
        self.limit_bytes = int(limit_bytes)
        self._reservations: dict[str, int] = {}
        self.high_water_bytes = 0

    @property
    def reserved_bytes(self) -> int:
        return sum(self._reservations.values())

    @property
    def available_bytes(self) -> int:
        return self.limit_bytes - self.reserved_bytes

    def reserve(self, label: str, nbytes: int) -> None:
        """Reserve before allocation, failing without changing accounting."""

        if not label or label in self._reservations:
            raise ValueError("byte reservation labels must be nonempty and unique")
        if isinstance(nbytes, bool) or nbytes < 0:
            raise ValueError("reserved bytes must be a nonnegative integer")
        proposed = self.reserved_bytes + int(nbytes)
        if proposed > self.limit_bytes:
            raise MemoryError(
                f"byte budget exceeded: requested={nbytes}, reserved={self.reserved_bytes}, "
                f"limit={self.limit_bytes}"
            )
        self._reservations[label] = int(nbytes)
        self.high_water_bytes = max(self.high_water_bytes, proposed)

    def release(self, label: str) -> int:
        """Release and return one exact reservation."""

        try:
            return self._reservations.pop(label)
        except KeyError as exc:
            raise KeyError(f"unknown byte reservation {label!r}") from exc


@dataclass(frozen=True)
class TrajectoryAdmissionPlan:
    """Byte-complete transaction bound for sequential rollout admission.

    ``retained_bytes`` and ``max_inflight_bytes`` are reserved before sampling.
    Each live trajectory then consumes ``resident_bytes_per_start`` until its
    unique stopped leaf is reached.  The planner only delays starts; it never
    changes a trajectory's logical address or sampled work.
    """

    limit_bytes: int
    retained_bytes: int
    max_inflight_bytes: int
    resident_bytes_per_start: int

    def __post_init__(self) -> None:
        values = (
            self.limit_bytes,
            self.retained_bytes,
            self.max_inflight_bytes,
            self.resident_bytes_per_start,
        )
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("admission byte-plan terms must be nonnegative integers")
        if self.resident_bytes_per_start == 0:
            raise ValueError("resident_bytes_per_start must be positive")

    @property
    def base_bytes(self) -> int:
        return self.retained_bytes + self.max_inflight_bytes

    def committed_bytes(self, active_starts: int) -> int:
        if isinstance(active_starts, bool) or active_starts < 0:
            raise ValueError("active starts must be a nonnegative integer")
        return self.base_bytes + int(active_starts) * self.resident_bytes_per_start

    def safe_admit(
        self,
        *,
        active_starts: int,
        unstarted: int,
        requested: int,
    ) -> int:
        """Return the largest transactional admission no greater than requested."""

        values = (active_starts, unstarted, requested)
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("admission counts must be nonnegative integers")
        available = self.limit_bytes - self.committed_bytes(active_starts)
        byte_capacity = max(0, available // self.resident_bytes_per_start)
        return min(int(unstarted), int(requested), int(byte_capacity))

    def require_progress_capacity(self, total_starts: int) -> None:
        """Fail before sampling if even one required live start cannot fit."""

        if isinstance(total_starts, bool) or total_starts < 0:
            raise ValueError("total starts must be a nonnegative integer")
        if total_starts and self.safe_admit(
            active_starts=0, unstarted=total_starts, requested=1
        ) != 1:
            raise MemoryError(
                "admission plan has no capacity for one trajectory start: "
                f"retained={self.retained_bytes}, "
                f"max_inflight={self.max_inflight_bytes}, "
                f"resident_per_start={self.resident_bytes_per_start}, "
                f"limit={self.limit_bytes}"
            )


@dataclass(frozen=True)
class ArenaField:
    """One struct-of-arrays column specification."""

    dtype: torch.dtype
    shape: tuple[int, ...] = ()
    fill: int | float = 0

    def __post_init__(self) -> None:
        if any(isinstance(size, bool) or size <= 0 for size in self.shape):
            raise ValueError("arena field dimensions must be positive integers")

    @property
    def row_bytes(self) -> int:
        elements = math.prod(self.shape) if self.shape else 1
        return elements * torch.empty((), dtype=self.dtype).element_size()


_OCCURRENCE_FIELDS: dict[str, ArenaField] = {
    "parent": ArenaField(torch.int64, fill=-1),
    "state_ref": ArenaField(torch.int32, fill=-1),
    "owner": ArenaField(torch.int32, fill=-1),
    "tree_id": ArenaField(torch.int32, fill=-1),
    "trajectory_id": ArenaField(torch.int64, fill=-1),
    "path_key": ArenaField(torch.int64),
    "child_begin": ArenaField(torch.int64, fill=-1),
    "child_count": ArenaField(torch.int32),
    "status": ArenaField(torch.uint8, fill=NEW),
}


class ChunkedOccurrenceArena:
    """Append-only chunked SoA storage addressed by integer handles.

    Optional fields let each rollout mode retain only what it consumes.  Chunks
    are allocated transactionally under :class:`ByteAccountant`; an allocation
    or hard-cap failure occurs before ``size`` or any occurrence status changes.
    """

    def __init__(
        self,
        capacity: int,
        chunk_nodes: int,
        device: torch.device | str,
        *,
        fields: Mapping[str, ArenaField] | None = None,
        accountant: ByteAccountant | None = None,
        label: str = "occurrence_arena",
    ) -> None:
        if isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("arena capacity must be a positive integer")
        if isinstance(chunk_nodes, bool) or chunk_nodes <= 0:
            raise ValueError("chunk_nodes must be a positive integer")
        if not label:
            raise ValueError("arena label must be nonempty")
        self.capacity = int(capacity)
        self.chunk_nodes = int(chunk_nodes)
        self.device = torch.device(device)
        self.fields = dict(_OCCURRENCE_FIELDS if fields is None else fields)
        if "status" not in self.fields:
            raise ValueError("an occurrence arena requires a status field")
        if not self.fields:
            raise ValueError("an occurrence arena requires at least one field")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if any(field.dtype not in integer_dtypes for field in self.fields.values()):
            raise ValueError("occurrence arena fields must use integer dtypes")
        self.accountant = accountant
        self.label = label
        self._chunks: dict[str, list[Tensor]] = {
            name: [] for name in self.fields
        }
        self._chunk_lengths: list[int] = []
        self._size = 0
        self._reserved_bytes = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_lengths)

    @property
    def row_bytes(self) -> int:
        return sum(field.row_bytes for field in self.fields.values())

    @property
    def reserved_bytes(self) -> int:
        return self._reserved_bytes

    @property
    def used_bytes(self) -> int:
        return self._size * self.row_bytes

    def _ensure_capacity(self, stop: int) -> None:
        required_chunks = math.ceil(stop / self.chunk_nodes)
        if required_chunks <= self.chunk_count:
            return
        new_lengths = [
            min(self.chunk_nodes, self.capacity - index * self.chunk_nodes)
            for index in range(self.chunk_count, required_chunks)
        ]
        added_bytes = sum(new_lengths) * self.row_bytes
        reservation = f"{self.label}_chunks_{self.chunk_count}_{required_chunks}"
        if self.accountant is not None:
            self.accountant.reserve(reservation, added_bytes)
        allocated: dict[str, list[Tensor]] = {name: [] for name in self.fields}
        try:
            for length in new_lengths:
                for name, field in self.fields.items():
                    allocated[name].append(
                        torch.full(
                            (length, *field.shape),
                            field.fill,
                            dtype=field.dtype,
                            device=self.device,
                        )
                    )
        except BaseException:
            if self.accountant is not None:
                self.accountant.release(reservation)
            raise
        for name, chunks in allocated.items():
            self._chunks[name].extend(chunks)
        self._chunk_lengths.extend(new_lengths)
        self._reserved_bytes += added_bytes

    def allocate(self, count: int) -> Tensor:
        """Allocate ``count`` NEW occurrences and return int64 handles ``[count]``."""

        if isinstance(count, bool) or count < 0:
            raise ValueError("occurrence count must be a nonnegative integer")
        stop = self._size + int(count)
        if stop > self.capacity:
            raise MemoryError(
                f"occurrence arena overflow: requested_stop={stop}, capacity={self.capacity}"
            )
        self._ensure_capacity(stop)
        start = self._size
        self._size = stop
        return torch.arange(start, stop, dtype=torch.int64, device=self.device)

    def require_capacity(self, count: int) -> None:
        """Validate one future allocation without mutating arena state."""

        if isinstance(count, bool) or count < 0:
            raise ValueError("occurrence count must be a nonnegative integer")
        stop = self._size + int(count)
        if stop > self.capacity:
            raise MemoryError(
                f"occurrence arena overflow: requested_stop={stop}, capacity={self.capacity}"
            )

    def _validate_handles(self, handles: Tensor) -> Tensor:
        handles = handles.to(device=self.device, dtype=torch.int64)
        if handles.ndim != 1:
            raise ValueError("occurrence handles must be one-dimensional")
        valid = ((handles >= 0) & (handles < self._size)).all()
        _device_assert(valid, "occurrence handle is out of range")
        return handles

    def _index_plan(self, handles: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        """Validate handles and compute chunk-local positions once."""

        handles = self._validate_handles(handles)
        chunk_ids = torch.div(handles, self.chunk_nodes, rounding_mode="floor")
        offsets = torch.remainder(handles, self.chunk_nodes)
        if self.device.type == "cuda":
            selectors = tuple(
                chunk_ids == chunk_id for chunk_id in range(self.chunk_count)
            )
        else:
            selectors = tuple(
                (chunk_ids == chunk_id).nonzero(as_tuple=False).squeeze(1)
                for chunk_id in range(self.chunk_count)
            )
        return handles, offsets, selectors

    def _gather_from_plan(
        self,
        names: tuple[str, ...],
        specs: Mapping[str, ArenaField],
        handles: Tensor,
        offsets: Tensor,
        selectors: tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        outputs = {
            name: torch.full(
                (handles.numel(), *field.shape),
                field.fill,
                dtype=field.dtype,
                device=self.device,
            )
            for name, field in specs.items()
        }
        for chunk_id, selector in enumerate(selectors):
            if self.device.type == "cuda":
                length = self._chunk_lengths[chunk_id]
                safe_offsets = offsets.clamp_max(length - 1)
                for name, field in specs.items():
                    shape = (handles.numel(),) + (1,) * len(field.shape)
                    outputs[name] = torch.where(
                        selector.reshape(shape),
                        self._chunks[name][chunk_id][safe_offsets],
                        outputs[name],
                    )
            else:
                selected_offsets = offsets[selector]
                for name in names:
                    outputs[name][selector] = self._chunks[name][chunk_id][
                        selected_offsets
                    ]
        return outputs

    def _scatter_from_plan(
        self,
        prepared: Mapping[str, Tensor],
        offsets: Tensor,
        selectors: tuple[Tensor, ...],
    ) -> None:
        for chunk_id, selector in enumerate(selectors):
            if self.device.type == "cuda":
                length = self._chunk_lengths[chunk_id]
                destination = offsets.clamp_max(length - 1)
                for name, value in prepared.items():
                    chunk = self._chunks[name][chunk_id]
                    current = chunk[destination]
                    shape = (selector.numel(),) + (1,) * (value.ndim - 1)
                    delta = torch.where(
                        selector.reshape(shape),
                        value - current,
                        torch.zeros_like(value),
                    )
                    # Semantic handles are unique. Nonmatching rows contribute
                    # exact integer zero even if their clamped offsets collide.
                    chunk.index_add_(0, destination, delta)
            else:
                selected_offsets = offsets[selector]
                for name, value in prepared.items():
                    self._chunks[name][chunk_id][selected_offsets] = value[selector]

    def gather_many(self, names: Iterable[str], handles: Tensor) -> dict[str, Tensor]:
        """Gather several fields while sharing one device chunk-index plan."""

        selected = tuple(names)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("gather field names must be nonempty and unique")
        try:
            specs = {name: self.fields[name] for name in selected}
        except KeyError as exc:
            raise KeyError(f"unknown occurrence field {exc.args[0]!r}") from exc
        handles, offsets, selectors = self._index_plan(handles)
        return self._gather_from_plan(selected, specs, handles, offsets, selectors)

    def gather(self, name: str, handles: Tensor) -> Tensor:
        """Gather one field without converting handles to host scalars."""

        return self.gather_many((name,), handles)[name]

    def scatter_many(self, handles: Tensor, values: Mapping[str, Tensor]) -> None:
        """Scatter several fields while sharing one device chunk-index plan."""

        if not values:
            raise ValueError("scatter values must contain at least one field")
        try:
            specs = {name: self.fields[name] for name in values}
        except KeyError as exc:
            raise KeyError(f"unknown occurrence field {exc.args[0]!r}") from exc
        handles, offsets, selectors = self._index_plan(handles)
        if handles.numel() > 1:
            ordered = handles.sort().values
            _device_assert(
                (ordered[1:] != ordered[:-1]).all(),
                "duplicate occurrence handle in one scatter",
            )
        prepared: dict[str, Tensor] = {}
        for name, value in values.items():
            field = specs[name]
            expected = (handles.numel(), *field.shape)
            if value.shape != expected or value.dtype != field.dtype:
                raise ValueError(
                    f"field {name!r} values must have shape {expected} and dtype {field.dtype}"
                )
            prepared[name] = value.to(self.device)
        self._scatter_from_plan(prepared, offsets, selectors)

    def scatter(self, name: str, handles: Tensor, values: Tensor) -> None:
        """Scatter one field using device index tensors only."""

        self.scatter_many(handles, {name: values})

    def fill(self, name: str, handles: Tensor, value: int | float) -> None:
        field = self.fields[name]
        values = torch.full(
            (handles.numel(), *field.shape),
            value,
            dtype=field.dtype,
            device=self.device,
        )
        self.scatter(name, handles, values)

    def transition(self, handles: Tensor, expected: int, new: int) -> None:
        """Apply one legal monotone status transition exactly once."""

        legal = {(NEW, QUEUED), (QUEUED, EVALUATING), (EVALUATING, DONE)}
        if (expected, new) not in legal:
            raise ValueError(f"illegal occurrence status transition {expected}->{new}")
        handles, offsets, selectors = self._index_plan(handles)
        if handles.numel() > 1:
            ordered = handles.sort().values
            _device_assert(
                (ordered[1:] != ordered[:-1]).all(),
                "duplicate occurrence handle in one status transition",
            )
        current = self._gather_from_plan(
            ("status",),
            {"status": self.fields["status"]},
            handles,
            offsets,
            selectors,
        )["status"]
        _device_assert(
            (current == expected).all(),
            f"occurrence status is not {expected} before transition to {new}",
        )
        self._scatter_from_plan(
            {"status": torch.full_like(current, new)}, offsets, selectors
        )


class DeviceWorkQueue:
    """Bounded insertion-order whose storage and live metadata reside on one device."""

    def __init__(self, capacity: int, device: torch.device | str) -> None:
        if isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("queue capacity must be a positive integer")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.storage = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.head = torch.zeros((), dtype=torch.int64, device=self.device)
        self.tail = torch.zeros((), dtype=torch.int64, device=self.device)
        self.count = torch.zeros((), dtype=torch.int64, device=self.device)
        # A deterministic mirror avoids reading device scalars at scheduler
        # boundaries while the device tensors remain kernel-visible metadata.
        self._host_count = 0
        # Logical enqueue-time runs mirror insertion-order mutations exactly.  They make
        # queue age observable without reading a device handle or clock scalar.
        self._age_runs: deque[tuple[int, int]] = deque()
        self._last_logical_time = 0

    @property
    def reserved_bytes(self) -> int:
        return self.storage.numel() * self.storage.element_size() + 3 * 8

    @property
    def host_count(self) -> int:
        return self._host_count

    def oldest_age(self, logical_time: int) -> int:
        """Return the exact oldest wait in scheduler ticks."""

        if isinstance(logical_time, bool) or logical_time < 0:
            raise ValueError("logical time must be a nonnegative integer")
        if not self._age_runs:
            return 0
        oldest = self._age_runs[0][1]
        if logical_time < oldest:
            raise ValueError("logical time cannot precede the oldest waiting row")
        return int(logical_time) - oldest

    def enqueue(
        self,
        handles: Tensor,
        *,
        arena: ChunkedOccurrenceArena | None = None,
        logical_time: int = 0,
    ) -> None:
        """Append handles, optionally transitioning them NEW -> QUEUED."""

        handles = handles.to(device=self.device, dtype=torch.int64)
        if handles.ndim != 1:
            raise ValueError("work handles must be one-dimensional")
        if isinstance(logical_time, bool) or logical_time < self._last_logical_time:
            raise ValueError("queue logical times must be nonnegative and monotone")
        added = int(handles.numel())
        if self._host_count + added > self.capacity:
            raise MemoryError(
                f"insertion-order overflow: depth={self._host_count}, added={added}, "
                f"capacity={self.capacity}"
            )
        if arena is not None:
            arena.transition(handles, NEW, QUEUED)
        positions = torch.remainder(
            self.tail + torch.arange(added, device=self.device), self.capacity
        )
        self.storage[positions] = handles
        self.tail.copy_(torch.remainder(self.tail + added, self.capacity))
        self.count.add_(added)
        self._host_count += added
        self._last_logical_time = int(logical_time)
        if added:
            if self._age_runs and self._age_runs[-1][1] == logical_time:
                previous, timestamp = self._age_runs.pop()
                self._age_runs.append((previous + added, timestamp))
            else:
                self._age_runs.append((added, int(logical_time)))

    def peek(self, count: int | None = None) -> Tensor:
        """Return a insertion-order-ordered copy without changing queue state."""

        selected = self._host_count if count is None else int(count)
        if selected < 0 or selected > self._host_count:
            raise ValueError("peek count is outside the live insertion-order depth")
        positions = torch.remainder(
            self.head + torch.arange(selected, device=self.device), self.capacity
        )
        return self.storage[positions].clone()

    def dequeue(
        self,
        count: int,
        *,
        arena: ChunkedOccurrenceArena | None = None,
    ) -> Tensor:
        """Pop insertion-order handles, optionally transitioning QUEUED -> EVALUATING."""

        if isinstance(count, bool) or count < 0 or count > self._host_count:
            raise ValueError("dequeue count is outside the live insertion-order depth")
        handles = self.peek(count)
        if arena is not None:
            arena.transition(handles, QUEUED, EVALUATING)
        self.head.copy_(torch.remainder(self.head + count, self.capacity))
        self.count.sub_(count)
        self._host_count -= count
        remaining = count
        while remaining:
            run_count, timestamp = self._age_runs.popleft()
            consumed = min(remaining, run_count)
            remaining -= consumed
            if consumed < run_count:
                self._age_runs.appendleft((run_count - consumed, timestamp))
        return handles

    @staticmethod
    def mark_done(arena: ChunkedOccurrenceArena, handles: Tensor) -> None:
        arena.transition(handles, EVALUATING, DONE)


@dataclass(frozen=True)
class MaskedStablePartition:
    """Fixed-shape stable partitions with device-resident validity/counts."""

    candidate_handles: Tensor  # int64 [F,B]
    valid_mask: Tensor  # bool [F,B], valid prefix in every row
    selected_count: Tensor  # int64 [F]
    family_values: Tensor  # same integer dtype as routing ids, [F]


def stable_partition_masked(
    handles: Tensor,
    family_ids: Tensor,
    family_values: Iterable[int],
) -> MaskedStablePartition:
    """Return one stable fixed-width candidate row per execution family.

    Counts remain device tensors. No boolean indexing, ``nonzero``, scalar read,
    or host transfer is used, so CUDA callers can feed the representation into
    masked queues/kernels without introducing a dynamic-shape synchronization.
    Invalid tails deliberately retain real input handles and must be ignored via
    ``valid_mask`` by every consumer.
    """

    values = tuple(int(value) for value in family_values)
    if not values or len(set(values)) != len(values):
        raise ValueError("family values must be nonempty and unique")
    if handles.ndim != 1 or family_ids.shape != handles.shape:
        raise ValueError("handles and family ids must be matching vectors")
    if handles.dtype != torch.int64 or family_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("partition handles must be int64 and family ids integer")
    if handles.device != family_ids.device:
        raise ValueError("partition handles and family ids must share a device")
    family_tensor = torch.tensor(values, dtype=family_ids.dtype, device=handles.device)
    membership = family_ids[:, None] == family_tensor[None, :]
    _device_assert(
        membership.sum(1).eq(1).all(),
        "every routed handle must belong to exactly one execution family",
    )
    candidates: list[Tensor] = []
    masks: list[Tensor] = []
    for column in range(len(values)):
        mask = membership[:, column]
        order = torch.argsort((~mask).to(torch.uint8), stable=True)
        candidates.append(handles[order])
        masks.append(mask[order])
    valid_mask = torch.stack(masks)
    return MaskedStablePartition(
        candidate_handles=torch.stack(candidates),
        valid_mask=valid_mask,
        selected_count=valid_mask.sum(1, dtype=torch.int64),
        family_values=family_tensor,
    )


class ReusableStagingSlabs:
    """Preallocated family staging tensors with exact byte accounting."""

    def __init__(
        self,
        capacity: int,
        specs: Mapping[str, ArenaField],
        device: torch.device | str,
        *,
        accountant: ByteAccountant | None = None,
        label: str = "staging_slabs",
    ) -> None:
        if isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("staging capacity must be a positive integer")
        if not specs:
            raise ValueError("at least one staging slab is required")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.specs = dict(specs)
        self.reserved_bytes = capacity * sum(spec.row_bytes for spec in specs.values())
        if accountant is not None:
            accountant.reserve(label, self.reserved_bytes)
        try:
            self.tensors = {
                name: torch.empty(
                    (capacity, *spec.shape), dtype=spec.dtype, device=self.device
                )
                for name, spec in specs.items()
            }
        except BaseException:
            if accountant is not None:
                accountant.release(label)
            raise

    def view(self, name: str, rows: int) -> Tensor:
        if isinstance(rows, bool) or rows < 0 or rows > self.capacity:
            raise ValueError("staging view rows are outside slab capacity")
        try:
            return self.tensors[name][:rows]
        except KeyError as exc:
            raise KeyError(f"unknown staging slab {name!r}") from exc


class DrawPurpose(IntEnum):
    """Disjoint counter-based RNG namespaces from the rollout pseudocode."""

    PRIVATE = 1
    ROUND = 2
    POSTERIOR = 3
    ACTION = 4
    CHANCE = 5
    COMPLETION = 6
    LEARNER = 7


_MASK63 = (1 << 63) - 1
LOGICAL_RNG_VERSION = "counter_hash_v2"


def _mix63(value: Tensor) -> Tensor:
    value = torch.bitwise_xor(value, value >> 30)
    value = (value * 0x3C79AC492BA7B653) & _MASK63
    value = torch.bitwise_xor(value, value >> 27)
    value = (value * 0x1C69B3F74AC4AE35) & _MASK63
    return torch.bitwise_xor(value, value >> 31) & _MASK63


def _mix63_scalar(value: int) -> int:
    """Scalar counterpart used to pre-hash iteration/purpose namespaces."""

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
    trajectory: Tensor,
    path: Tensor,
    draw: Tensor | int,
    purpose: DrawPurpose | int,
) -> Tensor:
    """Return schedule- and placement-independent int64 keys ``[...]``."""

    if trajectory.device != path.device:
        raise ValueError("trajectory and path addresses must share a device")
    trajectory, path = torch.broadcast_tensors(
        trajectory.to(torch.int64), path.to(torch.int64)
    )
    draw_tensor = torch.as_tensor(draw, dtype=torch.int64, device=trajectory.device)
    trajectory, path, draw_tensor = torch.broadcast_tensors(
        trajectory, path, draw_tensor
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
    # Independent odd multipliers construct one counter from the logical
    # address.  One avalanche round is sufficient because the components are
    # counters, not entropy sources; hashing every component separately only
    # adds tiny GPU launches without improving schedule independence.
    counter = (
        trajectory * 0x1C69B3F74AC4AE35
        + path * 0x3C79AC492BA7B653
        + draw_tensor * 0x1F83D9ABFB41BD6B
        + namespace
    ) & _MASK63
    return _mix63(counter)


def child_path_keys(parent_path: Tensor, child_ordinal: Tensor | int) -> Tensor:
    """Extend an abstract path by its one-based child ordinal."""

    ordinal = torch.as_tensor(
        child_ordinal, dtype=torch.int64, device=parent_path.device
    )
    parent_path, ordinal = torch.broadcast_tensors(parent_path.to(torch.int64), ordinal)
    if ordinal.device.type == "cuda":
        torch._assert_async((ordinal > 0).all(), "child ordinals must be positive")
    elif not bool((ordinal > 0).all()):
        raise ValueError("child ordinals must be positive")
    return _mix63(torch.bitwise_xor(parent_path, (ordinal + 0x243F6A8885A308D3) & _MASK63))


def logical_uniform(keys: Tensor, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Map int64 keys to open-interval uniforms without mutable RNG state."""

    if keys.dtype != torch.int64:
        raise ValueError("logical RNG keys must be int64")
    # Use 24 high bits: exactly representable in fp32 and identical on CPU/CUDA.
    mantissa = torch.bitwise_and(keys >> 39, (1 << 24) - 1).to(torch.float32)
    return ((mantissa + 0.5) * (1.0 / (1 << 24))).to(dtype)


def logical_categorical(probability: Tensor, keys: Tensor) -> Tensor:
    """Sample one categorical index per row from logical-address keys."""

    if probability.ndim != 2 or keys.shape != probability.shape[:1]:
        raise ValueError("probability must be [B,E] and keys [B]")
    if probability.device != keys.device:
        raise ValueError("probabilities and logical keys must share a device")
    valid = torch.isfinite(probability).all() & (probability >= 0).all()
    mass = probability.sum(1)
    _device_assert(valid & (mass > 0).all(), "categorical rows must be finite and positive")
    return logical_categorical_from_valid_probability(probability, keys, mass=mass)


def logical_categorical_from_valid_probability(
    probability: Tensor,
    keys: Tensor,
    *,
    mass: Tensor | None = None,
) -> Tensor:
    """Sample backend-validated nonnegative rows without repeated reductions."""

    if probability.ndim != 2 or keys.shape != probability.shape[:1]:
        raise ValueError("probability must be [B,E] and keys [B]")
    if probability.device != keys.device:
        raise ValueError("probabilities and logical keys must share a device")
    if keys.dtype != torch.int64:
        raise ValueError("logical RNG keys must be int64")
    if mass is None:
        mass = probability.sum(1)
    elif mass.shape != probability.shape[:1] or mass.device != probability.device:
        raise ValueError("categorical mass must be device-local [B]")
    cdf = probability.cumsum(1)
    threshold = (logical_uniform(keys) * mass).unsqueeze(1).contiguous()
    return torch.searchsorted(cdf, threshold, right=False).squeeze(1).clamp_max(
        probability.shape[1] - 1
    )



def arena_fields(
    required: Iterable[str],
    extra: Mapping[str, ArenaField] | None = None,
) -> dict[str, ArenaField]:
    """Build a mode-minimal occurrence schema from contracted common fields."""

    names = set(required) | {"status"}
    unknown = names - _OCCURRENCE_FIELDS.keys()
    if unknown:
        raise KeyError(f"unknown common occurrence fields: {sorted(unknown)}")
    result = {name: _OCCURRENCE_FIELDS[name] for name in names}
    if extra:
        overlap = result.keys() & extra.keys()
        if overlap:
            raise ValueError(f"extra arena fields overlap common fields: {sorted(overlap)}")
        result.update(extra)
    return result
