"""Schema-derived paged, sealed, rollout, and training storage.

One :class:`TensorSchema` drives allocation, capacity accounting, sealing,
validation, and diagnostics.  Growing stores expose only logical integer row
ids; tensor addresses are never identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from pbg.core.execution import ByteAccountant


ROLLOUT_STORAGE_SCHEMA_VERSION = "pbg-rollout-storage-v2"
TRAINING_STORAGE_SCHEMA_VERSION = "pbg-training-storage-v2"


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class TensorFieldSchema:
    """One structure-of-arrays column with a fixed trailing shape."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...] = ()
    fill: int | float | bool = 0
    alignment: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor field name must be nonempty")
        shape = tuple(int(value) for value in self.shape)
        if any(value <= 0 for value in shape):
            raise ValueError("tensor field dimensions must be positive")
        object.__setattr__(self, "shape", shape)
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("field alignment must be a positive power of two")
        _dtype_size(self.dtype)

    @property
    def elements_per_row(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def row_payload_bytes(self) -> int:
        return self.elements_per_row * _dtype_size(self.dtype)

    def payload_bytes(self, rows: int) -> int:
        return int(rows) * self.row_payload_bytes

    def capacity_bytes(self, rows: int) -> int:
        return _align(self.payload_bytes(rows), self.alignment)


@dataclass(frozen=True)
class TensorSchema:
    """Immutable allocation and byte-accounting source of truth."""

    name: str
    fields: tuple[TensorFieldSchema, ...]
    chunk_header_bytes: int = 0
    allocator_slack_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("schema name must be nonempty")
        fields = tuple(self.fields)
        if not fields or len({field.name for field in fields}) != len(fields):
            raise ValueError("schema fields must be nonempty and uniquely named")
        object.__setattr__(self, "fields", fields)
        if any(
            isinstance(value, bool) or value < 0
            for value in (self.chunk_header_bytes, self.allocator_slack_bytes)
        ):
            raise ValueError("schema overhead byte counts must be nonnegative")

    @property
    def by_name(self) -> Mapping[str, TensorFieldSchema]:
        return MappingProxyType({field.name: field for field in self.fields})

    @property
    def row_payload_bytes(self) -> int:
        return sum(field.row_payload_bytes for field in self.fields)

    def payload_bytes(self, rows: int) -> int:
        if isinstance(rows, bool) or rows < 0:
            raise ValueError("row count must be nonnegative")
        return sum(field.payload_bytes(rows) for field in self.fields)

    def capacity_bytes(self, rows: int, *, chunks: int = 1) -> int:
        if isinstance(rows, bool) or rows < 0 or isinstance(chunks, bool) or chunks < 0:
            raise ValueError("capacity rows/chunks must be nonnegative")
        if rows == 0 and chunks:
            raise ValueError("an empty allocation cannot own chunks")
        if rows and chunks == 0:
            raise ValueError("a nonempty allocation must own at least one chunk")
        return (
            sum(field.capacity_bytes(rows) for field in self.fields)
            + chunks * (self.chunk_header_bytes + self.allocator_slack_bytes)
        )

    def allocate(self, rows: int, device: torch.device | str) -> dict[str, Tensor]:
        if isinstance(rows, bool) or rows < 0:
            raise ValueError("allocation rows must be nonnegative")
        target = torch.device(device)
        return {
            spec.name: torch.full(
                (rows, *spec.shape), spec.fill, dtype=spec.dtype, device=target
            )
            for spec in self.fields
        }

    def validate(self, columns: Mapping[str, Tensor], rows: int) -> None:
        if set(columns) != {field.name for field in self.fields}:
            raise ValueError(f"{self.name} columns do not match the immutable schema")
        devices = set()
        for spec in self.fields:
            value = columns[spec.name]
            expected = (rows, *spec.shape)
            if tuple(value.shape) != expected or value.dtype != spec.dtype:
                raise ValueError(
                    f"{self.name}.{spec.name} must have shape {expected} and "
                    f"dtype {spec.dtype}, got {tuple(value.shape)} {value.dtype}"
                )
            devices.add(value.device)
        if len(devices) > 1:
            raise ValueError(f"{self.name} columns must share one device")


def paged_capacity_bytes(schema: TensorSchema, rows: int, page_rows: int) -> int:
    """Return the exact capacity of the pages needed for ``rows``.

    Alignment and per-page overhead apply independently to every allocation,
    exactly as they do in :class:`PagedTensorArena`.  Treating several pages as
    one large allocation can undercount field alignment and is therefore not a
    valid admission forecast.
    """

    if isinstance(rows, bool) or rows < 0:
        raise ValueError("row count must be nonnegative")
    if isinstance(page_rows, bool) or page_rows <= 0:
        raise ValueError("page_rows must be positive")
    full, remainder = divmod(int(rows), int(page_rows))
    total = full * schema.capacity_bytes(int(page_rows), chunks=1)
    if remainder:
        # Arenas allocate a full final page unless a separate maximum_rows
        # truncates it. Production rollout owners have no count capacity.
        total += schema.capacity_bytes(int(page_rows), chunks=1)
    return total

@dataclass(frozen=True)
class SealedTensorBlock:
    """Exact contiguous destination columns with no spare logical rows."""

    schema: TensorSchema
    columns: Mapping[str, Tensor]
    rows: int
    allocated_capacity_bytes: int

    def __post_init__(self) -> None:
        columns = MappingProxyType(dict(self.columns))
        object.__setattr__(self, "columns", columns)
        self.schema.validate(columns, self.rows)
        expected = self.schema.capacity_bytes(self.rows, chunks=int(self.rows > 0))
        if self.allocated_capacity_bytes != expected:
            raise ValueError(
                f"sealed block accounting mismatch: declared={self.allocated_capacity_bytes}, "
                f"schema={expected}"
            )

    def __getitem__(self, name: str) -> Tensor:
        return self.columns[name]

    @property
    def logical_payload_bytes(self) -> int:
        return self.schema.payload_bytes(self.rows)

    @property
    def tensor_storage_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size() for value in self.columns.values()
        )

    def assert_exact_payload(self) -> None:
        if self.tensor_storage_bytes != self.logical_payload_bytes:
            raise RuntimeError(
                f"{self.schema.name} destination payload is not exact: "
                f"tensors={self.tensor_storage_bytes}, schema={self.logical_payload_bytes}"
            )


class PagedTensorArena:
    """Append-only schema allocation with stable logical integer row ids.

    A page is the indivisible allocation/admission unit.  Reserving rows first
    admits every newly required page as one transaction and publishes neither
    page nor cursor movement unless all allocations succeed.
    """

    def __init__(
        self,
        schema: TensorSchema,
        page_rows: int,
        device: torch.device | str,
        *,
        accountant: ByteAccountant | None = None,
        label: str | None = None,
        maximum_rows: int | None = None,
    ) -> None:
        if isinstance(page_rows, bool) or page_rows <= 0:
            raise ValueError("page_rows must be a positive integer")
        if maximum_rows is not None and (
            isinstance(maximum_rows, bool) or maximum_rows <= 0
        ):
            raise ValueError("maximum_rows must be positive or None")
        self.schema = schema
        self.page_rows = int(page_rows)
        self.device = torch.device(device)
        self.accountant = accountant
        self.label = schema.name if label is None else str(label)
        if not self.label:
            raise ValueError("arena label must be nonempty")
        self.maximum_rows = maximum_rows
        self._pages: list[dict[str, Tensor]] = []
        self._lengths: list[int] = []
        self._reservation_labels: list[str] = []
        self._size = 0
        self._released = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def capacity_rows(self) -> int:
        return sum(self._lengths)

    @property
    def allocated_capacity_bytes(self) -> int:
        return sum(
            self.schema.capacity_bytes(length, chunks=1) for length in self._lengths
        )

    @property
    def logical_payload_bytes(self) -> int:
        return self.schema.payload_bytes(self._size)

    @property
    def page_slack_bytes(self) -> int:
        return self.allocated_capacity_bytes - self.schema.payload_bytes(
            self.capacity_rows
        ) + self.schema.payload_bytes(self.capacity_rows - self._size)

    def _check_live(self) -> None:
        if self._released:
            raise RuntimeError("paged arena has been released")

    def reset(self, *, release_pages: bool = False) -> None:
        """Return the logical cursor to zero, optionally dropping admitted pages.

        Retaining pages makes repeated iterations reuse the same byte-admitted
        allocation.  Dropping them releases the ledger reservations but keeps
        the arena live so a later iteration can admit a different realized
        extent.
        """

        self._check_live()
        self._size = 0
        if not release_pages:
            return
        self._pages.clear()
        self._lengths.clear()
        if self.accountant is not None:
            self.accountant.release_many(self._reservation_labels)
        self._reservation_labels.clear()

    def lookup_rows(self, name: str, keys: Tensor) -> Tensor:
        """Return logical rows for unique scalar keys, or ``-1`` when absent.

        The search is page-bounded: it never builds a capacity-wide validity
        map or a contiguous copy of every stored key.  The named schema field
        must be scalar and every stored key must be unique.
        """

        self._check_live()
        spec = self.schema.by_name[name]
        if spec.shape:
            raise ValueError("lookup key field must be scalar")
        keys = keys.to(device=self.device, dtype=spec.dtype)
        if keys.ndim != 1:
            raise ValueError("lookup keys must be one-dimensional")
        result = torch.full(
            (keys.numel(),), -1, dtype=torch.int64, device=self.device
        )
        cursor = 0
        for columns, capacity in zip(self._pages, self._lengths, strict=True):
            used = min(capacity, self._size - cursor)
            if used <= 0:
                break
            matches = keys[:, None] == columns[name][None, :used]
            hit = matches.any(1)
            candidate = cursor + matches.to(torch.int64).argmax(1)
            result = torch.where(hit & (result < 0), candidate, result)
            cursor += used
        return result

    def _next_page_lengths(self, stop: int) -> list[int]:
        lengths: list[int] = []
        capacity = self.capacity_rows
        while stop > capacity:
            if self.maximum_rows is not None and capacity >= self.maximum_rows:
                raise MemoryError(f"{self.label} maximum row capacity exhausted")
            length = self.page_rows
            if self.maximum_rows is not None:
                length = min(length, self.maximum_rows - capacity)
            lengths.append(length)
            capacity += length
        return lengths

    def _allocate_pages(self, lengths: Iterable[int]) -> None:
        self._check_live()
        selected = tuple(int(length) for length in lengths)
        if not selected:
            return
        first = len(self._pages)
        labels = tuple(
            f"{self.label}.page.{first + offset}" for offset in range(len(selected))
        )
        reservations = {
            label: self.schema.capacity_bytes(length, chunks=1)
            for label, length in zip(labels, selected, strict=True)
        }
        if self.accountant is not None:
            self.accountant.reserve_many(reservations)
        try:
            pages = [self.schema.allocate(length, self.device) for length in selected]
        except BaseException:
            if self.accountant is not None:
                self.accountant.release_many(labels)
            raise
        self._pages.extend(pages)
        self._lengths.extend(selected)
        self._reservation_labels.extend(labels)

    def reserve_rows(self, count: int) -> Tensor:
        """Reserve stable ids before their payload is populated."""

        self._check_live()
        if isinstance(count, bool) or count < 0:
            raise ValueError("reserved row count must be nonnegative")
        stop = self._size + int(count)
        if self.maximum_rows is not None and stop > self.maximum_rows:
            raise MemoryError(
                f"{self.label} overflow: requested_stop={stop}, maximum={self.maximum_rows}"
            )
        self._allocate_pages(self._next_page_lengths(stop))
        start = self._size
        self._size = stop
        return torch.arange(start, stop, dtype=torch.int64, device=self.device)

    def append(self, values: Mapping[str, Tensor]) -> Tensor:
        """Append matching schema rows and return their stable logical ids."""

        if set(values) != {field.name for field in self.schema.fields}:
            raise ValueError("append columns do not match schema")
        counts = {int(value.shape[0]) for value in values.values() if value.ndim >= 1}
        if len(counts) != 1:
            raise ValueError("append columns must have one matching leading extent")
        count = counts.pop()
        for spec in self.schema.fields:
            value = values[spec.name]
            if tuple(value.shape) != (count, *spec.shape) or value.dtype != spec.dtype:
                raise ValueError(f"append value for {spec.name!r} violates its schema")
            if value.device != self.device:
                raise ValueError("append tensors must already be on the arena device")
        ids = self.reserve_rows(count)
        self.write(ids, values)
        return ids

    def write(self, rows: Tensor, values: Mapping[str, Tensor]) -> None:
        """Populate already-reserved unique logical rows."""

        self._check_live()
        rows = rows.to(device=self.device, dtype=torch.int64)
        if rows.ndim != 1:
            raise ValueError("logical row ids must be one-dimensional")
        if rows.numel():
            torch._assert_async(
                ((rows >= 0) & (rows < self._size)).all(),
                "logical row ids must be allocated rows",
            )
            if self.device.type == "cpu" and rows.unique().numel() != rows.numel():
                raise ValueError("logical row ids must be unique")
        for name, value in values.items():
            try:
                spec = self.schema.by_name[name]
            except KeyError as exc:
                raise KeyError(f"unknown schema field {name!r}") from exc
            if tuple(value.shape) != (rows.numel(), *spec.shape) or value.dtype != spec.dtype:
                raise ValueError(f"write value for {name!r} violates its schema")
            if value.device != self.device:
                raise ValueError("write tensors must already be on the arena device")
        page_id = torch.div(rows, self.page_rows, rounding_mode="floor")
        offset = torch.remainder(rows, self.page_rows)
        # Each page update uses a fixed, admitted page-width match matrix.
        # This avoids CUDA ``nonzero`` shape discovery and its host sync.
        for index, (columns, length) in enumerate(
            zip(self._pages, self._lengths, strict=True)
        ):
            positions = torch.arange(length, device=self.device)
            matches = (page_id[:, None] == index) & (
                offset[:, None] == positions[None, :]
            )
            present = matches.any(0)
            if rows.numel() == 0:
                continue
            source = matches.to(torch.int64).argmax(0)
            for name, value in values.items():
                candidate = value[source]
                mask = present.reshape(
                    (length,) + (1,) * (candidate.ndim - 1)
                )
                columns[name].copy_(torch.where(mask, candidate, columns[name]))

    def gather(self, name: str, rows: Tensor) -> Tensor:
        self._check_live()
        spec = self.schema.by_name[name]
        rows = rows.to(device=self.device, dtype=torch.int64)
        if rows.ndim != 1:
            raise ValueError("gather row ids must be one-dimensional")
        if rows.numel():
            torch._assert_async(
                ((rows >= 0) & (rows < self._size)).all(),
                "gather row ids are outside the logical arena",
            )
        output = torch.zeros(
            (rows.numel(), *spec.shape), dtype=spec.dtype, device=self.device
        )
        page_id = torch.div(rows, self.page_rows, rounding_mode="floor")
        offset = torch.remainder(rows, self.page_rows)
        for index, (columns, length) in enumerate(
            zip(self._pages, self._lengths, strict=True)
        ):
            local = offset.clamp_max(length - 1)
            candidate = columns[name][local]
            mask = (page_id == index).reshape(
                (rows.numel(),) + (1,) * len(spec.shape)
            )
            output = torch.where(mask, candidate, output)
        return output

    def require_rows(self, additional_rows: int) -> None:
        """Admit future rows without advancing the logical row cursor."""

        if isinstance(additional_rows, bool) or additional_rows < 0:
            raise ValueError("required rows must be nonnegative")
        stop = self._size + int(additional_rows)
        if self.maximum_rows is not None and stop > self.maximum_rows:
            raise MemoryError(
                f"{self.label} overflow: requested_stop={stop}, maximum={self.maximum_rows}"
            )
        self._allocate_pages(self._next_page_lengths(stop))

    def rollback_page_growth(self, page_count: int) -> None:
        """Remove empty pages published after a failed grouped admission."""

        if not 0 <= page_count <= self.page_count:
            raise ValueError("invalid page-growth rollback checkpoint")
        if self._size > sum(self._lengths[:page_count]):
            raise RuntimeError("cannot roll back pages containing logical rows")
        labels = self._reservation_labels[page_count:]
        del self._pages[page_count:]
        del self._lengths[page_count:]
        del self._reservation_labels[page_count:]
        if self.accountant is not None:
            self.accountant.release_many(labels)



    def seal(
        self,
        *,
        accountant: ByteAccountant | None = None,
        label: str | None = None,
        scratch_rows: int = 1,
        release_source: bool = True,
        row_order: Tensor | None = None,
    ) -> tuple[SealedTensorBlock, "SealMetrics"]:
        """Copy stable chunks once into an exact contiguous destination.

        ``row_order`` optionally gives the source row for each destination row.
        It is topology-sized metadata, not payload scratch, and lets schedulers
        seal their out-of-order family packs without first concatenating and
        permuting every wide input column.
        """

        self._check_live()
        if isinstance(scratch_rows, bool) or scratch_rows <= 0:
            raise ValueError("scratch_rows must be positive")
        if row_order is not None:
            row_order = row_order.to(device=self.device, dtype=torch.int64)
            if row_order.shape != (self._size,):
                raise ValueError("seal row_order must contain every logical row")
            if self._size and not torch.equal(
                row_order.sort().values,
                torch.arange(self._size, dtype=torch.int64, device=self.device),
            ):
                raise ValueError("seal row_order must be a permutation")
        ledger = self.accountant if accountant is None else accountant
        if ledger is not None and self.accountant is None:
            raise ValueError("accounted sealing requires source chunks on the same ledger")
        if (
            ledger is not None
            and self.accountant is not None
            and ledger is not self.accountant
        ):
            raise ValueError("source and seal reservations must use one byte ledger")
        if ledger is not None and sum(
            ledger.reservation(label) for label in self._reservation_labels
        ) != self.allocated_capacity_bytes:
            raise RuntimeError("source chunk ledger does not match schema capacity")
        destination_bytes = self.schema.capacity_bytes(
            self._size, chunks=int(self._size > 0)
        )
        bounded_rows = min(scratch_rows, max(self._size, 1))
        scratch_bytes = self.schema.payload_bytes(bounded_rows)
        destination_label = f"{self.label}.sealed" if label is None else str(label)
        scratch_label = f"{destination_label}.scratch"
        reservations = {
            destination_label: destination_bytes,
            scratch_label: scratch_bytes,
        }
        if ledger is not None:
            ledger.reserve_many(reservations)
        try:
            destination = self.schema.allocate(self._size, self.device)
            # One reusable bounded scratch column set.  It deliberately has no
            # payload proportional to total rows and replaces repeated cat.
            scratch = self.schema.allocate(bounded_rows, self.device)
            cursor = 0
            if row_order is None:
                for columns, capacity in zip(self._pages, self._lengths, strict=True):
                    used = min(capacity, self._size - cursor)
                    copied = 0
                    while copied < used:
                        count = min(bounded_rows, used - copied)
                        for spec in self.schema.fields:
                            scratch[spec.name][:count].copy_(
                                columns[spec.name][copied : copied + count]
                            )
                            destination[spec.name][
                                cursor + copied : cursor + copied + count
                            ].copy_(scratch[spec.name][:count])
                        copied += count
                    cursor += used
            else:
                while cursor < self._size:
                    count = min(bounded_rows, self._size - cursor)
                    selected = row_order[cursor : cursor + count]
                    page_id = torch.div(
                        selected, self.page_rows, rounding_mode="floor"
                    )
                    offset = torch.remainder(selected, self.page_rows)
                    for spec in self.schema.fields:
                        target = scratch[spec.name][:count]
                        target.zero_()
                        for page_index, (columns, capacity) in enumerate(
                            zip(self._pages, self._lengths, strict=True)
                        ):
                            local = offset.clamp_max(capacity - 1)
                            candidate = columns[spec.name][local]
                            mask = (page_id == page_index).reshape(
                                (count,) + (1,) * len(spec.shape)
                            )
                            target.copy_(torch.where(mask, candidate, target))
                        destination[spec.name][cursor : cursor + count].copy_(target)
                    cursor += count
            if cursor != self._size:
                raise RuntimeError("seal copy extent does not match logical row count")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            block = SealedTensorBlock(
                schema=self.schema,
                columns=destination,
                rows=self._size,
                allocated_capacity_bytes=destination_bytes,
            )
            block.assert_exact_payload()
        except BaseException:
            if ledger is not None:
                ledger.release_many(reservations)
            raise
        del scratch
        if ledger is not None:
            ledger.release(scratch_label)
        metrics = SealMetrics(
            source_capacity_bytes=self.allocated_capacity_bytes,
            destination_capacity_bytes=destination_bytes,
            scratch_capacity_bytes=scratch_bytes,
            peak_accounted_bytes=(
                self.allocated_capacity_bytes + destination_bytes + scratch_bytes
            ),
            logical_payload_bytes=block.logical_payload_bytes,
            source_pages=self.page_count,
        )
        if release_source:
            self.release()
        return block, metrics

    def release(self) -> None:
        """Drop source chunks and their accounting reservations exactly once."""

        self._check_live()
        self._pages.clear()
        self._lengths.clear()
        if self.accountant is not None:
            self.accountant.release_many(self._reservation_labels)
        self._reservation_labels.clear()
        self._released = True


class PagedRowAccumulator:
    """Schema-on-first-write paged owner for one homogeneous row stream.

    The first realized pack fixes an immutable schema.  Later packs move into
    admitted pages immediately; sealing is the only source-plus-destination
    overlap and uses :class:`PagedTensorArena`'s bounded scratch copy.
    """

    def __init__(
        self,
        label: str,
        page_rows: int,
        device: torch.device | str,
        *,
        accountant: ByteAccountant | None = None,
    ) -> None:
        self.label = str(label)
        self.page_rows = int(page_rows)
        self.device = torch.device(device)
        self.accountant = accountant
        self.arena: PagedTensorArena | None = None

    @property
    def rows(self) -> int:
        return 0 if self.arena is None else self.arena.size

    def append(self, columns: Mapping[str, Tensor]) -> Tensor:
        if not columns:
            raise ValueError("paged row packs require at least one field")
        if self.arena is None:
            schema = TensorSchema(
                self.label,
                tuple(
                    TensorFieldSchema(
                        name,
                        value.dtype,
                        tuple(value.shape[1:]),
                        alignment=min(max(value.element_size(), 1), 16),
                    )
                    for name, value in columns.items()
                ),
            )
            self.arena = PagedTensorArena(
                schema,
                self.page_rows,
                self.device,
                accountant=self.accountant,
                label=self.label,
            )
        return self.arena.append(columns)

    def seal(self, *, row_order: Tensor | None = None) -> tuple[SealedTensorBlock, SealMetrics]:
        if self.arena is None:
            raise RuntimeError("cannot seal an unrealized row stream")
        return self.arena.seal(
            scratch_rows=self.page_rows,
            row_order=row_order,
            release_source=True,
        )


def admit_arena_growth(requests: Mapping[PagedTensorArena, int]) -> None:
    """Admit and allocate several arena extents as one publication transaction.

    This is the child-wave primitive: every newly required page is reserved
    before any allocation, all tensor pages are built off-owner, and arena page
    lists are updated only after the entire group succeeds.
    """

    selected = tuple(requests.items())
    if not selected:
        return
    accountants = {arena.accountant for arena, _ in selected}
    if len(accountants) != 1:
        raise ValueError("grouped arena growth requires one shared byte accountant")
    accountant = next(iter(accountants))
    plans: list[tuple[PagedTensorArena, tuple[int, ...], tuple[str, ...]]] = []
    reservations: dict[str, int] = {}
    for arena, additional_rows in selected:
        arena._check_live()
        if isinstance(additional_rows, bool) or additional_rows < 0:
            raise ValueError("grouped arena row counts must be nonnegative")
        stop = arena.size + int(additional_rows)
        if arena.maximum_rows is not None and stop > arena.maximum_rows:
            raise MemoryError(
                f"{arena.label} overflow: requested_stop={stop}, "
                f"maximum={arena.maximum_rows}"
            )
        lengths = tuple(arena._next_page_lengths(stop))
        first = arena.page_count
        labels = tuple(
            f"{arena.label}.page.{first + offset}"
            for offset in range(len(lengths))
        )
        plans.append((arena, lengths, labels))
        reservations.update(
            {
                label: arena.schema.capacity_bytes(length, chunks=1)
                for label, length in zip(labels, lengths, strict=True)
            }
        )
    labels = tuple(reservations)
    if not labels:
        return
    if accountant is not None:
        accountant.reserve_many(reservations)
    pending: list[tuple[PagedTensorArena, tuple[int, ...], tuple[str, ...], list[dict[str, Tensor]]]] = []
    try:
        for arena, lengths, arena_labels in plans:
            pages = [arena.schema.allocate(length, arena.device) for length in lengths]
            pending.append((arena, lengths, arena_labels, pages))
    except BaseException:
        if accountant is not None:
            accountant.release_many(labels)
        raise
    for arena, lengths, arena_labels, pages in pending:
        arena._pages.extend(pages)
        arena._lengths.extend(lengths)
        arena._reservation_labels.extend(arena_labels)


class PagedTensorColumn:
    """Tensor-style row indexing over one field of a paged arena."""

    def __init__(self, arena: PagedTensorArena, name: str) -> None:
        self.arena = arena
        self.name = name
        self.spec = arena.schema.by_name[name]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.arena.size, *self.spec.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.dtype

    @property
    def device(self) -> torch.device:
        return self.arena.device

    def numel(self) -> int:
        return self.arena.size * self.spec.elements_per_row

    def _rows(self, selector: Any) -> tuple[Tensor, tuple[int, ...], bool]:
        if isinstance(selector, slice):
            start, stop, step = selector.indices(self.arena.size)
            rows = torch.arange(start, stop, step, device=self.device)
            return rows, (rows.numel(),), False
        if isinstance(selector, int):
            row = selector if selector >= 0 else self.arena.size + selector
            return torch.tensor([row], dtype=torch.int64, device=self.device), (), True
        rows = torch.as_tensor(selector, device=self.device)
        if rows.dtype == torch.bool:
            if rows.ndim != 1 or rows.numel() != self.arena.size:
                raise IndexError("paged boolean row selector has the wrong extent")
            rows = rows.nonzero(as_tuple=False).squeeze(1)
        else:
            rows = rows.to(torch.int64)
        return rows.reshape(-1), tuple(rows.shape), False

    @staticmethod
    def _select(full: Tensor, row_shape: tuple[int, ...], rest: tuple[Any, ...]) -> Tensor:
        if not rest:
            return full
        first = rest[0]
        if isinstance(first, Tensor) and tuple(first.shape) == row_shape:
            flat = full.reshape(-1, *full.shape[len(row_shape) :])
            selected = flat[
                torch.arange(flat.shape[0], device=flat.device),
                first.to(device=flat.device, dtype=torch.int64).reshape(-1),
            ]
            selected = selected.reshape(*row_shape, *selected.shape[1:])
            return selected[(slice(None),) * len(row_shape) + rest[1:]]
        return full[(slice(None),) * len(row_shape) + rest]

    @staticmethod
    def _assign(
        full: Tensor,
        row_shape: tuple[int, ...],
        rest: tuple[Any, ...],
        value: Any,
    ) -> None:
        if not rest:
            full[...] = value
            return
        first = rest[0]
        if isinstance(first, Tensor) and tuple(first.shape) == row_shape:
            flat = full.reshape(-1, *full.shape[len(row_shape) :])
            positions = torch.arange(flat.shape[0], device=flat.device)
            columns = first.to(device=flat.device, dtype=torch.int64).reshape(-1)
            if len(rest) == 1:
                flat[positions, columns] = value
            else:
                selected = flat[positions, columns]
                selected[(slice(None),) * len(row_shape) + rest[1:]] = value
                flat[positions, columns] = selected
            return
        full[(slice(None),) * len(row_shape) + rest] = value

    def __getitem__(self, key: Any) -> Tensor:
        parts = key if isinstance(key, tuple) else (key,)
        rows, row_shape, scalar = self._rows(parts[0])
        full = self.arena.gather(self.name, rows)
        if scalar:
            full = full[0]
            row_shape = ()
        else:
            full = full.reshape(*row_shape, *self.spec.shape)
        return self._select(full, row_shape, parts[1:])

    def __setitem__(self, key: Any, value: Any) -> None:
        parts = key if isinstance(key, tuple) else (key,)
        rows, row_shape, scalar = self._rows(parts[0])
        full = self.arena.gather(self.name, rows)
        if scalar:
            shaped = full[0]
            self._assign(shaped, (), parts[1:], value)
            full = shaped.reshape(1, *self.spec.shape)
        else:
            shaped = full.reshape(*row_shape, *self.spec.shape)
            self._assign(shaped, row_shape, parts[1:], value)
            full = shaped.reshape(rows.numel(), *self.spec.shape)
        self.arena.write(rows, {self.name: full})


class RecyclableTensorPool:
    """Page-backed mutable slot owner with explicit last-consumer release.

    Logical rollout/topology ids must not use these handles as identities.
    Handles are physical live-slot addresses and may be reused only after the
    caller releases their last mutable-state consumer.  The append-only arena
    remains the sole page allocator and byte ledger owner.
    """

    def __init__(
        self,
        schema: TensorSchema,
        page_rows: int,
        device: torch.device | str,
        *,
        accountant: ByteAccountant | None = None,
        label: str | None = None,
        maximum_rows: int | None = None,
    ) -> None:
        self.arena = PagedTensorArena(
            schema,
            page_rows,
            device,
            accountant=accountant,
            label=label,
            maximum_rows=maximum_rows,
        )
        self._free: list[Tensor] = []
        self._live_rows = 0
        self.high_water_live_rows = 0

    @property
    def schema(self) -> TensorSchema:
        return self.arena.schema

    @property
    def device(self) -> torch.device:
        return self.arena.device

    @property
    def capacity_rows(self) -> int:
        return self.arena.capacity_rows

    @property
    def live_rows(self) -> int:
        return self._live_rows

    @property
    def allocated_capacity_bytes(self) -> int:
        return self.arena.allocated_capacity_bytes

    @property
    def live_payload_bytes(self) -> int:
        return self.schema.payload_bytes(self.live_rows)

    @property
    def slack_bytes(self) -> int:
        return self.allocated_capacity_bytes - self.live_payload_bytes

    def allocate(self, count: int) -> Tensor:
        if isinstance(count, bool) or count < 0:
            raise ValueError("pool allocation count must be nonnegative")
        requested = int(count)
        available = sum(batch.numel() for batch in self._free)
        reused_count = min(requested, available)
        fresh_count = requested - reused_count
        fresh_start = self.arena.size
        fresh = self.arena.reserve_rows(fresh_count)
        pieces: list[Tensor] = []
        remaining = reused_count
        while remaining:
            batch = self._free.pop()
            take = min(remaining, batch.numel())
            pieces.append(batch[:take])
            if take < batch.numel():
                self._free.append(batch[take:])
            remaining -= take
        if fresh_count:
            pieces.append(fresh)
        handles = (
            torch.cat(pieces)
            if len(pieces) > 1
            else (
                pieces[0]
                if pieces
                else torch.empty(0, dtype=torch.int64, device=self.device)
            )
        )
        self._live_rows += requested
        self.high_water_live_rows = max(self.high_water_live_rows, self.live_rows)
        return handles

    def required_fresh_rows(self, count: int) -> int:
        """Return rows not satisfiable from already released handles."""

        if isinstance(count, bool) or count < 0:
            raise ValueError("pool allocation count must be nonnegative")
        available = sum(batch.numel() for batch in self._free)
        return max(0, int(count) - available)

    def require_capacity(self, count: int) -> None:
        """Admit pages for a future allocation without consuming handles."""

        self.arena.require_rows(self.required_fresh_rows(count))

    def _device_handles(self, rows: Tensor | Iterable[int]) -> Tensor:
        handles = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
        if handles.ndim != 1:
            raise ValueError("pool handles must be one-dimensional")
        if handles.numel():
            torch._assert_async(
                ((handles >= 0) & (handles < self.arena.size)).all(),
                "pool handle is outside admitted storage",
            )
        return handles

    def _assert_not_free(self, handles: Tensor) -> None:
        for free in self._free:
            torch._assert_async(
                ~(handles[:, None] == free[None, :]).any(),
                "pool handle has already reached its last consumer",
            )

    def release_rows(self, rows: Tensor | Iterable[int]) -> None:
        handles = self._device_handles(rows)
        if handles.numel() > 1:
            ordered = handles.sort().values
            torch._assert_async(
                (ordered[1:] != ordered[:-1]).all(),
                "released pool handles must be unique",
            )
        self._assert_not_free(handles)
        if handles.numel():
            self._free.append(handles.clone())
        self._live_rows -= handles.numel()
        if self._live_rows < 0:
            raise RuntimeError("pool live-row accounting underflow")

    def write(self, rows: Tensor, values: Mapping[str, Tensor]) -> None:
        handles = self._device_handles(rows)
        self._assert_not_free(handles)
        self.arena.write(handles, values)

    def gather(self, name: str, rows: Tensor) -> Tensor:
        handles = self._device_handles(rows)
        self._assert_not_free(handles)
        return self.arena.gather(name, handles)

    def reset(self, *, release_pages: bool = False) -> None:
        """Release every iteration-local consumer and optionally its pages."""

        self._free.clear()
        self._live_rows = 0
        self.high_water_live_rows = 0
        self.arena.reset(release_pages=release_pages)

    def release(self) -> None:
        self._free.clear()
        self._live_rows = 0
        self.high_water_live_rows = 0
        self.arena.release()


@dataclass(frozen=True)
class SealMetrics:
    source_capacity_bytes: int
    destination_capacity_bytes: int
    scratch_capacity_bytes: int
    peak_accounted_bytes: int
    logical_payload_bytes: int
    source_pages: int


@dataclass(frozen=True)
class PartitionRolloutBlock:
    """Aligned natural input/decision/action columns for one partition."""

    partition_id: int
    inputs: SealedTensorBlock
    decisions: SealedTensorBlock
    actions: SealedTensorBlock | None = None
    action_ptr: Tensor | None = None
    input_pools: Mapping[str, SealedTensorBlock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.inputs.rows != self.decisions.rows:
            raise ValueError("partition input and decision rows must align")
        if self.actions is None:
            if self.action_ptr is not None:
                raise ValueError("sequential partitions cannot own an action pointer")
        else:
            if (
                self.action_ptr is None
                or self.action_ptr.dtype != torch.int32
                or self.action_ptr.shape != (self.decisions.rows + 1,)
            ):
                raise ValueError("branched action_ptr must be int32 [D+1]")
            ptr = self.action_ptr.to(torch.int64)
            if int(ptr[0]) != 0 or int(ptr[-1]) != self.actions.rows or bool(
                (ptr[1:] < ptr[:-1]).any()
            ):
                raise ValueError("branched action_ptr is not valid CSR")
        pools = MappingProxyType(dict(self.input_pools))
        for name, block in pools.items():
            ref_name = f"{name}_ref"
            if ref_name not in self.inputs.columns:
                raise ValueError(f"rollout input pool {name!r} lacks {ref_name!r}")
            refs = self.inputs.columns[ref_name]
            if refs.dtype != torch.int32 or refs.shape != (self.inputs.rows,):
                raise ValueError("rollout input-pool references must be int32 [D]")
            if refs.numel():
                torch._assert_async(
                    ((refs >= 0) & (refs < block.rows)).all(),
                    "rollout input-pool reference is out of range",
                )
        object.__setattr__(self, "input_pools", pools)


@dataclass(frozen=True)
class RolloutStorage:
    """Sealed pre-target store; topology and leaves have explicit lifetimes."""

    mode: str
    partitions: Mapping[int, PartitionRolloutBlock]
    topology: Any
    leaves: SealedTensorBlock
    seal_metrics: Mapping[str, SealMetrics] = field(default_factory=dict)
    schema_version: str = ROLLOUT_STORAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mode not in {"normal", "sequential_rb", "branched_rb"}:
            raise ValueError("unknown rollout storage mode")
        if self.schema_version != ROLLOUT_STORAGE_SCHEMA_VERSION:
            raise ValueError("unsupported rollout storage schema version")
        partitions = {int(key): value for key, value in self.partitions.items()}
        if any(key != value.partition_id for key, value in partitions.items()):
            raise ValueError("partition block keys must equal stable partition ids")
        object.__setattr__(self, "partitions", MappingProxyType(partitions))
        object.__setattr__(self, "seal_metrics", MappingProxyType(dict(self.seal_metrics)))


@dataclass(frozen=True)
class SealedDecisionInputs:
    """Exact natural decision rows emitted by a rollout scheduler."""

    node_ids: Tensor
    partition_id: Tensor
    partition_row: Tensor
    inputs: Mapping[int, SealedTensorBlock]
    input_pools: Mapping[int, Mapping[str, SealedTensorBlock]] = field(
        default_factory=dict
    )
    seal_metrics: Mapping[str, SealMetrics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = self.node_ids.numel()
        for name, value in (
            ("node_ids", self.node_ids),
            ("partition_id", self.partition_id),
            ("partition_row", self.partition_row),
        ):
            if value.dtype != torch.int32 or value.shape != (count,):
                raise ValueError(f"sealed decision {name} must be int32 [D]")
        inputs = MappingProxyType({int(key): value for key, value in self.inputs.items()})
        pools = MappingProxyType(
            {
                int(key): MappingProxyType(dict(value))
                for key, value in self.input_pools.items()
            }
        )
        for partition, block in inputs.items():
            selected = self.partition_id == partition
            rows = self.partition_row[selected].to(torch.int64)
            if rows.numel() != block.rows or (
                rows.numel()
                and not torch.equal(
                    rows.sort().values,
                    torch.arange(block.rows, device=rows.device),
                )
            ):
                raise ValueError("sealed decision partition rows are not exact")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "input_pools", pools)
        object.__setattr__(self, "seal_metrics", MappingProxyType(dict(self.seal_metrics)))

    def gather_partition_inputs(
        self, partition_id: int, partition_rows: Tensor
    ) -> Mapping[str, Tensor]:
        """Gather bounded natural rows and resolve their immutable pools."""

        partition_id = int(partition_id)
        block = self.inputs[partition_id]
        device = next(iter(block.columns.values())).device
        rows = partition_rows.to(device=device, dtype=torch.int64)
        gathered = {name: value[rows] for name, value in block.columns.items()}
        for name, pool in self.input_pools.get(partition_id, {}).items():
            refs = gathered.pop(f"{name}_ref").to(torch.int64)
            for field_name, value in pool.columns.items():
                resolved = f"{name}.{field_name}"
                if resolved in gathered:
                    raise ValueError(f"resolved input name collision: {resolved!r}")
                gathered[resolved] = value[refs]
        return MappingProxyType(gathered)


@dataclass(frozen=True)
class TrainingPartition:
    """One immutable aligned partition block consumed by optimization."""

    partition_id: int
    inputs: SealedTensorBlock
    columns: Mapping[str, Tensor]
    action_columns: Mapping[str, Tensor] | None = None
    input_pools: Mapping[str, SealedTensorBlock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        columns = MappingProxyType(dict(self.columns))
        object.__setattr__(self, "columns", columns)
        rows = self.inputs.rows
        for name, value in columns.items():
            if value.ndim == 0 or value.shape[0] != rows:
                raise ValueError(f"training decision column {name!r} is not row aligned")
            if value.requires_grad:
                raise ValueError("training storage tensors must be detached")
        if self.action_columns is not None:
            action_columns = MappingProxyType(dict(self.action_columns))
            action_rows = {
                int(value.shape[0])
                for name, value in action_columns.items()
                if name != "action_ptr"
                if value.ndim > 0
            }
            if len(action_rows) > 1 or any(
                value.ndim == 0 or value.requires_grad
                for value in action_columns.values()
            ):
                raise ValueError("training action columns must be aligned and detached")
            if "action_ptr" in action_columns:
                ptr = action_columns["action_ptr"]
                count = next(iter(action_rows), 0)
                if (
                    ptr.dtype != torch.int32
                    or ptr.shape != (rows + 1,)
                    or int(ptr[0]) != 0
                    or int(ptr[-1]) != count
                    or bool((ptr[1:] < ptr[:-1]).any())
                ):
                    raise ValueError("training action_ptr is not valid CSR")
            object.__setattr__(self, "action_columns", action_columns)
        pools = MappingProxyType(dict(self.input_pools))
        for name, block in pools.items():
            if not name or not isinstance(block, SealedTensorBlock):
                raise TypeError("input pools require nonempty names and sealed blocks")
            ref_name = f"{name}_ref"
            if ref_name not in self.inputs.columns:
                raise ValueError(f"input pool {name!r} has no aligned {ref_name!r}")
            refs = self.inputs.columns[ref_name]
            if refs.dtype != torch.int32 or refs.shape != (rows,):
                raise ValueError(f"input pool reference {ref_name!r} must be int32 [D]")
            if refs.numel():
                torch._assert_async(
                    ((refs >= 0) & (refs < block.rows)).all(),
                    f"input pool reference {ref_name!r} is out of range",
                )
        object.__setattr__(self, "input_pools", pools)

    def gather_inputs(self, rows: Tensor) -> Mapping[str, Tensor]:
        """Gather one bounded subspan and resolve exact immutable input pools."""

        rows = rows.to(
            dtype=torch.int64,
            device=next(iter(self.inputs.columns.values())).device,
        )
        gathered = {name: value[rows] for name, value in self.inputs.columns.items()}
        for name, block in self.input_pools.items():
            refs = gathered.pop(f"{name}_ref").to(torch.int64)
            for field_name, value in block.columns.items():
                resolved = f"{name}.{field_name}"
                if resolved in gathered:
                    raise ValueError(f"resolved input name collision: {resolved!r}")
                gathered[resolved] = value[refs]
        return MappingProxyType(gathered)


@dataclass(frozen=True)
class SequentialPosteriorPool:
    """Deduplicated exact posterior rows and trajectory/player addresses."""

    posterior: Tensor
    trajectory: Tensor
    player: Tensor

    def __post_init__(self) -> None:
        count = self.posterior.shape[0]
        if self.posterior.ndim != 2 or self.posterior.dtype != torch.float32:
            raise ValueError("posterior pool must be fp32 [U,X]")
        if self.trajectory.shape != (count,) or self.player.shape != (count,):
            raise ValueError("posterior pool addresses must align with rows")
        if self.trajectory.dtype != torch.int32 or self.player.dtype != torch.int32:
            raise ValueError("posterior pool addresses must be int32")
        pairs = torch.stack((self.trajectory, self.player), 1)
        if count and pairs.unique(dim=0).shape[0] != count:
            raise ValueError("posterior pool contains a duplicate trajectory/player pair")


@dataclass(frozen=True)
class SamplingMetadata:
    log_z: Mapping[int, Tensor]
    alias_probability: Mapping[int, Tensor] = field(default_factory=dict)
    alias_index: Mapping[int, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_z", MappingProxyType(dict(self.log_z)))
        object.__setattr__(
            self, "alias_probability", MappingProxyType(dict(self.alias_probability))
        )
        object.__setattr__(self, "alias_index", MappingProxyType(dict(self.alias_index)))
        for value in self.log_z.values():
            if value.numel() != 1 or value.dtype != torch.float64:
                raise ValueError("each partition log Z must be one fp64 scalar")
            if bool(torch.isnan(value)) or bool(torch.isposinf(value)):
                raise ValueError("partition log Z must be finite or negative infinity")
        if set(self.alias_probability) != set(self.alias_index):
            raise ValueError("alias probability/index tables must have identical keys")
        for key in self.alias_probability:
            probability = self.alias_probability[key]
            alias = self.alias_index[key]
            if (
                probability.ndim != 1
                or probability.dtype != torch.float64
                or alias.shape != probability.shape
                or alias.dtype != torch.int32
            ):
                raise ValueError("Walker alias tables must be fp64/int32 aligned vectors")


@dataclass(frozen=True)
class TrainingStorage:
    """Only immutable natural inputs, exact targets, and sampling metadata."""

    mode: str
    partitions: Mapping[int, TrainingPartition]
    sampling: SamplingMetadata
    posterior_pool: SequentialPosteriorPool | None = None
    schema_version: str = TRAINING_STORAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "partitions",
            MappingProxyType({int(key): value for key, value in self.partitions.items()}),
        )
        if self.schema_version != TRAINING_STORAGE_SCHEMA_VERSION:
            raise ValueError("unsupported training storage schema version")
        if self.mode == "sequential_rb" and self.posterior_pool is None:
            raise ValueError("sequential RB training storage requires a posterior pool")
        if self.mode != "sequential_rb" and self.posterior_pool is not None:
            raise ValueError("only sequential RB owns a posterior pool")
        for partition in self.partitions.values():
            columns = partition.columns
            rows = partition.inputs.rows
            event_count = int(columns["legal_event_ids"].shape[1])
            model_width = (
                1 if self.mode == "normal" else int(columns["old_value"].shape[1])
            )
            base = rollout_decision_schema(
                self.mode,
                model_width=model_width,
                event_count=event_count,
                dynamic_heads=False,
            )
            base.validate({name: columns[name] for name in base.by_name}, rows)
            exact_width = (
                1
                if self.mode == "normal"
                else int(
                    columns[
                        "lambda_return"
                        if self.mode == "sequential_rb"
                        else "value_hat"
                    ].shape[1]
                )
            )
            dynamic_targets = "trace_variance_hat" in columns
            target = target_decision_schema(
                self.mode,
                exact_width=exact_width,
                dynamic_heads=dynamic_targets,
            )
            if set(columns) != set(base.by_name) | set(target.by_name):
                raise ValueError("training decision columns do not match declared schemas")
            target.validate({name: columns[name] for name in target.by_name}, rows)
            if self.mode == "branched_rb":
                if partition.action_columns is None:
                    raise ValueError("branched training partition omitted ragged actions")
                actions = partition.action_columns
                action_base = rollout_action_schema(model_width)
                action_rows = int(actions["event"].shape[0])
                action_base.validate(
                    {name: actions[name] for name in action_base.by_name}, action_rows
                )
                action_target = target_action_schema(exact_width)
                action_target.validate(
                    {name: actions[name] for name in action_target.by_name}, action_rows
                )
                if set(actions) != (
                    set(action_base.by_name) | set(action_target.by_name) | {"action_ptr"}
                ):
                    raise ValueError("training action columns do not match declared schemas")

    def tensor_storage_bytes(self) -> int:
        """Return unique retained tensor storage bytes (views count once)."""

        from pbg.core.memory_control import unique_tensor_bytes

        return unique_tensor_bytes(self)


def occurrence_schema(*, n_players: int, mode: str) -> TensorSchema:
    """Normative compact occurrence source schema for one rollout mode."""

    if n_players <= 0:
        raise ValueError("n_players must be positive")
    fields = [
        TensorFieldSchema("parent", torch.int32, fill=-1, alignment=4),
        TensorFieldSchema("state_ref", torch.int32, fill=-1, alignment=4),
        TensorFieldSchema("owner", torch.int32, fill=-1, alignment=4),
        TensorFieldSchema("status", torch.uint8),
        TensorFieldSchema("rng_root", torch.int64, alignment=8),
        TensorFieldSchema("rng_path", torch.int64, alignment=8),
        TensorFieldSchema("child_begin", torch.int32, fill=-1, alignment=4),
        TensorFieldSchema("degree", torch.int32, alignment=4),
        TensorFieldSchema("decision_partition", torch.int32, fill=-1, alignment=4),
        TensorFieldSchema("decision_row", torch.int32, fill=-1, alignment=4),
        TensorFieldSchema("leaf_row", torch.int32, fill=-1, alignment=4),
    ]
    if mode != "branched_rb":
        fields.extend(
            (
                TensorFieldSchema("root_id", torch.int32, fill=-1, alignment=4),
                TensorFieldSchema("decision_ordinal", torch.int32, fill=-1, alignment=4),
            )
        )
    else:
        fields.extend(
            (
                TensorFieldSchema("log_weight", torch.float32, alignment=4),
                TensorFieldSchema("nominal_count", torch.int32, alignment=4),
                TensorFieldSchema("executed_count", torch.int32, alignment=4),
                TensorFieldSchema("log_player_sum", torch.float32, alignment=4),
                TensorFieldSchema("log_ancestor_product", torch.float32, alignment=4),
                TensorFieldSchema("queue_cost", torch.float32, alignment=4),
            )
        )
    return TensorSchema(
        "occurrence",
        tuple(fields),
        chunk_header_bytes=64,
        allocator_slack_bytes=64,
    )


def topology_schema(mode: str) -> TensorSchema:
    if mode == "branched_rb":
        names = (
            "parent",
            "child_begin",
            "degree",
            "decision_partition",
            "decision_row",
            "leaf_row",
        )
    elif mode in {"normal", "sequential_rb"}:
        names = ("decision_partition", "decision_row")
    else:
        raise ValueError("unknown topology mode")
    return TensorSchema(
        f"{mode}_topology",
        tuple(TensorFieldSchema(name, torch.int32, fill=-1, alignment=4) for name in names),
    )


def rollout_policy_source_schema(
    mode: str, *, exact_width: int, dynamic_heads: bool = True
) -> TensorSchema:
    """Paged scheduler policy rows before partition-aware sealing."""

    fields = [TensorFieldSchema("node_id", torch.int32, alignment=4)]
    if mode in {"normal", "sequential_rb"}:
        fields.extend(
            (
                TensorFieldSchema(
                    "old_logp", torch.float32, (exact_width,), alignment=16
                ),
                TensorFieldSchema(
                    "old_v", torch.float32, (exact_width,), alignment=16
                ),
            )
        )
    elif mode == "branched_rb":
        fields.append(
            TensorFieldSchema("old_v", torch.float32, (exact_width,), alignment=16)
        )
        if dynamic_heads:
            fields.extend(
                (
                    TensorFieldSchema("old_t", torch.float32, alignment=4),
                    TensorFieldSchema("old_c", torch.float32, alignment=4),
                )
            )
    else:
        raise ValueError("unknown rollout mode")
    return TensorSchema("rollout_policy_source", tuple(fields))


def rollout_decision_schema(
    mode: str,
    *,
    model_width: int,
    event_count: int,
    dynamic_heads: bool = True,
) -> TensorSchema:
    """Exact retained decision columns on one partition's natural width."""

    rb = mode != "normal"
    fields = [
        TensorFieldSchema("owner", torch.int32, alignment=4),
        TensorFieldSchema(
            "old_value",
            torch.float32,
            (() if not rb else (model_width,)),
            alignment=16,
        ),
        TensorFieldSchema(
            "legal_event_ids", torch.int32, (event_count,), alignment=16
        ),
        TensorFieldSchema("legal_event_mask", torch.bool, (event_count,)),
    ]
    if mode in {"normal", "sequential_rb"}:
        fields.extend(
            (
                TensorFieldSchema("event", torch.int32, alignment=4),
                TensorFieldSchema(
                    "old_log_prob",
                    torch.float32,
                    (() if not rb else (model_width,)),
                    alignment=16,
                ),
            )
        )
        if mode == "normal":
            fields.append(
                TensorFieldSchema("realized_type", torch.int32, alignment=4)
            )
    elif mode == "branched_rb":
        fields.extend(
            (
                TensorFieldSchema("log_weight", torch.float64, alignment=8),
                TensorFieldSchema("nominal_child_count", torch.int32, alignment=4),
                TensorFieldSchema("executed_child_count", torch.int32, alignment=4),
            )
        )
        if dynamic_heads:
            fields.extend(
                (
                    TensorFieldSchema(
                        "frozen_trace_variance", torch.float32, alignment=4
                    ),
                    TensorFieldSchema("frozen_cost", torch.float32, alignment=4),
                )
            )
    else:
        raise ValueError("unknown rollout mode")
    return TensorSchema(f"{mode}_decision", tuple(fields))


def rollout_action_schema(model_width: int) -> TensorSchema:
    """Exact ragged branched action rows."""

    return TensorSchema(
        "branched_rb_action",
        (
            TensorFieldSchema("event", torch.int32, alignment=4),
            TensorFieldSchema(
                "old_log_prob", torch.float32, (model_width,), alignment=16
            ),
        ),
    )


def rollout_action_source_schema(exact_width: int) -> TensorSchema:
    """Paged scheduler edge-policy rows before ragged action sealing."""

    return TensorSchema(
        "branched_rb_action_source",
        (
            TensorFieldSchema("edge_id", torch.int64, alignment=8),
            TensorFieldSchema(
                "old_logp", torch.float32, (exact_width,), alignment=16
            ),
        ),
    )


def stopped_leaf_source_schema(mode: str, *, players: int, exact_width: int) -> TensorSchema:
    """Paged last-consumer leaf capture, including its sole log-reach owner."""

    fields = [
        TensorFieldSchema("node_id", torch.int32, alignment=4),
        TensorFieldSchema(
            "log_reach", torch.float32, (players, exact_width), alignment=16
        ),
    ]
    if mode == "normal":
        fields.append(
            TensorFieldSchema("reward", torch.float32, (players,), alignment=16)
        )
    elif mode in {"sequential_rb", "branched_rb"}:
        fields.extend(_rb_completion_public_fields(players))
    else:
        raise ValueError("unknown rollout mode")
    return TensorSchema(f"{mode}_stopped_leaf_source", tuple(fields))


def _rb_completion_public_fields(players: int) -> tuple[TensorFieldSchema, ...]:
    return (
        TensorFieldSchema("owner", torch.int32, alignment=4),
        TensorFieldSchema("fold_payoff", torch.float32, (players,), alignment=16),
        TensorFieldSchema("board_cards", torch.uint8, (5,)),
        TensorFieldSchema("stratum_level", torch.int32, (players,), alignment=16),
        TensorFieldSchema(
            "payoff_baseline", torch.float32, (players,), alignment=16
        ),
        TensorFieldSchema("stratum_amount", torch.float32, (players,), alignment=16),
    )


def rollout_leaf_schema(mode: str, *, players: int, exact_width: int) -> TensorSchema:
    """Exact post-seal leaf fields; RB reach is represented once as weights."""

    if mode == "normal":
        fields = (
            TensorFieldSchema("reward", torch.float32, (players,), alignment=16),
        )
    elif mode in {"sequential_rb", "branched_rb"}:
        fields = (
            *_rb_completion_public_fields(players),
            TensorFieldSchema("completion_key", torch.int64, alignment=8),
            TensorFieldSchema(
                "range_weights",
                torch.float32,
                (players, exact_width),
                alignment=16,
            ),
        )
    else:
        raise ValueError("unknown rollout mode")
    return TensorSchema(f"{mode}_leaf", fields)


def target_decision_schema(
    mode: str, *, exact_width: int, dynamic_heads: bool = True
) -> TensorSchema:
    """Target-only decision columns appended to immutable training storage."""

    if mode == "normal":
        fields = (TensorFieldSchema("lambda_return", torch.float32),)
    elif mode == "sequential_rb":
        fields = (
            TensorFieldSchema("posterior_ref", torch.int32, alignment=4),
            TensorFieldSchema(
                "lambda_return", torch.float32, (exact_width,), alignment=16
            ),
        )
    elif mode == "branched_rb":
        fields = [
            TensorFieldSchema(
                "value_hat", torch.float32, (exact_width,), alignment=16
            )
        ]
        if dynamic_heads:
            fields.extend(
                (
                    TensorFieldSchema("trace_variance_hat", torch.float32),
                    TensorFieldSchema("trace_variance_mask", torch.bool),
                    TensorFieldSchema("cost_hat", torch.float32),
                )
            )
        fields = tuple(fields)
    else:
        raise ValueError("unknown rollout mode")
    return TensorSchema(f"{mode}_decision_target", fields)


def target_action_schema(exact_width: int) -> TensorSchema:
    return TensorSchema(
        "branched_rb_action_target",
        (
            TensorFieldSchema(
                "child_posterior", torch.float32, (exact_width,), alignment=16
            ),
            TensorFieldSchema(
                "advantage_hat", torch.float32, (exact_width,), alignment=16
            ),
        ),
    )


def sequential_posterior_schema(exact_width: int) -> TensorSchema:
    return TensorSchema(
        "sequential_posterior_pool",
        (
            TensorFieldSchema(
                "posterior", torch.float32, (exact_width,), alignment=16
            ),
            TensorFieldSchema("trajectory", torch.int32, alignment=4),
            TensorFieldSchema("player", torch.int32, alignment=4),
        ),
    )
