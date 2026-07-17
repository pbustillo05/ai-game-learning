"""Immutable game-neutral partition and completion boundaries.

Core rollout code routes by integer partition id and consumes only these
contracts.  Diagnostic names are deliberately opaque: no scheduler or target
builder is allowed to infer game semantics from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor


class NaturalForward(Protocol):
    """Curried model boundary for one natural partition."""

    def __call__(self, inputs: Mapping[str, Tensor]) -> Any: ...


class CompletionForward(Protocol):
    """Opaque bounded leaf-completion boundary."""

    def __call__(self, leaves: Mapping[str, Tensor], *, rows: Tensor) -> Any: ...


def _freeze_mapping(value: Mapping[str, int]) -> Mapping[str, int]:
    result = {str(name): int(size) for name, size in value.items()}
    if any(not name or size < 0 for name, size in result.items()):
        raise ValueError("byte-contract names must be nonempty and values nonnegative")
    return MappingProxyType(result)


@dataclass(frozen=True)
class ByteWorkContract:
    """Worst-case retained/transient contract for one partition.

    Byte terms are named so phase ledgers can identify the limiting field.
    ``max_rows`` is a hard physical-pack limit; ``batch_quanta`` describes
    legal pack granularities without imposing a universal alignment rule.
    """

    retained_per_row: Mapping[str, int] = field(default_factory=dict)
    transient_per_pack: Mapping[str, int] = field(default_factory=dict)
    max_rows: int = 1
    batch_quanta: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retained_per_row", _freeze_mapping(self.retained_per_row))
        object.__setattr__(
            self, "transient_per_pack", _freeze_mapping(self.transient_per_pack)
        )
        if isinstance(self.max_rows, bool) or self.max_rows <= 0:
            raise ValueError("max_rows must be a positive integer")
        quanta = tuple(int(value) for value in self.batch_quanta)
        if not quanta or any(value <= 0 for value in quanta):
            raise ValueError("batch_quanta must contain positive integers")
        if tuple(sorted(set(quanta))) != quanta:
            raise ValueError("batch_quanta must be strictly increasing")
        if quanta[0] != 1:
            raise ValueError("batch_quanta must include singleton execution")
        object.__setattr__(self, "batch_quanta", quanta)

    @property
    def retained_row_bytes(self) -> int:
        return sum(self.retained_per_row.values())

    @property
    def transient_pack_bytes(self) -> int:
        return sum(self.transient_per_pack.values())


@dataclass(frozen=True)
class CompletionSpec:
    """Immutable declaration of a game's stopped-leaf completion service."""

    leaf_fields: tuple[str, ...]
    exact_reach_order: tuple[str, ...]
    output_fields: tuple[str, ...]
    complete: CompletionForward
    workspace_bytes: int | Callable[[int], int]
    key_fields: tuple[str, ...] = ("completion_key",)

    def __post_init__(self) -> None:
        for label, values in (
            ("leaf_fields", self.leaf_fields),
            ("exact_reach_order", self.exact_reach_order),
            ("output_fields", self.output_fields),
            ("key_fields", self.key_fields),
        ):
            frozen = tuple(str(value) for value in values)
            if (
                not frozen
                or any(not value for value in frozen)
                or len(set(frozen)) != len(frozen)
            ):
                raise ValueError(f"{label} must contain unique nonempty names")
            object.__setattr__(self, label, frozen)
        if not set(self.key_fields).issubset(self.leaf_fields):
            raise ValueError("completion key fields must be owned by the leaf schema")
        if not callable(self.complete):
            raise TypeError("completion boundary must be callable")
        if not callable(self.workspace_bytes):
            if isinstance(self.workspace_bytes, bool) or self.workspace_bytes < 0:
                raise ValueError("completion workspace must be nonnegative")

    def workspace_bound(self, rows: int) -> int:
        if isinstance(rows, bool) or rows < 0:
            raise ValueError("completion rows must be nonnegative")
        value = self.workspace_bytes(rows) if callable(self.workspace_bytes) else self.workspace_bytes
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("completion workspace callback returned an invalid byte count")
        return value


@dataclass(frozen=True)
class PartitionSpec:
    """One stable exact/model domain and its natural model boundary."""

    partition_id: int
    name: str
    router: Callable[[Any], int | Tensor]
    exact_width: int
    model_width: int
    exact_to_model: tuple[int, ...]
    forward: NaturalForward
    byte_work: ByteWorkContract
    input_schema: Any
    preorder: int | None = None
    completion: CompletionSpec | None = None

    def __post_init__(self) -> None:
        if isinstance(self.partition_id, bool) or self.partition_id < 0:
            raise ValueError("partition_id must be a nonnegative integer")
        if not self.name:
            raise ValueError("partition name must be nonempty")
        if not callable(self.router) or not callable(self.forward):
            raise TypeError("partition router and forward boundary must be callable")
        if any(
            isinstance(value, bool) or value <= 0
            for value in (self.exact_width, self.model_width)
        ):
            raise ValueError("partition widths must be positive integers")
        mapping = tuple(int(value) for value in self.exact_to_model)
        if len(mapping) != self.exact_width:
            raise ValueError("exact_to_model must have one entry per exact state")
        if any(value < 0 or value >= self.model_width for value in mapping):
            raise ValueError("exact_to_model contains an out-of-domain model row")
        object.__setattr__(self, "exact_to_model", mapping)
        if self.preorder is not None and (
            isinstance(self.preorder, bool) or self.preorder < 0
        ):
            raise ValueError("partition preorder must be nonnegative or None")

    @property
    def m(self) -> int:
        return self.partition_id

    @property
    def X_m(self) -> int:
        return self.exact_width

    @property
    def Y_m(self) -> int:
        return self.model_width

    def phi(self, *, device: torch.device | str | None = None) -> Tensor:
        return torch.tensor(self.exact_to_model, dtype=torch.int64, device=device)

    def assert_model_width(self, value: Tensor, *, dimension: int = -1) -> None:
        if value.ndim == 0 or value.shape[dimension] != self.model_width:
            raise ValueError(
                f"partition {self.partition_id} ({self.name}) requires natural "
                f"model width {self.model_width}, got shape {tuple(value.shape)}"
            )

    def assert_exact_width(self, value: Tensor, *, dimension: int = -1) -> None:
        if value.ndim == 0 or value.shape[dimension] != self.exact_width:
            raise ValueError(
                f"partition {self.partition_id} ({self.name}) requires exact "
                f"width {self.exact_width}, got shape {tuple(value.shape)}"
            )


@dataclass(frozen=True)
class PartitionSet:
    """Validated immutable collection indexed only by stable integer ids."""

    specs: tuple[PartitionSpec, ...]

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        if not specs:
            raise ValueError("at least one partition is required")
        ids = tuple(spec.partition_id for spec in specs)
        if len(set(ids)) != len(ids):
            raise ValueError("partition ids must be unique")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(
            self, "_by_id", MappingProxyType({spec.partition_id: spec for spec in specs})
        )

    def __getitem__(self, partition_id: int) -> PartitionSpec:
        try:
            return self._by_id[int(partition_id)]
        except KeyError as exc:
            raise KeyError(f"unknown partition id {partition_id}") from exc

    def route(self, public_state: Any) -> int | Tensor:
        """Route through the common pure router and validate scalar ids.

        Games normally install the same router callable on every spec.  Keeping
        it on the spec makes the interface self-contained while this method
        supplies the collection-level convenience boundary.
        """

        routed = self.specs[0].router(public_state)
        if isinstance(routed, Tensor):
            return routed
        result = int(routed)
        self[result]
        return result

    def scheduler_order(self, eligible: Sequence[int]) -> tuple[int, ...]:
        """Return ids ordered by declared preorder and stable collection order."""

        positions = {spec.partition_id: index for index, spec in enumerate(self.specs)}
        return tuple(
            sorted(
                (int(value) for value in eligible),
                key=lambda value: (
                    math.inf if self[value].preorder is None else self[value].preorder,
                    positions[value],
                ),
            )
        )
