"""Device-indexed cache for evaluator-owned folded-player summaries."""

from __future__ import annotations

import torch
from torch import Tensor

from pbg.core.storage import (
    ByteAccountant,
    PagedTensorArena,
    TensorFieldSchema,
    TensorSchema,
)
from pbg.games.poker.folded import PartitionSummaryBatch


_FOLDER_FIELDS = frozenset(
    {
        "folder_log_mass",
        "folder_card_mass",
        "folder_mask",
        "folder_line_meta",
        "folder_seat",
    }
)


def poker_folded_summary_schema(
    *,
    n_cards: int,
    beta_rank: int,
    max_folders: int,
    line_dim: int = 8,
    include_anchor: bool = True,
    name: str = "poker_folded_summary",
) -> TensorSchema:
    """Return the sole schema for sparse evaluator summary anchors."""

    if min(n_cards, beta_rank, max_folders, line_dim) <= 0:
        raise ValueError("folded-summary schema dimensions must be positive")
    hands = n_cards * (n_cards - 1) // 2
    fields = [TensorFieldSchema("anchor", torch.int64, alignment=8)] if include_anchor else []
    fields.extend(
        (
            TensorFieldSchema("hand_log_ratio", torch.float32, (hands,), alignment=16),
            TensorFieldSchema("beta_eigenvalue", torch.float32, (beta_rank,), alignment=16),
            TensorFieldSchema(
                "beta_eigenvector", torch.float32, (n_cards, beta_rank), alignment=16
            ),
            TensorFieldSchema(
                "folder_log_mass", torch.float32, (max_folders,), alignment=16
            ),
            TensorFieldSchema(
                "folder_card_mass",
                torch.float32,
                (max_folders, n_cards),
                alignment=16,
            ),
            TensorFieldSchema("tight_cluster_scalar", torch.float32, alignment=4),
            TensorFieldSchema("folder_mask", torch.bool, (max_folders,)),
            TensorFieldSchema(
                "folder_line_meta",
                torch.float32,
                (max_folders, line_dim),
                alignment=16,
            ),
            TensorFieldSchema(
                "folder_seat", torch.int64, (max_folders,), alignment=16
            ),
        )
    )
    return TensorSchema(
        name,
        tuple(fields),
        chunk_header_bytes=64 if include_anchor else 0,
        allocator_slack_bytes=64 if include_anchor else 0,
    )


class DevicePartitionSummaryCache:
    """Sparse page-backed summaries indexed by dependency-anchor handles.

    Only realized distinct anchors own rows. Lookup scans admitted key pages;
    there is no validity vector or summary payload shaped by rollout capacity.
    Folder dimensions use fixed per-row maximum storage and are sliced to the
    host-known natural partition on return.
    """

    def __init__(
        self,
        max_folders: int,
        device: torch.device | str,
        *,
        n_cards: int,
        beta_rank: int,
        line_dim: int = 8,
        page_rows: int = 64,
        accountant: ByteAccountant | None = None,
        label: str = "poker_folded_summary",
    ) -> None:
        if max_folders < 0:
            raise ValueError("partition cache dimensions are invalid")
        self.max_folders = int(max_folders)
        self.device = torch.device(device)
        self.page_rows = int(page_rows)
        self.accountant = accountant
        self.label = str(label)
        if not self.label:
            raise ValueError("partition cache label must be nonempty")
        self.schema = (
            None
            if self.max_folders == 0
            else poker_folded_summary_schema(
                n_cards=n_cards,
                beta_rank=beta_rank,
                max_folders=self.max_folders,
                line_dim=line_dim,
            )
        )
        self.arena: PagedTensorArena | None = None
        self._field_names: frozenset[str] | None = None
        self._metadata: dict[str, object] = {}

    @property
    def capacity(self) -> int:
        """Return admitted sparse rows, not a configured rollout maximum."""

        return 0 if self.arena is None else self.arena.capacity_rows

    @property
    def size(self) -> int:
        return 0 if self.arena is None else self.arena.size

    @property
    def allocated_capacity_bytes(self) -> int:
        return 0 if self.arena is None else self.arena.allocated_capacity_bytes

    def reset(self, *, release_pages: bool = False) -> None:
        """Invalidate anchors and optionally release admitted sparse pages."""

        if self.arena is not None:
            self.arena.reset(release_pages=release_pages)
        self._metadata.clear()

    def contains(self, keys: Tensor) -> Tensor:
        """Return device-local cache-hit flags for int state keys ``[B]``."""

        return self._lookup(keys) >= 0

    def lookup_rows(self, keys: Tensor) -> Tensor:
        """Return stable owned row ids, or ``-1`` for absent anchors."""

        return self._lookup(keys)

    def _lookup(self, keys: Tensor) -> Tensor:
        keys = keys.to(device=self.device, dtype=torch.int64)
        if keys.ndim != 1:
            raise ValueError("partition cache keys must be one-dimensional")
        if self.arena is None:
            return torch.full(
                keys.shape, -1, dtype=torch.int64, device=self.device
            )
        return self.arena.lookup_rows("anchor", keys)

    def _ensure_arena(self, computed: PartitionSummaryBatch) -> PagedTensorArena:
        names = frozenset(computed.fields)
        if self._field_names is not None:
            if names != self._field_names:
                raise ValueError(
                    "partition summary field schema changed within one rollout"
                )
            assert self.arena is not None
            return self.arena
        if self.max_folders == 0:
            raise RuntimeError("an all-live game cannot allocate folded summaries")
        assert self.schema is not None
        expected = {field.name: field for field in self.schema.fields if field.name != "anchor"}
        if names != set(expected):
            raise ValueError("partition summary fields do not match the declared schema")
        for name, value in computed.fields.items():
            spec = expected[name]
            shape_ok = (
                value.ndim >= 2
                and value.shape[1] <= self.max_folders
                and tuple(value.shape[2:]) == spec.shape[1:]
                if name in _FOLDER_FIELDS
                else tuple(value.shape[1:]) == spec.shape
            )
            if not shape_ok or value.dtype != spec.dtype or value.device != self.device:
                raise ValueError(f"partition field {name} violates the declared schema")
        self._field_names = names
        self.arena = PagedTensorArena(
            self.schema,
            self.page_rows,
            self.device,
            accountant=self.accountant,
            label=self.label,
        )
        return self.arena

    def _storage_columns(
        self,
        computed: PartitionSummaryBatch,
        source_rows: Tensor,
        anchors: Tensor,
        folder_count: int,
    ) -> dict[str, Tensor]:
        columns: dict[str, Tensor] = {"anchor": anchors.to(torch.int64)}
        for name, value in computed.fields.items():
            selected = value[source_rows]
            if name in _FOLDER_FIELDS:
                padded = torch.empty(
                    (source_rows.numel(), self.max_folders, *value.shape[2:]),
                    dtype=value.dtype,
                    device=self.device,
                )
                if folder_count:
                    padded[:, :folder_count] = selected
                columns[name] = padded
            else:
                columns[name] = selected
        return columns

    def merge(
        self,
        *,
        keys: Tensor,
        computed: PartitionSummaryBatch,
        compute_mask: Tensor,
        folder_count: int,
    ) -> PartitionSummaryBatch:
        """Store missing rows and return cached/new fields in request order."""

        if not 0 <= folder_count <= self.max_folders:
            raise ValueError("folder_count is outside the cache geometry")
        keys = keys.to(device=self.device, dtype=torch.int64)
        if compute_mask.shape != keys.shape or compute_mask.dtype != torch.bool:
            raise TypeError("compute_mask must be bool with the cache-key shape")
        for name, value in computed.fields.items():
            if value.shape[0] != keys.numel() or value.device != self.device:
                raise ValueError(f"partition field {name} has the wrong batch or device")
            if name in _FOLDER_FIELDS and value.shape[1] != folder_count:
                raise ValueError(f"partition field {name} changed folder geometry")
        arena = self._ensure_arena(computed)
        rows = self._lookup(keys)
        missing = rows < 0
        torch._assert_async(
            (missing == compute_mask).all(),
            "partition cache compute mask disagrees with sparse lookup",
        )
        missing_positions = missing.nonzero(as_tuple=False).squeeze(1)
        missing_keys = keys[missing_positions]
        unique_keys, inverse = torch.unique(
            missing_keys, sorted=True, return_inverse=True
        )
        first = torch.full(
            (unique_keys.numel(),),
            keys.numel(),
            dtype=torch.int64,
            device=self.device,
        )
        first.scatter_reduce_(
            0, inverse, missing_positions, reduce="amin", include_self=True
        )
        new_rows = arena.append(
            self._storage_columns(
                computed, first, unique_keys, folder_count
            )
        )
        rows[missing_positions] = new_rows[inverse]

        output = {
            name: (
                arena.gather(name, rows)[:, :folder_count]
                if name in _FOLDER_FIELDS
                else arena.gather(name, rows)
            )
            for name in computed.fields
        }
        self._metadata.update(computed.metadata)
        return PartitionSummaryBatch(
            fields=output,
            metadata={
                **self._metadata,
                "cache": "device_indexed_street_or_fold",
                "cache_masking": "evaluator_row_mask",
            },
        )

    def gather(self, keys: Tensor, *, folder_count: int) -> PartitionSummaryBatch:
        """Gather already-computed dependency anchors without reevaluation."""

        if not 0 <= folder_count <= self.max_folders:
            raise ValueError("folder_count is outside the cache geometry")
        keys = keys.to(device=self.device, dtype=torch.int64)
        rows = self._lookup(keys)
        if keys.numel():
            torch._assert_async(
                (rows >= 0).all(),
                "one or more folded-summary anchors are not cached",
            )
        if self.arena is None:
            raise KeyError("folded-summary cache is empty")
        fields = {
            name: (
                self.arena.gather(name, rows)[:, :folder_count]
                if name in _FOLDER_FIELDS
                else self.arena.gather(name, rows)
            )
            for name in self._field_names or ()
        }
        return PartitionSummaryBatch(
            fields=fields,
            metadata={**self._metadata, "cache": "sealed_sparse_anchor_source"},
        )

    @staticmethod
    def storage_nbytes(
        capacity: int,
        *,
        n_cards: int,
        beta_rank: int,
        max_folders: int,
        line_dim: int = 8,
        include_anchor: bool = False,
    ) -> int:
        """Return schema capacity bytes for ``capacity`` sparse rows.

        Sealed pools omit the anchor field. Mutable rollout caches set
        ``include_anchor`` and admit this amount one page at a time.
        """

        if min(capacity, n_cards, beta_rank, max_folders, line_dim) < 0:
            raise ValueError("partition cache dimensions must be nonnegative")
        if capacity == 0 or max_folders == 0:
            return 0
        schema = poker_folded_summary_schema(
            n_cards=n_cards,
            beta_rank=beta_rank,
            max_folders=max_folders,
            line_dim=line_dim,
            include_anchor=include_anchor,
        )
        return schema.capacity_bytes(capacity, chunks=1)
