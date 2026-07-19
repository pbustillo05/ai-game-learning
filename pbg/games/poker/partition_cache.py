"""Device-indexed cache for evaluator-owned folded-player summaries."""

from __future__ import annotations

import torch
from torch import Tensor

from pbg.games.poker.folded import PartitionSummaryBatch


_FOLDER_FIELDS = frozenset(
    {
        "folder_log_mass",
        "folder_single_card_log_ratio",
        "folder_card_marginal",
        "folder_card_mass",
        "folder_delta",
        "folder_output_valid",
        "folder_mask",
        "folder_line_meta",
        "folder_seat",
    }
)


class DevicePartitionSummaryCache:
    """Cache summary tensors by dependency-anchor state handle.

    The cache has no per-row Python keys or values. A device bool validity map
    selects cached versus newly evaluated rows, and every field is gathered in
    one indexed batch. Folder dimensions use fixed maximum storage and are
    sliced to the host-known execution family on return.
    """

    def __init__(
        self,
        capacity: int,
        max_folders: int,
        device: torch.device | str,
    ) -> None:
        if capacity <= 0 or max_folders < 0:
            raise ValueError("partition cache dimensions are invalid")
        self.capacity = int(capacity)
        self.max_folders = int(max_folders)
        self.device = torch.device(device)
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self._fields: dict[str, Tensor] = {}
        self._field_names: frozenset[str] | None = None
        self._metadata: dict[str, object] = {}

    def reset(self) -> None:
        """Invalidate all anchors without reallocating field storage."""

        self.valid.zero_()
        self._metadata.clear()

    def contains(self, keys: Tensor) -> Tensor:
        """Return device-local cache-hit flags for int state keys ``[B]``."""

        keys = keys.to(device=self.device, dtype=torch.int64)
        return self.valid[keys]

    @staticmethod
    def _broadcast_rows(mask: Tensor, value: Tensor) -> Tensor:
        return mask.reshape(mask.shape + (1,) * (value.ndim - 1))

    def _allocate_field(self, name: str, value: Tensor) -> Tensor:
        if name in _FOLDER_FIELDS:
            shape = (self.capacity, self.max_folders, *value.shape[2:])
        else:
            shape = (self.capacity, *value.shape[1:])
        return torch.empty(shape, dtype=value.dtype, device=self.device)

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
        names = frozenset(computed.fields)
        if self._field_names is None:
            self._field_names = names
        elif names != self._field_names:
            raise ValueError("partition summary field schema changed within one rollout")

        output: dict[str, Tensor] = {}
        for name, value in computed.fields.items():
            if value.shape[0] != keys.numel() or value.device != self.device:
                raise ValueError(f"partition field {name} has the wrong batch or device")
            storage = self._fields.get(name)
            if storage is None:
                storage = self._allocate_field(name, value)
                self._fields[name] = storage
            if name in _FOLDER_FIELDS:
                expected = (keys.numel(), folder_count, *storage.shape[2:])
                if value.shape != expected:
                    raise ValueError(f"partition field {name} changed folder geometry")
                current = storage[keys, :folder_count]
                selected = torch.where(
                    self._broadcast_rows(compute_mask, value), value, current
                )
                storage[keys, :folder_count] = selected
                output[name] = storage[keys, :folder_count]
            else:
                if value.shape[1:] != storage.shape[1:]:
                    raise ValueError(f"partition field {name} changed shape")
                current = storage[keys]
                selected = torch.where(
                    self._broadcast_rows(compute_mask, value), value, current
                )
                storage[keys] = selected
                output[name] = storage[keys]

        self.valid[keys] = torch.ones_like(keys, dtype=torch.bool)
        self._metadata.update(computed.metadata)
        return PartitionSummaryBatch(
            fields=output,
            metadata={
                **self._metadata,
                "cache": "device_indexed_street_or_fold",
                "cache_masking": "evaluator_row_mask",
            },
        )

    @staticmethod
    def storage_nbytes(
        capacity: int,
        *,
        n_cards: int,
        beta_rank: int,
        max_folders: int,
        line_dim: int = 8,
    ) -> int:
        """Return the conservative fully materialized production-cache bound."""

        if min(capacity, n_cards, beta_rank, max_folders, line_dim) < 0:
            raise ValueError("partition cache dimensions must be nonnegative")
        d, r, f = n_cards, beta_rank, max_folders
        h = d * (d - 1) // 2
        float_values = h + r + d * r + 1
        float_values += f + f * d + f * line_dim
        bool_values = 1 + f  # cache validity and folder mask
        int64_values = f
        return capacity * (4 * float_values + bool_values + 8 * int64_values)
