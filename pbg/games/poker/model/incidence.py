"""Generic gather, scatter, and row-permutation tensor helpers.

This module deliberately has no knowledge of poker rules.  Its maps are built
from boolean compatibility rows supplied by the representation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CompactMap:
    """A padded compact representation of fixed-width rows.

    ``compact_to_full`` and ``valid`` are ``[B,Q]``. ``full_to_compact`` is
    ``[B,H]`` and contains ``-1`` for rows omitted from the compact form.
    """

    compact_to_full: Tensor
    full_to_compact: Tensor
    valid: Tensor

    @property
    def batch_size(self) -> int:
        return self.compact_to_full.shape[0]

    @property
    def compact_size(self) -> int:
        return self.compact_to_full.shape[1]

    @property
    def full_size(self) -> int:
        return self.full_to_compact.shape[1]


def build_compact_map(compatibility: Tensor) -> CompactMap:
    """Build a stable, padded map from supplied bool rows ``[B,H]``."""

    if not isinstance(compatibility, Tensor) or compatibility.ndim != 2:
        raise ValueError("compatibility must have shape [B,H]")
    if compatibility.dtype != torch.bool:
        raise TypeError("compatibility must have dtype torch.bool")
    batch, full_size = compatibility.shape
    counts = compatibility.sum(dim=1)
    compact_size = int(counts.max().item()) if batch else 0
    compact_to_full = torch.zeros(
        (batch, compact_size), dtype=torch.long, device=compatibility.device
    )
    valid = torch.zeros_like(compact_to_full, dtype=torch.bool)
    full_to_compact = torch.full(
        (batch, full_size), -1, dtype=torch.long, device=compatibility.device
    )
    full_ids = torch.arange(full_size, device=compatibility.device)
    for row in range(batch):
        selected = full_ids[compatibility[row]]
        count = selected.numel()
        compact_to_full[row, :count] = selected
        valid[row, :count] = True
        full_to_compact[row, selected] = torch.arange(count, device=compatibility.device)
    return CompactMap(compact_to_full, full_to_compact, valid)


def compact_gather(values: Tensor, mapping: CompactMap, dim: int = 1) -> Tensor:
    """Gather the full-row dimension of ``values`` into padded compact rows.

    The batch dimension must be zero.  Padded compact entries are returned as
    zero, independently of the value at full row zero.
    """

    if not isinstance(values, Tensor) or values.ndim < 2:
        raise ValueError("values must have a batch and full-row dimension")
    dim = dim % values.ndim
    if dim == 0:
        raise ValueError("the batch dimension cannot be gathered")
    if values.shape[0] != mapping.batch_size or values.shape[dim] != mapping.full_size:
        raise ValueError("values do not match the compact map")
    order = [0, dim] + [axis for axis in range(1, values.ndim) if axis != dim]
    inverse = [order.index(axis) for axis in range(values.ndim)]
    moved = values.permute(order)
    suffix = moved.shape[2:]
    index = mapping.compact_to_full.reshape(
        mapping.batch_size, mapping.compact_size, *([1] * len(suffix))
    ).expand(mapping.batch_size, mapping.compact_size, *suffix)
    gathered = moved.gather(1, index)
    mask = mapping.valid.reshape(
        mapping.batch_size, mapping.compact_size, *([1] * len(suffix))
    )
    return gathered.masked_fill(~mask, 0).permute(inverse)


def compact_scatter(
    values: Tensor,
    mapping: CompactMap,
    dim: int = 1,
    *,
    fill_value: float | int | bool = 0,
) -> Tensor:
    """Scatter padded compact rows back to the map's fixed row width."""

    if not isinstance(values, Tensor) or values.ndim < 2:
        raise ValueError("values must have a batch and compact-row dimension")
    dim = dim % values.ndim
    if dim == 0:
        raise ValueError("the batch dimension cannot be scattered")
    if values.shape[0] != mapping.batch_size or values.shape[dim] != mapping.compact_size:
        raise ValueError("values do not match the compact map")
    order = [0, dim] + [axis for axis in range(1, values.ndim) if axis != dim]
    inverse = [order.index(axis) for axis in range(values.ndim)]
    moved = values.permute(order)
    suffix = moved.shape[2:]
    output = torch.full(
        (mapping.batch_size, mapping.full_size, *suffix),
        fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    if mapping.compact_size and bool(mapping.valid.any()):
        batch_ids = torch.arange(mapping.batch_size, device=values.device)[:, None]
        batch_ids = batch_ids.expand_as(mapping.valid)[mapping.valid]
        full_ids = mapping.compact_to_full.to(values.device)[mapping.valid]
        output[batch_ids, full_ids] = moved[mapping.valid]
    return output.permute(inverse)


def permute_rows(values: Tensor, old_to_new: Tensor, dim: int = 1) -> Tensor:
    """Apply a supplied bijective row map without interpreting its semantics."""

    if old_to_new.ndim != 1 or old_to_new.dtype not in (torch.int32, torch.int64):
        raise TypeError("old_to_new must be an integer vector")
    dim = dim % values.ndim
    if values.shape[dim] != old_to_new.numel():
        raise ValueError("row-map length does not match values")
    expected = torch.arange(old_to_new.numel(), device=old_to_new.device)
    if not torch.equal(torch.sort(old_to_new.long()).values, expected):
        raise ValueError("old_to_new must be a bijection")
    inverse = torch.empty_like(old_to_new, dtype=torch.long)
    inverse[old_to_new.long()] = expected
    return values.index_select(dim, inverse.to(values.device))
