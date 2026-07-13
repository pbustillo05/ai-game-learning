"""Generic gather, scatter, and row-permutation tensor helpers.

This module deliberately has no knowledge of poker rules.  Its maps are built
from boolean compatibility rows supplied by the representation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pbg.games.poker.model.primitives import LN


class IncidenceResidual(nn.Module):
    """Exact unnormalized card/identical-row mass residual for one layer."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = LN(d_model)
        self.source = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model)
        self.identical = nn.Linear(d_model, d_model, bias=False)
        self.own = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        rows: Tensor,
        mass: Tensor,
        card_incidence: Tensor,
        identical_scale: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``j, O_card, O_equal, O_own`` with shape ``[B,P,R,D]``.

        ``card_incidence`` is the two-hot postflop map or orbit-averaged
        preflop map ``[B,R,52]``/``[R,52]``. ``identical_scale`` is one for
        combo rows and ``1/n_k`` for class-orbit rows.
        """

        if rows.ndim != 4 or mass.shape != rows.shape[:3]:
            raise ValueError("incidence rows/mass must have shapes [B,P,R,D]/[B,P,R]")
        batch, _, row_count, _ = rows.shape
        if card_incidence.ndim == 2:
            card_incidence = card_incidence[None].expand(batch, -1, -1)
        if card_incidence.shape != (batch, row_count, 52):
            raise ValueError("card_incidence must have shape [R,52] or [B,R,52]")
        if identical_scale.ndim == 1:
            identical_scale = identical_scale[None].expand(batch, -1)
        if identical_scale.shape != (batch, row_count):
            raise ValueError("identical_scale must have shape [R] or [B,R]")
        normalized = self.norm(rows)
        source = self.source(normalized) * mass[..., None]
        card_accumulator = torch.einsum(
            "brc,bprd->bpcd", card_incidence.to(source.dtype), source
        )
        card_overlap = torch.einsum(
            "brc,bpcd->bprd", card_incidence.to(source.dtype), card_accumulator
        )
        same = source * identical_scale[:, None, :, None].to(source.dtype)
        total_card = card_overlap.sum(1, keepdim=True)
        total_same = same.sum(1, keepdim=True)
        other_equal = total_same - same
        other_card = total_card - card_overlap - 2.0 * other_equal
        own_card = card_overlap - 2.0 * same
        residual = self.output(
            torch.nn.functional.gelu(
                self.query(normalized)
                + other_card
                + self.identical(other_equal)
                + self.own(own_card)
            )
        )
        return residual, other_card, other_equal, own_card


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
