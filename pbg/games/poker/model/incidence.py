"""Generic gather, scatter, and row-permutation tensor helpers.

This module deliberately has no knowledge of poker rules.  Its maps are built
from boolean compatibility rows supplied by the representation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from pbg.core.tensor_ops import Linear, Embedding

from pbg.games.poker.model.primitives import LN, accelerator_assert


class IncidenceResidual(nn.Module):
    """Exact unnormalized card/identical-row mass residual for one layer."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = LN(d_model)
        self.source = Linear(d_model, d_model, bias=False)
        self.query = Linear(d_model, d_model)
        self.identical = Linear(d_model, d_model, bias=False)
        self.own = Linear(d_model, d_model, bias=False)
        self.output = Linear(d_model, d_model)

    def forward(
        self,
        rows: Tensor,
        mass: Tensor,
        card_incidence: Tensor,
        identical_scale: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``j, O_card, O_equal, O_own`` with shape ``[B,P,R,D]``.

        ``card_incidence`` is the two-hot postflop map or orbit-averaged
        preflop map ``[B,R,D]``/``[R,D]``. ``identical_scale`` is one for
        combo rows and ``1/n_k`` for class-orbit rows.
        """

        if rows.ndim != 4 or mass.shape != rows.shape[:3]:
            raise ValueError("incidence rows/mass must have shapes [B,P,R,D]/[B,P,R]")
        batch, _, row_count, _ = rows.shape
        if card_incidence.ndim == 2:
            card_incidence = card_incidence[None].expand(batch, -1, -1)
        if (
            card_incidence.ndim != 3
            or card_incidence.shape[:2] != (batch, row_count)
            or card_incidence.shape[2] <= 0
        ):
            raise ValueError("card_incidence must have shape [R,D] or [B,R,D]")
        if identical_scale.ndim == 1:
            identical_scale = identical_scale[None].expand(batch, -1)
        if identical_scale.shape != (batch, row_count):
            raise ValueError("identical_scale must have shape [R] or [B,R]")
        normalized = self.norm(rows)
        projected = self.source(normalized)
        # Incidence mass reductions retain FP32 accumulation; only the working
        # residual/projection boundary returns to the selected compute dtype.
        with torch.autocast(rows.device.type, enabled=False):
            source = projected.float() * mass.float()[..., None]
            incidence = card_incidence.float()
            card_accumulator = torch.einsum("brc,bprd->bpcd", incidence, source)
            card_overlap = torch.einsum("brc,bpcd->bprd", incidence, card_accumulator)
            same = source * identical_scale[:, None, :, None].float()
            total_card = card_overlap.sum(1, keepdim=True)
            total_same = same.sum(1, keepdim=True)
            other_equal = total_same - same
            other_card = total_card - card_overlap - 2.0 * other_equal
            own_card = card_overlap - 2.0 * same
        query = self.query(normalized)
        residual = self.output(
            torch.nn.functional.gelu(
                query
                + other_card.to(query.dtype)
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


def build_compact_map(
    compatibility: Tensor,
    compact_size: int | None = None,
) -> CompactMap:
    """Build a stable, padded map from supplied bool rows ``[B,H]``.

    Production callers provide the fixed compatible-combo count for the street.
    The conservative default retains full width without a device-to-host read.
    """

    if not isinstance(compatibility, Tensor) or compatibility.ndim != 2:
        raise ValueError("compatibility must have shape [B,H]")
    if compatibility.dtype != torch.bool:
        raise TypeError("compatibility must have dtype torch.bool")
    batch, full_size = compatibility.shape
    counts = compatibility.sum(dim=1)
    if compact_size is None:
        compact_size = full_size
    if isinstance(compact_size, bool) or not 0 <= compact_size <= full_size:
        raise ValueError("compact_size must be in [0,H]")
    enough_space = (counts <= compact_size).all()
    accelerator_assert(enough_space, "compact_size omits compatible rows")
    compact_to_full = (
        compatibility[:, :compact_size].to(torch.long).clone().zero_()
    )
    rank = compatibility.long().cumsum(dim=1) - 1
    full_to_compact = torch.where(compatibility, rank, -torch.ones_like(rank))
    if compact_size:
        full_ids = torch.arange(full_size, device=compatibility.device)[None].expand(batch, -1)
        compact_to_full = compact_to_full.scatter_add(
            1,
            rank.clamp(0, compact_size - 1),
            torch.where(compatibility, full_ids, torch.zeros_like(full_ids)),
        )
    valid = (
        torch.arange(compact_size, device=compatibility.device)[None]
        < counts[:, None]
    )
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
    fill_value: float | bool = 0,
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
    if mapping.compact_size:
        index = mapping.full_to_compact.clamp_min(0).reshape(
            mapping.batch_size, mapping.full_size, *([1] * len(suffix))
        ).expand(mapping.batch_size, mapping.full_size, *suffix)
        output = moved.gather(1, index)
        selected = (mapping.full_to_compact >= 0).reshape(
            mapping.batch_size, mapping.full_size, *([1] * len(suffix))
        )
        output = output.masked_fill(~selected, fill_value)
    else:
        output = torch.full(
            (mapping.batch_size, mapping.full_size, *suffix),
            fill_value,
            dtype=values.dtype,
            device=values.device,
        )
    return output.permute(inverse)
