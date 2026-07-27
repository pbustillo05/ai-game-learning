"""Repository-owned compact folded-range statistics for queued poker."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from pbg.games.poker.queued.hooks import FOLDED, NLHEGeometry


class FoldedStatisticsError(ValueError):
    pass


def folded_statistics(
    geometry: NLHEGeometry,
    *,
    board_cards: Tensor,
    range_weights: Tensor,
    status: Tensor,
    combo_cards: Tensor,
    beta_rank: int,
    line_dim: int,
) -> Mapping[str, Tensor]:
    """Build deterministic FP32 model-ready features without evaluator calls.

    Folded seats are kept in seat order.  All reductions use dense fixed-order
    FP32 arithmetic; masked slots remain exact zeros.
    """

    batch, players, hands = range_weights.shape
    if (
        board_cards.device.type != "cuda"
        or range_weights.dtype != torch.float32
        or status.shape != (batch, players)
        or hands != geometry.private_hands
        or combo_cards.shape != (hands, 2)
    ):
        raise FoldedStatisticsError("folded statistic inputs violate poker geometry")
    folder_mask = status.eq(FOLDED)
    finite = torch.isfinite(range_weights)
    weights = range_weights.masked_fill(~finite, 0.0).clamp_min(0.0)
    weights = weights * folder_mask[..., None]
    mass = weights.sum(-1)
    card_incidence = torch.zeros(
        (hands, geometry.deck_cards),
        dtype=torch.float32,
        device=weights.device,
    )
    card_incidence.scatter_(
        1,
        (combo_cards - geometry.first_card).to(torch.int64),
        1.0,
    )
    card_mass = torch.einsum("bph,hc->bpc", weights, card_incidence)
    marginal = card_mass.sum(1)
    pair_pressure = marginal[:, (combo_cards[:, 0] - geometry.first_card).long()]
    pair_pressure = pair_pressure + marginal[
        :, (combo_cards[:, 1] - geometry.first_card).long()
    ]
    mean_pressure = pair_pressure.mean(-1, keepdim=True)
    hand_log_ratio = torch.log1p(pair_pressure) - torch.log1p(mean_pressure)
    return MappingProxyType(
        {
            "hand_log_ratio": hand_log_ratio.contiguous(),
            "beta_eigenvalue": torch.zeros(
                (batch, beta_rank), dtype=torch.float32, device=weights.device
            ),
            "beta_eigenvector": torch.zeros(
                (batch, geometry.deck_cards, beta_rank),
                dtype=torch.float32,
                device=weights.device,
            ),
            "folder_log_mass": torch.where(
                folder_mask,
                torch.log(mass.clamp_min(torch.finfo(torch.float32).tiny)),
                torch.zeros_like(mass),
            ).contiguous(),
            "folder_card_mass": card_mass.contiguous(),
            "folder_line_meta": torch.zeros(
                (batch, players, line_dim),
                dtype=torch.float32,
                device=weights.device,
            ),
            "folder_mask": folder_mask.contiguous(),
            "folder_output_valid": folder_mask.logical_and(mass > 0).contiguous(),
            "folder_seat": torch.arange(
                players, dtype=torch.int64, device=weights.device
            )[None].expand(batch, -1).contiguous(),
            "tight_cluster_scalar": torch.zeros(
                batch, dtype=torch.float32, device=weights.device
            ),
        }
    )


__all__ = ["FoldedStatisticsError", "folded_statistics"]
