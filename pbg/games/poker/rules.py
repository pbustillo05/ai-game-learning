"""Shared accelerator-native NLHE rules and affine pot accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import Tensor

from pbg.games.poker.cards import DeckGeometry

FOLD = 0
CHECK_CALL = 1
RAISE_OFFSET = 2
PREFLOP = 0
RIVER = 3
LIVE = 0
FOLDED = 1
ALL_IN = 2


@dataclass(frozen=True)
class PokerConfig:
    """Configuration for one fixed compact-deck NLHE cash-game run."""

    n_players: int = 6
    stack_bb: float = 100.0
    rank_count: int = 13
    n_ticks: int = field(init=False)
    geometry: DeckGeometry = field(init=False, repr=False)

    sb: ClassVar[float] = 0.5
    bb: ClassVar[float] = 1.0
    chip_unit_bb: ClassVar[float] = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.n_players, bool) or not 2 <= self.n_players <= 6:
            raise ValueError("n_players must be an integer in [2,6]")
        if not math.isfinite(self.stack_bb) or not 8.0 <= self.stack_bb <= 128.0:
            raise ValueError("stack_bb must be finite and in [8,128]")
        ticks = round(self.stack_bb / self.chip_unit_bb)
        if not math.isclose(self.stack_bb, ticks * self.chip_unit_bb, abs_tol=1e-9):
            raise ValueError("stack_bb must be an integer multiple of 0.5 bb")
        geometry = DeckGeometry(self.rank_count)
        if 2 * self.n_players + 5 >= geometry.n_cards:
            raise ValueError("compact deck must satisfy 2*n_players + 5 < 4*rank_count")
        object.__setattr__(self, "n_ticks", ticks)
        object.__setattr__(self, "geometry", geometry)


@dataclass
class PotStrata:
    """Public affine coefficients supplied to evaluator showdown methods."""

    stratum_players: Tensor
    stratum_amount: Tensor
    payoff_baseline: Tensor


def assemble_showdown_payoff(strata: PotStrata, showdown_share: Tensor) -> Tensor:
    """Assemble chip payoffs without inspecting cards or comparing hands."""

    if showdown_share.dtype != torch.float32 or showdown_share.ndim != 3:
        raise TypeError("showdown_share must be float32 [B,K,P]")
    if showdown_share.shape != strata.stratum_players.shape:
        raise ValueError("showdown_share shape must match stratum_players")
    if strata.stratum_amount.shape != showdown_share.shape[:2]:
        raise ValueError("stratum_amount must have shape [B,K]")
    if strata.payoff_baseline.shape != (showdown_share.shape[0], showdown_share.shape[2]):
        raise ValueError("payoff_baseline must have shape [B,P]")
    tensors = (strata.stratum_players, strata.stratum_amount, strata.payoff_baseline, showdown_share)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all payoff tensors must be on one device")
    if bool((showdown_share < -1e-6).any()):
        raise ValueError("showdown shares must be nonnegative")
    if bool((showdown_share.masked_select(~strata.stratum_players).abs() > 1e-6).any()):
        raise ValueError("ineligible players must have zero showdown share")
    positive = strata.stratum_amount > 1e-6
    totals = showdown_share.sum(dim=-1)
    if bool((torch.abs(totals[positive] - 1.0) > 1e-5).any()):
        raise ValueError("shares in each positive stratum must sum to one")
    payoff = strata.payoff_baseline + (strata.stratum_amount[:, :, None] * showdown_share).sum(dim=1)
    if bool((payoff.sum(dim=-1).abs() > 2e-4).any()):
        raise ValueError("assembled chip payoffs must be conserved")
    return payoff


__all__ = [
    "ALL_IN", "CHECK_CALL", "FOLD", "FOLDED", "LIVE", "PREFLOP",
    "RAISE_OFFSET", "RIVER", "PokerConfig", "PotStrata", "assemble_showdown_payoff",
]
