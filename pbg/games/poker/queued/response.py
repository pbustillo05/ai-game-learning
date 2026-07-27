"""Neutral shared-response probe laws for the queued poker consumer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


SHARED_RESPONSE_SEMANTICS = "shared_seat_approximate_response_v1"


def training_probe_seats(keys: Tensor, players: int) -> Tensor:
    if keys.dtype != torch.int64 or keys.ndim != 1:
        raise TypeError("probe keys must be one-dimensional int64")
    if isinstance(players, bool) or not isinstance(players, int) or players < 2:
        raise ValueError("players must be an integer of at least two")
    return keys.remainder(players).to(torch.int32)


def balanced_evaluation_probe_seats(
    trajectory_ids: Tensor, players: int
) -> Tensor:
    if trajectory_ids.dtype != torch.int64 or trajectory_ids.ndim != 1:
        raise TypeError("trajectory ids must be one-dimensional int64")
    if isinstance(players, bool) or not isinstance(players, int) or players < 2:
        raise ValueError("players must be an integer of at least two")
    if bool(trajectory_ids.lt(0).any()):
        raise ValueError("trajectory ids must be nonnegative")
    return torch.div(
        trajectory_ids, players, rounding_mode="floor"
    ).remainder(players).to(torch.int32)


@dataclass(frozen=True)
class PairedProbeStatistics:
    mean_gain: float
    standard_error: float
    per_seat_gain: tuple[float, ...]
    approximate_nashconv_lower_bound: float


def paired_probe_statistics(
    differences: Tensor, seats: Tensor, players: int
) -> PairedProbeStatistics:
    if differences.ndim != 1 or not differences.dtype.is_floating_point:
        raise TypeError("paired differences must be a floating vector")
    if seats.ndim != 1 or seats.dtype not in (torch.int32, torch.int64):
        raise TypeError("paired seats must be an integer vector")
    if differences.shape != seats.shape or differences.device != seats.device:
        raise ValueError("paired differences and seats must be aligned")
    if differences.numel() <= 1:
        raise ValueError("paired evaluation requires at least two rows")
    if isinstance(players, bool) or not isinstance(players, int) or players < 2:
        raise ValueError("players must be an integer of at least two")
    if not bool(torch.isfinite(differences).all()):
        raise ValueError("paired differences must be finite")
    counts = torch.bincount(seats.to(torch.int64), minlength=players)
    if (
        bool(((seats < 0) | (seats >= players)).any())
        or bool(counts.eq(0).any())
        or not bool(counts.eq(counts[0]).all())
    ):
        raise ValueError("paired evaluation must be positive and seat-balanced")
    values = differences.to(torch.float64)
    per_seat = torch.stack(
        tuple(values[seats.eq(player)].mean() for player in range(players))
    )
    mean = values.mean()
    lower_bound = per_seat.sum()
    if not bool(
        torch.isclose(
            lower_bound,
            players * mean,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ):
        raise RuntimeError("balanced paired-gain identities disagree")
    standard_error = values.std(unbiased=True) / math.sqrt(values.numel())
    return PairedProbeStatistics(
        mean_gain=float(mean),
        standard_error=float(standard_error),
        per_seat_gain=tuple(float(value) for value in per_seat),
        approximate_nashconv_lower_bound=float(lower_bound),
    )


__all__ = [
    "PairedProbeStatistics",
    "SHARED_RESPONSE_SEMANTICS",
    "balanced_evaluation_probe_seats",
    "paired_probe_statistics",
    "training_probe_seats",
]
