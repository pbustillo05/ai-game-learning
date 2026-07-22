"""Authoritative shared-seat approximate-response probe primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


SHARED_RESPONSE_SEMANTICS = "shared_seat_approximate_response_v1"


def training_probe_seats(keys: Tensor, n_players: int) -> Tensor:
    """Map counter-based root keys to the declared uniform probe-seat law."""

    if keys.dtype != torch.int64:
        raise TypeError("probe-seat keys must be int64")
    if keys.ndim != 1:
        raise ValueError("probe-seat keys must be one-dimensional")
    if isinstance(n_players, bool) or not isinstance(n_players, int) or n_players < 2:
        raise ValueError("n_players must be an integer of at least two")
    return torch.remainder(keys, n_players).to(torch.int32)


def balanced_evaluation_probe_seats(
    trajectory_ids: Tensor, n_players: int
) -> Tensor:
    """Return the seat component of the mandatory ``P^2`` evaluation design."""

    if trajectory_ids.dtype != torch.int64 or trajectory_ids.ndim != 1:
        raise TypeError("evaluation trajectory ids must be one-dimensional int64")
    if isinstance(n_players, bool) or not isinstance(n_players, int) or n_players < 2:
        raise ValueError("n_players must be an integer of at least two")
    if bool((trajectory_ids < 0).any()):
        raise ValueError("evaluation trajectory ids must be nonnegative")
    return torch.remainder(
        torch.div(trajectory_ids, n_players, rounding_mode="floor"), n_players
    ).to(torch.int32)


@dataclass(frozen=True)
class PairedProbeStatistics:
    """Balanced paired-gain summary for one fixed probe checkpoint."""

    mean_gain: float
    standard_error: float
    per_seat_gain: tuple[float, ...]
    approximate_nashconv_lower_bound: float


def paired_probe_statistics(
    differences: Tensor, seats: Tensor, n_players: int
) -> PairedProbeStatistics:
    """Aggregate Eq. ``approximate-nashconv-estimator`` on balanced rows."""

    if differences.ndim != 1 or not differences.dtype.is_floating_point:
        raise TypeError("paired differences must be a one-dimensional floating tensor")
    if seats.ndim != 1 or seats.dtype not in (torch.int32, torch.int64):
        raise TypeError("paired seats must be a one-dimensional integer tensor")
    if differences.shape != seats.shape or differences.device != seats.device:
        raise ValueError("paired differences and seats must have the same shape and device")
    if differences.numel() <= 1:
        raise ValueError("paired evaluation requires at least two rows")
    if isinstance(n_players, bool) or not isinstance(n_players, int) or n_players < 2:
        raise ValueError("n_players must be an integer of at least two")
    if not bool(torch.isfinite(differences).all()):
        raise ValueError("paired differences must be finite")
    if bool(((seats < 0) | (seats >= n_players)).any()):
        raise ValueError("paired seats are out of range")

    values = differences.to(torch.float64)
    counts = torch.bincount(seats.to(torch.int64), minlength=n_players)
    if bool((counts == 0).any()) or not bool((counts == counts[0]).all()):
        raise ValueError("paired evaluation must contain equal positive rows per seat")
    per_seat_tensor = torch.stack(
        [values[seats == player].mean() for player in range(n_players)]
    )
    mean = values.mean()
    lower_bound = per_seat_tensor.sum()
    expected = float(n_players) * mean
    if not bool(torch.isclose(lower_bound, expected, rtol=1.0e-12, atol=1.0e-12)):
        raise RuntimeError("balanced paired-gain identities disagree")
    standard_error = values.std(unbiased=True) / math.sqrt(values.numel())
    return PairedProbeStatistics(
        mean_gain=float(mean),
        standard_error=float(standard_error),
        per_seat_gain=tuple(float(value) for value in per_seat_tensor),
        approximate_nashconv_lower_bound=float(lower_bound),
    )


__all__ = [
    "PairedProbeStatistics",
    "SHARED_RESPONSE_SEMANTICS",
    "balanced_evaluation_probe_seats",
    "paired_probe_statistics",
    "training_probe_seats",
]
