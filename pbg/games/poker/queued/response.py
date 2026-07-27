"""Neutral shared-response probe laws for the queued poker consumer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable

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
    confidence_level: float
    confidence_interval: tuple[float, float]
    per_seat_gain: tuple[float, ...]
    approximate_nashconv_lower_bound: float


_T95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


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
    degrees = values.numel() - 1
    critical = _T95[degrees] if degrees < len(_T95) else 1.96
    half_width = critical * standard_error
    return PairedProbeStatistics(
        mean_gain=float(mean),
        standard_error=float(standard_error),
        confidence_level=0.95,
        confidence_interval=(
            float(mean - half_width),
            float(mean + half_width),
        ),
        per_seat_gain=tuple(float(value) for value in per_seat),
        approximate_nashconv_lower_bound=float(lower_bound),
    )


@dataclass(frozen=True)
class SharedResponseEvaluationReport:
    semantics: str
    target_sha256: str
    initial_model_sha256: str
    trajectory_count: int
    trajectory_id_sha256: str
    statistics: PairedProbeStatistics


@dataclass(frozen=True)
class SequentialSharedResponseEvaluator:
    """Execute paired mixed/baseline rollouts on identical logical work."""

    players: int
    target_sha256: str
    initial_model_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("target_sha256", self.target_sha256),
            ("initial_model_sha256", self.initial_model_sha256),
        ):
            if (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")

    def evaluate(
        self,
        trajectory_ids: Tensor,
        mixed_rollout: Callable[[Tensor, Tensor], Tensor],
        baseline_rollout: Callable[[Tensor, Tensor], Tensor],
    ) -> SharedResponseEvaluationReport:
        seats = balanced_evaluation_probe_seats(
            trajectory_ids, self.players
        )
        mixed = mixed_rollout(trajectory_ids, seats)
        baseline = baseline_rollout(trajectory_ids, seats)
        if (
            mixed.shape != trajectory_ids.shape
            or baseline.shape != trajectory_ids.shape
            or mixed.device != trajectory_ids.device
            or baseline.device != trajectory_ids.device
        ):
            raise ValueError("paired rollout outputs must align to trajectory ids")
        differences = mixed.float() - baseline.float()
        identity = hashlib.sha256(
            trajectory_ids.detach().to("cpu", torch.int64).numpy().tobytes()
        ).hexdigest()
        return SharedResponseEvaluationReport(
            semantics=SHARED_RESPONSE_SEMANTICS,
            target_sha256=self.target_sha256,
            initial_model_sha256=self.initial_model_sha256,
            trajectory_count=trajectory_ids.numel(),
            trajectory_id_sha256=identity,
            statistics=paired_probe_statistics(
                differences, seats, self.players
            ),
        )


__all__ = [
    "PairedProbeStatistics",
    "SHARED_RESPONSE_SEMANTICS",
    "SequentialSharedResponseEvaluator",
    "SharedResponseEvaluationReport",
    "balanced_evaluation_probe_seats",
    "paired_probe_statistics",
    "training_probe_seats",
]
