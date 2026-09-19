"""Neutral shared-response probe laws for the queued poker consumer."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

SHARED_RESPONSE_SEMANTICS = "shared_seat_approximate_response_v1"
FIXED_TARGET_RESPONSE_SEMANTICS = "fixed_target_paired_response_gain_v2_1"


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


def probe_evaluation_design(
    trajectory_ids: Tensor, players: int
) -> Tensor:
    """Validate complete consecutive P-by-P seat/button blocks."""

    block_size = players * players
    if (
        trajectory_ids.numel() == 0
        or trajectory_ids.numel() % block_size
    ):
        raise ValueError(
            "paired evaluation requires complete P^2 trajectory blocks"
        )
    first = int(trajectory_ids[0])
    expected = torch.arange(
        first,
        first + trajectory_ids.numel(),
        dtype=torch.int64,
        device=trajectory_ids.device,
    )
    if first % block_size or not bool(trajectory_ids.eq(expected).all()):
        raise ValueError(
            "paired trajectory ids must be consecutive P^2 blocks"
        )
    seats = balanced_evaluation_probe_seats(trajectory_ids, players)
    buttons = trajectory_ids.remainder(players).to(torch.int64)
    pair = seats.to(torch.int64) * players + buttons
    pair_counts = torch.bincount(pair, minlength=block_size)
    if not bool(pair_counts.eq(trajectory_ids.numel() // block_size).all()):
        raise ValueError(
            "each probe-seat/button pair must occur once per block"
        )
    return seats


@dataclass(frozen=True)
class PairedProbeStatistics:
    mean_gain: float
    standard_error: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    per_seat_gain: tuple[float, ...]
    approximate_nashconv_lower_bound: float


@dataclass(frozen=True)
class FixedTargetPairedStatistics:
    """Signed paired-gain estimates over independent logical-deal clusters."""

    mean_gain: float
    standard_error: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    per_seat_gain: tuple[float, ...]
    per_seat_standard_error: tuple[float, ...]
    per_seat_confidence_interval: tuple[tuple[float, float], ...]
    sum_achieved_response_gain: float
    sum_achieved_response_gain_standard_error: float
    sum_achieved_response_gain_confidence_interval: tuple[float, float]
    independent_cluster_count: int
    sampling_unit: str


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


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction for regularized beta."""

    maximum_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), floor) * (1.0 if d >= 0 else -1.0)
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + even) * (a + even))
        )
        d = 1.0 + coefficient * d
        d = 1.0 / max(abs(d), floor) * (1.0 if d >= 0 else -1.0)
        c = 1.0 + coefficient / c
        c = max(abs(c), floor) * (1.0 if c >= 0 else -1.0)
        result *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        d = 1.0 / max(abs(d), floor) * (1.0 if d >= 0 else -1.0)
        c = 1.0 + coefficient / c
        c = max(abs(c), floor) * (1.0 if c >= 0 else -1.0)
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise RuntimeError("Student-t beta fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - (
        front * _beta_continued_fraction(b, a, 1.0 - x) / b
    )


def _student_t_critical_95(degrees: int) -> float:
    if degrees <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if degrees < len(_T95):
        return _T95[degrees]
    lower, upper = 0.0, 16.0
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        x = degrees / (degrees + middle * middle)
        cdf = 1.0 - 0.5 * _regularized_beta(
            x, 0.5 * degrees, 0.5
        )
        if cdf < 0.975:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def fixed_target_paired_statistics(
    differences: Tensor,
    seats: Tensor,
    players: int,
    *,
    cluster_ids: Tensor,
) -> FixedTargetPairedStatistics:
    """Estimate signed gain, clustering every replay of one deal together."""

    if differences.ndim != 1 or not differences.dtype.is_floating_point:
        raise TypeError("paired differences must be a floating vector")
    if seats.ndim != 1 or seats.dtype not in (torch.int32, torch.int64):
        raise TypeError("paired seats must be an integer vector")
    if cluster_ids.ndim != 1 or cluster_ids.dtype != torch.int64:
        raise TypeError("cluster ids must be a one-dimensional int64 vector")
    if (
        differences.shape != seats.shape
        or differences.shape != cluster_ids.shape
        or differences.device != seats.device
        or differences.device != cluster_ids.device
    ):
        raise ValueError("paired differences, seats, and clusters must be aligned")
    if differences.numel() <= 1 or not bool(torch.isfinite(differences).all()):
        raise ValueError("paired differences must contain finite replicated values")
    if isinstance(players, bool) or not isinstance(players, int) or players < 2:
        raise ValueError("players must be an integer of at least two")
    seat64 = seats.to(torch.int64)
    if bool(((seat64 < 0) | (seat64 >= players)).any()):
        raise ValueError("paired seats are outside the table")

    unique_cluster, inverse = torch.unique(
        cluster_ids, sorted=True, return_inverse=True
    )
    cluster_count = unique_cluster.numel()
    if cluster_count <= 1:
        raise ValueError("paired evaluation needs at least two independent clusters")
    cell = inverse * players + seat64
    cell_count = torch.zeros(
        cluster_count * players, dtype=torch.int64, device=differences.device
    )
    cell_count.scatter_add_(0, cell, torch.ones_like(cell))
    cell_sum = torch.zeros(
        cluster_count * players, dtype=torch.float64, device=differences.device
    )
    cell_sum.scatter_add_(0, cell, differences.to(torch.float64))
    cell_count = cell_count.reshape(cluster_count, players)
    present = cell_count.gt(0)
    cell_value = (
        cell_sum.reshape(cluster_count, players)
        / cell_count.clamp_min(1).to(torch.float64)
    )
    single_seat = bool(present.sum(1).eq(1).all())
    complete_seat = bool(present.all())
    if not single_seat and not complete_seat:
        raise ValueError(
            "cluster design must use one seat per deal or all seats per shared deal"
        )

    per_seat_mean = []
    per_seat_se = []
    per_seat_interval = []
    per_seat_count = []
    for seat in range(players):
        values = cell_value[present[:, seat], seat]
        if values.numel() <= 1:
            raise ValueError("each seat needs at least two independent deal clusters")
        mean = values.mean()
        se = values.std(unbiased=True) / math.sqrt(values.numel())
        critical = _student_t_critical_95(values.numel() - 1)
        per_seat_mean.append(mean)
        per_seat_se.append(se)
        per_seat_interval.append(
            (float(mean - critical * se), float(mean + critical * se))
        )
        per_seat_count.append(values.numel())

    mean = torch.stack(per_seat_mean).mean()
    if complete_seat:
        cluster_values = cell_value.mean(1)
        standard_error = cluster_values.std(unbiased=True) / math.sqrt(cluster_count)
        degrees = cluster_count - 1
        sampling_unit = "logical_deal_cluster_all_seats"
    else:
        component_variance = torch.stack(per_seat_se).square() / (players * players)
        variance = component_variance.sum()
        standard_error = variance.sqrt()
        denominator = sum(
            float(value * value) / (count - 1)
            for value, count in zip(
                component_variance.tolist(), per_seat_count, strict=True
            )
            if value > 0
        )
        degrees = (
            min(per_seat_count) - 1
            if denominator == 0
            else max(1, round(float(variance * variance) / denominator))
        )
        sampling_unit = "logical_deal_cluster_seat_stratified"
    critical = _student_t_critical_95(degrees)
    half_width = critical * standard_error
    sum_gain = players * mean
    sum_standard_error = players * standard_error
    return FixedTargetPairedStatistics(
        mean_gain=float(mean),
        standard_error=float(standard_error),
        confidence_level=0.95,
        confidence_interval=(float(mean - half_width), float(mean + half_width)),
        per_seat_gain=tuple(float(value) for value in per_seat_mean),
        per_seat_standard_error=tuple(float(value) for value in per_seat_se),
        per_seat_confidence_interval=tuple(per_seat_interval),
        sum_achieved_response_gain=float(sum_gain),
        sum_achieved_response_gain_standard_error=float(sum_standard_error),
        sum_achieved_response_gain_confidence_interval=(
            float(sum_gain - players * half_width),
            float(sum_gain + players * half_width),
        ),
        independent_cluster_count=cluster_count,
        sampling_unit=sampling_unit,
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
    critical = _student_t_critical_95(degrees)
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
    initialization: str
    trajectory_count: int
    trajectory_id_sha256: str
    statistics: PairedProbeStatistics


@dataclass(frozen=True)
class SequentialSharedResponseEvaluator:
    """Execute paired mixed/baseline rollouts on identical logical work."""

    players: int
    target_sha256: str
    initial_model_sha256: str
    initialization: str = "random"

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
        if self.initialization not in ("random", "target"):
            raise ValueError("probe initialization must be random or target")

    def evaluate(
        self,
        trajectory_ids: Tensor,
        mixed_rollout: Callable[[Tensor, Tensor], Tensor],
        baseline_rollout: Callable[[Tensor, Tensor], Tensor],
    ) -> SharedResponseEvaluationReport:
        seats = probe_evaluation_design(trajectory_ids, self.players)
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
            initialization=self.initialization,
            trajectory_count=trajectory_ids.numel(),
            trajectory_id_sha256=identity,
            statistics=paired_probe_statistics(
                differences, seats, self.players
            ),
        )


__all__ = [
    "FIXED_TARGET_RESPONSE_SEMANTICS",
    "SHARED_RESPONSE_SEMANTICS",
    "FixedTargetPairedStatistics",
    "PairedProbeStatistics",
    "SequentialSharedResponseEvaluator",
    "SharedResponseEvaluationReport",
    "balanced_evaluation_probe_seats",
    "fixed_target_paired_statistics",
    "paired_probe_statistics",
    "probe_evaluation_design",
]
