"""Strict paired best-response probe report consumer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pbg.games.poker.queued.response import (
    SHARED_RESPONSE_SEMANTICS,
    SequentialSharedResponseEvaluator,
)


@dataclass(frozen=True)
class BRProbeReport:
    semantics: str
    target_sha256: str
    initial_model_sha256: str
    initialization: str
    trajectory_count: int
    trajectory_id_sha256: str
    mean_gain_bb_per_hand: float
    standard_error_bb_per_hand: float
    approximate_nashconv_bb_per_hand: float
    per_seat_gain_bb_per_hand: tuple[float, ...]

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_br_probe_observations(
    *,
    players: int,
    target_sha256: str,
    initial_model_sha256: str,
    initialization: str,
    trajectory_ids: torch.Tensor,
    mixed_payoff: torch.Tensor,
    baseline_payoff: torch.Tensor,
) -> BRProbeReport:
    evaluator = SequentialSharedResponseEvaluator(
        players, target_sha256, initial_model_sha256, initialization
    )
    report = evaluator.evaluate(
        trajectory_ids,
        lambda _ids, _seats: mixed_payoff,
        lambda _ids, _seats: baseline_payoff,
    )
    statistics = report.statistics
    return BRProbeReport(
        semantics=SHARED_RESPONSE_SEMANTICS,
        target_sha256=report.target_sha256,
        initial_model_sha256=report.initial_model_sha256,
        initialization=report.initialization,
        trajectory_count=report.trajectory_count,
        trajectory_id_sha256=report.trajectory_id_sha256,
        mean_gain_bb_per_hand=statistics.mean_gain,
        standard_error_bb_per_hand=statistics.standard_error,
        approximate_nashconv_bb_per_hand=(
            statistics.approximate_nashconv_lower_bound
        ),
        per_seat_gain_bb_per_hand=statistics.per_seat_gain,
    )


def evaluate_br_probe(path: str | Path, **_unused: object) -> BRProbeReport:
    value = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema",
        "players",
        "target_sha256",
        "initial_model_sha256",
        "initialization",
        "trajectory_ids",
        "mixed_payoff",
        "baseline_payoff",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["schema"] != "pbg-queued-paired-probe-v1"
    ):
        raise RuntimeError(
            "BR evaluation accepts only queued paired-probe observations"
        )
    return evaluate_br_probe_observations(
        **{name: value[name] for name in expected - {"schema"}}
    )


__all__ = [
    "BRProbeReport",
    "evaluate_br_probe",
    "evaluate_br_probe_observations",
]
