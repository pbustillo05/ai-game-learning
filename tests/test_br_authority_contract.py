from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pbg.core.br import (
    SHARED_RESPONSE_SEMANTICS,
    balanced_evaluation_probe_seats,
    paired_probe_statistics,
    training_probe_seats,
)


ROOT = Path(__file__).resolve().parents[1]


def test_training_probe_seat_law_is_keyed_and_schedule_independent() -> None:
    keys = torch.tensor([0, 8, 16, 24, 32, 40], dtype=torch.int64)
    expected = torch.tensor([0, 2, 1, 0, 2, 1], dtype=torch.int32)
    seats = training_probe_seats(keys, 3)
    assert torch.equal(seats, expected)

    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    permuted = training_probe_seats(keys[permutation], 3)
    assert torch.equal(permuted, seats[permutation])


@pytest.mark.parametrize("players", (2, 3, 6))
def test_evaluation_design_balances_every_seat_button_pair(players: int) -> None:
    ids = torch.arange(players * players, dtype=torch.int64)
    seats = balanced_evaluation_probe_seats(ids, players)
    buttons = torch.remainder(ids, players).to(torch.int32)
    pairs = {(int(seat), int(button)) for seat, button in zip(seats, buttons)}
    assert len(pairs) == players * players
    assert pairs == {
        (seat, button)
        for seat in range(players)
        for button in range(players)
    }


def test_paired_estimator_is_sum_of_per_seat_unilateral_gains() -> None:
    differences = torch.tensor(
        [1.0, 3.0, -2.0, 2.0, 0.5, 1.5], dtype=torch.float64
    )
    seats = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
    result = paired_probe_statistics(differences, seats, 3)

    assert result.per_seat_gain == pytest.approx((2.0, 0.0, 1.0))
    assert result.mean_gain == pytest.approx(1.0)
    assert result.approximate_nashconv_lower_bound == pytest.approx(3.0)
    assert result.approximate_nashconv_lower_bound == pytest.approx(
        3 * result.mean_gain
    )

    exact_best_response_gains = torch.tensor([2.5, 0.75, 1.25])
    achieved = torch.tensor(result.per_seat_gain)
    assert bool((achieved <= exact_best_response_gains).all())
    assert result.approximate_nashconv_lower_bound <= float(
        exact_best_response_gains.sum()
    )


def test_paired_estimator_rejects_unbalanced_seat_counts() -> None:
    with pytest.raises(ValueError, match="equal positive rows per seat"):
        paired_probe_statistics(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0, 0, 1], dtype=torch.int32),
            2,
        )


def test_authority_and_production_semantics_are_version_aligned() -> None:
    theory = (ROOT / "docs/theory/core_math.tex").read_text(encoding="utf-8")
    pseudocode = (
        ROOT / "docs/theory/pseudocode/shared_response_probe.tex"
    ).read_text(encoding="utf-8")
    guidance = (ROOT / "docs/training_and_evaluation.md").read_text(
        encoding="utf-8"
    )

    for label in (
        "eq:shared-response-class",
        "eq:probe-seat-law",
        "eq:shared-probe-empirical-loss",
        "eq:probe-public-cutoff",
        "eq:approximate-nashconv-estimator",
    ):
        assert f"\\label{{{label}}}" in theory
    assert "TrainSharedApproximateResponse" in pseudocode
    assert "EvaluateSharedApproximateResponse" in pseudocode
    assert SHARED_RESPONSE_SEMANTICS.replace("_", "\\_") in pseudocode
    assert SHARED_RESPONSE_SEMANTICS in guidance
