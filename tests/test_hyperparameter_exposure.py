from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.hyperparameter_search import exposure
from pbg.core.contracts import PrivateEstimator
from pbg.games.poker.queued.hooks import PAD_CARD


def _board(rows: int) -> torch.Tensor:
    return torch.full((rows, 5), PAD_CARD, dtype=torch.uint8)


def _combos(cards: range) -> torch.Tensor:
    return torch.tensor(list(itertools.combinations(cards, 2)), dtype=torch.int32)


def test_exact_combo_entropy_clock_and_sampled_clock() -> None:
    combo_cards = _combos(range(48, 52))
    log_reach = torch.full((2, 2, 6), -torch.inf)
    log_reach[0, 0, :2] = math.log(0.5)
    log_reach[0, 1] = math.log(1.0 / 6.0)
    log_reach[1, 0] = math.log(1.0 / 6.0)
    log_reach[1, 1, :4] = math.log(0.25)
    actor = torch.tensor([0, 1], dtype=torch.int64)

    result = exposure.current_posterior_exposure(
        log_reach,
        actor,
        _board(2),
        combo_cards,
    )
    assert int(result["fresh_player_decisions"]) == 2
    assert float(result["sum_posterior_exp_entropy"]) == pytest.approx(6.0)
    assert float(result["sum_sqrt_posterior_exp_entropy"]) == pytest.approx(
        math.sqrt(2.0) + 2.0
    )
    assert float(result["sum_posterior_entropy_nats"]) == pytest.approx(
        math.log(2.0) + math.log(4.0)
    )

    sampled = exposure.current_posterior_exposure(
        log_reach,
        actor,
        _board(2),
        combo_cards,
        selected=torch.tensor([True, False]),
        sampled=True,
    )
    assert int(sampled["fresh_player_decisions"]) == 1
    assert float(sampled["sum_posterior_exp_entropy"]) == 1.0
    assert float(sampled["sum_sqrt_posterior_exp_entropy"]) == 1.0


def test_nonuniform_opponent_blockers_change_actor_marginal() -> None:
    combo_cards = _combos(range(48, 52))
    log_reach = torch.full((1, 2, 6), -torch.inf)
    log_reach[0, 0] = math.log(1.0 / 6.0)
    log_reach[0, 1, 0] = 0.0

    measured = exposure.current_posterior_exposure(
        log_reach,
        torch.tensor([0]),
        _board(1),
        combo_cards,
    )

    # Actor reach alone is uniform over six hands. Once the opponent is known
    # to hold (48, 49), only its disjoint complement (50, 51) remains possible.
    assert float(measured["sum_posterior_exp_entropy"]) == pytest.approx(1.0)
    assert float(measured["sum_posterior_entropy_nats"]) == pytest.approx(0.0)


def test_board_conflicts_are_excluded_and_class_width_fails_explicitly() -> None:
    combo_cards = _combos(range(47, 52))
    board = _board(1)
    board[0, 0] = 47
    actor = torch.tensor([0])
    valid = torch.zeros((1, 2, 10))
    conflict = combo_cards.eq(47).any(1)
    valid[:, :, conflict] = -torch.inf
    measured = exposure.current_posterior_exposure(valid, actor, board, combo_cards)
    assert float(measured["sum_posterior_exp_entropy"]) == pytest.approx(6.0)

    invalid = valid.clone()
    invalid[0, 0, int(torch.nonzero(conflict)[0])] = 0.0
    with pytest.raises(RuntimeError, match="known-board conflict"):
        exposure.current_posterior_exposure(invalid, actor, board, combo_cards)

    with pytest.raises(
        exposure.ExactComboPosteriorUnavailable,
        match="exact combo-card domain",
    ):
        exposure.current_posterior_exposure(
            torch.zeros((1, 2, 2)), actor, _board(1), combo_cards
        )


def test_gradient_covariance_discount_requires_repeated_gradient_evidence() -> None:
    first = torch.tensor([1.0, -1.0, 1.0, -1.0])
    independent_second = torch.tensor([1.0, 1.0, -1.0, -1.0])
    independent = torch.stack((first, independent_second), 1).unsqueeze(-1)
    independent_diagnostic = exposure.gradient_covariance_diagnostic(independent)
    assert independent_diagnostic["variance_design_effect"] == pytest.approx(1.0)
    assert independent_diagnostic["conservative_scheduling_discount"] == 1.0

    correlated = torch.stack((first, first), 1).unsqueeze(-1)
    correlated_diagnostic = exposure.gradient_covariance_diagnostic(correlated)
    assert correlated_diagnostic["variance_design_effect"] == pytest.approx(2.0)
    assert correlated_diagnostic[
        "variance_equivalent_component_count"
    ] == pytest.approx(1.0)
    records = [
        {
            "sum_posterior_exp_entropy": 100.0,
            "sum_sqrt_posterior_exp_entropy": 20.0,
        }
    ]
    discounted = exposure.scheduling_exposure(
        records, exponent=1.0, covariance_diagnostic=correlated_diagnostic
    )
    assert discounted == {
        "exponent": 1.0,
        "raw_exposure": 100.0,
        "correlation_discount": 2.0,
        "scheduled_exposure": 50.0,
        "completion_effective_hands_used": False,
        "is_independent_sample_count": False,
    }
    square_root = exposure.scheduling_exposure(records, exponent=0.5)
    assert square_root["raw_exposure"] == 20.0
    with pytest.raises(ValueError, match="repeated-gradient provenance"):
        exposure.scheduling_exposure(
            records,
            covariance_diagnostic={"conservative_scheduling_discount": 2.0},
        )


def test_advantage_variance_is_only_reported_as_dispersion() -> None:
    diagnostic = exposure.weighted_advantage_dispersion(
        torch.tensor([[0.25, 0.75]]),
        torch.tensor([[-1.0, 1.0]]),
    )
    assert diagnostic["mean_weighted_advantage_variance"] == pytest.approx(0.75)
    assert "not a correlation estimate" in diagnostic["interpretation"]
    assert "independent-hand" in diagnostic["interpretation"]


class _FakeMachine:
    combo_cards_value = _combos(range(48, 52))

    def combo_cards(self, _ids: torch.Tensor) -> torch.Tensor:
        return self.combo_cards_value

    def state_from_model_input(
        self, _input_type: str, model_input: torch.Tensor, _private: torch.Tensor
    ) -> SimpleNamespace:
        return SimpleNamespace(
            to_act=torch.zeros(model_input.shape[0], dtype=torch.int32),
            board_cards=_board(model_input.shape[0]),
        )


class _FakeHook:
    def __init__(self) -> None:
        self.geometry = SimpleNamespace(private_hands=6, players=2)
        self.owners = SimpleNamespace(device=torch.device("cpu"))
        self.machine = _FakeMachine()
        self.mode = SimpleNamespace(private_estimator=PrivateEstimator.RAO_BLACKWELL)

    def _resolved_model_context(self, _batch: object) -> tuple[torch.Tensor, None]:
        reach = torch.full((1, 2, 6), -torch.inf)
        reach[0, 0, :2] = math.log(0.5)
        reach[0, 1] = math.log(1.0 / 6.0)
        return reach, None

    def _forward_model_input(self, batch: object) -> torch.Tensor:
        return batch.fields["model_input"].reshape(-1, 6)  # type: ignore[attr-defined]


def test_legacy_decision_hook_remains_an_isolated_entropy_primitive() -> None:
    hook = _FakeHook()
    batch = SimpleNamespace(
        rows=1,
        lanes=torch.tensor([0]),
        input_type="pre",
        fields={
            "model_input": torch.zeros((1, 1, 6)),
            "true_profile": torch.zeros((1, 1, 2), dtype=torch.int32),
        },
    )
    accumulators = {}
    exposure._instrument_sequential_hook(hook, accumulators)
    hook._resolved_model_context(batch)
    assert float(accumulators[0]["pre"]["sum_posterior_exp_entropy"]) == pytest.approx(
        2.0
    )
