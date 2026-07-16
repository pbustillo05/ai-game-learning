from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from pbg.games.poker.engine import ALL_IN, FOLDED, LIVE, NLHEEngine, PokerConfig, _State


@dataclass
class _CompletionResult:
    p_hat: torch.Tensor
    v_hat: torch.Tensor
    vvec_hat: torch.Tensor
    u_hat: torch.Tensor
    n_eff: torch.Tensor
    equity_share: torch.Tensor
    output_valid: torch.Tensor
    sample_count: torch.Tensor
    estimator_stddev: torch.Tensor
    metadata: dict[str, object]


@dataclass
class _JointResult:
    combo_ids: torch.Tensor
    accepted: torch.Tensor
    attempt_count: torch.Tensor
    acceptance_rate: torch.Tensor
    metadata: dict[str, object]


@dataclass
class _ShowdownResult:
    winner_mask: torch.Tensor
    showdown_share: torch.Tensor
    metadata: dict[str, object]


def _install_fake(
    monkeypatch: pytest.MonkeyPatch, *, version: str = "pbg-v3"
) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    package = ModuleType("poker_gpu_evaluator")
    package.__path__ = []  # type: ignore[attr-defined]
    facade = ModuleType("poker_gpu_evaluator.pbg")
    facade.API_VERSION = version  # type: ignore[attr-defined]
    combo_table = torch.triu_indices(52, 52, offset=1).T.to(torch.uint8)
    facade.combo_cards = lambda rank_count: combo_table.clone()  # type: ignore[attr-defined]

    def complete(**kwargs: Any) -> _CompletionResult:
        calls["complete"] = kwargs
        board = kwargs["board_cards"]
        batch, players = kwargs["range_weights"].shape[:2]
        strata = kwargs["stratum_amount"].shape[1]
        device = board.device
        return _CompletionResult(
            p_hat=torch.full((batch, players, 1326), 1 / 1326, device=device),
            v_hat=torch.arange(batch * players * 1326, dtype=torch.float32, device=device).reshape(batch, players, 1326),
            vvec_hat=torch.arange(batch * players, dtype=torch.float32, device=device).reshape(batch, players),
            u_hat=torch.arange(batch, dtype=torch.float32, device=device),
            n_eff=torch.full((batch,), 7.0, device=device),
            equity_share=torch.zeros((batch, players, 1326, strata), device=device),
            output_valid=torch.ones((batch, players, 1326), dtype=torch.bool, device=device),
            sample_count=torch.full((batch,), kwargs["config"].n_samples, dtype=torch.int64, device=device),
            estimator_stddev=torch.zeros((batch, players, 1326), device=device),
            metadata={"backend": "fake", "api_version": "pbg-v3"},
        )

    def sample_joint_profiles(**kwargs: Any) -> _JointResult:
        calls["sample"] = kwargs
        batch, players = kwargs["dealt_mask"].shape
        samples = kwargs["n_samples"]
        device = kwargs["range_weights"].device
        profiles = torch.full((batch, samples, players), -1, dtype=torch.int32, device=device)
        for seat in range(players):
            profiles[:, :, seat] = torch.where(
                kwargs["dealt_mask"][:, seat, None],
                torch.full((batch, samples), 10 + seat, dtype=torch.int32, device=device),
                profiles[:, :, seat],
            )
        return _JointResult(
            combo_ids=profiles,
            accepted=torch.ones((batch, samples), dtype=torch.bool, device=device),
            attempt_count=torch.ones((batch, samples), dtype=torch.int32, device=device),
            acceptance_rate=torch.ones((batch,), dtype=torch.float32, device=device),
            metadata={"backend": "fake"},
        )

    def evaluate_showdowns(**kwargs: Any) -> _ShowdownResult:
        calls["showdown"] = kwargs
        players = kwargs["combo_ids"].shape[1]
        device = kwargs["combo_ids"].device
        shares = torch.tensor(
            [[[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]],
            dtype=torch.float32,
            device=device,
        )
        assert players == 3
        return _ShowdownResult(
            winner_mask=torch.tensor([[3, 2]], dtype=torch.int32, device=device),
            showdown_share=shares,
            metadata={"backend": "fake"},
        )

    facade.complete = complete  # type: ignore[attr-defined]
    facade.sample_joint_profiles = sample_joint_profiles  # type: ignore[attr-defined]
    facade.evaluate_showdowns = evaluate_showdowns  # type: ignore[attr-defined]
    package.pbg = facade  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator", package)
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator.pbg", facade)
    return calls


def _preterm_engine() -> tuple[NLHEEngine, int]:
    game = NLHEEngine(PokerConfig(n_players=3))
    ref = game.states.append(
        _State(
            street=1,
            board=(0, 5, 9),
            stacks=(90.0, 80.0, 80.0),
            committed_street=(10.0, 20.0, 20.0),
            committed_total=(10.0, 20.0, 20.0),
            status=(FOLDED, LIVE, ALL_IN),
            actor=-1,
            last_full_raise=10.0,
            current_bet=20.0,
            acted=(True, True, True),
            button=0,
            phase=3,
            betting_closed=True,
            n_raises=1,
            action_history=(),
        )
    )
    return game, ref


def test_completion_exact_mapping_padding_seeds_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "poker_gpu_evaluator", raising=False)
    monkeypatch.delitem(sys.modules, "poker_gpu_evaluator.pbg", raising=False)
    module = importlib.reload(importlib.import_module("pbg.games.poker.mc"))
    assert "poker_gpu_evaluator" not in sys.modules
    calls = _install_fake(monkeypatch)
    game, ref = _preterm_engine()
    tree = SimpleNamespace(state_ref=torch.tensor([ref], dtype=torch.int32))
    log_r = torch.full((1, 4, 1326), -torch.inf, dtype=torch.float32)
    log_r[0, 0] = torch.linspace(-100.0, -90.0, 1326)
    log_r[0, 1] = -3.0
    log_r[0, 2] = torch.linspace(-1.0, 0.0, 1326)
    gen = torch.Generator().manual_seed(123)
    expected_gen = torch.Generator().manual_seed(123)
    expected_seed = torch.empty(1, dtype=torch.int64).random_(generator=expected_gen)

    completion = module.PokerGPUCompletion(backend="cpu", n_partials=4)
    result = completion.complete(
        game,
        tree,
        torch.tensor([0], dtype=torch.int32),
        module.ReachView(log_r),
        17,
        gen,
    )
    call = calls["complete"]
    assert set(call) == {
        "board_cards",
        "range_weights",
        "dealt_mask",
        "contender_mask",
        "stratum_players",
        "payoff_baseline",
        "stratum_amount",
        "seed",
        "config",
        "rank_count",
    }
    assert call["rank_count"] == 13
    assert call["board_cards"].tolist() == [[0, 5, 9, 255, 255]]
    assert call["board_cards"].dtype == torch.uint8 and call["board_cards"].is_contiguous()
    assert call["dealt_mask"].tolist() == [[True, True, True, False]]
    assert call["contender_mask"].tolist() == [[False, True, True, False]]
    assert call["stratum_amount"].tolist() == [[30.0, 20.0]]
    assert call["stratum_players"].tolist() == [
        [[False, True, True, False], [False, True, True, False]]
    ]
    assert call["payoff_baseline"].tolist() == [[-10.0, -20.0, -20.0, 0.0]]
    assert torch.equal(call["seed"], expected_seed)
    assert call["config"].n_samples == 17
    assert call["config"].n_partials == 4
    assert call["config"].backend == "cpu"
    weights = call["range_weights"]
    assert weights.dtype == torch.float32 and weights.is_contiguous()
    assert torch.equal(weights.amax(dim=-1), torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
    assert result.p_hat.shape == (1, 4, 1326)
    assert result.equity_share is not None and result.equity_share.shape == (1, 4, 1326, 2)
    assert result.sample_count is not None and result.sample_count.item() == 17
    assert result.metadata == {"backend": "fake", "api_version": "pbg-v3"}

    logical_seed = torch.tensor([987654321], dtype=torch.int64)
    completion.complete_keyed(
        game,
        tree,
        torch.tensor([0], dtype=torch.int32),
        module.ReachView(log_r),
        17,
        logical_seed,
    )
    assert torch.equal(calls["complete"]["seed"], logical_seed)


def test_rejection_returns_whole_profiles_and_reports_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake(monkeypatch)
    from pbg.games.poker.mc import AcceptanceError, PokerGPURejectionSampler

    rows = torch.zeros((2, 4, 1326), dtype=torch.float32)
    board = torch.tensor([[0, 4, 8, -1, -1], [-1, -1, -1, -1, -1]], dtype=torch.int16)
    dealt = torch.tensor([[True, False, True, True], [True, True, False, False]])
    profiles = PokerGPURejectionSampler(max_attempts=91).sample(
        rows, board, dealt, 3, torch.Generator().manual_seed(9)
    )
    assert profiles.shape == (2, 3, 4)
    assert torch.equal(profiles == -1, (~dealt)[:, None, :].expand_as(profiles))
    call = calls["sample"]
    assert set(call) == {
        "board_cards",
        "range_weights",
        "dealt_mask",
        "n_samples",
        "seed",
        "max_attempts",
        "backend",
        "rank_count",
    }
    assert call["rank_count"] == 13
    assert call["max_attempts"] == 91
    assert call["board_cards"].tolist()[1] == [255] * 5

    logical_seeds = torch.tensor([1234, 5678], dtype=torch.int64)
    seeded = PokerGPURejectionSampler(max_attempts=92).sample_seeded(
        rows, board, dealt, 3, logical_seeds
    )
    assert seeded.shape == profiles.shape
    assert torch.equal(calls["sample"]["seed"], logical_seeds)
    assert calls["sample"]["max_attempts"] == 92

    facade = sys.modules["poker_gpu_evaluator.pbg"]

    original = facade.sample_joint_profiles

    def partial(**kwargs: Any) -> _JointResult:
        result = original(**kwargs)
        result.accepted[1, 2] = False
        result.combo_ids[1, 2] = -1
        result.attempt_count[1, 2] = kwargs["max_attempts"]
        result.acceptance_rate[1] = 0.25
        return result

    facade.sample_joint_profiles = partial  # type: ignore[attr-defined]
    # The adapter caches the facade callable per instance, so use a fresh one.
    try:
        with pytest.raises(AcceptanceError) as exc_info:
            PokerGPURejectionSampler(max_attempts=7).sample(
                rows, board, dealt, 3, torch.Generator().manual_seed(9)
            )
    finally:
        facade.sample_joint_profiles = original  # type: ignore[attr-defined]
    assert exc_info.value.failed_indices.tolist() == [[1, 2]]
    assert exc_info.value.attempt_count[1, 2].item() == 7
    assert exc_info.value.acceptance_rate.tolist() == [1.0, 0.25]


def test_showdown_stipulated_strata_validate_and_assemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake(monkeypatch)
    from pbg.games.poker.engine import PotStrata, assemble_showdown_payoff
    from pbg.games.poker.mc import PokerGPUShowdownEvaluator

    board = torch.tensor([[0, 4, 8, 12, 16]], dtype=torch.int16)
    combos = torch.tensor([[20, 300, -1]], dtype=torch.int32)
    active = torch.tensor([[True, True, False]])
    eligible = torch.tensor([[[True, True, False], [False, True, False]]])
    result = PokerGPUShowdownEvaluator(backend="cpu").evaluate(
        board_cards=board,
        combo_ids=combos,
        active_mask=active,
        stratum_players=eligible,
    )
    assert result.winner_mask.tolist() == [[3, 2]]
    assert result.showdown_share.tolist() == [[[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]]
    call = calls["showdown"]
    assert set(call) == {
        "board_cards",
        "combo_ids",
        "active_mask",
        "stratum_players",
        "backend",
        "rank_count",
    }
    assert call["rank_count"] == 13
    strata = PotStrata(
        stratum_players=eligible,
        stratum_amount=torch.tensor([[30.0, 10.0]]),
        payoff_baseline=torch.tensor([[-10.0, -20.0, -10.0]]),
    )
    assert torch.equal(
        assemble_showdown_payoff(strata, result.showdown_share),
        torch.tensor([[5.0, 5.0, -10.0]]),
    )


def test_version_result_contract_and_forbidden_implementation_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake(monkeypatch, version="pbg-v1")
    from pbg.games.poker.mc import PokerGPUShowdownEvaluator

    with pytest.raises(RuntimeError, match="expected API_VERSION 'pbg-v3'"):
        PokerGPUShowdownEvaluator().evaluate(
            board_cards=torch.tensor([[0, 4, 8, 12, 16]], dtype=torch.uint8),
            combo_ids=torch.tensor([[1, 2]], dtype=torch.int32),
            active_mask=torch.tensor([[True, True]]),
            stratum_players=torch.tensor([[[True, True]]]),
        )

    source = (Path(__file__).parents[1] / "pbg/games/poker/mc.py").read_text().lower()
    forbidden = ("evaluate7", "hand_rank", "batch_eval", "load_inline", "cpp_extension")
    assert not any(token in source for token in forbidden)


def test_real_sibling_facade_version_when_installed() -> None:
    facade = pytest.importorskip(
        "poker_gpu_evaluator.pbg",
        reason="poker_gpu_evaluator pbg-v3 facade is not installed",
    )
    assert facade.API_VERSION == "pbg-v3"
    expected = torch.triu_indices(52, 52, offset=1).T.to(torch.uint8)
    assert torch.equal(facade.combo_cards(13), expected)
