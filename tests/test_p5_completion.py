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


@dataclass(frozen=True)
class _Info:
    contract_version: str = "1"
    package_version: str = "test"
    build_revision: str = "fake"
    rank_count: int = 13
    deck_geometry_version: str = "global-suffix-lex-v1"
    category_order_version: str = "cumulative-five-card-v1"
    folder_feature_version: str = "compact-folder-v1"
    cpu_rng_version: str = "cpu-test"
    cuda_rng_version: str = "cuda-test"
    mps_rng_version: str = "mps-test"


class _FakeEvaluator:
    def __init__(self, calls: dict[str, dict[str, Any]]) -> None:
        self.calls = calls
        self.info = _Info()
        self.combo_cards = torch.triu_indices(52, 52, offset=1).T.to(torch.uint8)

    def complete(self, **kwargs: Any) -> Any:
        self.calls["complete"] = kwargs
        batch, players, hands = kwargs["range_weights"].shape
        device = kwargs["range_weights"].device
        return SimpleNamespace(
            p_hat=torch.full((batch, players, hands), 1 / hands, device=device),
            v_hat=torch.arange(
                batch * players * hands, dtype=torch.float32, device=device
            ).reshape(batch, players, hands),
            n_eff=torch.full((batch,), 7.0, device=device),
        )

    def sample_joint_profiles(self, **kwargs: Any) -> torch.Tensor:
        self.calls["sample"] = kwargs
        ranges = kwargs["range_weights"]
        batch, players = ranges.shape[:2]
        dealt = ranges.sum(2) > 0
        output = torch.full(
            (batch, kwargs["n_samples"], players),
            -1,
            dtype=torch.int32,
            device=ranges.device,
        )
        for seat in range(players):
            output[:, :, seat] = torch.where(
                dealt[:, seat, None],
                torch.full_like(output[:, :, seat], 10 + seat),
                output[:, :, seat],
            )
        return output

    def evaluate_showdowns(self, **kwargs: Any) -> torch.Tensor:
        self.calls["showdown"] = kwargs
        return torch.tensor(
            [[5.0, 5.0, -10.0]],
            dtype=torch.float32,
            device=kwargs["combo_ids"].device,
        )


def _install_fake(
    monkeypatch: pytest.MonkeyPatch, *, version: str = "1"
) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    package = ModuleType("poker_gpu_evaluator")
    package.CONTRACT_VERSION = version  # type: ignore[attr-defined]
    package.bind = lambda *, rank_count: _FakeEvaluator(calls)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator", package)
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


def test_completion_exact_bound_mapping_padding_and_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "poker_gpu_evaluator", raising=False)
    module = importlib.reload(importlib.import_module("pbg.games.poker.mc"))
    assert "poker_gpu_evaluator" not in sys.modules
    calls = _install_fake(monkeypatch)
    game, ref = _preterm_engine()
    tree = SimpleNamespace(state_ref=torch.tensor([ref], dtype=torch.int32))
    log_r = torch.full((1, 4, 1326), -torch.inf, dtype=torch.float32)
    log_r[0, 0] = torch.linspace(-100.0, -90.0, 1326)
    log_r[0, 1] = -3.0
    log_r[0, 2] = torch.linspace(-1.0, 0.0, 1326)
    expected_gen = torch.Generator().manual_seed(123)
    expected_seed = torch.empty(1, dtype=torch.int64).random_(generator=expected_gen)

    result = module.PokerGPUCompletion(backend="cpu").complete(
        game,
        tree,
        torch.tensor([0], dtype=torch.int32),
        module.ReachView(log_r),
        17,
        torch.Generator().manual_seed(123),
    )
    call = calls["complete"]
    assert set(call) == {
        "board_cards", "range_weights", "stratum_level", "payoff_baseline",
        "stratum_amount", "n_samples", "seed",
    }
    assert call["board_cards"].tolist() == [[0, 5, 9, 255, 255]]
    assert call["stratum_level"].tolist() == [[-1, 1, 1, -1]]
    assert call["payoff_baseline"].tolist() == [[-10.0, -20.0, -20.0, 0.0]]
    assert call["stratum_amount"].tolist() == [[30.0, 20.0]]
    assert call["n_samples"] == 17
    assert torch.equal(call["seed"], expected_seed)
    assert torch.equal(
        call["range_weights"].amax(dim=-1), torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    )
    assert result.p_hat.shape == (1, 4, 1326)
    assert result.n_eff is not None and result.n_eff.item() == 7
    assert result.metadata["contract_version"] == "1"


def test_joint_profiles_use_zero_ranges_for_undealt_seats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake(monkeypatch)
    from pbg.games.poker.mc import PokerGPURejectionSampler

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
        "board_cards", "range_weights", "n_samples", "seed", "max_attempts"
    }
    assert call["max_attempts"] == 91
    assert torch.equal(call["range_weights"].sum(2) > 0, dealt)


def test_showdown_passes_stratum_levels_and_returns_payoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake(monkeypatch)
    from pbg.games.poker.mc import PokerGPUShowdownEvaluator

    result = PokerGPUShowdownEvaluator(backend="cpu").evaluate(
        board_cards=torch.tensor([[0, 4, 8, 12, 16]], dtype=torch.int16),
        combo_ids=torch.tensor([[20, 300, -1]], dtype=torch.int32),
        stratum_level=torch.tensor([[0, 1, -1]], dtype=torch.int32),
        payoff_baseline=torch.tensor([[-10.0, -20.0, -10.0]]),
        stratum_amount=torch.tensor([[30.0, 10.0]]),
    )
    assert result.payoff.tolist() == [[5.0, 5.0, -10.0]]
    assert set(calls["showdown"]) == {
        "board_cards", "combo_ids", "stratum_level", "payoff_baseline",
        "stratum_amount",
    }


def test_version_contract_and_forbidden_implementation_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake(monkeypatch, version="0")
    from pbg.games.poker.mc import PokerGPUShowdownEvaluator

    with pytest.raises(RuntimeError, match="expected CONTRACT_VERSION '1'"):
        PokerGPUShowdownEvaluator().evaluate(
            board_cards=torch.tensor([[0, 4, 8, 12, 16]], dtype=torch.uint8),
            combo_ids=torch.tensor([[1, 2]], dtype=torch.int32),
            stratum_level=torch.tensor([[0, 0]], dtype=torch.int32),
            payoff_baseline=torch.tensor([[-1.0, -1.0]]),
            stratum_amount=torch.tensor([[2.0]]),
        )
    source = (Path(__file__).parents[1] / "pbg/games/poker/mc.py").read_text().lower()
    assert not any(token in source for token in ("evaluate7", "hand_rank", "load_inline"))


def test_real_sibling_bound_contract_when_installed() -> None:
    package = pytest.importorskip("poker_gpu_evaluator")
    assert package.CONTRACT_VERSION == "1"
    evaluator = package.bind(rank_count=13)
    expected = torch.triu_indices(52, 52, offset=1).T.to(torch.uint8)
    assert torch.equal(evaluator.combo_cards, expected)
