from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pbg.core.config import load_config
from pbg.eval import arena as arena_module
from pbg.eval.arena import arena
from pbg.games.poker.cards import hand_index, hand_cards
from pbg.games.poker.model.preflop import PolicyOutput


class _CheckCallModel(nn.Module):
    def forward(self, public, ranges, **kwargs):
        del kwargs
        batch = ranges.shape[0]
        type_logits = torch.tensor([-5.0, 5.0, -10.0]).expand(
            batch, 1326, 3
        ).clone()
        raw = {
            "type_logits": type_logits,
            "omega": torch.ones((batch, 1326, 1)),
            "mu": torch.full((batch, 1326, 1), 0.5),
            "s": torch.full((batch, 1326, 1), 0.2),
            "iota": torch.zeros((batch, 1326)),
        }
        return PolicyOutput(raw=raw, v=torch.zeros((batch, 1326)))


def _install_fake_sibling(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    package = ModuleType("poker_gpu_evaluator")
    facade = ModuleType("poker_gpu_evaluator.pbg")
    facade.API_VERSION = "pbg-v3"  # type: ignore[attr-defined]
    combo_table = torch.tensor(
        [hand_cards(combo) for combo in range(1326)], dtype=torch.uint8
    )
    facade.combo_cards = lambda rank_count: combo_table.clone()  # type: ignore[attr-defined]
    calls: dict[str, object] = {"showdowns": []}

    def sample_joint_profiles(**kwargs):
        calls["profiles"] = kwargs
        samples = kwargs["n_samples"]
        players = kwargs["dealt_mask"].shape[1]
        combos = [hand_index(2 * seat, 2 * seat + 1) for seat in range(players)]
        profile = torch.tensor(combos, dtype=torch.int32).view(1, 1, players)
        profile = profile.expand(1, samples, players).clone()
        return SimpleNamespace(
            combo_ids=profile,
            accepted=torch.ones((1, samples), dtype=torch.bool),
            attempt_count=torch.ones((1, samples), dtype=torch.int32),
            acceptance_rate=torch.ones(1),
            metadata={"backend": "fake"},
        )

    def evaluate_showdowns(**kwargs):
        calls["showdowns"].append(kwargs)
        eligible = kwargs["stratum_players"]
        shares = torch.zeros_like(eligible, dtype=torch.float32)
        winner_mask = torch.zeros(eligible.shape[:2], dtype=torch.int32)
        for batch in range(eligible.shape[0]):
            for stratum in range(eligible.shape[1]):
                winner = int(eligible[batch, stratum].nonzero()[0])
                shares[batch, stratum, winner] = 1.0
                winner_mask[batch, stratum] = 1 << winner
        return SimpleNamespace(
            winner_mask=winner_mask,
            showdown_share=shares,
            metadata={"backend": "fake"},
        )

    facade.sample_joint_profiles = sample_joint_profiles  # type: ignore[attr-defined]
    facade.evaluate_showdowns = evaluate_showdowns  # type: ignore[attr-defined]
    package.pbg = facade  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator", package)
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator.pbg", facade)
    return calls


def _poker_cfg():
    cfg = load_config("configs/nlhe6_h100.yaml")
    cfg.game.poker = {"n_players": 2, "stack_bb": 10.0, "rank_count": 13}
    cfg.model = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 4,
    }
    return cfg


@pytest.mark.parametrize("argmax_actions", [False, True])
def test_duplicate_self_play_is_zero_rotates_seats_and_crosses_showdown_facade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argmax_actions: bool,
) -> None:
    calls = _install_fake_sibling(monkeypatch)
    model = _CheckCallModel()
    monkeypatch.setattr(arena_module, "_load_model", lambda path, cfg: model)
    cfg = _poker_cfg()
    report = arena(
        tmp_path / "same.pt",
        tmp_path / "same.pt",
        cfg,
        4,
        torch.Generator().manual_seed(120),
        np.random.default_rng(120),
        argmax_actions=argmax_actions,
    )
    assert report.bb100_mean == 0.0
    assert report.bb100_ci95 == 0.0
    assert abs(report.bb100_mean) <= 3.0 * report.bb100_ci95
    assert report.per_seat_bb100 == [0.0, 0.0]
    assert report.n_hands == 4 and report.seat_pairing == "duplicate"
    # All four physical check-down hands are evaluator-settled, and the fake's
    # stipulated single-winner shares still pass local chip-conservation checks.
    assert len(calls["showdowns"]) == 4
    profile_call = calls["profiles"]
    assert profile_call["n_samples"] == 2


def test_real_sibling_combo_export_matches_every_local_row() -> None:
    facade = pytest.importorskip("poker_gpu_evaluator.pbg")
    combo_cards = facade.combo_cards(13)
    assert facade.API_VERSION == "pbg-v3"
    assert combo_cards.dtype == torch.uint8
    assert combo_cards.tolist() == [
        list(hand_cards(combo)) for combo in range(1326)
    ]
