from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pbg.core.config import load_config
from pbg.core.training import build_game_bundle
from pbg.eval import arena as arena_module
from pbg.eval.arena import arena
from pbg.eval.exploitability import exploitability_curve, exploitability_of_checkpoint
from pbg.games.poker.cards import hand_index, hand_cards
from pbg.games.poker.model.preflop import PolicyOutput
from pbg.games.synthetic.exact import exploitability, uniform_policy_tables


def _synthetic_checkpoint(path: Path, iteration: int) -> object:
    cfg = load_config("configs/kuhn_smoke.yaml")
    torch.manual_seed(0)
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    torch.save(
        {"model": bundle.model.state_dict(), "iteration": iteration}, path
    )
    return cfg


def test_fresh_kuhn_checkpoint_agrees_with_uniform_and_curve_is_sorted(
    tmp_path: Path,
) -> None:
    cfg = _synthetic_checkpoint(tmp_path / "z.pt", 9)
    _synthetic_checkpoint(tmp_path / "a.pt", 2)
    _synthetic_checkpoint(tmp_path / "m.pt", 5)
    cfg.train.out_dir = str(tmp_path / "out")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    game = bundle.rollout_backend.game
    uniform = exploitability(game, uniform_policy_tables(game))
    observed = exploitability_of_checkpoint(tmp_path / "z.pt", cfg)
    assert observed == pytest.approx(uniform, rel=0.10)

    rows = exploitability_curve(cfg, tmp_path / "*.pt")
    assert [row["iter"] for row in rows] == [2, 5, 9]
    written = [
        json.loads(line)
        for line in (tmp_path / "out" / "exploitability.jsonl").read_text().splitlines()
    ]
    assert written == rows


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
    facade.API_VERSION = "pbg-v2"  # type: ignore[attr-defined]
    facade.COMBO_CARDS = torch.tensor(  # type: ignore[attr-defined]
        [hand_cards(combo) for combo in range(1326)], dtype=torch.uint8
    )
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
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.game.name = "poker"
    cfg.game.synthetic = None
    cfg.game.poker = {
        "n_players": 2,
        "stack_bb": 10.0,
        "sb": 0.5,
        "bb": 1.0,
        "n_ticks": 1,
    }
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
    assert facade.COMBO_CARDS.dtype == torch.uint8
    assert facade.COMBO_CARDS.tolist() == [
        list(hand_cards(combo)) for combo in range(1326)
    ]
