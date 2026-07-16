from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from pbg.config import ConfigError, load_config
from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch
from pbg.games.poker.bundle import build_bundle
from pbg.games.poker.engine import CHECK_CALL, LIVE, RAISE_OFFSET, PokerConfig
from pbg.games.poker.model.capacity import (
    max_player_nodes,
    max_policy_sequence_tokens,
    max_public_history_tokens,
)
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig
from pbg.games.poker.rollout_backend import PokerRolloutBackend


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("players", range(2, 7))
@pytest.mark.parametrize("stack", (8.0, 8.5, 128.0))
def test_lossless_history_bound_covers_documented_player_nodes_and_separators(
    players: int, stack: float
) -> None:
    expected_nodes = math.ceil(stack * players + 3 * players - stack + 1)
    assert max_player_nodes(players, stack) == expected_nodes
    assert max_public_history_tokens(players, stack) == expected_nodes + 3


def test_omitted_history_capacity_is_derived_before_model_allocation(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["completion"]["backend"] = "cpu"
    raw["rollout"]["node_budget"] = 8
    raw["model"].update(
        {
            "d_model": 8,
            "n_heads": 2,
            "n_layers_pre": 1,
            "n_layers_post": 1,
            "m_raise": 1,
            "k_act": 4,
            "class_embedding_dim": 4,
        }
    )
    path = tmp_path / "derived_history.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    bundle = build_bundle(cfg, torch.device("cpu"))
    expected = max_public_history_tokens(6, 100.0)
    assert expected == 522
    assert bundle.model.preflop.cfg.max_history_tokens == expected
    assert bundle.model.postflop.cfg.max_history_tokens == expected
    assert bundle.rollout_backend.max_rollout_steps == expected


def test_explicitly_small_history_capacity_is_a_dotted_load_error(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["model"]["max_history_tokens"] = 521
    path = tmp_path / "short_history.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"root\.model\.max_history_tokens"):
        load_config(path)


def test_worst_history_attention_footprint_is_checked_against_profile(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["game"]["stack_bb"] = 128.0
    path = tmp_path / "deep.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    width = max_policy_sequence_tokens(
        n_players=6, stack_bb=128.0, rank_count=13, k_act=7
    )
    assert width > 8192
    with pytest.raises(ConfigError, match=r"hardware\.training_max_tokens"):
        load_config(path, hardware_profile="local_3x2080ti")


@pytest.mark.parametrize("players", range(2, 7))
@pytest.mark.parametrize("stack", (8.0, 128.0))
def test_adversarial_minimum_raise_trajectory_reaches_exact_history_bound(
    players: int, stack: float
) -> None:
    """Exercise the legal trajectory attaining the documented maximum.

    The last player who could close a round makes the smallest full raise until
    one quarter of the stack is committed on that street. Everyone else calls.
    This distributes the minimum-raise chain over all four streets and attains
    the exact player-node-plus-separator bound.
    """

    game_cfg = PokerConfig(n_players=players, stack_bb=stack, rank_count=13)
    bound = max_public_history_tokens(players, stack)
    backend = PokerRolloutBackend(
        game_cfg,
        bound + 1,
        "cpu",
        max_rollout_steps=bound,
    )
    state = backend.sequential_start_states(1)
    used_cards: list[int] = []
    street_cap = stack / 4.0

    while True:
        owner = int(backend.classify(state)[0])
        if owner in {PRETERM, TERMINAL}:
            break
        if owner == CHANCE:
            street = int(backend.states.street[state.long()][0])
            card_count = 3 if street == 0 else 1
            dealt = [
                card
                for card in game_cfg.geometry.active_cards
                if card not in used_cards
            ][:card_count]
            used_cards.extend(dealt)
            cards = torch.tensor(
                [dealt + [-1] * (3 - card_count)], dtype=torch.int16
            )
            state = backend.step(
                state,
                EventBatch(torch.zeros(1, dtype=torch.int32), {"cards": cards}),
            )
            continue

        row = state.long()
        actor = int(backend.states.actor[row][0])
        status = backend.states.status[row][0]
        acted = backend.states.acted[row][0]
        committed = backend.states.committed_street[row][0]
        current_bet = backend.states.current_bet[row][0]
        other_needs_action = (status == LIVE) & (
            (~acted) | (committed + 1.0e-6 < current_bet)
        )
        other_needs_action[actor] = False
        event_ids, legal = backend.legal_events(state)
        raises = event_ids[0, legal[0] & (event_ids[0] >= RAISE_OFFSET)]
        smallest_target = (
            float(backend.raise_targets(state)[0, int(raises[0] - RAISE_OFFSET)])
            if raises.numel()
            else float("inf")
        )
        event = (
            int(raises[0])
            if raises.numel()
            and not bool(other_needs_action.any())
            and smallest_target <= street_cap
            else CHECK_CALL
        )
        state = backend.step(
            state, EventBatch(torch.tensor([event], dtype=torch.int32))
        )

    history = backend.encode_public(state).history
    assert history is not None
    assert int(backend.states.history_length[state.long()][0]) == bound
    assert history.mask.shape == (1, bound)
    assert bool(history.mask.all())

    model_cfg = PokerModelConfig(
        d_model=8,
        n_heads=2,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=1,
        k_act=4,
        max_players=players,
        class_embedding_dim=4,
        max_history_tokens=bound,
    )
    postflop = PostflopNet(model_cfg)
    public = backend.encode_public(state)
    relative = torch.arange(players)[None]
    tokens, mask = postflop._public_tokens(public, None, torch.float32, relative)
    assert tokens.shape[:2] == mask.shape
    assert int(mask.sum()) >= bound
