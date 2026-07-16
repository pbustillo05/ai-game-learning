from __future__ import annotations

import math

import pytest
import torch

from pbg.games.poker.cards import DeckGeometry, category_profile
from pbg.games.poker.engine import PokerConfig
from pbg.games.poker.bundle import estimate_rollout_memory
from pbg.games.poker.rollout_backend import PokerRolloutBackend


EXPECTED_ORDERS = {
    4: ("quads", "full_house", "trips", "two_pair", "pair", "high_card"),
    5: (
        "straight_flush", "flush", "quads", "full_house", "straight",
        "trips", "two_pair", "pair", "high_card",
    ),
    6: (
        "straight_flush", "flush", "quads", "full_house", "straight",
        "trips", "two_pair", "pair", "high_card",
    ),
    7: (
        "straight_flush", "flush", "quads", "full_house", "straight",
        "trips", "two_pair", "pair", "high_card",
    ),
    **{
        rank_count: (
            "straight_flush", "quads", "flush", "full_house", "straight",
            "trips", "two_pair", "pair", "high_card",
        )
        for rank_count in range(8, 12)
    },
    **{
        rank_count: (
            "straight_flush", "quads", "full_house", "flush", "straight",
            "trips", "two_pair", "pair", "high_card",
        )
        for rank_count in (12, 13)
    },
}


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_compact_geometry_is_a_bijection(rank_count: int) -> None:
    geometry = DeckGeometry(rank_count)
    assert geometry.n_cards == 4 * rank_count
    assert geometry.n_hands == math.comb(4 * rank_count, 2)
    assert geometry.n_classes == rank_count**2
    assert geometry.active_cards == tuple(range(52 - 4 * rank_count, 52))
    assert len(set(geometry.combo_cards)) == geometry.n_hands
    assert sorted(hand for members in geometry.class_memberships for hand in members) == list(
        range(geometry.n_hands)
    )
    for hand, cards in enumerate(geometry.combo_cards):
        assert geometry.hand_index(*cards) == hand
        assert geometry.hand_cards(hand) == cards


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_category_counts_and_orders(rank_count: int) -> None:
    profile = category_profile(rank_count)
    r = rank_count
    total = math.comb(4 * r, 5)
    sequences = max(0, r - 4) + int(r == 13)
    choose_r_5 = math.comb(r, 5) if r >= 5 else 0
    assert profile.denominator == total
    assert profile.numerators == {
        "high_card": total,
        "pair": total - 4**5 * choose_r_5,
        "two_pair": 144 * (r - 2) * math.comb(r, 2) + 24 * r * (r - 1),
        "trips": 64 * r * math.comb(r - 1, 2) + 28 * r * (r - 1),
        "straight": 4**5 * sequences,
        "flush": 4 * choose_r_5,
        "full_house": 24 * r * (r - 1),
        "quads": 4 * r * (r - 1),
        "straight_flush": 4 * sequences,
    }
    assert profile.strength_order == EXPECTED_ORDERS[rank_count]


def test_documented_probability_degeneracies() -> None:
    four = category_profile(4)
    assert four.probability("high_card") == 1
    assert four.probability("pair") == 1
    assert four.numerators["straight"] == four.numerators["flush"] == 0

    five = category_profile(5)
    assert five.numerators["flush"] == five.numerators["straight_flush"]

    eight = category_profile(8)
    assert eight.numerators["flush"] == eight.numerators["quads"] == 224

    twelve = category_profile(12)
    assert twelve.numerators["flush"] == twelve.numerators["full_house"] == 3168


def test_board_conflicts_reject_inactive_cards() -> None:
    geometry = DeckGeometry(4)
    board = torch.tensor([36, 40, 44, -1, -1], dtype=torch.int16)
    mask = geometry.board_conflict_mask(board)
    assert mask.shape == (120,)
    assert int(mask.sum()) == 42
    with pytest.raises(ValueError, match="outside the active deck"):
        geometry.board_conflict_mask(torch.tensor([35, -1, -1, -1, -1]))


@pytest.mark.parametrize("players,stack,ranks", [(2, 8.0, 4), (6, 8.0, 5), (6, 128.0, 13)])
def test_supported_configuration_corners(players: int, stack: float, ranks: int) -> None:
    config = PokerConfig(n_players=players, stack_bb=stack, rank_count=ranks)
    assert config.geometry.rank_count == ranks
    assert config.n_ticks == round(2 * stack)
    assert (config.sb, config.bb, config.chip_unit_bb) == (0.5, 1.0, 0.5)


@pytest.mark.parametrize(
    "values,match",
    [
        ({"n_players": 1}, "n_players"),
        ({"n_players": 7}, "n_players"),
        ({"stack_bb": 7.5}, "stack_bb"),
        ({"stack_bb": 128.5}, "stack_bb"),
        ({"stack_bb": 8.25}, "multiple of 0.5"),
        ({"rank_count": 3}, "rank_count"),
        ({"rank_count": 14}, "rank_count"),
        ({"n_players": 6, "rank_count": 4}, "2\\*n_players"),
    ],
)
def test_configuration_rejects_out_of_contract_values(
    values: dict[str, int | float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        PokerConfig(**values)


def test_rollout_memory_uses_compact_private_width() -> None:
    short = estimate_rollout_memory(
        n_trees=2, node_budget=8, n_players=2, reach_storage="full", rank_count=4
    )
    full = estimate_rollout_memory(
        n_trees=2, node_budget=8, n_players=2, reach_storage="full", rank_count=13
    )
    assert short.reach_cache_bytes * 1326 == full.reach_cache_bytes * 120
    assert short.frontier_reach_bytes * 1326 == full.frontier_reach_bytes * 120


def test_reduced_deck_rollout_fails_loudly_until_pbg_v3_exists() -> None:
    with pytest.raises(RuntimeError, match="pbg-v3"):
        PokerRolloutBackend(PokerConfig(n_players=2, stack_bb=8, rank_count=4), 4, "cpu")
