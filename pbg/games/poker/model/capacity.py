"""Static lossless-history and attention-capacity bounds for NLHE."""

from __future__ import annotations

import math

from pbg.games.poker.cards import DeckGeometry


STREET_SEPARATOR_COUNT = 3


def max_player_nodes(n_players: int, stack_bb: float) -> int:
    """Return the conservative documented maximum player-node count."""

    if isinstance(n_players, bool) or not isinstance(n_players, int) or n_players < 2:
        raise ValueError("n_players must be an integer at least two")
    if not math.isfinite(stack_bb) or stack_bb <= 0:
        raise ValueError("stack_bb must be finite and positive")
    return math.ceil(stack_bb * n_players + 3 * n_players - stack_bb + 1)


def max_public_history_tokens(n_players: int, stack_bb: float) -> int:
    """Bound lossless player actions plus the three street separators."""

    return max_player_nodes(n_players, stack_bb) + STREET_SEPARATOR_COUNT


def max_policy_sequence_tokens(
    *,
    n_players: int,
    stack_bb: float,
    rank_count: int,
    k_act: int,
) -> int:
    """Return the widest legal one-row pre/postflop transformer sequence."""

    geometry = DeckGeometry(rank_count)
    history = max_public_history_tokens(n_players, stack_bb)
    n_ticks = round(stack_bb / 0.5)
    event_count = 3 + n_ticks
    scalar_count = 3 * n_players + 13 + event_count + n_ticks
    preflop = n_players * geometry.n_classes + scalar_count + 1 + history + k_act
    flop_hands = math.comb(geometry.n_cards - 3, 2)
    postflop = n_players * flop_hands + scalar_count + 6 + history + k_act
    return max(preflop, postflop)


def max_policy_partition_sequence_tokens(
    *,
    n_players: int,
    stack_bb: float,
    rank_count: int,
    k_act: int,
) -> dict[str, int]:
    """Return conservative one-row bounds for each sampled input partition."""

    geometry = DeckGeometry(rank_count)
    history = max_public_history_tokens(n_players, stack_bb)
    n_ticks = round(stack_bb / 0.5)
    event_count = 3 + n_ticks
    scalar_count = 3 * n_players + 13 + event_count + n_ticks
    result = {
        "pre": (
            n_players * geometry.n_classes
            + scalar_count
            + 1
            + history
            + k_act
        )
    }
    flop_hands = math.comb(geometry.n_cards - 3, 2)
    for live_players in range(2, n_players + 1):
        result[f"post_{live_players}"] = (
            live_players * flop_hands
            + scalar_count
            + 6
            + history
            + k_act
        )
    return result
