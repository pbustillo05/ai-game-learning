"""Pure configurations for Kuhn, Leduc, and scalable synthetic games."""

from __future__ import annotations

from pbg.games.synthetic.family import SyntheticDialConfig


def kuhn() -> SyntheticDialConfig:
    """Return the standard two-player, three-card Kuhn configuration."""

    return SyntheticDialConfig(
        n_players=2,
        deck_size=3,
        n_suits=1,
        hand_size=1,
        n_rounds=1,
        board_per_round=(0,),
        max_raises=1,
        bet_size=(1,),
        ante=1,
    )


def leduc() -> SyntheticDialConfig:
    """Return the standard two-player, six-card limit Leduc configuration."""

    return SyntheticDialConfig(
        n_players=2,
        deck_size=6,
        n_suits=2,
        hand_size=1,
        n_rounds=2,
        board_per_round=(0, 1),
        max_raises=2,
        bet_size=(2, 4),
        ante=1,
    )


def dial(public_scale: int, private_scale: int, n_players: int) -> SyntheticDialConfig:
    """Map positive integer dials to independently growing public/private games.

    ``public_scale`` adds betting rounds and raises. ``private_scale`` increases
    the private hand size every other step and always expands the deck. Dial
    rounds reveal no board cards, so changing the public dial does not change
    the private type count.
    """

    if public_scale < 1 or private_scale < 1:
        raise ValueError("public_scale and private_scale must be positive")
    if n_players < 2:
        raise ValueError("n_players must be at least 2")
    n_rounds = public_scale
    board_per_round = (0,) * n_rounds
    hand_size = 1 + (private_scale - 1) // 2
    minimum_deck = n_players * hand_size + 1
    target_deck = minimum_deck + 2 * private_scale
    deck_size = target_deck + target_deck % 2
    return SyntheticDialConfig(
        n_players=n_players,
        deck_size=deck_size,
        n_suits=2,
        hand_size=hand_size,
        n_rounds=n_rounds,
        board_per_round=board_per_round,
        max_raises=public_scale,
        bet_size=tuple(2 ** index for index in range(n_rounds)),
        ante=1,
    )
