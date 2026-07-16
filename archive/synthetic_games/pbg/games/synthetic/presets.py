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


def dial(card_scale: int, action_scale: int) -> SyntheticDialConfig:
    """Return one point on the two-axis, two-player poker family.

    ``card_scale`` grows the deck, hence both the private type space and every
    board-chance support. ``action_scale`` grows the number of streets, the
    raise cap (tree depth), and the raise-size menu (tree width). The rules and
    one-card private-hand/showdown representation are otherwise unchanged.

    The named anchors are exact: ``dial(1, 1) == kuhn()`` and
    ``dial(2, 2) == leduc()``. At larger action scales, every betting round has
    ``action_scale - 1`` distinct raise increments. At most four streets are
    exposed, matching the intended poker depth axis.
    """

    if card_scale < 1 or action_scale < 1:
        raise ValueError("card_scale and action_scale must be positive")
    if action_scale > 4:
        raise ValueError("action_scale cannot exceed four poker streets")
    deck_size = 3 if card_scale == 1 else 2 * card_scale + 2
    n_suits = 1 if card_scale == 1 else 2
    n_rounds = action_scale
    board_per_round = (0,) + (1,) * (n_rounds - 1)
    cards_needed = 2 + sum(board_per_round)
    if cards_needed > deck_size:
        raise ValueError(
            "this card/action point cannot deal two private hands and its board"
        )
    option_count = max(1, action_scale - 1)
    bases = tuple(action_scale * 2**round_index for round_index in range(n_rounds))
    raise_sizes = tuple(
        tuple(base * multiple for multiple in range(1, option_count + 1))
        for base in bases
    )
    return SyntheticDialConfig(
        n_players=2,
        deck_size=deck_size,
        n_suits=n_suits,
        hand_size=1,
        n_rounds=n_rounds,
        board_per_round=board_per_round,
        max_raises=action_scale,
        bet_size=bases,
        raise_sizes=raise_sizes,
        ante=1,
    )
