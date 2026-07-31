from __future__ import annotations

from collections import Counter

from pbg.games.poker.cards import (
    DeckGeometry,
)

GEOMETRY = DeckGeometry(13)


def test_complete_combo_map_and_preflop_classes() -> None:
    combo_index = {
        combo: index for index, combo in enumerate(GEOMETRY.combo_cards)
    }
    assert len(combo_index) == GEOMETRY.n_hands

    members = [
        hand
        for group in GEOMETRY.class_memberships
        for hand in group
    ]
    assert sorted(members) == list(range(GEOMETRY.n_hands))
    for k in range(GEOMETRY.n_classes):
        assert all(
            GEOMETRY.class_of_combo[hand] == k
            for hand in GEOMETRY.class_memberships[k]
        )
    assert Counter(
        len(group) for group in GEOMETRY.class_memberships
    ) == {
        4: 78,
        6: 13,
        12: 78,
    }

    # Standard matrix orientation: AA=0, AKs=1, AKo=13, 22=168.
    assert GEOMETRY.class_of_combo[combo_index[(48, 49)]] == 0
    assert GEOMETRY.class_of_combo[combo_index[(44, 48)]] == 1
    assert GEOMETRY.class_of_combo[combo_index[(45, 48)]] == 13
    assert GEOMETRY.class_of_combo[combo_index[(0, 1)]] == 168
