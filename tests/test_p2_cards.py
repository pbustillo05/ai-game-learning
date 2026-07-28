from __future__ import annotations

from collections import Counter

import pytest
import torch

from pbg.games.poker.cards import (
    DeckGeometry,
)


GEOMETRY = DeckGeometry(13)


def test_complete_combo_map_and_preflop_classes() -> None:
    seen: set[int] = set()
    for a in range(52):
        for b in range(a + 1, 52):
            h = GEOMETRY.hand_index(a, b)
            assert GEOMETRY.hand_cards(h) == (a, b)
            assert GEOMETRY.hand_index(b, a) == h
            seen.add(h)
    assert seen == set(range(GEOMETRY.n_hands))

    members = [h for k in range(GEOMETRY.n_classes) for h in GEOMETRY.class_members(k)]
    assert sorted(members) == list(range(GEOMETRY.n_hands))
    for k in range(GEOMETRY.n_classes):
        assert all(GEOMETRY.class_of_combo[h] == k for h in GEOMETRY.class_members(k))
    assert Counter(
        len(GEOMETRY.class_members(k)) for k in range(GEOMETRY.n_classes)
    ) == {
        4: 78,
        6: 13,
        12: 78,
    }

    # Standard matrix orientation: AA=0, AKs=1, AKo=13, 22=168.
    assert GEOMETRY.class_of_combo[GEOMETRY.hand_index(48, 49)] == 0
    assert GEOMETRY.class_of_combo[GEOMETRY.hand_index(44, 48)] == 1
    assert GEOMETRY.class_of_combo[GEOMETRY.hand_index(45, 48)] == 13
    assert GEOMETRY.class_of_combo[GEOMETRY.hand_index(0, 1)] == 168


def test_board_conflict_mask_supports_local_and_boundary_padding() -> None:
    board = torch.tensor([0, 7, 51, -1, -1], dtype=torch.int16)
    mask = GEOMETRY.board_conflict_mask(board)
    assert mask.shape == (GEOMETRY.n_hands,)
    for h in range(GEOMETRY.n_hands):
        assert bool(mask[h]) == bool(
            set(GEOMETRY.hand_cards(h)) & {0, 7, 51}
        )

    batched = GEOMETRY.board_conflict_mask(
        torch.tensor([[0, 7, 51, 255, 255], [4, 5, 6, 8, 255]], dtype=torch.uint8)
    )
    assert torch.equal(batched[0], mask)
    assert int(batched[1].sum()) == GEOMETRY.n_hands - (48 * 47 // 2)
    with pytest.raises(ValueError):
        GEOMETRY.board_conflict_mask(
            torch.tensor([3, 3, -1, -1, -1], dtype=torch.int8)
        )
