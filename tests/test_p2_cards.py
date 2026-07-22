from __future__ import annotations

from collections import Counter

import pytest
import torch

from pbg.games.poker.cards import (
    DeckGeometry,
    N_HANDS,
    board_conflict_mask,
    hand_cards,
    sample_uniform_complement,
)


GEOMETRY = DeckGeometry(13)


def test_complete_combo_map_and_preflop_classes() -> None:
    seen: set[int] = set()
    for a in range(52):
        for b in range(a + 1, 52):
            h = GEOMETRY.hand_index(a, b)
            assert hand_cards(h) == (a, b)
            assert GEOMETRY.hand_index(b, a) == h
            seen.add(h)
    assert seen == set(range(N_HANDS))

    members = [h for k in range(GEOMETRY.n_classes) for h in GEOMETRY.class_members(k)]
    assert sorted(members) == list(range(N_HANDS))
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
    mask = board_conflict_mask(board)
    assert mask.shape == (N_HANDS,)
    for h in range(N_HANDS):
        assert bool(mask[h]) == bool(set(hand_cards(h)) & {0, 7, 51})

    batched = board_conflict_mask(
        torch.tensor([[0, 7, 51, 255, 255], [4, 5, 6, 8, 255]], dtype=torch.uint8)
    )
    assert torch.equal(batched[0], mask)
    assert int(batched[1].sum()) == N_HANDS - (48 * 47 // 2)
    with pytest.raises(ValueError):
        board_conflict_mask(torch.tensor([3, 3, -1, -1, -1], dtype=torch.int8))


def test_complement_sampler_legality_reproducibility_and_uniformity() -> None:
    unavailable = torch.tensor([[0, 1, 2, -1], [48, 49, 50, 51]], dtype=torch.int16)
    counts = torch.tensor([3, 1], dtype=torch.int32)
    a = sample_uniform_complement(
        unavailable, counts, torch.Generator().manual_seed(90210)
    )
    b = sample_uniform_complement(
        unavailable, counts, torch.Generator().manual_seed(90210)
    )
    assert torch.equal(a, b)
    assert len(set(a[0].tolist())) == 3
    assert set(a[0].tolist()).isdisjoint({0, 1, 2})
    assert int(a[1, 1]) == -1 and int(a[1, 2]) == -1

    n = 20_000
    draws = sample_uniform_complement(
        torch.tensor([[0, 1]] * n, dtype=torch.int16),
        torch.ones(n, dtype=torch.int32),
        torch.Generator().manual_seed(717),
    )[:, 0]
    frequency = torch.bincount(draws.to(torch.long), minlength=52)[2:].float()
    expected = n / 50
    # Six standard deviations plus a small discrete margin.
    assert float((frequency - expected).abs().max()) < 6 * (expected**0.5) + 3
