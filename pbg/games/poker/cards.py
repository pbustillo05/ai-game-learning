"""Poker card identifiers and deterministic representation utilities.

This module deliberately contains no hand evaluation.  A card id is
``4 * rank + suit`` and a hand id follows the lexicographic enumeration in
``archive/POKER_GPU_EVALUATOR_IO.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

CATEGORY_NAMES = (
    "high_card",
    "pair",
    "two_pair",
    "trips",
    "straight",
    "flush",
    "full_house",
    "quads",
    "straight_flush",
)
CONVENTIONAL_STRENGTH = tuple(reversed(CATEGORY_NAMES))


@dataclass(frozen=True)
class CategoryProfile:
    """Exact cumulative five-card category frequencies for one active deck."""

    denominator: int
    numerators: Mapping[str, int]
    strength_order: tuple[str, ...]

    def probability(self, category: str) -> Fraction:
        """Return the exact probability of containing ``category``."""

        return Fraction(self.numerators[category], self.denominator)


def category_profile(rank_count: int) -> CategoryProfile:
    """Return exact cumulative category counts and rarity-consistent strength.

    Categories use minimal-essence containment: a full house contains trips,
    two pair, and a pair; a straight flush contains a straight and a flush;
    and every holding contains high card.  Impossible categories are omitted
    from ``strength_order``.  Equal counts retain conventional poker order.
    """

    if isinstance(rank_count, bool) or not isinstance(rank_count, int):
        raise TypeError("rank_count must be an integer")
    if not 4 <= rank_count <= 13:
        raise ValueError("rank_count must be in [4,13]")
    r = rank_count
    total = math.comb(4 * r, 5)
    sequences = max(0, r - 4) + int(r == 13)
    choose_r_5 = math.comb(r, 5) if r >= 5 else 0
    counts = {
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
    conventional_index = {
        category: index for index, category in enumerate(CONVENTIONAL_STRENGTH)
    }
    order = tuple(
        sorted(
            (category for category in CATEGORY_NAMES if counts[category] > 0),
            key=lambda category: (counts[category], conventional_index[category]),
        )
    )
    return CategoryProfile(total, MappingProxyType(counts), order)


@dataclass(frozen=True)
class DeckGeometry:
    """Immutable compact geometry for the highest ``rank_count`` ranks.

    Card values remain global evaluator IDs.  The active deck is the suffix
    ``52-4R .. 51`` while combo and class IDs are compact and deck-local.
    """

    rank_count: int
    n_cards: int = field(init=False)
    n_hands: int = field(init=False)
    n_classes: int = field(init=False)
    first_card: int = field(init=False)
    active_cards: tuple[int, ...] = field(init=False)
    combo_cards: tuple[tuple[int, int], ...] = field(init=False)
    class_of_combo: tuple[int, ...] = field(init=False)
    class_memberships: tuple[tuple[int, ...], ...] = field(init=False)
    categories: CategoryProfile = field(init=False)
    _combo_index: Mapping[tuple[int, int], int] = field(init=False, repr=False)

    def __deepcopy__(self, memo: dict[int, object]) -> DeckGeometry:
        """Return this immutable value object during model snapshot copies."""

        memo[id(self)] = self
        return self

    def __post_init__(self) -> None:
        r = self.rank_count
        if isinstance(r, bool) or not isinstance(r, int):
            raise TypeError("rank_count must be an integer")
        if not 4 <= r <= 13:
            raise ValueError("rank_count must be in [4,13]")
        d = 4 * r
        first = 52 - d
        cards = tuple(range(first, 52))
        combos = tuple((a, b) for a in cards for b in cards if a < b)

        def class_id(combo: tuple[int, int]) -> int:
            a, b = combo
            rank_a = a // 4 - (13 - r)
            rank_b = b // 4 - (13 - r)
            high, low = max(rank_a, rank_b), min(rank_a, rank_b)
            high_pos, low_pos = r - 1 - high, r - 1 - low
            if rank_a == rank_b or a % 4 == b % 4:
                return high_pos * r + low_pos
            return low_pos * r + high_pos

        classes = tuple(class_id(combo) for combo in combos)
        object.__setattr__(self, "n_cards", d)
        object.__setattr__(self, "n_hands", math.comb(d, 2))
        object.__setattr__(self, "n_classes", r * r)
        object.__setattr__(self, "first_card", first)
        object.__setattr__(self, "active_cards", cards)
        object.__setattr__(self, "combo_cards", combos)
        object.__setattr__(
            self, "_combo_index", MappingProxyType({combo: i for i, combo in enumerate(combos)})
        )
        object.__setattr__(self, "class_of_combo", classes)
        object.__setattr__(
            self,
            "class_memberships",
            tuple(tuple(i for i, value in enumerate(classes) if value == k) for k in range(r * r)),
        )
        object.__setattr__(self, "categories", category_profile(r))

    def hand_index(self, a: int, b: int) -> int:
        """Return the compact ID of an unordered active-card pair."""

        if a == b:
            raise ValueError("a hand must contain two distinct cards")
        key = (a, b) if a < b else (b, a)
        try:
            return self._combo_index[key]
        except KeyError as exc:
            raise ValueError("hand contains a card outside the active deck") from exc

    def hand_cards(self, hand: int) -> tuple[int, int]:
        if isinstance(hand, bool) or not isinstance(hand, int) or not 0 <= hand < self.n_hands:
            raise ValueError(f"hand ids must be integers in [0,{self.n_hands})")
        return self.combo_cards[hand]

    def class_members(self, class_id: int) -> tuple[int, ...]:
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < self.n_classes
        ):
            raise ValueError(f"class ids must be integers in [0,{self.n_classes})")
        return self.class_memberships[class_id]

    def board_conflict_mask(self, board: Tensor) -> Tensor:
        """Return compact combo conflicts for ``[5]`` or ``[B,5]`` boards."""

        if not isinstance(board, Tensor) or board.ndim not in (1, 2):
            raise ValueError("board must be a tensor with shape [5] or [B,5]")
        squeeze = board.ndim == 1
        rows = board.unsqueeze(0) if squeeze else board
        if rows.shape[-1] != 5:
            raise ValueError("board must have exactly five slots")
        real = (rows >= self.first_card) & (rows < 52)
        padding = (rows == -1) | (rows == 255)
        if not bool((real | padding).all()):
            raise ValueError("board contains a card outside the active deck")
        for row, valid in zip(rows, real, strict=True):
            cards = row[valid]
            if cards.numel() != torch.unique(cards).numel():
                raise ValueError("board cards must be unique")
        combos = torch.tensor(self.combo_cards, dtype=rows.dtype, device=rows.device)
        conflict = (
            (combos[None, :, :, None] == rows[:, None, None, :])
            & real[:, None, None, :]
        ).any(dim=-1).any(dim=-1)
        return conflict[0] if squeeze else conflict
