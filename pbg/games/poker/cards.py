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

N_CARDS = 52
N_HANDS = 1326
N_CLASSES = 169

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
DECK_GEOMETRY_VERSION = "compact-high-ranks-v1"


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


STANDARD_DECK = DeckGeometry(13)


def _check_card(card: int) -> None:
    if isinstance(card, bool) or not isinstance(card, int) or not 0 <= card < N_CARDS:
        raise ValueError("card ids must be integers in [0, 52)")


def hand_index(a: int, b: int) -> int:
    """Return the lexicographic id of the unordered two-card hand ``{a,b}``."""

    _check_card(a)
    _check_card(b)
    if a == b:
        raise ValueError("a hand must contain two distinct cards")
    if a > b:
        a, b = b, a
    return a * 51 - a * (a - 1) // 2 + b - a - 1


_HAND_CARDS: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in range(N_CARDS) for b in range(a + 1, N_CARDS)
)
if len(_HAND_CARDS) != N_HANDS:  # pragma: no cover - import-time invariant
    raise AssertionError("bad combo table")


def hand_cards(h: int) -> tuple[int, int]:
    """Return the ascending card ids represented by hand id ``h``."""

    if isinstance(h, bool) or not isinstance(h, int) or not 0 <= h < N_HANDS:
        raise ValueError("hand ids must be integers in [0, 1326)")
    return _HAND_CARDS[h]


def validate_evaluator_combo_cards(
    combo_cards: Tensor, rank_count: int = 13
) -> None:
    """Require the sibling facade to use the matching compact combo map."""

    geometry = DeckGeometry(rank_count)
    if not isinstance(combo_cards, Tensor):
        raise RuntimeError("poker_gpu_evaluator.pbg combo_cards must be a torch.Tensor")
    if combo_cards.shape != (geometry.n_hands, 2) or combo_cards.dtype != torch.uint8:
        raise RuntimeError(
            "poker_gpu_evaluator.pbg combo_cards has the wrong compact shape or dtype"
        )
    expected = torch.tensor(
        geometry.combo_cards, dtype=torch.uint8, device=combo_cards.device
    )
    if not torch.equal(combo_cards, expected):
        raise RuntimeError(
            "poker_gpu_evaluator.pbg.COMBO_CARDS is incompatible with the local "
            "lexicographic hand_index/hand_cards mapping"
        )


def _class_id(a: int, b: int) -> int:
    rank_a, suit_a = divmod(a, 4)
    rank_b, suit_b = divmod(b, 4)
    high, low = max(rank_a, rank_b), min(rank_a, rank_b)
    high_pos, low_pos = 12 - high, 12 - low
    if high == low:
        return high_pos * 13 + high_pos
    if suit_a == suit_b:
        return high_pos * 13 + low_pos
    return low_pos * 13 + high_pos


_CLASS_OF_HAND: tuple[int, ...] = tuple(_class_id(a, b) for a, b in _HAND_CARDS)
_CLASS_MEMBERS: tuple[tuple[int, ...], ...] = tuple(
    tuple(h for h, class_id in enumerate(_CLASS_OF_HAND) if class_id == k)
    for k in range(N_CLASSES)
)


def class_of_hand(h: int) -> int:
    """Map a combo to its standard ``AA, AKs, ..., 22`` 13x13 class id."""

    hand_cards(h)  # validation
    return _CLASS_OF_HAND[h]


def class_members(k: int) -> list[int]:
    """Return combo ids in preflop class ``k`` in lexicographic order."""

    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < N_CLASSES:
        raise ValueError("class ids must be integers in [0, 169)")
    return list(_CLASS_MEMBERS[k])


def class_size(k: int) -> int:
    """Return 6 for a pair, 4 for suited, and 12 for offsuit classes."""

    return len(class_members(k))


def suit_perm_apply(perm: tuple[int, int, int, int], h: int) -> int:
    """Apply a permutation of suits to a combo id."""

    if tuple(sorted(perm)) != (0, 1, 2, 3):
        raise ValueError("perm must be a permutation of (0, 1, 2, 3)")
    a, b = hand_cards(h)
    mapped_a = 4 * (a // 4) + perm[a % 4]
    mapped_b = 4 * (b // 4) + perm[b % 4]
    return hand_index(mapped_a, mapped_b)


def board_conflict_mask(board: Tensor) -> Tensor:
    """Return a hand-conflict mask for boards shaped ``[5]`` or ``[B,5]``.

    Both local ``-1`` padding and evaluator-boundary ``255`` padding are
    accepted.  The result is bool ``[1326]`` or ``[B,1326]`` respectively.
    """

    if not isinstance(board, Tensor) or board.ndim not in (1, 2):
        raise ValueError("board must be a tensor with shape [5] or [B,5]")
    squeeze = board.ndim == 1
    rows = board.unsqueeze(0) if squeeze else board
    if rows.shape[-1] != 5:
        raise ValueError("board must have exactly five slots")
    integer_dtypes = {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if rows.dtype not in integer_dtypes:
        raise TypeError("board must have an integer dtype")
    real = (rows >= 0) & (rows < N_CARDS)
    padding = (rows == -1) | (rows == 255)
    if not bool((real | padding).all()):
        raise ValueError("board contains an invalid card id")
    for row, valid in zip(rows, real, strict=True):
        cards = row[valid]
        if cards.numel() != torch.unique(cards).numel():
            raise ValueError("board cards must be unique")
    combo_cards = torch.tensor(_HAND_CARDS, dtype=rows.dtype, device=rows.device)
    conflict = (
        (combo_cards[None, :, :, None] == rows[:, None, None, :])
        & real[:, None, None, :]
    ).any(dim=-1).any(dim=-1)
    return conflict[0] if squeeze else conflict


def sample_uniform_complement(
    unavailable_cards: Tensor,
    n_cards: Tensor,
    gen: torch.Generator,
    *,
    rank_count: int = 13,
) -> Tensor:
    """Sample ordered, uniform draws without replacement from each complement.

    ``unavailable_cards`` is int16 ``[B,U]`` with ``-1`` padding and
    ``n_cards`` is int32 ``[B]``.  Output is int16 ``[B,max(n_cards)]`` with a
    ``-1`` padded suffix.
    """

    if unavailable_cards.ndim != 2 or unavailable_cards.dtype != torch.int16:
        raise TypeError("unavailable_cards must be int16 [B,U]")
    if n_cards.ndim != 1 or n_cards.dtype != torch.int32:
        raise TypeError("n_cards must be int32 [B]")
    if unavailable_cards.shape[0] != n_cards.shape[0]:
        raise ValueError("batch dimensions do not match")
    if unavailable_cards.device != n_cards.device:
        raise ValueError("inputs must be on one device")
    if not isinstance(gen, torch.Generator):
        raise TypeError("gen must be a torch.Generator")
    geometry = DeckGeometry(rank_count)
    valid = unavailable_cards >= 0
    active = (unavailable_cards >= geometry.first_card) & (
        unavailable_cards < N_CARDS
    )
    if not bool((active | ~valid).all()):
        raise ValueError("unavailable cards must be active-deck ids or -1")
    if not bool((n_cards >= 0).all()):
        raise ValueError("n_cards must be nonnegative")
    for row, mask in zip(unavailable_cards, valid, strict=True):
        cards = row[mask]
        if cards.numel() != torch.unique(cards).numel():
            raise ValueError("unavailable cards must not contain duplicates")
    available_count = geometry.n_cards - valid.sum(dim=1)
    if not bool((n_cards <= available_count).all()):
        raise ValueError("requested draw is larger than the card complement")

    batch = unavailable_cards.shape[0]
    max_draw = int(n_cards.max().item()) if batch else 0
    result = torch.full(
        (batch, max_draw), -1, dtype=torch.int16, device=unavailable_cards.device
    )
    if max_draw == 0:
        return result

    blocked = torch.zeros(
        (batch, geometry.n_cards), dtype=torch.bool, device=result.device
    )
    if valid.any():
        row_ids = torch.arange(batch, device=result.device)[:, None].expand_as(valid)
        blocked[
            row_ids[valid],
            unavailable_cards[valid].to(torch.long) - geometry.first_card,
        ] = True
    # Sorting independent continuous keys is a uniform random permutation of
    # each row's complement, hence every ordered prefix has the required law.
    keys = torch.rand(
        (batch, geometry.n_cards), generator=gen, device=result.device
    )
    keys.masked_fill_(blocked, float("inf"))
    draws = (
        torch.argsort(keys, dim=1)[:, :max_draw] + geometry.first_card
    ).to(torch.int16)
    keep = torch.arange(max_draw, device=result.device)[None, :] < n_cards[:, None]
    result[keep] = draws[keep]
    return result
