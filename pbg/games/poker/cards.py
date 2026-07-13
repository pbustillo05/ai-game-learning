"""Poker card identifiers and deterministic representation utilities.

This module deliberately contains no hand evaluation.  A card id is
``4 * rank + suit`` and a hand id follows the lexicographic enumeration in
``archive/POKER_GPU_EVALUATOR_IO.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor

N_CARDS = 52
N_HANDS = 1326
N_CLASSES = 169


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


def validate_evaluator_combo_cards(combo_cards: Tensor) -> None:
    """Require the sibling facade to use this repository's frozen combo map."""

    if not isinstance(combo_cards, Tensor):
        raise RuntimeError("poker_gpu_evaluator.pbg.COMBO_CARDS must be a torch.Tensor")
    if combo_cards.shape != (N_HANDS, 2) or combo_cards.dtype != torch.uint8:
        raise RuntimeError(
            "poker_gpu_evaluator.pbg.COMBO_CARDS must be uint8 [1326,2]"
        )
    expected = torch.tensor(_HAND_CARDS, dtype=torch.uint8, device=combo_cards.device)
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
    valid = unavailable_cards >= 0
    if not bool(((unavailable_cards < N_CARDS) | ~valid).all()):
        raise ValueError("unavailable cards must be in [0,52) or -1")
    if not bool((n_cards >= 0).all()):
        raise ValueError("n_cards must be nonnegative")
    for row, mask in zip(unavailable_cards, valid, strict=True):
        cards = row[mask]
        if cards.numel() != torch.unique(cards).numel():
            raise ValueError("unavailable cards must not contain duplicates")
    available_count = N_CARDS - valid.sum(dim=1)
    if not bool((n_cards <= available_count).all()):
        raise ValueError("requested draw is larger than the card complement")

    batch = unavailable_cards.shape[0]
    max_draw = int(n_cards.max().item()) if batch else 0
    result = torch.full(
        (batch, max_draw), -1, dtype=torch.int16, device=unavailable_cards.device
    )
    if max_draw == 0:
        return result

    blocked = torch.zeros((batch, N_CARDS), dtype=torch.bool, device=result.device)
    if valid.any():
        row_ids = torch.arange(batch, device=result.device)[:, None].expand_as(valid)
        blocked[row_ids[valid], unavailable_cards[valid].to(torch.long)] = True
    # Sorting independent continuous keys is a uniform random permutation of
    # each row's complement, hence every ordered prefix has the required law.
    keys = torch.rand((batch, N_CARDS), generator=gen, device=result.device)
    keys.masked_fill_(blocked, float("inf"))
    draws = torch.argsort(keys, dim=1)[:, :max_draw].to(torch.int16)
    keep = torch.arange(max_draw, device=result.device)[None, :] < n_cards[:, None]
    result[keep] = draws[keep]
    return result
