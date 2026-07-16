"""Poker action-likelihood reach bookkeeping.

The rows in this module are deliberately marginal, unnormalized likelihood
products.  They are not a joint private-card distribution.  Log rows may be
shifted by a row-wise constant for numerical stability, so only differences
within a row are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from pbg.games.poker.cards import DeckGeometry

LINE_META_SIZE = 8
"""Width of public fold-line metadata; fields after the caller's data are zero padded."""


@dataclass
class FoldedPlayer:
    """A frozen fold-time range and its public context.

    ``f_weights`` is float32 ``[H]``, ``board_at_fold`` is int8 ``[5]``, and
    ``line_meta`` is float32 ``[D_line]``.  The weights retain the arbitrary
    positive row scale used by :class:`RangeState`.
    """

    f_weights: Tensor
    board_at_fold: Tensor
    line_meta: Tensor


class RangeState:
    """One reach table for rollout nodes, indexed by ``(slot, player)``.

    Args:
        n_slots: Number of initially allocated rollout-node slots ``S``.
        n_players: Number of seats ``P``.
        device: Device on which all state tensors are stored.

    ``log_r`` is float32 ``[S,P,H]``, ``live`` is bool ``[S,P]``, and
    ``board`` is int8 ``[S,5]``.  Public fold-line metadata can be written to
    ``line_meta[S,P,8]`` by the rollout owner before :meth:`fold`; this module
    has no engine dependency from which to derive it.
    """

    def __init__(
        self,
        n_slots: int,
        n_players: int,
        device: torch.device | str | None = None,
        rank_count: int = 13,
    ) -> None:
        if isinstance(n_slots, bool) or not isinstance(n_slots, int) or n_slots < 0:
            raise ValueError("n_slots must be a nonnegative integer")
        if (
            isinstance(n_players, bool)
            or not isinstance(n_players, int)
            or n_players <= 0
        ):
            raise ValueError("n_players must be a positive integer")
        self.n_players = n_players
        self.geometry = DeckGeometry(rank_count)
        self.n_hands = self.geometry.n_hands
        self.n_classes = self.geometry.n_classes
        self.log_r = torch.zeros(
            (n_slots, n_players, self.n_hands), dtype=torch.float32, device=device
        )
        self.live = torch.ones(
            (n_slots, n_players), dtype=torch.bool, device=self.log_r.device
        )
        self.board = torch.full(
            (n_slots, 5), -1, dtype=torch.int8, device=self.log_r.device
        )
        self.line_meta = torch.zeros(
            (n_slots, n_players, LINE_META_SIZE),
            dtype=torch.float32,
            device=self.log_r.device,
        )
        self._row_max = torch.zeros(
            (n_slots, n_players), dtype=torch.float32, device=self.log_r.device
        )
        self._class_of_hand = torch.tensor(
            self.geometry.class_of_combo,
            dtype=torch.long,
            device=self.log_r.device,
        )

    @property
    def n_slots(self) -> int:
        """Return the current number of allocated rows ``S``."""

        return self.log_r.shape[0]

    def _indices(
        self, values: Tensor | Sequence[int], name: str, upper: int
    ) -> Tensor:
        if isinstance(values, Tensor):
            if values.ndim != 1:
                raise ValueError(f"{name} must have shape [B]")
            if values.dtype not in (
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError(f"{name} must have an integer dtype")
            result = values.to(device=self.log_r.device, dtype=torch.long)
        else:
            result = torch.as_tensor(values, dtype=torch.long, device=self.log_r.device)
            if result.ndim != 1:
                raise ValueError(f"{name} must have shape [B]")
        if result.numel() and not bool(((result >= 0) & (result < upper)).all()):
            raise IndexError(f"{name} contains an out-of-range index")
        return result

    @staticmethod
    def _require_unique_pairs(slots: Tensor, players: Tensor) -> None:
        if slots.numel() == 0:
            return
        pairs = torch.stack((slots, players), dim=1)  # [B,2]
        if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
            raise ValueError("each (slot, player) pair may occur only once")

    def _stabilize(self, slots: Tensor, players: Tensor) -> None:
        """Shift selected finite log rows whose maxima have drifted below -30."""

        if slots.numel() == 0:
            return
        rows = self.log_r[slots, players]  # [B,H]
        row_max = rows.max(dim=1).values  # [B]
        if not bool(torch.isfinite(row_max).all()):
            raise ValueError("a reach row has no possible hand")
        shift = torch.where(row_max < -30.0, row_max, torch.zeros_like(row_max))
        self.log_r[slots, players] = rows - shift[:, None]  # [B,H]
        self._row_max[slots, players] = row_max - shift

    def clone_rows(self, src_slots: Tensor | Sequence[int]) -> Tensor:
        """Append independent copies of ``src_slots`` and return int32 ``[B]`` ids."""

        src = self._indices(src_slots, "src_slots", self.n_slots)
        start = self.n_slots
        self.log_r = torch.cat((self.log_r, self.log_r[src].clone()), dim=0)
        self.live = torch.cat((self.live, self.live[src].clone()), dim=0)
        self.board = torch.cat((self.board, self.board[src].clone()), dim=0)
        self.line_meta = torch.cat(
            (self.line_meta, self.line_meta[src].clone()), dim=0
        )
        self._row_max = torch.cat(
            (self._row_max, self._row_max[src].clone()), dim=0
        )
        return torch.arange(
            start,
            start + src.numel(),
            dtype=torch.int32,
            device=self.log_r.device,
        )  # [B]

    def update_action(
        self,
        slots: Tensor | Sequence[int],
        players: Tensor | Sequence[int],
        event_ids: Tensor | Sequence[int],
        policy_log_rows: Tensor,
    ) -> None:
        """Multiply actor reaches by taken-action likelihoods in log space.

        ``slots``, ``players``, and ``event_ids`` have shape ``[B]``.
        ``policy_log_rows`` is float ``[S,H,E]`` indexed by the global slot id,
        or float ``[B,H,E]`` aligned with the selected rows.  Only the selected
        actor row in each slot is changed.
        """

        if not isinstance(policy_log_rows, Tensor) or policy_log_rows.ndim != 3:
            raise ValueError("policy_log_rows must have shape [S,H,E] or [B,H,E]")
        slot_ids = self._indices(slots, "slots", self.n_slots)
        player_ids = self._indices(players, "players", self.n_players)
        if slot_ids.shape != player_ids.shape:
            raise ValueError("slots and players must have the same shape")
        event = self._indices(event_ids, "event_ids", policy_log_rows.shape[-1])
        if event.shape != slot_ids.shape:
            raise ValueError("event_ids must have the same shape as slots")
        self._require_unique_pairs(slot_ids, player_ids)
        if policy_log_rows.shape[1] != self.n_hands:
            raise ValueError(
                f"policy_log_rows must have {self.n_hands} compact hand rows"
            )
        if not policy_log_rows.is_floating_point():
            raise TypeError("policy_log_rows must have a floating dtype")
        if policy_log_rows.device != self.log_r.device:
            raise ValueError("policy_log_rows and range state must share a device")
        batch = slot_ids.numel()
        if policy_log_rows.shape[0] == self.n_slots:
            policy_rows = policy_log_rows[slot_ids]  # [B,H,E]
        elif policy_log_rows.shape[0] == batch:
            policy_rows = policy_log_rows  # [B,H,E]
        else:
            raise ValueError("policy_log_rows first dimension must be S or B")
        if event.numel() and int(event.max()) >= policy_rows.shape[2]:
            raise IndexError("event_ids contains an out-of-range index")
        taken = policy_rows[
            torch.arange(batch, device=self.log_r.device), :, event
        ]  # [B,H]
        if bool(torch.isnan(taken).any()) or bool(torch.isposinf(taken).any()):
            raise ValueError("policy log probabilities may not contain NaN or +inf")
        self.log_r[slot_ids, player_ids] += taken.detach().to(torch.float32)
        self._stabilize(slot_ids, player_ids)

    def update_board(
        self,
        slots: Tensor | Sequence[int],
        new_cards: Tensor,
    ) -> None:
        """Append public cards and mask conflicting hands in selected slots.

        ``slots`` is integer ``[B]``.  ``new_cards`` is integer ``[B,C]`` with
        optional ``-1`` padding; for a single selected slot, ``[C]`` is also
        accepted.  Every player's current row is masked, including folded rows;
        previously returned fold snapshots are independent tensors.
        """

        slot_ids = self._indices(slots, "slots", self.n_slots)
        if torch.unique(slot_ids).numel() != slot_ids.numel():
            raise ValueError("each slot may occur only once")
        if not isinstance(new_cards, Tensor) or new_cards.ndim not in (1, 2):
            raise ValueError("new_cards must have shape [B,C] or [C] for B=1")
        if new_cards.ndim == 1:
            if slot_ids.numel() != 1:
                raise ValueError("one-dimensional new_cards requires one slot")
            cards = new_cards.unsqueeze(0)
        else:
            cards = new_cards
        if cards.shape[0] != slot_ids.numel():
            raise ValueError("new_cards batch dimension must equal len(slots)")
        if cards.dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("new_cards must have an integer dtype")
        if cards.device != self.log_r.device:
            raise ValueError("new_cards and range state must share a device")
        cards_i16 = cards.to(torch.int16)
        valid_cards = (cards_i16 == -1) | (
            (cards_i16 >= self.geometry.first_card) & (cards_i16 < 52)
        )
        if not bool(valid_cards.all()):
            raise ValueError("new_cards must contain active-deck card ids or -1")

        current = self.board[slot_ids].to(torch.int16)  # [B,5]
        for row in range(slot_ids.numel()):
            existing = current[row][current[row] >= 0]
            added = cards_i16[row][cards_i16[row] >= 0]
            combined = torch.cat((existing, added))  # [<=5]
            if combined.numel() > 5:
                raise ValueError("a board cannot contain more than five cards")
            if torch.unique(combined).numel() != combined.numel():
                raise ValueError("board cards must be unique")
            current[row].fill_(-1)
            current[row, : combined.numel()] = combined
        self.board[slot_ids] = current.to(torch.int8)
        conflict = self.geometry.board_conflict_mask(self.board[slot_ids])  # [B,H]
        rows = self.log_r[slot_ids]  # [B,P,H]
        self.log_r[slot_ids] = rows.masked_fill(conflict[:, None, :], -torch.inf)
        all_players = torch.arange(self.n_players, device=self.log_r.device)
        stab_slots = slot_ids[:, None].expand(-1, self.n_players).reshape(-1)
        stab_players = all_players[None, :].expand(slot_ids.numel(), -1).reshape(-1)
        self._stabilize(stab_slots, stab_players)

    def fold(
        self,
        slots: Tensor | Sequence[int],
        players: Tensor | Sequence[int],
    ) -> list[FoldedPlayer]:
        """Freeze selected fold-time rows and mark their seats non-live.

        Inputs are integer ``[B]``.  The returned list has ``B`` snapshots,
        each containing ``f_weights[H]``, ``board_at_fold[5]``, and
        ``line_meta[D_line]``.
        """

        slot_ids = self._indices(slots, "slots", self.n_slots)
        player_ids = self._indices(players, "players", self.n_players)
        if slot_ids.shape != player_ids.shape:
            raise ValueError("slots and players must have the same shape")
        self._require_unique_pairs(slot_ids, player_ids)
        if slot_ids.numel() and not bool(self.live[slot_ids, player_ids].all()):
            raise ValueError("a player may only fold once in a slot")
        weights = self.log_r[slot_ids, player_ids].exp().clone()  # [B,H]
        boards = self.board[slot_ids].clone()  # [B,5]
        metadata = self.line_meta[slot_ids, player_ids].clone()  # [B,D_line]
        self.live[slot_ids, player_ids] = False
        return [
            FoldedPlayer(weights[i], boards[i], metadata[i])
            for i in range(slot_ids.numel())
        ]

    def normalized(
        self,
        slots: Tensor | Sequence[int],
        players: Tensor | Sequence[int],
    ) -> Tensor:
        """Return normalized marginal reaches as float32 ``[B,H]``."""

        slot_ids = self._indices(slots, "slots", self.n_slots)
        player_ids = self._indices(players, "players", self.n_players)
        if slot_ids.shape != player_ids.shape:
            raise ValueError("slots and players must have the same shape")
        rows = self.log_r[slot_ids, player_ids]  # [B,H]
        if rows.numel() and not bool(torch.isfinite(rows.max(dim=1).values).all()):
            raise ValueError("a reach row has no possible hand")
        return torch.softmax(rows, dim=1, dtype=torch.float32)  # [B,H]

    def to_class(self, slots: Tensor | Sequence[int]) -> Tensor:
        """Aggregate normalized combo reaches into float32 ``[B,P,K]``."""

        slot_ids = self._indices(slots, "slots", self.n_slots)
        rows = self.log_r[slot_ids]  # [B,P,H]
        if rows.numel() and not bool(torch.isfinite(rows.max(dim=2).values).all()):
            raise ValueError("a reach row has no possible hand")
        normalized = torch.softmax(rows, dim=2, dtype=torch.float32)  # [B,P,H]
        result = torch.zeros(
            (slot_ids.numel(), self.n_players, self.n_classes),
            dtype=torch.float32,
            device=self.log_r.device,
        )
        class_ids = self._class_of_hand.view(1, 1, self.n_hands).expand(
            slot_ids.numel(), self.n_players, -1
        )  # [B,P,H]
        result.scatter_add_(2, class_ids, normalized)
        return result  # [B,P,K]
