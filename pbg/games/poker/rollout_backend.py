"""Device-native batched no-limit hold'em rollout backend."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch
from pbg.games.poker.cards import N_CLASSES, N_HANDS
from pbg.games.poker.engine import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    FOLDED,
    LIVE,
    RAISE_OFFSET,
    RIVER,
    PokerConfig,
    PotStrata,
    assemble_showdown_payoff,
)
from pbg.games.poker.folded import (
    FoldSnapshotBatch,
    PartitionSummaryBatch,
    PokerGPUPartitionAdapter,
)
from pbg.games.poker.mc import PokerGPURejectionSampler, PokerGPUShowdownEvaluator
from pbg.games.poker.model.postflop import PostflopMasks
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerPublicBatch,
    PolicyOutput,
    PreflopMasks,
)
from pbg.games.poker.tensor_state import PokerTensorStateStore


_PLAYER = 0
_CHANCE = 1
_TERMINAL = 2
_PRETERM = 3
_EPS = 1e-6
_CHANCE_OFFSETS = (100_000, 200_000, 300_000)


@dataclass
class CompactPokerDecisionReach:
    """Ragged live-seat reach rows retained for optimization."""

    row_ptr: Tensor  # int64 [D+1]
    seat_ids: Tensor  # int64 [T]
    log_r: Tensor  # float32 [T,1326]
    n_rows: int


class PokerRolloutBackend:
    """Batched Torch implementation of public NLHE navigation.

    Args:
        cfg: Public table configuration.
        capacity: Maximum number of state rows resident in this pipeline.
        device: Owning CPU or CUDA device.
        max_rollout_steps: Maximum public transitions in one sampled path.
        beta_fix: Fixed branching intensities by public-stage bucket.
        profile_sampler: Exact sibling-facade joint-profile adapter.
        partition_adapter: Sibling-facade neutral-partition adapter.
        evaluator_backend: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Every returned tensor remains on ``device``.  The CPU and CUDA paths run
    the same state-transition code; :class:`NLHEEngine` remains the separate
    list-based correctness oracle used only by tests and evaluation.
    """

    def __init__(
        self,
        cfg: PokerConfig,
        capacity: int,
        device: torch.device | str,
        *,
        max_rollout_steps: int | None = None,
        beta_fix: Mapping[str, float] | Any | None = None,
        profile_sampler: PokerGPURejectionSampler | Any | None = None,
        partition_adapter: PokerGPUPartitionAdapter | Any | None = None,
        evaluator_backend: str = "auto",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_players = cfg.n_players
        self.x_sizes = (N_HANDS,) * cfg.n_players
        self.n_ticks = cfg.n_ticks
        self.e_max = RAISE_OFFSET + cfg.n_ticks
        self.raise_unit_values = (
            torch.arange(1, cfg.n_ticks + 1, dtype=torch.float32, device=self.device)
            * float(cfg.chip_unit_bb)
        )
        if evaluator_backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("evaluator_backend must be 'auto', 'cpu', or 'cuda'")
        if max_rollout_steps is None:
            raises_per_street = math.ceil(cfg.stack_bb / cfg.bb) + 1
            max_rollout_steps = 4 * cfg.n_players * (raises_per_street + 1) + 3
        self.max_rollout_steps = int(max_rollout_steps)
        self.states = PokerTensorStateStore(
            capacity, self.max_rollout_steps, self.device, cfg.n_players
        )
        self.state_store = self.states
        table = getattr(beta_fix, "beta_fix", beta_fix)
        self.beta_fix = dict(table or {"default": 1.0})
        if "default" not in self.beta_fix:
            raise ValueError("beta_fix requires a 'default' entry")
        if any(not math.isfinite(float(value)) or float(value) < 1.0 for value in self.beta_fix.values()):
            raise ValueError("rollout beta values must be finite and at least one")
        self.profile_sampler = profile_sampler or PokerGPURejectionSampler(
            backend=evaluator_backend
        )
        self.showdown_evaluator = PokerGPUShowdownEvaluator(backend=evaluator_backend)
        self.partition_adapter = partition_adapter or PokerGPUPartitionAdapter()
        combo_cards = torch.triu_indices(52, 52, offset=1, device=self.device).T
        self.combo_cards = combo_cards.to(torch.int16)  # [H,2]
        a, b = combo_cards[:, 0], combo_cards[:, 1]
        rank_a, rank_b = a // 4, b // 4
        high_pos = 12 - torch.maximum(rank_a, rank_b)
        low_pos = 12 - torch.minimum(rank_a, rank_b)
        self.combo_class = torch.where(
            (rank_a == rank_b) | ((a % 4) == (b % 4)),
            high_pos * 13 + low_pos,
            low_pos * 13 + high_pos,
        ).to(torch.int64)  # [H]
        self.batch_size_histograms: dict[str, list[int]] = {
            "preflop": [],
            **{f"postflop_{live}": [] for live in range(2, cfg.n_players + 1)},
        }
        self.execution_profile = None
        self.policy_batch_metrics: deque[dict[str, int | str]] = deque(maxlen=4096)
        self._partition_cache: dict[int, PartitionSummaryBatch] = {}
        self.game = self

    def root_states(self, n: int) -> Tensor:
        """Reset storage, post blinds, and return int32 root handles ``[B]``."""

        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        buttons = torch.remainder(
            torch.arange(n, dtype=torch.int32, device=self.device), self.n_players
        )  # [B]
        return self._root_states_with_buttons(buttons)

    def _root_states_with_buttons(self, buttons: Tensor) -> Tensor:
        """Reset storage and initialize roots for explicit buttons ``[B]``."""

        n = buttons.shape[0]
        self.states.reset()
        self._partition_cache.clear()
        handles = self.states.allocate_all(n)  # [B]
        rows = handles.to(torch.int64)
        if self.n_players == 2:
            sb_seat = buttons
            bb_seat = torch.remainder(buttons + 1, 2)
        else:
            sb_seat = torch.remainder(buttons + 1, self.n_players)
            bb_seat = torch.remainder(buttons + 2, self.n_players)
        stacks = torch.full(
            (n, self.n_players), float(self.cfg.stack_bb),
            dtype=torch.float32, device=self.device,
        )  # [B,P]
        committed = torch.zeros_like(stacks)  # [B,P]
        batch = torch.arange(n, device=self.device)
        sb_post = torch.minimum(stacks[batch, sb_seat.long()], torch.full((n,), float(self.cfg.sb), device=self.device))
        bb_post = torch.minimum(stacks[batch, bb_seat.long()], torch.full((n,), float(self.cfg.bb), device=self.device))
        committed[batch, sb_seat.long()] += sb_post
        committed[batch, bb_seat.long()] += bb_post
        stacks -= committed
        status = torch.where(stacks <= _EPS, ALL_IN, LIVE).to(torch.int8)  # [B,P]
        first = torch.remainder(bb_seat + 1, self.n_players)
        offsets = torch.arange(self.n_players, device=self.device)[None, :]  # [1,P]
        candidates = torch.remainder(first[:, None] + offsets, self.n_players)
        candidate_live = status.gather(1, candidates.long()) == LIVE
        actor = candidates.gather(1, candidate_live.to(torch.int64).argmax(1, keepdim=True)).squeeze(1)
        has_actor = candidate_live.any(1)
        self.states.stacks[rows] = stacks
        self.states.committed_street[rows] = committed
        self.states.committed_total[rows] = committed
        self.states.status[rows] = status
        self.states.acted[rows] = False
        self.states.reopen_mask[rows] = status == LIVE
        self.states.actor[rows] = torch.where(has_actor, actor, -torch.ones_like(actor)).to(torch.int32)
        self.states.street[rows] = 0
        self.states.button[rows] = buttons
        self.states.n_raises[rows] = 0
        self.states.phase[rows] = torch.where(has_actor, _PLAYER, _PRETERM).to(torch.int32)
        self.states.board[rows] = -1
        self.states.last_full_raise[rows] = float(self.cfg.bb)
        self.states.current_bet[rows] = committed.max(1).values
        self.states.history_parent[rows] = -1
        self.states.history_event[rows] = -1
        self.states.history_actor[rows] = -1
        self.states.history_street[rows] = -1
        self.states.history_action[rows] = -1
        self.states.history_length[rows] = 0
        self.states.fold_payoff[rows] = 0.0
        self.states.board_at_fold[rows] = 255
        self.states.fold_line_meta[rows] = 0.0
        self.states.partition_cache_key[rows] = -1
        return handles

    def rotate_button(self, states: Tensor) -> Tensor:
        """Return fresh roots ``[B]`` with each supplied button advanced once."""

        self._validate_handles(states)
        buttons = torch.remainder(
            self.states.button[states.to(torch.int64)].clone() + 1, self.n_players
        )
        return self._root_states_with_buttons(buttons)

    def _validate_handles(self, states: Tensor) -> None:
        if states.ndim != 1 or states.dtype != torch.int32:
            raise TypeError("states must be int32 [B]")
        if states.device != self.device:
            raise ValueError("states must be on the backend device")

    def classify(self, states: Tensor) -> Tensor:
        """Return int32 owner or leaf codes ``[B]``."""

        self._validate_handles(states)
        rows = states.to(torch.int64)
        phase = self.states.phase[rows]
        actor = self.states.actor[rows]
        return torch.where(
            phase == _CHANCE,
            torch.full_like(actor, CHANCE),
            torch.where(
                phase == _TERMINAL,
                torch.full_like(actor, TERMINAL),
                torch.where(phase == _PRETERM, torch.full_like(actor, PRETERM), actor),
            ),
        ).to(torch.int32)  # [B]

    owner_of = classify

    def is_terminal(self, states: Tensor) -> Tensor:
        """Return fold-terminal flags ``[B]``."""

        return self.classify(states) == TERMINAL

    def is_pre_terminal(self, states: Tensor) -> Tensor:
        """Return completion-cut flags ``[B]``."""

        return self.classify(states) == PRETERM

    def raise_targets(self, states: Tensor) -> Tensor:
        """Return every fixed-denomination raise-to amount ``[B,n_ticks]``."""

        self._validate_handles(states)
        return self.raise_unit_values[None].expand(states.shape[0], -1)

    legal_raise_targets = raise_targets

    def legal_events(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Return player event ids and legal masks ``[B,E_max]``."""

        self._validate_handles(states)
        rows = states.to(torch.int64)
        actor = self.states.actor[rows].to(torch.int64)
        ids = torch.arange(self.e_max, dtype=torch.int32, device=self.device)[None, :]
        ids = ids.expand(states.shape[0], -1).clone()  # [B,E]
        mask = torch.zeros_like(ids, dtype=torch.bool)  # [B,E]
        player = self.states.phase[rows] == _PLAYER
        to_call = self.states.current_bet[rows] - self.states.committed_street[rows, actor]
        mask[:, FOLD] = player & (to_call > _EPS)
        mask[:, CHECK_CALL] = player
        maximum = self.states.committed_street[rows, actor] + self.states.stacks[rows, actor]
        can_raise = player & self.states.reopen_mask[rows, actor] & (
            maximum > self.states.current_bet[rows] + _EPS
        )
        minimum = self.states.current_bet[rows] + self.states.last_full_raise[rows]
        short = maximum + _EPS < minimum
        targets = self.raise_unit_values[None]
        full_raises = (targets + _EPS >= minimum[:, None]) & (
            targets <= maximum[:, None] + _EPS
        )
        short_allin = torch.isclose(
            targets, maximum[:, None], rtol=0.0, atol=_EPS
        )
        mask[:, RAISE_OFFSET:] = can_raise[:, None] & torch.where(
            short[:, None], short_allin, full_raises
        )
        return ids, mask

    def step(self, states: Tensor, events: EventBatch) -> Tensor:
        """Apply player or chance events and return successor handles ``[B]``."""

        self._validate_handles(states)
        if events.public_id.shape != states.shape:
            raise ValueError("events and states must have matching shape [B]")
        phase = self.states.phase[states.to(torch.int64)]
        if "cards" in events.payload:
            torch._assert((phase == _CHANCE).all(), "card payload requires chance states")
            return self._step_chance(states, events.payload["cards"])
        torch._assert((phase == _PLAYER).all(), "player event requires player states")
        return self._step_player(states, events.public_id)

    def _step_player(self, states: Tensor, event_ids: Tensor) -> Tensor:
        """Apply batched legal player actions ``[B]``."""

        legal_ids, legal_mask = self.legal_events(states)
        selected = legal_ids == event_ids[:, None]
        torch._assert((selected & legal_mask).any(1).all(), "illegal poker event")
        parent_rows = states.to(torch.int64)
        parent_actor = self.states.actor[parent_rows].to(torch.int64)
        parent_batch = torch.arange(states.shape[0], device=self.device)
        parent_targets = self.raise_targets(states)
        parent_to_call = (
            self.states.current_bet[parent_rows]
            - self.states.committed_street[parent_rows, parent_actor]
        ).clamp_min(0.0)
        parent_maximum = (
            self.states.committed_street[parent_rows, parent_actor]
            + self.states.stacks[parent_rows, parent_actor]
        )
        parent_minimum = torch.minimum(
            parent_maximum,
            self.states.current_bet[parent_rows] + self.states.last_full_raise[parent_rows],
        )
        parent_tick = (event_ids - RAISE_OFFSET).clamp(0, self.n_ticks - 1).to(torch.int64)
        parent_raise_to = parent_targets[parent_batch, parent_tick]
        parent_call_paid = torch.minimum(
            self.states.stacks[parent_rows, parent_actor], parent_to_call
        )
        successors = self.states.append_copies(states)  # [B]
        rows = successors.to(torch.int64)
        actor = self.states.actor[rows].to(torch.int64)
        batch = torch.arange(states.shape[0], device=self.device)
        fold = event_ids == FOLD
        call = event_ids == CHECK_CALL
        raise_event = event_ids >= RAISE_OFFSET
        self.states.status[rows[fold], actor[fold]] = FOLDED
        self.states.acted[rows[fold], actor[fold]] = True
        to_call = (self.states.current_bet[rows] - self.states.committed_street[rows, actor]).clamp_min(0.0)
        addition_call = torch.minimum(self.states.stacks[rows, actor], to_call)
        selected_call = torch.where(call, addition_call, torch.zeros_like(addition_call))
        self.states.stacks[rows, actor] -= selected_call
        self.states.committed_street[rows, actor] += selected_call
        self.states.committed_total[rows, actor] += selected_call
        self.states.acted[rows[call], actor[call]] = True
        targets = parent_targets
        tick = (event_ids - RAISE_OFFSET).clamp(0, self.n_ticks - 1).to(torch.int64)
        target = targets[batch, tick]
        old_bet = self.states.current_bet[rows]
        old_committed = self.states.committed_street[rows, actor]
        addition_raise = torch.where(raise_event, target - old_committed, torch.zeros_like(target))
        self.states.stacks[rows, actor] -= addition_raise
        self.states.committed_street[rows, actor] += addition_raise
        self.states.committed_total[rows, actor] += addition_raise
        increment = target - old_bet
        full_raise = raise_event & (increment + _EPS >= self.states.last_full_raise[rows])
        self.states.current_bet[rows] = torch.where(raise_event, target, old_bet)
        self.states.last_full_raise[rows] = torch.where(
            full_raise, increment, self.states.last_full_raise[rows]
        )
        self.states.n_raises[rows] += full_raise.to(torch.int32)
        self.states.acted[rows[full_raise]] = False
        self.states.acted[rows[raise_event], actor[raise_event]] = True
        all_in = self.states.stacks[rows, actor] <= _EPS
        self.states.stacks[rows, actor] = torch.where(
            all_in, torch.zeros_like(self.states.stacks[rows, actor]), self.states.stacks[rows, actor]
        )
        self.states.status[rows, actor] = torch.where(
            all_in, torch.full_like(self.states.status[rows, actor], ALL_IN), self.states.status[rows, actor]
        )
        length = self.states.history_length[rows]
        torch._assert((length < self.max_rollout_steps).all(), "poker action history capacity exceeded")
        self.states.history_event[rows] = event_ids
        self.states.history_actor[rows] = parent_actor.to(torch.int8)
        self.states.history_street[rows] = self.states.street[parent_rows].to(torch.int8)
        self.states.history_action[rows] = torch.where(
            raise_event,
            torch.full_like(event_ids, 2),
            torch.where(call, torch.ones_like(event_ids), torch.zeros_like(event_ids)),
        ).to(torch.int8)
        self.states.history_amount[rows] = torch.where(
            raise_event,
            parent_raise_to,
            torch.where(call, parent_call_paid, torch.zeros_like(parent_call_paid)),
        )
        self.states.history_pot_before[rows] = self.states.committed_total[parent_rows].sum(1)
        self.states.history_call_amount[rows] = parent_to_call
        self.states.history_current_bet[rows] = self.states.current_bet[parent_rows]
        self.states.history_min_raise_to[rows] = parent_minimum
        self.states.history_max_raise_to[rows] = parent_maximum
        self.states.history_length[rows] += 1
        if bool(fold.any()):
            fold_rows = rows[fold]
            fold_actor = actor[fold]
            fold_board = self.states.board[parent_rows[fold]]
            self.states.board_at_fold[fold_rows, fold_actor] = torch.where(
                fold_board < 0, torch.full_like(fold_board, 255), fold_board
            ).to(torch.uint8)
            self.states.fold_line_meta[fold_rows, fold_actor] = torch.stack(
                (
                    self.states.street[parent_rows[fold]].to(torch.float32),
                    self.states.committed_total[parent_rows[fold], fold_actor],
                    self.states.committed_street[parent_rows[fold], fold_actor],
                    self.states.committed_total[parent_rows[fold]].sum(1),
                    parent_to_call[fold],
                    self.states.n_raises[parent_rows[fold]].to(torch.float32),
                    self.states.history_length[parent_rows[fold]].to(torch.float32),
                    (fold_board >= 0).sum(1).to(torch.float32),
                ),
                dim=1,
            )
            self.states.partition_cache_key[fold_rows] = successors[fold]
        self.states.reopen_mask[rows] = (~self.states.acted[rows]) & (self.states.status[rows] == LIVE)

        contesting = self.states.status[rows] != FOLDED
        contest_count = contesting.sum(1)
        fold_terminal = contest_count == 1
        winner = contesting.to(torch.int64).argmax(1)
        baseline = -self.states.committed_total[rows]
        pot = self.states.committed_total[rows].sum(1)
        payoff = baseline.clone()
        payoff[batch, winner] += pot
        self.states.fold_payoff[rows] = torch.where(
            fold_terminal[:, None], payoff, self.states.fold_payoff[rows]
        )

        offsets = torch.arange(1, self.n_players + 1, device=self.device)[None, :]
        candidates = torch.remainder(actor[:, None] + offsets, self.n_players)
        candidate_live = self.states.status[rows].gather(1, candidates) == LIVE
        candidate_committed = self.states.committed_street[rows].gather(1, candidates)
        candidate_acted = self.states.acted[rows].gather(1, candidates)
        needs_action = candidate_live & (
            (~candidate_acted) | (candidate_committed + _EPS < self.states.current_bet[rows, None])
        )
        has_next = needs_action.any(1)
        next_actor = candidates.gather(1, needs_action.to(torch.int64).argmax(1, keepdim=True)).squeeze(1)
        actionable = (self.states.status[rows] == LIVE).sum(1)
        finish_round = (~fold_terminal) & (~has_next)
        completion = finish_round & ((self.states.street[rows] == RIVER) | (actionable <= 1))
        chance = finish_round & (~completion)
        self.states.actor[rows] = torch.where(has_next, next_actor, -torch.ones_like(next_actor)).to(torch.int32)
        self.states.phase[rows] = torch.where(
            fold_terminal,
            torch.full_like(self.states.phase[rows], _TERMINAL),
            torch.where(
                completion,
                torch.full_like(self.states.phase[rows], _PRETERM),
                torch.where(chance, torch.full_like(self.states.phase[rows], _CHANCE), self.states.phase[rows]),
            ),
        )
        return successors

    def _step_chance(self, states: Tensor, cards: Tensor) -> Tensor:
        """Deal padded chance cards ``[B,3]`` and begin the next street."""

        if cards.ndim != 2 or cards.shape != (states.shape[0], 3):
            raise ValueError("chance cards must have shape [B,3]")
        successors = self.states.append_copies(states)  # [B]
        rows = successors.to(torch.int64)
        old_street = self.states.street[states.to(torch.int64)]
        flop = old_street == 0
        expected = torch.where(flop, 3, 1)
        real = cards >= 0
        torch._assert((real.sum(1) == expected).all(), "wrong number of chance cards")
        torch._assert(((cards < 52) | (~real)).all(), "chance card outside deck")
        sorted_cards = torch.sort(cards.masked_fill(~real, 127), dim=1).values.to(torch.int16)
        sorted_cards = sorted_cards.masked_fill(~real, -1)
        self.states.board[rows[flop], :3] = sorted_cards[flop, :3]
        later = ~flop
        destination = old_street[later] + 2
        self.states.board[rows[later], destination.to(torch.int64)] = cards[later, 0]
        self.states.street[rows] = old_street + 1
        self.states.committed_street[rows] = 0.0
        self.states.acted[rows] = False
        self.states.reopen_mask[rows] = self.states.status[rows] == LIVE
        self.states.last_full_raise[rows] = float(self.cfg.bb)
        self.states.current_bet[rows] = 0.0
        self.states.n_raises[rows] = 0
        start = torch.remainder(self.states.button[rows] + 1, self.n_players)
        offsets = torch.arange(self.n_players, device=self.device)[None, :]
        candidates = torch.remainder(start[:, None] + offsets, self.n_players)
        live = self.states.status[rows].gather(1, candidates.long()) == LIVE
        torch._assert(live.any(1).all(), "chance state has no actionable seat")
        actor = candidates.gather(1, live.to(torch.int64).argmax(1, keepdim=True)).squeeze(1)
        self.states.actor[rows] = actor.to(torch.int32)
        self.states.phase[rows] = _PLAYER
        length = self.states.history_length[rows]
        torch._assert((length < self.max_rollout_steps).all(), "poker action history capacity exceeded")
        self.states.history_street[rows] = self.states.street[rows].to(torch.int8)
        self.states.history_action[rows] = 3
        parent_rows = states.to(torch.int64)
        self.states.history_pot_before[rows] = self.states.committed_total[parent_rows].sum(1)
        self.states.history_current_bet[rows] = self.states.current_bet[parent_rows]
        self.states.history_length[rows] += 1
        self.states.partition_cache_key[rows] = successors
        return successors

    def _packed_history(self, states: Tensor) -> PackedPublicHistory:
        """Reconstruct parent-linked histories as a padded chronological batch."""

        rows = states.to(torch.int64)
        lengths = self.states.history_length[rows].to(torch.int64)
        width = int(lengths.max().item()) if states.numel() else 0
        shape = (states.shape[0], width)
        actor = torch.full(shape, -1, dtype=torch.int64, device=self.device)
        street = torch.full(shape, -1, dtype=torch.int64, device=self.device)
        action = torch.full(shape, -1, dtype=torch.int64, device=self.device)
        mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        floats = {
            name: torch.zeros(shape, dtype=torch.float32, device=self.device)
            for name in (
                "amount", "pot_before", "call_amount", "current_bet",
                "min_raise_to", "max_raise_to",
            )
        }
        cursor = states.to(torch.int32)
        batch = torch.arange(states.shape[0], device=self.device)
        for offset in range(width):
            position = lengths - 1 - offset
            valid = position >= 0
            safe_cursor = cursor.clamp_min(0).to(torch.int64)
            selected_batch = batch[valid]
            selected_pos = position[valid]
            selected_rows = safe_cursor[valid]
            actor[selected_batch, selected_pos] = self.states.history_actor[selected_rows].long()
            street[selected_batch, selected_pos] = self.states.history_street[selected_rows].long()
            action[selected_batch, selected_pos] = self.states.history_action[selected_rows].long()
            mask[selected_batch, selected_pos] = True
            for target, source in (
                ("amount", "history_amount"),
                ("pot_before", "history_pot_before"),
                ("call_amount", "history_call_amount"),
                ("current_bet", "history_current_bet"),
                ("min_raise_to", "history_min_raise_to"),
                ("max_raise_to", "history_max_raise_to"),
            ):
                floats[target][selected_batch, selected_pos] = getattr(self.states, source)[selected_rows]
            cursor = torch.where(valid, self.states.history_parent[safe_cursor], cursor)
        return PackedPublicHistory(
            actor=actor, street=street, action=action, mask=mask, **floats
        )

    def sample_profiles(self, states: Tensor, reach: Tensor, gen: torch.Generator) -> Tensor:
        """Draw one exact collision-conditioned profile as int32 ``[B,P]``."""

        self._validate_handles(states)
        rows = states.to(torch.int64)
        dealt = torch.ones(
            (states.shape[0], self.n_players), dtype=torch.bool, device=self.device
        )  # [B,P]
        profiles = self.profile_sampler.sample(
            reach, self.states.board[rows], dealt, 1, gen
        )
        return profiles[:, 0].to(torch.int32)  # [B,P]

    def sampled_terminal_reward(
        self, states: Tensor, profiles: Tensor, gen: torch.Generator
    ) -> Tensor:
        """Draw missing board cards and settle one sampled profile per leaf."""

        self._validate_handles(states)
        if profiles.shape != (states.shape[0], self.n_players):
            raise ValueError("profiles must have shape [B,P]")
        rows = states.to(torch.int64)
        owner = self.classify(states)
        if bool(((owner != TERMINAL) & (owner != PRETERM)).any()):
            raise ValueError("sampled terminal reward requires terminal or pre-terminal rows")
        reward = torch.empty(
            (states.shape[0], self.n_players), dtype=torch.float32, device=self.device
        )
        folded = owner == TERMINAL
        if bool(folded.any()):
            reward[folded] = self.states.fold_payoff[rows[folded]]

        showdown = owner == PRETERM
        if bool(showdown.any()):
            selected_rows = rows[showdown]
            selected_profiles = profiles[showdown].to(torch.int32)
            board = self.states.board[selected_rows].clone()
            private = self.combo_cards[selected_profiles.to(torch.int64)].reshape(
                selected_profiles.shape[0], -1
            )
            unavailable = torch.cat((board, private), dim=1)
            valid = unavailable >= 0
            blocked = torch.zeros(
                (board.shape[0], 52), dtype=torch.bool, device=self.device
            )
            batch = torch.arange(board.shape[0], device=self.device)[:, None].expand_as(valid)
            blocked[batch[valid], unavailable[valid].to(torch.int64)] = True
            keys = torch.rand(
                (board.shape[0], 52), generator=gen, device=self.device
            ).masked_fill(blocked, torch.inf)
            runout = torch.argsort(keys, dim=1)[:, :5].to(torch.int16)
            missing = board < 0
            missing_rank = (missing.to(torch.int64).cumsum(1) - 1).clamp_min(0)
            complete_board = torch.where(
                missing, runout.gather(1, missing_rank), board
            ).to(torch.uint8).contiguous()
            status = self.states.status[selected_rows]
            active = status != FOLDED
            combos = selected_profiles.clone().masked_fill(~active, -1)
            strata = self.pot_strata(states[showdown])
            # The public strata tensor is padded to P rows.  The evaluator
            # requires even zero-amount padding rows to name an eligible seat;
            # assigning all active seats there is payoff-neutral.
            empty = ~strata.stratum_players.any(2)
            evaluator_players = torch.where(
                empty[:, :, None],
                active[:, None, :].expand_as(strata.stratum_players),
                strata.stratum_players,
            )
            settlement = PotStrata(
                evaluator_players, strata.stratum_amount, strata.payoff_baseline
            )
            result = self.showdown_evaluator.evaluate(
                board_cards=complete_board,
                combo_ids=combos.contiguous(),
                active_mask=active.contiguous(),
                stratum_players=evaluator_players.contiguous(),
            )
            reward[showdown] = assemble_showdown_payoff(
                settlement, result.showdown_share
            )
        return reward

    def sample_events(
        self,
        states: Tensor,
        profiles: Tensor,
        policy_out: PolicyOutput | None,
        adapter: Any,
        gen: torch.Generator,
    ) -> EventBatch:
        """Draw player actions or profile-complement chance cards ``[B]``."""

        owners = self.classify(states)
        if policy_out is not None:
            legal_ids, legal_mask = self.legal_events(states)
            public_id = adapter.draw(
                policy_out, profiles, owners.to(torch.int64), legal_ids, legal_mask, gen
            )
            return EventBatch(public_id=public_id.to(torch.int32))
        rows = states.to(torch.int64)
        board = self.states.board[rows]
        private = self.combo_cards[profiles.to(torch.int64)].reshape(states.shape[0], -1)
        unavailable = torch.cat((board, private), dim=1)  # [B,5+2P]
        valid = unavailable >= 0
        blocked = torch.zeros((states.shape[0], 52), dtype=torch.bool, device=self.device)
        batch_rows = torch.arange(states.shape[0], device=self.device)[:, None].expand_as(valid)
        blocked[batch_rows[valid], unavailable[valid].to(torch.int64)] = True
        keys = torch.rand((states.shape[0], 52), generator=gen, device=self.device)
        keys.masked_fill_(blocked, torch.inf)
        draw = torch.argsort(keys, dim=1)[:, :3].to(torch.int16)  # [B,3]
        count = torch.where(self.states.street[rows] == 0, 3, 1)
        keep = torch.arange(3, device=self.device)[None, :] < count[:, None]
        cards = draw.masked_fill(~keep, -1)  # [B,3]
        sorted_cards = torch.sort(cards.masked_fill(cards < 0, 127), dim=1).values
        c0, c1, c2 = sorted_cards[:, 0].to(torch.int32), sorted_cards[:, 1].to(torch.int32), sorted_cards[:, 2].to(torch.int32)
        flop_label = c0 + c1 * (c1 - 1) // 2 + c2 * (c2 - 1) * (c2 - 2) // 6
        one_label = cards[:, 0].to(torch.int32)
        street = self.states.street[rows]
        label = torch.where(street == 0, flop_label + _CHANCE_OFFSETS[0], one_label + _CHANCE_OFFSETS[1])
        label = torch.where(street == 2, one_label + _CHANCE_OFFSETS[2], label)
        return EventBatch(public_id=label.to(torch.int32), payload={"cards": cards})

    def reach_update(
        self,
        reach: Tensor,
        states: Tensor,
        events: EventBatch,
        policy_out: PolicyOutput | None,
    ) -> Tensor:
        """Update action likelihoods or board compatibility ``[B,P,H]``."""

        result = reach.clone()  # [B,P,H]
        if policy_out is not None:
            legal_ids, legal_mask = self.legal_events(states)
            from pbg.games.poker.adapter import PokerPolicyAdapter

            adapter = PokerPolicyAdapter(self.n_ticks)
            selected = adapter.log_prob(
                policy_out, events.public_id[:, None], legal_ids, legal_mask
            )[:, 0]  # [B,H]
            actor = self.classify(states).to(torch.int64)
            batch = torch.arange(states.shape[0], device=self.device)
            result[batch, actor] += selected.detach().float()
            return result
        cards = events.payload["cards"]
        real = cards >= 0
        conflict = (
            (self.combo_cards[None, :, :, None] == cards[:, None, None, :])
            & real[:, None, None, :]
        ).any(3).any(2)  # [B,H]
        return result.masked_fill(conflict[:, None, :], -torch.inf)

    def replay_chance_reach(
        self, reach: Tensor, parent_states: Tensor, child_states: Tensor
    ) -> Tensor:
        """Replay a dealt-board compatibility update from successor state data.

        ``TreeBatch`` deliberately stores compact public labels only.  The poker
        state store already retains each child board, so replay can apply the
        exact same hand-conflict mask as the rollout without retaining a
        poker-specific ``EventBatch.payload[\"cards\"]`` on every tree edge.
        Reapplying already-known parent board cards is idempotent.
        """

        self._validate_handles(parent_states)
        self._validate_handles(child_states)
        if reach.ndim != 3 or reach.shape != (
            parent_states.shape[0], self.n_players, N_HANDS
        ):
            raise ValueError("reach must have shape [B,P,1326]")
        if parent_states.shape != child_states.shape:
            raise ValueError("parent_states and child_states must have matching shapes")
        child_board = self.states.board[child_states.to(torch.int64)]  # [B,5]
        real = child_board >= 0
        conflict = (
            (self.combo_cards[None, :, :, None] == child_board[:, None, None, :])
            & real[:, None, None, :]
        ).any(3).any(2)  # [B,H]
        return reach.masked_fill(conflict[:, None, :], -torch.inf)

    def encode_public(self, states: Tensor) -> PokerPublicBatch:
        """Construct model public inputs from handles ``[B]``."""

        rows = states.to(torch.int64)
        legal_ids, legal = self.legal_events(states)
        targets = self.raise_targets(states)
        fold_legal = legal[:, FOLD].to(torch.float32)
        raise_fraction = legal[:, RAISE_OFFSET:].to(torch.float32).mean(1)
        legality = torch.stack((fold_legal, legal[:, CHECK_CALL].float(), raise_fraction), dim=1)
        raise_legal = legal[:, RAISE_OFFSET:]
        lower = targets.masked_fill(~raise_legal, torch.inf).min(1).values
        upper = targets.masked_fill(~raise_legal, -torch.inf).max(1).values
        has_raise = raise_legal.any(1)
        lower = torch.where(has_raise, lower, torch.zeros_like(lower))
        upper = torch.where(has_raise, upper, torch.zeros_like(upper))
        bounds = torch.stack((lower, upper), dim=1)
        actor = self.states.actor[rows].to(torch.int64)
        call_amount = (
            self.states.current_bet[rows]
            - self.states.committed_street[rows, actor]
        ).clamp_min(0.0)
        del legal_ids
        return PokerPublicBatch(
            street=self.states.street[rows].to(torch.int64),
            board=self.states.board[rows].to(torch.int64),
            pot=self.states.committed_total[rows].sum(1),
            stacks=self.states.stacks[rows],
            committed=self.states.committed_total[rows],
            to_act=actor,
            legality_summary=legality,
            n_ticks=torch.full((states.shape[0],), self.n_ticks, dtype=torch.int64, device=self.device),
            raise_grid_bounds=bounds,
            status=self.states.status[rows].to(torch.int64),
            seat_mask=torch.ones((states.shape[0], self.n_players), dtype=torch.bool, device=self.device),
            button=self.states.button[rows].to(torch.int64),
            committed_street=self.states.committed_street[rows],
            call_amount=call_amount,
            current_bet=self.states.current_bet[rows],
            min_raise_to=lower,
            max_raise_to=upper,
            last_full_raise=self.states.last_full_raise[rows],
            legal_action_mask=legal,
            raise_ticks=targets,
            history=self._packed_history(states),
        )

    @staticmethod
    def _slice_public(pub: PokerPublicBatch, positions: Tensor) -> PokerPublicBatch:
        values: dict[str, Any] = {}
        for name in pub.__dataclass_fields__:
            value = getattr(pub, name)
            if isinstance(value, PackedPublicHistory):
                values[name] = PackedPublicHistory(
                    **{
                        field: getattr(value, field)[positions]
                        for field in value.__dataclass_fields__
                    }
                )
            else:
                values[name] = None if value is None else value[positions]
        return PokerPublicBatch(**values)

    @staticmethod
    def _combine_outputs(parts: list[tuple[Tensor, PolicyOutput]], batch: int) -> PolicyOutput:
        if not parts:
            raise ValueError("cannot combine an empty policy batch")
        sample = parts[0][1]
        raw = {
            name: torch.empty((batch, *value.shape[1:]), dtype=value.dtype, device=value.device)
            for name, value in sample.raw.items()
        }
        value = torch.empty((batch, *sample.v.shape[1:]), dtype=sample.v.dtype, device=sample.v.device)
        vvec = None if sample.vvec is None else torch.empty(
            (batch, *sample.vvec.shape[1:]), dtype=sample.vvec.dtype, device=sample.vvec.device
        )
        u = None if sample.u is None else torch.empty(
            (batch, *sample.u.shape[1:]), dtype=sample.u.dtype, device=sample.u.device
        )
        for positions, output in parts:
            for name in raw:
                raw[name][positions] = output.raw[name]
            value[positions] = output.v
            if vvec is not None and output.vvec is not None:
                vvec[positions] = output.vvec
            if u is not None and output.u is not None:
                u[positions] = output.u
        return PolicyOutput(raw=raw, v=value, vvec=vvec, u=u)

    def plan_policy_batches(
        self, model: nn.Module, states: Tensor, phase: str
    ) -> list[Tensor]:
        """Return deterministic pre/post/live-count batches within profile limits."""

        if phase not in {"training", "inference"}:
            raise ValueError("phase must be 'training' or 'inference'")
        if states.ndim != 1:
            raise ValueError("states must be one-dimensional")
        if states.numel() == 0:
            return []
        profile = self.execution_profile
        if profile is None:
            return [torch.arange(states.numel(), device=states.device)]
        source = model.module if hasattr(model, "module") else model
        post_model = getattr(source, "postflop", None)
        if post_model is None:
            return [torch.arange(states.numel(), device=states.device)]
        cfg = post_model.cfg
        rows = states.to(torch.int64)
        street = self.states.street[rows].to(torch.int64)
        live = (self.states.status[rows] != FOLDED).sum(1).to(torch.int64)
        history = self.states.history_length[rows].to(torch.int64)
        scalar_count = 3 * self.n_players + 13 + self.e_max + self.n_ticks
        pre_tokens = (
            self.n_players * N_CLASSES + scalar_count + 1 + history + cfg.k_act
        )
        combo_count = torch.where(
            street == 1,
            torch.full_like(street, 1176),
            torch.where(
                street == 2,
                torch.full_like(street, 1128),
                torch.full_like(street, 1081),
            ),
        )
        post_tokens = live * combo_count + scalar_count + 6 + history + cfg.k_act
        token_count = torch.where(street == 0, pre_tokens, post_tokens)
        family = torch.where(street == 0, torch.zeros_like(live), live)
        max_tokens = (
            profile.training_max_tokens
            if phase == "training"
            else profile.inference_max_tokens
        )
        max_pairs = (
            profile.training_max_token_pairs
            if phase == "training"
            else profile.inference_max_token_pairs
        )
        result: list[Tensor] = []
        for family_id in torch.unique(family, sorted=True).tolist():
            positions = (family == family_id).nonzero(as_tuple=False).squeeze(1)
            order = torch.argsort(token_count[positions], stable=True)
            ordered = positions[order]
            current: list[int] = []
            current_max = 0
            for position, length in zip(
                ordered.tolist(), token_count[ordered].tolist(), strict=True
            ):
                if length > max_tokens or length * length > max_pairs:
                    raise MemoryError(
                        f"one policy sequence needs {length} tokens/{length * length} "
                        f"token pairs, beyond hardware profile {profile.name!r}"
                    )
                candidate_max = max(current_max, int(length))
                candidate_batch = len(current) + 1
                if current and (
                    candidate_batch * candidate_max > max_tokens
                    or candidate_batch * candidate_max * candidate_max > max_pairs
                ):
                    result.append(torch.tensor(current, device=states.device, dtype=torch.long))
                    current = []
                    current_max = 0
                current.append(int(position))
                current_max = max(current_max, int(length))
            if current:
                result.append(torch.tensor(current, device=states.device, dtype=torch.long))
        for positions in result:
            lengths = token_count[positions]
            maximum = int(lengths.max().item())
            self.policy_batch_metrics.append(
                {
                    "phase": phase,
                    "batch_size": int(positions.numel()),
                    "max_tokens": maximum,
                    "padded_tokens": int(positions.numel()) * maximum,
                    "padded_token_pairs": int(positions.numel()) * maximum * maximum,
                }
            )
        return result

    def policy_forward(self, frozen_model: nn.Module, states: Tensor, reach: Tensor) -> PolicyOutput:
        """Route and bounded-batch policy calls, returning combo rows ``[B,H]``."""

        if not torch.is_grad_enabled() and self.execution_profile is not None:
            plans = self.plan_policy_batches(frozen_model, states, "inference")
            if len(plans) > 1:
                parts = [
                    (positions, self._policy_forward_direct(
                        frozen_model, states[positions], reach[positions]
                    ))
                    for positions in plans
                ]
                return self._combine_outputs(parts, states.shape[0])
        return self._policy_forward_direct(frozen_model, states, reach)

    def compact_decision_reach(
        self, states: Tensor, reach: Tensor
    ) -> CompactPokerDecisionReach:
        """Discard folded full ranges after their final rollout/completion use."""

        self._validate_handles(states)
        if reach.shape != (states.numel(), self.n_players, N_HANDS):
            raise ValueError("reach must have shape [D,P,1326]")
        rows = states.to(device=self.device, dtype=torch.int64)
        live = self.states.status[rows] != FOLDED  # [D,P]
        counts = live.sum(1, dtype=torch.int64)
        row_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.int64, device=self.device), counts.cumsum(0))
        )
        seats = torch.arange(self.n_players, device=self.device)[None].expand_as(live)
        return CompactPokerDecisionReach(
            row_ptr=row_ptr,
            seat_ids=seats[live].to(torch.int64),
            log_r=reach[live].contiguous(),
            n_rows=int(states.numel()),
        )

    def expand_decision_reach(
        self, compact: CompactPokerDecisionReach, rows: Tensor
    ) -> Tensor:
        """Materialize dense policy rows only for the sampled minibatch."""

        if not isinstance(compact, CompactPokerDecisionReach):
            raise TypeError("expected CompactPokerDecisionReach")
        rows = rows.to(device=self.device, dtype=torch.int64)
        if rows.ndim != 1:
            raise ValueError("decision reach rows are out of range")
        valid_rows = ((rows >= 0) & (rows < compact.n_rows)).all()
        if rows.device.type == "cuda":
            torch._assert_async(valid_rows, "decision reach rows are out of range")
        elif not bool(valid_rows):
            raise ValueError("decision reach rows are out of range")
        starts = compact.row_ptr[rows]
        counts = compact.row_ptr[rows + 1] - starts
        batch_rows = torch.repeat_interleave(
            torch.arange(rows.numel(), device=self.device), counts
        )
        repeated_starts = torch.repeat_interleave(starts, counts)
        output_starts = torch.repeat_interleave(
            counts.cumsum(0) - counts, counts
        )
        offsets = torch.arange(batch_rows.numel(), device=self.device) - output_starts
        source = repeated_starts + offsets
        result = torch.full(
            (rows.numel(), self.n_players, N_HANDS),
            -torch.inf,
            dtype=torch.float32,
            device=self.device,
        )
        result[batch_rows, compact.seat_ids[source]] = compact.log_r[source]
        return result

    def _policy_forward_direct(
        self, frozen_model: nn.Module, states: Tensor, reach: Tensor
    ) -> PolicyOutput:
        """Route one already-budgeted street-family policy batch."""

        self._validate_handles(states)
        if reach.shape != (states.shape[0], self.n_players, N_HANDS):
            raise ValueError("reach must have shape [B,P,1326]")
        finite = torch.isfinite(reach)
        gauge = reach.masked_fill(~finite, -torch.inf).max(2, keepdim=True).values
        weights = torch.exp(reach - gauge).masked_fill(~finite, 0.0)  # [B,P,H]
        pub = self.encode_public(states)
        rows = states.to(torch.int64)
        parts: list[tuple[Tensor, PolicyOutput]] = []
        preflop = (self.states.street[rows] == 0).nonzero().squeeze(1)
        if preflop.numel():
            selected = weights[preflop]
            class_ranges = torch.zeros(
                (preflop.shape[0], self.n_players, N_CLASSES),
                dtype=torch.float32, device=self.device,
            )  # [Bp,P,K]
            class_ids = self.combo_class[None, None, :].expand_as(selected)
            class_ranges.scatter_add_(2, class_ids, selected)
            pre_pub = self._slice_public(pub, preflop)
            masks = PreflopMasks(
                seat_mask=pre_pub.seat_mask,
                status=pre_pub.status,
            )
            output = frozen_model.forward_preflop(
                pre_pub, class_ranges, pre_pub.to_act, masks
            )
            if output.v.shape[1] == N_CLASSES:
                output = PolicyOutput(
                    raw={name: value[:, self.combo_class] for name, value in output.raw.items()},
                    v=output.v[:, self.combo_class], vvec=output.vvec, u=output.u,
                )
            parts.append((preflop, output))
            self.batch_size_histograms["preflop"].append(preflop.shape[0])

        post = (self.states.street[rows] != 0).nonzero().squeeze(1)
        if post.numel():
            source_model = frozen_model.module if hasattr(frozen_model, "module") else frozen_model
            post_model = getattr(source_model, "postflop", None)
            consumes_partition = bool(
                getattr(
                    post_model,
                    "requires_partition_summary",
                    getattr(getattr(post_model, "cfg", None), "evaluator_summary_dim", 0) > 0,
                )
            )
            post_street = self.states.street[rows[post]]
            n_live = (self.states.status[rows[post]] != FOLDED).sum(1)
            for street in range(1, RIVER + 1):
                compact_size = math.comb(52 - (street + 2), 2)
                for live in range(2, self.n_players + 1):
                    positions = post[(post_street == street) & (n_live == live)]
                    if positions.numel() == 0:
                        continue
                    post_pub = self._slice_public(pub, positions)
                    compatibility = self._compatibility(post_pub.board)
                    partition = None
                    if consumes_partition:
                        partition = self._partition_summary(
                            states[positions], post_pub.board, weights[positions],
                            post_pub.status, live
                        )
                    masks = PostflopMasks(
                        compatibility=compatibility,
                        seat_mask=post_pub.seat_mask,
                        combo_mask=(weights[positions] > 0),
                        status=post_pub.status,
                        evaluator_summary=partition,
                        live_seat_ids=torch.where(
                            post_pub.status != FOLDED,
                            torch.arange(self.n_players, device=self.device)[None],
                            self.n_players,
                        ).sort(1).values[:, :live],
                        compact_size=compact_size,
                    )
                    output = frozen_model.forward_postflop(
                        post_pub, weights[positions], post_pub.to_act, masks
                    )
                    parts.append((positions, output))
                    self.batch_size_histograms[
                        f"postflop_{live}"
                    ].append(positions.shape[0])
        return self._combine_outputs(parts, states.shape[0])

    def _compatibility(self, board: Tensor) -> Tensor:
        """Return board-compatible combo mask ``[B,H]``."""

        real = board >= 0
        conflict = (
            (self.combo_cards[None, :, :, None] == board[:, None, None, :])
            & real[:, None, None, :]
        ).any(3).any(2)
        return ~conflict

    def _partition_summary(
        self, states: Tensor, board: Tensor, weights: Tensor, status: Tensor, n_live: int
    ) -> PartitionSummaryBatch:
        """Reuse summaries until their street/fold dependency anchor changes."""

        rows = states.to(torch.int64)
        keys = self.states.partition_cache_key[rows].clone()
        missing_anchor = keys < 0
        if bool(missing_anchor.any()):
            keys[missing_anchor] = states[missing_anchor]
            self.states.partition_cache_key[rows[missing_anchor]] = states[missing_anchor]
        key_values = [int(value) for value in keys.detach().cpu().tolist()]
        representatives: dict[int, int] = {}
        for position, key in enumerate(key_values):
            if key not in self._partition_cache and key not in representatives:
                representatives[key] = position
        if representatives:
            selected = torch.tensor(
                list(representatives.values()), dtype=torch.long, device=self.device
            )
            computed = self._partition_summary_uncached(
                states[selected], board[selected], weights[selected], status[selected], n_live
            )
            for offset, key in enumerate(representatives):
                self._partition_cache[key] = PartitionSummaryBatch(
                    fields={name: value[offset : offset + 1] for name, value in computed.fields.items()},
                    metadata=dict(computed.metadata),
                )
        cached = [self._partition_cache[key] for key in key_values]
        field_names = tuple(cached[0].fields)
        return PartitionSummaryBatch(
            fields={
                name: torch.cat([summary.fields[name] for summary in cached], dim=0)
                for name in field_names
            },
            metadata={
                "cache": "street_or_fold",
                "cache_hits": len(key_values) - len(representatives),
                "cache_misses": len(representatives),
                "folder_api_version": computed.metadata.get("folder_api_version")
                if representatives else cached[0].metadata.get("folder_api_version"),
            },
        )

    def _partition_summary_uncached(
        self, states: Tensor, board: Tensor, weights: Tensor, status: Tensor, n_live: int
    ) -> PartitionSummaryBatch:
        """Call the sibling folder facade for dependency-distinct rows."""

        batch = board.shape[0]
        folded_mask = status == FOLDED
        order = torch.argsort(folded_mask.to(torch.int32), dim=1)
        ordered = weights.gather(
            1, order[:, :, None].expand(-1, -1, N_HANDS)
        )  # [B,P,H]
        live_weights = ordered[:, :n_live].contiguous()  # [B,L,H]
        folded_weights = ordered[:, n_live:].contiguous()  # [B,F,H]
        folders = self.n_players - n_live
        state_rows = states.to(torch.int64)
        batch_rows = state_rows[:, None].expand(-1, folders)
        folded = FoldSnapshotBatch(
            reach_weights=folded_weights,
            board_at_fold=self.states.board_at_fold[
                batch_rows, order[:, n_live:]
            ].contiguous(),
            line_meta=self.states.fold_line_meta[
                batch_rows, order[:, n_live:]
            ].contiguous(),
            folder_mask=torch.ones((batch, folders), dtype=torch.bool, device=self.device),
            seat_ids=order[:, n_live:].to(torch.int64).contiguous(),
        )
        return self.partition_adapter.summarize(
            board_cards=board, live_range_weights=live_weights, folded=folded
        )

    def beta_fn(self, states: Tensor) -> Tensor:
        """Return configured stage-fixed branching intensities as fp32 ``[B]``."""

        self._validate_handles(states)
        rows = states.to(torch.int64)
        street = self.states.street[rows]
        phase = self.states.phase[rows]
        result = torch.full(
            (states.shape[0],), float(self.beta_fix["default"]),
            dtype=torch.float32, device=self.device,
        )  # [B]
        player_names = ("preflop", "flop", "turn", "river")
        chance_names = ("chance_to_flop", "chance_to_turn", "chance_to_river")
        for index, name in enumerate(player_names):
            if name in self.beta_fix:
                result[(phase == _PLAYER) & (street == index)] = float(self.beta_fix[name])
        for index, name in enumerate(chance_names):
            if name in self.beta_fix:
                result[(phase == _CHANCE) & (street == index)] = float(self.beta_fix[name])
        return result

    def pot_strata(self, states: Tensor) -> PotStrata:
        """Construct padded public pot strata ``[B,P,P]`` and amounts ``[B,P]``."""

        rows = states.to(torch.int64)
        committed = self.states.committed_total[rows]
        levels = torch.sort(committed, dim=1).values
        previous = torch.cat((torch.zeros_like(levels[:, :1]), levels[:, :-1]), dim=1)
        contributors = (committed[:, None, :] + _EPS >= levels[:, :, None]).sum(2)
        amount = (levels - previous) * contributors.to(torch.float32)
        eligible = (
            (self.states.status[rows, None, :] != FOLDED)
            & (committed[:, None, :] + _EPS >= levels[:, :, None])
            & (amount[:, :, None] > _EPS)
        )
        return PotStrata(eligible, amount, -committed)

    def completion_inputs(self, states: Tensor, reach: Tensor) -> dict[str, Tensor]:
        """Build device-local tensors for the public sibling completion facade.

        ``states`` is int32 ``[B]`` and ``reach`` is log reach
        float32 ``[B,P,1326]``.  Returned fields match the tensor arguments of
        ``poker_gpu_evaluator.pbg.complete`` except for caller-owned seed and
        completion configuration.
        """

        self._validate_handles(states)
        if reach.shape != (states.shape[0], self.n_players, N_HANDS):
            raise ValueError("reach must have shape [B,P,1326]")
        rows = states.to(torch.int64)
        dealt = torch.ones(
            (states.shape[0], self.n_players), dtype=torch.bool, device=self.device
        )  # [B,P]
        finite = torch.isfinite(reach)
        row_max = reach.masked_fill(~finite, -torch.inf).max(2, keepdim=True).values
        weights = torch.exp(reach - row_max).masked_fill(~finite, 0.0).contiguous()
        strata = self.pot_strata(states)
        board = torch.where(
            self.states.board[rows] < 0,
            torch.full_like(self.states.board[rows], 255),
            self.states.board[rows],
        ).to(torch.uint8).contiguous()
        return {
            "board_cards": board,
            "range_weights": weights,
            "dealt_mask": dealt,
            "contender_mask": (self.states.status[rows] != FOLDED).contiguous(),
            "stratum_players": strata.stratum_players.contiguous(),
            "payoff_baseline": strata.payoff_baseline.contiguous(),
            "stratum_amount": strata.stratum_amount.contiguous(),
        }

    def reward(self, profiles: Tensor, terminal_states: Tensor) -> Tensor:
        """Return locally known fold payoff rows ``[B,P]``."""

        del profiles
        return self.states.fold_payoff[terminal_states.to(torch.int64)]
