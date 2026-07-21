"""Device-native batched no-limit hold'em rollout backend."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch
from pbg.core.observability import ExecutionMetrics
from pbg.core.execution import (
    child_path_keys,
    logical_categorical_from_valid_probability,
    logical_uniform,
)
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
)
from pbg.games.poker.folded import (
    FoldSnapshotBatch,
    PartitionSummaryBatch,
    PokerGPUPartitionAdapter,
)
from pbg.games.poker.mc import (
    PokerGPURejectionSampler,
    PokerGPUShowdownEvaluator,
    stratum_levels,
)
from pbg.games.poker.model_inputs import DeviceHistoryMaterializer
from pbg.games.poker.partition_cache import DevicePartitionSummaryCache
from pbg.games.poker.policy_planning import plan_policy_batches as _plan_policy_batches
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
    log_r: Tensor  # float32 [T,H]
    n_rows: int


class PokerRolloutBackend:
    """Batched Torch implementation of public NLHE navigation.

    Args:
        cfg: Public table configuration.
        capacity: Maximum number of state rows resident in this pipeline.
        device: Owning CPU or CUDA device.
        max_rollout_steps: Maximum public transitions in one sampled path.
        beta_table: Fixed branching intensities by public-stage bucket.
        profile_sampler: Exact sibling-facade joint-profile adapter.
        partition_adapter: Sibling-facade neutral-partition adapter.
        evaluator_backend: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Every returned tensor remains on ``device``.  The CPU and CUDA paths run
    the same state-transition code; :class:`NLHEEngine` remains the separate
    list-based correctness oracle used only by tests and evaluation.
    """

    # Folded-seat ranges still affect preflop tokens and reach-derived features.
    # Retain the exact rollout input until an immutable folded-summary cache is
    # part of the model-input contract.
    requires_full_decision_reach = True

    def __init__(
        self,
        cfg: PokerConfig,
        capacity: int,
        device: torch.device | str,
        *,
        max_rollout_steps: int | None = None,
        beta_table: Mapping[str, float] | Any | None = None,
        profile_sampler: PokerGPURejectionSampler | Any | None = None,
        partition_adapter: PokerGPUPartitionAdapter | Any | None = None,
        evaluator_backend: str = "auto",
    ) -> None:
        self.cfg = cfg
        self.geometry = cfg.geometry
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.n_players = cfg.n_players
        self.n_cards = self.geometry.n_cards
        self.n_hands = self.geometry.n_hands
        self.n_classes = self.geometry.n_classes
        self.x_sizes = (self.n_hands,) * cfg.n_players
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
        self.history_materializer = DeviceHistoryMaterializer(self.states)
        self.partition_summary_cache = DevicePartitionSummaryCache(
            capacity, max(0, cfg.n_players - 2), self.device
        )
        table = getattr(beta_table, "beta_table", beta_table)
        self.beta_table = dict(table or {"default": 1.0})
        if "default" not in self.beta_table:
            raise ValueError("beta_table requires a 'default' entry")
        if any(not math.isfinite(float(value)) or float(value) < 1.0 for value in self.beta_table.values()):
            raise ValueError("rollout beta values must be finite and at least one")
        self.profile_sampler = profile_sampler or PokerGPURejectionSampler(
            backend=evaluator_backend, rank_count=cfg.rank_count
        )
        self.showdown_evaluator = PokerGPUShowdownEvaluator(
            backend=evaluator_backend, rank_count=cfg.rank_count
        )
        self.partition_adapter = partition_adapter or PokerGPUPartitionAdapter(
            rank_count=cfg.rank_count
        )
        initial_beta = int(getattr(self.partition_adapter, "beta_rank", 16))
        self._fallback_partition_service = (
            (self.partition_adapter, self.partition_summary_cache)
            if not hasattr(self.partition_adapter, "beta_rank")
            else None
        )
        self._partition_services = {
            initial_beta: (self.partition_adapter, self.partition_summary_cache)
        }
        self._partition_materialized = {initial_beta: False}
        self.combo_cards = torch.tensor(
            self.geometry.combo_cards,
            dtype=torch.int16,
            device=self.device,
        )
        self.combo_class = torch.tensor(
            self.geometry.class_of_combo,
            dtype=torch.int64,
            device=self.device,
        )
        self.batch_size_histograms: dict[str, list[int]] = {
            "preflop": [],
            **{f"postflop_{live}": [] for live in range(2, cfg.n_players + 1)},
        }
        self.execution_profile = None
        # Aggregate counters are intentionally lossless.  The former bounded
        # deque silently discarded early calls in long iterations.
        self.execution_metrics = ExecutionMetrics()
        self.execution_families = (
            "pre",
            *(f"post_{live}" for live in range(2, cfg.n_players + 1)),
        )
        self.execution_family_values = (0, *range(2, cfg.n_players + 1))
        self.execution_family_priorities = (
            0,
            *(1 for _ in range(2, cfg.n_players + 1)),
        )
        self.game = self

    def _dynamic_positions(self, mask: Tensor, source: str) -> Tensor:
        """Compact a mask and record CUDA's implicit dynamic-shape sync."""

        if mask.device.type != "cuda":
            return mask.nonzero(as_tuple=False).squeeze(1)
        started = time.perf_counter()
        positions = mask.nonzero(as_tuple=False).squeeze(1)
        self.execution_metrics.record_host_synchronization(
            seconds=time.perf_counter() - started, source=source
        )
        return positions

    def sequential_start_states(self, count: int) -> Tensor:
        """Reset storage and return ``count`` deterministically rotated starts."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        buttons = torch.remainder(
            torch.arange(count, dtype=torch.int32, device=self.device), self.n_players
        )  # [B]
        return self._initial_states_with_buttons(buttons)

    def branched_root_state(self) -> Tensor:
        """Reset storage and return exactly one canonical root for this rank."""

        buttons = torch.zeros(1, dtype=torch.int32, device=self.device)
        return self._initial_states_with_buttons(buttons)

    def _initial_states_with_buttons(self, buttons: Tensor) -> Tensor:
        """Reset storage and initialize roots for explicit buttons ``[B]``."""

        n = buttons.shape[0]
        self.states.reset()
        for _, cache in self._partition_services.values():
            cache.reset()
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
        self.states.history_jump[rows] = -1
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
        return self._initial_states_with_buttons(buttons)

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

    def probe_fold_mask(self, states: Tensor, seats: Tensor) -> Tensor:
        """Return rows where the selected probe seat has folded."""

        self._validate_handles(states)
        if seats.shape != states.shape or seats.device != self.device:
            raise ValueError("probe seats must be device-local [B]")
        rows = states.to(torch.int64)
        return self.states.status[
            rows, seats.to(torch.int64)
        ] == FOLDED

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
        player = self._dynamic_positions(phase == _PLAYER, "poker_step_player")
        chance = self._dynamic_positions(phase == _CHANCE, "poker_step_chance")
        torch._assert(
            player.numel() + chance.numel() == states.numel(),
            "poker step requires player or chance states",
        )
        if chance.numel() and "cards" not in events.payload:
            raise ValueError("chance events require a card payload")

        # A branched frontier may contain player and chance rows at the same
        # depth.  Generic event collation pads chance-only payloads across the
        # combined batch, so payload-key presence cannot classify every row.
        successors = torch.empty_like(states)
        if player.numel():
            successors[player] = self._step_player(
                states[player], events.public_id[player]
            )
        if chance.numel():
            successors[chance] = self._step_chance(
                states[chance], events.payload["cards"][chance]
            )
        return successors

    def step_player_batch(self, states: Tensor, events: EventBatch) -> Tensor:
        """Apply a scheduler-declared homogeneous player batch without compaction."""

        self._validate_handles(states)
        if events.public_id.shape != states.shape:
            raise ValueError("events and states must have matching shape [B]")
        phase = self.states.phase[states.to(torch.int64)]
        torch._assert_async((phase == _PLAYER).all(), "player batch contains non-player state")
        return self._step_player(states, events.public_id)

    def step_chance_batch(self, states: Tensor, events: EventBatch) -> Tensor:
        """Apply a scheduler-declared homogeneous chance batch without compaction."""

        self._validate_handles(states)
        if events.public_id.shape != states.shape or "cards" not in events.payload:
            raise ValueError("chance events require matching states and card payload")
        phase = self.states.phase[states.to(torch.int64)]
        torch._assert_async((phase == _CHANCE).all(), "chance batch contains non-chance state")
        return self._step_chance(states, events.payload["cards"])

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
        torch._assert(
            (
                (
                    (cards >= self.geometry.first_card)
                    & (cards < 52)
                )
                | (~real)
            ).all(),
            "chance card outside deck",
        )
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
        """Gather parent-linked histories without a trajectory-length loop."""

        return self.history_materializer.pack(states)

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
        self.execution_metrics.record_evaluator(rows=int(states.numel()))
        return profiles[:, 0].to(torch.int32)  # [B,P]

    def sample_profiles_keyed(
        self, states: Tensor, reach: Tensor, keys: Tensor
    ) -> Tensor:
        """Draw profiles from caller-owned logical-address seeds ``[B]``."""

        self._validate_handles(states)
        if keys.shape != states.shape or keys.dtype != torch.int64:
            raise ValueError("profile keys must be int64 [B]")
        rows = states.to(torch.int64)
        dealt = torch.ones(
            (states.shape[0], self.n_players), dtype=torch.bool, device=self.device
        )
        seeded = getattr(self.profile_sampler, "sample_seeded", None)
        if callable(seeded):
            profiles = seeded(
                reach,
                self.states.board[rows],
                dealt,
                1,
                keys.to(device=self.device, dtype=torch.int64),
            )
        else:
            # Test doubles predating the keyed facade are deterministic.  Real
            # evaluator adapters expose sample_seeded and never take this path.
            fallback = torch.Generator(device=self.device).manual_seed(0)
            profiles = self.profile_sampler.sample(
                reach, self.states.board[rows], dealt, 1, fallback
            )
        self.execution_metrics.record_evaluator(rows=int(states.numel()))
        return profiles[:, 0].to(torch.int32)

    def sampled_terminal_reward(
        self, states: Tensor, profiles: Tensor, gen: torch.Generator
    ) -> Tensor:
        """Draw missing board cards and settle one sampled profile per leaf."""

        return self._sampled_terminal_reward_impl(states, profiles, gen=gen, keys=None)

    def sampled_terminal_reward_keyed(
        self, states: Tensor, profiles: Tensor, keys: Tensor
    ) -> Tensor:
        """Settle leaves with schedule-independent runout keys ``[B]``."""

        if keys.shape != states.shape or keys.dtype != torch.int64:
            raise ValueError("terminal reward keys must be int64 [B]")
        return self._sampled_terminal_reward_impl(states, profiles, gen=None, keys=keys)

    def _sampled_terminal_reward_impl(
        self,
        states: Tensor,
        profiles: Tensor,
        *,
        gen: torch.Generator | None,
        keys: Tensor | None,
    ) -> Tensor:
        """Common sampled-showdown implementation for mutable or keyed RNG."""

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
                (board.shape[0], self.n_cards),
                dtype=torch.bool,
                device=self.device,
            )
            batch = torch.arange(board.shape[0], device=self.device)[:, None].expand_as(valid)
            blocked[
                batch[valid],
                unavailable[valid].to(torch.int64) - self.geometry.first_card,
            ] = True
            if keys is None:
                if gen is None:
                    raise RuntimeError("sampled terminal reward requires an RNG source")
                scores = torch.rand(
                    (board.shape[0], self.n_cards),
                    generator=gen,
                    device=self.device,
                )
            else:
                showdown_keys = keys[showdown]
                card_ordinals = torch.arange(
                    1, self.n_cards + 1, dtype=torch.int64, device=self.device
                )
                scores = logical_uniform(
                    child_path_keys(showdown_keys[:, None], card_ordinals[None, :])
                )
            scores = scores.masked_fill(blocked, torch.inf)
            runout = (
                torch.argsort(scores, dim=1)[:, :5]
                + self.geometry.first_card
            ).to(torch.int16)
            missing = board < 0
            missing_rank = (missing.to(torch.int64).cumsum(1) - 1).clamp_min(0)
            complete_board = torch.where(
                missing, runout.gather(1, missing_rank), board
            ).to(torch.uint8).contiguous()
            status = self.states.status[selected_rows]
            active = status != FOLDED
            combos = selected_profiles.clone().masked_fill(~active, -1)
            strata = self.pot_strata(states[showdown])
            result = self.showdown_evaluator.evaluate(
                board_cards=complete_board,
                combo_ids=combos.contiguous(),
                stratum_level=stratum_levels(strata.stratum_players),
                payoff_baseline=strata.payoff_baseline.contiguous(),
                stratum_amount=strata.stratum_amount.contiguous(),
            )
            self.execution_metrics.record_evaluator(rows=int(selected_rows.numel()))
            reward[showdown] = result.payoff
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
        blocked = torch.zeros(
            (states.shape[0], self.n_cards),
            dtype=torch.bool,
            device=self.device,
        )
        batch_rows = torch.arange(states.shape[0], device=self.device)[:, None].expand_as(valid)
        blocked[
            batch_rows[valid],
            unavailable[valid].to(torch.int64) - self.geometry.first_card,
        ] = True
        keys = torch.rand(
            (states.shape[0], self.n_cards),
            generator=gen,
            device=self.device,
        )
        keys.masked_fill_(blocked, torch.inf)
        draw = (
            torch.argsort(keys, dim=1)[:, :3] + self.geometry.first_card
        ).to(torch.int16)  # [B,3]
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

    def sample_events_keyed(
        self,
        states: Tensor,
        profiles: Tensor,
        policy_out: PolicyOutput | None,
        adapter: Any,
        keys: Tensor,
    ) -> EventBatch:
        """Draw actions/cards from schedule-independent logical keys ``[B]``."""

        if keys.shape != states.shape or keys.dtype != torch.int64:
            raise ValueError("event keys must be int64 [B]")
        owners = self.classify(states)
        if policy_out is not None:
            legal_ids, legal_mask = self.legal_events(states)
            selected = adapter.select_actor_row(
                policy_out, profiles, owners.to(torch.int64)
            )
            probability = adapter.probabilities(selected, legal_ids, legal_mask)
            columns = logical_categorical_from_valid_probability(probability, keys)
            public_id = legal_ids.gather(1, columns[:, None]).squeeze(1)
            return EventBatch(public_id=public_id.to(torch.int32))

        rows = states.to(torch.int64)
        board = self.states.board[rows]
        private = self.combo_cards[profiles.to(torch.int64)].reshape(states.shape[0], -1)
        unavailable = torch.cat((board, private), dim=1)
        valid = unavailable >= 0
        blocked = torch.zeros(
            (states.shape[0], self.n_cards),
            dtype=torch.bool,
            device=self.device,
        )
        batch_rows = torch.arange(states.shape[0], device=self.device)[:, None].expand_as(valid)
        blocked[
            batch_rows[valid],
            unavailable[valid].to(torch.int64) - self.geometry.first_card,
        ] = True
        card_ordinals = torch.arange(
            1, self.n_cards + 1, dtype=torch.int64, device=self.device
        )
        card_keys = child_path_keys(keys[:, None], card_ordinals[None, :])
        scores = logical_uniform(card_keys).masked_fill(blocked, torch.inf)
        draw = (
            torch.argsort(scores, dim=1)[:, :3] + self.geometry.first_card
        ).to(torch.int16)
        count = torch.where(self.states.street[rows] == 0, 3, 1)
        keep = torch.arange(3, device=self.device)[None, :] < count[:, None]
        cards = draw.masked_fill(~keep, -1)
        sorted_cards = torch.sort(cards.masked_fill(cards < 0, 127), dim=1).values
        c0 = sorted_cards[:, 0].to(torch.int32)
        c1 = sorted_cards[:, 1].to(torch.int32)
        c2 = sorted_cards[:, 2].to(torch.int32)
        flop_label = c0 + c1 * (c1 - 1) // 2 + c2 * (c2 - 1) * (c2 - 2) // 6
        one_label = cards[:, 0].to(torch.int32)
        street = self.states.street[rows]
        label = torch.where(
            street == 0, flop_label + _CHANCE_OFFSETS[0], one_label + _CHANCE_OFFSETS[1]
        )
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
        t = None if sample.t is None else torch.empty(
            (batch, *sample.t.shape[1:]), dtype=sample.t.dtype, device=sample.t.device
        )
        cost = None if sample.c is None else torch.empty(
            (batch, *sample.c.shape[1:]), dtype=sample.c.dtype, device=sample.c.device
        )
        for positions, output in parts:
            for name in raw:
                raw[name][positions] = output.raw[name]
            value[positions] = output.v
            if t is not None and output.t is not None:
                t[positions] = output.t
            if cost is not None and output.c is not None:
                cost[positions] = output.c
        return PolicyOutput(raw=raw, v=value, t=t, c=cost)

    def _planned_policy_batches(
        self, model: nn.Module, states: Tensor, phase: str
    ) -> list[tuple[Tensor, str]]:
        """Return deterministic bounded batches with host-known family names."""

        if phase not in {"training", "inference"}:
            raise ValueError("phase must be 'training' or 'inference'")
        if states.ndim != 1:
            raise ValueError("states must be one-dimensional")
        if states.numel() == 0:
            return []
        profile = self.execution_profile
        if profile is None:
            return [(torch.arange(states.numel(), device=states.device), "")]
        source = model.module if hasattr(model, "module") else model
        post_model = getattr(source, "postflop", None)
        if post_model is None:
            return [(torch.arange(states.numel(), device=states.device), "")]
        token_count, family = self._policy_work_rows(model, states)
        return [
            (plan.positions, plan.family)
            for plan in _plan_policy_batches(
                token_count,
                family,
                phase=phase,
                profile=profile,
                metrics=self.execution_metrics,
            )
        ]

    def plan_policy_batches(
        self, model: nn.Module, states: Tensor, phase: str
    ) -> list[Tensor]:
        """Return deterministic pre/post/live-count batches within profile limits."""

        return [
            positions
            for positions, _ in self._planned_policy_batches(model, states, phase)
        ]

    def _policy_work_rows(
        self, model: nn.Module, states: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return per-row padded-work lengths and execution-family ids."""

        source = model.module if hasattr(model, "module") else model
        post_model = getattr(source, "postflop", None)
        k_act = int(getattr(getattr(post_model, "cfg", None), "k_act", 0))
        rows = states.to(torch.int64)
        street = self.states.street[rows].to(torch.int64)
        live = (self.states.status[rows] != FOLDED).sum(1).to(torch.int64)
        history = self.states.history_length[rows].to(torch.int64)
        if self.device.type == "cuda":
            history = torch.full_like(history, self.max_rollout_steps)
        scalar_count = 3 * self.n_players + 13 + self.e_max + self.n_ticks
        pre_tokens = (
            self.n_players * self.n_classes
            + scalar_count
            + 1
            + history
            + k_act
        )
        combo_count = torch.tensor(
            [
                self.n_hands,
                math.comb(self.n_cards - 3, 2),
                math.comb(self.n_cards - 4, 2),
                math.comb(self.n_cards - 5, 2),
            ],
            dtype=torch.int64,
            device=self.device,
        )[street.clamp(0, 3)]
        post_tokens = live * combo_count + scalar_count + 6 + history + k_act
        return (
            torch.where(street == 0, pre_tokens, post_tokens),
            torch.where(street == 0, torch.zeros_like(live), live),
        )

    def rollout_family_ids(self, states: Tensor) -> Tensor:
        """Return ``0=pre`` or ``A=post_A`` without street partitioning."""

        rows = states.to(torch.int64)
        street = self.states.street[rows]
        live = (self.states.status[rows] != FOLDED).sum(1).to(torch.int64)
        return torch.where(street == 0, torch.zeros_like(live), live)

    def training_partition_specs(self):
        """Return poker-owned natural-width partitions for generic training."""

        from pbg.games.poker.partitions import poker_partition_specs

        def router(states: Any) -> int | Tensor:
            routed = self.rollout_family_ids(states)
            if isinstance(routed, Tensor):
                return torch.where(routed == 0, routed, routed - 1).to(torch.int64)
            value = int(routed)
            return 0 if value == 0 else value - 1

        forwards = {
            partition_id: (
                lambda model, inputs, partition_id=partition_id: self.policy_forward_partition(
                    model, partition_id, inputs
                )
            )
            for partition_id in range(self.n_players)
        }
        return poker_partition_specs(
            rank_count=self.cfg.rank_count,
            n_players=self.n_players,
            router=router,
            forwards=forwards,
        )

    def rollout_row_cost(
        self, model: nn.Module, states: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return padded-token width and non-model workspace estimates ``[B]``."""

        source = model.module if hasattr(model, "module") else model
        post_model = getattr(source, "postflop", None)
        k_act = int(getattr(getattr(post_model, "cfg", None), "k_act", 0))
        rows = states.to(torch.int64)
        street = self.states.street[rows].to(torch.int64)
        live = (self.states.status[rows] != FOLDED).sum(1).to(torch.int64)
        history = self.states.history_length[rows].to(torch.int64)
        if self.device.type == "cuda":
            history = torch.full_like(history, self.max_rollout_steps)
        scalar_count = 3 * self.n_players + 13 + self.e_max + self.n_ticks
        pre_tokens = (
            self.n_players * self.n_classes
            + scalar_count
            + 1
            + history
            + k_act
        )
        # post_A deliberately mixes flop/turn/river in one invocation. Its
        # compact tensor is therefore padded to the flop width for every row;
        # only the realized history component varies within the family.
        post_tokens = (
            live * math.comb(self.n_cards - 3, 2)
            + scalar_count
            + 6
            + history
            + k_act
        )
        tokens = torch.where(street == 0, pre_tokens, post_tokens)
        return tokens, torch.zeros_like(tokens)


    def _record_policy_model_call(
        self,
        model: nn.Module,
        states: Tensor,
        positions: Tensor,
        family: str,
    ) -> None:
        """Aggregate one actual direct model call, including padded work."""

        if self.device.type == "cuda":
            maximum = self._family_maximum_width(model, family)
        else:
            token_count, _ = self._policy_work_rows(model, states[positions])
            maximum = int(token_count.max())
        rows = int(positions.numel())
        self.execution_metrics.record_model_call(
            "training" if torch.is_grad_enabled() else "inference",
            family,
            rows,
            padded_tokens=rows * maximum,
            padded_token_pairs=rows * maximum * maximum,
        )

    def _select_partition_service(self, model: nn.Module) -> None:
        """Select a model-compatible folded-player summary cache lazily."""

        source = model.module if hasattr(model, "module") else model
        post = getattr(source, "postflop", None)
        beta = int(getattr(getattr(post, "cfg", None), "folder_beta_rank", 16))
        service = self._partition_services.get(beta)
        if service is None and self._fallback_partition_service is not None:
            service = self._fallback_partition_service
            self._partition_services[beta] = service
        if service is None:
            service = (
                PokerGPUPartitionAdapter(
                    beta_rank=beta, rank_count=self.cfg.rank_count
                ),
                DevicePartitionSummaryCache(
                    self.states.capacity,
                    max(0, self.n_players - 2),
                    self.device,
                ),
            )
            self._partition_services[beta] = service
        self.partition_adapter, self.partition_summary_cache = service

    def register_policy_model(self, model: nn.Module) -> None:
        """Register one policy's folded-summary geometry for memory planning."""

        self._select_partition_service(model)
        source = model.module if hasattr(model, "module") else model
        post = getattr(source, "postflop", None)
        cfg = getattr(post, "cfg", None)
        beta = int(getattr(cfg, "folder_beta_rank", 16))
        materialized = int(getattr(cfg, "evaluator_summary_dim", 0)) > 0
        self._partition_materialized[beta] = (
            self._partition_materialized.get(beta, False) or materialized
        )

    def partition_cache_memory_bound(self) -> int:
        """Return the bound for every registered model-compatible cache."""

        total = 0
        for beta, materialized in self._partition_materialized.items():
            total += (
                DevicePartitionSummaryCache.storage_nbytes(
                    self.states.capacity,
                    n_cards=self.geometry.n_cards,
                    beta_rank=beta,
                    max_folders=max(0, self.n_players - 2),
                )
                if materialized
                else self.states.capacity
            )
        return total

    def policy_forward(self, frozen_model: nn.Module, states: Tensor, reach: Tensor) -> PolicyOutput:
        """Route and bounded-batch policy calls, returning combo rows ``[B,H]``."""

        self.register_policy_model(frozen_model)
        if not torch.is_grad_enabled() and self.execution_profile is not None:
            plans = self._planned_policy_batches(
                frozen_model, states, "inference"
            )
            if all(family for _, family in plans):
                parts = [
                    (
                        positions,
                        self.policy_forward_family(
                            frozen_model,
                            family,
                            states[positions],
                            reach[positions],
                        ),
                    )
                    for positions, family in plans
                ]
                return self._combine_outputs(parts, states.shape[0])
        return self._policy_forward_direct(frozen_model, states, reach)

    def policy_forward_family(
        self,
        frozen_model: nn.Module,
        family: str,
        states: Tensor,
        reach: Tensor,
    ) -> PolicyOutput:
        """Execute one already-routed family, allowing mixed postflop streets."""

        self._select_partition_service(frozen_model)

        self._validate_handles(states)
        if states.ndim != 1 or states.numel() == 0:
            raise ValueError("a family policy call requires a nonempty state vector")
        if reach.shape != (states.shape[0], self.n_players, self.n_hands):
            raise ValueError("reach has the wrong compact hand width")
        rows = states.to(torch.int64)
        street = self.states.street[rows]
        live = (self.states.status[rows] != FOLDED).sum(1)
        if family == "pre":
            expected = street == 0
        elif family.startswith("post_"):
            try:
                expected_live = int(family.removeprefix("post_"))
            except ValueError as exc:
                raise ValueError(f"invalid poker execution family {family!r}") from exc
            if not 2 <= expected_live <= self.n_players:
                raise ValueError(f"invalid poker execution family {family!r}")
            expected = (street != 0) & (live == expected_live)
        else:
            raise ValueError(f"invalid poker execution family {family!r}")
        if states.device.type == "cuda":
            torch._assert_async(expected.all(), "family queue contains a misrouted poker row")
        elif not bool(expected.all()):
            raise ValueError("family queue contains a misrouted poker row")

        finite = torch.isfinite(reach)
        gauge = reach.masked_fill(~finite, -torch.inf).max(2, keepdim=True).values
        weights = torch.exp(reach - gauge).masked_fill(~finite, 0.0)
        pub = self.encode_public(states)
        positions = torch.arange(states.numel(), device=self.device)
        if family == "pre":
            class_ranges = torch.zeros(
                (states.numel(), self.n_players, self.n_classes),
                dtype=torch.float32,
                device=self.device,
            )
            class_ids = self.combo_class[None, None, :].expand_as(weights)
            class_ranges.scatter_add_(2, class_ids, weights)
            masks = PreflopMasks(seat_mask=pub.seat_mask, status=pub.status)
            output = frozen_model.forward_preflop(pub, class_ranges, pub.to_act, masks)
            if output.v.shape[1] == self.n_classes:
                output = PolicyOutput(
                    raw={
                        name: value[:, self.combo_class]
                        for name, value in output.raw.items()
                    },
                    v=output.v[:, self.combo_class],
                    t=output.t,
                    c=output.c,
                )
            self.batch_size_histograms["preflop"].append(states.numel())
        else:
            source = frozen_model.module if hasattr(frozen_model, "module") else frozen_model
            post_model = getattr(source, "postflop", None)
            consumes_partition = bool(
                getattr(
                    post_model,
                    "requires_partition_summary",
                    getattr(getattr(post_model, "cfg", None), "evaluator_summary_dim", 0) > 0,
                )
            )
            expected_live = int(family.removeprefix("post_"))
            compatibility = self._compatibility(pub.board)
            partition = None
            if consumes_partition:
                partition = self._partition_summary(
                    states, pub.board, weights, pub.status, expected_live
                )
            live_seat_ids = torch.where(
                pub.status != FOLDED,
                torch.arange(self.n_players, device=self.device)[None],
                self.n_players,
            ).sort(1).values[:, :expected_live]
            masks = PostflopMasks(
                compatibility=compatibility,
                seat_mask=pub.seat_mask,
                combo_mask=weights > 0,
                status=pub.status,
                evaluator_summary=partition,
                live_seat_ids=live_seat_ids,
                # Flop has the widest compatible set. Turn/river rows are
                # padded and masked inside this same post_A invocation.
                compact_size=math.comb(self.n_cards - 3, 2),
            )
            output = frozen_model.forward_postflop(pub, weights, pub.to_act, masks)
            self.batch_size_histograms[f"postflop_{expected_live}"].append(
                states.numel()
            )

        # Family routing and row count are host-known at this declared launch
        # boundary. Padded work uses the calibrated worst-case family width and
        # therefore introduces no hidden device scalar read.
        width = self._family_maximum_width(frozen_model, family)
        phase = "training" if torch.is_grad_enabled() else "inference"
        self.execution_metrics.record_model_call(
            phase,
            family,
            states.numel(),
            padded_tokens=states.numel() * width,
            padded_token_pairs=states.numel() * width * width,
        )
        return output

    def policy_forward_partition(
        self,
        frozen_model: nn.Module,
        partition_id: int,
        inputs: Mapping[str, Tensor],
    ) -> PolicyOutput:
        """Forward one sealed natural-width learner partition.

        Preflop stays on ``K=R**2`` rows. Postflop stores only live ``[A,H]``
        ranges and reads evaluator-owned folded summaries through the immutable
        dependency anchor populated during rollout.
        """

        states = inputs.get("state_ref")
        natural = inputs.get("range_weights")
        if not isinstance(states, Tensor) or not isinstance(natural, Tensor):
            raise TypeError("sealed poker inputs require state_ref and range_weights")
        states = states.to(device=self.device, dtype=torch.int32)
        rows = states.to(torch.int64)
        pub = self.encode_public(states)
        if partition_id == 0:
            expected = (states.numel(), self.n_players, self.n_classes)
            if tuple(natural.shape) != expected:
                raise ValueError(f"preflop natural ranges must have shape {expected}")
            if not bool((self.states.street[rows] == 0).all()):
                raise ValueError("preflop partition contains a postflop state")
            source = frozen_model.module if hasattr(frozen_model, "module") else frozen_model
            output = source.preflop(
                pub,
                natural.to(torch.float32),
                pub.to_act,
                PreflopMasks(seat_mask=pub.seat_mask, status=pub.status),
            )
            if output.v.shape[1] != self.n_classes:
                raise ValueError("preflop learner forward expanded its natural domain")
            family = "pre"
        else:
            live = int(partition_id) + 1
            expected = (states.numel(), live, self.n_hands)
            if tuple(natural.shape) != expected:
                raise ValueError(f"postflop natural ranges must have shape {expected}")
            status = self.states.status[rows]
            actual_live = (status != FOLDED).sum(1)
            if not bool(((self.states.street[rows] != 0) & (actual_live == live)).all()):
                raise ValueError("postflop partition router/domain mismatch")
            live_seat_ids = torch.where(
                status != FOLDED,
                torch.arange(self.n_players, device=self.device)[None],
                self.n_players,
            ).sort(1).values[:, :live]
            weights = torch.ones(
                (states.numel(), self.n_players, self.n_hands),
                dtype=torch.float32,
                device=self.device,
            )
            batch = torch.arange(states.numel(), device=self.device)[:, None]
            weights[batch, live_seat_ids] = natural.to(torch.float32)
            source = frozen_model.module if hasattr(frozen_model, "module") else frozen_model
            post_model = getattr(source, "postflop", None)
            consumes_partition = bool(
                getattr(
                    post_model,
                    "requires_partition_summary",
                    getattr(getattr(post_model, "cfg", None), "evaluator_summary_dim", 0) > 0,
                )
            )
            summary = (
                self._partition_summary(states, pub.board, weights, status, live)
                if consumes_partition
                else None
            )
            masks = PostflopMasks(
                compatibility=self._compatibility(pub.board),
                seat_mask=pub.seat_mask,
                combo_mask=weights > 0,
                status=pub.status,
                evaluator_summary=summary,
                live_seat_ids=live_seat_ids,
                compact_size=math.comb(self.n_cards - 3, 2),
            )
            output = frozen_model.forward_postflop(pub, weights, pub.to_act, masks)
            if output.v.shape[1] != self.n_hands:
                raise ValueError("postflop learner forward changed the canonical domain")
            family = f"post_{live}"
        width = self._family_maximum_width(frozen_model, family)
        self.execution_metrics.record_model_call(
            "training" if torch.is_grad_enabled() else "inference",
            family,
            states.numel(),
            padded_tokens=states.numel() * width,
            padded_token_pairs=states.numel() * width * width,
        )
        return output

    def policy_forward_family_actual_cost(
        self,
        frozen_model: nn.Module,
        family: str,
        states: Tensor,
        reach: Tensor,
    ) -> PolicyOutput:
        """Greedily split one logical family block using its actual row costs.

        The cost vector is the sole device-to-host scheduler control transfer
        for the logical block.  Planning on that fixed-size vector amortizes
        the boundary across all physical calls and keeps queue/RNG identities
        independent of the resulting width order.
        """

        if states.ndim != 1 or states.numel() == 0:
            raise ValueError("a family policy block requires a nonempty state vector")
        # A singleton has exactly one possible physical plan.  Avoid reading a
        # cost vector that cannot change that plan; narrow branched trees hit
        # this case frequently.
        if states.numel() == 1:
            return self.policy_forward_family(frozen_model, family, states, reach)
        profile = self.execution_profile
        if profile is None:
            return self.policy_forward_family(frozen_model, family, states, reach)

        # CUDA history tensors are materialized at the family-wide bounded
        # width, so every row in this already-routed family has identical
        # padded cost. The queue calibration has already enforced the profile;
        # a width-vector host transfer cannot improve this physical plan.
        if self.device.type == "cuda":
            return self.policy_forward_family(frozen_model, family, states, reach)

        widths, _ = self.rollout_row_cost(frozen_model, states)
        host_widths = widths

        max_tokens = int(profile.inference_max_tokens)
        max_pairs = int(profile.inference_max_token_pairs)
        # Keep indexed input/output destinations on the rollout device.  The
        # current prototype still reads the fixed-size width vector to plan
        # greedy boundaries, but reconstructing every physical pack from a
        # Python list used to add one H2D index transfer per model call.  A
        # stable device argsort has precisely the requested width/insertion-order tiebreak
        # order and lets all spans below remain device views.
        ordered_positions = torch.argsort(widths, stable=True)
        # Do not transfer the permutation back merely to plan its boundaries:
        # the already-declared width vector supplies the same stable ordering
        # on the host, while ``ordered_positions`` remains the sole index
        # source used for device gathers/scatters.
        ordered = sorted(
            range(states.numel()), key=lambda position: (int(host_widths[position]), position)
        )
        plans: list[tuple[int, int]] = []
        current_start = 0
        current_rows = 0
        current_max = 0
        for ordered_index in range(states.numel()):
            position = ordered[ordered_index]
            width = int(host_widths[position])
            if width > max_tokens or width * width > max_pairs:
                raise MemoryError(
                    f"one {family} row needs {width} tokens/{width * width} token pairs"
                )
            candidate_max = max(current_max, width)
            candidate_rows = current_rows + 1
            if current_rows and (
                candidate_rows * candidate_max > max_tokens
                or candidate_rows * candidate_max * candidate_max > max_pairs
            ):
                plans.append((current_start, current_rows))
                current_start = ordered_index
                current_rows = 0
                current_max = 0
            current_rows += 1
            current_max = max(current_max, width)
        if current_rows:
            plans.append((current_start, current_rows))

        if len(plans) == 1 and plans[0] == (0, states.numel()):
            return self.policy_forward_family(frozen_model, family, states, reach)

        parts: list[tuple[Tensor, PolicyOutput]] = []
        for start, count in plans:
            positions = ordered_positions.narrow(0, start, count)
            parts.append(
                (
                    positions,
                    self.policy_forward_family(
                        frozen_model, family, states[positions], reach[positions]
                    ),
                )
            )
        return self._combine_outputs(parts, states.numel())

    def _family_maximum_width(self, model: nn.Module, family: str) -> int:
        """Return the host-known padded width used by queue calibration."""

        source = model.module if hasattr(model, "module") else model
        post_model = getattr(source, "postflop", None)
        k_act = int(getattr(getattr(post_model, "cfg", None), "k_act", 0))
        scalar_count = 3 * self.n_players + 13 + self.e_max + self.n_ticks
        if family == "pre":
            return (
                self.n_players * self.n_classes
                + scalar_count
                + 1
                + self.max_rollout_steps
                + k_act
            )
        live = int(family.removeprefix("post_"))
        return (
            live * math.comb(self.n_cards - 3, 2)
            + scalar_count
            + 6
            + self.max_rollout_steps
            + k_act
        )

    def training_transient_bytes(
        self, model: nn.Module, family: str, rows: int, *, amp: bool
    ) -> int:
        """Conservative activation/workspace bound for one learner subspan."""

        if isinstance(rows, bool) or rows <= 0:
            raise ValueError("training workspace rows must be positive")
        source = model.module if hasattr(model, "module") else model
        branch = source.preflop if family == "pre" else source.postflop
        cfg = branch.cfg
        tokens = self._family_maximum_width(source, family)
        layers = int(cfg.n_layers_pre if family == "pre" else cfg.n_layers_post)
        scalar_bytes = 2 if amp else 4
        # Match the configured bounded-memory attention implementation.  Flex
        # attention does not materialize the quadratic score/probability
        # matrices; the old unconditional dense estimate overstated measured
        # postflop learner peaks by roughly an order of magnitude and reduced
        # the production physical batch to three rows.  Chunked dense retains
        # only one query chunk's fp32 score/probability workspace at a time.
        profile = self.execution_profile
        backend = "auto" if profile is None else str(profile.attention_backend)
        query_chunk = 128 if profile is None else int(profile.query_chunk_size)
        d_head = int(cfg.d_model) // int(cfg.n_heads)
        uses_fused_attention = backend in {"flex", "sdpa"} or (
            backend == "auto" and d_head >= 16
        )
        attention = (
            8 * tokens * int(cfg.d_model) * scalar_bytes
            if uses_fused_attention
            else 2
            * int(cfg.n_heads)
            * min(tokens, query_chunk)
            * tokens
            * 4
        )
        ffn_hidden = int(cfg.ffn_hidden or (4 * cfg.d_model))
        features = tokens * (
            12 * int(cfg.d_model) + 4 * ffn_hidden
        ) * scalar_bytes
        per_row = layers * (attention + features)
        natural = (
            self.n_players * self.n_classes
            if family == "pre"
            else (int(family.removeprefix("post_")) * self.n_hands)
        ) * 4
        # A 2x envelope covers gathered inputs, loss tensors, fused-attention
        # workspaces, allocator bins, and the measured forward/backward delta
        # while retaining the hardware profile's independent VRAM reserve.
        return math.ceil(rows * (per_row + natural) * 2.0)

    def compact_decision_reach(
        self, states: Tensor, reach: Tensor
    ) -> CompactPokerDecisionReach:
        """Discard folded full ranges after their final rollout/completion use."""

        self._validate_handles(states)
        if reach.shape != (states.numel(), self.n_players, self.n_hands):
            raise ValueError("reach has the wrong compact hand width")
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
            (rows.numel(), self.n_players, self.n_hands),
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
        if reach.shape != (states.shape[0], self.n_players, self.n_hands):
            raise ValueError("reach has the wrong compact hand width")
        finite = torch.isfinite(reach)
        gauge = reach.masked_fill(~finite, -torch.inf).max(2, keepdim=True).values
        weights = torch.exp(reach - gauge).masked_fill(~finite, 0.0)  # [B,P,H]
        pub = self.encode_public(states)
        rows = states.to(torch.int64)
        parts: list[tuple[Tensor, PolicyOutput]] = []
        preflop = self._dynamic_positions(
            self.states.street[rows] == 0, "poker_forward_preflop"
        )
        if preflop.numel():
            selected = weights[preflop]
            class_ranges = torch.zeros(
                (preflop.shape[0], self.n_players, self.n_classes),
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
            if output.v.shape[1] == self.n_classes:
                output = PolicyOutput(
                    raw={name: value[:, self.combo_class] for name, value in output.raw.items()},
                    v=output.v[:, self.combo_class], t=output.t, c=output.c,
                )
            parts.append((preflop, output))
            self.batch_size_histograms["preflop"].append(preflop.shape[0])
            self._record_policy_model_call(
                frozen_model, states, preflop, "pre"
            )

        post = self._dynamic_positions(
            self.states.street[rows] != 0, "poker_forward_postflop"
        )
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
                compact_size = math.comb(
                    self.n_cards - (street + 2), 2
                )
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
                    self._record_policy_model_call(
                        frozen_model, states, positions, f"post_{live}"
                    )
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
        """Merge evaluator rows through device-indexed dependency anchors."""

        rows = states.to(torch.int64)
        inherited = self.states.partition_cache_key[rows]
        keys = torch.where(inherited >= 0, inherited, states).to(torch.int64)
        self.states.partition_cache_key[rows] = keys.to(torch.int32)
        hits = self.partition_summary_cache.contains(keys)
        compute_mask = ~hits
        # Cached rows may arrive after folded full reaches were discarded. Give
        # the masked evaluator rows finite dummy weights; their results are
        # ignored by the device merge and the real cached tensors are gathered.
        evaluator_weights = torch.where(
            hits[:, None, None], torch.ones_like(weights), weights
        )
        computed = self._partition_summary_uncached(
            states,
            board,
            evaluator_weights,
            status,
            n_live,
            compute_mask=compute_mask,
        )
        return self.partition_summary_cache.merge(
            keys=keys,
            computed=computed,
            compute_mask=compute_mask,
            folder_count=self.n_players - n_live,
        )

    def _partition_summary_uncached(
        self,
        states: Tensor,
        board: Tensor,
        weights: Tensor,
        status: Tensor,
        n_live: int,
        *,
        compute_mask: Tensor | None = None,
    ) -> PartitionSummaryBatch:
        """Call the sibling folder facade for dependency-distinct rows."""

        batch = board.shape[0]
        folded_mask = status == FOLDED
        order = torch.argsort(folded_mask.to(torch.int32), dim=1)
        ordered = weights.gather(
            1, order[:, :, None].expand(-1, -1, self.n_hands)
        )  # [B,P,H]
        live_weights = ordered[:, :n_live].contiguous()  # [B,L,H]
        folded_weights = ordered[:, n_live:].contiguous()  # [B,F,H]
        folders = self.n_players - n_live
        if compute_mask is None:
            compute_mask = torch.ones(batch, dtype=torch.bool, device=self.device)
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
            folder_mask=compute_mask[:, None].expand(-1, folders).contiguous(),
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
            (states.shape[0],), float(self.beta_table["default"]),
            dtype=torch.float32, device=self.device,
        )  # [B]
        player_names = ("preflop", "flop", "turn", "river")
        chance_names = ("chance_to_flop", "chance_to_turn", "chance_to_river")
        for index, name in enumerate(player_names):
            if name in self.beta_table:
                result[(phase == _PLAYER) & (street == index)] = float(self.beta_table[name])
        for index, name in enumerate(chance_names):
            if name in self.beta_table:
                result[(phase == _CHANCE) & (street == index)] = float(self.beta_table[name])
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
        """Build device-local tensors for bound sibling completion.

        ``states`` is int32 ``[B]`` and ``reach`` is log reach
        float32 ``[B,P,H]``.  Returned fields match the tensor arguments of
        ``evaluator.complete`` except for caller-owned seed and sample count.
        """

        self._validate_handles(states)
        if reach.shape != (states.shape[0], self.n_players, self.n_hands):
            raise ValueError("reach has the wrong compact hand width")
        rows = states.to(torch.int64)
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
            "stratum_level": stratum_levels(strata.stratum_players),
            "payoff_baseline": strata.payoff_baseline.contiguous(),
            "stratum_amount": strata.stratum_amount.contiguous(),
        }

    def reward(self, profiles: Tensor, terminal_states: Tensor) -> Tensor:
        """Return locally known fold payoff rows ``[B,P]``."""

        del profiles
        return self.states.fold_payoff[terminal_states.to(torch.int64)]
