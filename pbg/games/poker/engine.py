"""Public no-limit hold'em navigation and affine pot accounting.

The engine knows chips, seats, actions, and public cards. It intentionally
does not know how to compare private hands; production showdown payoffs cross
the evaluator boundary already assembled. :func:`assemble_showdown_payoff`
remains a public affine-accounting validator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL
from pbg.games.poker.cards import DeckGeometry

FOLD = 0
CHECK_CALL = 1
RAISE_OFFSET = 2

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3

LIVE = 0
FOLDED = 1
ALL_IN = 2

_PLAYER = 0
_CHANCE = 1
_TERMINAL = 2
_PRETERM = 3
_EPS = 1e-6


@dataclass(frozen=True)
class PokerConfig:
    """Configuration for one fixed compact-deck NLHE cash-game run."""

    n_players: int = 6
    stack_bb: float = 100.0
    rank_count: int = 13
    n_ticks: int = field(init=False)
    geometry: DeckGeometry = field(init=False, repr=False)

    sb: ClassVar[float] = 0.5
    bb: ClassVar[float] = 1.0
    chip_unit_bb: ClassVar[float] = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_players, bool)
            or not isinstance(self.n_players, int)
            or not 2 <= self.n_players <= 6
        ):
            raise ValueError("n_players must be an integer in [2,6]")
        if not math.isfinite(self.stack_bb) or not 8.0 <= self.stack_bb <= 128.0:
            raise ValueError("stack_bb must be finite and in [8,128]")
        ticks = round(self.stack_bb / self.chip_unit_bb)
        if not math.isclose(
            self.stack_bb, ticks * self.chip_unit_bb, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("stack_bb must be an integer multiple of 0.5 bb")
        geometry = DeckGeometry(self.rank_count)
        if 2 * self.n_players + 5 >= geometry.n_cards:
            raise ValueError("compact deck must satisfy 2*n_players + 5 < 4*rank_count")
        object.__setattr__(self, "n_ticks", ticks)
        object.__setattr__(self, "geometry", geometry)


@dataclass(frozen=True)
class PublicHistoryEvent:
    """One lossless public action or street-separator record."""

    actor: int
    street: int
    action: int
    amount: float
    pot_before: float
    call_amount: float
    current_bet: float
    min_raise_to: float
    max_raise_to: float


@dataclass(frozen=True)
class _State:
    street: int
    board: tuple[int, ...]
    stacks: tuple[float, ...]
    committed_street: tuple[float, ...]
    committed_total: tuple[float, ...]
    status: tuple[int, ...]
    actor: int
    last_full_raise: float
    current_bet: float
    acted: tuple[bool, ...]
    button: int
    phase: int
    betting_closed: bool
    n_raises: int
    action_history: tuple[int, ...]
    fold_payoff: tuple[float, ...] | None = None
    public_history: tuple[PublicHistoryEvent, ...] = ()


class StateTable:
    """Append-only structure-of-arrays table addressed by int32 references."""

    def __init__(self, n_players: int) -> None:
        self.n_players = n_players
        self.street: list[int] = []
        self.board: list[tuple[int, ...]] = []
        self.stacks: list[tuple[float, ...]] = []
        self.committed_street: list[tuple[float, ...]] = []
        self.street_commitments = self.committed_street
        self.committed_total: list[tuple[float, ...]] = []
        self.committed = self.committed_total
        self.status: list[tuple[int, ...]] = []
        self.actor: list[int] = []
        self.to_act = self.actor
        self.last_full_raise: list[float] = []
        self.last_raise = self.last_full_raise
        self.current_bet: list[float] = []
        self.acted: list[tuple[bool, ...]] = []
        self.reopen: list[tuple[bool, ...]] = []
        self.button: list[int] = []
        self.pot: list[float] = []
        self.hand_over: list[bool] = []
        self.betting_closed: list[bool] = []
        self.phase: list[int] = []
        self.n_raises: list[int] = []
        self.action_history: list[tuple[int, ...]] = []
        self.fold_payoff: list[tuple[float, ...] | None] = []
        self.public_history: list[tuple[PublicHistoryEvent, ...]] = []

    def append(self, state: _State) -> int:
        ref = len(self.street)
        self.street.append(state.street)
        self.board.append(state.board + (-1,) * (5 - len(state.board)))
        self.stacks.append(state.stacks)
        self.committed_street.append(state.committed_street)
        self.committed_total.append(state.committed_total)
        self.status.append(state.status)
        self.actor.append(state.actor)
        self.last_full_raise.append(state.last_full_raise)
        self.current_bet.append(state.current_bet)
        self.acted.append(state.acted)
        self.reopen.append(
            tuple(not flag and value == LIVE for flag, value in zip(state.acted, state.status))
        )
        self.button.append(state.button)
        self.pot.append(float(sum(state.committed_total)))
        self.hand_over.append(state.phase == _TERMINAL)
        self.betting_closed.append(state.betting_closed)
        self.phase.append(state.phase)
        self.n_raises.append(state.n_raises)
        self.action_history.append(state.action_history)
        self.fold_payoff.append(state.fold_payoff)
        self.public_history.append(state.public_history)
        return ref

    def state(self, ref: int) -> _State:
        board = tuple(card for card in self.board[ref] if card >= 0)
        return _State(
            street=self.street[ref],
            board=board,
            stacks=self.stacks[ref],
            committed_street=self.committed_street[ref],
            committed_total=self.committed_total[ref],
            status=self.status[ref],
            actor=self.actor[ref],
            last_full_raise=self.last_full_raise[ref],
            current_bet=self.current_bet[ref],
            acted=self.acted[ref],
            button=self.button[ref],
            phase=self.phase[ref],
            betting_closed=self.betting_closed[ref],
            n_raises=self.n_raises[ref],
            action_history=self.action_history[ref],
            fold_payoff=self.fold_payoff[ref],
            public_history=self.public_history[ref],
        )

    def __len__(self) -> int:
        return len(self.street)


@dataclass
class PotStrata:
    """Public affine coefficients supplied to evaluator showdown methods."""

    stratum_players: Tensor  # bool [B,K,P]
    stratum_amount: Tensor  # float32 [B,K]
    payoff_baseline: Tensor  # float32 [B,P]


def assemble_showdown_payoff(strata: PotStrata, showdown_share: Tensor) -> Tensor:
    """Assemble chip payoffs without inspecting cards or comparing hands."""

    if showdown_share.dtype != torch.float32 or showdown_share.ndim != 3:
        raise TypeError("showdown_share must be float32 [B,K,P]")
    if showdown_share.shape != strata.stratum_players.shape:
        raise ValueError("showdown_share shape must match stratum_players")
    if strata.stratum_amount.shape != showdown_share.shape[:2]:
        raise ValueError("stratum_amount must have shape [B,K]")
    if strata.payoff_baseline.shape != (
        showdown_share.shape[0],
        showdown_share.shape[2],
    ):
        raise ValueError("payoff_baseline must have shape [B,P]")
    tensors = (
        strata.stratum_players,
        strata.stratum_amount,
        strata.payoff_baseline,
        showdown_share,
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all payoff tensors must be on one device")
    if bool((showdown_share < -1e-6).any()):
        raise ValueError("showdown shares must be nonnegative")
    if bool((showdown_share.masked_select(~strata.stratum_players).abs() > 1e-6).any()):
        raise ValueError("ineligible players must have zero showdown share")
    positive = strata.stratum_amount > 1e-6
    totals = showdown_share.sum(dim=-1)
    if bool((torch.abs(totals[positive] - 1.0) > 1e-5).any()):
        raise ValueError("shares in each positive stratum must sum to one")
    payoff = strata.payoff_baseline + (
        strata.stratum_amount[:, :, None] * showdown_share
    ).sum(dim=1)
    if bool((payoff.sum(dim=-1).abs() > 2e-4).any()):
        raise ValueError("assembled chip payoffs must be conserved")
    return payoff


class NLHEEngine:
    """Append-only public no-limit hold'em state machine."""

    def __init__(self, cfg: PokerConfig):
        self.cfg = cfg
        self.n_players = cfg.n_players
        self.n_ticks = cfg.n_ticks
        self.n_player_events = RAISE_OFFSET + cfg.n_ticks
        self.e_max = self.n_player_events
        self.states = StateTable(cfg.n_players)
        self.state_table = self.states
        self._roots: dict[int, int] = {}
        self._transition_cache: dict[tuple[int, int], int] = {}
        self._root_ref = self.new_hand(0)

    def _blind_seats(self, button: int) -> tuple[int, int]:
        if self.n_players == 2:
            return button, (button + 1) % 2
        return (button + 1) % self.n_players, (button + 2) % self.n_players

    def _first_preflop(self, button: int) -> int:
        _, bb_seat = self._blind_seats(button)
        return (bb_seat + 1) % self.n_players

    def _next_live_from(self, start: int, status: tuple[int, ...]) -> int:
        for offset in range(self.n_players):
            seat = (start + offset) % self.n_players
            if status[seat] == LIVE:
                return seat
        return -1

    def new_hand(self, button: int = 0) -> int:
        """Post blinds and append a fresh root with the requested button seat."""

        if isinstance(button, bool) or not isinstance(button, int):
            raise TypeError("button must be an integer seat")
        button %= self.n_players
        if button in self._roots:
            return self._roots[button]
        stacks = [float(self.cfg.stack_bb)] * self.n_players
        street_commit = [0.0] * self.n_players
        total_commit = [0.0] * self.n_players
        status = [LIVE] * self.n_players
        sb_seat, bb_seat = self._blind_seats(button)
        for seat, blind in ((sb_seat, self.cfg.sb), (bb_seat, self.cfg.bb)):
            posted = min(stacks[seat], float(blind))
            stacks[seat] -= posted
            street_commit[seat] += posted
            total_commit[seat] += posted
            if stacks[seat] <= _EPS:
                stacks[seat] = 0.0
                status[seat] = ALL_IN
        status_tuple = tuple(status)
        actor = self._next_live_from(self._first_preflop(button), status_tuple)
        phase = _PLAYER if actor >= 0 else _PRETERM
        state = _State(
            street=PREFLOP,
            board=(),
            stacks=tuple(stacks),
            committed_street=tuple(street_commit),
            committed_total=tuple(total_commit),
            status=status_tuple,
            actor=actor,
            last_full_raise=float(self.cfg.bb),
            current_bet=max(street_commit),
            acted=(False,) * self.n_players,
            button=button,
            phase=phase,
            betting_closed=phase != _PLAYER,
            n_raises=0,
            action_history=(),
        )
        ref = self.states.append(state)
        self._roots[button] = ref
        return ref

    def next_hand(self, state_ref: int) -> int:
        """Return a fresh-hand root after rotating the button one seat."""

        self._validate_ref(state_ref)
        return self.new_hand((self.states.button[state_ref] + 1) % self.n_players)

    def rotate_button(self, state_refs: Tensor) -> Tensor:
        refs, device = self._refs(state_refs)
        return torch.tensor([self.next_hand(ref) for ref in refs], dtype=torch.int32, device=device)

    def initial_states(self, g: int, button: int = 0) -> Tensor:
        if g < 0:
            raise ValueError("g must be nonnegative")
        return torch.full((g,), self.new_hand(button), dtype=torch.int32)

    def _validate_ref(self, ref: int) -> None:
        if not 0 <= ref < len(self.states):
            raise ValueError(f"invalid state reference {ref}")

    def _refs(self, state_refs: Tensor) -> tuple[list[int], torch.device]:
        if state_refs.ndim != 1:
            raise ValueError("state_refs must have shape [B]")
        refs = [int(value) for value in state_refs.detach().cpu().tolist()]
        for ref in refs:
            self._validate_ref(ref)
        return refs, state_refs.device

    def owner_of(self, state_refs: Tensor) -> Tensor:
        refs, device = self._refs(state_refs)
        owners: list[int] = []
        for ref in refs:
            phase = self.states.phase[ref]
            if phase == _CHANCE:
                owners.append(CHANCE)
            elif phase == _TERMINAL:
                owners.append(TERMINAL)
            elif phase == _PRETERM:
                owners.append(PRETERM)
            else:
                owners.append(self.states.actor[ref])
        return torch.tensor(owners, dtype=torch.int32, device=device)

    def is_terminal(self, state_refs: Tensor) -> Tensor:
        refs, device = self._refs(state_refs)
        return torch.tensor(
            [self.states.phase[ref] == _TERMINAL for ref in refs],
            dtype=torch.bool,
            device=device,
        )

    def is_pre_terminal(self, state_refs: Tensor) -> Tensor:
        refs, device = self._refs(state_refs)
        return torch.tensor(
            [self.states.phase[ref] == _PRETERM for ref in refs],
            dtype=torch.bool,
            device=device,
        )

    def _raise_grid(self, state: _State) -> tuple[list[float], list[bool]]:
        actor = state.actor
        unit = self.cfg.chip_unit_bb
        targets = [unit * (tick + 1) for tick in range(self.n_ticks)]
        legal = [False] * self.n_ticks
        if actor < 0 or state.phase != _PLAYER or state.acted[actor]:
            return targets, legal
        maximum = state.committed_street[actor] + state.stacks[actor]
        if maximum <= state.current_bet + _EPS:
            return targets, legal
        minimum = state.current_bet + state.last_full_raise
        maximum_tick = round(maximum / unit) - 1
        if maximum + _EPS < minimum:
            legal[maximum_tick] = True
            return targets, legal
        minimum_tick = max(0, math.ceil((minimum - _EPS) / unit) - 1)
        for tick in range(minimum_tick, maximum_tick + 1):
            legal[tick] = True
        return targets, legal

    def raise_targets(self, state_refs: Tensor) -> Tensor:
        """Return every fixed-denomination raise-to amount ``[B,n_ticks]``."""

        refs, device = self._refs(state_refs)
        rows = [self._raise_grid(self.states.state(ref))[0] for ref in refs]
        return torch.tensor(rows, dtype=torch.float32, device=device)

    legal_raise_targets = raise_targets

    def legal_events(self, state_refs: Tensor) -> tuple[Tensor, Tensor]:
        refs, device = self._refs(state_refs)
        batch = len(refs)
        ids = torch.arange(self.e_max, dtype=torch.int32, device=device).expand(batch, -1).clone()
        mask = torch.zeros((batch, self.e_max), dtype=torch.bool, device=device)
        for row, ref in enumerate(refs):
            state = self.states.state(ref)
            if state.phase != _PLAYER:
                continue
            actor = state.actor
            to_call = max(0.0, state.current_bet - state.committed_street[actor])
            mask[row, FOLD] = to_call > _EPS
            mask[row, CHECK_CALL] = True
            _, raises = self._raise_grid(state)
            mask[row, RAISE_OFFSET:] = torch.tensor(raises, dtype=torch.bool, device=device)
        return ids, mask

    def _still_contesting(self, status: tuple[int, ...]) -> list[int]:
        return [seat for seat, value in enumerate(status) if value != FOLDED]

    def _needs_action(self, state: _State, seat: int) -> bool:
        return state.status[seat] == LIVE and (
            not state.acted[seat]
            or state.committed_street[seat] + _EPS < state.current_bet
        )

    def _find_next_actor(self, state: _State, after: int) -> int:
        for offset in range(1, self.n_players + 1):
            seat = (after + offset) % self.n_players
            if self._needs_action(state, seat):
                return seat
        return -1

    def _finish_fold(self, state: _State) -> _State:
        winner = self._still_contesting(state.status)[0]
        payoff = [-amount for amount in state.committed_total]
        payoff[winner] += sum(state.committed_total)
        return _State(
            **{
                **state.__dict__,
                "actor": -1,
                "phase": _TERMINAL,
                "betting_closed": True,
                "fold_payoff": tuple(payoff),
            }
        )

    def _finish_round(self, state: _State) -> _State:
        actionable = [seat for seat, value in enumerate(state.status) if value == LIVE]
        if state.street == RIVER or len(actionable) <= 1:
            return _State(
                **{
                    **state.__dict__,
                    "actor": -1,
                    "phase": _PRETERM,
                    "betting_closed": True,
                }
            )
        return _State(
            **{
                **state.__dict__,
                "actor": -1,
                "phase": _CHANCE,
                "betting_closed": True,
            }
        )

    def _apply(self, state: _State, event: int) -> _State:
        actor = state.actor
        stacks = list(state.stacks)
        street_commit = list(state.committed_street)
        total_commit = list(state.committed_total)
        status = list(state.status)
        acted = list(state.acted)
        current_bet = state.current_bet
        last_full_raise = state.last_full_raise
        n_raises = state.n_raises
        targets, _ = self._raise_grid(state)
        call_amount = max(0.0, current_bet - street_commit[actor])
        maximum = street_commit[actor] + stacks[actor]
        minimum = min(maximum, current_bet + last_full_raise)
        action = 0 if event == FOLD else 1 if event == CHECK_CALL else 2
        amount = (
            min(stacks[actor], call_amount)
            if event == CHECK_CALL
            else targets[event - RAISE_OFFSET]
            if event >= RAISE_OFFSET
            else 0.0
        )
        history_event = PublicHistoryEvent(
            actor=actor,
            street=state.street,
            action=action,
            amount=float(amount),
            pot_before=float(sum(state.committed_total)),
            call_amount=float(call_amount),
            current_bet=float(current_bet),
            min_raise_to=float(minimum),
            max_raise_to=float(maximum),
        )

        if event == FOLD:
            status[actor] = FOLDED
            acted[actor] = True
        elif event == CHECK_CALL:
            call = min(stacks[actor], max(0.0, current_bet - street_commit[actor]))
            stacks[actor] -= call
            street_commit[actor] += call
            total_commit[actor] += call
            acted[actor] = True
            if stacks[actor] <= _EPS:
                stacks[actor] = 0.0
                status[actor] = ALL_IN
        else:
            target = targets[event - RAISE_OFFSET]
            addition = target - street_commit[actor]
            old_bet = current_bet
            stacks[actor] -= addition
            street_commit[actor] = target
            total_commit[actor] += addition
            current_bet = target
            increment = target - old_bet
            is_full = increment + _EPS >= last_full_raise
            if is_full:
                last_full_raise = increment
                acted = [False] * self.n_players
                n_raises += 1
            acted[actor] = True
            if stacks[actor] <= _EPS:
                stacks[actor] = 0.0
                status[actor] = ALL_IN

        updated = _State(
            street=state.street,
            board=state.board,
            stacks=tuple(stacks),
            committed_street=tuple(street_commit),
            committed_total=tuple(total_commit),
            status=tuple(status),
            actor=actor,
            last_full_raise=last_full_raise,
            current_bet=current_bet,
            acted=tuple(acted),
            button=state.button,
            phase=_PLAYER,
            betting_closed=False,
            n_raises=n_raises,
            action_history=state.action_history + (event,),
            public_history=state.public_history + (history_event,),
        )
        if len(self._still_contesting(updated.status)) == 1:
            return self._finish_fold(updated)
        next_actor = self._find_next_actor(updated, actor)
        if next_actor < 0:
            return self._finish_round(updated)
        return _State(**{**updated.__dict__, "actor": next_actor})

    def step(self, state_refs: Tensor, event_ids: Tensor) -> Tensor:
        """Apply legal player events and return int32 successor refs ``[B]``."""

        refs, device = self._refs(state_refs)
        if event_ids.ndim != 1 or event_ids.shape != state_refs.shape:
            raise ValueError("event_ids must have shape [B]")
        events = [int(value) for value in event_ids.detach().cpu().tolist()]
        _, legal = self.legal_events(state_refs)
        successors: list[int] = []
        for row, (ref, event) in enumerate(zip(refs, events, strict=True)):
            if not 0 <= event < self.e_max or not bool(legal[row, event]):
                raise ValueError(f"illegal event {event} at state {ref}")
            key = (ref, event)
            successor = self._transition_cache.get(key)
            if successor is None:
                successor = self.states.append(self._apply(self.states.state(ref), event))
                self._transition_cache[key] = successor
            successors.append(successor)
        return torch.tensor(successors, dtype=torch.int32, device=device)

    def deal(self, state_refs: Tensor, cards: Tensor) -> Tensor:
        """Apply supplied public chance cards and begin the next betting street.

        A flop row contains three cards and is canonicalized by sorting.  Turn
        and river rows contain one card.  ``-1`` suffix padding is accepted.
        """

        refs, device = self._refs(state_refs)
        if cards.ndim == 1:
            cards = cards[:, None]
        if cards.ndim != 2 or cards.shape[0] != len(refs):
            raise ValueError("cards must have shape [B,C]")
        if cards.device != device:
            raise ValueError("cards and state_refs must be on one device")
        if cards.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise TypeError("cards must have an integer dtype")
        card_rows = cards.detach().cpu().tolist()
        successors: list[int] = []
        for ref, padded in zip(refs, card_rows, strict=True):
            state = self.states.state(ref)
            if state.phase != _CHANCE:
                raise ValueError("deal is only valid at a public chance node")
            supplied = [int(card) for card in padded if int(card) != -1]
            expected = 3 if state.street == PREFLOP else 1
            if len(supplied) != expected:
                raise ValueError(f"street requires exactly {expected} public card(s)")
            if any(
                card < self.cfg.geometry.first_card or card >= 52 for card in supplied
            ):
                raise ValueError("public card ids must belong to the active deck")
            if len(set(supplied)) != len(supplied) or set(supplied) & set(state.board):
                raise ValueError("public cards must be unique")
            if expected == 3:
                supplied.sort()
            board = state.board + tuple(supplied)
            street = state.street + 1
            acted = (False,) * self.n_players
            street_commit = (0.0,) * self.n_players
            live_status = state.status
            actor = self._next_live_from((state.button + 1) % self.n_players, live_status)
            if actor < 0:
                raise AssertionError("chance node has no player able to act")
            next_state = _State(
                street=street,
                board=board,
                stacks=state.stacks,
                committed_street=street_commit,
                committed_total=state.committed_total,
                status=live_status,
                actor=actor,
                last_full_raise=float(self.cfg.bb),
                current_bet=0.0,
                acted=acted,
                button=state.button,
                phase=_PLAYER,
                betting_closed=False,
                n_raises=0,
                action_history=state.action_history,
                public_history=state.public_history + (
                    PublicHistoryEvent(
                        actor=-1,
                        street=street,
                        action=3,
                        amount=0.0,
                        pot_before=float(sum(state.committed_total)),
                        call_amount=0.0,
                        current_bet=0.0,
                        min_raise_to=0.0,
                        max_raise_to=0.0,
                    ),
                ),
            )
            successors.append(self.states.append(next_state))
        return torch.tensor(successors, dtype=torch.int32, device=device)

    def reward(self, xprofile: Tensor, terminal_state_refs: Tensor) -> Tensor:
        """Return locally known fold-win payoffs for terminal states."""

        del xprofile
        refs, device = self._refs(terminal_state_refs)
        rows: list[tuple[float, ...]] = []
        for ref in refs:
            payoff = self.states.fold_payoff[ref]
            if self.states.phase[ref] != _TERMINAL or payoff is None:
                raise ValueError("reward is only local for fold-win terminal states")
            rows.append(payoff)
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def pot_strata(self, state_refs: Tensor) -> PotStrata:
        """Construct padded nested side-pot strata from public contributions."""

        refs, device = self._refs(state_refs)
        all_rows: list[list[tuple[float, tuple[bool, ...]]]] = []
        max_strata = 0
        for ref in refs:
            state = self.states.state(ref)
            levels = sorted({amount for amount in state.committed_total if amount > _EPS})
            previous = 0.0
            row: list[tuple[float, tuple[bool, ...]]] = []
            for level in levels:
                contributors = sum(amount + _EPS >= level for amount in state.committed_total)
                amount = (level - previous) * contributors
                eligible = tuple(
                    value != FOLDED and contribution + _EPS >= level
                    for value, contribution in zip(state.status, state.committed_total)
                )
                if amount > _EPS:
                    row.append((amount, eligible))
                previous = level
            all_rows.append(row)
            max_strata = max(max_strata, len(row))
        players = torch.zeros(
            (len(refs), max_strata, self.n_players), dtype=torch.bool, device=device
        )
        amounts = torch.zeros((len(refs), max_strata), dtype=torch.float32, device=device)
        baseline = torch.empty((len(refs), self.n_players), dtype=torch.float32, device=device)
        for batch, (ref, row) in enumerate(zip(refs, all_rows, strict=True)):
            baseline[batch] = -torch.tensor(
                self.states.committed_total[ref], dtype=torch.float32, device=device
            )
            for stratum, (amount, eligible) in enumerate(row):
                amounts[batch, stratum] = amount
                players[batch, stratum] = torch.tensor(eligible, dtype=torch.bool, device=device)
        return PotStrata(players, amounts, baseline)

    def assemble_showdown_payoff(
        self, strata: PotStrata, showdown_share: Tensor
    ) -> Tensor:
        return assemble_showdown_payoff(strata, showdown_share)
