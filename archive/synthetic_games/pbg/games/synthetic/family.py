"""Parametric finite card games for exercising generic PBG algorithms."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch
from pbg.core.observability import ExecutionMetrics
from pbg.core.queued import (
    logical_categorical_from_valid_probability,
)

FOLD = 0
CHECK_CALL = 1
RAISE = 2
N_PLAYER_EVENTS = 3
CHANCE_EVENT_OFFSET = 3


@dataclass(frozen=True)
class SyntheticDialConfig:
    """Configure a finite capped, discrete-raise card game.

    ``board_per_round[i]`` cards are revealed as one chance event immediately
    before betting round ``i``. ``raise_sizes[i]`` supplies the legal raise
    increments in round ``i``; omitted rows use the legacy singleton
    ``bet_size``. Private hands use colexicographic type indices.
    """

    n_players: int
    deck_size: int
    n_suits: int
    hand_size: int
    n_rounds: int
    board_per_round: tuple[int, ...]
    max_raises: int
    bet_size: tuple[int, ...]
    raise_sizes: tuple[tuple[int, ...], ...] | None = None
    ante: int = 1

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce a complete deal."""

        object.__setattr__(self, "board_per_round", tuple(self.board_per_round))
        object.__setattr__(self, "bet_size", tuple(self.bet_size))
        raise_sizes = self.raise_sizes
        if raise_sizes is None:
            raise_sizes = tuple((size,) for size in self.bet_size)
        else:
            raise_sizes = tuple(tuple(round_sizes) for round_sizes in raise_sizes)
        object.__setattr__(self, "raise_sizes", raise_sizes)
        if self.n_players < 2:
            raise ValueError("n_players must be at least 2")
        if self.deck_size <= 0 or self.n_suits <= 0:
            raise ValueError("deck_size and n_suits must be positive")
        if self.deck_size % self.n_suits:
            raise ValueError("deck_size must be divisible by n_suits")
        if not 1 <= self.hand_size <= self.deck_size:
            raise ValueError("hand_size must be in [1, deck_size]")
        if self.n_rounds <= 0:
            raise ValueError("n_rounds must be positive")
        if len(self.board_per_round) != self.n_rounds:
            raise ValueError("board_per_round must have one entry per round")
        if self.board_per_round[0] != 0 or any(n < 0 for n in self.board_per_round):
            raise ValueError("board_per_round must start with zero and be nonnegative")
        if len(self.bet_size) != self.n_rounds or any(size <= 0 for size in self.bet_size):
            raise ValueError("bet_size must contain one positive size per round")
        if len(raise_sizes) != self.n_rounds or any(not sizes for sizes in raise_sizes):
            raise ValueError("raise_sizes must contain one nonempty row per round")
        if any(
            not all(size > 0 for size in sizes)
            or tuple(sorted(set(sizes))) != sizes
            for sizes in raise_sizes
        ):
            raise ValueError("each raise_sizes row must be positive, unique, and increasing")
        if tuple(sizes[0] for sizes in raise_sizes) != self.bet_size:
            raise ValueError("bet_size must equal the first raise size in each round")
        if self.max_raises < 0 or self.ante < 0:
            raise ValueError("max_raises and ante must be nonnegative")
        cards_needed = self.n_players * self.hand_size + sum(self.board_per_round)
        if cards_needed > self.deck_size:
            raise ValueError("deck is too small for all private and public cards")


@dataclass(frozen=True)
class _State:
    round: int
    raises_this_round: int
    to_act: int
    committed: tuple[int, ...]
    live: tuple[bool, ...]
    board: tuple[int, ...]
    round_committed: tuple[int, ...]
    acted: tuple[bool, ...]
    phase: int  # 0 decision, 1 chance, 2 terminal


class StateTable:
    """Structure-of-arrays storage addressed by int32 state references."""

    def __init__(self, n_players: int, board_capacity: int):
        """Create an empty table for ``P``-seat states and padded public boards."""

        self.n_players = n_players
        self.board_capacity = board_capacity
        self.round: list[int] = []
        self.raises_this_round: list[int] = []
        self.to_act: list[int] = []
        self.committed: list[tuple[int, ...]] = []
        self.live: list[tuple[bool, ...]] = []
        self.board: list[tuple[int, ...]] = []
        self.round_committed: list[tuple[int, ...]] = []
        self.acted: list[tuple[bool, ...]] = []
        self.phase: list[int] = []
        self.history_hash: list[int] = []

    def append(self, state: _State) -> int:
        """Append one state and return its integer reference."""

        ref = len(self.round)
        self.round.append(state.round)
        self.raises_this_round.append(state.raises_this_round)
        self.to_act.append(state.to_act)
        self.committed.append(state.committed)
        self.live.append(state.live)
        padded = state.board + (-1,) * (self.board_capacity - len(state.board))
        self.board.append(padded)
        self.round_committed.append(state.round_committed)
        self.acted.append(state.acted)
        self.phase.append(state.phase)
        self.history_hash.append(ref)
        return ref

    def state(self, ref: int) -> _State:
        """Reconstruct the immutable state stored at ``ref``."""

        board = tuple(card for card in self.board[ref] if card >= 0)
        return _State(
            round=self.round[ref],
            raises_this_round=self.raises_this_round[ref],
            to_act=self.to_act[ref],
            committed=self.committed[ref],
            live=self.live[ref],
            board=board,
            round_committed=self.round_committed[ref],
            acted=self.acted[ref],
            phase=self.phase[ref],
        )

    def __len__(self) -> int:
        return len(self.round)


def colex_index(cards: Iterable[int]) -> int:
    """Return the colexicographic rank of a sorted distinct card set."""

    ordered = tuple(sorted(cards))
    if len(set(ordered)) != len(ordered):
        raise ValueError("colex card sets must contain distinct cards")
    return sum(math.comb(card, index + 1) for index, card in enumerate(ordered))


class SyntheticPBG:
    """Finite public-action Bayesian card game implementing ``PBGGame``."""

    def __init__(self, cfg: SyntheticDialConfig):
        """Build private-type maps and the canonical public root state."""

        self.cfg = cfg
        self.n_players = cfg.n_players
        self.n_types = math.comb(cfg.deck_size, cfg.hand_size)
        self.x_sizes = (self.n_types,) * cfg.n_players
        hands = list(itertools.combinations(range(cfg.deck_size), cfg.hand_size))
        hands.sort(key=colex_index)
        self.hands: tuple[tuple[int, ...], ...] = tuple(hands)
        if tuple(colex_index(hand) for hand in self.hands) != tuple(range(self.n_types)):
            raise AssertionError("colex hand map is not contiguous")
        self._hand_to_type = {hand: index for index, hand in enumerate(self.hands)}
        self._hand_sets = tuple(frozenset(hand) for hand in self.hands)

        self.n_player_events = 2 + max(len(sizes) for sizes in cfg.raise_sizes)
        self.chance_event_offset = self.n_player_events

        chance_sizes = {count for count in cfg.board_per_round if count > 0}
        self._chance_combos: dict[int, tuple[tuple[int, ...], ...]] = {}
        self._chance_event_to_cards: dict[int, dict[int, tuple[int, ...]]] = {}
        for count in chance_sizes:
            combos = list(itertools.combinations(range(cfg.deck_size), count))
            combos.sort(key=colex_index)
            self._chance_combos[count] = tuple(combos)
            self._chance_event_to_cards[count] = {
                self.chance_event_offset + colex_index(combo): combo for combo in combos
            }
        self.e_max = max(
            [self.n_player_events]
            + [math.comb(cfg.deck_size, count) for count in chance_sizes]
        )

        self.states = StateTable(cfg.n_players, sum(cfg.board_per_round))
        self.state_table = self.states
        root = _State(
            round=0,
            raises_this_round=0,
            to_act=0,
            committed=(cfg.ante,) * cfg.n_players,
            live=(True,) * cfg.n_players,
            board=(),
            round_committed=(0,) * cfg.n_players,
            acted=(False,) * cfg.n_players,
            phase=0,
        )
        self._root_ref = self.states.append(root)
        self._transition_cache: dict[tuple[int, int], int] = {}

    def hand_of_type(self, xtype: int) -> tuple[int, ...]:
        """Return the sorted private cards represented by ``xtype``."""

        return self.hands[xtype]

    def type_of_hand(self, cards: Iterable[int]) -> int:
        """Return the colex type index for a private card set."""

        return self._hand_to_type[tuple(sorted(cards))]

    def sample_mu(
        self, g: int, gen: torch.Generator | np.random.Generator
    ) -> Tensor:
        """Sample collision-free private profiles as int32 ``[G, P]``."""

        if g < 0:
            raise ValueError("g must be nonnegative")
        result = torch.empty((g, self.n_players), dtype=torch.int32)  # [G, P]
        needed = self.n_players * self.cfg.hand_size
        for row in range(g):
            if isinstance(gen, torch.Generator):
                dealt = torch.randperm(self.cfg.deck_size, generator=gen)[:needed].tolist()
            else:
                dealt = gen.permutation(self.cfg.deck_size)[:needed].tolist()
            for player in range(self.n_players):
                start = player * self.cfg.hand_size
                hand = tuple(sorted(dealt[start : start + self.cfg.hand_size]))
                result[row, player] = self._hand_to_type[hand]
        return result

    def root_states(self, g: int) -> Tensor:
        """Return int32 references to ``G`` copies of the canonical public root ``[G]``."""

        if g < 0:
            raise ValueError("g must be nonnegative")
        return torch.full((g,), self._root_ref, dtype=torch.int32)  # [G]

    @staticmethod
    def _refs(state_refs: Tensor) -> tuple[list[int], torch.device]:
        if state_refs.ndim != 1:
            raise ValueError("state_refs must have shape [B]")
        return [int(ref) for ref in state_refs.detach().cpu().tolist()], state_refs.device

    def owner_of(self, state_refs: Tensor) -> Tensor:
        """Return owner codes as int32 ``[B]``."""

        refs, device = self._refs(state_refs)
        owners: list[int] = []
        for ref in refs:
            phase = self.states.phase[ref]
            if phase == 2:
                owners.append(TERMINAL)
            elif phase == 1:
                owners.append(CHANCE)
            else:
                owners.append(self.states.to_act[ref])
        return torch.tensor(owners, dtype=torch.int32, device=device)  # [B]

    def _legal_for_ref(self, ref: int) -> list[int]:
        phase = self.states.phase[ref]
        if phase == 2:
            return []
        if phase == 1:
            count = self.cfg.board_per_round[self.states.round[ref]]
            board = {card for card in self.states.board[ref] if card >= 0}
            return [
                self.chance_event_offset + colex_index(combo)
                for combo in self._chance_combos[count]
                if board.isdisjoint(combo)
            ]
        actor = self.states.to_act[ref]
        current_bet = max(self.states.round_committed[ref])
        facing = self.states.round_committed[ref][actor] < current_bet
        events = [CHECK_CALL]
        if facing:
            events.insert(0, FOLD)
        if self.states.raises_this_round[ref] < self.cfg.max_raises:
            events.extend(
                RAISE + index
                for index in range(len(self.cfg.raise_sizes[self.states.round[ref]]))
            )
        return events

    def legal_events(self, state_refs: Tensor) -> tuple[Tensor, Tensor]:
        """Return padded event ids and validity masks, each ``[B, E_max]``."""

        refs, device = self._refs(state_refs)
        event_ids = torch.zeros((len(refs), self.e_max), dtype=torch.int32, device=device)
        mask = torch.zeros((len(refs), self.e_max), dtype=torch.bool, device=device)
        for row, ref in enumerate(refs):
            events = self._legal_for_ref(ref)
            if events:
                event_ids[row, : len(events)] = torch.tensor(
                    events, dtype=torch.int32, device=device
                )
                mask[row, : len(events)] = True
        return event_ids, mask

    def chance_probs(self, state_refs: Tensor, xprofile: Tensor) -> Tensor:
        """Return collision-conditioned chance probabilities ``[B, E_max]``."""

        refs, device = self._refs(state_refs)
        if xprofile.shape != (len(refs), self.n_players):
            raise ValueError("xprofile must have shape [B, P]")
        profiles = xprofile.detach().cpu().to(torch.int64).tolist()
        probs = torch.zeros((len(refs), self.e_max), dtype=torch.float32, device=device)
        for row, (ref, profile) in enumerate(zip(refs, profiles, strict=True)):
            if self.states.phase[ref] != 1:
                continue
            unavailable = {card for card in self.states.board[ref] if card >= 0}
            for xtype in profile:
                unavailable.update(self.hands[xtype])
            valid_columns: list[int] = []
            for column, event in enumerate(self._legal_for_ref(ref)):
                count = self.cfg.board_per_round[self.states.round[ref]]
                cards = self._chance_event_to_cards[count][event]
                if unavailable.isdisjoint(cards):
                    valid_columns.append(column)
            if not valid_columns:
                raise RuntimeError("chance state has no outcome compatible with private profile")
            probs[row, valid_columns] = 1.0 / len(valid_columns)
        return probs

    def _next_live(self, live: tuple[bool, ...], actor: int) -> int:
        for offset in range(1, self.n_players + 1):
            candidate = (actor + offset) % self.n_players
            if live[candidate]:
                return candidate
        raise RuntimeError("state has no live player")

    @staticmethod
    def _round_closed(
        live: tuple[bool, ...], acted: tuple[bool, ...], contributions: tuple[int, ...]
    ) -> bool:
        current_bet = max(contributions)
        return all(not live[p] or (acted[p] and contributions[p] == current_bet) for p in range(len(live)))

    def _after_round(self, state: _State) -> _State:
        next_round = state.round + 1
        if next_round >= self.cfg.n_rounds:
            return _State(
                round=state.round,
                raises_this_round=state.raises_this_round,
                to_act=-1,
                committed=state.committed,
                live=state.live,
                board=state.board,
                round_committed=state.round_committed,
                acted=state.acted,
                phase=2,
            )
        phase = 1 if self.cfg.board_per_round[next_round] else 0
        first_live = next(p for p, live in enumerate(state.live) if live)
        return _State(
            round=next_round,
            raises_this_round=0,
            to_act=-1 if phase == 1 else first_live,
            committed=state.committed,
            live=state.live,
            board=state.board,
            round_committed=(0,) * self.n_players,
            acted=(False,) * self.n_players,
            phase=phase,
        )

    def _apply_player_event(self, state: _State, event: int) -> _State:
        actor = state.to_act
        live = list(state.live)
        acted = list(state.acted)
        contributions = list(state.round_committed)
        committed = list(state.committed)
        current_bet = max(contributions)
        if event == FOLD:
            live[actor] = False
            acted[actor] = True
        elif event == CHECK_CALL:
            payment = current_bet - contributions[actor]
            contributions[actor] = current_bet
            committed[actor] += payment
            acted[actor] = True
        elif RAISE <= event < RAISE + len(self.cfg.raise_sizes[state.round]):
            target = current_bet + self.cfg.raise_sizes[state.round][event - RAISE]
            payment = target - contributions[actor]
            contributions[actor] = target
            committed[actor] += payment
            acted = [False] * self.n_players
            acted[actor] = True
        else:
            raise ValueError(f"unknown player event {event}")

        live_tuple = tuple(live)
        next_state = _State(
            round=state.round,
            raises_this_round=state.raises_this_round + int(event >= RAISE),
            to_act=self._next_live(live_tuple, actor) if sum(live_tuple) > 1 else -1,
            committed=tuple(committed),
            live=live_tuple,
            board=state.board,
            round_committed=tuple(contributions),
            acted=tuple(acted),
            phase=0,
        )
        if sum(live_tuple) == 1:
            return _State(**{**next_state.__dict__, "phase": 2})
        if self._round_closed(live_tuple, tuple(acted), tuple(contributions)):
            return self._after_round(next_state)
        return next_state

    def _apply_chance_event(self, state: _State, event: int) -> _State:
        count = self.cfg.board_per_round[state.round]
        cards = self._chance_event_to_cards[count].get(event)
        if cards is None:
            raise ValueError(f"unknown chance event {event}")
        if not set(state.board).isdisjoint(cards):
            raise ValueError("chance event repeats a public card")
        first_live = next(p for p, live in enumerate(state.live) if live)
        return _State(
            round=state.round,
            raises_this_round=0,
            to_act=first_live,
            committed=state.committed,
            live=state.live,
            board=tuple(sorted((*state.board, *cards))),
            round_committed=(0,) * self.n_players,
            acted=(False,) * self.n_players,
            phase=0,
        )

    def step(self, state_refs: Tensor, event_ids: Tensor) -> Tensor:
        """Apply one legal event per row and return int32 successors ``[B]``."""

        refs, device = self._refs(state_refs)
        if event_ids.ndim != 1 or event_ids.shape[0] != len(refs):
            raise ValueError("event_ids must have shape [B]")
        events = [int(event) for event in event_ids.detach().cpu().tolist()]
        successors: list[int] = []
        for ref, event in zip(refs, events, strict=True):
            if event not in self._legal_for_ref(ref):
                raise ValueError(f"event {event} is illegal at state {ref}")
            cache_key = (ref, event)
            successor = self._transition_cache.get(cache_key)
            if successor is None:
                state = self.states.state(ref)
                new_state = (
                    self._apply_chance_event(state, event)
                    if state.phase == 1
                    else self._apply_player_event(state, event)
                )
                successor = self.states.append(new_state)
                self._transition_cache[cache_key] = successor
            successors.append(successor)
        return torch.tensor(successors, dtype=torch.int32, device=device)  # [B]

    def is_terminal(self, state_refs: Tensor) -> Tensor:
        """Return terminal flags ``[B]``."""

        refs, device = self._refs(state_refs)
        return torch.tensor(
            [self.states.phase[ref] == 2 for ref in refs], dtype=torch.bool, device=device
        )  # [B]

    def is_pre_terminal(self, state_refs: Tensor) -> Tensor:
        """Return flags for the final public chance prefix ``[B]``."""

        refs, device = self._refs(state_refs)
        flags = [
            self.states.phase[ref] == 1
            and self.states.round[ref] == self.cfg.n_rounds - 1
            for ref in refs
        ]
        return torch.tensor(flags, dtype=torch.bool, device=device)  # [B]

    def _strength(self, xtype: int, board: tuple[int, ...]) -> tuple[int, int]:
        ranks = tuple(card // self.cfg.n_suits for card in self.hands[xtype])
        board_ranks = {card // self.cfg.n_suits for card in board}
        paired = int(any(rank in board_ranks for rank in ranks))
        return paired, max(ranks)

    def reward(self, xprofile: Tensor, terminal_state_refs: Tensor) -> Tensor:
        """Return zero-sum terminal chip rewards as float32 ``[B, P]``."""

        refs, device = self._refs(terminal_state_refs)
        if xprofile.shape != (len(refs), self.n_players):
            raise ValueError("xprofile must have shape [B, P]")
        profiles = xprofile.detach().cpu().to(torch.int64).tolist()
        rewards = torch.empty(
            (len(refs), self.n_players), dtype=torch.float32, device=device
        )  # [B, P]
        for row, (ref, profile) in enumerate(zip(refs, profiles, strict=True)):
            state = self.states.state(ref)
            if state.phase != 2:
                raise ValueError("reward requires terminal states")
            payoff = [-float(value) for value in state.committed]
            pot = float(sum(state.committed))
            live_players = [p for p, live in enumerate(state.live) if live]
            if len(live_players) == 1:
                winners = live_players
            else:
                strengths = {p: self._strength(profile[p], state.board) for p in live_players}
                best = max(strengths.values())
                winners = [p for p, strength in strengths.items() if strength == best]
            share = pot / len(winners)
            for winner in winners:
                payoff[winner] += share
            rewards[row] = torch.tensor(payoff, dtype=torch.float32, device=device)
        return rewards

    def _event_cards(self, ref: int, event: int) -> tuple[int, ...]:
        count = self.cfg.board_per_round[self.states.round[ref]]
        return self._chance_event_to_cards[count][event]

    def reach_update(
        self,
        log_r_rows: Tensor,
        state_refs: Tensor,
        event_ids: Tensor,
        policy_rows: Tensor,
    ) -> Tensor:
        """Apply action/chance likelihoods to log reach rows ``[B, P, X]``."""

        squeeze = log_r_rows.ndim == 2
        rows = log_r_rows.unsqueeze(0) if squeeze else log_r_rows
        policies = policy_rows.unsqueeze(0) if policy_rows.ndim == 3 else policy_rows
        refs, _ = self._refs(state_refs)
        if rows.shape != (len(refs), self.n_players, self.n_types):
            raise ValueError("log_r_rows must have shape [B, P, X]")
        if policies.ndim != 4 or policies.shape[:3] != rows.shape:
            raise ValueError("policy_rows must have shape [B, P, X, E]")
        if event_ids.shape != (len(refs),):
            raise ValueError("event_ids must have shape [B]")
        result = rows.clone()  # [B, P, X]
        events = event_ids.detach().cpu().tolist()
        for batch, (ref, event) in enumerate(zip(refs, events, strict=True)):
            phase = self.states.phase[ref]
            if phase == 0:
                actor = self.states.to_act[ref]
                if event >= policies.shape[-1]:
                    raise ValueError("policy_rows does not contain selected event")
                result[batch, actor] += torch.log(policies[batch, actor, :, event])
            elif phase == 1:
                revealed = set(self._event_cards(ref, int(event)))
                conflict = torch.tensor(
                    [not revealed.isdisjoint(hand) for hand in self._hand_sets],
                    dtype=torch.bool,
                    device=result.device,
                )  # [X]
                result[batch, :, conflict] = -torch.inf
        return result.squeeze(0) if squeeze else result

    def _profile_feasible(self, profile: tuple[int, ...], board: set[int]) -> bool:
        used = set(board)
        for xtype in profile:
            hand = self._hand_sets[xtype]
            if not used.isdisjoint(hand):
                return False
            used.update(hand)
        return True

    @staticmethod
    def _normalized_weights(log_weights: Tensor) -> Tensor:
        finite = torch.isfinite(log_weights)
        if not finite.any():
            raise RuntimeError("posterior reach has zero mass")
        shifted = log_weights - log_weights[finite].max()
        weights = torch.where(finite, torch.exp(shifted), torch.zeros_like(shifted))
        return weights / weights.sum()

    def sample_profiles(
        self,
        state_refs: Tensor,
        reach_rows: Tensor,
        n: int,
        gen: torch.Generator | np.random.Generator,
    ) -> Tensor:
        """Draw collision-conditioned posterior profiles as int32 ``[B, n, P]``."""

        if n < 0:
            raise ValueError("n must be nonnegative")
        squeeze = reach_rows.ndim == 2
        rows = reach_rows.unsqueeze(0) if squeeze else reach_rows
        refs, device = self._refs(state_refs)
        if rows.shape != (len(refs), self.n_players, self.n_types):
            raise ValueError("reach_rows must have shape [B, P, X]")
        output = torch.empty(
            (len(refs), n, self.n_players), dtype=torch.int32, device=device
        )  # [B, n, P]
        if n == 0:
            return output
        exact = math.prod(self.x_sizes) <= 2_000_000
        for batch, ref in enumerate(refs):
            board = {card for card in self.states.board[ref] if card >= 0}
            if exact:
                profiles: list[tuple[int, ...]] = []
                log_weights: list[Tensor] = []
                for profile in itertools.product(range(self.n_types), repeat=self.n_players):
                    if self._profile_feasible(profile, board):
                        profiles.append(profile)
                        log_weight = torch.zeros(
                            (), dtype=rows.dtype, device=rows.device
                        )
                        for player, xtype in enumerate(profile):
                            log_weight = log_weight + rows[batch, player, xtype]
                        log_weights.append(log_weight)
                weights = self._normalized_weights(
                    torch.stack(log_weights).to(torch.float64)
                )
                if isinstance(gen, torch.Generator):
                    indices = torch.multinomial(weights, n, replacement=True, generator=gen)
                    profile_table = torch.tensor(
                        profiles, dtype=torch.int32, device=device
                    )
                    output[batch] = profile_table[indices.to(device)]
                else:
                    indices = gen.choice(
                        len(profiles),
                        size=n,
                        replace=True,
                        p=weights.detach().cpu().numpy(),
                    )
                    chosen = [profiles[int(index)] for index in indices]
                    output[batch] = torch.tensor(
                        chosen, dtype=torch.int32, device=device
                    )
                continue

            player_probs = [
                self._normalized_weights(rows[batch, p])
                for p in range(self.n_players)
            ]
            accepted: list[tuple[int, ...]] = []
            attempts = 0
            max_attempts = max(10_000, 10_000 * n)
            while len(accepted) < n and attempts < max_attempts:
                attempts += 1
                if isinstance(gen, torch.Generator):
                    profile = tuple(
                        int(torch.multinomial(prob, 1, generator=gen).item())
                        for prob in player_probs
                    )
                else:
                    profile = tuple(
                        int(
                            gen.choice(
                                self.n_types, p=prob.detach().cpu().numpy()
                            )
                        )
                        for prob in player_probs
                    )
                if self._profile_feasible(profile, board):
                    accepted.append(profile)
            if len(accepted) != n:
                raise RuntimeError("posterior rejection sampler exhausted its attempt budget")
            output[batch] = torch.tensor(accepted, dtype=torch.int32, device=device)
        return output

    def sample_profiles_keyed(
        self,
        state_refs: Tensor,
        reach_rows: Tensor,
        keys: Tensor,
    ) -> Tensor:
        """Draw profiles from schedule-independent keys ``[B,S]``.

        The keyed path enumerates the exact finite joint support, matching the
        logical-address rollout sampler.  Large synthetic supports retain the
        existing explicit feasibility guard instead of silently falling back
        to queue-order mutable RNG.
        """

        if keys.ndim != 2 or keys.dtype != torch.int64:
            raise ValueError("synthetic profile keys must be int64 [B,S]")
        squeeze = reach_rows.ndim == 2
        rows = reach_rows.unsqueeze(0) if squeeze else reach_rows
        refs, device = self._refs(state_refs)
        if keys.shape[0] != len(refs) or keys.device != device:
            raise ValueError("synthetic profile keys must match rows and device")
        if rows.shape != (len(refs), self.n_players, self.n_types):
            raise ValueError("reach_rows must have shape [B,P,X]")
        if math.prod(self.x_sizes) > 2_000_000:
            raise RuntimeError(
                "keyed synthetic profile enumeration exceeds the exact feasibility guard"
            )
        samples = int(keys.shape[1])
        output = torch.empty(
            (len(refs), samples, self.n_players),
            dtype=torch.int32,
            device=device,
        )
        for batch, ref in enumerate(refs):
            board = {card for card in self.states.board[ref] if card >= 0}
            profiles: list[tuple[int, ...]] = []
            log_weights: list[Tensor] = []
            for profile in itertools.product(
                range(self.n_types), repeat=self.n_players
            ):
                if not self._profile_feasible(profile, board):
                    continue
                profiles.append(profile)
                weight = torch.zeros((), dtype=rows.dtype, device=rows.device)
                for player, xtype in enumerate(profile):
                    weight = weight + rows[batch, player, xtype]
                log_weights.append(weight)
            weights = self._normalized_weights(
                torch.stack(log_weights).to(torch.float64)
            )
            chosen = logical_categorical_from_valid_probability(
                weights[None].expand(samples, -1), keys[batch]
            )
            table = torch.tensor(profiles, dtype=torch.int32, device=device)
            output[batch] = table[chosen]
        return output

    def encode_public(self, state_refs: Tensor) -> Tensor:
        """Return perfect public-history keys as int64 ``[B]``."""

        refs, device = self._refs(state_refs)
        return torch.tensor(
            [self.states.history_hash[ref] for ref in refs], dtype=torch.int64, device=device
        )  # [B]


class SyntheticRolloutBackend:
    """Device-resident tensor boundary backed by a Python state oracle.

    Opaque state lookup and mutation remain a small CPU control-plane operation.
    State handles, reaches, sampling, model calls, targets, and losses remain on
    ``device``.
    """

    def __init__(
        self,
        game: SyntheticPBG,
        beta_fix: Mapping[str, float] | Any | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.game = game
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.n_players = game.n_players
        self.x_sizes = game.x_sizes
        self.max_rollout_steps = (
            game.cfg.n_rounds * game.n_players * (game.cfg.max_raises + 1)
            + sum(count > 0 for count in game.cfg.board_per_round)
        )
        table = getattr(beta_fix, "beta_fix", beta_fix)
        self.beta_fix = dict(table or {"default": 1.0})
        if "default" not in self.beta_fix:
            raise ValueError("beta_fix requires a 'default' entry")
        self.execution_metrics = ExecutionMetrics()
        self.execution_families = ("synthetic",)
        self.execution_family_values = (0,)
        self.execution_family_priorities = (0,)
        # Unit row costs make every configured fixed-quantum masked pack safe
        # without reading a device-computed prefix length on the host.
        self.queued_static_pack_safe = True

    def root_states(self, n: int) -> Tensor:
        """Return opaque int32 root handles ``[B]`` on the backend device."""

        return self.game.root_states(n).to(self.device)

    def classify(self, states: Tensor) -> Tensor:
        """Return owner codes ``[B]`` using the corrected completion cut."""

        refs, _ = self.game._refs(states)
        result: list[int] = []
        for ref in refs:
            state = self.game.states.state(ref)
            if state.phase == 2:
                result.append(TERMINAL if sum(state.live) == 1 else PRETERM)
            elif state.phase == 1:
                result.append(CHANCE)
            else:
                result.append(state.to_act)
        return torch.tensor(result, dtype=torch.int32, device=self.device)  # [B]

    def legal_events(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Return public event ids/masks ``[B,E_max]``."""

        return self.game.legal_events(states)

    def policy_forward(self, frozen_model: Any, states: Tensor, reach: Tensor) -> Any:
        """Evaluate all private types at player states ``[B]``."""

        if states.numel() > 1:
            unique, inverse = torch.unique(states, sorted=True, return_inverse=True)
            if unique.numel() < states.numel():
                order = torch.argsort(states)
                sorted_states = states[order]
                first_mask = torch.ones_like(sorted_states, dtype=torch.bool)
                first_mask[1:] = sorted_states[1:] != sorted_states[:-1]
                first = order[first_mask]
                compact = self.policy_forward(
                    frozen_model, states[first], reach[first]
                )
                replacements = {
                    name: value[inverse]
                    for name in compact.__dataclass_fields__
                    if isinstance((value := getattr(compact, name)), Tensor)
                    and value.ndim > 0
                    and value.shape[0] == unique.numel()
                }
                return replace(compact, **replacements)
        legal_ids, legal_mask = self.legal_events(states)
        width = int(getattr(frozen_model, "e_max", self.game.n_player_events))
        head_mask = torch.zeros(
            (states.shape[0], width), dtype=torch.bool, device=self.device
        )  # [B,E]
        rows = torch.arange(states.shape[0], device=self.device)[:, None].expand_as(legal_ids)
        head_mask[rows[legal_mask], legal_ids[legal_mask].to(torch.int64)] = True
        kwargs: dict[str, Tensor] = {}
        if getattr(frozen_model, "uses_reach_context", False):
            refs, _ = self.game._refs(states)
            kwargs = {
                "reach": reach,
                "actor": torch.tensor(
                    [self.game.states.to_act[ref] for ref in refs],
                    dtype=torch.long,
                    device=self.device,
                ),
                "live": torch.tensor(
                    [self.game.states.live[ref] for ref in refs],
                    dtype=torch.bool,
                    device=self.device,
                ),
            }
        output = frozen_model(self.game.encode_public(states), head_mask, **kwargs)
        rows = int(states.numel())
        self.execution_metrics.record_model_call(
            "training" if torch.is_grad_enabled() else "inference",
            "synthetic",
            rows,
            padded_tokens=rows,
            padded_token_pairs=rows,
        )
        return output

    def plan_policy_batches(
        self, model: Any, states: Tensor, phase: str
    ) -> list[Tensor]:
        """Synthetic models fit in one physical CPU batch."""

        del model
        if phase not in {"training", "inference"}:
            raise ValueError("phase must be 'training' or 'inference'")
        return [torch.arange(states.numel(), device=states.device)] if states.numel() else []

    def sample_profiles(
        self, states: Tensor, reach: Tensor, gen: torch.Generator
    ) -> Tensor:
        """Draw one posterior profile per state as int32 ``[B,P]``."""

        return self.game.sample_profiles(states, reach, 1, gen)[:, 0]  # [B,P]

    def sample_profiles_keyed(
        self, states: Tensor, reach: Tensor, keys: Tensor
    ) -> Tensor:
        """Draw exact finite-game profiles from logical-address keys ``[B]``."""

        if keys.shape != states.shape or keys.dtype != torch.int64:
            raise ValueError("profile keys must be int64 [B]")
        return self.game.sample_profiles_keyed(
            states, reach, keys[:, None]
        )[:, 0]

    def sampled_terminal_reward(
        self, states: Tensor, profiles: Tensor, gen: torch.Generator
    ) -> Tensor:
        """Settle one sampled private profile at each synthetic leaf."""

        del gen
        return self.game.reward(profiles, states).float()

    def sampled_terminal_reward_keyed(
        self, states: Tensor, profiles: Tensor, keys: Tensor
    ) -> Tensor:
        """Settle deterministic synthetic leaves under logical completion keys."""

        if keys.shape != states.shape or keys.dtype != torch.int64:
            raise ValueError("terminal reward keys must be int64 [B]")
        return self.game.reward(profiles, states).float()

    @staticmethod
    def _log_prob(
        adapter: Any, policy_out: Any, legal_ids: Tensor, legal_mask: Tensor
    ) -> Tensor:
        try:
            value = adapter.log_prob(policy_out, legal_ids, legal_ids, legal_mask)
        except TypeError:
            try:
                value = adapter.log_prob(policy_out, legal_ids, legal_mask)
            except TypeError:
                value = adapter.log_prob(policy_out, legal_ids)
        if value.shape[1] != legal_ids.shape[1] and value.shape[2] == legal_ids.shape[1]:
            value = value.transpose(1, 2)
        return value.float().masked_fill(~legal_mask[..., None], -torch.inf)  # [B,E,X]

    def sample_events(
        self,
        states: Tensor,
        profiles: Tensor,
        policy_out: Any | None,
        adapter: Any,
        gen: torch.Generator,
    ) -> EventBatch:
        """Draw one public event per row and return an opaque event batch."""

        legal_ids, legal_mask = self.legal_events(states)
        owners = self.classify(states)
        if policy_out is None:
            probability = self.game.chance_probs(states, profiles)
        else:
            all_logp = self._log_prob(adapter, policy_out, legal_ids, legal_mask)
            actor_type = profiles.gather(1, owners.to(torch.int64)[:, None]).squeeze(1)
            rows = torch.arange(states.shape[0], device=self.device)
            probability = torch.exp(all_logp[rows, :, actor_type])
        probability = probability.masked_fill(~legal_mask, 0.0)
        normalizer = probability.sum(dim=1, keepdim=True)
        if bool((normalizer <= 0).any()):
            raise RuntimeError("a rollout event distribution has zero mass")
        columns = torch.multinomial(probability / normalizer, 1, generator=gen)
        public_id = legal_ids.gather(1, columns).squeeze(1).to(torch.int32)  # [B]
        return EventBatch(public_id=public_id)

    def sample_events_keyed(
        self,
        states: Tensor,
        profiles: Tensor,
        policy_out: Any | None,
        adapter: Any,
        keys: Tensor,
    ) -> EventBatch:
        """Draw one public event per row from logical-address keys."""

        if keys.shape != states.shape or keys.dtype != torch.int64:
            raise ValueError("event keys must be int64 [B]")
        legal_ids, legal_mask = self.legal_events(states)
        owners = self.classify(states)
        if policy_out is None:
            probability = self.game.chance_probs(states, profiles)
        else:
            all_logp = self._log_prob(adapter, policy_out, legal_ids, legal_mask)
            actor_type = profiles.gather(1, owners.to(torch.int64)[:, None]).squeeze(1)
            rows = torch.arange(states.shape[0], device=self.device)
            probability = torch.exp(all_logp[rows, :, actor_type])
        probability = probability.masked_fill(~legal_mask, 0.0)
        columns = logical_categorical_from_valid_probability(probability, keys)
        public_id = legal_ids.gather(1, columns[:, None]).squeeze(1)
        return EventBatch(public_id=public_id.to(torch.int32))

    def rollout_family_ids(self, states: Tensor) -> Tensor:
        """Route every synthetic player occurrence to its one family."""

        return torch.zeros(states.shape, dtype=torch.int64, device=states.device)

    def rollout_row_cost(
        self, model: Any, states: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return unit padded work for the small synthetic executor."""

        del model
        ones = torch.ones(states.shape, dtype=torch.int64, device=states.device)
        return ones, torch.zeros_like(ones)

    def step(self, states: Tensor, events: EventBatch) -> Tensor:
        """Apply ``events`` and return successor handles ``[B]``."""

        return self.game.step(states, events.public_id)

    def reach_update(
        self,
        reach: Tensor,
        states: Tensor,
        events: EventBatch,
        policy_out: Any | None,
    ) -> Tensor:
        """Apply selected action/chance likelihoods to log reach ``[B,P,X]``."""

        result = reach.clone()  # [B,P,X]
        refs, _ = self.game._refs(states)
        event_ids = events.public_id.detach().cpu().tolist()
        if policy_out is not None:
            log_policy = torch.log_softmax(policy_out.raw, dim=-1)  # [B,X,E]
        for row, (ref, event) in enumerate(zip(refs, event_ids, strict=True)):
            state = self.game.states.state(ref)
            if state.phase == 0:
                result[row, state.to_act] += log_policy[row, :, int(event)]
            else:
                revealed = set(self.game._event_cards(ref, int(event)))
                conflict = torch.tensor(
                    [not revealed.isdisjoint(hand) for hand in self.game._hand_sets],
                    dtype=torch.bool,
                    device=self.device,
                )  # [X]
                result[row, :, conflict] = -torch.inf
        return result

    def beta_fn(self, states: Tensor) -> Tensor:
        """Return configured fixed intensity for synthetic stage buckets ``[B]``."""

        refs, _ = self.game._refs(states)
        values: list[float] = []
        for ref in refs:
            round_index = self.game.states.round[ref]
            if self.game.states.phase[ref] == 1:
                bucket = f"chance_round_{round_index}"
            else:
                bucket = f"round_{round_index}"
            # Retain historical spellings for programmatic callers while the
            # shipped presets use the P13B public-stage names above.
            legacy = f"round{round_index}_{'chance' if self.game.states.phase[ref] == 1 else 'player'}"
            values.append(
                float(self.beta_fix.get(bucket, self.beta_fix.get(legacy, self.beta_fix["default"])))
            )
        result = torch.tensor(values, dtype=torch.float32, device=self.device)  # [B]
        if not bool(torch.isfinite(result).all()) or bool((result < 1.0).any()):
            raise ValueError("rollout beta must be finite and at least one")
        return result
