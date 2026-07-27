"""Queued NLHE hooks for the self-play sequential sampled path.

The module owns poker geometry, public transitions, direct model inputs,
collision-free private roots, sampled leaves, and the evaluator-v2 showdown
boundary.  Generic queued modules remain game-opaque.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from pbg.queued.contracts import (
    ModeSchema,
    Partition,
    PartitionContract,
    PartitionMap,
    PrivateEstimator,
    RolloutMode,
    TrainingRegime,
)
from pbg.queued.rollout import (
    CommitCoverage,
    CommitPlan,
    CompletionBatch,
    CompletionPlan,
    DrawPurpose,
    FamilyBatch,
    RolloutOwnerRegistry,
    SampleRequest,
    SampleWorkspace,
    SemanticFamilyFIFO,
    TransitionRequest,
    child_path_keys,
    logical_rng_keys,
    stable_route,
)
from pbg.queued.storage import DeviceDestinationRequest, MonotoneSlab
from pbg.games.poker.model.capacity import max_public_history_tokens
from pbg.games.poker.queued.policy import (
    CHECK_CALL,
    FOLD,
    RAISE_OFFSET,
    PokerLegality,
    PokerPolicyOperator,
)
from pbg.queued.learner import ModelEvaluation, PolicyEvaluation


PAD_CARD = 255
LIVE = 0
FOLDED = 1
ALL_IN = 2


class QueuedPokerError(RuntimeError):
    """The queued poker state or evaluator contract was violated."""


@dataclass(frozen=True)
class NLHEGeometry:
    players: int
    stack_bb: float
    rank_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.players, bool)
            or not isinstance(self.players, int)
            or not 2 <= self.players <= 6
        ):
            raise ValueError("players must be an integer in [2,6]")
        if (
            isinstance(self.stack_bb, bool)
            or not isinstance(self.stack_bb, (int, float))
            or not math.isfinite(self.stack_bb)
            or not 8 <= self.stack_bb <= 128
            or not math.isclose(self.stack_bb * 2, round(self.stack_bb * 2))
        ):
            raise ValueError("stack_bb must be a half-integer in [8,128]")
        if (
            isinstance(self.rank_count, bool)
            or not isinstance(self.rank_count, int)
            or not 4 <= self.rank_count <= 13
        ):
            raise ValueError("rank_count must be an integer in [4,13]")
        if 2 * self.players + 5 >= self.deck_cards:
            raise ValueError("NLHE geometry must satisfy 2P+5<4R")

    @property
    def deck_cards(self) -> int:
        return 4 * self.rank_count

    @property
    def first_card(self) -> int:
        return 52 - self.deck_cards

    @property
    def private_hands(self) -> int:
        return self.deck_cards * (self.deck_cards - 1) // 2

    @property
    def stack_ticks(self) -> int:
        return int(round(self.stack_bb * 2))

    @property
    def player_node_bound(self) -> int:
        return int(math.ceil(self.stack_bb * self.players + 3 * self.players - self.stack_bb + 1))

    @property
    def direct_model_input_width(self) -> int:
        return (
            8
            + 9 * self.players
            + self.players * self.private_hands
            + 10 * max_public_history_tokens(self.players, self.stack_bb)
        )

    def combo_id(self, first: Tensor, second: Tensor) -> Tensor:
        low = torch.minimum(first, second).to(torch.int64) - self.first_card
        high = torch.maximum(first, second).to(torch.int64) - self.first_card
        return (
            low * (2 * self.deck_cards - low - 1) // 2 + high - low - 1
        ).to(torch.int32)


def sampled_partitions(players: int) -> PartitionContract:
    if not 2 <= players <= 6:
        raise ValueError("players must lie in [2,6]")
    inputs = (Partition("pre", 1),) + tuple(
        Partition(f"post_{live}", 1) for live in range(2, players + 1)
    )
    return PartitionContract(
        inputs=inputs,
        outputs=(Partition("all", 1),),
        folded=(Partition("pre", 1), Partition("post", 1)),
        input_to_output=tuple(
            PartitionMap(partition.name, "all") for partition in inputs
        ),
        conversions=(),
    )


@dataclass(frozen=True)
class PokerState:
    """Accelerator-resident public state for a batch of independent hands."""

    board_cards: Tensor
    combo_ids: Tensor
    button: Tensor
    street: Tensor
    to_act: Tensor
    stacks: Tensor
    committed: Tensor
    committed_street: Tensor
    status: Tensor
    current_bet: Tensor
    last_full_raise: Tensor
    pending: Tensor
    terminal: Tensor

    @property
    def rows(self) -> int:
        return self.board_cards.shape[0]

    @property
    def device(self) -> torch.device:
        return self.board_cards.device

    def validate(self, geometry: NLHEGeometry) -> None:
        batch, players = self.rows, geometry.players
        contracts = {
            "board_cards": (self.board_cards, torch.uint8, (batch, 5)),
            "combo_ids": (self.combo_ids, torch.int32, (batch, players)),
            "button": (self.button, torch.int32, (batch,)),
            "street": (self.street, torch.int32, (batch,)),
            "to_act": (self.to_act, torch.int32, (batch,)),
            "stacks": (self.stacks, torch.float32, (batch, players)),
            "committed": (self.committed, torch.float32, (batch, players)),
            "committed_street": (
                self.committed_street,
                torch.float32,
                (batch, players),
            ),
            "status": (self.status, torch.int8, (batch, players)),
            "current_bet": (self.current_bet, torch.float32, (batch,)),
            "last_full_raise": (self.last_full_raise, torch.float32, (batch,)),
            "pending": (self.pending, torch.bool, (batch, players)),
            "terminal": (self.terminal, torch.bool, (batch,)),
        }
        for name, (value, dtype, shape) in contracts.items():
            if (
                value.device != self.device
                or value.device.type != "cuda"
                or value.dtype != dtype
                or value.shape != shape
            ):
                raise QueuedPokerError(
                    f"{name} must be CUDA {dtype} with shape {shape}"
                )
        torch._assert_async(
            ((self.button >= 0) & (self.button < players)).all(),
            "button is outside the table",
        )
        torch._assert_async(
            ((self.street >= 0) & (self.street <= 3)).all(),
            "street is outside NLHE",
        )
        torch._assert_async(
            (self.stacks >= 0).all(), "poker stacks cannot be negative"
        )
        total = self.stacks + self.committed
        torch._assert_async(
            torch.isclose(
                total.sum(-1),
                torch.full_like(total.sum(-1), geometry.players * geometry.stack_bb),
                atol=1.0e-5,
                rtol=0.0,
            ).all(),
            "public chip accounting is not conserved",
        )


@dataclass(frozen=True)
class PokerHistory:
    actor: Tensor
    street: Tensor
    action: Tensor
    amount: Tensor
    pot_before: Tensor
    call_amount: Tensor
    current_bet: Tensor
    min_raise_to: Tensor
    max_raise_to: Tensor
    mask: Tensor


def _next_seat(mask: Tensor, after: Tensor) -> Tensor:
    players = mask.shape[1]
    offsets = torch.arange(1, players + 1, device=mask.device)
    candidates = (after[:, None].to(torch.int64) + offsets) % players
    eligible = mask.gather(1, candidates)
    first = eligible.to(torch.int64).argmax(-1)
    return candidates.gather(1, first[:, None]).squeeze(1).to(torch.int32)


class PokerStateMachine:
    """Vectorized equal-stack no-rake/no-ante NLHE public transitions."""

    def __init__(self, geometry: NLHEGeometry) -> None:
        self.geometry = geometry

    def initial_roots(
        self,
        root_count: int,
        *,
        seed: int,
        iteration: int,
        retry_epoch: int,
        device: torch.device | str,
    ) -> PokerState:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("queued poker roots require CUDA")
        if root_count <= 0:
            raise ValueError("root_count must be positive")
        roots = torch.arange(root_count, dtype=torch.int64, device=device)
        cards = torch.arange(
            self.geometry.first_card, 52, dtype=torch.int64, device=device
        )
        card_draw = torch.arange(
            self.geometry.deck_cards, dtype=torch.int64, device=device
        )
        keys = logical_rng_keys(
            seed=seed,
            iteration=iteration,
            root=roots[:, None].expand(-1, self.geometry.deck_cards),
            path=torch.zeros(
                (root_count, self.geometry.deck_cards),
                dtype=torch.int64,
                device=device,
            ),
            draw=card_draw[None, :],
            purpose=DrawPurpose.PRIVATE,
            retry_epoch=retry_epoch,
        )
        permutation = torch.argsort(keys, dim=1, stable=True)
        private_cards = cards[permutation[:, : 2 * self.geometry.players]].reshape(
            root_count, self.geometry.players, 2
        )
        combo_ids = self.geometry.combo_id(
            private_cards[..., 0], private_cards[..., 1]
        )
        button = (roots % self.geometry.players).to(torch.int32)
        seats = torch.arange(self.geometry.players, device=device)
        if self.geometry.players == 2:
            small_blind = button
            big_blind = (button + 1) % 2
            to_act = button.clone()
        else:
            small_blind = (button + 1) % self.geometry.players
            big_blind = (button + 2) % self.geometry.players
            to_act = (button + 3) % self.geometry.players
        committed = torch.zeros(
            (root_count, self.geometry.players),
            dtype=torch.float32,
            device=device,
        )
        committed.scatter_(1, small_blind[:, None].to(torch.int64), 0.5)
        committed.scatter_(1, big_blind[:, None].to(torch.int64), 1.0)
        stacks = torch.full_like(committed, float(self.geometry.stack_bb)) - committed
        status = torch.zeros_like(committed, dtype=torch.int8)
        pending = torch.ones_like(committed, dtype=torch.bool)
        state = PokerState(
            board_cards=torch.full(
                (root_count, 5), PAD_CARD, dtype=torch.uint8, device=device
            ),
            combo_ids=combo_ids,
            button=button,
            street=torch.zeros(root_count, dtype=torch.int32, device=device),
            to_act=to_act,
            stacks=stacks,
            committed=committed,
            committed_street=committed.clone(),
            status=status,
            current_bet=torch.ones(root_count, dtype=torch.float32, device=device),
            last_full_raise=torch.ones(root_count, dtype=torch.float32, device=device),
            pending=pending,
            terminal=torch.zeros(root_count, dtype=torch.bool, device=device),
        )
        state.validate(self.geometry)
        return state

    def input_type(self, state: PokerState) -> Tensor:
        """Return device family ids: pre=0 and post_A=A-1."""

        contenders = state.status.ne(FOLDED).sum(-1, dtype=torch.int32)
        return torch.where(
            state.street.eq(0), torch.zeros_like(contenders), contenders - 1
        )

    def legality(self, state: PokerState) -> PokerLegality:
        state.validate(self.geometry)
        rows = state.rows
        events = torch.arange(
            RAISE_OFFSET + self.geometry.stack_ticks,
            dtype=torch.int32,
            device=state.device,
        )[None, :].expand(rows, -1)
        mask = torch.zeros_like(events, dtype=torch.bool)
        row = torch.arange(rows, device=state.device)
        actor = state.to_act.to(torch.int64)
        actor_stack = state.stacks[row, actor]
        actor_street = state.committed_street[row, actor]
        call = (state.current_bet - actor_street).clamp_min(0.0)
        mask[:, FOLD] = call > 0
        mask[:, CHECK_CALL].fill_(True)
        target = (
            torch.arange(
                1,
                self.geometry.stack_ticks + 1,
                dtype=torch.float32,
                device=state.device,
            )
            * 0.5
        )
        maximum = actor_street + actor_stack
        minimum = state.current_bet + state.last_full_raise
        full_raise = (target[None, :] >= minimum[:, None]) & (
            target[None, :] <= maximum[:, None]
        )
        short_all_in = target[None, :].eq(maximum[:, None]) & (
            maximum[:, None] > state.current_bet[:, None]
        )
        multiple_actors = state.status.eq(LIVE).sum(-1).gt(1)
        mask[:, RAISE_OFFSET:] = (full_raise | short_all_in) & multiple_actors[:, None]
        mask &= ~state.terminal[:, None]
        return PokerLegality(events.contiguous(), mask.contiguous())

    def empty_history(self, rows: int, device: torch.device) -> PokerHistory:
        tokens = max_public_history_tokens(
            self.geometry.players, self.geometry.stack_bb
        )
        integer = torch.full(
            (rows, tokens), -1, dtype=torch.int64, device=device
        )
        numeric = torch.zeros(
            (rows, tokens), dtype=torch.float32, device=device
        )
        return PokerHistory(
            integer.clone(),
            integer.clone(),
            integer.clone(),
            numeric.clone(),
            numeric.clone(),
            numeric.clone(),
            numeric.clone(),
            numeric.clone(),
            numeric.clone(),
            torch.zeros_like(numeric, dtype=torch.bool),
        )

    def direct_model_input(
        self,
        state: PokerState,
        log_reach: Tensor | None = None,
        history: PokerHistory | None = None,
    ) -> Tensor:
        """Return the retained public, reach, and history model tensor."""

        state.validate(self.geometry)
        scale = float(self.geometry.stack_bb)
        board = state.board_cards.float()
        board = torch.where(
            state.board_cards == PAD_CARD,
            torch.full_like(board, -1.0),
            (board - self.geometry.first_card) / max(1, self.geometry.deck_cards - 1),
        )
        seats = torch.nn.functional.one_hot(
            state.to_act.to(torch.int64), self.geometry.players
        ).float()
        button = torch.nn.functional.one_hot(
            state.button.to(torch.int64), self.geometry.players
        ).float()
        status = torch.nn.functional.one_hot(
            state.status.to(torch.int64), 3
        ).float().flatten(1)
        public = torch.cat(
            (
                state.street[:, None].float() / 3.0,
                board,
                seats,
                button,
                state.stacks / scale,
                state.committed / scale,
                state.committed_street / scale,
                status,
                state.pending.float(),
                (state.current_bet / scale)[:, None],
                (state.last_full_raise / scale)[:, None],
            ),
            1,
        )
        if log_reach is None:
            log_reach = torch.zeros(
                (
                    state.rows,
                    self.geometry.players,
                    self.geometry.private_hands,
                ),
                dtype=torch.float32,
                device=state.device,
            )
        if history is None:
            history = self.empty_history(state.rows, state.device)
        history_values = tuple(
            getattr(history, name).float()
            for name in history.__dataclass_fields__
        )
        return torch.cat(
            (public, log_reach.flatten(1), *history_values), 1
        ).contiguous()

    def model_context(
        self, model_input: Tensor
    ) -> tuple[Tensor, PokerHistory]:
        players = self.geometry.players
        hands = self.geometry.private_hands
        tokens = max_public_history_tokens(
            players, self.geometry.stack_bb
        )
        offset = 8 + 9 * players
        reach = model_input[:, offset : offset + players * hands].float()
        offset += players * hands
        columns = []
        for _ in range(10):
            columns.append(model_input[:, offset : offset + tokens])
            offset += tokens
        if offset != model_input.shape[1]:
            raise QueuedPokerError("retained model context layout is not exact")
        return reach.reshape(-1, players, hands), PokerHistory(
            columns[0].round().to(torch.int64),
            columns[1].round().to(torch.int64),
            columns[2].round().to(torch.int64),
            columns[3].float(),
            columns[4].float(),
            columns[5].float(),
            columns[6].float(),
            columns[7].float(),
            columns[8].float(),
            columns[9].gt(0.5),
        )

    @staticmethod
    def _append_history(
        history: PokerHistory,
        active: Tensor,
        values: Mapping[str, Tensor],
    ) -> PokerHistory:
        length = history.mask.sum(1, dtype=torch.int64)
        torch._assert_async(
            (~active | length.lt(history.mask.shape[1])).all(),
            "poker public history capacity exceeded",
        )
        position = torch.arange(
            history.mask.shape[1], device=history.mask.device
        )[None].eq(length[:, None])
        selected = active[:, None] & position
        result = {}
        for name in history.__dataclass_fields__:
            current = getattr(history, name)
            if name == "mask":
                result[name] = current | selected
            else:
                value = values[name].to(current.dtype)[:, None]
                result[name] = torch.where(selected, value, current)
        return PokerHistory(**result)

    def append_player_history(
        self, state: PokerState, action: Tensor, history: PokerHistory
    ) -> PokerHistory:
        legality = self.legality(state)
        raise_mask = legality.mask[:, RAISE_OFFSET:]
        targets = (
            torch.arange(
                1,
                self.geometry.stack_ticks + 1,
                dtype=torch.float32,
                device=state.device,
            )
            * 0.5
        )
        lower = targets[None].masked_fill(
            ~raise_mask, torch.inf
        ).min(1).values
        upper = targets[None].masked_fill(
            ~raise_mask, -torch.inf
        ).max(1).values
        has_raise = raise_mask.any(1)
        lower = torch.where(has_raise, lower, torch.zeros_like(lower))
        upper = torch.where(has_raise, upper, torch.zeros_like(upper))
        actor = state.to_act.to(torch.int64)
        actor_stack = state.stacks.gather(1, actor[:, None]).squeeze(1)
        actor_committed = state.committed_street.gather(
            1, actor[:, None]
        ).squeeze(1)
        call = (state.current_bet - actor_committed).clamp_min(0.0)
        paid_call = torch.minimum(call, actor_stack)
        raise_to = (action.float() - RAISE_OFFSET + 1.0) * 0.5
        raised = action.ge(RAISE_OFFSET)
        called = action.eq(CHECK_CALL)
        amount = torch.where(
            raised,
            raise_to,
            torch.where(called, paid_call, torch.zeros_like(call)),
        )
        action_type = torch.where(
            raised,
            torch.full_like(action, 2),
            torch.where(called, torch.ones_like(action), torch.zeros_like(action)),
        )
        return self._append_history(
            history,
            ~state.terminal,
            {
                "actor": actor,
                "street": state.street,
                "action": action_type,
                "amount": amount,
                "pot_before": state.committed.sum(1),
                "call_amount": call,
                "current_bet": state.current_bet,
                "min_raise_to": lower,
                "max_raise_to": upper,
            },
        )

    def append_chance_history(
        self,
        state: PokerState,
        active: Tensor,
        history: PokerHistory,
    ) -> PokerHistory:
        zeros = torch.zeros_like(state.current_bet)
        return self._append_history(
            history,
            active,
            {
                "actor": torch.full_like(state.to_act, -1),
                "street": state.street,
                "action": torch.full_like(state.to_act, 3),
                "amount": zeros,
                "pot_before": state.committed.sum(1),
                "call_amount": zeros,
                "current_bet": state.current_bet,
                "min_raise_to": zeros,
                "max_raise_to": zeros,
            },
        )

    def state_from_model_input(
        self, model_input: Tensor, combo_ids: Tensor
    ) -> PokerState:
        """Recover exact public transition state from the retained input row."""

        players = self.geometry.players
        if (
            model_input.device.type != "cuda"
            or model_input.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or model_input.ndim != 2
            or model_input.shape[1] != self.geometry.direct_model_input_width
            or combo_ids.shape != (model_input.shape[0], players)
            or combo_ids.dtype != torch.int32
            or combo_ids.device != model_input.device
        ):
            raise QueuedPokerError("retained model input cannot reconstruct poker state")
        value = model_input.float()
        offset = 0

        def take(width: int) -> Tensor:
            nonlocal offset
            result = value[:, offset : offset + width]
            offset += width
            return result

        street = torch.round(take(1)[:, 0] * 3).to(torch.int32)
        board_value = take(5)
        board = torch.where(
            board_value < 0,
            torch.full_like(board_value, PAD_CARD),
            torch.round(
                board_value * max(1, self.geometry.deck_cards - 1)
                + self.geometry.first_card
            ),
        ).to(torch.uint8)
        to_act = take(players).argmax(-1).to(torch.int32)
        button = take(players).argmax(-1).to(torch.int32)
        scale = float(self.geometry.stack_bb)
        stacks = take(players) * scale
        committed = take(players) * scale
        committed_street = take(players) * scale
        status = take(3 * players).reshape(-1, players, 3).argmax(-1).to(torch.int8)
        pending = take(players).gt(0.5)
        current_bet = take(1)[:, 0] * scale
        last_full_raise = take(1)[:, 0] * scale
        if offset != 8 + 9 * players:
            raise QueuedPokerError("retained public input layout is not exact")
        state = PokerState(
            board,
            combo_ids,
            button,
            street,
            to_act,
            stacks,
            committed,
            committed_street,
            status,
            current_bet,
            last_full_raise,
            pending,
            torch.zeros(model_input.shape[0], dtype=torch.bool, device=model_input.device),
        )
        state.validate(self.geometry)
        return state

    def step_player(self, state: PokerState, action: Tensor) -> PokerState:
        legality = self.legality(state)
        if action.shape != (state.rows,):
            raise QueuedPokerError("player action must have shape [B]")
        matches = legality.event_ids.eq(action[:, None].to(legality.event_ids))
        torch._assert_async(
            (matches & legality.mask).sum(-1).eq(1).all(),
            "player transition received an illegal action",
        )
        actor = state.to_act.to(torch.int64)
        stacks = state.stacks.clone()
        committed = state.committed.clone()
        committed_street = state.committed_street.clone()
        status = state.status.clone()
        pending = state.pending.clone()
        current_bet = state.current_bet.clone()
        last_full_raise = state.last_full_raise.clone()
        fold = action.eq(FOLD)
        call_action = action.eq(CHECK_CALL)
        raise_action = action.ge(RAISE_OFFSET)
        actor_index = actor[:, None]
        actor_status = status.gather(1, actor_index).squeeze(1)
        actor_status = torch.where(
            fold, torch.full_like(actor_status, FOLDED), actor_status
        )
        status.scatter_(1, actor_index, actor_status[:, None])
        pending.scatter_(
            1,
            actor_index,
            torch.zeros_like(actor_index, dtype=torch.bool),
        )
        actor_committed = committed_street.gather(
            1, actor_index
        ).squeeze(1)
        actor_stack = stacks.gather(1, actor_index).squeeze(1)
        call = (current_bet - actor_committed).clamp_min(0.0)
        paid_call = torch.minimum(call, actor_stack)
        paid = torch.where(call_action, paid_call, torch.zeros_like(paid_call))
        raise_to = (action.float() - RAISE_OFFSET + 1.0) * 0.5
        paid_raise = (raise_to - actor_committed).clamp_min(0.0)
        paid = torch.where(raise_action, paid_raise, paid)
        paid = torch.where(fold, torch.zeros_like(paid), paid)
        next_actor_stack = actor_stack - paid
        stacks.scatter_(1, actor_index, next_actor_stack[:, None])
        committed.scatter_add_(1, actor_index, paid[:, None])
        committed_street.scatter_add_(1, actor_index, paid[:, None])
        all_in = next_actor_stack.eq(0) & ~fold
        actor_status = torch.where(
            all_in, torch.full_like(actor_status, ALL_IN), actor_status
        )
        status.scatter_(1, actor_index, actor_status[:, None])
        old_bet = current_bet.clone()
        current_bet = torch.where(raise_action, raise_to, current_bet)
        raise_size = raise_to - old_bet
        full_raise = raise_action & (raise_size >= last_full_raise)
        last_full_raise = torch.where(full_raise, raise_size, last_full_raise)
        active = status.eq(LIVE)
        pending = torch.where(full_raise[:, None] & active, active, pending)
        pending.scatter_(
            1,
            actor_index,
            torch.zeros_like(actor_index, dtype=torch.bool),
        )
        contenders = status.ne(FOLDED)
        terminal = contenders.sum(-1).le(1)
        outstanding = pending & active
        round_complete = ~outstanding.any(-1)
        terminal |= round_complete & state.street.eq(3)
        next_actor = _next_seat(active, actor.to(torch.int32))
        result = PokerState(
            state.board_cards,
            state.combo_ids,
            state.button,
            state.street,
            next_actor,
            stacks,
            committed,
            committed_street,
            status,
            current_bet,
            last_full_raise,
            pending,
            terminal,
        )
        result.validate(self.geometry)
        return result

    def needs_chance(self, state: PokerState) -> Tensor:
        active = state.status.eq(LIVE)
        return (
            ~state.terminal
            & ~state.pending.logical_and(active).any(-1)
            & state.street.lt(3)
        )

    def deal_next_street(
        self,
        state: PokerState,
        logical_key: Tensor,
    ) -> PokerState:
        """Deal the next public street from the unused compact deck."""

        needs = self.needs_chance(state)
        if logical_key.shape != (state.rows, self.geometry.deck_cards):
            raise QueuedPokerError("chance keys must be [B,D]")
        cards = torch.arange(
            self.geometry.first_card, 52, dtype=torch.int64, device=state.device
        )
        used = torch.zeros(
            (state.rows, self.geometry.deck_cards),
            dtype=torch.bool,
            device=state.device,
        )
        combo_cards = self.combo_cards(state.combo_ids)
        used.scatter_(1, (combo_cards.flatten(1) - self.geometry.first_card), True)
        board_valid = state.board_cards != PAD_CARD
        used |= (
            board_valid[..., None]
            & state.board_cards.long()[..., None].eq(cards[None, None, :])
        ).any(1)
        ordering_key = torch.where(
            used, torch.full_like(logical_key, torch.iinfo(torch.int64).max), logical_key
        )
        order = torch.argsort(ordering_key, dim=1, stable=True)
        dealt = cards[order[:, :3]]
        board = state.board_cards.clone()
        flop = state.street.eq(0) & needs
        turn = state.street.eq(1) & needs
        river = state.street.eq(2) & needs
        board[:, :3] = torch.where(
            flop[:, None], dealt[:, :3].to(torch.uint8), board[:, :3]
        )
        board[:, 3] = torch.where(
            turn, dealt[:, 0].to(torch.uint8), board[:, 3]
        )
        board[:, 4] = torch.where(
            river, dealt[:, 0].to(torch.uint8), board[:, 4]
        )
        street = state.street + needs.to(torch.int32)
        active = state.status.eq(LIVE)
        has_betting_opponent = active.sum(-1).gt(1)
        next_pending = active & has_betting_opponent[:, None]
        pending = torch.where(needs[:, None], next_pending, state.pending)
        committed_street = torch.where(
            needs[:, None], torch.zeros_like(state.committed_street), state.committed_street
        )
        zero = torch.zeros_like(state.current_bet)
        next_actor = _next_seat(active, state.button)
        result = PokerState(
            board,
            state.combo_ids,
            state.button,
            street,
            torch.where(needs, next_actor, state.to_act),
            state.stacks,
            state.committed,
            committed_street,
            state.status,
            torch.where(needs, zero, state.current_bet),
            torch.where(needs, torch.ones_like(zero), state.last_full_raise),
            pending,
            state.terminal
            | (
                street.eq(3)
                & ~pending.logical_and(active).any(-1)
            ),
        )
        result.validate(self.geometry)
        return result

    def deal_next_street_keyed(
        self,
        state: PokerState,
        *,
        seed: int,
        iteration: int,
        root: Tensor,
        path: Tensor,
        draw: int,
        retry_epoch: Tensor | int,
    ) -> PokerState:
        """Own chance labels at schedule-independent logical RNG addresses."""

        if (
            root.shape != (state.rows,)
            or path.shape != root.shape
            or root.device != state.device
            or path.device != state.device
        ):
            raise QueuedPokerError("chance root/path addresses must be CUDA [B]")
        card_ordinal = torch.arange(
            self.geometry.deck_cards, dtype=torch.int64, device=state.device
        )
        active = self.needs_chance(state)
        expanded_retry = (
            retry_epoch[:, None]
            if isinstance(retry_epoch, Tensor) and retry_epoch.shape == root.shape
            else retry_epoch
        )
        key = logical_rng_keys(
            seed=seed,
            iteration=iteration,
            root=root[:, None].expand(-1, self.geometry.deck_cards),
            path=path[:, None].expand(-1, self.geometry.deck_cards),
            draw=draw * self.geometry.deck_cards + card_ordinal[None, :],
            purpose=DrawPurpose.CHANCE,
            retry_epoch=expanded_retry,
            active=active[:, None],
        )
        return self.deal_next_street(state, key)

    def combo_cards(self, combo_ids: Tensor) -> Tensor:
        """Invert contract-v2 lexicographic combo ids on the accelerator."""

        ids = combo_ids.to(torch.int64)
        first = torch.arange(
            self.geometry.deck_cards - 1,
            dtype=torch.int64,
            device=ids.device,
        )
        starts = first * (2 * self.geometry.deck_cards - first - 1) // 2
        low = (ids[..., None] >= starts).sum(-1) - 1
        remaining = ids - starts[low]
        high = low + remaining + 1
        return torch.stack(
            (low + self.geometry.first_card, high + self.geometry.first_card), -1
        )

    def sampled_leaf(self, state: PokerState) -> Mapping[str, Tensor]:
        torch._assert_async(
            state.terminal.all(),
            "sampled leaves require terminal poker states",
        )
        contenders = state.status.ne(FOLDED)
        early = contenders.sum(-1).eq(1)
        baseline = -state.committed
        amount = torch.zeros(
            (state.rows, self.geometry.players),
            dtype=torch.float32,
            device=state.device,
        )
        amount[:, 0] = state.committed.sum(-1)
        level = torch.where(
            contenders,
            torch.zeros_like(state.combo_ids),
            torch.full_like(state.combo_ids, -1),
        )
        winner = contenders.to(torch.float32)
        baseline = torch.where(
            early[:, None],
            baseline + winner * amount[:, :1],
            baseline,
        )
        amount = torch.where(early[:, None], torch.zeros_like(amount), amount)
        level = torch.where(
            early[:, None], torch.full_like(level, -1), level
        )
        combo = torch.where(
            early[:, None], torch.full_like(state.combo_ids, -1), state.combo_ids
        )
        board = state.board_cards.clone()
        if board.shape[0]:
            private_cards = self.combo_cards(state.combo_ids).flatten(1)
            used = torch.zeros(
                (state.rows, self.geometry.deck_cards),
                dtype=torch.bool,
                device=state.device,
            )
            used.scatter_(
                1, private_cards - self.geometry.first_card, True
            )
            board_valid = board.ne(PAD_CARD)
            cards = torch.arange(
                self.geometry.first_card,
                52,
                dtype=torch.int64,
                device=state.device,
            )
            used |= (
                board_valid[..., None]
                & board.long()[..., None].eq(cards[None, None, :])
            ).any(1)
            unused_order = torch.argsort(used.to(torch.uint8), dim=1, stable=True)
            unused_cards = cards[unused_order]
            missing = board.eq(PAD_CARD)
            missing_ordinal = missing.cumsum(1).sub(1).clamp_min(0)
            completion = unused_cards.gather(1, missing_ordinal)
            board = torch.where(
                early[:, None] & missing,
                completion.to(torch.uint8),
                board,
            )
        return MappingProxyType(
            {
                "board_cards": board,
                "combo_ids": combo,
                "stratum_level": level,
                "payoff_baseline": baseline,
                "stratum_amount": amount,
            }
        )


@dataclass(frozen=True)
class PokerForwardInput:
    """Direct flattened model arguments for one stable family prefix."""

    input_type: str
    model_input: Tensor
    sampled_private_state: Tensor
    owner: Tensor


@dataclass(frozen=True)
class PokerTransitionTrace:
    """Ephemeral successful-transition data used to seal CPU topology."""

    input_type: str
    logical_rng_key: Tensor
    next_logical_rng_key: Tensor
    next_input_type: Tensor
    terminal: Tensor
    owner: Tensor
    action: Tensor
    familywork_row: Tensor


class SequentialSampledPokerHooks:
    """NLHE implementation of the Phase 4 opaque rollout hook boundary.

    The hook writes no retained owner directly.  It returns one atomic plan
    containing the aligned heads plus exactly one leaf or next-family row for
    every scheduled player node.
    """

    def __init__(
        self,
        geometry: NLHEGeometry,
        owners: RolloutOwnerRegistry,
        queues: Mapping[str, SemanticFamilyFIFO],
        *,
        seed: int,
        iteration: int,
        policy: PokerPolicyOperator | None = None,
    ) -> None:
        if owners.n_replica != 1:
            raise ValueError("ordinary Phase 5 poker hooks require one replica")
        expected = tuple(item.name for item in sampled_partitions(geometry.players).inputs)
        if tuple(queues) != expected:
            raise ValueError("poker queues do not cover the sampled input preorder")
        self.geometry = geometry
        self.machine = PokerStateMachine(geometry)
        self.owners = owners
        self.queues = MappingProxyType(dict(queues))
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.policy = PokerPolicyOperator() if policy is None else policy
        self.input_index = {name: index for index, name in enumerate(expected)}
        self._route_family_values = (-1, *range(len(self.queues)))
        self._route_families = torch.tensor(
            self._route_family_values,
            dtype=torch.int32,
            device=self.owners.device,
        )
        self.last_trace: PokerTransitionTrace | None = None

    def model_input(self, batch: FamilyBatch) -> PokerForwardInput:
        shape = (batch.lane_count * batch.rows,)
        return PokerForwardInput(
            batch.input_type,
            batch.fields["model_input"].reshape(
                *shape, self.geometry.direct_model_input_width
            ),
            batch.fields["sampled_private_state"].reshape(
                *shape, self.geometry.players
            ),
            batch.fields["owner"].reshape(*shape),
        )

    def sample(self, request: SampleRequest) -> SampleWorkspace:
        forward = PokerForwardInput(
            request.batch.input_type,
            request.batch.fields["model_input"].reshape(
                -1, self.geometry.direct_model_input_width
            ),
            request.batch.fields["sampled_private_state"].reshape(
                -1, self.geometry.players
            ),
            request.batch.fields["owner"].reshape(-1),
        )
        state = self.machine.state_from_model_input(
            forward.model_input, forward.sampled_private_state
        )
        legality = self.machine.legality(state)
        raw = {
            name: value
            for name, value in request.model.values.items()
            if name != "value"
        }
        row = torch.arange(state.rows, device=state.device)
        actor_hand = state.combo_ids[
            row, forward.owner.to(torch.int64)
        ]
        sampled = self.policy.sample_actor(
            raw,
            legality,
            actor_hand,
            request.rng_key.reshape(-1),
        )
        shape = (request.batch.lane_count, request.batch.rows)
        return SampleWorkspace(
            MappingProxyType(
                {
                    "action": sampled.action.reshape(shape),
                    "selected_log_policy": sampled.selected_log_policy.reshape(shape),
                }
            ),
            torch.full(
                (request.batch.lane_count,),
                request.batch.rows,
                dtype=torch.int64,
                device=state.device,
            ),
        )

    @staticmethod
    def _defaults(
        slab: MonotoneSlab, rows: int
    ) -> dict[str, Tensor]:
        return {
            field.name: torch.full(
                (slab.n_replica, rows, *field.shape),
                field.fill,
                dtype=field.dtype,
                device=slab.device,
            )
            for field in slab.schema.fields
        }

    @staticmethod
    def _gather_route(
        destination: dict[str, Tensor],
        source: Mapping[str, Tensor],
        source_row: Tensor,
        lanes: Tensor,
    ) -> None:
        for name, value in source.items():
            index = source_row
            for _ in value.shape[2:]:
                index = index.unsqueeze(-1)
            index = index.expand((value.shape[0], source_row.shape[1], *value.shape[2:]))
            destination[name].index_copy_(0, lanes, value.gather(1, index))

    def transition(self, request: TransitionRequest) -> CommitPlan:
        batch = request.batch
        lane_count, rows = batch.lane_count, batch.rows
        flat_input = batch.fields["model_input"].reshape(
            -1, self.geometry.direct_model_input_width
        )
        flat_private = batch.fields["sampled_private_state"].reshape(
            -1, self.geometry.players
        )
        state = self.machine.state_from_model_input(flat_input, flat_private)
        log_reach, history = self.machine.model_context(flat_input)
        action = request.sample.values["action"].reshape(-1)
        raw = MappingProxyType(
            {
                name: value
                for name, value in request.model.values.items()
                if name != "value"
            }
        )
        selected_log = self.policy.selected_log_pi(
            raw, self.machine.legality(state), action
        ).detach().float()
        actor = state.to_act.to(torch.int64)
        actor_reach = log_reach.gather(
            1,
            actor[:, None, None].expand(-1, 1, self.geometry.private_hands),
        ).squeeze(1)
        log_reach = log_reach.scatter(
            1,
            actor[:, None, None].expand(-1, 1, self.geometry.private_hands),
            (actor_reach + selected_log)[:, None],
        )
        history = self.machine.append_player_history(
            state, action, history
        )
        next_state = self.machine.step_player(state, action)
        logical = batch.fields["logical_rng_key"]
        root = logical[..., 0].reshape(-1)
        path = logical[..., 1].reshape(-1)
        retry = self.owners.retry_epoch()[batch.lanes, None].expand(
            lane_count, rows
        ).reshape(-1)
        for generation in range(3):
            chance_active = self.machine.needs_chance(next_state)
            next_state = self.machine.deal_next_street_keyed(
                next_state,
                seed=self.seed,
                iteration=self.iteration,
                root=root,
                path=path,
                draw=generation,
                retry_epoch=retry,
            )
            history = self.machine.append_chance_history(
                next_state, chance_active, history
            )
        board = next_state.board_cards
        cards = self.machine.combo_cards(
            torch.arange(
                self.geometry.private_hands,
                dtype=torch.int32,
                device=state.device,
            )
        )
        conflict = (
            board.ne(PAD_CARD)[..., None, None]
            & board.long()[..., None, None].eq(cards[None, :, :])
        ).any(1).any(-1)
        log_reach = log_reach.masked_fill(
            conflict[:, None, :], -torch.inf
        )
        terminal = next_state.terminal.reshape(lane_count, rows)
        next_family = self.machine.input_type(next_state).reshape(
            lane_count, rows
        )
        route_family = torch.where(
            terminal,
            torch.full_like(next_family, -1),
            next_family,
        )
        route = stable_route(
            route_family,
            torch.ones_like(route_family, dtype=torch.bool),
            self._route_family_values,
            declared_values=self._route_families,
        )

        n_replica = self.owners.n_replica
        full_counts = torch.zeros(
            (n_replica,), dtype=torch.int64, device=state.device
        )
        full_counts.index_fill_(0, batch.lanes, rows)
        address = torch.full(
            (n_replica,),
            torch.iinfo(torch.int64).max,
            dtype=torch.int64,
            device=state.device,
        )
        address.index_copy_(0, batch.lanes, logical[..., 0].amin(1))
        destinations: list[DeviceDestinationRequest] = []

        leaf = next(
            slab for slab in self.owners.slabs if slab.schema.name == "Leaf"
        )
        leaf_values = self._defaults(leaf, rows)
        leaf_source = self.machine.sampled_leaf(
            replace(
                next_state,
                terminal=torch.ones_like(next_state.terminal),
            )
        )
        reshaped_leaf = {
            name: value.reshape(lane_count, rows, *value.shape[1:])
            for name, value in leaf_source.items()
        }
        self._gather_route(
            leaf_values, reshaped_leaf, route.source_row[:, 0], batch.lanes
        )
        leaf_counts = torch.zeros_like(full_counts)
        leaf_counts.index_copy_(0, batch.lanes, route.counts[:, 0])
        destinations.append(
            DeviceDestinationRequest(
                leaf, leaf_counts, leaf_values, batch.input_type, address
            )
        )

        sigma = next(
            slab
            for slab in self.owners.slabs
            if slab.schema.name == f"SigmaHead.{batch.input_type}"
        )
        sigma_values = self._defaults(sigma, rows)
        value = request.model.values["value"]
        owner = batch.fields["owner"].reshape(-1).to(torch.int64)
        private = flat_private
        flat_row = torch.arange(owner.numel(), device=state.device)
        actor_hand = private[flat_row, owner].to(torch.int64)
        selected_value = value[flat_row, actor_hand].reshape(
            lane_count, rows, 1
        )
        sigma_values["old_value"].index_copy_(
            0, batch.lanes, selected_value
        )
        destinations.append(
            DeviceDestinationRequest(
                sigma, full_counts, sigma_values, batch.input_type, address
            )
        )

        action_head = next(
            slab
            for slab in self.owners.slabs
            if slab.schema.name == f"ActionHead.{batch.input_type}"
        )
        action_values = self._defaults(action_head, rows)
        action_values["action"].index_copy_(
            0, batch.lanes, request.sample.values["action"]
        )
        action_values["old_log_policy"].index_copy_(
            0,
            batch.lanes,
            request.sample.values["selected_log_policy"][..., None],
        )
        destinations.append(
            DeviceDestinationRequest(
                action_head,
                full_counts,
                action_values,
                batch.input_type,
                address,
            )
        )

        next_path = child_path_keys(path, 1).reshape(lane_count, rows)
        next_input = self.machine.direct_model_input(
            next_state, log_reach, history
        ).reshape(
            lane_count, rows, -1
        )
        next_private = next_state.combo_ids.reshape(
            lane_count, rows, self.geometry.players
        )
        next_owner = next_state.to_act.reshape(lane_count, rows)
        next_logical = torch.stack(
            (root.reshape(lane_count, rows), next_path), -1
        )
        family_source = {
            "logical_rng_key": next_logical,
            "true_profile": next_private,
            "model_input": next_input,
            "owner": next_owner,
            "sampled_private_state": next_private,
        }
        published_queues: list[SemanticFamilyFIFO] = []
        for family_index, (name, queue) in enumerate(self.queues.items()):
            values = self._defaults(queue.family, rows)
            self._gather_route(
                values,
                family_source,
                route.source_row[:, family_index + 1],
                batch.lanes,
            )
            counts = torch.zeros_like(full_counts)
            counts.index_copy_(
                0, batch.lanes, route.counts[:, family_index + 1]
            )
            destinations.append(
                DeviceDestinationRequest(
                    queue.family, counts, values, name, address
                )
            )
            published_queues.append(queue)

        extent = 3 * rows
        valid = torch.zeros(
            (n_replica, extent), dtype=torch.bool, device=state.device
        )
        source_ordinal = torch.zeros(
            (n_replica, extent), dtype=torch.int64, device=state.device
        )
        destination = torch.zeros_like(source_ordinal)
        destination_ordinal = torch.zeros_like(source_ordinal)
        ordinal = torch.arange(rows, dtype=torch.int64, device=state.device)
        for local_lane, lane in ((0, 0),):
            valid[lane].fill_(True)
            source_ordinal[lane, 0::3].copy_(ordinal)
            source_ordinal[lane, 1::3].copy_(ordinal)
            source_ordinal[lane, 2::3].copy_(ordinal)
            destination[lane, 0::3].fill_(1)
            destination[lane, 1::3].fill_(2)
            destination_ordinal[lane, 0::3].copy_(ordinal)
            destination_ordinal[lane, 1::3].copy_(ordinal)
            selected_family = route_family[local_lane]
            route_ordinal = torch.zeros_like(ordinal)
            route_destination = torch.zeros_like(ordinal)
            for family_value in (-1, *range(len(self.queues))):
                selected = selected_family.eq(family_value)
                route_ordinal = torch.where(
                    selected,
                    selected.cumsum(0, dtype=torch.int64) - 1,
                    route_ordinal,
                )
                request_index = 0 if family_value == -1 else 3 + family_value
                route_destination = torch.where(
                    selected,
                    torch.full_like(route_destination, request_index),
                    route_destination,
                )
            destination[lane, 2::3].copy_(route_destination)
            destination_ordinal[lane, 2::3].copy_(route_ordinal)

        self.last_trace = PokerTransitionTrace(
            batch.input_type,
            logical.detach(),
            next_logical.detach(),
            next_family.detach(),
            terminal.detach(),
            batch.fields["owner"].detach(),
            request.sample.values["action"].detach(),
            batch.row.detach(),
        )
        return CommitPlan(
            tuple(destinations),
            CommitCoverage(
                valid, source_ordinal, destination, destination_ordinal
            ),
            published_queues=tuple(published_queues),
        )


class PokerLearnerPolicy:
    """Reconstruct public legality and invoke the canonical policy operation."""

    def __init__(
        self,
        geometry: NLHEGeometry,
        policy: PokerPolicyOperator | None = None,
    ) -> None:
        self.machine = PokerStateMachine(geometry)
        self.policy = PokerPolicyOperator() if policy is None else policy

    def __call__(
        self,
        _input_type: str,
        model_input: Tensor,
        raw: Mapping[str, Tensor],
        action: Tensor,
        sampled_private_state: Tensor,
        owner: Tensor,
        reference_raw: Mapping[str, Tensor] | None,
    ) -> PolicyEvaluation:
        del owner
        state = self.machine.state_from_model_input(
            model_input, sampled_private_state.to(torch.int32)
        )
        return self.policy.learner(
            raw,
            self.machine.legality(state),
            action,
            sampled_private_state,
            state.to_act.to(torch.int64),
            reference_raw,
        )


class SampledShowdownBinding:
    """Exact evaluator contract-v2 ``Leaf -> LayerWork`` hook."""

    def __init__(
        self,
        geometry: NLHEGeometry,
        layer_work: MonotoneSlab,
        *,
        n_replica: int,
        pack_cap: int,
    ) -> None:
        try:
            import poker_gpu_evaluator as poker
        except ImportError as exc:
            raise QueuedPokerError("poker-gpu-evaluator is required") from exc
        if str(getattr(poker, "CONTRACT_VERSION", "")) != "2":
            raise QueuedPokerError("poker evaluator contract v2 is required")
        self.evaluator = poker.bind(rank_count=geometry.rank_count)
        info = self.evaluator.info
        if str(info.contract_version) != "2" or int(info.rank_count) != geometry.rank_count:
            raise QueuedPokerError("bound evaluator identity differs from NLHE geometry")
        if layer_work.schema.name != "LayerWork.all":
            raise QueuedPokerError("sampled showdown requires LayerWork.all")
        self.geometry = geometry
        self.layer_work = layer_work
        self.n_replica = n_replica
        self.pack_cap = pack_cap

    def leaf_to_layer_work(self, batch: CompletionBatch) -> CompletionPlan:
        fields = batch.fields
        payoff = self.evaluator.evaluate_showdowns(
            board_cards=fields["board_cards"].reshape(-1, 5).contiguous(),
            combo_ids=fields["combo_ids"].reshape(
                -1, self.geometry.players
            ).contiguous(),
            stratum_level=fields["stratum_level"].reshape(
                -1, self.geometry.players
            ).contiguous(),
            payoff_baseline=fields["payoff_baseline"].reshape(
                -1, self.geometry.players
            ).contiguous(),
            stratum_amount=fields["stratum_amount"].reshape(
                -1, self.geometry.players
            ).contiguous(),
        ).reshape(batch.lanes.numel(), batch.rows, self.geometry.players)
        values = torch.zeros(
            (
                self.n_replica,
                self.pack_cap,
                2,
                self.geometry.players,
                1,
            ),
            dtype=torch.float32,
            device=payoff.device,
        )
        values[batch.lanes, : batch.rows, 0, :, 0] = payoff
        counts = torch.zeros(
            self.n_replica, dtype=torch.int64, device=payoff.device
        )
        counts[batch.lanes] = batch.rows
        valid = torch.zeros(
            (self.n_replica, self.pack_cap),
            dtype=torch.bool,
            device=payoff.device,
        )
        valid[batch.lanes, : batch.rows] = True
        ordinal = torch.arange(
            self.pack_cap, dtype=torch.int64, device=payoff.device
        )[None, :].expand(self.n_replica, -1)
        source = ordinal.clone()
        destination = torch.zeros_like(ordinal)
        coverage = CommitCoverage(valid, source, destination, ordinal)
        return CompletionPlan(
            (
                DeviceDestinationRequest(
                    self.layer_work,
                    counts,
                    values={"return_value": values},
                    input_type="all",
                    logical_address=torch.where(
                        counts > 0,
                        batch.row[:, 0],
                        torch.full_like(counts, torch.iinfo(torch.int64).max),
                    ),
                ),
            ),
            coverage,
        )


def validate_phase5_mode(schema: ModeSchema) -> None:
    if (
        schema.training_regime is not TrainingRegime.SELF_PLAY
        or schema.rollout_mode is not RolloutMode.SEQUENTIAL
        or schema.private_estimator is not PrivateEstimator.SAMPLED
    ):
        raise ValueError("Phase 5 poker hooks require self-play sequential sampled")


__all__ = [
    "ALL_IN",
    "FOLDED",
    "LIVE",
    "NLHEGeometry",
    "PAD_CARD",
    "PokerForwardInput",
    "PokerHistory",
    "PokerLearnerPolicy",
    "PokerState",
    "PokerStateMachine",
    "PokerTransitionTrace",
    "QueuedPokerError",
    "SampledShowdownBinding",
    "SequentialSampledPokerHooks",
    "sampled_partitions",
    "validate_phase5_mode",
]
