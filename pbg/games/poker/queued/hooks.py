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
    ConversionSpec,
    ModeSchema,
    Partition,
    PartitionContract,
    PartitionMap,
    PrivateEstimator,
    RolloutMode,
    TrainingRegime,
    mode_schema,
)
from pbg.queued.rollout import (
    CommitCoverage,
    CommitPlan,
    CompletionBatch,
    CompletionPlan,
    DrawPurpose,
    DeferredUnaryTopologyPublication,
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
        """Width of the sealed family row, including folded statistics."""

        return self.base_model_input_width + self.folded_summary_width

    @property
    def base_model_input_width(self) -> int:
        return (
            8
            + 10 * self.players
            + self.players * self.private_hands
            + 10 * max_public_history_tokens(self.players, self.stack_bb)
        )

    @property
    def folded_summary_width(self) -> int:
        if self.players == 2:
            # A heads-up fold is terminal, so no later player node can consume
            # folded context.
            return 0
        cards = self.deck_cards
        return (
            self.private_hands
            + cards
            + cards * cards
            + self.players
            + self.players * cards
            + 2
        )

    def combo_id(self, first: Tensor, second: Tensor) -> Tensor:
        low = torch.minimum(first, second).to(torch.int64) - self.first_card
        high = torch.maximum(first, second).to(torch.int64) - self.first_card
        return (
            low * (2 * self.deck_cards - low - 1) // 2 + high - low - 1
        ).to(torch.int32)


def sampled_partitions(
    players: int, geometry: NLHEGeometry | None = None
) -> PartitionContract:
    if not 2 <= players <= 6:
        raise ValueError("players must lie in [2,6]")
    inputs = (Partition("pre", 1),) + tuple(
        Partition(f"post_{live}", 1) for live in range(2, players + 1)
    )
    return PartitionContract(
        inputs=inputs,
        outputs=(Partition("all", 1),),
        folded=(
            Partition(
                "pre", 1 if geometry is None else geometry.rank_count**2
            ),
            Partition(
                "post", 1 if geometry is None else geometry.private_hands
            ),
        ),
        input_to_output=tuple(
            PartitionMap(partition.name, "all") for partition in inputs
        ),
        conversions=(),
    )


def rb_partitions(geometry: NLHEGeometry) -> PartitionContract:
    """Natural learner domains for sequential Rao--Blackwell poker."""

    k = geometry.rank_count**2
    h = geometry.private_hands
    inputs = (Partition("pre", k),) + tuple(
        Partition(f"post_{live}", h)
        for live in range(2, geometry.players + 1)
    )
    return PartitionContract(
        inputs=inputs,
        outputs=(Partition("pre", k), Partition("post", h)),
        folded=(Partition("pre", k), Partition("post", h)),
        input_to_output=(
            PartitionMap("pre", "pre"),
            *(
                PartitionMap(f"post_{live}", "post")
                for live in range(2, geometry.players + 1)
            ),
        ),
        conversions=(ConversionSpec("post", "pre", "sum_classes"),),
    )


def poker_partitions(
    geometry: NLHEGeometry, schema: ModeSchema
) -> PartitionContract:
    if schema.private_estimator is PrivateEstimator.RAO_BLACKWELL:
        return rb_partitions(geometry)
    return sampled_partitions(geometry.players, geometry)


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
    can_raise: Tensor
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
            "can_raise": (self.can_raise, torch.bool, (batch, players)),
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
        logical_base: int = 0,
        device: torch.device | str,
    ) -> PokerState:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("queued poker roots require CUDA")
        if root_count <= 0:
            raise ValueError("root_count must be positive")
        roots = (
            torch.arange(root_count, dtype=torch.int64, device=device)
            + int(logical_base)
        )
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
            can_raise=pending.clone(),
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
        can_raise = state.can_raise.gather(1, actor[:, None]).squeeze(1)
        mask[:, RAISE_OFFSET:] = (
            (full_raise | short_all_in)
            & multiple_actors[:, None]
            & can_raise[:, None]
        )
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
                state.can_raise.float(),
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
        base = torch.cat(
            (public, log_reach.flatten(1), *history_values), 1
        )
        summary = torch.zeros(
            (state.rows, self.geometry.folded_summary_width),
            dtype=base.dtype,
            device=base.device,
        )
        return torch.cat((base, summary), 1).contiguous()

    def model_context(
        self, model_input: Tensor
    ) -> tuple[Tensor, PokerHistory]:
        players = self.geometry.players
        hands = self.geometry.private_hands
        tokens = max_public_history_tokens(
            players, self.geometry.stack_bb
        )
        offset = 8 + 10 * players
        reach = model_input[:, offset : offset + players * hands].float()
        offset += players * hands
        columns = []
        for _ in range(10):
            columns.append(model_input[:, offset : offset + tokens])
            offset += tokens
        if offset != self.geometry.base_model_input_width:
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
                # The separator belongs to the street being entered, while
                # every public arithmetic field is captured before the
                # transition resets street commitments and the current bet.
                "street": state.street + 1,
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
        can_raise = take(players).gt(0.5)
        if offset != 8 + 10 * players:
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
            can_raise,
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
        can_raise = state.can_raise.clone()
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
        # A full raise reopens raising for every live player.  A short all-in
        # does not reopen raising, but every live player still below the new
        # current bet must act again to call or fold.
        can_raise = torch.where(full_raise[:, None] & active, active, can_raise)
        can_raise.scatter_(
            1,
            actor_index,
            torch.zeros_like(actor_index, dtype=torch.bool),
        )
        pending = torch.where(full_raise[:, None] & active, active, pending)
        owes_call = active & committed_street.lt(current_bet[:, None])
        pending |= owes_call
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
            can_raise,
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
        can_raise = torch.where(needs[:, None], next_pending, state.can_raise)
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
            can_raise,
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
            contenders,
            state.combo_ids,
            torch.full_like(state.combo_ids, -1),
        )
        combo = torch.where(
            early[:, None], torch.full_like(combo, -1), combo
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

    def rb_leaf(
        self,
        state: PokerState,
        log_reach: Tensor,
        *,
        seed: Tensor,
        selected_player: Tensor | None = None,
    ) -> Mapping[str, Tensor]:
        """Capture evaluator-v2 completion inputs in exact-hand space."""

        if (
            log_reach.shape
            != (state.rows, self.geometry.players, self.geometry.private_hands)
            or log_reach.dtype != torch.float32
            or log_reach.device != state.device
        ):
            raise QueuedPokerError("RB leaf reaches must be CUDA FP32 [B,P,H]")
        sampled = self.sampled_leaf(
            replace(state, terminal=torch.ones_like(state.terminal))
        )
        finite = torch.isfinite(log_reach)
        gauge = log_reach.masked_fill(~finite, -torch.inf).amax(
            -1, keepdim=True
        )
        range_weights = torch.exp(
            log_reach - gauge.masked_fill(~torch.isfinite(gauge), 0.0)
        ).masked_fill(~finite, 0.0)
        result: dict[str, Tensor] = {
            # RB completion retains the public pre-terminal board.  Unlike
            # sampled showdown, an early fold must not be padded with invented
            # cards because that would change its pre/post output partition.
            "board_cards": state.board_cards.contiguous(),
            "stratum_level": sampled["stratum_level"],
            "payoff_baseline": sampled["payoff_baseline"],
            "stratum_amount": sampled["stratum_amount"],
            "range_weights": range_weights.contiguous(),
            "seed": seed.to(dtype=torch.int64, device=state.device).contiguous(),
        }
        if selected_player is not None:
            result["selected_player"] = selected_player.to(
                dtype=torch.int32, device=state.device
            ).contiguous()
        return MappingProxyType(result)


@dataclass(frozen=True)
class PokerForwardInput:
    """Direct flattened model arguments for one stable family prefix."""

    input_type: str
    model_input: Tensor
    sampled_private_state: Tensor
    owner: Tensor
    role_selector: Tensor | None = None


class SequentialPokerHooks:
    """NLHE implementation shared by every sequential mode.

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
        mode: ModeSchema | None = None,
        policy: PokerPolicyOperator | None = None,
        probe_cutoff: bool = False,
    ) -> None:
        if owners.n_replica != 1:
            raise ValueError("ordinary Phase 5 poker hooks require one replica")
        self.mode = (
            mode_schema("self_play", "sequential", "sampled")
            if mode is None
            else mode
        )
        if self.mode.rollout_mode is not RolloutMode.SEQUENTIAL:
            raise ValueError("sequential poker hooks require sequential mode")
        expected = tuple(
            item.name for item in poker_partitions(geometry, self.mode).inputs
        )
        if tuple(queues) != expected:
            raise ValueError("poker queues do not cover the sampled input preorder")
        self.geometry = geometry
        self.machine = PokerStateMachine(geometry)
        self.owners = owners
        self.queues = MappingProxyType(dict(queues))
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.policy = PokerPolicyOperator() if policy is None else policy
        if type(probe_cutoff) is not bool:
            raise TypeError("probe_cutoff must be bool")
        if probe_cutoff and (
            self.mode.training_regime is not TrainingRegime.BEST_RESPONSE
            or self.mode.private_estimator is not PrivateEstimator.RAO_BLACKWELL
        ):
            raise ValueError("probe cutoff requires sequential BR/RB")
        self.probe_cutoff = probe_cutoff
        self.input_index = {name: index for index, name in enumerate(expected)}
        self._route_family_values = (-1, *range(len(self.queues)))
        self._route_families = torch.tensor(
            self._route_family_values,
            dtype=torch.int32,
            device=self.owners.device,
        )
        self.last_topology: DeferredUnaryTopologyPublication | None = None
        self._topology_sequence = 0

    def _combo_class(self) -> Tensor:
        ranks = self.geometry.rank_count
        cards = self.machine.combo_cards(
            torch.arange(
                self.geometry.private_hands,
                dtype=torch.int32,
                device=self.owners.device,
            )
        ).to(torch.int64)
        rank = cards // 4 - (13 - ranks)
        high = rank.max(1).values
        low = rank.min(1).values
        high_position = ranks - 1 - high
        low_position = ranks - 1 - low
        suited_or_pair = rank[:, 0].eq(rank[:, 1]) | cards[:, 0].remainder(
            4
        ).eq(cards[:, 1].remainder(4))
        return torch.where(
            suited_or_pair,
            high_position * ranks + low_position,
            low_position * ranks + high_position,
        )

    def _owner_slab(self, owner: str, partition: str) -> MonotoneSlab:
        name = f"{owner}.{partition}"
        try:
            return next(
                slab for slab in self.owners.slabs if slab.schema.name == name
            )
        except StopIteration as exc:
            raise QueuedPokerError(f"missing retained owner {name}") from exc

    def _resolved_model_input(self, batch: FamilyBatch) -> Tensor:
        """Refresh folded reaches from the sole retained FoldedRanges owner."""

        model_input = batch.fields["model_input"].reshape(
            -1, self.geometry.direct_model_input_width
        )
        references = batch.fields["folded_range_row"].reshape(
            -1, self.geometry.players
        ).to(torch.int64)
        refreshed = model_input.clone()
        log_reach, _ = self.machine.model_context(refreshed)
        log_reach = log_reach.clone()
        rows = torch.arange(refreshed.shape[0], device=refreshed.device)
        classes = self._combo_class()
        class_count = torch.bincount(
            classes, minlength=self.geometry.rank_count**2
        ).float()
        state = self.machine.state_from_model_input(
            refreshed,
            torch.zeros(
                (refreshed.shape[0], self.geometry.players),
                dtype=torch.int32,
                device=refreshed.device,
            ),
        )
        combo_cards = self.machine.combo_cards(
            torch.arange(
                self.geometry.private_hands,
                dtype=torch.int32,
                device=refreshed.device,
            )
        )
        board_conflict = (
            state.board_cards.ne(PAD_CARD)[..., None, None]
            & state.board_cards.long()[..., None, None].eq(
                combo_cards[None]
            )
        ).any(1).any(-1)
        pre = self._owner_slab("FoldedRanges", "pre").columns["range"][0]
        post = self._owner_slab("FoldedRanges", "post").columns["range"][0]
        pre_log_mass = self._owner_slab(
            "FoldedRanges", "pre"
        ).columns["log_mass"][0]
        post_log_mass = self._owner_slab(
            "FoldedRanges", "post"
        ).columns["log_mass"][0]
        for seat in range(self.geometry.players):
            reference = references[:, seat]
            pre_mask = reference.ge(0)
            post_mask = reference.lt(-1)
            safe_pre = reference.clamp_min(0)
            safe_post = (-reference - 2).clamp_min(0)
            if pre.shape[1] == 1:
                pre_weight = pre.index_select(0, safe_pre).expand(
                    -1, self.geometry.private_hands
                ) / self.geometry.private_hands
            else:
                pre_weight = pre.index_select(0, safe_pre).index_select(
                    1, classes
                )
                pre_weight = pre_weight / class_count.index_select(
                    0, classes
                )[None]
            if post.shape[1] == 1:
                post_weight = post.index_select(0, safe_post).expand(
                    -1, self.geometry.private_hands
                ) / self.geometry.private_hands
            else:
                post_weight = post.index_select(0, safe_post)
            weight = torch.where(pre_mask[:, None], pre_weight, post_weight)
            log_mass = torch.where(
                pre_mask,
                pre_log_mass.index_select(0, safe_pre),
                post_log_mass.index_select(0, safe_post),
            )
            weight = weight * torch.exp(log_mass)[:, None]
            weight = weight.masked_fill(board_conflict, 0.0)
            active = pre_mask | post_mask
            weight = weight.masked_fill(~active[:, None], 0.0)
            log_weight = torch.log(
                weight.clamp_min(torch.finfo(torch.float32).tiny)
            ).masked_fill(~weight.gt(0), -torch.inf)
            log_reach[rows, seat] = torch.where(
                active[:, None], log_weight, log_reach[rows, seat]
            )
        offset = 8 + 10 * self.geometry.players
        refreshed[
            :, offset : offset + self.geometry.players * self.geometry.private_hands
        ] = log_reach.flatten(1).to(refreshed.dtype)
        return refreshed

    def _forward_model_input(self, batch: FamilyBatch) -> Tensor:
        return batch.fields["model_input"].reshape(
            -1, self.geometry.direct_model_input_width
        )

    def model_input(self, batch: FamilyBatch) -> PokerForwardInput:
        shape = (batch.lane_count * batch.rows,)
        return PokerForwardInput(
            batch.input_type,
            self._forward_model_input(batch).reshape(
                *shape, self.geometry.direct_model_input_width
            ),
            batch.fields.get(
                "sampled_private_state", batch.fields["true_profile"]
            ).reshape(
                *shape, self.geometry.players
            ),
            batch.fields["owner"].reshape(*shape),
            (
                batch.fields["role_selector"].reshape(*shape)
                if "role_selector" in batch.fields
                else None
            ),
        )

    def sample(self, request: SampleRequest) -> SampleWorkspace:
        forward = PokerForwardInput(
            request.batch.input_type,
            self._forward_model_input(request.batch).reshape(
                -1, self.geometry.direct_model_input_width
            ),
            request.batch.fields.get(
                "sampled_private_state", request.batch.fields["true_profile"]
            ).reshape(
                -1, self.geometry.players
            ),
            request.batch.fields["owner"].reshape(-1),
            (
                request.batch.fields["role_selector"].reshape(-1)
                if "role_selector" in request.batch.fields
                else None
            ),
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

    def _natural_rows(self, value: Tensor, input_type: str) -> Tensor:
        """Select one deterministic exact-hand representative per pre class."""

        if input_type != "pre":
            return value.float()
        ranks = self.geometry.rank_count
        cards = self.machine.combo_cards(
            torch.arange(
                self.geometry.private_hands,
                dtype=torch.int32,
                device=value.device,
            )
        )
        rank = cards.long() // 4 - (13 - ranks)
        high, low = rank.max(1).values, rank.min(1).values
        hp, lp = ranks - 1 - high, ranks - 1 - low
        suited_or_pair = rank[:, 0].eq(rank[:, 1]) | cards[:, 0].remainder(4).eq(
            cards[:, 1].remainder(4)
        )
        classes = torch.where(
            suited_or_pair, hp * ranks + lp, lp * ranks + hp
        )
        representatives = torch.full(
            (ranks * ranks,), -1, dtype=torch.int64, device=value.device
        )
        # Reverse assignment leaves the smallest exact-hand row in each class.
        for combo in range(self.geometry.private_hands - 1, -1, -1):
            representatives[classes[combo]] = combo
        torch._assert_async(
            representatives.ge(0).all(), "preflop class map is incomplete"
        )
        return value.index_select(1, representatives).float()

    def _tight_folder_mask(
        self, history: PokerHistory, status: Tensor
    ) -> Tensor:
        """Classify acted-then-folded seats; instant preflop folds are loose."""

        tight = torch.zeros_like(status, dtype=torch.bool)
        for seat in range(self.geometry.players):
            owned = history.mask & history.actor.eq(seat)
            voluntarily_invested = owned & (
                (
                    history.action.eq(CHECK_CALL)
                    & history.amount.gt(0)
                )
                | history.action.ge(RAISE_OFFSET)
            )
            tight[:, seat] = (
                status[:, seat].eq(FOLDED) & voluntarily_invested.any(-1)
            )
        return tight

    def transition(self, request: TransitionRequest) -> CommitPlan:
        batch = request.batch
        lane_count, rows = batch.lane_count, batch.rows
        flat_input = self._forward_model_input(batch)
        flat_private = batch.fields.get(
            "sampled_private_state", batch.fields["true_profile"]
        ).reshape(
            -1, self.geometry.players
        )
        state = self.machine.state_from_model_input(flat_input, flat_private)
        resolved_input = self._resolved_model_input(batch)
        log_reach, history = self.machine.model_context(resolved_input)
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
        fold_log_reach = log_reach
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
            history = self.machine.append_chance_history(
                next_state, chance_active, history
            )
            next_state = self.machine.deal_next_street_keyed(
                next_state,
                seed=self.seed,
                iteration=self.iteration,
                root=root,
                path=path,
                draw=generation,
                retry_epoch=retry,
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
        sampled_mode = (
            self.mode.private_estimator is PrivateEstimator.SAMPLED
        )
        response = self.mode.training_regime is TrainingRegime.BEST_RESPONSE
        owner = batch.fields["owner"].reshape(-1).to(torch.int64)
        selected_player = (
            batch.fields["role_selector"].reshape(-1).to(torch.int64)
            if response
            else owner
        )
        probe_cutoff = (
            action.eq(FOLD)
            & actor.eq(selected_player)
            & self.probe_cutoff
        )
        terminal = (
            next_state.terminal | probe_cutoff
        ).reshape(lane_count, rows)
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
        trained_flat = (
            owner.eq(selected_player)
            if response
            else torch.ones_like(owner, dtype=torch.bool)
        )
        trained = trained_flat.reshape(lane_count, rows)
        trained_order = torch.argsort(~trained, dim=1, stable=True)
        trained_counts = torch.zeros_like(full_counts)
        trained_counts.index_copy_(
            0, batch.lanes, trained.sum(1, dtype=torch.int64)
        )

        leaf = next(
            slab for slab in self.owners.slabs if slab.schema.name == "Leaf"
        )
        leaf_values = self._defaults(leaf, rows)
        terminal_state = replace(
            next_state, terminal=torch.ones_like(next_state.terminal)
        )
        leaf_source = (
            self.machine.sampled_leaf(terminal_state)
            if sampled_mode
            else self.machine.rb_leaf(
                terminal_state,
                log_reach,
                seed=logical_rng_keys(
                    seed=self.seed,
                    iteration=self.iteration,
                    root=root,
                    path=path,
                    draw=0,
                    purpose=DrawPurpose.COMPLETION,
                    retry_epoch=retry,
                ),
                selected_player=selected_player if response else None,
            )
        )
        if sampled_mode and response:
            leaf_source = MappingProxyType(
                {
                    **dict(leaf_source),
                    "selected_player": selected_player.to(torch.int32),
                }
            )
        if response:
            leaf_values_source = dict(leaf_source)
            leaf_values_source["probe_cutoff"] = probe_cutoff.to(torch.uint8)
            if self.probe_cutoff:
                baseline = leaf_values_source["payoff_baseline"].clone()
                row_index = torch.arange(
                    baseline.shape[0], device=baseline.device
                )
                selected_committed = next_state.committed[
                    row_index, selected_player
                ]
                baseline[row_index, selected_player] = torch.where(
                    probe_cutoff,
                    -selected_committed,
                    baseline[row_index, selected_player],
                )
                leaf_values_source["payoff_baseline"] = baseline
            leaf_source = MappingProxyType(leaf_values_source)
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
        private = flat_private
        flat_row = torch.arange(owner.numel(), device=state.device)
        selected_value = (
            value[flat_row, private[flat_row, owner].to(torch.int64), None]
            if sampled_mode
            else self._natural_rows(value, batch.input_type)
        ).reshape(lane_count, rows, -1)
        selected_value = selected_value.gather(
            1,
            trained_order[..., None].expand(
                -1, -1, selected_value.shape[-1]
            ),
        )
        sigma_values["old_value"].index_copy_(
            0, batch.lanes, selected_value
        )
        destinations.append(
            DeviceDestinationRequest(
                sigma, trained_counts, sigma_values, batch.input_type, address
            )
        )

        action_head = next(
            slab
            for slab in self.owners.slabs
            if slab.schema.name == f"ActionHead.{batch.input_type}"
        )
        action_values = self._defaults(action_head, rows)
        action_values["action"].index_copy_(
            0,
            batch.lanes,
            request.sample.values["action"].gather(1, trained_order),
        )
        retained_log = (
            request.sample.values["selected_log_policy"].reshape(-1, 1)
            if sampled_mode
            else self._natural_rows(selected_log, batch.input_type)
        ).reshape(lane_count, rows, -1)
        retained_log = retained_log.gather(
            1,
            trained_order[..., None].expand(-1, -1, retained_log.shape[-1]),
        )
        action_values["old_log_policy"].index_copy_(
            0, batch.lanes, retained_log
        )
        destinations.append(
            DeviceDestinationRequest(
                action_head,
                trained_counts,
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
        }
        folded_reference = batch.fields["folded_range_row"].reshape(
            -1, self.geometry.players
        ).to(torch.int64)
        fold = action.eq(FOLD)
        finite_fold = torch.isfinite(fold_log_reach)
        fold_gauge = fold_log_reach.masked_fill(
            ~finite_fold, -torch.inf
        ).amax(-1, keepdim=True)
        fold_weight = torch.exp(
            fold_log_reach
            - fold_gauge.masked_fill(~torch.isfinite(fold_gauge), 0.0)
        ).masked_fill(~finite_fold, 0.0)
        actor_weight = fold_weight[
            torch.arange(fold_weight.shape[0], device=state.device), actor
        ]
        actor_weight /= actor_weight.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        folded_partition = "pre" if batch.input_type == "pre" else "post"
        folded_slab = self._owner_slab("FoldedRanges", folded_partition)
        fold_order = torch.argsort(~fold, stable=True)
        folded_rows = actor_weight
        if folded_partition == "pre":
            classes = self._combo_class()
            folded_rows = torch.zeros(
                (
                    actor_weight.shape[0],
                    self.geometry.rank_count**2,
                ),
                dtype=torch.float32,
                device=state.device,
            )
            folded_rows.scatter_add_(
                1,
                classes[None].expand_as(actor_weight),
                actor_weight,
            )
        if folded_slab.columns["range"].shape[-1] == 1:
            folded_rows = folded_rows.sum(-1, keepdim=True)
        folded_values = self._defaults(folded_slab, rows)
        folded_values["range"][0].copy_(
            folded_rows.index_select(0, fold_order)
        )
        actor_log_mass = torch.logsumexp(
            fold_log_reach[
                torch.arange(fold_log_reach.shape[0], device=state.device),
                actor,
            ],
            -1,
        )
        folded_values["log_mass"][0].copy_(
            actor_log_mass.index_select(0, fold_order)
        )
        fold_counts = torch.zeros(
            (n_replica,), dtype=torch.int64, device=state.device
        )
        fold_counts.index_copy_(
            0,
            batch.lanes,
            fold.reshape(lane_count, rows).sum(1, dtype=torch.int64),
        )
        fold_base = (
            folded_slab.reserved_tails_device()
            .index_select(0, batch.lanes)[:, None]
            .expand(lane_count, rows)
            .reshape(-1)
        )
        fold_ordinal = fold.cumsum(0, dtype=torch.int64) - 1
        encoded_reference = (
            fold_base + fold_ordinal
            if folded_partition == "pre"
            else -(fold_base + fold_ordinal + 2)
        )
        folded_reference = folded_reference.scatter(
            1,
            actor[:, None],
            torch.where(
                fold,
                encoded_reference,
                folded_reference.gather(1, actor[:, None]).squeeze(1),
            )[:, None],
        )
        family_source["folded_range_row"] = folded_reference.reshape(
            lane_count, rows, self.geometry.players
        ).to(torch.int32)
        refresh = fold | next_state.street.ne(state.street)
        refresh_rows = torch.nonzero(refresh, as_tuple=False).flatten()
        from pbg.games.poker.queued.folded import (
            folded_statistics,
            pack_folded_statistics,
        )

        next_input_flat = next_input.reshape(
            -1, self.geometry.direct_model_input_width
        )
        if self.geometry.folded_summary_width:
            refreshed_summary = folded_statistics(
                self.geometry,
                board_cards=next_state.board_cards.index_select(0, refresh_rows),
                range_weights=torch.exp(
                    log_reach.index_select(0, refresh_rows).float()
                ).masked_fill(
                    ~torch.isfinite(
                        log_reach.index_select(0, refresh_rows)
                    ),
                    0.0,
                ),
                status=next_state.status.index_select(0, refresh_rows),
                combo_cards=cards,
                beta_rank=self.geometry.deck_cards,
                line_dim=0,
                tight_mask=self._tight_folder_mask(
                    history, next_state.status
                ).index_select(0, refresh_rows),
            )
            packed_summary = pack_folded_statistics(
                self.geometry, refreshed_summary
            )
            next_input_flat[
                :, self.geometry.base_model_input_width :
            ].copy_(
                flat_input[:, self.geometry.base_model_input_width :]
            )
            next_input_flat[
                :, self.geometry.base_model_input_width :
            ].index_copy_(0, refresh_rows, packed_summary.to(next_input.dtype))
        if sampled_mode:
            family_source["sampled_private_state"] = next_private
        if response:
            family_source["role_selector"] = batch.fields["role_selector"]
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

        folded_destination = len(destinations)
        destinations.append(
            DeviceDestinationRequest(
                folded_slab,
                fold_counts,
                folded_values,
                batch.input_type,
                address,
            )
        )

        extent = 4 * rows
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
            valid[lane, 2::4].fill_(True)
            valid[lane, 0::4].copy_(trained[local_lane])
            valid[lane, 1::4].copy_(trained[local_lane])
            valid[lane, 3::4].copy_(fold.reshape(lane_count, rows)[local_lane])
            source_ordinal[lane, 0::4].copy_(ordinal)
            source_ordinal[lane, 1::4].copy_(ordinal)
            source_ordinal[lane, 2::4].copy_(ordinal)
            source_ordinal[lane, 3::4].copy_(ordinal)
            destination[lane, 0::4].fill_(1)
            destination[lane, 1::4].fill_(2)
            destination[lane, 3::4].fill_(folded_destination)
            compact_ordinal = trained[local_lane].cumsum(
                0, dtype=torch.int64
            ) - 1
            destination_ordinal[lane, 0::4].copy_(
                compact_ordinal.clamp_min(0)
            )
            destination_ordinal[lane, 1::4].copy_(
                compact_ordinal.clamp_min(0)
            )
            destination_ordinal[lane, 3::4].copy_(
                fold.reshape(lane_count, rows)[local_lane].cumsum(
                    0, dtype=torch.int64
                ).sub(1).clamp_min(0)
            )
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
            destination[lane, 2::4].copy_(route_destination)
            destination_ordinal[lane, 2::4].copy_(route_ordinal)

        topology_next_family = next_family[0]
        if not sampled_mode:
            leaf_family = torch.where(
                next_state.board_cards.reshape(
                    lane_count, rows, 5
                )[0, :, 0].eq(PAD_CARD),
                torch.zeros_like(topology_next_family),
                torch.ones_like(topology_next_family),
            )
            topology_next_family = torch.where(
                terminal[0], leaf_family, topology_next_family
            )
        topology = DeferredUnaryTopologyPublication(
            lane=0,
            sequence=self._topology_sequence,
            input_type_index=self.input_index[batch.input_type],
            logical_rng_key=logical[0],
            next_logical_rng_key=next_logical[0],
            next_input_type=topology_next_family,
            terminal=terminal[0],
            owner=(
                torch.zeros_like(batch.fields["owner"][0])
                if response
                else batch.fields["owner"][0]
            ),
            action=request.sample.values["action"][0],
            familywork_row=batch.row[0],
            leaf_reservation_index=0,
            leaf_target_ordinal=route_ordinal,
            trained=trained[0],
            sigma_reservation_index=1,
            action_reservation_index=2,
            trained_target_ordinal=compact_ordinal.clamp_min(0),
        )
        self._topology_sequence += 1
        self.last_topology = topology
        return CommitPlan(
            tuple(destinations),
            CommitCoverage(
                valid, source_ordinal, destination, destination_ordinal
            ),
            published_queues=tuple(published_queues),
            deferred_topology=(topology,),
        )


# Phase 5 imports retain their public spelling while using the shared engine.
SequentialSampledPokerHooks = SequentialPokerHooks


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
        if sampled_private_state.shape != (
            model_input.shape[0],
            self.machine.geometry.players,
        ):
            sampled_private_state = torch.zeros(
                (model_input.shape[0], self.machine.geometry.players),
                dtype=torch.int32,
                device=model_input.device,
            )
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
        utility: str = "raw_bb",
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
        if utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("showdown utility must be raw_bb or stack_normalized")
        self.utility_scale = (
            1.0 if utility == "raw_bb" else 1.0 / geometry.stack_bb
        )

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
        ).reshape(
            batch.lanes.numel(), batch.rows, self.geometry.players
        ).mul(self.utility_scale)
        p_train = self.layer_work.schema.by_name[
            "return_value"
        ].shape[1]
        if p_train == 1:
            if "selected_player" not in fields:
                raise QueuedPokerError(
                    "best-response sampled leaf omitted selected_player"
                )
            selected = fields["selected_player"].reshape(
                batch.lanes.numel(), batch.rows
            ).to(torch.int64)
            payoff = payoff.gather(2, selected[..., None])
        elif p_train != self.geometry.players:
            raise QueuedPokerError("sampled LayerWork has invalid trained width")
        values = torch.zeros(
            (
                self.n_replica,
                self.pack_cap,
                2,
                p_train,
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


class RBCompletionBinding:
    """Evaluator-v2 completion followed by FP32 mass/moment publication."""

    def __init__(
        self,
        geometry: NLHEGeometry,
        layer_work: Mapping[str, MonotoneSlab],
        algebra: object,
        *,
        mode: ModeSchema,
        n_replica: int,
        pack_cap: int,
        n_samples: int,
        utility: str = "raw_bb",
    ) -> None:
        from pbg.queued.targets import TargetAlgebra

        if mode.private_estimator is not PrivateEstimator.RAO_BLACKWELL:
            raise ValueError("RB completion requires an RB schema")
        if not isinstance(algebra, TargetAlgebra):
            raise TypeError("RB completion requires the target algebra")
        if isinstance(n_samples, bool) or n_samples <= 0:
            raise ValueError("completion samples must be positive")
        try:
            import poker_gpu_evaluator as poker
        except ImportError as exc:
            raise QueuedPokerError("poker-gpu-evaluator is required") from exc
        if str(getattr(poker, "CONTRACT_VERSION", "")) != "2":
            raise QueuedPokerError("poker evaluator contract v2 is required")
        self.evaluator = poker.bind(rank_count=geometry.rank_count)
        self.geometry = geometry
        self.layer_work = MappingProxyType(dict(layer_work))
        self.algebra = algebra
        self.mode = mode
        self.n_replica = int(n_replica)
        self.pack_cap = int(pack_cap)
        self.n_samples = int(n_samples)
        if utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("completion utility must be raw_bb or stack_normalized")
        self.utility_scale = 1.0 if utility == "raw_bb" else 1.0 / geometry.stack_bb

    def leaf_to_layer_work(self, batch: CompletionBatch) -> CompletionPlan:
        from pbg.queued.targets import rb_completion

        fields = batch.fields
        flat = batch.lanes.numel() * batch.rows
        common = {
            "board_cards": fields["board_cards"].reshape(flat, 5).contiguous(),
            "range_weights": fields["range_weights"].reshape(
                flat, self.geometry.players, self.geometry.private_hands
            ).contiguous(),
            "stratum_level": fields["stratum_level"].reshape(
                flat, self.geometry.players
            ).contiguous(),
            "payoff_baseline": fields["payoff_baseline"].reshape(
                flat, self.geometry.players
            ).contiguous(),
            "stratum_amount": fields["stratum_amount"].reshape(
                flat, -1
            ).contiguous(),
            "n_samples": self.n_samples,
            "seed": fields["seed"].reshape(flat).contiguous(),
        }
        response = self.mode.training_regime is TrainingRegime.BEST_RESPONSE
        p_train = 1 if response else self.geometry.players
        if response:
            selected = fields["selected_player"].reshape(flat).to(torch.int64)
            cutoff = fields["probe_cutoff"].reshape(flat).bool()
            ordinary_row = torch.nonzero(~cutoff, as_tuple=False).flatten()
            p_hat = torch.zeros(
                (flat, 1, self.geometry.private_hands),
                dtype=torch.float32,
                device=selected.device,
            )
            v_hat = torch.zeros_like(p_hat)
            if ordinary_row.numel():
                selected_common = {
                    name: (
                        value.index_select(0, ordinary_row)
                        if isinstance(value, Tensor)
                        else value
                    )
                    for name, value in common.items()
                }
                completed = self.evaluator.complete_selected(
                    **selected_common,
                    selected_player=selected.index_select(
                        0, ordinary_row
                    ).to(torch.int32).contiguous(),
                )
                p_hat.index_copy_(
                    0,
                    ordinary_row,
                    completed.p_hat.reshape(
                        -1, 1, self.geometry.private_hands
                    ).float(),
                )
                v_hat.index_copy_(
                    0,
                    ordinary_row,
                    completed.v_hat.reshape(
                        -1, 1, self.geometry.private_hands
                    ).float(),
                )
            cutoff_row = torch.nonzero(cutoff, as_tuple=False).flatten()
            if cutoff_row.numel():
                cutoff_selected = selected.index_select(0, cutoff_row)
                ranges = common["range_weights"].index_select(
                    0, cutoff_row
                )
                row = torch.arange(
                    cutoff_row.numel(), device=selected.device
                )
                mass = ranges[row, cutoff_selected]
                mass = mass / mass.sum(-1, keepdim=True).clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                baseline = common["payoff_baseline"].index_select(
                    0, cutoff_row
                )[row, cutoff_selected]
                p_hat.index_copy_(0, cutoff_row, mass[:, None])
                v_hat.index_copy_(
                    0,
                    cutoff_row,
                    baseline[:, None, None].expand(
                        -1, 1, self.geometry.private_hands
                    ),
                )
        else:
            completed = self.evaluator.complete(**common)
            p_hat = completed.p_hat.reshape(
                flat, p_train, self.geometry.private_hands
            ).float().contiguous()
            v_hat = completed.v_hat.reshape_as(p_hat).float()
        v_hat.mul_(self.utility_scale)
        lane_count = batch.lanes.numel()
        pre = fields["board_cards"].reshape(
            lane_count, batch.rows, 5
        )[:, :, 0].eq(PAD_CARD)
        route_family = (~pre).to(torch.int32)
        route = stable_route(
            route_family,
            torch.ones_like(route_family, dtype=torch.bool),
            (0, 1),
            declared_values=torch.tensor(
                (0, 1), dtype=torch.int32, device=p_hat.device
            ),
        )
        p_hat_rows = p_hat.reshape(
            lane_count, batch.rows, p_train, self.geometry.private_hands
        )
        v_hat_rows = v_hat.reshape_as(p_hat_rows)
        destination_names = ("pre", "post")
        requests: list[DeviceDestinationRequest] = []
        for destination_index, name in enumerate(destination_names):
            slab = self.layer_work[name]
            values = {
                field.name: torch.full(
                    (self.n_replica, self.pack_cap, *field.shape),
                    field.fill,
                    dtype=field.dtype,
                    device=p_hat.device,
                )
                for field in slab.schema.fields
            }
            source_row = route.source_row[:, destination_index]
            gather = source_row[..., None, None].expand_as(p_hat_rows)
            selected_p = p_hat_rows.gather(1, gather).reshape(
                flat, p_train, self.geometry.private_hands
            ).contiguous()
            selected_v = v_hat_rows.gather(1, gather).reshape_as(
                selected_p
            ).contiguous()
            work = rb_completion(
                selected_p,
                selected_v,
                branched=False,
                algebra=self.algebra,
                source_output="post",
                destination_output=name,
            )
            values["mass"][batch.lanes, : batch.rows, 0] = work.mass.reshape(
                lane_count, batch.rows, p_train, -1
            )
            values["moment"][batch.lanes, : batch.rows, 0] = work.moment.reshape(
                lane_count, batch.rows, p_train, -1
            )
            counts = torch.zeros(
                self.n_replica, dtype=torch.int64, device=p_hat.device
            )
            counts.index_copy_(
                0, batch.lanes, route.counts[:, destination_index]
            )
            requests.append(
                DeviceDestinationRequest(
                    slab,
                    counts,
                    values,
                    name,
                    torch.where(
                        counts > 0,
                        batch.row[:, 0],
                        torch.full_like(counts, torch.iinfo(torch.int64).max),
                    ),
                )
            )
        valid = torch.zeros(
            (self.n_replica, self.pack_cap),
            dtype=torch.bool,
            device=p_hat.device,
        )
        valid[batch.lanes, : batch.rows] = True
        source = torch.arange(
            self.pack_cap, dtype=torch.int64, device=p_hat.device
        )[None].expand(self.n_replica, -1).clone()
        destination = torch.zeros_like(source)
        ordinal = torch.zeros_like(source)
        destination[batch.lanes, : batch.rows] = route_family.to(torch.int64)
        ordinal[batch.lanes, : batch.rows] = torch.where(
            pre,
            pre.cumsum(1, dtype=torch.int64) - 1,
            (~pre).cumsum(1, dtype=torch.int64) - 1,
        )
        return CompletionPlan(
            tuple(requests),
            CommitCoverage(valid, source, destination, ordinal),
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
    "QueuedPokerError",
    "SampledShowdownBinding",
    "RBCompletionBinding",
    "SequentialSampledPokerHooks",
    "SequentialPokerHooks",
    "sampled_partitions",
    "rb_partitions",
    "poker_partitions",
    "validate_phase5_mode",
]
