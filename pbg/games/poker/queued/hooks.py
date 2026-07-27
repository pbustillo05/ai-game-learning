"""Queued NLHE hooks for the self-play sequential sampled path.

The module owns poker geometry, public transitions, direct model inputs,
collision-free private roots, sampled leaves, and the evaluator-v2 showdown
boundary.  Generic queued modules remain game-opaque.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    CompletionBatch,
    CompletionPlan,
    DrawPurpose,
    logical_rng_keys,
)
from pbg.queued.storage import DeviceDestinationRequest, MonotoneSlab
from pbg.games.poker.queued.policy import (
    CHECK_CALL,
    FOLD,
    RAISE_OFFSET,
    PokerLegality,
)


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
        return 8 + 9 * self.players

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

        live = state.status.eq(LIVE).sum(-1, dtype=torch.int32)
        return torch.where(state.street.eq(0), torch.zeros_like(live), live - 1)

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
        mask[:, CHECK_CALL] = True
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

    def direct_model_input(self, state: PokerState) -> Tensor:
        """Return the retained fixed-width public tensor without a DTO adapter."""

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
        return torch.cat(
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
        ).contiguous()

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
        if offset != model_input.shape[1]:
            raise QueuedPokerError("retained model input layout is not exact")
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
        row = torch.arange(state.rows, device=state.device)
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
        status[row[fold], actor[fold]] = FOLDED
        pending[row, actor] = False
        call = (current_bet - committed_street[row, actor]).clamp_min(0.0)
        paid_call = torch.minimum(call, stacks[row, actor])
        paid = torch.where(call_action, paid_call, torch.zeros_like(paid_call))
        raise_to = (action.float() - RAISE_OFFSET + 1.0) * 0.5
        paid_raise = (raise_to - committed_street[row, actor]).clamp_min(0.0)
        paid = torch.where(raise_action, paid_raise, paid)
        paid = torch.where(fold, torch.zeros_like(paid), paid)
        stacks[row, actor] -= paid
        committed[row, actor] += paid
        committed_street[row, actor] += paid
        all_in = stacks[row, actor].eq(0) & ~fold
        status[row[all_in], actor[all_in]] = ALL_IN
        old_bet = current_bet.clone()
        current_bet = torch.where(raise_action, raise_to, current_bet)
        raise_size = raise_to - old_bet
        full_raise = raise_action & (raise_size >= last_full_raise)
        last_full_raise = torch.where(full_raise, raise_size, last_full_raise)
        active = status.eq(LIVE)
        pending = torch.where(full_raise[:, None] & active, active, pending)
        pending[row, actor] = False
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
        used.scatter_(
            1,
            (state.board_cards.long() - self.geometry.first_card).clamp(
                0, self.geometry.deck_cards - 1
            ),
            board_valid,
        )
        ordering_key = torch.where(
            used, torch.full_like(logical_key, torch.iinfo(torch.int64).max), logical_key
        )
        order = torch.argsort(ordering_key, dim=1, stable=True)
        dealt = cards[order[:, :3]]
        board = state.board_cards.clone()
        flop = state.street.eq(0) & needs
        turn = state.street.eq(1) & needs
        river = state.street.eq(2) & needs
        board[flop, :3] = dealt[flop, :3].to(torch.uint8)
        board[turn, 3] = dealt[turn, 0].to(torch.uint8)
        board[river, 4] = dealt[river, 0].to(torch.uint8)
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
            state.terminal,
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
        key = logical_rng_keys(
            seed=seed,
            iteration=iteration,
            root=root[:, None].expand(-1, self.geometry.deck_cards),
            path=path[:, None].expand(-1, self.geometry.deck_cards),
            draw=draw * self.geometry.deck_cards + card_ordinal[None, :],
            purpose=DrawPurpose.CHANCE,
            retry_epoch=retry_epoch,
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
        amount[early] = 0
        combo = torch.where(
            early[:, None], torch.full_like(state.combo_ids, -1), state.combo_ids
        )
        return MappingProxyType(
            {
                "board_cards": state.board_cards,
                "combo_ids": combo,
                "stratum_level": level,
                "payoff_baseline": baseline,
                "stratum_amount": amount,
            }
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
    "PokerState",
    "PokerStateMachine",
    "QueuedPokerError",
    "SampledShowdownBinding",
    "sampled_partitions",
    "validate_phase5_mode",
]
