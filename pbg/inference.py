"""Stateful model-only NLHE inference through queued poker operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor

from pbg.devices import assert_module_device, resolve_execution_device
from pbg.games.poker.queued import (
    CHECK_CALL,
    FOLD,
    NLHEGeometry,
    PokerPolicyOperator,
    PokerState,
    PokerStateMachine,
    build_model_from_config,
)
from pbg.games.poker.queued.hooks import FOLDED, LIVE, PAD_CARD, RAISE_OFFSET
from pbg.inference_artifact import load_inference_source


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_ActionMode = Literal["none", "argmax", "sample"]


@dataclass(frozen=True)
class HandHandle:
    value: int


@dataclass(frozen=True)
class PokerAction:
    kind: str
    event_id: int
    raise_to: float | None = None
    all_in: bool = False


@dataclass(frozen=True)
class Decision:
    hand: HandHandle
    private_hand_id: int
    actions: tuple[PokerAction, ...]
    probabilities: Tensor
    value_bb: float
    seat_values_bb: tuple[float, ...] | None
    selected_action: PokerAction | None = None


@dataclass(frozen=True)
class FullStrategy:
    hands: tuple[HandHandle, ...]
    actions: tuple[tuple[PokerAction, ...], ...]
    probabilities: Tensor
    values_bb: Tensor
    legal_mask: Tensor
    compatible_hands: Tensor
    seat_values_bb: Tensor | None


@dataclass
class _Hand:
    state: PokerState
    log_reach: Tensor
    history: object


def parse_card(card: str | int, *, first_card: int = 0) -> int:
    if isinstance(card, bool):
        raise TypeError("card must be an integer id or two-character string")
    if isinstance(card, int):
        value = card
    elif isinstance(card, str) and len(card.strip()) == 2:
        text = card.strip()
        rank = _RANKS.find(text[0].upper())
        suit = _SUITS.find(text[1].lower())
        if rank < 0 or suit < 0:
            raise ValueError(f"invalid card {card!r}")
        value = 4 * rank + suit
    else:
        raise TypeError("card must be an integer id or two-character string")
    if not first_card <= value < 52:
        raise ValueError(f"card {card!r} is outside the configured active deck")
    return value


class PokerInferenceSession:
    """Own one strict artifact, accelerator model, and live public hands."""

    def __init__(
        self,
        source: dict,
        *,
        device: str | torch.device,
        precision: str = "fp32",
        mode: str = "latency",
        max_batch_rows: int | None = None,
    ) -> None:
        self.device = resolve_execution_device(device, workload="inference")
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision must be fp32 or fp16")
        if precision == "fp16" and self.device.type != "mps":
            raise ValueError("fp16 inference is supported only on MPS")
        if mode not in ("latency", "throughput"):
            raise ValueError("mode must be latency or throughput")
        if max_batch_rows is not None and max_batch_rows <= 0:
            raise ValueError("max_batch_rows must be positive")
        scientific = source.get("scientific_config")
        if not isinstance(scientific, dict):
            raise RuntimeError("inference source has no scientific configuration")
        game = scientific["game"]
        self.geometry = NLHEGeometry(
            game["players"], game["stack_bb"], game["rank_count"]
        )
        self.model = build_model_from_config(scientific, self.device)
        self.model.load_state_dict(source["model"], strict=True)
        self.model.eval().requires_grad_(False)
        self.model.configure_execution("sdpa", 128)
        assert_module_device(self.model, self.device, "inference model")
        self.machine = PokerStateMachine(self.geometry)
        self.policy = PokerPolicyOperator()
        self.precision = precision
        self.mode = mode
        self.max_batch_rows = max_batch_rows
        self.iteration = int(source["iteration"])
        self.source_checkpoint_sha256 = str(source["source_checkpoint_sha256"])
        self._hands: dict[int, _Hand] = {}
        self._next = 0
        self._combo_cards = self.machine.combo_cards(
            torch.arange(
                self.geometry.private_hands,
                dtype=torch.int32,
                device=self.device,
            )
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        precision: str = "fp32",
        mode: str = "latency",
        max_batch_rows: int | None = None,
    ) -> "PokerInferenceSession":
        return cls(
            load_inference_source(path),
            device=device,
            precision=precision,
            mode=mode,
            max_batch_rows=max_batch_rows,
        )

    def new_hands(self, buttons: Sequence[int]) -> list[HandHandle]:
        if any(not 0 <= int(button) < self.geometry.players for button in buttons):
            raise ValueError("button is outside the table")
        result = []
        for button in buttons:
            state = self.machine.initial_roots(
                1,
                seed=0,
                iteration=self._next,
                retry_epoch=0,
                logical_base=int(button),
                device=self.device,
            )
            state = replace(
                state,
                button=torch.tensor(
                    [button], dtype=torch.int32, device=self.device
                ),
            )
            handle = HandHandle(self._next)
            self._next += 1
            self._hands[handle.value] = _Hand(
                state,
                torch.zeros(
                    (1, self.geometry.players, self.geometry.private_hands),
                    dtype=torch.float32,
                    device=self.device,
                ),
                self.machine.empty_history(1, self.device),
            )
            result.append(handle)
        return result

    def new_hand(self, button: int = 0) -> HandHandle:
        return self.new_hands((button,))[0]

    def _hand(self, handle: HandHandle) -> _Hand:
        if not isinstance(handle, HandHandle) or handle.value not in self._hands:
            raise ValueError("unknown inference hand handle")
        return self._hands[handle.value]

    def _family(self, state: PokerState) -> str:
        if bool(state.terminal[0]):
            raise ValueError("hand has no further player decision")
        if bool(self.machine.needs_chance(state)[0]):
            raise ValueError("hand is at a chance node; deal public cards first")
        return (
            "pre"
            if int(state.street[0]) == 0
            else f"post_{int(state.status.ne(FOLDED).sum())}"
        )

    def _evaluate(self, item: _Hand):
        family = self._family(item.state)
        if self.device.type == "mps" and family != "pre":
            raise RuntimeError(
                "postflop MPS inference is unsupported because evaluator contract "
                "v2 does not provide folded-summary operations on MPS"
            )
        model_input = self.machine.direct_model_input(
            item.state, item.log_reach, item.history
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.precision == "fp16",
        ):
            evaluation = self.model.forward_partition(family, model_input)
            legality = self.machine.legality(item.state)
            probabilities = self.policy.probabilities(
                evaluation.policy_raw, legality
            ).float()
        return evaluation, legality, probabilities

    def _private_id(self, cards: Sequence[str | int]) -> int:
        if len(cards) != 2:
            raise ValueError("private hand must contain two cards")
        values = sorted(
            parse_card(card, first_card=self.geometry.first_card) for card in cards
        )
        if values[0] == values[1]:
            raise ValueError("private cards must be distinct")
        matches = self._combo_cards.eq(
            torch.tensor(values, device=self.device)
        ).all(1)
        index = torch.nonzero(matches, as_tuple=False)
        if index.numel() != 1:
            raise ValueError("private hand is outside the compact deck")
        return int(index[0, 0])

    def _actions(self, legality) -> tuple[PokerAction, ...]:
        actions = []
        for event in torch.nonzero(legality.mask[0], as_tuple=False).flatten().tolist():
            if event == FOLD:
                actions.append(PokerAction("fold", event))
            elif event == CHECK_CALL:
                actions.append(PokerAction("check_call", event))
            else:
                target = (event - RAISE_OFFSET + 1) * 0.5
                actions.append(PokerAction("raise_to", event, target))
        return tuple(actions)

    def predict(
        self,
        hand: HandHandle,
        private_hand: Sequence[str | int],
        *,
        action_mode: _ActionMode = "none",
        generator: torch.Generator | None = None,
    ) -> Decision:
        item = self._hand(hand)
        evaluation, legality, all_probability = self._evaluate(item)
        private = self._private_id(private_hand)
        board = item.state.board_cards[0]
        cards = self._combo_cards[private]
        if ((cards[:, None] == board[None]) & board[None].ne(PAD_CARD)).any():
            raise ValueError("private hand conflicts with the public board")
        actions = self._actions(legality)
        columns = torch.nonzero(legality.mask[0], as_tuple=False).flatten()
        probability = all_probability[0, private].index_select(0, columns).cpu()
        selected = None
        if action_mode == "argmax":
            selected = actions[int(probability.argmax())]
        elif action_mode == "sample":
            selected = actions[
                int(torch.multinomial(probability, 1, generator=generator))
            ]
        elif action_mode != "none":
            raise ValueError("action_mode must be none, argmax, or sample")
        return Decision(
            hand,
            private,
            actions,
            probability,
            float(evaluation.value[0, private].float().cpu()),
            None,
            selected,
        )

    def predict_many(
        self,
        hands: Sequence[HandHandle],
        private_hands: Sequence[Sequence[str | int]],
        *,
        action_mode: _ActionMode = "none",
        generator: torch.Generator | None = None,
    ) -> list[Decision]:
        if len(hands) != len(private_hands):
            raise ValueError("hands and private_hands differ in length")
        return [
            self.predict(
                hand,
                private,
                action_mode=action_mode,
                generator=generator,
            )
            for hand, private in zip(hands, private_hands, strict=True)
        ]

    def full_strategy(self, hands: Sequence[HandHandle]) -> FullStrategy:
        evaluations = [self._evaluate(self._hand(hand)) for hand in hands]
        probabilities = torch.cat([value[2] for value in evaluations]).cpu()
        values = torch.cat([value[0].value.float() for value in evaluations]).cpu()
        legal = torch.cat([value[1].mask for value in evaluations]).cpu()
        actions = tuple(self._actions(value[1]) for value in evaluations)
        compatible = torch.stack(
            tuple(
                ~(
                    (
                        self._combo_cards[:, :, None]
                        == self._hand(hand).state.board_cards[0][None, None]
                    )
                    & self._hand(hand).state.board_cards[0][None, None].ne(PAD_CARD)
                )
                .any(2)
                .any(1)
                for hand in hands
            )
        ).cpu()
        return FullStrategy(
            tuple(hands), actions, probabilities, values, legal, compatible, None
        )

    def actor(self, hand: HandHandle) -> int:
        """Return the seat owning the current decision."""

        item = self._hand(hand)
        if bool(item.state.terminal[0]) or bool(self.machine.needs_chance(item.state)[0]):
            raise ValueError("hand is not at a player decision")
        return int(item.state.to_act[0])

    def private_cards(self, hand: HandHandle, seat: int) -> tuple[int, int]:
        """Return the already-dealt compact-deck cards for one seat."""

        item = self._hand(hand)
        if not 0 <= seat < self.geometry.players:
            raise ValueError("seat is outside the table")
        cards = self.machine.combo_cards(item.state.combo_ids)[0, seat]
        return int(cards[0]), int(cards[1])

    def private_hand_id(self, hand: HandHandle, seat: int) -> int:
        item = self._hand(hand)
        if not 0 <= seat < self.geometry.players:
            raise ValueError("seat is outside the table")
        return int(item.state.combo_ids[0, seat])

    def is_terminal(self, hand: HandHandle) -> bool:
        return bool(self._hand(hand).state.terminal[0])

    def needs_chance(self, hand: HandHandle) -> bool:
        return bool(self.machine.needs_chance(self._hand(hand).state)[0])

    def policy_table(self, hand: HandHandle) -> tuple[tuple[PokerAction, ...], Tensor]:
        """Return legal actions and the full private-hand policy on-device."""

        _evaluation, legality, probabilities = self._evaluate(self._hand(hand))
        return self._actions(legality), probabilities[0]

    def advance(
        self,
        hand: HandHandle,
        action: str | PokerAction | tuple,
        *,
        policy_probabilities: Tensor | None = None,
    ) -> None:
        item = self._hand(hand)
        _evaluation, legality, probabilities = self._evaluate(item)
        actions = self._actions(legality)
        if isinstance(action, PokerAction):
            event = action.event_id
        elif isinstance(action, str):
            normalized = action.lower().replace("-", "_")
            matches = [
                value
                for value in actions
                if value.kind == normalized
                or normalized in ("check", "call")
                and value.kind == "check_call"
            ]
            if len(matches) != 1:
                raise ValueError(f"action {action!r} is not legal")
            event = matches[0].event_id
        elif (
            isinstance(action, (tuple, list))
            and len(action) == 2
            and action[0] == "raise_to"
        ):
            matches = [
                value
                for value in actions
                if value.raise_to is not None
                and abs(value.raise_to - float(action[1])) < 1.0e-6
            ]
            if len(matches) != 1:
                raise ValueError("raise-to amount is not legal")
            event = matches[0].event_id
        else:
            raise TypeError("unsupported poker action")
        actor = int(item.state.to_act[0])
        policy = probabilities[0] if policy_probabilities is None else policy_probabilities
        if (
            policy.device != self.device
            or policy.dtype not in (torch.float16, torch.float32, torch.bfloat16)
            or policy.shape != probabilities.shape[1:]
            or not bool(torch.isfinite(policy).all())
            or bool((policy < 0).any())
        ):
            raise ValueError(
                "external policy probabilities must be finite accelerator [H,E]"
            )
        selected = policy[:, event].float().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        item.log_reach[:, actor].add_(selected.log())
        event_tensor = torch.tensor([event], dtype=torch.int32, device=self.device)
        item.history = self.machine.append_player_history(
            item.state, event_tensor, item.history
        )
        item.state = self.machine.step_player(item.state, event_tensor)
        if self.device.type == "mps" and item.state.status.eq(FOLDED).any():
            if not bool(item.state.terminal[0]):
                raise RuntimeError(
                    "MPS folded-state inference is unsupported before CPU staging"
                )

    def deal_keyed(
        self,
        hand: HandHandle,
        *,
        seed: int,
        trajectory_id: int,
        decision_index: int,
    ) -> None:
        """Execute one schedule-independent chance transition on-device."""

        if self.device.type == "mps":
            raise RuntimeError(
                "MPS chance/postflop inference is unsupported before CPU staging"
            )
        item = self._hand(hand)
        if not bool(self.machine.needs_chance(item.state)[0]):
            raise ValueError("hand is not at a chance node")
        item.history = self.machine.append_chance_history(
            item.state,
            torch.ones(1, dtype=torch.bool, device=self.device),
            item.history,
        )
        item.state = self.machine.deal_next_street_keyed(
            item.state,
            seed=seed,
            iteration=0,
            root=torch.tensor(
                [trajectory_id], dtype=torch.int64, device=self.device
            ),
            path=torch.tensor(
                [decision_index], dtype=torch.int64, device=self.device
            ),
            draw=int(item.state.street[0]),
            retry_epoch=0,
        )

    def terminal_payoffs(self, hand: HandHandle) -> Tensor:
        """Settle one terminal equal-stack hand on the selected accelerator."""

        item = self._hand(hand)
        state = item.state
        if not bool(state.terminal[0]):
            raise ValueError("hand is not terminal")
        contenders = state.status[0].ne(FOLDED)
        baseline = -state.committed
        pot = state.committed.sum(-1)
        if int(contenders.sum()) == 1:
            result = baseline.clone()
            winner = torch.nonzero(contenders, as_tuple=False)[0, 0]
            result[0, winner] += pot[0]
            return result[0]
        if bool(state.board_cards[0].eq(PAD_CARD).any()):
            raise RuntimeError("showdown terminal omitted a complete board")
        import poker_gpu_evaluator as poker

        evaluator = poker.bind(rank_count=self.geometry.rank_count)
        combo_ids = torch.where(
            contenders[None],
            state.combo_ids,
            torch.full_like(state.combo_ids, -1),
        ).to(torch.int32)
        stratum_level = torch.where(
            contenders[None],
            torch.zeros_like(state.combo_ids, dtype=torch.int32),
            torch.full_like(state.combo_ids, -1, dtype=torch.int32),
        )
        amounts = torch.zeros(
            (1, self.geometry.players),
            dtype=torch.float32,
            device=self.device,
        )
        amounts[:, 0] = pot
        return evaluator.evaluate_showdowns(
            board_cards=state.board_cards,
            combo_ids=combo_ids,
            stratum_level=stratum_level,
            payoff_baseline=baseline,
            stratum_amount=amounts,
        )[0]

    def advance_many(
        self, hands: Sequence[HandHandle], actions: Sequence[object]
    ) -> None:
        if len(hands) != len(actions):
            raise ValueError("hands and actions differ in length")
        for hand, action in zip(hands, actions, strict=True):
            self.advance(hand, action)

    def deal(self, hand: HandHandle, cards: Sequence[str | int]) -> None:
        if self.device.type == "mps":
            raise RuntimeError(
                "MPS chance/postflop inference is unsupported before CPU staging"
            )
        item = self._hand(hand)
        if not bool(self.machine.needs_chance(item.state)[0]):
            raise ValueError("hand is not at a chance node")
        expected = 3 if int(item.state.street[0]) == 0 else 1
        if len(cards) != expected:
            raise ValueError(f"chance transition requires {expected} public card(s)")
        parsed = [
            parse_card(card, first_card=self.geometry.first_card) for card in cards
        ]
        if len(set(parsed)) != len(parsed):
            raise ValueError("public cards must be unique")
        board = item.state.board_cards.clone()
        if expected == 3:
            board[0, :3] = torch.tensor(parsed, dtype=torch.uint8, device=self.device)
        else:
            board[0, int(item.state.street[0]) + 2] = parsed[0]
        active = item.state.status.eq(LIVE)
        next_actor = (item.state.button + 1).remainder(self.geometry.players)
        item.history = self.machine.append_chance_history(
            item.state,
            torch.ones(1, dtype=torch.bool, device=self.device),
            item.history,
        )
        item.state = replace(
            item.state,
            board_cards=board,
            street=item.state.street + 1,
            to_act=next_actor,
            committed_street=torch.zeros_like(item.state.committed_street),
            current_bet=torch.zeros_like(item.state.current_bet),
            last_full_raise=torch.ones_like(item.state.last_full_raise),
            pending=active,
            can_raise=active,
        )
        item.state.validate(self.geometry)

    def deal_many(
        self,
        hands: Sequence[HandHandle],
        cards: Sequence[Sequence[str | int]],
    ) -> None:
        if len(hands) != len(cards):
            raise ValueError("hands and boards differ in length")
        for hand, board in zip(hands, cards, strict=True):
            self.deal(hand, board)


__all__ = [
    "Decision",
    "FullStrategy",
    "HandHandle",
    "PokerAction",
    "PokerInferenceSession",
    "parse_card",
]
