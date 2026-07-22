"""Stateful accelerator-native NLHE inference on CUDA or Apple MPS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL
from pbg.games.poker.adapter import PokerPolicyAdapter
from pbg.games.poker.bundle import build_model_from_config
from pbg.games.poker.rules import (
    CHECK_CALL,
    FOLD,
    FOLDED,
    RAISE_OFFSET,
    PokerConfig,
)
from pbg.games.poker.folded import PartitionSummaryBatch, PokerGPUPartitionAdapter
from pbg.games.poker.model.postflop import PostflopMasks
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerPublicBatch,
    PreflopMasks,
)
from pbg.games.poker.model.primitives import MHA
from pbg.inference_artifact import load_inference_source
from pbg.core.game import EventBatch
from pbg.core.devices import assert_module_device, resolve_execution_device
from pbg.games.poker.rollout_backend import PokerRolloutBackend


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_ActionMode = Literal["none", "argmax", "sample"]


@dataclass(frozen=True)
class HandHandle:
    """Opaque handle for one live inference hand."""

    value: int


@dataclass(frozen=True)
class PokerAction:
    """One legal public action in human-facing units."""

    kind: str
    event_id: int
    raise_to: float | None = None
    all_in: bool = False


@dataclass(frozen=True)
class Decision:
    """Selected-private-hand policy and value result."""

    hand: HandHandle
    private_hand_id: int
    actions: tuple[PokerAction, ...]
    probabilities: Tensor
    value_bb: float
    seat_values_bb: tuple[float, ...] | None
    selected_action: PokerAction | None = None


@dataclass(frozen=True)
class FullStrategy:
    """CPU result for all private hands in one or more public spots."""

    hands: tuple[HandHandle, ...]
    actions: tuple[tuple[PokerAction, ...], ...]
    probabilities: Tensor  # float32 [B,H,E]
    values_bb: Tensor  # float32 [B,H]
    legal_mask: Tensor  # bool [B,E]
    compatible_hands: Tensor  # bool [B,H]
    seat_values_bb: Tensor | None  # float32 [B,P]


@dataclass
class _HandState:
    ref: int
    log_reach: Tensor  # float32 [P,H], model device


def parse_card(card: str | int, *, first_card: int = 0) -> int:
    """Parse a global card id or conventional two-character card string."""

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


def _move_public(pub: PokerPublicBatch, device: torch.device) -> PokerPublicBatch:
    values: dict[str, Any] = {}
    for name in pub.__dataclass_fields__:
        value = getattr(pub, name)
        if isinstance(value, PackedPublicHistory):
            values[name] = PackedPublicHistory(
                **{
                    field: getattr(value, field).to(device=device)
                    for field in value.__dataclass_fields__
                }
            )
        else:
            values[name] = None if value is None else value.to(device=device)
    return PokerPublicBatch(**values)


def _move_summary(
    summary: PartitionSummaryBatch, device: torch.device
) -> PartitionSummaryBatch:
    return PartitionSummaryBatch(
        fields={
            name: value.to(device=device) for name, value in summary.fields.items()
        },
        metadata=dict(summary.metadata),
    )


class PokerInferenceSession:
    """Own a checkpoint-backed policy/value model and stateful public hands."""

    def __init__(
        self,
        source: dict[str, Any],
        *,
        device: str | torch.device,
        precision: str = "auto",
        mode: str = "auto",
        max_batch_rows: int | None = None,
    ) -> None:
        self.device = resolve_execution_device(device, workload="inference")
        if precision not in {"auto", "fp32", "fp16"}:
            raise ValueError("precision must be 'auto', 'fp32', or 'fp16'")
        if mode not in {"auto", "latency", "throughput"}:
            raise ValueError("mode must be 'auto', 'latency', or 'throughput'")
        if max_batch_rows is not None and (
            isinstance(max_batch_rows, bool) or max_batch_rows <= 0
        ):
            raise ValueError("max_batch_rows must be positive when supplied")
        if precision == "auto":
            self.precision = "fp16" if self.device.type == "mps" else "fp32"
        else:
            self.precision = precision
        if self.device.type != "mps" and self.precision == "fp16":
            raise ValueError("fp16 inference is currently supported only on MPS")
        self.mode = mode
        self._max_batch_rows_override = max_batch_rows

        scientific = source["config"]
        game = scientific.get("game")
        if not isinstance(game, dict):
            raise RuntimeError("inference source has no game configuration")
        self.game_config = PokerConfig(
            n_players=game["n_players"],
            stack_bb=game["stack_bb"],
            rank_count=game["rank_count"],
        )
        self.model = build_model_from_config(scientific, self.device)
        self.model.load_state_dict(source["model"], strict=True)
        self.model.eval().requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, MHA):
                module.configure_execution(
                    "sdpa" if self.device.type == "mps" else "auto", 128
                )
        assert_module_device(self.model, self.device, "inference model")
        self.adapter = PokerPolicyAdapter(self.game_config.n_ticks)
        self.partition_adapter = PokerGPUPartitionAdapter(
            beta_rank=self.model.postflop.cfg.folder_beta_rank,
            rank_count=self.game_config.rank_count,
        )
        self.backend = PokerRolloutBackend(
            self.game_config,
            self.device,
            storage_page_rows=max(16, int(max_batch_rows or 64)),
            max_rollout_steps=self.model.preflop.cfg.max_history_tokens,
            partition_adapter=self.partition_adapter,
        )
        self.backend.states.reset()
        self.iteration = int(source["iteration"])
        self.source_checkpoint_sha256 = str(source["source_checkpoint_sha256"])
        self._hands: dict[int, _HandState] = {}
        self._next_hand_id = 0
        self._combo_cards_cpu = torch.tensor(
            self.game_config.geometry.combo_cards, dtype=torch.int64
        )
        self._combo_cards_device = self._combo_cards_cpu.to(self.device)
        self._combo_class_device = torch.tensor(
            self.game_config.geometry.class_of_combo,
            dtype=torch.int64,
            device=self.device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        precision: str = "auto",
        mode: str = "auto",
        max_batch_rows: int | None = None,
    ) -> "PokerInferenceSession":
        """Load a model-only artifact or full training checkpoint."""

        return cls(
            load_inference_source(path),
            device=device,
            precision=precision,
            mode=mode,
            max_batch_rows=max_batch_rows,
        )

    def new_hand(self, button: int = 0) -> HandHandle:
        """Create one inference hand with uniform initial private ranges."""

        handle = HandHandle(self._next_hand_id)
        self._next_hand_id += 1
        self._hands[handle.value] = _HandState(
            ref=int(self.backend.append_initial_states(
                torch.tensor([button], dtype=torch.int32, device=self.device)
            ).to("cpu")[0]),
            log_reach=torch.zeros(
                (self.game_config.n_players, self.game_config.geometry.n_hands),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        return handle

    def new_hands(self, buttons: Sequence[int]) -> list[HandHandle]:
        """Create a batch of independent hands in input order."""

        if not buttons:
            return []
        roots = self.backend.append_initial_states(
            torch.tensor(buttons, dtype=torch.int32, device=self.device)
        ).to("cpu").tolist()
        result: list[HandHandle] = []
        for ref in roots:
            handle = HandHandle(self._next_hand_id)
            self._next_hand_id += 1
            self._hands[handle.value] = _HandState(
                ref=int(ref),
                log_reach=torch.zeros(
                    (self.game_config.n_players, self.game_config.geometry.n_hands),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            result.append(handle)
        return result

    def _state(self, hand: HandHandle) -> _HandState:
        if not isinstance(hand, HandHandle) or hand.value not in self._hands:
            raise ValueError("unknown inference hand handle")
        return self._hands[hand.value]

    def _owner(self, state: _HandState) -> int:
        return int(
            self.backend.classify(torch.tensor(
                [state.ref], dtype=torch.int32, device=self.device
            )).to("cpu")[0]
        )

    def _require_decision(self, hand: HandHandle) -> _HandState:
        state = self._state(hand)
        owner = self._owner(state)
        if owner == CHANCE:
            raise ValueError("hand is at a chance node; deal public cards first")
        if owner in {TERMINAL, PRETERM}:
            raise ValueError("hand has no further player decision")
        return state

    def _parse_private_hand(self, cards: Sequence[str | int]) -> int:
        if len(cards) != 2:
            raise ValueError("private_hand must contain exactly two cards")
        parsed = [
            parse_card(card, first_card=self.game_config.geometry.first_card)
            for card in cards
        ]
        if parsed[0] == parsed[1]:
            raise ValueError("private cards must be distinct")
        return self.game_config.geometry.hand_index(*sorted(parsed))

    def _resolve_action(self, ref: int, action: Any) -> int:
        refs = torch.tensor([ref], dtype=torch.int32, device=self.device)
        _, legal = self.backend.legal_events(refs)
        row = refs.to(torch.int64)
        actor = self.backend.states.actor[row].to(torch.int64)
        to_call = float((
            self.backend.states.current_bet[row]
            - self.backend.states.committed_street[row, actor]
        ).clamp_min(0).to("cpu")[0])
        if isinstance(action, PokerAction):
            event = action.event_id
        elif isinstance(action, str):
            name = action.strip().lower().replace("-", "_")
            if name == "fold":
                event = FOLD
            elif name in {"check_call", "check", "call"}:
                if name == "check" and to_call > 1e-6:
                    raise ValueError("check is illegal while facing a bet")
                if name == "call" and to_call <= 1e-6:
                    raise ValueError("call is illegal when checking is available")
                event = CHECK_CALL
            elif name in {"all_in", "allin"}:
                target = float((
                    self.backend.states.committed_street[row, actor]
                    + self.backend.states.stacks[row, actor]
                ).to("cpu")[0])
                event = self._raise_event_for(ref, target)
            else:
                raise ValueError(f"unknown poker action {action!r}")
        elif (
            isinstance(action, (tuple, list))
            and len(action) == 2
            and str(action[0]).lower().replace("-", "_") == "raise_to"
        ):
            event = self._raise_event_for(ref, float(action[1]))
        else:
            raise TypeError(
                "action must be fold/check/call/all_in, PokerAction, or ('raise_to', bb)"
            )
        if not 0 <= event < self.backend.e_max or not bool(legal[0, event].to("cpu")):
            raise ValueError(f"event {event} is illegal in the current public state")
        return int(event)

    def _raise_event_for(self, ref: int, target: float) -> int:
        if not math.isfinite(target):
            raise ValueError("raise-to amount must be finite")
        refs = torch.tensor([ref], dtype=torch.int32, device=self.device)
        targets = self.backend.raise_targets(refs)[0]
        _, legal = self.backend.legal_events(refs)
        matches = torch.isclose(
            targets, torch.tensor(target, device=self.device), atol=1e-6, rtol=0.0
        )
        columns = (matches & legal[0, RAISE_OFFSET:]).nonzero(as_tuple=False)
        if columns.numel() != 1:
            raise ValueError(
                f"raise-to {target:g} bb is not legal in the current state"
            )
        return RAISE_OFFSET + int(columns[0, 0])

    def _actions_for(self, ref: int) -> tuple[PokerAction, ...]:
        refs = torch.tensor([ref], dtype=torch.int32, device=self.device)
        _, legal = self.backend.legal_events(refs)
        targets = self.backend.raise_targets(refs)[0]
        row = refs.to(torch.int64)
        actor = self.backend.states.actor[row].to(torch.int64)
        to_call = float((self.backend.states.current_bet[row]
            - self.backend.states.committed_street[row, actor]).clamp_min(0).to("cpu")[0])
        maximum = float((self.backend.states.committed_street[row, actor]
            + self.backend.states.stacks[row, actor]).to("cpu")[0])
        actions: list[PokerAction] = []
        for event in legal[0].nonzero(as_tuple=False).squeeze(1).to("cpu").tolist():
            if event == FOLD:
                actions.append(PokerAction("fold", event))
            elif event == CHECK_CALL:
                actions.append(
                    PokerAction("call" if to_call > 1e-6 else "check", event)
                )
            else:
                target = float(targets[event - RAISE_OFFSET].to("cpu"))
                actions.append(
                    PokerAction(
                        "raise_to",
                        event,
                        raise_to=target,
                        all_in=math.isclose(target, maximum, abs_tol=1e-6),
                    )
                )
        return tuple(actions)

    def _autocast(self):
        if self.device.type == "mps" and self.precision == "fp16":
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return nullcontext()

    def _physical_batch_rows(self, family: tuple[int, int, int], logical: int) -> int:
        if self.mode == "latency" or (self.mode == "auto" and logical == 1):
            return 1
        street, live, history = family
        scalar_count = (
            3 * self.game_config.n_players
            + 13
            + self.backend.e_max
            + self.game_config.n_ticks
        )
        if street == 0:
            tokens = (
                self.game_config.n_players * self.game_config.geometry.n_classes
                + scalar_count
                + 1
                + history
                + self.model.preflop.cfg.k_act
            )
        else:
            tokens = (
                live * math.comb(self.game_config.geometry.n_cards - (street + 2), 2)
                + scalar_count
                + 6
                + history
                + self.model.postflop.cfg.k_act
            )
        token_cap = max(1, 131_072 // tokens)
        pair_cap = max(1, 268_435_456 // (tokens * tokens))
        admitted = min(logical, 64, token_cap, pair_cap)
        if self.device.type == "mps" and live < self.game_config.n_players:
            recommended = int(torch.mps.recommended_max_memory())
            driver = int(torch.mps.driver_allocated_memory())
            reserve = max(2 * 1024**3, recommended // 4)
            available = max(256 * 1024**2, recommended - reserve - driver)
            per_row = (64 if street == 0 else 128 + 32 * live) * 1024**2
            admitted = min(admitted, max(1, available // per_row))
        else:
            admitted = min(admitted, 256)
        if self._max_batch_rows_override is not None:
            admitted = min(admitted, self._max_batch_rows_override)
        return max(1, admitted)

    @staticmethod
    def _weights(log_reach: Tensor) -> Tensor:
        finite = torch.isfinite(log_reach)
        gauge = log_reach.masked_fill(~finite, -torch.inf).amax(2, keepdim=True)
        return torch.exp(log_reach - gauge).masked_fill(~finite, 0.0)

    def _compatibility(self, board: Tensor) -> Tensor:
        real = board >= 0
        conflict = (
            (
                (self._combo_cards_device[None, :, :, None] == board[:, None, None, :])
                & real[:, None, None, :]
            )
            .any(3)
            .any(2)
        )
        return ~conflict

    def _partition_summary(
        self,
        refs: Tensor,
        pub: PokerPublicBatch,
        weights: Tensor,
        live: int,
    ) -> PartitionSummaryBatch:
        if self.device.type == "mps":
            raise RuntimeError(
                "postflop MPS inference is unavailable because evaluator contract v2 "
                "defers summarize_folders on MPS"
            )
        return self.backend._partition_summary_uncached(
            refs,
            pub.board,
            weights,
            pub.status,
            live,
            compute_mask=torch.ones(refs.shape, dtype=torch.bool, device=self.device),
        )

    def _evaluate_full_device(
        self, handles: Sequence[HandHandle]
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor]:
        if not handles:
            raise ValueError("at least one hand is required")
        states = [self._require_decision(hand) for hand in handles]
        refs = torch.tensor(
            [state.ref for state in states], dtype=torch.int32, device=self.device
        )
        legal_ids_all, legal_mask_all = self.backend.legal_events(refs)
        batch = len(handles)
        n_hands = self.game_config.geometry.n_hands
        e_max = self.backend.e_max
        probabilities = torch.empty(
            (batch, n_hands, e_max), dtype=torch.float32, device=self.device
        )
        values = torch.empty((batch, n_hands), dtype=torch.float32, device=self.device)
        seat_values: Tensor | None = None

        groups: dict[tuple[int, int, int], list[int]] = {}
        controls = torch.stack((
            self.backend.states.street[refs.long()].to(torch.int32),
            (self.backend.states.status[refs.long()] != FOLDED).sum(1).to(torch.int32),
            self.backend.states.history_length[refs.long()].to(torch.int32),
        ), dim=1).to("cpu").tolist()
        for index, (street, live, history) in enumerate(controls):
            history_bucket = 0 if history == 0 else 1 << (history - 1).bit_length()
            groups.setdefault((street, live, history_bucket), []).append(index)

        with torch.inference_mode():
            for family, positions in groups.items():
                physical = self._physical_batch_rows(family, len(positions))
                for start in range(0, len(positions), physical):
                    selected = positions[start : start + physical]
                    selected_states = [states[index] for index in selected]
                    selected_refs = torch.tensor(
                        [state.ref for state in selected_states],
                        dtype=torch.int32,
                        device=self.device,
                    )
                    pub = self.backend.encode_public(selected_refs)
                    log_reach = torch.stack(
                        [state.log_reach for state in selected_states]
                    )
                    weights = self._weights(log_reach)
                    with self._autocast():
                        if family[0] == 0:
                            class_ranges = torch.zeros(
                                (
                                    len(selected),
                                    self.game_config.n_players,
                                    self.game_config.geometry.n_classes,
                                ),
                                dtype=torch.float32,
                                device=self.device,
                            )
                            ids = self._combo_class_device.view(1, 1, -1).expand_as(
                                weights
                            )
                            class_ranges.scatter_add_(2, ids, weights)
                            output = self.model.forward_preflop(
                                pub,
                                class_ranges,
                                pub.to_act,
                                PreflopMasks(
                                    seat_mask=pub.seat_mask, status=pub.status
                                ),
                            )
                        else:
                            summary = self._partition_summary(
                                selected_refs,
                                pub,
                                weights,
                                family[1],
                            )
                            compatibility = self._compatibility(pub.board)
                            live_ids = (
                                torch.where(
                                    pub.status != FOLDED,
                                    torch.arange(
                                        self.game_config.n_players, device=self.device
                                    )[None],
                                    self.game_config.n_players,
                                )
                                .sort(1)
                                .values[:, : family[1]]
                            )
                            output = self.model.forward_postflop(
                                pub,
                                weights,
                                pub.to_act,
                                PostflopMasks(
                                    compatibility=compatibility,
                                    seat_mask=pub.seat_mask,
                                    combo_mask=weights > 0,
                                    status=pub.status,
                                    evaluator_summary=summary,
                                    live_seat_ids=live_ids,
                                    compact_size=math.comb(
                                        self.game_config.geometry.n_cards
                                        - (family[0] + 2),
                                        2,
                                    ),
                                ),
                            )
                        index = torch.tensor(selected, device=self.device)
                        legal_ids = legal_ids_all[index]
                        legal_mask = legal_mask_all[index]
                        part_probability = self.adapter.probabilities(
                            output, legal_ids, legal_mask
                        )
                    destination = torch.tensor(
                        selected, dtype=torch.int64, device=self.device
                    )
                    probabilities[destination] = part_probability.float()
                    values[destination] = output.v.float()
        return (
            probabilities,
            values,
            seat_values,
            legal_ids_all,
            legal_mask_all,
        )

    def predict(
        self,
        hand: HandHandle,
        private_hand: Sequence[str | int],
        *,
        action_mode: _ActionMode = "none",
        seed: int | None = None,
    ) -> Decision:
        """Return the current actor's selected-hand policy and value."""

        return self.predict_many(
            [hand], [private_hand], action_mode=action_mode, seed=seed
        )[0]

    def predict_many(
        self,
        hands: Sequence[HandHandle],
        private_hands: Sequence[Sequence[str | int]],
        *,
        action_mode: _ActionMode = "none",
        seed: int | None = None,
    ) -> list[Decision]:
        """Evaluate selected private hands while preserving request order."""

        if len(hands) != len(private_hands) or not hands:
            raise ValueError(
                "hands and private_hands must have the same nonzero length"
            )
        if any(not isinstance(hand, HandHandle) for hand in hands):
            raise ValueError("every hand must be a HandHandle")
        if len({hand.value for hand in hands}) != len(hands):
            raise ValueError("one hand may appear only once in a batched request")
        if action_mode not in {"none", "argmax", "sample"}:
            raise ValueError("action_mode must be 'none', 'argmax', or 'sample'")
        if action_mode == "sample" and seed is None:
            raise ValueError("sample action_mode requires an explicit seed")
        combo_ids = [self._parse_private_hand(cards) for cards in private_hands]
        probability, values, seat_values, _, legal_mask = self._evaluate_full_device(
            hands
        )
        rows = torch.arange(len(hands), device=self.device)
        combos = torch.tensor(combo_ids, dtype=torch.int64, device=self.device)
        selected_probability = probability[rows, combos].to(device="cpu")
        selected_value = values[rows, combos].to(device="cpu")
        selected_seats = None if seat_values is None else seat_values.to(device="cpu")
        legal_mask_cpu = legal_mask.to(device="cpu")
        generator = None
        if action_mode == "sample":
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        results: list[Decision] = []
        for row, (hand, combo) in enumerate(zip(hands, combo_ids, strict=True)):
            state = self._state(hand)
            board = self.backend.states.board[state.ref].to("cpu").tolist()
            cards = self._combo_cards_cpu[combo]
            if any(int(card) in board for card in cards):
                raise ValueError("private_hand conflicts with the public board")
            actions = self._actions_for(state.ref)
            columns = legal_mask_cpu[row].nonzero(as_tuple=False).squeeze(1)
            row_probability = selected_probability[row, columns].float()
            chosen = None
            if action_mode == "argmax":
                chosen = actions[int(row_probability.argmax())]
            elif action_mode == "sample":
                choice = int(
                    torch.multinomial(row_probability, 1, generator=generator)[0]
                )
                chosen = actions[choice]
            results.append(
                Decision(
                    hand=hand,
                    private_hand_id=combo,
                    actions=actions,
                    probabilities=row_probability,
                    value_bb=float(selected_value[row]),
                    seat_values_bb=(
                        None
                        if selected_seats is None
                        else tuple(float(value) for value in selected_seats[row])
                    ),
                    selected_action=chosen,
                )
            )
        return results

    def full_strategy(self, hands: HandHandle | Sequence[HandHandle]) -> FullStrategy:
        """Return all-hand policy/value rows on CPU for analysis."""

        selected = [hands] if isinstance(hands, HandHandle) else list(hands)
        probability, values, seat_values, _, legal_mask = self._evaluate_full_device(
            selected
        )
        refs = torch.tensor(
            [self._state(hand).ref for hand in selected],
            dtype=torch.int64,
            device=self.device,
        )
        boards = self.backend.states.board[refs].to(torch.int64)
        compatible = self._compatibility(boards)
        return FullStrategy(
            hands=tuple(selected),
            actions=tuple(
                self._actions_for(self._state(hand).ref) for hand in selected
            ),
            probabilities=probability.to(device="cpu"),
            values_bb=values.to(device="cpu"),
            legal_mask=legal_mask.to(device="cpu"),
            compatible_hands=compatible.to(device="cpu"),
            seat_values_bb=(
                None if seat_values is None else seat_values.to(device="cpu")
            ),
        )

    def advance(self, hand: HandHandle, action: Any) -> HandHandle:
        """Apply one observed legal player action and update belief reaches."""

        self.advance_many([hand], [action])
        return hand

    def advance_many(
        self, hands: Sequence[HandHandle], actions: Sequence[Any]
    ) -> list[HandHandle]:
        """Apply a batch of observed actions after one grouped model evaluation."""

        if len(hands) != len(actions) or not hands:
            raise ValueError("hands and actions must have the same nonzero length")
        if any(not isinstance(hand, HandHandle) for hand in hands):
            raise ValueError("every hand must be a HandHandle")
        if len({hand.value for hand in hands}) != len(hands):
            raise ValueError("one hand may appear only once in a batched request")
        states = [self._require_decision(hand) for hand in hands]
        events = [
            self._resolve_action(state.ref, action)
            for state, action in zip(states, actions, strict=True)
        ]
        probability, _, _, _, _ = self._evaluate_full_device(hands)
        for row, (hand, state, event) in enumerate(
            zip(hands, states, events, strict=True)
        ):
            ref = torch.tensor([state.ref], dtype=torch.int32, device=self.device)
            actor = int(self.backend.states.actor[ref.long()].to("cpu")[0])
            selected = probability[row, :, event].clamp_min(self.adapter.eps)
            state.log_reach[actor] += torch.log(selected).float()
            finite = torch.isfinite(state.log_reach[actor])
            gauge = state.log_reach[actor].masked_fill(~finite, -torch.inf).amax()
            state.log_reach[actor] -= gauge
            successor = self.backend.step(
                ref,
                EventBatch(torch.tensor([event], dtype=torch.int32, device=self.device)),
            )
            state.ref = int(successor.to("cpu")[0])
        return list(hands)

    def deal(self, hand: HandHandle, cards: Sequence[str | int]) -> HandHandle:
        """Deal the required flop, turn, or river cards and mask all ranges."""

        self.deal_many([hand], [cards])
        return hand

    def deal_many(
        self,
        hands: Sequence[HandHandle],
        card_rows: Sequence[Sequence[str | int]],
    ) -> list[HandHandle]:
        """Apply public chance cards to independent hands in input order."""

        if len(hands) != len(card_rows) or not hands:
            raise ValueError("hands and card_rows must have the same nonzero length")
        if any(not isinstance(hand, HandHandle) for hand in hands):
            raise ValueError("every hand must be a HandHandle")
        if len({hand.value for hand in hands}) != len(hands):
            raise ValueError("one hand may appear only once in a batched request")
        parsed_rows: list[list[int]] = []
        states = [self._state(hand) for hand in hands]
        for state, cards in zip(states, card_rows, strict=True):
            if self._owner(state) != CHANCE:
                raise ValueError("deal is only valid at a public chance node")
            parsed = [
                parse_card(card, first_card=self.game_config.geometry.first_card)
                for card in cards
            ]
            ref = torch.tensor([state.ref], dtype=torch.int64, device=self.device)
            street = int(self.backend.states.street[ref].to("cpu")[0])
            expected = 3 if street == 0 else 1
            if len(parsed) != expected:
                raise ValueError(f"street requires exactly {expected} public card(s)")
            board = self.backend.states.board[ref].to("cpu")[0].tolist()
            if len(set(parsed)) != len(parsed) or set(parsed) & {c for c in board if c >= 0}:
                raise ValueError("public cards must be unique")
            parsed_rows.append(parsed)
        for state, cards in zip(states, parsed_rows, strict=True):
            conflict = (
                (
                    self._combo_cards_device[:, :, None]
                    == torch.tensor(cards, device=self.device)[None, None, :]
                )
                .any(2)
                .any(1)
            )
            state.log_reach[:, conflict] = -torch.inf
            padded = cards + [-1] * (3 - len(cards))
            ref = torch.tensor([state.ref], dtype=torch.int32, device=self.device)
            successor = self.backend.step(
                ref,
                EventBatch(
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    {"cards": torch.tensor([padded], dtype=torch.int16, device=self.device)},
                ),
            )
            state.ref = int(successor.to("cpu")[0])
        return list(hands)


__all__ = [
    "Decision",
    "FullStrategy",
    "HandHandle",
    "PokerAction",
    "PokerInferenceSession",
    "parse_card",
]
