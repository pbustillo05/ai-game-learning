"""Stateful, model-only NLHE inference on CPU or Apple MPS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from pbg.core.game import CHANCE, PRETERM, TERMINAL
from pbg.games.poker.adapter import PokerGame, PokerPolicyAdapter
from pbg.games.poker.bundle import build_model_from_config
from pbg.games.poker.engine import (
    CHECK_CALL,
    FOLD,
    FOLDED,
    RAISE_OFFSET,
    NLHEEngine,
    PokerConfig,
)
from pbg.games.poker.folded import (
    FoldSnapshotBatch,
    PartitionSummaryBatch,
    PokerGPUPartitionAdapter,
)
from pbg.games.poker.model.postflop import PostflopMasks
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerPublicBatch,
    PreflopMasks,
)
from pbg.games.poker.model.primitives import MHA
from pbg.inference_artifact import load_inference_source


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
class _FoldRecord:
    board_at_fold: Tensor  # uint8 [5], CPU
    line_meta: Tensor  # float32 [8], CPU


@dataclass
class _HandState:
    ref: int
    log_reach: Tensor  # float32 [P,H], model device
    folds: dict[int, _FoldRecord]
    fold_version: int = 0


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
        device: str | torch.device = "auto",
        precision: str = "auto",
        mode: str = "auto",
        max_batch_rows: int | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        if precision not in {"auto", "fp32", "fp16"}:
            raise ValueError("precision must be 'auto', 'fp32', or 'fp16'")
        if mode not in {"auto", "latency", "throughput"}:
            raise ValueError("mode must be 'auto', 'latency', or 'throughput'")
        if max_batch_rows is not None and (
            isinstance(max_batch_rows, bool) or max_batch_rows <= 0
        ):
            raise ValueError("max_batch_rows must be positive when supplied")
        if (
            self.device.type == "mps"
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        ):
            raise RuntimeError(
                "PYTORCH_ENABLE_MPS_FALLBACK=1 hides unsupported model operations; "
                "unset it before creating an MPS inference session"
            )
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
        self.engine = NLHEEngine(self.game_config)
        self._game = PokerGame(
            self.engine,
            profile_sampler=object(),
            showdown_evaluator=object(),
            partition_adapter=object(),
        )
        self.model = build_model_from_config(scientific, torch.device("cpu"))
        self.model.load_state_dict(source["model"], strict=True)
        self.model.eval().requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, MHA):
                module.configure_execution(
                    "sdpa" if self.device.type == "mps" else "auto", 128
                )
        self.model.to(self.device)
        self.adapter = PokerPolicyAdapter(self.game_config.n_ticks)
        self.partition_adapter = PokerGPUPartitionAdapter(
            beta_rank=self.model.postflop.cfg.folder_beta_rank,
            rank_count=self.game_config.rank_count,
        )
        self.iteration = int(source["iteration"])
        self.source_checkpoint_sha256 = str(source["source_checkpoint_sha256"])
        self._hands: dict[int, _HandState] = {}
        self._next_hand_id = 0
        self._partition_cache: dict[
            tuple[int, tuple[int, ...], int], PartitionSummaryBatch
        ] = {}
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
        device: str | torch.device = "auto",
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

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        resolved = torch.device(device)
        if resolved.type not in {"cpu", "mps"}:
            raise ValueError("inference device must be 'auto', 'cpu', or 'mps'")
        if resolved.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            reason = (
                "unavailable on this Mac" if built else "absent from this PyTorch build"
            )
            raise RuntimeError(f"MPS is {reason}")
        return resolved

    def new_hand(self, button: int = 0) -> HandHandle:
        """Create one inference hand with uniform initial private ranges."""

        handle = HandHandle(self._next_hand_id)
        self._next_hand_id += 1
        self._hands[handle.value] = _HandState(
            ref=self.engine.new_hand(button),
            log_reach=torch.zeros(
                (self.game_config.n_players, self.game_config.geometry.n_hands),
                dtype=torch.float32,
                device=self.device,
            ),
            folds={},
        )
        return handle

    def new_hands(self, buttons: Sequence[int]) -> list[HandHandle]:
        """Create a batch of independent hands in input order."""

        return [self.new_hand(button) for button in buttons]

    def _state(self, hand: HandHandle) -> _HandState:
        if not isinstance(hand, HandHandle) or hand.value not in self._hands:
            raise ValueError("unknown inference hand handle")
        return self._hands[hand.value]

    def _owner(self, state: _HandState) -> int:
        return int(
            self.engine.owner_of(torch.tensor([state.ref], dtype=torch.int32))[0]
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
        refs = torch.tensor([ref], dtype=torch.int32)
        _, legal = self.engine.legal_events(refs)
        state = self.engine.states.state(ref)
        to_call = max(0.0, state.current_bet - state.committed_street[state.actor])
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
                target = state.committed_street[state.actor] + state.stacks[state.actor]
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
        if not 0 <= event < self.engine.e_max or not bool(legal[0, event]):
            raise ValueError(f"event {event} is illegal in the current public state")
        return int(event)

    def _raise_event_for(self, ref: int, target: float) -> int:
        if not math.isfinite(target):
            raise ValueError("raise-to amount must be finite")
        refs = torch.tensor([ref], dtype=torch.int32)
        targets = self.engine.raise_targets(refs)[0]
        _, legal = self.engine.legal_events(refs)
        matches = torch.isclose(targets, torch.tensor(target), atol=1e-6, rtol=0.0)
        columns = (matches & legal[0, RAISE_OFFSET:]).nonzero(as_tuple=False)
        if columns.numel() != 1:
            raise ValueError(
                f"raise-to {target:g} bb is not legal in the current state"
            )
        return RAISE_OFFSET + int(columns[0, 0])

    def _actions_for(self, ref: int) -> tuple[PokerAction, ...]:
        refs = torch.tensor([ref], dtype=torch.int32)
        _, legal = self.engine.legal_events(refs)
        targets = self.engine.raise_targets(refs)[0]
        state = self.engine.states.state(ref)
        to_call = max(0.0, state.current_bet - state.committed_street[state.actor])
        maximum = state.committed_street[state.actor] + state.stacks[state.actor]
        actions: list[PokerAction] = []
        for event in legal[0].nonzero(as_tuple=False).squeeze(1).tolist():
            if event == FOLD:
                actions.append(PokerAction("fold", event))
            elif event == CHECK_CALL:
                actions.append(
                    PokerAction("call" if to_call > 1e-6 else "check", event)
                )
            else:
                target = float(targets[event - RAISE_OFFSET])
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
            + self.engine.e_max
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
        if self.device.type == "mps":
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
        handles: Sequence[HandHandle],
        states: Sequence[_HandState],
        pub_cpu: PokerPublicBatch,
        weights: Tensor,
        live: int,
    ) -> PartitionSummaryBatch:
        keys = [
            (
                handle.value,
                tuple(int(card) for card in self.engine.states.board[state.ref]),
                state.fold_version,
            )
            for handle, state in zip(handles, states, strict=True)
        ]
        missing = [
            index for index, key in enumerate(keys) if key not in self._partition_cache
        ]
        if missing:
            status = pub_cpu.status
            if status is None:
                raise RuntimeError("postflop public state is missing seat status")
            weights_cpu = (
                weights[missing].detach().to(device="cpu", dtype=torch.float32)
            )
            ordered_seats = [
                sorted(
                    range(self.game_config.n_players),
                    key=lambda seat: int(status[index, seat] == FOLDED),
                )
                for index in missing
            ]
            live_weights = torch.stack(
                [
                    weights_cpu[row, seats[:live]]
                    for row, seats in enumerate(ordered_seats)
                ]
            )
            folder_count = self.game_config.n_players - live
            if folder_count:
                folded_weights = torch.stack(
                    [
                        weights_cpu[row, seats[live:]]
                        for row, seats in enumerate(ordered_seats)
                    ]
                )
                fold_boards = torch.stack(
                    [
                        torch.stack(
                            [
                                states[index].folds[seat].board_at_fold
                                for seat in seats[live:]
                            ]
                        )
                        for index, seats in zip(missing, ordered_seats, strict=True)
                    ]
                )
                line_meta = torch.stack(
                    [
                        torch.stack(
                            [
                                states[index].folds[seat].line_meta
                                for seat in seats[live:]
                            ]
                        )
                        for index, seats in zip(missing, ordered_seats, strict=True)
                    ]
                )
            else:
                n_hands = self.game_config.geometry.n_hands
                folded_weights = torch.empty(
                    (len(missing), 0, n_hands), dtype=torch.float32
                )
                fold_boards = torch.empty((len(missing), 0, 5), dtype=torch.uint8)
                line_meta = torch.empty((len(missing), 0, 8), dtype=torch.float32)
            folder_seats = torch.tensor(
                [seats[live:] for seats in ordered_seats], dtype=torch.int64
            )
            folded = FoldSnapshotBatch(
                reach_weights=folded_weights.contiguous(),
                board_at_fold=fold_boards.contiguous(),
                line_meta=line_meta.contiguous(),
                folder_mask=torch.ones((len(missing), folder_count), dtype=torch.bool),
                seat_ids=folder_seats.contiguous(),
            )
            computed = self.partition_adapter.summarize(
                board_cards=pub_cpu.board[missing].to(torch.int16).contiguous(),
                live_range_weights=live_weights.contiguous(),
                folded=folded,
            )
            for row, index in enumerate(missing):
                self._partition_cache[keys[index]] = PartitionSummaryBatch(
                    fields={
                        name: value[row : row + 1].clone()
                        for name, value in computed.fields.items()
                    },
                    metadata=dict(computed.metadata),
                )
        first = self._partition_cache[keys[0]]
        return PartitionSummaryBatch(
            fields={
                name: torch.cat(
                    [self._partition_cache[key].fields[name] for key in keys], dim=0
                )
                for name in first.fields
            },
            metadata=dict(first.metadata),
        )

    def _evaluate_full_device(
        self, handles: Sequence[HandHandle]
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor]:
        if not handles:
            raise ValueError("at least one hand is required")
        states = [self._require_decision(hand) for hand in handles]
        refs = torch.tensor([state.ref for state in states], dtype=torch.int32)
        legal_ids_cpu, legal_mask_cpu = self.engine.legal_events(refs)
        batch = len(handles)
        n_hands = self.game_config.geometry.n_hands
        e_max = self.engine.e_max
        probabilities = torch.empty(
            (batch, n_hands, e_max), dtype=torch.float32, device=self.device
        )
        values = torch.empty((batch, n_hands), dtype=torch.float32, device=self.device)
        seat_values: Tensor | None = None

        groups: dict[tuple[int, int, int], list[int]] = {}
        for index, state in enumerate(states):
            public = self.engine.states.state(state.ref)
            live = sum(status != FOLDED for status in public.status)
            history = len(public.public_history)
            history_bucket = 0 if history == 0 else 1 << (history - 1).bit_length()
            groups.setdefault((public.street, live, history_bucket), []).append(index)

        with torch.inference_mode():
            for family, positions in groups.items():
                physical = self._physical_batch_rows(family, len(positions))
                for start in range(0, len(positions), physical):
                    selected = positions[start : start + physical]
                    selected_handles = [handles[index] for index in selected]
                    selected_states = [states[index] for index in selected]
                    selected_refs = torch.tensor(
                        [state.ref for state in selected_states], dtype=torch.int32
                    )
                    pub_cpu = self._game.encode_public(selected_refs)
                    pub = _move_public(pub_cpu, self.device)
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
                            summary_cpu = self._partition_summary(
                                selected_handles,
                                selected_states,
                                pub_cpu,
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
                                    evaluator_summary=_move_summary(
                                        summary_cpu, self.device
                                    ),
                                    live_seat_ids=live_ids,
                                    compact_size=math.comb(
                                        self.game_config.geometry.n_cards
                                        - (family[0] + 2),
                                        2,
                                    ),
                                ),
                            )
                        legal_ids = legal_ids_cpu[selected].to(self.device)
                        legal_mask = legal_mask_cpu[selected].to(self.device)
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
            legal_ids_cpu.to(self.device),
            legal_mask_cpu.to(self.device),
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
            board = self.engine.states.board[state.ref]
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
        boards = torch.tensor(
            [self.engine.states.board[self._state(hand).ref] for hand in selected],
            dtype=torch.int64,
            device=self.device,
        )
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
            public = self.engine.states.state(state.ref)
            actor = public.actor
            selected = probability[row, :, event].clamp_min(self.adapter.eps)
            state.log_reach[actor] += torch.log(selected).float()
            finite = torch.isfinite(state.log_reach[actor])
            gauge = state.log_reach[actor].masked_fill(~finite, -torch.inf).amax()
            state.log_reach[actor] -= gauge
            if event == FOLD:
                board = torch.tensor(
                    public.board + (-1,) * (5 - len(public.board)),
                    dtype=torch.int16,
                )
                board_at_fold = torch.where(
                    board < 0, torch.full_like(board, 255), board
                ).to(torch.uint8)
                to_call = max(
                    0.0,
                    public.current_bet - public.committed_street[actor],
                )
                line = torch.tensor(
                    [
                        public.street,
                        public.committed_total[actor],
                        public.committed_street[actor],
                        sum(public.committed_total),
                        to_call,
                        public.n_raises,
                        len(public.public_history),
                        len(public.board),
                    ],
                    dtype=torch.float32,
                )
                state.folds[actor] = _FoldRecord(board_at_fold, line)
                state.fold_version += 1
            successor = self.engine.step(
                torch.tensor([state.ref], dtype=torch.int32),
                torch.tensor([event], dtype=torch.int32),
            )
            state.ref = int(successor[0])
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
            public = self.engine.states.state(state.ref)
            expected = 3 if public.street == 0 else 1
            if len(parsed) != expected:
                raise ValueError(f"street requires exactly {expected} public card(s)")
            if len(set(parsed)) != len(parsed) or set(parsed) & set(public.board):
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
            successor = self.engine.deal(
                torch.tensor([state.ref], dtype=torch.int32),
                torch.tensor([cards], dtype=torch.int16),
            )
            state.ref = int(successor[0])
        return list(hands)


__all__ = [
    "Decision",
    "FullStrategy",
    "HandHandle",
    "PokerAction",
    "PokerInferenceSession",
    "parse_card",
]
