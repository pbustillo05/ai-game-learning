"""Authoritative two-partition queued poker model operation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping
import torch
from torch import Tensor, nn

from pbg.queued.learner import ModelEvaluation
from pbg.queued.contracts import RolloutMode, mode_schema
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.model.capacity import max_public_history_tokens
from pbg.games.poker.model.postflop import PostflopMasks, PostflopNet
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerModelConfig,
    PokerPublicBatch,
    PolicyOutput,
    PreflopMasks,
    PreflopNet,
)
from pbg.games.poker.model.primitives import MHA
from pbg.games.poker.queued.hooks import (
    FOLD,
    FOLDED,
    NLHEGeometry,
    PAD_CARD,
    PokerHistory,
    PokerState,
    PokerStateMachine,
)
from pbg.games.poker.queued.folded import unpack_folded_statistics


class PokerModelError(RuntimeError):
    """A partition model violated the queued model operation."""


def _fold_line_metadata(
    history: PokerHistory,
    status: Tensor,
    *,
    line_dim: int,
    stack_bb: float,
) -> Tensor:
    """Encode each folded seat's retained action line at its fold event."""

    batch, tokens = history.mask.shape
    players = status.shape[1]
    result = torch.zeros(
        (batch, players, line_dim),
        dtype=torch.float32,
        device=status.device,
    )
    position = torch.arange(tokens, device=status.device)[None]
    for seat in range(players):
        folder = status[:, seat].eq(FOLDED)
        owned = history.mask & history.actor.eq(seat)
        fold = owned & history.action.eq(FOLD)
        torch._assert_async(
            (~folder | fold.any(-1)).all(),
            "folded seat has no retained fold-line event",
        )
        last = torch.where(
            fold, position, torch.full_like(position, -1)
        ).max(-1).values
        gather = last.clamp_min(0)[:, None]
        through_fold = owned & position.le(last[:, None])
        call = through_fold & history.action.eq(1)
        raise_ = through_fold & history.action.eq(2)
        fields = (
            history.street.gather(1, gather).squeeze(1).float() / 3.0,
            through_fold.sum(-1, dtype=torch.float32) / max(tokens, 1),
            (history.amount * through_fold).sum(-1) / float(stack_bb),
            call.sum(-1, dtype=torch.float32),
            raise_.sum(-1, dtype=torch.float32),
            history.amount.gather(1, gather).squeeze(1) / float(stack_bb),
            history.pot_before.gather(1, gather).squeeze(1) / float(stack_bb),
            history.call_amount.gather(1, gather).squeeze(1) / float(stack_bb),
        )
        for column, value in enumerate(fields[:line_dim]):
            result[:, seat, column] = torch.where(
                folder, value, torch.zeros_like(value)
            )
    return result


def _combo_class(geometry: NLHEGeometry) -> Tensor:
    classes = []
    ranks = geometry.rank_count
    for first in range(geometry.first_card, 52):
        for second in range(first + 1, 52):
            rank_first = first // 4 - (13 - ranks)
            rank_second = second // 4 - (13 - ranks)
            high, low = max(rank_first, rank_second), min(rank_first, rank_second)
            high_position, low_position = ranks - 1 - high, ranks - 1 - low
            classes.append(
                high_position * ranks + low_position
                if rank_first == rank_second or first % 4 == second % 4
                else low_position * ranks + high_position
            )
    return torch.tensor(classes, dtype=torch.int64)


def _validate_output(
    evaluation: ModelEvaluation,
    *,
    batch: int,
    width: int,
    device: torch.device,
) -> None:
    if evaluation.value.shape != (batch, width) or evaluation.value.device != device:
        raise PokerModelError("partition value output has the wrong domain")
    expected = {"type_logits", "omega", "mu", "s", "iota"}
    if set(evaluation.policy_raw) != expected:
        raise PokerModelError("partition policy output has unknown or missing fields")
    prefix = (batch, width)
    if evaluation.policy_raw["type_logits"].shape != prefix + (3,):
        raise PokerModelError("partition type logits have the wrong domain")
    mixture_shape = evaluation.policy_raw["omega"].shape
    if (
        len(mixture_shape) != 3
        or mixture_shape[:2] != prefix
        or evaluation.policy_raw["mu"].shape != mixture_shape
        or evaluation.policy_raw["s"].shape != mixture_shape
        or evaluation.policy_raw["iota"].shape != prefix
    ):
        raise PokerModelError("partition raise outputs have the wrong domain")
    if any(value.device != device for value in evaluation.policy_raw.values()):
        raise PokerModelError("partition policy output changed device")


class QueuedPokerModel(nn.Module):
    """Own preflop/postflop parameter separation and natural-domain routing.

    Each supplied partition network consumes the direct retained model-input
    tensor and returns :class:`ModelEvaluation`.  Preflop operates on its
    natural $R^2$ class domain and is lifted once to evaluator combo order;
    every ``post_A`` family operates directly on the $H$ combo domain.
    """

    def __init__(
        self,
        geometry: NLHEGeometry,
        preflop: nn.Module,
        postflop: nn.Module,
    ) -> None:
        super().__init__()
        if preflop is postflop:
            raise ValueError("preflop and postflop parameter owners must be distinct")
        self.geometry = geometry
        self.preflop = preflop
        self.postflop = postflop
        self.register_buffer("combo_class", _combo_class(geometry), persistent=True)
        deck = DeckGeometry(geometry.rank_count)
        self.register_buffer(
            "combo_cards",
            torch.tensor(deck.combo_cards, dtype=torch.int64),
            persistent=False,
        )

    def configure_execution(
        self, attention_backend: str, query_chunk_size: int
    ) -> None:
        """Apply the admitted hardware execution profile to every attention."""

        for module in self.modules():
            if isinstance(module, MHA):
                module.configure_execution(
                    attention_backend, query_chunk_size
                )

    @staticmethod
    def _call_direct(module: nn.Module, model_input: Tensor) -> ModelEvaluation:
        result = module(model_input)
        if not isinstance(result, ModelEvaluation):
            raise PokerModelError("partition network must return ModelEvaluation")
        return result

    @staticmethod
    def _evaluation(result: PolicyOutput) -> ModelEvaluation:
        return ModelEvaluation(MappingProxyType(dict(result.raw)), result.v)

    def _state(self, model_input: Tensor) -> PokerState:
        combo_ids = torch.zeros(
            (model_input.shape[0], self.geometry.players),
            dtype=torch.int32,
            device=model_input.device,
        )
        return PokerStateMachine(self.geometry).state_from_model_input(
            model_input, combo_ids
        )

    def _public(
        self, state: PokerState, history: PokerHistory
    ) -> tuple[PokerPublicBatch, Tensor]:
        machine = PokerStateMachine(self.geometry)
        legality = machine.legality(state)
        raise_mask = legality.mask[:, 2:]
        raise_ticks = (
            torch.arange(
                1,
                self.geometry.stack_ticks + 1,
                dtype=torch.float32,
                device=state.device,
            )[None]
            * 0.5
        ).expand(state.rows, -1)
        lower = raise_ticks.masked_fill(~raise_mask, torch.inf).min(1).values
        upper = raise_ticks.masked_fill(~raise_mask, -torch.inf).max(1).values
        has_raise = raise_mask.any(1)
        lower = torch.where(has_raise, lower, torch.zeros_like(lower))
        upper = torch.where(has_raise, upper, torch.zeros_like(upper))
        actor = state.to_act.to(torch.int64)
        actor_committed = state.committed_street.gather(
            1, actor[:, None]
        ).squeeze(1)
        legality_summary = torch.stack(
            (
                legality.mask[:, 0].float(),
                legality.mask[:, 1].float(),
                raise_mask.float().mean(1),
            ),
            1,
        )
        public = PokerPublicBatch(
            street=state.street.to(torch.int64),
            board=torch.where(
                state.board_cards.eq(PAD_CARD),
                torch.full_like(state.board_cards, -1, dtype=torch.int64),
                state.board_cards.to(torch.int64),
            ),
            pot=state.committed.sum(1),
            stacks=state.stacks,
            committed=state.committed,
            to_act=actor,
            legality_summary=legality_summary,
            n_ticks=torch.full(
                (state.rows,),
                self.geometry.stack_ticks,
                dtype=torch.int64,
                device=state.device,
            ),
            raise_grid_bounds=torch.stack((lower, upper), 1),
            status=state.status.to(torch.int64),
            seat_mask=torch.ones_like(state.status, dtype=torch.bool),
            button=state.button.to(torch.int64),
            committed_street=state.committed_street,
            call_amount=(
                state.current_bet - actor_committed
            ).clamp_min(0.0),
            current_bet=state.current_bet,
            min_raise_to=lower,
            max_raise_to=upper,
            last_full_raise=state.last_full_raise,
            legal_action_mask=legality.mask,
            raise_ticks=raise_ticks,
            history=PackedPublicHistory(
                **{
                    name: getattr(history, name)
                    for name in history.__dataclass_fields__
                }
            ),
        )
        board = public.board
        compatible = ~(
            self.combo_cards[None, :, :, None].eq(
                board[:, None, None, :]
            )
            & board[:, None, None, :].ge(0)
        ).any(3).any(2)
        return public, compatible

    def _call_preflop(
        self, model_input: Tensor
    ) -> ModelEvaluation:
        state = self._state(model_input)
        log_reach, history = PokerStateMachine(
            self.geometry
        ).model_context(model_input)
        public, _compatible = self._public(state, history)
        finite = torch.isfinite(log_reach)
        gauge = log_reach.masked_fill(
            ~finite, -torch.inf
        ).max(2, keepdim=True).values
        weights = torch.exp(log_reach - gauge).masked_fill(~finite, 0.0)
        class_ranges = torch.zeros(
            (
                state.rows,
                self.geometry.players,
                self.geometry.rank_count**2,
            ),
            dtype=torch.float32,
            device=state.device,
        )
        classes = self.combo_class[None, None, :].expand_as(weights)
        class_ranges.scatter_add_(2, classes, weights)
        return self._evaluation(
            self.preflop(
                public,
                class_ranges,
                public.to_act,
                PreflopMasks(
                    seat_mask=public.seat_mask,
                    status=public.status,
                ),
            )
        )

    def _call_postflop(
        self, input_type: str, model_input: Tensor
    ) -> ModelEvaluation:
        state = self._state(model_input)
        log_reach, history = PokerStateMachine(
            self.geometry
        ).model_context(model_input)
        public, compatible = self._public(state, history)
        live = int(input_type[5:])
        seats = torch.arange(
            self.geometry.players, device=state.device
        )[None].expand(state.rows, -1)
        ordered = torch.where(
            state.status.ne(FOLDED), seats, self.geometry.players
        ).sort(1).values
        live_ids = ordered[:, :live]
        torch._assert_async(
            live_ids.lt(self.geometry.players).all(),
            "postflop family disagrees with live-seat state",
        )
        finite = torch.isfinite(log_reach)
        gauge = log_reach.masked_fill(
            ~finite, -torch.inf
        ).max(2, keepdim=True).values
        ranges = torch.exp(log_reach - gauge).masked_fill(~finite, 0.0)
        line_dim = getattr(self.postflop.cfg, "folder_line_dim", 0)
        summary = unpack_folded_statistics(
            self.geometry,
            beta_rank=getattr(self.postflop.cfg, "folder_beta_rank", 0),
            line_metadata=_fold_line_metadata(
                history,
                state.status,
                line_dim=line_dim,
                stack_bb=self.geometry.stack_bb,
            ),
            status=state.status,
            packed=model_input[:, self.geometry.base_model_input_width :].float(),
        )
        return self._evaluation(
            self.postflop(
                public,
                ranges,
                public.to_act,
                PostflopMasks(
                    compatibility=compatible,
                    seat_mask=public.seat_mask,
                    combo_mask=ranges.bool(),
                    status=public.status,
                    evaluator_summary=summary,
                    live_seat_ids=live_ids,
                ),
            )
        )

    def forward_partition(
        self, input_type: str, model_input: Tensor
    ) -> ModelEvaluation:
        if (
            model_input.device.type != "cuda"
            or model_input.ndim != 2
            or not model_input.is_floating_point()
        ):
            raise PokerModelError("queued poker model input must be CUDA [B,F]")
        batch = model_input.shape[0]
        if input_type == "pre":
            compact = (
                self._call_preflop(model_input)
                if isinstance(self.preflop, PreflopNet)
                else self._call_direct(self.preflop, model_input)
            )
            _validate_output(
                compact,
                batch=batch,
                width=self.geometry.rank_count**2,
                device=model_input.device,
            )
            index = self.combo_class
            raw = MappingProxyType(
                {name: value[:, index] for name, value in compact.policy_raw.items()}
            )
            result = ModelEvaluation(raw, compact.value[:, index])
        elif input_type.startswith("post_") and input_type[5:].isdigit():
            live = int(input_type[5:])
            if not 2 <= live <= self.geometry.players:
                raise PokerModelError("postflop family has an invalid live-seat count")
            if (
                self.geometry.folded_summary_width
                and not isinstance(self.postflop, PostflopNet)
            ):
                state = self._state(model_input)
                if not bool(
                    (
                        ~state.status.eq(FOLDED).any(-1)
                        | model_input[:, -1].gt(0.5)
                    ).all()
                ):
                    raise PokerModelError(
                        "postflop folded seats require sealed folded summaries"
                    )
            result = (
                self._call_postflop(input_type, model_input)
                if isinstance(self.postflop, PostflopNet)
                else self._call_direct(self.postflop, model_input)
            )
        else:
            raise PokerModelError(f"unknown poker input partition {input_type!r}")
        _validate_output(
            result,
            batch=batch,
            width=self.geometry.private_hands,
            device=model_input.device,
        )
        return result

    def forward_natural_partition(
        self, input_type: str, model_input: Tensor
    ) -> ModelEvaluation:
        """Return K-wide preflop or H-wide postflop learner outputs."""

        if input_type == "pre":
            result = (
                self._call_preflop(model_input)
                if isinstance(self.preflop, PreflopNet)
                else self._call_direct(self.preflop, model_input)
            )
            _validate_output(
                result,
                batch=model_input.shape[0],
                width=self.geometry.rank_count**2,
                device=model_input.device,
            )
            return result
        if input_type.startswith("post_") and input_type[5:].isdigit():
            result = (
                self._call_postflop(input_type, model_input)
                if isinstance(self.postflop, PostflopNet)
                else self._call_direct(self.postflop, model_input)
            )
            _validate_output(
                result,
                batch=model_input.shape[0],
                width=self.geometry.private_hands,
                device=model_input.device,
            )
            return result
        raise PokerModelError(f"unknown poker input partition {input_type!r}")

    def forward_sequential_partition(
        self, input_type: str, model_input: Tensor
    ) -> ModelEvaluation:
        """Phase 6 rollout path with repository-owned folded summaries."""

        return self.forward_partition(input_type, model_input)

    def learner_operation(
        self,
        input_type: str,
        model_input: Tensor,
        _sampled_private_state: Tensor,
        _owner: Tensor,
    ) -> ModelEvaluation:
        return self.forward_sequential_partition(input_type, model_input)

    def rollout_operation(
        self,
        input_type: str,
        model_input: Tensor,
        _private_state: Tensor,
        _owner: Tensor,
    ) -> ModelEvaluation:
        return self.forward_sequential_partition(input_type, model_input)

    def sequential_rollout_operation(
        self,
        input_type: str,
        model_input: Tensor,
        _private_state: Tensor,
        _owner: Tensor,
    ) -> ModelEvaluation:
        return self.forward_sequential_partition(input_type, model_input)

    def rb_learner_operation(
        self,
        input_type: str,
        model_input: Tensor,
        _private_state: Tensor,
        _owner: Tensor,
    ) -> ModelEvaluation:
        return self.forward_natural_partition(input_type, model_input)


def build_model_from_config(
    scientific: Mapping[str, object],
    device: torch.device | str,
) -> QueuedPokerModel:
    """Reconstruct the selected queued architecture from scientific state.

    The offline converter intentionally removes architecture values that are
    functions of other scientific values.  This factory is the sole inverse:
    it derives the action width, sequential head selection, and lossless
    history capacity before constructing the exact parameter owners.
    """

    mode = scientific.get("mode")
    if not isinstance(mode, Mapping):
        raise ValueError("scientific.mode must select a sequential mode")
    try:
        schema = mode_schema(**dict(mode))
    except (TypeError, ValueError) as exc:
        raise ValueError("scientific.mode is not a closed queued mode") from exc
    if schema.rollout_mode is not RolloutMode.SEQUENTIAL:
        raise ValueError("scientific.mode must select sequential rollout")
    game = scientific.get("game")
    model = scientific.get("model")
    if not isinstance(game, Mapping) or not isinstance(model, Mapping):
        raise ValueError(
            "scientific configuration must contain game and model objects"
        )
    game_values = dict(game)
    if "players" in game_values:
        players = game_values["players"]
        if "n_players" in game_values:
            raise ValueError("scientific.game contains duplicate player counts")
    else:
        players = game_values.get("n_players")
    try:
        geometry = NLHEGeometry(
            players=int(players),
            stack_bb=float(game_values["stack_bb"]),
            rank_count=int(game_values["rank_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scientific.game is not a valid NLHE geometry") from exc

    model_values = dict(model)
    if "epsilon" in model_values:
        if "eps" in model_values:
            raise ValueError("scientific.model contains duplicate epsilon values")
        model_values["eps"] = model_values.pop("epsilon")
    derived_action_width = 3 + int(model_values.get("m_raise", 4))
    if "k_act" in model_values and model_values["k_act"] != derived_action_width:
        raise ValueError("scientific.model.k_act disagrees with m_raise")
    model_values["k_act"] = derived_action_width
    if (
        "use_branching_heads" in model_values
        and model_values["use_branching_heads"] is not False
    ):
        raise ValueError(
            "scientific.model.use_branching_heads is invalid for sequential mode"
        )
    model_values["use_branching_heads"] = False
    required_history = max_public_history_tokens(
        geometry.players, geometry.stack_bb
    )
    if (
        "max_history_tokens" in model_values
        and int(model_values["max_history_tokens"]) != required_history
    ):
        raise ValueError(
            "scientific.model.max_history_tokens is not the derived exact bound"
        )
    model_values["max_history_tokens"] = required_history
    model_config = PokerModelConfig.from_dict(model_values)
    if model_config.max_players < geometry.players:
        raise ValueError("scientific.model.max_players does not cover all seats")
    if model_config.folder_beta_rank > geometry.deck_cards:
        raise ValueError(
            "scientific.model.folder_beta_rank exceeds the compact deck"
        )

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError(
            "Phase 5 queued poker model construction requires CUDA; "
            "a supported MPS inference composition lands at cutover"
        )
    if target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    deck = DeckGeometry(geometry.rank_count)
    result = QueuedPokerModel(
        geometry,
        PreflopNet(model_config, deck),
        PostflopNet(model_config, deck),
    ).to(target)
    if any(parameter.device != target for parameter in result.parameters()):
        raise PokerModelError("queued poker model construction changed device")
    return result


__all__ = ["PokerModelError", "QueuedPokerModel", "build_model_from_config"]
