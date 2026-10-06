"""Authoritative two-partition queued poker model operation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import Tensor, nn

from pbg.core.contracts import mode_schema
from pbg.core.learner import ModelEvaluation
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.model.capacity import max_public_history_tokens
from pbg.games.poker.model.shared import SharedPokerParameters
from pbg.games.poker.model.postflop import PostflopMasks, PostflopEncoder
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerModelConfig,
    PokerPublicBatch,
    PolicyOutput,
    PreflopMasks,
    PreflopEncoder,
)
from pbg.games.poker.model.primitives import MHA, accelerator_assert
from pbg.games.poker.queued.folded import unpack_folded_statistics
from pbg.games.poker.queued.hooks import (
    FOLD,
    FOLDED,
    PAD_CARD,
    NLHEGeometry,
    PokerHistory,
    PokerState,
    PokerStateMachine,
)


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

    _batch, tokens = history.mask.shape
    players = status.shape[1]
    position = torch.arange(tokens, device=status.device)[None]
    seat_rows: list[Tensor] = []
    for seat in range(players):
        folder = status[:, seat].eq(FOLDED)
        owned = history.mask & history.actor.eq(seat)
        fold = owned & history.action.eq(FOLD)
        accelerator_assert(
            (~folder | fold.any(-1)).all(),
            "folded seat has no retained fold-line event",
        )
        last = torch.where(fold, position, torch.full_like(position, -1)).max(-1).values
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
        seat_rows.append(
            torch.stack(
                tuple(
                    torch.where(folder, value, torch.zeros_like(value))
                    for value in fields[:line_dim]
                ),
                -1,
            )
            if line_dim
            else history.amount[:, :0]
        )
    return torch.stack(tuple(seat_rows), 1)


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
    if set(evaluation.public_heads).difference({"trace", "cost"}):
        raise PokerModelError("partition public output has unknown fields")
    if evaluation.public_heads and (
        set(evaluation.public_heads) != {"trace", "cost"}
        or any(
            value.shape != (batch,) or value.device != device
            for value in evaluation.public_heads.values()
        )
    ):
        raise PokerModelError("partition trace/cost outputs have the wrong shape")


class QueuedPokerModel(nn.Module):
    """One shared model, routed through native preflop/postflop encoders."""

    def __init__(self, geometry: NLHEGeometry, config: PokerModelConfig) -> None:
        super().__init__()
        self.geometry = geometry
        self.shared = SharedPokerParameters(config, geometry.deck_cards)
        deck = DeckGeometry(geometry.rank_count)
        self.pre_encoder = PreflopEncoder(config, deck)
        self.post_encoder = PostflopEncoder(config, deck)
        self.register_buffer("combo_class", _combo_class(geometry), persistent=True)
        deck = DeckGeometry(geometry.rank_count)
        self.register_buffer(
            "combo_cards",
            torch.tensor(deck.combo_cards, dtype=torch.int64),
            persistent=False,
        )

    @property
    def replica_shared_buffers(self) -> tuple[str, ...]:
        """Every model buffer is immutable deck/class geometry."""

        return tuple(name for name, _ in self.named_buffers())

    def configure_execution(
        self, attention_backend: str, query_chunk_size: int | None = None
    ) -> None:
        """Apply the admitted hardware execution profile to every attention."""

        for module in self.modules():
            if isinstance(module, MHA):
                module.configure_execution(attention_backend, query_chunk_size)

    @staticmethod
    def _evaluation(result: PolicyOutput) -> ModelEvaluation:
        public = (
            MappingProxyType({})
            if result.t is None and result.c is None
            else MappingProxyType({"trace": result.t, "cost": result.c})
        )
        if (result.t is None) != (result.c is None):
            raise PokerModelError("partition returned only one public branch head")
        return ModelEvaluation(MappingProxyType(dict(result.raw)), result.v, public)

    def _state(self, input_type: str, model_input: Tensor) -> PokerState:
        combo_ids = torch.zeros(
            (model_input.shape[0], self.geometry.players),
            dtype=torch.int32,
            device=model_input.device,
        )
        return PokerStateMachine(self.geometry).state_from_model_input(
            input_type, model_input, combo_ids
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
        actor_committed = state.committed_street.gather(1, actor[:, None]).squeeze(1)
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
            call_amount=(state.current_bet - actor_committed).clamp_min(0.0),
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
            self.combo_cards[None, :, :, None].eq(board[:, None, None, :])
            & board[:, None, None, :].ge(0)
        ).any(3).any(2)
        return public, compatible

    def _call_preflop(self, model_input: Tensor, selected_private: Tensor | None = None) -> ModelEvaluation:
        state = self._state("pre", model_input)
        log_reach, history = PokerStateMachine(self.geometry).model_context(
            "pre", model_input
        )
        public, _compatible = self._public(state, history)
        finite = torch.isfinite(log_reach)
        gauge = log_reach.masked_fill(~finite, -torch.inf).max(2, keepdim=True).values
        class_ranges = torch.exp(log_reach - gauge).masked_fill(~finite, 0.0)
        return self._evaluation(
            self.pre_encoder(
                self.shared, public,
                class_ranges,
                public.to_act,
                PreflopMasks(
                    seat_mask=public.seat_mask,
                    status=public.status,
                ),
                selected_private=selected_private,
            )
        )

    def _call_postflop(self, input_type: str, model_input: Tensor, selected_private: Tensor | None = None) -> ModelEvaluation:
        state = self._state(input_type, model_input)
        log_reach, history = PokerStateMachine(self.geometry).model_context(
            input_type, model_input
        )
        public, compatible = self._public(state, history)
        live = int(input_type[5:])
        seats = torch.arange(self.geometry.players, device=state.device)[None].expand(
            state.rows, -1
        )
        ordered = (
            torch.where(state.status.ne(FOLDED), seats, self.geometry.players)
            .sort(1)
            .values
        )
        live_ids = ordered[:, :live]
        accelerator_assert(
            live_ids.lt(self.geometry.players).all(),
            "postflop family disagrees with live-seat state",
        )
        finite = torch.isfinite(log_reach)
        gauge = log_reach.masked_fill(~finite, -torch.inf).max(2, keepdim=True).values
        ranges = torch.exp(log_reach - gauge).masked_fill(~finite, 0.0)
        line_dim = getattr(self.post_encoder.cfg, "folder_line_dim", 0)
        summary = unpack_folded_statistics(
            self.geometry,
            beta_rank=getattr(self.post_encoder.cfg, "folder_beta_rank", 0),
            line_metadata=_fold_line_metadata(
                history,
                state.status,
                line_dim=line_dim,
                stack_bb=self.geometry.stack_bb,
            ),
            status=state.status,
            packed=model_input[
                :, self.geometry.postflop_base_model_input_width :
            ].float(),
        )
        return self._evaluation(
            self.post_encoder(
                self.shared, public,
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
                selected_private=selected_private,
            )
        )

    def forward_partition(
        self, input_type: str, model_input: Tensor,
        selected_private: Tensor | None = None,
    ) -> ModelEvaluation:
        if (
            model_input.device.type not in ("cuda", "mps")
            or model_input.ndim != 2
            or not model_input.is_floating_point()
        ):
            raise PokerModelError(
                "queued poker model input must be accelerator [B,F]"
            )
        batch = model_input.shape[0]
        if input_type == "pre":
            result = self._call_preflop(model_input, selected_private)
            _validate_output(
                result,
                batch=batch,
                width=self.geometry.private_classes if selected_private is None else 1,
                device=model_input.device,
            )
        elif input_type.startswith("post_") and input_type[5:].isdigit():
            live = int(input_type[5:])
            if not 2 <= live <= self.geometry.players:
                raise PokerModelError("postflop family has an invalid live-seat count")
            result = self._call_postflop(input_type, model_input, selected_private)
        else:
            raise PokerModelError(f"unknown poker input partition {input_type!r}")
        if input_type != "pre":
            _validate_output(
                result,
                batch=batch,
                width=self.geometry.private_hands if selected_private is None else 1,
                device=model_input.device,
            )
        return result

    def forward_natural_partition(
        self, input_type: str, model_input: Tensor
    ) -> ModelEvaluation:
        """Return K-wide preflop or H-wide postflop learner outputs."""

        return self.forward_partition(input_type, model_input)

    def forward_sequential_partition(
        self, input_type: str, model_input: Tensor
    ) -> ModelEvaluation:
        """Phase 6 rollout path with repository-owned folded summaries."""

        return self.forward_partition(input_type, model_input)

    def learner_operation(
        self,
        input_type: str,
        model_input: Tensor,
        sampled_private_state: Tensor,
        owner: Tensor,
    ) -> ModelEvaluation:
        owner64 = owner.to(torch.int64)
        combo = sampled_private_state.gather(
            1, owner64[:, None]
        ).squeeze(1).to(torch.int64)
        index = self.combo_class.index_select(0, combo) if input_type == "pre" else combo
        return self.forward_partition(input_type, model_input, index)

    def rollout_operation(
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

    def forward(
        self,
        input_type: str,
        model_input: Tensor,
        _private_state: Tensor,
        _owner: Tensor,
        *,
        natural_domain: bool = False,
    ) -> ModelEvaluation:
        """Functional-call entry shared by replica rollout/reference inference."""

        if type(natural_domain) is not bool:
            raise TypeError("natural_domain must be bool")
        return (
            self.forward_natural_partition(input_type, model_input)
            if natural_domain
            else self.forward_sequential_partition(input_type, model_input)
        )


def build_model_from_config(
    scientific: Mapping[str, object],
    device: torch.device | str,
) -> QueuedPokerModel:
    """Reconstruct the selected queued architecture from scientific state.

    Architecture values that are functions of other scientific values are
    derived here rather than accepted as duplicated inputs. This factory
    derives the action width, sequential head selection, and lossless history
    capacity before constructing the exact parameter owners.
    """

    mode = scientific.get("mode")
    if not isinstance(mode, Mapping):
        raise ValueError(  # noqa: TRY004 - stable configuration error contract
            "scientific.mode must select a sequential mode"
        )
    try:
        schema = mode_schema(**dict(mode))
    except (TypeError, ValueError) as exc:
        raise ValueError("scientific.mode is not a closed queued mode") from exc
    game = scientific.get("game")
    model = scientific.get("model")
    if not isinstance(game, Mapping) or not isinstance(model, Mapping):
        raise ValueError(  # noqa: TRY004 - stable configuration error contract
            "scientific configuration must contain game and model objects"
        )
    game_values = dict(game)
    if "n_players" in game_values:
        raise ValueError("scientific.game.n_players is not supported")
    try:
        geometry = NLHEGeometry(
            players=int(game_values["players"]),
            stack_bb=float(game_values["stack_bb"]),
            rank_count=int(game_values["rank_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scientific.game is not a valid NLHE geometry") from exc

    model_values = dict(model)
    layers = model_values.get("layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise ValueError("scientific.model.layers must be a positive integer; split-model artifacts are unsupported")
    raises = model_values.pop("raises", None)
    if not isinstance(raises, Mapping) or set(raises) != {
        "components",
        "minimum_scale",
    }:
        raise ValueError("scientific.model.raises is required and must be closed")
    model_values["m_raise"] = raises["components"]
    model_values["s_min"] = raises["minimum_scale"]
    if "seat_capacity" in model_values:
        model_values["max_players"] = model_values.pop("seat_capacity")
    else:
        model_values.setdefault("max_players", geometry.players)
    forbidden_derived = (
        "eps",
        "k_act",
        "use_branching_heads",
        "max_history_tokens",
    )
    stale = next((name for name in forbidden_derived if name in model_values), None)
    if stale is not None:
        raise ValueError(f"scientific.model.{stale} is not supported")
    try:
        model_values["eps"] = model_values.pop("epsilon")
    except KeyError as exc:
        raise ValueError("scientific.model.epsilon is required") from exc
    derived_action_width = 3 + int(model_values.get("m_raise", 4))
    model_values["k_act"] = derived_action_width
    expected_branching_heads = schema.rollout_mode.is_branched
    model_values["use_branching_heads"] = expected_branching_heads
    required_history = max_public_history_tokens(geometry.players, geometry.stack_bb)
    model_values["max_history_tokens"] = required_history
    model_config = PokerModelConfig.from_dict(model_values)
    if model_config.max_players < geometry.players:
        raise ValueError("scientific.model.max_players does not cover all seats")
    if model_config.folder_beta_rank > geometry.deck_cards:
        raise ValueError("scientific.model.folder_beta_rank exceeds the compact deck")

    target = torch.device(device)
    if target.type not in ("cuda", "mps"):
        raise ValueError("queued poker model construction requires CUDA or MPS")
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    result = QueuedPokerModel(geometry, model_config).to(target)
    if any(parameter.device != target for parameter in result.parameters()):
        raise PokerModelError("queued poker model construction changed device")
    return result


__all__ = ["PokerModelError", "QueuedPokerModel", "build_model_from_config"]
