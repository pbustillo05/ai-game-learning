"""Authoritative two-partition queued poker model operation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping
import torch
from torch import Tensor, nn

from pbg.queued.learner import ModelEvaluation
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.model.capacity import max_public_history_tokens
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet
from pbg.games.poker.queued.hooks import NLHEGeometry


class PokerModelError(RuntimeError):
    """A partition model violated the queued model operation."""


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

    @staticmethod
    def _call(module: nn.Module, model_input: Tensor) -> ModelEvaluation:
        result = module(model_input)
        if not isinstance(result, ModelEvaluation):
            raise PokerModelError("partition network must return ModelEvaluation")
        return result

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
            compact = self._call(self.preflop, model_input)
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
            result = self._call(self.postflop, model_input)
        else:
            raise PokerModelError(f"unknown poker input partition {input_type!r}")
        _validate_output(
            result,
            batch=batch,
            width=self.geometry.private_hands,
            device=model_input.device,
        )
        return result

    def learner_operation(
        self,
        input_type: str,
        model_input: Tensor,
        _sampled_private_state: Tensor,
        _owner: Tensor,
    ) -> ModelEvaluation:
        return self.forward_partition(input_type, model_input)


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
    expected_mode = {
        "training_regime": "self_play",
        "rollout_mode": "sequential",
        "private_estimator": "sampled",
    }
    if not isinstance(mode, Mapping) or dict(mode) != expected_mode:
        raise ValueError(
            "scientific.mode must select self_play/sequential/sampled"
        )
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
    if target.type not in ("cuda", "mps"):
        raise ValueError("queued poker model construction requires cuda or mps")
    if target.type == "cuda" and target.index is None:
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
