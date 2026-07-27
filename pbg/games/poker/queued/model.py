"""Authoritative two-partition queued poker model operation."""

from __future__ import annotations

from types import MappingProxyType
import torch
from torch import Tensor, nn

from pbg.queued.learner import ModelEvaluation
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

    def learner_operation(self, input_type: str):
        def operation(
            model_input: Tensor,
            _sampled_private_state: Tensor,
            _owner: Tensor,
        ) -> ModelEvaluation:
            return self.forward_partition(input_type, model_input)

        return operation


__all__ = ["PokerModelError", "QueuedPokerModel"]
