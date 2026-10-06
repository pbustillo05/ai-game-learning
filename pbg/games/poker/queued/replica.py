"""Poker binding for the generic queued replica inference owner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from torch import Tensor, nn

from pbg.core.learner import ModelEvaluation
from pbg.core.replica import (
    FunctionalReplicaInference,
    ReplicaModuleOwner,
)


def encode_model_evaluation(evaluation: object) -> Mapping[str, Tensor]:
    if not isinstance(evaluation, ModelEvaluation):
        raise TypeError("poker replica inference requires ModelEvaluation")
    return MappingProxyType(
        {
            **{
                f"policy.{name}": value for name, value in evaluation.policy_raw.items()
            },
            "value": evaluation.value,
            **{
                f"public.{name}": value
                for name, value in evaluation.public_heads.items()
            },
        }
    )


def decode_model_evaluation(values: Mapping[str, Tensor]) -> ModelEvaluation:
    policy = {
        name.removeprefix("policy."): value
        for name, value in values.items()
        if name.startswith("policy.")
    }
    public = {
        name.removeprefix("public."): value
        for name, value in values.items()
        if name.startswith("public.")
    }
    if "value" not in values or not policy:
        raise ValueError("encoded poker model output is incomplete")
    if set(values) != {
        "value",
        *(f"policy.{name}" for name in policy),
        *(f"public.{name}" for name in public),
    }:
        raise ValueError("encoded poker model output has unknown fields")
    return ModelEvaluation(
        MappingProxyType(policy),
        values["value"],
        MappingProxyType(public),
    )


class PokerReplicaInference:
    """Explicit-K rollout and reference inference in natural lane shape."""

    def __init__(self, modules: Sequence[nn.Module] | ReplicaModuleOwner) -> None:
        self.modules = (
            modules
            if isinstance(modules, ReplicaModuleOwner)
            else ReplicaModuleOwner(modules)
        )
        self.operation = FunctionalReplicaInference(
            self.modules,
            encode_output=encode_model_evaluation,
        )

    def __call__(
        self,
        input_type: str,
        model_input: Tensor,
        private_state: Tensor,
        owner: Tensor,
        *,
        active_lanes: Tensor,
        natural_domain: bool = False,
        lane_ids: Sequence[int] | None = None,
    ) -> ModelEvaluation:
        if (
            model_input.ndim != 3
            or private_state.ndim < 2
            or owner.ndim != 2
            or model_input.shape[:2] != owner.shape
            or private_state.shape[:2] != owner.shape
            or model_input.shape[0] != active_lanes.numel()
        ):
            raise ValueError(
                "poker replica inference requires aligned [active,row,...] inputs"
            )
        encoded = self.operation(
            active_lanes,
            model_input,
            private_state,
            owner,
            static_arguments=(input_type,),
            static_keywords={"natural_domain": natural_domain},
            lane_ids=lane_ids,
        )
        return decode_model_evaluation(encoded)

__all__ = [
    "PokerReplicaInference",
    "decode_model_evaluation",
    "encode_model_evaluation",
]
