"""Poker binding for the generic queued replica inference owner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor, nn

from pbg.queued.contracts import ModeSchema, validate_replica_group
from pbg.queued.learner import ModelEvaluation
from pbg.queued.replica import (
    FunctionalReplicaInference,
    ReplicaModuleOwner,
    ReplicaOptimizerGroup,
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
    """Replica-vmapped rollout and reference inference in natural lane shape."""

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

    def lane_operation(
        self, lane: int, *, natural_domain: bool = False
    ) -> PokerReplicaLaneOperation:
        return PokerReplicaLaneOperation(
            self,
            lane,
            natural_domain=natural_domain,
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
        )
        return decode_model_evaluation(encoded)

    def rollout_mapping(
        self,
        input_type: str,
        model_input: Tensor,
        private_state: Tensor,
        owner: Tensor,
        *,
        active_lanes: Tensor,
    ) -> Mapping[str, Tensor]:
        evaluation = self(
            input_type,
            model_input,
            private_state,
            owner,
            active_lanes=active_lanes,
        )
        return MappingProxyType(
            {
                **{
                    name: value.float() for name, value in evaluation.policy_raw.items()
                },
                "value": evaluation.value.float(),
            }
        )


class PokerReplicaLaneOperation:
    """A production ``ModelOperation`` view for one replica-owned lane."""

    def __init__(
        self,
        inference: PokerReplicaInference,
        lane: int,
        *,
        natural_domain: bool,
    ) -> None:
        inference.modules._check_lane(lane)
        self.inference = inference
        self.lane = lane
        self.natural_domain = bool(natural_domain)
        self._active = torch.tensor(
            [lane], dtype=torch.int64, device=inference.modules.device
        )

    def __call__(
        self,
        input_type: str,
        model_input: Tensor,
        private_state: Tensor,
        owner: Tensor,
    ) -> ModelEvaluation:
        evaluation = self.inference(
            input_type,
            model_input.unsqueeze(0),
            private_state.unsqueeze(0),
            owner.unsqueeze(0),
            active_lanes=self._active,
            natural_domain=self.natural_domain,
        )
        return ModelEvaluation(
            MappingProxyType(
                {
                    name: value.squeeze(0)
                    for name, value in evaluation.policy_raw.items()
                }
            ),
            evaluation.value.squeeze(0),
            MappingProxyType(
                {
                    name: value.squeeze(0)
                    for name, value in evaluation.public_heads.items()
                }
            ),
        )


class _PokerIteration(Protocol):
    model: nn.Module
    optimizer: object
    mode: ModeSchema
    rollout_operation: object
    model_operation: object

    def run(self, *, iteration: int) -> object: ...


@dataclass(frozen=True)
class ReplicaPokerGroupResult:
    results: tuple[object | None, ...]
    failures: Mapping[int, str]
    versions: tuple[int, ...]


class ReplicaPokerIterationGroup:
    """Production owner/scheduler over existing poker iteration composition.

    The individual iteration remains the sole rollout, completion, target,
    learner, and optimizer path. This owner changes only the lane selected for
    its parameters, RNG/state, and no-grad functional inference.
    """

    def __init__(
        self,
        iterations: Sequence[_PokerIteration],
        *,
        inference: PokerReplicaInference,
        optimizers: ReplicaOptimizerGroup,
    ) -> None:
        self.iterations = tuple(iterations)
        self.inference = inference
        self.optimizers = optimizers
        count = inference.modules.n_replica
        if (
            not self.iterations
            or len(self.iterations) != count
            or optimizers.modules is not inference.modules
        ):
            raise ValueError("replica poker lane owners are not aligned")
        validate_replica_group(tuple(value.mode for value in self.iterations))
        for lane, value in enumerate(self.iterations):
            if (
                value.model is not inference.modules.lane_module(lane)
                or value.optimizer is not optimizers.optimizers[lane]
                or not isinstance(value.rollout_operation, PokerReplicaLaneOperation)
                or value.rollout_operation.inference is not inference
                or value.rollout_operation.lane != lane
                or getattr(value.model_operation, "__self__", None)
                is not inference.modules.lane_module(lane)
            ):
                raise ValueError(
                    "poker iteration is not bound to its replica-owned lane"
                )

    def run(
        self,
        *,
        iteration: int,
        active_lanes: Sequence[int] | None = None,
    ) -> ReplicaPokerGroupResult:
        lanes = (
            tuple(range(len(self.iterations)))
            if active_lanes is None
            else tuple(sorted(int(value) for value in active_lanes))
        )
        self.optimizers._mask(lanes)
        results: list[object | None] = [None] * len(self.iterations)
        failures: dict[int, str] = {}
        prior_steps = tuple(
            optimizer.step_count for optimizer in self.optimizers.optimizers
        )
        for lane in lanes:
            try:
                results[lane] = self.iterations[lane].run(iteration=iteration)
            except Exception as exc:  # noqa: BLE001 - lane-local disposition
                failures[lane] = f"{type(exc).__name__}: {exc}"
        for lane in lanes:
            delta = self.optimizers.optimizers[lane].step_count - prior_steps[lane]
            if lane in failures:
                if delta:
                    raise RuntimeError(
                        "failed replica lane crossed an optimizer boundary"
                    )
            elif delta < 0:
                raise RuntimeError("replica optimizer counter moved backward")
            else:
                self.optimizers.versions[lane] += delta
        return ReplicaPokerGroupResult(
            tuple(results),
            MappingProxyType(failures),
            tuple(int(value) for value in self.optimizers.versions.tolist()),
        )


__all__ = [
    "PokerReplicaInference",
    "PokerReplicaLaneOperation",
    "ReplicaPokerGroupResult",
    "ReplicaPokerIterationGroup",
    "decode_model_evaluation",
    "encode_model_evaluation",
]
