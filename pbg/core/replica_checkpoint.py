"""Independent ordinary-checkpoint interop for physically stacked replicas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pbg.core.checkpointing import load_checkpoint_payload, save_checkpoint
from pbg.core.config import RootConfig
from pbg.core.replica_learner import FunctionalReplicaLearner, StackedModuleReplicas


@dataclass(frozen=True)
class OrdinaryReplicaLane:
    """Ephemeral ordinary objects backed by one stacked lane."""

    model: nn.Module
    optimizer: torch.optim.AdamW


class ReplicaCheckpointManager:
    """Export and restore each lane using the repository checkpoint schema."""

    def __init__(
        self,
        learner: FunctionalReplicaLearner,
        *,
        model_prefix: str = "bundle.",
        reference: StackedModuleReplicas | None = None,
    ) -> None:
        self.learner = learner
        self.model_prefix = model_prefix
        self.reference = reference

    @staticmethod
    def _unwrap(wrapper: nn.Module, prefix: str) -> nn.Module:
        if not prefix:
            return wrapper
        name = prefix.removesuffix(".")
        result = wrapper.get_submodule(name)
        if not isinstance(result, nn.Module):
            raise TypeError("replica checkpoint prefix does not name a module")
        return result

    def ordinary_lane(self, lane: int) -> OrdinaryReplicaLane:
        wrapper = self.learner.replicas.lane_module(lane)
        model = self._unwrap(wrapper, self.model_prefix)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learner.optimizer.lr,
            betas=self.learner.optimizer.betas,
            eps=self.learner.optimizer.eps,
            weight_decay=self.learner.optimizer.weight_decay,
        )
        self.learner.optimizer.export_lane_to_optimizer(
            lane,
            model,
            optimizer,
            parameter_prefix=self.model_prefix,
        )
        return OrdinaryReplicaLane(model, optimizer)

    def save_lane(
        self,
        path: str | Path,
        lane: int,
        *,
        scaler: Any,
        iteration: int,
        cfg: RootConfig,
        torch_generator: torch.Generator,
        numpy_generator: np.random.Generator,
        **metadata: Any,
    ) -> None:
        ordinary = self.ordinary_lane(lane)
        reference_model = None
        if self.reference is not None:
            reference_wrapper = self.reference.lane_module(lane)
            reference_model = self._unwrap(reference_wrapper, self.model_prefix)
        save_checkpoint(
            path,
            ordinary.model,
            ordinary.optimizer,
            scaler,
            iteration,
            cfg,
            torch_generator,
            numpy_generator,
            reference_model=reference_model,
            **metadata,
        )

    @torch.no_grad()
    def load_lane(
        self,
        path: str | Path,
        lane: int,
        *,
        map_location: torch.device | str,
    ) -> dict[str, Any]:
        payload = load_checkpoint_payload(path, map_location=map_location)
        ordinary = self.ordinary_lane(lane)
        ordinary.model.load_state_dict(payload["model"])
        ordinary.optimizer.load_state_dict(payload["optimizer"])
        self.learner.replicas.load_lane_module_state(
            lane,
            ordinary.model.state_dict(),
            add_prefix=self.model_prefix,
        )
        self.learner.optimizer.import_lane_from_optimizer(
            lane,
            ordinary.model,
            ordinary.optimizer,
            parameter_prefix=self.model_prefix,
        )
        if self.reference is not None:
            reference_state = payload["reference_model"]
            if reference_state is None:
                raise RuntimeError("replica checkpoint omitted its reference model")
            self.reference.load_lane_module_state(
                lane,
                reference_state,
                add_prefix=self.model_prefix,
            )
        return payload


__all__ = ["OrdinaryReplicaLane", "ReplicaCheckpointManager"]
