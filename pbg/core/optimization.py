"""Dense FP32 gradient ownership and Adam/AdamW transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class OptimizationError(RuntimeError):
    """The dense optimizer contract was violated."""


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in ("adam", "adamw"):
            raise ValueError("name must be 'adam' or 'adamw'")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if not math.isfinite(self.beta1) or not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be finite in [0,1)")
        if not math.isfinite(self.beta2) or not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be finite in [0,1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass
class LossScaler:
    """One explicit scaler state for FP16; FP32/BF16 use unit scale."""

    enabled: bool = False
    scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    growth_tracker: int = 0

    def __post_init__(self) -> None:
        if not self.enabled:
            self.scale = 1.0
        if (
            self.scale <= 0
            or self.growth_factor <= 1
            or not 0 < self.backoff_factor < 1
            or self.growth_interval <= 0
        ):
            raise ValueError("loss scaler configuration is invalid")

    def update(self, finite: bool) -> None:
        if not self.enabled:
            self.scale = 1.0
            self.growth_tracker = 0
            return
        if not finite:
            self.scale *= self.backoff_factor
            self.growth_tracker = 0
            return
        self.growth_tracker += 1
        if self.growth_tracker == self.growth_interval:
            self.scale *= self.growth_factor
            self.growth_tracker = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "scale": self.scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "growth_tracker": self.growth_tracker,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> LossScaler:
        expected = {
            "enabled",
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "growth_tracker",
        }
        if set(state) != expected:
            raise ValueError("loss scaler state has unknown or missing fields")
        return cls(**dict(state))


class DenseGradientOwner:
    """One FP32 allocation to which every trainable ``.grad`` is bound."""

    def __init__(self, module: nn.Module, *, flat: Tensor | None = None) -> None:
        self.parameters = tuple(
            parameter for parameter in module.parameters() if parameter.requires_grad
        )
        if not self.parameters:
            raise ValueError("dense gradients require trainable parameters")
        devices = {parameter.device for parameter in self.parameters}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("queued dense gradients require one CUDA device")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters):
            raise TypeError("queued master parameters must be FP32")
        self.device = next(iter(devices))
        size = sum(parameter.numel() for parameter in self.parameters)
        if flat is None:
            flat = torch.zeros(
                size,
                dtype=torch.float32,
                device=self.device,
            )
        elif (
            flat.shape != (size,)
            or flat.dtype != torch.float32
            or flat.device != self.device
            or not flat.is_contiguous()
        ):
            raise ValueError(
                "external dense gradient storage must be contiguous FP32 [N] "
                "on the module device"
            )
        self.flat = flat
        self.views: tuple[Tensor, ...] = self._views()
        self.bind()

    def _views(self) -> tuple[Tensor, ...]:
        offset = 0
        result = []
        for parameter in self.parameters:
            end = offset + parameter.numel()
            result.append(self.flat[offset:end].view_as(parameter))
            offset = end
        return tuple(result)

    def bind(self) -> None:
        for parameter, view in zip(self.parameters, self.views, strict=True):
            parameter.grad = view
        self.assert_bound()

    def assert_bound(self) -> None:
        base = self.flat.untyped_storage().data_ptr()
        for parameter, view in zip(self.parameters, self.views, strict=True):
            if (
                parameter.grad is None
                or parameter.grad.untyped_storage().data_ptr() != base
                or parameter.grad.data_ptr() != view.data_ptr()
            ):
                raise OptimizationError("a parameter lost its dense gradient view")

    def zero_(self) -> None:
        self.flat.zero_()
        self.bind()


@dataclass(frozen=True)
class OptimizerStep:
    applied: bool
    gradients_finite: bool
    grad_norm: float
    clip_scale: float
    loss_scale_before: float
    loss_scale_after: float
    step: int


class DenseAdam:
    """Selected Adam/AdamW over the owning FP32 parameter/gradient vectors."""

    def __init__(
        self,
        module: nn.Module,
        config: OptimizerConfig,
        *,
        scaler: LossScaler | None = None,
        gradient_flat: Tensor | None = None,
        exp_avg: Tensor | None = None,
        exp_avg_sq: Tensor | None = None,
    ) -> None:
        self.config = config
        self.gradients = DenseGradientOwner(module, flat=gradient_flat)
        self.parameters = self.gradients.parameters
        self.buffers = tuple(module.buffers())
        self.exp_avg = self._state_vector(exp_avg, "exp_avg")
        self.exp_avg_sq = self._state_vector(exp_avg_sq, "exp_avg_sq")
        self.step_count = 0
        self.scaler = LossScaler() if scaler is None else scaler
        self._parameter_snapshot = torch.empty_like(self.gradients.flat)
        self._parameter_snapshot_views = self._flat_views(
            self._parameter_snapshot
        )
        self._buffer_snapshots = tuple(
            torch.empty_like(buffer) for buffer in self.buffers
        )
        self._exp_avg_snapshot = torch.empty_like(self.exp_avg)
        self._exp_avg_sq_snapshot = torch.empty_like(self.exp_avg_sq)
        self._transaction_state: tuple[int, dict[str, object]] | None = None

    def _state_vector(self, value: Tensor | None, name: str) -> Tensor:
        if value is None:
            return torch.zeros_like(self.gradients.flat)
        if (
            value.shape != self.gradients.flat.shape
            or value.dtype != torch.float32
            or value.device != self.gradients.device
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"external optimizer {name} must match the dense FP32 gradient row"
            )
        return value

    def _flat_views(self, value: Tensor) -> tuple[Tensor, ...]:
        if value.ndim != 1 or value.numel() != self.gradients.flat.numel():
            raise OptimizationError("optimizer vector has the wrong shape")
        offset = 0
        result = []
        for parameter in self.parameters:
            end = offset + parameter.numel()
            result.append(value[offset:end].view_as(parameter))
            offset = end
        return tuple(result)

    @property
    def transaction_bytes(self) -> int:
        """Resident accelerator bytes reserved for exact update rollback."""

        tensors = (
            self._parameter_snapshot,
            self._exp_avg_snapshot,
            self._exp_avg_sq_snapshot,
            *self._buffer_snapshots,
        )
        return sum(value.numel() * value.element_size() for value in tensors)

    @torch.no_grad()
    def begin_transaction(self) -> None:
        """Snapshot every mutable learner tensor before the first transition."""

        if self._transaction_state is not None:
            raise OptimizationError("dense optimizer transaction is already active")
        self.gradients.assert_bound()
        torch._foreach_copy_(self._parameter_snapshot_views, self.parameters)
        for destination, source in zip(
            self._buffer_snapshots, self.buffers, strict=True
        ):
            destination.copy_(source)
        self._exp_avg_snapshot.copy_(self.exp_avg)
        self._exp_avg_sq_snapshot.copy_(self.exp_avg_sq)
        self._transaction_state = (
            self.step_count,
            self.scaler.state_dict(),
        )

    @torch.no_grad()
    def rollback_transaction(self) -> None:
        """Restore the exact pre-transaction model, optimizer, and scaler."""

        if self._transaction_state is None:
            raise OptimizationError("dense optimizer transaction is not active")
        torch._foreach_copy_(self.parameters, self._parameter_snapshot_views)
        for destination, source in zip(
            self.buffers, self._buffer_snapshots, strict=True
        ):
            destination.copy_(source)
        self.exp_avg.copy_(self._exp_avg_snapshot)
        self.exp_avg_sq.copy_(self._exp_avg_sq_snapshot)
        step_count, scaler_state = self._transaction_state
        self.step_count = step_count
        self.scaler = LossScaler.from_state_dict(scaler_state)
        self._transaction_state = None
        self.zero_grad()

    def commit_transaction(self) -> None:
        """Publish all transitions by discarding the rollback point."""

        if self._transaction_state is None:
            raise OptimizationError("dense optimizer transaction is not active")
        self._transaction_state = None

    @torch.no_grad()
    def step(self) -> OptimizerStep:
        self.gradients.assert_bound()
        before = float(self.scaler.scale)
        if self.scaler.enabled:
            self.gradients.flat.div_(before)
        finite = bool(torch.isfinite(self.gradients.flat).all())
        if not finite:
            self.gradients.flat.zero_()
            self.scaler.update(False)
            return OptimizerStep(
                False,
                False,
                float("inf"),
                0.0,
                before,
                float(self.scaler.scale),
                self.step_count,
            )
        norm_tensor = torch.linalg.vector_norm(self.gradients.flat)
        norm = float(norm_tensor)
        clip_scale = min(1.0, self.config.grad_clip / max(norm, 1.0e-12))
        self.gradients.flat.mul_(clip_scale)
        self.step_count += 1
        beta1, beta2 = self.config.beta1, self.config.beta2
        gradient = self.gradients.flat
        if self.config.name == "adam" and self.config.weight_decay:
            torch._foreach_add_(
                self.gradients.views,
                self.parameters,
                alpha=self.config.weight_decay,
            )
        self.exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        correction1 = 1.0 - beta1**self.step_count
        correction2 = 1.0 - beta2**self.step_count
        denominator = self.exp_avg_sq.sqrt().div_(math.sqrt(correction2))
        denominator.add_(self.config.epsilon)
        direction = self.exp_avg / correction1 / denominator
        if self.config.name == "adamw" and self.config.weight_decay:
            torch._foreach_mul_(
                self.parameters,
                1.0 - self.config.learning_rate * self.config.weight_decay,
            )
        torch._foreach_add_(
            self.parameters,
            self._flat_views(direction),
            alpha=-self.config.learning_rate,
        )
        self.scaler.update(True)
        return OptimizerStep(
            True,
            True,
            norm,
            clip_scale,
            before,
            float(self.scaler.scale),
            self.step_count,
        )

    def zero_grad(self) -> None:
        self.gradients.zero_()

    def state_dict(self) -> dict[str, object]:
        if self._transaction_state is not None:
            raise OptimizationError(
                "cannot checkpoint an active optimizer transaction"
            )
        return {
            "schema": "queued-dense-adam-v1",
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
            },
            "step": self.step_count,
            "exp_avg": self.exp_avg.detach().clone(),
            "exp_avg_sq": self.exp_avg_sq.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._transaction_state is not None:
            raise OptimizationError(
                "cannot restore an active optimizer transaction"
            )
        expected = {"schema", "config", "step", "exp_avg", "exp_avg_sq"}
        if set(state) != expected or state["schema"] != "queued-dense-adam-v1":
            raise ValueError("queued optimizer state schema is invalid")
        if state["config"] != {
            name: getattr(self.config, name)
            for name in self.config.__dataclass_fields__
        }:
            raise ValueError("optimizer scientific configuration drifted")
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("optimizer step is invalid")
        for name, destination in (
            ("exp_avg", self.exp_avg),
            ("exp_avg_sq", self.exp_avg_sq),
        ):
            source = state[name]
            if (
                not isinstance(source, Tensor)
                or source.shape != destination.shape
                or source.dtype != torch.float32
            ):
                raise ValueError(f"optimizer {name} has the wrong contract")
            destination.copy_(source.to(destination.device))
        self.step_count = step


__all__ = [
    "DenseAdam",
    "DenseGradientOwner",
    "LossScaler",
    "OptimizationError",
    "OptimizerConfig",
    "OptimizerStep",
]
