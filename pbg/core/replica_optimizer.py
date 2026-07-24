"""Lane-wise AdamW for physically stacked, statistically independent replicas.

Every tensor has a leading replica dimension ``K``.  Reductions in this module
stop at that boundary: gradient finiteness, global norm clipping, optimizer
state, and the AdamW step counter are all decided independently per lane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor
from torch import nn


@dataclass(frozen=True)
class ReplicaOptimizerStepResult:
    """Per-replica outcome of one optimizer boundary."""

    gradients_finite: Tensor
    optimizer_step_applied: Tensor
    grad_norm_pre_clip: Tensor
    grad_clip_scale: Tensor


class ReplicaAdamW:
    """AdamW over tensors laid out as ``[K, *parameter_shape]``.

    This is intentionally not a ``torch.optim.Optimizer`` subclass.  PyTorch
    optimizers interpret a stacked tensor as one parameter and therefore expose
    only one step counter and one finite/clipping decision.  Here those
    decisions are part of the replica dimension.
    """

    def __init__(
        self,
        parameters: Mapping[str, Tensor],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not parameters:
            raise ValueError("replica AdamW requires at least one parameter")
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("lr must be finite and nonnegative")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("AdamW betas must lie in [0, 1)")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")

        self.parameters = dict(parameters)
        first = next(iter(self.parameters.values()))
        if first.ndim < 1:
            raise ValueError("replica parameters require a leading K dimension")
        if first.device.type != "cuda":
            raise ValueError("replica AdamW is CUDA-only")
        self.replicas = int(first.shape[0])
        if self.replicas <= 0:
            raise ValueError("replica count K must be positive")

        for name, parameter in self.parameters.items():
            if parameter.ndim < 1 or parameter.shape[0] != self.replicas:
                raise ValueError(
                    f"parameter {name!r} does not share leading replica dimension "
                    f"K={self.replicas}"
                )
            if parameter.device != first.device:
                raise ValueError("all replica parameters must share one CUDA device")
            if parameter.dtype != torch.float32:
                raise TypeError("replica AdamW requires fp32 master parameters")

        self.lr = float(lr)
        self.betas = (float(beta1), float(beta2))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = torch.zeros(
            self.replicas, dtype=torch.float32, device=first.device
        )
        self.exp_avg = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.parameters.items()
        }
        self.exp_avg_sq = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.parameters.items()
        }

    @staticmethod
    def _lane_view(values: Tensor, tensor: Tensor) -> Tensor:
        return values.reshape((values.shape[0],) + (1,) * (tensor.ndim - 1))

    def _validate_gradients(self, gradients: Mapping[str, Tensor]) -> None:
        if gradients.keys() != self.parameters.keys():
            missing = sorted(self.parameters.keys() - gradients.keys())
            extra = sorted(gradients.keys() - self.parameters.keys())
            raise ValueError(
                f"gradient layout differs from parameters; missing={missing}, "
                f"extra={extra}"
            )
        for name, gradient in gradients.items():
            parameter = self.parameters[name]
            if gradient.shape != parameter.shape:
                raise ValueError(f"gradient {name!r} has the wrong shape")
            if gradient.device != parameter.device or gradient.dtype != torch.float32:
                raise TypeError(
                    f"gradient {name!r} must match its fp32 CUDA parameter"
                )

    def _gradient_statistics(
        self, gradients: Mapping[str, Tensor], grad_clip: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.step_count.device
        finite = torch.ones(self.replicas, dtype=torch.bool, device=device)
        squared_norm = torch.zeros(
            self.replicas, dtype=torch.float32, device=device
        )
        for gradient in gradients.values():
            flat = gradient.reshape(self.replicas, -1)
            finite = finite & torch.isfinite(flat).all(dim=1)
            squared_norm = squared_norm + flat.square().sum(dim=1)

        norm = squared_norm.sqrt()
        finite = finite & torch.isfinite(norm)
        reported_norm = torch.where(
            finite, norm, torch.full_like(norm, float("inf"))
        )
        if math.isinf(grad_clip):
            scale = torch.ones_like(norm)
        else:
            scale = (float(grad_clip) / norm.clamp_min(1.0e-12)).clamp(max=1.0)
        scale = torch.where(finite, scale, torch.zeros_like(scale))
        return finite, reported_norm, scale

    @torch.no_grad()
    def step(
        self,
        gradients: Mapping[str, Tensor],
        *,
        grad_clip: float = math.inf,
    ) -> ReplicaOptimizerStepResult:
        """Apply one independent finite-gated, globally clipped step per lane."""

        if math.isnan(grad_clip) or grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        self._validate_gradients(gradients)
        finite, grad_norm, clip_scale = self._gradient_statistics(
            gradients, grad_clip
        )
        applied = finite
        self.step_count.add_(applied.to(self.step_count.dtype))

        beta1, beta2 = self.betas
        # Clamp only protects the unselected zero-step lanes from invalid bias
        # corrections. Their parameters and moments are restored by ``where``.
        safe_step = self.step_count.clamp_min(1.0)
        bias_correction1 = 1.0 - torch.pow(
            torch.full_like(safe_step, beta1), safe_step
        )
        bias_correction2_sqrt = (
            1.0
            - torch.pow(torch.full_like(safe_step, beta2), safe_step)
        ).sqrt()
        step_size = self.lr / bias_correction1

        for name, parameter in self.parameters.items():
            mask = self._lane_view(applied, parameter)
            scale = self._lane_view(clip_scale, parameter)
            gradient = torch.where(mask, gradients[name] * scale, 0.0)

            old_avg = self.exp_avg[name]
            old_avg_sq = self.exp_avg_sq[name]
            candidate_avg = old_avg.lerp(gradient, 1.0 - beta1)
            candidate_avg_sq = old_avg_sq.mul(beta2).addcmul(
                gradient, gradient, value=1.0 - beta2
            )
            new_avg = torch.where(mask, candidate_avg, old_avg)
            new_avg_sq = torch.where(mask, candidate_avg_sq, old_avg_sq)
            old_avg.copy_(new_avg)
            old_avg_sq.copy_(new_avg_sq)

            lane_step_size = self._lane_view(step_size, parameter)
            lane_bias2 = self._lane_view(bias_correction2_sqrt, parameter)
            denominator = new_avg_sq.sqrt().div(lane_bias2).add(self.eps)
            # Decoupled decay is not multiplied by the bias-corrected step
            # size.
            candidate_parameter = parameter * (
                1.0 - self.lr * self.weight_decay
            ) - lane_step_size * new_avg / denominator
            parameter.copy_(torch.where(mask, candidate_parameter, parameter))

        return ReplicaOptimizerStepResult(
            gradients_finite=finite,
            optimizer_step_applied=applied,
            grad_norm_pre_clip=grad_norm,
            grad_clip_scale=clip_scale,
        )

    def state_dict(self) -> dict[str, object]:
        """Return an independently restorable stacked optimizer state."""

        return {
            "schema": "pbg-replica-adamw-v1",
            "replicas": self.replicas,
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step": self.step_count.detach().clone(),
            "exp_avg": {
                name: value.detach().clone() for name, value in self.exp_avg.items()
            },
            "exp_avg_sq": {
                name: value.detach().clone()
                for name, value in self.exp_avg_sq.items()
            },
        }

    def _ordinary_parameters(
        self, module: nn.Module, parameter_prefix: str
    ) -> dict[str, nn.Parameter]:
        ordinary = dict(module.named_parameters())
        expected = {
            name.removeprefix(parameter_prefix)
            if parameter_prefix and name.startswith(parameter_prefix)
            else name
            for name in self.parameters
        }
        if set(ordinary) != expected:
            raise ValueError(
                "ordinary module parameter layout differs from stacked replicas"
            )
        return ordinary

    @staticmethod
    def _validate_ordinary_group(
        optimizer: torch.optim.Optimizer,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        if len(optimizer.param_groups) != 1:
            raise ValueError("replica checkpoint export requires one parameter group")
        group = optimizer.param_groups[0]
        observed = (
            float(group["lr"]),
            tuple(float(value) for value in group["betas"]),
            float(group["eps"]),
            float(group["weight_decay"]),
        )
        expected = (lr, betas, eps, weight_decay)
        if observed != expected:
            raise ValueError(
                "ordinary optimizer hyperparameters differ from replica AdamW"
            )

    @torch.no_grad()
    def export_lane_to_optimizer(
        self,
        lane: int,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        parameter_prefix: str = "",
    ) -> None:
        """Populate one ordinary AdamW optimizer for a lane checkpoint."""

        if not 0 <= lane < self.replicas:
            raise IndexError("replica lane is outside [0,K)")
        self._validate_ordinary_group(
            optimizer,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        ordinary = self._ordinary_parameters(module, parameter_prefix)
        optimizer.state.clear()
        for stacked_name, average in self.exp_avg.items():
            name = (
                stacked_name.removeprefix(parameter_prefix)
                if parameter_prefix
                else stacked_name
            )
            parameter = ordinary[name]
            optimizer.state[parameter] = {
                "step": self.step_count[lane].detach().clone(),
                "exp_avg": average[lane].detach().clone(),
                "exp_avg_sq": self.exp_avg_sq[stacked_name][lane]
                .detach()
                .clone(),
            }

    @torch.no_grad()
    def import_lane_from_optimizer(
        self,
        lane: int,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        parameter_prefix: str = "",
    ) -> None:
        """Restore one lane's moments and step from an ordinary checkpoint."""

        if not 0 <= lane < self.replicas:
            raise IndexError("replica lane is outside [0,K)")
        self._validate_ordinary_group(
            optimizer,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        ordinary = self._ordinary_parameters(module, parameter_prefix)
        lane_step: Tensor | None = None
        for stacked_name in self.parameters:
            name = (
                stacked_name.removeprefix(parameter_prefix)
                if parameter_prefix
                else stacked_name
            )
            parameter = ordinary[name]
            state = optimizer.state.get(parameter)
            if not state:
                step = self.step_count.new_zeros(())
                average = torch.zeros_like(parameter)
                average_sq = torch.zeros_like(parameter)
            else:
                step = state.get("step")
                average = state.get("exp_avg")
                average_sq = state.get("exp_avg_sq")
                if (
                    not isinstance(step, Tensor)
                    or not isinstance(average, Tensor)
                    or not isinstance(average_sq, Tensor)
                ):
                    raise TypeError("ordinary AdamW state is incomplete")
            if lane_step is None:
                lane_step = step.to(self.step_count)
            elif not torch.equal(lane_step, step.to(lane_step)):
                raise ValueError("ordinary AdamW parameters have unequal step counts")
            self.exp_avg[stacked_name][lane].copy_(average)
            self.exp_avg_sq[stacked_name][lane].copy_(average_sq)
        assert lane_step is not None
        self.step_count[lane].copy_(lane_step)

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore state only when the complete lane/parameter layout matches."""

        if state.get("schema") != "pbg-replica-adamw-v1":
            raise ValueError("unsupported replica AdamW state schema")
        if int(state.get("replicas", -1)) != self.replicas:
            raise ValueError("replica AdamW state has the wrong K")
        step = state.get("step")
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(step, Tensor) or not isinstance(exp_avg, Mapping):
            raise TypeError("invalid replica AdamW state")
        if not isinstance(exp_avg_sq, Mapping):
            raise TypeError("invalid replica AdamW squared-moment state")
        if step.shape != self.step_count.shape:
            raise ValueError("replica AdamW step state has the wrong shape")
        self.step_count.copy_(step)
        for name, parameter in self.parameters.items():
            average = exp_avg.get(name)
            average_sq = exp_avg_sq.get(name)
            if not isinstance(average, Tensor) or average.shape != parameter.shape:
                raise ValueError(f"replica AdamW first moment {name!r} is invalid")
            if (
                not isinstance(average_sq, Tensor)
                or average_sq.shape != parameter.shape
            ):
                raise ValueError(f"replica AdamW second moment {name!r} is invalid")
            self.exp_avg[name].copy_(average)
            self.exp_avg_sq[name].copy_(average_sq)
