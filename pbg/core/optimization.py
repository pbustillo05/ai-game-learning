"""Distributed gradient reduction and finite-gated optimizer updates."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class OptimizerStepResult:
    """Observable result of one common finite-gradient decision."""

    gradients_finite: bool
    optimizer_step_applied: bool
    grad_norm_pre_clip: float
    grad_clip_scale: float
    loss_scale_before: float
    loss_scale_after: float

    def metrics(self) -> dict[str, float]:
        return {
            "reduced_gradients_finite": float(self.gradients_finite),
            "optimizer_step_applied": float(self.optimizer_step_applied),
            "optimizer_step_skipped_nonfinite": float(not self.gradients_finite),
            "grad_norm_pre_clip": self.grad_norm_pre_clip,
            "grad_clip_scale": self.grad_clip_scale,
            "loss_scale_before": self.loss_scale_before,
            "loss_scale_after": self.loss_scale_after,
        }


def sync_scaled_gradients(
    model: nn.Module, world: int, *, metrics: Any | None = None
) -> bool:
    """SUM one reusable flat fp32 buffer and return the common finite decision."""

    source = model.module if hasattr(model, "module") else model
    parameters = [parameter for parameter in source.parameters() if parameter.requires_grad]
    total = sum(parameter.numel() for parameter in parameters)
    if total == 0:
        return True
    cache = getattr(source, "_pbg_flat_gradient_buffers", None)
    if cache is None:
        cache = {}
        source._pbg_flat_gradient_buffers = cache
    key = (str(parameters[0].device), total)
    flat = cache.get(key)
    if flat is None or flat.device != parameters[0].device:
        flat = torch.empty(total, dtype=torch.float32, device=parameters[0].device)
        cache[key] = flat
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        if parameter.grad is None:
            flat[offset : offset + count].zero_()
        else:
            flat[offset : offset + count].copy_(
                parameter.grad.detach().reshape(-1).to(torch.float32)
            )
        offset += count
    if world > 1:
        started = time.perf_counter()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if metrics is not None:
            metrics.record_collective(
                nbytes=flat.numel() * flat.element_size(),
                seconds=time.perf_counter() - started,
            )
    finite = bool(torch.isfinite(flat).all())
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.copy_(
            flat[offset : offset + count].view_as(parameter).to(parameter.grad.dtype)
        )
        offset += count
    return finite


def _loss_scale(scaler: Any) -> float:
    return float(scaler.get_scale()) if scaler.is_enabled() else 1.0


def _backoff_loss_scale(scaler: Any, current: float) -> float:
    """Apply the explicit common nonfinite transition to an enabled scaler."""

    if not scaler.is_enabled():
        return current
    get_backoff = getattr(scaler, "get_backoff_factor", None)
    backoff = float(get_backoff()) if callable(get_backoff) else 0.5
    updated = current * backoff
    scaler.update(new_scale=updated)
    return _loss_scale(scaler)


def finite_gated_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    gradients_finite: bool,
    grad_clip: float,
) -> OptimizerStepResult:
    """Unscale, clip, and step only when the reduced gradients are finite.

    A false common decision clears the unusable gradients and, for FP16, backs
    the shared loss scale off once without invoking local scaler inf checks.
    Disabled scalers (FP32/BF16/no-AMP) retain their unit scale.
    """

    scale_before = _loss_scale(scaler)
    if not gradients_finite:
        scale_after = _backoff_loss_scale(scaler, scale_before)
        optimizer.zero_grad(set_to_none=True)
        return OptimizerStepResult(
            gradients_finite=False,
            optimizer_step_applied=False,
            grad_norm_pre_clip=float("inf"),
            grad_clip_scale=0.0,
            loss_scale_before=scale_before,
            loss_scale_after=scale_after,
        )

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip, error_if_nonfinite=True
    )
    grad_norm_value = float(grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return OptimizerStepResult(
        gradients_finite=True,
        optimizer_step_applied=True,
        grad_norm_pre_clip=grad_norm_value,
        grad_clip_scale=min(1.0, float(grad_clip) / max(grad_norm_value, 1.0e-12)),
        loss_scale_before=scale_before,
        loss_scale_after=_loss_scale(scaler),
    )
