"""Direct flat-gradient storage and finite-gated distributed updates.

The flat buffer is the owning storage for every trainable ``.grad``.  Backward
therefore accumulates directly into the buffer that is SUM-reduced; there is no
post-backward flatten/copy pass and no DDP bucket averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class FlatGradientBuffer:
    """One fp32 SUM buffer with parameter gradients bound as stable views."""

    def __init__(self, model: nn.Module) -> None:
        source = model.module if hasattr(model, "module") else model
        self.parameters = tuple(
            parameter for parameter in source.parameters() if parameter.requires_grad
        )
        if any(parameter.dtype != torch.float32 for parameter in self.parameters):
            raise TypeError(
                "direct fp32 gradient views require fp32 master parameters; use "
                "autocast for reduced-precision forwards"
            )
        devices = {parameter.device for parameter in self.parameters}
        if len(devices) > 1:
            raise ValueError("one flat gradient buffer cannot span devices")
        device = next(iter(devices), torch.device("cpu"))
        self.flat = torch.zeros(
            sum(parameter.numel() for parameter in self.parameters),
            dtype=torch.float32,
            device=device,
        )
        self._views: tuple[Tensor, ...] = self._make_views()
        self.bind()

    def _make_views(self) -> tuple[Tensor, ...]:
        views: list[Tensor] = []
        offset = 0
        for parameter in self.parameters:
            stop = offset + parameter.numel()
            views.append(self.flat[offset:stop].view_as(parameter))
            offset = stop
        if offset != self.flat.numel():
            raise RuntimeError("flat gradient layout does not cover its buffer")
        return tuple(views)

    @property
    def nbytes(self) -> int:
        return self.flat.numel() * self.flat.element_size()

    def bind(self) -> None:
        """Restore every ``.grad`` alias (optimizers may have cleared it)."""

        for parameter, view in zip(self.parameters, self._views, strict=True):
            parameter.grad = view
        self.assert_aliasing()

    def zero_(self) -> None:
        self.flat.zero_()
        self.bind()

    def assert_aliasing(self) -> None:
        base = self.flat.untyped_storage().data_ptr()
        for parameter, view in zip(self.parameters, self._views, strict=True):
            if parameter.grad is None:
                raise RuntimeError("a trainable parameter lost its flat gradient view")
            if parameter.grad.untyped_storage().data_ptr() != base:
                raise RuntimeError("parameter gradient does not alias the flat buffer")
            if parameter.grad.data_ptr() != view.data_ptr():
                raise RuntimeError("parameter gradient has the wrong flat-buffer offset")

    def state_dict(self) -> dict[str, int]:
        """Layout metadata only; gradients are step-transient, not checkpointed."""

        return {
            "num_parameters": len(self.parameters),
            "numel": self.flat.numel(),
            "nbytes": self.nbytes,
        }


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
    model: nn.Module,
    world: int,
    *,
    metrics: Any | None = None,
    flat_buffer: FlatGradientBuffer | None = None,
) -> bool:
    """SUM the owning fp32 gradient buffer and return the common finite decision.

    New callers must bind ``flat_buffer`` before backward.  The compatibility
    fallback is intentionally strict: it only accepts gradients that already
    alias a cached direct buffer and never copies gradients into one.
    """

    source = model.module if hasattr(model, "module") else model
    if flat_buffer is None:
        flat_buffer = getattr(source, "_pbg_direct_flat_gradient_buffer", None)
        if flat_buffer is None:
            raise RuntimeError(
                "a FlatGradientBuffer must be allocated and bound before backward"
            )
    if not isinstance(flat_buffer, FlatGradientBuffer):
        raise TypeError("flat_buffer must be a FlatGradientBuffer")
    flat_buffer.assert_aliasing()
    flat = flat_buffer.flat
    if flat.numel() == 0:
        return True
    if world > 1:
        started = time.perf_counter()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if metrics is not None:
            metrics.record_collective(
                nbytes=flat.numel() * flat.element_size(),
                seconds=time.perf_counter() - started,
            )
    return bool(torch.isfinite(flat).all())


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
        # Keep direct gradient views alive.  The owner is zeroed once at the
        # next optimizer step; clearing to None would destroy the layout.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.zero_()
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
