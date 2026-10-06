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


def adam_transition(
    masters: Tensor, gradients: Tensor, first: Tensor, second: Tensor,
    controls: Tensor, steps: Tensor, trackers: Tensor, enabled: Tensor,
    decoupled: Tensor, active: Tensor, fatal: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fixed-shape device transition shared by singleton and replica Adam.

    Controls are [lr,beta1,beta2,epsilon,decay,clip,scale,growth,backoff,interval].
    Gradients are dense, including zero rows for unused parameters. No Python
    scalar depends on device values; callers own publication and fatal checks.
    """

    lr, beta1, beta2, epsilon, decay, clip, scale, growth, backoff, interval = controls.unbind(1)
    before = scale.clone()
    attempt = active & ~fatal
    gradient = gradients / scale[:, None]
    norm = torch.linalg.vector_norm(gradient, dim=1)
    finite = torch.isfinite(gradient).all(1) & torch.isfinite(norm)
    apply = attempt & finite
    overflow = attempt & ~finite
    fatal.logical_or_(overflow & ~enabled)
    clipping = (clip / norm.clamp_min(1.0e-12)).clamp_max(1.0)
    clipping = torch.where(finite, clipping, torch.zeros_like(clipping))
    clipped = torch.where(apply[:, None], gradient * clipping[:, None], torch.zeros_like(gradient))
    coupled = torch.where(decoupled, torch.zeros_like(decay), decay)
    update_gradient = clipped + coupled[:, None] * masters
    next_first = beta1[:, None] * first + (1.0 - beta1[:, None]) * update_gradient
    next_second = beta2[:, None] * second + (1.0 - beta2[:, None]) * update_gradient.square()
    next_steps = steps + apply.to(steps.dtype)
    correction1 = 1.0 - beta1.pow(next_steps)
    correction2 = 1.0 - beta2.pow(next_steps)
    denominator = next_second.sqrt() / correction2.clamp_min(1.0e-30).sqrt()[:, None] + epsilon[:, None]
    direction = next_first / correction1.clamp_min(1.0e-30)[:, None] / denominator
    decayed = masters * (1.0 - lr * torch.where(decoupled, decay, torch.zeros_like(decay)))[:, None]
    candidate = decayed - lr[:, None] * direction
    state_finite = torch.isfinite(candidate).all(1) & torch.isfinite(next_first).all(1) & torch.isfinite(next_second).all(1)
    fatal.logical_or_(apply & ~state_finite)
    apply = apply & state_finite
    next_steps = steps + apply.to(steps.dtype)
    masters.copy_(torch.where(apply[:, None], candidate, masters))
    first.copy_(torch.where(apply[:, None], next_first, first))
    second.copy_(torch.where(apply[:, None], next_second, second))
    gradients.copy_(torch.where(attempt[:, None], torch.where(apply[:, None], update_gradient, torch.zeros_like(gradient)), gradients))
    steps.copy_(next_steps)
    next_tracker = trackers + (apply & enabled).to(trackers.dtype)
    grow = apply & enabled & next_tracker.ge(interval)
    next_tracker = torch.where((overflow & enabled) | grow, torch.zeros_like(trackers), next_tracker)
    next_scale = torch.where(overflow & enabled, scale * backoff, torch.where(grow, scale * growth, scale))
    controls[:, 6].copy_(next_scale)
    trackers.copy_(next_tracker)
    reported_norm = torch.where(finite, norm, torch.full_like(norm, float("inf")))
    return apply, finite, reported_norm, clipping, before, next_scale.clone(), next_steps.clone()


class DenseAdam:
    """One device-side Adam program over flat independent lane rows.

    Ordinary callers supply one module. Replica callers supply a sequence and
    the shared master's flat allocation. Lane handles below expose metadata and
    checkpoint views; they never implement another optimizer transition.
    """

    def __init__(
        self, module: nn.Module | tuple[nn.Module, ...],
        config: OptimizerConfig | tuple[OptimizerConfig, ...], *,
        scaler: LossScaler | None = None,
        scalers: tuple[LossScaler, ...] | None = None,
        parameter_flat: Tensor | None = None,
        gradient_flat: Tensor | None = None,
        exp_avg: Tensor | None = None, exp_avg_sq: Tensor | None = None,
    ) -> None:
        self.modules = (module,) if isinstance(module, nn.Module) else tuple(module)
        self.count = len(self.modules)
        if not self.count:
            raise ValueError("dense optimizer requires at least one lane")
        self.configs = ((config,) * self.count if isinstance(config, OptimizerConfig) else tuple(config))
        if len(self.configs) != self.count:
            raise ValueError("one optimizer configuration is required per lane")
        self._scalers = list(scalers if scalers is not None else ((scaler or LossScaler(),) if self.count == 1 else tuple(LossScaler() for _ in self.modules)))
        if len(self._scalers) != self.count:
            raise ValueError("one loss scaler is required per lane")
        parameters = tuple(tuple(p for p in m.parameters() if p.requires_grad) for m in self.modules)
        if not parameters[0] or any(tuple(p.shape for p in lane) != tuple(p.shape for p in parameters[0]) for lane in parameters):
            raise ValueError("optimizer lanes require identical nonempty parameter layouts")
        self.device = parameters[0][0].device
        if self.device.type != "cuda" or any(p.device != self.device or p.dtype != torch.float32 for lane in parameters for p in lane):
            raise ValueError("dense optimizer requires CUDA FP32 masters")
        width = sum(p.numel() for p in parameters[0])

        def storage(value: Tensor | None, *, zero: bool) -> Tensor:
            if value is None:
                return (torch.zeros if zero else torch.empty)((self.count, width), dtype=torch.float32, device=self.device)
            if value.numel() != self.count * width or value.dtype != torch.float32 or value.device != self.device or not value.is_contiguous():
                raise ValueError("external optimizer storage must be contiguous CUDA FP32 [K,N]")
            return value.view(self.count, width)

        self.masters = storage(parameter_flat, zero=False)
        if parameter_flat is None:
            # Bind once at construction. There is no per-step master repacking.
            with torch.no_grad():
                for lane, values in enumerate(parameters):
                    begin = 0
                    for parameter in values:
                        end = begin + parameter.numel()
                        view = self.masters[lane, begin:end].view_as(parameter)
                        view.copy_(parameter)
                        parameter.data = view
                        begin = end
        else:
            for lane, values in enumerate(parameters):
                begin = 0
                for parameter in values:
                    if parameter.data_ptr() != self.masters[lane, begin:].data_ptr():
                        raise ValueError("external master storage does not own the named parameters")
                    begin += parameter.numel()
        self.gradient_rows = storage(gradient_flat, zero=True)
        self.gradient_owners = tuple(DenseGradientOwner(m, flat=self.gradient_rows[lane]) for lane, m in enumerate(self.modules))
        self.first = storage(exp_avg, zero=True)
        self.second = storage(exp_avg_sq, zero=True)
        self.steps = torch.zeros(self.count, dtype=torch.int64, device=self.device)
        self.trackers = torch.tensor([s.growth_tracker for s in self._scalers], dtype=torch.int64, device=self.device)
        self.enabled = torch.tensor([s.enabled for s in self._scalers], dtype=torch.bool, device=self.device)
        self.decoupled = torch.tensor([c.name == "adamw" for c in self.configs], dtype=torch.bool, device=self.device)
        self.fatal = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self.controls = torch.tensor([
            [c.learning_rate, c.beta1, c.beta2, c.epsilon, c.weight_decay, c.grad_clip,
             s.scale, s.growth_factor, s.backoff_factor, s.growth_interval]
            for c, s in zip(self.configs, self._scalers, strict=True)
        ], dtype=torch.float32, device=self.device)
        self._mutable = (self.masters, self.first, self.second, self.steps, self.trackers, self.enabled, self.fatal, self.controls,
                         *(buffer for m in self.modules for buffer in m.buffers()))
        self._snapshots = tuple(torch.empty_like(value) for value in self._mutable)
        self._transaction_state: tuple[dict[str, object], ...] | None = None
        self.lanes = tuple(DenseAdamLane(self, lane) for lane in range(self.count))
        self._transition = adam_transition
        from pbg.core.execution_program import TensorProgramRunner
        self.program_runner = TensorProgramRunner()

    @property
    def config(self) -> OptimizerConfig:
        return self.configs[0]

    @property
    def gradients(self) -> DenseGradientOwner:
        return self.gradient_owners[0]

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        return self.gradients.parameters

    @property
    def exp_avg(self) -> Tensor:
        return self.first[0] if self.count == 1 else self.first

    @property
    def exp_avg_sq(self) -> Tensor:
        return self.second[0] if self.count == 1 else self.second

    @property
    def step_count(self) -> int:
        return int(self.steps[0].item())

    @step_count.setter
    def step_count(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("optimizer step is invalid")
        self.steps[0] = value

    @property
    def scaler(self) -> LossScaler:
        return self.lanes[0].scaler

    @scaler.setter
    def scaler(self, value: LossScaler) -> None:
        self.lanes[0].scaler = value

    @property
    def transaction_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self._snapshots)

    def configure_execution(self, *, mode: str, cache_entries: int, cache_bytes: int,
                            reserve_bytes: int = 0, peak_workspace_bytes: int = 0) -> None:
        from pbg.core.execution_program import TensorProgramRunner
        self.program_runner = TensorProgramRunner(mode=mode, max_entries=cache_entries,
                                                  byte_budget=cache_bytes, reserve_bytes=reserve_bytes,
                                                  peak_workspace_bytes=peak_workspace_bytes)
        self.configure_compilation(mode != "eager")

    def configure_compilation(self, enabled: bool) -> None:
        self._transition = (torch.compile(adam_transition, fullgraph=True, dynamic=False,
                                         mode="max-autotune-no-cudagraphs") if enabled else adam_transition)

    @torch.no_grad()
    def begin_transaction(self) -> None:
        if self._transaction_state is not None:
            raise OptimizationError("dense optimizer transaction is already active")
        for owner in self.gradient_owners:
            owner.assert_bound()
        for snapshot, value in zip(self._snapshots, self._mutable, strict=True):
            snapshot.copy_(value)
        self._transaction_state = tuple(s.state_dict() for s in self._scalers)

    @torch.no_grad()
    def rollback_transaction(self) -> None:
        if self._transaction_state is None:
            raise OptimizationError("dense optimizer transaction is not active")
        for value, snapshot in zip(self._mutable, self._snapshots, strict=True):
            value.copy_(snapshot)
        self._scalers = [LossScaler.from_state_dict(state) for state in self._transaction_state]
        self._transaction_state = None
        self.zero_grad()

    def commit_transaction(self) -> None:
        if self._transaction_state is None:
            raise OptimizationError("dense optimizer transaction is not active")
        if bool(self.fatal.any()):
            raise OptimizationError("fatal nonfinite loss/gradient prevents optimizer publication")
        self._transaction_state = None

    @torch.no_grad()
    def step_device(self, active: Tensor):
        if active.shape != (self.count,) or active.dtype != torch.bool or active.device != self.device:
            raise ValueError("optimizer active mask must be CUDA bool [K]")
        return self._transition(self.masters, self.gradient_rows, self.first, self.second,
                                self.controls, self.steps, self.trackers, self.enabled,
                                self.decoupled, active, self.fatal)

    @staticmethod
    def materialize_steps(result) -> tuple[OptimizerStep, ...]:
        applied, finite, norm, clip, before, after, steps = result
        # One reporting transfer after the tensor transition, never a decision
        # that changes the transition itself. Training batches these at publish.
        rows = torch.stack(tuple(value.double() for value in (applied, finite, norm, clip, before, after, steps)), dim=1).cpu().tolist()
        return tuple(OptimizerStep(bool(a), bool(f), n, c, b, z, int(s)) for a, f, n, c, b, z, s in rows)

    def step(self) -> OptimizerStep:
        if self.count != 1:
            raise OptimizationError("multi-lane optimizer requires step_device with an active mask")
        return self.materialize_steps(self.step_device(torch.ones(1, dtype=torch.bool, device=self.device)))[0]

    def zero_grad(self) -> None:
        self.gradient_rows.zero_()
        for owner in self.gradient_owners:
            owner.assert_bound()

    def state_dict(self) -> dict[str, object]:
        if self.count != 1:
            raise OptimizationError("serialize replica optimizer lane views explicitly")
        return self.lanes[0].state_dict()

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self.count != 1:
            raise OptimizationError("restore replica optimizer lane views explicitly")
        self.lanes[0].load_state_dict(state)


class DenseAdamLane:
    """A zero-copy checkpoint/control view, with no optimizer implementation."""

    def __init__(self, owner: DenseAdam, lane: int) -> None:
        self.owner, self.lane = owner, lane

    @property
    def config(self) -> OptimizerConfig:
        return self.owner.configs[self.lane]

    @property
    def scaler(self) -> LossScaler:
        result = self.owner._scalers[self.lane]
        result.scale = float(self.owner.controls[self.lane, 6].item())
        result.growth_tracker = int(self.owner.trackers[self.lane].item())
        return result

    @scaler.setter
    def scaler(self, value: LossScaler) -> None:
        self.owner._scalers[self.lane] = value
        self.owner.controls[self.lane, 6:] = torch.tensor(
            [value.scale, value.growth_factor, value.backoff_factor, value.growth_interval],
            dtype=torch.float32, device=self.owner.device)
        self.owner.trackers[self.lane] = value.growth_tracker
        self.owner.enabled[self.lane] = value.enabled

    @property
    def loss_scale(self) -> Tensor:
        return self.owner.controls[self.lane, 6]

    @property
    def gradients(self) -> DenseGradientOwner:
        return self.owner.gradient_owners[self.lane]

    @property
    def exp_avg(self) -> Tensor:
        return self.owner.first[self.lane]

    @property
    def exp_avg_sq(self) -> Tensor:
        return self.owner.second[self.lane]

    @property
    def step_count(self) -> int:
        return int(self.owner.steps[self.lane].item())

    def begin_transaction(self) -> None:
        self.owner.begin_transaction()

    def rollback_transaction(self) -> None:
        self.owner.rollback_transaction()

    def commit_transaction(self) -> None:
        self.owner.commit_transaction()

    def zero_grad(self) -> None:
        self.gradients.zero_()

    def step(self) -> OptimizerStep:
        active = torch.zeros(self.owner.count, dtype=torch.bool, device=self.owner.device)
        active[self.lane] = True
        return self.owner.materialize_steps(self.owner.step_device(active))[self.lane]

    def state_dict(self) -> dict[str, object]:
        if self.owner._transaction_state is not None:
            raise OptimizationError("cannot checkpoint an active optimizer transaction")
        return {
            "schema": "queued-dense-adam-v2",
            "config": {name: getattr(self.config, name) for name in self.config.__dataclass_fields__},
            "step": self.step_count,
            "exp_avg": self.exp_avg.detach().clone(),
            "exp_avg_sq": self.exp_avg_sq.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self.owner._transaction_state is not None:
            raise OptimizationError("cannot restore an active optimizer transaction")
        if set(state) != {"schema", "config", "step", "exp_avg", "exp_avg_sq"} or state["schema"] != "queued-dense-adam-v2":
            raise ValueError("queued optimizer state schema is invalid")
        if state["config"] != {name: getattr(self.config, name) for name in self.config.__dataclass_fields__}:
            raise ValueError("optimizer scientific configuration drifted")
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("optimizer step is invalid")
        for name, target in (("exp_avg", self.exp_avg), ("exp_avg_sq", self.exp_avg_sq)):
            source = state[name]
            if not isinstance(source, Tensor) or source.shape != target.shape or source.dtype != torch.float32:
                raise ValueError(f"optimizer {name} has the wrong contract")
            target.copy_(source.to(target.device))
        self.owner.steps[self.lane] = step
        self.owner.fatal[self.lane] = False


__all__ = [
    "DenseAdam",
    "DenseAdamLane",
    "adam_transition",
    "DenseGradientOwner",
    "LossScaler",
    "OptimizationError",
    "OptimizerConfig",
    "OptimizerStep",
]
