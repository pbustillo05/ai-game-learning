"""Replica ownership, no-grad inference, and distributed learner scheduling.

Replicas do not define another learner or optimizer.  This module only owns
the leading lane axis, binds zero-copy lane modules to it, schedules the
existing per-lane learner operation, and synchronizes the resulting ordered
dense FP32 gradient rows when distributed execution requires it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.func import functional_call, stack_module_state, vmap

from pbg.queued.optimization import (
    DenseAdam,
    LossScaler,
    OptimizerConfig,
    OptimizerStep,
)


class ReplicaError(RuntimeError):
    """A replica ownership, inference, or scheduling contract was violated."""


def _owner(module: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    path, _, leaf = qualified_name.rpartition(".")
    return (module.get_submodule(path) if path else module), leaf


class ReplicaModuleOwner:
    """One stacked parameter/buffer owner with zero-copy trainable lane views."""

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        modules = tuple(modules)
        if not modules:
            raise ValueError("replica ownership requires at least one module")
        devices = {
            value.device
            for module in modules
            for value in (*module.parameters(), *module.buffers())
        }
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("replica modules must occupy one CUDA device")
        if len({module.training for module in modules}) != 1:
            raise ValueError("replica modules must share train/eval state")
        parameter_names = tuple(dict(modules[0].named_parameters()))
        buffer_names = tuple(dict(modules[0].named_buffers()))
        state_names = tuple(modules[0].state_dict())
        for module in modules[1:]:
            if (
                tuple(dict(module.named_parameters())) != parameter_names
                or tuple(dict(module.named_buffers())) != buffer_names
                or tuple(module.state_dict()) != state_names
            ):
                raise ValueError("replica modules have different state layouts")
        parameters, buffers = stack_module_state(list(modules))
        if any(value.dtype != torch.float32 for value in parameters.values()):
            raise TypeError("replica master parameters must be FP32")
        self.parameters = dict(parameters)
        self.buffers = dict(buffers)
        self.n_replica = len(modules)
        self.device = next(iter(devices))
        self.training = modules[0].training
        self._parameter_names = parameter_names
        self._buffer_names = buffer_names
        self._persistent_names = state_names
        self._code = copy.deepcopy(modules[0]).to("meta")
        self._lanes: dict[int, nn.Module] = {}

    @property
    def parameter_count(self) -> int:
        return sum(value[0].numel() for value in self.parameters.values())

    def _check_lane(self, lane: int) -> None:
        if isinstance(lane, bool) or not isinstance(lane, int):
            raise TypeError("replica lane must be an integer")
        if not 0 <= lane < self.n_replica:
            raise IndexError("replica lane is outside [0,K)")

    def lane_module(self, lane: int) -> nn.Module:
        """Return the sole trainable module whose tensors alias one lane."""

        self._check_lane(lane)
        if lane in self._lanes:
            return self._lanes[lane]
        module = copy.deepcopy(self._code)
        for name in self._parameter_names:
            target, leaf = _owner(module, name)
            target._parameters[leaf] = nn.Parameter(
                self.parameters[name][lane],
                requires_grad=True,
            )
        for name in self._buffer_names:
            target, leaf = _owner(module, name)
            target._buffers[leaf] = self.buffers[name][lane]
        module.train(self.training)
        if any(value.device.type == "meta" for value in module.state_dict().values()):
            raise ReplicaError("zero-copy lane binding retained meta storage")
        self._lanes[lane] = module
        return module

    def assert_zero_copy(self) -> None:
        """Prove every materialized lane still aliases its stacked owner."""

        for lane, module in self._lanes.items():
            for name, parameter in module.named_parameters():
                if parameter.data_ptr() != self.parameters[name][lane].data_ptr():
                    raise ReplicaError("a lane parameter lost its zero-copy binding")
            for name, buffer in module.named_buffers():
                if buffer.data_ptr() != self.buffers[name][lane].data_ptr():
                    raise ReplicaError("a lane buffer lost its zero-copy binding")

    def lane_state_dict(self, lane: int) -> dict[str, Tensor]:
        self._check_lane(lane)
        combined = {**self.parameters, **self.buffers}
        return {
            name: combined[name][lane].detach().clone()
            for name in self._persistent_names
        }

    @torch.no_grad()
    def load_lane_state_dict(self, lane: int, state: Mapping[str, Tensor]) -> None:
        self._check_lane(lane)
        if tuple(state) != self._persistent_names:
            raise ValueError("replica lane state layout differs")
        combined = {**self.parameters, **self.buffers}
        for name, source in state.items():
            destination = combined[name][lane]
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError(f"replica lane tensor {name!r} differs")
            destination.copy_(source.to(destination.device))

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "pbg-queued-replica-module-v1",
            "n_replica": self.n_replica,
            "lanes": tuple(
                self.lane_state_dict(lane) for lane in range(self.n_replica)
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            set(state) != {"schema", "n_replica", "lanes"}
            or state["schema"] != "pbg-queued-replica-module-v1"
        ):
            raise ValueError("replica module state schema is invalid")
        if state["n_replica"] != self.n_replica:
            raise ValueError("replica module lane count differs")
        lanes = state["lanes"]
        if not isinstance(lanes, (tuple, list)) or len(lanes) != self.n_replica:
            raise ValueError("replica module lanes are invalid")
        for lane, lane_state in enumerate(lanes):
            if not isinstance(lane_state, Mapping):
                raise TypeError("replica module lane state is invalid")
            self.load_lane_state_dict(lane, lane_state)


OutputEncoder = Callable[[Any], Mapping[str, Tensor]]


class FunctionalReplicaInference:
    """One no-grad ``vmap(functional_call)`` over only the active lane subset."""

    def __init__(
        self,
        owner: ReplicaModuleOwner,
        *,
        encode_output: OutputEncoder,
    ) -> None:
        self.owner = owner
        self.encode_output = encode_output

    def __call__(
        self,
        active_lanes: Tensor,
        *lane_arguments: Tensor,
        static_arguments: tuple[object, ...] = (),
        static_keywords: Mapping[str, object] | None = None,
    ) -> Mapping[str, Tensor]:
        if (
            active_lanes.device != self.owner.device
            or active_lanes.dtype != torch.int64
            or active_lanes.ndim != 1
            or not active_lanes.is_contiguous()
            or active_lanes.numel() == 0
        ):
            raise ValueError(
                "active lanes must be a nonempty contiguous CUDA int64 row"
            )
        ordered = active_lanes.sort().values
        torch._assert_async(
            (
                ordered.ge(0)
                & ordered.lt(self.owner.n_replica)
                & torch.cat(
                    (
                        torch.ones(1, dtype=torch.bool, device=ordered.device),
                        ordered[1:].ne(ordered[:-1]),
                    )
                )
            ).all(),
            "active replica lanes must be unique addresses in [0,K)",
        )
        active = int(active_lanes.numel())
        for argument in lane_arguments:
            if (
                not isinstance(argument, Tensor)
                or argument.device != self.owner.device
                or argument.ndim == 0
                or argument.shape[0] != active
            ):
                raise ValueError(
                    "every inference tensor must be device-local with leading "
                    "active-lane extent"
                )
        parameters = {
            name: value.index_select(0, active_lanes)
            for name, value in self.owner.parameters.items()
        }
        buffers = {
            name: value.index_select(0, active_lanes)
            for name, value in self.owner.buffers.items()
        }
        keywords = {} if static_keywords is None else dict(static_keywords)

        def one_lane(
            lane_parameters: Mapping[str, Tensor],
            lane_buffers: Mapping[str, Tensor],
            *arguments: Tensor,
        ) -> dict[str, Tensor]:
            result = functional_call(
                self.owner._code,
                (lane_parameters, lane_buffers),
                (*static_arguments, *arguments),
                keywords,
                strict=True,
            )
            encoded = dict(self.encode_output(result))
            if not encoded or any(
                not isinstance(value, Tensor) for value in encoded.values()
            ):
                raise TypeError(
                    "replica inference encoder must return a nonempty tensor mapping"
                )
            return encoded

        operation = vmap(
            one_lane,
            in_dims=(0, 0, *(0 for _ in lane_arguments)),
            randomness="error",
        )
        with torch.no_grad():
            result = operation(parameters, buffers, *lane_arguments)
        if any(
            value.device != self.owner.device
            or value.shape[0] != active
            or value.requires_grad
            for value in result.values()
        ):
            raise ReplicaError("replica inference output lost lane ownership")
        return MappingProxyType(dict(result))


class DistributedSUM:
    """NCCL-only active-lane agreement and dense gradient SUM."""

    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError("distributed SUM requires an initialized process group")
        self.group = group
        if dist.get_backend(group) != "nccl":
            raise ValueError("queued distributed training is CUDA/NCCL-only")

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.group)

    def agree_active(self, active: Tensor) -> None:
        if active.device.type != "cuda" or active.dtype != torch.bool:
            raise ValueError("distributed active mask must be CUDA bool")
        low = active.to(torch.uint8)
        high = low.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=self.group)
        if not bool(low.eq(high).all()):
            raise ReplicaError("distributed ranks selected asymmetric active lanes")

    def agree_failures(self, failed: Tensor) -> Tensor:
        if failed.device.type != "cuda" or failed.dtype != torch.bool:
            raise ValueError("distributed failure mask must be CUDA bool")
        common = failed.to(torch.uint8)
        dist.all_reduce(common, op=dist.ReduceOp.MAX, group=self.group)
        return common.bool()

    def sum_gradients(self, gradients: Tensor, active: Tensor) -> None:
        if (
            gradients.device.type != "cuda"
            or gradients.dtype != torch.float32
            or gradients.ndim != 2
            or active.shape != gradients.shape[:1]
        ):
            raise ValueError("distributed gradients must be dense CUDA FP32 [K,N]")
        self.agree_active(active)
        gradients.masked_fill_(~active[:, None], 0.0)
        dist.all_reduce(gradients, op=dist.ReduceOp.SUM, group=self.group)


@dataclass(frozen=True)
class ReplicaStepResult:
    lanes: tuple[int, ...]
    optimizer_steps: tuple[OptimizerStep | None, ...]
    failures: Mapping[int, str]
    versions: tuple[int, ...]


@dataclass(frozen=True)
class ReplicaMemoryReport:
    """Exact named resident bytes for stacked model and learner state."""

    parameter_bytes: int
    buffer_bytes: int
    gradient_bytes: int
    optimizer_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.buffer_bytes
            + self.gradient_bytes
            + self.optimizer_bytes
        )

    def assert_reconciled(
        self,
        modules: ReplicaModuleOwner,
        optimizers: ReplicaOptimizerGroup,
    ) -> None:
        expected = optimizers.memory_report()
        if self != expected or optimizers.modules is not modules:
            raise ReplicaError("replica named resident bytes do not reconcile")


class ReplicaOptimizerGroup:
    """One ordered dense replica-axis owner serving ordinary ``DenseAdam``."""

    def __init__(
        self,
        modules: ReplicaModuleOwner,
        config: OptimizerConfig,
        *,
        scalers: Sequence[LossScaler] | None = None,
    ) -> None:
        self.modules = modules
        count = modules.n_replica
        if scalers is None:
            scalers = tuple(LossScaler() for _ in range(count))
        if len(scalers) != count:
            raise ValueError("one loss scaler is required per replica lane")
        width = modules.parameter_count
        self.gradients = torch.zeros(
            (count, width), dtype=torch.float32, device=modules.device
        )
        self.exp_avg = torch.zeros_like(self.gradients)
        self.exp_avg_sq = torch.zeros_like(self.gradients)
        self.optimizers = tuple(
            DenseAdam(
                modules.lane_module(lane),
                config,
                scaler=scalers[lane],
                gradient_flat=self.gradients[lane],
                exp_avg=self.exp_avg[lane],
                exp_avg_sq=self.exp_avg_sq[lane],
            )
            for lane in range(count)
        )
        self.versions = torch.zeros(count, dtype=torch.int64)
        modules.assert_zero_copy()

    def memory_report(self) -> ReplicaMemoryReport:
        return ReplicaMemoryReport(
            parameter_bytes=sum(
                value.numel() * value.element_size()
                for value in self.modules.parameters.values()
            ),
            buffer_bytes=sum(
                value.numel() * value.element_size()
                for value in self.modules.buffers.values()
            ),
            gradient_bytes=self.gradients.numel() * self.gradients.element_size(),
            optimizer_bytes=(
                self.exp_avg.numel() * self.exp_avg.element_size()
                + self.exp_avg_sq.numel() * self.exp_avg_sq.element_size()
            ),
        )

    def _mask(self, lanes: Sequence[int]) -> Tensor:
        mask = torch.zeros(
            self.modules.n_replica,
            dtype=torch.bool,
            device=self.modules.device,
        )
        normalized = tuple(int(lane) for lane in lanes)
        if len(set(normalized)) != len(normalized):
            raise ValueError("active learner lanes must be unique")
        for lane in normalized:
            self.modules._check_lane(lane)
            mask[lane] = True
        return mask

    def zero_grad(self, lanes: Sequence[int]) -> None:
        for lane in lanes:
            self.optimizers[int(lane)].zero_grad()

    def step(
        self,
        lanes: Sequence[int],
        *,
        distributed: DistributedSUM | None = None,
    ) -> tuple[OptimizerStep | None, ...]:
        active = self._mask(lanes)
        if distributed is not None:
            distributed.sum_gradients(self.gradients, active)
        result: list[OptimizerStep | None] = [None] * self.modules.n_replica
        for lane in lanes:
            normalized = int(lane)
            step = self.optimizers[normalized].step()
            result[normalized] = step
            if step.applied:
                self.versions[normalized] += 1
        return tuple(result)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "pbg-queued-replica-optimizer-v1",
            "n_replica": self.modules.n_replica,
            "versions": self.versions.clone(),
            "lanes": tuple(optimizer.state_dict() for optimizer in self.optimizers),
            "scalers": tuple(
                optimizer.scaler.state_dict() for optimizer in self.optimizers
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {"schema", "n_replica", "versions", "lanes", "scalers"}
        if (
            set(state) != expected
            or state["schema"] != "pbg-queued-replica-optimizer-v1"
        ):
            raise ValueError("replica optimizer state schema is invalid")
        if state["n_replica"] != self.modules.n_replica:
            raise ValueError("replica optimizer lane count differs")
        versions = state["versions"]
        lanes = state["lanes"]
        scalers = state["scalers"]
        if (
            not isinstance(versions, Tensor)
            or versions.shape != self.versions.shape
            or versions.dtype != torch.int64
            or not isinstance(lanes, (tuple, list))
            or len(lanes) != self.modules.n_replica
            or not isinstance(scalers, (tuple, list))
            or len(scalers) != self.modules.n_replica
        ):
            raise ValueError("replica optimizer state has invalid lane data")
        self.versions.copy_(versions)
        for lane, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(lanes[lane])
            optimizer.scaler = LossScaler.from_state_dict(scalers[lane])


class ReplicaLearnerScheduler:
    """Schedule the existing lane learner operation without padding."""

    def __init__(
        self,
        optimizers: ReplicaOptimizerGroup,
        *,
        distributed: DistributedSUM | None = None,
    ) -> None:
        self.optimizers = optimizers
        self.distributed = distributed

    def run_step(
        self,
        active_lanes: Sequence[int],
        forward_loss_backward: Callable[[int, nn.Module, DenseAdam], Any],
    ) -> ReplicaStepResult:
        lanes = tuple(sorted(int(lane) for lane in active_lanes))
        self.optimizers._mask(lanes)
        self.optimizers.zero_grad(lanes)
        failures: dict[int, str] = {}
        for lane in lanes:
            try:
                forward_loss_backward(
                    lane,
                    self.optimizers.modules.lane_module(lane),
                    self.optimizers.optimizers[lane],
                )
            except Exception as exc:  # noqa: BLE001 - lane-local disposition
                failures[lane] = f"{type(exc).__name__}: {exc}"
                self.optimizers.gradients[lane].zero_()
        if self.distributed is not None:
            local = torch.zeros(
                self.optimizers.modules.n_replica,
                dtype=torch.bool,
                device=self.optimizers.modules.device,
            )
            for lane in failures:
                local[lane] = True
            common = self.distributed.agree_failures(local)
            for lane in lanes:
                if bool(common[lane]) and lane not in failures:
                    failures[lane] = "failure on another distributed rank"
                    self.optimizers.gradients[lane].zero_()
        successful = tuple(lane for lane in lanes if lane not in failures)
        steps = self.optimizers.step(
            successful,
            distributed=self.distributed,
        )
        return ReplicaStepResult(
            lanes,
            steps,
            MappingProxyType(failures),
            tuple(int(value) for value in self.optimizers.versions.tolist()),
        )


__all__ = [
    "DistributedSUM",
    "FunctionalReplicaInference",
    "ReplicaError",
    "ReplicaLearnerScheduler",
    "ReplicaMemoryReport",
    "ReplicaModuleOwner",
    "ReplicaOptimizerGroup",
    "ReplicaStepResult",
]
