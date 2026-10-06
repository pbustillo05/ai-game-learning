"""Replica ownership, no-grad inference, and distributed learner scheduling.

Replicas do not define another learner or optimizer.  This module only owns
the leading lane axis, binds zero-copy lane modules to it, schedules the
existing per-lane learner operation, and synchronizes the resulting ordered
dense FP32 gradient rows when distributed execution requires it.
"""

from __future__ import annotations

import copy
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from pbg.core.optimization import (
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
        source_parameters = tuple(dict(module.named_parameters()) for module in modules)
        if any(value.dtype != torch.float32 for lane in source_parameters for value in lane.values()):
            raise TypeError("replica master parameters must be FP32")
        width = sum(value.numel() for value in source_parameters[0].values())
        device = next(iter(devices))
        # Containers are storage, not autograd leaves. Only named views own grads.
        self.master = torch.empty((len(modules), width), dtype=torch.float32, device=device)
        self.gradients = (torch.zeros_like(self.master)
                          if any(value.requires_grad for value in source_parameters[0].values())
                          else self.master.new_empty(0))
        self.parameters: dict[str, Tensor] = {}
        self.parameter_layout: dict[str, tuple[int, int, tuple[int, ...]]] = {}
        offset = 0
        with torch.no_grad():
            for name in parameter_names:
                original = source_parameters[0][name]
                end = offset + original.numel()
                shape = tuple(original.shape)
                view = self.master[:, offset:end].view(len(modules), *shape)
                for lane, values in enumerate(source_parameters):
                    if values[name].shape != original.shape or values[name].requires_grad != original.requires_grad:
                        raise ValueError("replica parameter shapes or trainability differ")
                    view[lane].copy_(values[name])
                leaf = nn.Parameter(view, requires_grad=original.requires_grad)
                if original.requires_grad:
                    leaf.grad = self.gradients[:, offset:end].view_as(leaf)
                self.parameters[name] = leaf
                self.parameter_layout[name] = (offset, end, shape)
                offset = end
        source_buffers = tuple(dict(module.named_buffers()) for module in modules)
        self.buffers = {
            name: torch.stack(tuple(values[name] for values in source_buffers))
            for name in buffer_names
        }
        self.n_replica = len(modules)
        self.device = next(iter(devices))
        self.training = modules[0].training
        self._parameter_names = parameter_names
        self._buffer_names = buffer_names
        self._persistent_names = state_names
        self._code = copy.deepcopy(modules[0]).to("meta")
        self._lanes: dict[int, nn.Module] = {}
        self._cohorts: dict[tuple[int, int], nn.Module] = {}

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
                requires_grad=self.parameters[name].requires_grad,
            )
            if target._parameters[leaf].requires_grad:
                begin, end, shape = self.parameter_layout[name]
                target._parameters[leaf].grad = self.gradients[lane, begin:end].view(shape)
        for name in self._buffer_names:
            target, leaf = _owner(module, name)
            target._buffers[leaf] = self.buffers[name][lane]
        module.train(self.training)
        if any(value.device.type == "meta" for value in module.state_dict().values()):
            raise ReplicaError("zero-copy lane binding retained meta storage")
        module._replica_owner_ref = weakref.ref(self)
        module._replica_lane = lane
        self._lanes[lane] = module
        return module

    def cohort_module(self, start: int, count: int) -> nn.Module:
        """Bind an exact contiguous lane run to named stacked leaf/grad views."""

        self._check_lane(start)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or start + count > self.n_replica:
            raise ValueError("cohort must be a nonempty contiguous resident lane run")
        key = (start, count)
        if key in self._cohorts:
            return self._cohorts[key]
        module = copy.deepcopy(self._code)
        for name, (begin, end, shape) in self.parameter_layout.items():
            target, leaf = _owner(module, name)
            parameter = nn.Parameter(
                self.master[start:start + count, begin:end].view(count, *shape),
                requires_grad=self.parameters[name].requires_grad,
            )
            if parameter.requires_grad:
                parameter.grad = self.gradients[start:start + count, begin:end].view_as(parameter)
            target._parameters[leaf] = parameter
        # Poker buffers are immutable geometry, shared across decision batches.
        # Other tensor programs explicitly declare which buffers are shared.
        shared = set(getattr(module, "replica_shared_buffers", ()))
        for name in self._buffer_names:
            target, leaf = _owner(module, name)
            target._buffers[leaf] = (self.buffers[name][start] if name in shared
                                    else self.buffers[name][start:start + count])
        module.train(self.training)
        self._cohorts[key] = module
        return module

    def assert_zero_copy(self) -> None:
        """Prove every materialized lane still aliases its stacked owner."""

        if self.master.requires_grad or self.gradients.requires_grad:
            raise ReplicaError("flat storage must not be a trainable leaf")
        for name, (begin, end, shape) in self.parameter_layout.items():
            parameter = self.parameters[name]
            if parameter.data_ptr() != self.master[:, begin:end].data_ptr():
                raise ReplicaError("named parameter lost flat master ownership")
            if parameter.requires_grad and (parameter.grad is None or parameter.grad.data_ptr() != self.gradients[:, begin:end].data_ptr()):
                raise ReplicaError("named parameter lost flat gradient ownership")
        for lane, module in self._lanes.items():
            for name, parameter in module.named_parameters():
                if parameter.data_ptr() != self.parameters[name][lane].data_ptr():
                    raise ReplicaError("a lane parameter lost its zero-copy binding")
            for name, buffer in module.named_buffers():
                if buffer.data_ptr() != self.buffers[name][lane].data_ptr():
                    raise ReplicaError("a lane buffer lost its zero-copy binding")

        for (start, count), module in self._cohorts.items():
            for name, parameter in module.named_parameters():
                begin, end, _shape = self.parameter_layout[name]
                expected = self.master[start:start + count, begin:end]
                if parameter.data_ptr() != expected.data_ptr():
                    raise ReplicaError("cohort parameter lost its zero-copy binding")
                if parameter.requires_grad and (parameter.grad is None or parameter.grad.data_ptr() != self.gradients[start:start + count, begin:end].data_ptr()):
                    raise ReplicaError("cohort gradient lost its zero-copy binding")

    def layout_manifest(self) -> dict[str, object]:
        """Logical-to-physical parameter map, including packed QKV ordering."""

        return {
            "schema": "pbg-flat-replica-layout-v1",
            "replicas": self.n_replica,
            "physical_width": self.master.shape[1],
            "padding_elements": 0,
            "parameters": tuple({
                "name": name, "begin": begin, "end": end, "shape": shape,
                "packing": "Q,K,V" if ".w_qkv." in name else "native",
            } for name, (begin, end, shape) in self.parameter_layout.items()),
        }

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
            "schema": "pbg-queued-replica-module-v2",
            "n_replica": self.n_replica,
            "lanes": tuple(
                self.lane_state_dict(lane) for lane in range(self.n_replica)
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            set(state) != {"schema", "n_replica", "lanes"}
            or state["schema"] != "pbg-queued-replica-module-v2"
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
    """No-grad calls to the shared explicit-K program on contiguous lane runs."""

    def __init__(self, owner: ReplicaModuleOwner, *, encode_output: OutputEncoder) -> None:
        self.owner = owner
        self.encode_output = encode_output

    def __call__(
        self, active_lanes: Tensor, *lane_arguments: Tensor,
        static_arguments: tuple[object, ...] = (),
        static_keywords: Mapping[str, object] | None = None,
        lane_ids: Sequence[int] | None = None,
    ) -> Mapping[str, Tensor]:
        if (active_lanes.device != self.owner.device or active_lanes.dtype != torch.int64
            or active_lanes.ndim != 1 or not active_lanes.is_contiguous() or active_lanes.numel() == 0):
            raise ValueError("active lanes must be a nonempty contiguous CUDA int64 row")
        # Production FIFO callers already own this bounded host schedule. The
        # optional conversion is only for standalone API/diagnostic calls.
        lanes = tuple(active_lanes.cpu().tolist()) if lane_ids is None else tuple(lane_ids)
        if len(lanes) != active_lanes.numel() or len(set(lanes)) != len(lanes):
            raise ValueError("active lanes must be unique and match the host cohort")
        for lane in lanes:
            self.owner._check_lane(lane)
        if lane_ids is not None:
            expected = torch.tensor(lanes, dtype=torch.int64, device=active_lanes.device)
            torch._assert_async((active_lanes == expected).all(), "device lanes disagree with host cohort")
        if not lane_arguments or any(
            not isinstance(arg, Tensor) or arg.device != self.owner.device
            or arg.ndim < 2 or arg.shape[:2] != lane_arguments[0].shape[:2]
            or arg.shape[0] != len(lanes) for arg in lane_arguments
        ):
            raise ValueError("inference arguments require aligned [active,rows,...] tensors")
        rows = lane_arguments[0].shape[1]
        outputs: dict[str, list[Tensor]] = {}
        begin = 0
        with torch.no_grad():
            while begin < len(lanes):
                end = begin + 1
                while end < len(lanes) and lanes[end] == lanes[end - 1] + 1:
                    end += 1
                module = self.owner.cohort_module(lanes[begin], end - begin)
                arguments = tuple(arg[begin:end].flatten(0, 1) for arg in lane_arguments)
                encoded = self.encode_output(module(*static_arguments, *arguments, **dict(static_keywords or {})))
                if not encoded or any(not isinstance(value, Tensor) for value in encoded.values()):
                    raise TypeError("replica inference encoder must return tensor fields")
                for name, value in encoded.items():
                    if value.shape[0] != (end - begin) * rows or value.device != self.owner.device:
                        raise ReplicaError("cohort inference changed decision ownership")
                    outputs.setdefault(name, []).append(value.reshape(end - begin, rows, *value.shape[1:]))
                begin = end
        return MappingProxyType({name: torch.cat(values, dim=0) if len(values) > 1 else values[0]
                                 for name, values in outputs.items()})


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

    def barrier(self) -> None:
        """Keep rank-varying work ahead of shared lazy CUDA modules phase-safe."""

        dist.barrier(
            group=self.group,
            device_ids=[torch.cuda.current_device()],
        )

    def maximum_steps(self, local_steps: int, *, device: torch.device) -> int:
        """Return one rank-common optimizer-call budget."""

        if isinstance(local_steps, bool) or local_steps < 0:
            raise ValueError("local optimizer step count must be nonnegative")
        count = torch.tensor([local_steps], dtype=torch.int64, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.group)
        return int(count.item())


@dataclass(frozen=True)
class ReplicaMemoryReport:
    """Exact named resident bytes for stacked model and learner state."""

    parameter_bytes: int
    buffer_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    transaction_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.buffer_bytes
            + self.gradient_bytes
            + self.optimizer_bytes
            + self.transaction_bytes
        )

    def assert_reconciled(
        self,
        modules: ReplicaModuleOwner,
        optimizers: ReplicaOptimizerGroup,
    ) -> None:
        tensors = optimizers.resident_tensors()
        unique = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                  for value in tensors}
        if self.total_bytes != sum(unique.values()) or optimizers.modules is not modules:
            raise ReplicaError("replica named resident bytes do not reconcile")


class ReplicaOptimizerGroup:
    """One vectorized dense optimizer with zero-copy lane control views."""

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
        self.optimizer = DenseAdam(
            tuple(modules.lane_module(lane) for lane in range(count)), config,
            scalers=tuple(scalers), parameter_flat=modules.master,
            gradient_flat=modules.gradients,
        )
        self.gradients = modules.gradients
        self.exp_avg = self.optimizer.first
        self.exp_avg_sq = self.optimizer.second
        self.optimizers = self.optimizer.lanes
        self.versions = torch.zeros(count, dtype=torch.int64, device=modules.device)
        self._version_snapshot = torch.empty_like(self.versions)
        self._transaction_versions: Tensor | None = None
        modules.assert_zero_copy()

    def _optimizer_tensors(self) -> tuple[Tensor, ...]:
        owner = self.optimizer
        return (owner.first, owner.second, owner.steps, owner.trackers,
                owner.enabled, owner.decoupled, owner.fatal, owner.controls, self.versions)

    def resident_tensors(self) -> tuple[Tensor, ...]:
        """Owning allocations, excluding named leaf and lane aliases."""
        return (self.modules.master, *self.modules.buffers.values(), self.gradients,
                *self._optimizer_tensors(), *self.optimizer._snapshots,
                self._version_snapshot)

    def memory_report(self) -> ReplicaMemoryReport:
        def size(values):
            return sum(value.numel() * value.element_size() for value in values)

        report = ReplicaMemoryReport(
            parameter_bytes=size((self.modules.master,)),
            buffer_bytes=size(self.modules.buffers.values()),
            gradient_bytes=size((self.gradients,)),
            optimizer_bytes=size(self._optimizer_tensors()),
            transaction_bytes=self.optimizer.transaction_bytes + size((self._version_snapshot,)),
        )
        report.assert_reconciled(self.modules, self)
        return report

    def begin_transaction(self) -> None:
        """Create one rollback point spanning every learner lane."""

        if self._transaction_versions is not None:
            raise ReplicaError("replica optimizer transaction is already active")
        self.optimizer.begin_transaction()
        self._version_snapshot.copy_(self.versions)
        self._transaction_versions = self._version_snapshot

    def rollback_transaction(self) -> None:
        """Restore all lanes after any local or rank-common learner failure."""

        if self._transaction_versions is None:
            raise ReplicaError("replica optimizer transaction is not active")
        self.optimizer.rollback_transaction()
        self.versions.copy_(self._transaction_versions)
        self._transaction_versions = None

    def commit_transaction(self) -> None:
        """Publish every lane transition together."""

        if self._transaction_versions is None:
            raise ReplicaError("replica optimizer transaction is not active")
        self.optimizer.commit_transaction()
        self._transaction_versions = None

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
        active = self._mask(lanes)
        self.gradients.masked_fill_(active[:, None], 0.0)

    def step_device(self, lanes: Sequence[int], *, distributed: DistributedSUM | None = None):
        active = self._mask(lanes)
        if distributed is not None:
            distributed.sum_gradients(self.gradients, active)
        result = self.optimizer.step_device(active)
        self.versions.add_(result[0].to(self.versions.dtype))
        return result

    def step(self, lanes: Sequence[int], *, distributed: DistributedSUM | None = None) -> tuple[OptimizerStep | None, ...]:
        result = DenseAdam.materialize_steps(self.step_device(lanes, distributed=distributed))
        enabled = set(lanes)
        return tuple(value if lane in enabled else None for lane, value in enumerate(result))

    def state_dict(self) -> dict[str, object]:
        if self._transaction_versions is not None:
            raise ReplicaError(
                "cannot checkpoint an active replica optimizer transaction"
            )
        return {
            "schema": "pbg-queued-replica-optimizer-v2",
            "n_replica": self.modules.n_replica,
            "versions": self.versions.clone(),
            "lanes": tuple(optimizer.state_dict() for optimizer in self.optimizers),
            "scalers": tuple(
                optimizer.scaler.state_dict() for optimizer in self.optimizers
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._transaction_versions is not None:
            raise ReplicaError(
                "cannot restore an active replica optimizer transaction"
            )
        expected = {"schema", "n_replica", "versions", "lanes", "scalers"}
        if (
            set(state) != expected
            or state["schema"] != "pbg-queued-replica-optimizer-v2"
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

    def lane_stepper(
        self,
        lane: int,
        *,
        distributed: DistributedSUM | None = None,
    ) -> ReplicaLaneOptimizer:
        """Expose ordinary optimizer calls while retaining group ownership."""

        self.modules._check_lane(lane)
        return ReplicaLaneOptimizer(self, lane, distributed=distributed)


class ReplicaLaneOptimizer:
    """Scheduling-only view of one dense optimizer lane."""

    def __init__(
        self,
        group: ReplicaOptimizerGroup,
        lane: int,
        *,
        distributed: DistributedSUM | None,
    ) -> None:
        self.group = group
        self.lane = lane
        self.distributed = distributed

    @property
    def scaler(self) -> LossScaler:
        return self.group.optimizers[self.lane].scaler

    def zero_grad(self) -> None:
        self.group.zero_grad((self.lane,))

    def agree_failure(self, failed: bool) -> bool:
        if self.distributed is None:
            return bool(failed)
        local = torch.zeros(
            self.group.modules.n_replica,
            dtype=torch.bool,
            device=self.group.modules.device,
        )
        local[self.lane] = bool(failed)
        return bool(self.distributed.agree_failures(local)[self.lane])

    def common_step_count(self, local_steps: int) -> int:
        """Agree the number of ordered SUM/optimizer transitions across ranks."""

        if self.distributed is None:
            return int(local_steps)
        return self.distributed.maximum_steps(
            local_steps,
            device=self.group.modules.device,
        )

    def step(self) -> OptimizerStep:
        result = self.group.step(
            (self.lane,),
            distributed=self.distributed,
        )[self.lane]
        if result is None:
            raise ReplicaError("active optimizer lane did not produce a step")
        return result


__all__ = [
    "DistributedSUM",
    "FunctionalReplicaInference",
    "ReplicaError",
    "ReplicaLaneOptimizer",
    "ReplicaMemoryReport",
    "ReplicaModuleOwner",
    "ReplicaOptimizerGroup",
]
