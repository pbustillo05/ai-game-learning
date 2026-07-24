"""Functional learner primitives for independent replicas on one CUDA device."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad_and_value, stack_module_state, vmap

from pbg.core.replica_optimizer import ReplicaAdamW, ReplicaOptimizerStepResult


class ReplicaTransformUnsupported(RuntimeError):
    """The selected model kernel cannot execute under the replica transform."""


@dataclass
class StackedModuleReplicas:
    """One stateless module plus physically stacked parameters and buffers."""

    module: nn.Module
    parameters: dict[str, Tensor]
    buffers: dict[str, Tensor]
    replicas: int

    @classmethod
    def from_modules(cls, modules: Sequence[nn.Module]) -> StackedModuleReplicas:
        if not modules:
            raise ValueError("at least one module replica is required")
        devices = {
            value.device
            for module in modules
            for value in (*module.parameters(), *module.buffers())
        }
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("module replicas must occupy one CUDA device")
        training_modes = {module.training for module in modules}
        if len(training_modes) != 1:
            raise ValueError("all module replicas must share train/eval mode")
        parameters, buffers = stack_module_state(list(modules))
        if any(value.dtype != torch.float32 for value in parameters.values()):
            raise TypeError("replica learner requires fp32 master parameters")
        # The base module supplies structure and code only. Concrete storage is
        # owned exclusively by the stacked state above.
        base = copy.deepcopy(modules[0]).to("meta")
        return cls(base, dict(parameters), dict(buffers), len(modules))

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "pbg-stacked-module-replicas-v1",
            "replicas": self.replicas,
            "parameters": {
                name: value.detach().clone()
                for name, value in self.parameters.items()
            },
            "buffers": {
                name: value.detach().clone() for name, value in self.buffers.items()
            },
        }

    def with_module(self, module: nn.Module) -> StackedModuleReplicas:
        """Return another stateless code view over the same stacked storage."""

        base = copy.deepcopy(module).to("meta")
        parameter_names = set(dict(base.named_parameters()))
        buffer_names = set(dict(base.named_buffers()))
        if parameter_names != set(self.parameters):
            raise ValueError("replica module view has a different parameter layout")
        if buffer_names != set(self.buffers):
            raise ValueError("replica module view has a different buffer layout")
        return StackedModuleReplicas(
            module=base,
            parameters=self.parameters,
            buffers=self.buffers,
            replicas=self.replicas,
        )

    def frozen_copy(self) -> StackedModuleReplicas:
        """Clone one detached stacked reference without ordinary model copies."""

        module = copy.deepcopy(self.module)
        module.eval()
        return StackedModuleReplicas(
            module=module,
            parameters={
                name: value.detach().clone()
                for name, value in self.parameters.items()
            },
            buffers={
                name: value.detach().clone() for name, value in self.buffers.items()
            },
            replicas=self.replicas,
        )

    @torch.no_grad()
    def copy_lane_from(
        self, destination_lane: int, source: StackedModuleReplicas, source_lane: int
    ) -> None:
        """Refresh one lane from another layout-identical stacked owner."""

        if not 0 <= destination_lane < self.replicas:
            raise IndexError("destination replica lane is outside [0,K)")
        if not 0 <= source_lane < source.replicas:
            raise IndexError("source replica lane is outside [0,K)")
        if self.parameters.keys() != source.parameters.keys():
            raise ValueError("replica parameter layouts differ")
        if self.buffers.keys() != source.buffers.keys():
            raise ValueError("replica buffer layouts differ")
        for name, target in self.parameters.items():
            target[destination_lane].copy_(source.parameters[name][source_lane])
        for name, target in self.buffers.items():
            target[destination_lane].copy_(source.buffers[name][source_lane])

    def lane_module_state(
        self, lane: int, *, strip_prefix: str = ""
    ) -> dict[str, Tensor]:
        """Return one ordinary module state for an independent checkpoint."""

        if not 0 <= lane < self.replicas:
            raise IndexError("replica lane is outside [0,K)")
        persistent = set(self.module.state_dict())
        combined = {**self.parameters, **self.buffers}
        result: dict[str, Tensor] = {}
        for name in persistent:
            if name not in combined:
                raise RuntimeError(
                    f"stacked module state omitted persistent tensor {name!r}"
                )
            target = name
            if strip_prefix:
                if not name.startswith(strip_prefix):
                    raise ValueError(
                        f"state tensor {name!r} lacks prefix {strip_prefix!r}"
                    )
                target = name.removeprefix(strip_prefix)
            result[target] = combined[name][lane].detach().clone()
        return result

    @torch.no_grad()
    def load_lane_module_state(
        self,
        lane: int,
        state: Mapping[str, Tensor],
        *,
        add_prefix: str = "",
    ) -> None:
        """Restore one ordinary module state into its stacked lane."""

        if not 0 <= lane < self.replicas:
            raise IndexError("replica lane is outside [0,K)")
        combined = {**self.parameters, **self.buffers}
        expected = {
            name.removeprefix(add_prefix)
            if add_prefix and name.startswith(add_prefix)
            else name
            for name in combined
            if name in self.module.state_dict()
        }
        if set(state) != expected:
            raise ValueError("ordinary lane state has a different tensor layout")
        for ordinary_name, value in state.items():
            stacked_name = f"{add_prefix}{ordinary_name}"
            target = combined.get(stacked_name)
            if target is None:
                raise KeyError(f"stacked lane has no tensor {stacked_name!r}")
            if target[lane].shape != value.shape:
                raise ValueError(f"lane tensor {ordinary_name!r} has the wrong shape")
            target[lane].copy_(value)

    def lane_module(self, lane: int) -> nn.Module:
        """Materialize a code object whose tensors are zero-copy lane views.

        The returned module is inference-only. Its parameters are detached
        views of the stacked master storage, so rollout observes every later
        replica optimizer update without retaining a second model copy.
        """

        if not 0 <= lane < self.replicas:
            raise IndexError("replica lane is outside [0,K)")
        result = copy.deepcopy(self.module)

        def owner(qualified_name: str) -> tuple[nn.Module, str]:
            path, _, leaf = qualified_name.rpartition(".")
            module = result.get_submodule(path) if path else result
            return module, leaf

        for name, stacked in self.parameters.items():
            module, leaf = owner(name)
            module._parameters[leaf] = nn.Parameter(
                stacked[lane].detach(), requires_grad=False
            )
        for name, stacked in self.buffers.items():
            module, leaf = owner(name)
            module._buffers[leaf] = stacked[lane].detach()
        if any(value.device.type == "meta" for value in result.parameters()):
            raise RuntimeError("replica lane module retained a meta parameter")
        return result


LossFunction = Callable[..., Tensor]


def _policy_output_tensors(output: Any) -> dict[str, Tensor]:
    raw = getattr(output, "raw", None)
    value = getattr(output, "v", None)
    if not isinstance(raw, Mapping) or not isinstance(value, Tensor):
        raise TypeError("replica reference output must expose raw tensor fields and v")
    tensors = {f"raw.{name}": tensor for name, tensor in raw.items()}
    if any(not isinstance(tensor, Tensor) for tensor in tensors.values()):
        raise TypeError("replica reference raw fields must be tensors")
    tensors["v"] = value
    for name in ("t", "c"):
        tensor = getattr(output, name, None)
        if tensor is not None:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"replica reference {name} field must be a tensor")
            tensors[name] = tensor
    return tensors


def replica_forward(
    replicas: StackedModuleReplicas,
    model_arguments: Sequence[Any],
) -> dict[str, Tensor]:
    """Run one no-grad functional forward per lane and stack its tensor leaves."""

    if not model_arguments:
        raise ValueError("replica forward requires at least one lane argument")
    preparer = getattr(replicas.module, "prepare_replica_arguments", None)
    if callable(preparer):
        model_arguments = preparer(
            replicas.parameters,
            replicas.buffers,
            tuple(model_arguments),
        )

    def one_lane(
        parameters: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        *arguments: Any,
    ) -> dict[str, Tensor]:
        output = functional_call(
            replicas.module, (parameters, buffers), arguments, strict=True
        )
        return _policy_output_tensors(output)

    transformed = vmap(
        one_lane,
        in_dims=(0, 0, *(0 for _ in model_arguments)),
        randomness="error",
    )
    try:
        with torch.no_grad():
            output = transformed(
                replicas.parameters, replicas.buffers, *model_arguments
            )
    except RuntimeError as exc:
        message = str(exc)
        if not any(
            marker in message
            for marker in ("vmap", "functorch", "batching rule", "Batching rule")
        ):
            raise
        raise ReplicaTransformUnsupported(
            "reference forward is not compatible with the replica transform"
        ) from exc
    return {name: tensor.detach() for name, tensor in output.items()}


def replica_grad_and_value(
    replicas: StackedModuleReplicas,
    loss_fn: LossFunction,
    model_arguments: Sequence[Any],
    loss_arguments: Sequence[Any] = (),
) -> tuple[dict[str, Tensor], Tensor]:
    """Return ``grad(loss[k], params[k])`` and one scalar loss per lane.

    Every tensor leaf in the argument sequences must carry the same leading
    ``K`` dimension. Random draws inside the model are rejected; stochastic
    inputs must be materialized from each replica's independent RNG before this
    call.
    """

    if not model_arguments:
        raise ValueError("replica learner requires at least one lane argument")
    preparer = getattr(replicas.module, "prepare_replica_arguments", None)
    if callable(preparer):
        model_arguments = preparer(
            replicas.parameters,
            replicas.buffers,
            tuple(model_arguments),
        )
    loss_preparer = getattr(loss_fn, "prepare_replica_loss_arguments", None)
    if callable(loss_preparer):
        loss_arguments = loss_preparer(tuple(loss_arguments))
    model_count = len(model_arguments)
    lane_arguments = (*model_arguments, *loss_arguments)

    def one_lane(
        parameters: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        *arguments: Any,
    ) -> Tensor:
        model_args = arguments[:model_count]
        output = functional_call(
            replicas.module, (parameters, buffers), model_args, strict=True
        )
        loss = loss_fn(output, *arguments)
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ValueError("replica loss function must return one scalar per lane")
        return loss

    transformed = vmap(
        grad_and_value(one_lane),
        in_dims=(0, 0, *(0 for _ in lane_arguments)),
        randomness="error",
    )
    try:
        gradients, losses = transformed(
            replicas.parameters, replicas.buffers, *lane_arguments
        )
    except RuntimeError as exc:
        message = str(exc)
        transform_markers = (
            "vmap",
            "functorch",
            "batching rule",
            "Batching rule",
            "BatchedFallback",
            "saved tensor hooks",
            "LSE is not correctly aligned",
        )
        if not any(marker in message for marker in transform_markers):
            raise
        raise ReplicaTransformUnsupported(
            "model backward is not compatible with the replica transform; "
            "profile a grouped/batched kernel for this execution family"
        ) from exc
    return dict(gradients), losses


@dataclass(frozen=True)
class ReplicaLearnerStep:
    losses: Tensor
    optimizer: ReplicaOptimizerStepResult


class FunctionalReplicaLearner:
    """Benchmark boundary joining lane-wise autograd to lane-wise AdamW."""

    def __init__(
        self,
        replicas: StackedModuleReplicas,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.01,
        grad_clip: float,
    ) -> None:
        self.replicas = replicas
        self.grad_clip = float(grad_clip)
        self.optimizer = ReplicaAdamW(
            replicas.parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def step(
        self,
        loss_fn: LossFunction,
        model_arguments: Sequence[Any],
        loss_arguments: Sequence[Any] = (),
    ) -> ReplicaLearnerStep:
        gradients, losses = replica_grad_and_value(
            self.replicas, loss_fn, model_arguments, loss_arguments
        )
        outcome = self.optimizer.step(gradients, grad_clip=self.grad_clip)
        return ReplicaLearnerStep(losses=losses.detach(), optimizer=outcome)


__all__ = [
    "FunctionalReplicaLearner",
    "ReplicaLearnerStep",
    "ReplicaTransformUnsupported",
    "StackedModuleReplicas",
    "replica_forward",
    "replica_grad_and_value",
]
