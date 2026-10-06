"""Aligned sequential sampling and the sole queued learner operation.

The learner consumes only the sealed owners produced by the queued rollout and
target executor.  It never receives topology, rollout queues, leaves, or work
arenas.  A physical batch is one CUDA index vector applied to the five aligned
owner families, followed by one model forward, one loss, and an immediate
backward.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor

from pbg.core.capacity import GeneratedOwners
from pbg.core.contracts import (
    ModeSchema,
    PrivateEstimator,
    RolloutMode,
)
from pbg.core.cuda_alias import alias_table_cuda
from pbg.core.storage import (
    DeviceDestinationRequest,
    MonotoneSlab,
    PublicationTransaction,
)


class LearnerError(RuntimeError):
    """A sealed learner input or loss contract was violated."""


@dataclass(frozen=True)
class LearnerDTypes:
    """Explicit numerical boundaries shared by every learner lane."""

    model: torch.dtype
    control: torch.dtype = torch.float32
    posterior: torch.dtype = torch.float32
    gradient: torch.dtype = torch.float32
    optimizer: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.model not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("model dtype must be FP32, FP16, or BF16")
        if any(
            value != torch.float32
            for value in (
                self.control,
                self.posterior,
                self.gradient,
                self.optimizer,
            )
        ):
            raise ValueError(
                "learner controls, posterior, gradients, and optimizer must be FP32"
            )


def allocate_statistical_counts(
    mass: Tensor,
    variance: Tensor,
    seconds_per_draw: Tensor,
    total_count: int,
) -> Tensor:
    """Deterministic positive-stratum cost-aware integer allocation."""

    if (
        mass.ndim != 1
        or variance.shape != mass.shape
        or seconds_per_draw.shape != mass.shape
        or mass.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("allocation inputs must be aligned floating vectors")
    if (
        not bool(torch.isfinite(mass).all())
        or bool((mass < 0).any())
        or not bool(torch.isfinite(variance).all())
        or bool((variance < 0).any())
        or not bool(torch.isfinite(seconds_per_draw).all())
        or bool((seconds_per_draw <= 0).any())
    ):
        raise ValueError("allocation mass, variance, and cost are invalid")
    if isinstance(total_count, bool) or not isinstance(total_count, int):
        raise TypeError("total_count must be an integer")
    active = mass > 0
    active_count = int(active.sum())
    if total_count < active_count or total_count <= 0:
        raise ValueError("total_count cannot cover every positive-mass stratum")
    counts = active.to(torch.int64)
    remaining = total_count - active_count
    if remaining == 0:
        return counts.to(torch.int32)
    score = torch.where(
        active,
        mass * torch.sqrt(variance / seconds_per_draw),
        torch.zeros_like(mass),
    )
    if not bool(score.sum() > 0):
        score = active.to(score)
    raw = remaining * score / score.sum()
    extra = torch.floor(raw).to(torch.int64)
    counts += extra
    leftover = remaining - int(extra.sum())
    if leftover:
        fractional = raw - extra
        order = torch.argsort(fractional, descending=True, stable=True)
        counts[order[:leftover]] += 1
    return counts.to(torch.int32)


@dataclass
class ReferenceOutputCache:
    """Admitted immutable reference outputs for one exact sealed presentation."""

    fields: Mapping[str, Tensor]
    allocated_bytes: int
    released: bool = False

    @staticmethod
    def eligible(
        *,
        byte_budget: int,
        resident_bytes: int,
        build_bytes: int,
        payload_bytes: int,
        predicted_saved_seconds: float,
        predicted_build_seconds: float,
    ) -> bool:
        if min(byte_budget, resident_bytes, build_bytes, payload_bytes) < 0:
            raise ValueError("reference cache byte values must be nonnegative")
        if (
            not math.isfinite(predicted_saved_seconds)
            or not math.isfinite(predicted_build_seconds)
            or min(predicted_saved_seconds, predicted_build_seconds) < 0
        ):
            raise ValueError("reference cache time predictions must be nonnegative")
        return (
            resident_bytes + build_bytes + payload_bytes <= byte_budget
            and predicted_saved_seconds > predicted_build_seconds
        )

    @classmethod
    def admit(
        cls,
        fields: Mapping[str, Tensor],
        *,
        byte_budget: int,
        resident_bytes: int,
        build_bytes: int,
        predicted_saved_seconds: float,
        predicted_build_seconds: float,
    ) -> ReferenceOutputCache | None:
        payload = sum(value.numel() * value.element_size() for value in fields.values())
        if not cls.eligible(
            byte_budget=byte_budget,
            resident_bytes=resident_bytes,
            build_bytes=build_bytes,
            payload_bytes=payload,
            predicted_saved_seconds=predicted_saved_seconds,
            predicted_build_seconds=predicted_build_seconds,
        ):
            return None
        cached = MappingProxyType(
            {name: value.detach().clone() for name, value in fields.items()}
        )
        return cls(cached, payload)

    def gather(self, indices: Tensor) -> Mapping[str, Tensor]:
        if self.released:
            raise LearnerError("reference output cache was released")
        return MappingProxyType(
            {
                name: value[indices.to(torch.int64)]
                for name, value in self.fields.items()
            }
        )

    def release(self) -> None:
        self.fields = MappingProxyType({})
        self.allocated_bytes = 0
        self.released = True


def _nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ValueLossConfig:
    """Value objective with only fields applicable to its selected kind."""

    kind: str = "mse"
    coefficient: float = 0.5
    epsilon: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("mse", "clipped"):
            raise ValueError("value loss kind must be 'mse' or 'clipped'")
        _nonnegative_finite(self.coefficient, "value coefficient")
        if self.kind == "mse" and self.epsilon is not None:
            raise ValueError("MSE value loss cannot carry a clip epsilon")
        if self.kind == "clipped":
            if self.epsilon is None:
                raise ValueError("clipped value loss requires epsilon")
            _nonnegative_finite(self.epsilon, "value clip epsilon")


@dataclass(frozen=True)
class EntropyLossConfig:
    """Entropy objective with an optional independent linear horizon."""

    kind: str = "constant"
    coefficient: float = 0.01
    iterations: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("constant", "linear_decay"):
            raise ValueError("entropy kind must be 'constant' or 'linear_decay'")
        _nonnegative_finite(self.coefficient, "entropy coefficient")
        if self.kind == "constant" and self.iterations is not None:
            raise ValueError("constant entropy cannot carry an iteration horizon")
        if self.kind == "linear_decay" and (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations <= 0
        ):
            raise ValueError("linear entropy decay requires a positive horizon")

    def at(self, iteration: int) -> float:
        if iteration < 0:
            raise ValueError("iteration must be nonnegative")
        if self.kind == "constant":
            return self.coefficient
        assert self.iterations is not None
        return self.coefficient * max(0.0, 1.0 - float(iteration) / self.iterations)


@dataclass(frozen=True)
class BranchingLossConfig:
    """Public-head objective, present only for branched rollout."""

    trace_coefficient: float
    cost_coefficient: float
    trace_epsilon: float

    def __post_init__(self) -> None:
        _nonnegative_finite(self.trace_coefficient, "trace coefficient")
        _nonnegative_finite(self.cost_coefficient, "cost coefficient")
        if not math.isfinite(self.trace_epsilon) or self.trace_epsilon <= 0:
            raise ValueError("trace epsilon must be finite and positive")


@dataclass(frozen=True)
class LossConfig:
    """Closed, structured loss surface shared by every queued mode."""

    policy_clip: float = 0.2
    negative_advantage_ratio_cap: float | None = None
    value: ValueLossConfig = field(default_factory=ValueLossConfig)
    entropy: EntropyLossConfig = field(default_factory=EntropyLossConfig)
    reference_kl_coefficient: float | None = None
    branching: BranchingLossConfig | None = None

    def __post_init__(self) -> None:
        _nonnegative_finite(self.policy_clip, "policy clip")
        if self.negative_advantage_ratio_cap is not None and (
            not math.isfinite(self.negative_advantage_ratio_cap)
            or self.negative_advantage_ratio_cap <= 1 + self.policy_clip
        ):
            raise ValueError(
                "negative-advantage ratio cap must exceed the PPO upper clip"
            )
        if self.reference_kl_coefficient is not None and (
            not math.isfinite(self.reference_kl_coefficient)
            or self.reference_kl_coefficient <= 0
        ):
            raise ValueError("reference KL coefficient must be finite and positive")

    def entropy_at(self, iteration: int) -> float:
        return self.entropy.at(iteration)


@dataclass(frozen=True)
class LearnerSchedule:
    """Exact statistical presentation schedule for sequential modes."""

    batch_size: int
    epochs: int | None = 1
    gradient_steps: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if (self.epochs is None) == (self.gradient_steps is None):
            raise ValueError("select exactly one of epochs or gradient_steps")
        if self.epochs is not None and (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise ValueError("epochs must be positive when supplied")
        if self.gradient_steps is not None and (
            isinstance(self.gradient_steps, bool)
            or not isinstance(self.gradient_steps, int)
            or self.gradient_steps <= 0
        ):
            raise ValueError("gradient_steps must be positive when supplied")


@dataclass(frozen=True)
class ModelEvaluation:
    """One model result before the canonical policy reduction."""

    policy_raw: Mapping[str, Tensor]
    value: Tensor
    public_heads: Mapping[str, Tensor] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class PolicyEvaluation:
    """Selected current policy and full learner reductions."""

    selected_log_policy: Tensor
    entropy: Tensor
    reference_kl: Tensor


class ModelOperation(Protocol):
    def __call__(
        self,
        input_type: str,
        model_input: Tensor,
        sampled_private_state: Tensor,
        owner: Tensor,
    ) -> ModelEvaluation: ...


def compact_best_response_evaluation(
    learned_operation: ModelOperation,
    frozen_operation: ModelOperation,
    *,
    input_type: str,
    model_input: Tensor,
    sampled_private_state: Tensor,
    owner: Tensor,
    role_selector: Tensor,
) -> ModelEvaluation:
    """Evaluate learned/frozen BR rows once each and restore FIFO order."""

    trained = owner.to(torch.int64).eq(role_selector.to(torch.int64))
    trained_row = torch.nonzero(trained, as_tuple=False).flatten()
    frozen_row = torch.nonzero(~trained, as_tuple=False).flatten()

    def compact_call(operation: ModelOperation, row: Tensor) -> ModelEvaluation | None:
        if row.numel() == 0:
            return None
        return operation(
            input_type,
            model_input.index_select(0, row),
            sampled_private_state.index_select(0, row),
            owner.index_select(0, row),
        )

    learned = compact_call(learned_operation, trained_row)
    frozen = compact_call(frozen_operation, frozen_row)
    prototype = learned if learned is not None else frozen
    if prototype is None:
        raise LearnerError("best-response model batch is empty")

    def merge(
        learned_value: Tensor | None,
        frozen_value: Tensor | None,
    ) -> Tensor:
        source = learned_value if learned_value is not None else frozen_value
        assert source is not None
        result = source.new_zeros((trained.shape[0], *source.shape[1:]))
        if learned_value is not None:
            result.index_copy_(0, trained_row, learned_value)
        if frozen_value is not None:
            result.index_copy_(0, frozen_row, frozen_value)
        return result

    policy_names = tuple(prototype.policy_raw)
    public_names = tuple(prototype.public_heads)
    for item in (learned, frozen):
        if item is not None and (
            tuple(item.policy_raw) != policy_names
        ):
            raise LearnerError("trained and frozen BR model schemas differ")
    return ModelEvaluation(
        MappingProxyType(
            {
                name: merge(
                    None if learned is None else learned.policy_raw[name],
                    None if frozen is None else frozen.policy_raw[name],
                )
                for name in policy_names
            }
        ),
        merge(
            None if learned is None else learned.value,
            None if frozen is None else frozen.value,
        ),
        MappingProxyType(
            {
                name: merge(
                    None if learned is None else learned.public_heads[name],
                    None if frozen is None else frozen.public_heads.get(name),
                )
                for name in public_names
            }
        ),
    )


def build_reference_output_cache(
    operation: ModelOperation,
    *,
    input_type: str,
    model_input: Tensor,
    sampled_private_state: Tensor,
    owner: Tensor,
    pack_cap: int,
    precision: str,
    byte_budget: int,
    resident_bytes: int,
    predicted_saved_seconds: float,
    predicted_build_seconds: float,
) -> ReferenceOutputCache | None:
    """Build one cache only after a bounded first chunk proves admission."""

    rows = int(owner.numel())
    if (
        rows <= 0
        or model_input.shape[0] != rows
        or sampled_private_state.shape[0] != rows
        or isinstance(pack_cap, bool)
        or not isinstance(pack_cap, int)
        or pack_cap <= 0
    ):
        raise LearnerError("reference cache inputs or pack capacity are invalid")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("reference cache precision is invalid")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise LearnerError("selected CUDA device does not support BF16")
    autocast_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]

    # Reject a known unprofitable cache before launching a reference forward.
    # Payload admission is checked again once the first chunk proves its size.
    if not ReferenceOutputCache.eligible(
        byte_budget=byte_budget,
        resident_bytes=resident_bytes,
        build_bytes=0,
        payload_bytes=0,
        predicted_saved_seconds=predicted_saved_seconds,
        predicted_build_seconds=predicted_build_seconds,
    ):
        return None

    def evaluate(begin: int, end: int) -> Mapping[str, Tensor]:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=precision != "fp32",
            ),
        ):
            result = operation(
                input_type,
                model_input[begin:end],
                sampled_private_state[begin:end],
                owner[begin:end],
            )
        return result.policy_raw

    first_end = min(rows, pack_cap)
    first = evaluate(0, first_end)
    if not first or any(value.shape[0] != first_end for value in first.values()):
        raise LearnerError("reference operation returned invalid policy rows")
    payload_bytes = sum(
        rows * value[0].numel() * value.element_size() for value in first.values()
    )
    if not ReferenceOutputCache.eligible(
        byte_budget=byte_budget,
        resident_bytes=resident_bytes,
        build_bytes=payload_bytes,
        payload_bytes=payload_bytes,
        predicted_saved_seconds=predicted_saved_seconds,
        predicted_build_seconds=predicted_build_seconds,
    ):
        return None
    pieces: dict[str, list[Tensor]] = {name: [value] for name, value in first.items()}
    for begin in range(first_end, rows, pack_cap):
        end = min(rows, begin + pack_cap)
        current = evaluate(begin, end)
        if tuple(current) != tuple(pieces) or any(
            value.shape[0] != end - begin for value in current.values()
        ):
            raise LearnerError("reference operation policy schema changed")
        for name, value in current.items():
            pieces[name].append(value)
    fields = MappingProxyType(
        {name: torch.cat(values) for name, values in pieces.items()}
    )
    del pieces, first
    cache = ReferenceOutputCache.admit(
        fields,
        byte_budget=byte_budget,
        resident_bytes=resident_bytes,
        build_bytes=payload_bytes,
        predicted_saved_seconds=predicted_saved_seconds,
        predicted_build_seconds=predicted_build_seconds,
    )
    if cache is None:
        raise LearnerError("pre-admitted reference cache changed size")
    return cache


class PolicyOperation(Protocol):
    def __call__(
        self,
        input_type: str,
        model_input: Tensor,
        raw: Mapping[str, Tensor],
        action: Tensor,
        sampled_private_state: Tensor,
        owner: Tensor,
        reference_raw: Mapping[str, Tensor] | None,
    ) -> PolicyEvaluation: ...


@dataclass(frozen=True)
class SequentialLearnerBatch:
    """Aligned rows exposed through the canonical unit-span action schema."""

    input_type: str
    indices: Tensor
    fields: Mapping[str, Tensor]
    logical_step: int = 0
    logical_rows: int | None = None
    statistical_weight: float | Tensor | None = None

    @property
    def rows(self) -> int:
        return int(self.indices.numel())

    @property
    def objective_rows(self) -> int:
        return self.rows if self.logical_rows is None else self.logical_rows

    @property
    def objective_weight(self) -> float | Tensor:
        return (
            self.rows / self.objective_rows
            if self.statistical_weight is None
            else self.statistical_weight
        )

    @property
    def sigma_index(self) -> Tensor:
        return self.indices

    @property
    def action_ptr(self) -> Tensor:
        return torch.arange(
            self.rows + 1,
            dtype=torch.int64,
            device=self.indices.device,
        )

    @property
    def action_index(self) -> Tensor:
        return self.indices

    @property
    def sigma_fields(self) -> Mapping[str, Tensor]:
        return self.fields

    @property
    def action_fields(self) -> Mapping[str, Tensor]:
        return self.fields

    @property
    def action_rows(self) -> int:
        return self.rows


class SealedSequentialInputs:
    """Direct self-play payload or the one allowed best-response compaction."""

    def __init__(
        self,
        slab: MonotoneSlab,
        *,
        input_type: str,
        lane: int,
    ) -> None:
        expected = (
            ("model_input", "sampled_private_state", "owner")
            if "sampled_private_state" in slab.columns
            else ("model_input", "owner")
        )
        if tuple(slab.columns) != expected:
            raise LearnerError("sealed sequential input has the wrong field contract")
        if slab.schema.name not in (
            f"FamilyRetained.{input_type}",
            f"LearnerRetained.{input_type}",
        ):
            raise LearnerError("sealed input is not a generated retained owner")
        self.slab = slab
        self.input_type = input_type
        self.lane = lane
        self.rows = slab.tail(lane)
        self.device = slab.device
        self.fields = expected

    @classmethod
    def seal(
        cls,
        family_hot: MonotoneSlab,
        retained: MonotoneSlab,
        destination: MonotoneSlab | None = None,
        *,
        input_type: str,
        lane: int = 0,
    ) -> SealedSequentialInputs:
        if family_hot.schema.name != f"FamilyWork.{input_type}":
            raise LearnerError("input sealing is bound to the wrong family owner")
        if retained.schema.name != f"FamilyRetained.{input_type}":
            raise LearnerError("input sealing is bound to the wrong retained owner")
        if not 0 <= lane < family_hot.n_replica:
            raise LearnerError("input sealing lane is outside replicas")
        fields = (
            ("model_input", "sampled_private_state", "owner")
            if "sampled_private_state" in retained.columns
            else ("model_input", "owner")
        )
        if (
            tuple(retained.columns) != fields
            or retained.n_replica != family_hot.n_replica
            or retained.device != family_hot.device
            or retained.failure_state is not family_hot.failure_state
        ):
            raise LearnerError("generated retained owner is incompatible")
        rows = family_hot.tail(lane)
        if retained.tail(lane) != rows:
            raise LearnerError("family hot and retained rows are not aligned")
        if "role_selector" not in family_hot.columns:
            if destination is not None:
                raise LearnerError("self-play retained inputs must not be copied")
            return cls(retained, input_type=input_type, lane=lane)
        if destination is None:
            raise LearnerError("best response requires trained-row compaction")
        if (
            destination.schema.name != f"LearnerRetained.{input_type}"
            or tuple(destination.columns) != fields
            or destination.n_replica != retained.n_replica
            or destination.device != retained.device
            or destination.failure_state is not retained.failure_state
        ):
            raise LearnerError("best-response retained destination is incompatible")
        selected = torch.nonzero(
            family_hot.columns["role_selector"][lane, :rows].to(torch.int64)
            == retained.columns["owner"][lane, :rows].to(torch.int64),
            as_tuple=False,
        ).flatten()
        rows = int(selected.numel())
        if rows <= 0:
            return cls(destination, input_type=input_type, lane=lane)
        values = {name: retained.columns[name][lane, selected][None] for name in fields}
        counts = torch.zeros(
            retained.n_replica, dtype=torch.int64, device=retained.device
        )
        counts[lane] = rows
        address = torch.full_like(counts, torch.iinfo(torch.int64).max)
        address[lane] = 0
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    destination,
                    counts,
                    values,
                    input_type,
                    address,
                ),
            ),
            phase="sequential_retained_compaction",
            active=torch.arange(retained.n_replica, device=retained.device).eq(lane),
        )
        transaction.fill_all()
        success = transaction.publish()
        torch._assert_async(success[lane], "sequential retained compaction failed")
        return cls(destination, input_type=input_type, lane=lane)

    @property
    def columns(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                name: value[self.lane, : self.rows]
                for name, value in self.slab.columns.items()
            }
        )

    def release(self) -> None:
        # GeneratedOwners owns the slab lifetime.
        return None


class AlignedSequentialStore:
    """Validated five-owner view for one sequential input partition."""

    def __init__(
        self,
        owners: GeneratedOwners,
        input_type: str,
        *,
        sealed_inputs: SealedSequentialInputs,
        lane: int = 0,
    ) -> None:
        mode = owners.plan.mode
        if mode.rollout_mode is not RolloutMode.SEQUENTIAL:
            raise LearnerError("aligned stores require a sequential mode")
        if input_type not in tuple(item.name for item in owners.plan.partitions.inputs):
            raise LearnerError(f"unknown learner input partition {input_type!r}")
        if not 0 <= lane < owners.plan.manifest.n_replica:
            raise LearnerError("learner lane is outside the replica axis")
        if (
            sealed_inputs.input_type != input_type
            or sealed_inputs.lane != lane
            or sealed_inputs.device != owners.plan.device
        ):
            raise LearnerError("sealed learner input identity differs from owners")
        owner_fields = {
            "SigmaHead": ("old_value",),
            "SigmaTarget": ("value_target",),
            "ActionHead": ("action", "old_log_policy"),
            "ActionTarget": (
                ("advantage_target",)
                if mode.private_estimator is PrivateEstimator.SAMPLED
                else ("posterior", "advantage_target")
            ),
        }
        slabs = {name: owners.slab(name, input_type) for name in owner_fields}
        counts = {name: slab.tail(lane) for name, slab in slabs.items()}
        counts["SealedInput"] = sealed_inputs.rows
        if len(set(counts.values())) != 1:
            raise LearnerError(
                "sequential learner owners are not exactly row aligned: "
                + ", ".join(f"{name}={count}" for name, count in counts.items())
            )
        for owner, fields in owner_fields.items():
            missing = set(fields).difference(slabs[owner].columns)
            if missing:
                raise LearnerError(
                    f"{owner}.{input_type} lacks fields {sorted(missing)}"
                )
        self.owners = owners
        self.mode = mode
        self.input_type = input_type
        self.lane = lane
        self.sealed_inputs = sealed_inputs
        self.slabs = MappingProxyType(slabs)
        self.owner_fields = MappingProxyType(owner_fields)
        self.rows = next(iter(counts.values()))
        self.device = next(iter(slabs.values())).device

    def gather(self, indices: Tensor) -> SequentialLearnerBatch:
        if (
            indices.device != self.device
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.ndim != 1
        ):
            raise LearnerError("learner indices must be a CUDA integer vector")
        if indices.numel() == 0:
            raise LearnerError("physical learner batches cannot be empty")
        torch._assert_async(
            ((indices >= 0) & (indices < self.rows)).all(),
            "learner index is outside the sealed extent",
        )
        selected = indices.to(torch.int64)
        fields: dict[str, Tensor] = {}
        for name, value in self.sealed_inputs.columns.items():
            fields[name] = value[selected]
        for owner, names in self.owner_fields.items():
            columns = self.slabs[owner].columns
            for name in names:
                if name in fields:
                    raise LearnerError(f"duplicate learner field {name!r}")
                fields[name] = columns[name][self.lane, selected]
        return SequentialLearnerBatch(
            self.input_type, indices, MappingProxyType(fields)
        )


class SequentialBatchSampler:
    """Pooled fixed-step sampling or exact per-partition epoch traversal."""

    def __init__(
        self,
        stores: Mapping[str, AlignedSequentialStore],
        schedule: LearnerSchedule,
        *,
        generator: torch.Generator,
        statistical_counts: Mapping[str, int] | None = None,
        statistical_weights: Mapping[str, float] | None = None,
        reproducibility: str = "strict",
    ) -> None:
        if not stores:
            raise ValueError("at least one aligned learner store is required")
        self.stores = MappingProxyType(dict(stores))
        self.schedule = schedule
        self.generator = generator
        devices = {store.device for store in stores.values()}
        lanes = {store.lane for store in stores.values()}
        if len(devices) != 1 or len(lanes) != 1:
            raise LearnerError("one sampler cannot span devices or replica lanes")
        if generator.device != next(iter(devices)):
            raise LearnerError("learner generator must live on the learner CUDA device")
        if statistical_counts is not None:
            if set(statistical_counts) != set(stores):
                raise LearnerError(
                    "statistical allocation must cover every learner partition"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in statistical_counts.values()
            ):
                raise LearnerError(
                    "statistical allocation counts must be nonnegative integers"
                )
            if not any(statistical_counts.values()):
                raise LearnerError("statistical allocation cannot be empty")
        if statistical_weights is not None:
            if statistical_counts is None or set(statistical_weights) != set(stores):
                raise LearnerError(
                    "statistical weights require the same complete partition set"
                )
            if any(
                not math.isfinite(value) or value < 0
                for value in statistical_weights.values()
            ) or not math.isclose(
                sum(statistical_weights.values()), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-8
            ):
                raise LearnerError(
                    "statistical weights must be nonnegative and sum to one"
                )
        self.statistical_counts = (
            None
            if statistical_counts is None
            else MappingProxyType(dict(statistical_counts))
        )
        self.statistical_weights = (
            None
            if statistical_weights is None
            else MappingProxyType(dict(statistical_weights))
        )
        if reproducibility not in ("strict", "seeded", "throughput"):
            raise LearnerError("unknown learner reproducibility profile")
        self.reproducibility = reproducibility

    @classmethod
    def from_costs(
        cls,
        stores: Mapping[str, AlignedSequentialStore],
        schedule: LearnerSchedule,
        *,
        generator: torch.Generator,
        mass: Tensor,
        variance: Tensor,
        seconds_per_draw: Tensor,
        total_count: int,
        reproducibility: str = "strict",
    ) -> SequentialBatchSampler:
        names = tuple(stores)
        if mass.numel() != len(names):
            raise LearnerError("cost-aware allocation must match learner partitions")
        counts = allocate_statistical_counts(
            mass, variance, seconds_per_draw, total_count
        )
        total_mass = float(mass.sum())
        if not math.isfinite(total_mass) or total_mass <= 0:
            raise LearnerError("cost-aware allocation requires positive total mass")
        return cls(
            stores,
            schedule,
            generator=generator,
            statistical_counts={
                name: int(counts[index]) for index, name in enumerate(names)
            },
            statistical_weights={
                name: float(mass[index]) / total_mass
                for index, name in enumerate(names)
            },
            reproducibility=reproducibility,
        )

    @property
    def statistical_presentations(self) -> int:
        if self.statistical_counts is not None:
            count = sum(self.statistical_counts.values())
            return (
                count
                if self.schedule.gradient_steps is None
                else count * self.schedule.gradient_steps
            )
        rows = sum(store.rows for store in self.stores.values())
        if self.schedule.gradient_steps is None:
            assert self.schedule.epochs is not None
            return rows * self.schedule.epochs
        return self.schedule.gradient_steps * self.schedule.batch_size

    @property
    def optimizer_steps(self) -> int:
        """Number of logical optimizer transitions without consuming RNG."""

        if self.schedule.gradient_steps is not None:
            return self.schedule.gradient_steps
        assert self.schedule.epochs is not None
        if self.statistical_counts is None:
            return self.schedule.epochs * sum(
                (store.rows + self.schedule.batch_size - 1) // self.schedule.batch_size
                for store in self.stores.values()
            )
        total = 0
        for name, requested in self.statistical_counts.items():
            rows = self.stores[name].rows
            full, remainder = divmod(requested, rows)
            total += full * (
                (rows + self.schedule.batch_size - 1) // self.schedule.batch_size
            )
            if remainder:
                total += (
                    remainder + self.schedule.batch_size - 1
                ) // self.schedule.batch_size
        return total

    def batches(self) -> Iterator[SequentialLearnerBatch]:
        if self.statistical_counts is not None:
            logical_steps = self.schedule.gradient_steps or 1
            logical_rows = sum(self.statistical_counts.values())
            if (
                self.schedule.gradient_steps is not None
                and logical_rows != self.schedule.batch_size
            ):
                raise LearnerError(
                    "fixed-step statistical allocation must sum to batch_size"
                )
            standalone_step = 0
            for logical_step in range(logical_steps):
                for name, requested in self.statistical_counts.items():
                    if requested == 0:
                        continue
                    store = self.stores[name]
                    if self.schedule.gradient_steps is not None:
                        pieces: list[Tensor] = []
                        remaining = requested
                        while remaining:
                            permutation = torch.randperm(
                                store.rows,
                                dtype=torch.int64,
                                device=store.device,
                                generator=self.generator,
                            )
                            take = min(store.rows, remaining)
                            pieces.append(permutation[:take])
                            remaining -= take
                        gathered = store.gather(torch.cat(pieces))
                        yield SequentialLearnerBatch(
                            gathered.input_type,
                            gathered.indices,
                            gathered.fields,
                            logical_step=logical_step,
                            logical_rows=logical_rows,
                            statistical_weight=(
                                None
                                if self.statistical_weights is None
                                else self.statistical_weights[name]
                            ),
                        )
                        continue
                    emitted = 0
                    while emitted < requested:
                        permutation = torch.randperm(
                            store.rows,
                            dtype=torch.int64,
                            device=store.device,
                            generator=self.generator,
                        )
                        take = min(store.rows, requested - emitted)
                        for begin in range(0, take, self.schedule.batch_size):
                            end = min(begin + self.schedule.batch_size, take)
                            gathered = store.gather(permutation[begin:end])
                            yield SequentialLearnerBatch(
                                gathered.input_type,
                                gathered.indices,
                                gathered.fields,
                                logical_step=(
                                    logical_step
                                    if self.schedule.gradient_steps is not None
                                    else standalone_step
                                ),
                                logical_rows=(
                                    logical_rows
                                    if self.schedule.gradient_steps is not None
                                    else gathered.rows
                                ),
                            )
                            if self.schedule.gradient_steps is None:
                                standalone_step += 1
                        emitted += take
            return
        emitted = 0
        maximum = self.schedule.gradient_steps
        names = (
            tuple(
                sorted(
                    self.stores,
                    key=lambda name: (-self.stores[name].rows, name),
                )
            )
            if self.reproducibility == "throughput"
            else tuple(self.stores)
        )
        if self.schedule.gradient_steps is not None:
            offsets: list[tuple[str, int, int]] = []
            begin = 0
            for name in names:
                end = begin + self.stores[name].rows
                offsets.append((name, begin, end))
                begin = end
            if begin <= 0:
                raise LearnerError("fixed-step sampling requires retained rows")
            for step in range(self.schedule.gradient_steps):
                pooled = torch.randint(
                    begin,
                    (self.schedule.batch_size,),
                    dtype=torch.int64,
                    device=next(iter(self.stores.values())).device,
                    generator=self.generator,
                )
                for name, offset, end in offsets:
                    selected = pooled[(pooled >= offset) & (pooled < end)] - offset
                    if selected.numel() == 0:
                        continue
                    gathered = self.stores[name].gather(selected)
                    yield SequentialLearnerBatch(
                        gathered.input_type,
                        gathered.indices,
                        gathered.fields,
                        logical_step=step,
                        logical_rows=self.schedule.batch_size,
                    )
            return
        epoch = 0
        assert self.schedule.epochs is not None
        while maximum is None or emitted < maximum:
            epoch += 1
            if epoch > self.schedule.epochs and maximum is None:
                break
            if epoch > self.schedule.epochs and maximum is not None:
                epoch = 1
            made_progress = False
            for name in names:
                store = self.stores[name]
                if store.rows == 0:
                    continue
                permutation = torch.randperm(
                    store.rows,
                    dtype=torch.int64,
                    device=store.device,
                    generator=self.generator,
                )
                for begin in range(0, store.rows, self.schedule.batch_size):
                    if maximum is not None and emitted >= maximum:
                        return
                    end = min(begin + self.schedule.batch_size, store.rows)
                    made_progress = True
                    gathered = store.gather(permutation[begin:end])
                    yield SequentialLearnerBatch(
                        gathered.input_type,
                        gathered.indices,
                        gathered.fields,
                        logical_step=emitted,
                        logical_rows=gathered.rows,
                    )
                    emitted += 1
            if not made_progress:
                return
            if maximum is None and epoch >= self.schedule.epochs:
                return


@dataclass(frozen=True)
class BranchedLearnerBatch:
    """One ragged branched batch with no inactive action representation."""

    input_type: str
    sigma_index: Tensor
    action_ptr: Tensor
    action_index: Tensor
    sigma_fields: Mapping[str, Tensor]
    action_fields: Mapping[str, Tensor]
    logical_step: int = 0
    logical_rows: int | None = None
    statistical_weight: float | Tensor | None = None

    def __post_init__(self) -> None:
        device = self.sigma_index.device
        if (
            device.type != "cuda"
            or self.sigma_index.dtype not in (torch.int32, torch.int64)
            or self.sigma_index.ndim != 1
            or self.sigma_index.numel() == 0
            or self.action_ptr.device != device
            or self.action_ptr.dtype not in (torch.int32, torch.int64)
            or self.action_ptr.shape != (self.sigma_index.numel() + 1,)
            or self.action_index.device != device
            or self.action_index.dtype not in (torch.int32, torch.int64)
            or self.action_index.ndim != 1
        ):
            raise LearnerError("ragged learner indices have an invalid contract")
        torch._assert_async(
            (
                self.action_ptr[0].eq(0)
                & self.action_ptr[-1].eq(self.action_index.numel())
                & (self.action_ptr[1:] > self.action_ptr[:-1]).all()
            ),
            "ragged learner action spans must be positive and exhaustive",
        )

    @property
    def rows(self) -> int:
        return int(self.sigma_index.numel())

    @property
    def action_rows(self) -> int:
        return int(self.action_index.numel())

    @property
    def objective_rows(self) -> int:
        return self.rows if self.logical_rows is None else self.logical_rows

    @property
    def objective_weight(self) -> float | Tensor:
        return (
            self.rows / self.objective_rows
            if self.statistical_weight is None
            else self.statistical_weight
        )


class SealedBranchedInputs:
    """Direct/compacted retained payload plus metadata-only ragged spans."""

    def __init__(
        self,
        retained: MonotoneSlab,
        index: MonotoneSlab,
        *,
        input_type: str,
        lane: int,
    ) -> None:
        if index.schema.name != f"LearnerIndex.{input_type}":
            raise LearnerError("branched input is not the generated learner owner")
        required = {
            "family_row",
            "target_degree",
            "action_begin",
            "action_count",
            "log_sampling_weight",
        }
        if set(index.columns) != required:
            raise LearnerError("branched learner index is not metadata-only")
        retained_fields = (
            ("model_input", "sampled_private_state", "owner")
            if "sampled_private_state" in retained.columns
            else ("model_input", "owner")
        )
        if (
            retained.schema.name
            not in (
                f"FamilyRetained.{input_type}",
                f"LearnerRetained.{input_type}",
            )
            or tuple(retained.columns) != retained_fields
        ):
            raise LearnerError("branched retained payload is incompatible")
        self.retained = retained
        self.index = index
        self.input_type = input_type
        self.lane = lane
        self.rows = index.tail(lane)
        self.device = index.device

    @classmethod
    def seal(
        cls,
        family_retained: MonotoneSlab,
        index_destination: MonotoneSlab,
        payload_destination: MonotoneSlab | None = None,
        *,
        input_type: str,
        family_row: Tensor,
        action_begin: Tensor,
        action_count: Tensor,
        target_degree: Tensor,
        log_sampling_weight: Tensor,
        lane: int = 0,
    ) -> SealedBranchedInputs:
        if family_retained.schema.name != f"FamilyRetained.{input_type}":
            raise LearnerError("branched sealing is bound to the wrong retained owner")
        if index_destination.schema.name != f"LearnerIndex.{input_type}":
            raise LearnerError("branched input destination has the wrong identity")
        if set(index_destination.columns) != {
            "family_row",
            "target_degree",
            "action_begin",
            "action_count",
            "log_sampling_weight",
        }:
            raise LearnerError("branched learner index must contain metadata only")
        rows = int(family_row.numel())
        device = family_retained.device
        for name, value in (
            ("family_row", family_row),
            ("action_begin", action_begin),
            ("action_count", action_count),
            ("target_degree", target_degree),
            ("log_sampling_weight", log_sampling_weight),
        ):
            if value.device != device or value.shape != (rows,):
                raise LearnerError(f"{name} must be device-local [trained rows]")
        torch._assert_async(
            (
                family_row.ge(0)
                & family_row.lt(family_retained.tail(lane))
                & action_begin.ge(0)
                & action_count.ge(1)
                & target_degree.eq(action_count)
                & torch.isfinite(log_sampling_weight)
            ).all(),
            "branched seal metadata is invalid",
        )
        retained_fields = (
            ("model_input", "sampled_private_state", "owner")
            if "sampled_private_state" in family_retained.columns
            else ("model_input", "owner")
        )
        if payload_destination is None:
            retained = family_retained
            retained_row = family_row.to(torch.int32)
        else:
            if (
                payload_destination.schema.name != f"LearnerRetained.{input_type}"
                or tuple(payload_destination.columns) != retained_fields
                or payload_destination.n_replica != family_retained.n_replica
                or payload_destination.device != device
                or payload_destination.failure_state
                is not family_retained.failure_state
            ):
                raise LearnerError("best-response retained destination is incompatible")
            retained = payload_destination
            retained_row = torch.arange(rows, dtype=torch.int32, device=device)
        lane_values = {
            "family_row": retained_row,
            "target_degree": target_degree.to(torch.int32),
            "action_begin": action_begin.to(torch.int32),
            "action_count": action_count.to(torch.int32),
            "log_sampling_weight": log_sampling_weight.float(),
        }
        index_values = {
            field.name: torch.full(
                (index_destination.n_replica, max(rows, 1), *field.shape),
                field.fill,
                dtype=field.dtype,
                device=device,
            )
            for field in index_destination.schema.fields
        }
        for name, value in lane_values.items():
            index_values[name][lane, :rows].copy_(value)
        counts = torch.zeros(
            family_retained.n_replica, dtype=torch.int64, device=device
        )
        counts[lane] = rows
        logical = torch.full_like(counts, torch.iinfo(torch.int64).max)
        logical[lane] = 0
        requests = [
            DeviceDestinationRequest(
                index_destination,
                counts,
                index_values,
                input_type,
                logical,
            )
        ]
        if payload_destination is not None:
            selected = family_row.to(torch.int64)
            payload_values = {
                field.name: torch.full(
                    (
                        payload_destination.n_replica,
                        max(rows, 1),
                        *field.shape,
                    ),
                    field.fill,
                    dtype=field.dtype,
                    device=device,
                )
                for field in payload_destination.schema.fields
            }
            for name in retained_fields:
                payload_values[name][lane, :rows].copy_(
                    family_retained.columns[name][lane, selected]
                )
            requests.append(
                DeviceDestinationRequest(
                    payload_destination,
                    counts,
                    payload_values,
                    input_type,
                    logical,
                )
            )
        transaction = PublicationTransaction.begin_device(
            tuple(requests),
            phase="branched_retained_seal",
            active=torch.arange(family_retained.n_replica, device=device).eq(lane),
        )
        transaction.fill_all()
        success = transaction.publish()
        torch._assert_async(success[lane], "branched input seal failed")
        return cls(
            retained,
            index_destination,
            input_type=input_type,
            lane=lane,
        )

    @property
    def columns(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                name: value[self.lane, : self.rows]
                for name, value in self.index.columns.items()
            }
        )

    @property
    def aligned_payload_columns(self) -> Mapping[str, Tensor]:
        row = self.columns["family_row"].to(torch.int64)
        return MappingProxyType(
            {
                name: value[self.lane].index_select(0, row)
                for name, value in self.retained.columns.items()
            }
        )

    def release(self) -> None:
        return None


class RaggedBranchedStore:
    """Validated sigma owners plus exact output-partition action spans."""

    def __init__(
        self,
        owners: GeneratedOwners,
        input_type: str,
        *,
        sealed_inputs: SealedBranchedInputs,
        lane: int = 0,
    ) -> None:
        mode = owners.plan.mode
        if not mode.rollout_mode.is_branched:
            raise LearnerError("ragged stores require branched mode")
        output_type = owners.plan.partitions.output_for(input_type).name
        sigma = {
            "SigmaHead": owners.slab("SigmaHead", input_type),
            "SigmaTarget": owners.slab("SigmaTarget", input_type),
        }
        action = {
            "ActionHead": owners.slab("ActionHead", output_type),
            "ActionTarget": owners.slab("ActionTarget", output_type),
        }
        sigma_counts = {name: slab.tail(lane) for name, slab in sigma.items()}
        sigma_counts["LearnerIndex"] = sealed_inputs.rows
        if len(set(sigma_counts.values())) != 1:
            raise LearnerError(
                "branched sigma owners are not row aligned: "
                + ", ".join(f"{name}={count}" for name, count in sigma_counts.items())
            )
        columns = sealed_inputs.columns
        if sealed_inputs.rows:
            torch._assert_async(
                (
                    columns["action_begin"].ge(0)
                    & columns["action_count"].ge(1)
                    & (
                        columns["action_begin"].to(torch.int64)
                        + columns["action_count"].to(torch.int64)
                    ).le(action["ActionHead"].tail(lane))
                ).all(),
                "branched action spans exceed the sealed action extent",
            )
        if action["ActionHead"].tail(lane) != action["ActionTarget"].tail(lane):
            raise LearnerError("branched action heads and targets are not aligned")
        self.owners = owners
        self.mode = mode
        self.input_type = input_type
        self.output_type = output_type
        self.lane = lane
        self.sealed_inputs = sealed_inputs
        self.sigma_slabs = MappingProxyType(sigma)
        self.action_slabs = MappingProxyType(action)
        self.rows = sealed_inputs.rows
        self.device = owners.plan.device

    def gather(
        self,
        sigma_index: Tensor,
        *,
        expected_action_rows: int,
    ) -> BranchedLearnerBatch:
        if (
            sigma_index.device != self.device
            or sigma_index.dtype not in (torch.int32, torch.int64)
            or sigma_index.ndim != 1
            or sigma_index.numel() == 0
        ):
            raise LearnerError("branched sigma indices must be CUDA integers")
        selected = sigma_index.to(torch.int64)
        if (
            isinstance(expected_action_rows, bool)
            or not isinstance(expected_action_rows, int)
            or expected_action_rows <= 0
        ):
            raise LearnerError("expected ragged action extent must be positive")
        torch._assert_async(
            (selected.ge(0) & selected.lt(self.rows)).all(),
            "branched sigma index is outside its sealed extent",
        )
        index = self.sealed_inputs.columns
        begin = index["action_begin"][selected].to(torch.int64)
        count = index["action_count"][selected].to(torch.int64)
        ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=self.device),
                count.cumsum(0),
            )
        )
        action_index = torch.repeat_interleave(
            begin,
            count,
            output_size=expected_action_rows,
        )
        action_index += torch.arange(
            expected_action_rows,
            dtype=torch.int64,
            device=self.device,
        ) - torch.repeat_interleave(
            ptr[:-1],
            count,
            output_size=expected_action_rows,
        )
        retained_row = index["family_row"][selected].to(torch.int64)
        sigma_fields: dict[str, Tensor] = {
            name: value[self.lane, retained_row]
            for name, value in self.sealed_inputs.retained.columns.items()
        }
        sigma_fields.update(
            {
                name: value[selected]
                for name, value in index.items()
                if name not in ("family_row", "action_begin", "action_count")
            }
        )
        for slab in self.sigma_slabs.values():
            for name, value in slab.columns.items():
                if name in sigma_fields:
                    raise LearnerError(f"duplicate branched sigma field {name!r}")
                sigma_fields[name] = value[self.lane, selected]
        action_fields: dict[str, Tensor] = {}
        for slab in self.action_slabs.values():
            for name, value in slab.columns.items():
                if name in action_fields:
                    raise LearnerError(f"duplicate branched action field {name!r}")
                action_fields[name] = value[self.lane, action_index]
        return BranchedLearnerBatch(
            self.input_type,
            sigma_index,
            ptr,
            action_index,
            MappingProxyType(sigma_fields),
            MappingProxyType(action_fields),
        )


def _alias_table(log_weights: Tensor) -> tuple[Tensor, Tensor]:
    """Build one deterministic Walker table entirely on CUDA in float64."""

    if (
        log_weights.device.type != "cuda"
        or log_weights.ndim != 1
        or log_weights.numel() == 0
    ):
        raise LearnerError("alias log weights must be a nonempty CUDA vector")
    work = log_weights.detach().double()
    torch._assert_async(torch.isfinite(work).all(), "alias log weights must be finite")
    size = work.numel()
    scaled = (torch.exp(work - torch.logsumexp(work, 0)) * size).contiguous()
    probability = torch.empty_like(scaled)
    alias = torch.empty(size, dtype=torch.int32, device=work.device)
    small = torch.empty(size, dtype=torch.int64, device=work.device)
    large = torch.empty_like(small)
    alias_table_cuda().build_alias_table(
        scaled,
        probability,
        alias,
        small,
        large,
    )
    return probability, alias


class BranchedBatchSampler:
    """Alias-sample decision mass and gather exact ragged action spans."""

    def __init__(
        self,
        stores: Mapping[str, RaggedBranchedStore],
        schedule: LearnerSchedule,
        *,
        generator: torch.Generator,
        sigma_pack_cap: int,
        action_pack_cap: int,
        statistical_counts: Mapping[str, int] | None = None,
        family_sigma_caps: Mapping[str, int] | None = None,
    ) -> None:
        if not stores:
            raise ValueError("at least one branched learner store is required")
        self.stores = MappingProxyType(dict(stores))
        self.schedule = schedule
        self.generator = generator
        devices = {store.device for store in stores.values()}
        if len(devices) != 1 or generator.device != next(iter(devices)):
            raise LearnerError("branched sampler must remain on one CUDA device")
        log_z: dict[str, Tensor] = {}
        alias_tables: dict[str, tuple[Tensor, Tensor]] = {}
        for name, store in stores.items():
            log_weight = store.sealed_inputs.columns["log_sampling_weight"].double()
            log_z[name] = torch.logsumexp(log_weight, 0)
            alias_tables[name] = _alias_table(log_weight)
            del log_weight
        self.log_Z = MappingProxyType(log_z)
        self.alias = MappingProxyType(alias_tables)
        log_mass = torch.stack(tuple(self.log_Z.values()))
        mass = torch.softmax(log_mass, 0)
        self.stratum_probability = MappingProxyType(
            {name: mass[index] / mass.sum() for index, name in enumerate(stores)}
        )
        if statistical_counts is None:
            stratum_count = mass.numel()
            if schedule.batch_size < stratum_count:
                raise LearnerError("branched batch size cannot cover every stratum")
            remaining = schedule.batch_size - stratum_count
            raw = mass * remaining
            extra = raw.floor().to(torch.int64)
            leftover = remaining - extra.sum()
            order = torch.argsort(
                raw - extra,
                descending=True,
                stable=True,
            )
            rank = torch.empty_like(order)
            rank.scatter_(
                0,
                order,
                torch.arange(stratum_count, device=mass.device),
            )
            allocated = 1 + extra + rank.lt(leftover)
            host_allocated = torch.empty(
                stratum_count,
                dtype=torch.int64,
                pin_memory=True,
            )
            host_allocated.copy_(allocated, non_blocking=True)
            ready = torch.cuda.Event(blocking=False)
            ready.record(torch.cuda.current_stream(mass.device))
            while not ready.query():
                time.sleep(0)
            statistical_counts = {
                name: int(host_allocated[index]) for index, name in enumerate(stores)
            }
        if set(statistical_counts) != set(stores) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in statistical_counts.values()
        ):
            raise LearnerError(
                "branched statistical counts must positively cover every stratum"
            )
        self.statistical_counts = MappingProxyType(dict(statistical_counts))
        if (
            isinstance(sigma_pack_cap, bool)
            or not isinstance(sigma_pack_cap, int)
            or isinstance(action_pack_cap, bool)
            or not isinstance(action_pack_cap, int)
            or sigma_pack_cap <= 0
            or action_pack_cap <= 0
        ):
            raise LearnerError("branched physical pack caps must be positive")
        self.family_sigma_caps = {} if family_sigma_caps is None else dict(family_sigma_caps)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in self.family_sigma_caps.values()):
            raise ValueError("family sigma caps must be positive integers")
        self.sigma_pack_cap = sigma_pack_cap
        self.action_pack_cap = action_pack_cap

    def _draw(self, name: str, count: int) -> Tensor:
        threshold, alias = self.alias[name]
        column = torch.randint(
            threshold.numel(),
            (count,),
            dtype=torch.int64,
            device=threshold.device,
            generator=self.generator,
        )
        uniform = torch.rand(
            count,
            dtype=torch.float64,
            device=threshold.device,
            generator=self.generator,
        )
        return torch.where(
            uniform < threshold[column],
            column,
            alias[column].to(torch.int64),
        )

    @property
    def statistical_presentations(self) -> int:
        one_pass = sum(self.statistical_counts.values())
        return one_pass * (
            self.schedule.gradient_steps
            if self.schedule.gradient_steps is not None
            else self.schedule.epochs
        )

    @property
    def optimizer_steps(self) -> int:
        return (
            self.schedule.gradient_steps
            if self.schedule.gradient_steps is not None
            else self.schedule.epochs
        )

    def batches(self) -> Iterator[BranchedLearnerBatch]:
        logical_rows = sum(self.statistical_counts.values())
        for logical_step in range(self.optimizer_steps):
            for name, requested in self.statistical_counts.items():
                draw = self._draw(name, requested)
                action_counts = (
                    self.stores[name]
                    .sealed_inputs.columns["action_count"]
                    .index_select(0, draw)
                    .to(torch.int64)
                    .cpu()
                    .tolist()
                )
                emitted = 0
                while emitted < requested:
                    take = 0
                    actions = 0
                    sigma_limit = min(
                        self.schedule.batch_size,
                        self.sigma_pack_cap,
                        self.family_sigma_caps.get(name, self.sigma_pack_cap),
                        requested - emitted,
                    )
                    while take < sigma_limit:
                        next_actions = int(action_counts[emitted + take])
                        if next_actions > self.action_pack_cap:
                            raise LearnerError(
                                "one ragged decision exceeds action pack capacity"
                            )
                        if take and actions + next_actions > self.action_pack_cap:
                            break
                        actions += next_actions
                        take += 1
                    gathered = self.stores[name].gather(
                        draw[emitted : emitted + take],
                        expected_action_rows=actions,
                    )
                    yield BranchedLearnerBatch(
                        gathered.input_type,
                        gathered.sigma_index,
                        gathered.action_ptr,
                        gathered.action_index,
                        gathered.sigma_fields,
                        gathered.action_fields,
                        logical_step=logical_step,
                        logical_rows=logical_rows,
                        statistical_weight=(
                            self.stratum_probability[name] * (take / requested)
                        ),
                    )
                    emitted += take

    def release(self) -> None:
        """Release alias and stratum tensors at their final optimizer consumer."""

        self.alias = MappingProxyType({})
        self.log_Z = MappingProxyType({})
        self.stratum_probability = MappingProxyType({})
        self.stores = MappingProxyType({})


@dataclass(frozen=True)
class LossResult:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    reference_kl: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    negative_advantage_cap_fraction: Tensor
    negative_advantage_log_ratio_max: Tensor
    negative_advantage_influence_max: Tensor


def _scalar_private_row(value: Tensor, private: Tensor, owner: Tensor) -> Tensor:
    """Select owner-combo values when a model exposes ``[B,H]`` rows."""

    if value.ndim == 1:
        return value.float()
    if value.ndim != 2 or private.ndim != 2 or owner.shape != value.shape[:1]:
        raise LearnerError("private-row selection requires [B,H], [B,P], and [B]")
    owner64 = owner.to(torch.int64)
    torch._assert_async(
        ((owner64 >= 0) & (owner64 < private.shape[1])).all(),
        "learner owner is outside the private profile",
    )
    combo = private.gather(1, owner64[:, None]).squeeze(1).to(torch.int64)
    torch._assert_async(
        ((combo >= 0) & (combo < value.shape[1])).all(),
        "sampled private combo is outside the model domain",
    )
    return value.gather(1, combo[:, None]).squeeze(1).float()


def _select_sampled_private(value: Tensor, private: Tensor, owner: Tensor) -> Tensor:
    """Gather one owner-combo row while preserving trailing head fields."""

    if value.ndim < 2 or private.ndim != 2 or owner.shape != value.shape[:1]:
        raise LearnerError("sampled output selection requires [B,H,...], [B,P], [B]")
    owner64 = owner.to(torch.int64)
    torch._assert_async(
        ((owner64 >= 0) & (owner64 < private.shape[1])).all(),
        "learner owner is outside the private profile",
    )
    combo = private.gather(1, owner64[:, None]).to(torch.int64)
    if value.shape[1] == 1:
        return value
    torch._assert_async(
        ((combo >= 0) & (combo < value.shape[1])).all(),
        "sampled private combo is outside the model domain",
    )
    index = combo.reshape(combo.shape[0], 1, *((1,) * (value.ndim - 2))).expand(
        combo.shape[0], 1, *value.shape[2:]
    )
    return value.gather(1, index)


class LearnerOperation:
    """One tensor loss with independent lane reductions and external backward."""

    def __init__(
        self,
        mode: ModeSchema,
        loss: LossConfig,
        model_operation: ModelOperation,
        policy_operation: PolicyOperation,
        *,
        precision: str = "fp32",
    ) -> None:
        self.mode = mode
        self.loss = loss
        self.model_operation = model_operation
        self.policy_operation = policy_operation
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("learner precision must be fp32, fp16, or bf16")
        from pbg.core.precision import resolve_precision
        self.precision_policy = resolve_precision(precision, torch.device("cuda"))
        self.precision = precision
        self.dtypes = LearnerDTypes(
            {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[precision]
        )

    def forward_loss(
        self,
        batch: SequentialLearnerBatch | BranchedLearnerBatch,
        *,
        iteration: int,
        reference_operation: ModelOperation | None = None,
        reference_cache: ReferenceOutputCache | None = None,
        lane_count: int = 1,
        loss_controls: Tensor | None = None,
        evaluation: ModelEvaluation | None = None,
        reference_raw: Mapping[str, Tensor] | None = None,
    ) -> LossResult:
        if not isinstance(batch, (SequentialLearnerBatch, BranchedLearnerBatch)):
            raise LearnerError("learner batch has no supported presentation")
        rows = batch.rows
        if lane_count <= 0 or rows % lane_count or batch.action_rows % lane_count:
            raise LearnerError("loss cohort must contain exact equal-shape lanes")
        if loss_controls is None:
            loss_controls = self.controls((self.loss,), iteration, batch.sigma_index.device)
        if loss_controls.shape != (lane_count, 10) or loss_controls.dtype != torch.float32:
            raise LearnerError("loss controls must be FP32 [K,10]")
        def action_control(column: int) -> Tensor:
            return loss_controls[:, column, None].expand(-1, batch.action_rows // lane_count).reshape(-1, 1)
        def sigma_control(column: int) -> Tensor:
            return loss_controls[:, column, None].expand(-1, rows // lane_count).reshape(-1, 1)
        def lane_mean(value: Tensor) -> Tensor:
            return value.reshape(lane_count, -1).mean(-1)
        sigma_index = batch.sigma_index
        action_ptr = batch.action_ptr
        sigma = batch.sigma_fields
        action = batch.action_fields
        counts = action_ptr[1:] - action_ptr[:-1]
        action_rows = batch.action_rows
        if "target_degree" in sigma:
            torch._assert_async(
                counts.eq(sigma["target_degree"].to(counts.dtype)).all(),
                "learner action spans disagree with protected target degrees",
            )
        else:
            torch._assert_async(
                counts.eq(1).all(),
                "aligned learner rows must have unit action spans",
            )
        action_to_sigma = torch.repeat_interleave(
            torch.arange(rows, device=sigma_index.device),
            counts,
            output_size=action_rows,
        )
        sampled = self.mode.private_estimator is PrivateEstimator.SAMPLED
        sigma_private = (
            sigma["sampled_private_state"] if sampled else sigma["owner"][:, None]
        )
        autocast_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            if evaluation is None:
                evaluation = self.model_operation(
                    batch.input_type, sigma["model_input"], sigma_private, sigma["owner"],
                )
            natural_raw = (
                {
                    name: _select_sampled_private(
                        value,
                        sigma_private,
                        sigma["owner"],
                    )
                    for name, value in evaluation.policy_raw.items()
                }
                if sampled
                else dict(evaluation.policy_raw)
            )
            natural_value = (
                _select_sampled_private(
                    evaluation.value,
                    sigma_private,
                    sigma["owner"],
                )
                if sampled
                else evaluation.value
            )
            reference = None
            if reference_raw is not None:
                reference_sigma = dict(reference_raw)
            elif reference_cache is not None:
                if reference_operation is not None:
                    raise LearnerError(
                        "reference cache and operation are mutually exclusive"
                    )
                cached = reference_cache.gather(sigma_index)
                reference_sigma = (
                    {
                        name: _select_sampled_private(
                            value,
                            sigma_private,
                            sigma["owner"],
                        )
                        for name, value in cached.items()
                    }
                    if sampled
                    else dict(cached)
                )
            elif reference_operation is not None:
                with torch.no_grad():
                    reference_evaluation = reference_operation(
                        batch.input_type, sigma["model_input"], sigma_private, sigma["owner"],
                    )
                reference_sigma = (
                    {
                        name: _select_sampled_private(
                            value,
                            sigma_private,
                            sigma["owner"],
                        )
                        for name, value in reference_evaluation.policy_raw.items()
                    }
                    if sampled
                    else dict(reference_evaluation.policy_raw)
                )
            else:
                reference_sigma = None
            repeated_raw = {
                name: value.index_select(0, action_to_sigma)
                for name, value in natural_raw.items()
            }
            if reference_sigma is not None:
                reference = {
                    name: value.index_select(0, action_to_sigma)
                    for name, value in reference_sigma.items()
                }
        repeated_owner = sigma["owner"].index_select(0, action_to_sigma)
        repeated_private = sigma_private.index_select(0, action_to_sigma)
        policy_private = (
            torch.zeros_like(repeated_private) if sampled else repeated_owner[:, None]
        )
        policy = self.policy_operation(
            batch.input_type,
            sigma["model_input"].index_select(0, action_to_sigma),
            repeated_raw,
            action["action"].to(torch.int32),
            policy_private,
            repeated_owner.to(torch.int64),
            reference,
        )
        if sampled:
            current_log = _scalar_private_row(
                policy.selected_log_policy,
                policy_private,
                repeated_owner,
            ).reshape(action_rows, 1)
            entropy = _scalar_private_row(
                policy.entropy, policy_private, repeated_owner
            ).reshape(action_rows, 1)
            reference_kl = _scalar_private_row(
                policy.reference_kl, policy_private, repeated_owner
            ).reshape(action_rows, 1)
            current_value = _scalar_private_row(
                natural_value,
                torch.zeros_like(sigma_private),
                sigma["owner"],
            ).reshape(rows, 1)
            old_log = action["old_log_policy"].float().reshape(action_rows, -1)[:, :1]
            advantage = (
                action["advantage_target"].float().reshape(action_rows, -1)[:, :1]
            )
            old_value = sigma["old_value"].float().reshape(rows, -1)[:, :1]
            target_value = sigma["value_target"].float().reshape(rows, -1)[:, :1]
            posterior = torch.ones_like(current_log)
        else:
            current_log = policy.selected_log_policy.float()
            entropy = policy.entropy.float()
            reference_kl = policy.reference_kl.float()
            current_value = natural_value.float()
            old_log = action["old_log_policy"].float()
            advantage = action["advantage_target"].float()
            posterior = action["posterior"].float()
            old_value = sigma["old_value"].float()
            target_value = sigma["value_target"].float()
        if (
            current_log.shape != entropy.shape
            or current_log.shape != reference_kl.shape
            or current_log.shape != old_log.shape
            or current_log.shape != advantage.shape
            or current_log.shape != posterior.shape
            or current_value.shape != old_value.shape
            or current_value.shape != target_value.shape
            or current_log.shape[0] != action_rows
            or current_value.shape[0] != rows
            or current_log.shape[1:] != current_value.shape[1:]
        ):
            raise LearnerError("learner fields are not sigma/action aligned")
        posterior = posterior / posterior.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        log_ratio = current_log - old_log
        ratio = log_ratio.exp()
        clip = action_control(0)
        clipped = ratio.clamp(1 - clip, 1 + clip)
        negative = advantage.lt(0)
        log_cap = action_control(1)
        negative_ratio = torch.minimum(log_ratio, log_cap).exp()
        policy_ratio = torch.where(negative, negative_ratio, ratio)
        negative_capped = negative & log_ratio.gt(log_cap)
        policy_row = (
            posterior
            * -torch.minimum(
                policy_ratio * advantage,
                clipped * advantage,
            )
        ).sum(-1)
        entropy_row = (posterior * entropy).sum(-1)
        reference_row = (posterior * reference_kl).sum(-1)
        approximate_row = (posterior * ((ratio - 1) - log_ratio)).sum(-1)
        clipped_row = (
            posterior * ratio.sub(1).abs().gt(clip).float()
        ).sum(-1)
        negative_cap_row = (
            posterior * negative_capped.float()
        ).sum(-1)
        supported_negative = negative & posterior.gt(0)
        negative_log_ratio_max = torch.where(
            supported_negative,
            log_ratio,
            torch.zeros_like(log_ratio),
        ).reshape(lane_count, -1).amax(-1)
        negative_influence_max = (
            posterior
            * advantage.neg().clamp_min(0)
            * torch.where(negative, policy_ratio, torch.zeros_like(policy_ratio))
        ).reshape(lane_count, -1).amax(-1)
        sigma_posterior = torch.zeros_like(current_value, dtype=torch.float32)
        sigma_posterior.scatter_add_(
            0,
            action_to_sigma[:, None].expand_as(posterior),
            posterior,
        )
        sigma_posterior /= counts.float()[:, None]
        sigma_posterior /= sigma_posterior.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        value_error = (current_value - target_value).square()
        value_epsilon = sigma_control(3)
        clipped_value = old_value + (current_value - old_value).clamp(-value_epsilon, value_epsilon)
        value_error = torch.where(
            sigma_control(9).bool(),
            torch.maximum(value_error, (clipped_value - target_value).square()), value_error,
        )
        value_row = (sigma_posterior * value_error).sum(-1)

        def mean_by_sigma(rows: Tensor) -> Tensor:
            result = torch.zeros(
                current_value.shape[0],
                dtype=torch.float32,
                device=rows.device,
            )
            result.scatter_add_(0, action_to_sigma, rows)
            return result / counts.float()

        sigma_policy = mean_by_sigma(policy_row)
        sigma_entropy = mean_by_sigma(entropy_row)
        sigma_reference = mean_by_sigma(reference_row)
        sigma_approximate = mean_by_sigma(approximate_row)
        sigma_clipped = mean_by_sigma(clipped_row)
        sigma_negative_capped = mean_by_sigma(negative_cap_row)
        if "trace_target" in sigma or "cost_target" in sigma:
            if self.loss.branching is None:
                raise LearnerError("branched public targets require a branching loss")
            if (
                set(evaluation.public_heads) != {"trace", "cost"}
                or "trace_target" not in sigma
                or "cost_target" not in sigma
                or "target_degree" not in sigma
            ):
                raise LearnerError("public heads and targets must share one schema")
            trace_prediction = evaluation.public_heads["trace"].float()
            cost_prediction = evaluation.public_heads["cost"].float()
            trace_target = sigma["trace_target"].float()
            cost_target = sigma["cost_target"].float()
            if not all(
                value.shape == (rows,)
                for value in (
                    trace_prediction,
                    cost_prediction,
                    trace_target,
                    cost_target,
                )
            ):
                raise LearnerError("public heads and targets are not sigma aligned")
            eps = sigma_control(8).squeeze(-1)
            trace_ratio = (trace_target + eps) / (
                trace_prediction.clamp_min(torch.finfo(torch.float32).tiny) + eps
            )
            trace_loss = torch.where(
                sigma["target_degree"].ge(2),
                trace_ratio - trace_ratio.log() - 1,
                torch.zeros_like(trace_ratio),
            ).reshape(lane_count, -1).mean(-1)
            torch._assert_async(
                cost_target.gt(0).all(),
                "cost targets must be positive",
            )
            cost_ratio = cost_target / cost_prediction.clamp_min(
                torch.finfo(torch.float32).tiny
            )
            cost_loss = lane_mean(cost_ratio - cost_ratio.log() - 1)
        else:
            trace_loss = current_value.new_zeros((lane_count,))
            cost_loss = current_value.new_zeros((lane_count,))
        policy_loss = lane_mean(sigma_policy)
        value_loss = lane_mean(value_row)
        entropy_mean = lane_mean(sigma_entropy)
        reference_kl_mean = lane_mean(sigma_reference)
        total = (
            policy_loss + loss_controls[:, 2] * value_loss
            - loss_controls[:, 4] * entropy_mean
            + loss_controls[:, 5] * reference_kl_mean
            + loss_controls[:, 6] * trace_loss + loss_controls[:, 7] * cost_loss
        )
        return LossResult(
            total,
            policy_loss,
            value_loss,
            entropy_mean,
            reference_kl_mean,
            lane_mean(sigma_approximate),
            lane_mean(sigma_clipped),
            lane_mean(sigma_negative_capped),
            negative_log_ratio_max,
            negative_influence_max,
        )

    @staticmethod
    def controls(losses: tuple[LossConfig, ...], iteration: int, device: torch.device) -> Tensor:
        """Resolve independent scientific controls into device inputs."""

        return torch.tensor([
            [loss.policy_clip,
             math.inf if loss.negative_advantage_ratio_cap is None else math.log(loss.negative_advantage_ratio_cap),
             loss.value.coefficient, loss.value.epsilon or 0.0, loss.entropy_at(iteration),
             loss.reference_kl_coefficient or 0.0,
             0.0 if loss.branching is None else loss.branching.trace_coefficient,
             0.0 if loss.branching is None else loss.branching.cost_coefficient,
             1.0e-8 if loss.branching is None else loss.branching.trace_epsilon,
             float(loss.value.kind == "clipped")]
            for loss in losses
        ], dtype=torch.float32, device=device)

    def forward_loss_backward(
        self, batch: SequentialLearnerBatch | BranchedLearnerBatch, *, iteration: int,
        reference_operation: ModelOperation | None = None,
        reference_cache: ReferenceOutputCache | None = None,
        loss_scale: float | Tensor = 1.0, objective_weight: float | Tensor = 1.0,
    ) -> LossResult:
        """Singleton diagnostic interface to the same tensor loss program."""

        result = self.forward_loss(batch, iteration=iteration,
                                   reference_operation=reference_operation,
                                   reference_cache=reference_cache)
        if self.precision != "fp16" and not bool(torch.isfinite(result.total).all()):
            raise LearnerError("learner loss is nonfinite")
        weight = torch.as_tensor(objective_weight, dtype=torch.float32, device=batch.sigma_index.device)
        if weight.numel() != 1:
            raise LearnerError("physical objective weight must be one device scalar")
        torch._assert_async((torch.isfinite(weight) & weight.gt(0) & weight.le(1)).all(),
                            "physical objective weight must lie in (0,1]")
        (result.total * loss_scale * weight).sum().backward()
        return LossResult(**{name: getattr(result, name).squeeze(0) for name in result.__dataclass_fields__})


@dataclass
class ReferenceState:
    """Checkpointed reference refresh state and effective-hand cadence."""

    reference_iteration: int = 0
    effective_hands_rollout: float = 0.0
    interval_iterations: int | None = None
    interval_effective_hands: float | None = None

    def __post_init__(self) -> None:
        if self.reference_iteration < 0 or self.effective_hands_rollout < 0:
            raise ValueError("reference counters must be nonnegative")
        if (self.interval_iterations is None) == (
            self.interval_effective_hands is None
        ):
            raise ValueError("select exactly one reference refresh cadence")
        if self.interval_iterations is not None and self.interval_iterations <= 0:
            raise ValueError("reference iteration cadence must be positive")
        if self.interval_effective_hands is not None and (
            not math.isfinite(self.interval_effective_hands)
            or self.interval_effective_hands <= 0
        ):
            raise ValueError("reference effective-hand cadence must be positive")

    def refresh_before_iteration(self, iteration: int) -> bool:
        """Apply the preserved interval cadence at the iteration boundary."""

        if iteration < 0:
            raise ValueError("reference iteration must be nonnegative")
        if self.interval_iterations is None:
            return False
        due = iteration % self.interval_iterations == 0
        if due:
            self.reference_iteration = iteration
            self.effective_hands_rollout = 0.0
        return due

    def observe(self, *, iteration: int, effective_hands: float) -> bool:
        """Observe one completed boundary for the effective-hand cadence."""

        if effective_hands < 0 or not math.isfinite(effective_hands):
            raise ValueError("effective hands must be finite and nonnegative")
        if iteration < 0:
            raise ValueError("reference iteration must be nonnegative")
        if self.interval_effective_hands is None:
            return False
        self.effective_hands_rollout += effective_hands
        due = self.effective_hands_rollout >= self.interval_effective_hands
        if due:
            self.reference_iteration = iteration
            self.effective_hands_rollout = 0.0
        return due

    def state_dict(self) -> dict[str, object]:
        return {
            "reference_iteration": self.reference_iteration,
            "effective_hands_rollout": self.effective_hands_rollout,
            "interval_iterations": self.interval_iterations,
            "interval_effective_hands": self.interval_effective_hands,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> ReferenceState:
        expected = {
            "reference_iteration",
            "effective_hands_rollout",
            "interval_iterations",
            "interval_effective_hands",
        }
        if set(state) != expected:
            raise ValueError("reference state has unknown or missing fields")
        return cls(**dict(state))


__all__ = [
    "AlignedSequentialStore",
    "BranchedBatchSampler",
    "BranchedLearnerBatch",
    "BranchingLossConfig",
    "EntropyLossConfig",
    "LearnerDTypes",
    "LearnerError",
    "LearnerOperation",
    "LearnerSchedule",
    "LossConfig",
    "LossResult",
    "ModelEvaluation",
    "PolicyEvaluation",
    "RaggedBranchedStore",
    "ReferenceOutputCache",
    "ReferenceState",
    "SealedBranchedInputs",
    "SealedSequentialInputs",
    "SequentialBatchSampler",
    "SequentialLearnerBatch",
    "ValueLossConfig",
    "allocate_statistical_counts",
    "build_reference_output_cache",
    "compact_best_response_evaluation",
]
