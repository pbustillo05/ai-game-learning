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

from pbg.queued.capacity import GeneratedOwners
from pbg.queued.contracts import (
    ModeSchema,
    PrivateEstimator,
    RolloutMode,
)
from pbg.queued.cuda_alias import alias_table_cuda
from pbg.queued.storage import (
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


@dataclass(frozen=True)
class LossConfig:
    """Closed loss surface shared by every queued mode."""

    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    reference_kl_coefficient: float = 0.0
    value_loss: str = "mse"
    entropy_schedule: str = "constant"
    entropy_decay_iterations: int | None = None
    trace_coefficient: float = 0.0
    cost_coefficient: float = 0.0
    trace_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        finite = (
            self.clip_epsilon,
            self.value_clip_epsilon,
            self.value_coefficient,
            self.entropy_coefficient,
            self.reference_kl_coefficient,
            self.trace_coefficient,
            self.cost_coefficient,
            self.trace_epsilon,
        )
        if any(not math.isfinite(value) or value < 0 for value in finite):
            raise ValueError(
                "loss coefficients and clips must be finite and nonnegative"
            )
        if self.value_loss not in ("mse", "clipped"):
            raise ValueError("value_loss must be 'mse' or 'clipped'")
        if self.trace_epsilon <= 0:
            raise ValueError("trace_epsilon must be positive")
        if self.entropy_schedule not in ("constant", "linear_decay"):
            raise ValueError("entropy_schedule must be 'constant' or 'linear_decay'")
        if self.entropy_schedule == "linear_decay" and (
            isinstance(self.entropy_decay_iterations, bool)
            or not isinstance(self.entropy_decay_iterations, int)
            or self.entropy_decay_iterations <= 0
        ):
            raise ValueError(
                "linear entropy decay requires a positive independent horizon"
            )

    def entropy_at(self, iteration: int) -> float:
        if iteration < 0:
            raise ValueError("iteration must be nonnegative")
        if self.entropy_schedule == "constant":
            return self.entropy_coefficient
        assert self.entropy_decay_iterations is not None
        fraction = max(0.0, 1.0 - float(iteration) / self.entropy_decay_iterations)
        return self.entropy_coefficient * fraction


@dataclass(frozen=True)
class LearnerSchedule:
    """Exact statistical presentation schedule for sequential modes."""

    batch_size: int
    epochs: int = 1
    gradient_steps: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer")
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
        result = source.new_empty((trained.shape[0], *source.shape[1:]))
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
            or tuple(item.public_heads) != public_names
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
                    None if frozen is None else frozen.public_heads[name],
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
    """The exact aligned rows gathered by one permutation slice."""

    input_type: str
    indices: Tensor
    fields: Mapping[str, Tensor]
    logical_step: int = 0
    logical_rows: int | None = None

    @property
    def rows(self) -> int:
        return int(self.indices.numel())

    @property
    def objective_rows(self) -> int:
        return self.rows if self.logical_rows is None else self.logical_rows


class SealedSequentialInputs:
    """Generated retained learner-input owner after rollout-hot release."""

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
        if slab.schema.name != f"LearnerIndex.{input_type}":
            raise LearnerError("sealed input is not the generated learner owner")
        self.slab = slab
        self.input_type = input_type
        self.lane = lane
        self.rows = slab.tail(lane)
        self.device = slab.device
        self.fields = expected

    @classmethod
    def seal(
        cls,
        family: MonotoneSlab,
        destination: MonotoneSlab,
        *,
        input_type: str,
        lane: int = 0,
    ) -> SealedSequentialInputs:
        if family.schema.name != f"FamilyWork.{input_type}":
            raise LearnerError("input sealing is bound to the wrong family owner")
        if not 0 <= lane < family.n_replica:
            raise LearnerError("input sealing lane is outside replicas")
        by_name = family.schema.by_name
        fields = (
            ("model_input", "sampled_private_state", "owner")
            if "sampled_private_state" in destination.columns
            else ("model_input", "owner")
        )
        missing = set(fields).difference(by_name)
        if missing:
            raise LearnerError(f"family input lacks fields {sorted(missing)}")
        if (
            destination.schema.name != f"LearnerIndex.{input_type}"
            or tuple(destination.columns) != fields
            or destination.n_replica != family.n_replica
            or destination.device != family.device
            or destination.failure_state is not family.failure_state
        ):
            raise LearnerError("generated learner-input owner is incompatible")
        rows = family.tail(lane)
        selected: Tensor | None = None
        if "role_selector" in family.columns:
            selected = torch.nonzero(
                family.columns["role_selector"][lane, :rows].to(torch.int64)
                == family.columns["owner"][lane, :rows].to(torch.int64),
                as_tuple=False,
            ).flatten()
            rows = int(selected.numel())
        if rows <= 0:
            return cls(destination, input_type=input_type, lane=lane)
        values = {
            name: (
                family.columns[name][:, :rows]
                if selected is None
                else family.columns[name][lane, selected][None]
            )
            for name in fields
        }
        counts = torch.zeros(family.n_replica, dtype=torch.int64, device=family.device)
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
            phase="sequential_input_seal",
            active=torch.arange(family.n_replica, device=family.device).eq(lane),
        )
        transaction.fill_all()
        success = transaction.publish()
        torch._assert_async(success[lane], "sequential input seal failed")
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
    """One lane-local ``randperm`` per partition and epoch with exact tails."""

    def __init__(
        self,
        stores: Mapping[str, AlignedSequentialStore],
        schedule: LearnerSchedule,
        *,
        generator: torch.Generator,
        statistical_counts: Mapping[str, int] | None = None,
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
        self.statistical_counts = (
            None
            if statistical_counts is None
            else MappingProxyType(dict(statistical_counts))
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
        return cls(
            stores,
            schedule,
            generator=generator,
            statistical_counts={
                name: int(counts[index]) for index, name in enumerate(names)
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
            return rows * self.schedule.epochs
        return self.schedule.gradient_steps * self.schedule.batch_size

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
            for step in range(self.schedule.gradient_steps):
                store = self.stores[names[step % len(names)]]
                pieces: list[Tensor] = []
                remaining = self.schedule.batch_size
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
                    logical_step=step,
                    logical_rows=gathered.rows,
                )
            return
        epoch = 0
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
    """Compacted retained inputs and ragged spans for one input partition."""

    def __init__(
        self,
        slab: MonotoneSlab,
        *,
        input_type: str,
        lane: int,
    ) -> None:
        if slab.schema.name != f"LearnerIndex.{input_type}":
            raise LearnerError("branched input is not the generated learner owner")
        required = {
            "model_input",
            "owner",
            "target_degree",
            "action_begin",
            "action_count",
            "log_sampling_weight",
        }
        if "sampled_private_state" in slab.columns:
            required.add("sampled_private_state")
        if set(slab.columns) != required:
            raise LearnerError("sealed branched input has the wrong fields")
        self.slab = slab
        self.input_type = input_type
        self.lane = lane
        self.rows = slab.tail(lane)
        self.device = slab.device

    @classmethod
    def seal(
        cls,
        family: MonotoneSlab,
        destination: MonotoneSlab,
        *,
        input_type: str,
        family_row: Tensor,
        action_begin: Tensor,
        action_count: Tensor,
        target_degree: Tensor,
        log_sampling_weight: Tensor,
        lane: int = 0,
    ) -> SealedBranchedInputs:
        if family.schema.name != f"FamilyWork.{input_type}":
            raise LearnerError("branched input sealing is bound to the wrong family")
        if destination.schema.name != f"LearnerIndex.{input_type}":
            raise LearnerError("branched input destination has the wrong identity")
        rows = int(family_row.numel())
        device = family.device
        for name, value in (
            ("family_row", family_row),
            ("action_begin", action_begin),
            ("action_count", action_count),
            ("target_degree", target_degree),
            ("log_sampling_weight", log_sampling_weight),
        ):
            if value.device != device or value.shape != (rows,):
                raise LearnerError(f"{name} must be device-local [trained rows]")
        if rows == 0:
            return cls(destination, input_type=input_type, lane=lane)
        torch._assert_async(
            (
                family_row.ge(0)
                & family_row.lt(family.tail(lane))
                & action_begin.ge(0)
                & action_count.ge(1)
                & target_degree.eq(action_count)
                & torch.isfinite(log_sampling_weight)
            ).all(),
            "branched seal metadata is invalid",
        )
        selected = family_row.to(torch.int64)
        lane_values = {
            "model_input": family.columns["model_input"][lane, selected],
            "owner": family.columns["owner"][lane, selected],
            "target_degree": target_degree.to(torch.int32),
            "action_begin": action_begin.to(torch.int32),
            "action_count": action_count.to(torch.int32),
            "log_sampling_weight": log_sampling_weight.float(),
        }
        if "sampled_private_state" in destination.columns:
            if "sampled_private_state" not in family.columns:
                raise LearnerError("sampled branched seal omitted private state")
            lane_values["sampled_private_state"] = family.columns[
                "sampled_private_state"
            ][lane, selected]
        values = {
            field.name: torch.full(
                (destination.n_replica, rows, *field.shape),
                field.fill,
                dtype=field.dtype,
                device=device,
            )
            for field in destination.schema.fields
        }
        for name, value in lane_values.items():
            values[name][lane].copy_(value)
        counts = torch.zeros(family.n_replica, dtype=torch.int64, device=device)
        counts[lane] = rows
        logical = torch.full_like(counts, torch.iinfo(torch.int64).max)
        logical[lane] = 0
        transaction = PublicationTransaction.begin_device(
            (
                DeviceDestinationRequest(
                    destination,
                    counts,
                    values,
                    input_type,
                    logical,
                ),
            ),
            phase="branched_input_seal",
            active=torch.arange(family.n_replica, device=device).eq(lane),
        )
        transaction.fill_all()
        success = transaction.publish()
        torch._assert_async(success[lane], "branched input seal failed")
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
        sigma_fields: dict[str, Tensor] = {
            name: value[selected]
            for name, value in index.items()
            if name not in ("action_begin", "action_count")
        }
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

    def batches(self) -> Iterator[BranchedLearnerBatch]:
        steps = (
            self.schedule.gradient_steps
            if self.schedule.gradient_steps is not None
            else self.schedule.epochs
        )
        logical_rows = sum(self.statistical_counts.values())
        for logical_step in range(steps):
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

    def detached_metrics(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in (
                "total",
                "policy",
                "value",
                "entropy",
                "reference_kl",
                "approximate_kl",
                "clip_fraction",
            )
        }


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
    """One authoritative forward/loss/immediate-backward operation."""

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
        self.precision = precision
        self.dtypes = LearnerDTypes(
            {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[precision]
        )

    def forward_loss_backward(
        self,
        batch: SequentialLearnerBatch | BranchedLearnerBatch,
        *,
        iteration: int,
        reference_operation: ModelOperation | None = None,
        reference_cache: ReferenceOutputCache | None = None,
        loss_scale: float = 1.0,
        objective_weight: float | Tensor = 1.0,
    ) -> LossResult:
        if self.mode.rollout_mode.is_branched:
            if not isinstance(batch, BranchedLearnerBatch):
                raise LearnerError("branched learner requires a ragged batch")
            return self._forward_branched_loss_backward(
                batch,
                iteration=iteration,
                reference_operation=reference_operation,
                reference_cache=reference_cache,
                loss_scale=loss_scale,
                objective_weight=objective_weight,
            )
        if not isinstance(batch, SequentialLearnerBatch):
            raise LearnerError("sequential learner requires an aligned batch")
        fields = batch.fields
        autocast_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise LearnerError("selected CUDA device does not support BF16")
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            evaluation = self.model_operation(
                batch.input_type,
                fields["model_input"],
                fields.get("sampled_private_state", fields["owner"][:, None]),
                fields["owner"],
            )
            sampled = self.mode.private_estimator is PrivateEstimator.SAMPLED
            if sampled:
                evaluation = ModelEvaluation(
                    MappingProxyType(
                        {
                            name: _select_sampled_private(
                                value,
                                fields["sampled_private_state"],
                                fields["owner"],
                            )
                            for name, value in evaluation.policy_raw.items()
                        }
                    ),
                    _select_sampled_private(
                        evaluation.value,
                        fields["sampled_private_state"],
                        fields["owner"],
                    ),
                    evaluation.public_heads,
                )
            if reference_cache is not None:
                if reference_operation is not None:
                    raise LearnerError(
                        "reference cache and reference operation are mutually exclusive"
                    )
                cached = reference_cache.gather(batch.indices)
                reference = (
                    {
                        name: _select_sampled_private(
                            value,
                            fields["sampled_private_state"],
                            fields["owner"],
                        )
                        for name, value in cached.items()
                    }
                    if sampled
                    else dict(cached)
                )
            else:
                reference = (
                    None
                    if reference_operation is None
                    else {
                        name: (
                            _select_sampled_private(
                                value,
                                fields["sampled_private_state"],
                                fields["owner"],
                            )
                            if sampled
                            else value
                        )
                        for name, value in reference_operation(
                            batch.input_type,
                            fields["model_input"],
                            fields.get(
                                "sampled_private_state", fields["owner"][:, None]
                            ),
                            fields["owner"],
                        ).policy_raw.items()
                    }
                )
        compact_private = (
            torch.zeros_like(fields["sampled_private_state"])
            if sampled
            else fields["owner"][:, None]
        )
        policy = self.policy_operation(
            batch.input_type,
            fields["model_input"],
            evaluation.policy_raw,
            fields["action"].to(torch.int32),
            compact_private,
            fields["owner"].to(torch.int64),
            reference,
        )
        if sampled:
            current_log = _scalar_private_row(
                policy.selected_log_policy, compact_private, fields["owner"]
            )
            entropy = _scalar_private_row(
                policy.entropy, compact_private, fields["owner"]
            )
            reference_kl = _scalar_private_row(
                policy.reference_kl, compact_private, fields["owner"]
            )
            current_value = _scalar_private_row(
                evaluation.value, compact_private, fields["owner"]
            )
        else:
            current_log = policy.selected_log_policy.float()
            entropy = policy.entropy.float()
            reference_kl = policy.reference_kl.float()
            current_value = evaluation.value.float()
        old_log = fields["old_log_policy"].float()
        old_value = fields["old_value"].float()
        target_value = fields["value_target"].float()
        advantage = fields["advantage_target"].float()
        if sampled:
            old_log = old_log.reshape(batch.rows, -1)[:, 0]
            old_value = old_value.reshape(batch.rows, -1)[:, 0]
            target_value = target_value.reshape(batch.rows, -1)[:, 0]
            advantage = advantage.reshape(batch.rows, -1)[:, 0]
        if not all(
            tensor.shape == current_log.shape
            for tensor in (
                entropy,
                reference_kl,
                current_value,
                old_log,
                old_value,
                target_value,
                advantage,
            )
        ):
            raise LearnerError("learner fields are not batch/domain aligned")
        posterior = (
            torch.ones_like(current_log) if sampled else fields["posterior"].float()
        )
        if posterior.shape != current_log.shape:
            raise LearnerError("RB posterior is not aligned to the natural domain")
        posterior_mass = posterior.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        posterior = posterior / posterior_mass
        log_ratio = current_log - old_log
        ratio = torch.exp(log_ratio)
        clipped_ratio = ratio.clamp(
            1.0 - self.loss.clip_epsilon,
            1.0 + self.loss.clip_epsilon,
        )
        policy_term = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
        error = (current_value - target_value).square()
        if self.loss.value_loss == "clipped":
            clipped_value = old_value + (current_value - old_value).clamp(
                -self.loss.value_clip_epsilon,
                self.loss.value_clip_epsilon,
            )
            error = torch.maximum(error, (clipped_value - target_value).square())
        if sampled:
            policy_loss = policy_term.mean()
            value_loss = error.mean()
            entropy_mean = entropy.mean()
            reference_kl_mean = reference_kl.mean()
        else:
            policy_loss = (posterior * policy_term).sum(-1).mean()
            value_loss = (posterior * error).sum(-1).mean()
            entropy_mean = (posterior * entropy).sum(-1).mean()
            reference_kl_mean = (posterior * reference_kl).sum(-1).mean()
        total = (
            policy_loss
            + self.loss.value_coefficient * value_loss
            - self.loss.entropy_at(iteration) * entropy_mean
            + self.loss.reference_kl_coefficient * reference_kl_mean
        )
        if self.precision != "fp16" and not torch.isfinite(total):
            raise LearnerError("learner loss is nonfinite")
        if (
            not math.isfinite(objective_weight)
            or objective_weight <= 0
            or objective_weight > 1
        ):
            raise LearnerError("physical objective weight must lie in (0,1]")
        (total * float(loss_scale) * objective_weight).backward()
        return LossResult(
            total,
            policy_loss,
            value_loss,
            entropy_mean,
            reference_kl_mean,
            (
                (((ratio - 1.0) - log_ratio) * posterior).sum(-1).mean()
                if not sampled
                else ((ratio - 1.0) - log_ratio).mean()
            ),
            (
                ((torch.abs(ratio - 1.0) > self.loss.clip_epsilon).float() * posterior)
                .sum(-1)
                .mean()
                if not sampled
                else (torch.abs(ratio - 1.0) > self.loss.clip_epsilon).float().mean()
            ),
        )

    def _forward_branched_loss_backward(
        self,
        batch: BranchedLearnerBatch,
        *,
        iteration: int,
        reference_operation: ModelOperation | None,
        reference_cache: ReferenceOutputCache | None,
        loss_scale: float,
        objective_weight: float | Tensor,
    ) -> LossResult:
        if self.mode.private_estimator is PrivateEstimator.RAO_BLACKWELL:
            return self._forward_branched_rb_loss_backward(
                batch,
                iteration=iteration,
                reference_operation=reference_operation,
                reference_cache=reference_cache,
                loss_scale=loss_scale,
                objective_weight=objective_weight,
            )
        sigma = batch.sigma_fields
        action = batch.action_fields
        counts = batch.action_ptr[1:] - batch.action_ptr[:-1]
        torch._assert_async(
            counts.eq(sigma["target_degree"].to(counts.dtype)).all(),
            "ragged action spans disagree with protected target degrees",
        )
        action_to_sigma = torch.repeat_interleave(
            torch.arange(batch.rows, device=batch.sigma_index.device),
            counts,
            output_size=batch.action_rows,
        )
        autocast_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise LearnerError("selected CUDA device does not support BF16")
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            evaluation = self.model_operation(
                batch.input_type,
                sigma["model_input"],
                sigma["sampled_private_state"],
                sigma["owner"],
            )
            natural_raw = {
                name: _select_sampled_private(
                    value,
                    sigma["sampled_private_state"],
                    sigma["owner"],
                )
                for name, value in evaluation.policy_raw.items()
            }
            natural_value = _select_sampled_private(
                evaluation.value,
                sigma["sampled_private_state"],
                sigma["owner"],
            )
            repeated_raw = {
                name: value.index_select(0, action_to_sigma)
                for name, value in natural_raw.items()
            }
            repeated_private = sigma["sampled_private_state"].index_select(
                0, action_to_sigma
            )
            repeated_owner = sigma["owner"].index_select(0, action_to_sigma)
            reference = None
            if reference_cache is not None:
                if reference_operation is not None:
                    raise LearnerError(
                        "reference cache and operation are mutually exclusive"
                    )
                reference = {
                    name: _select_sampled_private(
                        value,
                        sigma["sampled_private_state"],
                        sigma["owner"],
                    ).index_select(0, action_to_sigma)
                    for name, value in reference_cache.gather(batch.sigma_index).items()
                }
            elif reference_operation is not None:
                reference_evaluation = reference_operation(
                    batch.input_type,
                    sigma["model_input"],
                    sigma["sampled_private_state"],
                    sigma["owner"],
                )
                reference = {
                    name: _select_sampled_private(
                        value,
                        sigma["sampled_private_state"],
                        sigma["owner"],
                    ).index_select(0, action_to_sigma)
                    for name, value in reference_evaluation.policy_raw.items()
                }
        compact_private = torch.zeros_like(repeated_private)
        policy = self.policy_operation(
            batch.input_type,
            sigma["model_input"].index_select(0, action_to_sigma),
            repeated_raw,
            action["action"].to(torch.int32),
            compact_private,
            repeated_owner.to(torch.int64),
            reference,
        )
        current_log = _scalar_private_row(
            policy.selected_log_policy, compact_private, repeated_owner
        )
        entropy = _scalar_private_row(policy.entropy, compact_private, repeated_owner)
        reference_kl = _scalar_private_row(
            policy.reference_kl, compact_private, repeated_owner
        )
        current_value = _scalar_private_row(
            natural_value,
            torch.zeros_like(sigma["sampled_private_state"]),
            sigma["owner"],
        )
        old_log = action["old_log_policy"].reshape(batch.action_rows, -1)[:, 0]
        advantage = action["advantage_target"].reshape(batch.action_rows, -1)[:, 0]
        old_value = sigma["old_value"].reshape(batch.rows, -1)[:, 0]
        target_value = sigma["value_target"].reshape(batch.rows, -1)[:, 0]
        if not (
            current_log.shape
            == entropy.shape
            == reference_kl.shape
            == old_log.shape
            == advantage.shape
            == (batch.action_rows,)
            and current_value.shape
            == old_value.shape
            == target_value.shape
            == (batch.rows,)
        ):
            raise LearnerError("branched learner fields are not ragged-aligned")
        log_ratio = current_log.float() - old_log.float()
        ratio = log_ratio.exp()
        clipped = ratio.clamp(
            1 - self.loss.clip_epsilon,
            1 + self.loss.clip_epsilon,
        )
        policy_row = -torch.minimum(
            ratio * advantage.float(), clipped * advantage.float()
        )
        sigma_policy = torch.zeros(
            batch.rows, dtype=torch.float32, device=current_log.device
        )
        sigma_entropy = torch.zeros_like(sigma_policy)
        sigma_reference = torch.zeros_like(sigma_policy)
        sigma_approximate = torch.zeros_like(sigma_policy)
        sigma_clipped = torch.zeros_like(sigma_policy)
        denominator = counts.float()
        for destination, source in (
            (sigma_policy, policy_row),
            (sigma_entropy, entropy.float()),
            (sigma_reference, reference_kl.float()),
            (sigma_approximate, (ratio - 1) - log_ratio),
            (
                sigma_clipped,
                ratio.sub(1).abs().gt(self.loss.clip_epsilon).float(),
            ),
        ):
            destination.scatter_add_(0, action_to_sigma, source)
            destination.div_(denominator)
        value_error = (current_value.float() - target_value.float()).square()
        if self.loss.value_loss == "clipped":
            clipped_value = old_value.float() + (
                current_value.float() - old_value.float()
            ).clamp(
                -self.loss.value_clip_epsilon,
                self.loss.value_clip_epsilon,
            )
            value_error = torch.maximum(
                value_error,
                (clipped_value - target_value.float()).square(),
            )
        public = evaluation.public_heads
        if set(public) != {"trace", "cost"}:
            raise LearnerError("branched model must expose public trace/cost heads")
        trace_prediction = public["trace"].float()
        cost_prediction = public["cost"].float()
        trace_target = sigma["trace_target"].float()
        cost_target = sigma["cost_target"].float()
        if not all(
            value.shape == (batch.rows,)
            for value in (
                trace_prediction,
                cost_prediction,
                trace_target,
                cost_target,
            )
        ):
            raise LearnerError("branched public heads and targets are not aligned")
        eps = self.loss.trace_epsilon
        trace_ratio = (trace_target + eps) / (
            trace_prediction.clamp_min(torch.finfo(torch.float32).tiny) + eps
        )
        trace_deviance = trace_ratio - trace_ratio.log() - 1
        trace_valid = sigma["target_degree"].ge(2)
        trace_loss = torch.where(
            trace_valid, trace_deviance, torch.zeros_like(trace_deviance)
        ).mean()
        torch._assert_async(
            cost_target.gt(0).all(),
            "branched cost targets must be positive",
        )
        cost_ratio = cost_target / cost_prediction.clamp_min(
            torch.finfo(torch.float32).tiny
        )
        cost_loss = (cost_ratio - cost_ratio.log() - 1).mean()
        policy_loss = sigma_policy.mean()
        value_loss = value_error.mean()
        entropy_mean = sigma_entropy.mean()
        reference_kl_mean = sigma_reference.mean()
        total = (
            policy_loss
            + self.loss.value_coefficient * value_loss
            - self.loss.entropy_at(iteration) * entropy_mean
            + self.loss.reference_kl_coefficient * reference_kl_mean
            + self.loss.trace_coefficient * trace_loss
            + self.loss.cost_coefficient * cost_loss
        )
        if self.precision != "fp16" and not torch.isfinite(total):
            raise LearnerError("branched learner loss is nonfinite")
        if isinstance(objective_weight, Tensor):
            if (
                objective_weight.device != batch.sigma_index.device
                or objective_weight.numel() != 1
            ):
                raise LearnerError(
                    "physical objective weight must be one device scalar"
                )
            torch._assert_async(
                torch.isfinite(objective_weight)
                & objective_weight.gt(0)
                & objective_weight.le(1),
                "physical objective weight must lie in (0,1]",
            )
        elif not math.isfinite(objective_weight) or not 0 < objective_weight <= 1:
            raise LearnerError("physical objective weight must lie in (0,1]")
        (total * float(loss_scale) * objective_weight).backward()
        return LossResult(
            total,
            policy_loss,
            value_loss,
            entropy_mean,
            reference_kl_mean,
            sigma_approximate.mean(),
            sigma_clipped.mean(),
        )

    def _forward_branched_rb_loss_backward(
        self,
        batch: BranchedLearnerBatch,
        *,
        iteration: int,
        reference_operation: ModelOperation | None,
        reference_cache: ReferenceOutputCache | None,
        loss_scale: float,
        objective_weight: float | Tensor,
    ) -> LossResult:
        """Natural-domain RB loss over exact ragged action rows."""

        sigma = batch.sigma_fields
        action = batch.action_fields
        counts = batch.action_ptr[1:] - batch.action_ptr[:-1]
        torch._assert_async(
            counts.eq(sigma["target_degree"].to(counts.dtype)).all(),
            "ragged RB spans disagree with protected target degrees",
        )
        action_to_sigma = torch.repeat_interleave(
            torch.arange(batch.rows, device=batch.sigma_index.device),
            counts,
            output_size=batch.action_rows,
        )
        compact_private = sigma["owner"][:, None]
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
            evaluation = self.model_operation(
                batch.input_type,
                sigma["model_input"],
                compact_private,
                sigma["owner"],
            )
            repeated_raw = {
                name: value.index_select(0, action_to_sigma)
                for name, value in evaluation.policy_raw.items()
            }
            reference = None
            if reference_cache is not None:
                if reference_operation is not None:
                    raise LearnerError(
                        "reference cache and operation are mutually exclusive"
                    )
                reference = {
                    name: value.index_select(0, action_to_sigma)
                    for name, value in reference_cache.gather(batch.sigma_index).items()
                }
            elif reference_operation is not None:
                reference_evaluation = reference_operation(
                    batch.input_type,
                    sigma["model_input"],
                    compact_private,
                    sigma["owner"],
                )
                reference = {
                    name: value.index_select(0, action_to_sigma)
                    for name, value in reference_evaluation.policy_raw.items()
                }
        repeated_owner = sigma["owner"].index_select(0, action_to_sigma)
        policy = self.policy_operation(
            batch.input_type,
            sigma["model_input"].index_select(0, action_to_sigma),
            repeated_raw,
            action["action"].to(torch.int32),
            repeated_owner[:, None],
            repeated_owner,
            reference,
        )
        current_log = policy.selected_log_policy.float()
        entropy = policy.entropy.float()
        reference_kl = policy.reference_kl.float()
        current_value = evaluation.value.float()
        old_log = action["old_log_policy"].float()
        advantage = action["advantage_target"].float()
        posterior = action["posterior"].float()
        old_value = sigma["old_value"].float()
        target_value = sigma["value_target"].float()
        if not (
            current_log.shape
            == entropy.shape
            == reference_kl.shape
            == old_log.shape
            == advantage.shape
            == posterior.shape
            and current_value.shape == old_value.shape == target_value.shape
            and current_log.shape[0] == batch.action_rows
            and current_value.shape[0] == batch.rows
            and current_log.shape[1:] == current_value.shape[1:]
        ):
            raise LearnerError(
                "branched RB fields are not ragged/natural-domain aligned"
            )
        posterior = posterior / posterior.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        log_ratio = current_log - old_log
        ratio = log_ratio.exp()
        clipped = ratio.clamp(
            1 - self.loss.clip_epsilon,
            1 + self.loss.clip_epsilon,
        )
        policy_row = (
            posterior
            * -torch.minimum(
                ratio * advantage,
                clipped * advantage,
            )
        ).sum(-1)
        entropy_row = (posterior * entropy).sum(-1)
        reference_row = (posterior * reference_kl).sum(-1)
        approximate_row = (posterior * ((ratio - 1) - log_ratio)).sum(-1)
        clipped_row = (
            posterior * ratio.sub(1).abs().gt(self.loss.clip_epsilon).float()
        ).sum(-1)
        sigma_posterior = torch.zeros_like(current_value)
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
        if self.loss.value_loss == "clipped":
            clipped_value = old_value + (current_value - old_value).clamp(
                -self.loss.value_clip_epsilon,
                self.loss.value_clip_epsilon,
            )
            value_error = torch.maximum(
                value_error,
                (clipped_value - target_value).square(),
            )
        value_row = (sigma_posterior * value_error).sum(-1)

        def mean_by_sigma(rows: Tensor) -> Tensor:
            result = torch.zeros(batch.rows, dtype=torch.float32, device=rows.device)
            result.scatter_add_(0, action_to_sigma, rows)
            return result / counts.float()

        sigma_policy = mean_by_sigma(policy_row)
        sigma_entropy = mean_by_sigma(entropy_row)
        sigma_reference = mean_by_sigma(reference_row)
        sigma_approximate = mean_by_sigma(approximate_row)
        sigma_clipped = mean_by_sigma(clipped_row)
        public = evaluation.public_heads
        if set(public) != {"trace", "cost"}:
            raise LearnerError("branched model must expose public trace/cost heads")
        trace_prediction = public["trace"].float()
        cost_prediction = public["cost"].float()
        trace_target = sigma["trace_target"].float()
        cost_target = sigma["cost_target"].float()
        eps = self.loss.trace_epsilon
        trace_ratio = (trace_target + eps) / (
            trace_prediction.clamp_min(torch.finfo(torch.float32).tiny) + eps
        )
        trace_loss = torch.where(
            sigma["target_degree"].ge(2),
            trace_ratio - trace_ratio.log() - 1,
            torch.zeros_like(trace_ratio),
        ).mean()
        torch._assert_async(
            cost_target.gt(0).all(),
            "branched cost targets must be positive",
        )
        cost_ratio = cost_target / cost_prediction.clamp_min(
            torch.finfo(torch.float32).tiny
        )
        cost_loss = (cost_ratio - cost_ratio.log() - 1).mean()
        policy_loss = sigma_policy.mean()
        value_loss = value_row.mean()
        entropy_mean = sigma_entropy.mean()
        reference_kl_mean = sigma_reference.mean()
        total = (
            policy_loss
            + self.loss.value_coefficient * value_loss
            - self.loss.entropy_at(iteration) * entropy_mean
            + self.loss.reference_kl_coefficient * reference_kl_mean
            + self.loss.trace_coefficient * trace_loss
            + self.loss.cost_coefficient * cost_loss
        )
        if not torch.isfinite(total):
            raise LearnerError("branched RB learner loss is nonfinite")
        if isinstance(objective_weight, Tensor):
            torch._assert_async(
                torch.isfinite(objective_weight)
                & objective_weight.gt(0)
                & objective_weight.le(1),
                "physical objective weight must lie in (0,1]",
            )
        elif not math.isfinite(objective_weight) or not 0 < objective_weight <= 1:
            raise LearnerError("physical objective weight must lie in (0,1]")
        (total * float(loss_scale) * objective_weight).backward()
        return LossResult(
            total,
            policy_loss,
            value_loss,
            entropy_mean,
            reference_kl_mean,
            sigma_approximate.mean(),
            sigma_clipped.mean(),
        )


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

    def observe(self, *, iteration: int, effective_hands: float) -> bool:
        if effective_hands < 0 or not math.isfinite(effective_hands):
            raise ValueError("effective hands must be finite and nonnegative")
        self.effective_hands_rollout += effective_hands
        if self.interval_iterations is not None:
            due = iteration - self.reference_iteration >= self.interval_iterations
        else:
            assert self.interval_effective_hands is not None
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
    "allocate_statistical_counts",
    "build_reference_output_cache",
    "compact_best_response_evaluation",
]
