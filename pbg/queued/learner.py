"""Aligned sequential sampling and the sole queued learner operation.

The learner consumes only the sealed owners produced by the queued rollout and
target executor.  It never receives topology, rollout queues, leaves, or work
arenas.  A physical batch is one CUDA index vector applied to the five aligned
owner families, followed by one model forward, one loss, and an immediate
backward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterator, Mapping, Protocol

import torch
from torch import Tensor, nn

from pbg.queued.capacity import GeneratedOwners
from pbg.queued.contracts import (
    ModeSchema,
    PrivateEstimator,
    RolloutMode,
    TrainingRegime,
)
from pbg.queued.storage import (
    DeviceDestinationRequest,
    MonotoneSlab,
    PublicationTransaction,
)


class LearnerError(RuntimeError):
    """A sealed learner input or loss contract was violated."""


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
    ) -> "ReferenceOutputCache | None":
        if min(byte_budget, resident_bytes, build_bytes) < 0:
            raise ValueError("reference cache byte values must be nonnegative")
        if (
            not math.isfinite(predicted_saved_seconds)
            or not math.isfinite(predicted_build_seconds)
            or min(predicted_saved_seconds, predicted_build_seconds) < 0
        ):
            raise ValueError("reference cache time predictions must be nonnegative")
        payload = sum(value.numel() * value.element_size() for value in fields.values())
        if (
            resident_bytes + build_bytes + payload > byte_budget
            or predicted_saved_seconds <= predicted_build_seconds
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

    def __post_init__(self) -> None:
        finite = (
            self.clip_epsilon,
            self.value_clip_epsilon,
            self.value_coefficient,
            self.entropy_coefficient,
            self.reference_kl_coefficient,
        )
        if any(not math.isfinite(value) or value < 0 for value in finite):
            raise ValueError("loss coefficients and clips must be finite and nonnegative")
        if self.value_loss not in ("mse", "clipped"):
            raise ValueError("value_loss must be 'mse' or 'clipped'")
        if self.entropy_schedule not in ("constant", "linear_decay"):
            raise ValueError(
                "entropy_schedule must be 'constant' or 'linear_decay'"
            )
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
        fraction = max(
            0.0, 1.0 - float(iteration) / self.entropy_decay_iterations
        )
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
    ) -> "SealedSequentialInputs":
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
        counts = torch.zeros(
            family.n_replica, dtype=torch.int64, device=family.device
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
            phase="sequential_input_seal",
            active=torch.arange(
                family.n_replica, device=family.device
            ).eq(lane),
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
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
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
    ) -> "SequentialBatchSampler":
        names = tuple(stores)
        if mass.numel() != len(names):
            raise LearnerError(
                "cost-aware allocation must match learner partitions"
            )
        counts = allocate_statistical_counts(
            mass, variance, seconds_per_draw, total_count
        )
        return cls(
            stores,
            schedule,
            generator=generator,
            statistical_counts={
                name: int(counts[index])
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


def _select_sampled_private(
    value: Tensor, private: Tensor, owner: Tensor
) -> Tensor:
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
    index = combo.reshape(
        combo.shape[0], 1, *((1,) * (value.ndim - 2))
    ).expand(combo.shape[0], 1, *value.shape[2:])
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
        if mode.rollout_mode is not RolloutMode.SEQUENTIAL:
            raise LearnerError("the aligned learner operation requires sequential mode")
        self.mode = mode
        self.loss = loss
        self.model_operation = model_operation
        self.policy_operation = policy_operation
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("learner precision must be fp32, fp16, or bf16")
        self.precision = precision

    def forward_loss_backward(
        self,
        batch: SequentialLearnerBatch,
        *,
        iteration: int,
        reference_operation: ModelOperation | None = None,
        reference_cache: ReferenceOutputCache | None = None,
        loss_scale: float = 1.0,
        objective_weight: float = 1.0,
    ) -> LossResult:
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
                            fields.get("sampled_private_state", fields["owner"][:, None]),
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
        if not all(tensor.shape == current_log.shape for tensor in (
            entropy, reference_kl, current_value, old_log, old_value,
            target_value, advantage,
        )):
            raise LearnerError("learner fields are not batch/domain aligned")
        posterior = (
            torch.ones_like(current_log)
            if sampled
            else fields["posterior"].float()
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
        policy_term = -torch.minimum(
            ratio * advantage, clipped_ratio * advantage
        )
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
                if not sampled else ((ratio - 1.0) - log_ratio).mean()
            ),
            (
                (
                    (torch.abs(ratio - 1.0) > self.loss.clip_epsilon).float()
                    * posterior
                ).sum(-1).mean()
                if not sampled
                else (torch.abs(ratio - 1.0) > self.loss.clip_epsilon).float().mean()
            ),
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
        if (
            self.interval_effective_hands is not None
            and (
                not math.isfinite(self.interval_effective_hands)
                or self.interval_effective_hands <= 0
            )
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
    def from_state_dict(cls, state: Mapping[str, object]) -> "ReferenceState":
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
    "allocate_statistical_counts",
    "LearnerError",
    "LearnerOperation",
    "LearnerSchedule",
    "LossConfig",
    "LossResult",
    "ModelEvaluation",
    "PolicyEvaluation",
    "ReferenceState",
    "ReferenceOutputCache",
    "SealedSequentialInputs",
    "SequentialBatchSampler",
    "SequentialLearnerBatch",
]
