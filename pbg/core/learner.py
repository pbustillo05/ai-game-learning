"""Bounded stratified learner over immutable :class:`TrainingStorage`.

This module is the production-v1 storage-to-loss and physical-execution
boundary.  Statistical draws are index-only; gathered tensors and autograd
graphs exist for one ordered physical subspan and are released after backward.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import inspect
import math
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

from pbg.core.adaptive_branching import gamma_deviance
from pbg.core.distributed_failure import DistributedFailureAgreement
from pbg.core.losses import _entropy_coefficient
from pbg.core.optimization import (
    FlatGradientBuffer,
    OptimizerStepResult,
    finite_gated_optimizer_step,
    sync_scaled_gradients,
)
from pbg.core.partition import PartitionSet, PartitionSpec
from pbg.core.storage import TrainingPartition, TrainingStorage
from pbg.core.targets import finalize_training_storage


_FORBIDDEN_LEARNER_FIELDS = frozenset(
    {
        "topology",
        "tree",
        "reach",
        "log_r",
        "leaf_reach",
        "completion",
        "frontier",
        "rollout_queue",
        "rollout_state",
        "occurrence",
        "replay",
        "q_hat",
        "qHat",
        "W",
        "S",
    }
)


def validate_learner_input(storage: TrainingStorage, partitions: PartitionSet) -> None:
    """Enforce the immutable, partition-addressed learner boundary."""

    if type(storage) is not TrainingStorage:
        raise TypeError("DistributedFixedRolloutUpdate accepts only TrainingStorage")
    finalize_training_storage(storage)
    known = {spec.partition_id for spec in partitions.specs}
    if not set(storage.partitions).issubset(known):
        raise ValueError("training storage contains an unknown partition")
    if set(storage.sampling.log_z) != set(storage.partitions):
        raise ValueError("every local partition must retain one fp64 log Z")
    for partition_id, block in storage.partitions.items():
        if type(block) is not TrainingPartition:
            raise TypeError("learner partitions must be immutable TrainingPartition values")
        if block.partition_id != partition_id:
            raise ValueError("partition key and stable id disagree")
        overlap = _FORBIDDEN_LEARNER_FIELDS.intersection(block.columns)
        if block.action_columns is not None:
            overlap |= _FORBIDDEN_LEARNER_FIELDS.intersection(block.action_columns)
        if overlap:
            raise ValueError(f"learner input retained forbidden fields {sorted(overlap)}")
        spec = partitions[partition_id]
        if block.inputs.schema != spec.input_schema:
            raise ValueError("training input schema does not match its opaque PartitionSpec")
        log_z = storage.sampling.log_z[partition_id]
        if block.inputs.rows == 0 and not bool(torch.isneginf(log_z)):
            raise ValueError("empty local strata must have log Z = -inf")
        if block.inputs.rows and not bool(torch.isfinite(log_z)):
            raise ValueError("nonempty local strata require finite log Z")
        branched = storage.mode == "branched_rb"
        has_alias = partition_id in storage.sampling.alias_probability
        if branched != has_alias and block.inputs.rows:
            raise ValueError("branched strata require aliases; sequential strata are uniform")


@dataclass(frozen=True)
class GlobalMassExchange:
    """One common-order asynchronous all-gather of local partition log masses."""

    local: Tensor
    gathered: tuple[Tensor, ...]
    work: Any | None
    started: float

    def wait(self) -> tuple[Tensor, Tensor]:
        if self.work is not None:
            self.work.wait()
        all_logs = torch.stack(self.gathered)
        log_z = torch.logsumexp(all_logs.flatten(), 0)
        if not bool(torch.isfinite(log_z)):
            raise ValueError("distributed fixed rollout has zero decision mass")
        return all_logs, log_z


def begin_log_mass_exchange(
    storage: TrainingStorage,
    partitions: PartitionSet,
    *,
    group: Any | None = None,
) -> GlobalMassExchange:
    """Launch IAllGather without a preceding barrier, including empty strata."""

    device = next(iter(storage.sampling.log_z.values())).device
    local = torch.stack(
        [
            storage.sampling.log_z.get(
                spec.partition_id,
                torch.tensor(-torch.inf, dtype=torch.float64, device=device),
            ).to(device=device, dtype=torch.float64)
            for spec in partitions.specs
        ]
    )
    started = time.perf_counter()
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size(group)
        gathered = tuple(torch.empty_like(local) for _ in range(world))
        work = dist.all_gather(list(gathered), local, group=group, async_op=True)
    else:
        gathered = (local,)
        work = None
    return GlobalMassExchange(local, gathered, work, started)


def global_stratum_alphas(all_log_z: Tensor) -> Tensor:
    """Return exact ``alpha[g,m]`` values, with zeros for empty strata."""

    if all_log_z.ndim != 2 or all_log_z.dtype != torch.float64:
        raise ValueError("all_log_z must be fp64 [world, partitions]")
    log_total = torch.logsumexp(all_log_z.flatten(), 0)
    if not bool(torch.isfinite(log_total)):
        raise ValueError("global log mass must be finite")
    return torch.where(
        torch.isfinite(all_log_z),
        torch.exp(all_log_z - log_total),
        torch.zeros_like(all_log_z),
    )


def allocate_fixed_statistical_counts(
    mass: Tensor,
    variance: Tensor,
    seconds_per_draw: Tensor,
    total_count: int,
) -> Tensor:
    """Allocate a fixed draw count with a mass/variance/cost proxy.

    Every positive-mass stratum receives one draw.  Remaining draws use the
    largest-remainder apportionment of ``mass * sqrt(variance / cost)``.  The
    per-stratum estimator remains exactly unbiased because execution applies
    the corresponding exact stratum coefficient divided by the returned
    positive count.
    """

    if mass.ndim != 1 or variance.shape != mass.shape or seconds_per_draw.shape != mass.shape:
        raise ValueError("statistical allocation inputs must be aligned vectors")
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
        raise ValueError("mass must be finite and nonnegative")
    if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise ValueError("lagged variances must be finite and nonnegative")
    if not bool(torch.isfinite(seconds_per_draw).all()) or bool((seconds_per_draw <= 0).any()):
        raise ValueError("measured draw costs must be finite and positive")
    if isinstance(total_count, bool) or total_count <= 0:
        raise ValueError("total statistical count must be positive")
    active = mass > 0
    active_count = int(active.sum())
    if total_count < active_count:
        raise ValueError("total count cannot cover every positive-mass stratum")
    counts = active.to(torch.int64)
    remaining = total_count - active_count
    if remaining == 0:
        return counts.to(torch.int32)
    score = torch.where(
        active,
        mass * torch.sqrt(variance / seconds_per_draw),
        torch.zeros_like(mass),
    )
    total_score = score.sum()
    if not bool(total_score > 0):
        score = active.to(mass.dtype)
        total_score = score.sum()
    raw = remaining * score / total_score
    extra = torch.floor(raw).to(torch.int64)
    counts += extra
    leftover = remaining - int(extra.sum())
    if leftover:
        fractions = raw - extra.to(raw)
        # Stable sorting makes lower partition positions the deterministic
        # tie-break, matching the fixed partition order used by the learner.
        order = torch.argsort(fractions, descending=True, stable=True)
        counts[order[:leftover]] += 1
    return counts.to(torch.int32)


class ReusableIndexBuffer:
    """One fixed int32 destination for all statistical index draws."""

    def __init__(self, capacity: int, device: torch.device | str) -> None:
        if isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("index-buffer capacity must be positive")
        self.tensor = torch.empty(capacity, dtype=torch.int32, device=device)

    @property
    def capacity(self) -> int:
        return self.tensor.numel()

    def draw_uniform(self, rows: int, count: int, generator: torch.Generator) -> Tensor:
        if rows <= 0 or count <= 0 or count > self.capacity:
            raise ValueError("uniform draw exceeds its row domain or index cap")
        view = self.tensor[:count]
        torch.randint(rows, (count,), generator=generator, device=view.device, out=view)
        return view

    def draw_alias(
        self,
        probability: Tensor,
        alias: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        if count <= 0 or count > self.capacity:
            raise ValueError("alias draw exceeds the index-buffer cap")
        if probability.ndim != 1 or alias.shape != probability.shape:
            raise ValueError("invalid Walker alias table")
        # Columns occupy the reusable destination; uniforms are the only draw
        # temporary and are released before any storage rows are gathered.
        view = self.tensor[:count]
        torch.randint(
            probability.numel(), (count,), generator=generator, device=view.device, out=view
        )
        uniforms = torch.rand(
            count, dtype=torch.float64, generator=generator, device=view.device
        )
        use_alias = uniforms >= probability[view.to(torch.int64)]
        view[use_alias] = alias[view[use_alias].to(torch.int64)]
        return view


@dataclass(frozen=True)
class PhysicalShape:
    rows: int
    transient_bytes: int
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.transient_bytes < 0:
            raise ValueError("physical shapes require positive rows and nonnegative bytes")


@dataclass(frozen=True)
class PhysicalSubspan:
    begin: int
    end: int
    shape_rows: int
    transient_bytes: int
    activation_checkpointing: bool = False

    def select(self, indices: Tensor) -> Tensor:
        return indices[self.begin : self.end]


class PhysicalPlanner:
    """Order-preserving finite-shape planner with an exact tail."""

    def __init__(
        self,
        shapes: Sequence[PhysicalShape],
        *,
        tail_predictor: Callable[[int], int] | None = None,
    ) -> None:
        unique = {shape.rows: shape for shape in shapes}
        if not unique:
            raise ValueError("at least one warmed physical shape is required")
        self.shapes = tuple(unique[key] for key in sorted(unique))
        self.tail_predictor = tail_predictor

    def predicted_bytes(self, rows: int) -> int:
        exact = next((shape for shape in self.shapes if shape.rows == rows), None)
        if exact is not None:
            return exact.transient_bytes
        if self.tail_predictor is not None:
            value = int(self.tail_predictor(rows))
        else:
            singleton = self.shapes[0]
            value = math.ceil(rows / singleton.rows) * singleton.transient_bytes
        if value < 0:
            raise ValueError("physical tail predictor returned negative bytes")
        return value

    def plan(self, indices: Tensor, available_bytes: int) -> tuple[PhysicalSubspan, ...]:
        if indices.ndim != 1 or indices.dtype != torch.int32:
            raise TypeError("PhysicalPlan consumes one int32 statistical index vector")
        if available_bytes < 0:
            raise ValueError("available transient bytes must be nonnegative")
        if indices.numel() and self.predicted_bytes(1) > available_bytes:
            raise MemoryError("uncached singleton learner path cannot fit")
        fitting = [shape for shape in self.shapes if shape.transient_bytes <= available_bytes]
        if not fitting and indices.numel():
            raise MemoryError("no calibrated physical shape fits")
        largest = max(fitting, key=lambda shape: shape.rows) if fitting else None
        spans: list[PhysicalSubspan] = []
        cursor = 0
        total = indices.numel()
        while cursor < total:
            assert largest is not None
            remaining = total - cursor
            if remaining >= largest.rows:
                count = largest.rows
                shape = largest
                transient = shape.transient_bytes
                checkpoint = shape.activation_checkpointing
            else:
                count = remaining
                transient = self.predicted_bytes(count)
                checkpoint = False
                if transient > available_bytes:
                    # A smaller warmed shape may be safer than a nonlinear tail.
                    candidates = [s for s in fitting if s.rows <= remaining]
                    if not candidates:
                        raise MemoryError("exact physical tail cannot fit")
                    shape = max(candidates, key=lambda item: item.rows)
                    count = shape.rows
                    transient = shape.transient_bytes
                    checkpoint = shape.activation_checkpointing
            spans.append(
                PhysicalSubspan(cursor, cursor + count, count, transient, checkpoint)
            )
            cursor += count
        return tuple(spans)


PhysicalPlan = PhysicalPlanner


@dataclass(frozen=True)
class ReferenceCacheCandidate:
    partition_id: int
    resident_bytes: int
    build_workspace_bytes: int
    uncached_training_workspace_bytes: int
    cached_training_workspace_bytes: int
    build_seconds: float
    uncached_training_seconds: float
    cached_training_seconds: float

    def __post_init__(self) -> None:
        byte_values = (
            self.resident_bytes,
            self.build_workspace_bytes,
            self.uncached_training_workspace_bytes,
            self.cached_training_workspace_bytes,
        )
        if any(value < 0 for value in byte_values):
            raise ValueError("reference-cache byte predictions must be nonnegative")
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.build_seconds,
                self.uncached_training_seconds,
                self.cached_training_seconds,
            )
        ):
            raise ValueError("reference-cache time predictions must be finite and nonnegative")


@dataclass(frozen=True)
class ReferenceCacheChoice:
    selected: frozenset[int]
    predicted_seconds: float
    predicted_peak_bytes: int


def select_reference_cache_set(
    candidates: Sequence[ReferenceCacheCandidate],
    *,
    resident_base_bytes: int,
    hard_bytes: int,
) -> ReferenceCacheChoice:
    """Exact finite subset search charging destination and max transients."""

    ordered = tuple(sorted(candidates, key=lambda item: item.partition_id))
    if len({item.partition_id for item in ordered}) != len(ordered):
        raise ValueError("reference-cache candidates must have unique partition ids")
    if resident_base_bytes < 0 or hard_bytes < 0:
        raise ValueError("reference-cache memory bounds must be nonnegative")
    best: ReferenceCacheChoice | None = None
    for mask in range(1 << len(ordered)):
        selected = frozenset(
            item.partition_id for index, item in enumerate(ordered) if mask & (1 << index)
        )
        resident = resident_base_bytes + sum(
            item.resident_bytes for item in ordered if item.partition_id in selected
        )
        training_workspace = max(
            (
                item.cached_training_workspace_bytes
                if item.partition_id in selected
                else item.uncached_training_workspace_bytes
                for item in ordered
            ),
            default=0,
        )
        build_workspace = max(
            (
                item.build_workspace_bytes
                for item in ordered
                if item.partition_id in selected
            ),
            default=0,
        )
        peak = resident + max(training_workspace, build_workspace)
        if peak > hard_bytes:
            continue
        seconds = sum(
            (
                item.build_seconds + item.cached_training_seconds
                if item.partition_id in selected
                else item.uncached_training_seconds
            )
            for item in ordered
        )
        choice = ReferenceCacheChoice(selected, seconds, peak)
        if best is None or (choice.predicted_seconds, tuple(sorted(choice.selected))) < (
            best.predicted_seconds,
            tuple(sorted(best.selected)),
        ):
            best = choice
    if best is None:
        raise MemoryError("the required empty-cache singleton path is infeasible")
    return best


def _map_output(value: Any, fn: Callable[[Tensor], Tensor]) -> Any:
    if isinstance(value, Tensor):
        return fn(value)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return type(value)((key, _map_output(item, fn)) for key, item in value.items())
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{field.name: _map_output(getattr(value, field.name), fn) for field in fields(value)}
        )
    if isinstance(value, tuple):
        return type(value)(_map_output(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_output(item, fn) for item in value]
    raise TypeError(f"reference output contains unsupported leaf {type(value).__name__}")


def output_nbytes(value: Any) -> int:
    total = 0

    def count(tensor: Tensor) -> Tensor:
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    _map_output(value, count)
    return total


def _output_rows(value: Any, rows: Tensor) -> Any:
    return _map_output(value, lambda tensor: tensor[rows])


def _empty_output_like(value: Any, rows: int) -> Any:
    def allocate(tensor: Tensor) -> Tensor:
        if tensor.ndim == 0:
            raise ValueError("natural reference outputs must be batch-row aligned")
        return torch.empty((rows, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)

    return _map_output(value, allocate)


def _scatter_output(destination: Any, rows: Tensor, source: Any) -> None:
    def leaves(value: Any) -> list[Tensor]:
        result: list[Tensor] = []
        _map_output(value, lambda tensor: result.append(tensor) or tensor)
        return result

    destination_leaves = leaves(destination)
    source_leaves = leaves(source)
    if len(destination_leaves) != len(source_leaves):
        raise ValueError("reference cache output structure changed during build")
    for target, value in zip(destination_leaves, source_leaves, strict=True):
        target[rows] = value


def _forward(spec: PartitionSpec, model: nn.Module, inputs: Mapping[str, Tensor]) -> Any:
    """Call the opaque natural forward, supporting the explicit v1 signature."""

    signature = inspect.signature(spec.forward)
    if len(signature.parameters) >= 2:
        output = spec.forward(model, inputs)  # type: ignore[misc]
    else:
        output = spec.forward(inputs)
    value = getattr(output, "v", None)
    if not isinstance(value, Tensor):
        raise TypeError("natural policy output must expose a tensor value field 'v'")
    spec.assert_model_width(value)
    raw = getattr(output, "raw", None)
    if isinstance(raw, Mapping):
        for name, tensor in raw.items():
            if not isinstance(tensor, Tensor) or tensor.ndim < 2:
                raise TypeError(f"natural policy output field {name!r} is not row aligned")
            if tensor.shape[1] != spec.model_width:
                raise ValueError(
                    f"natural policy output field {name!r} expanded model width "
                    f"{spec.model_width} to {tensor.shape[1]}"
                )
    return output


def _gather_inputs(partition: TrainingPartition, rows: Tensor) -> Mapping[str, Tensor]:
    return partition.gather_inputs(rows)


def _column(columns: Mapping[str, Tensor], *names: str) -> Tensor:
    for name in names:
        if name in columns:
            return columns[name]
    raise KeyError(f"required learner column is absent: {names}")


def _exact_rows(value: Tensor, phi: Tensor) -> Tensor:
    return value.index_select(-1, phi.to(value.device))


def _legal(partition: TrainingPartition, rows: Tensor) -> tuple[Tensor, Tensor]:
    ids = _column(partition.columns, "legal_event_ids", "legal_ids")[rows]
    mask = _column(partition.columns, "legal_event_mask", "legal_mask")[rows]
    if ids.ndim != 2 or mask.shape != ids.shape or mask.dtype != torch.bool:
        raise ValueError("legal event ids/mask must be aligned [B,E]")
    return ids, mask


def _policy_terms(
    output: Any,
    reference: Any | None,
    legal_ids: Tensor,
    legal_mask: Tensor,
    adapter: Any,
    phi: Tensor,
    *,
    prevalidated: bool = False,
) -> tuple[Callable[[Tensor], Tensor], Tensor, Tensor]:
    """Return selected-logp closure, exact entropy, and exact current||ref KL."""

    fused = getattr(adapter, "policy_terms", None)
    natural_log_rows: Tensor | None = None
    if callable(fused):
        natural_log_rows, natural_entropy, natural_kl = fused(
            output, reference, legal_ids, legal_mask
        )
        entropy = _exact_rows(natural_entropy, phi)
        kl = _exact_rows(natural_kl, phi)
    else:
        entropy = _exact_rows(adapter.entropy(output, legal_ids, legal_mask), phi)
        if reference is None:
            kl = torch.zeros_like(entropy)
        else:
            try:
                natural_kl = adapter.kl(output, reference, legal_ids, legal_mask)
            except TypeError:
                natural_kl = adapter.kl(output, reference)
            kl = _exact_rows(natural_kl, phi)

    def selected(events: Tensor) -> Tensor:
        if natural_log_rows is None:
            natural = adapter.log_prob(output, events, legal_ids, legal_mask)
            return _exact_rows(natural, phi)
        selected_events = events if events.ndim == 2 else events[:, None]
        matches = (
            legal_ids[:, None, :].to(selected_events.device)
            == selected_events[:, :, None]
        ) & legal_mask[:, None, :].to(selected_events.device)
        if not prevalidated:
            _require_selected = matches.any(-1)
            if _require_selected.device.type == "cuda":
                torch._assert_async(
                    _require_selected.all(),
                    "selected events must select supplied legal events",
                )
            elif not bool(_require_selected.all()):
                raise ValueError(
                    "selected events must select supplied legal events"
                )
        columns = matches.to(torch.int64).argmax(-1).to(natural_log_rows.device)
        natural = natural_log_rows.gather(
            -1,
            columns[:, None, :].expand(
                natural_log_rows.shape[0], natural_log_rows.shape[1], -1
            ),
        ).transpose(1, 2)
        return _exact_rows(natural, phi)

    return selected, entropy, kl


def _clipped_policy(logp: Tensor, old_logp: Tensor, advantage: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    ratio = torch.exp(logp - old_logp)
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
    loss = -torch.minimum(ratio * advantage, clipped * advantage)
    return loss, ratio, ratio != clipped


@dataclass(frozen=True)
class ModeLossResult:
    per_decision: Tensor
    stats: Mapping[str, Tensor]


def mode_loss(
    mode: str,
    spec: PartitionSpec,
    storage: TrainingStorage,
    partition: TrainingPartition,
    rows: Tensor,
    output: Any,
    adapter: Any,
    loss_cfg: Any,
    *,
    iteration: int = 0,
    reference: Any | None = None,
    prevalidated: bool = False,
    entropy_coefficient_override: Tensor | None = None,
) -> ModeLossResult:
    """Exact production-v1 storage-to-math map for one physical subspan."""

    rows = rows.to(dtype=torch.int64, device=next(iter(partition.columns.values())).device)
    phi = spec.phi(device=rows.device)
    legal_ids, legal_mask = _legal(partition, rows)
    c_v = float(getattr(loss_cfg, "c_v"))
    c_kl = float(getattr(loss_cfg, "c_kl", 0.0))
    eps = float(getattr(loss_cfg, "clip_eps"))
    natural_value = output.v
    entropy_coefficient = (
        _entropy_coefficient(loss_cfg, iteration)
        if entropy_coefficient_override is None
        else entropy_coefficient_override.to(natural_value)
    )
    clipped_value = getattr(loss_cfg, "value_loss", "mse") == "clipped"
    eps_v = float(getattr(loss_cfg, "clip_eps_v"))
    branching_t = natural_value.new_zeros(natural_value.shape[0])
    branching_c = natural_value.new_zeros(natural_value.shape[0])
    branching_t_raw = natural_value.new_zeros(natural_value.shape[0])
    branching_c_raw = natural_value.new_zeros(natural_value.shape[0])

    columns = partition.columns
    if mode == "normal":
        exact_type = _column(columns, "realized_type", "actor_type")[rows].long()
        model_type = phi[exact_type]
        batch = torch.arange(rows.numel(), device=rows.device)
        event = _column(columns, "event")[rows].long()[:, None]
        logp_natural = adapter.log_prob(output, event, legal_ids, legal_mask)[:, 0]
        current_logp = logp_natural[batch, model_type]
        old_logp = _column(columns, "old_log_prob", "old_logp")[rows].to(current_logp)
        target = _column(columns, "lambda_return")[rows].to(current_logp)
        old_value = _column(columns, "old_value", "old_v")[rows].to(current_logp)
        advantage = target - old_value
        policy, ratio, was_clipped = _clipped_policy(current_logp, old_logp, advantage, eps)
        current_value = natural_value[batch, model_type]
        plain_value = (current_value - target).square()
        if clipped_value:
            v_clipped = old_value + (current_value - old_value).clamp(-eps_v, eps_v)
            plain_value = torch.maximum(plain_value, (v_clipped - target).square())
        value = c_v * plain_value
        natural_entropy = adapter.entropy(output, legal_ids, legal_mask)
        selected_entropy = natural_entropy[batch, model_type]
        if reference is None:
            selected_kl = torch.zeros_like(selected_entropy)
        else:
            try:
                natural_kl = adapter.kl(
                    output, reference, legal_ids, legal_mask
                )
            except TypeError:
                natural_kl = adapter.kl(output, reference)
            selected_kl = natural_kl[batch, model_type]
        entropy_loss = -entropy_coefficient * selected_entropy
        kl_loss = c_kl * selected_kl
        per_decision = policy + value + entropy_loss + kl_loss
        weight = torch.ones_like(per_decision)
    else:
        selected_logp, entropy, reference_kl = _policy_terms(
            output,
            reference,
            legal_ids,
            legal_mask,
            adapter,
            phi,
            prevalidated=prevalidated,
        )
        exact_value = _exact_rows(natural_value, phi)

        def value_elements(target: Tensor, old_natural: Tensor) -> Tensor:
            old = _exact_rows(old_natural, phi)
            plain = (exact_value - target).square()
            if clipped_value:
                clipped = old + (exact_value - old).clamp(-eps_v, eps_v)
                plain = torch.maximum(plain, (clipped - target).square())
            return c_v * plain

    if mode == "sequential_rb":
        if storage.posterior_pool is None:
            raise ValueError("sequential RB requires its posterior pool")
        posterior_ref = _column(columns, "posterior_ref")[rows].long()
        weight = storage.posterior_pool.posterior[posterior_ref].to(exact_value)
        target = _column(columns, "lambda_return")[rows].to(exact_value)
        old_value_natural = _column(columns, "old_value", "old_v")[rows].to(exact_value)
        advantage = target - _exact_rows(old_value_natural, phi)
        event = _column(columns, "event")[rows].long()[:, None]
        logp = selected_logp(event)[:, 0]
        old_logp = _exact_rows(
            _column(columns, "old_log_prob", "old_logp")[rows].to(logp), phi
        )
        policy_element, ratio, was_clipped = _clipped_policy(logp, old_logp, advantage, eps)
        value_element = value_elements(target, old_value_natural)
        entropy_element = -entropy_coefficient * entropy
        kl_element = c_kl * reference_kl
        policy = (weight * policy_element).sum(1)
        value = (weight * value_element).sum(1)
        entropy_loss = (weight * entropy_element).sum(1)
        kl_loss = (weight * kl_element).sum(1)
        per_decision = policy + value + entropy_loss + kl_loss
    elif mode == "branched_rb":
        actions = partition.action_columns
        if actions is None:
            raise ValueError("branched RB requires action columns")
        ptr = _column(actions, "action_ptr").long()
        starts = ptr[rows]
        counts = ptr[rows + 1] - starts
        torch._assert_async(
            (counts > 0).all(),
            "every branched decision requires a sampled action",
        )
        maximum = partition.max_actions_per_decision
        if maximum is None or maximum <= 0:
            raise ValueError("branched partition omitted its immutable action bound")
        torch._assert_async(
            (counts <= maximum).all(),
            "sampled action span exceeds the immutable partition bound",
        )
        slots = torch.arange(maximum, device=rows.device)
        active = slots[None] < counts[:, None]
        padded_action_rows = starts[:, None] + slots[None]
        safe_action_rows = padded_action_rows.clamp_max(
            _column(actions, "event").numel() - 1
        )
        action_rows = safe_action_rows.flatten()
        active_rows = active.flatten()
        decision_index = (
            torch.arange(rows.numel(), device=rows.device)[:, None]
            .expand(-1, maximum)
            .flatten()
        )
        child_posterior = _column(actions, "child_posterior")[action_rows].to(exact_value)
        child_posterior = torch.where(
            active_rows[:, None], child_posterior, torch.zeros_like(child_posterior)
        )
        weight = child_posterior.reshape(rows.numel(), maximum, -1).sum(1)
        weight = weight / counts[:, None].to(weight)
        # Policy output is per decision; repeat natural rows only for this
        # physical action span, never as retained storage.
        repeated_output = _output_rows(output, decision_index)
        repeated_legal_ids = legal_ids[decision_index]
        repeated_legal_mask = legal_mask[decision_index]
        events = _column(actions, "event")[action_rows].long()[:, None]
        # A fixed-width ragged pack may index the following decision's first
        # action in an inactive tail slot.  Give those inert rows a legal event
        # from their own decision before calling the strict policy adapter;
        # their posterior and every reported ratio/clip contribution remain
        # masked below.
        safe_event = repeated_legal_ids.gather(
            1, repeated_legal_mask.to(torch.int64).argmax(1, keepdim=True)
        )
        events = torch.where(active_rows[:, None], events, safe_event)
        action_logp = _exact_rows(
            adapter.log_prob(
                repeated_output, events, repeated_legal_ids, repeated_legal_mask
            )[:, 0],
            phi,
        )
        old_action_logp = _exact_rows(
            _column(actions, "old_log_prob", "old_logp")[action_rows].to(action_logp),
            phi,
        )
        advantage = _column(actions, "advantage_hat")[action_rows].to(action_logp)
        policy_element, ratio, was_clipped = _clipped_policy(
            action_logp, old_action_logp, advantage, eps
        )
        action_policy = (child_posterior * policy_element).sum(1)
        policy = action_policy.reshape(rows.numel(), maximum).sum(1) / counts.to(
            action_policy
        )
        ratio = torch.where(active_rows[:, None], ratio, torch.ones_like(ratio))
        was_clipped = active_rows[:, None] & was_clipped
        target = _column(columns, "value_hat")[rows].to(exact_value)
        old_value_natural = _column(columns, "old_value", "old_v")[rows].to(exact_value)
        value = (weight * value_elements(target, old_value_natural)).sum(1)
        entropy_loss = (weight * (-entropy_coefficient * entropy)).sum(1)
        kl_loss = (weight * (c_kl * reference_kl)).sum(1)
        if "trace_variance_hat" in columns or "cost_hat" in columns:
            if output.t is None or output.c is None:
                raise ValueError("adaptive branching targets require model T/C heads")
            if "trace_variance_hat" not in columns or "cost_hat" not in columns:
                raise ValueError("adaptive branching target columns are incomplete")
            t_mask = _column(columns, "trace_variance_mask")[rows]
            if t_mask.dtype != torch.bool:
                raise TypeError("trace_variance_mask must be boolean")
            branching_t_raw = gamma_deviance(
                output.t.float(),
                _column(columns, "trace_variance_hat")[rows].to(output.t).float(),
                shift=float(getattr(loss_cfg, "epsilon_t")),
            )
            branching_t_raw = torch.where(
                t_mask, branching_t_raw, torch.zeros_like(branching_t_raw)
            )
            branching_c_raw = gamma_deviance(
                output.c.float(),
                _column(columns, "cost_hat")[rows].to(output.c).float(),
            )
            branching_t = float(getattr(loss_cfg, "c_t")) * branching_t_raw
            branching_c = float(getattr(loss_cfg, "c_c")) * branching_c_raw
        per_decision = policy + value + entropy_loss + kl_loss + branching_t + branching_c
    elif mode != "normal":
        raise ValueError(f"unsupported production-v1 mode {mode!r}")

    stats = MappingProxyType(
        {
            "rows": per_decision.new_ones(()) * per_decision.numel(),
            "loss_sum": per_decision.detach().sum(),
            "policy_sum": policy.detach().sum(),
            "value_sum": value.detach().sum(),
            "entropy_loss_sum": entropy_loss.detach().sum(),
            "reference_kl_sum": kl_loss.detach().sum(),
            "trace_variance_raw_sum": branching_t_raw.detach().sum(),
            "cost_raw_sum": branching_c_raw.detach().sum(),
            "trace_variance_loss_sum": branching_t.detach().sum(),
            "cost_loss_sum": branching_c.detach().sum(),
            "posterior_mass": weight.detach().sum(),
            "ratio_max": ratio.detach().max(),
            "clip_num": was_clipped.detach().to(torch.float32).sum(),
        }
    )
    return ModeLossResult(per_decision, stats)
@dataclass(frozen=True)
class DistributedLearnerConfig:
    """Measured execution inputs for one fixed-rollout update."""

    statistical_counts: Mapping[int, int]
    physical_shapes: Mapping[int, tuple[PhysicalShape, ...]]
    available_transient_bytes: int
    hard_bytes: int | None = None
    resident_base_bytes: int = 0
    reference_candidates: tuple[ReferenceCacheCandidate, ...] = ()
    reference_build_shapes: Mapping[int, tuple[PhysicalShape, ...]] = MappingProxyType({})
    sampled_reference_cache: bool = False
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        counts = {int(key): int(value) for key, value in self.statistical_counts.items()}
        if any(value <= 0 for value in counts.values()):
            raise ValueError("statistical B counts must be arbitrary positive integers")
        shapes = {int(key): tuple(value) for key, value in self.physical_shapes.items()}
        if any(not value for value in shapes.values()):
            raise ValueError("every configured stratum needs calibrated physical shapes")
        build_shapes = {
            int(key): tuple(value) for key, value in self.reference_build_shapes.items()
        }
        if any(not value for value in build_shapes.values()):
            raise ValueError("configured reference build shapes cannot be empty")
        if not set(build_shapes).issubset(shapes):
            raise ValueError("reference build shapes require a training stratum")
        if self.available_transient_bytes < 0 or self.resident_base_bytes < 0:
            raise ValueError("learner memory bounds must be nonnegative")
        object.__setattr__(self, "statistical_counts", MappingProxyType(counts))
        object.__setattr__(self, "physical_shapes", MappingProxyType(shapes))
        object.__setattr__(self, "reference_build_shapes", MappingProxyType(build_shapes))


@dataclass(frozen=True)
class DistributedUpdateResult:
    steps: tuple[OptimizerStepResult, ...]
    metrics: Mapping[str, float]


def posterior_effective_hand_counts(storage: TrainingStorage) -> Tensor:
    """Return one entropy-effective hand count per retained decision row.

    Normal sampling presents one realized private hand per row.  Sequential
    Rao--Blackwell rows reference an exact posterior in the deduplicated pool;
    repeated decision references remain repeated here because this vector is
    decision-aligned, not pool-aligned.
    """

    first_partition = next(iter(storage.partitions.values()))
    device = next(iter(first_partition.inputs.columns.values())).device
    rows = sum(partition.inputs.rows for partition in storage.partitions.values())
    if storage.mode == "normal":
        return torch.ones(rows, dtype=torch.float64, device=device)
    if storage.mode != "sequential_rb" or storage.posterior_pool is None:
        return torch.ones(rows, dtype=torch.float64, device=device)
    posterior = storage.posterior_pool.posterior
    if posterior.ndim != 2 or posterior.shape[0] == 0:
        raise ValueError("sequential posterior pool must contain fp64-compatible rows")
    pool_counts = torch.empty(
        posterior.shape[0], dtype=torch.float64, device=posterior.device
    )
    # Bound fp64 entropy workspace independently of rollout width.
    for begin in range(0, posterior.shape[0], 256):
        stop = min(begin + 256, posterior.shape[0])
        values = posterior[begin:stop].to(torch.float64)
        torch._assert_async(
            (torch.isfinite(values) & (values >= 0)).all(),
            "sequential posterior pool must be finite and nonnegative",
        )
        mass = values.sum(1)
        torch._assert_async(
            (mass > 0).all(),
            "sequential posterior pool rows must have positive mass",
        )
        probability = values / mass[:, None]
        entropy = -torch.where(
            probability > 0,
            probability * torch.log(probability),
            torch.zeros_like(probability),
        ).sum(1)
        pool_counts[begin:stop] = torch.exp(entropy)
    return torch.cat(
        [
            pool_counts[partition.columns["posterior_ref"].to(torch.int64)]
            for partition in storage.partitions.values()
        ]
    )


class DistributedFixedRolloutUpdate:
    """Bounded fixed-rollout optimizer consuming only ``TrainingStorage``."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        partitions: PartitionSet,
        adapter: Any,
        loss_cfg: Any,
        config: DistributedLearnerConfig,
        *,
        reference_model: nn.Module | None = None,
        group: Any | None = None,
        failure_agreement: DistributedFailureAgreement | None = None,
    ) -> None:
        self.model = model.module if hasattr(model, "module") else model
        self.optimizer = optimizer
        self.scaler = scaler
        self.partitions = partitions
        self.adapter = adapter
        self.loss_cfg = loss_cfg
        self.config = config
        self.reference_model = reference_model
        self.group = group
        self.failure_agreement = failure_agreement or DistributedFailureAgreement(
            next(self.model.parameters()).device,
            group=group,
        )
        self.flat_gradient = FlatGradientBuffer(self.model)
        self.index_buffer = ReusableIndexBuffer(
            max(config.statistical_counts.values()),
            next(self.model.parameters()).device,
        )
        self._planners = {
            key: PhysicalPlanner(value) for key, value in config.physical_shapes.items()
        }
        self._reference_build_planners = {
            key: PhysicalPlanner(config.reference_build_shapes.get(key, value))
            for key, value in config.physical_shapes.items()
        }

    def _forward_partition(
        self,
        spec: PartitionSpec,
        model: nn.Module,
        inputs: Mapping[str, Tensor],
    ) -> Any:
        device = next(model.parameters()).device
        with torch.autocast(
            device_type=device.type,
            enabled=self.config.amp_enabled,
            dtype=self.config.amp_dtype,
        ):
            return _forward(spec, model, inputs)

    def _cache_choice(
        self,
        candidates: Sequence[ReferenceCacheCandidate] | None = None,
    ) -> ReferenceCacheChoice:
        hard = (
            self.config.resident_base_bytes
            + self.flat_gradient.nbytes
            + self.config.available_transient_bytes
            if self.config.hard_bytes is None
            else self.config.hard_bytes
        )
        return select_reference_cache_set(
            self.config.reference_candidates if candidates is None else candidates,
            resident_base_bytes=self.config.resident_base_bytes + self.flat_gradient.nbytes,
            hard_bytes=hard,
        )

    def _build_reference_caches(
        self,
        storage: TrainingStorage,
        selected: frozenset[int],
        *,
        candidates: Sequence[ReferenceCacheCandidate] | None = None,
        cache_rows: Mapping[int, Tensor] | None = None,
    ) -> tuple[dict[int, Any], dict[str, float]]:
        if not selected:
            return {}, {"reference_cache_partitions": 0.0}
        if self.reference_model is None:
            raise ValueError("a selected reference cache requires reference_model")
        caches: dict[int, Any] = {}
        probes: dict[int, Any] = {}
        metrics = {
            "reference_cache_partitions": float(len(selected)),
            "reference_cache_build_rows": 0.0,
            "reference_cache_build_calls": 0.0,
            "reference_cache_bytes": 0.0,
            "reference_cache_build_seconds": 0.0,
        }
        try:
            device = self.index_buffer.tensor.device
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # Singleton probes establish the no-OOM output contract.  Reserve
            # every final destination before the first multirow build forward.
            predicted = {
                item.partition_id: item.resident_bytes
                for item in (
                    self.config.reference_candidates if candidates is None else candidates
                )
            }
            for partition_id in sorted(selected):
                partition = storage.partitions[partition_id]
                one = torch.zeros(1, dtype=torch.int64, device=self.index_buffer.tensor.device)
                with torch.no_grad():
                    probes[partition_id] = self._forward_partition(
                        self.partitions[partition_id],
                        self.reference_model,
                        _gather_inputs(partition, one),
                    )
                caches[partition_id] = _empty_output_like(
                    probes[partition_id],
                    partition.inputs.rows
                    if cache_rows is None
                    else cache_rows[partition_id].numel(),
                )
                actual_bytes = output_nbytes(caches[partition_id])
                if actual_bytes != predicted[partition_id]:
                    raise MemoryError(
                        "exact reference-cache destination byte contract failed"
                    )
                metrics["reference_cache_bytes"] += actual_bytes
            started = time.perf_counter()
            for partition_id in sorted(selected):
                partition = storage.partitions[partition_id]
                planner = self._reference_build_planners[partition_id]
                selected_rows = (
                    torch.arange(
                        partition.inputs.rows,
                        dtype=torch.int64,
                        device=self.index_buffer.tensor.device,
                    )
                    if cache_rows is None
                    else cache_rows[partition_id].to(torch.int64)
                )
                positions = torch.arange(
                    selected_rows.numel(),
                    dtype=torch.int32,
                    device=self.index_buffer.tensor.device,
                )
                for span in planner.plan(positions, self.config.available_transient_bytes):
                    destination_rows = span.select(positions).long()
                    rows = selected_rows[destination_rows]
                    with torch.no_grad():
                        output = self._forward_partition(
                            self.partitions[partition_id],
                            self.reference_model,
                            _gather_inputs(partition, rows),
                        )
                    _scatter_output(caches[partition_id], destination_rows, output)
                    metrics["reference_cache_build_rows"] += float(rows.numel())
                    metrics["reference_cache_build_calls"] += 1.0
                    histogram_key = f"reference_cache_build_rows_{rows.numel()}"
                    metrics[histogram_key] = metrics.get(histogram_key, 0.0) + 1.0
                    del output
            metrics["reference_cache_build_seconds"] = time.perf_counter() - started
            if device.type == "cuda":
                metrics["reference_cache_build_peak_allocated_bytes"] = float(
                    torch.cuda.max_memory_allocated(device)
                )
                metrics["reference_cache_build_peak_reserved_bytes"] = float(
                    torch.cuda.max_memory_reserved(device)
                )
            return caches, metrics
        except (MemoryError, RuntimeError):
            # Admission/build failure is a hard transaction rollback.  It
            # either precedes learner draws or reuses the already-predrawn
            # sampled plan, so targets and randomness are unchanged and the
            # mandatory empty-cache path remains valid.
            caches.clear()
            return {}, {
                "reference_cache_partitions": 0.0,
                "reference_cache_rollback": 1.0,
                "reference_cache_build_rows": 0.0,
                "reference_cache_build_calls": 0.0,
                "reference_cache_bytes": 0.0,
                "reference_cache_build_seconds": 0.0,
            }

    def __call__(
        self,
        storage: TrainingStorage,
        generator: torch.Generator,
        *,
        optimizer_steps: int = 1,
        iteration: int = 0,
        grad_clip: float = math.inf,
    ) -> DistributedUpdateResult:
        def prove_mandatory_feasibility() -> tuple[Tensor, int, dict[int, int]]:
            validate_learner_input(storage, self.partitions)
            if optimizer_steps <= 0:
                raise ValueError("optimizer_steps must be positive")
            # This tensor and every uncached singleton forward must fit before
            # any rank is allowed to launch the mass collective.
            singleton = torch.zeros(
                1, dtype=torch.int32, device=self.index_buffer.tensor.device
            )
            for partition_id, partition in storage.partitions.items():
                if partition.inputs.rows:
                    self._planners[partition_id].plan(
                        singleton, self.config.available_transient_bytes
                    )
            reference_singleton_rows = 0
            singleton_output_bytes: dict[int, int] = {}
            if float(getattr(self.loss_cfg, "c_kl", 0.0)) > 0.0:
                if self.reference_model is None:
                    raise ValueError("positive reference KL requires reference_model")
                singleton_rows = singleton.long()
                for partition_id, partition in storage.partitions.items():
                    if not partition.inputs.rows:
                        continue
                    with torch.no_grad():
                        singleton_output = self._forward_partition(
                            self.partitions[partition_id],
                            self.reference_model,
                            _gather_inputs(partition, singleton_rows),
                        )
                    reference_singleton_rows += 1
                    singleton_output_bytes[partition_id] = output_nbytes(
                        singleton_output
                    )
                    del singleton_output
            return singleton, reference_singleton_rows, singleton_output_bytes

        singleton, reference_singleton_rows, singleton_output_bytes = (
            self.failure_agreement.run(
                "mandatory learner singleton feasibility",
                prove_mandatory_feasibility,
            )
        )
        exchange = begin_log_mass_exchange(
            storage, self.partitions, group=self.group
        )
        predrawn: dict[int, Tensor] | None = None
        cache_rows: dict[int, Tensor] | None = None
        if self.config.sampled_reference_cache and singleton_output_bytes:
            # Draw the complete logical index plan using the exact original
            # step/partition call order.  A cache rollback reuses this plan, so
            # it neither rerolls the sampled iteration nor changes RNG state.
            predrawn = {
                partition_id: torch.empty(
                    (optimizer_steps, count),
                    dtype=torch.int32,
                    device=self.index_buffer.tensor.device,
                )
                for partition_id, count in self.config.statistical_counts.items()
                if storage.partitions.get(partition_id) is not None
                and storage.partitions[partition_id].inputs.rows
            }
            for step_index in range(optimizer_steps):
                for partition_spec in self.partitions.specs:
                    partition_id = partition_spec.partition_id
                    partition = storage.partitions.get(partition_id)
                    if partition is None or not partition.inputs.rows:
                        continue
                    count = self.config.statistical_counts[partition_id]
                    if storage.mode == "branched_rb":
                        drawn = self.index_buffer.draw_alias(
                            storage.sampling.alias_probability[partition_id],
                            storage.sampling.alias_index[partition_id],
                            count,
                            generator,
                        )
                    else:
                        drawn = self.index_buffer.draw_uniform(
                            partition.inputs.rows, count, generator
                        )
                    predrawn[partition_id][step_index].copy_(drawn)
            cache_rows = {
                partition_id: torch.unique(indices.flatten(), sorted=True)
                for partition_id, indices in predrawn.items()
            }
            candidates = tuple(
                ReferenceCacheCandidate(
                    partition_id=partition_id,
                    resident_bytes=(
                        rows.numel() * singleton_output_bytes[partition_id]
                    ),
                    build_workspace_bytes=max(
                        shape.transient_bytes
                        for shape in self.config.reference_build_shapes.get(
                            partition_id,
                            self.config.physical_shapes[partition_id],
                        )
                    ),
                    uncached_training_workspace_bytes=max(
                        shape.transient_bytes
                        for shape in self.config.physical_shapes[partition_id]
                    ),
                    cached_training_workspace_bytes=max(
                        shape.transient_bytes
                        for shape in self.config.physical_shapes[partition_id]
                    ),
                    build_seconds=float(rows.numel()),
                    uncached_training_seconds=float(
                        optimizer_steps * self.config.statistical_counts[partition_id]
                    ),
                    cached_training_seconds=0.0,
                )
                for partition_id, rows in cache_rows.items()
            )
            choice = self._cache_choice(candidates)
            caches, metrics = self._build_reference_caches(
                storage,
                choice.selected,
                candidates=candidates,
                cache_rows=cache_rows,
            )
        else:
            # Full-partition candidates remain available for callers with
            # externally calibrated cache contracts.
            choice = self._cache_choice()
            caches, metrics = self._build_reference_caches(storage, choice.selected)
        metrics["reference_singleton_feasibility_rows"] = float(
            reference_singleton_rows
        )
        all_logs, _ = exchange.wait()
        alphas = global_stratum_alphas(all_logs)
        rank = dist.get_rank(self.group) if dist.is_available() and dist.is_initialized() else 0
        steps: list[OptimizerStepResult] = []
        sampled_total = 0
        physical_total = 0
        tail_total = 0
        model_rows = 0
        reference_rows = int(metrics.get("reference_cache_build_rows", 0.0))
        objective_mass = torch.zeros(
            (), dtype=torch.float32, device=self.index_buffer.tensor.device
        )
        subspan_histogram: dict[int, int] = {}
        partition_sampled: dict[int, int] = {}
        partition_model_rows: dict[int, int] = {}
        partition_reference_rows: dict[int, int] = {}
        activation_checkpointed = 0
        objective_terms: dict[str, Tensor] = {}
        ratio_maximum: Tensor | None = None
        effective_by_partition: dict[int, Tensor] = {}
        effective_offset = 0
        effective_counts = posterior_effective_hand_counts(storage)
        rollout_effective_hands = effective_counts.sum()
        learner_effective_hands = effective_counts.new_zeros(())
        for partition_id, partition in storage.partitions.items():
            effective_by_partition[partition_id] = effective_counts[
                effective_offset : effective_offset + partition.inputs.rows
            ]
            effective_offset += partition.inputs.rows
        timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        cuda_device = self.index_buffer.tensor.device

        def timing_start() -> torch.cuda.Event | None:
            if cuda_device.type != "cuda":
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def timing_end(name: str, started: torch.cuda.Event | None) -> None:
            if started is None:
                return
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            timing_events.setdefault(name, []).append((started, ended))

        def accumulate_local_step(step_index: int) -> None:
            nonlocal sampled_total
            nonlocal physical_total
            nonlocal tail_total
            nonlocal model_rows
            nonlocal reference_rows
            nonlocal objective_mass
            nonlocal activation_checkpointed
            nonlocal ratio_maximum
            nonlocal learner_effective_hands
            self.flat_gradient.zero_()
            for position, partition_spec in enumerate(self.partitions.specs):
                partition_id = partition_spec.partition_id
                partition = storage.partitions.get(partition_id)
                if partition is None or partition.inputs.rows == 0:
                    continue
                count = self.config.statistical_counts[partition_id]
                if predrawn is not None:
                    indices = predrawn[partition_id][step_index]
                elif storage.mode == "branched_rb":
                    indices = self.index_buffer.draw_alias(
                        storage.sampling.alias_probability[partition_id],
                        storage.sampling.alias_index[partition_id],
                        count,
                        generator,
                    )
                else:
                    indices = self.index_buffer.draw_uniform(
                        partition.inputs.rows, count, generator
                    )
                alpha = alphas[rank, position].to(torch.float32)
                # Keep stratum mass device-resident. Converting each alpha to
                # Python here serialized every partition behind prior kernels.
                objective_mass = objective_mass + alpha.detach()
                planner = self._planners[partition_id]
                spans = planner.plan(indices, self.config.available_transient_bytes)
                sampled_total += count
                learner_effective_hands = (
                    learner_effective_hands
                    + effective_by_partition[partition_id][indices.to(torch.int64)].sum()
                )
                partition_sampled[partition_id] = partition_sampled.get(partition_id, 0) + count
                physical_total += len(spans)
                tail_total += sum(span.shape_rows not in {s.rows for s in planner.shapes} for span in spans)
                for span in spans:
                    subspan_histogram[span.shape_rows] = (
                        subspan_histogram.get(span.shape_rows, 0) + 1
                    )
                    activation_checkpointed += int(span.activation_checkpointing)
                    phase_event = timing_start()
                    rows = span.select(indices).long()
                    inputs = _gather_inputs(partition, rows)
                    timing_end("gather", phase_event)
                    reference_output = None
                    if float(getattr(self.loss_cfg, "c_kl", 0.0)) > 0.0:
                        phase_event = timing_start()
                        if self.reference_model is None:
                            raise ValueError("positive reference KL requires reference_model")
                        if partition_id in caches:
                            cache_positions = (
                                rows
                                if cache_rows is None
                                else torch.searchsorted(
                                    cache_rows[partition_id],
                                    rows.to(cache_rows[partition_id].dtype),
                                )
                            )
                            reference_output = _output_rows(
                                caches[partition_id], cache_positions
                            )
                        else:
                            with torch.no_grad():
                                reference_output = self._forward_partition(
                                    partition_spec, self.reference_model, inputs
                                )
                            reference_rows += rows.numel()
                            partition_reference_rows[partition_id] = (
                                partition_reference_rows.get(partition_id, 0)
                                + rows.numel()
                            )
                        timing_end("reference", phase_event)
                    phase_event = timing_start()
                    output = self._forward_partition(
                        partition_spec, self.model, inputs
                    )
                    timing_end("forward", phase_event)
                    model_rows += rows.numel()
                    partition_model_rows[partition_id] = (
                        partition_model_rows.get(partition_id, 0) + rows.numel()
                    )
                    phase_event = timing_start()
                    with torch.autocast(
                        device_type=self.index_buffer.tensor.device.type,
                        enabled=self.config.amp_enabled,
                        dtype=self.config.amp_dtype,
                    ):
                        result = mode_loss(
                            storage.mode,
                            partition_spec,
                            storage,
                            partition,
                            rows,
                            output,
                            self.adapter,
                            self.loss_cfg,
                            iteration=iteration,
                            reference=reference_output,
                        )
                    timing_end("loss", phase_event)
                    normalization = alpha.to(result.per_decision) / count
                    for name in (
                        "loss_sum",
                        "policy_sum",
                        "value_sum",
                        "entropy_loss_sum",
                        "reference_kl_sum",
                        "trace_variance_raw_sum",
                        "cost_raw_sum",
                        "trace_variance_loss_sum",
                        "cost_loss_sum",
                    ):
                        term = normalization * result.stats[name]
                        objective_terms[name] = objective_terms.get(name, term.new_zeros(())) + term
                    current_ratio = result.stats["ratio_max"].detach()
                    ratio_maximum = (
                        current_ratio
                        if ratio_maximum is None
                        else torch.maximum(ratio_maximum, current_ratio)
                    )
                    contribution = alpha.to(result.per_decision) / count * result.per_decision.sum()
                    phase_event = timing_start()
                    self.scaler.scale(contribution).backward()
                    timing_end("backward", phase_event)
                    del result, contribution, output, reference_output, inputs, rows

        if cuda_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cuda_device)
        for step_index in range(optimizer_steps):
            self.failure_agreement.run(
                f"local backward step {step_index}",
                lambda step_index=step_index: accumulate_local_step(step_index),
            )
            phase_event = timing_start()
            finite = sync_scaled_gradients(
                dist.get_world_size(self.group)
                if dist.is_available() and dist.is_initialized()
                else 1,
                flat_buffer=self.flat_gradient,
            )
            timing_end("gradient_sync", phase_event)
            phase_event = timing_start()
            step = self.failure_agreement.run(
                f"optimizer transition step {step_index}",
                lambda: finite_gated_optimizer_step(
                    self.model,
                    self.optimizer,
                    self.scaler,
                    gradients_finite=finite,
                    grad_clip=grad_clip,
                ),
            )
            steps.append(step)
            timing_end("optimizer_update", phase_event)
        metrics.update(
            {
                "statistical_rows": float(sampled_total),
                "physical_subspans": float(physical_total),
                "physical_tails": float(tail_total),
                "model_rows": float(model_rows),
                "reference_rows": float(reference_rows),
                "exact_objective_mass": float(objective_mass),
                "flat_gradient_bytes": float(self.flat_gradient.nbytes),
                "reference_cache_predicted_seconds": choice.predicted_seconds,
                "reference_cache_predicted_peak_bytes": float(choice.predicted_peak_bytes),
                "activation_checkpointed_subspans": float(activation_checkpointed),
                "posterior_effective_hands_rollout": float(rollout_effective_hands),
                "posterior_effective_hand_presentations_learner": float(
                    learner_effective_hands
                ),
            }
        )
        metrics["posterior_effective_hands_per_learner_sample"] = (
            metrics["posterior_effective_hand_presentations_learner"]
            / max(metrics["statistical_rows"], 1.0)
        )
        for partition_id, count in partition_sampled.items():
            metrics[f"partition_{partition_id}_statistical_rows"] = float(count)
            metrics[f"partition_{partition_id}_model_rows"] = float(
                partition_model_rows.get(partition_id, 0)
            )
            metrics[f"partition_{partition_id}_reference_rows"] = float(
                partition_reference_rows.get(partition_id, 0)
            )
        for rows, count in subspan_histogram.items():
            metrics[f"physical_subspan_rows_{rows}"] = float(count)
        metric_names = {
            "loss_sum": "loss",
            "policy_sum": "policy",
            "value_sum": "value",
            "entropy_loss_sum": "entropy_loss",
            "reference_kl_sum": "reference_kl",
            "trace_variance_raw_sum": "trace_variance_raw",
            "cost_raw_sum": "cost_raw",
            "trace_variance_loss_sum": "trace_variance_loss",
            "cost_loss_sum": "cost_loss",
        }
        for source_name, target_name in metric_names.items():
            metrics[target_name] = float(objective_terms.get(source_name, torch.zeros(())))
        metrics["ratio_max"] = (
            0.0 if ratio_maximum is None else float(ratio_maximum)
        )
        if timing_events:
            torch.cuda.synchronize(cuda_device)
            for name, events in timing_events.items():
                metrics[f"optimization_cuda_{name}_seconds"] = sum(
                    started.elapsed_time(ended) for started, ended in events
                ) / 1000.0
        if steps:
            for name, value in steps[-1].metrics().items():
                metrics[name] = float(value)
            grad_norms = [step.grad_norm_pre_clip for step in steps]
            clip_scales = [step.grad_clip_scale for step in steps]
            clipped = [scale < 1.0 - 1.0e-12 for scale in clip_scales]
            metrics.update(
                {
                    "grad_norm_pre_clip_step_mean": sum(grad_norms) / len(grad_norms),
                    "grad_norm_pre_clip_step_min": min(grad_norms),
                    "grad_norm_pre_clip_step_max": max(grad_norms),
                    "grad_clip_scale_step_mean": sum(clip_scales) / len(clip_scales),
                    "grad_clip_scale_step_min": min(clip_scales),
                    "grad_clipped_step_count": float(sum(clipped)),
                    "grad_clipped_step_fraction": sum(clipped) / len(clipped),
                    "optimizer_steps_applied_count": float(
                        sum(step.optimizer_step_applied for step in steps)
                    ),
                    "optimizer_steps_nonfinite_count": float(
                        sum(not step.gradients_finite for step in steps)
                    ),
                }
            )
        if self.index_buffer.tensor.device.type == "cuda":
            metrics["training_peak_allocated_bytes"] = float(
                torch.cuda.max_memory_allocated(self.index_buffer.tensor.device)
            )
            metrics["training_peak_reserved_bytes"] = float(
                torch.cuda.max_memory_reserved(self.index_buffer.tensor.device)
            )
        caches.clear()
        return DistributedUpdateResult(tuple(steps), MappingProxyType(metrics))
