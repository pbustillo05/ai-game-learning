"""One schema-generated layered CUDA target executor.

The implementation is vectorized by reverse topology layer.  Python schedules
only immutable layer/domain groups; it never iterates over a node or action,
and it performs no device-to-host read between layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor

from pbg.core.capacity import GeneratedOwners
from pbg.core.cuda_layer import layered_target_cuda
from pbg.core.contracts import (
    ModeSchema,
    PartitionContract,
    PrivateEstimator,
    TrainingRegime,
)
from pbg.core.storage import (
    DestinationRequest,
    DeviceFailure,
    ExactBlock,
    PublicationTransaction,
    SoAField,
    SoASchema,
    StorageError,
)
from pbg.core.topology import LayerPlan


RB_GEMM_LIBRARY_WORKSPACE_BYTES = 8 << 20
CUDA_LIBRARY_ALLOCATION_ALIGNMENT = 64 << 10


class TargetError(RuntimeError):
    """Layered target planning or execution failed."""


def rb_conversion_workspace_row_bytes(
    source_width: int, destination_width: int
) -> int:
    """Bound FP32 posterior/moment conversion without a K-by-H product."""

    if source_width <= 0 or destination_width <= 0:
        raise ValueError("RB conversion widths must be positive")
    return 4 * (3 * source_width + 2 * destination_width)


def rb_conversion_mask_bytes(
    source_width: int, destination_width: int
) -> int:
    """Resident CUDA bytes for the deterministic class-reduction matrix."""

    if source_width <= 0 or destination_width <= 0:
        raise ValueError("RB conversion widths must be positive")
    return 4 * source_width * destination_width


def rb_conversion_library_workspace_bytes(
    source_width: int, destination_width: int
) -> int:
    """Resident reduction mask plus the versioned cuBLAS workspace bound."""

    raw = (
        rb_conversion_mask_bytes(source_width, destination_width)
        + RB_GEMM_LIBRARY_WORKSPACE_BYTES
    )
    return (
        (raw + CUDA_LIBRARY_ALLOCATION_ALIGNMENT - 1)
        // CUDA_LIBRARY_ALLOCATION_ALIGNMENT
        * CUDA_LIBRARY_ALLOCATION_ALIGNMENT
    )


def _lambda(value: float) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or not 0.0 <= result <= 1.0:
        raise TargetError("lambda must be finite and lie in [0, 1]")
    return result


@dataclass(frozen=True, order=True)
class ConversionMap:
    """Deterministic component map for one declared class-sum conversion."""

    source: str
    destination: str
    destination_index: tuple[int, ...]


class TargetAlgebra:
    """Validated identity and deterministic class-sum endpoint conversions."""

    def __init__(
        self,
        partitions: PartitionContract,
        conversions: Sequence[ConversionMap] = (),
    ) -> None:
        self.partitions = partitions
        self.input_names = tuple(item.name for item in partitions.inputs)
        self.output_names = tuple(item.name for item in partitions.outputs)
        self.widths = tuple(item.width for item in partitions.outputs)
        supplied = {
            (item.source, item.destination): item for item in conversions
        }
        expected = {
            (item.source, item.destination): item
            for item in partitions.conversions
        }
        if set(supplied) != set(expected):
            raise TargetError(
                "conversion maps must cover every declared conversion exactly"
            )
        masks: dict[tuple[int, int], Tensor] = {}
        for pair, specification in expected.items():
            if specification.operation != "sum_classes":
                raise TargetError(
                    f"unsupported endpoint conversion {specification.operation!r}"
                )
            mapping = supplied[pair]
            source = self.output_names.index(mapping.source)
            destination = self.output_names.index(mapping.destination)
            if len(mapping.destination_index) != self.widths[source]:
                raise TargetError("conversion map does not cover its source width")
            if any(
                isinstance(index, bool)
                or index < 0
                or index >= self.widths[destination]
                for index in mapping.destination_index
            ):
                raise TargetError("conversion destination index is out of range")
            # Rows are destination classes and columns retain exact source
            # logical order.  Multiplication plus ordered reduction avoids
            # nondeterministic atomic scatter additions.
            mask = torch.zeros(
                (self.widths[destination], self.widths[source]),
                dtype=torch.float32,
            )
            for source_index, destination_index in enumerate(
                mapping.destination_index
            ):
                mask[destination_index, source_index] = 1.0
            masks[(source, destination)] = mask
        self._cpu_masks = MappingProxyType(masks)

    def device(self, device: torch.device | str) -> "_DeviceAlgebra":
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise TargetError("queued targets require CUDA")
        return _DeviceAlgebra(
            self.widths,
            {
                pair: mask.to(device=resolved)
                for pair, mask in self._cpu_masks.items()
            },
        )


@dataclass(frozen=True)
class _DeviceAlgebra:
    widths: tuple[int, ...]
    masks: Mapping[tuple[int, int], Tensor]

    def convert(self, value: Tensor, source: int, destination: int) -> Tensor:
        if source == destination:
            return value
        try:
            mask = self.masks[(source, destination)]
        except KeyError as exc:
            raise TargetError(
                f"unsupported output conversion {source} -> {destination}"
            ) from exc
        # GEMM avoids materializing the prohibited [..., destination, source]
        # product while preserving the fixed source-order reduction.
        return torch.matmul(value, mask.t())


@dataclass(frozen=True)
class SampledLeafWork:
    return_value: Tensor
    trace_vector: Tensor | None = None


@dataclass(frozen=True)
class RBLeafWork:
    mass: Tensor
    moment: Tensor
    trace_vector: Tensor | None = None


LeafWork = SampledLeafWork | RBLeafWork


def rb_completion(
    p_hat: Tensor,
    v_hat: Tensor,
    *,
    branched: bool,
    algebra: TargetAlgebra,
    source_output: str,
    destination_output: str | None = None,
    device_algebra: _DeviceAlgebra | None = None,
) -> RBLeafWork:
    """Create mass/moment work, converting before conditionalization."""

    if (
        p_hat.device.type != "cuda"
        or p_hat.device != v_hat.device
        or p_hat.dtype != torch.float32
        or v_hat.dtype != torch.float32
        or p_hat.shape != v_hat.shape
        or p_hat.ndim != 3
        or not p_hat.is_contiguous()
        or not v_hat.is_contiguous()
    ):
        raise TargetError(
            "RB completion must be matching CUDA FP32 [leaf, P_train, width]"
        )
    if bool((p_hat < 0).any()):
        raise TargetError("RB completion mass must be nonnegative")
    source = algebra.output_names.index(source_output)
    destination = (
        source
        if destination_output is None
        else algebra.output_names.index(destination_output)
    )
    if device_algebra is None:
        device_algebra = algebra.device(p_hat.device)
    mass = device_algebra.convert(p_hat, source, destination)
    moment = device_algebra.convert(p_hat * v_hat, source, destination)
    trace = (p_hat * v_hat).sum(-1) if branched else None
    return RBLeafWork(mass=mass, moment=moment, trace_vector=trace)


def _schema(name: str, fields: Sequence[SoAField]) -> SoASchema:
    return SoASchema(name=name, fields=tuple(fields), lifetime="layer_plan")


class DeviceLayerPlan:
    """Exact CUDA upload of an immutable CPU ``LayerPlan``."""

    def __init__(self, plan: LayerPlan, device: torch.device | str) -> None:
        self.plan = plan
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise TargetError("LayerPlan uploads require CUDA")
        node_schema, child_schema = self._schemas()
        self.nodes = ExactBlock(node_schema, plan.node_count, self.device)
        self.children = ExactBlock(child_schema, plan.edge_count, self.device)
        node_sources = {
            "replica": plan.replica,
            "child_begin": plan.child_begin,
            "child_count": plan.child_count,
            "destination_work_row": plan.destination_work_row,
            "input_type": plan.input_type,
            "output_type": plan.output_type,
            "owner_slot": plan.owner_slot,
            "nominal_degree": plan.nominal_degree,
            "target_degree": plan.target_degree,
            "sigma_head_row": plan.sigma_head_row,
            "sigma_target_row": plan.sigma_target_row,
            "action_begin": plan.action_begin,
        }
        child_sources = {
            "source_work_row": plan.child_source_work_row,
            "source_output_type": plan.child_source_output_type,
            "path_length": plan.child_path_length,
            "cost_multiplier": plan.child_cost_multiplier,
        }
        for block, sources in (
            (self.nodes, node_sources),
            (self.children, child_sources),
        ):
            for name, source in sources.items():
                block.columns[name].copy_(source)
        self._released = False

    @staticmethod
    def _schemas() -> tuple[SoASchema, SoASchema]:
        i32 = torch.int32
        return (
            _schema(
                "LayerPlanNode",
                tuple(
                    SoAField(name, i32)
                    for name in (
                        "replica",
                        "child_begin",
                        "child_count",
                        "destination_work_row",
                        "input_type",
                        "output_type",
                        "owner_slot",
                        "nominal_degree",
                        "target_degree",
                        "sigma_head_row",
                        "sigma_target_row",
                        "action_begin",
                    )
                ),
            ),
            _schema(
                "LayerPlanChild",
                (
                    SoAField("source_work_row", i32),
                    SoAField("source_output_type", i32),
                    SoAField("path_length", i32),
                    SoAField("cost_multiplier", torch.float32),
                ),
            ),
        )

    @classmethod
    def required_bytes(cls, plan: LayerPlan) -> int:
        node_schema, child_schema = cls._schemas()
        return node_schema.allocation_bytes(
            plan.node_count
        ) + child_schema.allocation_bytes(plan.edge_count)

    @property
    def allocated_bytes(self) -> int:
        if self._released:
            raise StorageError("device layer plan has been released")
        return sum(
            block.allocated_bytes
            for block in (
                self.nodes,
                self.children,
            )
        )

    def release(self) -> None:
        if self._released:
            raise StorageError("device layer plan has been released")
        for block in (self.children, self.nodes):
            block.release()
        self._released = True


@dataclass(frozen=True)
class TargetCounters:
    layer_count: int
    node_count_per_layer: tuple[int, ...]
    child_edges_read: int
    mass_moment_bytes_read: int
    mass_moment_bytes_written: int
    target_bytes_written: int
    scratch_bytes: int
    kernel_launches: int
    cuda_seconds: float


@dataclass(frozen=True)
class TargetResult:
    published_lanes: tuple[int, ...]
    failed_lanes: tuple[int, ...]
    counters: TargetCounters


class LayeredTargetExecutor:
    """The sole target executor generated from one immutable ``ModeSchema``."""

    def __init__(
        self,
        mode: ModeSchema,
        algebra: TargetAlgebra,
        *,
        lambda_: float,
    ) -> None:
        self.mode = mode
        self.algebra = algebra
        self.lambda_ = _lambda(lambda_)
        self.sampled = mode.private_estimator is PrivateEstimator.SAMPLED
        self.branched = mode.rollout_mode.is_branched

    def required_scratch_bytes(
        self, plan: LayerPlan, n_replica: int
    ) -> int:
        output_count = len(plan.output_names)
        input_count = len(plan.input_names)
        action_count = output_count if self.branched else input_count
        max_width = max(self.algebra.widths)
        pointer_entries = (
            5 * output_count
            + 3 * input_count
            + 3 * n_replica * input_count
            + 2 * n_replica * action_count
        )
        scalar_entries = 2 * output_count + input_count + n_replica
        transient_bytes = (
            pointer_entries * 8
            + scalar_entries * 4
            + output_count * output_count * max_width * 4
        )
        return DeviceLayerPlan.required_bytes(plan) + transient_bytes

    def execute(
        self,
        owners: GeneratedOwners,
        plan: LayerPlan,
    ) -> TargetResult:
        if owners.plan.mode != self.mode:
            raise TargetError("executor mode differs from generated owners")
        if owners.plan.partitions != self.algebra.partitions:
            raise TargetError("target algebra differs from generated owners")
        if plan.input_names != self.algebra.input_names:
            raise TargetError("LayerPlan inputs differ from target algebra")
        if plan.output_names != self.algebra.output_names:
            raise TargetError("LayerPlan outputs differ from target algebra")
        if plan.output_widths != self.algebra.widths:
            raise TargetError("LayerPlan widths differ from target algebra")
        if plan.branched != self.branched:
            raise TargetError("LayerPlan rollout family differs from executor")
        if plan.supported_conversions != tuple(
            sorted(self.algebra._cpu_masks)
        ):
            raise TargetError("LayerPlan conversions differ from target algebra")
        device = owners.plan.device
        n_replica = owners.plan.manifest.n_replica
        p_train = (
            owners.plan.layout.player_count
            if self.mode.training_regime is TrainingRegime.SELF_PLAY
            else 1
        )
        if len(plan.arena_rows) != n_replica:
            raise TargetError("LayerPlan replica count differs from owners")
        plan.assert_admitted(owners.plan.manifest)
        input_output = torch.tensor(
            [
                plan.output_names.index(
                    self.algebra.partitions.output_for(name).name
                )
                for name in plan.input_names
            ],
            dtype=torch.int32,
        )
        trained = plan.input_type >= 0
        if bool(
            (
                trained
                & (
                    plan.output_type
                    != input_output[plan.input_type.clamp_min(0)]
                )
            ).any()
        ):
            raise TargetError("LayerPlan input/output binding is invalid")
        if bool(
            (
                (plan.owner_slot >= p_train)
                | (plan.owner_slot < -1)
            ).any()
        ):
            raise TargetError("LayerPlan owner slot is outside P_train")
        max_width = max(self.algebra.widths)
        required_scratch = self.required_scratch_bytes(plan, n_replica)
        if required_scratch > owners.plan.manifest.movement_scratch_bytes:
            raise TargetError(
                "layer executor scratch exceeds admitted movement scratch bytes"
            )
        for output, name in enumerate(plan.output_names):
            slab = owners.slab("LayerWork", name)
            for lane in range(n_replica):
                if slab.tail(lane) < plan.arena_rows[lane][output]:
                    raise TargetError("LayerPlan addresses unpublished work rows")
        for input_type, name in enumerate(plan.input_names):
            slab = owners.slab("SigmaHead", name)
            for lane in range(n_replica):
                rows = plan.replica == lane
                rows &= plan.input_type == input_type
                if bool(rows.any()):
                    maximum = int(plan.sigma_head_row[rows].max())
                    if maximum >= slab.tail(lane):
                        raise TargetError(
                            "LayerPlan addresses an unpublished sigma head"
                        )
        device_plan = DeviceLayerPlan(plan, device)
        if device_plan.allocated_bytes != DeviceLayerPlan.required_bytes(plan):
            device_plan.release()
            raise TargetError("LayerPlan device bytes differ from exact preflight")
        transactions: dict[int, PublicationTransaction] = {}
        sigma_views: dict[tuple[int, int, str], Tensor] = {}
        action_views: dict[tuple[int, int, str], Tensor] = {}
        try:
            for lane in range(n_replica):
                requests: list[DestinationRequest] = []
                tags: list[tuple[str, int]] = []
                for input_type, name in enumerate(plan.input_names):
                    count = plan.sigma_rows[lane][input_type]
                    if count:
                        requests.append(
                            DestinationRequest(
                                owners.slab("SigmaTarget", name),
                                lane,
                                count,
                                name,
                                input_type,
                            )
                        )
                        tags.append(("sigma", input_type))
                action_names = (
                    plan.output_names if self.branched else plan.input_names
                )
                for action_type, name in enumerate(action_names):
                    count = plan.action_rows[lane][action_type]
                    if count:
                        requests.append(
                            DestinationRequest(
                                owners.slab("ActionTarget", name),
                                lane,
                                count,
                                name,
                                len(plan.input_names) + action_type,
                            )
                        )
                        tags.append(("action", action_type))
                if not requests:
                    continue
                transaction = PublicationTransaction.begin(
                    requests, phase="layered_targets"
                )
                transactions[lane] = transaction
                for index, (kind, partition) in enumerate(tags):
                    reservation = transaction.reservations[index]
                    transaction.fill_defaults(index)
                    for schema_field in reservation._slab.schema.fields:
                        field = schema_field.name
                        key = (lane, partition, field)
                        if kind == "sigma":
                            sigma_views[key] = reservation.view(field)
                        else:
                            action_views[key] = reservation.view(field)

            empty = torch.empty(0, dtype=torch.float32, device=device)
            output_slabs = [
                owners.slab("LayerWork", name) for name in plan.output_names
            ]
            work0 = [
                slab.columns["return_value" if self.sampled else "mass"]
                for slab in output_slabs
            ]
            work1 = (
                [empty for _ in output_slabs]
                if self.sampled
                else [slab.columns["moment"] for slab in output_slabs]
            )
            traces = (
                [slab.columns["trace_vector"] for slab in output_slabs]
                if self.branched
                else [empty for _ in output_slabs]
            )
            costs = (
                [slab.columns["cost"] for slab in output_slabs]
                if self.branched
                else [empty for _ in output_slabs]
            )
            deltas = (
                [slab.columns["delta"] for slab in output_slabs]
                if self.branched
                else [empty for _ in output_slabs]
            )
            head_slabs = [
                owners.slab("SigmaHead", name) for name in plan.input_names
            ]
            head_v = [slab.columns["old_value"] for slab in head_slabs]
            head_t = [
                slab.columns["old_trace"] if self.branched else empty
                for slab in head_slabs
            ]
            head_c = [
                slab.columns["old_cost"] if self.branched else empty
                for slab in head_slabs
            ]
            sigma_v: list[Tensor] = []
            sigma_t: list[Tensor] = []
            sigma_c: list[Tensor] = []
            for lane in range(n_replica):
                for input_type in range(len(plan.input_names)):
                    sigma_v.append(
                        sigma_views.get((lane, input_type, "value_target"), empty)
                    )
                    sigma_t.append(
                        sigma_views.get((lane, input_type, "trace_target"), empty)
                    )
                    sigma_c.append(
                        sigma_views.get((lane, input_type, "cost_target"), empty)
                    )
            action_names = (
                plan.output_names if self.branched else plan.input_names
            )
            action_p: list[Tensor] = []
            action_a: list[Tensor] = []
            for lane in range(n_replica):
                for action_type in range(len(action_names)):
                    action_p.append(
                        action_views.get((lane, action_type, "posterior"), empty)
                    )
                    action_a.append(
                        action_views.get(
                            (lane, action_type, "advantage_target"), empty
                        )
                    )
            conversion = torch.full(
                (
                    len(plan.output_names),
                    len(plan.output_names),
                    max_width,
                ),
                -1,
                dtype=torch.int32,
            )
            for source in range(len(plan.output_names)):
                for destination in range(len(plan.output_names)):
                    if source == destination:
                        conversion[source, destination, : self.algebra.widths[source]] = (
                            torch.arange(
                                self.algebra.widths[source], dtype=torch.int32
                            )
                        )
                    elif (source, destination) in self.algebra._cpu_masks:
                        mask = self.algebra._cpu_masks[(source, destination)]
                        for component in range(self.algebra.widths[source]):
                            selected = torch.nonzero(
                                mask[:, component], as_tuple=False
                            )
                            if selected.numel():
                                conversion[source, destination, component] = selected[0, 0]
            conversion = conversion.flatten().to(device=device)
            extension = layered_target_cuda()
            start_events = [
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            ]
            start_events[0].record()
            failure_min = extension.run_layers(
                plan.layer_ptr,
                list(device_plan.nodes.columns.values()),
                list(device_plan.children.columns.values()),
                work0,
                work1,
                traces,
                costs,
                deltas,
                head_v,
                head_t,
                head_c,
                sigma_v,
                sigma_t,
                sigma_c,
                action_p,
                action_a,
                conversion,
                [slab.capacity for slab in output_slabs],
                [slab.capacity for slab in head_slabs],
                list(self.algebra.widths),
                n_replica,
                len(plan.input_names),
                len(plan.output_names),
                len(action_names),
                p_train,
                max_width,
                self.lambda_,
                self.sampled,
                self.branched,
            )
            start_events[1].record()
            # The sole target boundary synchronization validates and publishes;
            # no layer above read a device scalar or synchronized the stream.
            start_events[1].synchronize()
            elapsed = start_events[0].elapsed_time(start_events[1]) / 1000.0
            failed_lanes: list[int] = []
            published_lanes: list[int] = []
            for lane in range(n_replica):
                logical = int(failure_min[lane].item())
                failed = logical != torch.iinfo(torch.int32).max
                if failed:
                    failed_lanes.append(lane)
                    owners.failure_state.record(
                        DeviceFailure(
                            lane=lane,
                            status=1,
                            phase="targets",
                            failure_code="invalid_target",
                            input_type="layer",
                            logical_address=logical,
                            requested_extent=1,
                            available_extent=0,
                        )
                    )
                    if lane in transactions:
                        transactions[lane].abort()
                else:
                    published_lanes.append(lane)
                    if lane in transactions:
                        transactions[lane].publish()
            rb_read_bytes = 0
            rb_written_bytes = 0
            if not self.sampled:
                rb_read_bytes = 8 * (
                    p_train * plan.rb_first_source_components
                    + plan.rb_owner_source_components
                )
                rb_written_bytes = (
                    8 * p_train * plan.rb_written_components
                )
            result = TargetResult(
                published_lanes=tuple(published_lanes),
                failed_lanes=tuple(failed_lanes),
                counters=TargetCounters(
                    layer_count=plan.layer_count,
                    node_count_per_layer=tuple(
                        int(plan.layer_ptr[layer + 1] - plan.layer_ptr[layer])
                        for layer in range(plan.layer_count)
                    ),
                    child_edges_read=plan.target_edge_count,
                    mass_moment_bytes_read=rb_read_bytes,
                    mass_moment_bytes_written=rb_written_bytes,
                    target_bytes_written=sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in (*sigma_v, *sigma_t, *sigma_c, *action_p, *action_a)
                    ),
                    scratch_bytes=required_scratch,
                    kernel_launches=plan.layer_count,
                    cuda_seconds=elapsed,
                ),
            )
        except BaseException:
            for transaction in transactions.values():
                transaction.abort()
            raise
        finally:
            device_plan.release()
        return result

__all__ = [
    "ConversionMap",
    "DeviceLayerPlan",
    "LayeredTargetExecutor",
    "RBLeafWork",
    "SampledLeafWork",
    "TargetAlgebra",
    "TargetCounters",
    "TargetError",
    "TargetResult",
    "rb_completion",
]
