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

from pbg.queued.capacity import GeneratedOwners
from pbg.queued.cuda_layer import layered_target_cuda
from pbg.queued.contracts import (
    ModeSchema,
    PartitionContract,
    PrivateEstimator,
    TrainingRegime,
)
from pbg.queued.storage import (
    DestinationRequest,
    DeviceFailure,
    ExactBlock,
    PublicationTransaction,
    SoAField,
    SoASchema,
    StorageError,
)
from pbg.queued.topology import LayerPlan


class TargetError(RuntimeError):
    """Layered target planning or execution failed."""


def _lambda(value: float) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or not 0.0 <= result <= 1.0:
        raise TargetError("lambda must be finite and lie in [0, 1]")
    return result


def conditionalize(moment: Tensor, mass: Tensor) -> Tensor:
    """Return an FP32 conditional value with exact zeros at zero mass."""

    if moment.dtype != torch.float32 or mass.dtype != torch.float32:
        raise TargetError("target conditionalization requires FP32")
    valid = mass > 0
    safe = torch.where(valid, mass, torch.ones_like(mass))
    return torch.where(valid, moment / safe, torch.zeros_like(moment))


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
        # [..., source] -> [..., destination], summing source order exactly.
        return (value.unsqueeze(-2) * mask).sum(dim=-1)


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


def sampled_completion(
    payoff: Tensor, *, branched: bool
) -> SampledLeafWork:
    """Create sampled leaf work from selected FP32 payoff components."""

    if payoff.dtype != torch.float32 or payoff.ndim != 2:
        raise TargetError("sampled payoff must be FP32 [leaf, P_train]")
    return SampledLeafWork(
        return_value=payoff.unsqueeze(-1),
        trace_vector=payoff if branched else None,
    )


def rb_completion(
    p_hat: Tensor,
    v_hat: Tensor,
    *,
    branched: bool,
    algebra: TargetAlgebra,
    source_output: str,
    destination_output: str | None = None,
) -> RBLeafWork:
    """Create mass/moment work, converting before conditionalization."""

    if (
        p_hat.dtype != torch.float32
        or v_hat.dtype != torch.float32
        or p_hat.shape != v_hat.shape
        or p_hat.ndim != 3
    ):
        raise TargetError(
            "RB completion must be matching FP32 [leaf, P_train, width]"
        )
    if bool((p_hat < 0).any()):
        raise TargetError("RB completion mass must be nonnegative")
    source = algebra.output_names.index(source_output)
    destination = (
        source
        if destination_output is None
        else algebra.output_names.index(destination_output)
    )
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
        i32 = torch.int32
        f32 = torch.float32
        self.nodes = ExactBlock(
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
            plan.node_count,
            self.device,
        )
        self.children = ExactBlock(
            _schema(
                "LayerPlanChild",
                (
                    SoAField("source_work_row", i32),
                    SoAField("source_output_type", i32),
                    SoAField("path_length", i32),
                    SoAField("cost_multiplier", f32),
                ),
            ),
            plan.edge_count,
            self.device,
        )
        self.carries = ExactBlock(
            _schema(
                "LayerPlanCarry",
                (
                    SoAField("replica", i32),
                    SoAField("source_work_row", i32),
                    SoAField("destination_work_row", i32),
                    SoAField("output_type", i32),
                ),
            ),
            plan.carry_count,
            self.device,
        )
        self.movement = ExactBlock(
            _schema(
                "LayerPlanMovement",
                (
                    SoAField("replica", i32),
                    SoAField("source_leaf_row", i32),
                    SoAField("destination_work_row", i32),
                    SoAField("output_type", i32),
                ),
            ),
            plan.movement_count,
            self.device,
        )
        self.pointers = ExactBlock(
            _schema(
                "LayerPlanPointer",
                (
                    SoAField("layer_ptr", i32, (plan.layer_count + 1,)),
                    SoAField("carry_ptr", i32, (plan.layer_count + 1,)),
                ),
            ),
            1,
            self.device,
        )
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
        carry_sources = {
            "replica": plan.carry_replica,
            "source_work_row": plan.carry_source_work_row,
            "destination_work_row": plan.carry_destination_work_row,
            "output_type": plan.carry_output_type,
        }
        movement_sources = {
            "replica": plan.movement_replica,
            "source_leaf_row": plan.movement_source_leaf_row,
            "destination_work_row": plan.movement_destination_work_row,
            "output_type": plan.movement_output_type,
        }
        for block, sources in (
            (self.nodes, node_sources),
            (self.children, child_sources),
            (self.carries, carry_sources),
            (self.movement, movement_sources),
        ):
            for name, source in sources.items():
                block.columns[name].copy_(source.to(device=self.device))
        self.pointers.columns["layer_ptr"][0].copy_(
            plan.layer_ptr.to(device=self.device)
        )
        self.pointers.columns["carry_ptr"][0].copy_(
            plan.carry_ptr.to(device=self.device)
        )
        self._released = False

    @property
    def allocated_bytes(self) -> int:
        if self._released:
            raise StorageError("device layer plan has been released")
        return sum(
            block.allocated_bytes
            for block in (
                self.nodes,
                self.children,
                self.carries,
                self.movement,
                self.pointers,
            )
        )

    def release(self) -> None:
        if self._released:
            raise StorageError("device layer plan has been released")
        for block in (
            self.pointers,
            self.movement,
            self.carries,
            self.children,
            self.nodes,
        ):
            block.release()
        self._released = True


def _zero_values(slab, count: int) -> dict[str, Tensor]:
    return {
        field.name: torch.zeros(
            (count, *field.shape),
            dtype=field.dtype,
            device=slab.device,
        )
        for field in slab.schema.fields
    }


def initialize_work_arenas(
    owners: GeneratedOwners,
    plan: LayerPlan,
    completion: Mapping[tuple[int, str], LeafWork],
) -> None:
    """Apply the exact leaf movement plan and atomically publish work arenas."""

    n_replica = owners.plan.manifest.n_replica
    if len(plan.arena_rows) != n_replica:
        raise TargetError("LayerPlan replica count differs from generated owners")
    for lane in range(n_replica):
        requests: list[DestinationRequest] = []
        values: list[dict[str, Tensor]] = []
        for output, name in enumerate(plan.output_names):
            count = plan.arena_rows[lane][output]
            if count == 0:
                continue
            slab = owners.slab("LayerWork", name)
            requests.append(
                DestinationRequest(
                    slab=slab,
                    lane=lane,
                    count=count,
                    input_type=name,
                    logical_address=output,
                )
            )
            current = _zero_values(slab, count)
            selected = (
                (plan.movement_replica == lane)
                & (plan.movement_output_type == output)
            )
            destinations = plan.movement_destination_work_row[selected].long()
            sources = plan.movement_source_leaf_row[selected].long()
            if destinations.numel():
                try:
                    payload = completion[(lane, name)]
                except KeyError as exc:
                    raise TargetError(
                        f"missing completion payload for lane {lane}, {name}"
                    ) from exc
                if isinstance(payload, SampledLeafWork):
                    if "return_value" not in current:
                        raise TargetError("sampled completion used with RB work schema")
                    current["return_value"][destinations, 0] = (
                        payload.return_value[sources]
                    )
                    if "trace_vector" in current:
                        if payload.trace_vector is None:
                            raise TargetError("branched completion lacks trace vector")
                        current["trace_vector"][destinations, 0] = (
                            payload.trace_vector[sources]
                        )
                else:
                    if "mass" not in current:
                        raise TargetError("RB completion used with sampled work schema")
                    current["mass"][destinations, 0] = payload.mass[sources]
                    current["moment"][destinations, 0] = payload.moment[sources]
                    if "trace_vector" in current:
                        if payload.trace_vector is None:
                            raise TargetError("branched completion lacks trace vector")
                        current["trace_vector"][destinations, 0] = (
                            payload.trace_vector[sources]
                        )
            values.append(current)
        if not requests:
            continue
        transaction = PublicationTransaction.begin(
            requests, phase="completion_movement"
        )
        try:
            for index, current in enumerate(values):
                transaction.fill(index, current)
            transaction.publish()
        except BaseException:
            transaction.abort()
            raise


@dataclass(frozen=True)
class TargetCounters:
    layer_count: int
    node_count_per_layer: tuple[int, ...]
    carry_count: int
    carry_payload_bytes: int
    child_edges_read: int
    mass_moment_bytes_read: int
    mass_moment_bytes_written: int
    target_bytes_written: int
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

    def execute(
        self,
        owners: GeneratedOwners,
        plan: LayerPlan,
    ) -> TargetResult:
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
        device_plan = DeviceLayerPlan(plan, device)
        if device_plan.allocated_bytes > owners.plan.manifest.movement_scratch_bytes:
            device_plan.release()
            raise TargetError("exact LayerPlan upload exceeds admitted scratch bytes")
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
                    zero = _zero_values(reservation._slab, reservation.rows)
                    transaction.fill(index, zero)
                    for field in zero:
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
            max_width = max(self.algebra.widths)
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
                [
                    plan.output_names.index(
                        self.algebra.partitions.output_for(name).name
                    )
                    for name in plan.input_names
                ],
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
            result = TargetResult(
                published_lanes=tuple(published_lanes),
                failed_lanes=tuple(failed_lanes),
                counters=TargetCounters(
                    layer_count=plan.layer_count,
                    node_count_per_layer=tuple(
                        int(plan.layer_ptr[layer + 1] - plan.layer_ptr[layer])
                        for layer in range(plan.layer_count)
                    ),
                    carry_count=plan.carry_count,
                    carry_payload_bytes=0,
                    child_edges_read=plan.edge_count,
                    mass_moment_bytes_read=(
                        0
                        if self.sampled
                        else plan.edge_count * p_train * max_width * 8
                    ),
                    mass_moment_bytes_written=(
                        0
                        if self.sampled
                        else plan.node_count * p_train * max_width * 8
                    ),
                    target_bytes_written=sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in (*sigma_v, *sigma_t, *sigma_c, *action_p, *action_a)
                    ),
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
    "conditionalize",
    "initialize_work_arenas",
    "rb_completion",
    "sampled_completion",
]
