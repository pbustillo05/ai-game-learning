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
from pbg.queued.topology import LayerGroup, LayerPlan


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


def _direct_sibling_sum(value: Tensor, target_mask: Tensor) -> Tensor:
    """Sum siblings directly in logical order, excluding each target child."""

    count = value.shape[1]
    ordinal = torch.arange(count, device=value.device)
    sibling = (
        target_mask[:, None, :]
        & (ordinal[None, :, None] != ordinal[None, None, :])
    )
    shape = sibling.shape + (1,) * (value.ndim - 2)
    return (value[:, None, ...] * sibling.reshape(shape)).sum(dim=2)


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
        algebra = self.algebra.device(device)
        nodes = device_plan.nodes.columns
        children = device_plan.children.columns

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

            lane_failed = torch.zeros(
                n_replica, dtype=torch.bool, device=device
            )
            node_invalid = torch.zeros(
                plan.node_count, dtype=torch.bool, device=device
            )
            carry_payload_bytes = 0
            mass_read = 0
            mass_written = 0
            target_written = 0
            # One logical executor launch is enqueued per reverse layer.  Its
            # immutable partition groups are vectorized subspans, not
            # node/action launches.
            kernel_launches = plan.layer_count
            start_events = [
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            ]
            start_events[0].record()

            for layer in range(plan.layer_count):
                src_arena = layer & 1
                dst_arena = (layer + 1) & 1
                for group in (
                    item for item in plan.groups if item.layer == layer
                ):
                    invalid, written, mm_read, mm_written = self._execute_group(
                        owners,
                        plan,
                        device_plan,
                        algebra,
                        group,
                        src_arena,
                        dst_arena,
                        p_train,
                        lane_failed,
                        sigma_views,
                        action_views,
                    )
                    node_invalid[group.begin : group.end] = invalid
                    target_written += written
                    mass_read += mm_read
                    mass_written += mm_written
                carry_payload_bytes += self._execute_carries(
                    owners,
                    plan,
                    device_plan,
                    layer,
                    src_arena,
                    dst_arena,
                    lane_failed,
                )

            start_events[1].record()
            # The sole target boundary synchronization validates and publishes;
            # no layer above read a device scalar or synchronized the stream.
            start_events[1].synchronize()
            elapsed = start_events[0].elapsed_time(start_events[1]) / 1000.0
            failed_lanes: list[int] = []
            published_lanes: list[int] = []
            for lane in range(n_replica):
                failed = bool(lane_failed[lane].item())
                if failed:
                    failed_lanes.append(lane)
                    logical = self._minimum_failure_row(
                        device_plan.nodes.columns["replica"],
                        node_invalid,
                        lane,
                    )
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
                    carry_payload_bytes=carry_payload_bytes,
                    child_edges_read=plan.edge_count,
                    mass_moment_bytes_read=mass_read,
                    mass_moment_bytes_written=mass_written,
                    target_bytes_written=target_written,
                    kernel_launches=kernel_launches,
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

    @staticmethod
    def _minimum_failure_row(
        replica: Tensor, invalid: Tensor, lane: int
    ) -> int:
        lane_mask = replica == lane
        candidates = torch.nonzero(
            invalid & lane_mask, as_tuple=False
        ).flatten()
        return int(candidates[0].item()) if candidates.numel() else 0

    def _execute_group(
        self,
        owners: GeneratedOwners,
        plan: LayerPlan,
        device_plan: DeviceLayerPlan,
        algebra: _DeviceAlgebra,
        group: LayerGroup,
        src_arena: int,
        dst_arena: int,
        p_train: int,
        lane_failed: Tensor,
        sigma_views: Mapping[tuple[int, int, str], Tensor],
        action_views: Mapping[tuple[int, int, str], Tensor],
    ) -> tuple[Tensor, int, int, int]:
        nodes = device_plan.nodes.columns
        children = device_plan.children.columns
        sl = slice(group.begin, group.end)
        replica = nodes["replica"][sl].long()
        count = nodes["child_count"][sl].long()
        target_count = nodes["target_degree"][sl].long()
        destination = nodes["destination_work_row"][sl].long()
        begin = nodes["child_begin"][sl].long()
        owner_slot = nodes["owner_slot"][sl].long()
        batch = replica.numel()
        n_max = max(1, owners.plan.manifest.beta_max)
        ordinal = torch.arange(n_max, device=replica.device)
        edge = begin[:, None] + ordinal[None, :]
        valid = ordinal[None, :] < count[:, None]
        target = ordinal[None, :] < target_count[:, None]
        safe_edge = torch.where(valid, edge, torch.zeros_like(edge))
        encoded_source_row = children["source_work_row"][safe_edge].long()
        source_type = children["source_output_type"][safe_edge].long()
        width = algebra.widths[group.output_type]
        payload_shape = (batch, n_max, p_train, width)
        if self.sampled:
            child_primary = torch.zeros(
                payload_shape, dtype=torch.float32, device=replica.device
            )
            child_secondary = None
        else:
            child_primary = torch.zeros(
                payload_shape, dtype=torch.float32, device=replica.device
            )
            child_secondary = torch.zeros_like(child_primary)
        child_trace = (
            torch.zeros(
                (batch, n_max, p_train),
                dtype=torch.float32,
                device=replica.device,
            )
            if self.branched
            else None
        )
        child_cost = (
            torch.zeros(
                (batch, n_max), dtype=torch.float32, device=replica.device
            )
            if self.branched
            else None
        )
        child_delta = torch.zeros_like(child_cost) if self.branched else None

        for source, source_name in enumerate(plan.output_names):
            work = owners.slab("LayerWork", source_name).columns
            for source_arena in (0, 1):
                arena_mask = (
                    encoded_source_row >= 0
                    if source_arena == 0
                    else encoded_source_row < 0
                )
                mask = valid & (source_type == source) & arena_mask
                decoded_row = torch.where(
                    encoded_source_row >= 0,
                    encoded_source_row,
                    -encoded_source_row - 1,
                )
                safe_row = torch.where(
                    mask, decoded_row, torch.zeros_like(decoded_row)
                )
                if self.sampled:
                    gathered = work["return_value"][
                        replica[:, None], safe_row, source_arena
                    ]
                    converted = algebra.convert(
                        gathered, source, group.output_type
                    )
                    child_primary = torch.where(
                        mask[..., None, None], converted, child_primary
                    )
                else:
                    gathered_mass = work["mass"][
                        replica[:, None], safe_row, source_arena
                    ]
                    gathered_moment = work["moment"][
                        replica[:, None], safe_row, source_arena
                    ]
                    converted_mass = algebra.convert(
                        gathered_mass, source, group.output_type
                    )
                    converted_moment = algebra.convert(
                        gathered_moment, source, group.output_type
                    )
                    child_primary = torch.where(
                        mask[..., None, None], converted_mass, child_primary
                    )
                    assert child_secondary is not None
                    child_secondary = torch.where(
                        mask[..., None, None],
                        converted_moment,
                        child_secondary,
                    )
                if self.branched:
                    assert child_trace is not None
                    assert child_cost is not None and child_delta is not None
                    child_trace = torch.where(
                        mask[..., None],
                        work["trace_vector"][
                            replica[:, None], safe_row, source_arena
                        ],
                        child_trace,
                    )
                    child_cost = torch.where(
                        mask,
                        work["cost"][
                            replica[:, None], safe_row, source_arena
                        ],
                        child_cost,
                    )
                    child_delta = torch.where(
                        mask,
                        work["delta"][
                            replica[:, None], safe_row, source_arena
                        ],
                        child_delta,
                    )

        divisor = target_count.to(torch.float32).reshape(batch, 1, 1)
        target_payload_mask = target[..., None, None]
        parent_primary = (
            child_primary * target_payload_mask
        ).sum(dim=1) / divisor
        if child_secondary is not None:
            parent_secondary = (
                child_secondary * target_payload_mask
            ).sum(dim=1) / divisor
        else:
            parent_secondary = None
        trained = owner_slot >= 0
        safe_owner = owner_slot.clamp_min(0)
        row_index = torch.arange(batch, device=replica.device)
        active = ~lane_failed[replica]
        target_bytes = 0
        target_finite = torch.ones(
            batch, dtype=torch.bool, device=replica.device
        )

        if group.input_type >= 0:
            old_value = owners.slab(
                "SigmaHead", plan.input_names[group.input_type]
            ).columns["old_value"][
                replica, nodes["sigma_head_row"][sl].long()
            ]
            if self.sampled:
                q = child_primary[
                    row_index[:, None],
                    ordinal[None, :],
                    safe_owner[:, None],
                ]
                sibling_sum = _direct_sibling_sum(
                    child_primary[
                        row_index[:, None],
                        ordinal[None, :],
                        safe_owner[:, None],
                    ],
                    target,
                )
                sibling_mean = sibling_sum / (
                    (target_count - 1).clamp_min(1).to(torch.float32)[
                        :, None, None
                    ]
                )
                baseline = torch.where(
                    (target_count > 1)[:, None, None],
                    self.lambda_ * sibling_mean
                    + (1.0 - self.lambda_) * old_value[:, None],
                    old_value[:, None],
                )
                advantage = q - baseline
                sigma_value = parent_primary[row_index, safe_owner]
                propagated = parent_primary.clone()
                propagated[row_index, safe_owner] = (
                    self.lambda_ * sigma_value
                    + (1.0 - self.lambda_) * old_value
                )
                parent_primary = propagated
                posterior = None
            else:
                assert parent_secondary is not None
                assert child_secondary is not None
                child_mass = child_primary[
                    row_index[:, None],
                    ordinal[None, :],
                    safe_owner[:, None],
                ]
                child_moment = child_secondary[
                    row_index[:, None],
                    ordinal[None, :],
                    safe_owner[:, None],
                ]
                q = conditionalize(child_moment, child_mass)
                sibling_mass = _direct_sibling_sum(child_mass, target)
                sibling_moment = _direct_sibling_sum(child_moment, target)
                sibling_value = conditionalize(sibling_moment, sibling_mass)
                baseline = torch.where(
                    (target_count > 1)[:, None, None],
                    self.lambda_ * sibling_value
                    + (1.0 - self.lambda_) * old_value[:, None],
                    old_value[:, None],
                )
                advantage = q - baseline
                sigma_value = conditionalize(
                    parent_secondary[row_index, safe_owner],
                    parent_primary[row_index, safe_owner],
                )
                propagated = parent_secondary.clone()
                propagated[row_index, safe_owner] = (
                    self.lambda_ * parent_secondary[row_index, safe_owner]
                    + (1.0 - self.lambda_)
                    * parent_primary[row_index, safe_owner]
                    * old_value
                )
                parent_secondary = propagated
                posterior = child_mass
            target_finite &= torch.isfinite(sigma_value).flatten(1).all(1)
            target_finite &= torch.where(
                target[..., None],
                torch.isfinite(advantage),
                torch.ones_like(advantage, dtype=torch.bool),
            ).flatten(1).all(1)
            if posterior is not None:
                target_finite &= torch.where(
                    target[..., None],
                    torch.isfinite(posterior) & (posterior >= 0),
                    torch.ones_like(posterior, dtype=torch.bool),
                ).flatten(1).all(1)

            action_partition = (
                group.output_type if self.branched else group.input_type
            )
            action_rows = nodes["action_begin"][sl].long()[:, None] + ordinal
            flat_valid = target & trained[:, None] & active[:, None]
            for lane in range(owners.plan.manifest.n_replica):
                lane_rows = flat_valid & (replica == lane)[:, None]
                rows = action_rows[lane_rows]
                if rows.numel():
                    action_views[
                        (lane, action_partition, "advantage_target")
                    ][rows] = advantage[lane_rows]
                    target_bytes += (
                        rows.numel() * width * torch.tensor([], dtype=torch.float32).element_size()
                    )
                    if posterior is not None:
                        action_views[
                            (lane, action_partition, "posterior")
                        ][rows] = posterior[lane_rows]
                        target_bytes += rows.numel() * width * 4
                lane_nodes = trained & active & (replica == lane)
                sigma_rows = nodes["sigma_target_row"][sl].long()[lane_nodes]
                if sigma_rows.numel():
                    sigma_views[
                        (lane, group.input_type, "value_target")
                    ][sigma_rows] = sigma_value[lane_nodes]
                    target_bytes += sigma_rows.numel() * width * 4
        else:
            sigma_value = None

        if self.branched:
            assert child_trace is not None
            assert child_cost is not None and child_delta is not None
            trace_mean = (
                child_trace * target[..., None]
            ).sum(1) / target_count.to(torch.float32)[:, None]
            delta_bar = (
                child_delta * target
            ).sum(1) / target_count.to(torch.float32)
            path = children["path_length"][safe_edge].to(torch.float32)
            multiplier = children["cost_multiplier"][safe_edge]
            cost_raw = (
                (path + multiplier * child_cost) * target
            ).sum(1) / target_count.to(torch.float32)
            if group.input_type >= 0:
                old_trace = owners.slab(
                    "SigmaHead", plan.input_names[group.input_type]
                ).columns["old_trace"][
                    replica, nodes["sigma_head_row"][sl].long()
                ]
                old_cost = owners.slab(
                    "SigmaHead", plan.input_names[group.input_type]
                ).columns["old_cost"][
                    replica, nodes["sigma_head_row"][sl].long()
                ]
                centered = child_trace - trace_mean[:, None]
                trace_exec = (
                    centered.square().sum(-1) * target
                ).sum(1) / (target_count - 1).clamp_min(1).to(torch.float32)
                trace_hat = (trace_exec - delta_bar).clamp_min(0.0)
                has_trace = target_count >= 2
                t_star = torch.where(has_trace, trace_hat, old_trace)
                parent_delta = (
                    (t_star + delta_bar) / target_count.to(torch.float32)
                    - t_star / nodes["nominal_degree"][sl].to(torch.float32)
                )
                parent_cost = (
                    self.lambda_ * cost_raw + (1.0 - self.lambda_) * old_cost
                )
                target_finite &= (
                    torch.isfinite(trace_hat)
                    & torch.isfinite(cost_raw)
                    & torch.isfinite(parent_delta)
                    & torch.isfinite(parent_cost)
                )
                for lane in range(owners.plan.manifest.n_replica):
                    lane_nodes = trained & active & (replica == lane)
                    sigma_rows = nodes["sigma_target_row"][sl].long()[lane_nodes]
                    if sigma_rows.numel():
                        sigma_views[
                            (lane, group.input_type, "trace_target")
                        ][sigma_rows] = torch.where(
                            has_trace[lane_nodes],
                            trace_hat[lane_nodes],
                            torch.zeros_like(trace_hat[lane_nodes]),
                        )
                        sigma_views[
                            (lane, group.input_type, "cost_target")
                        ][sigma_rows] = cost_raw[lane_nodes]
                        target_bytes += sigma_rows.numel() * 8
            else:
                parent_delta = delta_bar / target_count.to(torch.float32)
                parent_cost = cost_raw
        else:
            trace_mean = parent_delta = parent_cost = None

        finite = (
            torch.isfinite(child_primary)
            .where(
                valid[..., None, None],
                torch.ones_like(child_primary, dtype=torch.bool),
            )
            .flatten(1)
            .all(1)
        )
        finite &= torch.isfinite(parent_primary).flatten(1).all(1)
        if parent_secondary is not None:
            assert child_secondary is not None
            child_valid = valid[..., None, None]
            finite &= torch.where(
                child_valid,
                torch.isfinite(child_secondary),
                torch.ones_like(child_secondary, dtype=torch.bool),
            ).flatten(1).all(1)
            finite &= torch.where(
                child_valid,
                child_primary >= 0,
                torch.ones_like(child_primary, dtype=torch.bool),
            ).flatten(1).all(1)
            finite &= torch.where(
                child_valid & (child_primary == 0),
                child_secondary.abs() <= 1.0e-7,
                torch.ones_like(child_secondary, dtype=torch.bool),
            ).flatten(1).all(1)
            finite &= torch.isfinite(parent_secondary).flatten(1).all(1)
            finite &= (parent_primary >= 0).flatten(1).all(1)
            zero_mass = parent_primary == 0
            finite &= (
                (~zero_mass)
                | (parent_secondary.abs() <= 1.0e-7)
            ).flatten(1).all(1)
        if self.branched:
            assert trace_mean is not None
            assert parent_delta is not None and parent_cost is not None
            finite &= torch.isfinite(trace_mean).flatten(1).all(1)
            finite &= torch.isfinite(parent_delta) & torch.isfinite(parent_cost)
        finite &= target_finite
        invalid = ~finite
        for lane in range(owners.plan.manifest.n_replica):
            lane_failed[lane] |= (invalid & (replica == lane)).any()
        write = active & finite & ~lane_failed[replica]
        destination_work = owners.slab(
            "LayerWork", plan.output_names[group.output_type]
        ).columns
        selected_replica = replica[write]
        selected_destination = destination[write]
        if self.sampled:
            destination_work["return_value"][
                selected_replica, selected_destination, dst_arena
            ] = parent_primary[write]
        else:
            destination_work["mass"][
                selected_replica, selected_destination, dst_arena
            ] = parent_primary[write]
            assert parent_secondary is not None
            destination_work["moment"][
                selected_replica, selected_destination, dst_arena
            ] = parent_secondary[write]
        if self.branched:
            assert trace_mean is not None
            assert parent_cost is not None and parent_delta is not None
            destination_work["trace_vector"][
                selected_replica, selected_destination, dst_arena
            ] = trace_mean[write]
            destination_work["cost"][
                selected_replica, selected_destination, dst_arena
            ] = parent_cost[write]
            destination_work["delta"][
                selected_replica, selected_destination, dst_arena
            ] = parent_delta[write]

        elements = batch * n_max * p_train * width
        mm_read = 0 if self.sampled else elements * 2 * 4
        mm_written = (
            0 if self.sampled else batch * p_train * width * 2 * 4
        )
        return invalid, target_bytes, mm_read, mm_written

    def _execute_carries(
        self,
        owners: GeneratedOwners,
        plan: LayerPlan,
        device_plan: DeviceLayerPlan,
        layer: int,
        src_arena: int,
        dst_arena: int,
        lane_failed: Tensor,
    ) -> int:
        first = int(plan.carry_ptr[layer])
        stop = int(plan.carry_ptr[layer + 1])
        if first == stop:
            return 0
        carries = device_plan.carries.columns
        payload_bytes = 0
        for output, name in enumerate(plan.output_names):
            host_selected = (
                plan.carry_output_type[first:stop] == output
            ).nonzero(as_tuple=False).flatten()
            if not host_selected.numel():
                continue
            index = host_selected.to(device=owners.plan.device) + first
            replica = carries["replica"][index].long()
            active = ~lane_failed[replica]
            source = carries["source_work_row"][index].long()
            destination = carries["destination_work_row"][index].long()
            work = owners.slab("LayerWork", name).columns
            for field in self.mode.owner("LayerWork").fields:
                value = work[field][replica, source, src_arena]
                work[field][
                    replica[active], destination[active], dst_arena
                ] = value[active]
                payload_bytes += value[active].numel() * value.element_size()
        return payload_bytes


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
