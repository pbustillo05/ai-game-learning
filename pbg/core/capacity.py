"""Immutable capacity admission and generated queued-storage ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

import torch

from pbg.core.contracts import (
    CAPACITY_VOCABULARY,
    ModeSchema,
    Partition,
    PartitionContract,
    RolloutMode,
    TrainingRegime,
)
from pbg.core.storage import (
    DeviceFailure,
    DeviceFailureState,
    MonotoneSlab,
    SoAField,
    SoASchema,
    StorageError,
)


class CapacityError(ValueError):
    """An immutable manifest does not cover its declared execution."""


class AdmissionError(MemoryError):
    """A validated phase cannot fit its immutable memory budget."""


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CapacityError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CapacityError(f"{label} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True, order=True)
class PartitionRows:
    """One immutable partition capacity."""

    name: str
    rows: int

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise CapacityError(f"invalid capacity partition {self.name!r}")
        _nonnegative_int(self.rows, f"{self.name} rows")


@dataclass(frozen=True, order=True)
class NamedShape:
    """One immutable game-supplied fixed physical shape."""

    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise CapacityError(f"invalid shape name {self.name!r}")
        shape = tuple(self.shape)
        if not shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in shape
        ):
            raise CapacityError(f"{self.name} must have a positive fixed shape")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class ExecutionGeometry:
    """Game-supplied scalar bounds needed by generic capacity admission."""

    actor_count: int
    occurrence_bound: int
    model_token_bound: int
    model_token_pair_bound: int

    def __post_init__(self) -> None:
        _positive_int(self.actor_count, "actor_count")
        _positive_int(self.occurrence_bound, "occurrence_bound")
        _positive_int(self.model_token_bound, "model_token_bound")
        _positive_int(self.model_token_pair_bound, "model_token_pair_bound")
        if self.model_token_pair_bound < self.model_token_bound:
            raise CapacityError("model token-pair bound is smaller than token bound")


@dataclass(frozen=True)
class HardwareLimits:
    """Immutable accelerator identity and hard admission limits."""

    total_memory_bytes: int
    usable_memory_bytes: int
    reserve_bytes: int = 0

    def __post_init__(self) -> None:
        total = _positive_int(self.total_memory_bytes, "total_memory_bytes")
        budget = _positive_int(self.usable_memory_bytes, "usable_memory_bytes")
        reserve = _nonnegative_int(self.reserve_bytes, "reserve_bytes")
        if budget + reserve > total:
            raise CapacityError("usable memory plus reserve exceeds observed device memory")


@dataclass(frozen=True)
class PhysicalLayout:
    """Opaque fixed shapes needed to materialize a mode's declared fields."""

    model_input_shapes: tuple[NamedShape, ...]
    player_count: int
    stratum_count: int
    evaluator_private_width: int
    reference_cache_bytes: int = 0
    field_alignment: int = 64
    model_dtype: torch.dtype = torch.float32
    opaque_fields: tuple[SoAField, ...] = ()

    def __post_init__(self) -> None:
        shapes = tuple(self.model_input_shapes)
        if len({item.name for item in shapes}) != len(shapes):
            raise CapacityError("model input shapes contain duplicate partitions")
        object.__setattr__(self, "model_input_shapes", shapes)
        _positive_int(self.player_count, "layout player_count")
        _positive_int(self.stratum_count, "stratum_count")
        _positive_int(self.evaluator_private_width, "evaluator_private_width")
        _nonnegative_int(self.reference_cache_bytes, "reference_cache_bytes")
        if (
            isinstance(self.field_alignment, bool)
            or not isinstance(self.field_alignment, int)
            or self.field_alignment <= 0
            or self.field_alignment & (self.field_alignment - 1)
            or self.field_alignment > 256
        ):
            raise CapacityError(
                "field_alignment must be a power of two no greater than 256"
            )
        if self.model_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise CapacityError("model_dtype must be fp32, fp16, or bf16")
        opaque = tuple(self.opaque_fields)
        if len({item.name for item in opaque}) != len(opaque):
            raise CapacityError("opaque field contracts contain duplicate names")
        object.__setattr__(self, "opaque_fields", opaque)

    @property
    def input_shapes(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(
            {item.name: item.shape for item in self.model_input_shapes}
        )


@dataclass(frozen=True)
class CapacityManifest:
    """The sole immutable source for every fixed run capacity."""

    geometry: ExecutionGeometry
    root_count: int
    n_replica: int
    beta_max: int
    player_node_cap: tuple[PartitionRows, ...]
    trained_player_cap: tuple[PartitionRows, ...]
    action_cap: tuple[PartitionRows, ...]
    folded_range_cap: tuple[PartitionRows, ...]
    leaf_cap: int
    topology_cap: int
    topology_edge_cap: int
    family_pack_cap: int
    chance_frontier_cap: int
    completion_pack_cap: int
    learner_sigma_pack_cap: int
    learner_action_pack_cap: int
    topology_packet_cap: int
    target_arena_cap: int
    rollout_workspace_bytes: int
    queue_drain_workspace_bytes: int
    completion_workspace_bytes: int
    align_workspace_bytes: int
    target_workspace_bytes: int
    learner_workspace_bytes: int
    hardware: HardwareLimits

    def __post_init__(self) -> None:
        _positive_int(self.root_count, "root_count")
        _positive_int(self.n_replica, "n_replica")
        _positive_int(self.beta_max, "beta_max")
        for name in CAPACITY_VOCABULARY.partitioned_rows:
            values = tuple(getattr(self, name))
            if not values or len({item.name for item in values}) != len(values):
                raise CapacityError(f"{name} must have unique nonempty partitions")
            object.__setattr__(self, name, values)
        for name in (
            *CAPACITY_VOCABULARY.scalar_rows,
            *CAPACITY_VOCABULARY.pack_limits,
        ):
            _positive_int(getattr(self, name), name)
        for name in CAPACITY_VOCABULARY.workspaces:
            _nonnegative_int(getattr(self, name), name)

    def partition_rows(self, name: str) -> Mapping[str, int]:
        if name not in CAPACITY_VOCABULARY.partitioned_rows:
            raise KeyError(f"unknown partitioned capacity {name!r}")
        return MappingProxyType(
            {item.name: item.rows for item in getattr(self, name)}
        )

class ExecutionPhase(IntEnum):
    """Ordered allocation and last-consumer boundaries."""

    ROLLOUT = 0
    QUEUE_DRAIN = 1
    COMPLETION = 2
    ALIGN_AND_SEAL = 3
    TARGETS = 4
    LEARNER = 5
    FINAL_BACKWARD = 6


@dataclass(frozen=True, order=True)
class OwnerKey:
    owner: str
    partition: str

    @property
    def label(self) -> str:
        return self.owner if not self.partition else f"{self.owner}.{self.partition}"


@dataclass(frozen=True)
class OwnerAllocation:
    """One generated CUDA owner instance."""

    key: OwnerKey
    schema: SoASchema
    capacity: int
    start_phase: ExecutionPhase
    last_consumer: ExecutionPhase
    analytical_bytes: int


@dataclass(frozen=True)
class PhaseAdmission:
    phase: ExecutionPhase
    owner_bytes: int
    bounded_workspace_bytes: int
    guard_bytes: int
    required_bytes: int
    configured_budget_bytes: int


@dataclass(frozen=True)
class StoragePlan:
    """A fully validated, allocation-free generated owner plan."""

    mode: ModeSchema
    partitions: PartitionContract
    manifest: CapacityManifest
    layout: PhysicalLayout
    allocations: tuple[OwnerAllocation, ...]
    deferred_cpu_owners: tuple[str, ...]
    admissions: tuple[PhaseAdmission, ...]
    device: torch.device

    def admission(self, phase: ExecutionPhase) -> PhaseAdmission:
        return self.admissions[int(phase)]


_LIFETIMES: Mapping[str, tuple[ExecutionPhase, ExecutionPhase]] = MappingProxyType(
    {
        "FamilyWork": (ExecutionPhase.ROLLOUT, ExecutionPhase.ALIGN_AND_SEAL),
        "FamilyRetained": (
            ExecutionPhase.ROLLOUT,
            ExecutionPhase.FINAL_BACKWARD,
        ),
        "LearnerRetained": (
            ExecutionPhase.ALIGN_AND_SEAL,
            ExecutionPhase.FINAL_BACKWARD,
        ),
        "SigmaHead": (ExecutionPhase.ROLLOUT, ExecutionPhase.FINAL_BACKWARD),
        "ActionHead": (ExecutionPhase.ROLLOUT, ExecutionPhase.FINAL_BACKWARD),
        "Leaf": (ExecutionPhase.ROLLOUT, ExecutionPhase.COMPLETION),
        "FoldedRanges": (
            ExecutionPhase.ROLLOUT,
            ExecutionPhase.ALIGN_AND_SEAL,
        ),
        "QueueMetadata": (ExecutionPhase.ROLLOUT, ExecutionPhase.QUEUE_DRAIN),
        "LayerWork": (ExecutionPhase.COMPLETION, ExecutionPhase.TARGETS),
        "SigmaTarget": (ExecutionPhase.TARGETS, ExecutionPhase.FINAL_BACKWARD),
        "ActionTarget": (ExecutionPhase.TARGETS, ExecutionPhase.FINAL_BACKWARD),
        "LearnerIndex": (
            ExecutionPhase.ALIGN_AND_SEAL,
            ExecutionPhase.FINAL_BACKWARD,
        ),
    }
)

_CPU_OWNERS = ("Topology", "TopologyChild")
BRANCHED_ALIAS_PEAK_ROW_BYTES = 44
BRANCHED_LOG_MASS_ROW_BYTES = 12


def _output_width(partitions: PartitionContract, input_name: str) -> int:
    return partitions.output_for(input_name).width


def _partition_width(
    owner_axis: str,
    partition: Partition,
    partitions: PartitionContract,
) -> int:
    if owner_axis == "input":
        return _output_width(partitions, partition.name)
    return partition.width


def _field(
    mode: ModeSchema,
    owner_name: str,
    field_name: str,
    partition: Partition | None,
    partitions: PartitionContract,
    layout: PhysicalLayout,
) -> SoAField:
    alignment = layout.field_alignment
    players = layout.player_count
    width = (
        1
        if partition is None
        else _partition_width(
            mode.owner(owner_name).partition_axis, partition, partitions
        )
    )
    integer_fill = -1

    if field_name == "logical_rng_key":
        return SoAField(field_name, torch.int64, (2,), alignment)
    if field_name in ("true_profile", "sampled_private_state"):
        return SoAField(field_name, torch.int32, (players,), alignment, integer_fill)
    if field_name == "folded_range_row":
        return SoAField(field_name, torch.int32, (players,), alignment, integer_fill)
    if field_name in (
        "role_selector",
        "response_cutoff",
    ):
        return SoAField(field_name, torch.uint8, (), alignment)
    if field_name == "model_input":
        if partition is None or partition.name not in layout.input_shapes:
            raise CapacityError("model_input requires a declared input-partition shape")
        return SoAField(
            field_name,
            layout.model_dtype,
            layout.input_shapes[partition.name],
            alignment,
        )
    if field_name in (
        "owner",
        "nominal_degree",
        "executed_degree",
        "target_degree",
        "action",
        "trained_actor",
        "head",
        "ready_tail",
        "reserved_tail",
        "family_row",
        "action_begin",
        "action_count",
    ):
        fill = (
            integer_fill
            if field_name
            in (
                "owner",
                "action",
                "trained_actor",
                "family_row",
                "action_begin",
            )
            else 0
        )
        return SoAField(field_name, torch.int32, (), alignment, fill)
    if field_name == "stratum_level":
        return SoAField(field_name, torch.int32, (players,), alignment, integer_fill)
    if field_name == "payoff_baseline":
        return SoAField(field_name, torch.float32, (players,), alignment)
    if field_name == "stratum_amount":
        return SoAField(
            field_name, torch.float32, (layout.stratum_count,), alignment
        )
    if field_name == "range_weights":
        return SoAField(
            field_name,
            torch.float32,
            (players, layout.evaluator_private_width),
            alignment,
        )
    if field_name == "seed":
        return SoAField(field_name, torch.int64, (), alignment)
    if field_name == "range":
        if partition is None:
            raise CapacityError("folded range requires a partition")
        return SoAField(field_name, torch.float32, (partition.width,), alignment)
    if field_name == "log_mass":
        return SoAField(field_name, torch.float32, (), alignment)
    if field_name in (
        "old_value",
        "old_log_policy",
        "value_target",
        "posterior",
        "advantage_target",
    ):
        return SoAField(field_name, torch.float32, (width,), alignment)
    if field_name in ("return_value", "mass", "moment"):
        p_train = (
            players if mode.training_regime is TrainingRegime.SELF_PLAY else 1
        )
        # One row owns both fixed ping-pong arenas and every trained channel.
        return SoAField(
            field_name, torch.float32, (2, p_train, width), alignment
        )
    if field_name == "trace_vector":
        p_train = (
            players if mode.training_regime is TrainingRegime.SELF_PLAY else 1
        )
        return SoAField(field_name, torch.float32, (2, p_train), alignment)
    if field_name in (
        "path_mass",
        "log_player_ancestor_sum",
        "log_ancestor_product",
        "branch_beta",
        "frontier_cost_contribution",
        "old_trace",
        "old_cost",
        "frontier_cost",
        "trace_target",
        "cost_target",
        "log_sampling_weight",
    ):
        return SoAField(field_name, torch.float32, (), alignment)
    if field_name in ("cost", "delta"):
        return SoAField(field_name, torch.float32, (2,), alignment)
    for opaque in layout.opaque_fields:
        if opaque.name == field_name:
            return opaque
    raise CapacityError(f"no physical field contract for {owner_name}.{field_name}")


def _axis_partitions(
    axis: str, partitions: PartitionContract
) -> tuple[Partition | None, ...]:
    if axis == "input":
        return tuple(partitions.inputs)
    if axis == "output":
        return tuple(partitions.outputs)
    if axis == "folded":
        return tuple(partitions.folded)
    if axis in ("leaf",):
        return (None,)
    raise CapacityError(f"unsupported CUDA owner partition axis {axis!r}")


def _owner_capacity(
    owner: str,
    partition: Partition | None,
    manifest: CapacityManifest,
) -> int:
    if owner in ("FamilyWork", "FamilyRetained"):
        source = "player_node_cap"
    elif owner in (
        "SigmaHead",
        "SigmaTarget",
        "LearnerIndex",
        "LearnerRetained",
    ):
        source = "trained_player_cap"
    elif owner in ("ActionHead", "ActionTarget"):
        source = "action_cap"
    elif owner == "FoldedRanges":
        source = "folded_range_cap"
    elif owner == "Leaf":
        return manifest.leaf_cap
    elif owner == "QueueMetadata":
        return 1
    elif owner == "LayerWork":
        return manifest.target_arena_cap
    else:
        raise CapacityError(f"unknown CUDA owner capacity for {owner!r}")
    if partition is None:
        raise CapacityError(f"{owner} requires a partition capacity")
    try:
        return manifest.partition_rows(source)[partition.name]
    except KeyError as exc:
        raise CapacityError(
            f"{source} does not cover partition {partition.name!r}"
        ) from exc


def _validate_manifest(
    mode: ModeSchema,
    partitions: PartitionContract,
    manifest: CapacityManifest,
    layout: PhysicalLayout,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        raise CapacityError("queued storage manifests require CUDA")
    if not torch.cuda.is_available():
        raise CapacityError("CUDA is unavailable")
    if layout.player_count != manifest.geometry.actor_count:
        raise CapacityError("physical layout and game geometry player counts differ")
    if set(layout.input_shapes) != {item.name for item in partitions.inputs}:
        raise CapacityError(
            "physical layout must cover every input partition exactly"
        )
    if layout.evaluator_private_width < max(item.width for item in partitions.outputs):
        raise CapacityError("evaluator private width does not cover output domains")
    if mode.output_domain == "scalar" and (
        len(partitions.outputs) != 1 or partitions.outputs[0].width != 1
    ):
        raise CapacityError("sampled mode requires one scalar output partition")
    if mode.output_domain == "private_state" and not partitions.conversions:
        raise CapacityError(
            "private-state mode requires an explicit endpoint conversion"
        )
    if mode.rollout_mode is RolloutMode.SEQUENTIAL and manifest.beta_max != 1:
        raise CapacityError("sequential manifests require beta_max=1")

    expected_keys = {
        "player_node_cap": {item.name for item in partitions.inputs},
        "trained_player_cap": {item.name for item in partitions.inputs},
        "action_cap": {
            item.name
            for item in (
                partitions.inputs
                if mode.action_partition_axis == "input"
                else partitions.outputs
            )
        },
        "folded_range_cap": {item.name for item in partitions.folded},
    }
    for capacity_name, expected in expected_keys.items():
        actual = set(manifest.partition_rows(capacity_name))
        if actual != expected:
            raise CapacityError(
                f"{capacity_name} partitions are {sorted(actual)}, "
                f"expected {sorted(expected)}"
            )

    player_rows = sum(manifest.partition_rows("player_node_cap").values())
    trained_rows = sum(manifest.partition_rows("trained_player_cap").values())
    action_rows = sum(manifest.partition_rows("action_cap").values())
    folded_rows = sum(manifest.partition_rows("folded_range_cap").values())
    geometry = manifest.geometry
    minimum_player_rows = manifest.root_count * geometry.occurrence_bound
    if player_rows < minimum_player_rows:
        raise CapacityError(
            f"player_node_cap={player_rows} does not cover geometry bound "
            f"{minimum_player_rows}"
        )
    if trained_rows > player_rows:
        raise CapacityError("trained player capacity exceeds player-node capacity")
    if action_rows < trained_rows:
        raise CapacityError("action capacity does not cover trained-node geometry")
    if action_rows < manifest.beta_max:
        raise CapacityError("action capacity does not cover beta_max")
    if manifest.leaf_cap < manifest.root_count:
        raise CapacityError("leaf capacity does not cover root/branch geometry")
    if manifest.topology_cap < manifest.root_count:
        raise CapacityError("topology capacity does not cover distinguished roots")
    if manifest.topology_edge_cap < max(
        1, manifest.topology_cap - manifest.root_count
    ):
        raise CapacityError("topology edge capacity cannot connect its node bound")
    if geometry.actor_count == 2 and folded_rows != 0:
        raise CapacityError("heads-up folded-range capacity must be zero")
    if geometry.actor_count > 2 and folded_rows < len(partitions.folded):
        raise CapacityError("folded-range capacity does not cover its partitions")
    if manifest.topology_edge_cap < manifest.beta_max:
        raise CapacityError("topology edge capacity does not cover beta_max")
    if manifest.target_arena_cap < manifest.leaf_cap + manifest.topology_cap:
        raise CapacityError("target arenas do not cover leaf plus topology rows")

    pack_bounds = {
        "family_pack_cap": player_rows,
        "chance_frontier_cap": player_rows + manifest.leaf_cap,
        "completion_pack_cap": manifest.leaf_cap,
        "learner_sigma_pack_cap": trained_rows,
        "learner_action_pack_cap": action_rows,
        "topology_packet_cap": manifest.topology_cap
        + manifest.topology_edge_cap,
    }
    for name, upper in pack_bounds.items():
        if getattr(manifest, name) > upper:
            raise CapacityError(f"{name} exceeds its covered row capacity")

    properties = torch.cuda.get_device_properties(device)
    actual_total = int(properties.total_memory)
    if manifest.hardware.total_memory_bytes != actual_total:
        raise CapacityError("resolved manifest device memory is stale")
    if manifest.hardware.usable_memory_bytes > actual_total:
        raise CapacityError("usable memory exceeds selected CUDA device memory")


def _bounded_workspace(
    phase: ExecutionPhase,
    manifest: CapacityManifest,
) -> int:
    return {
        ExecutionPhase.ROLLOUT: manifest.rollout_workspace_bytes,
        ExecutionPhase.QUEUE_DRAIN: manifest.queue_drain_workspace_bytes,
        ExecutionPhase.COMPLETION: manifest.completion_workspace_bytes,
        ExecutionPhase.ALIGN_AND_SEAL: manifest.align_workspace_bytes,
        ExecutionPhase.TARGETS: manifest.target_workspace_bytes,
        ExecutionPhase.LEARNER: manifest.learner_workspace_bytes,
    }.get(phase, 0)


def build_storage_plan(
    mode: ModeSchema,
    partitions: PartitionContract,
    manifest: CapacityManifest,
    layout: PhysicalLayout,
    device: torch.device | str,
    *,
    rollout_only: bool = False,
) -> StoragePlan:
    """Validate a manifest and generate every declared CUDA owner."""

    resolved = torch.device(device)
    _validate_manifest(mode, partitions, manifest, layout, resolved)
    allocations: list[OwnerAllocation] = []
    accounted_owner_names: set[str] = set()
    for owner in mode.owners:
        if owner.name in _CPU_OWNERS:
            accounted_owner_names.add(owner.name)
            continue
        try:
            start_phase, last_consumer = _LIFETIMES[owner.name]
        except KeyError as exc:
            raise CapacityError(f"owner {owner.name!r} has no lifetime plan") from exc
        if (
            owner.name == "FamilyRetained"
            and mode.training_regime is TrainingRegime.BEST_RESPONSE
        ):
            last_consumer = ExecutionPhase.ALIGN_AND_SEAL
        for partition in _axis_partitions(owner.partition_axis, partitions):
            if rollout_only and start_phase is not ExecutionPhase.ROLLOUT:
                continue
            partition_name = "" if partition is None else partition.name
            schema = SoASchema(
                name=OwnerKey(owner.name, partition_name).label,
                fields=tuple(
                    _field(
                        mode,
                        owner.name,
                        field,
                        partition,
                        partitions,
                        layout,
                    )
                    for field in owner.fields
                ),
                lifetime=owner.lifetime,
            )
            capacity = _owner_capacity(owner.name, partition, manifest)
            allocations.append(
                OwnerAllocation(
                    key=OwnerKey(owner.name, partition_name),
                    schema=schema,
                    capacity=capacity,
                    start_phase=start_phase,
                    last_consumer=last_consumer,
                    analytical_bytes=MonotoneSlab.analytical_bytes(
                        schema, capacity, manifest.n_replica
                    ),
                )
            )
        accounted_owner_names.add(owner.name)
    declared_names = {owner.name for owner in mode.owners}
    if accounted_owner_names != declared_names:
        raise CapacityError("generated owner plan does not cover the mode schema")

    failure_bytes = DeviceFailureState.analytical_bytes(manifest.n_replica)
    admissions: list[PhaseAdmission] = []
    for phase in ExecutionPhase:
        owner_bytes = failure_bytes + sum(
            allocation.analytical_bytes
            for allocation in allocations
            if allocation.start_phase <= phase <= allocation.last_consumer
        )
        bounded = _bounded_workspace(phase, manifest)
        required = owner_bytes + bounded
        admission = PhaseAdmission(
            phase=phase,
            owner_bytes=owner_bytes,
            bounded_workspace_bytes=bounded,
            guard_bytes=manifest.hardware.reserve_bytes,
            required_bytes=required,
            configured_budget_bytes=manifest.hardware.usable_memory_bytes,
        )
        deferred_layer_work = (
            ExecutionPhase.COMPLETION
            <= phase
            <= ExecutionPhase.TARGETS
        )
        deferred_bytes = (
            sum(
                allocation.analytical_bytes
                for allocation in allocations
                if allocation.key.owner == "LayerWork"
                and allocation.start_phase <= phase <= allocation.last_consumer
            )
            if deferred_layer_work
            else 0
        )
        if (
            required - deferred_bytes
            > manifest.hardware.usable_memory_bytes
        ):
            raise AdmissionError(
                f"{phase.name} non-deferred owners require "
                f"{required - deferred_bytes} bytes but immutable budget is "
                f"{manifest.hardware.usable_memory_bytes}"
            )
        admissions.append(admission)

    free_bytes, _ = torch.cuda.mem_get_info(resolved)
    peak_required = max(
        item.required_bytes
        for item in admissions
        if not (
            ExecutionPhase.COMPLETION
            <= item.phase
            <= ExecutionPhase.TARGETS
        )
    )
    reserve = manifest.hardware.reserve_bytes
    if peak_required + reserve > int(free_bytes):
        raise AdmissionError(
            f"validated plan plus reserve requires {peak_required + reserve} free bytes, selected device "
            f"currently has {int(free_bytes)}"
        )
    return StoragePlan(
        mode=mode,
        partitions=partitions,
        manifest=manifest,
        layout=layout,
        allocations=tuple(allocations),
        deferred_cpu_owners=_CPU_OWNERS,
        admissions=tuple(admissions),
        device=resolved,
    )


@dataclass(frozen=True)
class MemoryReport:
    """One quiescent reconciliation of configured and observed bytes."""

    phase: ExecutionPhase
    configured_budget_bytes: int
    analytical_live_bytes: int
    allocated_owner_bytes: int
    high_water_payload_bytes: int
    bounded_workspace_bytes: int
    guard_bytes: int
    admitted_required_bytes: int
    observed_allocated_bytes: int
    observed_reserved_bytes: int
    observed_peak_allocated_bytes: int
    observed_peak_reserved_bytes: int

    def assert_reconciled(self) -> None:
        if self.allocated_owner_bytes != self.analytical_live_bytes:
            raise StorageError(
                "named allocated bytes do not match schema analytical bytes"
            )
        if self.admitted_required_bytes > self.configured_budget_bytes:
            raise StorageError("observed phase was not admitted by its manifest")
        if self.high_water_payload_bytes > self.allocated_owner_bytes:
            raise StorageError("high-water payload exceeds owner allocation")
        if self.observed_allocated_bytes < self.allocated_owner_bytes:
            raise StorageError("CUDA allocator observation is below named owners")
        if self.observed_reserved_bytes < self.observed_allocated_bytes:
            raise StorageError("CUDA reserved bytes are below allocated bytes")
        if self.observed_peak_allocated_bytes < self.observed_allocated_bytes:
            raise StorageError("CUDA allocated peak is below its current value")
        if self.observed_peak_reserved_bytes < self.observed_reserved_bytes:
            raise StorageError("CUDA reserved peak is below its current value")


class GeneratedOwners:
    """The sole owner of a generated plan's CUDA allocations and releases."""

    def __init__(self, plan: StoragePlan) -> None:
        self.plan = plan
        self.failure_state = DeviceFailureState(
            plan.manifest.n_replica, plan.device
        )
        self._live: dict[OwnerKey, MonotoneSlab] = {}
        self._released: set[OwnerKey] = set()
        self._release_counts: dict[OwnerKey, int] = {}
        self._entered_phase: ExecutionPhase | None = None
        self._released_failure_state = False
        self._realized_capacities: dict[OwnerKey, int] = {}
        self._runtime_admissions: dict[ExecutionPhase, PhaseAdmission] = {}

    @property
    def layer_work_is_bound(self) -> bool:
        expected = {
            item.key
            for item in self.plan.allocations
            if item.key.owner == "LayerWork"
        }
        return expected == set(self._realized_capacities)

    def bind_layer_work(
        self,
        output_names: tuple[str, ...],
        arena_rows: tuple[tuple[int, ...], ...],
    ) -> None:
        """Bind the planner's exact immutable arena extents before completion."""

        if self._entered_phase is not ExecutionPhase.ROLLOUT:
            raise StorageError("LayerWork extents must bind during rollout sealing")
        if self.layer_work_is_bound:
            raise StorageError("LayerWork extents were already bound")
        if (
            len(arena_rows) != self.plan.manifest.n_replica
            or any(len(rows) != len(output_names) for rows in arena_rows)
        ):
            raise StorageError("LayerWork extents have invalid replica partitions")
        expected_names = tuple(item.name for item in self.plan.partitions.outputs)
        if output_names != expected_names:
            raise StorageError("LayerWork extents differ from output partitions")
        for output, name in enumerate(output_names):
            capacity = max(
                (int(rows[output]) for rows in arena_rows),
                default=0,
            )
            if capacity < 0 or capacity > self.plan.manifest.target_arena_cap:
                raise AdmissionError(
                    f"LayerWork.{name} exact extent exceeds target_arena_cap"
                )
            self._realized_capacities[OwnerKey("LayerWork", name)] = capacity
        try:
            self._build_runtime_admissions()
        except BaseException:
            self._realized_capacities.clear()
            self._runtime_admissions.clear()
            raise

    def bind_empty_layer_work_for_abort(self) -> None:
        """Bind zero extents solely so a pre-plan failed attempt can unwind."""

        if self.layer_work_is_bound:
            return
        output_names = tuple(item.name for item in self.plan.partitions.outputs)
        arena_rows = tuple(
            tuple(0 for _name in output_names)
            for _lane in range(self.plan.manifest.n_replica)
        )
        self.bind_layer_work(output_names, arena_rows)

    def _allocation_capacity(self, allocation: OwnerAllocation) -> int:
        return self._realized_capacities.get(
            allocation.key, allocation.capacity
        )

    def _allocation_bytes(self, allocation: OwnerAllocation) -> int:
        capacity = self._allocation_capacity(allocation)
        if capacity == allocation.capacity:
            return allocation.analytical_bytes
        return MonotoneSlab.analytical_bytes(
            allocation.schema,
            capacity,
            self.plan.manifest.n_replica,
        )

    def _build_runtime_admissions(self) -> None:
        for phase in (
            ExecutionPhase.COMPLETION,
            ExecutionPhase.ALIGN_AND_SEAL,
            ExecutionPhase.TARGETS,
        ):
            owner_bytes = DeviceFailureState.analytical_bytes(
                self.plan.manifest.n_replica
            ) + sum(
                self._allocation_bytes(allocation)
                for allocation in self.plan.allocations
                if allocation.start_phase <= phase <= allocation.last_consumer
            )
            planned = self.plan.admission(phase)
            required = owner_bytes + planned.bounded_workspace_bytes
            if required > planned.configured_budget_bytes:
                raise AdmissionError(
                    f"{phase.name} exact LayerWork plan requires {required} bytes "
                    f"but immutable budget is {planned.configured_budget_bytes}"
                )
            self._runtime_admissions[phase] = PhaseAdmission(
                phase=phase,
                owner_bytes=owner_bytes,
                bounded_workspace_bytes=planned.bounded_workspace_bytes,
                guard_bytes=planned.guard_bytes,
                required_bytes=required,
                configured_budget_bytes=planned.configured_budget_bytes,
            )

    def admission(self, phase: ExecutionPhase) -> PhaseAdmission:
        return self._runtime_admissions.get(phase, self.plan.admission(phase))

    @property
    def live_slabs(self) -> tuple[MonotoneSlab, ...]:
        """All sanctioned live retained owners in stable allocation order."""

        return tuple(
            self._live[key]
            for key in sorted(self._live, key=lambda item: item.label)
        )

    def enter_phase(
        self,
        phase: ExecutionPhase,
        *,
        failure_injector: Callable[[OwnerAllocation], None] | None = None,
    ) -> None:
        if self._released_failure_state:
            raise StorageError("generated owners have been closed")
        if self._entered_phase is None:
            if phase is not ExecutionPhase.ROLLOUT:
                raise StorageError("owner lifecycle must start at ROLLOUT")
        elif int(phase) != int(self._entered_phase) + 1:
            raise StorageError("owner phases must be entered exactly once in order")
        if self._entered_phase is not None:
            due = {
                item.key
                for item in self.plan.allocations
                if item.last_consumer is self._entered_phase
            }
            if due.intersection(self._live):
                raise StorageError(
                    f"owners past {self._entered_phase.name} last consumer "
                    "have not been explicitly released"
                )

        if (
            phase is ExecutionPhase.COMPLETION
            and not self.layer_work_is_bound
        ):
            raise StorageError(
                "LayerWork exact extents must be bound before completion"
            )
        if phase is ExecutionPhase.COMPLETION:
            torch.cuda.synchronize(self.plan.device)
            torch.cuda.empty_cache()
        admission = self.admission(phase)
        if admission.required_bytes > admission.configured_budget_bytes:
            raise AdmissionError(f"{phase.name} failed immutable admission")
        starts = tuple(
            item for item in self.plan.allocations if item.start_phase is phase
        )
        if phase is ExecutionPhase.COMPLETION:
            free_bytes, _ = torch.cuda.mem_get_info(self.plan.device)
            incremental = (
                sum(self._allocation_bytes(item) for item in starts)
                + admission.bounded_workspace_bytes
                + admission.guard_bytes
            )
            if incremental > int(free_bytes):
                raise AdmissionError(
                    f"{phase.name} exact allocation and workspace require "
                    f"{incremental} free bytes but selected device has "
                    f"{int(free_bytes)}"
                )
        created: list[OwnerKey] = []
        try:
            for allocation in starts:
                if failure_injector is not None:
                    failure_injector(allocation)
                self._live[allocation.key] = MonotoneSlab(
                    allocation.schema,
                    self._allocation_capacity(allocation),
                    self.plan.manifest.n_replica,
                    self.plan.device,
                    failure_state=self.failure_state,
                    label=allocation.key.label,
                    allow_empty=(
                        allocation.key.owner in {"LayerWork", "FoldedRanges"}
                    ),
                )
                created.append(allocation.key)
        except BaseException:
            for key in reversed(created):
                self._live.pop(key).release()
            raise
        self._entered_phase = phase

    def slab(self, owner: str, partition: str = "") -> MonotoneSlab:
        key = OwnerKey(owner, partition)
        try:
            return self._live[key]
        except KeyError as exc:
            if key in self._released:
                raise StorageError(f"{key.label} has passed its last consumer") from exc
            raise StorageError(f"{key.label} is not live in this phase") from exc

    def release_after(self, phase: ExecutionPhase) -> tuple[OwnerKey, ...]:
        if phase is not self._entered_phase:
            raise StorageError("release boundary must match the entered phase")
        due = tuple(
            item.key
            for item in self.plan.allocations
            if item.last_consumer is phase
        )
        for key in due:
            try:
                slab = self._live.pop(key)
            except KeyError as exc:
                raise StorageError(f"{key.label} was not live at last consumer") from exc
            slab.release()
            self._released.add(key)
            self._release_counts[key] = self._release_counts.get(key, 0) + 1
        return due

    def abort_failed_lane(self, lane: int) -> DeviceFailure:
        failure = self.failure_state.snapshot(lane)
        if failure is None:
            raise StorageError("lane has no recorded failure")
        for slab in self._live.values():
            slab._prepare_reset_failed_lane(lane)
        for slab in self._live.values():
            slab.reset_failed_lane(lane)
        self.failure_state.clear_lane(lane)
        return failure

    def memory_report(self) -> MemoryReport:
        if self._entered_phase is None:
            raise StorageError("no execution phase has been entered")
        torch.cuda.synchronize(self.plan.device)
        admission = self.admission(self._entered_phase)
        allocated = self.failure_state.allocated_bytes + sum(
            slab.allocated_bytes for slab in self._live.values()
        )
        analytical = DeviceFailureState.analytical_bytes(
            self.plan.manifest.n_replica
        ) + sum(
            self._allocation_bytes(allocation)
            for allocation in self.plan.allocations
            if allocation.key in self._live
        )
        high_water = 0
        allocations = {item.key: item for item in self.plan.allocations}
        for key, slab in self._live.items():
            allocation = allocations[key]
            row_bytes = sum(field.row_bytes for field in allocation.schema.fields)
            high_water += sum(
                slab.high_water(lane) * row_bytes
                for lane in range(self.plan.manifest.n_replica)
            )
        report = MemoryReport(
            phase=self._entered_phase,
            configured_budget_bytes=admission.configured_budget_bytes,
            analytical_live_bytes=analytical,
            allocated_owner_bytes=allocated,
            high_water_payload_bytes=high_water,
            bounded_workspace_bytes=admission.bounded_workspace_bytes,
            guard_bytes=admission.guard_bytes,
            admitted_required_bytes=admission.required_bytes,
            observed_allocated_bytes=int(torch.cuda.memory_allocated(self.plan.device)),
            observed_reserved_bytes=int(torch.cuda.memory_reserved(self.plan.device)),
            observed_peak_allocated_bytes=int(
                torch.cuda.max_memory_allocated(self.plan.device)
            ),
            observed_peak_reserved_bytes=int(
                torch.cuda.max_memory_reserved(self.plan.device)
            ),
        )
        report.assert_reconciled()
        return report

    def close(self) -> None:
        if self._entered_phase is not ExecutionPhase.FINAL_BACKWARD:
            raise StorageError("owners may close only after final backward")
        due = {
            item.key
            for item in self.plan.allocations
            if item.last_consumer is ExecutionPhase.FINAL_BACKWARD
        }
        if due.intersection(self._live):
            raise StorageError("final-backward owners have not been released")
        if self._live:
            raise StorageError("owners remain live after their last consumer")
        if set(self._release_counts) != {item.key for item in self.plan.allocations}:
            raise StorageError("not every generated owner was released")
        if any(count != 1 for count in self._release_counts.values()):
            raise StorageError("a generated owner was not released exactly once")
        self.failure_state.release()
        self._released_failure_state = True


__all__ = [
    "BRANCHED_ALIAS_PEAK_ROW_BYTES",
    "BRANCHED_LOG_MASS_ROW_BYTES",
    "AdmissionError",
    "CapacityError",
    "CapacityManifest",
    "ExecutionGeometry",
    "ExecutionPhase",
    "GeneratedOwners",
    "HardwareLimits",
    "MemoryReport",
    "NamedShape",
    "OwnerAllocation",
    "OwnerKey",
    "PartitionRows",
    "PhaseAdmission",
    "PhysicalLayout",
    "StoragePlan",
    "build_storage_plan",
]
