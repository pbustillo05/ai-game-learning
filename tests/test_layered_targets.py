from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pbg.core.capacity import (
    CapacityManifest,
    ExecutionGeometry,
    ExecutionPhase,
    GeneratedOwners,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
    build_storage_plan,
)
from pbg.core.contracts import (
    ConversionSpec,
    Partition,
    PartitionContract,
    PartitionMap,
    PrivateEstimator,
    RolloutMode,
    TrainingRegime,
    mode_schema,
)
from pbg.core.storage import (
    DestinationRequest,
    MonotoneSlab,
    PublicationTransaction,
    SoAField,
)
from pbg.core.targets import (
    ConversionMap,
    DeviceLayerPlan,
    LayeredTargetExecutor,
    RBLeafWork,
    SampledLeafWork,
    TargetAlgebra,
    TargetError,
    rb_completion,
)
from pbg.core.topology import (
    LayerPlan,
    TargetKind,
    Topology,
    TopologyBindings,
    TopologyError,
    build_layer_plan,
)
from tests.static_gates import core_firewall_violations
from tests.storage_support import published_rows

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="queued targets require CUDA"
)
REPOSITORY = Path(__file__).resolve().parents[1]


def sampled_completion(
    payoff: torch.Tensor, *, branched: bool
) -> SampledLeafWork:
    if (
        payoff.device.type != "cuda"
        or payoff.dtype != torch.float32
        or payoff.ndim != 2
        or not payoff.is_contiguous()
    ):
        raise TargetError("sampled payoff must be CUDA FP32 [leaf, P_train]")
    return SampledLeafWork(
        return_value=payoff.unsqueeze(-1),
        trace_vector=payoff if branched else None,
    )


def initialize_work_arenas(
    owners: GeneratedOwners,
    plan: LayerPlan,
    completion: Mapping[
        tuple[int, str], SampledLeafWork | RBLeafWork
    ],
) -> None:
    n_replica = owners.plan.manifest.n_replica
    expected_inputs = tuple(
        item.name for item in owners.plan.partitions.inputs
    )
    expected_outputs = tuple(
        item.name for item in owners.plan.partitions.outputs
    )
    if plan.input_names != expected_inputs or plan.output_names != expected_outputs:
        raise TargetError("LayerPlan partitions differ from generated owners")
    if plan.branched != owners.plan.mode.rollout_mode.is_branched:
        raise TargetError("LayerPlan rollout family differs from generated owners")
    if len(plan.arena_rows) != n_replica:
        raise TargetError("LayerPlan replica count differs from generated owners")
    plan.assert_admitted(owners.plan.manifest)
    p_train = (
        owners.plan.layout.player_count
        if owners.plan.mode.training_regime is TrainingRegime.SELF_PLAY
        else 1
    )
    required_completion: set[tuple[int, str]] = set()
    for movement in range(plan.movement_count):
        lane = int(plan.movement_replica[movement])
        output = int(plan.movement_output_type[movement])
        required_completion.add((lane, plan.output_names[output]))
    if set(completion) != required_completion:
        raise TargetError("completion payload keys differ from movement plan")
    for lane in range(n_replica):
        requests: list[DestinationRequest] = []
        operations: list[
            tuple[
                int,
                str,
                torch.Tensor,
                torch.Tensor,
                SampledLeafWork | RBLeafWork | None,
            ]
        ] = []
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
            selected = (
                (plan.movement_replica == lane)
                & (plan.movement_output_type == output)
            )
            destinations = plan.movement_destination_work_row[selected].long()
            sources = plan.movement_source_leaf_row[selected].long()
            payload = completion.get((lane, name)) if destinations.numel() else None
            if destinations.numel():
                if not isinstance(payload, (SampledLeafWork, RBLeafWork)):
                    raise TargetError("completion payload type is not supported")
                expects_sampled = (
                    owners.plan.mode.private_estimator
                    is PrivateEstimator.SAMPLED
                )
                if expects_sampled != isinstance(payload, SampledLeafWork):
                    raise TargetError(
                        "completion payload estimator differs from LayerWork schema"
                    )
                leaf_rows = plan.leaf_rows[lane]
                width = plan.output_widths[output]
                work_shape = (leaf_rows, p_train, width)
                trace_shape = (leaf_rows, p_train)
                tensors = (
                    (("return_value", payload.return_value),)
                    if isinstance(payload, SampledLeafWork)
                    else (("mass", payload.mass), ("moment", payload.moment))
                )
                for field, tensor in tensors:
                    if (
                        tensor.device != owners.plan.device
                        or tensor.dtype != torch.float32
                        or tuple(tensor.shape) != work_shape
                        or not tensor.is_contiguous()
                    ):
                        raise TargetError(
                            f"completion {field} must be contiguous CUDA FP32 "
                            f"{work_shape} on {owners.plan.device}"
                        )
                trace = payload.trace_vector
                if plan.branched:
                    if (
                        trace is None
                        or trace.device != owners.plan.device
                        or trace.dtype != torch.float32
                        or tuple(trace.shape) != trace_shape
                        or not trace.is_contiguous()
                    ):
                        raise TargetError(
                            "branched completion trace must match its closed "
                            "CUDA FP32 leaf/player extent"
                        )
                elif trace is not None:
                    raise TargetError("sequential completion must omit trace_vector")
                finite = all(
                    bool(torch.isfinite(tensor).all())
                    for _, tensor in tensors
                )
                if trace is not None:
                    finite = finite and bool(torch.isfinite(trace).all())
                if not finite:
                    raise TargetError("completion payload must be finite")
                if isinstance(payload, RBLeafWork):
                    invalid_mass = bool((payload.mass < 0).any())
                    invalid_zero = bool(
                        (
                            (payload.mass == 0)
                            & (payload.moment.abs() > 1.0e-7)
                        ).any()
                    )
                    if invalid_mass or invalid_zero:
                        raise TargetError("RB completion violates mass/moment laws")
            operations.append((output, name, destinations, sources, payload))
        if not requests:
            continue
        transaction = PublicationTransaction.begin(
            requests, phase="completion_movement"
        )
        try:
            for index, (_, _, destinations, sources, payload) in enumerate(
                operations
            ):
                transaction.fill_defaults(index)
                if payload is None:
                    continue
                reservation = transaction.reservations[index]
                fields = tuple(
                    field.name for field in reservation._slab.schema.fields
                )
                sources_device = sources.to(
                    device=owners.plan.device, dtype=torch.int64
                )
                if isinstance(payload, SampledLeafWork):
                    if "return_value" not in fields:
                        raise TargetError(
                            "sampled completion used with RB work schema"
                        )
                    torch.index_select(
                        payload.return_value,
                        0,
                        sources_device,
                        out=reservation.view("return_value")[
                            : destinations.numel(), 0
                        ],
                    )
                    if "trace_vector" in fields:
                        if payload.trace_vector is None:
                            raise TargetError(
                                "branched completion lacks trace vector"
                            )
                        torch.index_select(
                            payload.trace_vector,
                            0,
                            sources_device,
                            out=reservation.view("trace_vector")[
                                : destinations.numel(), 0
                            ],
                        )
                else:
                    if "mass" not in fields:
                        raise TargetError(
                            "RB completion used with sampled work schema"
                        )
                    torch.index_select(
                        payload.mass,
                        0,
                        sources_device,
                        out=reservation.view("mass")[
                            : destinations.numel(), 0
                        ],
                    )
                    torch.index_select(
                        payload.moment,
                        0,
                        sources_device,
                        out=reservation.view("moment")[
                            : destinations.numel(), 0
                        ],
                    )
                    if "trace_vector" in fields:
                        if payload.trace_vector is None:
                            raise TargetError(
                                "branched completion lacks trace vector"
                            )
                        torch.index_select(
                            payload.trace_vector,
                            0,
                            sources_device,
                            out=reservation.view("trace_vector")[
                                : destinations.numel(), 0
                            ],
                        )
            transaction.publish()
        except BaseException:
            transaction.abort()
            raise


def _i32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def _partitions(estimator: PrivateEstimator) -> PartitionContract:
    if estimator is PrivateEstimator.SAMPLED:
        return PartitionContract(
            inputs=(Partition("pre", 1), Partition("post2", 1)),
            outputs=(Partition("all", 1),),
            folded=(Partition("pre", 2), Partition("post", 4)),
            input_to_output=(
                PartitionMap("pre", "all"),
                PartitionMap("post2", "all"),
            ),
            conversions=(),
        )
    return PartitionContract(
        inputs=(Partition("pre", 2), Partition("post2", 4)),
        outputs=(Partition("pre", 2), Partition("post", 4)),
        folded=(Partition("pre", 2), Partition("post", 4)),
        input_to_output=(
            PartitionMap("pre", "pre"),
            PartitionMap("post2", "post"),
        ),
        conversions=(ConversionSpec("post", "pre", "sum_classes"),),
    )


def _algebra(estimator: PrivateEstimator) -> TargetAlgebra:
    partitions = _partitions(estimator)
    maps = (
        ()
        if estimator is PrivateEstimator.SAMPLED
        else (ConversionMap("post", "pre", (0, 0, 1, 1)),)
    )
    return TargetAlgebra(partitions, maps)


def _layout() -> PhysicalLayout:
    return PhysicalLayout(
        model_input_shapes=(
            NamedShape("pre", (4,)),
            NamedShape("post2", (4,)),
        ),
        player_count=2,
        stratum_count=2,
        evaluator_private_width=4,
        opaque_fields=(
            SoAField("leaf_public_state", torch.uint8, (5,), fill=255),
            SoAField("leaf_private_profile", torch.int32, (2,), fill=-1),
        ),
    )


def _manifest(mode, device: torch.device) -> CapacityManifest:
    partitions = _partitions(mode.private_estimator)
    action_partitions = (
        partitions.outputs if mode.rollout_mode.is_branched else partitions.inputs
    )
    beta = 1 if mode.rollout_mode is RolloutMode.SEQUENTIAL else 3
    return CapacityManifest(
        geometry=ExecutionGeometry(2, 15, 64, 4096),
        root_count=1,
        n_replica=1,
        beta_max=beta,
        player_node_cap=(
            PartitionRows("pre", 16),
            PartitionRows("post2", 16),
        ),
        trained_player_cap=(
            PartitionRows("pre", 8),
            PartitionRows("post2", 8),
        ),
        action_cap=tuple(
            PartitionRows(partition.name, 16)
            for partition in action_partitions
        ),
        folded_range_cap=(
            PartitionRows("pre", 0),
            PartitionRows("post", 0),
        ),
        leaf_cap=8,
        topology_cap=8,
        topology_edge_cap=16,
        family_pack_cap=8,
        chance_frontier_cap=8,
        completion_pack_cap=8,
        learner_sigma_pack_cap=8,
        learner_action_pack_cap=8,
        topology_packet_cap=8,
        target_arena_cap=16,
        rollout_workspace_bytes=1024 * 1024,
        queue_drain_workspace_bytes=1024,
        completion_workspace_bytes=2048,
        align_workspace_bytes=1024 * 1024 + 1024,
        target_workspace_bytes=1024 * 1024 + 1024,
        learner_workspace_bytes=1024 * 1024,
        hardware=HardwareLimits(
            torch.cuda.get_device_properties(device).total_memory,
            64 * 1024 * 1024,
        ),
    )


def _topology(
    mode,
    *,
    unbalanced: bool = False,
    structural_unary: bool = False,
) -> tuple[Topology, TopologyBindings]:
    branched = mode.rollout_mode.is_branched
    if not branched:
        topology = Topology(
            parent=_i32([-1]),
            child_begin=_i32([0]),
            executed_degree=_i32([1]),
            nominal_degree=_i32([1]),
            target_degree=_i32([1]),
            owner=_i32([0]),
            familywork_row=_i32([0]),
            sigma_head_row=_i32([0]),
            action_begin=_i32([0]),
            label=_i32([7]),
            path_length=_i32([1]),
            target_kind=_i32([int(TargetKind.LEAF)]),
            target_row=_i32([0]),
        )
        bindings = TopologyBindings(
            input_type=_i32([0]),
            output_type=_i32([0]),
            sigma_target_row=_i32([0]),
            child_output_type=_i32([0]),
        )
        return topology, bindings
    if structural_unary:
        topology = Topology(
            parent=_i32([-1, 0]),
            child_begin=_i32([0, 1]),
            executed_degree=_i32([1, 1]),
            nominal_degree=_i32([1, 1]),
            target_degree=_i32([1, 1]),
            owner=_i32([0, -1]),
            familywork_row=_i32([0, -1]),
            sigma_head_row=_i32([0, -1]),
            action_begin=_i32([0, -1]),
            label=_i32([4, 9]),
            path_length=_i32([1, 2]),
            target_kind=_i32(
                [int(TargetKind.TOPOLOGY), int(TargetKind.LEAF)]
            ),
            target_row=_i32([1, 0]),
        )
        bindings = TopologyBindings(
            input_type=_i32([0, -1]),
            output_type=_i32([0, 0]),
            sigma_target_row=_i32([0, -1]),
            child_output_type=_i32([0, 0]),
        )
        return topology, bindings
    if unbalanced:
        topology = Topology(
            parent=_i32([-1, 0]),
            child_begin=_i32([0, 2]),
            executed_degree=_i32([2, 2]),
            nominal_degree=_i32([2, 2]),
            target_degree=_i32([2, 2]),
            owner=_i32([0, 0]),
            familywork_row=_i32([0, 1]),
            sigma_head_row=_i32([0, 1]),
            action_begin=_i32([0, 2]),
            label=_i32([3, 3, 5, 6]),
            path_length=_i32([1, 1, 1, 1]),
            target_kind=_i32(
                [
                    int(TargetKind.LEAF),
                    int(TargetKind.TOPOLOGY),
                    int(TargetKind.LEAF),
                    int(TargetKind.LEAF),
                ]
            ),
            target_row=_i32([0, 1, 1, 2]),
        )
        bindings = TopologyBindings(
            input_type=_i32([0, 0]),
            output_type=_i32([0, 0]),
            sigma_target_row=_i32([0, 1]),
            child_output_type=_i32([0, 0, 0, 0]),
        )
        return topology, bindings
    topology = Topology(
        parent=_i32([-1]),
        child_begin=_i32([0]),
        executed_degree=_i32([2]),
        nominal_degree=_i32([2]),
        target_degree=_i32([2]),
        owner=_i32([0]),
        familywork_row=_i32([0]),
        sigma_head_row=_i32([0]),
        action_begin=_i32([0]),
        label=_i32([7, 7]),  # duplicate labels retain distinct child identity
        path_length=_i32([1, 1]),
        target_kind=_i32([int(TargetKind.LEAF), int(TargetKind.LEAF)]),
        target_row=_i32([0, 1]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0]),
        output_type=_i32([0]),
        sigma_target_row=_i32([0]),
        child_output_type=_i32([0, 0]),
    )
    return topology, bindings


def _fill_heads(
    owners: GeneratedOwners, mode, rows: int, *, lane: int = 0
) -> None:
    slab = owners.slab("SigmaHead", "pre")
    reservation = slab.reserve(
        lane, rows, phase="model_commit", input_type="pre", logical_address=0
    )
    values = {
        "old_value": torch.tensor(
            [[2.0] * slab.schema.fields[0].shape[-1]] * rows,
            dtype=torch.float32,
            device=slab.device,
        )
    }
    if mode.rollout_mode.is_branched:
        values["old_trace"] = torch.full(
            (rows,), 0.25, dtype=torch.float32, device=slab.device
        )
        values["old_cost"] = torch.full(
            (rows,), 3.0, dtype=torch.float32, device=slab.device
        )
    reservation.fill(values)
    PublicationTransaction((reservation,)).publish()


def _enter_through_targets(
    owners: GeneratedOwners,
    mode,
    plan,
    completion,
    head_rows: int,
):
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    owners.bind_layer_work(plan.output_names, plan.arena_rows)
    if head_rows:
        for lane in range(owners.plan.manifest.n_replica):
            _fill_heads(owners, mode, head_rows, lane=lane)
    owners.release_after(ExecutionPhase.ROLLOUT)
    owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
    owners.release_after(ExecutionPhase.QUEUE_DRAIN)
    owners.enter_phase(ExecutionPhase.COMPLETION)
    for output, name in enumerate(plan.output_names):
        work = owners.slab("LayerWork", name)
        exact_rows = max(
            rows[output] for rows in plan.arena_rows
        )
        assert work.capacity == exact_rows
        assert work.allocated_bytes == MonotoneSlab.analytical_bytes(
            work.schema,
            exact_rows,
            owners.plan.manifest.n_replica,
        )
    initialize_work_arenas(owners, plan, completion)
    owners.release_after(ExecutionPhase.COMPLETION)
    owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
    owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)
    owners.enter_phase(ExecutionPhase.TARGETS)


def _close(owners: GeneratedOwners) -> None:
    owners.release_after(ExecutionPhase.TARGETS)
    owners.enter_phase(ExecutionPhase.LEARNER)
    owners.release_after(ExecutionPhase.LEARNER)
    owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
    owners.release_after(ExecutionPhase.FINAL_BACKWARD)
    owners.close()


def test_topology_compaction_preserves_unary_path_cost() -> None:
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology, bindings = _topology(mode, structural_unary=True)
    topology.validate(bindings, p_train=2)
    compact, compact_bindings = topology.compact(bindings, p_train=2)
    assert compact.node_count == 1
    assert compact.edge_count == 1
    assert compact.path_length.tolist() == [3]
    assert compact.label.tolist() == [4]
    assert compact.target_kind.tolist() == [int(TargetKind.LEAF)]
    compact.validate(compact_bindings, p_train=2, compact=True)
def test_layer_plan_handles_duplicates_unbalanced_last_consumers_and_dense_rows() -> None:
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology, bindings = _topology(mode, unbalanced=True)
    plan = build_layer_plan(
        mode, _partitions(mode.private_estimator), (topology,), (bindings,), p_train=2
    )
    assert plan.layer_count == 2
    assert plan.node_count == 2
    assert plan.edge_count == 4
    assert not hasattr(plan, "carry_count")
    assert plan.sigma_rows == ((2, 0),)
    assert plan.action_rows == ((4,),)
    assert plan.arena_rows[0][0] == 3


def test_protected_root_prefix_excludes_refill_child_from_targets() -> None:
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology = Topology(
        parent=_i32([-1]),
        child_begin=_i32([0]),
        executed_degree=_i32([3]),
        nominal_degree=_i32([2]),
        target_degree=_i32([2]),
        owner=_i32([0]),
        familywork_row=_i32([0]),
        sigma_head_row=_i32([0]),
        action_begin=_i32([0]),
        label=_i32([1, 2, 3]),
        path_length=_i32([1, 1, 1]),
        target_kind=_i32([int(TargetKind.LEAF)] * 3),
        target_row=_i32([0, 1, 2]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0]),
        output_type=_i32([0]),
        sigma_target_row=_i32([0]),
        child_output_type=_i32([0, 0, 0]),
    )
    plan = build_layer_plan(
        mode,
        _partitions(mode.private_estimator),
        (topology,),
        (bindings,),
        p_train=2,
    )
    assert plan.edge_count == 3
    assert plan.action_rows == ((2,),)
    assert plan.target_degree.tolist() == [2]


@CUDA
def test_protected_root_prefix_is_numerical_and_ignores_refill_child() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology = Topology(
        parent=_i32([-1]),
        child_begin=_i32([0]),
        executed_degree=_i32([3]),
        nominal_degree=_i32([2]),
        target_degree=_i32([2]),
        owner=_i32([0]),
        familywork_row=_i32([0]),
        sigma_head_row=_i32([0]),
        action_begin=_i32([0]),
        label=_i32([1, 2, 3]),
        path_length=_i32([1, 1, 1]),
        target_kind=_i32([int(TargetKind.LEAF)] * 3),
        target_row=_i32([0, 1, 2]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0]),
        output_type=_i32([0]),
        sigma_target_row=_i32([0]),
        child_output_type=_i32([0, 0, 0]),
    )
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "all"): sampled_completion(
                torch.tensor(
                    [[2.0, -2.0], [4.0, -4.0], [100.0, -100.0]],
                    device=device,
                ),
                branched=True,
            )
        },
        1,
    )
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=1.0
    ).execute(owners, plan)
    sigma = published_rows(
        owners.slab("SigmaTarget", "pre"), 0, "value_target"
    )
    action = published_rows(
        owners.slab("ActionTarget", "all"), 0, "advantage_target"
    )
    assert torch.equal(sigma[0], torch.tensor([3.0], device=device))
    assert action.shape[0] == 2
    assert result.counters.child_edges_read == 2
    _close(owners)


def test_layer_plan_rejects_unsupported_conversion_and_raw_addresses() -> None:
    mode = mode_schema("self_play", "sequential", "rao_blackwell")
    topology, bindings = _topology(mode)
    unsupported = replace(
        bindings,
        input_type=_i32([1]),
        output_type=_i32([1]),
        child_output_type=_i32([0]),
    )
    with pytest.raises(TopologyError, match="unsupported conversion"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (topology,),
            (unsupported,),
            p_train=2,
        )


def test_topology_rejects_overlapping_or_orphan_child_spans() -> None:
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology = Topology(
        parent=_i32([-1, -1]),
        child_begin=_i32([0, 0]),
        executed_degree=_i32([1, 1]),
        nominal_degree=_i32([1, 1]),
        target_degree=_i32([1, 1]),
        owner=_i32([0, 0]),
        familywork_row=_i32([0, 1]),
        sigma_head_row=_i32([0, 1]),
        action_begin=_i32([0, 1]),
        label=_i32([1, 2]),
        path_length=_i32([1, 1]),
        target_kind=_i32([int(TargetKind.LEAF), int(TargetKind.LEAF)]),
        target_row=_i32([0, 1]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0, 0]),
        output_type=_i32([0, 0]),
        sigma_target_row=_i32([0, 1]),
        child_output_type=_i32([0, 0]),
    )
    with pytest.raises(TopologyError, match="overlap"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (topology,),
            (bindings,),
            p_train=2,
        )
    orphaned = replace(
        topology,
        child_begin=_i32([0, 1]),
        label=_i32([1, 2, 3]),
        path_length=_i32([1, 1, 1]),
        target_kind=_i32([int(TargetKind.LEAF)] * 3),
        target_row=_i32([0, 1, 2]),
    )
    orphan_bindings = replace(
        bindings, child_output_type=_i32([0, 0, 0])
    )
    with pytest.raises(TopologyError, match="orphan"):
        orphaned.validate(orphan_bindings, p_train=2)


def test_layer_plan_rejects_duplicate_sigma_and_action_destinations() -> None:
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology = Topology(
        parent=_i32([-1, -1]),
        child_begin=_i32([0, 1]),
        executed_degree=_i32([1, 1]),
        nominal_degree=_i32([1, 1]),
        target_degree=_i32([1, 1]),
        owner=_i32([0, 0]),
        familywork_row=_i32([0, 1]),
        sigma_head_row=_i32([0, 1]),
        action_begin=_i32([0, 1]),
        label=_i32([1, 2]),
        path_length=_i32([1, 1]),
        target_kind=_i32([int(TargetKind.LEAF), int(TargetKind.LEAF)]),
        target_row=_i32([0, 1]),
    )
    duplicate_sigma = TopologyBindings(
        input_type=_i32([0, 0]),
        output_type=_i32([0, 0]),
        sigma_target_row=_i32([0, 0]),
        child_output_type=_i32([0, 0]),
    )
    with pytest.raises(TopologyError, match="sigma target rows must be unique"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (topology,),
            (duplicate_sigma,),
            p_train=2,
        )
    unique_sigma = replace(
        duplicate_sigma, sigma_target_row=_i32([0, 1])
    )
    with pytest.raises(TopologyError, match="action target spans"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (replace(topology, action_begin=_i32([0, 0])),),
            (unique_sigma,),
            p_train=2,
        )


@CUDA
def test_layer_plan_rejects_inflated_or_gapped_target_extents() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode,
        _partitions(mode.private_estimator),
        (topology,),
        (bindings,),
        p_train=2,
        manifest=manifest,
    )
    with pytest.raises(TopologyError, match="sigma targets.*exact extent"):
        replace(plan, sigma_rows=((2, 0),)).assert_admitted(manifest)
    with pytest.raises(TopologyError, match="action targets.*exact extent"):
        replace(plan, action_rows=((2, 0),)).assert_admitted(manifest)
    with pytest.raises(TopologyError, match="share one identity"):
        replace(
            plan,
            sigma_target_row=_i32([1]),
            sigma_rows=((2, 0),),
        ).assert_admitted(manifest)
    with pytest.raises(TopologyError, match="share one identity"):
        replace(
            plan,
            action_begin=_i32([1]),
            action_rows=((2, 0),),
        ).assert_admitted(manifest)


def test_layer_plan_rejects_nonpermutation_leaf_rows() -> None:
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    with pytest.raises(TopologyError, match="dense exact permutation"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (replace(topology, target_row=_i32([1])),),
            (bindings,),
            p_train=2,
        )


def test_sequential_rejects_nonunit_nominal_degree() -> None:
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    with pytest.raises(TopologyError, match="N=N_tilde=1"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (replace(topology, nominal_degree=_i32([2])),),
            (bindings,),
            p_train=2,
        )


def test_sequential_rejects_permuted_aligned_target_rows() -> None:
    mode = mode_schema("self_play", "sequential", "sampled")
    topology = Topology(
        parent=_i32([-1, -1]),
        child_begin=_i32([0, 1]),
        executed_degree=_i32([1, 1]),
        nominal_degree=_i32([1, 1]),
        target_degree=_i32([1, 1]),
        owner=_i32([0, 0]),
        familywork_row=_i32([0, 1]),
        sigma_head_row=_i32([0, 1]),
        action_begin=_i32([1, 0]),
        label=_i32([1, 2]),
        path_length=_i32([1, 1]),
        target_kind=_i32([int(TargetKind.LEAF), int(TargetKind.LEAF)]),
        target_row=_i32([0, 1]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0, 0]),
        output_type=_i32([0, 0]),
        sigma_target_row=_i32([0, 1]),
        child_output_type=_i32([0, 0]),
    )
    with pytest.raises(TopologyError, match="share one identity"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (topology,),
            (bindings,),
            p_train=2,
        )


@CUDA
def test_sequential_mutated_plan_rejects_permuted_row_identity() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode,
        _partitions(mode.private_estimator),
        (topology,),
        (bindings,),
        p_train=2,
        manifest=manifest,
    )
    with pytest.raises(TopologyError, match="share one identity"):
        replace(
            plan,
            sigma_head_row=_i32([1]),
            sigma_target_row=_i32([0]),
        ).assert_admitted(manifest)
    with pytest.raises(TopologyError, match="share one identity"):
        replace(
            plan,
            action_begin=_i32([1]),
            action_rows=((2, 0),),
        ).assert_admitted(manifest)


def test_sampled_completion_rejects_cpu_state() -> None:
    with pytest.raises(TargetError, match="CUDA FP32"):
        sampled_completion(torch.ones((1, 2)), branched=False)


@CUDA
def test_completion_payload_contract_rejects_broadcast_before_reservation() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    invalid = SampledLeafWork(torch.ones((1, 1, 1), device=device))
    with pytest.raises(TargetError, match="closed|must be contiguous"):
        _enter_through_targets(
            owners, mode, plan, {(0, "all"): invalid}, 1
        )
    assert owners.slab("LayerWork", "all").tail(0) == 0
    crossed = RBLeafWork(
        torch.ones((1, 2, 1), device=device),
        torch.ones((1, 2, 1), device=device),
    )
    with pytest.raises(TargetError, match="estimator differs"):
        initialize_work_arenas(
            owners, plan, {(0, "all"): crossed}
        )
    work = owners.slab("LayerWork", "all")
    assert work.tail(0) == 0
    assert work.reserved_tail(0) == 0
    initialize_work_arenas(
        owners,
        plan,
        {
            (0, "all"): sampled_completion(
                torch.ones((1, 2), device=device), branched=False
            )
        },
    )
    owners.release_after(ExecutionPhase.COMPLETION)
    owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
    owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)
    owners.enter_phase(ExecutionPhase.TARGETS)
    _close(owners)


@CUDA
def test_negative_post_mass_cannot_cancel_before_h_to_k_validation() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "rao_blackwell")
    topology, bindings = _topology(mode)
    bindings = replace(bindings, child_output_type=_i32([1]))
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    mass = torch.tensor(
        [[[-1.0, 2.0, 0.0, 0.0], [-1.0, 2.0, 0.0, 0.0]]],
        device=device,
    )
    invalid = RBLeafWork(mass, torch.zeros_like(mass))
    with pytest.raises(TargetError, match="mass/moment"):
        _enter_through_targets(
            owners, mode, plan, {(0, "post"): invalid}, 1
        )
    assert owners.slab("LayerWork", "post").tail(0) == 0
    valid_mass = mass.abs()
    initialize_work_arenas(
        owners,
        plan,
        {(0, "post"): RBLeafWork(valid_mass, torch.zeros_like(mass))},
    )
    owners.release_after(ExecutionPhase.COMPLETION)
    owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
    owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)
    owners.enter_phase(ExecutionPhase.TARGETS)
    _close(owners)


@CUDA
def test_configured_maximum_sequential_depth_is_admitted() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    nodes = 8
    topology = Topology(
        parent=_i32([-1, *range(nodes - 1)]),
        child_begin=_i32(range(nodes)),
        executed_degree=_i32([1] * nodes),
        nominal_degree=_i32([1] * nodes),
        target_degree=_i32([1] * nodes),
        owner=_i32([0] * nodes),
        familywork_row=_i32(range(nodes)),
        sigma_head_row=_i32(range(nodes)),
        action_begin=_i32(range(nodes)),
        label=_i32(range(nodes)),
        path_length=_i32([1] * nodes),
        target_kind=_i32(
            [int(TargetKind.TOPOLOGY)] * (nodes - 1)
            + [int(TargetKind.LEAF)]
        ),
        target_row=_i32([*range(1, nodes), 0]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0] * nodes),
        output_type=_i32([0] * nodes),
        sigma_target_row=_i32(range(nodes)),
        child_output_type=_i32([0] * nodes),
    )
    plan = build_layer_plan(
        mode,
        _partitions(mode.private_estimator),
        (topology,),
        (bindings,),
        p_train=2,
        manifest=_manifest(mode, device),
    )
    assert plan.layer_count == nodes


@CUDA
def test_branched_nominal_degree_cannot_exceed_beta_max() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology, bindings = _topology(mode)
    topology = replace(topology, nominal_degree=_i32([4]))
    with pytest.raises(TopologyError, match="child span"):
        build_layer_plan(
            mode,
            _partitions(mode.private_estimator),
            (topology,),
            (bindings,),
            p_train=2,
            manifest=_manifest(mode, device),
        )


@CUDA
def test_signed_sampled_return_is_valid() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    with pytest.raises(TopologyError, match="share one identity"):
        replace(plan, action_begin=_i32([16])).assert_admitted(manifest)
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "all"): sampled_completion(
                torch.tensor([[-3.0, 3.0]], device=device), branched=False
            )
        },
        1,
    )
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=0.5
    ).execute(owners, plan)
    assert result.published_lanes == (0,)
    assert torch.equal(
        published_rows(
            owners.slab("SigmaTarget", "pre"), 0, "value_target"
        )[0],
        torch.tensor([-3.0], device=device),
    )
    _close(owners)


@CUDA
@pytest.mark.parametrize(
    ("bad_mass", "bad_moment"),
    ((-1.0, -2.0), (0.0, 1.0)),
)
def test_rb_validates_every_work_channel(
    bad_mass: float, bad_moment: float
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "rao_blackwell")
    topology, bindings = _topology(mode)
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    mass = torch.ones((1, 2, 2), device=device)
    moment = torch.full((1, 2, 2), 2.0, device=device)
    _enter_through_targets(
        owners,
        mode,
        plan,
        {(0, "pre"): RBLeafWork(mass, moment)},
        1,
    )
    work = owners.slab("LayerWork", "pre")
    work.columns["mass"][0, 0, 0, 1, 0] = bad_mass
    work.columns["moment"][0, 0, 0, 1, 0] = bad_moment
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=0.5
    ).execute(owners, plan)
    assert result.failed_lanes == (0,)
    assert owners.slab("SigmaTarget", "pre").tail(0) == 0
    owners.abort_failed_lane(0)
    _close(owners)


@CUDA
@pytest.mark.parametrize(
    "regime", (TrainingRegime.SELF_PLAY, TrainingRegime.BEST_RESPONSE)
)
@pytest.mark.parametrize(
    "rollout",
    (
        RolloutMode.SEQUENTIAL,
        RolloutMode.BRANCHED_FIXED,
        RolloutMode.BRANCHED_KKT,
    ),
)
@pytest.mark.parametrize(
    "estimator",
    (PrivateEstimator.SAMPLED, PrivateEstimator.RAO_BLACKWELL),
)
def test_all_twelve_modes_match_direct_one_node_reference(
    regime, rollout, estimator
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema(regime, rollout, estimator)
    topology, bindings = _topology(mode)
    partitions = _partitions(estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode,
        partitions,
        (topology,),
        (bindings,),
        p_train=2 if regime is TrainingRegime.SELF_PLAY else 1,
        manifest=manifest,
    )
    storage_plan = build_storage_plan(
        mode, partitions, manifest, _layout(), device
    )
    owners = GeneratedOwners(storage_plan)
    p_train = 2 if regime is TrainingRegime.SELF_PLAY else 1
    leaves = 1 if rollout is RolloutMode.SEQUENTIAL else 2
    if estimator is PrivateEstimator.SAMPLED:
        payoff = torch.arange(
            1, leaves * p_train + 1, dtype=torch.float32, device=device
        ).reshape(leaves, p_train)
        payload: SampledLeafWork | RBLeafWork = sampled_completion(
            payoff, branched=rollout.is_branched
        )
    else:
        mass = torch.ones(
            (leaves, p_train, 2), dtype=torch.float32, device=device
        )
        value = torch.arange(
            1,
            leaves * p_train * 2 + 1,
            dtype=torch.float32,
            device=device,
        ).reshape(leaves, p_train, 2)
        payload = rb_completion(
            mass,
            value,
            branched=rollout.is_branched,
            algebra=_algebra(estimator),
            source_output="pre",
        )
    _enter_through_targets(
        owners, mode, plan, {(0, plan.output_names[0]): payload}, 1
    )
    result = LayeredTargetExecutor(
        mode, _algebra(estimator), lambda_=0.5
    ).execute(owners, plan)
    assert result.published_lanes == (0,)
    assert result.failed_lanes == ()
    assert result.counters.layer_count == 1
    assert result.counters.scratch_bytes == LayeredTargetExecutor(
        mode, _algebra(estimator), lambda_=0.5
    ).required_scratch_bytes(plan, manifest.n_replica)
    assert result.counters.scratch_bytes <= manifest.target_workspace_bytes
    sigma = published_rows(
        owners.slab("SigmaTarget", "pre"), 0, "value_target"
    )
    action_name = "pre" if not rollout.is_branched else plan.output_names[0]
    advantage = published_rows(
        owners.slab("ActionTarget", action_name), 0, "advantage_target"
    )
    if estimator is PrivateEstimator.SAMPLED:
        owner_payoff = payoff[:, 0]
        assert torch.allclose(sigma[0], owner_payoff.mean().reshape(1))
        expected = []
        for child in range(leaves):
            baseline = (
                torch.tensor(2.0, device=device)
                if leaves == 1
                else 0.5 * owner_payoff[1 - child] + 1.0
            )
            expected.append(owner_payoff[child] - baseline)
        assert torch.allclose(
            advantage[:, 0], torch.stack(expected)
        )
    else:
        owner_value = value[:, 0]
        assert torch.allclose(sigma[0], owner_value.mean(0))
        posterior = published_rows(
            owners.slab("ActionTarget", action_name), 0, "posterior"
        )
        assert torch.equal(posterior, torch.ones_like(posterior))
        expected = []
        for child in range(leaves):
            baseline = (
                torch.full((2,), 2.0, device=device)
                if leaves == 1
                else 0.5 * owner_value[1 - child]
                + torch.ones(2, device=device)
            )
            expected.append(owner_value[child] - baseline)
        assert torch.allclose(advantage, torch.stack(expected))
    if rollout.is_branched:
        trace = published_rows(
            owners.slab("SigmaTarget", "pre"), 0, "trace_target"
        )
        cost = published_rows(
            owners.slab("SigmaTarget", "pre"), 0, "cost_target"
        )
        trace_vector = payload.trace_vector
        assert trace_vector is not None
        expected_trace = (
            (trace_vector - trace_vector.mean(0, keepdim=True)).square().sum()
            / (leaves - 1)
        )
        assert torch.allclose(trace, expected_trace.reshape(1))
        assert torch.equal(cost, torch.ones_like(cost))
        work = owners.slab("LayerWork", plan.output_names[0])
        destination = int(plan.destination_work_row[0])
        arena = 1
        assert float(work.columns["cost"][0, destination, arena]) == 2.0
        assert float(work.columns["delta"][0, destination, arena]) == 0.0
    _close(owners)


@CUDA
def test_unbalanced_last_consumer_slots_match_reference_without_copy_traffic() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology, bindings = _topology(mode, unbalanced=True)
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    payoff = torch.tensor(
        [[2.0, -2.0], [4.0, -4.0], [8.0, -8.0]], device=device
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {(0, "all"): sampled_completion(payoff, branched=True)},
        2,
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        acc_events=True,
    ) as profile:
        result = LayeredTargetExecutor(
            mode, _algebra(mode.private_estimator), lambda_=1.0
        ).execute(owners, plan)
    layer_kernels = [
        event
        for event in profile.events()
        if "layered_target_kernel" in event.name
    ]
    assert len(layer_kernels) == plan.layer_count
    assert not hasattr(result.counters, "carry_count")
    assert not hasattr(result.counters, "carry_payload_bytes")
    sigma = published_rows(
        owners.slab("SigmaTarget", "pre"), 0, "value_target"
    )
    # Node 1 is mean(4,8)=6, root is mean(2,6)=4.
    assert torch.equal(sigma[:, 0], torch.tensor([4.0, 6.0], device=device))
    _close(owners)


@CUDA
def test_unbalanced_rb_two_layer_recurrence_matches_direct_reference() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "branched.KKT", "rao_blackwell")
    topology, bindings = _topology(mode, unbalanced=True)
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    values = torch.tensor(
        [
            [[2.0, 2.0], [2.0, 2.0]],
            [[4.0, 4.0], [4.0, 4.0]],
            [[8.0, 8.0], [8.0, 8.0]],
        ],
        device=device,
    )
    mass = torch.ones_like(values)
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "pre"): rb_completion(
                mass,
                values,
                branched=True,
                algebra=_algebra(mode.private_estimator),
                source_output="pre",
            )
        },
        2,
    )
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=1.0
    ).execute(owners, plan)
    assert result.published_lanes == (0,)
    sigma = published_rows(
        owners.slab("SigmaTarget", "pre"), 0, "value_target"
    )
    assert torch.equal(
        sigma,
        torch.tensor([[4.0, 4.0], [6.0, 6.0]], device=device),
    )
    assert result.counters.mass_moment_bytes_read == 256
    assert result.counters.mass_moment_bytes_written == 64
    _close(owners)


@CUDA
def test_sequential_two_level_player_clock_specialization() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology = Topology(
        parent=_i32([-1, 0]),
        child_begin=_i32([0, 1]),
        executed_degree=_i32([1, 1]),
        nominal_degree=_i32([1, 1]),
        target_degree=_i32([1, 1]),
        owner=_i32([0, 0]),
        familywork_row=_i32([0, 1]),
        sigma_head_row=_i32([0, 1]),
        action_begin=_i32([0, 1]),
        label=_i32([1, 2]),
        path_length=_i32([1, 1]),
        target_kind=_i32(
            [int(TargetKind.TOPOLOGY), int(TargetKind.LEAF)]
        ),
        target_row=_i32([1, 0]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0, 0]),
        output_type=_i32([0, 0]),
        sigma_target_row=_i32([0, 1]),
        child_output_type=_i32([0, 0]),
    )
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "all"): sampled_completion(
                torch.tensor([[10.0, -10.0]], device=device), branched=False
            )
        },
        2,
    )
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=0.5
    ).execute(owners, plan)
    assert result.counters.layer_count == 2
    sigma = published_rows(
        owners.slab("SigmaTarget", "pre"), 0, "value_target"
    )
    advantage = published_rows(
        owners.slab("ActionTarget", "pre"), 0, "advantage_target"
    )
    # Descendant target is 10, then propagation bootstraps to 6 for its parent.
    assert torch.equal(sigma[:, 0], torch.tensor([6.0, 10.0], device=device))
    assert torch.equal(
        advantage[:, 0], torch.tensor([4.0, 8.0], device=device)
    )
    _close(owners)


@CUDA
def test_headless_frozen_structural_row_never_accesses_heads_or_targets() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("best_response", "branched.fixed", "sampled")
    topology = Topology(
        parent=_i32([-1]),
        child_begin=_i32([0]),
        executed_degree=_i32([2]),
        nominal_degree=_i32([2]),
        target_degree=_i32([2]),
        owner=_i32([-1]),
        familywork_row=_i32([-1]),
        sigma_head_row=_i32([-1]),
        action_begin=_i32([-1]),
        label=_i32([1, 1]),
        path_length=_i32([1, 1]),
        target_kind=_i32([int(TargetKind.LEAF), int(TargetKind.LEAF)]),
        target_row=_i32([0, 1]),
    )
    bindings = TopologyBindings(
        input_type=_i32([-1]),
        output_type=_i32([0]),
        sigma_target_row=_i32([-1]),
        child_output_type=_i32([0, 0]),
    )
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=1, manifest=manifest
    )
    assert plan.sigma_rows == ((0, 0),)
    assert plan.action_rows == ((0,),)
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "all"): sampled_completion(
                torch.tensor([[1.0], [3.0]], device=device), branched=True
            )
        },
        0,
    )
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=0.5
    ).execute(owners, plan)
    assert result.published_lanes == (0,)
    assert owners.slab("SigmaTarget", "pre").tail(0) == 0
    assert owners.slab("ActionTarget", "all").tail(0) == 0
    _close(owners)


@CUDA
def test_post_to_pre_reduces_mass_and_moment_before_conditionalization() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    algebra = _algebra(PrivateEstimator.RAO_BLACKWELL)
    mass = torch.tensor(
        [[[1.0, 3.0, 0.0, 0.0]]], dtype=torch.float32, device=device
    )
    value = torch.tensor(
        [[[10.0, 2.0, 100.0, 200.0]]],
        dtype=torch.float32,
        device=device,
    )
    payload = rb_completion(
        mass,
        value,
        branched=False,
        algebra=algebra,
        source_output="post",
        destination_output="pre",
    )
    assert torch.equal(payload.mass, torch.tensor([[[4.0, 0.0]]], device=device))
    assert torch.equal(
        payload.moment, torch.tensor([[[16.0, 0.0]]], device=device)
    )


@CUDA
def test_internal_layer_post_to_pre_conversion_is_deterministic() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "rao_blackwell")
    topology, bindings = _topology(mode)
    bindings = replace(bindings, child_output_type=_i32([1]))
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    mass = torch.tensor(
        [[[1.0, 3.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]]],
        device=device,
    )
    value = torch.tensor(
        [[[10.0, 2.0, 100.0, 200.0], [10.0, 2.0, 100.0, 200.0]]],
        device=device,
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "post"): rb_completion(
                mass,
                value,
                branched=False,
                algebra=_algebra(mode.private_estimator),
                source_output="post",
            )
        },
        1,
    )
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=1.0
    ).execute(owners, plan)
    assert result.published_lanes == (0,)
    sigma = published_rows(
        owners.slab("SigmaTarget", "pre"), 0, "value_target"
    )
    posterior = published_rows(
        owners.slab("ActionTarget", "pre"), 0, "posterior"
    )
    assert torch.equal(sigma[0], torch.tensor([4.0, 0.0], device=device))
    assert torch.equal(posterior[0], torch.tensor([4.0, 0.0], device=device))
    _close(owners)


@CUDA
def test_executor_rejects_mutated_heads_and_schema_mismatch_prelaunch() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "all"): sampled_completion(
                torch.tensor([[1.0, -1.0]], device=device), branched=False
            )
        },
        1,
    )
    unpublished = replace(plan, sigma_head_row=_i32([3]))
    with pytest.raises(TopologyError, match="share one identity"):
        LayeredTargetExecutor(
            mode, _algebra(mode.private_estimator), lambda_=0.5
        ).execute(owners, unpublished)
    wrong_mode = mode_schema("best_response", "sequential", "sampled")
    with pytest.raises(TargetError, match="mode differs"):
        LayeredTargetExecutor(
            wrong_mode, _algebra(mode.private_estimator), lambda_=0.5
        ).execute(owners, plan)
    _close(owners)


@CUDA
def test_failure_is_isolated_across_replica_axis() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    partitions = _partitions(mode.private_estimator)
    manifest = replace(_manifest(mode, device), n_replica=2)
    plan = build_layer_plan(
        mode,
        partitions,
        (topology, topology),
        (bindings, bindings),
        p_train=2,
        manifest=manifest,
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {
            (0, "all"): sampled_completion(
                torch.tensor([[3.0, -3.0]], device=device), branched=False
            ),
            (1, "all"): sampled_completion(
                torch.tensor([[1.0, 1.0]], device=device),
                branched=False,
            ),
        },
        1,
    )
    owners.slab("LayerWork", "all").columns["return_value"][
        1, 0, 0, 0, 0
    ] = float("nan")
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=0.5
    ).execute(owners, plan)
    assert result.published_lanes == (0,)
    assert result.failed_lanes == (1,)
    slab = owners.slab("SigmaTarget", "pre")
    assert slab.tail(0) == 1
    assert slab.tail(1) == 0
    owners.abort_failed_lane(1)
    _close(owners)


@CUDA
def test_direct_sibling_sum_avoids_total_minus_child_cancellation() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.tensor(
        [[[1.0e8], [1.0], [1.0], [-1.0e8]]],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones((1, 4), dtype=torch.bool, device=device)
    ordinal = torch.arange(4, device=device)
    sibling = mask[:, None, :] & (
        ordinal[None, :, None] != ordinal[None, None, :]
    )
    direct = (
        values[:, None] * sibling[..., None]
    ).sum(dim=2)
    total_minus = values.sum(1, keepdim=True) - values
    # For the middle child the direct logical-order sum is exactly zero, while
    # total-minus-child loses the unit term in the total and returns -1.
    assert float(direct[0, 1, 0]) == 0.0
    assert float(total_minus[0, 1, 0]) == -1.0


@CUDA
def test_cuda_executor_uses_direct_sibling_sum_for_concentrated_rows() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology = Topology(
        parent=_i32([-1]),
        child_begin=_i32([0]),
        executed_degree=_i32([3]),
        nominal_degree=_i32([3]),
        target_degree=_i32([3]),
        owner=_i32([0]),
        familywork_row=_i32([0]),
        sigma_head_row=_i32([0]),
        action_begin=_i32([0]),
        label=_i32([0, 1, 2]),
        path_length=_i32([1, 1, 1]),
        target_kind=_i32([int(TargetKind.LEAF)] * 3),
        target_row=_i32([0, 1, 2]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0]),
        output_type=_i32([0]),
        sigma_target_row=_i32([0]),
        child_output_type=_i32([0, 0, 0]),
    )
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode,
        partitions,
        (topology,),
        (bindings,),
        p_train=2,
        manifest=manifest,
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    payoff = torch.tensor(
        (
            (1.0e8, 0.0),
            (1.0, 0.0),
            (-1.0e8, 0.0),
        ),
        dtype=torch.float32,
        device=device,
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {(0, "all"): sampled_completion(payoff, branched=True)},
        1,
    )
    LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=1.0
    ).execute(owners, plan)
    advantage = published_rows(
        owners.slab("ActionTarget", "all"), 0, "advantage_target"
    )
    assert float(advantage[1, 0]) == 1.0
    _close(owners)


@CUDA
def test_failure_keeps_targets_unpublished_and_records_lane() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    partitions = _partitions(mode.private_estimator)
    manifest = _manifest(mode, device)
    plan = build_layer_plan(
        mode, partitions, (topology,), (bindings,), p_train=2, manifest=manifest
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, _layout(), device)
    )
    payoff = torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=device)
    _enter_through_targets(
        owners,
        mode,
        plan,
        {(0, "all"): sampled_completion(payoff, branched=False)},
        1,
    )
    owners.slab("LayerWork", "all").columns["return_value"][
        0, 0, 0, 0, 0
    ] = float("nan")
    result = LayeredTargetExecutor(
        mode, _algebra(mode.private_estimator), lambda_=0.5
    ).execute(owners, plan)
    assert result.published_lanes == ()
    assert result.failed_lanes == (0,)
    assert owners.slab("SigmaTarget", "pre").tail(0) == 0
    assert owners.slab("ActionTarget", "pre").tail(0) == 0
    failure = owners.failure_state.snapshot(0)
    assert failure is not None
    assert failure.phase == "targets"
    owners.abort_failed_lane(0)
    _close(owners)


def test_executor_has_no_python_node_or_action_loop_or_interlayer_sync() -> None:
    source = inspect.getsource(LayeredTargetExecutor)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            text = ast.get_source_segment(source, node) or ""
            assert "range(plan.node_count)" not in text
            assert "range(plan.edge_count)" not in text
            assert "range(target_count)" not in text
    execute_source = inspect.getsource(LayeredTargetExecutor.execute)
    prefix = execute_source.split("start_events[1].synchronize()", 1)[0]
    assert ".item()" not in prefix
    assert ".cpu()" not in prefix
    assert ".tolist()" not in prefix
    device_plan_source = inspect.getsource(DeviceLayerPlan)
    assert "LayerPlanCarry" not in device_plan_source
    assert "carry_ptr" not in device_plan_source
    assert "carry" not in inspect.getsource(LayerPlan).lower()


def test_queued_firewall_remains_closed() -> None:
    assert core_firewall_violations(REPOSITORY / "pbg" / "core") == ()
