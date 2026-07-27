from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pbg.queued import (
    CapacityManifest,
    ConversionMap,
    ConversionSpec,
    DestinationRequest,
    DeviceLayerPlan,
    ExecutionPhase,
    ExternalWorkspaceBounds,
    GameGeometry,
    GeneratedOwners,
    HardwareLimits,
    LayeredTargetExecutor,
    NamedShape,
    Partition,
    PartitionContract,
    PartitionMap,
    PartitionRows,
    PhysicalLayout,
    PrivateEstimator,
    PublicationTransaction,
    RBLeafWork,
    RolloutMode,
    SampledLeafWork,
    TargetAlgebra,
    TargetError,
    TargetKind,
    Topology,
    TopologyBindings,
    TopologyError,
    TopologyPacket,
    TrainingRegime,
    assemble_topology_packets,
    build_layer_plan,
    build_storage_plan,
    initialize_work_arenas,
    mode_schema,
    rb_completion,
    sampled_completion,
)
from pbg.queued.static_gates import queued_firewall_violations


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="queued targets require CUDA"
)
REPOSITORY = Path(__file__).resolve().parents[1]


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
        family_workspace_row_bytes=32,
        chance_frontier_row_bytes=32,
        completion_workspace_row_bytes=64,
        learner_sigma_workspace_row_bytes=32,
        learner_action_workspace_row_bytes=32,
        topology_packet_row_bytes=64,
    )


def _manifest(mode, device: torch.device) -> CapacityManifest:
    partitions = _partitions(mode.private_estimator)
    action_partitions = (
        partitions.outputs if mode.rollout_mode.is_branched else partitions.inputs
    )
    beta = 1 if mode.rollout_mode is RolloutMode.SEQUENTIAL else 3
    return CapacityManifest(
        geometry=GameGeometry(2, 8.0, 4, 64, 4096),
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
            PartitionRows("pre", 2),
            PartitionRows("post", 2),
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
        movement_scratch_bytes=1024 * 1024,
        library_workspace_bytes=1024,
        evaluator_workspace_bytes=1024,
        memory_guard_bytes=1024,
        hardware=HardwareLimits(
            torch.cuda.get_device_properties(device).total_memory,
            64 * 1024 * 1024,
            1024,
            1_000_000,
            64,
        ),
        required_external=ExternalWorkspaceBounds(0, 0),
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
    if head_rows:
        for lane in range(owners.plan.manifest.n_replica):
            _fill_heads(owners, mode, head_rows, lane=lane)
    owners.release_after(ExecutionPhase.ROLLOUT)
    owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
    owners.release_after(ExecutionPhase.QUEUE_DRAIN)
    owners.enter_phase(ExecutionPhase.COMPLETION)
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


def test_topology_packets_compact_unary_rows_and_preserve_path_cost() -> None:
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology, bindings = _topology(mode, structural_unary=True)
    packet = TopologyPacket(0, 0, 0, 0, topology, bindings)
    compact, compact_bindings = assemble_topology_packets(
        (packet,), p_train=2
    )
    assert compact.node_count == 1
    assert compact.edge_count == 1
    assert compact.path_length.tolist() == [3]
    assert compact.label.tolist() == [4]
    assert compact.target_kind.tolist() == [int(TargetKind.LEAF)]
    compact.validate(compact_bindings, p_train=2, compact=True)


def test_topology_packets_reject_partial_commit_and_sequence_gap() -> None:
    mode = mode_schema("self_play", "sequential", "sampled")
    topology, bindings = _topology(mode)
    with pytest.raises(TopologyError, match="committed"):
        TopologyPacket(1, 0, 0, 0, topology, bindings)
    packet = TopologyPacket(1, 0, 0, 1, topology, bindings)
    with pytest.raises(TopologyError, match="sequence"):
        assemble_topology_packets((packet,), p_train=2)


def test_layer_plan_handles_duplicates_unbalanced_last_consumers_and_dense_rows() -> None:
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    topology, bindings = _topology(mode, unbalanced=True)
    plan = build_layer_plan(
        mode, _partitions(mode.private_estimator), (topology,), (bindings,), p_train=2
    )
    assert plan.layer_count == 2
    assert plan.node_count == 2
    assert plan.edge_count == 4
    assert plan.carry_count == 0
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
    with pytest.raises(TopologyError, match="action target span"):
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
        owners.slab("SigmaTarget", "pre").published(0, "value_target")[0],
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
    mass = torch.tensor(
        [[[1.0, 1.0], [bad_mass, 1.0]]], device=device
    )
    moment = torch.tensor(
        [[[2.0, 2.0], [bad_moment, 2.0]]], device=device
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {(0, "pre"): RBLeafWork(mass, moment)},
        1,
    )
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
    assert result.counters.scratch_bytes <= manifest.movement_scratch_bytes
    sigma = owners.slab("SigmaTarget", "pre").published(0, "value_target")
    action_name = "pre" if not rollout.is_branched else plan.output_names[0]
    advantage = owners.slab(
        "ActionTarget", action_name
    ).published(0, "advantage_target")
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
        posterior = owners.slab(
            "ActionTarget", action_name
        ).published(0, "posterior")
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
        trace = owners.slab("SigmaTarget", "pre").published(
            0, "trace_target"
        )
        cost = owners.slab("SigmaTarget", "pre").published(
            0, "cost_target"
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
def test_unbalanced_last_consumer_slots_match_reference_without_carry_traffic() -> None:
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
    assert result.counters.carry_count == 0
    assert result.counters.carry_payload_bytes == 0
    sigma = owners.slab("SigmaTarget", "pre").published(0, "value_target")
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
    sigma = owners.slab("SigmaTarget", "pre").published(0, "value_target")
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
    sigma = owners.slab("SigmaTarget", "pre").published(0, "value_target")
    advantage = owners.slab(
        "ActionTarget", "pre"
    ).published(0, "advantage_target")
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
    sigma = owners.slab("SigmaTarget", "pre").published(0, "value_target")
    posterior = owners.slab("ActionTarget", "pre").published(0, "posterior")
    assert torch.equal(sigma[0], torch.tensor([4.0, 0.0], device=device))
    assert torch.equal(posterior[0], torch.tensor([4.0, 0.0], device=device))
    _close(owners)


@CUDA
def test_executor_rejects_unpublished_heads_and_schema_mismatch_prelaunch() -> None:
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
    with pytest.raises(TargetError, match="unpublished sigma head"):
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
                torch.tensor([[float("nan"), 1.0]], device=device),
                branched=False,
            ),
        },
        1,
    )
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
    payoff = torch.tensor(
        [[float("nan"), 1.0]], dtype=torch.float32, device=device
    )
    _enter_through_targets(
        owners,
        mode,
        plan,
        {(0, "all"): sampled_completion(payoff, branched=False)},
        1,
    )
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


def test_queued_firewall_remains_closed() -> None:
    assert queued_firewall_violations(REPOSITORY / "pbg" / "queued") == ()
