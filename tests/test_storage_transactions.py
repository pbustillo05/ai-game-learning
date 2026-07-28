from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from pbg.core.capacity import (
    AdmissionError,
    CapacityError,
    CapacityManifest,
    ExecutionGeometry,
    ExecutionPhase,
    ExternalWorkspaceBounds,
    GeneratedOwners,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
    build_storage_plan,
)
from pbg.core.contracts import (
    CAPACITY_VOCABULARY,
    MODE_SCHEMAS,
    ConversionSpec,
    Partition,
    PartitionContract,
    PartitionMap,
    PrivateEstimator,
    RolloutMode,
)
from pbg.core.storage import (
    DestinationRequest,
    DeviceDestinationRequest,
    DeviceFailure,
    DeviceFailureState,
    ExactBlock,
    MonotoneSlab,
    PublicationTransaction,
    SoAField,
    SoASchema,
    StorageError,
    StorageOverflow,
)
from tests.static_gates import core_firewall_violations


REPOSITORY = Path(__file__).resolve().parents[1]
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for queued storage"
)


def _partitions(estimator: PrivateEstimator) -> PartitionContract:
    outputs = (
        (Partition("all", 1),)
        if estimator is PrivateEstimator.SAMPLED
        else (Partition("pre", 16), Partition("post", 28))
    )
    mappings = (
        (
            PartitionMap("pre", "all"),
            PartitionMap("post2", "all"),
        )
        if estimator is PrivateEstimator.SAMPLED
        else (
            PartitionMap("pre", "pre"),
            PartitionMap("post2", "post"),
        )
    )
    conversions = (
        ()
        if estimator is PrivateEstimator.SAMPLED
        else (ConversionSpec("post", "pre", "sum_classes"),)
    )
    return PartitionContract(
        inputs=(Partition("pre", 16), Partition("post2", 28)),
        outputs=outputs,
        folded=(Partition("pre", 16), Partition("post", 28)),
        input_to_output=mappings,
        conversions=conversions,
    )


def _layout() -> PhysicalLayout:
    return PhysicalLayout(
        model_input_shapes=(
            NamedShape("pre", (12,)),
            NamedShape("post2", (14,)),
        ),
        player_count=2,
        stratum_count=3,
        evaluator_private_width=28,
        family_workspace_row_bytes=64,
        chance_frontier_row_bytes=48,
        completion_workspace_row_bytes=96,
        learner_sigma_workspace_row_bytes=64,
        learner_action_workspace_row_bytes=48,
        topology_packet_row_bytes=32,
        opaque_fields=(
            SoAField("leaf_public_state", torch.uint8, (5,), fill=255),
            SoAField("leaf_private_profile", torch.int32, (2,), fill=-1),
        ),
    )


def _manifest(
    mode,
    *,
    device: torch.device,
    memory_budget_bytes: int = 64 * 1024 * 1024,
) -> CapacityManifest:
    action_names = (
        ("pre", "post2")
        if mode.action_partition_axis == "input"
        else tuple(item.name for item in _partitions(mode.private_estimator).outputs)
    )
    beta = 1 if mode.rollout_mode is RolloutMode.SEQUENTIAL else 2
    action_rows = 32 if beta == 1 else 96
    return CapacityManifest(
        geometry=ExecutionGeometry(
            actor_count=2,
            occurrence_bound=15,
            model_token_bound=100,
            model_token_pair_bound=10_000,
        ),
        root_count=2,
        n_replica=2,
        beta_max=beta,
        player_node_cap=(
            PartitionRows("pre", 40),
            PartitionRows("post2", 40),
        ),
        trained_player_cap=(
            PartitionRows("pre", 20),
            PartitionRows("post2", 20),
        ),
        action_cap=tuple(
            PartitionRows(name, action_rows) for name in action_names
        ),
        folded_range_cap=(
            PartitionRows("pre", 4),
            PartitionRows("post", 4),
        ),
        leaf_cap=8,
        topology_cap=8,
        topology_edge_cap=16,
        family_pack_cap=8,
        chance_frontier_cap=8,
        completion_pack_cap=4,
        learner_sigma_pack_cap=8,
        learner_action_pack_cap=8,
        topology_packet_cap=8,
        target_arena_cap=16,
        movement_scratch_bytes=4096,
        library_workspace_bytes=2048,
        evaluator_workspace_bytes=2048,
        memory_guard_bytes=4096,
        hardware=HardwareLimits(
            total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
            memory_budget_bytes=memory_budget_bytes,
            training_max_tokens=1024,
            training_max_token_pairs=1_000_000,
            query_chunk_size=128,
        ),
        required_external=ExternalWorkspaceBounds(
            library_bytes=1024,
            evaluator_bytes=1024,
        ),
    )


def _values(
    slab: MonotoneSlab, count: int, value: int | float = 1
) -> dict[str, torch.Tensor]:
    return {
        field.name: torch.full(
            (count, *field.shape),
            value,
            dtype=field.dtype,
            device=slab.device,
        )
        for field in slab.schema.fields
    }


def _small_schema(name: str) -> SoASchema:
    return SoASchema(
        name=name,
        fields=(
            SoAField("index", torch.int32, alignment=64, fill=-1),
            SoAField("value", torch.float32, (3,), alignment=64),
        ),
        lifetime="test_commit_to_test_consumer",
    )


@CUDA
def test_generated_owners_reset_every_live_slab_after_device_rollback() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = MODE_SCHEMAS[0]
    plan = build_storage_plan(
        mode,
        _partitions(mode.private_estimator),
        _manifest(mode, device=device),
        _layout(),
        device,
    )
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    leaf = owners.slab("Leaf")
    counts = torch.tensor([1, 0], dtype=torch.int64, device=device)
    values = {
        field.name: torch.full(
            (2, 1, *field.shape),
            field.fill,
            dtype=field.dtype,
            device=device,
        )
        for field in leaf.schema.fields
    }
    transaction = PublicationTransaction.begin_device(
        (
            DeviceDestinationRequest(
                leaf,
                counts,
                values,
                "pre",
                torch.tensor([7, 8], dtype=torch.int64, device=device),
            ),
        ),
        phase="rollout_commit",
        active=torch.tensor([True, False], device=device),
    )
    transaction.fill_all()
    transaction.publish()
    transaction.rollback_published()
    failure = owners.failure_state.snapshot(0)
    assert failure is not None
    assert failure.failure_code == "transaction_rollback"
    owners.abort_failed_lane(0)
    assert all(slab.epoch(0) == 1 for slab in owners.live_slabs)
    assert all(slab.epoch(1) == 0 for slab in owners.live_slabs)
    for slab in owners.live_slabs:
        slab.release()
    owners.failure_state.release()


def test_schema_and_capacity_records_are_deeply_immutable() -> None:
    schema = _small_schema("sample")
    with pytest.raises(FrozenInstanceError):
        schema.lifetime = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        schema.fields[0].shape = (2,)  # type: ignore[misc]

    assert schema.payload_bytes(7, 2) == 7 * 2 * (4 + 3 * 4)
    assert schema.allocation_bytes(7, 2) >= schema.payload_bytes(7, 2)
    assert schema.allocation_bytes(7, 2) % schema.alignment == 0


@CUDA
def test_all_twelve_modes_allocate_only_their_generated_cuda_owners() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    for mode in MODE_SCHEMAS:
        manifest = _manifest(mode, device=device)
        plan = build_storage_plan(
            mode,
            _partitions(mode.private_estimator),
            manifest,
            _layout(),
            device,
        )
        assert manifest.configured_names == frozenset(
            (
                *CAPACITY_VOCABULARY.partitioned_rows,
                *CAPACITY_VOCABULARY.scalar_rows,
                *CAPACITY_VOCABULARY.pack_limits,
                *CAPACITY_VOCABULARY.workspaces,
            )
        )
        declared = {owner.name for owner in mode.owners}
        assert plan.cuda_owner_names == declared - {"Topology", "TopologyChild"}
        assert set(plan.deferred_cpu_owners) == {"Topology", "TopologyChild"}
        assert plan.cuda_owner_names | set(plan.deferred_cpu_owners) == declared
        assert len({item.key for item in plan.allocations}) == len(plan.allocations)
        for allocation in plan.allocations:
            owner = mode.owner(allocation.key.owner)
            assert tuple(field.name for field in allocation.schema.fields) == owner.fields
            assert allocation.schema.lifetime == owner.lifetime
            assert allocation.analytical_bytes > 0

        owners = GeneratedOwners(plan)
        seen: set = set()
        for phase in ExecutionPhase:
            owners.enter_phase(phase)
            seen.update(owners.live_keys)
            report = owners.memory_report()
            report.assert_reconciled()
            owners.release_after(phase)
        assert seen == {item.key for item in plan.allocations}
        assert owners.released_keys == seen
        owners.close()


@CUDA
def test_monotone_slab_uses_one_aligned_arena_and_never_repeats_addresses() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    failures = DeviceFailureState(2, device)
    slab = MonotoneSlab(
        _small_schema("monotone"), 5, 2, device, failure_state=failures
    )
    with pytest.raises(AttributeError, match="immutable"):
        slab.capacity = 6  # type: ignore[misc]
    assert slab.allocated_bytes == MonotoneSlab.analytical_bytes(
        slab.schema, 5, 2
    )
    backing_pointers = {
        value.untyped_storage().data_ptr() for value in slab.columns.values()
    }
    assert len(backing_pointers) == 1
    for field in slab.schema.fields:
        assert slab.columns[field.name].data_ptr() % field.alignment == 0

    first = slab.reserve(
        0,
        2,
        phase="rollout",
        input_type="pre",
        logical_address=10,
    )
    first.fill(_values(slab, 2, 3))
    PublicationTransaction((first,)).publish()
    second = slab.reserve(
        0,
        3,
        phase="rollout",
        input_type="pre",
        logical_address=11,
    )
    second.fill(_values(slab, 3, 4))
    PublicationTransaction((second,)).publish()
    assert first.address == (first.address.__class__)(0, 0, 2)
    assert second.address == (second.address.__class__)(0, 2, 5)
    assert slab.tail(0) == slab.reserved_tail(0) == slab.high_water(0) == 5
    assert torch.equal(
        slab.published(0, "index"),
        torch.tensor([3, 3, 4, 4, 4], dtype=torch.int32, device=device),
    )
    with pytest.raises(ValueError, match="positive"):
        slab.reserve(
            1,
            0,
            phase="rollout",
            input_type="pre",
            logical_address=12,
        )
    slab.release()
    failures.release()


@CUDA
def test_transaction_failure_injection_publishes_every_destination_or_none() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    failures = DeviceFailureState(1, device)
    left = MonotoneSlab(
        _small_schema("left"), 4, 1, device, failure_state=failures
    )
    right = MonotoneSlab(
        _small_schema("right"), 4, 1, device, failure_state=failures
    )
    transaction = PublicationTransaction.begin(
        (
            DestinationRequest(left, 0, 2, "pre", 20),
            DestinationRequest(right, 0, 2, "pre", 20),
        ),
        phase="commit",
    )
    transaction.fill(0, _values(left, 2, 5))
    transaction.fill(1, _values(right, 2, 6))
    with pytest.raises(RuntimeError, match="injected"):
        transaction.publish(
            failure_injector=lambda: (_ for _ in ()).throw(
                RuntimeError("injected publication failure")
            )
        )
    assert left.tail(0) == right.tail(0) == 0
    assert left.reserved_tail(0) == right.reserved_tail(0) == 2
    left.reset_failed_lane(0)
    right.reset_failed_lane(0)
    assert left.epoch(0) == right.epoch(0) == 1

    committed = PublicationTransaction.begin(
        (
            DestinationRequest(left, 0, 1, "pre", 21),
            DestinationRequest(right, 0, 1, "pre", 21),
        ),
        phase="commit",
    )
    committed.fill(0, _values(left, 1, 7))
    committed.fill(1, _values(right, 1, 8))
    committed.publish()
    assert left.tail(0) == right.tail(0) == 1
    assert int(left.published(0, "index")[0]) == 7
    assert int(right.published(0, "index")[0]) == 8
    left.release()
    right.release()
    failures.release()


@CUDA
def test_overflow_selects_minimum_address_without_partial_reservation() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    failures = DeviceFailureState(2, device)
    left = MonotoneSlab(
        _small_schema("overflow_left"), 2, 2, device, failure_state=failures
    )
    right = MonotoneSlab(
        _small_schema("overflow_right"), 1, 2, device, failure_state=failures
    )
    with pytest.raises(StorageOverflow) as raised:
        PublicationTransaction.begin(
            (
                DestinationRequest(left, 0, 3, "pre", 90),
                DestinationRequest(right, 0, 2, "post2", 30),
            ),
            phase="reserve",
            sampler_attempt_count=7,
        )
    assert raised.value.failure.logical_address == 30
    assert raised.value.failure.input_type == "post2"
    assert raised.value.failure.sampler_attempt_count == 7
    assert left.tail(0) == left.reserved_tail(0) == 0
    assert right.tail(0) == right.reserved_tail(0) == 0
    assert failures.snapshot(0) == raised.value.failure
    assert failures.snapshot(1) is None
    left.reset_failed_lane(0)
    right.reset_failed_lane(0)
    failures.clear_lane(0)
    left.release()
    right.release()
    failures.release()


@CUDA
def test_concurrent_failure_batch_keeps_each_lanes_minimum_address() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    failures = DeviceFailureState(2, device)
    failures.record_many(
        (
            DeviceFailure(0, 1, "commit", "overflow", "post2", 40, 4, 1),
            DeviceFailure(1, 1, "sample", "exhausted", "pre", 22, 9, 0),
            DeviceFailure(0, 1, "reserve", "overflow", "pre", 7, 5, 2),
            DeviceFailure(1, 1, "commit", "overflow", "post2", 30, 8, 1),
        )
    )
    assert failures.snapshot(0).logical_address == 7  # type: ignore[union-attr]
    assert failures.snapshot(1).logical_address == 22  # type: ignore[union-attr]
    failures.record(
        DeviceFailure(0, 1, "later", "overflow", "pre", 99, 1, 0)
    )
    assert failures.snapshot(0).logical_address == 7  # type: ignore[union-attr]
    failures.release()


@CUDA
def test_failed_lane_retry_does_not_corrupt_completed_lane() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    failures = DeviceFailureState(2, device)
    slab = MonotoneSlab(
        _small_schema("lanes"), 2, 2, device, failure_state=failures
    )
    completed = slab.reserve(
        1,
        2,
        phase="commit",
        input_type="pre",
        logical_address=1,
    )
    completed.fill(_values(slab, 2, 9))
    PublicationTransaction((completed,)).publish()
    lane_one_before = slab.published(1, "value").clone()

    with pytest.raises(StorageOverflow):
        slab.reserve(
            0,
            3,
            phase="commit",
            input_type="pre",
            logical_address=2,
        )
    slab.reset_failed_lane(0)
    failures.clear_lane(0)
    assert slab.epoch(0) == 1
    assert slab.tail(0) == 0
    assert slab.tail(1) == 2
    assert torch.equal(slab.published(1, "value"), lane_one_before)
    retry = slab.reserve(
        0,
        1,
        phase="commit",
        input_type="pre",
        logical_address=3,
    )
    assert retry.address.epoch == 1
    retry.fill(_values(slab, 1, 10))
    PublicationTransaction((retry,)).publish()
    assert slab.high_water(0, lifetime=True) == 1
    slab.release()
    failures.release()


@CUDA
def test_generated_owner_retry_discards_whole_lane_attempt_only() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = MODE_SCHEMAS[0]
    plan = build_storage_plan(
        mode,
        _partitions(mode.private_estimator),
        _manifest(mode, device=device),
        _layout(),
        device,
    )
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    family = owners.slab("FamilyWork", "pre")
    completed = family.reserve(
        1,
        1,
        phase="commit",
        input_type="pre",
        logical_address=1,
    )
    completed.fill(_values(family, 1, 3))
    PublicationTransaction((completed,)).publish()
    lane_one_before = family.published(1, "model_input").clone()

    leaf = owners.slab("Leaf")
    with pytest.raises(StorageOverflow):
        leaf.reserve(
            0,
            leaf.capacity + 1,
            phase="commit",
            input_type="pre",
            logical_address=8,
        )
    failure = owners.abort_failed_lane(0)
    assert failure.logical_address == 8
    for key in owners.live_keys:
        slab = owners.slab(key.owner, key.partition)
        assert slab.epoch(0) == 1
        assert slab.tail(0) == 0
    assert family.epoch(1) == 0
    assert family.tail(1) == 1
    assert torch.equal(family.published(1, "model_input"), lane_one_before)

    owners.release_after(ExecutionPhase.ROLLOUT)
    for phase in tuple(ExecutionPhase)[1:]:
        owners.enter_phase(phase)
        owners.release_after(phase)
    owners.close()


@CUDA
def test_exact_block_has_immutable_extent_exact_bytes_and_explicit_release() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    block = ExactBlock(_small_schema("exact"), 3, device)
    with pytest.raises(AttributeError, match="immutable"):
        block.rows = 4  # type: ignore[misc]
    assert block.rows == 3
    assert block.allocated_bytes == block.schema.allocation_bytes(3)
    assert block.payload_bytes == block.schema.payload_bytes(3)
    assert len(
        {value.untyped_storage().data_ptr() for value in block.columns.values()}
    ) == 1
    with pytest.raises(TypeError):
        block.columns["extra"] = torch.empty(0, device=device)  # type: ignore[index]
    block.release()
    with pytest.raises(StorageError, match="released"):
        _ = block.columns
    with pytest.raises(StorageError, match="released"):
        block.release()


@CUDA
def test_manifest_rejects_uncovered_geometry_hardware_and_workspaces() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = MODE_SCHEMAS[0]
    manifest = _manifest(mode, device=device)
    partitions = _partitions(mode.private_estimator)
    layout = _layout()

    with pytest.raises(CapacityError, match="player_node_cap"):
        build_storage_plan(
            mode,
            partitions,
            replace(
                manifest,
                player_node_cap=(
                    PartitionRows("pre", 2),
                    PartitionRows("post2", 2),
                ),
            ),
            layout,
            device,
        )
    with pytest.raises(CapacityError, match="partitions"):
        build_storage_plan(
            mode,
            partitions,
            replace(
                manifest,
                folded_range_cap=(PartitionRows("pre", 8),),
            ),
            layout,
            device,
        )
    with pytest.raises(CapacityError, match="token limit"):
        build_storage_plan(
            mode,
            partitions,
            replace(
                manifest,
                hardware=replace(manifest.hardware, training_max_tokens=50),
            ),
            layout,
            device,
        )
    with pytest.raises(CapacityError, match="workspace"):
        replace(manifest, evaluator_workspace_bytes=512)
    with pytest.raises(AdmissionError):
        build_storage_plan(
            mode,
            partitions,
            _manifest(mode, device=device, memory_budget_bytes=1024),
            layout,
            device,
        )


@CUDA
def test_phase_allocation_is_transactional_and_release_is_at_last_consumer() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = MODE_SCHEMAS[-1]
    plan = build_storage_plan(
        mode,
        _partitions(mode.private_estimator),
        _manifest(mode, device=device),
        _layout(),
        device,
    )
    owners = GeneratedOwners(plan)
    calls = 0

    def fail_third(_allocation) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected allocation failure")

    with pytest.raises(RuntimeError, match="injected allocation"):
        owners.enter_phase(ExecutionPhase.ROLLOUT, failure_injector=fail_third)
    assert owners.live_keys == frozenset()

    owners.enter_phase(ExecutionPhase.ROLLOUT)
    queue_key = next(
        item.key for item in plan.allocations if item.key.owner == "QueueMetadata"
    )
    owners.slab(queue_key.owner, queue_key.partition)
    owners.release_after(ExecutionPhase.ROLLOUT)
    owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
    with pytest.raises(StorageError, match="explicitly released"):
        owners.enter_phase(ExecutionPhase.COMPLETION)
    owners.release_after(ExecutionPhase.QUEUE_DRAIN)
    with pytest.raises(StorageError, match="last consumer"):
        owners.slab(queue_key.owner, queue_key.partition)

    for phase in tuple(ExecutionPhase)[2:]:
        owners.enter_phase(phase)
        owners.release_after(phase)
    owners.close()


def test_manifest_vocabulary_and_firewall_remain_closed() -> None:
    assert len(CAPACITY_VOCABULARY.partitioned_rows) == 4
    assert len(CAPACITY_VOCABULARY.scalar_rows) == 3
    assert len(CAPACITY_VOCABULARY.pack_limits) == 6
    assert len(CAPACITY_VOCABULARY.workspaces) == 5
    names = frozenset(
        (
            *CAPACITY_VOCABULARY.partitioned_rows,
            *CAPACITY_VOCABULARY.scalar_rows,
            *CAPACITY_VOCABULARY.pack_limits,
            *CAPACITY_VOCABULARY.workspaces,
        )
    )
    assert len(names) == 18
    assert core_firewall_violations(REPOSITORY / "pbg" / "core") == ()
    source = (REPOSITORY / "pbg" / "core" / "storage.py").read_text(
        encoding="utf-8"
    )
    assert "class Paged" not in source
    assert "class Recyclable" not in source
    assert "def grow" not in source
