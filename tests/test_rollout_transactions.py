from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess
import sys
import textwrap
from types import MappingProxyType

import pytest
import torch

from pbg.core import (
    AsyncTopologyPublisher,
    BoundedChanceClosure,
    ChanceFrontier,
    CommitCoverage,
    CommitPlan,
    CompletionFIFO,
    CompletionPlan,
    DeviceDestinationRequest,
    DeviceFailureState,
    PublicationTransaction,
    DrawPurpose,
    EndpointPatch,
    FamilyQueueState,
    FamilyScheduler,
    LOGICAL_RNG_VERSION,
    MonotoneSlab,
    Partition,
    PartitionContract,
    PartitionMap,
    RolloutTransaction,
    RolloutOwnerRegistry,
    SampleWorkspace,
    SemanticFamilyFIFO,
    SoAField,
    SoASchema,
    StorageError,
    Topology,
    TopologyBindings,
    TopologyDraftPacket,
    TopologyError,
    TopologyPublication,
    child_path_keys,
    device_exclusive_scan,
    logical_categorical,
    logical_rng_keys,
    logical_uniform,
    mode_schema,
    stable_route,
)


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="queued rollout requires CUDA"
)


def _schema(name: str, fields: tuple[SoAField, ...]) -> SoASchema:
    return SoASchema(name, fields, "test_rollout_lifetime")


def _family_schema(name: str = "pre") -> SoASchema:
    return _schema(
        f"FamilyWork.{name}",
        (
            SoAField("logical_rng_key", torch.int64, (2,)),
            SoAField("model_input", torch.float32, (2,)),
            SoAField("owner", torch.int32, fill=-1),
        ),
    )


def _metadata_schema(name: str = "pre") -> SoASchema:
    return _schema(
        f"QueueMetadata.{name}",
        (
            SoAField("head", torch.int32),
            SoAField("ready_tail", torch.int32),
            SoAField("reserved_tail", torch.int32),
        ),
    )


def _partitions() -> PartitionContract:
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


def _values(
    slab: MonotoneSlab,
    max_rows: int,
    *,
    base: int = 1,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for offset, field in enumerate(slab.schema.fields):
        values[field.name] = torch.full(
            (slab.n_replica, max_rows, *field.shape),
            base + offset,
            dtype=field.dtype,
            device=slab.device,
        )
    return values


def _request(
    slab: MonotoneSlab,
    counts: torch.Tensor,
    *,
    max_rows: int,
    base: int = 1,
    logical: int = 10,
    input_type: str = "pre",
) -> DeviceDestinationRequest:
    return DeviceDestinationRequest(
        slab=slab,
        counts=counts,
        values=_values(slab, max_rows, base=base),
        input_type=input_type,
        logical_address=torch.arange(
            logical,
            logical + slab.n_replica,
            dtype=torch.int64,
            device=slab.device,
        ),
    )


def _slab(
    schema: SoASchema,
    capacity: int,
    failures: DeviceFailureState,
) -> MonotoneSlab:
    return MonotoneSlab(
        schema,
        capacity,
        failures.n_replica,
        failures.device,
        failure_state=failures,
    )


def _registry(
    slabs: tuple[MonotoneSlab, ...],
    *,
    family_pack_cap: int = 8,
    chance_frontier_cap: int = 8,
    completion_pack_cap: int = 8,
    topology_packet_cap: int = 8,
    topology_cap: int = 32,
    topology_edge_cap: int = 32,
) -> RolloutOwnerRegistry:
    return RolloutOwnerRegistry.for_testing(
        slabs,
        family_pack_cap=family_pack_cap,
        chance_frontier_cap=chance_frontier_cap,
        completion_pack_cap=completion_pack_cap,
        topology_packet_cap=topology_packet_cap,
        topology_cap=topology_cap,
        topology_edge_cap=topology_edge_cap,
    )


def _coverage(
    counts: torch.Tensor,
    rows: int,
    *,
    destination: int = 0,
) -> CommitCoverage:
    ordinal = torch.arange(rows, dtype=torch.int64, device=counts.device)
    ordinal = ordinal.expand(counts.numel(), rows)
    valid = ordinal < counts[:, None]
    return CommitCoverage(
        valid=valid,
        source_ordinal=ordinal,
        destination=torch.full_like(ordinal, destination),
        destination_ordinal=ordinal,
    )


def _i32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, pin_memory=True)


def _draft(
    sequence: int = 0,
    *,
    device: torch.device | str = "cuda",
    target_ordinal: int = 0,
) -> tuple[TopologyDraftPacket, EndpointPatch]:
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
        target_kind=_i32([-1]),
        target_row=_i32([-1]),
    )
    bindings = TopologyBindings(
        input_type=_i32([0]),
        output_type=_i32([0]),
        sigma_target_row=_i32([0]),
        child_output_type=_i32([-1]),
    )
    draft = TopologyDraftPacket(
        sequence=sequence,
        node_begin=0,
        edge_begin=0,
        topology=topology,
        bindings=bindings,
        unresolved_edges=(0,),
        declared_node_count=1,
        declared_edge_count=1,
        validation_event=torch.cuda.Event(blocking=False),
        ready_event=torch.cuda.Event(blocking=False),
        device_valid=torch.ones(
            1, dtype=torch.bool, device=device
        ),
    )
    draft.validation_event.record(torch.cuda.current_stream(device))
    draft.ready_event.record(torch.cuda.current_stream(device))
    patch = EndpointPatch(
        sequence=sequence,
        edge=_i32([0]),
        target_kind=_i32([1]),
        reservation_index=_i32([0]),
        preflight_target_row=_i32([0]),
        target_ordinal=torch.tensor(
            [target_ordinal], dtype=torch.int64, device=device
        ),
        child_output_type=_i32([0]),
        commit_sequence=sequence,
    )
    return draft, patch


def test_scheduler_is_full_first_preorder_load_age_and_never_pads_lanes() -> None:
    fixed = mode_schema("self_play", "branched.fixed", "sampled")
    kkt = mode_schema("self_play", "branched.KKT", "sampled")
    scheduler = FamilyScheduler(("pre", "post2"), 4)
    states = (
        FamilyQueueState(0, "post2", 1, 0, 6, 9, 3, fixed),
        FamilyQueueState(1, "pre", 0, 0, 3, 9, 0, kkt),
        FamilyQueueState(2, "post2", 1, 2, 8, 9, 1, kkt),
        FamilyQueueState(3, "post2", 1, 0, 2, 9, 0, fixed),
    )
    selected = scheduler.select(states, allow_partial=True)
    assert selected is not None
    assert selected.input_type == "post2"
    assert selected.rows == 4
    assert selected.lanes == (0, 2)
    assert selected.lane_schemas == (fixed, kkt)
    assert not selected.forced_partial

    partial = scheduler.select(states[1:2] + states[3:4], allow_partial=True)
    assert partial is not None
    assert partial.input_type == "pre"
    assert partial.rows == 3
    assert partial.lanes == (1,)
    assert partial.forced_partial
    assert scheduler.select(states[1:2], allow_partial=False) is None

    bad = FamilyQueueState(0, "other", 0, 0, 1, 1, 0, fixed)
    with pytest.raises(Exception, match="undeclared queue"):
        scheduler.select((bad,), allow_partial=True)


def test_scheduler_rejects_a_physically_incompatible_cohort() -> None:
    sampled = mode_schema("self_play", "sequential", "sampled")
    rb = mode_schema("self_play", "sequential", "rao_blackwell")
    scheduler = FamilyScheduler(("pre",), 2)
    states = (
        FamilyQueueState(0, "pre", 0, 0, 2, 2, 0, sampled),
        FamilyQueueState(1, "pre", 0, 0, 2, 2, 0, rb),
    )
    with pytest.raises(Exception, match="physically incompatible"):
        scheduler.select(states, allow_partial=True)


@CUDA
def test_device_counts_scans_reservations_and_publication_is_lane_atomic() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(3, device)
    left = _slab(
        _schema("left", (SoAField("value", torch.int32),)),
        4,
        failures,
    )
    right = _slab(
        _schema("right", (SoAField("value", torch.int32),)),
        4,
        failures,
    )
    prior_counts = torch.tensor([0, 1, 0], dtype=torch.int64, device=device)
    prior = PublicationTransaction.begin_device(
        (_request(right, prior_counts, max_rows=1, base=3),),
        phase="prior_commit",
        active=torch.tensor([False, True, False], device=device),
    )
    prior.fill_all()
    prior.publish()
    left_counts = torch.tensor([2, 2, 1], dtype=torch.int64, device=device)
    right_counts = torch.tensor([2, 4, 1], dtype=torch.int64, device=device)
    transaction = PublicationTransaction.begin_device(
        (
            _request(left, left_counts, max_rows=2, base=5),
            _request(right, right_counts, max_rows=4, base=7),
        ),
        phase="rollout_commit",
        active=torch.ones(3, dtype=torch.bool, device=device),
    )
    transaction.fill_all()
    marker = torch.full((3,), -1, dtype=torch.int64, device=device)
    success = transaction.publish(commit_marker=marker, sequence=11)
    torch.cuda.synchronize(device)

    assert success.tolist() == [True, False, True]
    assert marker.tolist() == [11, -1, 11]
    assert [left.tail(lane) for lane in range(3)] == [2, 0, 1]
    assert [right.tail(lane) for lane in range(3)] == [2, 1, 1]
    assert left.published(0, "value").tolist() == [5, 5]
    assert left.published(2, "value").tolist() == [5]
    assert right.published(0, "value").tolist() == [7, 7]
    assert right.published(2, "value").tolist() == [7]
    failure = failures.snapshot(1)
    assert failure is not None
    assert failure.logical_address == 11
    assert failure.requested_extent == 4
    assert failure.available_extent == 3
    assert failure.input_type == "pre"
    assert left.published(1, "value").numel() == 0
    assert right.published(1, "value").tolist() == [3]

    left.reset_failed_lane(1)
    right.reset_failed_lane(1)
    failures.clear_lane(1)
    assert left.epoch(1) == right.epoch(1) == 1
    left.release()
    right.release()
    failures.release()

    pack_failures = DeviceFailureState(1, device)
    packed = _slab(
        _schema("packed", (SoAField("value", torch.int32),)),
        4,
        pack_failures,
    )
    invalid_count = torch.tensor([3], dtype=torch.int64, device=device)
    invalid = PublicationTransaction.begin_device(
        (_request(packed, invalid_count, max_rows=2, logical=40),),
        phase="rollout_commit",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    invalid.fill_all()
    assert not invalid.publish().item()
    assert packed.tail(0) == 0
    failure = pack_failures.snapshot(0)
    assert failure is not None
    assert failure.requested_extent == 3
    assert failure.available_extent == 2
    packed.reset_failed_lane(0)
    pack_failures.clear_lane(0)
    packed.release()
    pack_failures.release()

    boundary_failures = DeviceFailureState(1, device)
    boundary = _slab(
        _schema("boundary", (SoAField("value", torch.int32),)),
        4,
        boundary_failures,
    )
    two = torch.tensor([2], dtype=torch.int64, device=device)
    initial = PublicationTransaction.begin_device(
        (_request(boundary, two, max_rows=2, base=3),),
        phase="initial",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    initial.fill_all()
    initial.publish()
    final = PublicationTransaction.begin_device(
        (_request(boundary, two, max_rows=4, base=9),),
        phase="final",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    final.fill_all()
    final.publish()
    assert boundary.published(0, "value").tolist() == [3, 3, 9, 9]
    boundary.release()
    boundary_failures.release()


@CUDA
def test_device_stable_route_declares_every_row_once_without_reordering() -> None:
    device = torch.device("cuda", 0)
    family = torch.tensor(
        [[2, 1, 2, 3, 1], [3, 3, 1, 2, 2]],
        dtype=torch.int32,
        device=device,
    )
    active = torch.tensor(
        [[True, True, True, False, True], [True, False, True, True, True]],
        device=device,
    )
    route = stable_route(family, active, (1, 2, 3))
    assert route.counts.tolist() == [[2, 2, 0], [1, 2, 1]]
    assert route.offsets.tolist() == [[0, 2, 4], [0, 1, 3]]
    for lane in range(2):
        observed: list[int] = []
        for family_index in range(3):
            rows = route.source_row[lane, family_index][
                route.valid[lane, family_index]
            ].tolist()
            assert rows == sorted(rows)
            observed.extend(rows)
        expected = torch.arange(5, device=device)[active[lane]].tolist()
        assert sorted(observed) == expected
    offsets, totals = device_exclusive_scan(route.counts)
    assert torch.equal(offsets, route.offsets)
    assert totals.tolist() == [4, 4]


@CUDA
def test_device_publication_failure_injection_exposes_none_and_poisons() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(1, device)
    slab = _slab(
        _schema("destination", (SoAField("value", torch.int32),)),
        3,
        failures,
    )
    counts = torch.tensor([2], dtype=torch.int64, device=device)
    transaction = PublicationTransaction.begin_device(
        (_request(slab, counts, max_rows=2),),
        phase="rollout_commit",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    transaction.fill_all()
    with pytest.raises(RuntimeError, match="injected"):
        transaction.publish(
            failure_injector=lambda: (_ for _ in ()).throw(
                RuntimeError("injected")
            )
        )
    assert slab.tail(0) == slab.reserved_tail(0) == 0
    with pytest.raises(StorageError, match="poisoned"):
        slab.preflight(0, 1)
    slab.reset_failed_lane(0)
    assert slab.epoch(0) == 1
    slab.release()
    failures.release()


@CUDA
def test_device_materialization_has_no_hidden_nonzero_or_scalar_sync() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(2, device)
    slab = _slab(
        _schema("profiled", (SoAField("value", torch.int32),)),
        8,
        failures,
    )
    counts = torch.tensor([4, 3], dtype=torch.int64, device=device)
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        acc_events=True,
    ) as profile:
        transaction = PublicationTransaction.begin_device(
            (_request(slab, counts, max_rows=4),),
            phase="rollout_commit",
            active=torch.ones(2, dtype=torch.bool, device=device),
        )
        transaction.fill_all()
        transaction.publish()
        torch.cuda.synchronize(device)
    keys = {event.key for event in profile.key_averages()}
    assert "aten::nonzero" not in keys
    assert "aten::_local_scalar_dense" not in keys
    names = tuple(event.name.lower() for event in profile.events())
    assert not any(
        "memcpy dtoh" in name
        or "device to host" in name
        or "memcpy htod" in name
        or "host to device" in name
        for name in names
    )
    slab.release()
    failures.release()


@CUDA
def test_undeclared_device_route_fails_terminally_without_skipping() -> None:
    script = """
import torch
from pbg.core import stable_route
family = torch.tensor([[1, 99, 2]], dtype=torch.int32, device="cuda")
active = torch.ones_like(family, dtype=torch.bool)
stable_route(family, active, (1, 2))
torch.cuda.synchronize()
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "declared queue family" in completed.stderr


@CUDA
@pytest.mark.parametrize(
    ("body", "message"),
    (
        (
            """
from types import MappingProxyType
import torch
from pbg.core import CommitCoverage, DeviceDestinationRequest, DeviceFailureState, FamilyBatch, MonotoneSlab, SoAField, SoASchema
device = torch.device("cuda")
failure = DeviceFailureState(1, device)
slab = MonotoneSlab(SoASchema("Leaf", (SoAField("x", torch.int32),), "test"), 2, 1, device, failure_state=failure)
counts = torch.ones(1, dtype=torch.int64, device=device)
request = DeviceDestinationRequest(slab, counts, {"x": torch.ones((1, 1), dtype=torch.int32, device=device)}, "pre", torch.zeros(1, dtype=torch.int64, device=device))
batch = FamilyBatch("pre", torch.zeros(1, dtype=torch.int64, device=device), torch.arange(2, device=device)[None], MappingProxyType({}))
coverage = CommitCoverage(torch.tensor([[True, False]], device=device), torch.tensor([[0, 1]], dtype=torch.int64, device=device), torch.zeros((1, 2), dtype=torch.int64, device=device), torch.tensor([[0, 0]], dtype=torch.int64, device=device))
coverage.validate(batch, (request,), torch.ones(1, dtype=torch.bool, device=device), descriptor_cap=2, workspace_byte_cap=1024)
torch.cuda.synchronize()
""",
            "skipped a scheduled source row",
        ),
        (
            """
from types import MappingProxyType
import torch
from pbg.core import BoundedChanceClosure, ChanceFrontier, DeviceFailureState, MonotoneSlab, RolloutOwnerRegistry, SoAField, SoASchema
device = torch.device("cuda")
failure = DeviceFailureState(1, device)
slab = MonotoneSlab(SoASchema("Leaf", (SoAField("x", torch.int32),), "test"), 1, 1, device, failure_state=failure)
owners = RolloutOwnerRegistry.for_testing((slab,), family_pack_cap=1, chance_frontier_cap=1, completion_pack_cap=1, topology_packet_cap=1, topology_cap=1, topology_edge_cap=1)
frontier = ChanceFrontier(torch.ones((1,1), dtype=torch.bool, device=device), torch.zeros((1,1), dtype=torch.int64, device=device), torch.zeros((1,1), dtype=torch.int64, device=device), MappingProxyType({}))
BoundedChanceClosure(seed=1, iteration=1, owners=owners, max_generations=1).run(frontier, lambda current, key, generation: key, lambda current, sampled, generation: current)
torch.cuda.synchronize()
""",
            "exhausted its generation cap",
        ),
    ),
)
def test_terminal_device_contract_failures(body: str, message: str) -> None:
    completed = subprocess.run(
        (sys.executable, "-c", body),
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert message in completed.stderr


@CUDA
def test_logical_rng_is_invariant_to_order_subdivision_stream_and_placement() -> None:
    device = torch.device("cuda", 0)
    assert LOGICAL_RNG_VERSION == "counter_hash_v4"
    root = torch.tensor([3, 5, 8, 13, 21, 34], dtype=torch.int64, device=device)
    path = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device=device)
    lane = torch.tensor([2, 0, 1, 2, 0, 1], dtype=torch.int64, device=device)
    keys = logical_rng_keys(
        seed=91,
        iteration=7,
        root=root,
        path=path,
        draw=4,
        purpose=DrawPurpose.ACTION,
        lane=lane,
    )
    order = torch.tensor([4, 1, 5, 0, 3, 2], dtype=torch.int64, device=device)
    permuted = logical_rng_keys(
        seed=91,
        iteration=7,
        root=root[order],
        path=path[order],
        draw=4,
        purpose=DrawPurpose.ACTION,
        lane=lane[order],
    )
    rebuilt = torch.empty_like(keys)
    rebuilt[order] = permuted
    split = torch.cat(
        (
            logical_rng_keys(
                seed=91,
                iteration=7,
                root=root[:2],
                path=path[:2],
                draw=4,
                purpose=DrawPurpose.ACTION,
                lane=lane[:2],
            ),
            logical_rng_keys(
                seed=91,
                iteration=7,
                root=root[2:],
                path=path[2:],
                draw=4,
                purpose=DrawPurpose.ACTION,
                lane=lane[2:],
            ),
        )
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        other_stream = logical_rng_keys(
            seed=91,
            iteration=7,
            root=root,
            path=path,
            draw=4,
            purpose=DrawPurpose.ACTION,
            lane=lane,
        )
    stream.synchronize()
    assert torch.equal(keys, rebuilt)
    assert torch.equal(keys, split)
    assert torch.equal(keys, other_stream)
    assert not torch.equal(
        keys,
        logical_rng_keys(
            seed=91,
            iteration=7,
            root=root,
            path=path,
            draw=4,
            purpose=DrawPurpose.CHANCE,
            lane=lane,
        ),
    )
    assert not torch.equal(
        keys,
        logical_rng_keys(
            seed=91,
            iteration=7,
            root=root,
            path=path,
            draw=4,
            purpose=DrawPurpose.ACTION,
            lane=lane,
            retry_epoch=1,
        ),
    )

    active = torch.tensor(
        [True, False, True, False, True, False], device=device
    )
    masked = logical_rng_keys(
        seed=91,
        iteration=7,
        root=root,
        path=path,
        draw=4,
        purpose=DrawPurpose.ACTION,
        lane=lane,
        active=active,
    )
    assert torch.equal(masked[active], keys[active])
    assert torch.equal(masked[~active], torch.zeros(3, dtype=torch.int64, device=device))
    assert torch.equal(
        child_path_keys(path, 1),
        child_path_keys(path, torch.ones_like(path)),
    )
    probability = torch.tensor(
        [[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32, device=device
    )
    selected = logical_categorical(probability, keys[:2])
    assert selected.shape == (2,)
    assert torch.all((logical_uniform(keys) > 0) & (logical_uniform(keys) < 1))
    endpoint_key = torch.tensor(
        [((1 << 24) - 1) << 39],
        dtype=torch.int64,
        device=device,
    )
    assert float(logical_uniform(endpoint_key)[0]) < 1.0
    assert int(
        logical_categorical(
            torch.tensor([[1.0, 0.0]], device=device),
            endpoint_key,
        )[0]
    ) == 0


@CUDA
@pytest.mark.parametrize(
    "rollout_mode", ["sequential", "branched.fixed"]
)
def test_one_model_sample_commit_path_serves_sequential_and_branched(
    rollout_mode: str,
) -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(3, device)
    family = _slab(_family_schema(), 8, failures)
    metadata = _slab(_metadata_schema(), 1, failures)
    leaf = _slab(
        _schema("Leaf", (SoAField("result", torch.int32),)),
        8,
        failures,
    )
    fifo = SemanticFamilyFIFO("pre", _partitions(), family, metadata)
    fifo.initialize()
    counts = torch.tensor([2, 0, 2], dtype=torch.int64, device=device)
    family_values = _values(family, 2)
    family_values["logical_rng_key"] = torch.tensor(
        [
            [[10, 0], [10, 1]],
            [[-1, -1], [-1, -1]],
            [[20, 0], [20, 1]],
        ],
        dtype=torch.int64,
        device=device,
    )
    roots = PublicationTransaction.begin_device(
        (
            DeviceDestinationRequest(
                family,
                counts,
                family_values,
                "pre",
                torch.tensor([10, 0, 20], dtype=torch.int64, device=device),
            ),
        ),
        phase="root_commit",
        active=torch.tensor([True, False, True], device=device),
    )
    roots.fill_all()
    roots.publish()
    fifo.publish_family_tail(
        torch.tensor([True, False, True], device=device)
    )
    schema = mode_schema("self_play", rollout_mode, "sampled")
    states = (
        FamilyQueueState(0, "pre", 0, 0, 2, 1, 0, schema),
        FamilyQueueState(2, "pre", 0, 0, 2, 1, 0, schema),
    )
    scheduled = FamilyScheduler(("pre",), 2).select(
        states, allow_partial=True
    )
    assert scheduled is not None
    owners = _registry((family, metadata, leaf), family_pack_cap=2)
    publisher = AsyncTopologyPublisher(owners, p_train=1)
    draft, patch = _draft(0, device=device)
    seen: dict[str, object] = {}

    class Hooks:
        def model_input(self, batch):
            seen["batch_lanes"] = batch.lanes
            return batch.fields["model_input"]

        def sample(self, request):
            seen["keys"] = request.rng_key
            return SampleWorkspace(
                MappingProxyType({"label": request.rng_key & 7}),
                torch.tensor([2, 2], dtype=torch.int64, device=device),
            )

        def transition(self, request):
            values = {
                "result": torch.zeros(
                    (3, 2), dtype=torch.int32, device=device
                )
            }
            values["result"][request.batch.lanes] = request.sample.values[
                "label"
            ].to(torch.int32)
            return CommitPlan(
                (
                    DeviceDestinationRequest(
                        leaf,
                        counts,
                        values,
                        "pre",
                        torch.tensor(
                            [10, 0, 20], dtype=torch.int64, device=device
                        ),
                    ),
                ),
                _coverage(counts, 2),
                topology=(TopologyPublication(0, draft, patch),),
            )

    def model(value, lanes):
        seen["model_lanes"] = lanes
        return {"policy": value.sum(-1)}

    engine = RolloutTransaction(
        schema,
        seed=123,
        iteration=4,
        topology=publisher,
        owners=owners,
    )
    result = engine.execute(fifo, scheduled, model, Hooks())
    torch.cuda.synchronize(device)
    assert result.success.tolist() == [True, False, True]
    assert result.draw_count.tolist() == [2, 0, 2]
    assert seen["batch_lanes"].tolist() == [0, 2]
    assert seen["model_lanes"].tolist() == [0, 2]
    assert seen["keys"].shape == (2, 2)
    assert fifo.head.tolist() == [2, 0, 2]
    assert [leaf.tail(lane) for lane in range(3)] == [2, 0, 2]
    expected = (seen["keys"] & 7).to(torch.int32)
    assert torch.equal(leaf.published(0, "result"), expected[0])
    assert torch.equal(leaf.published(2, "result"), expected[1])
    assert publisher.committed_packets == ()
    committed = publisher.drain(block=True)
    assert len(committed) == 1
    assert committed[0].lane == 0
    committed[0].packet.topology.validate(
        committed[0].packet.bindings, p_train=1
    )

    family.release()
    metadata.release()
    leaf.release()
    failures.release()


@CUDA
def test_sample_cannot_publish_retained_storage() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(1, device)
    family = _slab(_family_schema(), 4, failures)
    metadata = _slab(_metadata_schema(), 1, failures)
    leaf = _slab(
        _schema("Leaf", (SoAField("result", torch.int32),)),
        4,
        failures,
    )
    fifo = SemanticFamilyFIFO("pre", _partitions(), family, metadata)
    fifo.initialize()
    counts = torch.ones(1, dtype=torch.int64, device=device)
    root = PublicationTransaction.begin_device(
        (_request(family, counts, max_rows=1),),
        phase="root_commit",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    root.fill_all()
    root.publish()
    fifo.publish_family_tail(torch.ones(1, dtype=torch.bool, device=device))
    schema = mode_schema("self_play", "sequential", "sampled")
    scheduled = FamilyScheduler(("pre",), 1).select(
        (FamilyQueueState(0, "pre", 0, 0, 1, 1, 0, schema),),
        allow_partial=True,
    )
    assert scheduled is not None

    class BadHooks:
        def model_input(self, batch):
            return batch.fields["model_input"]

        def sample(self, request):
            leaf._state[:, MonotoneSlab._PUBLISHED].fill_(1)
            return SampleWorkspace(
                MappingProxyType({"label": request.rng_key}),
                torch.ones(1, dtype=torch.int64, device=device),
            )

        def transition(self, request):
            raise AssertionError("transition must not run")

    owners = _registry((family, metadata, leaf), family_pack_cap=1)
    failure_guard = owners.guard()
    failures._records[0, 0].fill_(1)
    with pytest.raises(Exception, match="failure records"):
        owners.assert_unchanged(failure_guard, "sample")
    failures.clear_lane(0)
    engine = RolloutTransaction(
        schema,
        seed=1,
        iteration=1,
        topology=AsyncTopologyPublisher(owners, p_train=1),
        owners=owners,
    )
    with pytest.raises(Exception, match="sample wrote"):
        engine.execute(
            fifo,
            scheduled,
            lambda value, lanes: {"policy": value.sum(-1)},
            BadHooks(),
        )
    torch.cuda.synchronize(device)
    leaf._state[:, MonotoneSlab._PUBLISHED].zero_()
    family.release()
    metadata.release()
    leaf.release()
    failures.release()


@CUDA
def test_branched_fanout_publishes_a_declared_cross_family_fifo() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(1, device)
    source_family = _slab(_family_schema("pre"), 4, failures)
    source_metadata = _slab(_metadata_schema("pre"), 1, failures)
    target_family = _slab(_family_schema("post2"), 4, failures)
    target_metadata = _slab(_metadata_schema("post2"), 1, failures)
    source = SemanticFamilyFIFO(
        "pre", _partitions(), source_family, source_metadata
    )
    target = SemanticFamilyFIFO(
        "post2", _partitions(), target_family, target_metadata
    )
    source.initialize()
    target.initialize()
    one = torch.ones(1, dtype=torch.int64, device=device)
    root = PublicationTransaction.begin_device(
        (_request(source_family, one, max_rows=1),),
        phase="root",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    root.fill_all()
    root.publish()
    source.publish_family_tail(torch.ones(1, dtype=torch.bool, device=device))
    schema = mode_schema("self_play", "branched.fixed", "sampled")
    scheduled = FamilyScheduler(("pre", "post2"), 1).select(
        (FamilyQueueState(0, "pre", 0, 0, 1, 1, 0, schema),),
        allow_partial=True,
    )
    assert scheduled is not None
    owners = _registry(
        (
            source_family,
            source_metadata,
            target_family,
            target_metadata,
        ),
        family_pack_cap=1,
    )
    publisher = AsyncTopologyPublisher(owners, p_train=1)
    duplicate_coverage = _coverage(one, 1)
    with pytest.raises(ValueError, match="repeats a published queue"):
        CommitPlan(
            (
                _request(
                    target_family,
                    one,
                    max_rows=1,
                    input_type="post2",
                ),
            ),
            duplicate_coverage,
            published_queues=(target, target),
        )

    class Hooks:
        def model_input(self, batch):
            return batch.fields["model_input"]

        def sample(self, request):
            return SampleWorkspace(
                MappingProxyType({"label": request.rng_key}),
                torch.ones(1, dtype=torch.int64, device=device),
            )

        def transition(self, request):
            counts = torch.full(
                (1,), 2, dtype=torch.int64, device=device
            )
            ordinal = torch.arange(2, dtype=torch.int64, device=device)[None]
            coverage = CommitCoverage(
                valid=torch.ones((1, 2), dtype=torch.bool, device=device),
                source_ordinal=torch.zeros(
                    (1, 2), dtype=torch.int64, device=device
                ),
                destination=torch.zeros(
                    (1, 2), dtype=torch.int64, device=device
                ),
                destination_ordinal=ordinal,
            )
            return CommitPlan(
                (
                    _request(
                        target_family,
                        counts,
                        max_rows=2,
                        input_type="post2",
                    ),
                ),
                coverage,
                published_queues=(target,),
            )

    result = RolloutTransaction(
        schema,
        seed=5,
        iteration=0,
        topology=publisher,
        owners=owners,
    ).execute(
        source,
        scheduled,
        lambda value, lanes: {"policy": value.sum(-1)},
        Hooks(),
    )
    result.event.synchronize()
    assert source.head.tolist() == [1]
    assert target.ready_tail.tolist() == [2]
    assert target_family.tail(0) == 2
    source_family.release()
    source_metadata.release()
    target_family.release()
    target_metadata.release()
    failures.release()


@CUDA
def test_post_publish_exception_rolls_back_destination_and_source_visibility() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(1, device)
    family = _slab(_family_schema(), 2, failures)
    metadata = _slab(_metadata_schema(), 1, failures)
    leaf = _slab(
        _schema("Leaf", (SoAField("result", torch.int32),)),
        2,
        failures,
    )
    fifo = SemanticFamilyFIFO("pre", _partitions(), family, metadata)
    fifo.initialize()
    counts = torch.ones(1, dtype=torch.int64, device=device)
    root = PublicationTransaction.begin_device(
        (_request(family, counts, max_rows=1),),
        phase="root",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    root.fill_all()
    root.publish()
    fifo.publish_family_tail(torch.ones(1, dtype=torch.bool, device=device))
    schema = mode_schema("self_play", "sequential", "sampled")
    scheduled = FamilyScheduler(("pre",), 1).select(
        (FamilyQueueState(0, "pre", 0, 0, 1, 1, 0, schema),),
        allow_partial=True,
    )
    assert scheduled is not None
    owners = _registry(
        (family, metadata, leaf),
        family_pack_cap=1,
        completion_pack_cap=1,
    )
    publisher = AsyncTopologyPublisher(owners, p_train=1)

    class Hooks:
        def model_input(self, batch):
            return batch.fields["model_input"]

        def sample(self, request):
            return SampleWorkspace(
                MappingProxyType({"label": request.rng_key}),
                torch.ones(1, dtype=torch.int64, device=device),
            )

        def transition(self, request):
            destination = _request(leaf, counts, max_rows=1)
            return CommitPlan((destination,), _coverage(counts, 1))

    engine = RolloutTransaction(
        schema,
        seed=1,
        iteration=0,
        topology=publisher,
        owners=owners,
    )
    with pytest.raises(RuntimeError, match="after publish"):
        engine.execute(
            fifo,
            scheduled,
            lambda value, lanes: {"policy": value.sum(-1)},
            Hooks(),
            post_publish_failure_injector=lambda: (_ for _ in ()).throw(
                RuntimeError("after publish")
            ),
        )
    torch.cuda.synchronize(device)
    assert leaf.tail(0) == leaf.reserved_tail(0) == 0
    assert fifo.head.tolist() == [0]
    failure = failures.snapshot(0)
    assert failure is not None
    assert failure.failure_code == "transaction_rollback"
    publisher.assert_drained()
    for slab in (family, metadata, leaf):
        slab.reset_failed_lane(0)
    failures.clear_lane(0)
    assert family.epoch(0) == metadata.epoch(0) == leaf.epoch(0) == 1
    family.release()
    metadata.release()
    leaf.release()
    failures.release()


@CUDA
def test_topology_endpoints_bind_divergent_reservations_and_keep_replica_identity() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(2, device)
    leaf = _slab(
        _schema("Leaf", (SoAField("terminal", torch.int32),)),
        4,
        failures,
    )
    owners = _registry((leaf,))
    publisher = AsyncTopologyPublisher(owners, p_train=1)
    one = torch.ones(2, dtype=torch.int64, device=device)
    prior = PublicationTransaction.begin_device(
        (_request(leaf, one, max_rows=1),),
        phase="prior",
        active=torch.tensor([False, True], device=device),
    )
    prior.fill_all()
    prior.publish()
    request = _request(leaf, one, max_rows=1, base=7)
    transaction = PublicationTransaction.begin_device(
        (request,),
        phase="rollout",
        active=torch.ones(2, dtype=torch.bool, device=device),
    )
    publications = tuple(
        TopologyPublication(
            lane,
            *_draft(0, device=device),
        )
        for lane in (0, 1)
    )
    for publication in publications:
        publisher.preflight(publication, (request,))
        publisher.stage(publication)
    resolved = tuple(
        publisher.resolve_rows(publication, transaction)
        for publication in publications
    )
    marker = torch.full((2,), -1, dtype=torch.int64, device=device)
    stream = torch.cuda.current_stream(device)
    pending = publisher.prepare_defer(
        publications, resolved, marker, stream=stream
    )
    transaction.fill_all()
    transaction.publish(
        commit_marker=marker,
        sequence=torch.zeros_like(marker),
    )
    publisher.finalize_defer(pending, marker, stream=stream)
    accepted = publisher.drain()
    if not accepted:
        accepted = publisher.drain(block=True)
    assert tuple(item.lane for item in accepted) == (0, 1)
    assert tuple(
        item.packet.topology.target_row.item() for item in accepted
    ) == (0, 1)
    assert tuple(item.packet.sequence for item in accepted) == (0, 0)
    publisher.assert_drained()
    leaf.release()
    failures.release()


@CUDA
def test_topology_staging_enforces_cumulative_manifest_caps() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(2, device)
    leaf = _slab(
        _schema("Leaf", (SoAField("terminal", torch.int32),)),
        2,
        failures,
    )
    owners = _registry(
        (leaf,),
        topology_packet_cap=2,
        topology_cap=1,
        topology_edge_cap=1,
    )
    publisher = AsyncTopologyPublisher(owners, p_train=1)
    first = TopologyPublication(0, *_draft(0, device=device))
    second = TopologyPublication(1, *_draft(0, device=device))
    publisher.stage(first)
    with pytest.raises(TopologyError, match="cumulative node cap"):
        publisher.stage(second)
    publisher.abort(0, 0)
    publisher.assert_drained()
    leaf.release()
    failures.release()


@CUDA
def test_bounded_chance_closure_uses_fixed_launches_and_inactive_draws_nothing() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(2, device)
    owner = _slab(
        _schema("Leaf", (SoAField("terminal", torch.int32),)),
        4,
        failures,
    )
    owners = _registry((owner,), chance_frontier_cap=4)
    active = torch.tensor(
        [[True, True, False, False], [True, False, False, False]],
        dtype=torch.bool,
        device=device,
    )
    frontier = ChanceFrontier(
        active=active,
        root=torch.tensor(
            [[0, 1, 0, 0], [2, 0, 0, 0]],
            dtype=torch.int64,
            device=device,
        ),
        path=torch.zeros((2, 4), dtype=torch.int64, device=device),
        payload=MappingProxyType(
            {"generation": torch.zeros((2, 4), device=device)}
        ),
    )
    calls: list[int] = []

    def sample(current, key, generation):
        calls.append(generation)
        torch._assert_async(
            (key[~current.active] == 0).all(),
            "inactive chance row received a draw",
        )
        return key

    def commit(current, sampled, generation):
        del sampled
        next_active = current.active & (generation < 1)
        payload = {
            "generation": torch.full_like(
                current.payload["generation"], generation + 1
            )
        }
        return ChanceFrontier(
            next_active,
            current.root,
            child_path_keys(
                current.path,
                torch.full_like(current.path, generation + 1),
            ),
            MappingProxyType(payload),
        )

    result = BoundedChanceClosure(
        seed=17,
        iteration=3,
        owners=owners,
        max_generations=4,
    ).run(frontier, sample, commit)
    torch.cuda.synchronize(device)
    assert calls == [0, 1, 2, 3]
    assert result.generation_draws.tolist() == [3, 3, 0, 0]
    assert not result.frontier.active.any()

    source = inspect.getsource(BoundedChanceClosure.run)
    tree = ast.parse(textwrap.dedent(source))
    calls_in_source = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"item", "cpu", "tolist", "numpy"}.intersection(calls_in_source)
    owner.release()
    failures.release()


@CUDA
def test_completion_fifo_consumes_exact_stable_prefix_and_isolates_failure() -> None:
    device = torch.device("cuda", 0)
    failures = DeviceFailureState(2, device)
    leaf = _slab(
        _schema("Leaf", (SoAField("terminal", torch.int32),)),
        4,
        failures,
    )
    work = _slab(
        _schema("LayerWork.all", (SoAField("return_value", torch.float32),)),
        2,
        failures,
    )
    leaf_counts = torch.tensor([2, 2], dtype=torch.int64, device=device)
    leaves = PublicationTransaction.begin_device(
        (_request(leaf, leaf_counts, max_rows=2, base=9),),
        phase="leaf_commit",
        active=torch.ones(2, dtype=torch.bool, device=device),
    )
    leaves.fill_all()
    leaves.publish()
    prior_counts = torch.tensor([0, 1], dtype=torch.int64, device=device)
    prior = PublicationTransaction.begin_device(
        (_request(work, prior_counts, max_rows=1, base=4),),
        phase="prior_layer_work",
        active=torch.tensor([False, True], device=device),
    )
    prior.fill_all()
    prior.publish()
    owners = _registry(
        (leaf, work), completion_pack_cap=2, family_pack_cap=2
    )
    fifo = CompletionFIFO(
        leaf,
        owners,
        head=torch.zeros(2, dtype=torch.int64, device=device),
        observed_epoch=torch.zeros(2, dtype=torch.int64, device=device),
    )

    def complete(batch):
        values = {
            "return_value": batch.fields["terminal"].to(torch.float32)
        }
        counts = torch.tensor([2, 3], dtype=torch.int64, device=device)
        return CompletionPlan(
            (
                DeviceDestinationRequest(
                    work,
                    counts,
                    values=values,
                    input_type="all",
                    logical_address=torch.tensor(
                        [30, 31], dtype=torch.int64, device=device
                    ),
                ),
            ),
            _coverage(counts, 2),
        )

    ready = torch.cuda.Event(blocking=False)
    ready.record()

    def illegal(batch):
        counts = torch.full(
            (2,), 2, dtype=torch.int64, device=device
        )
        return CompletionPlan(
            (
                DeviceDestinationRequest(
                    leaf,
                    counts,
                    values={"terminal": batch.fields["terminal"]},
                    input_type="all",
                    logical_address=torch.tensor(
                        [40, 41], dtype=torch.int64, device=device
                    ),
                ),
            ),
            _coverage(counts, 2),
        )

    with pytest.raises(Exception, match="LayerWork"):
        fifo.complete((0, 1), 2, illegal, wait_for=ready)
    assert fifo.head.tolist() == [0, 0]
    success = fifo.complete((0, 1), 2, complete, wait_for=ready)
    torch.cuda.synchronize(device)
    assert success.tolist() == [True, False]
    assert fifo.head.tolist() == [2, 0]
    assert work.tail(0) == 2
    assert work.tail(1) == 1
    failure = failures.snapshot(1)
    assert failure is not None and failure.phase == "completion"
    leaf.reset_failed_lane(1)
    work.reset_failed_lane(1)
    failures.clear_lane(1)
    fifo._observe_retry_epoch()
    assert fifo.observed_epoch.tolist() == [0, 1]
    assert fifo.head.tolist() == [2, 0]
    fifo.release()
    with pytest.raises(Exception, match="released"):
        _ = fifo.ready_tail
    leaf.release()
    work.release()
    failures.release()


def test_rollout_hot_paths_do_not_read_cuda_scalars_on_host() -> None:
    hot = (
        PublicationTransaction.begin_device,
        CommitCoverage.validate,
        SemanticFamilyFIFO.publish_family_tail,
        SemanticFamilyFIFO.stable_prefix,
        SemanticFamilyFIFO.consume,
        RolloutTransaction.execute,
        BoundedChanceClosure.run,
        stable_route,
    )
    forbidden = {"item", "cpu", "tolist", "numpy"}
    for function in hot:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        assert not forbidden.intersection(attributes), function.__qualname__
