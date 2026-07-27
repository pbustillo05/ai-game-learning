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

from pbg.queued import (
    AsyncTopologyPublisher,
    BoundedChanceClosure,
    ChanceFrontier,
    CommitPlan,
    CompletionFIFO,
    ConversionSpec,
    DeviceDestinationRequest,
    DeviceFailureState,
    DevicePublicationTransaction,
    DrawPurpose,
    EndpointPatch,
    FamilyQueueState,
    FamilyScheduler,
    MonotoneSlab,
    Partition,
    PartitionContract,
    PartitionMap,
    RolloutMode,
    RolloutTransaction,
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


def _i32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def _draft(sequence: int = 0) -> tuple[TopologyDraftPacket, EndpointPatch]:
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
    )
    patch = EndpointPatch(
        sequence=sequence,
        edge=_i32([0]),
        target_kind=_i32([1]),
        target_row=_i32([0]),
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
    prior = DevicePublicationTransaction.begin(
        (_request(right, prior_counts, max_rows=1, base=3),),
        phase="prior_commit",
        active=torch.tensor([False, True, False], device=device),
    )
    prior.fill_all()
    prior.publish()
    left_counts = torch.tensor([2, 2, 1], dtype=torch.int64, device=device)
    right_counts = torch.tensor([2, 4, 1], dtype=torch.int64, device=device)
    transaction = DevicePublicationTransaction.begin(
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
    invalid = DevicePublicationTransaction.begin(
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
    initial = DevicePublicationTransaction.begin(
        (_request(boundary, two, max_rows=2, base=3),),
        phase="initial",
        active=torch.ones(1, dtype=torch.bool, device=device),
    )
    initial.fill_all()
    initial.publish()
    final = DevicePublicationTransaction.begin(
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
    transaction = DevicePublicationTransaction.begin(
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
        transaction = DevicePublicationTransaction.begin(
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
        "memcpy dtoh" in name or "device to host" in name for name in names
    )
    slab.release()
    failures.release()


@CUDA
def test_undeclared_device_route_fails_terminally_without_skipping() -> None:
    script = """
import torch
from pbg.queued import stable_route
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
def test_logical_rng_is_invariant_to_order_subdivision_stream_and_placement() -> None:
    device = torch.device("cuda", 0)
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
    roots = DevicePublicationTransaction.begin(
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
    publisher = AsyncTopologyPublisher(2)
    draft, patch = _draft(0)
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
        retained_slabs=(family, leaf),
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
    committed = publisher.drain()
    assert len(committed) == 1
    committed[0].topology.validate(committed[0].bindings, p_train=1)

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
    root = DevicePublicationTransaction.begin(
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

    engine = RolloutTransaction(
        schema,
        seed=1,
        iteration=1,
        topology=AsyncTopologyPublisher(2),
        retained_slabs=(family, leaf),
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


def test_topology_descriptors_publish_only_with_complete_endpoint_patch() -> None:
    too_small = AsyncTopologyPublisher(1)
    oversized, _ = _draft(0)
    with pytest.raises(TopologyError, match="packet cap"):
        too_small.stage(oversized)

    publisher = AsyncTopologyPublisher(2)
    draft, patch = _draft(0)
    publisher.stage(draft)
    bad = EndpointPatch(
        sequence=0,
        edge=_i32([]),
        target_kind=_i32([]),
        target_row=_i32([]),
        child_output_type=_i32([]),
        commit_sequence=0,
    )
    with pytest.raises(TopologyError, match="cover unresolved"):
        publisher.commit(bad)

    publisher = AsyncTopologyPublisher(2)
    publisher.stage(draft)
    packet = publisher.commit(patch)
    packet.topology.validate(packet.bindings, p_train=1)
    assert packet.topology.target_row.tolist() == [0]
    assert len(publisher.committed_packets) == 1
    publisher.assert_drained()

    second, _ = _draft(1)
    publisher.stage(second)
    with pytest.raises(TopologyError, match="uncommitted"):
        publisher.assert_drained()
    publisher.abort(1)
    publisher.assert_drained()


@CUDA
def test_topology_acceptance_waits_for_device_commit_marker_and_drops_failure() -> None:
    device = torch.device("cuda", 0)
    publisher = AsyncTopologyPublisher(2)
    draft0, patch0 = _draft(0)
    publisher.stage(draft0)
    marker0 = torch.tensor([0], dtype=torch.int64, device=device)
    publisher.defer(TopologyPublication(0, draft0, patch0), marker0)
    assert publisher.committed_packets == ()
    accepted = publisher.drain()
    assert len(accepted) == 1
    accepted[0].topology.validate(accepted[0].bindings, p_train=1)

    draft1, patch1 = _draft(1)
    publisher.stage(draft1)
    marker1 = torch.tensor([-1], dtype=torch.int64, device=device)
    publisher.defer(TopologyPublication(0, draft1, patch1), marker1)
    assert publisher.drain() == ()
    publisher.assert_drained()


@CUDA
def test_bounded_chance_closure_uses_fixed_launches_and_inactive_draws_nothing() -> None:
    device = torch.device("cuda", 0)
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
        capacity=4,
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
    leaves = DevicePublicationTransaction.begin(
        (_request(leaf, leaf_counts, max_rows=2, base=9),),
        phase="leaf_commit",
        active=torch.ones(2, dtype=torch.bool, device=device),
    )
    leaves.fill_all()
    leaves.publish()
    fifo = CompletionFIFO(leaf, 2)

    def complete(batch):
        values = {
            "return_value": batch.fields["terminal"].to(torch.float32)
        }
        return (
            DeviceDestinationRequest(
                work,
                torch.tensor([2, 3], dtype=torch.int64, device=device),
                values=values,
                input_type="all",
                logical_address=torch.tensor(
                    [30, 31], dtype=torch.int64, device=device
                ),
            ),
        )

    success = fifo.complete((0, 1), 2, complete)
    torch.cuda.synchronize(device)
    assert success.tolist() == [True, False]
    assert fifo.head.tolist() == [2, 0]
    assert work.tail(0) == 2
    assert work.tail(1) == 0
    failure = failures.snapshot(1)
    assert failure is not None and failure.phase == "completion"
    leaf.reset_failed_lane(1)
    work.reset_failed_lane(1)
    failures.clear_lane(1)
    leaf.release()
    work.release()
    failures.release()


def test_rollout_hot_paths_do_not_read_cuda_scalars_on_host() -> None:
    hot = (
        DevicePublicationTransaction.__init__,
        DevicePublicationTransaction.fill,
        DevicePublicationTransaction.publish,
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
