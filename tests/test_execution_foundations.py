from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from pbg.core.observability import ExecutionMetrics
from pbg.core.execution import (
    DONE,
    EVALUATING,
    NEW,
    QUEUED,
    ArenaField,
    ByteAccountant,
    ChunkedOccurrenceArena,
    DeviceWorkQueue,
    DrawPurpose,
    TrajectoryAdmissionPlan,
    arena_fields,
    child_path_keys,
    logical_categorical_from_valid_probability,
    logical_rng_keys,
    select_work_queue,
    stable_partition_masked,
)
from pbg.core.training import (
    _analytical_memory_plan,
    _rank_local_config,
    _validate_memory_calibration,
)
from pbg.core.learner import ReusableIndexBuffer
from pbg.core.targets import _walker_alias
from pbg.core.game import GameBundle
from pbg.core.storage import TensorFieldSchema, TensorSchema
from pbg.core.optimization import (
    FlatGradientBuffer,
    finite_gated_optimizer_step,
    sync_scaled_gradients,
)


def test_four_phase_memory_plan_is_schema_derived_without_padded_action_owner() -> None:
    model = torch.nn.Linear(3, 2)
    input_schema = TensorSchema(
        "test_natural_input",
        (TensorFieldSchema("range_weights", torch.float32, (2, 1326)),),
    )
    backend = SimpleNamespace(
        n_players=2,
        x_sizes=(1326, 1326),
        e_max=66,
        max_rollout_steps=42,
        states=SimpleNamespace(storage=torch.empty(17, dtype=torch.uint8)),
        partition_summary_cache=SimpleNamespace(
            arena=torch.empty(43, dtype=torch.bool)
        ),
        training_partition_specs=lambda: SimpleNamespace(
            specs=(SimpleNamespace(input_schema=input_schema, model_width=1326),)
        ),
    )
    bundle = GameBundle(
        model=model,
        rollout_backend=backend,
        policy_adapter=SimpleNamespace(),
        completion_backend=SimpleNamespace(),
    )
    cfg = SimpleNamespace(
        training=SimpleNamespace(mode="rao_blackwell"),
        rollout=SimpleNamespace(
            roots_per_gpu=10,
            storage_page_rows=16,
            family_batch_rows=64,
            max_active_trajectories=64,
        ),
    )

    plan = _analytical_memory_plan(cfg, bundle, has_reference=True)
    assert plan["memory_plan_backend_resident_bytes"] == 60
    assert "memory_plan_rollout_decisions_bytes" not in plan
    assert "memory_plan_decision_inputs_bytes" not in plan
    for phase in ("rollout", "seal", "target", "optimization"):
        assert plan[f"memory_plan_{phase}_phase_bytes"] == (
            plan[f"memory_plan_{phase}_fixed_bytes"]
            + plan[f"memory_plan_{phase}_variable_bytes"]
        )
    assert plan["memory_plan_seal_phase_bytes"] > plan["memory_plan_rollout_phase_bytes"]
    assert plan["memory_plan_training_storage_bytes"] > 0
    assert plan["memory_plan_sealed_rollout_bytes"] > 0
    assert plan["memory_plan_target_variable_bytes"] == (
        plan["memory_plan_sealed_rollout_bytes"]
        + plan["memory_plan_target_destination_bytes"]
        + plan["memory_plan_completion_workspace_bytes"]
        + plan["memory_plan_frontier_workspace_bytes"]
    )
    assert plan["memory_plan_structural_peak_bytes"] == max(
        plan[f"memory_plan_{phase}_phase_bytes"]
        for phase in ("rollout", "seal", "target", "optimization")
    )


def test_memory_calibration_reports_actual_hard_headroom_without_synthetic_margin(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pbg.core.training._memory_limit",
        lambda cfg, device, account_current_free=False: 1_000,
    )
    metrics = {"iteration_peak_reserved_bytes": 801.0}
    _validate_memory_calibration(SimpleNamespace(), torch.device("cuda"), metrics)
    assert metrics["calibrated_actual_headroom_bytes"] == 199.0
    assert metrics["memory_hard_limit_exceeded"] == 0.0


def test_execution_metrics_are_lossless_and_match_hand_counts() -> None:
    metrics = ExecutionMetrics()
    for _ in range(5_000):
        metrics.record_model_call(
            "inference", "pre", 2, padded_tokens=20, padded_token_pairs=200
        )
    metrics.record_policy_pack(
        "inference", "post_3", rows=3, padded_tokens=90, padded_token_pairs=2_700
    )
    metrics.record_policy_pack(
        "inference", "post_3", rows=1, padded_tokens=20, padded_token_pairs=400
    )
    metrics.record_queue("post_3", depth=7, oldest_age=2.5)
    metrics.record_queue("post_3", depth=4, oldest_age=3.0)
    metrics.add("forced_drains", 2)
    metrics.record_evaluator(rows=11)
    metrics.record_target_bytes(120)
    metrics.record_target_bytes(80)
    metrics.record_collective(nbytes=256, seconds=0.125)
    snapshot = metrics.snapshot()

    assert snapshot["execution_model_calls"] == 5_000
    assert snapshot["execution_model_rows"] == 10_000
    assert snapshot["execution_inference_pre_model_calls"] == 5_000
    assert snapshot["execution_policy_pack_calls"] == 2
    assert snapshot["execution_policy_pack_rows"] == 4
    assert snapshot["execution_padded_tokens"] == 100_000
    assert snapshot["execution_padded_token_pairs"] == 1_000_000
    assert snapshot["execution_policy_pack_padded_tokens"] == 110
    assert snapshot["execution_policy_pack_padded_token_pairs"] == 3_100
    assert snapshot["execution_queue_depth_max"] == 7
    assert snapshot["execution_queue_oldest_age_max"] == 3.0
    assert snapshot["execution_evaluator_rows"] == 11
    assert snapshot["execution_target_bytes"] == 200
    assert snapshot["execution_target_bytes_max"] == 120
    assert snapshot["execution_collective_bytes"] == 256


def test_declared_control_sync_and_transfers_are_accounted_separately() -> None:
    metrics = ExecutionMetrics()
    metrics.record_scheduler_control_synchronization(seconds=0.002, nbytes=32)
    metrics.record_transfer("h2d", nbytes=48)
    snapshot = metrics.snapshot()

    assert snapshot["execution_scheduler_control_synchronizations"] == 1
    assert snapshot["execution_scheduler_control_bytes"] == 32
    assert snapshot["execution_undeclared_host_synchronizations"] == 0
    assert snapshot["execution_d2h_transfers"] == 1
    assert snapshot["execution_d2h_bytes"] == 32
    assert snapshot["execution_h2d_transfers"] == 1
    assert snapshot["execution_h2d_bytes"] == 48


def test_alias_table_reconstructs_law_and_draws_from_it() -> None:
    probability = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    threshold, alias = _walker_alias(probability)
    reconstructed = torch.zeros_like(probability)
    for column in range(probability.numel()):
        reconstructed[column] += threshold[column] / probability.numel()
        reconstructed[alias[column]] += (1.0 - threshold[column]) / probability.numel()
    assert torch.allclose(reconstructed, probability)

    gen = torch.Generator().manual_seed(5)
    draws = ReusableIndexBuffer(40_000, probability.device).draw_alias(
        threshold, alias, 40_000, gen
    )
    empirical = torch.bincount(draws, minlength=3).to(torch.float64) / draws.numel()
    assert torch.allclose(empirical, probability, atol=0.01, rtol=0.0)


def test_flat_gradient_sum_zero_fills_absent_parameters_and_reuses_buffer() -> None:
    model = torch.nn.Linear(2, 1)
    flat = FlatGradientBuffer(model)
    flat.zero_()
    parameters = list(model.parameters())
    parameters[0].grad.copy_(torch.tensor([[2.0, -3.0]]))

    assert sync_scaled_gradients(1, flat_buffer=flat)
    assert torch.equal(parameters[0].grad, torch.tensor([[2.0, -3.0]]))
    assert torch.equal(parameters[1].grad, torch.zeros_like(parameters[1]))
    first = flat.flat

    parameters[0].grad.fill_(float("inf"))
    assert not sync_scaled_gradients(1, flat_buffer=flat)
    assert first is flat.flat


def test_nonfinite_common_decision_skips_disabled_scaler_optimizer_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model = torch.nn.ParameterList([parameter])
    flat = FlatGradientBuffer(model)
    flat.zero_()
    parameter.grad.fill_(float("inf"))
    common_finite = sync_scaled_gradients(1, flat_buffer=flat)

    result = finite_gated_optimizer_step(
        model,
        optimizer,
        scaler,
        gradients_finite=common_finite,
        grad_clip=1.0,
    )

    assert not result.optimizer_step_applied
    assert result.loss_scale_before == result.loss_scale_after == 1.0
    assert result.grad_clip_scale == 0.0
    assert torch.equal(parameter.detach(), torch.tensor([1.0], dtype=torch.float32))
    assert torch.equal(parameter.grad, torch.zeros_like(parameter))


class _RecordingEnabledScaler:
    def __init__(self) -> None:
        self.scale_value = 1024.0
        self.unscale_calls = 0
        self.step_calls = 0
        self.explicit_updates: list[float | None] = []

    def is_enabled(self) -> bool:
        return True

    def get_scale(self) -> float:
        return self.scale_value

    def get_backoff_factor(self) -> float:
        return 0.5

    def unscale_(self, optimizer) -> None:
        del optimizer
        self.unscale_calls += 1

    def step(self, optimizer) -> None:
        self.step_calls += 1
        optimizer.step()

    def update(self, new_scale=None) -> None:
        self.explicit_updates.append(new_scale)
        if new_scale is not None:
            self.scale_value = float(new_scale)


def test_nonfinite_common_decision_controls_fp16_scaler_transition() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    scaler = _RecordingEnabledScaler()
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("inf"))

    result = finite_gated_optimizer_step(
        model,
        optimizer,
        scaler,
        gradients_finite=False,
        grad_clip=1.0,
    )

    assert not result.optimizer_step_applied
    assert scaler.unscale_calls == scaler.step_calls == 0
    assert scaler.explicit_updates == [512.0]
    assert result.loss_scale_before == 1024.0
    assert result.loss_scale_after == 512.0
    assert torch.equal(model.weight, before)


def test_sequential_roots_are_a_common_per_gpu_width() -> None:
    cfg = SimpleNamespace(
        training=SimpleNamespace(mode="rao_blackwell"),
        rollout=SimpleNamespace(roots_per_gpu=10),
    )
    ranks = [_rank_local_config(cfg, rank, 3) for rank in range(3)]
    assert [item.rollout.roots_per_gpu for item in ranks] == [10, 10, 10]
    assert all(item.rollout.trajectory_rank_counts == (10, 10, 10) for item in ranks)


def test_arena_allocation_is_chunked_exact_and_overflow_is_transactional() -> None:
    fields = arena_fields(("state_ref",))
    row_bytes = sum(field.row_bytes for field in fields.values())
    accountant = ByteAccountant(limit_bytes=4 * row_bytes)
    arena = ChunkedOccurrenceArena(
        capacity=5,
        chunk_nodes=2,
        device="cpu",
        fields=fields,
        accountant=accountant,
    )
    handles = arena.allocate(3)
    arena.scatter("state_ref", handles, torch.tensor([7, 8, 9], dtype=torch.int32))

    assert arena.chunk_count == 2
    assert arena.reserved_bytes == 4 * row_bytes
    assert arena.used_bytes == 3 * row_bytes
    assert accountant.reserved_bytes == arena.reserved_bytes
    assert torch.equal(arena.gather("state_ref", handles), torch.tensor([7, 8, 9]))
    before = (arena.size, arena.reserved_bytes, accountant.reserved_bytes)
    with pytest.raises(MemoryError):
        arena.allocate(2)
    assert (arena.size, arena.reserved_bytes, accountant.reserved_bytes) == before


def test_occurrence_arena_requires_integer_metadata_and_unique_scatter_handles() -> None:
    fields = arena_fields(("state_ref",))
    fields["float_state"] = ArenaField(torch.float32)
    with pytest.raises(ValueError, match="integer dtypes"):
        ChunkedOccurrenceArena(4, 2, "cpu", fields=fields)

    arena = ChunkedOccurrenceArena(
        4, 2, "cpu", fields=arena_fields(("state_ref",))
    )
    handles = arena.allocate(2)
    with pytest.raises(RuntimeError, match="duplicate occurrence handle"):
        arena.scatter(
            "state_ref",
            handles[[0, 0]],
            torch.tensor([1, 2], dtype=torch.int32),
        )


def test_masked_stable_partition_is_fixed_shape_stable_and_complete() -> None:
    handles = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    family = torch.tensor([3, 2, 3, 5, 2, 3], dtype=torch.int32)
    partition = stable_partition_masked(handles, family, (2, 3, 5))
    assert partition.candidate_handles.shape == (3, 6)
    assert partition.valid_mask.shape == (3, 6)
    assert torch.equal(partition.selected_count, torch.tensor([2, 3, 1]))
    assert torch.equal(
        partition.candidate_handles[0][partition.valid_mask[0]],
        torch.tensor([11, 14]),
    )
    assert torch.equal(
        partition.candidate_handles[1][partition.valid_mask[1]],
        torch.tensor([10, 12, 15]),
    )
    assert torch.equal(
        partition.candidate_handles[2][partition.valid_mask[2]],
        torch.tensor([13]),
    )
    assert all(
        bool((~row).cummax(0).values.eq(~row).all()) for row in partition.valid_mask
    )

    with pytest.raises(RuntimeError, match="exactly one execution family"):
        stable_partition_masked(handles, family, (2, 3))


def test_arena_multi_field_gather_scatter_share_cross_chunk_indexing() -> None:
    arena = ChunkedOccurrenceArena(
        7,
        2,
        "cpu",
        fields=arena_fields(("state_ref", "trajectory_id", "path_key")),
    )
    handles = arena.allocate(7)
    selected = handles[torch.tensor([6, 1, 4, 3])]
    values = {
        "state_ref": torch.tensor([16, 11, 14, 13], dtype=torch.int32),
        "trajectory_id": torch.tensor([106, 101, 104, 103], dtype=torch.int64),
        "path_key": torch.tensor([206, 201, 204, 203], dtype=torch.int64),
    }

    arena.scatter_many(selected, values)
    gathered = arena.gather_many(tuple(values), selected)

    assert gathered.keys() == values.keys()
    for name, expected in values.items():
        assert torch.equal(gathered[name], expected)
    assert torch.equal(arena.gather("state_ref", selected), values["state_ref"])


def test_work_queue_status_lifecycle_duplicate_guard_and_overflow_are_exact() -> None:
    arena = ChunkedOccurrenceArena(6, 2, "cpu")
    handles = arena.allocate(6)
    queue = DeviceWorkQueue(3, "cpu")
    queue.enqueue(handles[:3], arena=arena)
    assert queue.host_count == 3
    assert torch.equal(arena.gather("status", handles[:3]), torch.full((3,), QUEUED))

    popped = queue.dequeue(2, arena=arena)
    assert torch.equal(popped, handles[:2])
    assert torch.equal(arena.gather("status", popped), torch.full((2,), EVALUATING))
    DeviceWorkQueue.mark_done(arena, popped)
    assert torch.equal(arena.gather("status", popped), torch.full((2,), DONE))

    depth = queue.host_count
    status = arena.gather("status", handles).clone()
    with pytest.raises(RuntimeError, match="status"):
        queue.enqueue(handles[2:3], arena=arena)
    assert queue.host_count == depth
    assert torch.equal(arena.gather("status", handles), status)

    with pytest.raises(MemoryError, match="insertion-order overflow"):
        queue.enqueue(handles[3:], arena=arena)
    assert queue.host_count == depth
    assert torch.equal(arena.gather("status", handles[3:]), torch.full((3,), NEW))


def test_work_queue_age_tracks_oldest_live_enqueue_without_device_reads() -> None:
    arena = ChunkedOccurrenceArena(6, 3, "cpu")
    handles = arena.allocate(6)
    queue = DeviceWorkQueue(6, "cpu")
    queue.enqueue(handles[:2], arena=arena, logical_time=3)
    queue.enqueue(handles[2:5], arena=arena, logical_time=7)

    assert queue.oldest_age(9) == 6
    first = queue.dequeue(2, arena=arena)
    DeviceWorkQueue.mark_done(arena, first)
    assert queue.oldest_age(9) == 2
    queue.enqueue(handles[5:], arena=arena, logical_time=10)
    assert queue.oldest_age(12) == 5

    rest = queue.dequeue(4, arena=arena)
    DeviceWorkQueue.mark_done(arena, rest)
    assert queue.oldest_age(99) == 0
    with pytest.raises(ValueError, match="monotone"):
        queue.enqueue(torch.empty(0, dtype=torch.int64), logical_time=9)


def test_work_queue_capacity_and_prefix_staging_are_exactly_accounted() -> None:
    accountant = ByteAccountant(10_000)
    queue = DeviceWorkQueue(
        6,
        "cpu",
        staging_rows=4,
        accountant=accountant,
        label="test_queue",
    )
    assert queue.reserved_bytes == (6 + 4 + 1) * 8
    assert accountant.reserved_bytes == queue.reserved_bytes

    queue.enqueue(torch.tensor([0, 1, 2, 3], dtype=torch.int64))
    assert torch.equal(queue.dequeue(3), torch.tensor([0, 1, 2]))
    queue.enqueue(torch.tensor([4, 5, 6, 7], dtype=torch.int64))
    assert torch.equal(queue.dequeue(4), torch.tensor([3, 4, 5, 6]))
    assert torch.equal(queue.dequeue(1), torch.tensor([7]))
    queue.release()
    assert accountant.reserved_bytes == 0


def test_queue_scheduler_is_globally_full_first_then_preorder_fill_age() -> None:
    partition_ids = (0, 1)
    priorities = (0, 1)
    queues = {
        partition_id: DeviceWorkQueue(8, "cpu", staging_rows=3)
        for partition_id in partition_ids
    }
    queues[0].enqueue(torch.tensor([0, 1], dtype=torch.int64), logical_time=0)
    queues[1].enqueue(
        torch.tensor([2, 3, 4], dtype=torch.int64), logical_time=1
    )
    selected = select_work_queue(
        partition_ids, priorities, queues, 3, allow_partial=False, logical_time=2
    )
    assert selected is not None
    assert (selected.partition_id, selected.rows, selected.forced_partial) == (
        1, 3, False
    )

    queues[1].dequeue(3)
    selected = select_work_queue(
        partition_ids, priorities, queues, 3, allow_partial=True, logical_time=3
    )
    assert selected is not None
    assert (selected.partition_id, selected.rows, selected.forced_partial) == (
        0, 2, True
    )


def test_safe_admit_reserves_retained_and_max_inflight_before_live_rows() -> None:
    plan = TrajectoryAdmissionPlan(
        limit_bytes=1_000,
        retained_bytes=300,
        max_inflight_bytes=200,
        resident_bytes_per_start=100,
    )
    plan.require_progress_capacity(10)

    assert plan.base_bytes == 500
    assert plan.safe_admit(active_starts=0, unstarted=10, requested=8) == 5
    assert plan.safe_admit(active_starts=3, unstarted=7, requested=8) == 2
    assert plan.safe_admit(active_starts=5, unstarted=5, requested=8) == 0
    assert plan.committed_bytes(5) == plan.limit_bytes
    # Releasing two completed resident rows permits exactly two new starts;
    # retained and maximum in-flight reservations never change.
    assert plan.safe_admit(active_starts=3, unstarted=5, requested=5) == 2

    impossible = TrajectoryAdmissionPlan(599, 300, 200, 100)
    with pytest.raises(MemoryError, match="no capacity for one"):
        impossible.require_progress_capacity(1)


def test_logical_rng_is_invariant_to_schedule_and_gpu_assignment() -> None:
    trajectory = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    parent_path = torch.zeros(4, dtype=torch.int64)
    path = child_path_keys(parent_path, torch.tensor([1, 2, 1, 3]))
    keys = logical_rng_keys(
        seed=17,
        iteration=9,
        trajectory=trajectory,
        path=path,
        draw=1,
        purpose=DrawPurpose.ACTION,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    scheduled = logical_rng_keys(
        seed=17,
        iteration=9,
        trajectory=trajectory[permutation],
        path=path[permutation],
        draw=1,
        purpose=DrawPurpose.ACTION,
    )
    restored = torch.empty_like(scheduled)
    restored[permutation] = scheduled

    assert torch.equal(restored, keys)
    # Device/rank placement is intentionally absent from the logical address.
    assigned_gpu = torch.tensor([1, 0, 1, 0])
    reassigned_gpu = 1 - assigned_gpu
    assert not torch.equal(assigned_gpu, reassigned_gpu)
    assert torch.equal(
        keys,
        logical_rng_keys(
            seed=17,
            iteration=9,
            trajectory=trajectory,
            path=path,
            draw=1,
            purpose=DrawPurpose.ACTION,
        ),
    )
    posterior = logical_rng_keys(
        seed=17,
        iteration=9,
        trajectory=trajectory,
        path=path,
        draw=1,
        purpose=DrawPurpose.POSTERIOR,
    )
    assert not torch.equal(keys, posterior)
    probability = torch.tensor([[0.2, 0.8]]).expand(4, -1)
    assert torch.equal(
        logical_categorical_from_valid_probability(probability, keys),
        logical_categorical_from_valid_probability(
            probability[permutation], scheduled
        )[permutation.argsort()],
    )


def test_logical_rng_namespaces_and_draw_ordinals_are_disjoint_on_address_grid() -> None:
    trajectory = torch.arange(32, dtype=torch.int64)
    path = child_path_keys(
        torch.zeros_like(trajectory), torch.arange(1, 33, dtype=torch.int64)
    )
    streams = []
    for iteration in (0, 11):
        for purpose in DrawPurpose:
            for draw in (0, 1, 7):
                streams.append(
                    logical_rng_keys(
                        seed=29,
                        iteration=iteration,
                        trajectory=trajectory,
                        path=path,
                        draw=draw,
                        purpose=purpose,
                    )
                )
    keys = torch.cat(streams)
    assert keys.unique().numel() == keys.numel()

    unnormalized = torch.tensor([[1.0, 3.0], [7.0, 2.0]])
    sample_keys = keys[:2]
    assert torch.equal(
        logical_categorical_from_valid_probability(unnormalized, sample_keys),
        logical_categorical_from_valid_probability(
            unnormalized / unnormalized.sum(1, keepdim=True), sample_keys
        ),
    )


def test_queue_hot_primitives_have_no_explicit_scalar_host_transfer() -> None:
    banned = {"cpu", "numpy", "tolist", "item"}
    for target in (
        ChunkedOccurrenceArena.gather,
        ChunkedOccurrenceArena.gather_many,
        ChunkedOccurrenceArena.scatter,
        ChunkedOccurrenceArena.scatter_many,
        ChunkedOccurrenceArena.transition,
        DeviceWorkQueue.enqueue,
        DeviceWorkQueue.peek,
        DeviceWorkQueue.dequeue,
        logical_rng_keys,
        logical_categorical_from_valid_probability,
        stable_partition_masked,
    ):
        syntax = ast.parse(textwrap.dedent(inspect.getsource(target)))
        used = {
            node.attr
            for node in ast.walk(syntax)
            if isinstance(node, ast.Attribute) and node.attr in banned
        }
        assert not used, f"{target.__qualname__} uses host-transfer calls {used}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_queue_arena_and_rng_execute_transactionally() -> None:
    """Exercise the Phase-1 primitives on-device with only tiny allocations."""

    device = torch.device("cuda", torch.cuda.current_device())
    fields = arena_fields(("state_ref", "trajectory_id", "path_key"))
    row_bytes = sum(field.row_bytes for field in fields.values())
    accountant = ByteAccountant(limit_bytes=8 * row_bytes)
    arena = ChunkedOccurrenceArena(
        capacity=10,
        chunk_nodes=4,
        device=device,
        fields=fields,
        accountant=accountant,
    )
    handles = arena.allocate(8)
    states = torch.arange(8, dtype=torch.int32, device=device) + 40
    trajectories = torch.arange(100, 108, dtype=torch.int64, device=device)
    paths = child_path_keys(
        torch.zeros(8, dtype=torch.int64, device=device),
        torch.arange(1, 9, dtype=torch.int64, device=device),
    )
    arena.scatter("state_ref", handles, states)
    arena.scatter("trajectory_id", handles, trajectories)
    arena.scatter("path_key", handles, paths)

    queue = DeviceWorkQueue(5, device)
    queue.enqueue(handles[:5], arena=arena)
    first = queue.dequeue(3, arena=arena)
    DeviceWorkQueue.mark_done(arena, first)
    # A dequeued prefix is the queue's bounded staging view and is valid only
    # through its last consumer; preserve the test observation before reuse.
    first_observed = first.cpu()
    queue.enqueue(handles[5:8], arena=arena)  # force ring wraparound
    wrapped = queue.dequeue(5, arena=arena)
    DeviceWorkQueue.mark_done(arena, wrapped)

    keys = logical_rng_keys(
        seed=17,
        iteration=3,
        trajectory=arena.gather("trajectory_id", handles),
        path=arena.gather("path_key", handles),
        draw=1,
        purpose=DrawPurpose.ACTION,
    )
    probability = torch.tensor([[0.25, 0.75]], device=device).expand(8, -1)
    samples = logical_categorical_from_valid_probability(probability, keys)
    partition = stable_partition_masked(
        handles,
        torch.tensor([0, 1, 0, 2, 1, 0, 2, 1], dtype=torch.int32, device=device),
        (0, 1, 2),
    )
    torch.cuda.synchronize(device)

    assert torch.equal(arena.gather("state_ref", handles).cpu(), states.cpu())
    assert torch.equal(first_observed, torch.tensor([0, 1, 2]))
    assert torch.equal(wrapped.cpu(), torch.tensor([3, 4, 5, 6, 7]))
    assert torch.equal(
        arena.gather("status", handles).cpu(), torch.full((8,), DONE)
    )
    assert samples.shape == (8,)
    assert torch.equal(partition.selected_count.cpu(), torch.tensor([3, 3, 2]))
    assert torch.equal(
        partition.candidate_handles[0][partition.valid_mask[0]].cpu(),
        torch.tensor([0, 2, 5]),
    )

    # Both hard overflows are decided from already-known capacities before any
    # CUDA write/status transition is enqueued.
    before_size = arena.size
    before_bytes = (arena.reserved_bytes, accountant.reserved_bytes)
    with pytest.raises(MemoryError, match="occurrence arena overflow"):
        arena.allocate(3)
    assert arena.size == before_size
    assert (arena.reserved_bytes, accountant.reserved_bytes) == before_bytes

    fresh = ChunkedOccurrenceArena(3, 3, device)
    fresh_handles = fresh.allocate(3)
    small_queue = DeviceWorkQueue(2, device)
    with pytest.raises(MemoryError, match="insertion-order overflow"):
        small_queue.enqueue(fresh_handles, arena=fresh)
    torch.cuda.synchronize(device)
    assert small_queue.host_count == 0
    assert torch.equal(
        fresh.gather("status", fresh_handles).cpu(), torch.full((3,), NEW)
    )
