from __future__ import annotations

import math

import pytest
import torch

from pbg.core.execution import ByteAccountant
from pbg.core.partition import (
    ByteWorkContract,
    CompletionSpec,
    PartitionSet,
    PartitionSpec,
)
from pbg.core.storage import (
    PagedTensorArena,
    PartitionRolloutBlock,
    RecyclableTensorPool,
    RolloutStorage,
    SealedTensorBlock,
    TensorFieldSchema,
    TensorSchema,
    TrainingPartition,
    occurrence_schema,
)
from pbg.core.targets import (
    build_branched_rb_targets,
    build_normal_targets,
    build_sequential_rb_targets,
    plan_branched_frontier,
)
from pbg.core.topology import BranchedTreeSpan, SequentialDecisionCSR
from pbg.games.poker.partitions import assert_poker_storage_widths, poker_partition_specs


def _sealed(schema: TensorSchema, **columns: torch.Tensor) -> SealedTensorBlock:
    rows = next(iter(columns.values())).shape[0]
    return SealedTensorBlock(
        schema,
        columns,
        rows,
        schema.capacity_bytes(rows, chunks=int(rows > 0)),
    )


def _input(rows: int, width: int = 2) -> SealedTensorBlock:
    schema = TensorSchema(
        "input", (TensorFieldSchema("model_input", torch.float32, (width,)),)
    )
    return _sealed(schema, model_input=torch.arange(rows * width).reshape(rows, width).float())


def _spec(exact: int, model: int, mapping: tuple[int, ...]) -> PartitionSet:
    return PartitionSet(
        (
            PartitionSpec(
                0,
                "opaque",
                lambda _: 0,
                exact,
                model,
                mapping,
                lambda value: value,
                ByteWorkContract(max_rows=8),
                _input(0, model).schema,
            ),
        )
    )


def test_schema_transaction_page_slack_and_exact_multipage_seal() -> None:
    schema = TensorSchema(
        "rows",
        (
            TensorFieldSchema("id", torch.int32, alignment=8),
            TensorFieldSchema("payload", torch.float32, (3,), alignment=16),
        ),
        chunk_header_bytes=32,
        allocator_slack_bytes=16,
    )
    per_chunk = schema.capacity_bytes(2)
    destination = schema.capacity_bytes(5)
    scratch = schema.payload_bytes(1)
    accountant = ByteAccountant(3 * per_chunk + destination + scratch)
    arena = PagedTensorArena(schema, 2, "cpu", accountant=accountant)
    values = {
        "id": torch.arange(5, dtype=torch.int32),
        "payload": torch.arange(15, dtype=torch.float32).reshape(5, 3),
    }
    ids = arena.append(values)
    assert torch.equal(ids, torch.arange(5))
    assert arena.page_count == 3
    assert arena.capacity_rows == 6
    assert arena.page_slack_bytes > schema.payload_bytes(1)
    source_bytes = arena.allocated_capacity_bytes

    block, metrics = arena.seal(scratch_rows=1)
    assert torch.equal(block["id"], values["id"])
    assert torch.equal(block["payload"], values["payload"])
    assert block.tensor_storage_bytes == block.logical_payload_bytes == schema.payload_bytes(5)
    assert metrics.source_pages == 3
    assert metrics.peak_accounted_bytes == source_bytes + destination + scratch
    assert accountant.high_water_bytes == metrics.peak_accounted_bytes
    assert accountant.reserved_bytes == destination


def test_paged_reservation_is_atomic_and_recyclable_slots_follow_last_consumer() -> None:
    schema = TensorSchema(
        "live",
        (TensorFieldSchema("payload", torch.float32, (2,), alignment=16),),
        chunk_header_bytes=32,
        allocator_slack_bytes=16,
    )
    page_bytes = schema.capacity_bytes(2)
    accountant = ByteAccountant(page_bytes)
    arena = PagedTensorArena(schema, 2, "cpu", accountant=accountant)
    with pytest.raises(MemoryError):
        arena.reserve_rows(3)
    assert arena.size == arena.capacity_rows == arena.page_count == 0
    assert accountant.snapshot() == {}

    pool = RecyclableTensorPool(schema, 2, "cpu")
    first = pool.allocate(2)
    pool.write(first, {"payload": torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    assert pool.capacity_rows == pool.live_rows == pool.high_water_live_rows == 2
    released = first[:1]
    pool.release_rows(released)
    with pytest.raises(RuntimeError, match="last consumer"):
        pool.gather("payload", released)
    reused = pool.allocate(1)
    assert reused.item() == released.item()
    pool.write(reused, {"payload": torch.tensor([[9.0, 8.0]])})
    assert torch.equal(pool.gather("payload", reused), torch.tensor([[9.0, 8.0]]))
    assert pool.capacity_rows == 2


def test_byte_accountant_group_reservation_rolls_back_before_allocation() -> None:
    accountant = ByteAccountant(10)
    accountant.reserve("resident", 4)
    before = accountant.snapshot()
    with pytest.raises(MemoryError):
        accountant.reserve_many({"destination": 5, "scratch": 2})
    assert accountant.snapshot() == before
    with pytest.raises(RuntimeError):
        with accountant.transaction({"destination": 3, "scratch": 2}):
            raise RuntimeError("allocation failed")
    assert accountant.snapshot() == before


def test_sequential_csr_seals_stable_root_ordinal_order() -> None:
    topology = SequentialDecisionCSR.build(
        root_id=torch.tensor([1, 0, 1, 0]),
        decision_ordinal=torch.tensor([1, 1, 0, 0]),
        decision_partition=torch.tensor([0, 0, 1, 1]),
        decision_row=torch.tensor([1, 0, 0, 1]),
        leaf_root_id=torch.tensor([1, 0]),
        leaf_row=torch.tensor([1, 0]),
        root_count=2,
    )
    assert torch.equal(topology.trajectory_ptr, torch.tensor([0, 2, 4], dtype=torch.int32))
    assert torch.equal(topology.decision_partition, torch.tensor([1, 0, 1, 0], dtype=torch.int32))
    assert torch.equal(topology.decision_row, torch.tensor([1, 0, 0, 1], dtype=torch.int32))
    assert torch.equal(topology.leaf_row, torch.tensor([0, 1], dtype=torch.int32))
    assert topology.nbytes == 4 * (2 * 4 + 2 * 2 + 1)


def test_dynamic_occurrence_schema_and_protected_root_action_prefix() -> None:
    fields = occurrence_schema(n_players=2, mode="branched_rb").by_name
    for name in (
        "nominal_count",
        "executed_count",
        "log_player_sum",
        "log_ancestor_product",
        "queue_cost",
    ):
        assert name in fields

    topology = BranchedTreeSpan(
        parent=torch.tensor([-1, 0, 0, 0], dtype=torch.int32),
        child_begin=torch.tensor([1, -1, -1, -1], dtype=torch.int32),
        degree=torch.tensor([3, 0, 0, 0], dtype=torch.int32),
        decision_partition=torch.tensor([0, -1, -1, -1], dtype=torch.int32),
        decision_row=torch.tensor([0, -1, -1, -1], dtype=torch.int32),
        leaf_row=torch.tensor([-1, 0, 1, 2], dtype=torch.int32),
        root_target_degree=2,
        pre_refill_node_count=3,
    )
    topology.validate(
        partition_rows={0: 1},
        action_ptr={0: torch.tensor([0, 2], dtype=torch.int32)},
        leaf_rows=3,
    )
    assert topology.protected_root_degree == 2
    assert topology.pre_refill_nodes == 3
    assert topology.nbytes == 6 * 4 * 4 + 8


def test_training_partition_resolves_sparse_immutable_input_pool() -> None:
    input_schema = TensorSchema(
        "decision_input",
        (
            TensorFieldSchema("model_input", torch.float32, (2,)),
            TensorFieldSchema("folded_summary_ref", torch.int32),
        ),
    )
    inputs = _sealed(
        input_schema,
        model_input=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        folded_summary_ref=torch.tensor([1, 0], dtype=torch.int32),
    )
    summary_schema = TensorSchema(
        "summary", (TensorFieldSchema("beta", torch.float32, (3,)),)
    )
    summaries = _sealed(
        summary_schema,
        beta=torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
    )
    partition = TrainingPartition(
        0,
        inputs,
        {"owner": torch.tensor([0, 1], dtype=torch.int32)},
        input_pools={"folded_summary": summaries},
    )
    gathered = partition.gather_inputs(torch.tensor([0, 1]))
    assert "folded_summary_ref" not in gathered
    assert torch.equal(
        gathered["folded_summary.beta"],
        torch.tensor([[20.0, 21.0, 22.0], [10.0, 11.0, 12.0]]),
    )


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_normal_streaming_player_clock_lambda_return(lam: float) -> None:
    decision_schema = TensorSchema(
        "normal_decision",
        (
            TensorFieldSchema("owner", torch.int32),
            TensorFieldSchema("event", torch.int32),
            TensorFieldSchema("realized_type", torch.int32),
            TensorFieldSchema("old_value", torch.float32),
            TensorFieldSchema("old_log_prob", torch.float32),
        ),
    )
    decisions = _sealed(
        decision_schema,
        owner=torch.tensor([0, 0], dtype=torch.int32),
        event=torch.tensor([1, 2], dtype=torch.int32),
        realized_type=torch.tensor([0, 0], dtype=torch.int32),
        old_value=torch.tensor([10.0, 20.0]),
        old_log_prob=torch.tensor([-0.1, -0.2]),
    )
    leaf_schema = TensorSchema(
        "normal_leaf", (TensorFieldSchema("reward", torch.float32, (2,)),)
    )
    rollout = RolloutStorage(
        "normal",
        {0: PartitionRolloutBlock(0, _input(2), decisions)},
        SequentialDecisionCSR(
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        ),
        _sealed(leaf_schema, reward=torch.tensor([[3.0, -3.0]])),
    )
    training = build_normal_targets(rollout, _spec(2, 2, (0, 1)), lam)
    result = training.partitions[0].columns["lambda_return"]
    expected_first = 3.0 if lam == 1.0 else 20.0
    assert torch.equal(result, torch.tensor([expected_first, 3.0]))
    assert "advantage" not in training.partitions[0].columns
    assert "v_target" not in training.partitions[0].columns


class _Completion:
    def __init__(self, posterior: torch.Tensor, value: torch.Tensor) -> None:
        self.posterior = posterior
        self.value = value

    def __call__(self, leaves, *, rows):
        return {"p_hat": self.posterior[rows], "v_hat": self.value[rows]}


def _completion(posterior: torch.Tensor, value: torch.Tensor) -> CompletionSpec:
    return CompletionSpec(
        ("key",),
        ("exact",),
        ("p_hat", "v_hat"),
        _Completion(posterior, value),
        0,
        key_fields=("key",),
    )


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_sequential_rb_exact_targets_deduplicate_posterior_and_use_phi(lam: float) -> None:
    # Exact width four, natural model width two: this is the H -> K contract.
    decision_schema = TensorSchema(
        "seq_decision",
        (
            TensorFieldSchema("owner", torch.int32),
            TensorFieldSchema("event", torch.int32),
            TensorFieldSchema("old_value", torch.float32, (2,)),
            TensorFieldSchema("old_log_prob", torch.float32, (2,)),
        ),
    )
    decisions = _sealed(
        decision_schema,
        owner=torch.tensor([0, 0], dtype=torch.int32),
        event=torch.tensor([1, 2], dtype=torch.int32),
        old_value=torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        old_log_prob=torch.zeros(2, 2),
    )
    leaf_schema = TensorSchema("rb_leaf", (TensorFieldSchema("key", torch.int64),))
    rollout = RolloutStorage(
        "sequential_rb",
        {0: PartitionRolloutBlock(0, _input(2), decisions)},
        SequentialDecisionCSR(
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        ),
        _sealed(leaf_schema, key=torch.tensor([7], dtype=torch.int64)),
    )
    posterior = torch.tensor([[[1.0, 0.0, 2.0, 1.0]]])
    value = torch.tensor([[[2.0, 99.0, 4.0, 8.0]]])
    training = build_sequential_rb_targets(
        rollout,
        _spec(4, 2, (0, 0, 1, 1)),
        _completion(posterior, value),
        lam,
    )
    assert training.posterior_pool is not None
    assert training.posterior_pool.posterior.shape == (1, 4)
    assert torch.equal(training.partitions[0].columns["posterior_ref"], torch.tensor([0, 0], dtype=torch.int32))
    targets = training.partitions[0].columns["lambda_return"]
    assert torch.equal(targets[1], torch.tensor([2.0, 0.0, 4.0, 8.0]))
    expected_first = (
        torch.tensor([2.0, 0.0, 4.0, 8.0])
        if lam == 1.0
        else torch.tensor([30.0, 0.0, 40.0, 40.0])
    )
    assert torch.equal(targets[0], expected_first)
    assert targets.shape[1] == 4
    assert decisions["old_value"].shape[1] == 2


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_branched_bounded_targets_multiple_children_and_zero_mass(lam: float) -> None:
    decision_schema = TensorSchema(
        "branch_decision",
        (
            TensorFieldSchema("owner", torch.int32),
            TensorFieldSchema("old_value", torch.float32, (2,)),
            TensorFieldSchema("log_weight", torch.float32),
        ),
    )
    action_schema = TensorSchema(
        "branch_action",
        (
            TensorFieldSchema("event", torch.int32),
            TensorFieldSchema("old_log_prob", torch.float32, (2,)),
        ),
    )
    decisions = _sealed(
        decision_schema,
        owner=torch.tensor([0], dtype=torch.int32),
        old_value=torch.tensor([[5.0, 7.0]]),
        log_weight=torch.tensor([0.0]),
    )
    actions = _sealed(
        action_schema,
        event=torch.tensor([1, 2], dtype=torch.int32),
        old_log_prob=torch.zeros(2, 2),
    )
    topology = BranchedTreeSpan(
        parent=torch.tensor([-1, 0, 0], dtype=torch.int32),
        child_begin=torch.tensor([1, -1, -1], dtype=torch.int32),
        degree=torch.tensor([2, 0, 0], dtype=torch.int32),
        decision_partition=torch.tensor([0, -1, -1], dtype=torch.int32),
        decision_row=torch.tensor([0, -1, -1], dtype=torch.int32),
        leaf_row=torch.tensor([-1, 0, 1], dtype=torch.int32),
    )
    assert plan_branched_frontier(topology, leaf_microbatch_rows=2).peak_slots == 2
    leaf_schema = TensorSchema("branch_leaf", (TensorFieldSchema("key", torch.int64),))
    rollout = RolloutStorage(
        "branched_rb",
        {
            0: PartitionRolloutBlock(
                0,
                _input(1),
                decisions,
                actions,
                torch.tensor([0, 2], dtype=torch.int32),
            )
        },
        topology,
        _sealed(leaf_schema, key=torch.tensor([1, 2], dtype=torch.int64)),
    )
    posterior = torch.tensor([[[1.0, 0.0]], [[3.0, 0.0]]])
    value = torch.tensor([[[2.0, 100.0]], [[4.0, 200.0]]])
    training = build_branched_rb_targets(
        rollout,
        _spec(2, 2, (0, 1)),
        _completion(posterior, value),
        lam,
        leaf_microbatch_rows=2,
    )
    partition = training.partitions[0]
    assert torch.equal(partition.columns["value_hat"], torch.tensor([[3.5, 0.0]]))
    assert partition.action_columns is not None
    assert torch.equal(
        partition.action_columns["child_posterior"], posterior[:, 0]
    )
    if lam == 1.0:
        expected = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    else:
        expected = torch.tensor([[-3.0, -7.0], [-1.0, -7.0]])
    assert torch.equal(partition.action_columns["advantage_hat"], expected)
    assert "q_hat" not in partition.columns
    assert "parent_posterior" not in partition.columns
    assert torch.equal(training.sampling.log_z[0], torch.tensor(0.0, dtype=torch.float64))


def test_branched_root_uses_ordinary_declared_multiplicity() -> None:
    topology = BranchedTreeSpan(
        parent=torch.tensor([-1, 0], dtype=torch.int32),
        child_begin=torch.tensor([1, -1], dtype=torch.int32),
        degree=torch.tensor([1, 0], dtype=torch.int32),
        decision_partition=torch.tensor([0, -1], dtype=torch.int32),
        decision_row=torch.tensor([0, -1], dtype=torch.int32),
        leaf_row=torch.tensor([-1, 0], dtype=torch.int32),
    )
    topology.validate(leaf_rows=1)
    assert int(topology.degree[0]) == 1


def test_full_deck_poker_natural_model_and_exact_target_widths() -> None:
    specs = poker_partition_specs(
        rank_count=13,
        n_players=6,
        router=lambda _: 0,
        forwards={partition_id: (lambda rows: rows) for partition_id in range(6)},
    )
    pre = specs[0]
    assert pre.model_width == 169
    assert pre.exact_width == 1326
    assert pre.input_schema.by_name["range_weights"].shape == (6, 169)
    for live in range(2, 7):
        post = specs[live - 1]
        assert post.model_width == post.exact_width == 1326
        assert post.input_schema.by_name["range_weights"].shape == (live, 1326)
        assert ("folded_summary_ref" in post.input_schema.by_name) == (live < 6)
    assert_poker_storage_widths(
        specs,
        partition_id=0,
        model_rows=torch.zeros(3, 169),
        exact_rows=torch.zeros(3, 1326),
    )
    with pytest.raises(ValueError, match="natural model width 169"):
        assert_poker_storage_widths(
            specs, partition_id=0, model_rows=torch.zeros(3, 1326)
        )
