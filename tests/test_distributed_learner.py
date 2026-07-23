from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from pbg.core.distributed_failure import (
    DistributedFailureAgreement,
    DistributedPhaseFailure,
)
from pbg.core.learner import (
    DistributedFixedRolloutUpdate,
    DistributedLearnerConfig,
    PhysicalPlanner,
    PhysicalShape,
    ReferenceCacheCandidate,
    ReusableIndexBuffer,
    allocate_fixed_statistical_counts,
    global_stratum_alphas,
    mode_loss,
    posterior_effective_hand_counts,
    select_reference_cache_set,
    validate_learner_input,
)
from pbg.core.optimization import FlatGradientBuffer, sync_scaled_gradients
from pbg.core.partition import ByteWorkContract, PartitionSet, PartitionSpec
from pbg.core.storage import (
    SamplingMetadata,
    SealedTensorBlock,
    SequentialPosteriorPool,
    TensorFieldSchema,
    TensorSchema,
    TrainingPartition,
    TrainingStorage,
)


@dataclass
class _Output:
    logits: torch.Tensor  # [B,Y,E]
    v: torch.Tensor  # [B,Y]


class _Model(torch.nn.Module):
    def __init__(self, value: float = 0.2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def output(self, inputs):
        x = inputs["x"]
        rows = torch.cat((x, x + 0.4), 1)
        v = self.weight * rows
        logits = torch.stack((self.weight * rows, -self.weight * rows), -1)
        return _Output(logits, v)


class _FailingModel(_Model):
    def __init__(self, value: float, *, fail_forward: bool) -> None:
        super().__init__(value)
        self.fail_forward = fail_forward

    def output(self, inputs):
        if self.fail_forward:
            raise RuntimeError("injected rank-local forward failure")
        return super().output(inputs)


class _Adapter:
    def _logp(self, output):
        return torch.log_softmax(output.logits, -1)

    def log_prob(self, output, events, legal_ids, legal_mask):
        del legal_ids
        logp = self._logp(output).masked_fill(~legal_mask[:, None, :], -torch.inf)
        index = events[:, :, None, None].expand(-1, -1, logp.shape[1], 1)
        return logp[:, None].expand(-1, events.shape[1], -1, -1).gather(-1, index).squeeze(-1)

    def entropy(self, output, legal_ids, legal_mask):
        del legal_ids
        logp = self._logp(output).masked_fill(~legal_mask[:, None, :], -torch.inf)
        probability = torch.where(torch.isfinite(logp), logp.exp(), 0.0)
        return -(probability * torch.where(torch.isfinite(logp), logp, 0.0)).sum(-1)

    def kl(self, output, reference, legal_ids, legal_mask):
        del legal_ids
        logp = self._logp(output).masked_fill(~legal_mask[:, None, :], -torch.inf)
        ref = self._logp(reference).masked_fill(~legal_mask[:, None, :], -torch.inf)
        probability = torch.where(torch.isfinite(logp), logp.exp(), 0.0)
        difference = torch.where(torch.isfinite(logp), logp - ref, 0.0)
        return (probability * difference).sum(-1)


def test_fixed_statistical_allocation_is_positive_exact_and_cost_aware() -> None:
    counts = allocate_fixed_statistical_counts(
        torch.tensor([0.6, 0.4], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.tensor([1.0, 16.0], dtype=torch.float64),
        64,
    )
    assert counts.tolist() == [54, 10]
    assert int(counts.sum()) == 64
    assert bool((counts > 0).all())


def test_fixed_statistical_allocation_rejects_uncovered_strata() -> None:
    with pytest.raises(ValueError, match="cannot cover"):
        allocate_fixed_statistical_counts(
            torch.ones(2), torch.ones(2), torch.ones(2), 1
        )


def _sealed(schema: TensorSchema, **columns: torch.Tensor) -> SealedTensorBlock:
    rows = next(iter(columns.values())).shape[0]
    return SealedTensorBlock(
        schema,
        columns,
        rows,
        schema.capacity_bytes(rows, chunks=int(rows > 0)),
    )


def _fixture(mode: str, *, rows: int = 3):
    input_schema = TensorSchema("learner_input", (TensorFieldSchema("x", torch.float32, (1,)),))
    inputs = _sealed(input_schema, x=torch.linspace(0.2, 0.8, rows)[:, None])
    common = {
        "owner": torch.zeros(rows, dtype=torch.int32),
        "event": torch.tensor([0, 1, 0], dtype=torch.int32)[:rows],
        "old_value": torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5]])[:rows],
        "old_log_prob": torch.tensor([[-0.5, -0.4], [-0.7, -0.6], [-0.3, -0.2]])[:rows],
        "legal_event_ids": torch.tensor([[0, 1]] * rows, dtype=torch.int32),
        "legal_event_mask": torch.tensor([[True, True]] * rows),
    }
    posterior_pool = None
    actions = None
    if mode == "normal":
        common["old_value"] = torch.tensor([0.1, 0.4, 0.3])[:rows]
        common["old_log_prob"] = torch.tensor([-0.5, -0.6, -0.3])[:rows]
        common["realized_type"] = torch.tensor([0, 3, 2], dtype=torch.int32)[:rows]
        common["lambda_return"] = torch.tensor([0.7, -0.2, 0.4])[:rows]
    elif mode == "sequential_rb":
        common["posterior_ref"] = torch.arange(rows, dtype=torch.int32)
        common["lambda_return"] = torch.tensor(
            [[0.7, 0.0, -0.2, 0.1], [0.3, 0.2, 0.0, -0.1], [0.2, 0.4, 0.1, 0.0]]
        )[:rows]
        posterior_pool = SequentialPosteriorPool(
            torch.tensor(
                [[0.5, 0.0, 0.25, 0.25], [0.0, 0.4, 0.6, 0.0], [0.2, 0.3, 0.0, 0.5]]
            )[:rows],
            torch.arange(rows, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
        )
    else:
        common.pop("event")
        common.pop("old_log_prob")
        common["log_weight"] = torch.zeros(rows, dtype=torch.float64)
        common["nominal_child_count"] = torch.tensor([2, 1, 2], dtype=torch.int32)[:rows]
        common["executed_child_count"] = torch.tensor([2, 1, 2], dtype=torch.int32)[:rows]
        common["value_hat"] = torch.tensor(
            [[0.7, 0.0, -0.2, 0.1], [0.3, 0.2, 0.0, -0.1], [0.2, 0.4, 0.1, 0.0]]
        )[:rows]
        counts = torch.tensor([2, 1, 2], dtype=torch.int32)[:rows]
        ptr = torch.cat(
            (torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32))
        )
        action_rows = int(ptr[-1])
        actions = {
            "action_ptr": ptr,
            "event": torch.arange(action_rows, dtype=torch.int32) % 2,
            "old_log_prob": torch.full((action_rows, 2), -0.5),
            "child_posterior": torch.arange(action_rows * 4, dtype=torch.float32).reshape(action_rows, 4) % 3 / 4,
            "advantage_hat": torch.linspace(-0.5, 0.7, action_rows * 4).reshape(action_rows, 4),
        }
    partition = TrainingPartition(0, inputs, common, actions)
    sampling = SamplingMetadata(
        {0: torch.tensor(math.log(rows), dtype=torch.float64)},
        {0: torch.ones(rows, dtype=torch.float64)} if mode == "branched_rb" else {},
        {0: torch.arange(rows, dtype=torch.int32)} if mode == "branched_rb" else {},
    )
    storage = TrainingStorage(mode, {0: partition}, sampling, posterior_pool)
    spec = PartitionSpec(
        0,
        "opaque",
        lambda _: 0,
        4,
        2,
        (0, 0, 1, 1),
        lambda model, values: model.output(values),
        ByteWorkContract(max_rows=8),
        input_schema,
    )
    return storage, PartitionSet((spec,)), partition


def _loss_cfg(**updates):
    values = dict(
        clip_eps=0.2,
        clip_eps_v=0.1,
        c_v=0.7,
        c_h=0.03,
        c_kl=0.2,
        entropy_schedule="const",
        value_loss="clipped",
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_training_storage_rejects_transient_branched_controller_predictions() -> None:
    storage, _, partition = _fixture("branched_rb")
    columns = dict(partition.columns)
    columns["frozen_trace_variance"] = torch.zeros(partition.inputs.rows)
    bad_partition = TrainingPartition(
        partition.partition_id,
        partition.inputs,
        columns,
        partition.action_columns,
    )
    with pytest.raises(ValueError, match="declared schemas"):
        TrainingStorage(
            storage.mode,
            {partition.partition_id: bad_partition},
            storage.sampling,
        )


def test_log_mass_empty_strata_and_log_domain_stability() -> None:
    logs = torch.tensor(
        [[10000.0, -torch.inf], [9999.0, -10000.0]], dtype=torch.float64
    )
    alpha = global_stratum_alphas(logs)
    assert torch.isfinite(alpha).all()
    assert alpha[0, 1] == 0
    assert alpha.sum() == pytest.approx(1.0)
    assert alpha[0, 0] / alpha[1, 0] == pytest.approx(math.e)


def test_arbitrary_index_draws_alias_distribution_and_reusable_storage() -> None:
    buffer = ReusableIndexBuffer(100_003, "cpu")
    generator = torch.Generator().manual_seed(7)
    probability = torch.tensor([0.4, 0.8, 1.0], dtype=torch.float64)
    alias = torch.tensor([2, 2, 2], dtype=torch.int32)
    draws = buffer.draw_alias(probability, alias, 100_003, generator)
    assert draws.dtype == torch.int32
    assert draws.untyped_storage().data_ptr() == buffer.tensor.untyped_storage().data_ptr()
    observed = torch.bincount(draws.long(), minlength=3).float() / draws.numel()
    expected = torch.tensor([0.4 / 3, 0.8 / 3, 1 - 1.2 / 3])
    assert torch.allclose(observed, expected, atol=0.006, rtol=0)


def test_physical_plan_preserves_order_repeats_and_one_row_tail() -> None:
    indices = torch.tensor([4, 1, 4, 0, 3, 3, 2], dtype=torch.int32)
    planner = PhysicalPlanner(
        (PhysicalShape(1, 10), PhysicalShape(3, 25)),
        tail_predictor=lambda rows: rows * 10,
    )
    spans = planner.plan(indices, 25)
    rebuilt = torch.cat([span.select(indices) for span in spans])
    assert torch.equal(rebuilt, indices)
    assert [span.shape_rows for span in spans] == [3, 3, 1]
    with pytest.raises(MemoryError, match="singleton"):
        planner.plan(indices, 9)


def test_reference_cache_exact_subset_charges_build_workspace_and_empty_fallback() -> None:
    candidates = (
        ReferenceCacheCandidate(0, 20, 70, 30, 25, 2.0, 12.0, 3.0),
        ReferenceCacheCandidate(1, 30, 10, 50, 20, 1.0, 10.0, 4.0),
    )
    choice = select_reference_cache_set(candidates, resident_base_bytes=10, hard_bytes=85)
    # Cache 0 alone would have resident 30 + build 70 = 100, so its apparently
    # faster training path is inadmissible.  Cache 1 is feasible and optimal.
    assert choice.selected == frozenset({1})
    empty = select_reference_cache_set(candidates, resident_base_bytes=10, hard_bytes=60)
    assert empty.selected == frozenset()


@pytest.mark.parametrize("mode", ["normal", "sequential_rb", "branched_rb"])
@pytest.mark.parametrize("value_loss", ["mse", "clipped"])
def test_mode_loss_cache_recompute_and_split_gradient_equality(mode, value_loss) -> None:
    storage, specs, partition = _fixture(mode)
    model = _Model()
    reference = _Model(-0.15)
    adapter = _Adapter()
    cfg = _loss_cfg(value_loss=value_loss)
    rows = torch.arange(partition.inputs.rows)

    def evaluate(selected, ref_output=None):
        inputs = {name: value[selected] for name, value in partition.inputs.columns.items()}
        output = model.output(inputs)
        if ref_output is None:
            ref_output = reference.output(inputs)
        return mode_loss(
            mode,
            specs[0],
            storage,
            partition,
            selected,
            output,
            adapter,
            cfg,
            reference=ref_output,
        ).per_decision

    with torch.no_grad():
        full_reference = reference.output(partition.inputs.columns)
    cached = _Output(full_reference.logits[rows], full_reference.v[rows])
    assert torch.allclose(evaluate(rows), evaluate(rows, cached), atol=0, rtol=0)

    model.zero_grad(set_to_none=True)
    evaluate(rows).sum().backward()
    unsplit = model.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    evaluate(rows[:2]).sum().backward()
    evaluate(rows[2:]).sum().backward()
    assert torch.allclose(model.weight.grad, unsplit, atol=2e-7, rtol=2e-6)


def test_branched_mode_loss_uses_legal_inert_events_for_ragged_padding() -> None:
    storage, specs, partition = _fixture("branched_rb")
    # Decision one owns only action row two (event zero). Its inactive padded
    # slot physically points at decision two's first action (event one), which
    # is deliberately illegal for decision one.
    partition.columns["legal_event_mask"][1] = torch.tensor([True, False])

    class StrictAdapter(_Adapter):
        def log_prob(self, output, events, legal_ids, legal_mask):
            selected_is_legal = (
                (legal_ids[:, None, :] == events[:, :, None])
                & legal_mask[:, None, :]
            ).any(-1)
            if not bool(selected_is_legal.all()):
                raise ValueError("ragged padding selected an illegal event")
            return super().log_prob(output, events, legal_ids, legal_mask)

    rows = torch.arange(partition.inputs.rows)
    output = _Model().output(partition.inputs.columns)
    result = mode_loss(
        "branched_rb",
        specs[0],
        storage,
        partition,
        rows,
        output,
        StrictAdapter(),
        _loss_cfg(),
    )
    assert torch.isfinite(result.per_decision).all()


def test_normal_mode_loss_matches_direct_production_formula() -> None:
    storage, specs, partition = _fixture("normal")
    model = _Model()
    reference = _Model(-0.15)
    adapter = _Adapter()
    cfg = _loss_cfg()
    rows = torch.arange(partition.inputs.rows)
    inputs = partition.inputs.columns
    output = model.output(inputs)
    reference_output = reference.output(inputs)
    actual = mode_loss(
        "normal",
        specs[0],
        storage,
        partition,
        rows,
        output,
        adapter,
        cfg,
        reference=reference_output,
    ).per_decision

    phi = torch.tensor(specs[0].exact_to_model)
    exact_type = partition.columns["realized_type"].long()
    model_type = phi[exact_type]
    batch = torch.arange(rows.numel())
    legal_ids = partition.columns["legal_event_ids"]
    legal_mask = partition.columns["legal_event_mask"]
    events = partition.columns["event"].long()[:, None]
    logp = adapter.log_prob(output, events, legal_ids, legal_mask)[:, 0][batch, model_type]
    old_logp = partition.columns["old_log_prob"]
    target = partition.columns["lambda_return"]
    old_value = partition.columns["old_value"]
    advantage = target - old_value
    ratio = torch.exp(logp - old_logp)
    clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    policy = -torch.minimum(ratio * advantage, clipped * advantage)
    value = output.v[batch, model_type]
    clipped_value = old_value + (value - old_value).clamp(-cfg.clip_eps_v, cfg.clip_eps_v)
    value_loss = cfg.c_v * torch.maximum(
        (value - target).square(), (clipped_value - target).square()
    )
    entropy = adapter.entropy(output, legal_ids, legal_mask)[batch, model_type]
    kl = adapter.kl(output, reference_output, legal_ids, legal_mask)[batch, model_type]
    expected = policy + value_loss - cfg.c_h * entropy + cfg.c_kl * kl
    assert torch.allclose(actual, expected, atol=0, rtol=0)


def test_flat_gradient_aliases_one_buffer_and_has_no_copy_flatten() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    flat = FlatGradientBuffer(model)
    flat.zero_()
    model(torch.ones(4, 2)).sum().backward()
    flat.assert_aliasing()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sync_scaled_gradients(1, flat_buffer=flat)
    assert flat.nbytes == sum(parameter.numel() * 4 for parameter in model.parameters())


def test_distributed_update_accepts_only_training_storage_and_arbitrary_tail() -> None:
    storage, specs, _ = _fixture("normal")
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    config = DistributedLearnerConfig(
        statistical_counts={0: 5},
        physical_shapes={0: (PhysicalShape(1, 10), PhysicalShape(2, 18))},
        available_transient_bytes=20,
    )
    learner = DistributedFixedRolloutUpdate(
        model,
        optimizer,
        scaler,
        specs,
        _Adapter(),
        _loss_cfg(c_kl=0.0),
        config,
    )
    result = learner(storage, torch.Generator().manual_seed(4), grad_clip=10.0)
    assert result.steps[0].optimizer_step_applied
    assert result.metrics["statistical_rows"] == 5
    assert result.metrics["physical_subspans"] == 3
    assert result.metrics["flat_gradient_bytes"] == 4
    assert result.metrics["posterior_effective_hands_rollout"] == 3
    assert result.metrics["posterior_effective_hand_presentations_learner"] == 5
    with pytest.raises(TypeError, match="only TrainingStorage"):
        validate_learner_input(SimpleNamespace(), specs)


def test_entropy_effective_hands_count_fresh_rows_and_learner_reuse() -> None:
    storage, specs, _ = _fixture("sequential_rb")
    assert storage.posterior_pool is not None
    storage.posterior_pool.posterior.fill_(0.25)
    counts = posterior_effective_hand_counts(storage)
    assert torch.equal(counts, torch.full((3,), 4.0, dtype=torch.float64))
    model = _Model()
    learner = DistributedFixedRolloutUpdate(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        torch.amp.GradScaler("cpu", enabled=False),
        specs,
        _Adapter(),
        _loss_cfg(c_kl=0.0),
        DistributedLearnerConfig(
            statistical_counts={0: 5},
            physical_shapes={0: (PhysicalShape(2, 18),)},
            available_transient_bytes=20,
        ),
    )
    result = learner(
        storage,
        torch.Generator().manual_seed(11),
        optimizer_steps=2,
        grad_clip=10.0,
    )
    assert result.metrics["posterior_effective_hands_rollout"] == 12
    assert result.metrics["posterior_effective_hand_presentations_learner"] == 40
    assert result.metrics["posterior_effective_hands_per_learner_sample"] == 4


def test_sampled_reference_cache_preserves_update_and_rng() -> None:
    storage, specs, _ = _fixture("normal")
    cached_model = _Model()
    uncached_model = _Model()
    cached = DistributedFixedRolloutUpdate(
        cached_model,
        torch.optim.SGD(cached_model.parameters(), lr=0.01),
        torch.amp.GradScaler("cpu", enabled=False),
        specs,
        _Adapter(),
        _loss_cfg(),
        DistributedLearnerConfig(
            statistical_counts={0: 5},
            physical_shapes={0: (PhysicalShape(2, 18),)},
            reference_build_shapes={0: (PhysicalShape(3, 20),)},
            available_transient_bytes=20,
            hard_bytes=1024,
            sampled_reference_cache=True,
        ),
        reference_model=_Model(-0.15),
    )
    uncached = DistributedFixedRolloutUpdate(
        uncached_model,
        torch.optim.SGD(uncached_model.parameters(), lr=0.01),
        torch.amp.GradScaler("cpu", enabled=False),
        specs,
        _Adapter(),
        _loss_cfg(),
        DistributedLearnerConfig(
            statistical_counts={0: 5},
            physical_shapes={0: (PhysicalShape(2, 18),)},
            available_transient_bytes=20,
            hard_bytes=1024,
        ),
        reference_model=_Model(-0.15),
    )
    cached_generator = torch.Generator().manual_seed(19)
    uncached_generator = torch.Generator().manual_seed(19)
    cached_result = cached(
        storage, cached_generator, optimizer_steps=2, grad_clip=10.0
    )
    uncached_result = uncached(
        storage, uncached_generator, optimizer_steps=2, grad_clip=10.0
    )
    assert torch.equal(cached_model.weight, uncached_model.weight)
    assert torch.equal(cached_generator.get_state(), uncached_generator.get_state())
    assert cached_result.metrics["reference_cache_partitions"] == 1
    assert cached_result.metrics["reference_cache_build_rows"] <= 3
    assert cached_result.metrics["reference_cache_build_calls"] == 1
    assert cached_result.metrics["reference_rows"] < uncached_result.metrics["reference_rows"]
    assert cached_result.metrics["optimizer_steps_applied_count"] == 2
    assert cached_result.metrics["optimizer_steps_nonfinite_count"] == 0
    assert cached_result.metrics["grad_clipped_step_fraction"] == 0.0
    assert (
        cached_result.metrics["grad_norm_pre_clip_step_min"]
        <= cached_result.metrics["grad_norm_pre_clip_step_mean"]
        <= cached_result.metrics["grad_norm_pre_clip_step_max"]
    )


def _storage_to(storage: TrainingStorage, device: torch.device) -> TrainingStorage:
    rebuilt = {}
    for partition_id, partition in storage.partitions.items():
        inputs = SealedTensorBlock(
            partition.inputs.schema,
            {name: value.to(device) for name, value in partition.inputs.columns.items()},
            partition.inputs.rows,
            partition.inputs.allocated_capacity_bytes,
        )
        rebuilt[partition_id] = TrainingPartition(
            partition_id,
            inputs,
            {name: value.to(device) for name, value in partition.columns.items()},
            None
            if partition.action_columns is None
            else {name: value.to(device) for name, value in partition.action_columns.items()},
            input_pools={
                name: SealedTensorBlock(
                    block.schema,
                    {field: value.to(device) for field, value in block.columns.items()},
                    block.rows,
                    block.allocated_capacity_bytes,
                )
                for name, block in partition.input_pools.items()
            },
        )
    sampling = SamplingMetadata(
        {key: value.to(device) for key, value in storage.sampling.log_z.items()},
        {key: value.to(device) for key, value in storage.sampling.alias_probability.items()},
        {key: value.to(device) for key, value in storage.sampling.alias_index.items()},
    )
    pool = storage.posterior_pool
    if pool is not None:
        pool = SequentialPosteriorPool(
            pool.posterior.to(device), pool.trajectory.to(device), pool.player.to(device)
        )
    return TrainingStorage(storage.mode, rebuilt, sampling, pool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["normal", "sequential_rb", "branched_rb"])
def test_tiny_cuda_bounded_update_cache_and_recompute(mode) -> None:
    device = torch.device("cuda")
    storage, specs, partition = _fixture(mode)
    storage = _storage_to(storage, device)
    model = _Model().to(device)
    reference = _Model(-0.15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    cache_bytes = partition.inputs.rows * 24  # logits [Y,2] plus value [Y], fp32
    config = DistributedLearnerConfig(
        statistical_counts={0: 5},
        physical_shapes={0: (PhysicalShape(1, 4096), PhysicalShape(2, 6144))},
        available_transient_bytes=8192,
        hard_bytes=32 * 1024 * 1024,
        reference_candidates=(
            ReferenceCacheCandidate(0, cache_bytes, 8192, 8192, 6144, 0.1, 2.0, 0.2),
        ),
    )
    learner = DistributedFixedRolloutUpdate(
        model,
        optimizer,
        scaler,
        specs,
        _Adapter(),
        _loss_cfg(),
        config,
        reference_model=reference,
    )
    result = learner(
        storage,
        torch.Generator(device=device).manual_seed(9),
        grad_clip=10.0,
    )
    recompute_model = _Model().to(device)
    recompute_reference = _Model(-0.15).to(device)
    recompute = DistributedFixedRolloutUpdate(
        recompute_model,
        torch.optim.Adam(recompute_model.parameters(), lr=0.001),
        torch.amp.GradScaler("cuda", enabled=False),
        specs,
        _Adapter(),
        _loss_cfg(),
        DistributedLearnerConfig(
            statistical_counts={0: 5},
            physical_shapes={0: (PhysicalShape(1, 4096), PhysicalShape(2, 6144))},
            available_transient_bytes=8192,
        ),
        reference_model=recompute_reference,
    )
    recompute_result = recompute(
        storage,
        torch.Generator(device=device).manual_seed(9),
        grad_clip=10.0,
    )
    assert result.steps[0].optimizer_step_applied
    assert recompute_result.steps[0].optimizer_step_applied
    assert torch.equal(model.weight, recompute_model.weight)
    assert result.metrics["reference_cache_partitions"] == 1
    assert result.metrics["reference_cache_build_rows"] == partition.inputs.rows
    assert result.metrics["training_peak_reserved_bytes"] > 0
    assert result.metrics["flat_gradient_bytes"] == 4
    assert recompute_result.metrics["reference_rows"] == 5
    print(
        json.dumps(
            {
                "mode": mode,
                "statistical_rows": result.metrics["statistical_rows"],
                "physical_subspans": result.metrics["physical_subspans"],
                "reference_cache_bytes": result.metrics["reference_cache_bytes"],
                "reference_cache_build_peak_reserved_bytes": result.metrics[
                    "reference_cache_build_peak_reserved_bytes"
                ],
                "training_peak_allocated_bytes": result.metrics[
                    "training_peak_allocated_bytes"
                ],
                "training_peak_reserved_bytes": result.metrics[
                    "training_peak_reserved_bytes"
                ],
                "flat_gradient_bytes": result.metrics["flat_gradient_bytes"],
            },
            sort_keys=True,
        )
    )


def _two_cuda_learner_worker(
    rank: int, world: int, port: int, output_dir: str, mode: str
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world),
            "LOCAL_RANK": str(rank),
            "NCCL_SOCKET_IFNAME": "lo",
        }
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        rows = 1 if rank == 0 else 3
        storage, specs, _ = _fixture(mode, rows=rows)
        storage = _storage_to(storage, torch.device("cuda", rank))
        # Constant rows make each sampled stratum mean exact despite arbitrary B.
        block = storage.partitions[0]
        block.inputs.columns["x"].fill_(0.5)
        block.columns["lambda_return"].fill_(float(2 * rank))
        block.columns["old_value"].zero_()
        block.columns["old_log_prob"].fill_(-math.log(2.0))
        block.columns["event"].zero_()
        if mode == "normal":
            block.columns["realized_type"].zero_()
        else:
            assert storage.posterior_pool is not None
            storage.posterior_pool.posterior.fill_(0.25)
        model = _Model(0.0).to(rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        learner = DistributedFixedRolloutUpdate(
            model,
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            specs,
            _Adapter(),
            _loss_cfg(c_kl=0.0, c_h=0.0, value_loss="mse"),
            DistributedLearnerConfig(
                statistical_counts={0: 3 + rank},
                physical_shapes={0: (PhysicalShape(1, 4096), PhysicalShape(2, 6144))},
                available_transient_bytes=8192,
            ),
        )
        result = learner(
            storage,
            torch.Generator(device=torch.device("cuda", rank)).manual_seed(20 + rank),
            grad_clip=100.0,
        )
        Path(output_dir, f"learner_{mode}_rank_{rank}.json").write_text(
            json.dumps(
                {
                    "weight": float(model.weight.detach()),
                    "mass": result.metrics["exact_objective_mass"],
                    "rows": result.metrics["statistical_rows"],
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs unavailable")
@pytest.mark.parametrize("mode", ("normal", "sequential_rb"))
def test_two_cuda_unequal_strata_exact_sum(tmp_path: Path, mode: str) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(
        _two_cuda_learner_worker,
        args=(2, port, str(tmp_path), mode),
        nprocs=2,
        join=True,
    )
    records = [
        json.loads((tmp_path / f"learner_{mode}_rank_{rank}.json").read_text())
        for rank in range(2)
    ]
    assert records[0]["weight"] == pytest.approx(records[1]["weight"], abs=0)
    assert [record["mass"] for record in records] == pytest.approx([0.25, 0.75])
    assert [record["rows"] for record in records] == [3.0, 4.0]


def _two_cuda_asymmetric_failure_worker(
    rank: int, world: int, port: int, output_dir: str, failure_phase: str
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world),
            "LOCAL_RANK": str(rank),
            "NCCL_SOCKET_IFNAME": "lo",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    try:
        if failure_phase == "before_targets":
            import pbg.core.training as training_module

            model = _Model(0.0).to(rank)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            cfg = SimpleNamespace(
                hardware=SimpleNamespace(precision="fp32"),
                rollout=SimpleNamespace(roots_per_gpu=1),
            )
            bundle = SimpleNamespace(
                model=model,
                rollout_backend=SimpleNamespace(execution_metrics=None),
            )

            def injected_build(*args, **kwargs):
                del args, kwargs
                if rank == 1:
                    raise RuntimeError("injected rank-local target failure")
                return object()

            training_module._build_iteration_data = injected_build
            agreement = DistributedFailureAgreement(torch.device("cuda", rank))
            try:
                training_module._one_iteration(
                    cfg,
                    bundle,
                    optimizer,
                    torch.amp.GradScaler("cuda", enabled=False),
                    torch.Generator(device=torch.device("cuda", rank)),
                    0,
                    failure_agreement=agreement,
                )
            except DistributedPhaseFailure as exc:
                record = {
                    "phase": exc.phase,
                    "failing_rank": exc.failing_rank,
                    "weight": float(model.weight.detach()),
                }
            else:
                record = {"unexpected_success": True}
            Path(
                output_dir, f"failure_{failure_phase}_rank_{rank}.json"
            ).write_text(json.dumps(record), encoding="utf-8")
            return

        storage, specs, _ = _fixture("normal", rows=2)
        storage = _storage_to(storage, torch.device("cuda", rank))
        model = _FailingModel(
            0.0,
            fail_forward=failure_phase == "before_backward" and rank == 1,
        ).to(rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        learner = DistributedFixedRolloutUpdate(
            model,
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            specs,
            _Adapter(),
            _loss_cfg(c_kl=0.0, c_h=0.0, value_loss="mse"),
            DistributedLearnerConfig(
                statistical_counts={0: 2},
                physical_shapes={0: (PhysicalShape(1, 4096),)},
                available_transient_bytes=8192,
            ),
        )
        if failure_phase == "before_mass" and rank == 1:
            def fail_plan(*args, **kwargs):
                del args, kwargs
                raise RuntimeError("injected rank-local singleton failure")

            learner._planners[0].plan = fail_plan
        try:
            learner(
                storage,
                torch.Generator(device=torch.device("cuda", rank)).manual_seed(
                    30 + rank
                ),
                grad_clip=100.0,
            )
        except DistributedPhaseFailure as exc:
            record = {
                "phase": exc.phase,
                "failing_rank": exc.failing_rank,
                "weight": float(model.weight.detach()),
            }
        else:
            record = {"unexpected_success": True}
        Path(output_dir, f"failure_{failure_phase}_rank_{rank}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs unavailable")
@pytest.mark.parametrize(
    "failure_phase", ("before_targets", "before_mass", "before_backward")
)
def test_two_cuda_asymmetric_mandatory_failure_is_rank_common(
    tmp_path: Path, failure_phase: str
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(
        _two_cuda_asymmetric_failure_worker,
        args=(2, port, str(tmp_path), failure_phase),
        nprocs=2,
        join=True,
    )
    records = [
        json.loads(
            (tmp_path / f"failure_{failure_phase}_rank_{rank}.json").read_text()
        )
        for rank in range(2)
    ]
    expected_phase = {
        "before_targets": "rollout, seal, completion, and targets",
        "before_mass": "mandatory learner singleton feasibility",
        "before_backward": "local backward step 0",
    }[failure_phase]
    assert [record["failing_rank"] for record in records] == [1, 1]
    assert [record["phase"] for record in records] == [expected_phase, expected_phase]
    assert [record["weight"] for record in records] == [0.0, 0.0]
