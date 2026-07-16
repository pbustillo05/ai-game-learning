from __future__ import annotations

import pytest
import torch

from pbg.core.distributed import (
    equal_rank_mean_baseline,
    global_mass_objective,
    stratum_objective_scale,
    weighted_decision_stats,
)
from pbg.core.observability import ExecutionMetrics


def test_exact_execution_counters_match_a_hand_computed_workload() -> None:
    metrics = ExecutionMetrics()
    metrics.record_policy_pack(
        "inference",
        "pre",
        rows=3,
        padded_tokens=30,
        padded_token_pairs=300,
    )
    metrics.record_model_call(
        "inference",
        "pre",
        2,
        padded_tokens=20,
        padded_token_pairs=200,
    )
    metrics.record_model_call(
        "inference",
        "pre",
        1,
        padded_tokens=10,
        padded_token_pairs=100,
    )
    metrics.record_queue("pre", depth=7, oldest_age=0.125)
    metrics.record_queue("pre", depth=4, oldest_age=0.250)
    metrics.record_forced_drain("pre", rows=1)
    metrics.record_cache(lookups=5, hits=3, evictions=1, saved_rows=4)
    metrics.record_evaluator(rows=6, rejection_work=19)
    metrics.record_host_synchronization(seconds=0.002, source="mask compaction")
    metrics.record_completion(rows=8)
    metrics.record_target_bytes(1_024)
    metrics.record_collective(
        nbytes=4_096, seconds=0.003, exposed_seconds=0.001
    )
    metrics.record_rank_wait(seconds=0.004)

    snapshot = metrics.snapshot()
    assert snapshot["execution_model_calls"] == 2
    assert snapshot["execution_model_rows"] == 3
    assert snapshot["execution_padded_tokens"] == 30
    assert snapshot["execution_padded_token_pairs"] == 300
    assert snapshot["execution_policy_pack_calls"] == 1
    assert snapshot["execution_policy_pack_rows"] == 3
    assert snapshot["execution_policy_pack_padded_tokens"] == 30
    assert snapshot["execution_policy_pack_padded_token_pairs"] == 300
    assert snapshot["execution_model_rows_max"] == 2
    assert snapshot["execution_padded_tokens_max"] == 20
    assert snapshot["execution_padded_token_pairs_max"] == 200
    assert snapshot["execution_queue_depth_max"] == 7
    assert snapshot["execution_queue_oldest_age_max"] == pytest.approx(0.250)
    assert snapshot["execution_forced_drains"] == 1
    assert snapshot["execution_forced_drain_rows"] == 1
    assert snapshot["execution_cache_lookups"] == 5
    assert snapshot["execution_cache_hits"] == 3
    assert snapshot["execution_cache_evictions"] == 1
    assert snapshot["execution_cache_saved_rows"] == 4
    assert snapshot["execution_evaluator_calls"] == 1
    assert snapshot["execution_evaluator_rows"] == 6
    assert snapshot["execution_evaluator_rejection_work"] == 19
    assert snapshot["execution_host_synchronizations"] == 1
    assert snapshot["execution_host_sync_mask_compaction_host_synchronizations"] == 1
    assert snapshot[
        "execution_host_sync_mask_compaction_host_synchronization_seconds"
    ] == pytest.approx(0.002)
    assert snapshot["execution_completion_batches"] == 1
    assert snapshot["execution_completion_rows"] == 8
    assert snapshot["execution_target_bytes"] == 1_024
    assert snapshot["execution_target_bytes_max"] == 1_024
    assert snapshot["execution_collective_calls"] == 1
    assert snapshot["execution_collective_bytes"] == 4_096
    assert snapshot["execution_collective_exposed_seconds"] == pytest.approx(0.001)
    assert snapshot["execution_rank_wait_seconds"] == pytest.approx(0.004)
    assert snapshot["execution_inference_pre_model_calls"] == 2
    assert snapshot["execution_inference_pre_batch_size_1_calls"] == 1
    assert snapshot["execution_inference_pre_batch_size_2_calls"] == 1


def test_model_call_counter_does_not_truncate_long_iterations() -> None:
    metrics = ExecutionMetrics()
    for _ in range(5_000):
        metrics.record_model_call(
            "training",
            "post_6",
            1,
            padded_tokens=1_081,
            padded_token_pairs=1_081**2,
        )
    snapshot = metrics.snapshot()
    assert snapshot["execution_model_calls"] == 5_000
    assert snapshot["execution_training_post_6_model_calls"] == 5_000
    assert snapshot["execution_training_post_6_batch_size_1_calls"] == 5_000


def test_global_decision_mass_gradient_differs_from_equal_rank_mean() -> None:
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    rank_zero = weighted_decision_stats(
        torch.stack(((theta - 0.0).square(),)),
        torch.ones(1, dtype=torch.float64),
    )
    rank_one = weighted_decision_stats(
        torch.stack(tuple((theta - 2.0).square() for _ in range(3))),
        torch.ones(3, dtype=torch.float64),
    )
    numerators = torch.stack((rank_zero.numerator, rank_one.numerator))
    masses = torch.stack((rank_zero.mass, rank_one.mass))

    global_loss = global_mass_objective(numerators, masses)
    equal_rank_loss = equal_rank_mean_baseline(numerators, masses)
    global_gradient = torch.autograd.grad(global_loss, theta, retain_graph=True)[0]
    equal_rank_gradient = torch.autograd.grad(equal_rank_loss, theta)[0]

    assert float(global_loss.detach()) == pytest.approx(3.0)
    assert float(equal_rank_loss.detach()) == pytest.approx(2.0)
    assert float(global_gradient) == pytest.approx(-3.0)
    assert float(equal_rank_gradient) == pytest.approx(-2.0)
    assert float(global_gradient) != float(equal_rank_gradient)
    assert float(stratum_objective_scale(masses[0], masses.sum())) == pytest.approx(
        0.25
    )
