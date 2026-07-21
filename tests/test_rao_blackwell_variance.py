from __future__ import annotations

import itertools

import pytest
import torch

from pbg.eval.rao_blackwell_variance import (
    SufficientStatistics,
    assigned_jobs,
    canonical_jobs,
    consecutive_blocks,
    cpu_oracle,
    logical_seed,
    theoretical_variance,
    weighted_ratio_moments,
)


def test_weighted_ratio_moments_reduce_to_uniform_sample_formulas() -> None:
    values = torch.tensor([[1.0, 3.0, 8.0]], dtype=torch.float64)
    weights = torch.ones_like(values)
    mean, world, mean_variance = weighted_ratio_moments(values, weights)
    assert mean.item() == pytest.approx(4.0)
    assert world.item() == pytest.approx(values.var(unbiased=True).item())
    assert mean_variance.item() == pytest.approx(world.item() / 3)


def test_sufficient_statistics_apply_both_noise_corrections() -> None:
    stats = SufficientStatistics()
    stats.update_trajectory([1.0, 2.0], [4.0, 5.0], [0.2, 0.3], 3.0, 8.0, 0.7)
    stats.update_trajectory([4.0], [6.0], [0.4], 4.0, 6.0, 0.8)
    result = stats.components()
    assert result["minibatch_N_1_raw"] == pytest.approx((3 * 21 - 49) / 4)
    assert result["minibatch_noise_correction"] == pytest.approx((3 * 0.9 - 1.5) / 4)
    assert result["rollout_1"] == pytest.approx(0.5 - 1.5 / 2)
    assert result["montecarlo_N_inf"] == pytest.approx(14 / 4)
    assert result["montecarlo_N_1"] == pytest.approx(3 * 15 / 4)


def test_cpu_oracle_checks_shared_completion_covariance() -> None:
    result = cpu_oracle()
    assert result["direct_completion"] == pytest.approx(result["predicted_completion"])


def test_candidate_formula_has_exact_axis_dependences() -> None:
    kwargs = dict(rollout_1=24.0, minibatch_N_1=12.0, montecarlo_N_inf=8.0, montecarlo_N_1=20.0)
    value = theoretical_variance(**kwargs, N=4, B=2, n=5)
    assert value == pytest.approx(24 / 4 + 12 / 2 + ((1 - 1 / 2) * 8 + 20 / 2) / 5)


def test_jobs_are_rank_invariant_and_blocks_are_disjoint() -> None:
    checkpoints = [37, 296]
    expected = set(canonical_jobs(checkpoints, 5))
    actual = set(itertools.chain.from_iterable(assigned_jobs(checkpoints, 5, rank, 3) for rank in range(3)))
    assert actual == expected
    assert consecutive_blocks(12, 3) == ((0, 3), (3, 6), (6, 9), (9, 12))
    assert logical_seed(1, "completion", 37, 2) == logical_seed(1, "completion", 37, 2)
    assert logical_seed(1, "completion", 37, 2) != logical_seed(1, "rollout", 37, 2)
