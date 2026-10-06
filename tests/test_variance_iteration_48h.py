from __future__ import annotations

import torch

from experiments.variance_iteration_48h.protocol import (
    direct_clipped_actor_loss,
    gradient_variance_from_gram,
    logical_seed,
    percentile_interval,
    signed_moment_clipped_actor_loss,
)
from experiments.variance_iteration_48h.reference import signed_batch_losses


def test_signed_moment_cache_matches_direct_with_both_advantage_signs() -> None:
    ratio = torch.tensor([[0.6, 0.9, 1.1, 1.7], [1.5, 0.5, 1.0, 1.3]])
    advantage = torch.tensor([[2.0, -3.0, 0.0, -0.25], [-1.0, 4.0, -2.0, 3.0]])
    posterior = torch.tensor([[0.1, 0.2, 0.3, 0.4], [4.0, 3.0, 2.0, 1.0]])
    cached = signed_moment_clipped_actor_loss(
        ratio,
        advantage.clamp_min(0),
        advantage.clamp_max(0),
        posterior,
        0.2,
    )
    direct = direct_clipped_actor_loss(ratio, advantage, posterior, 0.2)
    torch.testing.assert_close(cached, direct, rtol=1e-6, atol=1e-7)


def test_completion_world_signed_cache_matches_sparse_direct_average() -> None:
    ratio = torch.tensor([[0.7, 1.4, 0.9], [1.3, 0.6, 1.0]])
    world_hand = torch.tensor([[0, 0, 1, 1, 2], [2, 1, 1, 0, 0]])
    world_advantage = torch.tensor(
        [[2.0, -1.0, -3.0, 4.0, 0.5], [-2.0, 1.0, -4.0, 3.0, -1.0]]
    )
    rows, width = ratio.shape
    count = torch.zeros((rows, width))
    positive = torch.zeros_like(count)
    negative = torch.zeros_like(count)
    count.scatter_add_(1, world_hand, torch.ones_like(world_advantage))
    positive.scatter_add_(1, world_hand, world_advantage.clamp_min(0))
    negative.scatter_add_(1, world_hand, world_advantage.clamp_max(0))
    moments = {
        "posterior": count / world_hand.shape[1],
        "positive": positive / count.clamp_min(1),
        "negative": negative / count.clamp_min(1),
        "world_hand": world_hand.to(torch.int16),
        "world_advantage": world_advantage,
    }
    cached, direct = signed_batch_losses(ratio, moments, clip_epsilon=0.2)
    torch.testing.assert_close(cached, direct, rtol=1e-6, atol=1e-7)


def test_gram_variance_matches_direct_and_bootstrap_multiplicity() -> None:
    gradients = torch.tensor(
        [[1.0, 2.0, -1.0], [3.0, 0.0, 2.0], [-2.0, 1.0, 4.0]],
        dtype=torch.float64,
    )
    gram = gradients @ gradients.T
    direct = ((gradients - gradients.mean(0)).square().sum()) / 2
    torch.testing.assert_close(gradient_variance_from_gram(gram), direct)

    multiplicity = torch.tensor([2, 0, 1])
    resampled = gradients[torch.tensor([0, 0, 2])]
    bootstrap_direct = ((resampled - resampled.mean(0)).square().sum()) / 2
    torch.testing.assert_close(
        gradient_variance_from_gram(gram, multiplicity), bootstrap_direct
    )


def test_logical_seed_is_stable_and_stream_separated() -> None:
    first = logical_seed(123, "train", "normal", 4)
    assert first == logical_seed(123, "train", "normal", 4)
    assert first != logical_seed(123, "reference", "normal", 4)
    assert first != logical_seed(123, "train", "normal", 5)
    assert 0 <= first < 2**63


def test_percentile_interval_uses_stable_float64_quantiles() -> None:
    low, high = percentile_interval([0.0, 1.0, 2.0, 3.0])
    assert 0.0 <= low < high <= 3.0
