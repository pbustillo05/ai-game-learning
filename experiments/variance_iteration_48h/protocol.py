"""Pure protocol helpers for the strict 48-hour experiment.

The production runner lives in :mod:`runner`; this module deliberately keeps
the statistical identities and seed construction CPU-testable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

SCHEMA = "pbg-variance-iteration-48h-v1"

REPOSITORY = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL: dict[str, Any] = {
    "schema": SCHEMA,
    "received_utc": "2026-09-22T00:00:00Z",
    "checkpoints": {
        "normal10k": {
            "path": "runs/experiments/nlhe_hu_32bb_normal_60h_20260812/checkpoint_00010000.pt",
            "config": "configs/experiments/nlhe_hu_32bb_normal_60h_20260812.yaml",
        },
        "rb10k": {
            "path": "runs/experiments/nlhe_hu_32bb_rb_dualclip5_60h_20260814/checkpoint_00010000.pt",
            "config": "configs/experiments/nlhe_hu_32bb_rb_dualclip5_60h_20260814.yaml",
        },
    },
    "arms": ("sampled", "rb1024"),
    "gradient": {
        "public_rollouts": 384,
        "decision_records": 256,
        "repetitions": 32,
        "lambda": 1.0,
        "rb_completions": 1024,
        "actor_policy_clip": 1_000_000.0,
    },
    "optimization": {
        "public_rollouts": 384,
        "decision_records": 256,
        "repetitions": 6,
        "maximum_steps": 100,
        "endpoints": (0, 10, 20, 38, 100),
        "primary_endpoint": 38,
        "rb_completions": 1024,
    },
    "reference": {
        "public_rollouts": 256,
        "completions": 4096,
        "precision_public_rollouts": 32,
        "precision_completions": (4096, 8192),
    },
    "bootstrap_draws": 20_000,
    "base_seed": 48_202_609_22,
}


def logical_seed(base: int, purpose: str, *address: object) -> int:
    """Return a stable nonnegative int64 seed independent of process rank."""

    encoded = json.dumps(
        [int(base), str(purpose), *address],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def signed_moment_clipped_actor_loss(
    ratio: Tensor,
    advantage_positive: Tensor,
    advantage_negative: Tensor,
    posterior: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """Expected clipped PPO loss from cached positive/negative moments.

    All tensors are aligned on the actor-hand axis. ``advantage_positive`` and
    ``advantage_negative`` are respectively ``max(A, 0)`` and ``min(A, 0)``;
    posterior normalization is explicit so invalid zero-mass rows cannot pass
    silently.
    """

    if not (
        ratio.shape
        == advantage_positive.shape
        == advantage_negative.shape
        == posterior.shape
    ):
        raise ValueError("signed PPO tensors must have identical shapes")
    if ratio.ndim < 1 or not math.isfinite(clip_epsilon) or clip_epsilon < 0:
        raise ValueError("signed PPO inputs are invalid")
    weight = posterior.float()
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    value = (
        torch.minimum(ratio, clipped) * advantage_positive
        + torch.maximum(ratio, clipped) * advantage_negative
    )
    return -(weight * value).sum(-1)


def direct_clipped_actor_loss(
    ratio: Tensor,
    advantage: Tensor,
    posterior: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """Direct posterior average used to verify the signed-moment cache."""

    if not (ratio.shape == advantage.shape == posterior.shape):
        raise ValueError("direct PPO tensors must have identical shapes")
    weight = posterior.float()
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -(weight * torch.minimum(ratio * advantage, clipped * advantage)).sum(-1)


def gradient_variance_from_gram(gram: Tensor, weights: Tensor | None = None) -> Tensor:
    """Total sample variance of vectors represented by a Gram matrix.

    Integer bootstrap multiplicities may be supplied in ``weights``. Their sum
    is the bootstrap sample size, so repeated paired IDs are handled exactly.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] < 2:
        raise ValueError("gradient Gram matrix must be square with at least two rows")
    matrix = gram.double()
    if weights is None:
        weight = torch.ones(matrix.shape[0], dtype=torch.float64)
    else:
        if weights.shape != (matrix.shape[0],):
            raise ValueError("bootstrap weights do not match the Gram matrix")
        weight = weights.double()
    count = weight.sum()
    if float(count) <= 1 or bool((weight < 0).any()):
        raise ValueError("variance weights require at least two nonnegative draws")
    diagonal = matrix.diagonal()
    centered_sum = (weight * diagonal).sum() - weight @ matrix @ weight / count
    # Small negative values can arise from a long FP32 dot-product reduction.
    tolerance = (
        128.0
        * torch.finfo(torch.float64).eps
        * max(1.0, float((weight * diagonal.abs()).sum()))
    )
    if float(centered_sum) < -tolerance:
        raise ValueError("Gram matrix implies a negative gradient variance")
    return centered_sum.clamp_min(0.0) / (count - 1.0)


def percentile_interval(
    values: Sequence[float], level: float = 0.95
) -> tuple[float, float]:
    if not values or not 0 < level < 1:
        raise ValueError("percentile interval inputs are invalid")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    tail = (1.0 - level) / 2.0
    quantiles = torch.quantile(
        tensor, torch.tensor([tail, 1.0 - tail], dtype=tensor.dtype)
    )
    return float(quantiles[0]), float(quantiles[1])


def paired_bootstrap_indices(repetitions: int, draws: int, seed: int) -> Tensor:
    if repetitions < 2 or draws <= 0:
        raise ValueError("paired bootstrap dimensions are invalid")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randint(
        repetitions,
        (draws, repetitions),
        generator=generator,
        dtype=torch.int64,
    )


def protocol_copy() -> dict[str, Any]:
    """Return a JSON-safe mutable protocol copy."""

    return json.loads(json.dumps(DEFAULT_PROTOCOL))


__all__ = [
    "DEFAULT_PROTOCOL",
    "REPOSITORY",
    "SCHEMA",
    "canonical_sha256",
    "direct_clipped_actor_loss",
    "file_sha256",
    "gradient_variance_from_gram",
    "logical_seed",
    "paired_bootstrap_indices",
    "percentile_interval",
    "protocol_copy",
    "signed_moment_clipped_actor_loss",
]
