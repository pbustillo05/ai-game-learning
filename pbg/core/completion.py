"""Game-neutral data transfer objects for leaf completion.

The core package deliberately knows nothing about poker.  In particular,
``ReachView.payload`` and the diagnostic fields on
``BatchedCompletionStats`` are opaque values ferried between a game adapter
and the rollout/value code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass
class CompletionStats:
    """Completion statistics for one leaf (all floating tensors are fp32)."""

    p_hat: Tensor  # [P_max,X_max]
    v_hat: Tensor  # [P_max,X_max]
    vvec_hat: Tensor  # [P_max]
    u_hat: Tensor  # []
    n_eff: Tensor  # []


@dataclass
class BatchedCompletionStats:
    """Completion statistics for ``B`` leaves.

    ``p_hat`` and ``v_hat`` are the learning interface.  The remaining tensors
    are optional completion diagnostics and must be collapsed with
    :meth:`for_learning` before target construction.
    """

    p_hat: Tensor  # [B,P_max,X_max]
    v_hat: Tensor  # [B,P_max,X_max]
    vvec_hat: Tensor | None = None  # [B,P_max]
    u_hat: Tensor | None = None  # [B]
    n_eff: Tensor | None = None  # [B]
    equity_share: Tensor | None = None
    output_valid: Tensor | None = None
    sample_count: Tensor | None = None
    estimator_stddev: Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def for_learning(self) -> "BatchedCompletionStats":
        """Validate estimator quality and retain only tensors used by targets.

        Invalid posterior mass is rejected.  Scalar quality summaries are
        copied into metadata so training can log them without retaining the
        evaluator's ``[B,P,X]`` validity/error tensors or ``[B,P,X,K]`` equity
        tensor.
        """

        if self.p_hat.ndim != 3 or self.v_hat.shape != self.p_hat.shape:
            raise ValueError("completion p_hat and v_hat must share shape [B,P,X]")
        if not self.p_hat.is_floating_point() or not self.v_hat.is_floating_point():
            raise TypeError("completion p_hat and v_hat must be floating point")
        if self.p_hat.device != self.v_hat.device:
            raise ValueError("completion p_hat and v_hat must share a device")
        if not bool(self.p_hat.isfinite().all()) or bool((self.p_hat < 0).any()):
            raise ValueError("completion posterior mass must be finite and nonnegative")
        if not bool(self.v_hat.isfinite().all()):
            raise ValueError("completion conditional values must be finite")

        batch = self.p_hat.shape[0]
        quality: dict[str, float] = {}
        if self.n_eff is not None:
            if self.n_eff.shape != (batch,) or self.n_eff.device != self.p_hat.device:
                raise ValueError("completion n_eff must be device-local [B]")
            if not bool(self.n_eff.isfinite().all()) or bool((self.n_eff <= 0).any()):
                raise ValueError("completion n_eff must be finite and positive")
            quality["completion_min_n_eff"] = float(self.n_eff.min())
            quality["completion_mean_n_eff"] = float(self.n_eff.mean())
        if self.sample_count is not None:
            if self.sample_count.shape != (batch,) or self.sample_count.device != self.p_hat.device:
                raise ValueError("completion sample_count must be device-local [B]")
            if bool((self.sample_count <= 0).any()):
                raise ValueError("completion sample_count must be positive")
            if self.n_eff is not None and bool(
                (self.n_eff > self.sample_count.to(self.n_eff.dtype) + 1e-4).any()
            ):
                raise ValueError("completion n_eff cannot exceed sample_count")
            quality["completion_min_sample_count"] = float(self.sample_count.min())

        if self.output_valid is not None:
            if self.output_valid.shape != self.p_hat.shape:
                raise ValueError("completion output_valid must match p_hat")
            if (
                self.output_valid.dtype != torch.bool
                or self.output_valid.device != self.p_hat.device
            ):
                raise ValueError("completion output_valid must be device-local bool")
            invalid_mass = self.p_hat.masked_select(~self.output_valid).sum()
            quality["completion_invalid_posterior_mass"] = float(invalid_mass)
            if float(invalid_mass) > 1e-6:
                raise ValueError("completion assigned posterior mass to invalid outputs")

        if self.estimator_stddev is not None:
            if self.estimator_stddev.shape != self.p_hat.shape:
                raise ValueError("completion estimator_stddev must match p_hat")
            observed = self.p_hat > 0
            if self.output_valid is not None:
                observed &= self.output_valid
            selected = self.estimator_stddev[observed]
            if selected.numel():
                if bool(torch.isinf(selected).any()) or bool((selected < 0).any()):
                    raise ValueError(
                        "completion estimator_stddev must be nonnegative or NaN "
                        "on posterior support"
                    )
                finite = selected.isfinite()
                quality["completion_stddev_coverage"] = float(finite.float().mean())
                if bool(finite.any()):
                    quality["completion_max_estimator_stddev"] = float(selected[finite].max())
                    quality["completion_mean_estimator_stddev"] = float(selected[finite].mean())

        metadata = dict(self.metadata)
        metadata.update(quality)
        return BatchedCompletionStats(
            p_hat=self.p_hat,
            v_hat=self.v_hat,
            metadata=metadata,
        )

    def row(self, index: int) -> CompletionStats:
        """Return the five learning statistics for one batch row."""

        if self.vvec_hat is None or self.u_hat is None or self.n_eff is None:
            raise RuntimeError("learning-only completion rows omit diagnostic moments")

        return CompletionStats(
            p_hat=self.p_hat[index],
            v_hat=self.v_hat[index],
            vvec_hat=self.vvec_hat[index],
            u_hat=self.u_hat[index],
            n_eff=self.n_eff[index],
        )


@dataclass
class ReachView:
    """Selected log-reach rows plus an optional game-owned payload.

    ``log_r`` is float32 ``[B,P_max,X_max]``.  Core never interprets
    ``payload``; it exists for adapters that need additional public bookkeeping.
    """

    log_r: Tensor
    payload: Any = None

    @property
    def rows(self) -> Tensor:
        """Alias used by generic rollout plumbing."""

        return self.log_r


@runtime_checkable
class MCCompletion(Protocol):
    """Protocol implemented by game-specific completion backends."""

    def complete(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: ReachView | Tensor | Any,
        n_samples: int,
        gen: Any,
    ) -> BatchedCompletionStats:
        """Complete selected leaves and return one statistics row per node."""

        ...
