"""Game-neutral data transfer objects for leaf completion.

The core package deliberately knows nothing about poker.  In particular,
``ReachView.payload`` and the diagnostic fields on
``BatchedCompletionStats`` are opaque values ferried between a game adapter
and the rollout/value code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

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

    The first five fields are the game-neutral learning interface.  Remaining
    tensor fields are optional evaluator diagnostics; keeping them here avoids
    throwing information away at a game boundary while core algorithms remain
    free to ignore them.
    """

    p_hat: Tensor  # [B,P_max,X_max]
    v_hat: Tensor  # [B,P_max,X_max]
    vvec_hat: Tensor  # [B,P_max]
    u_hat: Tensor  # [B]
    n_eff: Tensor  # [B]
    equity_share: Tensor | None = None
    output_valid: Tensor | None = None
    sample_count: Tensor | None = None
    estimator_stddev: Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def row(self, index: int) -> CompletionStats:
        """Return the five learning statistics for one batch row."""

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
