"""Generic policy-output and action-adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor


@dataclass
class PolicyOutput:
    """Game-owned policy payload and actor-type values ``[B,X]``."""

    raw: Any
    v: Tensor


class PolicyAdapter(Protocol):
    """Normalize game-specific policies over complete legal event rows."""

    def log_prob(
        self,
        out: PolicyOutput,
        selected_event_ids: Tensor,
        legal_event_ids: Tensor,
        legal_event_mask: Tensor,
    ) -> Tensor:
        """Return selected log probabilities as fp32 ``[B,E_selected,X]``."""

        ...

    def entropy(
        self,
        out: PolicyOutput,
        legal_event_ids: Tensor,
        legal_event_mask: Tensor,
    ) -> Tensor:
        """Return legal-row entropy as fp32 ``[B,X]``."""

        ...
