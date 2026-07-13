"""Generic game and device-resident rollout boundary definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from pbg.core.completion import MCCompletion

CHANCE = -1
TERMINAL = -2
PRETERM = -3

StateBatch = Tensor


@dataclass
class EventBatch:
    """Public events for ``B`` states.

    ``public_id`` is int32 ``[B]``.  Every opaque payload tensor has leading
    dimension ``B`` and lives on the same device.
    """

    public_id: Tensor
    payload: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.public_id.ndim != 1 or self.public_id.dtype != torch.int32:
            raise ValueError("public_id must be int32 [B]")
        for name, value in self.payload.items():
            if value.ndim < 1 or value.shape[0] != self.public_id.shape[0]:
                raise ValueError(f"event payload {name!r} must have leading dimension B")
            if value.device != self.public_id.device:
                raise ValueError(f"event payload {name!r} must share public_id.device")


@runtime_checkable
class RolloutBackend(Protocol):
    """Device-neutral batched public rollout interface."""

    device: torch.device
    n_players: int
    x_sizes: tuple[int, ...]
    max_rollout_steps: int

    def root_states(self, n: int) -> StateBatch:
        """Return device-local opaque int32 state handles ``[B]``."""
        ...

    def classify(self, states: StateBatch) -> Tensor:
        """Return int32 owner/leaf codes ``[B]``."""
        ...

    def legal_events(self, states: StateBatch) -> tuple[Tensor, Tensor]:
        """Return int32 ids and bool masks ``[B,E_max]``."""
        ...

    def policy_forward(
        self, frozen_model: nn.Module, states: StateBatch, reach: Tensor
    ) -> Any:
        """Return game-owned policy output for states/reach ``[B]``/``[B,P,X]``."""
        ...

    def plan_policy_batches(
        self, model: nn.Module, states: StateBatch, phase: str
    ) -> list[Tensor]:
        """Partition sampled rows into bounded physical policy batches."""
        ...

    def sample_profiles(
        self, states: StateBatch, reach: Tensor, gen: torch.Generator
    ) -> Tensor:
        """Draw one posterior private profile as int32 ``[B,P]``."""
        ...

    def sample_events(
        self,
        states: StateBatch,
        profiles: Tensor,
        policy_out: Any | None,
        adapter: Any,
        gen: torch.Generator,
    ) -> EventBatch:
        """Draw one public event per input row as an :class:`EventBatch`."""
        ...

    def step(self, states: StateBatch, events: EventBatch) -> StateBatch:
        """Apply ``B`` events and return successor handles ``[B]``."""
        ...

    def reach_update(
        self,
        reach: Tensor,
        states: StateBatch,
        events: EventBatch,
        policy_out: Any | None,
    ) -> Tensor:
        """Return updated fp32 log reach rows ``[B,P,X]``."""
        ...

    def beta_fn(self, states: StateBatch) -> Tensor:
        """Return fixed branching intensities as fp32 ``[B]``."""
        ...

    def sampled_terminal_reward(
        self, states: StateBatch, profiles: Tensor, gen: torch.Generator
    ) -> Tensor:
        """Settle sampled terminal/pre-terminal rows as fp32 ``[B,P]``."""
        ...


@dataclass
class GameBundle:
    """Typed collection of model and game-owned rollout/completion adapters."""

    model: nn.Module
    rollout_backend: RolloutBackend
    policy_adapter: Any
    completion_backend: MCCompletion | None


@runtime_checkable
class PBGGame(Protocol):
    """Interface implemented by every public-action Bayesian game."""

    n_players: int
    x_sizes: tuple[int, ...]

    def sample_mu(self, g: int, gen: Any) -> Tensor:
        """Sample private profiles as int32 ``[G, P]``."""
        ...

    def root_states(self, g: int) -> Tensor:
        """Return int32 root state references ``[G]``."""
        ...

    def owner_of(self, state_refs: Tensor) -> Tensor:
        """Return int32 owner codes ``[B]``."""
        ...

    def legal_events(self, state_refs: Tensor) -> tuple[Tensor, Tensor]:
        """Return event ids and validity masks, each ``[B, E_max]``."""
        ...

    def chance_probs(self, state_refs: Tensor, xprofile: Tensor) -> Tensor:
        """Return chance probabilities ``[B, E_max]`` for profiles ``[B, P]``."""
        ...

    def step(self, state_refs: Tensor, event_ids: Tensor) -> Tensor:
        """Apply events and return int32 successor state references ``[B]``."""
        ...

    def is_terminal(self, state_refs: Tensor) -> Tensor:
        """Return terminal flags ``[B]``."""
        ...

    def is_pre_terminal(self, state_refs: Tensor) -> Tensor:
        """Return pre-terminal flags ``[B]``."""
        ...

    def reward(self, xprofile: Tensor, terminal_state_refs: Tensor) -> Tensor:
        """Return terminal rewards as float32 ``[B, P]``."""
        ...

    def reach_update(
        self,
        log_r_rows: Tensor,
        state_refs: Tensor,
        event_ids: Tensor,
        policy_rows: Tensor,
    ) -> Tensor:
        """Update and return log-reach rows with shape ``[B, P, X_max]``."""
        ...

    def sample_profiles(
        self,
        state_refs: Tensor,
        reach_rows: Tensor,
        n: int,
        gen: Any,
    ) -> Tensor:
        """Draw joint posterior profiles as int32 ``[B, n, P]``."""
        ...

    def encode_public(self, state_refs: Tensor) -> Any:
        """Encode public states ``[B]`` in a game-owned model input."""
        ...
