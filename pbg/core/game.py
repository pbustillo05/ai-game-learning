"""Frozen generic game protocol and public-tree owner constants."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor

CHANCE = -1
TERMINAL = -2
PRETERM = -3


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

