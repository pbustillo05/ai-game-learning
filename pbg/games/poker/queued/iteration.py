"""Compact poker binding for the generic iteration lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pbg.core.capacity import GeneratedOwners
from pbg.core.iteration import IterationAttempt


@dataclass(frozen=True)
class PokerIterationHooks:
    """Construct poker phase work without owning the phase state machine."""

    attempt_factory: Callable[[GeneratedOwners], IterationAttempt]

    def build_attempt(self, owners: GeneratedOwners) -> IterationAttempt:
        return self.attempt_factory(owners)


__all__ = ["PokerIterationHooks"]
