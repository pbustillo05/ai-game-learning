"""Runtime scientific configuration compiled from the canonical run schema."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from pbg.core.learner import LearnerSchedule, LossConfig
from pbg.core.optimization import OptimizerConfig


@dataclass(frozen=True)
class PokerGameConfig:
    players: int
    stack_bb: float
    rank_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.players, bool)
            or not isinstance(self.players, int)
            or not 2 <= self.players <= 6
        ):
            raise ValueError("players must be an integer in [2,6]")
        if (
            isinstance(self.stack_bb, bool)
            or not isinstance(self.stack_bb, (int, float))
            or not math.isfinite(self.stack_bb)
            or not 8 <= self.stack_bb <= 128
            or not math.isclose(self.stack_bb * 2, round(self.stack_bb * 2))
        ):
            raise ValueError("stack_bb must be a half-integer in [8,128]")
        if (
            isinstance(self.rank_count, bool)
            or not isinstance(self.rank_count, int)
            or not 4 <= self.rank_count <= 13
        ):
            raise ValueError("rank_count must be an integer in [4,13]")
        if 2 * self.players + 5 >= 4 * self.rank_count:
            raise ValueError("game must satisfy 2P+5<4R")


@dataclass(frozen=True)
class PrecisionConfig:
    name: str = "fp32"

    def __post_init__(self) -> None:
        if self.name not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16, or bf16")


@dataclass(frozen=True)
class PokerModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers_pre: int = 4
    n_layers_post: int = 4
    m_raise: int = 4
    s_min: float = 0.02
    max_players: int = 6
    class_embedding_dim: int = 32
    ffn_hidden: int | None = None
    folder_beta_rank: int = 16
    folder_line_dim: int = 8
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "n_heads",
            "n_layers_pre",
            "n_layers_post",
            "m_raise",
            "max_players",
            "class_embedding_dim",
            "folder_line_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.folder_beta_rank, bool)
            or not isinstance(self.folder_beta_rank, int)
            or self.folder_beta_rank < 0
        ):
            raise ValueError("folder_beta_rank must be a nonnegative integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 2 <= self.max_players <= 6:
            raise ValueError("max_players must be in [2,6]")
        if self.ffn_hidden is not None and (
            isinstance(self.ffn_hidden, bool)
            or not isinstance(self.ffn_hidden, int)
            or self.ffn_hidden <= 0
        ):
            raise ValueError("ffn_hidden must be positive when supplied")
        if (
            not math.isfinite(self.s_min)
            or self.s_min <= 0
            or not math.isfinite(self.epsilon)
            or self.epsilon <= 0
        ):
            field = "s_min" if not math.isfinite(self.s_min) or self.s_min <= 0 else "epsilon"
            raise ValueError(f"{field} must be finite and positive")


@dataclass(frozen=True)
class ReferenceConfig:
    interval_iterations: int | None = 100
    interval_effective_hands: float | None = None
    cache_outputs: bool = False

    def __post_init__(self) -> None:
        if (self.interval_iterations is None) == (
            self.interval_effective_hands is None
        ):
            raise ValueError("select exactly one reference refresh cadence")
        if self.interval_iterations is not None and self.interval_iterations <= 0:
            raise ValueError("interval_iterations must be positive")
        if self.interval_effective_hands is not None and (
            not math.isfinite(self.interval_effective_hands)
            or self.interval_effective_hands <= 0
        ):
            raise ValueError("interval_effective_hands must be positive")
        if type(self.cache_outputs) is not bool:
            raise TypeError("cache_outputs must be bool")


@dataclass(frozen=True)
class PokerScientificConfig:
    """Validated runtime values shared by every execution mode."""

    game: PokerGameConfig
    model: PokerModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    schedule: LearnerSchedule
    precision: PrecisionConfig
    reference: ReferenceConfig
    lambda_: float
    root_count: int
    root_count_minimum: int
    seed: int
    reproducibility: str = "strict"
    utility: str = "raw_bb"

    def __post_init__(self) -> None:
        if (
            isinstance(self.root_count, bool)
            or not isinstance(self.root_count, int)
            or isinstance(self.root_count_minimum, bool)
            or not isinstance(self.root_count_minimum, int)
            or not 1 <= self.root_count_minimum <= self.root_count
        ):
            raise ValueError("root_count must cover its positive minimum")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.reproducibility not in ("strict", "seeded", "throughput"):
            raise ValueError("reproducibility must be strict, seeded, or throughput")
        if self.utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("utility must be raw_bb or stack_normalized")
        if not math.isfinite(self.lambda_) or not 0 <= self.lambda_ <= 1:
            raise ValueError("lambda_ must be finite in [0,1]")
        if self.model.max_players < self.game.players:
            raise ValueError("model max_players does not cover game players")
        if self.model.folder_beta_rank > 4 * self.game.rank_count:
            raise ValueError("folder_beta_rank exceeds the compact deck")


@dataclass(frozen=True)
class SequentialModeConfig:
    """Runtime product for the single sequential pipeline."""

    mode: Mapping[str, str]
    common: PokerScientificConfig
    applicable: Mapping[str, object]


@dataclass(frozen=True)
class BranchedModeConfig:
    """Runtime product for the single branched pipeline."""

    mode: Mapping[str, str]
    common: PokerScientificConfig
    branching: Mapping[str, object]
    cost_safeguard: Mapping[str, object]
    applicable: Mapping[str, object]


__all__ = [
    "BranchedModeConfig",
    "PokerGameConfig",
    "PokerModelConfig",
    "PokerScientificConfig",
    "PrecisionConfig",
    "ReferenceConfig",
    "SequentialModeConfig",
]
