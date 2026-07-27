"""Closed Phase 5 configuration with dotted-path diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping, TypeVar

from pbg.queued.learner import LearnerSchedule, LossConfig
from pbg.queued.optimization import OptimizerConfig


T = TypeVar("T")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected an object")
    return value


def _closed(
    cls: type[T], value: object, path: str, aliases: Mapping[str, str] | None = None
) -> T:
    source = dict(_mapping(value, path))
    aliases = {} if aliases is None else aliases
    translated: dict[str, object] = {}
    allowed = {field.name for field in fields(cls)}
    for key, item in source.items():
        name = aliases.get(key, key)
        if name not in allowed:
            raise ValueError(f"{path}.{key}: unknown configuration field")
        if name in translated:
            raise ValueError(f"{path}.{key}: duplicate configuration field")
        translated[name] = item
    try:
        return cls(**translated)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        field = next(
            (name for name in allowed if name in message), "configuration"
        )
        raise ValueError(f"{path}.{field}: {message}") from exc


@dataclass(frozen=True)
class Phase5GameConfig:
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
class Phase5Precision:
    name: str = "fp32"

    def __post_init__(self) -> None:
        if self.name not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16, or bf16")


@dataclass(frozen=True)
class Phase5ModelConfig:
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
        integers = (
            self.d_model,
            self.n_heads,
            self.n_layers_pre,
            self.n_layers_post,
            self.m_raise,
            self.max_players,
            self.class_embedding_dim,
            self.folder_line_dim,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers
        ):
            raise ValueError("model dimensions must be positive integers")
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
            raise ValueError("model scale and epsilon must be positive")

    @property
    def action_head_width(self) -> int:
        return 3 + self.m_raise


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
class Phase5Config:
    """Scientific Phase 5 values; capacities remain a separate manifest."""

    game: Phase5GameConfig
    model: Phase5ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    schedule: LearnerSchedule
    precision: Phase5Precision
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
            raise ValueError(
                "reproducibility must be strict, seeded, or throughput"
            )
        if self.utility not in ("raw_bb", "stack_normalized"):
            raise ValueError("utility must be raw_bb or stack_normalized")
        if not math.isfinite(self.lambda_) or not 0 <= self.lambda_ <= 1:
            raise ValueError("lambda_ must be finite in [0,1]")
        if self.model.max_players < self.game.players:
            raise ValueError("model max_players does not cover game players")
        if self.model.folder_beta_rank > 4 * self.game.rank_count:
            raise ValueError("folder_beta_rank exceeds the compact deck")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Phase5Config":
        source = dict(_mapping(value, "root"))
        allowed = {
            "mode",
            "game",
            "loss",
            "model",
            "optimizer",
            "schedule",
            "precision",
            "reference",
            "lambda",
            "root_count",
            "root_count_minimum",
            "seed",
            "reproducibility",
            "utility",
        }
        unknown = sorted(set(source).difference(allowed))
        if unknown:
            raise ValueError(f"root.{unknown[0]}: unknown configuration field")
        mode = _mapping(source.get("mode"), "root.mode")
        expected_mode = {
            "training_regime": "self_play",
            "rollout_mode": "sequential",
            "private_estimator": "sampled",
        }
        if dict(mode) != expected_mode:
            unexpected = next(
                (
                    key
                    for key in set(mode).union(expected_mode)
                    if mode.get(key) != expected_mode.get(key)
                ),
                "mode",
            )
            raise ValueError(
                f"root.mode.{unexpected}: Phase 5 supports only "
                "self_play/sequential/sampled"
            )
        required = {
            "game",
            "loss",
            "model",
            "optimizer",
            "schedule",
            "precision",
            "reference",
            "lambda",
            "root_count",
            "root_count_minimum",
            "seed",
        }
        missing = sorted(required.difference(source))
        if missing:
            raise ValueError(f"root.{missing[0]}: missing required field")
        return cls(
            game=_closed(Phase5GameConfig, source["game"], "root.game"),
            model=_closed(Phase5ModelConfig, source["model"], "root.model"),
            loss=_closed(LossConfig, source["loss"], "root.loss"),
            optimizer=_closed(
                OptimizerConfig, source["optimizer"], "root.optimizer"
            ),
            schedule=_closed(
                LearnerSchedule, source["schedule"], "root.schedule"
            ),
            precision=_closed(
                Phase5Precision, source["precision"], "root.precision"
            ),
            reference=_closed(
                ReferenceConfig, source["reference"], "root.reference"
            ),
            lambda_=source["lambda"],
            root_count=source["root_count"],
            root_count_minimum=source["root_count_minimum"],
            seed=source["seed"],
            reproducibility=source.get("reproducibility", "strict"),
            utility=source.get("utility", "raw_bb"),
        )

    def scientific_mapping(self) -> dict[str, object]:
        def values(item: object) -> dict[str, object]:
            return {
                field.name: getattr(item, field.name)
                for field in fields(item)
            }

        return {
            "mode": {
                "training_regime": "self_play",
                "rollout_mode": "sequential",
                "private_estimator": "sampled",
            },
            "game": values(self.game),
            "model": values(self.model),
            "loss": values(self.loss),
            "optimizer": values(self.optimizer),
            "schedule": values(self.schedule),
            "precision": values(self.precision),
            "reference": values(self.reference),
            "lambda": self.lambda_,
            "root_count": self.root_count,
            "root_count_minimum": self.root_count_minimum,
            "seed": self.seed,
            "reproducibility": self.reproducibility,
            "utility": self.utility,
        }


__all__ = [
    "Phase5Config",
    "Phase5GameConfig",
    "Phase5ModelConfig",
    "Phase5Precision",
    "ReferenceConfig",
]
