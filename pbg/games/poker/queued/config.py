"""Strict NLHE scientific configuration with dotted-path diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TypeVar

from pbg.core.learner import LearnerSchedule, LossConfig
from pbg.core.optimization import OptimizerConfig

T = TypeVar("T")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected an object")  # noqa: TRY004
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _closed(cls: type[T], value: object, path: str) -> T:
    source = dict(_mapping(value, path))
    translated: dict[str, object] = {}
    field_names = tuple(field.name for field in fields(cls))
    allowed = set(field_names)
    for key, item in source.items():
        if key not in allowed:
            raise ValueError(f"{path}.{key}: unknown configuration field")
        if key in translated:
            raise ValueError(f"{path}.{key}: duplicate configuration field")
        translated[key] = item
    try:
        return cls(**translated)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        field = next(
            (name for name in field_names if name in message),
            "configuration",
        )
        raise ValueError(f"{path}.{field}: {message}") from exc


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
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
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
            field = (
                "s_min"
                if not math.isfinite(self.s_min) or self.s_min <= 0
                else "epsilon"
            )
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
    """Scientific Phase 5 values; capacities remain a separate manifest."""

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

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PokerScientificConfig:
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
            game=_closed(PokerGameConfig, source["game"], "root.game"),
            model=_closed(PokerModelConfig, source["model"], "root.model"),
            loss=_closed(LossConfig, source["loss"], "root.loss"),
            optimizer=_closed(OptimizerConfig, source["optimizer"], "root.optimizer"),
            schedule=_closed(LearnerSchedule, source["schedule"], "root.schedule"),
            precision=_closed(PrecisionConfig, source["precision"], "root.precision"),
            reference=_closed(ReferenceConfig, source["reference"], "root.reference"),
            lambda_=source["lambda"],
            root_count=source["root_count"],
            root_count_minimum=source["root_count_minimum"],
            seed=source["seed"],
            reproducibility=source.get("reproducibility", "strict"),
            utility=source.get("utility", "raw_bb"),
        )

    def scientific_mapping(self) -> dict[str, object]:
        def values(item: object) -> dict[str, object]:
            return {field.name: getattr(item, field.name) for field in fields(item)}

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


@dataclass(frozen=True)
class SequentialModeConfig:
    """Closed Phase 6 scientific configuration for all sequential rows."""

    mode: Mapping[str, str]
    common: PokerScientificConfig
    applicable: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SequentialModeConfig:
        source = dict(_mapping(value, "root"))
        mode = dict(_mapping(source.get("mode"), "root.mode"))
        allowed_mode = {
            "training_regime": ("self_play", "best_response"),
            "rollout_mode": ("sequential",),
            "private_estimator": ("sampled", "rao_blackwell"),
        }
        if set(mode) != set(allowed_mode):
            raise ValueError("root.mode: sequential mode axes are incomplete")
        for name, choices in allowed_mode.items():
            if mode[name] not in choices:
                raise ValueError(f"root.mode.{name}: invalid sequential axis")
        response = mode["training_regime"] == "best_response"
        rb = mode["private_estimator"] == "rao_blackwell"
        extensions = {
            "completion",
            "frozen_profile",
            "response_initialization",
            "probe",
        }
        base_source = {
            key: item for key, item in source.items() if key not in extensions
        }
        base_source["mode"] = {
            "training_regime": "self_play",
            "rollout_mode": "sequential",
            "private_estimator": "sampled",
        }
        common = PokerScientificConfig.from_mapping(base_source)
        applicable: dict[str, object] = {}
        if rb:
            completion = dict(_mapping(source.get("completion"), "root.completion"))
            if set(completion) != {"n_samples"}:
                raise ValueError("root.completion: RB requires exactly n_samples")
            if (
                isinstance(completion["n_samples"], bool)
                or not isinstance(completion["n_samples"], int)
                or completion["n_samples"] <= 0
            ):
                raise ValueError("root.completion.n_samples: must be positive")
            applicable["completion"] = completion
        elif "completion" in source:
            raise ValueError("root.completion: inapplicable to sampled mode")
        if response:
            frozen = dict(_mapping(source.get("frozen_profile"), "root.frozen_profile"))
            if set(frozen) != {"sha256"} or not _is_sha256(frozen["sha256"]):
                raise ValueError("root.frozen_profile.sha256: content hash is required")
            initialization = source.get("response_initialization")
            if initialization not in ("random", "target"):
                raise ValueError(
                    "root.response_initialization: expected random or target"
                )
            applicable["frozen_profile"] = frozen
            applicable["response_initialization"] = initialization
            if "probe" in source:
                if not rb:
                    raise ValueError(
                        "root.probe: shared-response probe requires sequential RB"
                    )
                probe = dict(_mapping(source["probe"], "root.probe"))
                if set(probe) != {
                    "target_sha256",
                    "initial_model_sha256",
                } or (
                    probe["target_sha256"] != frozen["sha256"]
                    or not _is_sha256(probe["initial_model_sha256"])
                ):
                    raise ValueError(
                        "root.probe: target and initial model identities are required"
                    )
                applicable["probe"] = probe
        elif any(
            name in source
            for name in ("frozen_profile", "response_initialization", "probe")
        ):
            raise ValueError(
                "root.frozen_profile: response fields require best_response"
            )
        return cls(
            MappingProxyType(mode),
            common,
            MappingProxyType(applicable),
        )

    def scientific_mapping(self) -> dict[str, object]:
        result = self.common.scientific_mapping()
        result["mode"] = dict(self.mode)
        result.update(
            {
                name: dict(value) if isinstance(value, Mapping) else value
                for name, value in self.applicable.items()
            }
        )
        return result


@dataclass(frozen=True)
class BranchedModeConfig:
    """Closed scientific configuration for every branched mode."""

    mode: Mapping[str, str]
    common: PokerScientificConfig
    branching: Mapping[str, object]
    cost_safeguard: Mapping[str, object]
    applicable: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> BranchedModeConfig:
        source = dict(_mapping(value, "root"))
        mode = dict(_mapping(source.get("mode"), "root.mode"))
        if set(mode) != {
            "training_regime",
            "rollout_mode",
            "private_estimator",
        }:
            raise ValueError("root.mode: branched mode axes are incomplete")
        if mode["training_regime"] not in ("self_play", "best_response"):
            raise ValueError("root.mode.training_regime: invalid branched axis")
        if mode["rollout_mode"] not in ("branched.fixed", "branched.KKT"):
            raise ValueError("root.mode.rollout_mode: invalid branched axis")
        if mode["private_estimator"] not in ("sampled", "rao_blackwell"):
            raise ValueError(
                "root.mode.private_estimator: invalid branched axis"
            )
        rb = mode["private_estimator"] == "rao_blackwell"
        extensions = {
            "branching",
            "cost_safeguard",
            "completion",
            "frozen_profile",
            "response_initialization",
        }
        base_source = {
            key: item for key, item in source.items() if key not in extensions
        }
        base_source["mode"] = {
            "training_regime": "self_play",
            "rollout_mode": "sequential",
            "private_estimator": "sampled",
        }
        common = PokerScientificConfig.from_mapping(base_source)
        branching = dict(_mapping(source.get("branching"), "root.branching"))
        shared = {
            "fixed_beta",
            "beta_max",
            "minimum_tree_size",
            "minimum_scale",
            "backoff_factor",
            "safe_tree_size",
        }
        adaptive = {
            "warmup_iterations",
            "target_tree_size",
            "gamma_kappa",
            "eta_kappa",
            "kappa_min",
            "kappa_max",
            "log_kappa",
        }
        expected = (
            shared
            if mode["rollout_mode"] == "branched.fixed"
            else shared | adaptive
        )
        if set(branching) != expected:
            unknown = sorted(set(branching).difference(expected))
            missing = sorted(expected.difference(branching))
            name = unknown[0] if unknown else missing[0]
            detail = "unknown" if unknown else "missing"
            raise ValueError(f"root.branching.{name}: {detail} field")
        fixed_beta = dict(
            _mapping(branching["fixed_beta"], "root.branching.fixed_beta")
        )
        valid_fixed_beta = (
            {"player", "chance"},
            {"player_pre", "player_post", "chance"},
        )
        if set(fixed_beta) not in valid_fixed_beta or (
            mode["rollout_mode"] == "branched.KKT"
            and set(fixed_beta) != {"player", "chance"}
        ):
            raise ValueError(
                "root.branching.fixed_beta: fixed mode requires player/chance "
                "or player_pre/player_post/chance; KKT requires player/chance"
            )
        beta_max_raw = branching["beta_max"]
        if isinstance(beta_max_raw, Mapping):
            beta_max = dict(beta_max_raw)
            valid_beta_max = (
                mode["rollout_mode"] == "branched.fixed"
                and set(fixed_beta)
                == {"player_pre", "player_post", "chance"}
                and set(beta_max)
                == {"player_pre", "player_post", "chance"}
                and all(
                    not isinstance(item, bool)
                    and isinstance(item, int)
                    and item >= 1
                    for item in beta_max.values()
                )
            )
        else:
            beta_max = beta_max_raw
            valid_beta_max = (
                not isinstance(beta_max, bool)
                and isinstance(beta_max, int)
                and beta_max >= 1
            )
        numeric = {
            name: item
            for name, item in branching.items()
            if name not in ("fixed_beta", "beta_max", "minimum_tree_size")
        }
        numeric.update(fixed_beta)
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in numeric.values()
        ):
            raise ValueError("root.branching: numeric controller values must be finite")
        if (
            isinstance(branching["minimum_tree_size"], bool)
            or not isinstance(branching["minimum_tree_size"], int)
            or branching["minimum_tree_size"] <= 0
            or not valid_beta_max
            or any(
                not 1
                <= fixed_beta[name]
                <= (
                    beta_max[name]
                    if isinstance(beta_max, dict)
                    else beta_max
                )
                for name in fixed_beta
            )
            or (
                "minimum_scale" in branching
                and not 0 < branching["minimum_scale"] <= 1
            )
            or not 0 < branching["backoff_factor"] < 1
            or branching["safe_tree_size"] < 1
        ):
            raise ValueError("root.branching: controller bounds are invalid")
        if mode["rollout_mode"] == "branched.KKT" and (
            isinstance(branching["warmup_iterations"], bool)
            or not isinstance(branching["warmup_iterations"], int)
            or branching["warmup_iterations"] < 0
            or branching["target_tree_size"] <= 0
            or not 0 < branching["gamma_kappa"] <= 1
            or not 0 < branching["eta_kappa"] <= 1
            or not 0 < branching["kappa_min"] <= branching["kappa_max"]
            or not math.log(branching["kappa_min"])
            <= branching["log_kappa"]
            <= math.log(branching["kappa_max"])
        ):
            raise ValueError("root.branching: KKT controller bounds are invalid")
        safeguard = dict(_mapping(source.get("cost_safeguard"), "root.cost_safeguard"))
        if set(safeguard) != {
            "gamma_deviance_max",
            "p95_underprediction_max",
            "consecutive_iterations",
        }:
            raise ValueError("root.cost_safeguard: calibration fields are incomplete")
        safeguard_numeric = (
            safeguard["gamma_deviance_max"],
            safeguard["p95_underprediction_max"],
        )
        if (
            not isinstance(safeguard["consecutive_iterations"], int)
            or isinstance(safeguard["consecutive_iterations"], bool)
            or safeguard["consecutive_iterations"] <= 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in safeguard_numeric
            )
            or safeguard["gamma_deviance_max"] < 0
            or safeguard["p95_underprediction_max"] < 1
        ):
            raise ValueError("root.cost_safeguard: calibration bounds are invalid")
        applicable: dict[str, object] = {}
        if rb:
            completion = dict(
                _mapping(source.get("completion"), "root.completion")
            )
            if set(completion) != {"n_samples"}:
                raise ValueError("root.completion: RB requires exactly n_samples")
            if (
                isinstance(completion["n_samples"], bool)
                or not isinstance(completion["n_samples"], int)
                or completion["n_samples"] <= 0
            ):
                raise ValueError(
                    "root.completion.n_samples: must be positive"
                )
            applicable["completion"] = completion
        elif "completion" in source:
            raise ValueError(
                "root.completion: inapplicable to sampled mode"
            )
        response = mode["training_regime"] == "best_response"
        if response:
            frozen = dict(_mapping(source.get("frozen_profile"), "root.frozen_profile"))
            if set(frozen) != {"sha256"} or not _is_sha256(frozen["sha256"]):
                raise ValueError("root.frozen_profile.sha256: content hash is required")
            initialization = source.get("response_initialization")
            if initialization not in ("random", "target"):
                raise ValueError(
                    "root.response_initialization: expected random or target"
                )
            applicable["frozen_profile"] = frozen
            applicable["response_initialization"] = initialization
        elif any(
            name in source for name in ("frozen_profile", "response_initialization")
        ):
            raise ValueError(
                "root.frozen_profile: response fields require best_response"
            )
        return cls(
            MappingProxyType(mode),
            common,
            MappingProxyType(
                {
                    **branching,
                    "fixed_beta": MappingProxyType(fixed_beta),
                    "beta_max": (
                        MappingProxyType(beta_max)
                        if isinstance(beta_max, dict)
                        else beta_max
                    ),
                }
            ),
            MappingProxyType(safeguard),
            MappingProxyType(applicable),
        )

    def scientific_mapping(self) -> dict[str, object]:
        result = self.common.scientific_mapping()
        result["mode"] = dict(self.mode)
        result["branching"] = {
            name: dict(value) if isinstance(value, Mapping) else value
            for name, value in self.branching.items()
        }
        result["cost_safeguard"] = dict(self.cost_safeguard)
        result.update(
            {
                name: dict(value) if isinstance(value, Mapping) else value
                for name, value in self.applicable.items()
            }
        )
        return result


__all__ = [
    "BranchedModeConfig",
    "PokerGameConfig",
    "PokerModelConfig",
    "PokerScientificConfig",
    "PrecisionConfig",
    "ReferenceConfig",
    "SequentialModeConfig",
]
