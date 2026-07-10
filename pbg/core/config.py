"""Strict dataclass configuration loading and explicit random generators."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np
import torch
import yaml


class ConfigError(ValueError):
    """Raised when a configuration does not match the frozen schema."""


@dataclass
class GameConfig:
    """Select a game and retain its game-owned configuration mapping."""

    name: str
    synthetic: dict[str, Any] | None = None
    poker: dict[str, Any] | None = None


@dataclass
class RolloutConfig:
    """Configure stored-tree sampling and branching intensities."""

    n_trees: int
    beta_mode: str
    beta_fix: dict[str, float]
    beta_avg: dict[str, float]
    beta_max: dict[str, float]
    warmup_iters: int
    node_budget: int
    reach_storage: str


@dataclass
class CompletionConfig:
    """Configure Monte Carlo completion at pre-terminal nodes."""

    n_samples: int
    backend: str


@dataclass
class LossConfig:
    """Configure the policy, value, entropy, KL, and moment losses."""

    name: str
    lam: float
    clip_eps: float
    clip_eps_v: float
    c_v: float
    c_h: float
    c_kl: float
    c_u: float
    c_vvec: float
    ref_period: int | None
    entropy_schedule: str
    use_moment_heads: bool = False
    use_suit_consistency: bool = False


@dataclass
class TrainConfig:
    """Configure optimization, reproducibility, and checkpoint output."""

    iters: int
    gd_steps_per_iter: int
    minibatch_size: int
    lr: float
    adam_betas: tuple[float, float]
    grad_clip: float
    amp: bool
    seed: int
    checkpoint_every: int
    out_dir: str


@dataclass
class RootConfig:
    """Top-level configuration shared by all game implementations."""

    game: GameConfig
    rollout: RolloutConfig
    completion: CompletionConfig
    model: dict[str, Any]
    loss: LossConfig
    train: TrainConfig
    debug_asserts: bool = False


_ConfigT = TypeVar("_ConfigT")


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))


def _convert_value(value: Any, annotation: Any, path: str) -> Any:
    """Convert a YAML value to an annotated dataclass field value."""

    origin = get_origin(annotation)
    args = get_args(annotation)
    if is_dataclass(annotation):
        return _parse_dataclass(annotation, value, path)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        candidates = tuple(arg for arg in args if arg is not type(None))
        errors: list[str] = []
        for candidate in candidates:
            try:
                return _convert_value(value, candidate, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(
            f"{path}: does not match {_type_name(annotation)} ({'; '.join(errors)})"
        )
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        key_type, value_type = args or (Any, Any)
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            converted_key = (
                key if key_type is Any else _convert_value(key, key_type, f"{path}.<key>")
            )
            converted[converted_key] = (
                item if value_type is Any else _convert_value(item, value_type, f"{path}.{key}")
            )
        return converted
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _convert_value(item, args[0], f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(
            _convert_value(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, args, strict=True))
        )
    if annotation is Any:
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"{path}: expected bool")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"{path}: expected int")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise ConfigError(f"{path}: expected float")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str")
        return value
    return value


def _parse_dataclass(cls: type[_ConfigT], value: Any, path: str) -> _ConfigT:
    """Recursively instantiate one dataclass while rejecting unknown keys."""

    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping")
    field_map = {field.name: field for field in fields(cls)}
    for key in value:
        if key not in field_map:
            raise ConfigError(f"{path}.{key}: unknown configuration key")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, field in field_map.items():
        child_path = f"{path}.{name}"
        if name in value:
            kwargs[name] = _convert_value(value[name], hints[name], child_path)
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ConfigError(f"{child_path}: missing required configuration key")
    return cls(**kwargs)


def _validate_config(cfg: RootConfig) -> None:
    """Validate frozen enumerations and the external evaluator boundary."""

    if cfg.game.name not in {"synthetic", "poker"}:
        raise ConfigError("root.game.name: expected 'synthetic' or 'poker'")
    if cfg.game.name == "synthetic" and cfg.game.synthetic is None:
        raise ConfigError("root.game.synthetic: required when game.name is 'synthetic'")
    if cfg.game.name == "poker":
        if cfg.game.poker is None:
            raise ConfigError("root.game.poker: required when game.name is 'poker'")
        if cfg.completion.backend != "poker_gpu":
            raise ConfigError("root.completion.backend: poker requires 'poker_gpu'")
    if cfg.rollout.beta_mode not in {"sequential", "fixed", "neyman"}:
        raise ConfigError("root.rollout.beta_mode: expected 'sequential', 'fixed', or 'neyman'")
    if cfg.rollout.reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ConfigError(
            "root.rollout.reach_storage: expected 'full', 'decisions_only', or 'recompute'"
        )
    if cfg.loss.entropy_schedule not in {"const", "linear_decay"}:
        raise ConfigError("root.loss.entropy_schedule: expected 'const' or 'linear_decay'")


def load_config(path: str | Path) -> RootConfig:
    """Load ``path`` as a strict :class:`RootConfig`.

    Unknown keys and absent required fields raise :class:`ConfigError` with a
    dotted path to the offending location.
    """

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{config_path}: could not load configuration: {exc}") from exc
    if raw is None:
        raw = {}
    cfg = _parse_dataclass(RootConfig, raw, "root")
    _validate_config(cfg)
    return cfg


def seed_all(seed: int, rank: int = 0) -> tuple[torch.Generator, np.random.Generator]:
    """Return explicit Torch and NumPy generators seeded for one rank.

    The rank-local seed is ``seed * 1000 + rank``. Torch's process generators
    are also seeded; NumPy global state is deliberately untouched.
    """

    rank_seed = seed * 1000 + rank
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(rank_seed)
    np_gen = np.random.default_rng(rank_seed)
    return torch_gen, np_gen

