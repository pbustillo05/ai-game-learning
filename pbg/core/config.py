"""Strict dataclass configuration loading and explicit random generators."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
import math
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
    """Configure the sole active game: equal-stack NLHE cash poker."""

    n_players: int
    stack_bb: float
    rank_count: int


@dataclass
class RolloutConfig:
    """Configure rollout storage and optional branching intensity.

    ``sequential_starts`` is a global logical-work count in normal and
    sequential RB.  The training launcher partitions it into collision-free
    rank-local prefixes.  Branched RB ignores the count and always samples one
    root per rank; it is retained there as the explicit value ``1`` so one
    frozen schema can serve every mode.
    """

    sequential_starts: int
    beta: dict[str, float] | None
    node_budget: int
    reach_storage: str
    family_batch_rows: int | None = None
    max_active_trajectories: int | None = None
    arena_chunk_nodes: int | None = None
    execution_memory_bytes: int | None = None


@dataclass
class CompletionConfig:
    """Configure Monte Carlo completion at pre-terminal nodes."""

    n_samples: int | None
    backend: str


@dataclass
class LossConfig:
    """Configure the minimal clipped-policy, value, and entropy losses."""

    name: str
    lam: float
    clip_eps: float
    clip_eps_v: float
    c_v: float
    c_h: float
    entropy_schedule: str
    value_loss: str = "mse"
    c_kl: float = 0.0


@dataclass
class TrainingConfig:
    """Select the estimator pipeline and configure optimization."""

    mode: str
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
    reproducibility: str = "strict"
    kl_ref_interval: int = 1
    optimizer: str = "adam"
    ppo_epochs: int | None = None


@dataclass
class HardwareProfileConfig:
    """Execution-only limits selected independently of scientific settings."""

    name: str
    scheduler: str
    attention_backend: str
    query_chunk_size: int
    training_max_tokens: int
    training_max_token_pairs: int
    inference_max_tokens: int
    inference_max_token_pairs: int
    reserve_vram_gib: float
    max_memory_fraction: float
    max_devices: int
    expected_compute_capability: tuple[int, int] | None = None
    min_vram_gib: float | None = None
    max_vram_gib: float | None = None
    precision: str = "fp16"
    worker_count: int = 0
    checkpoint_every: int | None = None
    slurm: dict[str, Any] | None = None


@dataclass
class RootConfig:
    """Top-level configuration shared by all game implementations."""

    game: GameConfig
    rollout: RolloutConfig
    completion: CompletionConfig
    model: dict[str, Any]
    loss: LossConfig
    training: TrainingConfig
    hardware: HardwareProfileConfig | None = None

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
    """Validate game-independent frozen enumerations."""

    if cfg.completion.backend not in {"auto", "cpu", "cuda"}:
        raise ConfigError(
            "root.completion.backend: poker requires 'auto', 'cpu', or 'cuda'"
        )
    modes = {"normal", "rao_blackwell", "branched_rao_blackwell"}
    if cfg.training.mode not in modes:
        raise ConfigError(
            "root.training.mode: expected 'normal', 'rao_blackwell', or "
            "'branched_rao_blackwell'"
        )
    if cfg.training.mode == "branched_rao_blackwell":
        if cfg.rollout.beta is None or "default" not in cfg.rollout.beta:
            raise ConfigError(
                "root.rollout.beta.default: required for branched_rao_blackwell"
            )
        for bucket, beta in cfg.rollout.beta.items():
            if not math.isfinite(beta) or beta < 1.0:
                raise ConfigError(
                    f"root.rollout.beta.{bucket}: expected a finite value at least one"
                )
        if cfg.rollout.sequential_starts != 1:
            raise ConfigError(
                "root.rollout.sequential_starts: use 1 for branched_rao_blackwell; "
                "production creates exactly one root per rank"
            )
    elif cfg.rollout.beta is not None:
        raise ConfigError(
            f"root.rollout.beta: only valid for branched_rao_blackwell, not {cfg.training.mode}"
        )
    if cfg.rollout.sequential_starts <= 0:
        raise ConfigError("root.rollout.sequential_starts: expected a positive integer")
    if cfg.rollout.reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ConfigError(
            "root.rollout.reach_storage: expected 'full', 'decisions_only', or 'recompute'"
        )
    for name in (
        "family_batch_rows",
        "max_active_trajectories",
        "arena_chunk_nodes",
        "execution_memory_bytes",
    ):
        value = getattr(cfg.rollout, name)
        if value is not None and (isinstance(value, bool) or value <= 0):
            raise ConfigError(f"root.rollout.{name}: expected a positive integer")
    if cfg.loss.entropy_schedule not in {"const", "linear_decay"}:
        raise ConfigError("root.loss.entropy_schedule: expected 'const' or 'linear_decay'")
    if cfg.loss.value_loss not in {"mse", "clipped"}:
        raise ConfigError("root.loss.value_loss: expected 'mse' or 'clipped'")
    if not math.isfinite(cfg.loss.c_kl) or cfg.loss.c_kl < 0.0:
        raise ConfigError("root.loss.c_kl: expected a finite nonnegative value")
    if isinstance(cfg.training.kl_ref_interval, bool) or cfg.training.kl_ref_interval <= 0:
        raise ConfigError("root.training.kl_ref_interval: expected a positive integer")
    if cfg.training.optimizer not in {"adam", "adamw"}:
        raise ConfigError("root.training.optimizer: expected 'adam' or 'adamw'")
    if cfg.training.ppo_epochs is not None:
        if (
            isinstance(cfg.training.ppo_epochs, bool)
            or cfg.training.ppo_epochs <= 0
        ):
            raise ConfigError("root.training.ppo_epochs: expected a positive integer")
        if cfg.training.gd_steps_per_iter != 1:
            raise ConfigError(
                "root.training.gd_steps_per_iter: must equal one when ppo_epochs is set"
            )
        if cfg.training.mode == "branched_rao_blackwell":
            raise ConfigError(
                "root.training.ppo_epochs: shuffled epochs currently support only "
                "normal and sequential Rao--Blackwell training"
            )
    if cfg.training.reproducibility not in {"strict", "seeded", "throughput"}:
        raise ConfigError(
            "root.training.reproducibility: expected 'strict', 'seeded', or 'throughput'"
        )
    from pbg.core.losses import COMPOSITES

    loss_spec = COMPOSITES.get(cfg.loss.name)
    if loss_spec is None:
        raise ConfigError(f"root.loss.name: unknown composite {cfg.loss.name!r}")
    if cfg.training.mode not in loss_spec.modes:
        raise ConfigError(
            f"root.loss.name: {cfg.loss.name!r} does not support mode {cfg.training.mode!r}"
        )
    if cfg.training.mode == "normal":
        if cfg.completion.n_samples is not None:
            raise ConfigError(
                "root.completion.n_samples: must be null in normal mode"
            )
    elif (
        isinstance(cfg.completion.n_samples, bool)
        or not isinstance(cfg.completion.n_samples, int)
        or cfg.completion.n_samples <= 0
    ):
        raise ConfigError(
            "root.completion.n_samples: RB modes require a positive integer"
        )

    profile = cfg.hardware
    if profile is not None:
        if profile.scheduler not in {"none", "slurm"}:
            raise ConfigError("hardware.scheduler: expected 'none' or 'slurm'")
        if profile.attention_backend not in {"flex", "chunked_dense", "auto"}:
            raise ConfigError(
                "hardware.attention_backend: expected 'flex', 'chunked_dense', or 'auto'"
            )
        if profile.precision not in {"fp16", "bf16", "fp32"}:
            raise ConfigError("hardware.precision: expected 'fp16', 'bf16', or 'fp32'")
        positive = {
            "query_chunk_size": profile.query_chunk_size,
            "training_max_tokens": profile.training_max_tokens,
            "training_max_token_pairs": profile.training_max_token_pairs,
            "inference_max_tokens": profile.inference_max_tokens,
            "inference_max_token_pairs": profile.inference_max_token_pairs,
            "max_devices": profile.max_devices,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or value <= 0:
                raise ConfigError(f"hardware.{name}: expected a positive integer")
        if not 0.0 < profile.max_memory_fraction <= 1.0:
            raise ConfigError("hardware.max_memory_fraction: expected a value in (0,1]")
        if profile.reserve_vram_gib < 0.0:
            raise ConfigError("hardware.reserve_vram_gib: expected a nonnegative value")
        if profile.scheduler == "slurm" and profile.slurm is None:
            raise ConfigError("hardware.slurm: required when scheduler is 'slurm'")


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: could not load {label}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: {label} must be a mapping")
    return raw


def _hardware_path(profile: str | Path) -> Path:
    candidate = Path(profile)
    if candidate.exists() or candidate.suffix in {".yaml", ".yml"}:
        return candidate
    root = Path(__file__).resolve().parents[2]
    return root / "configs" / "hardware" / f"{candidate.name}.yaml"


def parse_config(
    path: str | Path, *, hardware_profile: str | Path | None = None
) -> RootConfig:
    """Parse ``path`` as a strict game-independent :class:`RootConfig`.

    Unknown keys and absent required fields raise :class:`ConfigError` with a
    dotted path to the offending location.
    """

    config_path = Path(path)
    raw = _load_yaml(config_path, "configuration")
    cfg = _parse_dataclass(RootConfig, raw, "root")
    if hardware_profile is not None:
        profile_path = _hardware_path(hardware_profile)
        profile_raw = _load_yaml(profile_path, "hardware profile")
        cfg.hardware = _parse_dataclass(
            HardwareProfileConfig, profile_raw, "hardware"
        )
    _validate_config(cfg)
    return cfg


def seed_all(
    seed: int,
    rank: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Generator, np.random.Generator]:
    """Return explicit Torch and NumPy generators seeded for one rank.

    The rank-local seed is ``seed * 1000 + rank``. Torch's process generators
    are also seeded; NumPy global state is deliberately untouched.
    """

    rank_seed = seed * 1000 + rank
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    torch_gen = torch.Generator(device=torch.device(device or "cpu"))
    torch_gen.manual_seed(rank_seed)
    np_gen = np.random.default_rng(rank_seed)
    return torch_gen, np_gen
