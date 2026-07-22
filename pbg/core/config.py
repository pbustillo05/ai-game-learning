"""Strict dataclass configuration loading and explicit random generators."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
import math
from pathlib import Path
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

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
class AdaptiveKKTConfig:
    """Dynamic public branching controller parameters."""

    beta_max: float
    warmup_iters: int
    target_tree_size: float
    kappa_initial: float
    kappa_min: float
    kappa_max: float
    gamma_kappa: float
    eta_kappa: float


@dataclass
class RolloutConfig:
    """Configure rollout storage and optional branching intensity.

    Sequential widths are explicitly per GPU.  ``roots_per_gpu`` is a mutable
    runtime view initialized from ``sequential_roots_per_gpu_initial``; the
    checkpointed memory controller may change it only between iterations.
    """

    sequential_roots_per_gpu_initial: int
    beta: dict[str, float] | None
    branching_mode: Literal["fixed", "dynamic"] | None = None
    sequential_roots_per_gpu_minimum: int | None = None
    sequential_roots_per_gpu_maximum: int | None = None
    family_batch_rows: int | None = None
    max_active_trajectories: int | None = None
    storage_page_rows: int | None = None
    execution_memory_bytes: int | None = None
    memory_calibration_window: int = 32
    memory_growth_fraction: float = 0.10
    memory_growth_delay: int = 8
    memory_prediction_z: float = 3.0
    adaptive_kkt: AdaptiveKKTConfig | None = None
    branching_min_nodes: int | None = None
    safe_tree_size: float | None = None
    queue_capacity_rows: int | None = None

    def __post_init__(self) -> None:
        self.roots_per_gpu = int(self.sequential_roots_per_gpu_initial)


@dataclass
class CompletionConfig:
    """Configure Monte Carlo completion work."""

    n_samples: int | None


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
    utility_normalization: str = "none"
    entropy_decay_iters: int | None = None
    c_t: float = 0.0
    c_c: float = 0.0
    epsilon_t: float = 1.0e-6


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
class ProbeConfig:
    """Train one shared approximate best response against a fixed checkpoint."""

    target_checkpoint: str
    target_sha256: str | None = None
    initialization: str = "random"


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

    device: Literal["cuda", "mps"]
    game: GameConfig
    rollout: RolloutConfig
    completion: CompletionConfig
    model: dict[str, Any]
    loss: LossConfig
    training: TrainingConfig
    probe: ProbeConfig | None = None
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
    if origin is Literal:
        if value not in args:
            expected = ", ".join(repr(item) for item in args)
            raise ConfigError(f"{path}: expected one of {expected}")
        return value
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
    """Validate all game-independent values in the frozen schema."""

    def positive(path: str, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{path}: expected a finite positive value")

    def nonnegative(path: str, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"{path}: expected a finite nonnegative value")

    modes = {"normal", "rao_blackwell", "branched_rao_blackwell"}
    if cfg.training.mode not in modes:
        raise ConfigError(
            "root.training.mode: expected 'normal', 'rao_blackwell', or "
            "'branched_rao_blackwell'"
        )
    if cfg.probe is not None:
        if cfg.training.mode != "rao_blackwell":
            raise ConfigError(
                "root.probe: best-response probes require training.mode 'rao_blackwell'"
            )
        if not cfg.probe.target_checkpoint.strip():
            raise ConfigError("root.probe.target_checkpoint: expected a nonempty path")
        if cfg.probe.initialization not in {"random", "target"}:
            raise ConfigError(
                "root.probe.initialization: expected 'random' or 'target'"
            )
        if cfg.probe.target_sha256 is not None:
            digest = cfg.probe.target_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ConfigError(
                    "root.probe.target_sha256: expected 64 lowercase hexadecimal characters"
                )
            cfg.probe.target_sha256 = digest
    if cfg.training.mode == "branched_rao_blackwell":
        if cfg.rollout.branching_mode is None:
            raise ConfigError(
                "root.rollout.branching_mode: branched_rao_blackwell requires "
                "an explicit 'fixed' or 'dynamic' mode"
            )
        if cfg.rollout.beta is None or "default" not in cfg.rollout.beta:
            raise ConfigError(
                "root.rollout.beta.default: required for branched_rao_blackwell"
            )
        for bucket, beta in cfg.rollout.beta.items():
            if not math.isfinite(beta) or beta < 1.0:
                raise ConfigError(
                    f"root.rollout.beta.{bucket}: expected a finite value at least one"
                )
        if cfg.rollout.sequential_roots_per_gpu_initial != 1:
            raise ConfigError(
                "root.rollout.sequential_roots_per_gpu_initial: use 1 for "
                "branched_rao_blackwell; "
                "production creates exactly one root per rank"
            )
        minimum_nodes = cfg.rollout.branching_min_nodes
        if minimum_nodes is not None and (
            isinstance(minimum_nodes, bool)
            or not isinstance(minimum_nodes, int)
            or minimum_nodes <= 1
        ):
            raise ConfigError(
                "root.rollout.branching_min_nodes: expected an integer greater than one"
            )
        queue_capacity = cfg.rollout.queue_capacity_rows
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity <= 0
        ):
            raise ConfigError(
                "root.rollout.queue_capacity_rows: branched_rao_blackwell requires "
                "a positive integer"
            )
        safe_tree_size = cfg.rollout.safe_tree_size
        if safe_tree_size is not None:
            positive("root.rollout.safe_tree_size", safe_tree_size)
            if safe_tree_size < 1.0:
                raise ConfigError("root.rollout.safe_tree_size: must be at least one")
    else:
        if cfg.rollout.branching_mode is not None:
            raise ConfigError(
                "root.rollout.branching_mode: only valid for branched_rao_blackwell"
            )
        if cfg.rollout.beta is not None:
            raise ConfigError(
                f"root.rollout.beta: only valid for branched_rao_blackwell, not {cfg.training.mode}"
            )
        if cfg.rollout.branching_min_nodes is not None:
            raise ConfigError(
                "root.rollout.branching_min_nodes: only valid for branched_rao_blackwell"
            )
        if cfg.rollout.safe_tree_size is not None:
            raise ConfigError(
                "root.rollout.safe_tree_size: only valid for branched_rao_blackwell"
            )
        if cfg.rollout.queue_capacity_rows is not None:
            raise ConfigError(
                "root.rollout.queue_capacity_rows: only valid for branched_rao_blackwell"
            )
    adaptive = cfg.rollout.adaptive_kkt
    if cfg.rollout.branching_mode == "dynamic" and adaptive is None:
        raise ConfigError(
            "root.rollout.adaptive_kkt: required when branching_mode is 'dynamic'"
        )
    if cfg.rollout.branching_mode != "dynamic" and adaptive is not None:
        raise ConfigError(
            "root.rollout.adaptive_kkt: only valid when branching_mode is 'dynamic'"
        )
    if adaptive is not None:
        for name in (
            "beta_max", "target_tree_size", "kappa_initial",
            "kappa_min", "kappa_max", "gamma_kappa", "eta_kappa",
        ):
            positive(f"root.rollout.adaptive_kkt.{name}", getattr(adaptive, name))
        if isinstance(adaptive.warmup_iters, bool) or adaptive.warmup_iters < 0:
            raise ConfigError(
                "root.rollout.adaptive_kkt.warmup_iters: expected a nonnegative integer"
            )
        if adaptive.beta_max < 1.0:
            raise ConfigError("root.rollout.adaptive_kkt.beta_max: must be at least one")
        if not adaptive.kappa_min <= adaptive.kappa_initial <= adaptive.kappa_max:
            raise ConfigError(
                "root.rollout.adaptive_kkt kappa values must satisfy min <= initial <= max"
            )
        if adaptive.gamma_kappa > 1.0 or adaptive.eta_kappa > 1.0:
            raise ConfigError(
                "root.rollout.adaptive_kkt gamma_kappa and eta_kappa must be in (0,1]"
            )
    if cfg.training.mode == "branched_rao_blackwell":
        if not bool(cfg.model.get("use_branching_heads", False)):
            raise ConfigError(
                "root.model.use_branching_heads: must be true in both branching modes"
            )
        if cfg.loss.c_t <= 0.0 or cfg.loss.c_c <= 0.0:
            raise ConfigError(
                "root.loss.c_t and root.loss.c_c must be positive in both branching modes"
            )
    initial = cfg.rollout.sequential_roots_per_gpu_initial
    minimum = cfg.rollout.sequential_roots_per_gpu_minimum
    maximum = cfg.rollout.sequential_roots_per_gpu_maximum
    if minimum is None:
        minimum = initial
        cfg.rollout.sequential_roots_per_gpu_minimum = minimum
    if maximum is None:
        maximum = initial
        cfg.rollout.sequential_roots_per_gpu_maximum = maximum
    if any(isinstance(value, bool) or value <= 0 for value in (initial, minimum, maximum)):
        raise ConfigError("root.rollout sequential per-GPU root widths must be positive integers")
    if not minimum <= initial <= maximum:
        raise ConfigError(
            "root.rollout sequential per-GPU root widths must satisfy minimum <= initial <= maximum"
        )
    if cfg.training.mode == "branched_rao_blackwell" and (minimum, maximum) != (1, 1):
        raise ConfigError("branched_rao_blackwell requires all per-GPU root widths equal one")
    if cfg.rollout.memory_calibration_window <= 0 or cfg.rollout.memory_growth_delay <= 0:
        raise ConfigError("root.rollout memory windows/delays must be positive integers")
    if not 0.0 < cfg.rollout.memory_growth_fraction < 1.0:
        raise ConfigError("root.rollout.memory_growth_fraction: expected a value in (0,1)")
    if not math.isfinite(cfg.rollout.memory_prediction_z) or cfg.rollout.memory_prediction_z <= 0:
        raise ConfigError("root.rollout.memory_prediction_z: expected a finite positive value")
    for name in (
        "family_batch_rows",
        "max_active_trajectories",
        "storage_page_rows",
        "execution_memory_bytes",
    ):
        value = getattr(cfg.rollout, name)
        if value is not None and (isinstance(value, bool) or value <= 0):
            raise ConfigError(f"root.rollout.{name}: expected a positive integer")
    if cfg.loss.entropy_schedule not in {"const", "linear_decay"}:
        raise ConfigError("root.loss.entropy_schedule: expected 'const' or 'linear_decay'")
    if cfg.loss.utility_normalization not in {"none", "stack"}:
        raise ConfigError(
            "root.loss.utility_normalization: expected 'none' or 'stack'"
        )
    entropy_decay_iters = cfg.loss.entropy_decay_iters
    if entropy_decay_iters is not None and (
        isinstance(entropy_decay_iters, bool)
        or not isinstance(entropy_decay_iters, int)
        or entropy_decay_iters <= 0
    ):
        raise ConfigError(
            "root.loss.entropy_decay_iters: expected a positive integer or null"
        )
    if cfg.loss.entropy_schedule == "const" and entropy_decay_iters is not None:
        raise ConfigError(
            "root.loss.entropy_decay_iters: only valid with linear_decay"
        )
    if cfg.loss.value_loss not in {"mse", "clipped"}:
        raise ConfigError("root.loss.value_loss: expected 'mse' or 'clipped'")
    if not math.isfinite(cfg.loss.lam) or not 0.0 <= cfg.loss.lam <= 1.0:
        raise ConfigError("root.loss.lam: expected a finite value in [0,1]")
    if not math.isfinite(cfg.loss.clip_eps) or not 0.0 < cfg.loss.clip_eps <= 1.0:
        raise ConfigError("root.loss.clip_eps: expected a finite value in (0,1]")
    positive("root.loss.clip_eps_v", cfg.loss.clip_eps_v)
    nonnegative("root.loss.c_v", cfg.loss.c_v)
    nonnegative("root.loss.c_h", cfg.loss.c_h)
    nonnegative("root.loss.c_kl", cfg.loss.c_kl)
    nonnegative("root.loss.c_t", cfg.loss.c_t)
    nonnegative("root.loss.c_c", cfg.loss.c_c)
    positive("root.loss.epsilon_t", cfg.loss.epsilon_t)
    if cfg.training.iters <= 0:
        raise ConfigError("root.training.iters: expected a positive integer")
    if cfg.training.gd_steps_per_iter <= 0:
        raise ConfigError("root.training.gd_steps_per_iter: expected a positive integer")
    if cfg.training.minibatch_size <= 0:
        raise ConfigError("root.training.minibatch_size: expected a positive integer")
    positive("root.training.lr", cfg.training.lr)
    for index, beta in enumerate(cfg.training.adam_betas):
        if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
            raise ConfigError(
                f"root.training.adam_betas[{index}]: expected a finite value in [0,1)"
            )
    positive("root.training.grad_clip", cfg.training.grad_clip)
    if cfg.training.seed < 0:
        raise ConfigError("root.training.seed: expected a nonnegative integer")
    if cfg.training.checkpoint_every <= 0:
        raise ConfigError("root.training.checkpoint_every: expected a positive integer")
    if not cfg.training.out_dir.strip():
        raise ConfigError("root.training.out_dir: expected a nonempty path")
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
        if profile.attention_backend not in {"flex", "sdpa", "chunked_dense", "auto"}:
            raise ConfigError(
                "hardware.attention_backend: expected 'flex', 'sdpa', "
                "'chunked_dense', or 'auto'"
            )
        if profile.precision not in {"fp16", "bf16", "fp32"}:
            raise ConfigError("hardware.precision: expected 'fp16', 'bf16', or 'fp32'")
        positive_integers = {
            "query_chunk_size": profile.query_chunk_size,
            "training_max_tokens": profile.training_max_tokens,
            "training_max_token_pairs": profile.training_max_token_pairs,
            "inference_max_tokens": profile.inference_max_tokens,
            "inference_max_token_pairs": profile.inference_max_token_pairs,
            "max_devices": profile.max_devices,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or value <= 0:
                raise ConfigError(f"hardware.{name}: expected a positive integer")
        if not math.isfinite(profile.max_memory_fraction) or not 0.0 < profile.max_memory_fraction <= 1.0:
            raise ConfigError("hardware.max_memory_fraction: expected a finite value in (0,1]")
        nonnegative("hardware.reserve_vram_gib", profile.reserve_vram_gib)
        if profile.worker_count < 0:
            raise ConfigError("hardware.worker_count: expected a nonnegative integer")
        if profile.checkpoint_every is not None and profile.checkpoint_every <= 0:
            raise ConfigError("hardware.checkpoint_every: expected a positive integer or null")
        if profile.expected_compute_capability is not None:
            major, minor = profile.expected_compute_capability
            if major <= 0 or minor < 0:
                raise ConfigError(
                    "hardware.expected_compute_capability: expected [positive major, nonnegative minor]"
                )
        for name in ("min_vram_gib", "max_vram_gib"):
            value = getattr(profile, name)
            if value is not None:
                positive(f"hardware.{name}", value)
        if (
            profile.min_vram_gib is not None
            and profile.max_vram_gib is not None
            and profile.min_vram_gib > profile.max_vram_gib
        ):
            raise ConfigError(
                "hardware.min_vram_gib: must not exceed hardware.max_vram_gib"
            )
        if not profile.name.strip():
            raise ConfigError("hardware.name: expected a nonempty name")
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


def parse_config_mapping(
    raw: dict[str, Any], *, hardware_profile: str | Path | None = None
) -> RootConfig:
    """Parse an in-memory scientific configuration, such as a checkpoint payload."""

    cfg = _parse_dataclass(RootConfig, raw, "root")
    if hardware_profile is not None:
        profile_path = _hardware_path(hardware_profile)
        cfg.hardware = _parse_dataclass(
            HardwareProfileConfig,
            _load_yaml(profile_path, "hardware profile"),
            "hardware",
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
