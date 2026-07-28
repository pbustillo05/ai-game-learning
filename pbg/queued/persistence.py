"""Strict queued checkpoints and the one-time legacy converter."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from pbg.queued.config import Phase5Config, SequentialModeConfig
from pbg.queued.learner import ReferenceState
from pbg.queued.optimization import DenseAdam, LossScaler

CHECKPOINT_SCHEMA = "pbg-queued-checkpoint-v2"
QUEUED_CHECKPOINT_V1 = "pbg-queued-checkpoint-v1"
CONVERSION_SCHEMA = "pbg-queued-conversion-v1"
LEGACY_SCHEMA = "pbg-checkpoint-v8"
LEGACY_SCHEMA_V7 = "pbg-checkpoint-v7"


class PersistenceError(RuntimeError):
    """Checkpoint identity, schema, or resume state is invalid."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clone_cpu(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return value


def _atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass
class RunState:
    run_id: str
    iteration: int = 0
    root_count: int = 1
    retry_epoch: int = 0
    attempt_logical_base: int = 0
    logical_work: dict[str, int] | None = None
    recovered_overflow: bool = False
    overflow_recoveries: list[dict[str, object]] | None = None
    metrics_prefix: list[dict[str, object]] | None = None
    branch_controller_state: dict[str, object] | None = None
    cost_safeguard_state: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or self.iteration < 0
            or self.root_count <= 0
            or self.retry_epoch < 0
            or self.attempt_logical_base < 0
        ):
            raise ValueError("run identity and iteration are invalid")
        if self.logical_work is None:
            self.logical_work = {
                "trajectories": 0,
                "completion_samples": 0,
                "minibatches": 0,
                "optimizer_steps": 0,
            }
        if self.overflow_recoveries is None:
            self.overflow_recoveries = []
        if self.metrics_prefix is None:
            self.metrics_prefix = []
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.logical_work.values()
        ):
            raise ValueError("logical work counters must be nonnegative integers")
        iterations = [int(item["iteration"]) for item in self.metrics_prefix]
        if iterations and iterations != list(range(iterations[0], iterations[-1] + 1)):
            raise ValueError("metrics prefix must be gap-free")

    def state_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "root_count": self.root_count,
            "retry_epoch": self.retry_epoch,
            "attempt_logical_base": self.attempt_logical_base,
            "logical_work": dict(self.logical_work),
            "recovered_overflow": self.recovered_overflow,
            "overflow_recoveries": list(self.overflow_recoveries),
            "metrics_prefix": list(self.metrics_prefix),
            "branch_controller_state": (
                None
                if self.branch_controller_state is None
                else dict(self.branch_controller_state)
            ),
            "cost_safeguard_state": (
                None
                if self.cost_safeguard_state is None
                else dict(self.cost_safeguard_state)
            ),
        }


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: DenseAdam,
    scaler: LossScaler,
    reference_model: nn.Module,
    reference_state: ReferenceState,
    run_state: RunState,
    scientific_config: Mapping[str, object],
    execution_config: Mapping[str, object],
    generator: torch.Generator,
) -> str:
    """Atomically persist the exact completed optimizer boundary."""

    state = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "scientific_config": dict(scientific_config),
        "scientific_sha256": _canonical_hash(scientific_config),
        "execution_config": dict(execution_config),
        "model": _clone_cpu(model.state_dict()),
        "optimizer": _clone_cpu(optimizer.state_dict()),
        "scaler": scaler.state_dict(),
        "reference_model": _clone_cpu(reference_model.state_dict()),
        "reference_state": reference_state.state_dict(),
        "run_state": run_state.state_dict(),
        "explicit_rng": generator.get_state().cpu(),
        "ambient_cpu_rng": torch.get_rng_state(),
        "ambient_cuda_rng": tuple(
            value.cpu() for value in torch.cuda.get_rng_state_all()
        ),
    }
    destination = Path(path)
    _atomic_torch_save(state, destination)
    return sha256_file(destination)


_CHECKPOINT_FIELDS = {
    "checkpoint_schema",
    "scientific_config",
    "scientific_sha256",
    "execution_config",
    "model",
    "optimizer",
    "scaler",
    "reference_model",
    "reference_state",
    "run_state",
    "explicit_rng",
    "ambient_cpu_rng",
    "ambient_cuda_rng",
}


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: DenseAdam,
    reference_model: nn.Module,
    expected_scientific_config: Mapping[str, object],
    generator: torch.Generator,
    restore_ambient_rng: bool = True,
) -> tuple[LossScaler, ReferenceState, RunState, Mapping[str, object]]:
    """Strictly resume queued state; legacy schemas are never loaded here."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != _CHECKPOINT_FIELDS:
        raise PersistenceError("checkpoint has unknown or missing top-level fields")
    if state["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise PersistenceError(
            "runtime accepts only queued checkpoints; use the offline converter"
        )
    expected_hash = _canonical_hash(expected_scientific_config)
    if state["scientific_sha256"] != expected_hash or state[
        "scientific_config"
    ] != dict(expected_scientific_config):
        raise PersistenceError("resume rejected scientific configuration drift")
    model.load_state_dict(state["model"], strict=True)
    reference_model.load_state_dict(state["reference_model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler = LossScaler.from_state_dict(state["scaler"])
    optimizer.scaler = scaler
    reference = ReferenceState.from_state_dict(state["reference_state"])
    run_data = state["run_state"]
    expected_run = {
        "run_id",
        "iteration",
        "root_count",
        "retry_epoch",
        "attempt_logical_base",
        "logical_work",
        "recovered_overflow",
        "overflow_recoveries",
        "metrics_prefix",
        "branch_controller_state",
        "cost_safeguard_state",
    }
    if not isinstance(run_data, dict) or set(run_data) != expected_run:
        raise PersistenceError("run state has unknown or missing fields")
    run = RunState(**run_data)
    generator.set_state(state["explicit_rng"])
    if restore_ambient_rng:
        torch.set_rng_state(state["ambient_cpu_rng"])
        cuda_states = state["ambient_cuda_rng"]
        if len(cuda_states) != torch.cuda.device_count():
            raise PersistenceError("checkpoint CUDA RNG rank/device count differs")
        torch.cuda.set_rng_state_all(list(cuda_states))
    return scaler, reference, run, state["execution_config"]


def convert_queued_checkpoint_v1(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Convert one exact queued-v1 checkpoint into strict queued-v2."""

    source = Path(source_path)
    destination = Path(destination_path)
    if not source.is_file():
        raise PersistenceError("queued-v1 checkpoint source does not exist")
    if destination.exists():
        raise PersistenceError("queued-v2 conversion destination already exists")
    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != _CHECKPOINT_FIELDS:
        raise PersistenceError("queued-v1 checkpoint fields are invalid")
    if state["checkpoint_schema"] != QUEUED_CHECKPOINT_V1:
        raise PersistenceError("offline conversion requires queued checkpoint v1")
    prior_run_fields = {
        "run_id",
        "iteration",
        "root_count",
        "retry_epoch",
        "attempt_logical_base",
        "logical_work",
        "recovered_overflow",
        "overflow_recoveries",
        "metrics_prefix",
    }
    run_data = state["run_state"]
    if not isinstance(run_data, dict) or set(run_data) != prior_run_fields:
        raise PersistenceError("queued-v1 run state is not exact")
    converted = {
        **state,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "run_state": {
            **run_data,
            "branch_controller_state": None,
            "cost_safeguard_state": None,
        },
    }
    source_sha256 = sha256_file(source)
    _atomic_torch_save(converted, destination)
    result = {
        "schema": CONVERSION_SCHEMA,
        "source_schema": QUEUED_CHECKPOINT_V1,
        "result_schema": CHECKPOINT_SCHEMA,
        "source_sha256": source_sha256,
        "result_sha256": sha256_file(destination),
        "destination": str(destination),
    }
    _atomic_json_save(
        result,
        destination.with_suffix(destination.suffix + ".conversion.json"),
    )
    return result


_LEGACY_FIELDS_V7 = {
    "adaptive_work_state",
    "branching_controller_schema_version",
    "branching_controller_state",
    "checkpoint_schema_version",
    "config",
    "controller_schema_version",
    "hardware_profile",
    "initialization",
    "iteration",
    "logical_work",
    "model",
    "numpy_generator",
    "optimizer",
    "probe_target",
    "rank_rng_states",
    "reference_effective_hands_rollout",
    "reference_iteration",
    "reference_model",
    "run_id",
    "scaler",
    "storage_schema_version",
    "torch_cpu_rng",
    "torch_cuda_rng",
    "torch_generator",
}
_LEGACY_FIELDS_V8 = _LEGACY_FIELDS_V7 | {
    "cost_safeguard_controller_schema_version",
    "cost_safeguard_controller_state",
}


def _convert_optimizer(
    legacy: Mapping[str, object], training: Mapping[str, object]
) -> dict[str, object]:
    if set(legacy) != {"state", "param_groups"}:
        raise PersistenceError("legacy optimizer has unknown fields")
    groups = legacy["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise PersistenceError("Phase 5 converter requires one optimizer group")
    group = groups[0]
    parameter_ids = group["params"]
    state = legacy["state"]
    if set(state) != set(parameter_ids):
        raise PersistenceError("legacy optimizer parameter state is incomplete")
    exp_avg = torch.cat(
        tuple(state[key]["exp_avg"].reshape(-1) for key in parameter_ids)
    )
    exp_avg_sq = torch.cat(
        tuple(state[key]["exp_avg_sq"].reshape(-1) for key in parameter_ids)
    )
    steps = {int(float(state[key]["step"])) for key in parameter_ids}
    if len(steps) != 1:
        raise PersistenceError("legacy optimizer parameters have divergent steps")
    betas = tuple(training["adam_betas"])
    optimizer_name = str(training["optimizer"])
    config = {
        "name": optimizer_name,
        "learning_rate": float(group["lr"]),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "epsilon": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "grad_clip": float(training["grad_clip"]),
    }
    return {
        "schema": "queued-dense-adam-v1",
        "config": config,
        "step": steps.pop(),
        "exp_avg": exp_avg.float(),
        "exp_avg_sq": exp_avg_sq.float(),
    }


def _convert_scaler(legacy: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(legacy) != expected:
        raise PersistenceError("legacy scaler has unknown or missing fields")
    scale = legacy["scale"]
    tracker = legacy["_growth_tracker"]
    return {
        "enabled": True,
        "scale": float(scale),
        "growth_factor": float(legacy["growth_factor"]),
        "backoff_factor": float(legacy["backoff_factor"]),
        "growth_interval": int(legacy["growth_interval"]),
        "growth_tracker": int(tracker),
    }


def convert_legacy_sequential(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
    *,
    expected_training_regime: str | None = None,
    expected_estimator: str | None = None,
) -> dict[str, object]:
    """Convert an approved sequential sampled, RB, or probe artifact offline."""

    source = Path(source_path)
    destination = Path(destination_path)
    if destination.exists():
        raise PersistenceError("conversion destination already exists")
    source_hash = sha256_file(source)
    legacy = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(legacy, dict):
        raise PersistenceError("legacy checkpoint must be a mapping")
    source_schema = legacy.get("checkpoint_schema_version")
    expected_fields = {
        LEGACY_SCHEMA_V7: _LEGACY_FIELDS_V7,
        LEGACY_SCHEMA: _LEGACY_FIELDS_V8,
    }.get(source_schema)
    if expected_fields is None or set(legacy) != expected_fields:
        raise PersistenceError("legacy checkpoint schema has unknown or missing fields")
    if source_schema == LEGACY_SCHEMA and (
        legacy["cost_safeguard_controller_schema_version"] is not None
        or legacy["cost_safeguard_controller_state"] is not None
    ):
        raise PersistenceError("sequential artifact cannot carry a cost safeguard")
    config = legacy["config"]
    legacy_mode = config["training"]["mode"]
    if config["rollout"]["branching_mode"] is not None:
        raise PersistenceError("sequential converter rejects branched artifacts")
    completion_samples = config["completion"]["n_samples"]
    if legacy_mode == "nor" + "mal" and completion_samples is None:
        regime, estimator = "self_play", "sampled"
    elif legacy_mode == "rao_blackwell" and completion_samples is not None:
        probe = config.get("probe")
        if (probe is None) != (legacy["probe_target"] is None):
            raise PersistenceError(
                "probe target metadata disagrees with legacy configuration"
            )
        regime, estimator = (
            ("best_response", "rao_blackwell")
            if probe is not None
            else ("self_play", "rao_blackwell")
        )
    else:
        raise PersistenceError("legacy artifact is not an approved sequential mode")
    if (
        expected_training_regime is not None and regime != expected_training_regime
    ) or (expected_estimator is not None and estimator != expected_estimator):
        raise PersistenceError("legacy artifact mode differs from requested conversion")
    training = config["training"]
    legacy_loss = config["loss"]
    model = dict(config["model"])
    model["epsilon"] = model.pop("eps")
    game = dict(config["game"])
    game["players"] = game.pop("n_players")
    effective_hand_cadence = training.get("kl_ref_effective_hands_rollout")
    interval_cadence = (
        None if effective_hand_cadence is not None else int(training["kl_ref_interval"])
    )
    optimizer_state = _convert_optimizer(legacy["optimizer"], training)
    scientific = {
        "mode": {
            "training_regime": "self_play",
            "rollout_mode": "sequential",
            "private_estimator": "sampled",
        },
        "game": game,
        "model": {
            key: value
            for key, value in model.items()
            if key not in ("k_act", "use_branching_heads", "max_history_tokens")
        },
        "loss": {
            "clip_epsilon": float(legacy_loss["clip_eps"]),
            "value_clip_epsilon": float(legacy_loss["clip_eps_v"]),
            "value_coefficient": float(legacy_loss["c_v"]),
            "entropy_coefficient": float(legacy_loss["c_h"]),
            "reference_kl_coefficient": float(legacy_loss["c_kl"]),
            "value_loss": str(legacy_loss["value_loss"]),
            "entropy_schedule": (
                "constant"
                if legacy_loss["entropy_schedule"] == "const"
                else str(legacy_loss["entropy_schedule"])
            ),
            "entropy_decay_iterations": legacy_loss["entropy_decay_iters"],
        },
        "optimizer": dict(optimizer_state["config"]),
        "schedule": {
            "batch_size": int(training["minibatch_size"]),
            "epochs": int(training.get("ppo_epochs") or 1),
            "gradient_steps": (
                None
                if training.get("gd_steps_per_iter") is None
                else int(training["gd_steps_per_iter"])
            ),
        },
        "precision": {"name": "fp16" if bool(training["amp"]) else "fp32"},
        "reference": {
            "interval_iterations": interval_cadence,
            "interval_effective_hands": (
                None
                if effective_hand_cadence is None
                else float(effective_hand_cadence)
            ),
            "cache_outputs": True,
        },
        "lambda": float(legacy_loss["lam"]),
        "root_count": int(legacy["adaptive_work_state"]["current_roots_per_gpu"]),
        "root_count_minimum": int(
            legacy["adaptive_work_state"]["minimum_roots_per_gpu"]
        ),
        "seed": int(training["seed"]),
        "reproducibility": str(training["reproducibility"]),
        "utility": (
            "stack_normalized"
            if legacy_loss["utility_normalization"] == "stack"
            else "raw_bb"
        ),
    }
    # The converter itself must produce the strict public scientific schema,
    # not merely a mapping sufficient to reconstruct model parameters.
    scientific = Phase5Config.from_mapping(scientific).scientific_mapping()
    scientific["mode"] = {
        "training_regime": regime,
        "rollout_mode": "sequential",
        "private_estimator": estimator,
    }
    if estimator == "rao_blackwell":
        scientific["completion"] = {
            "n_samples": int(completion_samples),
            "strata": config["completion"].get("strata", "default"),
        }
    if regime == "best_response":
        probe = config["probe"]
        configured_target = probe.get("target_sha256") or probe.get("checkpoint_sha256")
        target_metadata = legacy["probe_target"]
        metadata_target = (
            target_metadata.get("sha256")
            if isinstance(target_metadata, Mapping)
            else target_metadata
        )
        candidates = tuple(
            value for value in (configured_target, metadata_target) if value is not None
        )
        if (
            not candidates
            or any(
                not isinstance(value, str) or len(value) != 64 for value in candidates
            )
            or len(set(candidates)) != 1
        ):
            raise PersistenceError("probe target must have a content hash")
        target_hash = candidates[0]
        if (
            isinstance(target_metadata, Mapping)
            and target_metadata.get("semantics", "shared_seat_approximate_response_v1")
            != "shared_seat_approximate_response_v1"
        ):
            raise PersistenceError("probe target uses unknown response semantics")
        initialization_metadata = legacy["initialization"]
        initial_hash = (
            initialization_metadata.get("initial_model_sha256")
            if isinstance(initialization_metadata, Mapping)
            else None
        )
        if not isinstance(initial_hash, str) or len(initial_hash) != 64:
            raise PersistenceError(
                "probe initialization must preserve initial_model_sha256"
            )
        initialization = str(probe.get("initialization", "random"))
        metadata_kind = (
            initialization_metadata.get("kind")
            if isinstance(initialization_metadata, Mapping)
            else None
        )
        expected_kind = "scratch" if initialization == "random" else "target"
        if metadata_kind not in (None, expected_kind):
            raise PersistenceError(
                "probe initialization metadata disagrees with configuration"
            )
        scientific["frozen_profile"] = {"sha256": target_hash}
        scientific["response_initialization"] = initialization
        scientific["probe"] = {
            "target_sha256": target_hash,
            "initial_model_sha256": initial_hash,
        }
    scientific = SequentialModeConfig.from_mapping(scientific).scientific_mapping()
    execution = {
        "hardware_profile": dict(legacy["hardware_profile"]),
        "checkpoint_cadence": config["training"].get("checkpoint_every"),
    }
    scaler = (
        _convert_scaler(legacy["scaler"])
        if bool(training["amp"])
        else LossScaler(enabled=False).state_dict()
    )
    metrics_prefix: list[dict[str, object]] = []
    metrics_path = source.parent / "metrics.jsonl"
    if metrics_path.is_file():
        with open(metrics_path, encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                item_iteration = int(item["iter"])
                if item_iteration >= int(legacy["iteration"]):
                    break
                metrics_prefix.append(
                    {
                        "iteration": item_iteration,
                        "statistical_presentations": int(
                            item.get("statistical_rows", 0)
                        ),
                        "minibatches": int(item.get("minibatches", 0)),
                        "optimizer_steps": int(
                            item.get("optimizer_steps_applied_count", 0)
                        ),
                    }
                )
    run_state = RunState(
        run_id=legacy["run_id"],
        iteration=int(legacy["iteration"]),
        root_count=int(legacy["adaptive_work_state"]["current_roots_per_gpu"]),
        logical_work=dict(legacy["logical_work"]),
        metrics_prefix=metrics_prefix,
    )
    converted = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "scientific_config": scientific,
        "scientific_sha256": _canonical_hash(scientific),
        "execution_config": execution,
        "model": _clone_cpu(legacy["model"]),
        "optimizer": optimizer_state,
        "scaler": scaler,
        "reference_model": _clone_cpu(legacy["reference_model"]),
        "reference_state": {
            "reference_iteration": int(legacy["reference_iteration"]),
            "effective_hands_rollout": float(
                legacy["reference_effective_hands_rollout"]
            ),
            "interval_iterations": interval_cadence,
            "interval_effective_hands": (
                None
                if effective_hand_cadence is None
                else float(effective_hand_cadence)
            ),
        },
        "run_state": run_state.state_dict(),
        "explicit_rng": legacy["torch_generator"].clone(),
        "ambient_cpu_rng": legacy["torch_cpu_rng"].clone(),
        "ambient_cuda_rng": tuple(value.clone() for value in legacy["torch_cuda_rng"]),
        "conversion": {
            "schema": CONVERSION_SCHEMA,
            "source_sha256": source_hash,
            "source_schema": source_schema,
            "mode": f"{regime}/sequential/{estimator}",
        },
    }
    # Conversion provenance is recorded beside the strict runtime payload,
    # never accepted as a runtime field.
    runtime = dict(converted)
    provenance = runtime.pop("conversion")
    _atomic_torch_save(runtime, destination)
    result = {
        **provenance,
        "result_sha256": sha256_file(destination),
        "destination": str(destination),
    }
    metadata = destination.with_suffix(destination.suffix + ".conversion.json")
    _atomic_json_save(result, metadata)
    external = {}
    for name in ("run.json", "status.json", "metrics.jsonl"):
        candidate = source.parent / name
        if candidate.is_file():
            external[name] = {
                "sha256": sha256_file(candidate),
                "bytes": candidate.stat().st_size,
            }
    if external:
        artifact = {
            "schema": "pbg-queued-run-artifacts-v1",
            "checkpoint_iteration": int(legacy["iteration"]),
            "checkpoint_run_id": str(legacy["run_id"]),
            "source_files": external,
            "converted_metrics_rows": len(metrics_prefix),
        }
        artifact_path = destination.with_suffix(
            destination.suffix + ".run-artifacts.json"
        )
        _atomic_json_save(artifact, artifact_path)
    return result


def convert_legacy_sequential_sampled(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Phase 5 compatibility spelling for the sole offline converter."""

    return convert_legacy_sequential(
        source_path,
        destination_path,
        expected_training_regime="self_play",
        expected_estimator="sampled",
    )


def trim_metrics_after_checkpoint(
    metrics: Sequence[Mapping[str, object]], iteration: int
) -> list[dict[str, object]]:
    """Back up externally if desired, then retain the exact completed prefix."""

    retained = [dict(item) for item in metrics if int(item["iteration"]) <= iteration]
    values = [int(item["iteration"]) for item in retained]
    if values and values != list(range(values[0], values[-1] + 1)):
        raise PersistenceError("metrics before the checkpoint are not gap-free")
    return retained


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONVERSION_SCHEMA",
    "LEGACY_SCHEMA",
    "LEGACY_SCHEMA_V7",
    "QUEUED_CHECKPOINT_V1",
    "PersistenceError",
    "RunState",
    "convert_legacy_sequential",
    "convert_legacy_sequential_sampled",
    "convert_queued_checkpoint_v1",
    "load_checkpoint",
    "save_checkpoint",
    "sha256_file",
    "trim_metrics_after_checkpoint",
]
