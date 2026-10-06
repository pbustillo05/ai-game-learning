"""CUDA runner for the strict paired variance/iteration experiment.

Every command writes one atomic, independently resumable job artifact.  The
runner intentionally composes the public queued engine and only instruments
its already-created sampler and optimizer boundaries; rollout, completion,
target construction, caching, loss, and update implementations remain the
production implementations.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from experiments.variance_iteration_48h.protocol import (
    REPOSITORY,
    SCHEMA,
    canonical_sha256,
    file_sha256,
    logical_seed,
    protocol_copy,
)
from experiments.variance_iteration_48h.reference import (
    direct_reference_moments,
    signed_batch_losses,
)
from pbg.config import RootConfig, load_config_mapping
from pbg.core.learner import ReferenceState, SequentialLearnerBatch
from pbg.core.optimization import LossScaler
from pbg.core.persistence import model_state_sha256
from pbg.runtime import TrainingSession


class ExperimentError(RuntimeError):
    """A predeclared experiment invariant was violated."""


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_json(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_yaml(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(dict(value), sort_keys=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_torch_save(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        raise ExperimentError(f"experiment manifest does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ExperimentError("experiment manifest schema differs")
    return value


def initialize(run_dir: str | Path) -> dict[str, Any]:
    destination = Path(run_dir)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        return load_manifest(destination)
    protocol = protocol_copy()
    checkpoint_identity: dict[str, object] = {}
    for label, item in protocol["checkpoints"].items():
        path = _repository_path(item["path"])
        if not path.exists():
            raise ExperimentError(f"missing {label} checkpoint: {path}")
        checkpoint_identity[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "config": str(_repository_path(item["config"])),
            "config_sha256": file_sha256(_repository_path(item["config"])),
        }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY, text=True
    )
    manifest: dict[str, Any] = {
        **protocol,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(destination.resolve()),
        "checkpoint_identity": checkpoint_identity,
        "source": {
            "git_revision": revision,
            "git_status_at_init": status.splitlines(),
            "runner_sha256": file_sha256(Path(__file__)),
        },
        "scope": {
            "locked": False,
            "gradient_repetitions": protocol["gradient"]["repetitions"],
            "optimization_repetitions": protocol["optimization"]["repetitions"],
            "maximum_steps": protocol["optimization"]["maximum_steps"],
            "endpoints": protocol["optimization"]["endpoints"],
            "selection_rule": (
                "Before inspecting effects: cap all optimization at 38 steps if "
                "forecast exceeds hour 40; then reduce gradient repetitions to 16"
            ),
        },
    }
    manifest["protocol_sha256"] = canonical_sha256(protocol)
    for name in (
        "gradient",
        "optimization",
        "reference",
        "reference_precision",
        "checks",
        "logs",
        "configs",
        "figures",
        "analysis",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest)
    return manifest


def lock_scope(
    run_dir: str | Path,
    *,
    gradient_pair_seconds: float,
    optimization_pair_seconds: float,
    available_gpus: int,
) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    scope = dict(manifest["scope"])
    if scope.get("locked"):
        return manifest
    if min(gradient_pair_seconds, optimization_pair_seconds, available_gpus) <= 0:
        raise ValueError("pilot timings and GPU count must be positive")
    gradient_repetitions = int(manifest["gradient"]["repetitions"])
    optimization_repetitions = int(manifest["optimization"]["repetitions"])
    checkpoint_count = len(manifest["checkpoints"])
    forecast_seconds = (
        checkpoint_count * gradient_repetitions * gradient_pair_seconds
        + checkpoint_count * optimization_repetitions * optimization_pair_seconds
    ) / available_gpus
    maximum_steps = int(manifest["optimization"]["maximum_steps"])
    decision = "default"
    if forecast_seconds > 40 * 3600:
        maximum_steps = 38
        optimization_pair_seconds *= 38.0 / 100.0
        forecast_seconds = (
            checkpoint_count * gradient_repetitions * gradient_pair_seconds
            + checkpoint_count * optimization_repetitions * optimization_pair_seconds
        ) / available_gpus
        decision = "optimization_capped_at_38"
    if forecast_seconds > 40 * 3600:
        gradient_repetitions = 16
        forecast_seconds = (
            checkpoint_count * gradient_repetitions * gradient_pair_seconds
            + checkpoint_count * optimization_repetitions * optimization_pair_seconds
        ) / available_gpus
        decision = "optimization_capped_at_38_and_gradient_repetitions_16"
    endpoints = [0, 10, 20, 38] + ([100] if maximum_steps >= 100 else [])
    scope.update(
        {
            "locked": True,
            "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gradient_repetitions": gradient_repetitions,
            "optimization_repetitions": optimization_repetitions,
            "maximum_steps": maximum_steps,
            "endpoints": endpoints,
            "available_gpus": available_gpus,
            "pilot_gradient_pair_seconds": gradient_pair_seconds,
            "pilot_optimization_pair_seconds": optimization_pair_seconds,
            "forecast_wall_seconds_excluding_reference_and_analysis": forecast_seconds,
            "decision": decision,
        }
    )
    manifest["scope"] = scope
    atomic_json(Path(run_dir) / "manifest.json", manifest)
    return manifest


def _checkpoint_spec(manifest: Mapping[str, Any], checkpoint: str) -> Mapping[str, Any]:
    try:
        return manifest["checkpoint_identity"][checkpoint]
    except KeyError as exc:
        raise ExperimentError(f"unknown checkpoint {checkpoint!r}") from exc


def _load_source(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentError("source configuration is not a mapping")
    return value


def build_config_source(
    manifest: Mapping[str, Any],
    *,
    checkpoint: str,
    arm: str,
    purpose: str,
    device: str,
    seed: int,
    roots: int,
    steps: int,
    completion_samples: int = 1024,
) -> dict[str, Any]:
    """Derive one exact arm from the checkpoint's original run config."""

    if arm not in ("sampled", "rb1024", "reference"):
        raise ValueError("unknown experiment arm")
    if purpose not in ("gradient", "optimization", "reference", "precision"):
        raise ValueError("unknown experiment purpose")
    spec = _checkpoint_spec(manifest, checkpoint)
    source = copy.deepcopy(_load_source(spec["config"]))
    scientific = source["scientific"]
    run = source["run"]
    estimator = "sampled" if arm == "sampled" else "rao_blackwell"
    scientific["mode"]["private_estimator"] = estimator
    if estimator == "rao_blackwell":
        scientific["completion"] = {"samples": int(completion_samples)}
    else:
        scientific.pop("completion", None)
    scientific["schedule"] = {"batch_size": 256, "steps": int(steps)}
    if purpose in ("gradient", "reference", "precision"):
        scientific["objective"] = {
            "policy_clip": (
                float(manifest["gradient"]["actor_policy_clip"])
                if purpose == "gradient"
                else 0.2
            ),
            "value": {"kind": "mse", "coefficient": 0.0},
            "entropy": {"kind": "constant", "coefficient": 0.0},
        }
        run["execution"]["reference_cache"] = "off"
    if purpose == "gradient":
        scientific["lambda"] = 1.0
    run["seed"] = int(seed)
    run["reproducibility"] = "strict"
    run["workload"]["roots"] = int(roots)
    run["recovery"]["minimum_roots"] = int(roots)
    execution = run["execution"]
    execution["device"] = device
    execution["distributed"] = False
    # Smaller completion packs keep RB-4096/8192 reference jobs below the
    # immutable 11 GiB admission bound; they do not alter logical worlds.
    if purpose in ("reference", "precision"):
        execution["microbatch"]["completion_rows"] = min(
            int(execution["microbatch"]["completion_rows"]), 32
        )
    run["iterations"] = 1
    run["checkpoint_every"] = 1
    run["output_dir"] = (
        f"runs/experiments/variance_iteration_internal/{purpose}_{checkpoint}_{arm}"
    )
    return source


def _load_payload(manifest: Mapping[str, Any], checkpoint: str) -> dict[str, Any]:
    spec = _checkpoint_spec(manifest, checkpoint)
    path = Path(spec["path"])
    if file_sha256(path) != spec["sha256"]:
        raise ExperimentError(f"{checkpoint} checkpoint content changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, dict)
        or value.get("checkpoint_schema") != "pbg-queued-checkpoint-v3"
    ):
        raise ExperimentError("checkpoint is not a queued v3 checkpoint")
    return value


def _restore_session(
    cfg: RootConfig,
    payload: Mapping[str, Any],
    *,
    benchmark_telemetry: bool = True,
) -> TrainingSession:
    session = TrainingSession(
        cfg,
        initial_model_state=payload["model"],
        benchmark_telemetry=benchmark_telemetry,
    )
    optimizer = session.optimizers.optimizers[0]
    optimizer.load_state_dict(payload["optimizer"])
    optimizer.scaler = LossScaler.from_state_dict(payload["scaler"])
    if session.reference_modules is not None:
        reference_model = payload.get("reference_model")
        reference_state = payload.get("reference_state")
        if not isinstance(reference_model, Mapping) or not isinstance(
            reference_state, Mapping
        ):
            raise ExperimentError("checkpoint omitted required reference state")
        session.reference_modules.lane_module(0).load_state_dict(
            reference_model, strict=True
        )
        session.reference_states = (ReferenceState.from_state_dict(reference_state),)
    # We start at the saved scientific iteration while intentionally replacing
    # its workload with this experiment's fresh, independently seeded batch.
    session.run_states[0].iteration = int(payload["run_state"]["iteration"])
    session.run_states[0].root_count = cfg.scientific.common.root_count
    session.run_states[0].metrics_prefix.clear()
    session.run_states[0].logical_work = {
        "trajectories": 0,
        "completion_samples": 0,
        "minibatches": 0,
        "optimizer_steps": 0,
    }
    session.run_states[0].retry_epoch = 0
    session.run_states[0].attempt_logical_base = 0
    return session


def _update_hash(digest: hashlib._Hash, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _public_contract_hash(stores: Mapping[str, object]) -> tuple[str, dict[str, int]]:
    digest = hashlib.sha256()
    rows: dict[str, int] = {}
    for name in sorted(stores):
        store = stores[name]
        count = int(store.rows)
        rows[name] = count
        digest.update(name.encode())
        digest.update(str(count).encode())
        columns = store.sealed_inputs.columns
        _update_hash(digest, f"{name}/model_input", columns["model_input"])
        _update_hash(digest, f"{name}/owner", columns["owner"])
        action = store.slabs["ActionHead"].columns["action"][store.lane, :count]
        _update_hash(digest, f"{name}/action", action)
    return digest.hexdigest(), rows


class _RecordingSampler:
    def __init__(self, sampler: object, destination: dict[str, object]) -> None:
        self.sampler = sampler
        self.destination = destination

    def __getattr__(self, name: str) -> object:
        return getattr(self.sampler, name)

    def batches(self) -> Iterator[object]:
        digest = hashlib.sha256()
        rows = 0
        logical_steps: set[int] = set()
        for batch in self.sampler.batches():
            digest.update(batch.input_type.encode())
            digest.update(int(batch.logical_step).to_bytes(8, "little", signed=False))
            _update_hash(digest, "indices", batch.indices)
            rows += int(batch.rows)
            logical_steps.add(int(batch.logical_step))
            yield batch
        self.destination["minibatch_schedule_sha256"] = digest.hexdigest()
        self.destination["decision_presentations"] = rows
        self.destination["logical_steps"] = len(logical_steps)


def _instrument_sampler(
    session: TrainingSession,
    *,
    minibatch_seed: int,
    capture: dict[str, object],
) -> None:
    original_factory = session._iteration_engine

    def factory():
        engine = original_factory()
        original_prepare = engine._prepare_learner

        def prepare(owners, state, lane):
            prepared = original_prepare(owners, state, lane)
            if lane != 0:
                raise ExperimentError("48-hour experiment requires one replica lane")
            prepared.sampler.generator.manual_seed(int(minibatch_seed))
            public_hash, partition_rows = _public_contract_hash(prepared.sampler.stores)
            capture["public_contract_sha256"] = public_hash
            capture["partition_rows"] = partition_rows
            capture["prepare_calls"] = int(capture.get("prepare_calls", 0)) + 1
            return dataclasses.replace(
                prepared,
                sampler=_RecordingSampler(prepared.sampler, capture),
            )

        engine._prepare_learner = prepare
        return engine

    session._iteration_engine = factory


def _model_state_cpu(module: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _last_metrics(session: TrainingSession) -> dict[str, object]:
    if not session.run_states[0].metrics_prefix:
        raise ExperimentError("session produced no metrics row")
    return copy.deepcopy(session.run_states[0].metrics_prefix[-1])


def _install_gradient_capture(optimizer: object, destination: list[Tensor]) -> None:
    original_step = optimizer.step

    def capture_step():
        if destination:
            raise ExperimentError("actor-gradient job reached multiple steps")
        value = optimizer.gradients.flat.detach().clone()
        scale = float(optimizer.scaler.scale)
        if optimizer.scaler.enabled:
            value.div_(scale)
        destination.append(value.cpu())
        return original_step()

    optimizer.step = capture_step


def _install_endpoint_capture(
    optimizer: object,
    module: torch.nn.Module,
    session: TrainingSession,
    endpoints: Sequence[int],
    candidates: dict[int, dict[str, Tensor]],
    endpoint_seconds: dict[int, float],
    timing: dict[str, float],
) -> dict[str, int]:
    original_step = optimizer.step
    state = {"applied": 0}

    def capture_step():
        result = original_step()
        if result.applied:
            state["applied"] += 1
        applied = state["applied"]
        if applied in endpoints and applied not in candidates:
            torch.cuda.synchronize(session.device)
            candidates[applied] = _model_state_cpu(module)
            endpoint_seconds[applied] = time.perf_counter() - timing["started"]
        return result

    optimizer.step = capture_step
    return state


def _job_config(
    run_dir: Path,
    name: str,
    source: Mapping[str, object],
) -> RootConfig:
    path = run_dir / "configs" / f"{name}.yaml"
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != source:
            raise ExperimentError(f"existing job config differs: {path}")
    else:
        atomic_yaml(path, source)
    return load_config_mapping(source)


def _cleanup_session(session: TrainingSession | None) -> None:
    if session is not None:
        session.close()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def run_gradient_pair(
    run_dir: str | Path,
    checkpoint: str,
    repetition: int,
    device: str,
) -> dict[str, Any]:
    directory = Path(run_dir)
    manifest = load_manifest(directory)
    repetitions = int(manifest["scope"]["gradient_repetitions"])
    if not 0 <= repetition < repetitions:
        raise ValueError("gradient repetition is outside the locked scope")
    pair_dir = directory / "gradient" / checkpoint / f"rep_{repetition:02d}"
    pair_path = pair_dir / "pair.json"
    if pair_path.exists():
        return json.loads(pair_path.read_text(encoding="utf-8"))
    payload = _load_payload(manifest, checkpoint)
    base_seed = int(manifest["base_seed"])
    rollout_seed = logical_seed(base_seed, "gradient-rollout", checkpoint, repetition)
    decision_seed = logical_seed(
        base_seed, "gradient-decisions", checkpoint, repetition
    )
    rows: dict[str, dict[str, object]] = {}
    for arm in manifest["arms"]:
        output = pair_dir / f"{arm}.pt"
        metadata_path = pair_dir / f"{arm}.json"
        if output.exists() and metadata_path.exists():
            rows[arm] = json.loads(metadata_path.read_text(encoding="utf-8"))
            continue
        source = build_config_source(
            manifest,
            checkpoint=checkpoint,
            arm=arm,
            purpose="gradient",
            device=device,
            seed=rollout_seed,
            roots=int(manifest["gradient"]["public_rollouts"]),
            steps=1,
            completion_samples=int(manifest["gradient"]["rb_completions"]),
        )
        cfg = _job_config(
            directory,
            f"gradient_{checkpoint}_rep{repetition:02d}_{arm}",
            source,
        )
        session: TrainingSession | None = None
        capture: dict[str, object] = {}
        try:
            session = _restore_session(cfg, payload)
            _instrument_sampler(session, minibatch_seed=decision_seed, capture=capture)
            optimizer = session.optimizers.optimizers[0]
            gradients: list[Tensor] = []
            _install_gradient_capture(optimizer, gradients)
            torch.cuda.reset_peak_memory_stats(session.device)
            torch.cuda.synchronize(session.device)
            started = time.perf_counter()
            session.step()
            torch.cuda.synchronize(session.device)
            seconds = time.perf_counter() - started
            if len(gradients) != 1:
                raise ExperimentError(
                    "gradient capture did not observe exactly one step"
                )
            gradient = gradients[0]
            metrics = _last_metrics(session)
            if float(metrics["objective"]["clip_fraction"]) != 0.0:
                raise ExperimentError(
                    "nominally unclipped actor gradient clipped a row"
                )
            metadata = {
                "schema": SCHEMA,
                "experiment": "actor_gradient",
                "checkpoint": checkpoint,
                "checkpoint_sha256": manifest["checkpoint_identity"][checkpoint][
                    "sha256"
                ],
                "repetition": repetition,
                "arm": arm,
                "device": str(session.device),
                "rollout_seed": rollout_seed,
                "decision_seed": decision_seed,
                "initial_model_sha256": model_state_sha256(payload["model"]),
                "gradient_elements": gradient.numel(),
                "gradient_l2": float(torch.linalg.vector_norm(gradient.double())),
                "gradient_sum": float(gradient.double().sum()),
                "seconds_including_targets": seconds,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(session.device)
                ),
                "loss_contract": {
                    "lambda": 1.0,
                    "actor": "unclipped importance-ratio objective",
                    "policy_clip_guard": float(
                        manifest["gradient"]["actor_policy_clip"]
                    ),
                    "value_coefficient": 0.0,
                    "entropy_coefficient": 0.0,
                    "reference_kl_coefficient": 0.0,
                    "baseline": "detached fixed old-policy value target",
                    "normalization": "mean over one pooled 256-decision logical batch",
                    "advantage_scaling": "stack_normalized utility; no per-arm normalization",
                    "precision": cfg.scientific.common.precision.name,
                    "captured": "before clipping/update; explicitly divided by loss scale",
                },
                "capture": capture,
                "metrics": metrics,
            }
            atomic_torch_save(output, {"metadata": metadata, "gradient": gradient})
            metadata["artifact_sha256"] = file_sha256(output)
            atomic_json(metadata_path, metadata)
            rows[arm] = metadata
        finally:
            _cleanup_session(session)
    sampled = rows["sampled"]
    rb = rows["rb1024"]
    for field in (
        "initial_model_sha256",
        "rollout_seed",
        "decision_seed",
    ):
        if sampled[field] != rb[field]:
            raise ExperimentError(f"gradient pair differs in {field}")
    for field in ("public_contract_sha256", "minibatch_schedule_sha256"):
        if sampled["capture"][field] != rb["capture"][field]:
            raise ExperimentError(f"gradient pair differs in {field}")
    pair = {
        "schema": SCHEMA,
        "experiment": "actor_gradient_pair",
        "checkpoint": checkpoint,
        "repetition": repetition,
        "status": "complete",
        "arms": rows,
        "pair_seconds": sum(
            float(row["seconds_including_targets"]) for row in rows.values()
        ),
        "verified_same_public_data": True,
        "verified_same_decision_schedule": True,
    }
    atomic_json(pair_path, pair)
    return pair


def run_optimization_pair(
    run_dir: str | Path,
    checkpoint: str,
    repetition: int,
    device: str,
) -> dict[str, Any]:
    directory = Path(run_dir)
    manifest = load_manifest(directory)
    # Repetition zero is the pre-lock representative paired timing pilot.  It
    # uses the full default geometry and is retained only if the subsequent
    # timing-only scope rule keeps it in the locked queue.
    if not manifest["scope"]["locked"] and repetition != 0:
        raise ExperimentError(
            "only repetition zero may run as the pre-lock optimization pilot"
        )
    repetitions = int(manifest["scope"]["optimization_repetitions"])
    if not 0 <= repetition < repetitions:
        raise ValueError("optimization repetition is outside the locked scope")
    maximum_steps = int(manifest["scope"]["maximum_steps"])
    endpoints = tuple(int(value) for value in manifest["scope"]["endpoints"])
    pair_dir = directory / "optimization" / checkpoint / f"rep_{repetition:02d}"
    pair_path = pair_dir / "pair.json"
    if pair_path.exists():
        return json.loads(pair_path.read_text(encoding="utf-8"))
    payload = _load_payload(manifest, checkpoint)
    base_seed = int(manifest["base_seed"])
    rollout_seed = logical_seed(
        base_seed, "optimization-rollout", checkpoint, repetition
    )
    minibatch_seed = logical_seed(
        base_seed, "optimization-minibatches", checkpoint, repetition
    )
    rows: dict[str, dict[str, object]] = {}
    for arm in manifest["arms"]:
        output = pair_dir / f"{arm}_candidates.pt"
        metadata_path = pair_dir / f"{arm}.json"
        if output.exists() and metadata_path.exists():
            rows[arm] = json.loads(metadata_path.read_text(encoding="utf-8"))
            continue
        source = build_config_source(
            manifest,
            checkpoint=checkpoint,
            arm=arm,
            purpose="optimization",
            device=device,
            seed=rollout_seed,
            roots=int(manifest["optimization"]["public_rollouts"]),
            steps=maximum_steps,
            completion_samples=int(manifest["optimization"]["rb_completions"]),
        )
        cfg = _job_config(
            directory,
            f"optimization_{checkpoint}_rep{repetition:02d}_{arm}",
            source,
        )
        session: TrainingSession | None = None
        capture: dict[str, object] = {}
        try:
            session = _restore_session(cfg, payload)
            _instrument_sampler(session, minibatch_seed=minibatch_seed, capture=capture)
            module = session.modules.lane_module(0)
            optimizer = session.optimizers.optimizers[0]
            candidates: dict[int, dict[str, Tensor]] = {0: _model_state_cpu(module)}
            endpoint_seconds: dict[int, float] = {0: 0.0}
            timing = {"started": 0.0}
            step_state = _install_endpoint_capture(
                optimizer,
                module,
                session,
                endpoints,
                candidates,
                endpoint_seconds,
                timing,
            )
            torch.cuda.reset_peak_memory_stats(session.device)
            torch.cuda.synchronize(session.device)
            timing["started"] = time.perf_counter()
            session.step()
            torch.cuda.synchronize(session.device)
            seconds = time.perf_counter() - timing["started"]
            missing = set(endpoints).difference(candidates)
            if missing:
                raise ExperimentError(
                    f"optimization omitted endpoints {sorted(missing)}"
                )
            metrics = _last_metrics(session)
            hashes = {
                str(step): model_state_sha256(state)
                for step, state in candidates.items()
            }
            metadata = {
                "schema": SCHEMA,
                "experiment": "one_iteration_optimization",
                "checkpoint": checkpoint,
                "checkpoint_sha256": manifest["checkpoint_identity"][checkpoint][
                    "sha256"
                ],
                "repetition": repetition,
                "arm": arm,
                "device": str(session.device),
                "rollout_seed": rollout_seed,
                "minibatch_seed": minibatch_seed,
                "starting_model_sha256": hashes["0"],
                "starting_optimizer_step": int(payload["optimizer"]["step"]),
                "starting_optimizer_state_sha256": canonical_sha256(
                    {
                        "step": int(payload["optimizer"]["step"]),
                        "config": payload["optimizer"]["config"],
                        "exp_avg_sum": float(
                            payload["optimizer"]["exp_avg"].double().sum()
                        ),
                        "exp_avg_sq_sum": float(
                            payload["optimizer"]["exp_avg_sq"].double().sum()
                        ),
                    }
                ),
                "reference_model_sha256": model_state_sha256(
                    payload["reference_model"]
                ),
                "scaler_state": payload["scaler"],
                "maximum_steps": maximum_steps,
                "applied_steps": step_state["applied"],
                "endpoints": list(endpoints),
                "endpoint_seconds_including_targets": {
                    str(key): value for key, value in endpoint_seconds.items()
                },
                "candidate_sha256": hashes,
                "seconds_including_targets": seconds,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(session.device)
                ),
                "capture": capture,
                "metrics": metrics,
                "target_cache_contract": (
                    "one target construction in one prepared learner; targets reused "
                    "for every fixed-step minibatch"
                ),
            }
            atomic_torch_save(
                output,
                {
                    "metadata": metadata,
                    "candidates": {str(k): v for k, v in candidates.items()},
                },
            )
            metadata["artifact_sha256"] = file_sha256(output)
            atomic_json(metadata_path, metadata)
            rows[arm] = metadata
        finally:
            _cleanup_session(session)
    sampled = rows["sampled"]
    rb = rows["rb1024"]
    for field in (
        "starting_model_sha256",
        "starting_optimizer_step",
        "starting_optimizer_state_sha256",
        "reference_model_sha256",
        "scaler_state",
        "rollout_seed",
        "minibatch_seed",
        "maximum_steps",
        "endpoints",
    ):
        if sampled[field] != rb[field]:
            raise ExperimentError(f"optimization pair differs in {field}")
    for field in ("public_contract_sha256", "minibatch_schedule_sha256"):
        if sampled["capture"][field] != rb["capture"][field]:
            raise ExperimentError(f"optimization pair differs in {field}")
    if sampled["capture"]["prepare_calls"] != 1 or rb["capture"]["prepare_calls"] != 1:
        raise ExperimentError("optimization target/preparation was not cached once")
    pair = {
        "schema": SCHEMA,
        "experiment": "one_iteration_optimization_pair",
        "checkpoint": checkpoint,
        "repetition": repetition,
        "status": "complete",
        "arms": rows,
        "pair_seconds": sum(
            float(row["seconds_including_targets"]) for row in rows.values()
        ),
        "verified_same_public_data": True,
        "verified_same_minibatch_schedule": True,
        "verified_same_starting_state": True,
        "verified_one_target_construction_per_arm": True,
    }
    atomic_json(pair_path, pair)
    return pair


def _reference_batch_loss(
    learner: object,
    batch: SequentialLearnerBatch,
    signed: Mapping[str, Mapping[str, Tensor]],
    *,
    clip_epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Return cached/direct held-out loss rows without constructing gradients."""

    sigma = batch.sigma_fields
    action = batch.action_fields
    counts = batch.action_ptr[1:] - batch.action_ptr[:-1]
    action_to_sigma = torch.repeat_interleave(
        torch.arange(batch.rows, device=batch.sigma_index.device),
        counts,
        output_size=batch.action_rows,
    )
    sigma_private = sigma["owner"][:, None]
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        learner.precision
    ]
    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda", dtype=dtype, enabled=learner.precision != "fp32"
        ),
    ):
        evaluation = learner.model_operation(
            batch.input_type,
            sigma["model_input"],
            sigma_private,
            sigma["owner"],
        )
        repeated_raw = {
            name: value.index_select(0, action_to_sigma)
            for name, value in evaluation.policy_raw.items()
        }
        repeated_owner = sigma["owner"].index_select(0, action_to_sigma)
        policy = learner.policy_operation(
            batch.input_type,
            sigma["model_input"].index_select(0, action_to_sigma),
            repeated_raw,
            action["action"].to(torch.int32),
            repeated_owner[:, None],
            repeated_owner.to(torch.int64),
            None,
        )
        current_log = policy.selected_log_policy.float()
    old_log = action["old_log_policy"].float()
    indices = batch.indices.to(torch.int64)
    source = signed[batch.input_type]
    moments = {name: value.index_select(0, indices) for name, value in source.items()}
    if not (
        current_log.shape
        == old_log.shape
        == moments["positive"].shape
        == moments["negative"].shape
        == moments["posterior"].shape
    ):
        raise ExperimentError("held-out signed action fields are not aligned")
    ratio = (current_log - old_log).exp()
    cached_action, direct_action = signed_batch_losses(
        ratio, moments, clip_epsilon=clip_epsilon
    )

    def mean_by_sigma(value: Tensor) -> Tensor:
        result = torch.zeros(batch.rows, dtype=torch.float32, device=value.device)
        result.scatter_add_(0, action_to_sigma, value)
        return result / counts.float()

    return mean_by_sigma(cached_action), mean_by_sigma(direct_action)


def _candidate_inventory(
    run_dir: Path,
    checkpoint: str,
    manifest: Mapping[str, Any],
    *,
    selected_only: bool,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    repetitions = int(manifest["scope"]["optimization_repetitions"])
    endpoints = tuple(int(value) for value in manifest["scope"]["endpoints"])
    if selected_only:
        repetitions = min(repetitions, 1)
        endpoints = tuple(
            value for value in endpoints if value in (0, 38, max(endpoints))
        )
    unique: dict[str, dict[str, object]] = {}
    aliases: dict[str, list[str]] = {}
    for repetition in range(repetitions):
        base = run_dir / "optimization" / checkpoint / f"rep_{repetition:02d}"
        for arm in manifest["arms"]:
            metadata_path = base / f"{arm}.json"
            artifact = base / f"{arm}_candidates.pt"
            if not metadata_path.exists() or not artifact.exists():
                raise ExperimentError(f"missing optimization artifact: {artifact}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if file_sha256(artifact) != metadata["artifact_sha256"]:
                raise ExperimentError(f"candidate artifact changed: {artifact}")
            for endpoint in endpoints:
                candidate_id = f"{checkpoint}/rep{repetition:02d}/{arm}/step{endpoint}"
                digest = metadata["candidate_sha256"][str(endpoint)]
                aliases.setdefault(digest, []).append(candidate_id)
                unique.setdefault(
                    digest,
                    {
                        "sha256": digest,
                        "artifact": str(artifact),
                        "endpoint": str(endpoint),
                    },
                )
    return list(unique.values()), aliases


def _evaluate_candidates(
    module: torch.nn.Module,
    learner: object,
    stores: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    aliases: Mapping[str, Sequence[str]],
    signed: Mapping[str, Mapping[str, Tensor]],
    *,
    physical_cap: int,
    clip_epsilon: float,
) -> tuple[dict[str, float], float, int]:
    results: dict[str, float] = {}
    maximum_error = 0.0
    total_rows_expected = sum(int(store.rows) for store in stores.values())
    loaded_path: str | None = None
    loaded_payload: Mapping[str, object] | None = None
    for candidate in candidates:
        artifact = str(candidate["artifact"])
        if artifact != loaded_path:
            if loaded_payload is not None:
                del loaded_payload
            loaded_payload = torch.load(
                artifact, map_location="cpu", weights_only=False
            )
            loaded_path = artifact
        assert loaded_payload is not None
        state = loaded_payload["candidates"][str(candidate["endpoint"])]
        module.load_state_dict(state, strict=True)
        total = 0.0
        rows = 0
        for name in sorted(stores):
            store = stores[name]
            for begin in range(0, int(store.rows), physical_cap):
                end = min(begin + physical_cap, int(store.rows))
                indices = torch.arange(
                    begin,
                    end,
                    dtype=torch.int64,
                    device=store.device,
                )
                batch = store.gather(indices)
                cached, direct = _reference_batch_loss(
                    learner, batch, signed, clip_epsilon=clip_epsilon
                )
                maximum_error = max(
                    maximum_error,
                    float((cached - direct).abs().max()),
                )
                total += float(cached.double().sum())
                rows += int(cached.numel())
        if rows != total_rows_expected:
            raise ExperimentError("held-out evaluation omitted decision rows")
        value = total / rows
        for candidate_id in aliases[str(candidate["sha256"])]:
            results[candidate_id] = value
    if loaded_payload is not None:
        del loaded_payload
    return results, maximum_error, total_rows_expected


def run_reference(
    run_dir: str | Path,
    checkpoint: str,
    device: str,
    *,
    precision_index: int | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    manifest = load_manifest(directory)
    if not manifest["scope"]["locked"]:
        raise ExperimentError("scope must be locked before held-out evaluation")
    if precision_index is None:
        purpose = "reference"
        roots = int(manifest["reference"]["public_rollouts"])
        completions = int(manifest["reference"]["completions"])
        selected_only = False
        output = directory / "reference" / f"{checkpoint}.json"
        stream_name = "primary"
    else:
        budgets = tuple(
            int(value) for value in manifest["reference"]["precision_completions"]
        )
        if not 0 <= precision_index < len(budgets):
            raise ValueError("precision index is outside configured budgets")
        purpose = "precision"
        roots = int(manifest["reference"]["precision_public_rollouts"])
        completions = budgets[precision_index]
        selected_only = True
        output = (
            directory / "reference_precision" / (f"{checkpoint}_m{completions}.json")
        )
        stream_name = f"precision-{precision_index}"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    payload = _load_payload(manifest, checkpoint)
    candidates, aliases = _candidate_inventory(
        directory, checkpoint, manifest, selected_only=selected_only
    )
    base_seed = int(manifest["base_seed"])
    rollout_seed = logical_seed(base_seed, "reference-public", checkpoint, roots)
    completion_mask = logical_seed(
        base_seed, "reference-completion", checkpoint, stream_name, completions
    )
    board_seed = logical_seed(
        base_seed, "reference-board", checkpoint, stream_name, completions
    )
    source = build_config_source(
        manifest,
        checkpoint=checkpoint,
        arm="reference",
        purpose=purpose,
        device=device,
        seed=rollout_seed,
        roots=roots,
        steps=1,
        # These production RB targets only scaffold aligned held-out rows.  The
        # declared objective below uses its own direct completion worlds.
        completion_samples=min(1024, completions),
    )
    cfg = _job_config(
        directory,
        f"{purpose}_{checkpoint}_n{roots}_m{completions}",
        source,
    )
    session: TrainingSession | None = None
    capture: dict[str, object] = {}
    try:
        session = _restore_session(cfg, payload)
        original_factory = session._iteration_engine
        theta0 = copy.deepcopy(payload["model"])

        def factory():
            engine = original_factory()
            original_completion = engine._execute_completion
            original_prepare = engine._prepare_learner

            def completion(owners, state):
                leaf = owners.slab("Leaf")
                mask = int(completion_mask & ((1 << 63) - 1))
                for lane in range(owners.plan.manifest.n_replica):
                    count = leaf.tail(lane)
                    if count:
                        seeds = leaf.columns["seed"][lane, :count]
                        seeds.bitwise_xor_(mask)
                        seeds.bitwise_and_((1 << 63) - 1)
                capture["completion_seed_mask"] = mask
                result = original_completion(owners, state)
                signed, metadata = direct_reference_moments(
                    owners,
                    state,
                    engine.operations,
                    completions=completions,
                    board_seed=board_seed,
                    lambda_=float(engine.config.lambda_),
                )
                capture["signed_reference"] = signed
                capture["direct_reference_metadata"] = metadata
                return result

            def prepare(owners, state, lane):
                prepared = original_prepare(owners, state, lane)
                stores = prepared.sampler.stores
                public_hash, partition_rows = _public_contract_hash(stores)
                capture["public_contract_sha256"] = public_hash
                capture["partition_rows"] = partition_rows
                physical_cap = min(
                    owners.plan.manifest.learner_sigma_pack_cap,
                    owners.plan.manifest.learner_action_pack_cap,
                )
                evaluated, maximum_error, rows = _evaluate_candidates(
                    session.modules.lane_module(0),
                    prepared.iteration.learner,
                    stores,
                    candidates,
                    aliases,
                    capture["signed_reference"],
                    physical_cap=physical_cap,
                    clip_epsilon=0.2,
                )
                capture["objectives"] = evaluated
                capture["signed_cache_max_abs_error"] = maximum_error
                capture["decision_rows"] = rows
                capture["unique_candidate_count"] = len(candidates)
                capture["candidate_count_with_aliases"] = sum(
                    len(values) for values in aliases.values()
                )
                session.modules.lane_module(0).load_state_dict(theta0, strict=True)
                return prepared

            engine._execute_completion = completion
            engine._prepare_learner = prepare
            return engine

        session._iteration_engine = factory
        torch.cuda.reset_peak_memory_stats(session.device)
        torch.cuda.synchronize(session.device)
        started = time.perf_counter()
        session.step()
        torch.cuda.synchronize(session.device)
        seconds = time.perf_counter() - started
        if "objectives" not in capture:
            raise ExperimentError("held-out evaluator did not reach prepared targets")
        if float(capture["signed_cache_max_abs_error"]) > 2e-5:
            raise ExperimentError("signed-moment cache differs from direct averaging")
        theta_losses = {
            value
            for key, value in capture["objectives"].items()
            if key.endswith("/step0")
        }
        if len(theta_losses) != 1:
            raise ExperimentError("held-out step-zero objective is not unique")
        result = {
            "schema": SCHEMA,
            "experiment": purpose,
            "checkpoint": checkpoint,
            "checkpoint_sha256": manifest["checkpoint_identity"][checkpoint]["sha256"],
            "device": str(session.device),
            "public_rollouts": roots,
            "completion_samples": completions,
            "scaffold_completion_samples": min(1024, completions),
            "rollout_seed": rollout_seed,
            "completion_seed_mask": completion_mask,
            "board_seed": board_seed,
            "theta0_loss": next(iter(theta_losses)),
            "objectives": capture["objectives"],
            "public_contract_sha256": capture["public_contract_sha256"],
            "partition_rows": capture["partition_rows"],
            "decision_rows": capture["decision_rows"],
            "signed_cache_max_abs_error": capture["signed_cache_max_abs_error"],
            "direct_reference_metadata": capture["direct_reference_metadata"],
            "unique_candidate_count": capture["unique_candidate_count"],
            "candidate_count_with_aliases": capture["candidate_count_with_aliases"],
            "seconds": seconds,
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(session.device)
            ),
            "objective_contract": {
                "name": "expected ordinary sampled-target clipped PPO actor loss",
                "policy_clip": 0.2,
                "old_policy": "fixed checkpoint policy",
                "advantage": "fixed checkpoint baseline; production lambda",
                "normalization": "uniform mean over every held-out decision row",
                "conditioning": "actor hand before action ratio",
                "posterior": "collision-conditioned RB completion weights",
                "cache": (
                    "positive/negative conditional moments from direct exact "
                    "joint-profile and uniform-runout completion worlds"
                ),
            },
        }
        atomic_json(output, result)
        return result
    finally:
        _cleanup_session(session)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--run-dir", required=True)
    lock = subparsers.add_parser("lock")
    lock.add_argument("--run-dir", required=True)
    lock.add_argument("--gradient-pair-seconds", required=True, type=float)
    lock.add_argument("--optimization-pair-seconds", required=True, type=float)
    lock.add_argument("--available-gpus", required=True, type=int)
    for name in ("gradient", "optimize"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument(
            "--checkpoint", required=True, choices=("normal10k", "rb10k")
        )
        command.add_argument("--repetition", required=True, type=int)
        command.add_argument("--device", required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--run-dir", required=True)
    reference.add_argument(
        "--checkpoint", required=True, choices=("normal10k", "rb10k")
    )
    reference.add_argument("--device", required=True)
    precision = subparsers.add_parser("precision")
    precision.add_argument("--run-dir", required=True)
    precision.add_argument(
        "--checkpoint", required=True, choices=("normal10k", "rb10k")
    )
    precision.add_argument("--device", required=True)
    precision.add_argument("--index", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "init":
        result = initialize(args.run_dir)
    elif args.command == "lock":
        result = lock_scope(
            args.run_dir,
            gradient_pair_seconds=args.gradient_pair_seconds,
            optimization_pair_seconds=args.optimization_pair_seconds,
            available_gpus=args.available_gpus,
        )
    elif args.command == "gradient":
        result = run_gradient_pair(
            args.run_dir, args.checkpoint, args.repetition, args.device
        )
    elif args.command == "optimize":
        result = run_optimization_pair(
            args.run_dir, args.checkpoint, args.repetition, args.device
        )
    elif args.command == "reference":
        result = run_reference(args.run_dir, args.checkpoint, args.device)
    else:
        result = run_reference(
            args.run_dir,
            args.checkpoint,
            args.device,
            precision_index=args.index,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
