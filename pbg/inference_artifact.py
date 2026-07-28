"""Strict model-only artifacts for queued CUDA/MPS inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from pbg.games.poker.queued.config import BranchedModeConfig, SequentialModeConfig
from pbg.core.persistence import (
    CHECKPOINT_SCHEMA,
    DISTRIBUTED_CHECKPOINT_SCHEMA,
    REPLICA_CHECKPOINT_SCHEMA,
    load_distributed_checkpoint_manifest,
    model_state_sha256,
    sha256_file,
    validate_checkpoint_payload,
)

INFERENCE_ARTIFACT_SCHEMA_VERSION = "pbg-queued-inference-v1"
_FIELDS = {
    "artifact_schema",
    "source_checkpoint_schema",
    "source_checkpoint_sha256",
    "model_sha256",
    "payload_sha256",
    "iteration",
    "scientific_config",
    "model",
}


def _payload_sha256(value: dict[str, Any]) -> str:
    identity = {
        "artifact_schema": value["artifact_schema"],
        "source_checkpoint_schema": value["source_checkpoint_schema"],
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "iteration": value["iteration"],
        "scientific_config": value["scientific_config"],
        "model_sha256": value["model_sha256"],
    }
    return hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _load_artifact(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, dict)
        or set(value) != _FIELDS
        or value.get("artifact_schema") != INFERENCE_ARTIFACT_SCHEMA_VERSION
        or value.get("source_checkpoint_schema")
        not in (
            CHECKPOINT_SCHEMA,
            DISTRIBUTED_CHECKPOINT_SCHEMA,
            REPLICA_CHECKPOINT_SCHEMA,
        )
    ):
        raise RuntimeError("runtime accepts only strict queued inference artifacts")
    scientific = value["scientific_config"]
    if not isinstance(scientific, dict) or not isinstance(
        scientific.get("mode"), dict
    ):
        raise RuntimeError("inference artifact scientific configuration is invalid")
    try:
        parsed = (
            SequentialModeConfig.from_mapping(scientific)
            if scientific["mode"].get("rollout_mode") == "sequential"
            else BranchedModeConfig.from_mapping(scientific)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"inference artifact scientific configuration is invalid: {exc}"
        ) from exc
    if parsed.scientific_mapping() != scientific:
        raise RuntimeError(
            "inference artifact scientific configuration is not canonical"
        )
    if not isinstance(value["model"], dict):
        raise RuntimeError("inference artifact model state is invalid")
    for name in ("source_checkpoint_sha256", "model_sha256", "payload_sha256"):
        digest = value[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"inference artifact {name} is invalid")
    if model_state_sha256(value["model"]) != value["model_sha256"]:
        raise RuntimeError("inference artifact model content hash differs")
    if _payload_sha256(value) != value["payload_sha256"]:
        raise RuntimeError("inference artifact payload hash differs")
    return value



def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_inference_source(path: str | Path) -> dict[str, Any]:
    value = _load_artifact(Path(path))
    return {
        "model": value["model"],
        "scientific_config": value["scientific_config"],
        "iteration": int(value["iteration"]),
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_schema": value["source_checkpoint_schema"],
    }


def export_inference_artifact(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    lane: int | None = None,
) -> dict[str, Any]:
    source = Path(checkpoint_path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_schema = (
        checkpoint.get("checkpoint_schema")
        if isinstance(checkpoint, dict)
        else None
    )
    source_digest = sha256_file(source)
    if source_schema == DISTRIBUTED_CHECKPOINT_SCHEMA:
        rank_paths = load_distributed_checkpoint_manifest(
            source,
            expected_scientific_config=checkpoint["scientific_config"],
            world_size=int(checkpoint["world_size"]),
        )
        checkpoint = torch.load(
            rank_paths[0], map_location="cpu", weights_only=False
        )
    try:
        checkpoint = validate_checkpoint_payload(checkpoint)
    except RuntimeError as exc:
        raise RuntimeError(
            "inference export accepts only queued checkpoints or manifests "
            "with the strict schema"
        ) from exc
    if source_schema == REPLICA_CHECKPOINT_SCHEMA or (
        source_schema == DISTRIBUTED_CHECKPOINT_SCHEMA
        and checkpoint.get("checkpoint_schema") == REPLICA_CHECKPOINT_SCHEMA
    ):
        if (
            isinstance(lane, bool)
            or not isinstance(lane, int)
            or not 0 <= lane < int(checkpoint.get("n_replica", 0))
        ):
            raise RuntimeError(
                "replica inference export requires an explicit valid lane"
            )
        scientific_config = checkpoint["scientific_config"][lane]
        model_state = checkpoint["model"]["lanes"][lane]
        run_state = checkpoint["run_state"][lane]
    else:
        if lane is not None:
            raise RuntimeError("ordinary inference export has no lane selector")
        scientific_config = checkpoint.get("scientific_config")
        model_state = checkpoint.get("model")
        run_state = checkpoint.get("run_state")
    if not isinstance(run_state, dict) or not isinstance(
        run_state.get("iteration"), int
    ):
        raise RuntimeError("queued checkpoint run state is invalid")
    model = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model_state.items()
    }
    artifact = {
        "artifact_schema": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "source_checkpoint_schema": source_schema,
        "source_checkpoint_sha256": source_digest,
        "iteration": run_state["iteration"],
        "scientific_config": scientific_config,
        "model": model,
        "model_sha256": model_state_sha256(model),
    }
    artifact["payload_sha256"] = _payload_sha256(artifact)
    destination = Path(output_path)
    _atomic_torch(destination, artifact)
    return {
        "path": str(destination),
        "schema_version": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "iteration": artifact["iteration"],
        "source_checkpoint_sha256": artifact["source_checkpoint_sha256"],
        "model_sha256": artifact["model_sha256"],
        "artifact_sha256": sha256_file(destination),
    }




__all__ = [
    "INFERENCE_ARTIFACT_SCHEMA_VERSION",
    "export_inference_artifact",
    "load_inference_source",
    "sha256_file",
]
