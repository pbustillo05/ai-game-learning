"""Strict model-only artifacts for queued CUDA/MPS inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from pbg.queued.persistence import CHECKPOINT_SCHEMA, sha256_file


INFERENCE_ARTIFACT_SCHEMA_VERSION = "pbg-queued-inference-v1"
_FIELDS = {
    "artifact_schema",
    "source_checkpoint_schema",
    "source_checkpoint_sha256",
    "iteration",
    "scientific_config",
    "model",
}


def _load_artifact(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, dict)
        or set(value) != _FIELDS
        or value.get("artifact_schema") != INFERENCE_ARTIFACT_SCHEMA_VERSION
        or value.get("source_checkpoint_schema") != CHECKPOINT_SCHEMA
    ):
        raise RuntimeError("runtime accepts only strict queued inference artifacts")
    scientific = value["scientific_config"]
    if (
        not isinstance(scientific, dict)
        or not isinstance(scientific.get("mode"), dict)
        or not isinstance(scientific.get("game"), dict)
        or not isinstance(scientific.get("model"), dict)
    ):
        raise RuntimeError("inference artifact scientific configuration is invalid")
    if not isinstance(value["model"], dict):
        raise RuntimeError("inference artifact model state is invalid")
    digest = value["source_checkpoint_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("inference artifact source hash is invalid")
    return value


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
    checkpoint_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    source = Path(checkpoint_path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = {"checkpoint_schema", "scientific_config", "model", "run_state"}
    if (
        not isinstance(checkpoint, dict)
        or not required.issubset(checkpoint)
        or checkpoint["checkpoint_schema"] != CHECKPOINT_SCHEMA
    ):
        raise RuntimeError(
            "inference export accepts only ordinary queued checkpoints"
        )
    run_state = checkpoint["run_state"]
    if not isinstance(run_state, dict) or not isinstance(
        run_state.get("iteration"), int
    ):
        raise RuntimeError("queued checkpoint run state is invalid")
    artifact = {
        "artifact_schema": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "source_checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_checkpoint_sha256": sha256_file(source),
        "iteration": run_state["iteration"],
        "scientific_config": checkpoint["scientific_config"],
        "model": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in checkpoint["model"].items()
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(artifact, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "path": str(destination),
        "schema_version": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "iteration": artifact["iteration"],
        "source_checkpoint_sha256": artifact["source_checkpoint_sha256"],
    }


__all__ = [
    "INFERENCE_ARTIFACT_SCHEMA_VERSION",
    "export_inference_artifact",
    "load_inference_source",
    "sha256_file",
]
