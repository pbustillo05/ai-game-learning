"""Portable, model-only poker inference artifacts."""

from __future__ import annotations

import copy
import hashlib
import os
import string
from pathlib import Path
from typing import Any

import torch

from pbg.core.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint_payload,
)


INFERENCE_ARTIFACT_SCHEMA_VERSION = "pbg-inference-v1"
_ARTIFACT_FIELDS = frozenset(
    {
        "inference_artifact_schema_version",
        "source_checkpoint_schema_version",
        "source_checkpoint_sha256",
        "iteration",
        "config",
        "model",
    }
)


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_artifact(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("inference artifact must be a mapping")
    if (
        payload.get("inference_artifact_schema_version")
        != INFERENCE_ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "unsupported inference artifact schema version "
            f"{payload.get('inference_artifact_schema_version')!r}"
        )
    if set(payload) != _ARTIFACT_FIELDS:
        raise RuntimeError(
            "inference artifact fields do not match the current schema; "
            f"missing={sorted(_ARTIFACT_FIELDS.difference(payload))}, "
            f"extra={sorted(set(payload).difference(_ARTIFACT_FIELDS))}"
        )
    if payload["source_checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("inference artifact source checkpoint schema is obsolete")
    if not isinstance(payload["model"], dict):
        raise RuntimeError("inference artifact model must be a state mapping")
    config = payload["config"]
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("game"), dict)
        or not isinstance(config.get("model"), dict)
        or set(config) != {"game", "model"}
    ):
        raise RuntimeError(
            "inference artifact must contain game and model configuration"
        )
    digest = payload["source_checkpoint_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in string.hexdigits for character in digest)
    ):
        raise RuntimeError("inference artifact source checkpoint hash is invalid")
    return payload


def load_inference_source(path: str | Path) -> dict[str, Any]:
    """Load an inference artifact or full training checkpoint on CPU.

    The returned mapping has the normalized keys ``model``, ``config``,
    ``iteration``, ``source_checkpoint_sha256``, and ``source_schema_version``.
    Checkpoints are trusted project artifacts; the compact exported form can be
    read with Torch's restricted weights-only loader.
    """

    source = Path(path)
    try:
        restricted = torch.load(source, map_location="cpu", weights_only=True)
    except Exception:
        restricted = None
    if (
        isinstance(restricted, dict)
        and "inference_artifact_schema_version" in restricted
    ):
        payload = _validate_artifact(restricted)
        return {
            "model": payload["model"],
            "config": copy.deepcopy(payload["config"]),
            "iteration": int(payload["iteration"]),
            "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
            "source_schema_version": payload.get("source_checkpoint_schema_version"),
        }

    checkpoint = load_checkpoint_payload(source)
    return {
        "model": checkpoint["model"],
        "config": copy.deepcopy(checkpoint["config"]),
        "iteration": int(checkpoint["iteration"]),
        "source_checkpoint_sha256": sha256_file(source),
        "source_schema_version": checkpoint["checkpoint_schema_version"],
    }


def export_inference_artifact(
    checkpoint_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Atomically export model weights and scientific configuration only."""

    source = Path(checkpoint_path)
    payload = load_checkpoint_payload(source)
    config = payload["config"]
    if not isinstance(config.get("game"), dict) or not isinstance(
        config.get("model"), dict
    ):
        raise RuntimeError("checkpoint must contain current game and model configuration")
    artifact = {
        "inference_artifact_schema_version": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "source_checkpoint_schema_version": payload["checkpoint_schema_version"],
        "source_checkpoint_sha256": sha256_file(source),
        "iteration": int(payload["iteration"]),
        "config": {
            "game": copy.deepcopy(config["game"]),
            "model": copy.deepcopy(config["model"]),
        },
        "model": {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in payload["model"].items()
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, target)
    return {
        "path": str(target),
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
