"""Canonical sealing and validation for resolved execution manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


class ExecutionManifestError(ValueError):
    """A resolved execution manifest is malformed or has drifted."""


def _canonical_hash(value: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExecutionManifestError(
            "resolved execution manifest is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def seal_execution_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a detached manifest with a digest over every resolved field."""

    if "resolved_sha256" in payload:
        raise ExecutionManifestError("resolved_sha256 may only be added by the sealer")
    detached = dict(payload)
    return {**detached, "resolved_sha256": _canonical_hash(detached)}


def validate_execution_manifest(
    value: object,
    *,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate self-integrity and, for resume, exact expected identity."""

    if not isinstance(value, Mapping):
        raise ExecutionManifestError("resolved execution manifest must be a mapping")
    manifest = dict(value)
    digest = manifest.pop("resolved_sha256", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or digest != _canonical_hash(manifest)
    ):
        raise ExecutionManifestError("resolved execution manifest hash is invalid")
    validated = {**manifest, "resolved_sha256": digest}
    if expected is not None:
        expected_validated = validate_execution_manifest(expected)
        if validated != expected_validated:
            raise ExecutionManifestError("resolved execution manifest drifted")
    return validated


__all__ = [
    "ExecutionManifestError",
    "seal_execution_manifest",
    "validate_execution_manifest",
]
