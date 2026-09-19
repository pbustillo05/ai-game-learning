#!/usr/bin/env python3
"""Hash a completed v2.1 handoff directory and the exact working tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(("git", *arguments), cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    artifact_root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    artifacts = []
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(repository)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    tracked_diff = git(repository, "diff", "--binary", "HEAD", "--")
    status = git(repository, "status", "--porcelain=v1", "-z")
    result = {
        "schema": "pbg-fixed-target-br-handoff-manifest-v2.1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_version": "fixed_target_paired_response_gain_v2_1",
        "artifact_root": str(artifact_root.relative_to(repository)),
        "git_revision": git(repository, "rev-parse", "HEAD").decode().strip(),
        "tracked_worktree_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "worktree_status_sha256": hashlib.sha256(status).hexdigest(),
        "generator": str(Path(__file__).resolve().relative_to(repository)),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
