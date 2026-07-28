"""Content and accelerator identities shared by production consumers."""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any

from pbg.games.poker.queued.evaluator import bound_evaluator


def code_identity() -> dict[str, str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", "pbg", "tools"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
        diff = b""
    return {
        "git_revision": revision,
        "working_tree_sha256": hashlib.sha256(diff).hexdigest(),
    }


def accelerator_identity(rank_count: int) -> dict[str, Any]:
    info = bound_evaluator(rank_count).info
    evaluator = {
        name: value
        for name in dir(info)
        if not name.startswith("_")
        and isinstance(
            (value := getattr(info, name)),
            (str, int, float, bool, type(None)),
        )
    }
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = "unavailable"
    return {"evaluator": evaluator, "nvidia_driver": driver}


__all__ = ["accelerator_identity", "code_identity"]
