from __future__ import annotations

import ast
from pathlib import Path


def test_runpod_selection_uses_the_public_session() -> None:
    path = Path("pbg/cli/runpod_replica_selection.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "pbg.runtime" in imports
    assert not any(
        value == "pbg.queued" or value.startswith("pbg.queued.")
        for value in imports
    )


def test_runpod_selection_has_strict_deadline_and_replica_arguments() -> None:
    source = Path("pbg/cli/runpod_replica_selection.py").read_text(encoding="utf-8")
    assert "--deadline-minutes" in source
    assert "--replicas" in source
    assert "manifest.n_replica" in source
