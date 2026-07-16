from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "pbg"


def _module_name(path: Path) -> tuple[str, str]:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    module = ".".join(parts)
    package = module if is_package else module.rpartition(".")[0]
    return module, package


def _import_targets(path: Path) -> list[str]:
    _, package = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                base = importlib.util.resolve_name(relative, package)
            else:
                base = node.module or ""
            targets.append(base)
    return targets


def test_t12_import_direction() -> None:
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        module, _ = _module_name(path)
        for target in _import_targets(path):
            if module.startswith("pbg.core") and target.startswith(("pbg.games", "pbg.eval")):
                violations.append(f"{module} imports {target}")
            if module.startswith("pbg.games.poker") and target.startswith("pbg."):
                if not target.startswith(("pbg.core", "pbg.games.poker")):
                    violations.append(f"{module} imports {target}")
    assert violations == []


def test_retired_synthetic_package_is_not_importable() -> None:
    assert importlib.util.find_spec("pbg.games.synthetic") is None
