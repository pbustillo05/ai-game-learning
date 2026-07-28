"""Read-only static gates for the final core and accepted product roots."""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from typing import Iterable


def _source_package(path: Path) -> tuple[str, ...]:
    parts = path.with_suffix("").parts
    try:
        package_start = parts.index("pbg")
    except ValueError:
        return ()
    module_parts = parts[package_start:]
    if module_parts[-1] == "__init__":
        return module_parts[:-1]
    return module_parts[:-1]


def module_imports(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    source_package = _source_package(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = max(0, len(source_package) - node.level + 1)
                base_parts = source_package[:keep]
                module_parts = tuple((node.module or "").split("."))
                resolved = ".".join(base_parts + tuple(filter(None, module_parts)))
            else:
                resolved = node.module or ""
            if resolved:
                imports.add(resolved)
                imports.update(
                    f"{resolved}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
    return frozenset(imports)


def core_firewall_violations(package: Path) -> tuple[str, ...]:
    """Return generic-core imports of game-specific poker implementations."""

    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        for imported in module_imports(path):
            if imported == "pbg.games.poker" or imported.startswith(
                "pbg.games.poker."
            ):
                violations.append(f"{path}:{imported}")
    return tuple(violations)


def forbidden_literal_violations(
    package: Path, forbidden: Iterable[str]
) -> tuple[str, ...]:
    blocked = frozenset(forbidden)
    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in blocked:
                violations.append(f"{path}:{node.lineno}:{node.value}")
    return tuple(violations)


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def local_import_graph(root: Path) -> dict[str, frozenset[str]]:
    graph: dict[str, frozenset[str]] = {}
    for path in sorted((root / "pbg").rglob("*.py")):
        graph[_module_name(root, path)] = frozenset(
            imported
            for imported in module_imports(path)
            if imported == "pbg" or imported.startswith("pbg.")
        )
    return graph


def reachable_modules(
    graph: dict[str, frozenset[str]], root_module: str
) -> frozenset[str]:
    reached: set[str] = set()
    queue = deque((root_module,))
    while queue:
        module = queue.popleft()
        if module in reached:
            continue
        reached.add(module)
        for imported in graph.get(module, ()):
            candidate = imported
            while candidate and candidate not in graph:
                candidate = candidate.rpartition(".")[0]
            if candidate and candidate not in reached:
                queue.append(candidate)
    return frozenset(reached)


def accepted_root_violations(
    root: Path, production_roots: Iterable[str]
) -> tuple[str, ...]:
    """Return missing roots, roots bypassing core, or roots reaching stale imports."""

    graph = local_import_graph(root)
    violations: list[str] = []
    for production_root in production_roots:
        if production_root not in graph:
            violations.append(f"{production_root}:missing")
            continue
        reached = reachable_modules(graph, production_root)
        if not any(
            module == "pbg.core" or module.startswith("pbg.core.")
            for module in reached
        ):
            violations.append(f"{production_root}:does not reach pbg.core")
            continue
        stale = sorted(
            imported
            for module in reached
            for imported in graph.get(module, ())
            if imported == "pbg.queued" or imported.startswith("pbg.queued.")
        )
        if stale:
            violations.append(f"{production_root}:reaches {stale[0]}")
    return tuple(violations)


def stale_cutover_violations(root: Path) -> tuple[str, ...]:
    """Find source or import remnants of the isolated replacement namespace."""

    violations: list[str] = []
    stale_package = root / "pbg" / "queued"
    if stale_package.exists():
        violations.append(str(stale_package.relative_to(root)))
    for path in sorted((root / "pbg").rglob("*.py")):
        for imported in module_imports(path):
            if imported == "pbg.queued" or imported.startswith("pbg.queued."):
                violations.append(f"{path.relative_to(root)}:{imported}")
    return tuple(violations)


__all__ = [
    "accepted_root_violations",
    "core_firewall_violations",
    "forbidden_literal_violations",
    "local_import_graph",
    "module_imports",
    "reachable_modules",
    "stale_cutover_violations",
]
