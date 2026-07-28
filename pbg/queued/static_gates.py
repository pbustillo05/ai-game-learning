"""Read-only static gates for the queued import boundary."""

from __future__ import annotations

import ast
import hashlib
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


def queued_firewall_violations(package: Path) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        for imported in module_imports(path):
            if imported == "pbg.core" or imported.startswith("pbg.core."):
                violations.append(f"{path}:{imported}")
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


def dual_engine_roots(
    root: Path, production_roots: Iterable[str]
) -> tuple[str, ...]:
    graph = local_import_graph(root)
    violations: list[str] = []
    for production_root in production_roots:
        reached = reachable_modules(graph, production_root)
        old_reached = any(
            module == "pbg.core" or module.startswith("pbg.core.")
            for module in reached
        )
        queued_reached = any(
            module == "pbg.queued"
            or module.startswith("pbg.queued.")
            or module == "pbg.games.poker.queued"
            or module.startswith("pbg.games.poker.queued.")
            for module in reached
        )
        if old_reached and queued_reached:
            violations.append(production_root)
    return tuple(violations)


def migrated_similarity_violations(
    root: Path,
    production_roots: Iterable[str],
    *,
    minimum_nodes: int = 24,
) -> tuple[str, ...]:
    """Find exact nontrivial function bodies copied from the legacy core."""

    if minimum_nodes <= 0:
        raise ValueError("minimum_nodes must be positive")
    graph = local_import_graph(root)
    reached = set().union(
        *(reachable_modules(graph, module) for module in production_roots)
    )

    def bodies(paths: Iterable[Path]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for path in sorted(paths):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                count = sum(1 for _ in ast.walk(node))
                if count < minimum_nodes:
                    continue
                normalized = ast.dump(
                    ast.Module(body=node.body, type_ignores=[]),
                    annotate_fields=True,
                    include_attributes=False,
                )
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                result.setdefault(digest, []).append(
                    f"{path.relative_to(root)}:{node.lineno}:{node.name}"
                )
        return result

    legacy = bodies((root / "pbg/core").rglob("*.py"))
    migrated_paths = [
        root / f"{module.replace('.', '/')}.py"
        for module in reached
        if not (module == "pbg.core" or module.startswith("pbg.core."))
    ]
    migrated_paths.extend(
        root / module.replace(".", "/") / "__init__.py"
        for module in reached
        if not (module == "pbg.core" or module.startswith("pbg.core."))
    )
    migrated = bodies(path for path in migrated_paths if path.is_file())
    return tuple(
        f"{old} == {new}"
        for digest in sorted(set(legacy).intersection(migrated))
        for old in legacy[digest]
        for new in migrated[digest]
    )


__all__ = [
    "dual_engine_roots",
    "forbidden_literal_violations",
    "local_import_graph",
    "migrated_similarity_violations",
    "module_imports",
    "queued_firewall_violations",
    "reachable_modules",
]
