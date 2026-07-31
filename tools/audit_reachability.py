#!/usr/bin/env python3
"""Conservative static reachability audit for the active PBG Python package.

This is a deletion *candidate* generator, not a proof that code is dead.  It
keeps production reachability separate from references in tests, documentation,
and tools so a reviewer can distinguish test-only code from wholly unreferenced
code.  Dynamic Python features can always require manual follow-up.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Public library surfaces which are not necessarily reached by an installed CLI.
# Keep this short: an obsolete root hides everything below it from this audit.
DECLARED_PUBLIC_ROOTS = (
    "pbg.runtime.train",
    "pbg.config.load_config",
    "pbg.config.load_config_mapping",
    "pbg.eval.br_probe.evaluate_br_probe",
    "pbg.inference.PokerInferenceSession",
    "pbg.inference.PokerInferenceSession.predict",
    "pbg.inference.PokerInferenceSession.full_strategy",
    "pbg.inference.PokerInferenceSession.advance_many",
    "pbg.inference.PokerInferenceSession.deal_many",
    "pbg.inference.parse_card",
    "pbg.inference_artifact.load_inference_source",
    "pbg.inference_artifact.export_inference_artifact",
    "pbg.benchmark.benchmark_training",
    "pbg.benchmark.compare_benchmark",
)

MAIN_GUARD_RE = re.compile(r"^pbg(?:\.[A-Za-z_]\w*)+$")
DOC_SUFFIXES = {".md", ".rst", ".tex", ".txt"}

# Python and framework protocols invoke these methods without a source-level
# ``object.method(...)`` call.  Keep the list explicit and deliberately short:
# adding a name weakens the method-level deletion screen for every reachable
# class that defines it.
IMPLICIT_METHOD_NAMES = frozenset(
    {
        "__call__",
        "__deepcopy__",
        "__enter__",
        "__eq__",
        "__exit__",
        "__getitem__",
        "__hash__",
        "__init__",
        "__iter__",
        "__len__",
        "__new__",
        "__next__",
        "__post_init__",
        "__repr__",
        "__setitem__",
        "__str__",
        "forward",
    }
)


@dataclass(frozen=True)
class Definition:
    qname: str
    module: str
    name: str
    kind: str
    path: Path
    line: int
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Assign | ast.AnnAssign


@dataclass
class ModuleInfo:
    name: str
    path: Path
    tree: ast.Module
    package: str
    bindings: dict[str, str] = field(default_factory=dict)


@dataclass
class AuditResult:
    repository: Path
    roots: list[tuple[str, str]]
    missing_roots: list[tuple[str, str]]
    definitions: dict[str, Definition]
    reachable: set[str]
    edges: dict[str, set[str]]
    external_references: dict[str, Counter[str]]
    dynamic_hazards: dict[str, list[str]]

    @property
    def unreachable(self) -> list[Definition]:
        return sorted(
            (item for name, item in self.definitions.items() if name not in self.reachable),
            key=lambda item: (str(item.path), item.line, item.qname),
        )


def module_name(repository: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(repository).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    module = ".".join(parts)
    package = module if is_package else module.rpartition(".")[0]
    return module, package


def parse_modules(repository: Path, package: str) -> dict[str, ModuleInfo]:
    package_dir = repository / package.replace(".", "/")
    modules: dict[str, ModuleInfo] = {}
    for path in sorted(package_dir.rglob("*.py")):
        name, parent = module_name(repository, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules[name] = ModuleInfo(name, path, tree, parent)
    return modules


def import_target(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    relative = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative, package)


def collect_definitions(modules: dict[str, ModuleInfo]) -> dict[str, Definition]:
    definitions: dict[str, Definition] = {}
    for module in modules.values():
        for statement in module.tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qname = f"{module.name}.{statement.name}"
                definitions[qname] = Definition(
                    qname, module.name, statement.name,
                    type(statement).__name__, module.path, statement.lineno, statement,
                )
                if isinstance(statement, ast.ClassDef):
                    for member in statement.body:
                        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        method_qname = f"{qname}.{member.name}"
                        definitions[method_qname] = Definition(
                            method_qname,
                            module.name,
                            member.name,
                            "method",
                            module.path,
                            member.lineno,
                            member,
                        )
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target_node in targets:
                    if not isinstance(target_node, ast.Name) or not target_node.id.isupper():
                        continue
                    qname = f"{module.name}.{target_node.id}"
                    definitions[qname] = Definition(
                        qname, module.name, target_node.id, "constant",
                        module.path, statement.lineno, statement,
                    )
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    module.bindings.setdefault(local, alias.name if alias.asname else local)
            elif isinstance(node, ast.ImportFrom):
                base = import_target(node, module.package)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    target = f"{base}.{alias.name}" if base else alias.name
                    module.bindings.setdefault(alias.asname or alias.name, target)
    return definitions


def attribute_parts(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return list(reversed(parts))


class ReferenceVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module: ModuleInfo,
        definitions: dict[str, Definition],
        modules: dict[str, ModuleInfo],
        *,
        current_class: str | None = None,
    ) -> None:
        self.module = module
        self.definitions = definitions
        self.modules = modules
        self.current_class = current_class
        self.methods_by_name: dict[str, tuple[str, ...]] = defaultdict(tuple)
        grouped: dict[str, list[str]] = defaultdict(list)
        for qname, definition in definitions.items():
            if definition.kind == "method":
                grouped[definition.name].append(qname)
        self.methods_by_name = {
            name: tuple(symbols) for name, symbols in grouped.items()
        }
        self.references: set[str] = set()
        self.hazards: list[str] = []

    def resolve_parts(self, parts: list[str]) -> str | None:
        first, *rest = parts
        if first in {"self", "cls"} and self.current_class and rest:
            target = ".".join((self.current_class, *rest))
            if target in self.definitions:
                return target
        local = f"{self.module.name}.{first}"
        if local in self.definitions:
            if rest:
                target = ".".join((local, *rest))
                if target in self.definitions:
                    return target
            return local
        if first in self.module.bindings:
            target = self.module.bindings[first]
            if rest:
                target = ".".join((target, *rest))
            return target
        return None

    def add_target(self, target: str | None) -> None:
        if target in self.definitions or target in self.modules:
            self.references.add(target)  # type: ignore[arg-type]

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.add_target(self.resolve_parts([node.id]))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts = attribute_parts(node)
        resolved = self.resolve_parts(parts) if parts else None
        if parts:
            self.add_target(resolved)
        if resolved is None and isinstance(node.ctx, ast.Load):
            self.references.update(self.methods_by_name.get(node.attr, ()))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.add_target(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = import_target(node, self.module.package)
        self.add_target(base)
        for alias in node.names:
            self.add_target(f"{base}.{alias.name}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "setattr", "hasattr"}:
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                attr = node.args[1].value
                parts = attribute_parts(node.args[0])
                if isinstance(attr, str) and parts:
                    target = self.resolve_parts([*parts, attr])
                    self.add_target(target)
                    if target is None:
                        self.references.update(self.methods_by_name.get(attr, ()))
            else:
                self.hazards.append(f"dynamic {node.func.id} at line {node.lineno}")
        elif isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
            self.hazards.append(f"{node.func.id} at line {node.lineno}")
        self.generic_visit(node)


def definition_runtime_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Yield expressions evaluated while a function/class is being defined."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield from node.decorator_list
        yield from node.args.defaults
        yield from (value for value in node.args.kw_defaults if value is not None)
        if node.returns:
            yield node.returns
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if arg.annotation:
                yield arg.annotation
        if node.args.vararg and node.args.vararg.annotation:
            yield node.args.vararg.annotation
        if node.args.kwarg and node.args.kwarg.annotation:
            yield node.args.kwarg.annotation
    elif isinstance(node, ast.ClassDef):
        yield from node.decorator_list
        yield from node.bases
        yield from (keyword.value for keyword in node.keywords)


def build_graph(
    modules: dict[str, ModuleInfo], definitions: dict[str, Definition]
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    edges: dict[str, set[str]] = defaultdict(set)
    hazards: dict[str, list[str]] = defaultdict(list)
    for module in modules.values():
        init_visitor = ReferenceVisitor(module, definitions, modules)
        for statement in module.tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for runtime_node in definition_runtime_nodes(statement):
                    init_visitor.visit(runtime_node)
            else:
                init_visitor.visit(statement)
        edges[module.name].update(init_visitor.references)
        hazards[module.name].extend(init_visitor.hazards)

    for definition in definitions.values():
        class_qname = (
            definition.qname.rpartition(".")[0]
            if definition.kind == "method"
            else None
        )
        visitor = ReferenceVisitor(
            modules[definition.module],
            definitions,
            modules,
            current_class=class_qname,
        )
        if isinstance(definition.node, ast.ClassDef):
            for runtime_node in definition_runtime_nodes(definition.node):
                visitor.visit(runtime_node)
            for statement in definition.node.body:
                if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visitor.visit(statement)
            for method_name in IMPLICIT_METHOD_NAMES:
                method = f"{definition.qname}.{method_name}"
                if method in definitions:
                    visitor.references.add(method)
        else:
            visitor.visit(definition.node)
        edges[definition.qname].update(visitor.references - {definition.qname})
        # Importing/reaching a symbol necessarily executes its module initializer.
        edges[definition.qname].add(definition.module)
        hazards[definition.qname].extend(visitor.hazards)
    return dict(edges), dict(hazards)


def script_roots(repository: Path, modules: dict[str, ModuleInfo]) -> list[tuple[str, str]]:
    roots: list[tuple[str, str]] = []
    pyproject = repository / "pyproject.toml"
    if pyproject.exists():
        in_scripts = False
        assignment = re.compile(r'^([A-Za-z0-9_.-]+)\s*=\s*["\']([^"\']+)["\']\s*$')
        for raw in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped.startswith("["):
                in_scripts = stripped == "[project.scripts]"
                continue
            match = assignment.match(stripped) if in_scripts else None
            if match:
                command, value = match.groups()
                target = value.split("[", 1)[0].replace(":", ".")
                roots.append((target, f"pyproject script {command}"))

    for module in modules.values():
        has_main_guard = any(
            isinstance(statement, ast.If) and is_main_guard(statement.test)
            for statement in module.tree.body
        )
        target = f"{module.name}.main"
        if has_main_guard and target in {
            f"{module.name}.{statement.name}"
            for statement in module.tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }:
            roots.append((target, "__main__ guard"))

    tool_dir = repository / "tools"
    if tool_dir.exists():
        module_pattern = re.compile(r"(?:-m\s+|python\s+-m\s+)(pbg(?:\.[A-Za-z_]\w*)+)")
        for path in tool_dir.iterdir():
            if not path.is_file() or path.name == Path(__file__).name:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for match in module_pattern.finditer(text):
                target = f"{match.group(1)}.main"
                roots.append((target, f"launcher {path.relative_to(repository)}"))
    return roots


def is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        and len(node.ops) == len(node.comparators) == 1
        and isinstance(node.ops[0], ast.Eq)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "__main__"
    )


def read_roots_file(path: Path) -> list[tuple[str, str]]:
    roots = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.split("#", 1)[0].strip()
        if value:
            if not MAIN_GUARD_RE.fullmatch(value):
                raise ValueError(f"invalid root at {path}:{number}: {value!r}")
            roots.append((value, str(path)))
    return roots


def transitive_reachability(roots: Iterable[str], edges: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    queue = deque(roots)
    while queue:
        item = queue.popleft()
        if item in seen:
            continue
        seen.add(item)
        queue.extend(edges.get(item, ()))
    return seen


def external_reference_counts(
    repository: Path, definitions: dict[str, Definition]
) -> dict[str, Counter[str]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for qname, definition in definitions.items():
        by_name[definition.name].append(qname)
    result: dict[str, Counter[str]] = defaultdict(Counter)

    areas = (("tests", repository / "tests"), ("docs", repository / "docs"),
             ("tools", repository / "tools"))
    for label, directory in areas:
        if not directory.exists():
            continue
        paths = directory.rglob("*") if label != "tools" else directory.iterdir()
        for path in paths:
            if not path.is_file() or path == Path(__file__):
                continue
            if path.suffix not in ({".py"} | DOC_SUFFIXES | {".sh", ""}):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            counts = Counter(re.findall(r"\b[A-Za-z_]\w*\b", text))
            for name, count in counts.items():
                if name in by_name:
                    for qname in by_name[name]:
                        result[qname][label] += count
    return dict(result)


def audit_repository(
    repository: Path,
    *,
    package: str = "pbg",
    extra_roots: Iterable[tuple[str, str]] = (),
    include_public_roots: bool = True,
) -> AuditResult:
    repository = repository.resolve()
    modules = parse_modules(repository, package)
    definitions = collect_definitions(modules)
    edges, hazards = build_graph(modules, definitions)
    roots = script_roots(repository, modules)
    if include_public_roots:
        roots.extend((root, "declared public API") for root in DECLARED_PUBLIC_ROOTS)
    roots.extend(extra_roots)
    roots = list(dict.fromkeys(roots))
    accepted = [(root, source) for root, source in roots if root in definitions or root in modules]
    missing = [(root, source) for root, source in roots if root not in definitions and root not in modules]
    reachable = transitive_reachability((root for root, _ in accepted), edges)
    external = external_reference_counts(repository, definitions)
    return AuditResult(repository, accepted, missing, definitions, reachable, edges, external, hazards)


def git_blame(repository: Path, definitions: Iterable[Definition]) -> dict[tuple[Path, int], str]:
    result: dict[tuple[Path, int], str] = {}
    by_path: dict[Path, list[int]] = defaultdict(list)
    for definition in definitions:
        by_path[definition.path].append(definition.line)
    for path, lines in by_path.items():
        relative = path.relative_to(repository)
        try:
            completed = subprocess.run(
                ["git", "blame", "--line-porcelain", "--", str(relative)],
                cwd=repository, check=True, text=True, capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        current_line = 0
        commit = summary = date = ""
        wanted = set(lines)
        for raw in completed.stdout.splitlines():
            header = re.match(r"^([0-9a-f^]+) \d+ (\d+)(?: \d+)?$", raw)
            if header:
                commit, current_line = header.group(1), int(header.group(2))
                summary = date = ""
            elif raw.startswith("author-time "):
                from datetime import datetime, timezone
                date = datetime.fromtimestamp(int(raw.split()[1]), timezone.utc).date().isoformat()
            elif raw.startswith("summary "):
                summary = raw[8:]
            elif raw.startswith("\t") and current_line in wanted:
                result[(path, current_line)] = f"{commit[:8]} {date} {summary}".strip()
    return result


def retention_evidence_closure(
    result: AuditResult,
    *,
    sources: set[str] | None = None,
    include_dynamic: bool = True,
) -> set[str]:
    """Propagate selected evidence into private helpers of a candidate."""
    unreachable = {item.qname for item in result.unreachable}
    seeds = {
        name for name in unreachable
        if (
            any(
                count and (sources is None or source in sources)
                for source, count in result.external_references.get(name, {}).items()
            )
            or include_dynamic and (
                result.dynamic_hazards.get(name)
                or result.dynamic_hazards.get(result.definitions[name].module)
            )
        )
    }
    closure: set[str] = set()
    queue = deque(seeds)
    while queue:
        name = queue.popleft()
        if name in closure:
            continue
        closure.add(name)
        queue.extend(target for target in result.edges.get(name, ()) if target in unreachable)
    return closure


def candidate_level(
    result: AuditResult,
    definition: Definition,
    retention: tuple[set[str], set[str]] | None = None,
) -> str:
    if retention is None:
        strong = retention_evidence_closure(result, sources={"tests", "tools"})
        any_evidence = retention_evidence_closure(result)
    else:
        strong, any_evidence = retention
    if definition.qname in strong:
        return "REVIEW"
    if definition.qname in any_evidence:
        return "DOCS_ONLY"
    return "HIGH"


def json_payload(result: AuditResult) -> dict[str, object]:
    retention = (
        retention_evidence_closure(result, sources={"tests", "tools"}),
        retention_evidence_closure(result),
    )
    candidates = []
    for item in result.unreachable:
        candidates.append({
            "symbol": item.qname,
            "path": str(item.path.relative_to(result.repository)),
            "line": item.line,
            "kind": item.kind,
            "priority": candidate_level(result, item, retention),
            "external_references": dict(result.external_references.get(item.qname, {})),
            "dynamic_hazards": result.dynamic_hazards.get(item.qname, [])
                + result.dynamic_hazards.get(item.module, []),
        })
    return {
        "repository": str(result.repository),
        "roots": [{"symbol": root, "source": source} for root, source in result.roots],
        "missing_roots": [{"symbol": root, "source": source} for root, source in result.missing_roots],
        "definitions_indexed": len(result.definitions),
        "reachable_definitions": len(result.reachable & result.definitions.keys()),
        "candidates": candidates,
    }


def print_report(result: AuditResult, *, show_git: bool) -> None:
    candidates = result.unreachable
    retention = (
        retention_evidence_closure(result, sources={"tests", "tools"}),
        retention_evidence_closure(result),
    )
    blame = git_blame(result.repository, candidates) if show_git else {}
    reachable_definitions = len(result.reachable & result.definitions.keys())
    print(f"repository: {result.repository}")
    print(f"definitions indexed: {len(result.definitions)}")
    print(f"accepted roots: {len(result.roots)}")
    for root, source in result.roots:
        print(f"  ROOT {root} ({source})")
    for root, source in result.missing_roots:
        print(f"  MISSING ROOT {root} ({source})")
    print(f"reachable definitions: {reachable_definitions}")
    print(f"review candidates: {len(candidates)}")
    print("NOTE: static reachability is evidence for manual review, never proof of dead code.")
    print("      Test/docs/tool references do not make a symbol production-reachable.\n")

    for level in ("HIGH", "DOCS_ONLY", "REVIEW"):
        selected = [item for item in candidates if candidate_level(result, item, retention) == level]
        if not selected:
            continue
        title = {
            "HIGH": "no external references detected",
            "DOCS_ONLY": "mentioned only by documentation/audits",
            "REVIEW": "referenced by tests/tools or affected by dynamic dispatch",
        }[level]
        print(f"[{level}: {title}]")
        for item in selected:
            relative = item.path.relative_to(result.repository)
            details = [f"{relative}:{item.line}", item.kind]
            refs = result.external_references.get(item.qname, Counter())
            if refs:
                details.append("refs=" + ",".join(f"{key}:{value}" for key, value in sorted(refs.items())))
            hazards = result.dynamic_hazards.get(item.qname, []) + result.dynamic_hazards.get(item.module, [])
            if hazards:
                details.append("dynamic=" + "; ".join(sorted(set(hazards))))
            if (item.path, item.line) in blame:
                details.append("last-change=" + blame[(item.path, item.line)])
            print(f"  {item.qname} ({'; '.join(details)})")
        print()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    result.add_argument("--package", default="pbg")
    result.add_argument("--root", action="append", default=[], help="additional fully-qualified root")
    result.add_argument("--roots-file", action="append", type=Path, default=[])
    result.add_argument("--no-public-roots", action="store_true", help="use only discovered and supplied roots")
    result.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    result.add_argument("--git", action="store_true", help="add line-level git blame (slower)")
    result.add_argument("--fail-on-high", action="store_true", help="exit 1 if HIGH candidates exist")
    result.add_argument(
        "--fail-on-candidates",
        action="store_true",
        help="exit 1 if any production-unreachable definition exists",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    extra: list[tuple[str, str]] = []
    for root in args.root:
        if not MAIN_GUARD_RE.fullmatch(root):
            parser().error(f"invalid --root: {root!r}")
        extra.append((root, "command line"))
    try:
        for path in args.roots_file:
            extra.extend(read_roots_file(path))
        result = audit_repository(
            args.repository, package=args.package, extra_roots=extra,
            include_public_roots=not args.no_public_roots,
        )
    except (OSError, SyntaxError, ValueError) as error:
        print(f"audit_reachability.py: error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(json_payload(result), indent=2, sort_keys=True))
    else:
        print_report(result, show_git=args.git)
    retention = (
        retention_evidence_closure(result, sources={"tests", "tools"}),
        retention_evidence_closure(result),
    )
    high = any(candidate_level(result, item, retention) == "HIGH" for item in result.unreachable)
    failed = (args.fail_on_high and high) or (
        args.fail_on_candidates and bool(result.unreachable)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
