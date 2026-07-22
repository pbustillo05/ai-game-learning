from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "audit_reachability.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("audit_reachability", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve forward annotations through the registered module.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reachability_separates_production_and_test_evidence(tmp_path: Path) -> None:
    audit = _audit_module()
    package = tmp_path / "demo"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "live.py").write_text(
        "USED = 10\n"
        "STALE_CONSTANT = 20\n\n"
        "def helper():\n"
        "    return USED\n\n"
        "def entry():\n"
        "    return helper()\n\n"
        "def test_only():\n"
        "    return 2\n\n"
        "def stale():\n"
        "    return 3\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_live.py").write_text(
        "from demo.live import test_only\n\ndef test_it():\n    assert test_only() == 2\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\ndemo = "demo.live:entry"\n', encoding="utf-8"
    )

    result = audit.audit_repository(tmp_path, package="demo", include_public_roots=False)

    assert "demo.live.entry" in result.reachable
    assert "demo.live.helper" in result.reachable
    assert "demo.live.USED" in result.reachable
    assert "demo.live.STALE_CONSTANT" not in result.reachable
    assert "demo.live.test_only" not in result.reachable
    assert "demo.live.stale" not in result.reachable
    retention = audit.retention_evidence_closure(result)
    assert "demo.live.test_only" in retention
    assert "demo.live.stale" not in retention
    assert audit.candidate_level(result, result.definitions["demo.live.test_only"]) == "REVIEW"
    assert audit.candidate_level(result, result.definitions["demo.live.stale"]) == "HIGH"


def test_main_guard_is_discovered_without_treating_error_print_as_root(tmp_path: Path) -> None:
    audit = _audit_module()
    package = tmp_path / "demo"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        "def work():\n    return 1\n\n"
        "def main():\n    return work()\n\n"
        "if __name__ == '__main__':\n"
        "    try:\n        main()\n"
        "    except Exception as exc:\n        print(exc)\n",
        encoding="utf-8",
    )

    result = audit.audit_repository(tmp_path, package="demo", include_public_roots=False)

    assert ("demo.cli.main", "__main__ guard") in result.roots
    assert result.missing_roots == []
    assert "demo.cli.main" in result.reachable
    assert "demo.cli.work" in result.reachable
