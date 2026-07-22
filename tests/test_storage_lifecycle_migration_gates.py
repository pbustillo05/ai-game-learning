"""Executable sole-path gates for the authoritative storage lifecycle."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "pbg"


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _imports(relative: str) -> set[str]:
    tree = ast.parse(_source(relative), filename=relative)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_training_orchestrator_uses_rollout_storage_and_authoritative_targets_directly() -> None:
    source = _source("pbg/core/training.py")
    imports = _imports("pbg/core/training.py")
    for retired in (
        "_RolloutSealInput",
        "_seal_iteration_training_storage",
        "SampledTree",
        "RolloutDecisionRows",
        "StoppedLeafRows",
        "PagedTreeSource",
        "streamed_branched_value",
        "build_normal_dataset_from_sampled",
    ):
        assert retired not in source
    assert {
        "build_normal_targets",
        "build_sequential_rb_targets",
        "build_branched_rb_targets",
    }.issubset(imports)


def test_poker_training_inputs_do_not_retain_mutable_state_handles() -> None:
    partition_source = _source("pbg/games/poker/partitions.py")
    forward_source = _source("pbg/games/poker/rollout_backend.py")
    assert 'TensorFieldSchema("state_ref"' not in partition_source
    assert 'inputs.get("state_ref")' not in forward_source


def test_retired_rollout_and_target_artifacts_are_absent_from_production() -> None:
    retired = {
        "DeviceTreeBuffer",
        "PolicyCache",
        "ReachCache",
        "DecisionReachCache",
        "NormalDataset",
        "NodeTargets",
        "streamed_branched_value",
        "_RolloutSealInput",
        "_seal_iteration_training_storage",
    }
    found: dict[str, list[str]] = {name: [] for name in retired}
    for path in PRODUCTION.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in retired:
            if name in source:
                found[name].append(str(path.relative_to(ROOT)))
    assert {name: paths for name, paths in found.items() if paths} == {}


def test_node_budget_is_absent_from_production_python() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in PRODUCTION.rglob("*.py")
        if "node_budget" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_production_rollout_and_targets_do_not_grow_payload_with_torch_cat() -> None:
    offenders: list[str] = []
    for relative in (
        "pbg/core/rollout.py",
        "pbg/core/branching.py",
        "pbg/core/targets.py",
    ):
        tree = ast.parse(_source(relative), filename=relative)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "cat"
        ]
        if calls:
            offenders.append(relative)
    assert offenders == []
