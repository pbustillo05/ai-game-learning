from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.static_gates import (
    accepted_root_violations,
    core_firewall_violations,
    local_import_graph,
    reachable_modules,
    stale_cutover_violations,
)

ROOT = Path(__file__).parents[1]


def test_generic_core_keeps_its_game_firewall() -> None:
    assert core_firewall_violations(ROOT / "pbg/core") == ()
    assert stale_cutover_violations(ROOT) == ()


def test_public_roots_reach_the_final_core() -> None:
    roots = (
        "pbg.cli.train",
        "pbg.cli.train_replicas",
        "pbg.cli.benchmark",
        "pbg.cli.br_eval",
        "pbg.cli.export_inference",
        "pbg.cli.benchmark_inference",
        "pbg.cli.runpod_replica_selection",
    )
    assert accepted_root_violations(ROOT, roots) == ()
    graph = local_import_graph(ROOT)
    for root in roots:
        reached = reachable_modules(graph, root)
        assert any(
            value == "pbg.core" or value.startswith("pbg.core.")
            for value in reached
        ), root


def test_retired_synthetic_package_is_not_importable() -> None:
    assert importlib.util.find_spec("pbg.games.synthetic") is None
