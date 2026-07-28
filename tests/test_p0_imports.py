from __future__ import annotations

import importlib.util
from pathlib import Path

from pbg.queued.static_gates import (
    local_import_graph,
    queued_firewall_violations,
    reachable_modules,
)


ROOT = Path(__file__).parents[1]


def test_queued_generic_core_keeps_its_game_and_legacy_firewall() -> None:
    assert queued_firewall_violations(ROOT / "pbg/queued") == ()


def test_public_roots_cannot_reach_the_old_core() -> None:
    graph = local_import_graph(ROOT)
    for root in (
        "pbg.cli.train",
        "pbg.cli.train_replicas",
        "pbg.cli.benchmark",
        "pbg.cli.br_eval",
        "pbg.cli.export_inference",
        "pbg.cli.benchmark_inference",
        "pbg.cli.runpod_replica_selection",
    ):
        reached = reachable_modules(graph, root)
        assert not any(
            value == "pbg.core" or value.startswith("pbg.core.")
            for value in reached
        ), root


def test_retired_synthetic_package_is_not_importable() -> None:
    assert importlib.util.find_spec("pbg.games.synthetic") is None
