from types import SimpleNamespace
import json

import torch

from pbg.core.checkpointing import validate_metrics_prefix
from pbg.core.training import _decision_street_metrics


def test_decision_street_metrics_counts_and_normalizes() -> None:
    backend = SimpleNamespace(
        states=SimpleNamespace(street=torch.tensor([0, 1, 1, 2, 3, 3]))
    )
    tree = SimpleNamespace(
        state_ref=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    )
    metrics: dict[str, float] = {}

    _decision_street_metrics(metrics, backend, tree, torch.arange(6))

    assert metrics["decision_nodes_preflop"] == 1.0
    assert metrics["decision_nodes_flop"] == 2.0
    assert metrics["decision_nodes_turn"] == 1.0
    assert metrics["decision_nodes_river"] == 2.0
    assert metrics["decision_fraction_flop"] == 2.0 / 6.0


def test_resume_validation_preserves_and_trims_uncheckpointed_metrics(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"iter": index, "run_id": "run"}
        for index in range(4)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    validate_metrics_prefix(path, completed=2, run_id="run")

    assert [json.loads(line)["iter"] for line in path.read_text().splitlines()] == [0, 1]
    backup = tmp_path / "metrics.jsonl.before_resume_0000002.bak"
    assert backup.exists()
    assert [json.loads(line)["iter"] for line in backup.read_text().splitlines()] == [0, 1, 2, 3]
