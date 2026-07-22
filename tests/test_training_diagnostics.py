from types import SimpleNamespace
import json

import torch

from pbg.core.checkpointing import validate_metrics_prefix
from pbg.core.storage import (
    SealedDecisionInputs,
    SealedTensorBlock,
    TensorFieldSchema,
    TensorSchema,
)
from pbg.core.training import _decision_street_metrics


def test_decision_street_metrics_counts_and_normalizes() -> None:
    streets = torch.tensor([0, 1, 1, 2, 3, 3], dtype=torch.int32)
    schema = TensorSchema(
        "diagnostic_inputs", (TensorFieldSchema("street", torch.int32),)
    )
    block = SealedTensorBlock(
        schema,
        {"street": streets},
        streets.numel(),
        schema.capacity_bytes(streets.numel()),
    )
    inputs = SealedDecisionInputs(
        node_ids=torch.arange(6, dtype=torch.int32),
        partition_id=torch.zeros(6, dtype=torch.int32),
        partition_row=torch.arange(6, dtype=torch.int32),
        inputs={0: block},
    )
    sample = SimpleNamespace(
        decision_source={"node_id": inputs.node_ids}, decision_inputs=inputs
    )
    metrics: dict[str, float] = {}

    _decision_street_metrics(metrics, SimpleNamespace(), sample)

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
