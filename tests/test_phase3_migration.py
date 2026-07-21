from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pbg.config import load_config
from pbg.core.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from pbg.core.storage import (
    TRAINING_STORAGE_SCHEMA_VERSION,
    SamplingMetadata,
    TrainingStorage,
)
from pbg.games.poker.partitions import poker_partition_specs


def _scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def test_checkpoint_v3_and_recent_legacy_metadata_migration(tmp_path: Path) -> None:
    cfg = load_config("configs/acceptance/nlhe_hu_normal.yaml")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(7)
    numpy = np.random.default_rng(9)
    path = tmp_path / "current.pt"
    save_checkpoint(path, model, optimizer, _scaler(), 3, cfg, generator, numpy)
    header = inspect_checkpoint(path)
    assert header["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert header["storage_schema_version"] == TRAINING_STORAGE_SCHEMA_VERSION
    restored = load_checkpoint(path, model)
    assert restored["iter"] == 3

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "checkpoint_schema_version",
        "storage_schema_version",
        "controller_schema_version",
    ):
        payload.pop(key)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    legacy_header = inspect_checkpoint(legacy)
    assert legacy_header["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert legacy_header["migrated_from_checkpoint_schema"] == "pbg-checkpoint-v2"

    payload.pop("torch_generator")
    partial = tmp_path / "partial.pt"
    torch.save(payload, partial)
    with pytest.raises(RuntimeError, match="legacy checkpoint is partial"):
        inspect_checkpoint(partial)


def test_storage_schema_and_full_deck_partition_domains_are_strict() -> None:
    device = torch.device("cpu")
    empty = SamplingMetadata(
        {0: torch.tensor(-torch.inf, dtype=torch.float64, device=device)}
    )
    with pytest.raises(ValueError, match="storage schema"):
        TrainingStorage("normal", {}, empty, schema_version="legacy-tree-v0")

    class Output:
        v = torch.empty((0, 169))

    specs = poker_partition_specs(
        rank_count=13,
        n_players=2,
        router=lambda _: 0,
        forwards={0: lambda _: Output(), 1: lambda _: Output()},
    )
    pre, post = specs[0], specs[1]
    assert (pre.exact_width, pre.model_width) == (1326, 169)
    assert (post.exact_width, post.model_width) == (1326, 1326)
    pre_fields = {field.name: field for field in pre.input_schema.fields}
    post_fields = {field.name: field for field in post.input_schema.fields}
    assert pre_fields["range_weights"].shape == (2, 169)
    assert post_fields["range_weights"].shape == (2, 1326)
    assert "folded_summary_ref" not in {field.name for field in post.input_schema.fields}
