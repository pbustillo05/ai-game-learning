from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from pbg.config import ConfigError, load_config, load_config_mapping
from pbg.devices import resolve_execution_device, validate_workload_capabilities
from pbg.inference_artifact import (
    INFERENCE_ARTIFACT_SCHEMA_VERSION,
    export_inference_artifact,
    load_inference_source,
)
from pbg.queued.persistence import CHECKPOINT_SCHEMA
from pbg.queued.static_gates import local_import_graph, reachable_modules


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/train/nlhe_hu_8bb_normal_queued_smoke.yaml"
ROOTS = (
    "pbg.cli.train",
    "pbg.cli.train_replicas",
    "pbg.cli.benchmark",
    "pbg.cli.br_eval",
    "pbg.cli.export_inference",
    "pbg.cli.benchmark_inference",
    "pbg.cli.runpod_replica_selection",
)


def test_public_config_is_one_closed_three_axis_composition() -> None:
    config = load_config(CONFIG)
    assert config.mode == {
        "training_regime": "self_play",
        "rollout_mode": "sequential",
        "private_estimator": "sampled",
    }
    assert config.execution.manifest.root_count == 1
    assert config.execution.layout.input_shapes == {
        "pre": (448,),
        "post_2": (448,),
    }
    raw = {
        "scientific": config.scientific_mapping(),
        "execution": {
            **config.execution.mapping(),
            "frozen_checkpoint": None,
        },
    }
    assert load_config_mapping(raw) == config


def test_public_config_rejects_legacy_aliases_and_profile_overlays() -> None:
    with pytest.raises(ConfigError, match="profile overlays"):
        load_config(CONFIG, hardware_profile="legacy")
    import yaml

    raw = yaml.safe_load(CONFIG.read_text())
    raw["scientific"]["training"] = {"mode": "normal"}
    with pytest.raises(ConfigError, match="unknown configuration field"):
        load_config_mapping(raw)


def test_every_accepted_root_reaches_only_the_queued_replacement() -> None:
    graph = local_import_graph(ROOT)
    for root in ROOTS:
        reached = reachable_modules(graph, root)
        assert not any(
            module == "pbg.core" or module.startswith("pbg.core.")
            for module in reached
        ), root
        assert any(
            module == "pbg.queued"
            or module.startswith("pbg.queued.")
            or module.startswith("pbg.games.poker.queued")
            for module in reached
        ), root


def test_accelerator_contract_has_no_cpu_auto_or_mps_training_fallback(
    monkeypatch,
) -> None:
    for value in ("cpu", "auto", torch.device("cpu")):
        with pytest.raises(ValueError, match="cuda.*mps"):
            resolve_execution_device(value, workload="inference")
    with pytest.raises(RuntimeError, match="contract v2"):
        validate_workload_capabilities(torch.device("mps"), workload="training")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never falls back"):
        resolve_execution_device("cuda", workload="training")


def test_inference_export_accepts_only_queued_schema(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "scientific_config": config.scientific_mapping(),
            "model": {"weight": torch.ones(2)},
            "run_state": {"iteration": 3},
        },
        checkpoint,
    )
    artifact = tmp_path / "inference.pt"
    report = export_inference_artifact(checkpoint, artifact)
    assert report["schema_version"] == INFERENCE_ARTIFACT_SCHEMA_VERSION
    source = load_inference_source(artifact)
    assert source["iteration"] == 3
    assert source["scientific_config"] == config.scientific_mapping()
    legacy = tmp_path / "legacy.pt"
    torch.save({"checkpoint_schema_version": "pbg-checkpoint-v8"}, legacy)
    with pytest.raises(RuntimeError, match="only ordinary queued"):
        export_inference_artifact(legacy, tmp_path / "rejected.pt")
