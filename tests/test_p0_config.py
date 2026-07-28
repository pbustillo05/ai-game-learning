from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from pbg.config import ConfigError, load_config, load_config_mapping
from pbg.games.poker.queued.config import SequentialModeConfig


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/train/nlhe_hu_8bb_normal_smoke.yaml"


def _raw() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_public_config_loads_complete_scientific_and_execution_contract() -> None:
    cfg = load_config(CONFIG)
    assert isinstance(cfg.scientific, SequentialModeConfig)
    assert cfg.mode == {
        "training_regime": "self_play",
        "rollout_mode": "sequential",
        "private_estimator": "sampled",
    }
    assert cfg.scientific.common.game.players == 2
    assert cfg.scientific.common.game.stack_bb == 8.0
    assert cfg.scientific.common.game.rank_count == 4
    assert cfg.scientific.common.model.d_model == 16
    assert cfg.scientific.common.loss.value_loss == "mse"
    assert cfg.scientific.common.optimizer.name == "adamw"
    assert cfg.scientific.common.reproducibility == "strict"
    assert cfg.execution.device == "cuda:0"
    assert cfg.execution.manifest.root_count == 1
    assert cfg.execution.layout.input_shapes == {
        "pre": (448,),
        "post_2": (448,),
    }


def test_public_config_round_trip_is_exact() -> None:
    cfg = load_config(CONFIG)
    rebuilt = load_config_mapping(
        {
            "scientific": cfg.scientific_mapping(),
            "execution": cfg.execution.mapping(),
        }
    )
    assert rebuilt == cfg


@pytest.mark.parametrize("legacy_root", ["game", "rollout", "training", "hardware"])
def test_legacy_root_shapes_are_rejected_with_dotted_paths(
    legacy_root: str,
) -> None:
    raw = _raw()
    raw[legacy_root] = {}
    with pytest.raises(ConfigError, match=rf"root\.{legacy_root}"):
        load_config_mapping(raw)


def test_runtime_hardware_overlay_is_rejected() -> None:
    with pytest.raises(TypeError, match="hardware_profile"):
        load_config(  # type: ignore[call-arg]
            CONFIG, hardware_profile="local_3x2080ti"
        )


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("scientific",), "surprise"),
        (("scientific", "mode"), "legacy_mode"),
        (("scientific", "game"), "ante"),
        (("scientific", "model"), "unknown_head"),
        (("scientific", "loss"), "ignored_coefficient"),
        (("scientific", "optimizer"), "momentum"),
        (("execution",), "auto_device"),
        (("execution", "manifest"), "capacity_growth"),
        (("execution", "manifest", "hardware"), "profile"),
        (("execution", "layout"), "cpu_staging"),
    ],
)
def test_unknown_nested_key_reports_full_path(
    path: tuple[str, ...],
    field: str,
) -> None:
    raw = _raw()
    target = raw
    for component in path:
        target = target[component]  # type: ignore[index,assignment]
    target[field] = 1
    dotted = r"\.".join((*path, field))
    with pytest.raises(ConfigError, match=rf"root\.{dotted}"):
        load_config_mapping(raw)


@pytest.mark.parametrize("device", ["cpu", "mps", "auto", ""])
def test_training_device_must_be_explicit_cuda(device: str) -> None:
    raw = _raw()
    raw["execution"]["device"] = device  # type: ignore[index]
    with pytest.raises(ConfigError, match=r"root\.execution\.device"):
        load_config_mapping(raw)


@pytest.mark.parametrize(
    ("path", "value", "pattern"),
    [
        (("execution", "iterations"), 0, r"root\.execution\.iterations"),
        (
            ("execution", "checkpoint_every"),
            0,
            r"root\.execution\.checkpoint_every",
        ),
        (
            ("execution", "query_chunk_size"),
            0,
            r"root\.execution\.query_chunk_size",
        ),
        (
            ("execution", "manifest", "root_count"),
            0,
            r"root\.execution\.manifest",
        ),
        (
            ("execution", "manifest", "family_pack_cap"),
            0,
            r"root\.execution\.manifest",
        ),
        (
            ("execution", "layout", "field_alignment"),
            0,
            r"root\.execution\.layout",
        ),
        (
            ("scientific", "root_count"),
            0,
            r"root\.scientific\.root_count",
        ),
        (
            ("scientific", "lambda"),
            -0.01,
            r"root\.scientific\.lambda",
        ),
        (
            ("scientific", "optimizer", "learning_rate"),
            -1.0,
            r"root\.scientific\.optimizer\.learning_rate",
        ),
        (
            ("scientific", "schedule", "batch_size"),
            0,
            r"root\.scientific\.schedule\.batch_size",
        ),
    ],
)
def test_numeric_boundaries_reject_with_dotted_paths(
    path: tuple[str, ...],
    value: object,
    pattern: str,
) -> None:
    raw = _raw()
    target = raw
    for component in path[:-1]:
        target = target[component]  # type: ignore[index,assignment]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=pattern):
        load_config_mapping(raw)


def test_manifest_root_count_must_match_scientific_root_count() -> None:
    raw = _raw()
    raw["execution"]["manifest"]["root_count"] = 2  # type: ignore[index]
    with pytest.raises(ConfigError, match="differs from scientific root_count"):
        load_config_mapping(raw)


def test_self_play_rejects_frozen_target() -> None:
    raw = _raw()
    raw["execution"]["frozen_checkpoint"] = "target.pt"  # type: ignore[index]
    with pytest.raises(ConfigError, match="inapplicable to self play"):
        load_config_mapping(raw)


def test_input_mapping_is_copied_not_mutated() -> None:
    raw = _raw()
    before = copy.deepcopy(raw)
    load_config_mapping(raw)
    assert raw == before
