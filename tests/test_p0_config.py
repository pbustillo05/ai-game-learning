from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pbg.core.config import ConfigError, load_config, seed_all
from pbg.core.logging_util import MetricsLogger
from pbg.core.training import _configure_reproducibility


ROOT = Path(__file__).parents[1]


def test_nlhe_loads_every_config_field() -> None:
    cfg = load_config(ROOT / "configs" / "train" / "nlhe6.yaml")
    assert cfg.game.name == "poker"
    assert cfg.game.poker == {"n_players": 6, "stack_bb": 100.0, "rank_count": 13}
    assert cfg.rollout.beta_fix["preflop"] == 1.0
    assert cfg.completion.n_samples == 256
    assert cfg.completion.backend == "cuda"
    assert cfg.model["d_model"] == 192
    assert cfg.loss.lam == pytest.approx(0.95)
    assert cfg.loss.value_loss == "clipped"
    assert cfg.train.adam_betas == (0.9, 0.999)
    assert cfg.train.reproducibility == "throughput"


def test_hardware_profile_is_composed_without_changing_scientific_config() -> None:
    cfg = load_config(
        ROOT / "configs" / "train" / "nlhe6.yaml",
        hardware_profile="local_3x2080ti",
    )
    assert cfg.hardware is not None
    assert cfg.hardware.name == "local_3x2080ti"
    assert cfg.hardware.expected_compute_capability == (7, 5)
    assert cfg.hardware.training_max_tokens == 8192
    assert cfg.training.minibatch_size == 128


def test_unknown_nested_key_reports_full_path(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["rollout"]["surprise"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"root\.rollout\.surprise"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "pattern"),
    [
        ("game.poker", "mispelled_ante", r"root\.game\.poker\.mispelled_ante"),
        ("model", "e_max", r"unknown poker model configuration key"),
    ],
)
def test_game_owned_mappings_reject_unknown_keys(
    tmp_path: Path, section: str, key: str, pattern: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    target = raw
    for component in section.split("."):
        target = target[component]
    target[key] = 3
    path = tmp_path / "bad_owned.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=pattern):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "pattern"),
    [
        ("game.poker", "stacks_bb", r"root\.game\.poker\.stacks_bb"),
        ("model", "unknown_head", r"unknown poker model configuration key"),
    ],
)
def test_poker_owned_mappings_are_strict_at_top_level_load(
    tmp_path: Path, section: str, key: str, pattern: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    target = raw
    for component in section.split("."):
        target = target[component]
    target[key] = 3
    path = tmp_path / "bad_poker_owned.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=pattern):
        load_config(path)


@pytest.mark.parametrize("field", ["debug_asserts", "use_suit_consistency"])
def test_removed_noop_controls_are_rejected(tmp_path: Path, field: str) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    if field == "debug_asserts":
        raw[field] = True
    else:
        raw["loss"][field] = True
    path = tmp_path / "noop.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_training_mode_validates_beta_completion_and_loss_axes(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["training"]["mode"] = "normal"
    raw["completion"]["n_samples"] = None
    raw["loss"]["name"] = "ppo"
    raw["rollout"]["beta"] = None
    path = tmp_path / "normal.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.training.mode == "normal"
    assert cfg.rollout.beta is None

    raw["rollout"]["beta"] = {"default": 2.0}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="only valid for branched"):
        load_config(path)

    raw["rollout"]["beta"] = None
    raw["loss"]["name"] = "rb_ppo"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="does not support mode 'normal'"):
        load_config(path)


def test_seed_all_returns_repeatable_explicit_generators() -> None:
    torch_a, numpy_a = seed_all(19, rank=2)
    torch_b, numpy_b = seed_all(19, rank=2)
    torch_other, numpy_other = seed_all(19, rank=3)
    draw_torch_a = torch.rand(8, generator=torch_a)
    draw_torch_b = torch.rand(8, generator=torch_b)
    draw_numpy_a = numpy_a.random(8)
    draw_numpy_b = numpy_b.random(8)
    assert torch.equal(draw_torch_a, draw_torch_b)
    assert np.array_equal(draw_numpy_a, draw_numpy_b)
    assert not torch.equal(draw_torch_a, torch.rand(8, generator=torch_other))
    assert not np.array_equal(draw_numpy_a, numpy_other.random(8))


def test_reproducibility_profiles_change_backend_controls() -> None:
    _configure_reproducibility("strict")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark

    _configure_reproducibility("seeded")
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark

    _configure_reproducibility("throughput")
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark
    _configure_reproducibility("strict")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "reproducibility", "sometimes"),
        ("loss", "value_loss", "silent_noop"),
    ],
)
def test_behavioral_profile_enums_are_strict(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw[section][field] = value
    path = tmp_path / "bad_profile.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_metrics_logger_appends_json_lines(tmp_path: Path) -> None:
    logger = MetricsLogger(tmp_path)
    logger.log(4, {"loss": 1.5})
    logger.log(5, {"loss": 1.0, "entropy": 0.25})
    logger.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {"step": 4, "loss": 1.5},
        {"step": 5, "loss": 1.0, "entropy": 0.25},
    ]
