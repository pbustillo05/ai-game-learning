from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pbg.core.config import ConfigError, load_config, seed_all
from pbg.core.logging_util import MetricsLogger


ROOT = Path(__file__).parents[1]


def test_kuhn_smoke_loads_every_config_field() -> None:
    cfg = load_config(ROOT / "configs" / "kuhn_smoke.yaml")
    assert cfg.game.name == "synthetic"
    assert cfg.game.synthetic is not None
    assert cfg.rollout.beta_fix == {"default": 1.0}
    assert cfg.completion.n_samples == 256
    assert cfg.completion.backend == "exact"
    assert cfg.model["n_buckets"] == 128
    assert cfg.loss.lam == pytest.approx(0.95)
    assert cfg.train.adam_betas == (0.9, 0.999)
    assert cfg.debug_asserts is True


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
    raw = yaml.safe_load((ROOT / "configs" / "kuhn_smoke.yaml").read_text())
    raw["rollout"]["surprise"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"root\.rollout\.surprise"):
        load_config(path)


def test_training_mode_validates_beta_completion_and_loss_axes(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "kuhn_smoke.yaml").read_text())
    raw["training"]["mode"] = "normal"
    raw["completion"]["n_samples"] = None
    raw["loss"]["name"] = "ppo"
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
