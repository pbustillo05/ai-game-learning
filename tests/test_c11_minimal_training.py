from __future__ import annotations

import inspect

import pytest
import torch
import yaml

from pbg.core.config import ConfigError, load_config
from pbg.core.minibatch import sample_minibatch
from pbg.core.training import build_game_bundle, train


def test_fixed_beta_minimal_training_and_bundle_boundary(tmp_path) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.training.mode = "branched_rao_blackwell"
    cfg.rollout.beta = {"default": 2.0}
    cfg.rollout.n_trees = 4
    cfg.train.iters = 1
    cfg.train.minibatch_size = 4
    cfg.train.out_dir = str(tmp_path / "fixed")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert bundle.model is not None
    assert bundle.rollout_backend.device == torch.device("cpu")
    train(cfg)
    assert (tmp_path / "fixed" / "metrics.jsonl").exists()


def test_removed_config_fields_are_unknown_and_minibatch_has_no_node_lookup(tmp_path) -> None:
    raw = yaml.safe_load(open("configs/kuhn_smoke.yaml", encoding="utf-8"))
    raw["loss"]["c_kl"] = 1.0
    path = tmp_path / "obsolete.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"root\.loss\.c_kl"):
        load_config(path)
    source = inspect.getsource(sample_minibatch)
    assert "target_lookup" not in source
    assert "node_ids.tolist" not in source


def test_normal_mode_trains_without_completion_or_posterior_targets(tmp_path) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.training.mode = "normal"
    cfg.rollout.beta = None
    cfg.completion.n_samples = None
    cfg.loss.name = "ppo"
    cfg.rollout.n_trees = 8
    cfg.train.iters = 1
    cfg.train.minibatch_size = 8
    cfg.train.out_dir = str(tmp_path / "normal")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert bundle.completion_backend is None
    train(cfg)
    assert (tmp_path / "normal" / "metrics.jsonl").exists()
