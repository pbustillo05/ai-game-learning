from __future__ import annotations

import inspect

import pytest
import torch
import yaml

from pbg.core.config import ConfigError, load_config
from pbg.core.minibatch import sample_minibatch
from pbg.core.config import seed_all
from pbg.core.training import build_game_bundle, train
import pbg.core.training as training_module


def test_fixed_beta_minimal_training_and_bundle_boundary(tmp_path) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.training.mode = "branched_rao_blackwell"
    cfg.rollout.beta = {"default": 2.0}
    cfg.rollout.n_trees = 1
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
    raw["loss"]["c_ws"] = 1.0
    path = tmp_path / "obsolete.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"root\.loss\.c_ws"):
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


def test_streamed_normal_ppo_covers_each_chunk_without_replacement(
    monkeypatch,
) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.training.mode = "normal"
    cfg.rollout.beta = None
    cfg.completion.n_samples = None
    cfg.loss.name = "ppo"
    cfg.loss.c_kl = 0.0
    cfg.rollout.n_trees = 8
    cfg.rollout.trajectory_chunk_size = 2
    cfg.train.gd_steps_per_iter = 1
    cfg.train.ppo_epochs = 2
    cfg.train.minibatch_size = 100

    bundle = build_game_bundle(cfg, torch.device("cpu"))
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch_gen, _ = seed_all(cfg.train.seed)
    selected: list[torch.Tensor] = []
    real_gather = training_module.gather_normal_minibatch

    def tracked_gather(dataset, owned_bundle, rows):
        selected.append(rows.detach().cpu().clone())
        return real_gather(dataset, owned_bundle, rows)

    monkeypatch.setattr(training_module, "gather_normal_minibatch", tracked_gather)
    metrics = training_module._one_iteration(
        cfg,
        bundle,
        optimizer,
        scaler,
        torch_gen,
        0,
    )

    assert metrics["trajectory_chunks"] == 4
    assert metrics["ppo_epochs"] == 2
    assert metrics["optimizer_steps"] == 8
    assert metrics["decision_epoch_coverage"] == pytest.approx(1.0)
    assert metrics["optimized_decision_rows"] == 2 * metrics["decisions"]
    assert len(selected) == 8
    for rows in selected:
        assert torch.equal(torch.sort(rows).values, torch.arange(rows.numel()))


def test_streamed_sequential_rb_ppo_covers_each_chunk_without_replacement(
    monkeypatch,
) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.loss.c_kl = 0.0
    cfg.rollout.n_trees = 8
    cfg.rollout.trajectory_chunk_size = 2
    cfg.train.gd_steps_per_iter = 1
    cfg.train.ppo_epochs = 2
    cfg.train.minibatch_size = 100

    bundle = build_game_bundle(cfg, torch.device("cpu"))
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch_gen, _ = seed_all(cfg.train.seed)
    selected: list[torch.Tensor] = []
    real_gather = training_module.gather_minibatch_rows

    def tracked_gather(tree, targets, policy_cache, reach_cache, owned_bundle, rows):
        selected.append(rows.detach().cpu().clone())
        return real_gather(
            tree, targets, policy_cache, reach_cache, owned_bundle, rows
        )

    monkeypatch.setattr(training_module, "gather_minibatch_rows", tracked_gather)
    metrics = training_module._one_iteration(
        cfg,
        bundle,
        optimizer,
        scaler,
        torch_gen,
        0,
    )

    assert metrics["trajectory_chunks"] == 4
    assert metrics["ppo_epochs"] == 2
    assert metrics["optimizer_steps"] == 8
    assert metrics["decision_epoch_coverage"] == pytest.approx(1.0)
    assert metrics["optimized_decision_rows"] == 2 * metrics["decisions"]
    assert len(selected) == 8
    for rows in selected:
        assert torch.equal(torch.sort(rows).values, torch.arange(rows.numel()))
