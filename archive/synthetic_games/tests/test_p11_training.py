from __future__ import annotations

import copy
import gc
import json
import weakref

import numpy as np
import pytest
import torch

from pbg.core.config import load_config, seed_all
from pbg.core.training import build_game_bundle, load_checkpoint, save_checkpoint, train
import pbg.core.training as training_module


def test_kuhn_smoke_and_checkpoint_restores_rng(tmp_path) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.rollout.n_trees = 16
    cfg.completion.n_samples = 32
    cfg.train.iters = 2
    cfg.train.minibatch_size = 16
    cfg.train.checkpoint_every = 1
    cfg.train.out_dir = str(tmp_path / "run")
    train(cfg)
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == cfg.train.iters
    assert (tmp_path / "run" / "ckpt_0000002.pt").exists()

    bundle = build_game_bundle(cfg, torch.device("cpu"))
    model = bundle.model
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch_gen, np_gen = seed_all(cfg.train.seed)
    checkpoint = tmp_path / "roundtrip.pt"
    save_checkpoint(checkpoint, model, optimizer, scaler, 7, cfg, torch_gen, np_gen)
    expected_torch = torch.rand(5, generator=torch_gen)
    expected_numpy = np_gen.random(5)
    expected_model = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    metadata = load_checkpoint(
        checkpoint, model, optimizer, scaler, torch_gen, np_gen
    )
    assert metadata["iter"] == 7
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected_model.items())
    assert torch.equal(torch.rand(5, generator=torch_gen), expected_torch)
    assert np.array_equal(np_gen.random(5), expected_numpy)


def test_strict_training_resume_is_bitwise_continuation(tmp_path, monkeypatch) -> None:
    def small_cfg(out_dir):
        cfg = load_config("configs/kuhn_smoke.yaml")
        cfg.rollout.n_trees = 8
        cfg.completion.n_samples = 16
        cfg.train.iters = 3
        cfg.train.minibatch_size = 8
        cfg.train.checkpoint_every = 1
        cfg.train.out_dir = str(out_dir)
        cfg.train.reproducibility = "strict"
        return cfg

    uninterrupted = small_cfg(tmp_path / "uninterrupted")
    train(uninterrupted)

    resumed = small_cfg(tmp_path / "resumed")
    original_iteration = training_module._one_iteration

    def interrupt_after_checkpoint(*args, **kwargs):
        iteration = args[5] if len(args) > 5 else kwargs["iteration"]
        if iteration == 1:
            raise RuntimeError("simulated interruption")
        return original_iteration(*args, **kwargs)

    monkeypatch.setattr(training_module, "_one_iteration", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(resumed)
    monkeypatch.setattr(training_module, "_one_iteration", original_iteration)
    train(
        resumed,
        resume_from=tmp_path / "resumed" / "ckpt_0000001.pt",
    )

    expected = torch.load(
        tmp_path / "uninterrupted" / "ckpt_0000003.pt",
        map_location="cpu",
        weights_only=False,
    )
    actual = torch.load(
        tmp_path / "resumed" / "ckpt_0000003.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert expected["iteration"] == actual["iteration"] == 3
    assert all(
        torch.equal(expected["model"][name], actual["model"][name])
        for name in expected["model"]
    )
    expected_metrics = [
        {key: value for key, value in json.loads(line).items() if key != "run_id"}
        for line in (tmp_path / "uninterrupted" / "metrics.jsonl").read_text().splitlines()
    ]
    actual_metrics = [
        {key: value for key, value in json.loads(line).items() if key != "run_id"}
        for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
    ]
    assert actual_metrics == expected_metrics


def test_rollout_model_and_completion_result_die_at_optimization_boundary(
    monkeypatch,
) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.rollout.n_trees = 4
    cfg.completion.n_samples = 1
    bundle = build_game_bundle(cfg, torch.device("cpu"))

    model_refs: list[weakref.ReferenceType] = []
    result_refs: list[weakref.ReferenceType] = []
    real_deepcopy = copy.deepcopy

    def tracked_deepcopy(value, memo=None):
        clone = real_deepcopy(value, memo)
        if isinstance(value, torch.nn.Module):
            model_refs.append(weakref.ref(clone))
        return clone

    completion = bundle.completion_backend
    assert completion is not None
    real_complete = completion.complete

    def tracked_complete(*args, **kwargs):
        result = real_complete(*args, **kwargs)
        result_refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(training_module.copy, "deepcopy", tracked_deepcopy)
    monkeypatch.setattr(completion, "complete", tracked_complete)
    data = training_module._build_iteration_data(
        cfg,
        bundle,
        torch.Generator().manual_seed(cfg.train.seed),
        training_module._loss_spec(cfg),
    )
    gc.collect()

    assert data.targets is not None
    assert model_refs and model_refs[0]() is None
    assert result_refs and result_refs[0]() is None


def test_completion_microbatches_release_previous_result_before_next(
    monkeypatch,
) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.rollout.n_trees = 4
    cfg.completion.n_samples = 1
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    completion = bundle.completion_backend
    assert completion is not None
    real_complete = completion.complete
    result_refs: list[weakref.ReferenceType] = []

    def tracked_complete(*args, **kwargs):
        gc.collect()
        if result_refs:
            assert result_refs[-1]() is None
        result = real_complete(*args, **kwargs)
        result_refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(completion, "complete", tracked_complete)
    monkeypatch.setattr(training_module, "_completion_batch_rows", lambda *args: 1)
    data = training_module._build_iteration_data(
        cfg,
        bundle,
        torch.Generator().manual_seed(cfg.train.seed),
        training_module._loss_spec(cfg),
    )
    gc.collect()

    assert len(result_refs) > 1
    assert all(reference() is None for reference in result_refs)
    assert data.tree is None
    assert data.decision_reach.owners is not None
