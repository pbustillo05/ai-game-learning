from __future__ import annotations

import copy

import numpy as np
import torch

from pbg.core.branching import IntensityController
from pbg.core.config import load_config, seed_all
from pbg.core.training import build_game_bundle, load_checkpoint, save_checkpoint, train


def test_kuhn_smoke_and_checkpoint_restores_rng(tmp_path) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.train.out_dir = str(tmp_path / "run")
    train(cfg)
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == cfg.train.iters
    assert (tmp_path / "run" / "ckpt_0000002.pt").exists()

    game, model, _, _, bucket_fn = build_game_bundle(cfg)
    del game
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    controller = IntensityController(cfg.rollout, bucket_fn, use_moment_heads=False)
    torch_gen, np_gen = seed_all(cfg.train.seed)
    checkpoint = tmp_path / "roundtrip.pt"
    save_checkpoint(checkpoint, model, opt, scaler, controller, 7, cfg, torch_gen, np_gen)
    expected_torch = torch.rand(5, generator=torch_gen)
    expected_numpy = np_gen.random(5)
    expected_model = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    metadata = load_checkpoint(
        checkpoint, model, opt, scaler, controller, torch_gen, np_gen
    )
    assert metadata["iter"] == 7
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected_model.items())
    assert torch.equal(torch.rand(5, generator=torch_gen), expected_torch)
    assert np.array_equal(np_gen.random(5), expected_numpy)
