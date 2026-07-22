from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import yaml

from pbg.config import ConfigError, load_config
from pbg.core.br import SHARED_RESPONSE_SEMANTICS
from pbg.core.checkpointing import save_checkpoint, scientific_config
from pbg.core.game import PRETERM, PROBE_FOLD, EventBatch
from pbg.core.rollout import ProbeRollout, sample_trees_sequential
from pbg.core.training import _build_iteration_data, _seal_sequential_rollout_storage
from pbg.core.utility import utility_scale
from pbg.eval.br_probe import evaluate_br_probe
from pbg.games.poker.bundle import build_bundle
from pbg.runtime import train


ROOT = Path(__file__).resolve().parents[1]


def _force_fold_policy(model: torch.nn.Module) -> None:
    """Make the checkpoint-binding integration path terminate immediately."""

    with torch.no_grad():
        for street_model in (model.preflop, model.postflop):
            street_model.type_head.affine.weight.zero_()
            street_model.type_head.affine.bias.copy_(
                torch.tensor([20.0, -20.0, -20.0])
            )


def _probe_config(tmp_path: Path):
    raw = yaml.safe_load(
        (ROOT / "configs/train/nlhe_hu_32bb_rb.yaml").read_text()
    )
    raw["game"].update(stack_bb=8.0, rank_count=4)
    raw["completion"]["n_samples"] = 4
    raw["rollout"].update(
        sequential_roots_per_gpu_initial=2,
        sequential_roots_per_gpu_minimum=2,
        sequential_roots_per_gpu_maximum=2,
        family_batch_rows=2,
        max_active_trajectories=2,
        storage_page_rows=8,
        execution_memory_bytes=1 << 28,
    )
    raw["model"].update(
        d_model=8,
        n_heads=2,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=1,
        k_act=4,
    )
    raw["training"].update(
        gd_steps_per_iter=1,
        minibatch_size=4,
        amp=False,
    )
    raw["probe"] = {
        "target_checkpoint": str(tmp_path / "target.pt"),
        "target_sha256": "a" * 64,
        "initialization": "random",
    }
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def _three_player_cuda_probe_config(tmp_path: Path):
    raw = yaml.safe_load(
        (ROOT / "configs/train/nlhe_hu_32bb_rb.yaml").read_text()
    )
    raw["game"].update(n_players=3, stack_bb=8.0, rank_count=4)
    raw["completion"]["n_samples"] = 4
    raw["rollout"].update(
        sequential_roots_per_gpu_initial=1,
        sequential_roots_per_gpu_minimum=1,
        sequential_roots_per_gpu_maximum=1,
        family_batch_rows=1,
        max_active_trajectories=1,
        storage_page_rows=8,
        execution_memory_bytes=1 << 28,
    )
    raw["model"].update(
        d_model=8,
        n_heads=2,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=1,
        k_act=4,
        max_players=3,
        class_embedding_dim=4,
        folder_beta_rank=4,
    )
    raw["training"].update(
        iters=1,
        gd_steps_per_iter=1,
        minibatch_size=3,
        lr=1.0e-4,
        checkpoint_every=1,
        out_dir=str(tmp_path / "cuda_probe"),
        amp=False,
    )
    raw["probe"] = {
        "target_checkpoint": str(tmp_path / "target.pt"),
        "target_sha256": "a" * 64,
        "initialization": "target",
    }
    path = tmp_path / "three_player_cuda_probe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path, hardware_profile="local_3x2080ti")


def test_probe_config_requires_sequential_rb_and_hashes_content(tmp_path: Path) -> None:
    cfg = _probe_config(tmp_path)
    assert cfg.probe is not None and cfg.probe.initialization == "random"
    stored = scientific_config(cfg)
    assert "target_checkpoint" not in stored["probe"]
    assert stored["probe"]["target_sha256"] == "a" * 64

    raw = yaml.safe_load((tmp_path / "probe.yaml").read_text())
    raw["training"]["mode"] = "normal"
    raw["completion"]["n_samples"] = None
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="best-response probes require"):
        load_config(bad)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_three_player_probe_fold_cuda_train_resume_and_evaluate(tmp_path: Path) -> None:
    """Exercise the real synthetic cutoff while two opponents remain live."""

    cfg = _three_player_cuda_probe_config(tmp_path)
    device = torch.device("cuda", 0)
    initial = build_bundle(cfg, device)
    _force_fold_policy(initial.model)
    initial.opponent_model = copy.deepcopy(initial.model).eval().requires_grad_(False)
    cfg.rollout.probe_seats = [0]

    sampled = sample_trees_sequential(
        initial, cfg, torch.Generator(device=device).manual_seed(7), iteration=0
    )
    assert isinstance(sampled, ProbeRollout)
    assert sampled.sample.leaf_source.rows == 1
    assert sampled.sample.leaf_source["owner"].item() == PROBE_FOLD
    reward = sampled.sample.leaf_source["reward"][0]
    assert reward[0].item() == 0.0
    assert not bool(reward[1:].any())

    rollout, _, completion, _ = _seal_sequential_rollout_storage(
        cfg,
        initial,
        sampled,
        iteration=0,
        gen=torch.Generator(device=device).manual_seed(7),
    )
    assert completion is not None
    assert completion.selected_owner_field == "selected_owner"
    assert torch.equal(
        rollout.leaves["selected_owner"], torch.tensor([0], device=device)
    )
    rows = torch.tensor([0], dtype=torch.int64, device=device)
    compact = completion.complete(rollout.leaves.columns, rows=rows)
    dense = initial.completion_backend.complete_sealed(
        initial.rollout_backend,
        rollout.leaves.columns,
        rows=rows,
        n_samples=int(cfg.completion.n_samples),
    ).for_learning()
    assert compact.p_hat.shape == (1, 1, initial.rollout_backend.n_hands)
    assert compact.v_hat.shape == compact.p_hat.shape
    assert dense.p_hat.shape[1] == 3
    assert torch.equal(compact.p_hat[:, 0], dense.p_hat[:, 0])
    assert torch.equal(
        compact.v_hat[:, 0], dense.v_hat[:, 0] * utility_scale(cfg)
    )
    assert completion.workspace_bound(1) >= (
        3 * initial.rollout_backend.n_hands * 12
        + initial.rollout_backend.n_hands * 8
    )

    hands = initial.rollout_backend.n_hands
    preterminal_leaves = {
        "owner": torch.tensor([PRETERM], dtype=torch.int32, device=device),
        "selected_owner": torch.tensor([2], dtype=torch.int32, device=device),
        "range_weights": torch.ones((1, 3, hands), device=device),
        "completion_key": torch.tensor([91], dtype=torch.int64, device=device),
        "board_cards": torch.full((1, 5), 255, dtype=torch.uint8, device=device),
        "stratum_level": torch.zeros((1, 3), dtype=torch.int32, device=device),
        "payoff_baseline": torch.zeros((1, 3), device=device),
        "stratum_amount": torch.tensor([[3.0, 0.0, 0.0]], device=device),
        "fold_payoff": torch.zeros((1, 3), device=device),
    }
    selected_preterminal = initial.completion_backend.complete_selected_sealed(
        initial.rollout_backend,
        preterminal_leaves,
        rows=rows,
        n_samples=4,
    ).for_learning()
    dense_preterminal = initial.completion_backend.complete_sealed(
        initial.rollout_backend,
        preterminal_leaves,
        rows=rows,
        n_samples=4,
    ).for_learning()
    assert selected_preterminal.p_hat.shape == (1, 1, hands)
    assert torch.equal(
        selected_preterminal.p_hat[:, 0], dense_preterminal.p_hat[:, 2]
    )
    assert torch.equal(
        selected_preterminal.v_hat[:, 0], dense_preterminal.v_hat[:, 2]
    )
    del compact, dense, rollout, completion
    del selected_preterminal, dense_preterminal, preterminal_leaves

    target = tmp_path / "target.pt"
    target_cfg = copy.deepcopy(cfg)
    target_cfg.probe = None
    target_optimizer = torch.optim.Adam(initial.model.parameters(), lr=cfg.training.lr)
    target_scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(
        target,
        initial.model,
        target_optimizer,
        target_scaler,
        0,
        target_cfg,
        torch.Generator(device=device).manual_seed(19),
        np.random.default_rng(23),
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    cfg.probe.target_sha256 = digest
    del sampled, initial
    torch.cuda.empty_cache()

    train(cfg)
    checkpoint = Path(cfg.training.out_dir) / "ckpt_0000001.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["probe_target"]["semantics"] == SHARED_RESPONSE_SEMANTICS
    assert payload["probe_target"]["sha256"] == digest
    assert payload["logical_work"]["optimizer_steps"] == 1

    resumed = copy.deepcopy(cfg)
    train(resumed, resume_from=checkpoint)

    report = evaluate_br_probe(
        checkpoint,
        target_checkpoint=target,
        hardware_profile="local_3x2080ti",
        n_hands=9,
        batch_size=9,
        seed=11,
        device="cuda:0",
    )
    assert report.semantics == SHARED_RESPONSE_SEMANTICS
    assert math.isfinite(report.mean_gain_bb_per_hand)
