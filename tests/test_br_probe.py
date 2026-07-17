from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil

import pytest
import torch
import yaml

from pbg.config import ConfigError, load_config
from pbg.core.checkpointing import scientific_config
from pbg.core.game import PROBE_FOLD, EventBatch
from pbg.core.rollout import ProbeRollout, sample_trees_sequential
from pbg.core.tree import TreeBatch
from pbg.core.training import _build_iteration_data, _loss_spec
from pbg.eval.br_probe import evaluate_br_probe
from pbg.games.poker.bundle import build_bundle
from pbg.runtime import train


ROOT = Path(__file__).resolve().parents[1]


def _probe_config(tmp_path: Path):
    raw = yaml.safe_load(
        (ROOT / "configs/acceptance/nlhe_hu_sequential_rb.yaml").read_text()
    )
    raw["completion"]["backend"] = "cpu"
    raw["completion"]["n_samples"] = 4
    raw["rollout"].update(
        sequential_starts=4,
        node_budget=32,
        family_batch_rows=4,
        max_active_trajectories=4,
        arena_chunk_nodes=32,
        execution_memory_bytes=1 << 28,
    )
    raw["model"].update(d_model=32, n_heads=4, n_layers_post=1)
    raw["probe"] = {
        "target_checkpoint": str(tmp_path / "target.pt"),
        "target_sha256": "a" * 64,
        "initialization": "random",
    }
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


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


def test_probe_rollout_and_backward_retain_only_selected_player(tmp_path: Path) -> None:
    cfg = _probe_config(tmp_path)
    bundle = build_bundle(cfg, torch.device("cpu"))
    target_cfg = copy.deepcopy(cfg)
    target_cfg.model["m_raise"] = 1
    target_cfg.model["k_act"] = 4
    bundle.opponent_model = (
        build_bundle(target_cfg, torch.device("cpu"))
        .model.eval()
        .requires_grad_(False)
    )
    gen = torch.Generator().manual_seed(123)
    sampled = sample_trees_sequential(bundle, cfg, gen, iteration=0)
    assert isinstance(sampled, ProbeRollout)
    nodes = sampled.policy.node_ids.to(torch.int64)
    owners = sampled.tree.owner[nodes]
    selected = sampled.seats[sampled.tree.tree_id[nodes].to(torch.int64)]
    assert torch.equal(owners, selected)
    assert torch.equal(sampled.policy.node_ids, sampled.decision_reach.node_ids)

    cfg.rollout.max_active_trajectories = 1
    rescheduled = sample_trees_sequential(
        bundle, cfg, torch.Generator().manual_seed(999), iteration=0
    )
    assert isinstance(rescheduled, ProbeRollout)
    assert torch.equal(rescheduled.seats, sampled.seats)

    data = _build_iteration_data(
        cfg,
        bundle,
        torch.Generator().manual_seed(321),
        _loss_spec(cfg),
        iteration=0,
    )
    assert torch.equal(data.targets.node_ids, data.policy_cache.node_ids)
    assert torch.equal(data.targets.node_ids, data.decision_reach.node_ids)
    target_nodes = data.targets.node_ids.to(torch.int64)
    # Compact data no longer retains the tree, so rollout ownership is checked
    # above and exact target/cache alignment is checked here.
    assert target_nodes.numel() > 0


def test_identical_probe_has_exact_zero_paired_gain(tmp_path: Path) -> None:
    cfg = _probe_config(tmp_path)
    bundle = build_bundle(cfg, torch.device("cpu"))
    target = tmp_path / "target.pt"
    target_config = scientific_config(cfg)
    target_config.pop("probe")
    torch.save(
        {"model": bundle.model.state_dict(), "config": target_config, "iteration": 0},
        target,
    )
    target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    cfg.probe.target_checkpoint = str(target)
    cfg.probe.target_sha256 = target_hash
    probe = tmp_path / "probe.pt"
    torch.save(
        {
            "model": bundle.model.state_dict(),
            "config": scientific_config(cfg),
            "probe_target": {"path": str(target), "sha256": target_hash},
        },
        probe,
    )
    report = evaluate_br_probe(
        probe,
        target_checkpoint=target,
        hardware_profile="local_3x2080ti",
        n_hands=4,
        batch_size=4,
        seed=7,
        device="cpu",
    )
    assert report.mean_gain_bb_per_hand == 0.0
    assert report.ci95_bb_per_hand == 0.0
    assert report.n_replays == 8
    assert report.evaluation_completion_samples == 8


def test_probe_fold_completion_is_selected_posterior_and_public_payoff(
    tmp_path: Path,
) -> None:
    cfg = _probe_config(tmp_path)
    bundle = build_bundle(cfg, torch.device("cpu"))
    backend = bundle.rollout_backend
    root = backend.sequential_start_states(1)
    seat = backend.classify(root).to(torch.int64)
    folded = backend.step_player_batch(
        root, EventBatch(torch.zeros(1, dtype=torch.int32))
    )
    tree = TreeBatch(
        parent=torch.tensor([-1], dtype=torch.int32),
        child_ptr=torch.tensor([0, 0], dtype=torch.int32),
        child_node=torch.empty(0, dtype=torch.int32),
        child_event=torch.empty(0, dtype=torch.int32),
        nu=torch.empty(0, dtype=torch.float32),
        tree_id=torch.tensor([0], dtype=torch.int32),
        owner=torch.tensor([PROBE_FOLD], dtype=torch.int32),
        depth_p=torch.zeros((1, 2), dtype=torch.int32),
        depth_player=torch.zeros(1, dtype=torch.int32),
        state_ref=folded,
        path_weight=torch.ones(1),
    )
    tree.validate()
    stats = bundle.completion_backend.complete_selected_keyed(
        backend,
        tree,
        torch.tensor([0], dtype=torch.int32),
        torch.zeros((1, 2, backend.x_sizes[0])),
        8,
        torch.tensor([12345], dtype=torch.int64),
        seat,
    )
    selected = int(seat[0])
    opponent = 1 - selected
    posterior = stats.p_hat[0, selected]
    assert posterior.sum() == pytest.approx(1.0)
    assert not bool(stats.p_hat[0, opponent].any())
    assert not bool(stats.v_hat[0, opponent].any())
    payoff = -backend.states.committed_total[folded.long(), seat].item()
    assert torch.all(stats.v_hat[0, selected, posterior > 0] == payoff)


def test_probe_checkpoint_binds_hash_allows_move_and_target_initialization(
    tmp_path: Path,
) -> None:
    cfg = _probe_config(tmp_path)
    initial = build_bundle(cfg, torch.device("cpu"))
    target = tmp_path / "target.pt"
    target_config = scientific_config(cfg)
    target_config.pop("probe")
    torch.save(
        {"model": initial.model.state_dict(), "config": target_config, "iteration": 0},
        target,
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    cfg.probe.target_sha256 = digest
    cfg.probe.initialization = "target"
    cfg.training.lr = 0.0
    cfg.training.iters = 1
    cfg.training.checkpoint_every = 1
    cfg.training.out_dir = str(tmp_path / "run")
    train(cfg)

    checkpoint = Path(cfg.training.out_dir) / "ckpt_0000001.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["probe_target"]["sha256"] == digest
    assert payload["logical_work"]["trajectories"] == 4
    assert payload["logical_work"]["completion_samples"] == 16
    assert payload["logical_work"]["minibatches"] >= 1
    assert payload["logical_work"]["optimizer_steps"] == 1
    for name, value in initial.model.state_dict().items():
        assert torch.equal(payload["model"][name], value)

    moved = tmp_path / "moved_target.pt"
    shutil.copy2(target, moved)
    resumed = copy.deepcopy(cfg)
    resumed.probe.target_checkpoint = str(moved)
    train(resumed, resume_from=checkpoint)

    rejected = copy.deepcopy(resumed)
    rejected.probe.target_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256"):
        train(rejected, resume_from=checkpoint)
