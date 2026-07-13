from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from pbg.core.config import load_config
from pbg.core.game import GameBundle
from pbg.core.training import GAME_FACTORIES, build_game_bundle, load_checkpoint, train
from pbg.games.poker.adapter import PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.cards import validate_evaluator_combo_cards
from pbg.games.poker.folded import PokerGPUPartitionAdapter
from pbg.games.poker.mc import PokerGPUCompletion
from pbg.games.poker.rollout_backend import PokerRolloutBackend


ROOT = Path(__file__).parents[1]
PRESETS = (
    "kuhn_smoke.yaml",
    "leduc.yaml",
    "synthetic_dial_example.yaml",
    "nlhe6_h100.yaml",
)
DEFERRED = {
    "beta_avg", "beta_max", "warmup_iters", "use_moment_heads",
    "ref_period", "c_kl", "c_u", "c_vvec",
}


def _small_poker_cfg(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "kuhn_smoke.yaml")
    cfg.game.name = "poker"
    cfg.game.synthetic = None
    cfg.game.poker = {
        "n_players": 2,
        "stack_bb": 2.0,
        "sb": 0.5,
        "bb": 1.0,
        "n_ticks": 2,
    }
    cfg.rollout.n_trees = 2
    cfg.rollout.node_budget = 64
    cfg.rollout.beta_mode = "sequential"
    cfg.rollout.beta_fix = {"default": 1.0}
    cfg.rollout.reach_storage = "full"
    cfg.completion.n_samples = 8
    cfg.completion.backend = "cpu"
    cfg.model = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 4,
    }
    cfg.train.iters = 3
    cfg.train.gd_steps_per_iter = 1
    cfg.train.minibatch_size = 2
    cfg.train.checkpoint_every = 1
    cfg.train.out_dir = str(tmp_path / "poker_cpu")
    return cfg


def _keys(value):
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_keys(child))
        return result
    return set()


def test_all_first_run_presets_load_strictly_without_deferred_settings() -> None:
    loaded = {name: load_config(ROOT / "configs" / name) for name in PRESETS}
    for name in PRESETS:
        raw = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert not (_keys(raw) & DEFERRED)
    assert loaded["kuhn_smoke.yaml"].rollout.n_trees == 512
    assert loaded["leduc.yaml"].rollout.beta_fix == {
        "default": 1.0, "round_0": 1.0, "chance_round_1": 2.0, "round_1": 2.0,
    }
    nlhe = loaded["nlhe6_h100.yaml"]
    assert nlhe.game.poker["n_players"] == 6
    assert nlhe.completion.backend == "cuda"
    assert nlhe.train.iters == 10_000 and nlhe.train.checkpoint_every == 100


def test_explicit_registry_builds_one_owning_poker_model(tmp_path: Path) -> None:
    cfg = _small_poker_cfg(tmp_path)
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert GAME_FACTORIES["poker"] == "pbg.games.poker.bundle:build_bundle"
    assert isinstance(bundle, GameBundle)
    assert isinstance(bundle.model, PokerModelBundle)
    assert isinstance(bundle.rollout_backend, PokerRolloutBackend)
    assert isinstance(bundle.policy_adapter, PokerPolicyAdapter)
    assert isinstance(bundle.completion_backend, PokerGPUCompletion)
    model_parameter_ids = {id(parameter) for parameter in bundle.model.parameters()}
    backend_parameter_ids = {
        id(value) for value in vars(bundle.rollout_backend).values()
        if isinstance(value, torch.nn.Parameter)
    }
    assert model_parameter_ids.isdisjoint(backend_parameter_ids)


def test_real_sibling_combo_table_and_partition_signature() -> None:
    pbg = pytest.importorskip("poker_gpu_evaluator.pbg")
    assert pbg.API_VERSION == "pbg-v2"
    validate_evaluator_combo_cards(pbg.COMBO_CARDS)
    assert tuple(inspect.signature(pbg.partition_many).parameters) == (
        "board_cards", "range_weights", "row_mask", "field_order", "backend",
    )
    source = inspect.getsource(PokerGPUPartitionAdapter.summarize)
    assert "query_combo_ids" not in source
    assert "query_card_sets" not in source
    assert "seed=" not in source


def test_three_iteration_cpu_sibling_smoke_is_finite_and_checkpoints(tmp_path: Path) -> None:
    pytest.importorskip("poker_gpu_evaluator.pbg")
    cfg = _small_poker_cfg(tmp_path)
    train(cfg)
    metrics = [
        json.loads(line)
        for line in (Path(cfg.train.out_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 3
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    checkpoint = Path(cfg.train.out_dir) / "ckpt_0000003.pt"
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    metadata = load_checkpoint(checkpoint, bundle.model)
    assert metadata["iter"] == 3
    backend = bundle.rollout_backend
    roots = backend.root_states(2)
    inputs = backend.completion_inputs(
        roots, torch.zeros((2, 2, 1326), dtype=torch.float32)
    )
    assert torch.allclose(
        inputs["payoff_baseline"].sum(1) + inputs["stratum_amount"].sum(1),
        torch.zeros(2),
        atol=1e-6,
    )
