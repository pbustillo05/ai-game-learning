from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from pbg.config import load_config
from pbg.core.game import GameBundle
from pbg.core.training import load_checkpoint
from pbg.runtime import build_game_bundle, train
from pbg.games.poker.adapter import PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.cards import validate_evaluator_combo_cards
from pbg.games.poker.folded import PokerGPUPartitionAdapter
from pbg.games.poker.mc import PokerGPUCompletion
from pbg.games.poker.bundle import estimate_rollout_memory
from pbg.games.poker.rollout_backend import PokerRolloutBackend


ROOT = Path(__file__).parents[1]
PRESETS = (
    "nlhe6_h100.yaml",
)
DEFERRED = {
    "beta_avg", "beta_max", "warmup_iters", "use_moment_heads",
    "ref_period", "c_kl", "c_u", "c_vvec",
}


def _small_poker_cfg(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "nlhe6_h100.yaml")
    cfg.game.n_players = 2
    cfg.game.stack_bb = 8.0
    cfg.game.rank_count = 13
    cfg.rollout.sequential_starts = 2
    cfg.rollout.node_budget = 64
    cfg.training.mode = "rao_blackwell"
    cfg.rollout.beta = None
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
    cfg.training.iters = 3
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 2
    cfg.training.checkpoint_every = 1
    cfg.training.out_dir = str(tmp_path / "poker_cpu")
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
    nlhe = loaded["nlhe6_h100.yaml"]
    assert nlhe.game.n_players == 6
    assert nlhe.rollout.sequential_starts == 1 and nlhe.rollout.node_budget == 4096
    assert nlhe.training.minibatch_size == 6
    assert nlhe.completion.backend == "cuda"
    assert nlhe.training.iters == 10_000 and nlhe.training.checkpoint_every == 100

def test_poker_rollout_memory_plan_bounds_recompute_and_full_storage() -> None:
    safe = estimate_rollout_memory(
        root_count=8, node_budget=512, n_players=6, reach_storage="recompute"
    )
    legacy = estimate_rollout_memory(
        root_count=256, node_budget=8192, n_players=6, reach_storage="recompute"
    )
    full = estimate_rollout_memory(
        root_count=256, node_budget=8192, n_players=6, reach_storage="full"
    )
    gib = 1024**3
    assert safe.capacity == 4096
    assert safe.peak_bytes < gib
    assert legacy.frontier_reach_bytes > 100 * gib
    assert full.reach_cache_bytes > 60 * gib

    execution = estimate_rollout_memory(
        root_count=256,
        node_budget=8192,
        n_players=6,
        reach_storage="recompute",
        sequential_recompute_live_rows=True,
    )
    assert execution.frontier_reach_bytes == 2 * 256 * 6 * 1326 * 4
    assert execution.frontier_reach_bytes < legacy.frontier_reach_bytes


def test_h100_bootstrap_bundle_constructs_without_capacity_sized_history() -> None:
    cfg = load_config(ROOT / "configs" / "nlhe6_h100.yaml")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    backend = bundle.rollout_backend
    assert backend.rollout_memory_plan.peak_bytes == 306_149_260
    assert "action_history" not in vars(backend.states)
    assert backend.states.history_parent.shape == (4_096,)


def test_nlhe_builder_constructs_one_owning_poker_model(tmp_path: Path) -> None:
    cfg = _small_poker_cfg(tmp_path)
    bundle = build_game_bundle(cfg, torch.device("cpu"))
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
    assert pbg.API_VERSION == "pbg-v3"
    validate_evaluator_combo_cards(pbg.combo_cards(13))
    assert tuple(inspect.signature(pbg.partition_many).parameters) == (
        "rank_count", "board_cards", "range_weights", "row_mask", "field_order", "backend",
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
        for line in (Path(cfg.training.out_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 3
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    checkpoint = Path(cfg.training.out_dir) / "ckpt_0000003.pt"
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    metadata = load_checkpoint(checkpoint, bundle.model)
    assert metadata["iter"] == 3
    backend = bundle.rollout_backend
    roots = backend.sequential_start_states(2)
    inputs = backend.completion_inputs(
        roots, torch.zeros((2, 2, 1326), dtype=torch.float32)
    )
    assert torch.allclose(
        inputs["payoff_baseline"].sum(1) + inputs["stratum_amount"].sum(1),
        torch.zeros(2),
        atol=1e-6,
    )


def test_normal_poker_settles_sampled_showdown_without_completion(tmp_path: Path) -> None:
    pytest.importorskip("poker_gpu_evaluator.pbg")
    cfg = _small_poker_cfg(tmp_path)
    cfg.training.mode = "normal"
    cfg.completion.n_samples = None
    cfg.loss.name = "ppo"
    cfg.training.iters = 1
    cfg.training.out_dir = str(tmp_path / "poker_normal")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert bundle.completion_backend is None
    train(cfg)
    record = json.loads(
        (Path(cfg.training.out_dir) / "metrics.jsonl").read_text().splitlines()[0]
    )
    assert torch.isfinite(torch.tensor(record["loss"]))
