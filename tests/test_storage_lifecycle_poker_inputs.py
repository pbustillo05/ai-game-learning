from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pbg.config import load_config
from pbg.core.branching import sample_branched_tree
from pbg.core.game import EventBatch
from pbg.core.training import _build_iteration_data
from pbg.games.poker.bundle import build_bundle
from pbg.games.poker.engine import CHECK_CALL


ROOT = Path(__file__).resolve().parents[1]


def _tiny_normal_config():
    cfg = load_config(ROOT / "configs" / "nlhe6_h100.yaml")
    cfg.game.n_players = 2
    cfg.game.stack_bb = 8.0
    cfg.game.rank_count = 4
    cfg.rollout.roots_per_gpu = 4
    cfg.rollout.sequential_roots_per_gpu_initial = 4
    cfg.rollout.beta = None
    cfg.rollout.branching_mode = None
    cfg.completion.n_samples = None
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
        "use_branching_heads": False,
    }
    cfg.training.mode = "normal"
    cfg.training.amp = False
    cfg.training.minibatch_size = 2
    cfg.loss.name = "ppo"
    cfg.loss.c_t = 0.0
    cfg.loss.c_c = 0.0
    return cfg


def test_direct_rollout_seals_complete_state_independent_poker_inputs() -> None:
    cfg = _tiny_normal_config()
    bundle = build_bundle(cfg, torch.device("cpu"))
    generator = torch.Generator().manual_seed(17)
    data = _build_iteration_data(
        cfg,
        bundle,
        generator,
        frozen_model=bundle.model,
    )
    storage = data.storage
    specs = data.partition_specs

    assert storage is not None and specs is not None
    assert storage.partitions
    for partition_id, partition in storage.partitions.items():
        assert "state_ref" not in partition.inputs.columns
        assert "range_weights" in partition.inputs.columns
        assert "history_mask" in partition.inputs.columns
        rows = torch.arange(partition.inputs.rows, dtype=torch.int64)
        inputs = partition.gather_inputs(rows)
        output = specs[partition_id].forward(bundle.model, inputs)
        assert output.v.shape == (rows.numel(), specs[partition_id].model_width)
    assert bundle.rollout_backend.states.arena.allocated_capacity_bytes == 0


def test_all_live_postflop_forward_uses_no_persistent_summary_owner() -> None:
    cfg = _tiny_normal_config()
    bundle = build_bundle(cfg, torch.device("cpu"))
    backend = bundle.rollout_backend
    root = backend.sequential_start_states(1)
    chance = backend.step_player_batch(
        root, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32))
    )
    chance = backend.step_player_batch(
        chance, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32))
    )
    flop = backend.step_chance_batch(
        chance,
        EventBatch(
            torch.zeros(1, dtype=torch.int32),
            {"cards": torch.tensor([[36, 40, 44]], dtype=torch.int16)},
        ),
    )
    inputs = backend.serialize_decision_inputs(flop)
    inputs["range_weights"] = torch.ones(
        (1, backend.n_players, backend.n_hands), dtype=torch.float32
    )
    output = backend.policy_forward_partition(bundle.model, 1, inputs)
    assert output.v.shape == (1, backend.n_hands)
    assert backend.partition_summary_cache.allocated_capacity_bytes == 0


def test_branched_mandatory_child_no_fit_stops_before_draw_and_releases_pages() -> None:
    cfg = _tiny_normal_config()
    cfg.rollout.roots_per_gpu = 1
    cfg.rollout.sequential_roots_per_gpu_initial = 1
    cfg.rollout.storage_page_rows = 1
    cfg.rollout.execution_memory_bytes = 4_000
    cfg.rollout.branching_mode = "fixed"
    cfg.rollout.beta = {
        name: 1.0
        for name in (
            "default",
            "preflop",
            "flop",
            "turn",
            "river",
            "chance_to_flop",
            "chance_to_turn",
            "chance_to_river",
        )
    }
    cfg.model["use_branching_heads"] = True
    cfg.training.mode = "branched_rao_blackwell"
    cfg.loss.name = "rb_ppo"
    cfg.loss.c_t = 1.0
    cfg.loss.c_c = 1.0
    cfg.completion.n_samples = 4
    bundle = build_bundle(cfg, torch.device("cpu"))
    backend = bundle.rollout_backend
    forwards = 0
    original_forward = backend.policy_forward_family_actual_cost

    def record_forward(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        return original_forward(*args, **kwargs)

    backend.policy_forward_family_actual_cost = record_forward

    def forbid_posterior_draw(*args, **kwargs):
        raise AssertionError("posterior draw occurred before child-wave admission")

    backend.sample_profiles_keyed = forbid_posterior_draw
    with pytest.raises(MemoryError, match="byte budget exceeded"):
        sample_branched_tree(
            bundle,
            cfg,
            torch.Generator().manual_seed(3),
        )

    assert forwards == 1
    assert backend.states.arena.allocated_capacity_bytes == 0
    assert backend.states.history_arena.allocated_capacity_bytes == 0
    assert backend.partition_summary_cache.allocated_capacity_bytes == 0
