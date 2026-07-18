from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import torch
import yaml

from pbg.config import load_config
from pbg.runtime import train
from pbg.games.poker.cards import DeckGeometry, category_profile
from pbg.core.game import EventBatch
from pbg.games.poker.engine import PokerConfig
from pbg.games.poker.engine import CHECK_CALL
from pbg.games.poker.folded import FoldSnapshotBatch, PokerGPUPartitionAdapter
from pbg.games.poker.bundle import estimate_rollout_memory
from pbg.games.poker.adapter import PokerModelBundle
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet
from pbg.games.poker.rollout_backend import PokerRolloutBackend


EXPECTED_ORDERS = {
    4: ("quads", "full_house", "trips", "two_pair", "pair", "high_card"),
    5: (
        "straight_flush", "flush", "quads", "full_house", "straight",
        "trips", "two_pair", "pair", "high_card",
    ),
    6: (
        "straight_flush", "flush", "quads", "full_house", "straight",
        "trips", "two_pair", "pair", "high_card",
    ),
    7: (
        "straight_flush", "flush", "quads", "full_house", "straight",
        "trips", "two_pair", "pair", "high_card",
    ),
    **{
        rank_count: (
            "straight_flush", "quads", "flush", "full_house", "straight",
            "trips", "two_pair", "pair", "high_card",
        )
        for rank_count in range(8, 12)
    },
    **{
        rank_count: (
            "straight_flush", "quads", "full_house", "flush", "straight",
            "trips", "two_pair", "pair", "high_card",
        )
        for rank_count in (12, 13)
    },
}


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_compact_geometry_is_a_bijection(rank_count: int) -> None:
    geometry = DeckGeometry(rank_count)
    assert geometry.n_cards == 4 * rank_count
    assert geometry.n_hands == math.comb(4 * rank_count, 2)
    assert geometry.n_classes == rank_count**2
    assert geometry.active_cards == tuple(range(52 - 4 * rank_count, 52))
    assert len(set(geometry.combo_cards)) == geometry.n_hands
    assert sorted(hand for members in geometry.class_memberships for hand in members) == list(
        range(geometry.n_hands)
    )
    for hand, cards in enumerate(geometry.combo_cards):
        assert geometry.hand_index(*cards) == hand
        assert geometry.hand_cards(hand) == cards


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_category_counts_and_orders(rank_count: int) -> None:
    profile = category_profile(rank_count)
    r = rank_count
    total = math.comb(4 * r, 5)
    sequences = max(0, r - 4) + int(r == 13)
    choose_r_5 = math.comb(r, 5) if r >= 5 else 0
    assert profile.denominator == total
    assert profile.numerators == {
        "high_card": total,
        "pair": total - 4**5 * choose_r_5,
        "two_pair": 144 * (r - 2) * math.comb(r, 2) + 24 * r * (r - 1),
        "trips": 64 * r * math.comb(r - 1, 2) + 28 * r * (r - 1),
        "straight": 4**5 * sequences,
        "flush": 4 * choose_r_5,
        "full_house": 24 * r * (r - 1),
        "quads": 4 * r * (r - 1),
        "straight_flush": 4 * sequences,
    }
    assert profile.strength_order == EXPECTED_ORDERS[rank_count]


def test_documented_probability_degeneracies() -> None:
    four = category_profile(4)
    assert four.probability("high_card") == 1
    assert four.probability("pair") == 1
    assert four.numerators["straight"] == four.numerators["flush"] == 0

    five = category_profile(5)
    assert five.numerators["flush"] == five.numerators["straight_flush"]

    eight = category_profile(8)
    assert eight.numerators["flush"] == eight.numerators["quads"] == 224

    twelve = category_profile(12)
    assert twelve.numerators["flush"] == twelve.numerators["full_house"] == 3168


def test_board_conflicts_reject_inactive_cards() -> None:
    geometry = DeckGeometry(4)
    board = torch.tensor([36, 40, 44, -1, -1], dtype=torch.int16)
    mask = geometry.board_conflict_mask(board)
    assert mask.shape == (120,)
    assert int(mask.sum()) == 42
    with pytest.raises(ValueError, match="outside the active deck"):
        geometry.board_conflict_mask(torch.tensor([35, -1, -1, -1, -1]))


@pytest.mark.parametrize("players,stack,ranks", [(2, 8.0, 4), (6, 8.0, 5), (6, 128.0, 13)])
def test_supported_configuration_corners(players: int, stack: float, ranks: int) -> None:
    config = PokerConfig(n_players=players, stack_bb=stack, rank_count=ranks)
    assert config.geometry.rank_count == ranks
    assert config.n_ticks == round(2 * stack)
    assert (config.sb, config.bb, config.chip_unit_bb) == (0.5, 1.0, 0.5)


@pytest.mark.parametrize(
    "players,rank_count",
    [
        (players, rank_count)
        for rank_count in range(4, 14)
        for players in range(2, 7)
        if 2 * players + 5 < 4 * rank_count
    ],
)
def test_every_legal_player_and_rank_count_pair_builds_compact_rollout_state(
    players: int, rank_count: int
) -> None:
    config = PokerConfig(n_players=players, stack_bb=8, rank_count=rank_count)
    geometry = config.geometry
    backend = PokerRolloutBackend(config, 2, "cpu")
    state = backend.sequential_start_states(1)

    assert state.shape == (1,)
    assert backend.n_cards == geometry.n_cards
    assert backend.n_hands == geometry.n_hands
    assert backend.n_classes == geometry.n_classes
    assert backend.x_sizes == (geometry.n_hands,) * players


@pytest.mark.parametrize(
    "values,match",
    [
        ({"n_players": 1}, "n_players"),
        ({"n_players": 7}, "n_players"),
        ({"stack_bb": 7.5}, "stack_bb"),
        ({"stack_bb": 128.5}, "stack_bb"),
        ({"stack_bb": 8.25}, "multiple of 0.5"),
        ({"rank_count": 3}, "rank_count"),
        ({"rank_count": 14}, "rank_count"),
        ({"n_players": 6, "rank_count": 4}, "2\\*n_players"),
    ],
)
def test_configuration_rejects_out_of_contract_values(
    values: dict[str, int | float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        PokerConfig(**values)


def test_rollout_memory_uses_compact_private_width() -> None:
    short = estimate_rollout_memory(
        root_count=2, node_budget=8, n_players=2, rank_count=4
    )
    full = estimate_rollout_memory(
        root_count=2, node_budget=8, n_players=2, rank_count=13
    )
    assert short.reach_cache_bytes * 1326 == full.reach_cache_bytes * 120
    assert short.frontier_reach_bytes * 1326 == full.frontier_reach_bytes * 120


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_every_rank_count_runs_preflop_and_postflop_policy_boundaries(
    rank_count: int,
) -> None:
    game_cfg = PokerConfig(n_players=2, stack_bb=8, rank_count=rank_count)
    geometry = game_cfg.geometry
    backend = PokerRolloutBackend(game_cfg, 16, "cpu")
    model_cfg = PokerModelConfig(
        d_model=4,
        n_heads=1,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=1,
        k_act=4,
        max_players=2,
        class_embedding_dim=2,
        ffn_hidden=4,
        folder_beta_rank=min(4, geometry.n_cards),
    )
    model = PokerModelBundle(
        PreflopNet(model_cfg, geometry),
        PostflopNet(model_cfg, geometry),
    ).eval()
    state = backend.sequential_start_states(1)
    reach = torch.zeros((1, 2, geometry.n_hands), dtype=torch.float32)
    with torch.no_grad():
        preflop = backend.policy_forward(model, state, reach)
    assert preflop.v.shape == (1, geometry.n_hands)
    assert all(
        value.shape[:2] == (1, geometry.n_hands)
        for value in preflop.raw.values()
    )

    for _ in range(2):
        state = backend.step(
            state,
            EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32)),
        )
    cards = torch.tensor([geometry.active_cards[:3]], dtype=torch.int16)
    chance_event = EventBatch(
        torch.zeros(1, dtype=torch.int32), {"cards": cards}
    )
    reach = backend.reach_update(reach, state, chance_event, None)
    state = backend.step(state, chance_event)
    with torch.no_grad():
        postflop = backend.policy_forward(model, state, reach)
    assert postflop.v.shape == (1, geometry.n_hands)
    assert all(
        value.shape[:2] == (1, geometry.n_hands)
        for value in postflop.raw.values()
    )


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_every_rank_count_runs_folded_partition_boundary(rank_count: int) -> None:
    pytest.importorskip("poker_gpu_evaluator.pbg")
    geometry = DeckGeometry(rank_count)
    board = torch.tensor(
        [[*geometry.active_cards[:3], -1, -1]], dtype=torch.int16
    )
    fold_board = torch.tensor(
        [[[*geometry.active_cards[:3], 255, 255]]], dtype=torch.uint8
    )
    live = torch.ones((1, 2, geometry.n_hands), dtype=torch.float32)
    folded = FoldSnapshotBatch(
        reach_weights=torch.ones((1, 1, geometry.n_hands), dtype=torch.float32),
        board_at_fold=fold_board,
        line_meta=torch.zeros((1, 1, 3), dtype=torch.float32),
        folder_mask=torch.ones((1, 1), dtype=torch.bool),
        seat_ids=torch.tensor([[1]], dtype=torch.int64),
    )

    summary = PokerGPUPartitionAdapter(
        beta_rank=min(4, geometry.n_cards), rank_count=rank_count
    ).summarize(
        board_cards=board,
        live_range_weights=live,
        folded=folded,
    )

    assert summary.fields["single_card_log_ratio"].shape == (1, geometry.n_cards)
    assert summary.fields["pair_card_log_ratio"].shape == (
        1,
        geometry.n_cards,
        geometry.n_cards,
    )
    assert summary.fields["folder_card_marginal"].shape == (
        1,
        1,
        geometry.n_cards,
    )
    assert summary.fields["output_valid"].tolist() == [True]

    masked = FoldSnapshotBatch(
        reach_weights=folded.reach_weights,
        board_at_fold=folded.board_at_fold,
        line_meta=folded.line_meta,
        folder_mask=torch.zeros_like(folded.folder_mask),
        seat_ids=folded.seat_ids,
    )
    neutral = PokerGPUPartitionAdapter(
        beta_rank=min(4, geometry.n_cards), rank_count=rank_count
    ).summarize(
        board_cards=board,
        live_range_weights=live,
        folded=masked,
    )
    assert torch.count_nonzero(neutral.fields["single_card_log_ratio"]) == 0
    assert torch.count_nonzero(neutral.fields["pair_card_log_ratio"]) == 0


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_every_rank_count_completes_a_boundary_training_iteration(
    rank_count: int, tmp_path: Path
) -> None:
    root = Path(__file__).parents[1]
    raw = yaml.safe_load(
        (root / "configs" / "acceptance" / "nlhe_hu_normal.yaml").read_text()
    )
    raw["game"]["rank_count"] = rank_count
    raw["rollout"].update(
        {
            "sequential_roots_per_gpu_initial": 2,
            "node_budget": 32,
            "family_batch_rows": 2,
            "max_active_trajectories": 2,
            "arena_chunk_nodes": 32,
        }
    )
    raw["completion"]["backend"] = "cpu"
    raw["model"] = {
        "d_model": 4,
        "n_heads": 1,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 2,
        "ffn_hidden": 4,
        "folder_beta_rank": 4,
    }
    raw["training"]["minibatch_size"] = 2
    raw["training"]["out_dir"] = str(tmp_path / f"rank_{rank_count}")
    path = tmp_path / f"rank_{rank_count}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(path)
    train(cfg)
    records = [
        json.loads(line)
        for line in (Path(cfg.training.out_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert math.isfinite(records[0]["loss"])
    assert records[0]["optimizer_steps"] == 1.0


@pytest.mark.parametrize("rank_count", range(4, 14))
def test_every_rank_count_completes_a_rao_blackwell_training_iteration(
    rank_count: int, tmp_path: Path
) -> None:
    root = Path(__file__).parents[1]
    raw = yaml.safe_load(
        (root / "configs" / "acceptance" / "nlhe_hu_sequential_rb.yaml").read_text()
    )
    raw["game"]["rank_count"] = rank_count
    raw["rollout"].update(
        {
            "sequential_roots_per_gpu_initial": 2,
            "node_budget": 32,
            "family_batch_rows": 2,
            "max_active_trajectories": 2,
            "arena_chunk_nodes": 32,
        }
    )
    raw["completion"].update({"backend": "cpu", "n_samples": 2})
    raw["model"] = {
        "d_model": 4,
        "n_heads": 1,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 2,
        "ffn_hidden": 4,
        "folder_beta_rank": 4,
    }
    raw["training"]["minibatch_size"] = 2
    raw["training"]["out_dir"] = str(tmp_path / f"rb_rank_{rank_count}")
    path = tmp_path / f"rb_rank_{rank_count}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(path)
    train(cfg)
    record = json.loads(
        (Path(cfg.training.out_dir) / "metrics.jsonl").read_text().splitlines()[0]
    )
    assert math.isfinite(record["loss"])
    assert record["optimizer_steps"] == 1.0
    assert record["completion_mean_n_eff"] > 1.99
