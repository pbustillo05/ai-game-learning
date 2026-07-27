from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from pbg.config import ConfigError, load_config
from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch
from pbg.games.poker.bundle import build_bundle
from pbg.games.poker.rules import CHECK_CALL, LIVE, RAISE_OFFSET, PokerConfig
from pbg.games.poker.model.capacity import (
    max_player_nodes,
    max_policy_partition_sequence_tokens,
    max_policy_sequence_tokens,
    max_public_history_tokens,
)
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig
from pbg.games.poker.rollout_backend import PokerRolloutBackend


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("players", range(2, 7))
@pytest.mark.parametrize("stack", (8.0, 8.5, 128.0))
def test_lossless_history_bound_covers_documented_player_nodes_and_separators(
    players: int, stack: float
) -> None:
    expected_nodes = math.ceil(stack * players + 3 * players - stack + 1)
    assert max_player_nodes(players, stack) == expected_nodes
    assert max_public_history_tokens(players, stack) == expected_nodes + 3


def test_partition_attention_bounds_preserve_the_global_worst_case() -> None:
    bounds = max_policy_partition_sequence_tokens(
        n_players=2, stack_bb=32.0, rank_count=13, k_act=7
    )
    assert bounds == {"pre": 538, "post_2": 2557}
    assert max(bounds.values()) == max_policy_sequence_tokens(
        n_players=2, stack_bb=32.0, rank_count=13, k_act=7
    )



def test_explicitly_small_history_capacity_is_a_dotted_load_error(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["model"]["max_history_tokens"] = 521
    path = tmp_path / "short_history.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"root\.model\.max_history_tokens"):
        load_config(path)


def test_worst_history_attention_footprint_is_checked_against_profile(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["game"]["stack_bb"] = 128.0
    path = tmp_path / "deep.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    width = max_policy_sequence_tokens(
        n_players=6, stack_bb=128.0, rank_count=13, k_act=7
    )
    assert width > 8192
    with pytest.raises(ConfigError, match=r"hardware\.training_max_tokens"):
        load_config(path, hardware_profile="quest_slurm")
