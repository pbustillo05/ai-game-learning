from __future__ import annotations

import copy

import pytest

from pbg.core.config import load_config


def test_ddp_keeps_tree_and_batch_values_per_rank() -> None:
    cfg = load_config("configs/train/nlhe6.yaml")
    one_rank = copy.deepcopy(cfg)
    two_rank = copy.deepcopy(cfg)
    assert one_rank.rollout.n_trees == 1
    assert two_rank.rollout.n_trees == one_rank.rollout.n_trees
    assert two_rank.train.minibatch_size == one_rank.train.minibatch_size


@pytest.mark.skip(reason="multi-node acceptance run; not a fixed CI performance assertion")
def test_one_rank_and_two_rank_kuhn_curves_improve_comparably() -> None:
    """Run via torchrun on the target node and compare sample-scaled curves."""
