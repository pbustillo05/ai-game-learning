from __future__ import annotations

import torch
import pytest

from pbg.games.synthetic.exact import enumerate_public_tree
from pbg.games.synthetic.family import SyntheticPBG
from pbg.games.synthetic.model import (
    SyntheticNet,
    SyntheticPolicyAdapter,
    export_policy_tables,
)
from pbg.games.synthetic.presets import kuhn


def test_adapter_log_prob_normalizes_and_entropy_is_exact() -> None:
    net = SyntheticNet(3, 3, {"n_buckets": 32, "hidden_dim": 16})
    keys = torch.tensor([3, 7], dtype=torch.int64)
    legal = torch.tensor([[False, True, True], [True, True, False]])
    output = net(keys, legal)
    adapter = SyntheticPolicyAdapter()
    selected = torch.tensor([[1, 2], [0, 1]], dtype=torch.int64)
    ids = torch.arange(3, dtype=torch.int32).expand(2, -1)
    log_prob = adapter.log_prob(output, selected, ids, legal)
    assert log_prob.shape == (2, 2, 3)
    assert torch.allclose(
        torch.logsumexp(log_prob, dim=1), torch.zeros((2, 3)), atol=1e-6
    )

    probability = torch.softmax(output.raw, dim=-1)
    log_policy = torch.log_softmax(output.raw, dim=-1)
    numerical = -torch.where(
        probability > 0.0, probability * log_policy, torch.zeros_like(probability)
    ).sum(dim=-1)
    assert torch.allclose(adapter.entropy(output, ids, legal), numerical)
    assert torch.allclose(adapter.kl(output, output), torch.zeros((2, 3)), atol=1e-7)


def test_synthetic_public_embedding_rejects_aliasing_keys() -> None:
    net = SyntheticNet(3, 3, {"n_buckets": 8, "hidden_dim": 4})
    legal = torch.ones((1, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="collision-free"):
        net(torch.tensor([8], dtype=torch.int64), legal)


def test_exported_model_policy_tables_are_legal_and_normalized() -> None:
    game = SyntheticPBG(kuhn())
    tree = enumerate_public_tree(game)
    net = SyntheticNet(game.n_types, game.n_player_events, {"n_buckets": 64})
    tables = export_policy_tables(net, game, tree)
    assert len(tables) == 4
    for table in tables.values():
        assert table.shape == (game.n_types, game.n_player_events)
        assert torch.allclose(
            torch.from_numpy(table.sum(axis=-1)), torch.ones(game.n_types, dtype=torch.float64)
        )
