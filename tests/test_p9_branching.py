from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pbg.core.game import GameBundle, PRETERM, TERMINAL
from pbg.core.rollout import sample_trees
from pbg.games.synthetic.exact import enumerate_public_tree
from pbg.games.synthetic.family import SyntheticPBG, SyntheticRolloutBackend
from pbg.games.synthetic.model import (
    SyntheticNet,
    SyntheticPolicyAdapter,
    export_policy_tables,
)
from pbg.games.synthetic.presets import kuhn


def _cfg(mode: str, n_trees: int, beta: float = 1.0, budget: int = 128):
    return SimpleNamespace(
        n_trees=n_trees,
        beta_mode=mode,
        beta_fix={"default": beta},
        node_budget=budget,
        reach_storage="recompute",
    )


def _bundle(game: SyntheticPBG, net: SyntheticNet, beta: float) -> GameBundle:
    return GameBundle(
        model=net,
        rollout_backend=SyntheticRolloutBackend(game, {"default": beta}),
        policy_adapter=SyntheticPolicyAdapter(),
        completion_backend=None,
    )


def _sample(game: SyntheticPBG, net: SyntheticNet, cfg: SimpleNamespace, seed: int):
    return sample_trees(
        _bundle(game, net, float(cfg.beta_fix["default"])),
        cfg,
        torch.Generator().manual_seed(seed),
    )


def _terminal_probabilities(game: SyntheticPBG, net: SyntheticNet) -> dict[int, float]:
    index = enumerate_public_tree(game)
    tables = export_policy_tables(net, game, index)
    profiles = np.asarray(
        [(a, b) for a in range(3) for b in range(3) if a != b], dtype=np.int64
    )
    expected: dict[int, float] = {}

    def walk(node: int, mass: np.ndarray) -> None:
        owner = int(index.owners[node])
        if owner == TERMINAL:
            expected[int(index.node_keys[node])] = float(mass.sum())
            return
        table = tables[int(index.node_keys[node])]
        for event, child in index.children[node]:
            walk(child, mass * table[profiles[:, owner], event])

    walk(0, np.full(6, 1.0 / 6.0, dtype=np.float64))
    return expected


def test_sequential_mode_is_byte_deterministic_for_five_seeds() -> None:
    torch.manual_seed(900)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(game.n_types, game.n_player_events, {"hidden_dim": 8})
    fields = (
        "parent", "child_ptr", "child_node", "child_event", "nu", "tree_id",
        "owner", "depth_p", "depth_player", "state_ref", "path_weight",
    )
    for seed in range(901, 906):
        left = _sample(game, net, _cfg("sequential", 64), seed)[0]
        right = _sample(game, net, _cfg("sequential", 64), seed)[0]
        for field in fields:
            assert torch.equal(getattr(left, field), getattr(right, field))


def test_fixed_beta_one_has_the_sequential_terminal_law() -> None:
    torch.manual_seed(906)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(
        game.n_types, game.n_player_events, {"n_buckets": 64, "hidden_dim": 8}
    )
    expected = _terminal_probabilities(game, net)
    tree, _, _ = _sample(game, net, _cfg("fixed", 20_000), 907)
    leaves = ((tree.owner == TERMINAL) | (tree.owner == PRETERM)).nonzero(
        as_tuple=False
    ).squeeze(1)
    keys = game.encode_public(tree.state_ref[leaves]).numpy()
    observed = np.asarray([(keys == key).sum() for key in expected], dtype=np.float64)
    target = 20_000 * np.asarray(list(expected.values()))
    # df=4 chi-square 0.001 upper critical value; H0 failure probability < .001.
    assert float((np.square(observed - target) / target).sum()) < 18.467
    assert torch.equal(tree.nu, torch.ones_like(tree.nu))


def test_labeled_leaf_measure_matches_exact_public_law_at_beta_three() -> None:
    torch.manual_seed(908)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(
        game.n_types, game.n_player_events, {"n_buckets": 64, "hidden_dim": 8}
    )
    expected = _terminal_probabilities(game, net)
    n_trees = 5_000
    tree, _, _ = _sample(game, net, _cfg("fixed", n_trees, 3.0), 909)
    leaves = ((tree.owner == TERMINAL) | (tree.owner == PRETERM)).nonzero(
        as_tuple=False
    ).squeeze(1)
    keys = game.encode_public(tree.state_ref[leaves]).numpy()
    weights = tree.path_weight[leaves]
    observed = np.asarray(
        [float(weights[torch.from_numpy(keys == key)].sum()) for key in expected]
    )
    target = n_trees * np.asarray(list(expected.values()))
    # Importance-weighted labeled leaves retain the exact public terminal law.
    assert float((np.square(observed - target) / target).sum()) < 18.467
    assert abs(float(weights.sum()) - n_trees) < 0.05
