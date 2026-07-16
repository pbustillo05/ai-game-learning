from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pbg.core.game import GameBundle, PRETERM, TERMINAL
from pbg.core.rollout import sample_trees
from pbg.games.synthetic.family import SyntheticPBG, SyntheticRolloutBackend
from pbg.games.synthetic.exact import enumerate_public_tree
from pbg.games.synthetic.model import (
    SyntheticNet,
    SyntheticPolicyAdapter,
    export_policy_tables,
)
from pbg.games.synthetic.presets import kuhn, leduc


def _cfg(mode: str, n_trees: int = 256):
    return SimpleNamespace(
        n_trees=n_trees,
        beta_mode="sequential",
        node_budget=128,
        reach_storage=mode,
    )


def _rollout(game, net, mode: str, seed: int = 811, n_trees: int = 256):
    bundle = GameBundle(
        net, SyntheticRolloutBackend(game), SyntheticPolicyAdapter(), None
    )
    return sample_trees(
        bundle, _cfg(mode, n_trees), torch.Generator().manual_seed(seed)
    )


def _leaf_ids(tree):
    return ((tree.owner == PRETERM) | (tree.owner == TERMINAL)).nonzero(
        as_tuple=False
    ).squeeze(1)


def test_sequential_rollout_is_one_path_per_tree_and_stops_at_valid_leaves() -> None:
    torch.manual_seed(810)
    for preset in (kuhn, leduc):
        game = SyntheticPBG(preset())
        net = SyntheticNet(game.n_types, game.n_player_events, {"n_buckets": 256})
        tree, _, _ = _rollout(game, net, "full")
        counts = tree.child_ptr[1:] - tree.child_ptr[:-1]
        assert bool((counts <= 1).all())
        assert torch.equal(tree.nu, torch.ones_like(tree.nu))
        leaves = _leaf_ids(tree)
        assert leaves.numel() == 256
        assert tree.tree_id[leaves].sort().values.tolist() == list(range(256))


def test_reach_modes_agree_exactly_and_policy_cache_matches_adapter() -> None:
    torch.manual_seed(812)
    game = SyntheticPBG(leduc())
    net = SyntheticNet(
        game.n_types, game.n_player_events, {"n_buckets": 512, "hidden_dim": 16}
    )
    results = []
    for mode in ("full", "decisions_only", "recompute"):
        tree, reach, policy = _rollout(game, net, mode)
        leaves = _leaf_ids(tree)
        results.append((tree, reach.rows_for(tree, leaves, SyntheticRolloutBackend(game), policy), policy))
    reference_tree, reference_rows, _ = results[0]
    for tree, rows, _ in results[1:]:
        assert torch.equal(tree.state_ref, reference_tree.state_ref)
        assert torch.equal(tree.child_event, reference_tree.child_event)
        assert torch.equal(rows, reference_rows)

    tree, _, policy = results[0]
    node_ids = policy.node_ids.long()
    refs = tree.state_ref[node_ids]
    legal_ids, legal_mask = game.legal_events(refs)
    head_mask = torch.zeros((len(refs), game.n_player_events), dtype=torch.bool)
    batch = torch.arange(len(refs))[:, None].expand_as(legal_ids)
    head_mask[batch[legal_mask], legal_ids[legal_mask].long()] = True
    with torch.no_grad():
        output = net(game.encode_public(refs), head_mask)
        edge = tree.child_ptr[node_ids]
        events = tree.child_event[edge.long()]
        recomputed = SyntheticPolicyAdapter().log_prob(
            output, events[:, None], legal_ids, legal_mask
        )[:, 0]
    assert torch.allclose(policy.old_logp_child[:, 0], recomputed, atol=1e-6, rtol=0.0)
    assert torch.allclose(policy.old_v, output.v, atol=1e-6, rtol=0.0)


def test_50k_sequential_terminal_history_law_matches_exact_policy() -> None:
    torch.manual_seed(813)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(
        game.n_types, game.n_player_events, {"n_buckets": 64, "hidden_dim": 8}
    )
    exact_tree = enumerate_public_tree(game)
    tables = export_policy_tables(net, game, exact_tree)
    profiles = np.asarray(
        [(first, second) for first in range(3) for second in range(3) if first != second],
        dtype=np.int64,
    )
    expected: dict[int, float] = {}

    def walk(index: int, profile_mass: np.ndarray) -> None:
        owner = int(exact_tree.owners[index])
        if owner == TERMINAL:
            expected[int(exact_tree.node_keys[index])] = float(profile_mass.sum())
            return
        table = tables[int(exact_tree.node_keys[index])]
        for event, child in exact_tree.children[index]:
            likelihood = table[profiles[:, owner], event]
            walk(child, profile_mass * likelihood)

    walk(0, np.full(len(profiles), 1.0 / len(profiles), dtype=np.float64))
    tree, _, _ = _rollout(game, net, "recompute", seed=813, n_trees=50_000)
    leaves = _leaf_ids(tree)
    keys = game.encode_public(tree.state_ref[leaves]).cpu().numpy()
    observed = np.asarray([(keys == key).sum() for key in expected], dtype=np.float64)
    target = 50_000 * np.asarray(list(expected.values()), dtype=np.float64)
    statistic = float((np.square(observed - target) / target).sum())
    # df=4: chi-square survival probability exceeds 1e-3 below 18.467.
    assert statistic < 18.467


def test_sequential_rejects_invalid_mode_and_enforces_node_budget() -> None:
    torch.manual_seed(814)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(game.n_types, game.n_player_events)
    invalid = _cfg("full", 1)
    invalid.beta_mode = "neyman"
    bundle = GameBundle(net, SyntheticRolloutBackend(game), SyntheticPolicyAdapter(), None)
    try:
        sample_trees(bundle, invalid, torch.Generator().manual_seed(815))
    except ValueError as exc:
        assert "sequential" in str(exc) and "fixed" in str(exc)
    else:
        raise AssertionError("rollout accepted an invalid beta mode")

    tiny = _cfg("full", 1)
    tiny.node_budget = 1
    try:
        sample_trees(bundle, tiny, torch.Generator().manual_seed(816))
    except RuntimeError as exc:
        assert "capacity nodes" in str(exc)
    else:
        raise AssertionError("rollout ignored its hard node budget")
