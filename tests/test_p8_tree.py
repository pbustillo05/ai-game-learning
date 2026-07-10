from __future__ import annotations

from dataclasses import replace

import torch

from pbg.core.game import CHANCE, PRETERM, TERMINAL
from pbg.core.tree import AliasTable, TreeBuilder, build_decision_alias


def _hand_tree():
    builder = TreeBuilder(2)
    root = builder.add_root(10, 0, 0)
    left = builder.add_child(root, 11, 4, 0.25, CHANCE)
    right = builder.add_child(root, 12, 5, 0.75, TERMINAL)
    builder.add_child(left, 13, 8, 1.0, PRETERM)
    return builder.finalize(), (root, left, right)


def test_builder_finalize_computes_csr_depth_and_path_weight() -> None:
    tree, (root, left, right) = _hand_tree()
    tree.validate()
    assert tree.parent.tolist() == [-1, root, root, left]
    assert tree.child_ptr.tolist() == [0, 2, 3, 3, 3]
    assert tree.child_node.tolist() == [left, right, 3]
    assert tree.child_event.tolist() == [4, 5, 8]
    assert torch.equal(
        tree.depth_p,
        torch.tensor([[0, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.int32),
    )
    assert tree.depth_player.tolist() == [0, 1, 1, 1]
    assert torch.equal(tree.path_weight, torch.tensor([1.0, 0.25, 0.75, 0.25]))


def test_validate_rejects_bad_nu_and_path_weight() -> None:
    tree, _ = _hand_tree()
    bad_nu = tree.nu.clone()
    bad_nu[0] = 0.5
    try:
        replace(tree, nu=bad_nu).validate()
    except ValueError as exc:
        assert "nu rows" in str(exc)
    else:
        raise AssertionError("invalid nu row was accepted")

    bad_path = tree.path_weight.clone()
    bad_path[1] = 0.5
    try:
        replace(tree, path_weight=bad_path).validate()
    except ValueError as exc:
        assert "path_weight" in str(exc)
    else:
        raise AssertionError("invalid path weight was accepted")


def test_alias_table_matches_weights_and_decision_alias_excludes_nonplayers() -> None:
    weights = torch.tensor([1.0, 2.0, 7.0], dtype=torch.float64)
    table = AliasTable.build(weights)
    n = 1_000_000
    draws = table.sample(n, torch.Generator().manual_seed(803))
    frequencies = torch.bincount(draws.long(), minlength=3).double() / n
    target = weights / weights.sum()
    # The 6e-3 bound is over 12 standard deviations for the rarest cell;
    # the seeded false-failure probability under H0 is negligible.
    assert torch.all((frequencies - target).abs() < 6e-3)

    tree, _ = _hand_tree()
    decision = build_decision_alias(tree)
    assert decision.outcomes.tolist() == [0]
    assert torch.equal(
        decision.sample(100, torch.Generator().manual_seed(804)),
        torch.zeros(100, dtype=torch.int32),
    )
