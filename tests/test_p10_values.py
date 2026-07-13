from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pbg.core.completion import BatchedCompletionStats
from pbg.core.game import CHANCE, TERMINAL
from pbg.core.tree import TreeBuilder
from pbg.core.values import (
    HeadEvals,
    branched_value,
    branched_value_reference,
)


@dataclass
class _Game:
    n_players: int = 2
    x_sizes: tuple[int, int] = (2, 3)


def _branched_fixture(moment_heads: bool = True):
    builder = TreeBuilder(2)
    # Tree zero contains both chance aggregation and repeated decisions by p0.
    root = builder.add_root(0, 0, tree_id=0)
    left = builder.add_child(root, 1, 10, 0.25, CHANCE)
    right = builder.add_child(root, 2, 11, 0.75, 1)
    deep = builder.add_child(left, 3, 12, 1.0, 0)
    builder.add_child(deep, 4, 13, 0.4, TERMINAL)
    builder.add_child(deep, 5, 14, 0.6, TERMINAL)
    builder.add_child(right, 6, 15, 1.0, TERMINAL)
    # A second tree makes the max-resident-rows path observable.
    other = builder.add_root(7, 1, tree_id=1)
    builder.add_child(other, 8, 16, 1.0, TERMINAL)
    tree = builder.finalize()

    gen = torch.Generator().manual_seed(1001)
    leaves = int((tree.child_ptr[1:] == tree.child_ptr[:-1]).sum())
    p_hat = torch.rand((leaves, 2, 3), generator=gen, dtype=torch.float64)
    v_hat = torch.randn((leaves, 2, 3), generator=gen, dtype=torch.float64)
    # Player zero has only two valid types; poison padding to verify masking.
    p_hat[:, 0, 2] = 999.0
    v_hat[:, 0, 2] = 999.0
    stats = BatchedCompletionStats(
        p_hat=p_hat,
        v_hat=v_hat,
        vvec_hat=torch.randn((leaves, 2), generator=gen, dtype=torch.float64),
        u_hat=torch.rand(leaves, generator=gen, dtype=torch.float64),
        n_eff=torch.ones(leaves, dtype=torch.float64),
    )
    decision_ids = ((tree.owner >= 0) & (tree.owner < 2)).nonzero(as_tuple=False).squeeze(1)
    permutation = torch.tensor([2, 0, 3, 1])
    head_ids = decision_ids[permutation]
    head_v = torch.randn((4, 3), generator=gen, dtype=torch.float64)
    head_v[:, 2] = 777.0
    heads = HeadEvals(
        node_ids=head_ids.to(torch.int32),
        v=head_v,
        vvec=torch.randn((4, 2), generator=gen, dtype=torch.float64)
        if moment_heads
        else None,
        u=torch.randn(4, generator=gen, dtype=torch.float64) if moment_heads else None,
    )
    return tree, stats, heads, (root, deep)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("moment_heads", [False, True])
def test_vectorized_pass_matches_independent_fp64_reference(lam, moment_heads) -> None:
    tree, stats, heads, _ = _branched_fixture(moment_heads)
    actual = branched_value(tree, stats, None, heads, lam, _Game())
    expected = branched_value_reference(tree, stats, None, heads, lam, _Game())
    for field in ("p_hat", "q_hat", "v_hat", "vvec_hat", "u_hat", "nu_row"):
        assert torch.allclose(getattr(actual, field), getattr(expected, field), atol=1e-12, rtol=0)
    assert torch.equal(actual.node_ids, expected.node_ids)
    assert torch.equal(actual.child_event_row, expected.child_event_row)
    assert torch.equal(actual.p_hat[:, 2], torch.where(tree.owner[actual.node_ids.long()] == 0, 0.0, actual.p_hat[:, 2]))


def test_sequential_lambda_return_and_capture_then_mix_boundary() -> None:
    tree, stats, heads, (root, deep) = _branched_fixture(False)
    lam = 0.37
    targets = branched_value(tree, stats, None, heads, lam, _Game())
    row_for = {int(node): row for row, node in enumerate(targets.node_ids.tolist())}
    head_for = {int(node): row for row, node in enumerate(heads.node_ids.tolist())}
    deep_row = row_for[deep]
    deep_leaf_return = (
        targets.nu_row[deep_row, :, None] * targets.q_hat[deep_row]
    ).sum(0)
    expected_root_left = lam * deep_leaf_return + (1.0 - lam) * heads.v[head_for[deep]]
    expected_root_left[2] = 0.0
    assert torch.allclose(targets.q_hat[row_for[root], 0], expected_root_left, atol=1e-12, rtol=0)

    zero = branched_value(tree, stats, None, heads, 0.0, _Game())
    assert torch.equal(zero.q_hat[row_for[root], 0, :2], heads.v[head_for[deep], :2])
    assert zero.q_hat[row_for[root], 0, 2] == 0


def test_lambda_one_ignores_heads_and_tiny_memory_cap_matches() -> None:
    tree, stats, heads, _ = _branched_fixture(True)
    base = branched_value(tree, stats, None, heads, 1.0, _Game())
    perturbed = HeadEvals(
        heads.node_ids,
        heads.v + 10_000,
        heads.vvec - 20_000,
        heads.u + 30_000,
    )
    changed = branched_value(tree, stats, None, perturbed, 1.0, _Game(), max_resident_rows=1)
    for field in ("p_hat", "q_hat", "v_hat", "vvec_hat", "u_hat", "nu_row"):
        assert torch.equal(getattr(base, field), getattr(changed, field))


def test_rejects_half_precision_accumulators() -> None:
    tree, stats, heads, _ = _branched_fixture(False)
    stats.p_hat = stats.p_hat.half()
    with pytest.raises(TypeError, match="fp16"):
        branched_value(tree, stats, None, heads, 0.5, _Game())
