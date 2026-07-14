from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pbg.core.completion import BatchedCompletionStats
from pbg.core.game import TERMINAL
from pbg.core.posterior_values import NodeTargets as CorrectedTargets
from pbg.core.tree import TreeBuilder
from pbg.core.values import HeadEvals, NodeTargets, branched_value, streamed_branched_value


@dataclass
class _Game:
    n_players: int = 2
    x_sizes: tuple[int, int] = (2, 3)


def _fixture():
    builder = TreeBuilder(2)
    root = builder.add_root(0, 0, tree_id=0)
    left = builder.add_child(root, 1, 10, 0.25, 1)
    builder.add_child(root, 2, 11, 0.75, TERMINAL)
    builder.add_child(left, 3, 12, 0.5, TERMINAL)
    builder.add_child(left, 4, 13, 0.5, TERMINAL)
    other = builder.add_root(5, 1, tree_id=1)
    builder.add_child(other, 6, 14, 1.0, TERMINAL)
    tree = builder.finalize()
    p_hat = torch.tensor(
        [
            [[0.8, 0.2, 999.0], [0.2, 0.3, 0.5]],
            [[0.1, 0.9, 999.0], [0.4, 0.4, 0.2]],
            [[0.6, 0.4, 999.0], [0.1, 0.2, 0.7]],
            [[0.5, 0.5, 999.0], [0.3, 0.3, 0.4]],
        ]
    )
    v_hat = torch.tensor(
        [
            [[2.0, -1.0, 999.0], [1.0, 2.0, 3.0]],
            [[-2.0, 4.0, 999.0], [3.0, 2.0, 1.0]],
            [[1.0, 3.0, 999.0], [-1.0, 0.0, 2.0]],
            [[-1.0, 1.0, 999.0], [2.0, -2.0, 1.0]],
        ]
    )
    stats = BatchedCompletionStats(
        p_hat=p_hat,
        v_hat=v_hat,
        vvec_hat=torch.zeros((4, 2)),
        u_hat=torch.zeros(4),
        n_eff=torch.ones(4),
    )
    decisions = ((tree.owner >= 0) & (tree.owner < 2)).nonzero().squeeze(1)
    heads = HeadEvals(decisions.to(torch.int32), torch.tensor(
        [[0.5, -0.5, 777.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]
    ))
    return tree, stats, heads


def _run(lam: float, **kwargs) -> NodeTargets:
    tree, stats, heads = _fixture()
    return branched_value(
        tree, stats, None, heads, lam, _Game(), configured_node_budget=5,
        emit_q=True, **kwargs,
    )


def test_values_module_is_the_corrected_public_api_and_masks_padding() -> None:
    assert NodeTargets is CorrectedTargets
    targets = _run(0.5)
    tree, _, _ = _fixture()
    actors = tree.owner[targets.node_ids.long()]
    assert torch.equal(
        targets.parent_p_hat[:, 2],
        torch.where(actors == 0, 0.0, targets.parent_p_hat[:, 2]),
    )
    assert targets.q_hat is not None
    padding = (~targets.child_mask[:, :, None]).expand_as(targets.a_hat)
    assert torch.equal(targets.a_hat.masked_select(padding), torch.zeros(3))


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_lambda_boundaries_are_finite_and_q_is_opt_in(lam: float) -> None:
    tree, stats, heads = _fixture()
    targets = branched_value(
        tree, stats, None, heads, lam, _Game(), configured_node_budget=5
    )
    assert targets.q_hat is None
    assert torch.isfinite(targets.a_hat).all()
    assert torch.isfinite(targets.v_hat).all()


def test_streamed_leaf_batches_match_full_completion() -> None:
    tree, stats, heads = _fixture()
    base = _run(0.95)
    leaves = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero().squeeze(1)
    batches = [
        (
            leaves[first : first + 1].to(torch.int32),
            BatchedCompletionStats(
                p_hat=stats.p_hat[first : first + 1],
                v_hat=stats.v_hat[first : first + 1],
            ),
        )
        for first in range(leaves.numel())
    ]
    chunked = streamed_branched_value(
        tree,
        leaves,
        batches,
        None,
        heads,
        0.95,
        _Game(),
        configured_node_budget=5,
        completion_batch_rows=1,
        emit_q=True,
    )
    for field in ("parent_p_hat", "child_p_hat", "a_hat", "v_hat", "q_hat"):
        assert torch.equal(getattr(base, field), getattr(chunked, field))


def test_rejects_half_precision_accumulators() -> None:
    tree, stats, heads = _fixture()
    stats.p_hat = stats.p_hat.half()
    with pytest.raises(TypeError, match="fp16"):
        branched_value(
            tree, stats, None, heads, 0.5, _Game(), configured_node_budget=5
        )
