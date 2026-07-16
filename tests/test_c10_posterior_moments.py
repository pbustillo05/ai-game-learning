from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pbg.core.completion import BatchedCompletionStats
from pbg.core.game import TERMINAL
from pbg.core.values import (
    HeadEvals,
    streamed_branched_value,
)
from pbg.core.tree import TreeBatch, TreeBuilder


@dataclass
class _Game:
    n_players: int = 1
    x_sizes: tuple[int, ...] = (2,)


def _stats(p_hat: torch.Tensor, v_hat: torch.Tensor, **metadata):
    leaves = p_hat.shape[0]
    return BatchedCompletionStats(
        p_hat=p_hat.float(),
        v_hat=v_hat.float(),
        vvec_hat=torch.zeros((leaves, 1)),
        u_hat=torch.zeros(leaves),
        n_eff=torch.ones(leaves),
        metadata=metadata,
    )


def _targets(tree, stats, heads, lam, *, emit_q=False, max_workspace_bytes=1 << 30):
    leaves = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero().squeeze(1)
    return streamed_branched_value(
        tree,
        leaves,
        [(leaves.to(torch.int32), stats)],
        None,
        heads,
        lam,
        _Game(),
        configured_node_budget=3,
        completion_batch_rows=max(1, int(leaves.numel())),
        emit_q=emit_q,
        max_workspace_bytes=max_workspace_bytes,
    )


def test_completion_learning_view_releases_diagnostics_and_summarizes_quality() -> None:
    stats = BatchedCompletionStats(
        p_hat=torch.tensor([[[0.25, 0.75]]]),
        v_hat=torch.tensor([[[2.0, -1.0]]]),
        vvec_hat=torch.tensor([[-0.25]]),
        u_hat=torch.tensor([1.75]),
        n_eff=torch.tensor([7.0]),
        equity_share=torch.zeros((1, 1, 2, 3)),
        output_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        sample_count=torch.tensor([8]),
        estimator_stddev=torch.tensor([[[0.2, 0.4]]]),
    )
    learning = stats.for_learning()
    assert learning.p_hat is stats.p_hat and learning.v_hat is stats.v_hat
    for field in (
        "vvec_hat", "u_hat", "n_eff", "equity_share", "output_valid",
        "sample_count", "estimator_stddev",
    ):
        assert getattr(learning, field) is None
    assert learning.metadata["completion_min_n_eff"] == 7.0
    assert learning.metadata["completion_max_estimator_stddev"] == pytest.approx(0.4)
    with pytest.raises(RuntimeError, match="omit diagnostic moments"):
        learning.row(0)


def test_completion_learning_view_rejects_invalid_posterior_mass() -> None:
    stats = BatchedCompletionStats(
        p_hat=torch.tensor([[[0.25, 0.75]]]),
        v_hat=torch.zeros((1, 1, 2)),
        output_valid=torch.tensor([[[False, True]]]),
    )
    with pytest.raises(ValueError, match="posterior mass to invalid"):
        stats.for_learning()


def _posterior_shift_fixture():
    builder = TreeBuilder(1)
    root = builder.add_root(0, 0)
    # C10 averages abstract children uniformly; legacy empirical weights are
    # deliberately unequal here so accidentally reusing P10's ``nu`` fails.
    builder.add_child(root, 1, 7, 0.25, TERMINAL)
    builder.add_child(root, 2, 7, 0.75, TERMINAL)  # repeated label is distinct
    tree = builder.finalize()
    stats = _stats(
        torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        torch.tensor([[[2.0, 123.0]], [[456.0, 8.0]]]),
    )
    heads = HeadEvals(torch.tensor([root], dtype=torch.int32), torch.tensor([[10.0, 10.0]]))
    return tree, stats, heads


def test_posterior_moment_aggregation_and_leave_one_out() -> None:
    tree, stats, heads = _posterior_shift_fixture()
    target = _targets(tree, stats, heads, 0.5, emit_q=True)

    # Directly averaging the two conditional leaf values would give [1,4].
    assert torch.equal(target.parent_p_hat, torch.tensor([[0.5, 0.5]]))
    assert torch.equal(target.v_hat, torch.tensor([[2.0, 8.0]]))
    assert torch.equal(target.child_p_hat[0], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(target.q_hat[0], torch.tensor([[2.0, 0.0], [0.0, 8.0]]))
    assert torch.equal(target.a_hat[0], torch.tensor([[-3.0, -9.0], [-6.0, 3.0]]))
    assert torch.equal(target.child_event_row, torch.tensor([[7, 7]], dtype=torch.int32))
    assert torch.equal(target.child_mask, torch.tensor([[True, True]]))
    assert torch.equal(target.child_count, torch.tensor([2], dtype=torch.int32))


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_single_child_baseline_is_old_value(lam: float) -> None:
    builder = TreeBuilder(1)
    root = builder.add_root(0, 0)
    builder.add_child(root, 1, 3, 1.0, TERMINAL)
    tree = builder.finalize()
    stats = _stats(torch.tensor([[[0.25, 0.75]]]), torch.tensor([[[4.0, -2.0]]]))
    heads = HeadEvals(torch.tensor([root], dtype=torch.int32), torch.tensor([[1.0, 3.0]]))
    target = _targets(tree, stats, heads, lam)
    assert target.q_hat is None
    assert torch.allclose(target.a_hat[0, 0], torch.tensor([3.0, -5.0]))
    assert torch.allclose(target.v_hat[0], torch.tensor([4.0, -2.0]))
    assert (
        target.child_p_hat.untyped_storage().data_ptr()
        == target.parent_p_hat.untyped_storage().data_ptr()
    )


def test_frontier_byte_budget_is_loud() -> None:
    tree, stats, heads = _posterior_shift_fixture()
    target = _targets(tree, stats, heads, 0.95, max_workspace_bytes=1 << 20)
    assert torch.equal(target.parent_p_hat, torch.tensor([[0.5, 0.5]]))
    with pytest.raises(MemoryError, match="byte budget"):
        _targets(tree, stats, heads, 0.95, max_workspace_bytes=1)


def test_rejects_low_precision_leaf_accumulators() -> None:
    tree, stats, heads = _posterior_shift_fixture()
    stats.p_hat = stats.p_hat.half()
    with pytest.raises(TypeError, match="fp16"):
        _targets(tree, stats, heads, 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_small_cpu_cuda_parity() -> None:
    tree, stats, heads = _posterior_shift_fixture()
    cpu = _targets(tree, stats, heads, 0.5)
    device = torch.device("cuda")
    gpu_tree = TreeBatch(**{
        name: getattr(tree, name).to(device)
        for name in TreeBatch.__dataclass_fields__
    })
    gpu_stats = _stats(stats.p_hat.to(device), stats.v_hat.to(device))
    gpu_heads = HeadEvals(heads.node_ids.to(device), heads.v.to(device))
    gpu = _targets(gpu_tree, gpu_stats, gpu_heads, 0.5)
    for field in ("parent_p_hat", "child_p_hat", "a_hat", "v_hat"):
        assert torch.equal(getattr(cpu, field), getattr(gpu, field).cpu())
