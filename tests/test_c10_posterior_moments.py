from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pbg.core.completion import BatchedCompletionStats
from pbg.core.game import TERMINAL
from pbg.core.values import (
    AuxiliaryChannelSpec,
    HeadEvals,
    branched_value,
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
    target = branched_value(
        tree, stats, None, heads, 0.5, _Game(), configured_node_budget=3, emit_q=True
    )

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
    target = branched_value(tree, stats, None, heads, lam, _Game(), configured_node_budget=2)
    assert target.q_hat is None
    assert torch.allclose(target.a_hat[0, 0], torch.tensor([3.0, -5.0]))
    assert torch.allclose(target.v_hat[0], torch.tensor([4.0, -2.0]))


def test_auxiliary_capture_precedes_player_bootstrap() -> None:
    builder = TreeBuilder(1)
    root = builder.add_root(0, 0)
    child = builder.add_child(root, 1, 4, 1.0, 0)
    builder.add_child(child, 2, 5, 1.0, TERMINAL)
    tree = builder.finalize()
    stats = _stats(torch.tensor([[[0.5, 0.5]]]), torch.tensor([[[2.0, 4.0]]]))
    stats.aux_scalar = torch.tensor([6.0])
    heads = HeadEvals(
        torch.tensor([root, child], dtype=torch.int32),
        torch.zeros((2, 2)),
        aux={"scalar": torch.tensor([10.0, 2.0])},
    )
    spec = AuxiliaryChannelSpec(
        "dummy", "aux_scalar", "scalar", "every_player_node", "captured"
    )
    target = branched_value(
        tree, stats, None, heads, 0.5, _Game(), configured_node_budget=3,
        auxiliary_reducers=(spec,),
    )
    # Child captures 6 before mixing it to 4; root then captures propagated 4.
    assert torch.equal(target.aux["captured"], torch.tensor([4.0, 6.0]))

    no_aux = branched_value(tree, stats, None, heads, 0.5, _Game(), configured_node_budget=3)
    assert no_aux.aux == {}


def test_chunking_matches_and_configuration_cap_is_loud() -> None:
    tree, stats, heads = _posterior_shift_fixture()
    base = branched_value(
        tree, stats, None, heads, 0.95, _Game(), configured_node_budget=3,
        max_resident_rows=100,
    )
    chunked = branched_value(
        tree, stats, None, heads, 0.95, _Game(), configured_node_budget=3,
        max_resident_rows=6,
    )
    for field in ("parent_p_hat", "child_p_hat", "a_hat", "v_hat"):
        assert torch.equal(getattr(base, field), getattr(chunked, field))
    with pytest.raises(ValueError, match="configured tree"):
        branched_value(
            tree, stats, None, heads, 0.95, _Game(), configured_node_budget=3,
            max_resident_rows=2,
        )


def test_rejects_low_precision_leaf_accumulators() -> None:
    tree, stats, heads = _posterior_shift_fixture()
    stats.p_hat = stats.p_hat.half()
    with pytest.raises(TypeError, match="fp16"):
        branched_value(tree, stats, None, heads, 0.5, _Game(), configured_node_budget=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_small_cpu_cuda_parity() -> None:
    tree, stats, heads = _posterior_shift_fixture()
    cpu = branched_value(tree, stats, None, heads, 0.5, _Game(), configured_node_budget=3)
    device = torch.device("cuda")
    gpu_tree = TreeBatch(**{
        name: getattr(tree, name).to(device)
        for name in TreeBatch.__dataclass_fields__
    })
    gpu_stats = _stats(stats.p_hat.to(device), stats.v_hat.to(device))
    gpu_heads = HeadEvals(heads.node_ids.to(device), heads.v.to(device))
    gpu = branched_value(
        gpu_tree, gpu_stats, None, gpu_heads, 0.5, _Game(), configured_node_budget=3
    )
    for field in ("parent_p_hat", "child_p_hat", "a_hat", "v_hat"):
        assert torch.equal(getattr(cpu, field), getattr(gpu, field).cpu())
