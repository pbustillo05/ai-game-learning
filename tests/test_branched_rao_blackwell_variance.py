from __future__ import annotations

import torch

from pbg.core.game import CHANCE, PRETERM
from pbg.core.tree import DiagnosticTreeBuilder
from pbg.eval.branched_rao_blackwell_variance import (
    _propagate_functional_coefficients,
    _tree_layout,
    load_config,
    projected_loss_and_child_partials,
    reconstruct_variance,
)


def test_projected_leave_one_out_partials_match_autograd() -> None:
    generator = torch.Generator().manual_seed(17)
    decisions, edges, hands, probes = 3, 3, 5, 2
    count = torch.tensor([1, 2, 3], dtype=torch.int64)
    mask = torch.arange(edges)[None] < count[:, None]
    p = (torch.rand((decisions, edges, hands), generator=generator, dtype=torch.float64) + 0.2)
    p = p.masked_fill(~mask[:, :, None], 0.0).requires_grad_()
    m = torch.randn((decisions, edges, hands), generator=generator, dtype=torch.float64)
    m = (m * p.detach()).masked_fill(~mask[:, :, None], 0.0).requires_grad_()
    old = torch.randn((decisions, hands), generator=generator, dtype=torch.float64)
    d_value = torch.randn((probes, decisions, hands), generator=generator, dtype=torch.float64)
    d_logp = torch.randn(
        (probes, decisions, edges, hands), generator=generator, dtype=torch.float64
    ).masked_fill(~mask[None, :, :, None], 0.0)

    projected, partial_p, partial_m = projected_loss_and_child_partials(
        p, m, old, d_value, d_logp, mask, count, c_v=0.0005
    )
    for probe in range(probes):
        wanted_p, wanted_m = torch.autograd.grad(
            projected[probe].sum(), (p, m), retain_graph=True
        )
        torch.testing.assert_close(partial_p[probe], wanted_p, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(partial_m[probe], wanted_m, rtol=1e-12, atol=1e-12)


def test_child_summary_partials_propagate_through_chance_average() -> None:
    builder = DiagnosticTreeBuilder(2)
    root = builder.add_root(0, 0)
    chance = builder.add_child(root, 1, 4, 1.0, CHANCE)
    left = builder.add_child(chance, 2, 5, 0.5, PRETERM)
    right = builder.add_child(chance, 3, 6, 0.5, PRETERM)
    tree = builder.finalize()
    layout = _tree_layout(tree)
    children = torch.tensor([[chance]])
    mask = torch.ones((1, 1), dtype=torch.bool)
    owners = torch.tensor([0])
    local_p = torch.tensor([[[2.0, -4.0]]])
    local_m = torch.tensor([[[6.0, 8.0]]])
    leaf_p, leaf_m = _propagate_functional_coefficients(
        tree=tree,
        layout=layout,
        children=children,
        child_mask=mask,
        owners=owners,
        local_p=local_p,
        local_m=local_m,
        functional_weights=torch.tensor([[3.0]], dtype=torch.float64),
        leaf_ids=torch.tensor([left, right]),
    )
    torch.testing.assert_close(
        leaf_p[:, :, 0], torch.tensor([[[3.0, -6.0], [3.0, -6.0]]])
    )
    torch.testing.assert_close(
        leaf_m[:, :, 0], torch.tensor([[[9.0, 12.0], [9.0, 12.0]]])
    )
    assert not bool(leaf_p[:, :, 1].any())
    assert not bool(leaf_m[:, :, 1].any())


def test_variance_surface_uses_branch_specific_A_and_one_over_n() -> None:
    first = reconstruct_variance(tree=2.0, minibatch=8.0, pair=12.0, node=20.0, B=4, n=10)
    doubled_B = reconstruct_variance(tree=2.0, minibatch=8.0, pair=12.0, node=20.0, B=8, n=10)
    doubled_n = reconstruct_variance(tree=2.0, minibatch=8.0, pair=12.0, node=20.0, B=4, n=20)
    assert first == 2.0 + 2.0 + ((0.75 * 12.0) + 5.0) / 10.0
    assert doubled_B < first
    assert doubled_n == 2.0 + 2.0 + (first - 4.0) / 2.0


def test_approved_config_has_only_checkpoint_1110() -> None:
    config = load_config(
        "configs/experiments/nlhe_hu_32bb_branched_rao_blackwell_variance.yaml"
    )
    assert config["checkpoint"] == 1110
    assert config["rollout"]["roots_per_gpu"] == 1
    assert config["rollout"]["branching_upper_cutoff_nodes"] == 4000
