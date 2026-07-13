from __future__ import annotations

from types import SimpleNamespace

import torch

from pbg.core.losses import l_clip, rb_ppo
from pbg.core.minibatch import MinibatchSample
from pbg.core.values import NodeTargets
from pbg.games.synthetic.model import PolicyOutput, SyntheticPolicyAdapter


def _fixture(dtype=torch.float64):
    logits = torch.tensor([[[0.2, -0.1, -torch.inf], [-0.3, 0.4, -torch.inf]]], dtype=dtype)
    old_out = PolicyOutput(logits.clone(), torch.tensor([[0.1, -0.2]], dtype=dtype))
    adapter = SyntheticPolicyAdapter()
    events = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    targets = NodeTargets(
        node_ids=torch.tensor([0], dtype=torch.int32),
        p_hat=torch.tensor([[0.4, 0.6]], dtype=dtype),
        q_hat=torch.tensor([[[1.1, -0.5], [-0.2, 0.7], [999.0, 999.0]]], dtype=dtype),
        v_hat=torch.tensor([[0.3, -0.1]], dtype=dtype),
        vvec_hat=torch.zeros((1, 2), dtype=dtype), u_hat=torch.zeros(1, dtype=dtype),
        nu_row=torch.tensor([[0.25, 0.75, 0.0]], dtype=dtype),
        child_event_row=events,
    )
    sample = MinibatchSample(
        node_ids=targets.node_ids, targets=targets,
        old_logp_child=adapter.log_prob(old_out, events.long()).detach(),
        old_v=old_out.v.detach(), state_refs=torch.tensor([0], dtype=torch.int32),
        valid_type_mask=torch.ones((1, 2), dtype=torch.bool),
        child_mask=torch.tensor([[True, True, False]]),
    )
    cfg = SimpleNamespace(
        clip_eps=0.2, clip_eps_v=0.2, c_v=0.5, c_h=0.0, c_kl=0.0,
        c_u=0.0, c_vvec=0.0, ref_period=None, entropy_schedule="const",
        use_moment_heads=False,
    )
    return logits, adapter, targets, sample, cfg


def test_clip_gradient_matches_unclipped_analytic_direction_and_clips() -> None:
    logits, adapter, targets, sample, cfg = _fixture()
    current = logits.clone().requires_grad_()
    out = PolicyOutput(current, sample.old_v.clone())
    element, _ = l_clip(out, sample, sample, adapter, cfg)
    weighted = (element * targets.p_hat[:, None] * targets.nu_row[:, :, None]).sum()
    actual = torch.autograd.grad(weighted, current)[0]

    # Inside the trust region l_clip is exactly -rho*A.  Differentiate that
    # expression independently as the fp64 oracle.
    oracle_logits = logits.clone().requires_grad_()
    oracle_out = PolicyOutput(oracle_logits, sample.old_v.clone())
    logp = adapter.log_prob(oracle_out, targets.child_event_row.long())
    ratio = torch.exp(logp - sample.old_logp_child)
    advantage = targets.q_hat - sample.old_v[:, None]
    oracle = (-ratio * advantage * targets.p_hat[:, None] * targets.nu_row[:, :, None]).sum()
    expected = torch.autograd.grad(oracle, oracle_logits)[0]
    assert torch.allclose(actual, expected, atol=1e-8, rtol=0)

    shifted = logits.clone()
    shifted[0, 0, 0] += 4.0
    shifted.requires_grad_()
    clipped_out = PolicyOutput(shifted, sample.old_v.clone())
    clipped, _ = l_clip(clipped_out, sample, sample, adapter, cfg)
    # Positive advantage for child zero/type zero is on the flat clipped branch.
    grad = torch.autograd.grad(clipped[0, 0, 0], shifted)[0]
    assert torch.equal(grad, torch.zeros_like(grad))


def test_composite_masks_padding_and_identity_metrics() -> None:
    logits, adapter, targets, sample, cfg = _fixture()
    current = logits.clone().requires_grad_()
    out = PolicyOutput(current, sample.old_v.clone().requires_grad_())
    loss, metrics = rb_ppo(out, sample, sample, adapter, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert current.grad[0, :, 2].abs().sum() == 0
    assert metrics["approx_kl"] == 0.0
    assert metrics["clip_frac"] == 0.0
    assert metrics["ratio_max"] == 1.0
