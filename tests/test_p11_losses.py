from __future__ import annotations

from types import SimpleNamespace

import torch

from pbg.core.losses import l_clip, l_v, l_v_clip, rb_ppo
from pbg.core.minibatch import MinibatchSample
from pbg.core.values import NodeTargets
from pbg.games.synthetic.model import PolicyOutput, SyntheticPolicyAdapter


def _fixture(dtype=torch.float64):
    logits = torch.tensor([[[0.2, -0.1, -torch.inf], [-0.3, 0.4, -torch.inf]]], dtype=dtype)
    old_out = PolicyOutput(logits.clone(), torch.tensor([[0.1, -0.2]], dtype=dtype))
    adapter = SyntheticPolicyAdapter()
    events = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    targets = NodeTargets(
        node_ids=torch.tensor([0], dtype=torch.int32),
        parent_p_hat=torch.tensor([[0.4, 0.6]], dtype=dtype),
        child_p_hat=torch.tensor(
            [[[0.9, 0.1], [0.2, 0.8], [0.0, 0.0]]], dtype=dtype
        ),
        a_hat=torch.tensor(
            [[[1.0, -0.3], [-0.4, 0.7], [999.0, 999.0]]], dtype=dtype
        ),
        v_hat=torch.tensor([[0.3, -0.1]], dtype=dtype),
        child_event_row=events,
        child_mask=torch.tensor([[True, True, False]]),
        child_count=torch.tensor([2], dtype=torch.int32),
    )
    legal_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    legal_mask = torch.ones_like(legal_ids, dtype=torch.bool)
    cached = adapter.log_prob(old_out, events[:, :2], legal_ids, legal_mask).detach()
    cached = torch.cat((cached, torch.full((1, 1, 2), -torch.inf, dtype=dtype)), dim=1)
    sample = MinibatchSample(
        decision_rows=torch.tensor([0], dtype=torch.int32),
        state_refs=torch.tensor([0], dtype=torch.int32),
        reach=torch.zeros((1, 2, 2), dtype=torch.float32),
        targets=targets,
        old_logp_child=cached,
        old_v=old_out.v.detach(),
        valid_type_mask=torch.ones((1, 2), dtype=torch.bool),
        legal_event_ids=legal_ids,
        legal_event_mask=legal_mask,
    )
    cfg = SimpleNamespace(
        clip_eps=0.2, clip_eps_v=0.2, c_v=0.5, c_h=0.0,
        entropy_schedule="const",
    )
    return logits, adapter, targets, sample, cfg


def test_clip_uses_direct_advantage_and_clips() -> None:
    logits, adapter, targets, sample, cfg = _fixture()
    current = logits.clone().requires_grad_()
    out = PolicyOutput(current, sample.old_v.clone())
    element, _ = l_clip(out, sample, sample, adapter, cfg)
    actual = torch.autograd.grad(element[0, 0, 0], current, retain_graph=True)[0]

    oracle_logits = logits.clone().requires_grad_()
    oracle_out = PolicyOutput(oracle_logits, sample.old_v.clone())
    logp = adapter.log_prob(
        oracle_out, targets.child_event_row[:, :2], sample.legal_event_ids,
        sample.legal_event_mask,
    )
    ratio = torch.exp(logp - sample.old_logp_child[:, :2])
    oracle = -ratio[0, 0, 0] * targets.a_hat[0, 0, 0]
    expected = torch.autograd.grad(oracle, oracle_logits)[0]
    assert torch.allclose(actual, expected, atol=1e-8, rtol=0)

    shifted = logits.clone()
    shifted[0, 0, 0] += 4.0
    shifted.requires_grad_()
    clipped, _ = l_clip(PolicyOutput(shifted, sample.old_v), sample, sample, adapter, cfg)
    assert torch.equal(
        torch.autograd.grad(clipped[0, 0, 0], shifted)[0], torch.zeros_like(shifted)
    )


def test_composite_masks_padding_and_identity_metrics() -> None:
    logits, adapter, _, sample, cfg = _fixture()
    current = logits.clone().requires_grad_()
    value = sample.old_v.clone().requires_grad_()
    loss, metrics = rb_ppo(PolicyOutput(current, value), sample, sample, adapter, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert current.grad[0, :, 2].abs().sum() == 0
    assert metrics["approx_kl"] == 0.0
    assert metrics["clip_frac"] == 0.0
    assert metrics["ratio_max"] == 1.0


def test_value_loss_control_selects_a_traceably_different_objective() -> None:
    logits, adapter, _, sample, cfg = _fixture()
    output = PolicyOutput(logits, torch.tensor([[0.3, -0.1]], dtype=logits.dtype))
    sample.old_v.zero_()
    mse, _ = l_v(output, sample, sample, adapter, cfg)
    clipped, _ = l_v_clip(output, sample, sample, adapter, cfg)
    assert not torch.equal(mse, clipped)

    cfg.value_loss = "mse"
    mse_composite, _ = rb_ppo(output, sample, sample, adapter, cfg)
    cfg.value_loss = "clipped"
    clipped_composite, _ = rb_ppo(output, sample, sample, adapter, cfg)
    assert mse_composite != clipped_composite
