from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pbg.core.losses import _entropy_coefficient, l_clip, ppo, l_v, l_v_clip, rb_ppo
from pbg.core.minibatch import MinibatchSample
from pbg.core.normal import NormalMinibatch
from pbg.core.values import NodeTargets
from tests.categorical_policy_fixture import CategoricalPolicyAdapter, PolicyOutput


def _fixture(dtype=torch.float64):
    logits = torch.tensor([[[0.2, -0.1, -torch.inf], [-0.3, 0.4, -torch.inf]]], dtype=dtype)
    old_out = PolicyOutput(logits.clone(), torch.tensor([[0.1, -0.2]], dtype=dtype))
    adapter = CategoricalPolicyAdapter()
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


def test_padding_uses_a_legal_event_when_column_zero_is_illegal() -> None:
    logits, adapter, targets, sample, cfg = _fixture()
    targets.child_event_row[:] = torch.tensor([[1, 1, -1]], dtype=torch.int32)
    sample.legal_event_mask[:] = torch.tensor([[False, True]])
    cached = adapter.log_prob(
        PolicyOutput(logits, sample.old_v),
        torch.tensor([[1, 1, 1]], dtype=torch.int32),
        sample.legal_event_ids,
        sample.legal_event_mask,
    )
    sample.old_logp_child[:] = cached
    loss, _ = rb_ppo(
        PolicyOutput(logits.clone().requires_grad_(), sample.old_v),
        sample,
        sample,
        adapter,
        cfg,
    )
    assert torch.isfinite(loss)


def test_normal_and_rb_modes_use_the_same_exact_kl_term() -> None:
    logits, adapter, _, sample, cfg = _fixture()
    cfg.c_kl = 0.2
    cfg.c_v = 0.0
    cfg.c_h = 0.0
    sample.targets.a_hat.zero_()
    rb_logits = logits.clone().requires_grad_()
    current = PolicyOutput(rb_logits, sample.old_v)
    reference = PolicyOutput(logits + torch.tensor([[[0.3, -0.2, 0.0]]]), sample.old_v)
    rb_loss, rb_metrics = rb_ppo(
        current, sample, sample, adapter, cfg, reference=reference
    )
    rb_gradient = torch.autograd.grad(rb_loss, rb_logits)[0]

    actor_type = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    repeats = actor_type.numel()
    normal = NormalMinibatch(
        state_refs=torch.zeros(repeats, dtype=torch.int32),
        reach=torch.zeros((repeats, 2, 2)),
        actor_type=actor_type,
        event=torch.zeros(repeats, dtype=torch.int32),
        old_logp=torch.zeros(repeats),
        old_v=torch.zeros(repeats),
        advantage=torch.zeros(repeats),
        v_target=torch.zeros(repeats),
        legal_event_ids=sample.legal_event_ids.expand(repeats, -1),
        legal_event_mask=sample.legal_event_mask.expand(repeats, -1),
    )
    normal_logits = logits.clone().requires_grad_()
    expanded_current = PolicyOutput(
        normal_logits.expand(repeats, -1, -1), current.v.expand(repeats, -1)
    )
    expanded_reference = PolicyOutput(
        reference.raw.expand(repeats, -1, -1), reference.v.expand(repeats, -1)
    )
    normal.old_logp = adapter.log_prob(
        expanded_current,
        normal.event[:, None],
        normal.legal_event_ids,
        normal.legal_event_mask,
    )[:, 0].gather(1, actor_type[:, None]).squeeze(1)
    normal_loss, normal_metrics = ppo(
        expanded_current,
        normal,
        normal,
        adapter,
        cfg,
        reference=expanded_reference,
    )
    normal_gradient = torch.autograd.grad(normal_loss, normal_logits)[0]
    assert torch.allclose(
        torch.tensor(normal_metrics["reference_kl"]),
        torch.tensor(rb_metrics["reference_kl"]),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(normal_gradient, rb_gradient, atol=1e-12, rtol=0)


@torch.no_grad()
def test_linear_entropy_schedule_uses_training_horizon_in_both_composites() -> None:
    logits, adapter, _, rb_sample, loss_cfg = _fixture()
    loss_cfg.c_v = 0.0
    loss_cfg.c_h = 0.01
    loss_cfg.entropy_schedule = "linear_decay"
    root_cfg = SimpleNamespace(
        loss=loss_cfg,
        training=SimpleNamespace(iters=100),
    )
    rb_sample.targets.a_hat.zero_()
    rb_output = PolicyOutput(logits, rb_sample.old_v)

    actor_type = torch.tensor([0], dtype=torch.int64)
    normal_sample = NormalMinibatch(
        state_refs=torch.tensor([0], dtype=torch.int32),
        reach=torch.zeros((1, 1, 2)),
        actor_type=actor_type,
        event=torch.tensor([0], dtype=torch.int32),
        old_logp=adapter.log_prob(
            rb_output,
            torch.tensor([[0]], dtype=torch.int32),
            rb_sample.legal_event_ids,
            rb_sample.legal_event_mask,
        )[0, 0, 0][None],
        old_v=rb_output.v[:, 0],
        advantage=torch.zeros(1, dtype=logits.dtype),
        v_target=rb_output.v[:, 0],
        legal_event_ids=rb_sample.legal_event_ids,
        legal_event_mask=rb_sample.legal_event_mask,
    )

    for iteration, expected in ((0, 0.01), (50, 0.005), (100, 0.0)):
        assert _entropy_coefficient(root_cfg, iteration) == pytest.approx(expected)
        _, rb_metrics = rb_ppo(
            rb_output,
            rb_sample,
            rb_sample,
            adapter,
            root_cfg,
            iteration=iteration,
        )
        _, normal_metrics = ppo(
            rb_output,
            normal_sample,
            normal_sample,
            adapter,
            root_cfg,
            iteration=iteration,
        )
        assert rb_metrics["entropy_loss"] == pytest.approx(
            -expected * rb_metrics["entropy"]
        )
        assert normal_metrics["entropy_loss"] == pytest.approx(
            -expected * normal_metrics["entropy"]
        )


def test_linear_entropy_schedule_never_guesses_a_horizon() -> None:
    cfg = SimpleNamespace(c_h=0.01, entropy_schedule="linear_decay")
    with pytest.raises(ValueError, match="training.iters"):
        _entropy_coefficient(cfg, 0)
