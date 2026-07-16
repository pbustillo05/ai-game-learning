from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pbg.core.losses import COMPOSITES, LossSpec, finalize_metric_stats
from pbg.core.normal import NormalMinibatch, index_normal_minibatch
from pbg.core.minibatch import MinibatchSample
from pbg.core.values import NodeTargets
from tests.categorical_policy_fixture import CategoricalPolicyAdapter, PolicyOutput


def test_registered_minimal_loss_uses_duplicate_children_independently() -> None:
    adapter = CategoricalPolicyAdapter()
    logits = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    output = PolicyOutput(logits, torch.zeros((1, 1), requires_grad=True))
    legal_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    legal_mask = torch.ones_like(legal_ids, dtype=torch.bool)
    selected = torch.tensor([[0, 0]], dtype=torch.int32)
    cached = adapter.log_prob(output, selected, legal_ids, legal_mask).detach()
    targets = NodeTargets(
        node_ids=torch.tensor([0], dtype=torch.int32),
        parent_p_hat=torch.ones((1, 1)),
        child_p_hat=torch.ones((1, 2, 1)),
        a_hat=torch.tensor([[[1.0], [2.0]]]),
        v_hat=torch.zeros((1, 1)),
        child_event_row=selected,
        child_mask=torch.ones((1, 2), dtype=torch.bool),
        child_count=torch.tensor([2], dtype=torch.int32),
    )
    sample = MinibatchSample(
        decision_rows=torch.tensor([0], dtype=torch.int32),
        state_refs=torch.tensor([0], dtype=torch.int32),
        reach=torch.zeros((1, 1, 1)),
        targets=targets,
        old_logp_child=cached,
        old_v=torch.zeros((1, 1)),
        valid_type_mask=torch.ones((1, 1), dtype=torch.bool),
        legal_event_ids=legal_ids,
        legal_event_mask=legal_mask,
    )
    cfg = SimpleNamespace(
        clip_eps=0.2, clip_eps_v=0.2, c_v=0.0, c_h=0.0,
        entropy_schedule="const",
    )
    spec = COMPOSITES["rb_ppo"]
    assert isinstance(spec, LossSpec)
    assert spec.required_targets == frozenset(
        {"parent_p_hat", "child_p_hat", "a_hat", "v_hat"}
    )
    loss, metrics = spec.fn(output, sample, sample, adapter, cfg)
    assert loss == -1.5
    assert metrics["ratio_max"] == 1.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_normal_ppo_selects_only_the_sampled_private_type() -> None:
    adapter = CategoricalPolicyAdapter()
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    output = PolicyOutput(logits, torch.tensor([[1.0, 7.0]], requires_grad=True))
    legal_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    legal_mask = torch.ones_like(legal_ids, dtype=torch.bool)
    old_logp = adapter.log_prob(
        output, torch.tensor([[1]], dtype=torch.int32), legal_ids, legal_mask
    )[0, 0, 1].detach()[None]
    sample = NormalMinibatch(
        state_refs=torch.tensor([0], dtype=torch.int32),
        reach=torch.zeros((1, 1, 2)),
        actor_type=torch.tensor([1]),
        event=torch.tensor([1], dtype=torch.int32),
        old_logp=old_logp,
        old_v=torch.tensor([7.0]),
        advantage=torch.tensor([2.0]),
        v_target=torch.tensor([9.0]),
        legal_event_ids=legal_ids,
        legal_event_mask=legal_mask,
    )
    cfg = SimpleNamespace(
        clip_eps=0.2, clip_eps_v=0.2, c_v=0.5, c_h=0.0,
        entropy_schedule="const",
    )
    loss, metrics = COMPOSITES["ppo"].fn(output, sample, sample, adapter, cfg)
    assert loss == 0.0  # -2 policy + 0.5 * (7 - 9)^2 value
    assert metrics["ratio_max"] == 1.0
    loss.backward()
    assert torch.equal(logits.grad[0, 0], torch.zeros(2))
    assert torch.isfinite(logits.grad[0, 1]).all()


def test_physical_batch_loss_and_metrics_equal_effective_batch() -> None:
    adapter = CategoricalPolicyAdapter()
    base_logits = torch.tensor(
        [[[0.2, -0.1], [0.0, 0.3]], [[-0.2, 0.4], [0.5, -0.3]]]
    )
    base_value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    legal_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    legal_mask = torch.ones_like(legal_ids, dtype=torch.bool)
    events = torch.tensor([1, 0], dtype=torch.int32)
    actor_type = torch.tensor([1, 0])
    reference = PolicyOutput(base_logits, base_value)
    logp = adapter.log_prob(
        reference, events[:, None], legal_ids, legal_mask
    )[:, 0].gather(1, actor_type[:, None]).squeeze(1)
    sample = NormalMinibatch(
        state_refs=torch.arange(2, dtype=torch.int32),
        reach=torch.zeros((2, 1, 2)),
        actor_type=actor_type,
        event=events,
        old_logp=logp,
        old_v=base_value.gather(1, actor_type[:, None]).squeeze(1),
        advantage=torch.tensor([1.0, -0.5]),
        v_target=torch.tensor([2.5, 2.0]),
        legal_event_ids=legal_ids,
        legal_event_mask=legal_mask,
    )
    cfg = SimpleNamespace(
        clip_eps=0.2, clip_eps_v=0.2, c_v=0.5, c_h=0.01,
        entropy_schedule="const",
    )
    full_logits = base_logits.clone().requires_grad_()
    full_value = base_value.clone().requires_grad_()
    full_loss, full_metrics = COMPOSITES["ppo"].fn(
        PolicyOutput(full_logits, full_value), sample, sample, adapter, cfg
    )
    full_loss.backward()

    split_logits = base_logits.clone().requires_grad_()
    split_value = base_value.clone().requires_grad_()
    parts = []
    for row in (torch.tensor([0]), torch.tensor([1])):
        physical = index_normal_minibatch(sample, row)
        loss_sum, stats = COMPOSITES["ppo"].fn(
            PolicyOutput(split_logits[row], split_value[row]),
            physical,
            physical,
            adapter,
            cfg,
            reduction="sum",
        )
        (loss_sum / 2).backward()
        parts.append(stats)
    torch.testing.assert_close(split_logits.grad, full_logits.grad)
    torch.testing.assert_close(split_value.grad, full_value.grad)
    split_metrics = finalize_metric_stats(parts)
    assert split_metrics == pytest.approx(full_metrics)
