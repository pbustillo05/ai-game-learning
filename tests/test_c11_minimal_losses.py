from __future__ import annotations

from types import SimpleNamespace

import torch

from pbg.core.losses import COMPOSITES, LossSpec
from pbg.core.minibatch import MinibatchSample
from pbg.core.values import NodeTargets
from pbg.games.synthetic.model import PolicyOutput, SyntheticPolicyAdapter


def test_registered_minimal_loss_uses_duplicate_children_independently() -> None:
    adapter = SyntheticPolicyAdapter()
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

