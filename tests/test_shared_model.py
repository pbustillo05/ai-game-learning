"""Shared-model ownership and CUDA numerical contracts (no retired model oracle)."""

import copy

import pytest
import torch

from pbg.config import ConfigError, _parse_model
from pbg.core.persistence import PersistenceError, validate_checkpoint_payload
from pbg.games.poker.model.preflop import PokerModelConfig
from pbg.games.poker.model.primitives import FFN
from pbg.games.poker.queued.hooks import NLHEGeometry
from pbg.games.poker.queued.model import QueuedPokerModel
from tests.test_p6_preflop import _cuda_inputs as pre_inputs
from tests.test_p7_postflop import _cuda_inputs as post_inputs

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def config():
    return PokerModelConfig(d_model=16, n_heads=4, layers=1, m_raise=2,
                            k_act=5, max_players=2, class_embedding_dim=4,
                            ffn_hidden=24, use_branching_heads=True)


def test_shared_owner_has_no_duplicate_registered_parameters():
    with torch.device("meta"):
        model = QueuedPokerModel(NLHEGeometry(2, 32, 13), config())
    named = list(model.named_parameters(remove_duplicate=False))
    assert len(named) == len({id(value) for _, value in named})
    assert [name for name, _ in named if name.endswith("value_head.affine.weight")] == [
        "shared.value_head.affine.weight"
    ]
    assert len(model.shared.attentions) == 1
    assert model.shared.layer_ffns[0].static["pre"].in_features == 60
    assert model.shared.layer_ffns[0].static["post"].in_features == 155
    assert not any("attentions" in name for name, _ in model.pre_encoder.named_parameters())
    assert not any("attentions" in name for name, _ in model.post_encoder.named_parameters())


@pytest.mark.parametrize("layers", [True, 0, -1, 1.5, {"pre": 1, "post": 1}])
def test_shared_config_rejects_split_or_invalid_depth(layers):
    with pytest.raises(ConfigError, match="model.layers"):
        _parse_model({"layers": layers}, players=2)


def test_shared_config_has_one_canonical_depth():
    internal, public = _parse_model({"layers": 6}, players=2)
    assert internal["layers"] == public["layers"] == 6
    assert "n_layers_pre" not in internal and "n_layers_post" not in internal


def test_split_checkpoint_is_explicitly_rejected():
    with pytest.raises(PersistenceError, match="split-model"):
        validate_checkpoint_payload({"checkpoint_schema": "pbg-queued-checkpoint-v3"})


@CUDA
def test_factored_readout_matches_broadcast_forward_and_gradients():
    torch.manual_seed(601)
    factored = FFN(17, 8, 24).cuda()
    dense = copy.deepcopy(factored)
    rows = torch.randn(2, 7, 5, device="cuda", requires_grad=True)
    context = torch.randn(2, 12, device="cuda", requires_grad=True)
    other_rows = rows.detach().clone().requires_grad_()
    other_context = context.detach().clone().requires_grad_()
    actual = factored.forward_factored(rows, context)
    expected = dense(torch.cat((other_rows, other_context[:, None].expand(-1, 7, -1)), -1))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for left, right in zip((rows, context, *factored.parameters()), (other_rows, other_context, *dense.parameters()), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=3e-5, rtol=3e-5)


@CUDA
@pytest.mark.parametrize("family", ["pre", "post"])
def test_selected_head_matches_full_readout_and_parameter_gradients(family):
    torch.manual_seed(602)
    model = QueuedPokerModel(NLHEGeometry(2, 32, 13), config()).cuda()
    if family == "pre":
        pub, ranges = pre_inputs(players=2)
        encoder, masks = model.pre_encoder, None
        selected = torch.tensor([4, 21], device="cuda")
    else:
        pub, ranges, masks = post_inputs()
        masks.compact_size = 12
        encoder = model.post_encoder
        selected = torch.tensor([4, 21], device="cuda")
    full = encoder(model.shared, pub, ranges, masks=masks)
    sampled = encoder(model.shared, pub, ranges, masks=masks, selected_private=selected)
    batch = torch.arange(2, device="cuda")
    full_fields = [full.v[batch, selected], *(v[batch, selected] for v in full.raw.values())]
    small_fields = [sampled.v[:, 0], *(v[:, 0] for v in sampled.raw.values())]
    for left, right in zip(full_fields, small_fields, strict=True):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    parameters = tuple(model.parameters())
    a = torch.autograd.grad(sum(x.square().sum() for x in full_fields), parameters, allow_unused=True)
    b = torch.autograd.grad(sum(x.square().sum() for x in small_fields), parameters, allow_unused=True)
    for left, right in zip(a, b, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)


@CUDA
def test_shared_gradient_is_weighted_sum_of_both_families():
    torch.manual_seed(603)
    model = QueuedPokerModel(NLHEGeometry(2, 32, 13), config()).cuda()
    pre_pub, pre_ranges = pre_inputs(players=2)
    post_pub, post_ranges, post_masks = post_inputs()
    post_masks.compact_size = 12
    parameters = tuple(model.shared.parameters())
    pre_loss = model.pre_encoder(model.shared, pre_pub, pre_ranges).v.square().mean()
    post_loss = model.post_encoder(model.shared, post_pub, post_ranges, masks=post_masks).v.square().mean()
    pre_grads = torch.autograd.grad(pre_loss, parameters, retain_graph=True, allow_unused=True)
    post_grads = torch.autograd.grad(post_loss, parameters, retain_graph=True, allow_unused=True)
    (0.3 * pre_loss + 0.7 * post_loss).backward()
    for parameter, pre, post in zip(parameters, pre_grads, post_grads, strict=True):
        if pre is None and post is None:
            assert parameter.grad is None
        else:
            expected = 0.3 * (0 if pre is None else pre) + 0.7 * (0 if post is None else post)
            torch.testing.assert_close(parameter.grad, expected, atol=3e-5, rtol=3e-5)
