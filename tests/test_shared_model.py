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


@CUDA
@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_explicit_replica_forward_backward_matches_independent_calls(lanes):
    from dataclasses import fields, is_dataclass
    from pbg.core.replica import ReplicaModuleOwner

    torch.manual_seed(604)
    models = [QueuedPokerModel(NLHEGeometry(2, 32, 13), config()).cuda() for _ in range(lanes)]
    owner = ReplicaModuleOwner(models)
    batched = owner.cohort_module(0, lanes)
    public, ranges = [], []
    for lane in range(lanes):
        pub, reach = pre_inputs(players=2)
        pub.pot = pub.pot + lane
        public.append(pub)
        ranges.append(reach.roll(lane, dims=-1))

    def concatenate(values):
        if values[0] is None:
            return None
        if is_dataclass(values[0]):
            return type(values[0])(**{f.name: concatenate([getattr(x, f.name) for x in values]) for f in fields(values[0])})
        return torch.cat(values, dim=0)

    out = batched.pre_encoder(batched.shared, concatenate(public), torch.cat(ranges))
    independent = [model.pre_encoder(model.shared, pub, reach) for model, pub, reach in zip(models, public, ranges, strict=True)]
    torch.testing.assert_close(out.v, torch.cat([x.v for x in independent]), atol=3e-5, rtol=3e-5)
    before = {name: parameter.grad.data_ptr() for name, parameter in batched.named_parameters()}
    out.v.square().sum().backward()
    for expected in independent:
        expected.v.square().sum().backward()
    for lane, model in enumerate(models):
        for name, expected in model.named_parameters():
            begin, end, shape = owner.parameter_layout[name]
            actual = owner.gradients[lane, begin:end].view(shape)
            torch.testing.assert_close(actual, torch.zeros_like(expected) if expected.grad is None else expected.grad,
                                       atol=5e-5, rtol=5e-5)
    saved = owner.gradients.clone()
    second = batched.pre_encoder(batched.shared, concatenate(public), torch.cat(ranges))
    second.v.square().sum().backward()
    torch.testing.assert_close(owner.gradients, 2 * saved, atol=5e-5, rtol=5e-5)
    assert before == {name: parameter.grad.data_ptr() for name, parameter in batched.named_parameters()}
    owner.assert_zero_copy()


def _scientific_contract():
    return {
        "game": {"players": 2, "stack_bb": 32, "rank_count": 13},
        "model": {"layers": 2, "d_model": 32, "folder_beta_rank": 16,
                  "folder_line_dim": 8, "raises": {"components": 4, "minimum_scale": 0.02},
                  "epsilon": 1e-8},
        "run": {"utility": "raw_bb"},
        "mode": {"rollout_mode": "sequential", "private_estimator": "sampled",
                 "training_regime": "self_play"},
    }


def test_frozen_opponent_may_have_different_internal_width_and_depth():
    from pbg.runtime import _validate_frozen_compatibility
    current = _scientific_contract()
    frozen = copy.deepcopy(current)
    frozen["model"].update(layers=6, d_model=128)
    _validate_frozen_compatibility(current, frozen, initialize=False)


@pytest.mark.parametrize("section,field,value", [
    ("game", "players", 3), ("game", "stack_bb", 64),
    ("model", "folder_beta_rank", 8), ("model", "folder_line_dim", 4),
    ("model", "raises", {"components": 2, "minimum_scale": 0.02}),
    ("run", "utility", "stack_normalized"),
])
def test_frozen_opponent_rejects_incompatible_input_or_policy_contract(section, field, value):
    from pbg.runtime import _validate_frozen_compatibility
    current = _scientific_contract()
    frozen = copy.deepcopy(current)
    frozen[section][field] = value
    with pytest.raises(RuntimeError, match="frozen checkpoint"):
        _validate_frozen_compatibility(current, frozen, initialize=False)


@CUDA
@pytest.mark.parametrize("precision,dtype,rtol,atol", [
    ("fp16", torch.float16, 8e-3, 8e-4),
    ("bf16", torch.bfloat16, 3e-2, 3e-3),
])
@pytest.mark.parametrize("family", ["pre", "post"])
def test_native_precision_model_outputs_and_gradients(precision, dtype, rtol, atol, family):
    from pbg.core.precision import resolve_precision
    if precision == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("native BF16 is unavailable on this architecture")
    resolve_precision(precision, torch.device("cuda"))
    torch.manual_seed(610)
    reference = QueuedPokerModel(NLHEGeometry(2, 32, 13), config()).cuda()
    candidate = copy.deepcopy(reference)
    if family == "pre":
        public, ranges = pre_inputs(players=2)
        def forward(model):
            return model.pre_encoder(model.shared, public, ranges).v
    else:
        public, ranges, masks = post_inputs()
        masks.compact_size = 12
        def forward(model):
            return model.post_encoder(model.shared, public, ranges, masks=masks).v
    observed = []
    handle = candidate.shared.attentions[0].register_forward_hook(
        lambda module, args, result: observed.append(result.dtype))
    expected = forward(reference)
    expected.float().square().mean().backward()
    with torch.autocast("cuda", dtype=dtype):
        actual = forward(candidate)
        loss = actual.float().square().mean()
    loss.backward()
    handle.remove()
    assert observed == [dtype]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    for left, right in zip(candidate.parameters(), reference.parameters(), strict=True):
        assert left.dtype == torch.float32
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert left.grad.dtype == torch.float32
            assert bool(torch.isfinite(left.grad).all())
            torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol)
