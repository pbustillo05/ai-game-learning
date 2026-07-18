import pytest
import torch

from pbg.games.poker.model.heads import dlogistic_tick, raise_tick_probs
from pbg.games.poker.model.primitives import MHA


def test_mha_sdpa_matches_explicit_and_masks_are_exact() -> None:
    torch.manual_seed(60)
    mha = MHA(24, 4)
    query = torch.randn(2, 3, 24)
    context = torch.randn(2, 5, 24)
    key_mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, False, True]]
    )
    key_bias = torch.zeros(2, 5, dtype=torch.float32)
    q_head, k_head, v_head = mha._project(query, context)
    sdpa = mha._sdpa_attention(q_head, k_head, v_head, key_mask)
    explicit, probability = mha._explicit_attention(
        q_head, k_head, v_head, key_bias, key_mask
    )
    torch.testing.assert_close(sdpa, explicit, atol=1e-5, rtol=1e-5)
    masked = ~key_mask[:, None, None, :].expand_as(probability)
    assert torch.count_nonzero(probability.masked_select(masked)) == 0


def test_key_bias_is_not_tempered() -> None:
    torch.manual_seed(61)
    mha = MHA(16, 4)
    with torch.no_grad():
        mha.w_q.weight.zero_()
        mha.w_q.bias.zero_()
        mha.w_k.weight.zero_()
        mha.w_k.bias.zero_()
    query = torch.randn(2, 3, 16)
    context = torch.randn(2, 5, 16)
    bias = torch.randn(2, 5, dtype=torch.float32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    before = mha.attention_weights(query, context, bias, mask)
    with torch.no_grad():
        mha.theta.mul_(2.0)
    after = mha.attention_weights(query, context, bias, mask)
    torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)


def test_query_chunk_attention_matches_explicit_forward_and_backward() -> None:
    torch.manual_seed(62)
    explicit = MHA(24, 4)
    chunked = MHA(24, 4)
    chunked.load_state_dict(explicit.state_dict())
    chunked.configure_execution("chunked_dense", query_chunk_size=2)
    assert chunked.state_dict().keys() == explicit.state_dict().keys()
    query_a = torch.randn(2, 5, 24, requires_grad=True)
    query_b = query_a.detach().clone().requires_grad_()
    bias = torch.randn(2, 5, dtype=torch.float32)
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )
    q_a, k_a, v_a = explicit._project(query_a, query_a)
    q_b, k_b, v_b = chunked._project(query_b, query_b)
    wanted, _ = explicit._explicit_attention(q_a, k_a, v_a, bias, mask)
    actual = chunked._chunked_attention(q_b, k_b, v_b, bias, mask)
    torch.testing.assert_close(actual, wanted, atol=1e-5, rtol=1e-5)
    wanted.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(query_b.grad, query_a.grad, atol=1e-5, rtol=1e-5)
    for left, right in zip(explicit.parameters(), chunked.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-5, rtol=1e-5)


def test_biased_sdpa_matches_explicit_forward_and_backward() -> None:
    torch.manual_seed(63)
    explicit = MHA(24, 4)
    sdpa = MHA(24, 4)
    sdpa.load_state_dict(explicit.state_dict())
    query_a = torch.randn(2, 5, 24, requires_grad=True)
    query_b = query_a.detach().clone().requires_grad_()
    bias = torch.randn(2, 5, dtype=torch.float32)
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )
    q_a, k_a, v_a = explicit._project(query_a, query_a)
    q_b, k_b, v_b = sdpa._project(query_b, query_b)
    wanted, _ = explicit._explicit_attention(q_a, k_a, v_a, bias, mask)
    actual = sdpa._sdpa_biased_attention(q_b, k_b, v_b, bias, mask)
    torch.testing.assert_close(actual, wanted, atol=1e-5, rtol=1e-5)
    wanted.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(query_b.grad, query_a.grad, atol=1e-5, rtol=1e-5)
    for left, right in zip(explicit.parameters(), sdpa.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_biased_sdpa_matches_flex_throughput_precision() -> None:
    torch.manual_seed(64)
    flex = MHA(128, 4).cuda().train()
    sdpa = MHA(128, 4).cuda().train()
    sdpa.load_state_dict(flex.state_dict())
    flex.configure_execution("flex", query_chunk_size=128)
    sdpa.configure_execution("sdpa", query_chunk_size=128)
    query_a = torch.randn(2, 128, 128, device="cuda", requires_grad=True)
    query_b = query_a.detach().clone().requires_grad_()
    bias = 3 * torch.randn(2, 128, device="cuda", dtype=torch.float32)
    mask = torch.rand(2, 128, device="cuda") > 0.05
    with torch.autocast("cuda", dtype=torch.float16):
        output_a = flex.forward_biased(query_a, query_a, bias, mask)
        output_b = sdpa.forward_biased(query_b, query_b, bias, mask)
    torch.testing.assert_close(output_b, output_a, atol=2e-3, rtol=2e-2)
    output_a.float().square().mean().backward()
    output_b.float().square().mean().backward()
    torch.testing.assert_close(query_b.grad, query_a.grad, atol=1e-5, rtol=5e-2)
    for left, right in zip(flex.parameters(), sdpa.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-4, rtol=5e-2)


def test_dlogistic_ticks_and_allin_atom() -> None:
    mu = torch.tensor([[0.15, 0.50, 0.85], [0.20, 0.60, 0.90]])
    scale = torch.tensor([[1e-20, 0.03, 5.0], [0.01, 0.2, 1e20]])
    probability = dlogistic_tick(64, mu, scale)
    torch.testing.assert_close(
        probability.sum(dim=-1), torch.ones_like(mu), atol=1e-6, rtol=0.0
    )
    low = dlogistic_tick(64, torch.tensor([0.2]), torch.tensor([0.03]))
    high = dlogistic_tick(64, torch.tensor([0.8]), torch.tensor([0.03]))
    assert int(low.argmax()) < int(high.argmax())

    omega = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    mu = torch.sigmoid(torch.randn(2, 3, 4))
    scale = torch.rand(2, 3, 4) + 0.02
    ticks = raise_tick_probs(omega, mu, scale, torch.ones(2, 3), 32)
    assert torch.count_nonzero(ticks[..., :-1]) == 0
    assert torch.equal(ticks[..., -1], torch.ones(2, 3))
