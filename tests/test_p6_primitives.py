import pytest
import torch

from pbg.games.poker.model.primitives import MHA


@pytest.mark.parametrize("backend", ("auto", "flex"))
def test_attention_rejects_implicit_or_unpublished_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="unknown attention backend"):
        MHA(16, 4).configure_execution(backend, query_chunk_size=8)


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
    before = mha(query, context, bias, mask)
    with torch.no_grad():
        mha.theta.mul_(2.0)
    after = mha(query, context, bias, mask)
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)


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
