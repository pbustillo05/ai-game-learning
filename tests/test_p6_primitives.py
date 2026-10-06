import math

import pytest
import torch

from pbg.games.poker.model.primitives import MHA

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


def _explicit_attention(
    mha: MHA,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_bias: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    temperature = mha.temperature.float()[None, :, None, None]
    content = torch.matmul(query.float(), key.float().transpose(-2, -1))
    content = content / (temperature * math.sqrt(mha.d_head))
    logits = content + key_bias[:, None, None, :]
    logits = logits.masked_fill(~key_mask[:, None, None, :], -torch.inf)
    probability = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    return torch.matmul(probability, value.float()).to(value.dtype)


@pytest.mark.parametrize("backend", ("auto", "flex"))
def test_attention_rejects_implicit_or_unpublished_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="unknown attention backend"):
        MHA(16, 4).configure_execution(backend, query_chunk_size=8)


def test_attention_rejects_cpu_before_projection() -> None:
    mha = MHA(16, 4)
    query = torch.randn(2, 3, 16)
    with pytest.raises(ValueError, match="requires CUDA or MPS"):
        mha(query, query, None, None)


@CUDA
def test_key_bias_is_not_tempered() -> None:
    torch.manual_seed(61)
    device = torch.device("cuda", torch.cuda.current_device())
    mha = MHA(16, 4).to(device)
    with torch.no_grad():
        mha.w_qkv.weight[:2 * mha.d_model].zero_()
        mha.w_qkv.bias[:2 * mha.d_model].zero_()
    query = torch.randn(2, 3, 16, device=device)
    context = torch.randn(2, 5, 16, device=device)
    bias = torch.randn(2, 5, dtype=torch.float32, device=device)
    mask = torch.ones(2, 5, dtype=torch.bool, device=device)
    q_head, k_head, v_head = mha._project(query, context)
    before = _explicit_attention(mha, q_head, k_head, v_head, bias, mask)
    with torch.no_grad():
        mha.theta.mul_(2.0)
    after = _explicit_attention(mha, q_head, k_head, v_head, bias, mask)
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)


@CUDA
def test_query_chunk_attention_matches_explicit_forward_and_backward() -> None:
    torch.manual_seed(62)
    device = torch.device("cuda", torch.cuda.current_device())
    explicit = MHA(24, 4).to(device)
    chunked = MHA(24, 4).to(device)
    chunked.load_state_dict(explicit.state_dict())
    chunked.configure_execution("chunked_dense", query_chunk_size=2)
    assert chunked.state_dict().keys() == explicit.state_dict().keys()
    query_a = torch.randn(2, 5, 24, device=device, requires_grad=True)
    query_b = query_a.detach().clone().requires_grad_()
    bias = torch.randn(2, 5, dtype=torch.float32, device=device)
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]],
        device=device,
    )
    q_a, k_a, v_a = explicit._project(query_a, query_a)
    q_b, k_b, v_b = chunked._project(query_b, query_b)
    wanted = _explicit_attention(explicit, q_a, k_a, v_a, bias, mask)
    actual = chunked._chunked_attention(q_b, k_b, v_b, bias, mask)
    torch.testing.assert_close(actual, wanted, atol=1e-5, rtol=1e-5)
    wanted.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(query_b.grad, query_a.grad, atol=1e-5, rtol=1e-5)
    for left, right in zip(explicit.parameters(), chunked.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-5, rtol=1e-5)


@CUDA
@pytest.mark.parametrize("d_model", (16, 24, 128))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_biased_sdpa_matches_explicit_forward_and_backward(d_model, dtype) -> None:
    torch.manual_seed(63)
    device = torch.device("cuda", torch.cuda.current_device())
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("native BF16 is unavailable")
    explicit = MHA(d_model, 4).to(device)
    sdpa = MHA(d_model, 4).to(device)
    sdpa.load_state_dict(explicit.state_dict())
    query_a = torch.randn(2, 5, d_model, device=device, requires_grad=True)
    query_b = query_a.detach().clone().requires_grad_()
    bias = torch.randn(2, 5, dtype=torch.float32, device=device)
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]],
        device=device,
    )
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        q_a, k_a, v_a = explicit._project(query_a, query_a)
        q_b, k_b, v_b = sdpa._project(query_b, query_b)
        # The mathematical reference keeps its score arithmetic in FP32.
        with torch.autocast("cuda", enabled=False):
            wanted = _explicit_attention(explicit, q_a, k_a, v_a, bias, mask)
        actual = sdpa._sdpa_biased_attention(q_b, k_b, v_b, bias, mask)
    rtol, atol = ((1e-5, 1e-5) if dtype == torch.float32 else
                 (8e-3, 8e-4) if dtype == torch.float16 else (3e-2, 3e-3))
    torch.testing.assert_close(actual, wanted, atol=atol, rtol=rtol)
    wanted.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(query_b.grad, query_a.grad, atol=atol, rtol=rtol)
    for left, right in zip(explicit.parameters(), sdpa.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=atol, rtol=rtol)
            error = torch.linalg.vector_norm(right.grad - left.grad)
            reference_norm = torch.linalg.vector_norm(left.grad)
            assert float(error / reference_norm.clamp_min(1e-12)) <= rtol
