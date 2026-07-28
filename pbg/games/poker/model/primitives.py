"""Normalization, attention, and feed-forward primitives for poker models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

def accelerator_assert(condition: Tensor, message: str) -> None:
    """Issue an asynchronous check unless an enclosing transform owns it."""

    if torch._C._functorch.is_batchedtensor(condition):
        return
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class LN(nn.LayerNorm):
    """Standard layer normalization with learned affine parameters."""


def gelu(x: Tensor) -> Tensor:
    """Apply the GELU nonlinearity without introducing another layer type."""

    return F.gelu(x)


class FFN(nn.Module):
    """Affine-GELU-affine map over the final dimension of an input tensor."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        hidden = 4 * out_dim if hidden is None else hidden
        if in_dim <= 0 or out_dim <= 0 or hidden <= 0:
            raise ValueError("FFN dimensions must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Return an output shaped ``[*x.shape[:-1], out_dim]``."""

        if x.shape[-1] != self.in_dim:
            raise ValueError(f"expected final dimension {self.in_dim}, got {x.shape[-1]}")
        return self.w2(gelu(self.w1(x)))


class MHA(nn.Module):
    """Multi-head attention with learned content temperatures and key priors.

    ``key_bias`` is added after the temperature-scaled content score. A zero
    bias uses PyTorch SDPA; a nonzero bias uses the explicit implementation.
    Both paths use the same projections and output affine map.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        # Execution-only attributes are deliberately absent from state_dict.
        self.attention_backend = "sdpa"
        self.query_chunk_size = 128
        # softplus(theta) + .5 == 1 at initialization.
        initial_theta = math.log(math.expm1(0.5))
        self.theta = nn.Parameter(torch.full((n_heads,), initial_theta))

    @property
    def temperature(self) -> Tensor:
        """Learned positive temperature ``[n_heads]``."""

        return F.softplus(self.theta) + 0.5

    def _project(self, q: Tensor, kv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, n_query, _ = q.shape
        n_key = kv.shape[1]
        q_head = self.w_q(q).reshape(batch, n_query, self.n_heads, self.d_head)
        k_head = self.w_k(kv).reshape(batch, n_key, self.n_heads, self.d_head)
        v_head = self.w_v(kv).reshape(batch, n_key, self.n_heads, self.d_head)
        return (
            q_head.transpose(1, 2),
            k_head.transpose(1, 2),
            v_head.transpose(1, 2),
        )

    def _validate(
        self,
        q: Tensor,
        kv: Tensor,
        key_bias: Tensor | None,
        key_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if q.ndim != 3 or kv.ndim != 3:
            raise ValueError("q and kv must have shapes [B,N,d]")
        if q.shape[0] != kv.shape[0] or q.shape[2] != self.d_model or kv.shape[2] != self.d_model:
            raise ValueError("q and kv batch/model dimensions do not match this MHA")
        batch, n_key = kv.shape[:2]
        if key_bias is None:
            key_bias = torch.zeros((batch, n_key), dtype=torch.float32, device=kv.device)
        if key_mask is None:
            key_mask = torch.ones((batch, n_key), dtype=torch.bool, device=kv.device)
        if key_bias.shape != (batch, n_key) or key_bias.dtype != torch.float32:
            raise TypeError("key_bias must be float32 [B,Nk]")
        if key_mask.shape != (batch, n_key) or key_mask.dtype != torch.bool:
            raise TypeError("key_mask must be bool [B,Nk]")
        if key_bias.device != kv.device or key_mask.device != kv.device or q.device != kv.device:
            raise ValueError("all attention tensors must be on one device")
        return key_bias, key_mask

    def _sdpa_biased_attention(
        self,
        q_head: Tensor,
        k_head: Tensor,
        v_head: Tensor,
        key_bias: Tensor,
        key_mask: Tensor,
    ) -> Tensor:
        """Use native SDPA with the exact additive reach/key bias."""

        temperature = self.temperature.to(dtype=q_head.dtype)[None, :, None, None]
        # Keep reach/key priors in their declared fp32 precision while the
        # content projections continue to use the configured autocast dtype.
        additive_mask = key_bias[:, None, None, :]
        additive_mask = additive_mask.masked_fill(
            ~key_mask[:, None, None, :], -torch.inf
        )
        return F.scaled_dot_product_attention(
            q_head / temperature,
            k_head,
            v_head,
            attn_mask=additive_mask,
            dropout_p=0.0,
        )

    def configure_execution(self, backend: str, query_chunk_size: int) -> None:
        """Select a mathematically exact bounded-memory execution strategy."""

        if backend not in {"sdpa", "chunked_dense"}:
            raise ValueError("unknown attention backend")
        if isinstance(query_chunk_size, bool) or query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.attention_backend = backend
        self.query_chunk_size = int(query_chunk_size)

    def _chunked_attention(
        self,
        q_head: Tensor,
        k_head: Tensor,
        v_head: Tensor,
        key_bias: Tensor,
        key_mask: Tensor,
    ) -> Tensor:
        """Exact full-key softmax with bounded query-score workspace."""

        temperature = self.temperature.float()[None, :, None, None]
        scale = 1.0 / math.sqrt(self.d_head)

        def attend(query_chunk: Tensor, key: Tensor, value: Tensor) -> Tensor:
            content = torch.matmul(
                query_chunk.float(), key.float().transpose(-2, -1)
            )
            logits = content * scale / temperature
            logits = logits + key_bias[:, None, None, :]
            logits = logits.masked_fill(~key_mask[:, None, None, :], -torch.inf)
            probability = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
            return torch.matmul(probability.to(value.dtype), value)

        chunks: list[Tensor] = []
        for start in range(0, q_head.shape[2], self.query_chunk_size):
            query_chunk = q_head[:, :, start : start + self.query_chunk_size]
            if self.training and torch.is_grad_enabled():
                chunks.append(
                    checkpoint(
                        attend,
                        query_chunk,
                        k_head,
                        v_head,
                        use_reentrant=False,
                    )
                )
            else:
                chunks.append(attend(query_chunk, k_head, v_head))
        return torch.cat(chunks, dim=2)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        key_bias: Tensor | None,
        key_mask: Tensor | None,
    ) -> Tensor:
        """Attend from ``q [B,Nq,d]`` to ``kv [B,Nk,d]`` and return ``[B,Nq,d]``."""

        if q.device.type not in ("cuda", "mps"):
            raise ValueError("attention forward requires CUDA or MPS")
        key_bias, key_mask = self._validate(q, kv, key_bias, key_mask)
        q_head, k_head, v_head = self._project(q, kv)
        if self.attention_backend == "sdpa":
            attended = self._sdpa_biased_attention(
                q_head, k_head, v_head, key_bias, key_mask
            )
        else:
            attended = self._chunked_attention(
                q_head, k_head, v_head, key_bias, key_mask
            )
        merged = attended.transpose(1, 2).reshape(q.shape[0], q.shape[1], self.d_model)
        return self.w_o(merged)

    def forward_biased(
        self,
        q: Tensor,
        kv: Tensor,
        key_bias: Tensor,
        key_mask: Tensor,
    ) -> Tensor:
        """Explicit steady-state entry point for reach-biased attention."""

        return self(q, kv, key_bias, key_mask)
