"""One explicit-replica tensor program for projections, embeddings and norms.

Model inputs flatten the independent lane and decision axes to [K*B,...].
Weights retain [K,...]; a singleton module supplies the same program with K=1.
Only feature axes are normalized. Matrix dimensions remain contiguous views.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _accelerator(value: Tensor) -> None:
    if value.device.type not in ("cuda", "mps"):
        raise ValueError("model tensor operations require CUDA or MPS")


def compute_dtype(value: Tensor) -> torch.dtype:
    return (torch.get_autocast_dtype(value.device.type)
            if torch.is_autocast_enabled(value.device.type) else value.dtype)


def linear(value: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Independent lane BMMs; no packed copies of the master parameters."""

    _accelerator(value)
    weights = weight.unsqueeze(0) if weight.ndim == 2 else weight
    lanes = weights.shape[0]
    if value.shape[0] % lanes:
        raise ValueError("projection input does not contain whole replica batches")
    rows = value.reshape(lanes, -1, value.shape[-1])
    result = torch.bmm(rows, weights.transpose(1, 2))
    if bias is not None:
        biases = bias.unsqueeze(0) if bias.ndim == 1 else bias
        result = result + biases.to(result.dtype)[:, None, :]
    return result.reshape(*value.shape[:-1], weights.shape[-2])


def broadcast_parameter(value: Tensor, batch: int, event_ndim: int) -> Tensor:
    """Broadcast a small lane-local token/temperature across its decisions."""

    lane_value = value.unsqueeze(0) if value.ndim == event_ndim else value
    lanes = lane_value.shape[0]
    if batch % lanes:
        raise ValueError("parameter broadcast does not contain whole replica batches")
    return lane_value[:, None].expand(lanes, batch // lanes, *lane_value.shape[1:]).reshape(batch, *lane_value.shape[1:])


class Linear(nn.Linear):
    def forward(self, value: Tensor) -> Tensor:
        return linear(value, self.weight, self.bias)


class Embedding(nn.Embedding):
    def forward(self, index: Tensor) -> Tensor:
        _accelerator(index)
        weights = self.weight.unsqueeze(0) if self.weight.ndim == 2 else self.weight
        lanes = weights.shape[0]
        if index.ndim == 0 or index.shape[0] % lanes:
            raise ValueError("embedding indices require a leading replica/decision axis")
        addresses = index.reshape(lanes, -1)
        result = weights.gather(1, addresses[..., None].expand(-1, -1, weights.shape[-1]))
        result = result.reshape(*index.shape, weights.shape[-1])
        # Masters/backward remain FP32; stored activations follow autocast.
        return result.to(compute_dtype(result))


class LayerNorm(nn.LayerNorm):
    def forward(self, value: Tensor) -> Tensor:
        _accelerator(value)
        result = F.layer_norm(value, self.normalized_shape, None, None, self.eps)
        if self.weight is not None:
            weight = broadcast_parameter(self.weight, value.shape[0], 1)
            shape = (value.shape[0],) + (1,) * (value.ndim - 2) + (value.shape[-1],)
            result = result * weight.to(result.dtype).reshape(shape)
        if self.bias is not None:
            bias = broadcast_parameter(self.bias, value.shape[0], 1)
            result = result + bias.to(result.dtype).reshape(shape)
        return result
