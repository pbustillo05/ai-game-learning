"""Policy, value, and optional public branching output heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pbg.core.tensor_ops import Linear

from pbg.games.poker.model.primitives import FFN


class TypeHead(nn.Module):
    """Emit contextual passive/call/raise logits for each actor row."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.affine = Linear(in_dim, 3)

    def forward(self, z_type: Tensor) -> Tensor:
        """Map row features ``[...,D]`` to type logits ``[...,3]``."""

        if z_type.shape[-1] != self.in_dim:
            raise ValueError("type feature dimension does not match the head")
        return self.affine(z_type)


class RaiseMixtureHead(nn.Module):
    """Emit mixture weights, locations, scales, and an all-in atom."""

    def __init__(self, in_dim: int, m_raise: int = 4, s_min: float = 0.02) -> None:
        super().__init__()
        if in_dim <= 0 or m_raise <= 0 or s_min <= 0.0:
            raise ValueError("head dimensions and s_min must be positive")
        self.in_dim = in_dim
        self.m_raise = m_raise
        self.s_min = float(s_min)
        self.mixture_affine = Linear(in_dim, 3 * m_raise)
        self.allin_affine = Linear(in_dim, 1)

    def forward(
        self, z_raise: Tensor, z_allin: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``omega, mu, s [...,M]`` and ``iota [...]``."""

        if z_raise.shape[-1] != self.in_dim:
            raise ValueError("raise feature dimension does not match the head")
        if z_allin is None:
            z_allin = z_raise
        if z_allin.shape != z_raise.shape:
            raise ValueError("all-in and raise features must have identical shapes")
        raw = self.mixture_affine(z_raise)
        weight, location, scale = raw.split(self.m_raise, dim=-1)
        omega = torch.softmax(weight, dim=-1)
        mu = torch.sigmoid(location)
        s = F.softplus(scale) + self.s_min
        iota = torch.sigmoid(self.allin_affine(z_allin)).squeeze(-1)
        return omega, mu, s, iota


class ValueHead(nn.Module):
    """Emit one scalar value per actor private-type row."""

    def __init__(
        self,
        in_dim: int,
        feature_dim: int | None = None,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        feature_dim = in_dim if feature_dim is None else feature_dim
        self.features = FFN(in_dim, feature_dim, hidden)
        self.affine = Linear(feature_dim, 1)

    def forward(self, z_value: Tensor) -> Tensor:
        """Return values shaped ``z_value.shape[:-1]``."""

        return self.affine(self.features(z_value)).squeeze(-1)


class PublicBranchingHeads(nn.Module):
    """Emit public trace-variance ``T`` and subtree-cost ``C`` scalars."""

    def __init__(self, public_dim: int, feature_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        self.public_dim = public_dim
        self.public_features = FFN(public_dim, feature_dim, hidden)
        self.t_features = FFN(feature_dim, feature_dim, hidden)
        self.c_features = FFN(feature_dim, feature_dim, hidden)
        self.t_affine = Linear(feature_dim, 1)
        self.c_affine = Linear(feature_dim, 1)

    def forward(self, public: Tensor) -> tuple[Tensor, Tensor]:
        if public.ndim != 2 or public.shape[-1] != self.public_dim:
            raise ValueError("public branching input must have shape [B,public_dim]")
        features = self.public_features(public)
        # Heads remain fp32 under autocast: exp in fp16 overflows at ordinary
        # cost scales, while the controller consumes frozen fp32 quantities.
        t_logit = self.t_affine(self.t_features(features)).squeeze(-1).float()
        c_logit = self.c_affine(self.c_features(features)).squeeze(-1).float()
        t = torch.sigmoid(t_logit)
        c = torch.exp(c_logit.clamp(min=-30.0, max=30.0))
        return t, c
