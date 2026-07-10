"""Policy, value, and optional public-moment output heads."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pbg.games.poker.model.primitives import FFN


class TypeHead(nn.Module):
    """Emit contextual passive/call/raise logits for each actor row."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.affine = nn.Linear(in_dim, 3)

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
        self.mixture_affine = nn.Linear(in_dim, 3 * m_raise)
        self.allin_affine = nn.Linear(in_dim, 1)

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


def dlogistic_tick(n_ticks: int, mu: Tensor, s: Tensor) -> Tensor:
    """Return a normalized discretized logistic distribution ``[...,n_ticks]``."""

    if isinstance(n_ticks, bool) or not isinstance(n_ticks, int) or n_ticks <= 0:
        raise ValueError("n_ticks must be a positive integer")
    if mu.shape != s.shape:
        raise ValueError("mu and s must have identical shapes")
    if not mu.is_floating_point() or not s.is_floating_point():
        raise TypeError("mu and s must be floating point")
    if bool((s <= 0).any()):
        raise ValueError("logistic scales must be positive")
    # fp32 accumulation is important for wide mixtures under fp16 autocast.
    work_dtype = torch.float64 if mu.dtype == torch.float64 or s.dtype == torch.float64 else torch.float32
    location = mu.to(work_dtype)[..., None]
    scale = s.to(work_dtype)[..., None]
    edges = torch.linspace(0.0, 1.0, n_ticks + 1, dtype=work_dtype, device=mu.device)
    standardized = (edges - location) / scale
    lower = standardized[..., :-1]
    upper = standardized[..., 1:]
    # sigmoid(upper) - sigmoid(lower), written without subtracting nearly
    # equal CDF values. This remains normalized even for extremely wide s.
    probability = (
        torch.sigmoid(upper)
        * torch.sigmoid(-lower)
        * (-torch.expm1(lower - upper))
    ).clamp_min(0.0)
    lower_zero = standardized[..., :1]
    upper_one = standardized[..., -1:]
    normalization = (
        torch.sigmoid(upper_one)
        * torch.sigmoid(-lower_zero)
        * (-torch.expm1(lower_zero - upper_one))
    )
    probability = probability / normalization.clamp_min(torch.finfo(work_dtype).tiny)
    # Preserve normalization when cancellation at extreme scales loses ulps.
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(work_dtype).tiny
    )
    return probability.to(torch.promote_types(mu.dtype, s.dtype))


def raise_tick_probs(
    omega: Tensor,
    mu: Tensor,
    s: Tensor,
    iota: Tensor,
    n_ticks: int,
) -> Tensor:
    """Compose the logistic mixture and all-in atom as ``[...,n_ticks]``."""

    if omega.shape != mu.shape or omega.shape != s.shape:
        raise ValueError("omega, mu, and s must have identical [...,M] shapes")
    if iota.shape != omega.shape[:-1]:
        raise ValueError("iota must have shape omega.shape[:-1]")
    logistic = dlogistic_tick(n_ticks, mu, s)
    mixture = (omega[..., None] * logistic).sum(dim=-2)
    allin = torch.zeros_like(mixture)
    allin[..., -1] = 1.0
    return (1.0 - iota[..., None]) * mixture + iota[..., None] * allin


class BinomialTickHead(nn.Module):
    """Flag-gated binomial raise-tick ablation with an all-in atom."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.prob_affine = nn.Linear(in_dim, 1)
        self.allin_affine = nn.Linear(in_dim, 1)

    def forward(self, z_raise: Tensor, n_ticks: int) -> Tensor:
        """Return binomial-plus-all-in probabilities ``[...,n_ticks]``."""

        if n_ticks <= 0 or z_raise.shape[-1] != self.in_dim:
            raise ValueError("invalid tick count or feature dimension")
        if n_ticks == 1:
            return torch.ones(z_raise.shape[:-1] + (1,), dtype=z_raise.dtype, device=z_raise.device)
        p = torch.sigmoid(self.prob_affine(z_raise)).squeeze(-1)
        iota = torch.sigmoid(self.allin_affine(z_raise)).squeeze(-1)
        count = n_ticks - 1
        ticks = torch.arange(n_ticks, dtype=torch.float32, device=z_raise.device)
        log_choose = (
            torch.lgamma(torch.tensor(float(count + 1), device=z_raise.device))
            - torch.lgamma(ticks + 1.0)
            - torch.lgamma(float(count) - ticks + 1.0)
        )
        p32 = p.float().clamp(torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps)
        log_prob = log_choose + ticks * torch.log(p32[..., None]) + (count - ticks) * torch.log1p(-p32[..., None])
        base = torch.softmax(log_prob, dim=-1).to(z_raise.dtype)
        allin = torch.zeros_like(base)
        allin[..., -1] = 1.0
        return (1.0 - iota[..., None]) * base + iota[..., None] * allin


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
        self.affine = nn.Linear(feature_dim, 1)

    def forward(self, z_value: Tensor) -> Tensor:
        """Return values shaped ``z_value.shape[:-1]``."""

        return self.affine(self.features(z_value)).squeeze(-1)


class MomentHeads(nn.Module):
    """Emit public per-seat values and a nonnegative squared-return moment."""

    def __init__(self, public_dim: int, seat_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        self.public_dim = public_dim
        self.seat_dim = seat_dim
        self.public_features = FFN(public_dim, seat_dim, hidden)
        self.seat_features = FFN(2 * seat_dim, seat_dim, hidden)
        self.value_affine = nn.Linear(seat_dim, 1)
        self.u_features = FFN(seat_dim, seat_dim, hidden)
        self.u_affine = nn.Linear(seat_dim, 1)

    def forward(
        self, public: Tensor, seats: Tensor, seat_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``Vvec [B,P]`` and ``U [B]`` with padded seats zeroed."""

        if public.ndim != 2 or public.shape[-1] != self.public_dim:
            raise ValueError("public must have shape [B,public_dim]")
        if seats.ndim != 3 or seats.shape[0] != public.shape[0] or seats.shape[-1] != self.seat_dim:
            raise ValueError("seats must have shape [B,P,seat_dim]")
        if seat_mask.shape != seats.shape[:2] or seat_mask.dtype != torch.bool:
            raise TypeError("seat_mask must be bool [B,P]")
        public_features = self.public_features(public)
        expanded = public_features[:, None, :].expand_as(seats)
        seat_hidden = self.seat_features(torch.cat((expanded, seats), dim=-1))
        vvec = self.value_affine(seat_hidden).squeeze(-1)
        vvec = vvec.masked_fill(~seat_mask, 0.0)
        u = F.softplus(self.u_affine(self.u_features(public_features)).squeeze(-1))
        return vvec, u
