"""Canonical queued poker policy operation.

All three entry points use the same FP32 factorized-policy composition.  The
implementation is accelerator-native PyTorch so autograd covers every policy
field while rollout avoids materializing a private-hand by action tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from pbg.games.poker.model.primitives import accelerator_assert
from pbg.core.learner import PolicyEvaluation
from pbg.core.rollout import logical_categorical

FOLD = 0
CHECK_CALL = 1
RAISE_OFFSET = 2
_POLICY_FIELDS = frozenset({"type_logits", "omega", "mu", "s", "iota"})


class PokerPolicyError(ValueError):
    """Raw policy or public legality violates the canonical operation."""


@dataclass(frozen=True)
class PokerLegality:
    """Fixed-width public legal event rows."""

    event_ids: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if (
            self.event_ids.device.type not in ("cuda", "mps")
            or self.event_ids.ndim != 2
            or self.event_ids.dtype not in (torch.int32, torch.int64)
            or self.mask.shape != self.event_ids.shape
            or self.mask.dtype != torch.bool
            or self.mask.device != self.event_ids.device
            or self.event_ids.shape[1] <= RAISE_OFFSET
        ):
            raise PokerPolicyError(
                "legality requires accelerator integer ids and bool mask [B,E]"
            )
        accelerator_assert(
            self.mask.any(-1).all(), "every poker row needs a legal action"
        )

    @property
    def raise_ticks(self) -> int:
        return self.event_ids.shape[1] - RAISE_OFFSET


@dataclass(frozen=True)
class ActorSample:
    action: Tensor
    selected_log_policy: Tensor


def _validate_raw(raw: Mapping[str, Tensor], legality: PokerLegality) -> None:
    if set(raw) != _POLICY_FIELDS:
        raise PokerPolicyError(
            f"raw policy fields must be exactly {sorted(_POLICY_FIELDS)}"
        )
    logits = raw["type_logits"]
    if (
        logits.device != legality.event_ids.device
        or logits.shape[0] != legality.event_ids.shape[0]
        or logits.shape[-1] != 3
        or not logits.is_floating_point()
    ):
        raise PokerPolicyError("type_logits must be floating [...,3] and batch aligned")
    prefix = logits.shape[:-1]
    mixture_shape = raw["omega"].shape
    if (
        len(mixture_shape) != len(prefix) + 1
        or mixture_shape[:-1] != prefix
        or mixture_shape[-1] <= 0
        or raw["mu"].shape != mixture_shape
        or raw["s"].shape != mixture_shape
        or raw["iota"].shape != prefix
    ):
        raise PokerPolicyError("raise-mixture fields do not share the policy prefix")
    if any(
        value.device != logits.device or not value.is_floating_point()
        for value in raw.values()
    ):
        raise PokerPolicyError("policy fields must be floating on one accelerator")
    finite = torch.stack(tuple(torch.isfinite(value).all() for value in raw.values()))
    positive = (raw["s"] > 0).all()
    mixture_law = (
        (raw["omega"] >= 0).all()
        & torch.isclose(
            raw["omega"].float().sum(-1),
            torch.ones_like(raw["omega"].float().sum(-1)),
            atol=2.0e-5,
            rtol=2.0e-5,
        ).all()
        & ((raw["mu"] >= 0) & (raw["mu"] <= 1)).all()
        & ((raw["iota"] >= 0) & (raw["iota"] <= 1)).all()
    )
    accelerator_assert(finite.all(), "raw poker policy must be finite")
    accelerator_assert(positive, "raise scales must be positive")
    accelerator_assert(mixture_law, "raw raise mixture law is invalid")


def _discretized_logistic(ticks: int, location: Tensor, scale: Tensor) -> Tensor:
    location = location.float()[..., None]
    scale = scale.float()[..., None]
    edges = torch.linspace(
        0.0, 1.0, ticks + 1, dtype=torch.float32, device=location.device
    )
    standardized = (edges - location) / scale
    lower, upper = standardized[..., :-1], standardized[..., 1:]
    probability = (
        torch.sigmoid(upper)
        * torch.sigmoid(-lower)
        * (-torch.expm1(lower - upper))
    ).clamp_min(0.0)
    normalizer = (
        torch.sigmoid(standardized[..., -1:])
        * torch.sigmoid(-standardized[..., :1])
        * (-torch.expm1(standardized[..., :1] - standardized[..., -1:]))
    )
    tiny = torch.finfo(torch.float32).tiny
    probability = probability / normalizer.clamp_min(tiny)
    return probability / probability.sum(-1, keepdim=True).clamp_min(tiny)


class PokerPolicyOperator:
    """Sole composition used by rollout, retained heads, and the learner."""

    def __init__(self, *, epsilon: float = 1.0e-12) -> None:
        if epsilon <= 0:
            raise ValueError("policy epsilon must be positive")
        self.epsilon = float(epsilon)

    def probabilities(
        self, raw: Mapping[str, Tensor], legality: PokerLegality
    ) -> Tensor:
        _validate_raw(raw, legality)
        logits = raw["type_logits"].float()
        prefix = logits.shape[:-1]
        ticks = legality.raise_ticks
        mixture = (
            raw["omega"].float()[..., None]
            * _discretized_logistic(ticks, raw["mu"], raw["s"])
        ).sum(-2)
        event_shape = (prefix[0],) + (1,) * (len(prefix) - 1) + (
            legality.event_ids.shape[1],
        )
        mask = legality.mask.reshape(event_shape).expand(prefix + (event_shape[-1],))
        raise_mask = mask[..., RAISE_OFFSET:]
        mixture = mixture.masked_fill(~raise_mask, 0.0)
        mixture_mass = mixture.sum(-1, keepdim=True)
        legal_raise_count = raise_mask.sum(-1, keepdim=True)
        uniform = raise_mask.float() / legal_raise_count.clamp_min(1)
        conditional_raise = torch.where(
            mixture_mass > 0,
            mixture / mixture_mass.clamp_min(self.epsilon),
            uniform,
        )
        tick_index = torch.arange(ticks, device=logits.device)
        all_in_index = torch.where(
            raise_mask,
            tick_index,
            torch.full_like(tick_index, -1),
        ).amax(-1)
        all_in = (
            tick_index == all_in_index[..., None]
        ).float() * (all_in_index >= 0)[..., None]
        raise_probability = (
            (1.0 - raw["iota"].float()[..., None]) * conditional_raise
            + raw["iota"].float()[..., None] * all_in
        ).masked_fill(~raise_mask, 0.0)
        facing = mask[..., FOLD]
        full_logits = torch.cat(
            (
                logits[..., 0, None],
                torch.where(facing, logits[..., 1], logits[..., 0])[..., None],
                logits[..., 2, None]
                + torch.where(
                    raise_probability > 0,
                    torch.log(raise_probability.clamp_min(self.epsilon)),
                    torch.full_like(raise_probability, -torch.inf),
                ),
            ),
            -1,
        )
        ids = legality.event_ids.to(torch.int64).reshape(event_shape).expand_as(mask)
        in_range = (ids >= 0) & (ids < full_logits.shape[-1])
        accelerator_assert(
            (~mask | in_range).all(), "legal poker event id is out of range"
        )
        selected_logits = full_logits.gather(-1, ids.clamp(0, full_logits.shape[-1] - 1))
        selected_logits = selected_logits.masked_fill(~mask, -torch.inf)
        probability = torch.softmax(selected_logits, -1).masked_fill(~mask, 0.0)
        return probability

    def selected_log_pi(
        self,
        raw: Mapping[str, Tensor],
        legality: PokerLegality,
        action: Tensor,
    ) -> Tensor:
        probability = self.probabilities(raw, legality)
        if action.shape != (probability.shape[0],):
            raise PokerPolicyError("selected action must have shape [B]")
        matches = legality.event_ids.eq(action[:, None].to(legality.event_ids))
        matches &= legality.mask
        accelerator_assert(
            matches.sum(-1).eq(1).all(),
            "selected action must occur exactly once in legal events",
        )
        column = matches.to(torch.int64).argmax(-1)
        column = column.reshape((probability.shape[0],) + (1,) * (probability.ndim - 2) + (1,))
        column = column.expand(probability.shape[:-1] + (1,))
        return torch.log(
            probability.gather(-1, column).squeeze(-1).clamp_min(self.epsilon)
        )

    def policy_reductions(
        self,
        raw: Mapping[str, Tensor],
        legality: PokerLegality,
        reference_raw: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        probability = self.probabilities(raw, legality)
        log_probability = torch.log(probability.clamp_min(self.epsilon))
        entropy = -(probability * log_probability).sum(-1)
        if reference_raw is None:
            return entropy, torch.zeros_like(entropy)
        reference = self.probabilities(reference_raw, legality)
        kl = (
            probability
            * (log_probability - torch.log(reference.clamp_min(self.epsilon)))
        ).sum(-1)
        return entropy, kl

    def sample_actor(
        self,
        raw: Mapping[str, Tensor],
        legality: PokerLegality,
        actor_hand: Tensor,
        logical_key: Tensor,
    ) -> ActorSample:
        if (
            raw["type_logits"].ndim != 3
            or actor_hand.shape != logical_key.shape
            or actor_hand.shape != raw["type_logits"].shape[:1]
            or actor_hand.device != raw["type_logits"].device
            or logical_key.device != raw["type_logits"].device
            or logical_key.dtype != torch.int64
        ):
            raise PokerPolicyError(
                "actor sampling requires policy [B,H,E] and aligned hands/keys"
            )
        hand = actor_hand.to(torch.int64)
        accelerator_assert(
            ((hand >= 0) & (hand < raw["type_logits"].shape[1])).all(),
            "actor hand is outside the policy domain",
        )
        row = torch.arange(hand.numel(), device=hand.device)
        selected_raw = MappingProxyType(
            {name: value[row, hand] for name, value in raw.items()}
        )
        # Composition now has prefix [B], so rollout materializes only [B,E].
        actor_probability = self.probabilities(selected_raw, legality)
        column = logical_categorical(actor_probability, logical_key)
        action = legality.event_ids.gather(1, column[:, None]).squeeze(1).to(torch.int32)
        selected = torch.log(
            actor_probability.gather(1, column[:, None]).squeeze(1).clamp_min(
                self.epsilon
            )
        )
        return ActorSample(action, selected)

    def learner(
        self,
        raw: Mapping[str, Tensor],
        legality: PokerLegality,
        action: Tensor,
        sampled_private_state: Tensor,
        owner: Tensor,
        reference_raw: Mapping[str, Tensor] | None,
    ) -> PolicyEvaluation:
        del sampled_private_state, owner
        selected = self.selected_log_pi(raw, legality, action)
        entropy, reference_kl = self.policy_reductions(
            raw, legality, reference_raw
        )
        return PolicyEvaluation(selected, entropy, reference_kl)


class BoundPokerPolicy:
    """Bind public legality once for the generic learner callback."""

    def __init__(self, operator: PokerPolicyOperator, legality: PokerLegality) -> None:
        self.operator = operator
        self.legality = legality

    def __call__(
        self,
        raw: Mapping[str, Tensor],
        action: Tensor,
        sampled_private_state: Tensor,
        owner: Tensor,
        reference_raw: Mapping[str, Tensor] | None,
    ) -> PolicyEvaluation:
        return self.operator.learner(
            raw,
            self.legality,
            action,
            sampled_private_state,
            owner,
            reference_raw,
        )


__all__ = [
    "CHECK_CALL",
    "FOLD",
    "RAISE_OFFSET",
    "ActorSample",
    "BoundPokerPolicy",
    "PokerLegality",
    "PokerPolicyError",
    "PokerPolicyOperator",
]
