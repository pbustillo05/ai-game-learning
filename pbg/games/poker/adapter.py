"""Poker public navigation and factorized policy adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from pbg.games.poker.rules import CHECK_CALL, FOLD, RAISE_OFFSET

from pbg.games.poker.model.heads import dlogistic_tick
from pbg.games.poker.model.postflop import PostflopMasks, PostflopNet
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerPublicBatch,
    PolicyOutput,
    PreflopMasks,
    PreflopNet,
)


def _raw_mapping(output: PolicyOutput | Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    return output.raw if isinstance(output, PolicyOutput) else output


def _require_device(condition: Tensor, message: str) -> None:
    """Validate a scalar condition without synchronizing CUDA to the host."""

    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class PokerPolicyAdapter:
    """Compose contextual action types and conditional raise ticks."""

    def __init__(
        self,
        n_ticks: int | None = None,
        eps: float = 1e-12,
        exact_to_model: Tensor | tuple[int, ...] | None = None,
    ) -> None:
        if n_ticks is not None and n_ticks <= 0:
            raise ValueError("n_ticks must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.n_ticks = n_ticks
        self.eps = float(eps)
        self.exact_to_model = exact_to_model

    def select_actor_row(
        self,
        output: PolicyOutput | Mapping[str, Tensor], profiles: Tensor, actor: Tensor
    ) -> dict[str, Tensor]:
        """Select exactly the actor combo stipulated by each fresh profile."""

        raw = _raw_mapping(output)
        if profiles.ndim == 3:
            if profiles.shape[1] != 1:
                raise ValueError("profiles must contain one fresh profile per spot")
            profiles = profiles[:, 0]
        if profiles.ndim != 2 or actor.shape != (profiles.shape[0],):
            raise ValueError("profiles and actor must have shapes [B,P] and [B]")
        actor = actor.to(device=profiles.device, dtype=torch.long)
        combo = profiles.gather(1, actor[:, None]).squeeze(1).long()
        raw_widths = {value.shape[1] for value in raw.values() if value.ndim >= 2}
        if len(raw_widths) != 1:
            raise ValueError("raw policy fields must share one compact hand width")
        n_hands = next(iter(raw_widths))
        if self.exact_to_model is not None and n_hands != len(self.exact_to_model):
            mapping = torch.as_tensor(
                self.exact_to_model, dtype=torch.int64, device=combo.device
            )
            _require_device(
                ((combo >= 0) & (combo < mapping.numel())).all(),
                "the stipulated actor combo is invalid",
            )
            combo = mapping[combo]
        _require_device(
            ((combo >= 0) & (combo < n_hands)).all(),
            "the stipulated actor combo is invalid",
        )
        batch_ids = torch.arange(profiles.shape[0], device=profiles.device)
        selected: dict[str, Tensor] = {}
        for name, value in raw.items():
            if value.shape[:2] != (profiles.shape[0], n_hands):
                raise ValueError(
                    f"raw field {name} must begin [B,{n_hands}]"
                )
            selected[name] = value[batch_ids.to(value.device), combo.to(value.device)]
        return selected

    def probabilities(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        legal_event_ids: Tensor,
        legal_mask: Tensor,
        *,
        _prevalidated: bool = False,
    ) -> Tensor:
        """Return normalized probabilities over the supplied event columns."""

        fields = _raw_mapping(raw)
        required = {"type_logits", "omega", "mu", "s", "iota"}
        if not required.issubset(fields):
            raise ValueError("raw policy output is missing factorized head fields")
        type_logits = fields["type_logits"]
        if type_logits.ndim < 2 or type_logits.shape[-1] != 3:
            raise ValueError("type_logits must have shape [B,...,3]")
        batch = type_logits.shape[0]
        if legal_event_ids.shape != legal_mask.shape or legal_event_ids.ndim != 2:
            raise ValueError("legal event ids and masks must have shape [B,E]")
        if legal_event_ids.shape[0] != batch or legal_mask.dtype != torch.bool:
            raise TypeError("legal mask must be bool and batch-aligned")
        # Poker backends expose the complete contiguous
        # fold/check-call/raise-tick row, so its static width determines the
        # number of raise ticks without synchronizing a device scalar.
        inferred = legal_event_ids.shape[1] - RAISE_OFFSET
        n_ticks = self.n_ticks if self.n_ticks is not None else inferred
        if n_ticks <= 0:
            raise ValueError("the legal event set has no raise-tick width")
        type_logits = type_logits.float()
        if not _prevalidated:
            _require_device(
                torch.stack(
                    [
                        torch.isfinite(fields[name]).all()
                        for name in sorted(required)
                    ]
                ).all(),
                "raw policy fields must be finite",
            )
        type_probability = torch.softmax(type_logits, dim=-1)
        prefix = type_logits.shape[:-1]
        omega = fields["omega"].float()
        mu = fields["mu"].float()
        scale = fields["s"].float()
        iota = fields["iota"].float()
        logistic = dlogistic_tick(
            n_ticks, mu, scale, validate=not _prevalidated
        )
        mixture = (omega[..., None] * logistic).sum(dim=-2)
        raise_mask = legal_mask[:, RAISE_OFFSET:].to(type_logits.device)
        mask_shape = (batch,) + (1,) * (mixture.ndim - 2) + (n_ticks,)
        expanded_raise_mask = raise_mask.reshape(mask_shape).expand_as(mixture)
        mixture = mixture.masked_fill(~expanded_raise_mask, 0.0)
        mixture = mixture / mixture.sum(-1, keepdim=True).clamp_min(self.eps)
        tick_index = torch.arange(n_ticks, device=type_logits.device)[None]
        allin_index = torch.where(
            raise_mask,
            tick_index,
            torch.full_like(tick_index, -1),
        ).max(1).values
        has_raise = allin_index >= 0
        allin = (
            torch.arange(n_ticks, device=type_logits.device)[None, :]
            == allin_index[:, None]
        ).to(torch.float32) * has_raise.to(torch.float32)[:, None]
        allin = allin.reshape(mask_shape).expand_as(mixture)
        ticks = (1.0 - iota[..., None]) * mixture + iota[..., None] * allin
        ticks = ticks.masked_fill(~expanded_raise_mask, 0.0)

        # Preserve the ordinary probability-space path exactly.  This keeps
        # seeded rollout behavior unchanged except on the zero-mass corner
        # which previously raised a device-side assertion.
        facing_bet = ((legal_event_ids == FOLD) & legal_mask).any(dim=1)
        facing = facing_bet.reshape((batch,) + (1,) * (type_logits.ndim - 2))
        full = torch.cat(
            (
                torch.where(
                    facing, type_probability[..., 0], 0.0
                )[..., None],
                torch.where(
                    facing, type_probability[..., 1], type_probability[..., 0]
                )[..., None],
                type_probability[..., 2, None] * ticks,
            ),
            dim=-1,
        )

        # Condition on the currently legal raise grid before taking logs.  In
        # exact arithmetic the mixture already has positive mass on every
        # interval, but extreme learned locations/scales can underflow an
        # entire legal tail in fp32.  A uniform conditional fallback is used
        # only for that zero-mass numerical corner.
        tick_mass = ticks.sum(-1, keepdim=True)
        legal_raise_count = expanded_raise_mask.sum(-1, keepdim=True)
        uniform_legal_raise = expanded_raise_mask.to(ticks) / legal_raise_count.clamp_min(1)
        stable_ticks = torch.where(
            (tick_mass > 0) | (legal_raise_count == 0),
            ticks / tick_mass.clamp_min(self.eps),
            uniform_legal_raise,
        )

        # Build unnormalized event *logits* and softmax only over supplied
        # legal events.  The previous probability-space implementation first
        # softmaxed all three action types, then discarded illegal types.  If
        # an illegal type exceeded every legal type by roughly 100 logits,
        # every supplied legal probability underflowed to exactly zero.
        tick_logits = torch.where(
            stable_ticks > 0,
            torch.log(stable_ticks.clamp_min(self.eps)),
            torch.full_like(stable_ticks, -torch.inf),
        )
        full_logits = torch.cat(
            (
                type_logits[..., 0, None],
                torch.where(
                    facing, type_logits[..., 1], type_logits[..., 0]
                )[..., None],
                type_logits[..., 2, None] + tick_logits,
            ),
            dim=-1,
        )
        ids = legal_event_ids.to(device=full_logits.device, dtype=torch.long)
        if not _prevalidated:
            _require_device(
                ((ids >= 0) & (ids < full_logits.shape[-1])).all(),
                "legal event id is outside the factorized event space",
            )
        event_shape = (batch,) + (1,) * (type_logits.ndim - 2) + (ids.shape[1],)
        ids = ids.reshape(event_shape).expand(prefix + (ids.shape[1],))
        mask = legal_mask.to(full_logits.device).reshape(event_shape).expand_as(ids)
        if not _prevalidated:
            _require_device(
                mask.any(-1).all(),
                "each policy row must supply at least one legal event",
            )
        probability = full.gather(-1, ids).masked_fill(~mask, 0.0)
        normalizer = probability.sum(dim=-1, keepdim=True)
        ordinary = probability / normalizer.clamp_min(self.eps)
        legal_logits = full_logits.gather(-1, ids).masked_fill(~mask, -torch.inf)
        fallback = torch.softmax(legal_logits, dim=-1).masked_fill(~mask, 0.0)
        probability = torch.where(normalizer > 0, ordinary, fallback)
        if not _prevalidated:
            _require_device(
                torch.isfinite(probability).all(),
                "legal-event probabilities must be finite",
            )
        return probability

    compose = probabilities

    def policy_terms(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        reference: PolicyOutput | Mapping[str, Tensor] | None,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compose log-probabilities, entropy, and KL with one current pass."""

        return self._policy_terms(
            raw,
            reference,
            legal_event_ids,
            legal_mask,
            prevalidated=False,
        )

    def _policy_terms(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        reference: PolicyOutput | Mapping[str, Tensor] | None,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
        *,
        prevalidated: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        probability = self.probabilities(
            raw,
            legal_event_ids,
            legal_mask,
            _prevalidated=prevalidated,
        )
        event_shape = (probability.shape[0],) + (1,) * (probability.ndim - 2) + (
            probability.shape[-1],
        )
        mask = legal_mask.to(probability.device).reshape(event_shape).expand_as(
            probability
        )
        log_probability = torch.where(
            mask,
            torch.log(probability.clamp_min(self.eps)),
            torch.full_like(probability, -torch.inf),
        )
        entropy = -(probability * torch.where(mask, log_probability, 0.0)).sum(-1)
        if reference is None:
            kl = torch.zeros_like(entropy)
        else:
            reference_probability = self.probabilities(
                reference,
                legal_event_ids,
                legal_mask,
                _prevalidated=prevalidated,
            )
            reference_log = torch.log(reference_probability.clamp_min(self.eps))
            kl = (
                probability
                * (torch.where(mask, log_probability, 0.0) - reference_log)
            ).sum(-1)
        return log_probability, entropy, kl

    def policy_terms_prevalidated(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        reference: PolicyOutput | Mapping[str, Tensor] | None,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compose a pack whose inputs were validated before a transform."""

        return self._policy_terms(
            raw,
            reference,
            legal_event_ids,
            legal_mask,
            prevalidated=True,
        )

    def log_prob(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        selected_event_ids: Tensor,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Return selected legal-event log probabilities ``[B,E_selected,X]``."""

        return self._log_prob(
            raw,
            selected_event_ids,
            legal_event_ids,
            legal_mask,
            prevalidated=False,
        )

    def _log_prob(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        selected_event_ids: Tensor,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
        *,
        prevalidated: bool,
    ) -> Tensor:
        probability = self.probabilities(
            raw,
            legal_event_ids,
            legal_mask,
            _prevalidated=prevalidated,
        )
        event_shape = (probability.shape[0],) + (1,) * (probability.ndim - 2) + (probability.shape[-1],)
        mask = legal_mask.to(probability.device).reshape(event_shape).expand_as(probability)
        log_rows = torch.where(
            mask, torch.log(probability.clamp_min(self.eps)),
            torch.full_like(probability, -torch.inf),
        )
        if selected_event_ids.ndim == 1:
            selected_event_ids = selected_event_ids[:, None]
        matches = (
            legal_event_ids[:, None, :].to(selected_event_ids.device)
            == selected_event_ids[:, :, None]
        ) & legal_mask[:, None, :].to(selected_event_ids.device)
        if not prevalidated:
            _require_device(
                matches.any(-1).all(),
                "selected_event_ids must select supplied legal events",
            )
        columns = matches.to(torch.int64).argmax(-1).to(log_rows.device)  # [B,E_selected]
        if log_rows.ndim != 3:
            raise ValueError("poker policy log rows must have shape [B,X,E_legal]")
        gathered = log_rows.gather(
            -1, columns[:, None, :].expand(log_rows.shape[0], log_rows.shape[1], -1)
        )  # [B,X,E_selected]
        return gathered.transpose(1, 2)  # [B,E_selected,X]

    def log_prob_prevalidated(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        selected_event_ids: Tensor,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Select events from a pack validated before a transform."""

        return self._log_prob(
            raw,
            selected_event_ids,
            legal_event_ids,
            legal_mask,
            prevalidated=True,
        )

    def entropy(self, raw: PolicyOutput | Mapping[str, Tensor], legal_event_ids: Tensor, legal_mask: Tensor) -> Tensor:
        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        return -(probability * torch.log(probability.clamp_min(self.eps))).sum(dim=-1)

    def entropy_prevalidated(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        probability = self.probabilities(
            raw, legal_event_ids, legal_mask, _prevalidated=True
        )
        return -(probability * torch.log(probability.clamp_min(self.eps))).sum(
            dim=-1
        )

    def kl(
        self, raw: PolicyOutput | Mapping[str, Tensor], reference: PolicyOutput | Mapping[str, Tensor],
        legal_event_ids: Tensor, legal_mask: Tensor,
    ) -> Tensor:
        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        reference_probability = self.probabilities(reference, legal_event_ids, legal_mask)
        return (probability * (torch.log(probability.clamp_min(self.eps)) -
                               torch.log(reference_probability.clamp_min(self.eps)))).sum(dim=-1)

    def kl_prevalidated(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        reference: PolicyOutput | Mapping[str, Tensor],
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        probability = self.probabilities(
            raw, legal_event_ids, legal_mask, _prevalidated=True
        )
        reference_probability = self.probabilities(
            reference, legal_event_ids, legal_mask, _prevalidated=True
        )
        return (
            probability
            * (
                torch.log(probability.clamp_min(self.eps))
                - torch.log(reference_probability.clamp_min(self.eps))
            )
        ).sum(dim=-1)

    def sample(
        self, raw: PolicyOutput | Mapping[str, Tensor], legal_event_ids: Tensor,
        legal_mask: Tensor, gen: torch.Generator,
    ) -> Tensor:
        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        if probability.ndim != 2:
            raise ValueError("sample requires one already-selected private row per spot")
        column = torch.multinomial(probability, 1, generator=gen).squeeze(1)
        return legal_event_ids.gather(1, column[:, None].to(legal_event_ids.device)).squeeze(1).to(torch.int32)

    def draw(
        self,
        output: PolicyOutput | Mapping[str, Tensor],
        profiles: Tensor,
        actor: Tensor,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
        gen: torch.Generator,
    ) -> Tensor:
        """Select stipulated actor rows and draw one legal public event."""

        selected = self.select_actor_row(output, profiles, actor)
        return self.sample(selected, legal_event_ids, legal_mask, gen)


class PokerModelBundle(nn.Module):
    """Own and route the two street-specific poker models."""

    def __init__(self, preflop: PreflopNet, postflop: PostflopNet) -> None:
        super().__init__()
        if preflop.geometry != postflop.geometry:
            raise ValueError("preflop and postflop models must share deck geometry")
        self.preflop = preflop
        self.postflop = postflop
        self.geometry = preflop.geometry
        self.n_hands = self.geometry.n_hands
        self.n_classes = self.geometry.n_classes
        combo_class = torch.tensor(
            self.geometry.class_of_combo, dtype=torch.long
        )
        self.register_buffer("combo_class", combo_class, persistent=True)

    @staticmethod
    def _slice_public(pub: PokerPublicBatch, rows: Tensor) -> PokerPublicBatch:
        values = {}
        for name in pub.__dataclass_fields__:
            value = getattr(pub, name)
            if isinstance(value, PackedPublicHistory):
                values[name] = PackedPublicHistory(
                    **{
                        field: getattr(value, field)[rows]
                        for field in value.__dataclass_fields__
                    }
                )
            else:
                values[name] = None if value is None else value[rows]
        return PokerPublicBatch(**values)

    @staticmethod
    def _slice_post_masks(masks: PostflopMasks, rows: Tensor, batch: int) -> PostflopMasks:
        values: dict[str, Any] = {}
        for name in masks.__dataclass_fields__:
            value = getattr(masks, name)
            if isinstance(value, Tensor):
                values[name] = value[rows] if value.ndim and value.shape[0] == batch else value
            elif isinstance(value, Mapping):
                values[name] = {key: item[rows] for key, item in value.items()}
            elif hasattr(value, "fields"):
                values[name] = type(value)({key: item[rows] for key, item in value.fields.items()}, value.metadata)
            else:
                values[name] = value
        return PostflopMasks(**values)

    @staticmethod
    def _slice_pre_masks(masks: PreflopMasks | None, rows: Tensor, batch: int) -> PreflopMasks | None:
        if masks is None:
            return None
        values: dict[str, Tensor | None] = {}
        for name in masks.__dataclass_fields__:
            value = getattr(masks, name)
            values[name] = value[rows] if isinstance(value, Tensor) and value.ndim and value.shape[0] == batch else value
        return PreflopMasks(**values)

    def forward_preflop(
        self, pub: PokerPublicBatch, class_ranges: Tensor, actor_seat: Tensor | None = None,
        masks: PreflopMasks | None = None,
    ) -> PolicyOutput:
        """Run preflop on its natural class domain."""

        return self.preflop(pub, class_ranges, actor_seat, masks)

    def forward_postflop(
        self, pub: PokerPublicBatch, combo_ranges: Tensor, actor_seat: Tensor | None = None,
        masks: PostflopMasks | None = None,
    ) -> PolicyOutput:
        """Run the compact postflop network without changing row semantics."""

        return self.postflop(pub, combo_ranges, actor_seat, masks)

    def forward(
        self, pub: PokerPublicBatch, ranges: Tensor, actor_seat: Tensor | None = None,
        *, preflop_masks: PreflopMasks | None = None, postflop_masks: PostflopMasks | None = None,
    ) -> PolicyOutput:
        """Route a combo-range batch and return uniformly combo-indexed rows."""

        if ranges.ndim != 3 or ranges.shape[-1] != self.n_hands:
            raise ValueError(
                f"bundle ranges must have shape [B,P,{self.n_hands}]"
            )
        batch = ranges.shape[0]
        actor = pub.to_act if actor_seat is None else actor_seat
        pre_rows = (pub.street == 0).nonzero(as_tuple=False).squeeze(1)
        post_rows = (pub.street != 0).nonzero(as_tuple=False).squeeze(1)
        outputs: dict[int, PolicyOutput] = {}
        if pre_rows.numel():
            selected = ranges[pre_rows]
            class_ranges = torch.zeros(
                selected.shape[:2] + (self.n_classes,),
                dtype=selected.dtype,
                device=selected.device,
            )
            ids = self.combo_class.view(1, 1, -1).expand_as(selected)
            class_ranges.scatter_add_(2, ids, selected)
            compact = self.forward_preflop(
                self._slice_public(pub, pre_rows), class_ranges, actor[pre_rows],
                self._slice_pre_masks(preflop_masks, pre_rows, batch),
            )
            lifted = PolicyOutput(
                {key: value[:, self.combo_class] for key, value in compact.raw.items()},
                compact.v[:, self.combo_class], compact.t, compact.c,
            )
            for local, global_row in enumerate(pre_rows.tolist()):
                outputs[global_row] = PolicyOutput({k: v[local:local+1] for k, v in lifted.raw.items()},
                                                   lifted.v[local:local+1],
                                                   None if lifted.t is None else lifted.t[local:local+1],
                                                   None if lifted.c is None else lifted.c[local:local+1])
        if post_rows.numel():
            if postflop_masks is None:
                raise ValueError("postflop rows require postflop_masks")
            post = self.postflop(self._slice_public(pub, post_rows), ranges[post_rows], actor[post_rows],
                                 self._slice_post_masks(postflop_masks, post_rows, batch))
            for local, global_row in enumerate(post_rows.tolist()):
                outputs[global_row] = PolicyOutput({k: v[local:local+1] for k, v in post.raw.items()},
                                                   post.v[local:local+1],
                                                   None if post.t is None else post.t[local:local+1],
                                                   None if post.c is None else post.c[local:local+1])
        raw = {key: torch.cat([outputs[row].raw[key] for row in range(batch)]) for key in next(iter(outputs.values())).raw}
        value = torch.cat([outputs[row].v for row in range(batch)])
        t = None if any(outputs[row].t is None for row in range(batch)) else torch.cat([outputs[row].t for row in range(batch)])
        cost = None if any(outputs[row].c is None for row in range(batch)) else torch.cat([outputs[row].c for row in range(batch)])
        return PolicyOutput(raw, value, t, cost)



__all__ = ["PokerModelBundle", "PokerPolicyAdapter"]
