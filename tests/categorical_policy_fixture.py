"""Game-neutral categorical policy fixture for core loss tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PolicyOutput:
    raw: Tensor
    v: Tensor
    vvec: Tensor | None = None
    u: Tensor | None = None


class CategoricalPolicyAdapter:
    def _legal(self, out: PolicyOutput, ids: Tensor, mask: Tensor) -> Tensor:
        legal = torch.zeros_like(out.raw, dtype=torch.int32)
        expanded = ids.to(out.raw.device).long()[:, None].expand(
            out.raw.shape[0], out.raw.shape[1], -1
        )
        legal.scatter_add_(
            2,
            expanded,
            mask.to(device=out.raw.device, dtype=torch.int32)[:, None].expand_as(expanded),
        )
        return legal > 0

    def log_prob(
        self, out: PolicyOutput, event_ids: Tensor, legal_ids: Tensor, legal_mask: Tensor
    ) -> Tensor:
        legal = self._legal(out, legal_ids, legal_mask)
        log_policy = torch.log_softmax(out.raw.masked_fill(~legal, -torch.inf), dim=-1)
        selected = event_ids[:, None] if event_ids.ndim == 1 else event_ids
        gather = selected[:, None].expand(out.raw.shape[0], out.raw.shape[1], -1)
        return torch.gather(log_policy, 2, gather).transpose(1, 2)

    def entropy(self, out: PolicyOutput, legal_ids: Tensor, legal_mask: Tensor) -> Tensor:
        legal = self._legal(out, legal_ids, legal_mask)
        finite_logits = torch.where(legal, out.raw, torch.zeros_like(out.raw))
        maximum = torch.where(
            legal, out.raw, torch.full_like(out.raw, -torch.inf)
        ).amax(dim=-1, keepdim=True)
        unnormalized = torch.where(
            legal, torch.exp(finite_logits - maximum), torch.zeros_like(out.raw)
        )
        normalizer = unnormalized.sum(dim=-1, keepdim=True)
        probability = unnormalized / normalizer
        log_probability = torch.where(
            legal,
            finite_logits - maximum - torch.log(normalizer),
            torch.zeros_like(out.raw),
        )
        return -(probability * log_probability).sum(-1)

    def kl(
        self,
        out_new: PolicyOutput,
        out_ref: PolicyOutput,
        legal_ids: Tensor | None = None,
        legal_mask: Tensor | None = None,
    ) -> Tensor:
        del legal_ids, legal_mask
        finite = torch.isfinite(out_new.raw) & torch.isfinite(out_ref.raw)
        floor = torch.finfo(out_new.raw.dtype).min
        new_logits = out_new.raw.masked_fill(~finite, floor)
        ref_logits = out_ref.raw.masked_fill(~finite, floor)
        log_new = torch.log_softmax(new_logits, dim=-1)
        log_ref = torch.log_softmax(ref_logits, dim=-1)
        probability = torch.softmax(new_logits, dim=-1)
        difference = torch.where(
            probability > 0, log_new - log_ref, torch.zeros_like(log_new)
        )
        return (probability * difference).sum(-1)
