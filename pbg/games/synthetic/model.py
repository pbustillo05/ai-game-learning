"""Small neural policy/value model for the synthetic game family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from pbg.games.synthetic.exact import PolicyTables, PublicTreeIndex
from pbg.games.synthetic.family import SyntheticPBG


# CONTRACT-QUESTION: PolicyOutput is frozen in core/policy_adapter.py, but P00
# explicitly required that file to remain docstring-only and P01 may not edit it.
@dataclass
class PolicyOutput:
    """Synthetic model output with values ``[B, X]`` and masked raw logits."""

    raw: Tensor  # float32 [B, X, E]
    v: Tensor  # float32 [B, X]
    vvec: Tensor | None = None  # float32 [B, P]
    u: Tensor | None = None  # float32 [B]


class SyntheticNet(nn.Module):
    """Embed public keys and private types, then emit categorical logits and V."""

    def __init__(
        self,
        x_max: int,
        e_max: int,
        cfg: Mapping[str, Any] | None = None,
        *,
        n_buckets: int = 4096,
        hidden_dim: int = 64,
    ):
        """Create the two-layer MLP from a game-owned configuration mapping."""

        super().__init__()
        values = dict(cfg or {})
        n_buckets = int(values.get("n_buckets", n_buckets))
        hidden_dim = int(values.get("hidden_dim", hidden_dim))
        if x_max <= 0 or e_max <= 0 or n_buckets <= 0 or hidden_dim <= 0:
            raise ValueError("x_max, e_max, n_buckets, and hidden_dim must be positive")
        self.x_max = x_max
        self.e_max = e_max
        self.n_buckets = n_buckets
        self.hidden_dim = hidden_dim
        self.public_embedding = nn.Embedding(n_buckets, hidden_dim)
        self.type_embedding = nn.Embedding(x_max, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, e_max)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, pub_keys: Tensor, legal_mask: Tensor) -> PolicyOutput:
        """Return masked logits ``[B, X, E]`` and values ``[B, X]``.

        ``pub_keys`` is int64 ``[B]`` and ``legal_mask`` is bool ``[B, E]``.
        Each row must contain at least one legal player event.
        """

        if pub_keys.ndim != 1:
            raise ValueError("pub_keys must have shape [B]")
        if legal_mask.shape != (pub_keys.shape[0], self.e_max) or legal_mask.dtype != torch.bool:
            raise ValueError("legal_mask must be bool [B, E]")
        if not legal_mask.any(dim=-1).all():
            raise ValueError("every model row must contain a legal event")
        batch = pub_keys.shape[0]
        public = self.public_embedding(torch.remainder(pub_keys, self.n_buckets))
        public = public[:, None, :].expand(batch, self.x_max, self.hidden_dim)
        type_ids = torch.arange(self.x_max, device=pub_keys.device)
        private = self.type_embedding(type_ids)[None, :, :].expand(
            batch, self.x_max, self.hidden_dim
        )
        hidden = self.mlp(torch.cat((public, private), dim=-1))  # [B, X, H]
        logits = self.policy_head(hidden)  # [B, X, E]
        logits = logits.masked_fill(~legal_mask[:, None, :], -torch.inf)  # [B, X, E]
        value = self.value_head(hidden).squeeze(-1)  # [B, X]
        return PolicyOutput(raw=logits, v=value)


class SyntheticPolicyAdapter:
    """Exact categorical policy operations over synthetic event logits."""

    def log_prob(
        self,
        out: PolicyOutput,
        event_ids: Tensor,
        legal_event_ids: Tensor,
        legal_event_mask: Tensor,
    ) -> Tensor:
        """Map int64 event ids ``[B, E_sel]`` to log-probs ``[B, E_sel, X]``."""

        if legal_event_ids.ndim != 2 or legal_event_mask.shape != legal_event_ids.shape:
            raise ValueError("legal events and mask must have shape [B,E_legal]")
        if legal_event_ids.shape[0] != out.raw.shape[0]:
            raise ValueError("legal event rows must align with policy rows")
        full_legal_count = torch.zeros_like(out.raw, dtype=torch.int32)
        ids = legal_event_ids.to(out.raw.device).long()[:, None, :].expand(
            out.raw.shape[0], out.raw.shape[1], -1
        )
        full_legal_count.scatter_add_(
            2,
            ids,
            legal_event_mask.to(device=out.raw.device, dtype=torch.int32)[:, None, :].expand_as(ids),
        )
        full_legal = full_legal_count > 0
        logits = out.raw.masked_fill(~full_legal, -torch.inf)
        if event_ids.ndim == 1:
            event_ids = event_ids[:, None]
        if event_ids.ndim != 2 or event_ids.shape[0] != out.raw.shape[0]:
            raise ValueError("event_ids must have shape [B, E_sel]")
        if ((event_ids < 0) | (event_ids >= out.raw.shape[-1])).any():
            raise ValueError("event id is outside the categorical head")
        log_policy = torch.log_softmax(logits, dim=-1)  # [B, X, E]
        gather_ids = event_ids[:, None, :].expand(
            out.raw.shape[0], out.raw.shape[1], event_ids.shape[1]
        )
        selected = torch.gather(log_policy, 2, gather_ids)  # [B, X, E_sel]
        return selected.transpose(1, 2)  # [B, E_sel, X]

    def entropy(
        self,
        out: PolicyOutput,
        legal_event_ids: Tensor,
        legal_event_mask: Tensor,
    ) -> Tensor:
        """Return exact categorical entropy ``[B, X]``."""

        # Avoid differentiating through ``0 * -inf`` at masked events.  This is
        # algebraically log_softmax on the finite (legal) subset.
        legal_count = torch.zeros_like(out.raw, dtype=torch.int32)
        ids = legal_event_ids.to(out.raw.device).long()[:, None, :].expand(
            out.raw.shape[0], out.raw.shape[1], -1
        )
        legal_count.scatter_add_(
            2,
            ids,
            legal_event_mask.to(device=out.raw.device, dtype=torch.int32)[:, None, :].expand_as(ids),
        )
        legal = legal_count > 0
        finite_logits = torch.where(legal, out.raw, torch.zeros_like(out.raw))
        maximum = torch.where(legal, out.raw, torch.full_like(out.raw, -torch.inf)).amax(
            dim=-1, keepdim=True
        )
        unnormalized = torch.where(
            legal, torch.exp(finite_logits - maximum), torch.zeros_like(out.raw)
        )
        normalizer = unnormalized.sum(dim=-1, keepdim=True)
        policy = unnormalized / normalizer
        log_policy = torch.where(
            legal, finite_logits - maximum - torch.log(normalizer), torch.zeros_like(out.raw)
        )
        terms = policy * log_policy
        return -terms.sum(dim=-1)  # [B, X]

    def kl(self, out_new: PolicyOutput, out_ref: PolicyOutput) -> Tensor:
        """Return ``KL(new || ref)`` for every node/type as ``[B, X]``."""

        if out_new.raw.shape != out_ref.raw.shape:
            raise ValueError("new and reference logits must have identical shapes")
        log_new = torch.log_softmax(out_new.raw, dim=-1)  # [B, X, E]
        log_ref = torch.log_softmax(out_ref.raw, dim=-1)  # [B, X, E]
        probability = torch.softmax(out_new.raw, dim=-1)  # [B, X, E]
        difference = torch.where(
            probability > 0.0, log_new - log_ref, torch.zeros_like(log_new)
        )
        return (probability * difference).sum(dim=-1)  # [B, X]

    def centered_logits(self, out: PolicyOutput, event_ids: Tensor) -> Tensor:
        """Return selected logits centered over legal events ``[B, E_sel, X]``."""

        finite = torch.isfinite(out.raw)
        count = finite.sum(dim=-1, keepdim=True).clamp_min(1)
        total = torch.where(finite, out.raw, torch.zeros_like(out.raw)).sum(
            dim=-1, keepdim=True
        )
        centered = out.raw - total / count
        temporary = PolicyOutput(raw=centered, v=out.v, vvec=out.vvec, u=out.u)
        if event_ids.ndim == 1:
            event_ids = event_ids[:, None]
        gather_ids = event_ids[:, None, :].expand(
            centered.shape[0], centered.shape[1], event_ids.shape[1]
        )
        return torch.gather(temporary.raw, 2, gather_ids).transpose(1, 2)


def export_policy_tables(
    net: SyntheticNet, game: SyntheticPBG, tree_index: PublicTreeIndex
) -> PolicyTables:
    """Evaluate ``net`` at every player node and return float64 ``[X, E]`` tables."""

    decision_indices = [
        index
        for index, owner in enumerate(tree_index.owners)
        if 0 <= int(owner) < game.n_players
    ]
    if not decision_indices:
        return {}
    device = next(net.parameters()).device
    refs = torch.tensor(
        [tree_index.nodes[index] for index in decision_indices],
        dtype=torch.int32,
        device=device,
    )
    pub_keys = game.encode_public(refs)
    event_ids, event_mask = game.legal_events(refs)
    legal_mask = torch.zeros(
        (len(decision_indices), net.e_max), dtype=torch.bool, device=device
    )  # [B, E]
    for row in range(len(decision_indices)):
        events = event_ids[row, event_mask[row]].to(torch.int64)
        if (events >= net.e_max).any():
            raise ValueError("network event head is too small for a player event")
        legal_mask[row, events] = True
    was_training = net.training
    net.eval()
    with torch.no_grad():
        output = net(pub_keys, legal_mask)
        probabilities = torch.softmax(output.raw, dim=-1).cpu().numpy().astype(np.float64)
    net.train(was_training)
    return {
        int(tree_index.node_keys[index]): probabilities[row]
        for row, index in enumerate(decision_indices)
    }
