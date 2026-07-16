"""Policy/value models for the finite synthetic-poker family."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from pbg.core.game import CHANCE
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


@dataclass(frozen=True)
class SyntheticModelConfig:
    """Strict configuration for synthetic MHA or the explicit MLP baseline."""

    architecture: str = "mha"
    hidden_dim: int = 64
    n_buckets: int | None = None
    n_heads: int = 4
    n_layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be positive")
        if self.n_buckets is not None and self.n_buckets <= 0:
            raise ValueError("model.n_buckets must be positive when supplied")
        if self.architecture not in {"mha", "mlp_baseline"}:
            raise ValueError("model.architecture must be 'mha' or 'mlp_baseline'")
        if self.n_heads <= 0 or self.hidden_dim % self.n_heads:
            raise ValueError("model.n_heads must positively divide model.hidden_dim")
        if self.n_layers <= 0:
            raise ValueError("model.n_layers must be positive")

    @classmethod
    def from_dict(
        cls, values: Mapping[str, Any] | None
    ) -> SyntheticModelConfig:
        """Parse a synthetic model mapping while rejecting unknown keys."""

        data = dict(values or {})
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                f"unknown synthetic model configuration key: {unknown[0]}"
            )
        return cls(**data)


class SyntheticNet(nn.Module):
    """Explicit small-MLP baseline for synthetic poker experiments."""

    uses_reach_context = False

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
        values = SyntheticModelConfig.from_dict(cfg)
        n_buckets = n_buckets if values.n_buckets is None else values.n_buckets
        hidden_dim = values.hidden_dim if cfg is not None else hidden_dim
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
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

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
        if bool(((pub_keys < 0) | (pub_keys >= self.n_buckets)).any()):
            raise ValueError(
                "public key is outside the collision-free synthetic embedding capacity"
            )
        public = self.public_embedding(pub_keys)
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


class _ReachBiasedLayer(nn.Module):
    """Pre-norm attention layer with an untempered additive key log-prior."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.temperature = nn.Parameter(torch.ones(heads))
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.norm_incidence = nn.LayerNorm(width)
        self.incidence_source = nn.Linear(width, width, bias=False)
        self.incidence_equal = nn.Linear(width, width, bias=False)
        self.incidence_own = nn.Linear(width, width, bias=False)
        self.incidence_out = nn.Linear(width, width)

    def _incidence(
        self, rows: Tensor, mass: Tensor, card_incidence: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return exact other-card, identical-row, and own-card accumulators."""

        normalized = self.norm_incidence(rows)
        source = self.incidence_source(normalized) * mass[..., None]
        card_accumulator = torch.einsum("xc,bpxd->bpcd", card_incidence, source)
        card_overlap = torch.einsum("xc,bpcd->bpxd", card_incidence, card_accumulator)
        same = source
        other_equal = same.sum(1, keepdim=True) - same
        other_card = card_overlap.sum(1, keepdim=True) - card_overlap - 2.0 * other_equal
        own_card = card_overlap - 2.0 * same
        return other_card, other_equal, own_card

    def forward(
        self,
        sequence: Tensor,
        key_log_prior: Tensor,
        key_valid: Tensor,
        *,
        row_count: int,
        players: int,
        types: int,
        mass: Tensor,
        card_incidence: Tensor,
    ) -> Tensor:
        """Update a mixed row/public/action sequence while preserving row geometry."""

        batch, tokens, _ = sequence.shape
        normalized = self.norm_attention(sequence)
        qkv = self.qkv(normalized).reshape(batch, tokens, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        content = torch.matmul(query, key.transpose(-1, -2))
        scale = self.temperature.abs().clamp_min(1.0e-3)[:, None, None]
        scores = content / (math.sqrt(self.head_width) * scale)
        scores = scores + key_log_prior[:, None, None, :]
        scores = scores.masked_fill(~key_valid[:, None, None, :], -torch.inf)
        probability = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probability, value).transpose(1, 2).reshape(
            batch, tokens, self.width
        )
        sequence = sequence + self.attention_out(attended)
        sequence = sequence + self.ffn(self.norm_ffn(sequence))

        rows = sequence[:, :row_count].reshape(batch, players, types, self.width)
        other_card, other_equal, own_card = self._incidence(
            rows, mass, card_incidence.to(rows.dtype)
        )
        residual = self.incidence_out(
            torch.nn.functional.gelu(
                self.norm_incidence(rows)
                + other_card
                + self.incidence_equal(other_equal)
                + self.incidence_own(own_card)
            )
        )
        sequence = sequence.clone()
        sequence[:, :row_count] = (rows + residual).reshape(batch, row_count, self.width)
        return sequence


class SyntheticMHANet(nn.Module):
    """Scaled MHA poker design over every player/type row.

    Synthetic games have a discrete raise menu rather than the continuous NLHE
    tick lattice, so learned per-event action anchors are the game-appropriate
    factored head. Folded ranges are integrated into a compact per-card pressure
    feature instead of being retained as attention rows.
    """

    uses_reach_context = True

    def __init__(
        self,
        x_max: int,
        e_max: int,
        n_players: int,
        card_incidence: Tensor,
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        values = SyntheticModelConfig.from_dict(cfg)
        if values.architecture != "mha":
            raise ValueError("SyntheticMHANet requires model.architecture: mha")
        if card_incidence.ndim != 2 or card_incidence.shape[0] != x_max:
            raise ValueError("card_incidence must have shape [X,C]")
        if card_incidence.dtype != torch.bool:
            raise TypeError("card_incidence must be boolean")
        n_buckets = values.n_buckets or 4096
        self.x_max = x_max
        self.e_max = e_max
        self.n_players = n_players
        self.n_buckets = n_buckets
        self.hidden_dim = values.hidden_dim
        self.register_buffer("card_incidence", card_incidence.float(), persistent=True)
        self.public_embedding = nn.Embedding(n_buckets, values.hidden_dim)
        self.type_embedding = nn.Embedding(x_max, values.hidden_dim)
        self.player_embedding = nn.Embedding(n_players, values.hidden_dim)
        self.status_embedding = nn.Embedding(3, values.hidden_dim)
        self.action_anchors = nn.Parameter(torch.empty(e_max, values.hidden_dim))
        nn.init.normal_(self.action_anchors, std=values.hidden_dim ** -0.5)
        self.range_features = nn.Linear(2, values.hidden_dim)
        self.card_features = nn.Linear(card_incidence.shape[1], values.hidden_dim, bias=False)
        self.fold_pressure = nn.Linear(1, values.hidden_dim)
        self.public_token = nn.Linear(values.hidden_dim, values.hidden_dim)
        self.layers = nn.ModuleList(
            _ReachBiasedLayer(values.hidden_dim, values.n_heads)
            for _ in range(values.n_layers)
        )
        self.final_norm = nn.LayerNorm(values.hidden_dim)
        self.policy_query = nn.Linear(values.hidden_dim, values.hidden_dim, bias=False)
        self.policy_anchor = nn.Linear(values.hidden_dim, values.hidden_dim, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(e_max))
        self.value_head = nn.Linear(values.hidden_dim, 1)
        nn.init.zeros_(self.policy_query.weight)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def _context(
        self,
        pub_keys: Tensor,
        reach: Tensor | None,
        actor: Tensor | None,
        live: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = pub_keys.numel()
        device = pub_keys.device
        if reach is None:
            reach = torch.zeros(
                (batch, self.n_players, self.x_max), device=device, dtype=torch.float32
            )
        if reach.shape != (batch, self.n_players, self.x_max):
            raise ValueError("reach must have shape [B,P,X]")
        finite = torch.isfinite(reach)
        if not finite.any(-1).all():
            raise ValueError("every synthetic player range must have positive support")
        safe = torch.where(finite, reach.float(), torch.full_like(reach.float(), -torch.inf))
        log_mass = safe - torch.logsumexp(safe, dim=-1, keepdim=True)
        mass = torch.where(finite, torch.exp(log_mass), torch.zeros_like(log_mass))
        if actor is None:
            actor = torch.zeros(batch, dtype=torch.long, device=device)
        actor = actor.to(device=device, dtype=torch.long)
        if actor.shape != (batch,) or bool(((actor < 0) | (actor >= self.n_players)).any()):
            raise ValueError("actor must be integer [B] within the player range")
        if live is None:
            live = torch.ones((batch, self.n_players), dtype=torch.bool, device=device)
        live = live.to(device=device)
        if live.shape != (batch, self.n_players) or live.dtype != torch.bool:
            raise TypeError("live must be bool [B,P]")
        return log_mass, mass, actor, live

    def forward(
        self,
        pub_keys: Tensor,
        legal_mask: Tensor,
        *,
        reach: Tensor | None = None,
        actor: Tensor | None = None,
        live: Tensor | None = None,
    ) -> PolicyOutput:
        """Return all-actor-type policy/value rows for one public batch."""

        if pub_keys.ndim != 1:
            raise ValueError("pub_keys must have shape [B]")
        if legal_mask.shape != (pub_keys.shape[0], self.e_max) or legal_mask.dtype != torch.bool:
            raise ValueError("legal_mask must be bool [B, E]")
        if not legal_mask.any(dim=-1).all():
            raise ValueError("every model row must contain a legal event")
        if bool(((pub_keys < 0) | (pub_keys >= self.n_buckets)).any()):
            raise ValueError("public key is outside the collision-free synthetic embedding capacity")
        batch = pub_keys.numel()
        log_mass, mass, actor, live = self._context(pub_keys, reach, actor, live)
        width = self.hidden_dim
        type_ids = torch.arange(self.x_max, device=pub_keys.device)
        player_ids = torch.arange(self.n_players, device=pub_keys.device)
        public = self.public_embedding(pub_keys)
        card = self.card_features(self.card_incidence.to(public.dtype))
        status = torch.where(
            live,
            torch.ones_like(live, dtype=torch.long),
            torch.full_like(live, 2, dtype=torch.long),
        )
        status.scatter_(1, actor[:, None], 0)

        folded_mass = mass * (~live)[..., None]
        folded_cards = torch.einsum(
            "bpx,xc->bc", folded_mass, self.card_incidence.to(mass.dtype)
        )
        type_pressure = torch.einsum(
            "xc,bc->bx", self.card_incidence.to(mass.dtype), folded_cards
        )
        rows = (
            self.type_embedding(type_ids)[None, None]
            + self.player_embedding(player_ids)[None, :, None]
            + self.status_embedding(status)[:, :, None]
            + card[None, None]
            + public[:, None, None]
            + self.range_features(torch.stack((mass, log_mass.clamp_min(-30.0)), dim=-1))
            + self.fold_pressure(type_pressure[:, None, :, None])
        )
        row_count = self.n_players * self.x_max
        public_rows = self.public_token(public)[:, None]
        anchors = self.action_anchors[None].expand(batch, -1, -1)
        sequence = torch.cat((rows.reshape(batch, row_count, width), public_rows, anchors), dim=1)
        row_valid = (finite := torch.isfinite(log_mass)) & live[..., None]
        key_valid = torch.cat(
            (
                row_valid.reshape(batch, row_count),
                torch.ones((batch, 1 + self.e_max), dtype=torch.bool, device=pub_keys.device),
            ),
            dim=1,
        )
        key_log_prior = torch.cat(
            (
                torch.where(finite, log_mass, torch.zeros_like(log_mass)).reshape(batch, row_count),
                torch.zeros((batch, 1 + self.e_max), device=pub_keys.device),
            ),
            dim=1,
        )
        for layer in self.layers:
            sequence = layer(
                sequence,
                key_log_prior,
                key_valid,
                row_count=row_count,
                players=self.n_players,
                types=self.x_max,
                mass=mass * live[..., None],
                card_incidence=self.card_incidence,
            )
        sequence = self.final_norm(sequence)
        final_rows = sequence[:, :row_count].reshape(
            batch, self.n_players, self.x_max, width
        )
        gather = actor[:, None, None, None].expand(-1, 1, self.x_max, width)
        actor_rows = final_rows.gather(1, gather).squeeze(1)
        action_rows = sequence[:, row_count + 1 :]
        logits = torch.einsum(
            "bxd,bed->bxe", self.policy_query(actor_rows), self.policy_anchor(action_rows)
        ) / math.sqrt(width)
        logits = logits + self.policy_bias
        logits = logits.masked_fill(~legal_mask[:, None], -torch.inf)
        return PolicyOutput(raw=logits, v=self.value_head(actor_rows).squeeze(-1))


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

    def kl(
        self,
        out_new: PolicyOutput,
        out_ref: PolicyOutput,
        legal_event_ids: Tensor | None = None,
        legal_event_mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``KL(new || ref)`` for every node/type as ``[B, X]``."""

        del legal_event_ids, legal_event_mask
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
    net: nn.Module, game: SyntheticPBG, tree_index: PublicTreeIndex
) -> PolicyTables:
    """Evaluate policy-consistent belief inputs down an enumerated public tree."""

    device = next(net.parameters()).device
    reaches: dict[int, Tensor] = {
        0: torch.zeros((game.n_players, game.n_types), dtype=torch.float32, device=device)
    }
    tables: PolicyTables = {}
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for index, ref in enumerate(tree_index.nodes):
            reach = reaches[index]
            owner = int(tree_index.owners[index])
            policy: Tensor | None = None
            if 0 <= owner < game.n_players:
                refs = torch.tensor([ref], dtype=torch.int32, device=device)
                event_ids, event_mask = game.legal_events(refs)
                legal_mask = torch.zeros((1, net.e_max), dtype=torch.bool, device=device)
                events = event_ids[0, event_mask[0]].long()
                legal_mask[0, events] = True
                kwargs: dict[str, Tensor] = {}
                if getattr(net, "uses_reach_context", False):
                    state = game.states.state(ref)
                    kwargs = {
                        "reach": reach[None],
                        "actor": torch.tensor([owner], device=device),
                        "live": torch.tensor([state.live], dtype=torch.bool, device=device),
                    }
                output = net(game.encode_public(refs), legal_mask, **kwargs)
                policy = torch.softmax(output.raw[0], dim=-1)
                tables[int(tree_index.node_keys[index])] = (
                    policy.cpu().numpy().astype(np.float64)
                )
            for event, child in tree_index.children[index]:
                child_reach = reach.clone()
                if 0 <= owner < game.n_players:
                    assert policy is not None
                    child_reach[owner] += torch.log(policy[:, event])
                elif owner == CHANCE:
                    cards = set(game._event_cards(ref, event))
                    conflict = torch.tensor(
                        [not cards.isdisjoint(hand) for hand in game._hand_sets],
                        dtype=torch.bool,
                        device=device,
                    )
                    child_reach[:, conflict] = -torch.inf
                reaches[child] = child_reach
    net.train(was_training)
    return tables
