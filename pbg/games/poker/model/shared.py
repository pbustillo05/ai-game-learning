"""Unique parameter owner used by both native poker input families."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from pbg.core.tensor_ops import Linear, Embedding, feature_cat

from pbg.games.poker.model.heads import RaiseMixtureHead, TypeHead, ValueHead
from pbg.games.poker.model.incidence import IncidenceResidual
from pbg.games.poker.model.primitives import FFN, LN, MHA, gelu

if TYPE_CHECKING:
    from pbg.games.poker.model.preflop import PokerModelConfig


class FamilyFFN(nn.Module):
    """Shared dynamic map and output, with native-width static adapters."""

    def __init__(self, d: int, hidden: int, pre: int, post: int) -> None:
        super().__init__()
        self.common = Linear(4 * d, hidden)
        self.static = nn.ModuleDict({
            "pre": Linear(pre, hidden, bias=False),
            "post": Linear(post, hidden, bias=False),
        })
        self.output = Linear(hidden, d)

    def forward(self, common: Tensor, static: Tensor, family: str) -> Tensor:
        return self.output(gelu(self.common(common) + self.static[family](static)))


class SharedPokerParameters(nn.Module):
    """One registered trunk/readout; family encoders receive it as an argument."""

    def __init__(self, cfg: PokerModelConfig, n_cards: int) -> None:
        super().__init__()
        d, dc = cfg.d_model, cfg.class_embedding_dim
        hidden = cfg.ffn_hidden or 4 * d
        self.seat_embedding = Embedding(cfg.max_players, d)
        self.status_embedding = Embedding(3, d)
        self.actor_embedding = Embedding(2, d)
        self.public_scalar_affine = Linear(1, d)
        # pot, stack, commitment, street, legality, ticks, bounds,
        # street commitment, exact betting amounts, action mask, raises, fold.
        self.public_type_embedding = Embedding(13, d)
        self.history_numeric_affine = Linear(6, d)
        self.history_action_embedding = Embedding(4, d)
        self.history_street_embedding = Embedding(4, d)
        self.history_position_embedding = Embedding(cfg.max_history_tokens, d)
        self.state_token = nn.Parameter(torch.zeros(d))
        self.anchor_tokens = nn.Parameter(torch.empty(cfg.k_act, d))
        self.anchor_seat_embedding = Embedding(cfg.max_players, d)
        nn.init.normal_(self.anchor_tokens, std=d**-0.5)
        self.layer_norms = nn.ModuleList(LN(d) for _ in range(cfg.layers))
        self.attentions = nn.ModuleList(MHA(d, cfg.n_heads) for _ in range(cfg.layers))
        self.incidence_residuals = nn.ModuleList(IncidenceResidual(d) for _ in range(cfg.layers))
        pre = dc + 4 + n_cards
        post = dc + 3 + n_cards + 4 * cfg.folder_beta_rank + 2 * d
        self.layer_ffns = nn.ModuleList(
            FamilyFFN(d, hidden, pre, post) for _ in range(cfg.layers)
        )
        self.final_norm = LN(d)
        self.actor_seat_features = FFN(4 * d, d, hidden)
        self.seat_features = FFN(d + 3, d, hidden)
        self.global_features = FFN((2 + cfg.max_players) * d, d, hidden)
        readout_dim = (2 + cfg.k_act) * d + dc
        self.type_features = FFN(readout_dim, d, hidden)
        self.raise_features = FFN(readout_dim, d, hidden)
        self.type_head = TypeHead(d)
        self.raise_head = RaiseMixtureHead(d, cfg.m_raise, cfg.s_min)
        self.value_head = ValueHead(readout_dim, feature_dim=d, hidden=hidden)

    def readout(
        self, actor_rows: Tensor, private_embedding: Tensor,
        anchors: Tensor, global_feature: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Project the decision context once for all private rows."""

        rows = feature_cat((actor_rows, private_embedding), dim=-1)
        context = torch.cat((anchors.flatten(1), global_feature), dim=-1)
        z_type = self.type_features.forward_factored(rows, context)
        z_raise = self.raise_features.forward_factored(rows, context)
        omega, mu, scale, iota = self.raise_head(z_raise, z_type)
        value_features = self.value_head.features.forward_factored(rows, context)
        value = self.value_head.affine(value_features).squeeze(-1)
        return {
            "type_logits": self.type_head(z_type), "omega": omega,
            "mu": mu, "s": scale, "iota": iota,
        }, value
