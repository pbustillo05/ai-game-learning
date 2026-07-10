"""Compact combo-level postflop policy and value network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from pbg.games.poker.cards import N_HANDS
from pbg.games.poker.folded import PartitionSummaryBatch
from pbg.games.poker.model.heads import MomentHeads, RaiseMixtureHead, TypeHead, ValueHead
from pbg.games.poker.model.incidence import (
    CompactMap,
    build_compact_map,
    compact_gather,
    compact_scatter,
    permute_rows,
)
from pbg.games.poker.model.preflop import PokerModelConfig, PokerPublicBatch, PolicyOutput
from pbg.games.poker.model.primitives import FFN, LN, MHA


@dataclass
class PostflopMasks:
    """Representation-boundary inputs for compact postflop tokenization.

    ``compatibility`` is the required public bool map ``[B,1326]``.
    ``combo_mask`` may further mask player rows as bool ``[B,P,1326]``.
    Evaluator summaries may be a ready row tensor or an opaque field mapping;
    fields are only gathered, flattened, concatenated, and learned-projected.
    """

    compatibility: Tensor
    seat_mask: Tensor | None = None
    combo_mask: Tensor | None = None
    status: Tensor | None = None
    combo_embeddings: Tensor | None = None
    evaluator_summary: Tensor | Mapping[str, Tensor] | PartitionSummaryBatch | None = None
    fold_line_features: Tensor | None = None


def _summary_fields(
    summary: Tensor | Mapping[str, Tensor] | PartitionSummaryBatch | None,
    mapping: CompactMap,
    players: int,
    dtype: torch.dtype,
) -> Tensor | None:
    """Turn supplied fields into generic compact row features ``[B,P,Q,S]``."""

    if summary is None:
        return None
    if isinstance(summary, PartitionSummaryBatch):
        summary = summary.fields
    values = [summary] if isinstance(summary, Tensor) else [summary[key] for key in sorted(summary)]
    result: list[Tensor] = []
    batch, compact = mapping.batch_size, mapping.compact_size
    for value in values:
        if not isinstance(value, Tensor) or value.shape[:1] != (batch,):
            raise ValueError("each evaluator summary field must have leading dimension B")
        if value.device != mapping.compact_to_full.device:
            raise ValueError("evaluator summary fields and compatibility must share a device")
        value = value.to(dtype=dtype)
        if value.ndim >= 3 and value.shape[1:3] == (players, mapping.full_size):
            row = compact_gather(value, mapping, dim=2)
            if row.ndim == 3:
                row = row[..., None]
            else:
                row = row.flatten(start_dim=3)
        elif value.ndim >= 2 and value.shape[1] == mapping.full_size:
            row = compact_gather(value, mapping, dim=1)
            if row.ndim == 2:
                row = row[..., None]
            else:
                row = row.flatten(start_dim=2)
            row = row[:, None].expand(-1, players, -1, -1)
        elif value.ndim >= 2 and value.shape[1] == players:
            flat = value.flatten(start_dim=2) if value.ndim > 2 else value[..., None]
            row = flat[:, :, None, :].expand(-1, -1, compact, -1)
        else:
            flat = value.reshape(batch, -1)
            row = flat[:, None, None, :].expand(-1, players, compact, -1)
        result.append(row)
    return torch.cat(result, dim=-1) if result else None


def suit_row_permutation(
    permutation: tuple[int, int, int, int], device: torch.device | str | None = None
) -> Tensor:
    """Return the old-to-new combo-row map for a suit relabeling."""

    if tuple(sorted(permutation)) != (0, 1, 2, 3):
        raise ValueError("permutation must contain each suit exactly once")
    cards = torch.triu_indices(52, 52, offset=1).T
    mapped = (cards // 4) * 4 + torch.tensor(permutation)[cards % 4]
    low, high = mapped.min(dim=1).values, mapped.max(dim=1).values
    # Lexicographic triangular-row offset, purely a representation transform.
    row = low * 51 - low * (low - 1) // 2 + (high - low - 1)
    return row.to(device=device, dtype=torch.long)


def permute_suit_representation(
    board: Tensor,
    row_tensors: Mapping[str, Tensor],
    permutation: tuple[int, int, int, int],
    *,
    row_dim: int = 1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Relabel board ids and fixed-width combo-row tensors consistently."""

    row_map = suit_row_permutation(permutation, board.device)
    valid = board >= 0
    mapped_board = board.clone()
    mapped_board[valid] = (board[valid] // 4) * 4 + torch.tensor(
        permutation, dtype=board.dtype, device=board.device
    )[board[valid] % 4]
    return mapped_board, {
        name: permute_rows(value, row_map.to(value.device), dim=row_dim)
        for name, value in row_tensors.items()
    }


class PostflopNet(nn.Module):
    """Pre-LN transformer over evaluator-selected live combo rows."""

    def __init__(self, cfg: PokerModelConfig | Mapping[str, object] | None = None) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, PokerModelConfig) else PokerModelConfig.from_dict(cfg)
        c = self.cfg
        d, dc = c.d_model, c.class_embedding_dim
        self.rank_embedding = nn.Embedding(13, dc)
        self.suit_embedding = nn.Embedding(4, dc)
        self.combo_features = FFN(3 * dc, dc, c.ffn_hidden)
        self.summary_projection = nn.Linear(c.evaluator_summary_dim, dc) if c.evaluator_summary_dim else None
        self.row_affine = nn.Linear(2 + dc + 1 + (dc if self.summary_projection else 0), d)
        self.seat_embedding = nn.Embedding(c.max_players, d)
        self.status_embedding = nn.Embedding(3, d)
        self.actor_embedding = nn.Embedding(2, d)
        self.public_scalar_affine = nn.Linear(1, d)
        self.public_type_embedding = nn.Embedding(6, d)
        self.board_projection = nn.Linear(dc, d)
        self.state_token = nn.Parameter(torch.zeros(d))
        self.anchor_tokens = nn.Parameter(torch.empty(c.k_act, d))
        self.anchor_seat_embedding = nn.Embedding(c.max_players, d)
        nn.init.normal_(self.anchor_tokens, std=d**-0.5)

        side_dim = dc + 2 + (dc if self.summary_projection else 0)
        self.layer_norms = nn.ModuleList(LN(d) for _ in range(c.n_layers_post))
        self.attentions = nn.ModuleList(MHA(d, c.n_heads) for _ in range(c.n_layers_post))
        self.layer_ffns = nn.ModuleList(FFN(d + side_dim, d, c.ffn_hidden) for _ in range(c.n_layers_post))
        self.final_norm = LN(d)
        self.global_features = FFN(3 * d, d, c.ffn_hidden)
        self.seat_features = FFN(d + 3, d, c.ffn_hidden)
        readout_dim = d + c.k_act * d + dc + d
        self.type_features = FFN(readout_dim, d, c.ffn_hidden)
        self.raise_features = FFN(readout_dim, d, c.ffn_hidden)
        self.type_head = TypeHead(d)
        self.raise_head = RaiseMixtureHead(d, c.m_raise, c.s_min)
        self.value_head = ValueHead(readout_dim, feature_dim=d, hidden=c.ffn_hidden)
        self.moment_heads = MomentHeads(2 * d, d, c.ffn_hidden) if c.use_moment_heads else None
        self.register_buffer("combo_cards", torch.triu_indices(52, 52, offset=1).T, persistent=True)

    def _card_embedding(self, cards: Tensor) -> Tensor:
        safe = cards.clamp(0, 51).long()
        return self.rank_embedding(safe // 4) + self.suit_embedding(safe % 4)

    def _combo_embedding(self, ids: Tensor, supplied: Tensor | None, dtype: torch.dtype) -> Tensor:
        batch, compact = ids.shape
        if supplied is not None:
            expected = (N_HANDS, self.cfg.class_embedding_dim)
            if supplied.shape == expected:
                supplied = supplied[None].expand(batch, -1, -1)
            if supplied.shape != (batch,) + expected:
                raise ValueError("combo_embeddings must have shape [1326,Dc] or [B,1326,Dc]")
            return supplied.to(device=ids.device, dtype=dtype).gather(
                1, ids[..., None].expand(-1, -1, supplied.shape[-1])
            )
        cards = self.combo_cards[ids]
        first, second = self._card_embedding(cards[..., 0]), self._card_embedding(cards[..., 1])
        return self.combo_features(torch.cat((first, second, first * second), dim=-1)).to(dtype)

    def _validate(
        self, pub: PokerPublicBatch, ranges: Tensor, actor_seat: Tensor | None, masks: PostflopMasks
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, CompactMap]:
        if ranges.ndim != 3 or ranges.shape[-1] != N_HANDS or not ranges.is_floating_point():
            raise ValueError("combo_ranges must be floating point [B,P,1326]")
        batch, players = ranges.shape[:2]
        if players > self.cfg.max_players:
            raise ValueError("combo_ranges has more seats than max_players")
        if masks.compatibility.shape != (batch, N_HANDS) or masks.compatibility.dtype != torch.bool:
            raise TypeError("compatibility must be bool [B,1326]")
        if masks.compatibility.device != ranges.device:
            raise ValueError("compatibility and ranges must share a device")
        actor = pub.to_act if actor_seat is None else actor_seat
        actor = actor.to(device=ranges.device, dtype=torch.long)
        if actor.shape != (batch,) or bool(((actor < 0) | (actor >= players)).any()):
            raise ValueError("actor_seat must contain a valid seat for each batch row")
        seat_mask = masks.seat_mask if masks.seat_mask is not None else pub.seat_mask
        if seat_mask is None:
            seat_mask = torch.ones((batch, players), dtype=torch.bool, device=ranges.device)
        if seat_mask.shape != (batch, players) or seat_mask.dtype != torch.bool:
            raise TypeError("seat_mask must be bool [B,P]")
        status = masks.status if masks.status is not None else pub.status
        if status is None:
            status = torch.zeros((batch, players), dtype=torch.long, device=ranges.device)
        status = status.to(device=ranges.device, dtype=torch.long)
        if status.shape != (batch, players) or bool(((status < 0) | (status > 2)).any()):
            raise ValueError("status must contain ids in [0,2]")
        seat_mask = seat_mask & (status != 1)
        if not bool(seat_mask.gather(1, actor[:, None]).all()):
            raise ValueError("the actor seat must be present")
        combo_mask = masks.combo_mask
        if combo_mask is None:
            combo_mask = masks.compatibility[:, None, :].expand(-1, players, -1)
        if combo_mask.shape != ranges.shape or combo_mask.dtype != torch.bool:
            raise TypeError("combo_mask must be bool [B,P,1326]")
        combo_mask = combo_mask & masks.compatibility[:, None, :] & seat_mask[:, :, None]
        if not bool(combo_mask[torch.arange(batch, device=ranges.device), actor].any(dim=1).all()):
            raise ValueError("each actor must have at least one compatible combo row")
        return actor, seat_mask, status, combo_mask, build_compact_map(masks.compatibility)

    def _public_tokens(self, pub: PokerPublicBatch, fold: Tensor | None, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        batch, players = pub.stacks.shape
        groups = [pub.pot[:, None], pub.stacks, pub.committed, pub.street.to(dtype)[:, None],
                  pub.legality_summary.to(dtype), pub.n_ticks.to(dtype)[:, None], pub.raise_grid_bounds.to(dtype)]
        if fold is not None:
            if fold.shape[0] != batch or not fold.is_floating_point():
                raise ValueError("fold_line_features must be floating point with leading B")
            flat = fold.flatten(start_dim=1).to(dtype)
            groups.extend((flat.mean(dim=1, keepdim=True), flat.square().mean(dim=1, keepdim=True)))
        scalar = torch.cat([value.to(dtype) for value in groups], dim=1)
        scalar_tokens = self.public_scalar_affine(scalar[..., None])
        scalar_tokens = scalar_tokens + self.public_type_embedding(
            torch.zeros(scalar.shape, dtype=torch.long, device=scalar.device)
        )
        board_mask = pub.board >= 0
        board_tokens = self.board_projection(self._card_embedding(pub.board).to(dtype))
        board_tokens = board_tokens + self.public_type_embedding(
            torch.full(pub.board.shape, 1, dtype=torch.long, device=pub.board.device)
        )
        state = self.state_token.to(dtype)[None, None, :].expand(batch, 1, -1)
        token = torch.cat((state, scalar_tokens, board_tokens), dim=1)
        valid = torch.cat((torch.ones((batch, 1 + scalar.shape[1]), dtype=torch.bool, device=scalar.device), board_mask), dim=1)
        return token, valid

    def forward(
        self,
        pub: PokerPublicBatch,
        combo_ranges: Tensor,
        actor_seat: Tensor | None = None,
        masks: PostflopMasks | None = None,
    ) -> PolicyOutput:
        """Return fixed-width actor outputs while running only supplied rows."""

        if masks is None:
            raise ValueError("postflop forward requires supplied compatibility")
        actor, seat_mask, status, combo_mask, mapping = self._validate(pub, combo_ranges, actor_seat, masks)
        batch, players = combo_ranges.shape[:2]
        c, d = self.cfg, self.cfg.d_model
        compact_ranges = compact_gather(combo_ranges, mapping, dim=2)
        compact_mask = compact_gather(combo_mask, mapping, dim=2).bool()
        combo_emb = self._combo_embedding(mapping.compact_to_full, masks.combo_embeddings, combo_ranges.dtype)
        summary = _summary_fields(masks.evaluator_summary, mapping, players, combo_ranges.dtype)
        if self.summary_projection is None:
            if summary is not None:
                raise ValueError("evaluator_summary_dim must be configured to supply summaries")
            summary_projected = None
        else:
            if summary is None:
                summary = torch.zeros(
                    (batch, players, mapping.compact_size, c.evaluator_summary_dim),
                    dtype=combo_ranges.dtype, device=combo_ranges.device,
                )
            if summary.shape[-1] != c.evaluator_summary_dim:
                raise ValueError("concatenated evaluator summary width does not match evaluator_summary_dim")
            summary_projected = self.summary_projection(summary)
        log_range = torch.log(compact_ranges.clamp_min(c.eps))
        expanded_combo = combo_emb[:, None].expand(-1, players, -1, -1)
        features = [compact_ranges[..., None], log_range[..., None], expanded_combo,
                    torch.ones_like(compact_ranges)[..., None]]
        if summary_projected is not None:
            features.append(summary_projected)
        row = self.row_affine(torch.cat(features, dim=-1))
        seats = torch.arange(players, device=row.device)[None].expand(batch, -1)
        if pub.button is not None:
            relative = torch.remainder(seats - pub.button[:, None].long(), players)
        else:
            relative = seats
        row = row + self.seat_embedding(relative)[:, :, None]
        row = row + self.status_embedding(status)[:, :, None]
        actor_flag = seats == actor[:, None]
        row = row + self.actor_embedding(actor_flag.long())[:, :, None]
        row_flat = row.reshape(batch, players * mapping.compact_size, d)
        public, public_mask = self._public_tokens(pub, masks.fold_line_features, combo_ranges.dtype)
        actor_relative = relative.gather(1, actor[:, None]).squeeze(1)
        anchors = self.anchor_tokens.to(combo_ranges.dtype)[None] + self.anchor_seat_embedding(actor_relative)[:, None]
        token = torch.cat((row_flat, public, anchors), dim=1)
        row_mask = compact_mask.reshape(batch, -1)
        key_mask = torch.cat((row_mask, public_mask, torch.ones((batch, c.k_act), dtype=torch.bool, device=row.device)), dim=1)
        normalized_reach = compact_ranges.masked_fill(~compact_mask, 0)
        normalized_reach = normalized_reach / normalized_reach.sum(dim=2, keepdim=True).clamp_min(c.eps)
        key_bias = torch.zeros(token.shape[:2], dtype=torch.float32, device=token.device)
        key_bias[:, :row_flat.shape[1]] = torch.log(normalized_reach.clamp_min(c.eps)).reshape(batch, -1).float()
        side_parts = [expanded_combo, compact_ranges[..., None], log_range[..., None]]
        if summary_projected is not None:
            side_parts.append(summary_projected)
        side_rows = torch.cat(side_parts, dim=-1)
        side = torch.zeros(token.shape[:2] + (side_rows.shape[-1],), dtype=side_rows.dtype, device=token.device)
        side[:, :row_flat.shape[1]] = side_rows.reshape(batch, row_flat.shape[1], -1)
        x = token
        for norm, attention, ffn in zip(self.layer_norms, self.attentions, self.layer_ffns, strict=True):
            normalized_x = norm(x)
            residual = x + attention(normalized_x, normalized_x, key_bias, key_mask)
            x = residual + ffn(torch.cat((norm(residual), side), dim=-1))
        final = self.final_norm(x)
        final_rows = final[:, :row_flat.shape[1]].reshape(batch, players, mapping.compact_size, d)
        final_public = final[:, row_flat.shape[1]:-c.k_act]
        final_anchors = final[:, -c.k_act:]
        public_float = public_mask.to(final_public.dtype)
        public_mean = (final_public * public_float[..., None]).sum(1) / public_float.sum(1, keepdim=True)
        actor_rows = final_rows[torch.arange(batch, device=actor.device), actor]
        actor_mask = compact_mask[torch.arange(batch, device=actor.device), actor]
        actor_mass = compact_ranges[torch.arange(batch, device=actor.device), actor] * actor_mask
        actor_pool = (actor_rows * actor_mass[..., None]).sum(1) / actor_mass.sum(1, keepdim=True).clamp_min(c.eps)
        anchor_pool = final_anchors.mean(1)
        global_feature = self.global_features(torch.cat((public_mean, actor_pool, anchor_pool), dim=-1))
        readout = torch.cat((actor_rows, final_anchors.reshape(batch, 1, -1).expand(-1, mapping.compact_size, -1),
                             combo_emb, global_feature[:, None].expand(-1, mapping.compact_size, -1)), dim=-1)
        z_type, z_raise = self.type_features(readout), self.raise_features(readout)
        type_logits = self.type_head(z_type).masked_fill(~actor_mask[..., None], 0)
        omega, mu, scale, iota = self.raise_head(z_raise, z_type)
        omega = omega.masked_fill(~actor_mask[..., None], 0)
        mu = mu.masked_fill(~actor_mask[..., None], 0)
        scale = scale.masked_fill(~actor_mask[..., None], 0)
        iota = iota.masked_fill(~actor_mask, 0)
        value = self.value_head(readout).masked_fill(~actor_mask, 0)
        raw = {
            "type_logits": compact_scatter(type_logits, mapping, dim=1),
            "omega": compact_scatter(omega, mapping, dim=1),
            "mu": compact_scatter(mu, mapping, dim=1),
            "s": compact_scatter(scale, mapping, dim=1),
            "iota": compact_scatter(iota, mapping, dim=1),
        }
        full_value = compact_scatter(value, mapping, dim=1)
        vvec: Tensor | None = None
        u: Tensor | None = None
        if self.moment_heads is not None:
            mass = (compact_ranges * compact_mask).sum(2)
            pooled = (final_rows * (compact_ranges * compact_mask)[..., None]).sum(2) / mass[..., None].clamp_min(c.eps)
            seat_meta = torch.stack((mass, status.to(mass.dtype) / 2, relative.to(mass.dtype) / max(players - 1, 1)), dim=-1)
            seat_summary = self.seat_features(torch.cat((pooled, seat_meta), dim=-1))
            vvec, u = self.moment_heads(torch.cat((global_feature, public_mean), dim=-1), seat_summary, seat_mask)
        return PolicyOutput(raw=raw, v=full_value, vvec=vvec, u=u)
