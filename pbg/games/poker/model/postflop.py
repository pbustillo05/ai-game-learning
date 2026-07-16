"""Compact combo-level postflop policy and value network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.folded import PartitionSummaryBatch
from pbg.games.poker.model.heads import MomentHeads, RaiseMixtureHead, TypeHead, ValueHead
from pbg.games.poker.model.incidence import (
    CompactMap,
    IncidenceResidual,
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

    ``compatibility`` is the required public bool map ``[B,H]``.
    ``combo_mask`` may further mask player rows as bool ``[B,P,H]``.
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
    live_seat_ids: Tensor | None = None
    compact_size: int | None = None


def _require_device(condition: Tensor, message: str) -> None:
    """Validate a scalar condition without synchronizing CUDA to the host."""

    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _summary_fields(
    summary: Tensor | Mapping[str, Tensor] | PartitionSummaryBatch | None,
    mapping: CompactMap,
    players: int,
    dtype: torch.dtype,
) -> Tensor | None:
    """Turn supplied fields into generic compact row features ``[B,P,Q,S]``."""

    if summary is None:
        return None
    if isinstance(summary, PartitionSummaryBatch) or hasattr(summary, "fields"):
        summary = summary.fields
    semantic = {
        "base_log_z", "single_card_log_ratio", "pair_card_log_ratio", "output_valid"
    }
    values = (
        [summary]
        if isinstance(summary, Tensor)
        else [
            summary[key]
            for key in sorted(summary)
            if key not in semantic and not key.startswith("folder_")
        ]
    )
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
    permutation: tuple[int, int, int, int],
    device: torch.device | str | None = None,
    *,
    rank_count: int = 13,
) -> Tensor:
    """Return the old-to-new combo-row map for a suit relabeling."""

    if tuple(sorted(permutation)) != (0, 1, 2, 3):
        raise ValueError("permutation must contain each suit exactly once")
    geometry = DeckGeometry(rank_count)
    cards = torch.tensor(geometry.combo_cards, dtype=torch.long)
    mapped = (cards // 4) * 4 + torch.tensor(permutation)[cards % 4]
    row = torch.tensor(
        [geometry.hand_index(int(a), int(b)) for a, b in mapped],
        dtype=torch.long,
    )
    return row.to(device=device, dtype=torch.long)


def permute_suit_representation(
    board: Tensor,
    row_tensors: Mapping[str, Tensor],
    permutation: tuple[int, int, int, int],
    *,
    row_dim: int = 1,
    rank_count: int = 13,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Relabel board ids and fixed-width combo-row tensors consistently."""

    row_map = suit_row_permutation(
        permutation, board.device, rank_count=rank_count
    )
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

    requires_partition_summary = True

    def __init__(
        self,
        cfg: PokerModelConfig | Mapping[str, object] | None = None,
        geometry: DeckGeometry | int = 13,
    ) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, PokerModelConfig) else PokerModelConfig.from_dict(cfg)
        self.geometry = (
            geometry if isinstance(geometry, DeckGeometry) else DeckGeometry(geometry)
        )
        self.n_cards = self.geometry.n_cards
        self.n_hands = self.geometry.n_hands
        c = self.cfg
        if c.folder_beta_rank > self.n_cards:
            raise ValueError(
                "folder_beta_rank cannot exceed the active compact deck size"
            )
        d, dc = c.d_model, c.class_embedding_dim
        geo_dim = self.n_cards + 2 * c.folder_beta_rank
        self.rank_embedding = nn.Embedding(13, dc)
        self.suit_embedding = nn.Embedding(4, dc)
        self.combo_features = FFN(3 * dc, dc, c.ffn_hidden)
        self.summary_projection = nn.Linear(c.evaluator_summary_dim, dc) if c.evaluator_summary_dim else None
        self.row_affine = nn.Linear(
            3 + dc + geo_dim + d + 1 + (dc if self.summary_projection else 0), d
        )
        self.folder_context = FFN(1 + c.folder_line_dim, d, c.ffn_hidden)
        self.tight_delta_projection = nn.Linear(self.n_cards, d, bias=False)
        self.tight_contraction_projection = nn.Linear(1, d, bias=False)
        self.seat_embedding = nn.Embedding(c.max_players, d)
        self.status_embedding = nn.Embedding(3, d)
        self.actor_embedding = nn.Embedding(2, d)
        self.public_scalar_affine = nn.Linear(1, d)
        self.public_type_embedding = nn.Embedding(12, d)
        self.history_numeric_affine = nn.Linear(6, d)
        self.history_action_embedding = nn.Embedding(4, d)
        self.history_street_embedding = nn.Embedding(4, d)
        self.history_position_embedding = nn.Embedding(c.max_history_tokens, d)
        self.board_projection = nn.Linear(dc, d)
        self.state_token = nn.Parameter(torch.zeros(d))
        self.anchor_tokens = nn.Parameter(torch.empty(c.k_act, d))
        self.anchor_seat_embedding = nn.Embedding(c.max_players, d)
        nn.init.normal_(self.anchor_tokens, std=d**-0.5)

        static_side_dim = (
            dc + 3 + geo_dim + 2 * c.folder_beta_rank + 2 * d
            + (dc if self.summary_projection else 0)
        )
        side_dim = static_side_dim + 3 * d
        self.layer_norms = nn.ModuleList(LN(d) for _ in range(c.n_layers_post))
        self.attentions = nn.ModuleList(MHA(d, c.n_heads) for _ in range(c.n_layers_post))
        self.incidence_residuals = nn.ModuleList(
            IncidenceResidual(d) for _ in range(c.n_layers_post)
        )
        self.layer_ffns = nn.ModuleList(FFN(d + side_dim, d, c.ffn_hidden) for _ in range(c.n_layers_post))
        self.final_norm = LN(d)
        self.actor_seat_features = FFN(4 * d, d, c.ffn_hidden)
        self.global_features = FFN((2 + c.max_players) * d, d, c.ffn_hidden)
        self.seat_features = FFN(d + 3, d, c.ffn_hidden)
        self.fold_features = FFN(d + 3, d, c.ffn_hidden)
        readout_dim = d + c.k_act * d + dc + d
        self.type_features = FFN(readout_dim, d, c.ffn_hidden)
        self.raise_features = FFN(readout_dim, d, c.ffn_hidden)
        self.type_head = TypeHead(d)
        self.raise_head = RaiseMixtureHead(d, c.m_raise, c.s_min)
        self.value_head = ValueHead(readout_dim, feature_dim=d, hidden=c.ffn_hidden)
        self.moment_heads = MomentHeads(2 * d, d, c.ffn_hidden) if c.use_moment_heads else None
        combo_cards = torch.tensor(self.geometry.combo_cards, dtype=torch.long)
        self.register_buffer("combo_cards", combo_cards, persistent=True)
        combo_ordinals = combo_cards - self.geometry.first_card
        self.register_buffer("combo_ordinals", combo_ordinals, persistent=True)
        two_hot = torch.zeros((self.n_hands, self.n_cards), dtype=torch.float32)
        two_hot.scatter_(1, combo_ordinals, 1.0)
        self.register_buffer("combo_two_hot", two_hot, persistent=True)

    def _folded_geometry(
        self,
        summary: Tensor | Mapping[str, Tensor] | PartitionSummaryBatch | None,
        mapping: CompactMap,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Return evaluator-derived Lambda and signed beta coordinates."""

        batch, compact = mapping.batch_size, mapping.compact_size
        device = mapping.compact_to_full.device
        combo_ids = mapping.compact_to_full
        cards = self.combo_ordinals[combo_ids]
        geometry = self.combo_two_hot[combo_ids].to(dtype)
        fields: Mapping[str, Tensor] | None
        if isinstance(summary, PartitionSummaryBatch) or hasattr(summary, "fields"):
            fields = summary.fields
        elif isinstance(summary, Mapping):
            fields = summary
        else:
            fields = None
        alpha = None if fields is None else fields.get("single_card_log_ratio")
        beta = None if fields is None else fields.get("pair_card_log_ratio")
        valid = None if fields is None else fields.get("output_valid")
        if alpha is None:
            lambda_hand = torch.zeros((batch, compact), dtype=dtype, device=device)
        else:
            if alpha.shape != (batch, self.n_cards):
                raise ValueError(
                    f"single_card_log_ratio must have shape [B,{self.n_cards}]"
                )
            alpha = alpha.to(device=device, dtype=dtype)
            lambda_hand = alpha.gather(1, cards[..., 0]) + alpha.gather(1, cards[..., 1])
        if beta is not None:
            if beta.shape != (batch, self.n_cards, self.n_cards):
                raise ValueError("pair_card_log_ratio has the wrong deck width")
            beta = beta.to(device=device, dtype=torch.float32)
            batch_ids = torch.arange(batch, device=device)[:, None]
            lambda_hand = lambda_hand + beta[
                batch_ids, cards[..., 0], cards[..., 1]
            ].to(dtype)
        rank = self.cfg.folder_beta_rank
        if rank:
            signed = torch.zeros((batch, compact, 2 * rank), dtype=dtype, device=device)
            supplied_value = None if fields is None else fields.get("beta_eigenvalue")
            supplied_vector = None if fields is None else fields.get("beta_eigenvector")
            if supplied_value is not None or supplied_vector is not None:
                if supplied_value is None or supplied_vector is None:
                    raise ValueError("beta eigenvalue/vector fields must be supplied together")
                if supplied_value.shape[0] != batch or supplied_value.shape[1] < rank:
                    raise ValueError("beta_eigenvalue has insufficient rank")
                if supplied_vector.shape != (
                    batch, self.n_cards, supplied_value.shape[1]
                ):
                    raise ValueError("beta_eigenvector has the wrong shape")
                selected_value = supplied_value[:, :rank].to(device=device, dtype=torch.float32)
                selected_vector = supplied_vector[:, :, :rank].to(device=device, dtype=torch.float32)
            elif beta is not None:
                symmetric = 0.5 * (beta + beta.transpose(1, 2))
                eigenvalue, eigenvector = torch.linalg.eigh(symmetric)
                indices = torch.topk(eigenvalue.abs(), k=rank, dim=1).indices
                selected_value = eigenvalue.gather(1, indices)
                selected_vector = eigenvector.gather(
                    2, indices[:, None, :].expand(-1, self.n_cards, -1)
                )
            else:
                selected_value = None
                selected_vector = None
            if selected_value is not None and selected_vector is not None:
                first = selected_vector.gather(
                    1, cards[..., 0, None].expand(-1, -1, rank)
                )
                second = selected_vector.gather(
                    1, cards[..., 1, None].expand(-1, -1, rank)
                )
                coordinate = (first + second) * selected_value.abs().sqrt()[:, None, :]
                positive = coordinate.masked_fill(selected_value[:, None, :] < 0, 0.0)
                negative = coordinate.masked_fill(selected_value[:, None, :] >= 0, 0.0)
                signed = torch.cat((positive, negative), dim=-1).to(dtype)
            geometry = torch.cat((geometry, signed), dim=-1)
        if valid is not None:
            if valid.shape != (batch,) or valid.dtype != torch.bool:
                raise TypeError("output_valid must be bool [B]")
            lambda_hand = lambda_hand.masked_fill(~valid[:, None], 0.0)
            geometry = torch.cat(
                (
                    geometry[..., : self.n_cards],
                    geometry[..., self.n_cards :].masked_fill(
                        ~valid[:, None, None], 0.0
                    ),
                ),
                dim=-1,
            )
        lambda_hand = lambda_hand.masked_fill(~mapping.valid, 0.0)
        geometry = geometry.masked_fill(~mapping.valid[..., None], 0.0)
        return lambda_hand, geometry

    def _folder_context(
        self,
        summary: Tensor | Mapping[str, Tensor] | PartitionSummaryBatch | None,
        mapping: CompactMap,
        players: int,
        dtype: torch.dtype,
        board: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build per-card pressure, per-folder seat context, and products."""

        if isinstance(summary, PartitionSummaryBatch) or hasattr(summary, "fields"):
            fields: Mapping[str, Tensor] | None = summary.fields
        elif isinstance(summary, Mapping):
            fields = summary
        else:
            fields = None
        batch, compact = mapping.batch_size, mapping.compact_size
        device = mapping.compact_to_full.device
        empty_pressure = torch.zeros((batch, compact, self.cfg.d_model), dtype=dtype, device=device)
        empty_seats = torch.zeros((batch, players, self.cfg.d_model), dtype=dtype, device=device)
        if fields is None or "folder_log_mass" not in fields:
            return empty_pressure, empty_seats, empty_pressure
        log_mass = fields["folder_log_mass"].to(device=device, dtype=dtype)
        folder_count = log_mass.shape[1]
        single = fields.get("folder_single_card_log_ratio")
        line = fields.get("folder_line_meta")
        mask = fields.get("folder_mask")
        seat = fields.get("folder_seat")
        valid = fields.get("folder_output_valid")
        card_marginal = fields.get("folder_card_marginal")
        supplied_delta = fields.get("folder_delta")
        if card_marginal is None and (
            single is None
            or single.shape != (batch, folder_count, self.n_cards)
        ):
            raise ValueError("folder card marginals or single-card ratios are required")
        if card_marginal is not None and card_marginal.shape != (
            batch, folder_count, self.n_cards
        ):
            raise ValueError("folder_card_marginal has the wrong deck width")
        if line is None or line.shape != (batch, folder_count, self.cfg.folder_line_dim):
            raise ValueError("folder_line_meta has the wrong shape")
        if mask is None or mask.shape != (batch, folder_count) or mask.dtype != torch.bool:
            raise TypeError("folder_mask must be bool [B,F]")
        if valid is not None:
            if valid.shape != mask.shape or valid.dtype != torch.bool:
                raise TypeError("folder_output_valid must be bool [B,F]")
            mask = mask & valid
        marginal = (
            (1.0 - single.to(device=device, dtype=dtype).exp()).clamp(0.0, 1.0)
            if single is not None
            else card_marginal.to(device=device, dtype=dtype)
            / log_mass.exp().clamp_min(self.cfg.eps)[..., None]
        )
        context = self.folder_context(
            torch.cat((log_mass[..., None], line.to(device=device, dtype=dtype)), dim=-1)
        )
        context = context * mask[..., None]
        unnormalized = (
            log_mass.exp()[..., None] * marginal
            if card_marginal is None
            else card_marginal.to(device=device, dtype=dtype)
        ) * mask[..., None]
        card_pressure = torch.einsum("bfc,bfd->bcd", unnormalized, context)
        cards = self.combo_ordinals[mapping.compact_to_full]
        pressure = card_pressure.gather(
            1, cards[..., 0, None].expand(-1, -1, self.cfg.d_model)
        ) + card_pressure.gather(
            1, cards[..., 1, None].expand(-1, -1, self.cfg.d_model)
        )
        folder_pool = context * unnormalized.sum(2, keepdim=True)
        seat_context = empty_seats.clone()
        if seat is not None:
            if seat.shape != (batch, folder_count):
                raise ValueError("folder_seat must have shape [B,F]")
            safe_seat = seat.to(device=device, dtype=torch.long).clamp(0, players - 1)
            seat_context.scatter_add_(
                1,
                safe_seat[..., None].expand(-1, -1, self.cfg.d_model),
                context + folder_pool,
            )
        board_ordinal = (board - self.geometry.first_card).clamp(
            0, self.n_cards - 1
        ).long()
        board_indicator = torch.nn.functional.one_hot(
            board_ordinal, self.n_cards
        ).bool() & (board >= 0)[..., None]
        available = ~board_indicator.any(1)
        real_count = (board >= 0).sum(1)
        uniform = available.to(dtype) * (
            2.0 / (self.n_cards - real_count).to(dtype)
        )[:, None]
        if supplied_delta is not None:
            if supplied_delta.shape != (
                batch, folder_count, self.n_cards
            ):
                raise ValueError("folder_delta has the wrong deck width")
            delta = supplied_delta.to(device=device, dtype=dtype) * mask[..., None]
        else:
            delta = (marginal - uniform[:, None, :]) * mask[..., None]
        projected_delta = self.tight_delta_projection(delta)
        total_delta = projected_delta.sum(1)
        pair_products = 0.5 * (
            total_delta.square() - projected_delta.square().sum(1)
        )
        tight_scalar = fields.get("tight_cluster_scalar")
        if tight_scalar is not None:
            if tight_scalar.shape != (batch,):
                raise ValueError("tight_cluster_scalar must have shape [B]")
            pair_products = pair_products + self.tight_contraction_projection(
                tight_scalar.to(device=device, dtype=dtype)[:, None]
            )
        pressure = pressure.masked_fill(~mapping.valid[..., None], 0.0)
        tight = pair_products[:, None, :].expand(-1, compact, -1)
        tight = tight.masked_fill(~mapping.valid[..., None], 0.0)
        return pressure, seat_context, tight

    def _card_embedding(self, cards: Tensor) -> Tensor:
        safe = cards.clamp(0, 51).long()
        return self.rank_embedding(safe // 4) + self.suit_embedding(safe % 4)

    def _combo_embedding(self, ids: Tensor, supplied: Tensor | None, dtype: torch.dtype) -> Tensor:
        batch, compact = ids.shape
        if supplied is not None:
            expected = (self.n_hands, self.cfg.class_embedding_dim)
            if supplied.shape == expected:
                supplied = supplied[None].expand(batch, -1, -1)
            if supplied.shape != (batch,) + expected:
                raise ValueError("combo_embeddings has the wrong compact hand width")
            return supplied.to(device=ids.device, dtype=dtype).gather(
                1, ids[..., None].expand(-1, -1, supplied.shape[-1])
            )
        cards = self.combo_cards[ids]
        first, second = self._card_embedding(cards[..., 0]), self._card_embedding(cards[..., 1])
        return self.combo_features(torch.cat((first, second, first * second), dim=-1)).to(dtype)

    def _validate(
        self, pub: PokerPublicBatch, ranges: Tensor, actor_seat: Tensor | None, masks: PostflopMasks
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, CompactMap]:
        if (
            ranges.ndim != 3
            or ranges.shape[-1] != self.n_hands
            or not ranges.is_floating_point()
        ):
            raise ValueError(
                f"combo_ranges must be floating point [B,P,{self.n_hands}]"
            )
        batch, players = ranges.shape[:2]
        if players > self.cfg.max_players:
            raise ValueError("combo_ranges has more seats than max_players")
        if (
            masks.compatibility.shape != (batch, self.n_hands)
            or masks.compatibility.dtype != torch.bool
        ):
            raise TypeError(
                f"compatibility must be bool [B,{self.n_hands}]"
            )
        if masks.compatibility.device != ranges.device:
            raise ValueError("compatibility and ranges must share a device")
        actor = pub.to_act if actor_seat is None else actor_seat
        actor = actor.to(device=ranges.device, dtype=torch.long)
        if actor.shape != (batch,):
            raise ValueError("actor_seat must contain a valid seat for each batch row")
        _require_device(
            ((actor >= 0) & (actor < players)).all(),
            "actor_seat must contain a valid seat for each batch row",
        )
        seat_mask = masks.seat_mask if masks.seat_mask is not None else pub.seat_mask
        if seat_mask is None:
            seat_mask = torch.ones((batch, players), dtype=torch.bool, device=ranges.device)
        if seat_mask.shape != (batch, players) or seat_mask.dtype != torch.bool:
            raise TypeError("seat_mask must be bool [B,P]")
        status = masks.status if masks.status is not None else pub.status
        if status is None:
            status = torch.zeros((batch, players), dtype=torch.long, device=ranges.device)
        status = status.to(device=ranges.device, dtype=torch.long)
        if status.shape != (batch, players):
            raise ValueError("status must contain ids in [0,2]")
        _require_device(
            ((status >= 0) & (status <= 2)).all(),
            "status must contain ids in [0,2]",
        )
        seat_mask = seat_mask & (status != 1)
        _require_device(
            seat_mask.gather(1, actor[:, None]).all(),
            "the actor seat must be present",
        )
        combo_mask = masks.combo_mask
        if combo_mask is None:
            combo_mask = masks.compatibility[:, None, :].expand(-1, players, -1)
        if combo_mask.shape != ranges.shape or combo_mask.dtype != torch.bool:
            raise TypeError("combo_mask must be bool with the range shape")
        combo_mask = combo_mask & masks.compatibility[:, None, :] & seat_mask[:, :, None]
        _require_device(
            combo_mask[
                torch.arange(batch, device=ranges.device), actor
            ].any(dim=1).all(),
            "each actor must have at least one compatible combo row",
        )
        live_ids = masks.live_seat_ids
        if live_ids is None:
            seats = torch.arange(players, device=ranges.device)[None].expand(batch, -1)
            ordered = torch.where(seat_mask, seats, players).sort(dim=1).values
            live_count = (
                int(seat_mask.sum(1).max().item())
                if ranges.device.type == "cpu"
                else players
            )
            selected = ordered[:, :live_count]
            live_valid = selected < players
            live_ids = selected.clamp_max(players - 1)
        else:
            if live_ids.ndim != 2 or live_ids.shape[0] != batch:
                raise ValueError("live_seat_ids must have shape [B,L]")
            live_ids = live_ids.to(device=ranges.device, dtype=torch.long)
            _require_device(
                ((live_ids >= 0) & (live_ids < players)).all(),
                "live_seat_ids contains an invalid table seat",
            )
            live_valid = seat_mask.gather(1, live_ids)
        _require_device(
            ((live_ids == actor[:, None]) & live_valid).any(1).all(),
            "live_seat_ids must contain each actor",
        )
        return (
            actor, seat_mask, live_ids, live_valid, status, combo_mask,
            build_compact_map(masks.compatibility, masks.compact_size),
        )

    def _public_tokens(
        self,
        pub: PokerPublicBatch,
        fold: Tensor | None,
        dtype: torch.dtype,
        relative: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, players = pub.stacks.shape
        groups = [
            pub.pot[:, None], pub.stacks, pub.committed, pub.street.to(dtype)[:, None],
            pub.legality_summary.to(dtype), pub.n_ticks.to(dtype)[:, None],
            pub.raise_grid_bounds.to(dtype),
        ]
        type_groups = [
            (1, 0), (players, 1), (players, 2), (1, 3),
            (pub.legality_summary.shape[1], 4), (1, 5), (2, 6),
        ]
        if pub.committed_street is not None:
            groups.append(pub.committed_street.to(dtype))
            type_groups.append((players, 7))
        exact = [
            value.to(dtype)[:, None]
            for value in (
                pub.call_amount, pub.current_bet, pub.min_raise_to,
                pub.max_raise_to, pub.last_full_raise,
            )
            if value is not None
        ]
        if exact:
            groups.append(torch.cat(exact, dim=1))
            type_groups.append((len(exact), 8))
        if pub.legal_action_mask is not None:
            groups.append(pub.legal_action_mask.to(dtype))
            type_groups.append((pub.legal_action_mask.shape[1], 9))
        if pub.raise_ticks is not None:
            groups.append(pub.raise_ticks.to(dtype))
            type_groups.append((pub.raise_ticks.shape[1], 10))
        if fold is not None:
            if fold.shape[0] != batch or not fold.is_floating_point():
                raise ValueError("fold_line_features must be floating point with leading B")
            flat = fold.flatten(start_dim=1).to(dtype)
            groups.extend((flat.mean(dim=1, keepdim=True), flat.square().mean(dim=1, keepdim=True)))
            type_groups.append((2, 11))
        scalar = torch.cat([value.to(dtype) for value in groups], dim=1)
        scalar_tokens = self.public_scalar_affine(scalar[..., None])
        scalar_type = torch.cat([
            torch.full((count,), kind, dtype=torch.long, device=scalar.device)
            for count, kind in type_groups
        ])
        scalar_tokens = scalar_tokens + self.public_type_embedding(scalar_type)[None]
        scalar_tokens[:, 1 : 1 + players] += self.seat_embedding(relative)
        scalar_tokens[:, 1 + players : 1 + 2 * players] += self.seat_embedding(relative)
        board_mask = pub.board >= 0
        board_tokens = self.board_projection(self._card_embedding(pub.board).to(dtype))
        board_tokens = board_tokens + self.public_type_embedding(
            torch.full(pub.board.shape, 1, dtype=torch.long, device=pub.board.device)
        )
        state = self.state_token.to(dtype)[None, None, :].expand(batch, 1, -1)
        token = torch.cat((state, scalar_tokens, board_tokens), dim=1)
        scalar_valid = torch.ones(scalar.shape, dtype=torch.bool, device=scalar.device)
        if pub.seat_mask is not None:
            scalar_valid[:, 1 : 1 + players] = pub.seat_mask
            scalar_valid[:, 1 + players : 1 + 2 * players] = pub.seat_mask
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=scalar.device),
                scalar_valid,
                board_mask,
            ),
            dim=1,
        )
        if pub.history is not None and pub.history.mask.shape[1]:
            history = pub.history
            if history.mask.shape[1] > self.cfg.max_history_tokens:
                raise ValueError("public history exceeds max_history_tokens")
            numeric = torch.stack(
                (
                    history.amount, history.pot_before, history.call_amount,
                    history.current_bet, history.min_raise_to, history.max_raise_to,
                ),
                dim=-1,
            ).to(dtype)
            history_token = self.history_numeric_affine(numeric)
            history_token += self.history_action_embedding(history.action.clamp(0, 3).long())
            history_token += self.history_street_embedding(history.street.clamp(0, 3).long())
            history_token += self.history_position_embedding(
                torch.arange(history.mask.shape[1], device=scalar.device)
            )[None]
            safe_actor = history.actor.clamp(0, players - 1).long()
            history_token += self.seat_embedding(relative.gather(1, safe_actor)) * (
                history.actor >= 0
            )[..., None]
            token = torch.cat((token, history_token), dim=1)
            valid = torch.cat((valid, history.mask), dim=1)
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
        (
            actor, seat_mask, live_ids, live_valid, status, combo_mask, mapping
        ) = self._validate(pub, combo_ranges, actor_seat, masks)
        batch, players = combo_ranges.shape[:2]
        live_players = live_ids.shape[1]
        c, d = self.cfg, self.cfg.d_model
        all_compact_ranges = compact_gather(combo_ranges, mapping, dim=2)
        all_compact_mask = compact_gather(combo_mask, mapping, dim=2).bool()
        live_row_index = live_ids[..., None].expand(-1, -1, mapping.compact_size)
        compact_ranges = all_compact_ranges.gather(1, live_row_index)
        compact_mask = all_compact_mask.gather(1, live_row_index) & live_valid[..., None]
        combo_emb = self._combo_embedding(mapping.compact_to_full, masks.combo_embeddings, combo_ranges.dtype)
        lambda_hand, folded_geometry = self._folded_geometry(
            masks.evaluator_summary, mapping, combo_ranges.dtype
        )
        folder_pressure, folder_seat_context, tight_products = self._folder_context(
            masks.evaluator_summary, mapping, players, combo_ranges.dtype, pub.board
        )
        summary = _summary_fields(masks.evaluator_summary, mapping, players, combo_ranges.dtype)
        if summary is not None:
            summary = summary.gather(
                1,
                live_ids[..., None, None].expand(
                    -1, -1, mapping.compact_size, summary.shape[-1]
                ),
            )
        if self.summary_projection is None:
            # The evaluator DTO also carries mandatory folder-correction fields
            # consumed above.  A zero generic-summary width disables only this
            # optional projection; it does not forbid the richer DTO.
            summary_projected = None
        else:
            if summary is None:
                summary = torch.zeros(
                    (batch, live_players, mapping.compact_size, c.evaluator_summary_dim),
                    dtype=combo_ranges.dtype, device=combo_ranges.device,
                )
            if summary.shape[-1] != c.evaluator_summary_dim:
                raise ValueError("concatenated evaluator summary width does not match evaluator_summary_dim")
            summary_projected = self.summary_projection(summary)
        log_range = torch.log(compact_ranges.clamp_min(c.eps))
        base_reach = compact_ranges.masked_fill(~compact_mask, 0.0)
        base_reach = base_reach / base_reach.sum(dim=2, keepdim=True).clamp_min(c.eps)
        corrected_log_reach = torch.log(base_reach.clamp_min(c.eps)) + lambda_hand[:, None, :]
        corrected_log_reach = corrected_log_reach.masked_fill(~compact_mask, -torch.inf)
        log_normalizer = torch.logsumexp(corrected_log_reach, dim=2, keepdim=True)
        corrected_log_reach = torch.where(
            compact_mask,
            corrected_log_reach - log_normalizer.masked_fill(~torch.isfinite(log_normalizer), 0.0),
            torch.zeros_like(corrected_log_reach),
        )
        expanded_combo = combo_emb[:, None].expand(-1, live_players, -1, -1)
        expanded_geometry = folded_geometry[:, None].expand(-1, live_players, -1, -1)
        expanded_pressure = folder_pressure[:, None].expand(-1, live_players, -1, -1)
        expanded_tight = tight_products[:, None].expand(-1, live_players, -1, -1)
        features = [
            compact_ranges[..., None], log_range[..., None],
            corrected_log_reach[..., None], expanded_combo, expanded_geometry,
            expanded_pressure,
            torch.ones_like(compact_ranges)[..., None],
        ]
        if summary_projected is not None:
            features.append(summary_projected)
        row = self.row_affine(torch.cat(features, dim=-1))
        seats = torch.arange(players, device=row.device)[None].expand(batch, -1)
        if pub.button is not None:
            relative = torch.remainder(seats - pub.button[:, None].long(), players)
        else:
            relative = seats
        live_relative = relative.gather(1, live_ids)
        live_status = status.gather(1, live_ids)
        row = row + self.seat_embedding(live_relative)[:, :, None]
        row = row + self.status_embedding(live_status)[:, :, None]
        actor_flag = live_ids == actor[:, None]
        row = row + self.actor_embedding(actor_flag.long())[:, :, None]
        row_flat = row.reshape(batch, live_players * mapping.compact_size, d)
        public, public_mask = self._public_tokens(
            pub, masks.fold_line_features, combo_ranges.dtype, relative
        )
        actor_relative = relative.gather(1, actor[:, None]).squeeze(1)
        anchors = self.anchor_tokens.to(combo_ranges.dtype)[None] + self.anchor_seat_embedding(actor_relative)[:, None]
        token = torch.cat((row_flat, public, anchors), dim=1)
        row_mask = compact_mask.reshape(batch, -1)
        key_mask = torch.cat((row_mask, public_mask, torch.ones((batch, c.k_act), dtype=torch.bool, device=row.device)), dim=1)
        key_bias = torch.zeros(token.shape[:2], dtype=torch.float32, device=token.device)
        key_bias[:, :row_flat.shape[1]] = corrected_log_reach.reshape(batch, -1).float()
        beta_square = expanded_geometry[..., self.n_cards :].square()
        side_parts = [
            expanded_combo, compact_ranges[..., None], log_range[..., None],
            corrected_log_reach[..., None], expanded_geometry, beta_square,
            expanded_pressure, expanded_tight,
        ]
        if summary_projected is not None:
            side_parts.append(summary_projected)
        side_rows = torch.cat(side_parts, dim=-1)
        x = token
        incidence_mass = corrected_log_reach.exp() * compact_mask
        identical_scale = torch.ones(
            (batch, mapping.compact_size), dtype=combo_ranges.dtype, device=combo_ranges.device
        )
        for norm, attention, incidence, ffn in zip(
            self.layer_norms, self.attentions, self.incidence_residuals,
            self.layer_ffns, strict=True
        ):
            row_state = x[:, :row_flat.shape[1]].reshape(
                batch, live_players, mapping.compact_size, d
            )
            j, overlap_card, overlap_equal, _ = incidence(
                row_state,
                incidence_mass,
                folded_geometry[..., : self.n_cards],
                identical_scale,
            )
            normalized_x = norm(x)
            residual = x + attention.forward_biased(
                normalized_x, normalized_x, key_bias, key_mask
            )
            dynamic = torch.cat(
                (j, overlap_card * overlap_card, overlap_card * overlap_equal), dim=-1
            )
            complete_side = torch.cat((side_rows, dynamic), dim=-1)
            side = torch.zeros(
                token.shape[:2] + (complete_side.shape[-1],),
                dtype=complete_side.dtype,
                device=token.device,
            )
            side[:, :row_flat.shape[1]] = complete_side.reshape(
                batch, row_flat.shape[1], -1
            )
            x = residual + ffn(torch.cat((norm(residual), side), dim=-1))
        final = self.final_norm(x)
        final_rows = final[:, :row_flat.shape[1]].reshape(
            batch, live_players, mapping.compact_size, d
        )
        final_public = final[:, row_flat.shape[1]:-c.k_act]
        final_anchors = final[:, -c.k_act:]
        public_float = public_mask.to(final_public.dtype)
        public_mean = (final_public * public_float[..., None]).sum(1) / public_float.sum(1, keepdim=True)
        actor_live = ((live_ids == actor[:, None]) & live_valid).to(torch.int64).argmax(1)
        actor_rows = final_rows[torch.arange(batch, device=actor.device), actor_live]
        actor_mask = compact_mask[torch.arange(batch, device=actor.device), actor_live]
        actor_anchor_input = torch.cat(
            (
                final_anchors[:, 0], final_anchors[:, 1], final_anchors[:, 2],
                final_anchors[:, 3:].mean(1),
            ),
            dim=-1,
        )
        actor_summary = self.actor_seat_features(actor_anchor_input)
        mass = incidence_mass.sum(2)
        pooled = (final_rows * incidence_mass[..., None]).sum(2) / mass[..., None].clamp_min(c.eps)
        live_seat_meta = torch.stack(
            (
                mass,
                live_status.to(mass.dtype) / 2.0,
                live_relative.to(mass.dtype) / max(c.max_players - 1, 1),
            ),
            dim=-1,
        )
        live_summary = self.seat_features(
            torch.cat((pooled, live_seat_meta), dim=-1)
        )
        full_mass = torch.zeros(
            (batch, players), dtype=mass.dtype, device=mass.device
        ).scatter_add(1, live_ids, mass * live_valid.to(mass.dtype))
        seat_meta = torch.stack(
            (
                full_mass,
                status.to(mass.dtype) / 2.0,
                relative.to(mass.dtype) / max(c.max_players - 1, 1),
            ),
            dim=-1,
        )
        folded_summary = self.fold_features(
            torch.cat((folder_seat_context, seat_meta), dim=-1)
        )
        scattered_live = torch.zeros_like(folded_summary).scatter_add(
            1,
            live_ids[..., None].expand(-1, -1, d),
            live_summary * live_valid[..., None],
        )
        seat_summary = torch.where(
            seat_mask[..., None], scattered_live, folded_summary
        )
        actor_selector = torch.nn.functional.one_hot(actor, players).to(torch.bool)
        seat_summary = torch.where(actor_selector[..., None], actor_summary[:, None], seat_summary)
        seat_summary = seat_summary.masked_fill(~(pub.seat_mask if pub.seat_mask is not None else torch.ones_like(seat_mask))[..., None], 0.0)
        relative_index = relative[..., None].expand(-1, -1, d)
        ordered_seats = torch.zeros_like(seat_summary).scatter(1, relative_index, seat_summary)
        padded_seats = torch.zeros(
            (batch, c.max_players, d), dtype=seat_summary.dtype, device=seat_summary.device
        )
        padded_seats[:, :players] = ordered_seats
        global_feature = self.global_features(
            torch.cat(
                (public_mean, actor_summary, padded_seats.reshape(batch, -1)), dim=-1
            )
        )
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
        # Padded/impossible private rows are excluded by posterior masks in the
        # loss, but the factorized adapter still evaluates a complete fixed-width
        # row. Give those rows a harmless valid distribution rather than the
        # scatter fill value (notably, a zero logistic scale).
        full_actor_mask = compact_scatter(actor_mask, mapping, dim=1)
        invalid = ~full_actor_mask
        raw["omega"] = raw["omega"].masked_fill(invalid[..., None], 1.0 / c.m_raise)
        raw["mu"] = raw["mu"].masked_fill(invalid[..., None], 0.5)
        raw["s"] = raw["s"].masked_fill(invalid[..., None], 1.0)
        raw["iota"] = raw["iota"].masked_fill(invalid, 0.5)
        full_value = compact_scatter(value, mapping, dim=1)
        vvec: Tensor | None = None
        u: Tensor | None = None
        if self.moment_heads is not None:
            vvec, u = self.moment_heads(torch.cat((global_feature, public_mean), dim=-1), seat_summary, seat_mask)
        return PolicyOutput(raw=raw, v=full_value, vvec=vvec, u=u)
