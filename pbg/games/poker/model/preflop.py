"""The class-level preflop poker policy and value network."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from pbg.games.poker.cards import N_CLASSES, class_size
from pbg.games.poker.model.heads import MomentHeads, RaiseMixtureHead, TypeHead, ValueHead
from pbg.games.poker.model.primitives import FFN, LN, MHA


@dataclass(frozen=True)
class PokerModelConfig:
    """Configuration of the poker transformer and its output heads."""

    d_model: int = 256
    n_heads: int = 8
    n_layers_pre: int = 4
    n_layers_post: int = 4
    m_raise: int = 4
    k_act: int = 7
    s_min: float = 0.02
    n_buckets_line: int = 64
    max_players: int = 9
    class_embedding_dim: int = 32
    evaluator_summary_dim: int = 0
    ffn_hidden: int | None = None
    use_moment_heads: bool = False
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_layers_pre <= 0 or self.n_layers_post <= 0 or self.m_raise <= 0:
            raise ValueError("layer and mixture counts must be positive")
        if self.k_act != 3 + self.m_raise:
            raise ValueError("k_act must equal 3 + m_raise")
        if self.s_min <= 0.0 or self.eps <= 0.0:
            raise ValueError("s_min and eps must be positive")
        if not 2 <= self.max_players <= 9:
            raise ValueError("max_players must be in [2,9]")
        if self.class_embedding_dim <= 0 or self.evaluator_summary_dim < 0:
            raise ValueError("embedding dimensions are invalid")
        if self.ffn_hidden is not None and self.ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive when supplied")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> PokerModelConfig:
        """Parse ``RootConfig.model`` while rejecting unknown model keys."""

        data = dict(values or {})
        if "p_max" in data:
            if "max_players" in data:
                raise ValueError("specify only one of p_max and max_players")
            data["max_players"] = data.pop("p_max")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown poker model configuration key: {unknown[0]}")
        return cls(**data)


@dataclass
class PokerPublicBatch:
    """Public poker tensors consumed by both street-specific networks.

    ``street`` is int ``[B]``; ``board`` is int ``[B,5]`` with padded slots;
    ``pot`` is float ``[B]``; ``stacks`` and ``committed`` are float ``[B,P]``
    in table-seat order; ``to_act`` is int ``[B]``; ``legality_summary`` is
    float ``[B,L]``; ``n_ticks`` is numeric ``[B]``; and
    ``raise_grid_bounds`` is float ``[B,2]``. Optional ``status`` is int
    ``[B,P]`` (live/folded/all-in), ``seat_mask`` is bool ``[B,P]``, and
    ``button`` is int ``[B]`` for conversion to button-relative embeddings.
    """

    street: Tensor
    board: Tensor
    pot: Tensor
    stacks: Tensor
    committed: Tensor
    to_act: Tensor
    legality_summary: Tensor
    n_ticks: Tensor
    raise_grid_bounds: Tensor
    status: Tensor | None = None
    seat_mask: Tensor | None = None
    button: Tensor | None = None


@dataclass
class PreflopMasks:
    """Optional row metadata and masks for a preflop batch."""

    seat_mask: Tensor | None = None  # bool [B,P]
    class_mask: Tensor | None = None  # bool [B,P,169]
    status: Tensor | None = None  # int [B,P]
    class_ids: Tensor | None = None  # int [169] or [B,169]
    class_embeddings: Tensor | None = None  # float [169,Dc] or [B,169,Dc]
    evaluator_summary: Tensor | None = None  # float [B,P,169,S]


@dataclass
class PolicyOutput:
    """Poker model output with class rows along dimension one."""

    raw: dict[str, Tensor]
    v: Tensor  # [B,169]
    vvec: Tensor | None = None  # [B,P_max]
    u: Tensor | None = None  # [B]


def _mask_value(masks: PreflopMasks | Mapping[str, Tensor] | Tensor | None, name: str) -> Tensor | None:
    if masks is None:
        return None
    if isinstance(masks, Tensor):
        return masks if name == "seat_mask" else None
    if isinstance(masks, Mapping):
        return masks.get(name)
    return getattr(masks, name)


class PreflopNet(nn.Module):
    """Pre-LN transformer over all seated-player preflop class rows."""

    def __init__(self, cfg: PokerModelConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, PokerModelConfig) else PokerModelConfig.from_dict(cfg)
        c = self.cfg
        d = c.d_model
        dc = c.class_embedding_dim

        self.class_embedding = nn.Embedding(N_CLASSES, dc)
        self.class_row_affine = nn.Linear(dc + 4, d)
        self.seat_embedding = nn.Embedding(c.max_players, d)
        self.status_embedding = nn.Embedding(3, d)
        self.actor_embedding = nn.Embedding(2, d)
        self.public_scalar_affine = nn.Linear(1, d)
        self.public_type_embedding = nn.Embedding(6, d)
        self.state_token = nn.Parameter(torch.zeros(d))
        self.anchor_tokens = nn.Parameter(torch.empty(c.k_act, d))
        self.anchor_seat_embedding = nn.Embedding(c.max_players, d)
        nn.init.normal_(self.anchor_tokens, std=d**-0.5)

        self.summary_projection = (
            nn.Linear(c.evaluator_summary_dim, dc) if c.evaluator_summary_dim else None
        )
        side_dim = dc + 2 + (dc if c.evaluator_summary_dim else 0)
        self.layer_norms = nn.ModuleList(LN(d) for _ in range(c.n_layers_pre))
        self.attentions = nn.ModuleList(MHA(d, c.n_heads) for _ in range(c.n_layers_pre))
        self.layer_ffns = nn.ModuleList(
            FFN(d + side_dim, d, c.ffn_hidden) for _ in range(c.n_layers_pre)
        )
        self.final_norm = LN(d)
        self.actor_seat_ffn = FFN(4 * d, d, c.ffn_hidden)
        self.seat_ffn = FFN(d + 3, d, c.ffn_hidden)
        self.global_ffn = FFN((2 + c.max_players) * d, d, c.ffn_hidden)

        readout_dim = d + c.k_act * d + dc + d
        self.type_features = FFN(readout_dim, d, c.ffn_hidden)
        self.raise_features = FFN(readout_dim, d, c.ffn_hidden)
        self.type_head = TypeHead(d)
        self.raise_head = RaiseMixtureHead(d, c.m_raise, c.s_min)
        self.value_head = ValueHead(readout_dim, feature_dim=d, hidden=c.ffn_hidden)
        self.moment_heads = MomentHeads(2 * d, d, c.ffn_hidden) if c.use_moment_heads else None

        sizes = torch.tensor([class_size(k) for k in range(N_CLASSES)], dtype=torch.float32)
        self.register_buffer("class_sizes", sizes, persistent=True)

    def _validate_public(self, pub: PokerPublicBatch, class_ranges: Tensor) -> tuple[int, int]:
        if class_ranges.ndim != 3 or class_ranges.shape[-1] != N_CLASSES:
            raise ValueError("class_ranges must have shape [B,P,169]")
        if not class_ranges.is_floating_point():
            raise TypeError("class_ranges must be floating point")
        batch, players = class_ranges.shape[:2]
        if players > self.cfg.max_players:
            raise ValueError("class_ranges has more seats than max_players")
        shapes = {
            "street": (batch,), "pot": (batch,), "stacks": (batch, players),
            "committed": (batch, players), "to_act": (batch,), "n_ticks": (batch,),
            "raise_grid_bounds": (batch, 2),
        }
        for name, shape in shapes.items():
            if getattr(pub, name).shape != shape:
                raise ValueError(f"pub.{name} must have shape {shape}")
        if pub.board.shape != (batch, 5):
            raise ValueError("pub.board must have shape [B,5]")
        if pub.legality_summary.ndim != 2 or pub.legality_summary.shape[0] != batch:
            raise ValueError("pub.legality_summary must have shape [B,L]")
        tensors = (
            pub.street, pub.board, pub.pot, pub.stacks, pub.committed, pub.to_act,
            pub.legality_summary, pub.n_ticks, pub.raise_grid_bounds, class_ranges,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all public and range tensors must be on one device")
        return batch, players

    def _metadata(
        self,
        pub: PokerPublicBatch,
        class_ranges: Tensor,
        actor_seat: Tensor | None,
        masks: PreflopMasks | Mapping[str, Tensor] | Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        batch, players = class_ranges.shape[:2]
        device = class_ranges.device
        actor = pub.to_act if actor_seat is None else actor_seat
        if actor.shape != (batch,):
            raise ValueError("actor_seat must have shape [B]")
        actor = actor.to(device=device, dtype=torch.long)
        if bool(((actor < 0) | (actor >= players)).any()):
            raise ValueError("actor seat is outside the supplied seats")

        seat_mask = _mask_value(masks, "seat_mask")
        if seat_mask is None:
            seat_mask = pub.seat_mask
        if seat_mask is None:
            seat_mask = torch.ones((batch, players), dtype=torch.bool, device=device)
        if seat_mask.shape != (batch, players) or seat_mask.dtype != torch.bool:
            raise TypeError("seat_mask must be bool [B,P]")
        if not bool(seat_mask.gather(1, actor[:, None]).all()):
            raise ValueError("the actor seat must be present")

        class_mask = _mask_value(masks, "class_mask")
        if class_mask is None:
            class_mask = torch.ones_like(class_ranges, dtype=torch.bool)
        if class_mask.shape != class_ranges.shape or class_mask.dtype != torch.bool:
            raise TypeError("class_mask must be bool [B,P,169]")
        class_mask = class_mask & seat_mask[:, :, None]

        status = _mask_value(masks, "status")
        if status is None:
            status = pub.status
        if status is None:
            status = torch.zeros((batch, players), dtype=torch.long, device=device)
        status = status.to(device=device, dtype=torch.long)
        if status.shape != (batch, players) or bool(((status < 0) | (status > 2)).any()):
            raise ValueError("status must contain live/folded/all-in ids [B,P]")

        class_ids = _mask_value(masks, "class_ids")
        if class_ids is None:
            class_ids = torch.arange(N_CLASSES, device=device)[None, :].expand(batch, -1)
        elif class_ids.ndim == 1:
            class_ids = class_ids[None, :].expand(batch, -1)
        class_ids = class_ids.to(device=device, dtype=torch.long)
        if class_ids.shape != (batch, N_CLASSES) or bool(((class_ids < 0) | (class_ids >= N_CLASSES)).any()):
            raise ValueError("class_ids must have shape [169] or [B,169]")

        supplied_embedding = _mask_value(masks, "class_embeddings")
        if supplied_embedding is not None:
            embedding_shape = supplied_embedding.shape
            expected = (N_CLASSES, self.cfg.class_embedding_dim)
            batched_expected = (batch,) + expected
            if embedding_shape not in (expected, batched_expected):
                raise ValueError("class_embeddings has the wrong shape")

        summary = _mask_value(masks, "evaluator_summary")
        if self.summary_projection is None:
            if summary is not None:
                raise ValueError("evaluator_summary_dim must be configured to supply a summary")
        else:
            if summary is None or summary.shape != class_ranges.shape + (self.cfg.evaluator_summary_dim,):
                raise ValueError("evaluator_summary must have shape [B,P,169,S]")

        button = pub.button
        seat_ids = torch.arange(players, device=device)[None, :].expand(batch, -1)
        if button is not None:
            if button.shape != (batch,):
                raise ValueError("pub.button must have shape [B]")
            relative_seats = torch.remainder(seat_ids - button[:, None].long(), players)
        else:
            relative_seats = seat_ids
        return actor, seat_mask, class_mask, status, class_ids, summary, relative_seats

    def _public_tokens(
        self,
        pub: PokerPublicBatch,
        seat_mask: Tensor,
        relative_seats: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, players = seat_mask.shape
        dtype = pub.pot.dtype
        device = pub.pot.device
        scalar_groups = (
            pub.pot[:, None], pub.stacks, pub.committed,
            pub.street.to(dtype)[:, None], pub.legality_summary.to(dtype),
            torch.cat((pub.n_ticks.to(dtype)[:, None], pub.raise_grid_bounds.to(dtype)), dim=1),
        )
        scalars = torch.cat(scalar_groups, dim=1)
        type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=device),
                torch.ones(players, dtype=torch.long, device=device),
                torch.full((players,), 2, dtype=torch.long, device=device),
                torch.full((1,), 3, dtype=torch.long, device=device),
                torch.full((pub.legality_summary.shape[1],), 4, dtype=torch.long, device=device),
                torch.full((3,), 5, dtype=torch.long, device=device),
            )
        )
        token = self.public_scalar_affine(scalars[..., None]) + self.public_type_embedding(type_ids)[None, :, :]
        seat_position = self.seat_embedding(relative_seats)
        token[:, 1 : 1 + players] = token[:, 1 : 1 + players] + seat_position
        token[:, 1 + players : 1 + 2 * players] = token[:, 1 + players : 1 + 2 * players] + seat_position
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=device), seat_mask, seat_mask,
                torch.ones((batch, 1 + pub.legality_summary.shape[1] + 3), dtype=torch.bool, device=device),
            ), dim=1,
        )
        public_mean = (token * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        state = self.state_token[None, :] + public_mean
        return torch.cat((token, state[:, None, :]), dim=1), torch.cat(
            (valid, torch.ones((batch, 1), dtype=torch.bool, device=device)), dim=1
        )

    def forward(
        self,
        pub: PokerPublicBatch,
        class_ranges: Tensor,
        actor_seat: Tensor | None = None,
        masks: PreflopMasks | Mapping[str, Tensor] | Tensor | None = None,
    ) -> PolicyOutput:
        """Return policy parameters and values for every actor class row.

        ``class_ranges`` is float ``[B,P,169]``. ``actor_seat`` defaults to
        ``pub.to_act``. Masks may be :class:`PreflopMasks`, a mapping, or a
        bool seat mask directly.
        """

        batch, players = self._validate_public(pub, class_ranges)
        actor, seat_mask, class_mask, status, class_ids, summary, relative_seats = self._metadata(
            pub, class_ranges, actor_seat, masks
        )
        c = self.cfg
        d = c.d_model
        dc = c.class_embedding_dim

        supplied_embedding = _mask_value(masks, "class_embeddings")
        if supplied_embedding is None:
            class_emb = self.class_embedding(class_ids)
        else:
            if supplied_embedding.ndim == 2:
                supplied_embedding = supplied_embedding[None, :, :].expand(batch, -1, -1)
            class_emb = supplied_embedding.to(device=class_ranges.device, dtype=class_ranges.dtype)
        expanded_emb = class_emb[:, None, :, :].expand(-1, players, -1, -1)
        sizes = self.class_sizes[class_ids].to(class_ranges.dtype) / 12.0
        size_feature = sizes[:, None, :].expand(-1, players, -1)
        log_range = torch.log(class_ranges.clamp_min(c.eps))
        constant = torch.ones_like(class_ranges)
        row_feature = torch.cat(
            (class_ranges[..., None], log_range[..., None], size_feature[..., None], expanded_emb, constant[..., None]),
            dim=-1,
        )
        row = self.class_row_affine(row_feature)
        row = row + self.seat_embedding(relative_seats)[:, :, None, :]
        row = row + self.status_embedding(status)[:, :, None, :]
        actor_flag = torch.arange(players, device=row.device)[None, :] == actor[:, None]
        row = row + self.actor_embedding(actor_flag.long())[:, :, None, :]
        row_flat = row.reshape(batch, players * N_CLASSES, d)

        public_token, public_mask = self._public_tokens(pub, seat_mask, relative_seats)
        actor_relative = relative_seats.gather(1, actor[:, None]).squeeze(1)
        anchors = self.anchor_tokens[None, :, :] + self.anchor_seat_embedding(actor_relative)[:, None, :]
        token = torch.cat((row_flat, public_token, anchors), dim=1)
        row_mask_flat = class_mask.reshape(batch, players * N_CLASSES)
        key_mask = torch.cat(
            (row_mask_flat, public_mask, torch.ones((batch, c.k_act), dtype=torch.bool, device=row.device)), dim=1
        )

        masked_range = class_ranges * class_mask.to(class_ranges.dtype)
        normalized = masked_range / masked_range.sum(dim=-1, keepdim=True).clamp_min(c.eps)
        key_bias = torch.zeros(token.shape[:2], dtype=torch.float32, device=token.device)
        key_bias[:, : players * N_CLASSES] = torch.log(normalized.clamp_min(c.eps)).reshape(
            batch, players * N_CLASSES
        ).float()

        side_class = torch.cat((expanded_emb, class_ranges[..., None], log_range[..., None]), dim=-1)
        if self.summary_projection is not None:
            assert summary is not None
            side_class = torch.cat((side_class, self.summary_projection(summary)), dim=-1)
        side = torch.zeros(
            token.shape[:2] + (side_class.shape[-1],), dtype=side_class.dtype, device=token.device
        )
        side[:, : players * N_CLASSES] = side_class.reshape(batch, players * N_CLASSES, -1)

        x = token
        for layer_norm, attention, layer_ffn in zip(
            self.layer_norms, self.attentions, self.layer_ffns, strict=True
        ):
            normalized_x = layer_norm(x)
            x_hat = x + attention(normalized_x, normalized_x, key_bias, key_mask)
            x = x_hat + layer_ffn(torch.cat((layer_norm(x_hat), side), dim=-1))

        final = self.final_norm(x)
        final_rows = final[:, : players * N_CLASSES].reshape(batch, players, N_CLASSES, d)
        final_public = final[:, players * N_CLASSES : -c.k_act]
        final_anchors = final[:, -c.k_act :]
        public_valid = public_mask.to(final_public.dtype)
        public_mean = (final_public * public_valid[..., None]).sum(dim=1) / public_valid.sum(dim=1, keepdim=True)

        actor_anchor_input = torch.cat(
            (final_anchors[:, 0], final_anchors[:, 1], final_anchors[:, 2], final_anchors[:, 3:].mean(dim=1)), dim=-1
        )
        actor_summary = self.actor_seat_ffn(actor_anchor_input)
        mass = masked_range.sum(dim=-1)
        pooled = (final_rows * masked_range[..., None]).sum(dim=2) / mass[..., None].clamp_min(c.eps)
        seat_meta = torch.stack(
            (
                mass,
                status.to(mass.dtype) / 2.0,
                relative_seats.to(mass.dtype) / max(c.max_players - 1, 1),
            ), dim=-1,
        )
        seat_summary = self.seat_ffn(torch.cat((pooled, seat_meta), dim=-1))
        actor_selector = torch.nn.functional.one_hot(actor, players).to(torch.bool)
        seat_summary = torch.where(actor_selector[..., None], actor_summary[:, None, :], seat_summary)
        seat_summary = seat_summary.masked_fill(~seat_mask[..., None], 0.0)
        relative_index = relative_seats[..., None].expand(-1, -1, d)
        ordered_seats = torch.zeros_like(seat_summary).scatter(1, relative_index, seat_summary)
        ordered_mask = torch.zeros_like(seat_mask).scatter(1, relative_seats, seat_mask)
        padded_seats = torch.zeros((batch, c.max_players, d), dtype=seat_summary.dtype, device=seat_summary.device)
        padded_seats[:, :players] = ordered_seats
        global_feature = self.global_ffn(
            torch.cat((public_mean, actor_summary, padded_seats.reshape(batch, c.max_players * d)), dim=-1)
        )

        batch_ids = torch.arange(batch, device=actor.device)
        actor_rows = final_rows[batch_ids, actor]
        readout = torch.cat(
            (
                actor_rows,
                final_anchors.reshape(batch, 1, c.k_act * d).expand(-1, N_CLASSES, -1),
                class_emb,
                global_feature[:, None, :].expand(-1, N_CLASSES, -1),
            ), dim=-1,
        )
        z_type = self.type_features(readout)
        z_raise = self.raise_features(readout)
        type_logits = self.type_head(z_type)
        omega, mu, scale, iota = self.raise_head(z_raise, z_type)
        value = self.value_head(readout)

        vvec: Tensor | None = None
        u: Tensor | None = None
        if self.moment_heads is not None:
            moment_public = torch.cat((global_feature, public_mean), dim=-1)
            padded_mask = torch.zeros((batch, c.max_players), dtype=torch.bool, device=seat_mask.device)
            padded_mask[:, :players] = ordered_mask
            relative_vvec, u = self.moment_heads(moment_public, padded_seats, padded_mask)
            vvec = torch.zeros_like(relative_vvec)
            vvec[:, :players] = relative_vvec[:, :players].gather(1, relative_seats)
        return PolicyOutput(
            raw={"type_logits": type_logits, "omega": omega, "mu": mu, "s": scale, "iota": iota},
            v=value,
            vvec=vvec,
            u=u,
        )
