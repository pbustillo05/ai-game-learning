"""The class-level preflop poker policy and value network."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from pbg.core.tensor_ops import Linear, Embedding, broadcast_parameter

from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.model.shared import SharedPokerParameters
from pbg.games.poker.model.heads import (
    PublicBranchingHeads,
)
from pbg.games.poker.model.primitives import (
    accelerator_assert,
)


def _require_device(condition: Tensor, message: str) -> None:
    """Validate a scalar condition without synchronizing CUDA to the host."""

    accelerator_assert(condition, message)


@dataclass(frozen=True)
class PokerModelConfig:
    """Configuration of the poker transformer and its output heads."""

    d_model: int = 256
    n_heads: int = 8
    layers: int = 6
    m_raise: int = 4
    k_act: int = 7
    s_min: float = 0.02
    max_players: int = 9
    class_embedding_dim: int = 32
    ffn_hidden: int | None = None
    use_branching_heads: bool = False
    max_history_tokens: int = 256
    folder_beta_rank: int = 16
    folder_line_dim: int = 8
    eps: float = 1e-8

    def __post_init__(self) -> None:
        integer_fields = (
            "d_model", "n_heads", "layers", "m_raise",
            "k_act", "max_players", "class_embedding_dim",
            "max_history_tokens", "folder_beta_rank",
            "folder_line_dim",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.ffn_hidden is not None and (
            isinstance(self.ffn_hidden, bool) or not isinstance(self.ffn_hidden, int)
        ):
            raise TypeError("ffn_hidden must be an integer when supplied")
        if type(self.use_branching_heads) is not bool:
            raise TypeError("use_branching_heads must be bool")
        for name in ("s_min", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
        if not math.isfinite(self.s_min) or not math.isfinite(self.eps):
            raise ValueError("s_min and eps must be finite")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.layers <= 0 or self.m_raise <= 0:
            raise ValueError("layer and mixture counts must be positive")
        if self.k_act != 3 + self.m_raise:
            raise ValueError("k_act must equal 3 + m_raise")
        if self.s_min <= 0.0 or self.eps <= 0.0:
            raise ValueError("s_min and eps must be positive")
        if not 2 <= self.max_players <= 9:
            raise ValueError("max_players must be in [2,9]")
        if self.class_embedding_dim <= 0:
            raise ValueError("class_embedding_dim must be positive")
        if self.ffn_hidden is not None and self.ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive when supplied")
        if self.max_history_tokens <= 0:
            raise ValueError("max_history_tokens must be positive")
        if not 0 <= self.folder_beta_rank <= 52:
            raise ValueError("folder_beta_rank must be in [0,52]")
        if self.folder_line_dim <= 0:
            raise ValueError("folder_line_dim must be positive")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> PokerModelConfig:
        """Parse ``RootConfig.model`` while rejecting unknown model keys."""

        data = dict(values or {})
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                f"root.model.{unknown[0]}: unknown poker model configuration key"
            )
        try:
            return cls(**data)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            mentioned = [name for name in allowed if name in message]
            field = (
                min(mentioned, key=message.index) if mentioned else "configuration"
            )
            raise ValueError(f"root.model.{field}: {message}") from exc


@dataclass
class PackedPublicHistory:
    """Lossless padded public trajectory consumed by both poker networks.

    Every field has shape ``[B,T]``. ``action`` uses ``0=fold``,
    ``1=check/call``, ``2=raise``, and ``3=street separator``. ``amount`` is
    the chips called for check/call and the exact raise-to amount for raises.
    The remaining floating-point fields record the pre-action arithmetic.
    """

    actor: Tensor
    street: Tensor
    action: Tensor
    amount: Tensor
    pot_before: Tensor
    call_amount: Tensor
    current_bet: Tensor
    min_raise_to: Tensor
    max_raise_to: Tensor
    mask: Tensor


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
    committed_street: Tensor | None = None
    call_amount: Tensor | None = None
    current_bet: Tensor | None = None
    min_raise_to: Tensor | None = None
    max_raise_to: Tensor | None = None
    last_full_raise: Tensor | None = None
    legal_action_mask: Tensor | None = None
    raise_ticks: Tensor | None = None
    history: PackedPublicHistory | None = None


@dataclass
class PreflopMasks:
    """Optional row metadata and masks for a preflop batch."""

    seat_mask: Tensor | None = None  # bool [B,P]
    class_mask: Tensor | None = None  # bool [B,P,K]
    status: Tensor | None = None  # int [B,P]
    class_ids: Tensor | None = None  # int [K] or [B,K]
    class_embeddings: Tensor | None = None  # float [K,Dc] or [B,K,Dc]


@dataclass
class PolicyOutput:
    """Poker model output with class rows along dimension one."""

    raw: dict[str, Tensor]
    v: Tensor  # [B,K]
    t: Tensor | None = None  # [B]
    c: Tensor | None = None  # [B]


def _mask_value(masks: PreflopMasks | Mapping[str, Tensor] | Tensor | None, name: str) -> Tensor | None:
    if masks is None:
        return None
    if isinstance(masks, Tensor):
        return masks if name == "seat_mask" else None
    if isinstance(masks, Mapping):
        return masks.get(name)
    return getattr(masks, name)


class PreflopEncoder(nn.Module):
    """Pre-LN transformer over all seated-player preflop class rows."""

    def __init__(
        self,
        cfg: PokerModelConfig | Mapping[str, Any] | None = None,
        geometry: DeckGeometry | int = 13,
    ) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, PokerModelConfig) else PokerModelConfig.from_dict(cfg)
        self.geometry = (
            geometry if isinstance(geometry, DeckGeometry) else DeckGeometry(geometry)
        )
        self.n_cards = self.geometry.n_cards
        self.n_hands = self.geometry.n_hands
        self.n_classes = self.geometry.n_classes
        c = self.cfg
        d = c.d_model
        dc = c.class_embedding_dim

        self.class_embedding = Embedding(self.n_classes, dc)
        class_geometry_dim = self.n_cards + 2
        self.class_row_affine = Linear(dc + 4 + class_geometry_dim, d)
        branching_dim = (2 + c.max_players) * d
        self.branching_heads = (
            PublicBranchingHeads(branching_dim, d, c.ffn_hidden)
            if c.use_branching_heads else None
        )

        sizes = torch.tensor(
            [len(members) for members in self.geometry.class_memberships],
            dtype=torch.float32,
        )
        self.register_buffer("class_sizes", sizes, persistent=True)
        combo_cards = torch.tensor(self.geometry.combo_cards, dtype=torch.long)
        combo_ordinals = combo_cards - self.geometry.first_card
        combo_class = torch.tensor(self.geometry.class_of_combo, dtype=torch.long)
        orbit = torch.zeros((self.n_classes, self.n_cards), dtype=torch.float32)
        orbit.index_put_(
            (
                combo_class[:, None].expand(-1, 2).reshape(-1),
                combo_ordinals.reshape(-1),
            ),
            torch.ones(combo_cards.numel(), dtype=torch.float32),
            accumulate=True,
        )
        orbit = orbit / sizes[:, None]
        overlap = orbit @ orbit.T
        within = overlap.diagonal()
        cross = (overlap.sum(1) - within) / (self.n_classes - 1)
        self.register_buffer("class_card_geometry", orbit, persistent=True)
        self.register_buffer(
            "class_overlap_scalars", torch.stack((within, cross), dim=1), persistent=True
        )

    def _validate_public(self, pub: PokerPublicBatch, class_ranges: Tensor) -> tuple[int, int]:
        if class_ranges.ndim != 3 or class_ranges.shape[-1] != self.n_classes:
            raise ValueError(
                f"class_ranges must have shape [B,P,{self.n_classes}]"
            )
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
        optional_shapes = {
            "committed_street": (batch, players),
            "call_amount": (batch,),
            "current_bet": (batch,),
            "min_raise_to": (batch,),
            "max_raise_to": (batch,),
            "last_full_raise": (batch,),
        }
        for name, shape in optional_shapes.items():
            value = getattr(pub, name)
            if value is not None and value.shape != shape:
                raise ValueError(f"pub.{name} must have shape {shape}")
        if pub.legal_action_mask is not None and (
            pub.legal_action_mask.ndim != 2
            or pub.legal_action_mask.shape[0] != batch
        ):
            raise ValueError("pub.legal_action_mask must have shape [B,E]")
        if pub.raise_ticks is not None and (
            pub.raise_ticks.ndim != 2 or pub.raise_ticks.shape[0] != batch
        ):
            raise ValueError("pub.raise_ticks must have shape [B,N]")
        if pub.history is not None:
            history_shape = pub.history.mask.shape
            for name in pub.history.__dataclass_fields__:
                if getattr(pub.history, name).shape != history_shape:
                    raise ValueError("every public-history field must have shape [B,T]")
            if history_shape[0] != batch or pub.history.mask.dtype != torch.bool:
                raise TypeError("public history mask must be bool [B,T]")
            if history_shape[1] > self.cfg.max_history_tokens:
                raise ValueError("public history exceeds max_history_tokens")
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, players = class_ranges.shape[:2]
        device = class_ranges.device
        actor = pub.to_act if actor_seat is None else actor_seat
        if actor.shape != (batch,):
            raise ValueError("actor_seat must have shape [B]")
        actor = actor.to(device=device, dtype=torch.long)
        _require_device(
            ((actor >= 0) & (actor < players)).all(),
            "actor seat is outside the supplied seats",
        )

        seat_mask = _mask_value(masks, "seat_mask")
        if seat_mask is None:
            seat_mask = pub.seat_mask
        if seat_mask is None:
            seat_mask = torch.ones((batch, players), dtype=torch.bool, device=device)
        if seat_mask.shape != (batch, players) or seat_mask.dtype != torch.bool:
            raise TypeError("seat_mask must be bool [B,P]")
        _require_device(
            seat_mask.gather(1, actor[:, None]).all(),
            "the actor seat must be present",
        )

        class_mask = _mask_value(masks, "class_mask")
        if class_mask is None:
            class_mask = torch.ones_like(class_ranges, dtype=torch.bool)
        if class_mask.shape != class_ranges.shape or class_mask.dtype != torch.bool:
            raise TypeError("class_mask must be bool with the class-range shape")
        class_mask = class_mask & seat_mask[:, :, None]

        status = _mask_value(masks, "status")
        if status is None:
            status = pub.status
        if status is None:
            status = torch.zeros((batch, players), dtype=torch.long, device=device)
        status = status.to(device=device, dtype=torch.long)
        if status.shape != (batch, players):
            raise ValueError("status must contain live/folded/all-in ids [B,P]")
        _require_device(
            ((status >= 0) & (status <= 2)).all(),
            "status must contain live/folded/all-in ids [B,P]",
        )

        class_ids = _mask_value(masks, "class_ids")
        if class_ids is None:
            class_ids = torch.arange(self.n_classes, device=device)[None, :].expand(batch, -1)
        elif class_ids.ndim == 1:
            class_ids = class_ids[None, :].expand(batch, -1)
        class_ids = class_ids.to(device=device, dtype=torch.long)
        if class_ids.shape != (batch, self.n_classes):
            raise ValueError(
                f"class_ids must have shape [{self.n_classes}] or [B,{self.n_classes}]"
            )
        _require_device(
            ((class_ids >= 0) & (class_ids < self.n_classes)).all(),
            "class_ids are outside compact class geometry",
        )

        supplied_embedding = _mask_value(masks, "class_embeddings")
        if supplied_embedding is not None:
            embedding_shape = supplied_embedding.shape
            expected = (self.n_classes, self.cfg.class_embedding_dim)
            batched_expected = (batch,) + expected
            if embedding_shape not in (expected, batched_expected):
                raise ValueError("class_embeddings has the wrong shape")

        button = pub.button
        seat_ids = torch.arange(players, device=device)[None, :].expand(batch, -1)
        if button is not None:
            if button.shape != (batch,):
                raise ValueError("pub.button must have shape [B]")
            relative_seats = torch.remainder(seat_ids - button[:, None].long(), players)
        else:
            relative_seats = seat_ids
        return actor, seat_mask, class_mask, status, class_ids, relative_seats

    def _public_tokens(
        self,
        shared: SharedPokerParameters,
        pub: PokerPublicBatch,
        seat_mask: Tensor,
        relative_seats: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, players = seat_mask.shape
        dtype = pub.pot.dtype
        device = pub.pot.device
        scalar_groups = [
            pub.pot[:, None], pub.stacks, pub.committed,
            pub.street.to(dtype)[:, None], pub.legality_summary.to(dtype),
            pub.n_ticks.to(dtype)[:, None], pub.raise_grid_bounds.to(dtype),
        ]
        type_groups = [
            (1, 0), (players, 1), (players, 2), (1, 3),
            (pub.legality_summary.shape[1], 4), (1, 5), (2, 6),
        ]
        if pub.committed_street is not None:
            scalar_groups.append(pub.committed_street.to(dtype))
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
            scalar_groups.append(torch.cat(exact, dim=1))
            type_groups.append((len(exact), 8))
        if pub.legal_action_mask is not None:
            scalar_groups.append(pub.legal_action_mask.to(dtype))
            type_groups.append((pub.legal_action_mask.shape[1], 9))
        if pub.raise_ticks is not None:
            scalar_groups.append(pub.raise_ticks.to(dtype))
            type_groups.append((pub.raise_ticks.shape[1], 10))
        scalars = torch.cat(scalar_groups, dim=1)
        type_ids = torch.cat([
            torch.full((count,), kind, dtype=torch.long, device=device)
            for count, kind in type_groups
        ])
        token = shared.public_scalar_affine(scalars[..., None]) + shared.public_type_embedding(type_ids[None].expand(batch, -1))
        seat_position = shared.seat_embedding(relative_seats)
        seat_prefix = torch.zeros(
            (batch, 1, self.cfg.d_model), dtype=token.dtype, device=device
        )
        seat_suffix = torch.zeros(
            (batch, scalars.shape[1] - 1 - 2 * players, self.cfg.d_model),
            dtype=token.dtype,
            device=device,
        )
        token = token + torch.cat(
            (seat_prefix, seat_position, seat_position, seat_suffix), dim=1
        )
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=device),
                seat_mask,
                seat_mask,
                torch.ones(
                    (batch, scalars.shape[1] - 1 - 2 * players),
                    dtype=torch.bool,
                    device=device,
                ),
            ),
            dim=1,
        )
        if pub.history is not None and pub.history.mask.shape[1]:
            history = pub.history
            numeric = torch.stack(
                (
                    history.amount, history.pot_before, history.call_amount,
                    history.current_bet, history.min_raise_to, history.max_raise_to,
                ),
                dim=-1,
            ).to(dtype)
            history_token = shared.history_numeric_affine(numeric)
            history_token = history_token + shared.history_action_embedding(
                history.action.clamp(0, 3).long()
            )
            history_token = history_token + shared.history_street_embedding(
                history.street.clamp(0, 3).long()
            )
            history_token = history_token + shared.history_position_embedding(
                torch.arange(history.mask.shape[1], device=device)[None].expand(batch, -1)
            )
            safe_actor = history.actor.clamp(0, players - 1).long()
            history_token = history_token + shared.seat_embedding(
                relative_seats.gather(1, safe_actor)
            ) * (history.actor >= 0)[..., None]
            token = torch.cat((token, history_token), dim=1)
            valid = torch.cat((valid, history.mask), dim=1)
        public_mean = (token * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        state = broadcast_parameter(shared.state_token, batch, 1).to(token.dtype) + public_mean
        return torch.cat((token, state[:, None, :]), dim=1), torch.cat(
            (valid, torch.ones((batch, 1), dtype=torch.bool, device=device)), dim=1
        )

    def forward(
        self,
        shared: SharedPokerParameters,
        pub: PokerPublicBatch,
        class_ranges: Tensor,
        actor_seat: Tensor | None = None,
        masks: PreflopMasks | Mapping[str, Tensor] | Tensor | None = None,
        *,
        selected_private: Tensor | None = None,
        _prepared_metadata: tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
        ]
        | None = None,
    ) -> PolicyOutput:
        """Return policy parameters and values for every actor class row.

        ``class_ranges`` is float ``[B,P,K]``. ``actor_seat`` defaults to
        ``pub.to_act``. Masks may be :class:`PreflopMasks`, a mapping, or a
        bool seat mask directly.
        """

        if _prepared_metadata is None:
            batch, players = self._validate_public(pub, class_ranges)
            prepared = self._metadata(pub, class_ranges, actor_seat, masks)
        else:
            batch, players = class_ranges.shape[:2]
            prepared = _prepared_metadata
        actor, seat_mask, class_mask, status, class_ids, relative_seats = prepared
        c = self.cfg
        d = c.d_model
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
        geometry = torch.cat(
            (
                self.class_card_geometry[class_ids],
                self.class_overlap_scalars[class_ids],
            ),
            dim=-1,
        ).to(class_ranges.dtype)
        expanded_geometry = geometry[:, None].expand(-1, players, -1, -1)
        log_range = torch.log(class_ranges.clamp_min(c.eps))
        constant = torch.ones_like(class_ranges)
        row_feature = torch.cat(
            (
                class_ranges[..., None], log_range[..., None], size_feature[..., None],
                expanded_emb, expanded_geometry, constant[..., None],
            ),
            dim=-1,
        )
        row = self.class_row_affine(row_feature)
        row = row + shared.seat_embedding(relative_seats)[:, :, None, :]
        row = row + shared.status_embedding(status)[:, :, None, :]
        actor_flag = torch.arange(players, device=row.device)[None, :] == actor[:, None]
        row = row + shared.actor_embedding(actor_flag.long())[:, :, None, :]
        row_flat = row.reshape(batch, players * self.n_classes, d)

        public_token, public_mask = self._public_tokens(shared, pub, seat_mask, relative_seats)
        actor_relative = relative_seats.gather(1, actor[:, None]).squeeze(1)
        anchors = broadcast_parameter(shared.anchor_tokens, batch, 2).to(row.dtype) + shared.anchor_seat_embedding(actor_relative)[:, None, :]
        token = torch.cat((row_flat, public_token, anchors), dim=1)
        row_mask_flat = class_mask.reshape(batch, players * self.n_classes)
        key_mask = torch.cat(
            (row_mask_flat, public_mask, torch.ones((batch, c.k_act), dtype=torch.bool, device=row.device)), dim=1
        )

        masked_range = class_ranges * class_mask.to(class_ranges.dtype)
        normalized = masked_range / masked_range.sum(dim=-1, keepdim=True).clamp_min(c.eps)
        key_bias = torch.cat(
            (
                torch.log(normalized.clamp_min(c.eps))
                .reshape(batch, players * self.n_classes)
                .float(),
                torch.zeros(
                    (batch, token.shape[1] - players * self.n_classes),
                    dtype=torch.float32,
                    device=token.device,
                ),
            ),
            dim=1,
        )

        side_class = torch.cat(
            (expanded_emb, class_ranges[..., None], log_range[..., None], expanded_geometry),
            dim=-1,
        )
        x = token
        incidence_mass = normalized
        identical_scale = (1.0 / self.class_sizes[class_ids]).to(class_ranges.dtype)
        for layer_norm, attention, incidence, layer_ffn in zip(
            shared.layer_norms, shared.attentions, shared.incidence_residuals,
            shared.layer_ffns, strict=True
        ):
            row_state = x[:, : players * self.n_classes].reshape(
                batch, players, self.n_classes, d
            )
            j, overlap_card, overlap_equal, _ = incidence(
                row_state,
                incidence_mass,
                self.class_card_geometry[class_ids].to(class_ranges.dtype),
                identical_scale,
            )
            normalized_x = layer_norm(x)
            x_hat = x + attention.forward_biased(
                normalized_x, normalized_x, key_bias, key_mask
            )
            dynamic = torch.cat(
                (j, overlap_card * overlap_card, overlap_card * overlap_equal), dim=-1
            )
            def pad_rows(value: Tensor) -> Tensor:
                rows = value.reshape(batch, players * self.n_classes, -1)
                return torch.nn.functional.pad(rows, (0, 0, 0, token.shape[1] - rows.shape[1]))
            x = x_hat + layer_ffn(
                torch.cat((layer_norm(x_hat), pad_rows(dynamic)), dim=-1),
                pad_rows(side_class), "pre",
            )

        final = shared.final_norm(x)
        final_rows = final[:, : players * self.n_classes].reshape(
            batch, players, self.n_classes, d
        )
        final_public = final[:, players * self.n_classes : -c.k_act]
        final_anchors = final[:, -c.k_act :]
        public_valid = public_mask.to(final_public.dtype)
        public_mean = (final_public * public_valid[..., None]).sum(dim=1) / public_valid.sum(dim=1, keepdim=True)

        actor_anchor_input = torch.cat(
            (final_anchors[:, 0], final_anchors[:, 1], final_anchors[:, 2], final_anchors[:, 3:].mean(dim=1)), dim=-1
        )
        actor_summary = shared.actor_seat_features(actor_anchor_input)
        mass = masked_range.sum(dim=-1)
        pooled = (final_rows * masked_range[..., None]).sum(dim=2) / mass[..., None].clamp_min(c.eps)
        seat_meta = torch.stack(
            (
                mass,
                status.to(mass.dtype) / 2.0,
                relative_seats.to(mass.dtype) / max(c.max_players - 1, 1),
            ), dim=-1,
        )
        seat_summary = shared.seat_features(torch.cat((pooled, seat_meta), dim=-1))
        actor_selector = (
            torch.arange(players, device=actor.device)[None, :]
            == actor[:, None]
        )
        seat_summary = torch.where(actor_selector[..., None], actor_summary[:, None, :], seat_summary)
        seat_summary = seat_summary.masked_fill(~seat_mask[..., None], 0.0)
        relative_index = relative_seats[..., None].expand(-1, -1, d)
        ordered_seats = torch.zeros_like(seat_summary).scatter(1, relative_index, seat_summary)
        padded_seats = torch.cat(
            (
                ordered_seats,
                torch.zeros(
                    (batch, c.max_players - players, d),
                    dtype=seat_summary.dtype,
                    device=seat_summary.device,
                ),
            ),
            dim=1,
        )
        global_feature = shared.global_features(
            torch.cat((public_mean, actor_summary, padded_seats.reshape(batch, c.max_players * d)), dim=-1)
        )

        batch_ids = torch.arange(batch, device=actor.device)
        actor_rows = final_rows[batch_ids, actor]
        if selected_private is not None:
            selected = selected_private[:, None, None]
            actor_rows = actor_rows.gather(1, selected.expand(-1, 1, d))
            class_emb = class_emb.gather(1, selected.expand(-1, 1, c.class_embedding_dim))
        raw, value = shared.readout(actor_rows, class_emb, final_anchors, global_feature)

        t: Tensor | None = None
        cost: Tensor | None = None
        if self.branching_heads is not None:
            branching_public = torch.cat(
                (global_feature, public_mean, padded_seats.flatten(1)), dim=-1
            )
            t, cost = self.branching_heads(branching_public)
        return PolicyOutput(
            raw=raw,
            v=value,
            t=t,
            c=cost,
        )
