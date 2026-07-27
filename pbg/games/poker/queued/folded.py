"""Authority-conforming folded-range statistics for queued poker."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from pbg.games.poker.queued.hooks import FOLDED, NLHEGeometry, PAD_CARD


class FoldedStatisticsError(ValueError):
    pass


_REMOVAL_QUERY_CHUNK = 64


def folded_statistics_workspace_row_bytes(geometry: NLHEGeometry) -> int:
    """Conservative per-row bound for the finite-difference CUDA kernel."""

    if geometry.players == 2:
        return 0
    cards = geometry.deck_cards
    players = geometry.players
    hands = geometry.private_hands
    queries = 1 + cards + cards * (cards - 1) // 2
    chunk = min(queries, _REMOVAL_QUERY_CHUNK)
    fp32_values = (
        players * cards * cards
        + chunk
        * (
            2 * players * cards
            + players * players
            + 8 * players
            + 8
        )
        + 2 * queries
        # Triple-cluster evaluation holds the survives/edge/matrix-term
        # hand-domain operands concurrently for one query chunk.
        + 10 * chunk * hands
        + hands
        + 3 * cards * cards
    )
    return 4 * fp32_values


def _removal_queries(cards: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    pair = torch.triu_indices(cards, cards, offset=1, device=device)
    rows = 1 + cards + pair.shape[1]
    removal = torch.zeros((rows, cards), dtype=torch.bool, device=device)
    removal[1 : 1 + cards] = torch.eye(
        cards, dtype=torch.bool, device=device
    )
    ordinal = torch.arange(pair.shape[1], device=device) + 1 + cards
    removal[ordinal, pair[0]] = True
    removal[ordinal, pair[1]] = True
    return removal, pair, ordinal


def _range_matrix(
    probability: Tensor, combo_local: Tensor, cards: int
) -> Tensor:
    batch, folders, hands = probability.shape
    matrix = probability.new_zeros((batch, folders, cards * cards))
    first = combo_local[:, 0] * cards + combo_local[:, 1]
    second = combo_local[:, 1] * cards + combo_local[:, 0]
    for index in (first, second):
        matrix.scatter_add_(
            2,
            index[None, None].expand(batch, folders, hands),
            probability,
        )
    return matrix.reshape(batch, folders, cards, cards)


def _phi_queries(
    probability: Tensor,
    range_matrix: Tensor,
    folder_mask: Tensor,
    removal: Tensor,
    combo_local: Tensor,
    base_cards: Tensor,
    tight_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate the §3.4 estimator on every zero/one/two-card removal."""

    batch, folders, _ = probability.shape
    queries = removal.shape[0]
    remove = removal.float()
    allowed = ~removal
    removed_count = remove.sum(-1)
    available = base_cards[:, None] - removed_count[None]
    marginal = range_matrix.sum(-1)
    linear = torch.einsum("bfc,qc->bqf", marginal, remove)
    quadratic = 0.5 * torch.einsum(
        "qc,bfcd,qd->bqf", remove, range_matrix, remove
    )
    surviving_mass = (1.0 - linear + quadratic).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    removed_column = torch.einsum(
        "bfcd,qd->bqfc", range_matrix, remove
    )
    conditional_marginal = (
        (marginal[:, None] - removed_column)
        * allowed[None, :, None]
        / surviving_mass[..., None]
    )
    flat = 2.0 / available.clamp_min(1.0)
    delta = conditional_marginal - (
        flat[:, :, None, None] * allowed[None, :, None]
    )

    phi = (
        torch.log(surviving_mass)
        * folder_mask[:, None].to(torch.float32)
    ).sum(-1)
    folder_count = folder_mask.sum(-1)
    denominator = available * (available - 1.0) * 0.5
    for packed in range(1, folders):
        numerator_cards = available - 2.0 * packed
        numerator = numerator_cards * (numerator_cards - 1.0) * 0.5
        active = folder_count[:, None] > packed
        phi = phi + torch.where(
            active,
            torch.log(
                (numerator / denominator).clamp_min(
                    torch.finfo(torch.float32).tiny
                )
            ),
            torch.zeros_like(phi),
        )

    survives = (
        allowed[:, combo_local[:, 0]]
        & allowed[:, combo_local[:, 1]]
    ).float()
    marginal_dot = torch.einsum(
        "bqfc,bqgc->bqfg",
        conditional_marginal,
        conditional_marginal,
    )
    identical = 2.0 * torch.einsum(
        "bfh,bgh,qh->bqfg", probability, probability, survives
    )
    identical = identical / (
        surviving_mass[:, :, :, None] * surviving_mass[:, :, None, :]
    )
    pair_disjoint = (
        1.0 - marginal_dot + 0.5 * identical
    ).clamp_min(torch.finfo(torch.float32).tiny)
    uniform_pair = (
        1.0
        - 4.0 / available
        + 1.0 / denominator
    ).clamp_min(torch.finfo(torch.float32).tiny)
    for first in range(folders):
        for second in range(first + 1, folders):
            active = folder_mask[:, first] & folder_mask[:, second]
            phi = phi + torch.where(
                active[:, None],
                torch.log(
                    pair_disjoint[:, :, first, second] / uniform_pair
                ),
                torch.zeros_like(phi),
            )

    triple = torch.zeros(
        (batch, queries), dtype=torch.float32, device=probability.device
    )
    for first in range(folders):
        for second in range(first + 1, folders):
            wings = tight_mask[:, first] & tight_mask[:, second]
            left = delta[:, :, first]
            right = delta[:, :, second]
            for hub in range(folders):
                if hub in (first, second):
                    continue
                active = wings & folder_mask[:, hub]
                edge = (
                    left[..., combo_local[:, 0]]
                    * right[..., combo_local[:, 1]]
                    + left[..., combo_local[:, 1]]
                    * right[..., combo_local[:, 0]]
                )
                matrix_term = (
                    edge
                    * survives[None]
                    * probability[:, None, hub]
                ).sum(-1) / surviving_mass[:, :, hub]
                hub_marginal = conditional_marginal[:, :, hub]
                diagonal = (hub_marginal * left * right).sum(-1)
                outer = (hub_marginal * left).sum(-1) * (
                    hub_marginal * right
                ).sum(-1)
                contribution = matrix_term + diagonal - outer
                triple = triple + torch.where(
                    active[:, None], contribution, torch.zeros_like(triple)
                )
    return phi + triple, triple


def _folded_statistics_fp32(
    geometry: NLHEGeometry,
    *,
    board_cards: Tensor,
    range_weights: Tensor,
    status: Tensor,
    combo_cards: Tensor,
    beta_rank: int,
    line_dim: int,
    line_metadata: Tensor | None = None,
    tight_mask: Tensor | None = None,
) -> Mapping[str, Tensor]:
    r"""Compute $\Lambda_B$ and signed $\beta_B$ factors from exact combos."""

    batch, players, hands = range_weights.shape
    cards = geometry.deck_cards
    if (
        board_cards.device.type != "cuda"
        or board_cards.shape != (batch, 5)
        or range_weights.dtype != torch.float32
        or status.shape != (batch, players)
        or hands != geometry.private_hands
        or combo_cards.shape != (hands, 2)
        or not 0 <= beta_rank <= cards
        or line_dim < 0
        or (line_dim > 0 and line_metadata is None)
        or (
            line_metadata is not None
            and (
                line_metadata.shape != (batch, players, line_dim)
                or line_metadata.dtype != torch.float32
                or line_metadata.device != range_weights.device
            )
        )
    ):
        raise FoldedStatisticsError("folded statistic inputs violate poker geometry")
    folder_mask = status.eq(FOLDED)
    if tight_mask is None:
        tight_mask = folder_mask
    if (
        tight_mask.shape != folder_mask.shape
        or tight_mask.dtype != torch.bool
        or tight_mask.device != range_weights.device
    ):
        raise FoldedStatisticsError("tight-folder mask violates folded seats")
    torch._assert_async(
        (~tight_mask | folder_mask).all(),
        "tight-folder mask includes a live seat",
    )
    finite = torch.isfinite(range_weights)
    raw = range_weights.masked_fill(~finite, 0.0).clamp_min(0.0)
    raw = raw * folder_mask[..., None]
    board_valid = board_cards.ne(PAD_CARD)
    board_conflict = (
        board_valid[..., None, None]
        & board_cards.long()[..., None, None].eq(combo_cards[None])
    ).any(1).any(-1)
    raw = raw.masked_fill(board_conflict[:, None], 0.0)
    mass = raw.sum(-1)
    probability = raw / mass.clamp_min(
        torch.finfo(torch.float32).tiny
    )[..., None]
    combo_local = (combo_cards - geometry.first_card).to(torch.int64)
    matrix = _range_matrix(probability, combo_local, cards)
    removal, pair, pair_ordinal = _removal_queries(cards, raw.device)
    base_cards = cards - board_valid.sum(-1, dtype=torch.float32)
    phi_parts = []
    triple_parts = []
    for begin in range(0, removal.shape[0], _REMOVAL_QUERY_CHUNK):
        phi_part, triple_part = _phi_queries(
            probability,
            matrix,
            folder_mask,
            removal[begin : begin + _REMOVAL_QUERY_CHUNK],
            combo_local,
            base_cards,
            tight_mask,
        )
        phi_parts.append(phi_part)
        triple_parts.append(triple_part)
    phi = torch.cat(phi_parts, 1)
    triple = torch.cat(triple_parts, 1)
    phi_zero = phi[:, :1]
    alpha = phi[:, 1 : 1 + cards] - phi_zero
    beta = torch.zeros(
        (batch, cards, cards), dtype=torch.float32, device=raw.device
    )
    pair_value = (
        phi.index_select(1, pair_ordinal)
        - phi[:, 1 + pair[0]]
        - phi[:, 1 + pair[1]]
        + phi_zero
    )
    beta[:, pair[0], pair[1]] = pair_value
    beta[:, pair[1], pair[0]] = pair_value
    board_local = (board_cards.long() - geometry.first_card).clamp(
        0, cards - 1
    )
    board_card_mask = torch.zeros(
        (batch, cards), dtype=torch.bool, device=raw.device
    )
    board_card_mask.scatter_(1, board_local, board_valid)
    beta = beta.masked_fill(
        board_card_mask[:, :, None] | board_card_mask[:, None, :], 0.0
    )
    alpha = alpha.masked_fill(board_card_mask, 0.0)
    hand_log_ratio = (
        alpha[:, combo_local[:, 0]]
        + alpha[:, combo_local[:, 1]]
        + beta[:, combo_local[:, 0], combo_local[:, 1]]
    )

    if beta_rank:
        eigenvalue, eigenvector = torch.linalg.eigh(beta)
        order = torch.argsort(
            eigenvalue.abs(), dim=-1, descending=True, stable=True
        )[:, :beta_rank]
        eigenvalue = eigenvalue.gather(1, order)
        eigenvector = eigenvector.gather(
            2, order[:, None].expand(-1, cards, -1)
        )
        pivot = eigenvector.abs().argmax(1)
        orientation = eigenvector.gather(1, pivot[:, None]).squeeze(1)
        orientation = torch.where(
            orientation < 0,
            -torch.ones_like(orientation),
            torch.ones_like(orientation),
        )
        eigenvector = eigenvector * orientation[:, None]
    else:
        eigenvalue = raw.new_zeros((batch, 0))
        eigenvector = raw.new_zeros((batch, cards, 0))

    card_incidence = raw.new_zeros((hands, cards))
    card_incidence.scatter_(1, combo_local, 1.0)
    card_mass = torch.einsum("bph,hc->bpc", raw, card_incidence)
    line_meta = (
        raw.new_zeros((batch, players, line_dim))
        if line_metadata is None
        else line_metadata * folder_mask[..., None]
    )
    return MappingProxyType(
        {
            "hand_log_ratio": hand_log_ratio.contiguous(),
            "beta_eigenvalue": eigenvalue.contiguous(),
            "beta_eigenvector": eigenvector.contiguous(),
            "folder_log_mass": torch.where(
                folder_mask,
                torch.log(mass.clamp_min(torch.finfo(torch.float32).tiny)),
                torch.zeros_like(mass),
            ).contiguous(),
            "folder_card_mass": card_mass.contiguous(),
            "folder_line_meta": line_meta.contiguous(),
            "folder_mask": folder_mask.contiguous(),
            "folder_output_valid": folder_mask.logical_and(mass > 0).contiguous(),
            "folder_seat": torch.arange(
                players, dtype=torch.int64, device=raw.device
            )[None].expand(batch, -1).contiguous(),
            "tight_cluster_scalar": triple[:, 0].contiguous(),
        }
    )


def folded_statistics(
    geometry: NLHEGeometry,
    *,
    board_cards: Tensor,
    range_weights: Tensor,
    status: Tensor,
    combo_cards: Tensor,
    beta_rank: int,
    line_dim: int,
    line_metadata: Tensor | None = None,
    tight_mask: Tensor | None = None,
) -> Mapping[str, Tensor]:
    """Run the authoritative finite differences outside mixed precision."""

    with torch.autocast(device_type="cuda", enabled=False):
        return _folded_statistics_fp32(
            geometry,
            board_cards=board_cards,
            range_weights=range_weights,
            status=status,
            combo_cards=combo_cards,
            beta_rank=beta_rank,
            line_dim=line_dim,
            line_metadata=line_metadata,
            tight_mask=tight_mask,
        )


def pack_folded_statistics(
    geometry: NLHEGeometry, summary: Mapping[str, Tensor]
) -> Tensor:
    """Pack model-consumed statistics into the retained family row."""

    batch = summary["hand_log_ratio"].shape[0]
    value = torch.cat(
        (
            summary["hand_log_ratio"],
            summary["beta_eigenvalue"],
            summary["beta_eigenvector"].flatten(1),
            summary["folder_log_mass"],
            summary["folder_card_mass"].flatten(1),
            summary["tight_cluster_scalar"][:, None],
            torch.ones(
                (batch, 1),
                dtype=summary["hand_log_ratio"].dtype,
                device=summary["hand_log_ratio"].device,
            ),
        ),
        1,
    ).contiguous()
    if value.shape != (batch, geometry.folded_summary_width):
        raise FoldedStatisticsError("packed folded summary has the wrong width")
    return value


def unpack_folded_statistics(
    geometry: NLHEGeometry,
    packed: Tensor,
    *,
    beta_rank: int,
    line_metadata: Tensor,
    status: Tensor,
) -> Mapping[str, Tensor]:
    """Recover the model view without folded-range access or recomputation."""

    batch = packed.shape[0]
    cards = geometry.deck_cards
    hands = geometry.private_hands
    players = geometry.players
    if geometry.folded_summary_width == 0:
        folder_mask = status.eq(FOLDED)
        return MappingProxyType(
            {
                "hand_log_ratio": packed.new_zeros((batch, hands)),
                "beta_eigenvalue": packed.new_zeros((batch, beta_rank)),
                "beta_eigenvector": packed.new_zeros(
                    (batch, cards, beta_rank)
                ),
                "folder_log_mass": packed.new_zeros((batch, players)),
                "folder_card_mass": packed.new_zeros(
                    (batch, players, cards)
                ),
                "folder_line_meta": line_metadata * folder_mask[..., None],
                "folder_mask": folder_mask,
                "folder_output_valid": folder_mask,
                "folder_seat": torch.arange(
                    players, dtype=torch.int64, device=packed.device
                )[None].expand(batch, -1),
                "tight_cluster_scalar": packed.new_zeros((batch,)),
            }
        )
    if packed.shape != (batch, geometry.folded_summary_width):
        raise FoldedStatisticsError("retained folded summary has the wrong width")
    offset = 0

    def take(width: int) -> Tensor:
        nonlocal offset
        result = packed[:, offset : offset + width]
        offset += width
        return result

    hand_log_ratio = take(hands)
    eigenvalue = take(cards)[:, :beta_rank]
    eigenvector = take(cards * cards).reshape(batch, cards, cards)[
        :, :, :beta_rank
    ]
    folder_log_mass = take(players)
    folder_card_mass = take(players * cards).reshape(batch, players, cards)
    tight_cluster_scalar = take(1)[:, 0]
    summary_valid = take(1)[:, 0].gt(0.5)
    folder_mask = status.eq(FOLDED)
    torch._assert_async(
        (~folder_mask.any(-1) | summary_valid).all(),
        "postflop folded seats require a sealed folded summary",
    )
    return MappingProxyType(
        {
            "hand_log_ratio": hand_log_ratio,
            "beta_eigenvalue": eigenvalue,
            "beta_eigenvector": eigenvector,
            "folder_log_mass": folder_log_mass,
            "folder_card_mass": folder_card_mass,
            "folder_line_meta": line_metadata * folder_mask[..., None],
            "folder_mask": folder_mask,
            "folder_output_valid": folder_mask,
            "folder_seat": torch.arange(
                players, dtype=torch.int64, device=packed.device
            )[None].expand(batch, -1),
            "tight_cluster_scalar": tight_cluster_scalar,
        }
    )


__all__ = [
    "FoldedStatisticsError",
    "folded_statistics",
    "folded_statistics_workspace_row_bytes",
    "pack_folded_statistics",
    "unpack_folded_statistics",
]
