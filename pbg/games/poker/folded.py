"""Evaluator-owned partition-summary data transfer boundary.

This module only batches fold snapshots and forwards them to the stable
``poker_gpu_evaluator.pbg`` facade.  Private-card collision and partition
calculations remain entirely evaluator-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

from pbg.games.poker.cards import DeckGeometry, validate_evaluator_combo_cards


PAD_CARD = 255
EXPECTED_API_VERSION = "pbg-v3"
EXPECTED_FOLDER_API_VERSION = "pbg-folder-v1"


@dataclass
class FoldSnapshotBatch:
    """Batched frozen fold-time inputs.

    ``reach_weights`` is float32 ``[B,F,H]``, ``board_at_fold`` is uint8
    ``[B,F,5]`` with 255 padding, ``line_meta`` is float32 ``[B,F,D]``, and
    ``folder_mask`` is bool ``[B,F]``.  Masked fold rows are ignored by the
    evaluator but retain their tensor slots for stable batching.
    """

    reach_weights: Tensor
    board_at_fold: Tensor
    line_meta: Tensor
    folder_mask: Tensor
    seat_ids: Tensor | None = None


@dataclass
class PartitionSummaryBatch:
    """Opaque evaluator summary tensors and host metadata.

    The compact model-ready fields include combined unary/pair corrections,
    signed beta factors, per-folder mass/card pressure/delta, tight contraction,
    fold-line/seat metadata, and validity. Bulky evaluator-owned per-folder
    pair/covariance tensors are validated at the boundary but not retained.
    """

    fields: dict[str, Tensor]
    metadata: dict[str, object]


class PokerGPUPartitionAdapter:
    """Thin, lazy adapter for evaluator-owned model-ready folder summaries."""

    def __init__(self, beta_rank: int = 16, rank_count: int = 13) -> None:
        if isinstance(rank_count, bool) or not 4 <= rank_count <= 13:
            raise ValueError("rank_count must be an integer in [4,13]")
        self.rank_count = int(rank_count)
        self.geometry = DeckGeometry(self.rank_count)
        if (
            isinstance(beta_rank, bool)
            or not isinstance(beta_rank, int)
            or not 0 <= beta_rank <= self.geometry.n_cards
        ):
            raise ValueError(
                f"beta_rank must be an integer in [0,{self.geometry.n_cards}]"
            )
        self.beta_rank = beta_rank
        self._partition_folders: Callable[..., Any] | None = None
        self._result_type: type[Any] | None = None

    def _load_facade(self) -> tuple[Callable[..., Any], type[Any]]:
        if self._partition_folders is not None and self._result_type is not None:
            return self._partition_folders, self._result_type
        try:
            from poker_gpu_evaluator.pbg import (
                API_VERSION,
                FOLDER_API_VERSION,
                FolderSummaryResult,
                combo_cards,
                partition_folders,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "poker_gpu_evaluator with the public pbg-v3 facade is required; "
                "install the sibling package (an editable install is sufficient)"
            ) from exc
        if API_VERSION != EXPECTED_API_VERSION:
            raise RuntimeError(
                "incompatible poker_gpu_evaluator facade: expected API_VERSION "
                f"{EXPECTED_API_VERSION!r}, got {API_VERSION!r}; update the sibling package"
            )
        if FOLDER_API_VERSION != EXPECTED_FOLDER_API_VERSION:
            raise RuntimeError(
                "incompatible evaluator folder facade: expected FOLDER_API_VERSION "
                f"{EXPECTED_FOLDER_API_VERSION!r}, got {FOLDER_API_VERSION!r}"
            )
        validate_evaluator_combo_cards(
            combo_cards(self.rank_count), self.rank_count
        )
        self._partition_folders = partition_folders
        self._result_type = FolderSummaryResult
        return partition_folders, FolderSummaryResult

    def _normalize_board(self, board_cards: Tensor) -> Tensor:
        if not isinstance(board_cards, Tensor):
            raise TypeError("board_cards must be a torch.Tensor")
        if board_cards.ndim != 2 or board_cards.shape[1] != 5:
            raise ValueError("board_cards must have shape [B,5]")
        if board_cards.dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("board_cards must have an integer dtype")
        padding = (board_cards == -1) | (board_cards == PAD_CARD)  # [B,5]
        valid = padding | (
            (board_cards >= self.geometry.first_card) & (board_cards < 52)
        )  # [B,5]
        if not bool(valid.all()):
            raise ValueError("board_cards must contain card ids in [0,52), -1, or 255")
        after_padding = padding.to(torch.int32).cumsum(dim=1) > 0  # [B,5]
        if bool((~padding & after_padding).any()):
            raise ValueError("real board cards must precede padding")
        return torch.where(
            padding,
            torch.full_like(board_cards, PAD_CARD),
            board_cards,
        ).to(dtype=torch.uint8).contiguous()  # [B,5]

    def _validate_inputs(
        self,
        board_cards: Tensor,
        live_range_weights: Tensor,
        folded: FoldSnapshotBatch,
    ) -> None:
        if not isinstance(live_range_weights, Tensor):
            raise TypeError("live_range_weights must be a torch.Tensor")
        if (
            live_range_weights.ndim != 3
            or live_range_weights.shape[2] != self.geometry.n_hands
        ):
            raise ValueError(
                f"live_range_weights must have shape [B,L,{self.geometry.n_hands}]"
            )
        if live_range_weights.dtype != torch.float32:
            raise TypeError("live_range_weights must have dtype torch.float32")
        if not all(
            hasattr(folded, name)
            for name in ("reach_weights", "board_at_fold", "line_meta", "folder_mask")
        ):
            raise TypeError("folded must provide the FoldSnapshotBatch fields")

        reach = folded.reach_weights
        fold_board = folded.board_at_fold
        line_meta = folded.line_meta
        mask = folded.folder_mask
        seat_ids = folded.seat_ids
        if reach.ndim != 3 or reach.shape[2] != self.geometry.n_hands:
            raise ValueError(
                f"folded.reach_weights must have shape [B,F,{self.geometry.n_hands}]"
            )
        batch_size, n_folders = reach.shape[:2]
        if reach.dtype != torch.float32:
            raise TypeError("folded.reach_weights must have dtype torch.float32")
        if fold_board.shape != (batch_size, n_folders, 5):
            raise ValueError("folded.board_at_fold must have shape [B,F,5]")
        if fold_board.dtype != torch.uint8:
            raise TypeError("folded.board_at_fold must have dtype torch.uint8")
        fold_padding = fold_board == PAD_CARD  # [B,F,5]
        fold_valid = fold_padding | (
            (fold_board >= self.geometry.first_card) & (fold_board < 52)
        )  # [B,F,5]
        if not bool(fold_valid.all()):
            raise ValueError("folded.board_at_fold must contain card ids in [0,52) or 255")
        fold_after_padding = fold_padding.to(torch.int32).cumsum(dim=2) > 0  # [B,F,5]
        if bool((~fold_padding & fold_after_padding).any()):
            raise ValueError("real fold-time board cards must precede padding")
        if line_meta.ndim != 3 or line_meta.shape[:2] != (batch_size, n_folders):
            raise ValueError("folded.line_meta must have shape [B,F,D]")
        if line_meta.dtype != torch.float32:
            raise TypeError("folded.line_meta must have dtype torch.float32")
        if mask.shape != (batch_size, n_folders):
            raise ValueError("folded.folder_mask must have shape [B,F]")
        if mask.dtype != torch.bool:
            raise TypeError("folded.folder_mask must have dtype torch.bool")
        if seat_ids is not None:
            if seat_ids.shape != (batch_size, n_folders) or seat_ids.dtype not in (
                torch.int32, torch.int64
            ):
                raise TypeError("folded.seat_ids must be integer [B,F]")

        if board_cards.shape[0] != batch_size or live_range_weights.shape[0] != batch_size:
            raise ValueError("all inputs must have the same B dimension")
        tensors = (live_range_weights, reach, fold_board, line_meta, mask)
        if any(tensor.device != board_cards.device for tensor in tensors):
            raise ValueError("all input tensors must be on the same device")
        if seat_ids is not None and seat_ids.device != board_cards.device:
            raise ValueError("folded.seat_ids is on the wrong device")

    def _validate_result(
        self,
        result: Any,
        result_type: type[Any],
        batch_size: int,
        n_folders: int,
        beta_rank: int,
        device: torch.device,
    ) -> None:
        if not isinstance(result, result_type):
            raise TypeError("partition_folders returned an unexpected result type")
        n_cards = self.geometry.n_cards
        expected = (
            ("base_log_z", (batch_size,), torch.float32),
            ("single_card_log_ratio", (batch_size, n_cards), torch.float32),
            ("pair_card_log_ratio", (batch_size, n_cards, n_cards), torch.float32),
            ("beta_eigenvalue", (batch_size, beta_rank), torch.float32),
            ("beta_eigenvector", (batch_size, n_cards, beta_rank), torch.float32),
            ("folder_log_mass", (batch_size, n_folders), torch.float32),
            ("folder_single_card_log_ratio", (batch_size, n_folders, n_cards), torch.float32),
            ("folder_pair_card_log_ratio", (batch_size, n_folders, n_cards, n_cards), torch.float32),
            ("folder_card_marginal", (batch_size, n_folders, n_cards), torch.float32),
            ("folder_delta", (batch_size, n_folders, n_cards), torch.float32),
            ("folder_sigma", (batch_size, n_folders, n_cards, n_cards), torch.float32),
            ("sigma_total", (batch_size, n_cards, n_cards), torch.float32),
            ("tight_cluster_scalar", (batch_size,), torch.float32),
            ("folder_output_valid", (batch_size, n_folders), torch.bool),
            ("output_valid", (batch_size,), torch.bool),
        )
        for name, shape, dtype in expected:
            value = getattr(result, name, None)
            if not isinstance(value, Tensor):
                raise TypeError(f"partition result {name} must be a torch.Tensor")
            if value.shape != shape:
                raise ValueError(f"partition result {name} must have shape {list(shape)}")
            if value.dtype != dtype:
                raise TypeError(f"partition result {name} has the wrong dtype")
            if value.device != device:
                raise ValueError(f"partition result {name} is on the wrong device")
            if not value.is_contiguous():
                raise ValueError(f"partition result {name} must be contiguous")
        if not isinstance(getattr(result, "metadata", None), dict):
            raise TypeError("partition result metadata must be a dict")
        if any(isinstance(value, Tensor) for value in result.metadata.values()):
            raise TypeError("partition result metadata may not contain tensors")

    def summarize(
        self,
        *,
        board_cards: Tensor,
        live_range_weights: Tensor,
        folded: FoldSnapshotBatch,
    ) -> PartitionSummaryBatch:
        """Return evaluator summaries for a batch of public spots.

        ``board_cards`` is integer ``[B,5]`` with local -1 or boundary 255
        padding. ``live_range_weights`` is float32 ``[B,L,H]``. Fold inputs
        have the shapes documented by :class:`FoldSnapshotBatch`.
        Returned fields have shapes documented by :class:`PartitionSummaryBatch`.
        """

        boundary_board = self._normalize_board(board_cards)  # [B,5]
        self._validate_inputs(boundary_board, live_range_weights, folded)
        batch_size = boundary_board.shape[0]
        # The correction is the folded subsystem integrated out. Live ranges are accepted
        # only because they remain part of the versioned call-site contract;
        # including them in the neutral partition law would double-count live
        # collision geometry and produce a non-neutral correction with K=0.
        if folded.reach_weights.shape[1] == 0 or not bool(folded.folder_mask.any()):
            zeros = torch.zeros(
                (batch_size,), dtype=torch.float32, device=boundary_board.device
            )
            n_folders = folded.reach_weights.shape[1]
            n_cards = self.geometry.n_cards
            fields = {
                "base_log_z": zeros,
                "single_card_log_ratio": torch.zeros((batch_size, n_cards), dtype=torch.float32, device=boundary_board.device),
                "pair_card_log_ratio": torch.zeros((batch_size, n_cards, n_cards), dtype=torch.float32, device=boundary_board.device),
                "beta_eigenvalue": torch.zeros((batch_size, self.beta_rank), dtype=torch.float32, device=boundary_board.device),
                "beta_eigenvector": torch.zeros((batch_size, n_cards, self.beta_rank), dtype=torch.float32, device=boundary_board.device),
                "folder_log_mass": torch.zeros((batch_size, n_folders), dtype=torch.float32, device=boundary_board.device),
                "folder_single_card_log_ratio": torch.zeros((batch_size, n_folders, n_cards), dtype=torch.float32, device=boundary_board.device),
                "folder_card_marginal": torch.zeros((batch_size, n_folders, n_cards), dtype=torch.float32, device=boundary_board.device),
                "folder_delta": torch.zeros((batch_size, n_folders, n_cards), dtype=torch.float32, device=boundary_board.device),
                "tight_cluster_scalar": zeros,
                "folder_output_valid": folded.folder_mask.clone(),
                "output_valid": torch.ones((batch_size,), dtype=torch.bool, device=boundary_board.device),
                "folder_mask": folded.folder_mask,
                "folder_line_meta": folded.line_meta,
                "folder_seat": folded.seat_ids if folded.seat_ids is not None else torch.arange(
                    n_folders, dtype=torch.int64, device=boundary_board.device
                )[None].expand(batch_size, -1),
            }
            return PartitionSummaryBatch(
                fields=fields,
                metadata={"api_version": EXPECTED_API_VERSION, "neutral_no_folders": True},
            )

        partition_folders, result_type = self._load_facade()
        range_weights = folded.reach_weights.contiguous()  # [B,F,H]
        row_mask = folded.folder_mask.contiguous()  # [B,F]
        result = partition_folders(
            rank_count=self.rank_count,
            board_cards=boundary_board,
            range_weights=range_weights,
            row_mask=row_mask,
            beta_rank=self.beta_rank,
        )
        n_folders = folded.reach_weights.shape[1]
        self._validate_result(
            result, result_type, batch_size, n_folders, self.beta_rank, boundary_board.device
        )
        fields = {
            name: getattr(result, name)
            for name in (
                "base_log_z", "single_card_log_ratio", "pair_card_log_ratio",
                "beta_eigenvalue", "beta_eigenvector", "folder_log_mass",
                "folder_single_card_log_ratio", "folder_card_marginal",
                "folder_delta", "tight_cluster_scalar", "folder_output_valid",
                "output_valid",
            )
        }
        fields.update({
            "folder_mask": folded.folder_mask,
            "folder_line_meta": folded.line_meta,
            "folder_seat": (
                folded.seat_ids
                if folded.seat_ids is not None
                else torch.arange(n_folders, device=boundary_board.device)[None].expand(
                    batch_size, -1
                )
            ),
        })
        if result.metadata.get("folder_api_version") != EXPECTED_FOLDER_API_VERSION:
            raise RuntimeError(
                "incompatible evaluator folder summary: expected folder_api_version "
                f"{EXPECTED_FOLDER_API_VERSION!r}"
            )
        return PartitionSummaryBatch(fields=fields, metadata=result.metadata)
