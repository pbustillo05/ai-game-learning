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

from pbg.games.poker.cards import validate_evaluator_combo_cards


N_CARDS = 52
N_COMBOS = 1326
PAD_CARD = 255
EXPECTED_API_VERSION = "pbg-v2"


@dataclass
class FoldSnapshotBatch:
    """Batched frozen fold-time inputs.

    ``reach_weights`` is float32 ``[B,F,1326]``, ``board_at_fold`` is uint8
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

    Standard fields are ``base_log_z`` float32 ``[B]``,
    ``single_card_log_ratio`` float32 ``[B,52]``, optional
    ``pair_card_log_ratio`` float32 ``[B,52,52]``, and ``output_valid`` bool
    ``[B]``.  The tensors are the evaluator result tensors themselves.
    """

    fields: dict[str, Tensor]
    metadata: dict[str, object]


class PokerGPUPartitionAdapter:
    """Thin, lazy adapter for ``poker_gpu_evaluator.pbg.partition_many``."""

    def __init__(self) -> None:
        self._partition_many: Callable[..., Any] | None = None
        self._result_type: type[Any] | None = None

    def _load_facade(self) -> tuple[Callable[..., Any], type[Any]]:
        if self._partition_many is not None and self._result_type is not None:
            return self._partition_many, self._result_type
        try:
            from poker_gpu_evaluator.pbg import (
                API_VERSION,
                COMBO_CARDS,
                PartitionSummaryResult,
                partition_many,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "poker_gpu_evaluator with the public pbg-v2 facade is required; "
                "install the sibling package (an editable install is sufficient)"
            ) from exc
        if API_VERSION != EXPECTED_API_VERSION:
            raise RuntimeError(
                "incompatible poker_gpu_evaluator facade: expected API_VERSION "
                f"{EXPECTED_API_VERSION!r}, got {API_VERSION!r}; update the sibling package"
            )
        validate_evaluator_combo_cards(COMBO_CARDS)
        self._partition_many = partition_many
        self._result_type = PartitionSummaryResult
        return partition_many, PartitionSummaryResult

    @staticmethod
    def _normalize_board(board_cards: Tensor) -> Tensor:
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
        valid = padding | ((board_cards >= 0) & (board_cards < N_CARDS))  # [B,5]
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

    @staticmethod
    def _validate_inputs(
        board_cards: Tensor,
        live_range_weights: Tensor,
        folded: FoldSnapshotBatch,
    ) -> None:
        if not isinstance(live_range_weights, Tensor):
            raise TypeError("live_range_weights must be a torch.Tensor")
        if live_range_weights.ndim != 3 or live_range_weights.shape[2] != N_COMBOS:
            raise ValueError("live_range_weights must have shape [B,L,1326]")
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
        if reach.ndim != 3 or reach.shape[2] != N_COMBOS:
            raise ValueError("folded.reach_weights must have shape [B,F,1326]")
        batch_size, n_folders = reach.shape[:2]
        if reach.dtype != torch.float32:
            raise TypeError("folded.reach_weights must have dtype torch.float32")
        if fold_board.shape != (batch_size, n_folders, 5):
            raise ValueError("folded.board_at_fold must have shape [B,F,5]")
        if fold_board.dtype != torch.uint8:
            raise TypeError("folded.board_at_fold must have dtype torch.uint8")
        fold_padding = fold_board == PAD_CARD  # [B,F,5]
        fold_valid = fold_padding | (fold_board < N_CARDS)  # [B,F,5]
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

    @staticmethod
    def _validate_result(result: Any, result_type: type[Any], batch_size: int, device: torch.device) -> None:
        if not isinstance(result, result_type):
            raise TypeError("partition_many returned an unexpected result type")
        expected = (
            ("base_log_z", (batch_size,), torch.float32),
            ("single_card_log_ratio", (batch_size, N_CARDS), torch.float32),
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
        pair = getattr(result, "pair_card_log_ratio", None)
        if pair is not None:
            if not isinstance(pair, Tensor):
                raise TypeError("partition result pair_card_log_ratio must be a Tensor or None")
            if pair.shape != (batch_size, N_CARDS, N_CARDS):
                raise ValueError("partition result pair_card_log_ratio must have shape [B,52,52]")
            if pair.dtype != torch.float32:
                raise TypeError("partition result pair_card_log_ratio has the wrong dtype")
            if pair.device != device:
                raise ValueError("partition result pair_card_log_ratio is on the wrong device")
            if not pair.is_contiguous():
                raise ValueError("partition result pair_card_log_ratio must be contiguous")
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
        padding. ``live_range_weights`` is float32 ``[B,L,1326]``. Fold inputs
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
            return PartitionSummaryBatch(
                fields={
                    "base_log_z": zeros,
                    "single_card_log_ratio": torch.zeros(
                        (batch_size, N_CARDS),
                        dtype=torch.float32,
                        device=boundary_board.device,
                    ),
                    "pair_card_log_ratio": torch.zeros(
                        (batch_size, N_CARDS, N_CARDS),
                        dtype=torch.float32,
                        device=boundary_board.device,
                    ),
                    "output_valid": torch.ones(
                        (batch_size,), dtype=torch.bool, device=boundary_board.device
                    ),
                },
                metadata={"api_version": EXPECTED_API_VERSION, "neutral_no_folders": True},
            )

        partition_many, result_type = self._load_facade()
        range_weights = folded.reach_weights.contiguous()  # [B,F,1326]
        row_mask = folded.folder_mask.contiguous()  # [B,F]
        result = partition_many(
            board_cards=boundary_board,
            range_weights=range_weights,
            row_mask=row_mask,
        )
        self._validate_result(result, result_type, batch_size, boundary_board.device)

        n_folders = folded.reach_weights.shape[1]
        folder_base = torch.zeros(
            (batch_size, n_folders), dtype=torch.float32, device=boundary_board.device
        )
        folder_single = torch.zeros(
            (batch_size, n_folders, N_CARDS),
            dtype=torch.float32,
            device=boundary_board.device,
        )
        folder_pair = torch.zeros(
            (batch_size, n_folders, N_CARDS, N_CARDS),
            dtype=torch.float32,
            device=boundary_board.device,
        )
        folder_valid = torch.zeros(
            (batch_size, n_folders), dtype=torch.bool, device=boundary_board.device
        )
        active = folded.folder_mask.nonzero(as_tuple=False)
        if active.numel():
            active_board = boundary_board[active[:, 0]].contiguous()
            active_range = folded.reach_weights[
                active[:, 0], active[:, 1]
            ][:, None, :].contiguous()
            active_mask = torch.ones(
                (active.shape[0], 1), dtype=torch.bool, device=boundary_board.device
            )
            per_folder = partition_many(
                board_cards=active_board,
                range_weights=active_range,
                row_mask=active_mask,
            )
            self._validate_result(
                per_folder, result_type, active.shape[0], boundary_board.device
            )
            folder_base[active[:, 0], active[:, 1]] = per_folder.base_log_z
            folder_single[active[:, 0], active[:, 1]] = per_folder.single_card_log_ratio
            folder_valid[active[:, 0], active[:, 1]] = per_folder.output_valid
            if per_folder.pair_card_log_ratio is not None:
                folder_pair[active[:, 0], active[:, 1]] = per_folder.pair_card_log_ratio

        fields = {
            "base_log_z": result.base_log_z,
            "single_card_log_ratio": result.single_card_log_ratio,
            "output_valid": result.output_valid,
            "folder_log_mass": folder_base,
            "folder_single_card_log_ratio": folder_single,
            "folder_pair_card_log_ratio": folder_pair,
            "folder_output_valid": folder_valid,
            "folder_mask": folded.folder_mask,
            "folder_line_meta": folded.line_meta,
            "folder_seat": (
                folded.seat_ids
                if folded.seat_ids is not None
                else torch.arange(n_folders, device=boundary_board.device)[None].expand(
                    batch_size, -1
                )
            ),
        }
        if result.pair_card_log_ratio is not None:
            fields["pair_card_log_ratio"] = result.pair_card_log_ratio
        return PartitionSummaryBatch(fields=fields, metadata=result.metadata)
