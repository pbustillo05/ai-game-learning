"""Evaluator-owned partition-summary data transfer boundary.

This module only batches fold snapshots and forwards them to the bound
``poker_gpu_evaluator`` API. Private-card collision and partition
calculations remain entirely evaluator-owned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from typing import Any

import torch
from torch import Tensor

from pbg.games.poker.cards import DeckGeometry, validate_evaluator_combo_cards


PAD_CARD = 255
EXPECTED_CONTRACT_VERSION = "1"


def _require_device(predicate: Tensor, message: str) -> None:
    """Validate a scalar predicate without synchronizing a CUDA caller."""

    if predicate.device.type == "cuda":
        torch._assert_async(predicate, message)
    elif not bool(predicate):
        raise ValueError(message)


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

    The compact model-ready fields include hand log ratios, signed beta factors,
    per-folder mass/card pressure, tight contraction, and fold-line/seat metadata.
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
        self._evaluator: Any | None = None

    def _bound_evaluator(self) -> Any:
        if self._evaluator is not None:
            return self._evaluator
        try:
            package = importlib.import_module("poker_gpu_evaluator")
            contract_version = package.CONTRACT_VERSION
            evaluator = package.bind(rank_count=self.rank_count)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                "poker_gpu_evaluator with the bound contract-v1 API is required; "
                "install the sibling package (an editable install is sufficient)"
            ) from exc
        if contract_version != EXPECTED_CONTRACT_VERSION:
            raise RuntimeError(
                "incompatible poker_gpu_evaluator: expected CONTRACT_VERSION "
                f"{EXPECTED_CONTRACT_VERSION!r}, got {contract_version!r}; "
                "update the sibling package"
            )
        validate_evaluator_combo_cards(evaluator.combo_cards, self.rank_count)
        if evaluator.info.contract_version != contract_version:
            raise RuntimeError("bound evaluator contract version disagrees with the package")
        self._evaluator = evaluator
        return evaluator

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
        _require_device(
            valid.all(),
            "board_cards must contain card ids in [0,52), -1, or 255",
        )
        after_padding = padding.to(torch.int32).cumsum(dim=1) > 0  # [B,5]
        _require_device(
            (~padding & after_padding).logical_not().all(),
            "real board cards must precede padding",
        )
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
        _require_device(
            fold_valid.all(),
            "folded.board_at_fold must contain card ids in [0,52) or 255",
        )
        fold_after_padding = fold_padding.to(torch.int32).cumsum(dim=2) > 0  # [B,F,5]
        _require_device(
            (~fold_padding & fold_after_padding).logical_not().all(),
            "real fold-time board cards must precede padding",
        )
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
        batch_size: int,
        n_folders: int,
        beta_rank: int,
        device: torch.device,
    ) -> None:
        n_cards = self.geometry.n_cards
        expected = (
            ("hand_log_ratio", (batch_size, self.geometry.n_hands), torch.float32),
            ("beta_eigenvalue", (batch_size, beta_rank), torch.float32),
            ("beta_eigenvector", (batch_size, n_cards, beta_rank), torch.float32),
            ("folder_log_mass", (batch_size, n_folders), torch.float32),
            ("folder_card_mass", (batch_size, n_folders, n_cards), torch.float32),
            ("tight_cluster_scalar", (batch_size,), torch.float32),
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
        # only because they remain part of the local call-site contract;
        # including them in the neutral partition law would double-count live
        # collision geometry and produce a non-neutral correction with K=0.
        if folded.reach_weights.shape[1] == 0:
            zeros = torch.zeros(
                (batch_size,), dtype=torch.float32, device=boundary_board.device
            )
            n_folders = folded.reach_weights.shape[1]
            n_cards = self.geometry.n_cards
            fields = {
                "hand_log_ratio": torch.zeros((batch_size, self.geometry.n_hands), dtype=torch.float32, device=boundary_board.device),
                "beta_eigenvalue": torch.zeros((batch_size, self.beta_rank), dtype=torch.float32, device=boundary_board.device),
                "beta_eigenvector": torch.zeros((batch_size, n_cards, self.beta_rank), dtype=torch.float32, device=boundary_board.device),
                "folder_log_mass": torch.zeros((batch_size, n_folders), dtype=torch.float32, device=boundary_board.device),
                "folder_card_mass": torch.zeros((batch_size, n_folders, n_cards), dtype=torch.float32, device=boundary_board.device),
                "tight_cluster_scalar": zeros,
                "folder_mask": folded.folder_mask,
                "folder_line_meta": folded.line_meta,
                "folder_seat": folded.seat_ids if folded.seat_ids is not None else torch.arange(
                    n_folders, dtype=torch.int64, device=boundary_board.device
                )[None].expand(batch_size, -1),
            }
            return PartitionSummaryBatch(
                fields=fields,
                metadata={
                    "contract_version": EXPECTED_CONTRACT_VERSION,
                    "neutral_no_folders": True,
                },
            )

        evaluator = self._bound_evaluator()
        range_weights = folded.reach_weights.contiguous()  # [B,F,H]
        row_mask = folded.folder_mask.contiguous()  # [B,F]
        result = evaluator.summarize_folders(
            board_cards=boundary_board,
            folder_weights=range_weights,
            row_mask=row_mask,
            beta_rank=self.beta_rank,
        )
        n_folders = folded.reach_weights.shape[1]
        self._validate_result(
            result, batch_size, n_folders, self.beta_rank, boundary_board.device
        )
        fields = {
            name: getattr(result, name)
            for name in (
                "hand_log_ratio", "beta_eigenvalue", "beta_eigenvector",
                "folder_log_mass", "folder_card_mass", "tight_cluster_scalar",
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
        return PartitionSummaryBatch(fields=fields, metadata=asdict(evaluator.info))
