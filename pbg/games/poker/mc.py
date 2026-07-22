"""Thin adapters to the bound :mod:`poker_gpu_evaluator` API.

Poker collision laws, profile sampling, runouts, hand ranking, and winner
selection are all owned by the sibling evaluator.  This module only translates
public engine state and validates the stable tensor contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from typing import Any, Mapping

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats, ReachView
from pbg.games.poker.cards import DeckGeometry, validate_evaluator_combo_cards

PAD_CARD = 255
EXPECTED_CONTRACT_VERSION = "2"
SUPPORTED_CONTRACT_VERSIONS = frozenset({"2"})


class EvaluatorContractError(RuntimeError):
    """The sibling returned a value outside its frozen public contract."""


def _require_device(
    predicate: Tensor,
    message: str,
    error_type: type[Exception] = ValueError,
) -> None:
    """Validate a scalar predicate without synchronizing a CUDA caller."""

    if predicate.device.type == "cuda":
        torch._assert_async(predicate, message)
    elif not bool(predicate):
        raise error_type(message)


@dataclass
class ShowdownResult:
    """Validated evaluator-owned fixed-showdown payoff."""

    payoff: Tensor  # float32 [B,P]
    metadata: Mapping[str, object]


def _load_evaluator(rank_count: int) -> Any:
    """Import, bind, and validate the production evaluator lazily."""

    try:
        package = importlib.import_module("poker_gpu_evaluator")
        contract_version = package.CONTRACT_VERSION
        evaluator = package.bind(rank_count=rank_count)
        combo_cards = evaluator.combo_cards
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "poker_gpu_evaluator with the bound contract-v2 API is required; "
            "install or update the sibling package (an editable install is sufficient)"
        ) from exc
    if contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        raise RuntimeError(
            "incompatible poker_gpu_evaluator: expected CONTRACT_VERSION "
            f"{EXPECTED_CONTRACT_VERSION!r}, got {contract_version!r}; "
            "update the sibling package"
        )
    validate_evaluator_combo_cards(combo_cards, rank_count)
    if evaluator.info.contract_version != contract_version:
        raise RuntimeError("bound evaluator contract version disagrees with the package")
    return evaluator


def _evaluator_metadata(evaluator: Any) -> Mapping[str, object]:
    """Return immutable evaluator identity as host-only metadata."""

    try:
        return asdict(evaluator.info)
    except (TypeError, AttributeError) as exc:
        raise EvaluatorContractError("bound evaluator must expose dataclass info") from exc


def _require_accelerator(device: torch.device, operation: str) -> None:
    if device.type not in {"cuda", "mps"}:
        raise ValueError(f"{operation} requires CUDA or MPS tensors")


def stratum_levels(stratum_players: Tensor) -> Tensor:
    """Convert padded nested eligibility ``[B,K,P]`` to highest levels.

    Engine strata use a fixed ``K=P`` width and may have zero-amount empty
    padding rows. The evaluator still evaluates every named level, so those
    payoff-neutral rows inherit all contenders before conversion.
    """

    if not isinstance(stratum_players, Tensor) or stratum_players.ndim != 3:
        raise TypeError("stratum_players must be a Tensor with shape [B,K,P]")
    if stratum_players.dtype != torch.bool:
        raise TypeError("stratum_players must have dtype torch.bool")
    contenders = stratum_players.any(dim=1)
    empty = ~stratum_players.any(dim=2)
    normalized = torch.where(
        empty[:, :, None], contenders[:, None, :], stratum_players
    )
    levels = torch.arange(
        stratum_players.shape[1], dtype=torch.int32, device=stratum_players.device
    )
    return torch.where(
        normalized,
        levels[None, :, None],
        torch.full((), -1, dtype=torch.int32, device=stratum_players.device),
    ).amax(dim=1).contiguous()


def _normalize_board(board_cards: Tensor, rank_count: int = 13) -> Tensor:
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
    geometry = DeckGeometry(rank_count)
    padding = (board_cards == -1) | (board_cards == PAD_CARD)  # [B,5]
    valid = padding | (
        (board_cards >= geometry.first_card) & (board_cards < 52)
    )  # [B,5]
    _require_device(
        valid.all(), "board_cards must contain card ids in [0,52), -1, or 255"
    )
    after_padding = padding.to(torch.int32).cumsum(dim=1) > 0  # [B,5]
    _require_device(
        (padding | ~after_padding).all(),
        "real board cards must precede padding",
    )
    real_counts = (~padding).sum(dim=1)
    _require_device(
        (
            (real_counts == 0)
            | (real_counts == 3)
            | (real_counts == 4)
            | (real_counts == 5)
        ).all(),
        "a board must contain 0, 3, 4, or 5 real cards",
    )
    return torch.where(
        padding, torch.full_like(board_cards, PAD_CARD), board_cards
    ).to(dtype=torch.uint8).contiguous()  # [B,5]


def _require_tensor(
    result: Any,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    operation: str,
) -> Tensor:
    value = getattr(result, name, None)
    if not isinstance(value, Tensor):
        raise EvaluatorContractError(f"{operation} result {name} must be a torch.Tensor")
    if value.shape != shape:
        raise EvaluatorContractError(
            f"{operation} result {name} must have shape {list(shape)}, got {list(value.shape)}"
        )
    if value.dtype != dtype:
        raise EvaluatorContractError(f"{operation} result {name} has the wrong dtype")
    if value.device != device:
        raise EvaluatorContractError(f"{operation} result {name} is on the wrong device")
    if not value.is_contiguous():
        raise EvaluatorContractError(f"{operation} result {name} must be contiguous")
    return value


def _allocate_seeds(batch_size: int, gen: torch.Generator) -> Tensor:
    if not isinstance(gen, torch.Generator):
        raise TypeError("gen must be a torch.Generator")
    device = torch.device(getattr(gen, "device", "cpu"))
    return torch.empty((batch_size,), dtype=torch.int64, device=device).random_(generator=gen)


def _linear_reach(
    log_rows: Tensor, dealt_mask: Tensor, rank_count: int = 13
) -> Tensor:
    if not isinstance(log_rows, Tensor):
        raise TypeError("reach rows must be a torch.Tensor")
    n_hands = DeckGeometry(rank_count).n_hands
    if log_rows.ndim != 3 or log_rows.shape[2] != n_hands:
        raise ValueError(f"reach rows must have shape [B,P,{n_hands}]")
    if log_rows.dtype != torch.float32:
        raise TypeError("reach rows must have dtype torch.float32")
    if dealt_mask.shape != log_rows.shape[:2] or dealt_mask.dtype != torch.bool:
        raise ValueError("dealt_mask must be bool [B,P]")
    if dealt_mask.device != log_rows.device:
        raise ValueError("reach rows and dealt_mask must share a device")
    finite_input = ~torch.isnan(log_rows) & ~torch.isposinf(log_rows)
    _require_device(
        (~dealt_mask[:, :, None] | finite_input).all(),
        "dealt reach rows may not contain NaN or +inf",
    )
    row_max = log_rows.max(dim=-1).values  # [B,P]
    _require_device(
        (~dealt_mask | torch.isfinite(row_max)).all(),
        "every dealt reach row must have positive finite support",
    )
    shifted = log_rows - torch.where(dealt_mask, row_max, torch.zeros_like(row_max))[:, :, None]
    weights = torch.where(dealt_mask[:, :, None], shifted.exp(), torch.zeros_like(shifted))
    return weights.to(torch.float32).contiguous()  # [B,P,H]


def _reach_rows(reach: ReachView | Tensor | Any) -> Tensor:
    if isinstance(reach, Tensor):
        return reach
    for name in ("log_r", "rows"):
        value = getattr(reach, name, None)
        if isinstance(value, Tensor):
            return value
    raise TypeError("reach must be a Tensor or expose Tensor log_r/rows")


class PokerGPUCompletion:
    """Adapter for evaluator-owned Monte Carlo completion."""

    def __init__(self, *, rank_count: int = 13) -> None:
        if isinstance(rank_count, bool) or not 4 <= rank_count <= 13:
            raise ValueError("rank_count must be an integer in [4,13]")
        self.rank_count = int(rank_count)
        self._evaluator: Any | None = None

    def _bound_evaluator(self) -> Any:
        if self._evaluator is None:
            self._evaluator = _load_evaluator(self.rank_count)
        return self._evaluator

    def _validate_result(
        self,
        result: Any,
        batch_size: int,
        p_max: int,
        device: torch.device,
        metadata: Mapping[str, object],
    ) -> BatchedCompletionStats:
        op = "complete"
        n_hands = DeckGeometry(self.rank_count).n_hands
        p_hat = _require_tensor(result, "p_hat", (batch_size, p_max, n_hands), torch.float32, device, op)
        v_hat = _require_tensor(result, "v_hat", (batch_size, p_max, n_hands), torch.float32, device, op)
        n_eff = _require_tensor(result, "n_eff", (batch_size,), torch.float32, device, op)
        return BatchedCompletionStats(
            p_hat=p_hat,
            v_hat=v_hat,
            n_eff=n_eff,
            metadata=metadata,
        )


class PokerGPURejectionSampler:
    """Adapter for exact whole-profile all-or-raise sampling."""

    def __init__(
        self,
        *,
        max_attempts: int = 100_000,
        rank_count: int = 13,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        self.max_attempts = max_attempts
        if isinstance(rank_count, bool) or not 4 <= rank_count <= 13:
            raise ValueError("rank_count must be an integer in [4,13]")
        self.rank_count = int(rank_count)
        self._evaluator: Any | None = None

    def _bound_evaluator(self) -> Any:
        if self._evaluator is None:
            self._evaluator = _load_evaluator(self.rank_count)
        return self._evaluator

    def sample(
        self,
        reach_rows: Tensor,
        board: Tensor,
        dealt_mask: Tensor,
        n: int,
        gen: torch.Generator,
    ) -> Tensor:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        board_cards = _normalize_board(board, self.rank_count)
        if dealt_mask.dtype != torch.bool or dealt_mask.ndim != 2:
            raise TypeError("dealt_mask must be bool [B,P]")
        batch_size = dealt_mask.shape[0]
        if board_cards.shape[0] != batch_size or board_cards.device != reach_rows.device:
            raise ValueError("board, reach rows, and dealt_mask must share B and device")
        seed = _allocate_seeds(batch_size, gen)
        return self.sample_seeded(
            reach_rows, board_cards, dealt_mask, n, seed,
            board_is_normalized=True,
        )

    def sample_seeded(
        self,
        reach_rows: Tensor,
        board: Tensor,
        dealt_mask: Tensor,
        n: int,
        seed: Tensor,
        *,
        board_is_normalized: bool = False,
    ) -> Tensor:
        """Sample with one caller-owned deterministic seed per batch row."""

        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        board_cards = (
            board
            if board_is_normalized
            else _normalize_board(board, self.rank_count)
        )
        if dealt_mask.dtype != torch.bool or dealt_mask.ndim != 2:
            raise TypeError("dealt_mask must be bool [B,P]")
        range_weights = _linear_reach(reach_rows, dealt_mask, self.rank_count)
        _require_accelerator(reach_rows.device, "sample_joint_profiles")
        batch_size, p_max = dealt_mask.shape
        if board_cards.shape[0] != batch_size or board_cards.device != reach_rows.device:
            raise ValueError("board, reach rows, and dealt_mask must share B and device")
        if seed.shape != (batch_size,) or seed.dtype != torch.int64:
            raise ValueError("seed must be int64 [B]")
        if seed.device != reach_rows.device:
            raise ValueError("seed and reach rows must share a device")
        seed = seed.contiguous()
        combo_ids = self._bound_evaluator().sample_joint_profiles(
            board_cards=board_cards,
            range_weights=range_weights,
            n_samples=n,
            seed=seed,
            max_attempts=self.max_attempts,
        )
        op = "sample_joint_profiles"
        if not isinstance(combo_ids, Tensor):
            raise EvaluatorContractError(f"{op} must return a torch.Tensor")
        if combo_ids.shape != (batch_size, n, p_max):
            raise EvaluatorContractError(f"{op} returned the wrong shape")
        if combo_ids.dtype != torch.int32 or combo_ids.device != reach_rows.device:
            raise EvaluatorContractError(f"{op} returned the wrong dtype or device")
        if not combo_ids.is_contiguous():
            raise EvaluatorContractError(f"{op} result must be contiguous")
        inactive = ~dealt_mask[:, None, :].expand_as(combo_ids)
        _require_device(
            (~inactive | (combo_ids == -1)).all(),
            "sample_joint_profiles must return -1 for undealt seats",
            EvaluatorContractError,
        )
        dealt = dealt_mask[:, None, :].expand_as(combo_ids)
        _require_device(
            (~dealt | (combo_ids >= 0)).all(),
            "sampled profiles must contain every dealt seat",
            EvaluatorContractError,
        )
        _require_device(
            (~dealt | (combo_ids < DeckGeometry(self.rank_count).n_hands)).all(),
            "sampled profile contains an invalid combo id",
            EvaluatorContractError,
        )
        return combo_ids


class PokerGPUShowdownEvaluator:
    """Adapter for exact fixed-showdown payoffs."""

    def __init__(self, *, rank_count: int = 13) -> None:
        if isinstance(rank_count, bool) or not 4 <= rank_count <= 13:
            raise ValueError("rank_count must be an integer in [4,13]")
        self.rank_count = int(rank_count)
        self._evaluator: Any | None = None

    def _bound_evaluator(self) -> Any:
        if self._evaluator is None:
            self._evaluator = _load_evaluator(self.rank_count)
        return self._evaluator

    def evaluate(
        self,
        *,
        board_cards: Tensor,
        combo_ids: Tensor,
        stratum_level: Tensor,
        payoff_baseline: Tensor,
        stratum_amount: Tensor,
    ) -> ShowdownResult:
        board = _normalize_board(board_cards, self.rank_count)
        _require_device(
            (board != PAD_CARD).all(),
            "fixed showdown requires a final five-card board",
        )
        if combo_ids.dtype != torch.int32 or combo_ids.ndim != 2:
            raise TypeError("combo_ids must be int32 [B,P]")
        batch_size, p_max = combo_ids.shape
        if stratum_level.shape != (batch_size, p_max) or stratum_level.dtype != torch.int32:
            raise TypeError("stratum_level must be int32 [B,P]")
        if payoff_baseline.shape != (batch_size, p_max) or payoff_baseline.dtype != torch.float32:
            raise TypeError("payoff_baseline must be float32 [B,P]")
        if stratum_amount.ndim != 2 or stratum_amount.shape[0] != batch_size or stratum_amount.dtype != torch.float32:
            raise TypeError("stratum_amount must be float32 [B,K]")
        tensors = (board, combo_ids, stratum_level, payoff_baseline, stratum_amount)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all showdown tensors must share a device")
        _require_accelerator(board.device, "evaluate_showdowns")
        evaluator = self._bound_evaluator()
        payoff = evaluator.evaluate_showdowns(
            board_cards=board,
            combo_ids=combo_ids.contiguous(),
            stratum_level=stratum_level.contiguous(),
            payoff_baseline=payoff_baseline.contiguous(),
            stratum_amount=stratum_amount.contiguous(),
        )
        op = "evaluate_showdowns"
        if not isinstance(payoff, Tensor) or payoff.shape != (batch_size, p_max):
            raise EvaluatorContractError(f"{op} must return float32 [B,P]")
        if payoff.dtype != torch.float32 or payoff.device != board.device:
            raise EvaluatorContractError(f"{op} returned the wrong dtype or device")
        if not payoff.is_contiguous():
            raise EvaluatorContractError(f"{op} returned an invalid payoff")
        _require_device(
            torch.isfinite(payoff).all(),
            f"{op} returned an invalid payoff",
            EvaluatorContractError,
        )
        _require_device(
            (payoff.sum(dim=1).abs() <= 2e-4).all(),
            f"{op} returned a non-conserved payoff",
            EvaluatorContractError,
        )
        return ShowdownResult(payoff, _evaluator_metadata(evaluator))
