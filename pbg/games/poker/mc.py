"""Thin adapters to the public ``poker_gpu_evaluator.pbg`` facade.

Poker collision laws, profile sampling, runouts, hand ranking, and winner
selection are all owned by the sibling evaluator.  This module only translates
public engine state and validates the stable tensor contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats, ReachView
from pbg.games.poker.cards import validate_evaluator_combo_cards
from pbg.games.poker.engine import FOLDED, NLHEEngine

N_COMBOS = 1326
PAD_CARD = 255
EXPECTED_API_VERSION = "pbg-v3"


class EvaluatorContractError(RuntimeError):
    """The sibling returned a value outside its frozen public contract."""


class AcceptanceError(RuntimeError):
    """One or more requested exact joint profiles were not accepted."""

    def __init__(
        self,
        failed_indices: Tensor,
        acceptance_rate: Tensor,
        attempt_count: Tensor,
        metadata: Mapping[str, object],
    ) -> None:
        self.failed_indices = failed_indices
        self.acceptance_rate = acceptance_rate
        self.attempt_count = attempt_count
        self.metadata = metadata
        indices = failed_indices.detach().cpu().tolist()
        rates = acceptance_rate.detach().cpu().tolist()
        super().__init__(
            "poker_gpu_evaluator failed to accept requested joint profiles; "
            f"failed_indices={indices}, acceptance_rate={rates}, metadata={dict(metadata)!r}"
        )


@dataclass(frozen=True)
class _CompletionConfig:
    """Structural counterpart of the facade's frozen completion config."""

    n_samples: int
    n_partials: int = 8
    backend: str = "auto"


@dataclass
class ShowdownResult:
    """Validated evaluator-owned winners and tie shares."""

    winner_mask: Tensor  # int32 [B,K]
    showdown_share: Tensor  # float32 [B,K,P]
    metadata: Mapping[str, object]


def _load_operation(name: str) -> Callable[..., Any]:
    """Import and version-check one stable facade operation, lazily."""

    try:
        facade = importlib.import_module("poker_gpu_evaluator.pbg")
        api_version = facade.API_VERSION
        combo_cards = facade.combo_cards(13)
        operation = getattr(facade, name)
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "poker_gpu_evaluator with the public pbg-v3 facade is required; "
            "install or update the sibling package (an editable install is sufficient)"
        ) from exc
    if api_version != EXPECTED_API_VERSION:
        raise RuntimeError(
            "incompatible poker_gpu_evaluator facade: expected API_VERSION "
            f"{EXPECTED_API_VERSION!r}, got {api_version!r}; update the sibling package"
        )
    validate_evaluator_combo_cards(combo_cards)
    return operation


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
    valid = padding | ((board_cards >= 0) & (board_cards < 52))  # [B,5]
    if not bool(valid.all()):
        raise ValueError("board_cards must contain card ids in [0,52), -1, or 255")
    after_padding = padding.to(torch.int32).cumsum(dim=1) > 0  # [B,5]
    if bool((~padding & after_padding).any()):
        raise ValueError("real board cards must precede padding")
    real_counts = (~padding).sum(dim=1)
    if not bool(((real_counts == 0) | (real_counts == 3) | (real_counts == 4) | (real_counts == 5)).all()):
        raise ValueError("a board must contain 0, 3, 4, or 5 real cards")
    return torch.where(
        padding, torch.full_like(board_cards, PAD_CARD), board_cards
    ).to(dtype=torch.uint8).contiguous()  # [B,5]


def _validate_metadata(metadata: Any, operation: str) -> Mapping[str, object]:
    if not isinstance(metadata, dict):
        raise EvaluatorContractError(f"{operation} metadata must be a dict")
    if any(isinstance(value, Tensor) for value in metadata.values()):
        raise EvaluatorContractError(f"{operation} metadata may not contain tensors")
    return metadata


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


def _linear_reach(log_rows: Tensor, dealt_mask: Tensor) -> Tensor:
    if not isinstance(log_rows, Tensor):
        raise TypeError("reach rows must be a torch.Tensor")
    if log_rows.ndim != 3 or log_rows.shape[2] != N_COMBOS:
        raise ValueError("reach rows must have shape [B,P,1326]")
    if log_rows.dtype != torch.float32:
        raise TypeError("reach rows must have dtype torch.float32")
    if dealt_mask.shape != log_rows.shape[:2] or dealt_mask.dtype != torch.bool:
        raise ValueError("dealt_mask must be bool [B,P]")
    if dealt_mask.device != log_rows.device:
        raise ValueError("reach rows and dealt_mask must share a device")
    if bool(torch.isnan(log_rows[dealt_mask]).any()) or bool(torch.isposinf(log_rows[dealt_mask]).any()):
        raise ValueError("dealt reach rows may not contain NaN or +inf")
    row_max = log_rows.max(dim=-1).values  # [B,P]
    if bool((~torch.isfinite(row_max[dealt_mask])).any()):
        raise ValueError("every dealt reach row must have positive finite support")
    shifted = log_rows - torch.where(dealt_mask, row_max, torch.zeros_like(row_max))[:, :, None]
    weights = torch.where(dealt_mask[:, :, None], shifted.exp(), torch.zeros_like(shifted))
    return weights.to(torch.float32).contiguous()  # [B,P,1326]


def _reach_rows(reach: ReachView | Tensor | Any) -> Tensor:
    if isinstance(reach, Tensor):
        return reach
    for name in ("log_r", "rows"):
        value = getattr(reach, name, None)
        if isinstance(value, Tensor):
            return value
    raise TypeError("reach must be a Tensor or expose Tensor log_r/rows")


def _engine(game: Any) -> NLHEEngine:
    candidate = getattr(game, "engine", game)
    if not isinstance(candidate, NLHEEngine):
        raise TypeError("game must be an NLHEEngine or expose one as game.engine")
    return candidate


class PokerGPUCompletion:
    """Adapter for evaluator-owned Monte Carlo completion."""

    def __init__(
        self, *, backend: str = "auto", n_partials: int = 8, rank_count: int = 13
    ) -> None:
        if backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
        if isinstance(n_partials, bool) or not isinstance(n_partials, int) or n_partials <= 0:
            raise ValueError("n_partials must be a positive integer")
        self.backend = backend
        self.n_partials = n_partials
        if isinstance(rank_count, bool) or not 4 <= rank_count <= 13:
            raise ValueError("rank_count must be an integer in [4,13]")
        self.rank_count = int(rank_count)
        self._complete: Callable[..., Any] | None = None

    def _operation(self) -> Callable[..., Any]:
        if self._complete is None:
            self._complete = _load_operation("complete")
        return self._complete

    def complete(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: ReachView | Tensor | Any,
        n_samples: int,
        gen: torch.Generator,
    ) -> BatchedCompletionStats:
        return self._complete_impl(
            game, tree, node_ids, reach, n_samples, gen=gen, seed=None
        )

    def complete_keyed(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: ReachView | Tensor | Any,
        n_samples: int,
        keys: Tensor,
    ) -> BatchedCompletionStats:
        """Complete rows using caller-owned evaluator stream seeds ``[B]``."""

        return self._complete_impl(
            game, tree, node_ids, reach, n_samples, gen=None, seed=keys
        )

    def _complete_impl(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: ReachView | Tensor | Any,
        n_samples: int,
        *,
        gen: torch.Generator | None,
        seed: Tensor | None,
    ) -> BatchedCompletionStats:
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(node_ids, Tensor) or node_ids.ndim != 1:
            raise ValueError("node_ids must be a one-dimensional Tensor")
        if node_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("node_ids must have dtype int32 or int64")
        if not hasattr(tree, "state_ref") or not isinstance(tree.state_ref, Tensor):
            raise TypeError("tree must expose a state_ref Tensor")
        state_refs = tree.state_ref[node_ids.to(device=tree.state_ref.device, dtype=torch.long)]
        engine = _engine(game)
        rows = _reach_rows(reach)
        batch_size, p_max = rows.shape[:2]
        if batch_size != node_ids.numel():
            raise ValueError("reach rows and node_ids must have the same B dimension")
        if p_max < engine.n_players:
            raise ValueError("reach rows have fewer seats than the poker engine")
        state_refs = state_refs.to(device=rows.device, dtype=torch.int32)
        refs = [int(value) for value in state_refs.detach().cpu().tolist()]
        boards = torch.tensor(
            [engine.states.board[ref] for ref in refs], dtype=torch.int16, device=rows.device
        )
        board_cards = _normalize_board(boards)
        dealt_mask = torch.zeros((batch_size, p_max), dtype=torch.bool, device=rows.device)
        dealt_mask[:, : engine.n_players] = True
        contender_mask = torch.zeros_like(dealt_mask)
        for batch, ref in enumerate(refs):
            contender_mask[batch, : engine.n_players] = torch.tensor(
                [status != FOLDED for status in engine.states.status[ref]],
                dtype=torch.bool,
                device=rows.device,
            )
        range_weights = _linear_reach(rows, dealt_mask)
        strata = engine.pot_strata(state_refs)
        k = strata.stratum_amount.shape[1]
        stratum_players = torch.zeros(
            (batch_size, k, p_max), dtype=torch.bool, device=rows.device
        )
        stratum_players[:, :, : engine.n_players] = strata.stratum_players
        payoff_baseline = torch.zeros((batch_size, p_max), dtype=torch.float32, device=rows.device)
        payoff_baseline[:, : engine.n_players] = strata.payoff_baseline
        if seed is None:
            if gen is None:
                raise RuntimeError("mutable poker completion requires a generator")
            seed = _allocate_seeds(batch_size, gen)
        elif (
            seed.shape != (batch_size,)
            or seed.dtype != torch.int64
            or seed.device != rows.device
        ):
            raise ValueError("completion seeds must be device-local int64 [B]")
        seed = seed.contiguous()
        result = self._operation()(
            rank_count=self.rank_count,
            board_cards=board_cards,
            range_weights=range_weights,
            dealt_mask=dealt_mask.contiguous(),
            contender_mask=contender_mask.contiguous(),
            stratum_players=stratum_players.contiguous(),
            payoff_baseline=payoff_baseline.contiguous(),
            stratum_amount=strata.stratum_amount.contiguous(),
            seed=seed,
            config=_CompletionConfig(n_samples, self.n_partials, self.backend),
        )
        metrics = getattr(game, "execution_metrics", None)
        if metrics is not None:
            metrics.record_evaluator(rows=batch_size)
        return self._validate_result(result, batch_size, p_max, k, rows.device)

    @staticmethod
    def _validate_result(
        result: Any, batch_size: int, p_max: int, k: int, device: torch.device
    ) -> BatchedCompletionStats:
        op = "complete"
        p_hat = _require_tensor(result, "p_hat", (batch_size, p_max, N_COMBOS), torch.float32, device, op)
        v_hat = _require_tensor(result, "v_hat", (batch_size, p_max, N_COMBOS), torch.float32, device, op)
        vvec_hat = _require_tensor(result, "vvec_hat", (batch_size, p_max), torch.float32, device, op)
        u_hat = _require_tensor(result, "u_hat", (batch_size,), torch.float32, device, op)
        n_eff = _require_tensor(result, "n_eff", (batch_size,), torch.float32, device, op)
        equity_share = _require_tensor(
            result, "equity_share", (batch_size, p_max, N_COMBOS, k), torch.float32, device, op
        )
        output_valid = _require_tensor(result, "output_valid", (batch_size, p_max, N_COMBOS), torch.bool, device, op)
        sample_count = _require_tensor(result, "sample_count", (batch_size,), torch.int64, device, op)
        estimator_stddev = _require_tensor(
            result, "estimator_stddev", (batch_size, p_max, N_COMBOS), torch.float32, device, op
        )
        metadata = _validate_metadata(getattr(result, "metadata", None), op)
        return BatchedCompletionStats(
            p_hat=p_hat,
            v_hat=v_hat,
            vvec_hat=vvec_hat,
            u_hat=u_hat,
            n_eff=n_eff,
            equity_share=equity_share,
            output_valid=output_valid,
            sample_count=sample_count,
            estimator_stddev=estimator_stddev,
            metadata=metadata,
        )


class PokerGPURejectionSampler:
    """Adapter for exact whole-profile independent rejection sampling."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        max_attempts: int = 100_000,
        rank_count: int = 13,
    ) -> None:
        if backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        self.backend = backend
        self.max_attempts = max_attempts
        if isinstance(rank_count, bool) or not 4 <= rank_count <= 13:
            raise ValueError("rank_count must be an integer in [4,13]")
        self.rank_count = int(rank_count)
        self._sample: Callable[..., Any] | None = None

    def _operation(self) -> Callable[..., Any]:
        if self._sample is None:
            self._sample = _load_operation("sample_joint_profiles")
        return self._sample

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
        board_cards = _normalize_board(board)
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
        board_cards = board if board_is_normalized else _normalize_board(board)
        if dealt_mask.dtype != torch.bool or dealt_mask.ndim != 2:
            raise TypeError("dealt_mask must be bool [B,P]")
        range_weights = _linear_reach(reach_rows, dealt_mask)
        batch_size, p_max = dealt_mask.shape
        if board_cards.shape[0] != batch_size or board_cards.device != reach_rows.device:
            raise ValueError("board, reach rows, and dealt_mask must share B and device")
        if seed.shape != (batch_size,) or seed.dtype != torch.int64:
            raise ValueError("seed must be int64 [B]")
        if seed.device != reach_rows.device:
            raise ValueError("seed and reach rows must share a device")
        seed = seed.contiguous()
        result = self._operation()(
            rank_count=self.rank_count,
            board_cards=board_cards,
            range_weights=range_weights,
            dealt_mask=dealt_mask.contiguous(),
            n_samples=n,
            seed=seed,
            max_attempts=self.max_attempts,
            backend=self.backend,
        )
        op = "sample_joint_profiles"
        combo_ids = _require_tensor(result, "combo_ids", (batch_size, n, p_max), torch.int32, reach_rows.device, op)
        accepted = _require_tensor(result, "accepted", (batch_size, n), torch.bool, reach_rows.device, op)
        attempt_count = _require_tensor(result, "attempt_count", (batch_size, n), torch.int32, reach_rows.device, op)
        acceptance_rate = _require_tensor(result, "acceptance_rate", (batch_size,), torch.float32, reach_rows.device, op)
        metadata = _validate_metadata(getattr(result, "metadata", None), op)
        inactive = ~dealt_mask[:, None, :].expand_as(combo_ids)
        if bool((combo_ids[inactive] != -1).any()):
            raise EvaluatorContractError("sample_joint_profiles must return -1 for undealt seats")
        if bool((combo_ids[dealt_mask[:, None, :].expand_as(combo_ids) & accepted[:, :, None]] < 0).any()):
            raise EvaluatorContractError("accepted profiles must contain every dealt seat")
        if not bool(accepted.all()):
            failed_mask = (~accepted)[:, :, None].expand_as(combo_ids)
            if bool((combo_ids[failed_mask] != -1).any()):
                raise EvaluatorContractError(
                    "failed sample_joint_profiles rows must contain only -1"
                )
            failed = (~accepted).nonzero(as_tuple=False).to(torch.int32).contiguous()
            raise AcceptanceError(failed, acceptance_rate, attempt_count, metadata)
        return combo_ids


class PokerGPUShowdownEvaluator:
    """Adapter for exact fixed-showdown winner masks and shares."""

    def __init__(self, *, backend: str = "auto", rank_count: int = 13) -> None:
        if backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
        self.backend = backend
        if isinstance(rank_count, bool) or not 4 <= rank_count <= 13:
            raise ValueError("rank_count must be an integer in [4,13]")
        self.rank_count = int(rank_count)
        self._evaluate: Callable[..., Any] | None = None

    def _operation(self) -> Callable[..., Any]:
        if self._evaluate is None:
            self._evaluate = _load_operation("evaluate_showdowns")
        return self._evaluate

    def evaluate(
        self,
        *,
        board_cards: Tensor,
        combo_ids: Tensor,
        active_mask: Tensor,
        stratum_players: Tensor,
    ) -> ShowdownResult:
        board = _normalize_board(board_cards)
        if bool((board == PAD_CARD).any()):
            raise ValueError("fixed showdown requires a final five-card board")
        if combo_ids.dtype != torch.int32 or combo_ids.ndim != 2:
            raise TypeError("combo_ids must be int32 [B,P]")
        batch_size, p_max = combo_ids.shape
        if active_mask.shape != (batch_size, p_max) or active_mask.dtype != torch.bool:
            raise TypeError("active_mask must be bool [B,P]")
        if stratum_players.ndim != 3 or stratum_players.shape[0] != batch_size or stratum_players.shape[2] != p_max:
            raise ValueError("stratum_players must have shape [B,K,P]")
        if stratum_players.dtype != torch.bool:
            raise TypeError("stratum_players must have dtype torch.bool")
        tensors = (board, combo_ids, active_mask, stratum_players)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all showdown tensors must share a device")
        if bool((stratum_players & ~active_mask[:, None, :]).any()):
            raise ValueError("every stratum-eligible player must be active")
        result = self._operation()(
            rank_count=self.rank_count,
            board_cards=board,
            combo_ids=combo_ids.contiguous(),
            active_mask=active_mask.contiguous(),
            stratum_players=stratum_players.contiguous(),
            backend=self.backend,
        )
        k = stratum_players.shape[1]
        op = "evaluate_showdowns"
        winner_mask = _require_tensor(result, "winner_mask", (batch_size, k), torch.int32, board.device, op)
        shares = _require_tensor(result, "showdown_share", (batch_size, k, p_max), torch.float32, board.device, op)
        metadata = _validate_metadata(getattr(result, "metadata", None), op)
        if bool((shares < 0).any()) or not bool(torch.isfinite(shares).all()):
            raise EvaluatorContractError("showdown shares must be finite and nonnegative")
        if bool((shares.masked_select(~stratum_players).abs() > 1e-6).any()):
            raise EvaluatorContractError("ineligible seats must have zero showdown share")
        if bool((torch.abs(shares.sum(dim=-1) - 1.0) > 1e-5).any()):
            raise EvaluatorContractError("showdown shares must sum to one per stratum")
        seat_bits = torch.bitwise_left_shift(
            torch.ones((p_max,), dtype=torch.int32, device=board.device),
            torch.arange(p_max, dtype=torch.int32, device=board.device),
        )
        mask_from_shares = ((shares > 0).to(torch.int32) * seat_bits).sum(dim=-1)
        if not bool(torch.equal(mask_from_shares, winner_mask)):
            raise EvaluatorContractError("winner_mask and positive showdown shares disagree")
        winners = (shares > 0).sum(dim=-1).to(torch.float32)
        expected = torch.where(shares > 0, winners.reciprocal()[:, :, None], torch.zeros_like(shares))
        if bool((torch.abs(shares - expected) > 1e-6).any()):
            raise EvaluatorContractError("showdown shares must split each winner mask equally")
        return ShowdownResult(winner_mask, shares, metadata)
