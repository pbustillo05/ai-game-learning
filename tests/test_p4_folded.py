from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


@dataclass(frozen=True)
class _Info:
    contract_version: str = "1"
    package_version: str = "test"
    build_revision: str = "fake"
    rank_count: int = 13
    deck_geometry_version: str = "global-suffix-lex-v1"
    category_order_version: str = "cumulative-five-card-v1"
    folder_feature_version: str = "compact-folder-v1"
    cpu_rng_version: str = "cpu-test"
    cuda_rng_version: str = "cuda-test"
    mps_rng_version: str = "mps-test"


class _FakeEvaluator:
    def __init__(self, calls: dict[str, Any]) -> None:
        self.calls = calls
        self.info = _Info()
        self.combo_cards = torch.triu_indices(52, 52, offset=1).T.to(torch.uint8)

    def summarize_folders(self, **kwargs: Any) -> Any:
        self.calls.setdefault("all", []).append(kwargs)
        board = kwargs["board_cards"]
        batch = board.shape[0]
        folders = kwargs["folder_weights"].shape[1]
        rank = kwargs["beta_rank"]
        device = board.device
        result = SimpleNamespace(
            hand_log_ratio=torch.arange(
                batch * 1326, dtype=torch.float32, device=device
            ).reshape(batch, 1326),
            beta_eigenvalue=torch.zeros((batch, rank), device=device),
            beta_eigenvector=torch.zeros((batch, 52, rank), device=device),
            folder_log_mass=torch.zeros((batch, folders), device=device),
            folder_card_mass=torch.zeros((batch, folders, 52), device=device),
            tight_cluster_scalar=torch.zeros(batch, device=device),
        )
        self.calls.setdefault("_results", []).append(result)
        return result


def _install_fake(
    monkeypatch: pytest.MonkeyPatch, *, version: str = "1", bad_combo: bool = False
) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    package = ModuleType("poker_gpu_evaluator")
    package.CONTRACT_VERSION = version  # type: ignore[attr-defined]

    def bind(*, rank_count: int) -> _FakeEvaluator:
        evaluator = _FakeEvaluator(calls)
        if bad_combo:
            evaluator.combo_cards[0] = torch.tensor([0, 2], dtype=torch.uint8)
        return evaluator

    package.bind = bind  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator", package)
    return calls


def _inputs(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, Any]:
    from pbg.games.poker.folded import FoldSnapshotBatch

    board = torch.tensor(
        [[0, 5, 9, -1, -1], [4, 8, 12, 16, 20]], dtype=torch.int16, device=device
    )
    live = torch.ones((2, 2, 1326), dtype=torch.float32, device=device)
    folded = FoldSnapshotBatch(
        reach_weights=torch.ones((2, 3, 1326), dtype=torch.float32, device=device),
        board_at_fold=torch.tensor(
            [
                [[0, 5, 9, 255, 255]] * 3,
                [[4, 8, 12, 255, 255], [4, 8, 12, 16, 255], [4, 8, 12, 16, 20]],
            ],
            dtype=torch.uint8,
            device=device,
        ),
        line_meta=torch.arange(
            2 * 3 * 4, dtype=torch.float32, device=device
        ).reshape(2, 3, 4),
        folder_mask=torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool, device=device
        ),
    )
    return board, live, folded


def test_import_is_lazy_and_maps_compact_folder_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "poker_gpu_evaluator", raising=False)
    module = importlib.reload(importlib.import_module("pbg.games.poker.folded"))
    assert "poker_gpu_evaluator" not in sys.modules
    calls = _install_fake(monkeypatch)
    board, live, folded = _inputs()
    output = module.PokerGPUPartitionAdapter().summarize(
        board_cards=board, live_range_weights=live, folded=folded
    )
    [call] = calls["all"]
    assert set(call) == {"board_cards", "folder_weights", "row_mask", "beta_rank"}
    assert call["board_cards"].tolist() == [[0, 5, 9, 255, 255], [4, 8, 12, 16, 20]]
    assert torch.equal(call["folder_weights"], folded.reach_weights)
    assert torch.equal(call["row_mask"], folded.folder_mask)
    assert set(output.fields) == {
        "hand_log_ratio", "beta_eigenvalue", "beta_eigenvector",
        "folder_log_mass", "folder_card_mass", "tight_cluster_scalar",
        "folder_mask", "folder_line_meta", "folder_seat",
    }
    assert output.fields["hand_log_ratio"].shape == (2, 1326)
    assert output.fields["folder_card_mass"].shape == (2, 3, 52)
    assert output.metadata["contract_version"] == "1"


def test_no_folder_summary_is_neutral_and_skips_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake(monkeypatch)
    from pbg.games.poker.folded import FoldSnapshotBatch, PokerGPUPartitionAdapter

    folded = FoldSnapshotBatch(
        reach_weights=torch.empty((2, 0, 1326), dtype=torch.float32),
        board_at_fold=torch.empty((2, 0, 5), dtype=torch.uint8),
        line_meta=torch.empty((2, 0, 8), dtype=torch.float32),
        folder_mask=torch.empty((2, 0), dtype=torch.bool),
    )
    output = PokerGPUPartitionAdapter().summarize(
        board_cards=torch.full((2, 5), -1, dtype=torch.int16),
        live_range_weights=torch.rand(2, 3, 1326),
        folded=folded,
    )
    assert "all" not in calls
    assert torch.count_nonzero(output.fields["hand_log_ratio"]) == 0
    assert output.fields["folder_card_mass"].shape == (2, 0, 52)


def test_version_result_and_combo_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pbg.games.poker.folded import PokerGPUPartitionAdapter

    board, live, folded = _inputs()
    _install_fake(monkeypatch, version="0")
    with pytest.raises(RuntimeError, match="expected CONTRACT_VERSION '1'"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board, live_range_weights=live, folded=folded
        )

    _install_fake(monkeypatch, bad_combo=True)
    with pytest.raises(RuntimeError, match="combo_cards is incompatible"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board, live_range_weights=live, folded=folded
        )

    calls = _install_fake(monkeypatch)
    original = _FakeEvaluator.summarize_folders

    def bad(self: _FakeEvaluator, **kwargs: Any) -> Any:
        result = original(self, **kwargs)
        result.hand_log_ratio = result.hand_log_ratio.double()
        return result

    monkeypatch.setattr(_FakeEvaluator, "summarize_folders", bad)
    with pytest.raises(TypeError, match="hand_log_ratio.*dtype"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board, live_range_weights=live, folded=folded
        )
    assert "all" in calls


def test_input_shape_and_dtype_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    from pbg.games.poker.folded import FoldSnapshotBatch, PokerGPUPartitionAdapter

    board, live, folded = _inputs()
    with pytest.raises(ValueError, match=r"\[B,5\]"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board[:, :4], live_range_weights=live, folded=folded
        )
    wrong = FoldSnapshotBatch(
        folded.reach_weights,
        folded.board_at_fold.to(torch.int16),
        folded.line_meta,
        folded.folder_mask,
    )
    with pytest.raises(TypeError, match="board_at_fold"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board, live_range_weights=live, folded=wrong
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mapping_has_no_cpu_or_numpy_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    from pbg.games.poker.folded import PokerGPUPartitionAdapter

    board, live, folded = _inputs("cuda")
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(torch.Tensor, "numpy", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    output = PokerGPUPartitionAdapter().summarize(
        board_cards=board, live_range_weights=live, folded=folded
    )
    assert all(value.device.type == "cuda" for value in output.fields.values())


def test_real_sibling_folder_smoke() -> None:
    package = pytest.importorskip("poker_gpu_evaluator")
    assert package.CONTRACT_VERSION == "2"
    evaluator = package.bind(rank_count=13)
    assert callable(evaluator.summarize_folders)
