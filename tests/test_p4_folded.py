from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest
import torch


@dataclass
class _FakeResult:
    base_log_z: torch.Tensor
    single_card_log_ratio: torch.Tensor
    pair_card_log_ratio: torch.Tensor | None
    output_valid: torch.Tensor
    metadata: dict[str, object]


def _install_fake(monkeypatch: pytest.MonkeyPatch, *, version: str = "pbg-v2") -> tuple[dict[str, Any], type[_FakeResult]]:
    calls: dict[str, Any] = {}
    package = ModuleType("poker_gpu_evaluator")
    package.__path__ = []  # type: ignore[attr-defined]
    facade = ModuleType("poker_gpu_evaluator.pbg")
    facade.API_VERSION = version  # type: ignore[attr-defined]
    facade.COMBO_CARDS = torch.triu_indices(52, 52, offset=1).T.to(torch.uint8)  # type: ignore[attr-defined]
    facade.PartitionSummaryResult = _FakeResult  # type: ignore[attr-defined]

    def partition_many(**kwargs: Any) -> _FakeResult:
        calls.setdefault("all", []).append(kwargs)
        board = kwargs["board_cards"]
        batch_size = board.shape[0]
        device = board.device
        result = _FakeResult(
            base_log_z=torch.arange(batch_size, dtype=torch.float32, device=device),
            single_card_log_ratio=torch.arange(
                batch_size * 52, dtype=torch.float32, device=device
            ).reshape(batch_size, 52),
            pair_card_log_ratio=torch.arange(
                batch_size * 52 * 52, dtype=torch.float32, device=device
            ).reshape(batch_size, 52, 52),
            output_valid=torch.ones(batch_size, dtype=torch.bool, device=device),
            metadata={"backend": "fake", "api_version": "pbg-v2"},
        )
        calls.setdefault("_results", []).append(result)
        return result

    facade.partition_many = partition_many  # type: ignore[attr-defined]
    package.pbg = facade  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator", package)
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator.pbg", facade)
    return calls, _FakeResult


def _inputs(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, Any]:
    from pbg.games.poker.folded import FoldSnapshotBatch

    board = torch.tensor(
        [[0, 5, 9, -1, -1], [4, 8, 12, 16, 20]],
        dtype=torch.int16,
        device=device,
    )
    live = torch.arange(2 * 2 * 1326, dtype=torch.float32, device=device).reshape(2, 2, 1326)
    folded = FoldSnapshotBatch(
        reach_weights=torch.arange(
            2 * 3 * 1326, dtype=torch.float32, device=device
        ).reshape(2, 3, 1326),
        board_at_fold=torch.tensor(
            [
                [[0, 5, 9, 255, 255], [0, 5, 9, 255, 255], [0, 5, 9, 255, 255]],
                [[4, 8, 12, 255, 255], [4, 8, 12, 16, 255], [4, 8, 12, 16, 20]],
            ],
            dtype=torch.uint8,
            device=device,
        ),
        line_meta=torch.arange(2 * 3 * 4, dtype=torch.float32, device=device).reshape(2, 3, 4),
        folder_mask=torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool, device=device
        ),
    )
    return board, live, folded


def test_import_is_lazy_and_exact_facade_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "poker_gpu_evaluator", raising=False)
    monkeypatch.delitem(sys.modules, "poker_gpu_evaluator.pbg", raising=False)
    folded_module = importlib.reload(importlib.import_module("pbg.games.poker.folded"))
    assert "poker_gpu_evaluator" not in sys.modules

    calls, result_type = _install_fake(monkeypatch)
    board, live, folded = _inputs()
    adapter = folded_module.PokerGPUPartitionAdapter()
    output = adapter.summarize(
        board_cards=board,
        live_range_weights=live,
        folded=folded,
    )

    sibling_result = calls["_results"][0]
    combined, per_folder = calls["all"]
    assert set(combined) == {"board_cards", "range_weights", "row_mask"}
    assert combined["board_cards"].dtype == torch.uint8
    assert combined["board_cards"].tolist() == [[0, 5, 9, 255, 255], [4, 8, 12, 16, 20]]
    assert combined["board_cards"].is_contiguous()
    assert combined["range_weights"].shape == (2, 3, 1326)
    assert torch.equal(combined["range_weights"], folded.reach_weights)
    assert combined["range_weights"].is_contiguous()
    assert combined["row_mask"].tolist() == folded.folder_mask.tolist()
    assert combined["row_mask"].is_contiguous()
    assert per_folder["range_weights"].shape == (3, 1, 1326)
    assert per_folder["row_mask"].all()
    assert isinstance(result_type(**{
        "base_log_z": output.fields["base_log_z"],
        "single_card_log_ratio": output.fields["single_card_log_ratio"],
        "pair_card_log_ratio": output.fields["pair_card_log_ratio"],
        "output_valid": output.fields["output_valid"],
        "metadata": output.metadata,
    }), result_type)
    assert set(output.fields) == {
        "base_log_z",
        "single_card_log_ratio",
        "pair_card_log_ratio",
        "output_valid",
        "folder_log_mass",
        "folder_single_card_log_ratio",
        "folder_pair_card_log_ratio",
        "folder_output_valid",
        "folder_mask",
        "folder_line_meta",
        "folder_seat",
    }
    assert output.fields["base_log_z"].shape == (2,)
    assert output.fields["single_card_log_ratio"].shape == (2, 52)
    assert output.fields["pair_card_log_ratio"].shape == (2, 52, 52)
    assert output.fields["output_valid"].dtype == torch.bool
    assert output.fields["base_log_z"] is sibling_result.base_log_z
    assert output.fields["single_card_log_ratio"] is sibling_result.single_card_log_ratio
    assert output.fields["pair_card_log_ratio"] is sibling_result.pair_card_log_ratio
    assert output.fields["output_valid"] is sibling_result.output_valid
    assert output.metadata is sibling_result.metadata


def test_no_folder_summary_is_exactly_neutral_and_skips_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _ = _install_fake(monkeypatch)
    from pbg.games.poker.folded import FoldSnapshotBatch, PokerGPUPartitionAdapter

    board = torch.full((2, 5), -1, dtype=torch.int16)
    folded = FoldSnapshotBatch(
        reach_weights=torch.empty((2, 0, 1326), dtype=torch.float32),
        board_at_fold=torch.empty((2, 0, 5), dtype=torch.uint8),
        line_meta=torch.empty((2, 0, 8), dtype=torch.float32),
        folder_mask=torch.empty((2, 0), dtype=torch.bool),
    )
    output = PokerGPUPartitionAdapter().summarize(
        board_cards=board,
        live_range_weights=torch.rand(2, 3, 1326),
        folded=folded,
    )
    assert "all" not in calls
    assert torch.count_nonzero(output.fields["single_card_log_ratio"]) == 0
    assert torch.count_nonzero(output.fields["pair_card_log_ratio"]) == 0
    assert output.fields["output_valid"].all()


def test_version_and_result_contract_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, version="pbg-v1")
    from pbg.games.poker.folded import PokerGPUPartitionAdapter

    board, live, folded = _inputs()
    with pytest.raises(RuntimeError, match="expected API_VERSION 'pbg-v2'"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board,
            live_range_weights=live,
            folded=folded,
        )

    calls, _ = _install_fake(monkeypatch)
    facade = sys.modules["poker_gpu_evaluator.pbg"]

    def bad_partition_many(**kwargs: Any) -> _FakeResult:
        calls.update(kwargs)
        return _FakeResult(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, 52),
            None,
            torch.ones(2, dtype=torch.bool),
            {},
        )

    facade.partition_many = bad_partition_many  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="base_log_z.*dtype"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board,
            live_range_weights=live,
            folded=folded,
        )


def test_input_shape_dtype_device_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    from pbg.games.poker.folded import FoldSnapshotBatch, PokerGPUPartitionAdapter

    board, live, folded = _inputs()
    adapter = PokerGPUPartitionAdapter()
    with pytest.raises(ValueError, match=r"\[B,5\]"):
        adapter.summarize(
            board_cards=board[:, :4], live_range_weights=live, folded=folded
        )
    wrong = FoldSnapshotBatch(
        folded.reach_weights,
        folded.board_at_fold.to(torch.int16),
        folded.line_meta,
        folded.folder_mask,
    )
    with pytest.raises(TypeError, match="board_at_fold"):
        adapter.summarize(
            board_cards=board, live_range_weights=live, folded=wrong
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mapping_has_no_cpu_or_numpy_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    from pbg.games.poker.folded import PokerGPUPartitionAdapter

    board, live, folded = _inputs("cuda")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adapter attempted a host round trip")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    output = PokerGPUPartitionAdapter().summarize(
        board_cards=board,
        live_range_weights=live,
        folded=folded,
    )
    assert all(value.device.type == "cuda" for value in output.fields.values())


def test_no_local_partition_or_private_assignment_math() -> None:
    from pathlib import Path

    path = Path(__file__).parents[1] / "pbg" / "games" / "poker" / "folded.py"
    source = path.read_text(encoding="utf-8")
    forbidden = (
        "Phi",
        "alpha_B",
        "beta_B",
        "lambda_g",
        "removal_set",
        "disjoint_assignment",
        "private_assignment",
        "import numpy",
        "evaluator.partition",
    )
    assert not any(symbol in source for symbol in forbidden)


def test_real_sibling_smoke() -> None:
    facade = pytest.importorskip("poker_gpu_evaluator.pbg")
    if facade.API_VERSION != "pbg-v2":
        pytest.skip("installed sibling does not expose the pbg-v2 facade")
    assert callable(facade.partition_many)
    expected = torch.triu_indices(52, 52, offset=1).T.to(torch.uint8)
    assert torch.equal(facade.COMBO_CARDS, expected)

    from pbg.games.poker.folded import FoldSnapshotBatch, PokerGPUPartitionAdapter

    folded = FoldSnapshotBatch(
        reach_weights=torch.empty((1, 0, 1326), dtype=torch.float32),
        board_at_fold=torch.empty((1, 0, 5), dtype=torch.uint8),
        line_meta=torch.empty((1, 0, 0), dtype=torch.float32),
        folder_mask=torch.empty((1, 0), dtype=torch.bool),
    )
    output = PokerGPUPartitionAdapter().summarize(
        board_cards=torch.full((1, 5), -1, dtype=torch.int16),
        live_range_weights=torch.ones((1, 1, 1326), dtype=torch.float32),
        folded=folded,
    )
    assert output.fields["output_valid"].tolist() == [True]


def test_incompatible_combo_table_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    facade = sys.modules["poker_gpu_evaluator.pbg"]
    facade.COMBO_CARDS[0] = torch.tensor([0, 2], dtype=torch.uint8)
    from pbg.games.poker.folded import PokerGPUPartitionAdapter

    board, live, folded = _inputs()
    with pytest.raises(RuntimeError, match="COMBO_CARDS is incompatible"):
        PokerGPUPartitionAdapter().summarize(
            board_cards=board, live_range_weights=live, folded=folded
        )
