from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from pbg.core.devices import resolve_execution_device, validate_workload_capabilities
from pbg.benchmark import benchmark_modes
from pbg.config import load_config
from pbg.eval.br_probe import evaluate_br_probe
from pbg.games.poker.bundle import build_bundle, build_model_from_config
from pbg.inference import PokerInferenceSession
from pbg.runtime import train


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("value", ["cpu", "auto", torch.device("cpu")])
def test_every_runtime_contract_rejects_cpu_and_auto(value) -> None:
    with pytest.raises(ValueError, match="cuda.*mps"):
        resolve_execution_device(value, workload="test workflow")


def test_cuda_unavailability_never_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never falls back to CPU"):
        resolve_execution_device("cuda", workload="training")


def test_mps_training_rejected_before_construction() -> None:
    with pytest.raises(RuntimeError, match="evaluator contract v2"):
        validate_workload_capabilities(torch.device("mps"), workload="training")


def test_inference_requires_an_explicit_device() -> None:
    with pytest.raises(TypeError):
        PokerInferenceSession.from_checkpoint("unused.pt")


def test_every_public_python_workflow_rejects_cpu_before_construction(tmp_path) -> None:
    cfg = load_config(ROOT / "configs" / "train" / "nlhe6.yaml")
    scientific = {"game": cfg.game.__dict__, "model": cfg.model}
    with pytest.raises(ValueError, match="production execution device"):
        build_bundle(cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="production execution device"):
        build_model_from_config(scientific, torch.device("cpu"))
    with pytest.raises(ValueError, match="production execution device"):
        benchmark_modes(cfg, ("normal",), warmup=0, repeats=1, device="cpu")
    cfg.device = "cpu"
    with pytest.raises(ValueError, match="production execution device"):
        train(cfg)
    with pytest.raises(ValueError, match="production execution device"):
        PokerInferenceSession({}, device="cpu")
    with pytest.raises(ValueError, match="production execution device"):
        evaluate_br_probe(tmp_path / "absent.pt", n_hands=4, device="cpu")


def test_production_sources_cannot_reach_deleted_legacy_types() -> None:
    banned = {"NLHEEngine", "PokerGame", "RangeState"}
    for path in (ROOT / "pbg").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not names & banned, f"{path} reaches {sorted(names & banned)}"
    for relative in (
        "pbg/games/poker/engine.py",
        "pbg/games/poker/ranges.py",
        "pbg/eval/arena.py",
    ):
        assert not (ROOT / relative).exists()


def test_public_clis_do_not_advertise_cpu_or_auto_devices() -> None:
    for path in (ROOT / "pbg" / "cli").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not any(
                isinstance(arg, ast.Constant) and arg.value == "--device"
                for arg in call.args
            ):
                continue
            for keyword in call.keywords:
                if keyword.arg == "choices":
                    advertised = {
                        item.value
                        for item in ast.walk(keyword.value)
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    }
                    assert not advertised & {"cpu", "auto"}, (path, advertised)
