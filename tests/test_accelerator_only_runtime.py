from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pbg.config import load_config
from pbg.devices import resolve_execution_device, validate_workload_capabilities
from pbg.inference import PokerInferenceSession
from pbg.runtime import TrainingSession


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/train/nlhe_hu_8bb_normal_queued_smoke.yaml"


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


def test_public_training_rejects_cpu_before_construction() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="training requires CUDA"):
        replace(config.execution, device="cpu")


def test_public_roots_do_not_import_old_core() -> None:
    for relative in (
        "pbg/config.py",
        "pbg/runtime.py",
        "pbg/benchmark.py",
        "pbg/inference.py",
        "pbg/inference_artifact.py",
        "pbg/eval/br_probe.py",
    ):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(
            value == "pbg.core" or value.startswith("pbg.core.")
            for value in imports
        ), relative


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
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    }
                    assert not advertised & {"cpu", "auto"}, (path, advertised)
