from __future__ import annotations

from pathlib import Path

from pbg.cli import rao_blackwell_variance as cli
from pbg.eval.rao_blackwell_variance import load_experiment_config


CONFIG = Path("configs/experiments/nlhe_hu_32bb_rao_blackwell_variance.yaml")


def test_scientific_config_defaults() -> None:
    cfg = load_experiment_config(CONFIG)
    assert cfg["checkpoints"] == [1110]
    assert cfg["candidates"]["N"] == [384, 768, 1536, 3072]
    assert cfg["oracle"]["initial_samples"] == 8192


def test_torchrun_local_rank_initializes_its_cuda_device(monkeypatch) -> None:
    selected = []
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        cli,
        "resolve_execution_device",
        lambda *_args, **_kwargs: cli.torch.device("cuda", 1),
    )
    monkeypatch.setattr(cli.torch.cuda, "set_device", selected.append)
    assert cli._device("cuda") == cli.torch.device("cuda", 1)
    assert selected == [cli.torch.device("cuda", 1)]
