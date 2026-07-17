from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket

import pytest
import torch

from pbg.config import load_config
from pbg.runtime import train


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("mode", "config_name", "roots"),
    (
        ("normal", "nlhe_hu_normal.yaml", 2),
        ("rao_blackwell", "nlhe_hu_sequential_rb.yaml", 2),
        ("branched_rao_blackwell", "nlhe_hu_branched_rb.yaml", 1),
    ),
)
def test_tiny_gpu_training_modes_report_bounded_phase_memory(
    tmp_path: Path, mode: str, config_name: str, roots: int
) -> None:
    """One real learner update per mode with deliberately tiny geometry."""

    cfg = load_config(
        Path("configs/acceptance") / config_name,
        hardware_profile="local_3x2080ti",
    )
    cfg.rollout.sequential_roots_per_gpu_initial = roots
    cfg.rollout.sequential_roots_per_gpu_minimum = roots
    cfg.rollout.sequential_roots_per_gpu_maximum = roots
    cfg.rollout.roots_per_gpu = roots
    cfg.rollout.node_budget = 64
    cfg.rollout.family_batch_rows = 2
    cfg.rollout.max_active_trajectories = roots
    cfg.rollout.arena_chunk_nodes = 8
    if mode != "normal":
        cfg.completion.n_samples = 4
        cfg.completion.backend = "cuda"
    cfg.model = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 4,
    }
    cfg.training.iters = 1
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 2
    cfg.training.checkpoint_every = 1
    cfg.training.out_dir = str(tmp_path / mode)
    cfg.training.amp = False

    train(cfg)

    record = json.loads(
        (Path(cfg.training.out_dir) / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert math.isfinite(record["loss"])
    assert record["trajectories"] == roots
    assert record["retained_target_bytes"] > 0
    assert record["iteration_peak_reserved_bytes"] > 0
    for phase in ("rollout", "optimization"):
        assert record[f"phase_{phase}_peak_reserved_bytes"] > 0
    print(
        json.dumps(
            {
                "mode": mode,
                "retained_target_bytes": record["retained_target_bytes"],
                "iteration_peak_allocated_bytes": record[
                    "iteration_peak_allocated_bytes"
                ],
                "iteration_peak_reserved_bytes": record[
                    "iteration_peak_reserved_bytes"
                ],
                "rollout_peak_reserved_bytes": record[
                    "phase_rollout_peak_reserved_bytes"
                ],
                "optimization_peak_reserved_bytes": record[
                    "phase_optimization_peak_reserved_bytes"
                ],
            },
            sort_keys=True,
        )
    )


def _two_gpu_worker(rank: int, world: int, port: int, output_dir: str) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world),
            "LOCAL_RANK": str(rank),
            "NCCL_SOCKET_IFNAME": "lo",
        }
    )
    cfg = load_config(
        "configs/acceptance/nlhe_hu_normal.yaml",
        hardware_profile="local_3x2080ti",
    )
    cfg.rollout.sequential_roots_per_gpu_initial = 1
    cfg.rollout.sequential_roots_per_gpu_minimum = 1
    cfg.rollout.sequential_roots_per_gpu_maximum = 1
    cfg.rollout.roots_per_gpu = 1
    cfg.rollout.node_budget = 32
    cfg.rollout.family_batch_rows = 1
    cfg.rollout.max_active_trajectories = 1
    cfg.rollout.arena_chunk_nodes = 8
    cfg.model = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 4,
    }
    cfg.training.iters = 1
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 4
    cfg.training.checkpoint_every = 1
    cfg.training.out_dir = output_dir
    cfg.training.amp = False
    train(cfg)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs unavailable")
def test_tiny_two_gpu_common_width_and_collective_update(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    output_dir = str(tmp_path / "two_gpu")
    torch.multiprocessing.spawn(
        _two_gpu_worker,
        args=(2, port, output_dir),
        nprocs=2,
        join=True,
    )
    record = json.loads(
        (Path(output_dir) / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert record["trajectories_rank_sum"] == 2
    assert record["sequential_roots_per_gpu"] == 1
