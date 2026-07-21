from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket

import pytest
import torch

from pbg.config import load_config
from pbg.core.config import AdaptiveKKTConfig
from pbg.runtime import train


def _tiny_config(mode: str, *, branching_mode: str = "fixed"):
    cfg = load_config(
        "configs/nlhe6_h100.yaml",
        hardware_profile="local_3x2080ti",
    )
    cfg.game.n_players = 2
    cfg.game.stack_bb = 8.0
    cfg.game.rank_count = 4
    cfg.training.mode = mode
    if mode == "normal":
        cfg.rollout.branching_mode = None
        cfg.rollout.beta = None
        cfg.completion.n_samples = None
        cfg.loss.name = "ppo"
        cfg.loss.c_t = 0.0
        cfg.loss.c_c = 0.0
    elif mode == "rao_blackwell":
        cfg.rollout.branching_mode = None
        cfg.rollout.beta = None
        cfg.loss.name = "rb_ppo"
        cfg.loss.c_t = 0.0
        cfg.loss.c_c = 0.0
    else:
        cfg.rollout.branching_mode = branching_mode
        cfg.rollout.beta = {"default": 1.0}
        cfg.rollout.adaptive_kkt = (
            None
            if branching_mode == "fixed"
            else AdaptiveKKTConfig(
                beta_max=1.0,
                warmup_iters=0,
                target_tree_size=4.0,
                safe_tree_size=16.0,
                kappa_initial=1.0,
                kappa_min=0.1,
                kappa_max=10.0,
                gamma_kappa=0.01,
                eta_kappa=0.1,
            )
        )
        cfg.loss.name = "rb_ppo"
        cfg.loss.c_t = 1.0
        cfg.loss.c_c = 1.0
    return cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("case_name", "mode", "branching_mode", "roots"),
    (
        ("normal", "normal", "fixed", 2),
        ("sequential_rb", "rao_blackwell", "fixed", 2),
        ("fixed_branched_rb", "branched_rao_blackwell", "fixed", 1),
        ("dynamic_branched_rb", "branched_rao_blackwell", "dynamic", 1),
    ),
)
def test_tiny_gpu_training_modes_report_bounded_phase_memory(
    tmp_path: Path,
    case_name: str,
    mode: str,
    branching_mode: str,
    roots: int,
) -> None:
    """One real learner update per mode with deliberately tiny geometry."""

    cfg = _tiny_config(mode, branching_mode=branching_mode)
    cfg.rollout.sequential_roots_per_gpu_initial = roots
    cfg.rollout.sequential_roots_per_gpu_minimum = roots
    cfg.rollout.sequential_roots_per_gpu_maximum = roots
    cfg.rollout.roots_per_gpu = roots
    cfg.rollout.family_batch_rows = 2
    cfg.rollout.max_active_trajectories = roots
    cfg.rollout.storage_page_rows = 8
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
        "use_branching_heads": mode == "branched_rao_blackwell",
    }
    cfg.training.iters = 1
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 2
    cfg.training.checkpoint_every = 1
    cfg.training.out_dir = str(tmp_path / case_name)
    cfg.training.amp = False

    train(cfg)

    record = json.loads(
        (Path(cfg.training.out_dir) / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert math.isfinite(record["loss"])
    assert record["trajectories"] == roots
    assert record["retained_target_bytes"] > 0
    assert record["iteration_peak_reserved_bytes"] > 0
    for phase in ("rollout", "seal", "targets", "optimization"):
        assert record[f"phase_{phase}_peak_reserved_bytes"] > 0
        planned_phase = "target" if phase == "targets" else phase
        assert record[f"memory_plan_{planned_phase}_phase_bytes"] > 0
    assert record["seal_source_pages"] > 0
    assert record["seal_source_capacity_bytes"] > 0
    assert record["seal_destination_capacity_bytes"] > 0
    assert record["rollout_storage_bytes"] > 0
    if mode == "branched_rao_blackwell":
        assert record["actions"] > 0
    else:
        assert record["actions"] == 0
    print(
        json.dumps(
            {
                "mode": case_name,
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
                "seal_peak_reserved_bytes": record[
                    "phase_seal_peak_reserved_bytes"
                ],
                "target_peak_reserved_bytes": record[
                    "phase_targets_peak_reserved_bytes"
                ],
                "optimization_peak_reserved_bytes": record[
                    "phase_optimization_peak_reserved_bytes"
                ],
                "seal_source_pages": record["seal_source_pages"],
                "seal_source_capacity_bytes": record["seal_source_capacity_bytes"],
                "seal_destination_capacity_bytes": record[
                    "seal_destination_capacity_bytes"
                ],
                "seal_scratch_capacity_bytes": record[
                    "seal_scratch_capacity_bytes"
                ],
                "actions": record["actions"],
                "decisions_per_second": record["decisions_per_second"],
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ("normal", "rao_blackwell"))
def test_tiny_sequential_gpu_soak_has_stable_iteration_owners(
    tmp_path: Path, mode: str
) -> None:
    cfg = _tiny_config(mode)
    cfg.rollout.sequential_roots_per_gpu_initial = 2
    cfg.rollout.sequential_roots_per_gpu_minimum = 2
    cfg.rollout.sequential_roots_per_gpu_maximum = 2
    cfg.rollout.roots_per_gpu = 2
    cfg.rollout.family_batch_rows = 2
    cfg.rollout.max_active_trajectories = 2
    cfg.rollout.storage_page_rows = 8
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
        "use_branching_heads": False,
    }
    cfg.training.iters = 3
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 2
    cfg.training.checkpoint_every = 3
    cfg.training.out_dir = str(tmp_path / f"{mode}_soak")
    cfg.training.amp = False
    train(cfg)

    records = [
        json.loads(line)
        for line in (Path(cfg.training.out_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3
    allocated = [record["iteration_end_allocated_bytes"] for record in records]
    assert max(allocated) - min(allocated) <= 4 * 1024**2
    assert all(record["memory_hard_limit_exceeded"] == 0 for record in records)
    assert all(record["seal_source_pages"] > 0 for record in records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("case", ("fixed_refill", "dynamic_safeguard"))
def test_tiny_branched_gpu_safety_soak_exercises_scientific_controls(
    tmp_path: Path, case: str
) -> None:
    branching_mode = "fixed" if case == "fixed_refill" else "dynamic"
    cfg = _tiny_config("branched_rao_blackwell", branching_mode=branching_mode)
    cfg.rollout.roots_per_gpu = 1
    cfg.rollout.family_batch_rows = 2
    cfg.rollout.max_active_trajectories = 2
    cfg.rollout.storage_page_rows = 8
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
        "use_branching_heads": True,
    }
    if case == "fixed_refill":
        cfg.rollout.branching_min_nodes = 32
    else:
        assert cfg.rollout.adaptive_kkt is not None
        cfg.rollout.adaptive_kkt.beta_max = 2.0
        cfg.rollout.adaptive_kkt.safe_tree_size = 1.0
        cfg.rollout.adaptive_kkt.kappa_initial = 10.0
        cfg.rollout.adaptive_kkt.kappa_max = 10.0
    cfg.training.iters = 2
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 2
    cfg.training.checkpoint_every = 2
    cfg.training.out_dir = str(tmp_path / case)
    cfg.training.amp = False
    train(cfg)

    records = [
        json.loads(line)
        for line in (Path(cfg.training.out_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert all(record["memory_hard_limit_exceeded"] == 0 for record in records)
    assert all(record["actions"] > 0 for record in records)
    if case == "fixed_refill":
        assert any(record.get("root_refill_count", 0) > 0 for record in records)
    else:
        assert any(record.get("safeguard_batches", 0) > 0 for record in records)
        assert any(
            record.get("safeguard_suppressed_children", 0) > 0
            for record in records
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tiny_gpu_pipeline_does_not_require_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _tiny_config("normal")
    cfg.rollout.sequential_roots_per_gpu_initial = 2
    cfg.rollout.sequential_roots_per_gpu_minimum = 2
    cfg.rollout.sequential_roots_per_gpu_maximum = 2
    cfg.rollout.roots_per_gpu = 2
    cfg.rollout.family_batch_rows = 2
    cfg.rollout.max_active_trajectories = 2
    cfg.rollout.storage_page_rows = 8
    cfg.model = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 4,
        "use_branching_heads": False,
    }
    cfg.training.iters = 1
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 2
    cfg.training.checkpoint_every = 1
    cfg.training.out_dir = str(tmp_path / "without_empty_cache")
    cfg.training.amp = False
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    train(cfg)
    record = json.loads(
        (Path(cfg.training.out_dir) / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert math.isfinite(record["loss"])
    assert record["memory_hard_limit_exceeded"] == 0


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
    cfg = _tiny_config("normal")
    cfg.rollout.sequential_roots_per_gpu_initial = 1
    cfg.rollout.sequential_roots_per_gpu_minimum = 1
    cfg.rollout.sequential_roots_per_gpu_maximum = 1
    cfg.rollout.roots_per_gpu = 1
    cfg.rollout.family_batch_rows = 1
    cfg.rollout.max_active_trajectories = 1
    cfg.rollout.storage_page_rows = 8
    cfg.model = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "m_raise": 1,
        "k_act": 4,
        "max_players": 2,
        "class_embedding_dim": 4,
        "use_branching_heads": False,
    }
    cfg.training.iters = 1
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 4
    cfg.training.checkpoint_every = 1
    cfg.training.out_dir = output_dir
    cfg.training.amp = False
    train(cfg)


def _two_gpu_dynamic_branched_worker(
    rank: int, world: int, port: int, output_dir: str
) -> None:
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
    cfg = _tiny_config("branched_rao_blackwell", branching_mode="dynamic")
    cfg.rollout.roots_per_gpu = 1
    cfg.rollout.family_batch_rows = 2
    cfg.rollout.max_active_trajectories = 2
    cfg.rollout.storage_page_rows = 8
    assert cfg.rollout.adaptive_kkt is not None
    cfg.rollout.adaptive_kkt.beta_max = 2.0
    cfg.rollout.adaptive_kkt.safe_tree_size = 1.0
    cfg.rollout.adaptive_kkt.kappa_initial = 10.0
    cfg.rollout.adaptive_kkt.kappa_max = 10.0
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
        "use_branching_heads": True,
    }
    cfg.training.iters = 1
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 2
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs unavailable")
def test_tiny_two_gpu_dynamic_branched_common_controller(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    output_dir = str(tmp_path / "two_gpu_dynamic_branched")
    torch.multiprocessing.spawn(
        _two_gpu_dynamic_branched_worker,
        args=(2, port, output_dir),
        nprocs=2,
        join=True,
    )
    record = json.loads(
        (Path(output_dir) / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert record["trajectories_rank_sum"] == 2
    assert record["safeguard_batches_rank_sum"] > 0
    assert record["safeguard_suppressed_children_rank_sum"] > 0
    assert record["memory_hard_limit_exceeded"] == 0
