from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from pbg.config import load_config
from pbg.core.optimization import (
    FlatGradientBuffer,
    finite_gated_optimizer_step,
    sync_scaled_gradients,
)
from pbg.core.training import (
    _add_distributed_metric_summaries,
    _scaler,
)


def _asymmetric_metric_worker(
    rank: int, world: int, rendezvous: str, output_dir: str
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
    )
    try:
        metrics = {"nodes": float(2 + 4 * rank), "loss": float(rank + 1)}
        if rank == 1:
            metrics.update(
                {
                    "completion_mean_estimator_stddev": 0.25,
                    "completion_max_estimator_stddev": 0.5,
                }
            )
        _add_distributed_metric_summaries(metrics, torch.device("cpu"), world)
        Path(output_dir, f"rank_{rank}.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def _unequal_mass_gradient_worker(
    rank: int, world: int, rendezvous: str, output_dir: str
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        model = torch.nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        flat = FlatGradientBuffer(model)
        flat.zero_()
        targets = torch.tensor([0.0]) if rank == 0 else torch.tensor([2.0, 2.0, 2.0])
        inputs = torch.ones((targets.numel(), 1))
        # Each rank contributes its unaveraged numerator divided by the exact
        # common global mass. The flat collective must SUM these terms.
        loss = (model(inputs).squeeze(1) - targets).square().sum() / 4.0
        loss.backward()
        assert sync_scaled_gradients(model, world, flat_buffer=flat)
        Path(output_dir, f"gradient_{rank}.json").write_text(
            json.dumps(
                {
                    "weight": float(model.weight.grad.squeeze()),
                    "bias": float(model.bias.grad.squeeze()),
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def _unequal_nonfinite_gradient_worker(
    rank: int, world: int, rendezvous: str, output_dir: str
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3.0)
        flat = FlatGradientBuffer(model)
        flat.zero_()
        model.weight.grad.fill_(1.0 if rank == 0 else float("inf"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        scaler = _scaler(False)
        common_finite = sync_scaled_gradients(model, world, flat_buffer=flat)
        result = finite_gated_optimizer_step(
            model,
            optimizer,
            scaler,
            gradients_finite=common_finite,
            grad_clip=1.0,
        )
        Path(output_dir, f"nonfinite_{rank}.json").write_text(
            json.dumps(
                {
                    "common_finite": common_finite,
                    "step_applied": result.optimizer_step_applied,
                    "weight": float(model.weight),
                    "scale": float(scaler.get_scale()),
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def test_ddp_keeps_tree_and_batch_values_per_rank() -> None:
    cfg = load_config("configs/train/nlhe6.yaml")
    one_rank = copy.deepcopy(cfg)
    two_rank = copy.deepcopy(cfg)
    assert one_rank.rollout.roots_per_gpu == 1
    assert two_rank.rollout.roots_per_gpu == one_rank.rollout.roots_per_gpu
    assert two_rank.training.minibatch_size == one_rank.training.minibatch_size


def test_distributed_metric_summaries_allow_rank_local_optional_metrics(
    tmp_path,
) -> None:
    world = 2
    torch.multiprocessing.spawn(
        _asymmetric_metric_worker,
        args=(world, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=world,
        join=True,
    )

    for rank in range(world):
        metrics = json.loads((tmp_path / f"rank_{rank}.json").read_text())
        assert metrics["nodes_rank_mean"] == pytest.approx(4.0)
        assert metrics["nodes_rank_max"] == pytest.approx(6.0)
        assert metrics["loss_rank_mean"] == pytest.approx(1.5)
        assert metrics["loss_rank_max"] == pytest.approx(2.0)
        assert metrics["completion_mean_estimator_stddev_rank_mean"] == pytest.approx(
            0.25
        )
        assert metrics["completion_mean_estimator_stddev_rank_max"] == pytest.approx(
            0.25
        )
        assert metrics["completion_max_estimator_stddev_rank_mean"] == pytest.approx(
            0.5
        )
        assert metrics["completion_max_estimator_stddev_rank_max"] == pytest.approx(
            0.5
        )


def test_unequal_rank_flat_sum_matches_concatenated_rollout_oracle(tmp_path) -> None:
    world = 2
    torch.multiprocessing.spawn(
        _unequal_mass_gradient_worker,
        args=(world, str(tmp_path / "gradient_rendezvous"), str(tmp_path)),
        nprocs=world,
        join=True,
    )
    oracle = torch.nn.Linear(1, 1, bias=True)
    with torch.no_grad():
        oracle.weight.zero_()
        oracle.bias.zero_()
    targets = torch.tensor([0.0, 2.0, 2.0, 2.0])
    loss = (oracle(torch.ones((4, 1))).squeeze(1) - targets).square().mean()
    loss.backward()

    for rank in range(world):
        reduced = json.loads((tmp_path / f"gradient_{rank}.json").read_text())
        assert reduced["weight"] == pytest.approx(float(oracle.weight.grad.squeeze()), abs=0)
        assert reduced["bias"] == pytest.approx(float(oracle.bias.grad.squeeze()), abs=0)


def test_unequal_rank_nonfinite_decision_skips_every_optimizer(tmp_path) -> None:
    world = 2
    torch.multiprocessing.spawn(
        _unequal_nonfinite_gradient_worker,
        args=(world, str(tmp_path / "nonfinite_rendezvous"), str(tmp_path)),
        nprocs=world,
        join=True,
    )
    for rank in range(world):
        record = json.loads((tmp_path / f"nonfinite_{rank}.json").read_text())
        assert not record["common_finite"]
        assert not record["step_applied"]
        assert record["weight"] == 3.0
        assert record["scale"] == 1.0
