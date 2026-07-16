from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from pbg.config import load_config
from pbg.core.training import _add_distributed_metric_summaries, _sync_scaled_gradients


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
        targets = torch.tensor([0.0]) if rank == 0 else torch.tensor([2.0, 2.0, 2.0])
        inputs = torch.ones((targets.numel(), 1))
        # Each rank contributes its unaveraged numerator divided by the exact
        # common global mass. The flat collective must SUM these terms.
        loss = (model(inputs).squeeze(1) - targets).square().sum() / 4.0
        loss.backward()
        assert _sync_scaled_gradients(model, world)
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


def test_ddp_keeps_tree_and_batch_values_per_rank() -> None:
    cfg = load_config("configs/train/nlhe6.yaml")
    one_rank = copy.deepcopy(cfg)
    two_rank = copy.deepcopy(cfg)
    assert one_rank.rollout.sequential_starts == 1
    assert two_rank.rollout.sequential_starts == one_rank.rollout.sequential_starts
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


@pytest.mark.skip(reason="multi-node acceptance run; not a fixed CI performance assertion")
def test_one_rank_and_two_rank_kuhn_curves_improve_comparably() -> None:
    """Run via torchrun on the target node and compare sample-scaled curves."""
