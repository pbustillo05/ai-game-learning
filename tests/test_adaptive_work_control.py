from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from pbg.config import load_config
from pbg.core.training import SequentialWorkController


def _observe(
    controller: SequentialWorkController,
    decisions: int,
    *,
    target: int = 100,
    minimum: int = 5,
    maximum: int = 40,
    alpha: float = 0.5,
    change: float = 0.25,
) -> tuple[int, dict[str, float | str]]:
    return controller.observe_rollout(
        local_decisions=decisions,
        rank_max_decisions=decisions,
        rank_max_decisions_per_start=(
            decisions / controller.current_starts_per_gpu
        ),
        target_decisions=target,
        minimum_starts=minimum,
        maximum_starts=maximum,
        ema_alpha=alpha,
        maximum_change_fraction=change,
        target_reuse=3.0,
        minibatch_size=8,
    )


def test_adaptive_ema_upward_downward_and_step_math() -> None:
    controller = SequentialWorkController(10)
    steps, first = _observe(controller, 50)
    assert controller.ema_decisions_per_start == 5.0
    assert controller.current_starts_per_gpu == 12  # rate-limited toward 20
    assert first["sequential_start_clamp_reason"] == "change_up"
    assert steps == math.ceil(3.0 * 50 / 8)
    _, second = _observe(controller, 120)
    assert controller.ema_decisions_per_start == 7.5
    assert controller.current_starts_per_gpu == 13
    assert second["decisions_per_start_ema"] == 7.5


def test_adaptive_bounds_and_both_change_directions() -> None:
    low = SequentialWorkController(20)
    _, low_metrics = _observe(low, 2_000, minimum=18, maximum=40)
    assert low.current_starts_per_gpu == 18
    assert low_metrics["sequential_start_clamp_reason"] == "minimum"

    high = SequentialWorkController(20)
    _, high_metrics = _observe(high, 20, minimum=5, maximum=22)
    assert high.current_starts_per_gpu == 22
    assert high_metrics["sequential_start_clamp_reason"] == "maximum"

    downward = SequentialWorkController(20)
    _, down_metrics = _observe(downward, 400, minimum=5, maximum=40)
    assert downward.current_starts_per_gpu == 15
    assert down_metrics["sequential_start_clamp_reason"] == "change_down"


@pytest.mark.parametrize(
    ("local", "rank_max", "rate"),
    ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.0), (1, 1, float("nan"))),
)
def test_adaptive_rejects_zero_or_invalid_observations(
    local: int, rank_max: int, rate: float
) -> None:
    with pytest.raises(ValueError, match="positive decision counts"):
        SequentialWorkController(10).observe_rollout(
            local_decisions=local,
            rank_max_decisions=rank_max,
            rank_max_decisions_per_start=rate,
            target_decisions=100,
            minimum_starts=5,
            maximum_starts=40,
            ema_alpha=0.2,
            maximum_change_fraction=0.25,
            target_reuse=3.0,
            minibatch_size=8,
        )


def test_candidate_enables_complete_adaptive_contract_and_legacy_is_fixed() -> None:
    adaptive = load_config("configs/train/nlhe_hu_32bb_rb.yaml")
    assert adaptive.rollout.target_decisions_per_gpu == 8000
    assert adaptive.training.target_node_reuse == 3.0
    legacy = load_config("configs/acceptance/nlhe_hu_sequential_rb.yaml")
    assert legacy.rollout.target_decisions_per_gpu is None
    assert legacy.training.target_node_reuse is None


def _distributed_controller_worker(
    rank: int, world: int, rendezvous: str, output_dir: str
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        per_rank = ((40, 90, 55, 120), (60, 70, 100, 80))[rank]

        def run(
            controller: SequentialWorkController,
            generator: torch.Generator,
            first: int,
            stop: int,
        ) -> tuple[list[dict[str, float | str]], list[float]]:
            metrics = []
            draws = []
            for index in range(first, stop):
                local = per_rank[index]
                control = torch.tensor(
                    [local, local / controller.current_starts_per_gpu],
                    dtype=torch.float64,
                )
                dist.all_reduce(control, op=dist.ReduceOp.MAX)
                steps, record = controller.observe_rollout(
                    local_decisions=local,
                    rank_max_decisions=int(control[0]),
                    rank_max_decisions_per_start=float(control[1]),
                    target_decisions=80,
                    minimum_starts=4,
                    maximum_starts=20,
                    ema_alpha=0.25,
                    maximum_change_fraction=0.25,
                    target_reuse=3.0,
                    minibatch_size=8,
                )
                record["steps"] = float(steps)
                metrics.append(record)
                draws.append(float(torch.rand((), generator=generator)))
            return metrics, draws

        full_controller = SequentialWorkController(8)
        full_gen = torch.Generator().manual_seed(100 + rank)
        full_metrics, full_draws = run(full_controller, full_gen, 0, 4)

        resumed_controller = SequentialWorkController(8)
        resumed_gen = torch.Generator().manual_seed(100 + rank)
        prefix_metrics, prefix_draws = run(resumed_controller, resumed_gen, 0, 2)
        saved_controller = resumed_controller.state_dict()
        saved_rng = resumed_gen.get_state()
        resumed_controller = SequentialWorkController.from_state_dict(saved_controller)
        resumed_gen = torch.Generator()
        resumed_gen.set_state(saved_rng)
        suffix_metrics, suffix_draws = run(resumed_controller, resumed_gen, 2, 4)

        Path(output_dir, f"adaptive_rank_{rank}.json").write_text(
            json.dumps(
                {
                    "full_metrics": full_metrics,
                    "resumed_metrics": prefix_metrics + suffix_metrics,
                    "full_draws": full_draws,
                    "resumed_draws": prefix_draws + suffix_draws,
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_controller_agrees_and_resume_preserves_metrics_and_rng(
    tmp_path: Path,
) -> None:
    world = 2
    torch.multiprocessing.spawn(
        _distributed_controller_worker,
        args=(world, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=world,
        join=True,
    )
    rows = [
        json.loads((tmp_path / f"adaptive_rank_{rank}.json").read_text())
        for rank in range(world)
    ]
    for row in rows:
        assert row["full_metrics"] == row["resumed_metrics"]
        assert row["full_draws"] == row["resumed_draws"]
    for iteration in range(4):
        assert rows[0]["full_metrics"][iteration]["steps"] == rows[1]["full_metrics"][iteration]["steps"]
        assert rows[0]["full_metrics"][iteration]["sequential_starts_per_gpu_next"] == rows[1]["full_metrics"][iteration]["sequential_starts_per_gpu_next"]
