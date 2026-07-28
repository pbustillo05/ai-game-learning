from __future__ import annotations

import pytest
import torch

from tests.legacy_config import load_config
from pbg.core.memory_control import (
    MEMORY_PHASES,
    PhaseMemoryForecaster,
    PhaseMemoryModel,
    SequentialMemoryController,
)


def _controller() -> SequentialMemoryController:
    forecaster = PhaseMemoryForecaster(
        {
            phase: PhaseMemoryModel(100, {"schema": 10})
            for phase in MEMORY_PHASES
        },
        window=4,
    )
    return SequentialMemoryController(
        minimum_roots_per_gpu=2,
        maximum_roots_per_gpu=40,
        current_roots_per_gpu=8,
        device_bytes=600,
        guard_bytes=100,
        forecaster=forecaster,
        growth_fraction=0.25,
        growth_delay=2,
    )


def test_repository_sequential_config_uses_explicit_per_gpu_root_contract() -> None:
    adaptive = load_config("configs/train/nlhe_hu_32bb_rb.yaml")
    assert adaptive.rollout.sequential_roots_per_gpu_initial == 2_560
    assert adaptive.rollout.sequential_roots_per_gpu_minimum == 512
    assert adaptive.rollout.sequential_roots_per_gpu_maximum == 3_072


def test_obsolete_global_and_decision_target_fields_are_rejected(tmp_path) -> None:
    source = tmp_path / "obsolete.yaml"
    source.write_text(
        """
device: cuda
game: {name: poker, n_players: 2, stack_bb: 8, rank_count: 4}
rollout:
  sequential_starts: 8
  target_decisions_per_gpu: 100
  beta: null
  storage_page_rows: 100
completion: {n_samples: 2}
model: {}
loss:
  {name: ppo_value_entropy, lam: 1.0, clip_eps: 0.2, clip_eps_v: 0.2,
   c_v: 0.5, c_h: 0.01, entropy_schedule: const}
training:
  {mode: normal, iters: 1, gd_steps_per_iter: 1, minibatch_size: 1,
   lr: 0.001, adam_betas: [0.9, 0.999], grad_clip: 1.0, amp: false,
   seed: 1, checkpoint_every: 1, out_dir: /tmp/pbg}
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="unknown configuration key"):
        load_config(source)


def test_rank_minimum_is_common_per_gpu_width_and_state_roundtrips() -> None:
    controller = _controller()
    # Local safe width is 40; the rank minimum is seven. Growth remains
    # disabled on cold start, so the frozen width stays eight until evidence.
    assert controller.local_safe_width() == 40
    assert controller.choose_root_width(
        rank_safe_widths=torch.tensor([40, 7, 30])
    ) == 7
    restored = SequentialMemoryController.from_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
