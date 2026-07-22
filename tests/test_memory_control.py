from __future__ import annotations

from dataclasses import dataclass

import torch

from pbg.core.memory_control import (
    MEMORY_PHASES,
    PhaseMemoryForecaster,
    PhaseMemoryModel,
    PhaseMemoryObservation,
    SequentialMemoryController,
    unique_tensor_bytes,
)


@dataclass
class _Aliased:
    original: torch.Tensor
    view: torch.Tensor


def test_unique_tensor_bytes_counts_aliased_storage_once() -> None:
    original = torch.zeros(32, dtype=torch.float32)
    value = _Aliased(original, original.view(8, 4))
    assert unique_tensor_bytes(value) == 32 * 4


def test_memory_commit_is_transactional_and_safety_decrease_is_unrestricted() -> None:
    forecaster = PhaseMemoryForecaster(
        {
            phase: PhaseMemoryModel(1_000, {"schema": 100})
            for phase in MEMORY_PHASES
        },
        window=4,
    )
    controller = SequentialMemoryController(
        minimum_roots_per_gpu=5,
        maximum_roots_per_gpu=40,
        current_roots_per_gpu=20,
        device_bytes=6_000,
        guard_bytes=1_000,
        forecaster=forecaster,
        growth_fraction=0.10,
        growth_delay=2,
    )
    observations = {
        phase: PhaseMemoryObservation(
            roots_per_gpu=20,
            fixed_bytes=1_000,
            variable_bytes={"schema": 8_000},
            predicted_peak_bytes=3_000,
            actual_peak_bytes=9_000,
        )
        for phase in MEMORY_PHASES
    }
    controller.observe_success(observations)
    assert controller.choose_root_width() == 10
    restored = SequentialMemoryController.from_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
