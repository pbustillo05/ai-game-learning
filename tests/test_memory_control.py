from __future__ import annotations

from dataclasses import dataclass

import torch

from pbg.core.checkpointing import (
    compatible_scientific_configs,
    config_dict_fingerprint,
    matches_legacy_run_fingerprint,
)
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


def test_legacy_checkpoint_without_probe_accepts_current_null_probe() -> None:
    legacy = {"training": {"checkpoint_every": 5}, "rollout": {}}
    current = {
        "training": {},
        "rollout": {
            "sequential_roots_per_gpu_minimum": 1,
            "sequential_roots_per_gpu_maximum": 8,
            "memory_calibration_window": 32,
            "memory_growth_fraction": 0.10,
            "memory_growth_delay": 8,
            "memory_prediction_z": 3.0,
        },
        "probe": None,
    }
    assert compatible_scientific_configs(legacy, current)


def test_legacy_run_fingerprint_allows_only_checkpoint_cadence_drift() -> None:
    original = {
        "game": {"n_players": 2},
        "training": {"iters": 100, "checkpoint_every": 10, "lr": 1e-4},
    }
    resumed = {
        "game": {"n_players": 2},
        "training": {"iters": 100, "checkpoint_every": 5, "lr": 1e-4},
    }
    recorded = config_dict_fingerprint(original)
    assert matches_legacy_run_fingerprint(recorded, resumed)
    changed_science = {
        **resumed,
        "training": {**resumed["training"], "lr": 2e-4},
    }
    assert not matches_legacy_run_fingerprint(recorded, changed_science)




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
