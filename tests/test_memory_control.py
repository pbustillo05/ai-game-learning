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
    PolicyMemoryForecaster,
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


def test_policy_forecast_roundtrip_and_policy_conditioned_scaling() -> None:
    forecast = PolicyMemoryForecaster(
        window=8, half_life=4.0, minimum_observations=4
    )
    for index in range(6):
        forecast.observe(
            starts=10,
            peak_reserved_bytes=10_000 + 200 * index,
            end_reserved_bytes=5_000,
            metrics={
                "nodes": 100 + 2 * index,
                "execution_training_post_2_model_rows": 50 + 2 * index,
                "execution_training_post_2_padded_token_pairs": 5_000 + 100 * index,
                "retained_target_bytes": 1_000,
            },
            ceiling_bytes=30_000,
        )
    assert forecast.predict(15) > forecast.predict(10)
    restored = PolicyMemoryForecaster.from_state_dict(forecast.state_dict())
    assert restored.state_dict() == forecast.state_dict()
    assert restored.predict(15) == forecast.predict(15)


def test_positive_surprise_widens_immediately_and_drift_freezes_growth() -> None:
    forecast = PolicyMemoryForecaster(
        window=16,
        half_life=8.0,
        minimum_observations=4,
        soft_exceedance_rate=0.25,
    )
    base = {
        "nodes": 100,
        "execution_training_post_2_model_rows": 50,
    }
    for _ in range(4):
        forecast.observe(
            starts=10,
            peak_reserved_bytes=10_000,
            end_reserved_bytes=5_000,
            metrics=base,
            ceiling_bytes=30_000,
        )
    old_bound = forecast.residual_bound_bytes
    for offset in range(4):
        forecast.observe(
            starts=10,
            peak_reserved_bytes=20_000 + offset,
            end_reserved_bytes=5_000,
            metrics=base,
            ceiling_bytes=30_000,
        )
    assert forecast.residual_bound_bytes >= old_bound
    assert forecast.freeze_upward
    assert forecast.calibration_epoch >= 1


def test_restart_rate_enters_one_calibration_epoch_and_requires_clean_data() -> None:
    forecast = PolicyMemoryForecaster(
        window=16,
        half_life=8.0,
        minimum_observations=4,
        soft_exceedance_rate=0.25,
    )
    metrics = {"nodes": 100}
    for _ in range(3):
        forecast.observe(
            starts=10,
            peak_reserved_bytes=10_000,
            end_reserved_bytes=5_000,
            metrics=metrics,
            ceiling_bytes=30_000,
        )
    result = forecast.observe(
        starts=10,
        peak_reserved_bytes=10_000,
        end_reserved_bytes=5_000,
        metrics=metrics,
        ceiling_bytes=30_000,
        restarted=True,
    )
    assert result["memory_forecast_restart_rate"] == 0.25
    assert not forecast.freeze_upward
    result = forecast.observe(
        starts=10,
        peak_reserved_bytes=10_000,
        end_reserved_bytes=5_000,
        metrics=metrics,
        ceiling_bytes=30_000,
        restarted=True,
    )
    assert result["memory_forecast_restart_rate"] > 0.25
    assert forecast.freeze_upward
    epoch = forecast.calibration_epoch
    forecast.observe(
        starts=10,
        peak_reserved_bytes=10_000,
        end_reserved_bytes=5_000,
        metrics=metrics,
        ceiling_bytes=30_000,
        restarted=True,
    )
    assert forecast.calibration_epoch == epoch


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
