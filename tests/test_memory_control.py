from __future__ import annotations

from dataclasses import dataclass

import torch

from pbg.core.checkpointing import (
    compatible_scientific_configs,
    config_dict_fingerprint,
    matches_legacy_run_fingerprint,
)
from pbg.core.memory_control import PolicyMemoryForecaster, unique_tensor_bytes
from pbg.core.training import SequentialWorkController


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
            "memory_forecast_window": 128,
            "memory_forecast_half_life": 32.0,
            "memory_forecast_min_observations": 40,
            "memory_soft_exceedance_rate": 0.025,
            "memory_max_start_increase_fraction": 0.10,
            "memory_retry_backoff_fraction": 0.20,
            "memory_max_rollout_retries": 2,
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
    controller = SequentialWorkController(
        current_starts_per_gpu=20,
        memory_forecaster=PolicyMemoryForecaster(
            window=8, half_life=4.0, minimum_observations=4
        ),
    )
    controller.stage_rollout(
        local_decisions=100,
        rank_max_decisions=100,
        rank_max_decisions_per_start=5.0,
        target_decisions=200,
        minimum_starts=5,
        maximum_starts=40,
        ema_alpha=1.0,
        target_reuse=3.0,
        minibatch_size=8,
    )
    assert controller.current_starts_per_gpu == 20
    controller.memory_forecaster.observe(
        starts=20,
        peak_reserved_bytes=20_000,
        end_reserved_bytes=1_000,
        metrics={"nodes": 200},
        ceiling_bytes=6_000,
    )
    result = controller.commit_memory_observation(
        metrics={
            "iteration_peak_reserved_bytes": 20_000,
            "iteration_end_reserved_bytes": 1_000,
            "nodes": 200,
        },
        ceiling_bytes=6_000,
        minimum_starts=5,
        maximum_starts=40,
        maximum_increase_fraction=0.10,
    )
    assert controller.current_starts_per_gpu < 20
    assert "memory" in result["sequential_start_clamp_reason"]
    restored = SequentialWorkController.from_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
