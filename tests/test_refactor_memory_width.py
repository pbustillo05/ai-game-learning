from __future__ import annotations

import pytest
import torch

from pbg.core.memory_control import (
    MEMORY_PHASES,
    MemoryAdmissionStop,
    PhaseMemoryForecaster,
    PhaseMemoryModel,
    PhaseMemoryObservation,
    SequentialMemoryController,
)


def _forecaster(per_root: int = 10, fixed: int = 100) -> PhaseMemoryForecaster:
    return PhaseMemoryForecaster(
        {
            phase: PhaseMemoryModel(fixed, {f"{phase}_schema": per_root})
            for phase in MEMORY_PHASES
        },
        window=3,
    )


def _controller(**kwargs) -> SequentialMemoryController:
    values = dict(
        minimum_roots_per_gpu=2,
        maximum_roots_per_gpu=100,
        current_roots_per_gpu=4,
        device_bytes=1_200,
        guard_bytes=100,
        forecaster=_forecaster(),
        growth_fraction=0.25,
        growth_delay=2,
    )
    values.update(kwargs)
    return SequentialMemoryController(**values)


def _observations(controller: SequentialMemoryController, *, extra: int = 0):
    roots = controller.current_roots_per_gpu
    predictions = controller.forecaster.predict(roots)
    return {
        phase: PhaseMemoryObservation(
            roots,
            100,
            {f"{phase}_schema": roots * 10 + extra},
            predictions[phase],
            100 + roots * 10 + extra,
        )
        for phase in MEMORY_PHASES
    }


def test_memory_only_binary_search_and_rank_minimum() -> None:
    controller = _controller()
    assert controller.local_safe_width() == 100
    assert controller.choose_root_width(rank_safe_widths=torch.tensor([80, 7, 20])) == 4
    controller.observe_success(_observations(controller))
    controller.successful_streak = 2
    assert controller.choose_root_width(rank_safe_widths=torch.tensor([80, 7, 20])) == 5


def test_unsafe_evidence_decreases_immediately_and_growth_is_delayed_capped() -> None:
    controller = _controller(
        current_roots_per_gpu=50,
        device_bytes=500,
        guard_bytes=100,
    )
    # fixed 100 + 10G <= 400, so G_safe=30 and decrease is unrestricted.
    assert controller.choose_root_width() == 30
    controller.device_bytes = 1_200
    controller.observe_success(_observations(controller))
    assert controller.choose_root_width() == 30
    controller.observe_success(_observations(controller))
    # 25% cap, despite a much larger safe maximum.
    assert controller.choose_root_width() == 37


def test_positive_surprise_widens_phase_model_and_can_reduce_next_width() -> None:
    controller = _controller(current_roots_per_gpu=10, device_bytes=500, guard_bytes=100)
    observations = _observations(controller, extra=200)
    controller.observe_success(observations)
    assert controller.forecaster.phase_model("rollout").bytes_per_root == 30
    # fixed 100 + 30G <= 400.
    assert controller.choose_root_width() == 10
    controller.current_roots_per_gpu = 20
    assert controller.choose_root_width() == 10


def test_checkpoint_roundtrip_preserves_controller_state() -> None:
    controller = _controller()
    controller.observe_success(_observations(controller))
    restored = SequentialMemoryController.from_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
    assert restored.choose_root_width() == controller.choose_root_width()


def test_minimum_rejection_and_transaction_failure_are_clean_no_reroll_stops() -> None:
    controller = _controller(device_bytes=150, guard_bytes=40)
    with pytest.raises(MemoryAdmissionStop, match="minimum sequential root width"):
        controller.choose_root_width()
    assert controller.current_roots_per_gpu == 4

    controller = _controller()
    frozen = controller.current_roots_per_gpu
    controller.record_transaction_failure("seal", 10_000)
    with pytest.raises(MemoryAdmissionStop, match="not retried"):
        controller.choose_root_width()
    assert controller.current_roots_per_gpu == frozen


def test_phase_ledger_rejects_owned_bytes_above_actual_peak() -> None:
    with pytest.raises(ValueError, match="ownership exceeds"):
        PhaseMemoryObservation(2, 100, {"chunks": 100}, 250, 199)
