from __future__ import annotations

import copy
import gc
import math
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from tests.legacy_config import ConfigError, load_config, load_config_mapping
from pbg.core.adaptive_branching import (
    AdaptiveKKTController,
    branched_attempt_address,
    FixedBetaController,
)
from pbg.core.branching import sample_branched_tree
from pbg.core.checkpointing import load_checkpoint, save_checkpoint
from pbg.core.game import GameBundle
from pbg.core.distributed_failure import (
    DistributedFailureAgreement,
    DistributedPhaseFailure,
)
from pbg.core.memory_control import (
    MEMORY_PHASES,
    BranchedMemoryRejection,
    PhaseMemoryForecaster,
    PhaseMemoryModel,
    SequentialMemoryController,
)
from tests.test_c09_labeled_branching import _Adapter, _LabeledBackend, _Model


def _fixed_controller(**overrides: float) -> FixedBetaController:
    values = {
        "target_tree_size": 100.0,
        "gamma": 0.5,
        "eta": 0.25,
        "scale_min": 0.5,
        "scale_max": 2.0,
        "memory_pressure_target": 0.8,
    }
    values.update(overrides)
    pressure = values.pop("memory_pressure_target")
    return FixedBetaController(
        scalar=AdaptiveKKTController(
            target_tree_size=values["target_tree_size"],
            gamma_kappa=values["gamma"],
            eta_kappa=values["eta"],
            kappa_min=values["scale_min"],
            kappa_max=values["scale_max"],
            log_kappa=0.0,
        ),
        memory_pressure_target=pressure,
    )


def _forecaster(fixed: int = 100, per_node: float = 10.0) -> PhaseMemoryForecaster:
    return PhaseMemoryForecaster(
        {
            phase: PhaseMemoryModel(
                fixed,
                {"analytic_schema_and_workspace": per_node},
            )
            for phase in MEMORY_PHASES
        }
    )


def _memory_controller(
    *, device_bytes: int, guard_bytes: int, forecaster: PhaseMemoryForecaster
) -> SequentialMemoryController:
    return SequentialMemoryController(
        device_bytes=device_bytes,
        guard_bytes=guard_bytes,
        forecaster=forecaster,
        minimum_roots_per_gpu=1,
        maximum_roots_per_gpu=1,
        current_roots_per_gpu=1,
    )


def test_fixed_beta_direction_damping_clipping_and_schedule_preservation() -> None:
    controller = _fixed_controller()
    first = controller.observe(25.0, 0.4)
    assert first["fixed_beta_scale"] == pytest.approx(math.sqrt(2.0))
    effective = controller.effective_schedule({"default": 2.0, "river": 3.0})
    assert effective["river"] / effective["default"] == pytest.approx(1.5)

    before = controller.scale
    controller.observe(400.0, 0.95)
    assert controller.scale < before

    for _ in range(20):
        controller.observe(1.0e-3, 0.1)
    assert controller.scale == pytest.approx(controller.scalar.kappa_max)
    for _ in range(20):
        controller.observe(1.0e8, 1.0)
    assert controller.scale == pytest.approx(controller.scalar.kappa_min)
    assert all(
        math.isfinite(value) and value >= 1.0
        for value in controller.effective_schedule(
            {"default": 1.0, "river": 1.25}
        ).values()
    )


def test_emergency_backoff_is_monotone_bounded_in_both_modes() -> None:
    fixed = _fixed_controller()
    fixed_values = [fixed.scale]
    for _ in range(20):
        fixed.emergency_backoff(0.8)
        fixed_values.append(fixed.scale)
    assert fixed_values == sorted(fixed_values, reverse=True)
    assert fixed.scale == pytest.approx(fixed.scalar.kappa_min)

    dynamic = AdaptiveKKTController(
        target_tree_size=100.0,
        gamma_kappa=0.5,
        eta_kappa=0.25,
        kappa_min=0.1,
        kappa_max=10.0,
        log_kappa=math.log(10.0),
    )
    kappa = [math.exp(dynamic.log_kappa)]
    for _ in range(30):
        dynamic.emergency_backoff(0.8)
        kappa.append(math.exp(dynamic.log_kappa))
    assert kappa == sorted(kappa, reverse=True)
    assert math.exp(dynamic.log_kappa) == pytest.approx(dynamic.kappa_min)


def test_fixed_beta_and_branched_memory_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = load_config("configs/train/nlhe6.yaml")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    beta = _fixed_controller()
    beta.observe(80.0, 0.6)
    memory = _memory_controller(
        device_bytes=10_000,
        guard_bytes=1_000,
        forecaster=_forecaster(),
    )
    path = tmp_path / "fixed.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        4,
        cfg,
        torch.Generator().manual_seed(3),
        np.random.default_rng(4),
        adaptive_work_state=memory.state_dict(),
        branching_controller_state=beta.state_dict(),
    )
    restored = load_checkpoint(path, model)
    assert (
        FixedBetaController.from_state_dict(
            restored["branching_controller_state"]
        ).state_dict()
        == beta.state_dict()
    )
    assert (
        SequentialMemoryController.from_state_dict(
            restored["adaptive_work_state"]
        ).state_dict()
        == memory.state_dict()
    )


def test_lower_ideal_upper_configuration_order_is_strict() -> None:
    raw = yaml.safe_load(Path("configs/train/nlhe6.yaml").read_text(encoding="utf-8"))
    raw["rollout"]["branching_min_nodes"] = 1000
    raw["rollout"]["branching_ideal_nodes"] = 2000.0
    raw["rollout"]["safe_tree_size"] = 3000.0
    load_config_mapping(copy.deepcopy(raw))
    for ideal in (1000.0, 3000.0):
        bad = copy.deepcopy(raw)
        bad["rollout"]["branching_ideal_nodes"] = ideal
        with pytest.raises(ConfigError, match="branching_min_nodes.*branching_ideal_nodes"):
            load_config_mapping(bad)


def test_accounting_rejection_preserves_reason_and_fresh_attempt_address() -> None:
    memory = _memory_controller(
        device_bytes=1_000,
        guard_bytes=100,
        forecaster=_forecaster(fixed=800, per_node=100.0),
    )
    cfg = SimpleNamespace(
        roots_per_gpu=1,
        storage_page_rows=32,
        branching_mode="fixed",
        adaptive_kkt=None,
        safe_tree_size=None,
        queue_capacity_rows=64,
    )
    with pytest.raises(BranchedMemoryRejection, match="child-wave owners") as caught:
        sample_branched_tree(
            GameBundle(_Model(), _LabeledBackend(2.3, two_levels=True), _Adapter(), None),
            cfg,
            torch.Generator().manual_seed(3),
            memory_controller=memory,
            attempt=0,
        )
    assert caught.value.predicted_bytes > caught.value.hard_ceiling_bytes

    generous = _memory_controller(
        device_bytes=1_000_000,
        guard_bytes=1,
        forecaster=_forecaster(),
    )
    samples = []
    for attempt in (0, 1):
        samples.append(
            sample_branched_tree(
                GameBundle(
                    _Model(),
                    _LabeledBackend(2.3, two_levels=True),
                    _Adapter(),
                    None,
                ),
                cfg,
                torch.Generator().manual_seed(3),
                memory_controller=generous,
                attempt=attempt,
            )
        )
    assert samples[0].branching_diagnostics["branching_attempt"] == 0
    assert samples[1].branching_diagnostics["branching_attempt"] == 1
    assert samples[0].branching_diagnostics["branching_attempt_address"] == 0
    assert samples[1].branching_diagnostics["branching_attempt_address"] == 1
    assert samples[0].topology.degree.tolist() != samples[1].topology.degree.tolist()


def test_branched_attempt_addresses_are_disjoint_across_ranks_and_attempts() -> None:
    addresses = {
        branched_attempt_address(attempt, rank=rank, world_size=3)
        for attempt in range(4)
        for rank in range(3)
    }
    assert addresses == set(range(12))
    with pytest.raises(ValueError, match="nonnegative integer"):
        branched_attempt_address(True)
    with pytest.raises(ValueError, match=r"\[0, world_size\)"):
        branched_attempt_address(0, rank=2, world_size=2)


def test_single_rank_nonmemory_retry_phase_failure_is_terminal() -> None:
    agreement = DistributedFailureAgreement(torch.device("cpu"))

    with pytest.raises(DistributedPhaseFailure) as caught:
        agreement.run_retryable(
            "branched attempt",
            lambda: (_ for _ in ()).throw(ValueError("invalid data")),
            retry_types=(MemoryError,),
        )

    assert caught.value.phase == "branched attempt"
    assert caught.value.failing_rank == 0
    assert caught.value.local_detail == "ValueError: invalid data"


def test_retryable_failure_does_not_retain_failed_attempt_traceback() -> None:
    class AttemptOwner:
        pass

    agreement = DistributedFailureAgreement(torch.device("cpu"))
    owner_ref: weakref.ReferenceType[AttemptOwner] | None = None

    def fail() -> None:
        nonlocal owner_ref
        owner = AttemptOwner()
        owner_ref = weakref.ref(owner)
        raise MemoryError("injected")

    result, error, rank = agreement.run_retryable(
        "branched attempt",
        fail,
        retry_types=(MemoryError,),
    )
    gc.collect()

    assert result is None
    assert isinstance(error, MemoryError)
    assert error.__traceback__ is None
    assert rank == 0
    assert owner_ref is not None and owner_ref() is None


def test_mocked_distributed_attempt_discards_for_any_rejecting_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreement = DistributedFailureAgreement(torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def remote_rank_one_rejected(value, op=None, group=None):
        del op, group
        value.fill_(3)  # world + rejecting rank

    monkeypatch.setattr(torch.distributed, "all_reduce", remote_rank_one_rejected)
    result, error, rank = agreement.run_retryable(
        "attempt",
        lambda: "rank-zero-result",
        retry_types=(MemoryError,),
    )
    assert result is None
    assert error is None
    assert rank == 1


def test_mocked_distributed_retry_transition_requires_equal_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreement = DistributedFailureAgreement(torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    calls = 0

    def disagree(value, op=None, group=None):
        nonlocal calls
        del op, group
        value.fill_(4 if calls == 0 else 5)
        calls += 1

    monkeypatch.setattr(torch.distributed, "all_reduce", disagree)
    with pytest.raises(DistributedPhaseFailure) as caught:
        agreement.check_equal_int("retry transition", 4)
    assert caught.value.local_detail == (
        "rank-common transition mismatch: min=4, max=5"
    )
