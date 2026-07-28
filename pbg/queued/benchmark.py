"""Frozen Phase 8 benchmark evaluation for queued branched RB."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass

FROZEN_PHASE8_SEEDS = (104830, 209559, 314288)


@dataclass(frozen=True)
class Phase8BenchmarkThresholds:
    branched_rollout_ms_per_decision: float = 9.433
    routing_total_bytes: int = 12_838_000_000
    routing_bytes_per_decision: float = 0.841 * 1_000_000
    base_host_synchronizations: int = 64
    child_reservation_commit_seconds: float = 17.71
    target_propagation_seconds: float = 0.500
    peak_growth_fraction: float = 0.05
    sequential_regression_fraction: float = 0.10


DEFAULT_PHASE8_BENCHMARK_THRESHOLDS = Phase8BenchmarkThresholds()


@dataclass(frozen=True)
class Phase8BenchmarkSample:
    seed: int
    retained_decisions: int
    target_layers: int
    rollout_seconds: float
    routing_traffic_bytes: int
    host_synchronizations: int
    child_reservation_commit_seconds: float
    target_propagation_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    @property
    def rollout_ms_per_decision(self) -> float:
        return 1000.0 * self.rollout_seconds / self.retained_decisions

    @property
    def routing_bytes_per_decision(self) -> float:
        return self.routing_traffic_bytes / self.retained_decisions


@dataclass(frozen=True)
class SequentialBenchmarkSample:
    """One matched current/baseline sequential-RB rollout measurement."""

    seed: int
    retained_decisions: int
    rollout_seconds: float
    baseline_ms_per_decision: float

    def __post_init__(self) -> None:
        if (
            self.seed < 0
            or self.retained_decisions <= 0
            or self.rollout_seconds <= 0
            or self.baseline_ms_per_decision <= 0
        ):
            raise ValueError("sequential benchmark sample is invalid")

    @property
    def rollout_ms_per_decision(self) -> float:
        return 1000.0 * self.rollout_seconds / self.retained_decisions

    @property
    def regression_ratio(self) -> float:
        return self.rollout_ms_per_decision / self.baseline_ms_per_decision


@dataclass(frozen=True)
class Phase8BenchmarkReport:
    revision: str
    identity: Mapping[str, str]
    samples: tuple[Phase8BenchmarkSample, ...]
    sequential_samples: tuple[SequentialBenchmarkSample, ...]
    medians: Mapping[str, float]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def _validate_frozen_iteration(
    iteration: object, seed: int
) -> Mapping[str, str]:
    plan = getattr(iteration, "plan", None)
    controller = getattr(iteration, "controller", None)
    arguments = getattr(iteration, "arguments", None)
    if plan is None or controller is None or not isinstance(arguments, Mapping):
        raise ValueError("benchmark factory did not return a queued iteration")
    mode = plan.mode
    if (
        mode.training_regime.value != "self_play"
        or mode.rollout_mode.value != "branched.fixed"
        or mode.private_estimator.value != "rao_blackwell"
    ):
        raise ValueError("benchmark iteration is not fixed branched RB self-play")
    geometry = arguments.get("geometry")
    if (
        geometry is None
        or geometry.players != 2
        or geometry.stack_bb != 32.0
        or geometry.rank_count != 13
    ):
        raise ValueError("benchmark geometry must be full-deck HU 32 bb")
    if (
        plan.manifest.root_count != 1
        or plan.manifest.family_pack_cap != 84
        or arguments.get("completion_samples") != 1024
        or arguments.get("seed") != seed
        or arguments.get("benchmark_telemetry") is not True
        or arguments.get("benchmark_rollout_only") is not True
    ):
        raise ValueError("benchmark logical-work identity is not frozen")
    fixed = getattr(controller, "fixed_beta", None)
    if not isinstance(fixed, Mapping) or dict(fixed) != {
        "player": 3.0,
        "chance": 3.0,
    }:
        raise ValueError("benchmark controller must use fixed beta=3")
    identity = arguments.get("benchmark_identity")
    required_identity = {
        "checkpoint_sha256",
        "scientific_config_sha256",
        "evaluator_version",
        "rng_version",
    }
    if (
        not isinstance(identity, Mapping)
        or set(identity) != required_identity
        or any(
            not isinstance(value, str) or not value
            for value in identity.values()
        )
    ):
        raise ValueError("benchmark identity is incomplete")
    return dict(identity)


def run_phase8_benchmark(
    iteration_factory: Callable[[int], object],
    *,
    revision: str,
    legacy_peak_allocated_bytes: int,
    sequential_samples: tuple[SequentialBenchmarkSample, ...],
    thresholds: Phase8BenchmarkThresholds = (
        DEFAULT_PHASE8_BENCHMARK_THRESHOLDS
    ),
) -> Phase8BenchmarkReport:
    """Run and evaluate the exact frozen three-seed queued workload."""

    if not revision:
        raise ValueError("benchmark revision must be nonempty")
    if legacy_peak_allocated_bytes <= 0:
        raise ValueError("legacy peak allocation must be positive")
    if (
        tuple(sample.seed for sample in sequential_samples)
        != FROZEN_PHASE8_SEEDS
    ):
        raise ValueError(
            "sequential benchmark must measure every frozen seed in order"
        )
    samples: list[Phase8BenchmarkSample] = []
    resolved_identity: Mapping[str, str] | None = None
    for seed in FROZEN_PHASE8_SEEDS:
        iteration = iteration_factory(seed)
        identity = _validate_frozen_iteration(iteration, seed)
        if resolved_identity is None:
            resolved_identity = identity
        elif identity != resolved_identity:
            raise ValueError("benchmark identity changed across seeds")
        result = iteration.run(iteration=0)
        telemetry = getattr(result, "telemetry", None)
        if telemetry is None:
            raise ValueError("benchmark iteration omitted queued telemetry")
        decisions = int(result.player_nodes)
        if decisions <= 0:
            raise ValueError("benchmark retained no player decisions")
        samples.append(
            Phase8BenchmarkSample(
                seed=seed,
                retained_decisions=decisions,
                target_layers=int(result.target_layers),
                rollout_seconds=float(telemetry.rollout_seconds),
                routing_traffic_bytes=int(
                    telemetry.routing_traffic_bytes
                ),
                host_synchronizations=int(
                    telemetry.host_synchronizations
                ),
                child_reservation_commit_seconds=float(
                    telemetry.child_reservation_commit_seconds
                ),
                target_propagation_seconds=float(
                    telemetry.target_propagation_seconds
                ),
                peak_allocated_bytes=int(
                    telemetry.peak_allocated_bytes
                ),
                peak_reserved_bytes=int(
                    telemetry.peak_reserved_bytes
                ),
            )
        )
    values = {
        "rollout_ms_per_decision": statistics.median(
            sample.rollout_ms_per_decision for sample in samples
        ),
        "routing_traffic_bytes": statistics.median(
            sample.routing_traffic_bytes for sample in samples
        ),
        "routing_bytes_per_decision": statistics.median(
            sample.routing_bytes_per_decision for sample in samples
        ),
        "host_synchronizations": statistics.median(
            sample.host_synchronizations for sample in samples
        ),
        "host_synchronization_limit": statistics.median(
            thresholds.base_host_synchronizations + sample.target_layers
            for sample in samples
        ),
        "child_reservation_commit_seconds": statistics.median(
            sample.child_reservation_commit_seconds
            for sample in samples
        ),
        "target_propagation_seconds": statistics.median(
            sample.target_propagation_seconds for sample in samples
        ),
        "peak_allocated_bytes": statistics.median(
            sample.peak_allocated_bytes for sample in samples
        ),
        "sequential_throughput_ratio": statistics.median(
            sample.regression_ratio for sample in sequential_samples
        ),
    }
    comparisons = (
        (
            "rollout_ms_per_decision",
            thresholds.branched_rollout_ms_per_decision,
        ),
        ("routing_traffic_bytes", thresholds.routing_total_bytes),
        (
            "routing_bytes_per_decision",
            thresholds.routing_bytes_per_decision,
        ),
        (
            "host_synchronizations",
            values["host_synchronization_limit"],
        ),
        (
            "child_reservation_commit_seconds",
            thresholds.child_reservation_commit_seconds,
        ),
        (
            "target_propagation_seconds",
            thresholds.target_propagation_seconds,
        ),
        (
            "peak_allocated_bytes",
            legacy_peak_allocated_bytes
            * (1 + thresholds.peak_growth_fraction),
        ),
        (
            "sequential_throughput_ratio",
            1 + thresholds.sequential_regression_fraction,
        ),
    )
    violations = tuple(
        f"{name}={values[name]:.6g} exceeds {limit:.6g}"
        for name, limit in comparisons
        if values[name] > limit
    )
    return Phase8BenchmarkReport(
        revision=revision,
        identity={} if resolved_identity is None else resolved_identity,
        samples=tuple(samples),
        sequential_samples=sequential_samples,
        medians=values,
        violations=violations,
    )


__all__ = [
    "DEFAULT_PHASE8_BENCHMARK_THRESHOLDS",
    "FROZEN_PHASE8_SEEDS",
    "Phase8BenchmarkReport",
    "Phase8BenchmarkSample",
    "Phase8BenchmarkThresholds",
    "SequentialBenchmarkSample",
    "run_phase8_benchmark",
]
