"""Matched-work benchmark consumer for the public queued runtime."""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from pbg.config import RootConfig
from pbg.identity import accelerator_identity, code_identity
from pbg.queued import LOGICAL_RNG_VERSION, sha256_file
from pbg.runtime import TrainingSession

_REQUIRED_SAMPLE_METRICS = (
    "seconds",
    "iteration",
    "player_nodes",
    "player_nodes_per_second",
    "incremental_allocated_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "logical_work_before",
    "logical_work_after",
    "structure",
    "telemetry",
)


def benchmark_training(
    cfg: RootConfig,
    *,
    warmup: int = 1,
    repeats: int = 5,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")
    samples = []
    benchmark_device: torch.device | None = None
    for index in range(warmup + repeats):
        # Every repeat starts from the identical model/optimizer/RNG/run state.
        # Warmup compiles the same production path but is never a measured row.
        session = TrainingSession(
            cfg, resume_from=checkpoint, benchmark_telemetry=True
        )
        benchmark_device = session.device
        iteration = session.run_states[0].iteration
        logical_before = tuple(
            dict(state.logical_work) for state in session.run_states
        )
        torch.cuda.synchronize(session.device)
        torch.cuda.reset_peak_memory_stats(session.device)
        before_allocated = torch.cuda.memory_allocated(session.device)
        started = time.perf_counter()
        result = session.step()
        torch.cuda.synchronize(session.device)
        elapsed = time.perf_counter() - started
        if index >= warmup:
            nodes = sum(
                int(getattr(value, "player_nodes", getattr(value, "nodes", 0)))
                for value in (
                    getattr(result, "results", ()) or (result,)
                )
                if value is not None
            )
            lane_values = tuple(
                value
                for value in (
                    getattr(result, "results", ()) or (result,)
                )
                if value is not None
            )
            telemetry = tuple(
                (
                    asdict(value.telemetry)
                    if is_dataclass(getattr(value, "telemetry", None))
                    else None
                )
                for value in lane_values
            )
            samples.append(
                {
                    "seconds": elapsed,
                    "iteration": iteration,
                    "logical_work_before": logical_before,
                    "logical_work_after": tuple(
                        dict(state.logical_work) for state in session.run_states
                    ),
                    "player_nodes": nodes,
                    "player_nodes_per_second": nodes / elapsed,
                    "incremental_allocated_bytes": (
                        torch.cuda.memory_allocated(session.device)
                        - before_allocated
                    ),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        session.device
                    ),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        session.device
                    ),
                    "structure": tuple(
                        dict(getattr(value, "observability", {}))
                        for value in lane_values
                    ),
                    "telemetry": telemetry,
                }
            )
        del result
        session.close()
        del session
    numeric = {
        key
        for key in set.intersection(*(set(sample) for sample in samples))
        if all(
            isinstance(sample[key], (int, float))
            and not isinstance(sample[key], bool)
            for sample in samples
        )
    }
    median = {
        key: float(statistics.median(sample[key] for sample in samples))
        for key in sorted(numeric)
    }
    assert benchmark_device is not None
    properties = torch.cuda.get_device_properties(benchmark_device)
    return {
        "schema": "pbg-queued-benchmark-v1",
        "mode": dict(cfg.mode),
        "scientific_config": cfg.scientific_mapping(),
        "execution": cfg.execution.mapping(),
        "identity": {
            "logical_rng_version": LOGICAL_RNG_VERSION,
            "checkpoint_sha256": (
                None if checkpoint is None else sha256_file(Path(checkpoint))
            ),
            "code": code_identity(),
            "accelerator": accelerator_identity(
                cfg.scientific.common.game.rank_count
            ),
            "required_sample_metrics": _REQUIRED_SAMPLE_METRICS,
        },
        "environment": {
            "device": str(benchmark_device),
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "warmup": warmup,
        "repeats": repeats,
        "median": median,
        "samples": samples,
    }


def compare_benchmark(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_regression: float = 0.10,
    allow_environment_mismatch: bool = False,
) -> list[str]:
    if not 0 <= max_regression < 1:
        raise ValueError("max_regression must be in [0,1)")
    failures = []
    for identity in ("mode", "scientific_config", "execution", "identity"):
        if current.get(identity) != baseline.get(identity):
            failures.append(f"{identity} identity differs")
    for report_name, report in (("current", current), ("baseline", baseline)):
        for index, sample in enumerate(report.get("samples", ())):
            missing = set(_REQUIRED_SAMPLE_METRICS).difference(sample)
            if missing:
                failures.append(
                    f"{report_name} sample {index} misses {sorted(missing)}"
                )
    if (
        not allow_environment_mismatch
        and current.get("environment") != baseline.get("environment")
    ):
        failures.append("environment identity differs")
    current_rate = float(current["median"]["player_nodes_per_second"])
    baseline_rate = float(baseline["median"]["player_nodes_per_second"])
    if current_rate < baseline_rate * (1.0 - max_regression):
        failures.append("player-node throughput regressed")
    return failures


__all__ = ["benchmark_training", "compare_benchmark"]
