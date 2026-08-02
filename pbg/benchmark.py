"""Matched-work benchmark consumer for the public queued runtime."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from pbg.config import RootConfig
from pbg.core.persistence import model_state_sha256, sha256_file
from pbg.core.rollout import LOGICAL_RNG_VERSION
from pbg.identity import accelerator_identity, code_identity
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


def _write_json_atomic(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def benchmark_training(
    cfg: RootConfig,
    *,
    warmup: int = 1,
    repeats: int = 5,
    checkpoint: str | Path | None = None,
    initial_model_state: Mapping[str, torch.Tensor] | None = None,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")
    if checkpoint is not None and initial_model_state is not None:
        raise ValueError(
            "checkpoint resume and model-only initialization are mutually exclusive"
        )
    initializer_sha256 = (
        None
        if initial_model_state is None
        else model_state_sha256(initial_model_state)
    )
    samples = []
    benchmark_device: torch.device | None = None
    progress = None if progress_path is None else Path(progress_path)
    if progress is not None:
        _write_json_atomic(
            {
                "schema": "pbg-queued-benchmark-progress-v1",
                "status": "starting",
                "warmup": warmup,
                "repeats": repeats,
                "completed_warmups": 0,
                "completed_repeats": 0,
                "samples": [],
            },
            progress,
        )
    for index in range(warmup + repeats):
        # Every repeat starts from the identical model/optimizer/RNG/run state.
        # Warmup compiles the same production path but is never a measured row.
        if progress is not None:
            _write_json_atomic(
                {
                    "schema": "pbg-queued-benchmark-progress-v1",
                    "status": "running",
                    "active_kind": "warmup" if index < warmup else "repeat",
                    "active_index": index if index < warmup else index - warmup,
                    "warmup": warmup,
                    "repeats": repeats,
                    "completed_warmups": min(index, warmup),
                    "completed_repeats": max(0, index - warmup),
                    "samples": samples,
                },
                progress,
            )
        session = TrainingSession(
            cfg,
            resume_from=checkpoint,
            benchmark_telemetry=True,
            initial_model_state=initial_model_state,
        )
        try:
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
            if index < warmup:
                del result
                if progress is not None:
                    _write_json_atomic(
                        {
                            "schema": "pbg-queued-benchmark-progress-v1",
                            "status": "running",
                            "warmup": warmup,
                            "repeats": repeats,
                            "completed_warmups": index + 1,
                            "completed_repeats": 0,
                            "samples": samples,
                        },
                        progress,
                    )
                continue
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
        except BaseException as error:
            if progress is not None:
                _write_json_atomic(
                    {
                        "schema": "pbg-queued-benchmark-progress-v1",
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "warmup": warmup,
                        "repeats": repeats,
                        "completed_repeats": len(samples),
                        "samples": samples,
                    },
                    progress,
                )
            raise
        finally:
            session.close()
            del session
        if progress is not None:
            _write_json_atomic(
                {
                    "schema": "pbg-queued-benchmark-progress-v1",
                    "status": "running",
                    "warmup": warmup,
                    "repeats": repeats,
                    "completed_warmups": warmup,
                    "completed_repeats": len(samples),
                    "samples": samples,
                },
                progress,
            )
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
    report = {
        "schema": "pbg-queued-benchmark-v1",
        "mode": dict(cfg.mode),
        "scientific_config": cfg.scientific_mapping(),
        "execution": cfg.execution.mapping(),
        "identity": {
            "logical_rng_version": LOGICAL_RNG_VERSION,
            "checkpoint_sha256": (
                None if checkpoint is None else sha256_file(Path(checkpoint))
            ),
            "initializer_model_sha256": initializer_sha256,
            "code": code_identity(),
            "accelerator": accelerator_identity(
                cfg.scientific.common.game.rank_count
            ),
            "required_sample_metrics": list(_REQUIRED_SAMPLE_METRICS),
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
    if progress is not None:
        _write_json_atomic(
            {
                "schema": "pbg-queued-benchmark-progress-v1",
                "status": "complete",
                "warmup": warmup,
                "repeats": repeats,
                "completed_warmups": warmup,
                "completed_repeats": len(samples),
                "samples": samples,
                "median": median,
            },
            progress,
        )
    return report


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

    def canonical(value: object) -> object:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )

    for identity in ("mode", "scientific_config", "execution", "identity"):
        if canonical(current.get(identity)) != canonical(
            baseline.get(identity)
        ):
            failures.append(f"{identity} identity differs")
    for report_name, report in (("current", current), ("baseline", baseline)):
        for index, sample in enumerate(report.get("samples", ())):
            missing = set(_REQUIRED_SAMPLE_METRICS).difference(sample)
            if missing:
                failures.append(
                    f"{report_name} sample {index} misses {sorted(missing)}"
                )
    current_work = {
        json.dumps(
            canonical(
                (
                    sample["logical_work_before"],
                    sample["logical_work_after"],
                )
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        for sample in current.get("samples", ())
        if {"logical_work_before", "logical_work_after"} <= set(sample)
    }
    baseline_work = {
        json.dumps(
            canonical(
                (
                    sample["logical_work_before"],
                    sample["logical_work_after"],
                )
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        for sample in baseline.get("samples", ())
        if {"logical_work_before", "logical_work_after"} <= set(sample)
    }
    if len(current_work) != 1:
        failures.append("current samples do not share one logical-work identity")
    if len(baseline_work) != 1:
        failures.append("baseline samples do not share one logical-work identity")
    if current_work != baseline_work:
        failures.append("logical work differs")
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
