"""Matched-work benchmark consumer for the public queued runtime."""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from pbg.config import RootConfig
from pbg.runtime import TrainingSession


def benchmark_training(
    cfg: RootConfig,
    *,
    warmup: int = 1,
    repeats: int = 5,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")
    session = TrainingSession(cfg, resume_from=checkpoint)
    samples = []
    for index in range(warmup + repeats):
        iteration = session.run_states[0].iteration
        torch.cuda.synchronize(session.device)
        before_allocated = torch.cuda.memory_allocated(session.device)
        started = time.perf_counter()
        result = (
            session._run_sequential(iteration)
            if cfg.mode["rollout_mode"] == "sequential"
            else session._run_branched(iteration)
        )
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
            samples.append(
                {
                    "seconds": elapsed,
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
                }
            )
        del result
    numeric = set.intersection(*(set(sample) for sample in samples))
    median = {
        key: float(statistics.median(sample[key] for sample in samples))
        for key in sorted(numeric)
    }
    properties = torch.cuda.get_device_properties(session.device)
    return {
        "schema": "pbg-queued-benchmark-v1",
        "mode": dict(cfg.mode),
        "scientific_config": cfg.scientific_mapping(),
        "execution": cfg.execution.mapping(),
        "environment": {
            "device": str(session.device),
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
    for identity in ("mode", "scientific_config", "execution"):
        if current.get(identity) != baseline.get(identity):
            failures.append(f"{identity} identity differs")
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
