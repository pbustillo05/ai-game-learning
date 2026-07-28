"""Matched-work benchmark consumer for the public queued runtime."""

from __future__ import annotations

import platform
import statistics
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from pbg.config import RootConfig
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
)


def _code_identity() -> dict[str, str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", "pbg"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
        diff = b""
    return {
        "git_revision": revision,
        "working_tree_sha256": hashlib.sha256(diff).hexdigest(),
    }


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
    for index in range(warmup + repeats):
        # Every repeat starts from the identical model/optimizer/RNG/run state.
        # Warmup compiles the same production path but is never a measured row.
        session = TrainingSession(cfg, resume_from=checkpoint)
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
                }
            )
        del result
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
    properties = torch.cuda.get_device_properties(session.device)
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
            "code": _code_identity(),
            "required_sample_metrics": _REQUIRED_SAMPLE_METRICS,
        },
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
