"""CUDA capability and throughput probe for replica-batched learner kernels."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from pbg.core.replica_learner import (
    FunctionalReplicaLearner,
    ReplicaTransformUnsupported,
    StackedModuleReplicas,
)
from pbg.games.poker.model.primitives import MHA


def _command_output(arguments: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _source_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    revision = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    status = _command_output(["git", "status", "--porcelain"], cwd=root)
    return {
        "repository": str(root),
        "revision": revision,
        "dirty": None if status is None else bool(status),
    }


def _kernel_count(run_step: Any) -> int:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as profile:
        run_step()
        torch.cuda.synchronize()
    cuda_type = torch.autograd.DeviceType.CUDA
    return sum(
        1 for event in profile.events() if event.device_type == cuda_type
    )


def _candidate(
    *,
    replicas: int,
    batch_rows: int,
    valid_rows: int,
    tokens: int,
    d_model: int,
    n_heads: int,
    attention_backend: str,
    precision: str,
    warmup: int,
    steps: int,
    grad_clip: float,
    profile_kernels: bool,
) -> dict[str, Any]:
    device = torch.device("cuda")
    modules = []
    for lane in range(replicas):
        torch.manual_seed(17_000 + lane)
        module = MHA(d_model, n_heads).to(device)
        module.configure_execution(attention_backend, query_chunk_size=128)
        module.train()
        modules.append(module)
    state = StackedModuleReplicas.from_modules(modules)
    del modules
    learner = FunctionalReplicaLearner(
        state,
        lr=3.0e-4,
        weight_decay=0.01,
        grad_clip=grad_clip,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(91_337)
    shape = (replicas, batch_rows, tokens, d_model)
    query = torch.randn(shape, device=device, generator=generator)
    key_value = torch.randn(shape, device=device, generator=generator)
    key_bias = torch.zeros(
        (replicas, batch_rows, tokens), dtype=torch.float32, device=device
    )
    key_mask = torch.ones(
        (replicas, batch_rows, tokens), dtype=torch.bool, device=device
    )
    row_mask = (
        torch.arange(batch_rows, device=device)[None, :]
        < torch.full((replicas, 1), valid_rows, device=device)
    )

    def loss_fn(
        output: torch.Tensor,
        _query: torch.Tensor,
        _key_value: torch.Tensor,
        _key_bias: torch.Tensor,
        _key_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        per_row = output.float().square().mean(dim=(1, 2))
        selected = torch.where(valid, per_row, torch.zeros_like(per_row))
        return selected.sum() / valid.sum().clamp_min(1)

    autocast_dtype = (
        torch.bfloat16 if precision == "bf16" else torch.float32
    )

    def run_step():
        with torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=precision == "bf16"
        ):
            return learner.step(
                loss_fn,
                (query, key_value, key_bias, key_mask),
                (row_mask,),
            )

    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    for _ in range(warmup):
        outcome = run_step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        outcome = run_step()
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1_000.0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    kernels = _kernel_count(run_step) if profile_kernels else None
    logical_rows = replicas * valid_rows * steps

    return {
        "replicas": replicas,
        "status": "ok",
        "seconds": seconds,
        "measured_steps": steps,
        "logical_rows": logical_rows,
        "logical_rows_per_second": logical_rows / seconds,
        "peak_allocated_bytes": peak_bytes,
        "incremental_peak_bytes": peak_bytes - baseline_bytes,
        "cuda_kernels_per_step": kernels,
        "cuda_kernels_per_logical_row": (
            None if kernels is None else kernels / (replicas * valid_rows)
        ),
        "losses": outcome.losses.tolist(),
        "grad_norm_pre_clip": outcome.optimizer.grad_norm_pre_clip.tolist(),
        "grad_clip_scale": outcome.optimizer.grad_clip_scale.tolist(),
        "optimizer_step_applied": (
            outcome.optimizer.optimizer_step_applied.tolist()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe torch.func replica batching on the exact poker MHA primitive"
        )
    )
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--batch-rows", type=int, default=2)
    parser.add_argument("--valid-rows", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "chunked_dense"),
        default="sdpa",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--minimum-throughput-gain", type=float, default=0.10)
    parser.add_argument("--max-peak-gib", type=float, default=70.0)
    parser.add_argument("--no-profile-kernels", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("replica learner benchmark requires CUDA")
    if (
        any(value <= 0 for value in args.replicas)
        or args.batch_rows <= 0
        or not 0 < args.valid_rows <= args.batch_rows
        or args.tokens <= 0
        or args.d_model <= 0
        or args.heads <= 0
        or args.warmup < 0
        or args.steps <= 0
        or not math.isfinite(args.grad_clip)
        or args.grad_clip <= 0.0
        or not math.isfinite(args.minimum_throughput_gain)
        or args.minimum_throughput_gain < 0.0
        or not math.isfinite(args.max_peak_gib)
        or args.max_peak_gib <= 0.0
    ):
        parser.error("invalid positive benchmark dimensions or step counts")

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    driver = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    driver = None if driver is None else driver.splitlines()[0].strip()
    report: dict[str, Any] = {
        "schema": "pbg-replica-learner-probe-v1",
        "scope": "poker_mha_primitive_not_end_to_end",
        "source": _source_metadata(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": driver,
        "device": properties.name,
        "device_index": device_index,
        "device_uuid": str(properties.uuid),
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "allocation": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_STEP_ID",
                "SLURM_TIMELIMIT",
            )
        },
        "precision": args.precision,
        "attention_backend": args.attention_backend,
        "batch_rows": args.batch_rows,
        "valid_rows": args.valid_rows,
        "tokens": args.tokens,
        "d_model": args.d_model,
        "heads": args.heads,
        "warmup": args.warmup,
        "steps": args.steps,
        "candidates": [],
    }
    for replicas in args.replicas:
        try:
            candidate = _candidate(
                replicas=replicas,
                batch_rows=args.batch_rows,
                valid_rows=args.valid_rows,
                tokens=args.tokens,
                d_model=args.d_model,
                n_heads=args.heads,
                attention_backend=args.attention_backend,
                precision=args.precision,
                warmup=args.warmup,
                steps=args.steps,
                grad_clip=args.grad_clip,
                profile_kernels=not args.no_profile_kernels,
            )
        except ReplicaTransformUnsupported as exc:
            root = exc.__cause__
            candidate = {
                "replicas": replicas,
                "status": "unsupported",
                "error": str(exc),
                "cause": None if root is None else f"{type(root).__name__}: {root}",
            }
        report["candidates"].append(candidate)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.01)
    selected = None
    previous = None
    gates = []
    for candidate in report["candidates"]:
        passed = candidate.get("status") == "ok"
        reasons = []
        if passed and candidate["peak_allocated_bytes"] >= args.max_peak_gib * 2**30:
            passed = False
            reasons.append("peak_memory")
        if passed and not all(candidate["optimizer_step_applied"]):
            passed = False
            reasons.append("nonfinite_lane")
        if passed and previous is not None:
            gain = (
                candidate["logical_rows_per_second"]
                / previous["logical_rows_per_second"]
                - 1.0
            )
            if gain < args.minimum_throughput_gain:
                passed = False
                reasons.append("throughput_gain")
            current_kernels = candidate["cuda_kernels_per_logical_row"]
            previous_kernels = previous["cuda_kernels_per_logical_row"]
            if (
                current_kernels is None
                or previous_kernels is None
                or current_kernels >= previous_kernels
            ):
                passed = False
                reasons.append("kernel_reduction")
        gates.append(
            {
                "replicas": candidate["replicas"],
                "passed": passed,
                "reasons": reasons,
            }
        )
        if candidate.get("status") == "ok":
            previous = candidate
        if passed:
            selected = candidate["replicas"]
    report["selection"] = {
        "replicas": selected,
        "minimum_throughput_gain": args.minimum_throughput_gain,
        "max_peak_gib": args.max_peak_gib,
        "gates": gates,
        "scope": "learner_microbenchmark_only",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
