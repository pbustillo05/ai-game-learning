"""CLI for the staged Rao--Blackwell variance experiment."""

from __future__ import annotations

import argparse
import copy
import gc
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from pbg.core.devices import resolve_execution_device, validate_workload_capabilities

from experiments.rao_blackwell_variance.eval.rao_blackwell_variance import (
    assigned_jobs,
    atomic_json,
    canonical_jobs,
    cpu_oracle,
    ensure_manifest,
    load_experiment_config,
    run_variance_job,
)


def _distributed_identity() -> tuple[int, int, int | None]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None
    return rank, world, local


def _device(argument: str | None) -> torch.device:
    rank, world, local = _distributed_identity()
    del rank
    if world > 1:
        assert local is not None
        if argument != "cuda":
            raise ValueError("distributed variance evaluation requires --device cuda")
        device = resolve_execution_device(
            "cuda", workload="variance evaluation", local_rank=local
        )
    else:
        if argument is None:
            raise ValueError("--device must explicitly select cuda or mps")
        device = resolve_execution_device(argument, workload="variance evaluation")
    validate_workload_capabilities(device, workload="variance evaluation")
    # A torch.device is only an address. CUDA memory APIs require the process's
    # current device to be initialized explicitly, especially under torchrun
    # where LOCAL_RANK differs across workers.
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _linear_fixture(device: torch.device, n_samples: int) -> tuple[Any, dict[str, Any]]:
    import poker_gpu_evaluator as poker

    evaluator = poker.bind(rank_count=4)
    board_row = torch.tensor([36, 37, 38, 39, 40], dtype=torch.uint8)
    combos = evaluator.combo_cards
    legal = torch.tensor(
        [index for index, cards in enumerate(combos.tolist()) if not set(cards).intersection(board_row.tolist())],
        dtype=torch.long,
    )
    ranges = torch.zeros((1, 1, combos.shape[0]), dtype=torch.float32)
    ranges[0, 0, legal[:4]] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = torch.zeros((1, 3, 1, combos.shape[0]), dtype=torch.float32)
    p[0, 0, 0, legal[:4]] = torch.tensor([0.0, 1.0, 2.0, 4.0])
    p[0, 1, 0, legal[:4]] = torch.tensor([2.0, -1.0, 3.0, 0.5])
    p[:, 2] = p[:, 0] + p[:, 1]
    kwargs: dict[str, Any] = {
        "board_cards": board_row[None],
        "range_weights": ranges,
        "stratum_level": torch.full((1, 1), -1, dtype=torch.int32),
        "payoff_baseline": torch.full((1, 1), 1.5, dtype=torch.float32),
        "stratum_amount": torch.zeros((1, 1), dtype=torch.float32),
        "p_coefficient": p,
        "m_coefficient": torch.zeros_like(p),
        "functional_mask": torch.ones((1, 3), dtype=torch.bool),
        "n_samples": int(n_samples),
        "seed": torch.tensor([918273], dtype=torch.int64),
    }
    for name, value in tuple(kwargs.items()):
        if isinstance(value, torch.Tensor) and name != "seed":
            kwargs[name] = value.to(device)
    return evaluator, kwargs


def _jvp_smoke(device: torch.device, rtol: float, atol: float) -> float:
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 4)).to(device)
    x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3) / 7
    parameters = dict(model.named_parameters())
    generator = torch.Generator(device="cpu").manual_seed(7711)
    direction = {
        name: (torch.randint(0, 2, value.shape, generator=generator, dtype=torch.int8).to(device=device, dtype=value.dtype) * 2 - 1)
        for name, value in parameters.items()
    }
    def call(p: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.functional_call(model, p, (x,), strict=True)
    output, tangent = torch.func.jvp(call, (parameters,), (direction,))
    weights = torch.linspace(-1, 1, output.numel(), device=device).reshape_as(output)
    gradients = torch.autograd.grad((model(x) * weights).sum(), tuple(model.parameters()))
    backward_projection = sum((gradient * direction[name]).sum() for (name, _), gradient in zip(model.named_parameters(), gradients, strict=True))
    jvp_projection = (tangent * weights).sum()
    torch.testing.assert_close(jvp_projection, backward_projection, rtol=rtol, atol=atol)
    return float((jvp_projection - backward_projection).abs().detach().cpu())


def run_smoke(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    oracle = cpu_oracle()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    before = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    evaluator, kwargs = _linear_fixture(device, int(config["smoke"]["oracle_samples"]))
    first = evaluator.complete_linear_moments(**kwargs)
    second = evaluator.complete_linear_moments(**kwargs)
    for field in first.__dataclass_fields__:
        torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)
    tolerance = config["numerics"]
    torch.testing.assert_close(
        first.mean[:, 2], first.mean[:, 0] + first.mean[:, 1],
        rtol=float(tolerance["functional_sum_rtol"]), atol=float(tolerance["functional_sum_atol"]),
    )
    if not bool(first.functional_valid.all()) or not bool(torch.isfinite(first.mean).all()):
        raise RuntimeError("linear-moment evaluator returned an invalid smoke result")
    jvp_error = _jvp_smoke(device, float(tolerance["jvp_backward_rtol"]), float(tolerance["jvp_backward_atol"]))
    # Imports, CUDA extension loading, and one-time framework caches are part
    # of worker initialization rather than leaked experiment allocations.
    del first, second, evaluator, kwargs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    pipeline = None
    if device.type == "cuda":
        bounded = copy.deepcopy(config)
        bounded["candidates"] = {
            "N": [int(config["smoke"]["trajectories"])],
            "B": [min(8, int(config["smoke"]["trajectories"]))],
            "n": [int(config["smoke"]["oracle_samples"])],
        }
        bounded["probes"]["count"] = int(config["smoke"]["probes"])
        bounded["streaming"]["probes_per_chunk"] = min(
            int(bounded["streaming"].get("probes_per_chunk", 1)),
            int(bounded["probes"]["count"]),
        )
        bounded["oracle"]["initial_samples"] = int(config["smoke"]["oracle_samples"])
        bounded["streaming"]["trajectories_per_chunk"] = int(config["smoke"]["trajectories"])
        middle = int(config["checkpoints"][len(config["checkpoints"]) // 2])
        pipeline = run_variance_job(bounded, middle, 0, device)
        if not pipeline.get("success") or not pipeline.get("statistics"):
            raise RuntimeError("bounded checkpoint pipeline smoke produced no statistics")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    gc.collect()
    cleanup_delta = 0
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        cleanup_delta = max(0, torch.cuda.memory_allocated(device) - before)
        allowed = float(tolerance["allocated_cleanup_tolerance_mib"]) * 1024**2
        if cleanup_delta > allowed:
            raise RuntimeError(f"CUDA smoke retained {cleanup_delta / 1024**2:.1f} MiB")
    checkpoints = config["checkpoints"]
    superbatches = int(config["public_sampling"]["superbatches"])
    inventory_one = set(assigned_jobs(checkpoints, superbatches, 0, 1))
    inventory_three = set().union(*(assigned_jobs(checkpoints, superbatches, rank, 3) for rank in range(3)))
    if inventory_one != inventory_three or len(inventory_three) != len(canonical_jobs(checkpoints, superbatches)):
        raise AssertionError("worker assignment changes the scientific job inventory")
    return {
        "success": True, "schema_version": "pbg-rb-variance-smoke-v1", "device": str(device),
        "cpu_oracle": oracle, "jvp_backward_absolute_error": jvp_error,
        "peak_allocated_bytes": peak, "cleanup_delta_bytes": cleanup_delta,
        "checkpoint_pipeline": pipeline,
    }


def _unimplemented_phase(phase: str) -> None:
    raise RuntimeError(
        f"phase {phase!r} requires the checkpoint rollout/moment worker; "
        "the statistical core and bounded evaluator smoke are installed, but this phase is not yet available"
    )


def _save_job(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_variance(
    config: dict[str, Any], device: torch.device, manifest: dict[str, Any],
    resume: bool, job_filter: tuple[int, int] | None = None,
) -> None:
    rank, world, _ = _distributed_identity()
    distributed = world > 1
    if distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    jobs = assigned_jobs(
        config["checkpoints"], int(config["public_sampling"]["superbatches"]), rank, world
    )
    if job_filter is not None:
        jobs = tuple(job for job in jobs if job == job_filter)
    checkpoint_hashes: dict[int, str] = {}
    from experiments.rao_blackwell_variance.eval.rao_blackwell_variance import checkpoint_sha256
    for checkpoint in config["checkpoints"]:
        path = Path(config["checkpoint_dir"]) / f"ckpt_{int(checkpoint):07d}.pt"
        checkpoint_hashes[int(checkpoint)] = checkpoint_sha256(path)
    for checkpoint, superbatch in jobs:
        target = Path(config["out_dir"]) / "jobs" / checkpoint_hashes[checkpoint] / f"{superbatch:03d}.pt"
        if target.exists() and resume:
            existing = torch.load(target, map_location="cpu", weights_only=False)
            if (
                existing.get("success") is True
                and existing.get("schema_version") == "pbg-rb-variance-v1"
                and existing.get("manifest_fingerprint") == manifest["fingerprint"]
                and existing.get("checkpoint_sha256") == checkpoint_hashes[checkpoint]
            ):
                continue
        payload = run_variance_job(config, checkpoint, superbatch, device)
        payload["manifest_fingerprint"] = manifest["fingerprint"]
        _save_job(target, payload)
    if distributed:
        torch.distributed.barrier()
    if rank == 0 and job_filter is None:
        inventory = []
        for checkpoint, superbatch in canonical_jobs(config["checkpoints"], int(config["public_sampling"]["superbatches"])):
            target = Path(config["out_dir"]) / "jobs" / checkpoint_hashes[checkpoint] / f"{superbatch:03d}.pt"
            if not target.exists():
                raise RuntimeError(f"missing variance job {target}")
            payload = torch.load(target, map_location="cpu", weights_only=False)
            if payload.get("success") is not True or payload.get("manifest_fingerprint") != manifest["fingerprint"]:
                raise RuntimeError(f"invalid variance job {target}")
            inventory.append({"checkpoint": checkpoint, "superbatch": superbatch, "path": str(target)})
        atomic_json(Path(config["out_dir"]) / "variance_inventory.json", {"success": True, "jobs": inventory})


def run_analysis(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    from experiments.rao_blackwell_variance.eval.rao_blackwell_variance import (
        SufficientStatistics, checkpoint_sha256, variance_cube,
    )
    out = Path(config["out_dir"])
    summary: dict[str, Any] = {
        "schema_version": "pbg-rb-variance-analysis-v1",
        "manifest_fingerprint": manifest["fingerprint"],
        "checkpoints": {},
    }
    cube_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    probes = int(config["probes"]["count"])
    superbatches = int(config["public_sampling"]["superbatches"])
    n_values = [int(value) for value in config["candidates"]["N"]]
    for checkpoint in config["checkpoints"]:
        digest = checkpoint_sha256(Path(config["checkpoint_dir"]) / f"ckpt_{int(checkpoint):07d}.pt")
        jobs = []
        for superbatch in range(superbatches):
            path = out / "jobs" / digest / f"{superbatch:03d}.pt"
            if not path.exists():
                raise RuntimeError(f"missing variance job {path}")
            job = torch.load(path, map_location="cpu", weights_only=False)
            if job.get("success") is not True or job.get("manifest_fingerprint") != manifest["fingerprint"]:
                raise RuntimeError(f"invalid variance job {path}")
            jobs.append(job)
        by_n: dict[int, dict[str, float]] = {}
        maximum_n = max(n_values)
        rollout_by_probe = []
        for probe in range(probes):
            pooled = SufficientStatistics()
            for job in jobs:
                pooled.merge(SufficientStatistics.fromdict(job["statistics"][f"N{maximum_n}/block0/probe{probe}"]))
            rollout_by_probe.append(pooled.components()["rollout_1"])
        rollout = sum(rollout_by_probe) / probes
        for n_value in n_values:
            endpoints: list[dict[str, float]] = []
            blocks = maximum_n // n_value
            for job in jobs:
                for block in range(blocks):
                    for probe in range(probes):
                        stats = SufficientStatistics.fromdict(job["statistics"][f"N{n_value}/block{block}/probe{probe}"])
                        endpoints.append(stats.components())
            component = {
                "rollout_1": rollout,
                "minibatch_N_1": sum(row["minibatch_N_1"] for row in endpoints) / len(endpoints),
                "montecarlo_N_inf": sum(row["montecarlo_N_inf"] for row in endpoints) / len(endpoints),
                "montecarlo_N_1": sum(row["montecarlo_N_1"] for row in endpoints) / len(endpoints),
            }
            if component["minibatch_N_1"] < 0 or component["rollout_1"] < 0:
                raise RuntimeError(f"oracle-noise correction is negative at checkpoint {checkpoint}, N={n_value}")
            by_n[n_value] = component
            component_rows.append({"checkpoint": int(checkpoint), "N": n_value, **component})
        rows = variance_cube(by_n, config["candidates"])
        for row in rows:
            row["checkpoint"] = int(checkpoint)
        cube_rows.extend(rows)
        summary["checkpoints"][str(checkpoint)] = {"components": by_n, "candidate_count": len(rows)}
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for path, rows in ((tables / "components.csv", component_rows), (tables / "variance_cube.csv", cube_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    atomic_json(out / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rao--Blackwell gradient-variance experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "calibrate-oracle", "variance", "performance", "validate", "analyze", "all"))
    parser.add_argument("--device", required=True, choices=("cuda", "mps"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--job", default=None, metavar="CHECKPOINT:SUPERBATCH",
        help="run only one canonical variance job (calibration/resume filter)",
    )
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    job_filter = None
    if args.job is not None:
        try:
            checkpoint_text, superbatch_text = args.job.split(":", 1)
            job_filter = (int(checkpoint_text), int(superbatch_text))
        except (ValueError, TypeError) as exc:
            parser.error("--job must have the form CHECKPOINT:SUPERBATCH")
        expected = set(canonical_jobs(config["checkpoints"], int(config["public_sampling"]["superbatches"])))
        if job_filter not in expected:
            parser.error("--job does not name a configured canonical job")
    rank, world, _ = _distributed_identity()
    manifest = ensure_manifest(config, {"device": args.device, "world_size": world}, resume=args.resume)
    device = _device(args.device)
    if args.phase == "smoke":
        result = run_smoke(config, device)
        result["manifest_fingerprint"] = manifest["fingerprint"]
        if rank == 0:
            atomic_json(Path(config["out_dir"]) / "smoke.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.phase == "variance":
        run_variance(config, device, manifest, args.resume, job_filter)
        return
    if args.phase == "analyze":
        if rank == 0:
            result = run_analysis(config, manifest)
            print(json.dumps(result, indent=2, sort_keys=True))
        return
    _unimplemented_phase(args.phase)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"rao-blackwell-variance: {exc}", file=sys.stderr)
        raise
