#!/usr/bin/env python3
"""Benchmark current compact sequential-RB rollout at explicit root widths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from pbg.config import load_config
from pbg.core.config import seed_all
from pbg.core.rollout import sample_trees_sequential
from pbg.runtime import build_game_bundle


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    common_keys = set.intersection(*(set(sample) for sample in samples))
    for key in sorted(common_keys):
        values = [float(sample[key]) for sample in samples]
        result[key] = {
            "min": min(values),
            "mean": statistics.fmean(values),
            "variance": statistics.pvariance(values),
            "p05": quantile(values, 0.05),
            "p25": quantile(values, 0.25),
            "p50": quantile(values, 0.50),
            "p75": quantile(values, 0.75),
            "p95": quantile(values, 0.95),
            "max": max(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--roots", type=int, action="append", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--family-batch-rows", type=int, default=8)
    parser.add_argument("--storage-page-rows", type=int, default=256)
    parser.add_argument("--execution-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats <= 0 or any(roots <= 0 for roots in args.roots):
        parser.error("repeats and every root width must be positive")

    cfg = copy.deepcopy(
        load_config(args.config, hardware_profile=args.hardware_profile)
    )
    cfg.training.mode = "rao_blackwell"
    cfg.training.reproducibility = "strict"
    cfg.rollout.branching_mode = None
    cfg.rollout.beta = None
    cfg.rollout.adaptive_kkt = None
    cfg.rollout.safe_tree_size = None
    cfg.rollout.branching_min_nodes = None
    cfg.rollout.family_batch_rows = args.family_batch_rows
    cfg.rollout.storage_page_rows = args.storage_page_rows
    cfg.rollout.execution_memory_bytes = args.execution_memory_bytes
    cfg.completion.n_samples = cfg.completion.n_samples or 64

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint has no model state")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    cfg.model["use_branching_heads"] = True
    cfg.loss.c_t = cfg.loss.c_t or 1.0
    cfg.loss.c_c = cfg.loss.c_c or 1.0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("sequential rollout benchmark requires CUDA")
    torch.cuda.set_device(device)
    seed_all(args.seed, 0, device)
    bundle = build_game_bundle(cfg, device)
    bundle.model.to(device)
    incompatible = bundle.model.load_state_dict(model_state, strict=False)
    if incompatible.unexpected_keys or any(
        "branching_heads" not in key for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "checkpoint load may omit only added branching heads: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    bundle.model.eval()
    generator, _ = seed_all(args.seed, 0, device)
    metrics = bundle.rollout_backend.execution_metrics

    records: list[dict[str, object]] = []
    for roots in args.roots:
        cfg.rollout.roots_per_gpu = roots
        cfg.rollout.sequential_roots_per_gpu_initial = roots
        cfg.rollout.sequential_roots_per_gpu_minimum = roots
        cfg.rollout.sequential_roots_per_gpu_maximum = roots
        samples: list[dict[str, float]] = []
        for repeat in range(args.repeats):
            repeat_seed = args.seed + 104_729 * repeat
            generator.manual_seed(repeat_seed)
            cfg.rollout.logical_seed = repeat_seed
            metrics.clear()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            started = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    enabled=bool(cfg.training.amp),
                    dtype=torch.float16,
                ):
                    sample = sample_trees_sequential(
                        bundle, cfg, generator, iteration=0
                    )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            decisions = int(sample.decision_inputs.node_ids.numel())
            row = {
                "roots": float(roots),
                "nodes": float(sample.node_count),
                "decisions": float(decisions),
                "leaves": float(sample.leaf_source.rows),
                "rollout_seconds": elapsed,
                "nodes_per_second": sample.node_count / elapsed,
                "decisions_per_second": decisions / elapsed,
                "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
                "peak_allocated_delta_bytes": float(
                    torch.cuda.max_memory_allocated(device) - baseline_allocated
                ),
                **metrics.snapshot(),
            }
            samples.append(row)
            print(
                f"roots={roots} repeat={repeat} nodes={sample.node_count} "
                f"decisions={decisions} seconds={elapsed:.3f}",
                flush=True,
            )
            del sample
        records.append(
            {"roots": roots, "samples": samples, "summary": summarize(samples)}
        )

    props = torch.cuda.get_device_properties(device)
    output = {
        "schema": "pbg-sequential-rollout-benchmark-v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": digest,
            "iteration": int(payload.get("iteration", -1)),
            "randomly_initialized_branching_head_keys": list(
                incompatible.missing_keys
            ),
        },
        "device": {"name": props.name, "total_memory_bytes": props.total_memory},
        "family_batch_rows": args.family_batch_rows,
        "execution_memory_bytes": args.execution_memory_bytes,
        "repeats": args.repeats,
        "seed": args.seed,
        "records": records,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
