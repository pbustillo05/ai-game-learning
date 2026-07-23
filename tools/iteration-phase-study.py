#!/usr/bin/env python3
"""Profile rollout, seal, and completion/target phases from a frozen model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from pbg.config import load_config
from pbg.core.config import seed_all
from pbg.core.training import _build_iteration_data
from pbg.runtime import build_game_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("branched", "sequential"), required=True)
    parser.add_argument("--roots", type=int, default=256)
    parser.add_argument("--beta", type=float, default=2.3)
    parser.add_argument("--safe-tree-size", type=float, default=1000.0)
    parser.add_argument("--family-batch-rows", type=int, default=8)
    parser.add_argument("--execution-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--completion-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=196030)
    parser.add_argument("--model-seed", type=int, default=91301)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = copy.deepcopy(
        load_config(args.config, hardware_profile=args.hardware_profile)
    )
    cfg.training.reproducibility = "strict"
    cfg.rollout.family_batch_rows = args.family_batch_rows
    cfg.rollout.storage_page_rows = 256
    cfg.rollout.execution_memory_bytes = args.execution_memory_bytes
    cfg.completion.n_samples = args.completion_samples
    cfg.model["use_branching_heads"] = True
    cfg.loss.c_t = cfg.loss.c_t or 1.0
    cfg.loss.c_c = cfg.loss.c_c or 1.0
    if args.mode == "branched":
        cfg.training.mode = "branched_rao_blackwell"
        cfg.rollout.roots_per_gpu = 1
        cfg.rollout.sequential_roots_per_gpu_initial = 1
        cfg.rollout.sequential_roots_per_gpu_minimum = 1
        cfg.rollout.sequential_roots_per_gpu_maximum = 1
        cfg.rollout.branching_mode = "fixed"
        cfg.rollout.beta = {"default": args.beta}
        cfg.rollout.adaptive_kkt = None
        cfg.rollout.safe_tree_size = args.safe_tree_size
        cfg.rollout.branching_min_nodes = None
        cfg.rollout.queue_capacity_rows = 4096
    else:
        cfg.training.mode = "rao_blackwell"
        cfg.rollout.roots_per_gpu = args.roots
        cfg.rollout.sequential_roots_per_gpu_initial = args.roots
        cfg.rollout.sequential_roots_per_gpu_minimum = args.roots
        cfg.rollout.sequential_roots_per_gpu_maximum = args.roots
        cfg.rollout.branching_mode = None
        cfg.rollout.beta = None
        cfg.rollout.adaptive_kkt = None
        cfg.rollout.safe_tree_size = None
        cfg.rollout.branching_min_nodes = None

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("phase study requires CUDA")
    torch.cuda.set_device(device)
    seed_all(args.model_seed, 0, device)
    bundle = build_game_bundle(cfg, device)
    bundle.model.to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = bundle.model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys or any(
        "branching_heads" not in key for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "checkpoint load may omit only added branching heads: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    generator, _ = seed_all(args.seed, 0, device)
    cfg.rollout.logical_seed = args.seed
    torch.cuda.synchronize(device)
    data = _build_iteration_data(
        cfg, bundle, generator, profile=True, iteration=0
    )
    torch.cuda.synchronize(device)
    metrics = dict(data.profile_metrics)
    metrics.update(bundle.rollout_backend.execution_metrics.snapshot())
    metrics["mode"] = args.mode
    metrics["completion_n_samples"] = float(args.completion_samples)
    metrics["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
    metrics["peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if key in {
                    "mode",
                    "nodes",
                    "decisions",
                    "leaves",
                    "phase_rollout_seconds",
                    "phase_seal_seconds",
                    "phase_backward_seconds",
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
