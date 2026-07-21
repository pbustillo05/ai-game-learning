"""CLI for comparative three-mode performance profiles and baseline gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pbg.benchmark import MODES, benchmark_modes, compare_benchmark
from pbg.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PBG training modes")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional read-only model checkpoint used to initialize every mode",
    )
    parser.add_argument("--hardware-profile", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="explicit torch device (for example cpu or cuda:0); default is auto",
    )
    parser.add_argument("--mode", action="append", choices=MODES, dest="modes")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--roots-per-gpu", type=int, default=None)
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="fixed branching intensity (branched mode only)",
    )
    parser.add_argument(
        "--branching-upper-cutoff-nodes",
        type=int,
        default=None,
        help="suppress further duplication near this fixed-mode node ceiling",
    )
    parser.add_argument("--family-batch-rows", type=int, default=None)
    parser.add_argument("--max-active-trajectories", type=int, default=None)
    parser.add_argument("--storage-page-rows", type=int, default=None)
    parser.add_argument("--completion-samples", type=int, default=None)
    parser.add_argument(
        "--completion-backend", choices=("auto", "cpu", "cuda"), default=None
    )
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--gd-steps", type=int, default=None)
    parser.add_argument("--output", default=None, help="optional JSON result/baseline path")
    parser.add_argument("--baseline", default=None, help="JSON baseline to enforce")
    parser.add_argument("--max-regression", type=float, default=0.20)
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, hardware_profile=args.hardware_profile)
    if args.roots_per_gpu is not None:
        cfg.rollout.roots_per_gpu = args.roots_per_gpu
    if args.beta is not None:
        if not math.isfinite(args.beta) or args.beta < 1.0:
            parser.error("--beta must be finite and at least one")
        cfg.rollout.beta = {"default": args.beta}
    if args.branching_upper_cutoff_nodes is not None:
        if args.branching_upper_cutoff_nodes <= 1:
            parser.error(
                "--branching-upper-cutoff-nodes must be greater than one"
            )
    if args.family_batch_rows is not None:
        cfg.rollout.family_batch_rows = args.family_batch_rows
    if args.max_active_trajectories is not None:
        cfg.rollout.max_active_trajectories = args.max_active_trajectories
    if args.storage_page_rows is not None:
        cfg.rollout.storage_page_rows = args.storage_page_rows
    if args.completion_samples is not None:
        cfg.completion.n_samples = args.completion_samples
    if args.completion_backend is not None:
        cfg.completion.backend = args.completion_backend
    if args.minibatch_size is not None:
        cfg.training.minibatch_size = args.minibatch_size
    if args.gd_steps is not None:
        cfg.training.gd_steps_per_iter = args.gd_steps
    if args.seed is not None:
        cfg.training.seed = args.seed
    result = benchmark_modes(
        cfg,
        args.modes or MODES,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        checkpoint=args.checkpoint,
        branching_upper_cutoff_nodes=args.branching_upper_cutoff_nodes,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        failures = compare_benchmark(
            result,
            baseline,
            max_regression=args.max_regression,
            allow_environment_mismatch=args.allow_environment_mismatch,
        )
        if failures:
            parser.exit(1, "performance gate failed:\n- " + "\n- ".join(failures) + "\n")


if __name__ == "__main__":
    main()
