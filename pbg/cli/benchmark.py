"""CLI for comparative three-mode performance profiles and baseline gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbg.benchmark import MODES, benchmark_modes, compare_benchmark
from pbg.core.config import load_config


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
    parser.add_argument("--sequential-starts", type=int, default=None)
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--queue-batch-rows", type=int, default=None)
    parser.add_argument(
        "--queue-packing",
        choices=("fixed_family_quantum", "actual_cost_greedy"),
        default=None,
    )
    parser.add_argument("--queue-admission-size", type=int, default=None)
    parser.add_argument("--arena-chunk-nodes", type=int, default=None)
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
    if args.sequential_starts is not None:
        cfg.rollout.sequential_starts = args.sequential_starts
    if args.node_budget is not None:
        cfg.rollout.node_budget = args.node_budget
    if args.queue_batch_rows is not None:
        cfg.rollout.queue_batch_rows = args.queue_batch_rows
    if args.queue_packing is not None:
        cfg.rollout.queue_packing = args.queue_packing
    if args.queue_admission_size is not None:
        cfg.rollout.queue_admission_size = args.queue_admission_size
    if args.arena_chunk_nodes is not None:
        cfg.rollout.arena_chunk_nodes = args.arena_chunk_nodes
    if args.completion_samples is not None:
        cfg.completion.n_samples = args.completion_samples
    if args.completion_backend is not None:
        cfg.completion.backend = args.completion_backend
    if args.minibatch_size is not None:
        cfg.training.minibatch_size = args.minibatch_size
    if args.gd_steps is not None:
        cfg.training.gd_steps_per_iter = args.gd_steps
    result = benchmark_modes(
        cfg,
        args.modes or MODES,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        checkpoint=args.checkpoint,
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
