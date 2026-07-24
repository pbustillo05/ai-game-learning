"""Replica-batched single-GPU training entry point."""

from __future__ import annotations

import argparse

from pbg.config import load_config
from pbg.core.replica_training import run_replica_training
from pbg.runtime import build_game_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train independent Normal/RB replicas in one CUDA process"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--resume-root", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--profile-kernel-iterations",
        type=int,
        nargs="*",
        default=(),
    )
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--verify-legacy-k1", action="store_true")
    parser.add_argument("--learner-benchmark-warmup", type=int, default=None)
    parser.add_argument("--learner-benchmark-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, hardware_profile=args.hardware_profile)
    if args.out_root is not None:
        cfg.training.out_dir = args.out_root
    run_replica_training(
        cfg,
        replicas=args.replicas,
        bundle_builder=build_game_bundle,
        resume_root=args.resume_root,
        max_iterations=args.max_iterations,
        profile_kernel_iterations=frozenset(
            args.profile_kernel_iterations
        ),
        write_checkpoints=not args.no_checkpoints,
        verify_legacy_k1=args.verify_legacy_k1,
        learner_benchmark_warmup=args.learner_benchmark_warmup,
        learner_benchmark_steps=args.learner_benchmark_steps,
    )


if __name__ == "__main__":
    main()
