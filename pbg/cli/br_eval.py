"""Command-line entry point for paired BR-probe evaluation."""

from __future__ import annotations

import argparse
import json

from pbg.eval.br_probe import evaluate_br_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BR-probe checkpoints")
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--hardware-profile", default="local_3x2080ti")
    parser.add_argument("--hands", type=int, required=True)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="paired rollout rows; defaults to the largest P^2 multiple up to 256",
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    reports = [
        evaluate_br_probe(
            path,
            target_checkpoint=args.target,
            hardware_profile=args.hardware_profile,
            n_hands=args.hands,
            batch_size=args.batch_size,
            seed=args.seed,
        ).asdict()
        for path in args.probe
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
