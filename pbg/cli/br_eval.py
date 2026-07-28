"""Command-line entry point for paired BR-probe evaluation."""

from __future__ import annotations

import argparse
import json

from pbg.eval.br_probe import evaluate_br_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired queued best-response evaluation"
    )
    parser.add_argument("--probe", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--hands", required=True, type=int)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    report = evaluate_br_probe(
        args.probe,
        target_artifact=args.target,
        n_hands=args.hands,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report.asdict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
