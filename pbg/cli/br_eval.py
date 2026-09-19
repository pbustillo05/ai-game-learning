"""Command-line entry point for paired BR-probe evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbg.eval.br_probe import evaluate_br_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired queued best-response evaluation"
    )
    parser.add_argument("--probe", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--hands", required=True, type=int)
    parser.add_argument("--split-namespace", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--logical-deal-start", type=int, default=0)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate_br_probe(
        args.probe,
        target_artifact=args.target,
        n_hands=args.hands,
        split_namespace=args.split_namespace,
        seed=args.seed,
        device=args.device,
        logical_deal_start=args.logical_deal_start,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report.asdict(), destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
