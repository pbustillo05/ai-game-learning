"""Command-line entry point for paired BR-probe evaluation."""

from __future__ import annotations

import argparse
import json

from pbg.eval.br_probe import evaluate_br_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report strict queued paired BR-probe observations"
    )
    parser.add_argument("--probe", nargs="+", required=True)
    args = parser.parse_args()
    reports = [
        evaluate_br_probe(path).asdict()
        for path in args.probe
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
