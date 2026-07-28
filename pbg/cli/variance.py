"""Command-line update-variance measurement through queued training."""

from __future__ import annotations

import argparse
import json

from pbg.config import load_config
from pbg.variance import estimate_update_variance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure frozen-boundary queued update variance"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    args = parser.parse_args()
    print(
        json.dumps(
            estimate_update_variance(
                load_config(args.config),
                repeats=args.repeats,
                checkpoint=args.checkpoint,
                seed_stride=args.seed_stride,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
