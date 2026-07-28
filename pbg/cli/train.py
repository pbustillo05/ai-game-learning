"""Strict queued training entry point."""

from __future__ import annotations

import argparse

from pbg.config import load_config
from pbg.runtime import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PBG model")
    parser.add_argument("--config", required=True, help="scientific YAML configuration")
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint in training.out_dir from which to resume safely",
    )
    args = parser.parse_args()
    train(
        load_config(args.config),
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
