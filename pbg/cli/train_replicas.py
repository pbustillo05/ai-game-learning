"""Replica training through the same strict queued composition."""

from __future__ import annotations

import argparse

from pbg.config import load_config
from pbg.runtime import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train queued replica lanes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.replicas != cfg.execution.manifest.n_replica:
        parser.error("--replicas must equal run.workload.replicas")
    if args.replicas <= 0:
        parser.error("replica count must be positive")
    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
