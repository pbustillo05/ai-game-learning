"""Two-launch distributed checkpoint smoke for Phase 10 validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pbg.config import load_config_mapping
from pbg.runtime import TrainingSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    source = Path("configs/train/nlhe_hu_8bb_normal_queued_smoke.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["execution"]["device"] = "cuda"
    raw["execution"]["distributed"] = True
    raw["execution"]["iterations"] = args.iterations
    raw["execution"]["output_dir"] = args.out
    raw["execution"]["run_id"] = "nlhe_hu_8bb_normal_distributed_smoke"
    TrainingSession(
        load_config_mapping(raw),
        resume_from=args.resume,
    ).run()


if __name__ == "__main__":
    main()
