"""Hardware-profile-aware training entry point."""

from __future__ import annotations

import argparse

from pbg.config import load_config
from pbg.runtime import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PBG model")
    parser.add_argument("--config", required=True, help="scientific YAML configuration")
    parser.add_argument(
        "--hardware-profile",
        default="local_3x2080ti",
        help=(
            "profile name under configs/hardware or an explicit YAML path "
            "(default: local_3x2080ti; Quest must be selected explicitly)"
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint in training.out_dir from which to resume safely",
    )
    args = parser.parse_args()
    train(
        load_config(args.config, hardware_profile=args.hardware_profile),
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
