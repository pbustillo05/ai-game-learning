"""Hardware-profile-aware training entry point."""

from __future__ import annotations

import argparse

from pbg.core.config import load_config
from pbg.core.training import train


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
    args = parser.parse_args()
    train(load_config(args.config, hardware_profile=args.hardware_profile))


if __name__ == "__main__":
    main()
