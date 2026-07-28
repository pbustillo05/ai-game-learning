"""Export a compact poker policy/value inference artifact."""

from __future__ import annotations

import argparse
import json

from pbg.config import load_config
from pbg.inference_artifact import (
    convert_legacy_inference_artifact,
    export_inference_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export model-only poker inference weights from a checkpoint"
    )
    parser.add_argument("checkpoint", help="training checkpoint to export")
    parser.add_argument("output", help="destination .pt artifact")
    parser.add_argument(
        "--legacy-scientific-config",
        help="offline-convert pbg-inference-v1 with this queued config",
    )
    parser.add_argument(
        "--lane",
        type=int,
        help="explicit lane for a queued replica checkpoint",
    )
    args = parser.parse_args()
    report = (
        convert_legacy_inference_artifact(
            args.checkpoint,
            args.output,
            scientific_config=load_config(
                args.legacy_scientific_config
            ).scientific_mapping(),
        )
        if args.legacy_scientific_config
        else export_inference_artifact(
            args.checkpoint, args.output, lane=args.lane
        )
    )
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
