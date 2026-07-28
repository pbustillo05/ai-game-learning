"""Export a compact poker policy/value inference artifact."""

from __future__ import annotations

import argparse
import json

from pbg.inference_artifact import export_inference_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export model-only poker inference weights from a checkpoint"
    )
    parser.add_argument("checkpoint", help="training checkpoint to export")
    parser.add_argument("output", help="destination .pt artifact")
    parser.add_argument(
        "--lane",
        type=int,
        help="explicit lane for a queued replica checkpoint",
    )
    args = parser.parse_args()
    report = export_inference_artifact(
        args.checkpoint, args.output, lane=args.lane
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
