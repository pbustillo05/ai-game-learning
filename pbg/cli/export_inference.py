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
    args = parser.parse_args()
    print(
        json.dumps(
            export_inference_artifact(args.checkpoint, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
