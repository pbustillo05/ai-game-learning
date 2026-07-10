"""JSON-lines metric logging with an optional Weights & Biases sink."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


class MetricsLogger:
    """Append scalar training metrics to ``metrics.jsonl``."""

    def __init__(self, out_dir: Path, use_wandb: bool = False):
        """Create the output directory and open its append-only metric stream."""
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = (self._out_dir / "metrics.jsonl").open(
            "a", encoding="utf-8"
        )
        self._wandb: Any | None = None
        if use_wandb:
            try:
                import wandb
            except ImportError:
                wandb = None
            self._wandb = wandb

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Append one JSON object and forward it to wandb when available."""
        if self._stream.closed:
            raise RuntimeError("cannot log with a closed MetricsLogger")
        record = {"step": step, **metrics}
        self._stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._stream.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        """Flush and close the JSON-lines stream; repeated calls are safe."""
        if not self._stream.closed:
            self._stream.close()

