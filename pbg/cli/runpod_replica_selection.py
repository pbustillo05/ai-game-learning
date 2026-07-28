"""Deadline-bounded H100 replica run using the production queued consumer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pbg.config import load_config
from pbg.runtime import TrainingSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Run queued H100 replica selection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--deadline-minutes", type=float, required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.deadline_minutes <= 0:
        parser.error("--deadline-minutes must be positive")
    cfg = load_config(args.config)
    if cfg.execution.manifest.n_replica != args.replicas or args.replicas <= 1:
        parser.error("replica count must match the strict manifest and exceed one")
    started = time.monotonic()
    session = TrainingSession(cfg, resume_from=args.resume)
    completed = 0
    while (
        min(state.iteration for state in session.run_states)
        < cfg.execution.iterations
        and time.monotonic() - started < args.deadline_minutes * 60
    ):
        iteration = min(state.iteration for state in session.run_states)
        if cfg.mode["rollout_mode"] == "sequential":
            session._run_sequential(iteration)
        else:
            session._run_branched(iteration)
        completed += 1
    checkpoint = cfg.execution.output_dir / "selection_final.pt"
    digest = session.checkpoint(checkpoint)
    report = {
        "schema": "pbg-queued-runpod-selection-v1",
        "replicas": args.replicas,
        "completed_iterations": completed,
        "elapsed_seconds": time.monotonic() - started,
        "deadline_exhausted": (
            time.monotonic() - started >= args.deadline_minutes * 60
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "lane_iterations": [state.iteration for state in session.run_states],
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
