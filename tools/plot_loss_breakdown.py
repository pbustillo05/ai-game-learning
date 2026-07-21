#!/usr/bin/env python3
"""Plot per-iteration loss components from a PBG metrics JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        metavar="N",
        help="centered rolling-mean window in iterations; 1 disables smoothing",
    )
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")

    rows = [
        json.loads(line)
        for line in args.metrics.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if "iter" in row and "policy" in row]
    iterations = [row["iter"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    series = (
        ("policy", "PPO policy / clip", "#1f77b4"),
        ("value", "Value", "#ff7f0e"),
        ("entropy_loss", "Entropy", "#2ca02c"),
        ("reference_kl", "Reference KL", "#d62728"),
        ("loss", "Total loss", "#000000"),
    )

    def smoothed(values: list[float]) -> list[float]:
        if args.smooth_window == 1:
            return values
        window = args.smooth_window
        radius = window // 2
        result = []
        for index in range(len(values)):
            start = max(0, index - radius)
            end = min(len(values), index + radius + 1)
            result.append(sum(values[start:end]) / (end - start))
        return result

    for key, label, color in series:
        values = smoothed([row[key] for row in rows])
        ax.plot(iterations, values, label=label, color=color, linewidth=1.4 if key != "loss" else 2.2)

    ax.axhline(0.0, color="#777777", linewidth=0.8)
    smoothing = "raw" if args.smooth_window == 1 else f"{args.smooth_window}-iteration rolling mean"
    ax.set_title(f"NLHE HU 32bb sequential-RB loss breakdown ({smoothing})")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Normalized objective contribution")
    ax.legend(loc="best", frameon=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
