#!/usr/bin/env python3
"""Reduce and plot the 64-seed pre/post-beta size experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CAPS = {
    "total": 150_000,
    "pre": 2_000,
    "post": 48_000,
    "leaf": 80_000,
    "player": 50_000,
}
SAFE_TREE_SIZE = 120_000


def _summary(values: list[int]) -> dict[str, float | int]:
    data = np.asarray(values)
    return {
        "minimum": int(data.min()),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "maximum": int(data.max()),
        "mean": float(data.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    source = Path(args.input_dir).resolve()

    successes: list[dict[str, object]] = []
    for path in sorted(source.glob("seed_[0-9][0-9]_*.json")):
        if path.name.endswith(".failure.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        observability = result["observability"]
        pre = int(observability["decision_counts_by_street"]["preflop"])
        player = int(result["player_nodes"])
        post = player - pre
        total = int(result["sampled_tree_nodes"])
        leaf = int(result["leaves"])
        pre_refill = int(result["pre_refill_nodes"])
        refill_count = int(result["refill_count"])
        record = {
            "seed": int(payload["workload"]["seed"]),
            "total": total,
            "pre": pre,
            "post": post,
            "player": player,
            "leaf": leaf,
            "chance": total - player - leaf,
            "pre_refill_nodes": pre_refill,
            "refill_count": refill_count,
            "underflow_refill": refill_count > 0 or pre_refill < 1_000,
            "final_total_above_safe_tree_size": total > SAFE_TREE_SIZE,
            "overflow": {
                name: value > CAPS[name]
                for name, value in {
                    "total": total,
                    "pre": pre,
                    "post": post,
                    "player": player,
                    "leaf": leaf,
                }.items()
            },
            "wall_seconds": float(result["wall_seconds"]),
            "peak_allocated_bytes": int(
                result["telemetry"]["peak_allocated_bytes"]
            ),
            "peak_reserved_bytes": int(
                result["telemetry"]["peak_reserved_bytes"]
            ),
            "source": path.name,
        }
        successes.append(record)

    failures: list[dict[str, object]] = []
    for path in sorted(source.glob("*.failure.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stderr = str(payload.get("stderr", ""))
        payload["capacity_overflow"] = (
            "RecoverableOverflow" in stderr or "capacity_overflow" in stderr
        )
        payload["source"] = path.name
        failures.append(payload)

    classes = ("total", "pre", "post", "leaf")
    summaries = {
        name: _summary([int(record[name]) for record in successes])
        for name in (*classes, "player", "chance")
    }
    overflow_records = [
        {
            "seed": record["seed"],
            "overflow": record["overflow"],
        }
        for record in successes
        if any(record["overflow"].values())
    ]
    refill_records = [
        {
            "seed": record["seed"],
            "pre_refill_nodes": record["pre_refill_nodes"],
            "refill_count": record["refill_count"],
        }
        for record in successes
        if record["underflow_refill"]
    ]
    safe_tree_records = [
        {
            "seed": record["seed"],
            "total": record["total"],
        }
        for record in successes
        if record["final_total_above_safe_tree_size"]
    ]
    summary = {
        "schema": "pbg-prepost-beta-distribution-summary-v1",
        "workload": {
            "game": "NLHE HU 32 bb, 13 ranks",
            "fixed_beta": {
                "player_pre": 14.0,
                "player_post": 1.58,
                "chance": 2.4,
            },
            "beta_max": {
                "player_pre": 15,
                "player_post": 3,
                "chance": 3,
            },
            "family_pack_cap": 64,
            "minimum_tree_size": 1_000,
            "safe_tree_size": SAFE_TREE_SIZE,
            "scientific_caps": CAPS,
            "seed_generator": "random.Random(20260729).randrange(1, 2**31)",
        },
        "requested_seed_count": 64,
        "success_count": len(successes),
        "failure_count": len(failures),
        "capacity_failure_count": sum(
            bool(record["capacity_overflow"]) for record in failures
        ),
        "successful_overflow_count": len(overflow_records),
        "underflow_refill_count": len(refill_records),
        "final_total_above_safe_tree_size_count": len(safe_tree_records),
        "summaries": summaries,
        "overflow_records": overflow_records,
        "underflow_refill_records": refill_records,
        "final_total_above_safe_tree_size_records": safe_tree_records,
        "failures": failures,
        "records": successes,
        "wall_seconds": {
            "sum": sum(float(record["wall_seconds"]) for record in successes),
            "median": statistics.median(
                float(record["wall_seconds"]) for record in successes
            )
            if successes
            else 0.0,
        },
    }
    (source / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    colors = {
        "total": "#4c78a8",
        "pre": "#7a9e63",
        "post": "#d8893f",
        "leaf": "#9560ad",
    }
    labels = {
        "total": "Total nodes",
        "pre": "Preflop player nodes",
        "post": "Postflop player nodes",
        "leaf": "Leaf nodes",
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for axis, name in zip(axes.flat, classes, strict=True):
        values = [int(record[name]) for record in successes]
        median = float(np.median(values))
        maximum = max(values)
        bins = min(12, max(5, round(np.sqrt(len(values)))))
        axis.hist(values, bins=bins, color=colors[name], alpha=0.95)
        axis.axvline(
            median,
            color="#222222",
            linestyle="--",
            linewidth=1.7,
            label=f"median {median:,.1f}",
        )
        axis.axvline(
            maximum,
            color="#b22222",
            linestyle=":",
            linewidth=1.7,
            label=f"max {maximum:,}",
        )
        axis.axvline(
            CAPS[name],
            color="#555555",
            linestyle="-.",
            linewidth=1.4,
            label=f"cap {CAPS[name]:,}",
        )
        axis.set_title(
            f"{labels[name]} — median {median:,.1f}, max {maximum:,}",
            fontsize=15,
        )
        axis.set_xlabel("nodes per tree")
        axis.set_ylabel("seeds")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=9)
    figure.suptitle(
        "NLHE HU 32 bb, fixed beta pre=14 / post=1.58 / chance=2.4 "
        f"({len(successes)}/64 successful seeds)",
        fontsize=20,
    )
    figure.savefig(source / "node_distribution.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
