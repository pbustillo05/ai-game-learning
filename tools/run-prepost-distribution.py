#!/usr/bin/env python3
"""Run one deterministic shard of the NLHE pre/post-beta size experiment."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--master-seed", type=int, default=20260729)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, shards)")
    if args.count <= 0 or args.shards <= 0:
        parser.error("--count and --shards must be positive")

    generator = random.Random(args.master_seed)
    seeds: list[int] = []
    while len(seeds) < args.count:
        seed = generator.randrange(1, 2**31)
        if seed not in seeds:
            seeds.append(seed)

    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    assigned = list(enumerate(seeds))[args.shard :: args.shards]
    records: list[dict[str, object]] = []
    for sequence, seed in assigned:
        output = destination / f"seed_{sequence:02d}_{seed}.json"
        failure = destination / f"seed_{sequence:02d}_{seed}.failure.json"
        if output.exists():
            records.append(
                {"sequence": sequence, "seed": seed, "status": "existing"}
            )
            continue
        command = [
            "./tools/poker-python",
            "tools/profile-queued-branching.py",
            "--config",
            "configs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing.yaml",
            "--scenario",
            "successful_seed104830",
            "--seed",
            str(seed),
            "--player-pre-beta",
            "14.0",
            "--player-post-beta",
            "1.58",
            "--chance-beta",
            "2.4",
            "--player-pre-beta-max",
            "15",
            "--player-post-beta-max",
            "3",
            "--chance-beta-max",
            "3",
            "--safe-tree-size",
            "120000",
            "--tree-only",
            "--large-tree-manifest",
            "--output",
            str(output),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        elapsed = time.perf_counter() - started
        status = "success" if completed.returncode == 0 else "failed"
        record = {
            "sequence": sequence,
            "seed": seed,
            "status": status,
            "returncode": completed.returncode,
            "launcher_seconds": elapsed,
        }
        records.append(record)
        if completed.returncode:
            failure.write_text(
                json.dumps(
                    {
                        **record,
                        "stderr": completed.stderr,
                        "command": command,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            f"shard={args.shard} sequence={sequence} seed={seed} "
            f"status={status} seconds={elapsed:.1f}",
            flush=True,
        )

    (destination / f"shard_{args.shard}.json").write_text(
        json.dumps(
            {
                "schema": "pbg-prepost-beta-distribution-shard-v1",
                "master_seed": args.master_seed,
                "seed_count": args.count,
                "shard": args.shard,
                "shards": args.shards,
                "seeds": seeds,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
