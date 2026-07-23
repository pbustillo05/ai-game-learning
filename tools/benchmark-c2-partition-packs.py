"""Controlled CUDA gate for C2 dequeue-time postflop compact sizing."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from pbg.benchmark import benchmark_modes
from pbg.config import load_config
from pbg.core.partition import PhysicalPack
from pbg.games.poker.rollout_backend import PokerPhysicalPack, PokerRolloutBackend


def _flop_padded_pack(
    original: Any,
    backend: PokerRolloutBackend,
    partition_id: int,
    states: torch.Tensor,
) -> PhysicalPack:
    realized = original(backend, partition_id, states)
    if partition_id == 0:
        return realized
    width = math.comb(backend.n_cards - 3, 2)
    descriptor = realized.descriptor
    if not isinstance(descriptor, PokerPhysicalPack):
        raise TypeError("poker pack planner returned the wrong descriptor")
    return PhysicalPack(
        rows=realized.rows,
        token_width=width,
        descriptor=replace(descriptor, compact_hand_width=width),
    )


def _timed_variant(
    cfg: Any,
    *,
    original: Any,
    flop_padded: bool,
    warmup: int,
    repeats: int,
    cutoff: int | None,
) -> dict[str, Any]:
    if flop_padded:
        PokerRolloutBackend._physical_pack = (
            lambda self, partition_id, states: _flop_padded_pack(
                original, self, partition_id, states
            )
        )
    else:
        PokerRolloutBackend._physical_pack = original
    return benchmark_modes(
        cfg,
        ("normal", "branched_rao_blackwell"),
        warmup=warmup,
        repeats=repeats,
        device="cuda",
        branching_upper_cutoff_nodes=cutoff,
    )


def _composition_probe(
    cfg: Any,
    *,
    original: Any,
    warmup: int,
    repeats: int,
    cutoff: int | None,
) -> list[dict[str, Any]]:
    observations: list[tuple[int, int, int, int, torch.Tensor]] = []

    def observed(
        backend: PokerRolloutBackend,
        partition_id: int,
        states: torch.Tensor,
    ) -> PhysicalPack:
        pack = original(backend, partition_id, states)
        descriptor = pack.descriptor
        if isinstance(descriptor, PokerPhysicalPack) and partition_id:
            rows = states.to(torch.int64)
            composition = torch.bincount(
                backend.states.street[rows].to(torch.int64), minlength=4
            ).detach()
            observations.append(
                (
                    partition_id,
                    descriptor.minimum_street,
                    descriptor.compact_hand_width,
                    pack.rows,
                    composition,
                )
            )
        return pack

    PokerRolloutBackend._physical_pack = observed
    benchmark_modes(
        cfg,
        ("branched_rao_blackwell",),
        warmup=warmup,
        repeats=repeats,
        device="cuda",
        branching_upper_cutoff_nodes=cutoff,
    )
    combined: Counter[tuple[int, int, int, int, tuple[int, ...]]] = Counter()
    for partition_id, minimum, width, rows, composition in observations:
        combined[
            (
                partition_id,
                minimum,
                width,
                rows,
                tuple(int(value) for value in composition.cpu().tolist()),
            )
        ] += 1
    return [
        {
            "partition_id": key[0],
            "minimum_street": key[1],
            "compact_hand_width": key[2],
            "rows": key[3],
            "street_rows": {
                "preflop": key[4][0],
                "flop": key[4][1],
                "turn": key[4][2],
                "river": key[4][3],
            },
            "packs": count,
        }
        for key, count in sorted(combined.items())
    ]


def _smoke_config(cfg: Any) -> None:
    cfg.game.n_players = 2
    cfg.game.stack_bb = 8.0
    cfg.game.rank_count = 4
    cfg.rollout.sequential_roots_per_gpu_initial = 32
    cfg.rollout.sequential_roots_per_gpu_minimum = 32
    cfg.rollout.sequential_roots_per_gpu_maximum = 32
    cfg.rollout.roots_per_gpu = 32
    cfg.rollout.family_batch_rows = 16
    cfg.rollout.max_active_trajectories = 32
    cfg.rollout.storage_page_rows = 32
    cfg.rollout.queue_capacity_rows = 512
    cfg.rollout.branching_mode = "fixed"
    cfg.rollout.beta = {"default": 1.5}
    cfg.rollout.safe_tree_size = None
    cfg.completion.n_samples = 2
    cfg.model = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers_pre": 1,
        "n_layers_post": 3,
        "m_raise": 2,
        "k_act": 5,
        "max_players": 2,
        "class_embedding_dim": 8,
        "use_branching_heads": True,
    }
    cfg.training.gd_steps_per_iter = 1
    cfg.training.minibatch_size = 16
    cfg.training.amp = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare C2 compact pack sizing with flop-padded post_A packs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--hardware-profile", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--branching-upper-cutoff-nodes", type=int, default=None)
    parser.add_argument("--max-sequential-regression", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--worker",
        choices=("flop-padded", "dequeue-compact", "composition"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.worker is not None:
        cfg = load_config(args.config, hardware_profile=args.hardware_profile)
        if args.smoke:
            _smoke_config(cfg)
        original = PokerRolloutBackend._physical_pack
        try:
            if args.worker == "composition":
                result = _composition_probe(
                    cfg,
                    original=original,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    cutoff=args.branching_upper_cutoff_nodes,
                )
            else:
                result = _timed_variant(
                    cfg,
                    original=original,
                    flop_padded=args.worker == "flop-padded",
                    warmup=args.warmup,
                    repeats=args.repeats,
                    cutoff=args.branching_upper_cutoff_nodes,
                )
        finally:
            PokerRolloutBackend._physical_pack = original
        print(json.dumps(result, sort_keys=True))
        return

    def worker(variant: str) -> Any:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            args.config,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--worker",
            variant,
        ]
        if args.hardware_profile is not None:
            command.extend(("--hardware-profile", args.hardware_profile))
        if args.branching_upper_cutoff_nodes is not None:
            command.extend(
                (
                    "--branching-upper-cutoff-nodes",
                    str(args.branching_upper_cutoff_nodes),
                )
            )
        if args.smoke:
            command.append("--smoke")
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    baseline = worker("flop-padded")
    optimized = worker("dequeue-compact")
    composition = worker("composition")

    def samples(result: dict[str, Any], mode: str) -> list[dict[str, float]]:
        return next(entry for entry in result["modes"] if entry["mode"] == mode)[
            "samples"
        ]

    def aggregate_seconds_per_decision(
        result: dict[str, Any], mode: str
    ) -> float:
        values = samples(result, mode)
        return sum(value["profiled_iteration_seconds"] for value in values) / sum(
            value["decisions"] for value in values
        )

    baseline_branched = aggregate_seconds_per_decision(
        baseline, "branched_rao_blackwell"
    )
    optimized_branched = aggregate_seconds_per_decision(
        optimized, "branched_rao_blackwell"
    )
    baseline_sequential = aggregate_seconds_per_decision(baseline, "normal")
    optimized_sequential = aggregate_seconds_per_decision(optimized, "normal")
    branched_gain = 1.0 - optimized_branched / baseline_branched
    sequential_regression = optimized_sequential / baseline_sequential - 1.0
    equal_decisions = all(
        [
            value["decisions"]
            for value in samples(baseline, mode)
        ]
        == [
            value["decisions"]
            for value in samples(optimized, mode)
        ]
        for mode in ("normal", "branched_rao_blackwell")
    )
    paired_branched_gains = [
        1.0
        - current["profiled_iteration_seconds"]
        / reference["profiled_iteration_seconds"]
        for reference, current in zip(
            samples(baseline, "branched_rao_blackwell"),
            samples(optimized, "branched_rao_blackwell"),
            strict=True,
        )
    ]
    passed = (
        equal_decisions
        and branched_gain > 0.0
        and sequential_regression <= args.max_sequential_regression
    )
    result = {
        "schema": "pbg-c2-pack-benchmark-v1",
        "passed": passed,
        "equal_realized_decisions": equal_decisions,
        "branched_end_to_end_gain": branched_gain,
        "paired_branched_end_to_end_gains": paired_branched_gains,
        "sequential_regression": sequential_regression,
        "max_sequential_regression": args.max_sequential_regression,
        "street_composition": composition,
        "flop_padded": baseline,
        "dequeue_compact": optimized,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        Path(args.output).write_text(encoded, encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
