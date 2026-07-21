#!/usr/bin/env python3
"""Measure fixed-intensity branched rollout geometry from a frozen checkpoint."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from pbg.config import load_config
from pbg.core.branching import BranchingCutoffReached, sample_branched_tree
from pbg.core.config import seed_all
from pbg.runtime import build_game_bundle


def _checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"checkpoint {path} has no model state")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return model, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "iteration": int(payload.get("iteration", -1)),
        "run_id": payload.get("run_id"),
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _summary(samples: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in samples[0]:
        if not all(isinstance(sample[key], (int, float)) for sample in samples):
            continue
        values = [sample[key] for sample in samples]
        result[key] = {
            "mean": float(statistics.fmean(values)),
            "p05": _quantile(values, 0.05),
            "p50": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
            "max": float(max(values)),
        }
    return result


def _street_geometry(
    backend: object, state_refs: torch.Tensor, owners: torch.Tensor
) -> dict[str, float]:
    states = getattr(backend, "states", None)
    street_rows = getattr(states, "street", None)
    if not isinstance(street_rows, torch.Tensor):
        return {}
    street = street_rows[state_refs.to(torch.int64)].to(torch.int64).clamp(0, 3)
    all_counts = torch.bincount(street, minlength=4)
    decision = (owners >= 0) & (owners < int(backend.n_players))
    decision_counts = torch.bincount(street[decision], minlength=4)
    all_host = all_counts.to("cpu").tolist()
    decision_host = decision_counts.to("cpu").tolist()
    labels = ("preflop", "flop", "turn", "river")
    return {
        **{
            f"nodes_{label}": float(all_host[index])
            for index, label in enumerate(labels)
        },
        **{
            f"decision_nodes_{label}": float(decision_host[index])
            for index, label in enumerate(labels)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beta", type=float, action="append", required=True)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--start-repeat", type=int, default=0)
    parser.add_argument("--node-budget", type=int, default=4096)
    parser.add_argument("--upper-cutoff", type=int, default=None)
    parser.add_argument("--stop-cutoff", type=int, default=None)
    parser.add_argument("--lower-cutoff", type=int, default=None)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--hard-memory-bytes", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats <= 0 or args.node_budget <= 0:
        parser.error("repeats and node-budget must be positive")
    if not 0 <= args.start_repeat < args.repeats:
        parser.error("start-repeat must be nonnegative and below repeats")
    if args.hard_memory_bytes is not None and args.hard_memory_bytes <= 0:
        parser.error("hard-memory-bytes must be positive")
    if args.upper_cutoff is not None and not 1 < args.upper_cutoff < args.node_budget:
        parser.error("upper-cutoff must be strictly between one and node-budget")
    if args.stop_cutoff is not None and not 1 < args.stop_cutoff < args.node_budget:
        parser.error("stop-cutoff must be strictly between one and node-budget")
    if args.lower_cutoff is not None and not 0 < args.lower_cutoff < args.node_budget:
        parser.error("lower-cutoff must be positive and below node-budget")
    if (
        args.lower_cutoff is not None
        and args.upper_cutoff is not None
        and args.lower_cutoff >= args.upper_cutoff
    ):
        parser.error("lower-cutoff must be below upper-cutoff")
    if any(not math.isfinite(beta) or beta < 1.0 for beta in args.beta):
        parser.error("every beta must be finite and at least one")

    cfg = load_config(args.config, hardware_profile=args.hardware_profile)
    cfg = copy.deepcopy(cfg)
    cfg.training.mode = "branched_rao_blackwell"
    cfg.training.reproducibility = "strict"
    cfg.rollout.roots_per_gpu = 1
    cfg.rollout.sequential_roots_per_gpu_initial = 1
    cfg.rollout.sequential_roots_per_gpu_minimum = 1
    cfg.rollout.sequential_roots_per_gpu_maximum = 1
    cfg.rollout.node_budget = args.node_budget
    cfg.rollout.beta = {"default": 1.0}
    cfg.rollout.branching_mode = "fixed"
    cfg.rollout.adaptive_kkt = None
    # Sequential checkpoints legitimately have no T/C branching heads.  Keep
    # those heads enabled so geometry benchmarks include production branching
    # compute, while requiring every trained trunk parameter from the
    # checkpoint below.
    checkpoint_path = Path(args.checkpoint)
    model_state, checkpoint_metadata = _checkpoint(checkpoint_path)
    cfg.model["use_branching_heads"] = True
    cfg.loss.c_t = cfg.loss.c_t or 1.0
    cfg.loss.c_c = cfg.loss.c_c or 1.0
    cfg.rollout.branching_upper_cutoff_nodes = args.upper_cutoff
    cfg.rollout.branching_stop_cutoff_nodes = args.stop_cutoff
    cfg.completion.n_samples = cfg.completion.n_samples or 64

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the branching sweep requires an available CUDA device")
    torch.cuda.set_device(device)
    seed_all(args.seed, 0, device)
    bundle = build_game_bundle(cfg, device)
    bundle.model.to(device)
    incompatible = bundle.model.load_state_dict(model_state, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if unexpected or any("branching_heads" not in key for key in missing):
        raise RuntimeError(
            "checkpoint load may omit only newly initialized branching heads; "
            f"missing={missing}, unexpected={unexpected}"
        )
    checkpoint_metadata["randomly_initialized_branching_head_keys"] = list(missing)
    bundle.model.eval()
    generator, _ = seed_all(args.seed, 0, device)
    backend = bundle.rollout_backend
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample_destination = destination.with_suffix(destination.suffix + ".samples.jsonl")
    sample_destination.write_text("", encoding="utf-8")

    records: list[dict[str, object]] = []
    for beta in args.beta:
        backend.beta_table = {"default": float(beta)}
        samples: list[dict[str, object]] = []
        for repeat in range(args.start_repeat, args.repeats):
            repeat_seed = args.seed + 104_729 * repeat
            generator.manual_seed(repeat_seed)
            cfg.rollout.logical_seed = repeat_seed
            metrics = getattr(backend, "execution_metrics", None)
            if metrics is not None:
                metrics.clear()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            started = time.perf_counter()
            diagnostics: dict[str, int] = {}
            cutoff = None
            tree = reach = policy = None
            try:
                with torch.inference_mode():
                    tree, reach, policy = sample_branched_tree(
                        bundle, cfg, generator, iteration=0, diagnostics=diagnostics
                    )
            except BranchingCutoffReached as caught:
                cutoff = caught
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            execution = {} if metrics is None else metrics.snapshot()
            if cutoff is None:
                assert tree is not None and reach is not None and policy is not None
                degree = tree.child_ptr[1:] - tree.child_ptr[:-1]
                leaf = degree == 0
                leaf_weight = tree.path_weight[leaf].double()
                leaf_depth = tree.depth_player[leaf].double()
                weight_total = float(leaf_weight.sum())
                weighted_depth = float((leaf_weight * leaf_depth).sum() / weight_total)
                geometry = {
                    "nodes": float(tree.owner.numel()),
                    "decisions": float(policy.node_ids.numel()),
                    "leaves": float(leaf.sum()),
                    "max_offspring": float(degree.max()),
                    "max_player_depth": float(tree.depth_player.max()),
                    "weighted_leaf_player_depth": weighted_depth,
                    "leaf_weight_sum": weight_total,
                    "stop_cutoff_reached": 0.0,
                    "pending_nodes": 0.0,
                    "queued_nodes": 0.0,
                    "frontier_nodes": 0.0,
                    "family_queue_nodes": {},
                    **_street_geometry(backend, tree.state_ref, tree.owner),
                }
            else:
                geometry = {
                    "nodes": float(cutoff.node_count),
                    "decisions": float(cutoff.completed_decisions),
                    "leaves": float(cutoff.completed_leaves),
                    "max_offspring": math.nan,
                    "max_player_depth": math.nan,
                    "weighted_leaf_player_depth": math.nan,
                    "leaf_weight_sum": math.nan,
                    "stop_cutoff_reached": 1.0,
                    "pending_nodes": float(cutoff.pending_nodes),
                    "queued_nodes": float(cutoff.queued_nodes),
                    "frontier_nodes": float(cutoff.frontier_nodes),
                    "family_queue_nodes": cutoff.family_queue_nodes,
                    **_street_geometry(
                        backend, cutoff.state_refs, cutoff.owners
                    ),
                }
            sample = {
                    **geometry,
                    "execution": execution,
                    "rollout_seconds": elapsed,
                    "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
                    "baseline_allocated_bytes": float(baseline_allocated),
                    "baseline_reserved_bytes": float(baseline_reserved),
                    "peak_allocated_delta_bytes": float(
                        torch.cuda.max_memory_allocated(device) - baseline_allocated
                    ),
                    "model_rows_max": execution.get(
                        "execution_model_rows_max", 0.0
                    ),
                    "padded_tokens_max": execution.get(
                        "execution_padded_tokens_max", 0.0
                    ),
                    "padded_token_pairs_max": execution.get(
                        "execution_padded_token_pairs_max", 0.0
                    ),
                    "queue_depth_max": execution.get(
                        "execution_queue_depth_max", 0.0
                    ),
                    "upper_cutoff_activated": float(
                        diagnostics["upper_cutoff_batches"] > 0
                    ),
                    "upper_cutoff_batches": float(
                        diagnostics["upper_cutoff_batches"]
                    ),
                    "upper_cutoff_first_node_count": float(
                        diagnostics["upper_cutoff_first_node_count"]
                    ),
                    "upper_cutoff_avoided_children": float(
                        diagnostics["upper_cutoff_avoided_children"]
                    ),
                    "below_lower_cutoff": float(
                        args.lower_cutoff is not None
                        and tree.owner.numel() < args.lower_cutoff
                    ),
                }
            samples.append(sample)
            with sample_destination.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"beta": beta, "repeat": repeat, **sample},
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(
                f"sample beta={beta:.4f} repeat={repeat} "
                f"nodes={int(sample['nodes'])} "
                f"stopped={int(sample['stop_cutoff_reached'])} "
                f"pending={int(sample['pending_nodes'])} "
                f"cutoff={int(diagnostics['upper_cutoff_batches'] > 0)} "
                f"peak_allocated={torch.cuda.max_memory_allocated(device)} "
                f"peak_reserved={torch.cuda.max_memory_reserved(device)}",
                flush=True,
            )
            if args.hard_memory_bytes is not None and max(
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            ) > args.hard_memory_bytes:
                raise RuntimeError(
                    "measured CUDA peak exceeded hard-memory-bytes="
                    f"{args.hard_memory_bytes}"
                )
            if tree is not None:
                del degree, leaf, leaf_depth, leaf_weight
            del policy, reach, tree
        records.append(
            {
                "beta": beta,
                "samples": samples,
                "summary": _summary(samples),
            }
        )

    props = torch.cuda.get_device_properties(device)
    output = {
        "schema": "pbg-fixed-branching-sweep-v1",
        "checkpoint": checkpoint_metadata,
        "game": {
            "n_players": cfg.game.n_players,
            "stack_bb": cfg.game.stack_bb,
            "rank_count": cfg.game.rank_count,
        },
        "device": {
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
        },
        "node_budget": args.node_budget,
        "analytical_rollout_memory_plan": asdict(backend.rollout_memory_plan),
        "analytical_rollout_peak_bytes": backend.rollout_memory_plan.peak_bytes,
        "upper_cutoff": args.upper_cutoff,
        "stop_cutoff": args.stop_cutoff,
        "lower_cutoff": args.lower_cutoff,
        "repeats": args.repeats,
        "start_repeat": args.start_repeat,
        "hard_memory_bytes": args.hard_memory_bytes,
        "seed": args.seed,
        "records": records,
    }
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for record in records:
        node = record["summary"]["nodes"]
        print(
            f"beta={record['beta']:.4f} "
            f"nodes p50={node['p50']:.1f} p95={node['p95']:.1f} "
            f"p99={node['p99']:.1f} max={node['max']:.0f}"
        )


if __name__ == "__main__":
    main()
