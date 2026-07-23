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
from pbg.core.branching import (
    BranchingCutoffReached,
    BranchingDiagnosticLimits,
    sample_branched_tree,
)
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
    common_keys = set.intersection(*(set(sample) for sample in samples))
    for key in sorted(common_keys):
        if not all(isinstance(sample[key], (int, float)) for sample in samples):
            continue
        values = [sample[key] for sample in samples]
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            continue
        result[key] = {
            "count": float(len(finite)),
            "min": float(min(finite)),
            "mean": float(statistics.fmean(finite)),
            "variance": float(statistics.pvariance(finite)),
            "stddev": float(statistics.pstdev(finite)),
            "p01": _quantile(finite, 0.01),
            "p05": _quantile(finite, 0.05),
            "p25": _quantile(finite, 0.25),
            "p50": _quantile(finite, 0.50),
            "p75": _quantile(finite, 0.75),
            "p90": _quantile(finite, 0.90),
            "p95": _quantile(finite, 0.95),
            "p99": _quantile(finite, 0.99),
            "max": float(max(finite)),
        }
    return result


def _decision_street_geometry(policy: object) -> dict[str, float]:
    captured = getattr(policy, "decision_inputs", None)
    inputs = getattr(captured, "inputs", None)
    partition_id = getattr(captured, "partition_id", None)
    partition_row = getattr(captured, "partition_row", None)
    if not isinstance(inputs, dict) and not hasattr(inputs, "items"):
        return {}
    street = torch.empty(
        partition_id.numel(), dtype=torch.int64, device=partition_id.device
    )
    for pid, block in inputs.items():
        selected = partition_id == int(pid)
        street[selected] = block["street"][
            partition_row[selected].to(torch.int64)
        ].to(torch.int64)
    decision_counts = torch.bincount(street.clamp(0, 3), minlength=4)
    decision_host = decision_counts.to("cpu").tolist()
    labels = ("preflop", "flop", "turn", "river")
    return {
        f"decision_nodes_{label}": float(decision_host[index])
        for index, label in enumerate(labels)
    }


def _cost_head_calibration(sample: object) -> dict[str, float]:
    """Compare frozen C predictions with realized descendants per child.

    Suppressed decisions are excluded because their realized subtree is not a
    draw from the nominal multiplicity.  A refilled root is excluded because
    its post-refill descendants intentionally do not define the protected C
    target.
    """

    topology = sample.topology
    parent = topology.parent.detach().to("cpu", torch.int64)
    degree = topology.degree.detach().to("cpu", torch.int64)
    subtree = torch.ones(parent.numel(), dtype=torch.float64)
    for node in range(parent.numel() - 1, 0, -1):
        subtree[int(parent[node])] += subtree[node]

    captured = sample.decision_inputs
    prediction = torch.empty(captured.node_ids.numel(), dtype=torch.float64)
    for partition_id, source in captured.decision_sources.items():
        selected = (captured.partition_id == int(partition_id)).to("cpu")
        old_c = source.columns.get("old_c")
        if old_c is None:
            return {}
        prediction[selected] = old_c.detach().to("cpu", torch.float64)

    nodes = sample.decision_node.detach().to("cpu", torch.int64)
    nominal = sample.decision_aux["nominal_child_count"].detach().to("cpu", torch.int64)
    executed = sample.decision_aux["executed_child_count"].detach().to("cpu", torch.int64)
    eligible = (degree[nodes] > 0) & (nominal == executed)
    if topology.pre_refill_node_count is not None:
        eligible &= nodes != 0
    realized = (subtree[nodes] - 1.0) / degree[nodes].clamp_min(1)
    prediction = prediction[eligible]
    realized = realized[eligible]
    if not prediction.numel():
        return {}
    error = prediction - realized
    relative = prediction / realized.clamp_min(1.0)
    log_error = torch.log(prediction.clamp_min(1.0e-30)) - torch.log(
        realized.clamp_min(1.0e-30)
    )
    centered_prediction = prediction - prediction.mean()
    centered_realized = realized - realized.mean()
    denominator = torch.sqrt(
        centered_prediction.square().sum() * centered_realized.square().sum()
    )
    correlation = (
        float((centered_prediction * centered_realized).sum() / denominator)
        if float(denominator) > 0.0
        else math.nan
    )
    return {
        "cost_calibration_rows": float(prediction.numel()),
        "cost_prediction_mean": float(prediction.mean()),
        "cost_realized_mean": float(realized.mean()),
        "cost_prediction_bias": float(error.mean()),
        "cost_prediction_mae": float(error.abs().mean()),
        "cost_prediction_median_ratio": float(relative.median()),
        "cost_prediction_under_fraction": float((prediction < realized).double().mean()),
        "cost_prediction_log_rmse": float(torch.sqrt(log_error.square().mean())),
        "cost_prediction_correlation": correlation,
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
    parser.add_argument("--storage-page-rows", type=int, default=256)
    parser.add_argument("--family-batch-rows", type=int, default=None)
    parser.add_argument("--queue-capacity-rows", type=int, default=4096)
    parser.add_argument("--execution-memory-bytes", type=int, default=None)
    parser.add_argument("--safe-tree-size", type=float, default=None)
    parser.add_argument("--branching-min-nodes", type=int, default=None)
    parser.add_argument("--upper-cutoff", type=int, default=None)
    parser.add_argument("--stop-cutoff", type=int, default=None)
    parser.add_argument("--lower-cutoff", type=int, default=None)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--hard-memory-bytes", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats <= 0 or args.storage_page_rows <= 0:
        parser.error("repeats and storage-page-rows must be positive")
    if not 0 <= args.start_repeat < args.repeats:
        parser.error("start-repeat must be nonnegative and below repeats")
    if args.hard_memory_bytes is not None and args.hard_memory_bytes <= 0:
        parser.error("hard-memory-bytes must be positive")
    if args.execution_memory_bytes is not None and args.execution_memory_bytes <= 0:
        parser.error("execution-memory-bytes must be positive")
    if args.family_batch_rows is not None and args.family_batch_rows <= 0:
        parser.error("family-batch-rows must be positive")
    if args.queue_capacity_rows <= 0:
        parser.error("queue-capacity-rows must be positive")
    if args.safe_tree_size is not None and args.safe_tree_size < 1.0:
        parser.error("safe-tree-size must be at least one")
    if args.branching_min_nodes is not None and args.branching_min_nodes <= 1:
        parser.error("branching-min-nodes must be greater than one")
    if args.upper_cutoff is not None and args.upper_cutoff <= 1:
        parser.error("upper-cutoff must be greater than one")
    if args.stop_cutoff is not None and args.stop_cutoff <= 1:
        parser.error("stop-cutoff must be greater than one")
    if args.lower_cutoff is not None and args.lower_cutoff <= 0:
        parser.error("lower-cutoff must be positive")
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
    cfg.rollout.storage_page_rows = args.storage_page_rows
    if args.family_batch_rows is not None:
        cfg.rollout.family_batch_rows = args.family_batch_rows
    cfg.rollout.queue_capacity_rows = args.queue_capacity_rows
    cfg.rollout.execution_memory_bytes = args.execution_memory_bytes
    cfg.rollout.safe_tree_size = args.safe_tree_size
    cfg.rollout.branching_min_nodes = args.branching_min_nodes
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
    diagnostic_limits = BranchingDiagnosticLimits(
        upper_nodes=args.upper_cutoff,
        stop_nodes=args.stop_cutoff,
    )
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
            admission_error = None
            sampled = None
            try:
                with torch.inference_mode():
                    with torch.autocast(
                        device_type=device.type,
                        enabled=bool(cfg.training.amp),
                        dtype=torch.float16,
                    ):
                        sampled = sample_branched_tree(
                            bundle,
                            cfg,
                            generator,
                            iteration=0,
                            diagnostics=diagnostics,
                            diagnostic_limits=diagnostic_limits,
                        )
            except BranchingCutoffReached as caught:
                cutoff = caught
            except MemoryError as caught:
                admission_error = str(caught)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            execution = {} if metrics is None else metrics.snapshot()
            if cutoff is None and admission_error is None:
                assert sampled is not None
                tree = sampled.topology
                policy = sampled.decision_inputs
                degree = tree.degree.detach().to("cpu", torch.int64)
                leaf = degree == 0
                parent = tree.parent.detach().to("cpu", torch.int64)
                is_player = tree.decision_partition.detach().to("cpu") >= 0
                path_weight = torch.ones(tree.nodes, dtype=torch.float64)
                player_depth = torch.zeros(tree.nodes, dtype=torch.int64)
                for node in range(1, tree.nodes):
                    ancestor = int(parent[node])
                    path_weight[node] = path_weight[ancestor] / degree[ancestor]
                    player_depth[node] = player_depth[ancestor] + is_player[ancestor]
                leaf_weight = path_weight[leaf]
                leaf_depth = player_depth[leaf].double()
                weight_total = float(leaf_weight.sum())
                weighted_depth = float((leaf_weight * leaf_depth).sum() / weight_total)
                geometry = {
                    "nodes": float(tree.nodes),
                    "decisions": float(policy.node_ids.numel()),
                    "leaves": float(leaf.sum()),
                    "max_offspring": float(degree.max()),
                    "max_player_depth": float(player_depth.max()),
                    "weighted_leaf_player_depth": weighted_depth,
                    "leaf_weight_sum": weight_total,
                    "stop_cutoff_reached": 0.0,
                    "pending_nodes": 0.0,
                    "queued_nodes": 0.0,
                    "frontier_nodes": 0.0,
                    "family_queue_nodes": {},
                    "memory_admission_stop": 0.0,
                    **{
                        key: float(value)
                        for key, value in sampled.branching_diagnostics.items()
                    },
                    **_decision_street_geometry(policy),
                    **_cost_head_calibration(sampled),
                }
            elif cutoff is not None:
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
                    "memory_admission_stop": 0.0,
                }
            else:
                geometry = {
                    "nodes": math.nan,
                    "decisions": math.nan,
                    "leaves": math.nan,
                    "max_offspring": math.nan,
                    "max_player_depth": math.nan,
                    "weighted_leaf_player_depth": math.nan,
                    "leaf_weight_sum": math.nan,
                    "stop_cutoff_reached": 0.0,
                    "pending_nodes": math.nan,
                    "queued_nodes": math.nan,
                    "frontier_nodes": math.nan,
                    "family_queue_nodes": {},
                    "memory_admission_stop": 1.0,
                    "memory_admission_error": admission_error,
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
                        and sampled is not None
                        and sampled.node_count < args.lower_cutoff
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
                f"nodes={sample['nodes']:.0f} "
                f"stopped={int(sample['stop_cutoff_reached'])} "
                f"pending={sample['pending_nodes']:.0f} "
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
            if sampled is not None:
                del degree, leaf, leaf_depth, leaf_weight, parent, path_weight, player_depth
            del sampled
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
        "storage_page_rows": args.storage_page_rows,
        "family_batch_rows": cfg.rollout.family_batch_rows,
        "queue_capacity_rows": cfg.rollout.queue_capacity_rows,
        "execution_memory_bytes": args.execution_memory_bytes,
        "safe_tree_size": args.safe_tree_size,
        "branching_min_nodes": args.branching_min_nodes,
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
        node = record["summary"].get("nodes")
        if node is None:
            print(f"beta={record['beta']:.4f} no completed node samples")
            continue
        print(
            f"beta={record['beta']:.4f} "
            f"nodes p50={node['p50']:.1f} p95={node['p95']:.1f} "
            f"p99={node['p99']:.1f} max={node['max']:.0f}"
        )


if __name__ == "__main__":
    main()
