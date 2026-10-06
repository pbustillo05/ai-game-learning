"""Bounded CUDA restoration of the July policy inputs, units, and objective.

This deliberately changes no production implementation. The smoke restores the
legacy exact-hand reach gauge only on its experiment-owned model instances.
It checks that the current runtime can train the recovered initialization; two
iterations cannot establish restored learning speed or poker strength.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from experiments.training_diagnosis.probe_policies import (
    OLD,
    OUT,
    install_legacy_preflop_inputs,
    summarize,
)
from pbg.config import load_config_mapping
from pbg.core.optimization import LossScaler
from pbg.games.poker.queued.hooks import PokerStateMachine
from pbg.inference import PokerInferenceSession
from pbg.runtime import TrainingSession

TEMPLATE = Path(
    "configs/experiments/nlhe_hu_32bb_rb_dualclip5_60h_20260814.yaml"
)


def historical_adapter(model, geometry):
    wrapper = SimpleNamespace(
        model=model,
        machine=PokerStateMachine(geometry),
        geometry=geometry,
    )
    install_legacy_preflop_inputs(wrapper)


def policy_snapshot(session):
    """Read three fixed states without perturbing ambient training RNG state."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(session.device)
    wrapper = None
    try:
        wrapper = PokerInferenceSession(
            {
                "scientific_config": session.config.scientific_mapping(),
                "model": session.modules.lane_module(0).state_dict(),
                "iteration": session.run_states[0].iteration,
                "source_checkpoint_sha256": "restoration-smoke-in-memory",
            },
            device=str(session.device),
        )
        install_legacy_preflop_inputs(wrapper)
        with torch.inference_mode():
            hand = wrapper.new_hand()
            snapshot = {"root": summarize(wrapper, hand)}
            wrapper.advance(hand, ("raise_to", 3.0))
            snapshot["bb_facing_3bb"] = summarize(wrapper, hand)
            wrapper.advance(hand, "call")
            wrapper.deal(hand, ["As", "7h", "2s"])
            snapshot["flop_A72_two_spades"] = summarize(wrapper, hand)
        return snapshot
    finally:
        del wrapper
        gc.collect()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, session.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("reference", "model"), default="reference")
    parser.add_argument("--checkpoint", type=Path, default=OLD / "ckpt_0001110.pt")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--roots", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--align-completion", action="store_true", help="Apply experiment-only completion-to-topology leaf alignment before targets")
    args = parser.parse_args()
    if min(args.iterations, args.roots, args.steps) <= 0:
        parser.error("iterations, roots, and steps must be positive")
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("restoration smoke requires an explicitly targeted CUDA GPU")
    torch.cuda.set_device(0)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state_key = "reference_model" if args.source == "reference" else "model"
    if state_key not in payload:
        raise ValueError(f"source checkpoint omits {state_key}")
    source = yaml.safe_load(TEMPLATE.read_text())
    objective = source["scientific"]["objective"]
    objective.pop("negative_advantage_ratio_cap", None)
    objective["policy_clip"] = 0.2
    objective["value"] = {"kind": "clipped", "epsilon": 0.2, "coefficient": 0.0005}
    objective["entropy"] = {"kind": "constant", "coefficient": 0.05}
    objective["reference_kl"] = {"coefficient": 0.2, "refresh": {"iterations": 1380}}
    source["scientific"]["optimizer"]["weight_decay"] = 0.01
    source["scientific"]["schedule"] = {"batch_size": 256, "steps": args.steps}
    run = source["run"]
    run["utility"] = "raw_bb"
    run["seed"] = args.seed
    run["workload"]["roots"] = args.roots
    run["recovery"]["minimum_roots"] = args.roots
    run["execution"]["device"] = "cuda:0"
    run["execution"]["reference_cache"] = "off"
    run["iterations"] = args.iterations
    alignment_tag = "aligned" if args.align_completion else "uncorrected"
    tag = f"{args.source}_roots{args.roots}_steps{args.steps}_seed{args.seed}_{alignment_tag}"
    run["output_dir"] = str(OUT / f"restoration_{tag}")
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"restoration_{tag}.json"
    (OUT / f"restoration_{tag}_settings.json").write_text(json.dumps(source, indent=2) + "\n")
    config = load_config_mapping(source)
    session = TrainingSession(config, initial_model_state=payload[state_key], benchmark_telemetry=True)
    historical_adapter(session.modules.lane_module(0), session.geometry)
    if session.reference_modules is None:
        raise RuntimeError("historical objective requires a frozen reference")
    historical_adapter(session.reference_modules.lane_module(0), session.geometry)
    alignment_audit = []
    if args.align_completion:
        from experiments.training_diagnosis.completion_alignment import install_completion_alignment
        install_completion_alignment(session, alignment_audit)
    optimizer = session.optimizers.optimizers[0]
    # 537700f:pbg/core/training.py:97 starts fresh FP16 runs at scale 1024.
    optimizer.scaler = LossScaler(enabled=True, scale=1024.0, growth_interval=2000)
    if optimizer.step_count != 0:
        raise RuntimeError("restoration requires fresh Adam moments and step counter")
    del payload
    results = {
        "source_checkpoint": str(args.checkpoint),
        "source_state_key": state_key,
        "source_interpretation": (
            "Recovered iteration-20 policy retained as the unchanged July KL reference"
            if args.source == "reference"
            else "July trained iteration-1110 policy; fresh optimizer/reference for this smoke"
        ),
        "arguments": {**vars(args), "checkpoint": str(args.checkpoint)},
        "source_config": source,
        "legacy_input_adapter": "exact-hand max gauge before preflop class summation",
        "value_head_scaling": 1.0,
        "completion_alignment_status": ("experimental correction enabled" if args.align_completion else "UNCORRECTED: known completion/topology row-order defect; invalid learning evidence"),
        "completion_alignment": alignment_audit,
        "initial_optimizer_step": optimizer.step_count,
        "initial_amp_scaler": optimizer.scaler.state_dict(),
        "cuda": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(session.device),
            "name": torch.cuda.get_device_name(session.device),
        },
        "scope": "Pipeline smoke only; reduced rollout/update budget is not a learning-speed reproduction.",
        "before": policy_snapshot(session),
        "iterations": [],
    }
    destination.write_text(json.dumps(results, indent=2) + "\n")
    try:
        for _ in range(args.iterations):
            torch.cuda.synchronize(session.device)
            torch.cuda.reset_peak_memory_stats(session.device)
            alignment_begin = len(alignment_audit)
            start = time.monotonic()
            session.step()
            torch.cuda.synchronize(session.device)
            elapsed = time.monotonic() - start
            memory = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device),
                "allocated_bytes": torch.cuda.memory_allocated(session.device),
                "reserved_bytes": torch.cuda.memory_reserved(session.device),
            }
            metrics = copy.deepcopy(session.run_states[0].metrics_prefix[-1])
            row = {
                "completed_iteration": session.run_states[0].iteration,
                "seconds": elapsed,
                "cuda_memory": memory,
                "policy": policy_snapshot(session),
                "metrics": metrics,
                "completion_alignment": copy.deepcopy(alignment_audit[alignment_begin:]),
            }
            results["iterations"].append(row)
            destination.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({
                "tag": tag,
                "completed_iteration": row["completed_iteration"],
                "seconds": elapsed,
                "player_nodes": metrics["player_nodes"],
                "applied_steps": metrics["optimizer_steps"],
                "raise": row["policy"]["root"]["raise"],
                "range_std": row["policy"]["root"]["raise_std_over_hands"],
            }), flush=True)
        parameters_finite = bool(torch.stack([
            parameter.isfinite().all()
            for parameter in session.modules.lane_module(0).parameters()
        ]).all())
        results["validation"] = {
            "completed_iterations": session.run_states[0].iteration,
            "applied_optimizer_steps": optimizer.step_count,
            "requested_optimizer_steps": args.iterations * args.steps,
            "model_parameters_finite": parameters_finite,
            "reference_iteration": session.reference_states[0].reference_iteration,
            "production_code_modified": False,
            "target_alignment_corrected": args.align_completion,
            "target_alignment_verified": bool(alignment_audit) and all(item.get("all_columns_verified", False) for item in alignment_audit),
            "completion_leaf_rows_aligned": sum(item["leaf_count"] for item in alignment_audit),
            "completion_leaf_rows_moved": sum(item["moved_rows"] for item in alignment_audit),
        }
        results["after"] = results["iterations"][-1]["policy"]
        destination.write_text(json.dumps(results, indent=2) + "\n")
        if args.align_completion and not results["validation"]["target_alignment_verified"]:
            raise RuntimeError("aligned restoration requires verified completion movement before target construction")
        if not parameters_finite or optimizer.step_count == 0:
            raise RuntimeError("restoration smoke did not produce finite updated model parameters")
    finally:
        session.close()


if __name__ == "__main__":
    main()
