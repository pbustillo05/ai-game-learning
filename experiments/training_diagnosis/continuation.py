"""Paired CUDA-only entropy ablations using untouched production iterations."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from pbg.config import load_config_mapping
from pbg.core.learner import ReferenceState
from pbg.core.optimization import LossScaler
from pbg.runtime import TrainingSession
from pbg.inference import PokerInferenceSession
from experiments.training_diagnosis.probe_policies import OUT, summarize


def root_stats(session):
    # The wrapper owns only inference state; use the already resident model.
    source = {
        "scientific_config": session.config.scientific_mapping(),
        "model": session.modules.lane_module(0).state_dict(),
        "iteration": session.run_states[0].iteration,
        "source_checkpoint_sha256": "diagnostic-in-memory",
    }
    wrapper = PokerInferenceSession(source, device="cuda:0")
    return summarize(wrapper, wrapper.new_hand())


def attribute_first_batch(session, capture, batches=4):
    factory = session._iteration_engine

    def instrumented_factory():
        engine = factory()
        original = engine._prepare_learner

        def prepare(owners, state, lane):
            prepared = original(owners, state, lane)
            if capture:
                return prepared
            learner = prepared.iteration.learner
            sampler_state = prepared.sampler.generator.get_state()
            parameters = tuple(session.modules.lane_module(0).parameters())
            terms = ("policy", "value", "entropy", "reference_kl")
            gradients = {term: [torch.zeros_like(p) for p in parameters] for term in terms}
            totals = {term: 0.0 for term in (*terms, "approximate_kl", "clip_fraction")}
            rows = 0
            # Suppress only the method's final backward. The returned tensors
            # then expose each exact production loss graph to autograd.grad.
            for batch in prepared.sampler.batches():
                if batch.logical_step >= batches:
                    break
                for physical, weight in prepared.iteration._physical_batches(batch):
                    with patch.object(torch.Tensor, "backward", lambda *args, **kwargs: None):
                        result = learner.forward_loss_backward(
                            physical,
                            iteration=session.run_states[0].iteration,
                            reference_operation=session.reference_operations[0],
                        )
                    rows += physical.rows
                    for term in totals:
                        totals[term] += float(getattr(result, term).detach()) * float(weight) / batches
                    for term in terms:
                        grad = torch.autograd.grad(getattr(result, term), parameters, retain_graph=True, allow_unused=True)
                        for accumulator, value in zip(gradients[term], grad):
                            if value is not None:
                                accumulator.add_(value.detach(), alpha=float(weight) / batches)
                    del result
            flattened = {term: torch.cat([v.flatten() for v in values]) for term, values in gradients.items()}
            coefficients = {"policy": 1.0, "value": learner.loss.value.coefficient, "entropy": -learner.loss.entropy.coefficient, "reference_kl": learner.loss.reference_kl_coefficient}
            norms = {term: float(value.norm()) for term, value in flattened.items()}
            cosines = {f"{a},{b}": float(torch.nn.functional.cosine_similarity(flattened[a][None], flattened[b][None])) for a in terms for b in terms if a < b}
            capture.update({"rows": rows, "logical_batches": batches, "loss_means": totals, "unweighted_gradient_norms": norms, "weighted_gradient_norms": {term: abs(coefficients[term]) * norms[term] for term in terms}, "raw_gradient_cosines": cosines, "partition_rows": {name: store.rows for name, store in prepared.sampler.stores.items()}})
            prepared.sampler.generator.set_state(sampler_state)
            session.optimizers.optimizers[0].zero_grad()
            return prepared

        engine._prepare_learner = prepare
        return engine

    session._iteration_engine = instrumented_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("normal", "rb"), default="normal")
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--entropy", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--roots", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=918273)
    parser.add_argument("--attribute", action="store_true")
    args = parser.parse_args()
    stem = "nlhe_hu_32bb_normal_60h_20260812" if args.run == "normal" else "nlhe_hu_32bb_rb_dualclip5_60h_20260814"
    path = Path("runs/experiments") / stem / f"checkpoint_{args.checkpoint:08d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source = yaml.safe_load((Path("configs/experiments") / f"{stem}.yaml").read_text())
    tag = f"{args.run}_i{args.checkpoint}_h{args.entropy:g}_seed{args.seed}"
    source["scientific"]["objective"]["entropy"]["coefficient"] = args.entropy
    source["scientific"]["schedule"]["steps"] = args.steps
    run = source["run"]
    run["seed"] = args.seed
    run["workload"]["roots"] = args.roots
    run["recovery"]["minimum_roots"] = args.roots
    run["execution"]["reference_cache"] = "off"
    run["iterations"] = args.checkpoint + args.iterations
    run["output_dir"] = str(OUT / tag)
    cfg = load_config_mapping(source)
    session = TrainingSession(cfg, initial_model_state=payload["model"], benchmark_telemetry=True)
    optimizer = session.optimizers.optimizers[0]
    optimizer.load_state_dict(payload["optimizer"])
    optimizer.scaler = LossScaler.from_state_dict(payload["scaler"])
    session.reference_modules.lane_module(0).load_state_dict(payload["reference_model"], strict=True)
    session.reference_states = (ReferenceState.from_state_dict(payload["reference_state"]),)
    session.run_states[0].iteration = args.checkpoint
    capture = {}
    if args.attribute:
        attribute_first_batch(session, capture)
    results = {"source_checkpoint": str(path), "arguments": vars(args), "source_config": source, "attribution": capture, "before": root_stats(session), "iterations": []}
    destination = OUT / f"continuation_{tag}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(args.iterations):
            start = time.monotonic()
            session.step()
            row = {"iteration": session.run_states[0].iteration, "seconds": time.monotonic()-start, "root": root_stats(session), "metrics": copy.deepcopy(session.run_states[0].metrics_prefix[-1])}
            results["iterations"].append(row)
            destination.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({"tag": tag, "iteration": row["iteration"], "seconds": row["seconds"], "raise": row["root"]["raise"], "range_std": row["root"]["raise_std_over_hands"], "size_uniform_kl": row["root"]["raise_kl_to_uniform"]}), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
