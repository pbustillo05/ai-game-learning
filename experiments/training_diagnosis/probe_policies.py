"""Read-only CUDA checkpoint probes for the September 2026 diagnosis.

The legacy model state has exactly the same keys and shapes as queued states.
For this experiment only it is loaded strictly with the equivalent queued
architecture; no checkpoint or production compatibility surface is changed.
Legacy checkpoints receive the historical exact-hand reach gauge before
class summation. This experimental input adapter was checked on CUDA against
historical 537700f preflop and postflop networks (legacy_input_audit.py).
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from pbg.inference import PokerInferenceSession
from pbg.games.poker.model.preflop import PreflopMasks

BASE = Path("runs/experiments")
RB = BASE / "nlhe_hu_32bb_rb_dualclip5_60h_20260814"
NORMAL = BASE / "nlhe_hu_32bb_normal_60h_20260812"
OLD = Path("runs/nlhe_hu_32bb_full_sequential_rb")
OUT = BASE / "nlhe_hu_32bb_rb_training_diagnosis_20260925"


def install_legacy_preflop_inputs(session):
    """Reproduce 537700f rollout input units for legacy checkpoint probes."""
    def historical_call(model_input):
        state = session.model._state("pre", model_input)
        log_class, history = session.machine.model_context("pre", model_input)
        public, _ = session.model._public(state, history)
        exact_log = session.machine.expand_preflop_reach(log_class)
        finite = torch.isfinite(exact_log)
        gauge = exact_log.masked_fill(~finite, -torch.inf).amax(-1, keepdim=True)
        exact_weight = (exact_log - gauge).exp().masked_fill(~finite, 0)
        ranges = exact_weight.new_zeros(exact_weight.shape[:2] + (session.geometry.private_classes,))
        ranges.scatter_add_(2, session.model.combo_class[None, None, :].expand_as(exact_weight), exact_weight)
        return session.model._evaluation(session.model.preflop(public, ranges, public.to_act, PreflopMasks(seat_mask=public.seat_mask, status=public.status)))
    session.model._call_preflop = historical_call


def session_for(path: Path, template: dict) -> PokerInferenceSession:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    session = PokerInferenceSession(
        {
            "scientific_config": payload.get("scientific_config", template),
            "model": payload["model"],
            "iteration": payload.get("iteration", payload.get("run_state", {}).get("iteration", 0)),
            "source_checkpoint_sha256": "diagnostic-direct-strict-state-load",
        },
        device="cuda:0",
    )
    if "scientific_config" not in payload:
        install_legacy_preflop_inputs(session)
    return session


def summarize(session, handle):
    item = session._hand(handle)
    evaluation, legal, probs = session._evaluate(item)
    p = probs[0]
    raw = evaluation.policy_raw
    if session._family(item.state) == "pre":
        p = p.index_select(0, session.model.combo_class)
        values = evaluation.value[0].index_select(0, session.model.combo_class)
    else:
        values = evaluation.value[0]
    weights = item.log_reach[0, session.actor(handle)].softmax(-1)
    raise_p = p[:, 2:].sum(-1)
    cond = p[:, 2:] / raise_p[:, None].clamp_min(1e-30)
    h_size = -(cond * cond.clamp_min(1e-30).log()).sum(-1)
    n_raises = legal.mask[0, 2:].sum()
    type_p = torch.stack((p[:, 0], p[:, 1], raise_p), -1)
    h_type = -(type_p * type_p.clamp_min(1e-30).log()).sum(-1)
    result = {
        "raw_type_logits_min_by_type": raw["type_logits"].reshape(-1, 3).amin(0).tolist(),
        "raw_type_logits_max_by_type": raw["type_logits"].reshape(-1, 3).amax(0).tolist(),
        "fold_probability_exact_zero_fraction": float(p[:, 0].eq(0).float().mean()),
        "call_probability_exact_zero_fraction": float(p[:, 1].eq(0).float().mean()),
        "fold_probability_below_1e12_fraction": float(p[:, 0].lt(1e-12).float().mean()),
        "call_probability_below_1e12_fraction": float(p[:, 1].lt(1e-12).float().mean()),
        "fold": float(weights @ p[:, 0]),
        "call_check": float(weights @ p[:, 1]),
        "raise": float(weights @ raise_p),
        "raise_std_over_hands": float(raise_p.std()),
        "raise_min": float(raise_p.min()),
        "raise_max": float(raise_p.max()),
        "legal_raise_sizes": int(n_raises),
        "type_entropy": float(weights @ h_type),
        "conditional_raise_entropy": float(weights @ h_size),
        "raise_kl_to_uniform": float(weights @ (n_raises.float().log() - h_size)),
        "entropy": float(weights @ (h_type + raise_p * h_size)),
        "raise_scale_mean": float(raw["s"].mean()),
        "allin_atom_mean": float(raw["iota"].mean()),
        "value_mean_native_units": float(weights @ values),
        "avg_raise_bb": float(weights @ (cond @ (torch.arange(cond.shape[-1], device=p.device) + 1).float().mul(.5))),
    }
    if session._family(item.state) == "pre":
        class_probability = probs[0]
        result["type_probabilities_by_class"] = torch.stack((class_probability[:, 0], class_probability[:, 1], class_probability[:, 2:].sum(-1)), -1).tolist()
    selected = {}
    for name, cards in {"AA": ["Ac", "Ad"], "KK": ["Kc", "Kd"], "AKs": ["Ac", "Kc"], "72o": ["7c", "2d"], "JTs": ["Jc", "Tc"]}.items():
        index = session._private_id(cards)
        selected[name] = type_p[index].tolist()
    result["hands_fold_call_raise"] = selected
    result["mean_event_probabilities"] = (weights @ p).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    template = torch.load(RB / "checkpoint_00010000.pt", map_location="cpu", weights_only=False, mmap=True)["scientific_config"]
    points = [("old", OLD / "ckpt_0001110.pt")]
    if not args.quick:
        points = [("old", OLD / f"ckpt_{i:07d}.pt") for i in (37, 111, 370, 740, 1110)]
    for name, folder in (("normal", NORMAL), ("rb", RB)):
        for i in ((1100, 10000) if args.quick else (50, 200, 500, 1100, 2500, 4000, 5400, 5450, 5500, 5600, 7000, 10000)):
            path = folder / f"checkpoint_{i:08d}.pt"
            if path.exists():
                points.append((name, path))
    results = []
    for name, path in points:
        session = session_for(path, template)
        hand = session.new_hand()
        row = {"run": name, "checkpoint": str(path), "iteration": session.iteration, "root": summarize(session, hand)}
        if args.quick or session.iteration in (1110, 1100, 10000):
            session.advance(hand, ("raise_to", 3.0))
            row["bb_facing_3bb"] = summarize(session, hand)
            session.advance(hand, "call")
            session.deal(hand, ["As", "7h", "2s"])
            row["flop_A72_two_spades"] = summarize(session, hand)
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ("root", "bb_facing_3bb", "flop_A72_two_spades")}), json.dumps({k:v for k,v in row["root"].items() if k not in ("hands_fold_call_raise", "mean_event_probabilities")}), flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ("policy_probes_quick.json" if args.quick else "policy_probes.json")).write_text(json.dumps(results, indent=2) + "\n")
        del session
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
