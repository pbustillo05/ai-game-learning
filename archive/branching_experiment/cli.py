"""CLI for read-only fixed-policy branching experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pbg.core.config import load_config
from pbg.experiments.branching import (
    FixedPolicyBranchingController,
    load_numeric_rules,
    run_branching_experiments,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure experimental branching controllers on frozen checkpoints"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--hardware-profile", default=None)
    parser.add_argument(
        "--controller",
        action="append",
        choices=sorted(FixedPolicyBranchingController.VARIANTS),
        dest="controllers",
    )
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--node-capacity", type=int, default=None)
    parser.add_argument(
        "--beta-schedule",
        default=None,
        help="JSON/YAML file or inline flat phase:street:depth mapping",
    )
    parser.add_argument(
        "--cost-table",
        default=None,
        help="JSON/YAML file or inline flat phase:street:depth mapping",
    )
    parser.add_argument("--soft-budget-fraction", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--kappa-eta", type=float, default=1.0)
    parser.add_argument("--target-size", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, hardware_profile=args.hardware_profile)
    if args.node_capacity is not None:
        if args.node_capacity <= 0:
            parser.error("--node-capacity must be positive")
        cfg.rollout.node_budget = args.node_capacity
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda:0"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    result = run_branching_experiments(
        cfg,
        args.checkpoint,
        variants=args.controllers or ["legacy"],
        trials=args.trials,
        seed=args.seed,
        device=device,
        schedule=load_numeric_rules(args.beta_schedule),
        costs=load_numeric_rules(args.cost_table),
        soft_budget_fraction=args.soft_budget_fraction,
        kappa=args.kappa,
        kappa_eta=args.kappa_eta,
        target_size=args.target_size,
        beta_max=args.beta_max,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
