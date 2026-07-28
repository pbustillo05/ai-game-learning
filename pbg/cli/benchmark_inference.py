"""Benchmark warmed model-only poker inference on CUDA or Apple MPS."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from pbg.inference import PokerInferenceSession


def _prepare_hands(
    session: PokerInferenceSession, batch: int, street: str
) -> tuple[list, list[list[int]]]:
    hands = session.new_hands(
        [index % session.game_config.n_players for index in range(batch)]
    )
    active = list(session.game_config.geometry.active_cards)
    private = [active[-2:] for _ in hands]
    if street == "preflop":
        return hands, private
    for hand in hands:
        while not bool(session.machine.needs_chance(session._hand(hand).state)[0]):
            state = session._hand(hand).state
            actor = state.to_act.long()
            to_call = float((
                state.current_bet
                - state.committed_street.gather(1, actor[:, None]).squeeze(1)
            ).clamp_min(0).cpu()[0])
            session.advance(hand, "call" if to_call > 0 else "check")
        session.deal(hand, active[:3])
    return hands, private


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark warmed poker policy/value inference"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--device", required=True, choices=("cuda", "mps"))
    parser.add_argument("--precision", default="fp32", choices=("fp32", "fp16"))
    parser.add_argument(
        "--mode", default="latency", choices=("latency", "throughput")
    )
    parser.add_argument("--street", default="preflop", choices=("preflop", "flop"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 8, 32])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.repeats <= 0
        or any(batch <= 0 for batch in args.batches)
    ):
        parser.error("warmup must be nonnegative and repeats/batches must be positive")

    session = PokerInferenceSession.from_checkpoint(
        args.checkpoint,
        device=args.device,
        precision=args.precision,
        mode=args.mode,
    )
    reports = []
    for batch in args.batches:
        hands, private = _prepare_hands(session, batch, args.street)
        for _ in range(args.warmup):
            session.predict_many(hands, private)
        if session.device.type == "mps":
            torch.mps.synchronize()
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            session.predict_many(hands, private)
            if session.device.type == "mps":
                torch.mps.synchronize()
            samples.append(time.perf_counter() - started)
        ordered = sorted(samples)
        p95_index = min(
            len(ordered) - 1,
            max(0, math.ceil(0.95 * len(ordered)) - 1),
        )
        median = statistics.median(samples)
        report = {
            "batch": batch,
            "p50_seconds": median,
            "p95_seconds": ordered[p95_index],
            "spots_per_second": batch / median,
        }
        if session.device.type == "mps":
            report.update(
                {
                    "mps_current_allocated_bytes": int(
                        torch.mps.current_allocated_memory()
                    ),
                    "mps_driver_allocated_bytes": int(
                        torch.mps.driver_allocated_memory()
                    ),
                    "mps_recommended_max_bytes": int(
                        torch.mps.recommended_max_memory()
                    ),
                }
            )
        reports.append(report)
    print(
        json.dumps(
            {
                "device": str(session.device),
                "precision": session.precision,
                "mode": session.mode,
                "street": args.street,
                "iteration": session.iteration,
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
