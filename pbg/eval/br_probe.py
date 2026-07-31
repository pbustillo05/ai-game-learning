"""Accelerator paired evaluation of queued best-response artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pbg.devices import resolve_execution_device
from pbg.games.poker.queued.response import (
    SHARED_RESPONSE_SEMANTICS,
    SequentialSharedResponseEvaluator,
)
from pbg.inference import PokerInferenceSession
from pbg.inference_artifact import load_inference_source


@dataclass(frozen=True)
class BRProbeReport:
    semantics: str
    target_sha256: str
    initial_model_sha256: str
    initialization: str
    trajectory_count: int
    trajectory_id_sha256: str
    mean_gain_bb_per_hand: float
    standard_error_bb_per_hand: float
    approximate_nashconv_bb_per_hand: float
    per_seat_gain_bb_per_hand: tuple[float, ...]
    device: str = ""
    seed: int = 0

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_br_probe_observations(
    *,
    players: int,
    target_sha256: str,
    initial_model_sha256: str,
    initialization: str,
    trajectory_ids: Tensor,
    mixed_payoff: Tensor,
    baseline_payoff: Tensor,
    device: str = "",
    seed: int = 0,
) -> BRProbeReport:
    """Reduce already-paired accelerator observations with the public law."""

    evaluator = SequentialSharedResponseEvaluator(
        players, target_sha256, initial_model_sha256, initialization
    )
    report = evaluator.evaluate(
        trajectory_ids,
        lambda _ids, _seats: mixed_payoff,
        lambda _ids, _seats: baseline_payoff,
    )
    statistics = report.statistics
    return BRProbeReport(
        semantics=SHARED_RESPONSE_SEMANTICS,
        target_sha256=report.target_sha256,
        initial_model_sha256=report.initial_model_sha256,
        initialization=report.initialization,
        trajectory_count=report.trajectory_count,
        trajectory_id_sha256=report.trajectory_id_sha256,
        mean_gain_bb_per_hand=statistics.mean_gain,
        standard_error_bb_per_hand=statistics.standard_error,
        approximate_nashconv_bb_per_hand=(
            statistics.approximate_nashconv_lower_bound
        ),
        per_seat_gain_bb_per_hand=statistics.per_seat_gain,
        device=device,
        seed=seed,
    )


def _uniform(seed: int, trajectory: int, decision: int) -> float:
    encoded = f"{seed}:{trajectory}:{decision}:paired-action".encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return (value + 0.5) / float(1 << 64)


def _select_action(
    probabilities: Tensor,
    actions: tuple[object, ...],
    *,
    uniform: float,
) -> object:
    cumulative = probabilities.double().cumsum(0)
    row = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(
                uniform,
                dtype=cumulative.dtype,
                device=cumulative.device,
            ),
        ).clamp_max(len(actions) - 1)
    )
    return actions[row]


def _assert_same_deal(
    left: PokerInferenceSession,
    left_hand: object,
    right: PokerInferenceSession,
    right_hand: object,
) -> None:
    for seat in range(left.geometry.players):
        if left.private_cards(left_hand, seat) != right.private_cards(
            right_hand, seat
        ):
            raise RuntimeError("paired sessions did not receive the same private deal")


def _play_one(
    probe: PokerInferenceSession,
    target: PokerInferenceSession,
    *,
    trajectory_id: int,
    probe_seat: int,
    seed: int,
    mixed: bool,
) -> float:
    """Play one shared-deal policy profile entirely through queued inference."""

    button = trajectory_id % target.geometry.players
    target_hand = target.new_hand(button)
    probe_hand = probe.new_hand(button)
    _assert_same_deal(target, target_hand, probe, probe_hand)
    decision_index = 0
    while not target.is_terminal(target_hand):
        if target.needs_chance(target_hand):
            # Public cards belong to the trajectory/street deal, not to the
            # number of policy decisions taken before reaching that street.
            # Keeping this address fixed makes mixed and baseline replays use
            # the same runout even when their action paths have different
            # lengths.
            target.deal_keyed(
                target_hand,
                seed=seed,
                trajectory_id=trajectory_id,
                decision_index=0,
            )
            probe.deal_keyed(
                probe_hand,
                seed=seed,
                trajectory_id=trajectory_id,
                decision_index=0,
            )
            continue
        actor = target.actor(target_hand)
        selected = probe if mixed and actor == probe_seat else target
        selected_hand = probe_hand if selected is probe else target_hand
        actions, policy = selected.policy_table(selected_hand)
        private = selected.private_policy_index(selected_hand, actor)
        columns = torch.tensor(
            [action.event_id for action in actions],
            dtype=torch.int64,
            device=policy.device,
        )
        action = _select_action(
            policy[private].index_select(0, columns),
            actions,
            uniform=_uniform(seed, trajectory_id, decision_index),
        )
        target.advance(
            target_hand,
            action,
            policy_probabilities=policy,
        )
        probe.advance(
            probe_hand,
            action,
            policy_probabilities=policy,
        )
        decision_index += 1
    payoff = target.terminal_payoffs(target_hand)
    return float(payoff[probe_seat].detach().cpu())


def _validate_sources(
    probe: dict[str, Any],
    target: dict[str, Any],
) -> tuple[int, str, str, str]:
    scientific = probe["scientific_config"]
    target_scientific = target["scientific_config"]
    mode = scientific.get("mode")
    if mode != {
        "training_regime": "best_response",
        "rollout_mode": "sequential",
        "private_estimator": "rao_blackwell",
    }:
        raise RuntimeError(
            "BR evaluation requires a sequential RB best-response artifact"
        )
    if scientific.get("game") != target_scientific.get("game"):
        raise RuntimeError("probe and target game geometry differ")
    if scientific.get("model") != target_scientific.get("model"):
        raise RuntimeError("probe and target model contracts differ")
    run = scientific.get("run")
    metadata = scientific.get("probe")
    initialization = scientific.get("response_initialization")
    if (
        not isinstance(run, dict)
        or run.get("frozen_sha256") != target["source_checkpoint_sha256"]
        or not isinstance(metadata, dict)
        or metadata.get("target_sha256") != target["source_checkpoint_sha256"]
        or initialization not in ("random", "target")
    ):
        raise RuntimeError("probe target content identity is invalid")
    return (
        int(scientific["game"]["players"]),
        str(target["source_checkpoint_sha256"]),
        str(metadata["initial_model_sha256"]),
        str(initialization),
    )


def evaluate_br_probe(
    probe_artifact: str | Path,
    *,
    target_artifact: str | Path,
    n_hands: int,
    seed: int = 1729,
    device: str | torch.device,
) -> BRProbeReport:
    """Execute balanced paired mixed/baseline rollouts on CUDA."""

    requested = resolve_execution_device(
        device, workload="best-response evaluation"
    )
    if requested.type != "cuda":
        raise RuntimeError(
            "BR evaluation requires CUDA completion/showdown operations"
        )
    probe_source = load_inference_source(probe_artifact)
    target_source = load_inference_source(target_artifact)
    players, target_hash, initial_hash, initialization = _validate_sources(
        probe_source, target_source
    )
    block = players * players
    if (
        isinstance(n_hands, bool)
        or not isinstance(n_hands, int)
        or n_hands <= 1
        or n_hands % block
    ):
        raise ValueError("n_hands must be a positive multiple of P^2")
    probe_mixed = PokerInferenceSession(probe_source, device=requested)
    target_mixed = PokerInferenceSession(target_source, device=requested)
    # A distinct pair owns the baseline state, while logical action/chance
    # addresses are shared with the mixed replay.
    probe_baseline = PokerInferenceSession(target_source, device=requested)
    target_baseline = PokerInferenceSession(target_source, device=requested)
    ids = torch.arange(n_hands, dtype=torch.int64, device=target_mixed.device)
    seats = torch.div(ids, players, rounding_mode="floor").remainder(players)
    mixed_values = []
    baseline_values = []
    for trajectory_id, seat in zip(ids.tolist(), seats.tolist(), strict=True):
        mixed_values.append(
            _play_one(
                probe_mixed,
                target_mixed,
                trajectory_id=trajectory_id,
                probe_seat=seat,
                seed=seed,
                mixed=True,
            )
        )
        baseline_values.append(
            _play_one(
                probe_baseline,
                target_baseline,
                trajectory_id=trajectory_id,
                probe_seat=seat,
                seed=seed,
                mixed=False,
            )
        )
    return evaluate_br_probe_observations(
        players=players,
        target_sha256=target_hash,
        initial_model_sha256=initial_hash,
        initialization=initialization,
        trajectory_ids=ids,
        mixed_payoff=torch.tensor(
            mixed_values, dtype=torch.float32, device=ids.device
        ),
        baseline_payoff=torch.tensor(
            baseline_values, dtype=torch.float32, device=ids.device
        ),
        device=str(target_mixed.device),
        seed=seed,
    )


__all__ = [
    "BRProbeReport",
    "evaluate_br_probe",
    "evaluate_br_probe_observations",
]
