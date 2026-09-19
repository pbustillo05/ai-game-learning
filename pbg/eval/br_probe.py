"""Accelerator paired evaluation of queued best-response artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pbg.devices import resolve_execution_device
from pbg.games.poker.queued.response import (
    FIXED_TARGET_RESPONSE_SEMANTICS,
    fixed_target_paired_statistics,
    probe_evaluation_design,
)
from pbg.inference import DealIdentity, PokerInferenceSession
from pbg.inference_artifact import load_inference_source


@dataclass(frozen=True)
class BRProbeObservation:
    logical_deal_id: int
    probe_seat: int
    button: int
    mixed_payoff_bb: float
    baseline_payoff_bb: float
    gain_bb_per_hand: float


EVALUATION_PROTOCOL_VERSION = FIXED_TARGET_RESPONSE_SEMANTICS
TARGET_RANGE_SEMANTICS = "fixed_target_reference_range_board_conditioned_v2"
RESPONSE_TRAINING_RANGE_SEMANTICS = "adaptive_mixed_profile_range_v1"


@dataclass(frozen=True)
class BRProbeReport:
    semantics: str
    protocol_version: str
    split_namespace: str
    evaluation_seed: int
    target_range_semantics: str
    response_training_range_semantics: str
    target_sha256: str
    response_sha256: str
    initial_model_sha256: str
    initialization: str
    trajectory_count: int
    independent_cluster_count: int
    trajectory_id_sha256: str
    observation_sha256: str
    mean_gain_bb_per_hand: float
    standard_error_bb_per_hand: float
    confidence_level: float
    confidence_interval_bb_per_hand: tuple[float, float]
    per_seat_gain_bb_per_hand: tuple[float, ...]
    per_seat_standard_error_bb_per_hand: tuple[float, ...]
    per_seat_confidence_interval_bb_per_hand: tuple[tuple[float, float], ...]
    sum_achieved_response_gain_bb_per_hand: float
    sum_achieved_response_gain_standard_error_bb_per_hand: float
    sum_achieved_response_gain_confidence_interval_bb_per_hand: tuple[float, float]
    sampling_unit: str
    observations: tuple[BRProbeObservation, ...]
    elapsed_seconds: float
    gpu_hours: float
    device: str = ""

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_br_probe_observations(
    *,
    players: int,
    protocol_version: str,
    split_namespace: str,
    evaluation_seed: int,
    target_sha256: str,
    response_sha256: str,
    initial_model_sha256: str,
    initialization: str,
    trajectory_ids: Tensor,
    mixed_payoff: Tensor,
    baseline_payoff: Tensor,
    device: str = "",
    elapsed_seconds: float = 0.0,
) -> BRProbeReport:
    """Reduce paired returns under the fixed-target protocol-v2 law."""

    if protocol_version != EVALUATION_PROTOCOL_VERSION:
        raise ValueError("unsupported BR evaluation protocol version")
    if not split_namespace:
        raise ValueError("evaluation split namespace must be nonempty")
    seats = probe_evaluation_design(trajectory_ids, players)
    if (
        mixed_payoff.shape != trajectory_ids.shape
        or baseline_payoff.shape != trajectory_ids.shape
        or mixed_payoff.device != trajectory_ids.device
        or baseline_payoff.device != trajectory_ids.device
    ):
        raise ValueError("paired payoff vectors must align to trajectory ids")
    difference = mixed_payoff.float() - baseline_payoff.float()
    statistics = fixed_target_paired_statistics(
        difference,
        seats,
        players,
        cluster_ids=trajectory_ids,
    )
    ids_cpu = trajectory_ids.detach().to("cpu", torch.int64)
    seats_cpu = seats.detach().to("cpu", torch.int64)
    mixed_cpu = mixed_payoff.detach().to("cpu", torch.float64)
    baseline_cpu = baseline_payoff.detach().to("cpu", torch.float64)
    rows = tuple(
        BRProbeObservation(
            logical_deal_id=int(logical_id),
            probe_seat=int(seat),
            button=int(logical_id) % players,
            mixed_payoff_bb=float(mixed),
            baseline_payoff_bb=float(baseline),
            gain_bb_per_hand=float(mixed - baseline),
        )
        for logical_id, seat, mixed, baseline in zip(
            ids_cpu.tolist(),
            seats_cpu.tolist(),
            mixed_cpu.tolist(),
            baseline_cpu.tolist(),
            strict=True,
        )
    )
    observation_payload = tuple(
        (
            row.logical_deal_id,
            row.probe_seat,
            row.button,
            row.mixed_payoff_bb,
            row.baseline_payoff_bb,
            row.gain_bb_per_hand,
        )
        for row in rows
    )
    return BRProbeReport(
        semantics=FIXED_TARGET_RESPONSE_SEMANTICS,
        protocol_version=protocol_version,
        split_namespace=split_namespace,
        evaluation_seed=evaluation_seed,
        target_range_semantics=TARGET_RANGE_SEMANTICS,
        response_training_range_semantics=RESPONSE_TRAINING_RANGE_SEMANTICS,
        target_sha256=target_sha256,
        response_sha256=response_sha256,
        initial_model_sha256=initial_model_sha256,
        initialization=initialization,
        trajectory_count=trajectory_ids.numel(),
        independent_cluster_count=statistics.independent_cluster_count,
        trajectory_id_sha256=hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest(),
        observation_sha256=hashlib.sha256(
            json.dumps(
                observation_payload,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        mean_gain_bb_per_hand=statistics.mean_gain,
        standard_error_bb_per_hand=statistics.standard_error,
        confidence_level=statistics.confidence_level,
        confidence_interval_bb_per_hand=statistics.confidence_interval,
        per_seat_gain_bb_per_hand=statistics.per_seat_gain,
        per_seat_standard_error_bb_per_hand=statistics.per_seat_standard_error,
        per_seat_confidence_interval_bb_per_hand=(
            statistics.per_seat_confidence_interval
        ),
        sum_achieved_response_gain_bb_per_hand=(
            statistics.sum_achieved_response_gain
        ),
        sum_achieved_response_gain_standard_error_bb_per_hand=(
            statistics.sum_achieved_response_gain_standard_error
        ),
        sum_achieved_response_gain_confidence_interval_bb_per_hand=(
            statistics.sum_achieved_response_gain_confidence_interval
        ),
        sampling_unit=statistics.sampling_unit,
        observations=rows,
        elapsed_seconds=elapsed_seconds,
        gpu_hours=elapsed_seconds / 3600.0,
        device=device,
    )


def _uniform(deal: DealIdentity, decision: int) -> float:
    encoded = (
        f"{deal.namespace_seed('action')}:{deal.logical_deal_id}:"
        f"{decision}:paired-action"
    ).encode()
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
    deal: DealIdentity,
    button: int,
    probe_seat: int,
    mixed: bool,
) -> float:
    """Play with separate actual-profile and fixed-target reference ranges."""

    target_hand = target.new_hand(button, deal=deal)
    probe_hand = probe.new_hand(button, deal=deal)
    _assert_same_deal(target, target_hand, probe, probe_hand)
    decision_index = 0
    while not target.is_terminal(target_hand):
        if target.needs_chance(target_hand):
            # Street chance uses the logical deal and street, not call order or
            # the number of decisions taken along a possibly divergent path.
            target.deal_keyed(target_hand, deal=deal, decision_index=0)
            probe.deal_keyed(probe_hand, deal=deal, decision_index=0)
            continue
        actor = target.actor(target_hand)
        target_actions, target_policy = target.policy_table(target_hand)
        if mixed and actor == probe_seat:
            actual_actions, actual_policy = probe.policy_table(probe_hand)
            if tuple(action.event_id for action in actual_actions) != tuple(
                action.event_id for action in target_actions
            ):
                raise RuntimeError("candidate and target legal action sets differ")
            actions = actual_actions
            selected_hand = probe_hand
        else:
            actions = target_actions
            actual_policy = target_policy
            selected_hand = target_hand
        private = (
            probe.private_policy_index(selected_hand, actor)
            if selected_hand is probe_hand
            else target.private_policy_index(selected_hand, actor)
        )
        columns = torch.tensor(
            [action.event_id for action in actions],
            dtype=torch.int64,
            device=actual_policy.device,
        )
        action = _select_action(
            actual_policy[private].index_select(0, columns),
            actions,
            uniform=_uniform(deal, decision_index),
        )
        # The actual mixed profile tracks the policy that generated the action.
        probe.advance(
            probe_hand,
            action,
            policy_probabilities=actual_policy,
            zero_probability_extension="error",
        )
        # The frozen target's policy definition tracks only its own reference
        # policy. Off-support public histories retain the prior target range.
        target.advance(
            target_hand,
            action,
            policy_probabilities=target_policy,
            zero_probability_extension="retain_prior",
        )
        decision_index += 1
    payoff = target.terminal_payoffs(target_hand)
    return float(payoff[probe_seat].detach().cpu())


def _validate_sources(
    probe: dict[str, Any],
    target: dict[str, Any],
) -> tuple[int, str, str, str, str]:
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
        str(probe["source_checkpoint_sha256"]),
        str(metadata["initial_model_sha256"]),
        str(initialization),
    )


def evaluate_br_probe(
    probe_artifact: str | Path,
    *,
    target_artifact: str | Path,
    n_hands: int,
    split_namespace: str,
    seed: int,
    device: str | torch.device,
    logical_deal_start: int = 0,
) -> BRProbeReport:
    """Execute balanced fixed-target paired response-gain rollouts on CUDA."""

    started = time.perf_counter()
    requested = resolve_execution_device(
        device, workload="best-response evaluation"
    )
    if requested.type != "cuda":
        raise RuntimeError(
            "BR evaluation requires CUDA completion/showdown operations"
        )
    if not split_namespace:
        raise ValueError("evaluation split namespace must be nonempty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("evaluation seed must be an integer")
    probe_source = load_inference_source(probe_artifact)
    target_source = load_inference_source(target_artifact)
    (
        players,
        target_hash,
        response_hash,
        initial_hash,
        initialization,
    ) = _validate_sources(probe_source, target_source)
    block = players * players
    if (
        isinstance(n_hands, bool)
        or not isinstance(n_hands, int)
        or n_hands <= 1
        or n_hands % block
    ):
        raise ValueError("n_hands must be a positive multiple of P^2")
    if (
        isinstance(logical_deal_start, bool)
        or not isinstance(logical_deal_start, int)
        or logical_deal_start < 0
        or logical_deal_start % block
    ):
        raise ValueError("logical_deal_start must be a nonnegative P^2 boundary")
    probe_mixed = PokerInferenceSession(probe_source, device=requested)
    target_mixed = PokerInferenceSession(target_source, device=requested)
    # A distinct pair owns baseline state; all four sessions use the same
    # explicit logical deals and schedule-independent random addresses.
    probe_baseline = PokerInferenceSession(target_source, device=requested)
    target_baseline = PokerInferenceSession(target_source, device=requested)
    ids = torch.arange(
        logical_deal_start,
        logical_deal_start + n_hands,
        dtype=torch.int64,
        device=target_mixed.device,
    )
    seats = probe_evaluation_design(ids, players)
    mixed_values = []
    baseline_values = []
    for logical_deal_id, seat in zip(
        ids.tolist(), seats.tolist(), strict=True
    ):
        deal = DealIdentity(
            EVALUATION_PROTOCOL_VERSION,
            split_namespace,
            seed,
            logical_deal_id,
        )
        button = logical_deal_id % players
        mixed_values.append(
            _play_one(
                probe_mixed,
                target_mixed,
                deal=deal,
                button=button,
                probe_seat=seat,
                mixed=True,
            )
        )
        baseline_values.append(
            _play_one(
                probe_baseline,
                target_baseline,
                deal=deal,
                button=button,
                probe_seat=seat,
                mixed=False,
            )
        )
    return evaluate_br_probe_observations(
        players=players,
        protocol_version=EVALUATION_PROTOCOL_VERSION,
        split_namespace=split_namespace,
        evaluation_seed=seed,
        target_sha256=target_hash,
        response_sha256=response_hash,
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
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "BRProbeObservation",
    "BRProbeReport",
    "EVALUATION_PROTOCOL_VERSION",
    "RESPONSE_TRAINING_RANGE_SEMANTICS",
    "TARGET_RANGE_SEMANTICS",
    "evaluate_br_probe",
    "evaluate_br_probe_observations",
]
