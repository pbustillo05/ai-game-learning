"""Bounded, resumable CUDA evaluation against one fixed opponent policy.

The default estimate is the paired payoff difference between a candidate and
its initialization baseline. Both policies face the same fixed opponent on the
same explicit deals, seats, buttons, and action-randomness addresses. Raw-match
mode instead reports candidate payoff directly and skips baseline simulation.

Campaign snapshots are intentionally experiment-local mappings rather than
production inference artifacts. Every snapshot must explicitly declare the
historical July preflop input convention; the adapter is installed on every
inference session in either mode.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.training_diagnosis.probe_policies import (
    install_legacy_preflop_inputs,
)
from pbg.core.persistence import model_state_sha256, sha256_file
from pbg.devices import resolve_execution_device
from pbg.eval.br_probe import _play_one
from pbg.games.poker.queued.response import (
    fixed_target_paired_statistics,
    probe_evaluation_design,
)
from pbg.inference import DealIdentity, PokerInferenceSession

EVALUATION_SCHEMA = "hyperparameter-fixed-opponent-evaluation-v1"
RAW_MATCH_SCHEMA = "hyperparameter-raw-fixed-opponent-match-v1"
DEAL_PROTOCOL = "hyperparameter-fixed-opponent-paired-deal-v1"
RAW_MATCH_DEAL_PROTOCOL = "hyperparameter-raw-match-deal-v1"
COMPARISON_SEMANTICS = "paired_policy_payoff_difference_vs_fixed_opponent_v1"
RAW_MATCH_SEMANTICS = "raw_policy_payoff_vs_fixed_opponent_v1"
LEGACY_INPUT_SEMANTICS = "july_exact_hand_preflop_reach_gauge_v1"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_campaign_snapshot(path: str | Path) -> dict[str, Any]:
    """Load the small experiment-owned inference mapping used by the campaign."""

    source_path = Path(path).resolve()
    payload = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"campaign snapshot {source_path} is not a mapping")
    model = payload.get("model")
    scientific = payload.get("scientific_config")
    metadata = payload.get("metadata", {})
    if (
        not isinstance(model, dict)
        or not model
        or not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in model.items()
        )
    ):
        raise RuntimeError(f"campaign snapshot {source_path} has no model state")
    if not isinstance(scientific, dict):
        raise TypeError(
            f"campaign snapshot {source_path} has no scientific_config mapping"
        )
    if not isinstance(metadata, dict):
        raise TypeError(f"campaign snapshot {source_path} has no metadata mapping")
    if (
        payload.get("legacy_preflop_inputs", metadata.get("legacy_preflop_inputs"))
        is not True
    ):
        raise RuntimeError(
            f"campaign snapshot {source_path} must declare legacy_preflop_inputs=true"
        )
    iteration = payload.get("iteration", metadata.get("iteration"))
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise RuntimeError(
            f"campaign snapshot {source_path} needs a nonnegative iteration"
        )
    file_hash = sha256_file(source_path)
    metadata = {
        **metadata,
        "legacy_preflop_inputs": True,
        "input_encoding": payload.get("input_encoding", metadata.get("input_encoding")),
        "utility": payload.get("utility", metadata.get("utility")),
        "source_checkpoint_sha256": payload.get(
            "source_checkpoint_sha256", metadata.get("source_checkpoint_sha256")
        ),
        "policy_role": payload.get("policy_role", metadata.get("policy_role")),
        "initialization_anchor": payload.get(
            "initialization_anchor", metadata.get("initialization_anchor")
        ),
    }
    if metadata["input_encoding"] not in (None, "july_exact_hand_preflop_v1"):
        raise RuntimeError(
            f"campaign snapshot {source_path} has an unsupported input_encoding"
        )
    return {
        "path": str(source_path),
        "file_sha256": file_hash,
        "model_sha256": model_state_sha256(model),
        "iteration": iteration,
        "metadata": metadata,
        "checkpoint_metadata": {
            "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
            "checkpoint": payload.get(
                "checkpoint_metadata", metadata.get("checkpoint")
            ),
            "utility": metadata["utility"],
            "input_encoding": metadata["input_encoding"],
        },
        "session_source": {
            "model": model,
            "scientific_config": scientific,
            "iteration": iteration,
            "source_checkpoint_sha256": file_hash,
        },
    }


def _initialization_anchor(candidate: dict[str, Any]) -> Path:
    metadata = candidate["metadata"]
    raw = metadata.get("initialization_anchor")
    if isinstance(raw, dict):
        raw = raw.get("path")
    if raw is None:
        # Temporary aliases make snapshots produced by early campaign drivers
        # readable while keeping ``initialization_anchor`` the documented key.
        for key in ("initialization_anchor_path", "initial_model_path"):
            if metadata.get(key) is not None:
                raw = metadata[key]
                break
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(
            "--baseline was omitted and candidate metadata has no "
            "initialization_anchor path"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = Path(candidate["path"]).parent / path
    return path.resolve()


def _validate_geometry(*sources: dict[str, Any]) -> int:
    scientific = [source["session_source"]["scientific_config"] for source in sources]
    games = [value.get("game") for value in scientific]
    models = [value.get("model") for value in scientific]
    if any(not isinstance(game, dict) for game in games):
        raise RuntimeError("campaign snapshot scientific config has no game mapping")
    if any(not isinstance(model, dict) for model in models):
        raise RuntimeError("campaign snapshot scientific config has no model mapping")
    if games[1:] != games[:-1]:
        raise RuntimeError("candidate, opponent, and baseline game geometries differ")
    if models[1:] != models[:-1]:
        raise RuntimeError("candidate, opponent, and baseline model contracts differ")
    players = games[0].get("players")
    if players != 2:
        raise RuntimeError(
            "the hyperparameter campaign evaluator currently requires heads-up"
        )
    return 2


def _source_identity(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": source["path"],
        "file_sha256": source["file_sha256"],
        "model_sha256": source["model_sha256"],
        "iteration": source["iteration"],
        "checkpoint_metadata": source["checkpoint_metadata"],
        "policy_role": source["metadata"].get("policy_role"),
    }


def _manifest(
    *,
    candidate: dict[str, Any],
    opponent: dict[str, Any],
    baseline: dict[str, Any],
    hands: int,
    seed: int,
    split: str,
    device: str,
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_SCHEMA,
        "semantics": COMPARISON_SEMANTICS,
        "deal_protocol": DEAL_PROTOCOL,
        "fixed_opponent_range_semantics": (
            "opponent_reference_range_board_conditioned_with_off_support_prior_retention"
        ),
        "legacy_input_semantics": LEGACY_INPUT_SEMANTICS,
        "candidate": _source_identity(candidate),
        "opponent": _source_identity(opponent),
        "baseline": _source_identity(baseline),
        "requested_hands": hands,
        "seed": seed,
        "split": split,
        "device": device,
        "players": 2,
        "candidate_path": candidate["path"],
        "opponent_path": opponent["path"],
        "baseline_path": baseline["path"],
        "candidate_sha256": candidate["file_sha256"],
        "opponent_sha256": opponent["file_sha256"],
        "baseline_sha256": baseline["file_sha256"],
    }


def _raw_match_manifest(
    *,
    candidate: dict[str, Any],
    opponent: dict[str, Any],
    hands: int,
    seed: int,
    split: str,
    device: str,
) -> dict[str, Any]:
    return {
        "schema": RAW_MATCH_SCHEMA,
        "semantics": RAW_MATCH_SEMANTICS,
        "raw_match": True,
        "deal_protocol": RAW_MATCH_DEAL_PROTOCOL,
        "fixed_opponent_range_semantics": (
            "opponent_reference_range_board_conditioned_with_off_support_prior_retention"
        ),
        "legacy_input_semantics": LEGACY_INPUT_SEMANTICS,
        "candidate": _source_identity(candidate),
        "opponent": _source_identity(opponent),
        "requested_hands": hands,
        "seed": seed,
        "split": split,
        "device": device,
        "players": 2,
        "candidate_path": candidate["path"],
        "opponent_path": opponent["path"],
        "baseline_path": None,
        "candidate_sha256": candidate["file_sha256"],
        "opponent_sha256": opponent["file_sha256"],
        "baseline_sha256": None,
        "baseline_evaluated": False,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_or_create_state(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        state = {
            **manifest,
            "status": "incomplete",
            "completed_hands": 0,
            "observations": [],
            "elapsed_seconds": 0.0,
        }
        _atomic_json(path, state)
        return state
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"evaluation output {path} is unreadable") from exc
    if not isinstance(state, dict):
        raise TypeError(f"evaluation output {path} is not a mapping")
    if state.get("status") == "complete":
        raise FileExistsError(f"refusing to overwrite completed evaluation {path}")
    for key, expected in manifest.items():
        if state.get(key) != expected:
            raise RuntimeError(f"evaluation resume invariant differs for {key}")
    observations = state.get("observations")
    if not isinstance(observations, list):
        raise TypeError("evaluation resume observations are invalid")
    for expected_id, row in enumerate(observations):
        if not isinstance(row, dict) or row.get("logical_deal_id") != expected_id:
            raise RuntimeError("evaluation resume deal ids are not consecutive")
        if row.get("probe_seat") != (expected_id // 2) % 2:
            raise RuntimeError("evaluation resume probe-seat schedule is invalid")
        if row.get("button") != expected_id % 2:
            raise RuntimeError("evaluation resume button schedule is invalid")
    if len(observations) > manifest["requested_hands"]:
        raise RuntimeError("evaluation resume has more deals than requested")
    if state.get("completed_hands") != len(observations):
        raise RuntimeError("evaluation resume completed_hands is inconsistent")
    return state


def _statistics(
    values: torch.Tensor, seats: torch.Tensor, ids: torch.Tensor
) -> dict[str, Any]:
    raw = asdict(
        fixed_target_paired_statistics(
            values.to(torch.float64), seats, 2, cluster_ids=ids
        )
    )
    return {
        "mean_bb_per_hand": raw["mean_gain"],
        "standard_error_bb_per_hand": raw["standard_error"],
        "confidence_level": raw["confidence_level"],
        "confidence_interval_bb_per_hand": raw["confidence_interval"],
        "per_seat_mean_bb_per_hand": raw["per_seat_gain"],
        "per_seat_standard_error_bb_per_hand": raw["per_seat_standard_error"],
        "per_seat_confidence_interval_bb_per_hand": raw["per_seat_confidence_interval"],
        "independent_cluster_count": raw["independent_cluster_count"],
        "sampling_unit": raw["sampling_unit"],
    }


def summarize_observations(state: dict[str, Any]) -> dict[str, Any]:
    """Return cluster-aware estimates for one complete paired design."""

    requested = state["requested_hands"]
    observations = state["observations"]
    if len(observations) != requested:
        raise ValueError("cannot summarize an incomplete evaluation")
    ids = torch.tensor(
        [row["logical_deal_id"] for row in observations], dtype=torch.int64
    )
    seats = probe_evaluation_design(ids, 2)
    candidate = torch.tensor(
        [row["candidate_payoff_bb"] for row in observations], dtype=torch.float64
    )
    baseline = torch.tensor(
        [row["baseline_payoff_bb"] for row in observations], dtype=torch.float64
    )
    difference = candidate - baseline
    candidate_stats = _statistics(candidate, seats, ids)
    baseline_stats = _statistics(baseline, seats, ids)
    paired_stats = _statistics(difference, seats, ids)
    result: dict[str, Any] = {
        "candidate_payoff_vs_fixed_opponent": candidate_stats,
        "baseline_payoff_vs_fixed_opponent": baseline_stats,
        "paired_candidate_minus_baseline": paired_stats,
        "mean_candidate_minus_baseline_bb_per_hand": paired_stats["mean_bb_per_hand"],
        "confidence_interval_candidate_minus_baseline_bb_per_hand": paired_stats[
            "confidence_interval_bb_per_hand"
        ],
        "direct_output_parity": {
            "same_model": state["candidate"]["model_sha256"]
            == state["baseline"]["model_sha256"],
            "all_payoffs_identical": bool(candidate.equal(baseline)),
            "hands_checked": requested,
        },
        "interpretation": (
            "Signed candidate-minus-baseline payoff difference against one fixed "
            "opponent under paired explicit deals; this is not exploitability."
        ),
    }
    target_as_baseline = (
        state["baseline"]["model_sha256"] == state["opponent"]["model_sha256"]
    )
    if target_as_baseline:
        result.update(
            mean_achieved_response_gain_bb_per_hand=paired_stats["mean_bb_per_hand"],
            confidence_interval_achieved_response_gain_bb_per_hand=paired_stats[
                "confidence_interval_bb_per_hand"
            ],
            mean_attack_gain_bb_per_hand=paired_stats["mean_bb_per_hand"],
            confidence_interval_attack_gain_bb_per_hand=paired_stats[
                "confidence_interval_bb_per_hand"
            ],
            achieved_response_interpretation=(
                "Candidate gain over target self-play against the same fixed "
                "target. When the candidate is a budget-matched trained response, "
                "this is achieved attack gain; it is not exploitability or a "
                "claim of an exact best response."
            ),
        )
    return result


def summarize_raw_match_observations(state: dict[str, Any]) -> dict[str, Any]:
    """Return raw candidate payoff statistics for one complete match."""

    requested = state["requested_hands"]
    observations = state["observations"]
    if len(observations) != requested:
        raise ValueError("cannot summarize an incomplete raw match")
    ids = torch.tensor(
        [row["logical_deal_id"] for row in observations], dtype=torch.int64
    )
    seats = probe_evaluation_design(ids, 2)
    candidate = torch.tensor(
        [row["candidate_payoff_bb"] for row in observations], dtype=torch.float64
    )
    statistics = _statistics(candidate, seats, ids)
    return {
        "candidate_payoff_vs_fixed_opponent": statistics,
        "mean_candidate_bb_per_hand": statistics["mean_bb_per_hand"],
        "confidence_interval_candidate_bb_per_hand": statistics[
            "confidence_interval_bb_per_hand"
        ],
        "baseline_evaluated": False,
        "interpretation": (
            "Raw candidate payoff against one fixed opponent under explicit "
            "seat-balanced deals."
        ),
    }


def evaluate_fixed_opponent(
    candidate_path: str | Path,
    *,
    opponent_path: str | Path,
    baseline_path: str | Path | None = None,
    hands: int,
    seed: int,
    split: str,
    output_path: str | Path,
    device: str | torch.device = "cuda:0",
    max_new_hands: int | None = None,
    time_limit_seconds: float | None = None,
    raw_match: bool = False,
) -> dict[str, Any]:
    """Run or resume a seat-balanced fixed-opponent evaluation."""

    hands = _positive_int(hands, "hands")
    if hands % 4:
        raise ValueError("hands must be a positive multiple of 4")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be nonempty")
    if not isinstance(raw_match, bool):
        raise TypeError("raw_match must be boolean")
    if raw_match and baseline_path is not None:
        raise ValueError("raw-match evaluation does not accept a baseline")
    if max_new_hands is not None:
        max_new_hands = _positive_int(max_new_hands, "max_new_hands")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, (int, float))
        or time_limit_seconds <= 0
    ):
        raise ValueError("time_limit_seconds must be positive")
    requested_device = resolve_execution_device(
        device, workload="fixed-opponent evaluation"
    )
    if requested_device.type != "cuda":
        raise RuntimeError("fixed-opponent evaluation requires CUDA")

    invocation_started = time.perf_counter()
    interval_started = invocation_started
    candidate = load_campaign_snapshot(candidate_path)
    opponent = load_campaign_snapshot(opponent_path)
    baseline: dict[str, Any] | None = None
    if raw_match:
        _validate_geometry(candidate, opponent)
        manifest = _raw_match_manifest(
            candidate=candidate,
            opponent=opponent,
            hands=hands,
            seed=seed,
            split=split,
            device=str(requested_device),
        )
    else:
        if baseline_path is None:
            baseline_path = _initialization_anchor(candidate)
        baseline = load_campaign_snapshot(baseline_path)
        _validate_geometry(candidate, opponent, baseline)
        manifest = _manifest(
            candidate=candidate,
            opponent=opponent,
            baseline=baseline,
            hands=hands,
            seed=seed,
            split=split,
            device=str(requested_device),
        )

    destination = Path(output_path).resolve()
    state = _load_or_create_state(destination, manifest)
    begin = len(state["observations"])
    remaining = hands - begin
    allowance = remaining if max_new_hands is None else min(remaining, max_new_hands)
    if allowance:
        sessions = [
            PokerInferenceSession(candidate["session_source"], device=requested_device),
            PokerInferenceSession(opponent["session_source"], device=requested_device),
        ]
        if baseline is not None:
            sessions.extend(
                (
                    PokerInferenceSession(
                        baseline["session_source"], device=requested_device
                    ),
                    PokerInferenceSession(
                        opponent["session_source"], device=requested_device
                    ),
                )
            )
        for session in sessions:
            install_legacy_preflop_inputs(session)
        candidate_session, candidate_target = sessions[:2]
        try:
            for logical_deal_id in range(begin, begin + allowance):
                if (
                    logical_deal_id > begin
                    and time_limit_seconds is not None
                    and time.perf_counter() - invocation_started >= time_limit_seconds
                ):
                    break
                seat = (logical_deal_id // 2) % 2
                button = logical_deal_id % 2
                protocol = RAW_MATCH_DEAL_PROTOCOL if raw_match else DEAL_PROTOCOL
                deal = DealIdentity(protocol, split, seed, logical_deal_id)
                candidate_payoff = _play_one(
                    candidate_session,
                    candidate_target,
                    deal=deal,
                    button=button,
                    probe_seat=seat,
                    mixed=True,
                )
                observation = {
                    "logical_deal_id": logical_deal_id,
                    "probe_seat": seat,
                    "button": button,
                    "candidate_payoff_bb": candidate_payoff,
                }
                if baseline is not None:
                    baseline_payoff = _play_one(
                        sessions[2],
                        sessions[3],
                        deal=deal,
                        button=button,
                        probe_seat=seat,
                        mixed=True,
                    )
                    if (
                        candidate["model_sha256"] == baseline["model_sha256"]
                        and candidate_payoff != baseline_payoff
                    ):
                        raise RuntimeError(
                            "identity parity failed for identical candidate and baseline models"
                        )
                    observation.update(
                        baseline_payoff_bb=baseline_payoff,
                        candidate_minus_baseline_bb=(
                            candidate_payoff - baseline_payoff
                        ),
                    )
                state["observations"].append(observation)
                state["completed_hands"] = len(state["observations"])
                state["elapsed_seconds"] = float(state["elapsed_seconds"]) + (
                    time.perf_counter() - interval_started
                )
                interval_started = time.perf_counter()
                _atomic_json(destination, state)
        finally:
            del sessions
            del candidate_session, candidate_target
            gc.collect()

    if len(state["observations"]) == hands:
        summary = (
            summarize_raw_match_observations(state)
            if raw_match
            else summarize_observations(state)
        )
        state.update(summary)
        state["status"] = "complete"
        _atomic_json(destination, state)
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--hands", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--raw-match",
        action="store_true",
        help="Evaluate only raw candidate payoff and skip empirical baseline games.",
    )
    parser.add_argument(
        "--max-new-hands",
        type=int,
        help="Stop after appending at most this many new deals; resume with the same command.",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        help="Stop between deals after this wall-time budget; resume with the same command.",
    )
    args = parser.parse_args(argv)
    result = evaluate_fixed_opponent(
        args.candidate,
        opponent_path=args.opponent,
        baseline_path=args.baseline,
        hands=args.hands,
        seed=args.seed,
        split=args.split,
        output_path=args.output,
        device=args.device,
        max_new_hands=args.max_new_hands,
        time_limit_seconds=args.time_limit_seconds,
        raw_match=args.raw_match,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": result["status"],
                "completed_hands": result["completed_hands"],
                "requested_hands": result["requested_hands"],
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "COMPARISON_SEMANTICS",
    "DEAL_PROTOCOL",
    "EVALUATION_SCHEMA",
    "RAW_MATCH_DEAL_PROTOCOL",
    "RAW_MATCH_SCHEMA",
    "RAW_MATCH_SEMANTICS",
    "evaluate_fixed_opponent",
    "load_campaign_snapshot",
    "summarize_observations",
    "summarize_raw_match_observations",
]
