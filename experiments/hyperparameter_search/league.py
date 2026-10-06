"""Frozen, balanced round-robin crossplay for campaign policies.

Every policy, including both private-estimator conditions and every paired
training seed, enters one shared opponent population. Each unordered policy
pair is played once with a seat-balanced explicit-deal schedule. The reverse
matrix entry is the zero-sum payoff from those same games, and self-play is the
declared zero diagonal.

Selection uses raw policy payoff against the uniform frozen population. The
raw-match evaluation path simulates each matchup once and does not run an
empirical self-play baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from experiments.hyperparameter_search.campaign import (
    ESTIMATORS,
    PYTHON,
    Job,
    atomic_json,
    digest,
    immutable_json,
    validate_evaluation,
)
from pbg.games.poker.queued.response import (
    fixed_target_paired_statistics,
    probe_evaluation_design,
)

if TYPE_CHECKING:
    from experiments.hyperparameter_search.campaign import Campaign

LEAGUE_SCHEMA = "frozen-uniform-policy-round-robin-v1"
STAGE_RECOMMENDED_HANDS = {"screen": 512, "confirm": 2048, "cold": 8192}
_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
_ARM = re.compile(r"[a-z0-9][a-z0-9_]*")


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _evaluation_seed(campaign: Campaign, stage: str, split: str) -> int:
    settings = campaign.profile.get("campaign", {})
    configured = settings.get(f"{stage}_league_seed", settings.get("league_seed"))
    if configured is not None:
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int)
            or configured < 0
        ):
            raise ValueError("league seed must be a nonnegative integer")
        return configured
    material = {
        "schema": LEAGUE_SCHEMA,
        "profile_sha256": digest(campaign.profile),
        "stage": stage,
        "split": split,
    }
    return int(digest(material)[:15], 16)


def _freeze_targets(targets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(targets, dict) or len(targets) < 2:
        raise ValueError("League requires at least two policies")
    frozen: dict[str, dict[str, Any]] = {}
    seen_conditions: set[tuple[str, str, int]] = set()
    for policy_id in sorted(targets):
        if not isinstance(policy_id, str) or _TOKEN.fullmatch(policy_id) is None:
            raise ValueError(f"Unsafe league policy id: {policy_id!r}")
        value = targets[policy_id]
        if not isinstance(value, dict):
            raise TypeError(f"League target {policy_id} must be a mapping")
        arm = value.get("arm")
        mode = value.get("private_estimator")
        seed = value.get("seed")
        if not isinstance(arm, str) or _ARM.fullmatch(arm) is None:
            raise ValueError(f"Unsafe league arm for {policy_id}")
        if mode not in ESTIMATORS:
            raise ValueError(f"Unknown private estimator for {policy_id}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"Training seed for {policy_id} must be an integer")
        condition = (arm, mode, seed)
        if condition in seen_conditions:
            raise ValueError(f"Duplicate arm/mode/seed league condition: {condition}")
        seen_conditions.add(condition)
        snapshot = Path(value.get("snapshot", ""))
        if not snapshot.is_absolute() or not snapshot.is_file():
            raise ValueError(
                f"League snapshot must be an existing absolute file: {snapshot}"
            )
        frozen[policy_id] = {
            "arm": arm,
            "private_estimator": mode,
            "seed": seed,
            "snapshot": str(snapshot.resolve()),
            "snapshot_sha256": _file_sha256(snapshot),
        }

    arms: dict[str, dict[int, set[str]]] = {}
    for value in frozen.values():
        arms.setdefault(value["arm"], {}).setdefault(value["seed"], set()).add(
            value["private_estimator"]
        )
    expected_seed_set: set[int] | None = None
    for arm, seeds in arms.items():
        if any(modes != set(ESTIMATORS) for modes in seeds.values()):
            raise ValueError(
                f"Every seed for arm {arm} must include both estimator modes"
            )
        seed_set = set(seeds)
        if expected_seed_set is None:
            expected_seed_set = seed_set
        elif seed_set != expected_seed_set:
            raise ValueError("All league arms must use identical paired training seeds")
    return frozen


def _orient_pair(
    left: str,
    right: str,
    positions: dict[str, int],
    participant_count: int,
) -> tuple[str, str]:
    first, second = positions[left], positions[right]
    distance = (second - first) % participant_count
    if participant_count % 2:
        return (left, right) if distance <= participant_count // 2 else (right, left)
    if distance < participant_count // 2:
        return left, right
    if distance > participant_count // 2:
        return right, left
    low = min(first, second)
    preferred = low if low % 2 == 0 else (low + participant_count // 2)
    return (left, right) if first == preferred else (right, left)


def balanced_pair_schedule(
    policy_ids: list[str] | tuple[str, ...], seed: int
) -> list[dict[str, Any]]:
    """Return a seeded round-robin order with exact unordered-pair coverage."""

    if len(policy_ids) < 2 or len(set(policy_ids)) != len(policy_ids):
        raise ValueError("Round-robin policy ids must be distinct")
    rng = random.Random(seed)
    participants = sorted(policy_ids)
    rng.shuffle(participants)
    positions = {policy_id: index for index, policy_id in enumerate(participants)}
    real_count = len(participants)
    rotating: list[str | None] = list(participants)
    if len(rotating) % 2:
        rotating.append(None)
    rounds: list[list[tuple[str, str]]] = []
    for _ in range(len(rotating) - 1):
        pairs = []
        for index in range(len(rotating) // 2):
            left = rotating[index]
            right = rotating[-1 - index]
            if left is None or right is None:
                continue
            pairs.append(_orient_pair(left, right, positions, real_count))
        rng.shuffle(pairs)
        rounds.append(pairs)
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    rng.shuffle(rounds)

    schedule = []
    for round_index, pairs in enumerate(rounds):
        for candidate_id, opponent_id in pairs:
            schedule.append(
                {
                    "order": len(schedule),
                    "round": round_index,
                    "candidate_id": candidate_id,
                    "opponent_id": opponent_id,
                }
            )
    observed = {
        frozenset((row["candidate_id"], row["opponent_id"])) for row in schedule
    }
    expected = real_count * (real_count - 1) // 2
    if len(schedule) != expected or len(observed) != expected:
        raise RuntimeError("Balanced round-robin construction lost pair coverage")
    return schedule


def freeze_league(
    campaign: Campaign,
    stage: str,
    targets: dict[str, dict[str, Any]],
    hands: int,
    split: str,
) -> dict[str, Any]:
    """Freeze policy identities and deterministic schedule for safe resume."""

    if not isinstance(stage, str) or _ARM.fullmatch(stage) is None:
        raise ValueError("League stage must be a safe lowercase token")
    if not isinstance(split, str) or not split:
        raise ValueError("League split must be nonempty")
    if isinstance(hands, bool) or not isinstance(hands, int) or hands < 4 or hands % 4:
        raise ValueError("League hands must be a positive multiple of four")
    frozen = _freeze_targets(targets)
    evaluation_seed = _evaluation_seed(campaign, stage, split)
    schedule = balanced_pair_schedule(list(frozen), evaluation_seed)
    manifest = {
        "schema": LEAGUE_SCHEMA,
        "profile_sha256": digest(campaign.profile),
        "stage": stage,
        "split": split,
        "evaluation_seed": evaluation_seed,
        "hands_per_pair": hands,
        "recommended_hands_for_stage": STAGE_RECOMMENDED_HANDS.get(stage),
        "policies": frozen,
        "policy_count": len(frozen),
        "pair_count": len(schedule),
        "population_opponent_weight": 1.0 / len(frozen),
        "self_play_diagonal": 0.0,
        "schedule": schedule,
        "backend": {
            "module": "experiments.hyperparameter_search.evaluation",
            "score_field": "candidate_payoff_vs_fixed_opponent.mean_bb_per_hand",
            "raw_match": True,
            "empirical_baseline_evaluated": False,
        },
    }
    immutable_json(campaign.output / "league" / stage / "manifest.json", manifest)
    return manifest


def _league_job(
    campaign: Campaign, manifest: dict[str, Any], row: dict, gpu: str
) -> Job:
    candidate = manifest["policies"][row["candidate_id"]]
    opponent = manifest["policies"][row["opponent_id"]]
    directory = (
        campaign.output
        / "league"
        / manifest["stage"]
        / "evaluations"
        / f"{row['order']:04d}_{row['candidate_id']}_vs_{row['opponent_id']}"
    )
    arguments = {
        "candidate": candidate["snapshot"],
        "opponent": opponent["snapshot"],
        "raw_match": True,
        "hands": manifest["hands_per_pair"],
        "seed": manifest["evaluation_seed"],
        "split": manifest["split"],
        "device": "cuda:0",
    }
    spec = {
        "schema": "league-evaluation-job-v1",
        "league_manifest_sha256": digest(manifest),
        "order": row["order"],
        "round": row["round"],
        "candidate_id": row["candidate_id"],
        "opponent_id": row["opponent_id"],
        "candidate_sha256": candidate["snapshot_sha256"],
        "opponent_sha256": opponent["snapshot_sha256"],
        "arguments": arguments,
    }
    immutable_json(directory / "job.json", spec)
    command = [str(PYTHON), "-m", "experiments.hyperparameter_search.evaluation"]
    for key, value in arguments.items():
        if key == "raw_match":
            if value is not True:
                raise RuntimeError("League raw-match job lost its mode flag")
            command.append("--raw-match")
        else:
            command += [f"--{key}", str(value)]
    result_path = directory / "result.json"
    command += ["--output", str(result_path)]
    return Job(
        str(directory.relative_to(campaign.output)),
        directory,
        command,
        [gpu],
        result_path,
        digest(spec),
        evaluation=True,
    )


def _validate_cell(value: dict[str, Any], job: Job) -> list[float]:
    validate_evaluation(value)
    spec = json.loads((job.directory / "job.json").read_text())
    arguments = spec["arguments"]
    for role in ("candidate", "opponent"):
        if value.get(f"{role}_path") != arguments[role]:
            raise ValueError(f"League {role} path differs from frozen job")
        if value.get(f"{role}_sha256") != spec[f"{role}_sha256"]:
            raise ValueError(f"League {role} hash differs from frozen job")
    if (
        value.get("raw_match") is not True
        or value.get("baseline_evaluated") is not False
        or value.get("baseline_path") is not None
        or value.get("baseline_sha256") is not None
    ):
        raise ValueError("League evaluation is not a baseline-free raw match")
    for name, expected in (
        ("requested_hands", arguments["hands"]),
        ("seed", arguments["seed"]),
        ("split", arguments["split"]),
    ):
        if value.get(name) != expected:
            raise ValueError(f"League {name} differs from frozen job")

    statistics = value.get("candidate_payoff_vs_fixed_opponent")
    if not isinstance(statistics, dict):
        raise TypeError("League evaluation lacks raw candidate payoff statistics")
    mean = _finite(statistics.get("mean_bb_per_hand"), "raw candidate payoff")
    interval = statistics.get("confidence_interval_bb_per_hand")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("League evaluation lacks raw candidate payoff interval")
    lower = _finite(interval[0], "raw payoff interval")
    upper = _finite(interval[1], "raw payoff interval")
    if lower > upper:
        raise ValueError("League raw payoff interval is reversed")

    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) != arguments["hands"]:
        raise ValueError("League evaluation lacks complete hand-level observations")
    payoffs = []
    for logical_id, observation in enumerate(observations):
        if (
            not isinstance(observation, dict)
            or observation.get("logical_deal_id") != logical_id
            or observation.get("probe_seat") != (logical_id // 2) % 2
            or observation.get("button") != logical_id % 2
        ):
            raise ValueError("League evaluation hand schedule is inconsistent")
        payoffs.append(
            _finite(observation.get("candidate_payoff_bb"), "candidate hand payoff")
        )
    if not math.isclose(sum(payoffs) / len(payoffs), mean, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("League raw payoff summary differs from its observations")
    return payoffs


def _cluster_statistics(values: list[float]) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float64)
    ids = torch.arange(len(values), dtype=torch.int64)
    seats = probe_evaluation_design(ids, 2)
    raw = asdict(fixed_target_paired_statistics(tensor, seats, 2, cluster_ids=ids))
    return {
        "mean_bb_per_hand": raw["mean_gain"],
        "standard_error_bb_per_hand": raw["standard_error"],
        "confidence_level": raw["confidence_level"],
        "confidence_interval_bb_per_hand": list(raw["confidence_interval"]),
        "independent_cluster_count": raw["independent_cluster_count"],
        "sampling_unit": raw["sampling_unit"],
    }


def _cycles(matrix: dict[str, dict[str, float]]) -> list[list[str]]:
    policies = sorted(matrix)
    found: list[list[str]] = []
    for first_index, first in enumerate(policies):
        for second_index in range(first_index + 1, len(policies)):
            second = policies[second_index]
            for third in policies[second_index + 1 :]:
                if (
                    matrix[first][second] > 0
                    and matrix[second][third] > 0
                    and matrix[third][first] > 0
                ):
                    found.append([first, second, third])
                elif (
                    matrix[second][first] > 0
                    and matrix[first][third] > 0
                    and matrix[third][second] > 0
                ):
                    found.append([second, first, third])
    return found


def summarize_league(
    manifest: dict[str, Any],
    cells: dict[int, tuple[dict[str, Any], Job]],
) -> dict[str, Any]:
    """Build raw zero-sum matrix, uniform-population scores, and paired intervals."""

    schedule = manifest["schedule"]
    if set(cells) != {row["order"] for row in schedule}:
        raise ValueError("League cell set is incomplete")
    policies = list(manifest["policies"])
    hands = manifest["hands_per_pair"]
    count = len(policies)
    matrix = {left: {right: 0.0 for right in policies} for left in policies}
    vectors = {
        left: {right: [0.0] * hands if left == right else None for right in policies}
        for left in policies
    }
    per_opponent = {
        left: {
            left: {
                "mean_bb_per_hand": 0.0,
                "confidence_interval_bb_per_hand": [0.0, 0.0],
                "population_weight": 1.0 / count,
                "source": "declared_zero_sum_self_play_diagonal",
            }
        }
        for left in policies
    }
    cell_provenance = []
    for row in schedule:
        value, job = cells[row["order"]]
        payoffs = _validate_cell(value, job)
        candidate_id = row["candidate_id"]
        opponent_id = row["opponent_id"]
        raw = value["candidate_payoff_vs_fixed_opponent"]
        mean = float(raw["mean_bb_per_hand"])
        interval = [float(x) for x in raw["confidence_interval_bb_per_hand"]]
        matrix[candidate_id][opponent_id] = mean
        matrix[opponent_id][candidate_id] = -mean
        vectors[candidate_id][opponent_id] = payoffs
        vectors[opponent_id][candidate_id] = [-payoff for payoff in payoffs]
        common = {
            "population_weight": 1.0 / count,
            "evaluation_path": str(job.result_path.resolve()),
            "evaluation_sha256": digest(value),
            "logical_deals": hands,
        }
        per_opponent[candidate_id][opponent_id] = {
            "mean_bb_per_hand": mean,
            "confidence_interval_bb_per_hand": interval,
            "source": "candidate_payoff_from_played_games",
            **common,
        }
        per_opponent[opponent_id][candidate_id] = {
            "mean_bb_per_hand": -mean,
            "confidence_interval_bb_per_hand": [-interval[1], -interval[0]],
            "source": "zero_sum_opponent_payoff_from_same_played_games",
            **common,
        }
        cell_provenance.append(
            {
                "order": row["order"],
                "candidate_id": candidate_id,
                "opponent_id": opponent_id,
                "result_path": str(job.result_path.resolve()),
                "result_sha256": digest(value),
            }
        )

    for left in policies:
        for right in policies:
            if vectors[left][right] is None:
                raise RuntimeError("League payoff observation matrix is incomplete")
            if not math.isclose(
                matrix[left][right], -matrix[right][left], abs_tol=1e-12
            ):
                raise RuntimeError("League payoff matrix is not zero-sum")

    policy_vectors = {
        policy_id: [
            sum(vectors[policy_id][opponent_id][deal] for opponent_id in policies)
            / count
            for deal in range(hands)
        ]
        for policy_id in policies
    }
    policy_scores = {}
    for policy_id in policies:
        statistics = _cluster_statistics(policy_vectors[policy_id])
        matrix_mean = sum(matrix[policy_id].values()) / count
        if not math.isclose(
            statistics["mean_bb_per_hand"], matrix_mean, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError("League population score disagrees with payoff matrix")
        policy_scores[policy_id] = {
            "arm": manifest["policies"][policy_id]["arm"],
            "private_estimator": manifest["policies"][policy_id]["private_estimator"],
            "seed": manifest["policies"][policy_id]["seed"],
            "mean_population_payoff_bb_per_hand": matrix_mean,
            "population_payoff_confidence": statistics,
            "opponent_count_including_self": count,
            "opponent_weight": 1.0 / count,
            "per_opponent": per_opponent[policy_id],
        }

    arm_ids: dict[str, list[str]] = {}
    for policy_id, identity in manifest["policies"].items():
        arm_ids.setdefault(identity["arm"], []).append(policy_id)
    arm_vectors = {
        arm: [
            sum(policy_vectors[policy_id][deal] for policy_id in ids) / len(ids)
            for deal in range(hands)
        ]
        for arm, ids in arm_ids.items()
    }
    arm_scores = {}
    for arm, ids in arm_ids.items():
        modes = {}
        seeds = {}
        for mode in ESTIMATORS:
            selected = [
                policy_id
                for policy_id in ids
                if manifest["policies"][policy_id]["private_estimator"] == mode
            ]
            modes[mode] = {
                "mean_population_payoff_bb_per_hand": sum(
                    policy_scores[policy_id]["mean_population_payoff_bb_per_hand"]
                    for policy_id in selected
                )
                / len(selected),
                "per_seed": {
                    str(manifest["policies"][policy_id]["seed"]): policy_scores[
                        policy_id
                    ]["mean_population_payoff_bb_per_hand"]
                    for policy_id in selected
                },
            }
        for seed in sorted(
            {manifest["policies"][policy_id]["seed"] for policy_id in ids}
        ):
            selected = [
                policy_id
                for policy_id in ids
                if manifest["policies"][policy_id]["seed"] == seed
            ]
            seeds[str(seed)] = {
                "mean_population_payoff_bb_per_hand": sum(
                    policy_scores[policy_id]["mean_population_payoff_bb_per_hand"]
                    for policy_id in selected
                )
                / len(selected),
                "per_mode": {
                    manifest["policies"][policy_id]["private_estimator"]: policy_scores[
                        policy_id
                    ]["mean_population_payoff_bb_per_hand"]
                    for policy_id in selected
                },
            }
        arm_scores[arm] = {
            "mean_population_payoff_bb_per_hand": sum(
                policy_scores[policy_id]["mean_population_payoff_bb_per_hand"]
                for policy_id in ids
            )
            / len(ids),
            "policy_weight": 1.0 / len(ids),
            "per_mode": modes,
            "paired_seed_scores": seeds,
            "population_payoff_confidence": _cluster_statistics(arm_vectors[arm]),
            "rb_minus_normal_confidence": _cluster_statistics(
                [
                    sum(
                        (
                            1
                            if manifest["policies"][policy_id]["private_estimator"]
                            == "rao_blackwell"
                            else -1
                        )
                        * policy_vectors[policy_id][deal]
                        for policy_id in ids
                    )
                    / (len(ids) / 2)
                    for deal in range(hands)
                ]
            ),
        }

    arms = sorted(arm_scores)
    arm_differences: dict[str, dict[str, Any]] = {arm: {} for arm in arms}
    for left in arms:
        for right in arms:
            if left == right:
                arm_differences[left][right] = {
                    "mean_left_minus_right_bb_per_hand": 0.0,
                    "confidence_interval_bb_per_hand": [0.0, 0.0],
                    "source": "identity",
                }
                continue
            delta = [
                arm_vectors[left][deal] - arm_vectors[right][deal]
                for deal in range(hands)
            ]
            statistics = _cluster_statistics(delta)
            arm_differences[left][right] = {
                "mean_left_minus_right_bb_per_hand": statistics["mean_bb_per_hand"],
                "confidence_interval_bb_per_hand": statistics[
                    "confidence_interval_bb_per_hand"
                ],
                "standard_error_bb_per_hand": statistics["standard_error_bb_per_hand"],
                "independent_cluster_count": statistics["independent_cluster_count"],
                "source": "matched_logical_deal_uniform_population_difference",
            }

    ordered_arms = sorted(
        arms,
        key=lambda arm: (
            -arm_scores[arm]["mean_population_payoff_bb_per_hand"],
            arm,
        ),
    )
    recommended = ordered_arms[0]
    public_arm_scores = {
        arm: {
            "mean_bb_per_hand": score["mean_population_payoff_bb_per_hand"],
            "per_mode": {
                mode: {
                    "mean_bb_per_hand": mode_score[
                        "mean_population_payoff_bb_per_hand"
                    ],
                    "per_seed": mode_score["per_seed"],
                }
                for mode, mode_score in score["per_mode"].items()
            },
            "paired_seed_scores": score["paired_seed_scores"],
            "policy_weight": score["policy_weight"],
            "population_payoff_confidence": score["population_payoff_confidence"],
            "rb_minus_normal_confidence": score["rb_minus_normal_confidence"],
        }
        for arm, score in arm_scores.items()
    }
    cycles = _cycles(matrix)
    return {
        "schema": LEAGUE_SCHEMA,
        "status": "complete",
        "complete": True,
        "stage": manifest["stage"],
        "split": manifest["split"],
        "evaluation_seed": manifest["evaluation_seed"],
        "recommended_arm": recommended,
        "ordered_arms": ordered_arms,
        "provisional": True,
        "selection_criterion": (
            "Highest raw payoff against the same uniform frozen policy population; "
            "both estimator modes and paired training seeds receive equal weight."
        ),
        "arm_scores": public_arm_scores,
        "arm_scores_higher_is_better": public_arm_scores,
        "policy_scores_higher_is_better": policy_scores,
        "payoff_matrix_bb_per_hand": matrix,
        "paired_arm_differences": arm_differences,
        "cycles": cycles,
        "cycle_count": len(cycles),
        "ranking_limitation": (
            "Pairwise crossplay can be non-transitive, so scalar population payoff "
            "is a declared selection rule rather than a total ordering of policies."
        ),
        "uncertainty": {
            "paired_difference_method": (
                "Matched logical-deal cluster intervals after averaging the fixed "
                "uniform opponent population."
            ),
            "conditions_on_frozen_models_and_population": True,
            "includes_training_seed_uncertainty": False,
            "multiple_comparison_adjusted": False,
            "cell_intervals_are_per_opponent": True,
        },
        "uniform_population": {
            "policy_ids": policies,
            "opponent_count_including_self": count,
            "weight_per_opponent": 1.0 / count,
            "self_play_diagonal": 0.0,
        },
        "league_manifest_sha256": digest(manifest),
        "cell_provenance": cell_provenance,
        "evaluation_pairs": manifest["pair_count"],
        "requested_hands_per_pair": hands,
        "requested_candidate_hands_total": manifest["pair_count"] * hands,
        "backend_simulated_games_total": manifest["pair_count"] * hands,
        "backend_baseline_games_evaluated": False,
    }


def run_league(
    campaign: Campaign,
    stage: str,
    targets: dict[str, dict[str, Any]],
    hands: int,
    split: str,
) -> dict[str, Any]:
    """Run or resume a frozen round robin in GPU-disjoint campaign waves."""

    if not campaign.gpus:
        raise ValueError("League requires campaign GPU assignments")
    manifest = freeze_league(campaign, stage, targets, hands, split)
    cells: dict[int, tuple[dict[str, Any], Job]] = {}
    schedule = manifest["schedule"]
    for begin in range(0, len(schedule), len(campaign.gpus)):
        rows = schedule[begin : begin + len(campaign.gpus)]
        jobs = [
            _league_job(campaign, manifest, row, campaign.gpus[index])
            for index, row in enumerate(rows)
        ]
        results = campaign.run_jobs(jobs)
        if len(results) != len(jobs):
            raise ValueError("Campaign returned an incomplete league wave")
        for row, job, value in zip(rows, jobs, results, strict=True):
            _validate_cell(value, job)
            cells[row["order"]] = (value, job)

    result = summarize_league(manifest, cells)
    destination = campaign.output / "league" / stage
    atomic_json(destination / "payoff_matrix.json", result["payoff_matrix_bb_per_hand"])
    atomic_json(destination / "result.json", result)
    return result


__all__ = [
    "LEAGUE_SCHEMA",
    "STAGE_RECOMMENDED_HANDS",
    "balanced_pair_schedule",
    "freeze_league",
    "run_league",
    "summarize_league",
]
