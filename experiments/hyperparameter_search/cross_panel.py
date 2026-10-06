"""Held-out common response pool for paired cold-stage finalists.

Select one response per target using the existing own-target validation scores,
freeze those snapshots, then evaluate the common pool on a fresh split. This
does not train additional responses or provide an exploitability certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.hyperparameter_search.campaign import (
    CI,
    ESTIMATORS,
    GAIN,
    PYTHON,
    Job,
    atomic_json,
    digest,
    immutable_json,
    pool_estimators,
    promote,
    validate_evaluation,
)

if TYPE_CHECKING:
    from experiments.hyperparameter_search.campaign import Campaign

SPLIT = "cross_panel_test_v1"


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _snapshot(result_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    if result.get("complete") is not True:
        raise ValueError(f"Cross-panel input worker is not complete: {result_path}")
    path = Path(result.get("snapshot", ""))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            f"Cross-panel snapshot must be an existing absolute file: {path}"
        )
    return {
        "snapshot": str(path.resolve()),
        "sha256": _file_sha256(path),
        "worker_result": str(result_path.resolve()),
        "worker_result_sha256": digest(result),
    }


def freeze_pool(campaign: Campaign, cold_scores: dict[str, dict]) -> dict:
    """Freeze selected response identities using old scores, never test scores."""
    if not cold_scores:
        raise ValueError("Cross-panel requires cold-stage finalists")
    settings = campaign.profile["campaign"]
    seed = settings["cold_seed"]
    hands = 4 if campaign.smoke else settings.get("cross_panel_hands", 512)
    if isinstance(hands, bool) or not isinstance(hands, int) or hands < 4 or hands % 2:
        raise ValueError("cross_panel_hands must be even and at least four")
    targets, attackers = {}, {}
    for arm in sorted(cold_scores):
        if not isinstance(arm, str) or not re.fullmatch(r"[a-z0-9_]+", arm):
            raise ValueError("Unsafe finalist arm name")
        modes = cold_scores[arm].get("per_mode", {})
        if set(modes) != set(ESTIMATORS):
            raise ValueError("Each finalist needs both estimator conditions")
        for mode in ESTIMATORS:
            target_id = f"{arm}__{mode}"
            directory = campaign.output / "cold" / arm / f"seed_{seed}" / mode
            targets[target_id] = {
                "arm": arm,
                "private_estimator": mode,
                **_snapshot(directory / "result.json"),
            }
            gains = modes[mode].get("restart_mean_gains")
            if (
                not isinstance(gains, list)
                or len(gains) != 3
                or any(
                    not isinstance(x, (float, int)) or not math.isfinite(x)
                    for x in gains
                )
            ):
                raise ValueError(
                    f"Three finite restart selection gains required: {target_id}"
                )
            # A deterministic tie break prevents unnecessary changes on resume.
            selected = max(range(3), key=lambda index: (gains[index], -index))
            attackers[target_id] = {
                "trained_against_target": target_id,
                "selected_restart": selected,
                "selection_gain_bb_per_hand": gains[selected],
                "selection_restart_gains": gains,
                **_snapshot(directory / f"response_{selected}" / "result.json"),
            }
    manifest = {
        "schema": "paired-finalist-common-response-panel-v1",
        "profile_sha256": digest(campaign.profile),
        "cold_scores_sha256": digest(cold_scores),
        "selection_split": "cold_heldout",
        "test_split": SPLIT,
        "seed": seed,
        "hands_per_pair": hands,
        "targets": targets,
        "attackers": attackers,
        "pair_count": len(targets) * len(attackers),
        "additional_training_seconds": 0,
        "note": "Restart selection uses previous own-target scores; all selected responses face every target on a fresh held-out split.",
    }
    immutable_json(campaign.output / "cross_panel" / "manifest.json", manifest)
    return manifest


def _panel_job(
    campaign: Campaign, manifest: dict, target_id: str, attacker_id: str, gpu: str
) -> Job:
    target = manifest["targets"][target_id]
    attacker = manifest["attackers"][attacker_id]
    directory = (
        campaign.output / "cross_panel" / "evaluations" / target_id / attacker_id
    )
    arguments = {
        "candidate": attacker["snapshot"],
        "opponent": target["snapshot"],
        "baseline": target["snapshot"],
        "hands": manifest["hands_per_pair"],
        "seed": manifest["seed"],
        "split": SPLIT,
        "device": "cuda:0",
    }
    spec = {
        "arguments": arguments,
        "panel_manifest_sha256": digest(manifest),
        "target_id": target_id,
        "attacker_id": attacker_id,
        "candidate_sha256": attacker["sha256"],
        "opponent_sha256": target["sha256"],
        "baseline_sha256": target["sha256"],
    }
    immutable_json(directory / "job.json", spec)
    command = [str(PYTHON), "-m", "experiments.hyperparameter_search.evaluation"]
    for key, value in arguments.items():
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


def _validate_pair(value: dict, job: Job) -> None:
    validate_evaluation(value)
    spec = json.loads((job.directory / "job.json").read_text())
    arguments = spec["arguments"]
    for name in ("candidate", "opponent", "baseline"):
        if value.get(f"{name}_path") != arguments[name]:
            raise ValueError(f"Cross-panel {name} path differs from frozen job")
        if value.get(f"{name}_sha256") != spec[f"{name}_sha256"]:
            raise ValueError(f"Cross-panel {name} hash differs from frozen job")
    for name, expected in (
        ("requested_hands", arguments["hands"]),
        ("seed", arguments["seed"]),
        ("split", arguments["split"]),
    ):
        if value.get(name) != expected:
            raise ValueError(f"Cross-panel {name} differs from frozen job")


def summarize_panel(
    manifest: dict, matrix: dict[str, dict[str, dict]], cold_scores: dict
) -> dict:
    target_scores = {}
    paired: dict[str, dict] = {}
    for target_id, identity in manifest["targets"].items():
        rows = matrix.get(target_id, {})
        if set(rows) != set(manifest["attackers"]):
            raise ValueError("Cross-panel matrix is incomplete")
        for value in rows.values():
            validate_evaluation(value)
        strongest = max(sorted(rows), key=lambda key: rows[key][GAIN])
        score = {
            "worst_response_gain_bb_per_hand": float(rows[strongest][GAIN]),
            "worst_response_interval": [
                max(float(value[CI][index]) for value in rows.values())
                for index in (0, 1)
            ],
            "strongest_heldout_attacker": strongest,
            "attacker_mean_gains": {
                key: float(value[GAIN]) for key, value in rows.items()
            },
            "interval_note": "Pointwise maxima of per-attack interval endpoints are descriptive; not multiplicity-adjusted confidence bounds.",
        }
        target_scores[target_id] = score
        paired.setdefault(identity["arm"], {})[identity["private_estimator"]] = score
    shared = {arm: pool_estimators(modes) for arm, modes in paired.items()}
    selection = promote(shared, 1)
    old_winner = min(
        cold_scores,
        key=lambda arm: (cold_scores[arm]["worst_response_gain_bb_per_hand"], arm),
    )
    winner = selection["selected"][0]
    return {
        "complete": True,
        "provisional": True,
        "estimator_selected": False,
        "recommended_arm": winner,
        "own_br_recommended_arm": old_winner,
        "disagrees_with_own_br_ranking": winner != old_winner,
        "target_scores": target_scores,
        "shared_arm_scores": shared,
        "selection": selection,
        "uncertainty": {
            "overlap_with_recommended": selection["uncertainty_overlap_with_leader"],
            "intervals_are_formal_pooled_ci": False,
            "restart_selection_and_test_splits_are_distinct": True,
            "remaining_postselection": "Finalist targets were selected adaptively, and the largest test attack is reported. A point winner is provisional.",
            "limitation": "A finite common response pool can miss exploitable weaknesses; this is achieved attack gain, not exact exploitability.",
        },
        "panel_manifest_sha256": digest(manifest),
        "additional_training_seconds": 0,
        "evaluation_pairs": manifest["pair_count"],
        "requested_hands_total": manifest["pair_count"] * manifest["hands_per_pair"],
    }


def run_cross_panel(campaign: Campaign, cold_scores: dict[str, dict]) -> dict:
    """Evaluate the frozen common pool in GPU-disjoint waves and recommend an arm."""
    if not campaign.gpus:
        raise ValueError("Cross-panel requires campaign GPU assignments")
    manifest = freeze_pool(campaign, cold_scores)
    pairs = [
        (target, attacker)
        for target in manifest["targets"]
        for attacker in manifest["attackers"]
    ]
    matrix: dict[str, dict[str, dict]] = {target: {} for target in manifest["targets"]}
    for begin in range(0, len(pairs), len(campaign.gpus)):
        wave = pairs[begin : begin + len(campaign.gpus)]
        jobs = [
            _panel_job(campaign, manifest, target, attacker, campaign.gpus[index])
            for index, (target, attacker) in enumerate(wave)
        ]
        results = campaign.run_jobs(jobs)
        if len(results) != len(jobs):
            raise ValueError("Campaign returned an incomplete cross-panel wave")
        for (target, attacker), job, value in zip(wave, jobs, results, strict=True):
            _validate_pair(value, job)
            matrix[target][attacker] = value
    summary = summarize_panel(manifest, matrix, cold_scores)
    # Each raw cell already has its own resumable evaluator result.json.
    atomic_json(campaign.output / "cross_panel" / "matrix.json", matrix)
    atomic_json(campaign.output / "cross_panel" / "result.json", summary)
    return summary
