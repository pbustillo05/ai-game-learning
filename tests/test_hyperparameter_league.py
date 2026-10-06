"""CPU-only structural tests for frozen policy league orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.hyperparameter_search import league
from experiments.hyperparameter_search.campaign import (
    atomic_json,
    score_evaluations,
    validate_evaluation,
)


@pytest.fixture
def registered(tmp_path: Path):
    output = tmp_path / "campaign"
    profile = {"campaign": {"league_seed": 761}}
    targets = {}
    for arm in ("arm_a", "arm_b"):
        for mode in league.ESTIMATORS:
            policy_id = f"{arm}__{mode}"
            snapshot = tmp_path / f"{policy_id}.pt"
            snapshot.write_bytes(policy_id.encode())
            targets[policy_id] = {
                "arm": arm,
                "private_estimator": mode,
                "seed": 17,
                "snapshot": str(snapshot.resolve()),
            }
    campaign = SimpleNamespace(
        output=output,
        profile=profile,
        smoke=False,
        gpus=["GPU-a", "GPU-b", "GPU-c"],
    )
    return campaign, targets


def _fake_launcher(campaign, strengths=None, cycle=False, incomplete=False):
    strengths = strengths or {
        "arm_a__sampled": 4.0,
        "arm_a__rao_blackwell": 2.0,
        "arm_b__sampled": 1.0,
        "arm_b__rao_blackwell": -1.0,
    }
    cycle_edges = {
        ("arm_a__sampled", "arm_a__rao_blackwell"): 1.0,
        ("arm_a__rao_blackwell", "arm_b__sampled"): 1.0,
        ("arm_b__sampled", "arm_a__sampled"): 1.0,
    }
    launched, waves = [], []

    def payoff(candidate, opponent):
        if not cycle:
            return strengths[candidate] - strengths[opponent]
        if (candidate, opponent) in cycle_edges:
            return cycle_edges[(candidate, opponent)]
        if (opponent, candidate) in cycle_edges:
            return -cycle_edges[(opponent, candidate)]
        return 0.0

    def run_jobs(jobs):
        waves.append(len(jobs))
        results = []
        for job in jobs:
            if job.result_path.exists():
                value = json.loads(job.result_path.read_text())
                if value.get("status") == "complete":
                    results.append(value)
                    continue
            spec = json.loads((job.directory / "job.json").read_text())
            arguments = spec["arguments"]
            mean = payoff(spec["candidate_id"], spec["opponent_id"])
            pattern = (-0.3, 0.1, -0.1, 0.3)
            observations = [
                {
                    "logical_deal_id": logical_id,
                    "probe_seat": (logical_id // 2) % 2,
                    "button": logical_id % 2,
                    "candidate_payoff_bb": mean + pattern[logical_id % 4],
                }
                for logical_id in range(arguments["hands"])
            ]
            value = {
                "schema": "hyperparameter-raw-fixed-opponent-match-v1",
                "semantics": "raw_policy_payoff_vs_fixed_opponent_v1",
                "raw_match": True,
                "status": "incomplete" if incomplete else "complete",
                "requested_hands": arguments["hands"],
                "completed_hands": (
                    arguments["hands"] - 1 if incomplete else arguments["hands"]
                ),
                "seed": arguments["seed"],
                "split": arguments["split"],
                "candidate_path": arguments["candidate"],
                "opponent_path": arguments["opponent"],
                "baseline_path": None,
                "candidate_sha256": spec["candidate_sha256"],
                "opponent_sha256": spec["opponent_sha256"],
                "baseline_sha256": None,
                "baseline_evaluated": False,
                "candidate_payoff_vs_fixed_opponent": {
                    "mean_bb_per_hand": mean,
                    "confidence_interval_bb_per_hand": [mean - 0.25, mean + 0.25],
                },
                "observations": observations,
            }
            atomic_json(job.result_path, value)
            launched.append(job)
            results.append(value)
        return results

    campaign.run_jobs = run_jobs
    return launched, waves


def test_round_robin_uses_one_shared_pool_raw_payoff_and_gpu_waves(registered):
    campaign, targets = registered
    launched, waves = _fake_launcher(campaign)

    result = league.run_league(
        campaign, "screen", targets, hands=4, split="screen_league_fresh"
    )

    assert len(launched) == 6
    assert waves == [3, 3]
    assert result["recommended_arm"] == "arm_a"
    assert result["ordered_arms"] == ["arm_a", "arm_b"]
    assert result["arm_scores"]["arm_a"]["mean_bb_per_hand"] == pytest.approx(1.5)
    assert result["arm_scores"]["arm_b"]["mean_bb_per_hand"] == pytest.approx(-1.5)
    assert set(result["arm_scores"]["arm_a"]["per_mode"]) == set(league.ESTIMATORS)
    assert result["paired_arm_differences"]["arm_a"]["arm_b"][
        "mean_left_minus_right_bb_per_hand"
    ] == pytest.approx(3.0)
    assert (
        result["paired_arm_differences"]["arm_a"]["arm_b"]["independent_cluster_count"]
        == 4
    )
    for arm in ("arm_a", "arm_b"):
        score = result["arm_scores"][arm]
        assert score["population_payoff_confidence"][
            "mean_bb_per_hand"
        ] == pytest.approx(score["mean_bb_per_hand"])
        assert score["rb_minus_normal_confidence"]["mean_bb_per_hand"] == pytest.approx(
            score["per_mode"]["rao_blackwell"]["mean_bb_per_hand"]
            - score["per_mode"]["sampled"]["mean_bb_per_hand"]
        )
    assert result["provisional"] is True
    assert result["backend_simulated_games_total"] == 24
    assert result["backend_baseline_games_evaluated"] is False

    matrix = result["payoff_matrix_bb_per_hand"]
    assert all(matrix[policy_id][policy_id] == 0 for policy_id in targets)
    for left in targets:
        for right in targets:
            assert matrix[left][right] == pytest.approx(-matrix[right][left])
    assert sum(sum(row.values()) for row in matrix.values()) == pytest.approx(0.0)
    for score in result["policy_scores_higher_is_better"].values():
        assert score["opponent_count_including_self"] == 4
        assert score["opponent_weight"] == 0.25
        assert len(score["per_opponent"]) == 4

    for job in launched:
        spec = json.loads((job.directory / "job.json").read_text())
        assert spec["arguments"]["raw_match"] is True
        assert "baseline" not in spec["arguments"]
        assert "--raw-match" in job.command
        assert job.command.count("--raw-match") == 1
        assert spec["arguments"]["hands"] == 4
        assert spec["arguments"]["seed"] == 761
        assert spec["arguments"]["split"] == "screen_league_fresh"
        assert len(job.gpus) == 1


def test_manifest_and_cells_resume_without_changing_frozen_inputs(registered):
    campaign, targets = registered
    launched, _waves = _fake_launcher(campaign)
    first = league.run_league(campaign, "screen", targets, 4, "fresh")
    second = league.run_league(campaign, "screen", targets, 4, "fresh")
    assert second == first
    assert len(launched) == 6

    changed = Path(targets["arm_a__sampled"]["snapshot"])
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Immutable"):
        league.run_league(campaign, "screen", targets, 4, "fresh")


def test_seeded_schedule_has_exact_balanced_pair_coverage() -> None:
    ids = tuple(f"policy_{index}" for index in range(7))
    first = league.balanced_pair_schedule(ids, 991)
    second = league.balanced_pair_schedule(tuple(reversed(ids)), 991)
    assert first == second
    assert len(first) == 21
    assert (
        len({frozenset((row["candidate_id"], row["opponent_id"])) for row in first})
        == 21
    )
    per_round = {}
    for row in first:
        per_round.setdefault(row["round"], []).extend(
            (row["candidate_id"], row["opponent_id"])
        )
    assert all(len(values) == len(set(values)) == 6 for values in per_round.values())


def test_nontransitive_cycles_are_reported_without_total_order_claim(registered):
    campaign, targets = registered
    _fake_launcher(campaign, cycle=True)
    result = league.run_league(campaign, "screen", targets, 4, "cycle_split")
    assert result["cycle_count"] >= 1
    assert any(
        set(cycle)
        == {
            "arm_a__sampled",
            "arm_a__rao_blackwell",
            "arm_b__sampled",
        }
        for cycle in result["cycles"]
    )
    assert "non-transitive" in result["ranking_limitation"]
    assert "total ordering" in result["ranking_limitation"]


def test_incomplete_match_never_emits_final_result(registered):
    campaign, targets = registered
    _fake_launcher(campaign, incomplete=True)
    with pytest.raises(ValueError, match="complete"):
        league.run_league(campaign, "screen", targets, 4, "incomplete")
    assert not (campaign.output / "league/screen/result.json").exists()


def test_pool_requires_safe_complete_paired_conditions(registered):
    campaign, targets = registered
    missing = dict(targets)
    del missing["arm_a__sampled"]
    with pytest.raises(ValueError, match="both estimator"):
        league.freeze_league(campaign, "screen", missing, 4, "fresh")

    unpaired = {key: dict(value) for key, value in targets.items()}
    unpaired["arm_b__sampled"]["seed"] = 99
    with pytest.raises(ValueError, match="both estimator"):
        league.freeze_league(campaign, "screen", unpaired, 4, "fresh")

    unsafe = dict(targets)
    unsafe["../escape"] = unsafe.pop("arm_a__sampled")
    with pytest.raises(ValueError, match="Unsafe"):
        league.freeze_league(campaign, "screen", unsafe, 4, "fresh")

    with pytest.raises(ValueError, match="multiple of four"):
        league.freeze_league(campaign, "screen", targets, 6, "fresh")


def test_arm_interval_is_computed_from_matched_deal_vectors(registered):
    campaign, targets = registered
    _fake_launcher(campaign)
    result = league.run_league(campaign, "confirm", targets, 8, "confirm_fresh")
    difference = result["paired_arm_differences"]["arm_a"]["arm_b"]
    lower, upper = difference["confidence_interval_bb_per_hand"]
    assert math.isfinite(lower) and math.isfinite(upper)
    assert lower <= difference["mean_left_minus_right_bb_per_hand"] <= upper
    assert difference["source"] == (
        "matched_logical_deal_uniform_population_difference"
    )
    assert result["uncertainty"]["conditions_on_frozen_models_and_population"] is True
    assert result["uncertainty"]["includes_training_seed_uncertainty"] is False


def test_campaign_validator_accepts_raw_match_but_br_scorer_rejects_it() -> None:
    value = {
        "raw_match": True,
        "status": "complete",
        "requested_hands": 4,
        "completed_hands": 4,
        "baseline_evaluated": False,
        "candidate_payoff_vs_fixed_opponent": {
            "mean_bb_per_hand": 0.25,
            "confidence_interval_bb_per_hand": [-0.5, 1.0],
        },
    }
    validate_evaluation(value)
    with pytest.raises(ValueError, match="does not accept raw-match"):
        score_evaluations([value, value, value])
