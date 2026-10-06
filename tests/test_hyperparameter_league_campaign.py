"""Protocol checks for the paired league search; no CUDA contexts."""

import copy
from pathlib import Path

import pytest
import yaml

from experiments.hyperparameter_search import league_campaign as c

PROFILE = Path("configs/profiles/nlhe_hu_32bb_shared_league_v2.yaml")


def profile():
    return yaml.safe_load(PROFILE.read_text())


def test_registered_workload_grid_and_historical_bridges():
    p = c.validate_profile(PROFILE)
    triples = {
        (
            a["overrides"]["run.workload.roots"],
            a["overrides"]["scientific.schedule.batch_size"],
            a["overrides"]["scientific.schedule.steps"],
        )
        for a in p["workload_arms"]
    }
    assert {
        (n, b, s) for n in (512, 1024) for b in (128, 256) for s in (24, 48)
    } <= triples
    assert {(2048, 128, 96), (2048, 256, 48)} <= triples
    assert 128 * 96 == 256 * 48
    for arm in p["workload_arms"]:
        assert (
            arm["overrides"]["run.recovery.minimum_roots"]
            == arm["overrides"]["run.workload.roots"]
        )
        assert "scientific.mode.private_estimator" not in arm["overrides"]


def test_budget_separates_training_league_and_sensitive_br():
    result = c.league_budget(profile(), smoke=False, scale=1)
    assert result["nominal_training_wall_seconds"] == 136 * 3600
    assert result["clock_BR_training_wall_seconds"] == 4 * 3600
    assert result["gradient_probe_jobs"] == 192
    assert result["stages"]["workload"]["league_pairs"] == 190
    assert result["stages"]["regularizer"]["league_pairs"] == 190
    assert result["stages"]["confirm"]["policies"] == 12
    assert result["stages"]["cold"]["hands_per_pair"] == 8192


def test_confirmation_keeps_control_and_deduplicates_equal_scientific_settings():
    p = profile()
    workloads = {a["name"]: a for a in p["workload_arms"]}
    regs = {a["name"]: a for a in p["regularizer_arms"]}
    chosen = c.confirmation_candidates(p, workloads, list(workloads), regs, list(regs))
    assert chosen[0]["name"] == "original_control"
    assert len(chosen) == 3
    assert len({c.config_signature(p, a["overrides"]) for a in chosen}) == 3
    assert p == profile()


def test_compatible_loss_combination_preserves_selected_workload():
    p = profile()
    workloads = {a["name"]: a for a in p["workload_arms"]}
    regs = {a["name"]: a for a in p["regularizer_arms"]}
    chosen = c.confirmation_candidates(
        p,
        workloads,
        ["n2048_b256_s48", "control"],
        regs,
        ["kl_weak", "entropy_zero", "control"],
    )
    combined = next(a for a in chosen if a["name"] == "workload_plus_loss")
    assert combined["overrides"]["run.workload.roots"] == 2048
    assert (
        combined["overrides"]["scientific.objective.reference_kl.coefficient"]
        == 0.003125
    )
    assert combined["overrides"]["control.entropy.initial_coefficient"] == 0


def test_uncalibrated_exponent_is_rejected(tmp_path):
    p = profile()
    p["control"]["posterior_clock"] = {"exponent": 0.5}
    f = tmp_path / "profile.yaml"
    f.write_text(yaml.safe_dump(p))
    with pytest.raises(ValueError, match="uncalibrated"):
        c.validate_profile(f)


def test_preparation_is_immutable_and_cuda_free(tmp_path, monkeypatch):
    monkeypatch.setattr(
        c, "resolve_gpus", lambda _: pytest.fail("No CUDA queries during prepare")
    )
    p = profile()
    output = tmp_path / "run"
    runner = c.LeagueCampaign(p, output, ["0", "1", "2"])
    runner.prepare(resume=False)
    assert len(list(output.glob("workload/*/seed_*/*/job.json"))) == 20
    runner.prepare(resume=True)
    changed = copy.deepcopy(p)
    changed["training"]["scientific"]["schedule"]["batch_size"] *= 2
    with pytest.raises(ValueError, match="Immutable"):
        c.LeagueCampaign(changed, output, ["0", "1", "2"]).prepare(resume=True)


def test_budget_uses_actual_br_worker_duration():
    p = profile()
    p["campaign"]["clock_audit_br_seconds"] = 7200
    assert (
        c.league_budget(p, smoke=False, scale=1)["clock_BR_training_wall_seconds"]
        == 8 * 3600
    )


def test_fixed_anchor_panel_uses_initial_snapshot_and_preserves_identity(tmp_path):
    import json

    runner = c.LeagueCampaign(
        profile(), tmp_path / "campaign", ["0", "1", "2"], smoke=True
    )
    targets = {}
    for mode in c.ESTIMATORS:
        directory = tmp_path / mode
        (directory / "snapshots").mkdir(parents=True)
        initial = directory / "snapshots" / "iteration_00000000.pt"
        initial.write_bytes(b"initial policy")
        final = directory / "inference.pt"
        final.write_bytes(mode.encode())
        targets[mode] = {
            "arm": "control",
            "private_estimator": mode,
            "seed": 123,
            "snapshot": str(final),
            "training_result": str(directory / "result.json"),
        }

    def run_jobs(jobs):
        results = []
        for job in jobs:
            spec = json.loads((job.directory / "job.json").read_text())
            assert "--raw-match" in job.command
            results.append(
                {
                    "candidate_sha256": spec["candidate_sha256"],
                    "opponent_sha256": spec["anchor_sha256"],
                    "raw_match": True,
                    "candidate_payoff_vs_fixed_opponent": {"mean_bb_per_hand": -0.1},
                }
            )
        return results

    runner.run_jobs = run_jobs
    runner.gpus = ["gpu0", "gpu1", "gpu2"]
    report = runner.anchor_panel("workload", targets)
    assert report["used_for_automatic_selection"] is False
    assert len(report["matches"]) == 2
    Path(report["anchor"]["snapshot"]).write_bytes(b"changed initial policy")
    with pytest.raises(ValueError, match="anchor changed"):
        runner.anchor_panel("confirm", targets)
