"""Cross-panel orchestration and selection tests; no CUDA or model execution."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from experiments.hyperparameter_search import cross_panel as panel
from experiments.hyperparameter_search.campaign import CI, GAIN, atomic_json


@pytest.fixture
def registered(tmp_path):
    output = tmp_path / "campaign"
    profile = {"campaign": {"cold_seed": 20261001, "cross_panel_hands": 512}}
    scores = {}
    for arm_index, arm in enumerate(("arm_a", "arm_b")):
        modes = {}
        for mode in panel.ESTIMATORS:
            directory = output / "cold" / arm / "seed_20261001" / mode
            for restart in (None, 0, 1, 2):
                job = directory if restart is None else directory / f"response_{restart}"
                job.mkdir(parents=True)
                snapshot = job / "inference.pt"
                snapshot.write_bytes(f"{arm}/{mode}/{restart}".encode())
                atomic_json(job / "result.json", {"complete": True, "snapshot": str(snapshot.resolve())})
            modes[mode] = {"restart_mean_gains": [1.0, 3.0, 2.0]}
        scores[arm] = {"worst_response_gain_bb_per_hand": 1.0 + arm_index, "per_mode": modes}
    campaign = SimpleNamespace(output=output, profile=profile, smoke=False, gpus=["GPU-a", "GPU-b", "GPU-c"])
    return campaign, scores


def fake_launcher(campaign, *, incomplete=False):
    launched, waves = [], []
    def run_jobs(jobs):
        waves.append(len(jobs))
        results = []
        for job in jobs:
            if job.result_path.exists():
                result = json.loads(job.result_path.read_text())
                if result.get("status") == "complete":
                    results.append(result)
                    continue
            spec = json.loads((job.directory / "job.json").read_text())
            arguments = spec["arguments"]
            # Attacks trained against B expose A's previously missed weakness.
            gain = 10.0 if spec["target_id"].startswith("arm_a") and spec["attacker_id"].startswith("arm_b") else 3.0
            result = {
                "status": "incomplete" if incomplete else "complete",
                "requested_hands": arguments["hands"],
                "completed_hands": 2 if incomplete else arguments["hands"],
                "seed": arguments["seed"], "split": arguments["split"],
                GAIN: gain, CI: [gain - 0.25, gain + 0.25],
            }
            for role in ("candidate", "opponent", "baseline"):
                result[f"{role}_path"] = arguments[role]
                result[f"{role}_sha256"] = spec[f"{role}_sha256"]
            atomic_json(job.result_path, result)
            launched.append(job)
            results.append(result)
        return results
    campaign.run_jobs = run_jobs
    return launched, waves


def test_frozen_restart_selection_uses_old_scores_and_stable_ties(registered):
    campaign, scores = registered
    scores["arm_a"]["per_mode"]["sampled"]["restart_mean_gains"] = [3.0, 3.0, 2.0]
    manifest = panel.freeze_pool(campaign, scores)
    assert len(manifest["targets"]) == len(manifest["attackers"]) == 4
    assert manifest["pair_count"] == 16
    assert manifest["attackers"]["arm_a__sampled"]["selected_restart"] == 0
    assert manifest["attackers"]["arm_b__rao_blackwell"]["selected_restart"] == 1
    assert manifest["test_split"] != manifest["selection_split"]


def test_full_cross_panel_changes_shared_recommendation_and_resumes(registered):
    campaign, scores = registered
    launched, waves = fake_launcher(campaign)
    result = panel.run_cross_panel(campaign, scores)
    assert len(launched) == 16
    assert waves == [3, 3, 3, 3, 3, 1]
    assert all(len(job.gpus) == 1 for job in launched)
    assert result["recommended_arm"] == "arm_b"
    assert result["own_br_recommended_arm"] == "arm_a"
    assert result["disagrees_with_own_br_ranking"] is True
    assert result["estimator_selected"] is False
    assert result["requested_hands_total"] == 8192
    assert result["additional_training_seconds"] == 0
    assert result["uncertainty"]["intervals_are_formal_pooled_ci"] is False
    assert len(result["shared_arm_scores"]["arm_b"]["per_mode"]) == 2
    assert panel.run_cross_panel(campaign, scores) == result
    assert len(launched) == 16
    assert (campaign.output / "cross_panel/matrix.json").is_file()
    for job in launched:
        assert job.command[2] == "experiments.hyperparameter_search.evaluation"
        spec = json.loads((job.directory / "job.json").read_text())
        assert spec["arguments"]["split"] == "cross_panel_test_v1"
        assert spec["arguments"]["seed"] == 20261001
        assert spec["arguments"]["baseline"] == spec["arguments"]["opponent"]


def test_changed_selected_snapshot_rejects_resume(registered):
    campaign, scores = registered
    manifest = panel.freeze_pool(campaign, scores)
    changed = Path(manifest["attackers"]["arm_a__sampled"]["snapshot"])
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Immutable"):
        panel.freeze_pool(campaign, scores)


def test_incomplete_worker_or_missing_mode_cannot_enter_pool(registered):
    campaign, scores = registered
    result_path = campaign.output / "cold/arm_a/seed_20261001/sampled/result.json"
    result = json.loads(result_path.read_text())
    result["complete"] = False
    atomic_json(result_path, result)
    with pytest.raises(ValueError, match="not complete"):
        panel.freeze_pool(campaign, scores)
    result["complete"] = True
    atomic_json(result_path, result)
    del scores["arm_a"]["per_mode"]["sampled"]
    with pytest.raises(ValueError, match="both estimator"):
        panel.freeze_pool(campaign, scores)


def test_incomplete_evaluation_never_emits_final_panel(registered):
    campaign, scores = registered
    fake_launcher(campaign, incomplete=True)
    with pytest.raises(ValueError, match="complete"):
        panel.run_cross_panel(campaign, scores)
    assert not (campaign.output / "cross_panel/result.json").exists()


def test_changed_evaluation_identity_is_rejected(registered):
    campaign, scores = registered
    fake_launcher(campaign)
    original = campaign.run_jobs
    def corrupted(jobs):
        values = original(jobs)
        values[0]["candidate_sha256"] = "different"
        return values
    campaign.run_jobs = corrupted
    with pytest.raises(ValueError, match="hash"):
        panel.run_cross_panel(campaign, scores)


def test_smoke_uses_four_hands_and_gpu_waves(registered):
    campaign, scores = registered
    campaign.smoke = True
    campaign.gpus = ["GPU-only"]
    launched, waves = fake_launcher(campaign)
    result = panel.run_cross_panel(campaign, scores)
    assert waves == [1] * 16
    assert result["requested_hands_total"] == 64
    assert all(job.gpus == ["GPU-only"] for job in launched)


def test_unsafe_names_and_nonfinite_selection_rejected(registered):
    campaign, scores = registered
    scores["../escape"] = scores.pop("arm_a")
    with pytest.raises(ValueError, match="Unsafe"):
        panel.freeze_pool(campaign, scores)
    scores["arm_a"] = scores.pop("../escape")
    scores["arm_a"]["per_mode"]["sampled"]["restart_mean_gains"] = [1.0, float("nan"), 2.0]
    with pytest.raises(ValueError, match="finite"):
        panel.freeze_pool(campaign, scores)
