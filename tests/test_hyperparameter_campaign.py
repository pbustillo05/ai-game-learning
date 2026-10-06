"""Orchestration tests: no model forwards, CUDA contexts, or GPU workloads."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from experiments.hyperparameter_search import campaign as c


@pytest.fixture
def profile():
    return {"kind": c.SCHEMA, "training": {"kind": "nlhe_run_v1", "scientific": {"optimizer": {"learning_rate": 0.00005}},
                                           "run": {"execution": {"microbatch": {"rollout_rows": 128}}}},
            "control": {"entropy": {"zero_after_decisions": 1000}},
            "best_response": {"overrides": {"scientific.optimizer.learning_rate": 0.0001}},
            "arms": [{"name": "control", "overrides": {}},
                     {"name": "faster", "overrides": {"scientific.optimizer.learning_rate": 0.0001,
                                                        "control.entropy.zero_after_decisions": 500}},
                     {"name": "lower", "overrides": {"scientific.optimizer.learning_rate": 0.000025}}],
            "campaign": {"screen_seconds": 7200, "confirm_seconds": 10800, "cold_seconds": 14400,
                         "screen_br_seconds": 1800, "confirm_br_seconds": 3600, "cold_br_seconds": 3600,
                         "screen_hands": 512, "confirm_hands": 2048, "cold_hands": 4096,
                         "screen_seed": 918273, "confirmation_seeds": [20260929, 20260930], "cold_seed": 20261001,
                         "br_seeds": [4101, 4102, 4103], "shortlist": 3, "cold_finalists": 2}}


def evaluation(mean, lower=None, upper=None):
    return {"status": "complete", "completed_hands": 4, "requested_hands": 4,
            c.GAIN: mean, c.CI: [mean - 1 if lower is None else lower, mean + 1 if upper is None else upper]}


def score(mean):
    return c.score_evaluations([evaluation(mean - 2), evaluation(mean - 1), evaluation(mean)])


def test_prepare_resume_is_cuda_free_and_immutable(profile, tmp_path, monkeypatch):
    source = tmp_path / "profile.yaml"
    source.write_text(yaml.safe_dump(profile))
    output = tmp_path / "campaign"
    monkeypatch.setattr(c, "resolve_gpus", lambda _: pytest.fail("prepare must not query CUDA"))
    args = ["--profile", str(source), "--output", str(output), "--gpus", "0,1", "--prepare-only"]
    assert c.main(args) == 0
    assert len(list((output / "screen").glob("*/seed_*/*/job.json"))) == 6
    assert c.main(args) == 1
    assert c.main([*args, "--resume"]) == 0
    profile["arms"][0]["overrides"]["scientific.optimizer.learning_rate"] = 0.0002
    source.write_text(yaml.safe_dump(profile))
    assert c.main([*args, "--resume"]) == 1


def test_restart_specs_ignore_arm_overrides_and_preserve_budget(profile, tmp_path):
    campaign = c.Campaign(profile, tmp_path, ["0", "1", "2"], scale=0.5)
    path, spec = campaign.worker_spec("confirm", "faster", 20260929, private_estimator="rao_blackwell", restart=1, target="/target/inference.pt")
    assert spec["overrides"] == {}
    assert spec["profile"]["best_response"] == profile["best_response"]
    assert spec["seed"] == 4102
    assert spec["initialization"] == "target"
    assert spec["budget_seconds"] == 1800
    assert campaign.worker_spec("confirm", "faster", 20260929, private_estimator="rao_blackwell", restart=1, target="/target/inference.pt")[0] == path
    with pytest.raises(ValueError, match="Immutable"):
        campaign.worker_spec("confirm", "faster", 20260929, private_estimator="rao_blackwell", restart=1, target="/different/inference.pt")


def test_budget_accounts_for_response_waves(profile):
    one = c.budget_plan(profile, 1, 1, False)
    three = c.budget_plan(profile, 3, 1, False)
    assert one["stages"]["screen"]["br_waves_per_trial"] == 3
    assert three["stages"]["screen"]["nominal_training_wall_seconds"] == 6 * (7200 + 1800)
    assert one["nominal_training_wall_seconds"] > three["nominal_training_wall_seconds"]


def test_promotion_preserves_control_and_marks_uncertainty():
    scores = {"control": score(4), "a": score(0), "b": score(0.5), "c": score(1)}
    result = c.promote(scores, 3, include_control=True)
    assert result["selected"] == ["control", "a", "b"]
    assert result["provisional"] is True
    assert "b" in result["uncertainty_overlap_with_leader"]
    assert c.promote(scores, 2)["selected"] == ["a", "b"]


def test_confirmation_requires_paired_seeds_and_aggregates():
    trials = {"control": {1: score(2), 2: score(4)}, "better": {1: score(1), 2: score(3)}}
    aggregate = c.aggregate_confirmation(trials, [1, 2])
    assert aggregate["better"]["worst_response_gain_bb_per_hand"] == 2
    assert aggregate["better"]["paired_gain_difference_vs_control"] == {"1": -1, "2": -1}
    del trials["better"][2]
    with pytest.raises(ValueError, match="paired"):
        c.aggregate_confirmation(trials, [1, 2])


def test_incomplete_or_nonfinite_evaluation_cannot_promote():
    bad = evaluation(1)
    bad["status"] = "incomplete"
    with pytest.raises(ValueError, match="complete"):
        c.score_evaluations([evaluation(0), bad, evaluation(2)])
    bad = evaluation(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        c.validate_evaluation(bad)


def test_resolve_uuid_and_refuse_only_selected_busy_gpu(monkeypatch):
    def smi(query, domain):
        return [["0", "GPU-a"], ["1", "GPU-b"]] if domain == "gpu" else [["GPU-b", "9123", "other-user"]]
    monkeypatch.setattr(c, "_smi", smi)
    assert c.resolve_gpus(["0", "GPU-b"]) == ["GPU-a", "GPU-b"]
    c.check_idle(["GPU-a"])
    with pytest.raises(RuntimeError, match="9123"):
        c.check_idle(["GPU-b"])
    with pytest.raises(ValueError, match="duplicate"):
        c.resolve_gpus(["0", "GPU-a"])


def test_environment_targets_gpu_and_clears_outer_rendezvous(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("MASTER_PORT", "9999")
    env = c.child_environment(["GPU-b"])
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert env["CUDA_HOME"] == str(c.CUDA_HOME)
    assert str(c.CUDA_HOME / "bin") in env["PATH"]
    assert "RANK" not in env and "MASTER_PORT" not in env


def test_resume_trusts_explicit_complete_result_and_checks_hash(profile, tmp_path):
    campaign = c.Campaign(profile, tmp_path, ["0"])
    path, spec = campaign.worker_spec("screen", "control", 918273, private_estimator="rao_blackwell")
    job = campaign._worker_job(path, spec, ["GPU-a"])
    assert campaign._completed(job) is None
    c.atomic_json(job.result_path, {"complete": False, "training_seconds": 3})
    assert campaign._completed(job) is None
    snapshot, checkpoint = path.parent / "inference.pt", path.parent / "rank0.pt"
    snapshot.write_bytes(b"snapshot")
    checkpoint.write_bytes(b"checkpoint")
    result = {"complete": True, "training_seconds": 7200, "snapshot": str(snapshot), "checkpoint": str(checkpoint), "metrics": {}}
    c.atomic_json(job.result_path, result)
    assert campaign._completed(job) == result
    result["training_seconds"] = 7300
    c.atomic_json(job.result_path, result)
    with pytest.raises(ValueError, match="changed"):
        campaign._completed(job)


def test_resolved_recommendation_applies_control_and_training_separately(profile):
    original = copy.deepcopy(profile)
    resolved = c.resolved_arm(profile, profile["arms"][1])
    assert resolved["training"]["scientific"]["optimizer"]["learning_rate"] == 0.0001
    assert resolved["control"]["entropy"]["zero_after_decisions"] == 500
    assert profile == original


def test_smoke_specs_use_one_iteration_and_single_arm(profile, tmp_path):
    campaign = c.Campaign(profile, tmp_path, ["0", "1"], smoke=True)
    campaign.prepare(resume=False)
    specs = list((tmp_path / "screen").glob("*/seed_*/*/job.json"))
    assert len(specs) == 2
    assert json.loads(specs[0].read_text())["max_iterations"] == 1
    assert c.budget_plan(profile, 2, 1, True)["stages"]["cold"]["trials"] == 0


def test_torchrun_uses_all_selected_uuids(profile, tmp_path):
    campaign = c.Campaign(profile, tmp_path, ["0", "1", "2"])
    path, spec = campaign.worker_spec("screen", "control", 918273, private_estimator="rao_blackwell")
    job = campaign._worker_job(path, spec, ["GPU-a", "GPU-b", "GPU-c"])
    assert "--nproc-per-node=3" in job.command
    assert job.command[-2:] == ["--spec", str(path)]


def test_output_lock_prevents_second_launcher(tmp_path):
    with c.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with c.campaign_lock(tmp_path):
                pytest.fail("second launcher acquired lock")


def test_pairing_selects_shared_parameters_not_estimator_winner():
    pooled = c.pool_estimators({"sampled": score(10), "rao_blackwell": score(2)})
    assert pooled["worst_response_gain_bb_per_hand"] == 6
    assert pooled["sampled_minus_rb_gain"] == 8
    assert pooled["estimator_weights"] == {"sampled": 0.5, "rao_blackwell": 0.5}
    with pytest.raises(ValueError, match="both estimator"):
        c.pool_estimators({"sampled": score(0)})


def test_prepare_materializes_both_modes_with_same_seed(profile, tmp_path):
    campaign = c.Campaign(profile, tmp_path, ["0", "1", "2"])
    campaign.prepare(resume=False)
    a = json.loads((tmp_path / "screen/control/seed_918273/sampled/job.json").read_text())
    b = json.loads((tmp_path / "screen/control/seed_918273/rao_blackwell/job.json").read_text())
    assert a["seed"] == b["seed"]
    assert a["budget_seconds"] == b["budget_seconds"]
    assert a["initialization"] == b["initialization"]
    assert a["private_estimator"] != b["private_estimator"]


def test_estimator_arm_rejected(profile, tmp_path):
    profile["arms"][1]["overrides"]["scientific.mode.private_estimator"] = "sampled"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(ValueError, match="paired"):
        c.load_profile(path)


def test_source_digest_covers_untracked_driver_and_production(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    (tmp_path / "pbg").mkdir()
    drivers = tmp_path / "experiments/hyperparameter_search"
    drivers.mkdir(parents=True)
    source = tmp_path / "pbg/new_untracked.py"
    source.write_text("a = 1")
    before = c.source_digest()
    source.write_text("a = 2")
    assert before != c.source_digest()
    before = c.source_digest()
    (drivers / "new_driver.py").write_text("b = 1")
    assert before != c.source_digest()


def test_completed_wave_is_skipped_on_resume(profile, tmp_path, monkeypatch):
    campaign = c.Campaign(profile, tmp_path, ["0"])
    path, spec = campaign.worker_spec("screen", "control", 918273, private_estimator="sampled")
    job = campaign._worker_job(path, spec, ["GPU-a"])
    snapshot, checkpoint = path.parent / "inference.pt", path.parent / "rank0.pt"
    snapshot.write_bytes(b"snapshot")
    checkpoint.write_bytes(b"checkpoint")
    calls = []
    class Process:
        pid = 424242
        def poll(self):
            return 0
    def popen(command, **kwargs):
        calls.append(command)
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a"
        c.atomic_json(job.result_path, {"complete": True, "training_seconds": 7200,
                                       "snapshot": str(snapshot), "checkpoint": str(checkpoint)})
        return Process()
    monkeypatch.setattr(c.subprocess, "Popen", popen)
    monkeypatch.setattr(c, "check_idle", lambda _: None)
    first = campaign.run_jobs([job])
    assert campaign.run_jobs([job]) == first
    assert len(calls) == 1
    assert json.loads((job.directory / "status.json").read_text())["state"] == "complete"
    assert len((tmp_path / "commands.jsonl").read_text().splitlines()) == 1


def test_termination_signals_only_owned_live_process_groups(profile, tmp_path, monkeypatch):
    campaign = c.Campaign(profile, tmp_path, ["0"])
    class Process:
        def __init__(self, pid, code):
            self.pid, self.code = pid, code
        def poll(self):
            return self.code
    campaign.children = {1234: Process(1234, None), 5678: Process(5678, 0)}
    signals = []
    monkeypatch.setattr(c.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    campaign._signal(c.signal.SIGTERM, None)
    assert campaign.interrupted
    assert signals == [(1234, c.signal.SIGTERM)]


def test_source_invalidated_result_is_not_resumed(profile, tmp_path):
    campaign = c.Campaign(profile, tmp_path, ["0"])
    path, spec = campaign.worker_spec("screen", "control", 918273, private_estimator="sampled")
    job = campaign._worker_job(path, spec, ["GPU-a"])
    campaign._status(job, "failed", source_invalidated=True)
    with pytest.raises(RuntimeError, match="source changed"):
        campaign._completed(job)


def test_combination_uses_strongest_nonconflicting_changes():
    arms = {'control': {'overrides': {}}, 'fast': {'overrides': {'scientific.optimizer.learning_rate': 0.0001}},
            'slower': {'overrides': {'scientific.optimizer.learning_rate': 0.00002}},
            'entropy': {'overrides': {'control.entropy.initial_coefficient': 0.0}}}
    result = c.combination_candidate(arms, ['control', 'fast', 'slower', 'entropy'])
    assert result['components'] == ['fast', 'entropy']
    assert len(result['overrides']) == 2
    assert c.combination_candidate(arms, ['fast', 'slower']) is None
