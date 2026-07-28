from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from pbg.config import ConfigError, load_config, load_config_mapping
from pbg.devices import resolve_execution_device, validate_workload_capabilities
from pbg.inference_artifact import (
    INFERENCE_ARTIFACT_SCHEMA_VERSION,
    convert_legacy_inference_artifact,
    export_inference_artifact,
    load_inference_source,
)
from pbg.eval.br_probe import evaluate_br_probe
from pbg.queued.persistence import (
    CHECKPOINT_SCHEMA,
    load_distributed_checkpoint_manifest,
    save_distributed_checkpoint_manifest,
)
from pbg.queued import sha256_file
from pbg.queued.static_gates import local_import_graph, reachable_modules
from pbg.queued.static_gates import migrated_similarity_violations
from pbg.runtime import TrainingSession
from pbg.variance import estimate_update_variance


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/train/nlhe_hu_8bb_normal_queued_smoke.yaml"
ROOTS = (
    "pbg.cli.train",
    "pbg.cli.train_replicas",
    "pbg.cli.benchmark",
    "pbg.cli.benchmark_replica_learner",
    "pbg.cli.br_eval",
    "pbg.cli.export_inference",
    "pbg.cli.benchmark_inference",
    "pbg.cli.runpod_replica_selection",
    "pbg.cli.variance",
)


def test_public_config_is_one_closed_three_axis_composition() -> None:
    config = load_config(CONFIG)
    assert config.mode == {
        "training_regime": "self_play",
        "rollout_mode": "sequential",
        "private_estimator": "sampled",
    }
    assert config.execution.manifest.root_count == 1
    assert config.execution.layout.input_shapes == {
        "pre": (448,),
        "post_2": (448,),
    }
    raw = {
        "scientific": config.scientific_mapping(),
        "execution": {
            **config.execution.mapping(),
            "frozen_checkpoint": None,
        },
    }
    assert load_config_mapping(raw) == config


def test_public_config_rejects_legacy_aliases_and_profile_overlays() -> None:
    with pytest.raises(ConfigError, match="profile overlays"):
        load_config(CONFIG, hardware_profile="legacy")
    import yaml

    raw = yaml.safe_load(CONFIG.read_text())
    raw["scientific"]["training"] = {"mode": "normal"}
    with pytest.raises(ConfigError, match="unknown configuration field"):
        load_config_mapping(raw)


def test_every_accepted_root_reaches_only_the_queued_replacement() -> None:
    graph = local_import_graph(ROOT)
    for root in ROOTS:
        reached = reachable_modules(graph, root)
        assert not any(
            module == "pbg.core" or module.startswith("pbg.core.")
            for module in reached
        ), root
        assert any(
            module == "pbg.queued"
            or module.startswith("pbg.queued.")
            or module.startswith("pbg.games.poker.queued")
            for module in reached
        ), root


def test_migrated_consumers_share_no_legacy_implementation_body() -> None:
    assert migrated_similarity_violations(ROOT, ROOTS) == ()


def test_accelerator_contract_has_no_cpu_auto_or_mps_training_fallback(
    monkeypatch,
) -> None:
    for value in ("cpu", "auto", torch.device("cpu")):
        with pytest.raises(ValueError, match="cuda.*mps"):
            resolve_execution_device(value, workload="inference")
    with pytest.raises(RuntimeError, match="contract v2"):
        validate_workload_capabilities(torch.device("mps"), workload="training")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never falls back"):
        resolve_execution_device("cuda", workload="training")


def test_inference_export_accepts_only_queued_schema(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "scientific_config": config.scientific_mapping(),
            "model": {"weight": torch.ones(2)},
            "run_state": {"iteration": 3},
        },
        checkpoint,
    )
    artifact = tmp_path / "inference.pt"
    report = export_inference_artifact(checkpoint, artifact)
    assert report["schema_version"] == INFERENCE_ARTIFACT_SCHEMA_VERSION
    source = load_inference_source(artifact)
    assert source["iteration"] == 3
    assert source["scientific_config"] == config.scientific_mapping()
    legacy = tmp_path / "legacy.pt"
    torch.save({"checkpoint_schema_version": "pbg-checkpoint-v8"}, legacy)
    with pytest.raises(RuntimeError, match="only queued"):
        export_inference_artifact(legacy, tmp_path / "rejected.pt")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["scientific_config"]["legacy_alias"] = True
    permissive = tmp_path / "permissive.pt"
    torch.save(payload, permissive)
    with pytest.raises(RuntimeError, match="scientific configuration"):
        load_inference_source(permissive)


def test_legacy_inference_conversion_is_offline_and_hash_audited(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    scientific = config.scientific_mapping()
    game = dict(scientific["game"])
    game["n_players"] = game.pop("players")
    model = dict(scientific["model"])
    model["eps"] = model.pop("epsilon")
    model["k_act"] = 3 + int(model["m_raise"])
    model["use_branching_heads"] = False
    model["max_history_tokens"] = 0  # derived legacy value is ignored
    source = tmp_path / "legacy_inference.pt"
    torch.save(
        {
            "inference_artifact_schema_version": "pbg-inference-v1",
            "source_checkpoint_schema_version": "pbg-checkpoint-v8",
            "source_checkpoint_sha256": "a" * 64,
            "iteration": 7,
            "config": {"game": game, "model": model},
            "model": {"weight": torch.ones(2)},
        },
        source,
    )
    destination = tmp_path / "queued_inference.pt"
    report = convert_legacy_inference_artifact(
        source,
        destination,
        scientific_config=scientific,
    )
    assert report["source_schema"] == "pbg-inference-v1"
    assert report["result_schema"] == INFERENCE_ARTIFACT_SCHEMA_VERSION
    assert len(report["source_sha256"]) == len(report["result_sha256"]) == 64
    assert destination.with_suffix(".pt.conversion.json").is_file()
    assert load_inference_source(destination)["scientific_config"] == scientific


def test_distributed_manifest_is_strict_and_content_addressed(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    rank_files = []
    rank_hashes = []
    for rank in range(2):
        path = tmp_path / f"rank_{rank:04d}.pt"
        path.write_bytes(f"rank-{rank}".encode())
        rank_files.append(path.name)
        rank_hashes.append(sha256_file(path))
    manifest = tmp_path / "distributed.pt"
    save_distributed_checkpoint_manifest(
        manifest,
        rank_files=rank_files,
        rank_sha256=rank_hashes,
        scientific_config=config.scientific_mapping(),
        execution_config=config.execution.mapping(),
    )
    assert load_distributed_checkpoint_manifest(
        manifest,
        expected_scientific_config=config.scientific_mapping(),
        world_size=2,
    ) == tuple(tmp_path / name for name in rank_files)
    (tmp_path / rank_files[1]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="rank hash"):
        load_distributed_checkpoint_manifest(
            manifest,
            expected_scientific_config=config.scientific_mapping(),
            world_size=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_replica_consumer_and_reference_cadence_resume(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["execution"]["manifest"]["n_replica"] = 2
    raw["execution"]["output_dir"] = str(tmp_path / "replica")
    raw["execution"]["run_id"] = "nlhe_hu_8bb_normal_replica_test"
    config = load_config_mapping(raw)
    session = TrainingSession(config)
    result = session.step()
    assert len(result.results) == 2
    assert tuple(state.iteration for state in session.run_states) == (1, 1)
    session.reference_states[0].reference_iteration = 7
    session.reference_states[0].effective_hands_rollout = 13.5
    session.reference_states[1].reference_iteration = 9
    session.reference_states[1].effective_hands_rollout = 21.5
    checkpoint = tmp_path / "replica.pt"
    session.checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match="explicit valid lane"):
        export_inference_artifact(checkpoint, tmp_path / "ambiguous.pt")
    artifact = tmp_path / "lane_1.pt"
    export_inference_artifact(checkpoint, artifact, lane=1)
    assert load_inference_source(artifact)["scientific_config"] == (
        config.scientific_mapping()
    )
    resumed = TrainingSession(config, resume_from=checkpoint)
    assert tuple(
        (
            state.reference_iteration,
            state.effective_hands_rollout,
        )
        for state in resumed.reference_states
    ) == ((7, 13.5), (9, 21.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_target_initialization_and_metrics_suffix_resume(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["execution"]["iterations"] = 2
    raw["execution"]["output_dir"] = str(tmp_path / "ordinary")
    raw["execution"]["run_id"] = "nlhe_hu_8bb_normal_resume_test"
    ordinary = load_config_mapping(raw)
    session = TrainingSession(ordinary)
    target = tmp_path / "target.pt"
    session.checkpoint(target)

    response_raw = copy.deepcopy(raw)
    response_raw["scientific"]["mode"]["training_regime"] = "best_response"
    response_raw["scientific"]["frozen_profile"] = {
        "sha256": sha256_file(target)
    }
    response_raw["scientific"]["response_initialization"] = "target"
    response_raw["execution"]["frozen_checkpoint"] = str(target)
    response_raw["execution"]["iterations"] = 1
    response = TrainingSession(load_config_mapping(response_raw))
    target_payload = torch.load(target, map_location="cpu", weights_only=False)
    for name, value in response.modules.lane_module(0).state_dict().items():
        torch.testing.assert_close(value.cpu(), target_payload["model"][name])

    session.run()
    checkpoint_one = (
        ordinary.execution.output_dir / "checkpoint_00000001.pt"
    )
    metrics = ordinary.execution.output_dir / "metrics.jsonl"
    before = [
        json.loads(line)["iteration"]
        for line in metrics.read_text(encoding="utf-8").splitlines()
    ]
    assert before == [0, 1]
    TrainingSession(ordinary, resume_from=checkpoint_one).run()
    after = [
        json.loads(line)["iteration"]
        for line in metrics.read_text(encoding="utf-8").splitlines()
    ]
    assert after == [0, 1]
    assert (
        ordinary.execution.output_dir
        / "metrics.jsonl.before_resume_0000001.bak"
    ).is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_end_to_end_br_and_variance_consumers(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["execution"]["output_dir"] = str(tmp_path / "target")
    raw["execution"]["run_id"] = "nlhe_hu_8bb_normal_br_target"
    target_config = load_config_mapping(raw)
    target_checkpoint = tmp_path / "target.pt"
    TrainingSession(target_config).checkpoint(target_checkpoint)
    target_artifact = tmp_path / "target_inference.pt"
    export_inference_artifact(target_checkpoint, target_artifact)

    target_hash = sha256_file(target_checkpoint)
    probe_raw = copy.deepcopy(raw)
    probe_raw["scientific"]["mode"] = {
        "training_regime": "best_response",
        "rollout_mode": "sequential",
        "private_estimator": "rao_blackwell",
    }
    probe_raw["scientific"]["completion"] = {
        "n_samples": 4,
        "strata": "default",
    }
    probe_raw["scientific"]["frozen_profile"] = {"sha256": target_hash}
    probe_raw["scientific"]["response_initialization"] = "target"
    probe_raw["scientific"]["probe"] = {
        "target_sha256": target_hash,
        "initial_model_sha256": target_hash,
    }
    probe_raw["execution"]["frozen_checkpoint"] = str(target_checkpoint)
    probe_raw["execution"]["output_dir"] = str(tmp_path / "probe")
    probe_raw["execution"]["run_id"] = "nlhe_hu_8bb_rb_br_probe"
    probe_checkpoint = tmp_path / "probe.pt"
    TrainingSession(load_config_mapping(probe_raw)).checkpoint(probe_checkpoint)
    probe_artifact = tmp_path / "probe_inference.pt"
    export_inference_artifact(probe_checkpoint, probe_artifact)
    report = evaluate_br_probe(
        probe_artifact,
        target_artifact=target_artifact,
        n_hands=4,
        seed=19,
        device="cuda:0",
    )
    assert report.trajectory_count == 4
    assert report.target_sha256 == target_hash
    assert report.approximate_nashconv_bb_per_hand == pytest.approx(0.0)

    variance = estimate_update_variance(target_config, repeats=2)
    assert variance["schema"] == "pbg-queued-update-variance-v1"
    assert variance["lanes"][0]["total_update_variance"] >= 0
