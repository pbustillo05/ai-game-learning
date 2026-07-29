from __future__ import annotations

import copy
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from pbg.benchmark import benchmark_training
from pbg.config import ConfigError, load_config, load_config_mapping
from pbg.devices import resolve_execution_device, validate_workload_capabilities
from pbg.eval.br_probe import _play_one, evaluate_br_probe
from pbg.games.poker.queued.evaluator import bound_evaluator
from pbg.inference_artifact import (
    INFERENCE_ARTIFACT_SCHEMA_VERSION,
    export_inference_artifact,
    load_inference_source,
)
from pbg.core.persistence import RunState, model_state_sha256, sha256_file
from pbg.core.persistence import (
    CHECKPOINT_SCHEMA,
    load_distributed_checkpoint_manifest,
    save_distributed_checkpoint_manifest,
)
from tests.static_gates import (
    accepted_root_violations,
    local_import_graph,
    reachable_modules,
    stale_cutover_violations,
)
from pbg.runtime import TrainingSession
from pbg.variance import estimate_update_variance

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/train/nlhe_hu_8bb_normal_smoke.yaml"
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
        "pre": (240,),
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


def test_run_state_requires_the_exact_nested_logical_work_schema() -> None:
    with pytest.raises(ValueError, match="exact nonnegative integer counters"):
        RunState("malformed-logical-work", logical_work={})


def test_public_config_rejects_legacy_aliases_and_profile_overlays() -> None:
    with pytest.raises(TypeError, match="hardware_profile"):
        load_config(CONFIG, hardware_profile="legacy")  # type: ignore[call-arg]
    import yaml

    raw = yaml.safe_load(CONFIG.read_text())
    raw["scientific"]["training"] = {"mode": "normal"}
    with pytest.raises(ConfigError, match="unknown configuration field"):
        load_config_mapping(raw)


def test_every_accepted_root_reaches_the_final_core() -> None:
    assert accepted_root_violations(ROOT, ROOTS) == ()
    assert stale_cutover_violations(ROOT) == ()
    graph = local_import_graph(ROOT)
    for root in ROOTS:
        reached = reachable_modules(graph, root)
        assert any(
            module == "pbg.core"
            or module.startswith("pbg.core.")
            for module in reached
        ), root


def test_operational_status_finds_current_checkpoint_names() -> None:
    launcher = (ROOT / "tools/run-training.sh").read_text(encoding="utf-8")
    assert "ckpt_*.pt" not in launcher
    assert "checkpoint_????????.pt" in launcher
    assert "failure_checkpoint_????????.pt" in launcher


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
    scientific = config.scientific_mapping()
    torch.save(
        {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "scientific_config": scientific,
            "scientific_sha256": hashlib.sha256(
                json.dumps(
                    scientific,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
            "execution_config": config.execution.mapping(),
            "model": {"weight": torch.ones(2)},
            "optimizer": {},
            "scaler": {},
            "reference_model": {},
            "reference_state": {},
            "run_state": {
                "run_id": "strict-export",
                "iteration": 3,
                "root_count": 1,
                "retry_epoch": 0,
                "attempt_logical_base": 0,
                "logical_work": {
                    "trajectories": 0,
                    "completion_samples": 0,
                    "minibatches": 0,
                    "optimizer_steps": 0,
                },
                "recovered_overflow": False,
                "overflow_recoveries": [],
                "metrics_prefix": [
                    {"iteration": value} for value in range(3)
                ],
                "branch_controller_state": None,
                "cost_safeguard_state": None,
            },
            "explicit_rng": torch.get_rng_state(),
            "ambient_cpu_rng": torch.get_rng_state(),
            "ambient_cuda_rng": (),
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
    raw["scientific"]["reference"]["interval_iterations"] = None
    raw["scientific"]["reference"]["interval_effective_hands"] = 0.5
    config = load_config_mapping(raw)
    session = TrainingSession(config)
    result = session.step()
    assert len(result.results) == 2
    assert tuple(state.iteration for state in session.run_states) == (1, 1)
    for lane, state in enumerate(session.run_states):
        metrics = state.metrics_prefix[-1]
        assert metrics["kl_ref_refresh_before_iteration"] is False
        assert metrics["kl_ref_refresh_after_iteration"] is True
        assert metrics["kl_ref_effective_hands_rollout_before"] == 0.0
        assert metrics["kl_ref_effective_hands_rollout_threshold"] == 0.5
        assert metrics["kl_ref_effective_hands_rollout_overshoot"] > 0.0
        assert metrics["kl_ref_effective_hands_rollout_after"] == 0.0
        assert metrics["reference_iteration_next"] == 1
        assert metrics["per_rank_lane_logical_work"][0][lane] == (
            state.logical_work
        )
        assert metrics["global_lane_logical_work"][lane] == (
            state.logical_work
        )
        assert metrics["global_logical_work"]["trajectories"] == 2
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
    target_session = TrainingSession(target_config)
    target_session.checkpoint(target_checkpoint)
    target_artifact = tmp_path / "target_inference.pt"
    export_inference_artifact(target_checkpoint, target_artifact)

    target_hash = sha256_file(target_checkpoint)
    initial_hash = model_state_sha256(target_session.modules.lane_module(0))
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
        "initial_model_sha256": initial_hash,
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


def test_paired_br_chance_address_is_independent_of_action_path() -> None:
    class Action:
        event_id = 0

    class Geometry:
        players = 2

    class Session:
        geometry = Geometry()

        def __init__(self) -> None:
            self.deal_addresses: list[int] = []

        def new_hand(self, _button: int) -> dict[str, int]:
            return {"phase": 0}

        def private_cards(
            self, _hand: object, seat: int
        ) -> tuple[int, int]:
            return (seat, seat + 2)

        def is_terminal(self, hand: dict[str, int]) -> bool:
            return hand["phase"] == 2

        def needs_chance(self, hand: dict[str, int]) -> bool:
            return hand["phase"] == 1

        def actor(self, _hand: object) -> int:
            return 0

        def policy_table(
            self, _hand: object
        ) -> tuple[tuple[Action], torch.Tensor]:
            return (Action(),), torch.ones((4, 1))

        def private_hand_id(self, _hand: object, _actor: int) -> int:
            return 0

        def private_policy_index(self, _hand: object, _actor: int) -> int:
            return 0

        def advance(
            self,
            hand: dict[str, int],
            _action: object,
            *,
            policy_probabilities: torch.Tensor,
        ) -> None:
            assert policy_probabilities.shape == (4, 1)
            hand["phase"] = 1

        def deal_keyed(
            self,
            hand: dict[str, int],
            *,
            seed: int,
            trajectory_id: int,
            decision_index: int,
        ) -> None:
            assert seed == 17
            assert trajectory_id == 5
            self.deal_addresses.append(decision_index)
            hand["phase"] = 2

        def terminal_payoffs(self, _hand: object) -> torch.Tensor:
            return torch.zeros(2)

    probe = Session()
    target = Session()
    assert _play_one(
        probe,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        trajectory_id=5,
        probe_seat=0,
        seed=17,
        mixed=True,
    ) == 0.0
    assert target.deal_addresses == [0]
    assert probe.deal_addresses == [0]


def test_evaluator_binding_has_one_process_owner_per_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def bind(*, rank_count: int) -> object:
        calls.append(rank_count)
        return SimpleNamespace(
            info=SimpleNamespace(
                contract_version="2",
                rank_count=rank_count,
            )
        )

    fake = SimpleNamespace(CONTRACT_VERSION="2", bind=bind)
    bound_evaluator.cache_clear()
    monkeypatch.setitem(sys.modules, "poker_gpu_evaluator", fake)
    try:
        first = bound_evaluator(4)
        assert bound_evaluator(4) is first
        assert bound_evaluator(5) is not first
        assert calls == [4, 5]
    finally:
        bound_evaluator.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_benchmark_reuses_process_workspaces_across_repeats(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["execution"]["output_dir"] = str(tmp_path / "benchmark")
    cfg = load_config_mapping(raw)
    report = benchmark_training(cfg, warmup=1, repeats=3)
    peaks = [sample["peak_allocated_bytes"] for sample in report["samples"]]
    del report
    gc.collect()
    torch.cuda.synchronize()
    assert peaks == [peaks[0]] * 3
