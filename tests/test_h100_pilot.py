from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pbg.config import load_config_mapping
from pbg.core.persistence import CHECKPOINT_SCHEMA, model_state_sha256
from pbg.h100_pilot import (
    H100_MEMORY_BUDGET_BYTES,
    Arm,
    PilotError,
    base_plan,
    export_model_initializer,
    load_model_initializer,
    pilot_config_mapping,
    select_replica_count,
)
from pbg.runtime import TrainingSession


def test_h100_plan_keeps_science_fixed_and_uses_bounded_execution_packs(monkeypatch) -> None:
    from pbg.games.poker.queued import capacity_compiler
    monkeypatch.setattr(capacity_compiler, "_device_total_memory", lambda device: 80_000_000_000)
    plan = base_plan()
    arms = [Arm(**value) for value in plan["arms"]]
    assert plan["fixed_batch_size"] == 256
    assert plan["fixed_gradient_steps"] == 116
    assert plan["production_precision"] == "bf16"
    for mode in ("normal", "rb"):
        scaling = [
            arm for arm in arms if arm.mode == mode and arm.stage == "replica_scaling"
        ]
        assert tuple(arm.replicas for arm in scaling) == (1, 16, 32)
        for arm in scaling:
            mapping = pilot_config_mapping(arm, output_dir="/tmp/pilot")
            scientific = mapping["scientific"]
            execution = mapping["run"]["execution"]
            assert scientific["schedule"] == {"batch_size": 256, "steps": 116}
            assert execution["precision"] == "bf16"
            assert execution["reference_cache"] == "off"
            assert execution["microbatch"] == {
                "rollout_rows": 8,
                "completion_rows": 64,
                "learner_rows": {"pre": 32, "post_2": 8},
            }
            assert "manifest" not in mapping["run"]
            resolved = load_config_mapping(mapping)
            assert resolved.execution.layout.model_dtype == torch.bfloat16
            assert resolved.execution.manifest.n_replica == arm.replicas


def test_local_smoke_changes_only_execution_scale_and_precision(monkeypatch) -> None:
    from pbg.games.poker.queued import capacity_compiler
    monkeypatch.setattr(capacity_compiler, "_device_total_memory", lambda device: 80_000_000_000)
    arm = Arm(
        "nlhe_hu_32bb_rb_local_test",
        "local_smoke",
        "rb",
        2,
        1,
        4,
        rollout_pack=1,
        completion_pack=1,
        learner_pack=1,
        warmup=0,
        repeats=1,
    )
    mapping = pilot_config_mapping(
        arm,
        output_dir="/tmp/pilot",
        hardware="local",
    )
    scientific = mapping["scientific"]
    assert scientific["game"] == {
        "players": 2,
        "stack_bb": 32.0,
        "rank_count": 13,
    }
    assert scientific["model"]["d_model"] == 128
    assert scientific["schedule"] == {"batch_size": 8, "steps": 1}
    assert mapping["run"]["execution"]["precision"] == "fp32"
    assert scientific["completion"]["samples"] == 4
    resolved = load_config_mapping(mapping)
    assert set(resolved.execution.manifest.partition_rows("action_cap")) == {
        "pre",
        "post_2",
    }


def test_model_initializer_is_hash_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    artifact = tmp_path / "initializer.pt"
    state = {
        "layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layer.bias": torch.tensor([0.25, -0.5]),
    }
    torch.save({"checkpoint_schema": CHECKPOINT_SCHEMA, "model": state}, checkpoint)
    report = export_model_initializer(checkpoint, artifact)
    loaded, metadata = load_model_initializer(artifact)
    assert report["model_sha256"] == model_state_sha256(state)
    assert metadata["model_sha256"] == report["model_sha256"]
    assert loaded.keys() == state.keys()
    for name, tensor in state.items():
        torch.testing.assert_close(loaded[name], tensor)

    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["model"]["layer.bias"][0] += 1
    torch.save(payload, artifact)
    with pytest.raises(PilotError, match="model hash differs"):
        load_model_initializer(artifact)


def test_replica_selection_requires_a_material_gain() -> None:
    rows = [
        {
            "replicas": 1,
            "player_nodes_per_second": 100.0,
            "peak_reserved_bytes": 20,
        },
        {
            "replicas": 2,
            "player_nodes_per_second": 109.9,
            "peak_reserved_bytes": 30,
        },
        {
            "replicas": 4,
            "player_nodes_per_second": 121.0,
            "peak_reserved_bytes": 40,
        },
        {
            "replicas": 8,
            "player_nodes_per_second": 150.0,
            "peak_reserved_bytes": H100_MEMORY_BUDGET_BYTES + 1,
        },
    ]
    assert select_replica_count(rows) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_model_only_initializer_reaches_every_replica_and_reference() -> None:
    source_arm = Arm(
        "nlhe_hu_32bb_normal_initializer_source",
        "local_smoke",
        "normal",
        1,
        1,
        None,
        rollout_pack=1,
        completion_pack=1,
        learner_pack=1,
        warmup=0,
        repeats=1,
    )
    source_config = pilot_config_mapping(
        source_arm,
        output_dir="/tmp/nlhe_hu_32bb_normal_initializer_source",
        hardware="local",
    )
    source = TrainingSession(load_config_mapping(source_config))
    try:
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in source.modules.lane_module(0).state_dict().items()
        }
    finally:
        source.close()

    target_arm = Arm(
        "nlhe_hu_32bb_normal_initializer_target",
        "local_smoke",
        "normal",
        2,
        1,
        None,
        rollout_pack=1,
        completion_pack=1,
        learner_pack=1,
        warmup=0,
        repeats=1,
    )
    target_config = pilot_config_mapping(
        target_arm,
        output_dir="/tmp/nlhe_hu_32bb_normal_initializer_target",
        hardware="local",
    )
    target = TrainingSession(
        load_config_mapping(target_config),
        initial_model_state=state,
    )
    try:
        expected = model_state_sha256(state)
        assert target.initial_model_sha256 == (expected, expected)
        assert target.reference_modules is not None
        assert tuple(
            model_state_sha256(target.reference_modules.lane_module(lane))
            for lane in range(2)
        ) == (expected, expected)
    finally:
        target.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("replicas", (1, 2))
def test_runtime_separates_rb_rollout_and_natural_learner_domains(
    replicas: int,
) -> None:
    arm = Arm(
        f"nlhe_hu_32bb_rb_runtime_domain_k{replicas}",
        "local_smoke",
        "rb",
        replicas,
        1,
        4,
        rollout_pack=1,
        completion_pack=1,
        learner_pack=1,
        warmup=0,
        repeats=1,
    )
    config = load_config_mapping(
        pilot_config_mapping(
            arm,
            output_dir=f"/tmp/{arm.name}",
            hardware="local",
        )
    )
    session = TrainingSession(config)
    try:
        assert {operation.__name__ for operation in session.model_operations} == {
            "rb_learner_operation"
        }
        assert {operation.__name__ for operation in session.reference_operations} == {
            "rb_learner_operation"
        }
        session.step()
        assert tuple(state.iteration for state in session.run_states) == (1, 16, 32) * replicas
    finally:
        session.close()


def test_shared_work_ledger_matches_hu_shape_arithmetic():
    from pbg.games.poker.model.work import shared_model_work_ledger
    for mode, expected in (("normal", (12.219138048, 142.262479872)),
                           ("rb", (12.665044992, 145.779305472))):
        arm = Arm("ledger", "replica_scaling", mode, 1, 64, None if mode == "normal" else 256)
        scientific = pilot_config_mapping(arm, output_dir="/tmp/ledger")["scientific"]
        ledger = shared_model_work_ledger(scientific)
        assert ledger["parameter_bytes_per_replica"] > 0
        for family, flops in zip(("pre", "post_2"), expected, strict=True):
            value = ledger["families"][family]
            assert value["readout_width"] == 928
            assert value["major_flops_per_presentation"] / 1e9 == pytest.approx(flops)


def test_split_initializer_is_rejected(tmp_path):
    source = tmp_path / "split.pt"
    torch.save({"checkpoint_schema": "legacy", "model": {"weight": torch.ones(1)}}, source)
    with pytest.raises(PilotError, match="split artifacts"):
        export_model_initializer(source, tmp_path / "initializer.pt")


def test_offline_prepare_never_queries_cuda(tmp_path, monkeypatch):
    from argparse import Namespace
    from pbg.cli.h100_pilot import _prepare
    def forbidden(*args, **kwargs):
        raise AssertionError("offline preparation queried CUDA")
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    _prepare(Namespace(directory=tmp_path / "run", configs_directory=tmp_path / "configs",
                       normal_checkpoint=None, rb_checkpoint=None))
    assert (tmp_path / "run" / "plan.json").is_file()
    assert len(list((tmp_path / "configs").glob("*.yaml"))) == len(base_plan()["arms"])


def test_pilot_resume_reuses_only_hash_bound_evidence(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    from types import SimpleNamespace
    from pbg.cli import h100_pilot as cli
    from pbg.h100_pilot import write_json
    identity = {"git_revision": "test-revision", "working_tree_sha256": "clean"}
    monkeypatch.setattr(cli, "code_identity", lambda: identity)
    root, configs = tmp_path / "run", tmp_path / "configs"
    cli._prepare(Namespace(directory=root, configs_directory=configs,
                           normal_checkpoint=None, rb_checkpoint=None))
    calls = []
    def subprocess_run(command, **kwargs):
        if "preflight" in command:
            write_json({"memory_budget_bytes": 70 << 30}, command[-1])
        else:
            kwargs["stdout"].write("all correctness gates passed\n")
        return SimpleNamespace(returncode=0)
    def invoke(arm, **kwargs):
        calls.append(arm.name)
        summary = {"arm": arm.name, "replicas": arm.replicas,
                   "player_nodes_per_second": 100 * arm.replicas,
                   "peak_reserved_bytes": 1 << 30, "seconds": 1.0}
        report = {"summary": summary}
        write_json(report, root / "reports" / f"{arm.name}.json")
        return report
    monkeypatch.setattr(cli.subprocess, "run", subprocess_run)
    monkeypatch.setattr(cli, "_invoke_arm", invoke)
    args = Namespace(plan=root / "plan.json", deadline_minutes=240, resume=False)
    cli._run(args)
    assert (root / "PILOT_COMPLETE").exists()
    assert len(calls) == 14
    args.resume = True
    cli._run(args)
    assert len(calls) == 14
    state = json.loads((root / "run_state.json").read_text())
    name = next(iter(state["artifacts"]))
    (root / "reports" / f"{name}.json").write_text("{}")
    with pytest.raises(PilotError, match="completed evidence changed"):
        cli._run(args)
