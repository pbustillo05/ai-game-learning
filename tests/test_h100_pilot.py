from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pbg.config import load_config_mapping
from pbg.core.persistence import model_state_sha256
from pbg.h100_pilot import (
    H100_MEMORY_BUDGET_BYTES,
    REPLICA_COUNTS,
    Arm,
    PilotError,
    base_plan,
    export_model_initializer,
    geometry_roots,
    load_model_initializer,
    pilot_config_mapping,
    select_replica_count,
)
from pbg.runtime import TrainingSession


def test_h100_plan_keeps_batch_fixed_and_sweeps_replicas() -> None:
    plan = base_plan()
    arms = [Arm(**value) for value in plan["arms"]]
    assert plan["fixed_batch_size"] == 256
    assert plan["fixed_gradient_steps"] == 116
    assert plan["production_precision"] == "bf16"
    for mode in ("normal", "rb"):
        scaling = [
            arm for arm in arms if arm.mode == mode and arm.stage == "replica_scaling"
        ]
        assert tuple(arm.replicas for arm in scaling) == REPLICA_COUNTS
        for arm in scaling:
            mapping = pilot_config_mapping(arm, output_dir="/tmp/pilot")
            scientific = mapping["scientific"]
            execution = mapping["run"]["execution"]
            assert scientific["schedule"] == {"batch_size": 256, "steps": 116}
            assert execution["precision"] == "bf16"
            assert "manifest" not in mapping["run"]
            resolved = load_config_mapping(mapping)
            assert resolved.execution.layout.model_dtype == torch.bfloat16
            assert resolved.execution.manifest.n_replica == arm.replicas


def test_local_smoke_changes_only_execution_scale_and_precision() -> None:
    arm = Arm(
        "nlhe_hu_32bb_rb_local_test",
        "local_smoke",
        "rb",
        2,
        1,
        4,
        family_pack=1,
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
    torch.save({"checkpoint_schema_version": "legacy", "model": state}, checkpoint)
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
    assert geometry_roots(1) == (2048, 3072, 4096)
    assert geometry_roots(8) == (384, 512, 768)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_model_only_initializer_reaches_every_replica_and_reference() -> None:
    source_arm = Arm(
        "nlhe_hu_32bb_normal_initializer_source",
        "local_smoke",
        "normal",
        1,
        1,
        None,
        family_pack=1,
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
        family_pack=1,
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
        family_pack=1,
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
        assert tuple(state.iteration for state in session.run_states) == (1,) * replicas
    finally:
        session.close()
