from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch
import yaml

from pbg.config import ConfigError, load_config, load_config_mapping
from pbg.core.execution_manifest import (
    ExecutionManifestError,
    seal_execution_manifest,
    validate_execution_manifest,
)
from pbg.games.poker.queued.config import SequentialModeConfig

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/train/nlhe_hu_8bb_normal_smoke.yaml"
PROFILE = ROOT / "configs/profiles/nlhe_hu_32bb_branching_profile_v1.yaml"


def _raw() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_public_config_compiles_one_resolved_runtime_contract() -> None:
    cfg = load_config(CONFIG)
    assert isinstance(cfg.scientific, SequentialModeConfig)
    assert cfg.mode == {
        "training_regime": "self_play",
        "rollout_mode": "sequential",
        "private_estimator": "sampled",
    }
    assert cfg.execution.layout.input_shapes == {
        "pre": (240,),
        "post_2": (448,),
    }
    manifest = cfg.execution.manifest
    assert manifest.topology_edge_cap == (
        manifest.topology_cap - manifest.root_count + manifest.leaf_cap
    )
    assert manifest.target_arena_cap == manifest.topology_cap + manifest.leaf_cap
    assert manifest.partition_rows("action_cap") == manifest.partition_rows(
        "trained_player_cap"
    )
    assert (
        manifest.hardware.total_memory_bytes
        == torch.cuda.get_device_properties(0).total_memory
    )


def test_source_round_trip_and_resolved_hash_are_deterministic() -> None:
    cfg = load_config(CONFIG)
    rebuilt = load_config_mapping(cfg.source_mapping())
    assert rebuilt.source_mapping() == cfg.source_mapping()
    assert rebuilt.execution.mapping() == cfg.execution.mapping()
    resolved = cfg.execution.mapping()
    assert resolved["schema"] == "pbg_resolved_execution_v1"
    assert resolved["derivation_version"] == "nlhe_capacity_v1"
    assert len(resolved["resolved_sha256"]) == 64
    json.dumps(resolved, sort_keys=True, allow_nan=False)


def _mutated_leaf(value: object, path: tuple[object, ...]) -> object:
    if not path:
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + 0.5
        if isinstance(value, str):
            return value + "_mutated"
        raise AssertionError(f"unsupported resolved leaf {type(value)}")
    head, *tail = path
    if isinstance(value, dict):
        return {**value, head: _mutated_leaf(value[head], tuple(tail))}
    if isinstance(value, list):
        result = list(value)
        result[int(head)] = _mutated_leaf(result[int(head)], tuple(tail))
        return result
    raise AssertionError("resolved path did not reach a container")


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if isinstance(value, dict):
        return [
            path
            for name, item in value.items()
            for path in _leaf_paths(item, (*prefix, name))
        ]
    if isinstance(value, list):
        return [
            path
            for index, item in enumerate(value)
            for path in _leaf_paths(item, (*prefix, index))
        ]
    return [prefix]


def test_every_resolved_leaf_is_hash_bound_and_resume_identity_bound() -> None:
    resolved = load_config(CONFIG).execution.mapping()
    payload = {name: value for name, value in resolved.items() if name != "resolved_sha256"}
    for path in _leaf_paths(payload):
        mutated_payload = _mutated_leaf(payload, path)
        tampered = {**mutated_payload, "resolved_sha256": resolved["resolved_sha256"]}
        with pytest.raises(ExecutionManifestError, match="hash"):
            validate_execution_manifest(tampered)
        resealed = seal_execution_manifest(mutated_payload)
        with pytest.raises(ExecutionManifestError, match="drifted"):
            validate_execution_manifest(resealed, expected=resolved)


@pytest.mark.parametrize("legacy_root", ["execution", "game", "hardware", "training"])
def test_legacy_root_shapes_are_rejected(legacy_root: str) -> None:
    raw = _raw()
    raw[legacy_root] = {}
    with pytest.raises(ConfigError, match=rf"root\.{legacy_root}"):
        load_config_mapping(raw)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("scientific",), "surprise"),
        (("scientific", "mode"), "legacy_mode"),
        (("scientific", "game"), "ante"),
        (("scientific", "model"), "unknown_head"),
        (("scientific", "objective"), "ignored_coefficient"),
        (("scientific", "optimizer"), "momentum"),
        (("run",), "manifest"),
        (("run", "execution"), "layout"),
        (("run", "execution", "memory"), "total_memory_bytes"),
    ],
)
def test_unknown_nested_key_reports_full_path(
    path: tuple[str, ...], field: str
) -> None:
    raw = _raw()
    target = raw
    for component in path:
        target = target[component]  # type: ignore[index,assignment]
    target[field] = 1
    dotted = r"\.".join((*path, field))
    with pytest.raises(ConfigError, match=rf"root\.{dotted}"):
        load_config_mapping(raw)


@pytest.mark.parametrize("device", ["cpu", "mps", "auto", ""])
def test_training_device_must_be_explicit_cuda(device: str) -> None:
    raw = _raw()
    raw["run"]["execution"]["device"] = device  # type: ignore[index]
    with pytest.raises((ConfigError, RuntimeError), match=r"device|CUDA"):
        load_config_mapping(raw)


@pytest.mark.parametrize(
    ("path", "value", "pattern"),
    [
        (("run", "iterations"), 0, r"root\.run\.iterations"),
        (("run", "checkpoint_every"), 0, r"root\.run\.checkpoint_every"),
        (("run", "workload", "roots"), 0, r"root\.run\.workload\.roots"),
        (
            ("run", "execution", "memory", "reserve_bytes"),
            -1,
            r"root\.run\.execution\.memory\.reserve_bytes",
        ),
        (("scientific", "lambda"), -0.01, r"root\.scientific\.lambda"),
        (
            ("scientific", "optimizer", "learning_rate"),
            -1.0,
            r"root\.scientific\.optimizer\.learning_rate",
        ),
        (
            ("scientific", "schedule", "batch_size"),
            0,
            r"root\.scientific\.schedule\.batch_size",
        ),
    ],
)
def test_numeric_boundaries_reject_with_dotted_paths(
    path: tuple[str, ...], value: object, pattern: str
) -> None:
    raw = _raw()
    target = raw
    for component in path[:-1]:
        target = target[component]  # type: ignore[index,assignment]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=pattern):
        load_config_mapping(raw)


def test_discriminated_unions_reject_inapplicable_placeholders() -> None:
    raw = _raw()
    raw["scientific"]["objective"]["value"]["epsilon"] = 0.2  # type: ignore[index]
    with pytest.raises(ConfigError, match=r"objective\.value\.epsilon"):
        load_config_mapping(raw)
    raw = _raw()
    raw["scientific"]["schedule"]["steps"] = 1  # type: ignore[index]
    with pytest.raises(ConfigError, match="select exactly one"):
        load_config_mapping(raw)
    raw = _raw()
    raw["run"]["execution"]["attention"] = {  # type: ignore[index]
        "kind": "sdpa",
        "query_rows": 128,
    }
    with pytest.raises(ConfigError, match="chunked_dense"):
        load_config_mapping(raw)


def test_self_play_rejects_frozen_target() -> None:
    raw = _raw()
    raw["run"]["frozen"] = {"path": "target.pt", "sha256": "0" * 64}  # type: ignore[index]
    with pytest.raises(ConfigError, match="response fields require best_response"):
        load_config_mapping(raw)


def test_distributed_policy_must_match_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw()
    raw["run"]["execution"]["distributed"] = True  # type: ignore[index]
    with pytest.raises(ConfigError, match="WORLD_SIZE"):
        load_config_mapping(raw)
    monkeypatch.setenv("WORLD_SIZE", "2")
    load_config_mapping(raw)


def test_input_mapping_is_copied_not_mutated() -> None:
    raw = _raw()
    before = copy.deepcopy(raw)
    load_config_mapping(raw)
    assert raw == before


def test_resolved_manifest_is_not_a_second_source_schema() -> None:
    cfg = load_config(CONFIG)
    with pytest.raises(ConfigError, match=r"root\.execution"):
        load_config_mapping(
            {
                "scientific": cfg.scientific_mapping(),
                "execution": cfg.execution.mapping(),
            }
        )


def test_profiler_scenarios_compile_without_mutation_or_null_placeholders() -> None:
    before = PROFILE.read_text(encoding="utf-8")
    loader = runpy.run_path(str(ROOT / "tools/profile-queued-branching.py"))[
        "_load_profile_config"
    ]
    successful = loader(PROFILE, "successful_seed104830")
    failed = loader(PROFILE, "failed_beta3_capacity")
    assert successful.root.source_mapping()["kind"] == "nlhe_run_v1"
    assert failed.scenario.cost_safeguard is None
    assert PROFILE.read_text(encoding="utf-8") == before
    assert ": null" not in before
