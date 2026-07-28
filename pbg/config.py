"""Strict public configuration for the queued NLHE engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from pbg.core.capacity import (
    CapacityManifest,
    ExternalWorkspaceBounds,
    ExecutionGeometry,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
)
from pbg.games.poker.queued.config import BranchedModeConfig, SequentialModeConfig
from pbg.games.poker.queued.hooks import NLHEGeometry
from pbg.core.storage import SoAField


class ConfigError(ValueError):
    """A public configuration is not closed or internally consistent."""


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected an object")
    return dict(value)


def _closed(source: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(source).difference(allowed))
    if unknown:
        raise ConfigError(f"{path}.{unknown[0]}: unknown configuration field")


def _rows(value: object, path: str) -> tuple[PartitionRows, ...]:
    source = _object(value, path)
    if not source:
        raise ConfigError(f"{path}: must be nonempty")
    try:
        return tuple(PartitionRows(str(name), rows) for name, rows in source.items())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class ExecutionConfig:
    device: str
    iterations: int
    checkpoint_every: int
    output_dir: Path
    run_id: str
    attention_backend: str
    query_chunk_size: int
    manifest: CapacityManifest
    layout: PhysicalLayout
    frozen_checkpoint: Path | None = None
    distributed: bool = False

    def __post_init__(self) -> None:
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ConfigError("root.execution.device: expected explicit CUDA") from exc
        if device.type != "cuda":
            raise ConfigError("root.execution.device: training requires CUDA")
        if self.iterations <= 0:
            raise ConfigError("root.execution.iterations: must be positive")
        if self.checkpoint_every <= 0:
            raise ConfigError("root.execution.checkpoint_every: must be positive")
        if not str(self.output_dir):
            raise ConfigError("root.execution.output_dir: must be nonempty")
        if not self.run_id:
            raise ConfigError("root.execution.run_id: must be nonempty")
        if self.attention_backend not in ("sdpa", "chunked_dense"):
            raise ConfigError(
                "root.execution.attention_backend: expected sdpa or chunked_dense"
            )
        if self.query_chunk_size <= 0:
            raise ConfigError("root.execution.query_chunk_size: must be positive")

    def mapping(self) -> dict[str, object]:
        manifest = self.manifest
        layout = self.layout
        return {
            "device": self.device,
            "iterations": self.iterations,
            "checkpoint_every": self.checkpoint_every,
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "attention_backend": self.attention_backend,
            "query_chunk_size": self.query_chunk_size,
            "distributed": self.distributed,
            "frozen_checkpoint": (
                None
                if self.frozen_checkpoint is None
                else str(self.frozen_checkpoint)
            ),
            "manifest": {
                "model_token_bound": manifest.geometry.model_token_bound,
                "model_token_pair_bound": (
                    manifest.geometry.model_token_pair_bound
                ),
                **{
                    name: getattr(manifest, name)
                    for name in (
                        "root_count",
                        "n_replica",
                        "beta_max",
                        "leaf_cap",
                        "topology_cap",
                        "topology_edge_cap",
                        "family_pack_cap",
                        "chance_frontier_cap",
                        "completion_pack_cap",
                        "learner_sigma_pack_cap",
                        "learner_action_pack_cap",
                        "topology_packet_cap",
                        "target_arena_cap",
                        "movement_scratch_bytes",
                        "library_workspace_bytes",
                        "evaluator_workspace_bytes",
                        "memory_guard_bytes",
                    )
                },
                **{
                    name: {
                        value.name: value.rows
                        for value in getattr(manifest, name)
                    }
                    for name in (
                        "player_node_cap",
                        "trained_player_cap",
                        "action_cap",
                        "folded_range_cap",
                    )
                },
                "hardware": {
                    name: getattr(manifest.hardware, name)
                    for name in manifest.hardware.__dataclass_fields__
                },
                "required_external": {
                    name: getattr(manifest.required_external, name)
                    for name in manifest.required_external.__dataclass_fields__
                },
            },
            "layout": {
                "model_input_shapes": {
                    value.name: list(value.shape)
                    for value in layout.model_input_shapes
                },
                **{
                    name: getattr(layout, name)
                    for name in (
                        "player_count",
                        "stratum_count",
                        "evaluator_private_width",
                        "family_workspace_row_bytes",
                        "chance_frontier_row_bytes",
                        "completion_workspace_row_bytes",
                        "learner_sigma_workspace_row_bytes",
                        "learner_action_workspace_row_bytes",
                        "topology_packet_row_bytes",
                        "reference_cache_bytes",
                        "field_alignment",
                    )
                },
                "model_dtype": str(layout.model_dtype).removeprefix("torch."),
            },
        }


@dataclass(frozen=True)
class RootConfig:
    """One immutable scientific row and one explicit execution manifest."""

    scientific: SequentialModeConfig | BranchedModeConfig
    execution: ExecutionConfig

    @property
    def mode(self) -> Mapping[str, str]:
        return self.scientific.mode

    def scientific_mapping(self) -> dict[str, object]:
        return self.scientific.scientific_mapping()


def _parse_execution(
    value: object,
    *,
    scientific: SequentialModeConfig | BranchedModeConfig,
) -> ExecutionConfig:
    source = _object(value, "root.execution")
    _closed(
        source,
        {
            "device",
            "iterations",
            "checkpoint_every",
            "output_dir",
            "run_id",
            "attention_backend",
            "query_chunk_size",
            "distributed",
            "frozen_checkpoint",
            "manifest",
            "layout",
        },
        "root.execution",
    )
    required_execution = {
        "device",
        "iterations",
        "checkpoint_every",
        "output_dir",
        "run_id",
        "attention_backend",
        "query_chunk_size",
        "manifest",
        "layout",
    }
    missing = sorted(required_execution.difference(source))
    if missing:
        raise ConfigError(f"root.execution.{missing[0]}: missing field")
    if type(source.get("distributed", False)) is not bool:
        raise ConfigError("root.execution.distributed: expected bool")
    for name in (
        "device",
        "output_dir",
        "run_id",
        "attention_backend",
    ):
        value = source[name]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"root.execution.{name}: expected nonempty string")
    for name in ("iterations", "checkpoint_every", "query_chunk_size"):
        value = source[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"root.execution.{name}: expected integer")
    manifest_raw = _object(source.get("manifest"), "root.execution.manifest")
    manifest_fields = {
        "model_token_bound",
        "model_token_pair_bound",
        "root_count",
        "n_replica",
        "beta_max",
        "player_node_cap",
        "trained_player_cap",
        "action_cap",
        "folded_range_cap",
        "leaf_cap",
        "topology_cap",
        "topology_edge_cap",
        "family_pack_cap",
        "chance_frontier_cap",
        "completion_pack_cap",
        "learner_sigma_pack_cap",
        "learner_action_pack_cap",
        "topology_packet_cap",
        "target_arena_cap",
        "movement_scratch_bytes",
        "library_workspace_bytes",
        "evaluator_workspace_bytes",
        "memory_guard_bytes",
        "hardware",
        "required_external",
    }
    _closed(manifest_raw, manifest_fields, "root.execution.manifest")
    missing = sorted(manifest_fields.difference(manifest_raw))
    if missing:
        raise ConfigError(f"root.execution.manifest.{missing[0]}: missing field")
    hardware_raw = _object(
        manifest_raw["hardware"], "root.execution.manifest.hardware"
    )
    hardware_fields = {
        "total_memory_bytes",
        "memory_budget_bytes",
        "training_max_tokens",
        "training_max_token_pairs",
        "query_chunk_size",
    }
    _closed(hardware_raw, hardware_fields, "root.execution.manifest.hardware")
    missing = sorted(hardware_fields.difference(hardware_raw))
    if missing:
        raise ConfigError(
            f"root.execution.manifest.hardware.{missing[0]}: missing field"
        )
    external_raw = _object(
        manifest_raw["required_external"],
        "root.execution.manifest.required_external",
    )
    _closed(
        external_raw,
        {"library_bytes", "evaluator_bytes"},
        "root.execution.manifest.required_external",
    )
    missing = sorted({"library_bytes", "evaluator_bytes"}.difference(external_raw))
    if missing:
        raise ConfigError(
            "root.execution.manifest.required_external."
            f"{missing[0]}: missing field"
        )
    common = scientific.common
    try:
        game_geometry = NLHEGeometry(
            common.game.players,
            common.game.stack_bb,
            common.game.rank_count,
        )
        manifest = CapacityManifest(
            geometry=ExecutionGeometry(
                game_geometry.players,
                game_geometry.player_node_bound,
                manifest_raw["model_token_bound"],
                manifest_raw["model_token_pair_bound"],
            ),
            root_count=manifest_raw["root_count"],
            n_replica=manifest_raw["n_replica"],
            beta_max=manifest_raw["beta_max"],
            player_node_cap=_rows(
                manifest_raw["player_node_cap"],
                "root.execution.manifest.player_node_cap",
            ),
            trained_player_cap=_rows(
                manifest_raw["trained_player_cap"],
                "root.execution.manifest.trained_player_cap",
            ),
            action_cap=_rows(
                manifest_raw["action_cap"], "root.execution.manifest.action_cap"
            ),
            folded_range_cap=_rows(
                manifest_raw["folded_range_cap"],
                "root.execution.manifest.folded_range_cap",
            ),
            **{
                name: manifest_raw[name]
                for name in manifest_fields
                - {
                    "root_count",
                    "n_replica",
                    "beta_max",
                    "model_token_bound",
                    "model_token_pair_bound",
                    "player_node_cap",
                    "trained_player_cap",
                    "action_cap",
                    "folded_range_cap",
                    "hardware",
                    "required_external",
                }
            },
            hardware=HardwareLimits(**hardware_raw),
            required_external=ExternalWorkspaceBounds(**external_raw),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"root.execution.manifest: {exc}") from exc
    if manifest.root_count != common.root_count:
        raise ConfigError(
            "root.execution.manifest.root_count: differs from scientific root_count"
        )
    layout_raw = _object(source.get("layout"), "root.execution.layout")
    layout_fields = {
        "model_input_shapes",
        "player_count",
        "stratum_count",
        "evaluator_private_width",
        "family_workspace_row_bytes",
        "chance_frontier_row_bytes",
        "completion_workspace_row_bytes",
        "learner_sigma_workspace_row_bytes",
        "learner_action_workspace_row_bytes",
        "topology_packet_row_bytes",
        "reference_cache_bytes",
        "field_alignment",
        "model_dtype",
    }
    _closed(layout_raw, layout_fields, "root.execution.layout")
    required_layout = layout_fields - {
        "reference_cache_bytes",
        "field_alignment",
        "model_dtype",
    }
    missing = sorted(required_layout.difference(layout_raw))
    if missing:
        raise ConfigError(f"root.execution.layout.{missing[0]}: missing field")
    shapes_raw = _object(
        layout_raw["model_input_shapes"],
        "root.execution.layout.model_input_shapes",
    )
    dtype_name = layout_raw.get("model_dtype", "float32")
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtypes:
        raise ConfigError(
            "root.execution.layout.model_dtype: expected float32, float16, or bfloat16"
        )
    try:
        layout = PhysicalLayout(
            model_input_shapes=tuple(
                NamedShape(str(name), tuple(shape))
                for name, shape in shapes_raw.items()
            ),
            **{
                name: layout_raw[name]
                for name in required_layout - {"model_input_shapes"}
            },
            reference_cache_bytes=layout_raw.get("reference_cache_bytes", 0),
            field_alignment=layout_raw.get("field_alignment", 64),
            model_dtype=dtypes[str(dtype_name)],
            opaque_fields=(
                SoAField(
                    "leaf_public_state",
                    torch.uint8,
                    (5,),
                    layout_raw.get("field_alignment", 64),
                    255,
                ),
                SoAField(
                    "leaf_private_profile",
                    torch.int32,
                    (common.game.players,),
                    layout_raw.get("field_alignment", 64),
                    -1,
                ),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"root.execution.layout: {exc}") from exc
    frozen = source.get("frozen_checkpoint")
    if frozen is not None and (
        not isinstance(frozen, str) or not frozen.strip()
    ):
        raise ConfigError(
            "root.execution.frozen_checkpoint: expected nonempty path or null"
        )
    if scientific.mode["training_regime"] == "best_response" and frozen is None:
        raise ConfigError(
            "root.execution.frozen_checkpoint: best response requires a target"
        )
    if scientific.mode["training_regime"] == "self_play" and frozen is not None:
        raise ConfigError(
            "root.execution.frozen_checkpoint: inapplicable to self play"
        )
    return ExecutionConfig(
        device=str(source.get("device")),
        iterations=source.get("iterations"),
        checkpoint_every=source.get("checkpoint_every"),
        output_dir=Path(str(source.get("output_dir"))),
        run_id=str(source.get("run_id")),
        attention_backend=str(source.get("attention_backend")),
        query_chunk_size=source.get("query_chunk_size"),
        distributed=source.get("distributed", False),
        frozen_checkpoint=None if frozen is None else Path(str(frozen)),
        manifest=manifest,
        layout=layout,
    )


def load_config_mapping(raw: Mapping[str, object]) -> RootConfig:
    source = _object(raw, "root")
    if set(source) != {"scientific", "execution"}:
        unknown = sorted(set(source).difference({"scientific", "execution"}))
        if unknown:
            raise ConfigError(f"root.{unknown[0]}: unknown configuration field")
        missing = sorted({"scientific", "execution"}.difference(source))
        raise ConfigError(f"root.{missing[0]}: missing field")
    scientific_raw = _object(source["scientific"], "root.scientific")
    mode = _object(scientific_raw.get("mode"), "root.scientific.mode")
    mode_fields = {
        "training_regime",
        "rollout_mode",
        "private_estimator",
    }
    unknown_mode = sorted(set(mode).difference(mode_fields))
    if unknown_mode:
        raise ConfigError(
            f"root.scientific.mode.{unknown_mode[0]}: unknown configuration field"
        )
    missing_mode = sorted(mode_fields.difference(mode))
    if missing_mode:
        raise ConfigError(f"root.scientific.mode.{missing_mode[0]}: missing field")
    try:
        scientific = (
            SequentialModeConfig.from_mapping(scientific_raw)
            if mode.get("rollout_mode") == "sequential"
            else BranchedModeConfig.from_mapping(scientific_raw)
        )
        execution = _parse_execution(source["execution"], scientific=scientific)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if message.startswith("root."):
            message = message.replace("root.", "root.scientific.", 1)
        elif "root_count" in message:
            message = f"root.scientific.root_count: {message}"
        elif "lambda_" in message:
            message = f"root.scientific.lambda: {message}"
        elif "seed" in message:
            message = f"root.scientific.seed: {message}"
        elif "reproducibility" in message:
            message = f"root.scientific.reproducibility: {message}"
        elif "utility" in message:
            message = f"root.scientific.utility: {message}"
        elif "game must satisfy" in message:
            message = f"root.scientific.game: {message}"
        elif "model max_players" in message:
            message = f"root.scientific.model.max_players: {message}"
        elif "folder_beta_rank" in message:
            message = f"root.scientific.model.folder_beta_rank: {message}"
        else:
            message = f"root.scientific.configuration: {message}"
        raise ConfigError(message) from exc
    return RootConfig(scientific, execution)


def load_config(path: str | Path) -> RootConfig:
    source = Path(path)
    try:
        raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{source}: unable to load configuration: {exc}") from exc
    return load_config_mapping(_object(raw, "root"))


__all__ = [
    "ConfigError",
    "ExecutionConfig",
    "RootConfig",
    "load_config",
    "load_config_mapping",
]
