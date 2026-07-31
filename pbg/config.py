"""Canonical user configuration and deterministic resolved-manifest compiler."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import yaml

from pbg.core.capacity import CapacityManifest, PhysicalLayout
from pbg.core.contracts import mode_schema
from pbg.games.poker.queued.capacity_compiler import (
    CAPACITY_DERIVATION_VERSION,
    WorkloadSpec,
    compile_capacity,
)
from pbg.games.poker.queued.config import BranchedModeConfig, SequentialModeConfig
from pbg.games.poker.queued.hooks import NLHEGeometry

RUN_CONFIG_KIND = "nlhe_run_v1"
RESOLVED_EXECUTION_SCHEMA = "pbg_resolved_execution_v1"


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


def _required(source: Mapping[str, object], names: set[str], path: str) -> None:
    missing = sorted(names.difference(source))
    if missing:
        raise ConfigError(f"{path}.{missing[0]}: missing field")


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path}: expected positive integer")
    return int(value)


def _nonnegative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{path}: expected nonnegative integer")
    return int(value)


def _number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected number")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ConfigError(f"{path}: expected {qualifier} number")
    return result


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: expected nonempty string")
    return value


def _sha256(value: object, path: str) -> str:
    result = _nonempty_string(value, path)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ConfigError(f"{path}: expected lowercase SHA-256")
    return result


def _canonical_hash(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ExecutionConfig:
    """Resolved execution state; never parsed from normal YAML."""

    device: str
    iterations: int
    checkpoint_every: int
    output_dir: Path
    run_id: str
    attention_backend: str
    query_chunk_size: int
    manifest: CapacityManifest
    layout: PhysicalLayout
    derivations: Mapping[str, str]
    source_sha256: str
    frozen_checkpoint: Path | None = None
    distributed: bool = False

    def __post_init__(self) -> None:
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ConfigError(
                "root.run.execution.device: expected explicit CUDA"
            ) from exc
        if device.type != "cuda":
            raise ConfigError("root.run.execution.device: training requires CUDA")
        if self.iterations <= 0 or self.checkpoint_every <= 0:
            raise ConfigError("root.run: iteration counts must be positive")
        if self.attention_backend not in ("sdpa", "chunked_dense"):
            raise ConfigError("root.run.execution.attention: invalid backend")
        if self.query_chunk_size <= 0:
            raise ConfigError("root.run.execution.attention: invalid query rows")

    def _payload(self) -> dict[str, object]:
        manifest = self.manifest
        layout = self.layout
        result: dict[str, object] = {
            "schema": RESOLVED_EXECUTION_SCHEMA,
            "derivation_version": CAPACITY_DERIVATION_VERSION,
            "derivations": dict(self.derivations),
            "source_sha256": self.source_sha256,
            "device": self.device,
            "iterations": self.iterations,
            "checkpoint_every": self.checkpoint_every,
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "attention": (
                "sdpa"
                if self.attention_backend == "sdpa"
                else {
                    "kind": "chunked_dense",
                    "query_rows": self.query_chunk_size,
                }
            ),
            "distributed": self.distributed,
            "manifest": {
                "geometry": {
                    name: getattr(manifest.geometry, name)
                    for name in manifest.geometry.__dataclass_fields__
                },
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
                    name: {row.name: row.rows for row in getattr(manifest, name)}
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
                    value.name: list(value.shape) for value in layout.model_input_shapes
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
        if self.frozen_checkpoint is not None:
            result["frozen_checkpoint"] = str(self.frozen_checkpoint)
        return result

    def mapping(self) -> dict[str, object]:
        """Serialize the generated immutable product for checkpoints/reports."""

        payload = self._payload()
        return {**payload, "resolved_sha256": _canonical_hash(payload)}


@dataclass(frozen=True)
class RootConfig:
    """One canonical user specification and its compiled runtime product."""

    scientific: SequentialModeConfig | BranchedModeConfig
    execution: ExecutionConfig
    _source_mapping: Mapping[str, object]

    @property
    def mode(self) -> Mapping[str, str]:
        return self.scientific.mode

    def scientific_mapping(self) -> dict[str, object]:
        """Fully resolved scientific identity used by checkpoints."""

        return self.scientific.scientific_mapping()

    def source_mapping(self) -> dict[str, object]:
        """Return the deterministic canonical user-specification mapping."""

        return _plain(self._source_mapping)  # type: ignore[return-value]


def _parse_model(
    value: object, *, players: int
) -> tuple[dict[str, object], dict[str, object]]:
    source = _object(value, "root.scientific.model")
    allowed = {
        "d_model",
        "n_heads",
        "layers",
        "raises",
        "seat_capacity",
        "class_embedding_dim",
        "ffn_hidden",
        "folder_beta_rank",
        "folder_line_dim",
        "epsilon",
    }
    _closed(source, allowed, "root.scientific.model")
    layers = _object(source.get("layers", {}), "root.scientific.model.layers")
    _closed(layers, {"pre", "post"}, "root.scientific.model.layers")
    raises = _object(source.get("raises", {}), "root.scientific.model.raises")
    _closed(raises, {"components", "minimum_scale"}, "root.scientific.model.raises")
    if "ffn_hidden" in source and source["ffn_hidden"] is None:
        raise ConfigError("root.scientific.model.ffn_hidden: omit to use the default")
    internal = {
        "d_model": source.get("d_model", 256),
        "n_heads": source.get("n_heads", 8),
        "n_layers_pre": layers.get("pre", 4),
        "n_layers_post": layers.get("post", 4),
        "m_raise": raises.get("components", 4),
        "s_min": raises.get("minimum_scale", 0.02),
        "max_players": source.get("seat_capacity", players),
        "class_embedding_dim": source.get("class_embedding_dim", 32),
        "ffn_hidden": source.get("ffn_hidden"),
        "folder_beta_rank": source.get("folder_beta_rank", 16),
        "folder_line_dim": source.get("folder_line_dim", 8),
        "epsilon": source.get("epsilon", 1.0e-8),
    }
    canonical: dict[str, object] = {
        "d_model": internal["d_model"],
        "n_heads": internal["n_heads"],
        "layers": {"pre": internal["n_layers_pre"], "post": internal["n_layers_post"]},
        "raises": {
            "components": internal["m_raise"],
            "minimum_scale": internal["s_min"],
        },
        "class_embedding_dim": internal["class_embedding_dim"],
        "folder_beta_rank": internal["folder_beta_rank"],
        "folder_line_dim": internal["folder_line_dim"],
        "epsilon": internal["epsilon"],
    }
    if internal["max_players"] != players:
        canonical["seat_capacity"] = internal["max_players"]
    if internal["ffn_hidden"] is not None:
        canonical["ffn_hidden"] = internal["ffn_hidden"]
    return internal, canonical


def _parse_objective(
    value: object, *, branched: bool
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source = _object(value, "root.scientific.objective")
    allowed = {"policy_clip", "value", "entropy", "reference_kl", "branching"}
    _closed(source, allowed, "root.scientific.objective")
    _required(source, {"policy_clip", "value", "entropy"}, "root.scientific.objective")

    value_spec = _object(source["value"], "root.scientific.objective.value")
    kind = value_spec.get("kind")
    expected_value = (
        {"kind", "coefficient", "epsilon"}
        if kind == "clipped"
        else {"kind", "coefficient"}
    )
    _closed(value_spec, expected_value, "root.scientific.objective.value")
    _required(value_spec, expected_value, "root.scientific.objective.value")
    if kind not in ("mse", "clipped"):
        raise ConfigError(
            "root.scientific.objective.value.kind: expected mse or clipped"
        )

    entropy = _object(source["entropy"], "root.scientific.objective.entropy")
    entropy_kind = entropy.get("kind")
    expected_entropy = (
        {"kind", "coefficient", "iterations"}
        if entropy_kind == "linear_decay"
        else {"kind", "coefficient"}
    )
    _closed(entropy, expected_entropy, "root.scientific.objective.entropy")
    _required(entropy, expected_entropy, "root.scientific.objective.entropy")
    if entropy_kind not in ("constant", "linear_decay"):
        raise ConfigError("root.scientific.objective.entropy.kind: invalid schedule")

    reference_internal = {
        "interval_iterations": 1,
        "interval_effective_hands": None,
        "cache_outputs": False,
    }
    reference_canonical: dict[str, object] | None = None
    reference_coefficient = 0.0
    if "reference_kl" in source:
        reference = _object(
            source["reference_kl"], "root.scientific.objective.reference_kl"
        )
        _closed(
            reference,
            {"coefficient", "refresh"},
            "root.scientific.objective.reference_kl",
        )
        _required(
            reference,
            {"coefficient", "refresh"},
            "root.scientific.objective.reference_kl",
        )
        reference_coefficient = _number(
            reference["coefficient"],
            "root.scientific.objective.reference_kl.coefficient",
            positive=True,
        )
        refresh = _object(
            reference["refresh"], "root.scientific.objective.reference_kl.refresh"
        )
        if set(refresh) == {"iterations"}:
            cadence = _positive_int(
                refresh["iterations"],
                "root.scientific.objective.reference_kl.refresh.iterations",
            )
            reference_internal = {
                "interval_iterations": cadence,
                "interval_effective_hands": None,
                "cache_outputs": False,
            }
        elif set(refresh) == {"effective_hands"}:
            cadence = _number(
                refresh["effective_hands"],
                "root.scientific.objective.reference_kl.refresh.effective_hands",
                positive=True,
            )
            reference_internal = {
                "interval_iterations": None,
                "interval_effective_hands": cadence,
                "cache_outputs": False,
            }
        else:
            raise ConfigError(
                "root.scientific.objective.reference_kl.refresh: select exactly one cadence"
            )
        reference_canonical = {
            "coefficient": reference_coefficient,
            "refresh": dict(refresh),
        }

    branching_loss = source.get("branching")
    if branched:
        values = _object(branching_loss, "root.scientific.objective.branching")
        expected = {"trace_coefficient", "cost_coefficient", "trace_epsilon"}
        _closed(values, expected, "root.scientific.objective.branching")
        _required(values, expected, "root.scientific.objective.branching")
    elif branching_loss is not None:
        raise ConfigError(
            "root.scientific.objective.branching: inapplicable to sequential rollout"
        )
    else:
        values = {
            "trace_coefficient": 0.0,
            "cost_coefficient": 0.0,
            "trace_epsilon": 1.0e-6,
        }

    loss = {
        "clip_epsilon": source["policy_clip"],
        "value_clip_epsilon": value_spec.get("epsilon", 0.0),
        "value_coefficient": value_spec["coefficient"],
        "entropy_coefficient": entropy["coefficient"],
        "reference_kl_coefficient": reference_coefficient,
        "value_loss": kind,
        "entropy_schedule": entropy_kind,
        "entropy_decay_iterations": entropy.get("iterations"),
        **values,
    }
    canonical: dict[str, object] = {
        "policy_clip": source["policy_clip"],
        "value": dict(value_spec),
        "entropy": dict(entropy),
    }
    if reference_canonical is not None:
        canonical["reference_kl"] = reference_canonical
    if branched:
        canonical["branching"] = dict(values)
    return loss, reference_internal, canonical


def _parse_optimizer(value: object) -> tuple[dict[str, object], dict[str, object]]:
    source = _object(value, "root.scientific.optimizer")
    kind = source.get("kind")
    expected = {"kind", "learning_rate", "beta1", "beta2", "epsilon", "grad_clip"}
    if kind == "adamw":
        expected.add("weight_decay")
    _closed(source, expected, "root.scientific.optimizer")
    _required(source, expected, "root.scientific.optimizer")
    if kind not in ("adam", "adamw"):
        raise ConfigError("root.scientific.optimizer.kind: expected adam or adamw")
    internal = {**source, "name": source["kind"]}
    internal.pop("kind")
    if kind == "adam":
        internal["weight_decay"] = 0.0
    return internal, dict(source)


def _parse_schedule(value: object) -> tuple[dict[str, object], dict[str, object]]:
    source = _object(value, "root.scientific.schedule")
    _closed(source, {"batch_size", "epochs", "steps"}, "root.scientific.schedule")
    _required(source, {"batch_size"}, "root.scientific.schedule")
    if ("epochs" in source) == ("steps" in source):
        raise ConfigError(
            "root.scientific.schedule: select exactly one of epochs or steps"
        )
    batch = _positive_int(source["batch_size"], "root.scientific.schedule.batch_size")
    if "steps" in source:
        count = _positive_int(source["steps"], "root.scientific.schedule.steps")
        return {"batch_size": batch, "epochs": 1, "gradient_steps": count}, {
            "batch_size": batch,
            "steps": count,
        }
    count = _positive_int(source["epochs"], "root.scientific.schedule.epochs")
    return {"batch_size": batch, "epochs": count, "gradient_steps": None}, {
        "batch_size": batch,
        "epochs": count,
    }


def _parse_attention(value: object) -> tuple[str, int, object]:
    if value == "sdpa":
        return "sdpa", 1, "sdpa"
    source = _object(value, "root.run.execution.attention")
    _closed(source, {"kind", "query_rows"}, "root.run.execution.attention")
    _required(source, {"kind", "query_rows"}, "root.run.execution.attention")
    if source["kind"] != "chunked_dense":
        raise ConfigError("root.run.execution.attention.kind: expected chunked_dense")
    rows = _positive_int(
        source["query_rows"], "root.run.execution.attention.query_rows"
    )
    return "chunked_dense", rows, {"kind": "chunked_dense", "query_rows": rows}


def _parse_microbatch(value: object) -> tuple[dict[str, int | None], object]:
    if value == "auto":
        return {"rollout": None, "completion": None, "learner": None}, "auto"
    source = _object(value, "root.run.execution.microbatch")
    allowed = {"rollout_rows", "completion_rows", "learner_rows"}
    _closed(source, allowed, "root.run.execution.microbatch")
    if not source:
        raise ConfigError(
            "root.run.execution.microbatch: expected at least one preference"
        )
    result: dict[str, int | None] = {
        "rollout": None,
        "completion": None,
        "learner": None,
    }
    for public, internal in (
        ("rollout_rows", "rollout"),
        ("completion_rows", "completion"),
        ("learner_rows", "learner"),
    ):
        if public in source:
            result[internal] = _positive_int(
                source[public], f"root.run.execution.microbatch.{public}"
            )
    return result, dict(source)


def _parse_branching(
    value: object,
    *,
    rollout_mode: str,
    tree_nodes: int,
    recovery: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], int]:
    source = _object(value, "root.scientific.branching")
    allowed = {"fixed_beta", "beta_max", "minimum_tree_size", "kkt", "cost_safeguard"}
    _closed(source, allowed, "root.scientific.branching")
    required = {"fixed_beta", "beta_max", "minimum_tree_size"}
    if rollout_mode == "branched.KKT":
        required.add("kkt")
    _required(source, required, "root.scientific.branching")
    if rollout_mode == "branched.fixed" and "kkt" in source:
        raise ConfigError(
            "root.scientific.branching.kkt: inapplicable to fixed branching"
        )
    beta = source["beta_max"]
    if isinstance(beta, Mapping):
        beta_values = dict(beta)
        beta_max = max(
            _positive_int(item, f"root.scientific.branching.beta_max.{name}")
            for name, item in beta_values.items()
        )
    else:
        beta_max = _positive_int(beta, "root.scientific.branching.beta_max")
    minimum_tree = _positive_int(
        source["minimum_tree_size"], "root.scientific.branching.minimum_tree_size"
    )
    if minimum_tree > tree_nodes:
        raise ConfigError(
            "root.scientific.branching.minimum_tree_size: exceeds tree_nodes"
        )
    internal: dict[str, object] = {
        "fixed_beta": source["fixed_beta"],
        "beta_max": source["beta_max"],
        "minimum_tree_size": minimum_tree,
        "minimum_scale": recovery.get("minimum_branch_scale", 0.25),
        "backoff_factor": recovery.get("branch_backoff_factor", 0.5),
        "safe_tree_size": float(tree_nodes),
    }
    canonical = {
        "fixed_beta": _plain(source["fixed_beta"]),
        "beta_max": _plain(source["beta_max"]),
        "minimum_tree_size": minimum_tree,
    }
    if rollout_mode == "branched.KKT":
        kkt = _object(source["kkt"], "root.scientific.branching.kkt")
        expected = {
            "warmup_iterations",
            "target_tree_size",
            "gamma_kappa",
            "eta_kappa",
            "kappa_min",
            "kappa_max",
            "log_kappa",
        }
        _closed(kkt, expected, "root.scientific.branching.kkt")
        _required(kkt, expected, "root.scientific.branching.kkt")
        target = _number(
            kkt["target_tree_size"],
            "root.scientific.branching.kkt.target_tree_size",
            positive=True,
        )
        if not minimum_tree <= target <= tree_nodes:
            raise ConfigError(
                "root.scientific.branching.kkt.target_tree_size: must lie between minimum_tree_size and tree_nodes"
            )
        internal.update(kkt)
        canonical["kkt"] = dict(kkt)
    safeguard = source.get(
        "cost_safeguard",
        {
            "gamma_deviance_max": 1.0,
            "p95_underprediction_max": 2.0,
            "consecutive_iterations": 1,
        },
    )
    safeguard_mapping = _object(safeguard, "root.scientific.branching.cost_safeguard")
    expected_safeguard = {
        "gamma_deviance_max",
        "p95_underprediction_max",
        "consecutive_iterations",
    }
    _closed(
        safeguard_mapping,
        expected_safeguard,
        "root.scientific.branching.cost_safeguard",
    )
    _required(
        safeguard_mapping,
        expected_safeguard,
        "root.scientific.branching.cost_safeguard",
    )
    canonical["cost_safeguard"] = dict(safeguard_mapping)
    return internal, canonical, beta_max


def load_config_mapping(raw: Mapping[str, object]) -> RootConfig:
    """Parse only the canonical small run specification."""

    source = _object(raw, "root")
    _closed(source, {"kind", "scientific", "run"}, "root")
    _required(source, {"kind", "scientific", "run"}, "root")
    if source["kind"] != RUN_CONFIG_KIND:
        raise ConfigError(f"root.kind: expected {RUN_CONFIG_KIND}")
    scientific_source = _object(source["scientific"], "root.scientific")
    run_source = _object(source["run"], "root.run")
    _closed(
        scientific_source,
        {
            "mode",
            "game",
            "model",
            "objective",
            "optimizer",
            "schedule",
            "lambda",
            "completion",
            "branching",
            "response_initialization",
            "probe",
        },
        "root.scientific",
    )
    _required(
        scientific_source,
        {"mode", "game", "model", "objective", "optimizer", "schedule", "lambda"},
        "root.scientific",
    )
    _closed(
        run_source,
        {
            "seed",
            "reproducibility",
            "utility",
            "workload",
            "recovery",
            "execution",
            "iterations",
            "checkpoint_every",
            "output_dir",
            "run_id",
            "frozen",
        },
        "root.run",
    )
    _required(
        run_source,
        {
            "seed",
            "reproducibility",
            "utility",
            "workload",
            "execution",
            "iterations",
            "checkpoint_every",
            "output_dir",
        },
        "root.run",
    )

    mode = _object(scientific_source["mode"], "root.scientific.mode")
    mode_fields = {"training_regime", "rollout_mode", "private_estimator"}
    _closed(mode, mode_fields, "root.scientific.mode")
    _required(mode, mode_fields, "root.scientific.mode")
    try:
        schema = mode_schema(**mode)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"root.scientific.mode: {exc}") from exc
    branched = schema.rollout_mode.is_branched
    rb = schema.private_estimator.value == "rao_blackwell"
    response = schema.training_regime.value == "best_response"

    game = _object(scientific_source["game"], "root.scientific.game")
    _closed(game, {"players", "stack_bb", "rank_count"}, "root.scientific.game")
    _required(game, {"players", "stack_bb", "rank_count"}, "root.scientific.game")
    try:
        geometry = NLHEGeometry(
            int(game["players"]), float(game["stack_bb"]), int(game["rank_count"])
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"root.scientific.game: {exc}") from exc

    workload_source = _object(run_source["workload"], "root.run.workload")
    _closed(workload_source, {"roots", "replicas", "tree_nodes"}, "root.run.workload")
    _required(workload_source, {"roots"}, "root.run.workload")
    roots = _positive_int(workload_source["roots"], "root.run.workload.roots")
    replicas = _positive_int(
        workload_source.get("replicas", 1), "root.run.workload.replicas"
    )
    tree_nodes = (
        _positive_int(workload_source["tree_nodes"], "root.run.workload.tree_nodes")
        if "tree_nodes" in workload_source
        else None
    )
    if branched and tree_nodes is None:
        raise ConfigError("root.run.workload.tree_nodes: required for branched rollout")
    if not branched and tree_nodes is not None:
        raise ConfigError(
            "root.run.workload.tree_nodes: inapplicable to sequential rollout"
        )

    recovery = _object(run_source.get("recovery", {}), "root.run.recovery")
    recovery_allowed = {"minimum_roots"}
    if branched:
        recovery_allowed |= {"minimum_branch_scale", "branch_backoff_factor"}
    _closed(recovery, recovery_allowed, "root.run.recovery")
    minimum_roots = _positive_int(
        recovery.get("minimum_roots", roots), "root.run.recovery.minimum_roots"
    )
    if minimum_roots > roots:
        raise ConfigError("root.run.recovery.minimum_roots: exceeds workload roots")

    model, canonical_model = _parse_model(
        scientific_source["model"], players=geometry.players
    )
    loss, reference, canonical_objective = _parse_objective(
        scientific_source["objective"], branched=branched
    )
    optimizer, canonical_optimizer = _parse_optimizer(scientific_source["optimizer"])
    schedule, canonical_schedule = _parse_schedule(scientific_source["schedule"])

    completion_samples = 1
    completion_canonical: dict[str, object] | None = None
    if rb:
        completion = _object(
            scientific_source.get("completion"), "root.scientific.completion"
        )
        _closed(completion, {"samples"}, "root.scientific.completion")
        _required(completion, {"samples"}, "root.scientific.completion")
        completion_samples = _positive_int(
            completion["samples"], "root.scientific.completion.samples"
        )
        completion_canonical = {"samples": completion_samples}
    elif "completion" in scientific_source:
        raise ConfigError(
            "root.scientific.completion: inapplicable to sampled estimator"
        )

    branching_internal: dict[str, object] | None = None
    branching_canonical: dict[str, object] | None = None
    beta_max = 1
    if branched:
        assert tree_nodes is not None
        branching_internal, branching_canonical, beta_max = _parse_branching(
            scientific_source.get("branching"),
            rollout_mode=str(mode["rollout_mode"]),
            tree_nodes=tree_nodes,
            recovery=recovery,
        )
    elif "branching" in scientific_source:
        raise ConfigError(
            "root.scientific.branching: inapplicable to sequential rollout"
        )

    frozen_path: Path | None = None
    frozen_sha: str | None = None
    if response:
        frozen = _object(run_source.get("frozen"), "root.run.frozen")
        _closed(frozen, {"path", "sha256"}, "root.run.frozen")
        _required(frozen, {"path", "sha256"}, "root.run.frozen")
        frozen_path = Path(_nonempty_string(frozen["path"], "root.run.frozen.path"))
        frozen_sha = _sha256(frozen["sha256"], "root.run.frozen.sha256")
        if scientific_source.get("response_initialization") not in ("random", "target"):
            raise ConfigError(
                "root.scientific.response_initialization: expected random or target"
            )
    elif (
        "frozen" in run_source
        or "response_initialization" in scientific_source
        or "probe" in scientific_source
    ):
        raise ConfigError("root.run.frozen: response fields require best_response")

    execution_source = _object(run_source["execution"], "root.run.execution")
    _closed(
        execution_source,
        {
            "device",
            "precision",
            "attention",
            "memory",
            "microbatch",
            "distributed",
            "reference_cache",
        },
        "root.run.execution",
    )
    _required(
        execution_source,
        {"device", "precision", "attention", "memory", "microbatch"},
        "root.run.execution",
    )
    device_name = _nonempty_string(
        execution_source["device"], "root.run.execution.device"
    )
    try:
        device = torch.device(device_name)
    except RuntimeError as exc:
        raise ConfigError("root.run.execution.device: expected explicit CUDA") from exc
    if device.type != "cuda":
        raise ConfigError("root.run.execution.device: training requires explicit CUDA")
    precision = execution_source["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ConfigError("root.run.execution.precision: expected fp32, fp16, or bf16")
    attention_backend, query_rows, canonical_attention = _parse_attention(
        execution_source["attention"]
    )
    microbatch, canonical_microbatch = _parse_microbatch(execution_source["microbatch"])
    memory = _object(execution_source["memory"], "root.run.execution.memory")
    _closed(memory, {"reserve_bytes"}, "root.run.execution.memory")
    _required(memory, {"reserve_bytes"}, "root.run.execution.memory")
    reserve_bytes = _nonnegative_int(
        memory["reserve_bytes"], "root.run.execution.memory.reserve_bytes"
    )
    distributed = execution_source.get("distributed", False)
    if type(distributed) is not bool:
        raise ConfigError("root.run.execution.distributed: expected bool")
    try:
        launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ConfigError("WORLD_SIZE must be an integer") from exc
    if bool(distributed) != (launch_world_size > 1):
        raise ConfigError(
            "root.run.execution.distributed: differs from launcher WORLD_SIZE"
        )

    reference_coefficient = float(loss["reference_kl_coefficient"])
    cache_policy = execution_source.get("reference_cache", "off")
    if reference_coefficient == 0 and cache_policy != "off":
        raise ConfigError("root.run.execution.reference_cache: requires reference_kl")
    if cache_policy == "off":
        cache_bytes = 0
        canonical_cache: object = "off"
    elif cache_policy == "auto":
        total = int(torch.cuda.get_device_properties(device).total_memory)
        cache_bytes = max(1, (total - reserve_bytes) // 20)
        canonical_cache = "auto"
        reference["cache_outputs"] = True
    else:
        cache = _object(cache_policy, "root.run.execution.reference_cache")
        _closed(cache, {"budget_bytes"}, "root.run.execution.reference_cache")
        _required(cache, {"budget_bytes"}, "root.run.execution.reference_cache")
        cache_bytes = _positive_int(
            cache["budget_bytes"], "root.run.execution.reference_cache.budget_bytes"
        )
        canonical_cache = {"budget_bytes": cache_bytes}
        reference["cache_outputs"] = True

    internal_scientific: dict[str, object] = {
        "mode": dict(mode),
        "game": dict(game),
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "schedule": schedule,
        "precision": {"name": precision},
        "reference": reference,
        "lambda": scientific_source["lambda"],
        "root_count": roots,
        "root_count_minimum": minimum_roots,
        "seed": run_source["seed"],
        "reproducibility": run_source["reproducibility"],
        "utility": run_source["utility"],
    }
    if completion_canonical is not None:
        internal_scientific["completion"] = {"n_samples": completion_samples}
    if branching_internal is not None:
        internal_scientific["branching"] = branching_internal
        internal_scientific["cost_safeguard"] = branching_canonical["cost_safeguard"]  # type: ignore[index]
    if response:
        internal_scientific["frozen_profile"] = {"sha256": frozen_sha}
        internal_scientific["response_initialization"] = scientific_source[
            "response_initialization"
        ]
        if "probe" in scientific_source:
            internal_scientific["probe"] = scientific_source["probe"]
    try:
        scientific = (
            BranchedModeConfig.from_mapping(internal_scientific)
            if branched
            else SequentialModeConfig.from_mapping(internal_scientific)
        )
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if message.startswith("root."):
            message = message.replace("root.", "root.scientific.", 1)
        elif "lambda_" in message:
            message = f"root.scientific.lambda: {message}"
        elif "seed" in message:
            message = f"root.run.seed: {message}"
        else:
            message = f"root.scientific.configuration: {message}"
        raise ConfigError(message) from exc

    canonical_scientific: dict[str, object] = {
        "mode": dict(mode),
        "game": dict(game),
        "model": canonical_model,
        "objective": canonical_objective,
        "optimizer": canonical_optimizer,
        "schedule": canonical_schedule,
        "lambda": scientific_source["lambda"],
    }
    if completion_canonical is not None:
        canonical_scientific["completion"] = completion_canonical
    if branching_canonical is not None:
        canonical_scientific["branching"] = branching_canonical
    if response:
        canonical_scientific["response_initialization"] = scientific_source[
            "response_initialization"
        ]
        if "probe" in scientific_source:
            canonical_scientific["probe"] = _plain(scientific_source["probe"])

    canonical_execution: dict[str, object] = {
        "device": device_name,
        "precision": precision,
        "attention": canonical_attention,
        "memory": {"reserve_bytes": reserve_bytes},
        "microbatch": canonical_microbatch,
    }
    if distributed:
        canonical_execution["distributed"] = True
    if canonical_cache != "off":
        canonical_execution["reference_cache"] = canonical_cache
    output_dir = Path(_nonempty_string(run_source["output_dir"], "root.run.output_dir"))
    run_id = _nonempty_string(
        run_source.get("run_id", output_dir.name), "root.run.run_id"
    )
    canonical_run: dict[str, object] = {
        "seed": run_source["seed"],
        "reproducibility": run_source["reproducibility"],
        "utility": run_source["utility"],
        "workload": {
            "roots": roots,
            **({"replicas": replicas} if replicas != 1 else {}),
            **({"tree_nodes": tree_nodes} if tree_nodes is not None else {}),
        },
        "execution": canonical_execution,
        "iterations": _positive_int(run_source["iterations"], "root.run.iterations"),
        "checkpoint_every": _positive_int(
            run_source["checkpoint_every"], "root.run.checkpoint_every"
        ),
        "output_dir": str(output_dir),
    }
    canonical_recovery: dict[str, object] = {}
    if minimum_roots != roots:
        canonical_recovery["minimum_roots"] = minimum_roots
    if branched:
        canonical_recovery.update(
            {
                "minimum_branch_scale": branching_internal["minimum_scale"],  # type: ignore[index]
                "branch_backoff_factor": branching_internal["backoff_factor"],  # type: ignore[index]
            }
        )
    if canonical_recovery:
        canonical_run["recovery"] = canonical_recovery
    if run_id != output_dir.name:
        canonical_run["run_id"] = run_id
    if response:
        canonical_run["frozen"] = {"path": str(frozen_path), "sha256": frozen_sha}
    canonical_source = {
        "kind": RUN_CONFIG_KIND,
        "scientific": canonical_scientific,
        "run": canonical_run,
    }
    source_sha256 = _canonical_hash(canonical_source)
    try:
        compiled = compile_capacity(
            mode=schema,
            geometry=geometry,
            model=scientific.common.model,
            precision=str(precision),
            device=device,
            workload=WorkloadSpec(
                roots,
                replicas,
                tree_nodes,
                reserve_bytes,
                microbatch["rollout"],
                microbatch["completion"],
                microbatch["learner"],
                cache_bytes,
            ),
            batch_size=scientific.common.schedule.batch_size,
            beta_max=beta_max,
            completion_samples=completion_samples,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"root.run: capacity compilation failed: {exc}") from exc
    execution = ExecutionConfig(
        device=device_name,
        iterations=canonical_run["iterations"],  # type: ignore[arg-type]
        checkpoint_every=canonical_run["checkpoint_every"],  # type: ignore[arg-type]
        output_dir=output_dir,
        run_id=run_id,
        attention_backend=attention_backend,
        query_chunk_size=(
            compiled.manifest.geometry.model_token_bound
            if attention_backend == "sdpa"
            else query_rows
        ),
        distributed=distributed,
        frozen_checkpoint=frozen_path,
        manifest=compiled.manifest,
        layout=compiled.layout,
        derivations=compiled.derivations,
        source_sha256=source_sha256,
    )
    return RootConfig(scientific, execution, MappingProxyType(canonical_source))


def load_config(path: str | Path) -> RootConfig:
    source = Path(path)
    try:
        raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{source}: unable to load configuration: {exc}") from exc
    return load_config_mapping(_object(raw, "root"))


__all__ = [
    "RESOLVED_EXECUTION_SCHEMA",
    "RUN_CONFIG_KIND",
    "ConfigError",
    "ExecutionConfig",
    "RootConfig",
    "load_config",
    "load_config_mapping",
]
