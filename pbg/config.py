"""Canonical user configuration and deterministic resolved-manifest compiler."""

from __future__ import annotations

import hashlib
import json
import math
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
from pbg.core.execution_manifest import seal_execution_manifest
from pbg.core.learner import (
    BranchingLossConfig,
    EntropyLossConfig,
    LearnerSchedule,
    LossConfig,
    ValueLossConfig,
)
from pbg.core.optimization import OptimizerConfig
from pbg.games.poker.queued.capacity_compiler import (
    CAPACITY_DERIVATION_VERSION,
    WorkloadSpec,
    compile_capacity,
)
from pbg.games.poker.queued.config import (
    BranchedModeConfig,
    PokerGameConfig,
    PokerModelSpec,
    PokerScientificConfig,
    PrecisionConfig,
    ReferenceConfig,
    SequentialModeConfig,
)
from pbg.games.poker.queued.hooks import NLHEGeometry

RUN_CONFIG_KIND = "nlhe_run_v1"
SCIENTIFIC_CONFIG_SCHEMA = "nlhe_scientific_v1"
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
    query_chunk_size: int | None
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
        if self.attention_backend == "sdpa" and self.query_chunk_size is not None:
            raise ConfigError("SDPA execution cannot carry query rows")
        if self.attention_backend == "chunked_dense" and (
            self.query_chunk_size is None or self.query_chunk_size <= 0
        ):
            raise ConfigError("chunked attention requires positive query rows")

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
                        "rollout_workspace_bytes",
                        "queue_drain_workspace_bytes",
                        "completion_workspace_bytes",
                        "align_workspace_bytes",
                        "target_workspace_bytes",
                        "learner_workspace_bytes",
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

        return seal_execution_manifest(self._payload())


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
        """Return the canonical, placeholder-free checkpoint identity."""

        source = self.source_mapping()
        scientific = dict(source["scientific"])
        run = dict(source["run"])
        identity_run: dict[str, object] = {
            name: _plain(run[name])
            for name in ("seed", "reproducibility", "utility", "workload")
        }
        if "recovery" in run:
            identity_run["recovery"] = _plain(run["recovery"])
        if "frozen" in run:
            identity_run["frozen_sha256"] = run["frozen"]["sha256"]
        return {
            "schema": SCIENTIFIC_CONFIG_SCHEMA,
            **_plain(scientific),
            "run": identity_run,
        }

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
) -> tuple[LossConfig, dict[str, object] | None, dict[str, object]]:
    source = _object(value, "root.scientific.objective")
    allowed = {
        "policy_clip",
        "negative_advantage_ratio_cap",
        "value",
        "entropy",
        "reference_kl",
        "branching",
    }
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

    reference_internal: dict[str, object] | None = None
    reference_canonical: dict[str, object] | None = None
    reference_coefficient: float | None = None
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
        values = None

    policy_clip = _number(
        source["policy_clip"], "root.scientific.objective.policy_clip"
    )
    negative_advantage_ratio_cap = (
        _number(
            source["negative_advantage_ratio_cap"],
            "root.scientific.objective.negative_advantage_ratio_cap",
            positive=True,
        )
        if "negative_advantage_ratio_cap" in source
        else None
    )
    if (
        negative_advantage_ratio_cap is not None
        and negative_advantage_ratio_cap <= 1 + policy_clip
    ):
        raise ConfigError(
            "root.scientific.objective.negative_advantage_ratio_cap: "
            "must exceed the PPO upper clip"
        )

    loss = LossConfig(
        policy_clip=policy_clip,
        negative_advantage_ratio_cap=negative_advantage_ratio_cap,
        value=ValueLossConfig(
            kind=str(kind),
            coefficient=_number(
                value_spec["coefficient"],
                "root.scientific.objective.value.coefficient",
            ),
            epsilon=(
                _number(
                    value_spec["epsilon"],
                    "root.scientific.objective.value.epsilon",
                )
                if kind == "clipped"
                else None
            ),
        ),
        entropy=EntropyLossConfig(
            kind=str(entropy_kind),
            coefficient=_number(
                entropy["coefficient"],
                "root.scientific.objective.entropy.coefficient",
            ),
            iterations=(
                _positive_int(
                    entropy["iterations"],
                    "root.scientific.objective.entropy.iterations",
                )
                if entropy_kind == "linear_decay"
                else None
            ),
        ),
        reference_kl_coefficient=reference_coefficient,
        branching=(
            BranchingLossConfig(
                trace_coefficient=_number(
                    values["trace_coefficient"],
                    "root.scientific.objective.branching.trace_coefficient",
                ),
                cost_coefficient=_number(
                    values["cost_coefficient"],
                    "root.scientific.objective.branching.cost_coefficient",
                ),
                trace_epsilon=_number(
                    values["trace_epsilon"],
                    "root.scientific.objective.branching.trace_epsilon",
                    positive=True,
                ),
            )
            if values is not None
            else None
        ),
    )
    canonical: dict[str, object] = {
        "policy_clip": source["policy_clip"],
        "value": dict(value_spec),
        "entropy": dict(entropy),
    }
    if "negative_advantage_ratio_cap" in source:
        canonical["negative_advantage_ratio_cap"] = source[
            "negative_advantage_ratio_cap"
        ]
    if reference_canonical is not None:
        canonical["reference_kl"] = reference_canonical
    if branched:
        assert values is not None
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
    try:
        OptimizerConfig(**internal)
    except (TypeError, ValueError) as exc:
        field = next(
            (name for name in expected if name in str(exc)),
            "configuration",
        )
        raise ConfigError(f"root.scientific.optimizer.{field}: {exc}") from exc
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
        return {"batch_size": batch, "epochs": None, "gradient_steps": count}, {
            "batch_size": batch,
            "steps": count,
        }
    count = _positive_int(source["epochs"], "root.scientific.schedule.epochs")
    return {"batch_size": batch, "epochs": count, "gradient_steps": None}, {
        "batch_size": batch,
        "epochs": count,
    }


def _parse_attention(value: object) -> tuple[str, int | None, object]:
    if value == "sdpa":
        return "sdpa", None, "sdpa"
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
    fixed_beta = _object(source["fixed_beta"], "root.scientific.branching.fixed_beta")
    split_fixed = {"player_pre", "player_post", "chance"}
    shared_fixed = {"player", "chance"}
    expected_fixed = (
        (shared_fixed, split_fixed)
        if rollout_mode == "branched.fixed"
        else (shared_fixed,)
    )
    if set(fixed_beta) not in expected_fixed:
        raise ConfigError(
            "root.scientific.branching.fixed_beta: invalid branch classes for mode"
        )
    for name, item in fixed_beta.items():
        _number(
            item,
            f"root.scientific.branching.fixed_beta.{name}",
            positive=True,
        )
    beta = source["beta_max"]
    if isinstance(beta, Mapping):
        beta_values = dict(beta)
        if set(fixed_beta) != split_fixed or set(beta_values) != split_fixed:
            raise ConfigError(
                "root.scientific.branching.beta_max: per-class bounds require split fixed beta"
            )
        beta_max = max(
            _positive_int(item, f"root.scientific.branching.beta_max.{name}")
            for name, item in beta_values.items()
        )
    else:
        beta_max = _positive_int(beta, "root.scientific.branching.beta_max")
        beta_values = {name: beta_max for name in fixed_beta}
    for name, item in fixed_beta.items():
        if float(item) < 1 or float(item) > int(beta_values[name]):
            raise ConfigError(
                f"root.scientific.branching.fixed_beta.{name}: outside [1,beta_max]"
            )
    minimum_tree = _positive_int(
        source["minimum_tree_size"], "root.scientific.branching.minimum_tree_size"
    )
    if minimum_tree > tree_nodes:
        raise ConfigError(
            "root.scientific.branching.minimum_tree_size: exceeds tree_nodes"
        )
    minimum_scale = _number(
        recovery.get("minimum_branch_scale", 0.25),
        "root.run.recovery.minimum_branch_scale",
        positive=True,
    )
    backoff = _number(
        recovery.get("branch_backoff_factor", 0.5),
        "root.run.recovery.branch_backoff_factor",
        positive=True,
    )
    if minimum_scale > 1 or backoff >= 1:
        raise ConfigError(
            "root.run.recovery: branch recovery factors must lie in (0,1]"
        )
    internal: dict[str, object] = {
        "fixed_beta": fixed_beta,
        "beta_max": beta_values if isinstance(beta, Mapping) else beta_max,
        "minimum_tree_size": minimum_tree,
        "minimum_scale": minimum_scale,
        "backoff_factor": backoff,
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
        warmup = _nonnegative_int(
            kkt["warmup_iterations"],
            "root.scientific.branching.kkt.warmup_iterations",
        )
        target = _number(
            kkt["target_tree_size"],
            "root.scientific.branching.kkt.target_tree_size",
            positive=True,
        )
        if not minimum_tree <= target <= tree_nodes:
            raise ConfigError(
                "root.scientific.branching.kkt.target_tree_size: must lie between minimum_tree_size and tree_nodes"
            )
        gamma = _number(
            kkt["gamma_kappa"],
            "root.scientific.branching.kkt.gamma_kappa",
            positive=True,
        )
        eta = _number(
            kkt["eta_kappa"],
            "root.scientific.branching.kkt.eta_kappa",
            positive=True,
        )
        kappa_min = _number(
            kkt["kappa_min"],
            "root.scientific.branching.kkt.kappa_min",
            positive=True,
        )
        kappa_max = _number(
            kkt["kappa_max"],
            "root.scientific.branching.kkt.kappa_max",
            positive=True,
        )
        log_kappa = _number(
            kkt["log_kappa"],
            "root.scientific.branching.kkt.log_kappa",
        )
        if (
            gamma > 1
            or eta > 1
            or kappa_min > kappa_max
            or not math.log(kappa_min) <= log_kappa <= math.log(kappa_max)
        ):
            raise ConfigError(
                "root.scientific.branching.kkt: controller bounds are invalid"
            )
        validated_kkt = {
            **kkt,
            "warmup_iterations": warmup,
            "target_tree_size": target,
            "gamma_kappa": gamma,
            "eta_kappa": eta,
            "kappa_min": kappa_min,
            "kappa_max": kappa_max,
            "log_kappa": log_kappa,
        }
        internal.update(validated_kkt)
        canonical["kkt"] = validated_kkt
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
    gamma_max = _number(
        safeguard_mapping["gamma_deviance_max"],
        "root.scientific.branching.cost_safeguard.gamma_deviance_max",
    )
    p95_max = _number(
        safeguard_mapping["p95_underprediction_max"],
        "root.scientific.branching.cost_safeguard.p95_underprediction_max",
    )
    consecutive = _positive_int(
        safeguard_mapping["consecutive_iterations"],
        "root.scientific.branching.cost_safeguard.consecutive_iterations",
    )
    if gamma_max < 0 or p95_max < 1:
        raise ConfigError(
            "root.scientific.branching.cost_safeguard: calibration bounds are invalid"
        )
    canonical["cost_safeguard"] = {
        "gamma_deviance_max": gamma_max,
        "p95_underprediction_max": p95_max,
        "consecutive_iterations": consecutive,
    }
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
        if "probe" in scientific_source:
            if not rb or branched:
                raise ConfigError(
                    "root.scientific.probe: requires sequential RB best response"
                )
            probe = _object(scientific_source["probe"], "root.scientific.probe")
            probe_fields = {"target_sha256", "initial_model_sha256"}
            _closed(probe, probe_fields, "root.scientific.probe")
            _required(probe, probe_fields, "root.scientific.probe")
            target_sha = _sha256(
                probe["target_sha256"], "root.scientific.probe.target_sha256"
            )
            _sha256(
                probe["initial_model_sha256"],
                "root.scientific.probe.initial_model_sha256",
            )
            if target_sha != frozen_sha:
                raise ConfigError(
                    "root.scientific.probe.target_sha256: differs from frozen checkpoint"
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

    reference_coefficient = loss.reference_kl_coefficient
    cache_policy = execution_source.get("reference_cache", "off")
    if reference_coefficient is None and cache_policy != "off":
        raise ConfigError("root.run.execution.reference_cache: requires reference_kl")
    if cache_policy == "off":
        cache_bytes = 0
        canonical_cache: object = "off"
    elif cache_policy == "auto":
        total = int(torch.cuda.get_device_properties(device).total_memory)
        cache_bytes = max(1, (total - reserve_bytes) // 20)
        canonical_cache = "auto"
        assert reference is not None
        reference["cache_outputs"] = True
    else:
        cache = _object(cache_policy, "root.run.execution.reference_cache")
        _closed(cache, {"budget_bytes"}, "root.run.execution.reference_cache")
        _required(cache, {"budget_bytes"}, "root.run.execution.reference_cache")
        cache_bytes = _positive_int(
            cache["budget_bytes"], "root.run.execution.reference_cache.budget_bytes"
        )
        canonical_cache = {"budget_bytes": cache_bytes}
        assert reference is not None
        reference["cache_outputs"] = True

    try:
        common = PokerScientificConfig(
            game=PokerGameConfig(**game),
            model=PokerModelSpec(**model),
            loss=loss,
            optimizer=OptimizerConfig(**optimizer),
            schedule=LearnerSchedule(**schedule),
            precision=PrecisionConfig(name=str(precision)),
            reference=(ReferenceConfig(**reference) if reference is not None else None),
            lambda_=scientific_source["lambda"],
            root_count=roots,
            root_count_minimum=minimum_roots,
            seed=run_source["seed"],
            reproducibility=run_source["reproducibility"],
            utility=run_source["utility"],
        )
        applicable: dict[str, object] = {}
        if completion_canonical is not None:
            applicable["completion_samples"] = completion_samples
        if response:
            applicable["frozen_sha256"] = frozen_sha
            applicable["response_initialization"] = scientific_source[
                "response_initialization"
            ]
            if "probe" in scientific_source:
                applicable["probe"] = MappingProxyType(
                    dict(_object(scientific_source["probe"], "root.scientific.probe"))
                )
        if branched:
            assert branching_internal is not None
            frozen_branching = {
                name: (
                    MappingProxyType(dict(value))
                    if isinstance(value, Mapping)
                    else value
                )
                for name, value in branching_internal.items()
            }
            scientific = BranchedModeConfig(
                MappingProxyType(dict(mode)),
                common,
                MappingProxyType(frozen_branching),
                MappingProxyType(dict(branching_canonical["cost_safeguard"])),
                MappingProxyType(applicable),
            )
        else:
            scientific = SequentialModeConfig(
                MappingProxyType(dict(mode)),
                common,
                MappingProxyType(applicable),
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
        query_chunk_size=query_rows,
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


def validate_scientific_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the one versioned scientific product stored in artifacts."""

    source = _object(value, "scientific")
    allowed = {
        "schema",
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
        "run",
    }
    _closed(source, allowed, "scientific")
    _required(
        source,
        {
            "schema",
            "mode",
            "game",
            "model",
            "objective",
            "optimizer",
            "schedule",
            "lambda",
            "run",
        },
        "scientific",
    )
    if source["schema"] != SCIENTIFIC_CONFIG_SCHEMA:
        raise ConfigError("scientific.schema: unsupported scientific schema")
    mode = _object(source["mode"], "scientific.mode")
    try:
        schema = mode_schema(**mode)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"scientific.mode: {exc}") from exc
    game = _object(source["game"], "scientific.game")
    try:
        geometry = NLHEGeometry(
            game["players"],
            game["stack_bb"],
            game["rank_count"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"scientific.game: {exc}") from exc
    _, canonical_model = _parse_model(source["model"], players=geometry.players)
    if canonical_model != source["model"]:
        raise ConfigError("scientific.model: mapping is not canonical")
    _, _, canonical_objective = _parse_objective(
        source["objective"], branched=schema.rollout_mode.is_branched
    )
    if canonical_objective != source["objective"]:
        raise ConfigError("scientific.objective: mapping is not canonical")
    _, canonical_optimizer = _parse_optimizer(source["optimizer"])
    if canonical_optimizer != source["optimizer"]:
        raise ConfigError("scientific.optimizer: mapping is not canonical")
    _, canonical_schedule = _parse_schedule(source["schedule"])
    if canonical_schedule != source["schedule"]:
        raise ConfigError("scientific.schedule: mapping is not canonical")
    lambda_value = _number(source["lambda"], "scientific.lambda")
    if not 0 <= lambda_value <= 1:
        raise ConfigError("scientific.lambda: expected a value in [0,1]")
    rb = schema.private_estimator.value == "rao_blackwell"
    if rb:
        completion = _object(source.get("completion"), "scientific.completion")
        _closed(completion, {"samples"}, "scientific.completion")
        _required(completion, {"samples"}, "scientific.completion")
        _positive_int(completion["samples"], "scientific.completion.samples")
    elif "completion" in source:
        raise ConfigError("scientific.completion: inapplicable to sampled mode")
    response = schema.training_regime.value == "best_response"
    if response != ("response_initialization" in source):
        raise ConfigError(
            "scientific.response_initialization: mode applicability mismatch"
        )
    run = _object(source["run"], "scientific.run")
    run_allowed = {
        "seed",
        "reproducibility",
        "utility",
        "workload",
        "recovery",
        "frozen_sha256",
    }
    _closed(run, run_allowed, "scientific.run")
    _required(run, {"seed", "reproducibility", "utility", "workload"}, "scientific.run")
    if isinstance(run["seed"], bool) or not isinstance(run["seed"], int):
        raise ConfigError("scientific.run.seed: expected integer")
    if run["reproducibility"] not in ("strict", "seeded", "throughput"):
        raise ConfigError("scientific.run.reproducibility: invalid policy")
    if run["utility"] not in ("raw_bb", "stack_normalized"):
        raise ConfigError("scientific.run.utility: invalid utility")
    workload = _object(run["workload"], "scientific.run.workload")
    _closed(workload, {"roots", "replicas", "tree_nodes"}, "scientific.run.workload")
    _required(workload, {"roots"}, "scientific.run.workload")
    _positive_int(workload["roots"], "scientific.run.workload.roots")
    if "replicas" in workload:
        _positive_int(workload["replicas"], "scientific.run.workload.replicas")
    tree_nodes = workload.get("tree_nodes")
    if schema.rollout_mode.is_branched:
        tree_count = _positive_int(tree_nodes, "scientific.run.workload.tree_nodes")
    elif tree_nodes is not None:
        raise ConfigError("scientific.run.workload.tree_nodes: inapplicable")
    else:
        tree_count = 0
    recovery = _object(run.get("recovery", {}), "scientific.run.recovery")
    recovery_allowed = {"minimum_roots"}
    if schema.rollout_mode.is_branched:
        recovery_allowed |= {"minimum_branch_scale", "branch_backoff_factor"}
    _closed(recovery, recovery_allowed, "scientific.run.recovery")
    if "minimum_roots" in recovery:
        minimum_roots = _positive_int(
            recovery["minimum_roots"], "scientific.run.recovery.minimum_roots"
        )
        if minimum_roots > int(workload["roots"]):
            raise ConfigError("scientific.run.recovery.minimum_roots: exceeds roots")
    if schema.rollout_mode.is_branched:
        _, canonical_branching, _ = _parse_branching(
            source.get("branching"),
            rollout_mode=schema.rollout_mode.value,
            tree_nodes=tree_count,
            recovery=recovery,
        )
        if canonical_branching != source["branching"]:
            raise ConfigError("scientific.branching: mapping is not canonical")
    elif "branching" in source:
        raise ConfigError("scientific.branching: mode applicability mismatch")
    if response != ("frozen_sha256" in run):
        raise ConfigError("scientific.run.frozen_sha256: mode applicability mismatch")
    if "frozen_sha256" in run:
        frozen_sha = _sha256(run["frozen_sha256"], "scientific.run.frozen_sha256")
        if source["response_initialization"] not in ("random", "target"):
            raise ConfigError("scientific.response_initialization: invalid policy")
        if "probe" in source:
            if not rb or schema.rollout_mode.is_branched:
                raise ConfigError("scientific.probe: requires sequential RB")
            probe = _object(source["probe"], "scientific.probe")
            probe_fields = {"target_sha256", "initial_model_sha256"}
            _closed(probe, probe_fields, "scientific.probe")
            _required(probe, probe_fields, "scientific.probe")
            if (
                _sha256(probe["target_sha256"], "scientific.probe.target_sha256")
                != frozen_sha
            ):
                raise ConfigError("scientific.probe.target_sha256: differs from frozen")
            _sha256(
                probe["initial_model_sha256"],
                "scientific.probe.initial_model_sha256",
            )
    elif "probe" in source:
        raise ConfigError("scientific.probe: requires best response")
    return _plain(source)  # type: ignore[return-value]


__all__ = [
    "RESOLVED_EXECUTION_SCHEMA",
    "RUN_CONFIG_KIND",
    "SCIENTIFIC_CONFIG_SCHEMA",
    "ConfigError",
    "ExecutionConfig",
    "RootConfig",
    "load_config",
    "load_config_mapping",
    "validate_scientific_mapping",
]
