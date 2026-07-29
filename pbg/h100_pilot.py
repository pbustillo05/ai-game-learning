"""Reproducible one-H100 pilot planning and model-only initialization."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from pbg.config import RootConfig, load_config_mapping
from pbg.core.persistence import model_state_sha256, sha256_file
from pbg.core.targets import (
    rb_conversion_library_workspace_bytes,
    rb_conversion_workspace_row_bytes,
)
from pbg.games.poker.model.capacity import (
    max_policy_partition_sequence_tokens,
)
from pbg.games.poker.queued.folded import (
    sequential_family_workspace_row_bytes,
)
from pbg.games.poker.queued.hooks import NLHEGeometry

INITIALIZER_SCHEMA = "pbg-h100-pilot-initializer-v1"
PLAN_SCHEMA = "pbg-h100-pilot-plan-v1"
RESULT_SCHEMA = "pbg-h100-pilot-result-v1"
GIB = 1 << 30
H100_DECLARED_TOTAL_BYTES = 80_000_000_000
H100_MEMORY_BUDGET_BYTES = 70 * GIB
H100_MEMORY_GUARD_BYTES = 10 * GIB
REPLICA_COUNTS = (1, 2, 4, 8)
RB_SAMPLE_COUNTS = (256, 512, 1024)


class PilotError(RuntimeError):
    """The H100 pilot cannot preserve its declared experiment contract."""


@dataclass(frozen=True)
class Arm:
    name: str
    stage: str
    mode: str
    replicas: int
    roots: int
    completion_samples: int | None
    family_pack: int = 84
    warmup: int = 1
    repeats: int = 2

    def __post_init__(self) -> None:
        if self.mode not in ("normal", "rb"):
            raise ValueError("pilot mode must be normal or rb")
        if self.replicas not in REPLICA_COUNTS:
            raise ValueError("pilot replicas must be one of 1, 2, 4, 8")
        if self.roots <= 0 or self.family_pack <= 0:
            raise ValueError("pilot roots and family pack must be positive")
        if self.mode == "rb" and (
            self.completion_samples is None or self.completion_samples <= 0
        ):
            raise ValueError("RB pilot arms require completion samples")
        if self.mode == "normal" and self.completion_samples is not None:
            raise ValueError("normal pilot arms cannot declare completion samples")
        if self.warmup < 0 or self.repeats <= 0:
            raise ValueError("pilot warmup/repeats are invalid")


def _atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tensor_state(value: object, *, source: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise PilotError(f"{source} has no nonempty model tensor state")
    state = dict(value)
    if any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise PilotError(f"{source} model state contains a non-tensor entry")
    return {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in state.items()
    }


def export_model_initializer(
    checkpoint: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    """Extract a portable, hash-bound model state from a trusted checkpoint."""

    source = Path(checkpoint)
    target = Path(destination)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise PilotError("initializer source is not a checkpoint mapping")
    state = _tensor_state(payload.get("model"), source=str(source))
    if "lanes" in state:
        raise PilotError("replica checkpoint model state is not a single initializer")
    report = {
        "schema": INITIALIZER_SCHEMA,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "model_sha256": model_state_sha256(state),
        "parameter_tensors": len(state),
    }
    _atomic_torch_save({**report, "model": state}, target)
    return {**report, "path": str(target), "artifact_sha256": sha256_file(target)}


def load_model_initializer(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Load and verify one model-only pilot initializer artifact."""

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "model_sha256",
        "parameter_tensors",
        "model",
    }:
        raise PilotError("initializer artifact has unknown or missing fields")
    if payload["schema"] != INITIALIZER_SCHEMA:
        raise PilotError("initializer artifact schema is unsupported")
    state = _tensor_state(payload["model"], source=str(source))
    if len(state) != payload["parameter_tensors"]:
        raise PilotError("initializer tensor count differs from its metadata")
    digest = model_state_sha256(state)
    if digest != payload["model_sha256"]:
        raise PilotError("initializer model hash differs from its metadata")
    metadata = {name: value for name, value in payload.items() if name != "model"}
    metadata["artifact_sha256"] = sha256_file(source)
    return state, metadata


def _scientific_mapping(
    *,
    mode: str,
    roots: int,
    completion_samples: int | None,
    precision: str,
    batch_size: int,
    gradient_steps: int,
) -> dict[str, object]:
    scientific: dict[str, object] = {
        "mode": {
            "training_regime": "self_play",
            "rollout_mode": "sequential",
            "private_estimator": (
                "sampled" if mode == "normal" else "rao_blackwell"
            ),
        },
        "game": {"players": 2, "stack_bb": 32.0, "rank_count": 13},
        "model": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers_pre": 4,
            "n_layers_post": 6,
            "m_raise": 2,
            "s_min": 0.02,
            "max_players": 2,
            "class_embedding_dim": 32,
            "ffn_hidden": None,
            "folder_beta_rank": 0,
            "folder_line_dim": 8,
            "epsilon": 1.0e-8,
        },
        "loss": {
            "clip_epsilon": 0.2,
            "value_clip_epsilon": 0.2,
            "value_coefficient": 0.125,
            "entropy_coefficient": 0.025,
            "reference_kl_coefficient": 0.05,
            "value_loss": "clipped",
            "entropy_schedule": "linear_decay",
            "entropy_decay_iterations": 500,
            "trace_coefficient": 0.0,
            "cost_coefficient": 0.0,
            "trace_epsilon": 1.0e-8,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 5.0e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "grad_clip": 0.5,
        },
        "schedule": {
            "batch_size": batch_size,
            "epochs": 1,
            "gradient_steps": gradient_steps,
        },
        "precision": {"name": precision},
        "reference": {
            "interval_iterations": None,
            "interval_effective_hands": (
                900_000.0 if mode == "normal" else 172_800_000.0
            ),
            "cache_outputs": False,
        },
        "lambda": 0.95,
        "root_count": roots,
        "root_count_minimum": roots,
        "seed": 104830,
        "reproducibility": "throughput",
        "utility": "stack_normalized",
    }
    if mode == "rb":
        scientific["completion"] = {
            "n_samples": completion_samples,
            "strata": "default",
        }
    return scientific


def pilot_config_mapping(
    arm: Arm,
    *,
    output_dir: str | Path,
    device: str = "cuda:0",
    hardware: str = "h100",
    local_total_memory_bytes: int | None = None,
) -> dict[str, object]:
    """Build one strict config for either the H100 pilot or local CUDA smoke."""

    if hardware not in ("h100", "local"):
        raise ValueError("hardware profile must be h100 or local")
    h100 = hardware == "h100"
    precision = "bf16" if h100 else "fp32"
    batch_size = 256 if h100 else 8
    gradient_steps = 116 if h100 else 1
    geometry = NLHEGeometry(2, 32.0, 13)
    token_bounds = max_policy_partition_sequence_tokens(
        n_players=2,
        stack_bb=32.0,
        rank_count=13,
        k_act=5,
    )
    token_bound = max(token_bounds.values())
    pair_bound = token_bound * token_bound
    pre_rows = 8 * arm.roots
    post_rows = 32 * arm.roots
    leaf_cap = 2 * arm.roots
    topology_cap = 42 * arm.roots
    topology_edge_cap = topology_cap - arm.roots
    # Folded ranges are retained for the complete rollout, so this is a
    # cumulative per-attempt capacity rather than a scheduler-pack bound.
    folded_rows = arm.roots
    learner_pack = min(arm.family_pack, batch_size)
    completion_pack = min(256 if h100 else 8, leaf_cap)
    chance_pack = min(max(2 * arm.family_pack, 8), pre_rows + post_rows + leaf_cap)
    topology_packet = min(
        max(4 * arm.family_pack, 8),
        topology_cap + topology_edge_cap,
    )
    if h100:
        total_memory = H100_DECLARED_TOTAL_BYTES
        memory_budget = H100_MEMORY_BUDGET_BYTES
        memory_guard = H100_MEMORY_GUARD_BYTES
        movement_scratch = 256 << 20
        evaluator_workspace = 512 << 20
        learner_workspace = 32 << 20
    else:
        if local_total_memory_bytes is None or local_total_memory_bytes <= 0:
            raise ValueError("local profile requires positive device memory")
        total_memory = local_total_memory_bytes
        memory_budget = min(9 * GIB, total_memory - 512 * 1024**2)
        if memory_budget <= 0:
            raise ValueError("local CUDA device is too small for the smoke profile")
        memory_guard = 256 << 20
        movement_scratch = 16 << 20
        evaluator_workspace = 64 << 20
        learner_workspace = 4 << 20
    rb = arm.mode == "rb"
    library_workspace = (
        rb_conversion_library_workspace_bytes(
            geometry.private_hands, geometry.rank_count**2
        )
        if rb
        else 8 << 20
    )
    action_cap = {"pre": pre_rows, "post_2": post_rows}
    run_id = arm.name
    result: dict[str, object] = {
        "scientific": _scientific_mapping(
            mode=arm.mode,
            roots=arm.roots,
            completion_samples=arm.completion_samples,
            precision=precision,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
        ),
        "execution": {
            "device": device,
            "iterations": 1,
            "checkpoint_every": 1,
            "output_dir": str(output_dir),
            "run_id": run_id,
            "distributed": False,
            "frozen_checkpoint": None,
            "attention_backend": "sdpa",
            "query_chunk_size": 128,
            "manifest": {
                "model_token_bound": token_bound,
                "model_token_pair_bound": pair_bound,
                "root_count": arm.roots,
                "n_replica": arm.replicas,
                "beta_max": 1,
                "player_node_cap": {"pre": pre_rows, "post_2": post_rows},
                "trained_player_cap": {"pre": pre_rows, "post_2": post_rows},
                "action_cap": action_cap,
                "folded_range_cap": {"pre": folded_rows, "post": folded_rows},
                "leaf_cap": leaf_cap,
                "topology_cap": topology_cap,
                "topology_edge_cap": topology_edge_cap,
                "family_pack_cap": min(arm.family_pack, pre_rows + post_rows),
                "chance_frontier_cap": chance_pack,
                "completion_pack_cap": completion_pack,
                "learner_sigma_pack_cap": learner_pack,
                "learner_action_pack_cap": learner_pack,
                "topology_packet_cap": topology_packet,
                "target_arena_cap": leaf_cap + topology_cap,
                "movement_scratch_bytes": movement_scratch,
                "library_workspace_bytes": library_workspace,
                "evaluator_workspace_bytes": evaluator_workspace,
                "memory_guard_bytes": memory_guard,
                "hardware": {
                    "total_memory_bytes": total_memory,
                    "memory_budget_bytes": memory_budget,
                    "training_max_tokens": learner_pack * token_bound,
                    "training_max_token_pairs": learner_pack * pair_bound,
                    "query_chunk_size": 128,
                },
                "required_external": {
                    "library_bytes": 0,
                    "evaluator_bytes": 0,
                },
            },
            "layout": {
                "model_input_shapes": {
                    "pre": [geometry.direct_model_input_width],
                    "post_2": [geometry.direct_model_input_width],
                },
                "player_count": 2,
                "stratum_count": 2,
                "evaluator_private_width": geometry.private_hands,
                "family_workspace_row_bytes": (
                    sequential_family_workspace_row_bytes(geometry, 2)
                ),
                "chance_frontier_row_bytes": 128 << 10,
                "completion_workspace_row_bytes": (
                    rb_conversion_workspace_row_bytes(
                        geometry.private_hands, geometry.rank_count**2
                    )
                    if rb
                    else 4096
                ),
                "learner_sigma_workspace_row_bytes": learner_workspace,
                "learner_action_workspace_row_bytes": learner_workspace,
                "topology_packet_row_bytes": 128,
                "reference_cache_bytes": 0,
                "field_alignment": 64,
                "model_dtype": "bfloat16" if h100 else "float32",
            },
        },
    }
    load_config_mapping(result)
    return result


def write_arm_config(
    arm: Arm,
    destination: str | Path,
    *,
    output_dir: str | Path,
    hardware: str = "h100",
    local_total_memory_bytes: int | None = None,
) -> RootConfig:
    target = Path(destination)
    mapping = pilot_config_mapping(
        arm,
        output_dir=output_dir,
        hardware=hardware,
        local_total_memory_bytes=local_total_memory_bytes,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(mapping, sort_keys=False),
        encoding="utf-8",
    )
    return load_config_mapping(mapping)


def geometry_roots(replicas: int) -> tuple[int, ...]:
    return {
        1: (2048, 3072, 4096),
        2: (1536, 2048, 3072),
        4: (768, 1024, 1536),
        8: (384, 512, 768),
    }[replicas]


def base_plan() -> dict[str, object]:
    arms: list[Arm] = [
        Arm("nlhe_hu_32bb_normal_h100_bf16_smoke", "smoke", "normal", 1, 4, None, warmup=0, repeats=1),
        Arm("nlhe_hu_32bb_rb_h100_bf16_smoke", "smoke", "rb", 1, 4, 16, warmup=0, repeats=1),
    ]
    arms.extend(
        Arm(
            f"nlhe_hu_32bb_{mode}_h100_replica_k{replicas}",
            "replica_scaling",
            mode,
            replicas,
            512,
            1024 if mode == "rb" else None,
        )
        for mode in ("normal", "rb")
        for replicas in REPLICA_COUNTS
    )
    return {
        "schema": PLAN_SCHEMA,
        "deadline_minutes": 170,
        "shutdown_reserve_minutes": 10,
        "replica_gain_threshold": 0.10,
        "memory_ceiling_bytes": H100_MEMORY_BUDGET_BYTES,
        "fixed_batch_size": 256,
        "fixed_gradient_steps": 116,
        "production_precision": "bf16",
        "replica_counts": list(REPLICA_COUNTS),
        "rb_sample_counts": list(RB_SAMPLE_COUNTS),
        "arms": [asdict(arm) for arm in arms],
    }


def select_replica_count(
    reports: Sequence[Mapping[str, object]],
    *,
    minimum_gain: float = 0.10,
    memory_ceiling_bytes: int = H100_MEMORY_BUDGET_BYTES,
) -> int:
    """Select the smallest K unless a larger K gains at least the threshold."""

    rows: dict[int, tuple[float, int]] = {}
    for report in reports:
        replicas = int(report["replicas"])
        rate = float(report["player_nodes_per_second"])
        peak = int(report["peak_reserved_bytes"])
        if rate > 0 and peak <= memory_ceiling_bytes:
            rows[replicas] = (rate, peak)
    if 1 not in rows:
        raise PilotError("replica selection has no successful K=1 baseline")
    selected = 1
    selected_rate = rows[1][0]
    for replicas in REPLICA_COUNTS[1:]:
        candidate = rows.get(replicas)
        if candidate is not None and candidate[0] >= selected_rate * (
            1.0 + minimum_gain
        ):
            selected = replicas
            selected_rate = candidate[0]
    return selected


def h100_preflight() -> dict[str, object]:
    """Require one visible BF16-capable H100 before spending pilot time."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PilotError("pilot requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if (
        "H100" not in properties.name.upper()
        or properties.major < 9
        or properties.total_memory < H100_DECLARED_TOTAL_BYTES
        or not torch.cuda.is_bf16_supported()
    ):
        raise PilotError(
            "pilot requires an H100 with compute capability >= 9.0, "
            "at least 80,000,000,000 bytes, and BF16 support"
        )
    return {
        "device": str(device),
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "bf16_supported": True,
    }


def benchmark_summary(
    arm: Arm,
    benchmark: Mapping[str, object],
) -> dict[str, object]:
    median = benchmark.get("median")
    samples = benchmark.get("samples")
    scientific = benchmark.get("scientific_config")
    if (
        not isinstance(median, Mapping)
        or not isinstance(samples, Sequence)
        or len(samples) != arm.repeats
        or not isinstance(scientific, Mapping)
        or not isinstance(scientific.get("schedule"), Mapping)
    ):
        raise PilotError("benchmark report is incomplete")
    gradient_steps = int(scientific["schedule"]["gradient_steps"])
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise PilotError("benchmark sample is not an object")
        before = sample.get("logical_work_before")
        after = sample.get("logical_work_after")
        if (
            not isinstance(before, Sequence)
            or not isinstance(after, Sequence)
            or len(before) != arm.replicas
            or len(after) != arm.replicas
            or int(sample.get("player_nodes", 0)) <= 0
        ):
            raise PilotError("benchmark sample lost replica/work identity")
        for lane_before, lane_after in zip(before, after, strict=True):
            if (
                not isinstance(lane_before, Mapping)
                or not isinstance(lane_after, Mapping)
                or any(int(value) != 0 for value in lane_before.values())
                or int(lane_after["trajectories"]) != arm.roots
                or int(lane_after["optimizer_steps"]) != gradient_steps
                or (
                    arm.mode == "rb"
                    and int(lane_after["completion_samples"]) <= 0
                )
            ):
                raise PilotError(
                    "benchmark lane did not complete its declared fresh work"
                )
    for name in (
        "player_nodes_per_second",
        "seconds",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        if not math.isfinite(float(median[name])) or float(median[name]) < 0:
            raise PilotError(f"benchmark median {name} is invalid")
    return {
        "schema": RESULT_SCHEMA,
        "arm": arm.name,
        "stage": arm.stage,
        "mode": arm.mode,
        "replicas": arm.replicas,
        "roots_per_replica": arm.roots,
        "completion_samples": arm.completion_samples,
        "player_nodes_per_second": float(median["player_nodes_per_second"]),
        "seconds": float(median["seconds"]),
        "peak_allocated_bytes": int(median["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(median["peak_reserved_bytes"]),
        "initializer_model_sha256": benchmark["identity"][
            "initializer_model_sha256"
        ],
    }


def write_json(value: object, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


__all__ = [
    "H100_MEMORY_BUDGET_BYTES",
    "INITIALIZER_SCHEMA",
    "PLAN_SCHEMA",
    "RB_SAMPLE_COUNTS",
    "REPLICA_COUNTS",
    "Arm",
    "PilotError",
    "base_plan",
    "benchmark_summary",
    "export_model_initializer",
    "geometry_roots",
    "h100_preflight",
    "load_model_initializer",
    "pilot_config_mapping",
    "select_replica_count",
    "write_arm_config",
    "write_json",
]
