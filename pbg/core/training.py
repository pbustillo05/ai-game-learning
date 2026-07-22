"""Minimal rollout/target/optimization training loop with DDP checkpoints."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import platform
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from pbg.core.br import SHARED_RESPONSE_SEMANTICS
from pbg.core.config import RootConfig, seed_all
from pbg.core.adaptive_branching import AdaptiveKKTController
from pbg.core.checkpointing import (
    _capture_rng_state,
    config_dict_fingerprint,
    _config_fingerprint,
    compatible_scientific_configs,
    legacy_config_fingerprint,
    matches_legacy_run_fingerprint,
    inspect_checkpoint,
    _scientific_config,
    _validate_metrics_prefix,
    load_checkpoint,
    save_checkpoint,
)
from pbg.core.game import GameBundle
from pbg.core.distributed_failure import (
    DistributedFailureAgreement,
    DistributedPhaseFailure,
)
from pbg.core.learner import (
    allocate_fixed_statistical_counts,
    DistributedFixedRolloutUpdate,
    DistributedLearnerConfig,
    PhysicalShape,
)
from pbg.core.memory_control import (
    MEMORY_PHASES,
    MemoryAdmissionStop,
    MemoryLedger,
    PhaseMemoryForecaster,
    PhaseMemoryModel,
    PhaseMemoryObservation,
    SequentialMemoryController,
)
from pbg.core.storage import (
    PartitionRolloutBlock,
    RolloutStorage,
    SamplingMetadata,
    SealedDecisionInputs,
    SealedRolloutSample,
    SealedTensorBlock,
    SequentialPosteriorPool,
    TensorFieldSchema,
    TensorSchema,
    TrainingPartition,
    TrainingStorage,
    paged_capacity_bytes,
    rollout_action_source_schema,
    rollout_action_schema,
    rollout_decision_schema,
    rollout_leaf_schema,
    rollout_policy_source_schema,
    sequential_posterior_schema,
    stopped_leaf_source_schema,
    target_action_schema,
    target_decision_schema,
    topology_schema,
)
from pbg.core.tree import PagedRolloutTopology
from pbg.core.targets import (
    _walker_alias,
    build_branched_rb_targets,
    build_normal_targets,
    build_sequential_rb_targets,
)
from pbg.core.topology import BranchedTreeSpan, SequentialDecisionCSR
from pbg.core.partition import CompletionSpec
from pbg.core.rollout import (
    _logical_seed,
    ProbeRollout,
    sample_trees,
    sample_trees_sequential,
    sequential_leaf_logical_rng_keys,
)
from pbg.core.execution import LOGICAL_RNG_VERSION, DrawPurpose, logical_rng_keys
from pbg.core.completion import BatchedCompletionStats
from pbg.core.utility import utility_conversion, utility_scale
from pbg.core.devices import (
    assert_module_device,
    resolve_execution_device,
    validate_workload_capabilities,
)


def _configure_reproducibility(profile: str) -> None:
    """Apply one explicit backend reproducibility/performance profile."""

    strict = profile == "strict"
    throughput = profile == "throughput"
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = throughput
    torch.backends.cudnn.allow_tf32 = throughput
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = throughput
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if throughput else "highest")


def _scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler(
            "cuda", enabled=enabled, init_scale=1024.0, growth_interval=2000
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled, init_scale=1024.0, growth_interval=2000
        )


def _scale_completion_utility(
    completion: BatchedCompletionStats, cfg: RootConfig
) -> BatchedCompletionStats:
    """Convert completed conditional values at the sole target boundary."""

    scale = utility_scale(cfg)
    if scale == 1.0:
        return completion
    return BatchedCompletionStats(
        p_hat=completion.p_hat,
        v_hat=completion.v_hat * scale,
        metadata={**completion.metadata, "training_utility_scale": scale},
    )


def _migrate_value_head_state(
    state: Mapping[str, Any], source_cfg: Any, target_cfg: Any
) -> tuple[dict[str, Any], float, tuple[str, ...]]:
    """Convert only final value affines between checkpoint utility units."""

    factor = utility_conversion(source_cfg, target_cfg)
    copied = dict(state)
    if factor == 1.0:
        return copied, factor, ()
    suffixes = ("value_head.affine.weight", "value_head.affine.bias")
    converted = tuple(sorted(name for name in copied if name.endswith(suffixes)))
    if len(converted) != 4:
        raise RuntimeError(
            "utility migration requires preflop/postflop value-head weights and biases"
        )
    for name in converted:
        value = copied[name]
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise RuntimeError(f"value-head checkpoint tensor {name!r} is invalid")
        copied[name] = value * factor
    return copied, factor, converted


def _precision(cfg: RootConfig, device: torch.device) -> tuple[bool, torch.dtype]:
    name = "fp16" if cfg.hardware is None else cfg.hardware.precision
    if device.type != "cuda" or name == "fp32" or not cfg.training.amp:
        return False, torch.float32
    return True, torch.bfloat16 if name == "bf16" else torch.float16


def _hardware_metadata(
    cfg: RootConfig, device: torch.device, world: int
) -> dict[str, Any]:
    """Validate the selected device and return reproducibility metadata."""

    profile = cfg.hardware
    metadata: dict[str, Any] = {
        "hardware_profile": None if profile is None else profile.name,
        "device": str(device),
        "world_size": world,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_toolkit": os.environ.get("CUDA_HOME"),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "scheduler_time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "reproducibility": cfg.training.reproducibility,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "logical_rng": LOGICAL_RNG_VERSION,
        "scheduler_control_sync_policy": "one_fixed_control_block_per_decision_v1",
        "execution_device": cfg.device,
        "execution_calibration": {
            "batch_rows": cfg.rollout.family_batch_rows,
            "max_active_trajectories": cfg.rollout.max_active_trajectories,
            "storage_page_rows": cfg.rollout.storage_page_rows,
        },
        "memory_limits": {
            "execution_memory_bytes": cfg.rollout.execution_memory_bytes,
            "reserve_vram_gib": None if cfg.hardware is None else cfg.hardware.reserve_vram_gib,
            "max_memory_fraction": None if cfg.hardware is None else cfg.hardware.max_memory_fraction,
        },
    }
    try:
        evaluator = importlib.import_module("poker_gpu_evaluator")
        metadata["evaluator_contract_version"] = evaluator.CONTRACT_VERSION
        metadata["evaluator_module"] = str(Path(evaluator.__file__).resolve())
    except (ImportError, AttributeError):
        metadata["evaluator_contract_version"] = None
        metadata["evaluator_module"] = None
    if cfg.training.mode == "branched_rao_blackwell":
        metadata["rank_start_quotas"] = [1] * world
        metadata["sequential_roots_per_gpu"] = None
    else:
        metadata["rank_start_quotas"] = [cfg.rollout.roots_per_gpu] * world
        metadata["sequential_roots_per_gpu"] = cfg.rollout.roots_per_gpu
    if profile is None:
        return metadata
    if world > profile.max_devices:
        raise RuntimeError(
            f"hardware profile {profile.name!r} allows at most {profile.max_devices} devices"
        )
    if device.type != "cuda":
        metadata["attention_backend"] = "mps"
        return metadata
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = 1024**3
    capability = (props.major, props.minor)
    if (
        profile.expected_compute_capability is not None
        and capability != profile.expected_compute_capability
    ):
        raise RuntimeError(
            f"hardware profile {profile.name!r} expects compute capability "
            f"{profile.expected_compute_capability}, found {capability}"
        )
    total_gib = total_bytes / gib
    if profile.min_vram_gib is not None and total_gib < profile.min_vram_gib:
        raise RuntimeError("visible GPU has less VRAM than the hardware profile requires")
    if profile.max_vram_gib is not None and total_gib > profile.max_vram_gib:
        raise RuntimeError("visible GPU has more VRAM than the selected validated tier")
    reserve = int(profile.reserve_vram_gib * gib)
    usable = min(free_bytes - reserve, int(total_bytes * profile.max_memory_fraction))
    if usable <= 0:
        raise MemoryError(
            f"hardware profile {profile.name!r} cannot preserve its "
            f"{profile.reserve_vram_gib:.1f} GiB reserve on {device}"
        )
    metadata.update(
        {
            "gpu_name": props.name,
            "compute_capability": capability,
            "total_vram_bytes": total_bytes,
            "free_vram_bytes": free_bytes,
            "usable_vram_bytes": usable,
            "attention_backend": profile.attention_backend,
            "query_chunk_size": profile.query_chunk_size,
            "training_max_tokens": profile.training_max_tokens,
            "training_max_token_pairs": profile.training_max_token_pairs,
            "inference_max_tokens": profile.inference_max_tokens,
            "inference_max_token_pairs": profile.inference_max_token_pairs,
        }
    )
    return metadata


def _rank_local_config(cfg: RootConfig, rank: int, world: int) -> RootConfig:
    """Return the common per-GPU width with a collision-free rank address."""

    local = copy.deepcopy(cfg)
    if cfg.training.mode == "branched_rao_blackwell":
        local.rollout.roots_per_gpu = 1
        local.rollout.trajectory_rank_counts = tuple(1 for _ in range(world))
        return local
    quotas = tuple(cfg.rollout.roots_per_gpu for _ in range(world))
    local.rollout.roots_per_gpu = cfg.rollout.roots_per_gpu
    # These attributes are execution metadata, not accepted YAML keys.
    local.rollout.trajectory_rank_counts = quotas
    return local




def _add_distributed_metric_summaries(
    metrics: dict[str, float], device: torch.device, world: int
) -> None:
    """Attach cross-rank means and maxima for launch-critical diagnostics."""

    if world <= 1:
        return
    names = (
        "nodes",
        "leaves",
        "decisions",
        "decision_nodes_preflop",
        "decision_nodes_flop",
        "decision_nodes_turn",
        "decision_nodes_river",
        "loss",
        "policy",
        "value",
        "entropy",
        "reference_kl",
        "grad_norm_pre_clip",
        "grad_clip_scale",
        "completion_mean_estimator_stddev",
        "completion_max_estimator_stddev",
        "trajectories",
        "completion_samples",
        "minibatches",
        "optimizer_steps",
        "safeguard_batches",
        "safeguard_suppressed_children",
        "root_refill_count",
        "root_refill_added_nodes",
        "counterfactual_tree_size",
    )
    present = torch.tensor(
        [name in metrics for name in names], dtype=torch.float64, device=device
    )
    local = torch.tensor(
        [metrics.get(name, 0.0) for name in names],
        dtype=torch.float64,
        device=device,
    )
    summed = torch.cat((local, present))
    maximum = torch.where(
        present.bool(), local, torch.full_like(local, -torch.inf)
    )
    summary_sync_start = time.perf_counter()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    metrics["diagnostic_metric_summary_sync_seconds"] = (
        time.perf_counter() - summary_sync_start
    )
    metrics["diagnostic_metric_summary_sync_bytes"] = float(
        (summed.numel() + maximum.numel()) * summed.element_size()
    )
    total, count = summed.chunk(2)
    for index, name in enumerate(names):
        if not bool(count[index]):
            continue
        metrics[f"{name}_rank_mean"] = float(total[index] / count[index])
        metrics[f"{name}_rank_max"] = float(maximum[index])
        metrics[f"{name}_rank_sum"] = float(total[index])


def _synchronize_memory_control_metrics(
    metrics: dict[str, Any], device: torch.device, world: int, n_players: int
) -> None:
    """Replace controller inputs with rank maxima before online fitting."""

    if world <= 1:
        return
    keys = [
        "nodes",
        "leaves",
        "decisions",
        "retained_target_bytes",
        "iteration_peak_allocated_bytes",
        "iteration_peak_reserved_bytes",
        "iteration_end_allocated_bytes",
        "iteration_end_reserved_bytes",
        "memory_iteration_start_allocated_bytes",
        "memory_iteration_start_reserved_bytes",
        "memory_iteration_start_peak_allocated_bytes",
        "memory_iteration_start_peak_reserved_bytes",
        "memory_preoptimization_allocated_bytes",
        "memory_preoptimization_reserved_bytes",
        "memory_preoptimization_peak_allocated_bytes",
        "memory_preoptimization_peak_reserved_bytes",
        "execution_padded_tokens",
        "execution_padded_token_pairs",
        "execution_padded_tokens_max",
        "execution_padded_token_pairs_max",
        "execution_queue_depth_max",
        "execution_queue_oldest_age_max",
        "execution_forced_drain_rows",
        "execution_forced_drains",
        "execution_model_calls",
        "execution_completion_batches",
        "execution_completion_rows",
        "trajectory_length_mean",
        "trajectory_length_max",
        "trajectory_length_p50",
        "trajectory_length_p90",
        "trajectory_length_p99",
        "trajectory_length_le_4",
        "trajectory_length_le_8",
        "trajectory_length_le_16",
        "trajectory_length_le_32",
        "trajectory_length_le_64",
    ]
    for phase in ("snapshot", "rollout", "completion", "backward", "seal", "optimization"):
        keys.extend(
            (
                f"phase_{phase}_peak_allocated_bytes",
                f"phase_{phase}_peak_reserved_bytes",
            )
        )
    for phase in ("inference", "training"):
        for family in ("pre", *(f"post_{live}" for live in range(2, n_players + 1))):
            base = f"execution_{phase}_{family}"
            keys.extend(
                (
                    f"{base}_model_calls",
                    f"{base}_model_rows",
                    f"{base}_model_rows_max",
                    f"{base}_padded_tokens",
                    f"{base}_padded_tokens_max",
                    f"{base}_padded_token_pairs",
                    f"{base}_padded_token_pairs_max",
                )
            )
    values = torch.tensor(
        [float(metrics.get(key, 0.0)) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    for key, value in zip(keys, values.tolist(), strict=True):
        if key in metrics or value != 0.0:
            metrics[f"{key}_rank_max"] = float(value)
            # Online fitting must be byte-identical on every rank.
            metrics[key] = float(value)


@dataclass
class IterationData:
    """Immutable training owner and diagnostics for one sampled iteration."""

    normal: bool
    completion_quality: dict[str, float]
    profile_metrics: dict[str, float]
    storage: TrainingStorage
    partition_specs: Any
    decision_count: int
    leaf_count: int


def _value_workspace_budget(cfg: RootConfig, device: torch.device) -> int:
    """Return an explicit byte budget for completion and backward reduction."""

    gib = 1024**3
    if device.type != "cuda":
        raise RuntimeError("training value workspaces require CUDA")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    profile = cfg.hardware
    if profile is None:
        budget = free_bytes // 2
    else:
        reserve = int(profile.reserve_vram_gib * gib)
        budget = min(
            free_bytes - reserve,
            int(total_bytes * profile.max_memory_fraction),
        )
    if budget <= 0:
        raise MemoryError("no device memory remains for completion/backward workspace")
    return int(budget)


def _completion_batch_rows(
    bundle: GameBundle, leaf_count: int, workspace_bytes: int
) -> int:
    """Choose a conservative completion microbatch from a byte budget."""

    if leaf_count <= 0:
        return 1
    p_max = int(bundle.rollout_backend.n_players)
    x_max = max(int(size) for size in bundle.rollout_backend.x_sizes)
    per_leaf = p_max * x_max * (3 * 4 + 1)
    byte_limited = max(1, (workspace_bytes // 8) // max(per_leaf, 1))
    return min(leaf_count, 256, int(byte_limited))


def _merge_completion_quality(
    parts: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """Merge scalar quality summaries emitted by completion microbatches."""

    if not parts:
        return {}
    result: dict[str, float] = {}
    keys = {key for _, values in parts for key in values}
    for key in keys:
        present = [(rows, values[key]) for rows, values in parts if key in values]
        if key.startswith("completion_min_"):
            result[key] = min(value for _, value in present)
        elif key.startswith("completion_max_"):
            result[key] = max(value for _, value in present)
        elif key == "completion_invalid_posterior_mass":
            result[key] = sum(value for _, value in present)
        else:
            denominator = sum(rows for rows, _ in present)
            result[key] = sum(rows * value for rows, value in present) / max(
                denominator, 1
            )
    return result


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Return unique tensor storage bytes reachable from a compact data object."""

    if seen is None:
        seen = set()
    if isinstance(value, Tensor):
        storage = value.untyped_storage()
        identity = storage.data_ptr()
        if identity in seen:
            return 0
        seen.add(identity)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item, seen) for item in value)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return sum(_tensor_bytes(getattr(value, name), seen) for name in fields)
    return 0


def _analytical_memory_plan(
    cfg: RootConfig, bundle: GameBundle, *, has_reference: bool
) -> dict[str, int]:
    """Build the four phase prior from the schemas that own real tensors."""

    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    parameter = sum(value.numel() * value.element_size() for value in model.parameters())
    opponent_parameter = 0 if bundle.opponent_model is None else sum(
        value.numel() * value.element_size()
        for value in bundle.opponent_model.parameters()
    )
    backend = bundle.rollout_backend
    players = int(backend.n_players)
    exact_width = max(int(value) for value in backend.x_sizes)
    events = int(getattr(backend, "e_max", 1))
    roots = int(cfg.rollout.roots_per_gpu)
    page_rows = int(getattr(cfg.rollout, "storage_page_rows", None) or 256)
    sequential = cfg.training.mode != "branched_rao_blackwell"
    mode = {
        "normal": "normal",
        "rao_blackwell": "sequential_rb",
        "branched_rao_blackwell": "branched_rb",
    }[cfg.training.mode]
    path_rows = int(getattr(backend, "max_rollout_steps", 1)) + 1
    realized_prior_rows = roots * path_rows if sequential else page_rows
    decision_rows = realized_prior_rows
    leaf_rows = roots if sequential else page_rows
    action_rows = decision_rows if sequential else page_rows

    def paged(schema: TensorSchema, rows: int) -> int:
        return paged_capacity_bytes(schema, rows, page_rows)

    def exact(schema: TensorSchema, rows: int) -> int:
        return schema.capacity_bytes(rows, chunks=int(rows > 0))

    partition_factory = getattr(backend, "training_partition_specs", None)
    if not callable(partition_factory):
        raise TypeError("memory planning requires game-owned partition schemas")
    partition_specs = tuple(partition_factory().specs)
    if not partition_specs:
        raise ValueError("memory planning requires at least one partition schema")
    input_schemas = tuple(spec.input_schema for spec in partition_specs)
    input_destination = max(exact(schema, decision_rows) for schema in input_schemas)
    input_source = max(
        paged(
            TensorSchema(
                f"{schema.name}_capture",
                (TensorFieldSchema("node_id", torch.int32, alignment=4), *schema.fields),
                chunk_header_bytes=schema.chunk_header_bytes,
                allocator_slack_bytes=schema.allocator_slack_bytes,
            ),
            decision_rows,
        )
        for schema in input_schemas
    )
    pool_factory = getattr(backend, "training_input_pool_schemas", None)
    pool_schemas = (
        ()
        if not callable(pool_factory)
        else tuple(
            pair
            for pools in pool_factory().values()
            for pair in pools.values()
        )
    )
    input_pool_source = max(
        (paged(source, decision_rows) for source, _ in pool_schemas),
        default=0,
    )
    input_pool_destination = max(
        (exact(destination, decision_rows) for _, destination in pool_schemas),
        default=0,
    )
    input_source += input_pool_source
    input_destination += input_pool_destination

    source_decision_schemas = tuple(
        rollout_policy_source_schema(
            mode, model_width=int(spec.model_width), dynamic_heads=True
        )
        for spec in partition_specs
    )
    decision_schemas = tuple(
        rollout_decision_schema(
            mode,
            model_width=int(spec.model_width),
            event_count=events,
            dynamic_heads=True,
        )
        for spec in partition_specs
    )
    source_action_schemas = tuple(
        rollout_action_source_schema(int(spec.model_width))
        for spec in partition_specs
    )
    action_schemas = tuple(
        rollout_action_schema(int(spec.model_width)) for spec in partition_specs
    )
    decision_source = max(
        paged(schema, decision_rows) for schema in source_decision_schemas
    )
    decision_destination = max(
        exact(schema, decision_rows) for schema in decision_schemas
    )
    training_decision_base = max(
        exact(
            rollout_decision_schema(
                mode,
                model_width=int(spec.model_width),
                event_count=events,
                dynamic_heads=False,
            ),
            decision_rows,
        )
        for spec in partition_specs
    )
    action_source = 0 if sequential else max(
        paged(schema, action_rows) for schema in source_action_schemas
    )
    action_destination = 0 if sequential else max(
        exact(schema, action_rows) for schema in action_schemas
    )

    source_leaf_schema = stopped_leaf_source_schema(
        mode, players=players, exact_width=exact_width
    )
    destination_leaf_schema = rollout_leaf_schema(
        mode, players=players, exact_width=exact_width
    )
    leaf_source = paged(source_leaf_schema, leaf_rows)
    leaf_destination = exact(destination_leaf_schema, leaf_rows)

    source_topology = paged(
        PagedRolloutTopology.node_schema(players), realized_prior_rows
    ) + paged(PagedRolloutTopology.edge_schema(), realized_prior_rows)
    sealed_topology = exact(topology_schema(mode), realized_prior_rows)
    if sequential:
        sealed_topology += exact(
            TensorSchema(
                "sequential_trajectory_ptr",
                (TensorFieldSchema("trajectory_ptr", torch.int32, alignment=4),),
            ),
            roots + 1,
        )
        sealed_topology += exact(
            TensorSchema(
                "sequential_leaf_row",
                (TensorFieldSchema("leaf_row", torch.int32, alignment=4),),
            ),
            roots,
        )
    else:
        sealed_topology += exact(
            TensorSchema(
                "branched_nominal_degree",
                (TensorFieldSchema("nominal_degree", torch.int32, alignment=4),),
            ),
            realized_prior_rows,
        )
    rollout_minimum = getattr(backend, "rollout_memory_plan", None)
    minimum_live_pages = int(getattr(rollout_minimum, "peak_bytes", 0))
    active_rows = min(
        roots,
        max(
            int(getattr(cfg.rollout, "family_batch_rows", None) or 1),
            int(getattr(cfg.rollout, "max_active_trajectories", None) or 1),
        ),
    )
    reach_pack = active_rows * players * exact_width * 4
    bounded_workspace = 4 * reach_pack
    source_rows_bytes = (
        source_topology
        + input_source
        + decision_source
        + action_source
        + leaf_source
        + minimum_live_pages
        + bounded_workspace
    )
    if not sequential:
        source_rows_bytes = max(
            source_rows_bytes,
            int(getattr(cfg.rollout, "execution_memory_bytes", None) or (1 << 30)),
        )
    sealed_rollout = (
        sealed_topology
        + input_destination
        + decision_destination
        + action_destination
        + leaf_destination
    )
    seal_scratch = max(
        schema.payload_bytes(min(page_rows, max(decision_rows, 1)))
        for schema in (
            *input_schemas,
            *source_decision_schemas,
            *decision_schemas,
            source_leaf_schema,
            destination_leaf_schema,
            *source_action_schemas,
            *action_schemas,
        )
    )

    target_schema = target_decision_schema(
        mode, exact_width=exact_width, dynamic_heads=True
    )
    target_destination = exact(target_schema, decision_rows)
    posterior_pool = (
        exact(sequential_posterior_schema(exact_width), decision_rows)
        if mode == "sequential_rb"
        else 0
    )
    action_target = (
        exact(target_action_schema(exact_width), action_rows)
        if mode == "branched_rb"
        else 0
    )
    sampling_metadata = len(partition_specs) * 8
    if mode == "branched_rb":
        sampling_metadata += decision_rows * (8 + 4)
    action_pointers = (decision_rows + 1) * 4 if mode == "branched_rb" else 0
    rollout_training_base = (
        input_destination + training_decision_base + action_destination
    )
    target_only_destination = (
        target_destination
        + action_target
        + posterior_pool
        + sampling_metadata
        + action_pointers
    )
    training_storage = rollout_training_base + target_only_destination
    completion_rows = min(max(leaf_rows, 1), 64)
    completion_workspace = (
        0
        if cfg.training.mode == "normal"
        else completion_rows * players * exact_width * (3 * 4 + 1)
    )
    frontier_workspace = (
        page_rows * players * exact_width * 2 * 4
        if not sequential
        else roots * players * exact_width * 4
    )

    resident_seen: set[int] = set()
    backend_resident = _tensor_bytes(vars(backend), resident_seen)
    for owned in (
        getattr(backend, "states", None),
        getattr(backend, "partition_summary_cache", None),
    ):
        if owned is not None:
            backend_resident += _tensor_bytes(vars(owned), resident_seen)
    for service in getattr(backend, "_partition_services", {}).values():
        backend_resident += _tensor_bytes(vars(service[1]), resident_seen)
    reference = parameter if has_reference else 0
    optimizer_and_grad = 3 * parameter
    gradient_sync = parameter
    rollout_fixed = parameter + opponent_parameter + reference + backend_resident
    seal_fixed = rollout_fixed
    target_fixed = rollout_fixed + optimizer_and_grad
    optimization_fixed = target_fixed + gradient_sync
    rollout_variable = source_rows_bytes
    seal_variable = source_rows_bytes + sealed_rollout + seal_scratch
    target_variable = (
        sealed_rollout
        + target_only_destination
        + completion_workspace
        + frontier_workspace
    )
    learner_rows = min(decision_rows, int(getattr(cfg.training, "minibatch_size", 1)))
    learner_workspace = learner_rows * exact_width * 6 * 4
    optimization_variable = training_storage + learner_workspace
    rollout_phase = rollout_fixed + rollout_variable
    seal_phase = seal_fixed + seal_variable
    target_phase = target_fixed + target_variable
    optimization_phase = optimization_fixed + optimization_variable
    return {
        "memory_plan_parameter_bytes": parameter,
        "memory_plan_opponent_parameter_bytes": opponent_parameter,
        "memory_plan_backend_resident_bytes": backend_resident,
        "memory_plan_rollout_source_page_bytes": source_rows_bytes,
        "memory_plan_input_pool_source_bytes": input_pool_source,
        "memory_plan_input_pool_destination_bytes": input_pool_destination,
        "memory_plan_sealed_rollout_bytes": sealed_rollout,
        "memory_plan_seal_scratch_bytes": seal_scratch,
        "memory_plan_target_destination_bytes": target_only_destination,
        "memory_plan_sampling_metadata_bytes": sampling_metadata,
        "memory_plan_action_pointer_bytes": action_pointers,
        "memory_plan_training_storage_bytes": training_storage,
        "memory_plan_completion_workspace_bytes": completion_workspace,
        "memory_plan_frontier_workspace_bytes": frontier_workspace,
        "memory_plan_learner_workspace_bytes": learner_workspace,
        "memory_plan_optimizer_and_grad_bytes": optimizer_and_grad,
        "memory_plan_gradient_sync_bytes": gradient_sync,
        "memory_plan_reference_bytes": reference,
        "memory_plan_rollout_phase_bytes": rollout_phase,
        "memory_plan_seal_phase_bytes": seal_phase,
        "memory_plan_target_phase_bytes": target_phase,
        "memory_plan_optimization_phase_bytes": optimization_phase,
        "memory_plan_rollout_fixed_bytes": rollout_fixed,
        "memory_plan_rollout_variable_bytes": rollout_variable,
        "memory_plan_seal_fixed_bytes": seal_fixed,
        "memory_plan_seal_variable_bytes": seal_variable,
        "memory_plan_target_fixed_bytes": target_fixed,
        "memory_plan_target_variable_bytes": target_variable,
        "memory_plan_optimization_fixed_bytes": optimization_fixed,
        "memory_plan_optimization_variable_bytes": optimization_variable,
        "memory_plan_structural_peak_bytes": max(
            rollout_phase, seal_phase, target_phase, optimization_phase
        ),
    }


def _memory_limit(
    cfg: RootConfig, device: torch.device, *, account_current_free: bool = True
) -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if cfg.hardware is None:
        absolute = int(total_bytes * 0.85)
        return min(int(free_bytes * 0.85), absolute) if account_current_free else absolute
    reserve = int(cfg.hardware.reserve_vram_gib * 1024**3)
    absolute = min(
        total_bytes - reserve, int(total_bytes * cfg.hardware.max_memory_fraction)
    )
    return min(free_bytes - reserve, absolute) if account_current_free else absolute


def _validate_memory_preflight(
    cfg: RootConfig, device: torch.device, plan: dict[str, int]
) -> None:
    """Reject a structurally impossible CUDA plan before rollout allocation."""

    if device.type != "cuda":
        raise RuntimeError("training memory preflight requires CUDA")
    # ``plan`` describes the complete phase peak, including model/backend
    # tensors already allocated when this check runs. Comparing it with current
    # free memory would double-count those residents.
    limit = _memory_limit(cfg, device, account_current_free=False)
    structural = plan["memory_plan_structural_peak_bytes"]
    if limit <= 0 or structural > limit:
        raise MemoryError(
            "analytical CUDA memory plan exceeds the configured safe limit: "
            f"structural_peak={structural}, safe_limit={limit}"
        )


def _validate_memory_calibration(
    cfg: RootConfig, device: torch.device, metrics: dict[str, float]
) -> None:
    """Gate later iterations using the measured byte-complete CUDA peak."""

    if device.type != "cuda":
        raise RuntimeError("training memory calibration requires CUDA")
    peaks = [
        value
        for key, value in metrics.items()
        if (
            key == "iteration_peak_reserved_bytes"
            or (key.startswith("phase_") and key.endswith("_peak_reserved_bytes"))
        )
    ]
    if not peaks:
        raise RuntimeError("first CUDA iteration did not emit memory calibration peaks")
    measured = int(max(peaks))
    metrics["calibrated_peak_reserved_bytes"] = float(measured)
    safe_limit = _memory_limit(cfg, device, account_current_free=False)
    metrics["calibrated_safe_limit_bytes"] = float(safe_limit)
    headroom = safe_limit - measured
    metrics["calibrated_actual_headroom_bytes"] = float(headroom)
    metrics["memory_hard_limit_exceeded"] = float(headroom < 0)


def _profiled_call(
    metrics: dict[str, float],
    name: str,
    device: torch.device,
    enabled: bool,
    function: Any,
) -> Any:
    """Run one benchmark phase with accurate optional CUDA boundaries."""

    if not enabled:
        return function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics[f"phase_{name}_seconds"] = time.perf_counter() - start
    if device.type == "cuda":
        metrics[f"phase_{name}_peak_allocated_bytes"] = float(
            torch.cuda.max_memory_allocated(device)
        )
        metrics[f"phase_{name}_peak_reserved_bytes"] = float(
            torch.cuda.max_memory_reserved(device)
        )
    return result


def _decision_street_metrics(
    profile_metrics: dict[str, float],
    backend: Any,
    sample: SealedRolloutSample,
) -> None:
    """Record decision-node counts by poker street for one sampled rollout.

    Street labels live in the backend's device-resident state store. This is
    called only on the diagnostic path, so the small count reduction is copied
    to the host once per iteration rather than synchronizing every decision.
    """

    node_ids = sample.decision_inputs.node_ids
    if node_ids.numel() == 0:
        return
    captured = sample.decision_inputs
    if not isinstance(captured, SealedDecisionInputs):
        raise RuntimeError("decision metrics require sealed immutable inputs")
    labels = torch.empty(
        node_ids.numel(), dtype=torch.int64, device=node_ids.device
    )
    for partition_id, block in captured.inputs.items():
        selected = captured.partition_id == partition_id
        labels[selected] = block["street"][
            captured.partition_row[selected].to(torch.int64)
        ].to(torch.int64)
    labels = labels.clamp(0, 3)
    counts = torch.bincount(labels, minlength=4).detach().cpu().tolist()
    names = ("preflop", "flop", "turn", "river")
    total = max(sum(int(value) for value in counts), 1)
    for name, count in zip(names, counts, strict=True):
        profile_metrics[f"decision_nodes_{name}"] = float(count)
        profile_metrics[f"decision_fraction_{name}"] = float(count) / total


def _trajectory_length_metrics(
    profile_metrics: dict[str, float], sample: SealedRolloutSample, starts: int
) -> None:
    """Record policy-conditioned realized trajectory-length diagnostics."""

    lengths = sample.trajectory_node_count
    if starts <= 0 or lengths is None or lengths.numel() == 0:
        return
    lengths = lengths.float()
    ordered = lengths.sort().values
    count = ordered.numel()
    profile_metrics["trajectory_length_mean"] = float(lengths.mean())
    profile_metrics["trajectory_length_max"] = float(lengths.max())
    for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        index = min(count - 1, max(0, math.ceil(count * quantile) - 1))
        profile_metrics[f"trajectory_length_{name}"] = float(ordered[index])
    for upper in (4, 8, 16, 32, 64):
        profile_metrics[f"trajectory_length_le_{upper}"] = float(
            (lengths <= upper).sum()
        )


def _record_branched_controller_metrics(
    cfg: RootConfig,
    sample: SealedRolloutSample,
    controller: Any | None,
    profile_metrics: dict[str, float],
    *,
    iteration: int,
    device: torch.device,
) -> None:
    """Preserve dynamic-KKT observations before rollout owners are released."""

    count = sample.decision_inputs.node_ids.numel()
    old_t = torch.empty(count, dtype=torch.float32, device=device)
    old_c = torch.empty_like(old_t)
    for partition_id, source in sample.decision_inputs.decision_sources.items():
        source_t = source.columns.get("old_t")
        source_c = source.columns.get("old_c")
        if source_t is None or source_c is None:
            raise RuntimeError("branched rollout did not retain frozen T/C heads")
        selected = sample.decision_inputs.partition_id == partition_id
        rows = sample.decision_inputs.partition_row[selected].to(torch.int64)
        old_t[selected] = source_t[rows]
        old_c[selected] = source_c[rows]
    profile_metrics.update(
        {
            key: float(value)
            for key, value in sample.branching_diagnostics.items()
        }
    )
    if cfg.rollout.branching_mode != "dynamic":
        return
    adaptive = cfg.rollout.adaptive_kkt
    if adaptive is None:
        raise RuntimeError("dynamic branching has no adaptive KKT configuration")
    decision_ids = sample.decision_node
    if decision_ids is None:
        raise RuntimeError("branched direct seal omitted compact decision nodes")
    executed = sample.decision_aux["executed_child_count"].to(torch.float32)
    nominal = sample.decision_aux["nominal_child_count"].to(torch.float32)
    beta_values = nominal if sample.branching_beta is None else sample.branching_beta
    profile_metrics.update(
        {
            "beta_mean": float(beta_values.mean()),
            "beta_max": float(beta_values.max()),
            "realized_child_count_mean": float(executed.mean()),
            "realized_child_count_max": float(executed.max()),
            "beta_min_clip_rate": float((beta_values <= 1.0).float().mean()),
            "beta_max_clip_rate": float(
                (beta_values >= float(adaptive.beta_max)).float().mean()
            ),
            "tree_size_vs_M": float(sample.node_count)
            / float(adaptive.target_tree_size),
        }
    )
    if iteration < int(adaptive.warmup_iters):
        return
    topology = sample.topology
    pre_refill_nodes = topology.pre_refill_node_count
    if pre_refill_nodes is None:
        counterfactual = float(sample.node_count) + float(
            ((nominal - executed) * old_c).sum()
        )
    else:
        original = decision_ids < int(pre_refill_nodes)
        original_executed = executed[original].clone()
        original_ids = decision_ids[original]
        root = original_ids == 0
        if bool(root.any()):
            protected = topology.root_target_degree
            if protected is None:
                raise RuntimeError(
                    "pre-refill node metadata omitted the protected root degree"
                )
            original_executed[root] = float(protected)
        counterfactual = float(pre_refill_nodes) + float(
            (
                (nominal[original] - original_executed)
                * old_c[original]
            ).sum()
        )
    del device
    profile_metrics["counterfactual_tree_size_local"] = counterfactual
    if controller is None:
        raise RuntimeError("adaptive rollout has no KKT controller")


def _commit_branched_controller_observation(
    metrics: dict[str, float],
    device: torch.device,
    agreement: DistributedFailureAgreement,
) -> None:
    """Synchronize KKT evidence only after every rank sealed valid targets."""

    local = metrics.pop("counterfactual_tree_size_local", None)
    if local is None:
        return
    shared = torch.tensor(local, dtype=torch.float64, device=device)
    if agreement.distributed:
        dist.all_reduce(shared, op=dist.ReduceOp.SUM)
        shared.div_(dist.get_world_size(agreement.group))
    counterfactual = float(shared)
    metrics["counterfactual_tree_size"] = counterfactual


def _build_iteration_data(
    cfg: RootConfig,
    bundle: GameBundle,
    torch_gen: torch.Generator,
    *,
    profile: bool = False,
    frozen_model: nn.Module | None = None,
    iteration: int = 0,
    branching_controller: Any | None = None,
    branching_diagnostic_limits: Any | None = None,
) -> IterationData:
    """Build compact training data in a frame that owns all frozen transients.

    Rollout and target construction are synchronous, so the live model is put
    in evaluation mode and remains immutable until this function returns.  A
    second full policy copy is deliberately not allocated.
    """

    source = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = next(source.parameters()).device
    amp_enabled, amp_dtype = _precision(cfg, device)
    execution_metrics = getattr(bundle.rollout_backend, "execution_metrics", None)
    profile_metrics: dict[str, float] = {}
    if frozen_model is None:
        frozen_model = _profiled_call(
            profile_metrics,
            "snapshot",
            device,
            profile,
            source.eval,
        )
    frozen_bundle = GameBundle(
        model=frozen_model,
        rollout_backend=bundle.rollout_backend,
        policy_adapter=bundle.policy_adapter,
        completion_backend=bundle.completion_backend,
        opponent_model=bundle.opponent_model,
        probe_target=bundle.probe_target,
    )
    normal = cfg.training.mode == "normal"

    def record_sealed_rollout(rollout: RolloutStorage) -> None:
        """Expose exact owner extents and admitted page-seal overlap."""

        metrics = tuple(rollout.seal_metrics.values())
        profile_metrics.update(
            {
                "rollout_storage_bytes": float(_tensor_bytes(rollout)),
                "actions": float(
                    sum(
                        0 if block.actions is None else block.actions.rows
                        for block in rollout.partitions.values()
                    )
                ),
                "seal_source_pages": float(sum(value.source_pages for value in metrics)),
                "seal_source_capacity_bytes": float(
                    sum(value.source_capacity_bytes for value in metrics)
                ),
                "seal_destination_capacity_bytes": float(
                    sum(value.destination_capacity_bytes for value in metrics)
                ),
                "seal_scratch_capacity_bytes": float(
                    sum(value.scratch_capacity_bytes for value in metrics)
                ),
                "seal_overlap_peak_bytes": float(
                    max((value.peak_accounted_bytes for value in metrics), default=0)
                ),
                "seal_logical_payload_bytes": float(
                    sum(value.logical_payload_bytes for value in metrics)
                ),
            }
        )

    def sample_normal_rollout() -> Any:
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            return sample_trees_sequential(
                frozen_bundle,
                cfg,
                torch_gen,
                return_profiles=True,
                iteration=iteration,
            )

    def sample_rb_rollout() -> Any:
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            return sample_trees(
                frozen_bundle, cfg, torch_gen, iteration=iteration,
                branching_controller=branching_controller,
                branching_diagnostic_limits=branching_diagnostic_limits,
            )

    with torch.no_grad():
        if normal:
            sampled = _profiled_call(
                profile_metrics,
                "rollout",
                device,
                profile,
                sample_normal_rollout,
            )
            if not isinstance(sampled, SealedRolloutSample):
                raise RuntimeError("normal rollout returned a probe sample")
            sampled_decisions = sampled.decision_inputs.node_ids.numel()
            sampled_nodes = sampled.node_count
            sampled_leaves = sampled.leaf_source.rows
            sampled_offspring_mean = sampled_offspring_max = None
            if profile:
                sampled_offspring_mean = float(
                    (sampled_nodes - sampled_leaves) / max(sampled_nodes, 1)
                )
                sampled_offspring_max = 1.0
                _decision_street_metrics(
                    profile_metrics,
                    bundle.rollout_backend,
                    sampled,
                )
                _trajectory_length_metrics(
                    profile_metrics, sampled, int(cfg.rollout.roots_per_gpu)
                )
            rollout, partitions, _, _ = _profiled_call(
                profile_metrics,
                "seal",
                device,
                profile,
                lambda: _seal_sequential_rollout_storage(
                    cfg,
                    frozen_bundle,
                    sampled,
                    iteration=iteration,
                    gen=torch_gen,
                ),
            )
            if profile:
                record_sealed_rollout(rollout)
            rollout_leaf_count = int(rollout.leaves.rows)
            del sampled
            storage = _profiled_call(
                profile_metrics,
                "backward",
                device,
                profile,
                lambda: build_normal_targets(
                    rollout, partitions, cfg.loss.lam
                ),
            )
            del rollout
            if execution_metrics is not None:
                execution_metrics.record_target_bytes(_tensor_bytes(storage))
            decisions = sum(
                partition.inputs.rows for partition in storage.partitions.values()
            )
            if decisions != sampled_decisions:
                raise ValueError("normal target table lost or duplicated decision rows")
            if profile:
                profile_metrics.update(
                    {
                        "trajectories": float(cfg.rollout.roots_per_gpu),
                        "nodes": float(sampled_nodes),
                        "leaves": float(sampled_leaves),
                        "decisions": float(decisions),
                        "terminal_samples": float(sampled_leaves),
                        "retained_target_bytes": float(_tensor_bytes(storage)),
                        "policy_decisions": float(sampled_decisions),
                        "offspring_mean": sampled_offspring_mean,
                        "offspring_max": sampled_offspring_max,
                    }
                )
            return IterationData(
                normal=True,
                completion_quality={},
                profile_metrics=profile_metrics,
                storage=storage,
                partition_specs=partitions,
                decision_count=decisions,
                leaf_count=rollout_leaf_count,
            )

        rb_sample = _profiled_call(
            profile_metrics,
            "rollout",
            device,
            profile,
            sample_rb_rollout,
        )
        if cfg.training.mode == "rao_blackwell":
            if isinstance(rb_sample, ProbeRollout):
                sample = rb_sample.sample
            else:
                sample = rb_sample
            sampled_nodes = sample.node_count
            sampled_offspring_mean = sampled_offspring_max = None
            if profile:
                sampled_offspring_mean = float(
                    (sampled_nodes - sample.leaf_source.rows) / max(sampled_nodes, 1)
                )
                sampled_offspring_max = 1.0
                _decision_street_metrics(
                    profile_metrics,
                    bundle.rollout_backend,
                    sample,
                )
                _trajectory_length_metrics(
                    profile_metrics, sample, int(cfg.rollout.roots_per_gpu)
                )
            rollout, partitions, completion, quality_parts = _profiled_call(
                profile_metrics,
                "seal",
                device,
                profile,
                lambda: _seal_sequential_rollout_storage(
                    cfg,
                    frozen_bundle,
                    rb_sample,
                    iteration=iteration,
                    gen=torch_gen,
                ),
            )
            if profile:
                record_sealed_rollout(rollout)
            rollout_leaf_count = int(rollout.leaves.rows)
            del rb_sample, sample
            if completion is None:
                raise RuntimeError("sequential RB seal omitted completion")
            workspace_bytes = _value_workspace_budget(cfg, device)
            completion_rows = _completion_batch_rows(
                bundle, rollout.leaves.rows, workspace_bytes
            )
            storage = _profiled_call(
                profile_metrics,
                "backward",
                device,
                profile,
                lambda: build_sequential_rb_targets(
                    rollout,
                    partitions,
                    completion,
                    cfg.loss.lam,
                    leaf_microbatch_rows=completion_rows,
                ),
            )
            del rollout, completion
            if execution_metrics is not None:
                execution_metrics.record_target_bytes(_tensor_bytes(storage))
            completion_quality = _merge_completion_quality(quality_parts)
            decisions = sum(
                partition.inputs.rows for partition in storage.partitions.values()
            )
            profile_metrics.update(
                {
                    "trajectories": float(cfg.rollout.roots_per_gpu),
                    "nodes": float(sampled_nodes),
                    "leaves": float(rollout_leaf_count),
                    "decisions": float(decisions),
                    "completion_samples": float(
                        rollout_leaf_count * int(cfg.completion.n_samples)
                    ),
                    "offspring_mean": sampled_offspring_mean,
                    "offspring_max": sampled_offspring_max,
                    "retained_target_bytes": float(_tensor_bytes(storage)),
                }
            )
            return IterationData(
                normal=False,
                completion_quality=completion_quality,
                profile_metrics=profile_metrics,
                storage=storage,
                partition_specs=partitions,
                decision_count=decisions,
                leaf_count=rollout_leaf_count,
            )
        if cfg.training.mode == "branched_rao_blackwell":
            if isinstance(rb_sample, ProbeRollout):
                raise RuntimeError("branched rollout cannot be an opponent probe")
            sampled_nodes = rb_sample.node_count
            _record_branched_controller_metrics(
                cfg,
                rb_sample,
                branching_controller,
                profile_metrics,
                iteration=iteration,
                device=device,
            )
            if profile:
                _decision_street_metrics(
                    profile_metrics,
                    bundle.rollout_backend,
                    rb_sample,
                )
            rollout, partitions, completion, quality_parts = _profiled_call(
                profile_metrics,
                "seal",
                device,
                profile,
                lambda: _seal_branched_rollout_storage(
                    cfg,
                    frozen_bundle,
                    rb_sample,
                    iteration=iteration,
                    gen=torch_gen,
                ),
            )
            if profile:
                record_sealed_rollout(rollout)
            rollout_leaf_count = int(rollout.leaves.rows)
            del rb_sample
            workspace_bytes = _value_workspace_budget(cfg, device)
            completion_rows = _completion_batch_rows(
                bundle, rollout.leaves.rows, workspace_bytes
            )
            storage = _profiled_call(
                profile_metrics,
                "backward",
                device,
                profile,
                lambda: build_branched_rb_targets(
                    rollout,
                    partitions,
                    completion,
                    cfg.loss.lam,
                    leaf_microbatch_rows=completion_rows,
                ),
            )
            del rollout, completion
            if execution_metrics is not None:
                execution_metrics.record_target_bytes(_tensor_bytes(storage))
            decisions = sum(
                partition.inputs.rows for partition in storage.partitions.values()
            )
            profile_metrics.update(
                {
                    "trajectories": 1.0,
                    "nodes": float(sampled_nodes),
                    "leaves": float(rollout_leaf_count),
                    "decisions": float(decisions),
                    "completion_samples": float(
                        rollout_leaf_count * int(cfg.completion.n_samples)
                    ),
                    "retained_target_bytes": float(_tensor_bytes(storage)),
                }
            )
            return IterationData(
                normal=False,
                completion_quality=_merge_completion_quality(quality_parts),
                profile_metrics=profile_metrics,
                storage=storage,
                partition_specs=partitions,
                decision_count=decisions,
                leaf_count=rollout_leaf_count,
            )
        raise RuntimeError(f"unsupported training mode {cfg.training.mode!r}")


def _natural_partition_set(cfg: RootConfig, bundle: GameBundle):
    """Obtain the game-owned opaque partition boundary for the learner."""

    backend = bundle.rollout_backend
    factory = getattr(backend, "training_partition_specs", None)
    if not callable(factory):
        raise TypeError("rollout backend does not expose training partition specs")
    return factory()


def _sealed_dynamic_block(
    label: str,
    columns: Mapping[str, Tensor],
    *,
    schema: TensorSchema | None = None,
) -> SealedTensorBlock:
    """Seal an exact game-owned auxiliary pool from its realized field shapes."""

    if not columns:
        raise ValueError("a sealed auxiliary block requires at least one field")
    rows = next(iter(columns.values())).shape[0]
    if schema is None:
        schema = TensorSchema(
            label,
            tuple(
                TensorFieldSchema(
                    name,
                    value.dtype,
                    tuple(value.shape[1:]),
                    alignment=min(max(value.element_size(), 1), 16),
                )
                for name, value in columns.items()
            ),
        )
    block = SealedTensorBlock(
        schema,
        {name: value.detach() for name, value in columns.items()},
        rows,
        schema.capacity_bytes(rows, chunks=int(rows > 0)),
    )
    block.assert_exact_payload()
    return block


def _seal_sequential_rollout_storage(
    cfg: RootConfig,
    bundle: GameBundle,
    sampled: Any,
    *,
    iteration: int,
    gen: torch.Generator,
) -> tuple[RolloutStorage, Any, CompletionSpec | None, list[tuple[int, dict[str, float]]]]:
    """Seal one family-scheduled sequential sample before target construction.

    None of the scheduler topology, stopped-reach capture, policy capture, or
    mutable poker-state owners cross this boundary. The returned topology is
    decision-only and every model/completion input is immutable.
    """

    probe = sampled if isinstance(sampled, ProbeRollout) else None
    sample = probe.sample if probe is not None else sampled
    if not isinstance(sample, SealedRolloutSample):
        raise TypeError("sequential seal requires a directly sealed rollout sample")
    root_profiles = sample.root_profiles
    backend = bundle.rollout_backend
    device = backend.device
    partitions = _natural_partition_set(cfg, bundle)
    captured = sample.decision_inputs
    if not isinstance(captured, SealedDecisionInputs):
        raise RuntimeError("sequential sealing requires immutable decision inputs")
    nodes = captured.node_ids.to(torch.int64)
    partition_id = captured.partition_id.to(torch.int64)

    blocks: dict[int, PartitionRolloutBlock] = {}
    global_to_partition_row = torch.full(
        (nodes.numel(),), -1, dtype=torch.int32, device=device
    )
    for partition in partitions.specs:
        selected = (partition_id == partition.partition_id).nonzero(
            as_tuple=False
        ).squeeze(1)
        if selected.numel() == 0:
            continue
        inputs = captured.inputs[partition.partition_id]
        source = captured.decision_sources[partition.partition_id]
        input_pools = dict(captured.input_pools.get(partition.partition_id, {}))
        captured_rows = captured.partition_row[selected].to(torch.int64)
        if not torch.equal(
            captured_rows,
            torch.arange(selected.numel(), dtype=torch.int64, device=device),
        ):
            raise RuntimeError("captured partition rows are not exact and contiguous")
        legal_ids = torch.arange(
            backend.e_max, dtype=torch.int32, device=device
        )[None].expand(selected.numel(), -1)
        decision_columns = {
            "owner": source["owner"].detach().to(torch.int32),
            "event": source["event"].detach().to(torch.int32),
            "old_log_prob": source["old_logp"].detach().to(torch.float32),
            "old_value": source["old_v"].detach().to(torch.float32),
            "legal_event_ids": legal_ids,
            "legal_event_mask": inputs["legal_action_mask"].detach().to(torch.bool),
        }
        if cfg.training.mode == "normal":
            if root_profiles is None:
                raise RuntimeError("normal rollout omitted sampled private profiles")
            realized_type = root_profiles[
                source["root_id"].to(torch.int64), source["owner"].to(torch.int64)
            ].to(torch.int32)
            model_type = torch.as_tensor(
                partition.exact_to_model, dtype=torch.int64, device=device
            )[realized_type.to(torch.int64)]
            row = torch.arange(source.rows, device=device)
            decision_columns["realized_type"] = realized_type
            decision_columns["old_log_prob"] = source["old_logp"][row, model_type]
            decision_columns["old_value"] = source["old_v"][row, model_type]
        decisions = _sealed_dynamic_block(
            f"poker_partition_{partition.partition_id}_decision",
            decision_columns,
            schema=rollout_decision_schema(
                "normal" if cfg.training.mode == "normal" else "sequential_rb",
                model_width=partition.model_width,
                event_count=backend.e_max,
                dynamic_heads=False,
            ),
        )
        global_to_partition_row[selected] = captured.partition_row[selected]
        blocks[partition.partition_id] = PartitionRolloutBlock(
            partition.partition_id,
            inputs,
            decisions,
            input_pools=input_pools,
        )

    leaf_source = sample.leaf_source
    leaf_roots = leaf_source["root_id"].to(torch.int32)
    root_count = int(cfg.rollout.roots_per_gpu)
    if not torch.equal(
        leaf_roots.to(torch.int64), torch.arange(root_count, device=device)
    ):
        raise RuntimeError("sequential rollout did not seal one leaf per root")
    completion: CompletionSpec | None = None
    quality_parts: list[tuple[int, dict[str, float]]] = []
    if cfg.training.mode == "normal":
        assert root_profiles is not None
        reward = leaf_source["reward"]
        reward = reward.float() * utility_scale(cfg)
        leaves = _sealed_dynamic_block(
            "normal_leaf",
            {"reward": reward},
            schema=rollout_leaf_schema(
                "normal", players=backend.n_players, exact_width=backend.n_hands
            ),
        )
        mode = "normal"
    else:
        if bundle.completion_backend is None or cfg.completion.n_samples is None:
            raise RuntimeError("sequential RB requires a completion service")
        leaf_reach = leaf_source["log_reach"].detach().to(torch.float32)
        captured_leaf = leaf_source.columns
        completion_inputs = {
            name: value
            for name, value in captured_leaf.items()
            if name
            not in {
                "node_id",
                "root_id",
                "path_key",
                "log_reach",
                "owner",
                "fold_payoff",
                "reward",
            }
        }
        completion_inputs["range_weights"] = backend.range_weights_from_log_reach(
            leaf_reach
        )
        leaf_owner = captured_leaf["owner"]
        fold_payoff = captured_leaf["fold_payoff"]
        completion_key = sequential_leaf_logical_rng_keys(
            leaf_source["root_id"],
            leaf_source["path_key"],
            cfg,
            gen,
            iteration=iteration,
            purpose=DrawPurpose.COMPLETION,
        )
        leaf_columns = {
            "owner": leaf_owner,
            "completion_key": completion_key.to(torch.int64),
            "fold_payoff": fold_payoff.detach().to(torch.float32),
            **{name: value.detach() for name, value in completion_inputs.items()},
        }
        leaves = _sealed_dynamic_block(
            "sequential_rb_leaf",
            leaf_columns,
            schema=rollout_leaf_schema(
                "sequential_rb",
                players=backend.n_players,
                exact_width=backend.n_hands,
            ),
        )
        service = bundle.completion_backend
        complete_sealed = getattr(service, "complete_sealed", None)
        if not callable(complete_sealed):
            raise TypeError("completion backend lacks immutable-leaf completion")

        def complete_leaf_rows(
            columns: Mapping[str, Tensor], *, rows: Tensor
        ) -> BatchedCompletionStats:
            result = complete_sealed(
                backend,
                columns,
                rows=rows,
                n_samples=int(cfg.completion.n_samples),
            ).for_learning()
            scaled = _scale_completion_utility(result, cfg)
            quality_parts.append(
                (
                    rows.numel(),
                    {
                        key: float(value)
                        for key, value in scaled.metadata.items()
                        if key.startswith("completion_")
                        and isinstance(value, (int, float))
                    },
                )
            )
            return scaled

        completion = CompletionSpec(
            tuple(leaf_columns),
            ("range_weights",),
            ("p_hat", "v_hat"),
            complete_leaf_rows,
            lambda count: int(count) * backend.n_players * backend.n_hands * 12,
            key_fields=("completion_key",),
        )
        mode = "sequential_rb"

    topology = sample.topology
    if not isinstance(topology, SequentialDecisionCSR):
        raise TypeError("sequential direct seal omitted decision-only CSR")
    rollout = RolloutStorage(
        mode,
        blocks,
        topology,
        leaves,
        sample.seal_metrics,
    )
    return rollout, partitions, completion, quality_parts


def _seal_branched_rollout_storage(
    cfg: RootConfig,
    bundle: GameBundle,
    sampled: Any,
    *,
    iteration: int,
    gen: torch.Generator,
) -> tuple[RolloutStorage, Any, CompletionSpec, list[tuple[int, dict[str, float]]]]:
    """Seal one rooted branched sample into ragged immutable storage."""

    if not isinstance(sampled, SealedRolloutSample):
        raise TypeError("branched seal requires a directly sealed rollout sample")
    backend = bundle.rollout_backend
    device = backend.device
    partitions = _natural_partition_set(cfg, bundle)
    captured = sampled.decision_inputs
    if not isinstance(captured, SealedDecisionInputs):
        raise RuntimeError("branched production requires captured immutable inputs")
    nodes = captured.node_ids.to(torch.int64)

    topology = sampled.topology
    if not isinstance(topology, BranchedTreeSpan):
        raise TypeError("branched direct seal omitted compact span topology")

    target_degree = sampled.decision_aux["executed_child_count"].to(torch.int64)
    nominal_decision = sampled.decision_aux["nominal_child_count"].to(torch.int32)

    blocks: dict[int, PartitionRolloutBlock] = {}
    for partition in partitions.specs:
        selected = (captured.partition_id == partition.partition_id).nonzero(
            as_tuple=False
        ).squeeze(1)
        if selected.numel() == 0:
            continue
        source = captured.decision_sources[partition.partition_id]
        captured_rows = captured.partition_row[selected].to(torch.int64)
        if not torch.equal(
            captured_rows,
            torch.arange(selected.numel(), dtype=torch.int64, device=device),
        ):
            raise RuntimeError("branched partition input rows are not contiguous")
        inputs = captured.inputs[partition.partition_id]
        decisions_columns = {
            "owner": sampled.decision_aux["owner"][selected]
            .detach()
            .to(torch.int32),
            "old_value": source["old_v"].detach().to(torch.float32),
            "legal_event_ids": torch.arange(
                backend.e_max, dtype=torch.int32, device=device
            )[None].expand(selected.numel(), -1),
            "legal_event_mask": inputs["legal_action_mask"].detach().to(torch.bool),
            "log_weight": sampled.decision_aux["log_weight"][selected]
            .detach()
            .to(torch.float64),
            "nominal_child_count": nominal_decision[selected],
            "executed_child_count": target_degree[selected].to(torch.int32),
        }
        old_t = source.columns.get("old_t")
        old_c = source.columns.get("old_c")
        if old_t is not None or old_c is not None:
            if old_t is None or old_c is None:
                raise RuntimeError("branched frozen T/C heads are incomplete")
            decisions_columns["frozen_trace_variance"] = old_t.detach()
            decisions_columns["frozen_cost"] = old_c.detach()
        decisions = _sealed_dynamic_block(
            f"poker_partition_{partition.partition_id}_branched_decision",
            decisions_columns,
            schema=rollout_decision_schema(
                "branched_rb",
                model_width=partition.model_width,
                event_count=backend.e_max,
                dynamic_heads="frozen_trace_variance" in decisions_columns,
            ),
        )
        action_source = captured.action_sources[partition.partition_id]
        action_ptr = captured.action_ptrs[partition.partition_id]
        actions = _sealed_dynamic_block(
            f"poker_partition_{partition.partition_id}_branched_action",
            {
                "event": action_source["event"].detach().to(torch.int32),
                "old_log_prob": action_source["old_logp"].detach().to(torch.float32),
            },
            schema=rollout_action_schema(partition.model_width),
        )
        blocks[partition.partition_id] = PartitionRolloutBlock(
            partition.partition_id,
            inputs,
            decisions,
            actions,
            action_ptr,
            captured.input_pools.get(partition.partition_id, {}),
        )

    leaf_source = sampled.leaf_source
    leaf_reach = leaf_source["log_reach"].detach().to(torch.float32)
    captured_leaf = leaf_source.columns
    completion_inputs = {
        name: value
        for name, value in captured_leaf.items()
        if name
        not in {
            "node_id",
            "root_id",
            "path_key",
            "log_reach",
            "owner",
            "fold_payoff",
        }
    }
    completion_inputs["range_weights"] = backend.range_weights_from_log_reach(
        leaf_reach
    )
    leaf_owner = captured_leaf["owner"]
    fold_payoff = captured_leaf["fold_payoff"]
    completion_key = logical_rng_keys(
        seed=_logical_seed(cfg, gen),
        iteration=iteration,
        trajectory=leaf_source["root_id"].to(torch.int64),
        path=leaf_source["path_key"].to(torch.int64),
        draw=0,
        purpose=DrawPurpose.COMPLETION,
    )
    leaf_columns = {
        "owner": leaf_owner,
        "completion_key": completion_key.to(torch.int64),
        "fold_payoff": fold_payoff.detach().to(torch.float32),
        **{name: value.detach() for name, value in completion_inputs.items()},
    }
    leaves = _sealed_dynamic_block(
        "branched_rb_leaf",
        leaf_columns,
        schema=rollout_leaf_schema(
            "branched_rb",
            players=backend.n_players,
            exact_width=backend.n_hands,
        ),
    )
    service = bundle.completion_backend
    if service is None or cfg.completion.n_samples is None:
        raise RuntimeError("branched RB requires a completion service")
    complete_sealed = getattr(service, "complete_sealed", None)
    if not callable(complete_sealed):
        raise TypeError("completion backend lacks immutable-leaf completion")
    quality_parts: list[tuple[int, dict[str, float]]] = []

    def complete_leaf_rows(
        columns: Mapping[str, Tensor], *, rows: Tensor
    ) -> BatchedCompletionStats:
        result = complete_sealed(
            backend,
            columns,
            rows=rows,
            n_samples=int(cfg.completion.n_samples),
        ).for_learning()
        scaled = _scale_completion_utility(result, cfg)
        quality_parts.append(
            (
                rows.numel(),
                {
                    key: float(value)
                    for key, value in scaled.metadata.items()
                    if key.startswith("completion_")
                    and isinstance(value, (int, float))
                },
            )
        )
        return scaled

    completion = CompletionSpec(
        tuple(leaf_columns),
        ("range_weights",),
        ("p_hat", "v_hat"),
        complete_leaf_rows,
        lambda count: int(count) * backend.n_players * backend.n_hands * 12,
        key_fields=("completion_key",),
    )
    rollout = RolloutStorage(
        "branched_rb",
        blocks,
        topology,
        leaves,
        sampled.seal_metrics,
    )
    return rollout, partitions, completion, quality_parts

def _validate_policy_identity(
    cfg: RootConfig,
    bundle: GameBundle,
    data: IterationData,
) -> dict[str, float]:
    """Validate frozen outputs using only immutable natural learner inputs."""

    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = next(model.parameters()).device
    amp_enabled, amp_dtype = _precision(cfg, device)
    was_training = model.training
    model.eval()
    maximum_logp_error = torch.zeros((), device=device)
    maximum_value_error = torch.zeros((), device=device)
    maximum_value_scale = torch.zeros((), device=device)
    ratio_min = torch.full((), torch.inf, device=device)
    ratio_max = torch.zeros((), device=device)
    row_cap = max(1, int(cfg.rollout.family_batch_rows or 64))
    with torch.no_grad():
        for partition_id, partition in data.storage.partitions.items():
            spec = data.partition_specs[partition_id]
            for first in range(0, partition.inputs.rows, row_cap):
                rows = torch.arange(
                    first,
                    min(first + row_cap, partition.inputs.rows),
                    dtype=torch.int64,
                    device=device,
                )
                inputs = partition.gather_inputs(rows)
                with torch.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                    dtype=amp_dtype,
                ):
                    output = spec.forward(model, inputs)
                legal_ids = partition.columns["legal_event_ids"][rows]
                legal_mask = partition.columns["legal_event_mask"][rows]
                if data.storage.mode == "branched_rb":
                    actions = partition.action_columns
                    if actions is None:
                        raise RuntimeError("branched identity gate omitted actions")
                    ptr = actions["action_ptr"].to(torch.int64)
                    starts = ptr[rows]
                    counts = ptr[rows + 1] - starts
                    width = int(counts.max())
                    slots = torch.arange(width, device=device)[None]
                    active = slots < counts[:, None]
                    action_rows = starts[:, None] + slots
                    safe_rows = action_rows.clamp_max(actions["event"].numel() - 1)
                    safe_event = legal_ids.gather(
                        1, legal_mask.to(torch.int64).argmax(1, keepdim=True)
                    )
                    events = torch.where(
                        active,
                        actions["event"][safe_rows],
                        safe_event.expand_as(action_rows),
                    )
                    selected = bundle.policy_adapter.log_prob(
                        output, events, legal_ids, legal_mask
                    ).float()
                    current_logp = selected[active]
                    cached_logp = actions["old_log_prob"][safe_rows][active]
                    current_value = output.v.float()
                    cached_value = partition.columns["old_value"][rows]
                else:
                    selected = bundle.policy_adapter.log_prob(
                        output,
                        partition.columns["event"][rows, None],
                        legal_ids,
                        legal_mask,
                    )[:, 0].float()
                if data.normal:
                    realized = partition.columns["realized_type"][rows].to(torch.int64)
                    model_type = spec.phi(device=device)[realized]
                    batch = torch.arange(rows.numel(), device=device)
                    current_logp = selected[batch, model_type]
                    current_value = output.v.float()[batch, model_type]
                    cached_logp = partition.columns["old_log_prob"][rows]
                    cached_value = partition.columns["old_value"][rows]
                elif data.storage.mode != "branched_rb":
                    current_logp = selected
                    current_value = output.v.float()
                    cached_logp = partition.columns["old_log_prob"][rows]
                    cached_value = partition.columns["old_value"][rows]
                log_ratio = current_logp - cached_logp.to(current_logp)
                value_error = current_value - cached_value.to(current_value)
                ratios = torch.exp(log_ratio)
                maximum_logp_error = torch.maximum(
                    maximum_logp_error, log_ratio.abs().max()
                )
                maximum_value_error = torch.maximum(
                    maximum_value_error, value_error.abs().max()
                )
                maximum_value_scale = torch.maximum(
                    maximum_value_scale,
                    torch.maximum(
                        current_value.abs().max(), cached_value.abs().max()
                    ),
                )
                ratio_min = torch.minimum(ratio_min, ratios.min())
                ratio_max = torch.maximum(ratio_max, ratios.max())
    if was_training:
        model.train()
    if dist.is_initialized():
        dist.all_reduce(maximum_logp_error, op=dist.ReduceOp.MAX)
        dist.all_reduce(maximum_value_error, op=dist.ReduceOp.MAX)
        dist.all_reduce(maximum_value_scale, op=dist.ReduceOp.MAX)
        dist.all_reduce(ratio_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(ratio_max, op=dist.ReduceOp.MAX)
    logp_tolerance = 2.0e-2 if amp_enabled else 2.0e-5
    value_tolerance = 1.0e-2 if amp_enabled else 2.0e-5
    ok = (
        (maximum_value_error <= value_tolerance * (1.0 + maximum_value_scale))
        & (maximum_logp_error <= logp_tolerance)
        & (ratio_min >= 1.0 - float(cfg.loss.clip_eps))
        & (ratio_max <= 1.0 + float(cfg.loss.clip_eps))
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    diagnostics = {
        "identity_max_abs_logp_error": float(maximum_logp_error),
        "identity_max_abs_value_error": float(maximum_value_error),
        "identity_ratio_min": float(ratio_min),
        "identity_ratio_max": float(ratio_max),
        "identity_logp_tolerance": float(logp_tolerance),
        "identity_value_tolerance": float(value_tolerance),
    }
    if not bool(ok):
        raise RuntimeError(
            "sealed old/current-policy identity gate failed: "
            f"max_abs_logp_error={diagnostics['identity_max_abs_logp_error']:.8g}, "
            f"max_abs_value_error={diagnostics['identity_max_abs_value_error']:.8g}, "
            f"ratio_min={diagnostics['identity_ratio_min']:.8g}, "
            f"ratio_max={diagnostics['identity_ratio_max']:.8g}"
        )
    return diagnostics


def _one_iteration(
    cfg: RootConfig,
    bundle: GameBundle,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    torch_gen: torch.Generator,
    iteration: int,
    *,
    profile: bool = False,
    reference_model: nn.Module | None = None,
    validate_identity: bool = False,
    branching_controller: Any | None = None,
    branching_diagnostic_limits: Any | None = None,
    failure_agreement: DistributedFailureAgreement | None = None,
) -> dict[str, Any]:
    execution_metrics = getattr(bundle.rollout_backend, "execution_metrics", None)
    if execution_metrics is not None:
        execution_metrics.clear()
    model = bundle.model
    source = model.module if hasattr(model, "module") else model
    device = next(model.parameters()).device
    agreement = failure_agreement or DistributedFailureAgreement(device)
    memory_ledger = MemoryLedger(device)
    memory_metrics: dict[str, float] = {}
    if device.type == "cuda":
        # Calibration must cover allocator growth and new cache bins on every
        # iteration, not only the initially profiled one.
        torch.cuda.reset_peak_memory_stats(device)
        memory_metrics.update(
            memory_ledger.snapshot(
                "iteration_start",
                owned={
                    "model": source,
                    "optimizer": optimizer.state,
                    "reference": reference_model,
                    "backend": bundle.rollout_backend,
                },
            )
        )
    amp_enabled, amp_dtype = _precision(cfg, device)
    total_trajectories = int(cfg.rollout.roots_per_gpu)
    identity_metrics: dict[str, float] = {}
    data = agreement.run(
        "rollout, seal, completion, and targets",
        lambda: _build_iteration_data(
            cfg,
            bundle,
            torch_gen,
            profile=profile,
            iteration=iteration,
            branching_controller=branching_controller,
            branching_diagnostic_limits=branching_diagnostic_limits,
        ),
    )
    _commit_branched_controller_observation(
        data.profile_metrics,
        device,
        agreement,
    )
    if profile:
        memory_metrics.update(
            memory_ledger.snapshot(
                "preoptimization",
                owned={
                    "model": source,
                    "optimizer": optimizer.state,
                    "reference": reference_model,
                    "backend": bundle.rollout_backend,
                    "iteration": data,
                },
            )
        )
    if validate_identity:
        if execution_metrics is None:
            identity_metrics = _validate_policy_identity(cfg, bundle, data)
        else:
            # The identity gate is launch/resume validation, not scientific
            # rollout or learner work.  Excluding it keeps exact execution
            # counters invariant under checkpoint continuation.
            with execution_metrics.suspended():
                identity_metrics = _validate_policy_identity(
                    cfg, bundle, data
                )
        if device.type == "cuda":
            # The launch/resume identity gate is not learner work. Drain its
            # kernels here so they cannot be charged to the following sealed
            # storage or optimization phase timer.
            torch.cuda.synchronize(device)

    decisions = data.decision_count
    leaves = data.leaf_count

    optimizer_steps_this_iteration = int(cfg.training.gd_steps_per_iter)

    storage = data.storage
    partition_specs = data.partition_specs
    retained_bytes = storage.tensor_storage_bytes()
    if data.profile_metrics is not None:
        for partition_id, partition in storage.partitions.items():
            partition_bytes = partition.inputs.tensor_storage_bytes + sum(
                value.numel() * value.element_size()
                for value in partition.columns.values()
            )
            if partition.action_columns is not None:
                partition_bytes += sum(
                    value.numel() * value.element_size()
                    for value in partition.action_columns.values()
                )
            data.profile_metrics.update(
                {
                    f"training_partition_{partition_id}_rows": float(
                        partition.inputs.rows
                    ),
                    f"training_partition_{partition_id}_bytes": float(
                        partition_bytes
                    ),
                    f"training_partition_{partition_id}_input_bytes": float(
                        partition.inputs.tensor_storage_bytes
                    ),
                }
            )
        data.profile_metrics.update(
            {
                "retained_target_bytes": float(retained_bytes),
                "training_storage_schema_version": 1.0,
                "preflop_model_width": float(bundle.rollout_backend.n_classes),
                "postflop_model_width": float(bundle.rollout_backend.n_hands),
                "rb_exact_width": float(bundle.rollout_backend.n_hands),
            }
        )
        # The target builder is the profiled backward-estimator phase. Expose
        # the design name as an explicit alias in release/soak telemetry.
        for suffix in ("seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            source_key = f"phase_backward_{suffix}"
            if source_key in data.profile_metrics:
                data.profile_metrics[f"phase_targets_{suffix}"] = data.profile_metrics[
                    source_key
                ]
    # The sealed store is now the sole iteration owner crossing into learning.
    completion_quality = data.completion_quality
    iteration_profile = data.profile_metrics or {}
    del data

    active_ids = tuple(sorted(storage.partitions))
    if not active_ids:
        raise ValueError("cannot optimize an empty fixed rollout")
    if cfg.training.minibatch_size < len(active_ids):
        raise ValueError(
            "training.minibatch_size must cover every nonempty execution partition"
        )
    local_logs = torch.stack(
        [storage.sampling.log_z[partition_id] for partition_id in active_ids]
    )
    local_mass = torch.exp(local_logs - torch.logsumexp(local_logs, 0))
    family_names = (
        "pre" if partition_id == 0 else f"post_{partition_id + 1}"
        for partition_id in active_ids
    )
    widths = torch.tensor(
        [bundle.rollout_backend._family_maximum_width(source, name) for name in family_names],
        dtype=torch.float64,
        device=device,
    )
    allocated = allocate_fixed_statistical_counts(
        local_mass,
        torch.ones_like(local_mass),
        widths.square(),
        cfg.training.minibatch_size,
    )
    statistical_counts = {
        partition_id: int(allocated[position])
        for position, partition_id in enumerate(active_ids)
    }
    if device.type == "cuda":
        # Rollout leaves large inactive allocator segments cached.  They are
        # reusable by PyTorch, but cudaMemGetInfo reports them as driver-used,
        # which would make the physical learner planner mistake cache slack for
        # live storage and collapse postflop subspans to singleton rows.  Return
        # only inactive cache blocks at this phase boundary before measuring the
        # learner workspace; live sealed storage remains allocated.
        torch.cuda.empty_cache()
    available_transient = _value_workspace_budget(cfg, device)
    physical_shapes: dict[int, tuple[PhysicalShape, ...]] = {}
    reference_build_shapes: dict[int, tuple[PhysicalShape, ...]] = {}
    for partition_id in active_ids:
        family_name = "pre" if partition_id == 0 else f"post_{partition_id + 1}"
        configured = int(cfg.rollout.family_batch_rows or statistical_counts[partition_id])
        row_cap = max(1, min(configured, statistical_counts[partition_id]))
        if cfg.hardware is not None:
            width = int(bundle.rollout_backend._family_maximum_width(source, family_name))
            token_cap = cfg.hardware.training_max_tokens // max(width, 1)
            pair_cap = cfg.hardware.training_max_token_pairs // max(width * width, 1)
            row_cap = max(1, min(row_cap, token_cap, pair_cap))
        predictor = getattr(bundle.rollout_backend, "training_transient_bytes", None)
        if not callable(predictor):
            raise TypeError("rollout backend lacks a learner workspace contract")
        while row_cap > 1 and predictor(
            source, family_name, row_cap, amp=amp_enabled
        ) > available_transient:
            row_cap = max(1, row_cap // 2)
        transient_bytes = predictor(
            source, family_name, row_cap, amp=amp_enabled
        )
        iteration_profile[
            f"partition_{partition_id}_physical_rows"
        ] = float(row_cap)
        iteration_profile[
            f"partition_{partition_id}_predicted_transient_bytes"
        ] = float(transient_bytes)
        physical_shapes[partition_id] = (
            PhysicalShape(row_cap, transient_bytes),
        )
        # Frozen-reference cache construction runs without autograd.  Its
        # measured row-11 workspace delta was less than half the training
        # delta, so admit a separately bounded shape up to twice the trained
        # row count (and never beyond the configured family pack).
        build_row_cap = min(configured, 2 * row_cap)
        build_transient = math.ceil(
            predictor(source, family_name, build_row_cap, amp=amp_enabled) / 2
        )
        while build_row_cap > row_cap and build_transient > available_transient:
            build_row_cap = max(row_cap, build_row_cap // 2)
            build_transient = math.ceil(
                predictor(source, family_name, build_row_cap, amp=amp_enabled) / 2
            )
        iteration_profile[
            f"partition_{partition_id}_reference_build_rows"
        ] = float(build_row_cap)
        iteration_profile[
            f"partition_{partition_id}_reference_build_predicted_transient_bytes"
        ] = float(build_transient)
        reference_build_shapes[partition_id] = (
            PhysicalShape(build_row_cap, build_transient),
        )
    learner_cfg = DistributedLearnerConfig(
        statistical_counts=statistical_counts,
        physical_shapes=physical_shapes,
        available_transient_bytes=available_transient,
        resident_base_bytes=retained_bytes,
        reference_build_shapes=reference_build_shapes,
        sampled_reference_cache=float(cfg.loss.c_kl) > 0.0,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    model.train()
    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    optimization_started = time.perf_counter()
    learner = agreement.run(
        "learner construction",
        lambda: DistributedFixedRolloutUpdate(
            source,
            optimizer,
            scaler,
            partition_specs,
            bundle.policy_adapter,
            cfg.loss,
            learner_cfg,
            reference_model=reference_model,
            failure_agreement=agreement,
        ),
    )
    if cfg.training.ppo_epochs is None:
        update_steps = optimizer_steps_this_iteration
    else:
        update_steps = cfg.training.ppo_epochs * math.ceil(
            decisions / cfg.training.minibatch_size
        )
    update = learner(
        storage,
        torch_gen,
        optimizer_steps=update_steps,
        iteration=iteration,
        grad_clip=cfg.training.grad_clip,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimization_seconds = time.perf_counter() - optimization_started
    metrics = dict(update.metrics)
    metrics.update(iteration_profile)
    metrics.update(memory_metrics)
    metrics.update(
        {
            "trajectories": float(total_trajectories),
            "decisions": float(decisions),
            "optimizer_steps": sum(
                float(step.optimizer_step_applied) for step in update.steps
            ),
            "minibatches": float(metrics.get("physical_subspans", 0.0)),
            "optimized_decision_rows": float(metrics.get("statistical_rows", 0.0)),
            "phase_optimization_seconds": optimization_seconds,
            "phase_optimization_host_seconds": optimization_seconds,
            "profiled_iteration_seconds": sum(
                float(iteration_profile.get(f"phase_{name}_seconds", 0.0))
                for name in ("snapshot", "rollout", "backward", "seal")
            ) + optimization_seconds,
            "target_bytes_per_decision": retained_bytes / max(decisions, 1),
        }
    )
    metrics["decisions_per_second"] = decisions / max(
        metrics["profiled_iteration_seconds"], 1.0e-12
    )
    metrics.update(_merge_completion_quality([(leaves, completion_quality)]))
    metrics.update(identity_metrics)
    if device.type == "cuda":
        metrics["phase_optimization_peak_allocated_bytes"] = float(
            metrics.get("training_peak_allocated_bytes", 0.0)
        )
        metrics["phase_optimization_peak_reserved_bytes"] = float(
            metrics.get("training_peak_reserved_bytes", 0.0)
        )
        allocated_peaks = [
            float(value)
            for key, value in metrics.items()
            if key.startswith("phase_") and key.endswith("_peak_allocated_bytes")
        ]
        reserved_peaks = [
            float(value)
            for key, value in metrics.items()
            if key.startswith("phase_") and key.endswith("_peak_reserved_bytes")
        ]
        metrics.update(
            {
                "iteration_peak_allocated_bytes": max(allocated_peaks),
                "iteration_peak_reserved_bytes": max(reserved_peaks),
                "iteration_end_allocated_bytes": float(torch.cuda.memory_allocated(device)),
                "iteration_end_reserved_bytes": float(torch.cuda.memory_reserved(device)),
            }
        )
        metrics["iteration_end_allocator_slack_bytes"] = (
            metrics["iteration_end_reserved_bytes"]
            - metrics["iteration_end_allocated_bytes"]
        )
    if execution_metrics is not None:
        snapshot = execution_metrics.snapshot()
        metrics.update(snapshot)
        totals = execution_metrics.totals()
        metrics["policy_physical_batches"] = float(totals.get("model_calls", 0))
        metrics["policy_planned_batches"] = float(totals.get("policy_pack_calls", 0))
        metrics["policy_total_padded_tokens"] = float(totals.get("padded_tokens", 0))
        metrics["policy_total_padded_token_pairs"] = float(
            totals.get("padded_token_pairs", 0)
        )
        metrics["policy_max_padded_tokens"] = snapshot["execution_padded_tokens_max"]
        metrics["policy_max_padded_token_pairs"] = snapshot[
            "execution_padded_token_pairs_max"
        ]
    counterfactual = metrics.get("counterfactual_tree_size")
    if counterfactual is not None:

        def commit_controller_observation() -> None:
            if branching_controller is None:
                raise RuntimeError("adaptive rollout has no KKT controller")
            metrics.update(branching_controller.observe(counterfactual))

        agreement.run(
            "adaptive KKT controller commit",
            commit_controller_observation,
        )
    del learner, storage
    return metrics



def run_training(
    cfg: RootConfig,
    *,
    bundle_builder: Callable[[RootConfig, torch.device], GameBundle],
    target_model_builder: Callable[[dict[str, Any], torch.device, Any | None], nn.Module]
    | None = None,
    resume_from: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
) -> None:
    """Train from scratch, initialize from model weights, or resume a run."""

    if resume_from is not None and init_checkpoint is not None:
        raise ValueError("resume_from and init_checkpoint are mutually exclusive")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    workload = "distributed training" if world > 1 else "training"
    requested = resolve_execution_device(
        cfg.device, workload=workload, local_rank=local_rank
    )
    validate_workload_capabilities(requested, workload=workload)
    if requested.type == "cuda" and cfg.hardware is None:
        raise RuntimeError(
            "CUDA training requires an explicit hardware profile with token, "
            "memory, and device limits"
        )
    if resume_from is not None:
        # Metadata-only CPU validation precedes bundle/model/optimizer CUDA
        # allocation and rejects incompatible partial state deterministically.
        inspect_checkpoint(resume_from)
    initialized_model_state: Mapping[str, Any] | None = None
    initialization_metadata: dict[str, Any] | None = None
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("model"), dict)
            or not isinstance(payload.get("config"), dict)
        ):
            raise RuntimeError("init checkpoint is not a PBG model checkpoint")
        source_config = payload["config"]
        source_game = source_config.get("game")
        expected_game = {
            "n_players": cfg.game.n_players,
            "stack_bb": cfg.game.stack_bb,
            "rank_count": cfg.game.rank_count,
        }
        if source_game != expected_game:
            raise RuntimeError("init checkpoint and target must use identical game settings")
        initialized_model_state, conversion, converted = _migrate_value_head_state(
            payload["model"], source_config, cfg
        )
        digest = hashlib.sha256()
        with Path(init_checkpoint).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        initialization_metadata = {
            "source_checkpoint": str(Path(init_checkpoint).resolve()),
            "source_checkpoint_sha256": digest.hexdigest(),
            "source_iteration": int(payload.get("iteration", -1)),
            "source_utility_normalization": source_config.get("loss", {}).get(
                "utility_normalization", "none"
            ),
            "target_utility_normalization": cfg.loss.utility_normalization,
            "value_head_conversion": conversion,
            "converted_tensors": list(converted),
        }
    target_payload: dict[str, Any] | None = None
    probe_target: dict[str, Any] | None = None
    if cfg.probe is not None:
        if target_model_builder is None:
            raise RuntimeError("probe training requires a target-model builder")
        target_path = Path(cfg.probe.target_checkpoint)
        digest = hashlib.sha256()
        with target_path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if (
            cfg.probe.target_sha256 is not None
            and cfg.probe.target_sha256 != actual_sha256
        ):
            raise RuntimeError("probe target checkpoint SHA-256 does not match configuration")
        cfg.probe.target_sha256 = actual_sha256
        target_payload = torch.load(target_path, map_location="cpu", weights_only=False)
        if not isinstance(target_payload, dict) or not isinstance(
            target_payload.get("config"), dict
        ) or "model" not in target_payload:
            raise RuntimeError("probe target is not a PBG model checkpoint")
        target_game = target_payload["config"].get("game")
        expected_game = {
            "n_players": cfg.game.n_players,
            "stack_bb": cfg.game.stack_bb,
            "rank_count": cfg.game.rank_count,
        }
        if target_game != expected_game:
            raise RuntimeError("probe and target checkpoints must use identical (P,S,R)")
        probe_target = {
            "semantics": SHARED_RESPONSE_SEMANTICS,
            "sha256": actual_sha256,
            "path": str(target_path),
            "run_id": target_payload.get("run_id"),
            "iteration": int(target_payload.get("iteration", -1)),
        }
    _configure_reproducibility(cfg.training.reproducibility)
    local_cfg = _rank_local_config(cfg, rank, world)
    allocation_cfg = local_cfg
    sequential_memory_control = cfg.training.mode != "branched_rao_blackwell"
    if sequential_memory_control:
        initial_per_gpu = int(cfg.rollout.sequential_roots_per_gpu_initial)
        maximum = int(cfg.rollout.sequential_roots_per_gpu_maximum)
        local_cfg.rollout.roots_per_gpu = initial_per_gpu
        local_cfg.rollout.trajectory_rank_counts = (initial_per_gpu,) * world
        allocation_cfg = copy.deepcopy(local_cfg)
        allocation_cfg.rollout.roots_per_gpu = maximum
        allocation_cfg.rollout.trajectory_rank_counts = (maximum,) * world
    _hardware_metadata(cfg, requested, world)
    torch_gen, np_gen = seed_all(cfg.training.seed, rank, requested)
    bundle = bundle_builder(allocation_cfg, requested)
    if not isinstance(bundle, GameBundle):
        raise TypeError("bundle builder must return GameBundle")
    device = torch.device(bundle.rollout_backend.device)
    if device != requested:
        raise RuntimeError(
            f"bundle rollout device {device} differs from selected device {requested}"
        )
    torch.cuda.set_device(device)
    if torch_gen.device != device:
        torch_gen, np_gen = seed_all(cfg.training.seed, rank, device)
    bundle.model.to(device)
    assert_module_device(bundle.model, device, "training model")
    if initialized_model_state is not None:
        try:
            bundle.model.load_state_dict(initialized_model_state)
        except RuntimeError as exc:
            if cfg.training.mode != "branched_rao_blackwell":
                raise RuntimeError(
                    "init checkpoint model architecture does not match the current config"
                ) from exc
            incompatible = bundle.model.load_state_dict(
                initialized_model_state, strict=False
            )
            allowed_missing = {
                name for name in bundle.model.state_dict()
                if ".branching_heads." in name or name.startswith("branching_heads.")
            }
            missing = set(incompatible.missing_keys)
            if incompatible.unexpected_keys or missing != allowed_missing:
                raise RuntimeError(
                    "init checkpoint is incompatible beyond absent T/C branching heads; "
                    f"missing={sorted(missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
                ) from exc
            warnings.warn(
                "initialized T/C branching heads from current defaults because the explicit "
                "init checkpoint predates them; optimizer/controller state starts fresh",
                RuntimeWarning,
                stacklevel=2,
            )
    if target_payload is not None:
        assert target_model_builder is not None and cfg.probe is not None
        opponent = target_model_builder(
            target_payload["config"], device, cfg.hardware
        )
        opponent.load_state_dict(target_payload["model"])
        opponent.eval().requires_grad_(False)
        bundle.opponent_model = opponent
        register_policy_model = getattr(
            bundle.rollout_backend, "register_policy_model", None
        )
        if callable(register_policy_model):
            register_policy_model(opponent)
            register_policy_model(bundle.model)
        bundle.probe_target = probe_target
        if cfg.probe.initialization == "target":
            try:
                bundle.model.load_state_dict(target_payload["model"])
            except RuntimeError as exc:
                raise RuntimeError(
                    "probe initialization 'target' requires identical architectures"
                ) from exc
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        # Each rank owns a plain model replica. Gradients are accumulated under
        # the exact local objective scale and reduced once through the explicit
        # flat SUM path; DDP's equal-rank averaging is intentionally not used.
        # Rank-addressed rollout RNG must not imply rank-addressed model
        # initialization: establish one identical learner before any rollout.
        for value in bundle.model.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer_cls = (
        torch.optim.AdamW
        if cfg.training.optimizer == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_cls(
        bundle.model.parameters(), lr=cfg.training.lr, betas=cfg.training.adam_betas
    )
    source_model = (
        bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    )
    reference_model: nn.Module | None = None
    reference_iteration: int | None = None
    if cfg.loss.c_kl > 0.0:
        reference_model = copy.deepcopy(source_model).eval()
        reference_model.requires_grad_(False)
    amp_enabled, amp_dtype = _precision(cfg, device)
    scaler = _scaler(bool(amp_enabled and amp_dtype == torch.float16))
    out_dir = Path(cfg.training.out_dir)
    start_iteration = 0
    run_id: str | None = None
    restored_memory_state: Mapping[str, Any] | None = None
    restored_branching_state: Mapping[str, Any] | None = None
    resume_path = None if resume_from is None else Path(resume_from)
    logical_work = {
        "trajectories": 0,
        "completion_samples": 0,
        "minibatches": 0,
        "optimizer_steps": 0,
    }

    if resume_path is not None:
        if resume_path.parent.resolve() != out_dir.resolve():
            raise RuntimeError("resume checkpoint must belong to training.out_dir")
        checkpoint_metadata = load_checkpoint(
            resume_path,
            bundle.model,
            optimizer,
            scaler,
            torch_gen,
            np_gen,
            map_location=device,
            rank=rank,
            reference_model=reference_model,
        )
        stored_logical_work = checkpoint_metadata.get("logical_work")
        if stored_logical_work is not None:
            if set(stored_logical_work) != set(logical_work):
                raise RuntimeError("checkpoint has malformed logical-work counters")
            logical_work = {
                key: int(stored_logical_work[key]) for key in logical_work
            }
        if cfg.probe is not None:
            stored_target = checkpoint_metadata.get("probe_target")
            if stored_target is None:
                raise RuntimeError("resume checkpoint has no probe target metadata")
            if stored_target.get("semantics") != SHARED_RESPONSE_SEMANTICS:
                raise RuntimeError("resume checkpoint has incompatible probe semantics")
            if stored_target.get("sha256") != cfg.probe.target_sha256:
                raise RuntimeError("resume probe target hash does not match checkpoint")
        if not compatible_scientific_configs(
            checkpoint_metadata["cfg_dict"], _scientific_config(cfg)
        ):
            raise RuntimeError("resume configuration does not match the checkpoint")
        start_iteration = int(checkpoint_metadata["iter"])
        if not 0 <= start_iteration <= cfg.training.iters:
            raise RuntimeError("checkpoint iteration is outside the configured schedule")
        run_id = checkpoint_metadata["run_id"]
        reference_iteration = checkpoint_metadata["reference_iteration"]
        initialization_metadata = checkpoint_metadata.get("initialization")
        if sequential_memory_control:
            restored_memory_state = checkpoint_metadata["adaptive_work_state"]
            if restored_memory_state is None:
                raise RuntimeError("sequential resume checkpoint has no memory controller state")
        if cfg.rollout.branching_mode == "dynamic":
            restored_branching_state = checkpoint_metadata.get(
                "branching_controller_state"
            )
            if restored_branching_state is None:
                raise RuntimeError(
                    "adaptive KKT resume checkpoint has no branching controller state"
                )
        elif checkpoint_metadata.get("branching_controller_state") is not None:
            raise RuntimeError(
                "fixed branching resume checkpoint unexpectedly contains dynamic "
                "controller state"
            )
        if not run_id:
            raise RuntimeError("checkpoint predates run identities and cannot be resumed safely")
        if device.type == "cuda":
            # torch.load(map_location=device) temporarily materializes checkpoint
            # tensors before load_state_dict copies them into the live model and
            # optimizer. Release those now-dead cache blocks so resume does not
            # carry an artificial allocator peak into rollout calibration.
            torch.cuda.empty_cache()

    memory_plan = _analytical_memory_plan(
        allocation_cfg, bundle, has_reference=reference_model is not None
    )
    _validate_memory_preflight(allocation_cfg, device, memory_plan)
    memory_controller: SequentialMemoryController | None = None
    branching_controller: AdaptiveKKTController | None = None
    adaptive_cfg = cfg.rollout.adaptive_kkt
    if cfg.rollout.branching_mode == "dynamic":
        if adaptive_cfg is None:
            raise RuntimeError("dynamic branching has no adaptive KKT configuration")
        branching_controller = (
            AdaptiveKKTController.from_state_dict(restored_branching_state)
            if restored_branching_state is not None
            else AdaptiveKKTController.from_config(adaptive_cfg)
        )
    if sequential_memory_control:
        maximum = int(cfg.rollout.sequential_roots_per_gpu_maximum)
        phase_parts = {
            "rollout": (
                memory_plan["memory_plan_rollout_fixed_bytes"],
                memory_plan["memory_plan_rollout_variable_bytes"],
            ),
            "seal": (
                memory_plan["memory_plan_seal_fixed_bytes"],
                memory_plan["memory_plan_seal_variable_bytes"],
            ),
            "targets": (
                memory_plan["memory_plan_target_fixed_bytes"],
                memory_plan["memory_plan_target_variable_bytes"],
            ),
            "optimization": (
                memory_plan["memory_plan_optimization_fixed_bytes"],
                memory_plan["memory_plan_optimization_variable_bytes"],
            ),
        }
        phase_totals = {
            phase: fixed + variable
            for phase, (fixed, variable) in phase_parts.items()
        }
        analytic = {
            phase: PhaseMemoryModel(
                fixed_bytes=fixed,
                upper_bytes_per_root={
                    "analytic_schema_and_workspace": variable / maximum
                },
            )
            for phase, (fixed, variable) in phase_parts.items()
        }
        if device.type == "cuda":
            _, device_bytes = torch.cuda.mem_get_info(device)
            hard = _memory_limit(cfg, device, account_current_free=False)
            guard_bytes = int(device_bytes) - int(hard)
        else:
            # CPU runs exercise controller state/geometry without a device
            # allocator ceiling.  Leave ample synthetic headroom while still
            # using the same four phase observations and hysteresis logic.
            device_bytes = 8 * max(phase_totals.values()) + 1024
            guard_bytes = 0
        memory_controller = (
            SequentialMemoryController.from_state_dict(restored_memory_state)
            if restored_memory_state is not None
            else SequentialMemoryController(
                minimum_roots_per_gpu=int(
                    cfg.rollout.sequential_roots_per_gpu_minimum
                ),
                maximum_roots_per_gpu=maximum,
                current_roots_per_gpu=int(
                    cfg.rollout.sequential_roots_per_gpu_initial
                ),
                device_bytes=int(device_bytes),
                guard_bytes=guard_bytes,
                forecaster=PhaseMemoryForecaster(
                    analytic,
                    window=int(cfg.rollout.memory_calibration_window),
                    prediction_z=float(cfg.rollout.memory_prediction_z),
                ),
                growth_fraction=float(cfg.rollout.memory_growth_fraction),
                growth_delay=int(cfg.rollout.memory_growth_delay),
            )
        )
        local_cfg.rollout.roots_per_gpu = memory_controller.current_roots_per_gpu
        local_cfg.rollout.trajectory_rank_counts = (
            memory_controller.current_roots_per_gpu,
        ) * world

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_path = out_dir / "run.json"
        if resume_path is None:
            occupied = any(
                path.exists()
                for path in (run_path, out_dir / "metrics.jsonl")
            ) or any(out_dir.glob("ckpt_*.pt"))
            if occupied:
                raise RuntimeError(
                    "output directory already contains a run; pass resume_from explicitly"
                )
            run_id = uuid.uuid4().hex
            run_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "config_sha256": _config_fingerprint(cfg),
                        "world_size": world,
                        "probe_target": probe_target,
                        "initialization": initialization_metadata,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            if not run_path.exists():
                raise RuntimeError("resume requires the existing run.json")
            run_record = json.loads(run_path.read_text(encoding="utf-8"))
            if run_record.get("run_id") != run_id:
                raise RuntimeError("checkpoint belongs to a different run identity")
            recorded_fingerprint = run_record.get("config_sha256")
            if recorded_fingerprint not in {
                _config_fingerprint(cfg),
                legacy_config_fingerprint(cfg),
                config_dict_fingerprint(checkpoint_metadata["cfg_dict"]),
            } and not matches_legacy_run_fingerprint(
                recorded_fingerprint, checkpoint_metadata["cfg_dict"]
            ):
                raise RuntimeError("run configuration fingerprint does not match")
            if run_record.get("world_size") != world:
                raise RuntimeError("resume requires the original DDP world size")
            _validate_metrics_prefix(
                out_dir / "metrics.jsonl", start_iteration, run_id
            )
            for memory_rank in range(world):
                memory_path = out_dir / f"memory_rank_{memory_rank}.jsonl"
                if memory_path.exists():
                    _validate_metrics_prefix(memory_path, start_iteration, run_id)
        metadata = _hardware_metadata(cfg, device, world)
        metadata.update(memory_plan)
        hardware_name = (
            "hardware.json"
            if resume_path is None
            else f"hardware_resume_{start_iteration:07d}.json"
        )
        with (out_dir / hardware_name).open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
    if world > 1:
        identity = [run_id]
        dist.broadcast_object_list(identity, src=0)
        run_id = identity[0]
        dist.barrier()
    assert run_id is not None
    failure_agreement = DistributedFailureAgreement(device)

    for iteration in range(start_iteration, cfg.training.iters):
        if memory_controller is not None:
            try:
                current = memory_controller.choose_root_width()
            except MemoryAdmissionStop as exc:
                # No current-iteration draw has occurred. Persist the unchanged
                # RNG/model state and the conservative controller evidence,
                # then stop without narrowing and rerolling the iteration.
                local_rng_state = _capture_rng_state(torch_gen, np_gen)
                if world > 1:
                    gathered_stop: list[dict[str, Any] | None] | None = (
                        [None] * world if rank == 0 else None
                    )
                    dist.gather_object(local_rng_state, gathered_stop, dst=0)
                    rank_rng_states = (
                        None
                        if rank != 0
                        else [item for item in gathered_stop or [] if item is not None]
                    )
                else:
                    rank_rng_states = [local_rng_state]
                if rank == 0:
                    save_checkpoint(
                        out_dir / f"ckpt_memory_stop_{iteration:07d}.pt",
                        bundle.model,
                        optimizer,
                        scaler,
                        iteration,
                        cfg,
                        torch_gen,
                        np_gen,
                        run_id=run_id,
                        rank_rng_states=rank_rng_states,
                        reference_model=reference_model,
                        reference_iteration=reference_iteration,
                        adaptive_work_state=memory_controller.state_dict(),
                        branching_controller_state=(
                            None if branching_controller is None
                            else branching_controller.state_dict()
                        ),
                        probe_target=probe_target,
                        logical_work=logical_work,
                        initialization=initialization_metadata,
                    )
                    (out_dir / "memory_stop.json").write_text(
                        json.dumps(
                            {
                                "iter": iteration,
                                "run_id": run_id,
                                "reason": str(exc),
                                "roots_per_gpu": memory_controller.current_roots_per_gpu,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                break
            local_cfg.rollout.roots_per_gpu = current
            local_cfg.rollout.trajectory_rank_counts = (current,) * world
        if reference_model is not None and iteration % cfg.training.kl_ref_interval == 0:
            reference_model.load_state_dict(source_model.state_dict())
            reference_model.eval()
            reference_iteration = iteration
        try:
            metrics = failure_agreement.run(
                "iteration execution",
                lambda: _one_iteration(
                    local_cfg,
                    bundle,
                    optimizer,
                    scaler,
                    torch_gen,
                    iteration,
                    profile=True,
                    reference_model=reference_model,
                    validate_identity=iteration == start_iteration,
                    branching_controller=branching_controller,
                    failure_agreement=failure_agreement,
                ),
            )
        except DistributedPhaseFailure as exc:
            # Every rank reaches this path with the same phase and first-failing
            # rank.  The audited mandatory boundaries precede the first update,
            # so their current model/optimizer state is exactly the last
            # completed iteration and can be checkpointed without rollback.
            checkpoint_safe = exc.phase in {
                "rollout, seal, completion, and targets",
                "adaptive KKT controller observation",
                "learner construction",
                "mandatory learner singleton feasibility",
                "local backward step 0",
            }
            local_rng_state = _capture_rng_state(torch_gen, np_gen)
            if world > 1:
                gathered_failure: list[dict[str, Any] | None] | None = (
                    [None] * world if rank == 0 else None
                )
                dist.gather_object(local_rng_state, gathered_failure, dst=0)
                gathered_details: list[str | None] | None = (
                    [None] * world if rank == 0 else None
                )
                dist.gather_object(exc.local_detail, gathered_details, dst=0)
                rank_rng_states = (
                    None
                    if rank != 0
                    else [
                        item
                        for item in gathered_failure or []
                        if item is not None
                    ]
                )
                failure_detail = (
                    None
                    if rank != 0 or gathered_details is None
                    else gathered_details[exc.failing_rank]
                )
            else:
                rank_rng_states = [local_rng_state]
                failure_detail = exc.local_detail
            checkpoint_name: str | None = None
            if rank == 0 and checkpoint_safe:
                checkpoint_name = f"ckpt_terminal_stop_{iteration:07d}.pt"
                save_checkpoint(
                    out_dir / checkpoint_name,
                    bundle.model,
                    optimizer,
                    scaler,
                    iteration,
                    cfg,
                    torch_gen,
                    np_gen,
                    run_id=run_id,
                    rank_rng_states=rank_rng_states,
                    reference_model=reference_model,
                    reference_iteration=reference_iteration,
                    adaptive_work_state=(
                        None
                        if memory_controller is None
                        else memory_controller.state_dict()
                    ),
                    branching_controller_state=(
                        None
                        if branching_controller is None
                        else branching_controller.state_dict()
                    ),
                    probe_target=probe_target,
                    logical_work=logical_work,
                    initialization=initialization_metadata,
                )
            if rank == 0:
                (out_dir / "terminal_stop.json").write_text(
                    json.dumps(
                        {
                            "iter": iteration,
                            "run_id": run_id,
                            "phase": exc.phase,
                            "failing_rank": exc.failing_rank,
                            "detail": failure_detail,
                            "checkpoint": checkpoint_name,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            break
        if resume_path is not None and iteration == start_iteration:
            for key in tuple(metrics):
                if key.startswith("identity_"):
                    metrics.pop(key)
        if iteration == 0:
            metrics.update({key: float(value) for key, value in memory_plan.items()})
        diagnostic = {
            key: value
            for key, value in metrics.items()
            if (
                key.startswith("memory_")
                or key.startswith("phase_") and "peak_" in key
                or key.startswith("execution_")
                and any(
                    marker in key
                    for marker in (
                        "model_calls",
                        "model_rows",
                        "padded_token",
                        "queue_",
                        "forced_drain",
                        "completion_",
                    )
                )
                or key.startswith("trajectory_length_")
                or key in {"nodes", "leaves", "decisions", "retained_target_bytes"}
            )
        }
        with (out_dir / f"memory_rank_{rank}.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        "iter": iteration,
                        "run_id": run_id,
                        "rank": rank,
                        "optimizer_step": logical_work["optimizer_steps"],
                        "entered_learning": True,
                        **diagnostic,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _add_distributed_metric_summaries(metrics, device, world)
        _synchronize_memory_control_metrics(
            metrics, device, world, int(bundle.rollout_backend.n_players)
        )
        _validate_memory_calibration(cfg, device, metrics)
        if memory_controller is not None:
            predicted = memory_controller.last_predicted_peaks
            fallback = int(
                metrics.get(
                    "iteration_peak_reserved_bytes",
                    max(predicted.values()),
                )
            )
            actual_keys = {
                # Allocated phase peaks are ownership/shape evidence and are
                # invariant to allocator-bin history across checkpoint resume.
                # Reserved/NVML peaks remain hard-guard diagnostics, but must
                # not perturb the deterministic next-root decision.
                "rollout": "phase_rollout_peak_allocated_bytes",
                "seal": "phase_seal_peak_allocated_bytes",
                "targets": "phase_backward_peak_allocated_bytes",
                "optimization": "phase_optimization_peak_allocated_bytes",
            }
            phase_observations = {}
            for phase in MEMORY_PHASES:
                actual = int(metrics.get(actual_keys[phase], fallback))
                expected = int(predicted[phase])
                analytic_fixed = memory_controller.forecaster.analytic_models[
                    phase
                ].fixed_bytes
                observed_fixed = min(analytic_fixed, actual)
                phase_observations[phase] = PhaseMemoryObservation(
                    roots_per_gpu=memory_controller.current_roots_per_gpu,
                    fixed_bytes=observed_fixed,
                    variable_bytes={
                        "analytic_schema_and_workspace": actual - observed_fixed
                    },
                    predicted_peak_bytes=expected,
                    actual_peak_bytes=actual,
                )
            metrics.update(memory_controller.observe_success(phase_observations))
        if reference_iteration is not None:
            metrics["reference_iteration"] = float(reference_iteration)
        for name in logical_work:
            distributed_key = (
                f"{name}_rank_max"
                if name == "optimizer_steps"
                else f"{name}_rank_sum"
            )
            value = metrics.get(
                distributed_key if world > 1 else name,
                0.0,
            )
            logical_work[name] += int(round(value))
            metrics[f"cumulative_{name}"] = float(logical_work[name])
        if "decisions_rank_sum" in metrics:
            global_decisions = max(metrics["decisions_rank_sum"], 1.0)
            for street in ("preflop", "flop", "turn", "river"):
                key = f"decision_nodes_{street}_rank_sum"
                if key in metrics:
                    metrics[f"decision_fraction_{street}_global"] = (
                        metrics[key] / global_decisions
                    )
        checkpoint_every = (
            cfg.training.checkpoint_every
            if cfg.hardware is None or cfg.hardware.checkpoint_every is None
            else cfg.hardware.checkpoint_every
        )
        checkpoint_due = (iteration + 1) % checkpoint_every == 0
        if rank == 0:
            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"iter": iteration, "run_id": run_id, **metrics},
                        sort_keys=True,
                    )
                    + "\n"
                )
        rank_rng_states: list[dict[str, Any]] | None = None
        if checkpoint_due:
            local_rng_state = _capture_rng_state(torch_gen, np_gen)
            if world > 1:
                gathered: list[dict[str, Any] | None] | None = (
                    [None] * world if rank == 0 else None
                )
                dist.gather_object(local_rng_state, gathered, dst=0)
                if rank == 0:
                    assert gathered is not None and all(item is not None for item in gathered)
                    rank_rng_states = [item for item in gathered if item is not None]
            else:
                rank_rng_states = [local_rng_state]
        if rank == 0 and checkpoint_due:
            save_checkpoint(
                out_dir / f"ckpt_{iteration + 1:07d}.pt",
                bundle.model,
                optimizer,
                scaler,
                iteration + 1,
                cfg,
                torch_gen,
                np_gen,
                run_id=run_id,
                rank_rng_states=rank_rng_states,
                reference_model=reference_model,
                reference_iteration=reference_iteration,
                adaptive_work_state=(
                    None
                    if memory_controller is None
                    else memory_controller.state_dict()
                ),
                branching_controller_state=(
                    None if branching_controller is None
                    else branching_controller.state_dict()
                ),
                probe_target=probe_target,
                logical_work=logical_work,
                initialization=initialization_metadata,
            )
    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
