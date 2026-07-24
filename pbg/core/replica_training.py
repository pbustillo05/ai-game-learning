"""Single-CUDA-process training for independent replica-batched experiments."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Callable
import uuid

import numpy as np
import torch

from pbg.core.checkpointing import scientific_config
from pbg.core.config import RootConfig, seed_all
from pbg.core.game import GameBundle
from pbg.core.learner import (
    DistributedFixedRolloutUpdate,
    DistributedLearnerConfig,
    PhysicalShape,
    allocate_fixed_statistical_counts,
    posterior_effective_hand_counts,
)
from pbg.core.preemption import PreemptionController
from pbg.core.replica_checkpoint import ReplicaCheckpointManager
from pbg.core.replica_learner import FunctionalReplicaLearner, StackedModuleReplicas
from pbg.core.replica_update import ReplicaFixedRolloutUpdate
from pbg.core.training import (
    _build_iteration_data,
    _configure_reproducibility,
    _precision,
    _scaler,
    _value_workspace_budget,
)
from pbg.games.poker.replica import PokerReplicaPartition


def _cuda_kernel_count(profile: torch.profiler.profile) -> int:
    cuda_type = torch.autograd.DeviceType.CUDA
    return sum(
        1 for event in profile.events() if event.device_type == cuda_type
    )


def _lane_config(cfg: RootConfig, lane: int) -> RootConfig:
    result = copy.deepcopy(cfg)
    result.training.out_dir = str(
        Path(cfg.training.out_dir) / f"replica_{lane:03d}"
    )
    roots = int(result.rollout.sequential_roots_per_gpu_initial)
    result.rollout.roots_per_gpu = roots
    result.rollout.trajectory_rank_counts = (roots,)
    return result


def _lane_counts(
    cfg: RootConfig, bundle: GameBundle, storage: object, model: torch.nn.Module
) -> dict[int, int]:
    active_ids = tuple(sorted(storage.partitions))
    if not active_ids:
        raise ValueError("cannot optimize an empty replica fixed rollout")
    if cfg.training.minibatch_size < len(active_ids):
        raise ValueError("minibatch size cannot cover every active family")
    logs = torch.stack(
        [storage.sampling.log_z[partition_id] for partition_id in active_ids]
    )
    mass = torch.exp(logs - torch.logsumexp(logs, 0))
    widths = torch.tensor(
        [
            bundle.rollout_backend._partition_maximum_width(model, partition_id)
            for partition_id in active_ids
        ],
        dtype=torch.float64,
        device=logs.device,
    )
    allocated = allocate_fixed_statistical_counts(
        mass,
        torch.ones_like(mass),
        widths.square(),
        cfg.training.minibatch_size,
    )
    active = {
        partition_id: int(allocated[position])
        for position, partition_id in enumerate(active_ids)
    }
    return {
        spec.partition_id: active.get(spec.partition_id, 1)
        for spec in bundle.rollout_backend.training_partition_specs().specs
    }


def _physical_config(
    cfg: RootConfig,
    bundles: list[GameBundle],
    counts: list[dict[int, int]],
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    retained_bytes: int,
) -> DistributedLearnerConfig:
    device = torch.device(bundles[0].rollout_backend.device)
    available = _value_workspace_budget(cfg, device)
    shapes: dict[int, tuple[PhysicalShape, ...]] = {}
    reference_shapes: dict[int, tuple[PhysicalShape, ...]] = {}
    specs = bundles[0].rollout_backend.training_partition_specs()
    for spec in specs.specs:
        partition_id = spec.partition_id
        configured = int(
            cfg.rollout.family_batch_rows
            or max(plan[partition_id] for plan in counts)
        )
        row_cap = max(
            1,
            min(
                configured,
                max(plan[partition_id] for plan in counts),
            ),
        )
        width = max(
            bundle.rollout_backend._partition_maximum_width(
                bundle.model, partition_id
            )
            for bundle in bundles
        )
        if cfg.hardware is not None:
            row_cap = max(
                1,
                min(
                    row_cap,
                    cfg.hardware.training_max_tokens // max(width, 1),
                    cfg.hardware.training_max_token_pairs // max(width * width, 1),
                ),
            )

        def aggregate_bytes(rows: int) -> int:
            return sum(
                int(
                    bundle.rollout_backend.training_transient_bytes(
                        bundle.model,
                        partition_id,
                        rows,
                        amp=amp_enabled,
                    )
                )
                for bundle in bundles
            )

        while row_cap > 1 and aggregate_bytes(row_cap) > available:
            row_cap = max(1, row_cap // 2)
        transient = aggregate_bytes(row_cap)
        if transient > available:
            raise MemoryError(
                f"replica partition {partition_id} singleton learner cannot fit"
            )
        shapes[partition_id] = (PhysicalShape(row_cap, transient),)
        build_rows = min(configured, 2 * row_cap)
        build_transient = math.ceil(aggregate_bytes(build_rows) / 2)
        while build_rows > row_cap and build_transient > available:
            build_rows = max(row_cap, build_rows // 2)
            build_transient = math.ceil(aggregate_bytes(build_rows) / 2)
        reference_shapes[partition_id] = (
            PhysicalShape(build_rows, build_transient),
        )
    return DistributedLearnerConfig(
        statistical_counts=counts[0],
        physical_shapes=shapes,
        available_transient_bytes=available,
        resident_base_bytes=retained_bytes,
        reference_build_shapes=reference_shapes,
        sampled_reference_cache=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )


def _restore_explicit_rng(
    payload: dict[str, object],
    torch_generator: torch.Generator,
    numpy_generator: np.random.Generator,
) -> None:
    torch_generator.set_state(payload["torch_generator"].cpu())
    numpy_generator.bit_generator.state = payload["numpy_generator"]


def run_replica_training(
    cfg: RootConfig,
    *,
    replicas: int,
    bundle_builder: Callable[[RootConfig, torch.device], GameBundle],
    resume_root: str | Path | None = None,
    max_iterations: int | None = None,
    profile_kernel_iterations: frozenset[int] = frozenset(),
    write_checkpoints: bool = True,
    verify_legacy_k1: bool = False,
    learner_benchmark_warmup: int | None = None,
    learner_benchmark_steps: int | None = None,
) -> None:
    """Run $K$ independent Normal or sequential-RB lanes on one visible GPU."""

    if isinstance(replicas, bool) or replicas <= 0:
        raise ValueError("replica count K must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "replica training launches one process per GPU; do not nest torchrun"
        )
    if cfg.training.mode not in {"normal", "rao_blackwell"}:
        raise ValueError("replica runtime supports only Normal and sequential RB")
    if cfg.training.optimizer != "adamw":
        raise ValueError("replica runtime currently requires the frozen AdamW setup")
    if cfg.training.ppo_epochs is not None:
        raise ValueError("replica runtime currently requires fixed gd_steps_per_iter")
    if cfg.probe is not None:
        raise ValueError("replica runtime does not mix BR-probe target ownership")
    if cfg.hardware is None:
        raise RuntimeError("replica CUDA training requires a hardware profile")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive or None")
    benchmark_requested = (
        learner_benchmark_warmup is not None
        or learner_benchmark_steps is not None
    )
    if benchmark_requested and (
        learner_benchmark_warmup is None
        or learner_benchmark_steps is None
        or learner_benchmark_warmup < 0
        or learner_benchmark_steps <= 0
    ):
        raise ValueError("learner benchmark requires nonnegative warmup and positive steps")
    if benchmark_requested and verify_legacy_k1:
        raise ValueError("legacy parity and learner benchmark are separate stages")
    if verify_legacy_k1 and replicas != 1:
        raise ValueError("legacy parity requires K=1")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    _configure_reproducibility(cfg.training.reproducibility)
    amp_enabled, amp_dtype = _precision(cfg, device)
    if amp_enabled and amp_dtype == torch.float16:
        raise ValueError(
            "replica AdamW has no shared loss scaler; use the H100 BF16 profile"
        )

    lane_cfgs = [_lane_config(cfg, lane) for lane in range(replicas)]
    generators: list[torch.Generator] = []
    numpy_generators: list[np.random.Generator] = []
    bundles: list[GameBundle] = []
    wrappers: list[PokerReplicaPartition] = []
    legacy_model: torch.nn.Module | None = None
    for lane, lane_cfg in enumerate(lane_cfgs):
        torch_generator, numpy_generator = seed_all(
            cfg.training.seed, lane, device
        )
        generators.append(torch_generator)
        numpy_generators.append(numpy_generator)
        bundle = bundle_builder(lane_cfg, device)
        if verify_legacy_k1:
            legacy_model = copy.deepcopy(bundle.model).train()
        bundles.append(bundle)
        wrappers.append(
            PokerReplicaPartition(
                bundle.model,
                partition_id=0,
                n_players=cfg.game.n_players,
                rank_count=cfg.game.rank_count,
            )
        )

    stacked = StackedModuleReplicas.from_modules(wrappers)
    reference = stacked.frozen_copy() if float(cfg.loss.c_kl) > 0.0 else None
    legacy_reference = (
        None
        if legacy_model is None or float(cfg.loss.c_kl) <= 0.0
        else copy.deepcopy(legacy_model).eval().requires_grad_(False)
    )
    legacy_optimizer = (
        None
        if legacy_model is None
        else torch.optim.AdamW(
            legacy_model.parameters(),
            lr=cfg.training.lr,
            betas=cfg.training.adam_betas,
            weight_decay=0.01,
        )
    )
    for lane, bundle in enumerate(bundles):
        lane_wrapper = stacked.lane_module(lane)
        bundle.model = lane_wrapper.bundle
        register = getattr(bundle.rollout_backend, "register_policy_model", None)
        if callable(register):
            register(bundle.model)
    del wrappers
    gc.collect()
    torch.cuda.empty_cache()

    learner = FunctionalReplicaLearner(
        stacked,
        lr=cfg.training.lr,
        betas=cfg.training.adam_betas,
        weight_decay=0.01,
        grad_clip=cfg.training.grad_clip,
    )
    checkpoint_manager = ReplicaCheckpointManager(
        learner, reference=reference
    )
    scalers = [_scaler(False) for _ in range(replicas)]
    run_ids = [str(uuid.uuid4()) for _ in range(replicas)]
    reference_iterations: list[int | None] = [
        0 if reference is not None else None for _ in range(replicas)
    ]
    reference_exposure = [0.0 for _ in range(replicas)]
    logical_work = [
        {
            "trajectories": 0,
            "completion_samples": 0,
            "minibatches": 0,
            "optimizer_steps": 0,
        }
        for _ in range(replicas)
    ]
    start_iteration = 0
    if resume_root is not None:
        starts = []
        for lane, lane_cfg in enumerate(lane_cfgs):
            lane_root = Path(resume_root) / f"replica_{lane:03d}"
            checkpoints = sorted(lane_root.glob("ckpt_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(
                    f"replica {lane} has no checkpoint under {lane_root}"
                )
            payload = checkpoint_manager.load_lane(
                checkpoints[-1], lane, map_location=device
            )
            if payload["config"] != scientific_config(lane_cfg):
                raise RuntimeError(f"replica {lane} resume config differs")
            _restore_explicit_rng(
                payload, generators[lane], numpy_generators[lane]
            )
            starts.append(int(payload["iteration"]))
            run_ids[lane] = payload["run_id"]
            reference_iterations[lane] = payload["reference_iteration"]
            reference_exposure[lane] = float(
                payload["reference_effective_hands_rollout"]
            )
            stored_work = payload["logical_work"]
            if stored_work is not None:
                logical_work[lane] = {
                    name: int(stored_work[name]) for name in logical_work[lane]
                }
        if len(set(starts)) != 1:
            raise RuntimeError("replica checkpoints must share one optimizer boundary")
        start_iteration = starts[0]

    for lane_cfg in lane_cfgs:
        out_dir = Path(lane_cfg.training.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    preemption = PreemptionController()
    preemption.install()
    try:
        stop_iteration = cfg.training.iters
        if max_iterations is not None:
            stop_iteration = min(
                stop_iteration, start_iteration + max_iterations
            )
        for iteration in range(start_iteration, stop_iteration):
            iteration_started = time.perf_counter()
            iteration_profile = None
            if iteration in profile_kernel_iterations:
                iteration_profile = torch.profiler.profile(
                    activities=(
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    )
                )
                iteration_profile.__enter__()
            if reference is not None and cfg.training.kl_ref_effective_hands_rollout is None:
                if iteration % cfg.training.kl_ref_interval == 0:
                    for lane in range(replicas):
                        reference.copy_lane_from(lane, stacked, lane)
                        reference_iterations[lane] = iteration
            built = [
                _build_iteration_data(
                    lane_cfgs[lane],
                    bundles[lane],
                    generators[lane],
                    profile=True,
                    iteration=iteration,
                )
                for lane in range(replicas)
            ]
            partition_specs = built[0].partition_specs
            layout = tuple(
                (
                    spec.partition_id,
                    spec.exact_width,
                    spec.model_width,
                    spec.exact_to_model,
                    spec.input_schema,
                )
                for spec in partition_specs.specs
            )
            if any(
                tuple(
                    (
                        spec.partition_id,
                        spec.exact_width,
                        spec.model_width,
                        spec.exact_to_model,
                        spec.input_schema,
                    )
                    for spec in item.partition_specs.specs
                )
                != layout
                for item in built[1:]
            ):
                raise RuntimeError("replica games produced different partition schemas")
            storages = [item.storage for item in built]
            counts = [
                _lane_counts(
                    lane_cfgs[lane],
                    bundles[lane],
                    storages[lane],
                    bundles[lane].model,
                )
                for lane in range(replicas)
            ]
            torch.cuda.empty_cache()
            physical = _physical_config(
                cfg,
                bundles,
                counts,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                retained_bytes=sum(
                    storage.tensor_storage_bytes() for storage in storages
                ),
            )
            learner_views = {
                spec.partition_id: stacked.with_module(
                    stacked.module.with_partition(spec.partition_id)
                )
                for spec in partition_specs.specs
            }
            reference_views = (
                None
                if reference is None
                else {
                    spec.partition_id: reference.with_module(
                        reference.module.with_partition(spec.partition_id)
                    )
                    for spec in partition_specs.specs
                }
            )
            updater = ReplicaFixedRolloutUpdate(
                learner,
                learner_views,
                partition_specs,
                bundles[0].policy_adapter,
                cfg.loss,
                physical,
                reference_partitions=reference_views,
                statistical_counts_by_replica=counts,
            )
            legacy_update = None
            legacy_generator = None
            if verify_legacy_k1:
                assert legacy_model is not None
                assert legacy_optimizer is not None
                legacy_generator = torch.Generator(device=device)
                legacy_generator.set_state(generators[0].get_state())
                legacy_learner = DistributedFixedRolloutUpdate(
                    legacy_model,
                    legacy_optimizer,
                    _scaler(False),
                    partition_specs,
                    bundles[0].policy_adapter,
                    cfg.loss,
                    physical,
                    reference_model=legacy_reference,
                )
                legacy_update = legacy_learner(
                    storages[0],
                    legacy_generator,
                    optimizer_steps=cfg.training.gd_steps_per_iter,
                    iteration=iteration,
                    grad_clip=cfg.training.grad_clip,
                )
            if benchmark_requested:
                assert learner_benchmark_warmup is not None
                assert learner_benchmark_steps is not None
                if learner_benchmark_warmup:
                    updater(
                        storages,
                        generators,
                        iterations=(iteration,) * replicas,
                        optimizer_steps=learner_benchmark_warmup,
                    )
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                benchmark_started = time.perf_counter()
                update = updater(
                    storages,
                    generators,
                    iterations=(iteration,) * replicas,
                    optimizer_steps=learner_benchmark_steps,
                )
                torch.cuda.synchronize(device)
                benchmark_seconds = time.perf_counter() - benchmark_started
                measured_peak_allocated = int(
                    torch.cuda.max_memory_allocated(device)
                )
                measured_peak_reserved = int(
                    torch.cuda.max_memory_reserved(device)
                )
                benchmark_profile = torch.profiler.profile(
                    activities=(
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    )
                )
                benchmark_profile.__enter__()
                kernel_probe = updater(
                    storages,
                    generators,
                    iterations=(iteration,) * replicas,
                    optimizer_steps=1,
                )
                torch.cuda.synchronize(device)
                benchmark_profile.__exit__(None, None, None)
                kernel_count = _cuda_kernel_count(benchmark_profile)
                logical_rows = sum(
                    metrics["statistical_rows"] for metrics in update.metrics
                )
                probe_rows = sum(
                    metrics["statistical_rows"]
                    for metrics in kernel_probe.metrics
                )
                benchmark_report = {
                    "schema": "pbg-full-replica-learner-benchmark-v1",
                    "mode": cfg.training.mode,
                    "replicas": replicas,
                    "warmup_steps": learner_benchmark_warmup,
                    "measured_steps": learner_benchmark_steps,
                    "seconds": benchmark_seconds,
                    "logical_rows": logical_rows,
                    "logical_rows_per_second": logical_rows
                    / max(benchmark_seconds, 1.0e-12),
                    "cuda_kernels": kernel_count,
                    "kernel_probe_logical_rows": probe_rows,
                    "cuda_kernels_per_logical_row": kernel_count
                    / max(probe_rows, 1.0),
                    "peak_allocated_bytes": measured_peak_allocated,
                    "peak_reserved_bytes": measured_peak_reserved,
                    "per_replica": list(update.metrics),
                    "objective_mass": [
                        metrics["exact_objective_mass"]
                        for metrics in update.metrics
                    ],
                    "finite": [
                        metrics["optimizer_steps_nonfinite"] == 0.0
                        for metrics in update.metrics
                    ],
                }
                with (
                    Path(cfg.training.out_dir) / "learner_benchmark.json"
                ).open("w", encoding="utf-8") as stream:
                    json.dump(
                        benchmark_report, stream, indent=2, sort_keys=True
                    )
                    stream.write("\n")
                if iteration_profile is not None:
                    iteration_profile.__exit__(None, None, None)
                del updater, storages, built
                gc.collect()
                torch.cuda.empty_cache()
                break
            started = time.perf_counter()
            update = updater(
                storages,
                generators,
                iterations=(iteration,) * replicas,
                optimizer_steps=cfg.training.gd_steps_per_iter,
            )
            torch.cuda.synchronize(device)
            optimization_seconds = time.perf_counter() - started
            if iteration_profile is not None:
                iteration_profile.__exit__(None, None, None)
            if verify_legacy_k1:
                assert legacy_update is not None
                assert legacy_generator is not None
                ordinary = checkpoint_manager.ordinary_lane(0)
                parameter_error = 0.0
                moment_error = 0.0
                moment_sq_error = 0.0
                legacy_parameters = dict(legacy_model.named_parameters())
                ordinary_parameters = dict(ordinary.model.named_parameters())
                for name, expected in legacy_parameters.items():
                    actual = ordinary_parameters[name]
                    parameter_error = max(
                        parameter_error,
                        float(
                            (actual.detach() - expected.detach()).abs().max()
                        ),
                    )
                    expected_state = legacy_optimizer.state[expected]
                    actual_state = ordinary.optimizer.state[actual]
                    moment_error = max(
                        moment_error,
                        float(
                            (
                                actual_state["exp_avg"]
                                - expected_state["exp_avg"]
                            )
                            .abs()
                            .max()
                        ),
                    )
                    moment_sq_error = max(
                        moment_sq_error,
                        float(
                            (
                                actual_state["exp_avg_sq"]
                                - expected_state["exp_avg_sq"]
                            )
                            .abs()
                            .max()
                        ),
                    )
                rng_equal = torch.equal(
                    generators[0].get_state(),
                    legacy_generator.get_state(),
                )
                parity = {
                    "schema": "pbg-replica-k1-legacy-parity-v1",
                    "mode": cfg.training.mode,
                    "iteration": iteration,
                    "parameter_max_abs_error": parameter_error,
                    "exp_avg_max_abs_error": moment_error,
                    "exp_avg_sq_max_abs_error": moment_sq_error,
                    "rng_equal": rng_equal,
                    "shared_training_storage": True,
                    "generator_cloned_before_either_update": True,
                    "replica_objective_mass": update.metrics[0][
                        "exact_objective_mass"
                    ],
                    "legacy_objective_mass": legacy_update.metrics[
                        "exact_objective_mass"
                    ],
                    "replica_statistical_rows": update.metrics[0][
                        "statistical_rows"
                    ],
                    "legacy_statistical_rows": legacy_update.metrics[
                        "statistical_rows"
                    ],
                }
                parity["passed"] = (
                    parameter_error <= 5.0e-5
                    and moment_error <= 5.0e-5
                    and moment_sq_error <= 5.0e-6
                    and rng_equal
                    and parity["replica_objective_mass"]
                    == parity["legacy_objective_mass"]
                    and parity["replica_statistical_rows"]
                    == parity["legacy_statistical_rows"]
                )
                with (
                    Path(cfg.training.out_dir) / "legacy_k1_parity.json"
                ).open("w", encoding="utf-8") as stream:
                    json.dump(parity, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                if not parity["passed"]:
                    raise RuntimeError(
                        f"K=1 legacy parity failed: {parity}"
                    )
            preemption_notice = preemption.notice()
            checkpoint_due = (
                (iteration + 1)
                % (
                    cfg.hardware.checkpoint_every
                    or cfg.training.checkpoint_every
                )
                == 0
                or preemption_notice.requested
            )
            for lane, item in enumerate(built):
                metrics = dict(item.profile_metrics)
                metrics.update(update.metrics[lane])
                metrics.update(
                    {
                        "replica": lane,
                        "replicas_per_gpu": replicas,
                        "iter": iteration,
                        "run_id": run_ids[lane],
                        "decisions": item.decision_count,
                        "leaves": item.leaf_count,
                        "optimizer_steps": update.metrics[lane][
                            "optimizer_steps_applied"
                        ],
                        "phase_optimization_seconds": optimization_seconds,
                        "profiled_iteration_seconds": optimization_seconds
                        + sum(
                            float(
                                item.profile_metrics.get(
                                    f"phase_{name}_seconds", 0.0
                                )
                            )
                            for name in ("snapshot", "rollout", "backward", "seal")
                        ),
                    }
                )
                metrics["decisions_per_second"] = item.decision_count / max(
                    metrics["profiled_iteration_seconds"], 1.0e-12
                )
                logical_work[lane]["trajectories"] += int(
                    lane_cfgs[lane].rollout.roots_per_gpu
                )
                logical_work[lane]["minibatches"] += int(
                    update.metrics[lane]["physical_subspans"]
                )
                logical_work[lane]["optimizer_steps"] += int(
                    update.metrics[lane]["optimizer_steps_applied"]
                )
                if reference is not None:
                    threshold = cfg.training.kl_ref_effective_hands_rollout
                    if threshold is not None:
                        exposure = float(
                            posterior_effective_hand_counts(storages[lane]).sum()
                        )
                        reference_exposure[lane] += exposure
                        if reference_exposure[lane] >= threshold:
                            reference.copy_lane_from(lane, stacked, lane)
                            reference_iterations[lane] = iteration + 1
                            reference_exposure[lane] = 0.0
                    metrics["reference_iteration"] = reference_iterations[lane]
                    metrics["reference_effective_hands_rollout"] = (
                        reference_exposure[lane]
                    )
                with (
                    Path(lane_cfgs[lane].training.out_dir) / "metrics.jsonl"
                ).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics, sort_keys=True) + "\n")
                if checkpoint_due and write_checkpoints:
                    checkpoint_manager.save_lane(
                        Path(lane_cfgs[lane].training.out_dir)
                        / f"ckpt_{iteration + 1:07d}.pt",
                        lane,
                        scaler=scalers[lane],
                        iteration=iteration + 1,
                        cfg=lane_cfgs[lane],
                        torch_generator=generators[lane],
                        numpy_generator=numpy_generators[lane],
                        run_id=run_ids[lane],
                        rank_rng_states=None,
                        reference_iteration=reference_iterations[lane],
                        reference_effective_hands_rollout=reference_exposure[lane],
                        adaptive_work_state=None,
                        branching_controller_state=None,
                        cost_safeguard_controller_state=None,
                        probe_target=None,
                        logical_work=logical_work[lane],
                        initialization={
                            "kind": "replica_batched_scratch",
                            "lane": lane,
                            "replicas": replicas,
                        },
                    )
                    if preemption_notice.requested:
                        (
                            Path(lane_cfgs[lane].training.out_dir)
                            / "preemption_stop.json"
                        ).write_text(
                            json.dumps(
                                {
                                    "iter": iteration + 1,
                                    "run_id": run_ids[lane],
                                    "signal": preemption_notice.signal_number,
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
            aggregate_seconds = time.perf_counter() - iteration_started
            aggregate = {
                "iter": iteration,
                "replicas_per_gpu": replicas,
                "mode": cfg.training.mode,
                "fresh_decisions": sum(item.decision_count for item in built),
                "fresh_decisions_per_second": sum(
                    item.decision_count for item in built
                )
                / max(aggregate_seconds, 1.0e-12),
                "physical_gpu_seconds": aggregate_seconds,
                "optimization_seconds": optimization_seconds,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "objective_mass": [
                    update.metrics[lane]["exact_objective_mass"]
                    for lane in range(replicas)
                ],
                "optimizer_steps_applied": [
                    update.metrics[lane]["optimizer_steps_applied"]
                    for lane in range(replicas)
                ],
                "cuda_kernels": (
                    None
                    if iteration_profile is None
                    else _cuda_kernel_count(iteration_profile)
                ),
            }
            aggregate["cuda_kernels_per_decision"] = (
                None
                if aggregate["cuda_kernels"] is None
                else aggregate["cuda_kernels"]
                / max(aggregate["fresh_decisions"], 1)
            )
            aggregate["phase_seconds"] = {
                name: sum(
                    float(
                        item.profile_metrics.get(
                            f"phase_{name}_seconds", 0.0
                        )
                    )
                    for item in built
                )
                for name in ("snapshot", "rollout", "backward", "seal")
            }
            with (Path(cfg.training.out_dir) / "aggregate_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(aggregate, sort_keys=True) + "\n")
            del updater, storages, built
            gc.collect()
            torch.cuda.empty_cache()
            if preemption_notice.requested:
                break
    finally:
        preemption.restore()


__all__ = ["run_replica_training"]
