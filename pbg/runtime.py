"""Public composition for ordinary, distributed, and replica queued training."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist

from pbg.config import RootConfig
from pbg.core.branching import (
    CostSafeguardController,
    FixedBranchController,
    KKTBranchController,
    controller_from_state_dict,
)
from pbg.core.capacity import GeneratedOwners, build_storage_plan
from pbg.core.contracts import mode_schema
from pbg.core.iteration import IterationConfig, IterationEngine, LaneContext
from pbg.core.learner import ReferenceState
from pbg.core.optimization import LossScaler
from pbg.core.persistence import (
    CHECKPOINT_SCHEMA,
    LOGICAL_WORK_KEYS,
    REPLICA_CHECKPOINT_SCHEMA,
    RunState,
    load_checkpoint,
    load_distributed_checkpoint_manifest,
    load_replica_checkpoint,
    model_state_sha256,
    save_checkpoint,
    save_distributed_checkpoint_manifest,
    save_replica_checkpoint,
    sha256_file,
)
from pbg.core.replica import (
    DistributedSUM,
    ReplicaModuleOwner,
    ReplicaOptimizerGroup,
)
from pbg.core.training import PreemptionNotice
from pbg.devices import resolve_execution_device, validate_workload_capabilities
from pbg.games.poker.queued.config import BranchedModeConfig
from pbg.games.poker.queued.hooks import (
    NLHEGeometry,
    poker_partitions,
)
from pbg.games.poker.queued.model import build_model_from_config
from pbg.games.poker.queued.operations import PokerOperations
from pbg.games.poker.queued.replica import (
    PokerReplicaInference,
)
from pbg.identity import accelerator_identity, code_identity


class RuntimeError(RuntimeError):
    """A public queued consumer cannot preserve its closed runtime contract."""


def _checkpoint_payload(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("checkpoint_schema") not in {
        CHECKPOINT_SCHEMA,
        REPLICA_CHECKPOINT_SCHEMA,
    }:
        raise RuntimeError("unsupported checkpoint schema")
    return value


def _frozen_state(cfg: RootConfig) -> dict[str, torch.Tensor] | None:
    path = cfg.execution.frozen_checkpoint
    if path is None:
        return None
    payload = _checkpoint_payload(path)
    expected = cfg.scientific.applicable["frozen_sha256"]
    if sha256_file(path) != expected:
        raise RuntimeError("frozen checkpoint content hash differs from configuration")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("frozen checkpoint has no model state")
    if "lanes" in model:
        raise RuntimeError(
            "a replica checkpoint cannot be an implicit frozen-profile selector"
        )
    return model


def _controller(config: BranchedModeConfig):
    values = dict(config.branching)
    fixed_beta = dict(values.pop("fixed_beta"))
    common = {
        "fixed_beta": fixed_beta,
        "beta_max": values.pop("beta_max"),
        "minimum_scale": values.pop("minimum_scale"),
        "backoff_factor": values.pop("backoff_factor"),
        "safe_tree_size": values.pop("safe_tree_size"),
    }
    values.pop("minimum_tree_size")
    if config.mode["rollout_mode"] == "branched.fixed":
        return FixedBranchController(**common)
    return KKTBranchController(
        **common,
        warmup_iterations=values["warmup_iterations"],
        target_tree_size=values["target_tree_size"],
        gamma_kappa=values["gamma_kappa"],
        eta_kappa=values["eta_kappa"],
        kappa_min=values["kappa_min"],
        kappa_max=values["kappa_max"],
        log_kappa=values["log_kappa"],
    )


class TrainingSession:
    """Own the complete public lifetime of one queued training run."""

    def __init__(
        self,
        config: RootConfig,
        *,
        resume_from: str | Path | None = None,
        benchmark_telemetry: bool = False,
        initial_model_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if initial_model_state is not None and (
            resume_from is not None or config.execution.frozen_checkpoint is not None
        ):
            raise ValueError(
                "model-only initialization is mutually exclusive with resume "
                "and frozen-profile initialization"
            )
        self.config = config
        self.resume_from = None if resume_from is None else Path(resume_from)
        self.benchmark_telemetry = bool(benchmark_telemetry)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        workload = (
            "distributed training" if config.execution.distributed else "training"
        )
        self.device = resolve_execution_device(
            config.execution.device,
            workload=workload,
            local_rank=local_rank,
        )
        validate_workload_capabilities(self.device, workload=workload)
        torch.cuda.set_device(self.device)
        if config.execution.distributed:
            self._owns_process_group = not dist.is_initialized()
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            self.distributed = DistributedSUM()
        else:
            self._owns_process_group = False
            self.distributed = None
        self.rank = dist.get_rank() if self.distributed is not None else 0
        rank_stride = 60 << 40
        if self.rank * rank_stride >= torch.iinfo(torch.int64).max:
            raise RuntimeError("distributed rank exceeds the logical address domain")
        self.rank_logical_base = self.rank * rank_stride
        scientific = config.scientific_mapping()
        self.scientific_sha256 = hashlib.sha256(
            json.dumps(
                scientific,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.mode = mode_schema(**dict(config.mode))
        self.geometry = NLHEGeometry(
            config.scientific.common.game.players,
            config.scientific.common.game.stack_bb,
            config.scientific.common.game.rank_count,
        )
        self.runtime_identity = {
            "code": code_identity(),
            "accelerator": accelerator_identity(self.geometry.rank_count),
        }
        self.plan = build_storage_plan(
            self.mode,
            poker_partitions(self.geometry, self.mode),
            config.execution.manifest,
            config.execution.layout,
            self.device,
        )
        count = self.plan.manifest.n_replica
        modules = []
        for lane in range(count):
            torch.manual_seed(config.scientific.common.seed + lane)
            module = build_model_from_config(scientific, self.device)
            module.configure_execution(
                config.execution.attention_backend,
                config.execution.query_chunk_size,
            )
            modules.append(module)
        self.inference = PokerReplicaInference(modules)
        self.modules: ReplicaModuleOwner = self.inference.modules
        if initial_model_state is not None:
            if not initial_model_state or any(
                not isinstance(name, str) or not isinstance(value, torch.Tensor)
                for name, value in initial_model_state.items()
            ):
                raise ValueError(
                    "model-only initialization requires a nonempty tensor state"
                )
            for lane in range(count):
                self.modules.lane_module(lane).load_state_dict(
                    initial_model_state, strict=True
                )
        rb = self.mode.private_estimator.value == "rao_blackwell"
        self.model_operations = tuple(
            (
                self.modules.lane_module(lane).rb_learner_operation
                if rb
                else self.modules.lane_module(lane).learner_operation
            )
            for lane in range(count)
        )
        self.optimizers = ReplicaOptimizerGroup(
            self.modules,
            config.scientific.common.optimizer,
            scalers=tuple(
                LossScaler(enabled=config.scientific.common.precision.name == "fp16")
                for _ in range(count)
            ),
        )
        self.generators = tuple(
            torch.Generator(device=self.device).manual_seed(
                config.scientific.common.seed + lane + self.rank * 1_000_003
            )
            for lane in range(count)
        )
        self.run_states = tuple(
            RunState(
                f"{config.execution.run_id}-lane-{lane}",
                root_count=config.scientific.common.root_count,
                attempt_logical_base=self.rank_logical_base,
            )
            for lane in range(count)
        )
        reference_config = config.scientific.common.reference
        if reference_config is None:
            self.reference_modules: ReplicaModuleOwner | None = None
            self.reference_operations = (None,) * count
            self.reference_states: tuple[ReferenceState, ...] | None = None
        else:
            self.reference_modules = ReplicaModuleOwner(
                tuple(
                    copy.deepcopy(self.modules.lane_module(lane))
                    .eval()
                    .requires_grad_(False)
                    for lane in range(count)
                )
            )
            self.reference_operations = tuple(
                (
                    self.reference_modules.lane_module(lane).rb_learner_operation
                    if rb
                    else self.reference_modules.lane_module(lane).learner_operation
                )
                for lane in range(count)
            )
            self.reference_states = tuple(
                ReferenceState(
                    interval_iterations=reference_config.interval_iterations,
                    interval_effective_hands=reference_config.interval_effective_hands,
                )
                for _ in range(count)
            )
        frozen = _frozen_state(config)
        self.frozen_inference: PokerReplicaInference | None = None
        if frozen is not None:
            frozen_modules = []
            for _ in range(count):
                module = build_model_from_config(scientific, self.device)
                module.load_state_dict(frozen, strict=True)
                module.configure_execution(
                    config.execution.attention_backend,
                    config.execution.query_chunk_size,
                )
                module.eval().requires_grad_(False)
                frozen_modules.append(module)
            self.frozen_inference = PokerReplicaInference(frozen_modules)
            if config.scientific.applicable.get("response_initialization") == "target":
                for lane in range(count):
                    self.modules.lane_module(lane).load_state_dict(frozen, strict=True)
                    if self.reference_modules is not None:
                        self.reference_modules.lane_module(lane).load_state_dict(
                            frozen, strict=True
                        )
        if self.distributed is not None:
            owners = (self.modules,) + (
                (self.reference_modules,) if self.reference_modules is not None else ()
            )
            for owner in owners:
                for lane in range(count):
                    for value in owner.lane_module(lane).state_dict().values():
                        dist.broadcast(value, src=0)
        self.initial_model_sha256 = tuple(
            model_state_sha256(self.modules.lane_module(lane)) for lane in range(count)
        )
        probe = config.scientific.applicable.get("probe")
        if (
            isinstance(probe, dict)
            and self.initial_model_sha256[0] != probe["initial_model_sha256"]
        ):
            raise RuntimeError(
                "probe initial model content hash differs from configuration"
            )
        self.controllers = (
            tuple(_controller(config.scientific) for _ in range(count))
            if isinstance(config.scientific, BranchedModeConfig)
            else ()
        )
        self.safeguards = (
            tuple(
                CostSafeguardController(**dict(config.scientific.cost_safeguard))
                for _ in range(count)
            )
            if isinstance(config.scientific, BranchedModeConfig)
            else ()
        )
        if self.controllers:
            for state, controller, safeguard in zip(
                self.run_states,
                self.controllers,
                self.safeguards,
                strict=True,
            ):
                state.branch_controller_state = controller.state_dict()
                state.cost_safeguard_state = safeguard.state_dict()
        if resume_from is not None:
            self._resume(Path(resume_from))

    def _prepare_metrics_for_resume(self, path: Path) -> None:
        if self.resume_from is None:
            return
        completed_values = {state.iteration for state in self.run_states}
        if len(completed_values) != 1:
            raise RuntimeError("resume checkpoint lane iterations differ")
        completed = completed_values.pop()
        expected = [
            {
                "iteration": iteration,
                "lanes": [state.metrics_prefix[iteration] for state in self.run_states],
            }
            for iteration in range(completed)
        ]
        if not path.exists():
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in expected),
                encoding="utf-8",
            )
            return
        raw_lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        try:
            records = [json.loads(line) for line in raw_lines]
        except json.JSONDecodeError as exc:
            raise RuntimeError("resume metrics file is not valid JSONL") from exc
        if len(records) < completed or records[:completed] != expected:
            raise RuntimeError(
                "metrics.jsonl is not the checkpoint's exact gap-free prefix"
            )
        if len(records) > completed:
            backup = path.with_name(f"{path.name}.before_resume_{completed:07d}.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(
                "\n".join(raw_lines[:completed]) + ("\n" if completed else ""),
                encoding="utf-8",
            )

    def _resume(self, path: Path) -> None:
        scientific = self.config.scientific_mapping()
        execution = self.config.execution.mapping()
        if self.distributed is not None and dist.get_world_size() > 1:
            rank_paths = load_distributed_checkpoint_manifest(
                path,
                expected_scientific_config=scientific,
                expected_execution_config=execution,
                world_size=dist.get_world_size(),
            )
            path = rank_paths[dist.get_rank()]
        if self.plan.manifest.n_replica == 1:
            optimizer = self.optimizers.optimizers[0]
            scaler, reference, state, _ = load_checkpoint(
                path,
                model=self.modules.lane_module(0),
                optimizer=optimizer,
                reference_model=(
                    None
                    if self.reference_modules is None
                    else self.reference_modules.lane_module(0)
                ),
                expected_scientific_config=scientific,
                expected_execution_config=execution,
                generator=self.generators[0],
            )
            optimizer.scaler = scaler
            self.reference_states = None if reference is None else (reference,)
            self.run_states = (state,)
        else:
            references, states, _ = load_replica_checkpoint(
                path,
                modules=self.modules,
                optimizers=self.optimizers,
                expected_scientific_configs=tuple(scientific for _ in self.run_states),
                expected_execution_config=execution,
                generators=self.generators,
                reference_modules=self.reference_modules,
            )
            self.reference_states = references
            self.run_states = states
        if self.controllers:
            restored = []
            restored_safeguards = []
            for state in self.run_states:
                if state.branch_controller_state is None:
                    raise RuntimeError("branched resume omitted controller state")
                restored.append(
                    controller_from_state_dict(state.branch_controller_state)
                )
                if state.cost_safeguard_state is None:
                    raise RuntimeError("branched resume omitted cost-safeguard state")
                restored_safeguards.append(
                    CostSafeguardController.from_state_dict(state.cost_safeguard_state)
                )
            self.controllers = tuple(restored)
            self.safeguards = tuple(restored_safeguards)

    def checkpoint(self, path: str | Path) -> str:
        scientific = self.config.scientific_mapping()
        destination = Path(path)
        distributed = self.distributed is not None and dist.get_world_size() > 1
        local_destination = (
            destination.with_name(
                f"{destination.stem}.rank_{dist.get_rank():04d}{destination.suffix}"
            )
            if distributed
            else destination
        )
        if self.plan.manifest.n_replica == 1:
            digest = save_checkpoint(
                local_destination,
                model=self.modules.lane_module(0),
                optimizer=self.optimizers.optimizers[0],
                scaler=self.optimizers.optimizers[0].scaler,
                reference_model=(
                    None
                    if self.reference_modules is None
                    else self.reference_modules.lane_module(0)
                ),
                reference_state=(
                    None if self.reference_states is None else self.reference_states[0]
                ),
                run_state=self.run_states[0],
                scientific_config=scientific,
                execution_config=self.config.execution.mapping(),
                generator=self.generators[0],
            )
        else:
            digest = save_replica_checkpoint(
                local_destination,
                modules=self.modules,
                optimizers=self.optimizers,
                run_states=self.run_states,
                scientific_configs=tuple(scientific for _ in self.run_states),
                execution_config=self.config.execution.mapping(),
                generators=self.generators,
                reference_modules=self.reference_modules,
                reference_states=self.reference_states,
            )
        if distributed:
            hashes: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(hashes, digest)
            if dist.get_rank() == 0:
                names = tuple(
                    f"{destination.stem}.rank_{rank:04d}{destination.suffix}"
                    for rank in range(dist.get_world_size())
                )
                manifest_hash = save_distributed_checkpoint_manifest(
                    destination,
                    rank_files=names,
                    rank_sha256=tuple(str(value) for value in hashes),
                    scientific_config=scientific,
                    execution_config=self.config.execution.mapping(),
                )
            else:
                manifest_hash = ""
            broadcast = [manifest_hash]
            dist.broadcast_object_list(broadcast, src=0)
            return str(broadcast[0])
        return digest

    def _poker_operations(self) -> PokerOperations:
        """Assemble one semantic NLHE bundle without selecting an engine."""

        count = self.plan.manifest.n_replica
        models = tuple(self.modules.lane_module(lane) for lane in range(count))
        completion_samples = int(
            self.config.scientific.applicable.get("completion_samples", 0)
        )
        shared_inference = count > 1 or self.distributed is not None
        return PokerOperations(
            geometry=self.geometry,
            plan=self.plan,
            modes=tuple(self.mode for _ in range(count)),
            models=models,
            rollout_operations=tuple(model.rollout_operation for model in models),
            frozen_operations=tuple(
                None
                if self.frozen_inference is None
                else self.frozen_inference.modules.lane_module(lane).rollout_operation
                for lane in range(count)
            ),
            seed=self.config.scientific.common.seed,
            precision=self.config.scientific.common.precision.name,
            utility=self.config.scientific.common.utility,
            completion_samples=completion_samples,
            response_cutoff="probe" in self.config.scientific.applicable,
            inference=self.inference if shared_inference else None,
            frozen_inference=(self.frozen_inference if shared_inference else None),
        )

    def _iteration_engine(self) -> IterationEngine:
        branched = self.mode.rollout_mode.is_branched
        minimum_tree_size = (
            int(self.config.scientific.branching["minimum_tree_size"])
            if isinstance(self.config.scientific, BranchedModeConfig)
            else 1
        )
        lanes = tuple(
            LaneContext(
                mode=self.mode,
                run_state=self.run_states[lane],
                generator=self.generators[lane],
                optimizer=self.optimizers.optimizers[lane],
                model_operation=self.model_operations[lane],
                loss=self.config.scientific.common.loss,
                model=self.modules.lane_module(lane),
                reference_operation=self.reference_operations[lane],
                branch_controller=(self.controllers[lane] if branched else None),
                cost_safeguard=(self.safeguards[lane] if branched else None),
                minimum_root_count=(self.config.scientific.common.root_count_minimum),
                minimum_tree_size=minimum_tree_size,
            )
            for lane in range(self.plan.manifest.n_replica)
        )
        return IterationEngine(
            config=IterationConfig(
                self.plan,
                lanes,
                self.config.scientific.common.schedule,
                self.optimizers,
                self.config.scientific.common.lambda_,
                precision=self.config.scientific.common.precision.name,
                cache_reference_outputs=(
                    self.config.scientific.common.reference is not None
                    and self.config.scientific.common.reference.cache_outputs
                ),
                reproducibility=(self.config.scientific.common.reproducibility),
                distributed=self.distributed,
                benchmark_telemetry=self.benchmark_telemetry,
            ),
            owner_factory=lambda: GeneratedOwners(self.plan),
            operations=self._poker_operations(),
        )

    @staticmethod
    def _lane_results(result: object) -> tuple[object | None, ...]:
        failures = getattr(result, "failures", {})
        if failures:
            raise RuntimeError(
                f"replica iteration returned uncommitted lane failures: {dict(failures)}"
            )
        values = getattr(result, "results", ())
        return tuple(values) if values else (result,)

    def step(self) -> object:
        """Execute and commit exactly one production iteration in memory."""

        iterations = {state.iteration for state in self.run_states}
        if len(iterations) != 1:
            raise RuntimeError("all active lanes must enter an iteration together")
        iteration = iterations.pop()
        reference_refresh_before: list[bool] = []
        if self.reference_states is not None:
            assert self.reference_modules is not None
            for lane, reference in enumerate(self.reference_states):
                refresh = reference.refresh_before_iteration(iteration)
                reference_refresh_before.append(refresh)
                if refresh:
                    self.reference_modules.lane_module(lane).load_state_dict(
                        self.modules.lane_module(lane).state_dict(),
                        strict=True,
                    )
        result = self._iteration_engine().run(iteration=iteration)
        lane_results = self._lane_results(result)
        local_nodes = [
            int(
                getattr(
                    lane_results[lane] if lane < len(lane_results) else None,
                    "player_nodes",
                    0,
                )
            )
            for lane in range(len(self.run_states))
        ]
        local_total = sum(local_nodes)
        samples_per_leaf = int(
            self.config.scientific.applicable.get("completion_samples", 0)
        )
        local_effective = [
            float(
                getattr(
                    lane_results[lane] if lane < len(lane_results) else None,
                    "posterior_effective_hands",
                    0.0,
                )
                or 0.0
            )
            for lane in range(len(self.run_states))
        ]
        if samples_per_leaf and any(value <= 0.0 for value in local_effective):
            raise RuntimeError(
                "RB completion omitted positive posterior effective hands"
            )
        if self.mode.private_estimator.value == "rao_blackwell":
            for lane, state in enumerate(self.run_states):
                lane_result = lane_results[lane] if lane < len(lane_results) else None
                leaves = int(getattr(lane_result, "leaves", 0))
                state.logical_work["completion_samples"] += leaves * samples_per_leaf
        if self.distributed is not None:
            lane_totals = torch.tensor(
                [*local_nodes, *local_effective],
                dtype=torch.float64,
                device=self.device,
            )
            gathered = [
                torch.empty_like(lane_totals) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered, lane_totals)
            split = len(self.run_states)
            per_rank_lane_nodes = [
                [int(item) for item in value[:split].tolist()] for value in gathered
            ]
            per_rank_lane_effective = [
                [float(item) for item in value[split:].tolist()] for value in gathered
            ]
            per_rank_nodes = [sum(value) for value in per_rank_lane_nodes]
            logical_values = torch.tensor(
                [
                    state.logical_work[name]
                    for state in self.run_states
                    for name in LOGICAL_WORK_KEYS
                ],
                dtype=torch.int64,
                device=self.device,
            )
            gathered_logical = [
                torch.empty_like(logical_values) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_logical, logical_values)
        else:
            per_rank_lane_nodes = [local_nodes]
            per_rank_lane_effective = [local_effective]
            per_rank_nodes = [local_total]
            gathered_logical = [
                torch.tensor(
                    [
                        state.logical_work[name]
                        for state in self.run_states
                        for name in LOGICAL_WORK_KEYS
                    ],
                    dtype=torch.int64,
                )
            ]
        per_rank_lane_logical_work = []
        for rank_values in gathered_logical:
            flat = [int(value) for value in rank_values.tolist()]
            per_rank_lane_logical_work.append(
                [
                    {
                        name: flat[lane * len(LOGICAL_WORK_KEYS) + index]
                        for index, name in enumerate(LOGICAL_WORK_KEYS)
                    }
                    for lane in range(len(self.run_states))
                ]
            )
        per_rank_logical_work = [
            {name: sum(lane[name] for lane in rank_lanes) for name in LOGICAL_WORK_KEYS}
            for rank_lanes in per_rank_lane_logical_work
        ]
        global_lane_logical_work = [
            {
                name: (
                    max(
                        rank_lanes[lane][name]
                        for rank_lanes in per_rank_lane_logical_work
                    )
                    if name == "optimizer_steps"
                    else sum(
                        rank_lanes[lane][name]
                        for rank_lanes in per_rank_lane_logical_work
                    )
                )
                for name in LOGICAL_WORK_KEYS
            }
            for lane in range(len(self.run_states))
        ]
        global_logical_work = {
            name: sum(lane[name] for lane in global_lane_logical_work)
            for name in LOGICAL_WORK_KEYS
        }
        global_nodes = sum(per_rank_nodes)
        global_lane_nodes = [
            sum(rank_nodes[lane] for rank_nodes in per_rank_lane_nodes)
            for lane in range(len(self.run_states))
        ]
        global_lane_effective = [
            sum(rank_values[lane] for rank_values in per_rank_lane_effective)
            for lane in range(len(self.run_states))
        ]
        for lane, state in enumerate(self.run_states):
            lane_result = lane_results[lane] if lane < len(lane_results) else None
            player_nodes = int(
                getattr(
                    lane_result,
                    "player_nodes",
                    getattr(lane_result, "nodes", 0),
                )
            )
            state.iteration = iteration + 1
            if not state.metrics_prefix:
                raise RuntimeError("learner omitted the iteration metrics row")
            state.metrics_prefix[-1].update(
                {
                    "player_nodes": player_nodes,
                    "leaves": int(getattr(lane_result, "leaves", 0)),
                    "target_layers": int(getattr(lane_result, "target_layers", 0)),
                    "mode": self.mode.key,
                    "lane": lane,
                    "rank": dist.get_rank() if self.distributed is not None else 0,
                    "world_size": (
                        dist.get_world_size() if self.distributed is not None else 1
                    ),
                    "per_rank_player_nodes": per_rank_nodes,
                    "per_rank_lane_player_nodes": per_rank_lane_nodes,
                    "global_player_nodes": global_nodes,
                    "global_lane_player_nodes": global_lane_nodes,
                    "per_rank_lane_effective_hands": per_rank_lane_effective,
                    "global_lane_effective_hands": global_lane_effective,
                    "logical_work": dict(state.logical_work),
                    "per_rank_lane_logical_work": (per_rank_lane_logical_work),
                    "per_rank_logical_work": per_rank_logical_work,
                    "global_lane_logical_work": global_lane_logical_work,
                    "global_logical_work": global_logical_work,
                    "structure": dict(getattr(lane_result, "observability", {})),
                    "completion_health": {
                        "leaf_rows": int(getattr(lane_result, "leaves", 0)),
                        "samples_per_leaf": int(
                            self.config.scientific.applicable.get(
                                "completion_samples", 0
                            )
                        ),
                        "logical_samples": int(
                            state.logical_work["completion_samples"]
                        ),
                        "posterior_effective_hands": (
                            global_lane_effective[lane] if samples_per_leaf else None
                        ),
                    },
                    "memory": {
                        "configured_budget_bytes": (
                            self.plan.manifest.hardware.usable_memory_bytes
                        ),
                        "admitted_phase_required_bytes": {
                            admission.phase.name: admission.required_bytes
                            for admission in self.plan.admissions
                        },
                        "allocated_bytes": torch.cuda.memory_allocated(self.device),
                        "reserved_bytes": torch.cuda.memory_reserved(self.device),
                        "peak_allocated_bytes": (
                            torch.cuda.max_memory_allocated(self.device)
                        ),
                        "peak_reserved_bytes": (
                            torch.cuda.max_memory_reserved(self.device)
                        ),
                        "replica_resident_bytes": (
                            self.optimizers.memory_report().total_bytes
                        ),
                    },
                    "identity": {
                        "run_id": state.run_id,
                        "scientific_sha256": self.scientific_sha256,
                        "initial_model_sha256": self.initial_model_sha256[lane],
                        "mode": self.mode.key,
                        **self.runtime_identity,
                    },
                }
            )
            if self.reference_states is not None:
                assert self.reference_modules is not None
                reference = self.reference_states[lane]
                exposure = float(
                    global_lane_effective[lane]
                    if samples_per_leaf
                    else global_lane_nodes[lane]
                )
                reference_before = reference.effective_hands_rollout
                reference_threshold = reference.interval_effective_hands
                reference_overshoot = (
                    max(reference_before + exposure - reference_threshold, 0.0)
                    if reference_threshold is not None
                    else 0.0
                )
                reference_refresh_after = reference.observe(
                    iteration=iteration + 1,
                    effective_hands=exposure,
                )
                if reference_refresh_after:
                    self.reference_modules.lane_module(lane).load_state_dict(
                        self.modules.lane_module(lane).state_dict(), strict=True
                    )
                state.metrics_prefix[-1].update(
                    {
                        "reference": reference.state_dict(),
                        "kl_ref_refresh_before_iteration": reference_refresh_before[
                            lane
                        ],
                        "kl_ref_effective_hands_rollout_before": reference_before,
                        "kl_ref_effective_hands_rollout_threshold": reference_threshold,
                        "kl_ref_refresh_after_iteration": reference_refresh_after,
                        "kl_ref_effective_hands_rollout_overshoot": (
                            reference_overshoot if reference_refresh_after else 0.0
                        ),
                        "kl_ref_effective_hands_rollout_after": reference.effective_hands_rollout,
                        "reference_iteration_next": reference.reference_iteration,
                    }
                )
            learner = getattr(lane_result, "learner", None)
            losses = tuple(getattr(learner, "losses", ()))
            if losses:
                state.metrics_prefix[-1]["objective"] = {
                    name: (
                        max(float(row[name]) for row in losses)
                        if name.endswith("_max")
                        else sum(float(row[name]) for row in losses) / len(losses)
                    )
                    for name in losses[0]
                }
            steps = tuple(getattr(learner, "optimizer_steps", ()))
            if steps:
                state.metrics_prefix[-1]["gradient"] = {
                    "finite": all(step.gradients_finite for step in steps),
                    "norm_mean": sum(step.grad_norm for step in steps) / len(steps),
                    "norm_max": max(step.grad_norm for step in steps),
                    "clip_scale_min": min(step.clip_scale for step in steps),
                    "applied_steps": sum(int(step.applied) for step in steps),
                    "loss_scale_before": steps[0].loss_scale_before,
                    "loss_scale_after": steps[-1].loss_scale_after,
                }
        if self.controllers:
            for state, controller, safeguard in zip(
                self.run_states,
                self.controllers,
                self.safeguards,
                strict=True,
            ):
                state.branch_controller_state = controller.state_dict()
                state.cost_safeguard_state = safeguard.state_dict()
        return result

    def run(self, *, deadline_monotonic: float | None = None) -> int:
        """Run configured iterations, optionally stopping before a deadline."""

        if deadline_monotonic is not None and not isinstance(
            deadline_monotonic, (int, float)
        ):
            raise TypeError("deadline_monotonic must be a monotonic timestamp")
        output = self.config.execution.output_dir
        output.mkdir(parents=True, exist_ok=True)
        write_rank = self.distributed is None or dist.get_rank() == 0
        if write_rank:
            self._prepare_metrics_for_resume(output / "metrics.jsonl")
        start = min(state.iteration for state in self.run_states)
        notice = PreemptionNotice()
        notice.install()
        completed = 0
        try:
            for iteration in range(start, self.config.execution.iterations):
                local_stop = (
                    deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic
                )
                if self._agree_flag(local_stop):
                    break
                local_error: Exception | None = None
                try:
                    result = self.step()
                except Exception as exc:  # noqa: BLE001 - rank-common below
                    local_error = exc
                    result = None
                if self._agree_flag(local_error is not None):
                    checkpointable = all(
                        state.iteration == iteration for state in self.run_states
                    )
                    if self._agree_flag(not checkpointable):
                        if local_error is not None:
                            raise local_error
                        raise RuntimeError(
                            "another rank failed after leaving the common "
                            "optimizer boundary"
                        )
                    if checkpointable:
                        self.checkpoint(
                            output / f"failure_checkpoint_{iteration:08d}.pt"
                        )
                    if local_error is not None:
                        raise local_error
                    raise RuntimeError(
                        "another distributed rank failed during the iteration"
                    )
                assert result is not None
                completed += 1
                if write_rank:
                    metrics = output / "metrics.jsonl"
                    with metrics.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "iteration": iteration,
                                    "lanes": [
                                        state.metrics_prefix[-1]
                                        for state in self.run_states
                                    ],
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                stop_requested = self._agree_flag(notice.requested)
                checkpoint_due = (
                    iteration + 1
                ) % self.config.execution.checkpoint_every == 0 or stop_requested
                if checkpoint_due:
                    self.checkpoint(output / f"checkpoint_{iteration + 1:08d}.pt")
                del result
                if stop_requested:
                    break
        finally:
            notice.restore()
        return completed

    def _agree_flag(self, value: bool) -> bool:
        if self.distributed is None:
            return bool(value)
        flag = torch.tensor([int(value)], dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def close(self) -> None:
        """Release a process group only after all collective consumers finish."""

        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False


def train(
    cfg: RootConfig,
    *,
    resume_from: str | Path | None = None,
) -> None:
    session = TrainingSession(cfg, resume_from=resume_from)
    try:
        session.run()
    finally:
        session.close()


__all__ = ["RuntimeError", "TrainingSession", "train"]
