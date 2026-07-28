"""Public composition for ordinary, distributed, and replica queued training."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import MappingProxyType

import torch
import torch.distributed as dist

from pbg.config import RootConfig
from pbg.devices import resolve_execution_device, validate_workload_capabilities
from pbg.games.poker.queued import (
    BranchedSampledPokerIteration,
    NLHEGeometry,
    PokerReplicaInference,
    ReplicaBranchedPokerIteration,
    SequentialPokerIteration,
    build_model_from_config,
    poker_partitions,
)
from pbg.queued import (
    CHECKPOINT_SCHEMA,
    REPLICA_CHECKPOINT_SCHEMA,
    BranchedModeConfig,
    CostSafeguardController,
    DistributedSUM,
    FixedBranchController,
    GeneratedOwners,
    KKTBranchController,
    LossScaler,
    ReferenceState,
    ReplicaModuleOwner,
    ReplicaOptimizerGroup,
    RunState,
    SequentialModeConfig,
    build_storage_plan,
    load_checkpoint,
    load_distributed_checkpoint_manifest,
    load_replica_checkpoint,
    mode_schema,
    save_checkpoint,
    save_distributed_checkpoint_manifest,
    save_replica_checkpoint,
    sha256_file,
    PreemptionNotice,
)
from pbg.queued.optimization import DenseAdam


class RuntimeError(RuntimeError):
    """A public queued consumer cannot preserve its closed runtime contract."""


def _checkpoint_payload(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("checkpoint_schema") not in {
        CHECKPOINT_SCHEMA,
        REPLICA_CHECKPOINT_SCHEMA,
    }:
        raise RuntimeError(
            "runtime accepts only queued checkpoints; convert legacy artifacts offline"
        )
    return value


def _frozen_state(cfg: RootConfig) -> dict[str, torch.Tensor] | None:
    path = cfg.execution.frozen_checkpoint
    if path is None:
        return None
    payload = _checkpoint_payload(path)
    expected = cfg.scientific.applicable["frozen_profile"]["sha256"]
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
    ) -> None:
        self.config = config
        self.resume_from = None if resume_from is None else Path(resume_from)
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
        self.optimizers = ReplicaOptimizerGroup(
            self.modules,
            config.scientific.common.optimizer,
            scalers=tuple(
                LossScaler(
                    enabled=config.scientific.common.precision.name == "fp16"
                )
                for _ in range(count)
            ),
        )
        self.generators = tuple(
            torch.Generator(device=self.device).manual_seed(
                config.scientific.common.seed + lane
            )
            for lane in range(count)
        )
        self.run_states = tuple(
            RunState(
                f"{config.execution.run_id}-lane-{lane}",
                root_count=config.scientific.common.root_count,
            )
            for lane in range(count)
        )
        self.reference_modules = ReplicaModuleOwner(
            tuple(copy.deepcopy(module).eval().requires_grad_(False) for module in modules)
        )
        self.reference_operations = tuple(
            self.reference_modules.lane_module(lane).learner_operation
            for lane in range(count)
        )
        self.reference_states = tuple(
            ReferenceState(
                interval_iterations=config.scientific.common.reference.interval_iterations,
                interval_effective_hands=(
                    config.scientific.common.reference.interval_effective_hands
                ),
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
            if (
                config.scientific.applicable.get("response_initialization")
                == "target"
            ):
                for lane in range(count):
                    self.modules.lane_module(lane).load_state_dict(
                        frozen, strict=True
                    )
                    self.reference_modules.lane_module(lane).load_state_dict(
                        frozen, strict=True
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
                "lanes": [
                    state.metrics_prefix[iteration] for state in self.run_states
                ],
            }
            for iteration in range(completed)
        ]
        if not path.exists():
            path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in expected
                ),
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
            backup = path.with_name(
                f"{path.name}.before_resume_{completed:07d}.bak"
            )
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(
                "\n".join(raw_lines[:completed])
                + ("\n" if completed else ""),
                encoding="utf-8",
            )

    def _resume(self, path: Path) -> None:
        scientific = self.config.scientific_mapping()
        if self.distributed is not None and dist.get_world_size() > 1:
            rank_paths = load_distributed_checkpoint_manifest(
                path,
                expected_scientific_config=scientific,
                world_size=dist.get_world_size(),
            )
            path = rank_paths[dist.get_rank()]
        if self.plan.manifest.n_replica == 1:
            optimizer = self.optimizers.optimizers[0]
            scaler, reference, state, _execution = load_checkpoint(
                path,
                model=self.modules.lane_module(0),
                optimizer=optimizer,
                reference_model=self.reference_modules.lane_module(0),
                expected_scientific_config=scientific,
                generator=self.generators[0],
            )
            optimizer.scaler = scaler
            self.reference_states = (reference,)
            self.run_states = (state,)
        else:
            references, states, _execution = load_replica_checkpoint(
                path,
                modules=self.modules,
                optimizers=self.optimizers,
                expected_scientific_configs=tuple(
                    scientific for _ in self.run_states
                ),
                generators=self.generators,
                reference_modules=self.reference_modules,
            )
            if references is None:
                raise RuntimeError("replica resume omitted reference cadence state")
            self.reference_states = references
            self.run_states = states
        if self.controllers:
            restored = []
            for state, fallback in zip(
                self.run_states, self.controllers, strict=True
            ):
                if state.branch_controller_state is None:
                    raise RuntimeError("branched resume omitted controller state")
                from pbg.queued import controller_from_state_dict

                restored.append(
                    controller_from_state_dict(state.branch_controller_state)
                )
            self.controllers = tuple(restored)

    def checkpoint(self, path: str | Path) -> str:
        scientific = self.config.scientific_mapping()
        destination = Path(path)
        distributed = self.distributed is not None and dist.get_world_size() > 1
        local_destination = (
            destination.with_name(
                f"{destination.stem}.rank_{dist.get_rank():04d}"
                f"{destination.suffix}"
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
                reference_model=self.reference_modules.lane_module(0),
                reference_state=self.reference_states[0],
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

    def _run_sequential(self, iteration: int):
        count = self.plan.manifest.n_replica
        models = tuple(self.modules.lane_module(lane) for lane in range(count))
        completion = self.config.scientific.applicable.get("completion", {})
        return SequentialPokerIteration(
            geometry=self.geometry,
            owners=GeneratedOwners(self.plan),
            mode=self.mode,
            model=models[0],
            model_operation=models[0].learner_operation,
            optimizer=self.optimizers.optimizers[0],
            loss=self.config.scientific.common.loss,
            schedule=self.config.scientific.common.schedule,
            generator=self.generators[0],
            seed=self.config.scientific.common.seed,
            lambda_=self.config.scientific.common.lambda_,
            precision=self.config.scientific.common.precision.name,
            utility=self.config.scientific.common.utility,
            cache_reference_outputs=(
                self.config.scientific.common.reference.cache_outputs
            ),
            reproducibility=self.config.scientific.common.reproducibility,
            run_state=self.run_states[0],
            reference_operation=self.reference_operations[0],
            frozen_operation=(
                None
                if self.frozen_inference is None
                else self.frozen_inference.modules.lane_module(0).rollout_operation
            ),
            completion_samples=int(completion.get("n_samples", 0)),
            probe_cutoff="probe" in self.config.scientific.applicable,
            replica_inference=self.inference if count > 1 else None,
            replica_optimizers=(
                self.optimizers
                if count > 1 or self.distributed is not None
                else None
            ),
            replica_losses=tuple(
                self.config.scientific.common.loss for _ in range(count)
            ),
            replica_generators=self.generators,
            replica_run_states=self.run_states,
            replica_model_operations=tuple(
                model.learner_operation for model in models
            ),
            replica_frozen_inference=self.frozen_inference,
            distributed=self.distributed,
        ).run(iteration=iteration)

    def _run_branched(self, iteration: int):
        assert isinstance(self.config.scientific, BranchedModeConfig)
        count = self.plan.manifest.n_replica
        completion = self.config.scientific.applicable.get("completion", {})
        if count > 1 or self.distributed is not None:
            return ReplicaBranchedPokerIteration(
                geometry=self.geometry,
                owners=GeneratedOwners(self.plan),
                modes=tuple(self.mode for _ in range(count)),
                controllers=self.controllers,
                inference=self.inference,
                optimizers=self.optimizers,
                losses=tuple(
                    self.config.scientific.common.loss for _ in range(count)
                ),
                schedule=self.config.scientific.common.schedule,
                generators=self.generators,
                run_states=self.run_states,
                seed=self.config.scientific.common.seed,
                lambda_=self.config.scientific.common.lambda_,
                precision=self.config.scientific.common.precision.name,
                utility=self.config.scientific.common.utility,
                completion_samples=int(completion.get("n_samples", 0)),
                frozen_inference=self.frozen_inference,
                distributed=self.distributed,
            ).run(iteration=iteration)
        model = self.modules.lane_module(0)
        arguments = {
            "geometry": self.geometry,
            "mode": self.mode,
            "model": model,
            "model_operation": (
                model.rb_learner_operation
                if self.mode.private_estimator.value == "rao_blackwell"
                else model.learner_operation
            ),
            "rollout_operation": model.rollout_operation,
            "optimizer": self.optimizers.optimizers[0],
            "loss": self.config.scientific.common.loss,
            "schedule": self.config.scientific.common.schedule,
            "generator": self.generators[0],
            "seed": self.config.scientific.common.seed,
            "lambda_": self.config.scientific.common.lambda_,
            "minimum_tree_size": self.config.scientific.branching[
                "minimum_tree_size"
            ],
            "cost_safeguard": self.safeguards[0],
            "precision": self.config.scientific.common.precision.name,
            "utility": self.config.scientific.common.utility,
            "cache_reference_outputs": (
                self.config.scientific.common.reference.cache_outputs
            ),
            "reference_operation": self.reference_operations[0],
            "frozen_operation": (
                None
                if self.frozen_inference is None
                else self.frozen_inference.modules.lane_module(0).rollout_operation
            ),
            "completion_samples": int(completion.get("n_samples", 0)),
        }
        return BranchedSampledPokerIteration(
            owner_factory=lambda: GeneratedOwners(self.plan),
            plan=self.plan,
            controller=self.controllers[0],
            run_state=self.run_states[0],
            arguments=MappingProxyType(arguments),
        ).run(iteration=iteration)

    @staticmethod
    def _lane_results(result: object) -> tuple[object | None, ...]:
        values = getattr(result, "results", ())
        return tuple(values) if values else (result,)

    def step(self) -> object:
        """Execute and commit exactly one production iteration in memory."""

        iterations = {state.iteration for state in self.run_states}
        if len(iterations) != 1:
            raise RuntimeError("all active lanes must enter an iteration together")
        iteration = iterations.pop()
        result = (
            self._run_sequential(iteration)
            if isinstance(self.config.scientific, SequentialModeConfig)
            else self._run_branched(iteration)
        )
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
        if self.distributed is not None:
            total = torch.tensor(
                [local_total], dtype=torch.int64, device=self.device
            )
            gathered = [torch.empty_like(total) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, total)
            per_rank_nodes = [int(value.item()) for value in gathered]
        else:
            per_rank_nodes = [local_total]
        global_nodes = sum(per_rank_nodes)
        for lane, (state, reference) in enumerate(
            zip(self.run_states, self.reference_states, strict=True)
        ):
            lane_result = lane_results[lane] if lane < len(lane_results) else None
            player_nodes = int(
                getattr(
                    lane_result,
                    "player_nodes",
                    getattr(lane_result, "nodes", 0),
                )
            )
            if reference.observe(
                iteration=state.iteration,
                effective_hands=float(player_nodes),
            ):
                self.reference_modules.lane_module(lane).load_state_dict(
                    self.modules.lane_module(lane).state_dict(),
                    strict=True,
                )
            if self.mode.private_estimator.value == "rao_blackwell":
                completion = self.config.scientific.applicable["completion"]
                leaves = int(getattr(lane_result, "leaves", 0))
                state.logical_work["completion_samples"] += (
                    leaves * int(completion["n_samples"])
                )
            state.iteration = iteration + 1
            if not state.metrics_prefix:
                raise RuntimeError("learner omitted the iteration metrics row")
            state.metrics_prefix[-1].update(
                {
                    "player_nodes": player_nodes,
                    "leaves": int(getattr(lane_result, "leaves", 0)),
                    "target_layers": int(
                        getattr(lane_result, "target_layers", 0)
                    ),
                    "mode": self.mode.key,
                    "lane": lane,
                    "rank": dist.get_rank() if self.distributed is not None else 0,
                    "world_size": (
                        dist.get_world_size()
                        if self.distributed is not None
                        else 1
                    ),
                    "per_rank_player_nodes": per_rank_nodes,
                    "global_player_nodes": global_nodes,
                    "logical_work": dict(state.logical_work),
                    "reference": reference.state_dict(),
                    "structure": dict(
                        getattr(lane_result, "observability", {})
                    ),
                    "completion_health": {
                        "leaf_rows": int(getattr(lane_result, "leaves", 0)),
                        "samples_per_leaf": int(
                            self.config.scientific.applicable.get(
                                "completion", {}
                            ).get("n_samples", 0)
                        ),
                        "logical_samples": int(
                            state.logical_work["completion_samples"]
                        ),
                    },
                    "identity": {
                        "run_id": state.run_id,
                        "scientific_sha256": self.scientific_sha256,
                        "mode": self.mode.key,
                    },
                }
            )
            learner = getattr(lane_result, "learner", None)
            losses = tuple(getattr(learner, "losses", ()))
            if losses:
                state.metrics_prefix[-1]["objective"] = {
                    name: sum(float(row[name]) for row in losses) / len(losses)
                    for name in losses[0]
                }
            steps = tuple(getattr(learner, "steps", ()))
            if steps:
                state.metrics_prefix[-1]["gradient"] = {
                    "finite": all(step.gradients_finite for step in steps),
                    "norm_mean": sum(step.grad_norm for step in steps)
                    / len(steps),
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
                if (
                    deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic
                ):
                    break
                try:
                    result = self.step()
                except Exception:
                    if all(
                        state.iteration == iteration
                        for state in self.run_states
                    ):
                        self.checkpoint(
                            output
                            / f"failure_checkpoint_{iteration:08d}.pt"
                        )
                    raise
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
                checkpoint_due = (
                    (iteration + 1) % self.config.execution.checkpoint_every == 0
                    or notice.requested
                )
                if checkpoint_due:
                    self.checkpoint(
                        output / f"checkpoint_{iteration + 1:08d}.pt"
                    )
                del result
                if notice.requested:
                    break
        finally:
            notice.restore()
            if self._owns_process_group and dist.is_initialized():
                dist.destroy_process_group()
        return completed


def train(
    cfg: RootConfig,
    *,
    resume_from: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
) -> None:
    if init_checkpoint is not None:
        raise RuntimeError(
            "runtime weight-only initialization is not supported; "
            "use the explicit offline artifact converter"
        )
    TrainingSession(cfg, resume_from=resume_from).run()


__all__ = ["RuntimeError", "TrainingSession", "train"]
