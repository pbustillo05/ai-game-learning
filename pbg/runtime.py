"""Public composition for ordinary, distributed, and replica queued training."""

from __future__ import annotations

import copy
import json
import os
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
    load_replica_checkpoint,
    mode_schema,
    save_checkpoint,
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
        "pbg-queued-checkpoint-v2",
        "pbg-queued-replica-checkpoint-v1",
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
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            self.distributed = DistributedSUM()
        else:
            self.distributed = None
        scientific = config.scientific_mapping()
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

    def _resume(self, path: Path) -> None:
        scientific = self.config.scientific_mapping()
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
            states, _execution = load_replica_checkpoint(
                path,
                modules=self.modules,
                optimizers=self.optimizers,
                expected_scientific_configs=tuple(
                    scientific for _ in self.run_states
                ),
                generators=self.generators,
                reference_modules=self.reference_modules,
            )
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
        if self.plan.manifest.n_replica == 1:
            return save_checkpoint(
                destination,
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
        return save_replica_checkpoint(
            destination,
            modules=self.modules,
            optimizers=self.optimizers,
            run_states=self.run_states,
            scientific_configs=tuple(scientific for _ in self.run_states),
            execution_config=self.config.execution.mapping(),
            generators=self.generators,
            reference_modules=self.reference_modules,
        )

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

    def run(self) -> None:
        output = self.config.execution.output_dir
        output.mkdir(parents=True, exist_ok=True)
        start = min(state.iteration for state in self.run_states)
        notice = PreemptionNotice()
        notice.install()
        try:
            for iteration in range(start, self.config.execution.iterations):
                result = (
                    self._run_sequential(iteration)
                    if isinstance(self.config.scientific, SequentialModeConfig)
                    else self._run_branched(iteration)
                )
                lane_results = tuple(
                    value
                    for value in (getattr(result, "results", ()) or (result,))
                )
                for lane, (state, reference) in enumerate(
                    zip(self.run_states, self.reference_states, strict=True)
                ):
                    lane_result = (
                        lane_results[lane]
                        if lane < len(lane_results)
                        else None
                    )
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
                    state.metrics_prefix[-1].update(
                        {
                            "player_nodes": player_nodes,
                            "mode": self.mode.key,
                            "lane": lane,
                        }
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
