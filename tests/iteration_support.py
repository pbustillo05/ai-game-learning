"""Test adapters that exercise poker only through the generic core engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from pbg.core.capacity import GeneratedOwners
from pbg.core.iteration import (
    IterationBatchResult,
    IterationConfig,
    IterationEngine,
    LaneContext,
)
from pbg.games.poker.queued.operations import PokerOperations


class _LocalAgreement:
    def agree(self, failure):
        return failure


def _lane_values(
    explicit: Sequence[object] | None,
    first: object,
    count: int,
) -> tuple[object, ...]:
    return (
        tuple(first for _lane in range(count))
        if explicit is None
        else tuple(explicit)
    )


def _result(value: IterationBatchResult):
    return value.results[0] if len(value.results) == 1 else value


def _sequential_engine(
    *,
    owners: GeneratedOwners,
    geometry,
    mode,
    model,
    model_operation,
    optimizer,
    loss,
    schedule,
    generator,
    seed,
    lambda_,
    rollout_operation=None,
    frozen_operation=None,
    reference_operation=None,
    run_state=None,
    precision="fp32",
    utility="raw_bb",
    cache_reference_outputs=False,
    allocation_inputs=None,
    reproducibility="strict",
    completion_samples=0,
    response_cutoff=False,
    replica_inference=None,
    replica_optimizers=None,
    replica_losses=None,
    replica_generators=None,
    replica_run_states=None,
    replica_model_operations=None,
    replica_reference_operations=None,
    replica_frozen_inference=None,
    distributed=None,
    owner_factory=None,
    minimum_root_count=1,
):
    count = owners.plan.manifest.n_replica
    models = (
        (model,)
        if replica_inference is None
        else tuple(
            replica_inference.modules.lane_module(lane)
            for lane in range(count)
        )
    )
    model_operations = _lane_values(
        replica_model_operations, model_operation, count
    )
    losses = _lane_values(replica_losses, loss, count)
    generators = _lane_values(replica_generators, generator, count)
    run_states = _lane_values(replica_run_states, run_state, count)
    references = _lane_values(
        replica_reference_operations, reference_operation, count
    )
    optimizer_owner = (
        optimizer if replica_optimizers is None else replica_optimizers
    )
    optimizers = (
        (optimizer,)
        if replica_optimizers is None
        else tuple(replica_optimizers.optimizers)
    )
    rollout_operations = tuple(
        (
            rollout_operation
            if count == 1 and rollout_operation is not None
            else getattr(
                models[lane],
                "rollout_operation",
                model_operations[lane],
            )
        )
        for lane in range(count)
    )
    frozen_operations = tuple(
        (
            frozen_operation
            if lane == 0 and frozen_operation is not None
            else (
                None
                if replica_frozen_inference is None
                else replica_frozen_inference.modules.lane_module(
                    lane
                ).rollout_operation
            )
        )
        for lane in range(count)
    )
    operations = PokerOperations(
        geometry=geometry,
        plan=owners.plan,
        modes=tuple(mode for _lane in range(count)),
        models=models,
        rollout_operations=rollout_operations,
        frozen_operations=frozen_operations,
        seed=seed,
        precision=precision,
        utility=utility,
        completion_samples=completion_samples,
        response_cutoff=response_cutoff,
        inference=replica_inference,
        frozen_inference=replica_frozen_inference,
    )
    lanes = tuple(
        LaneContext(
            mode=mode,
            run_state=run_states[lane],
            generator=generators[lane],
            optimizer=optimizers[lane],
            model_operation=model_operations[lane],
            loss=losses[lane],
            model=models[lane],
            reference_operation=references[lane],
            allocation_inputs=allocation_inputs,
            minimum_root_count=minimum_root_count,
        )
        for lane in range(count)
    )
    return IterationEngine(
        config=IterationConfig(
            owners.plan,
            lanes,
            schedule,
            optimizer_owner,
            lambda_,
            precision=precision,
            cache_reference_outputs=cache_reference_outputs,
            reproducibility=reproducibility,
            distributed=distributed,
        ),
        owner_factory=(
            (lambda: owners) if owner_factory is None else owner_factory
        ),
        operations=operations,
        rank_agreement=_LocalAgreement(),
    )


def _branched_engine(
    *,
    owners: GeneratedOwners,
    geometry,
    modes,
    controllers,
    safeguards,
    inference,
    reference_operations,
    optimizers,
    losses,
    schedule,
    generators,
    run_states,
    minimum_tree_sizes,
    seed,
    lambda_,
    precision="fp32",
    utility="raw_bb",
    completion_samples=0,
    posterior_max_attempts=256,
    cache_reference_outputs=False,
    frozen_inference=None,
    models=None,
    rollout_operations=None,
    learner_operations=None,
    frozen_operations=None,
    allocation_inputs=None,
    benchmark_telemetry=False,
    distributed=None,
    owner_factory=None,
):
    count = owners.plan.manifest.n_replica
    modes = tuple(modes)
    models = (
        tuple(models)
        if models is not None
        else tuple(
            inference.modules.lane_module(lane) for lane in range(count)
        )
    )
    rollout_operations = (
        tuple(rollout_operations)
        if rollout_operations is not None
        else tuple(model.rollout_operation for model in models)
    )
    learner_operations = (
        tuple(learner_operations)
        if learner_operations is not None
        else tuple(
            (
                model.learner_operation
                if modes[lane].private_estimator.value == "sampled"
                else model.rb_learner_operation
            )
            for lane, model in enumerate(models)
        )
    )
    frozen_operations = (
        tuple(frozen_operations)
        if frozen_operations is not None
        else tuple(
            None
            if frozen_inference is None
            else frozen_inference.modules.lane_module(
                lane
            ).rollout_operation
            for lane in range(count)
        )
    )
    allocations = (
        tuple(None for _lane in range(count))
        if allocation_inputs is None
        else tuple(allocation_inputs)
    )
    operations = PokerOperations(
        geometry=geometry,
        plan=owners.plan,
        modes=modes,
        models=models,
        rollout_operations=rollout_operations,
        frozen_operations=frozen_operations,
        seed=seed,
        precision=precision,
        utility=utility,
        completion_samples=completion_samples,
        posterior_max_attempts=posterior_max_attempts,
        inference=inference,
        frozen_inference=frozen_inference,
    )
    lanes = tuple(
        LaneContext(
            mode=modes[lane],
            run_state=run_states[lane],
            generator=generators[lane],
            optimizer=optimizers.optimizers[lane],
            model_operation=learner_operations[lane],
            loss=losses[lane],
            model=models[lane],
            reference_operation=reference_operations[lane],
            branch_controller=controllers[lane],
            cost_safeguard=safeguards[lane],
            allocation_inputs=allocations[lane],
            minimum_tree_size=minimum_tree_sizes[lane],
        )
        for lane in range(count)
    )
    return IterationEngine(
        config=IterationConfig(
            owners.plan,
            lanes,
            schedule,
            optimizers,
            lambda_,
            precision=precision,
            cache_reference_outputs=cache_reference_outputs,
            distributed=distributed,
            benchmark_telemetry=benchmark_telemetry,
        ),
        owner_factory=(
            (lambda: owners) if owner_factory is None else owner_factory
        ),
        operations=operations,
        rank_agreement=_LocalAgreement(),
    )


class EngineDrivenSequentialAttempt:
    def __init__(self, **arguments) -> None:
        self.arguments = arguments
        self.owners = arguments["owners"]

    def run(self, *, iteration: int):
        engine = _sequential_engine(**self.arguments)
        return _result(engine._run_once(self.owners, iteration=iteration))

    @property
    def operations(self) -> PokerOperations:
        return _sequential_engine(**self.arguments).operations


class EngineDrivenReplicaBranchedAttempt:
    def __init__(self, **arguments) -> None:
        arguments.setdefault(
            "minimum_tree_sizes",
            (1,) * arguments["owners"].plan.manifest.n_replica,
        )
        self.arguments = arguments
        self.owners = arguments["owners"]

    def run(self, *, iteration: int):
        engine = _branched_engine(**self.arguments)
        return engine._run_once(self.owners, iteration=iteration)


class _LaneStepper:
    def __init__(self, group: _SingleOptimizerGroup) -> None:
        self.group = group

    @property
    def scaler(self):
        return self.group.optimizers[0].scaler

    def zero_grad(self) -> None:
        self.group.optimizers[0].zero_grad()

    def agree_failure(self, failed: bool) -> bool:
        return bool(failed)

    def common_step_count(self, local_steps: int) -> int:
        return int(local_steps)

    def step(self):
        result = self.group.optimizers[0].step()
        if result.applied:
            self.group.versions[0] += 1
        return result


class _SingleOptimizerGroup:
    def __init__(self, optimizer) -> None:
        self.optimizers = (optimizer,)
        self.versions = torch.zeros(1, dtype=torch.int64)
        self._transaction_versions = None

    def begin_transaction(self) -> None:
        self.optimizers[0].begin_transaction()
        self._transaction_versions = self.versions.clone()

    def rollback_transaction(self) -> None:
        self.optimizers[0].rollback_transaction()
        self.versions.copy_(self._transaction_versions)
        self._transaction_versions = None

    def commit_transaction(self) -> None:
        self.optimizers[0].commit_transaction()
        self._transaction_versions = None

    def lane_stepper(self, lane: int, *, distributed=None):
        assert lane == 0 and distributed is None
        return _LaneStepper(self)


class EngineDrivenBranchedAttempt:
    def __init__(
        self,
        *,
        geometry,
        owners,
        mode,
        controller,
        model,
        model_operation,
        optimizer,
        loss,
        schedule,
        generator,
        seed,
        lambda_,
        rollout_operation=None,
        frozen_operation=None,
        reference_operation=None,
        run_state=None,
        minimum_tree_size=1,
        cost_safeguard=None,
        allocation_inputs=None,
        benchmark_telemetry=False,
        **arguments,
    ) -> None:
        group = _SingleOptimizerGroup(optimizer)
        self.owners = owners
        self.arguments = {
            "geometry": geometry,
            "owners": owners,
            "modes": (mode,),
            "controllers": (controller,),
            "safeguards": (cost_safeguard,),
            "inference": None,
            "reference_operations": (reference_operation,),
            "optimizers": group,
            "losses": (loss,),
            "schedule": schedule,
            "generators": (generator,),
            "run_states": (run_state,),
            "minimum_tree_sizes": (minimum_tree_size,),
            "seed": seed,
            "lambda_": lambda_,
            "models": (model,),
            "rollout_operations": (
                model_operation
                if rollout_operation is None
                else rollout_operation,
            ),
            "learner_operations": (model_operation,),
            "frozen_operations": (frozen_operation,),
            "allocation_inputs": (
                None if allocation_inputs is None else (allocation_inputs,)
            ),
            "benchmark_telemetry": benchmark_telemetry,
            **arguments,
        }

    def run(self, *, iteration: int):
        engine = _branched_engine(**self.arguments)
        return engine._run_once(
            self.owners, iteration=iteration
        ).results[0]


class SequentialRecoveryHarness:
    def __init__(
        self,
        *,
        owner_factory,
        plan,
        minimum_root_count,
        **arguments,
    ) -> None:
        self.owner_factory = owner_factory
        self.plan = plan
        self.minimum_root_count = minimum_root_count
        self.arguments = arguments

    def run(self, *, iteration: int):
        owners = self.owner_factory()
        first = True

        def owner_factory():
            nonlocal first
            if first:
                first = False
                return owners
            return self.owner_factory()

        engine = _sequential_engine(
            owners=owners,
            owner_factory=owner_factory,
            minimum_root_count=self.minimum_root_count,
            **self.arguments,
        )
        return _result(engine.run(iteration=iteration))


class BranchedRecoveryHarness:
    def __init__(
        self,
        *,
        owner_factory,
        plan,
        controller,
        run_state,
        arguments: Mapping[str, object],
        rank_agreement=None,
    ) -> None:
        if run_state.root_count != plan.manifest.root_count:
            raise ValueError(
                "branched run-state root count must match the storage plan"
            )
        self.owner_factory = owner_factory
        self.plan = plan
        self.controller = controller
        self.run_state = run_state
        self.arguments = dict(arguments)
        self.rank_agreement = rank_agreement

    def run(self, *, iteration: int):
        owners = self.owner_factory()
        first = True

        def owner_factory():
            nonlocal first
            if first:
                first = False
                return owners
            return self.owner_factory()

        arguments = self.arguments
        model = arguments["model"]
        optimizer = arguments["optimizer"]
        engine = _branched_engine(
            owners=owners,
            owner_factory=owner_factory,
            geometry=arguments["geometry"],
            modes=(arguments["mode"],),
            controllers=(self.controller,),
            safeguards=(arguments.get("cost_safeguard"),),
            inference=None,
            reference_operations=(arguments.get("reference_operation"),),
            optimizers=_SingleOptimizerGroup(optimizer),
            losses=(arguments["loss"],),
            schedule=arguments["schedule"],
            generators=(arguments["generator"],),
            run_states=(self.run_state,),
            minimum_tree_sizes=(arguments.get("minimum_tree_size", 1),),
            seed=arguments["seed"],
            lambda_=arguments["lambda_"],
            models=(model,),
            rollout_operations=(
                arguments.get(
                    "rollout_operation",
                    arguments["model_operation"],
                ),
            ),
            learner_operations=(arguments["model_operation"],),
            frozen_operations=(arguments.get("frozen_operation"),),
            completion_samples=arguments.get("completion_samples", 0),
            posterior_max_attempts=arguments.get(
                "posterior_max_attempts", 256
            ),
            precision=arguments.get("precision", "fp32"),
            utility=arguments.get("utility", "raw_bb"),
            cache_reference_outputs=arguments.get(
                "cache_reference_outputs", False
            ),
            benchmark_telemetry=arguments.get(
                "benchmark_telemetry", False
            ),
            allocation_inputs=(
                None
                if arguments.get("allocation_inputs") is None
                else (arguments["allocation_inputs"],)
            ),
        )
        if self.rank_agreement is not None:
            engine.rank_agreement = self.rank_agreement
        return _result(engine.run(iteration=iteration))


def discard_attempt(owners: GeneratedOwners) -> None:
    IterationEngine._finish_owners(owners, entered_index=0)
