"""Test adapters that exercise poker attempts through the sole core engine."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from pbg.core.capacity import GeneratedOwners
from pbg.core.iteration import (
    IterationConfig,
    IterationEngine,
    LaneContext,
)
from pbg.games.poker.queued.branching import (
    PokerBranchedProgram,
)
from pbg.games.poker.queued.iteration import PokerIterationHooks
from pbg.games.poker.queued.training import PokerSequentialProgram


class _LocalAgreement:
    def agree(self, failure):
        return failure


def run_attempt(attempt, *, iteration: int):
    engine = object.__new__(IterationEngine)
    engine.rank_agreement = _LocalAgreement()
    engine.hooks = PokerIterationHooks(lambda _owners: attempt)
    return engine._run_attempt(attempt.owners, iteration=iteration)


class EngineDrivenSequentialAttempt(PokerSequentialProgram):
    def run(self, *, iteration: int):
        return run_attempt(self, iteration=iteration)


class EngineDrivenReplicaBranchedAttempt(PokerBranchedProgram):
    def __init__(self, **arguments) -> None:
        arguments.setdefault(
            "minimum_tree_sizes",
            (1,) * arguments["owners"].plan.manifest.n_replica,
        )
        super().__init__(**arguments)

    def run(self, *, iteration: int):
        return run_attempt(self, iteration=iteration)


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
        assert self._transaction_versions is None
        self.optimizers[0].begin_transaction()
        self._transaction_versions = self.versions.clone()

    def rollback_transaction(self) -> None:
        assert self._transaction_versions is not None
        self.optimizers[0].rollback_transaction()
        self.versions.copy_(self._transaction_versions)
        self._transaction_versions = None

    def commit_transaction(self) -> None:
        assert self._transaction_versions is not None
        self.optimizers[0].commit_transaction()
        self._transaction_versions = None

    def zero_grad(self, lanes) -> None:
        for lane in lanes:
            self.optimizers[int(lane)].zero_grad()

    def lane_stepper(self, lane: int, *, distributed=None):
        assert lane == 0 and distributed is None
        return _LaneStepper(self)


class EngineDrivenBranchedAttempt(PokerBranchedProgram):
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
        super().__init__(
            geometry=geometry,
            owners=owners,
            modes=(mode,),
            controllers=(controller,),
            safeguards=(cost_safeguard,),
            inference=None,
            reference_operations=(reference_operation,),
            optimizers=_SingleOptimizerGroup(optimizer),
            losses=(loss,),
            schedule=schedule,
            generators=(generator,),
            run_states=(run_state,),
            minimum_tree_sizes=(minimum_tree_size,),
            seed=seed,
            lambda_=lambda_,
            models=(model,),
            rollout_operations=(
                model_operation
                if rollout_operation is None
                else rollout_operation,
            ),
            learner_operations=(model_operation,),
            frozen_operations=(frozen_operation,),
            allocation_inputs=(
                None if allocation_inputs is None else (allocation_inputs,)
            ),
            benchmark_telemetry=benchmark_telemetry,
            **arguments,
        )

    def run(self, *, iteration: int):
        return run_attempt(self, iteration=iteration)

    def finish_backward(self):
        return super().finish_backward().results[0]


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
        arguments = self.arguments
        lane = LaneContext(
            mode=arguments["mode"],
            run_state=arguments["run_state"],
            generator=arguments["generator"],
            optimizer=arguments["optimizer"],
            model_operation=arguments["model_operation"],
            loss=arguments["loss"],
            reference_operation=arguments.get("reference_operation"),
            minimum_root_count=self.minimum_root_count,
        )
        return IterationEngine(
            config=IterationConfig(self.plan, (lane,)),
            owner_factory=self.owner_factory,
            hooks=PokerIterationHooks(
                lambda owners: PokerSequentialProgram(
                    owners=owners, **arguments
                )
            ),
            rank_agreement=_LocalAgreement(),
        ).run(iteration=iteration)


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
            raise ValueError("branched root count differs from the storage plan")
        self.owner_factory = owner_factory
        self.plan = plan
        self.controller = controller
        self.run_state = run_state
        self.arguments = dict(arguments)
        self.rank_agreement = (
            _LocalAgreement() if rank_agreement is None else rank_agreement
        )

    def run(self, *, iteration: int):
        arguments = self.arguments
        lane = LaneContext(
            mode=arguments["mode"],
            run_state=self.run_state,
            generator=arguments["generator"],
            optimizer=arguments["optimizer"],
            model_operation=arguments["model_operation"],
            loss=arguments["loss"],
            reference_operation=arguments.get("reference_operation"),
            branch_controller=self.controller,
        )
        return IterationEngine(
            config=IterationConfig(self.plan, (lane,)),
            owner_factory=self.owner_factory,
            hooks=PokerIterationHooks(
                lambda owners: EngineDrivenBranchedAttempt(
                    owners=owners,
                    controller=self.controller,
                    run_state=self.run_state,
                    **arguments,
                )
            ),
            rank_agreement=self.rank_agreement,
        ).run(iteration=iteration)


def discard_attempt(owners: GeneratedOwners) -> None:
    IterationEngine._finish_owners(owners, entered_index=0)
