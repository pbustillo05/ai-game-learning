#!/usr/bin/env python3
"""Profile one representative queued branched-RB poker iteration.

This is a diagnostic consumer, not a training or checkpoint-conversion path.
It loads only the model tensor dictionary from an explicitly supplied
checkpoint and runs the current queued implementation with immutable,
conservative capacities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

import pbg.games.poker.queued.branching as branching_module
import pbg.games.poker.queued.folded as folded_module
from pbg.config import RootConfig, load_config_mapping
from pbg.core.branching import CostSafeguardController, FixedBranchController
from pbg.core.capacity import (
    GeneratedOwners,
    NamedShape,
    PartitionRows,
    build_storage_plan,
)
from pbg.core.contracts import mode_schema
from pbg.core.iteration import IterationConfig, IterationEngine, LaneContext
from pbg.core.optimization import DenseAdam
from pbg.core.persistence import RunState
from pbg.core.replica import ReplicaOptimizerGroup
from pbg.games.poker.queued.config import BranchedModeConfig
from pbg.games.poker.queued.hooks import (
    NLHEGeometry,
    PokerStateMachine,
    RBCompletionBinding,
    SequentialPokerHooks,
    rb_partitions,
)
from pbg.games.poker.queued.model import build_model_from_config
from pbg.games.poker.queued.operations import PokerOperations
from pbg.games.poker.queued.policy import PokerPolicyOperator
from pbg.games.poker.queued.rb_sampling import CollisionConditionedProfileSampler
from pbg.games.poker.queued.replica import PokerReplicaInference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class CudaOperationTimes:
    """Asynchronous CUDA-event attribution with no per-call synchronization."""

    def __init__(self) -> None:
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )
        self.rows: dict[str, int] = defaultdict(int)

    def call(
        self,
        name: str,
        rows: int,
        function: Callable[..., Any],
        *args: object,
        **kwargs: object,
    ) -> Any:
        started = torch.cuda.Event(enable_timing=True, blocking=False)
        finished = torch.cuda.Event(enable_timing=True, blocking=False)
        started.record()
        try:
            return function(*args, **kwargs)
        finally:
            finished.record()
            self.events[name].append((started, finished))
            self.rows[name] += int(rows)

    def report(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, events in self.events.items():
            if events:
                events[-1][1].synchronize()
            seconds = sum(
                started.elapsed_time(finished)
                for started, finished in events
            ) / 1000.0
            result[name] = {
                "calls": len(events),
                "work_rows": self.rows[name],
                "cuda_seconds": seconds,
                "cuda_ms_per_call": 1000.0 * seconds / max(1, len(events)),
            }
        return result


class _Patch:
    def __init__(self, owner: object, name: str, replacement: object) -> None:
        self.owner = owner
        self.name = name
        self.original = getattr(owner, name)
        self.replacement = replacement

    def __enter__(self) -> None:
        setattr(self.owner, self.name, self.replacement)

    def __exit__(self, *_error: object) -> None:
        setattr(self.owner, self.name, self.original)


def _instrument(timing: CudaOperationTimes) -> ExitStack:
    stack = ExitStack()

    original_sampler = CollisionConditionedProfileSampler.sample_children

    def sample_children(self: object, **request: object) -> object:
        source = request["source"]
        assert isinstance(source, torch.Tensor)
        return timing.call(
            "posterior_profile_sample",
            source.numel(),
            original_sampler,
            self,
            **request,
        )

    stack.enter_context(
        _Patch(
            CollisionConditionedProfileSampler,
            "sample_children",
            sample_children,
        )
    )

    original_actor = PokerPolicyOperator.sample_actor

    def sample_actor(
        self: object,
        raw: object,
        legality: object,
        actor_hand: torch.Tensor,
        logical_key: torch.Tensor,
    ) -> object:
        return timing.call(
            "policy_sample_actor",
            actor_hand.numel(),
            original_actor,
            self,
            raw,
            legality,
            actor_hand,
            logical_key,
        )

    stack.enter_context(
        _Patch(PokerPolicyOperator, "sample_actor", sample_actor)
    )

    original_selected = PokerPolicyOperator.selected_log_pi

    def selected_log_pi(
        self: object,
        raw: object,
        legality: object,
        action: torch.Tensor,
    ) -> object:
        return timing.call(
            "policy_selected_log_pi",
            action.numel(),
            original_selected,
            self,
            raw,
            legality,
            action,
        )

    stack.enter_context(
        _Patch(PokerPolicyOperator, "selected_log_pi", selected_log_pi)
    )

    original_resolve = SequentialPokerHooks._resolved_model_context

    def resolved_model_input(self: object, batch: object) -> object:
        rows = int(getattr(batch, "rows", 0))
        return timing.call(
            "resolve_folded_and_live_ranges",
            rows,
            original_resolve,
            self,
            batch,
        )

    stack.enter_context(
        _Patch(
            SequentialPokerHooks,
            "_resolved_model_context",
            resolved_model_input,
        )
    )

    original_condition = branching_module.action_conditioned_log_reach

    def condition(
        log_reach: torch.Tensor,
        selected_log_policy: torch.Tensor,
        actor: torch.Tensor,
    ) -> object:
        return timing.call(
            "action_conditioned_reach_update",
            log_reach.shape[0],
            original_condition,
            log_reach,
            selected_log_policy,
            actor,
        )

    stack.enter_context(
        _Patch(branching_module, "action_conditioned_log_reach", condition)
    )

    original_deal = PokerStateMachine.deal_next_street_keyed

    def deal_next_street_keyed(
        self: object, state: object, **request: object
    ) -> object:
        rows = int(getattr(state, "rows", 0))
        return timing.call(
            "chance_card_deal",
            rows,
            original_deal,
            self,
            state,
            **request,
        )

    stack.enter_context(
        _Patch(
            PokerStateMachine,
            "deal_next_street_keyed",
            deal_next_street_keyed,
        )
    )

    original_folded = folded_module.folded_statistics

    def folded_statistics(geometry: object, **request: object) -> object:
        board = request["board_cards"]
        assert isinstance(board, torch.Tensor)
        return timing.call(
            "folded_statistics",
            board.shape[0],
            original_folded,
            geometry,
            **request,
        )

    stack.enter_context(
        _Patch(folded_module, "folded_statistics", folded_statistics)
    )

    original_completion = RBCompletionBinding.leaf_to_layer_work

    def complete(
        self: object, batch: object
    ) -> object:
        rows = int(getattr(batch, "rows", 0))
        return timing.call(
            "rb_leaf_completion",
            rows,
            original_completion,
            self,
            batch,
        )

    stack.enter_context(
        _Patch(
            RBCompletionBinding,
            "leaf_to_layer_work",
            complete,
        )
    )
    return stack


@dataclass(frozen=True)
class ProfileScenario:
    """One exact historical or user-edited profiler invocation."""

    name: str
    output: Path
    fixed_beta: Mapping[str, float]
    minimum_tree_size: int
    safe_tree_size: float | None
    cost_safeguard: Mapping[str, object] | None
    completion_samples: int
    posterior_max_attempts: int
    seed: int
    iteration: int
    warmup: int
    historical_physical_gpu: int
    expected_outcome: str


@dataclass(frozen=True)
class ProfileConfig:
    """Strict diagnostic fields layered over one public runtime config."""

    source: Path
    root: RootConfig
    checkpoint: Path
    checkpoint_sha256: str
    scenario: ProfileScenario


def _load_profile_config(
    path: Path, requested_scenario: str | None
) -> ProfileConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("profile config root must be an object")
    source = dict(raw)
    if set(source) != {"scientific", "execution", "profile"}:
        raise ValueError(
            "profile config requires exactly scientific, execution, and profile"
        )
    profile_raw = source.pop("profile")
    if not isinstance(profile_raw, Mapping):
        raise TypeError("profile must be an object")
    profile = dict(profile_raw)
    expected_profile = {
        "checkpoint",
        "checkpoint_sha256",
        "default_scenario",
        "scenarios",
    }
    if set(profile) != expected_profile:
        unknown = sorted(set(profile).difference(expected_profile))
        missing = sorted(expected_profile.difference(profile))
        name = unknown[0] if unknown else missing[0]
        detail = "unknown" if unknown else "missing"
        raise ValueError(f"profile.{name}: {detail} field")
    for name in ("checkpoint", "checkpoint_sha256", "default_scenario"):
        if not isinstance(profile[name], str) or not profile[name]:
            raise ValueError(f"profile.{name}: expected nonempty string")
    digest = str(profile["checkpoint_sha256"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("profile.checkpoint_sha256: expected lowercase SHA-256")
    scenarios_raw = profile["scenarios"]
    if not isinstance(scenarios_raw, Mapping) or not scenarios_raw:
        raise ValueError("profile.scenarios: expected nonempty object")
    scenario_name = requested_scenario or str(profile["default_scenario"])
    if scenario_name not in scenarios_raw:
        raise ValueError(f"profile.scenarios.{scenario_name}: unknown scenario")
    scenario_raw = scenarios_raw[scenario_name]
    if not isinstance(scenario_raw, Mapping):
        raise TypeError(
            f"profile.scenarios.{scenario_name}: expected an object"
        )
    scenario = dict(scenario_raw)
    expected_scenario = {
        "output",
        "fixed_beta",
        "minimum_tree_size",
        "safe_tree_size",
        "cost_safeguard",
        "completion_samples",
        "posterior_max_attempts",
        "seed",
        "iteration",
        "warmup",
        "historical_physical_gpu",
        "expected_outcome",
    }
    if set(scenario) != expected_scenario:
        unknown = sorted(set(scenario).difference(expected_scenario))
        missing = sorted(expected_scenario.difference(scenario))
        name = unknown[0] if unknown else missing[0]
        detail = "unknown" if unknown else "missing"
        raise ValueError(
            f"profile.scenarios.{scenario_name}.{name}: {detail} field"
        )
    fixed_beta_raw = scenario["fixed_beta"]
    if (
        not isinstance(fixed_beta_raw, Mapping)
        or set(fixed_beta_raw) != {"player", "chance"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 1 <= float(value) <= 3
            for value in fixed_beta_raw.values()
        )
    ):
        raise ValueError(
            f"profile.scenarios.{scenario_name}.fixed_beta: invalid beta"
        )
    for name in (
        "minimum_tree_size",
        "completion_samples",
        "posterior_max_attempts",
        "seed",
        "iteration",
        "warmup",
        "historical_physical_gpu",
    ):
        value = scenario[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"profile.scenarios.{scenario_name}.{name}: expected integer"
            )
    for name in (
        "minimum_tree_size",
        "completion_samples",
        "posterior_max_attempts",
    ):
        if scenario[name] <= 0:
            raise ValueError(
                f"profile.scenarios.{scenario_name}.{name}: must be positive"
            )
    if scenario["iteration"] < 0:
        raise ValueError(
            f"profile.scenarios.{scenario_name}.iteration: must be nonnegative"
        )
    if scenario["warmup"] not in (0, 1):
        raise ValueError(
            f"profile.scenarios.{scenario_name}.warmup: expected zero or one"
        )
    if scenario["historical_physical_gpu"] < 0:
        raise ValueError(
            f"profile.scenarios.{scenario_name}.historical_physical_gpu: "
            "must be nonnegative"
        )
    if (
        not isinstance(scenario["output"], str)
        or not scenario["output"]
        or not isinstance(scenario["expected_outcome"], str)
        or not scenario["expected_outcome"]
    ):
        raise ValueError(
            f"profile.scenarios.{scenario_name}: output and expected_outcome "
            "must be nonempty strings"
        )
    safe_tree_size = scenario["safe_tree_size"]
    if safe_tree_size is not None and (
        isinstance(safe_tree_size, bool)
        or not isinstance(safe_tree_size, (int, float))
        or float(safe_tree_size) < 1
    ):
        raise ValueError(
            f"profile.scenarios.{scenario_name}.safe_tree_size: invalid bound"
        )
    safeguard_raw = scenario["cost_safeguard"]
    safeguard: Mapping[str, object] | None
    if safeguard_raw is None:
        safeguard = None
    elif isinstance(safeguard_raw, Mapping):
        values = dict(safeguard_raw)
        expected_safeguard = {
            "gamma_deviance_max",
            "p95_underprediction_max",
            "consecutive_iterations",
            "consecutive_successes",
            "active",
            "observations",
        }
        if set(values) != expected_safeguard:
            raise ValueError(
                f"profile.scenarios.{scenario_name}.cost_safeguard: "
                "unknown or missing state"
            )
        CostSafeguardController(**values)
        safeguard = values
    else:
        raise ValueError(
            f"profile.scenarios.{scenario_name}.cost_safeguard: "
            "expected object or null"
        )
    root = load_config_mapping(source)
    if (
        not isinstance(root.scientific, BranchedModeConfig)
        or dict(root.mode)
        != {
            "training_regime": "self_play",
            "rollout_mode": "branched.fixed",
            "private_estimator": "rao_blackwell",
        }
        or root.execution.manifest.n_replica != 1
        or root.scientific.common.root_count != 1
    ):
        raise ValueError(
            "profiler requires one-lane, one-root, self-play fixed branched RB"
        )
    return ProfileConfig(
        source=path.resolve(),
        root=root,
        checkpoint=Path(str(profile["checkpoint"])).resolve(),
        checkpoint_sha256=digest,
        scenario=ProfileScenario(
            name=scenario_name,
            output=Path(str(scenario["output"])).resolve(),
            fixed_beta={
                name: float(value) for name, value in fixed_beta_raw.items()
            },
            minimum_tree_size=int(scenario["minimum_tree_size"]),
            safe_tree_size=(
                None if safe_tree_size is None else float(safe_tree_size)
            ),
            cost_safeguard=safeguard,
            completion_samples=int(scenario["completion_samples"]),
            posterior_max_attempts=int(
                scenario["posterior_max_attempts"]
            ),
            seed=int(scenario["seed"]),
            iteration=int(scenario["iteration"]),
            warmup=int(scenario["warmup"]),
            historical_physical_gpu=int(
                scenario["historical_physical_gpu"]
            ),
            expected_outcome=str(scenario["expected_outcome"]),
        ),
    )


def _run_once(
    *,
    plan: object,
    config: RootConfig,
    geometry: NLHEGeometry,
    model: torch.nn.Module,
    optimizer: DenseAdam,
    device: torch.device,
    scenario: ProfileScenario,
    timing: CudaOperationTimes | None,
    beta_max: int | Mapping[str, int] | None = None,
) -> object:
    scientific = config.scientific
    assert isinstance(scientific, BranchedModeConfig)
    common = scientific.common
    controller = FixedBranchController(
        scenario.fixed_beta,
        (
            int(scientific.branching["beta_max"])
            if beta_max is None
            else beta_max
        ),
        minimum_scale=float(scientific.branching["minimum_scale"]),
        backoff_factor=float(scientific.branching["backoff_factor"]),
        safe_tree_size=scenario.safe_tree_size,
    )
    generator = torch.Generator(device=device).manual_seed(scenario.seed)
    run_state = RunState(
        config.execution.run_id,
        root_count=common.root_count,
    )
    safeguard = (
        None
        if scenario.cost_safeguard is None
        else CostSafeguardController(**dict(scenario.cost_safeguard))
    )
    context = _instrument(timing) if timing is not None else ExitStack()
    inference = PokerReplicaInference((model,))
    optimizers = ReplicaOptimizerGroup(
        inference.modules,
        optimizer.config,
        scalers=(optimizer.scaler,),
    )
    with context:
        lane_model = inference.modules.lane_module(0)
        operations = PokerOperations(
            geometry=geometry,
            plan=plan,
            modes=(plan.mode,),
            models=(lane_model,),
            rollout_operations=(lane_model.rollout_operation,),
            frozen_operations=(None,),
            seed=scenario.seed,
            precision=common.precision.name,
            utility=common.utility,
            completion_samples=scenario.completion_samples,
            posterior_max_attempts=scenario.posterior_max_attempts,
        )
        result = IterationEngine(
            config=IterationConfig(
                plan,
                (
                    LaneContext(
                        mode=plan.mode,
                        run_state=run_state,
                        generator=generator,
                        optimizer=optimizers.optimizers[0],
                        model_operation=lane_model.rb_learner_operation,
                        loss=common.loss,
                        model=lane_model,
                        branch_controller=controller,
                        cost_safeguard=safeguard,
                        minimum_tree_size=scenario.minimum_tree_size,
                    ),
                ),
                common.schedule,
                optimizers,
                common.lambda_,
                precision=common.precision.name,
                cache_reference_outputs=common.reference.cache_outputs,
                benchmark_telemetry=True,
            ),
            owner_factory=lambda: GeneratedOwners(plan),
            operations=operations,
        ).run(iteration=scenario.iteration)
    lane_result = result.results[0]
    if lane_result is None:
        raise RuntimeError("profiled iteration did not produce a lane result")
    return lane_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile one config-declared fixed-beta branching workload"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--player-pre-beta", type=float)
    parser.add_argument("--player-post-beta", type=float)
    parser.add_argument("--chance-beta", type=float)
    parser.add_argument("--large-tree-manifest", action="store_true")
    parser.add_argument("--safe-tree-size", type=float)
    parser.add_argument("--player-pre-beta-max", type=int)
    parser.add_argument("--player-post-beta-max", type=int, default=3)
    parser.add_argument("--chance-beta-max", type=int, default=3)
    args = parser.parse_args()

    profile = _load_profile_config(Path(args.config), args.scenario)
    scenario = profile.scenario
    split_values = (args.player_pre_beta, args.player_post_beta)
    stage_beta_max: Mapping[str, int] | None = None
    if any(value is not None for value in split_values):
        if not all(value is not None for value in split_values):
            parser.error(
                "--player-pre-beta and --player-post-beta must be supplied together"
            )
        chance_beta = (
            float(args.chance_beta)
            if args.chance_beta is not None
            else float(scenario.fixed_beta["chance"])
        )
        fixed_beta = {
            "player_pre": float(args.player_pre_beta),
            "player_post": float(args.player_post_beta),
            "chance": chance_beta,
        }
        pre_beta_max = (
            int(args.player_pre_beta_max)
            if args.player_pre_beta_max is not None
            else math.ceil(float(args.player_pre_beta))
        )
        stage_beta_max = {
            "player_pre": pre_beta_max,
            "player_post": int(args.player_post_beta_max),
            "chance": int(args.chance_beta_max),
        }
        if any(value < 1 for value in stage_beta_max.values()) or any(
            fixed_beta[name] > stage_beta_max[name]
            for name in fixed_beta
        ):
            parser.error("beta overrides exceed their positive stage maxima")
        scenario = ProfileScenario(
            **{
                **asdict(scenario),
                "fixed_beta": fixed_beta,
            }
        )
    elif args.chance_beta is not None:
        parser.error("--chance-beta requires the two stage-specific player betas")
    if args.seed is not None:
        scenario = ProfileScenario(
            **{**asdict(scenario), "seed": int(args.seed)}
        )
    if args.output is not None:
        scenario = ProfileScenario(
            **{**asdict(scenario), "output": Path(args.output).resolve()}
        )
    if args.safe_tree_size is not None:
        if args.safe_tree_size < 1:
            parser.error("--safe-tree-size must be at least one")
        scenario = ProfileScenario(
            **{
                **asdict(scenario),
                "safe_tree_size": float(args.safe_tree_size),
            }
        )
    config = profile.root
    if args.large_tree_manifest:
        experiment_geometry = NLHEGeometry(
            config.scientific.common.game.players,
            config.scientific.common.game.stack_bb,
            config.scientific.common.game.rank_count,
        )
        manifest = config.execution.manifest
        manifest = replace(
            manifest,
            beta_max=(
                manifest.beta_max
                if stage_beta_max is None
                else max(stage_beta_max.values())
            ),
            player_node_cap=(
                PartitionRows(
                    "pre",
                    2_000
                    * (
                        manifest.beta_max
                        if stage_beta_max is None
                        else stage_beta_max["player_pre"]
                    )
                    * (
                        manifest.beta_max
                        if stage_beta_max is None
                        else stage_beta_max["chance"]
                    )
                    ** 3,
                ),
                PartitionRows("post_2", 48_000),
            ),
            trained_player_cap=(
                PartitionRows("pre", 2_000),
                PartitionRows("post_2", 48_000),
            ),
            action_cap=(
                PartitionRows(
                    "pre",
                    2_000
                    * (
                        manifest.beta_max
                        if stage_beta_max is None
                        else stage_beta_max["player_pre"]
                    ),
                ),
                PartitionRows("post", 149_999),
            ),
            folded_range_cap=(
                PartitionRows(
                    "pre",
                    64
                    * (
                        manifest.beta_max
                        if stage_beta_max is None
                        else stage_beta_max["player_pre"]
                    ),
                ),
                PartitionRows("post", 192),
            ),
            leaf_cap=80_000,
            topology_cap=150_000,
            topology_edge_cap=149_999,
            family_pack_cap=64,
            chance_frontier_cap=(
                64
                * (
                    manifest.beta_max
                    if stage_beta_max is None
                    else stage_beta_max["player_pre"]
                )
                * (
                    manifest.beta_max
                    if stage_beta_max is None
                    else stage_beta_max["chance"]
                )
                ** 3
            ),
            completion_pack_cap=8_192,
            topology_packet_cap=(
                64
                * (
                    manifest.beta_max
                    if stage_beta_max is None
                    else stage_beta_max["player_pre"]
                )
                * sum(
                    (
                        manifest.beta_max
                        if stage_beta_max is None
                        else stage_beta_max["chance"]
                    )
                    ** generation
                    for generation in range(4)
                )
            ),
            learner_action_pack_cap=(
                20
                * (
                    manifest.beta_max
                    if stage_beta_max is None
                    else max(stage_beta_max.values())
                )
            ),
            target_arena_cap=230_000,
        )
        config = replace(
            config,
            execution=replace(
                config.execution,
                manifest=manifest,
                layout=replace(
                    config.execution.layout,
                    model_input_shapes=(
                        NamedShape(
                            "pre",
                            (experiment_geometry.preflop_model_input_width,),
                        ),
                        NamedShape(
                            "post_2",
                            (experiment_geometry.postflop_model_input_width,),
                        ),
                    ),
                ),
            ),
        )
    common = config.scientific.common
    device = torch.device(config.execution.device)
    torch.cuda.set_device(device)
    geometry = NLHEGeometry(
        common.game.players,
        common.game.stack_bb,
        common.game.rank_count,
    )
    scientific = config.scientific_mapping()
    model = build_model_from_config(scientific, device)
    model.configure_execution(
        config.execution.attention_backend,
        config.execution.query_chunk_size,
    )
    checkpoint = profile.checkpoint
    actual_checkpoint_sha256 = _sha256(checkpoint)
    if actual_checkpoint_sha256 != profile.checkpoint_sha256:
        raise ValueError("profile checkpoint SHA-256 differs from configuration")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise TypeError("checkpoint has no model tensor dictionary")
    model.load_state_dict(state, strict=True)
    del payload, state
    optimizer = DenseAdam(model, common.optimizer)
    mode = mode_schema(**dict(config.mode))
    plan = build_storage_plan(
        mode,
        rb_partitions(geometry),
        config.execution.manifest,
        config.execution.layout,
        device,
    )

    if scenario.warmup:
        _run_once(
            plan=plan,
            config=config,
            geometry=geometry,
            model=model,
            optimizer=optimizer,
            device=device,
            scenario=scenario,
            timing=None,
            beta_max=stage_beta_max,
        )
        torch.cuda.synchronize(device)

    timing = CudaOperationTimes()
    started = time.perf_counter()
    result = _run_once(
        plan=plan,
        config=config,
        geometry=geometry,
        model=model,
        optimizer=optimizer,
        device=device,
        scenario=scenario,
        timing=timing,
        beta_max=stage_beta_max,
    )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(device)
    try:
        git_head = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unknown"
    report = {
        "schema": "pbg-queued-branching-operation-profile-v1",
        "notes": [
            "CUDA operation scopes may be nested and must not be summed.",
            "HU folds are terminal, so folded_statistics is not exercised.",
            "Only model tensors are read from the supplied historical checkpoint.",
        ],
        "identity": {
            "git_head": git_head,
            "config": str(profile.source),
            "config_sha256": _sha256(profile.source),
            "scenario": scenario.name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_checkpoint_sha256,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": properties.name,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
        },
        "workload": {
            "scientific": scientific,
            "execution": config.execution.mapping(),
            "game": {
                "players": common.game.players,
                "stack_bb": common.game.stack_bb,
                "rank_count": common.game.rank_count,
            },
            "fixed_beta": dict(scenario.fixed_beta),
            "beta_max": (
                config.scientific.branching["beta_max"]
                if stage_beta_max is None
                else dict(stage_beta_max)
            ),
            "minimum_tree_size": scenario.minimum_tree_size,
            "safe_tree_size": scenario.safe_tree_size,
            "cost_safeguard": (
                None
                if scenario.cost_safeguard is None
                else dict(scenario.cost_safeguard)
            ),
            "completion_samples": scenario.completion_samples,
            "posterior_max_attempts": scenario.posterior_max_attempts,
            "seed": scenario.seed,
            "iteration": scenario.iteration,
            "warmup": scenario.warmup,
            "scientific_caps": {
                "player_pre": 2_000,
                "player_post": 48_000,
                "player_total": 50_000,
                "leaf": 80_000,
            }
            if args.large_tree_manifest
            else None,
            "historical_physical_gpu": scenario.historical_physical_gpu,
            "expected_outcome": scenario.expected_outcome,
        },
        "result": {
            "wall_seconds": wall_seconds,
            "player_nodes": result.player_nodes,
            "leaves": result.leaves,
            "executed_edges": result.executed_edges,
            "sampled_tree_nodes": result.sampled_tree_nodes,
            "pre_refill_nodes": result.pre_refill_nodes,
            "refill_count": result.refill_count,
            "duplicate_child_labels": result.duplicate_child_labels,
            "posterior_effective_hands": result.posterior_effective_hands,
            "telemetry": asdict(result.telemetry),
            "observability": dict(result.observability),
        },
        "operation_cuda": timing.report(),
    }
    destination = scenario.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
