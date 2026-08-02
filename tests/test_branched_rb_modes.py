from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from pbg.config import load_config_mapping
from pbg.core.branching import (
    CostSafeguardController,
    FixedBranchController,
    KKTBranchController,
)
from pbg.core.capacity import GeneratedOwners, PartitionRows, build_storage_plan
from pbg.core.contracts import mode_schema
from pbg.core.iteration import IterationTelemetry
from pbg.core.learner import (
    BranchingLossConfig,
    LearnerSchedule,
    LossConfig,
)
from pbg.core.optimization import DenseAdam, OptimizerConfig
from pbg.core.persistence import RunState
from pbg.games.poker.queued.branching import (
    BranchedPosteriorExhaustion,
    action_conditioned_log_reach,
)
from pbg.games.poker.queued.config import BranchedModeConfig
from pbg.games.poker.queued.hooks import (
    NLHEGeometry,
    rb_partitions,
)
from pbg.games.poker.queued.model import build_model_from_config
from pbg.games.poker.queued.rb_sampling import (
    POSTERIOR_MAX_ATTEMPTS,
    CollisionConditionedProfileSampler,
    RejectionObservation,
)
from tests.iteration_support import BranchedRecoveryHarness
from tests.test_branched_sampled_modes import (
    _manifest,
    _phase7_layout,
    _scientific_config,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase 8 requires CUDA")


def _config(regime: str, rollout: str) -> dict[str, object]:
    result = _scientific_config(regime, rollout)
    result["scientific"]["mode"]["private_estimator"] = "rao_blackwell"
    result["scientific"]["completion"] = {"samples": 4}
    return result


def _plan(
    device: torch.device,
    mode,
    geometry: NLHEGeometry,
    *,
    n_replica: int = 1,
):
    partitions = rb_partitions(geometry)
    manifest = replace(
        _manifest(device, geometry),
        n_replica=n_replica,
        action_cap=(
            PartitionRows("pre", 2048),
            PartitionRows("post", 2048),
        ),
        completion_workspace_bytes=16 << 20,
        learner_workspace_bytes=24 << 20,
    )
    layout = replace(
        _phase7_layout(geometry),
    )
    return build_storage_plan(mode, partitions, manifest, layout, device)


@pytest.mark.parametrize("regime", ("self_play", "best_response"))
@pytest.mark.parametrize("rollout", ("branched.fixed", "branched.KKT"))
def test_phase8_config_closes_all_four_rb_modes(regime: str, rollout: str) -> None:
    value = _config(regime, rollout)
    configured = load_config_mapping(value).scientific
    assert isinstance(configured, BranchedModeConfig)
    assert dict(configured.mode) == value["scientific"]["mode"]
    missing = dict(value)
    missing["scientific"] = dict(value["scientific"])
    del missing["scientific"]["completion"]
    with pytest.raises(ValueError, match="root.scientific.completion"):
        load_config_mapping(missing)


@CUDA
def test_collision_sampler_reuses_child_zero_and_is_subdivision_invariant() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    sampler = CollisionConditionedProfileSampler(geometry, max_attempts=16)
    weights = torch.ones(
        (2, 2, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    board = torch.full((2, 5), 255, dtype=torch.uint8, device=device)
    first = geometry.first_card
    inherited_cards = torch.tensor(
        (
            ((first, first + 1), (first + 2, first + 3)),
            ((first + 4, first + 5), (first + 6, first + 7)),
        ),
        dtype=torch.int64,
        device=device,
    )
    inherited = geometry.combo_id(inherited_cards[..., 0], inherited_cards[..., 1])
    source = torch.tensor((0, 0, 1, 1), dtype=torch.int64, device=device)
    ordinal = torch.tensor((0, 1, 0, 1), dtype=torch.int64, device=device)
    active = torch.ones(4, dtype=torch.bool, device=device)
    root = torch.tensor((5, 5, 6, 6), dtype=torch.int64, device=device)
    path = torch.tensor((11, 12, 21, 22), dtype=torch.int64, device=device)

    def sample(selected: torch.Tensor) -> torch.Tensor:
        return sampler.sample_children(
            range_weights=weights,
            board_cards=board,
            inherited_profile=inherited,
            source=source.index_select(0, selected),
            ordinal=ordinal.index_select(0, selected),
            active=active.index_select(0, selected),
            root=root.index_select(0, selected),
            path=path.index_select(0, selected),
            seed=73,
            iteration=4,
            retry_epoch=2,
        )

    whole = sample(torch.arange(4, device=device))
    observation = sampler.last_observation
    assert observation is not None
    assert observation.attempt_count[::2].tolist() == [0, 0]
    split = torch.cat(
        (
            sample(torch.arange(2, device=device)),
            sample(torch.arange(2, 4, device=device)),
        )
    )
    torch.testing.assert_close(whole, split)
    torch.testing.assert_close(whole[::2], inherited)
    cards = sampler._combo_cards(whole.flatten()).reshape(4, 2, 2)
    assert bool(cards[:, 0, :, None].ne(cards[:, 1, None, :]).all())


@CUDA
def test_collision_sampler_clamps_rounded_total_before_fixed_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    sampler = CollisionConditionedProfileSampler(geometry, max_attempts=1)
    first = geometry.first_card
    selected = geometry.combo_id(
        torch.tensor(
            ((first, first + 2), (first + 4, first + 6)),
            dtype=torch.int64,
            device=device,
        )[:, 0],
        torch.tensor(
            ((first, first + 2), (first + 4, first + 6)),
            dtype=torch.int64,
            device=device,
        )[:, 1],
    )
    weights = torch.zeros(
        (1, 2, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    weights[0, 0, selected[0]] = 1.0
    weights[0, 1, selected[1]] = 1.0
    monkeypatch.setattr(
        "pbg.games.poker.queued.rb_sampling.logical_uniform",
        lambda keys: torch.ones_like(keys, dtype=torch.float32),
    )
    sampled = sampler.sample_children(
        range_weights=weights,
        board_cards=torch.full((1, 5), 255, dtype=torch.uint8, device=device),
        inherited_profile=selected[None].to(torch.int32),
        source=torch.zeros(1, dtype=torch.int64, device=device),
        ordinal=torch.ones(1, dtype=torch.int64, device=device),
        active=torch.ones(1, dtype=torch.bool, device=device),
        root=torch.zeros(1, dtype=torch.int64, device=device),
        path=torch.ones(1, dtype=torch.int64, device=device),
        seed=1,
        iteration=0,
        retry_epoch=0,
    )
    torch.testing.assert_close(sampled[0], selected.to(torch.int32))
    assert not bool(sampler.last_observation.exhausted[0])


@CUDA
def test_root_refill_uses_fresh_posterior_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "rao_blackwell")
    plan = _plan(device, mode, geometry)
    observed_ordinals: list[int] = []

    def observe(
        sampler: CollisionConditionedProfileSampler,
        **request,
    ) -> torch.Tensor:
        active = request["active"]
        observed_ordinals.extend(request["ordinal"][active].detach().cpu().tolist())
        source = request["source"].to(torch.int64)
        sampler.last_observation = RejectionObservation(
            torch.zeros_like(source, dtype=torch.int32),
            torch.zeros_like(active),
            torch.full(
                (),
                torch.iinfo(torch.int64).max,
                dtype=torch.int64,
                device=device,
            ),
        )
        return request["inherited_profile"].index_select(0, source)

    monkeypatch.setattr(
        CollisionConditionedProfileSampler,
        "sample_children",
        observe,
    )
    model = build_model_from_config(
        _config("self_play", "branched.fixed")["scientific"], device
    )
    result = BranchedRecoveryHarness(
        owner_factory=lambda: GeneratedOwners(plan),
        plan=plan,
        controller=FixedBranchController({"player": 1.0, "chance": 1.0}, 2),
        run_state=RunState("phase8-refill-profile", root_count=1),
        arguments={
            "geometry": geometry,
            "mode": mode,
            "model": model,
            "model_operation": model.rb_learner_operation,
            "rollout_operation": model.rollout_operation,
            "optimizer": DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5)),
            "loss": LossConfig(
                branching=BranchingLossConfig(0.1, 0.1, 1.0e-6),
            ),
            "schedule": LearnerSchedule(batch_size=2, epochs=None, gradient_steps=1),
            "generator": torch.Generator(device=device).manual_seed(127),
            "seed": 127,
            "lambda_": 0.95,
            "minimum_tree_size": 100,
            "completion_samples": 4,
        },
    ).run(iteration=0)
    degree = result.root_executed_degree[0]
    assert degree > result.root_target_degree[0]
    assert sorted(value for value in observed_ordinals if value > 0) == list(
        range(1, degree)
    )


@CUDA
def test_collision_sampler_matches_the_exact_conditioned_sparse_law() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    sampler = CollisionConditionedProfileSampler(geometry, max_attempts=16)
    first = geometry.first_card

    def combo(a: int, b: int) -> int:
        value = geometry.combo_id(
            torch.tensor(a, dtype=torch.int64, device=device),
            torch.tensor(b, dtype=torch.int64, device=device),
        )
        return int(value)

    seat_zero = (
        combo(first, first + 1),
        combo(first + 2, first + 3),
    )
    seat_one = (
        combo(first, first + 2),
        combo(first + 4, first + 5),
    )
    weights = torch.zeros(
        (1, 2, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    weights[0, 0, seat_zero[0]] = 1
    weights[0, 0, seat_zero[1]] = 2
    weights[0, 1, seat_one[0]] = 1
    weights[0, 1, seat_one[1]] = 3
    samples = 4096
    sampled = sampler.sample_children(
        range_weights=weights,
        board_cards=torch.full((1, 5), 255, dtype=torch.uint8, device=device),
        inherited_profile=torch.tensor(
            ((seat_zero[0], seat_one[1]),),
            dtype=torch.int32,
            device=device,
        ),
        source=torch.zeros(samples, dtype=torch.int64, device=device),
        ordinal=torch.ones(samples, dtype=torch.int64, device=device),
        active=torch.ones(samples, dtype=torch.bool, device=device),
        root=torch.zeros(samples, dtype=torch.int64, device=device),
        path=torch.arange(1, samples + 1, dtype=torch.int64, device=device),
        seed=719,
        iteration=3,
        retry_epoch=0,
    )
    observation = sampler.last_observation
    assert observation is not None
    assert not bool(observation.exhausted.any())
    assert bool(sampled[:, 1].eq(seat_one[1]).all())
    first_probability = sampled[:, 0].eq(seat_zero[0]).float().mean()
    # Conditioning removes both proposals using seat_one[0], leaving
    # product masses 3 and 6 for the two valid joint profiles.
    assert float(first_probability) == pytest.approx(1 / 3, abs=0.04)


@CUDA
def test_action_conditioned_folded_range_drives_chance_profiles() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    first = geometry.first_card
    hands = geometry.combo_id(
        torch.tensor(
            (first, first + 2, first + 4),
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor(
            (first + 1, first + 3, first + 5),
            dtype=torch.int64,
            device=device,
        ),
    ).to(torch.int64)
    log_reach = torch.full(
        (1, 2, geometry.private_hands),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    log_reach[0, 0, hands[:2]] = torch.log(torch.tensor((0.25, 0.75), device=device))
    log_reach[0, 1, hands[2]] = 0
    selected_log = torch.full(
        (1, geometry.private_hands),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    selected_log[0, hands[:2]] = torch.log(torch.tensor((0.9, 0.1), device=device))
    conditioned = action_conditioned_log_reach(
        log_reach,
        selected_log,
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(conditioned[:, 1], log_reach[:, 1])
    weights = conditioned.exp()
    weights /= weights.sum(-1, keepdim=True)
    torch.testing.assert_close(
        weights[0, 0, hands[:2]],
        torch.tensor((0.75, 0.25), device=device),
    )

    samples = 2048
    sampler = CollisionConditionedProfileSampler(geometry, max_attempts=8)
    inherited = torch.tensor(
        ((int(hands[0]), int(hands[2])),),
        dtype=torch.int32,
        device=device,
    )
    profile = sampler.sample_children(
        range_weights=weights,
        board_cards=torch.full((1, 5), 255, dtype=torch.uint8, device=device),
        inherited_profile=inherited,
        source=torch.zeros(samples, dtype=torch.int64, device=device),
        ordinal=torch.ones(samples, dtype=torch.int64, device=device),
        active=torch.ones(samples, dtype=torch.bool, device=device),
        root=torch.zeros(samples, dtype=torch.int64, device=device),
        path=torch.arange(1, samples + 1, dtype=torch.int64, device=device),
        seed=811,
        iteration=2,
        retry_epoch=0,
    )
    assert bool(profile[:, 1].eq(hands[2]).all())
    assert float(profile[:, 0].eq(hands[0]).float().mean()) == (
        pytest.approx(0.75, abs=0.04)
    )


@CUDA
@pytest.mark.parametrize(
    ("players", "ranks"),
    ((2, 4), (3, 4), (4, 4), (5, 4), (6, 5)),
)
def test_conditioned_sampler_covers_player_rank_boundaries_and_zero_plateaus(
    players: int, ranks: int
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(players, 8.0, ranks)
    sampler = CollisionConditionedProfileSampler(geometry, max_attempts=8)
    first = geometry.first_card
    desired_cards = torch.tensor(
        [(first + 1 + 2 * seat, first + 2 + 2 * seat) for seat in range(players)],
        dtype=torch.int64,
        device=device,
    )
    desired = geometry.combo_id(desired_cards[:, 0], desired_cards[:, 1])
    conflicting = geometry.combo_id(
        torch.full((players,), first, dtype=torch.int64, device=device),
        desired_cards[:, 0],
    )
    weights = torch.zeros(
        (1, players, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    seat = torch.arange(players, device=device)
    weights[0, seat, desired.to(torch.int64)] = 1
    weights[0, seat, conflicting.to(torch.int64)] = 1_000
    board = torch.full((1, 5), 255, dtype=torch.uint8, device=device)
    board[0, 0] = first
    sampled = sampler.sample_children(
        range_weights=weights,
        board_cards=board,
        inherited_profile=desired[None].to(torch.int32),
        source=torch.zeros(1, dtype=torch.int64, device=device),
        ordinal=torch.ones(1, dtype=torch.int64, device=device),
        active=torch.ones(1, dtype=torch.bool, device=device),
        root=torch.tensor((players,), dtype=torch.int64, device=device),
        path=torch.tensor((ranks,), dtype=torch.int64, device=device),
        seed=101,
        iteration=7,
        retry_epoch=0,
    )
    torch.testing.assert_close(sampled[0], desired.to(torch.int32))


@CUDA
def test_rejection_tail_reports_minimum_address_and_full_attempt_count() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    sampler = CollisionConditionedProfileSampler(geometry, max_attempts=3)
    weights = torch.zeros(
        (1, 2, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    weights[:, :, 0] = 1
    inherited = torch.zeros((1, 2), dtype=torch.int32, device=device)
    sampler.sample_children(
        range_weights=weights,
        board_cards=torch.full((1, 5), 255, dtype=torch.uint8, device=device),
        inherited_profile=inherited,
        source=torch.zeros(1, dtype=torch.int64, device=device),
        ordinal=torch.ones(1, dtype=torch.int64, device=device),
        active=torch.ones(1, dtype=torch.bool, device=device),
        root=torch.tensor((9,), dtype=torch.int64, device=device),
        path=torch.tensor((17,), dtype=torch.int64, device=device),
        seed=3,
        iteration=2,
        retry_epoch=0,
    )
    observation = sampler.last_observation
    assert observation is not None
    assert observation.exhausted.tolist() == [True]
    assert observation.attempt_count.tolist() == [3]
    assert int(observation.minimum_failing_address) == 17


@CUDA
def test_rejection_exhaustion_blocks_commit_and_publishes_device_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "rao_blackwell")
    plan = _plan(device, mode, geometry)

    def exhaust(
        sampler: CollisionConditionedProfileSampler,
        **request,
    ) -> torch.Tensor:
        source = request["source"].to(torch.int64)
        inherited = request["inherited_profile"].index_select(0, source)
        fresh = request["active"] & request["ordinal"].gt(0)
        attempts = torch.where(
            fresh,
            torch.full_like(source, sampler.max_attempts, dtype=torch.int32),
            torch.zeros_like(source, dtype=torch.int32),
        )
        maximum = torch.full_like(request["path"], torch.iinfo(torch.int64).max)
        sampler.last_observation = RejectionObservation(
            attempts,
            fresh,
            torch.where(fresh, request["path"], maximum).amin(),
        )
        return inherited

    monkeypatch.setattr(
        CollisionConditionedProfileSampler,
        "sample_children",
        exhaust,
    )
    model = build_model_from_config(
        _config("self_play", "branched.fixed")["scientific"], device
    )
    with pytest.raises(BranchedPosteriorExhaustion) as raised:
        BranchedRecoveryHarness(
            owner_factory=lambda: GeneratedOwners(plan),
            plan=plan,
            controller=FixedBranchController({"player": 2.0, "chance": 1.0}, 2),
            run_state=RunState("phase8-exhaustion", root_count=1),
            arguments={
                "geometry": geometry,
                "mode": mode,
                "model": model,
                "model_operation": model.rb_learner_operation,
                "rollout_operation": model.rollout_operation,
                "optimizer": DenseAdam(
                    model,
                    OptimizerConfig(name="adamw", grad_clip=5),
                ),
                "loss": LossConfig(
                    branching=BranchingLossConfig(0.1, 0.1, 1.0e-6),
                ),
                "schedule": LearnerSchedule(
                    batch_size=2, epochs=None, gradient_steps=1
                ),
                "generator": torch.Generator(device=device).manual_seed(31),
                "seed": 31,
                "lambda_": 0.95,
                "completion_samples": 4,
            },
        ).run(iteration=0)
    failure = raised.value.failure
    assert failure.failure_code == "rejection_exhausted"
    assert failure.phase == "posterior_sample"
    assert failure.sampler_attempt_count == POSTERIOR_MAX_ATTEMPTS
    assert failure.logical_address >= 0


@CUDA
def test_safe_tree_uses_only_inherited_profiles_for_rb_player_and_chance(
    monkeypatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "rao_blackwell")
    plan = _plan(device, mode, geometry)

    def unexpected_posterior(
        _sampler: CollisionConditionedProfileSampler,
        **_request,
    ) -> torch.Tensor:
        raise AssertionError("safe-tree N=1 work entered posterior profile sampling")

    monkeypatch.setattr(
        CollisionConditionedProfileSampler,
        "sample_children",
        unexpected_posterior,
    )

    class RecordingController(FixedBranchController):
        def __init__(self) -> None:
            super().__init__(
                {"player": 2.0, "chance": 2.0},
                2,
                safe_tree_size=1,
            )
            self.chance_degrees: list[torch.Tensor] = []

        def decide(self, **request):
            decision = super().decide(**request)
            if not bool(request["owner_is_player"].any()):
                self.chance_degrees.append(decision.executed_degree.detach().cpu())
            return decision

    controller = RecordingController()
    safeguard = CostSafeguardController(1.0, 2.0, 1)
    safeguard.observe(0.1, 1.1, 1)
    model = build_model_from_config(
        _config("self_play", "branched.fixed")["scientific"], device
    )
    result = BranchedRecoveryHarness(
        owner_factory=lambda: GeneratedOwners(plan),
        plan=plan,
        controller=controller,
        run_state=RunState("phase8-safe-tree-profile", root_count=1),
        arguments={
            "geometry": geometry,
            "mode": mode,
            "model": model,
            "model_operation": model.rb_learner_operation,
            "rollout_operation": model.rollout_operation,
            "optimizer": DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5)),
            "loss": LossConfig(
                branching=BranchingLossConfig(0.1, 0.1, 1.0e-6),
            ),
            "schedule": LearnerSchedule(batch_size=2, epochs=None, gradient_steps=1),
            "generator": torch.Generator(device=device).manual_seed(613),
            "seed": 613,
            "lambda_": 0.95,
            "completion_samples": 4,
            "cost_safeguard": safeguard,
        },
    ).run(iteration=0)
    assert result.root_executed_degree == (1,)
    assert controller.chance_degrees
    assert all(degree.eq(1).all() for degree in controller.chance_degrees)


@CUDA
@pytest.mark.parametrize("regime", ("self_play", "best_response"))
@pytest.mark.parametrize("rollout", ("branched.fixed", "branched.KKT"))
def test_four_branched_rb_modes_complete_optimizer_step(
    regime: str, rollout: str
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema(regime, rollout, "rao_blackwell")
    plan = _plan(device, mode, geometry)
    model = build_model_from_config(_config(regime, rollout)["scientific"], device)
    frozen = (
        build_model_from_config(_config("self_play", rollout)["scientific"], device)
        if regime == "best_response"
        else None
    )
    controller = (
        FixedBranchController(
            {
                "player": (2.0 if regime == "self_play" else 1.0),
                "chance": 1.0,
            },
            2,
        )
        if rollout == "branched.fixed"
        else KKTBranchController(
            {"player": 1.0, "chance": 1.0},
            2,
            warmup_iterations=0,
            target_tree_size=64,
            gamma_kappa=0.2,
            eta_kappa=0.2,
            kappa_min=0.25,
            kappa_max=4,
            log_kappa=math.log(1.0),
        )
    )
    result = BranchedRecoveryHarness(
        owner_factory=lambda: GeneratedOwners(plan),
        plan=plan,
        controller=controller,
        run_state=RunState(f"phase8-{regime}-{rollout}", root_count=1),
        arguments={
            "geometry": geometry,
            "mode": mode,
            "model": model,
            "model_operation": model.rb_learner_operation,
            "rollout_operation": model.rollout_operation,
            "frozen_operation": (None if frozen is None else frozen.rollout_operation),
            "optimizer": DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5)),
            "loss": LossConfig(
                branching=BranchingLossConfig(0.1, 0.1, 1.0e-6),
            ),
            "schedule": LearnerSchedule(batch_size=2, epochs=None, gradient_steps=1),
            "generator": torch.Generator(device=device).manual_seed(991),
            "seed": 991,
            "lambda_": 0.95,
            "completion_samples": 4,
            "benchmark_telemetry": (
                regime == "self_play" and rollout == "branched.fixed"
            ),
        },
    ).run(iteration=0)
    assert result.player_nodes
    assert result.leaves
    assert result.posterior_effective_hands is not None
    assert 0 < result.posterior_effective_hands <= result.leaves * 4
    assert result.learner.optimizer_steps
    assert result.learner.optimizer_steps[0].applied
    if regime == "self_play" and rollout == "branched.fixed":
        assert isinstance(result.telemetry, IterationTelemetry)
        assert result.telemetry.rollout_seconds > 0
        assert result.telemetry.routing_traffic_bytes > 0
        assert result.telemetry.host_synchronizations > 0
        assert result.telemetry.child_reservation_commit_seconds > 0
        assert result.telemetry.target_propagation_seconds > 0
        assert result.telemetry.peak_allocated_bytes > 0
