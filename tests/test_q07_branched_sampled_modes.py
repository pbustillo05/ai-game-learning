from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest
import torch
from torch import nn

from pbg.games.poker.queued import (
    BranchedSampledPokerIteration,
    NLHEGeometry,
    sampled_partitions,
)
from pbg.queued import (
    BranchedModeConfig,
    BranchedOverflowRecovery,
    CapacityManifest,
    CostSafeguardController,
    DenseAdam,
    ExternalWorkspaceBounds,
    FixedBranchController,
    GameGeometry,
    GeneratedOwners,
    HardwareLimits,
    KKTBranchController,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    OptimizerConfig,
    PartitionRows,
    RunState,
    build_storage_plan,
    child_path_factors,
    controller_from_state_dict,
    decision_path_mass,
    log_kkt_intensity,
    mode_schema,
    randomized_round,
)
from pbg.queued.storage import DeviceFailure
from pbg.queued.training import RecoverableOverflow
from tests.test_q05_sequential_sampled_training import (
    _EndToEndPokerModel,
    _phase5_layout,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase 7 requires CUDA")


class _BranchedModel(_EndToEndPokerModel):
    def __init__(self, device: torch.device, hands: int) -> None:
        super().__init__(device, hands, check_down=True)
        self.trace_logit = nn.Parameter(torch.tensor(0.2, device=device))
        self.cost_logit = nn.Parameter(torch.tensor(0.1, device=device))

    def evaluate(
        self,
        input_type: str,
        model_input: torch.Tensor,
        private: torch.Tensor,
        owner: torch.Tensor,
    ) -> ModelEvaluation:
        ordinary = super().evaluate(input_type, model_input, private, owner)
        batch = model_input.shape[0]
        return ModelEvaluation(
            ordinary.policy_raw,
            ordinary.value,
            MappingProxyType(
                {
                    "trace": torch.sigmoid(self.trace_logit).expand(batch),
                    "cost": torch.exp(self.cost_logit).expand(batch),
                }
            ),
        )


def _manifest(device: torch.device) -> CapacityManifest:
    return CapacityManifest(
        geometry=GameGeometry(2, 8.0, 4, 64, 4096),
        root_count=1,
        n_replica=1,
        beta_max=2,
        player_node_cap=(
            PartitionRows("pre", 1024),
            PartitionRows("post_2", 1024),
        ),
        trained_player_cap=(
            PartitionRows("pre", 1024),
            PartitionRows("post_2", 1024),
        ),
        action_cap=(PartitionRows("all", 2048),),
        folded_range_cap=(
            PartitionRows("pre", 2),
            PartitionRows("post", 2),
        ),
        leaf_cap=1024,
        topology_cap=1024,
        topology_edge_cap=2048,
        family_pack_cap=32,
        chance_frontier_cap=1024,
        completion_pack_cap=256,
        learner_sigma_pack_cap=8,
        learner_action_pack_cap=16,
        topology_packet_cap=64,
        target_arena_cap=2048,
        movement_scratch_bytes=8 << 20,
        library_workspace_bytes=1 << 20,
        evaluator_workspace_bytes=8 << 20,
        memory_guard_bytes=1 << 20,
        hardware=HardwareLimits(
            torch.cuda.get_device_properties(device).total_memory,
            512 << 20,
            4096,
            16_000_000,
            256,
        ),
        required_external=ExternalWorkspaceBounds(0, 0),
    )


def _scientific_config(regime: str, rollout: str) -> dict[str, object]:
    value: dict[str, object] = {
        "mode": {
            "training_regime": regime,
            "rollout_mode": rollout,
            "private_estimator": "sampled",
        },
        "game": {"players": 2, "stack_bb": 8.0, "rank_count": 4},
        "model": {
            "d_model": 4,
            "n_heads": 1,
            "n_layers_pre": 1,
            "n_layers_post": 1,
            "m_raise": 2,
            "max_players": 2,
            "class_embedding_dim": 2,
            "ffn_hidden": 8,
            "folder_beta_rank": 0,
            "folder_line_dim": 2,
        },
        "loss": {
            "reference_kl_coefficient": 0.0,
            "trace_coefficient": 0.1,
            "cost_coefficient": 0.1,
        },
        "optimizer": {"name": "adamw", "grad_clip": 5.0},
        "schedule": {"batch_size": 2, "gradient_steps": 1},
        "precision": {"name": "fp32"},
        "reference": {
            "interval_iterations": 1,
            "interval_effective_hands": None,
            "cache_outputs": False,
        },
        "lambda": 0.95,
        "root_count": 1,
        "root_count_minimum": 1,
        "seed": 991,
        "branching": {
            "fixed_beta": {"player": 1.5, "chance": 1.25},
            "beta_max": 3.0,
            "minimum_tree_size": 16,
            "minimum_scale": 0.25,
            "backoff_factor": 0.5,
            "safe_tree_size": 512.0,
        },
        "cost_safeguard": {
            "gamma_deviance_max": 0.2,
            "p95_underprediction_max": 1.5,
            "consecutive_iterations": 4,
        },
    }
    if rollout == "branched.KKT":
        value["branching"].pop("minimum_scale")
        value["branching"].update(
            {
                "warmup_iterations": 5,
                "target_tree_size": 256.0,
                "gamma_kappa": 0.2,
                "eta_kappa": 0.2,
                "kappa_min": 0.25,
                "kappa_max": 4.0,
                "log_kappa": 0.0,
            }
        )
    if regime == "best_response":
        value["frozen_profile"] = {"sha256": "a" * 64}
        value["response_initialization"] = "target"
    return value


@pytest.mark.parametrize("regime", ("self_play", "best_response"))
@pytest.mark.parametrize("rollout", ("branched.fixed", "branched.KKT"))
def test_phase7_config_is_closed_for_exactly_four_sampled_modes(
    regime: str, rollout: str
) -> None:
    value = _scientific_config(regime, rollout)
    config = BranchedModeConfig.from_mapping(value)
    assert config.scientific_mapping()["mode"] == value["mode"]
    bad = dict(value)
    bad["completion"] = {"n_samples": 2, "strata": {}}
    with pytest.raises(ValueError, match="root.completion"):
        BranchedModeConfig.from_mapping(bad)


@CUDA
def test_rounding_and_kkt_path_math_match_direct_fp32_references() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    beta = torch.tensor([1.0, 1.25, 1.5, 1.75, 2.0], device=device)
    uniform = torch.tensor([0.2, 0.2, 0.6, 0.8, 0.1], device=device)
    assert randomized_round(beta, uniform).tolist() == [1, 2, 1, 1, 2]
    trace = torch.tensor([0.25, 0.5], device=device)
    cost = torch.tensor([1.0, 2.0], device=device)
    player_sum = torch.tensor([0.0, 0.4], device=device)
    ancestor = torch.tensor([0.0, 0.2], device=device)
    observed = log_kkt_intensity(
        trace,
        cost,
        log_kappa=0.3,
        log_player_ancestor_sum=player_sum,
        log_ancestor_product=ancestor,
        beta_max=4,
    )
    expected = torch.exp(
        0.5 * (0.3 + trace.log() - cost.log() + player_sum - ancestor)
    ).clamp(1, 4)
    torch.testing.assert_close(observed, expected)
    child_sum, child_product = child_path_factors(
        player_sum,
        ancestor,
        observed,
        child_is_player=torch.tensor([True, False], device=device),
    )
    torch.testing.assert_close(child_product, ancestor + observed.log())
    assert torch.isfinite(child_sum[0])


@CUDA
def test_fixed_and_kkt_share_decision_contract_and_round_the_root() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    root = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    path = torch.zeros_like(root)
    common = {
        "root": root,
        "path": path,
        "owner_is_player": torch.ones(4, dtype=torch.bool, device=device),
        "iteration": 0,
        "seed": 991,
        "retry_epoch": 0,
        "trace": torch.full((4,), 0.5, device=device),
        "cost": torch.ones(4, device=device),
        "log_player_ancestor_sum": torch.zeros(4, device=device),
        "log_ancestor_product": torch.zeros(4, device=device),
    }
    fixed = FixedBranchController({"player": 1.5, "chance": 1.25}, 3)
    kkt = KKTBranchController(
        {"player": 1.5, "chance": 1.25},
        3,
        warmup_iterations=2,
        target_tree_size=100,
        gamma_kappa=0.2,
        eta_kappa=0.2,
        kappa_min=0.25,
        kappa_max=4,
        log_kappa=0,
    )
    fixed_result = fixed.decide(
        schema=mode_schema("self_play", "branched.fixed", "sampled"),
        **common,
    )
    kkt_result = kkt.decide(
        schema=mode_schema("self_play", "branched.KKT", "sampled"),
        **common,
    )
    torch.testing.assert_close(fixed_result.nominal_degree, kkt_result.nominal_degree)
    assert fixed_result.nominal_degree.shape == root.shape
    restored = controller_from_state_dict(kkt.state_dict())
    assert restored.state_dict() == kkt.state_dict()
    guarded = FixedBranchController(
        {"player": 2.0, "chance": 2.0},
        2,
        safe_tree_size=1,
    ).decide(
        schema=mode_schema("self_play", "branched.fixed", "sampled"),
        **common,
        frontier_cost=torch.full((4,), 2.0, device=device),
    )
    assert guarded.nominal_degree.tolist() == [2, 2, 2, 2]
    assert guarded.executed_degree.tolist() == [1, 1, 1, 1]
    assert guarded.target_degree.tolist() == [1, 1, 1, 1]
    safeguard = CostSafeguardController(0.2, 1.5, 2)
    safeguard.observe(0.1, 1.2, 100)
    assert not safeguard.active
    safeguard.observe(0.1, 1.2, 100)
    assert safeguard.active
    restored_safeguard = CostSafeguardController.from_state_dict(safeguard.state_dict())
    assert restored_safeguard.state_dict() == safeguard.state_dict()


@CUDA
def test_decisions_are_pack_invariant_and_final_root_degree_sets_mass() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    schema = mode_schema("self_play", "branched.fixed", "sampled")
    controller = FixedBranchController({"player": 1.625, "chance": 1.25}, 3)
    root = torch.arange(7, dtype=torch.int64, device=device)
    path = torch.arange(11, 18, dtype=torch.int64, device=device)

    def decide(selected: torch.Tensor) -> torch.Tensor:
        count = selected.numel()
        return controller.decide(
            schema=schema,
            root=root.index_select(0, selected),
            path=path.index_select(0, selected),
            owner_is_player=torch.ones(count, dtype=torch.bool, device=device),
            iteration=4,
            seed=73,
            retry_epoch=2,
            trace=None,
            cost=None,
            log_player_ancestor_sum=torch.zeros(count, device=device),
            log_ancestor_product=torch.zeros(count, device=device),
        ).executed_degree

    whole = decide(torch.arange(7, device=device))
    split = torch.cat(
        (
            decide(torch.arange(3, device=device)),
            decide(torch.arange(3, 7, device=device)),
        )
    )
    torch.testing.assert_close(whole, split)
    mass = decision_path_mass(
        torch.tensor([-1, 0, 1, 0], dtype=torch.int32),
        torch.tensor([3, 2, 1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        mass, torch.tensor([1, 1 / 3, 1 / 6, 1 / 3], dtype=torch.float64)
    )


def test_whole_attempt_backoff_is_disjoint_and_other_failures_terminate() -> None:
    controller = FixedBranchController(
        {"player": 2.0, "chance": 1.0},
        2,
        minimum_scale=0.25,
    )
    run = RunState("phase7-recovery", root_count=3)
    calls = []
    discarded = []
    failure = DeviceFailure(
        lane=0,
        status=1,
        phase="rollout_commit",
        failure_code="capacity_overflow",
        input_type="pre",
        logical_address=19,
        requested_extent=9,
        available_extent=4,
    )

    def attempt(address):
        calls.append(address)
        if len(calls) == 1:
            raise RecoverableOverflow(failure, capacity=4, observed_demand=9)
        return "committed"

    recovery = BranchedOverflowRecovery(
        iteration=5,
        replica=0,
        controller=controller,
        run_state=run,
    )
    with pytest.warns(RuntimeWarning, match="RECOVERED_BRANCHED_OVERFLOW"):
        assert recovery.run(attempt, discarded.append) == "committed"
    assert len(discarded) == 1
    assert calls[1].retry_epoch == calls[0].retry_epoch + 1
    assert calls[1].logical_base == calls[0].logical_base + 3
    assert run.branch_controller_state == controller.state_dict()

    with pytest.raises(RuntimeError, match="terminal"):
        recovery.run(
            lambda _address: (_ for _ in ()).throw(RuntimeError("terminal")),
            discarded.append,
        )


@CUDA
@pytest.mark.parametrize("regime", ("self_play", "best_response"))
@pytest.mark.parametrize("rollout", ("branched.fixed", "branched.KKT"))
def test_four_sampled_branched_modes_complete_optimizer_step(
    regime: str, rollout: str
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema(regime, rollout, "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(2),
        _manifest(device),
        replace(
            _phase5_layout(geometry),
            player_count=2,
        ),
        device,
    )
    model = _BranchedModel(device, geometry.private_hands)
    frozen = (
        _BranchedModel(device, geometry.private_hands)
        if regime == "best_response"
        else None
    )
    controller = (
        FixedBranchController({"player": 2.0, "chance": 1.0}, 2)
        if rollout == "branched.fixed"
        else KKTBranchController(
            {"player": 2.0, "chance": 1.0},
            2,
            warmup_iterations=10,
            target_tree_size=256,
            gamma_kappa=0.2,
            eta_kappa=0.2,
            kappa_min=0.25,
            kappa_max=4,
            log_kappa=0,
        )
    )
    optimizer = DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5))
    run_state = RunState(f"phase7-{regime}-{rollout}", root_count=1)
    result = BranchedSampledPokerIteration(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        mode=mode,
        controller=controller,
        model=model,
        model_operation=model.evaluate,
        frozen_operation=None if frozen is None else frozen.evaluate,
        optimizer=optimizer,
        loss=LossConfig(
            reference_kl_coefficient=0,
            trace_coefficient=0.1,
            cost_coefficient=0.1,
        ),
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generator=torch.Generator(device=device).manual_seed(991),
        seed=991,
        lambda_=0.95,
        run_state=run_state,
    ).run(iteration=0)
    assert result.player_nodes > 8
    assert result.executed_edges > result.player_nodes
    assert result.leaves > 1
    assert result.duplicate_child_labels > 0
    assert result.learner.statistical_presentations == 2
    assert len(result.learner.optimizer_steps) == 1
    assert result.learner.optimizer_steps[0].applied
    assert optimizer.step_count == 1
    assert run_state.iteration == 1


@CUDA
@pytest.mark.parametrize(
    ("player_beta", "chance_beta", "minimum_tree_size"),
    ((1.0, 2.0, 1), (1.0, 1.0, 20)),
)
def test_chance_branching_and_protected_root_refill_complete_targets(
    player_beta: float,
    chance_beta: float,
    minimum_tree_size: int,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(2),
        _manifest(device),
        replace(_phase5_layout(geometry), player_count=2),
        device,
    )
    model = _BranchedModel(device, geometry.private_hands)
    optimizer = DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5))
    result = BranchedSampledPokerIteration(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        mode=mode,
        controller=FixedBranchController(
            {"player": player_beta, "chance": chance_beta}, 2
        ),
        model=model,
        model_operation=model.evaluate,
        optimizer=optimizer,
        loss=LossConfig(
            reference_kl_coefficient=0,
            trace_coefficient=0.1,
            cost_coefficient=0.1,
        ),
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generator=torch.Generator(device=device).manual_seed(123),
        seed=123,
        lambda_=0.95,
        minimum_tree_size=minimum_tree_size,
    ).run(iteration=0)
    assert result.learner.optimizer_steps[0].applied
    if chance_beta > 1:
        assert result.leaves > 1
        assert result.executed_edges > result.player_nodes
    else:
        assert result.player_nodes + result.leaves >= minimum_tree_size
        assert result.root_executed_degree[0] > result.root_target_degree[0]
        assert result.root_target_degree == (1,)
