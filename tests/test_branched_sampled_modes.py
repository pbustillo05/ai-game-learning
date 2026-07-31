from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from pbg.core.branching import (
    CostSafeguardController,
    FixedBranchController,
    KKTBranchController,
    RankCommonFailureAgreement,
    child_path_factors,
    controller_from_state_dict,
    decision_log_path_mass,
    log_kkt_intensity,
    randomized_round,
)
from pbg.core.capacity import (
    BRANCHED_ALIAS_PEAK_ROW_BYTES,
    CapacityManifest,
    ExecutionGeometry,
    ExecutionPhase,
    ExternalWorkspaceBounds,
    GeneratedOwners,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    build_storage_plan,
)
from pbg.core.contracts import mode_schema
from pbg.core.iteration import IterationEngine, LaneContext
from pbg.core.learner import (
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    ReferenceState,
    _alias_table,
)
from pbg.core.optimization import DenseAdam, OptimizerConfig
from pbg.core.persistence import (
    RunState,
    load_checkpoint,
    save_checkpoint,
)
from pbg.core.storage import DeviceFailure, StorageError
from pbg.core.topology import (
    StagedTopologyPacket,
    TopologyExpansionRecord,
    TopologyPacketStager,
)
from pbg.core.training import RecoverableOverflow
from pbg.games.poker.queued.config import BranchedModeConfig
from pbg.games.poker.queued.folded import folded_statistics_workspace_row_bytes
from pbg.games.poker.queued.hooks import (
    FOLDED,
    NLHEGeometry,
    PokerStateMachine,
    sampled_partitions,
)
from tests.iteration_support import (
    BranchedRecoveryHarness,
    EngineDrivenBranchedAttempt,
)
from tests.iteration_support import (
    discard_attempt as _discard_attempt,
)
from tests.test_sequential_sampled_training import (
    _EndToEndPokerModel,
    _phase5_layout,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase 7 requires CUDA")


class _BranchedModel(_EndToEndPokerModel):
    def __init__(self, device: torch.device, hands: int) -> None:
        super().__init__(device, hands, check_down=True)
        self.trace_logit = nn.Parameter(torch.tensor(0.2, device=device))
        self.cost_logit = nn.Parameter(torch.tensor(0.1, device=device))
        self.no_grad_batch_sizes: list[int] = []

    def evaluate(
        self,
        input_type: str,
        model_input: torch.Tensor,
        private: torch.Tensor,
        owner: torch.Tensor,
    ) -> ModelEvaluation:
        if not torch.is_grad_enabled():
            self.no_grad_batch_sizes.append(int(model_input.shape[0]))
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

def _manifest(
    device: torch.device,
    geometry: NLHEGeometry | None = None,
) -> CapacityManifest:
    geometry = NLHEGeometry(2, 8.0, 4) if geometry is None else geometry
    inputs = tuple(
        item.name for item in sampled_partitions(geometry).inputs
    )
    return CapacityManifest(
        geometry=ExecutionGeometry(
            geometry.players,
            geometry.player_node_bound,
            64,
            4096,
        ),
        root_count=1,
        n_replica=1,
        beta_max=2,
        player_node_cap=tuple(
            PartitionRows(name, 1024) for name in inputs
        ),
        trained_player_cap=tuple(
            PartitionRows(name, 1024) for name in inputs
        ),
        action_cap=(
            PartitionRows("all", 2048 * max(1, geometry.players - 1)),
        ),
        folded_range_cap=(
            PartitionRows("pre", 1024),
            PartitionRows("post", 1024),
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


_TOPOLOGY_PACKET_ROW_BYTES = 95


def _sampled_chance_frontier_row_bytes(geometry: NLHEGeometry) -> int:
    players = geometry.players
    hands = geometry.private_hands
    history = (
        geometry.postflop_base_model_input_width
        - 8
        - 10 * players
        - players * hands
    ) // 10
    return (
        512
        + 8 * geometry.postflop_model_input_width
        + 16 * players * hands
        + 128 * history
        + 64 * geometry.deck_cards
        + 128 * players
        + 3 * _TOPOLOGY_PACKET_ROW_BYTES
        + folded_statistics_workspace_row_bytes(geometry)
    )


def _phase7_layout(geometry: NLHEGeometry):
    return replace(
        _phase5_layout(geometry),
        chance_frontier_row_bytes=_sampled_chance_frontier_row_bytes(geometry),
        topology_packet_row_bytes=_TOPOLOGY_PACKET_ROW_BYTES,
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
            "epsilon": 1.0e-8,
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
            "beta_max": 3,
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
    if rollout == "branched.KKT":
        bad_kkt = {
            **value,
            "branching": {
                **value["branching"],
                "target_tree_size": 0,
            },
        }
        with pytest.raises(ValueError, match="KKT controller bounds"):
            BranchedModeConfig.from_mapping(bad_kkt)


def test_branch_controller_requires_integral_capacity_bound() -> None:
    invalid = _scientific_config("self_play", "branched.fixed")
    invalid["branching"] = {
        **invalid["branching"],
        "beta_max": 2.5,
    }
    with pytest.raises(ValueError, match="controller bounds"):
        BranchedModeConfig.from_mapping(invalid)
    with pytest.raises(ValueError, match="positive integer"):
        FixedBranchController(
            {"player": 1.0, "chance": 1.0},
            2.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive integer"):
        KKTBranchController(
            {"player": 1.0, "chance": 1.0},
            2.5,  # type: ignore[arg-type]
            warmup_iterations=0,
            target_tree_size=8,
            gamma_kappa=0.5,
            eta_kappa=0.5,
            kappa_min=0.25,
            kappa_max=4,
            log_kappa=0,
        )


def test_fixed_config_accepts_stage_specific_player_beta_only() -> None:
    value = _scientific_config("self_play", "branched.fixed")
    value["branching"] = {
        **value["branching"],
        "fixed_beta": {
            "player_pre": 14.0,
            "player_post": 2.0,
            "chance": 2.4,
        },
        "beta_max": {
            "player_pre": 15,
            "player_post": 3,
            "chance": 3,
        },
    }
    configured = BranchedModeConfig.from_mapping(value)
    assert dict(configured.branching["fixed_beta"]) == {
        "player_pre": 14.0,
        "player_post": 2.0,
        "chance": 2.4,
    }
    assert dict(configured.branching["beta_max"]) == {
        "player_pre": 15,
        "player_post": 3,
        "chance": 3,
    }
    value = _scientific_config("self_play", "branched.KKT")
    value["branching"] = {
        **value["branching"],
        "fixed_beta": {
            "player_pre": 3.0,
            "player_post": 2.0,
            "chance": 2.4,
        },
    }
    with pytest.raises(ValueError, match="KKT requires player/chance"):
        BranchedModeConfig.from_mapping(value)


@CUDA
def test_phase7_admission_covers_fixed_and_full_decision_workspaces() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    layout = _phase7_layout(geometry)
    manifest = _manifest(device)
    self_plan = build_storage_plan(
        mode_schema("self_play", "branched.fixed", "sampled"),
        sampled_partitions(geometry),
        manifest,
        layout,
        device,
    )
    response_plan = build_storage_plan(
        mode_schema("best_response", "branched.fixed", "sampled"),
        sampled_partitions(geometry),
        manifest,
        layout,
        device,
    )
    trained_rows = sum(
        manifest.partition_rows("trained_player_cap").values()
    )
    expected_learner = (
        manifest.learner_sigma_pack_cap
        * layout.learner_sigma_workspace_row_bytes
        + manifest.learner_action_pack_cap
        * layout.learner_action_workspace_row_bytes
        + manifest.library_workspace_bytes
        + trained_rows * BRANCHED_ALIAS_PEAK_ROW_BYTES
    )
    assert (
        self_plan.admission(
            ExecutionPhase.LEARNER
        ).bounded_workspace_bytes
        == expected_learner
    )
    self_rollout = self_plan.admission(
        ExecutionPhase.ROLLOUT
    ).bounded_workspace_bytes
    response_rollout = response_plan.admission(
        ExecutionPhase.ROLLOUT
    ).bounded_workspace_bytes
    assert (
        response_rollout - self_rollout
        == manifest.family_pack_cap * layout.family_workspace_row_bytes
    )
    assert (
        layout.chance_frontier_row_bytes
        == _sampled_chance_frontier_row_bytes(geometry)
    )
    assert (
        layout.topology_packet_row_bytes
        == _TOPOLOGY_PACKET_ROW_BYTES
    )
    with pytest.raises(ValueError, match="root count"):
        BranchedRecoveryHarness(
            owner_factory=lambda: GeneratedOwners(self_plan),
            plan=self_plan,
            controller=FixedBranchController(
                {"player": 1.0, "chance": 1.0}, 2
            ),
            run_state=RunState("phase7-root-bound", root_count=2),
            arguments={},
        )
    two_root_plan = build_storage_plan(
        self_plan.mode,
        sampled_partitions(geometry),
        replace(manifest, root_count=2),
        layout,
        device,
    )
    with pytest.raises(ValueError, match="root count"):
        BranchedRecoveryHarness(
            owner_factory=lambda: GeneratedOwners(two_root_plan),
            plan=two_root_plan,
            controller=FixedBranchController(
                {"player": 1.0, "chance": 1.0}, 2
            ),
            run_state=RunState("phase7-root-exact", root_count=1),
            arguments={},
        )


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
def test_cuda_alias_table_preserves_log_mass_without_cpu_underflow() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    log_weight = torch.tensor(
        [-1000.0, -3.0, -1.0, 0.0],
        dtype=torch.float64,
        device=device,
    )
    threshold, alias = _alias_table(log_weight)
    rows = log_weight.numel()
    observed = threshold / rows
    observed.scatter_add_(
        0,
        alias.to(torch.int64),
        (1 - threshold) / rows,
    )
    torch.testing.assert_close(
        observed,
        torch.softmax(log_weight, 0),
        atol=1.0e-15,
        rtol=1.0e-12,
    )


@CUDA
def test_cuda_alias_table_scales_to_large_decision_owner() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 65_536
    log_weight = torch.linspace(
        -40,
        0,
        rows,
        dtype=torch.float64,
        device=device,
    )
    threshold, alias = _alias_table(log_weight)
    torch.cuda.synchronize(device)
    assert threshold.shape == (rows,)
    assert alias.shape == (rows,)
    assert bool(torch.isfinite(threshold).all())
    assert bool(((threshold >= 0) & (threshold <= 1)).all())
    assert bool(((alias >= 0) & (alias < rows)).all())


@CUDA
def test_inactive_fixed_chance_rows_do_not_draw_or_transition() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    machine = PokerStateMachine(NLHEGeometry(2, 8.0, 4))
    initial = machine.initial_roots(
        2,
        seed=37,
        iteration=0,
        retry_epoch=0,
        device=device,
    )
    state = replace(initial, pending=torch.zeros_like(initial.pending))
    active = torch.tensor([True, False], dtype=torch.bool, device=device)
    dealt = machine.deal_next_street_keyed(
        state,
        seed=41,
        iteration=3,
        root=torch.tensor([5, 6], dtype=torch.int64, device=device),
        path=torch.tensor([7, 8], dtype=torch.int64, device=device),
        draw=0,
        retry_epoch=0,
        active=active,
    )
    assert int(dealt.street[0]) == 1
    for name in state.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(dealt, name)[1],
            getattr(state, name)[1],
        )


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
    guarded_chance = FixedBranchController(
        {"player": 2.0, "chance": 2.0},
        2,
        safe_tree_size=1,
    ).decide(
        schema=mode_schema("self_play", "branched.fixed", "sampled"),
        **{
            **common,
            "owner_is_player": torch.zeros_like(
                common["owner_is_player"]
            ),
            "trace": None,
            "cost": None,
        },
        frontier_cost=torch.full((4,), 2.0, device=device),
    )
    assert guarded_chance.nominal_degree.tolist() == [2, 2, 2, 2]
    assert guarded_chance.executed_degree.tolist() == [1, 1, 1, 1]
    split = FixedBranchController(
        {"player_pre": 6.0, "player_post": 3.0, "chance": 2.0},
        {"player_pre": 6, "player_post": 3, "chance": 3},
    )
    pre = split.decide(
        schema=mode_schema("self_play", "branched.fixed", "sampled"),
        **common,
        player_family="pre",
    )
    post = split.decide(
        schema=mode_schema("self_play", "branched.fixed", "sampled"),
        **common,
        player_family="post",
    )
    assert pre.executed_degree.tolist() == [6, 6, 6, 6]
    assert post.executed_degree.tolist() == [3, 3, 3, 3]
    assert split.max_degree(owner_is_player=True, player_family="pre") == 6
    assert split.max_degree(owner_is_player=True, player_family="post") == 3
    assert split.max_degree(owner_is_player=False) == 3
    guarded_kkt_chance = KKTBranchController(
        {"player": 2.0, "chance": 2.0},
        2,
        warmup_iterations=2,
        target_tree_size=100,
        gamma_kappa=0.2,
        eta_kappa=0.2,
        kappa_min=0.25,
        kappa_max=4,
        log_kappa=0,
        safe_tree_size=1,
    ).decide(
        schema=mode_schema("self_play", "branched.KKT", "sampled"),
        **{
            **common,
            "owner_is_player": torch.zeros_like(
                common["owner_is_player"]
            ),
            "trace": None,
            "cost": None,
        },
        frontier_cost=torch.full((4,), 2.0, device=device),
    )
    assert guarded_kkt_chance.nominal_degree.tolist() == [2, 2, 2, 2]
    assert guarded_kkt_chance.executed_degree.tolist() == [1, 1, 1, 1]
    safeguard = CostSafeguardController(0.2, 1.5, 2)
    safeguard.observe(0.1, 1.2, 100)
    assert not safeguard.active
    safeguard.observe(0.1, 1.2, 100)
    assert safeguard.active
    restored_safeguard = CostSafeguardController.from_state_dict(safeguard.state_dict())
    assert restored_safeguard.state_dict() == safeguard.state_dict()
    emergency = KKTBranchController(
        {"player": 2.0, "chance": 1.0},
        3,
        warmup_iterations=5,
        target_tree_size=100,
        gamma_kappa=0.5,
        eta_kappa=0.5,
        kappa_min=0.25,
        kappa_max=4,
        log_kappa=0,
    )
    before = emergency.decide(
        schema=mode_schema("self_play", "branched.KKT", "sampled"),
        **common,
    )
    emergency.backoff()
    after = emergency.decide(
        schema=mode_schema("self_play", "branched.KKT", "sampled"),
        **common,
    )
    assert int(after.executed_degree.sum()) < int(before.executed_degree.sum())
    old_log_kappa = emergency.log_kappa
    observed = emergency.observe(200, iteration=5)
    assert observed["ema_tree_size"] == pytest.approx(150)
    assert emergency.log_kappa <= old_log_kappa


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
    log_mass = decision_log_path_mass(
        torch.tensor([-1, 0, 1, 0], dtype=torch.int32),
        torch.tensor([3, 2, 1, 1], dtype=torch.int32),
        device=device,
    )
    torch.testing.assert_close(
        log_mass.exp(),
        torch.tensor([1, 1 / 3, 1 / 6, 1 / 3], device=device),
    )
    depth = 700
    deep_log_mass = decision_log_path_mass(
        torch.arange(-1, depth - 1, dtype=torch.int32),
        torch.full((depth,), 4, dtype=torch.int32),
        device=device,
    )
    assert bool(torch.isfinite(deep_log_mass).all())
    torch.testing.assert_close(
        deep_log_mass[-1],
        torch.tensor(
            -(depth - 1) * math.log(4),
            dtype=torch.float32,
            device=device,
        ),
        atol=2.0e-3,
        rtol=3.0e-6,
    )


def test_common_engine_branched_backoff_is_disjoint_and_durable() -> None:
    controller = FixedBranchController(
        {"player": 2.0, "chance": 1.0},
        2,
        minimum_scale=0.25,
    )
    run = RunState("phase7-recovery", root_count=3)
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

    local = RecoverableOverflow(failure, capacity=4, observed_demand=9)
    remote = RecoverableOverflow(
        replace(failure, logical_address=7),
        capacity=4,
        observed_demand=9,
    )
    agreement = RankCommonFailureAgreement(
        world_size=2,
        collect=lambda _local: (local, remote),
    )
    assert agreement.agree(local) is remote
    assert agreement.agree(remote) is remote
    engine = object.__new__(IterationEngine)
    engine.rank_agreement = RankCommonFailureAgreement(
        world_size=2,
        collect=lambda _local: (local, remote),
    )
    lane = LaneContext(
        mode=mode_schema("self_play", "branched.fixed", "sampled"),
        run_state=run,
        generator=torch.Generator().manual_seed(13),
        optimizer=object(),
        model_operation=object(),
        loss=LossConfig(),
        branch_controller=controller,
    )
    engine.config = SimpleNamespace(lanes=(lane,))

    agreed = engine._agree_failure(local)
    assert agreed is remote
    with pytest.warns(RuntimeWarning, match="RECOVERED_BRANCHED_OVERFLOW"):
        engine._recover(iteration=5, failure=agreed)
    assert run.retry_epoch == 1
    assert run.attempt_logical_base == 3
    assert run.recovered_overflow
    assert run.branch_controller_state == controller.state_dict()
    record = run.overflow_recoveries[0]
    assert record["failed_address"] == "0:7"
    assert record["replacement_address"] == "1:3"
    assert record["old_control"] != record["new_control"]


@CUDA
def test_branched_checkpoint_round_trip_preserves_controllers(
    tmp_path: Path,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    model = _BranchedModel(device, geometry.private_hands)
    reference = _BranchedModel(device, geometry.private_hands)
    optimizer = DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5))
    controller = KKTBranchController(
        {"player": 1.5, "chance": 1.25},
        3,
        warmup_iterations=0,
        target_tree_size=256,
        gamma_kappa=0.2,
        eta_kappa=0.2,
        kappa_min=0.25,
        kappa_max=4,
        log_kappa=0,
    )
    controller.backoff()
    safeguard = CostSafeguardController(0.2, 1.5, 1)
    safeguard.observe(0.1, 1.2, 8)
    run = RunState(
        "phase7-checkpoint",
        iteration=2,
        metrics_prefix=[{"iteration": value} for value in range(2)],
        branch_controller_state=controller.state_dict(),
        cost_safeguard_state=safeguard.state_dict(),
    )
    generator = torch.Generator(device=device).manual_seed(881)
    scientific = {"mode": "self_play/branched.KKT/sampled", "seed": 881}
    path = tmp_path / "phase7.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=optimizer.scaler,
        reference_model=reference,
        reference_state=ReferenceState(
            interval_iterations=1,
            interval_effective_hands=None,
        ),
        run_state=run,
        scientific_config=scientific,
        execution_config={"checkpoint_every": 1},
        generator=generator,
    )
    _, _, resumed, _ = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        reference_model=reference,
        expected_scientific_config=scientific,
        generator=generator,
    )
    assert (
        controller_from_state_dict(resumed.branch_controller_state).state_dict()
        == controller.state_dict()
    )
    assert (
        CostSafeguardController.from_state_dict(
            resumed.cost_safeguard_state
        ).state_dict()
        == safeguard.state_dict()
    )
    prior_state = torch.load(path, map_location="cpu", weights_only=False)
    prior_state["run_state"].pop("branch_controller_state")
    prior_state["run_state"].pop("cost_safeguard_state")
    prior_state["checkpoint_schema"] = "pbg-queued-checkpoint-v1"
    prior_path = tmp_path / "phase6-compatible.pt"
    torch.save(prior_state, prior_path)
    with pytest.raises(Exception, match="unsupported checkpoint schema"):
        load_checkpoint(
            prior_path,
            model=model,
            optimizer=optimizer,
            reference_model=reference,
            expected_scientific_config=scientific,
            generator=generator,
        )


@CUDA
def test_failed_branched_attempt_releases_every_generated_owner() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        _manifest(device),
        replace(_phase7_layout(geometry), player_count=2),
        device,
    )
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    owners.failure_state.record(
        DeviceFailure(
            lane=0,
            status=1,
            phase="rollout_commit",
            failure_code="capacity_overflow",
            input_type="pre",
            logical_address=3,
            requested_extent=12,
            available_extent=8,
        )
    )
    _discard_attempt(owners)
    assert frozenset(owners._released) == {
        allocation.key for allocation in plan.allocations
    }
    with pytest.raises(StorageError, match="released"):
        owners.failure_state.snapshot(0)


@CUDA
def test_healthy_branched_attempt_discard_releases_without_failure_record() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        _manifest(device),
        replace(_phase7_layout(geometry), player_count=2),
        device,
    )
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    _discard_attempt(owners)
    assert frozenset(owners._released) == {
        allocation.key for allocation in plan.allocations
    }


@CUDA
def test_untrained_unary_root_refill_restores_one_structural_root() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    recorder = TopologyPacketStager(
        input_names=("pre",),
        input_to_output=(0,),
        p_train=2,
        topology_cap=8,
        topology_edge_cap=8,
        topology_packet_cap=4,
        device=device,
    )

    def payload(*, refill: bool, child_path: int, leaf_row: int):
        record = TopologyExpansionRecord(
            node_active=torch.tensor([True]),
            edge_active=torch.tensor([True]),
            parent_key=torch.tensor([[0, 0]], dtype=torch.int64),
            child_key=torch.tensor([[0, child_path]], dtype=torch.int64),
            source_ordinal=torch.tensor([0], dtype=torch.int64),
            terminal=torch.tensor([True]),
            child_input=torch.tensor([0], dtype=torch.int32),
            child_leaf_row=torch.tensor([leaf_row], dtype=torch.int64),
            label=torch.tensor([child_path], dtype=torch.int32),
            owner=torch.tensor([0], dtype=torch.int32),
            input_type=torch.tensor([0], dtype=torch.int32),
            familywork_row=torch.tensor([0], dtype=torch.int32),
            sigma_head_row=torch.tensor([-1], dtype=torch.int32),
            action_begin=torch.tensor([-1], dtype=torch.int32),
            executed_degree=torch.tensor([1], dtype=torch.int32),
            nominal_degree=torch.tensor([1], dtype=torch.int32),
            target_degree=torch.tensor([1], dtype=torch.int32),
            trained=torch.tensor([False]),
            refill=torch.tensor([refill]),
            path_length=torch.tensor([1], dtype=torch.int32),
        )
        return StagedTopologyPacket(
            record,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(0, dtype=torch.float32),
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(False),
            torch.tensor(1, dtype=torch.int64),
        )

    recorder.accept(payload(refill=False, child_path=11, leaf_row=0))
    recorder.accept(payload(refill=True, child_path=12, leaf_row=1))
    topology, _, leaf_count = recorder.seal()
    assert topology.node_count == 1
    assert topology.executed_degree.tolist() == [2]
    assert topology.target_degree.tolist() == [1]
    assert topology.parent.tolist() == [-1]
    assert leaf_count == 2
    recorder.release()


@CUDA
def test_full_branched_iteration_is_family_pack_invariant() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")

    def execute(family_pack_cap: int) -> tuple[object, dict[str, torch.Tensor]]:
        torch.cuda.manual_seed(191)
        plan = build_storage_plan(
            mode,
            sampled_partitions(geometry),
            replace(_manifest(device), family_pack_cap=family_pack_cap),
            replace(_phase7_layout(geometry), player_count=2),
            device,
        )
        model = _BranchedModel(device, geometry.private_hands)
        result = EngineDrivenBranchedAttempt(
            geometry=geometry,
            owners=GeneratedOwners(plan),
            mode=mode,
            controller=FixedBranchController(
                {"player": 2.0, "chance": 1.0},
                2,
            ),
            model=model,
            model_operation=model.evaluate,
            optimizer=DenseAdam(
                model,
                OptimizerConfig(name="adamw", grad_clip=5),
            ),
            loss=LossConfig(
                reference_kl_coefficient=0,
                trace_coefficient=0.1,
                cost_coefficient=0.1,
            ),
            schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
            generator=torch.Generator(device=device).manual_seed(991),
            seed=991,
            lambda_=0.95,
        ).run(iteration=0)
        return result, {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    narrow, narrow_state = execute(1)
    wide, wide_state = execute(32)
    for field in (
        "player_nodes",
        "leaves",
        "target_layers",
        "executed_edges",
        "root_executed_degree",
        "root_target_degree",
        "duplicate_child_labels",
        "sampled_tree_nodes",
    ):
        assert getattr(narrow, field) == getattr(wide, field)
    assert narrow_state.keys() == wide_state.keys()
    for name in narrow_state:
        torch.testing.assert_close(
            narrow_state[name],
            wide_state[name],
            atol=0,
            rtol=0,
        )


@CUDA
def test_multi_pack_roots_use_complete_frontier_cost_forecast() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        replace(
            _manifest(device),
            root_count=3,
            family_pack_cap=1,
        ),
        replace(_phase7_layout(geometry), player_count=2),
        device,
    )
    model = _BranchedModel(device, geometry.private_hands)
    safeguard = CostSafeguardController(1, 2, 1)
    safeguard.observe(0.1, 1.1, 3)
    result = EngineDrivenBranchedAttempt(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        mode=mode,
        controller=FixedBranchController(
            {"player": 2.0, "chance": 1.0},
            2,
            safe_tree_size=6,
        ),
        model=model,
        model_operation=model.evaluate,
        optimizer=DenseAdam(
            model,
            OptimizerConfig(name="adamw", grad_clip=5),
        ),
        loss=LossConfig(
            reference_kl_coefficient=0,
            trace_coefficient=0.1,
            cost_coefficient=0.1,
        ),
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generator=torch.Generator(device=device).manual_seed(991),
        seed=991,
        lambda_=0.95,
        cost_safeguard=safeguard,
    ).run(iteration=0)
    assert result.root_executed_degree == (1, 1, 1)
    assert model.no_grad_batch_sizes
    assert max(model.no_grad_batch_sizes) <= 1


@CUDA
def test_kkt_feedback_is_invariant_to_root_batching() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.KKT", "sampled")

    def execute(root_count: int) -> KKTBranchController:
        torch.cuda.manual_seed(313)
        plan = build_storage_plan(
            mode,
            sampled_partitions(geometry),
                replace(
                    _manifest(device),
                    root_count=root_count,
                    beta_max=1,
                    family_pack_cap=1,
                ),
            replace(_phase7_layout(geometry), player_count=2),
            device,
        )
        model = _BranchedModel(device, geometry.private_hands)
        controller = KKTBranchController(
            {"player": 1.0, "chance": 1.0},
            1,
            warmup_iterations=0,
            target_tree_size=64,
            gamma_kappa=0.5,
            eta_kappa=0.2,
            kappa_min=1,
            kappa_max=1,
            log_kappa=0,
        )
        EngineDrivenBranchedAttempt(
            geometry=geometry,
            owners=GeneratedOwners(plan),
            mode=mode,
            controller=controller,
            model=model,
            model_operation=model.evaluate,
            optimizer=DenseAdam(
                model,
                OptimizerConfig(name="adamw", grad_clip=5),
            ),
            loss=LossConfig(
                reference_kl_coefficient=0,
                trace_coefficient=0.1,
                cost_coefficient=0.1,
            ),
            schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
            generator=torch.Generator(device=device).manual_seed(991),
            seed=991,
            lambda_=0.95,
        ).run(iteration=0)
        return controller

    single = execute(1)
    batched = execute(3)
    assert single.ema_tree_size == pytest.approx(batched.ema_tree_size)
    assert single.log_kappa == pytest.approx(batched.log_kappa)
    assert single.updates == batched.updates == 1


@CUDA
def test_ragged_statistical_work_subdivides_by_action_cap() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")

    def execute(action_cap: int) -> tuple[object, dict[str, torch.Tensor]]:
        torch.cuda.manual_seed(431)
        plan = build_storage_plan(
            mode,
            sampled_partitions(geometry),
            replace(
                _manifest(device),
                learner_sigma_pack_cap=8,
                learner_action_pack_cap=action_cap,
            ),
            replace(_phase7_layout(geometry), player_count=2),
            device,
        )
        model = _BranchedModel(device, geometry.private_hands)
        result = EngineDrivenBranchedAttempt(
            geometry=geometry,
            owners=GeneratedOwners(plan),
            mode=mode,
            controller=FixedBranchController(
                {"player": 2.0, "chance": 1.0},
                2,
            ),
            model=model,
            model_operation=model.evaluate,
            optimizer=DenseAdam(
                model,
                OptimizerConfig(name="adamw", grad_clip=5),
            ),
            loss=LossConfig(
                reference_kl_coefficient=0,
                trace_coefficient=0.1,
                cost_coefficient=0.1,
            ),
            schedule=LearnerSchedule(batch_size=8, gradient_steps=1),
            generator=torch.Generator(device=device).manual_seed(991),
            seed=991,
            lambda_=0.95,
        ).run(iteration=0)
        return result, {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    narrow, narrow_state = execute(2)
    wide, wide_state = execute(16)
    assert narrow.learner.statistical_presentations == 8
    assert wide.learner.statistical_presentations == 8
    assert narrow.learner.optimizer_steps[0].applied
    assert wide.learner.optimizer_steps[0].applied
    for name in narrow_state:
        torch.testing.assert_close(
            narrow_state[name],
            wide_state[name],
            atol=1.0e-6,
            rtol=1.0e-6,
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
        sampled_partitions(geometry),
        _manifest(device),
        replace(
            _phase7_layout(geometry),
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
            warmup_iterations=0,
            target_tree_size=256,
            gamma_kappa=0.2,
            eta_kappa=0.2,
            kappa_min=0.25,
            kappa_max=4,
            log_kappa=math.log(4),
        )
    )
    optimizer = DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5))
    run_state = RunState(f"phase7-{regime}-{rollout}", root_count=1)
    result = BranchedRecoveryHarness(
        owner_factory=lambda: GeneratedOwners(plan),
        plan=plan,
        controller=controller,
        run_state=run_state,
        arguments={
            "geometry": geometry,
            "mode": mode,
            "model": model,
            "model_operation": model.evaluate,
            "frozen_operation": None if frozen is None else frozen.evaluate,
            "optimizer": optimizer,
            "loss": LossConfig(
                reference_kl_coefficient=0,
                trace_coefficient=0.1,
                cost_coefficient=0.1,
            ),
            "schedule": LearnerSchedule(batch_size=2, gradient_steps=1),
            "generator": torch.Generator(device=device).manual_seed(991),
            "seed": 991,
            "lambda_": 0.95,
        },
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
    if rollout == "branched.KKT":
        assert controller.updates == 1
        assert run_state.branch_controller_state == controller.state_dict()


@CUDA
def test_best_response_rollout_selects_one_model_per_role_subspan() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("best_response", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        _manifest(device),
        _phase7_layout(geometry),
        device,
    )

    class RoleRecordingModel(_BranchedModel):
        def __init__(self):
            super().__init__(device, geometry.private_hands)
            self.rollout_owners: list[torch.Tensor] = []

        def evaluate(self, input_type, model_input, private, owner):
            if not torch.is_grad_enabled():
                self.rollout_owners.append(owner.detach().clone())
            return super().evaluate(
                input_type, model_input, private, owner
            )

    learned = RoleRecordingModel()
    frozen = RoleRecordingModel()
    EngineDrivenBranchedAttempt(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        mode=mode,
        controller=FixedBranchController(
            {"player": 1.0, "chance": 1.0}, 2
        ),
        model=learned,
        model_operation=learned.evaluate,
        frozen_operation=frozen.evaluate,
        optimizer=DenseAdam(
            learned,
            OptimizerConfig(name="adamw", grad_clip=5),
        ),
        loss=LossConfig(
            reference_kl_coefficient=0,
            trace_coefficient=0.1,
            cost_coefficient=0.1,
        ),
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generator=torch.Generator(device=device).manual_seed(701),
        seed=701,
        lambda_=0.95,
    ).run(iteration=0)
    assert learned.rollout_owners
    assert frozen.rollout_owners
    assert all(
        bool(owner.eq(0).all()) for owner in learned.rollout_owners
    )
    assert all(
        bool(owner.eq(1).all()) for owner in frozen.rollout_owners
    )


@CUDA
def test_multiplayer_best_response_root_uses_frozen_opponent_model() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(4, 8.0, 4)
    mode = mode_schema("best_response", "branched.fixed", "sampled")
    partitions = sampled_partitions(geometry)
    plan = build_storage_plan(
        mode,
        partitions,
        _manifest(device, geometry),
        replace(
            _phase7_layout(geometry),
            model_input_shapes=tuple(
                NamedShape(
                    partition.name,
                    (geometry.model_input_width(partition.name),),
                )
                for partition in partitions.inputs
            ),
            player_count=geometry.players,
            stratum_count=geometry.players,
        ),
        device,
    )

    class RoleRecordingModel(_BranchedModel):
        def __init__(self):
            super().__init__(device, geometry.private_hands)
            self.rollout_owners: list[torch.Tensor] = []

        def evaluate(self, input_type, model_input, private, owner):
            if not torch.is_grad_enabled():
                self.rollout_owners.append(owner.detach().clone())
            return super().evaluate(input_type, model_input, private, owner)

    learned = RoleRecordingModel()
    frozen = RoleRecordingModel()
    EngineDrivenBranchedAttempt(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        mode=mode,
        controller=FixedBranchController(
            {"player": 1.0, "chance": 1.0}, 2
        ),
        model=learned,
        model_operation=learned.evaluate,
        frozen_operation=frozen.evaluate,
        optimizer=DenseAdam(
            learned,
            OptimizerConfig(name="adamw", grad_clip=5),
        ),
        loss=LossConfig(
            reference_kl_coefficient=0,
            trace_coefficient=0.1,
            cost_coefficient=0.1,
        ),
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generator=torch.Generator(device=device).manual_seed(703),
        seed=703,
        lambda_=0.95,
    ).run(iteration=0)
    assert frozen.rollout_owners
    assert int(frozen.rollout_owners[0][0]) == 3
    assert not learned.rollout_owners or int(learned.rollout_owners[0][0]) != 3


@CUDA
def test_branched_allocator_and_reference_cache_controls_are_wired() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        replace(_manifest(device), learner_sigma_pack_cap=1),
        replace(
            _phase7_layout(geometry),
            reference_cache_bytes=4 << 20,
        ),
        device,
    )
    model = _BranchedModel(device, geometry.private_hands)
    reference = _BranchedModel(device, geometry.private_hands)
    result = EngineDrivenBranchedAttempt(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        mode=mode,
        controller=FixedBranchController(
            {"player": 2.0, "chance": 1.0}, 2
        ),
        model=model,
        model_operation=model.evaluate,
        reference_operation=reference.evaluate,
        cache_reference_outputs=True,
        allocation_inputs=(
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            8,
        ),
        optimizer=DenseAdam(
            model,
            OptimizerConfig(name="adamw", grad_clip=5),
        ),
        loss=LossConfig(
            reference_kl_coefficient=0.1,
            trace_coefficient=0.1,
            cost_coefficient=0.1,
        ),
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generator=torch.Generator(device=device).manual_seed(709),
        seed=709,
        lambda_=0.95,
    ).run(iteration=0)
    assert result.learner.statistical_presentations == 8
    assert len(reference.no_grad_batch_sizes) > 2
    assert max(reference.no_grad_batch_sizes) == 1


@CUDA
def test_three_player_nonterminal_fold_seals_branched_summary() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(3, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    partitions = sampled_partitions(geometry)
    layout = replace(
        _phase7_layout(geometry),
        model_input_shapes=tuple(
            NamedShape(
                partition.name,
                (geometry.model_input_width(partition.name),),
            )
            for partition in partitions.inputs
        ),
        player_count=geometry.players,
        stratum_count=geometry.players,
    )
    plan = build_storage_plan(
        mode,
        partitions,
        _manifest(device, geometry),
        layout,
        device,
    )

    class FoldCheckingModel(_BranchedModel):
        def __init__(self):
            super().__init__(device, geometry.private_hands)
            self.check_down = False
            self.forward_count = 0
            self.sealed_fold_seen = False

        def evaluate(self, input_type, model_input, private, owner):
            if not torch.is_grad_enabled():
                self.forward_count += 1
            if not torch.is_grad_enabled() and self.forward_count > 1:
                if input_type == "pre":
                    assert (
                        PokerStateMachine(geometry)
                        .model_context("pre", model_input)[0]
                        .shape[-1]
                        == geometry.private_classes
                    )
                    state = PokerStateMachine(
                        geometry
                    ).state_from_model_input(
                        "pre", model_input, private
                    )
                    torch._assert_async(
                        state.status.eq(FOLDED).any(-1).all(),
                        "nonterminal preflop fold was not retained",
                    )
                else:
                    torch._assert_async(
                        model_input[:, -1].gt(0.5).all(),
                        "nonterminal postflop fold omitted its sealed summary",
                    )
                self.sealed_fold_seen = True
            return super().evaluate(
                input_type, model_input, private, owner
            )

    model = FoldCheckingModel()
    result = EngineDrivenBranchedAttempt(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        mode=mode,
        controller=FixedBranchController(
            {"player": 1.0, "chance": 1.0}, 2
        ),
        model=model,
        model_operation=model.evaluate,
        optimizer=DenseAdam(
            model,
            OptimizerConfig(name="adamw", grad_clip=5),
        ),
        loss=LossConfig(
            reference_kl_coefficient=0,
            trace_coefficient=0.1,
            cost_coefficient=0.1,
        ),
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generator=torch.Generator(device=device).manual_seed(607),
        seed=607,
        lambda_=0.95,
    ).run(iteration=0)
    assert model.sealed_fold_seen
    assert result.learner.optimizer_steps[0].applied


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
        sampled_partitions(geometry),
        _manifest(device),
        replace(_phase7_layout(geometry), player_count=2),
        device,
    )
    model = _BranchedModel(device, geometry.private_hands)
    optimizer = DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5))
    result = EngineDrivenBranchedAttempt(
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
        cost_safeguard=CostSafeguardController(1.0e9, 1.0e9, 1),
    ).run(iteration=0)
    assert result.learner.optimizer_steps[0].applied
    if chance_beta > 1:
        assert result.leaves > 1
        assert result.executed_edges > result.player_nodes
    else:
        assert result.sampled_tree_nodes >= minimum_tree_size
        assert result.root_executed_degree[0] > result.root_target_degree[0]
        assert result.root_target_degree == (1,)
