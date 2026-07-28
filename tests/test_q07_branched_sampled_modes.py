from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from torch import nn

from pbg.games.poker.queued import (
    BranchedSampledPokerIteration as PublicBranchedSampledPokerIteration,
)
from pbg.games.poker.queued import (
    NLHEGeometry,
    PokerStateMachine,
    sampled_partitions,
)
from pbg.games.poker.queued.branching import (
    BranchedTopologyRecorder,
    _discard_attempt,
    _ExpansionRecord,
    _StagedBranchedPayload,
    sampled_branched_chance_frontier_row_bytes,
    sampled_branched_topology_packet_row_bytes,
)
from pbg.games.poker.queued.branching import (
    _BranchedSampledPokerAttempt as BranchedSampledPokerIteration,
)
from pbg.queued import (
    BranchedModeConfig,
    BranchedOverflowRecovery,
    CapacityManifest,
    CostSafeguardController,
    DenseAdam,
    ExecutionPhase,
    ExternalWorkspaceBounds,
    FixedBranchController,
    GameGeometry,
    GeneratedOwners,
    HardwareLimits,
    KKTBranchController,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    NamedShape,
    OptimizerConfig,
    PartitionRows,
    RankCommonFailureAgreement,
    ReferenceState,
    RunState,
    build_storage_plan,
    child_path_factors,
    controller_from_state_dict,
    convert_queued_checkpoint_v1,
    decision_log_path_mass,
    load_checkpoint,
    log_kkt_intensity,
    mode_schema,
    randomized_round,
    save_checkpoint,
)
from pbg.queued.capacity import BRANCHED_ALIAS_PEAK_ROW_BYTES
from pbg.queued.learner import _alias_table
from pbg.queued.rollout import (
    AsyncTopologyPublisher,
    DeferredBranchedTopologyPublication,
    RolloutOwnerRegistry,
)
from pbg.queued.storage import DeviceFailure, StorageError
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
        item.name for item in sampled_partitions(geometry.players).inputs
    )
    return CapacityManifest(
        geometry=GameGeometry(
            geometry.players,
            geometry.stack_bb,
            geometry.rank_count,
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


def _phase7_layout(geometry: NLHEGeometry):
    return replace(
        _phase5_layout(geometry),
        chance_frontier_row_bytes=(
            sampled_branched_chance_frontier_row_bytes(geometry)
        ),
        topology_packet_row_bytes=(
            sampled_branched_topology_packet_row_bytes(geometry)
        ),
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


@CUDA
def test_phase7_admission_covers_fixed_and_full_decision_workspaces() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    layout = _phase7_layout(geometry)
    manifest = _manifest(device)
    self_plan = build_storage_plan(
        mode_schema("self_play", "branched.fixed", "sampled"),
        sampled_partitions(2),
        manifest,
        layout,
        device,
    )
    response_plan = build_storage_plan(
        mode_schema("best_response", "branched.fixed", "sampled"),
        sampled_partitions(2),
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
        == sampled_branched_chance_frontier_row_bytes(geometry)
    )
    assert (
        layout.topology_packet_row_bytes
        == sampled_branched_topology_packet_row_bytes(geometry)
    )
    with pytest.raises(ValueError, match="root count"):
        PublicBranchedSampledPokerIteration(
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
        sampled_partitions(2),
        replace(manifest, root_count=2),
        layout,
        device,
    )
    with pytest.raises(ValueError, match="root count"):
        PublicBranchedSampledPokerIteration(
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
    rounds = iter(((local, remote), (None, None)))
    agreement = RankCommonFailureAgreement(
        world_size=2,
        collect=lambda _local: next(rounds),
    )
    recovery = BranchedOverflowRecovery(
        iteration=5,
        replica=0,
        controller=controller,
        run_state=run,
        rank_agreement=agreement,
    )
    with pytest.warns(RuntimeWarning, match="RECOVERED_BRANCHED_OVERFLOW"):
        assert recovery.run(attempt, discarded.append) == "committed"
    assert len(discarded) == 1
    assert discarded[0].failure.logical_address == 19
    assert calls[1].retry_epoch == calls[0].retry_epoch + 1
    assert calls[1].logical_base == calls[0].logical_base + 3
    assert run.branch_controller_state == controller.state_dict()

    with pytest.raises(RuntimeError, match="terminal"):
        recovery.run(
            lambda _address: (_ for _ in ()).throw(RuntimeError("terminal")),
            discarded.append,
        )

    healthy_discards = []
    healthy_rounds = iter(((None, remote), (None, None)))
    healthy_recovery = BranchedOverflowRecovery(
        iteration=6,
        replica=0,
        controller=FixedBranchController(
            {"player": 2.0, "chance": 1.0},
            2,
            minimum_scale=0.25,
        ),
        run_state=RunState("phase7-healthy-rank", root_count=1),
        rank_agreement=RankCommonFailureAgreement(
            world_size=2,
            collect=lambda _local: next(healthy_rounds),
        ),
    )
    healthy_calls = []
    with pytest.warns(RuntimeWarning, match="RECOVERED_BRANCHED_OVERFLOW"):
        assert (
            healthy_recovery.run(
                lambda address: healthy_calls.append(address) or "committed",
                healthy_discards.append,
            )
            == "committed"
        )
    assert len(healthy_calls) == 2
    assert healthy_discards == [remote]


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
    with pytest.raises(Exception, match="offline converter"):
        load_checkpoint(
            prior_path,
            model=model,
            optimizer=optimizer,
            reference_model=reference,
            expected_scientific_config=scientific,
            generator=generator,
        )
    converted_path = tmp_path / "phase6-converted.pt"
    report = convert_queued_checkpoint_v1(
        prior_path,
        converted_path,
    )
    assert report["source_schema"] == "pbg-queued-checkpoint-v1"
    assert report["result_schema"] == "pbg-queued-checkpoint-v2"
    assert (tmp_path / "phase6-converted.pt.conversion.json").is_file()
    _, _, migrated, _ = load_checkpoint(
        converted_path,
        model=model,
        optimizer=optimizer,
        reference_model=reference,
        expected_scientific_config=scientific,
        generator=generator,
    )
    assert migrated.branch_controller_state is None
    assert migrated.cost_safeguard_state is None


@CUDA
def test_failed_branched_attempt_releases_every_generated_owner() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(2),
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
    assert owners.released_keys == {
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
        sampled_partitions(2),
        _manifest(device),
        replace(_phase7_layout(geometry), player_count=2),
        device,
    )
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    _discard_attempt(owners)
    assert owners.released_keys == {
        allocation.key for allocation in plan.allocations
    }


@CUDA
def test_untrained_unary_root_refill_restores_one_structural_root() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    recorder = BranchedTopologyRecorder(
        input_names=("pre",),
        p_train=2,
        topology_cap=8,
        topology_edge_cap=8,
        topology_packet_cap=4,
        device=device,
    )

    def payload(*, refill: bool, child_path: int, leaf_row: int):
        record = _ExpansionRecord(
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
        return _StagedBranchedPayload(
            (record,),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(0, dtype=torch.float32),
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(False),
            torch.tensor(1, dtype=torch.int64),
        )

    recorder.accept_payload(payload(refill=False, child_path=11, leaf_row=0))
    recorder.accept_payload(payload(refill=True, child_path=12, leaf_row=1))
    topology, _, leaf_count = recorder.seal()
    assert topology.node_count == 1
    assert topology.executed_degree.tolist() == [2]
    assert topology.target_degree.tolist() == [1]
    assert topology.parent.tolist() == [-1]
    assert leaf_count == 2
    recorder.release()


@CUDA
def test_branched_topology_publication_accepts_only_committed_payloads() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    plan = build_storage_plan(
        mode,
        sampled_partitions(2),
        _manifest(device),
        replace(_phase7_layout(geometry), player_count=2),
        device,
    )
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    publisher = AsyncTopologyPublisher(
        RolloutOwnerRegistry.from_generated(owners),
        p_train=2,
    )
    for sequence, committed in enumerate((False, True)):
        payload = f"payload-{sequence}"
        publication = DeferredBranchedTopologyPublication(
            lane=0,
            sequence=sequence,
            declared_nodes=1,
            declared_edges=1,
            payload=payload,
        )
        publisher.stage_branched(publication)
        success = torch.tensor([committed], dtype=torch.bool, device=device)
        publisher.finalize_branched(
            publication,
            success,
            stream=torch.cuda.current_stream(device),
        )
        accepted = publisher.drain_branched(block=True)
        assert accepted == ((payload,) if committed else ())
    publisher.assert_drained()
    owners.release_after(ExecutionPhase.ROLLOUT)
    for phase in (
        ExecutionPhase.QUEUE_DRAIN,
        ExecutionPhase.COMPLETION,
        ExecutionPhase.ALIGN_AND_SEAL,
        ExecutionPhase.TARGETS,
        ExecutionPhase.LEARNER,
        ExecutionPhase.FINAL_BACKWARD,
    ):
        owners.enter_phase(phase)
        owners.release_after(phase)
    owners.close()


@CUDA
def test_full_branched_iteration_is_family_pack_invariant() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")

    def execute(family_pack_cap: int) -> tuple[object, dict[str, torch.Tensor]]:
        torch.cuda.manual_seed(191)
        plan = build_storage_plan(
            mode,
            sampled_partitions(2),
            replace(_manifest(device), family_pack_cap=family_pack_cap),
            replace(_phase7_layout(geometry), player_count=2),
            device,
        )
        model = _BranchedModel(device, geometry.private_hands)
        result = BranchedSampledPokerIteration(
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
        sampled_partitions(2),
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
    result = BranchedSampledPokerIteration(
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
            sampled_partitions(2),
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
        BranchedSampledPokerIteration(
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
            sampled_partitions(2),
            replace(
                _manifest(device),
                learner_sigma_pack_cap=8,
                learner_action_pack_cap=action_cap,
            ),
            replace(_phase7_layout(geometry), player_count=2),
            device,
        )
        model = _BranchedModel(device, geometry.private_hands)
        result = BranchedSampledPokerIteration(
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
        sampled_partitions(2),
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
    result = PublicBranchedSampledPokerIteration(
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
        sampled_partitions(2),
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
    BranchedSampledPokerIteration(
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
    partitions = sampled_partitions(geometry.players)
    plan = build_storage_plan(
        mode,
        partitions,
        _manifest(device, geometry),
        replace(
            _phase7_layout(geometry),
            model_input_shapes=tuple(
                NamedShape(
                    partition.name,
                    (geometry.direct_model_input_width,),
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
    BranchedSampledPokerIteration(
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
        sampled_partitions(2),
        replace(_manifest(device), learner_sigma_pack_cap=1),
        replace(
            _phase7_layout(geometry),
            reference_cache_bytes=4 << 20,
        ),
        device,
    )
    model = _BranchedModel(device, geometry.private_hands)
    reference = _BranchedModel(device, geometry.private_hands)
    result = BranchedSampledPokerIteration(
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
    partitions = sampled_partitions(geometry.players)
    layout = replace(
        _phase7_layout(geometry),
        model_input_shapes=tuple(
            NamedShape(
                partition.name,
                (geometry.direct_model_input_width,),
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
                torch._assert_async(
                    model_input[:, -1].gt(0.5).all(),
                    "nonterminal branched fold omitted its sealed summary",
                )
                self.sealed_fold_seen = True
            return super().evaluate(
                input_type, model_input, private, owner
            )

    model = FoldCheckingModel()
    result = BranchedSampledPokerIteration(
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
        sampled_partitions(2),
        _manifest(device),
        replace(_phase7_layout(geometry), player_count=2),
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
