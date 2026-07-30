from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from pbg.games.poker.queued.folded import (
    folded_statistics,
    folded_statistics_workspace_row_bytes,
    pack_folded_statistics,
    sequential_family_workspace_row_bytes,
)
from pbg.games.poker.queued.hooks import (
    FOLD,
    FOLDED,
    NLHEGeometry,
    PAD_CARD,
    PokerForwardInput,
    PokerStateMachine,
    SequentialPokerHooks,
    rb_partitions,
    sampled_partitions,
)
from pbg.games.poker.queued.model import (
    QueuedPokerModel,
    _combo_class,
    _fold_line_metadata,
    build_model_from_config,
)
from pbg.games.poker.queued.operations import PokerOperations
from tests.iteration_support import EngineDrivenSequentialAttempt
from pbg.games.poker.queued.response import (
    SequentialSharedResponseEvaluator,
    balanced_evaluation_probe_seats,
    paired_probe_statistics,
)
from pbg.core.capacity import (
    ExecutionGeometry,
    GeneratedOwners,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
    build_storage_plan,
)
from pbg.games.poker.queued.config import SequentialModeConfig
from pbg.core.contracts import mode_schema
from pbg.core.learner import (
    LearnerSchedule,
    LearnerOperation,
    LossConfig,
    ModelEvaluation,
    PolicyEvaluation,
    SequentialLearnerBatch,
)
from pbg.core.targets import (
    ConversionMap,
    TargetAlgebra,
    rb_conversion_library_workspace_bytes,
    rb_conversion_mask_bytes,
    rb_conversion_workspace_row_bytes,
)
from pbg.core.optimization import DenseAdam, OptimizerConfig
from pbg.core.rollout import (
    AsyncTopologyPublisher,
    RolloutOwnerRegistry,
    SemanticFamilyFIFO,
)
from pbg.core.storage import SoAField
from pbg.core.capacity import ExecutionPhase
from tests.test_sequential_sampled_training import (
    _PartitionNetwork,
    _phase5_layout,
    _phase5_manifest,
    _publish_rows,
)


class _FinitePartitionNetwork(_PartitionNetwork):
    def forward(self, model_input: torch.Tensor) -> ModelEvaluation:
        return super().forward(
            torch.nan_to_num(model_input, neginf=-20.0, posinf=20.0)
        )


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _config(regime: str, estimator: str) -> dict[str, object]:
    value: dict[str, object] = {
        "mode": {
            "training_regime": regime,
            "rollout_mode": "sequential",
            "private_estimator": estimator,
        },
        "game": {"players": 2, "stack_bb": 8.0, "rank_count": 4},
        "model": {
            "d_model": 4,
            "n_heads": 1,
            "max_players": 2,
            "epsilon": 1.0e-8,
        },
        "loss": {},
        "optimizer": {},
        "schedule": {"batch_size": 2},
        "precision": {},
        "reference": {},
        "lambda": 0.95,
        "root_count": 2,
        "root_count_minimum": 1,
        "seed": 7,
    }
    if estimator == "rao_blackwell":
        value["completion"] = {"n_samples": 8, "strata": "default"}
    if regime == "best_response":
        value["frozen_profile"] = {"sha256": "a" * 64}
        value["response_initialization"] = "random"
    return value


@pytest.mark.parametrize(
    ("regime", "estimator"),
    (
        ("self_play", "sampled"),
        ("best_response", "sampled"),
        ("self_play", "rao_blackwell"),
        ("best_response", "rao_blackwell"),
    ),
)
def test_all_sequential_public_configs_are_closed(
    regime: str, estimator: str
) -> None:
    config = SequentialModeConfig.from_mapping(_config(regime, estimator))
    assert config.scientific_mapping()["mode"] == {
        "training_regime": regime,
        "rollout_mode": "sequential",
        "private_estimator": estimator,
    }


@CUDA
def test_native_postflop_model_consumes_signed_folded_summary() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    scientific = _config("best_response", "rao_blackwell")
    scientific["game"] = {
        "players": 3,
        "stack_bb": 8.0,
        "rank_count": 4,
    }
    scientific["model"] = {
        "d_model": 8,
        "n_heads": 1,
        "n_layers_pre": 1,
        "n_layers_post": 1,
        "max_players": 3,
        "class_embedding_dim": 4,
        "ffn_hidden": 16,
        "folder_beta_rank": 4,
        "folder_line_dim": 4,
        "epsilon": 1.0e-8,
    }
    model = build_model_from_config(scientific, device)
    geometry = model.geometry
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        1, seed=11, iteration=0, retry_epoch=0, device=device
    )
    status = state.status.clone()
    status[:, 2] = FOLDED
    pending = state.pending.clone()
    pending[:, 2] = False
    state = replace(
        state,
        street=torch.ones_like(state.street),
        board_cards=torch.tensor(
            [[
                geometry.first_card,
                geometry.first_card + 1,
                geometry.first_card + 2,
                PAD_CARD,
                PAD_CARD,
            ]],
            dtype=torch.uint8,
            device=device,
        ),
        status=status,
        pending=pending,
        to_act=torch.zeros_like(state.to_act),
    )
    log_reach = torch.zeros(
        (1, geometry.players, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    log_reach[:, 2] = torch.linspace(
        -2.0, 0.0, geometry.private_hands, device=device
    )
    history = machine.empty_history(1, device)
    history.actor[:, 0] = 2
    history.street[:, 0] = 0
    history.action[:, 0] = FOLD
    history.mask[:, 0] = True
    line = _fold_line_metadata(
        history,
        state.status,
        line_dim=4,
        stack_bb=geometry.stack_bb,
    )
    assert line[0, 2, 0] == 0.0
    assert line[0, 2, 1] > 0.0
    model_input = machine.direct_model_input(
        "post_2", state, log_reach, history
    )
    summary = folded_statistics(
        geometry,
        board_cards=state.board_cards,
        range_weights=log_reach.exp(),
        status=state.status,
        combo_cards=machine.combo_cards(
            torch.arange(
                geometry.private_hands,
                dtype=torch.int32,
                device=device,
            )
        ),
        beta_rank=geometry.deck_cards,
        line_dim=0,
        tight_mask=torch.zeros_like(state.status, dtype=torch.bool),
    )
    model_input[:, geometry.postflop_base_model_input_width :] = (
        pack_folded_statistics(geometry, summary)
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        acc_events=True,
    ) as profile:
        evaluation = model.forward_sequential_partition(
            "post_2", model_input
        )
    scalar_events = {
        event.key
        for event in profile.key_averages()
        if event.key in {"aten::_local_scalar_dense", "aten::is_nonzero"}
    }
    assert not scalar_events
    assert evaluation.value.shape == (1, geometry.private_hands)
    assert torch.isfinite(evaluation.value).all()
    evaluation.value.mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.postflop.parameters()
    )


def test_probe_is_the_ordinary_best_response_rb_schema() -> None:
    value = _config("best_response", "rao_blackwell")
    value["probe"] = {
        "target_sha256": "a" * 64,
        "initial_model_sha256": "c" * 64,
    }
    config = SequentialModeConfig.from_mapping(value)
    assert config.applicable["probe"] == {
        "target_sha256": "a" * 64,
        "initial_model_sha256": "c" * 64,
    }
    assert mode_schema(
        **{
            "training_regime": config.mode["training_regime"],
            "rollout_mode": config.mode["rollout_mode"],
            "private_estimator": config.mode["private_estimator"],
        }
    ).key == "best_response/sequential/rao_blackwell"
    keys = torch.arange(12, dtype=torch.int64)
    seats = balanced_evaluation_probe_seats(keys, 3)
    assert seats.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0]
    # Use a complete P^2 design for the paired lower-bound report.
    seats = balanced_evaluation_probe_seats(torch.arange(9), 3)
    summary = paired_probe_statistics(
        torch.arange(9, dtype=torch.float64), seats, 3
    )
    assert summary.per_seat_gain == (1.0, 4.0, 7.0)
    assert summary.approximate_nashconv_lower_bound == 12.0
    assert summary.confidence_level == 0.95
    assert summary.confidence_interval[0] < summary.mean_gain
    assert summary.confidence_interval[1] > summary.mean_gain

    observed: list[tuple[torch.Tensor, torch.Tensor]] = []

    def rollout(ids: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        observed.append((ids.clone(), selected.clone()))
        return ids.float()

    report = SequentialSharedResponseEvaluator(
        3, "a" * 64, "c" * 64
    ).evaluate(
        torch.arange(9, dtype=torch.int64),
        rollout,
        lambda ids, selected: rollout(ids, selected) - 1.0,
    )
    assert report.semantics == "shared_seat_approximate_response_v1"
    assert report.target_sha256 == "a" * 64
    assert report.initial_model_sha256 == "c" * 64
    assert report.initialization == "random"
    assert report.statistics.mean_gain == 1.0
    assert len(observed) == 2
    assert all(
        torch.equal(left, right)
        for left, right in zip(observed[0], observed[1])
    )
    with pytest.raises(ValueError, match="consecutive"):
        SequentialSharedResponseEvaluator(
            3, "a" * 64, "c" * 64
        ).evaluate(
            torch.tensor(
                [0, 9, 18, 3, 12, 21, 6, 15, 24],
                dtype=torch.int64,
            ),
            rollout,
            rollout,
        )
    long_values = torch.arange(45, dtype=torch.float64)
    long_seats = torch.arange(45, dtype=torch.int64).remainder(3)
    long_summary = paired_probe_statistics(
        long_values, long_seats, 3
    )
    critical = (
        long_summary.confidence_interval[1] - long_summary.mean_gain
    ) / long_summary.standard_error
    assert critical > 1.96
    assert critical == pytest.approx(2.0154, abs=1.0e-3)


def test_sequential_config_rejects_inapplicable_mode_fields() -> None:
    sampled = _config("self_play", "sampled")
    sampled["completion"] = {"n_samples": 8, "strata": "default"}
    with pytest.raises(ValueError, match="inapplicable"):
        SequentialModeConfig.from_mapping(sampled)
    self_play = _config("self_play", "rao_blackwell")
    self_play["frozen_profile"] = {"sha256": "a" * 64}
    with pytest.raises(ValueError, match="best_response"):
        SequentialModeConfig.from_mapping(self_play)
    sampled_response = _config("best_response", "sampled")
    sampled_response["probe"] = {
        "target_sha256": "a" * 64,
        "initial_model_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="requires sequential RB"):
        SequentialModeConfig.from_mapping(sampled_response)



def test_rb_partitions_are_k_wide_preflop_and_h_wide_postflop() -> None:
    geometry = NLHEGeometry(3, 8.0, 4)
    partitions = rb_partitions(geometry)
    assert tuple((item.name, item.width) for item in partitions.outputs) == (
        ("pre", 16),
        ("post", 120),
    )
    assert partitions.output_for("pre").name == "pre"
    assert partitions.output_for("post_2").name == "post"
    assert partitions.output_for("post_3").name == "post"


def test_max_geometry_workspace_bounds_are_explicit() -> None:
    geometry = NLHEGeometry(6, 128.0, 13)
    assert geometry.private_hands == 1326
    assert geometry.folded_summary_width == 4402
    assert folded_statistics_workspace_row_bytes(geometry) == 3_691_536
    assert sequential_family_workspace_row_bytes(geometry, 4) == 4_290_160
    assert rb_conversion_workspace_row_bytes(1326, 169) == 17_264
    assert rb_conversion_mask_bytes(1326, 169) == 896_376
    assert (
        rb_conversion_library_workspace_bytes(1326, 169)
        == 9_306_112
    )


@CUDA
def test_full_deck_h_to_k_conversion_is_deterministic_and_bounded() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(6, 128.0, 13)
    classes = _combo_class(geometry)
    host_algebra = TargetAlgebra(
        rb_partitions(geometry),
        (
            ConversionMap(
                "post",
                "pre",
                tuple(int(value) for value in classes.tolist()),
            ),
        ),
    )
    value = torch.arange(
        8 * geometry.private_hands,
        dtype=torch.float32,
        device=device,
    ).reshape(8, geometry.private_hands)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    algebra = host_algebra.device(device)
    first = algebra.convert(value, 1, 0)
    second = algebra.convert(value, 1, 0)
    torch.cuda.synchronize()
    assert torch.equal(first, second)
    assert (
        torch.cuda.max_memory_allocated(device) - baseline
        <= rb_conversion_library_workspace_bytes(1326, 169)
        + 8 * rb_conversion_workspace_row_bytes(1326, 169)
    )


@CUDA
def test_full_deck_folded_statistics_stays_within_admitted_workspace() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(6, 128.0, 13)
    machine = PokerStateMachine(geometry)
    batch = 8
    state = machine.initial_roots(
        batch, seed=59, iteration=0, retry_epoch=0, device=device
    )
    status = state.status.clone()
    status[:, 3:] = FOLDED
    ranges = torch.rand(
        (batch, geometry.players, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    combo_cards = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    summary = folded_statistics(
        geometry,
        board_cards=state.board_cards,
        range_weights=ranges,
        status=status,
        combo_cards=combo_cards,
        beta_rank=16,
        line_dim=0,
        tight_mask=status.eq(FOLDED),
    )
    torch.cuda.synchronize()
    incremental_peak = (
        torch.cuda.max_memory_allocated(device) - baseline
    )
    assert torch.isfinite(summary["hand_log_ratio"]).all()
    assert incremental_peak <= (
        batch * folded_statistics_workspace_row_bytes(geometry)
    )


def test_every_model_forward_uses_the_sealed_family_input() -> None:
    hooks = object.__new__(SequentialPokerHooks)
    hooks.geometry = NLHEGeometry(3, 8.0, 4)
    pre_width = hooks.geometry.preflop_model_input_width
    post_width = hooks.geometry.postflop_model_input_width
    pre_raw = torch.zeros((1, 1, pre_width))
    post_raw = torch.zeros((1, 1, post_width))
    pre = SimpleNamespace(input_type="pre", fields={"model_input": pre_raw})
    post_two = SimpleNamespace(input_type="post_2", fields={"model_input": post_raw})
    post_three = SimpleNamespace(input_type="post_3", fields={"model_input": post_raw})
    assert torch.equal(
        hooks._forward_model_input(pre), pre_raw.reshape(1, pre_width)
    )
    assert torch.equal(
        hooks._forward_model_input(post_two), post_raw.reshape(1, post_width)
    )
    assert torch.equal(
        hooks._forward_model_input(post_three), post_raw.reshape(1, post_width)
    )


@CUDA
def test_repository_folded_statistics_are_deterministic_fp32() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(3, 8.0, 4)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        2, seed=19, iteration=0, retry_epoch=0, device=device
    )
    status = state.status.clone()
    status[:, 2] = FOLDED
    weights = torch.rand(
        (2, 3, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    combo_cards = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    arguments = dict(
        board_cards=state.board_cards,
        range_weights=weights,
        status=status,
        combo_cards=combo_cards,
        beta_rank=4,
        line_dim=3,
        line_metadata=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            ],
            dtype=torch.float32,
            device=device,
        ),
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        first = folded_statistics(geometry, **arguments)
        second = folded_statistics(geometry, **arguments)
    assert first.keys() == second.keys()
    assert all(value.dtype in (torch.float32, torch.bool, torch.int64)
               for value in first.values())
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert first["folder_mask"][:, 2].all()
    assert not first["folder_mask"][:, :2].any()
    assert torch.count_nonzero(first["beta_eigenvalue"]) > 0
    assert torch.count_nonzero(first["folder_line_meta"]) > 0
    torch.testing.assert_close(
        first["folder_log_mass"][:, 2],
        weights[:, 2].sum(-1).log(),
    )


@CUDA
def test_folded_lambda_and_signed_beta_match_one_folder_finite_differences() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        1, seed=31, iteration=0, retry_epoch=0, device=device
    )
    combo = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    raw = torch.arange(
        1,
        geometry.private_hands + 1,
        dtype=torch.float32,
        device=device,
    )
    probability = raw / raw.sum()
    ranges = torch.zeros(
        (1, 2, geometry.private_hands), dtype=torch.float32, device=device
    )
    ranges[0, 1] = raw
    status = state.status.clone()
    status[:, 1] = FOLDED
    result = folded_statistics(
        geometry,
        board_cards=state.board_cards,
        range_weights=ranges,
        status=status,
        combo_cards=combo,
        beta_rank=geometry.deck_cards,
        line_dim=0,
    )
    local = (combo - geometry.first_card).long()
    marginal = torch.zeros(
        geometry.deck_cards, dtype=torch.float32, device=device
    )
    marginal.scatter_add_(0, local[:, 0], probability)
    marginal.scatter_add_(0, local[:, 1], probability)
    alpha = torch.log1p(-marginal)
    beta_reference = torch.zeros(
        (geometry.deck_cards, geometry.deck_cards),
        dtype=torch.float32,
        device=device,
    )
    pair_survival = (
        1.0
        - marginal[local[:, 0]]
        - marginal[local[:, 1]]
        + probability
    )
    pair_beta = (
        torch.log(pair_survival)
        - alpha[local[:, 0]]
        - alpha[local[:, 1]]
    )
    beta_reference[local[:, 0], local[:, 1]] = pair_beta
    beta_reference[local[:, 1], local[:, 0]] = pair_beta
    lambda_reference = (
        alpha[local[:, 0]]
        + alpha[local[:, 1]]
        + pair_beta
    )
    torch.testing.assert_close(
        result["hand_log_ratio"][0], lambda_reference, atol=2.0e-5, rtol=2.0e-5
    )
    reconstructed = torch.einsum(
        "cr,r,dr->cd",
        result["beta_eigenvector"][0],
        result["beta_eigenvalue"][0],
        result["beta_eigenvector"][0],
    )
    torch.testing.assert_close(
        reconstructed, beta_reference, atol=2.0e-5, rtol=2.0e-5
    )
    assert (result["beta_eigenvalue"] < 0).any()
    assert (result["beta_eigenvalue"] > 0).any()


@CUDA
def test_folded_exact_combo_content_and_two_folder_oracle() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(3, 8.0, 4)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        1, seed=37, iteration=0, retry_epoch=0, device=device
    )
    combo = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    local = (combo - geometry.first_card).long()
    first = torch.arange(
        1,
        geometry.private_hands + 1,
        dtype=torch.float32,
        device=device,
    )
    second = torch.flip(first, dims=(0,))
    ranges = torch.zeros(
        (1, 3, geometry.private_hands), dtype=torch.float32, device=device
    )
    ranges[0, 1] = first
    ranges[0, 2] = second
    status = state.status.clone()
    status[:, 1:] = FOLDED
    result = folded_statistics(
        geometry,
        board_cards=state.board_cards,
        range_weights=ranges,
        status=status,
        combo_cards=combo,
        beta_rank=geometry.deck_cards,
        line_dim=0,
    )
    p = torch.stack((first / first.sum(), second / second.sum()))
    disjoint = (
        local[:, None, :, None]
        .ne(local[None, :, None, :])
        .all(-1)
        .all(-1)
    )
    base = torch.einsum("h,g,hg->", p[0], p[1], disjoint.float())
    for hand in range(0, geometry.private_hands, 17):
        removed = local[hand]
        survives = local.ne(removed[0]).all(-1) & local.ne(removed[1]).all(-1)
        partition = torch.einsum(
            "h,g,hg->",
            p[0] * survives,
            p[1] * survives,
            disjoint.float(),
        )
        torch.testing.assert_close(
            result["hand_log_ratio"][0, hand],
            torch.log(partition / base),
            atol=3.0e-5,
            rtol=3.0e-5,
        )


@CUDA
def test_evaluator_selected_completion_matches_the_selected_dense_slice() -> None:
    import poker_gpu_evaluator as poker

    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(3, 8.0, 4)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        1, seed=43, iteration=0, retry_epoch=0, device=device
    )
    leaf = machine.rb_leaf(
        state,
        torch.zeros(
            (1, geometry.players, geometry.private_hands),
            dtype=torch.float32,
            device=device,
        ),
        seed=torch.tensor([71], dtype=torch.int64, device=device),
    )
    common = {
        "board_cards": leaf["leaf_public_state"],
        "range_weights": leaf["range_weights"],
        "stratum_level": leaf["stratum_level"],
        "payoff_baseline": leaf["payoff_baseline"],
        "stratum_amount": leaf["stratum_amount"],
        "n_samples": 4,
        "seed": leaf["seed"],
    }
    evaluator = poker.bind(rank_count=geometry.rank_count)
    dense = evaluator.complete(**common)
    selected = torch.tensor([1], dtype=torch.int32, device=device)
    compact = evaluator.complete_selected(
        **common, selected_player=selected
    )
    torch.testing.assert_close(
        compact.p_hat, dense.p_hat[:, 1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        compact.v_hat, dense.v_hat[:, 1], rtol=0, atol=0
    )


@CUDA
def test_three_loose_folder_estimator_calibrates_to_exact_enumeration() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(4, 8.0, 4)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        1, seed=41, iteration=0, retry_epoch=0, device=device
    )
    combo = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    local = (combo - geometry.first_card).long()
    ordinal = torch.arange(
        geometry.private_hands, dtype=torch.float32, device=device
    )
    probability = torch.stack(
        tuple(
            1.0 + 0.1 * torch.sin(ordinal * (folder + 1) * 0.07)
            for folder in range(3)
        )
    )
    probability /= probability.sum(-1, keepdim=True)
    ranges = torch.zeros(
        (1, 4, geometry.private_hands), dtype=torch.float32, device=device
    )
    ranges[0, 1:] = probability
    status = state.status.clone()
    status[:, 1:] = FOLDED
    result = folded_statistics(
        geometry,
        board_cards=state.board_cards,
        range_weights=ranges,
        status=status,
        combo_cards=combo,
        beta_rank=geometry.deck_cards,
        line_dim=0,
    )
    disjoint = (
        local[:, None, :, None]
        .ne(local[None, :, None, :])
        .all(-1)
        .all(-1)
        .float()
    )

    def exact_partition(weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "h,i,j,hi,hj,ij->",
            weights[0],
            weights[1],
            weights[2],
            disjoint,
            disjoint,
            disjoint,
        )

    base = exact_partition(probability)
    for hand in range(0, geometry.private_hands, 17):
        removed = local[hand]
        survives = (
            local.ne(removed[0]).all(-1)
            & local.ne(removed[1]).all(-1)
        )
        oracle = torch.log(
            exact_partition(probability * survives) / base
        )
        torch.testing.assert_close(
            result["hand_log_ratio"][0, hand],
            oracle,
            atol=1.0e-3,
            rtol=1.0e-3,
        )


@CUDA
def test_folded_owner_refreshes_model_input_and_releases_at_seal() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "sequential", "sampled")
    partitions = sampled_partitions(geometry)
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            partitions,
            _phase5_manifest(device),
            _phase5_layout(),
            device,
        )
    )
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    registry = RolloutOwnerRegistry.from_generated(owners)
    queues = {
        name: SemanticFamilyFIFO(
            name,
            partitions,
            owners.slab("FamilyWork", name),
            owners.slab("FamilyRetained", name),
            owners.slab("QueueMetadata", name),
        )
        for name in ("pre", "post_2")
    }
    publisher = AsyncTopologyPublisher(
        registry,
        p_train=2,
        input_names=("pre", "post_2"),
        input_to_output=(0, 0),
    )
    hooks = SequentialPokerHooks(
        geometry,
        registry,
        queues,
        seed=3,
        iteration=0,
        mode=mode,
        topology_stagers=publisher.stagers,
    )
    folded = owners.slab("FoldedRanges", "pre")
    class_range = torch.arange(
        1, geometry.rank_count**2 + 1, dtype=torch.float32, device=device
    )
    class_range /= class_range.sum()
    _publish_rows(
        folded,
        1,
        {
            "range": class_range[None],
            "log_mass": torch.zeros((1,), device=device),
        },
    )
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        1, seed=5, iteration=0, retry_epoch=0, device=device
    )
    batch = SimpleNamespace(
        input_type="pre",
        fields={
            "model_input": machine.direct_model_input("pre", state)[None],
            "folded_range_row": torch.tensor(
                [[[0, -1]]], dtype=torch.int32, device=device
            ),
        }
    )
    reach, _ = hooks._resolved_model_context(batch)
    weights = reach[:, 0].exp()
    classes = hooks._combo_class()
    reconstructed = torch.zeros_like(class_range)
    reconstructed.scatter_add_(0, classes, weights[0])
    torch.testing.assert_close(reconstructed, class_range)
    second_reach, _ = hooks._resolved_model_context(batch)
    assert torch.equal(reach, second_reach)
    board = torch.tensor(
        [[geometry.first_card, geometry.first_card + 1, geometry.first_card + 2,
          PAD_CARD, PAD_CARD]],
        dtype=torch.uint8,
        device=device,
    )
    post_state = replace(
        state,
        street=torch.ones_like(state.street),
        board_cards=board,
    )
    batch.fields["model_input"] = machine.direct_model_input(
        "post_2", post_state
    )[None]
    batch.input_type = "post_2"
    chance_reach, _ = hooks._resolved_model_context(batch)
    combo_cards = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    conflict = combo_cards[None].eq(
        board[0, :3, None, None]
    ).any(0).any(-1)
    assert torch.isneginf(chance_reach[0, 0, conflict]).all()
    owners.release_after(ExecutionPhase.ROLLOUT)
    owners.bind_empty_layer_work_for_abort()
    for phase in (
        ExecutionPhase.QUEUE_DRAIN,
        ExecutionPhase.COMPLETION,
        ExecutionPhase.ALIGN_AND_SEAL,
    ):
        owners.enter_phase(phase)
        released = owners.release_after(phase)
    assert {key.partition for key in released if key.owner == "FoldedRanges"} == {
        "pre",
        "post",
    }
    for phase in (
        ExecutionPhase.TARGETS,
        ExecutionPhase.LEARNER,
        ExecutionPhase.FINAL_BACKWARD,
    ):
        owners.enter_phase(phase)
        owners.release_after(phase)
    owners.close()


class _RBModel(nn.Module):
    def __init__(self, width: int, device: torch.device) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.linspace(-0.2, 0.3, width, device=device))
        self.logit = nn.Parameter(torch.linspace(-0.1, 0.2, width, device=device))

    def operation(
        self, _name: str, model_input: torch.Tensor,
        _private: torch.Tensor, _owner: torch.Tensor
    ) -> ModelEvaluation:
        rows = model_input.shape[0]
        return ModelEvaluation(
            MappingProxyType({"logit": self.logit[None].expand(rows, -1)}),
            self.value[None].expand(rows, -1),
        )


def _policy(
    _name: str,
    _model_input: torch.Tensor,
    raw: MappingProxyType,
    _action: torch.Tensor,
    _private: torch.Tensor,
    _owner: torch.Tensor,
    _reference: object,
) -> PolicyEvaluation:
    logit = raw["logit"].float()
    return PolicyEvaluation(logit, torch.zeros_like(logit), torch.zeros_like(logit))


@CUDA
@pytest.mark.parametrize("regime", ("self_play", "best_response"))
def test_rb_loss_stays_in_natural_domain_and_uses_fp32_posterior(
    regime: str,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    width, rows = 16, 2
    model = _RBModel(width, device)
    posterior = torch.arange(
        1, rows * width + 1, dtype=torch.float32, device=device
    ).reshape(rows, width)
    posterior /= posterior.sum(-1, keepdim=True)
    fields = MappingProxyType(
        {
            "model_input": torch.zeros((rows, 1), device=device),
            "owner": torch.zeros(rows, dtype=torch.int32, device=device),
            "action": torch.ones(rows, dtype=torch.int32, device=device),
            "old_log_policy": torch.zeros((rows, width), device=device),
            "old_value": torch.zeros((rows, width), device=device),
            "value_target": torch.full((rows, width), 0.25, device=device),
            "posterior": posterior,
            "advantage_target": torch.full(
                (rows, width), 0.5, device=device
            ),
        }
    )
    batch = SequentialLearnerBatch(
        "pre", torch.arange(rows, device=device), fields
    )
    result = LearnerOperation(
        mode_schema(regime, "sequential", "rao_blackwell"),
        LossConfig(),
        model.operation,
        _policy,
    ).forward_loss_backward(batch, iteration=0)
    assert torch.isfinite(result.total)
    assert model.value.grad is not None and model.value.grad.dtype == torch.float32
    assert model.logit.grad is not None and model.logit.grad.dtype == torch.float32


@CUDA
def test_best_response_rollout_calls_each_model_only_for_its_role_rows() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    calls: dict[str, list[int]] = {}

    def operation(name: str):
        def run(
            _input_type: str,
            model_input: torch.Tensor,
            _private: torch.Tensor,
            _owner: torch.Tensor,
        ) -> ModelEvaluation:
            calls[name] = model_input[:, 0].to(torch.int64).tolist()
            rows = model_input.shape[0]
            value = model_input[:, :1] + (10.0 if name == "frozen" else 0.0)
            return ModelEvaluation(
                MappingProxyType({"logit": torch.zeros((rows, 1), device=device)}),
                value,
            )

        return run

    runner = object.__new__(PokerOperations)
    runner.modes = (
        mode_schema("best_response", "sequential", "sampled"),
    )
    runner.rollout_operations = (operation("trained"),)
    runner.frozen_operations = (operation("frozen"),)
    runner.inference = None
    runner.precision = "fp32"
    forward = PokerForwardInput(
        "pre",
        torch.arange(4, dtype=torch.float32, device=device)[:, None],
        torch.zeros((4, 2), dtype=torch.int32, device=device),
        torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=device),
        torch.tensor([0, 0, 2, 1], dtype=torch.uint8, device=device),
    )
    merged = runner._model(
        0,
        forward,
        torch.tensor([0], dtype=torch.int64, device=device),
    )
    assert calls == {"trained": [0, 2], "frozen": [1, 3]}
    torch.testing.assert_close(
        merged["value"][:, 0],
        torch.tensor([0.0, 11.0, 2.0, 13.0], device=device),
    )


@CUDA
@pytest.mark.parametrize(
    ("regime", "estimator"),
    (
        ("self_play", "sampled"),
        ("best_response", "sampled"),
        ("self_play", "rao_blackwell"),
        ("best_response", "rao_blackwell"),
    ),
)
def test_all_four_sequential_modes_complete_an_optimizer_step(
    regime: str, estimator: str
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema(regime, "sequential", estimator)
    rb = estimator == "rao_blackwell"
    layout = replace(
        _phase5_layout(),
        family_workspace_row_bytes=sequential_family_workspace_row_bytes(
            geometry, 1
        ),
    )
    manifest = _phase5_manifest(device)
    if rb:
        layout = replace(
            layout,
            evaluator_private_width=geometry.private_hands,
            completion_workspace_row_bytes=4096,
        )
        manifest = replace(
            manifest,
            library_workspace_bytes=rb_conversion_library_workspace_bytes(
                geometry.private_hands, geometry.rank_count**2
            ),
        )
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            rb_partitions(geometry) if rb else sampled_partitions(geometry),
            manifest,
            layout,
            device,
        )
    )

    def make_model() -> QueuedPokerModel:
        result = QueuedPokerModel(
            geometry,
            _FinitePartitionNetwork(
                geometry.preflop_model_input_width,
                geometry.rank_count**2,
                device,
            ),
            _FinitePartitionNetwork(
                geometry.postflop_model_input_width,
                geometry.private_hands,
                device,
            ),
        ).to(device)
        with torch.no_grad():
            for partition in (result.preflop, result.postflop):
                partition.affine.weight.zero_()
                partition.affine.bias.zero_()
        return result

    model = make_model()
    frozen = make_model() if regime == "best_response" else None
    generator = torch.Generator(device=device)
    generator.manual_seed(101)
    iteration_runner = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=owners,
        mode=mode,
        model=model,
        model_operation=(
            model.rb_learner_operation if rb else model.learner_operation
        ),
        rollout_operation=model.rollout_operation,
        frozen_operation=(
            None if frozen is None else frozen.rollout_operation
        ),
        optimizer=DenseAdam(
            model, OptimizerConfig(name="adamw", learning_rate=1.0e-3)
        ),
        loss=LossConfig(),
        schedule=LearnerSchedule(batch_size=1),
        generator=generator,
        seed=17,
        lambda_=0.95,
        completion_samples=4 if rb else 0,
        response_cutoff=regime == "best_response" and rb,
    )
    result = iteration_runner.run(iteration=0)
    assert result.learner.optimizer_steps
    assert all(step.applied for step in result.learner.optimizer_steps)
    if rb:
        assert result.posterior_effective_hands is not None
        assert 0 < result.posterior_effective_hands <= result.leaves * 4
    else:
        assert result.posterior_effective_hands is None


@CUDA
@pytest.mark.parametrize("estimator", ("sampled", "rao_blackwell"))
def test_multiway_best_response_uses_the_shared_learner_and_optimizer(
    estimator: str,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(3, 8.0, 4)
    mode = mode_schema("best_response", "sequential", estimator)
    partitions = (
        rb_partitions(geometry)
        if estimator == "rao_blackwell"
        else sampled_partitions(geometry)
    )
    inputs = tuple(item.name for item in partitions.inputs)
    base_manifest = _phase5_manifest(device)
    manifest = replace(
        base_manifest,
        geometry=ExecutionGeometry(3, 26, 64, 4096),
        player_node_cap=tuple(PartitionRows(name, 40) for name in inputs),
        trained_player_cap=tuple(PartitionRows(name, 40) for name in inputs),
        action_cap=tuple(PartitionRows(name, 40) for name in inputs),
        leaf_cap=32,
        topology_cap=64,
        topology_edge_cap=64,
        target_arena_cap=128,
        library_workspace_bytes=(
            rb_conversion_library_workspace_bytes(
                geometry.private_hands, geometry.rank_count**2
            )
            if estimator == "rao_blackwell"
            else base_manifest.library_workspace_bytes
        ),
    )
    layout = PhysicalLayout(
        model_input_shapes=tuple(
            NamedShape(name, (geometry.model_input_width(name),))
            for name in inputs
        ),
        player_count=3,
        stratum_count=3,
        evaluator_private_width=geometry.private_hands,
        family_workspace_row_bytes=sequential_family_workspace_row_bytes(
            geometry, 1
        ),
        chance_frontier_row_bytes=32,
        completion_workspace_row_bytes=4096,
        learner_sigma_workspace_row_bytes=4096,
        learner_action_workspace_row_bytes=4096,
        topology_packet_row_bytes=88,
        opaque_fields=(
            SoAField("leaf_public_state", torch.uint8, (5,), fill=255),
            SoAField("leaf_private_profile", torch.int32, (3,), fill=-1),
        ),
    )
    owners = GeneratedOwners(
        build_storage_plan(mode, partitions, manifest, layout, device)
    )

    def make_model() -> QueuedPokerModel:
        result = QueuedPokerModel(
            geometry,
            _FinitePartitionNetwork(
                geometry.preflop_model_input_width,
                geometry.rank_count**2,
                device,
            ),
            _FinitePartitionNetwork(
                geometry.postflop_model_input_width,
                geometry.private_hands,
                device,
            ),
        ).to(device)
        with torch.no_grad():
            for partition in (result.preflop, result.postflop):
                partition.affine.weight.zero_()
                partition.affine.bias.zero_()
        return result

    model = make_model()
    frozen = make_model()
    generator = torch.Generator(device=device).manual_seed(911)
    runner = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=owners,
        mode=mode,
        model=model,
        model_operation=(
            model.rb_learner_operation
            if estimator == "rao_blackwell"
            else model.learner_operation
        ),
        rollout_operation=model.rollout_operation,
        frozen_operation=frozen.rollout_operation,
        optimizer=DenseAdam(
            model, OptimizerConfig(name="adamw", learning_rate=1.0e-3)
        ),
        loss=LossConfig(),
        schedule=LearnerSchedule(batch_size=1),
        generator=generator,
        seed=23,
        lambda_=0.95,
        completion_samples=4 if estimator == "rao_blackwell" else 0,
    )
    result = runner.run(iteration=0)
    assert result.learner.optimizer_steps
    assert all(step.applied for step in result.learner.optimizer_steps)
    if estimator == "rao_blackwell":
        assert result.posterior_effective_hands is not None
        assert 0 < result.posterior_effective_hands <= result.leaves * 4
    else:
        assert result.posterior_effective_hands is None
