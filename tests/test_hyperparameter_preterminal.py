from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.hyperparameter_search import exposure
from experiments.hyperparameter_search.gradient_noise import paired_comparison
from experiments.hyperparameter_search.protocol import ExposureClock, validate_control
from pbg.core.contracts import PrivateEstimator, RolloutMode
from pbg.games.poker.queued.hooks import PAD_CARD

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def board(rows):
    return torch.full((rows, 5), PAD_CARD, dtype=torch.uint8, device="cuda")


def weights():
    result = torch.zeros((2, 2, 6), device="cuda")
    result[0, 0, :2] = 0.5
    result[0, 1] = 1 / 6
    result[1, 0, 0] = 1
    result[1, 1, 5] = 1
    return result


def combos():
    return torch.combinations(
        torch.arange(48, 52, device="cuda", dtype=torch.int32), r=2
    )


@cuda
def test_leaf_entropy_weights_every_ancestor_and_preserves_range_scale():
    ranges = weights()
    counts = torch.tensor([[[2, 1], [0, 1]], [[1, 0], [0, 2]]], device="cuda")
    result = exposure.preterminal_posterior_exposure(ranges, board(2), combos(), counts)
    assert int(result[0]["fresh_player_decisions"]) == 3
    assert int(result[1]["fresh_player_decisions"]) == 4
    assert float(result[0]["sum_posterior_exp_entropy"]) == pytest.approx(5)
    assert float(result[1]["sum_posterior_exp_entropy"]) == pytest.approx(6)
    # A later action narrows the leaf. All its earlier decisions must change too.
    ranges[0, 0, 1] = 0
    changed = exposure.preterminal_posterior_exposure(
        ranges, board(2), combos(), counts
    )
    assert sum(
        float(v["sum_posterior_exp_entropy"]) for v in changed.values()
    ) == pytest.approx(7)
    scaled = exposure.preterminal_posterior_exposure(
        ranges * 71, board(2), combos(), counts
    )
    for family in changed:
        torch.testing.assert_close(
            changed[family]["sum_posterior_exp_entropy"],
            scaled[family]["sum_posterior_exp_entropy"],
        )


def topology():
    return SimpleNamespace(
        node_count=3,
        executed_degree=torch.tensor([1, 1, 1], dtype=torch.int32),
        child_begin=torch.tensor([0, 1, 2], dtype=torch.int32),
        target_kind=torch.tensor([0, 1, 1], dtype=torch.int32),
        target_row=torch.tensor([1, 0, 1], dtype=torch.int32),
        owner=torch.tensor([0, 0, 1], dtype=torch.int32),
    )


def bindings():
    return SimpleNamespace(input_type=torch.tensor([0, 0, 1], dtype=torch.int32))


@cuda
def test_leaf_mapping_counts_each_ancestor_once():
    counts = exposure.sequential_leaf_decision_counts(
        topology(), bindings(), 2, 2, 2, "cuda"
    )
    torch.testing.assert_close(
        counts, torch.tensor([[[2, 0], [0, 0]], [[0, 0], [0, 1]]], device="cuda")
    )


@cuda
@pytest.mark.parametrize("sampled", [False, True])
def test_factory_reads_leaf_before_completion_and_not_intermediate_ranges(sampled):
    owners = SimpleNamespace(
        slab=lambda name: SimpleNamespace(
            columns={
                "range_weights": weights().unsqueeze(0),
                "leaf_public_state": board(2).unsqueeze(0),
            }
        )
    )
    event = torch.cuda.Event()
    event.record()
    state = SimpleNamespace(
        last_event=event,
        input_names=("pre", "post_2"),
        topology=SimpleNamespace(
            topologies=(topology(),), leaf_counts=(2,), bindings=(bindings(),)
        ),
    )
    called = []
    engine = SimpleNamespace(
        _execute_completion=lambda *a: called.append("completion"),
        _prepare_learner=lambda *a: "prepared",
    )
    session = SimpleNamespace(
        device=torch.device("cuda:0"),
        geometry=SimpleNamespace(players=2, rank_count=1),
        mode=SimpleNamespace(
            rollout_mode=RolloutMode.SEQUENTIAL,
            training_regime=SimpleNamespace(value="self_play"),
            private_estimator=PrivateEstimator.SAMPLED
            if sampled
            else PrivateEstimator.RAO_BLACKWELL,
        ),
        _iteration_engine=lambda: engine,
    )
    observations = []
    exposure.install_posterior_exposure_observer(session, observations)
    engine = session._iteration_engine()
    engine._execute_completion(owners, state)
    assert called == ["completion"]
    assert engine._prepare_learner(owners, state, 0) == "prepared"
    record = observations[0]
    assert record["fresh_player_decisions"] == 3
    assert record["sum_posterior_exp_entropy"] == pytest.approx(3 if sampled else 5)
    assert record["posterior_semantics"] == exposure.POSTERIOR_SEMANTICS
    assert record["completion_histogram_used"] is False
    assert record["completion_effective_hands_used"] is False


@cuda
def test_chunked_paired_statistics_match_dense_reference_across_chunk_boundary():
    generator = torch.Generator(device="cuda").manual_seed(902)
    rb = torch.randn((4, 65539), generator=generator, device="cuda")
    normal = 2 * rb + 0.1 * torch.randn(rb.shape, generator=generator, device="cuda")
    expected = (
        normal.double().var(0, unbiased=True).sum()
        / rb.double().var(0, unbiased=True).sum()
    )
    expected_noise = (normal.double() - rb.double()).var(0, unbiased=True).sum() / 4
    actual = paired_comparison(normal, rb, repetitions=100, seed=22)
    assert actual["variance_ratio_normal_over_rb"] == pytest.approx(
        float(expected), rel=1e-10
    )
    assert actual["paired_estimated_noise_of_mean_difference"] == pytest.approx(
        float(expected_noise), rel=1e-10
    )


def test_entropy_and_reference_clocks_can_be_changed_independently():
    from pathlib import Path

    control = yaml.safe_load(
        Path("configs/profiles/nlhe_hu_32bb_shared_loss_search_v4.yaml").read_text()
    )["control"]
    control["entropy_clock"] = "posterior_effective_hands"
    validate_control(control)
    clock = ExposureClock()
    due = clock.observe(
        100,
        128,
        1,
        400000,
        effective_hands=500000,
        schedule_clock=control["reference_clock"],
    )
    assert due is False
    assert clock.entropy_coefficient(control) == 0
    control["entropy_clock"] = "global_player_decisions"
    control["reference_clock"] = "posterior_effective_hands"
    assert clock.entropy_coefficient(control) > 0
    assert (
        clock.observe(
            100,
            128,
            1,
            400000,
            effective_hands=500000,
            schedule_clock=control["reference_clock"],
        )
        is True
    )


def test_v4_registers_shared_workloads_and_all_loss_clock_combinations(monkeypatch):
    from pathlib import Path

    from experiments.hyperparameter_search.league_campaign import (
        league_budget,
        validate_profile,
    )
    from experiments.hyperparameter_search.protocol import resolve_spec

    profile = validate_profile(
        Path("configs/profiles/nlhe_hu_32bb_shared_loss_search_v4.yaml")
    )
    assert len(profile["workload_arms"]) == 4
    assert len(profile["regularizer_arms"]) == 21
    assert (
        league_budget(profile, smoke=False, scale=1)["nominal_training_wall_seconds"]
        == 156 * 3600
    )
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_RANK", "0")
    clocks = {}
    for workload in profile["workload_arms"]:
        for arm in profile["regularizer_arms"]:
            for estimator in ("sampled", "rao_blackwell"):
                spec = {
                    "kind": "self_play",
                    "profile": profile,
                    "seed": 123,
                    "private_estimator": estimator,
                    "initialization": "model",
                    "output_dir": "/tmp/v4-config-check",
                    "overrides": {**workload["overrides"], **arm["overrides"]},
                }
                config, control = resolve_spec(spec, 0, 3)
                assert config["run"]["execution"]["distributed"] is True
                clocks[(arm["name"], estimator)] = (
                    control["entropy_clock"],
                    control["reference_clock"],
                )
    assert (
        clocks[("posterior_entropy_only", "sampled")]
        == clocks[("posterior_entropy_only", "rao_blackwell")]
    )
    assert clocks[("posterior_entropy_only", "sampled")] == (
        "posterior_effective_hands",
        "global_player_decisions",
    )
    assert clocks[("posterior_reference_only", "sampled")] == (
        "global_player_decisions",
        "posterior_effective_hands",
    )
