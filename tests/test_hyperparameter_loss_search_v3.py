"""Scientific protocol checks for the diagnosis-led profile revision."""

from pathlib import Path

import pytest
import torch

from experiments.hyperparameter_search.league_campaign import (
    league_budget,
    validate_profile,
)
from experiments.hyperparameter_search.protocol import resolve_spec
from pbg.config import load_config_mapping

PROFILE = Path("configs/profiles/nlhe_hu_32bb_shared_loss_search_v3.yaml")


def test_workloads_keep_presentations_per_root_fixed():
    p = validate_profile(PROFILE)
    triples = [
        (
            a["overrides"]["run.workload.roots"],
            a["overrides"]["scientific.schedule.batch_size"],
            a["overrides"]["scientific.schedule.steps"],
        )
        for a in p["workload_arms"]
    ]
    assert set(triples) == {
        (512, 128, 24),
        (1024, 128, 48),
        (1024, 256, 24),
        (2048, 256, 48),
    }
    assert all(b * s / n == 6 for n, b, s in triples)
    # Same data and number of presentations, different batch versus step count.
    assert 128 * 48 == 256 * 24


def test_loss_search_remains_primary_and_retains_the_prior_axes():
    p = validate_profile(PROFILE)
    names = {a["name"] for a in p["regularizer_arms"]}
    assert len(names) == 19
    assert {
        "entropy_zero",
        "entropy_half",
        "entropy_double",
        "entropy_fast",
        "entropy_slow",
        "kl_weak",
        "kl_strong",
        "reference_fast",
        "reference_fixed",
        "learning_rate_slow",
        "learning_rate_fast",
        "critic_weight_quarter",
        "critic_weight_4x",
        "critic_unclipped",
        "critic_clip_wide",
        "ppo_clip_tight",
        "lambda_09",
        "posterior_clock_raw",
        "control",
    } == names
    b = league_budget(p, smoke=False, scale=1)
    assert b["nominal_training_wall_seconds"] == 148 * 3600
    assert b["stages"]["workload"]["training_wall_seconds_upper_bound"] == 16 * 3600
    assert b["stages"]["regularizer"]["training_wall_seconds_upper_bound"] == 76 * 3600
    assert 76 / (76 + 16) > 0.8


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Capacity compilation requires CUDA"
)
def test_every_workload_loss_pair_resolves_for_both_modes_without_scientific_drift(
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "3")
    p = validate_profile(PROFILE)
    for workload in p["workload_arms"]:
        for arm in p["regularizer_arms"]:
            overrides = {**workload["overrides"], **arm["overrides"]}
            configs = []
            controls = []
            for mode in ("sampled", "rao_blackwell"):
                spec = {
                    "kind": "self_play",
                    "profile": p,
                    "overrides": overrides,
                    "private_estimator": mode,
                    "seed": 918273,
                    "smoke": False,
                }
                source, control = resolve_spec(spec, 0, 3)
                load_config_mapping(source)
                scientific = source["scientific"]
                scientific["mode"].pop("private_estimator")
                scientific.pop("completion", None)
                configs.append(scientific)
                controls.append(control)
            assert configs[0] == configs[1]
            assert controls[0] == controls[1]
