from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from experiments.hyperparameter_search.gradient_noise import (
    capture_gradient,
    comparable_config,
    covariance_summary,
    immutable_json,
    paired_comparison,
    probe_config,
    require_cuda_device,
    validate_groups,
)

ROOT = Path(__file__).parents[1]


def make_spec():
    profile = yaml.safe_load(
        (
            ROOT / "configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml"
        ).read_text()
    )
    return {
        "kind": "self_play",
        "profile": profile,
        "arm": "control",
        "seed": 123,
        "private_estimator": "rao_blackwell",
        "initialization": "model",
        "gradient_noise": True,
        "smoke": True,
        "output_dir": "/tmp/gradient-noise-test",
        "overrides": {},
    }


def groups():
    return {
        "groups": [
            {
                "label": "normal",
                "private_estimator": "sampled",
                "roots": 64,
                "batch": 16,
                "artifacts": ["normal0.json", "normal1.json"],
            },
            {
                "label": "rb",
                "private_estimator": "rao_blackwell",
                "roots": 64,
                "batch": 16,
                "artifacts": ["rb0.json", "rb1.json"],
            },
        ]
    }


def test_probe_forces_one_pure_pg_step_after_smoke_resolution():
    spec = make_spec()
    old = copy.deepcopy(spec)
    config = probe_config(spec)
    assert spec == old
    assert config["scientific"]["schedule"] == {"batch_size": 16, "steps": 1}
    assert config["scientific"]["objective"]["value"]["coefficient"] == 0
    assert config["scientific"]["objective"]["entropy"] == {
        "kind": "constant",
        "coefficient": 0.0,
    }
    assert "reference_kl" not in config["scientific"]["objective"]
    assert config["run"]["execution"]["reference_cache"] == "off"
    assert config["run"]["workload"]["roots"] == 64


@pytest.mark.parametrize(
    "change",
    [
        {"gradient_noise": False},
        {"kind": "best_response"},
        {"initialization": "random"},
    ],
)
def test_probe_rejects_unsupported_initialization_or_role(change):
    spec = make_spec()
    spec.update(change)
    with pytest.raises(ValueError):
        probe_config(spec)


def test_immutable_diagnostic_metadata(tmp_path):
    path = tmp_path / "spec.json"
    immutable_json(path, {"seed": 1})
    immutable_json(path, {"seed": 1})
    with pytest.raises(RuntimeError, match="Immutable"):
        immutable_json(path, {"seed": 2})


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_only_explicit_single_cuda_device(device):
    with pytest.raises(ValueError):
        require_cuda_device(device)


def test_complete_paired_manifest_is_accepted():
    manifest = groups()
    assert validate_groups(manifest) == manifest["groups"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m["groups"].pop(),
        lambda m: m["groups"][1].update(label="normal"),
        lambda m: m["groups"][1].update(roots=128),
        lambda m: m["groups"][1].update(artifacts=["same", "same"]),
        lambda m: m["groups"][1].update(batch=True),
        lambda m: m["groups"][1].update(artifacts=["only-one"]),
    ],
)
def test_invalid_paired_manifest_rejected(mutation):
    manifest = groups()
    mutation(manifest)
    with pytest.raises(ValueError):
        validate_groups(manifest)


def test_statistics_reject_cpu_storage():
    # Construction is an orchestration contract check, not production statistics.
    with pytest.raises(ValueError, match="CUDA"):
        covariance_summary(torch.empty((2, 3)))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_cuda_welford_known_trace_and_mean():
    gradients = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda")
    mean, trace, summary = covariance_summary(gradients)
    torch.testing.assert_close(
        mean, torch.tensor([3.0, 4.0], device="cuda", dtype=torch.float64)
    )
    assert trace.item() == pytest.approx(8.0)
    assert summary["noise_trace_of_mean"] == pytest.approx(8 / 3)
    assert summary["single_replicate_signal_to_noise"] == pytest.approx(25 / 8)


@cuda
def test_cuda_paired_bootstrap_preserves_shared_draws_and_known_ratio():
    # Every nondegenerate paired bootstrap has a variance ratio of four.
    rb = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0], [-1.0, 3.0]], device="cuda")
    normal = 2 * rb + torch.tensor([10.0, -3.0], device="cuda")
    summary = paired_comparison(normal, rb, repetitions=1000, seed=42)
    assert summary["variance_ratio_normal_over_rb"] == pytest.approx(4)
    assert summary[
        "variance_ratio_paired_bootstrap_95_percentile_interval"
    ] == pytest.approx([4, 4])
    assert (
        summary["bootstrap_valid_replicates"]
        + summary["bootstrap_degenerate_replicates"]
        == 1000
    )
    assert summary["paired_estimated_noise_of_mean_difference"] > 0


@cuda
def test_cuda_identical_pairs_have_zero_mean_difference_noise():
    gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 2.0]], device="cuda")
    result = paired_comparison(gradients, gradients)
    assert result["mean_l2_difference"] == 0
    assert result["paired_estimated_noise_of_mean_difference"] == 0
    assert result["variance_ratio_normal_over_rb"] == pytest.approx(1)
    assert result["mean_cosine_similarity"] == pytest.approx(1)


@cuda
def test_cuda_zero_variance_is_reported_without_infinity():
    gradients = torch.ones((3, 4), device="cuda")
    result = paired_comparison(gradients, gradients)
    assert result["variance_ratio_normal_over_rb"] is None
    assert result["variance_ratio_paired_bootstrap_95_percentile_interval"] is None
    assert result["bootstrap_degenerate_replicates"] == 1000


@cuda
def test_capture_is_unscaled_clone_before_optimizer_step_and_refuses_reuse():
    from types import SimpleNamespace

    flat = torch.tensor([2048.0, -1024.0], device="cuda")
    calls = []

    def original():
        calls.append(1)
        flat.zero_()
        return "applied"

    optimizer = SimpleNamespace(
        step=original,
        gradients=SimpleNamespace(flat=flat),
        scaler=SimpleNamespace(enabled=True, scale=1024.0),
    )
    captured = []
    capture_gradient(optimizer, captured)
    assert optimizer.step() == "applied"
    torch.testing.assert_close(captured[0], torch.tensor([2.0, -1.0], device="cuda"))
    assert calls == [1]
    with pytest.raises(RuntimeError, match="multiple"):
        optimizer.step()


def test_comparison_config_ignores_only_mode_and_replicate_identity():
    normal = probe_config(make_spec())
    rb = copy.deepcopy(normal)
    normal["scientific"]["mode"]["private_estimator"] = "sampled"
    normal["scientific"].pop("completion", None)
    rb["run"]["seed"] = 999
    rb["run"]["output_dir"] = "/tmp/another-replicate"
    assert comparable_config(normal) == comparable_config(rb)
    rb["scientific"]["lambda"] = 0.1
    assert comparable_config(normal) != comparable_config(rb)
