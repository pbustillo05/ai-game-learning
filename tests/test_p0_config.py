from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pbg.config import ConfigError, load_config
from pbg.core.config import seed_all
from pbg.core.logging_util import MetricsLogger
from pbg.core.training import _configure_reproducibility


ROOT = Path(__file__).parents[1]


def test_nlhe_loads_every_config_field() -> None:
    cfg = load_config(ROOT / "configs" / "train" / "nlhe6.yaml")
    assert cfg.game.n_players == 6
    assert cfg.game.stack_bb == 100.0
    assert cfg.game.rank_count == 13
    assert cfg.rollout.beta is not None
    assert cfg.rollout.beta["preflop"] == 1.0
    assert cfg.completion.n_samples == 256
    assert cfg.completion.backend == "cuda"
    assert cfg.model["d_model"] == 192
    assert cfg.loss.lam == pytest.approx(0.95)
    assert cfg.loss.value_loss == "clipped"
    assert cfg.training.adam_betas == (0.9, 0.999)
    assert cfg.training.reproducibility == "throughput"


def test_hardware_profile_is_composed_without_changing_scientific_config() -> None:
    cfg = load_config(
        ROOT / "configs" / "train" / "nlhe6.yaml",
        hardware_profile="local_3x2080ti",
    )
    assert cfg.hardware is not None
    assert cfg.hardware.name == "local_3x2080ti"
    assert cfg.hardware.expected_compute_capability == (7, 5)
    assert cfg.hardware.training_max_tokens == 8192
    assert cfg.training.minibatch_size == 128


def test_unknown_nested_key_reports_full_path(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["rollout"]["surprise"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"root\.rollout\.surprise"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "pattern"),
    [
        ("game", "mispelled_ante", r"root\.game\.mispelled_ante"),
        ("model", "e_max", r"unknown poker model configuration key"),
    ],
)
def test_game_owned_mappings_reject_unknown_keys(
    tmp_path: Path, section: str, key: str, pattern: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    target = raw
    for component in section.split("."):
        target = target[component]
    target[key] = 3
    path = tmp_path / "bad_owned.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=pattern):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "pattern"),
    [
        ("game", "stacks_bb", r"root\.game\.stacks_bb"),
        ("model", "unknown_head", r"unknown poker model configuration key"),
    ],
)
def test_poker_owned_mappings_are_strict_at_top_level_load(
    tmp_path: Path, section: str, key: str, pattern: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    target = raw
    for component in section.split("."):
        target = target[component]
    target[key] = 3
    path = tmp_path / "bad_poker_owned.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=pattern):
        load_config(path)


@pytest.mark.parametrize("field", ["debug_asserts", "use_suit_consistency"])
def test_removed_noop_controls_are_rejected(tmp_path: Path, field: str) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    if field == "debug_asserts":
        raw[field] = True
    else:
        raw["loss"][field] = True
    path = tmp_path / "noop.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_training_mode_validates_beta_completion_and_loss_axes(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["training"]["mode"] = "normal"
    raw["completion"]["n_samples"] = None
    raw["loss"]["name"] = "ppo"
    raw["rollout"]["beta"] = None
    raw["rollout"]["branching_mode"] = None
    path = tmp_path / "normal.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.training.mode == "normal"
    assert cfg.rollout.beta is None

    raw["rollout"]["beta"] = {"default": 2.0}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="only valid for branched"):
        load_config(path)

    raw["rollout"]["beta"] = None
    raw["loss"]["name"] = "rb_ppo"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="does not support mode 'normal'"):
        load_config(path)


@pytest.mark.parametrize(
    "key", ["executor", "trajectory_chunk_size", "queue_packing", "n_trees"]
)
def test_removed_rollout_implementation_keys_are_rejected(
    tmp_path: Path, key: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["rollout"][key] = "wavefront_v1" if key == "executor" else 4
    path = tmp_path / "removed.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_production_execution_config_is_explicit_and_strict(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["training"]["mode"] = "rao_blackwell"
    raw["rollout"]["beta"] = None
    raw["rollout"]["branching_mode"] = None
    raw["rollout"]["sequential_roots_per_gpu_initial"] = 10
    raw["rollout"].update(
        {
            "family_batch_rows": 8,
            "max_active_trajectories": 16,
            "arena_chunk_nodes": 32,
            "execution_memory_bytes": 1_000_000,
        }
    )
    path = tmp_path / "execution.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.rollout.family_batch_rows == 8
    assert cfg.rollout.max_active_trajectories == 16
    assert cfg.rollout.arena_chunk_nodes == 32
    assert cfg.rollout.execution_memory_bytes == 1_000_000

    raw["rollout"]["family_batch_rows"] = 0
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="family_batch_rows"):
        load_config(path)

    raw["rollout"]["family_batch_rows"] = 8
    raw["rollout"]["trajectory_start_index"] = 3
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(path)

def test_seed_all_returns_repeatable_explicit_generators() -> None:
    torch_a, numpy_a = seed_all(19, rank=2)
    torch_b, numpy_b = seed_all(19, rank=2)
    torch_other, numpy_other = seed_all(19, rank=3)
    draw_torch_a = torch.rand(8, generator=torch_a)
    draw_torch_b = torch.rand(8, generator=torch_b)
    draw_numpy_a = numpy_a.random(8)
    draw_numpy_b = numpy_b.random(8)
    assert torch.equal(draw_torch_a, draw_torch_b)
    assert np.array_equal(draw_numpy_a, draw_numpy_b)
    assert not torch.equal(draw_torch_a, torch.rand(8, generator=torch_other))
    assert not np.array_equal(draw_numpy_a, numpy_other.random(8))


def test_reproducibility_profiles_change_backend_controls() -> None:
    _configure_reproducibility("strict")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark

    _configure_reproducibility("seeded")
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark

    _configure_reproducibility("throughput")
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark
    _configure_reproducibility("strict")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "reproducibility", "sometimes"),
        ("loss", "value_loss", "silent_noop"),
    ],
)
def test_behavioral_profile_enums_are_strict(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw[section][field] = value
    path = tmp_path / "bad_profile.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_loss_unit_and_explicit_entropy_horizon_are_strict(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["loss"].update(
        {
            "utility_normalization": "stack",
            "entropy_schedule": "linear_decay",
            "entropy_decay_iters": 500,
        }
    )
    path = tmp_path / "normalized.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.loss.utility_normalization == "stack"
    assert cfg.loss.entropy_decay_iters == 500

    raw["loss"]["utility_normalization"] = "mystery"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="utility_normalization"):
        load_config(path)

    raw["loss"]["utility_normalization"] = "stack"
    raw["loss"]["entropy_decay_iters"] = 0
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="entropy_decay_iters"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field", "value", "pattern"),
    [
        ("rollout", "node_budget", 0, r"root\.rollout\.node_budget"),
        ("loss", "lam", -0.01, r"root\.loss\.lam"),
        ("loss", "lam", 1.01, r"root\.loss\.lam"),
        ("loss", "clip_eps", 0.0, r"root\.loss\.clip_eps"),
        ("loss", "clip_eps", 1.01, r"root\.loss\.clip_eps"),
        ("loss", "clip_eps_v", float("nan"), r"root\.loss\.clip_eps_v"),
        ("loss", "c_v", -0.01, r"root\.loss\.c_v"),
        ("loss", "c_h", float("inf"), r"root\.loss\.c_h"),
        ("loss", "c_kl", -0.01, r"root\.loss\.c_kl"),
        ("training", "iters", 0, r"root\.training\.iters"),
        ("training", "gd_steps_per_iter", 0, r"root\.training\.gd_steps_per_iter"),
        ("training", "minibatch_size", 0, r"root\.training\.minibatch_size"),
        ("training", "lr", -1.0, r"root\.training\.lr"),
        ("training", "grad_clip", 0.0, r"root\.training\.grad_clip"),
        ("training", "checkpoint_every", 0, r"root\.training\.checkpoint_every"),
        ("training", "seed", -1, r"root\.training\.seed"),
        ("training", "out_dir", "", r"root\.training\.out_dir"),
        ("model", "d_model", 0, r"root\.model\.d_model"),
        ("model", "s_min", float("nan"), r"root\.model\.s_min"),
        ("model", "folder_beta_rank", 53, r"root\.model\.folder_beta_rank"),
    ],
)
def test_all_scientific_numeric_boundaries_reject_with_dotted_paths(
    tmp_path: Path, section: str, field: str, value, pattern: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw[section][field] = value
    path = tmp_path / f"bad_{section}_{field}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=pattern):
        load_config(path)


@pytest.mark.parametrize(
    ("betas", "pattern"),
    [
        ((-0.01, 0.999), r"root\.training\.adam_betas\[0\]"),
        ((0.9, 1.0), r"root\.training\.adam_betas\[1\]"),
        ((0.9, float("nan")), r"root\.training\.adam_betas\[1\]"),
    ],
)
def test_adam_beta_boundaries_are_finite_and_dotted(
    tmp_path: Path, betas: tuple[float, float], pattern: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["training"]["adam_betas"] = list(betas)
    path = tmp_path / "bad_betas.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=pattern):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_count", -1),
        ("checkpoint_every", 0),
        ("reserve_vram_gib", float("nan")),
        ("max_memory_fraction", float("inf")),
        ("min_vram_gib", 0.0),
        ("max_vram_gib", -1.0),
    ],
)
def test_hardware_numeric_boundaries_reject_with_dotted_paths(
    tmp_path: Path, field: str, value
) -> None:
    profile = yaml.safe_load(
        (ROOT / "configs" / "hardware" / "local_3x2080ti.yaml").read_text()
    )
    profile[field] = value
    profile_path = tmp_path / "bad_hardware.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(ConfigError, match=rf"hardware\.{field}"):
        load_config(
            ROOT / "configs" / "train" / "nlhe6.yaml",
            hardware_profile=profile_path,
        )


def test_metrics_logger_appends_json_lines(tmp_path: Path) -> None:
    logger = MetricsLogger(tmp_path)
    logger.log(4, {"loss": 1.5})
    logger.log(5, {"loss": 1.0, "entropy": 0.25})
    logger.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {"step": 4, "loss": 1.5},
        {"step": 5, "loss": 1.0, "entropy": 0.25},
    ]
