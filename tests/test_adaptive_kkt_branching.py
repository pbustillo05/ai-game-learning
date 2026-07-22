from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
import numpy as np
from torch import Tensor, nn

from pbg.core.adaptive_branching import (
    AdaptiveKKTController,
    child_path_factors,
    gamma_deviance,
    log_kkt_intensity,
    randomized_round,
)
from pbg.core.branching import sample_branched_tree
from pbg.core.game import GameBundle, TERMINAL
from pbg.core.learner import mode_loss
from pbg.config import load_config, load_config_mapping, ConfigError
from pbg.core.checkpointing import load_checkpoint, save_checkpoint
from tests.categorical_policy_fixture import CategoricalPolicyAdapter
from tests.test_c09_labeled_branching import _Adapter, _LabeledBackend
from tests.test_p6_preflop import _inputs as _preflop_inputs
from tests.test_p7_postflop import _inputs as _postflop_inputs
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.heads import PublicBranchingHeads


def test_gamma_deviance_values_gradients_zero_target_and_arithmetic_mean() -> None:
    prediction = torch.tensor([2.0], requires_grad=True)
    target = torch.tensor([4.0])
    loss = gamma_deviance(prediction, target)
    assert loss.item() == pytest.approx(2.0 - math.log(2.0) - 1.0)
    loss.backward()
    assert prediction.grad.item() == pytest.approx((2.0 - 4.0) / 4.0)

    zero_prediction = torch.tensor([0.3], requires_grad=True)
    zero = gamma_deviance(zero_prediction, torch.zeros(1), shift=1.0e-3)
    assert torch.isfinite(zero).all()
    zero.backward()
    assert torch.isfinite(zero_prediction.grad).all()

    samples = torch.tensor([1.0, 3.0, 8.0])
    arithmetic_mean = samples.mean().requires_grad_()
    gamma_deviance(arithmetic_mean.expand_as(samples), samples).mean().backward()
    assert arithmetic_mean.grad.item() == pytest.approx(0.0, abs=1.0e-7)
    geometric_mean = samples.log().mean().exp().requires_grad_()
    gamma_deviance(geometric_mean.expand_as(samples), samples).mean().backward()
    assert abs(geometric_mean.grad.item()) > 1.0e-2


def test_preflop_and_postflop_emit_positive_public_t_and_c_heads() -> None:
    cfg = PokerModelConfig(
        d_model=16,
        n_heads=4,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=2,
        k_act=5,
        max_players=3,
        class_embedding_dim=4,
        ffn_hidden=24,
        use_branching_heads=True,
    )
    pre_pub, pre_ranges = _preflop_inputs(players=3)
    pre = PreflopNet(cfg)(pre_pub, pre_ranges)
    assert pre.t is not None and pre.t.shape == (2,)
    assert pre.c is not None and pre.c.shape == (2,)
    assert bool(((pre.t > 0) & (pre.t < 1)).all()) and bool((pre.c > 0).all())

    post_cfg = PokerModelConfig(
        d_model=16,
        n_heads=4,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=2,
        k_act=5,
        max_players=2,
        class_embedding_dim=4,
        ffn_hidden=24,
        use_branching_heads=True,
    )
    post_pub, post_ranges, masks = _postflop_inputs()
    post = PostflopNet(post_cfg)(post_pub, post_ranges, masks=masks)
    assert post.t is not None and post.t.shape == (2,)
    assert post.c is not None and post.c.shape == (2,)
    assert bool(((post.t > 0) & (post.t < 1)).all()) and bool((post.c > 0).all())


def test_legacy_vu_head_config_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="V/U heads were removed"):
        PokerModelConfig.from_dict({"use_moment_heads": True})


def test_branching_modes_are_explicit_typed_and_fail_early() -> None:
    raw = yaml.safe_load(Path("configs/train/nlhe6.yaml").read_text(encoding="utf-8"))
    fixed = load_config_mapping(copy.deepcopy(raw))
    assert fixed.rollout.branching_mode == "fixed"
    assert fixed.rollout.adaptive_kkt is None

    bad = copy.deepcopy(raw)
    bad["rollout"].pop("branching_mode")
    with pytest.raises(ConfigError, match="branching_mode.*explicit"):
        load_config_mapping(bad)

    bad = copy.deepcopy(raw)
    bad["training"]["mode"] = "rao_blackwell"
    bad["rollout"]["beta"] = None
    with pytest.raises(ConfigError, match="branching_mode.*only valid"):
        load_config_mapping(bad)

    raw["rollout"]["branching_mode"] = "dynamic"
    dynamic_section = {
        "beta_max": 4.0,
        "warmup_iters": 10,
        "target_tree_size": 3000.0,
        "safe_tree_size": 3800.0,
        "kappa_initial": 1.0,
        "kappa_min": 1.0e-4,
        "kappa_max": 1.0e4,
        "gamma_kappa": 0.1,
        "eta_kappa": 0.05,
    }
    raw["rollout"]["adaptive_kkt"] = dynamic_section
    cfg = load_config_mapping(raw)
    assert cfg.rollout.branching_mode == "dynamic"
    assert cfg.rollout.adaptive_kkt is not None
    assert cfg.rollout.adaptive_kkt.warmup_iters == 10

    bad = copy.deepcopy(raw)
    bad["rollout"]["adaptive_kkt"]["enabled"] = True
    with pytest.raises(ConfigError, match="adaptive_kkt.enabled.*unknown"):
        load_config_mapping(bad)

    bad = copy.deepcopy(raw)
    bad["rollout"]["branching_mode"] = "fixed"
    with pytest.raises(ConfigError, match="adaptive_kkt.*only valid"):
        load_config_mapping(bad)

    bad = copy.deepcopy(raw)
    bad["rollout"].pop("adaptive_kkt")
    with pytest.raises(ConfigError, match="adaptive_kkt.*required"):
        load_config_mapping(bad)

    bad = copy.deepcopy(raw)
    bad["rollout"]["branching_mode"] = "automatic"
    with pytest.raises(ConfigError, match="branching_mode"):
        load_config_mapping(bad)

    for mode in ("fixed", "dynamic"):
        bad = copy.deepcopy(raw)
        bad["rollout"]["branching_mode"] = mode
        if mode == "fixed":
            bad["rollout"].pop("adaptive_kkt")
        bad["model"]["use_branching_heads"] = False
        with pytest.raises(ConfigError, match="use_branching_heads"):
            load_config_mapping(bad)

    for coefficient in ("c_t", "c_c"):
        bad = copy.deepcopy(raw)
        bad["loss"][coefficient] = 0.0
        with pytest.raises(ConfigError, match="c_t and root.loss.c_c"):
            load_config_mapping(bad)


def test_log_domain_kkt_root_ancestor_factors_and_clipping() -> None:
    t = torch.tensor([0.25])
    cost = torch.tensor([1.0])
    root = log_kkt_intensity(
        t, cost, math.log(16.0), torch.zeros(1), torch.zeros(1), 10.0
    )
    assert root.item() == pytest.approx(2.0)

    log_sum, log_product = child_path_factors(
        torch.zeros(1), torch.zeros(1), root, torch.ones(1, dtype=torch.bool)
    )
    assert log_sum.exp().item() == pytest.approx(1.5)
    assert log_product.exp().item() == pytest.approx(2.0)
    child = log_kkt_intensity(t, cost, math.log(16.0), log_sum, log_product, 10.0)
    assert child.item() == pytest.approx(math.sqrt(3.0))
    clipped = log_kkt_intensity(t, cost, math.log(1.0e20), log_sum, log_product, 3.0)
    assert clipped.item() == 3.0


def test_randomized_rounding_has_configured_expectation() -> None:
    beta = torch.full((100_000,), 2.3)
    uniform = (torch.arange(beta.numel(), dtype=torch.float32) + 0.5) / beta.numel()
    rounded = randomized_round(beta, uniform)
    assert rounded.min() == 2 and rounded.max() == 3
    assert rounded.float().mean().item() == pytest.approx(2.3, abs=1.0e-5)


def test_damped_controller_update_saturation_and_persistence() -> None:
    controller = AdaptiveKKTController(
        target_tree_size=100.0,
        gamma_kappa=0.5,
        eta_kappa=0.25,
        kappa_min=0.5,
        kappa_max=2.0,
        log_kappa=0.0,
    )
    first = controller.observe(25.0)
    assert first["kkt_ema_tree_size"] == 25.0
    assert controller.log_kappa == pytest.approx(0.25 * math.log(4.0))
    controller.observe(1.0e-6)
    controller.observe(1.0e-6)
    assert controller.log_kappa == pytest.approx(math.log(2.0))
    restored = AdaptiveKKTController.from_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()


def test_controller_state_round_trips_through_training_checkpoint(tmp_path) -> None:
    cfg = load_config("configs/train/nlhe6.yaml")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    controller = AdaptiveKKTController(
        target_tree_size=100.0,
        gamma_kappa=0.2,
        eta_kappa=0.1,
        kappa_min=0.1,
        kappa_max=10.0,
        log_kappa=0.0,
    )
    controller.observe(80.0)
    path = tmp_path / "controller.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        4,
        cfg,
        torch.Generator().manual_seed(3),
        np.random.default_rng(4),
        branching_controller_state=controller.state_dict(),
    )
    metadata = load_checkpoint(path, model)
    restored = AdaptiveKKTController.from_state_dict(
        metadata["branching_controller_state"]
    )
    assert restored.state_dict() == controller.state_dict()


@dataclass
class _AdaptiveOutput:
    raw: Tensor
    v: Tensor
    t: Tensor
    c: Tensor


class _AdaptiveModel(nn.Module):
    def forward(self, states: Tensor) -> _AdaptiveOutput:
        batch = states.numel()
        value = states.float().reshape(batch, 1)
        cost = torch.where(states == 0, 10.0, 0.1).float()
        return _AdaptiveOutput(
            raw=torch.zeros((batch, 1, 2)),
            v=value,
            t=torch.full((batch,), 0.5),
            c=cost,
        )


def test_fixed_and_dynamic_warmup_ignore_heads_then_dynamic_uses_them() -> None:
    backend = _LabeledBackend(1.0, two_levels=True)
    bundle = GameBundle(_AdaptiveModel(), backend, _Adapter(), None)
    fixed_cfg = SimpleNamespace(
        roots_per_gpu=1, storage_page_rows=32, branching_mode="fixed", adaptive_kkt=None
    )
    dynamic = SimpleNamespace(
        warmup_iters=3, beta_max=3.0, safe_tree_size=1000.0
    )
    dynamic_cfg = SimpleNamespace(
        roots_per_gpu=1, storage_page_rows=32, branching_mode="dynamic",
        adaptive_kkt=dynamic,
    )
    controller = AdaptiveKKTController(
        target_tree_size=20.0,
        gamma_kappa=1.0,
        eta_kappa=1.0,
        kappa_min=1.0e-3,
        kappa_max=1.0e6,
        log_kappa=math.log(1000.0),
    )

    fixed_sample = sample_branched_tree(
        bundle, fixed_cfg, torch.Generator().manual_seed(5), iteration=20,
        controller=controller,
    )
    warmup_sample = sample_branched_tree(
        bundle, dynamic_cfg, torch.Generator().manual_seed(5), iteration=2,
    )
    fixed_degree = fixed_sample.topology.degree
    warmup_degree = warmup_sample.topology.degree
    assert torch.equal(warmup_degree, fixed_degree)
    assert backend.actual_cost_batch_sizes[-2:] == [1, 1]

    active_sample = sample_branched_tree(
        bundle, dynamic_cfg, torch.Generator().manual_seed(5), iteration=3,
        controller=controller,
    )
    active_degree = active_sample.topology.degree
    assert active_degree[:4].tolist() == [3, 3, 3, 3]

    with pytest.raises(ValueError, match="requires controller state"):
        sample_branched_tree(
            bundle, dynamic_cfg, torch.Generator().manual_seed(5), iteration=3,
        )


def test_queue_safeguard_is_batch_local_recovers_and_retains_counterfactual_counts() -> None:
    backend = _LabeledBackend(1.0, two_levels=True)
    adaptive = SimpleNamespace(
        warmup_iters=0,
        beta_max=3.0,
        safe_tree_size=5.0,
    )
    cfg = SimpleNamespace(
        roots_per_gpu=1,
        storage_page_rows=32,
        branching_mode="dynamic",
        adaptive_kkt=adaptive,
    )
    controller = AdaptiveKKTController(
        target_tree_size=20.0,
        gamma_kappa=1.0,
        eta_kappa=1.0,
        kappa_min=1.0e-3,
        kappa_max=1.0e6,
        log_kappa=math.log(1000.0),
    )
    sample = sample_branched_tree(
        GameBundle(_AdaptiveModel(), backend, _Adapter(), None),
        cfg,
        torch.Generator().manual_seed(4),
        controller=controller,
    )
    degree = sample.topology.degree
    assert degree[:2].tolist() == [1, 3]
    assert sample.topology.nominal_degree[:2].tolist() == [3, 3]
    assert sample.branching_diagnostics["safeguard_batches"] == 1
    old_c = sample.decision_source["old_c"]
    assert old_c.tolist() == pytest.approx([10.0, 0.1])
    counterfactual = float(sample.node_count) + float(
        ((sample.decision_aux["nominal_child_count"].float()
          - sample.decision_aux["executed_child_count"].float()) * old_c).sum()
    )
    assert counterfactual == pytest.approx(float(sample.node_count) + 20.0)


@dataclass
class _LearnerOutput:
    raw: Tensor
    v: Tensor
    t: Tensor
    c: Tensor


def test_real_mode_loss_reaches_both_heads_and_keeps_targets_detached() -> None:
    adapter = CategoricalPolicyAdapter()
    t_parameter = nn.Parameter(torch.tensor([0.4]))
    c_log_parameter = nn.Parameter(torch.tensor([0.0]))
    output = _LearnerOutput(
        raw=torch.zeros((1, 1, 2), requires_grad=True),
        v=torch.zeros((1, 1), requires_grad=True),
        t=torch.sigmoid(t_parameter),
        c=torch.exp(c_log_parameter),
    )
    columns = {
        "legal_event_ids": torch.tensor([[0, 1]], dtype=torch.int32),
        "legal_event_mask": torch.ones((1, 2), dtype=torch.bool),
        "old_value": torch.zeros((1, 1)),
        "value_hat": torch.zeros((1, 1)),
        "trace_variance_hat": torch.tensor([0.8]),
        "trace_variance_mask": torch.tensor([True]),
        "cost_hat": torch.tensor([3.0]),
    }
    actions = {
        "action_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "event": torch.tensor([0], dtype=torch.int32),
        "old_log_prob": torch.tensor([[-math.log(2.0)]]),
        "child_posterior": torch.ones((1, 1)),
        "advantage_hat": torch.zeros((1, 1)),
    }
    partition = SimpleNamespace(columns=columns, action_columns=actions)
    spec = SimpleNamespace(phi=lambda device: torch.tensor([0], device=device))
    storage = SimpleNamespace(posterior_pool=None)
    cfg = SimpleNamespace(
        c_v=0.0,
        c_kl=0.0,
        clip_eps=0.2,
        c_h=0.0,
        entropy_schedule="const",
        value_loss="mse",
        clip_eps_v=0.2,
        epsilon_t=1.0e-6,
        c_t=2.0,
        c_c=3.0,
    )
    result = mode_loss(
        "branched_rb", spec, storage, partition, torch.tensor([0]),
        output, adapter, cfg,
    )
    result.per_decision.sum().backward()
    assert t_parameter.grad is not None and t_parameter.grad.abs().item() > 0
    assert c_log_parameter.grad is not None and c_log_parameter.grad.abs().item() > 0
    assert not columns["trace_variance_hat"].requires_grad
    assert not columns["cost_hat"].requires_grad
    assert result.stats["trace_variance_loss_sum"] > 0
    assert result.stats["cost_loss_sum"] > 0



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tiny_cuda_branching_heads_and_gamma_losses() -> None:
    device = torch.device("cuda:0")
    head = PublicBranchingHeads(8, 4, 8).to(device)
    public = torch.randn((2, 8), device=device, requires_grad=True)
    t, cost = head(public)
    target_t = torch.tensor([0.0, 0.7], device=device)
    target_c = torch.tensor([1.0, 3.0], device=device)
    loss = gamma_deviance(t, target_t, shift=1.0e-6).mean()
    loss = loss + gamma_deviance(cost, target_c).mean()
    loss.backward()
    torch.cuda.synchronize(device)
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert public.grad is not None and torch.isfinite(public.grad).all()
