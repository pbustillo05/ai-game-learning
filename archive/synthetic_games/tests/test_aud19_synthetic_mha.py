from __future__ import annotations

import torch

from pbg.core.config import load_config
from pbg.core.training import build_game_bundle
from pbg.games.synthetic.model import SyntheticMHANet, SyntheticNet


def test_synthetic_poker_defaults_to_reach_conditioned_mha() -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert isinstance(bundle.model, SyntheticMHANet)
    model = bundle.model
    assert model.uses_reach_context
    assert len(model.layers) == 2
    torch.testing.assert_close(
        model.card_incidence.sum(1), torch.ones(model.x_max)
    )


def test_mha_reach_bias_and_fold_summary_change_type_values() -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    model = bundle.model
    assert isinstance(model, SyntheticMHANet)
    with torch.no_grad():
        model.value_head.weight.fill_(0.1)
    keys = torch.tensor([0], dtype=torch.long)
    legal = torch.tensor([[False, True, True]])
    reach = torch.zeros((1, 2, 3))
    actor = torch.tensor([0])
    live = torch.ones((1, 2), dtype=torch.bool)
    uniform = model(keys, legal, reach=reach, actor=actor, live=live).v
    skewed_reach = reach.clone()
    skewed_reach[0, 1] = torch.tensor([-8.0, 0.0, -8.0])
    skewed = model(keys, legal, reach=skewed_reach, actor=actor, live=live).v
    folded_live = live.clone()
    folded_live[0, 1] = False
    folded = model(keys, legal, reach=skewed_reach, actor=actor, live=folded_live).v
    assert not torch.allclose(uniform, skewed)
    assert not torch.allclose(skewed, folded)


def test_mlp_is_available_only_by_explicit_baseline_selection() -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.model["architecture"] = "mlp_baseline"
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert isinstance(bundle.model, SyntheticNet)


def test_duplicate_public_states_are_evaluated_once_and_expanded_exactly() -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    backend = bundle.rollout_backend
    roots = backend.root_states(8)
    reach = torch.zeros((8, 2, 3))
    batched = backend.policy_forward(bundle.model, roots, reach)
    single = backend.policy_forward(bundle.model, roots[:1], reach[:1])
    torch.testing.assert_close(batched.raw, single.raw.expand_as(batched.raw))
    torch.testing.assert_close(batched.v, single.v.expand_as(batched.v))
