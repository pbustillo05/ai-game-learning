from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pbg.core.config import load_config
from pbg.core.game import CHANCE, TERMINAL
from pbg.core.training import (
    _build_iteration_data,
    _loss_spec,
    build_game_bundle,
    train,
)
from pbg.games.synthetic import exact as exact_module
from pbg.games.synthetic.bundle import (
    SyntheticExactCompletion,
    SyntheticMCCompletion,
)
from pbg.games.synthetic.exact import (
    FeasibilityError,
    enumerate_public_tree,
    exact_completion_stats,
)
from pbg.games.synthetic.family import SyntheticPBG
from pbg.games.synthetic.model import export_policy_tables
from pbg.games.synthetic.presets import kuhn


def _terminal_ref(game: SyntheticPBG) -> int:
    frontier = [int(game.root_states(1)[0])]
    while frontier:
        ref = frontier.pop(0)
        if int(game.owner_of(torch.tensor([ref], dtype=torch.int32))[0]) == TERMINAL:
            return ref
        event_ids, event_mask = game.legal_events(torch.tensor([ref], dtype=torch.int32))
        for event in event_ids[0, event_mask[0]]:
            frontier.append(
                int(
                    game.step(
                        torch.tensor([ref], dtype=torch.int32),
                        event.reshape(1).to(torch.int32),
                    )[0]
                )
            )
    raise AssertionError("synthetic game has no terminal state")


def test_exact_completion_enumerates_the_collision_conditioned_posterior() -> None:
    game = SyntheticPBG(kuhn())
    ref = _terminal_ref(game)
    reach = torch.log(
        torch.tensor(
            [[[0.15, 0.35, 0.50], [0.60, 0.25, 0.15]]], dtype=torch.float64
        )
    )
    tree = SimpleNamespace(state_ref=torch.tensor([ref], dtype=torch.int32))
    exact = SyntheticExactCompletion(game).complete(
        None,
        tree,
        torch.tensor([0], dtype=torch.int32),
        reach,
        n_samples=1,
        gen=torch.Generator().manual_seed(1),
    )

    profiles = [
        profile
        for profile in itertools.product(range(game.n_types), repeat=game.n_players)
        if game._profile_feasible(profile, set())
    ]
    profile_tensor = torch.tensor(profiles, dtype=torch.int32)
    weights = torch.tensor(
        [
            torch.exp(reach[0, 0, first] + reach[0, 1, second])
            for first, second in profiles
        ],
        dtype=torch.float64,
    )
    weights /= weights.sum()
    rewards = game.reward(
        profile_tensor,
        torch.full((len(profiles),), ref, dtype=torch.int32),
    ).to(torch.float64)
    for player in range(game.n_players):
        for xtype in range(game.n_types):
            selected = profile_tensor[:, player] == xtype
            mass = weights[selected].sum()
            assert exact.p_hat[0, player, xtype] == pytest.approx(float(mass))
            expected = (weights[selected] * rewards[selected, player]).sum() / mass
            assert exact.v_hat[0, player, xtype] == pytest.approx(float(expected))
    assert exact.metadata["backend"] == "exact"
    assert exact.metadata["node_ids"].tolist() == [0]


def test_exact_completion_is_seed_and_sample_budget_independent() -> None:
    game = SyntheticPBG(kuhn())
    ref = _terminal_ref(game)
    refs = torch.tensor([ref], dtype=torch.int32)
    reach = torch.zeros((1, game.n_players, game.n_types))
    backend = SyntheticExactCompletion(game)
    tree = SimpleNamespace(state_ref=refs)
    left = backend.complete(
        None, tree, torch.tensor([0]), reach, 1, torch.Generator().manual_seed(2)
    )
    right = backend.complete(
        None, tree, torch.tensor([0]), reach, 10_000, torch.Generator().manual_seed(3)
    )
    assert torch.equal(left.p_hat, right.p_hat)
    assert torch.equal(left.v_hat, right.v_hat)


def test_exact_completion_fails_before_infeasible_enumeration(monkeypatch) -> None:
    game = SyntheticPBG(kuhn())
    ref = _terminal_ref(game)
    monkeypatch.setattr(exact_module, "MAX_EXACT_NODES", 8)
    with pytest.raises(FeasibilityError, match="private profile product"):
        exact_completion_stats(
            game,
            torch.tensor([ref], dtype=torch.int32),
            torch.zeros((1, game.n_players, game.n_types)),
        )


def test_synthetic_backend_dispatch_keeps_exact_and_mc_distinct() -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    exact_bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert isinstance(exact_bundle.completion_backend, SyntheticExactCompletion)
    cfg.completion.backend = "mc"
    mc_bundle = build_game_bundle(cfg, torch.device("cpu"))
    assert isinstance(mc_bundle.completion_backend, SyntheticMCCompletion)


def _exact_kuhn_root_gradient(bundle) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Enumerate the detached-target policy gradient at Kuhn's root logits."""

    game = bundle.rollout_backend.game
    index = enumerate_public_tree(game)
    tables = export_policy_tables(bundle.model, game, index)
    root = index.nodes[0]
    root_key = int(index.node_keys[0])
    policy = tables[root_key]
    profiles = [
        profile
        for profile in itertools.product(range(game.n_types), repeat=game.n_players)
        if game._profile_feasible(profile, set())
    ]

    def continuation(ref: int, profile: tuple[int, ...]) -> float:
        owner = int(game.owner_of(torch.tensor([ref], dtype=torch.int32))[0])
        profile_row = torch.tensor([profile], dtype=torch.int32)
        if owner == TERMINAL:
            return float(
                game.reward(profile_row, torch.tensor([ref], dtype=torch.int32))[0, 0]
            )
        event_ids, event_mask = game.legal_events(
            torch.tensor([ref], dtype=torch.int32)
        )
        legal = event_ids[0, event_mask[0]].tolist()
        if owner == CHANCE:
            probabilities = game.chance_probs(
                torch.tensor([ref], dtype=torch.int32), profile_row
            )[0]
        else:
            key = int(game.encode_public(torch.tensor([ref], dtype=torch.int32))[0])
            probabilities = tables[key][profile[owner]]
        result = 0.0
        for column, event in enumerate(legal):
            probability = float(
                probabilities[column] if owner == CHANCE else probabilities[event]
            )
            child = int(
                game.step(
                    torch.tensor([ref], dtype=torch.int32),
                    torch.tensor([event], dtype=torch.int32),
                )[0]
            )
            result += probability * continuation(child, profile)
        return result

    event_ids, event_mask = game.legal_events(torch.tensor([root], dtype=torch.int32))
    legal = event_ids[0, event_mask[0]].tolist()
    q_value = np.zeros((game.n_types, bundle.model.e_max), dtype=np.float64)
    type_count = np.zeros(game.n_types, dtype=np.float64)
    for profile in profiles:
        xtype = profile[0]
        type_count[xtype] += 1.0
        for event in legal:
            child = int(
                game.step(
                    torch.tensor([root], dtype=torch.int32),
                    torch.tensor([event], dtype=torch.int32),
                )[0]
            )
            q_value[xtype, event] += continuation(child, profile)
    q_value /= type_count[:, None]
    type_probability = type_count / type_count.sum()
    gradient = np.zeros_like(q_value)
    for xtype in range(game.n_types):
        baseline = sum(policy[xtype, event] * q_value[xtype, event] for event in legal)
        for event in legal:
            gradient[xtype, event] = (
                -type_probability[xtype]
                * policy[xtype, event]
                * (q_value[xtype, event] - baseline)
            )
    return (
        torch.from_numpy(gradient).float(),
        torch.from_numpy(policy).float(),
        root,
    )


@pytest.mark.parametrize(
    ("mode", "tolerance"),
    [
        ("normal", 0.02),
        ("rao_blackwell", 0.008),
        ("branched_rao_blackwell", 0.008),
    ],
)
def test_all_modes_match_the_exact_enumerated_kuhn_policy_gradient(
    mode: str, tolerance: float
) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.training.mode = mode
    cfg.rollout.n_trees = 2_000 if mode != "branched_rao_blackwell" else 1
    cfg.loss.lam = 1.0
    cfg.loss.c_v = 0.0
    cfg.loss.c_h = 0.0
    cfg.completion.n_samples = 1
    if mode == "normal":
        cfg.loss.name = "ppo"
        cfg.rollout.beta = None
        cfg.completion.n_samples = None
    elif mode == "rao_blackwell":
        cfg.rollout.beta = None
    else:
        cfg.rollout.beta = {"default": 1.0}

    torch.manual_seed(77)
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    exact_gradient, policy, root = _exact_kuhn_root_gradient(bundle)
    data = _build_iteration_data(
        cfg,
        bundle,
        torch.Generator().manual_seed(88),
        _loss_spec(cfg),
    )
    estimate = torch.zeros_like(exact_gradient)
    if mode == "normal":
        selected = data.dataset.state_refs == root
        rows = zip(
            data.dataset.actor_type[selected],
            data.dataset.event[selected],
            data.dataset.advantage[selected],
            strict=True,
        )
        for xtype_tensor, event_tensor, advantage in rows:
            xtype, event = int(xtype_tensor), int(event_tensor)
            score = -policy[xtype].clone()
            score[event] += 1.0
            estimate[xtype] -= advantage * score
        estimate /= int(selected.sum())
    else:
        samples = [data]
        if mode == "branched_rao_blackwell":
            samples.extend(
                _build_iteration_data(
                    cfg,
                    bundle,
                    torch.Generator().manual_seed(89 + iteration),
                    _loss_spec(cfg),
                )
                for iteration in range(299)
            )
        for sample in samples:
            selected = (sample.decision_reach.state_refs == root).nonzero(
                as_tuple=False
            ).squeeze(1)
            sample_estimate = torch.zeros_like(exact_gradient)
            for row in selected:
                count = int(sample.targets.child_count[row])
                for slot in sample.targets.child_mask[row].nonzero(
                    as_tuple=False
                ).squeeze(1):
                    event = int(sample.targets.child_event_row[row, slot])
                    score = -policy.clone()
                    score[:, event] += 1.0
                    sample_estimate -= (
                        sample.targets.child_p_hat[row, slot, :, None]
                        * sample.targets.a_hat[row, slot, :, None]
                        * score
                        / count
                    )
            estimate += sample_estimate / selected.numel()
        estimate /= len(samples)

    torch.testing.assert_close(estimate, exact_gradient, atol=tolerance, rtol=0.0)


@pytest.mark.parametrize("config_path", ["configs/kuhn_smoke.yaml", "configs/leduc.yaml"])
@pytest.mark.parametrize(
    ("mode", "loss_name"),
    [
        ("normal", "ppo"),
        ("rao_blackwell", "rb_ppo"),
        ("branched_rao_blackwell", "rb_ppo"),
    ],
)
def test_all_training_modes_cross_the_exact_small_game_gate(
    config_path: str, mode: str, loss_name: str, tmp_path
) -> None:
    cfg = load_config(config_path)
    cfg.training.mode = mode
    cfg.loss.name = loss_name
    cfg.loss.c_h = 0.0
    cfg.model["hidden_dim"] = 8
    cfg.rollout.n_trees = 1 if mode == "branched_rao_blackwell" else 2
    cfg.train.iters = 1
    cfg.train.minibatch_size = 2
    cfg.train.out_dir = str(tmp_path / config_path.replace("/", "_") / mode)
    if mode == "normal":
        cfg.rollout.beta = None
        cfg.completion.n_samples = None
    elif mode == "rao_blackwell":
        cfg.rollout.beta = None
        cfg.completion.n_samples = 1
        cfg.completion.backend = "exact"
    else:
        cfg.rollout.beta = {"default": 1.0}
        cfg.completion.n_samples = 1
        cfg.completion.backend = "exact"
    train(cfg)
    assert (tmp_path / config_path.replace("/", "_") / mode / "metrics.jsonl").exists()
