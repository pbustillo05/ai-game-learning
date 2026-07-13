from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from pbg.core.branching import IntensityController, NodeMeta
from pbg.core.game import TERMINAL
from pbg.core.rollout import sample_trees
from pbg.games.synthetic.exact import enumerate_public_tree
from pbg.games.synthetic.family import SyntheticPBG
from pbg.games.synthetic.model import (
    PolicyOutput,
    SyntheticNet,
    SyntheticPolicyAdapter,
    export_policy_tables,
)
from pbg.games.synthetic.presets import kuhn


def _cfg(mode: str, n_trees: int, beta: float = 1.0, budget: int = 128):
    return SimpleNamespace(
        n_trees=n_trees,
        beta_mode=mode,
        beta_fix={"default": beta},
        beta_avg={"default": beta},
        beta_max={"default": max(beta, 1.0)},
        warmup_iters=0,
        node_budget=budget,
        reach_storage="recompute",
    )


def _sample(game, net, cfg, seed, intensity=None, np_gen=None):
    return sample_trees(
        game,
        net,
        SyntheticPolicyAdapter(),
        cfg,
        None,
        intensity,
        torch.Generator().manual_seed(seed),
        np.random.default_rng(seed) if np_gen is None else np_gen,
    )


def _terminal_probabilities(game, net):
    index = enumerate_public_tree(game)
    tables = export_policy_tables(net, game, index)
    profiles = np.asarray(
        [(a, b) for a in range(3) for b in range(3) if a != b], dtype=np.int64
    )
    expected: dict[int, float] = {}

    def walk(node: int, mass: np.ndarray) -> None:
        owner = int(index.owners[node])
        if owner == TERMINAL:
            expected[int(index.node_keys[node])] = float(mass.sum())
            return
        table = tables[int(index.node_keys[node])]
        for event, child in index.children[node]:
            walk(child, mass * table[profiles[:, owner], event])

    walk(0, np.full(6, 1.0 / 6.0, dtype=np.float64))
    return expected


def test_sequential_mode_is_byte_deterministic_for_five_seeds() -> None:
    torch.manual_seed(900)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(game.n_types, game.n_player_events, {"hidden_dim": 8})
    fields = (
        "parent", "child_ptr", "child_node", "child_event", "nu", "tree_id",
        "owner", "depth_p", "depth_player", "state_ref", "path_weight",
    )
    for seed in range(901, 906):
        left = _sample(game, net, _cfg("sequential", 64), seed)[0]
        right = _sample(game, net, _cfg("sequential", 64), seed)[0]
        for field in fields:
            assert torch.equal(getattr(left, field), getattr(right, field))


def test_fixed_beta_one_has_the_sequential_terminal_law() -> None:
    torch.manual_seed(906)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(
        game.n_types, game.n_player_events, {"n_buckets": 64, "hidden_dim": 8}
    )
    expected = _terminal_probabilities(game, net)
    tree, _, _ = _sample(game, net, _cfg("fixed", 50_000), 907)
    leaves = (tree.owner == TERMINAL).nonzero(as_tuple=False).squeeze(1)
    keys = game.encode_public(tree.state_ref[leaves]).numpy()
    observed = np.asarray([(keys == key).sum() for key in expected], dtype=np.float64)
    target = 50_000 * np.asarray(list(expected.values()))
    assert float((np.square(observed - target) / target).sum()) < 18.467
    assert torch.equal(tree.nu, torch.ones_like(tree.nu))


def test_nu_leaf_measure_matches_exact_public_law_at_beta_three() -> None:
    torch.manual_seed(908)
    game = SyntheticPBG(kuhn())
    net = SyntheticNet(
        game.n_types, game.n_player_events, {"n_buckets": 64, "hidden_dim": 8}
    )
    expected = _terminal_probabilities(game, net)
    n_trees = 20_000
    tree, _, _ = _sample(game, net, _cfg("fixed", n_trees, 3.0), 909)
    leaves = (tree.owner == TERMINAL).nonzero(as_tuple=False).squeeze(1)
    keys = game.encode_public(tree.state_ref[leaves]).numpy()
    weights = tree.path_weight[leaves]
    observed = np.asarray(
        [float(weights[torch.from_numpy(keys == key)].sum()) for key in expected]
    )
    target = n_trees * np.asarray(list(expected.values()))
    # df=4 chi-square 0.001 upper critical value.
    assert float((np.square(observed - target) / target).sum()) < 18.467
    assert abs(float(weights.sum()) - n_trees) < 0.01


@dataclass
class _TwoEventGame:
    n_players: int = 2
    x_sizes: tuple[int, int] = (1, 1)
    n_player_events: int = 2

    def root_states(self, n):
        return torch.zeros(n, dtype=torch.int32)

    def owner_of(self, refs):
        return torch.where(refs == 0, torch.zeros_like(refs), torch.full_like(refs, TERMINAL))

    def is_terminal(self, refs):
        return refs != 0

    def is_pre_terminal(self, refs):
        return torch.zeros_like(refs, dtype=torch.bool)

    def legal_events(self, refs):
        return torch.tensor([[0, 1]], dtype=torch.int32).repeat(len(refs), 1), torch.ones((len(refs), 2), dtype=torch.bool)

    def step(self, refs, events):
        return events.to(torch.int32) + 1

    def sample_profiles(self, refs, reach, n, gen):
        return torch.zeros((len(refs), n, 2), dtype=torch.int32)

    def encode_public(self, refs):
        return refs.long()


class _FairPolicy:
    e_max = 2

    def __call__(self, refs, mask):
        return PolicyOutput(
            raw=torch.zeros((len(refs), 1, 2)), v=torch.zeros((len(refs), 1))
        )


class _FixedPoisson:
    def __init__(self, value):
        self.value = value

    def poisson(self, rate):
        return self.value


def test_duplicate_events_collapse_and_retain_binomial_multiplicity() -> None:
    game = _TwoEventGame()
    n_trees, draws = 4_000, 20
    tree, _, _ = _sample(
        game,
        _FairPolicy(),
        _cfg("fixed", n_trees, float(draws), budget=100),
        910,
        np_gen=_FixedPoisson(draws - 1),
    )
    roots = (tree.parent == -1).nonzero(as_tuple=False).squeeze(1)
    counts = []
    for root in roots.tolist():
        start, stop = int(tree.child_ptr[root]), int(tree.child_ptr[root + 1])
        event0 = 0
        for edge in range(start, stop):
            if int(tree.child_event[edge]) == 0:
                event0 = round(float(tree.nu[edge]) * draws)
        counts.append(event0)
        assert stop - start <= 2
    histogram = np.bincount(counts, minlength=draws + 1)
    expected = np.asarray(
        [n_trees * math.comb(draws, k) / 2**draws for k in range(draws + 1)]
    )
    mask = expected >= 5.0
    statistic = float(
        (np.square(histogram[mask] - expected[mask]) / expected[mask]).sum()
    )
    assert statistic < 32.0


def test_budget_clamp_forces_spine_continuation_and_records_metric() -> None:
    tree, _, _ = _sample(
        _TwoEventGame(),
        _FairPolicy(),
        _cfg("fixed", 128, 100.0, budget=1),
        911,
    )
    tree.validate()
    assert tree.budget_clamped
    assert tree.rollout_metrics == {"budget_clamped": True}
    assert bool(((tree.child_ptr[1:] - tree.child_ptr[:-1]) <= 1).all())
    assert int((tree.owner == TERMINAL).sum()) == 128


def test_intensity_controller_formula_fallback_mask_and_state_round_trip() -> None:
    cfg = SimpleNamespace(
        beta_fix={"default": 2.0, "high": 3.0},
        beta_avg={"default": 4.0},
        beta_max={"default": 5.0},
        warmup_iters=1,
    )
    controller = IntensityController(cfg, lambda meta: str(meta.state_ref), True)
    metas = [
        NodeMeta(0, 0, 0, 0, 0, seat_mask=torch.tensor([True, False])),
        NodeMeta(1, 0, 0, 0, 0),
        NodeMeta(2, 0, 0, 0, 0),
        NodeMeta("high", -1, 0, 0, 0),
    ]
    moments = {"vvec": torch.tensor([[1.0, 99.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), "u": torch.tensor([5.0, 0.01, 100.0, 100.0])}
    assert torch.equal(controller.beta(metas, moments), torch.tensor([2.0, 2.0, 2.0, 3.0]))
    controller.end_iteration({"0": [2.0], "1": [2.0], "2": [2.0], "high": [1.0]})
    # s rows are 2, .1, and 10: respectively beta_avg, low clip, high clip.
    assert torch.equal(controller.beta(metas, moments), torch.tensor([4.0, 1.0, 5.0, 3.0]))
    missing = NodeMeta("missing", 0, 0, 0, 0)
    assert float(
        controller.beta(
            missing,
            {"vvec": torch.zeros((1, 2)), "u": torch.ones(1)},
        )[0]
    ) == 2.0
    restored = IntensityController(cfg, lambda meta: str(meta.state_ref), False)
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
