from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from pbg.core.game import CHANCE, PRETERM, TERMINAL, GameBundle
from pbg.core.rollout import _sample_trees_sequential, sample_trees
from pbg.core.tree import DeviceTreeBuffer
from pbg.games.synthetic.family import SyntheticPBG, SyntheticRolloutBackend
from pbg.games.synthetic.model import SyntheticNet, SyntheticPolicyAdapter
from pbg.games.synthetic.presets import kuhn, leduc


def _bundle(game: SyntheticPBG) -> GameBundle:
    model = SyntheticNet(game.n_types, game.n_player_events, {"hidden_dim": 8})
    return GameBundle(
        model=model,
        rollout_backend=SyntheticRolloutBackend(game),
        policy_adapter=SyntheticPolicyAdapter(),
        completion_backend=None,  # Completion is deliberately outside C08 rollout.
    )


def _cfg(n_trees: int = 32, node_budget: int = 32, mode: str = "full"):
    return SimpleNamespace(
        n_trees=n_trees,
        node_budget=node_budget,
        beta_mode="sequential",
        reach_storage=mode,
    )


def _step(game: SyntheticPBG, state: int, event: int) -> int:
    return int(
        game.step(
            torch.tensor([state], dtype=torch.int32),
            torch.tensor([event], dtype=torch.int32),
        )[0]
    )


def test_device_sampler_stays_resident_and_builds_one_path_per_tree() -> None:
    torch.manual_seed(80)
    game = SyntheticPBG(leduc())
    tree, reach, policy = _sample_trees_sequential(
        _bundle(game), _cfg(), torch.Generator().manual_seed(81)
    )
    tree.validate()
    assert all(
        value.device.type == "cpu"
        for value in tree.__dict__.values()
        if isinstance(value, torch.Tensor)
    )
    assert torch.all(tree.child_ptr[1:] - tree.child_ptr[:-1] <= 1)
    assert int(((tree.owner == PRETERM) | (tree.owner == TERMINAL)).sum()) == 32
    assert reach.log_r.device == tree.parent.device
    assert policy.old_logp_child.device == tree.parent.device


def test_sequential_sampler_draws_one_private_profile_batch_at_roots() -> None:
    torch.manual_seed(84)
    game = SyntheticPBG(leduc())
    bundle = _bundle(game)
    backend = bundle.rollout_backend
    original = backend.sample_profiles
    sampled_states: list[torch.Tensor] = []

    def counted_sample_profiles(states, reach, gen):
        sampled_states.append(states.clone())
        return original(states, reach, gen)

    backend.sample_profiles = counted_sample_profiles
    _sample_trees_sequential(
        bundle, _cfg(n_trees=32), torch.Generator().manual_seed(85)
    )

    assert len(sampled_states) == 1
    assert torch.equal(sampled_states[0], game.root_states(32))


def test_corrected_leduc_completion_cut_rolls_through_board_and_final_betting() -> None:
    game = SyntheticPBG(leduc())
    backend = SyntheticRolloutBackend(game)
    root = int(game.root_states(1)[0])
    before_board = _step(game, _step(game, root, 1), 1)
    assert int(backend.classify(torch.tensor([before_board]))[0]) == CHANCE
    ids, mask = game.legal_events(torch.tensor([before_board], dtype=torch.int32))
    after_board = _step(game, before_board, int(ids[0, mask[0]][0]))
    assert int(backend.classify(torch.tensor([after_board]))[0]) >= 0
    after_betting = _step(game, _step(game, after_board, 1), 1)
    assert int(backend.classify(torch.tensor([after_betting]))[0]) == PRETERM
    fold = _step(game, _step(game, root, 2), 0)
    assert int(backend.classify(torch.tensor([fold]))[0]) == TERMINAL


def test_buffer_overflow_is_loud_and_returns_no_partial_tree() -> None:
    game = SyntheticPBG(kuhn())
    with pytest.raises(RuntimeError, match=r"requested nodes=.*capacity nodes="):
        _sample_trees_sequential(
            _bundle(game), _cfg(n_trees=2, node_budget=1),
            torch.Generator().manual_seed(82),
        )


def test_sample_trees_is_the_sole_dispatch_entry() -> None:
    game = SyntheticPBG(kuhn())
    cfg = _cfg(n_trees=1)
    tree, _, _ = sample_trees(_bundle(game), cfg, torch.Generator().manual_seed(83))
    assert tree.parent.numel() > 0


def test_production_hot_functions_contain_no_explicit_host_transfers() -> None:
    banned = {"cpu", "numpy", "tolist", "item"}
    for function in (
        _sample_trees_sequential,
        DeviceTreeBuffer.append_children,
        DeviceTreeBuffer.finalize,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in banned
        }
        assert not used
