from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

from pbg.core.branching import _sample_branched_labeled
from pbg.core.config import load_config
from pbg.core.game import GameBundle, PRETERM, TERMINAL
from pbg.core.training import build_game_bundle
from pbg.experiments.branching import (
    FixedPolicyBranchingController,
    gamma_deviance,
    reject_post_completion_refill,
    run_branching_experiments,
    update_kappa,
)
from pbg.games.synthetic.family import SyntheticPBG, SyntheticRolloutBackend
from pbg.games.synthetic.exact import enumerate_public_tree
from pbg.games.synthetic.model import (
    SyntheticNet,
    SyntheticPolicyAdapter,
    export_policy_tables,
)
from pbg.games.synthetic.presets import kuhn


ROOT = Path(__file__).parents[1]


def _synthetic_bundle():
    torch.manual_seed(100)
    game = SyntheticPBG(kuhn())
    index = enumerate_public_tree(game)
    model = SyntheticNet(
        game.n_types,
        game.n_player_events,
        {"n_buckets": 64, "hidden_dim": 8},
    )
    bundle = GameBundle(
        model=model,
        rollout_backend=SyntheticRolloutBackend(game, {"default": 2.0}),
        policy_adapter=SyntheticPolicyAdapter(),
        completion_backend=None,
    )
    return game, model, bundle, index


def _synthetic_run(
    seed: int, controller: FixedPolicyBranchingController | None
) -> tuple[object, SyntheticPBG]:
    game, _, bundle, _ = _synthetic_bundle()
    cfg = SimpleNamespace(
        n_trees=1,
        beta_mode="fixed",
        beta_fix={"default": 2.0},
        node_budget=256,
        reach_storage="full",
    )
    tree = _sample_branched_labeled(
        bundle,
        cfg,
        torch.Generator().manual_seed(seed),
        experiment_controller=controller,
    )[0]
    return tree, game


def _synthetic_sample(seed: int, controller: FixedPolicyBranchingController | None):
    return _synthetic_run(seed, controller)[0]


def _terminal_probabilities() -> dict[int, float]:
    game, model, _, index = _synthetic_bundle()
    tables = export_policy_tables(model, game, index)
    profiles = np.asarray(
        [(left, right) for left in range(3) for right in range(3) if left != right],
        dtype=np.int64,
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


def test_absent_and_legacy_controller_have_identical_topology() -> None:
    current = _synthetic_sample(101, None)
    instrumented = _synthetic_sample(
        101, FixedPolicyBranchingController(variant="legacy")
    )
    for field in (
        "parent",
        "child_ptr",
        "child_node",
        "child_event",
        "nu",
        "tree_id",
        "owner",
        "depth_p",
        "depth_player",
        "state_ref",
        "path_weight",
    ):
        assert torch.equal(getattr(current, field), getattr(instrumented, field))


def test_incremental_reserve_matches_oracle_after_every_wave() -> None:
    controller = FixedPolicyBranchingController(
        variant="fixed",
        schedule={"default": 2.0},
        costs={"default": 3.5},
        verify_reserve=True,
    )
    tree = _synthetic_sample(102, controller)
    assert controller.reserve == pytest.approx(float(tree.parent.numel()))


def test_gamma_deviance_has_arithmetic_mean_optimum_across_six_decades() -> None:
    singleton_costs = torch.logspace(0, 6, 7, dtype=torch.float64)
    singleton_log_means = singleton_costs.log().requires_grad_()
    gamma_deviance(singleton_log_means, singleton_costs).sum().backward()
    assert torch.allclose(
        singleton_log_means.grad,
        torch.zeros_like(singleton_log_means),
        atol=1e-12,
        rtol=0.0,
    )

    heavy_tail = torch.cat(
        (torch.ones(999, dtype=torch.float64), torch.tensor([1e6], dtype=torch.float64))
    )
    arithmetic = heavy_tail.mean()
    log_mean = arithmetic.log().detach().requires_grad_()
    gamma_deviance(log_mean.expand_as(heavy_tail), heavy_tail).mean().backward()
    geometric = heavy_tail.log().mean().exp()
    assert abs(float(log_mean.grad)) < 1e-12
    assert float(geometric) < float(arithmetic) / 100.0


def test_kappa_update_and_beta_cap_are_deterministic() -> None:
    assert update_kappa(16.0, 100.0, 25.0, 0.5) == pytest.approx(32.0)
    controller = FixedPolicyBranchingController(
        variant="cost_only",
        schedule={"default": 3.0},
        costs={"default": 1.0},
        kappa=16.0,
    )
    backend = _ToyBackend()
    buffer = _ToyBuffer()
    roots = torch.tensor([0], dtype=torch.int32)
    owner = torch.tensor([0], dtype=torch.int32)
    controller.start(
        backend=backend,
        buffer=buffer,
        root_nodes=roots,
        root_states=roots,
        root_owner=owner,
    )
    beta = controller.choose_beta(
        backend=backend,
        buffer=buffer,
        active_nodes=roots,
        active_states=roots,
        active_owner=owner,
        configured_beta=torch.ones(1),
    )
    assert beta.tolist() == [3.0]
    pending = controller._pending[0]
    assert pending.proposed_beta == 4.0
    assert pending.capped_beta == 3.0


def test_dynamic_budget_can_force_then_restore_branching_exactly() -> None:
    controller = FixedPolicyBranchingController(
        variant="fixed",
        schedule={"default": 3.0},
        costs={"player:-1:0": 5.0, "player:-1:1": 1.0, "default": 1.0},
        soft_budget=10.0,
        verify_reserve=True,
    )
    backend = _ToyBackend()
    buffer = _ToyBuffer()
    root = torch.tensor([0], dtype=torch.int32)
    owner = torch.tensor([0], dtype=torch.int32)
    configured = torch.ones(1)
    controller.start(
        backend=backend,
        buffer=buffer,
        root_nodes=root,
        root_states=root,
        root_owner=owner,
    )
    first = controller.choose_beta(
        backend=backend,
        buffer=buffer,
        active_nodes=root,
        active_states=root,
        active_owner=owner,
        configured_beta=configured,
    )
    assert first.tolist() == [3.0]
    admitted_first = controller.after_offspring(
        active_nodes=root,
        configured_beta=configured,
        effective_beta=first,
        offspring=torch.full((1,), 3, dtype=torch.int64),
    )
    assert admitted_first.tolist() == [1]
    child = torch.tensor([1], dtype=torch.int32)
    buffer.node_count = 2
    controller.after_children(
        backend=backend,
        buffer=buffer,
        active_nodes=root,
        active_states=root,
        active_owner=owner,
        offspring=admitted_first,
        parent_positions=torch.tensor([0], dtype=torch.int64),
        child_nodes=child,
        next_states=child,
        next_owner=owner,
        child_events=torch.tensor([7], dtype=torch.int32),
    )
    assert controller.reserve == 3.0
    second = controller.choose_beta(
        backend=backend,
        buffer=buffer,
        active_nodes=child,
        active_states=child,
        active_owner=owner,
        configured_beta=configured,
    )
    assert second.tolist() == [3.0]
    admitted_second = controller.after_offspring(
        active_nodes=child,
        configured_beta=configured,
        effective_beta=second,
        offspring=torch.full((1,), 3, dtype=torch.int64),
    )
    assert admitted_second.tolist() == [3]


def test_dynamic_switching_preserves_exact_weighted_leaf_mass() -> None:
    expected = _terminal_probabilities()
    observed = np.zeros(len(expected), dtype=np.float64)
    restored = False
    trials = 500
    for seed in range(110, 110 + trials):
        controller = FixedPolicyBranchingController(
            variant="fixed",
            schedule={"default": 3.0},
            costs={"player:-1:0": 5.0, "player:-1:1": 1.0, "default": 1.0},
            soft_budget=6.0,
        )
        tree, game = _synthetic_run(seed, controller)
        leaves = tree.child_ptr[1:] == tree.child_ptr[:-1]
        assert float(tree.path_weight[leaves].sum()) == pytest.approx(1.0, abs=1e-6)
        terminal = ((tree.owner == TERMINAL) | (tree.owner == PRETERM)).nonzero(
            as_tuple=False
        ).squeeze(1)
        keys = game.encode_public(tree.state_ref[terminal]).numpy()
        weights = tree.path_weight[terminal]
        observed += np.asarray(
            [float(weights[torch.from_numpy(keys == key)].sum()) for key in expected]
        )
        assert controller.records[0]["budget_admitted_beta"] == 1.0
        restored |= any(
            record["budget_admitted_beta"] > 1.0 for record in controller.records[1:]
        )
    assert restored
    target = trials * np.asarray(list(expected.values()))
    assert float((np.square(observed - target) / target).sum()) < 18.467


def test_post_completion_refill_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="post-completion refill"):
        reject_post_completion_refill()


def test_cpu_harness_emits_raw_and_grouped_measurements(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "configs" / "kuhn_smoke.yaml")
    cfg.training.mode = "branched_rao_blackwell"
    cfg.rollout.n_trees = 1
    cfg.rollout.beta = {"default": 1.0}
    cfg.rollout.node_budget = 128
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": bundle.model.state_dict(), "iteration": 7}, checkpoint)
    checkpoint_before = checkpoint.read_bytes()
    result = run_branching_experiments(
        cfg,
        [checkpoint],
        variants=["legacy"],
        trials=2,
        seed=103,
        device=torch.device("cpu"),
    )
    assert result["schema"] == "pbg-branching-experiment-v1"
    assert checkpoint.read_bytes() == checkpoint_before
    experiment = result["experiments"][0]
    assert experiment["checkpoint"]["iteration"] == 7
    assert experiment["summary"]["street_depth"]
    trial = experiment["trials"][0]
    assert {"nodes", "leaves", "decision_rows", "decision_law_ess"} <= set(
        trial["metrics"]
    )
    assert "arithmetic_bucket_prediction" in trial["nodes"][0]


class _ToyBackend:
    n_players = 1
    states = None

    def legal_events(self, states: Tensor) -> tuple[Tensor, Tensor]:
        ids = torch.arange(10, dtype=torch.int32).expand(states.numel(), -1)
        return ids, torch.ones_like(ids, dtype=torch.bool)


@dataclass
class _ToyBuffer:
    node_capacity: int = 8
    node_count: int = 1

    def __post_init__(self) -> None:
        self.depth_player = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
        self.path_weight = torch.ones(self.node_capacity, dtype=torch.float32)
