from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from pbg.core.config import load_config, seed_all
from pbg.core.game import GameBundle, PRETERM, TERMINAL
from pbg.core.normal import build_normal_dataset_from_sampled
from pbg.core.queued import DrawPurpose, FamilyPackingLimits
from pbg.core.rollout import (
    _global_trajectory_start,
    _sample_trees_sequential_queued,
    sample_trees,
    sample_trees_sequential,
    sequential_leaf_logical_rng_keys,
)
from pbg.core.values import HeadEvals, streamed_branched_value
from pbg.games.synthetic.bundle import SyntheticMCCompletion
from pbg.games.synthetic.exact import enumerate_public_tree
import pbg.core.rollout as rollout_module
from pbg.core.training import _build_iteration_data, _loss_spec, build_game_bundle
from pbg.games.synthetic.family import SyntheticPBG, SyntheticRolloutBackend
from pbg.games.synthetic.model import SyntheticNet, SyntheticPolicyAdapter
from pbg.games.synthetic.presets import kuhn, leduc


def _bundle(game: SyntheticPBG, model: SyntheticNet | None = None) -> GameBundle:
    return GameBundle(
        model=model
        or SyntheticNet(game.n_types, game.n_player_events, {"hidden_dim": 8}),
        rollout_backend=SyntheticRolloutBackend(game),
        policy_adapter=SyntheticPolicyAdapter(),
        completion_backend=None,
    )


def _cfg(
    *,
    n_trees: int = 24,
    node_budget: int = 32,
    batch: int = 4,
    admission: int = 8,
    reach_storage: str = "full",
) -> SimpleNamespace:
    return SimpleNamespace(
        n_trees=n_trees,
        node_budget=node_budget,
        beta_mode="sequential",
        reach_storage=reach_storage,
        executor="family_fifo_v1",
        queue_batch_rows=batch,
        queue_admission_size=admission,
        arena_chunk_nodes=16,
        logical_seed=91,
    )


def _canonical_paths(tree, policy) -> tuple[tuple[object, ...], ...]:
    policy_row = {
        int(node): row for row, node in enumerate(policy.node_ids.cpu().tolist())
    }
    result: list[tuple[object, ...]] = []
    roots = (tree.parent == -1).nonzero(as_tuple=False).squeeze(1)
    for root in roots.cpu().tolist():
        node = root
        path: list[object] = []
        while True:
            owner = int(tree.owner[node])
            if node in policy_row:
                path.append((owner, "decision"))
            first = int(tree.child_ptr[node])
            stop = int(tree.child_ptr[node + 1])
            if first == stop:
                path.append((owner, "leaf"))
                break
            assert stop == first + 1
            path.append(("event", int(tree.child_event[first])))
            node = int(tree.child_node[first])
        result.append(tuple(path))
    return tuple(result)


def test_queued_schedule_and_admission_do_not_change_logical_trajectories() -> None:
    torch.manual_seed(920)
    template_game = SyntheticPBG(leduc())
    template_model = SyntheticNet(
        template_game.n_types, template_game.n_player_events, {"hidden_dim": 8}
    )
    left_game = SyntheticPBG(leduc())
    right_game = SyntheticPBG(leduc())
    left_model = copy.deepcopy(template_model)
    right_model = copy.deepcopy(template_model)

    left = _sample_trees_sequential_queued(
        _bundle(left_game, left_model),
        _cfg(batch=1, admission=1),
        torch.Generator().manual_seed(17),
        return_profiles=True,
        iteration=5,
    )
    right = _sample_trees_sequential_queued(
        _bundle(right_game, right_model),
        _cfg(batch=7, admission=19),
        torch.Generator().manual_seed(999),
        return_profiles=True,
        iteration=5,
    )
    left_tree, _, left_policy, left_profiles = left
    right_tree, _, right_policy, right_profiles = right
    left_tree.validate()
    right_tree.validate()

    assert torch.equal(left_profiles, right_profiles)
    assert _canonical_paths(left_tree, left_policy) == _canonical_paths(
        right_tree, right_policy
    )
    assert int(((left_tree.owner == TERMINAL) | (left_tree.owner == PRETERM)).sum()) == 24


def test_larger_queued_resident_pool_reduces_model_calls_for_same_work() -> None:
    torch.manual_seed(923)
    template = SyntheticPBG(leduc())
    model = SyntheticNet(template.n_types, template.n_player_events, {"hidden_dim": 8})
    small_bundle = _bundle(SyntheticPBG(leduc()), copy.deepcopy(model))
    large_bundle = _bundle(SyntheticPBG(leduc()), copy.deepcopy(model))
    small = _sample_trees_sequential_queued(
        small_bundle,
        _cfg(n_trees=32, batch=1, admission=1),
        torch.Generator().manual_seed(1),
        iteration=6,
    )[0]
    large = _sample_trees_sequential_queued(
        large_bundle,
        _cfg(n_trees=32, batch=8, admission=32),
        torch.Generator().manual_seed(2),
        iteration=6,
    )[0]
    small_calls = small_bundle.rollout_backend.execution_metrics.snapshot()[
        "execution_model_calls"
    ]
    large_calls = large_bundle.rollout_backend.execution_metrics.snapshot()[
        "execution_model_calls"
    ]
    assert small.parent.numel() == large.parent.numel()
    assert small_calls > large_calls


def test_unequal_rank_quotas_use_collision_free_global_prefixes(monkeypatch) -> None:
    monkeypatch.setattr(rollout_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(rollout_module.dist, "is_initialized", lambda: True)
    rank = {"value": 0}
    monkeypatch.setattr(rollout_module.dist, "get_rank", lambda: rank["value"])
    cfg = SimpleNamespace(trajectory_rank_counts=(3, 5, 2))

    starts = []
    for current_rank, local_count in enumerate(cfg.trajectory_rank_counts):
        rank["value"] = current_rank
        starts.append(_global_trajectory_start(cfg, local_count))

    assert starts == [0, 3, 8]
    assert [
        set(range(start, start + count))
        for start, count in zip(starts, cfg.trajectory_rank_counts, strict=True)
    ] == [set(range(0, 3)), set(range(3, 8)), set(range(8, 10))]
    rank["value"] = 1
    with pytest.raises(ValueError, match="does not match"):
        _global_trajectory_start(cfg, 4)


@pytest.mark.parametrize("reach_storage", ["full", "decisions_only", "recompute"])
def test_queued_sequential_rollout_preserves_reach_and_old_output_contracts(
    reach_storage: str,
) -> None:
    torch.manual_seed(921)
    game = SyntheticPBG(leduc())
    bundle = _bundle(game)
    cfg = _cfg(reach_storage=reach_storage)
    tree, reach, policy = sample_trees(
        bundle, cfg, torch.Generator().manual_seed(21), iteration=3
    )
    tree.validate()
    decision_reach = reach.rows_for(tree, policy.node_ids, bundle.rollout_backend, policy)
    with torch.no_grad():
        current = bundle.rollout_backend.policy_forward(
            bundle.model, tree.state_ref[policy.node_ids.long()], decision_reach
        )
    torch.testing.assert_close(current.v.float(), policy.old_v)
    assert torch.isfinite(policy.old_logp_child).all()
    assert torch.all(tree.child_ptr[1:] - tree.child_ptr[:-1] <= 1)


def test_recompute_uses_one_live_reach_row_per_trajectory(monkeypatch) -> None:
    game = SyntheticPBG(kuhn())
    n_trees = 5
    node_budget = 17
    forbidden = (n_trees * node_budget, game.n_players, game.n_types)
    observed: list[tuple[int, ...]] = []
    original_empty = rollout_module.torch.empty

    def recording_empty(*args, **kwargs):
        if args and isinstance(args[0], (tuple, list)):
            observed.append(tuple(int(value) for value in args[0]))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(rollout_module.torch, "empty", recording_empty)
    tree, reach, _ = _sample_trees_sequential_queued(
        _bundle(game),
        _cfg(
            n_trees=n_trees,
            node_budget=node_budget,
            reach_storage="recompute",
        ),
        torch.Generator().manual_seed(31),
    )

    assert tree.parent.numel() > n_trees
    assert reach.log_r.shape == (n_trees, game.n_players, game.n_types)
    assert forbidden not in observed
    assert (n_trees, game.n_players, game.n_types) in observed


def test_queue_metrics_report_exact_nonzero_age_and_stratified_drains() -> None:
    game = SyntheticPBG(leduc())
    bundle = _bundle(game)
    _sample_trees_sequential_queued(
        bundle,
        _cfg(n_trees=12, batch=8, admission=3),
        torch.Generator().manual_seed(32),
    )
    snapshot = bundle.rollout_backend.execution_metrics.snapshot()

    assert snapshot["execution_queue_oldest_age_max"] > 0
    assert snapshot["execution_forced_drains"] > 0
    assert snapshot["execution_rollout_synthetic_forced_drains"] == snapshot[
        "execution_forced_drains"
    ]
    assert snapshot["execution_rollout_synthetic_forced_drain_rows"] == snapshot[
        "execution_forced_drain_rows"
    ]


def test_byte_safe_admission_bounds_resident_pool_and_fails_before_sampling() -> None:
    game = SyntheticPBG(kuhn())
    bundle = _bundle(game)
    backend = bundle.rollout_backend
    backend.queued_admission_byte_terms = lambda _cfg, _batch: (100, 50, 100)
    cfg = _cfg(n_trees=10, batch=4, admission=10)
    cfg.queue_memory_bytes = 300  # base 200 + exactly two resident starts

    tree, _, _ = _sample_trees_sequential_queued(
        bundle, cfg, torch.Generator().manual_seed(33), iteration=2
    )
    snapshot = backend.execution_metrics.snapshot()
    assert tree.parent.numel() > cfg.n_trees
    assert snapshot["execution_reserved_bytes_max"] == 300

    blocked = _bundle(SyntheticPBG(kuhn()))
    blocked.rollout_backend.queued_admission_byte_terms = (
        lambda _cfg, _batch: (100, 50, 100)
    )
    calls = {"profiles": 0}
    original = blocked.rollout_backend.sample_profiles_keyed

    def recording_profiles(states, reach, keys):
        calls["profiles"] += 1
        return original(states, reach, keys)

    blocked.rollout_backend.sample_profiles_keyed = recording_profiles
    cfg.queue_memory_bytes = 249
    with pytest.raises(MemoryError, match="no capacity for one"):
        _sample_trees_sequential_queued(
            blocked, cfg, torch.Generator().manual_seed(34), iteration=2
        )
    assert calls["profiles"] == 0
    assert blocked.rollout_backend.execution_metrics.snapshot()[
        "execution_model_calls"
    ] == 0


def test_scheduler_uses_calibrated_family_prefix_instead_of_row_limit() -> None:
    bundle = _bundle(SyntheticPBG(kuhn()))
    bundle.rollout_backend.queued_family_packing_limits = (
        lambda _family, _phase, _rows: FamilyPackingLimits(
            max_rows=4,
            max_padded_tokens=2,
            max_padded_token_pairs=2,
            max_workspace_bytes=16,
        )
    )
    _sample_trees_sequential_queued(
        bundle,
        _cfg(n_trees=12, batch=4, admission=12),
        torch.Generator().manual_seed(35),
    )
    snapshot = bundle.rollout_backend.execution_metrics.snapshot()

    assert snapshot["execution_model_rows_max"] <= 2
    assert snapshot["execution_model_calls"] > 1


def test_queued_normal_rollout_emits_only_scalar_sampled_targets() -> None:
    torch.manual_seed(922)
    game = SyntheticPBG(kuhn())
    bundle = _bundle(game)
    rollout = _cfg(n_trees=16, node_budget=16)
    sampled = sample_trees_sequential(
        bundle,
        rollout,
        torch.Generator().manual_seed(22),
        return_profiles=True,
        iteration=4,
    )
    cfg = SimpleNamespace(
        rollout=rollout,
        loss=SimpleNamespace(lam=0.95),
    )
    dataset = build_normal_dataset_from_sampled(
        bundle, cfg, torch.Generator().manual_seed(23), sampled
    )
    assert len(dataset) > 0
    for value in (
        dataset.actor_type,
        dataset.event,
        dataset.old_logp,
        dataset.old_v,
        dataset.advantage,
        dataset.v_target,
    ):
        assert value.ndim == 1 and value.shape == dataset.state_refs.shape


def test_queued_sequential_rb_builds_posterior_weighted_all_type_targets() -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.rollout.n_trees = 12
    cfg.rollout.executor = "family_fifo_v1"
    cfg.rollout.queue_batch_rows = 4
    cfg.rollout.queue_admission_size = 8
    cfg.rollout.arena_chunk_nodes = 16
    cfg.model["hidden_dim"] = 8
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    generator, _ = seed_all(cfg.training.seed, device="cpu")
    data = _build_iteration_data(
        cfg, bundle, generator, _loss_spec(cfg), iteration=7
    )
    assert not data.normal
    assert data.targets is not None
    assert data.targets.parent_p_hat.ndim == 2
    assert data.targets.parent_p_hat.shape[1] == max(bundle.rollout_backend.x_sizes)
    assert data.targets.child_p_hat.ndim == 3
    assert torch.isfinite(data.targets.parent_p_hat).all()
    assert torch.isfinite(data.targets.child_p_hat).all()


def _queued_nonexact_rb_targets(
    template_model: SyntheticNet,
    *,
    queue_rows: int,
    admission: int,
    completion_rows: int,
    mutable_seed: int,
):
    game = SyntheticPBG(leduc())
    # Production bundle construction pre-indexes the complete public tree so
    # opaque state refs/model embedding keys are identical across executors.
    enumerate_public_tree(game)
    bundle = GameBundle(
        model=copy.deepcopy(template_model),
        rollout_backend=SyntheticRolloutBackend(game),
        policy_adapter=SyntheticPolicyAdapter(),
        completion_backend=SyntheticMCCompletion(game),
    )
    cfg = _cfg(n_trees=12, batch=queue_rows, admission=admission)
    generator = torch.Generator().manual_seed(mutable_seed)
    tree, reach, policy = sample_trees(
        bundle, cfg, generator, iteration=19
    )
    leaves = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero(
        as_tuple=False
    ).squeeze(1)
    keys = sequential_leaf_logical_rng_keys(
        tree,
        cfg,
        generator,
        leaves,
        iteration=19,
        purpose=DrawPurpose.COMPLETION,
    )

    def batches():
        for first in range(0, leaves.numel(), completion_rows):
            selected = leaves[first : first + completion_rows].to(torch.int32)
            selected_reach = reach.rows_for(
                tree, selected, bundle.rollout_backend, policy
            )
            stats = bundle.completion_backend.complete_keyed(
                bundle.rollout_backend,
                tree,
                selected,
                selected_reach,
                9,
                keys[first : first + selected.numel()],
            )
            yield selected, stats

    targets = streamed_branched_value(
        tree,
        leaves.to(torch.int32),
        batches(),
        reach,
        HeadEvals(node_ids=policy.node_ids, v=policy.old_v),
        0.95,
        bundle.rollout_backend,
        configured_node_budget=cfg.node_budget,
        completion_batch_rows=completion_rows,
        max_workspace_bytes=1 << 28,
    )
    decision_nodes = targets.node_ids.to(torch.int64)
    stride = int(tree.depth_player.max()) + 1
    decision_order = torch.argsort(
        tree.tree_id[decision_nodes].to(torch.int64) * stride
        + tree.depth_player[decision_nodes].to(torch.int64)
    )
    leaf_order = torch.argsort(tree.tree_id[leaves].to(torch.int64))
    fields = {
        "state_ref": tree.state_ref[decision_nodes][decision_order],
        "old_v": policy.old_v[decision_order],
    }
    fields.update({
        name: getattr(targets, name)[decision_order]
        for name in (
            "parent_p_hat",
            "child_p_hat",
            "a_hat",
            "v_hat",
            "child_event_row",
            "child_mask",
            "child_count",
        )
    })
    return tree, policy, keys[leaf_order], fields


def test_nonexact_queued_rb_completion_and_targets_are_logically_addressed() -> None:
    """Queue and completion partitions cannot perturb sequential-MC targets."""

    torch.manual_seed(924)
    template_game = SyntheticPBG(leduc())
    template_model = SyntheticNet(
        template_game.n_types,
        template_game.n_player_events,
        {"hidden_dim": 8},
    )
    left = _queued_nonexact_rb_targets(
        template_model,
        queue_rows=1,
        admission=1,
        completion_rows=1,
        mutable_seed=3,
    )
    right = _queued_nonexact_rb_targets(
        template_model,
        queue_rows=7,
        admission=12,
        completion_rows=12,
        mutable_seed=999,
    )

    assert _canonical_paths(left[0], left[1]) == _canonical_paths(right[0], right[1])
    assert torch.equal(left[2], right[2])
    assert left[3].keys() == right[3].keys()
    for name in left[3]:
        torch.testing.assert_close(
            left[3][name],
            right[3][name],
            msg=lambda message, field=name: f"{field}: {message}",
        )


def test_iteration_builder_uses_keyed_completion_for_queued_sequential_rb(
    monkeypatch,
) -> None:
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.rollout.n_trees = 6
    cfg.rollout.executor = "family_fifo_v1"
    cfg.rollout.queue_batch_rows = 3
    cfg.rollout.queue_admission_size = 6
    cfg.rollout.arena_chunk_nodes = 8
    cfg.completion.backend = "mc"
    cfg.completion.n_samples = 3
    cfg.model["hidden_dim"] = 8
    bundle = build_game_bundle(cfg, torch.device("cpu"))

    def mutable_completion_is_forbidden(*args, **kwargs):
        raise AssertionError("queued sequential RB must use complete_keyed")

    monkeypatch.setattr(
        bundle.completion_backend, "complete", mutable_completion_is_forbidden
    )
    data = _build_iteration_data(
        cfg,
        bundle,
        torch.Generator().manual_seed(1234),
        _loss_spec(cfg),
        iteration=23,
    )
    assert data.targets is not None
    assert torch.isfinite(data.targets.a_hat).all()
    metrics = bundle.rollout_backend.execution_metrics.snapshot()
    assert metrics["execution_completion_rows"] == 6
    assert metrics["execution_completion_batches"] >= 1
    assert metrics["execution_target_bytes"] > 0
    assert metrics["execution_target_bytes_max"] == metrics["execution_target_bytes"]


def _executor_law_summary(mode: str, executor: str, model_state, seed: int):
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.training.mode = mode
    cfg.rollout.n_trees = 1024
    cfg.rollout.node_budget = 16
    cfg.rollout.executor = executor
    cfg.rollout.queue_batch_rows = 64
    cfg.rollout.queue_admission_size = 256
    cfg.rollout.arena_chunk_nodes = 256
    cfg.completion.backend = "mc"
    cfg.completion.n_samples = None if mode == "normal" else 16
    cfg.model["hidden_dim"] = 8
    bundle = build_game_bundle(cfg, torch.device("cpu"))
    if model_state is not None:
        bundle.model.load_state_dict(model_state)
    data = _build_iteration_data(
        cfg,
        bundle,
        torch.Generator().manual_seed(seed),
        _loss_spec(cfg),
        iteration=31,
    )
    if mode == "normal":
        dataset = data.dataset
        width = bundle.rollout_backend.game.n_player_events
        joint = dataset.actor_type * width + dataset.event.to(torch.int64)
        histogram = torch.bincount(
            joint, minlength=max(bundle.rollout_backend.x_sizes) * width
        ).to(torch.float64)
        histogram /= histogram.sum()
        scalar = torch.stack(
            (
                dataset.advantage.mean(),
                dataset.advantage.square().mean(),
                dataset.v_target.mean(),
                dataset.v_target.square().mean(),
            )
        ).to(torch.float64)
        return bundle.model.state_dict(), histogram, scalar

    targets = data.targets
    posterior = targets.parent_p_hat.mean(0).to(torch.float64)
    policy_value = (
        targets.child_p_hat * targets.a_hat * targets.child_mask[..., None]
    ).sum((1, 2))
    value_target = (targets.parent_p_hat * targets.v_hat).sum(1)
    scalar = torch.stack(
        (
            policy_value.mean(),
            policy_value.square().mean(),
            value_target.mean(),
            value_target.square().mean(),
        )
    ).to(torch.float64)
    return bundle.model.state_dict(), posterior, scalar


@pytest.mark.parametrize("mode", ["normal", "rao_blackwell"])
def test_wavefront_and_queued_sequential_estimators_match_statistically(mode: str) -> None:
    """The opt-in executor changes batching, not the finite-sample law."""

    torch.manual_seed(925)
    state, baseline_distribution, baseline_moments = _executor_law_summary(
        mode, "wavefront_v1", None, 101
    )
    _, queued_distribution, queued_moments = _executor_law_summary(
        mode, "family_fifo_v1", state, 707
    )

    total_variation = 0.5 * (baseline_distribution - queued_distribution).abs().sum()
    assert float(total_variation) < 0.08
    torch.testing.assert_close(
        baseline_moments,
        queued_moments,
        atol=0.12,
        rtol=0.12,
    )


def test_queued_overflow_is_loud_before_returning_a_partial_tree() -> None:
    game = SyntheticPBG(kuhn())
    with pytest.raises(RuntimeError, match="overflow before child expansion"):
        _sample_trees_sequential_queued(
            _bundle(game),
            _cfg(n_trees=2, node_budget=1, batch=2, admission=2),
            torch.Generator().manual_seed(24),
        )


def test_queued_core_scheduler_has_no_explicit_hot_path_host_transfer() -> None:
    syntax = ast.parse(
        textwrap.dedent(inspect.getsource(_sample_trees_sequential_queued))
    )
    banned = {"cpu", "numpy", "tolist", "item"}
    used = {
        node.attr
        for node in ast.walk(syntax)
        if isinstance(node, ast.Attribute) and node.attr in banned
    }
    assert not used


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_queued_sequential_rb_builds_targets_on_cuda() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = load_config("configs/kuhn_smoke.yaml")
    cfg.rollout.n_trees = 4
    cfg.rollout.executor = "family_fifo_v1"
    cfg.rollout.queue_batch_rows = 2
    cfg.rollout.queue_admission_size = 4
    cfg.rollout.arena_chunk_nodes = 8
    cfg.completion.n_samples = 1
    cfg.model["hidden_dim"] = 8
    bundle = build_game_bundle(cfg, device)
    generator, _ = seed_all(cfg.training.seed, device=device)
    data = _build_iteration_data(
        cfg, bundle, generator, _loss_spec(cfg), iteration=11
    )
    assert data.targets is not None
    assert data.targets.node_ids.device == device
    assert data.decision_reach is not None
    assert data.decision_reach.state_refs.device == device
    snapshot = bundle.rollout_backend.execution_metrics.snapshot()
    # Dynamic-shape nonzero compaction currently synchronizes CUDA.  Keep the
    # debt visible until the device-resident masked scheduler replaces it.
    assert snapshot["execution_host_synchronizations"] > 0
    assert snapshot["execution_host_synchronization_seconds"] >= 0.0
    torch.cuda.synchronize(device)
