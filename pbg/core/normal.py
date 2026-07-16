"""Compact sampled-private terminal-trajectory training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from pbg.core.game import GameBundle
from pbg.core.execution import DrawPurpose
from pbg.core.rollout import sequential_leaf_logical_rng_keys
from pbg.core.tree import DecisionReachCache


@dataclass
class NormalDataset:
    """Flat scalar decision rows retained by ordinary trajectory training."""

    state_refs: Tensor  # int32 [D]
    reach: DecisionReachCache
    actor_type: Tensor  # int64 [D]
    event: Tensor  # int32 [D]
    old_logp: Tensor  # float32 [D]
    old_v: Tensor  # float32 [D]
    advantage: Tensor  # float32 [D]
    v_target: Tensor  # float32 [D]

    def __len__(self) -> int:
        return int(self.state_refs.numel())


@dataclass
class NormalMinibatch:
    """A sampled minibatch of compact ordinary-training decision rows."""

    state_refs: Tensor
    reach: Tensor
    actor_type: Tensor
    event: Tensor
    old_logp: Tensor
    old_v: Tensor
    advantage: Tensor
    v_target: Tensor
    legal_event_ids: Tensor
    legal_event_mask: Tensor


def index_normal_minibatch(
    sample: NormalMinibatch, rows: Tensor
) -> NormalMinibatch:
    """Select physical rows from one already-sampled effective minibatch."""

    rows = rows.to(device=sample.state_refs.device, dtype=torch.int64)
    return NormalMinibatch(
        **{
            name: getattr(sample, name)[rows]
            for name in sample.__dataclass_fields__
        }
    )


def build_normal_dataset_from_sampled(
    bundle: GameBundle,
    cfg: Any,
    gen: torch.Generator,
    sampled: tuple[Any, Any, Any, Tensor],
    *,
    iteration: int = 0,
) -> NormalDataset:
    """Emit compact scalar targets from sampled terminal trajectories."""

    tree, reach_cache, policy_cache, root_profiles = sampled
    device = tree.owner.device
    p_max = int(bundle.rollout_backend.n_players)
    decisions = policy_cache.node_ids.to(torch.int64)
    if decisions.numel() == 0:
        x_max = max(int(size) for size in bundle.rollout_backend.x_sizes)
        return NormalDataset(
            state_refs=torch.empty(0, dtype=torch.int32, device=device),
            reach=DecisionReachCache.from_rows(
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty((0, p_max, x_max), dtype=torch.float32, device=device),
                bundle.rollout_backend,
            ),
            actor_type=torch.empty(0, dtype=torch.int64, device=device),
            event=torch.empty(0, dtype=torch.int32, device=device),
            old_logp=torch.empty(0, dtype=torch.float32, device=device),
            old_v=torch.empty(0, dtype=torch.float32, device=device),
            advantage=torch.empty(0, dtype=torch.float32, device=device),
            v_target=torch.empty(0, dtype=torch.float32, device=device),
        )

    leaf = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero(as_tuple=False).squeeze(1)
    leaf_tree = tree.tree_id[leaf].to(torch.int64)
    leaf_profiles = root_profiles[leaf_tree]
    keyed_reward = getattr(
        bundle.rollout_backend, "sampled_terminal_reward_keyed", None
    )
    if callable(keyed_reward):
        keys = sequential_leaf_logical_rng_keys(
            tree,
            cfg,
            gen,
            leaf,
            iteration=iteration,
            purpose=DrawPurpose.COMPLETION,
            draw=1,
        )
        terminal_reward = keyed_reward(
            tree.state_ref[leaf], leaf_profiles, keys
        ).float()
    else:
        terminal_reward = bundle.rollout_backend.sampled_terminal_reward(
            tree.state_ref[leaf], leaf_profiles, gen
        ).float()
    start_count = int(cfg.rollout.sequential_starts)
    reward_by_tree = torch.empty(
        (start_count, p_max), dtype=torch.float32, device=device
    )
    reward_by_tree[leaf_tree] = terminal_reward

    owners = tree.owner[decisions].to(torch.int64)
    decision_tree = tree.tree_id[decisions].to(torch.int64)
    actor_type = root_profiles[decision_tree, owners].to(torch.int64)
    rows = torch.arange(decisions.numel(), device=device)
    old_logp = policy_cache.old_logp_child[:, 0].gather(
        1, actor_type[:, None]
    ).squeeze(1)
    old_v = policy_cache.old_v.gather(1, actor_type[:, None]).squeeze(1)

    edge = tree.child_ptr[decisions].to(torch.int64)
    event = tree.child_event[edge].to(torch.int32)
    q_target = torch.empty_like(old_v)
    carry = reward_by_tree.clone()
    # Inclusive depth numbers each owner's decision nodes 1, 2, ... along a
    # trajectory; descending order therefore propagates later returns first.
    owner_depth = tree.depth_p[decisions, owners].to(torch.int64)
    for depth in range(int(bundle.rollout_backend.max_rollout_steps), -1, -1):
        selected = (owner_depth == depth).nonzero(as_tuple=False).squeeze(1)
        if selected.numel() == 0:
            continue
        selected_tree = decision_tree[selected]
        selected_owner = owners[selected]
        q = carry[selected_tree, selected_owner]
        q_target[selected] = q
        carry[selected_tree, selected_owner] = (
            float(cfg.loss.lam) * q
            + (1.0 - float(cfg.loss.lam)) * old_v[selected]
        )

    reach_rows = reach_cache.rows_for(
        tree,
        decisions.to(torch.int32),
        bundle.rollout_backend,
        policy_cache,
    ).float()
    state_refs = tree.state_ref[decisions].to(torch.int32)
    decision_reach = DecisionReachCache.from_rows(
        decisions.to(torch.int32),
        state_refs,
        reach_rows,
        bundle.rollout_backend,
    )
    return NormalDataset(
        state_refs=state_refs,
        reach=decision_reach,
        actor_type=actor_type,
        event=event,
        old_logp=old_logp.detach().float(),
        old_v=old_v.detach().float(),
        advantage=(q_target - old_v).detach().float(),
        v_target=q_target.detach().float(),
    )


def sample_normal_minibatch(
    dataset: NormalDataset,
    bundle: GameBundle,
    size: int,
    gen: torch.Generator,
) -> NormalMinibatch:
    """Uniformly sample the pooled empirical decision law."""

    if size < 0:
        raise ValueError("minibatch size must be nonnegative")
    if len(dataset) == 0 and size:
        raise ValueError("cannot sample from an empty normal dataset")
    rows = torch.randint(
        len(dataset), (size,), generator=gen, device=dataset.state_refs.device
    )
    return gather_normal_minibatch(dataset, bundle, rows)


def gather_normal_minibatch(
    dataset: NormalDataset,
    bundle: GameBundle,
    rows: Tensor,
) -> NormalMinibatch:
    """Gather explicit ordinary-training rows without resampling."""

    rows = rows.to(device=dataset.state_refs.device, dtype=torch.int64)
    states = dataset.state_refs[rows]
    legal_ids, legal_mask = bundle.rollout_backend.legal_events(states)
    return NormalMinibatch(
        state_refs=states,
        reach=dataset.reach.gather(rows, bundle.rollout_backend),
        actor_type=dataset.actor_type[rows],
        event=dataset.event[rows],
        old_logp=dataset.old_logp[rows],
        old_v=dataset.old_v[rows],
        advantage=dataset.advantage[rows],
        v_target=dataset.v_target[rows],
        legal_event_ids=legal_ids,
        legal_event_mask=legal_mask,
    )
