"""Tensor-aligned decision-law minibatch sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from pbg.core.game import GameBundle
from pbg.core.tree import (
    AliasTable,
    DecisionReachCache,
    PolicyCache,
    ReachCache,
    TreeBatch,
)
from pbg.core.values import NodeTargets


@dataclass
class MinibatchSample:
    """Device-local rows required by the minimal training objective."""

    decision_rows: Tensor  # int32 [B]
    state_refs: Tensor  # int32 [B]
    reach: Tensor  # float32 [B,P,X]
    targets: NodeTargets
    old_logp_child: Tensor  # float32 [B,E,X]
    old_v: Tensor  # float32 [B,X]
    valid_type_mask: Tensor  # bool [B,X]
    legal_event_ids: Tensor  # int32 [B,E_legal]
    legal_event_mask: Tensor  # bool [B,E_legal]


def index_minibatch(sample: MinibatchSample, rows: Tensor) -> MinibatchSample:
    """Select physical rows without changing their sampled semantics."""

    rows = rows.to(device=sample.state_refs.device, dtype=torch.int64)
    return MinibatchSample(
        decision_rows=sample.decision_rows[rows],
        state_refs=sample.state_refs[rows],
        reach=sample.reach[rows],
        targets=_gather_targets(sample.targets, rows),
        old_logp_child=sample.old_logp_child[rows],
        old_v=sample.old_v[rows],
        valid_type_mask=sample.valid_type_mask[rows],
        legal_event_ids=sample.legal_event_ids[rows],
        legal_event_mask=sample.legal_event_mask[rows],
    )


def _gather_targets(targets: NodeTargets, rows: Tensor) -> NodeTargets:
    """Gather target rows exclusively with tensor indexing."""

    return NodeTargets(
        node_ids=targets.node_ids[rows],
        parent_p_hat=targets.parent_p_hat[rows],
        child_p_hat=targets.child_p_hat[rows],
        a_hat=targets.a_hat[rows],
        v_hat=targets.v_hat[rows],
        child_event_row=targets.child_event_row[rows],
        child_mask=targets.child_mask[rows],
        child_count=targets.child_count[rows],
        q_hat=None if targets.q_hat is None else targets.q_hat[rows],
        aux={name: value[rows] for name, value in targets.aux.items()},
    )


def sample_minibatch(
    tree: TreeBatch | None,
    targets: NodeTargets,
    alias: AliasTable,
    policy_cache: PolicyCache,
    reach_cache: ReachCache | DecisionReachCache,
    bundle: GameBundle,
    B: int,
    gen: torch.Generator,
) -> MinibatchSample:
    """Sample decision rows and gather every cache in the same row order."""

    if B < 0:
        raise ValueError("B must be nonnegative")
    if not torch.equal(policy_cache.node_ids, targets.node_ids):
        raise ValueError("PolicyCache and NodeTargets node_ids must match exactly")
    rows = alias.sample(B, gen).to(device=targets.node_ids.device, dtype=torch.int64)
    if rows.numel():
        valid_rows = ((rows >= 0) & (rows < targets.node_ids.numel())).all()
        if rows.device.type == "cuda":
            torch._assert_async(valid_rows, "alias table produced an invalid decision row")
        elif not bool(valid_rows):
            raise ValueError("alias table produced an invalid decision row")
    gathered = _gather_targets(targets, rows)
    node_ids = gathered.node_ids.to(torch.int64)
    if isinstance(reach_cache, DecisionReachCache):
        if not torch.equal(reach_cache.node_ids, targets.node_ids):
            raise ValueError("DecisionReachCache and NodeTargets node_ids must match")
        state_refs = reach_cache.state_refs[rows].to(
            device=targets.node_ids.device, dtype=torch.int32
        )
        if reach_cache.owners is None:
            if tree is None:
                raise ValueError("compact decision reaches require cached owners")
            tree_rows = node_ids.to(tree.owner.device)
            owners = tree.owner[tree_rows].to(
                device=targets.node_ids.device, dtype=torch.int64
            )
        else:
            owners = reach_cache.owners[rows].to(
                device=targets.node_ids.device, dtype=torch.int64
            )
        reach = reach_cache.gather(rows, bundle.rollout_backend)
    else:
        if tree is None:
            raise ValueError("a full ReachCache requires the sampled topology")
        tree_rows = node_ids.to(tree.state_ref.device)
        state_refs = tree.state_ref[tree_rows].to(
            device=targets.node_ids.device, dtype=torch.int32
        )
        owners = tree.owner[tree_rows].to(
            device=targets.node_ids.device, dtype=torch.int64
        )
        reach = reach_cache.rows_for(
            tree, gathered.node_ids, bundle.rollout_backend, policy_cache
        )
    reach = reach.to(device=targets.node_ids.device, dtype=torch.float32)
    legal_ids, legal_mask = bundle.rollout_backend.legal_events(
        state_refs.to(bundle.rollout_backend.device)
    )
    legal_ids = legal_ids.to(device=targets.node_ids.device, dtype=torch.int32)
    legal_mask = legal_mask.to(device=targets.node_ids.device, dtype=torch.bool)

    x_max = targets.parent_p_hat.shape[1]
    widths = torch.tensor(
        bundle.rollout_backend.x_sizes, dtype=torch.int64, device=targets.node_ids.device
    )
    valid = torch.arange(x_max, device=targets.node_ids.device)[None, :] < widths[owners, None]
    return MinibatchSample(
        decision_rows=rows.to(torch.int32),
        state_refs=state_refs,
        reach=reach,
        targets=gathered,
        old_logp_child=policy_cache.old_logp_child[rows].to(torch.float32),
        old_v=policy_cache.old_v[rows].to(torch.float32),
        valid_type_mask=valid,
        legal_event_ids=legal_ids,
        legal_event_mask=legal_mask,
    )


sample = sample_minibatch
