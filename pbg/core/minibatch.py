"""Decision-law minibatch sampling over stored rollout trees."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pbg.core.tree import AliasTable, PolicyCache, TreeBatch
from pbg.core.values import NodeTargets


@dataclass
class MinibatchSample:
    """Rows required to train the current policy at sampled decision nodes."""

    node_ids: Tensor
    targets: NodeTargets
    old_logp_child: Tensor
    old_v: Tensor
    state_refs: Tensor
    valid_type_mask: Tensor
    child_mask: Tensor


def _gather_targets(targets: NodeTargets, rows: Tensor) -> NodeTargets:
    return NodeTargets(
        node_ids=targets.node_ids[rows],
        p_hat=targets.p_hat[rows],
        q_hat=targets.q_hat[rows],
        v_hat=targets.v_hat[rows],
        vvec_hat=targets.vvec_hat[rows],
        u_hat=targets.u_hat[rows],
        nu_row=targets.nu_row[rows],
        child_event_row=targets.child_event_row[rows],
    )


def sample_minibatch(
    tree: TreeBatch,
    targets: NodeTargets,
    alias: AliasTable,
    policy_cache: PolicyCache,
    B: int,
    torch_gen: torch.Generator,
) -> MinibatchSample:
    """Sample player nodes with replacement using only stored path weights."""

    if B < 0:
        raise ValueError("B must be nonnegative")
    node_ids = alias.sample(B, torch_gen).to(torch.int32)
    target_lookup = {int(node): row for row, node in enumerate(targets.node_ids.tolist())}
    try:
        rows = torch.tensor(
            [target_lookup[int(node)] for node in node_ids.tolist()], dtype=torch.long,
            device=targets.p_hat.device,
        )
    except KeyError as exc:
        raise KeyError(f"node {int(exc.args[0])} has no NodeTargets row") from exc
    gathered = _gather_targets(targets, rows)
    old_logp, old_v = policy_cache.rows_for(node_ids)
    device = targets.p_hat.device
    old_logp = old_logp.to(device=device, dtype=torch.float32)
    old_v = old_v.to(device=device, dtype=torch.float32)
    refs = tree.state_ref[node_ids.to(tree.state_ref.device).long()].to(
        device=device, dtype=torch.int32
    )
    ptr = tree.child_ptr.to(torch.int64)
    sampled_on_tree = node_ids.to(ptr.device).long()
    child_counts = ptr[sampled_on_tree + 1] - ptr[sampled_on_tree]
    child_mask = torch.arange(gathered.nu_row.shape[1], device=device)[None, :] < child_counts.to(
        device
    )[:, None]
    # P10 zeros padded actor types.  A zero-posterior valid type has zero weight
    # in every rb_ppo term, so treating it as masked is exactly equivalent.
    valid_type_mask = gathered.p_hat != 0
    return MinibatchSample(
        node_ids=node_ids.to(device), targets=gathered,
        old_logp_child=old_logp, old_v=old_v, state_refs=refs,
        valid_type_mask=valid_type_mask, child_mask=child_mask,
    )


# Short spelling retained for callers following the older CONTRACTS document.
sample = sample_minibatch
