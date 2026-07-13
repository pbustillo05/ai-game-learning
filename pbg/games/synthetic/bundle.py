"""Synthetic game factory for the generic training boundary."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.config import RootConfig
from pbg.core.game import GameBundle
from pbg.games.synthetic.family import (
    SyntheticDialConfig,
    SyntheticPBG,
    SyntheticRolloutBackend,
)
from pbg.games.synthetic.model import SyntheticNet, SyntheticPolicyAdapter


class SyntheticCompletion:
    """Exact-state-table Monte Carlo completion used by synthetic smoke runs."""

    def __init__(self, family: SyntheticPBG) -> None:
        self.family = family

    def complete(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: Tensor,
        n_samples: int,
        gen: torch.Generator,
    ) -> BatchedCompletionStats:
        """Complete leaves into fp32 tensors ``[L,P,X]`` and ``[L,P]``."""

        del game
        if n_samples <= 0:
            raise ValueError("completion.n_samples must be positive")
        refs = tree.state_ref[node_ids.to(tree.state_ref.device).long()]
        profiles = self.family.sample_profiles(refs, reach, n_samples, gen)  # [L,S,P]
        leaves = int(node_ids.numel())
        players, x_max = self.family.n_players, max(self.family.x_sizes)
        p_hat = torch.zeros((leaves, players, x_max), dtype=torch.float32, device=refs.device)
        v_hat = torch.zeros_like(p_hat)
        vvec = torch.zeros((leaves, players), dtype=torch.float32, device=refs.device)
        u_hat = torch.zeros(leaves, dtype=torch.float32, device=refs.device)
        for row in range(leaves):
            rewards = self.family.reward(profiles[row], refs[row].repeat(n_samples)).float()
            vvec[row] = rewards.mean(0)
            u_hat[row] = rewards.square().sum(-1).mean()
            for player, width in enumerate(self.family.x_sizes):
                types = profiles[row, :, player].long()
                counts = torch.bincount(types, minlength=width).float()
                p_hat[row, player, :width] = counts / float(n_samples)
                present = (counts > 0).nonzero(as_tuple=False).squeeze(1)
                for xtype in present.tolist():
                    v_hat[row, player, xtype] = rewards[types == xtype, player].mean()
        return BatchedCompletionStats(
            p_hat=p_hat,
            v_hat=v_hat,
            vvec_hat=vvec,
            u_hat=u_hat,
            n_eff=torch.full((leaves,), float(n_samples), dtype=torch.float32, device=refs.device),
            metadata={"node_ids": node_ids.to(torch.int32)},
        )


def build_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Build the synthetic family, model, adapter, rollout, and completion bundle."""

    if cfg.completion.backend not in {"synthetic", "cpu", "auto"}:
        raise ValueError("synthetic completion backend must be 'synthetic', 'cpu', or 'auto'")
    values = dict(cfg.game.synthetic or {})
    values["board_per_round"] = tuple(values["board_per_round"])
    values["bet_size"] = tuple(values["bet_size"])
    family = SyntheticPBG(SyntheticDialConfig(**values))
    # The synthetic state table is the deliberately retained CPU oracle.
    backend_device = torch.device("cpu")
    model = SyntheticNet(max(family.x_sizes), family.n_player_events, cfg.model).to(backend_device)
    del device
    return GameBundle(
        model=model,
        rollout_backend=SyntheticRolloutBackend(family, cfg.rollout.beta_fix, backend_device),
        policy_adapter=SyntheticPolicyAdapter(),
        completion_backend=SyntheticCompletion(family),
    )
