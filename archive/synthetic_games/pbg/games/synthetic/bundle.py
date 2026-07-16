"""Synthetic game factory for the generic training boundary."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.config import RootConfig
from pbg.core.game import GameBundle
from pbg.core.queued import child_path_keys
from pbg.games.synthetic.family import (
    SyntheticDialConfig,
    SyntheticPBG,
    SyntheticRolloutBackend,
)
from pbg.games.synthetic.exact import exact_completion_stats, enumerate_public_tree
from pbg.games.synthetic.model import (
    SyntheticModelConfig,
    SyntheticMHANet,
    SyntheticNet,
    SyntheticPolicyAdapter,
)


class SyntheticMCCompletion:
    """Monte Carlo completion for synthetic games."""

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

        return self._complete_impl(
            game, tree, node_ids, reach, n_samples, gen=gen, keys=None
        )

    def complete_keyed(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: Tensor,
        n_samples: int,
        keys: Tensor,
    ) -> BatchedCompletionStats:
        """Complete leaves from one caller-owned logical stream per row."""

        return self._complete_impl(
            game, tree, node_ids, reach, n_samples, gen=None, keys=keys
        )

    def _complete_impl(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: Tensor,
        n_samples: int,
        *,
        gen: torch.Generator | None,
        keys: Tensor | None,
    ) -> BatchedCompletionStats:
        """Shared mutable/keyed synthetic MC completion implementation."""

        del game
        if n_samples <= 0:
            raise ValueError("completion.n_samples must be positive")
        refs = tree.state_ref[node_ids.to(tree.state_ref.device).long()]
        if keys is None:
            if gen is None:
                raise RuntimeError("mutable synthetic completion requires a generator")
            profiles = self.family.sample_profiles(
                refs, reach, n_samples, gen
            )  # [L,S,P]
        else:
            if keys.shape != node_ids.shape or keys.dtype != torch.int64:
                raise ValueError("completion keys must be int64 [L]")
            if keys.device != refs.device:
                raise ValueError("completion keys and leaves must share a device")
            ordinals = torch.arange(
                1, n_samples + 1, dtype=torch.int64, device=refs.device
            )
            sample_keys = child_path_keys(keys[:, None], ordinals[None, :])
            profiles = self.family.sample_profiles_keyed(
                refs, reach, sample_keys
            )  # [L,S,P]
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


class SyntheticExactCompletion:
    """Feasibility-guarded enumerative completion for synthetic games."""

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
        """Return exact finite-game posterior and conditional value rows."""

        del game, gen
        if n_samples <= 0:
            raise ValueError("completion.n_samples must be positive")
        refs = tree.state_ref[node_ids.to(tree.state_ref.device).long()]
        result = exact_completion_stats(self.family, refs, reach)
        result.metadata = {
            **result.metadata,
            "node_ids": node_ids.to(torch.int32),
        }
        return result


# Backward-compatible spelling for callers that explicitly requested MC.
SyntheticCompletion = SyntheticMCCompletion


def validate_config(cfg: RootConfig) -> None:
    """Validate synthetic-owned game and model mappings without building them."""

    values = dict(cfg.game.synthetic or {})
    unknown = sorted(
        set(values) - {field.name for field in fields(SyntheticDialConfig)}
    )
    if unknown:
        raise ValueError(
            f"root.game.synthetic.{unknown[0]}: unknown configuration key"
        )
    try:
        SyntheticDialConfig(**values)
        SyntheticModelConfig.from_dict(cfg.model)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"root synthetic configuration: {exc}") from exc


def build_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Build the synthetic family, model, adapter, rollout, and completion bundle."""

    if cfg.completion.backend not in {"exact", "synthetic", "cpu", "auto", "mc"}:
        raise ValueError(
            "synthetic completion backend must be 'exact', 'mc', 'synthetic', 'cpu', or 'auto'"
        )
    values = dict(cfg.game.synthetic or {})
    values["board_per_round"] = tuple(values["board_per_round"])
    values["bet_size"] = tuple(values["bet_size"])
    if values.get("raise_sizes") is not None:
        values["raise_sizes"] = tuple(tuple(row) for row in values["raise_sizes"])
    family = SyntheticPBG(SyntheticDialConfig(**values))
    tree_index = enumerate_public_tree(family)
    public_keys = tree_index.node_keys
    if len(set(public_keys.tolist())) != len(public_keys):
        raise RuntimeError("synthetic public-history encoding is not collision-free")
    required_capacity = int(public_keys.max(initial=-1)) + 1
    model_cfg = SyntheticModelConfig.from_dict(cfg.model)
    if model_cfg.n_buckets is not None and model_cfg.n_buckets < required_capacity:
        raise ValueError(
            "model.n_buckets is smaller than the reachable synthetic public-state "
            f"capacity ({model_cfg.n_buckets} < {required_capacity})"
        )
    model_values = {
        "architecture": model_cfg.architecture,
        "hidden_dim": model_cfg.hidden_dim,
        "n_buckets": max(model_cfg.n_buckets or 0, required_capacity),
        "n_heads": model_cfg.n_heads,
        "n_layers": model_cfg.n_layers,
    }
    # State-table lookup is a Python control plane; all boundary tensors and
    # neural computation live on the requested device.
    backend_device = torch.device(device)
    if model_cfg.architecture == "mha":
        incidence = torch.zeros(
            (family.n_types, family.cfg.deck_size), dtype=torch.bool
        )
        for xtype, hand in enumerate(family.hands):
            incidence[xtype, torch.tensor(hand, dtype=torch.long)] = True
        model = SyntheticMHANet(
            max(family.x_sizes),
            family.n_player_events,
            family.n_players,
            incidence,
            model_values,
        ).to(backend_device)
    else:
        model = SyntheticNet(
            max(family.x_sizes), family.n_player_events, model_values
        ).to(backend_device)
    return GameBundle(
        model=model,
        rollout_backend=SyntheticRolloutBackend(family, cfg.rollout.beta_fix, backend_device),
        policy_adapter=SyntheticPolicyAdapter(),
        completion_backend=(
            None
            if cfg.training.mode == "normal"
            else (
                SyntheticExactCompletion(family)
                if cfg.completion.backend == "exact"
                else SyntheticMCCompletion(family)
            )
        ),
    )
