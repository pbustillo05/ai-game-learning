"""Poker-owned construction of the generic training bundle."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.config import RootConfig
from pbg.core.game import PRETERM, GameBundle
from pbg.games.poker.adapter import PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.engine import PokerConfig
from pbg.games.poker.folded import PokerGPUPartitionAdapter
from pbg.games.poker.mc import (
    PokerGPUCompletion,
    PokerGPURejectionSampler,
    _CompletionConfig,
    _allocate_seeds,
)
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet
from pbg.games.poker.rollout_backend import PokerRolloutBackend


def _reach_rows(reach: Any) -> Tensor:
    """Return game-neutral log reach rows ``[B,P,1326]``."""

    if isinstance(reach, Tensor):
        return reach
    for name in ("log_r", "rows"):
        value = getattr(reach, name, None)
        if isinstance(value, Tensor):
            return value
    raise TypeError("reach must be a Tensor or expose Tensor log_r/rows")


class _TensorPokerGPUCompletion(PokerGPUCompletion):
    """Run sibling completion directly from tensor-native rollout state.

    Pre-terminal showdowns use ``poker_gpu_evaluator.pbg.complete``. Public
    fold terminals use exact sibling joint-profile draws to estimate their
    collision-conditioned hand posterior; their chip payoff is already fixed
    by the public engine.
    """

    def complete(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: Any,
        n_samples: int,
        gen: torch.Generator,
    ) -> BatchedCompletionStats:
        """Return fp32 completion rows ``[B,P,1326]`` without host tensor copies."""

        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        rows = _reach_rows(reach)
        state_refs = tree.state_ref[node_ids.to(tree.state_ref.device).long()]
        if rows.shape != (node_ids.numel(), game.n_players, 1326):
            raise ValueError("reach rows must have shape [B,P,1326]")
        batch, players = rows.shape[:2]
        device = rows.device
        p_hat = torch.zeros((batch, players, 1326), dtype=torch.float32, device=device)
        v_hat = torch.zeros_like(p_hat)
        vvec_hat = torch.zeros((batch, players), dtype=torch.float32, device=device)
        u_hat = torch.zeros(batch, dtype=torch.float32, device=device)
        n_eff = torch.zeros(batch, dtype=torch.float32, device=device)
        output_valid = torch.zeros_like(p_hat, dtype=torch.bool)
        sample_count = torch.full((batch,), n_samples, dtype=torch.int64, device=device)
        estimator_stddev = torch.full_like(p_hat, torch.nan)

        owner = tree.owner[node_ids.to(tree.owner.device).long()]
        preterm = (owner == PRETERM).nonzero(as_tuple=False).squeeze(1)
        metadata: dict[str, object] = {"backend": self.backend, "fold_completion": "joint_profiles"}
        if preterm.numel():
            states = state_refs[preterm]
            selected_reach = rows[preterm]
            inputs = game.completion_inputs(states, selected_reach)
            seed = _allocate_seeds(preterm.numel(), gen)
            result = self._operation()(
                **inputs,
                seed=seed,
                config=_CompletionConfig(n_samples, self.n_partials, self.backend),
            )
            stats = self._validate_result(
                result,
                preterm.numel(),
                players,
                inputs["stratum_amount"].shape[1],
                device,
            )
            p_hat[preterm] = stats.p_hat
            v_hat[preterm] = stats.v_hat
            vvec_hat[preterm] = stats.vvec_hat
            u_hat[preterm] = stats.u_hat
            n_eff[preterm] = stats.n_eff
            if stats.output_valid is not None:
                output_valid[preterm] = stats.output_valid
            if stats.sample_count is not None:
                sample_count[preterm] = stats.sample_count
            if stats.estimator_stddev is not None:
                estimator_stddev[preterm] = stats.estimator_stddev
            metadata.update({f"showdown_{key}": value for key, value in stats.metadata.items()})

        terminal = (owner != PRETERM).nonzero(as_tuple=False).squeeze(1)
        if terminal.numel():
            states = state_refs[terminal]
            selected_reach = rows[terminal]
            dealt = torch.ones(
                (terminal.numel(), players), dtype=torch.bool, device=device
            )  # [Bt,P]
            boards = game.states.board[states.to(torch.int64)]
            profiles = game.profile_sampler.sample(
                selected_reach, boards, dealt, n_samples, gen
            )  # [Bt,S,P]
            counts = torch.zeros(
                (terminal.numel(), players, 1326), dtype=torch.float32, device=device
            )  # [Bt,P,H]
            index = profiles.permute(0, 2, 1).to(torch.int64)  # [Bt,P,S]
            counts.scatter_add_(2, index, torch.ones_like(index, dtype=torch.float32))
            posterior = counts / float(n_samples)
            reward = game.reward(profiles[:, 0], states).float()  # [Bt,P]
            p_hat[terminal] = posterior
            v_hat[terminal] = reward[:, :, None] * (posterior > 0).to(torch.float32)
            vvec_hat[terminal] = reward
            u_hat[terminal] = reward.square().sum(1)
            n_eff[terminal] = float(n_samples)
            output_valid[terminal] = posterior > 0

        return BatchedCompletionStats(
            p_hat=p_hat,
            v_hat=v_hat,
            vvec_hat=vvec_hat,
            u_hat=u_hat,
            n_eff=n_eff,
            output_valid=output_valid,
            sample_count=sample_count,
            estimator_stddev=estimator_stddev,
            metadata=metadata,
        )


def build_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Construct one owning poker model and its device-resident backends."""

    engine_cfg = PokerConfig(**dict(cfg.game.poker or {}))
    model_cfg = PokerModelConfig.from_dict(cfg.model)
    if model_cfg.max_players < engine_cfg.n_players:
        raise ValueError("model.max_players must cover every configured poker seat")
    model = PokerModelBundle(PreflopNet(model_cfg), PostflopNet(model_cfg)).to(device)
    adapter = PokerPolicyAdapter(engine_cfg.n_ticks)
    profiles = PokerGPURejectionSampler(backend=cfg.completion.backend)
    partition = PokerGPUPartitionAdapter()
    capacity = cfg.rollout.n_trees * cfg.rollout.node_budget
    backend = PokerRolloutBackend(
        engine_cfg,
        capacity,
        device,
        beta_fix=cfg.rollout.beta_fix,
        profile_sampler=profiles,
        partition_adapter=partition,
        evaluator_backend=cfg.completion.backend,
    )
    completion = _TensorPokerGPUCompletion(backend=cfg.completion.backend)
    return GameBundle(
        model=model,
        rollout_backend=backend,
        policy_adapter=adapter,
        completion_backend=completion,
    )
