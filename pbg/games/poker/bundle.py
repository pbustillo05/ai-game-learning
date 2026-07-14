"""Poker-owned construction of the generic training bundle."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.config import RootConfig
from pbg.core.game import PRETERM, GameBundle
from pbg.games.poker.adapter import PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.cards import N_HANDS
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
from pbg.games.poker.model.primitives import MHA
from pbg.games.poker.rollout_backend import PokerRolloutBackend
from pbg.games.poker.tensor_state import PokerTensorStateStore


@dataclass(frozen=True)
class PokerRolloutMemoryPlan:
    """Conservative byte plan for persistent and peak rollout allocations."""

    capacity: int
    state_bytes: int
    tree_bytes: int
    reach_cache_bytes: int
    frontier_reach_bytes: int
    policy_cache_bytes: int

    @property
    def peak_bytes(self) -> int:
        return sum(
            (
                self.state_bytes,
                self.tree_bytes,
                self.reach_cache_bytes,
                self.frontier_reach_bytes,
                self.policy_cache_bytes,
            )
        )


def estimate_rollout_memory(
    *, n_trees: int, node_budget: int, n_players: int, reach_storage: str
) -> PokerRolloutMemoryPlan:
    """Estimate worst-case rollout bytes before allocating a poker backend.

    The estimate intentionally counts two capacity-sized reach frontiers: a
    branched wave can retain the active rows while constructing its successors.
    It excludes model, completion, optimizer, and autograd allocations, so the
    CUDA preflight leaves half of currently free device memory for those phases.
    """

    if reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ValueError("unknown reach storage mode")
    if n_trees < 0 or node_budget <= 0 or n_players <= 0:
        raise ValueError("invalid rollout memory-plan dimensions")
    capacity = n_trees * node_budget
    edge_capacity = capacity - n_trees
    node_bytes = capacity * (6 * 4 + n_players * 4)
    edge_bytes = edge_capacity * 4 * 4
    reach_row_bytes = n_players * N_HANDS * 4
    reach_cache = (
        n_trees if reach_storage == "recompute" else capacity
    ) * reach_row_bytes
    return PokerRolloutMemoryPlan(
        capacity=capacity,
        state_bytes=PokerTensorStateStore.storage_nbytes(capacity, n_players),
        tree_bytes=node_bytes + edge_bytes,
        reach_cache_bytes=reach_cache,
        frontier_reach_bytes=2 * capacity * reach_row_bytes,
        # Selected policy likelihoods and values are retained only for realized
        # player edges/nodes, but this remains a safe capacity bound.
        policy_cache_bytes=(edge_capacity + capacity) * N_HANDS * 4,
    )


def _preflight_cuda_rollout(plan: PokerRolloutMemoryPlan, device: torch.device) -> None:
    """Reject a rollout plan that would crowd out model/optimizer allocations."""

    if device.type != "cuda":
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if plan.peak_bytes > free_bytes // 2:
        gib = 1024**3
        raise MemoryError(
            "poker rollout plan requires "
            f"{plan.peak_bytes / gib:.2f} GiB, exceeding half of the currently free "
            f"{free_bytes / gib:.2f} GiB on {device} (total {total_bytes / gib:.2f} GiB); "
            "reduce rollout.n_trees/node_budget or use a smaller storage plan"
        )


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


def validate_config(cfg: RootConfig) -> None:
    """Validate poker-owned game and model mappings without building them."""

    values = dict(cfg.game.poker or {})
    unknown = sorted(
        set(values) - {field.name for field in fields(PokerConfig) if field.init}
    )
    if unknown:
        raise ValueError(f"root.game.poker.{unknown[0]}: unknown configuration key")
    try:
        PokerConfig(**values)
        PokerModelConfig.from_dict(cfg.model)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"root poker configuration: {exc}") from exc


def build_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Construct one owning poker model and its device-resident backends."""

    engine_cfg = PokerConfig(**dict(cfg.game.poker or {}))
    model_cfg = PokerModelConfig.from_dict(cfg.model)
    if model_cfg.max_players < engine_cfg.n_players:
        raise ValueError("model.max_players must cover every configured poker seat")
    model = PokerModelBundle(PreflopNet(model_cfg), PostflopNet(model_cfg)).to(device)
    if cfg.hardware is not None:
        for module in model.modules():
            if isinstance(module, MHA):
                module.configure_execution(
                    cfg.hardware.attention_backend, cfg.hardware.query_chunk_size
                )
    adapter = PokerPolicyAdapter(engine_cfg.n_ticks)
    profiles = PokerGPURejectionSampler(backend=cfg.completion.backend)
    partition = PokerGPUPartitionAdapter(beta_rank=model_cfg.folder_beta_rank)
    plan = estimate_rollout_memory(
        n_trees=cfg.rollout.n_trees,
        node_budget=cfg.rollout.node_budget,
        n_players=engine_cfg.n_players,
        reach_storage=cfg.rollout.reach_storage,
    )
    _preflight_cuda_rollout(plan, device)
    backend = PokerRolloutBackend(
        engine_cfg,
        plan.capacity,
        device,
        beta_fix=cfg.rollout.beta_fix,
        profile_sampler=profiles,
        partition_adapter=partition,
        evaluator_backend=cfg.completion.backend,
    )
    backend.execution_profile = cfg.hardware
    backend.rollout_memory_plan = plan
    completion = (
        None
        if cfg.training.mode == "normal"
        else _TensorPokerGPUCompletion(backend=cfg.completion.backend)
    )
    return GameBundle(
        model=model,
        rollout_backend=backend,
        policy_adapter=adapter,
        completion_backend=completion,
    )
