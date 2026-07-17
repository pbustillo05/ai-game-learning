"""Poker-owned construction of the generic training bundle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.config import RootConfig
from pbg.core.game import PRETERM, PROBE_FOLD, GameBundle
from pbg.games.poker.adapter import PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.engine import PokerConfig
from pbg.games.poker.folded import PokerGPUPartitionAdapter
from pbg.games.poker.mc import (
    PokerGPUCompletion,
    PokerGPURejectionSampler,
    _CompletionConfig,
    _allocate_seeds,
)
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.capacity import (
    max_policy_sequence_tokens,
    max_public_history_tokens,
)
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet
from pbg.games.poker.model.primitives import MHA
from pbg.games.poker.partition_cache import DevicePartitionSummaryCache
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
    partition_cache_bytes: int

    @property
    def peak_bytes(self) -> int:
        return sum(
            (
                self.state_bytes,
                self.tree_bytes,
                self.reach_cache_bytes,
                self.frontier_reach_bytes,
                self.policy_cache_bytes,
                self.partition_cache_bytes,
            )
        )


def estimate_rollout_memory(
    *,
    root_count: int,
    node_budget: int,
    n_players: int,
    reach_storage: str,
    rank_count: int = 13,
    sequential_recompute_live_rows: bool = False,
    max_history_tokens: int = 1,
    folder_beta_rank: int = 16,
) -> PokerRolloutMemoryPlan:
    """Estimate worst-case rollout bytes before allocating a poker backend.

    The general estimate intentionally counts two capacity-sized reach
    frontiers: a branched wave can retain active rows while constructing its
    successors. Sequential execution in recompute mode is tighter: exactly one
    live occurrence exists per trajectory and execution allocates root and
    overwriteable working reaches with ``root_count`` rows each. It excludes model,
    completion, optimizer, and autograd allocations, so the CUDA preflight
    leaves half of currently free device memory for those phases.
    """

    if reach_storage not in {"full", "decisions_only", "recompute"}:
        raise ValueError("unknown reach storage mode")
    if root_count < 0 or node_budget <= 0 or n_players <= 0:
        raise ValueError("invalid rollout memory-plan dimensions")
    if sequential_recompute_live_rows and reach_storage != "recompute":
        raise ValueError("sequential live-row planning requires recompute storage")
    capacity = root_count * node_budget
    edge_capacity = capacity - root_count
    node_bytes = capacity * (6 * 4 + n_players * 4)
    edge_bytes = edge_capacity * 4 * 4
    n_hands = DeckGeometry(rank_count).n_hands
    reach_row_bytes = n_players * n_hands * 4
    reach_cache = (
        root_count if reach_storage == "recompute" else capacity
    ) * reach_row_bytes
    return PokerRolloutMemoryPlan(
        capacity=capacity,
        state_bytes=PokerTensorStateStore.storage_nbytes(
            capacity, n_players, max_history_tokens
        ),
        tree_bytes=node_bytes + edge_bytes,
        reach_cache_bytes=reach_cache,
        frontier_reach_bytes=(
            2 * root_count * reach_row_bytes
            if sequential_recompute_live_rows
            else 2 * capacity * reach_row_bytes
        ),
        # Selected policy likelihoods and values are retained only for realized
        # player edges/nodes, but this remains a safe capacity bound.
        policy_cache_bytes=(edge_capacity + capacity) * n_hands * 4,
        partition_cache_bytes=DevicePartitionSummaryCache.storage_nbytes(
            capacity,
            n_cards=4 * rank_count,
            beta_rank=folder_beta_rank,
            max_folders=max(0, n_players - 2),
        ),
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
            "reduce rollout.sequential_starts/node_budget or use a smaller storage plan"
        )


def _reach_rows(reach: Any) -> Tensor:
    """Return game-neutral log reach rows ``[B,P,H]``."""

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
        """Return fp32 completion rows ``[B,P,H]`` without host tensor copies."""

        return self._complete_impl(
            game, tree, node_ids, reach, n_samples, gen=gen, keys=None
        )

    def complete_keyed(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: Any,
        n_samples: int,
        keys: Tensor,
    ) -> BatchedCompletionStats:
        """Complete tensor-native leaves from logical-address seeds ``[B]``."""

        return self._complete_impl(
            game, tree, node_ids, reach, n_samples, gen=None, keys=keys
        )

    def complete_selected_keyed(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: Any,
        n_samples: int,
        keys: Tensor,
        players: Tensor,
    ) -> BatchedCompletionStats:
        """Complete only one selected player's posterior/value per leaf."""

        rows = _reach_rows(reach)
        if players.shape != node_ids.shape or players.device != rows.device:
            raise ValueError("selected completion players must be device-local [B]")
        p_hat = torch.zeros_like(rows, dtype=torch.float32)
        v_hat = torch.zeros_like(p_hat)
        owners = tree.owner[node_ids.to(tree.owner.device, dtype=torch.int64)]
        cutoff_mask = owners == PROBE_FOLD
        ordinary_mask = ~cutoff_mask

        if bool(ordinary_mask.any()):
            positions = ordinary_mask.nonzero(as_tuple=False).squeeze(1)
            completed = self._complete_impl(
                game,
                tree,
                node_ids[positions],
                rows[positions],
                n_samples,
                gen=None,
                keys=keys[positions],
            )
            selected = players[positions].to(torch.int64)
            target_rows = torch.arange(positions.numel(), device=rows.device)
            p_hat[positions, selected] = completed.p_hat[target_rows, selected]
            v_hat[positions, selected] = completed.v_hat[target_rows, selected]

        if bool(cutoff_mask.any()):
            positions = cutoff_mask.nonzero(as_tuple=False).squeeze(1)
            state_refs = tree.state_ref[
                node_ids[positions].to(tree.state_ref.device, dtype=torch.int64)
            ]
            selected_reach = rows[positions]
            dealt = torch.ones(
                (positions.numel(), game.n_players),
                dtype=torch.bool,
                device=rows.device,
            )
            profiles = game.profile_sampler.sample_seeded(
                selected_reach,
                game.states.board[state_refs.to(torch.int64)],
                dealt,
                n_samples,
                keys[positions].contiguous(),
            )
            selected = players[positions].to(torch.int64)
            combos = profiles.gather(
                2, selected[:, None, None].expand(-1, n_samples, 1)
            ).squeeze(2).to(torch.int64)
            counts = torch.zeros(
                (positions.numel(), rows.shape[2]),
                dtype=torch.float32,
                device=rows.device,
            )
            counts.scatter_add_(1, combos, torch.ones_like(combos, dtype=torch.float32))
            posterior = counts / float(n_samples)
            payoff = -game.states.committed_total[
                state_refs.to(torch.int64), selected
            ].to(torch.float32)
            p_hat[positions, selected] = posterior
            v_hat[positions, selected] = payoff[:, None] * (posterior > 0)

        return BatchedCompletionStats(
            p_hat=p_hat,
            v_hat=v_hat,
            metadata={"backend": self.backend, "selected_player": True},
        )

    def _complete_impl(
        self,
        game: Any,
        tree: Any,
        node_ids: Tensor,
        reach: Any,
        n_samples: int,
        *,
        gen: torch.Generator | None,
        keys: Tensor | None,
    ) -> BatchedCompletionStats:
        """Shared mutable/keyed tensor-native completion implementation."""

        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        rows = _reach_rows(reach)
        state_refs = tree.state_ref[node_ids.to(tree.state_ref.device).long()]
        n_hands = DeckGeometry(self.rank_count).n_hands
        if rows.shape != (node_ids.numel(), game.n_players, n_hands):
            raise ValueError(
                f"reach rows must have shape [B,P,{n_hands}]"
            )
        batch, players = rows.shape[:2]
        device = rows.device
        if keys is not None and (
            keys.shape != node_ids.shape
            or keys.dtype != torch.int64
            or keys.device != device
        ):
            raise ValueError("completion keys must be device-local int64 [B]")
        if keys is None and gen is None:
            raise RuntimeError("mutable poker completion requires a generator")
        p_hat = torch.zeros(
            (batch, players, n_hands), dtype=torch.float32, device=device
        )
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
            seed = (
                _allocate_seeds(preterm.numel(), gen)
                if keys is None
                else keys[preterm].contiguous()
            )
            result = self._operation()(
                rank_count=self.rank_count,
                **inputs,
                seed=seed,
                config=_CompletionConfig(n_samples, self.n_partials, self.backend),
            )
            metrics = getattr(game, "execution_metrics", None)
            if metrics is not None:
                metrics.record_evaluator(rows=int(preterm.numel()))
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
            if keys is None:
                profiles = game.profile_sampler.sample(
                    selected_reach, boards, dealt, n_samples, gen
                )  # [Bt,S,P]
            else:
                seeded = getattr(game.profile_sampler, "sample_seeded", None)
                if not callable(seeded):
                    raise TypeError(
                        "keyed poker completion requires a seeded joint-profile sampler"
                    )
                profiles = seeded(
                    selected_reach,
                    boards,
                    dealt,
                    n_samples,
                    keys[terminal].contiguous(),
                )  # [Bt,S,P]
            metrics = getattr(game, "execution_metrics", None)
            if metrics is not None:
                metrics.record_evaluator(rows=int(terminal.numel()))
            counts = torch.zeros(
                (terminal.numel(), players, n_hands),
                dtype=torch.float32,
                device=device,
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


def _resolved_model_config(
    cfg: RootConfig, poker_cfg: PokerConfig
) -> PokerModelConfig:
    """Return a model config with the lossless history capacity resolved."""

    model_cfg = PokerModelConfig.from_dict(cfg.model)
    if model_cfg.max_players < poker_cfg.n_players:
        raise ValueError(
            "root.model.max_players: must cover every configured poker seat"
        )
    if model_cfg.folder_beta_rank > poker_cfg.geometry.n_cards:
        raise ValueError(
            "root.model.folder_beta_rank: cannot exceed the active compact "
            f"deck size ({poker_cfg.geometry.n_cards})"
        )
    required_history = max_public_history_tokens(
        poker_cfg.n_players, poker_cfg.stack_bb
    )
    if (
        "max_history_tokens" in cfg.model
        and model_cfg.max_history_tokens < required_history
    ):
        raise ValueError(
            "root.model.max_history_tokens: explicitly configured capacity "
            f"{model_cfg.max_history_tokens} is smaller than the lossless "
            f"(P,S) requirement {required_history}"
        )
    if "max_history_tokens" not in cfg.model:
        model_cfg = replace(model_cfg, max_history_tokens=required_history)
    return model_cfg


def _validate_attention_capacity(
    cfg: RootConfig, poker_cfg: PokerConfig, model_cfg: PokerModelConfig
) -> None:
    profile = cfg.hardware
    if profile is None:
        return
    width = max_policy_sequence_tokens(
        n_players=poker_cfg.n_players,
        stack_bb=poker_cfg.stack_bb,
        rank_count=poker_cfg.rank_count,
        k_act=model_cfg.k_act,
    )
    pairs = width * width
    for phase in ("training", "inference"):
        token_limit = getattr(profile, f"{phase}_max_tokens")
        pair_limit = getattr(profile, f"{phase}_max_token_pairs")
        if width > token_limit:
            raise ValueError(
                f"hardware.{phase}_max_tokens: {token_limit} cannot accommodate "
                f"the worst legal one-row sequence ({width})"
            )
        if pairs > pair_limit:
            raise ValueError(
                f"hardware.{phase}_max_token_pairs: {pair_limit} cannot accommodate "
                f"the worst legal one-row sequence ({pairs} pairs)"
            )


def validate_config(cfg: RootConfig) -> None:
    """Validate poker-owned game and model mappings without building them."""

    try:
        poker_cfg = PokerConfig(
            n_players=cfg.game.n_players,
            stack_bb=cfg.game.stack_bb,
            rank_count=cfg.game.rank_count,
        )
        model_cfg = _resolved_model_config(cfg, poker_cfg)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if message.startswith("root.model."):
            raise ValueError(message) from exc
        field = (
            "n_players" if "n_players" in message
            else "stack_bb" if "stack_bb" in message
            else "rank_count" if "rank_count" in message
            else "rank_count" if "compact deck" in message
            else "configuration"
        )
        raise ValueError(f"root.game.{field}: {message}") from exc
    # Active poker has one preflop family plus post_A for every possible live
    # count A=2..P, hence P potentially nonempty learner strata.
    if cfg.training.minibatch_size < poker_cfg.n_players:
        raise ValueError(
            "root.training.minibatch_size: must cover every potentially nonempty "
            f"poker execution family ({poker_cfg.n_players})"
        )
    _validate_attention_capacity(cfg, poker_cfg, model_cfg)


def build_model_from_config(
    scientific: dict[str, Any], device: torch.device, hardware: Any | None = None
) -> PokerModelBundle:
    """Construct only a poker model from checkpoint scientific configuration."""

    game = scientific.get("game")
    model = scientific.get("model")
    if not isinstance(game, dict) or not isinstance(model, dict):
        raise ValueError("target checkpoint must contain game and model configuration")
    engine_cfg = PokerConfig(
        n_players=game["n_players"],
        stack_bb=game["stack_bb"],
        rank_count=game["rank_count"],
    )
    model_cfg = PokerModelConfig.from_dict(model)
    if model_cfg.max_players < engine_cfg.n_players:
        raise ValueError("target model does not cover every configured poker seat")
    if model_cfg.folder_beta_rank > engine_cfg.geometry.n_cards:
        raise ValueError("target folder beta rank exceeds the active compact deck")
    required = max_public_history_tokens(engine_cfg.n_players, engine_cfg.stack_bb)
    if "max_history_tokens" not in model:
        model_cfg = replace(model_cfg, max_history_tokens=required)
    if model_cfg.max_history_tokens < required:
        raise ValueError("target model history capacity is smaller than its game bound")
    if hardware is not None:
        width = max_policy_sequence_tokens(
            n_players=engine_cfg.n_players,
            stack_bb=engine_cfg.stack_bb,
            rank_count=engine_cfg.rank_count,
            k_act=model_cfg.k_act,
        )
        pairs = width * width
        for phase in ("training", "inference"):
            if width > getattr(hardware, f"{phase}_max_tokens"):
                raise ValueError(
                    f"target model exceeds hardware.{phase}_max_tokens"
                )
            if pairs > getattr(hardware, f"{phase}_max_token_pairs"):
                raise ValueError(
                    f"target model exceeds hardware.{phase}_max_token_pairs"
                )
    result = PokerModelBundle(
        PreflopNet(model_cfg, engine_cfg.geometry),
        PostflopNet(model_cfg, engine_cfg.geometry),
    ).to(device)
    if hardware is not None:
        for module in result.modules():
            if isinstance(module, MHA):
                module.configure_execution(
                    hardware.attention_backend, hardware.query_chunk_size
                )
    return result


def build_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Construct one owning poker model and its device-resident backends."""

    engine_cfg = PokerConfig(
        n_players=cfg.game.n_players,
        stack_bb=cfg.game.stack_bb,
        rank_count=cfg.game.rank_count,
    )
    model_cfg = _resolved_model_config(cfg, engine_cfg)
    _validate_attention_capacity(cfg, engine_cfg, model_cfg)
    model = PokerModelBundle(
        PreflopNet(model_cfg, engine_cfg.geometry),
        PostflopNet(model_cfg, engine_cfg.geometry),
    ).to(device)
    if cfg.hardware is not None:
        for module in model.modules():
            if isinstance(module, MHA):
                module.configure_execution(
                    cfg.hardware.attention_backend, cfg.hardware.query_chunk_size
                )
    adapter = PokerPolicyAdapter(engine_cfg.n_ticks)
    profiles = PokerGPURejectionSampler(
        backend=cfg.completion.backend, rank_count=engine_cfg.rank_count
    )
    partition = PokerGPUPartitionAdapter(
        beta_rank=model_cfg.folder_beta_rank, rank_count=engine_cfg.rank_count
    )
    resident_trees = cfg.rollout.sequential_starts
    plan = estimate_rollout_memory(
        root_count=resident_trees,
        node_budget=cfg.rollout.node_budget,
        n_players=engine_cfg.n_players,
        reach_storage=cfg.rollout.reach_storage,
        rank_count=engine_cfg.rank_count,
        sequential_recompute_live_rows=(
            cfg.training.mode != "branched_rao_blackwell"
            and cfg.rollout.reach_storage == "recompute"
        ),
        max_history_tokens=model_cfg.max_history_tokens,
        folder_beta_rank=model_cfg.folder_beta_rank,
    )
    _preflight_cuda_rollout(plan, device)
    backend = PokerRolloutBackend(
        engine_cfg,
        plan.capacity,
        device,
        beta_table=cfg.rollout.beta or {"default": 1.0},
        profile_sampler=profiles,
        partition_adapter=partition,
        evaluator_backend=cfg.completion.backend,
        max_rollout_steps=model_cfg.max_history_tokens,
    )
    backend.execution_profile = cfg.hardware
    backend.rollout_memory_plan = plan
    backend.register_policy_model(model)
    completion = (
        None
        if cfg.training.mode == "normal"
        else _TensorPokerGPUCompletion(
            backend=cfg.completion.backend, rank_count=engine_cfg.rank_count
        )
    )
    return GameBundle(
        model=model,
        rollout_backend=backend,
        policy_adapter=adapter,
        completion_backend=completion,
    )
