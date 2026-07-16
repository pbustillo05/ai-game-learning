"""Paired duplicate-deal checkpoint arenas driven by the CPU NLHE engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pbg.core.config import RootConfig
from pbg.core.game import CHANCE, PRETERM, TERMINAL
from pbg.games.poker.adapter import PokerGame, PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.cards import (
    N_HANDS,
    board_conflict_mask,
    hand_cards,
    sample_uniform_complement,
)
from pbg.games.poker.engine import FOLDED, NLHEEngine, PokerConfig
from pbg.games.poker.mc import PokerGPURejectionSampler, PokerGPUShowdownEvaluator
from pbg.games.poker.model.postflop import PostflopMasks, PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet


@dataclass
class ArenaReport:
    """Paired performance summary in big blinds per 100 physical hands."""

    bb100_mean: float
    bb100_ci95: float
    n_hands: int
    per_seat_bb100: list[float]
    seat_pairing: str = "duplicate"


def _poker_config(cfg: RootConfig) -> PokerConfig:
    return PokerConfig(
        n_players=cfg.game.n_players,
        stack_bb=cfg.game.stack_bb,
        rank_count=cfg.game.rank_count,
    )


def _build_model(cfg: RootConfig) -> PokerModelBundle:
    model_cfg = PokerModelConfig.from_dict(cfg.model)
    return PokerModelBundle(PreflopNet(model_cfg), PostflopNet(model_cfg)).cpu().eval()


def _load_model(path: str | Path, cfg: RootConfig) -> PokerModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint {path!s} does not contain a model state")
    model = _build_model(cfg)
    model.load_state_dict(payload["model"])
    return model


def _profile_runouts(
    n_pairs: int,
    n_players: int,
    torch_gen: torch.Generator,
) -> tuple[Tensor, Tensor]:
    sampler = PokerGPURejectionSampler(backend="cpu")
    log_reach = torch.zeros((1, n_players, N_HANDS), dtype=torch.float32)
    board = torch.full((1, 5), -1, dtype=torch.int16)
    dealt = torch.ones((1, n_players), dtype=torch.bool)
    profiles = sampler.sample(log_reach, board, dealt, n_pairs, torch_gen)[0]  # [D,P]
    unavailable = torch.tensor(
        [
            [card for combo in profile.tolist() for card in hand_cards(int(combo))]
            for profile in profiles
        ],
        dtype=torch.int16,
    )  # [D,2P]
    counts = torch.full((n_pairs,), 5, dtype=torch.int32)
    runouts = sample_uniform_complement(unavailable, counts, torch_gen)  # [D,5]
    return profiles, runouts


def _linear_ranges(log_reach: Tensor, board: tuple[int, ...]) -> Tensor:
    shifted = log_reach - log_reach.amax(dim=1, keepdim=True)
    ranges = torch.exp(shifted)
    padded = torch.tensor(board + (-1,) * (5 - len(board)), dtype=torch.int16)
    compatible = ~board_conflict_mask(padded)
    return ranges.masked_fill(~compatible[None, :], 0.0)


def _policy_probabilities(
    model: PokerModelBundle,
    game: PokerGame,
    adapter: PokerPolicyAdapter,
    ref: int,
    log_reach: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    refs = torch.tensor([ref], dtype=torch.int32)
    public = game.encode_public(refs)
    ranges = _linear_ranges(log_reach, game.engine.states.state(ref).board)[None]  # [1,P,H]
    postflop_masks = None
    if game.engine.states.street[ref] != 0:
        compatible = ~board_conflict_mask(public.board.to(torch.int16))
        street = game.engine.states.street[ref]
        postflop_masks = PostflopMasks(
            compatibility=compatible,
            compact_size=math.comb(52 - (street + 2), 2),
        )
    with torch.no_grad():
        output = model(public, ranges, postflop_masks=postflop_masks)
        legal_ids, legal_mask = game.legal_events(refs)
        probabilities = adapter.probabilities(output, legal_ids, legal_mask)  # [1,H,E]
    return probabilities[0], legal_ids[0], legal_mask[0]


def _uniform_for(
    uniforms: dict[tuple[Any, ...], float],
    key: tuple[Any, ...],
    torch_gen: torch.Generator,
) -> float:
    value = uniforms.get(key)
    if value is None:
        value = float(torch.rand((), generator=torch_gen))
        uniforms[key] = value
    return value


def _draw_event(
    probability: Tensor,
    legal_ids: Tensor,
    legal_mask: Tensor,
    uniform: float,
    argmax_actions: bool,
) -> int:
    legal_columns = legal_mask.nonzero(as_tuple=False).squeeze(1)
    row = probability[legal_columns]
    if argmax_actions:
        column = int(torch.argmax(row))
    else:
        column = int(torch.searchsorted(row.cumsum(0), torch.tensor(uniform), right=False))
        column = min(column, int(row.numel()) - 1)
    return int(legal_ids[legal_columns[column]])


def _settle_showdown(
    engine: NLHEEngine,
    ref: int,
    profile: Tensor,
    runout: Tensor,
    evaluator: PokerGPUShowdownEvaluator,
) -> Tensor:
    state = engine.states.state(ref)
    board = torch.tensor(state.board, dtype=torch.uint8)
    if board.numel() < 5:
        board = runout[:5].to(torch.uint8)
    board = board[None].contiguous()  # [1,5]
    status = torch.tensor(state.status, dtype=torch.long)[None]
    active = status != FOLDED
    combos = profile.to(torch.int32)[None].clone()
    combos.masked_fill_(~active, -1)
    strata = engine.pot_strata(torch.tensor([ref], dtype=torch.int32))
    result = evaluator.evaluate(
        board_cards=board,
        combo_ids=combos,
        active_mask=active,
        stratum_players=strata.stratum_players,
    )
    return engine.assemble_showdown_payoff(strata, result.showdown_share)[0]


def _replay(
    model_by_seat: list[PokerModelBundle],
    engine_cfg: PokerConfig,
    profile: Tensor,
    runout: Tensor,
    uniforms: dict[tuple[Any, ...], float],
    torch_gen: torch.Generator,
    *,
    argmax_actions: bool,
) -> Tensor:
    engine = NLHEEngine(engine_cfg)
    evaluator = PokerGPUShowdownEvaluator(backend="cpu")
    game = PokerGame(
        engine,
        profile_sampler=object(),
        showdown_evaluator=evaluator,
        partition_adapter=object(),
    )
    adapter = PokerPolicyAdapter(n_ticks=engine_cfg.n_ticks)
    ref = engine.new_hand(0)
    log_reach = torch.zeros((engine_cfg.n_players, N_HANDS), dtype=torch.float32)
    runout_offset = 0
    while True:
        owner = int(engine.owner_of(torch.tensor([ref], dtype=torch.int32))[0])
        if owner == TERMINAL:
            return engine.reward(profile[None], torch.tensor([ref], dtype=torch.int32))[0]
        if owner == PRETERM:
            return _settle_showdown(engine, ref, profile, runout, evaluator)
        if owner == CHANCE:
            count = 3 if engine.states.street[ref] == 0 else 1
            cards = runout[runout_offset : runout_offset + count]
            runout_offset += count
            ref = int(engine.deal(torch.tensor([ref], dtype=torch.int32), cards[None])[0])
            continue
        probability, legal_ids, legal_mask = _policy_probabilities(
            model_by_seat[owner], game, adapter, ref, log_reach
        )
        actor_combo = int(profile[owner])
        state = engine.states.state(ref)
        key = (owner, state.street, state.board, state.action_history)
        uniform = _uniform_for(uniforms, key, torch_gen)
        event = _draw_event(
            probability[actor_combo], legal_ids, legal_mask, uniform, argmax_actions
        )
        selected = probability[:, event]
        log_reach[owner] += torch.log(selected.clamp_min(adapter.eps))
        log_reach[owner] -= log_reach[owner].amax()
        ref = int(
            engine.step(
                torch.tensor([ref], dtype=torch.int32),
                torch.tensor([event], dtype=torch.int32),
            )[0]
        )


_T975 = (
    0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
    2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
    2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
)


def _student_t_critical(df: int) -> float:
    if df <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if df < len(_T975):
        return _T975[df]
    z = 1.959963984540054
    inverse = 1.0 / df
    return z + (z**3 + z) * inverse / 4.0 + (
        5 * z**5 + 16 * z**3 + 3 * z
    ) * inverse**2 / 96.0


def arena(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    cfg: RootConfig,
    n_hands: int,
    torch_gen: torch.Generator,
    np_gen: np.random.Generator,
    *,
    argmax_actions: bool = False,
) -> ArenaReport:
    """Evaluate A against B using paired deals and return a paired t interval."""

    engine_cfg = _poker_config(cfg)
    if isinstance(n_hands, bool) or not isinstance(n_hands, int) or n_hands <= 0:
        raise ValueError("n_hands must be a positive integer")
    if n_hands % (2 * engine_cfg.n_players):
        raise ValueError("n_hands must be a positive multiple of 2*n_players")
    if not isinstance(torch_gen, torch.Generator) or torch_gen.device.type != "cpu":
        raise TypeError("arena requires a CPU torch.Generator")
    if not isinstance(np_gen, np.random.Generator):
        raise TypeError("np_gen must be a numpy.random.Generator")
    model_a = _load_model(ckpt_a, cfg)
    model_b = _load_model(ckpt_b, cfg)
    n_pairs = n_hands // 2
    profiles, runouts = _profile_runouts(n_pairs, engine_cfg.n_players, torch_gen)
    observations: list[float] = []
    by_seat: list[list[float]] = [[] for _ in range(engine_cfg.n_players)]
    for pair in range(n_pairs):
        singled = pair % engine_cfg.n_players
        uniforms: dict[tuple[Any, ...], float] = {}
        ab_models = [model_b] * engine_cfg.n_players
        ab_models[singled] = model_a
        ba_models = [model_a] * engine_cfg.n_players
        ba_models[singled] = model_b
        payoff_ab = _replay(
            ab_models, engine_cfg, profiles[pair], runouts[pair], uniforms,
            torch_gen, argmax_actions=argmax_actions,
        )
        payoff_ba = _replay(
            ba_models, engine_cfg, profiles[pair], runouts[pair], uniforms,
            torch_gen, argmax_actions=argmax_actions,
        )
        if abs(float(payoff_ab.sum())) > 2e-4 or abs(float(payoff_ba.sum())) > 2e-4:
            raise ValueError("each physical arena replay must conserve chips")
        difference = float(payoff_ab[singled] - payoff_ba[singled])
        value = 100.0 * difference / (2.0 * engine_cfg.bb)
        observations.append(value)
        by_seat[singled].append(value)
    values = np.asarray(observations, dtype=np.float64)
    mean = float(values.mean())
    if n_pairs > 1:
        standard_error = float(values.std(ddof=1) / np.sqrt(n_pairs))
        ci95 = _student_t_critical(n_pairs - 1) * standard_error
    else:
        ci95 = 0.0
    return ArenaReport(
        bb100_mean=mean,
        bb100_ci95=float(ci95),
        n_hands=n_hands,
        per_seat_bb100=[float(np.mean(row)) for row in by_seat],
    )
