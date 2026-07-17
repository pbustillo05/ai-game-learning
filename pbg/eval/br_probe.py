"""Paired GPU evaluation of an approximate best-response probe."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pbg.config import load_config_mapping
from pbg.eval.arena import _student_t_critical
from pbg.core.execution import DrawPurpose
from pbg.core.game import PROBE_FOLD, GameBundle
from pbg.core.rollout import (
    ProbeRollout,
    sample_trees_sequential,
    sequential_leaf_logical_rng_keys,
)
from pbg.games.poker.bundle import build_bundle, build_model_from_config


@dataclass(frozen=True)
class BRProbeReport:
    """Paired approximate-exploitability lower-bound report."""

    mean_gain_bb_per_hand: float
    ci95_bb_per_hand: float
    mean_gain_bb100: float
    ci95_bb100: float
    approximate_nashconv_bb_per_hand: float
    per_seat_gain_bb_per_hand: list[float]
    n_deals: int
    n_replays: int
    evaluation_completion_samples: int
    training_logical_trajectories: int
    training_logical_completion_samples: int
    training_logical_minibatches: int
    training_logical_optimizer_steps: int
    wall_seconds: float
    seed: int
    probe_sha256: str
    target_sha256: str

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_reward(
    bundle: GameBundle,
    cfg: Any,
    sampled: ProbeRollout | tuple[Any, Any, Any, Tensor],
    seats: Tensor,
    gen: torch.Generator,
) -> Tensor:
    if isinstance(sampled, ProbeRollout):
        tree, profiles = sampled.tree, sampled.root_profiles
    else:
        tree, _, _, profiles = sampled
    leaf = (tree.child_ptr[1:] == tree.child_ptr[:-1]).nonzero(
        as_tuple=False
    ).squeeze(1)
    leaf_tree = tree.tree_id[leaf].to(torch.int64)
    leaf_seats = seats[leaf_tree].to(torch.int64)
    result = torch.empty(seats.shape[0], dtype=torch.float32, device=seats.device)
    cutoff = tree.owner[leaf] == PROBE_FOLD
    if bool(cutoff.any()):
        positions = cutoff.nonzero(as_tuple=False).squeeze(1)
        states = tree.state_ref[leaf[positions]].to(torch.int64)
        result[leaf_tree[positions]] = -bundle.rollout_backend.states.committed_total[
            states, leaf_seats[positions]
        ]
    ordinary = ~cutoff
    if bool(ordinary.any()):
        positions = ordinary.nonzero(as_tuple=False).squeeze(1)
        selected_leaf = leaf[positions]
        keys = sequential_leaf_logical_rng_keys(
            tree,
            cfg,
            gen,
            leaf,
            iteration=0,
            purpose=DrawPurpose.COMPLETION,
            draw=1,
        )[positions]
        reward = bundle.rollout_backend.sampled_terminal_reward_keyed(
            tree.state_ref[selected_leaf], profiles[leaf_tree[positions]], keys
        )
        rows = torch.arange(positions.numel(), device=seats.device)
        result[leaf_tree[positions]] = reward[rows, leaf_seats[positions]]
    return result


def evaluate_br_probe(
    probe_checkpoint: str | Path,
    *,
    target_checkpoint: str | Path | None = None,
    hardware_profile: str | Path = "local_3x2080ti",
    n_hands: int,
    batch_size: int | None = None,
    seed: int = 1729,
    device: torch.device | str | None = None,
) -> BRProbeReport:
    """Evaluate one probe against its target using paired stochastic rollouts."""

    started = time.perf_counter()
    probe_path = Path(probe_checkpoint)
    probe_payload = torch.load(probe_path, map_location="cpu", weights_only=False)
    target_meta = probe_payload.get("probe_target")
    if not isinstance(target_meta, dict) or not target_meta.get("sha256"):
        raise ValueError("probe checkpoint has no hash-bound target metadata")
    target_path = Path(target_checkpoint or target_meta.get("path", ""))
    if not target_path.is_file():
        raise ValueError("target checkpoint path is unavailable; pass target_checkpoint")
    target_hash = _sha256(target_path)
    if target_hash != target_meta["sha256"]:
        raise ValueError("target checkpoint SHA-256 does not match the probe")
    target_payload = torch.load(target_path, map_location="cpu", weights_only=False)

    raw = copy.deepcopy(probe_payload["config"])
    raw.setdefault("training", {})["checkpoint_every"] = 1
    raw["probe"] = {
        **raw.get("probe", {}),
        "target_checkpoint": str(target_path),
        "target_sha256": target_hash,
    }
    cfg = load_config_mapping(raw, hardware_profile=hardware_profile)
    players = cfg.game.n_players
    block = players * players
    if n_hands <= 1 or n_hands % block:
        raise ValueError("n_hands must be a positive multiple of P^2")
    if batch_size is None:
        batch_size = min(n_hands, max(block, (256 // block) * block))
    if batch_size <= 0 or batch_size % block:
        raise ValueError("batch_size must be a positive multiple of P^2")
    requested = torch.device(
        device
        or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    cfg.rollout.sequential_starts = min(batch_size, n_hands)
    bundle = build_bundle(cfg, requested)
    bundle.model.load_state_dict(probe_payload["model"])
    bundle.model.eval()
    target = build_model_from_config(target_payload["config"], requested, cfg.hardware)
    target.load_state_dict(target_payload["model"])
    target.eval().requires_grad_(False)
    bundle.opponent_model = target

    gen = torch.Generator(device=requested).manual_seed(seed)
    differences: list[Tensor] = []
    seat_rows: list[Tensor] = []
    for first in range(0, n_hands, batch_size):
        count = min(batch_size, n_hands - first)
        ids = torch.arange(first, first + count, device=requested, dtype=torch.int64)
        seats = torch.remainder(ids // players, players).to(torch.int32)
        cfg.rollout.sequential_starts = count
        cfg.rollout.trajectory_start_index = first
        cfg.rollout.probe_seats = seats
        mixed = sample_trees_sequential(bundle, cfg, gen, iteration=0)
        if not isinstance(mixed, ProbeRollout):
            raise RuntimeError("mixed evaluation did not produce a probe rollout")
        mixed_reward = _selected_reward(bundle, cfg, mixed, seats, gen)

        baseline_bundle = GameBundle(
            model=target,
            rollout_backend=bundle.rollout_backend,
            policy_adapter=bundle.policy_adapter,
            completion_backend=bundle.completion_backend,
        )
        baseline = sample_trees_sequential(
            baseline_bundle, cfg, gen, return_profiles=True, iteration=0
        )
        baseline_reward = _selected_reward(
            baseline_bundle, cfg, baseline, seats, gen
        )
        differences.append((mixed_reward - baseline_reward).detach().cpu())
        seat_rows.append(seats.cpu())

    values = torch.cat(differences).double().numpy()
    all_seats = torch.cat(seat_rows).numpy()
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    ci95 = _student_t_critical(values.size - 1) * standard_error
    per_seat = [float(values[all_seats == seat].mean()) for seat in range(players)]
    logical_work = probe_payload.get("logical_work") or {}
    return BRProbeReport(
        mean_gain_bb_per_hand=mean,
        ci95_bb_per_hand=ci95,
        mean_gain_bb100=100.0 * mean,
        ci95_bb100=100.0 * ci95,
        approximate_nashconv_bb_per_hand=players * mean,
        per_seat_gain_bb_per_hand=per_seat,
        n_deals=n_hands,
        n_replays=2 * n_hands,
        evaluation_completion_samples=2 * n_hands,
        training_logical_trajectories=int(logical_work.get("trajectories", 0)),
        training_logical_completion_samples=int(
            logical_work.get("completion_samples", 0)
        ),
        training_logical_minibatches=int(logical_work.get("minibatches", 0)),
        training_logical_optimizer_steps=int(
            logical_work.get("optimizer_steps", 0)
        ),
        wall_seconds=time.perf_counter() - started,
        seed=seed,
        probe_sha256=_sha256(probe_path),
        target_sha256=target_hash,
    )
