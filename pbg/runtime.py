"""Production NLHE bundle construction and training entry points."""

from __future__ import annotations

from pathlib import Path

import torch

from pbg.core.config import RootConfig
from pbg.core.game import GameBundle
from pbg.core.training import run_training
from pbg.games.poker.bundle import build_bundle, build_model_from_config


def build_game_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Construct the sole active NLHE model and runtime bundle."""

    bundle = build_bundle(cfg, device)
    if not isinstance(bundle, GameBundle):
        raise TypeError("NLHE bundle builder must return GameBundle")
    return bundle


def train(
    cfg: RootConfig,
    *,
    resume_from: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
) -> None:
    """Train from scratch, initialize from model weights, or resume."""

    run_training(
        cfg,
        bundle_builder=build_game_bundle,
        target_model_builder=build_model_from_config,
        resume_from=resume_from,
        init_checkpoint=init_checkpoint,
    )


__all__ = ["build_game_bundle", "train"]
