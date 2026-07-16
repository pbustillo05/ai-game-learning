"""Production NLHE bundle construction and training entry points."""

from __future__ import annotations

from pathlib import Path

import torch

from pbg.core.config import RootConfig
from pbg.core.game import GameBundle
from pbg.core.training import run_training
from pbg.games.poker.bundle import build_bundle


def build_game_bundle(cfg: RootConfig, device: torch.device) -> GameBundle:
    """Construct the sole active NLHE model and runtime bundle."""

    bundle = build_bundle(cfg, device)
    if not isinstance(bundle, GameBundle):
        raise TypeError("NLHE bundle builder must return GameBundle")
    return bundle


def train(cfg: RootConfig, *, resume_from: str | Path | None = None) -> None:
    """Train the production NLHE implementation from scratch or checkpoint."""

    run_training(cfg, bundle_builder=build_game_bundle, resume_from=resume_from)


__all__ = ["build_game_bundle", "train"]
