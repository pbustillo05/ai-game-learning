"""Strict configuration entry point for production NLHE training."""

from __future__ import annotations

from pathlib import Path

from pbg.core.config import ConfigError, RootConfig, parse_config, parse_config_mapping
from pbg.games.poker.bundle import validate_config


def load_config(
    path: str | Path, *, hardware_profile: str | Path | None = None
) -> RootConfig:
    """Load and validate one production NLHE configuration."""

    cfg = parse_config(path, hardware_profile=hardware_profile)
    try:
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return cfg


def load_config_mapping(
    raw: dict, *, hardware_profile: str | Path | None = None
) -> RootConfig:
    """Load and poker-validate an in-memory/checkpoint configuration."""

    cfg = parse_config_mapping(raw, hardware_profile=hardware_profile)
    try:
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return cfg


__all__ = ["ConfigError", "RootConfig", "load_config", "load_config_mapping"]
