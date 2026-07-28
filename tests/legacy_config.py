"""Test-only construction for the isolated pre-cutover comparison engine."""

from pathlib import Path

from pbg.core.config import ConfigError, RootConfig, parse_config, parse_config_mapping
from pbg.games.poker.bundle import validate_config


def load_config(
    path: str | Path, *, hardware_profile: str | Path | None = None
) -> RootConfig:
    config = parse_config(path, hardware_profile=hardware_profile)
    try:
        validate_config(config)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return config


def load_config_mapping(
    raw: dict, *, hardware_profile: str | Path | None = None
) -> RootConfig:
    config = parse_config_mapping(raw, hardware_profile=hardware_profile)
    try:
        validate_config(config)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return config


__all__ = ["ConfigError", "RootConfig", "load_config", "load_config_mapping"]
