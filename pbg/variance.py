"""Frozen-state update-variance consumer for the public queued runtime."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from pbg.benchmark import _code_identity
from pbg.config import RootConfig, load_config_mapping
from pbg.queued import sha256_file
from pbg.runtime import TrainingSession


def _device_state(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: _device_state(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_state(item) for item in value)
    if isinstance(value, list):
        return [_device_state(item) for item in value]
    return copy.deepcopy(value)


def _parameters(module: nn.Module) -> Tensor:
    return torch.cat(
        tuple(
            parameter.detach().float().flatten()
            for parameter in module.parameters()
        )
    )


def estimate_update_variance(
    cfg: RootConfig,
    *,
    repeats: int,
    checkpoint: str | Path | None = None,
    seed_stride: int = 1_000_003,
) -> dict[str, Any]:
    """Measure update variance from one identical frozen optimizer boundary."""

    if repeats < 2:
        raise ValueError("variance estimation requires at least two repeats")
    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    base = TrainingSession(cfg, resume_from=checkpoint)
    model_state = _device_state(base.modules.state_dict())
    reference_model_state = _device_state(base.reference_modules.state_dict())
    optimizer_state = _device_state(base.optimizers.state_dict())
    run_states = copy.deepcopy(base.run_states)
    reference_states = copy.deepcopy(base.reference_states)
    initial = tuple(
        _parameters(base.modules.lane_module(lane))
        for lane in range(base.plan.manifest.n_replica)
    )
    means = tuple(torch.zeros_like(value) for value in initial)
    m2 = tuple(torch.zeros_like(value) for value in initial)
    samples: list[dict[str, object]] = []
    base_scientific = cfg.scientific_mapping()
    for repeat in range(repeats):
        scientific = copy.deepcopy(base_scientific)
        scientific["seed"] = (
            int(base_scientific["seed"]) + repeat * seed_stride
        )
        local = load_config_mapping(
            {
                "scientific": scientific,
                "execution": cfg.execution.mapping(),
            }
        )
        session = TrainingSession(local)
        session.modules.load_state_dict(model_state)
        session.reference_modules.load_state_dict(reference_model_state)
        session.optimizers.load_state_dict(optimizer_state)
        session.run_states = copy.deepcopy(run_states)
        session.reference_states = copy.deepcopy(reference_states)
        result = session.step()
        lane_results = TrainingSession._lane_results(result)
        lane_rows = []
        for lane in range(session.plan.manifest.n_replica):
            update = _parameters(session.modules.lane_module(lane)) - initial[lane]
            delta = update - means[lane]
            means[lane].add_(delta / (repeat + 1))
            m2[lane].add_(delta * (update - means[lane]))
            lane_result = (
                lane_results[lane] if lane < len(lane_results) else None
            )
            lane_rows.append(
                {
                    "lane": lane,
                    "player_nodes": int(
                        getattr(lane_result, "player_nodes", 0)
                    ),
                    "update_l2": float(torch.linalg.vector_norm(update)),
                }
            )
        samples.append(
            {
                "repeat": repeat,
                "scientific_seed": scientific["seed"],
                "lanes": lane_rows,
            }
        )
        del result, session
    lanes = []
    for lane in range(len(initial)):
        total_variance = float(m2[lane].sum() / (repeats - 1))
        if not math.isfinite(total_variance) or total_variance < 0:
            raise RuntimeError("update variance is not finite and nonnegative")
        lanes.append(
            {
                "lane": lane,
                "parameter_count": initial[lane].numel(),
                "mean_update_l2": float(torch.linalg.vector_norm(means[lane])),
                "total_update_variance": total_variance,
            }
        )
    return {
        "schema": "pbg-queued-update-variance-v1",
        "mode": dict(cfg.mode),
        "scientific_config": base_scientific,
        "execution": cfg.execution.mapping(),
        "identity": {
            "code": _code_identity(),
            "checkpoint_sha256": (
                None if checkpoint is None else sha256_file(checkpoint)
            ),
            "seed_stride": seed_stride,
        },
        "repeats": repeats,
        "lanes": lanes,
        "samples": samples,
    }


__all__ = ["estimate_update_variance"]
