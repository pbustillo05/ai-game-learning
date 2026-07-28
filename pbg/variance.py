"""Frozen-state update-variance consumer for the public queued runtime."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from pbg.config import RootConfig, load_config_mapping
from pbg.identity import accelerator_identity, code_identity
from pbg.queued import model_state_sha256, sha256_file
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


def _tensor_bytes(value: object) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


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
    controllers = copy.deepcopy(base.controllers)
    safeguards = copy.deepcopy(base.safeguards)
    initial = tuple(
        _parameters(base.modules.lane_module(lane))
        for lane in range(base.plan.manifest.n_replica)
    )
    means = tuple(torch.zeros_like(value) for value in initial)
    m2 = tuple(torch.zeros_like(value) for value in initial)
    budget = base.plan.manifest.hardware.memory_budget_bytes
    admitted_peak = max(
        admission.required_bytes for admission in base.plan.admissions
    )
    variance_bytes = sum(
        value.numel() * value.element_size()
        for values in (initial, means, m2)
        for value in values
    ) + sum(
        _tensor_bytes(value)
        for value in (model_state, reference_model_state, optimizer_state)
    )
    if admitted_peak + variance_bytes > budget:
        base.close()
        raise RuntimeError(
            "variance statistics exceed the immutable CUDA memory budget"
        )
    torch.cuda.reset_peak_memory_stats(base.device)
    memory_start = torch.cuda.memory_allocated(base.device)
    variance_device = base.device
    base.close()
    del base
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
        session.controllers = copy.deepcopy(controllers)
        session.safeguards = copy.deepcopy(safeguards)
        session.initial_model_sha256 = tuple(
            model_state_sha256(session.modules.lane_module(lane))
            for lane in range(session.plan.manifest.n_replica)
        )
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
                "logical_work": tuple(
                    dict(state.logical_work) for state in session.run_states
                ),
                "allocated_bytes": torch.cuda.memory_allocated(session.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    session.device
                ),
            }
        )
        del result
        session.close()
        del session
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
            "code": code_identity(),
            "accelerator": accelerator_identity(
                cfg.scientific.common.game.rank_count
            ),
            "checkpoint_sha256": (
                None if checkpoint is None else sha256_file(checkpoint)
            ),
            "seed_stride": seed_stride,
        },
        "repeats": repeats,
        "lanes": lanes,
        "samples": samples,
        "memory": {
            "configured_budget_bytes": budget,
            "admitted_peak_bytes": admitted_peak,
            "variance_statistics_bytes": variance_bytes,
            "start_allocated_bytes": memory_start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                variance_device
            ),
            "end_allocated_bytes": torch.cuda.memory_allocated(
                variance_device
            ),
        },
    }


__all__ = ["estimate_update_variance"]
