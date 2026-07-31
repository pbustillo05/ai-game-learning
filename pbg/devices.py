"""Accelerator-only execution contract shared by public consumers."""

from __future__ import annotations

import os
from typing import Literal

import torch

ExecutionDevice = Literal["cuda", "mps"]
SUPPORTED_EXECUTION_DEVICES = ("cuda", "mps")


def resolve_execution_device(
    requested: str | torch.device,
    *,
    workload: str,
    local_rank: int | None = None,
) -> torch.device:
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"{workload} device must be explicitly 'cuda' or 'mps'"
        ) from exc
    if device.type not in SUPPORTED_EXECUTION_DEVICES:
        raise ValueError(
            f"{workload} device must be explicitly 'cuda' or 'mps'; "
            f"{device.type!r} is not a production execution device"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{workload} requested CUDA, but CUDA is unavailable; "
                "PBG never falls back to CPU"
            )
        index = device.index if device.index is not None else (local_rank or 0)
        if not 0 <= index < torch.cuda.device_count():
            raise RuntimeError(
                f"{workload} requested cuda:{index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        return torch.device("cuda", index)
    if device.index is not None:
        raise ValueError("MPS does not accept a device index")
    available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not available:
        built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        reason = "unavailable on this host" if built else "absent from this PyTorch build"
        raise RuntimeError(
            f"{workload} requested MPS, but MPS is {reason}; "
            "PBG never falls back to CPU"
        )
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 permits hidden CPU execution; "
            f"unset it before starting {workload}"
        )
    return torch.device("mps")


def validate_workload_capabilities(device: torch.device, *, workload: str) -> None:
    if workload == "distributed training" and device.type != "cuda":
        raise RuntimeError("distributed training is CUDA/NCCL-only")
    if device.type == "mps" and workload in {
        "training",
        "distributed training",
        "best-response evaluation",
        "training benchmark",
        "variance evaluation",
    }:
        raise RuntimeError(
            f"MPS {workload} is unavailable because evaluator contract v2 defers "
            "sample_joint_profiles, complete, complete_linear_moments, and "
            "summarize_folders on MPS; use CUDA"
        )


def assert_module_device(
    module: torch.nn.Module, device: torch.device, name: str
) -> None:
    mismatched = [
        label
        for label, tensor in (*module.named_parameters(), *module.named_buffers())
        if tensor.device != device
    ]
    if mismatched:
        raise RuntimeError(
            f"{name} contains tensors outside {device}: "
            f"{', '.join(mismatched[:5])}"
        )


__all__ = [
    "SUPPORTED_EXECUTION_DEVICES",
    "ExecutionDevice",
    "assert_module_device",
    "resolve_execution_device",
    "validate_workload_capabilities",
]
