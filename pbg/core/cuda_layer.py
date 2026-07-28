"""Lazy loader for the queued layered-target CUDA operator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def layered_target_cuda():
    """Build/load the sole C++-enqueued CUDA layer operator."""

    if not torch.cuda.is_available():
        raise RuntimeError("layered targets require CUDA")
    source = Path(__file__).with_name("csrc") / "layered_targets.cu"
    return load(
        name="pbg_queued_layered_targets_v1",
        sources=(str(source),),
        extra_cuda_cflags=("-O3", "--fmad=false"),
        extra_cflags=("-O3",),
        with_cuda=True,
        verbose=False,
    )


__all__ = ["layered_target_cuda"]
