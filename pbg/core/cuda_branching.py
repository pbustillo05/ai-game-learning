"""Lazy loader for queued branching CUDA primitives."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def branching_cuda():
    """Build/load linear device-side branching metadata operations."""

    if not torch.cuda.is_available():
        raise RuntimeError("branching metadata operations require CUDA")
    source = Path(__file__).with_name("csrc") / "branching_metadata.cu"
    return load(
        name="pbg_queued_branching_metadata_v1",
        sources=(str(source),),
        extra_cuda_cflags=("-O3", "--fmad=false"),
        extra_cflags=("-O3",),
        with_cuda=True,
        verbose=False,
    )


__all__ = ["branching_cuda"]
