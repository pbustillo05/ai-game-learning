"""Lazy loader for the queued CUDA alias-table builder."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def alias_table_cuda():
    """Build/load the linear-time device-resident Walker table operator."""

    if not torch.cuda.is_available():
        raise RuntimeError("alias-table construction requires CUDA")
    source = Path(__file__).with_name("csrc") / "alias_table.cu"
    return load(
        name="pbg_queued_alias_table_v1",
        sources=(str(source),),
        extra_cuda_cflags=("-O3", "--fmad=false"),
        extra_cflags=("-O3",),
        with_cuda=True,
        verbose=False,
    )


__all__ = ["alias_table_cuda"]
