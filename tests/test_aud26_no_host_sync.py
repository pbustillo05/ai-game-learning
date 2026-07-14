from __future__ import annotations

import inspect

import pytest
import torch

from pbg.core.tree import AliasTable
from pbg.games.poker.model.incidence import (
    build_compact_map,
    compact_gather,
    compact_scatter,
)
from pbg.games.poker.model.primitives import MHA


def test_cuda_hot_path_implementations_do_not_copy_or_branch_through_host() -> None:
    alias_source = inspect.getsource(AliasTable.build)
    assert ".cpu(" not in alias_source
    assert ".tolist(" not in alias_source
    attention_source = inspect.getsource(MHA.forward)
    assert "count_nonzero" not in attention_source
    assert ".item(" not in attention_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_alias_compaction_and_biased_attention_request_no_host_sync() -> None:
    device = torch.device("cuda")
    attention = MHA(16, 4).to(device)
    attention.configure_execution("chunked_dense", query_chunk_size=8)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        weights = torch.arange(1, 65, dtype=torch.float64, device=device)
        table = AliasTable.build(weights)
        draws = table.sample(512, torch.Generator(device=device).manual_seed(2601))
        assert draws.is_cuda

        compatibility = torch.zeros((2, 1326), dtype=torch.bool, device=device)
        compatibility[:, :1176] = True
        mapping = build_compact_map(compatibility, compact_size=1176)
        source = torch.randn((2, 1326, 16), device=device, requires_grad=True)
        compact = compact_gather(source, mapping)
        restored = compact_scatter(compact, mapping)

        rows = compact[:, :16]
        bias = torch.randn((2, 16), dtype=torch.float32, device=device)
        mask = torch.ones((2, 16), dtype=torch.bool, device=device)
        output = attention.forward_biased(rows, rows, bias, mask)
        (output.square().mean() + restored.square().mean()).backward()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
