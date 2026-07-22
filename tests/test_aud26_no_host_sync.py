from __future__ import annotations

import inspect

import pytest
import torch

from pbg.core.game import EventBatch
from pbg.core.execution import ByteAccountant
from pbg.games.poker.adapter import PokerModelBundle
from pbg.games.poker.rules import CHECK_CALL, FOLD, PokerConfig
from pbg.games.poker.folded import PartitionSummaryBatch, PokerGPUPartitionAdapter
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig, PreflopNet
from pbg.games.poker.model.incidence import (
    build_compact_map,
    compact_gather,
    compact_scatter,
)
from pbg.games.poker.model_inputs import DeviceHistoryMaterializer
from pbg.games.poker.partition_cache import DevicePartitionSummaryCache
from pbg.games.poker.model.primitives import MHA
from pbg.games.poker.rollout_backend import PokerRolloutBackend


class _DevicePartition:
    """Device-only evaluator stand-in for sync-debug policy construction."""

    beta_rank = 4

    @staticmethod
    def summarize(*, board_cards, live_range_weights, folded):
        batch = board_cards.shape[0]
        folders = folded.folder_mask.shape[1]
        device = board_cards.device
        hands = live_range_weights.shape[-1]
        cards = int((1 + (1 + 8 * hands) ** 0.5) / 2)
        rank = 4
        zeros = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)
        return PartitionSummaryBatch(
            fields={
                "hand_log_ratio": zeros(batch, hands),
                "beta_eigenvalue": zeros(batch, rank),
                "beta_eigenvector": zeros(batch, cards, rank),
                "folder_log_mass": zeros(batch, folders),
                "folder_card_mass": zeros(batch, folders, cards),
                "tight_cluster_scalar": zeros(batch),
                "folder_mask": folded.folder_mask,
                "folder_line_meta": folded.line_meta,
                "folder_seat": folded.seat_ids,
            },
            metadata={"backend": "device_test"},
        )


class _CountingPartition(_DevicePartition):
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, *, board_cards, live_range_weights, folded):
        self.calls += 1
        return super().summarize(
            board_cards=board_cards,
            live_range_weights=live_range_weights,
            folded=folded,
        )


def test_cuda_hot_path_implementations_do_not_copy_or_branch_through_host() -> None:
    attention_source = inspect.getsource(MHA.forward)
    assert "count_nonzero" not in attention_source
    assert ".item(" not in attention_source
    history_source = inspect.getsource(DeviceHistoryMaterializer.pack)
    assert ".cpu(" not in history_source
    assert ".item(" not in history_source
    assert ".tolist(" not in history_source
    cache_source = inspect.getsource(DevicePartitionSummaryCache.merge)
    assert ".cpu(" not in cache_source
    assert ".item(" not in cache_source
    assert ".tolist(" not in cache_source
    folded_boundary = inspect.getsource(PokerGPUPartitionAdapter._normalize_board)
    folded_boundary += inspect.getsource(PokerGPUPartitionAdapter._validate_inputs)
    assert "bool(" not in folded_boundary
    assert ".cpu(" not in folded_boundary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_alias_compaction_and_biased_attention_request_no_host_sync() -> None:
    device = torch.device("cuda")
    attention = MHA(16, 4).to(device)
    attention.configure_execution("chunked_dense", query_chunk_size=8)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_preflop_postflop_policy_inputs_request_no_host_sync() -> None:
    device = torch.device("cuda")
    cfg = PokerConfig(n_players=3, stack_bb=8.0)
    backend = PokerRolloutBackend(
        cfg,
        device,
        storage_page_rows=16,
        max_rollout_steps=32,
        partition_adapter=_DevicePartition(),
    )
    model_cfg = PokerModelConfig(
        d_model=4,
        n_heads=1,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=1,
        k_act=4,
        max_players=3,
        class_embedding_dim=2,
        ffn_hidden=4,
        folder_beta_rank=4,
        max_history_tokens=32,
    )
    model = PokerModelBundle(
        PreflopNet(model_cfg), PostflopNet(model_cfg)
    ).to(device).eval()
    root = backend.sequential_start_states(1)
    reach = torch.zeros((1, 3, 1326), dtype=torch.float32, device=device)

    # Prepare a one-folder flop before sync-debug mode so the assertion covers
    # model-input construction rather than public state navigation.
    state = backend.step_player_batch(
        root, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32, device=device))
    )
    state = backend.step_player_batch(
        state, EventBatch(torch.tensor([FOLD], dtype=torch.int32, device=device))
    )
    state = backend.step_player_batch(
        state, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32, device=device))
    )
    flop = backend.step_chance_batch(
        state,
        EventBatch(
            torch.zeros(1, dtype=torch.int32, device=device),
            {"cards": torch.tensor([[3, 7, 11]], dtype=torch.int16, device=device)},
        ),
    )

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.no_grad():
            pre = backend.policy_forward_family(model, "pre", root, reach)
            post = backend.policy_forward_family(model, "post_2", flop, reach)
            cached = backend.policy_forward_family(model, "post_2", flop, reach)
        assert pre.v.is_cuda and post.v.is_cuda and cached.v.is_cuda
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_folded_summaries_have_sparse_anchor_ownership_and_complete_accounting() -> None:
    device = torch.device("cuda")
    cfg = PokerConfig(n_players=3, stack_bb=8.0, rank_count=4)
    partition = _CountingPartition()
    backend = PokerRolloutBackend(
        cfg,
        device,
        storage_page_rows=8,
        max_rollout_steps=32,
        partition_adapter=partition,
    )
    model_cfg = PokerModelConfig(
        d_model=4, n_heads=1, n_layers_pre=1, n_layers_post=1,
        m_raise=1, k_act=4, max_players=3, class_embedding_dim=2,
        evaluator_summary_dim=0, ffn_hidden=4, folder_beta_rank=4,
        max_history_tokens=32,
    )
    model = PokerModelBundle(
        PreflopNet(model_cfg, cfg.geometry),
        PostflopNet(model_cfg, cfg.geometry),
    ).to(device).eval()
    backend.register_policy_model(model)
    expected_cache_bytes = DevicePartitionSummaryCache.storage_nbytes(
        8, n_cards=cfg.geometry.n_cards, beta_rank=model_cfg.folder_beta_rank,
        max_folders=1, include_anchor=True,
    )
    assert expected_cache_bytes > 0
    assert backend.partition_cache_memory_bound() == expected_cache_bytes

    reach = torch.zeros(
        (1, cfg.n_players, cfg.geometry.n_hands),
        dtype=torch.float32,
        device=device,
    )
    accountant = ByteAccountant(1 << 30)
    store = backend.begin_decision_input_captures(accountant=accountant)

    state = backend.sequential_start_states(1)
    for _ in range(cfg.n_players):
        state = backend.step_player_batch(
            state,
            EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32, device=device)),
        )
    flop_all_live = backend.step_chance_batch(
        state,
        EventBatch(
            torch.zeros(1, dtype=torch.int32, device=device),
            {"cards": torch.tensor([[36, 40, 44]], dtype=torch.int16, device=device)},
        ),
    )
    with torch.no_grad():
        backend.policy_forward_family(model, "post_3", flop_all_live, reach)
    assert partition.calls == 0

    state = backend.sequential_start_states(1)
    for event in (CHECK_CALL, FOLD, CHECK_CALL):
        state = backend.step_player_batch(
            state,
            EventBatch(torch.tensor([event], dtype=torch.int32, device=device)),
        )
    flop_after_fold = backend.step_chance_batch(
        state,
        EventBatch(
            torch.zeros(1, dtype=torch.int32, device=device),
            {"cards": torch.tensor([[36, 40, 44]], dtype=torch.int16, device=device)},
        ),
    )
    for node_id in (10, 11):
        with torch.no_grad():
            backend.policy_forward_family(model, "post_2", flop_after_fold, reach)
        capture = backend.capture_decision_inputs("post_2", flop_after_fold, reach)
        backend.append_decision_input_capture(
            store,
            torch.tensor([node_id], dtype=torch.int32, device=device),
            capture,
        )
    assert partition.calls == 1
    assert store.summary_caches[1].size == 1
    assert store.summary_caches[1].allocated_capacity_bytes == expected_cache_bytes

    sealed = backend.seal_decision_input_captures(store)
    assert sealed.input_pools[1]["folded_summary"].rows == 1
    assert sealed.inputs[1].columns["folded_summary_ref"].tolist() == [0, 0]
    assert sealed.node_ids.tolist() == [10, 11]
    assert accountant.reserved_bytes == (
        sealed.node_ids.numel() * 3 * 4
        + sealed.inputs[1].allocated_capacity_bytes
        + sealed.input_pools[1]["folded_summary"].allocated_capacity_bytes
    )
