import torch

from pbg.games.poker.model.incidence import build_compact_map, compact_gather, compact_scatter
from pbg.games.poker.model.postflop import (
    PostflopMasks,
    PostflopNet,
    permute_suit_representation,
    suit_row_permutation,
)
from pbg.games.poker.model.preflop import PokerModelConfig, PokerPublicBatch


def _inputs(summary_dim: int = 0):
    batch, players = 2, 2
    ranges = torch.rand(batch, players, 1326)
    compatibility = torch.zeros(batch, 1326, dtype=torch.bool)
    compatibility[0, :12] = True
    compatibility[1, 20:29] = True
    ranges = ranges * compatibility[:, None]
    ranges = ranges / ranges.sum(2, keepdim=True)
    pub = PokerPublicBatch(
        street=torch.tensor([1, 2]),
        board=torch.tensor([[0, 5, 10, -1, -1], [4, 9, 14, 19, -1]]),
        pot=torch.tensor([3.0, 7.0]),
        stacks=torch.full((batch, players), 99.0),
        committed=torch.ones(batch, players),
        to_act=torch.tensor([0, 1]),
        legality_summary=torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.5]]),
        n_ticks=torch.full((batch,), 4),
        raise_grid_bounds=torch.tensor([[2.0, 100.0], [3.0, 80.0]]),
        status=torch.zeros(batch, players, dtype=torch.long),
        seat_mask=torch.ones(batch, players, dtype=torch.bool),
        button=torch.tensor([1, 0]),
    )
    summary = None
    if summary_dim:
        summary = {"opaque_query_rows": torch.randn(batch, 1326, summary_dim)}
    return pub, ranges, PostflopMasks(compatibility, evaluator_summary=summary)


def _cfg(summary_dim: int = 0):
    return PokerModelConfig(
        d_model=16, n_heads=4, n_layers_pre=1, n_layers_post=1,
        m_raise=2, k_act=5, max_players=2, class_embedding_dim=4,
        evaluator_summary_dim=summary_dim, ffn_hidden=24, use_moment_heads=True,
    )


def test_compact_gather_scatter_is_ragged_and_differentiable():
    compatibility = torch.tensor([[True, False, True, False], [False, True, False, False]])
    mapping = build_compact_map(compatibility)
    source = torch.arange(8.0).reshape(2, 4).requires_grad_()
    compact = compact_gather(source, mapping)
    assert compact.tolist() == [[0.0, 2.0], [5.0, 0.0]]
    restored = compact_scatter(compact, mapping)
    assert restored.tolist() == [[0.0, 0.0, 2.0, 0.0], [0.0, 5.0, 0.0, 0.0]]
    restored.sum().backward()
    assert source.grad.tolist() == [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


def test_postflop_forward_scatter_summary_and_autocast():
    torch.manual_seed(70)
    pub, ranges, masks = _inputs(summary_dim=2)
    model = PostflopNet(_cfg(summary_dim=2)).eval()
    output = model(pub, ranges, masks=masks)
    assert output.raw["type_logits"].shape == (2, 1326, 3)
    assert output.raw["omega"].shape == output.raw["mu"].shape == output.raw["s"].shape == (2, 1326, 2)
    assert output.raw["iota"].shape == output.v.shape == (2, 1326)
    assert output.vvec is not None and output.vvec.shape == (2, 2)
    assert output.u is not None and output.u.shape == (2,)
    assert torch.count_nonzero(output.v[0, 12:]) == 0
    changed = {"opaque_query_rows": masks.evaluator_summary["opaque_query_rows"] + 2.0}
    other = model(pub, ranges, masks=PostflopMasks(masks.compatibility, evaluator_summary=changed))
    assert not torch.allclose(output.v[:, :9], other.v[:, :9])
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        mixed = model(pub, ranges, masks=masks)
    assert mixed.v.dtype == torch.float16


def test_configured_summary_accepts_absent_fields_as_zeros():
    pub, ranges, masks = _inputs()
    model = PostflopNet(_cfg(summary_dim=3))
    output = model(pub, ranges, masks=masks)
    assert torch.isfinite(output.v).all()


def test_suit_representation_relabels_board_and_rows_consistently():
    permutation = (2, 0, 3, 1)
    row_map = suit_row_permutation(permutation)
    assert torch.unique(row_map).numel() == 1326
    board = torch.tensor([[0, 5, 10, -1, -1]])
    rows = torch.arange(1326)[None]
    mapped_board, mapped = permute_suit_representation(board, {"rows": rows}, permutation)
    assert mapped_board.tolist() == [[2, 4, 11, -1, -1]]
    assert torch.equal(mapped["rows"][0, row_map], rows[0])

