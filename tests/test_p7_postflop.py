import torch
import pytest

from pbg.games.poker.folded import PartitionSummaryBatch
from pbg.games.poker.model.incidence import (
    IncidenceResidual,
    build_compact_map,
    compact_gather,
    compact_scatter,
)
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


def test_incidence_residual_exact_card_identical_and_own_channels():
    torch.manual_seed(71)
    layer = IncidenceResidual(3)
    rows = torch.randn(1, 3, 4, 3)
    mass = torch.rand(1, 3, 4)
    incidence = torch.zeros(4, 52)
    incidence[0, [0, 1]] = 1
    incidence[1, [0, 2]] = 1
    incidence[2, [3, 4]] = 1
    incidence[3, [1, 4]] = 1
    _, card, equal, own = layer(rows, mass, incidence, torch.ones(4))
    source = layer.source(layer.norm(rows)) * mass[..., None]
    expected_card = torch.zeros_like(card)
    expected_equal = torch.zeros_like(equal)
    expected_own = torch.zeros_like(own)
    for q in range(3):
        for h in range(4):
            for other in range(3):
                for g in range(4):
                    overlap = float((incidence[h] * incidence[g]).sum())
                    contribution = source[0, other, g]
                    if other == q:
                        expected_own[0, q, h] += contribution * (
                            overlap - 2.0 * float(g == h)
                        )
                    else:
                        expected_card[0, q, h] += contribution * (
                            overlap - 2.0 * float(g == h)
                        )
                        expected_equal[0, q, h] += contribution * float(g == h)
    torch.testing.assert_close(card, expected_card)
    torch.testing.assert_close(equal, expected_equal)
    torch.testing.assert_close(own, expected_own)


def test_signed_beta_coordinates_reconstruct_evaluator_pair_field():
    torch.manual_seed(72)
    cfg = _cfg()
    cfg = type(cfg)(**{**cfg.__dict__, "folder_beta_rank": 52})
    model = PostflopNet(cfg)
    compatibility = torch.zeros(1, 1326, dtype=torch.bool)
    compatibility[:, :8] = True
    mapping = build_compact_map(compatibility)
    matrix = torch.randn(1, 52, 52)
    matrix = (matrix + matrix.transpose(1, 2)) / 2
    matrix.diagonal(dim1=1, dim2=2).zero_()
    summary = PartitionSummaryBatch(
        fields={
            "single_card_log_ratio": torch.randn(1, 52),
            "pair_card_log_ratio": matrix,
            "output_valid": torch.ones(1, dtype=torch.bool),
        },
        metadata={},
    )
    correction, geometry = model._folded_geometry(summary, mapping, torch.float32)
    cards = model.combo_cards[mapping.compact_to_full]
    alpha = summary.fields["single_card_log_ratio"]
    expected = (
        alpha.gather(1, cards[..., 0])
        + alpha.gather(1, cards[..., 1])
        + matrix[torch.arange(1)[:, None], cards[..., 0], cards[..., 1]]
    )
    torch.testing.assert_close(correction, expected)
    positive = geometry[..., 52:104]
    negative = geometry[..., 104:]
    reconstructed = positive @ positive.transpose(1, 2) - negative @ negative.transpose(1, 2)
    two_hot = geometry[..., :52]
    direct = two_hot @ matrix @ two_hot.transpose(1, 2)
    torch.testing.assert_close(reconstructed, direct, atol=2e-4, rtol=2e-4)


def test_folder_pressure_and_folder_seat_readout_consume_semantic_fields():
    torch.manual_seed(73)
    pub, ranges, masks = _inputs()
    pub.status[:, 1] = 1
    pub.to_act.zero_()
    masks.status = pub.status
    batch = ranges.shape[0]
    fields = {
        "base_log_z": torch.zeros(batch),
        "single_card_log_ratio": torch.zeros(batch, 52),
        "pair_card_log_ratio": torch.zeros(batch, 52, 52),
        "output_valid": torch.ones(batch, dtype=torch.bool),
        "folder_log_mass": torch.zeros(batch, 1),
        "folder_single_card_log_ratio": torch.full((batch, 1, 52), -0.04),
        "folder_pair_card_log_ratio": torch.zeros(batch, 1, 52, 52),
        "folder_output_valid": torch.ones(batch, 1, dtype=torch.bool),
        "folder_mask": torch.ones(batch, 1, dtype=torch.bool),
        "folder_line_meta": torch.randn(batch, 1, 8),
        "folder_seat": torch.ones(batch, 1, dtype=torch.long),
    }
    masks.evaluator_summary = PartitionSummaryBatch(fields, {})
    model = PostflopNet(_cfg()).eval()
    first = model(pub, ranges, masks=masks)
    changed = {key: value.clone() for key, value in fields.items()}
    changed["folder_single_card_log_ratio"][..., :4] = -1.0
    masks.evaluator_summary = PartitionSummaryBatch(changed, {})
    second = model(pub, ranges, masks=masks)
    assert torch.isfinite(first.v).all()
    assert not torch.allclose(first.v, second.v)


def test_folded_seats_do_not_create_postflop_attention_rows():
    pub, ranges, masks = _inputs()
    pub.status[:, 1] = 1
    pub.to_act.zero_()
    masks.status = pub.status
    model = PostflopNet(_cfg()).eval()
    lengths: list[int] = []
    hook = model.attentions[0].register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1])
    )
    try:
        model(pub, ranges, masks=masks)
    finally:
        hook.remove()
    players = pub.stacks.shape[1]
    seats = torch.arange(players)[None].expand(pub.street.shape[0], -1)
    relative = torch.remainder(seats - pub.button[:, None], players)
    public, _ = model._public_tokens(pub, None, ranges.dtype, relative)
    expected = masks.compatibility.sum(1).max().item() + public.shape[1] + model.cfg.k_act
    assert lengths == [expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_postflop_conformance_paths_run_forward_backward_on_cuda():
    pub, ranges, masks = _inputs()
    pub = type(pub)(
        **{
            name: None if getattr(pub, name) is None else getattr(pub, name).cuda()
            for name in pub.__dataclass_fields__
        }
    )
    masks.compatibility = masks.compatibility.cuda()
    model = PostflopNet(_cfg()).cuda()
    output = model(pub, ranges.cuda(), masks=masks)
    (output.v.square().mean() + output.raw["type_logits"].square().mean()).backward()
    assert torch.isfinite(output.v).all()
