from tests.model_support import FamilyModelFixture
from collections.abc import Mapping
from dataclasses import dataclass

import pytest
import torch

from pbg.games.poker.model.incidence import (
    IncidenceResidual,
    build_compact_map,
    compact_gather,
    compact_scatter,
)
from pbg.games.poker.model.postflop import (
    PostflopMasks,
)
from pbg.games.poker.model.preflop import PokerModelConfig, PokerPublicBatch

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@dataclass(frozen=True)
class PartitionSummaryBatch:
    fields: Mapping[str, torch.Tensor]
    metadata: Mapping[str, object]


def _inputs():
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
    return pub, ranges, PostflopMasks(compatibility)


def _cuda_inputs():
    pub, ranges, masks = _inputs()
    pub = type(pub)(
        **{
            name: None if getattr(pub, name) is None else getattr(pub, name).cuda()
            for name in pub.__dataclass_fields__
        }
    )
    masks.compatibility = masks.compatibility.cuda()
    return pub, ranges.cuda(), masks


def _cfg():
    return PokerModelConfig(
        d_model=16, n_heads=4, layers=1,
        m_raise=2, k_act=5, max_players=2, class_embedding_dim=4,
        ffn_hidden=24, use_branching_heads=True,
    )


def test_compact_gather_scatter_is_ragged_and_differentiable():
    compatibility = torch.tensor([[True, False, True, False], [False, True, False, False]])
    mapping = build_compact_map(compatibility, compact_size=2)
    source = torch.arange(8.0).reshape(2, 4).requires_grad_()
    compact = compact_gather(source, mapping)
    assert compact.tolist() == [[0.0, 2.0], [5.0, 0.0]]
    restored = compact_scatter(compact, mapping)
    assert restored.tolist() == [[0.0, 0.0, 2.0, 0.0], [0.0, 5.0, 0.0, 0.0]]
    restored.sum().backward()
    assert source.grad.tolist() == [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


@CUDA
def test_postflop_forward_scatter_and_autocast():
    torch.manual_seed(70)
    pub, ranges, masks = _cuda_inputs()
    model = FamilyModelFixture(_cfg(), "post").cuda().eval()
    output = model(pub, ranges, masks=masks)
    assert output.raw["type_logits"].shape == (2, 1326, 3)
    assert output.raw["omega"].shape == output.raw["mu"].shape == output.raw["s"].shape == (2, 1326, 2)
    assert output.raw["iota"].shape == output.v.shape == (2, 1326)
    assert output.t is not None and output.t.shape == (2,)
    assert output.c is not None and output.c.shape == (2,)
    assert torch.count_nonzero(output.v[0, 12:]) == 0
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        mixed = model(pub, ranges, masks=masks)
    assert mixed.v.dtype == torch.float16


@CUDA
def test_incidence_residual_exact_card_identical_and_own_channels():
    with torch.device("cuda"):
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


def test_signed_beta_coordinates_consume_current_evaluator_eigenpairs():
    torch.manual_seed(72)
    cfg = _cfg()
    cfg = type(cfg)(**{**cfg.__dict__, "folder_beta_rank": 52})
    model = FamilyModelFixture(cfg, "post")
    compatibility = torch.zeros(1, 1326, dtype=torch.bool)
    compatibility[:, :8] = True
    mapping = build_compact_map(compatibility, compact_size=8)
    eigenvalue = torch.randn(1, 52)
    eigenvector = torch.linalg.qr(torch.randn(1, 52, 52)).Q
    summary = PartitionSummaryBatch(
        fields={
            "hand_log_ratio": torch.randn(1, 1326),
            "beta_eigenvalue": eigenvalue,
            "beta_eigenvector": eigenvector,
        },
        metadata={"contract_version": "2"},
    )
    correction, geometry = model.encoder._folded_geometry(summary, mapping, torch.float32)
    torch.testing.assert_close(correction, summary.fields["hand_log_ratio"][:, :8])
    positive = geometry[..., 52:104]
    negative = geometry[..., 104:]
    reconstructed = positive @ positive.transpose(1, 2) - negative @ negative.transpose(1, 2)
    two_hot = geometry[..., :52]
    matrix = eigenvector @ torch.diag_embed(eigenvalue) @ eigenvector.transpose(1, 2)
    direct = two_hot @ matrix @ two_hot.transpose(1, 2)
    torch.testing.assert_close(reconstructed, direct, atol=2e-4, rtol=2e-4)


@CUDA
def test_folder_pressure_and_folder_seat_readout_consume_semantic_fields():
    torch.manual_seed(73)
    pub, ranges, masks = _cuda_inputs()
    pub.status[:, 1] = 1
    pub.to_act.zero_()
    masks.status = pub.status
    masks.live_seat_ids = torch.zeros((pub.street.shape[0], 1), dtype=torch.long)
    masks.compact_size = int(masks.compatibility.sum(1).max())
    masks.live_seat_ids = torch.zeros((pub.street.shape[0], 1), dtype=torch.long)
    masks.compact_size = int(masks.compatibility.sum(1).max())
    batch = ranges.shape[0]
    device = ranges.device
    cfg = _cfg()
    fields = {
        "hand_log_ratio": torch.zeros(batch, 1326, device=device),
        "beta_eigenvalue": torch.zeros(
            batch, cfg.folder_beta_rank, device=device
        ),
        "beta_eigenvector": torch.zeros(
            batch, 52, cfg.folder_beta_rank, device=device
        ),
        "folder_log_mass": torch.zeros(batch, 1, device=device),
        "folder_card_mass": torch.full(
            (batch, 1, 52), 2.0 / 52.0, device=device
        ),
        "tight_cluster_scalar": torch.zeros(batch, device=device),
        "folder_mask": torch.ones(batch, 1, dtype=torch.bool, device=device),
        "folder_line_meta": torch.randn(batch, 1, 8, device=device),
        "folder_seat": torch.ones(batch, 1, dtype=torch.long, device=device),
    }
    masks.evaluator_summary = PartitionSummaryBatch(fields, {"contract_version": "2"})
    model = FamilyModelFixture(cfg, "post").cuda().eval()
    first = model(pub, ranges, masks=masks)
    changed = {key: value.clone() for key, value in fields.items()}
    changed["folder_card_mass"][..., :4] = 0.0
    masks.evaluator_summary = PartitionSummaryBatch(changed, {"contract_version": "2"})
    second = model(pub, ranges, masks=masks)
    assert torch.isfinite(first.v).all()
    assert not torch.allclose(first.v, second.v)


@CUDA
def test_compact_contract_v2_folder_features_feed_postflop_model():
    torch.manual_seed(74)
    pub, ranges, masks = _cuda_inputs()
    pub.status[:, 1] = 1
    pub.to_act.zero_()
    masks.status = pub.status
    batch = ranges.shape[0]
    device = ranges.device
    fields = {
        "hand_log_ratio": torch.randn(batch, 1326, device=device) * 0.01,
        "beta_eigenvalue": torch.randn(batch, 2, device=device),
        "beta_eigenvector": torch.randn(batch, 52, 2, device=device),
        "folder_log_mass": torch.zeros(batch, 1, device=device),
        "folder_card_mass": torch.rand(batch, 1, 52, device=device),
        "tight_cluster_scalar": torch.zeros(batch, device=device),
        "folder_mask": torch.ones(batch, 1, dtype=torch.bool, device=device),
        "folder_line_meta": torch.randn(batch, 1, 8, device=device),
        "folder_seat": torch.ones(batch, 1, dtype=torch.long, device=device),
    }
    masks.evaluator_summary = PartitionSummaryBatch(fields, {"contract_version": "2"})
    cfg = type(_cfg())(**{**_cfg().__dict__, "folder_beta_rank": 2})
    output = FamilyModelFixture(cfg, "post").cuda().eval()(pub, ranges, masks=masks)
    assert torch.isfinite(output.v).all()


@CUDA
def test_current_evaluator_summary_rejects_retired_field_aliases():
    with torch.device("cuda"):
        pub, ranges, masks = _inputs()
        pub.status[:, 1] = 1
        pub.to_act.zero_()
        masks.status = pub.status
        batch = ranges.shape[0]
        cfg = type(_cfg())(**{**_cfg().__dict__, "folder_beta_rank": 2})
        masks.evaluator_summary = PartitionSummaryBatch(
            {
                "hand_log_ratio": torch.zeros(batch, 1326),
                "beta_eigenvalue": torch.zeros(batch, 2),
                "beta_eigenvector": torch.zeros(batch, 52, 2),
                "folder_log_mass": torch.zeros(batch, 1),
                "folder_card_marginal": torch.ones(batch, 1, 52),
                "folder_mask": torch.ones(batch, 1, dtype=torch.bool),
                "folder_line_meta": torch.zeros(batch, 1, 8),
                "folder_seat": torch.ones(batch, 1, dtype=torch.long),
            },
            {"contract_version": "2"},
        )
        with pytest.raises(ValueError, match="folder card marginals"):
            FamilyModelFixture(cfg, "post").eval()(pub, ranges, masks=masks)


@CUDA
def test_folded_seats_do_not_create_postflop_attention_rows():
    pub, ranges, masks = _cuda_inputs()
    pub.status[:, 1] = 1
    pub.to_act.zero_()
    masks.status = pub.status
    masks.live_seat_ids = torch.zeros((pub.street.shape[0], 1), dtype=torch.long)
    masks.compact_size = int(masks.compatibility.sum(1).max())
    model = FamilyModelFixture(_cfg(), "post").cuda().eval()
    lengths: list[int] = []
    hook = model.shared.attentions[0].register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1])
    )
    try:
        model(pub, ranges, masks=masks)
    finally:
        hook.remove()
    players = pub.stacks.shape[1]
    seats = torch.arange(players, device=ranges.device)[None].expand(
        pub.street.shape[0], -1
    )
    relative = torch.remainder(seats - pub.button[:, None], players)
    public, _ = model.encoder._public_tokens(model.shared, pub, None, ranges.dtype, relative)
    expected = masks.compatibility.sum(1).max().item() + public.shape[1] + model.cfg.k_act
    assert lengths == [expected]


@CUDA
def test_postflop_conformance_paths_run_forward_backward_on_cuda():
    pub, ranges, masks = _cuda_inputs()
    model = FamilyModelFixture(_cfg(), "post").cuda()
    output = model(pub, ranges, masks=masks)
    (output.v.square().mean() + output.raw["type_logits"].square().mean()).backward()
    assert torch.isfinite(output.v).all()
