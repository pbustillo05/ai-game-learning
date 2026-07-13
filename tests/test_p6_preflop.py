import pytest
import torch

from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerModelConfig,
    PokerPublicBatch,
    PreflopMasks,
    PreflopNet,
)


def _config(*, moment: bool = False, players: int = 3) -> PokerModelConfig:
    return PokerModelConfig(
        d_model=16,
        n_heads=4,
        n_layers_pre=1,
        m_raise=2,
        k_act=5,
        max_players=players,
        class_embedding_dim=4,
        ffn_hidden=32,
        use_moment_heads=moment,
    )


def _inputs(batch: int = 2, players: int = 3) -> tuple[PokerPublicBatch, torch.Tensor]:
    torch.manual_seed(62)
    ranges = torch.rand(batch, players, 169)
    ranges = ranges / ranges.sum(dim=-1, keepdim=True)
    pub = PokerPublicBatch(
        street=torch.zeros(batch, dtype=torch.int64),
        board=torch.full((batch, 5), -1, dtype=torch.int64),
        pot=torch.rand(batch) + 1.0,
        stacks=torch.rand(batch, players) * 100.0,
        committed=torch.rand(batch, players) * 3.0,
        to_act=torch.arange(batch, dtype=torch.int64) % players,
        legality_summary=torch.rand(batch, 3),
        n_ticks=torch.full((batch,), 16, dtype=torch.int64),
        raise_grid_bounds=torch.tensor([[2.0, 100.0]]).expand(batch, -1).clone(),
        status=torch.zeros(batch, players, dtype=torch.int64),
        seat_mask=torch.ones(batch, players, dtype=torch.bool),
        button=torch.arange(batch, dtype=torch.int64) % players,
        committed_street=torch.rand(batch, players),
        call_amount=torch.rand(batch),
        current_bet=torch.rand(batch) + 1.0,
        min_raise_to=torch.rand(batch) + 2.0,
        max_raise_to=torch.full((batch,), 100.0),
        last_full_raise=torch.ones(batch),
        legal_action_mask=torch.ones(batch, 18, dtype=torch.bool),
        raise_ticks=torch.linspace(2.0, 100.0, 16)[None].expand(batch, -1).clone(),
        history=PackedPublicHistory(
            actor=torch.tensor([[0, 1], [1, -1]])[:batch],
            street=torch.tensor([[0, 0], [0, 1]])[:batch],
            action=torch.tensor([[1, 2], [0, 3]])[:batch],
            amount=torch.tensor([[0.5, 3.0], [0.0, 0.0]])[:batch],
            pot_before=torch.tensor([[1.5, 2.0], [1.5, 4.0]])[:batch],
            call_amount=torch.tensor([[0.5, 1.0], [1.0, 0.0]])[:batch],
            current_bet=torch.tensor([[1.0, 2.0], [1.0, 0.0]])[:batch],
            min_raise_to=torch.tensor([[2.0, 3.0], [2.0, 0.0]])[:batch],
            max_raise_to=torch.full((batch, 2), 100.0),
            mask=torch.ones(batch, 2, dtype=torch.bool),
        ),
    )
    return pub, ranges


def test_preflop_forward_shapes_fp32_and_autocast() -> None:
    model = PreflopNet(_config(moment=True))
    pub, ranges = _inputs()
    output = model(pub, ranges)
    assert output.raw["type_logits"].shape == (2, 169, 3)
    for name in ("omega", "mu", "s"):
        assert output.raw[name].shape == (2, 169, 2)
    assert output.raw["iota"].shape == output.v.shape == (2, 169)
    assert output.vvec is not None and output.vvec.shape == (2, 3)
    assert output.u is not None and output.u.shape == (2,)
    assert all(tensor.dtype == torch.float32 for tensor in output.raw.values())
    assert output.v.dtype == torch.float32

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        mixed = model(pub, ranges)
    assert all(tensor.dtype == torch.float16 for tensor in mixed.raw.values())
    assert mixed.v.dtype == torch.float16


def test_class_order_equivariance() -> None:
    torch.manual_seed(63)
    model = PreflopNet(_config(players=2)).eval()
    pub, ranges = _inputs(batch=1, players=2)
    permutation = torch.randperm(169)
    base = model(pub, ranges)
    permuted = model(
        pub,
        ranges[:, :, permutation],
        masks=PreflopMasks(class_ids=permutation),
    )
    for name in base.raw:
        torch.testing.assert_close(
            permuted.raw[name], base.raw[name][:, permutation], atol=1e-5, rtol=1e-5
        )
    torch.testing.assert_close(permuted.v, base.v[:, permutation], atol=1e-5, rtol=1e-5)


def test_masked_empty_seat_cannot_change_output() -> None:
    torch.manual_seed(64)
    model = PreflopNet(_config()).eval()
    pub, ranges = _inputs(batch=1)
    pub.seat_mask[:, -1] = False
    first = model(pub, ranges)
    changed_ranges = ranges.clone()
    changed_ranges[:, -1] = torch.rand_like(changed_ranges[:, -1]) * 1000.0
    pub.stacks[:, -1] = 99999.0
    pub.committed[:, -1] = 88888.0
    pub.status[:, -1] = 2
    second = model(pub, changed_ranges)
    for name in first.raw:
        torch.testing.assert_close(first.raw[name], second.raw[name], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first.v, second.v, atol=1e-5, rtol=1e-5)


def test_gradient_reaches_every_parameter() -> None:
    torch.manual_seed(65)
    model = PreflopNet(_config(moment=True))
    pub, ranges = _inputs(batch=2)
    pub.status[0] = torch.tensor([0, 1, 2])
    output = model(pub, ranges)
    tensors = list(output.raw.values()) + [output.v, output.vvec, output.u]
    loss = sum((tensor * torch.randn_like(tensor)).sum() for tensor in tensors if tensor is not None)
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    zero = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool((parameter.grad != 0).any())
    ]
    assert not missing
    assert not zero


def test_model_config_is_strict() -> None:
    with pytest.raises(ValueError, match="unknown poker model"):
        PokerModelConfig.from_dict({"d_modle": 16})


def test_preflop_orbit_geometry_is_exact_and_history_changes_output() -> None:
    model = PreflopNet(_config()).eval()
    torch.testing.assert_close(model.class_card_geometry.sum(1), torch.full((169,), 2.0))
    expected_overlap = model.class_card_geometry @ model.class_card_geometry.T
    torch.testing.assert_close(
        model.class_overlap_scalars[:, 0], expected_overlap.diagonal()
    )
    pub, ranges = _inputs(batch=1)
    first = model(pub, ranges)
    pub.history.action[0, 0] = 2
    pub.history.amount[0, 0] = 37.5
    second = model(pub, ranges)
    assert not torch.allclose(first.v, second.v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_preflop_conformance_paths_run_forward_backward_on_cuda() -> None:
    pub, ranges = _inputs(batch=1)
    values = {}
    for name in pub.__dataclass_fields__:
        value = getattr(pub, name)
        if isinstance(value, PackedPublicHistory):
            values[name] = PackedPublicHistory(
                **{
                    field: getattr(value, field).cuda()
                    for field in value.__dataclass_fields__
                }
            )
        else:
            values[name] = None if value is None else value.cuda()
    model = PreflopNet(_config()).cuda()
    output = model(PokerPublicBatch(**values), ranges.cuda())
    (output.v.square().mean() + output.raw["type_logits"].square().mean()).backward()
    assert torch.isfinite(output.v).all()
