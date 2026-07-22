from types import SimpleNamespace

import torch

from pbg.games.poker.adapter import PokerPolicyAdapter
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.rules import CHECK_CALL


GEOMETRY = DeckGeometry(13)


def _raw(batch=1):
    return {
        "type_logits": torch.tensor([[0.2, -0.1, 0.7]]).expand(batch, -1).clone(),
        "omega": torch.tensor([[0.4, 0.6]]).expand(batch, -1).clone(),
        "mu": torch.tensor([[0.25, 0.75]]).expand(batch, -1).clone(),
        "s": torch.tensor([[0.2, 0.3]]).expand(batch, -1).clone(),
        "iota": torch.full((batch,), 0.15),
    }


def test_composed_event_law_entropy_and_kl_match_direct_calculation():
    adapter = PokerPolicyAdapter(n_ticks=4)
    ids = torch.arange(6, dtype=torch.int32)[None].expand(2, -1).clone()
    legal = torch.tensor([[True, True, True, False, True, True],
                          [False, True, True, True, False, True]])
    raw = _raw(2)
    probability = adapter.probabilities(raw, ids, legal)
    torch.testing.assert_close(probability.sum(1), torch.ones(2))
    assert torch.equal(probability == 0, ~legal)
    direct_entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
    torch.testing.assert_close(adapter.entropy(raw, ids, legal), direct_entropy)
    reference = _raw(2)
    reference["type_logits"] += torch.tensor([[-0.5, 0.4, 0.1]])
    q = adapter.probabilities(reference, ids, legal)
    direct_kl = (probability * (probability.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(1)
    torch.testing.assert_close(adapter.kl(raw, reference, ids, legal), direct_kl)


def test_exact_profile_selects_only_stipulated_actor_row():
    raw = {name: value[:, None].expand(-1, 1326, *value.shape[1:]).clone()
           for name, value in _raw().items()}
    # Give each row a unique marker without changing any unrelated profile seat.
    raw["iota"] = torch.arange(1326.0)[None]
    profile = torch.tensor([[17, 901]], dtype=torch.int32)
    adapter = PokerPolicyAdapter()
    selected = adapter.select_actor_row(raw, profile, torch.tensor([1]))
    assert selected["iota"].item() == 901
    profile[0, 0] = 400
    selected_again = adapter.select_actor_row(raw, profile, torch.tensor([1]))
    assert selected_again["iota"].item() == 901


def test_adapter_composes_every_private_row_without_sampling_one():
    selected = _raw()
    raw = {name: value[:, None].expand(-1, 7, *value.shape[1:]).clone() for name, value in selected.items()}
    raw["type_logits"][:, :, 0] += torch.arange(7.0)[None] / 10
    ids = torch.arange(6, dtype=torch.int32)[None]
    legal = torch.tensor([[True, True, True, False, True, True]])
    probability = PokerPolicyAdapter(n_ticks=4).probabilities(raw, ids, legal)
    assert probability.shape == (1, 7, 6)
    torch.testing.assert_close(probability.sum(-1), torch.ones(1, 7))
    assert not torch.allclose(probability[:, 0], probability[:, -1])


def test_allin_atom_targets_highest_currently_legal_denomination():
    raw = _raw()
    raw["type_logits"][:] = torch.tensor([[-100.0, -100.0, 100.0]])
    raw["iota"].fill_(1.0)
    ids = torch.arange(10, dtype=torch.int32)[None]
    legal = torch.zeros((1, 10), dtype=torch.bool)
    legal[0, CHECK_CALL] = True
    legal[0, [4, 6, 7]] = True
    probability = PokerPolicyAdapter(n_ticks=8).probabilities(raw, ids, legal)
    assert probability[0, 7] == 1.0
    assert torch.count_nonzero(probability[0, 2:7]) == 0
    sampled = PokerPolicyAdapter(n_ticks=8).sample(
        raw, ids, legal, torch.Generator().manual_seed(17)
    )
    assert sampled.tolist() == [7]


def test_legal_event_conditioning_does_not_underflow_when_illegal_type_dominates():
    raw = _raw(2)
    raw["type_logits"][:] = torch.tensor(
        [[-1000.0, -1001.0, 1000.0], [-1000.0, -1001.0, 1000.0]]
    )
    ids = torch.arange(6, dtype=torch.int32)[None].expand(2, -1).clone()
    legal = torch.zeros((2, 6), dtype=torch.bool)
    legal[0, CHECK_CALL] = True
    legal[1, :2] = True

    probability = PokerPolicyAdapter(n_ticks=4).probabilities(raw, ids, legal)

    torch.testing.assert_close(probability.sum(1), torch.ones(2))
    assert probability[0, CHECK_CALL] == 1.0
    torch.testing.assert_close(
        probability[1, :2], torch.softmax(torch.tensor([-1000.0, -1001.0]), 0)
    )
    assert torch.count_nonzero(probability.masked_select(~legal)) == 0


def _advance_checks(engine, ref):
    while int(engine.owner_of(torch.tensor([ref], dtype=torch.int32))[0]) >= 0:
        ref = int(engine.step(torch.tensor([ref], dtype=torch.int32), torch.tensor([CHECK_CALL], dtype=torch.int32))[0])
    return ref
