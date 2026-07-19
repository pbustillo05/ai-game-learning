from types import SimpleNamespace

import torch

from pbg.games.poker.adapter import PokerGame, PokerPolicyAdapter
from pbg.games.poker.cards import hand_index
from pbg.games.poker.engine import CHECK_CALL, NLHEEngine, PokerConfig


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
    selected = PokerPolicyAdapter.select_actor_row(raw, profile, torch.tensor([1]))
    assert selected["iota"].item() == 901
    profile[0, 0] = 400
    selected_again = PokerPolicyAdapter.select_actor_row(raw, profile, torch.tensor([1]))
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


def test_chance_draw_uses_only_profile_and_board_complement():
    engine = NLHEEngine(PokerConfig(n_players=2))
    game = PokerGame(engine, profile_sampler=object(), showdown_evaluator=object(), partition_adapter=object())
    chance = _advance_checks(engine, engine.new_hand(0))
    profiles = torch.tensor([[hand_index(0, 1), hand_index(2, 3)]], dtype=torch.int32)
    generator = torch.Generator().manual_seed(73)
    seen = set()
    for _ in range(300):
        cards = game.sample_chance_cards(torch.tensor([chance], dtype=torch.int32), profiles, generator)
        assert cards.shape == (1, 3)
        assert not (set(cards[0].tolist()) & {0, 1, 2, 3})
        seen.add(int(cards[0, 0]))
    assert len(seen) > 40


class _FakeShowdown:
    def __init__(self):
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(payoff=torch.tensor([[1.0, -1.0]]))


def test_fixed_showdown_crosses_bound_api_and_uses_evaluator_payoff():
    engine = NLHEEngine(PokerConfig(n_players=2))
    fake = _FakeShowdown()
    game = PokerGame(engine, profile_sampler=object(), showdown_evaluator=fake, partition_adapter=object())
    ref = _advance_checks(engine, engine.new_hand(0))
    for cards in (torch.tensor([[4, 9, 14]], dtype=torch.int16),
                  torch.tensor([[19]], dtype=torch.int16),
                  torch.tensor([[24]], dtype=torch.int16)):
        ref = int(engine.deal(torch.tensor([ref], dtype=torch.int32), cards)[0])
        ref = _advance_checks(engine, ref)
    profile = torch.tensor([[hand_index(0, 1), hand_index(2, 3)]], dtype=torch.int32)
    payoff = game.reward(profile, torch.tensor([ref], dtype=torch.int32))
    assert len(fake.calls) == 1
    assert fake.calls[0]["board_cards"].shape == (1, 5)
    torch.testing.assert_close(payoff, torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(payoff.sum(1), torch.zeros(1))
