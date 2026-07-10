from __future__ import annotations

import random

import torch

from pbg.core.game import CHANCE, TERMINAL
from pbg.games.synthetic.exact import enumerate_public_tree
from pbg.games.synthetic.family import CHECK_CALL, FOLD, RAISE, SyntheticPBG
from pbg.games.synthetic.presets import kuhn, leduc


def _step(game: SyntheticPBG, ref: int, event: int) -> int:
    return int(
        game.step(
            torch.tensor([ref], dtype=torch.int32),
            torch.tensor([event], dtype=torch.int32),
        )[0]
    )


def _hand_rolled_kuhn_histories() -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
    decision = {(), (CHECK_CALL,), (RAISE,), (CHECK_CALL, RAISE)}
    terminal = {
        (CHECK_CALL, CHECK_CALL),
        (CHECK_CALL, RAISE, FOLD),
        (CHECK_CALL, RAISE, CHECK_CALL),
        (RAISE, FOLD),
        (RAISE, CHECK_CALL),
    }
    return decision, terminal


def test_kuhn_public_tree_matches_independent_standard_enumeration() -> None:
    game = SyntheticPBG(kuhn())
    tree = enumerate_public_tree(game)
    decision, terminal = _hand_rolled_kuhn_histories()

    # CONTRACT-QUESTION: P01 says "9 non-root decision states and 6 terminal
    # histories per deal ordering". Its specified standard Kuhn rules instead
    # yield 9 total public states: 4 decision histories and 5 terminal histories.
    assert len(tree.nodes) == len(decision) + len(terminal) == 9
    assert int(((tree.owners >= 0) & (tree.owners < 2)).sum()) == len(decision) == 4
    assert int((tree.owners == TERMINAL).sum()) == len(terminal) == 5


def test_private_deals_legality_and_chance_rows_are_valid() -> None:
    game = SyntheticPBG(leduc())
    generator = torch.Generator().manual_seed(101)
    profiles = game.sample_mu(200, generator)
    for profile in profiles.tolist():
        cards = [card for xtype in profile for card in game.hand_of_type(xtype)]
        assert len(cards) == len(set(cards))

    tree = enumerate_public_tree(game)
    for ref, owner in zip(tree.nodes, tree.owners, strict=True):
        ids, mask = game.legal_events(torch.tensor([ref], dtype=torch.int32))
        assert ids.dtype == torch.int32
        assert mask.dtype == torch.bool
        assert not mask[0].any() if owner == TERMINAL else mask[0].any()
        if owner == CHANCE:
            reach = torch.zeros((1, game.n_players, game.n_types), dtype=torch.float64)
            profile = game.sample_profiles(
                torch.tensor([ref], dtype=torch.int32), reach, 1, generator
            )[:, 0]
            probabilities = game.chance_probs(
                torch.tensor([ref], dtype=torch.int32), profile
            )
            assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1))
            assert (probabilities[~mask] == 0).all()
        elif owner != TERMINAL:
            legal = set(ids[0, mask[0]].tolist())
            assert CHECK_CALL in legal
            assert legal <= {FOLD, CHECK_CALL, RAISE}


def test_leduc_preterminal_is_final_chance_prefix_not_fold_terminal() -> None:
    game = SyntheticPBG(leduc())
    root = int(game.root_states(1)[0])
    after_checks = _step(game, _step(game, root, CHECK_CALL), CHECK_CALL)
    assert int(game.owner_of(torch.tensor([after_checks], dtype=torch.int32))[0]) == CHANCE
    assert bool(game.is_pre_terminal(torch.tensor([after_checks], dtype=torch.int32))[0])
    assert not bool(game.is_terminal(torch.tensor([after_checks], dtype=torch.int32))[0])

    chance_ids, chance_mask = game.legal_events(
        torch.tensor([after_checks], dtype=torch.int32)
    )
    board_state = _step(game, after_checks, int(chance_ids[0, chance_mask[0]][0]))
    assert not bool(game.is_pre_terminal(torch.tensor([board_state], dtype=torch.int32))[0])
    final = _step(game, _step(game, board_state, CHECK_CALL), CHECK_CALL)
    assert bool(game.is_terminal(torch.tensor([final], dtype=torch.int32))[0])
    assert not bool(game.is_pre_terminal(torch.tensor([final], dtype=torch.int32))[0])

    fold_terminal = _step(game, _step(game, root, RAISE), FOLD)
    assert bool(game.is_terminal(torch.tensor([fold_terminal], dtype=torch.int32))[0])
    assert not bool(game.is_pre_terminal(torch.tensor([fold_terminal], dtype=torch.int32))[0])


def test_rewards_are_zero_sum_on_10k_random_terminal_completions() -> None:
    game = SyntheticPBG(leduc())
    tree = enumerate_public_tree(game)
    terminal_refs = [
        ref for ref, owner in zip(tree.nodes, tree.owners, strict=True) if owner == TERMINAL
    ]
    chooser = random.Random(8128)
    refs = torch.tensor(
        [chooser.choice(terminal_refs) for _ in range(10_000)], dtype=torch.int32
    )
    reach = torch.zeros((len(refs), game.n_players, game.n_types), dtype=torch.float64)
    profiles = game.sample_profiles(
        refs, reach, 1, torch.Generator().manual_seed(8128)
    )[:, 0]
    rewards = game.reward(profiles, refs)
    assert rewards.dtype == torch.float32
    assert torch.equal(rewards.sum(dim=-1), torch.zeros(len(refs)))


def test_reach_update_matches_direct_action_and_collision_numerators() -> None:
    game = SyntheticPBG(leduc())
    root = int(game.root_states(1)[0])
    log_reach = torch.randn(
        (1, game.n_players, game.n_types),
        generator=torch.Generator().manual_seed(73),
        dtype=torch.float64,
    )
    policy = torch.rand(
        (1, game.n_players, game.n_types, game.n_player_events),
        generator=torch.Generator().manual_seed(74),
        dtype=torch.float64,
    )
    policy /= policy.sum(dim=-1, keepdim=True)
    updated = game.reach_update(
        log_reach,
        torch.tensor([root], dtype=torch.int32),
        torch.tensor([RAISE], dtype=torch.int32),
        policy,
    )
    expected = log_reach.clone()
    expected[0, 0] += torch.log(policy[0, 0, :, RAISE])
    assert torch.allclose(updated, expected, atol=1e-12, rtol=0.0)

    chance_ref = _step(game, _step(game, root, CHECK_CALL), CHECK_CALL)
    ids, mask = game.legal_events(torch.tensor([chance_ref], dtype=torch.int32))
    event = int(ids[0, mask[0]][2])
    revealed = set(game._event_cards(chance_ref, event))
    chance_updated = game.reach_update(
        log_reach,
        torch.tensor([chance_ref], dtype=torch.int32),
        torch.tensor([event], dtype=torch.int32),
        policy,
    )
    for xtype, hand in enumerate(game.hands):
        if not revealed.isdisjoint(hand):
            assert torch.isneginf(chance_updated[0, :, xtype]).all()
        else:
            assert torch.equal(chance_updated[0, :, xtype], log_reach[0, :, xtype])

