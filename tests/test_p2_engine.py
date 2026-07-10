from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pbg.core.game import CHANCE, PRETERM, TERMINAL
from pbg.games.poker.engine import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    FOLDED,
    LIVE,
    RAISE_OFFSET,
    NLHEEngine,
    PokerConfig,
    PotStrata,
    _State,
    assemble_showdown_payoff,
)


def _step(game: NLHEEngine, ref: int, event: int) -> int:
    return int(
        game.step(
            torch.tensor([ref], dtype=torch.int32),
            torch.tensor([event], dtype=torch.int32),
        )[0]
    )


def _check_call(game: NLHEEngine, ref: int) -> int:
    return _step(game, ref, CHECK_CALL)


def test_blinds_action_order_limp_fold_and_button_rotation() -> None:
    game = NLHEEngine(PokerConfig(n_players=3, n_ticks=5))
    root = int(game.root_states(1)[0])
    assert game.states.button[root] == 0
    assert game.states.committed_street[root] == (0.0, 0.5, 1.0)
    assert game.states.actor[root] == 0

    after_limp = _check_call(game, root)
    assert game.states.actor[after_limp] == 1
    after_fold = _step(game, after_limp, FOLD)
    assert game.states.status[after_fold][1] == FOLDED
    assert game.states.actor[after_fold] == 2
    chance = _check_call(game, after_fold)
    assert int(game.owner_of(torch.tensor([chance], dtype=torch.int32))[0]) == CHANCE

    flop = int(
        game.deal(
            torch.tensor([chance], dtype=torch.int32),
            torch.tensor([[11, 3, 8]], dtype=torch.int16),
        )[0]
    )
    assert game.states.board[flop] == (3, 8, 11, -1, -1)
    assert game.states.actor[flop] == 2  # first non-folded seat left of button

    next_root = game.next_hand(chance)
    assert game.states.button[next_root] == 1
    assert game.states.committed_street[next_root] == (1.0, 0.0, 0.5)
    assert game.states.actor[next_root] == 1


def test_heads_up_order_minimum_raise_grid_and_fold_payoff() -> None:
    game = NLHEEngine(PokerConfig(n_players=2, stack_bb=20.0, n_ticks=5))
    root = int(game.root_states(1)[0])
    assert game.states.committed_street[root] == (0.5, 1.0)
    assert game.states.actor[root] == 0  # button/small blind first preflop
    targets = game.raise_targets(torch.tensor([root], dtype=torch.int32))[0]
    assert torch.equal(targets, torch.tensor([2.0, 6.5, 11.0, 15.5, 20.0]))
    raised = _step(game, root, RAISE_OFFSET)
    assert game.states.current_bet[raised] == pytest.approx(2.0)
    assert game.states.last_full_raise[raised] == pytest.approx(1.0)
    assert game.raise_targets(torch.tensor([raised], dtype=torch.int32))[0, 0] == 3.0

    folded = _step(game, raised, FOLD)
    assert int(game.owner_of(torch.tensor([folded], dtype=torch.int32))[0]) == TERMINAL
    reward = game.reward(torch.empty((1, 2), dtype=torch.int32), torch.tensor([folded]))
    assert torch.equal(reward, torch.tensor([[1.0, -1.0]]))


def test_short_all_in_does_not_reopen_prior_actors() -> None:
    game = NLHEEngine(PokerConfig(n_players=3, stack_bb=100.0, n_ticks=4))
    # A public fixture with unequal effective stacks: seats 0 and 1 have acted
    # at 8, while seat 2 can only raise all-in to 10, short of the last full
    # raise size 7.  Seat 0 must still respond, but cannot re-raise.
    ref = game.states.append(
        _State(
            street=1,
            board=(0, 5, 10),
            stacks=(92.0, 92.0, 2.0),
            committed_street=(8.0, 8.0, 8.0),
            committed_total=(8.0, 8.0, 98.0),
            status=(LIVE, LIVE, LIVE),
            actor=2,
            last_full_raise=7.0,
            current_bet=8.0,
            acted=(True, True, False),
            button=0,
            phase=0,
            betting_closed=False,
            n_raises=1,
            action_history=(),
        )
    )
    _, legal = game.legal_events(torch.tensor([ref], dtype=torch.int32))
    assert legal[0, RAISE_OFFSET:].tolist() == [False, False, False, True]
    short = _step(game, ref, RAISE_OFFSET + 3)
    assert game.states.status[short][2] == ALL_IN
    assert game.states.actor[short] == 0
    _, legal = game.legal_events(torch.tensor([short], dtype=torch.int32))
    assert legal[0, FOLD] and legal[0, CHECK_CALL]
    assert not legal[0, RAISE_OFFSET:].any()


def test_scripted_multiway_all_in_closes_at_runout_preterminal() -> None:
    game = NLHEEngine(PokerConfig(n_players=3, stack_bb=10.0, n_ticks=5))
    ref = int(game.root_states(1)[0])
    ref = _step(game, ref, RAISE_OFFSET + 4)
    assert game.states.status[ref][0] == ALL_IN
    ref = _check_call(game, ref)
    assert game.states.status[ref][1] == ALL_IN
    ref = _check_call(game, ref)
    assert game.states.status[ref] == (ALL_IN, ALL_IN, ALL_IN)
    assert int(game.owner_of(torch.tensor([ref]))[0]) == PRETERM
    assert game.states.pot[ref] == pytest.approx(30.0)
    strata = game.pot_strata(torch.tensor([ref], dtype=torch.int32))
    assert torch.equal(strata.stratum_amount, torch.tensor([[30.0]]))
    assert strata.stratum_players.all()


def test_check_down_transitions_and_showdown_preterminal() -> None:
    game = NLHEEngine(PokerConfig(n_players=2, n_ticks=3))
    ref = int(game.root_states(1)[0])
    # Preflop call/check, then three checked streets.
    ref = _check_call(game, ref)
    ref = _check_call(game, ref)
    assert int(game.owner_of(torch.tensor([ref]))[0]) == CHANCE
    for cards in ([9, 2, 6], [22], [51]):
        ref = int(game.deal(torch.tensor([ref]), torch.tensor([cards], dtype=torch.int16))[0])
        assert game.states.actor[ref] == 1  # non-button acts first postflop
        ref = _check_call(game, ref)
        ref = _check_call(game, ref)
    assert int(game.owner_of(torch.tensor([ref]))[0]) == PRETERM
    assert bool(game.is_pre_terminal(torch.tensor([ref]))[0])
    assert not bool(game.is_terminal(torch.tensor([ref]))[0])
    assert game.states.board[ref] == (2, 6, 9, 22, 51)


def test_nested_pot_strata_and_affine_tied_assembly() -> None:
    game = NLHEEngine(PokerConfig(n_players=3))
    ref = game.states.append(
        _State(
            street=3,
            board=(0, 4, 8, 12, 16),
            stacks=(0.0, 50.0, 80.0),
            committed_street=(0.0, 0.0, 0.0),
            committed_total=(100.0, 50.0, 20.0),
            status=(ALL_IN, LIVE, FOLDED),
            actor=-1,
            last_full_raise=1.0,
            current_bet=0.0,
            acted=(True, True, True),
            button=0,
            phase=3,
            betting_closed=True,
            n_raises=0,
            action_history=(),
        )
    )
    strata = game.pot_strata(torch.tensor([ref], dtype=torch.int32))
    assert torch.equal(strata.stratum_amount, torch.tensor([[60.0, 60.0, 50.0]]))
    assert strata.stratum_players.tolist() == [
        [[True, True, False], [True, True, False], [True, False, False]]
    ]
    shares = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]])
    payoff = assemble_showdown_payoff(strata, shares)
    assert torch.equal(payoff, torch.tensor([[10.0, 10.0, -20.0]]))
    assert float(payoff.sum()) == 0.0

    with pytest.raises(ValueError, match="ineligible"):
        assemble_showdown_payoff(
            PotStrata(
                strata.stratum_players,
                strata.stratum_amount,
                strata.payoff_baseline,
            ),
            shares + torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0, 0, 0]]]),
        )


def test_engine_and_cards_have_no_private_settlement_implementation() -> None:
    root = Path(__file__).parents[1] / "pbg" / "games" / "poker"
    text = (root / "engine.py").read_text() + (root / "cards.py").read_text()
    forbidden = ("evaluate7", "hand_rank", "kicker_table", "winner_mask", "side_pot_winner")
    assert not any(symbol in text for symbol in forbidden)
