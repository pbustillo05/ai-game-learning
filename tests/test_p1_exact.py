from __future__ import annotations

import numpy as np
import pytest
import torch

from pbg.games.synthetic.exact import (
    FeasibilityError,
    best_response_value,
    enumerate_public_tree,
    exact_posterior,
    exploitability,
    uniform_policy_tables,
)
from pbg.games.synthetic.family import CHECK_CALL, FOLD, RAISE, SyntheticPBG
from pbg.games.synthetic.presets import dial, kuhn, leduc


def _step(game: SyntheticPBG, ref: int, event: int) -> int:
    return int(
        game.step(
            torch.tensor([ref], dtype=torch.int32),
            torch.tensor([event], dtype=torch.int32),
        )[0]
    )


def _kuhn_equilibrium(alpha: float) -> tuple[SyntheticPBG, dict[int, np.ndarray]]:
    game = SyntheticPBG(kuhn())
    tables = uniform_policy_tables(game)
    root = int(game.root_states(1)[0])
    after_check = _step(game, root, CHECK_CALL)
    after_bet = _step(game, root, RAISE)
    after_check_bet = _step(game, after_check, RAISE)

    def table(ref: int) -> np.ndarray:
        key = int(game.encode_public(torch.tensor([ref], dtype=torch.int32))[0])
        tables[key][:] = 0.0
        return tables[key]

    root_policy = table(root)
    root_policy[:, CHECK_CALL] = (1.0 - alpha, 1.0, 1.0 - 3.0 * alpha)
    root_policy[:, RAISE] = (alpha, 0.0, 3.0 * alpha)

    after_check_policy = table(after_check)
    after_check_policy[:, CHECK_CALL] = (2.0 / 3.0, 1.0, 0.0)
    after_check_policy[:, RAISE] = (1.0 / 3.0, 0.0, 1.0)

    check_bet_policy = table(after_check_bet)
    check_bet_policy[:, FOLD] = (1.0, 2.0 / 3.0 - alpha, 0.0)
    check_bet_policy[:, CHECK_CALL] = (0.0, alpha + 1.0 / 3.0, 1.0)

    bet_policy = table(after_bet)
    bet_policy[:, FOLD] = (1.0, 2.0 / 3.0, 0.0)
    bet_policy[:, CHECK_CALL] = (0.0, 1.0 / 3.0, 1.0)
    return game, tables


def test_uniform_kuhn_is_exploitable() -> None:
    game = SyntheticPBG(kuhn())
    assert exploitability(game, uniform_policy_tables(game)) > 0.0


@pytest.mark.parametrize("alpha", [0.0, 1.0 / 6.0, 1.0 / 3.0])
def test_closed_form_kuhn_equilibrium_has_zero_exploitability(alpha: float) -> None:
    game, tables = _kuhn_equilibrium(alpha)
    assert exploitability(game, tables) == pytest.approx(0.0, abs=1e-9)
    assert best_response_value(game, tables) == pytest.approx(
        np.array([-1.0 / 18.0, 1.0 / 18.0]), abs=1e-10
    )


def test_leduc_exact_posterior_matches_seeded_profile_sampling() -> None:
    game = SyntheticPBG(leduc())
    tree = enumerate_public_tree(game)
    root = int(game.root_states(1)[0])
    chance_ref = _step(game, _step(game, root, CHECK_CALL), CHECK_CALL)
    ids, mask = game.legal_events(torch.tensor([chance_ref], dtype=torch.int32))
    observed_ref = _step(game, chance_ref, int(ids[0, mask[0]][0]))
    node_index = tree.nodes.index(observed_ref)
    posterior = exact_posterior(game, tree, node_index)
    assert np.allclose(posterior.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)

    generator = torch.Generator().manual_seed(2027)
    profiles = game.sample_mu(60_000, generator)
    chance_refs = torch.full((len(profiles),), chance_ref, dtype=torch.int32)
    chance_probabilities = game.chance_probs(chance_refs, profiles)
    chosen_columns = torch.multinomial(
        chance_probabilities, 1, replacement=True, generator=generator
    )[:, 0]
    target_column = int((ids[0, mask[0]] == ids[0, mask[0]][0]).nonzero()[0])
    samples = profiles[chosen_columns == target_column].numpy()
    n = len(samples)
    assert n > 8_000
    for player in range(game.n_players):
        observed = np.bincount(samples[:, player], minlength=game.n_types)
        expected = posterior[player] * n
        positive = expected > 0.0
        chi_square = np.sum((observed[positive] - expected[positive]) ** 2 / expected[positive])
        # df <= 4 here. A threshold of 24 has tail probability below 1e-4.
        assert chi_square < 24.0
        assert (observed[~positive] == 0).all()


def test_large_private_dial_trips_exact_feasibility_guard() -> None:
    game = SyntheticPBG(dial(public_scale=1, private_scale=3, n_players=4))
    tree = enumerate_public_tree(game)
    with pytest.raises(FeasibilityError, match="private profile product"):
        exact_posterior(game, tree, 0)


def test_dial_axes_change_public_and_private_sizes_independently() -> None:
    base = dial(public_scale=1, private_scale=1, n_players=2)
    public = dial(public_scale=3, private_scale=1, n_players=2)
    private = dial(public_scale=1, private_scale=3, n_players=2)
    assert (public.deck_size, public.hand_size) == (base.deck_size, base.hand_size)
    assert public.n_rounds > base.n_rounds
    assert private.n_rounds == base.n_rounds
    assert SyntheticPBG(private).n_types > SyntheticPBG(base).n_types
