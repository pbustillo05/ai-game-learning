from __future__ import annotations

import torch

from pbg.games.poker.cards import (
    N_CLASSES,
    N_HANDS,
    board_conflict_mask,
    class_members,
    hand_index,
)
from pbg.games.poker.ranges import LINE_META_SIZE, RangeState


def test_scripted_action_updates_match_fp64_bayes_numerators() -> None:
    state = RangeState(2, 3)
    hands = torch.tensor(
        [hand_index(0, 1), hand_index(4, 5), hand_index(8, 9), hand_index(12, 13)]
    )
    state.log_r.fill_(-torch.inf)
    state.log_r[:, :, hands] = 0.0

    policies = torch.ones((2, N_HANDS, 2), dtype=torch.float32)
    policies[0, hands, 0] = torch.tensor([0.2, 0.4, 0.6, 0.8])
    policies[1, hands, 1] = torch.tensor([0.7, 0.3, 0.9, 0.5])
    state.update_action(
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        policies.log(),
    )

    followup = torch.ones((2, N_HANDS, 2), dtype=torch.float32)
    followup[0, hands, 1] = torch.tensor([0.9, 0.1, 0.5, 0.25])
    state.update_action(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1]),
        followup.log(),
    )

    expected0 = (
        torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
        * torch.tensor([0.9, 0.1, 0.5, 0.25], dtype=torch.float64)
    )
    expected0 /= expected0.sum()
    expected1 = torch.tensor([0.7, 0.3, 0.9, 0.5], dtype=torch.float64)
    expected1 /= expected1.sum()
    got = state.normalized(torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert torch.allclose(got[0, hands].double(), expected0, atol=1e-6, rtol=0)
    assert torch.allclose(got[1, hands].double(), expected1, atol=1e-6, rtol=0)
    outside = ~torch.isin(torch.arange(N_HANDS), hands)
    assert int(torch.count_nonzero(got[:, outside])) == 0


def test_board_masks_all_rows_without_hypothesis_renormalization() -> None:
    state = RangeState(2, 3)
    state.log_r[0, 0] = torch.linspace(-1.0, 1.0, N_HANDS)
    before = state.log_r[0].clone()
    state.update_board(
        torch.tensor([0]), torch.tensor([[0, 7, 51]], dtype=torch.int16)
    )
    conflict = board_conflict_mask(
        torch.tensor([0, 7, 51, -1, -1], dtype=torch.int8)
    )
    assert torch.isneginf(state.log_r[0, :, conflict]).all()
    assert torch.equal(state.log_r[0, :, ~conflict], before[:, ~conflict])
    normalized = state.normalized(torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]))
    assert torch.allclose(normalized.sum(dim=1), torch.ones(3))
    assert torch.equal(state.log_r[1], torch.zeros_like(state.log_r[1]))


def test_fold_snapshot_is_frozen_and_carries_public_line_metadata() -> None:
    state = RangeState(1, 3)
    log_weights = torch.linspace(-3.0, 0.0, N_HANDS)
    state.log_r[0, 1] = log_weights
    state.update_board(torch.tensor([0]), torch.tensor([3, 8, 15], dtype=torch.int16))
    state.line_meta[0, 1, :4] = torch.tensor([1.0, 2.0, 0.75, 0.5])
    pre_fold = state.log_r[0, 1].exp().clone()
    snapshot = state.fold(torch.tensor([0]), torch.tensor([1]))[0]
    assert not bool(state.live[0, 1])
    assert torch.equal(snapshot.f_weights, pre_fold)
    assert torch.equal(
        snapshot.board_at_fold,
        torch.tensor([3, 8, 15, -1, -1], dtype=torch.int8),
    )
    assert snapshot.line_meta.shape == (LINE_META_SIZE,)
    assert torch.equal(
        snapshot.line_meta[:4], torch.tensor([1.0, 2.0, 0.75, 0.5])
    )
    assert torch.equal(snapshot.line_meta[4:], torch.zeros(4))

    frozen_weights = snapshot.f_weights.clone()
    frozen_board = snapshot.board_at_fold.clone()
    state.update_board(torch.tensor([0]), torch.tensor([22], dtype=torch.int16))
    assert torch.equal(snapshot.f_weights, frozen_weights)
    assert torch.equal(snapshot.board_at_fold, frozen_board)


def test_class_aggregation_matches_brute_sums_and_uniform_class_sizes() -> None:
    state = RangeState(1, 2)
    uniform = state.to_class(torch.tensor([0]))
    expected = torch.tensor(
        [len(class_members(k)) / N_HANDS for k in range(N_CLASSES)]
    )
    assert torch.allclose(uniform[0, 0], expected, atol=1e-7, rtol=0)

    state.log_r[0, 1] = torch.linspace(-2.0, 2.0, N_HANDS)
    combo = state.normalized(torch.tensor([0]), torch.tensor([1]))[0]
    classes = state.to_class(torch.tensor([0]))[0, 1]
    brute = torch.stack([combo[class_members(k)].sum() for k in range(N_CLASSES)])
    assert torch.allclose(classes, brute, atol=1e-7, rtol=0)
    assert torch.allclose(classes.sum(), torch.tensor(1.0))


def test_clone_rows_are_independent_and_stability_preserves_probabilities() -> None:
    state = RangeState(2, 2)
    state.update_board(
        torch.tensor([0]), torch.tensor([2, 6, 10], dtype=torch.int16)
    )
    state.line_meta[0, 1, 0] = 3.0
    clone = state.clone_rows(torch.tensor([0, 1], dtype=torch.int32))
    assert torch.equal(clone, torch.tensor([2, 3], dtype=torch.int32))
    state.log_r[2, 0, 100] = -9.0
    state.board[2, 0] = 20
    state.live[2, 1] = False
    state.line_meta[2, 1, 0] = 9.0
    assert not torch.equal(state.log_r[2], state.log_r[0])
    assert int(state.board[0, 0]) == 2
    assert bool(state.live[0, 1])
    assert float(state.line_meta[0, 1, 0]) == 3.0

    before = state.normalized(torch.tensor([1]), torch.tensor([0]))
    policy = torch.full((1, N_HANDS, 1), -31.0)
    state.update_action(
        torch.tensor([1]), torch.tensor([0]), torch.tensor([0]), policy
    )
    after = state.normalized(torch.tensor([1]), torch.tensor([0]))
    assert torch.allclose(after, before)
    assert float(state.log_r[1, 0].max()) == 0.0
