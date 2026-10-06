from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from pbg.config import load_config
from pbg.core.persistence import CHECKPOINT_SCHEMA, model_state_sha256
from pbg.games.poker.queued.model import build_model_from_config
from pbg.inference import DealIdentity, PokerInferenceSession, parse_card
from pbg.inference_artifact import (
    INFERENCE_ARTIFACT_SCHEMA_VERSION,
    load_inference_source,
)

CONFIG = Path("configs/train/nlhe_hu_8bb_normal_smoke.yaml")


def _artifact(path: Path, device: str) -> Path:
    config = load_config(CONFIG)
    model = build_model_from_config(config.scientific_mapping(), device)
    model_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    artifact = {
        "artifact_schema": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "source_checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_checkpoint_sha256": "1" * 64,
        "iteration": 7,
        "scientific_config": config.scientific_mapping(),
        "model": model_state,
        "model_sha256": model_state_sha256(model_state),
    }
    identity = {
        name: artifact[name]
        for name in (
            "artifact_schema",
            "source_checkpoint_schema",
            "source_checkpoint_sha256",
            "iteration",
            "scientific_config",
            "model_sha256",
        )
    }
    artifact["payload_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    torch.save(artifact, path)
    return path


@pytest.mark.parametrize(
    ("scope", "legacy_key", "current_key", "value"),
    (
        ("game", "n_players", "players", 2),
        ("model", "eps", "epsilon", 1.0e-8),
        ("model", "k_act", None, 7),
        ("model", "use_branching_heads", None, False),
        ("model", "max_history_tokens", None, 32),
    ),
)
def test_model_factory_rejects_legacy_and_derived_scientific_fields(
    scope: str,
    legacy_key: str,
    current_key: str | None,
    value: object,
) -> None:
    scientific = copy.deepcopy(load_config(CONFIG).scientific_mapping())
    section = scientific[scope]
    assert isinstance(section, dict)
    if current_key is not None:
        section.pop(current_key)
    section[legacy_key] = value
    with pytest.raises(ValueError, match=legacy_key):
        build_model_from_config(scientific, "cpu")


def test_inference_artifact_rejects_legacy_and_invalid_hash(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {"inference_artifact_schema_version": "pbg-inference-v1"}, legacy
    )
    with pytest.raises(RuntimeError, match="strict shared-model"):
        load_inference_source(legacy)
    invalid = {
        "artifact_schema": INFERENCE_ARTIFACT_SCHEMA_VERSION,
        "source_checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_checkpoint_sha256": "z" * 64,
        "iteration": 0,
        "scientific_config": {},
        "model": {},
    }
    path = tmp_path / "invalid.pt"
    torch.save(invalid, path)
    with pytest.raises(RuntimeError):
        load_inference_source(path)


def test_card_parser_uses_global_compact_deck_ids() -> None:
    assert parse_card("2c") == 0
    assert parse_card("As") == 51
    with pytest.raises(ValueError, match="active deck"):
        parse_card("Tc", first_card=44)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stateful_singleton_batch_actions_and_export_identity(tmp_path: Path) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt", "cuda"), device="cuda"
    )
    first, second = session.new_hands((0, 1))
    singleton = session.predict(first, ("As", "Kd"), action_mode="argmax")
    assert singleton.selected_action in singleton.actions
    torch.testing.assert_close(singleton.probabilities.sum(), torch.tensor(1.0))
    batch = session.predict_many(
        (first, second), (("As", "Kd"), ("Qh", "Js"))
    )
    torch.testing.assert_close(batch[0].probabilities, singleton.probabilities)
    full = session.full_strategy((first, second))
    assert full.probabilities.shape[:2] == (
        2,
        session.geometry.private_classes,
    )
    assert full.values_bb.shape == (2, session.geometry.private_classes)
    session.advance(second, "call")
    assert session.source_checkpoint_sha256 == "1" * 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_explicit_deals_are_reproducible_split_distinct_and_order_invariant(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "deals.pt", "cuda")
    deals = tuple(
        DealIdentity("test_protocol_v2", "screening", 101, logical_id)
        for logical_id in range(12)
    )
    buttons = tuple(deal.logical_deal_id % 2 for deal in deals)
    ordered = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    ordered_hands = ordered.new_hands(buttons, deals=deals)
    ordered_cards = {
        deal.logical_deal_id: tuple(
            ordered.private_cards(hand, seat) for seat in range(2)
        )
        for deal, hand in zip(deals, ordered_hands, strict=True)
    }
    for deal, hand, button in zip(deals, ordered_hands, buttons, strict=True):
        flat = tuple(
            card
            for seat_cards in ordered_cards[deal.logical_deal_id]
            for card in seat_cards
        )
        assert len(flat) == len(set(flat))
        assert ordered.actor(hand) == button

    reversed_session = PokerInferenceSession.from_checkpoint(
        artifact, device="cuda:0"
    )
    reversed_hands = reversed_session.new_hands(
        tuple(reversed(buttons)), deals=tuple(reversed(deals))
    )
    reversed_cards = {
        deal.logical_deal_id: tuple(
            reversed_session.private_cards(hand, seat) for seat in range(2)
        )
        for deal, hand in zip(reversed(deals), reversed_hands, strict=True)
    }
    assert reversed_cards == ordered_cards

    # Intervening implicit work cannot change an explicitly addressed deal.
    ordered.new_hand(0)
    replay = ordered.new_hand(buttons[3], deal=deals[3])
    assert tuple(ordered.private_cards(replay, seat) for seat in range(2)) == (
        ordered_cards[deals[3].logical_deal_id]
    )

    validation = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    validation_deals = tuple(
        DealIdentity("test_protocol_v2", "validation", 101, logical_id)
        for logical_id in range(12)
    )
    validation_hands = validation.new_hands(buttons, deals=validation_deals)
    validation_cards = tuple(
        tuple(validation.private_cards(hand, seat) for seat in range(2))
        for hand in validation_hands
    )
    assert any(
        validation_cards[index] != ordered_cards[index] for index in range(12)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fixed_target_policy_is_invariant_to_response_ranges_and_private_deal(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "fixed_target.pt", "cuda")
    deal_a = DealIdentity("test_protocol_v2", "fixed-target", 303, 100)
    deal_b = DealIdentity("test_protocol_v2", "fixed-target", 303, 101)
    target_a = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    target_b = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    actual_a = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    actual_b = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    target_hand_a = target_a.new_hand(0, deal=deal_a)
    target_hand_b = target_b.new_hand(0, deal=deal_b)
    actual_hand_a = actual_a.new_hand(0, deal=deal_a)
    actual_hand_b = actual_b.new_hand(0, deal=deal_b)
    assert tuple(target_a.private_cards(target_hand_a, seat) for seat in range(2)) != (
        tuple(target_b.private_cards(target_hand_b, seat) for seat in range(2))
    )

    actions_a, target_policy_a = target_a.policy_table(target_hand_a)
    actions_b, target_policy_b = target_b.policy_table(target_hand_b)
    assert tuple(action.event_id for action in actions_a) == tuple(
        action.event_id for action in actions_b
    )
    torch.testing.assert_close(target_policy_a, target_policy_b)
    action = next(value for value in actions_a if value.kind == "check_call")
    legal = [value.event_id for value in actions_a]
    actual_policy_a = torch.zeros_like(target_policy_a)
    actual_policy_a[:, legal] = 1.0 / len(legal)
    actual_policy_b = torch.zeros_like(target_policy_b)
    selected = torch.linspace(
        0.1,
        0.9,
        actual_policy_b.shape[0],
        device=actual_policy_b.device,
    )
    actual_policy_b[:, action.event_id] = selected
    other = [event for event in legal if event != action.event_id]
    actual_policy_b[:, other] = (1.0 - selected[:, None]) / len(other)

    actual_a.advance(
        actual_hand_a,
        action,
        policy_probabilities=actual_policy_a,
        zero_probability_extension="error",
    )
    actual_b.advance(
        actual_hand_b,
        action,
        policy_probabilities=actual_policy_b,
        zero_probability_extension="error",
    )
    target_a.advance(
        target_hand_a,
        action,
        policy_probabilities=target_policy_a,
        zero_probability_extension="retain_prior",
    )
    target_b.advance(
        target_hand_b,
        action,
        policy_probabilities=target_policy_b,
        zero_probability_extension="retain_prior",
    )
    assert not torch.equal(
        actual_a._hand(actual_hand_a).log_reach,
        actual_b._hand(actual_hand_b).log_reach,
    )
    torch.testing.assert_close(
        target_a._hand(target_hand_a).log_reach,
        target_b._hand(target_hand_b).log_reach,
    )
    next_actions_a, next_policy_a = target_a.policy_table(target_hand_a)
    next_actions_b, next_policy_b = target_b.policy_table(target_hand_b)
    assert tuple(action.event_id for action in next_actions_a) == tuple(
        action.event_id for action in next_actions_b
    )
    torch.testing.assert_close(next_policy_a, next_policy_b)

    off_support = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    off_hand = off_support.new_hand(0, deal=deal_a)
    off_actions, off_policy = off_support.policy_table(off_hand)
    off_action = next(value for value in off_actions if value.kind == "check_call")
    prior = off_support._hand(off_hand).log_reach.clone()
    off_support.advance(
        off_hand,
        off_action,
        policy_probabilities=torch.zeros_like(off_policy),
        zero_probability_extension="retain_prior",
    )
    torch.testing.assert_close(off_support._hand(off_hand).log_reach, prior)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_ranges_keep_compatible_support_after_underflow_and_keyed_chance(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "range_support.pt", "cuda")
    deal = DealIdentity("test_protocol_v2_1", "range-support", 404, 7)
    session = PokerInferenceSession.from_checkpoint(artifact, device="cuda:0")
    hand = session.new_hand(0, deal=deal)

    # A softmax action may underflow to exact zero for some private rows. Once
    # the action has positive aggregate support those zeros are numerical, not
    # structural exclusions, and must not erase otherwise possible hands.
    actions, policy = session.policy_table(hand)
    call = next(value for value in actions if value.kind == "check_call")
    sparse = torch.zeros_like(policy)
    sparse[0, call.event_id] = 1.0
    session.advance(
        hand,
        call,
        policy_probabilities=sparse,
        zero_probability_extension="error",
    )
    actor_reach = session._hand(hand).log_reach[0, 0]
    assert bool(torch.isfinite(actor_reach).all())

    actions, policy = session.policy_table(hand)
    check = next(value for value in actions if value.kind == "check_call")
    session.advance(
        hand,
        check,
        policy_probabilities=policy,
        zero_probability_extension="error",
    )
    assert bool(session.machine.needs_chance(session._hand(hand).state)[0])

    # Simulate a reference history whose pre-chance conditional support has
    # numerically collapsed. The deterministic extension is uniform over the
    # hands compatible with the newly public board, never over conflicts.
    session._hand(hand).log_reach.fill_(-torch.inf)
    session.deal_keyed(hand, deal=deal, decision_index=2)
    item = session._hand(hand)
    board = item.state.board_cards[0]
    conflict = (
        (session._combo_cards[:, :, None] == board[None, None])
        & board[None, None].ne(255)
    ).any(2).any(1)
    assert bool(torch.isfinite(item.log_reach[:, :, ~conflict]).all())
    assert bool(torch.isneginf(item.log_reach[:, :, conflict]).all())
    assert session.policy_table(hand)[1].isfinite().all()

@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_mps_preflop_stays_on_metal_and_postflop_fails_early(
    tmp_path: Path,
) -> None:
    session = PokerInferenceSession.from_checkpoint(
        _artifact(tmp_path / "policy.pt", "mps"), device="mps"
    )
    assert all(value.device.type == "mps" for value in session.model.parameters())
    hand = session.new_hand()
    decision = session.predict(hand, ("As", "Kd"))
    torch.testing.assert_close(decision.probabilities.sum(), torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="unsupported before CPU staging"):
        session.deal(hand, ("Jh", "Qd", "Kc"))
