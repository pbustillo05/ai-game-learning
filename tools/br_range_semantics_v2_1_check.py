#!/usr/bin/env python3
"""Small CUDA audit of fixed-target range semantics for protocol v2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from pbg.inference import DealIdentity, PokerInferenceSession

PROTOCOL = "fixed_target_paired_response_gain_v2_1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_call(actions: tuple[object, ...]) -> object:
    return next(action for action in actions if action.kind == "check_call")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    deal = DealIdentity(PROTOCOL, "range-semantics-audit", 8675309, 0)

    # Reachable matched history: compare the old response-conditioned target
    # context with the corrected target-reference-conditioned context.
    probe = PokerInferenceSession.from_checkpoint(args.probe, device=args.device)
    corrected = PokerInferenceSession.from_checkpoint(args.target, device=args.device)
    adaptive = PokerInferenceSession.from_checkpoint(args.target, device=args.device)
    probe_hand = probe.new_hand(0, deal=deal)
    corrected_hand = corrected.new_hand(0, deal=deal)
    adaptive_hand = adaptive.new_hand(0, deal=deal)
    target_actions, target_policy = corrected.policy_table(corrected_hand)
    probe_actions, probe_policy = probe.policy_table(probe_hand)
    if tuple(action.event_id for action in target_actions) != tuple(
        action.event_id for action in probe_actions
    ):
        raise RuntimeError("reachable-history action sets differ")
    action = check_call(target_actions)
    root_actor = corrected.actor(corrected_hand)
    root_private = corrected.private_policy_index(corrected_hand, root_actor)
    target_action_probability = float(target_policy[root_private, action.event_id])
    response_action_probability = float(probe_policy[root_private, action.event_id])
    corrected.advance(
        corrected_hand,
        action,
        policy_probabilities=target_policy,
        zero_probability_extension="retain_prior",
    )
    adaptive.advance(
        adaptive_hand,
        action,
        policy_probabilities=probe_policy,
        zero_probability_extension="clamp",
    )
    corrected_actions, corrected_policy = corrected.policy_table(corrected_hand)
    adaptive_actions, adaptive_policy = adaptive.policy_table(adaptive_hand)
    if tuple(action.event_id for action in corrected_actions) != tuple(
        action.event_id for action in adaptive_actions
    ):
        raise RuntimeError("matched public history produced different legal actions")
    columns = torch.tensor(
        [action.event_id for action in corrected_actions],
        dtype=torch.int64,
        device=corrected_policy.device,
    )
    opponent = corrected.actor(corrected_hand)
    private_row = corrected.private_policy_index(corrected_hand, opponent)
    corrected_private = corrected_policy[private_row].index_select(0, columns)
    adaptive_private = adaptive_policy[private_row].index_select(0, columns)
    reachable_difference = (corrected_policy - adaptive_policy).abs()

    # Numerical underflow: the selected action has positive aggregate support,
    # although all but one natural private row are exact zero.
    underflow = PokerInferenceSession.from_checkpoint(args.target, device=args.device)
    underflow_hand = underflow.new_hand(0, deal=deal)
    actions, policy = underflow.policy_table(underflow_hand)
    action_underflow = check_call(actions)
    sparse = torch.zeros_like(policy)
    sparse[0, action_underflow.event_id] = 1.0
    actor = underflow.actor(underflow_hand)
    underflow_prior = underflow._hand(underflow_hand).log_reach[0, actor].clone()
    prior_finite = int(torch.isfinite(underflow_prior).sum())
    selected_natural = sparse[:, action_underflow.event_id].float()
    if underflow._family(underflow._hand(underflow_hand).state) == "pre":
        selected_combo = selected_natural.index_select(
            0, underflow.model.combo_class
        )
    else:
        selected_combo = selected_natural
    # Independent numerical reference: construct the Bayes update directly in
    # float64 log space, without calling advance or its support branch.
    tiny64 = torch.tensor(
        torch.finfo(torch.float32).tiny,
        dtype=torch.float64,
        device=underflow.device,
    )
    reference_log_reach64 = underflow_prior.double() + selected_combo.double().clamp_min(
        tiny64
    ).log()
    reference_log_posterior64 = reference_log_reach64 - torch.logsumexp(
        reference_log_reach64, dim=0
    )
    underflow.advance(
        underflow_hand,
        action_underflow,
        policy_probabilities=sparse,
        zero_probability_extension="error",
    )
    underflow_reach = underflow._hand(underflow_hand).log_reach[0, actor]
    actual_log_posterior64 = underflow_reach.double() - torch.logsumexp(
        underflow_reach.double(), dim=0
    )
    reference_unnormalized_error = float(
        (underflow_reach.double() - reference_log_reach64).abs().max()
    )
    reference_normalized_error = float(
        (actual_log_posterior64 - reference_log_posterior64).abs().max()
    )

    # Genuinely off-support history: no private row assigns the action positive
    # probability, so the deterministic fixed-policy extension retains prior.
    off_support = PokerInferenceSession.from_checkpoint(args.target, device=args.device)
    off_hand = off_support.new_hand(0, deal=deal)
    actions, policy = off_support.policy_table(off_hand)
    off_action = check_call(actions)
    off_prior = off_support._hand(off_hand).log_reach.clone()
    off_support.advance(
        off_hand,
        off_action,
        policy_probabilities=torch.zeros_like(policy),
        zero_probability_extension="retain_prior",
    )
    off_after = off_support._hand(off_hand).log_reach

    # Impossible hands: expose a keyed flop and check that every board conflict
    # is excluded while each seat keeps compatible posterior support.
    board_session = PokerInferenceSession.from_checkpoint(
        args.target, device=args.device
    )
    board_hand = board_session.new_hand(0, deal=deal)
    decision = 0
    while not board_session.needs_chance(board_hand):
        actions, policy = board_session.policy_table(board_hand)
        board_session.advance(
            board_hand,
            check_call(actions),
            policy_probabilities=policy,
            zero_probability_extension="retain_prior",
        )
        decision += 1
    board_session.deal_keyed(board_hand, deal=deal, decision_index=decision)
    item = board_session._hand(board_hand)
    board = item.state.board_cards[0]
    conflict = (
        (board_session._combo_cards[:, :, None] == board[None, None])
        & board[None, None].ne(255)
    ).any(2).any(1)
    compatible = ~conflict

    report = {
        "schema": "pbg-fixed-target-range-semantics-audit-v2.1",
        "protocol_version": PROTOCOL,
        "device": str(corrected.device),
        "probe_artifact": str(args.probe),
        "probe_artifact_sha256": sha256_file(args.probe),
        "target_artifact": str(args.target),
        "target_artifact_sha256": sha256_file(args.target),
        "deal_identity": {
            "split_namespace": deal.split_namespace,
            "evaluation_seed": deal.evaluation_seed,
            "logical_deal_id": deal.logical_deal_id,
        },
        "reachable_history": {
            "observed_action_event": action.event_id,
            "root_actor": root_actor,
            "deviation_response_seat": root_actor,
            "dealt_actor_private_row": root_private,
            "target_observed_action_probability": target_action_probability,
            "response_observed_action_probability": response_action_probability,
            "next_actor": opponent,
            "frozen_target_opponent_seat": opponent,
            "compared_policy_owner": "frozen_target_opponent",
            "max_abs_target_policy_difference": float(reachable_difference.max()),
            "mean_abs_target_policy_difference": float(reachable_difference.mean()),
            "dealt_opponent_private_row": private_row,
            "dealt_private_policy_l1_difference": float(
                (corrected_private - adaptive_private).abs().sum()
            ),
            "dealt_private_policy_total_variation": float(
                0.5 * (corrected_private - adaptive_private).abs().sum()
            ),
            "corrected_policy": corrected_private.detach().cpu().tolist(),
            "legacy_adaptive_policy": adaptive_private.detach().cpu().tolist(),
            "interpretation": "same reachable public history and private hand; only target range bookkeeping differs",
        },
        "numerical_underflow": {
            "selected_action_has_positive_aggregate_support": True,
            "prior_finite_private_hands": prior_finite,
            "posterior_finite_private_hands": int(torch.isfinite(underflow_reach).sum()),
            "minimum_finite_log_reach": float(underflow_reach[torch.isfinite(underflow_reach)].min()),
            "expected_fp32_log_tiny": float(
                torch.tensor(torch.finfo(torch.float32).tiny).log()
            ),
            "independent_float64_log_space_reference": {
                "construction": "prior_log_reach + log(max(selected_action_probability, float32_tiny)); normalize with logsumexp",
                "max_abs_unnormalized_log_reach_error": reference_unnormalized_error,
                "max_abs_normalized_log_posterior_error": reference_normalized_error,
                "tolerance": 1.0e-5,
                "matches": bool(
                    reference_unnormalized_error <= 1.0e-5
                    and reference_normalized_error <= 1.0e-5
                ),
            },
        },
        "genuinely_off_support": {
            "selected_action_has_positive_aggregate_support": False,
            "retained_prior_exactly": bool(torch.equal(off_prior, off_after)),
        },
        "impossible_public_card_conflicts": {
            "board_cards": [int(card) for card in board[board.ne(255)].tolist()],
            "conflicting_private_hands": int(conflict.sum()),
            "all_conflicting_log_reach_negative_infinity": bool(
                torch.isneginf(item.log_reach[:, :, conflict]).all()
            ),
            "compatible_private_hands": int(compatible.sum()),
            "finite_compatible_support_by_seat": [
                int(torch.isfinite(item.log_reach[0, seat, compatible]).sum())
                for seat in range(board_session.geometry.players)
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
