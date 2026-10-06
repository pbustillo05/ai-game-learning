"""Pre-terminal posterior exposure for sequential CUDA campaigns.

The active observer measures each player's exact combo marginal at the completed
pre-terminal prefix and weights its exp(entropy) by that player's fresh ancestor
decision count. No future chance reveal, learner reuse, or Monte Carlo sample
count enters the scheduling clock. The old per-decision helper is retained as
an entropy primitive; the active observer never measures intermediate ranges.
"""

from __future__ import annotations

import math
from collections.abc import MutableSequence
from typing import Any

import torch
from torch import Tensor

from pbg.core.contracts import PrivateEstimator, RolloutMode
from pbg.games.poker.queued.hooks import PAD_CARD, SequentialPokerHooks

EXPOSURE_SCHEMA = "preterminal-trajectory-weighted-posterior-exposure-v2"
POSTERIOR_SEMANTICS = (
    "heads_up_player_exact_combo_marginal_given_preterminal_history_v2"
)
GRADIENT_CORRELATION_PLAN = (
    "Replay matched public-history batches over independent rollout/completion "
    "seeds, retain posterior-weighted per-combo gradient sketches before combo "
    "summation, and pass [repetition, combo, feature] sketches to "
    "gradient_covariance_diagnostic. Advantage variance alone is insufficient."
)


class ExactComboPosteriorUnavailable(RuntimeError):
    """The observer was given a class-collapsed or otherwise inexact posterior."""


def _validate_exact_inputs(
    log_reach: Tensor,
    actor: Tensor,
    board_cards: Tensor,
    combo_cards: Tensor,
    selected: Tensor,
) -> None:
    if log_reach.ndim != 3 or not log_reach.dtype.is_floating_point:
        raise TypeError("exact log reach must be floating [decision, player, combo]")
    rows, players, hands = log_reach.shape
    if (
        actor.shape != (rows,)
        or actor.dtype not in (torch.int32, torch.int64)
        or actor.device != log_reach.device
    ):
        raise ValueError("actor must align with exact log reach")
    if (
        board_cards.shape != (rows, 5)
        or board_cards.device != log_reach.device
        or board_cards.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("known board cards must be integer [decision, 5]")
    if (
        combo_cards.shape != (hands, 2)
        or combo_cards.device != log_reach.device
        or combo_cards.dtype not in (torch.int16, torch.int32, torch.int64)
    ):
        raise ExactComboPosteriorUnavailable(
            "posterior width does not match the exact combo-card domain"
        )
    if (
        selected.shape != (rows,)
        or selected.dtype != torch.bool
        or selected.device != log_reach.device
    ):
        raise ValueError("selected decision mask must align with exact log reach")
    torch._assert_async(
        ((actor >= 0) & (actor < players)).all(),
        "actor lies outside the exact reach player axis",
    )


def current_posterior_exposure(
    log_reach: Tensor,
    actor: Tensor,
    board_cards: Tensor,
    combo_cards: Tensor,
    *,
    selected: Tensor | None = None,
    sampled: bool = False,
    decision_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Measure raw and square-root entropy clocks on one fresh decision batch.

    ``log_reach`` must still have exact combo width.  Passing a 169-class
    preflop posterior therefore fails instead of silently assuming uniform
    within-class mass.  The rollout hook calls this function before that
    collapse, where its reconstructed preflop exact reach is lossless because
    the preflop policy is class-invariant and compact storage preserves each
    class's total reach mass.
    """

    rows = log_reach.shape[0] if log_reach.ndim else 0
    if selected is None:
        selected = torch.ones(rows, dtype=torch.bool, device=log_reach.device)
    _validate_exact_inputs(log_reach, actor, board_cards, combo_cards, selected)
    if decision_weights is None:
        decision_weights = torch.ones(rows, dtype=torch.int64, device=log_reach.device)
    if (
        decision_weights.shape != (rows,)
        or decision_weights.device != log_reach.device
        or decision_weights.dtype != torch.int64
    ):
        raise ValueError("decision weights must be device-local int64 row counts")
    torch._assert_async(decision_weights.ge(0).all(), "negative decision count")
    count = (decision_weights * selected).sum(dtype=torch.int64)
    if sampled:
        value = count.to(torch.float64)
        return {
            "fresh_player_decisions": count,
            "sum_posterior_exp_entropy": value,
            "sum_sqrt_posterior_exp_entropy": value.clone(),
            "sum_posterior_entropy_nats": value.new_zeros(()),
        }

    players = log_reach.shape[1]
    if players != 2:
        raise ExactComboPosteriorUnavailable(
            "collision-conditioned exact posterior currently requires heads-up reach"
        )
    hands = combo_cards.shape[0]
    deck_cards = (1 + math.isqrt(1 + 8 * hands)) // 2
    if deck_cards * (deck_cards - 1) // 2 != hands:
        raise ExactComboPosteriorUnavailable(
            "combo-card domain is not a complete two-card deck domain"
        )
    first_card = 52 - deck_cards
    torch._assert_async(
        ((combo_cards >= first_card) & (combo_cards < 52)).all(),
        "combo-card domain is not the configured highest-card deck",
    )

    row = torch.arange(rows, device=log_reach.device)
    actor64 = actor.to(torch.int64)
    opponent64 = 1 - actor64
    actor_log_reach = log_reach[row, actor64].float()
    opponent_log_reach = log_reach[row, opponent64].float()
    known = board_cards.ne(PAD_CARD)
    conflict = (
        (
            known[..., None, None]
            & board_cards.to(combo_cards.dtype)[..., None, None].eq(
                combo_cards[None, None]
            )
        )
        .any(1)
        .any(-1)
    )
    torch._assert_async(
        (~conflict[:, None] | ~torch.isfinite(log_reach)).all(),
        "exact profile reach assigns mass to a known-board conflict",
    )

    # Stable unnormalized reaches are sufficient because both arbitrary scale
    # factors cancel when the actor marginal is normalized.
    actor_gauge = actor_log_reach.max(1, keepdim=True).values
    opponent_gauge = opponent_log_reach.max(1, keepdim=True).values
    actor_weight = torch.exp(actor_log_reach - actor_gauge).masked_fill(
        ~torch.isfinite(actor_log_reach), 0.0
    )
    opponent_weight = torch.exp(opponent_log_reach - opponent_gauge).masked_fill(
        ~torch.isfinite(opponent_log_reach), 0.0
    )
    card_index = (combo_cards.to(torch.int64) - first_card).transpose(0, 1)
    opponent_card_mass = opponent_weight.new_zeros((rows, deck_cards))
    opponent_card_mass.scatter_add_(1, card_index[0].expand(rows, -1), opponent_weight)
    opponent_card_mass.scatter_add_(1, card_index[1].expand(rows, -1), opponent_weight)
    compatible_opponent_mass = (
        opponent_weight.sum(1, keepdim=True)
        - opponent_card_mass[:, card_index[0]]
        - opponent_card_mass[:, card_index[1]]
        + opponent_weight
    ).clamp_min_(0.0)
    unnormalized = actor_weight * compatible_opponent_mass
    normalizer = unnormalized.sum(1)
    torch._assert_async(
        (~selected | normalizer.gt(0)).all(),
        "selected decision has no collision-compatible exact-combo posterior mass",
    )
    probability = (
        unnormalized
        / normalizer.clamp_min(torch.finfo(unnormalized.dtype).tiny)[:, None]
    )
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(1)
    selected_float = selected.to(torch.float64) * decision_weights.to(torch.float64)
    entropy64 = entropy.to(torch.float64)
    return {
        "fresh_player_decisions": count,
        "sum_posterior_exp_entropy": (entropy64.exp() * selected_float).sum(),
        "sum_sqrt_posterior_exp_entropy": (
            (0.5 * entropy64).exp() * selected_float
        ).sum(),
        "sum_posterior_entropy_nats": (entropy64 * selected_float).sum(),
    }


def weighted_advantage_dispersion(
    posterior: Tensor, advantage: Tensor
) -> dict[str, float]:
    """Describe within-posterior target variation without claiming independence."""

    if (
        posterior.ndim != 2
        or posterior.shape != advantage.shape
        or not posterior.dtype.is_floating_point
        or not advantage.dtype.is_floating_point
        or posterior.device != advantage.device
    ):
        raise ValueError("posterior and advantage must be aligned floating matrices")
    if not bool(torch.isfinite(posterior).all()) or not bool(
        torch.isfinite(advantage).all()
    ):
        raise ValueError("posterior and advantage must be finite")
    if bool((posterior < 0).any()):
        raise ValueError("posterior must be nonnegative")
    mass = posterior.sum(1, keepdim=True)
    if bool(mass.le(0).any()):
        raise ValueError("each posterior row must have positive mass")
    probability = posterior / mass
    mean = (probability * advantage).sum(1)
    variance = (probability * (advantage - mean[:, None]).square()).sum(1)
    second_moment = (probability * advantage.square()).sum(1)
    return {
        "decision_count": posterior.shape[0],
        "mean_weighted_advantage_variance": float(variance.mean()),
        "mean_weighted_advantage_second_moment": float(second_moment.mean()),
        "mean_absolute_weighted_advantage": float(mean.abs().mean()),
        "interpretation": (
            "Weighted advantage dispersion only; this is not a correlation "
            "estimate, effective sample size, or independent-hand count."
        ),
    }


def gradient_covariance_diagnostic(contributions: Tensor) -> dict[str, Any]:
    """Estimate a variance design effect from repeated per-combo gradients.

    ``contributions`` is ``[independent repetition, combo component, feature]``
    and must already include posterior weights.  Repeated rollouts are required:
    a single decision's advantage variance cannot identify gradient covariance.
    """

    if (
        contributions.ndim != 3
        or not contributions.dtype.is_floating_point
        or contributions.shape[0] < 2
        or contributions.shape[1] < 1
        or contributions.shape[2] < 1
    ):
        raise ValueError(
            "gradient contributions must be floating [repetition>=2, combo, feature]"
        )
    if not bool(torch.isfinite(contributions).all()):
        raise ValueError("gradient contributions must be finite")
    centered = contributions.double() - contributions.double().mean(0, keepdim=True)
    marginal_energy = centered.square().sum() / (contributions.shape[0] - 1)
    if float(marginal_energy) <= 0:
        raise ValueError("gradient covariance needs nonzero repeated variation")
    aggregate = centered.sum(1)
    aggregate_energy = aggregate.square().sum() / (contributions.shape[0] - 1)
    design_effect = float(aggregate_energy / marginal_energy)
    conservative_discount = max(1.0, design_effect)
    component_count = contributions.shape[1]
    variance_equivalent = (
        math.inf if design_effect == 0 else component_count / design_effect
    )
    return {
        "method": "repeated_posterior_weighted_gradient_covariance_energy_v1",
        "independent_repetitions": contributions.shape[0],
        "combo_components": component_count,
        "gradient_features": contributions.shape[2],
        "variance_design_effect": design_effect,
        "conservative_scheduling_discount": conservative_discount,
        "variance_equivalent_component_count": variance_equivalent,
        "interpretation": (
            "A repeated-gradient covariance diagnostic. The variance-equivalent "
            "component count is not labeled as independent hands or completion ESS."
        ),
    }


def scheduling_exposure(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    exponent: float = 1.0,
    covariance_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate an exposure clock, optionally with a measured covariance discount."""

    if exponent not in (1.0, 0.5):
        raise ValueError("posterior exposure exponent must be 1 or 0.5")
    field = (
        "sum_posterior_exp_entropy"
        if exponent == 1.0
        else "sum_sqrt_posterior_exp_entropy"
    )
    raw = sum(float(record[field]) for record in records)
    discount = 1.0
    if covariance_diagnostic is not None:
        if covariance_diagnostic.get("method") != (
            "repeated_posterior_weighted_gradient_covariance_energy_v1"
        ):
            raise ValueError("correlation discount lacks repeated-gradient provenance")
        discount = float(covariance_diagnostic["conservative_scheduling_discount"])
        if not math.isfinite(discount) or discount < 1:
            raise ValueError("correlation scheduling discount must be finite and >= 1")
    return {
        "exponent": exponent,
        "raw_exposure": raw,
        "correlation_discount": discount,
        "scheduled_exposure": raw / discount,
        "completion_effective_hands_used": False,
        "is_independent_sample_count": False,
    }


def _empty_accumulator(device: torch.device) -> dict[str, Tensor]:
    return {
        "fresh_player_decisions": torch.zeros((), dtype=torch.int64, device=device),
        "sum_posterior_exp_entropy": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "sum_sqrt_posterior_exp_entropy": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "sum_posterior_entropy_nats": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
    }


def _instrument_sequential_hook(
    hook: SequentialPokerHooks,
    accumulators: dict[int, dict[str, dict[str, Tensor]]],
) -> None:
    original = hook._resolved_model_context
    combo_ids = torch.arange(
        hook.geometry.private_hands,
        dtype=torch.int32,
        device=hook.owners.device,
    )
    combo_cards = hook.machine.combo_cards(combo_ids).to(torch.int32)

    def observed(batch: Any) -> Any:
        log_reach, history = original(batch)
        rows = batch.rows
        model_input = hook._forward_model_input(batch)
        private = batch.fields.get(
            "sampled_private_state", batch.fields["true_profile"]
        ).reshape(-1, hook.geometry.players)
        state = hook.machine.state_from_model_input(
            batch.input_type, model_input, private
        )
        actor = state.to_act.to(torch.int64)
        selected = torch.ones_like(actor, dtype=torch.bool)
        if "role_selector" in batch.fields:
            selected &= actor.eq(
                batch.fields["role_selector"].reshape(-1).to(torch.int64)
            )
        sampled = hook.mode.private_estimator is PrivateEstimator.SAMPLED
        for local_lane, lane_tensor in enumerate(batch.lanes):
            lane = int(lane_tensor)
            begin = local_lane * rows
            end = begin + rows
            lane_measured = current_posterior_exposure(
                log_reach[begin:end],
                actor[begin:end],
                state.board_cards[begin:end],
                combo_cards,
                selected=selected[begin:end],
                sampled=sampled,
            )
            lane_partitions = accumulators.setdefault(lane, {})
            if batch.input_type not in lane_partitions:
                lane_partitions[batch.input_type] = _empty_accumulator(log_reach.device)
            partition = lane_partitions[batch.input_type]
            for name, value in lane_measured.items():
                partition[name].add_(value)
        return log_reach, history

    hook._resolved_model_context = observed  # type: ignore[method-assign]


def sequential_leaf_decision_counts(
    topology, bindings, leaf_count, families, players, device
):
    """Map each trained decision to its unique descendant leaf, on CUDA.

    Counts retain the acting player and input partition. Pointer doubling has
    logarithmic depth and counts every fresh player node exactly once; learner
    resampling and completion multiplicity never enter this calculation.
    """
    if torch.device(device).type != "cuda":
        raise RuntimeError("pre-terminal exposure requires CUDA")
    nodes = topology.node_count
    degree = topology.executed_degree.to(device=device, dtype=torch.int64)
    torch._assert_async(degree.eq(1).all(), "exposure requires unbranched paths")
    edge = topology.child_begin.to(device=device, dtype=torch.int64)
    kind = topology.target_kind.to(device=device, dtype=torch.int64)[edge]
    target = topology.target_row.to(device=device, dtype=torch.int64)[edge]
    successor = torch.cat(
        (
            torch.where(kind.eq(1), nodes + target, target),
            torch.arange(nodes, nodes + leaf_count, device=device),
        )
    )
    torch._assert_async(
        ((successor >= 0) & (successor < nodes + leaf_count)).all(),
        "invalid pre-terminal descendant address",
    )
    for _ in range(max(1, nodes.bit_length())):
        successor = successor[successor]
    leaf = successor[:nodes] - nodes
    torch._assert_async(
        ((leaf >= 0) & (leaf < leaf_count)).all(),
        "pre-terminal paths did not reach leaves",
    )
    owner = topology.owner.to(device=device, dtype=torch.int64)
    family = bindings.input_type.to(device=device, dtype=torch.int64)
    selected = owner.ge(0)
    index = (leaf[selected] * players + owner[selected]) * families + family[selected]
    counts = torch.zeros(
        leaf_count * players * families, device=device, dtype=torch.int64
    )
    counts.scatter_add_(0, index, torch.ones_like(index))
    return counts.reshape(leaf_count, players, families)


def preterminal_posterior_exposure(
    range_weights, board_cards, combo_cards, counts, *, selected_player=None
):
    """Exact HU leaf marginals weighted by each player's trajectory decisions.

    Read the complete pre-terminal ranges after the last action and before
    completion sampling. Analytically marginalize opponent card collisions;
    this measures the distribution represented by the RB leaf rather than its
    finite Monte Carlo histogram. Entropy stays combo-wide even for ancestors
    whose learner representation is reduced to preflop classes.
    """
    if range_weights.device.type != "cuda":
        raise RuntimeError("pre-terminal posterior exposure requires CUDA")
    leaves, players, _hands = range_weights.shape
    if players != 2 or counts.shape[:2] != (
        leaves,
        1 if selected_player is not None else players,
    ):
        raise ValueError("pre-terminal counts and HU leaf ranges must align")
    totals = {
        name: _empty_accumulator(range_weights.device)
        for name in range(counts.shape[2])
    }
    # Bound transient combo matrices independently of rollout root count.
    for begin in range(0, leaves, 64):
        end = min(begin + 64, leaves)
        weights = range_weights[begin:end]
        torch._assert_async(
            (torch.isfinite(weights) & weights.ge(0)).all(),
            "invalid pre-terminal range weights",
        )
        log_reach = (
            weights.clamp_min(torch.finfo(weights.dtype).tiny)
            .log()
            .masked_fill(weights.eq(0), -torch.inf)
        )
        for slot in range(counts.shape[1]):
            actor = (
                torch.full(
                    (end - begin,), slot, device=weights.device, dtype=torch.int64
                )
                if selected_player is None
                else selected_player[begin:end].long()
            )
            for family in totals:
                decision_weights = counts[begin:end, slot, family]
                measured = current_posterior_exposure(
                    log_reach,
                    actor,
                    board_cards[begin:end],
                    combo_cards,
                    selected=decision_weights.gt(0),
                    decision_weights=decision_weights,
                )
                for key, value in measured.items():
                    totals[family][key].add_(value)
    return totals


def install_posterior_exposure_observer(
    session: Any, observations: MutableSequence[dict[str, Any]]
) -> None:
    """Observe pre-terminal leaf ranges, weighted by fresh ancestor decisions."""
    if session.device.type != "cuda":
        raise RuntimeError("posterior exposure observation requires CUDA")
    if session.mode.rollout_mode is not RolloutMode.SEQUENTIAL:
        raise RuntimeError("posterior exposure requires sequential rollout")
    factory = session._iteration_engine

    def instrumented_factory():
        engine = factory()
        records = {}
        original_completion = engine._execute_completion
        original_prepare = engine._prepare_learner

        def observed_completion(owners, state):
            torch.cuda.current_stream(session.device).wait_event(state.last_event)
            leaf_columns = owners.slab("Leaf").columns
            sampled = session.mode.private_estimator is PrivateEstimator.SAMPLED
            names = state.input_names
            players = (
                1
                if session.mode.training_regime.value == "best_response"
                else session.geometry.players
            )
            cards = torch.combinations(
                torch.arange(
                    52 - 4 * session.geometry.rank_count,
                    52,
                    device=session.device,
                    dtype=torch.int32,
                ),
                r=2,
            )
            for lane, topology in enumerate(state.topology.topologies):
                leaf_count = state.topology.leaf_counts[lane]
                if not topology.node_count:
                    continue
                counts = sequential_leaf_decision_counts(
                    topology,
                    state.topology.bindings[lane],
                    leaf_count,
                    len(names),
                    players,
                    session.device,
                )
                if sampled:
                    measured = {}
                    for family in range(len(names)):
                        count = counts[:, :, family].sum()
                        value = count.double()
                        measured[family] = {
                            "fresh_player_decisions": count,
                            "sum_posterior_exp_entropy": value,
                            "sum_sqrt_posterior_exp_entropy": value,
                            "sum_posterior_entropy_nats": value.new_zeros(()),
                        }
                else:
                    measured = preterminal_posterior_exposure(
                        leaf_columns["range_weights"][lane, :leaf_count],
                        leaf_columns["leaf_public_state"][lane, :leaf_count],
                        cards,
                        counts,
                        selected_player=(
                            leaf_columns["trained_actor"][lane, :leaf_count].reshape(-1)
                            if players == 1
                            else None
                        ),
                    )
                records[lane] = {
                    names[index]: values for index, values in measured.items()
                }
            return original_completion(owners, state)

        def observed_prepare(owners, state, lane):
            prepared = original_prepare(owners, state, lane)
            if lane not in records:
                raise RuntimeError(
                    "missing or duplicate pre-terminal posterior observation"
                )
            partitions = {}
            totals = {
                "fresh_player_decisions": 0,
                "sum_posterior_exp_entropy": 0.0,
                "sum_sqrt_posterior_exp_entropy": 0.0,
                "sum_posterior_entropy_nats": 0.0,
            }
            for name, values in records.pop(lane).items():
                item = {
                    key: (
                        int(value) if key == "fresh_player_decisions" else float(value)
                    )
                    for key, value in values.items()
                }
                partitions[name] = item
                for key in totals:
                    totals[key] += item[key]
            observations.append(
                {
                    "schema": EXPOSURE_SCHEMA,
                    "posterior_semantics": POSTERIOR_SEMANTICS,
                    "lane": lane,
                    "private_estimator": session.mode.private_estimator.value,
                    **totals,
                    "partitions": partitions,
                    "completion_effective_hands_used": False,
                    "completion_histogram_used": False,
                    "is_independent_sample_count": False,
                    "primary_clock_exponent": 1.0,
                    "weighting": "sum_leaf_player_decision_count_times_exp_entropy_of_preterminal_combo_marginal",
                    "correlation_discount_status": "unmeasured_no_discount_applied",
                }
            )
            return prepared

        engine._execute_completion = observed_completion
        engine._prepare_learner = observed_prepare
        return engine

    session._iteration_engine = instrumented_factory


__all__ = [
    "EXPOSURE_SCHEMA",
    "GRADIENT_CORRELATION_PLAN",
    "POSTERIOR_SEMANTICS",
    "ExactComboPosteriorUnavailable",
    "current_posterior_exposure",
    "gradient_covariance_diagnostic",
    "install_posterior_exposure_observer",
    "scheduling_exposure",
    "weighted_advantage_dispersion",
]
