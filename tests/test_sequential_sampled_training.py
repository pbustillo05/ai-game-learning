from __future__ import annotations

from tests.model_support import FixedOutputPokerModel

import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from torch import nn

from pbg.core.capacity import (
    CapacityManifest,
    ExecutionGeometry,
    ExecutionPhase,
    GeneratedOwners,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
    build_storage_plan,
)
from pbg.core.contracts import MODE_SCHEMAS
from pbg.core.execution_manifest import seal_execution_manifest
from pbg.core.iteration import IterationError
from pbg.core.learner import (
    AlignedSequentialStore,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    ModelEvaluation,
    ReferenceOutputCache,
    ReferenceState,
    SealedSequentialInputs,
    SequentialBatchSampler,
    SequentialLearnerBatch,
    allocate_statistical_counts,
)
from pbg.core.optimization import DenseAdam, LossScaler, OptimizerConfig
from pbg.core.persistence import (
    RunState,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)
from pbg.core.rollout import CompletionBatch
from pbg.core.storage import (
    DestinationRequest,
    DeviceFailureState,
    MonotoneSlab,
    PublicationTransaction,
    SoAField,
    SoASchema,
)
from pbg.core.training import (
    LearnerIteration,
    RecoverableOverflow,
)
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.model.capacity import max_policy_partition_sequence_tokens
from pbg.games.poker.queued.config import PokerScientificConfig
from pbg.games.poker.queued.hooks import (
    NLHEGeometry,
    PokerStateMachine,
    SampledShowdownBinding,
    rb_partitions,
    sampled_partitions,
)
from pbg.games.poker.queued.model import QueuedPokerModel, build_model_from_config
from pbg.games.poker.queued.policy import (
    CHECK_CALL,
    FOLD,
    PokerLegality,
    PokerPolicyOperator,
)
from tests.iteration_support import (
    EngineDrivenSequentialAttempt,
    SequentialRecoveryHarness,
)

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Phase 5 training requires CUDA"
)


def _reference_policy(
    raw: dict[str, torch.Tensor],
    event_ids: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    logits = raw["type_logits"].float()
    ticks = event_ids.shape[1] - 2
    location = raw["mu"].float()[..., None]
    scale = raw["s"].float()[..., None]
    edges = torch.linspace(
        0.0, 1.0, ticks + 1, dtype=torch.float32, device=logits.device
    )
    standardized = (edges - location) / scale
    lower, upper = standardized[..., :-1], standardized[..., 1:]
    tick_probability = (
        torch.sigmoid(upper) * torch.sigmoid(-lower) * (-torch.expm1(lower - upper))
    ).clamp_min(0.0)
    normalizer = (
        torch.sigmoid(standardized[..., -1:])
        * torch.sigmoid(-standardized[..., :1])
        * (-torch.expm1(standardized[..., :1] - standardized[..., -1:]))
    )
    tiny = torch.finfo(torch.float32).tiny
    tick_probability = tick_probability / normalizer.clamp_min(tiny)
    tick_probability = tick_probability / tick_probability.sum(
        -1, keepdim=True
    ).clamp_min(tiny)
    mixture = (raw["omega"].float()[..., None] * tick_probability).sum(-2)
    prefix = logits.shape[:-1]
    event_shape = (prefix[0],) + (1,) * (len(prefix) - 1) + (event_ids.shape[1],)
    mask = legal.reshape(event_shape).expand(prefix + (event_ids.shape[1],))
    raise_mask = mask[..., 2:]
    mixture = mixture.masked_fill(~raise_mask, 0.0)
    mixture_mass = mixture.sum(-1, keepdim=True)
    uniform = raise_mask.float() / raise_mask.sum(-1, keepdim=True).clamp_min(1)
    conditional = torch.where(
        mixture_mass > 0,
        mixture / mixture_mass.clamp_min(1.0e-12),
        uniform,
    )
    tick_index = torch.arange(ticks, device=logits.device)
    all_in_index = torch.where(
        raise_mask, tick_index, torch.full_like(tick_index, -1)
    ).amax(-1)
    all_in = (tick_index == all_in_index[..., None]).float() * (all_in_index >= 0)[
        ..., None
    ]
    raise_probability = (
        (1.0 - raw["iota"].float()[..., None]) * conditional
        + raw["iota"].float()[..., None] * all_in
    ).masked_fill(~raise_mask, 0.0)
    facing = mask[..., 0]
    full_logits = torch.cat(
        (
            logits[..., 0, None],
            torch.where(facing, logits[..., 1], logits[..., 0])[..., None],
            logits[..., 2, None]
            + torch.where(
                raise_probability > 0,
                torch.log(raise_probability.clamp_min(1.0e-12)),
                torch.full_like(raise_probability, -torch.inf),
            ),
        ),
        -1,
    )
    ids = event_ids.long().reshape(event_shape).expand_as(mask)
    selected = full_logits.gather(-1, ids)
    return torch.softmax(selected.masked_fill(~mask, -torch.inf), -1).masked_fill(
        ~mask, 0.0
    )


def test_nlhe_geometry_and_partition_boundary_matrix() -> None:
    for players in range(2, 7):
        for stack in (8.0, 8.5, 32.0, 127.5, 128.0):
            for ranks in range(4, 14):
                fits = 2 * players + 5 < 4 * ranks
                if fits:
                    geometry = NLHEGeometry(players, stack, ranks)
                    assert geometry.player_node_bound == math.ceil(
                        stack * players + 3 * players - stack + 1
                    )
                    partitions = sampled_partitions(geometry)
                    assert tuple(item.name for item in partitions.inputs) == (
                        ("pre",)
                        + tuple(f"post_{live}" for live in range(2, players + 1))
                    )
                    assert partitions.outputs[0].name == "all"
                else:
                    with pytest.raises(ValueError, match="2P"):
                        NLHEGeometry(players, stack, ranks)
    for invalid in (7.5, 8.25, 128.5):
        with pytest.raises(ValueError):
            NLHEGeometry(2, invalid, 13)


def test_full_deck_partition_widths_match_the_storage_contract() -> None:
    geometry = NLHEGeometry(2, 8.0, 13)
    assert geometry.private_classes == 169
    assert geometry.private_hands == 1326

    sampled = sampled_partitions(geometry)
    assert {item.name: item.width for item in sampled.inputs} == {
        "pre": 1,
        "post_2": 1,
    }
    assert {item.name: item.width for item in sampled.outputs} == {"all": 1}
    assert sampled.folded == ()

    rb = rb_partitions(geometry)
    assert {item.name: item.width for item in rb.inputs} == {
        "pre": 169,
        "post_2": 1326,
    }
    assert {item.name: item.width for item in rb.outputs} == {
        "pre": 169,
        "post": 1326,
    }
    assert rb.folded == ()

    multiway = NLHEGeometry(3, 8.0, 13)
    assert {item.name: item.width for item in sampled_partitions(multiway).folded} == {
        "pre": 169,
        "post": 1326,
    }
    assert {item.name: item.width for item in rb_partitions(multiway).folded} == {
        "pre": 169,
        "post": 1326,
    }


@CUDA
def test_multiway_short_all_in_repends_callers_without_reopening_raise() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    machine = PokerStateMachine(NLHEGeometry(3, 8.0, 4))
    state = machine.initial_roots(
        1,
        seed=71,
        iteration=0,
        retry_epoch=0,
        device=device,
    )
    # Button/root 0 acts first, seat 1 is the small blind, and seat 2 is the
    # big blind.  Seat 2's raise from 6 to 8 is all-in but smaller than the
    # preceding full-raise increment of 5.
    state = machine.step_player(
        state, torch.tensor([13], dtype=torch.int32, device=device)
    )
    state = machine.step_player(
        state, torch.tensor([CHECK_CALL], dtype=torch.int32, device=device)
    )
    state = machine.step_player(
        state, torch.tensor([17], dtype=torch.int32, device=device)
    )
    torch.cuda.synchronize(device)
    assert state.committed_street.tolist() == [[6.0, 6.0, 8.0]]
    assert state.pending.tolist() == [[True, True, False]]
    assert state.can_raise.tolist() == [[False, False, False]]
    assert state.to_act.tolist() == [0]
    assert not machine.needs_chance(state).item()
    legality = machine.legality(state)
    assert legality.mask[0, CHECK_CALL]
    assert not legality.mask[0, 2:].any()


@CUDA
def test_retry_logical_base_owns_private_profile_and_button_identity() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    machine = PokerStateMachine(NLHEGeometry(3, 8.0, 4))
    complete = machine.initial_roots(
        5,
        seed=91,
        iteration=3,
        retry_epoch=2,
        device=device,
    )
    replacement = machine.initial_roots(
        2,
        seed=91,
        iteration=3,
        retry_epoch=2,
        logical_base=3,
        device=device,
    )
    torch.testing.assert_close(replacement.combo_ids, complete.combo_ids[3:])
    torch.testing.assert_close(replacement.button, complete.button[3:])


@CUDA
def test_chance_separator_keeps_pretransition_public_arithmetic() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    machine = PokerStateMachine(NLHEGeometry(2, 8.0, 4))
    state = machine.initial_roots(
        1, seed=101, iteration=0, retry_epoch=0, device=device
    )
    history = machine.empty_history(1, device)
    action = torch.tensor([CHECK_CALL], dtype=torch.int32, device=device)
    history = machine.append_player_history(state, action, history)
    state = machine.step_player(state, action)
    history = machine.append_player_history(state, action, history)
    state = machine.step_player(state, action)
    active = machine.needs_chance(state)
    history = machine.append_chance_history(state, active, history)
    torch.cuda.synchronize(device)
    assert history.action[0, :3].tolist() == [1, 1, 3]
    assert history.street[0, :3].tolist() == [0, 0, 1]
    assert history.current_bet[0, 2].item() == 1.0
    assert history.pot_before[0, 2].item() == 2.0


def _raw(
    device: torch.device,
    *,
    batch: int = 2,
    hands: int = 5,
    mixture: int = 3,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(9182)
    result = {
        "type_logits": torch.randn(batch, hands, 3, generator=generator, device=device),
        "omega": torch.softmax(
            torch.randn(batch, hands, mixture, generator=generator, device=device),
            -1,
        ),
        "mu": torch.sigmoid(
            torch.randn(batch, hands, mixture, generator=generator, device=device)
        ),
        "s": torch.nn.functional.softplus(
            torch.randn(batch, hands, mixture, generator=generator, device=device)
        )
        + 0.02,
        "iota": torch.sigmoid(
            torch.randn(batch, hands, generator=generator, device=device)
        ),
    }
    if requires_grad:
        result = {
            name: value.detach().clone().requires_grad_()
            for name, value in result.items()
        }
    return result


@CUDA
def test_canonical_policy_matches_independent_materialization_and_gradients() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    ids = torch.arange(10, dtype=torch.int32, device=device)[None].expand(2, -1)
    legal = torch.tensor(
        [
            [True, True, True, False, True, True, False, True, False, True],
            [False, True, False, False, False, False, False, False, False, False],
        ],
        dtype=torch.bool,
        device=device,
    )
    legality = PokerLegality(ids.contiguous(), legal)
    queued_raw = _raw(device, requires_grad=True)
    reference_raw = {
        name: value.detach().clone().requires_grad_()
        for name, value in queued_raw.items()
    }
    queued = PokerPolicyOperator().probabilities(queued_raw, legality)
    reference = _reference_policy(reference_raw, ids, legal)
    torch.testing.assert_close(queued, reference, atol=1.0e-6, rtol=1.0e-5)
    weight = torch.linspace(-0.7, 1.1, queued.numel(), device=device).reshape_as(queued)
    (queued * weight).sum().backward()
    (reference * weight).sum().backward()
    for name in queued_raw:
        torch.testing.assert_close(
            queued_raw[name].grad,
            reference_raw[name].grad,
            atol=2.0e-6,
            rtol=2.0e-5,
        )


@CUDA
def test_policy_sampling_selected_head_and_learner_share_one_operation() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    ids = torch.arange(6, dtype=torch.int32, device=device)[None].expand(3, -1)
    mask = torch.tensor(
        [
            [True, True, True, False, True, True],
            [False, True, False, False, False, False],
            [True, True, False, True, False, True],
        ],
        dtype=torch.bool,
        device=device,
    )
    legality = PokerLegality(ids.contiguous(), mask)
    raw = _raw(device, batch=3, hands=7)
    hand = torch.tensor([2, 5, 1], dtype=torch.int32, device=device)
    keys = torch.tensor([71, 81, 91], dtype=torch.int64, device=device)

    class RecordingOperator(PokerPolicyOperator):
        def probabilities(self, selected_raw, selected_legality):
            self.rollout_shape = selected_raw["type_logits"].shape
            return super().probabilities(selected_raw, selected_legality)

    operator = RecordingOperator()
    sampled = operator.sample_actor(raw, legality, hand, keys)
    assert operator.rollout_shape == (3, 3)
    selected_all = operator.selected_log_pi(raw, legality, sampled.action)
    torch.testing.assert_close(
        sampled.selected_log_policy,
        selected_all[torch.arange(3, device=device), hand.to(torch.int64)],
    )
    assert sampled.action[1] == CHECK_CALL
    entropy, kl = operator.policy_reductions(raw, legality, raw)
    assert entropy.shape == kl.shape == (3, 7)
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=2.0e-7, rtol=0)


@CUDA
def test_root_profiles_are_collision_free_and_public_transition_is_conservative() -> (
    None
):
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(6, 8.0, 13)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        8, seed=104830, iteration=9, retry_epoch=0, device=device
    )
    cards = machine.combo_cards(state.combo_ids)
    for row in cards:
        assert torch.unique(row).numel() == 2 * geometry.players
    initial_input = machine.direct_model_input("pre", state)
    assert initial_input.device == device and initial_input.shape[0] == 8
    reconstructed = machine.state_from_model_input(
        "pre", initial_input, state.combo_ids
    )
    for name in state.__dataclass_fields__:
        if name != "terminal":
            torch.testing.assert_close(
                getattr(reconstructed, name), getattr(state, name)
            )
    initial_total = state.stacks + state.committed
    action = torch.full((8,), FOLD, dtype=torch.int32, device=device)
    next_state = machine.step_player(state, action)
    torch.testing.assert_close(next_state.stacks + next_state.committed, initial_total)
    assert next_state.terminal.logical_not().any()
    leaf_state = state
    # Folding in table order terminates every hand without requiring a board.
    while not bool(leaf_state.terminal.all()):
        legal = machine.legality(leaf_state)
        fold_or_call = torch.where(
            legal.mask[:, FOLD],
            torch.full_like(leaf_state.to_act, FOLD),
            torch.full_like(leaf_state.to_act, CHECK_CALL),
        )
        fold_or_call = torch.where(
            leaf_state.terminal,
            torch.full_like(fold_or_call, CHECK_CALL),
            fold_or_call,
        )
        active = ~leaf_state.terminal
        if not bool(active.all()):
            # Step only homogeneous active rows for this boundary test.
            break
        leaf_state = machine.step_player(leaf_state, fold_or_call)
    if bool(leaf_state.terminal.all()):
        leaf = machine.sampled_leaf(leaf_state)
        torch.testing.assert_close(
            leaf["payoff_baseline"].sum(-1) + leaf["stratum_amount"].sum(-1),
            torch.zeros(8, device=device),
            atol=1.0e-5,
            rtol=0,
        )
        import poker_gpu_evaluator as poker

        evaluator_payoff = poker.bind(
            rank_count=geometry.rank_count
        ).evaluate_showdowns(
            board_cards=leaf["leaf_public_state"],
            combo_ids=leaf["leaf_private_profile"],
            stratum_level=leaf["stratum_level"],
            payoff_baseline=leaf["payoff_baseline"],
            stratum_amount=leaf["stratum_amount"],
        )
        torch.testing.assert_close(
            evaluator_payoff, leaf["payoff_baseline"], atol=0, rtol=0
        )


@CUDA
def test_sampled_leaf_matches_exact_evaluator_rows_and_binding() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 13)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        2, seed=209559, iteration=1, retry_epoch=0, device=device
    )
    generation = 0
    while not bool(state.terminal.all()):
        if bool(machine.needs_chance(state).all()):
            draw = torch.arange(
                geometry.deck_cards,
                dtype=torch.int64,
                device=device,
            )[None].expand(state.rows, -1)
            keys = draw + generation * geometry.deck_cards + 1
            state = machine.deal_next_street(state, keys)
            generation += 1
        else:
            action = torch.full(
                (state.rows,), CHECK_CALL, dtype=torch.int32, device=device
            )
            state = machine.step_player(state, action)
    assert generation == 3
    assert state.board_cards.ne(255).all()
    leaf = machine.sampled_leaf(state)
    failures = DeviceFailureState(1, device)
    work = MonotoneSlab(
        SoASchema(
            "LayerWork.all",
            (SoAField("return_value", torch.float32, (2, 2, 1)),),
            "completion_to_target_publication",
        ),
        2,
        1,
        device,
        failure_state=failures,
    )
    binding = SampledShowdownBinding(geometry, work, n_replica=1, pack_cap=2)
    batch = CompletionBatch(
        lanes=torch.tensor([0], dtype=torch.int64, device=device),
        row=torch.tensor([[0, 1]], dtype=torch.int64, device=device),
        fields=MappingProxyType({name: value[None] for name, value in leaf.items()}),
    )
    plan = binding.leaf_to_layer_work(batch)
    expected = binding.evaluator.evaluate_showdowns(
        board_cards=leaf["leaf_public_state"],
        combo_ids=leaf["leaf_private_profile"],
        stratum_level=leaf["stratum_level"],
        payoff_baseline=leaf["payoff_baseline"],
        stratum_amount=leaf["stratum_amount"],
    )
    actual = plan.destinations[0].values["return_value"][0, :2, 0, :, 0]
    torch.testing.assert_close(actual, expected)
    normalized = SampledShowdownBinding(
        geometry,
        work,
        n_replica=1,
        pack_cap=2,
        utility="stack_normalized",
    ).leaf_to_layer_work(batch)
    normalized_value = normalized.destinations[0].values["return_value"][0, :2, 0, :, 0]
    torch.testing.assert_close(normalized_value, expected / geometry.stack_bb)
    realized = plan.coverage.validate(
        batch,
        plan.destinations,
        torch.tensor([True], device=device),
        descriptor_cap=8,
        workspace_byte_cap=4096,
    )
    assert realized[0].tolist() == [2]
    work.release()
    failures.release()


class _TinyModel(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.4, -0.8, 1.2], device=device))

    def forward(self) -> torch.Tensor:
        return self.weight


class _PartitionNetwork(nn.Module):
    def __init__(
        self, input_width: int, output_width: int, device: torch.device
    ) -> None:
        super().__init__()
        self.output_width = output_width
        self.affine = nn.Linear(input_width, output_width * 9, device=device)

    def forward(self, model_input: torch.Tensor) -> ModelEvaluation:
        fields = self.affine(model_input).reshape(
            model_input.shape[0], self.output_width, 9
        )
        return ModelEvaluation(
            MappingProxyType(
                {
                    "type_logits": fields[..., :3],
                    "omega": torch.softmax(fields[..., 3:4], -1),
                    "mu": torch.sigmoid(fields[..., 4:5]),
                    "s": torch.nn.functional.softplus(fields[..., 5:6]) + 0.02,
                    "iota": torch.sigmoid(fields[..., 6]),
                }
            ),
            fields[..., 7],
        )


@CUDA
def test_two_partition_model_routes_natural_domains_and_preserves_class_order() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    preflop = _PartitionNetwork(
        geometry.preflop_model_input_width, geometry.rank_count**2, device
    )
    postflop = _PartitionNetwork(
        geometry.postflop_model_input_width, geometry.private_hands, device
    )
    model = FixedOutputPokerModel(geometry, preflop, postflop).to(device)
    assert model.combo_class.cpu().tolist() == list(
        DeckGeometry(geometry.rank_count).class_of_combo
    )
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(3, seed=17, iteration=0, retry_epoch=0, device=device)
    model_input = machine.direct_model_input("pre", state)
    pre_reach, _ = machine.model_context("pre", model_input)
    assert pre_reach.shape == (
        state.rows,
        geometry.players,
        geometry.private_classes,
    )
    pre = model.forward_partition("pre", model_input)
    pre_rollout = model.rollout_operation(
        "pre", model_input, state.combo_ids, state.to_act
    )
    pre_learner = model.learner_operation(
        "pre", model_input, state.combo_ids, state.to_act
    )
    assert pre_rollout.value.shape == (state.rows, geometry.private_classes)
    assert pre_learner.value.shape == (state.rows, 1)
    assert all(
        value.shape[1] == geometry.private_classes
        for value in pre_rollout.policy_raw.values()
    )
    assert all(value.shape[1] == 1 for value in pre_learner.policy_raw.values())
    state = machine.step_player(
        state,
        torch.full((3,), CHECK_CALL, dtype=torch.int32, device=device),
    )
    state = machine.step_player(
        state,
        torch.full((3,), CHECK_CALL, dtype=torch.int32, device=device),
    )
    state = machine.deal_next_street(
        state,
        torch.arange(geometry.deck_cards, dtype=torch.int64, device=device)[
            None
        ].expand(3, -1),
    )
    post_input = machine.direct_model_input("post_2", state)
    post_reach, _ = machine.model_context("post_2", post_input)
    assert post_reach.shape == (
        state.rows,
        geometry.players,
        geometry.private_hands,
    )
    post = model.forward_partition("post_2", post_input)
    post_learner = model.learner_operation(
        "post_2", post_input, state.combo_ids, state.to_act
    )
    assert pre.value.shape == (3, geometry.private_classes)
    assert post.value.shape == (3, geometry.private_hands)
    assert post_learner.value.shape == (state.rows, 1)
    assert all(value.shape[1] == 1 for value in post_learner.policy_raw.values())
    loss = (
        pre.value.mean()
        + post.value.mean()
        + pre.policy_raw["type_logits"].mean()
        + post.policy_raw["type_logits"].mean()
    )
    loss.backward()
    assert preflop.affine.weight.grad is not None
    assert postflop.affine.weight.grad is not None


@CUDA
def test_phase5_postflop_fails_before_silent_folded_summary_omission() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(3, 8.0, 4)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(1, seed=29, iteration=0, retry_epoch=0, device=device)
    status = state.status.clone()
    status[:, 2] = 1
    state = replace(
        state,
        street=torch.ones_like(state.street),
        status=status,
    )
    model = FixedOutputPokerModel(
        geometry,
        _PartitionNetwork(
            geometry.preflop_model_input_width,
            geometry.rank_count**2,
            device,
        ),
        _PartitionNetwork(
            geometry.postflop_model_input_width,
            geometry.private_hands,
            device,
        ),
    )
    with pytest.raises(RuntimeError, match="folded summaries"):
        model.forward_partition("post_2", machine.direct_model_input("post_2", state))


def _phase5_manifest(device: torch.device) -> CapacityManifest:
    return CapacityManifest(
        geometry=ExecutionGeometry(2, 15, 64, 4096),
        root_count=1,
        n_replica=1,
        beta_max=1,
        player_node_cap=(
            PartitionRows("pre", 40),
            PartitionRows("post_2", 40),
        ),
        trained_player_cap=(
            PartitionRows("pre", 20),
            PartitionRows("post_2", 20),
        ),
        action_cap=(
            PartitionRows("pre", 20),
            PartitionRows("post_2", 20),
        ),
        folded_range_cap=(),
        leaf_cap=8,
        topology_cap=8,
        topology_edge_cap=8,
        family_pack_cap=8,
        chance_frontier_cap=8,
        completion_pack_cap=8,
        learner_sigma_pack_cap=8,
        learner_action_pack_cap=8,
        topology_packet_cap=8,
        target_arena_cap=16,
        rollout_workspace_bytes=1 << 20,
        queue_drain_workspace_bytes=1024,
        completion_workspace_bytes=2048,
        align_workspace_bytes=(1 << 20) + 1024,
        target_workspace_bytes=(1 << 20) + 1024,
        learner_workspace_bytes=1 << 20,
        hardware=HardwareLimits(
            torch.cuda.get_device_properties(device).total_memory,
            64 << 20,
        ),
    )


def _phase5_layout(
    geometry: NLHEGeometry | None = None,
) -> PhysicalLayout:
    geometry = NLHEGeometry(2, 8.0, 4) if geometry is None else geometry
    return PhysicalLayout(
        model_input_shapes=(
            NamedShape("pre", (geometry.preflop_model_input_width,)),
            NamedShape("post_2", (geometry.postflop_model_input_width,)),
        ),
        player_count=2,
        stratum_count=2,
        opaque_fields=(
            SoAField("leaf_public_state", torch.uint8, (5,), fill=255),
            SoAField(
                "leaf_private_profile",
                torch.int32,
                (geometry.players,),
                fill=-1,
            ),
        ),
    )


def _publish_rows(slab, rows: int, fields: dict[str, torch.Tensor]) -> None:
    transaction = PublicationTransaction.begin(
        (
            # Host-known exact sealed extents are legal at this boundary.
            DestinationRequest(slab, 0, rows, slab.label, 0),
        ),
        phase="phase5_fixture",
    )
    transaction.fill(0, fields)
    transaction.publish()


class _LearnerModel(nn.Module):
    def __init__(self, device: torch.device, hands: int = 4) -> None:
        super().__init__()
        self.hands = hands
        self.affine = nn.Linear(26, hands * 11, device=device)

    def evaluate(
        self,
        _input_type: str,
        model_input: torch.Tensor,
        _private: torch.Tensor,
        _owner: torch.Tensor,
    ) -> ModelEvaluation:
        batch = model_input.shape[0]
        fields = self.affine(model_input[:, :26].float()).reshape(batch, self.hands, 11)
        return ModelEvaluation(
            MappingProxyType(
                {
                    "type_logits": fields[..., :3],
                    "omega": torch.softmax(fields[..., 3:5], -1),
                    "mu": torch.sigmoid(fields[..., 5:7]),
                    "s": torch.nn.functional.softplus(fields[..., 7:9]) + 0.02,
                    "iota": torch.sigmoid(fields[..., 9]),
                }
            ),
            fields[..., 10],
        )


class _LearnerPolicy:
    def __init__(self) -> None:
        self.operator = PokerPolicyOperator()

    def __call__(
        self,
        _input_type,
        _model_input,
        raw,
        action,
        private,
        owner,
        reference_raw,
    ):
        batch = action.shape[0]
        device = action.device
        ids = torch.arange(4, dtype=torch.int32, device=device)[None].expand(batch, -1)
        legality = PokerLegality(
            ids.contiguous(), torch.ones_like(ids, dtype=torch.bool)
        )
        return self.operator.learner(
            raw, legality, action, private, owner, reference_raw
        )


class _EndToEndPokerModel(nn.Module):
    def __init__(self, device: torch.device, hands: int, *, check_down: bool) -> None:
        super().__init__()
        self.hands = hands
        self.check_down = check_down
        self.seen_input_types: list[str] = []
        self.public = nn.Linear(26, 1, device=device)
        self.type_logits = nn.Parameter(
            torch.tensor([24.0, -24.0, -24.0], device=device).expand(hands, -1).clone()
        )
        self.omega = nn.Parameter(torch.zeros(hands, 2, device=device))
        self.mu = nn.Parameter(torch.zeros(hands, 2, device=device))
        self.scale = nn.Parameter(torch.zeros(hands, 2, device=device))
        self.iota = nn.Parameter(torch.full((hands,), -4.0, device=device))
        self.value = nn.Parameter(torch.zeros(hands, device=device))

    def evaluate(
        self,
        input_type: str,
        model_input: torch.Tensor,
        _private: torch.Tensor,
        _owner: torch.Tensor,
    ) -> ModelEvaluation:
        self.seen_input_types.append(input_type)
        batch = model_input.shape[0]
        public = self.public(model_input[:, :26].float())
        type_logits = self.type_logits[None].expand(batch, -1, -1)
        if self.check_down:
            actor = model_input[:, 6:8].argmax(-1)
            committed_street = (
                model_input[:, 14:16].gather(1, actor[:, None]).squeeze(1)
            )
            facing = model_input[:, 24].float() > committed_street + 1.0e-6
            forced = torch.stack(
                (
                    torch.where(
                        facing,
                        torch.full_like(facing, -30.0, dtype=torch.float32),
                        torch.full_like(facing, 30.0, dtype=torch.float32),
                    ),
                    torch.where(
                        facing,
                        torch.full_like(facing, 30.0, dtype=torch.float32),
                        torch.full_like(facing, -30.0, dtype=torch.float32),
                    ),
                    torch.full_like(facing, -30.0, dtype=torch.float32),
                ),
                -1,
            )
            type_logits = (
                forced[:, None].expand(-1, self.hands, -1) + 0.001 * type_logits
            )
        return ModelEvaluation(
            MappingProxyType(
                {
                    "type_logits": type_logits + public[:, None] * 0.01,
                    "omega": torch.softmax(self.omega, -1)[None].expand(batch, -1, -1),
                    "mu": torch.sigmoid(self.mu)[None].expand(batch, -1, -1),
                    "s": (torch.nn.functional.softplus(self.scale) + 0.02)[None].expand(
                        batch, -1, -1
                    ),
                    "iota": torch.sigmoid(self.iota)[None].expand(batch, -1),
                }
            ),
            self.value[None].expand(batch, -1) + public,
        )


@CUDA
def test_complete_poker_iteration_runs_roots_to_optimizer() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = next(
        schema
        for schema in MODE_SCHEMAS
        if schema.key == "self_play/sequential/sampled"
    )
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            sampled_partitions(geometry),
            _phase5_manifest(device),
            _phase5_layout(),
            device,
        )
    )
    model = _EndToEndPokerModel(device, geometry.private_hands, check_down=True)
    optimizer = DenseAdam(model, OptimizerConfig(name="adamw", grad_clip=5.0))
    run_state = RunState("phase5-e2e", root_count=1)
    result = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=owners,
        mode=mode,
        model=model,
        model_operation=model.evaluate,
        optimizer=optimizer,
        loss=LossConfig(),
        schedule=LearnerSchedule(batch_size=1),
        generator=torch.Generator(device=device).manual_seed(991),
        seed=991,
        lambda_=0.95,
        run_state=run_state,
    ).run(iteration=0)
    assert result.player_nodes == 8
    assert result.leaves == 1
    assert result.target_layers == 8
    assert result.learner.statistical_presentations == 8
    assert len(result.learner.optimizer_steps) == 8
    assert all(step.applied for step in result.learner.optimizer_steps)
    assert optimizer.step_count == 8
    assert run_state.iteration == 1
    assert run_state.logical_work["trajectories"] == 1
    assert set(model.seen_input_types) == {"pre", "post_2"}


@CUDA
def test_complete_poker_iteration_runs_reconstructed_architecture() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 13)
    mode = next(
        schema
        for schema in MODE_SCHEMAS
        if schema.key == "self_play/sequential/sampled"
    )
    model = build_model_from_config(
        {
            "mode": {
                "training_regime": "self_play",
                "rollout_mode": "sequential",
                "private_estimator": "sampled",
            },
            "game": {
                "players": 2,
                "stack_bb": 8.0,
                "rank_count": 13,
            },
            "model": {
                "d_model": 4,
                "n_heads": 1,
                "layers": 1,
                "raises": {"components": 2, "minimum_scale": 0.02},
                "seat_capacity": 2,
                "class_embedding_dim": 2,
                "ffn_hidden": 8,
                "folder_beta_rank": 0,
                "folder_line_dim": 2,
                "epsilon": 1.0e-8,
            },
        },
        device,
    )
    model.configure_execution("sdpa")
    assert all(
        attention.attention_backend == "sdpa" and attention.query_chunk_size is None
        for attention in (
            *model.shared.attentions,
        )
    )
    with torch.no_grad():
        for network in (model.shared,):
            network.type_head.affine.weight.zero_()
            network.type_head.affine.bias.copy_(
                torch.tensor([-30.0, 30.0, -60.0], device=device)
            )
    token_bound = max(
        max_policy_partition_sequence_tokens(
            n_players=2,
            stack_bb=8.0,
            rank_count=13,
            k_act=5,
        ).values()
    )
    manifest = replace(
        _phase5_manifest(device),
        geometry=ExecutionGeometry(2, 15, token_bound, token_bound * token_bound),
        topology_cap=16,
        topology_edge_cap=16,
        topology_packet_cap=16,
        target_arena_cap=32,
    )
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            sampled_partitions(geometry),
            manifest,
            _phase5_layout(geometry),
            device,
        )
    )
    result = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=owners,
        mode=mode,
        model=model,
        model_operation=model.learner_operation,
        optimizer=DenseAdam(model, OptimizerConfig()),
        loss=LossConfig(),
        schedule=LearnerSchedule(batch_size=1),
        generator=torch.Generator(device=device).manual_seed(43),
        seed=43,
        lambda_=0.95,
        run_state=RunState("phase5-real-model", root_count=1),
    ).run(iteration=0)
    assert result.player_nodes == 8
    assert result.leaves == 1
    assert result.target_layers == 8
    assert result.learner.statistical_presentations == 8


@CUDA
def test_end_to_end_overflow_exposes_no_failed_attempt_rows() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = next(
        schema
        for schema in MODE_SCHEMAS
        if schema.key == "self_play/sequential/sampled"
    )
    manifest = replace(
        _phase5_manifest(device),
        player_node_cap=(
            PartitionRows("pre", 1),
            PartitionRows("post_2", 79),
        ),
    )
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            sampled_partitions(geometry),
            manifest,
            _phase5_layout(),
            device,
        )
    )
    model = _EndToEndPokerModel(device, geometry.private_hands, check_down=True)
    run_state = RunState("phase5-overflow", root_count=1)
    with pytest.raises(RecoverableOverflow) as raised:
        EngineDrivenSequentialAttempt(
            geometry=geometry,
            owners=owners,
            mode=mode,
            model=model,
            model_operation=model.evaluate,
            optimizer=DenseAdam(model, OptimizerConfig()),
            loss=LossConfig(),
            schedule=LearnerSchedule(batch_size=1),
            generator=torch.Generator(device=device).manual_seed(19),
            seed=19,
            lambda_=0.95,
            run_state=run_state,
        ).run(iteration=0)
    assert raised.value.failure.failure_code == "capacity_overflow"
    assert not owners._live
    assert len(owners._released) == len(owners.plan.allocations)


@CUDA
def test_end_to_end_overflow_rebuilds_and_retries_at_lower_roots() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = next(
        schema
        for schema in MODE_SCHEMAS
        if schema.key == "self_play/sequential/sampled"
    )
    manifest = replace(
        _phase5_manifest(device),
        root_count=2,
        player_node_cap=(
            PartitionRows("pre", 2),
            PartitionRows("post_2", 78),
        ),
    )
    plan = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        manifest,
        _phase5_layout(),
        device,
    )
    model = _EndToEndPokerModel(device, geometry.private_hands, check_down=True)
    run_state = RunState("phase5-retry", root_count=2)
    with pytest.warns(RuntimeWarning, match="RECOVERED_SEQUENTIAL_OVERFLOW"):
        result = SequentialRecoveryHarness(
            owner_factory=lambda: GeneratedOwners(plan),
            plan=plan,
            minimum_root_count=1,
            geometry=geometry,
            mode=mode,
            model=model,
            model_operation=model.evaluate,
            optimizer=DenseAdam(model, OptimizerConfig()),
            loss=LossConfig(),
            schedule=LearnerSchedule(batch_size=1),
            generator=torch.Generator(device=device).manual_seed(29),
            seed=29,
            lambda_=0.95,
            run_state=run_state,
        ).run(iteration=0)
    assert result.player_nodes == 8
    assert result.leaves == 1
    assert run_state.root_count == 1
    assert run_state.retry_epoch == 1
    assert run_state.attempt_logical_base == 2
    assert run_state.recovered_overflow
    assert len(run_state.overflow_recoveries) == 1
    record = run_state.overflow_recoveries[0]
    assert record["capacity"] == 2
    assert record["observed_demand"] == 4
    assert record["old_root_count"] == 2
    assert record["new_root_count"] == 1
    assert record["replacement_address"] == "1:2"


@CUDA
def test_end_to_end_overflow_rejects_retry_capacity_drift() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = next(
        schema
        for schema in MODE_SCHEMAS
        if schema.key == "self_play/sequential/sampled"
    )
    manifest = replace(
        _phase5_manifest(device),
        root_count=2,
        player_node_cap=(
            PartitionRows("pre", 2),
            PartitionRows("post_2", 78),
        ),
    )
    plan = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        manifest,
        _phase5_layout(),
        device,
    )
    drifted = build_storage_plan(
        mode,
        sampled_partitions(geometry),
        replace(
            manifest,
            player_node_cap=(
                PartitionRows("pre", 40),
                PartitionRows("post_2", 40),
            ),
        ),
        _phase5_layout(),
        device,
    )
    attempts = iter((plan, drifted))
    model = _EndToEndPokerModel(device, geometry.private_hands, check_down=True)
    run_state = RunState("phase5-cap-drift", root_count=2)
    with (
        pytest.warns(RuntimeWarning, match="RECOVERED_SEQUENTIAL_OVERFLOW"),
        pytest.raises(IterationError, match="immutable storage plan"),
    ):
        SequentialRecoveryHarness(
            owner_factory=lambda: GeneratedOwners(next(attempts)),
            plan=plan,
            minimum_root_count=1,
            geometry=geometry,
            mode=mode,
            model=model,
            model_operation=model.evaluate,
            optimizer=DenseAdam(model, OptimizerConfig()),
            loss=LossConfig(),
            schedule=LearnerSchedule(batch_size=1),
            generator=torch.Generator(device=device).manual_seed(31),
            seed=31,
            lambda_=0.95,
            run_state=run_state,
        ).run(iteration=0)


@CUDA
def test_aligned_five_array_batches_and_complete_optimizer_iteration() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = next(
        schema
        for schema in MODE_SCHEMAS
        if schema.key == "self_play/sequential/sampled"
    )
    partitions = sampled_partitions(geometry)
    plan = build_storage_plan(
        mode,
        partitions,
        _phase5_manifest(device),
        _phase5_layout(geometry),
        device,
    )
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    generator = torch.Generator(device=device).manual_seed(773)
    row_counts = {"pre": 5, "post_2": 3}
    for partition, rows in row_counts.items():
        family = owners.slab("FamilyWork", partition)
        retained = owners.slab("FamilyRetained", partition)
        family_values = {
            field.name: torch.full(
                (rows, *field.shape),
                field.fill,
                dtype=field.dtype,
                device=device,
            )
            for field in family.schema.fields
        }
        retained_values = {
            field.name: torch.full(
                (rows, *field.shape),
                field.fill,
                dtype=field.dtype,
                device=device,
            )
            for field in retained.schema.fields
        }
        retained_values["model_input"].normal_(generator=generator)
        retained_values["sampled_private_state"][:] = torch.tensor(
            [0, 1], dtype=torch.int32, device=device
        )
        retained_values["owner"][:] = torch.arange(rows, device=device) % 2
        _publish_rows(family, rows, family_values)
        _publish_rows(retained, rows, retained_values)
        sigma = owners.slab("SigmaHead", partition)
        _publish_rows(
            sigma,
            rows,
            {"old_value": torch.zeros((rows, 1), device=device)},
        )
        action = owners.slab("ActionHead", partition)
        _publish_rows(
            action,
            rows,
            {
                "action": torch.full(
                    (rows,), CHECK_CALL, dtype=torch.int32, device=device
                ),
                "old_log_policy": torch.full((rows, 1), -math.log(4), device=device),
            },
        )
    owners.bind_empty_layer_work_for_abort()
    owners.release_after(ExecutionPhase.ROLLOUT)
    owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
    owners.release_after(ExecutionPhase.QUEUE_DRAIN)
    owners.enter_phase(ExecutionPhase.COMPLETION)
    owners.release_after(ExecutionPhase.COMPLETION)
    owners.enter_phase(ExecutionPhase.ALIGN_AND_SEAL)
    sealed_inputs = {
        partition: SealedSequentialInputs.seal(
            owners.slab("FamilyWork", partition),
            owners.slab("FamilyRetained", partition),
            input_type=partition,
        )
        for partition in row_counts
    }
    for partition, sealed in sealed_inputs.items():
        retained = owners.slab("FamilyRetained", partition)
        assert sealed.slab is retained
        assert (
            sealed.columns["model_input"].data_ptr()
            == retained.columns["model_input"][0].data_ptr()
        )
        with pytest.raises(Exception, match="no generated|not live"):
            owners.slab("LearnerIndex", partition)
    owners.release_after(ExecutionPhase.ALIGN_AND_SEAL)
    owners.enter_phase(ExecutionPhase.TARGETS)
    for partition, rows in row_counts.items():
        _publish_rows(
            owners.slab("SigmaTarget", partition),
            rows,
            {"value_target": torch.linspace(-1, 1, rows, device=device)[:, None]},
        )
        _publish_rows(
            owners.slab("ActionTarget", partition),
            rows,
            {
                "advantage_target": torch.linspace(0.3, 1.1, rows, device=device)[
                    :, None
                ]
            },
        )
    owners.release_after(ExecutionPhase.TARGETS)
    owners.enter_phase(ExecutionPhase.LEARNER)
    stores = {
        partition: AlignedSequentialStore(
            owners, partition, sealed_inputs=sealed_inputs[partition]
        )
        for partition in row_counts
    }
    cost_sampler = SequentialBatchSampler.from_costs(
        stores,
        LearnerSchedule(batch_size=3),
        generator=torch.Generator(device=device).manual_seed(884),
        mass=torch.tensor([1.0, 2.0]),
        variance=torch.tensor([4.0, 1.0]),
        seconds_per_draw=torch.tensor([1.0, 4.0]),
        total_count=7,
    )
    cost_batches = tuple(cost_sampler.batches())
    assert cost_sampler.statistical_presentations == 7
    assert cost_sampler.optimizer_steps == len(cost_batches)
    assert sum(batch.rows for batch in cost_batches) == 7
    assert all(0 < batch.rows <= 3 for batch in cost_batches)

    def cost_stratified_sampler() -> SequentialBatchSampler:
        return SequentialBatchSampler.from_costs(
            stores,
            LearnerSchedule(batch_size=4, epochs=None, gradient_steps=2),
            generator=torch.Generator(device=device).manual_seed(885),
            mass=torch.tensor([3.0, 1.0]),
            variance=torch.ones(2),
            seconds_per_draw=torch.ones(2),
            total_count=4,
        )

    weighted_batches = tuple(cost_stratified_sampler().batches())
    for step in range(2):
        step_batches = [
            batch for batch in weighted_batches if batch.logical_step == step
        ]
        assert [batch.input_type for batch in step_batches] == ["pre", "post_2"]
        assert [batch.objective_weight for batch in step_batches] == [0.75, 0.25]
        assert sum(float(batch.objective_weight) for batch in step_batches) == 1.0

    pooled_generator = torch.Generator(device=device).manual_seed(886)
    pooled_sampler = SequentialBatchSampler(
        stores,
        LearnerSchedule(batch_size=4, epochs=None, gradient_steps=3),
        generator=pooled_generator,
    )
    pooled_batches = tuple(pooled_sampler.batches())
    oracle_generator = torch.Generator(device=device).manual_seed(886)
    for step in range(3):
        oracle = torch.randint(
            8,
            (4,),
            dtype=torch.int64,
            device=device,
            generator=oracle_generator,
        )
        step_batches = {
            batch.input_type: batch
            for batch in pooled_batches
            if batch.logical_step == step
        }
        assert sum(batch.rows for batch in step_batches.values()) == 4
        if bool((oracle < 5).any()):
            torch.testing.assert_close(step_batches["pre"].indices, oracle[oracle < 5])
        if bool((oracle >= 5).any()):
            torch.testing.assert_close(
                step_batches["post_2"].indices, oracle[oracle >= 5] - 5
            )

    mixed_sampler = cost_stratified_sampler()
    mixed_model = _LearnerModel(device)
    mixed_result = LearnerIteration(
        learner=LearnerOperation(
            mode,
            LossConfig(),
            mixed_model.evaluate,
            _LearnerPolicy(),
        ),
        optimizer=DenseAdam(mixed_model, OptimizerConfig(grad_clip=5.0)),
        physical_batch_cap={"pre": 2, "post_2": 1},
    ).optimize(mixed_sampler, iteration=0)
    assert mixed_result.statistical_presentations == 8
    assert mixed_sampler.optimizer_steps == 2
    assert len(mixed_result.optimizer_steps) == 2
    assert all(step.applied for step in mixed_result.optimizer_steps)
    sampler = SequentialBatchSampler(
        stores,
        LearnerSchedule(batch_size=3, epochs=2),
        generator=generator,
    )
    model = _LearnerModel(device)
    optimizer = DenseAdam(model, OptimizerConfig(grad_clip=5.0))
    learner = LearnerOperation(
        mode,
        LossConfig(),
        model.evaluate,
        _LearnerPolicy(),
    )
    result = LearnerIteration(
        learner=learner,
        optimizer=optimizer,
        physical_batch_cap={"pre": 2, "post_2": 1},
    ).optimize(sampler, iteration=0)
    assert result.statistical_presentations == 16
    assert sampler.optimizer_steps == 6
    assert [step.applied for step in result.optimizer_steps] == [True] * 6
    assert optimizer.step_count == 6
    for sealed in sealed_inputs.values():
        sealed.release()
    owners.release_after(ExecutionPhase.LEARNER)
    owners.enter_phase(ExecutionPhase.FINAL_BACKWARD)
    owners.release_after(ExecutionPhase.FINAL_BACKWARD)
    owners.close()


@CUDA
@pytest.mark.parametrize("name", ("adam", "adamw"))
def test_dense_optimizer_agrees_with_torch_oracle(name: str) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    ours = _TinyModel(device)
    oracle = _TinyModel(device)
    config = OptimizerConfig(
        name=name,
        learning_rate=0.003,
        beta1=0.8,
        beta2=0.95,
        epsilon=1.0e-7,
        weight_decay=0.02,
        grad_clip=100.0,
    )
    dense = DenseAdam(ours, config)
    torch_optimizer = (
        torch.optim.AdamW(
            oracle.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
        if name == "adamw"
        else torch.optim.Adam(
            oracle.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
    )
    for step in range(1, 8):
        dense.zero_grad()
        torch_optimizer.zero_grad()
        target = torch.tensor(
            [0.1 * step, -0.2, 0.7], dtype=torch.float32, device=device
        )
        ours_loss = (ours() - target).square().sum()
        oracle_loss = (oracle() - target).square().sum()
        ours_loss.backward()
        oracle_loss.backward()
        result = dense.step()
        torch_optimizer.step()
        assert result.applied and result.step == step
        torch.testing.assert_close(ours.weight, oracle.weight, atol=2e-7, rtol=2e-6)


@CUDA
def test_nonfinite_fp16_update_backs_off_once_without_mutating_parameters() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    model = _TinyModel(device)
    initial = model.weight.detach().clone()
    scaler = LossScaler(enabled=True, scale=128.0)
    optimizer = DenseAdam(model, OptimizerConfig(), scaler=scaler)
    optimizer.zero_grad()
    optimizer.gradients.flat.fill_(torch.inf)
    result = optimizer.step()
    assert not result.applied
    assert result.loss_scale_before == 128 and result.loss_scale_after == 64
    torch.testing.assert_close(model.weight, initial)


def test_cost_allocation_and_reference_cache_fallback_preserve_work() -> None:
    counts = allocate_statistical_counts(
        torch.tensor([0.5, 0.0, 0.3, 0.2], dtype=torch.float64),
        torch.tensor([4.0, 9.0, 1.0, 2.0], dtype=torch.float64),
        torch.tensor([2.0, 1.0, 0.5, 4.0], dtype=torch.float64),
        17,
    )
    assert counts.sum() == 17
    assert counts.tolist()[1] == 0
    assert all(counts[index] >= 1 for index in (0, 2, 3))
    fields = {"log_policy": torch.arange(6, dtype=torch.float32)}
    rejected = ReferenceOutputCache.admit(
        fields,
        byte_budget=8,
        resident_bytes=0,
        build_bytes=0,
        predicted_saved_seconds=2.0,
        predicted_build_seconds=1.0,
    )
    assert rejected is None
    admitted = ReferenceOutputCache.admit(
        fields,
        byte_budget=64,
        resident_bytes=8,
        build_bytes=8,
        predicted_saved_seconds=2.0,
        predicted_build_seconds=1.0,
    )
    assert admitted is not None
    torch.testing.assert_close(admitted.fields["log_policy"], fields["log_policy"])
    admitted.release()
    assert admitted.released


@CUDA
def test_admitted_and_uncached_reference_paths_have_identical_updates() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    mode = next(
        schema
        for schema in MODE_SCHEMAS
        if schema.key == "self_play/sequential/sampled"
    )
    current_uncached = _LearnerModel(device)
    current_cached = _LearnerModel(device)
    current_cached.load_state_dict(current_uncached.state_dict())
    reference = _LearnerModel(device)
    rows = 3
    fields = MappingProxyType(
        {
            "model_input": torch.randn(rows, 26, device=device),
            "sampled_private_state": torch.tensor(
                [[0, 1], [1, 2], [2, 3]],
                dtype=torch.int32,
                device=device,
            ),
            "owner": torch.tensor([0, 1, 0], dtype=torch.int32, device=device),
            "action": torch.full((rows,), CHECK_CALL, dtype=torch.int32, device=device),
            "old_log_policy": torch.full((rows, 1), -math.log(4), device=device),
            "old_value": torch.zeros(rows, 1, device=device),
            "value_target": torch.linspace(-0.5, 0.5, rows, device=device)[:, None],
            "advantage_target": torch.linspace(0.2, 0.8, rows, device=device)[:, None],
        }
    )
    batch = SequentialLearnerBatch(
        "pre",
        torch.arange(rows, dtype=torch.int64, device=device),
        fields,
    )
    with torch.no_grad():
        reference_fields = reference.evaluate(
            "pre",
            fields["model_input"],
            fields["sampled_private_state"],
            fields["owner"],
        ).policy_raw
    cache = ReferenceOutputCache.admit(
        reference_fields,
        byte_budget=1 << 20,
        resident_bytes=0,
        build_bytes=0,
        predicted_saved_seconds=2.0,
        predicted_build_seconds=1.0,
    )
    assert cache is not None
    loss = LossConfig(reference_kl_coefficient=0.3)
    uncached = LearnerOperation(
        mode, loss, current_uncached.evaluate, _LearnerPolicy()
    ).forward_loss_backward(
        batch,
        iteration=4,
        reference_operation=reference.evaluate,
    )
    cached = LearnerOperation(
        mode, loss, current_cached.evaluate, _LearnerPolicy()
    ).forward_loss_backward(
        batch,
        iteration=4,
        reference_cache=cache,
    )
    for name in uncached.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(cached, name), getattr(uncached, name), atol=0, rtol=0
        )
    torch.testing.assert_close(
        current_cached.affine.weight.grad,
        current_uncached.affine.weight.grad,
        atol=0,
        rtol=0,
    )
    cache.release()


@CUDA
def test_checkpoint_resume_is_strict_and_execution_drift_is_allowed(
    tmp_path: Path,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    model = _TinyModel(device)
    reference = _TinyModel(device)
    optimizer = DenseAdam(model, OptimizerConfig())
    optimizer.exp_avg.fill_(0.3)
    optimizer.exp_avg_sq.fill_(0.4)
    optimizer.step_count = 7
    generator = torch.Generator(device=device).manual_seed(41)
    generator_state = generator.get_state().clone()
    run = RunState(
        "strict-run",
        iteration=6,
        recovered_overflow=True,
        overflow_recoveries=[{"iteration": 2}],
        metrics_prefix=[{"iteration": value} for value in range(6)],
    )
    scientific = {"mode": "self_play/sequential/sampled", "seed": 41}
    path = tmp_path / "checkpoint.pt"
    execution_manifest = seal_execution_manifest(
        {"schema": "test_resolved_execution_v1", "checkpoint_every": 5}
    )
    digest = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=optimizer.scaler,
        reference_model=reference,
        reference_state=ReferenceState(
            interval_iterations=3, interval_effective_hands=None
        ),
        run_state=run,
        scientific_config=scientific,
        execution_config=execution_manifest,
        generator=generator,
    )
    assert digest == sha256_file(path)
    model.weight.data.zero_()
    reference.weight.data.zero_()
    optimizer.exp_avg.zero_()
    _scaler, reference_state, resumed, execution = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        reference_model=reference,
        expected_scientific_config=scientific,
        expected_execution_config=execution_manifest,
        generator=generator,
    )
    torch.testing.assert_close(
        model.weight, torch.tensor([0.4, -0.8, 1.2], device=device)
    )
    torch.testing.assert_close(optimizer.exp_avg, torch.full((3,), 0.3, device=device))
    assert resumed.recovered_overflow and resumed.iteration == 6
    assert reference_state.interval_iterations == 3
    assert execution == execution_manifest
    assert torch.equal(generator.get_state(), generator_state)
    with pytest.raises(Exception, match="scientific"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            reference_model=reference,
            expected_scientific_config={**scientific, "seed": 42},
            expected_execution_config=execution_manifest,
            generator=generator,
        )


def test_runtime_scientific_product_has_no_legacy_mapping_parser() -> None:
    assert not hasattr(PokerScientificConfig, "from_mapping")
