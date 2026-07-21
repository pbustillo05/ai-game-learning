from __future__ import annotations

import ast
import inspect
import random
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from pbg.config import load_config
from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch
from pbg.core.game import GameBundle
from pbg.core.normal import build_normal_dataset_from_sampled
from pbg.core.rollout import sample_trees, sample_trees_sequential
from pbg.core.tree import DecisionReachCache
from pbg.games.poker.adapter import PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.bundle import _TensorPokerGPUCompletion
from pbg.games.poker.engine import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    FOLDED,
    LIVE,
    RAISE_OFFSET,
    NLHEEngine,
    PokerConfig,
    _State,
)
from pbg.games.poker.folded import PokerGPUPartitionAdapter, PartitionSummaryBatch
from pbg.games.poker.mc import PokerGPURejectionSampler
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import PokerModelConfig, PolicyOutput, PreflopNet
from pbg.games.poker.rollout_backend import PokerRolloutBackend
from pbg.games.poker.tensor_state import PokerTensorStateStore


class _Partition:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def summarize(self, **kwargs: Any) -> PartitionSummaryBatch:
        self.calls.append(kwargs)
        board = kwargs["board_cards"]
        batch = board.shape[0]
        return PartitionSummaryBatch(
            fields={
                "base_log_z": torch.zeros(batch, dtype=torch.float32, device=board.device),
                "single_card_log_ratio": torch.zeros(
                    (batch, 52), dtype=torch.float32, device=board.device
                ),
                "output_valid": torch.ones(batch, dtype=torch.bool, device=board.device),
            },
            metadata={},
        )


class _Profiles:
    def __init__(self) -> None:
        self.devices: list[torch.device] = []

    def sample(
        self, reach: Tensor, board: Tensor, dealt: Tensor, n: int, gen: torch.Generator
    ) -> Tensor:
        del board, dealt, gen
        self.devices.append(reach.device)
        combos = torch.arange(reach.shape[1], dtype=torch.int32, device=reach.device)
        return combos[None, None, :].expand(reach.shape[0], n, -1).clone()


class _RecordingModel:
    def __init__(self) -> None:
        self.postflop = SimpleNamespace(
            cfg=SimpleNamespace(evaluator_summary_dim=0),
            requires_partition_summary=True,
        )
        self.calls: list[tuple[str, int, int | None]] = []

    @staticmethod
    def _output(batch: int, rows: int, device: torch.device) -> PolicyOutput:
        raw = {
            "type_logits": torch.zeros((batch, rows, 3), device=device),
            "omega": torch.zeros((batch, rows, 2), device=device),
            "mu": torch.zeros((batch, rows, 2), device=device),
            "s": torch.ones((batch, rows, 2), device=device),
            "iota": torch.zeros((batch, rows), device=device),
        }
        return PolicyOutput(raw=raw, v=torch.zeros((batch, rows), device=device))

    def forward_preflop(
        self, pub: Any, ranges: Tensor, actor: Tensor, masks: Any
    ) -> PolicyOutput:
        del pub, actor, masks
        self.calls.append(("preflop", ranges.shape[0], None))
        return self._output(ranges.shape[0], 169, ranges.device)

    def forward_postflop(
        self, pub: Any, ranges: Tensor, actor: Tensor, masks: Any
    ) -> PolicyOutput:
        del actor
        n_live = int((masks.status != FOLDED).sum(1)[0])
        self.calls.append(("postflop", ranges.shape[0], n_live))
        assert masks.compatibility.device == ranges.device
        assert pub.board.device == ranges.device
        return self._output(ranges.shape[0], 1326, ranges.device)


class _AlwaysCheckModel(_RecordingModel):
    """Deterministic legal check/call policy used to reach every board street."""

    @staticmethod
    def _output(batch: int, rows: int, device: torch.device) -> PolicyOutput:
        output = _RecordingModel._output(batch, rows, device)
        output.raw["type_logits"][..., 0] = 20.0
        output.raw["type_logits"][..., 1:] = -20.0
        return output

    @staticmethod
    def _prefer_check_call(output: PolicyOutput, pub: Any) -> PolicyOutput:
        facing = pub.legality_summary[:, 0].bool()
        logits = output.raw["type_logits"]
        logits[facing, :, 0] = -20.0
        logits[facing, :, 1] = 20.0
        return output

    def forward_preflop(
        self, pub: Any, ranges: Tensor, actor: Tensor, masks: Any
    ) -> PolicyOutput:
        del actor, masks
        return self._prefer_check_call(self._output(ranges.shape[0], 169, ranges.device), pub)

    def forward_postflop(
        self, pub: Any, ranges: Tensor, actor: Tensor, masks: Any
    ) -> PolicyOutput:
        del actor, masks
        return self._prefer_check_call(self._output(ranges.shape[0], 1326, ranges.device), pub)


class _AlwaysFoldModel(_RecordingModel):
    """Deterministic fold policy for a player-edge replay check."""

    @staticmethod
    def _output(batch: int, rows: int, device: torch.device) -> PolicyOutput:
        output = _RecordingModel._output(batch, rows, device)
        output.raw["type_logits"][..., 0] = 20.0
        output.raw["type_logits"][..., 1:] = -20.0
        return output


def _backend(
    cfg: PokerConfig,
    capacity: int = 4096,
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> PokerRolloutBackend:
    return PokerRolloutBackend(
        cfg,
        capacity,
        device,
        max_rollout_steps=256,
        profile_sampler=kwargs.pop("profile_sampler", _Profiles()),
        partition_adapter=kwargs.pop("partition_adapter", _Partition()),
        **kwargs,
    )


def _compare_state(cpu: NLHEEngine, cpu_ref: int, backend: PokerRolloutBackend, ref: Tensor) -> None:
    state = cpu.states.state(cpu_ref)
    row = ref.long()
    assert torch.allclose(backend.states.stacks[row][0], torch.tensor(state.stacks), atol=2e-5)
    assert torch.allclose(
        backend.states.committed_street[row][0], torch.tensor(state.committed_street), atol=2e-5
    )
    assert torch.allclose(
        backend.states.committed_total[row][0], torch.tensor(state.committed_total), atol=2e-5
    )
    assert tuple(backend.states.status[row][0].tolist()) == state.status
    assert tuple(backend.states.acted[row][0].tolist()) == state.acted
    assert int(backend.states.actor[row][0]) == state.actor
    assert int(backend.states.street[row][0]) == state.street
    assert tuple(card for card in backend.states.board[row][0].tolist() if card >= 0) == state.board
    assert float(backend.states.committed_total[row][0].sum() + backend.states.stacks[row][0].sum()) == pytest.approx(
        cpu.cfg.stack_bb * cpu.n_players
    )


def test_tensor_store_has_required_device_soa_and_prefix_allocation() -> None:
    store = PokerTensorStateStore(16, 20, "cpu", n_players=3)
    requested = torch.tensor([True, False, True, True])
    handles = store.allocate(requested)
    assert torch.equal(handles, torch.tensor([0, -1, 1, 2], dtype=torch.int32))
    required = {
        "stacks", "committed_street", "committed_total", "status", "acted",
            "reopen_mask", "actor", "street", "button", "n_raises", "phase",
            "board", "last_full_raise", "current_bet", "history_parent", "history_event",
            "history_length", "fold_payoff",
        }
    assert required.issubset(vars(store))
    assert "action_history" not in vars(store)
    assert store.history_parent.shape == store.history_event.shape == (16,)
    assert all(
        value.device.type == "cpu"
        for value in vars(store).values()
        if isinstance(value, Tensor)
    )


def test_post_family_forward_batches_flop_turn_and_river_together() -> None:
    partition = _Partition()
    backend = _backend(
        PokerConfig(n_players=2), 64, "cpu", partition_adapter=partition
    )
    states = backend.sequential_start_states(3)
    rows = states.to(torch.int64)
    backend.states.street[rows] = torch.tensor(
        [1, 2, 3], dtype=backend.states.street.dtype
    )
    backend.states.board[rows] = torch.tensor(
        [
            [0, 1, 2, -1, -1],
            [0, 1, 2, 3, -1],
            [0, 1, 2, 3, 4],
        ],
        dtype=backend.states.board.dtype,
    )
    reach = torch.zeros((3, 2, 1326), dtype=torch.float32)
    model = _RecordingModel()

    output = backend.policy_forward_family(model, "post_2", states, reach)

    assert output.v.shape == (3, 1326)
    assert model.calls == [("postflop", 3, 2)]
    assert len(partition.calls) == 1
    assert partition.calls[0]["board_cards"].shape == (3, 5)


@pytest.mark.parametrize("n_players", range(2, 7))
def test_random_legal_trajectories_match_cpu_engine_step_for_step(
    n_players: int,
) -> None:
    cfg = PokerConfig(n_players=n_players, stack_bb=10.0)
    for seed in range(4):
        rng = random.Random(seed)
        cpu = NLHEEngine(cfg)
        backend = _backend(cfg)
        cpu_ref = int(cpu.initial_states(1)[0])
        ref = backend.sequential_start_states(1)
        for _ in range(100):
            cpu_owner = int(cpu.owner_of(torch.tensor([cpu_ref], dtype=torch.int32))[0])
            assert int(backend.classify(ref)[0]) == cpu_owner
            if cpu_owner in {PRETERM, TERMINAL}:
                break
            parent_ref = ref.clone()
            if cpu_owner == CHANCE:
                street = cpu.states.street[cpu_ref]
                used = {card for card in cpu.states.board[cpu_ref] if card >= 0}
                count = 3 if street == 0 else 1
                dealt = [card for card in range(52) if card not in used][:count]
                padded = dealt + [-1] * (3 - count)
                cards = torch.tensor([padded], dtype=torch.int16)
                cpu_ref = int(cpu.deal(torch.tensor([cpu_ref]), cards)[0])
                ref = backend.step(ref, EventBatch(torch.zeros(1, dtype=torch.int32), {"cards": cards}))
                expected_history_event = -1
            else:
                cpu_ids, cpu_mask = cpu.legal_events(torch.tensor([cpu_ref], dtype=torch.int32))
                ids, mask = backend.legal_events(ref)
                assert torch.equal(ids, cpu_ids)
                assert torch.equal(mask, cpu_mask)
                assert torch.allclose(
                    backend.raise_targets(ref), cpu.raise_targets(torch.tensor([cpu_ref])), atol=2e-5
                )
                legal = cpu_ids[0, cpu_mask[0]].tolist()
                event = rng.choice(legal)
                cpu_ref = int(cpu.step(torch.tensor([cpu_ref]), torch.tensor([event]))[0])
                ref = backend.step(ref, EventBatch(torch.tensor([event], dtype=torch.int32)))
                expected_history_event = event
            row = ref.long()
            assert torch.equal(backend.states.history_parent[row], parent_ref)
            assert int(backend.states.history_event[row][0]) == expected_history_event
            _compare_state(cpu, cpu_ref, backend, ref)
        else:
            raise AssertionError("legal poker trajectory exceeded test horizon")


def test_parent_linked_history_reconstructs_exact_actions_arithmetic_and_streets() -> None:
    cfg = PokerConfig(n_players=2, stack_bb=20.0)
    backend = _backend(cfg)
    root = backend.sequential_start_states(1)
    raised = backend.step(
        root, EventBatch(torch.tensor([RAISE_OFFSET + 3], dtype=torch.int32))
    )
    called = backend.step(
        raised, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32))
    )
    flop = backend.step(
        called,
        EventBatch(
            torch.zeros(1, dtype=torch.int32),
            {"cards": torch.tensor([[3, 8, 11]], dtype=torch.int16)},
        ),
    )
    history = backend.encode_public(flop).history
    assert history is not None
    assert history.mask.tolist() == [[True, True, True]]
    assert history.actor.tolist() == [[0, 1, -1]]
    assert history.street.tolist() == [[0, 0, 1]]
    assert history.action.tolist() == [[2, 1, 3]]
    torch.testing.assert_close(history.amount, torch.tensor([[2.0, 1.0, 0.0]]))
    torch.testing.assert_close(history.pot_before, torch.tensor([[1.5, 3.0, 4.0]]))
    torch.testing.assert_close(history.call_amount, torch.tensor([[0.5, 1.0, 0.0]]))
    assert backend.states.history_parent.numel() == backend.states.capacity


def _install_state(backend: PokerRolloutBackend, state: _State) -> Tensor:
    ref = backend.sequential_start_states(1)
    row = ref.long()
    backend.states.stacks[row] = torch.tensor([state.stacks])
    backend.states.committed_street[row] = torch.tensor([state.committed_street])
    backend.states.committed_total[row] = torch.tensor([state.committed_total])
    backend.states.status[row] = torch.tensor([state.status], dtype=torch.int8)
    backend.states.acted[row] = torch.tensor([state.acted])
    backend.states.reopen_mask[row] = (~backend.states.acted[row]) & (backend.states.status[row] == LIVE)
    backend.states.actor[row] = state.actor
    backend.states.street[row] = state.street
    backend.states.button[row] = state.button
    backend.states.n_raises[row] = state.n_raises
    backend.states.phase[row] = state.phase
    backend.states.board[row] = -1
    backend.states.board[row, : len(state.board)] = torch.tensor(state.board, dtype=torch.int16)
    backend.states.last_full_raise[row] = state.last_full_raise
    backend.states.current_bet[row] = state.current_bet
    return ref


def test_short_all_in_reopen_and_side_pot_strata_match_oracle() -> None:
    cfg = PokerConfig(n_players=3, stack_bb=100.0)
    fixture = _State(
        street=1, board=(0, 5, 10), stacks=(92.0, 92.0, 2.0),
        committed_street=(8.0, 8.0, 8.0), committed_total=(8.0, 8.0, 98.0),
        status=(LIVE, LIVE, LIVE), actor=2, last_full_raise=7.0, current_bet=8.0,
        acted=(True, True, False), button=0, phase=0, betting_closed=False,
        n_raises=1, action_history=(),
    )
    cpu = NLHEEngine(cfg)
    cpu_ref = cpu.states.append(fixture)
    backend = _backend(cfg)
    ref = _install_state(backend, fixture)
    assert torch.equal(backend.legal_events(ref)[1], cpu.legal_events(torch.tensor([cpu_ref]))[1])
    event = RAISE_OFFSET + 19
    cpu_ref = int(cpu.step(torch.tensor([cpu_ref]), torch.tensor([event]))[0])
    ref = backend.step(ref, EventBatch(torch.tensor([event], dtype=torch.int32)))
    _compare_state(cpu, cpu_ref, backend, ref)
    assert not backend.legal_events(ref)[1][0, RAISE_OFFSET:].any()

    side = _State(
        street=3, board=(0, 4, 8, 12, 16), stacks=(0.0, 50.0, 80.0),
        committed_street=(0.0, 0.0, 0.0), committed_total=(100.0, 50.0, 20.0),
        status=(ALL_IN, LIVE, FOLDED), actor=-1, last_full_raise=1.0,
        current_bet=0.0, acted=(True, True, True), button=0, phase=3,
        betting_closed=True, n_raises=0, action_history=(),
    )
    side_ref = _install_state(backend, side)
    strata = backend.pot_strata(side_ref)
    assert torch.equal(strata.stratum_amount, torch.tensor([[60.0, 60.0, 50.0]]))
    assert torch.equal(strata.stratum_players, torch.tensor(
        [[[True, True, False], [True, True, False], [True, False, False]]]
    ))


def test_chance_labels_are_deterministic_street_disjoint_and_payload_is_padded() -> None:
    cfg = PokerConfig(n_players=2)
    backend = _backend(cfg)
    ref = backend.sequential_start_states(2)
    ref = backend.step(ref, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    ref = backend.step(ref, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    profiles = torch.tensor([[0, 100], [0, 100]], dtype=torch.int32)
    event = backend.sample_events(ref, profiles, None, object(), torch.Generator().manual_seed(9))
    assert event.payload["cards"].shape == (2, 3)
    same = event.payload["cards"][0].sort().values.equal(event.payload["cards"][1].sort().values)
    assert (event.public_id[0] == event.public_id[1]) == same
    flop = backend.step(ref, event)
    flop = backend.step(flop, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    flop = backend.step(flop, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    turn_event = backend.sample_events(flop, profiles, None, object(), torch.Generator().manual_seed(10))
    assert (turn_event.payload["cards"][:, 1:] == -1).all()
    assert (turn_event.public_id >= 200_000).all() and (turn_event.public_id < 300_000).all()
    assert (event.public_id >= 100_000).all() and (event.public_id < 200_000).all()


def test_mixed_player_and_chance_rows_step_in_one_branched_batch() -> None:
    """Chance payload padding must not reclassify neighboring player rows."""

    backend = _backend(PokerConfig(n_players=2, stack_bb=10.0))
    roots = backend.sequential_start_states(2)
    chance = backend.step(
        roots[:1], EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32))
    )
    chance = backend.step(
        chance, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32))
    )
    assert int(backend.classify(chance)[0]) == CHANCE

    events = EventBatch(
        torch.tensor([FOLD, 100_000], dtype=torch.int32),
        {"cards": torch.tensor([[0, 0, 0], [3, 7, 11]], dtype=torch.int16)},
    )
    successors = backend.step(torch.cat((roots[1:], chance)), events)

    assert int(backend.classify(successors[:1])[0]) == TERMINAL
    assert int(backend.states.street[successors[1:].long()][0]) == 1
    assert 0 <= int(backend.classify(successors[1:])[0]) < 2
    assert int(backend.states.history_event[successors.long()][0]) == FOLD
    assert int(backend.states.history_event[successors.long()][1]) == -1


def test_policy_routes_one_large_preflop_and_one_shared_postflop_queue() -> None:
    cfg = PokerConfig(n_players=2)
    partition = _Partition()
    backend = _backend(cfg, partition_adapter=partition)
    model = _RecordingModel()
    roots = backend.sequential_start_states(32)
    reach = torch.zeros((32, 2, 1326), dtype=torch.float32)
    output = backend.policy_forward(model, roots, reach)
    assert output.v.shape == (32, 1326)
    assert model.calls == [("preflop", 32, None)]
    chance = backend.step(roots, EventBatch(torch.full((32,), CHECK_CALL, dtype=torch.int32)))
    chance = backend.step(chance, EventBatch(torch.full((32,), CHECK_CALL, dtype=torch.int32)))
    cards = torch.tensor([[3, 7, 11]] * 32, dtype=torch.int16)
    flop = backend.step(chance, EventBatch(torch.full((32,), 100_000, dtype=torch.int32), {"cards": cards}))
    output = backend.policy_forward(model, flop, reach)
    assert output.v.shape == (32, 1326)
    assert model.calls[-1] == ("postflop", 32, 2)
    assert len(partition.calls) == 1
    assert set(partition.calls[0]) == {"board_cards", "live_range_weights", "folded"}
    assert backend.batch_size_histograms == {"preflop": [32], "postflop_2": [32]}


def test_device_partition_cache_masks_hits_and_invalidates_on_fold() -> None:
    cfg = PokerConfig(n_players=3, stack_bb=10.0)
    partition = _Partition()
    backend = _backend(cfg, partition_adapter=partition)
    state = _State(
        street=1, board=(0, 5, 10), stacks=(8.0, 8.0, 8.0),
        committed_street=(1.0, 1.0, 2.0), committed_total=(1.0, 1.0, 2.0),
        status=(LIVE, LIVE, LIVE), actor=0, last_full_raise=1.0,
        current_bet=2.0, acted=(False, False, False), button=0, phase=0,
        betting_closed=False, n_raises=1, action_history=(),
    )
    current = _install_state(backend, state)
    reach = torch.zeros((1, 3, 1326), dtype=torch.float32)
    model = _RecordingModel()
    backend.policy_forward(model, current, reach)
    assert len(partition.calls) == 1

    called = backend.step(current, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32)))
    backend.policy_forward(model, called, reach)
    assert len(partition.calls) == 2

    folded = backend.step(called, EventBatch(torch.tensor([FOLD], dtype=torch.int32)))
    backend.policy_forward(model, folded, reach)
    assert len(partition.calls) == 3
    assert partition.calls[-1]["folded"].folder_mask.tolist() == [[True]]

    decision_states = torch.cat((called, folded))
    decision_reach = torch.stack(
        (
            torch.arange(3 * 1326, dtype=torch.float32).reshape(3, 1326),
            torch.arange(3 * 1326, dtype=torch.float32).reshape(3, 1326) + 10_000,
        )
    )
    retained = DecisionReachCache.from_rows(
        torch.tensor([0, 1], dtype=torch.int32),
        decision_states,
        decision_reach,
        backend,
    )
    assert isinstance(retained.payload, Tensor)
    assert torch.equal(retained.payload, decision_reach)
    compact = backend.compact_decision_reach(decision_states, decision_reach)
    # Three live seats before the fold and two after it are retained; the
    # folded full row is gone while the model-ready folder summary stays cached.
    assert compact.log_r.shape == (5, 1326)
    sampled_rows = torch.tensor([1, 0, 1], dtype=torch.int64)
    expanded = backend.expand_decision_reach(compact, sampled_rows)
    status = backend.states.status[decision_states[sampled_rows].long()]
    expected = decision_reach[sampled_rows]
    assert torch.equal(expanded[status != FOLDED], expected[status != FOLDED])
    assert torch.isneginf(expanded[status == FOLDED]).all()
    backend.policy_forward(model, folded, expanded[:1])
    assert len(partition.calls) == 4
    assert partition.calls[-1]["folded"].folder_mask.tolist() == [[False]]


def test_policy_without_partition_consumer_skips_evaluator() -> None:
    cfg = PokerConfig(n_players=2)
    partition = _Partition()
    backend = _backend(cfg, partition_adapter=partition)
    model = _RecordingModel()
    model.postflop.requires_partition_summary = False
    roots = backend.sequential_start_states(1)
    chance = backend.step(roots, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32)))
    chance = backend.step(chance, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32)))
    flop = backend.step(
        chance,
        EventBatch(
            torch.tensor([100_000], dtype=torch.int32),
            {"cards": torch.tensor([[3, 7, 11]], dtype=torch.int16)},
        ),
    )
    backend.policy_forward(model, flop, torch.zeros((1, 2, 1326)))
    assert not partition.calls


def test_policy_planner_mixes_postflop_streets_and_splits_live_counts() -> None:
    cfg = PokerConfig(n_players=3)
    backend = _backend(cfg)
    backend.execution_profile = load_config(
        "configs/train/nlhe6.yaml", hardware_profile="local_3x2080ti"
    ).hardware
    states = backend.sequential_start_states(4)
    rows = states.long()
    backend.states.street[rows] = torch.tensor(
        [1, 2, 3, 1], dtype=backend.states.street.dtype
    )
    backend.states.status[rows] = torch.tensor(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]],
        dtype=backend.states.status.dtype,
    )
    model = PokerModelBundle(
        PreflopNet(PokerModelConfig(max_players=3)),
        PostflopNet(PokerModelConfig(max_players=3)),
    )
    plans = backend.plan_policy_batches(model, states, "training")
    assert len(plans) == 3
    assert all(
        len(set((backend.states.status[rows[plan]] != FOLDED).sum(1).tolist())) == 1
        for plan in plans
    )
    mixed = next(plan for plan in plans if plan.numel() == 2)
    assert set(backend.states.street[rows[mixed]].tolist()) == {2, 3}


def test_core_sampler_accumulates_many_trees_before_model_forward() -> None:
    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    profiles = _Profiles()
    backend = _backend(cfg, capacity=2048, profile_sampler=profiles)
    model = _RecordingModel()
    rollout = SimpleNamespace(
        roots_per_gpu=16, node_budget=64
    )
    tree, reach, policy = sample_trees_sequential(
        GameBundle(model, backend, PokerPolicyAdapter(cfg.n_ticks), None),
        rollout,
        torch.Generator().manual_seed(123),
    )
    tree.validate()
    assert int(((tree.owner == TERMINAL) | (tree.owner == PRETERM)).sum()) == 16
    assert reach.log_r.device == backend.device
    assert policy.old_v.shape[0] > len(model.calls) and policy.old_v.shape[1] == 1326
    assert all(batch > 1 for _, batch, _ in model.calls)
    assert profiles.devices == [backend.device]


def test_poker_family_scheduled_sequential_rollout_closes_chance_segments() -> None:
    """The opt-in queue runs poker paths across depths without pending chance rows."""

    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    profiles = _Profiles()
    backend = _backend(cfg, capacity=1024, profile_sampler=profiles)
    model = _AlwaysCheckModel()
    rollout = SimpleNamespace(
        roots_per_gpu=8,
        node_budget=64,
        family_batch_rows=4,
        max_active_trajectories=8,
        arena_chunk_nodes=32,
        logical_seed=123,
    )
    tree, reach, policy = sample_trees_sequential(
        GameBundle(model, backend, PokerPolicyAdapter(cfg.n_ticks), None),
        rollout,
        torch.Generator().manual_seed(999),
        iteration=2,
    )
    tree.validate()
    assert int(((tree.owner == TERMINAL) | (tree.owner == PRETERM)).sum()) == 8
    assert reach.log_r.shape[0] == 8
    assert policy.decision_reach is not None
    assert policy.decision_reach.node_ids.numel() == policy.node_ids.numel()
    assert policy.node_ids.numel() > 0
    assert bool((backend.states.street[tree.state_ref.long()] > 0).any())
    assert all(batch > 1 for _, batch, _ in model.calls)
    assert profiles.devices == [backend.device]



def test_poker_actual_cost_family_block_sorts_and_splits_without_changing_order() -> None:
    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    backend = _backend(cfg, capacity=256)
    backend.execution_profile = SimpleNamespace(
        name="tiny_actual_cost",
        inference_max_tokens=5_000,
        inference_max_token_pairs=6_000_000,
        training_max_tokens=5_000,
        training_max_token_pairs=6_000_000,
    )
    model = _AlwaysCheckModel()
    states = backend.sequential_start_states(4)
    reach = torch.zeros((4, 2, 1326))
    expected = backend.policy_forward_family(model, "pre", states, reach)
    actual = backend.policy_forward_family_actual_cost(model, "pre", states, reach)
    assert torch.equal(actual.v, expected.v)
    assert all(torch.equal(actual.raw[name], expected.raw[name]) for name in actual.raw)
    snapshot = backend.execution_metrics.snapshot()
    assert snapshot["execution_model_calls"] >= 2


def test_poker_actual_cost_charges_mixed_postflop_rows_at_flop_width() -> None:
    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    backend = _backend(cfg, capacity=16)
    model = _AlwaysCheckModel()
    states = backend.sequential_start_states(3)
    rows = states.to(torch.int64)
    backend.states.street[rows] = torch.tensor(
        [1, 2, 3], dtype=backend.states.street.dtype
    )
    backend.states.history_length[rows] = 4
    widths, workspace = backend.rollout_row_cost(model, states)
    assert widths.unique().numel() == 1
    assert torch.equal(workspace, torch.zeros_like(widths))


def test_poker_actual_cost_singleton_skips_cost_planning(monkeypatch) -> None:
    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    backend = _backend(cfg, capacity=16)
    backend.execution_profile = SimpleNamespace(
        name="tiny_actual_cost_singleton",
        inference_max_tokens=1_000,
        inference_max_token_pairs=6_000_000,
        training_max_tokens=1_000,
        training_max_token_pairs=6_000_000,
    )
    model = _AlwaysCheckModel()
    states = backend.sequential_start_states(1)
    reach = torch.zeros((1, 2, 1326))

    def unexpected_cost(*args, **kwargs):
        raise AssertionError("singleton actual-cost forward must not plan widths")

    monkeypatch.setattr(backend, "rollout_row_cost", unexpected_cost)
    output = backend.policy_forward_family_actual_cost(model, "pre", states, reach)

    assert output.v.shape == (1, 1326)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_poker_actual_cost_keeps_physical_pack_indices_on_device() -> None:
    """Bounded CUDA history makes a family width read unnecessary."""

    device = torch.device("cuda", torch.cuda.current_device())
    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    backend = _backend(cfg, capacity=32, device=device)
    backend.execution_profile = SimpleNamespace(
        name="tiny_actual_cost_cuda",
        inference_max_tokens=1_000,
        inference_max_token_pairs=6_000_000,
        training_max_tokens=1_000,
        training_max_token_pairs=6_000_000,
    )
    model = _AlwaysCheckModel()
    states = backend.sequential_start_states(4)
    reach = torch.zeros((4, 2, 1326), device=device)

    output = backend.policy_forward_family_actual_cost(model, "pre", states, reach)
    torch.cuda.synchronize(device)

    assert output.v.shape == (4, 1326)
    snapshot = backend.execution_metrics.snapshot()
    assert snapshot["execution_model_calls"] == 1
    assert snapshot["execution_scheduler_control_synchronizations"] == 0
    assert snapshot["execution_h2d_transfers"] == 0
    assert snapshot["execution_h2d_bytes"] == 0



def _execution_public_paths(tree) -> tuple[tuple[int, ...], ...]:
    paths: list[tuple[int, ...]] = []
    roots = (tree.parent == -1).nonzero(as_tuple=False).squeeze(1)
    for root in roots.cpu().tolist():
        node = root
        events: list[int] = []
        while True:
            first = int(tree.child_ptr[node])
            stop = int(tree.child_ptr[node + 1])
            if first == stop:
                break
            assert stop == first + 1
            events.append(int(tree.child_event[first]))
            node = int(tree.child_node[first])
        paths.append(tuple(events))
    return tuple(paths)


def _real_execution_poker_sample(
    device: torch.device, *, batch_rows: int, admission: int
):
    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    backend = _backend(
        cfg,
        capacity=512,
        device=device,
        profile_sampler=PokerGPURejectionSampler(
            backend="cuda" if device.type == "cuda" else "cpu"
        ),
        partition_adapter=PokerGPUPartitionAdapter(beta_rank=4),
    )
    model = _AlwaysCheckModel()
    rollout = SimpleNamespace(
        roots_per_gpu=4,
        node_budget=64,
        family_batch_rows=batch_rows,
        max_active_trajectories=admission,
        arena_chunk_nodes=16,
        logical_seed=741,
    )
    generator = torch.Generator(device=device).manual_seed(999)
    sampled = sample_trees_sequential(
        GameBundle(model, backend, PokerPolicyAdapter(cfg.n_ticks), None),
        rollout,
        generator,
        return_profiles=True,
        iteration=12,
    )
    return cfg, backend, model, sampled


def _assert_real_execution_poker_sample(cfg, backend, model, sampled) -> None:
    package = pytest.importorskip("poker_gpu_evaluator")
    evaluator = package.bind(rank_count=13)
    tree, reach, policy, profiles = sampled
    tree.validate()
    assert profiles.shape == (4, 2)
    cards = evaluator.combo_cards.to(profiles.device)[profiles.to(torch.int64)]
    ordered = cards.reshape(4, -1).sort(1).values
    assert bool((ordered[:, 1:] != ordered[:, :-1]).all())
    chance = tree.owner == CHANCE
    assert bool(((tree.child_ptr[1:] - tree.child_ptr[:-1])[chance] == 1).all())

    assert policy.decision_reach is not None
    rows = policy.decision_reach.gather(
        torch.arange(policy.node_ids.numel(), device=policy.node_ids.device), backend
    )
    states = tree.state_ref[policy.node_ids.to(torch.int64)]
    with torch.no_grad():
        current = backend.policy_forward(model, states, rows)
    legal_ids, legal_mask = backend.legal_events(states)
    edge_rows = tree.child_ptr[policy.node_ids.to(torch.int64)].to(torch.int64)
    selected = tree.child_event[edge_rows, None]
    current_logp = PokerPolicyAdapter(cfg.n_ticks).log_prob(
        current, selected, legal_ids, legal_mask
    ).float()
    torch.testing.assert_close(current.v.float(), policy.old_v)
    torch.testing.assert_close(current_logp, policy.old_logp_child)
    metrics = backend.execution_metrics.snapshot()
    assert metrics["execution_evaluator_rows"] == 4
    assert metrics["execution_evaluator_calls"] >= 1


def test_real_sibling_execution_poker_cpu_is_logically_addressed_and_collision_free() -> None:
    pytest.importorskip("poker_gpu_evaluator")
    left = _real_execution_poker_sample(torch.device("cpu"), batch_rows=1, admission=1)
    right = _real_execution_poker_sample(torch.device("cpu"), batch_rows=4, admission=4)
    _assert_real_execution_poker_sample(*left)
    _assert_real_execution_poker_sample(*right)
    assert torch.equal(left[3][3], right[3][3])
    assert _execution_public_paths(left[3][0]) == _execution_public_paths(right[3][0])

    normal_cfg = SimpleNamespace(
        rollout=SimpleNamespace(
            roots_per_gpu=4,
            logical_seed=741,
        ),
        loss=SimpleNamespace(lam=0.95),
    )
    datasets = []
    canonical_orders = []
    for game_cfg, backend, model, sampled in (left, right):
        dataset = build_normal_dataset_from_sampled(
                GameBundle(
                    model,
                    backend,
                    PokerPolicyAdapter(game_cfg.n_ticks),
                    None,
                ),
                normal_cfg,
                torch.Generator().manual_seed(123),
                sampled,
                iteration=12,
            )
        datasets.append(dataset)
        tree, _, policy, _ = sampled
        nodes = policy.node_ids.to(torch.int64)
        stride = int(tree.depth_player.max()) + 1
        canonical_orders.append(
            torch.argsort(
                tree.tree_id[nodes].to(torch.int64) * stride
                + tree.depth_player[nodes].to(torch.int64)
            )
        )
    for field in (
        "actor_type",
        "event",
        "old_logp",
        "old_v",
        "lambda_return",
    ):
        torch.testing.assert_close(
            getattr(datasets[0], field)[canonical_orders[0]],
            getattr(datasets[1], field)[canonical_orders[1]],
            msg=lambda message, name=field: f"{name}: {message}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_sibling_execution_poker_tiny_cuda() -> None:
    sampled = _real_execution_poker_sample(
        torch.device("cuda", torch.cuda.current_device()), batch_rows=2, admission=2
    )
    _assert_real_execution_poker_sample(*sampled)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    ("mode", "beta_table"),
    (
        ("rao_blackwell", {"default": 1.0}),
        ("branched_rao_blackwell", {"default": 1.0, "preflop": 2.0}),
    ),
)
def test_poker_last_consumer_reaches_are_schedule_repeatable(
    mode: str, beta_table: dict[str, float]
) -> None:
    """Leaf and decision reaches are written once and repeat exactly."""

    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    model = _AlwaysCheckModel()
    common = dict(
        roots_per_gpu=1 if mode == "branched_rao_blackwell" else 2,
        node_budget=128 if mode == "branched_rao_blackwell" else 64,
    )
    if mode == "branched_rao_blackwell":
        common.update(branching_mode="fixed", adaptive_kkt=None)
    full_backend = _backend(cfg, capacity=256, beta_table=beta_table)
    replay_backend = _backend(cfg, capacity=256, beta_table=beta_table)
    sampler = (
        sample_branched_tree
        if mode == "branched_rao_blackwell"
        else sample_trees_sequential
    )
    full_tree, full_reach, full_policy = sampler(
        GameBundle(model, full_backend, PokerPolicyAdapter(cfg.n_ticks), None),
        SimpleNamespace(**common),
        torch.Generator().manual_seed(731),
    )
    replay_tree, replay_reach, replay_policy = sampler(
        GameBundle(model, replay_backend, PokerPolicyAdapter(cfg.n_ticks), None),
        SimpleNamespace(**common),
        torch.Generator().manual_seed(731),
    )
    for field in ("parent", "child_ptr", "child_node", "child_event", "owner", "state_ref"):
        assert torch.equal(getattr(replay_tree, field), getattr(full_tree, field))
    assert torch.equal(replay_reach.node_index, full_reach.node_index)
    assert torch.equal(replay_reach.log_r, full_reach.log_r)
    assert full_policy.decision_reach is not None
    assert replay_policy.decision_reach is not None
    decision_rows = torch.arange(full_policy.node_ids.numel())
    expected = full_policy.decision_reach.gather(decision_rows, full_backend)
    actual = replay_policy.decision_reach.gather(decision_rows, replay_backend)
    assert torch.equal(actual, expected)

    streets = replay_backend.states.street[replay_tree.state_ref.long()]
    assert bool(((replay_tree.owner >= 0) & (streets > 0)).any())
    events = replay_tree.child_event
    assert bool(((events >= 100_000) & (events < 200_000)).any())
    assert bool(((events >= 200_000) & (events < 300_000)).any())
    assert bool(((events >= 300_000) & (events < 400_000)).any())
    if mode == "branched_rao_blackwell":
        has_duplicate_label = False
        for parent in range(replay_tree.parent.numel()):
            start, stop = int(replay_tree.child_ptr[parent]), int(replay_tree.child_ptr[parent + 1])
            labels = replay_tree.child_event[start:stop]
            has_duplicate_label |= labels.numel() > labels.unique().numel()
        assert has_duplicate_label


def test_poker_fold_reaches_are_retained_only_at_last_consumers() -> None:
    cfg = PokerConfig(n_players=2, stack_bb=8.0)
    rollout = SimpleNamespace(
        roots_per_gpu=2, node_budget=8
    )
    full_backend = _backend(cfg, capacity=32)
    replay_backend = _backend(cfg, capacity=32)
    model = _AlwaysFoldModel()
    full_tree, full_reach, full_policy = sample_trees_sequential(
        GameBundle(model, full_backend, PokerPolicyAdapter(cfg.n_ticks), None),
        rollout,
        torch.Generator().manual_seed(732),
    )
    replay_tree, replay_reach, replay_policy = sample_trees_sequential(
        GameBundle(model, replay_backend, PokerPolicyAdapter(cfg.n_ticks), None),
        SimpleNamespace(**dict(vars(rollout))),
        torch.Generator().manual_seed(732),
    )
    assert torch.equal(replay_reach.node_index, full_reach.node_index)
    assert torch.equal(replay_reach.log_r, full_reach.log_r)
    assert full_policy.decision_reach is not None
    assert replay_policy.decision_reach is not None
    rows = torch.arange(full_policy.node_ids.numel())
    assert torch.equal(
        full_policy.decision_reach.gather(rows, full_backend),
        replay_policy.decision_reach.gather(rows, replay_backend),
    )
    assert bool((replay_tree.child_event == FOLD).any())


def test_completion_inputs_are_device_facade_tensors_and_conserve_pots() -> None:
    backend = _backend(PokerConfig(n_players=3, stack_bb=10.0))
    roots = backend.sequential_start_states(3)
    reach = torch.zeros((3, 3, 1326), dtype=torch.float32)
    inputs = backend.completion_inputs(roots, reach)
    assert set(inputs) == {
        "board_cards", "range_weights", "stratum_level",
        "payoff_baseline", "stratum_amount",
    }
    assert inputs["board_cards"].dtype == torch.uint8
    assert (inputs["board_cards"] == 255).all()
    total = inputs["payoff_baseline"].sum(1) + inputs["stratum_amount"].sum(1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-6)


def test_real_tensor_poker_keyed_completion_mixes_showdown_and_fold_rows() -> None:
    """The tensor-native boundary keys both sibling completion paths directly."""

    pytest.importorskip("poker_gpu_evaluator")
    backend = _backend(
        PokerConfig(n_players=2, stack_bb=8.0),
        capacity=32,
        profile_sampler=PokerGPURejectionSampler(backend="cpu"),
    )
    roots = backend.sequential_start_states(2)

    fold = backend.step(
        roots[:1], EventBatch(torch.tensor([FOLD], dtype=torch.int32))
    )
    all_in = backend.step(
        roots[1:],
        EventBatch(torch.tensor([RAISE_OFFSET + backend.cfg.n_ticks - 1], dtype=torch.int32)),
    )
    showdown = backend.step(
        all_in, EventBatch(torch.tensor([CHECK_CALL], dtype=torch.int32))
    )
    states = torch.cat((showdown, fold))
    assert torch.equal(
        backend.classify(states), torch.tensor([PRETERM, TERMINAL], dtype=torch.int32)
    )

    tree = SimpleNamespace(
        state_ref=states,
        owner=torch.tensor([PRETERM, TERMINAL], dtype=torch.int32),
    )
    node_ids = torch.tensor([0, 1], dtype=torch.int32)
    reach = torch.zeros((2, 2, 1326), dtype=torch.float32)
    keys = torch.tensor([1_234_567, 7_654_321], dtype=torch.int64)
    completion = _TensorPokerGPUCompletion(backend="cpu")

    first = completion.complete_keyed(backend, tree, node_ids, reach, 8, keys)
    second = completion.complete_keyed(backend, tree, node_ids, reach, 8, keys)

    for name in (
        "p_hat",
        "v_hat",
        "n_eff",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    torch.testing.assert_close(first.p_hat.sum(2), torch.ones((2, 2)))
    assert first.metadata["fold_completion"] == "joint_profiles"
    assert first.metadata["showdown_contract_version"] == "1"

    metrics = backend.execution_metrics.snapshot()
    assert metrics["execution_evaluator_calls"] == 4
    assert metrics["execution_evaluator_rows"] == 4


def test_beta_buckets_use_exact_override_or_default() -> None:
    backend = _backend(
        PokerConfig(n_players=2),
        beta_table={"default": 1.25, "preflop": 2.0, "chance_to_flop": 3.0},
    )
    roots = backend.sequential_start_states(2)
    assert torch.equal(backend.beta_fn(roots), torch.full((2,), 2.0))
    chance = backend.step(roots, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    chance = backend.step(chance, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    assert torch.equal(backend.beta_fn(chance), torch.full((2,), 3.0))


def test_rollout_hot_path_has_no_explicit_host_transfers() -> None:
    banned = {"cpu", "numpy", "tolist", "item"}
    for target in (
        PokerTensorStateStore.allocate,
        PokerTensorStateStore.append_copies,
        PokerRolloutBackend.sequential_start_states,
        PokerRolloutBackend.legal_events,
        PokerRolloutBackend.step,
        PokerRolloutBackend.sample_events,
        PokerRolloutBackend.reach_update,
        PokerRolloutBackend.policy_forward,
    ):
        syntax = ast.parse(textwrap.dedent(inspect.getsource(target)))
        used = {
            node.attr
            for node in ast.walk(syntax)
            if isinstance(node, ast.Attribute) and node.attr in banned
        }
        assert not used


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_legal_event_conditioning_survives_illegal_type_underflow() -> None:
    device = torch.device("cuda:0")
    raw = {
        "type_logits": torch.tensor(
            [[-1000.0, -1001.0, 1000.0], [-1000.0, -1001.0, 1000.0]],
            device=device,
        ),
        "omega": torch.full((2, 2), 0.5, device=device),
        "mu": torch.tensor([[0.25, 0.75], [0.25, 0.75]], device=device),
        "s": torch.full((2, 2), 0.2, device=device),
        "iota": torch.full((2,), 0.15, device=device),
    }
    ids = torch.arange(6, dtype=torch.int32, device=device)[None].expand(2, -1)
    legal = torch.zeros((2, 6), dtype=torch.bool, device=device)
    legal[0, CHECK_CALL] = True
    legal[1, :2] = True

    probability = PokerPolicyAdapter(n_ticks=4).probabilities(raw, ids, legal)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(probability.sum(1), torch.ones(2, device=device))
    assert probability[0, CHECK_CALL] == 1.0
    assert bool(torch.isfinite(probability).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_outputs_and_sibling_inputs_remain_on_original_device() -> None:
    device = torch.device("cuda:0")
    profiles, partition = _Profiles(), _Partition()
    backend = _backend(
        PokerConfig(n_players=2), 1024, device,
        profile_sampler=profiles, partition_adapter=partition,
    )
    roots = backend.sequential_start_states(8)
    reach = torch.zeros((8, 2, 1326), device=device)
    sampled = backend.sample_profiles(roots, reach, torch.Generator(device=device).manual_seed(11))
    assert sampled.device == device and profiles.devices == [device]
    output = backend.policy_forward(_RecordingModel(), roots, reach)
    assert output.v.device == device
    chance = backend.step(roots, EventBatch(torch.full((8,), CHECK_CALL, dtype=torch.int32, device=device)))
    chance = backend.step(chance, EventBatch(torch.full((8,), CHECK_CALL, dtype=torch.int32, device=device)))
    cards = torch.tensor([[3, 7, 11]] * 8, dtype=torch.int16, device=device)
    flop = backend.step(chance, EventBatch(torch.full((8,), 100_000, dtype=torch.int32, device=device), {"cards": cards}))
    post = backend.policy_forward(_RecordingModel(), flop, reach)
    assert post.v.device == device
    assert partition.calls[0]["board_cards"].device == device
    assert all(
        value.device == device
        for value in vars(backend.states).values()
        if isinstance(value, Tensor)
    )
from pbg.core.branching import sample_branched_tree
