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

from pbg.core.game import CHANCE, PRETERM, TERMINAL, EventBatch
from pbg.core.game import GameBundle
from pbg.core.rollout import sample_trees
from pbg.games.poker.adapter import PokerPolicyAdapter
from pbg.games.poker.engine import (
    ALL_IN,
    CHECK_CALL,
    FOLDED,
    LIVE,
    RAISE_OFFSET,
    NLHEEngine,
    PokerConfig,
    _State,
)
from pbg.games.poker.folded import PartitionSummaryBatch
from pbg.games.poker.model.preflop import PolicyOutput
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
        self.postflop = SimpleNamespace(cfg=SimpleNamespace(evaluator_summary_dim=0))
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
        "board", "last_full_raise", "current_bet", "action_history",
        "history_length", "fold_payoff",
    }
    assert required.issubset(vars(store))
    assert all(value.device.type == "cpu" for value in vars(store).values() if isinstance(value, Tensor))


def test_random_legal_trajectories_match_cpu_engine_step_for_step() -> None:
    cfg = PokerConfig(n_players=3, stack_bb=10.0, n_ticks=5)
    for seed in range(20):
        rng = random.Random(seed)
        cpu = NLHEEngine(cfg)
        backend = _backend(cfg)
        cpu_ref = int(cpu.root_states(1)[0])
        ref = backend.root_states(1)
        for _ in range(100):
            cpu_owner = int(cpu.owner_of(torch.tensor([cpu_ref], dtype=torch.int32))[0])
            assert int(backend.classify(ref)[0]) == cpu_owner
            if cpu_owner in {PRETERM, TERMINAL}:
                break
            if cpu_owner == CHANCE:
                street = cpu.states.street[cpu_ref]
                used = {card for card in cpu.states.board[cpu_ref] if card >= 0}
                count = 3 if street == 0 else 1
                dealt = [card for card in range(52) if card not in used][:count]
                padded = dealt + [-1] * (3 - count)
                cards = torch.tensor([padded], dtype=torch.int16)
                cpu_ref = int(cpu.deal(torch.tensor([cpu_ref]), cards)[0])
                ref = backend.step(ref, EventBatch(torch.zeros(1, dtype=torch.int32), {"cards": cards}))
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
            _compare_state(cpu, cpu_ref, backend, ref)
        else:
            raise AssertionError("legal poker trajectory exceeded test horizon")


def _install_state(backend: PokerRolloutBackend, state: _State) -> Tensor:
    ref = backend.root_states(1)
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
    cfg = PokerConfig(n_players=3, stack_bb=100.0, n_ticks=4)
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
    event = RAISE_OFFSET + 3
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
    cfg = PokerConfig(n_players=2, n_ticks=3)
    backend = _backend(cfg)
    ref = backend.root_states(2)
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


def test_policy_routes_one_large_preflop_and_one_shared_postflop_queue() -> None:
    cfg = PokerConfig(n_players=2, n_ticks=3)
    partition = _Partition()
    backend = _backend(cfg, partition_adapter=partition)
    model = _RecordingModel()
    roots = backend.root_states(32)
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


def test_core_sampler_accumulates_many_trees_before_model_forward() -> None:
    cfg = PokerConfig(n_players=2, stack_bb=4.0, n_ticks=3)
    backend = _backend(cfg, capacity=2048)
    model = _RecordingModel()
    rollout = SimpleNamespace(
        n_trees=16, node_budget=64, beta_mode="sequential", reach_storage="full"
    )
    tree, reach, policy = sample_trees(
        GameBundle(model, backend, PokerPolicyAdapter(3), None),
        rollout,
        torch.Generator().manual_seed(123),
    )
    tree.validate()
    assert int(((tree.owner == TERMINAL) | (tree.owner == PRETERM)).sum()) == 16
    assert reach.log_r.device == backend.device
    assert policy.old_v.shape[0] > len(model.calls) and policy.old_v.shape[1] == 1326
    assert all(batch > 1 for _, batch, _ in model.calls)


def test_completion_inputs_are_device_facade_tensors_and_conserve_pots() -> None:
    backend = _backend(PokerConfig(n_players=3, stack_bb=10.0, n_ticks=3))
    roots = backend.root_states(3)
    reach = torch.zeros((3, 3, 1326), dtype=torch.float32)
    inputs = backend.completion_inputs(roots, reach)
    assert set(inputs) == {
        "board_cards", "range_weights", "dealt_mask", "contender_mask",
        "stratum_players", "payoff_baseline", "stratum_amount",
    }
    assert inputs["board_cards"].dtype == torch.uint8
    assert (inputs["board_cards"] == 255).all()
    total = inputs["payoff_baseline"].sum(1) + inputs["stratum_amount"].sum(1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-6)


def test_beta_buckets_use_exact_override_or_default() -> None:
    backend = _backend(
        PokerConfig(n_players=2, n_ticks=3),
        beta_fix={"default": 1.25, "preflop": 2.0, "chance_to_flop": 3.0},
    )
    roots = backend.root_states(2)
    assert torch.equal(backend.beta_fn(roots), torch.full((2,), 2.0))
    chance = backend.step(roots, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    chance = backend.step(chance, EventBatch(torch.full((2,), CHECK_CALL, dtype=torch.int32)))
    assert torch.equal(backend.beta_fn(chance), torch.full((2,), 3.0))


def test_rollout_hot_path_has_no_explicit_host_transfers() -> None:
    banned = {"cpu", "numpy", "tolist", "item"}
    for target in (
        PokerTensorStateStore.allocate,
        PokerTensorStateStore.append_copies,
        PokerRolloutBackend.root_states,
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
def test_cuda_outputs_and_sibling_inputs_remain_on_original_device() -> None:
    device = torch.device("cuda:0")
    profiles, partition = _Profiles(), _Partition()
    backend = _backend(
        PokerConfig(n_players=2, n_ticks=3), 1024, device,
        profile_sampler=profiles, partition_adapter=partition,
    )
    roots = backend.root_states(8)
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
