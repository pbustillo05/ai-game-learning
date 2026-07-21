from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import Tensor, nn

from pbg.core.branching import (
    BranchingCutoffReached,
    BranchingDiagnosticLimits,
    _index_policy_output,
    sample_branched_tree,
)
from pbg.config import ConfigError, load_config
from pbg.core.game import EventBatch, GameBundle, TERMINAL
from pbg.core.rollout import _selected_policy_logp
from pbg.core.tree import PagedTreeSource
from pbg.core.storage import (
    SealedDecisionInputs,
    SealedTensorBlock,
    TensorFieldSchema,
    TensorSchema,
)


ROOT = Path(__file__).parents[1]


@dataclass
class _Output:
    raw: Tensor
    v: Tensor


class _Model(nn.Module):
    def forward(self, states: Tensor) -> _Output:
        batch = states.shape[0]
        values = states.to(torch.float32).reshape(batch, 1)
        return _Output(
            raw=values[:, :, None].expand(-1, -1, 2).clone(),
            v=values,
        )


class _Adapter:
    def log_prob(
        self,
        output: _Output,
        selected: Tensor,
        legal_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        del legal_ids, legal_mask
        return torch.zeros(
            (*selected.shape, 1), dtype=torch.float32, device=selected.device
        )


class _LabeledBackend:
    device = torch.device("cpu")
    n_players = 2
    x_sizes = (1, 1)
    max_rollout_steps = 2

    def __init__(self, beta: float, two_levels: bool = False) -> None:
        self.beta = beta
        self.two_levels = two_levels
        self.forward_batch_sizes: list[int] = []
        self.actual_cost_batch_sizes: list[int] = []

    def _initial_state_batch(self, n: int) -> Tensor:
        return torch.zeros(n, dtype=torch.int32)

    def sequential_start_states(self, count: int) -> Tensor:
        return self._initial_state_batch(count)

    def branched_root_state(self) -> Tensor:
        return self._initial_state_batch(1)

    def classify(self, states: Tensor) -> Tensor:
        terminal_at = 2 if self.two_levels else 1
        return torch.where(states >= terminal_at, torch.full_like(states, TERMINAL), 0)

    def legal_events(self, states: Tensor) -> tuple[Tensor, Tensor]:
        ids = torch.tensor([[0, 1]], dtype=torch.int32).expand(states.shape[0], -1)
        mask = torch.ones_like(ids, dtype=torch.bool)
        if not self.two_levels:
            mask[:, 1] = False
        return ids, mask

    def policy_forward(self, model: nn.Module, states: Tensor, reach: Tensor) -> _Output:
        del reach
        self.forward_batch_sizes.append(int(states.numel()))
        return model(states)

    def policy_forward_family_actual_cost(
        self,
        model: nn.Module,
        family: str,
        states: Tensor,
        reach: Tensor,
    ) -> _Output:
        assert family == "default"
        self.actual_cost_batch_sizes.append(int(states.numel()))
        return self.policy_forward(model, states, reach)

    def sample_profiles(
        self, states: Tensor, reach: Tensor, gen: torch.Generator
    ) -> Tensor:
        del reach
        draw = torch.randint(2, (states.shape[0],), generator=gen, device=states.device)
        return torch.stack((draw, torch.zeros_like(draw)), dim=1).to(torch.int32)

    def sample_events(
        self,
        states: Tensor,
        profiles: Tensor,
        policy_out: _Output | None,
        adapter: _Adapter,
        gen: torch.Generator,
    ) -> EventBatch:
        del adapter, gen
        if policy_out is None:
            raise AssertionError("this rigged game has no chance nodes")
        public_id = torch.where(states == 0, torch.zeros_like(states), profiles[:, 0])
        return EventBatch(public_id=public_id.to(torch.int32))

    def step(self, states: Tensor, events: EventBatch) -> Tensor:
        del events
        return states + 1

    def reach_update(
        self, reach: Tensor, states: Tensor, events: EventBatch, policy_out: _Output | None
    ) -> Tensor:
        del states, events, policy_out
        return reach.clone()

    def beta_fn(self, states: Tensor) -> Tensor:
        return torch.full(states.shape, self.beta, dtype=torch.float32)

    def capture_decision_inputs(
        self, family: str, states: Tensor, reach: Tensor
    ) -> Tensor:
        del family, states
        return reach.detach().clone()

    @staticmethod
    def begin_decision_input_captures(
        *, accountant: object | None = None
    ) -> list[tuple[Tensor, Tensor]]:
        del accountant
        return []

    @staticmethod
    def append_decision_input_capture(
        store: list[tuple[Tensor, Tensor]], node_ids: Tensor, capture: Tensor
    ) -> None:
        store.append((node_ids, capture))

    def seal_decision_input_captures(
        self, records: list[tuple[Tensor, Tensor]]
    ) -> SealedDecisionInputs:
        if not records:
            empty = torch.empty(0, dtype=torch.int32)
            return SealedDecisionInputs(empty, empty, empty, {})
        nodes = torch.cat([node for node, _ in records]).to(torch.int32)
        reaches = torch.cat([reach for _, reach in records]).to(torch.float32)
        order = torch.argsort(nodes.to(torch.int64), stable=True)
        nodes, reaches = nodes[order], reaches[order]
        schema = TensorSchema(
            "labeled_test_decision_input",
            (TensorFieldSchema("reach", torch.float32, (2, 1)),),
        )
        block = SealedTensorBlock(
            schema,
            {"reach": reaches},
            nodes.numel(),
            schema.capacity_bytes(nodes.numel(), chunks=int(nodes.numel() > 0)),
        )
        rows = torch.arange(nodes.numel(), dtype=torch.int32)
        records.clear()
        return SealedDecisionInputs(nodes, torch.zeros_like(nodes), rows, {0: block})


def _cfg(
    root_count: int, beta: float, execution_memory_bytes: int = 1 << 20
) -> SimpleNamespace:
    return SimpleNamespace(
        roots_per_gpu=root_count,
        storage_page_rows=8,
        execution_memory_bytes=execution_memory_bytes,
        branching_mode="fixed",
        adaptive_kkt=None,
    )


def _sample(
    backend: _LabeledBackend,
    cfg: SimpleNamespace,
    seed: int = 1,
    *,
    diagnostic_limits: BranchingDiagnosticLimits | None = None,
):
    bundle = GameBundle(_Model(), backend, _Adapter(), None)
    return sample_branched_tree(
        bundle,
        cfg,
        torch.Generator().manual_seed(seed),
        diagnostic_limits=diagnostic_limits,
    )


def test_equal_labeled_siblings_remain_distinct_with_uniform_weights() -> None:
    tree, _, policy = _sample(_LabeledBackend(4.0), _cfg(1, 4.0))
    tree.validate()
    roots = (tree.parent == -1).nonzero().squeeze(1)
    counts = tree.child_ptr[roots + 1] - tree.child_ptr[roots]
    assert torch.equal(counts, torch.full_like(counts, 4))
    assert torch.equal(tree.child_event, torch.zeros_like(tree.child_event))
    assert torch.allclose(tree.nu, torch.full_like(tree.nu, 0.25), atol=0, rtol=0)
    assert torch.allclose(tree.path_weight[tree.child_node], tree.nu, atol=0, rtol=0)
    assert policy.action_ptr.tolist() == [0, 4]
    assert policy.old_logp_action.shape == (4, 1)


def test_duplicate_siblings_expand_independently_without_label_sorting() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    tree, _, _ = _sample(backend, _cfg(1, 2.0), 7)
    assert backend.forward_batch_sizes == [1, 2]
    roots = (tree.parent == -1).nonzero().squeeze(1)
    assert roots.numel() == 1
    for root in roots:
        first = tree.child_node[tree.child_ptr[root] : tree.child_ptr[root + 1]]
        assert torch.equal(
            tree.child_event[tree.child_ptr[root] : tree.child_ptr[root + 1]],
            torch.zeros(2, dtype=torch.int32),
        )
        descendants = [
            tree.child_node[tree.child_ptr[node] : tree.child_ptr[node + 1]]
            for node in first
        ]
        assert all(children.numel() == 2 for children in descendants)
        assert torch.cat(descendants).unique().numel() == 4


def test_branched_uses_one_root_and_preserves_duplicate_children() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    tree, _, policy = _sample(backend, _cfg(1, 2.0), 7)
    tree.validate()
    roots = (tree.parent == -1).nonzero().squeeze(1)
    assert roots.tolist() == [0]
    first = tree.child_node[tree.child_ptr[0] : tree.child_ptr[1]]
    assert first.numel() == 2
    descendants = [
        tree.child_node[tree.child_ptr[node] : tree.child_ptr[node + 1]]
        for node in first
    ]
    assert all(children.numel() == 2 for children in descendants)
    assert torch.cat(descendants).unique().numel() == 4
    assert backend.forward_batch_sizes == [1, 2]
    assert policy.action_ptr.tolist() == [0, 2, 4, 6]
    assert policy.old_logp_action.shape == (6, 1)


def test_branched_dispatches_actual_cost_family_blocks() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    cfg = _cfg(1, 2.0)
    tree, _, policy = _sample(backend, cfg, 7)

    tree.validate()
    assert backend.actual_cost_batch_sizes == [1, 2]
    assert backend.forward_batch_sizes == [1, 2]
    assert policy.action_ptr.tolist() == [0, 2, 4, 6]


def test_upper_cutoff_switches_later_expansions_to_spines() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    cfg = _cfg(1, 2.0)
    diagnostics: dict[str, int] = {}

    bundle = GameBundle(_Model(), backend, _Adapter(), None)
    tree, _, policy = sample_branched_tree(
        bundle,
        cfg,
        torch.Generator().manual_seed(7),
        diagnostics=diagnostics,
        diagnostic_limits=BranchingDiagnosticLimits(upper_nodes=4),
    )

    degree = tree.child_ptr[1:] - tree.child_ptr[:-1]
    assert tree.owner.numel() == 5
    assert int(degree[0]) == 2
    assert torch.equal(degree[1:3], torch.ones(2, dtype=torch.int64))
    assert policy.action_ptr.tolist() == [0, 2, 3, 4]
    assert diagnostics == {
        "upper_cutoff_batches": 1,
        "upper_cutoff_parents": 2,
        "upper_cutoff_avoided_children": 2,
        "upper_cutoff_first_node_count": 3,
    }


def test_stop_cutoff_reports_unexpanded_live_queue() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    cfg = _cfg(1, 2.0)
    bundle = GameBundle(_Model(), backend, _Adapter(), None)

    with pytest.raises(BranchingCutoffReached) as caught:
        sample_branched_tree(
            bundle,
            cfg,
            torch.Generator().manual_seed(7),
            diagnostic_limits=BranchingDiagnosticLimits(stop_nodes=3),
        )

    cutoff = caught.value
    assert cutoff.node_count == 3
    assert cutoff.pending_nodes == 2
    assert cutoff.queued_nodes == 0
    assert cutoff.frontier_nodes == 2
    assert cutoff.family_queue_nodes == {"default": 0}
    assert cutoff.completed_leaves == 0
    assert cutoff.completed_decisions == 1
    assert backend.actual_cost_batch_sizes == [1]


def test_root_fractional_rounding_is_keyed_and_repeatable() -> None:
    first, _, _ = _sample(_LabeledBackend(1.5), _cfg(1, 1.5), 19)
    second, _, _ = _sample(_LabeledBackend(1.5), _cfg(1, 1.5), 19)
    first_count = int(first.child_ptr[1] - first.child_ptr[0])
    second_count = int(second.child_ptr[1] - second.child_ptr[0])
    assert first_count in {1, 2}
    assert second_count == first_count


def test_root_refill_reaches_minimum_and_finalizes_uniform_weights() -> None:
    backend = _LabeledBackend(2.3)
    cfg = _cfg(1, 2.3)
    cfg.branching_min_nodes = 10
    diagnostics: dict[str, int] = {}

    tree, _, policy = sample_branched_tree(
        GameBundle(_Model(), backend, _Adapter(), None),
        cfg,
        torch.Generator().manual_seed(19),
        diagnostics=diagnostics,
    )

    root_degree = int(tree.child_ptr[1] - tree.child_ptr[0])
    assert tree.owner.numel() == 10
    assert tree.pre_refill_node_count in {3, 4}
    assert tree.root_pre_refill_child_count in {2, 3}
    assert root_degree == 9
    assert diagnostics["root_refill_count"] == (
        root_degree - tree.root_pre_refill_child_count
    )
    assert diagnostics["root_refill_added_nodes"] == (
        tree.owner.numel() - tree.pre_refill_node_count
    )
    assert int(tree.nominal_child_count[0]) == tree.root_pre_refill_child_count
    assert torch.allclose(
        tree.nu[:root_degree],
        torch.full((root_degree,), 1.0 / root_degree),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        tree.path_weight[1:],
        torch.full((root_degree,), 1.0 / root_degree),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert policy.node_ids.tolist() == [0]
    assert policy.action_ptr.tolist() == [0, tree.root_pre_refill_child_count]
    assert policy.old_logp_action.shape == (tree.root_pre_refill_child_count, 1)
    tree.validate()


def test_root_refill_is_disabled_without_minimum() -> None:
    ordinary, _, _ = _sample(_LabeledBackend(2.3), _cfg(1, 2.3), 19)
    assert ordinary.root_pre_refill_child_count is None
    assert ordinary.pre_refill_node_count is None


def test_branched_overflow_is_transactional() -> None:
    backend = _LabeledBackend(4.0)
    with pytest.raises(MemoryError, match="byte budget exceeded"):
        _sample(backend, _cfg(1, 4.0, execution_memory_bytes=1))
    assert backend.forward_batch_sizes == []


def test_policy_output_row_expansion_matches_explicit_sibling_forward() -> None:
    model = _Model()
    parent_states = torch.tensor([0, 1, 2], dtype=torch.int32)
    parent_output = model(parent_states)
    rows = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64)
    gathered = _index_policy_output(parent_output, rows)
    repeated = model(parent_states[rows])
    assert torch.equal(gathered.raw, repeated.raw)
    assert torch.equal(gathered.v, repeated.v)


def test_branched_sampler_rejects_multiple_root_replicas() -> None:
    with pytest.raises(ValueError, match="one root"):
        _sample(_LabeledBackend(2.0), _cfg(2, 2.0))


def test_strict_config_rejects_neyman_and_invalid_fixed_beta(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "train" / "nlhe6.yaml").read_text())
    raw["training"]["mode"] = "neyman"
    path = tmp_path / "neyman.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="training.mode"):
        load_config(path)
    raw["training"]["mode"] = "branched_rao_blackwell"
    raw["rollout"]["beta"] = {"default": float("nan")}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="finite value at least one"):
        load_config(path)
    raw["rollout"]["beta"] = {"default": 2.0}
    raw["rollout"]["sequential_roots_per_gpu_initial"] = 2
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="exactly one root per rank"):
        load_config(path)

    raw["rollout"]["sequential_roots_per_gpu_initial"] = 1
    raw["rollout"]["branching_min_nodes"] = 1
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="branching_min_nodes"):
        load_config(path)


def test_labeled_hot_path_has_no_explicit_host_transfers_or_label_sort() -> None:
    banned = {"cpu", "numpy", "tolist", "item", "sort", "argsort"}
    for function in (
        PagedTreeSource.append_children,
        PagedTreeSource.finalize,
        _selected_policy_logp,
    ):
        syntax = ast.parse(textwrap.dedent(inspect.getsource(function)))
        used = {
            node.attr
            for node in ast.walk(syntax)
            if isinstance(node, ast.Attribute) and node.attr in banned
        }
        assert not used
