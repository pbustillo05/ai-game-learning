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
from pbg.core.partition import ByteWorkContract, PartitionSet, PartitionSpec
from pbg.core.rollout import _selected_policy_logp
from pbg.core.tree import PagedRolloutTopology
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
        self.profile_batch_sizes: list[int] = []
        self.step_batch_sizes: list[int] = []
        self.reach_update_batch_sizes: list[int] = []

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

    def policy_forward_partition_actual_cost(
        self,
        model: nn.Module,
        partition_id: int,
        states: Tensor,
        reach: Tensor,
        *,
        physical_pack: object,
    ) -> _Output:
        assert partition_id == 0
        assert physical_pack is not None
        self.actual_cost_batch_sizes.append(int(states.numel()))
        return self.policy_forward(model, states, reach)

    def training_partition_specs(self) -> PartitionSet:
        return PartitionSet(
            (
                PartitionSpec(
                    partition_id=0,
                    name="default",
                    router=lambda states: torch.zeros_like(states, dtype=torch.int64),
                    exact_width=1,
                    model_width=1,
                    exact_to_model=(0,),
                    forward=lambda inputs: inputs,
                    byte_work=ByteWorkContract(max_rows=2**31 - 1),
                    input_schema=TensorSchema(
                        "dummy_partition",
                        (TensorFieldSchema("state", torch.int32),),
                    ),
                    preorder=0,
                ),
            )
        )

    def sample_profiles(
        self, states: Tensor, reach: Tensor, gen: torch.Generator
    ) -> Tensor:
        del reach
        self.profile_batch_sizes.append(int(states.numel()))
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
        self.step_batch_sizes.append(int(states.numel()))
        return states + 1

    def reach_update(
        self, reach: Tensor, states: Tensor, events: EventBatch, policy_out: _Output | None
    ) -> Tensor:
        del states, events, policy_out
        self.reach_update_batch_sizes.append(int(reach.shape[0]))
        return reach.clone()

    def beta_fn(self, states: Tensor) -> Tensor:
        return torch.full(states.shape, self.beta, dtype=torch.float32)

    def capture_decision_inputs(
        self, partition_id: int, states: Tensor, reach: Tensor
    ) -> Tensor:
        assert partition_id == 0
        del states
        return reach.detach().clone()

    @staticmethod
    def begin_decision_input_captures(
        *, mode: str = "branched_rb", dynamic_heads: bool = False,
        accountant: object | None = None
    ) -> dict[str, object]:
        del mode, dynamic_heads, accountant
        return {"next_decision": 0, "next_action": 0, "decisions": {}, "actions": {}}

    @staticmethod
    def reserve_decision_input_captures(store: dict[str, object], partition_id: int, count: int) -> Tensor:
        del partition_id
        start = int(store["next_decision"])
        store["next_decision"] = start + count
        return torch.arange(start, start + count, dtype=torch.int64)

    @staticmethod
    def reserve_action_captures(
        store: dict[str, object], partition_id: int, decision_rows: Tensor, counts: Tensor
    ) -> Tensor:
        del partition_id
        start = int(store["next_action"])
        total = int(counts.sum())
        store["next_action"] = start + total
        begins = start + torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)[:-1]))
        for row, begin, count in zip(decision_rows.tolist(), begins.tolist(), counts.tolist()):
            store["decisions"].setdefault(row, {})["action_begin"] = begin
            store["decisions"][row]["action_count"] = count
        return torch.arange(start, start + total, dtype=torch.int64)

    @staticmethod
    def write_decision_input_capture(
        store: dict[str, object], rows: Tensor, node_ids: Tensor, capture: Tensor,
        policy_columns: dict[str, Tensor]
    ) -> None:
        for index, row in enumerate(rows.tolist()):
            record = store["decisions"].setdefault(row, {})
            record.update({"node_id": node_ids[index], "reach": capture[index]})
            record.update({name: value[index] for name, value in policy_columns.items()})

    @staticmethod
    def write_action_captures(
        store: dict[str, object], partition_id: int, rows: Tensor,
        columns: dict[str, Tensor]
    ) -> None:
        del partition_id
        for index, row in enumerate(rows.tolist()):
            store["actions"][row] = {name: value[index] for name, value in columns.items()}

    def seal_decision_input_captures(
        self, records: dict[str, object]
    ) -> SealedDecisionInputs:
        if not records["decisions"]:
            empty = torch.empty(0, dtype=torch.int32)
            return SealedDecisionInputs(empty, empty, empty, {}, {})
        original = [records["decisions"][row] for row in sorted(records["decisions"])]
        ordered = sorted(original, key=lambda record: int(record["node_id"]))
        nodes = torch.stack([record["node_id"] for record in ordered]).to(torch.int32)
        reaches = torch.stack([record["reach"] for record in ordered]).to(torch.float32)
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
        has_dynamic_heads = "old_t" in ordered[0]
        policy_schema = TensorSchema(
            "labeled_test_policy",
            (
                TensorFieldSchema("node_id", torch.int32),
                TensorFieldSchema("old_v", torch.float32, (1,)),
                TensorFieldSchema("action_begin", torch.int64),
                TensorFieldSchema("action_count", torch.int32),
                *(
                    (TensorFieldSchema("old_t", torch.float32), TensorFieldSchema("old_c", torch.float32))
                    if has_dynamic_heads else ()
                ),
            ),
        )
        policy_columns = {
            "node_id": nodes,
            "old_v": torch.stack([record["old_v"] for record in ordered]),
            "action_begin": torch.tensor([record["action_begin"] for record in ordered]),
            "action_count": torch.tensor([record["action_count"] for record in ordered], dtype=torch.int32),
            **(
                {
                    "old_t": torch.stack([record["old_t"] for record in ordered]),
                    "old_c": torch.stack([record["old_c"] for record in ordered]),
                }
                if has_dynamic_heads else {}
            ),
        }
        policy = SealedTensorBlock(policy_schema, policy_columns, nodes.numel(), policy_schema.capacity_bytes(nodes.numel()))
        action_records = [
            records["actions"][index]
            for record in ordered
            for index in range(record["action_begin"], record["action_begin"] + record["action_count"])
        ]
        action_schema = TensorSchema(
            "labeled_test_action",
            (TensorFieldSchema("event", torch.int32), TensorFieldSchema("old_logp", torch.float32, (1,))),
        )
        action_columns = {
            "event": torch.stack([record["event"] for record in action_records]),
            "old_logp": torch.stack([record["old_logp"] for record in action_records]),
        }
        actions = SealedTensorBlock(action_schema, action_columns, len(action_records), action_schema.capacity_bytes(len(action_records)))
        ptr = torch.empty(nodes.numel() + 1, dtype=torch.int32)
        ptr[0] = 0
        ptr[1:] = policy_columns["action_count"].cumsum(0)
        rows = torch.arange(nodes.numel(), dtype=torch.int32)
        records.clear()
        return SealedDecisionInputs(
            nodes, torch.zeros_like(nodes), rows, {0: block}, {0: policy},
            {0: actions}, {0: ptr}
        )


def _cfg(
    root_count: int,
    beta: float,
    execution_memory_bytes: int = 1 << 20,
    *,
    family_batch_rows: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        roots_per_gpu=root_count,
        storage_page_rows=8,
        execution_memory_bytes=execution_memory_bytes,
        family_batch_rows=family_batch_rows,
        branching_mode="fixed",
        adaptive_kkt=None,
        safe_tree_size=None,
        queue_capacity_rows=256,
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
    sample = _sample(_LabeledBackend(4.0), _cfg(1, 4.0))
    sample.topology.validate(leaf_rows=sample.leaf_source.rows)
    assert int(sample.topology.degree[0]) == 4
    assert torch.equal(sample.decision_inputs.action_sources[0]["event"], torch.zeros(4, dtype=torch.int32))
    assert sample.decision_inputs.action_ptrs[0].tolist() == [0, 4]
    assert sample.decision_inputs.action_sources[0]["old_logp"].shape == (4, 1)


def test_duplicate_siblings_expand_independently_without_label_sorting() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    sample = _sample(backend, _cfg(1, 2.0), 7)
    tree = sample.topology
    assert backend.forward_batch_sizes == [1, 2]
    first = torch.arange(tree.child_begin[0], tree.child_begin[0] + tree.degree[0])
    descendants = [
        torch.arange(tree.child_begin[node], tree.child_begin[node] + tree.degree[node])
        for node in first
    ]
    assert all(children.numel() == 2 for children in descendants)
    assert torch.cat(descendants).unique().numel() == 4


def test_branched_uses_one_root_and_preserves_duplicate_children() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    sample = _sample(backend, _cfg(1, 2.0), 7)
    tree = sample.topology
    tree.validate(leaf_rows=sample.leaf_source.rows)
    roots = (tree.parent == -1).nonzero().squeeze(1)
    assert roots.tolist() == [0]
    first = torch.arange(tree.child_begin[0], tree.child_begin[0] + tree.degree[0])
    assert first.numel() == 2
    descendants = [
        torch.arange(tree.child_begin[node], tree.child_begin[node] + tree.degree[node])
        for node in first
    ]
    assert all(children.numel() == 2 for children in descendants)
    assert torch.cat(descendants).unique().numel() == 4
    assert backend.forward_batch_sizes == [1, 2]
    assert sample.decision_inputs.action_ptrs[0].tolist() == [0, 2, 4, 6]
    assert sample.decision_inputs.action_sources[0]["old_logp"].shape == (6, 1)


def test_branched_dispatches_actual_cost_family_blocks() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    cfg = _cfg(1, 2.0)
    sample = _sample(backend, cfg, 7)

    sample.topology.validate(leaf_rows=sample.leaf_source.rows)
    assert backend.actual_cost_batch_sizes == [1, 2]
    assert backend.forward_batch_sizes == [1, 2]
    assert sample.decision_inputs.action_ptrs[0].tolist() == [0, 2, 4, 6]


def test_child_materialization_is_bounded_and_physical_plan_invariant() -> None:
    narrow_backend = _LabeledBackend(4.0, two_levels=True)
    wide_backend = _LabeledBackend(4.0, two_levels=True)
    narrow = _sample(
        narrow_backend, _cfg(1, 4.0, family_batch_rows=2), seed=31
    )
    wide = _sample(
        wide_backend, _cfg(1, 4.0, family_batch_rows=3), seed=31
    )

    assert narrow_backend.profile_batch_sizes
    assert max(narrow_backend.profile_batch_sizes) <= 2
    assert max(narrow_backend.step_batch_sizes) <= 2
    assert max(narrow_backend.reach_update_batch_sizes) <= 2
    assert max(wide_backend.profile_batch_sizes) <= 3
    assert 2 in narrow_backend.profile_batch_sizes
    assert 3 in wide_backend.profile_batch_sizes

    for name in (
        "parent",
        "child_begin",
        "degree",
        "decision_partition",
        "decision_row",
        "leaf_row",
        "nominal_degree",
    ):
        assert torch.equal(getattr(narrow.topology, name), getattr(wide.topology, name))
    assert torch.equal(narrow.decision_inputs.action_ptrs[0], wide.decision_inputs.action_ptrs[0])
    assert torch.equal(narrow.decision_inputs.action_sources[0]["event"], wide.decision_inputs.action_sources[0]["event"])
    assert torch.equal(
        narrow.decision_inputs.action_sources[0]["old_logp"], wide.decision_inputs.action_sources[0]["old_logp"]
    )
    assert torch.equal(narrow.leaf_source["path_key"], wide.leaf_source["path_key"])


def test_upper_cutoff_switches_later_expansions_to_spines() -> None:
    backend = _LabeledBackend(2.0, two_levels=True)
    cfg = _cfg(1, 2.0)
    diagnostics: dict[str, int] = {}

    bundle = GameBundle(_Model(), backend, _Adapter(), None)
    sample = sample_branched_tree(
        bundle,
        cfg,
        torch.Generator().manual_seed(7),
        diagnostics=diagnostics,
        diagnostic_limits=BranchingDiagnosticLimits(upper_nodes=4),
    )

    degree = sample.topology.degree
    assert sample.node_count == 5
    assert int(degree[0]) == 2
    assert torch.equal(degree[1:3], torch.ones(2, dtype=torch.int64))
    assert sample.decision_inputs.action_ptrs[0].tolist() == [0, 2, 3, 4]
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
    first = _sample(_LabeledBackend(1.5), _cfg(1, 1.5), 19)
    second = _sample(_LabeledBackend(1.5), _cfg(1, 1.5), 19)
    first_count = int(first.topology.degree[0])
    second_count = int(second.topology.degree[0])
    assert first_count in {1, 2}
    assert second_count == first_count


def test_root_refill_reaches_minimum_and_finalizes_uniform_weights() -> None:
    backend = _LabeledBackend(2.3)
    cfg = _cfg(1, 2.3)
    cfg.branching_min_nodes = 10
    diagnostics: dict[str, int] = {}

    sample = sample_branched_tree(
        GameBundle(_Model(), backend, _Adapter(), None),
        cfg,
        torch.Generator().manual_seed(19),
        diagnostics=diagnostics,
    )

    tree = sample.topology
    root_degree = int(tree.degree[0])
    assert sample.node_count == 10
    assert tree.pre_refill_node_count in {3, 4}
    assert tree.root_target_degree in {2, 3}
    assert root_degree == 9
    assert diagnostics["root_refill_count"] == (
        root_degree - tree.root_target_degree
    )
    assert diagnostics["root_refill_added_nodes"] == (
        sample.node_count - tree.pre_refill_node_count
    )
    assert int(tree.nominal_degree[0]) == tree.root_target_degree
    assert sample.decision_inputs.decision_sources[0]["node_id"].tolist() == [0]
    assert sample.decision_inputs.action_ptrs[0].tolist() == [0, tree.root_target_degree]
    assert sample.decision_inputs.action_sources[0]["old_logp"].shape == (tree.root_target_degree, 1)
    tree.validate(leaf_rows=sample.leaf_source.rows)


def test_root_refill_is_disabled_without_minimum() -> None:
    ordinary = _sample(_LabeledBackend(2.3), _cfg(1, 2.3), 19).topology
    assert ordinary.root_target_degree is None
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
        PagedRolloutTopology.append_children,
        _selected_policy_logp,
    ):
        syntax = ast.parse(textwrap.dedent(inspect.getsource(function)))
        used = {
            node.attr
            for node in ast.walk(syntax)
            if isinstance(node, ast.Attribute) and node.attr in banned
        }
        assert not used
