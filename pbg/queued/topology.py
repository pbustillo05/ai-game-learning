"""Compact CPU topology and exact reverse-layer planning.

The topology is intentionally numerical and game-opaque.  Rollout packets use
absolute structural addresses; sealing compacts headless unary rows and emits
one immutable movement/layer plan for the CUDA target executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch
from torch import Tensor

from pbg.queued.capacity import CapacityManifest
from pbg.queued.contracts import ModeSchema, PartitionContract, RolloutMode


class TopologyError(ValueError):
    """A topology packet or sealed structural forest is invalid."""


class TargetKind(IntEnum):
    TOPOLOGY = 0
    LEAF = 1


def _cpu_vector(
    name: str,
    value: Tensor,
    length: int,
    dtype: torch.dtype,
) -> None:
    if (
        value.device.type != "cpu"
        or value.dtype != dtype
        or value.ndim != 1
        or value.numel() != length
        or not value.is_contiguous()
    ):
        raise TopologyError(f"{name} must be contiguous CPU {dtype} [{length}]")


@dataclass(frozen=True)
class TopologyBindings:
    """Planning metadata derived from partitioned rollout owners."""

    input_type: Tensor
    output_type: Tensor
    sigma_target_row: Tensor
    child_output_type: Tensor

    def validate(self, nodes: int, edges: int) -> None:
        _cpu_vector("input_type", self.input_type, nodes, torch.int32)
        _cpu_vector("output_type", self.output_type, nodes, torch.int32)
        _cpu_vector(
            "sigma_target_row", self.sigma_target_row, nodes, torch.int32
        )
        _cpu_vector(
            "child_output_type", self.child_output_type, edges, torch.int32
        )


@dataclass(frozen=True)
class Topology:
    """One replica-local compact structural forest on CPU."""

    parent: Tensor
    child_begin: Tensor
    executed_degree: Tensor
    nominal_degree: Tensor
    target_degree: Tensor
    owner: Tensor
    familywork_row: Tensor
    sigma_head_row: Tensor
    action_begin: Tensor
    label: Tensor
    path_length: Tensor
    target_kind: Tensor
    target_row: Tensor

    @property
    def node_count(self) -> int:
        return int(self.parent.numel())

    @property
    def edge_count(self) -> int:
        return int(self.label.numel())

    @property
    def nbytes(self) -> int:
        return 4 * (9 * self.node_count + 4 * self.edge_count)

    def validate(
        self,
        bindings: TopologyBindings,
        *,
        p_train: int,
        compact: bool = False,
    ) -> None:
        n = self.node_count
        e = self.edge_count
        for name in (
            "parent",
            "child_begin",
            "executed_degree",
            "nominal_degree",
            "target_degree",
            "owner",
            "familywork_row",
            "sigma_head_row",
            "action_begin",
        ):
            _cpu_vector(name, getattr(self, name), n, torch.int32)
        for name in ("label", "path_length", "target_kind", "target_row"):
            _cpu_vector(name, getattr(self, name), e, torch.int32)
        bindings.validate(n, e)
        if isinstance(p_train, bool) or p_train <= 0:
            raise TopologyError("p_train must be positive")
        if n == 0:
            if e:
                raise TopologyError("an empty topology cannot own edges")
            return

        parent = self.parent.tolist()
        begin = self.child_begin.tolist()
        degree = self.executed_degree.tolist()
        nominal = self.nominal_degree.tolist()
        target = self.target_degree.tolist()
        owner = self.owner.tolist()
        family = self.familywork_row.tolist()
        sigma = self.sigma_head_row.tolist()
        action = self.action_begin.tolist()
        input_type = bindings.input_type.tolist()
        output_type = bindings.output_type.tolist()
        sigma_target = bindings.sigma_target_row.tolist()
        target_kind = self.target_kind.tolist()
        target_row = self.target_row.tolist()
        child_output = bindings.child_output_type.tolist()
        path_length = self.path_length.tolist()

        incoming = [0] * n
        for node in range(n):
            if parent[node] < -1 or parent[node] >= n or parent[node] == node:
                raise TopologyError("parent address is outside the structural forest")
            if degree[node] < 1:
                raise TopologyError("retained structural nodes must have children")
            if not 1 <= target[node] <= degree[node]:
                raise TopologyError("target degree must satisfy 1 <= N_tgt <= N")
            if nominal[node] < 1:
                raise TopologyError("nominal degree must be positive")
            if begin[node] < 0 or begin[node] + degree[node] > e:
                raise TopologyError("node child span is outside TopologyChild")
            trained = owner[node] >= 0
            if trained:
                if owner[node] >= p_train:
                    raise TopologyError("owner slot is outside P_train")
                if min(
                    family[node],
                    sigma[node],
                    action[node],
                    input_type[node],
                    sigma_target[node],
                ) < 0:
                    raise TopologyError("trained topology row has a missing address")
            elif any(
                value != -1
                for value in (
                    family[node],
                    sigma[node],
                    action[node],
                    input_type[node],
                    sigma_target[node],
                )
            ):
                raise TopologyError("structural row cannot address heads or targets")
            if output_type[node] < 0:
                raise TopologyError("every topology row needs an output domain")
            if compact and not trained and degree[node] == nominal[node] == 1:
                raise TopologyError("headless ordinary unary row was not compressed")

            for edge in range(begin[node], begin[node] + degree[node]):
                if path_length[edge] < 1:
                    raise TopologyError("compressed path length must be positive")
                kind = target_kind[edge]
                row = target_row[edge]
                if kind == int(TargetKind.TOPOLOGY):
                    if row < 0 or row >= n:
                        raise TopologyError("topology edge target is out of range")
                    incoming[row] += 1
                    if child_output[edge] != output_type[row]:
                        raise TopologyError(
                            "child output domain disagrees with its topology target"
                        )
                elif kind == int(TargetKind.LEAF):
                    if row < 0:
                        raise TopologyError("leaf target row must be nonnegative")
                else:
                    raise TopologyError("unknown topology target kind")
                if child_output[edge] < 0:
                    raise TopologyError("every child endpoint needs an output domain")

        for node, count in enumerate(incoming):
            if parent[node] == -1:
                if count != 0:
                    raise TopologyError("a structural root has an incoming edge")
            elif count != 1:
                raise TopologyError("every nonroot structural row needs one edge")
        for node in range(n):
            if parent[node] >= 0:
                edge_slice = range(
                    begin[parent[node]],
                    begin[parent[node]] + degree[parent[node]],
                )
                if not any(
                    target_kind[edge] == int(TargetKind.TOPOLOGY)
                    and target_row[edge] == node
                    for edge in edge_slice
                ):
                    raise TopologyError("parent and child endpoint columns disagree")

        # Parent links must be acyclic even when packet rows are not preorder.
        for start in range(n):
            seen: set[int] = set()
            node = start
            while node >= 0:
                if node in seen:
                    raise TopologyError("topology parent relation contains a cycle")
                seen.add(node)
                node = parent[node]

    def compact(
        self, bindings: TopologyBindings, *, p_train: int
    ) -> tuple["Topology", TopologyBindings]:
        """Delete headless ordinary unary rows and summarize their paths."""

        self.validate(bindings, p_train=p_train)
        n = self.node_count
        if n == 0:
            return self, bindings
        degree = self.executed_degree.tolist()
        nominal = self.nominal_degree.tolist()
        owner = self.owner.tolist()
        keep = [
            owner[row] >= 0 or degree[row] >= 2 or nominal[row] >= 2
            for row in range(n)
        ]
        if all(keep):
            self.validate(bindings, p_train=p_train, compact=True)
            return self, bindings

        old_to_new: dict[int, int] = {}
        for old, retained in enumerate(keep):
            if retained:
                old_to_new[old] = len(old_to_new)
        columns = {
            name: getattr(self, name).tolist()
            for name in (
                "parent",
                "child_begin",
                "executed_degree",
                "nominal_degree",
                "target_degree",
                "owner",
                "familywork_row",
                "sigma_head_row",
                "action_begin",
                "label",
                "path_length",
                "target_kind",
                "target_row",
            )
        }
        input_type = bindings.input_type.tolist()
        output_type = bindings.output_type.tolist()
        sigma_target = bindings.sigma_target_row.tolist()
        child_output = bindings.child_output_type.tolist()

        new_parent = [-1] * len(old_to_new)
        new_begin: list[int] = []
        new_degree: list[int] = []
        new_nominal: list[int] = []
        new_target_degree: list[int] = []
        new_owner: list[int] = []
        new_family: list[int] = []
        new_sigma: list[int] = []
        new_action: list[int] = []
        new_input: list[int] = []
        new_output: list[int] = []
        new_sigma_target: list[int] = []
        new_label: list[int] = []
        new_path: list[int] = []
        new_kind: list[int] = []
        new_target: list[int] = []
        new_child_output: list[int] = []

        for old in range(n):
            if not keep[old]:
                continue
            new = old_to_new[old]
            new_begin.append(len(new_label))
            new_degree.append(degree[old])
            new_nominal.append(nominal[old])
            new_target_degree.append(columns["target_degree"][old])
            new_owner.append(owner[old])
            new_family.append(columns["familywork_row"][old])
            new_sigma.append(columns["sigma_head_row"][old])
            new_action.append(columns["action_begin"][old])
            new_input.append(input_type[old])
            new_output.append(output_type[old])
            new_sigma_target.append(sigma_target[old])
            first = columns["child_begin"][old]
            for edge in range(first, first + degree[old]):
                label = columns["label"][edge]
                path = columns["path_length"][edge]
                kind = columns["target_kind"][edge]
                row = columns["target_row"][edge]
                endpoint_output = child_output[edge]
                while kind == int(TargetKind.TOPOLOGY) and not keep[row]:
                    unary_edge = columns["child_begin"][row]
                    path += columns["path_length"][unary_edge]
                    label = columns["label"][unary_edge]
                    kind = columns["target_kind"][unary_edge]
                    row = columns["target_row"][unary_edge]
                    endpoint_output = child_output[unary_edge]
                if kind == int(TargetKind.TOPOLOGY):
                    row = old_to_new[row]
                    if new_parent[row] != -1:
                        raise TopologyError("compaction produced duplicate parents")
                    new_parent[row] = new
                new_label.append(label)
                new_path.append(path)
                new_kind.append(kind)
                new_target.append(row)
                new_child_output.append(endpoint_output)

        def i32(values: Sequence[int]) -> Tensor:
            return torch.tensor(values, dtype=torch.int32)

        compacted = Topology(
            parent=i32(new_parent),
            child_begin=i32(new_begin),
            executed_degree=i32(new_degree),
            nominal_degree=i32(new_nominal),
            target_degree=i32(new_target_degree),
            owner=i32(new_owner),
            familywork_row=i32(new_family),
            sigma_head_row=i32(new_sigma),
            action_begin=i32(new_action),
            label=i32(new_label),
            path_length=i32(new_path),
            target_kind=i32(new_kind),
            target_row=i32(new_target),
        )
        compact_bindings = TopologyBindings(
            input_type=i32(new_input),
            output_type=i32(new_output),
            sigma_target_row=i32(new_sigma_target),
            child_output_type=i32(new_child_output),
        )
        compacted.validate(compact_bindings, p_train=p_train, compact=True)
        return compacted, compact_bindings


@dataclass(frozen=True)
class TopologyPacket:
    """One committed absolute-address slice of a topology transfer."""

    sequence: int
    node_begin: int
    edge_begin: int
    commit_sequence: int
    topology: Topology
    bindings: TopologyBindings

    def __post_init__(self) -> None:
        if min(self.sequence, self.node_begin, self.edge_begin) < 0:
            raise TopologyError("packet addresses must be nonnegative")
        if self.commit_sequence != self.sequence:
            raise TopologyError("topology packet is not transactionally committed")
        self.bindings.validate(
            self.topology.node_count, self.topology.edge_count
        )


def assemble_topology_packets(
    packets: Sequence[TopologyPacket], *, p_train: int
) -> tuple[Topology, TopologyBindings]:
    """Assemble stable packet order without accepting a partial transfer."""

    if not packets:
        raise TopologyError("at least one topology packet is required")
    ordered = sorted(packets, key=lambda packet: packet.sequence)
    node_cursor = 0
    edge_cursor = 0
    for sequence, packet in enumerate(ordered):
        if packet.sequence != sequence:
            raise TopologyError("topology packet sequence has a gap")
        if packet.node_begin != node_cursor or packet.edge_begin != edge_cursor:
            raise TopologyError("topology packet extents are not contiguous")
        node_cursor += packet.topology.node_count
        edge_cursor += packet.topology.edge_count

    def join(name: str) -> Tensor:
        return torch.cat(
            tuple(getattr(packet.topology, name) for packet in ordered)
        ).contiguous()

    topology = Topology(
        parent=join("parent"),
        child_begin=join("child_begin"),
        executed_degree=join("executed_degree"),
        nominal_degree=join("nominal_degree"),
        target_degree=join("target_degree"),
        owner=join("owner"),
        familywork_row=join("familywork_row"),
        sigma_head_row=join("sigma_head_row"),
        action_begin=join("action_begin"),
        label=join("label"),
        path_length=join("path_length"),
        target_kind=join("target_kind"),
        target_row=join("target_row"),
    )
    bindings = TopologyBindings(
        input_type=torch.cat(
            tuple(packet.bindings.input_type for packet in ordered)
        ).contiguous(),
        output_type=torch.cat(
            tuple(packet.bindings.output_type for packet in ordered)
        ).contiguous(),
        sigma_target_row=torch.cat(
            tuple(packet.bindings.sigma_target_row for packet in ordered)
        ).contiguous(),
        child_output_type=torch.cat(
            tuple(packet.bindings.child_output_type for packet in ordered)
        ).contiguous(),
    )
    topology.validate(bindings, p_train=p_train)
    return topology.compact(bindings, p_train=p_train)


@dataclass(frozen=True, order=True)
class LayerGroup:
    layer: int
    output_type: int
    input_type: int
    begin: int
    end: int


@dataclass(frozen=True)
class LayerPlan:
    """Exact CPU movement and reverse-layer execution plan."""

    layer_ptr: Tensor
    carry_ptr: Tensor
    replica: Tensor
    child_begin: Tensor
    child_count: Tensor
    destination_work_row: Tensor
    input_type: Tensor
    output_type: Tensor
    owner_slot: Tensor
    nominal_degree: Tensor
    target_degree: Tensor
    sigma_head_row: Tensor
    sigma_target_row: Tensor
    action_begin: Tensor
    child_source_work_row: Tensor
    child_source_output_type: Tensor
    child_path_length: Tensor
    child_cost_multiplier: Tensor
    carry_replica: Tensor
    carry_source_work_row: Tensor
    carry_destination_work_row: Tensor
    carry_output_type: Tensor
    movement_replica: Tensor
    movement_source_leaf_row: Tensor
    movement_destination_work_row: Tensor
    movement_output_type: Tensor
    groups: tuple[LayerGroup, ...]
    arena_rows: tuple[tuple[int, ...], ...]
    sigma_rows: tuple[tuple[int, ...], ...]
    action_rows: tuple[tuple[int, ...], ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    @property
    def layer_count(self) -> int:
        return int(self.layer_ptr.numel()) - 1

    @property
    def node_count(self) -> int:
        return int(self.replica.numel())

    @property
    def edge_count(self) -> int:
        return int(self.child_source_work_row.numel())

    @property
    def carry_count(self) -> int:
        return int(self.carry_replica.numel())

    @property
    def movement_count(self) -> int:
        return int(self.movement_replica.numel())

    @property
    def carry_bytes(self) -> int:
        # Last-consumer slot planning leaves the legacy carry schema empty.
        return self.carry_count * 4 * 4

    @property
    def nbytes(self) -> int:
        tensors = (
            self.layer_ptr,
            self.carry_ptr,
            self.replica,
            self.child_begin,
            self.child_count,
            self.destination_work_row,
            self.input_type,
            self.output_type,
            self.owner_slot,
            self.nominal_degree,
            self.target_degree,
            self.sigma_head_row,
            self.sigma_target_row,
            self.action_begin,
            self.child_source_work_row,
            self.child_source_output_type,
            self.child_path_length,
            self.child_cost_multiplier,
            self.carry_replica,
            self.carry_source_work_row,
            self.carry_destination_work_row,
            self.carry_output_type,
            self.movement_replica,
            self.movement_source_leaf_row,
            self.movement_destination_work_row,
            self.movement_output_type,
        )
        return sum(value.numel() * value.element_size() for value in tensors)

    def assert_admitted(self, manifest: CapacityManifest) -> None:
        if self.node_count > manifest.topology_cap:
            raise TopologyError("realized LayerPlan exceeds topology_cap")
        if self.edge_count > manifest.topology_edge_cap:
            raise TopologyError("realized LayerPlan exceeds topology_edge_cap")
        if any(
            rows > manifest.target_arena_cap
            for lane_rows in self.arena_rows
            for rows in lane_rows
        ):
            raise TopologyError("realized LayerPlan exceeds target_arena_cap")
        if self.nbytes > manifest.movement_scratch_bytes:
            raise TopologyError("LayerPlan upload exceeds movement scratch admission")


def _depths(topology: Topology) -> list[int]:
    kinds = topology.target_kind.tolist()
    targets = topology.target_row.tolist()
    begin = topology.child_begin.tolist()
    degree = topology.executed_degree.tolist()
    state = [0] * topology.node_count
    depth = [0] * topology.node_count

    def visit(node: int) -> int:
        if state[node] == 1:
            raise TopologyError("topology child relation contains a cycle")
        if state[node] == 2:
            return depth[node]
        state[node] = 1
        child_depth = 0
        for edge in range(begin[node], begin[node] + degree[node]):
            if kinds[edge] == int(TargetKind.TOPOLOGY):
                child_depth = max(child_depth, visit(targets[edge]))
        depth[node] = child_depth + 1
        state[node] = 2
        return depth[node]

    for node in range(topology.node_count):
        visit(node)
    return depth


def build_layer_plan(
    mode: ModeSchema,
    partitions: PartitionContract,
    topologies: Sequence[Topology],
    bindings: Sequence[TopologyBindings],
    *,
    p_train: int,
    manifest: CapacityManifest | None = None,
) -> LayerPlan:
    """Compact replica forests and assign exact ping-pong work addresses."""

    if len(topologies) != len(bindings) or not topologies:
        raise TopologyError("one binding set is required per replica topology")
    if mode.rollout_mode is RolloutMode.SEQUENTIAL and any(
        int(value) != 1
        for topology in topologies
        for value in topology.executed_degree
    ):
        raise TopologyError("sequential topology requires N=1 at every row")
    compacted = [
        topology.compact(binding, p_train=p_train)
        for topology, binding in zip(topologies, bindings, strict=True)
    ]
    input_names = tuple(item.name for item in partitions.inputs)
    output_names = tuple(item.name for item in partitions.outputs)
    input_count = len(input_names)
    output_count = len(output_names)
    input_to_output = {
        index: output_names.index(partitions.output_for(name).name)
        for index, name in enumerate(input_names)
    }
    for topology, binding in compacted:
        topology.validate(binding, p_train=p_train, compact=True)
        if bool(
            ((binding.output_type < 0) | (binding.output_type >= output_count)).any()
        ):
            raise TopologyError("node output type is outside the partition contract")
        trained = binding.input_type >= 0
        if bool(
            (
                trained
                & (
                    (binding.input_type >= input_count)
                    | (
                        torch.tensor(
                            [
                                input_to_output.get(int(value), -1)
                                for value in binding.input_type
                            ],
                            dtype=torch.int32,
                        )
                        != binding.output_type
                    )
                )
            ).any()
        ):
            raise TopologyError("trained input/output partition binding is invalid")

    depths = [_depths(topology) for topology, _ in compacted]
    layer_count = max((max(value, default=0) for value in depths), default=0)
    # A nonnegative source slot addresses arena zero.  A negative slot ``-r-1``
    # addresses row ``r`` in arena one.  This preserves the fixed LayerPlan
    # child schema while allowing the selected last-consumer plan to retain
    # shorter branches in their producing arena without physical carries.
    current: dict[tuple[int, int, int, int], int] = {}
    movement: list[tuple[int, int, int, int]] = []
    arena_max = [[0] * output_count for _ in compacted]
    next_slot = [
        [[0, 0] for _ in range(output_count)] for _ in compacted
    ]
    free_slot: list[list[list[list[int]]]] = [
        [[[], []] for _ in range(output_count)] for _ in compacted
    ]

    for lane, (topology, binding) in enumerate(compacted):
        for edge in range(topology.edge_count):
            if int(topology.target_kind[edge]) != int(TargetKind.LEAF):
                continue
            output = int(binding.child_output_type[edge])
            if not 0 <= output < output_count:
                raise TopologyError("leaf output type is outside the contract")
            leaf = int(topology.target_row[edge])
            key = (lane, int(TargetKind.LEAF), leaf, output)
            if key in current:
                raise TopologyError("a leaf endpoint is referenced more than once")
            row = next_slot[lane][output][0]
            next_slot[lane][output][0] += 1
            current[key] = row
            movement.append((lane, leaf, row, output))
        for output in range(output_count):
            arena_max[lane][output] = next_slot[lane][output][0]

    node_records: list[tuple[int, ...]] = []
    child_records: list[tuple[int, int, int, float]] = []
    carry_records: list[tuple[int, int, int, int]] = []
    layer_ptr = [0]
    carry_ptr = [0]
    groups: list[LayerGroup] = []

    for layer in range(layer_count):
        selected: list[tuple[int, int, int, int]] = []
        for lane, ((topology, binding), lane_depth) in enumerate(
            zip(compacted, depths, strict=True)
        ):
            for node, depth in enumerate(lane_depth):
                if depth == layer + 1:
                    selected.append(
                        (
                            int(binding.output_type[node]),
                            int(binding.input_type[node]),
                            lane,
                            node,
                        )
                    )
        selected.sort()
        consumed: set[tuple[int, int, int, int]] = set()
        consumed_slots: set[tuple[int, int, int, int]] = set()
        new_results: dict[tuple[int, int, int, int], int] = {}
        root_destinations: list[tuple[int, int, int, int]] = []
        group_key: tuple[int, int] | None = None
        group_begin = len(node_records)
        destination_arena = (layer + 1) & 1

        for output, input_type, lane, node in selected:
            if group_key != (output, input_type):
                if group_key is not None:
                    groups.append(
                        LayerGroup(
                            layer,
                            group_key[0],
                            group_key[1],
                            group_begin,
                            len(node_records),
                        )
                    )
                group_key = (output, input_type)
                group_begin = len(node_records)
            topology, binding = compacted[lane]
            child_begin = len(child_records)
            first = int(topology.child_begin[node])
            count = int(topology.executed_degree[node])
            for edge in range(first, first + count):
                kind = int(topology.target_kind[edge])
                row = int(topology.target_row[edge])
                source_output = int(binding.child_output_type[edge])
                key = (lane, kind, row, source_output)
                try:
                    encoded_source = current[key]
                except KeyError as exc:
                    raise TopologyError(
                        "layer plan encountered an unavailable child endpoint"
                    ) from exc
                consumed.add(key)
                if encoded_source >= 0:
                    source_arena = 0
                    source_row = encoded_source
                else:
                    source_arena = 1
                    source_row = -encoded_source - 1
                consumed_slots.add((lane, source_output, source_arena, source_row))
                multiplier = (
                    float(topology.nominal_degree[row])
                    if kind == int(TargetKind.TOPOLOGY)
                    else 0.0
                )
                child_records.append(
                    (
                        encoded_source,
                        source_output,
                        int(topology.path_length[edge]),
                        multiplier,
                    )
                )
            available = free_slot[lane][output][destination_arena]
            if available:
                destination = available.pop()
            else:
                destination = next_slot[lane][output][destination_arena]
                next_slot[lane][output][destination_arena] += 1
            node_records.append(
                (
                    lane,
                    child_begin,
                    count,
                    destination,
                    input_type,
                    output,
                    int(topology.owner[node]),
                    int(topology.nominal_degree[node]),
                    int(topology.target_degree[node]),
                    int(topology.sigma_head_row[node]),
                    int(binding.sigma_target_row[node]),
                    int(topology.action_begin[node]),
                )
            )
            if int(topology.parent[node]) >= 0:
                encoded_destination = (
                    destination
                    if destination_arena == 0
                    else -destination - 1
                )
                new_results[
                    (lane, int(TargetKind.TOPOLOGY), node, output)
                ] = encoded_destination
            else:
                root_destinations.append(
                    (lane, output, destination_arena, destination)
                )
        if group_key is not None:
            groups.append(
                LayerGroup(
                    layer,
                    group_key[0],
                    group_key[1],
                    group_begin,
                    len(node_records),
                )
            )

        next_current = {
            key: source for key, source in current.items() if key not in consumed
        }
        next_current.update(new_results)
        current = next_current
        # Source slots become reusable only after every CTA in this layer has
        # finished reading them.  Root results have no consumer and are also
        # released at this ordered layer boundary.
        for lane, output, arena, row in sorted(consumed_slots):
            free_slot[lane][output][arena].append(row)
        for lane, output, arena, row in root_destinations:
            free_slot[lane][output][arena].append(row)
        for lane in range(len(compacted)):
            for output in range(output_count):
                arena_max[lane][output] = max(
                    arena_max[lane][output],
                    next_slot[lane][output][0],
                    next_slot[lane][output][1],
                )
        layer_ptr.append(len(node_records))
        carry_ptr.append(len(carry_records))

    def i32_column(records: Sequence[tuple], index: int) -> Tensor:
        return torch.tensor(
            [int(record[index]) for record in records], dtype=torch.int32
        )

    def f32_column(records: Sequence[tuple], index: int) -> Tensor:
        return torch.tensor(
            [float(record[index]) for record in records], dtype=torch.float32
        )

    sigma_rows = [[0] * input_count for _ in compacted]
    action_rows = [
        [0] * (input_count if not mode.rollout_mode.is_branched else output_count)
        for _ in compacted
    ]
    sigma_seen: dict[tuple[int, int], set[int]] = {}
    action_seen: dict[tuple[int, int], set[int]] = {}
    for record in node_records:
        lane, _, _, _, input_type, output, owner, _, target_degree, _, sigma, action = (
            record
        )
        if owner < 0:
            continue
        sigma_seen.setdefault((lane, input_type), set()).add(sigma)
        action_partition = output if mode.rollout_mode.is_branched else input_type
        action_seen.setdefault((lane, action_partition), set()).update(
            range(action, action + target_degree)
        )
    for (lane, partition), rows in sigma_seen.items():
        if rows != set(range(len(rows))):
            raise TopologyError("sigma target rows are not dense and unique")
        sigma_rows[lane][partition] = len(rows)
    for (lane, partition), rows in action_seen.items():
        if rows != set(range(len(rows))):
            raise TopologyError("action target spans are not dense and unique")
        action_rows[lane][partition] = len(rows)

    plan = LayerPlan(
        layer_ptr=torch.tensor(layer_ptr, dtype=torch.int32),
        carry_ptr=torch.tensor(carry_ptr, dtype=torch.int32),
        replica=i32_column(node_records, 0),
        child_begin=i32_column(node_records, 1),
        child_count=i32_column(node_records, 2),
        destination_work_row=i32_column(node_records, 3),
        input_type=i32_column(node_records, 4),
        output_type=i32_column(node_records, 5),
        owner_slot=i32_column(node_records, 6),
        nominal_degree=i32_column(node_records, 7),
        target_degree=i32_column(node_records, 8),
        sigma_head_row=i32_column(node_records, 9),
        sigma_target_row=i32_column(node_records, 10),
        action_begin=i32_column(node_records, 11),
        child_source_work_row=i32_column(child_records, 0),
        child_source_output_type=i32_column(child_records, 1),
        child_path_length=i32_column(child_records, 2),
        child_cost_multiplier=f32_column(child_records, 3),
        carry_replica=i32_column(carry_records, 0),
        carry_source_work_row=i32_column(carry_records, 1),
        carry_destination_work_row=i32_column(carry_records, 2),
        carry_output_type=i32_column(carry_records, 3),
        movement_replica=i32_column(movement, 0),
        movement_source_leaf_row=i32_column(movement, 1),
        movement_destination_work_row=i32_column(movement, 2),
        movement_output_type=i32_column(movement, 3),
        groups=tuple(groups),
        arena_rows=tuple(tuple(values) for values in arena_max),
        sigma_rows=tuple(tuple(values) for values in sigma_rows),
        action_rows=tuple(tuple(values) for values in action_rows),
        input_names=input_names,
        output_names=output_names,
    )
    if manifest is not None:
        plan.assert_admitted(manifest)
    return plan


__all__ = [
    "LayerGroup",
    "LayerPlan",
    "TargetKind",
    "Topology",
    "TopologyBindings",
    "TopologyError",
    "TopologyPacket",
    "assemble_topology_packets",
    "build_layer_plan",
]
