"""Compact CPU topology and exact reverse-layer planning.

The topology is intentionally numerical and game-opaque.  Rollout packets use
absolute structural addresses; sealing compacts headless unary rows and emits
one immutable movement/layer plan for the CUDA target executor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

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
        _cpu_vector("sigma_target_row", self.sigma_target_row, nodes, torch.int32)
        _cpu_vector("child_output_type", self.child_output_type, edges, torch.int32)


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
        _allow_zero_path: bool = False,
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
        edge_owner = [-1] * e
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
                if (
                    min(
                        family[node],
                        sigma[node],
                        action[node],
                        input_type[node],
                        sigma_target[node],
                    )
                    < 0
                ):
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
                if edge_owner[edge] != -1:
                    raise TopologyError("topology child spans overlap")
                edge_owner[edge] = node
                if path_length[edge] < (0 if _allow_zero_path else 1):
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
        if any(owner < 0 for owner in edge_owner):
            raise TopologyError("topology child spans leave orphan edges")

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
        self,
        bindings: TopologyBindings,
        *,
        p_train: int,
        _allow_zero_path: bool = False,
    ) -> tuple[Topology, TopologyBindings]:
        """Delete headless ordinary unary rows and summarize their paths."""

        self.validate(
            bindings,
            p_train=p_train,
            _allow_zero_path=_allow_zero_path,
        )
        n = self.node_count
        if n == 0:
            return self, bindings
        degree = self.executed_degree.tolist()
        nominal = self.nominal_degree.tolist()
        owner = self.owner.tolist()
        keep = [
            owner[row] >= 0 or degree[row] >= 2 or nominal[row] >= 2 for row in range(n)
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
        self.bindings.validate(self.topology.node_count, self.topology.edge_count)


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


@dataclass(frozen=True)
class LayerPlan:
    """Exact CPU movement and reverse-layer execution plan."""

    layer_ptr: Tensor
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
    movement_replica: Tensor
    movement_source_leaf_row: Tensor
    movement_destination_work_row: Tensor
    movement_output_type: Tensor
    arena_rows: tuple[tuple[int, ...], ...]
    sigma_rows: tuple[tuple[int, ...], ...]
    action_rows: tuple[tuple[int, ...], ...]
    leaf_rows: tuple[int, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    output_widths: tuple[int, ...]
    supported_conversions: tuple[tuple[int, int], ...]
    rb_first_source_components: int
    rb_owner_source_components: int
    rb_written_components: int
    target_edge_count: int
    movement_index_bytes: int
    branched: bool

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
    def movement_count(self) -> int:
        return int(self.movement_replica.numel())

    @property
    def nbytes(self) -> int:
        tensors = (
            self.layer_ptr,
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
            self.movement_replica,
            self.movement_source_leaf_row,
            self.movement_destination_work_row,
            self.movement_output_type,
        )
        return sum(value.numel() * value.element_size() for value in tensors)

    def assert_admitted(self, manifest: CapacityManifest) -> None:
        i32_node_fields = (
            "replica",
            "child_begin",
            "child_count",
            "destination_work_row",
            "input_type",
            "output_type",
            "owner_slot",
            "nominal_degree",
            "target_degree",
            "sigma_head_row",
            "sigma_target_row",
            "action_begin",
        )
        i32_edge_fields = (
            "child_source_work_row",
            "child_source_output_type",
            "child_path_length",
        )
        for name in i32_node_fields:
            _cpu_vector(name, getattr(self, name), self.node_count, torch.int32)
        for name in i32_edge_fields:
            _cpu_vector(name, getattr(self, name), self.edge_count, torch.int32)
        _cpu_vector(
            "child_cost_multiplier",
            self.child_cost_multiplier,
            self.edge_count,
            torch.float32,
        )
        _cpu_vector("layer_ptr", self.layer_ptr, self.layer_count + 1, torch.int32)
        layer_ptr = self.layer_ptr.tolist()
        if (
            not layer_ptr
            or layer_ptr[0] != 0
            or layer_ptr[-1] != self.node_count
            or any(left > right for left, right in zip(layer_ptr, layer_ptr[1:]))
        ):
            raise TopologyError("LayerPlan layer pointers are invalid")
        movement_fields = (
            "movement_replica",
            "movement_source_leaf_row",
            "movement_destination_work_row",
            "movement_output_type",
        )
        for name in movement_fields:
            _cpu_vector(name, getattr(self, name), self.movement_count, torch.int32)
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
        n_replica = manifest.n_replica
        input_count = len(self.input_names)
        output_count = len(self.output_names)
        if len(self.output_widths) != output_count or any(
            width <= 0 for width in self.output_widths
        ):
            raise TopologyError("LayerPlan output widths are invalid")
        trained_caps = manifest.partition_rows("trained_player_cap")
        action_caps = manifest.partition_rows("action_cap")
        if len(self.arena_rows) != n_replica:
            raise TopologyError("LayerPlan arena replica axis is invalid")
        if len(self.sigma_rows) != n_replica or len(self.action_rows) != n_replica:
            raise TopologyError("LayerPlan target replica axis is invalid")
        if len(self.leaf_rows) != n_replica:
            raise TopologyError("LayerPlan leaf replica axis is invalid")
        if any(not 0 <= rows <= manifest.leaf_cap for rows in self.leaf_rows):
            raise TopologyError("LayerPlan leaf extent is out of bounds")
        if any(len(rows) != output_count for rows in self.arena_rows):
            raise TopologyError("LayerPlan arena partition axis is invalid")
        if any(len(rows) != input_count for rows in self.sigma_rows):
            raise TopologyError("LayerPlan sigma partition axis is invalid")
        action_count = output_count if self.branched else input_count
        if any(len(rows) != action_count for rows in self.action_rows):
            raise TopologyError("LayerPlan action partition axis is invalid")
        if len(set(self.supported_conversions)) != len(
            self.supported_conversions
        ) or any(
            source == destination
            or not 0 <= source < output_count
            or not 0 <= destination < output_count
            for source, destination in self.supported_conversions
        ):
            raise TopologyError("LayerPlan conversion vocabulary is invalid")
        child_lane = [-1] * self.edge_count
        child_owner = [-1] * self.edge_count
        sigma_destinations: set[tuple[int, int, int]] = set()
        action_destinations: set[tuple[int, int, int]] = set()
        for first, stop in zip(layer_ptr, layer_ptr[1:]):
            destinations: set[tuple[int, int, int]] = set()
            for row in range(first, stop):
                key = (
                    int(self.replica[row]),
                    int(self.output_type[row]),
                    int(self.destination_work_row[row]),
                )
                if key in destinations:
                    raise TopologyError("LayerPlan layer destinations are not distinct")
                destinations.add(key)
        for row in range(self.node_count):
            lane = int(self.replica[row])
            output = int(self.output_type[row])
            destination = int(self.destination_work_row[row])
            if not 0 <= lane < n_replica or not 0 <= output < output_count:
                raise TopologyError("LayerPlan node lane/output address is invalid")
            if not 0 <= destination < self.arena_rows[lane][output]:
                raise TopologyError("LayerPlan node destination is outside its arena")
            begin = int(self.child_begin[row])
            count = int(self.child_count[row])
            target_count = int(self.target_degree[row])
            nominal = int(self.nominal_degree[row])
            refilled_root = target_count < count
            if (
                begin < 0
                or begin + count > self.edge_count
                or not 1 <= target_count <= count
                or target_count > manifest.beta_max
                or (count > manifest.beta_max and not refilled_root)
                or not 1 <= nominal <= manifest.beta_max
                or (
                    not self.branched
                    and (count != 1 or target_count != 1 or nominal != 1)
                )
            ):
                raise TopologyError("LayerPlan child span is out of bounds")
            for edge in range(begin, begin + count):
                if child_lane[edge] != -1:
                    raise TopologyError("LayerPlan child spans overlap")
                child_lane[edge] = lane
                child_owner[edge] = row
            input_type = int(self.input_type[row])
            owner = int(self.owner_slot[row])
            if input_type < 0 or owner < 0:
                if (
                    input_type != -1
                    or owner != -1
                    or min(
                        int(self.sigma_head_row[row]),
                        int(self.sigma_target_row[row]),
                        int(self.action_begin[row]),
                    )
                    != -1
                ):
                    raise TopologyError("LayerPlan structural row addresses targets")
                continue
            if input_type >= input_count or owner >= manifest.geometry.players:
                raise TopologyError("LayerPlan input type is out of bounds")
            input_name = self.input_names[input_type]
            if not 0 <= int(self.sigma_head_row[row]) < trained_caps[input_name]:
                raise TopologyError("LayerPlan sigma head address is out of bounds")
            if not 0 <= int(self.sigma_target_row[row]) < trained_caps[input_name]:
                raise TopologyError("LayerPlan sigma target address is out of bounds")
            if int(self.sigma_target_row[row]) >= self.sigma_rows[lane][input_type]:
                raise TopologyError(
                    "LayerPlan sigma target exceeds its realized reservation"
                )
            if not self.branched and (
                int(self.sigma_head_row[row]) != int(self.sigma_target_row[row])
                or int(self.action_begin[row]) != int(self.sigma_target_row[row])
            ):
                raise TopologyError(
                    "sequential head and target rows must share one identity"
                )
            sigma_key = (
                lane,
                input_type,
                int(self.sigma_target_row[row]),
            )
            if sigma_key in sigma_destinations:
                raise TopologyError("LayerPlan sigma targets are not unique")
            sigma_destinations.add(sigma_key)
            action_type = output if self.branched else input_type
            action_name = (
                self.output_names[action_type]
                if self.branched
                else self.input_names[action_type]
            )
            action_stop = int(self.action_begin[row]) + int(self.target_degree[row])
            if (
                int(self.action_begin[row]) < 0
                or action_stop > action_caps[action_name]
            ):
                raise TopologyError("LayerPlan action target span is out of bounds")
            if action_stop > self.action_rows[lane][action_type]:
                raise TopologyError(
                    "LayerPlan action target exceeds its realized reservation"
                )
            for action_row in range(int(self.action_begin[row]), action_stop):
                action_key = (lane, action_type, action_row)
                if action_key in action_destinations:
                    raise TopologyError("LayerPlan action targets overlap")
                action_destinations.add(action_key)
        if any(lane < 0 for lane in child_lane):
            raise TopologyError("LayerPlan child spans do not cover every edge")
        for edge in range(self.edge_count):
            output = int(self.child_source_output_type[edge])
            encoded = int(self.child_source_work_row[edge])
            if not 0 <= output < output_count:
                raise TopologyError("LayerPlan child output type is out of bounds")
            local_row = encoded if encoded >= 0 else -encoded - 1
            lane = child_lane[edge]
            if not 0 <= local_row < self.arena_rows[lane][output]:
                raise TopologyError("LayerPlan child source is outside its arena")
            owner_row = child_owner[edge]
            destination = int(self.output_type[owner_row])
            if (
                output != destination
                and (output, destination) not in self.supported_conversions
            ):
                raise TopologyError("LayerPlan uses an unsupported conversion")
            if int(self.child_path_length[edge]) < 1:
                raise TopologyError("LayerPlan child path length is invalid")
            if not bool(torch.isfinite(self.child_cost_multiplier[edge])):
                raise TopologyError("LayerPlan child cost multiplier is invalid")
        first_components = 0
        owner_components = 0
        written_components = 0
        for row in range(self.node_count):
            begin = int(self.child_begin[row])
            target_count = int(self.target_degree[row])
            source_components = sum(
                self.output_widths[int(self.child_source_output_type[edge])]
                for edge in range(begin, begin + target_count)
            )
            first_components += source_components
            if int(self.owner_slot[row]) >= 0:
                owner_components += target_count * source_components
            written_components += self.output_widths[int(self.output_type[row])]
        if (
            self.rb_first_source_components != first_components
            or self.rb_owner_source_components != owner_components
            or self.rb_written_components != written_components
        ):
            raise TopologyError("LayerPlan traffic summary is inconsistent")
        if self.target_edge_count != sum(int(value) for value in self.target_degree):
            raise TopologyError("LayerPlan target-edge summary is inconsistent")
        movement_destinations: set[tuple[int, int, int]] = set()
        movement_sources: list[set[int]] = [set() for _ in range(n_replica)]
        group_counts: dict[tuple[int, int], int] = {}
        for movement in range(self.movement_count):
            lane = int(self.movement_replica[movement])
            output = int(self.movement_output_type[movement])
            source = int(self.movement_source_leaf_row[movement])
            destination = int(self.movement_destination_work_row[movement])
            if (
                not 0 <= lane < n_replica
                or not 0 <= output < output_count
                or not 0 <= source < manifest.leaf_cap
                or not 0 <= destination < self.arena_rows[lane][output]
            ):
                raise TopologyError("LayerPlan movement address is out of bounds")
            key = (lane, output, destination)
            if key in movement_destinations:
                raise TopologyError("LayerPlan movement destinations overlap")
            movement_destinations.add(key)
            if source in movement_sources[lane]:
                raise TopologyError("LayerPlan leaf sources are not unique")
            movement_sources[lane].add(source)
            group = (lane, output)
            expected_destination = group_counts.get(group, 0)
            if destination != expected_destination:
                raise TopologyError("LayerPlan movement destinations are not dense")
            group_counts[group] = expected_destination + 1
            if source >= self.leaf_rows[lane]:
                raise TopologyError("LayerPlan movement exceeds sealed leaf rows")
        expected_index_bytes = max(group_counts.values(), default=0) * 8
        if self.movement_index_bytes != expected_index_bytes:
            raise TopologyError("LayerPlan movement scratch summary is inconsistent")
        if self.movement_index_bytes > manifest.movement_scratch_bytes:
            raise TopologyError("LayerPlan movement indices exceed admitted scratch")
        for lane, sources in enumerate(movement_sources):
            if sources != set(range(self.leaf_rows[lane])):
                raise TopologyError(
                    "LayerPlan leaf movement is not an exact permutation"
                )
        for lane in range(n_replica):
            for input_type, name in enumerate(self.input_names):
                rows = self.sigma_rows[lane][input_type]
                if rows < 0 or rows > trained_caps[name]:
                    raise TopologyError("realized sigma rows exceed trained capacity")
                actual = {
                    row
                    for actual_lane, actual_type, row in sigma_destinations
                    if actual_lane == lane and actual_type == input_type
                }
                if actual != set(range(rows)):
                    raise TopologyError(
                        "LayerPlan sigma targets do not match their exact extent"
                    )
            action_names = self.output_names if self.branched else self.input_names
            for action_type, name in enumerate(action_names):
                rows = self.action_rows[lane][action_type]
                if rows < 0 or rows > action_caps[name]:
                    raise TopologyError("realized action rows exceed action capacity")
                actual = {
                    row
                    for actual_lane, actual_type, row in action_destinations
                    if actual_lane == lane and actual_type == action_type
                }
                if actual != set(range(rows)):
                    raise TopologyError(
                        "LayerPlan action targets do not match their exact extent"
                    )


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
        for value in (*topology.executed_degree, *topology.nominal_degree)
    ):
        raise TopologyError("sequential topology requires N=N_tilde=1 at every row")
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
    supported_conversions = {
        (
            output_names.index(item.source),
            output_names.index(item.destination),
        )
        for item in partitions.conversions
    }
    output_widths = tuple(item.width for item in partitions.outputs)
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
        if manifest is not None:
            player_caps = manifest.partition_rows("player_node_cap")
            for node in range(topology.node_count):
                input_type = int(binding.input_type[node])
                if input_type < 0:
                    continue
                name = input_names[input_type]
                family_row = int(topology.familywork_row[node])
                if not 0 <= family_row < player_caps[name]:
                    raise TopologyError("topology family-work address is out of bounds")
        if mode.rollout_mode is RolloutMode.SEQUENTIAL:
            for node in range(topology.node_count):
                if int(topology.owner[node]) < 0:
                    continue
                aligned = (
                    int(topology.familywork_row[node]),
                    int(topology.sigma_head_row[node]),
                    int(binding.sigma_target_row[node]),
                    int(topology.action_begin[node]),
                )
                if len(set(aligned)) != 1:
                    raise TopologyError(
                        "sequential learner rows must share one identity"
                    )
        for node in range(topology.node_count):
            destination = int(binding.output_type[node])
            begin = int(topology.child_begin[node])
            count = int(topology.executed_degree[node])
            for edge in range(begin, begin + count):
                source = int(binding.child_output_type[edge])
                if (
                    source != destination
                    and (source, destination) not in supported_conversions
                ):
                    raise TopologyError("topology uses an unsupported conversion")

    depths = [_depths(topology) for topology, _ in compacted]
    layer_count = max((max(value, default=0) for value in depths), default=0)
    # A nonnegative source slot addresses arena zero.  A negative slot ``-r-1``
    # addresses row ``r`` in arena one.  This preserves the fixed LayerPlan
    # child schema while allowing the selected last-consumer plan to retain
    # shorter branches in their producing arena without physical carries.
    current: dict[tuple[int, int, int, int], int] = {}
    movement: list[tuple[int, int, int, int]] = []
    leaf_source_rows: list[set[int]] = [set() for _ in compacted]
    arena_max = [[0] * output_count for _ in compacted]
    next_slot = [[[0, 0] for _ in range(output_count)] for _ in compacted]
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
            if leaf in leaf_source_rows[lane]:
                raise TopologyError("a leaf row is referenced more than once")
            leaf_source_rows[lane].add(leaf)
            key = (lane, int(TargetKind.LEAF), leaf, output)
            if key in current:
                raise TopologyError("a leaf endpoint is referenced more than once")
            row = next_slot[lane][output][0]
            next_slot[lane][output][0] += 1
            current[key] = row
            movement.append((lane, leaf, row, output))
        for output in range(output_count):
            arena_max[lane][output] = next_slot[lane][output][0]
        if leaf_source_rows[lane] != set(range(len(leaf_source_rows[lane]))):
            raise TopologyError("leaf rows must form one dense exact permutation")

    node_records: list[tuple[int, ...]] = []
    child_records: list[tuple[int, int, int, float]] = []
    layer_ptr = [0]

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
        destination_arena = (layer + 1) & 1

        for output, input_type, lane, node in selected:
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
                    destination if destination_arena == 0 else -destination - 1
                )
                new_results[(lane, int(TargetKind.TOPOLOGY), node, output)] = (
                    encoded_destination
                )
            else:
                root_destinations.append((lane, output, destination_arena, destination))
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
    leaf_rows = [0] * len(compacted)
    movement_group_counts: dict[tuple[int, int], int] = {}
    for lane, leaf, _, output in movement:
        leaf_rows[lane] = max(leaf_rows[lane], leaf + 1)
        key = (lane, output)
        movement_group_counts[key] = movement_group_counts.get(key, 0) + 1
    sigma_seen: dict[tuple[int, int], set[int]] = {}
    action_seen: dict[tuple[int, int], set[int]] = {}
    for record in node_records:
        lane, _, _, _, input_type, output, owner, _, target_degree, _, sigma, action = (
            record
        )
        if owner < 0:
            continue
        sigma_key = (lane, input_type)
        sigma_partition_rows = sigma_seen.setdefault(sigma_key, set())
        if sigma in sigma_partition_rows:
            raise TopologyError("sigma target rows must be unique")
        sigma_partition_rows.add(sigma)
        action_partition = output if mode.rollout_mode.is_branched else input_type
        action_key = (lane, action_partition)
        action_partition_rows = action_seen.setdefault(action_key, set())
        action_span = set(range(action, action + target_degree))
        if action_partition_rows & action_span:
            raise TopologyError("action target spans must not overlap")
        action_partition_rows.update(action_span)
    rb_first_source_components = 0
    rb_owner_source_components = 0
    rb_written_components = 0
    for record in node_records:
        (
            _,
            child_begin,
            _,
            _,
            _,
            output,
            owner,
            _,
            target_degree,
            *_,
        ) = record
        source_components = sum(
            output_widths[int(child_records[edge][1])]
            for edge in range(child_begin, child_begin + target_degree)
        )
        rb_first_source_components += source_components
        if owner >= 0:
            rb_owner_source_components += target_degree * source_components
        rb_written_components += output_widths[output]
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
        movement_replica=i32_column(movement, 0),
        movement_source_leaf_row=i32_column(movement, 1),
        movement_destination_work_row=i32_column(movement, 2),
        movement_output_type=i32_column(movement, 3),
        arena_rows=tuple(tuple(values) for values in arena_max),
        sigma_rows=tuple(tuple(values) for values in sigma_rows),
        action_rows=tuple(tuple(values) for values in action_rows),
        leaf_rows=tuple(leaf_rows),
        input_names=input_names,
        output_names=output_names,
        output_widths=output_widths,
        supported_conversions=tuple(sorted(supported_conversions)),
        rb_first_source_components=rb_first_source_components,
        rb_owner_source_components=rb_owner_source_components,
        rb_written_components=rb_written_components,
        target_edge_count=sum(int(record[8]) for record in node_records),
        movement_index_bytes=max(movement_group_counts.values(), default=0) * 8,
        branched=mode.rollout_mode.is_branched,
    )
    if manifest is not None:
        plan.assert_admitted(manifest)
    return plan


__all__ = [
    "LayerPlan",
    "TargetKind",
    "Topology",
    "TopologyBindings",
    "TopologyError",
    "TopologyPacket",
    "assemble_topology_packets",
    "build_layer_plan",
]
