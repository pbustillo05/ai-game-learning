"""Immutable semantic contracts for all queued execution modes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Protocol

import torch


class ContractError(ValueError):
    """Raised when a queued contract is incomplete or inconsistent."""


class _ContractEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TrainingRegime(_ContractEnum):
    SELF_PLAY = "self_play"
    BEST_RESPONSE = "best_response"


class RolloutMode(_ContractEnum):
    SEQUENTIAL = "sequential"
    BRANCHED_FIXED = "branched.fixed"
    BRANCHED_KKT = "branched.KKT"

    @property
    def is_branched(self) -> bool:
        return self is not RolloutMode.SEQUENTIAL


class PrivateEstimator(_ContractEnum):
    SAMPLED = "sampled"
    RAO_BLACKWELL = "rao_blackwell"


@dataclass(frozen=True, order=True)
class Partition:
    """One game-defined numerical domain."""

    name: str
    width: int

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ContractError(f"invalid partition name {self.name!r}")
        if isinstance(self.width, bool) or self.width <= 0:
            raise ContractError(f"partition {self.name!r} has invalid width")


@dataclass(frozen=True, order=True)
class PartitionMap:
    """Map one input family to its numerical output domain."""

    input_name: str
    output_name: str


@dataclass(frozen=True, order=True)
class ConversionSpec:
    """One explicit endpoint conversion between output domains."""

    source: str
    destination: str
    operation: str

    def __post_init__(self) -> None:
        if not self.operation or not self.operation.isidentifier():
            raise ContractError(f"invalid conversion operation {self.operation!r}")


@dataclass(frozen=True)
class PartitionContract:
    """Closed game partition vocabulary supplied through an opaque hook."""

    inputs: tuple[Partition, ...]
    outputs: tuple[Partition, ...]
    folded: tuple[Partition, ...]
    input_to_output: tuple[PartitionMap, ...]
    conversions: tuple[ConversionSpec, ...]

    def __post_init__(self) -> None:
        self._require_unique("input", self.inputs)
        self._require_unique("output", self.outputs)
        self._require_unique("folded", self.folded)
        input_names = {item.name for item in self.inputs}
        output_names = {item.name for item in self.outputs}
        mapped_inputs = {item.input_name for item in self.input_to_output}
        if mapped_inputs != input_names or len(mapped_inputs) != len(self.input_to_output):
            raise ContractError("every input partition must have exactly one output")
        for item in self.input_to_output:
            if item.output_name not in output_names:
                raise ContractError(
                    f"input {item.input_name!r} maps to unknown output {item.output_name!r}"
                )
        conversion_pairs: set[tuple[str, str]] = set()
        for item in self.conversions:
            pair = (item.source, item.destination)
            if item.source not in output_names or item.destination not in output_names:
                raise ContractError(f"conversion {pair!r} uses an unknown output")
            if pair in conversion_pairs:
                raise ContractError(f"duplicate conversion {pair!r}")
            conversion_pairs.add(pair)

    @staticmethod
    def _require_unique(label: str, values: tuple[Partition, ...]) -> None:
        if not values:
            raise ContractError(f"{label} partition is empty")
        names = tuple(item.name for item in values)
        if len(set(names)) != len(names):
            raise ContractError(f"{label} partition contains duplicate names")

    def output_for(self, input_name: str) -> Partition:
        for mapping in self.input_to_output:
            if mapping.input_name == input_name:
                for output in self.outputs:
                    if output.name == mapping.output_name:
                        return output
        raise ContractError(f"unknown input partition {input_name!r}")


@dataclass(frozen=True)
class OwnerSchema:
    """One declared physical owner and its applicable fields."""

    name: str
    partition_axis: str
    fields: tuple[str, ...]
    lifetime: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ContractError(f"invalid owner name {self.name!r}")
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ContractError(f"owner {self.name!r} has empty or duplicate fields")
        if any(not field.isidentifier() for field in self.fields):
            raise ContractError(f"owner {self.name!r} has an invalid field")


@dataclass(frozen=True)
class ReplicaAxisRules:
    """State that is shared across or local to independent model lanes."""

    shared: tuple[str, ...]
    lane_local: tuple[str, ...]
    compatible_controller_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if set(self.shared).intersection(self.lane_local):
            raise ContractError("replica shared and lane-local rules overlap")


REPLICA_AXIS_RULES = ReplicaAxisRules(
    shared=(
        "mode_physical_signature",
        "game_geometry",
        "partitions",
        "capacities",
        "root_count",
        "schedule",
        "private_estimator",
    ),
    lane_local=(
        "semantic_mode",
        "weights",
        "loss_coefficients",
        "controller_state",
        "tails",
        "logical_rng",
        "failure",
        "learner_permutations",
    ),
    compatible_controller_pairs=(("branched.fixed", "branched.KKT"),),
)


@dataclass(frozen=True)
class CapacityVocabulary:
    """Closed public names required in an immutable capacity manifest."""

    partitioned_rows: tuple[str, ...]
    scalar_rows: tuple[str, ...]
    pack_limits: tuple[str, ...]
    workspaces: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            self.partitioned_rows,
            self.scalar_rows,
            self.pack_limits,
            self.workspaces,
        )
        flattened = tuple(name for group in groups for name in group)
        if any(not group for group in groups):
            raise ContractError("every capacity vocabulary group must be nonempty")
        if len(set(flattened)) != len(flattened):
            raise ContractError("capacity vocabulary contains duplicate names")
        if any(not name.isidentifier() for name in flattened):
            raise ContractError("capacity vocabulary contains an invalid name")


CAPACITY_VOCABULARY = CapacityVocabulary(
    partitioned_rows=(
        "player_node_cap",
        "trained_player_cap",
        "action_cap",
        "folded_range_cap",
    ),
    scalar_rows=("leaf_cap", "topology_cap", "topology_edge_cap"),
    pack_limits=(
        "family_pack_cap",
        "chance_frontier_cap",
        "completion_pack_cap",
        "learner_sigma_pack_cap",
        "learner_action_pack_cap",
        "topology_packet_cap",
    ),
    workspaces=(
        "target_arena_cap",
        "movement_scratch_bytes",
        "library_workspace_bytes",
        "evaluator_workspace_bytes",
        "memory_guard_bytes",
    ),
)


@dataclass(frozen=True)
class ModeSchema:
    """One immutable row of the twelve-mode product."""

    training_regime: TrainingRegime
    rollout_mode: RolloutMode
    private_estimator: PrivateEstimator
    output_domain: str
    action_partition_axis: str
    endpoint_conversion: str
    applicable_config: tuple[str, ...]
    required_hooks: tuple[str, ...]
    owners: tuple[OwnerSchema, ...]
    replica_rules: ReplicaAxisRules

    @property
    def key(self) -> str:
        return (
            f"{self.training_regime.value}/"
            f"{self.rollout_mode.value}/"
            f"{self.private_estimator.value}"
        )

    def owner(self, name: str) -> OwnerSchema:
        for owner in self.owners:
            if owner.name == name:
                return owner
        raise ContractError(f"{self.key} has no owner {name!r}")

    @property
    def physical_signature(self) -> tuple[object, ...]:
        rollout_shape = "branched" if self.rollout_mode.is_branched else "sequential"
        return (
            self.training_regime,
            rollout_shape,
            self.private_estimator,
            self.output_domain,
            self.action_partition_axis,
            tuple(
                (owner.name, owner.partition_axis, owner.fields, owner.lifetime)
                for owner in self.owners
            ),
        )

    def physically_compatible_with(self, other: "ModeSchema") -> bool:
        return self.physical_signature == other.physical_signature


def _owner(
    name: str,
    partition_axis: str,
    fields: tuple[str, ...],
    lifetime: str,
) -> OwnerSchema:
    return OwnerSchema(name, partition_axis, fields, lifetime)


def _owners(
    regime: TrainingRegime,
    rollout: RolloutMode,
    estimator: PrivateEstimator,
) -> tuple[OwnerSchema, ...]:
    branched = rollout.is_branched
    response = regime is TrainingRegime.BEST_RESPONSE
    sampled = estimator is PrivateEstimator.SAMPLED

    family_fields = [
        "logical_rng_key",
        "true_profile",
        "folded_range_row",
        "reservation_state",
        "publication_state",
        "model_input",
        "owner",
    ]
    if response:
        family_fields.append("role_selector")
    if sampled:
        family_fields.append("sampled_private_state")
    if branched:
        family_fields.extend(
            (
                "path_mass",
                "log_player_ancestor_sum",
                "log_ancestor_product",
                "branch_beta",
                "nominal_degree",
                "executed_degree",
                "target_degree",
            )
        )

    sigma_head_fields = ["old_value"]
    if branched:
        sigma_head_fields.extend(("old_trace", "old_cost"))

    leaf_fields = [
        "board_cards",
        "stratum_level",
        "payoff_baseline",
        "stratum_amount",
    ]
    if sampled:
        leaf_fields.append("combo_ids")
    else:
        leaf_fields.extend(("range_weights", "seed"))
    if response:
        leaf_fields.extend(("selected_player", "probe_cutoff"))

    layer_fields = ["return_value"] if sampled else ["mass", "moment"]
    if branched:
        layer_fields.extend(("trace_vector", "cost", "delta"))

    sigma_target_fields = ["value_target"]
    if branched:
        sigma_target_fields.extend(("trace_target", "cost_target"))

    action_target_fields = ["advantage_target"]
    if not sampled:
        action_target_fields.insert(0, "posterior")

    # Every sequential row retains the direct model input and owner in the
    # generated align-to-backward handoff.  Sampled rows additionally retain
    # their realized profile; RB rows stay in their natural output partition
    # and therefore need no sampled private state in the learner.
    learner_index_fields = ["model_input", "owner"]
    if sampled:
        learner_index_fields.insert(1, "sampled_private_state")
    if branched:
        learner_index_fields.extend(
            (
                "target_degree",
                "action_begin",
                "action_count",
                "log_sampling_weight",
            )
        )

    queue_fields = ["head", "ready_tail", "reserved_tail"]
    if branched:
        queue_fields.extend(("frontier_cost", "controller_state"))

    return (
        _owner(
            "FamilyWork",
            "input",
            tuple(family_fields),
            "root_commit_to_align_and_seal",
        ),
        _owner(
            "SigmaHead",
            "input",
            tuple(sigma_head_fields),
            "model_commit_to_final_backward",
        ),
        _owner(
            "ActionHead",
            "input" if not branched else "output",
            ("action", "old_log_policy"),
            "model_commit_to_final_backward",
        ),
        _owner("Leaf", "leaf", tuple(leaf_fields), "leaf_commit_to_completion"),
        _owner(
            "Topology",
            "structural",
            (
                "parent",
                "child_begin",
                "executed_degree",
                "nominal_degree",
                "target_degree",
                "owner",
                "familywork_row",
                "sigma_head_row",
                "action_begin",
            ),
            "topology_packet_to_layer_plan",
        ),
        _owner(
            "TopologyChild",
            "structural_edge",
            ("label", "path_length", "target_kind", "target_row"),
            "topology_packet_to_layer_plan",
        ),
        _owner(
            "FoldedRanges",
            "folded",
            ("range", "log_mass"),
            "fold_event_to_align_and_seal",
        ),
        _owner(
            "QueueMetadata",
            "input",
            tuple(queue_fields),
            "root_commit_to_queue_drain",
        ),
        _owner(
            "LayerWork",
            "output",
            tuple(layer_fields),
            "completion_to_target_publication",
        ),
        _owner(
            "SigmaTarget",
            "input",
            tuple(sigma_target_fields),
            "target_publication_to_final_backward",
        ),
        _owner(
            "ActionTarget",
            "input" if not branched else "output",
            tuple(action_target_fields),
            "target_publication_to_final_backward",
        ),
        _owner(
            "LearnerIndex",
            "input",
            tuple(learner_index_fields),
            "align_and_seal_to_final_backward",
        ),
    )


def _build_schema(
    regime: TrainingRegime,
    rollout: RolloutMode,
    estimator: PrivateEstimator,
) -> ModeSchema:
    branched = rollout.is_branched
    sampled = estimator is PrivateEstimator.SAMPLED
    response = regime is TrainingRegime.BEST_RESPONSE

    config = [
        "game_geometry",
        "root_count",
        "n_replica",
        "capacity_manifest",
        "loss",
        "optimizer",
        "precision",
        "reproducibility",
    ]
    if response:
        config.extend(("frozen_profile", "response_initialization"))
    if not sampled:
        config.extend(("completion_samples", "completion_strata"))
    if branched:
        config.extend(("beta_max", "minimum_tree_size", "safe_tree_size"))
        if rollout is RolloutMode.BRANCHED_FIXED:
            config.append("fixed_beta")
        else:
            config.append("kkt_controller")

    hooks = [
        "partitions",
        "initial_roots",
        "model_input",
        "sample",
        "transition",
        "leaf_to_layer_work",
    ]
    if response:
        hooks.append("role")
    hooks.append("folded_summary")

    return ModeSchema(
        training_regime=regime,
        rollout_mode=rollout,
        private_estimator=estimator,
        output_domain="scalar" if sampled else "private_state",
        action_partition_axis="output" if branched else "input",
        endpoint_conversion="identity" if sampled else "game_declared",
        applicable_config=tuple(config),
        required_hooks=tuple(hooks),
        owners=_owners(regime, rollout, estimator),
        replica_rules=REPLICA_AXIS_RULES,
    )


MODE_SCHEMAS = tuple(
    _build_schema(regime, rollout, estimator)
    for regime, rollout, estimator in product(
        TrainingRegime, RolloutMode, PrivateEstimator
    )
)

_SCHEMA_INDEX = {
    (schema.training_regime, schema.rollout_mode, schema.private_estimator): schema
    for schema in MODE_SCHEMAS
}


def _axis(enum_type: type[_ContractEnum], value: object, label: str) -> _ContractEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid {label} {value!r}") from exc


def mode_schema(
    training_regime: TrainingRegime | str,
    rollout_mode: RolloutMode | str,
    private_estimator: PrivateEstimator | str,
) -> ModeSchema:
    """Return the sole generated row for one valid semantic identity."""

    regime = _axis(TrainingRegime, training_regime, "training regime")
    rollout = _axis(RolloutMode, rollout_mode, "rollout mode")
    estimator = _axis(PrivateEstimator, private_estimator, "private estimator")
    return _SCHEMA_INDEX[(regime, rollout, estimator)]


def validate_replica_group(schemas: tuple[ModeSchema, ...]) -> None:
    """Require one physical schema while retaining every lane identity."""

    if not schemas:
        raise ContractError("replica group must contain at least one lane")
    first = schemas[0]
    for lane, schema in enumerate(schemas[1:], start=1):
        if not first.physically_compatible_with(schema):
            raise ContractError(
                f"replica lane {lane} ({schema.key}) is physically incompatible "
                f"with lane 0 ({first.key})"
            )


def validate_schema_on_cuda(schema: ModeSchema, device: torch.device | str) -> None:
    """Validate representative immutable metadata on the requested CUDA device."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ContractError("queued training schemas validate only on CUDA")
    if not torch.cuda.is_available():
        raise ContractError("CUDA is unavailable")
    marker = torch.tensor(
        [len(schema.owners), len(schema.applicable_config)],
        dtype=torch.int32,
        device=resolved,
    )
    if tuple(marker.cpu().tolist()) != (
        len(schema.owners),
        len(schema.applicable_config),
    ):
        raise ContractError("CUDA schema validation marker did not round-trip")


class GameHooks(Protocol):
    """Minimal opaque boundary implemented by the active game."""

    def partitions(self, schema: ModeSchema) -> PartitionContract: ...

    def initial_roots(self, request: object) -> object: ...

    def model_input(self, request: object) -> object: ...

    def sample(self, request: object) -> object: ...

    def transition(self, request: object) -> object: ...

    def folded_summary(self, request: object) -> object: ...

    def leaf_to_layer_work(self, request: object) -> object: ...

    def role(self, request: object) -> object: ...
