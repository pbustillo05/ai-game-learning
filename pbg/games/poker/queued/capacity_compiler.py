"""Deterministic compilation of a small NLHE run specification.

Normal configuration files never contain the objects produced here.  The
compiler is the sole authority for fixed storage addresses, physical shapes,
workspace bounds, and accelerator admission facts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from pbg.core.capacity import (
    BRANCHED_ALIAS_PEAK_ROW_BYTES,
    BRANCHED_LOG_MASS_ROW_BYTES,
    CapacityManifest,
    ExecutionGeometry,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
)
from pbg.core.contracts import ModeSchema, TrainingRegime
from pbg.core.storage import SoAField
from pbg.core.targets import (
    rb_conversion_library_workspace_bytes,
    rb_conversion_workspace_row_bytes,
)
from pbg.games.poker.model.capacity import max_policy_partition_sequence_tokens
from pbg.games.poker.queued.branching import (
    branched_rb_chance_frontier_row_bytes,
    branched_topology_packet_row_bytes,
    sampled_branched_chance_frontier_row_bytes,
)
from pbg.games.poker.queued.folded import sequential_family_workspace_row_bytes
from pbg.games.poker.queued.hooks import NLHEGeometry, poker_partitions
from pbg.games.poker.queued.rb_sampling import POSTERIOR_MAX_ATTEMPTS

CAPACITY_DERIVATION_VERSION = "nlhe_capacity_v3"
FIELD_ALIGNMENT = 64
_CUBLAS_WORKSPACE_BYTES = 8 << 20


@dataclass(frozen=True)
class WorkloadSpec:
    """The small workload/performance input accepted by the compiler."""

    roots: int
    replicas: int
    tree_nodes: int | None
    reserve_bytes: int
    rollout_microbatch: int | None
    completion_microbatch: int | None
    learner_microbatch: int | None
    reference_cache_bytes: int


@dataclass(frozen=True)
class CompiledCapacity:
    """One immutable resolved runtime product."""

    manifest: CapacityManifest
    layout: PhysicalLayout
    observed_total_memory_bytes: int
    derivations: Mapping[str, str]


def _align(value: int, alignment: int = 256) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _device_total_memory(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("capacity compilation requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise ValueError("capacity compilation requires available CUDA")
    return int(torch.cuda.get_device_properties(device).total_memory)


def _seat_occurrence_bound(geometry: NLHEGeometry) -> int:
    """Symmetric per-seat envelope used to prove BR trained-row capacity."""

    # A seat can act once on entry to each street and at most once per whole-bb
    # stack decrement.  This deliberately symmetric decomposition covers the
    # documented aggregate player-node bound and makes a selected seat's bound
    # independent of its label.
    bound = math.ceil(geometry.stack_bb) + 4
    if geometry.players * bound < geometry.player_node_bound:
        raise AssertionError("per-seat occurrence proof does not cover geometry")
    return bound


def _learner_workspace_row_bytes(
    geometry: NLHEGeometry,
    *,
    d_model: int,
    layers: int,
    model_dtype: torch.dtype,
    token_bound: int,
    action_width: int,
    hidden: int,
    static_width: int,
    heads: int,
    query_rows: int | None,
    training: bool = True,
) -> int:
    """Full-width live feature, saved-autograd and backward scratch envelope.

    SDPA is fused-only on CUDA; dense attention explicitly bounds query rows.
    Named terms include packed projections, residual/normalization inputs,
    both FFN hidden activations, dynamic/static adapters and FP32 incidence.
    """

    element_bytes = torch.empty((), dtype=model_dtype).element_size()
    saved_per_layer = token_bound * (18 * d_model + 2 * hidden + static_width)
    incidence_per_layer = 8 * token_bound * d_model
    saved = layers * (element_bytes * saved_per_layer + 4 * incidence_per_layer)
    working = element_bytes * token_bound * (12 * d_model + 2 * hidden + static_width)
    attention = (4 * heads * token_bound * min(query_rows, token_bound) * 4
                 if query_rows is not None else 4 * heads * token_bound * 4)
    readout = element_bytes * geometry.private_hands * (3 * hidden + (action_width + 3) * d_model)
    fp32_control = 4 * geometry.private_hands * (2 * action_width + 8)
    return _align((saved if training else 4 * incidence_per_layer)
                  + working + attention + readout + fp32_control)


def _chance_frontier_row_bytes(
    geometry: NLHEGeometry,
    *,
    rao_blackwell: bool,
) -> int:
    """Derived transient bytes for one chance-frontier row."""

    players = geometry.players
    hands = geometry.private_hands
    if rao_blackwell:
        # Posterior collision attempts plus accepted profile and legality masks.
        return _align(
            4 * (players * hands + POSTERIOR_MAX_ATTEMPTS * players)
            + POSTERIOR_MAX_ATTEMPTS * players
            + 8 * POSTERIOR_MAX_ATTEMPTS
        )
    return _align(4 * players * hands + 16 * players + 64)


def _completion_workspace_row_bytes(
    geometry: NLHEGeometry,
    *,
    rao_blackwell: bool,
) -> int:
    if not rao_blackwell:
        return _align(16 * geometry.players + 64)
    conversion = rb_conversion_workspace_row_bytes(
        geometry.private_hands, geometry.private_classes
    )
    publication = geometry.private_hands * 3 * 4
    return _align(conversion + publication)


def _evaluator_workspace_bytes(
    geometry: NLHEGeometry,
    *,
    rows: int,
    replicas: int,
    completion_samples: int,
    rao_blackwell: bool,
) -> int:
    """Bound every contract-v2 temporary and result owned during one call."""

    if not rao_blackwell:
        showdown_inputs = rows * replicas * geometry.players * (4 + 4 + 4)
        showdown_outputs = rows * replicas * geometry.players * 4
        return _align(showdown_inputs + showdown_outputs)
    batch = rows * replicas
    worlds = batch * completion_samples
    sampled_profiles = worlds * geometry.players * 4
    completed_boards = worlds * 5
    hand_ranks = worlds * geometry.players * 4
    weights_and_payoffs = worlds * (8 + 4 * geometry.players)
    results = batch * geometry.private_hands * geometry.players * 2 * 4
    return _align(
        sampled_profiles + completed_boards + hand_ranks + weights_and_payoffs + results
    )


def compile_capacity(
    *,
    mode: ModeSchema,
    geometry: NLHEGeometry,
    model: object,
    scientific: Mapping[str, object],
    precision: str,
    device: torch.device,
    workload: WorkloadSpec,
    batch_size: int,
    beta_max: int,
    completion_samples: int,
    query_rows: int | None = None,
    graph_cache_bytes: int = 0,
    family_learner_caps: Mapping[str, int] | None = None,
) -> CompiledCapacity:
    """Compile all derived storage/layout values from one canonical source."""

    total_memory = _device_total_memory(device)
    if total_memory >= 64 << 30:
        workload = replace(workload, reserve_bytes=max(workload.reserve_bytes, 10 << 30))
    if not 0 <= workload.reserve_bytes < total_memory:
        raise ValueError("memory reserve must be nonnegative and below device memory")
    if workload.roots <= 0 or workload.replicas <= 0:
        raise ValueError("workload roots and replicas must be positive")
    if beta_max <= 0:
        raise ValueError("branch degree bound must be positive")

    partitions = poker_partitions(geometry, mode)
    input_names = tuple(item.name for item in partitions.inputs)
    output_names = tuple(
        item.name
        for item in (
            partitions.inputs
            if mode.action_partition_axis == "input"
            else partitions.outputs
        )
    )
    seat_bound = _seat_occurrence_bound(geometry)
    sequential_structural = workload.roots * geometry.players * seat_bound
    if mode.rollout_mode.is_branched:
        if workload.tree_nodes is None:
            raise ValueError("branched workloads require one tree_nodes budget")
        topology_cap = workload.tree_nodes
        if topology_cap < workload.roots:
            raise ValueError("tree_nodes must cover every root")
    else:
        if workload.tree_nodes is not None:
            raise ValueError("tree_nodes is inapplicable to sequential rollout")
        topology_cap = sequential_structural

    # A beta-bounded forest with T structural vertices and Q roots has at most
    # (beta-1)T+Q terminal leaves.  These jointly imply E=(T-Q)+L.
    leaf_cap = (beta_max - 1) * topology_cap + workload.roots
    topology_edge_cap = topology_cap - workload.roots + leaf_cap
    target_arena_cap = topology_cap + leaf_cap

    player_cap = topology_cap
    player_rows = tuple(PartitionRows(name, player_cap) for name in input_names)
    if (
        mode.training_regime is TrainingRegime.SELF_PLAY
        or mode.rollout_mode.is_branched
    ):
        # The tree budget is an aggregate cap, but a branched tree can direct
        # every retained decision through the response seat.  Keep the full
        # bound until the tree builder exposes a stronger per-seat invariant.
        trained_cap = player_cap
    else:
        # Maximal-even root roles plus the symmetric seat occurrence proof.
        trained_cap = min(player_cap, workload.roots * seat_bound)
    trained_rows = tuple(PartitionRows(name, trained_cap) for name in input_names)
    if mode.rollout_mode.is_branched:
        total_trained = trained_cap * len(input_names)
        action_cap = max(beta_max, total_trained * beta_max)
        action_rows = tuple(PartitionRows(name, action_cap) for name in output_names)
    else:
        # Sequential target degree is exactly one, including row identity.
        action_rows = tuple(
            PartitionRows(item.name, item.rows) for item in trained_rows
        )

    folded_cap = topology_cap
    folded_rows = tuple(
        PartitionRows(item.name, folded_cap) for item in partitions.folded
    )

    action_width = 3 + int(model.m_raise)
    token_bounds = max_policy_partition_sequence_tokens(
        n_players=geometry.players,
        stack_bb=geometry.stack_bb,
        rank_count=geometry.rank_count,
        k_act=action_width,
    )
    token_bound = max(token_bounds.values())
    pair_bound = token_bound * token_bound

    family_rows = sum(item.rows for item in player_rows)
    trained_total = sum(item.rows for item in trained_rows)
    action_total = sum(item.rows for item in action_rows)
    auto_pack = max(1, min(batch_size, 64))
    family_pack = min(workload.rollout_microbatch or auto_pack, family_rows)
    completion_pack = min(
        workload.completion_microbatch or auto_pack,
        leaf_cap,
    )
    learner_pack = min(
        workload.learner_microbatch or batch_size,
        trained_total,
    )
    if mode.rollout_mode.is_branched:
        # One action expansion may deal flop, turn and river without another
        # player decision. Each of those three chance generations can branch.
        # Even the smallest rollout pack must accommodate one complete root.
        chance_rows_per_root = beta_max**4
        packet_rows_per_root = beta_max * sum(
            beta_max**generation for generation in range(4)
        )
        chance_pack = max(chance_rows_per_root, min(
            family_pack * chance_rows_per_root, family_rows + leaf_cap
        ))
        topology_packet = max(packet_rows_per_root, min(
            family_pack * packet_rows_per_root, topology_cap + topology_edge_cap
        ))
    else:
        chance_pack = min(family_pack * beta_max, family_rows + leaf_cap)
        topology_packet = min(
            family_pack * (beta_max + 1), topology_cap + topology_edge_cap
        )

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    rao_blackwell = mode.private_estimator.value == "rao_blackwell"
    family_workspace = sequential_family_workspace_row_bytes(
        geometry, int(model.m_raise)
    )
    chance_workspace = (
        (
            branched_rb_chance_frontier_row_bytes(
                geometry, max_attempts=POSTERIOR_MAX_ATTEMPTS
            )
            if rao_blackwell
            else sampled_branched_chance_frontier_row_bytes(geometry)
        )
        if mode.rollout_mode.is_branched
        else _chance_frontier_row_bytes(geometry, rao_blackwell=rao_blackwell)
    )
    completion_workspace = _completion_workspace_row_bytes(
        geometry,
        rao_blackwell=rao_blackwell,
    )
    def learner_row_workspace(tokens):
        return _learner_workspace_row_bytes(
            geometry,
            d_model=int(model.d_model),
            layers=int(model.layers),
            model_dtype=dtype,
            token_bound=tokens,
            action_width=action_width,
            hidden=int(model.ffn_hidden or 4 * model.d_model),
            static_width=int(model.class_embedding_dim + 3 + geometry.deck_cards
                             + 4 * model.folder_beta_rank + 2 * model.d_model),
            heads=int(model.n_heads), query_rows=query_rows,
        )
    learner_model_workspace = max(
        min(learner_pack, (family_learner_caps or {}).get(name, learner_pack))
        * learner_row_workspace(tokens) for name, tokens in token_bounds.items()
    )
    inference_workspace = _learner_workspace_row_bytes(
        geometry,
        d_model=int(model.d_model),
        layers=int(model.layers),
        model_dtype=dtype,
        token_bound=token_bound,
        action_width=action_width,
        hidden=int(model.ffn_hidden or 4 * model.d_model),
        static_width=int(model.class_embedding_dim + 3 + geometry.deck_cards
                         + 4 * model.folder_beta_rank + 2 * model.d_model),
        heads=int(model.n_heads), query_rows=query_rows, training=False,
    )
    movement_scratch = _align(
        64 * workload.replicas * (12 * topology_cap + 4 * topology_edge_cap + leaf_cap)
        + 4 * len(partitions.outputs) ** 2 * geometry.private_hands
    )
    library_workspace = graph_cache_bytes + _CUBLAS_WORKSPACE_BYTES + (
        rb_conversion_library_workspace_bytes(
            geometry.private_hands, geometry.private_classes
        )
        if rao_blackwell
        else 0
    )
    evaluator_workspace = _evaluator_workspace_bytes(
        geometry,
        rows=completion_pack,
        replicas=workload.replicas,
        completion_samples=completion_samples,
        rao_blackwell=rao_blackwell,
    )
    completion_destinations = len(partitions.outputs)
    completion_commit_workspace = _align(
        workload.replicas * completion_pack * (25 + 8 + 16 * completion_destinations)
        + workload.replicas * completion_destinations * 8
    )
    family_multiplier = (
        int(
            mode.rollout_mode.is_branched
            and mode.training_regime is TrainingRegime.BEST_RESPONSE
        )
        + 1
    )
    topology_rows = topology_packet + int(mode.rollout_mode.is_branched)
    rollout_workspace = (
        workload.replicas * family_multiplier * family_pack * family_workspace
        + workload.replicas * chance_pack * chance_workspace
        + workload.replicas
        * topology_rows
        * _align(branched_topology_packet_row_bytes(), 8)
        + workload.replicas * family_pack * inference_workspace
        + library_workspace
    )
    completion_phase_workspace = (
        workload.replicas * completion_pack * completion_workspace
        + evaluator_workspace
        + library_workspace
        + completion_commit_workspace
    )
    align_workspace = (
        movement_scratch
        + library_workspace
        + (
            topology_cap * BRANCHED_LOG_MASS_ROW_BYTES
            if mode.rollout_mode.is_branched
            else 0
        )
    )
    target_workspace = movement_scratch + library_workspace
    alias_workspace = (
        trained_total * BRANCHED_ALIAS_PEAK_ROW_BYTES
        if mode.rollout_mode.is_branched
        else 0
    )
    from pbg.games.poker.queued.model import model_storage_bytes
    parameter_bytes, _ = model_storage_bytes(scientific)
    optimizer_scratch = 16 * workload.replicas * parameter_bytes
    learner_phase_workspace = (
        workload.replicas * learner_model_workspace
        + workload.replicas * (
            learner_pack
            if not mode.rollout_mode.is_branched
            else min(action_total, learner_pack * beta_max)
        )
        * _align(4 * max(geometry.model_input_width(name) for name in input_names)
                 + 4 * geometry.private_hands * (12 * action_width + 32))
        + library_workspace
        + workload.replicas * alias_workspace
        + workload.reference_cache_bytes
        + optimizer_scratch
    )

    manifest = CapacityManifest(
        geometry=ExecutionGeometry(
            geometry.players,
            geometry.player_node_bound,
            token_bound,
            pair_bound,
        ),
        root_count=workload.roots,
        n_replica=workload.replicas,
        beta_max=beta_max,
        player_node_cap=player_rows,
        trained_player_cap=trained_rows,
        action_cap=action_rows,
        folded_range_cap=folded_rows,
        leaf_cap=leaf_cap,
        topology_cap=topology_cap,
        topology_edge_cap=topology_edge_cap,
        family_pack_cap=family_pack,
        chance_frontier_cap=chance_pack,
        completion_pack_cap=completion_pack,
        learner_sigma_pack_cap=learner_pack,
        learner_action_pack_cap=(
            learner_pack
            if not mode.rollout_mode.is_branched
            else min(action_total, learner_pack * beta_max)
        ),
        topology_packet_cap=topology_packet,
        target_arena_cap=target_arena_cap,
        rollout_workspace_bytes=rollout_workspace,
        queue_drain_workspace_bytes=library_workspace,
        completion_workspace_bytes=completion_phase_workspace,
        align_workspace_bytes=align_workspace,
        target_workspace_bytes=target_workspace,
        learner_workspace_bytes=learner_phase_workspace,
        hardware=HardwareLimits(
            total_memory_bytes=total_memory,
            usable_memory_bytes=total_memory - workload.reserve_bytes,
            reserve_bytes=workload.reserve_bytes,
        ),
    )
    layout = PhysicalLayout(
        model_input_shapes=tuple(
            NamedShape(name, (geometry.model_input_width(name),))
            for name in input_names
        ),
        player_count=geometry.players,
        stratum_count=geometry.players,
        reference_cache_bytes=workload.reference_cache_bytes,
        field_alignment=FIELD_ALIGNMENT,
        model_dtype=dtype,
        opaque_fields=(
            SoAField("leaf_public_state", torch.uint8, (5,), FIELD_ALIGNMENT, 255),
            SoAField(
                "leaf_private_profile",
                torch.int32,
                (geometry.players,),
                FIELD_ALIGNMENT,
                -1,
            ),
        ),
    )
    return CompiledCapacity(
        manifest,
        layout,
        total_memory,
        MappingProxyType(
            {
                "schema": CAPACITY_DERIVATION_VERSION,
                "geometry": "NLHEGeometry",
                "tokens": "max_policy_partition_sequence_tokens",
                "forest": "beta_bounded_forest_v1",
                "best_response": "branched_full_or_sequential_symmetric_bound_v1",
                "workspace": "full_width_cohort_saved_autograd_v2",
                "device_memory": "torch.cuda.get_device_properties",
            }
        ),
    )


__all__ = [
    "CAPACITY_DERIVATION_VERSION",
    "CompiledCapacity",
    "WorkloadSpec",
    "compile_capacity",
]
