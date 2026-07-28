"""Exact queued composition for the frozen Phase 8 HU 32 bb benchmark."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from pbg.games.poker.model.capacity import (
    max_policy_partition_sequence_tokens,
)
from pbg.games.poker.queued.branching import (
    BranchedPokerIteration,
    branched_rb_chance_frontier_row_bytes,
    sampled_branched_topology_packet_row_bytes,
)
from pbg.games.poker.queued.folded import (
    sequential_family_workspace_row_bytes,
)
from pbg.games.poker.queued.hooks import NLHEGeometry, rb_partitions
from pbg.games.poker.queued.model import build_model_from_config
from pbg.games.poker.queued.training import SequentialPokerIteration
from pbg.queued.benchmark import (
    FROZEN_PHASE8_SEEDS,
    SequentialBenchmarkSample,
    run_phase8_benchmark,
)
from pbg.queued.branching import (
    CostSafeguardController,
    FixedBranchController,
)
from pbg.queued.capacity import (
    CapacityManifest,
    ExternalWorkspaceBounds,
    GameGeometry,
    GeneratedOwners,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
    build_storage_plan,
)
from pbg.queued.contracts import mode_schema
from pbg.queued.cuda_layer import layered_target_cuda
from pbg.queued.learner import LearnerSchedule, LossConfig
from pbg.queued.optimization import DenseAdam, OptimizerConfig
from pbg.queued.persistence import RunState, sha256_file
from pbg.queued.rollout import LOGICAL_RNG_VERSION
from pbg.queued.targets import (
    rb_conversion_library_workspace_bytes,
    rb_conversion_workspace_row_bytes,
)


def _scientific_mapping(checkpoint: Mapping[str, object]) -> dict[str, object]:
    config = checkpoint["config"]
    model = config["model"]
    return {
        "mode": {
            "training_regime": "self_play",
            "rollout_mode": "branched.fixed",
            "private_estimator": "rao_blackwell",
        },
        "game": {
            "players": 2,
            "stack_bb": 32.0,
            "rank_count": 13,
        },
        "model": {
            name: model[name]
            for name in (
                "d_model",
                "n_heads",
                "n_layers_pre",
                "n_layers_post",
                "m_raise",
                "max_players",
                "class_embedding_dim",
                "ffn_hidden",
                "folder_beta_rank",
                "folder_line_dim",
            )
        },
    }


def _plan(
    device: torch.device,
    geometry: NLHEGeometry,
    *,
    m_raise: int,
) -> object:
    partitions = rb_partitions(geometry)
    token_bounds = max_policy_partition_sequence_tokens(
        n_players=geometry.players,
        stack_bb=geometry.stack_bb,
        rank_count=geometry.rank_count,
        k_act=5,
    )
    max_tokens = max(token_bounds.values())
    properties = torch.cuda.get_device_properties(device)
    inputs = tuple(partition.name for partition in partitions.inputs)
    manifest = CapacityManifest(
        geometry=GameGeometry(
            geometry.players,
            geometry.stack_bb,
            geometry.rank_count,
            max_tokens,
            max_tokens * max_tokens,
        ),
        root_count=1,
        n_replica=1,
        beta_max=3,
        player_node_cap=(
            PartitionRows("pre", 6_804),
            PartitionRows("post_2", 25_000),
        ),
        trained_player_cap=(
            PartitionRows("pre", 1_024),
            PartitionRows("post_2", 25_000),
        ),
        action_cap=(
            PartitionRows("pre", 1_024),
            PartitionRows("post", 35_000),
        ),
        folded_range_cap=(
            PartitionRows("pre", 1_024),
            PartitionRows("post", 25_000),
        ),
        leaf_cap=10_000,
        topology_cap=25_000,
        topology_edge_cap=35_000,
        family_pack_cap=84,
        chance_frontier_cap=6_804,
        completion_pack_cap=84,
        learner_sigma_pack_cap=256,
        learner_action_pack_cap=768,
        topology_packet_cap=10_080,
        target_arena_cap=35_000,
        movement_scratch_bytes=64 << 20,
        library_workspace_bytes=max(
            64 << 20,
            rb_conversion_library_workspace_bytes(
                geometry.private_hands,
                geometry.rank_count**2,
            ),
        ),
        evaluator_workspace_bytes=512 << 20,
        memory_guard_bytes=256 << 20,
        hardware=HardwareLimits(
            properties.total_memory,
            properties.total_memory - (384 << 20),
            max_tokens,
            max_tokens * max_tokens,
            256,
        ),
        required_external=ExternalWorkspaceBounds(0, 0),
    )
    layout = PhysicalLayout(
        model_input_shapes=tuple(
            NamedShape(
                name, (geometry.direct_model_input_width,)
            )
            for name in inputs
        ),
        player_count=geometry.players,
        stratum_count=2,
        evaluator_private_width=geometry.private_hands,
        family_workspace_row_bytes=(
            sequential_family_workspace_row_bytes(
                geometry, m_raise
            )
        ),
        chance_frontier_row_bytes=(
            branched_rb_chance_frontier_row_bytes(
                geometry, max_attempts=256
            )
        ),
        completion_workspace_row_bytes=(
            rb_conversion_workspace_row_bytes(
                geometry.private_hands,
                geometry.rank_count**2,
            )
        ),
        learner_sigma_workspace_row_bytes=128 << 10,
        learner_action_workspace_row_bytes=128 << 10,
        topology_packet_row_bytes=(
            sampled_branched_topology_packet_row_bytes(geometry)
        ),
        model_dtype=torch.float16,
    )
    return build_storage_plan(
        mode_schema(
            "self_play", "branched.fixed", "rao_blackwell"
        ),
        partitions,
        manifest,
        layout,
        device,
    )


_SEQUENTIAL_BASELINE_PATH = (
    Path(__file__).parents[4]
    / "benchmarks"
    / "phase8_sequential_rb_baseline.json"
)
_SEQUENTIAL_BASELINE_SCHEMA = "pbg.phase8.sequential-rb-baseline.v1"
_SEQUENTIAL_BASELINE_SOURCE_SHA256 = (
    "7d3961b852d1e3be1d123b0304e7488b07e87d72535da6b2769951404bb43797"
)


def _load_sequential_baseline() -> tuple[Mapping[str, object], ...]:
    """Load the immutable July sequential-RB regression evidence."""

    value = json.loads(_SEQUENTIAL_BASELINE_PATH.read_text())
    source = value.get("source")
    workload = value.get("workload")
    samples = value.get("samples")
    if (
        value.get("schema") != _SEQUENTIAL_BASELINE_SCHEMA
        or not isinstance(source, Mapping)
        or source
        != {
            "artifact": (
                "runs/experiments/"
                "nlhe_hu_32bb_rollout_diagnosis_20260724/summary.json"
            ),
            "sha256": _SEQUENTIAL_BASELINE_SOURCE_SHA256,
            "git_head": "c5cfd27a877c7e1a2aee9ebfe30723090874fe07",
            "checkpoint_sha256": (
                "854c449b3b55a158d0fe97533b75cd1ca3d98262ee31cafd"
                "95a657b6abfaaa28"
            ),
        }
        or not isinstance(workload, Mapping)
        or workload
        != {
            "game": "nlhe",
            "players": 2,
            "stack_bb": 32.0,
            "rank_count": 13,
            "mode": "sequential",
            "private_estimator": "rao_blackwell",
            "roots": 3072,
            "completion_samples": 1024,
            "warmup": 1,
            "timed_seed_offset": 104729,
            "timed_repeats": 1,
        }
        or not isinstance(samples, list)
        or not all(isinstance(sample, Mapping) for sample in samples)
        or tuple(sample["seed"] for sample in samples)
        != FROZEN_PHASE8_SEEDS
    ):
        raise ValueError("Phase 8 sequential baseline identity is invalid")
    for sample in samples:
        if (
            not isinstance(sample, Mapping)
            or int(sample.get("retained_decisions", 0)) <= 0
            or float(sample.get("rollout_seconds", 0.0)) <= 0.0
            or float(sample.get("rollout_ms_per_decision", 0.0)) <= 0.0
        ):
            raise ValueError("Phase 8 sequential baseline sample is invalid")
        derived = (
            1000.0
            * float(sample["rollout_seconds"])
            / int(sample["retained_decisions"])
        )
        if abs(derived - float(sample["rollout_ms_per_decision"])) > 1.0e-12:
            raise ValueError("Phase 8 sequential baseline timing is inconsistent")
    return tuple(samples)


def _sequential_plan(
    device: torch.device,
    geometry: NLHEGeometry,
    *,
    m_raise: int,
) -> object:
    """Right-sized storage for the matched 3,072-root sequential workload."""

    partitions = rb_partitions(geometry)
    token_bounds = max_policy_partition_sequence_tokens(
        n_players=geometry.players,
        stack_bb=geometry.stack_bb,
        rank_count=geometry.rank_count,
        k_act=5,
    )
    max_tokens = max(token_bounds.values())
    properties = torch.cuda.get_device_properties(device)
    manifest = CapacityManifest(
        geometry=GameGeometry(
            geometry.players,
            geometry.stack_bb,
            geometry.rank_count,
            max_tokens,
            max_tokens * max_tokens,
        ),
        root_count=3_072,
        n_replica=1,
        beta_max=1,
        player_node_cap=(
            PartitionRows("pre", 30_000),
            PartitionRows("post_2", 90_000),
        ),
        trained_player_cap=(
            PartitionRows("pre", 8_000),
            PartitionRows("post_2", 24_000),
        ),
        action_cap=(
            PartitionRows("pre", 8_000),
            PartitionRows("post_2", 24_000),
        ),
        folded_range_cap=(
            PartitionRows("pre", 8_000),
            PartitionRows("post", 24_000),
        ),
        leaf_cap=3_072,
        topology_cap=32_000,
        topology_edge_cap=32_000,
        family_pack_cap=84,
        chance_frontier_cap=84,
        completion_pack_cap=84,
        learner_sigma_pack_cap=256,
        learner_action_pack_cap=768,
        topology_packet_cap=10_080,
        target_arena_cap=35_072,
        movement_scratch_bytes=64 << 20,
        library_workspace_bytes=max(
            64 << 20,
            rb_conversion_library_workspace_bytes(
                geometry.private_hands,
                geometry.rank_count**2,
            ),
        ),
        evaluator_workspace_bytes=512 << 20,
        memory_guard_bytes=256 << 20,
        hardware=HardwareLimits(
            properties.total_memory,
            properties.total_memory - (384 << 20),
            max_tokens,
            max_tokens * max_tokens,
            256,
        ),
        required_external=ExternalWorkspaceBounds(0, 0),
    )
    layout = PhysicalLayout(
        model_input_shapes=tuple(
            NamedShape(
                partition.name,
                (geometry.direct_model_input_width,),
            )
            for partition in partitions.inputs
        ),
        player_count=geometry.players,
        stratum_count=2,
        evaluator_private_width=geometry.private_hands,
        family_workspace_row_bytes=sequential_family_workspace_row_bytes(
            geometry, m_raise
        ),
        chance_frontier_row_bytes=1,
        completion_workspace_row_bytes=rb_conversion_workspace_row_bytes(
            geometry.private_hands,
            geometry.rank_count**2,
        ),
        learner_sigma_workspace_row_bytes=128 << 10,
        learner_action_workspace_row_bytes=128 << 10,
        topology_packet_row_bytes=sampled_branched_topology_packet_row_bytes(
            geometry
        ),
        model_dtype=torch.float32,
    )
    return build_storage_plan(
        mode_schema("self_play", "sequential", "rao_blackwell"),
        partitions,
        manifest,
        layout,
        device,
    )


class FrozenHU32Phase8Factory:
    """Construct one exact queued run for each frozen logical seed."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.type == "cuda" and device.index is None
            else device
        )
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        # The July reference measured a profiled iteration after warmup.
        # Module/JIT load latency therefore belongs outside target timing.
        layered_target_cuda()
        self.scientific = _scientific_mapping(self.checkpoint)
        self.geometry = NLHEGeometry(2, 32.0, 13)
        m_raise = int(self.scientific["model"]["m_raise"])
        self.plan = _plan(
            self.device, self.geometry, m_raise=m_raise
        )
        self.sequential_plan = _sequential_plan(
            self.device, self.geometry, m_raise=m_raise
        )
        self.sequential_baseline = _load_sequential_baseline()
        scientific_bytes = json.dumps(
            self.scientific,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.identity = {
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "scientific_config_sha256": hashlib.sha256(
                scientific_bytes
            ).hexdigest(),
            "evaluator_version": "contract-v2",
            "rng_version": LOGICAL_RNG_VERSION,
        }

    def __call__(self, seed: int) -> BranchedPokerIteration:
        model = build_model_from_config(self.scientific, self.device)
        model.load_state_dict(self.checkpoint["model"], strict=True)
        hardware = self.checkpoint.get("hardware_profile")
        if not isinstance(hardware, Mapping):
            raise TypeError(
                "frozen Phase 8 checkpoint omits its hardware profile"
            )
        model.configure_execution(
            str(hardware["attention_backend"]),
            int(hardware["query_chunk_size"]),
        )
        controller = FixedBranchController(
            {"player": 3.0, "chance": 3.0},
            3,
            minimum_scale=1.0,
            safe_tree_size=10_000,
        )
        return BranchedPokerIteration(
            owner_factory=lambda: GeneratedOwners(self.plan),
            plan=self.plan,
            controller=controller,
            run_state=RunState(
                f"phase8-frozen-{seed}",
                root_count=1,
            ),
            arguments={
                "geometry": self.geometry,
                "mode": self.plan.mode,
                "model": model,
                "model_operation": model.rb_learner_operation,
                "rollout_operation": model.rollout_operation,
                "optimizer": DenseAdam(
                    model,
                    OptimizerConfig(
                        name="adamw",
                        learning_rate=5.0e-5,
                        beta1=0.9,
                        beta2=0.999,
                        grad_clip=0.5,
                    ),
                ),
                "loss": LossConfig(
                    clip_epsilon=0.2,
                    value_clip_epsilon=0.2,
                    value_coefficient=0.125,
                    entropy_coefficient=0.025,
                    reference_kl_coefficient=0.05,
                    value_loss="clipped",
                    entropy_schedule="linear_decay",
                    entropy_decay_iterations=500,
                    trace_coefficient=1.0,
                    cost_coefficient=1.0,
                ),
                "schedule": LearnerSchedule(
                    batch_size=256, gradient_steps=1
                ),
                "generator": torch.Generator(
                    device=self.device
                ).manual_seed(seed),
                "seed": seed,
                "lambda_": 0.95,
                "minimum_tree_size": 1_000,
                "cost_safeguard": CostSafeguardController(
                    0.1, 2.0, 8
                ),
                "precision": "fp16",
                "utility": "stack_normalized",
                "completion_samples": 1_024,
                "posterior_max_attempts": 256,
                "benchmark_telemetry": True,
                "benchmark_identity": self.identity,
                "benchmark_rollout_only": True,
            },
        )

    def measure_sequential(
        self,
    ) -> tuple[SequentialBenchmarkSample, ...]:
        """Measure the current matched sequential path for all frozen seeds."""

        def make_iteration(logical_seed: int) -> SequentialPokerIteration:
            model = build_model_from_config(self.scientific, self.device)
            model.load_state_dict(self.checkpoint["model"], strict=True)
            hardware = self.checkpoint.get("hardware_profile")
            if not isinstance(hardware, Mapping):
                raise TypeError(
                    "frozen Phase 8 checkpoint omits its hardware profile"
                )
            model.configure_execution(
                str(hardware["attention_backend"]),
                int(hardware["query_chunk_size"]),
            )
            return SequentialPokerIteration(
                geometry=self.geometry,
                owners=GeneratedOwners(self.sequential_plan),
                mode=self.sequential_plan.mode,
                model=model,
                model_operation=model.rb_learner_operation,
                rollout_operation=model.sequential_rollout_operation,
                optimizer=DenseAdam(
                    model,
                    OptimizerConfig(
                        name="adamw",
                        learning_rate=5.0e-5,
                        beta1=0.9,
                        beta2=0.999,
                        grad_clip=0.5,
                    ),
                ),
                loss=LossConfig(),
                schedule=LearnerSchedule(
                    batch_size=256, gradient_steps=1
                ),
                generator=torch.Generator(
                    device=self.device
                ).manual_seed(logical_seed),
                seed=logical_seed,
                lambda_=0.95,
                precision="fp16",
                utility="stack_normalized",
                completion_samples=1_024,
                benchmark_rollout_only=True,
            )

        result = []
        for seed, baseline in zip(
            FROZEN_PHASE8_SEEDS,
            self.sequential_baseline,
            strict=True,
        ):
            warmup_seed = seed - 104_729
            warmup = make_iteration(warmup_seed)
            warmup.run(iteration=0)
            del warmup
            iteration = make_iteration(seed)
            measured = iteration.run(iteration=0)
            if measured.telemetry is None:
                raise ValueError("sequential benchmark omitted telemetry")
            result.append(
                SequentialBenchmarkSample(
                    seed=seed,
                    retained_decisions=measured.player_nodes,
                    rollout_seconds=measured.telemetry.rollout_seconds,
                    baseline_ms_per_decision=float(
                        baseline["rollout_ms_per_decision"]
                    ),
                )
            )
        return tuple(result)


def run_frozen_hu32_phase8(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    revision: str,
    legacy_peak_allocated_bytes: int,
):
    factory = FrozenHU32Phase8Factory(checkpoint_path, device)
    sequential_samples = factory.measure_sequential()
    return run_phase8_benchmark(
        factory,
        revision=revision,
        legacy_peak_allocated_bytes=legacy_peak_allocated_bytes,
        sequential_samples=sequential_samples,
    )


__all__ = [
    "FrozenHU32Phase8Factory",
    "run_frozen_hu32_phase8",
]
