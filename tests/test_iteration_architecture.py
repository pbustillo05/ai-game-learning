from __future__ import annotations

import ast
import weakref
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from pbg.core.capacity import (
    CapacityManifest,
    ExecutionGeometry,
    ExecutionPhase,
    ExternalWorkspaceBounds,
    GeneratedOwners,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
    build_storage_plan,
)
from pbg.core.contracts import (
    Partition,
    PartitionContract,
    PartitionMap,
    mode_schema,
)
from pbg.core.iteration import (
    GameCompletion,
    GameRollout,
    IterationBatchResult,
    IterationEngine,
    RootDestination,
    RootPayload,
)
from pbg.core.rollout import (
    CommitCoverage,
    CommitPlan,
    CompletionPlan,
    SampleWorkspace,
    TopologyPublication,
)
from pbg.core.storage import DeviceDestinationRequest, SoAField
from pbg.core.targets import TargetAlgebra
from pbg.core.topology import TopologyExpansionRecord

REPOSITORY = Path(__file__).resolve().parents[1]
CORE = REPOSITORY / "pbg/core"
POKER = REPOSITORY / "pbg/games/poker/queued"
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="generic rollout requires CUDA"
)


class _Owners:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.plan = SimpleNamespace(manifest=SimpleNamespace(n_replica=1))
        self.failure_state = SimpleNamespace(snapshot=lambda _lane: None)

    def enter_phase(self, phase: ExecutionPhase) -> None:
        self.events.append(("enter", phase.name))

    def release_after(self, phase: ExecutionPhase) -> None:
        self.events.append(("release", phase.name))

    def close(self) -> None:
        self.events.append(("close", "owners"))


class _Agreement:
    def agree(self, failure):
        return failure


class _Optimizer:
    def __init__(self, events) -> None:
        self.events = events

    def begin_transaction(self) -> None:
        self.events.append(("semantic", "optimizer_begin"))

    def step(self) -> None:
        self.events.append(("semantic", "optimizer_step"))

    def commit_transaction(self) -> None:
        self.events.append(("semantic", "optimizer_commit"))


class _NonPokerOperations:
    """A synthetic game with no cards, streets, folds, or poker imports."""

    def __init__(self, events) -> None:
        self.events = events

    def root_destination(self, _plan):
        return RootDestination("origin")

    def root_payload(self, _request):
        self.events.append(("semantic", "root_values"))

    def rollout(self, _request):
        self.events.extend(
            (
                ("semantic", "model"),
                ("semantic", "sample"),
                ("semantic", "transition"),
            )
        )

    def completion(self, _request):
        self.events.append(("semantic", "completion_binding"))

    def target_algebra(self, _plan):
        self.events.append(("semantic", "target_algebra"))

    def learner_policy(self, _lane):
        self.events.append(("semantic", "learner_policy"))

    def observability(self, _lane):
        self.events.append(("semantic", "observability"))
        return MappingProxyType({})


class _SyntheticEngine(IterationEngine):
    def _execute_rollout(self, _owners, *, iteration: int):
        assert iteration == 7
        self.operations.root_payload(None)
        self.operations.rollout(None)
        return object()

    def _execute_completion(self, _owners, _state) -> None:
        self.operations.completion(None)

    def _seal_learner_inputs(self, _owners, _state) -> None:
        self.operations.events.append(("semantic", "seal"))

    def _execute_targets(self, _owners, _state) -> None:
        self.operations.target_algebra(None)
        self.operations.events.append(("semantic", "targets"))

    def _execute_learner(self, _owners, _state, *, iteration: int) -> None:
        assert iteration == 7
        self.operations.learner_policy(0)
        self.optimizer.begin_transaction()
        self.optimizer.step()
        self.optimizer.commit_transaction()

    def _build_result(self, _owners, _state):
        self.operations.observability(None)
        return IterationBatchResult((), MappingProxyType({}), ())


def test_non_poker_bundle_reaches_one_optimizer_transition(
    monkeypatch,
) -> None:
    monkeypatch.setattr("torch.cuda.synchronize", lambda _device=None: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    owners = _Owners()
    owners.plan.device = "cuda"
    operations = _NonPokerOperations(owners.events)
    engine = object.__new__(_SyntheticEngine)
    engine.operations = operations
    engine.optimizer = _Optimizer(owners.events)
    engine.rank_agreement = _Agreement()
    result = engine._run_once(owners, iteration=7)
    assert isinstance(result, IterationBatchResult)
    assert owners.events == [
        ("enter", "ROLLOUT"),
        ("semantic", "root_values"),
        ("semantic", "model"),
        ("semantic", "sample"),
        ("semantic", "transition"),
        ("release", "ROLLOUT"),
        ("enter", "QUEUE_DRAIN"),
        ("release", "QUEUE_DRAIN"),
        ("enter", "COMPLETION"),
        ("semantic", "completion_binding"),
        ("release", "COMPLETION"),
        ("enter", "ALIGN_AND_SEAL"),
        ("semantic", "seal"),
        ("release", "ALIGN_AND_SEAL"),
        ("enter", "TARGETS"),
        ("semantic", "target_algebra"),
        ("semantic", "targets"),
        ("release", "TARGETS"),
        ("enter", "LEARNER"),
        ("semantic", "learner_policy"),
        ("semantic", "optimizer_begin"),
        ("semantic", "optimizer_step"),
        ("semantic", "optimizer_commit"),
        ("release", "LEARNER"),
        ("enter", "FINAL_BACKWARD"),
        ("release", "FINAL_BACKWARD"),
        ("semantic", "observability"),
        ("close", "owners"),
    ]


def _neutral_plan(device: torch.device):
    mode = mode_schema("self_play", "sequential", "sampled")
    partitions = PartitionContract(
        inputs=(Partition("origin", 1),),
        outputs=(Partition("score", 1),),
        folded=(Partition("memory", 1),),
        input_to_output=(PartitionMap("origin", "score"),),
        conversions=(),
    )
    manifest = CapacityManifest(
        geometry=ExecutionGeometry(
            actor_count=1,
            occurrence_bound=4,
            model_token_bound=4,
            model_token_pair_bound=16,
        ),
        root_count=1,
        n_replica=1,
        beta_max=1,
        player_node_cap=(PartitionRows("origin", 4),),
        trained_player_cap=(PartitionRows("origin", 4),),
        action_cap=(PartitionRows("origin", 4),),
        folded_range_cap=(PartitionRows("memory", 4),),
        leaf_cap=4,
        topology_cap=4,
        topology_edge_cap=4,
        family_pack_cap=2,
        chance_frontier_cap=2,
        completion_pack_cap=2,
        learner_sigma_pack_cap=2,
        learner_action_pack_cap=2,
        topology_packet_cap=4,
        target_arena_cap=8,
        movement_scratch_bytes=4096,
        library_workspace_bytes=4096,
        evaluator_workspace_bytes=4096,
        memory_guard_bytes=4096,
        hardware=HardwareLimits(
            total_memory_bytes=torch.cuda.get_device_properties(
                device
            ).total_memory,
            memory_budget_bytes=64 * 1024 * 1024,
            training_max_tokens=16,
            training_max_token_pairs=256,
            query_chunk_size=4,
        ),
        required_external=ExternalWorkspaceBounds(
            library_bytes=0,
            evaluator_bytes=0,
        ),
    )
    layout = PhysicalLayout(
        model_input_shapes=(NamedShape("origin", (2,)),),
        player_count=1,
        stratum_count=1,
        evaluator_private_width=1,
        family_workspace_row_bytes=256,
        chance_frontier_row_bytes=256,
        completion_workspace_row_bytes=256,
        learner_sigma_workspace_row_bytes=256,
        learner_action_workspace_row_bytes=256,
        topology_packet_row_bytes=256,
        opaque_fields=(
            SoAField("leaf_public_state", torch.float32, (1,)),
            SoAField("leaf_private_profile", torch.int32, (1,), fill=-1),
        ),
    )
    return build_storage_plan(mode, partitions, manifest, layout, device)


def _unit_coverage(device: torch.device, rows: int) -> CommitCoverage:
    ordinal = torch.arange(rows, dtype=torch.int64, device=device)[None]
    return CommitCoverage(
        torch.ones((1, rows), dtype=torch.bool, device=device),
        ordinal,
        torch.zeros_like(ordinal),
        ordinal,
    )


class _NeutralRolloutHooks:
    def __init__(self, request) -> None:
        self.leaf = request.owners.slab("Leaf")
        self.sigma = request.owners.slab("SigmaHead", "origin")
        self.action = request.owners.slab("ActionHead", "origin")
        self.stager = request.stagers[0]
        self.sequence = 0

    def model_input(self, batch):
        return batch.fields["model_input"]

    def sample(self, request):
        return SampleWorkspace(
            MappingProxyType({}),
            torch.full(
                (request.batch.lane_count,),
                request.batch.rows,
                dtype=torch.int64,
                device=request.batch.row.device,
            ),
        )

    def transition(self, request):
        batch = request.batch
        device = batch.row.device
        rows = batch.rows
        destinations = (self.leaf, self.sigma, self.action)
        destination_values = tuple(
            {
                field.name: torch.full(
                    (1, rows, *field.shape),
                    field.fill,
                    dtype=field.dtype,
                    device=device,
                )
                for field in slab.schema.fields
            }
            for slab in destinations
        )
        destination_values[2]["action"].zero_()
        counts = torch.tensor([rows], dtype=torch.int64, device=device)
        ordinal = torch.arange(rows, dtype=torch.int64, device=device)
        logical = batch.fields["logical_rng_key"][0]
        one = torch.ones(rows, dtype=torch.int32, device=device)
        record = TopologyExpansionRecord(
            node_active=torch.ones(rows, dtype=torch.bool, device=device),
            edge_active=torch.ones(rows, dtype=torch.bool, device=device),
            parent_key=logical,
            child_key=torch.stack((logical[:, 0], logical[:, 1] + 1), dim=1),
            source_ordinal=ordinal,
            terminal=torch.ones(rows, dtype=torch.bool, device=device),
            child_input=torch.zeros(rows, dtype=torch.int32, device=device),
            child_leaf_row=(
                self.leaf.reserved_tails_device()[0] + ordinal
            ),
            label=torch.zeros(rows, dtype=torch.int32, device=device),
            owner=torch.zeros(rows, dtype=torch.int32, device=device),
            input_type=torch.zeros(rows, dtype=torch.int32, device=device),
            familywork_row=batch.row[0].to(torch.int32),
            sigma_head_row=(
                self.sigma.reserved_tails_device()[0] + ordinal
            ).to(torch.int32),
            action_begin=(
                self.action.reserved_tails_device()[0] + ordinal
            ).to(torch.int32),
            executed_degree=one,
            nominal_degree=one,
            target_degree=one,
            trained=torch.ones(rows, dtype=torch.bool, device=device),
            refill=torch.zeros(rows, dtype=torch.bool, device=device),
            path_length=one,
        )
        payload = self.stager.stage(
            (record,),
            slot=self.sequence,
            produced=torch.zeros(1, dtype=torch.int64, device=device),
            sampled_nodes=torch.zeros((), dtype=torch.int64, device=device),
            counterfactual_correction=torch.zeros(
                (), dtype=torch.float32, device=device
            ),
            frontier_overflow=torch.zeros(
                (), dtype=torch.bool, device=device
            ),
            frontier_demand=torch.tensor(
                rows, dtype=torch.int64, device=device
            ),
        )
        topology = TopologyPublication(
            lane=0,
            sequence=self.sequence,
            declared_nodes=rows,
            declared_edges=rows,
            payload=payload,
        )
        self.sequence += 1
        logical_address = torch.zeros(
            1, dtype=torch.int64, device=device
        )
        return CommitPlan(
            tuple(
                DeviceDestinationRequest(
                    slab,
                    counts,
                    values,
                    "score" if slab is self.leaf else "origin",
                    logical_address,
                )
                for slab, values in zip(
                    destinations, destination_values, strict=True
                )
            ),
            CommitCoverage(
                torch.ones(
                    (1, 3 * rows), dtype=torch.bool, device=device
                ),
                ordinal.repeat(3)[None],
                torch.arange(
                    3, dtype=torch.int64, device=device
                ).repeat_interleave(rows)[None],
                ordinal.repeat(3)[None],
            ),
            topology=(topology,),
        )


class _NeutralOperations:
    seed = 17
    error_type = RuntimeError

    def __init__(self) -> None:
        self.refs: dict[str, weakref.ReferenceType] = {}
        self.events: list[str] = []
        self.lifecycle: list[str] | None = None

    def root_destination(self, _plan):
        return RootDestination("origin")

    def root_payload(self, request):
        assert request.destination.input_type == "origin"
        logical = torch.arange(
            request.logical_base,
            request.logical_base + request.rows,
            dtype=torch.int64,
            device=request.plan.device,
        )
        return RootPayload(
            MappingProxyType(
                {
                    "logical_rng_key": torch.stack(
                        (logical, torch.zeros_like(logical)), dim=1
                    ),
                    "model_input": torch.ones(
                        (request.rows, 2),
                        dtype=torch.float32,
                        device=request.plan.device,
                    ),
                    "sampled_private_state": torch.zeros(
                        (request.rows, 1),
                        dtype=torch.int32,
                        device=request.plan.device,
                    ),
                    "owner": torch.zeros(
                        request.rows,
                        dtype=torch.int32,
                        device=request.plan.device,
                    ),
                }
            )
        )

    def rollout(self, request):
        hook = _NeutralRolloutHooks(request)
        rollout = GameRollout(
            (hook,),
            (lambda value, _lanes: {"signal": value.sum(-1)},),
            (None,),
        )
        self.refs = {
            "request": weakref.ref(request),
            "registry": weakref.ref(request.registry),
            "queue": weakref.ref(request.queues["origin"]),
            "hook": weakref.ref(hook),
            "rollout": weakref.ref(rollout),
        }
        return rollout

    def rollout_failure(self, _lane, _failure):
        return RuntimeError("neutral rollout failed")

    def logical_rollout_limit(self, _plan):
        return 4

    def target_algebra(self, plan):
        return TargetAlgebra(plan.partitions, ())

    def completion(self, context):
        assert all(reference() is None for reference in self.refs.values())
        self.events.append("completion")
        if self.lifecycle is not None:
            self.lifecycle.append("completion")
        work = context.owners.slab("LayerWork", "score")

        def complete(batch):
            rows = batch.rows
            values = {
                field.name: torch.zeros(
                    (1, rows, *field.shape),
                    dtype=field.dtype,
                    device=work.device,
                )
                for field in work.schema.fields
            }
            counts = torch.tensor(
                [rows], dtype=torch.int64, device=work.device
            )
            return CompletionPlan(
                (
                    DeviceDestinationRequest(
                        work,
                        counts,
                        values,
                        "score",
                        torch.zeros(
                            1, dtype=torch.int64, device=work.device
                        ),
                    ),
                ),
                _unit_coverage(work.device, rows),
            )

        return GameCompletion(complete)


@CUDA
def test_real_non_poker_rollout_declares_root_and_drops_rollout_graph(
    monkeypatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    plan = _neutral_plan(device)
    owners = GeneratedOwners(plan)
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    operations = _NeutralOperations()
    engine = object.__new__(IterationEngine)
    engine.config = SimpleNamespace(
        plan=plan,
        lanes=(
            SimpleNamespace(
                run_state=None,
                cost_safeguard=None,
                branch_controller=None,
                mode=plan.mode,
                minimum_tree_size=1,
            ),
        ),
        benchmark_telemetry=False,
        distributed=None,
    )
    engine.operations = operations

    state = engine._execute_rollout(owners, iteration=3)
    assert state.input_names == ("origin",)
    assert state.topology.leaf_counts == (1,)
    assert not hasattr(state, "queues")
    assert not hasattr(state, "rollout")
    assert all(reference() is None for reference in operations.refs.values())

    owners.release_after(ExecutionPhase.ROLLOUT)
    owners.enter_phase(ExecutionPhase.QUEUE_DRAIN)
    owners.release_after(ExecutionPhase.QUEUE_DRAIN)
    lifecycle: list[str] = []
    operations.lifecycle = lifecycle
    synchronize = torch.cuda.synchronize
    empty_cache = torch.cuda.empty_cache

    def record_synchronize(selected=None):
        lifecycle.append("synchronize")
        synchronize(selected)

    def record_empty_cache():
        lifecycle.append("empty_cache")
        empty_cache()

    monkeypatch.setattr(torch.cuda, "synchronize", record_synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", record_empty_cache)
    owners.enter_phase(ExecutionPhase.COMPLETION)
    engine._execute_completion(owners, state)
    assert lifecycle[:3] == ["synchronize", "empty_cache", "completion"]
    assert operations.events == ["completion"]
    IterationEngine._finish_owners(owners, entered_index=2)


def _calls(path: Path) -> tuple[ast.Call, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))


def test_core_has_no_poker_dependency_or_semantic_inspection() -> None:
    forbidden_terms = {
        "card",
        "street",
        "fold",
        "hand_class",
        "raise_action",
    }
    for path in CORE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name == "pbg.games.poker"
            or name.startswith("pbg.games.poker.")
            for name in imports
        ), path
        names = {
            node.id.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        assert not forbidden_terms.intersection(names), path


def test_poker_does_not_own_generic_lifecycle_or_execution() -> None:
    forbidden_calls = {
        "enter_phase",
        "release_after",
        "close",
        "select",
        "optimize",
        "barrier",
        "begin",
        "begin_device",
    }
    forbidden_constructors = {
        "SemanticFamilyFIFO",
        "FamilyScheduler",
        "AsyncTopologyPublisher",
        "RolloutOwnerRegistry",
        "CompletionFIFO",
        "SealedSequentialInputs",
        "SealedBranchedInputs",
        "AlignedSequentialStore",
        "RaggedBranchedStore",
        "SequentialBatchSampler",
        "BranchedBatchSampler",
        "ReferenceOutputCache",
        "LayeredTargetExecutor",
        "LearnerTransaction",
    }
    for path in POKER.glob("*.py"):
        calls = _calls(path)
        attributes = {
            node.func.attr
            for node in calls
            if isinstance(node.func, ast.Attribute)
        }
        constructors = {
            node.func.id
            for node in calls
            if isinstance(node.func, ast.Name)
        }
        assert not forbidden_calls.intersection(attributes), path
        assert not forbidden_constructors.intersection(constructors), path


def test_phase_sized_game_program_surface_is_deleted() -> None:
    forbidden = {
        "IterationAttempt",
        "IterationHooks",
        "RolloutLoop",
        "CompletionLoop",
        "PokerIterationHooks",
        "PokerSequentialProgram",
        "PokerBranchedProgram",
        "prepare_rollout",
        "finish_rollout",
        "finish_queue_drain",
        "prepare_completion",
        "finish_completion",
        "align_and_seal",
        "execute_targets",
        "execute_learner",
        "finish_backward",
    }
    production = tuple((REPOSITORY / "pbg").rglob("*.py"))
    definitions = {
        node.name
        for path in production
        for node in ast.walk(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert not forbidden.intersection(definitions)


def test_core_constructs_every_generic_iteration_mechanism() -> None:
    source = (CORE / "iteration.py").read_text(encoding="utf-8")
    required = {
        "SemanticFamilyFIFO(",
        "FamilyScheduler(",
        "AsyncTopologyPublisher(",
        "RolloutOwnerRegistry.from_generated(",
        "CompletionFIFO(",
        "SealedSequentialInputs.seal(",
        "SealedBranchedInputs.seal(",
        "AlignedSequentialStore(",
        "RaggedBranchedStore(",
        "SequentialBatchSampler(",
        "BranchedBatchSampler(",
        "build_reference_output_cache(",
        "LayeredTargetExecutor(",
        "LearnerTransaction(",
        "PublicationTransaction.begin_device(",
    }
    assert not {value for value in required if value not in source}


def test_runtime_constructs_one_engine_and_one_game_bundle() -> None:
    path = REPOSITORY / "pbg/runtime.py"
    calls = _calls(path)
    constructors = [
        node.func.id
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id in {"IterationEngine", "PokerOperations"}
    ]
    assert constructors.count("IterationEngine") == 1
    assert constructors.count("PokerOperations") == 1
    source = path.read_text(encoding="utf-8")
    assert "_sequential_attempt" not in source
    assert "_branched_attempt" not in source


def test_one_scheduler_completion_target_and_learner_caller() -> None:
    ownership = {
        ("scheduler", "select"): [CORE / "iteration.py"],
        ("completion", "complete"): [CORE / "iteration.py"],
    }
    for (receiver, method), expected in ownership.items():
        actual = []
        for path in (REPOSITORY / "pbg").rglob("*.py"):
            if any(
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == receiver
                and node.func.attr == method
                for node in _calls(path)
            ):
                actual.append(path)
        assert actual == expected


def test_no_retired_or_compatibility_objects_remain() -> None:
    forbidden_fragments = ("_Retired", "_Legacy", "_Deprecated")
    for path in (REPOSITORY / "pbg").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = (
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        )
        assert not any(
            fragment in name
            for name in names
            for fragment in forbidden_fragments
        ), path
