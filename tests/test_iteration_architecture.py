from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from pbg.core.capacity import ExecutionPhase
from pbg.core.iteration import IterationBatchResult, IterationEngine


REPOSITORY = Path(__file__).resolve().parents[1]
CORE = REPOSITORY / "pbg/core"
POKER = REPOSITORY / "pbg/games/poker/queued"


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
