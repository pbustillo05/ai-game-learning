from __future__ import annotations

import ast
from pathlib import Path

from pbg.core.capacity import ExecutionPhase
from pbg.core.iteration import (
    CompletionLoop,
    IterationEngine,
    RolloutLoop,
)
from pbg.core.rollout import FamilyScheduler
from pbg.games.poker.queued.branching import PokerBranchedProgram
from pbg.games.poker.queued.training import PokerSequentialProgram

REPOSITORY = Path(__file__).resolve().parents[1]
POKER = REPOSITORY / "pbg/games/poker/queued"


class _Agreement:
    def agree(self, failure):
        return failure


class _Owners:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def enter_phase(self, phase: ExecutionPhase) -> None:
        self.events.append(("enter", phase.name))

    def release_after(self, phase: ExecutionPhase) -> tuple[()]:
        self.events.append(("release", phase.name))
        return ()

    def close(self) -> None:
        self.events.append(("close", "owners"))


class _Attempt:
    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events

    def prepare_rollout(self, *, iteration: int):
        assert iteration == 7
        self.events.append(("work", "ROLLOUT"))
        return RolloutLoop(
            FamilyScheduler(("pre",), 1),
            lambda _time: (),
            lambda: False,
            {},
            None,
            (),
            0,
            (),
            (),
            lambda _pending: None,
            None,
            lambda _lane: RuntimeError("unused"),
            1,
        )

    def finish_rollout(self, *, logical_time: int, topology) -> None:
        assert logical_time == 0
        assert topology == "sealed"

    def finish_queue_drain(self) -> None:
        self.events.append(("work", "QUEUE_DRAIN"))

    def prepare_completion(self) -> CompletionLoop:
        self.events.append(("work", "COMPLETION"))
        return CompletionLoop(_EmptyCompletion(), (0,), None, None, "unused")

    def finish_completion(self) -> None:
        return None

    def align_and_seal(self) -> None:
        self.events.append(("work", "ALIGN_AND_SEAL"))

    def execute_targets(self) -> None:
        self.events.append(("work", "TARGETS"))

    def execute_learner(self, *, iteration: int) -> None:
        assert iteration == 7
        self.events.append(("work", "LEARNER"))

    def finish_backward(self):
        self.events.append(("work", "FINAL_BACKWARD"))
        return "committed"


class _EmptyCompletion:
    def release(self) -> None:
        return None


class _Hooks:
    def __init__(self, attempt: _Attempt) -> None:
        self.attempt = attempt

    def build_attempt(self, _owners):
        return self.attempt


def test_iteration_engine_owns_the_exact_phase_trace() -> None:
    owners = _Owners()
    engine = object.__new__(IterationEngine)
    engine.rank_agreement = _Agreement()
    engine.hooks = _Hooks(_Attempt(owners.events))
    engine.seal_topology = lambda _publisher: "sealed"
    result = engine._run_attempt(owners, iteration=7)
    assert result == "committed"
    assert owners.events == [
        event
        for phase in ExecutionPhase
        for event in (
            ("enter", phase.name),
            ("work", phase.name),
            ("release", phase.name),
        )
    ] + [("close", "owners")]


def test_poker_operations_do_not_own_storage_lifecycle_calls() -> None:
    forbidden = {"enter_phase", "release_after", "close"}
    for path in POKER.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        assert not forbidden.intersection(calls), path


def test_poker_attempts_do_not_encode_the_phase_sequence() -> None:
    for path in POKER.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert "ExecutionPhase" not in names, path


def test_retired_iteration_engines_are_absent_from_production() -> None:
    retired = {
        "SequentialPokerIteration",
        "RecoveringSequentialPokerIteration",
        "_BranchedSampledPokerAttempt",
        "BranchedSampledPokerIteration",
        "ReplicaBranchedPokerIteration",
        "SequentialOverflowRecovery",
        "BranchedOverflowRecovery",
        "PokerSequentialAttempt",
        "PokerBranchedAttempt",
        "PokerReplicaBranchedAttempt",
        "SequentialSampledIteration",
        "BranchedSampledIteration",
    }
    production = tuple((REPOSITORY / "pbg").rglob("*.py"))
    definitions: set[str] = set()
    for path in production:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        )
    assert not retired.intersection(definitions)
    assert {PokerSequentialProgram, PokerBranchedProgram}


def test_poker_programs_are_not_generator_or_run_engines() -> None:
    for path in (
        POKER / "training.py",
        POKER / "branching.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name not in {
                "PokerSequentialProgram",
                "PokerBranchedProgram",
            }:
                continue
            methods = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert "work" not in methods
            assert "run" not in methods
            assert not any(
                isinstance(child, (ast.Yield, ast.YieldFrom))
                for child in ast.walk(node)
            )


def test_one_topology_assembler_accepts_all_rollout_shapes() -> None:
    production = tuple((REPOSITORY / "pbg").rglob("*.py"))
    definitions = [
        (path, node.name)
        for path in production
        for node in ast.walk(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        if isinstance(node, ast.ClassDef)
        and node.name.endswith("TopologyAssembler")
    ]
    assert definitions == [
        (REPOSITORY / "pbg/core/topology.py", "TopologyAssembler")
    ]
    constructors = [
        path
        for path in production
        for node in ast.walk(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TopologyAssembler"
    ]
    assert constructors == [REPOSITORY / "pbg/core/topology.py"]
    topology_source = (REPOSITORY / "pbg/core/topology.py").read_text(
        encoding="utf-8"
    )
    hook_source = (POKER / "hooks.py").read_text(encoding="utf-8")
    assert "class TopologyPacketStager" in topology_source
    assert "class TopologyExpansionRecord" in topology_source
    assert "executed_degree=one" in hook_source
    assert "target_degree=one" in hook_source
    assert "TopologyPublication(" in hook_source


def test_one_learner_objective_accepts_unit_or_ragged_action_spans() -> None:
    path = REPOSITORY / "pbg/core/learner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    learner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LearnerOperation"
    )
    methods = {
        node.name
        for node in learner.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "forward_loss_backward" in methods
    assert not {
        "_forward_branched_loss_backward",
        "_forward_branched_rb_loss_backward",
    }.intersection(methods)
    sequential = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequentialLearnerBatch"
    )
    properties = {
        node.name
        for node in sequential.body
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        "sigma_index",
        "action_ptr",
        "action_index",
        "sigma_fields",
        "action_fields",
    }.issubset(properties)


def test_runtime_constructs_only_the_common_engine() -> None:
    path = REPOSITORY / "pbg/runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constructed = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.endswith("IterationEngine")
    ]
    assert len(constructed) == 1
    assert isinstance(constructed[0].func, ast.Name)
    assert constructed[0].func.id == "IterationEngine"


def test_common_engine_owns_the_only_family_scheduler_loop() -> None:
    calls: list[Path] = []
    for path in (REPOSITORY / "pbg").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "scheduler"
            and node.func.attr == "select"
            for node in ast.walk(tree)
        ):
            calls.append(path)
    assert calls == [REPOSITORY / "pbg/core/iteration.py"]


def test_common_engine_owns_the_only_completion_drain_loop() -> None:
    calls: list[Path] = []
    for path in (REPOSITORY / "pbg").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "completion"
            and node.func.attr == "complete"
            for node in ast.walk(tree)
        ):
            calls.append(path)
    assert calls == [REPOSITORY / "pbg/core/iteration.py"]


def test_common_engine_owns_rollout_transactions_and_topology_lifecycle() -> None:
    engine = (REPOSITORY / "pbg/core/iteration.py").read_text(
        encoding="utf-8"
    )
    assert "RolloutTransaction(" in engine
    assert "topology.accept(" in engine
    assert "seal_lane(" in engine
    assert "build_layer_plan(" in engine
    for path in POKER.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "RolloutTransaction(" not in source, path
        assert "build_layer_plan(" not in source, path


def test_production_modes_use_one_topology_publication_protocol() -> None:
    poker_source = "\n".join(
        path.read_text(encoding="utf-8") for path in POKER.glob("*.py")
    )
    assert "TopologyPublication(" in poker_source
    forbidden = {
        "TopologyDraftPacket",
        "EndpointPatch",
        "ReplicaTopologyPacket",
        "DeferredUnaryTopologyPublication",
        "DeferredBranchedTopologyPublication",
        "TopologyExpansionPublication",
        "expansion_topology",
        "deferred_topology",
    }
    production_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY / "pbg").rglob("*.py")
    )
    assert not {
        name for name in forbidden if name in production_source
    }
    definitions = [
        (path, node.name)
        for path in (REPOSITORY / "pbg").rglob("*.py")
        for node in ast.walk(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        if isinstance(node, ast.ClassDef)
        and node.name == "TopologyPacketStager"
    ]
    assert definitions == [
        (REPOSITORY / "pbg/core/topology.py", "TopologyPacketStager")
    ]


def test_no_retired_or_compatibility_iteration_objects_remain() -> None:
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
