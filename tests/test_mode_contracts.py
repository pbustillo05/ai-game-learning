from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pbg.core.contracts import (
    CAPACITY_VOCABULARY,
    MODE_SCHEMAS,
    ContractError,
    ConversionSpec,
    Partition,
    PartitionContract,
    PartitionMap,
    PrivateEstimator,
    RolloutMode,
    TrainingRegime,
    mode_schema,
    validate_replica_group,
)
from tests.static_gates import (
    accepted_root_violations,
    core_firewall_violations,
    forbidden_literal_violations,
    stale_cutover_violations,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CORE_PACKAGE = REPOSITORY / "pbg" / "core"
PRODUCTION_ROOTS = (
    "pbg.config",
    "pbg.runtime",
    "pbg.benchmark",
    "pbg.inference",
    "pbg.inference_artifact",
    "pbg.eval.br_probe",
    "pbg.cli.train",
    "pbg.cli.train_replicas",
    "pbg.cli.runpod_replica_selection",
    "pbg.cli.benchmark",
    "pbg.cli.br_eval",
    "pbg.cli.export_inference",
    "pbg.cli.benchmark_inference",
    "pbg.cli.benchmark_replica_learner",
)


def _fields(schema, owner: str) -> frozenset[str]:
    return frozenset(schema.owner(owner).fields)


def test_exact_twelve_row_product_and_lookup_identity() -> None:
    assert len(MODE_SCHEMAS) == 12
    identities = {
        (
            schema.training_regime,
            schema.rollout_mode,
            schema.private_estimator,
        )
        for schema in MODE_SCHEMAS
    }
    assert identities == set(
        (regime, rollout, estimator)
        for regime in TrainingRegime
        for rollout in RolloutMode
        for estimator in PrivateEstimator
    )
    for schema in MODE_SCHEMAS:
        assert (
            mode_schema(
                schema.training_regime.value,
                schema.rollout_mode.value,
                schema.private_estimator.value,
            )
            is schema
        )


@pytest.mark.parametrize(
    ("regime", "rollout", "estimator"),
    (
        ("selfplay", "sequential", "sampled"),
        ("self_play", "branched.dynamic", "sampled"),
        ("self_play", "branched.fixed", "integrated"),
    ),
)
def test_impossible_or_legacy_axis_values_are_rejected(
    regime: str, rollout: str, estimator: str
) -> None:
    with pytest.raises(ContractError):
        mode_schema(regime, rollout, estimator)


def test_rows_and_nested_owner_data_are_immutable() -> None:
    schema = MODE_SCHEMAS[0]
    with pytest.raises(FrozenInstanceError):
        schema.output_domain = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        schema.owners[0].fields = ("changed",)  # type: ignore[misc]


def test_inapplicable_fields_are_absent() -> None:
    for schema in MODE_SCHEMAS:
        family = _fields(schema, "FamilyWork")
        retained = _fields(schema, "FamilyRetained")
        leaf = _fields(schema, "Leaf")
        sigma_head = _fields(schema, "SigmaHead")
        sigma_target = _fields(schema, "SigmaTarget")
        action_target = _fields(schema, "ActionTarget")

        is_response = schema.training_regime is TrainingRegime.BEST_RESPONSE
        is_sampled = schema.private_estimator is PrivateEstimator.SAMPLED
        is_branched = schema.rollout_mode.is_branched

        assert ("role_selector" in family) is is_response
        assert not {"model_input", "sampled_private_state", "owner"} & family
        assert ("sampled_private_state" in retained) is is_sampled
        assert {"model_input", "owner"} <= retained
        assert ("LearnerRetained" in {owner.name for owner in schema.owners}) is (
            is_response
        )
        assert ("leaf_private_profile" in leaf) is is_sampled
        assert ("range_weights" in leaf) is (not is_sampled)
        assert ("seed" in leaf) is (not is_sampled)
        assert ("trained_actor" in leaf) is is_response
        assert ("old_trace" in sigma_head) is is_branched
        assert ("old_cost" in sigma_head) is is_branched
        assert ("trace_target" in sigma_target) is is_branched
        assert ("cost_target" in sigma_target) is is_branched
        assert ("posterior" in action_target) is (not is_sampled)
        if is_branched:
            assert _fields(schema, "LearnerIndex") == {
                "family_row",
                "target_degree",
                "action_begin",
                "action_count",
                "log_sampling_weight",
            }
        else:
            assert "LearnerIndex" not in {
                owner.name for owner in schema.owners
            }

        assert ("frozen_profile" in schema.applicable_config) is is_response
        assert ("completion_samples" in schema.applicable_config) is (
            not is_sampled
        )
        assert ("fixed_beta" in schema.applicable_config) is (
            schema.rollout_mode is RolloutMode.BRANCHED_FIXED
        )
        assert ("kkt_controller" in schema.applicable_config) is (
            schema.rollout_mode is RolloutMode.BRANCHED_KKT
        )


def test_fixed_and_kkt_lanes_share_physical_schema_but_keep_identity() -> None:
    for regime in TrainingRegime:
        for estimator in PrivateEstimator:
            fixed = mode_schema(regime, RolloutMode.BRANCHED_FIXED, estimator)
            kkt = mode_schema(regime, RolloutMode.BRANCHED_KKT, estimator)
            assert fixed.key != kkt.key
            assert fixed.physically_compatible_with(kkt)
            validate_replica_group((fixed, kkt, fixed))

            sequential = mode_schema(regime, RolloutMode.SEQUENTIAL, estimator)
            assert not fixed.physically_compatible_with(sequential)
            with pytest.raises(ContractError):
                validate_replica_group((fixed, sequential))

    with pytest.raises(ContractError):
        validate_replica_group(())


def test_partition_contract_is_closed_and_explicit() -> None:
    contract = PartitionContract(
        inputs=(
            Partition("pre", 16),
            Partition("post_2", 120),
            Partition("post_3", 120),
        ),
        outputs=(Partition("pre", 16), Partition("post", 120)),
        folded=(Partition("pre", 16), Partition("post", 120)),
        input_to_output=(
            PartitionMap("pre", "pre"),
            PartitionMap("post_2", "post"),
            PartitionMap("post_3", "post"),
        ),
        conversions=(ConversionSpec("post", "pre", "sum_classes"),),
    )
    assert contract.output_for("pre") == Partition("pre", 16)
    assert contract.output_for("post_3") == Partition("post", 120)
    with pytest.raises(ContractError):
        contract.output_for("river")

    with pytest.raises(ContractError):
        PartitionContract(
            inputs=(Partition("pre", 16),),
            outputs=(Partition("pre", 16),),
            folded=(Partition("pre", 16),),
            input_to_output=(),
            conversions=(),
        )


def test_capacity_manifest_vocabulary_is_closed_and_nonoverlapping() -> None:
    groups = (
        CAPACITY_VOCABULARY.partitioned_rows,
        CAPACITY_VOCABULARY.scalar_rows,
        CAPACITY_VOCABULARY.pack_limits,
        CAPACITY_VOCABULARY.workspaces,
    )
    names = tuple(name for group in groups for name in group)
    assert len(names) == len(set(names)) == 18
    assert "player_node_cap" in names
    assert "topology_edge_cap" in names
    assert "evaluator_workspace_bytes" in names
    assert "memory_guard_bytes" in names


def test_core_import_and_vocabulary_firewalls(tmp_path: Path) -> None:
    assert core_firewall_violations(CORE_PACKAGE) == ()
    assert forbidden_literal_violations(
        CORE_PACKAGE,
        (
            "normal",
            "branched_" + "rao_blackwell",
            "sequential_" + "rb",
            "branched_" + "rb",
            "dynamic",
        ),
    ) == ()
    assert accepted_root_violations(REPOSITORY, PRODUCTION_ROOTS) == ()
    assert stale_cutover_violations(REPOSITORY) == ()

    injected = tmp_path / "pbg" / "core"
    injected.mkdir(parents=True)
    (injected / "__init__.py").write_text(
        "from pbg.games.poker import cards\n", encoding="utf-8"
    )
    assert core_firewall_violations(injected)

    stale = tmp_path / "pbg" / "queued"
    stale.mkdir(parents=True)
    assert stale_cutover_violations(tmp_path) == ("pbg/queued",)


def test_cutover_deleted_legacy_core_and_poker_modules() -> None:
    assert not (REPOSITORY / "pbg" / "queued").exists()
    retired_core = {
        "adaptive_branching.py",
        "checkpointing.py",
        "completion.py",
        "execution.py",
        "memory_control.py",
        "partition.py",
        "tree.py",
        "utility.py",
    }
    assert not retired_core.intersection(
        path.name for path in CORE_PACKAGE.glob("*.py")
    )
    retired_poker = {
        "adapter.py",
        "bundle.py",
        "mc.py",
        "rollout_backend.py",
        "tensor_state.py",
    }
    poker = REPOSITORY / "pbg" / "games" / "poker"
    assert not retired_poker.intersection(path.name for path in poker.glob("*.py"))
