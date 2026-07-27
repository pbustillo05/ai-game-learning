from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from pbg.queued import (
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
    validate_schema_on_cuda,
)
from pbg.queued.static_gates import (
    dual_engine_roots,
    forbidden_literal_violations,
    queued_firewall_violations,
)


REPOSITORY = Path(__file__).resolve().parents[1]
QUEUED_PACKAGE = REPOSITORY / "pbg" / "queued"
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
        leaf = _fields(schema, "Leaf")
        sigma_head = _fields(schema, "SigmaHead")
        sigma_target = _fields(schema, "SigmaTarget")
        action_target = _fields(schema, "ActionTarget")
        learner_index = _fields(schema, "LearnerIndex")

        is_response = schema.training_regime is TrainingRegime.BEST_RESPONSE
        is_sampled = schema.private_estimator is PrivateEstimator.SAMPLED
        is_branched = schema.rollout_mode.is_branched

        assert ("role_selector" in family) is is_response
        assert ("sampled_private_state" in family) is is_sampled
        assert ("combo_ids" in leaf) is is_sampled
        assert ("range_weights" in leaf) is (not is_sampled)
        assert ("seed" in leaf) is (not is_sampled)
        assert ("selected_player" in leaf) is (is_response and not is_sampled)
        assert ("old_trace" in sigma_head) is is_branched
        assert ("old_cost" in sigma_head) is is_branched
        assert ("trace_target" in sigma_target) is is_branched
        assert ("cost_target" in sigma_target) is is_branched
        assert ("posterior" in action_target) is (not is_sampled)
        assert ("action_begin" in learner_index) is is_branched
        assert ("action_count" in learner_index) is is_branched
        assert ("log_sampling_weight" in learner_index) is is_branched

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


def test_queued_import_and_vocabulary_firewalls(tmp_path: Path) -> None:
    assert queued_firewall_violations(QUEUED_PACKAGE) == ()
    assert forbidden_literal_violations(
        QUEUED_PACKAGE,
        (
            "normal",
            "branched_" + "rao_blackwell",
            "sequential_" + "rb",
            "branched_" + "rb",
            "dynamic",
        ),
    ) == ()
    assert dual_engine_roots(REPOSITORY, PRODUCTION_ROOTS) == ()

    injected = tmp_path / "pbg" / "queued"
    injected.mkdir(parents=True)
    (injected / "__init__.py").write_text(
        "from pbg.core import storage\n", encoding="utf-8"
    )
    assert queued_firewall_violations(injected)

    package = tmp_path / "graph" / "pbg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for subpackage, module in (("core", "storage"), ("queued", "contracts")):
        directory = package / subpackage
        directory.mkdir()
        (directory / "__init__.py").write_text("", encoding="utf-8")
        (directory / f"{module}.py").write_text("", encoding="utf-8")
    (package / "root.py").write_text(
        "import pbg.core.storage\nimport pbg.queued.contracts\n",
        encoding="utf-8",
    )
    assert dual_engine_roots(tmp_path / "graph", ("pbg.root",)) == ("pbg.root",)


def test_representative_schemas_validate_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the repository acceptance run")
    device = torch.device("cuda", torch.cuda.current_device())
    for schema in (
        mode_schema("self_play", "sequential", "sampled"),
        mode_schema("best_response", "branched.fixed", "rao_blackwell"),
        mode_schema("self_play", "branched.KKT", "sampled"),
    ):
        validate_schema_on_cuda(schema, device)


def test_schema_validation_rejects_non_cuda_device() -> None:
    with pytest.raises(ContractError, match="only on CUDA"):
        validate_schema_on_cuda(MODE_SCHEMAS[0], "cpu")


def test_phase1_inventory_covers_legacy_modules_and_config_fields() -> None:
    ledger = (
        REPOSITORY / "docs" / "audit" / "storage_redesign_migration_ledger.md"
    ).read_text(encoding="utf-8")
    inventory = (
        REPOSITORY
        / "docs"
        / "audit"
        / "queued_phase1_inventory_20260726.md"
    ).read_text(encoding="utf-8")
    mode_contract = (
        REPOSITORY / "docs" / "queued" / "mode_contracts.md"
    ).read_text(encoding="utf-8")

    for path in sorted((REPOSITORY / "pbg" / "core").glob("*.py")):
        if path.name != "__init__.py":
            assert f"core.{path.stem}" in ledger
    for path in sorted((REPOSITORY / "pbg" / "games" / "poker").glob("*.py")):
        if path.name != "__init__.py":
            assert f"poker.{path.stem}" in ledger
    for path in sorted(
        (REPOSITORY / "pbg" / "games" / "poker" / "model").glob("*.py")
    ):
        if path.name != "__init__.py":
            assert path.stem in ledger

    expected_config_fields = (
        "n_players",
        "stack_bb",
        "rank_count",
        "sequential_roots_per_gpu_initial",
        "branching_mode",
        "family_batch_rows",
        "storage_page_rows",
        "completion.n_samples",
        "loss.name",
        "training.mode",
        "training.optimizer",
        "training.ppo_epochs",
        "probe.target_checkpoint",
        "hardware.attention_backend",
        "hardware.precision",
        "use_branching_heads",
        "max_history_tokens",
    )
    for field in expected_config_fields:
        assert field in inventory

    assert "Cartesian product has twelve rows" in mode_contract
    assert "negative-control" in mode_contract
