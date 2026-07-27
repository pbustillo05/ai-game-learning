from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from pbg.games.poker.queued.folded import folded_statistics
from pbg.games.poker.queued.hooks import (
    FOLDED,
    NLHEGeometry,
    PAD_CARD,
    PokerForwardInput,
    PokerStateMachine,
    SequentialPokerHooks,
    rb_partitions,
    sampled_partitions,
)
from pbg.games.poker.queued.model import QueuedPokerModel
from pbg.games.poker.queued.training import SequentialPokerIteration
from pbg.games.poker.queued.training import SequentialSharedResponseProbe
from pbg.games.poker.queued.response import (
    balanced_evaluation_probe_seats,
    paired_probe_statistics,
    training_probe_seats,
)
from pbg.queued.capacity import GeneratedOwners, build_storage_plan
from pbg.queued.config import SequentialModeConfig
from pbg.queued.contracts import mode_schema
from pbg.queued.learner import (
    LearnerSchedule,
    LearnerOperation,
    LossConfig,
    ModelEvaluation,
    PolicyEvaluation,
    SequentialLearnerBatch,
)
from pbg.queued.optimization import DenseAdam, OptimizerConfig
from pbg.queued.persistence import convert_legacy_sequential
from pbg.queued.rollout import (
    RolloutOwnerRegistry,
    SemanticFamilyFIFO,
)
from pbg.queued.capacity import ExecutionPhase
from tests.test_q05_sequential_sampled_training import (
    _PartitionNetwork,
    _legacy_checkpoint,
    _phase5_layout,
    _phase5_manifest,
    _publish_rows,
)


class _FinitePartitionNetwork(_PartitionNetwork):
    def forward(self, model_input: torch.Tensor) -> ModelEvaluation:
        return super().forward(
            torch.nan_to_num(model_input, neginf=-20.0, posinf=20.0)
        )


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _config(regime: str, estimator: str) -> dict[str, object]:
    value: dict[str, object] = {
        "mode": {
            "training_regime": regime,
            "rollout_mode": "sequential",
            "private_estimator": estimator,
        },
        "game": {"players": 2, "stack_bb": 8.0, "rank_count": 4},
        "model": {"d_model": 4, "n_heads": 1, "max_players": 2},
        "loss": {},
        "optimizer": {},
        "schedule": {"batch_size": 2},
        "precision": {},
        "reference": {},
        "lambda": 0.95,
        "root_count": 2,
        "root_count_minimum": 1,
        "seed": 7,
    }
    if estimator == "rao_blackwell":
        value["completion"] = {"n_samples": 8, "strata": "default"}
    if regime == "best_response":
        value["frozen_profile"] = {"sha256": "a" * 64}
        value["response_initialization"] = "random"
    return value


@pytest.mark.parametrize(
    ("regime", "estimator"),
    (
        ("self_play", "sampled"),
        ("best_response", "sampled"),
        ("self_play", "rao_blackwell"),
        ("best_response", "rao_blackwell"),
    ),
)
def test_all_sequential_public_configs_are_closed(
    regime: str, estimator: str
) -> None:
    config = SequentialModeConfig.from_mapping(_config(regime, estimator))
    assert config.scientific_mapping()["mode"] == {
        "training_regime": regime,
        "rollout_mode": "sequential",
        "private_estimator": estimator,
    }


def test_probe_is_the_ordinary_best_response_rb_schema() -> None:
    value = _config("best_response", "rao_blackwell")
    value["probe"] = {"target_sha256": "a" * 64}
    config = SequentialModeConfig.from_mapping(value)
    assert config.applicable["probe"] == {"target_sha256": "a" * 64}
    assert mode_schema(
        **{
            "training_regime": config.mode["training_regime"],
            "rollout_mode": config.mode["rollout_mode"],
            "private_estimator": config.mode["private_estimator"],
        }
    ).key == "best_response/sequential/rao_blackwell"
    keys = torch.arange(12, dtype=torch.int64)
    assert training_probe_seats(keys, 3).tolist() == [0, 1, 2] * 4
    seats = balanced_evaluation_probe_seats(keys, 3)
    assert seats.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0]
    # Use a complete P^2 design for the paired lower-bound report.
    seats = balanced_evaluation_probe_seats(torch.arange(9), 3)
    summary = paired_probe_statistics(
        torch.arange(9, dtype=torch.float64), seats, 3
    )
    assert summary.per_seat_gain == (1.0, 4.0, 7.0)
    assert summary.approximate_nashconv_lower_bound == 12.0


def test_sequential_config_rejects_inapplicable_mode_fields() -> None:
    sampled = _config("self_play", "sampled")
    sampled["completion"] = {"n_samples": 8, "strata": "default"}
    with pytest.raises(ValueError, match="inapplicable"):
        SequentialModeConfig.from_mapping(sampled)
    self_play = _config("self_play", "rao_blackwell")
    self_play["frozen_profile"] = {"sha256": "a" * 64}
    with pytest.raises(ValueError, match="best_response"):
        SequentialModeConfig.from_mapping(self_play)
    sampled_response = _config("best_response", "sampled")
    sampled_response["probe"] = {"target_sha256": "a" * 64}
    with pytest.raises(ValueError, match="requires sequential RB"):
        SequentialModeConfig.from_mapping(sampled_response)


@pytest.mark.parametrize(
    ("probe", "expected_regime"),
    ((False, "self_play"), (True, "best_response")),
)
def test_offline_converter_covers_sequential_rb_and_probe(
    tmp_path: Path, probe: bool, expected_regime: str
) -> None:
    legacy = _legacy_checkpoint()
    legacy["config"]["training"]["mode"] = "rao_blackwell"
    legacy["config"]["completion"] = {
        "n_samples": 8,
        "strata": "default",
    }
    if probe:
        legacy["config"]["probe"] = {
            "target_sha256": "b" * 64,
            "initialization": "target",
        }
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "queued.pt"
    torch.save(legacy, source)
    result = convert_legacy_sequential(
        source,
        destination,
        expected_training_regime=expected_regime,
        expected_estimator="rao_blackwell",
    )
    assert result["mode"] == (
        f"{expected_regime}/sequential/rao_blackwell"
    )
    converted = torch.load(destination, map_location="cpu", weights_only=False)
    config = SequentialModeConfig.from_mapping(converted["scientific_config"])
    assert config.applicable["completion"] == {
        "n_samples": 8,
        "strata": "default",
    }
    assert ("probe" in config.applicable) is probe
    with pytest.raises(Exception, match="already exists"):
        convert_legacy_sequential(source, destination)


def test_offline_converter_rejects_wrong_requested_rb_mode(tmp_path: Path) -> None:
    legacy = _legacy_checkpoint()
    legacy["config"]["training"]["mode"] = "rao_blackwell"
    legacy["config"]["completion"]["n_samples"] = 8
    source = tmp_path / "legacy.pt"
    torch.save(legacy, source)
    with pytest.raises(Exception, match="differs"):
        convert_legacy_sequential(
            source,
            tmp_path / "queued.pt",
            expected_training_regime="best_response",
            expected_estimator="rao_blackwell",
        )


def test_rb_partitions_are_k_wide_preflop_and_h_wide_postflop() -> None:
    geometry = NLHEGeometry(3, 8.0, 4)
    partitions = rb_partitions(geometry)
    assert tuple((item.name, item.width) for item in partitions.outputs) == (
        ("pre", 16),
        ("post", 120),
    )
    assert partitions.output_for("pre").name == "pre"
    assert partitions.output_for("post_2").name == "post"
    assert partitions.output_for("post_3").name == "post"


def test_multiway_preflop_and_folded_postflop_refresh_hot_references() -> None:
    hooks = object.__new__(SequentialPokerHooks)
    hooks.geometry = NLHEGeometry(3, 8.0, 4)
    width = hooks.geometry.direct_model_input_width
    raw = torch.zeros((1, 1, width))
    refreshed = torch.ones((1, width))
    hooks._resolved_model_input = lambda _batch: refreshed
    pre = SimpleNamespace(input_type="pre", fields={"model_input": raw})
    post_two = SimpleNamespace(input_type="post_2", fields={"model_input": raw})
    post_three = SimpleNamespace(input_type="post_3", fields={"model_input": raw})
    assert hooks._forward_model_input(pre) is refreshed
    assert hooks._forward_model_input(post_two) is refreshed
    assert torch.equal(hooks._forward_model_input(post_three), raw.reshape(1, width))


@CUDA
def test_repository_folded_statistics_are_deterministic_fp32() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(3, 8.0, 4)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        2, seed=19, iteration=0, retry_epoch=0, device=device
    )
    status = state.status.clone()
    status[:, 2] = FOLDED
    weights = torch.rand(
        (2, 3, geometry.private_hands),
        dtype=torch.float32,
        device=device,
    )
    combo_cards = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    arguments = dict(
        board_cards=state.board_cards,
        range_weights=weights,
        status=status,
        combo_cards=combo_cards,
        beta_rank=4,
        line_dim=3,
    )
    first = folded_statistics(geometry, **arguments)
    second = folded_statistics(geometry, **arguments)
    assert first.keys() == second.keys()
    assert all(value.dtype in (torch.float32, torch.bool, torch.int64)
               for value in first.values())
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert first["folder_mask"][:, 2].all()
    assert not first["folder_mask"][:, :2].any()
    assert torch.count_nonzero(first["beta_eigenvalue"]) > 0
    assert torch.count_nonzero(first["folder_line_meta"]) > 0


@CUDA
def test_folded_owner_refreshes_model_input_and_releases_at_seal() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "sequential", "sampled")
    partitions = sampled_partitions(2, geometry)
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            partitions,
            _phase5_manifest(device),
            _phase5_layout(),
            device,
        )
    )
    owners.enter_phase(ExecutionPhase.ROLLOUT)
    registry = RolloutOwnerRegistry.from_generated(owners)
    queues = {
        name: SemanticFamilyFIFO(
            name,
            partitions,
            owners.slab("FamilyWork", name),
            owners.slab("QueueMetadata", name),
        )
        for name in ("pre", "post_2")
    }
    hooks = SequentialPokerHooks(
        geometry, registry, queues, seed=3, iteration=0, mode=mode
    )
    folded = owners.slab("FoldedRanges", "pre")
    class_range = torch.arange(
        1, geometry.rank_count**2 + 1, dtype=torch.float32, device=device
    )
    class_range /= class_range.sum()
    _publish_rows(folded, 1, {"range": class_range[None]})
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(
        1, seed=5, iteration=0, retry_epoch=0, device=device
    )
    batch = SimpleNamespace(
        fields={
            "model_input": machine.direct_model_input(state)[None],
            "folded_range_row": torch.tensor(
                [[[0, -1]]], dtype=torch.int32, device=device
            ),
        }
    )
    refreshed = hooks._resolved_model_input(batch)
    reach, _ = machine.model_context(refreshed)
    weights = reach[:, 0].exp()
    classes = hooks._combo_class()
    reconstructed = torch.zeros_like(class_range)
    reconstructed.scatter_add_(0, classes, weights[0])
    torch.testing.assert_close(reconstructed, class_range)
    assert torch.equal(refreshed, hooks._resolved_model_input(batch))
    board = torch.tensor(
        [[geometry.first_card, geometry.first_card + 1, geometry.first_card + 2,
          PAD_CARD, PAD_CARD]],
        dtype=torch.uint8,
        device=device,
    )
    post_state = replace(
        state,
        street=torch.ones_like(state.street),
        board_cards=board,
    )
    batch.fields["model_input"] = machine.direct_model_input(post_state)[None]
    chance_refreshed = hooks._resolved_model_input(batch)
    chance_reach, _ = machine.model_context(chance_refreshed)
    combo_cards = machine.combo_cards(
        torch.arange(
            geometry.private_hands, dtype=torch.int32, device=device
        )
    )
    conflict = combo_cards[None].eq(
        board[0, :3, None, None]
    ).any(0).any(-1)
    assert torch.isneginf(chance_reach[0, 0, conflict]).all()
    owners.release_after(ExecutionPhase.ROLLOUT)
    for phase in (
        ExecutionPhase.QUEUE_DRAIN,
        ExecutionPhase.COMPLETION,
        ExecutionPhase.ALIGN_AND_SEAL,
    ):
        owners.enter_phase(phase)
        released = owners.release_after(phase)
    assert {key.partition for key in released if key.owner == "FoldedRanges"} == {
        "pre",
        "post",
    }
    for phase in (
        ExecutionPhase.TARGETS,
        ExecutionPhase.LEARNER,
        ExecutionPhase.FINAL_BACKWARD,
    ):
        owners.enter_phase(phase)
        owners.release_after(phase)
    owners.close()


class _RBModel(nn.Module):
    def __init__(self, width: int, device: torch.device) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.linspace(-0.2, 0.3, width, device=device))
        self.logit = nn.Parameter(torch.linspace(-0.1, 0.2, width, device=device))

    def operation(
        self, _name: str, model_input: torch.Tensor,
        _private: torch.Tensor, _owner: torch.Tensor
    ) -> ModelEvaluation:
        rows = model_input.shape[0]
        return ModelEvaluation(
            MappingProxyType({"logit": self.logit[None].expand(rows, -1)}),
            self.value[None].expand(rows, -1),
        )


def _policy(
    _name: str,
    _model_input: torch.Tensor,
    raw: MappingProxyType,
    _action: torch.Tensor,
    _private: torch.Tensor,
    _owner: torch.Tensor,
    _reference: object,
) -> PolicyEvaluation:
    logit = raw["logit"].float()
    return PolicyEvaluation(logit, torch.zeros_like(logit), torch.zeros_like(logit))


@CUDA
@pytest.mark.parametrize("regime", ("self_play", "best_response"))
def test_rb_loss_stays_in_natural_domain_and_uses_fp32_posterior(
    regime: str,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    width, rows = 16, 2
    model = _RBModel(width, device)
    posterior = torch.arange(
        1, rows * width + 1, dtype=torch.float32, device=device
    ).reshape(rows, width)
    posterior /= posterior.sum(-1, keepdim=True)
    fields = MappingProxyType(
        {
            "model_input": torch.zeros((rows, 1), device=device),
            "owner": torch.zeros(rows, dtype=torch.int32, device=device),
            "action": torch.ones(rows, dtype=torch.int32, device=device),
            "old_log_policy": torch.zeros((rows, width), device=device),
            "old_value": torch.zeros((rows, width), device=device),
            "value_target": torch.full((rows, width), 0.25, device=device),
            "posterior": posterior,
            "advantage_target": torch.full(
                (rows, width), 0.5, device=device
            ),
        }
    )
    batch = SequentialLearnerBatch(
        "pre", torch.arange(rows, device=device), fields
    )
    result = LearnerOperation(
        mode_schema(regime, "sequential", "rao_blackwell"),
        LossConfig(),
        model.operation,
        _policy,
    ).forward_loss_backward(batch, iteration=0)
    assert torch.isfinite(result.total)
    assert model.value.grad is not None and model.value.grad.dtype == torch.float32
    assert model.logit.grad is not None and model.logit.grad.dtype == torch.float32


@CUDA
def test_best_response_rollout_calls_each_model_only_for_its_role_rows() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    calls: dict[str, list[int]] = {}

    def operation(name: str):
        def run(
            _input_type: str,
            model_input: torch.Tensor,
            _private: torch.Tensor,
            _owner: torch.Tensor,
        ) -> ModelEvaluation:
            calls[name] = model_input[:, 0].to(torch.int64).tolist()
            rows = model_input.shape[0]
            value = model_input[:, :1] + (10.0 if name == "frozen" else 0.0)
            return ModelEvaluation(
                MappingProxyType({"logit": torch.zeros((rows, 1), device=device)}),
                value,
            )

        return run

    runner = object.__new__(SequentialPokerIteration)
    runner.mode = mode_schema("best_response", "sequential", "sampled")
    runner.rollout_operation = operation("trained")
    runner.frozen_operation = operation("frozen")
    runner.precision = "fp32"
    forward = PokerForwardInput(
        "pre",
        torch.arange(4, dtype=torch.float32, device=device)[:, None],
        torch.zeros((4, 2), dtype=torch.int32, device=device),
        torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=device),
        torch.tensor([0, 0, 2, 1], dtype=torch.uint8, device=device),
    )
    merged = runner._model(
        forward, torch.tensor([0], dtype=torch.int64, device=device)
    )
    assert calls == {"trained": [0, 2], "frozen": [1, 3]}
    torch.testing.assert_close(
        merged["value"][:, 0],
        torch.tensor([0.0, 11.0, 2.0, 13.0], device=device),
    )


@CUDA
@pytest.mark.parametrize(
    ("regime", "estimator"),
    (
        ("self_play", "sampled"),
        ("best_response", "sampled"),
        ("self_play", "rao_blackwell"),
        ("best_response", "rao_blackwell"),
    ),
)
def test_all_four_sequential_modes_complete_an_optimizer_step(
    regime: str, estimator: str
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema(regime, "sequential", estimator)
    rb = estimator == "rao_blackwell"
    layout = _phase5_layout()
    if rb:
        layout = replace(
            layout,
            evaluator_private_width=geometry.private_hands,
            completion_workspace_row_bytes=4096,
        )
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            rb_partitions(geometry) if rb else sampled_partitions(2),
            _phase5_manifest(device),
            layout,
            device,
        )
    )

    def make_model() -> QueuedPokerModel:
        result = QueuedPokerModel(
            geometry,
            _FinitePartitionNetwork(
                geometry.direct_model_input_width,
                geometry.rank_count**2,
                device,
            ),
            _FinitePartitionNetwork(
                geometry.direct_model_input_width,
                geometry.private_hands,
                device,
            ),
        ).to(device)
        with torch.no_grad():
            for partition in (result.preflop, result.postflop):
                partition.affine.weight.zero_()
                partition.affine.bias.zero_()
        return result

    model = make_model()
    frozen = make_model() if regime == "best_response" else None
    generator = torch.Generator(device=device)
    generator.manual_seed(101)
    iteration_runner = SequentialPokerIteration(
        geometry=geometry,
        owners=owners,
        mode=mode,
        model=model,
        model_operation=(
            model.rb_learner_operation if rb else model.learner_operation
        ),
        rollout_operation=model.sequential_rollout_operation,
        frozen_operation=(
            None if frozen is None else frozen.sequential_rollout_operation
        ),
        optimizer=DenseAdam(
            model, OptimizerConfig(name="adamw", learning_rate=1.0e-3)
        ),
        loss=LossConfig(),
        schedule=LearnerSchedule(batch_size=1),
        generator=generator,
        seed=17,
        lambda_=0.95,
        completion_samples=4 if rb else 0,
    )
    if regime == "best_response" and rb:
        probe = SequentialSharedResponseProbe(
            iteration_runner, target_sha256="a" * 64
        )
        assert probe.mode is mode
        result = probe.run(iteration=0)
    else:
        result = iteration_runner.run(iteration=0)
    assert result.learner.optimizer_steps
    assert all(step.applied for step in result.learner.optimizer_steps)
