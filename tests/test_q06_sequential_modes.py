from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest
import torch
from torch import nn

from pbg.games.poker.queued.folded import folded_statistics
from pbg.games.poker.queued.hooks import (
    FOLDED,
    NLHEGeometry,
    PokerStateMachine,
    rb_partitions,
    sampled_partitions,
)
from pbg.games.poker.queued.model import QueuedPokerModel
from pbg.games.poker.queued.training import SequentialPokerIteration
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
from tests.test_q05_sequential_sampled_training import (
    _PartitionNetwork,
    _phase5_layout,
    _phase5_manifest,
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
    result = SequentialPokerIteration(
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
    ).run(iteration=0)
    assert result.learner.optimizer_steps
    assert all(step.applied for step in result.learner.optimizer_steps)
