from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from pbg.core.learner import mode_loss
from pbg.core.learner import DistributedLearnerConfig, PhysicalShape
from pbg.core.optimization import FlatGradientBuffer
from pbg.core.replica_learner import (
    FunctionalReplicaLearner,
    StackedModuleReplicas,
    replica_forward,
)
from pbg.core.replica_checkpoint import ReplicaCheckpointManager
from pbg.core.replica_replay import ReplicaModeLoss
from pbg.core.replica_update import ReplicaFixedRolloutUpdate
from pbg.core.storage import (
    SamplingMetadata,
    SealedTensorBlock,
    SequentialPosteriorPool,
    TrainingPartition,
    TrainingStorage,
)
from pbg.games.poker.adapter import PokerModelBundle, PokerPolicyAdapter
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.model.postflop import PostflopNet
from pbg.games.poker.model.preflop import (
    PokerModelConfig,
    PreflopNet,
)
from pbg.games.poker.model.primitives import MHA
from pbg.games.poker.partitions import poker_partition_specs
from pbg.games.poker.replica import PokerReplicaPartition


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="replica learner is CUDA-only"
)


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.affine = torch.nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.affine(inputs)


def _masked_mse(
    output: torch.Tensor,
    _inputs: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    per_row = (output - targets).square().mean(dim=-1)
    selected = torch.where(valid, per_row, torch.zeros_like(per_row))
    return selected.sum() / valid.sum().clamp_min(1)


def _modules(replicas: int) -> list[_Linear]:
    result = []
    for lane in range(replicas):
        torch.manual_seed(1_000 + lane)
        result.append(_Linear().cuda().train())
    return result


def _poker_replica_module(seed: int, partition_id: int = 0) -> PokerReplicaPartition:
    torch.manual_seed(seed)
    config = PokerModelConfig(
        d_model=16,
        n_heads=4,
        n_layers_pre=1,
        n_layers_post=1,
        m_raise=2,
        k_act=5,
        max_players=2,
        class_embedding_dim=4,
        ffn_hidden=32,
        use_branching_heads=False,
        max_history_tokens=4,
        folder_beta_rank=0,
    )
    bundle = PokerModelBundle(
        PreflopNet(config),
        PostflopNet(config),
    )
    for module in bundle.modules():
        if isinstance(module, MHA):
            module.configure_execution("sdpa", query_chunk_size=128)
    return PokerReplicaPartition(
        bundle,
        partition_id=partition_id,
        n_players=2,
        rank_count=13,
    ).cuda().train()


def _postflop_replica_inputs(
    replicas: int, rows: int
) -> dict[str, torch.Tensor]:
    players = 2
    geometry = DeckGeometry(13)
    board = torch.tensor(
        [0, 5, 10, -1, -1], dtype=torch.int64, device="cuda"
    ).view(1, 1, 5).expand(replicas, rows, 5)
    combo_cards = torch.tensor(
        geometry.combo_cards, dtype=torch.int64, device="cuda"
    )
    conflict = (
        (
            combo_cards[None, None, :, :, None]
            == board[:, :, None, None, :]
        )
        & (board >= 0)[:, :, None, None, :]
    ).any(-1).any(-1)
    compatibility = ~conflict
    ranges = torch.rand(
        (replicas, rows, players, geometry.n_hands), device="cuda"
    )
    ranges = ranges * compatibility[:, :, None]
    ranges = ranges / ranges.sum(-1, keepdim=True)
    return {
        "street": torch.ones(
            (replicas, rows), dtype=torch.int64, device="cuda"
        ),
        "board": board,
        "pot": torch.full((replicas, rows), 4.0, device="cuda"),
        "stacks": torch.full(
            (replicas, rows, players), 30.0, device="cuda"
        ),
        "committed": torch.full(
            (replicas, rows, players), 2.0, device="cuda"
        ),
        "to_act": torch.zeros(
            (replicas, rows), dtype=torch.int64, device="cuda"
        ),
        "legality_summary": torch.ones(
            (replicas, rows, 3), device="cuda"
        ),
        "n_ticks": torch.full(
            (replicas, rows), 64, dtype=torch.int64, device="cuda"
        ),
        "raise_grid_bounds": torch.tensor(
            [2.0, 32.0], device="cuda"
        ).expand(replicas, rows, 2),
        "status": torch.zeros(
            (replicas, rows, players), dtype=torch.int64, device="cuda"
        ),
        "seat_mask": torch.ones(
            (replicas, rows, players), dtype=torch.bool, device="cuda"
        ),
        "button": torch.zeros(
            (replicas, rows), dtype=torch.int64, device="cuda"
        ),
        "committed_street": torch.zeros(
            (replicas, rows, players), device="cuda"
        ),
        "call_amount": torch.zeros((replicas, rows), device="cuda"),
        "current_bet": torch.zeros((replicas, rows), device="cuda"),
        "min_raise_to": torch.full((replicas, rows), 2.0, device="cuda"),
        "max_raise_to": torch.full((replicas, rows), 32.0, device="cuda"),
        "last_full_raise": torch.ones((replicas, rows), device="cuda"),
        "legal_action_mask": torch.ones(
            (replicas, rows, 66), dtype=torch.bool, device="cuda"
        ),
        "raise_ticks": torch.linspace(1.0, 32.0, 64, device="cuda")
        .view(1, 1, 64)
        .expand(replicas, rows, 64),
        "history_actor": torch.zeros(
            (replicas, rows, 2), dtype=torch.int8, device="cuda"
        ),
        "history_street": torch.ones(
            (replicas, rows, 2), dtype=torch.int8, device="cuda"
        ),
        "history_action": torch.ones(
            (replicas, rows, 2), dtype=torch.int8, device="cuda"
        ),
        "history_amount": torch.ones(
            (replicas, rows, 2), device="cuda"
        ),
        "history_pot_before": torch.full(
            (replicas, rows, 2), 3.0, device="cuda"
        ),
        "history_call_amount": torch.zeros(
            (replicas, rows, 2), device="cuda"
        ),
        "history_current_bet": torch.zeros(
            (replicas, rows, 2), device="cuda"
        ),
        "history_min_raise_to": torch.full(
            (replicas, rows, 2), 2.0, device="cuda"
        ),
        "history_max_raise_to": torch.full(
            (replicas, rows, 2), 32.0, device="cuda"
        ),
        "history_mask": torch.ones(
            (replicas, rows, 2), dtype=torch.bool, device="cuda"
        ),
        "range_weights": ranges,
    }


def test_replica_step_matches_independent_adamw_and_clipping() -> None:
    modules = _modules(3)
    references = copy.deepcopy(modules)
    state = StackedModuleReplicas.from_modules(modules)
    learner = FunctionalReplicaLearner(
        state,
        lr=2.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.03,
        grad_clip=0.25,
    )
    optimizers = [
        torch.optim.AdamW(
            module.parameters(),
            lr=2.0e-3,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.03,
        )
        for module in references
    ]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(82)
    inputs = torch.randn((3, 5, 3), device="cuda", generator=generator)
    targets = torch.randn((3, 5, 2), device="cuda", generator=generator)
    valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ],
        device="cuda",
    )

    for _ in range(3):
        outcome = learner.step(_masked_mse, (inputs,), (targets, valid))
        expected_losses = []
        expected_norms = []
        for lane, (module, optimizer) in enumerate(
            zip(references, optimizers, strict=True)
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_mse(
                module(inputs[lane]), inputs[lane], targets[lane], valid[lane]
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 0.25)
            optimizer.step()
            expected_losses.append(loss.detach())
            expected_norms.append(norm.detach())

        torch.testing.assert_close(
            outcome.losses, torch.stack(expected_losses), rtol=2.0e-5, atol=2.0e-6
        )
        torch.testing.assert_close(
            outcome.optimizer.grad_norm_pre_clip,
            torch.stack(expected_norms),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        for name, parameter in state.parameters.items():
            for lane, module in enumerate(references):
                expected = dict(module.named_parameters())[name]
                torch.testing.assert_close(
                    parameter[lane], expected, rtol=3.0e-5, atol=3.0e-6
                )
    assert learner.optimizer.step_count.tolist() == [3.0, 3.0, 3.0]


def test_nonfinite_lane_is_skipped_without_affecting_other_lanes() -> None:
    modules = _modules(3)
    state = StackedModuleReplicas.from_modules(modules)
    learner = FunctionalReplicaLearner(
        state,
        lr=1.0e-3,
        weight_decay=0.01,
        grad_clip=0.1,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in state.parameters.items()
    }
    inputs = torch.full((3, 4, 3), 0.2, device="cuda")
    targets = torch.zeros((3, 4, 2), device="cuda")
    targets[1] = float("nan")
    targets[2] = 100.0
    valid = torch.ones((3, 4), dtype=torch.bool, device="cuda")

    outcome = learner.step(_masked_mse, (inputs,), (targets, valid))

    assert outcome.optimizer.optimizer_step_applied.tolist() == [True, False, True]
    assert learner.optimizer.step_count.tolist() == [1.0, 0.0, 1.0]
    assert outcome.optimizer.grad_clip_scale[2] < 1.0
    for name, parameter in state.parameters.items():
        torch.testing.assert_close(parameter[1], before[name][1])
        assert bool((learner.optimizer.exp_avg[name][1] == 0).all())
        assert bool((learner.optimizer.exp_avg_sq[name][1] == 0).all())
        assert not torch.equal(parameter[0], before[name][0])
        assert not torch.equal(parameter[2], before[name][2])


def test_padding_values_create_no_loss_or_update_mass() -> None:
    modules_a = _modules(2)
    modules_b = copy.deepcopy(modules_a)
    state_a = StackedModuleReplicas.from_modules(modules_a)
    state_b = StackedModuleReplicas.from_modules(modules_b)
    learner_a = FunctionalReplicaLearner(
        state_a, lr=1.0e-3, weight_decay=0.0, grad_clip=1.0
    )
    learner_b = FunctionalReplicaLearner(
        state_b, lr=1.0e-3, weight_decay=0.0, grad_clip=1.0
    )
    inputs = torch.randn((2, 5, 3), device="cuda")
    targets_a = torch.randn((2, 5, 2), device="cuda")
    targets_b = targets_a.clone()
    valid = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, False],
        ],
        device="cuda",
    )
    targets_b[~valid] = 1.0e9

    outcome_a = learner_a.step(_masked_mse, (inputs,), (targets_a, valid))
    outcome_b = learner_b.step(_masked_mse, (inputs,), (targets_b, valid))

    torch.testing.assert_close(outcome_a.losses, outcome_b.losses)
    for name in state_a.parameters:
        torch.testing.assert_close(
            state_a.parameters[name], state_b.parameters[name]
        )


def test_lane_state_exports_to_an_ordinary_checkpoint_optimizer() -> None:
    modules = _modules(2)
    state = StackedModuleReplicas.from_modules(modules)
    learner = FunctionalReplicaLearner(
        state,
        lr=2.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.03,
        grad_clip=0.5,
    )
    inputs = torch.randn((2, 3, 3), device="cuda")
    targets = torch.randn((2, 3, 2), device="cuda")
    valid = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    learner.step(_masked_mse, (inputs,), (targets, valid))

    ordinary = _Linear().cuda()
    ordinary.load_state_dict(state.lane_module_state(1))
    optimizer = torch.optim.AdamW(
        ordinary.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.03,
    )
    learner.optimizer.export_lane_to_optimizer(1, ordinary, optimizer)

    for name, parameter in ordinary.named_parameters():
        torch.testing.assert_close(parameter, state.parameters[name][1])
        exported = optimizer.state[parameter]
        torch.testing.assert_close(
            exported["exp_avg"], learner.optimizer.exp_avg[name][1]
        )
        torch.testing.assert_close(
            exported["exp_avg_sq"], learner.optimizer.exp_avg_sq[name][1]
        )
        torch.testing.assert_close(
            exported["step"], learner.optimizer.step_count[1]
        )


def test_lane_rollout_module_is_a_zero_copy_read_only_view() -> None:
    modules = _modules(2)
    state = StackedModuleReplicas.from_modules(modules)
    lane = state.lane_module(1)

    for name, parameter in lane.named_parameters():
        assert not parameter.requires_grad
        assert parameter.data_ptr() == state.parameters[name][1].data_ptr()
    before = lane.affine.weight.detach().clone()
    saved = state.lane_module_state(1)
    with torch.no_grad():
        state.parameters["affine.weight"][1].add_(0.25)
    torch.testing.assert_close(lane.affine.weight, before + 0.25)
    state.load_lane_module_state(1, saved)
    torch.testing.assert_close(lane.affine.weight, before)


@pytest.mark.parametrize("mode", ["normal", "sequential_rb"])
def test_full_preflop_production_loss_matches_independent_trainers(
    mode: str,
) -> None:
    modules = []
    for lane in range(2):
        modules.append(_poker_replica_module(9_000 + lane))
    references = copy.deepcopy(modules)
    frozen_references = copy.deepcopy(modules)
    with torch.no_grad():
        for lane, module in enumerate(frozen_references):
            for parameter in module.parameters():
                parameter.add_(1.0e-3 * (lane + 1))
    for module in frozen_references:
        module.eval()
    state = StackedModuleReplicas.from_modules(modules)
    frozen_state = StackedModuleReplicas.from_modules(frozen_references)
    learner = FunctionalReplicaLearner(
        state, lr=1.0e-4, weight_decay=0.01, grad_clip=0.5
    )
    replicas, rows, players, classes = 2, 2, 2, 169
    ranges = torch.rand(
        (replicas, rows, players, classes), device="cuda"
    )
    ranges = ranges / ranges.sum(dim=-1, keepdim=True)
    inputs = {
        "street": torch.zeros(
            (replicas, rows), dtype=torch.int64, device="cuda"
        ),
        "board": torch.full(
            (replicas, rows, 5), -1, dtype=torch.int64, device="cuda"
        ),
        "pot": torch.full((replicas, rows), 2.0, device="cuda"),
        "stacks": torch.full(
            (replicas, rows, players), 31.0, device="cuda"
        ),
        "committed": torch.full(
            (replicas, rows, players), 1.0, device="cuda"
        ),
        "to_act": torch.zeros(
            (replicas, rows), dtype=torch.int64, device="cuda"
        ),
        "legality_summary": torch.ones(
            (replicas, rows, 3), device="cuda"
        ),
        "n_ticks": torch.full(
            (replicas, rows), 64, dtype=torch.int64, device="cuda"
        ),
        "raise_grid_bounds": torch.tensor(
            [2.0, 32.0], device="cuda"
        ).expand(replicas, rows, 2),
        "status": torch.zeros(
            (replicas, rows, players), dtype=torch.int64, device="cuda"
        ),
        "seat_mask": torch.ones(
            (replicas, rows, players), dtype=torch.bool, device="cuda"
        ),
        "button": torch.zeros(
            (replicas, rows), dtype=torch.int64, device="cuda"
        ),
        "committed_street": torch.full(
            (replicas, rows, players), 1.0, device="cuda"
        ),
        "call_amount": torch.zeros((replicas, rows), device="cuda"),
        "current_bet": torch.ones((replicas, rows), device="cuda"),
        "min_raise_to": torch.full((replicas, rows), 2.0, device="cuda"),
        "max_raise_to": torch.full((replicas, rows), 32.0, device="cuda"),
        "last_full_raise": torch.ones((replicas, rows), device="cuda"),
        "legal_action_mask": torch.ones(
            (replicas, rows, 66), dtype=torch.bool, device="cuda"
        ),
        "raise_ticks": torch.linspace(1.0, 32.0, 64, device="cuda")
        .view(1, 1, 64)
        .expand(replicas, rows, 64),
        "history_actor": torch.tensor(
            [0, 1], dtype=torch.int8, device="cuda"
        )
        .view(1, 1, 2)
        .expand(replicas, rows, 2),
        "history_street": torch.zeros(
            (replicas, rows, 2), dtype=torch.int8, device="cuda"
        ),
        "history_action": torch.ones(
            (replicas, rows, 2), dtype=torch.int8, device="cuda"
        ),
        "history_amount": torch.ones(
            (replicas, rows, 2), device="cuda"
        ),
        "history_pot_before": torch.ones(
            (replicas, rows, 2), device="cuda"
        ),
        "history_call_amount": torch.zeros(
            (replicas, rows, 2), device="cuda"
        ),
        "history_current_bet": torch.ones(
            (replicas, rows, 2), device="cuda"
        ),
        "history_min_raise_to": torch.full(
            (replicas, rows, 2), 2.0, device="cuda"
        ),
        "history_max_raise_to": torch.full(
            (replicas, rows, 2), 32.0, device="cuda"
        ),
        "history_mask": torch.ones(
            (replicas, rows, 2), dtype=torch.bool, device="cuda"
        ),
        "range_weights": ranges,
    }
    geometry = DeckGeometry(13)
    specs = poker_partition_specs(
        rank_count=13,
        n_players=2,
        router=lambda _: 0,
        forwards={0: lambda _: None, 1: lambda _: None},
        max_history_tokens=4,
        n_ticks=64,
        e_max=66,
    )
    spec = specs[0]
    adapter = PokerPolicyAdapter(
        64, exact_to_model=geometry.class_of_combo
    )
    loss_cfg = SimpleNamespace(
        clip_eps=0.2,
        clip_eps_v=0.2,
        c_v=0.125,
        c_h=0.025,
        c_kl=0.03,
        entropy_schedule="const",
        value_loss="clipped",
    )
    legal_ids = (
        torch.arange(66, dtype=torch.int32, device="cuda")
        .view(1, 1, 66)
        .expand(replicas, rows, 66)
    )
    pack = {
        "column.owner": torch.zeros(
            (replicas, rows), dtype=torch.int32, device="cuda"
        ),
        "column.event": torch.ones(
            (replicas, rows), dtype=torch.int32, device="cuda"
        ),
        "column.legal_event_ids": legal_ids,
        "column.legal_event_mask": torch.ones(
            (replicas, rows, 66), dtype=torch.bool, device="cuda"
        ),
        "objective_scale": torch.full(
            (replicas,), 1.0 / rows, device="cuda"
        ),
    }
    if mode == "normal":
        pack.update(
            {
                "column.realized_type": torch.tensor(
                    [[0, 41], [113, 702]],
                    dtype=torch.int32,
                    device="cuda",
                ),
                "column.old_value": torch.zeros(
                    (replicas, rows), device="cuda"
                ),
                "column.old_log_prob": torch.full(
                    (replicas, rows), -2.0, device="cuda"
                ),
                "column.lambda_return": torch.tensor(
                    [[0.5, -0.25], [0.75, 0.1]], device="cuda"
                ),
            }
        )
    else:
        posterior = torch.rand(
            (replicas, rows, geometry.n_hands), device="cuda"
        )
        posterior = posterior / posterior.sum(-1, keepdim=True)
        pack.update(
            {
                "column.posterior_ref": torch.arange(
                    rows, dtype=torch.int32, device="cuda"
                )[None].expand(replicas, -1),
                "column.old_value": torch.zeros(
                    (replicas, rows, geometry.n_classes), device="cuda"
                ),
                "column.old_log_prob": torch.full(
                    (replicas, rows, geometry.n_classes),
                    -2.0,
                    device="cuda",
                ),
                "column.lambda_return": torch.randn(
                    (replicas, rows, geometry.n_hands), device="cuda"
                ),
                "posterior": posterior,
            }
        )
    replica_loss = ReplicaModeLoss(
        mode=mode,
        spec=spec,
        adapter=adapter,
        loss_cfg=loss_cfg,
        iteration=0,
    )
    reference_output = replica_forward(frozen_state, (inputs,))
    pack.update(
        {
            f"reference.{name}": value
            for name, value in reference_output.items()
        }
    )

    outcome = learner.step(replica_loss, (inputs,), (pack,))
    assert outcome.optimizer.optimizer_step_applied.tolist() == [True, True]
    assert bool(torch.isfinite(outcome.losses).all())

    expected_losses = []
    for lane, reference in enumerate(references):
        optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1.0e-4, weight_decay=0.01
        )
        gradients = FlatGradientBuffer(reference)
        gradients.zero_()
        lane_inputs = {
            name: value[lane : lane + 1] for name, value in inputs.items()
        }
        prepared = reference.prepare_replica_arguments(
            {},
            {},
            (lane_inputs,),
        )[0]
        ordinary_inputs = {
            name: value[0] for name, value in prepared.items()
        }
        output = reference(ordinary_inputs)
        frozen_inputs = frozen_references[lane].prepare_replica_arguments(
            {},
            {},
            (lane_inputs,),
        )[0]
        frozen_output = frozen_references[lane](
            {name: value[0] for name, value in frozen_inputs.items()}
        )
        lane_pack = {name: value[lane] for name, value in pack.items()}
        columns = {
            name.removeprefix("column."): value
            for name, value in lane_pack.items()
            if name.startswith("column.")
        }
        partition = SimpleNamespace(
            columns=columns,
            action_columns=None,
            max_actions_per_decision=None,
        )
        storage = SimpleNamespace(
            mode=mode,
            posterior_pool=(
                None
                if mode == "normal"
                else SimpleNamespace(posterior=lane_pack["posterior"])
            ),
        )
        result = mode_loss(
            mode,
            spec,
            storage,
            partition,
            torch.arange(rows, device="cuda"),
            output,
            adapter,
            loss_cfg,
            reference=frozen_output,
        )
        contribution = result.per_decision.mean()
        contribution.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
        optimizer.step()
        expected_losses.append(contribution.detach())

    torch.testing.assert_close(
        outcome.losses,
        torch.stack(expected_losses),
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    for name, parameter in state.parameters.items():
        for lane, reference in enumerate(references):
            torch.testing.assert_close(
                parameter[lane],
                dict(reference.named_parameters())[name],
                rtol=5.0e-5,
                atol=5.0e-6,
            )

    storages = []
    for lane in range(replicas):
        input_columns = spec.input_schema.allocate(rows, "cuda")
        for name, destination in input_columns.items():
            source = inputs[name][lane]
            if destination.shape == source.shape:
                destination.copy_(source)
            else:
                destination[..., : source.shape[-1]].copy_(source)
        sealed = SealedTensorBlock(
            spec.input_schema,
            input_columns,
            rows,
            spec.input_schema.capacity_bytes(rows, chunks=1),
        )
        columns = {
            name.removeprefix("column."): value[lane].detach()
            for name, value in pack.items()
            if name.startswith("column.")
        }
        posterior_pool = None
        if mode == "sequential_rb":
            posterior_pool = SequentialPosteriorPool(
                pack["posterior"][lane].detach(),
                torch.arange(rows, dtype=torch.int32, device="cuda"),
                torch.zeros(rows, dtype=torch.int32, device="cuda"),
            )
        storages.append(
            TrainingStorage(
                mode=mode,
                partitions={
                    0: TrainingPartition(
                        partition_id=0,
                        inputs=sealed,
                        columns=columns,
                    )
                },
                sampling=SamplingMetadata(
                    {0: torch.zeros((), dtype=torch.float64, device="cuda")}
                ),
                posterior_pool=posterior_pool,
            )
        )
    learner_views = {
        0: state,
        1: state.with_module(state.module.with_partition(1)),
    }
    frozen_views = {
        0: frozen_state,
        1: frozen_state.with_module(frozen_state.module.with_partition(1)),
    }
    ordinary = ReplicaCheckpointManager(
        learner, reference=frozen_state
    ).ordinary_lane(1)
    assert set(dict(ordinary.model.named_parameters())) == {
        name.removeprefix("bundle.") for name in state.parameters
    }
    assert ordinary.optimizer.state
    update = ReplicaFixedRolloutUpdate(
        learner,
        learner_views,
        specs,
        adapter,
        loss_cfg,
        DistributedLearnerConfig(
            statistical_counts={0: rows, 1: 1},
            physical_shapes={
                0: (PhysicalShape(rows, 1),),
                1: (PhysicalShape(1, 1),),
            },
            available_transient_bytes=1,
        ),
        reference_partitions=frozen_views,
        statistical_counts_by_replica=(
            {0: rows, 1: 1},
            {0: 1, 1: 1},
        ),
    )
    generators = []
    for lane in range(replicas):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(15_000 + lane)
        generators.append(generator)
    replay = update(storages, generators, iterations=(1, 4))
    assert replay.losses.shape == (1, replicas)
    assert bool(torch.isfinite(replay.losses).all())
    assert [metric["statistical_rows"] for metric in replay.metrics] == [
        rows,
        1,
    ]
    assert all(metric["exact_objective_mass"] == 1.0 for metric in replay.metrics)


def test_full_postflop_pack_runs_through_replica_transform() -> None:
    modules = [
        _poker_replica_module(12_000 + lane, partition_id=1)
        for lane in range(2)
    ]
    state = StackedModuleReplicas.from_modules(modules)
    learner = FunctionalReplicaLearner(
        state, lr=1.0e-4, weight_decay=0.01, grad_clip=0.5
    )
    inputs = _postflop_replica_inputs(2, 1)
    valid = torch.ones((2, 1), dtype=torch.bool, device="cuda")

    def loss_fn(output, _inputs, row_mask):
        per_row = output.v.float().square().mean(dim=1)
        return torch.where(
            row_mask, per_row, torch.zeros_like(per_row)
        ).sum() / row_mask.sum()

    outcome = learner.step(loss_fn, (inputs,), (valid,))

    assert outcome.optimizer.optimizer_step_applied.tolist() == [True, True]
    assert bool(torch.isfinite(outcome.losses).all())
