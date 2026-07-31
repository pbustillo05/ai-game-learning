from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pbg.core.branching import (
    CostSafeguardController,
    FixedBranchController,
    KKTBranchController,
)
from pbg.core.capacity import (
    CapacityManifest,
    ExecutionGeometry,
    ExternalWorkspaceBounds,
    GeneratedOwners,
    HardwareLimits,
    NamedShape,
    PartitionRows,
    PhysicalLayout,
    build_storage_plan,
)
from pbg.core.contracts import mode_schema
from pbg.core.iteration import IterationBatchResult
from pbg.core.learner import (
    LearnerDTypes,
    LearnerOperation,
    LearnerSchedule,
    LossConfig,
    LossResult,
    ModelEvaluation,
    PolicyEvaluation,
    SequentialLearnerBatch,
)
from pbg.core.optimization import DenseAdam, LossScaler, OptimizerConfig
from pbg.core.persistence import (
    RunState,
    load_replica_checkpoint,
    save_replica_checkpoint,
)
from pbg.core.replica import (
    DistributedSUM,
    FunctionalReplicaInference,
    ReplicaMemoryReport,
    ReplicaModuleOwner,
    ReplicaOptimizerGroup,
)
from pbg.core.storage import SoAField
from pbg.core.training import LearnerIteration, TrainingError
from pbg.games.poker.queued.folded import sequential_family_workspace_row_bytes
from pbg.games.poker.queued.hooks import (
    NLHEGeometry,
    PokerForwardInput,
    sampled_partitions,
)
from pbg.games.poker.queued.model import QueuedPokerModel, build_model_from_config
from pbg.games.poker.queued.replica import PokerReplicaInference
from tests.iteration_support import (
    EngineDrivenReplicaBranchedAttempt,
    EngineDrivenSequentialAttempt,
)
from tests.test_branched_rb_modes import (
    _config as _branched_rb_config,
)
from tests.test_branched_rb_modes import (
    _plan as _branched_rb_plan,
)
from tests.test_branched_sampled_modes import (
    _manifest as _branched_manifest,
)
from tests.test_branched_sampled_modes import _phase7_layout

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="queued replica execution is CUDA-only"
)
MULTI_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="distributed queued learner validation requires two CUDA devices",
)


class ReplicaLearnerScheduler:
    """Test driver for the production lane optimizer group."""

    def __init__(
        self,
        optimizers: ReplicaOptimizerGroup,
        *,
        distributed: DistributedSUM | None = None,
    ) -> None:
        self.optimizers = optimizers
        self.distributed = distributed

    def run_step(self, active_lanes, forward_loss_backward):
        lanes = tuple(sorted(int(lane) for lane in active_lanes))
        self.optimizers._mask(lanes)
        self.optimizers.zero_grad(lanes)
        failures: dict[int, str] = {}
        for lane in lanes:
            try:
                forward_loss_backward(
                    lane,
                    self.optimizers.modules.lane_module(lane),
                    self.optimizers.optimizers[lane],
                )
            except Exception as exc:  # noqa: BLE001
                failures[lane] = f"{type(exc).__name__}: {exc}"
                self.optimizers.gradients[lane].zero_()
        if self.distributed is not None:
            local = torch.zeros(
                self.optimizers.modules.n_replica,
                dtype=torch.bool,
                device=self.optimizers.modules.device,
            )
            for lane in failures:
                local[lane] = True
            common = self.distributed.agree_failures(local)
            for lane in lanes:
                if bool(common[lane]) and lane not in failures:
                    failures[lane] = "failure on another distributed rank"
                    self.optimizers.gradients[lane].zero_()
        successful = tuple(lane for lane in lanes if lane not in failures)
        steps = self.optimizers.step(
            successful,
            distributed=self.distributed,
        )
        return SimpleNamespace(
            active_lanes=lanes,
            optimizer_steps=steps,
            failures=MappingProxyType(failures),
            versions=tuple(
                int(value) for value in self.optimizers.versions.tolist()
            ),
        )


class _Toy(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn((3, 2), generator=generator))
        self.bias = torch.nn.Parameter(torch.randn((2,), generator=generator))
        self.unused = torch.nn.Parameter(torch.full((2,), 0.25))
        self.register_buffer("offset", torch.tensor(float(seed % 7) / 10))

    def forward(self, kind: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
        if kind != "toy":
            raise ValueError("wrong static inference identity")
        return {
            "prediction": value @ self.weight + self.bias + self.offset,
            "label": value[:, :1],
        }


class _PokerPartition(torch.nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.width = width
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, value: torch.Tensor) -> ModelEvaluation:
        rows = value.shape[0]
        base = self.scale * torch.ones(
            (rows, self.width), dtype=value.dtype, device=value.device
        )
        return ModelEvaluation(
            {
                "type_logits": base[..., None].expand(-1, -1, 3),
                "omega": base[..., None],
                "mu": base[..., None],
                "s": base[..., None].abs().add(1),
                "iota": base,
            },
            base,
        )


class _CheckDownPartition(torch.nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        del input_width
        self.output_width = output_width
        self.public = torch.nn.Linear(26, 1)
        self.value_bias = torch.nn.Parameter(torch.zeros(output_width))

    def forward(self, value: torch.Tensor) -> ModelEvaluation:
        rows = value.shape[0]
        public = self.public(value[:, :26].float())
        actor = value[:, 6:8].argmax(-1)
        committed = value[:, 14:16].gather(1, actor[:, None]).squeeze(1)
        facing = value[:, 24].float().gt(committed + 1.0e-6)
        logits = torch.stack(
            (
                torch.where(facing, -30.0, 30.0),
                torch.where(facing, 30.0, -30.0),
                torch.full_like(public[:, 0], -60.0),
            ),
            -1,
        )[:, None].expand(-1, self.output_width, -1)
        base = public.expand(-1, self.output_width)
        return ModelEvaluation(
            MappingProxyType(
                {
                    "type_logits": logits + base[..., None] * 0.001,
                    "omega": torch.ones(
                        (rows, self.output_width, 1),
                        device=value.device,
                    ),
                    "mu": torch.full(
                        (rows, self.output_width, 1),
                        0.5,
                        device=value.device,
                    ),
                    "s": torch.ones(
                        (rows, self.output_width, 1),
                        device=value.device,
                    ),
                    "iota": torch.full(
                        (rows, self.output_width),
                        0.02,
                        device=value.device,
                    ),
                }
            ),
            self.value_bias[None] + base,
        )


class _BranchedCheckDownPartition(_CheckDownPartition):
    def forward(self, value: torch.Tensor) -> ModelEvaluation:
        ordinary = super().forward(value)
        rows = value.shape[0]
        return ModelEvaluation(
            ordinary.policy_raw,
            ordinary.value,
            MappingProxyType(
                {
                    "trace": torch.sigmoid(self.value_bias.mean()).expand(rows),
                    "cost": self.value_bias.mean().exp().expand(rows),
                }
            ),
        )


def _branched_check_down_model(geometry: NLHEGeometry) -> QueuedPokerModel:
    return QueuedPokerModel(
        geometry,
        _BranchedCheckDownPartition(
            geometry.preflop_model_input_width,
            geometry.rank_count**2,
        ),
        _BranchedCheckDownPartition(
            geometry.postflop_model_input_width,
            geometry.private_hands,
        ),
    ).cuda()


class _LearnerToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.2))
        self.value_weight = torch.nn.Parameter(torch.tensor(-0.1))

    def forward(
        self,
        _input_type: str,
        model_input: torch.Tensor,
        _private: torch.Tensor,
        _owner: torch.Tensor,
    ) -> ModelEvaluation:
        value = self.value_weight * model_input
        logit = self.logit * model_input
        return ModelEvaluation({"logit": logit}, value)


def _learner_policy(
    _input_type: str,
    _model_input: torch.Tensor,
    policy_raw: dict[str, torch.Tensor],
    _action: torch.Tensor,
    _private: torch.Tensor,
    _owner: torch.Tensor,
    _reference: dict[str, torch.Tensor] | None,
) -> PolicyEvaluation:
    selected = policy_raw["logit"]
    return PolicyEvaluation(
        selected,
        torch.zeros_like(selected),
        torch.zeros_like(selected),
    )


def _learner_batch() -> SequentialLearnerBatch:
    device = torch.device("cuda")
    rows = 4
    model_input = torch.linspace(0.2, 0.8, rows, device=device)[:, None]
    return SequentialLearnerBatch(
        "pre",
        torch.arange(rows, dtype=torch.int64, device=device),
        {
            "model_input": model_input,
            "sampled_private_state": torch.zeros(
                (rows, 1), dtype=torch.int32, device=device
            ),
            "owner": torch.zeros(rows, dtype=torch.uint8, device=device),
            "action": torch.ones(rows, dtype=torch.int32, device=device),
            "old_log_policy": torch.zeros((rows, 1), device=device),
            "old_value": torch.zeros((rows, 1), device=device),
            "value_target": torch.full((rows, 1), 0.5, device=device),
            "advantage_target": torch.linspace(-0.4, 0.7, rows, device=device)[:, None],
        },
    )


def _modules(count: int) -> list[_Toy]:
    return [_Toy(100 + lane).cuda() for lane in range(count)]


def _poker_manifest(
    device: torch.device, *, n_replica: int
) -> CapacityManifest:
    return CapacityManifest(
        geometry=ExecutionGeometry(2, 15, 64, 4096),
        root_count=1,
        n_replica=n_replica,
        beta_max=1,
        player_node_cap=(
            PartitionRows("pre", 40),
            PartitionRows("post_2", 40),
        ),
        trained_player_cap=(
            PartitionRows("pre", 20),
            PartitionRows("post_2", 20),
        ),
        action_cap=(
            PartitionRows("pre", 20),
            PartitionRows("post_2", 20),
        ),
        folded_range_cap=(
            PartitionRows("pre", 2),
            PartitionRows("post", 2),
        ),
        leaf_cap=8,
        topology_cap=8 * n_replica,
        topology_edge_cap=8 * n_replica,
        family_pack_cap=8,
        chance_frontier_cap=8,
        completion_pack_cap=8,
        learner_sigma_pack_cap=8,
        learner_action_pack_cap=8,
        topology_packet_cap=8,
        target_arena_cap=16 * n_replica,
        movement_scratch_bytes=1 << 20,
        library_workspace_bytes=1024,
        evaluator_workspace_bytes=1024,
        memory_guard_bytes=1024,
        hardware=HardwareLimits(
            torch.cuda.get_device_properties(device).total_memory,
            64 << 20,
            1024,
            1_000_000,
            64,
        ),
        required_external=ExternalWorkspaceBounds(0, 0),
    )


def _poker_layout(geometry: NLHEGeometry) -> PhysicalLayout:
    return PhysicalLayout(
        model_input_shapes=(
            NamedShape("pre", (geometry.preflop_model_input_width,)),
            NamedShape("post_2", (geometry.postflop_model_input_width,)),
        ),
        player_count=2,
        stratum_count=2,
        evaluator_private_width=4,
        family_workspace_row_bytes=sequential_family_workspace_row_bytes(
            geometry, 1
        ),
        chance_frontier_row_bytes=32,
        completion_workspace_row_bytes=64,
        learner_sigma_workspace_row_bytes=64,
        learner_action_workspace_row_bytes=64,
        topology_packet_row_bytes=88,
        opaque_fields=(
            SoAField("leaf_public_state", torch.uint8, (5,), fill=255),
            SoAField("leaf_private_profile", torch.int32, (2,), fill=-1),
        ),
    )


def _check_down_model(geometry: NLHEGeometry) -> QueuedPokerModel:
    return QueuedPokerModel(
        geometry,
        _CheckDownPartition(
            geometry.preflop_model_input_width,
            geometry.rank_count**2,
        ),
        _CheckDownPartition(
            geometry.postflop_model_input_width,
            geometry.private_hands,
        ),
    ).cuda()


def _distributed_sum_worker(
    rank: int,
    world_size: int,
    initialization_file: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{initialization_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        synchronizer = DistributedSUM()
        gradients = torch.full(
            (2, 7),
            float(rank + 1),
            dtype=torch.float32,
            device=torch.device("cuda", rank),
        )
        active = torch.tensor([True, False], device=gradients.device)
        synchronizer.sum_gradients(gradients, active)
        torch.testing.assert_close(gradients[0], torch.full_like(gradients[0], 3.0))
        assert bool(gradients[1].eq(0).all())
        failed = torch.zeros(2, dtype=torch.bool, device=gradients.device)
        if rank == 1:
            failed[1] = True
        common = synchronizer.agree_failures(failed)
        assert common.tolist() == [False, True]

        module_owner = ReplicaModuleOwner((_LearnerToy().cuda(),))
        optimizer_group = ReplicaOptimizerGroup(
            module_owner,
            OptimizerConfig(learning_rate=1.0e-3),
        )
        module = module_owner.lane_module(0)

        class _OneBatch:
            optimizer_steps = 1

            @staticmethod
            def batches():
                yield SequentialLearnerBatch(
                    "pre",
                    torch.tensor([0], dtype=torch.int64, device=gradients.device),
                    {
                        "model_input": torch.ones(
                            (1, 1), device=gradients.device
                        )
                    },
                )

        class _AsymmetricLearner:
            @staticmethod
            def forward_loss_backward(*_args, **_kwargs) -> LossResult:
                if rank == 1:
                    raise RuntimeError("injected pre-SUM learner failure")
                value = module.logit.square() + module.value_weight.square()
                value.backward()
                zero = value.detach() * 0
                return LossResult(value.detach(), zero, zero, zero, zero, zero, zero)

        with pytest.raises(
            TrainingError, match="failed .* distributed SUM|another distributed"
        ):
            LearnerIteration(
                learner=_AsymmetricLearner(),  # type: ignore[arg-type]
                optimizer=optimizer_group.lane_stepper(
                    0, distributed=synchronizer
                ),  # type: ignore[arg-type]
            ).optimize(_OneBatch(), iteration=0)  # type: ignore[arg-type]
        assert optimizer_group.versions.tolist() == [0]
    finally:
        dist.destroy_process_group()


@CUDA
@pytest.mark.parametrize("count", (1, 2, 4, 8))
def test_replica_vmap_matches_independent_active_lanes_without_padding(
    count: int,
) -> None:
    modules = _modules(count)
    owner = ReplicaModuleOwner(modules)
    inference = FunctionalReplicaInference(
        owner,
        encode_output=lambda value: value,
    )
    selected = tuple(range(count - 1, -1, -2))
    lanes = torch.tensor(selected, dtype=torch.int64, device="cuda")
    values = (
        torch.arange(len(selected) * 12, dtype=torch.float32, device="cuda").reshape(
            len(selected), 4, 3
        )
        / 17
    )
    actual = inference(
        lanes,
        values,
        static_arguments=("toy",),
    )
    expected = {
        name: torch.stack(
            tuple(
                modules[lane]("toy", values[row])[name]
                for row, lane in enumerate(selected)
            )
        )
        for name in ("prediction", "label")
    }
    assert actual["prediction"].shape[:2] == (len(selected), 4)
    assert actual["label"].shape == (len(selected), 4, 1)
    assert not actual["prediction"].requires_grad
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value)


@CUDA
def test_poker_rollout_vmap_preserves_lane_weights_and_natural_labels() -> None:
    geometry = NLHEGeometry(2, 8.0, 4)
    modules = [
        QueuedPokerModel(
            geometry,
            _PokerPartition(geometry.rank_count**2, 0.1 + lane),
            _PokerPartition(geometry.private_hands, 0.2 + lane),
        ).cuda()
        for lane in range(3)
    ]
    inference = PokerReplicaInference(modules)
    lanes = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
    rows = 3
    model_input = torch.zeros(
        (2, rows, geometry.preflop_model_input_width),
        device="cuda",
    )
    private = torch.zeros((2, rows, geometry.players), dtype=torch.int32, device="cuda")
    owner = torch.zeros((2, rows), dtype=torch.uint8, device="cuda")
    actual = inference(
        "pre",
        model_input,
        private,
        owner,
        active_lanes=lanes,
    )
    expected = torch.stack(
        (
            modules[2]
            .rollout_operation("pre", model_input[0], private[0], owner[0])
            .value,
            modules[0]
            .rollout_operation("pre", model_input[1], private[1], owner[1])
            .value,
        )
    )
    torch.testing.assert_close(actual.value, expected)
    assert actual.value.shape == (2, rows, geometry.private_classes)
    assert float(actual.value[0, 0, 0]) != float(actual.value[1, 0, 0])
@CUDA
def test_shared_owner_sequential_poker_runs_real_two_lane_transaction() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "sequential", "sampled")
    source = (_check_down_model(geometry), _check_down_model(geometry))
    ordinary_model = copy.deepcopy(source[0])
    ordinary_optimizer = DenseAdam(
        ordinary_model,
        OptimizerConfig(name="adamw", grad_clip=5.0),
    )
    ordinary = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=GeneratedOwners(
            build_storage_plan(
                mode,
                sampled_partitions(geometry),
                _poker_manifest(device, n_replica=1),
                _poker_layout(geometry),
                device,
            )
        ),
        mode=mode,
        model=ordinary_model,
        model_operation=ordinary_model.learner_operation,
        optimizer=ordinary_optimizer,
        loss=LossConfig(reference_kl_coefficient=0),
        schedule=LearnerSchedule(batch_size=1),
        generator=torch.Generator(device=device).manual_seed(710),
        seed=710,
        lambda_=0.95,
        run_state=RunState("ordinary-oracle", root_count=1),
    ).run(iteration=0)
    inference = PokerReplicaInference(source)
    optimizers = ReplicaOptimizerGroup(
        inference.modules,
        OptimizerConfig(name="adamw", grad_clip=5.0),
    )
    owners = GeneratedOwners(
        build_storage_plan(
            mode,
            sampled_partitions(geometry),
            _poker_manifest(device, n_replica=2),
            _poker_layout(geometry),
            device,
        )
    )
    modules = tuple(
        inference.modules.lane_module(lane) for lane in range(2)
    )
    generators = tuple(
        torch.Generator(device=device).manual_seed(710 + lane)
        for lane in range(2)
    )
    run_states = tuple(
        RunState(f"real-replica-{lane}", root_count=1)
        for lane in range(2)
    )
    result = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=owners,
        mode=mode,
        model=modules[0],
        model_operation=modules[0].learner_operation,
        optimizer=optimizers.optimizers[0],
        loss=LossConfig(reference_kl_coefficient=0),
        schedule=LearnerSchedule(batch_size=1),
        generator=generators[0],
        seed=710,
        lambda_=0.95,
        replica_inference=inference,
        replica_optimizers=optimizers,
        replica_generators=generators,
        replica_run_states=run_states,
        replica_model_operations=tuple(
            module.learner_operation for module in modules
        ),
    ).run(iteration=0)
    assert isinstance(result, IterationBatchResult)
    assert result.failures == {}
    assert result.versions == (8, 8)
    assert all(value is not None for value in result.results)
    for value in result.results:
        assert value is not None
        assert (value.player_nodes, value.leaves) == (8, 1)
        assert value.learner is not None
        assert value.learner.statistical_presentations == 8
    assert tuple(state.iteration for state in run_states) == (1, 1)
    inference.modules.assert_zero_copy()
    lane_zero = result.results[0]
    assert lane_zero is not None
    assert (
        lane_zero.player_nodes,
        lane_zero.leaves,
        lane_zero.target_layers,
    ) == (
        ordinary.player_nodes,
        ordinary.leaves,
        ordinary.target_layers,
    )
    for name, expected in ordinary_model.state_dict().items():
        torch.testing.assert_close(
            inference.modules.lane_state_dict(0)[name],
            expected,
            atol=3.0e-7,
            rtol=0,
        )


@CUDA
@pytest.mark.parametrize("count", (2, 4, 8))
def test_complete_sequential_transactions_isolate_every_replica_lane(
    count: int,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "sequential", "sampled")
    source = tuple(_check_down_model(geometry) for _ in range(count))
    with torch.no_grad():
        for lane, model in enumerate(source):
            model.preflop.value_bias.add_(lane * 0.01)
    inference = PokerReplicaInference(source)
    optimizers = ReplicaOptimizerGroup(
        inference.modules,
        OptimizerConfig(name="adamw", grad_clip=5.0),
    )
    modules = tuple(
        inference.modules.lane_module(lane) for lane in range(count)
    )
    learner_calls = [0 for _ in range(count)]

    class _CountedLearner:
        def __init__(self, lane: int, operation) -> None:
            self.lane = lane
            self.operation = operation

        def __call__(self, *arguments, **keywords):
            learner_calls[self.lane] += 1
            return self.operation(*arguments, **keywords)

    learner_operations = tuple(
        _CountedLearner(lane, module.learner_operation)
        for lane, module in enumerate(modules)
    )
    generators = tuple(
        torch.Generator(device=device).manual_seed(730 + lane)
        for lane in range(count)
    )
    initial_rng = tuple(generator.get_state() for generator in generators)
    run_states = tuple(
        RunState(f"complete-isolation-{count}-{lane}", root_count=1)
        for lane in range(count)
    )
    result = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=GeneratedOwners(
            build_storage_plan(
                mode,
                sampled_partitions(geometry),
                _poker_manifest(device, n_replica=count),
                _poker_layout(geometry),
                device,
            )
        ),
        mode=mode,
        model=modules[0],
        model_operation=learner_operations[0],
        optimizer=optimizers.optimizers[0],
        loss=LossConfig(reference_kl_coefficient=0),
        schedule=LearnerSchedule(batch_size=1),
        generator=generators[0],
        seed=730,
        lambda_=0.95,
        replica_inference=inference,
        replica_optimizers=optimizers,
        replica_generators=generators,
        replica_run_states=run_states,
        replica_model_operations=learner_operations,
    ).run(iteration=0)
    assert isinstance(result, IterationBatchResult)
    assert result.failures == {}
    assert result.versions == (8,) * count
    assert all(value is not None for value in result.results)
    assert tuple(state.iteration for state in run_states) == (1,) * count
    assert learner_calls == [8] * count
    final_rng = tuple(generator.get_state() for generator in generators)
    assert all(
        not torch.equal(before, after)
        for before, after in zip(initial_rng, final_rng, strict=True)
    )
    assert len({bytes(value.cpu().tolist()) for value in final_rng}) == count
    assert all(
        optimizers.gradients[lane].data_ptr()
        != optimizers.gradients[other].data_ptr()
        for lane in range(count)
        for other in range(lane)
    )
    inference.modules.assert_zero_copy()


@CUDA
@pytest.mark.parametrize("regime", ("self_play", "best_response"))
@pytest.mark.parametrize("estimator", ("sampled", "rao_blackwell"))
def test_shared_owner_branched_poker_preserves_fixed_kkt_lane_identity(
    regime: str,
    estimator: str,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    fixed = mode_schema(regime, "branched.fixed", estimator)
    kkt = mode_schema(regime, "branched.KKT", estimator)

    def models(selected_regime: str) -> tuple[torch.nn.Module, ...]:
        if estimator == "rao_blackwell":
            return (
                build_model_from_config(
                    _branched_rb_config(selected_regime, "branched.fixed"),
                    device,
                ),
                build_model_from_config(
                    _branched_rb_config(selected_regime, "branched.KKT"),
                    device,
                ),
            )
        return (
            _branched_check_down_model(geometry),
            _branched_check_down_model(geometry),
        )

    inference = PokerReplicaInference(models(regime))
    frozen = (
        PokerReplicaInference(models("self_play"))
        if regime == "best_response"
        else None
    )
    plan = (
        build_storage_plan(
            fixed,
            sampled_partitions(geometry),
            replace(_branched_manifest(device), n_replica=2),
            replace(_phase7_layout(geometry), player_count=2),
            device,
        )
        if estimator == "sampled"
        else _branched_rb_plan(
            device,
            fixed,
            geometry,
            n_replica=2,
        )
    )
    optimizers = ReplicaOptimizerGroup(
        inference.modules,
        OptimizerConfig(name="adamw", grad_clip=5.0),
    )
    run_states = (
        RunState(f"branched-fixed-{regime}-{estimator}", root_count=1),
        RunState(f"branched-kkt-{regime}-{estimator}", root_count=1),
    )
    controllers = (
        FixedBranchController(
            {"player": 1.0, "chance": 1.0},
            2,
        ),
        KKTBranchController(
            {"player": 1.0, "chance": 1.0},
            2,
            warmup_iterations=0,
            target_tree_size=256,
            gamma_kappa=0.2,
            eta_kappa=0.2,
            kappa_min=1.0e-6,
            kappa_max=1.0e-6,
            log_kappa=math.log(1.0e-6),
        ),
    )
    result = EngineDrivenReplicaBranchedAttempt(
        geometry=geometry,
        owners=GeneratedOwners(plan),
        modes=(fixed, kkt),
        controllers=controllers,
        safeguards=(
            CostSafeguardController(1.0, 10.0, 1),
            CostSafeguardController(1.0, 10.0, 1),
        ),
        reference_operations=(None, None),
        inference=inference,
        frozen_inference=frozen,
        optimizers=optimizers,
        losses=(
            LossConfig(
                reference_kl_coefficient=0,
                trace_coefficient=0.1,
                cost_coefficient=0.1,
            ),
        )
        * 2,
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generators=tuple(
            torch.Generator(device=device).manual_seed(919 + lane)
            for lane in range(2)
        ),
        run_states=run_states,
        seed=919,
        lambda_=0.95,
        completion_samples=4 if estimator == "rao_blackwell" else 0,
        posterior_max_attempts=16,
    ).run(iteration=0)
    assert result.failures == {}
    assert result.versions == (1, 1)
    assert all(value is not None for value in result.results)
    fixed_result, kkt_result = result.results
    assert fixed_result is not None
    assert kkt_result is not None
    assert fixed_result.root_executed_degree == (1,)
    assert all(value in (1, 2) for value in kkt_result.root_executed_degree)
    assert tuple(state.branch_controller_state["mode"] for state in run_states) == (
        "branched.fixed",
        "branched.KKT",
    )
    assert "updates" not in run_states[0].branch_controller_state
    assert run_states[1].branch_controller_state["updates"] == 1
    inference.modules.assert_zero_copy()


@CUDA
def test_branched_replica_learner_failure_aborts_before_any_lane_commit() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("self_play", "branched.fixed", "sampled")
    inference = PokerReplicaInference(
        tuple(_branched_check_down_model(geometry) for _ in range(2))
    )
    optimizers = ReplicaOptimizerGroup(
        inference.modules,
        OptimizerConfig(name="adamw", grad_clip=5.0),
    )
    healthy_module = inference.modules.lane_module(0)
    failed_module = inference.modules.lane_module(1)
    healthy_operation = healthy_module.learner_operation
    learner_calls = [0, 0]

    def count_healthy(*args, **kwargs):
        learner_calls[0] += 1
        return healthy_operation(*args, **kwargs)

    def fail_learner(*_args, **_kwargs):
        learner_calls[1] += 1
        raise RuntimeError("injected branched lane failure")

    healthy_module.learner_operation = count_healthy  # type: ignore[method-assign]
    failed_module.learner_operation = fail_learner  # type: ignore[method-assign]
    run_states = (
        RunState("branched-healthy-lane", root_count=1),
        RunState("branched-failed-lane", root_count=1),
    )
    generators = tuple(
        torch.Generator(device=device).manual_seed(950 + lane)
        for lane in range(2)
    )
    parameters_before = {
        name: value.clone()
        for name, value in inference.modules.parameters.items()
    }
    buffers_before = {
        name: value.clone()
        for name, value in inference.modules.buffers.items()
    }
    exp_avg_before = optimizers.exp_avg.clone()
    exp_avg_sq_before = optimizers.exp_avg_sq.clone()
    steps_before = tuple(
        optimizer.step_count for optimizer in optimizers.optimizers
    )
    scalers_before = tuple(
        optimizer.scaler.state_dict() for optimizer in optimizers.optimizers
    )
    rng_before = tuple(generator.get_state() for generator in generators)
    with pytest.raises(RuntimeError, match="every transition was rolled back"):
        EngineDrivenReplicaBranchedAttempt(
        geometry=geometry,
        owners=GeneratedOwners(
            build_storage_plan(
                mode,
                sampled_partitions(geometry),
                replace(_branched_manifest(device), n_replica=2),
                replace(_phase7_layout(geometry), player_count=2),
                device,
            )
        ),
        modes=(mode, mode),
        controllers=(
            FixedBranchController(
                {"player": 1.0, "chance": 1.0},
                2,
            ),
            FixedBranchController(
                {"player": 1.0, "chance": 1.0},
                2,
            ),
        ),
        safeguards=(
            CostSafeguardController(1.0, 10.0, 1),
            CostSafeguardController(1.0, 10.0, 1),
        ),
        reference_operations=(None, None),
        inference=inference,
        optimizers=optimizers,
        losses=(LossConfig(reference_kl_coefficient=0),) * 2,
        schedule=LearnerSchedule(batch_size=2, gradient_steps=1),
        generators=generators,
        run_states=run_states,
        seed=950,
        lambda_=0.95,
        ).run(iteration=0)
    assert tuple(int(value) for value in optimizers.versions.tolist()) == (0, 0)
    assert tuple(state.iteration for state in run_states) == (0, 0)
    assert learner_calls == [2, 1]
    assert bool(optimizers.gradients.eq(0).all())
    for name, expected in parameters_before.items():
        torch.testing.assert_close(inference.modules.parameters[name], expected)
    for name, expected in buffers_before.items():
        torch.testing.assert_close(inference.modules.buffers[name], expected)
    torch.testing.assert_close(optimizers.exp_avg, exp_avg_before)
    torch.testing.assert_close(optimizers.exp_avg_sq, exp_avg_sq_before)
    assert tuple(
        optimizer.step_count for optimizer in optimizers.optimizers
    ) == steps_before
    assert tuple(
        optimizer.scaler.state_dict() for optimizer in optimizers.optimizers
    ) == scalers_before
    assert all(
        torch.equal(generator.get_state(), expected)
        for generator, expected in zip(generators, rng_before, strict=True)
    )


@CUDA
def test_replica_best_response_groups_compact_learned_and_frozen_rows() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = NLHEGeometry(2, 8.0, 4)
    mode = mode_schema("best_response", "sequential", "sampled")
    learned_source = (_check_down_model(geometry), _check_down_model(geometry))
    frozen_source = (_check_down_model(geometry), _check_down_model(geometry))
    learned = PokerReplicaInference(learned_source)
    frozen = PokerReplicaInference(frozen_source)
    optimizers = ReplicaOptimizerGroup(
        learned.modules,
        OptimizerConfig(name="adamw"),
    )
    modules = tuple(learned.modules.lane_module(lane) for lane in range(2))
    iteration = EngineDrivenSequentialAttempt(
        geometry=geometry,
        owners=GeneratedOwners(
            build_storage_plan(
                mode,
                sampled_partitions(geometry),
                _poker_manifest(device, n_replica=2),
                _poker_layout(geometry),
                device,
            )
        ),
        mode=mode,
        model=modules[0],
        model_operation=modules[0].learner_operation,
        optimizer=optimizers.optimizers[0],
        loss=LossConfig(reference_kl_coefficient=0),
        schedule=LearnerSchedule(batch_size=1),
        generator=torch.Generator(device=device).manual_seed(812),
        seed=812,
        lambda_=0.95,
        frozen_operation=frozen.modules.lane_module(0).rollout_operation,
        replica_inference=learned,
        replica_frozen_inference=frozen,
        replica_optimizers=optimizers,
        replica_generators=tuple(
            torch.Generator(device=device).manual_seed(812 + lane)
            for lane in range(2)
        ),
        replica_run_states=tuple(
            RunState(f"br-lane-{lane}", root_count=1)
            for lane in range(2)
        ),
        replica_model_operations=tuple(
            module.learner_operation for module in modules
        ),
    )

    class _Observed:
        def __init__(self, operation: object) -> None:
            self.operation = operation
            self.calls: list[tuple[int, int]] = []

        def __call__(
            self, active_lanes: torch.Tensor, *arguments: torch.Tensor, **keywords
        ):
            self.calls.append((int(active_lanes.numel()), arguments[0].shape[1]))
            return self.operation(active_lanes, *arguments, **keywords)

    learned_calls = _Observed(learned.operation)
    frozen_calls = _Observed(frozen.operation)
    learned.operation = learned_calls  # type: ignore[assignment]
    frozen.operation = frozen_calls  # type: ignore[assignment]
    rows = 3
    model_input = torch.zeros(
        (2, rows, geometry.preflop_model_input_width), device=device
    )
    private = torch.zeros(
        (2, rows, geometry.players), dtype=torch.int32, device=device
    )
    owner = torch.tensor(
        [[0, 1, 0], [1, 0, 1]], dtype=torch.uint8, device=device
    )
    role = torch.tensor(
        [[0, 1, 1], [1, 1, 1]], dtype=torch.uint8, device=device
    )
    observed = iteration.operations._model(
        0,
        PokerForwardInput(
            "pre",
            model_input.flatten(0, 1),
            private.flatten(0, 1),
            owner.flatten(),
            role.flatten(),
        ),
        torch.tensor([0, 1], dtype=torch.int64, device=device),
    )
    assert learned_calls.calls == [(2, 2)]
    assert frozen_calls.calls == [(2, 1)]
    expected = []
    for lane in range(2):
        learned_value = learned_source[lane].rollout_operation(
            "pre", model_input[lane], private[lane], owner[lane]
        ).value
        frozen_value = frozen_source[lane].rollout_operation(
            "pre", model_input[lane], private[lane], owner[lane]
        ).value
        expected.append(
            torch.where(
                owner[lane].eq(role[lane])[:, None],
                learned_value,
                frozen_value,
            )
        )
    torch.testing.assert_close(
        observed["value"], torch.stack(expected).flatten(0, 1)
    )


@CUDA
def test_one_lane_uses_the_ordinary_learner_graph_for_complete_run_parity() -> None:
    source = _LearnerToy().cuda()
    oracle = copy.deepcopy(source)
    owner = ReplicaModuleOwner((source,))
    config = OptimizerConfig(
        learning_rate=3.0e-3,
        weight_decay=0.02,
        grad_clip=0.4,
    )
    group = ReplicaOptimizerGroup(owner, config)
    scheduler = ReplicaLearnerScheduler(group)
    oracle_optimizer = DenseAdam(oracle, config)
    mode = mode_schema("self_play", "sequential", "sampled")
    loss = LossConfig(
        entropy_coefficient=0,
        reference_kl_coefficient=0,
    )
    batch = _learner_batch()
    for iteration in range(3):
        result = scheduler.run_step(
            (0,),
            lambda _lane, module, optimizer, step=iteration: LearnerOperation(
                mode,
                loss,
                module,
                _learner_policy,
            ).forward_loss_backward(
                batch,
                iteration=step,
                loss_scale=optimizer.scaler.scale,
            ),
        )
        oracle_optimizer.zero_grad()
        LearnerOperation(
            mode,
            loss,
            oracle,
            _learner_policy,
        ).forward_loss_backward(
            batch,
            iteration=iteration,
            loss_scale=oracle_optimizer.scaler.scale,
        )
        expected = oracle_optimizer.step()
        actual = result.optimizer_steps[0]
        assert actual is not None
        assert actual.grad_norm == pytest.approx(expected.grad_norm, abs=0)
        for name, parameter in owner.lane_module(0).named_parameters():
            torch.testing.assert_close(
                parameter,
                dict(oracle.named_parameters())[name],
                atol=0,
                rtol=0,
            )


@CUDA
def test_lane_local_loss_controls_preserve_independent_updates() -> None:
    modules = (_LearnerToy().cuda(), _LearnerToy().cuda())
    owner = ReplicaModuleOwner(modules)
    group = ReplicaOptimizerGroup(owner, OptimizerConfig(learning_rate=3.0e-3))
    scheduler = ReplicaLearnerScheduler(group)
    mode = mode_schema("self_play", "sequential", "sampled")
    batch = _learner_batch()
    controls = (
        LossConfig(value_coefficient=0, entropy_coefficient=0),
        LossConfig(value_coefficient=2.0, entropy_coefficient=0),
    )

    def backward(lane: int, module: torch.nn.Module, optimizer: DenseAdam) -> None:
        LearnerOperation(
            mode,
            controls[lane],
            module,
            _learner_policy,
        ).forward_loss_backward(
            batch,
            iteration=0,
            loss_scale=optimizer.scaler.scale,
        )

    result = scheduler.run_step((0, 1), backward)
    assert result.failures == {}
    assert not torch.equal(
        owner.parameters["value_weight"][0],
        owner.parameters["value_weight"][1],
    )


@CUDA
@pytest.mark.parametrize("optimizer_name", ("adam", "adamw"))
def test_zero_copy_lanes_use_one_dense_gradient_axis_and_dense_oracle(
    optimizer_name: str,
) -> None:
    modules = _modules(4)
    oracle_modules = copy.deepcopy(modules)
    owner = ReplicaModuleOwner(modules)
    config = OptimizerConfig(
        name=optimizer_name,
        learning_rate=2.0e-3,
        weight_decay=0.03,
        grad_clip=0.2,
    )
    group = ReplicaOptimizerGroup(owner, config)
    scheduler = ReplicaLearnerScheduler(group)
    oracle_optimizers = [DenseAdam(module, config) for module in oracle_modules]
    inputs = torch.randn((4, 5, 3), device="cuda")
    targets = torch.randn((4, 5, 2), device="cuda")
    active = (0, 2, 3)
    calls: list[int] = []

    def backward(lane: int, module: torch.nn.Module, _optimizer: DenseAdam) -> None:
        calls.append(lane)
        (
            module("toy", inputs[lane])["prediction"] - targets[lane]
        ).square().mean().backward()

    result = scheduler.run_step(active, backward)
    assert calls == list(active)
    assert result.failures == {}
    assert group.gradients.shape == (4, owner.parameter_count)
    memory = group.memory_report()
    assert isinstance(memory, ReplicaMemoryReport)
    memory.assert_reconciled(owner, group)
    assert memory.gradient_bytes == 4 * owner.parameter_count * 4
    assert memory.optimizer_bytes == 2 * memory.gradient_bytes
    assert memory.transaction_bytes == (
        3 * memory.gradient_bytes + memory.buffer_bytes
    )
    assert bool(group.gradients[1].eq(0).all())
    assert group.versions.tolist() == [1, 0, 1, 1]
    owner.assert_zero_copy()

    for lane in active:
        oracle = oracle_optimizers[lane]
        oracle.zero_grad()
        (
            oracle_modules[lane]("toy", inputs[lane])["prediction"] - targets[lane]
        ).square().mean().backward()
        expected_step = oracle.step()
        observed_step = result.optimizer_steps[lane]
        assert observed_step is not None
        assert observed_step.grad_norm == pytest.approx(
            expected_step.grad_norm, rel=1.0e-6, abs=1.0e-7
        )
    for lane in range(4):
        actual = owner.lane_module(lane).state_dict()
        expected = (
            oracle_modules[lane].state_dict()
            if lane in active
            else modules[lane].state_dict()
        )
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])


@CUDA
def test_lane_failure_is_local_and_remote_failure_agreement_is_synchronous() -> None:
    owner = ReplicaModuleOwner(_modules(3))
    group = ReplicaOptimizerGroup(owner, OptimizerConfig())

    class _Agreement:
        def __init__(self) -> None:
            self.active: torch.Tensor | None = None

        def agree_failures(self, failed: torch.Tensor) -> torch.Tensor:
            result = failed.clone()
            result[2] = True
            return result

        def sum_gradients(self, gradients: torch.Tensor, active: torch.Tensor) -> None:
            self.active = active.clone()
            gradients.masked_fill_(~active[:, None], 0)

    agreement = _Agreement()
    scheduler = ReplicaLearnerScheduler(group, distributed=agreement)  # type: ignore[arg-type]
    calls = []

    def backward(lane: int, module: torch.nn.Module, _optimizer: DenseAdam) -> None:
        calls.append(lane)
        if lane == 1:
            raise RuntimeError("injected local lane failure")
        module("toy", torch.ones((2, 3), device="cuda"))["prediction"].sum().backward()

    result = scheduler.run_step((0, 1, 2), backward)
    assert calls == [0, 1, 2]
    assert set(result.failures) == {1, 2}
    assert result.optimizer_steps[0] is not None
    assert result.optimizer_steps[1] is None
    assert result.optimizer_steps[2] is None
    assert group.versions.tolist() == [1, 0, 0]
    assert agreement.active is not None
    assert agreement.active.tolist() == [True, False, False]


@MULTI_CUDA
def test_distributed_nccl_sum_and_asymmetric_failure_agreement(
    tmp_path: Path,
) -> None:
    initialization = tmp_path / "nccl-init"
    mp.spawn(
        _distributed_sum_worker,
        args=(2, str(initialization)),
        nprocs=2,
        join=True,
    )


@CUDA
def test_replica_checkpoint_round_trip_preserves_every_lane(tmp_path: Path) -> None:
    modules = ReplicaModuleOwner(_modules(2))
    optimizers = ReplicaOptimizerGroup(
        modules,
        OptimizerConfig(learning_rate=1.0e-3, weight_decay=0.01),
        scalers=(LossScaler(), LossScaler()),
    )
    scheduler = ReplicaLearnerScheduler(optimizers)
    value = torch.randn((2, 4, 3), device="cuda")
    scheduler.run_step(
        (0, 1),
        lambda lane, module, _optimizer: (
            module("toy", value[lane])["prediction"].square().mean().backward()
        ),
    )
    generators = tuple(
        torch.Generator(device="cuda").manual_seed(900 + lane) for lane in range(2)
    )
    runs = (
        RunState(
            "replica-lane-0",
            iteration=3,
            metrics_prefix=[{"iteration": value} for value in range(3)],
        ),
        RunState(
            "replica-lane-1",
            iteration=3,
            metrics_prefix=[{"iteration": value} for value in range(3)],
        ),
    )
    scientific = ({"lane": 0, "mode": "a"}, {"lane": 1, "mode": "b"})
    model_before = copy.deepcopy(modules.state_dict())
    optimizer_before = copy.deepcopy(optimizers.state_dict())
    rng_before = tuple(generator.get_state() for generator in generators)
    path = tmp_path / "replicas.pt"
    save_replica_checkpoint(
        path,
        modules=modules,
        optimizers=optimizers,
        run_states=runs,
        scientific_configs=scientific,
        execution_config={"hardware": "test"},
        generators=generators,
    )
    with torch.no_grad():
        modules.parameters["weight"].zero_()
    optimizers.exp_avg.fill_(17)
    for generator in generators:
        generator.manual_seed(1)
    _references, restored_runs, execution = load_replica_checkpoint(
        path,
        modules=modules,
        optimizers=optimizers,
        expected_scientific_configs=scientific,
        generators=generators,
        restore_ambient_rng=False,
    )
    assert execution == {"hardware": "test"}
    assert tuple(run.run_id for run in restored_runs) == (
        "replica-lane-0",
        "replica-lane-1",
    )
    for lane in range(2):
        for name, expected in model_before["lanes"][lane].items():
            torch.testing.assert_close(
                modules.state_dict()["lanes"][lane][name], expected
            )
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizers.state_dict()["lanes"][lane][name],
                optimizer_before["lanes"][lane][name],
            )
        assert torch.equal(generators[lane].get_state(), rng_before[lane])


def test_learner_dtype_contract_is_explicit() -> None:
    contract = LearnerDTypes(torch.bfloat16)
    assert contract.model == torch.bfloat16
    assert contract.control == torch.float32
    assert contract.posterior == torch.float32
    assert contract.gradient == torch.float32
    assert contract.optimizer == torch.float32
    with pytest.raises(ValueError, match="posterior"):
        LearnerDTypes(torch.float32, posterior=torch.bfloat16)
