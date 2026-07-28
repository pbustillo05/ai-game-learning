from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pbg.games.poker.queued import (
    NLHEGeometry,
    PokerReplicaInference,
    QueuedPokerModel,
    ReplicaPokerIterationGroup,
)
from pbg.queued import (
    DistributedSUM,
    FunctionalReplicaInference,
    LearnerDTypes,
    LearnerOperation,
    LossConfig,
    LossScaler,
    ModelEvaluation,
    OptimizerConfig,
    PolicyEvaluation,
    ReplicaLearnerScheduler,
    ReplicaMemoryReport,
    ReplicaModuleOwner,
    ReplicaOptimizerGroup,
    RunState,
    SequentialLearnerBatch,
    load_replica_checkpoint,
    mode_schema,
    save_replica_checkpoint,
)
from pbg.queued.optimization import DenseAdam
from pbg.queued.persistence import convert_legacy_lane_artifacts

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="queued replica execution is CUDA-only"
)
MULTI_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="distributed queued learner validation requires two CUDA devices",
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
        (2, rows, geometry.direct_model_input_width),
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
    assert actual.value.shape == (2, rows, geometry.private_hands)
    assert float(actual.value[0, 0, 0]) != float(actual.value[1, 0, 0])
    lane_operation = inference.lane_operation(2)
    lane_value = lane_operation("pre", model_input[0], private[0], owner[0]).value
    torch.testing.assert_close(lane_value, expected[0])


@CUDA
def test_production_poker_group_binds_vmap_rollout_and_ordinary_learner_lanes() -> None:
    geometry = NLHEGeometry(2, 8.0, 4)
    source = [
        QueuedPokerModel(
            geometry,
            _PokerPartition(geometry.rank_count**2, 0.3 + lane),
            _PokerPartition(geometry.private_hands, 0.4 + lane),
        ).cuda()
        for lane in range(2)
    ]
    inference = PokerReplicaInference(source)
    optimizers = ReplicaOptimizerGroup(
        inference.modules,
        OptimizerConfig(learning_rate=1.0e-3),
    )
    schemas = (
        mode_schema("self_play", "branched.fixed", "sampled"),
        mode_schema("self_play", "branched.KKT", "sampled"),
    )

    class _Iteration:
        def __init__(self, lane: int) -> None:
            self.model = inference.modules.lane_module(lane)
            self.optimizer = optimizers.optimizers[lane]
            self.mode = schemas[lane]
            self.rollout_operation = inference.lane_operation(lane)
            self.model_operation = self.model.learner_operation
            self.calls = 0

        def run(self, *, iteration: int) -> ModelEvaluation:
            del iteration
            self.calls += 1
            rows = 2
            model_input = torch.zeros(
                (rows, geometry.direct_model_input_width), device="cuda"
            )
            private = torch.zeros(
                (rows, geometry.players), dtype=torch.int32, device="cuda"
            )
            owner = torch.zeros(rows, dtype=torch.uint8, device="cuda")
            rollout = self.rollout_operation("pre", model_input, private, owner)
            self.optimizer.zero_grad()
            learned = self.model_operation("pre", model_input, private, owner)
            learned.value.mean().backward()
            self.optimizer.step()
            return rollout

    iterations = (_Iteration(0), _Iteration(1))
    group = ReplicaPokerIterationGroup(
        iterations,
        inference=inference,
        optimizers=optimizers,
    )
    result = group.run(iteration=0, active_lanes=(1,))
    assert result.failures == {}
    assert result.results[0] is None
    assert isinstance(result.results[1], ModelEvaluation)
    assert iterations[0].calls == 0
    assert iterations[1].calls == 1
    assert result.versions == (0, 1)


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
        RunState("replica-lane-0", iteration=3),
        RunState("replica-lane-1", iteration=3),
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
    restored_runs, execution = load_replica_checkpoint(
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


def test_distributed_and_replica_lane_artifacts_use_the_sole_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = (tmp_path / "source-0.pt", tmp_path / "source-1.pt")
    for source in sources:
        source.write_bytes(b"approved legacy lane")
    calls: list[tuple[Path, Path]] = []

    def convert(source: Path, destination: Path) -> dict[str, object]:
        calls.append((Path(source), Path(destination)))
        destination.write_bytes(b"queued lane")
        return {
            "source_sha256": f"{len(calls):064x}",
            "result_sha256": f"{len(calls) + 10:064x}",
        }

    monkeypatch.setattr(
        "pbg.queued.persistence.convert_legacy_sequential",
        convert,
    )
    destination = tmp_path / "converted"
    result = convert_legacy_lane_artifacts(
        sources,
        destination,
        execution_kind="replica",
    )
    assert result["lane_count"] == 2
    assert result["execution_kind"] == "replica"
    assert calls == [
        (sources[0], destination / "lane_000.pt"),
        (sources[1], destination / "lane_001.pt"),
    ]
    assert (destination / "manifest.json").is_file()
