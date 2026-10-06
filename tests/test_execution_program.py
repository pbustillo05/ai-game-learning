"""Precision metadata and CUDA compilation/capture lifecycle contracts."""

import pytest
import torch

from pbg.core.execution_program import TensorProgramRunner
from pbg.core.precision import resolve_precision
from pbg.core.tensor_ops import feature_cat

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def test_native_bf16_policy_rejects_sm75_without_emulation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(ValueError, match="native BF16"):
        resolve_precision("bf16", torch.device("cuda:0"))
    manifest = resolve_precision("fp16", torch.device("cuda:0")).manifest()
    assert manifest["activation"] == "float16"
    assert manifest["master"] == manifest["gradient"] == "float32"
    assert manifest["dynamic_loss_scaling"] is True


def test_native_bf16_policy_records_dtype_islands(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    manifest = resolve_precision("bf16", torch.device("cuda:0")).manifest()
    assert manifest["compute"] == manifest["activation"] == "bfloat16"
    assert manifest["incidence_accumulation"] == "float32"
    assert manifest["dynamic_loss_scaling"] is False
    assert manifest["backward_autocast"] == "off"


@CUDA
@pytest.mark.parametrize("precision", (torch.float16, torch.bfloat16))
def test_feature_assembly_does_not_promote_activation_bank(precision):
    if precision is torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("native BF16 is unavailable")
    with torch.autocast("cuda", dtype=precision):
        result = feature_cat((torch.ones(2, 4, device="cuda", dtype=precision),
                              torch.ones(2, 1, device="cuda", dtype=torch.float32)))
    assert result.dtype is precision
    assert result.shape == (2, 5)


@CUDA
@pytest.mark.parametrize("mode", ("compiled", "captured"))
def test_first_replay_warmup_reset_and_live_controls(mode):
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4], device="cuda"))
    parameter.grad = torch.zeros_like(parameter)
    pointer = parameter.grad.data_ptr()
    runner = TensorProgramRunner(mode=mode, byte_budget=256 << 20)

    def program(value, control):
        prediction = (parameter * value).sum()
        return prediction.detach(), prediction.square() * control

    expected = torch.zeros_like(parameter)
    for call in range(5):
        if call == 3:
            parameter.grad.zero_()
            expected.zero_()
            with torch.no_grad():
                parameter.add_(0.03)
        value = torch.tensor([0.3 + call, -0.6], device="cuda")
        control = torch.tensor(0.5 + call, device="cuda")
        expected.add_(2 * (parameter.detach() * value).sum() * value * control)
        output, = runner.run("two-features", program, (value, control), mutable=(parameter.grad,))
        torch.testing.assert_close(output, (parameter.detach() * value).sum())
        torch.testing.assert_close(parameter.grad, expected, rtol=3e-5, atol=3e-6)
        assert parameter.grad.data_ptr() == pointer
    assert len(runner.entries) == 1
    assert runner.replays == (4 if mode == "captured" else 0)
    if mode == "captured":
        entry = next(iter(runner.entries.values()))
        pool = sum(segment["total_size"] for segment in torch.cuda.memory_snapshot(
            mempool_id=entry.graph.pool(), include_traces=False))
        assert entry.resident_bytes >= pool > 0


@CUDA
def test_denied_shape_stays_uncaptured_and_cache_is_bounded():
    parameter = torch.nn.Parameter(torch.tensor(1., device="cuda"))
    parameter.grad = torch.zeros_like(parameter)
    runner = TensorProgramRunner(mode="captured", max_entries=1, byte_budget=0)

    def program(value):
        output = parameter * value
        return output.detach(), output

    for _ in range(4):
        runner.run("scalar", program, (torch.tensor(2., device="cuda"),), mutable=(parameter.grad,))
    assert len(runner.entries) == 0
    assert len(runner.denied) == 1
    assert runner.replays == 0
    torch.testing.assert_close(parameter.grad, torch.tensor(8., device="cuda"))


@CUDA
def test_oversized_graph_retains_warmed_compilation_and_live_gradients():
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4], device="cuda"))
    parameter.grad = torch.zeros_like(parameter)
    pointer = parameter.grad.data_ptr()
    # Staging and rollback fit, while the graph's allocator segment exceeds
    # this budget. Exercise actual post-capture denial, not its staging guard.
    runner = TensorProgramRunner(mode="captured", max_entries=1, byte_budget=64 << 10)

    def program(value, control):
        prediction = parameter * value
        return prediction.detach(), prediction.square().sum() * control

    expected = torch.zeros_like(parameter)
    for call in range(4):
        if call == 2:
            with torch.no_grad():
                parameter.add_(0.03)
        value = torch.tensor([0.3 + call, -0.6], device="cuda")
        control = torch.tensor(0.5 + call, device="cuda")
        expected.add_(2 * parameter.detach() * value.square() * control)
        output, = runner.run("oversized", program, (value, control), mutable=(parameter.grad,))
        torch.testing.assert_close(output, parameter.detach() * value)
        torch.testing.assert_close(parameter.grad, expected, rtol=3e-5, atol=3e-6)
        assert parameter.grad.data_ptr() == pointer
        if call:
            assert len(runner.entries) == 1
            entry = next(iter(runner.entries.values()))
            assert entry.graph is None and entry.static == () and entry.outputs == ()
            assert entry.resident_bytes == runner.resident_bytes == 0
            assert len(runner.denied) == 1
    assert runner.replays == 0
    assert runner.manifest()["captured_entries"] == 0
    assert runner.calls == runner.uncaptured == 4
    parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match="lost its gradient/buffer addresses"):
        runner.run("oversized", program, (value, control), mutable=(parameter.grad,))


def test_config_records_exact_family_caps_graph_policy_and_h100_reserve(monkeypatch):
    from pbg.config import load_config_mapping
    from pbg.games.poker.queued import capacity_compiler
    from tests.test_p0_config import _raw

    monkeypatch.setattr(capacity_compiler, "_device_total_memory", lambda device: 80_000_000_000)
    raw = _raw()
    raw["run"]["execution"]["microbatch"] = {"learner_rows": {"pre": 32, "post_2": 8}}
    raw["run"]["execution"]["tensor_program"] = {"mode": "captured", "cache_entries": 4, "cache_bytes": 1 << 30}
    cfg = load_config_mapping(raw)
    assert cfg.execution.learner_family_rows == {"pre": 32, "post_2": 8}
    assert cfg.execution.manifest.hardware.reserve_bytes >= 10 << 30
    assert cfg.execution.mapping()["tensor_program"]["mode"] == "captured"
    assert cfg.execution.mapping()["precision_policy"]["master"] == "float32"
    rebuilt = load_config_mapping(cfg.source_mapping())
    assert cfg.execution.mapping() == rebuilt.execution.mapping()


def test_token_admission_counts_full_postflop_storage():
    from pbg.games.poker.model.capacity import max_policy_partition_sequence_tokens
    bounds = max_policy_partition_sequence_tokens(n_players=2, stack_bb=32, rank_count=13, k_act=7)
    assert bounds["post_2"] > 2 * 1326


def test_optimizer_lane_does_not_retain_owner():
    import weakref
    from pbg.core.optimization import DenseAdam, DenseAdamLane
    owner = DenseAdam.__new__(DenseAdam)
    lane = DenseAdamLane(owner, 0)
    owner.lanes = (lane,)
    reference = weakref.ref(owner)
    del owner
    assert reference() is None
    with pytest.raises(ReferenceError):
        _ = lane.owner.configs


def test_reference_cache_budget_is_shared_across_resident_lanes(monkeypatch):
    from types import SimpleNamespace
    from pbg.core.iteration import IterationEngine
    import pbg.core.iteration as iteration
    count, total = 16, 1003
    engine = IterationEngine.__new__(IterationEngine)
    engine.config = SimpleNamespace(lanes=[SimpleNamespace(reference_operation=object()) for _ in range(count)],
        cache_reference_outputs=True, learner_family_rows={"pre": 32, "post_2": 8},
        schedule=SimpleNamespace(batch_size=256), precision="bf16")
    owners = SimpleNamespace(plan=SimpleNamespace(layout=SimpleNamespace(reference_cache_bytes=total),
                                                  manifest=SimpleNamespace(learner_sigma_pack_cap=32)))
    store = SimpleNamespace(rows=512, sealed_inputs=SimpleNamespace(columns={
        "model_input": torch.empty(512, 1, device="meta"), "owner": torch.empty(512, device="meta")}))
    budgets, caps = [], []
    def build(*args, **kwargs):
        budgets.append(kwargs["byte_budget"])
        caps.append(kwargs["pack_cap"])
        return SimpleNamespace(allocated_bytes=kwargs["byte_budget"])
    monkeypatch.setattr(iteration, "build_reference_output_cache", build)
    for lane in range(count):
        sampler = SimpleNamespace(statistical_counts=None, statistical_presentations=1024)
        assert engine._reference_caches(owners, lane, {"post_2": store}, sampler=sampler)
    assert sum(budgets) == total
    assert max(budgets) - min(budgets) == 1
    assert caps == [8] * count
