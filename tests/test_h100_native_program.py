"""Compile/capture gates for the actual shared poker tensor program."""

import copy

import pytest
import torch

from pbg.core.execution_program import TensorProgramRunner
from pbg.core.replica import ReplicaModuleOwner
from pbg.games.poker.model.preflop import PokerModelConfig
from pbg.games.poker.queued.hooks import NLHEGeometry, PokerStateMachine
from pbg.games.poker.queued.model import QueuedPokerModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ("compiled", "captured"))
@pytest.mark.parametrize("family", ("pre", "post_2"))
@pytest.mark.parametrize("lanes", (1, 2))
def test_native_bf16_program_replay_preserves_gradients_and_live_weights(mode, family, lanes):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("native BF16 unavailable")
    torch.manual_seed(621)
    geometry = NLHEGeometry(2, 8, 4)
    config = PokerModelConfig(d_model=16, n_heads=4, layers=1, m_raise=2,
                              k_act=5, max_players=2, class_embedding_dim=4,
                              ffn_hidden=24, folder_beta_rank=0)
    source = QueuedPokerModel(geometry, config).cuda()
    owner = ReplicaModuleOwner([copy.deepcopy(source) for _ in range(lanes)])
    module = owner.cohort_module(0, lanes)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(lanes, seed=622, iteration=0, retry_epoch=0, device="cuda")
    if family == "post_2":
        state.street.fill_(1)
        state.board_cards[:, :3] = torch.tensor([36, 41, 46], device="cuda", dtype=torch.uint8)
    inputs = machine.direct_model_input(family, state).to(torch.bfloat16)
    runner = TensorProgramRunner(mode=mode, byte_budget=256 << 20)

    def program(value, control):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module.forward_natural_partition(family, value)
            objective = output.value.float().square().mean() * control
        return output.value.detach(), objective

    pointers = tuple(parameter.grad.data_ptr() for parameter in module.parameters())
    for call in range(4):
        if call == 3:
            # Reference refresh/checkpoint reset must keep captured storage live.
            with torch.no_grad():
                owner.master.add_(0.001)
        control = torch.tensor(0.5 + call, device="cuda")
        owner.gradients.zero_()
        expected, loss = program(inputs, control)
        loss.backward()
        # Release the mathematical reference graph before side-stream capture;
        # retaining it also retains default-stream AccumulateGrad nodes.
        del loss
        gradients = owner.gradients.clone()
        owner.gradients.zero_()
        actual, = runner.run(family, program, (inputs, control), mutable=(owner.gradients,))
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-3)
        torch.testing.assert_close(owner.gradients, gradients, rtol=3e-2, atol=3e-3)
        assert bool(torch.isfinite(owner.gradients).all())
        assert pointers == tuple(parameter.grad.data_ptr() for parameter in module.parameters())
    assert len(runner.entries) == 1
    assert runner.replays == (3 if mode == "captured" else 0)
    owner.assert_zero_copy()
