"""Production-width eager native-precision parity and lane-isolation probes."""
import gc

import pytest
import torch

from pbg.core.replica import ReplicaModuleOwner
from pbg.games.poker.model.preflop import PokerModelConfig
from pbg.games.poker.queued.hooks import NLHEGeometry, PokerStateMachine
from pbg.games.poker.queued.model import QueuedPokerModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.mark.parametrize('players,lanes', [(2, 4), (6, 2)])
@pytest.mark.parametrize('family', ['pre', 'post'])
@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_full_width_cohort_parity_and_lane_isolation(players, lanes, family, selected, dtype):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('native BF16 unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(8701)
    geometry = NLHEGeometry(players, 32, 13)
    config = PokerModelConfig(d_model=128, n_heads=4, layers=6, m_raise=4,
        k_act=7, max_players=players, class_embedding_dim=32, ffn_hidden=512,
        folder_beta_rank=16)
    models = [QueuedPokerModel(geometry, config).cuda() for _ in range(lanes)]
    owner = ReplicaModuleOwner(models)
    cohort = owner.cohort_module(0, lanes)
    machine = PokerStateMachine(geometry)
    state = machine.initial_roots(lanes, seed=8702, iteration=0, retry_epoch=0, device='cuda')
    if family == 'post':
        state.street.fill_(1)
        state.board_cards[:, :3] = torch.tensor([0, 5, 10], device='cuda', dtype=torch.uint8)
    partition = 'pre' if family == 'pre' else f'post_{players}'
    inputs = machine.direct_model_input(partition, state).to(dtype)
    private = None
    if selected:
        private = torch.arange(lanes, device='cuda', dtype=torch.int64)
        if family == 'post':
            _, mask = models[0]._public(state, machine.empty_history(lanes, 'cuda'))
            private = mask.to(torch.int64).argmax(1)

    def forward(model, value, chosen):
        with torch.autocast('cuda', dtype=dtype):
            output = model.forward_partition(partition, value, chosen)
            tensors = (output.value, *output.policy_raw.values())
            loss = sum(x.float().square().reshape(value.shape[0], -1).mean(1).sum() for x in tensors)
        return tensors, loss

    values, objective = forward(cohort, inputs, private)
    objective.backward()
    del objective
    saved = owner.gradients.clone()
    detached = tuple(x.detach().clone() for x in values)
    del values
    rtol, atol = ((3e-2, 3e-3) if dtype == torch.bfloat16 else (8e-3, 8e-4))
    for lane, model in enumerate(models):
        result, loss = forward(model, inputs[lane:lane+1], None if private is None else private[lane:lane+1])
        loss.backward()
        for actual, expected in zip(detached, result):
            torch.testing.assert_close(actual[lane:lane+1], expected, rtol=rtol, atol=atol)
        for name, parameter in model.named_parameters():
            begin, end, shape = owner.parameter_layout[name]
            reference = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            torch.testing.assert_close(saved[lane, begin:end].view(shape), reference, rtol=rtol, atol=atol)
        del result, loss
    assert bool(torch.isfinite(saved).all())
    with torch.no_grad():
        owner.master[0].add_(0.001)
        changed, _ = forward(cohort, inputs, private)
    for before, after in zip(detached, changed):
        torch.testing.assert_close(before[1:], after[1:], rtol=0, atol=0)
    assert not torch.equal(detached[0][0], changed[0][0])
    owner.assert_zero_copy()
    del owner, cohort, models, saved, detached, changed
    gc.collect()
    torch.cuda.synchronize()


@pytest.mark.parametrize('lanes', [2, 16])
def test_strict_native_replica_resume_is_bitwise_equal(tmp_path, lanes):
    from pbg.config import load_config_mapping
    from pbg.h100_pilot import Arm, pilot_config_mapping
    from pbg.runtime import TrainingSession

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('native BF16 unavailable')
    raw = pilot_config_mapping(Arm('resume', 'smoke', 'normal', lanes, 4, None,
        rollout_pack=4, completion_pack=4, learner_pack=4), output_dir=tmp_path / 'runtime')
    raw['run']['reproducibility'] = 'strict'
    raw['run']['execution']['tensor_program'] = {'mode': 'eager', 'cache_entries': 8, 'cache_bytes': 0}
    raw['scientific']['objective']['reference_kl']['refresh'] = {'effective_hands': 0.5}
    session = TrainingSession(load_config_mapping(raw))
    checkpoint = tmp_path / 'resume.pt'
    try:
        assert session.backend_policy['deterministic_algorithms'] is True
        assert session.backend_policy['matmul_fp32_precision'] == 'ieee'
        session.step()
        assert all(reference.reference_iteration == 1 for reference in session.reference_states)
        session.checkpoint(checkpoint)
        addresses = tuple(parameter.data_ptr() for parameter in session.modules.parameters.values())
        session.step()
        expected = session.modules.master.detach().clone()
        logical_work = tuple(dict(state.logical_work) for state in session.run_states)
        session._resume(checkpoint)
        session.step()
        assert tuple(dict(state.logical_work) for state in session.run_states) == logical_work
        assert all(reference.reference_iteration == 2 for reference in session.reference_states)
        assert addresses == tuple(parameter.data_ptr() for parameter in session.modules.parameters.values())
        torch.testing.assert_close(session.modules.master, expected, rtol=0, atol=0)
        session.modules.assert_zero_copy()
        session.reference_modules.assert_zero_copy()
    finally:
        session.close()
        checkpoint.unlink(missing_ok=True)
