"""No padding or cross-lane update changes when exact FIFO tails differ."""
import copy
from dataclasses import replace

import pytest
import torch

from pbg.core.learner import LossConfig, ValueLossConfig
from pbg.core.optimization import DenseAdam, OptimizerConfig
from pbg.core.replica import ReplicaModuleOwner, ReplicaOptimizerGroup
from pbg.core.training import _ready_learner_cohort, optimize_prepared
from tests.test_cohort_training import _work
from tests.test_replica_identity import _LearnerToy, _learner_batch


def test_ready_cohort_prioritizes_fifo_family_then_largest_contiguous_run():
    lanes = (0, 1, 2, 3, 4)
    keys = {0: ('post', 8), 1: ('pre', 1), 2: ('pre', 2), 3: ('pre', 2), 4: ('post', 8)}
    ranks = {0: 1, 1: 0, 2: 0, 3: 0, 4: 1}
    batched = dict.fromkeys(lanes, True)
    assert _ready_learner_cohort(list(lanes), lanes=lanes, keys=keys,
                                family_ranks=ranks, batched=batched) == [2, 3]
    # A skipped resident lane cannot be padded or hidden inside a cohort.
    assert _ready_learner_cohort([0, 4], lanes=lanes, keys=keys,
                                family_ranks=ranks, batched=batched) == [0]
    batched[2] = False
    assert _ready_learner_cohort([2, 3], lanes=lanes, keys=keys,
                                family_ranks=ranks, batched=batched) == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('count', (2, 4, 16))
def test_uneven_family_tails_keep_postflop_cohorts_and_independent_updates(count, monkeypatch):
    import pbg.core.training as training

    torch.manual_seed(771)
    sources = [_LearnerToy().cuda() for _ in range(count)]
    for lane, model in enumerate(sources):
        with torch.no_grad():
            model.logit.add_(lane * 0.01)
    independent = [copy.deepcopy(model) for model in sources]
    owner = ReplicaModuleOwner(sources)
    config = OptimizerConfig(learning_rate=0.003, weight_decay=0.02, grad_clip=0.4)
    group = ReplicaOptimizerGroup(owner, config)
    optimizers = [DenseAdam(model, config) for model in independent]
    schedules, losses = [], []
    base = _learner_batch()
    for lane in range(count):
        batches = []
        for step in range(2):
            for family, rows in [('pre', 97+lane), ('post', 159-lane)]:
                indices = torch.arange(rows, device='cuda', dtype=torch.int64) % base.rows
                batches.append(replace(base, input_type=family, indices=indices.clone(),
                    fields={name: value.index_select(0, indices).clone() for name, value in base.fields.items()},
                    logical_step=step, logical_rows=256, statistical_weight=rows/256))
        schedules.append(batches)
        losses.append(LossConfig(value=ValueLossConfig(coefficient=0.2+lane)))
    works = [_work(owner.lane_module(lane), group.lane_stepper(lane, distributed=None),
                   schedules[lane], losses[lane]) for lane in range(count)]
    # Use the same production launch caps, while retaining test-local losses.
    for work in works:
        work.iteration.physical_batch_cap = {'pre': 32, 'post': 8}
        work.sampler.stores = dict.fromkeys(('pre', 'post'))
    calls, observed = [], [[] for _ in range(count)]
    original = training._run_learner_cohort

    def trace(works, packets, **kwargs):
        calls.append((len(works), packets[0][1].input_type, packets[0][1].rows))
        for lane, packet in zip(kwargs['lanes'], packets, strict=True):
            observed[lane].append((packet[0], packet[1].input_type, packet[1].indices.clone()))
        return original(works, packets, **kwargs)

    monkeypatch.setattr(training, '_run_learner_cohort', trace)
    actual = optimize_prepared(works, iteration=0, optimizer=group)
    monkeypatch.setattr(training, '_run_learner_cohort', original)
    assert sum(width == count and family == 'post' for width, family, _ in calls) >= 2 * 18
    for lane in range(count):
        expected_work = _work(independent[lane], optimizers[lane], schedules[lane], losses[lane])
        expected_work.iteration.physical_batch_cap = {'pre': 32, 'post': 8}
        expected = expected_work.optimize(iteration=0)
        assert actual[lane].statistical_presentations == expected.statistical_presentations == 512
        for step in range(2):
            for family in ('pre', 'post'):
                gathered = torch.cat([indices for batch_step, batch_family, indices in observed[lane]
                                      if batch_step == step and batch_family == family])
                reference = next(batch.indices for batch in schedules[lane]
                                 if batch.logical_step == step and batch.input_type == family)
                torch.testing.assert_close(gathered, reference, rtol=0, atol=0)
        for left, right in zip(owner.lane_module(lane).parameters(), independent[lane].parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(group.optimizer.first[lane], optimizers[lane].first[0])
        torch.testing.assert_close(group.optimizer.second[lane], optimizers[lane].second[0])
        for left, right in zip(actual[lane].losses, expected.losses, strict=True):
            for name in left:
                assert left[name] == pytest.approx(right[name], rel=3e-5, abs=3e-6)
    assert group.versions.tolist() == [2] * count
    owner.assert_zero_copy()
