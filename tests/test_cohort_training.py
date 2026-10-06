"""Numerical gates for the sole logical-step replica learner scheduler."""

import copy
from dataclasses import replace

import pytest
import torch

from pbg.core.contracts import mode_schema
from pbg.core.learner import LearnerOperation, LossConfig, ValueLossConfig
from pbg.core.optimization import DenseAdam, OptimizerConfig
from pbg.core.replica import ReplicaModuleOwner, ReplicaOptimizerGroup
from pbg.core.training import LearnerIteration, PreparedLearner, optimize_prepared
from tests.test_replica_identity import _LearnerToy, _learner_batch, _learner_policy

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


class _Schedule:
    def __init__(self, batches, steps=2):
        self.optimizer_steps = steps
        self._batches = batches

    def batches(self):
        yield from self._batches


def _work(model, optimizer, batches, loss):
    learner = LearnerOperation(mode_schema("self_play", "sequential", "sampled"),
                               loss, model, _learner_policy)
    return PreparedLearner(LearnerIteration(learner=learner, optimizer=optimizer,
                                           physical_batch_cap=2), _Schedule(batches))


@CUDA
@pytest.mark.parametrize("count", (1, 2, 4))
def test_cohort_scheduler_matches_independent_same_program(count, monkeypatch):
    import pbg.core.training as training

    sources = [_LearnerToy().cuda() for _ in range(count)]
    for lane, model in enumerate(sources):
        with torch.no_grad():
            model.logit.add_(lane * 0.04)
    independent = [copy.deepcopy(model) for model in sources]
    owner = ReplicaModuleOwner(sources)
    config = OptimizerConfig(learning_rate=0.003, weight_decay=0.02, grad_clip=0.4)
    group = ReplicaOptimizerGroup(owner, config)
    optimizers = [DenseAdam(model, config) for model in independent]
    schedules, losses = [], []
    for lane in range(count):
        batch = _learner_batch()
        batch.fields["model_input"].mul_(lane + 1)
        schedules.append([replace(batch, input_type=family, logical_step=step,
                                  statistical_weight=weight)
                          for step in range(2) for family, weight in (("pre", 0.3), ("post", 0.7))])
        losses.append(LossConfig(value=ValueLossConfig(coefficient=0.2 + lane)))
    works = [_work(owner.lane_module(lane), group.lane_stepper(lane, distributed=None),
                   schedules[lane], losses[lane]) for lane in range(count)]
    calls = []
    original = training._run_learner_cohort

    def observed(works, packets, **kwargs):
        calls.append((len(works), packets[0][1].input_type, packets[0][1].rows))
        return original(works, packets, **kwargs)

    monkeypatch.setattr(training, "_run_learner_cohort", observed)
    actual = optimize_prepared(works, iteration=0, optimizer=group)
    assert calls == [(count, family, 2) for _ in range(2) for family in ("pre", "pre", "post", "post")]
    for lane in range(count):
        expected = _work(independent[lane], optimizers[lane], schedules[lane], losses[lane]).optimize(iteration=0)
        assert actual[lane].statistical_presentations == expected.statistical_presentations == 16
        for left, right in zip(actual[lane].losses, expected.losses, strict=True):
            for name in left:
                assert left[name] == pytest.approx(right[name], rel=3e-5, abs=3e-6)
        for left, right in zip(owner.lane_module(lane).parameters(), independent[lane].parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(group.optimizer.first[lane], optimizers[lane].first[0])
        torch.testing.assert_close(group.optimizer.second[lane], optimizers[lane].second[0])
    assert group.versions.tolist() == [2] * count
    owner.assert_zero_copy()


@CUDA
def test_mixed_family_fifo_never_pads_or_skips_a_lane(monkeypatch):
    import pbg.core.training as training

    owner = ReplicaModuleOwner([_LearnerToy().cuda() for _ in range(2)])
    group = ReplicaOptimizerGroup(owner, OptimizerConfig())
    batch = _learner_batch()
    schedules = [[replace(batch, input_type=family, logical_step=step)
                  for step, family in enumerate(families)]
                 for families in (("pre", "post"), ("post", "pre"))]
    works = [_work(owner.lane_module(lane), group.lane_stepper(lane, distributed=None),
                   schedules[lane], LossConfig()) for lane in range(2)]
    calls = []
    original = training._run_learner_cohort

    def observed(works, packets, **kwargs):
        calls.append((tuple(kwargs["lanes"]), packets[0][1].input_type, packets[0][1].rows))
        return original(works, packets, **kwargs)

    monkeypatch.setattr(training, "_run_learner_cohort", observed)
    optimize_prepared(works, iteration=0, optimizer=group)
    assert calls == [((lane,), family, 2)
                     for pairs in (((0, "pre"), (1, "post")), ((0, "post"), (1, "pre")))
                     for lane, family in pairs for _ in range(2)]
    assert group.versions.tolist() == [2, 2]
