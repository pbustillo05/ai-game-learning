from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pbg.core.completion import BatchedCompletionStats
from pbg.core.training import _migrate_value_head_state, _scale_completion_utility
from pbg.core.utility import utility_conversion, utility_scale


def _cfg(convention: str, stack: float = 32.0) -> SimpleNamespace:
    return SimpleNamespace(
        game=SimpleNamespace(stack_bb=stack),
        loss=SimpleNamespace(utility_normalization=convention),
    )


def test_stack_utility_scale_and_conversion_are_explicit() -> None:
    raw = _cfg("none")
    normalized = _cfg("stack")
    assert utility_scale(raw) == 1.0
    assert utility_scale(normalized) == pytest.approx(1.0 / 32.0)
    assert utility_conversion(raw, normalized) == pytest.approx(1.0 / 32.0)
    assert utility_conversion(normalized, raw) == pytest.approx(32.0)


def test_completion_scaling_changes_values_only() -> None:
    posterior = torch.tensor([[[0.25, 0.75]]])
    values = torch.tensor([[[-16.0, 32.0]]])
    completion = BatchedCompletionStats(
        posterior, values, metadata={"completion_mean_n_eff": 2.0}
    )
    scaled = _scale_completion_utility(completion, _cfg("stack"))
    assert scaled.p_hat is posterior
    torch.testing.assert_close(scaled.v_hat, values / 32.0)
    assert scaled.metadata["completion_mean_n_eff"] == 2.0
    assert scaled.metadata["training_utility_scale"] == pytest.approx(1.0 / 32.0)


def test_checkpoint_migration_changes_only_final_value_affines() -> None:
    state = {
        "preflop.policy.weight": torch.tensor([3.0]),
        "postflop.policy.bias": torch.tensor([4.0]),
        "preflop.value_head.affine.weight": torch.tensor([[32.0, -16.0]]),
        "preflop.value_head.affine.bias": torch.tensor([8.0]),
        "postflop.value_head.affine.weight": torch.tensor([[64.0, 32.0]]),
        "postflop.value_head.affine.bias": torch.tensor([-4.0]),
    }
    migrated, factor, converted = _migrate_value_head_state(
        state, _cfg("none"), _cfg("stack")
    )
    assert factor == pytest.approx(1.0 / 32.0)
    assert len(converted) == 4
    assert migrated["preflop.policy.weight"] is state["preflop.policy.weight"]
    assert migrated["postflop.policy.bias"] is state["postflop.policy.bias"]
    torch.testing.assert_close(
        migrated["preflop.value_head.affine.weight"],
        torch.tensor([[1.0, -0.5]]),
    )
    torch.testing.assert_close(
        migrated["postflop.value_head.affine.bias"], torch.tensor([-0.125])
    )
    torch.testing.assert_close(
        state["preflop.value_head.affine.weight"],
        torch.tensor([[32.0, -16.0]]),
    )
