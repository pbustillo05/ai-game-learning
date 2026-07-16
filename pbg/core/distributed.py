"""Estimator-correct distributed decision-mass normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class WeightedDecisionStats:
    """One rank/stratum's frozen weighted numerator and decision mass."""

    numerator: Tensor
    mass: Tensor


def weighted_decision_stats(loss: Tensor, weight: Tensor) -> WeightedDecisionStats:
    """Return ``(sum_s weight_s loss_s, sum_s weight_s)`` without normalization."""

    if loss.ndim != 1 or weight.ndim != 1 or loss.shape != weight.shape:
        raise ValueError("loss and weight must be matching one-dimensional tensors")
    if loss.device != weight.device:
        raise ValueError("loss and weight must be on the same device")
    if not bool(torch.isfinite(loss).all()):
        raise ValueError("decision losses must be finite")
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("decision weights must be finite and nonnegative")
    return WeightedDecisionStats(
        numerator=(loss * weight).sum(),
        mass=weight.sum(),
    )


def global_mass_objective(
    numerators: Tensor,
    masses: Tensor,
) -> Tensor:
    """Return the exact globally normalized fixed-rollout objective.

    ``numerators`` and ``masses`` may index ranks, execution families, or their
    Cartesian product.  They are summed before division, so this function can
    never accidentally introduce an equal-rank average.
    """

    if numerators.shape != masses.shape:
        raise ValueError("numerators and masses must have matching shapes")
    if numerators.device != masses.device:
        raise ValueError("numerators and masses must be on the same device")
    if not bool(torch.isfinite(numerators).all()):
        raise ValueError("weighted numerators must be finite")
    if not bool(torch.isfinite(masses).all()) or bool((masses < 0).any()):
        raise ValueError("decision masses must be finite and nonnegative")
    total_mass = masses.sum()
    if not bool(total_mass > 0):
        raise ValueError("global decision mass must be positive")
    return numerators.sum() / total_mass


def stratum_objective_scale(local_mass: Tensor, global_mass: Tensor) -> Tensor:
    """Return ``alpha[g,m] = Z[g,m] / Z`` for a sampled local stratum."""

    if local_mass.numel() != 1 or global_mass.numel() != 1:
        raise ValueError("local and global masses must be scalar tensors")
    if local_mass.device != global_mass.device:
        raise ValueError("local and global masses must be on the same device")
    if not bool(torch.isfinite(local_mass)) or not bool(torch.isfinite(global_mass)):
        raise ValueError("decision masses must be finite")
    if bool(local_mass < 0) or not bool(global_mass > 0):
        raise ValueError("local mass must be nonnegative and global mass positive")
    return local_mass / global_mass
