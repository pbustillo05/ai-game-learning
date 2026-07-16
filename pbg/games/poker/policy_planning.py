"""Host-declared physical planning for device poker policy batches."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class PlannedPolicyBatch:
    """One device index view and its host-known execution family."""

    positions: Tensor
    family: str


def plan_policy_batches(
    token_count: Tensor,
    family: Tensor,
    *,
    phase: str,
    profile: Any,
    metrics: Any,
) -> list[PlannedPolicyBatch]:
    """Greedily pack family-homogeneous rows within one hardware profile.

    CUDA planning performs exactly one fixed-size device-to-host control
    transfer. The independently sorted device index vector supplies every
    physical gather, so no plan uploads Python-built indices back to CUDA.
    """

    if token_count.ndim != 1 or family.shape != token_count.shape:
        raise ValueError("policy cost and family vectors must share shape [B]")
    max_tokens = int(
        profile.training_max_tokens
        if phase == "training"
        else profile.inference_max_tokens
    )
    max_pairs = int(
        profile.training_max_token_pairs
        if phase == "training"
        else profile.inference_max_token_pairs
    )
    control = torch.stack((token_count, family), dim=1)
    if control.device.type == "cuda":
        started = time.perf_counter()
        host_control = control.to(device="cpu")
        metrics.record_scheduler_control_synchronization(
            seconds=time.perf_counter() - started,
            nbytes=control.numel() * control.element_size(),
        )
    else:
        host_control = control
    host_rows = host_control.tolist()
    host_order = sorted(
        range(token_count.numel()),
        key=lambda position: (
            int(host_rows[position][1]),
            int(host_rows[position][0]),
            position,
        ),
    )
    token_order = torch.argsort(token_count, stable=True)
    device_order = token_order[torch.argsort(family[token_order], stable=True)]

    plans: list[tuple[int, int, int, int]] = []
    start = 0
    count = 0
    current_family = -1
    current_max = 0
    for ordered_index, position in enumerate(host_order):
        length, family_id = (int(value) for value in host_rows[position])
        if length > max_tokens or length * length > max_pairs:
            raise MemoryError(
                f"one policy sequence needs {length} tokens/{length * length} "
                f"token pairs, beyond hardware profile {profile.name!r}"
            )
        family_changed = count > 0 and family_id != current_family
        candidate_max = max(current_max, length)
        candidate_count = count + 1
        capacity_exceeded = count > 0 and (
            candidate_count * candidate_max > max_tokens
            or candidate_count * candidate_max * candidate_max > max_pairs
        )
        if family_changed or capacity_exceeded:
            plans.append((start, count, current_family, current_max))
            start = ordered_index
            count = 0
            current_max = 0
        count += 1
        current_family = family_id
        current_max = max(current_max, length)
    if count:
        plans.append((start, count, current_family, current_max))

    result: list[PlannedPolicyBatch] = []
    for start, count, family_id, maximum in plans:
        positions = device_order.narrow(0, start, count)
        family_name = "pre" if family_id == 0 else f"post_{family_id}"
        result.append(PlannedPolicyBatch(positions, family_name))
        metrics.record_policy_pack(
            phase,
            family_name,
            rows=count,
            padded_tokens=count * maximum,
            padded_token_pairs=count * maximum * maximum,
        )
    return result
