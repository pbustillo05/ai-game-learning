"""Minimal posterior-weighted policy, value, and entropy losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


def _cfg(cfg: Any, name: str, default: Any = None) -> Any:
    source = getattr(cfg, "loss", cfg)
    return getattr(source, name, default)


def _parts(old: Any, targets: Any) -> tuple[Any, Tensor, Tensor]:
    target = getattr(targets, "targets", targets)
    source = targets if hasattr(targets, "old_logp_child") else old
    logp = getattr(source, "old_logp_child", None)
    value = getattr(source, "old_v", None)
    if logp is None or value is None:
        raise TypeError("old rollout log-probabilities and values are required")
    return target, logp, value


def _selected_logp(new: Any, targets: Any, adapter: Any) -> Tensor:
    target = getattr(targets, "targets", targets)
    selected = target.child_event_row.to(new.v.device)
    child_mask = target.child_mask.to(new.v.device)
    legal_ids = targets.legal_event_ids.to(new.v.device)
    legal_mask = targets.legal_event_mask.to(new.v.device)
    if legal_ids.ndim != 2 or not bool(legal_mask.any(1).all()):
        raise ValueError("every minibatch row must contain a legal event")
    safe = torch.where(child_mask, selected, legal_ids[:, :1].expand_as(selected))
    result = adapter.log_prob(new, safe, legal_ids, legal_mask)
    if result.shape != target.a_hat.shape:
        raise ValueError("adapter log probabilities are not aligned with a_hat")
    return result


def _ratio(new: Any, old: Any, targets: Any, adapter: Any) -> tuple[Tensor, Tensor]:
    target, cached, _ = _parts(old, targets)
    logp = _selected_logp(new, targets, adapter)
    active = target.child_mask.to(logp.device)[:, :, None]
    difference = torch.where(active, logp, torch.zeros_like(logp)) - torch.where(
        active, cached.to(logp), torch.zeros_like(logp)
    )
    return torch.exp(difference), target.a_hat.to(new.v)


def l_pg(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    """Return direct-advantage policy loss elements ``[B,E,X]``."""

    del cfg
    ratio, advantage = _ratio(new, old, targets, adapter)
    return -ratio * advantage, {"ratio": ratio}


def l_clip(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    """Return PPO-clipped direct-advantage elements ``[B,E,X]``."""

    ratio, advantage = _ratio(new, old, targets, adapter)
    eps = float(_cfg(cfg, "clip_eps"))
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
    return -torch.minimum(ratio * advantage, clipped * advantage), {
        "ratio": ratio,
        "clipped": ratio != clipped,
    }


def l_v(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    """Return actor-type squared value loss elements ``[B,X]``."""

    del old, adapter
    target = getattr(targets, "targets", targets)
    return float(_cfg(cfg, "c_v")) * (new.v - target.v_hat.to(new.v)) ** 2, {}


def l_v_clip(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    """Return clipped actor-type value loss elements ``[B,X]``."""

    del adapter
    target, _, old_v = _parts(old, targets)
    old_value = old_v.to(new.v)
    clipped = old_value + (new.v - old_value).clamp(
        -float(_cfg(cfg, "clip_eps_v")), float(_cfg(cfg, "clip_eps_v"))
    )
    wanted = target.v_hat.to(new.v)
    return float(_cfg(cfg, "c_v")) * torch.maximum(
        (new.v - wanted) ** 2, (clipped - wanted) ** 2
    ), {}


def _entropy_coefficient(cfg: Any, iteration: int) -> float:
    value = float(_cfg(cfg, "c_h"))
    if _cfg(cfg, "entropy_schedule", "const") == "linear_decay":
        iters = int(getattr(getattr(cfg, "train", cfg), "iters", 1))
        value *= max(0.0, 1.0 - float(iteration) / max(iters, 1))
    return value


def l_h(
    new: Any,
    old: Any,
    targets: Any,
    adapter: Any,
    cfg: Any,
    *,
    iteration: int = 0,
) -> tuple[Tensor, dict]:
    """Return entropy-regularization elements ``[B,X]``."""

    del old
    entropy = adapter.entropy(
        new,
        targets.legal_event_ids.to(new.v.device),
        targets.legal_event_mask.to(new.v.device),
    )
    return -_entropy_coefficient(cfg, iteration) * entropy, {"entropy": entropy}


def _sum(value: Tensor, mask: Tensor, dims: tuple[int, ...]) -> Tensor:
    return (value * mask.to(value)).sum(dim=dims)


def rb_ppo(
    new: Any,
    old: Any,
    targets: Any,
    adapter: Any,
    cfg: Any,
    *,
    iteration: int = 0,
) -> tuple[Tensor, dict[str, float]]:
    """Return ``L_clip[A_hat] + L_H + L_V[V_hat]`` averaged over rows."""

    target, _, _ = _parts(old, targets)
    valid = targets.valid_type_mask.to(new.v.device)
    child = target.child_mask.to(new.v.device)
    child_valid = child[:, :, None] & valid[:, None, :]
    child_p = target.child_p_hat.to(new.v)
    parent_p = target.parent_p_hat.to(new.v)
    count = target.child_count.to(new.v).clamp_min(1.0)[:, None, None]

    policy, policy_metrics = l_clip(new, old, targets, adapter, cfg)
    policy_weight = child_p / count * child_valid.to(child_p)
    per_row = (policy * policy_weight).sum((1, 2))

    value, _ = l_v(new, old, targets, adapter, cfg)
    entropy, entropy_metrics = l_h(new, old, targets, adapter, cfg, iteration=iteration)
    type_weight = parent_p * valid.to(parent_p)
    per_row = per_row + ((value + entropy) * type_weight).sum(1)

    denominator = policy_weight.sum().clamp_min(torch.finfo(policy_weight.dtype).eps)
    ratio = policy_metrics["ratio"]
    log_ratio = torch.log(ratio)
    approx_kl = (policy_weight * (ratio - 1.0 - log_ratio)).sum() / denominator
    clip_frac = (policy_weight * policy_metrics["clipped"].to(policy_weight)).sum() / denominator
    entropy_denominator = type_weight.sum().clamp_min(torch.finfo(type_weight.dtype).eps)
    wanted = target.v_hat.to(new.v)
    mean_wanted = (type_weight * wanted).sum() / entropy_denominator
    variance = (type_weight * (wanted - mean_wanted).square()).sum()
    residual = (type_weight * (wanted - new.v).square()).sum()
    explained = 1.0 - residual / variance.clamp_min(1e-12)
    metrics = {
        "loss": float(per_row.mean().detach()),
        "policy": float((policy * policy_weight).sum(1).sum(1).mean().detach()),
        "value": float((value * type_weight).sum(1).mean().detach()),
        "entropy_loss": float((entropy * type_weight).sum(1).mean().detach()),
        "approx_kl": float(approx_kl.detach()),
        "clip_frac": float(clip_frac.detach()),
        "entropy": float((entropy_metrics["entropy"] * type_weight).sum().detach() / entropy_denominator),
        "explained_variance": float(explained.detach()),
        "ratio_max": float(torch.where(child_valid, ratio, torch.zeros_like(ratio)).max().detach()),
    }
    return per_row.mean(), metrics


@dataclass(frozen=True)
class LossSpec:
    """Composite loss and the target fields it requires."""

    fn: Callable[..., tuple[Tensor, dict[str, float]]]
    required_targets: frozenset[str]


COMPOSITES: dict[str, LossSpec] = {
    "rb_ppo": LossSpec(
        rb_ppo, frozenset({"parent_p_hat", "child_p_hat", "a_hat", "v_hat"})
    )
}
