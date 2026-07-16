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
    safe_column = legal_mask.to(torch.int64).argmax(1, keepdim=True)
    safe_event = legal_ids.gather(1, safe_column).expand_as(selected)
    safe = torch.where(child_mask, selected, safe_event)
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
    """Return the entropy coefficient for a zero-based training iteration.

    Linear decay starts at ``c_h`` for iteration zero and reaches zero at
    ``training.iters``.  Callers that pass a loss-like object directly may
    expose ``iters`` on that object, but the horizon is never guessed.
    """

    value = float(_cfg(cfg, "c_h"))
    if _cfg(cfg, "entropy_schedule", "const") == "linear_decay":
        schedule = getattr(cfg, "training", cfg)
        iters = getattr(schedule, "iters", None)
        if isinstance(iters, bool) or not isinstance(iters, int) or iters <= 0:
            raise ValueError(
                "linear entropy decay requires a positive training.iters horizon"
            )
        value *= max(0.0, 1.0 - float(iteration) / float(iters))
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


def _policy_kl(new: Any, reference: Any, targets: Any, adapter: Any) -> Tensor:
    """Return exact legal-action ``KL(new || reference)`` as ``[B,X]``."""

    if reference is None:
        raise ValueError("a reference-policy output is required when c_kl is positive")
    legal_ids = targets.legal_event_ids.to(new.v.device)
    legal_mask = targets.legal_event_mask.to(new.v.device)
    try:
        result = adapter.kl(new, reference, legal_ids, legal_mask)
    except TypeError:
        result = adapter.kl(new, reference)
    if result.shape != new.v.shape:
        raise ValueError("adapter KL must have shape [B,X]")
    return result


def rb_ppo(
    new: Any,
    old: Any,
    targets: Any,
    adapter: Any,
    cfg: Any,
    *,
    iteration: int = 0,
    reduction: str = "mean",
    reference: Any | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Return clipped policy, value, entropy, and optional NashPG KL losses."""

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

    value_fn = l_v_clip if _cfg(cfg, "value_loss", "mse") == "clipped" else l_v
    value, _ = value_fn(new, old, targets, adapter, cfg)
    entropy, entropy_metrics = l_h(new, old, targets, adapter, cfg, iteration=iteration)
    type_weight = parent_p * valid.to(parent_p)
    c_kl = float(_cfg(cfg, "c_kl", 0.0))
    kl = (
        torch.zeros_like(new.v)
        if c_kl == 0.0
        else c_kl * _policy_kl(new, reference, targets, adapter)
    )
    per_row = per_row + ((value + entropy + kl) * type_weight).sum(1)

    policy_denominator = policy_weight.sum()
    denominator = policy_denominator.clamp_min(torch.finfo(policy_weight.dtype).eps)
    ratio = policy_metrics["ratio"]
    log_ratio = torch.log(ratio)
    raw_entropy_denominator = type_weight.sum()
    entropy_denominator = raw_entropy_denominator.clamp_min(
        torch.finfo(type_weight.dtype).eps
    )
    wanted = target.v_hat.to(new.v)
    residual = (type_weight * (wanted - new.v).square()).sum()
    stats = {
        "rows": per_row.new_tensor(per_row.shape[0]),
        "loss_sum": per_row.sum().detach(),
        "policy_sum": (policy * policy_weight).sum().detach(),
        "value_sum": (value * type_weight).sum().detach(),
        "entropy_loss_sum": (entropy * type_weight).sum().detach(),
        "reference_kl_sum": (kl * type_weight).sum().detach(),
        "kl_num": (policy_weight * (ratio - 1.0 - log_ratio)).sum().detach(),
        "kl_den": policy_denominator.detach(),
        "clip_num": (
            policy_weight * policy_metrics["clipped"].to(policy_weight)
        ).sum().detach(),
        "entropy_num": (entropy_metrics["entropy"] * type_weight).sum().detach(),
        "entropy_den": raw_entropy_denominator.detach(),
        "wanted_sum": (type_weight * wanted).sum().detach(),
        "wanted_sq_sum": (type_weight * wanted.square()).sum().detach(),
        "residual_sum": residual.detach(),
        "ratio_max": torch.where(
            child_valid, ratio, torch.zeros_like(ratio)
        ).max().detach(),
    }
    if reduction == "sum":
        return per_row.sum(), stats
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'sum'")
    return per_row.mean(), finalize_metric_stats([stats])


def ppo(
    new: Any,
    old: Any,
    targets: Any,
    adapter: Any,
    cfg: Any,
    *,
    iteration: int = 0,
    reduction: str = "mean",
    reference: Any | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Ordinary PPO with the same optional NashPG KL as the RB modes."""

    actor_type = targets.actor_type.to(new.v.device, dtype=torch.int64)
    rows = torch.arange(actor_type.numel(), device=new.v.device)
    logp_all = adapter.log_prob(
        new,
        targets.event.to(new.v.device)[:, None],
        targets.legal_event_ids.to(new.v.device),
        targets.legal_event_mask.to(new.v.device),
    )[:, 0]
    logp = logp_all[rows, actor_type]
    ratio = torch.exp(logp - targets.old_logp.to(logp))
    advantage = targets.advantage.to(logp)
    eps = float(_cfg(cfg, "clip_eps"))
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
    policy = -torch.minimum(ratio * advantage, clipped * advantage)

    value = new.v[rows, actor_type]
    wanted = targets.v_target.to(value)
    if _cfg(cfg, "value_loss", "mse") == "clipped":
        old_value = targets.old_v.to(value)
        clipped_value = old_value + (value - old_value).clamp(
            -float(_cfg(cfg, "clip_eps_v")), float(_cfg(cfg, "clip_eps_v"))
        )
        value_loss = float(_cfg(cfg, "c_v")) * torch.maximum(
            (value - wanted).square(), (clipped_value - wanted).square()
        )
    else:
        value_loss = float(_cfg(cfg, "c_v")) * (value - wanted).square()
    entropy_all = adapter.entropy(
        new,
        targets.legal_event_ids.to(new.v.device),
        targets.legal_event_mask.to(new.v.device),
    )
    entropy = entropy_all[rows, actor_type]
    entropy_loss = -_entropy_coefficient(cfg, iteration) * entropy
    c_kl = float(_cfg(cfg, "c_kl", 0.0))
    if c_kl == 0.0:
        kl_loss = torch.zeros_like(value)
    else:
        kl_all = _policy_kl(new, reference, targets, adapter)
        kl_loss = c_kl * kl_all[rows, actor_type]
    per_row = policy + value_loss + entropy_loss + kl_loss

    residual = (wanted - value).square().sum()
    rows_count = per_row.new_tensor(per_row.shape[0])
    stats = {
        "rows": rows_count,
        "loss_sum": per_row.sum().detach(),
        "policy_sum": policy.sum().detach(),
        "value_sum": value_loss.sum().detach(),
        "entropy_loss_sum": entropy_loss.sum().detach(),
        "reference_kl_sum": kl_loss.sum().detach(),
        "kl_num": (ratio - 1.0 - torch.log(ratio)).sum().detach(),
        "kl_den": rows_count,
        "clip_num": (ratio != clipped).to(ratio).sum().detach(),
        "entropy_num": entropy.sum().detach(),
        "entropy_den": rows_count,
        "wanted_sum": wanted.sum().detach(),
        "wanted_sq_sum": wanted.square().sum().detach(),
        "residual_sum": residual.detach(),
        "ratio_max": ratio.max().detach(),
    }
    if reduction == "sum":
        return per_row.sum(), stats
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'sum'")
    return per_row.mean(), finalize_metric_stats([stats])


def finalize_metric_stats(parts: list[dict[str, Tensor]]) -> dict[str, float]:
    """Finalize additive loss diagnostics from one or more physical batches."""

    if not parts:
        return {}
    summed = {
        key: torch.stack([part[key].float() for part in parts]).sum()
        for key in parts[0]
        if key != "ratio_max"
    }
    ratio_max = torch.stack([part["ratio_max"].float() for part in parts]).max()
    rows = summed["rows"].clamp_min(1.0)
    kl_den = summed["kl_den"].clamp_min(1e-12)
    entropy_den = summed["entropy_den"].clamp_min(1e-12)
    variance = summed["wanted_sq_sum"] - summed["wanted_sum"].square() / entropy_den
    explained = 1.0 - summed["residual_sum"] / variance.clamp_min(1e-12)
    return {
        "loss": float(summed["loss_sum"] / rows),
        "policy": float(summed["policy_sum"] / rows),
        "value": float(summed["value_sum"] / rows),
        "entropy_loss": float(summed["entropy_loss_sum"] / rows),
        "reference_kl": float(summed["reference_kl_sum"] / rows),
        "approx_kl": float(summed["kl_num"] / kl_den),
        "clip_frac": float(summed["clip_num"] / kl_den),
        "entropy": float(summed["entropy_num"] / entropy_den),
        "explained_variance": float(explained),
        "ratio_max": float(ratio_max),
    }


@dataclass(frozen=True)
class LossSpec:
    """Composite loss and the target fields it requires."""

    fn: Callable[..., tuple[Tensor, dict[str, Any]]]
    required_targets: frozenset[str]
    modes: frozenset[str] = frozenset(
        {"normal", "rao_blackwell", "branched_rao_blackwell"}
    )


COMPOSITES: dict[str, LossSpec] = {
    "ppo": LossSpec(
        ppo, frozenset({"advantage", "v_target"}), frozenset({"normal"})
    ),
    "rb_ppo": LossSpec(
        rb_ppo,
        frozenset({"parent_p_hat", "child_p_hat", "a_hat", "v_hat"}),
        frozenset({"rao_blackwell", "branched_rao_blackwell"}),
    )
}
