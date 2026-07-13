"""Policy-gradient, value, entropy, KL, and moment losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _cfg(cfg: Any, name: str, default: Any = None) -> Any:
    source = getattr(cfg, "loss", cfg)
    return getattr(source, name, default)


def _sample(old: Any, targets: Any) -> tuple[Any, Tensor, Tensor, Tensor]:
    target = getattr(targets, "targets", targets)
    source = targets if hasattr(targets, "old_logp_child") else old
    logp = getattr(source, "old_logp_child", getattr(source, "logp_child", None))
    value = getattr(source, "old_v", getattr(source, "v", None))
    if logp is None or value is None:
        raise TypeError("old rollout log-probabilities and values are required")
    return target, logp, value, getattr(targets, "child_mask", target.nu_row != 0)


def _ratio(new: Any, old: Any, targets: Any, adapter: Any) -> tuple[Tensor, Tensor, Any]:
    target, old_logp, old_v, child_mask = _sample(old, targets)
    logp = adapter.log_prob(new, target.child_event_row.to(new.v.device).long())
    if logp.shape != target.q_hat.shape:
        raise ValueError("adapter log probabilities are not aligned with q_hat")
    cached = old_logp.to(logp.device, logp.dtype)
    # Avoid inf/nan even before masking: padded cache columns conventionally
    # contain -inf, and IEEE arithmetic would turn finite - (-inf) into inf.
    active = child_mask.to(logp.device)[:, :, None]
    safe_logp = torch.where(active, logp, torch.zeros_like(logp))
    safe_cached = torch.where(active, cached, torch.zeros_like(cached))
    difference = safe_logp - safe_cached
    ratio = torch.exp(difference)
    advantage = target.q_hat.to(new.v.device, new.v.dtype) - old_v.to(
        new.v.device, new.v.dtype
    )[:, None, :]
    return ratio, advantage, target


def l_pg(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    del cfg
    ratio, advantage, _ = _ratio(new, old, targets, adapter)
    return -ratio * advantage, {"ratio": ratio}


def l_clip(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    ratio, advantage, _ = _ratio(new, old, targets, adapter)
    eps = float(_cfg(cfg, "clip_eps"))
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
    loss = -torch.minimum(ratio * advantage, clipped * advantage)
    return loss, {"ratio": ratio, "clipped": (ratio != clipped)}


def l_v(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    del adapter
    target = getattr(targets, "targets", targets)
    loss = float(_cfg(cfg, "c_v")) * (new.v - target.v_hat.to(new.v)) ** 2
    return loss, {}


def l_v_clip(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    del adapter
    target, _, old_v, _ = _sample(old, targets)
    old_v = old_v.to(new.v)
    eps = float(_cfg(cfg, "clip_eps_v"))
    clipped = old_v + (new.v - old_v).clamp(-eps, eps)
    wanted = target.v_hat.to(new.v)
    loss = float(_cfg(cfg, "c_v")) * torch.maximum(
        (new.v - wanted) ** 2, (clipped - wanted) ** 2
    )
    return loss, {}


def _entropy_coefficient(cfg: Any, iteration: int) -> float:
    value = float(_cfg(cfg, "c_h"))
    if _cfg(cfg, "entropy_schedule", "const") == "linear_decay":
        iters = int(getattr(getattr(cfg, "train", cfg), "iters", 1))
        value *= max(0.0, 1.0 - float(iteration) / max(iters, 1))
    return value


def l_h(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any, *, iteration: int = 0) -> tuple[Tensor, dict]:
    del old, targets
    entropy = adapter.entropy(new)
    return -_entropy_coefficient(cfg, iteration) * entropy, {"entropy": entropy}


def l_kl(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    del targets
    reference = getattr(old, "ref_out", getattr(old, "out_ref", old))
    value = adapter.kl(new, reference)
    return float(_cfg(cfg, "c_kl")) * value, {"kl": value}


def l_kl_sample(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    ratio, _, _ = _ratio(new, old, targets, adapter)
    value = ratio * torch.log(ratio) - ratio + 1.0
    return float(_cfg(cfg, "c_kl")) * value, {"ratio": ratio}


def l_u(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    del old, adapter
    target = getattr(targets, "targets", targets)
    if new.u is None:
        raise ValueError("moment loss requires the u head")
    return float(_cfg(cfg, "c_u")) * (new.u - target.u_hat.to(new.u)) ** 2, {}


def l_vvec(new: Any, old: Any, targets: Any, adapter: Any, cfg: Any) -> tuple[Tensor, dict]:
    del old, adapter
    target = getattr(targets, "targets", targets)
    if new.vvec is None:
        raise ValueError("moment loss requires the vvec head")
    seat_mask = getattr(targets, "seat_mask", torch.ones_like(target.vvec_hat, dtype=torch.bool))
    squared = (new.vvec - target.vvec_hat.to(new.vvec)) ** 2
    return float(_cfg(cfg, "c_vvec")) * (squared * seat_mask.to(squared)).sum(-1), {}


def _masked_sum(value: Tensor, mask: Tensor, dims: tuple[int, ...]) -> Tensor:
    return (value * mask.to(value)).sum(dim=dims)


def rb_ppo(
    new: Any,
    old: Any,
    targets: Any,
    adapter: Any,
    cfg: Any,
    *,
    iteration: int = 0,
    out_ref: Any | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Return the three-level posterior-weighted rb_ppo objective."""

    target, _, _, default_child = _sample(old, targets)
    valid = getattr(targets, "valid_type_mask", target.p_hat != 0).to(new.v.device)
    child = getattr(targets, "child_mask", default_child).to(new.v.device)
    p = target.p_hat.to(new.v)
    nu = target.nu_row.to(new.v)
    policy, policy_metrics = l_clip(new, old, targets, adapter, cfg)
    policy_mask = child[:, :, None] & valid[:, None, :]
    per_row = _masked_sum(policy * p[:, None, :] * nu[:, :, None], policy_mask, (1, 2))

    value, _ = l_v_clip(new, old, targets, adapter, cfg)
    entropy, entropy_metrics = l_h(new, old, targets, adapter, cfg, iteration=iteration)
    per_type = value + entropy
    if out_ref is not None and _cfg(cfg, "ref_period") is not None:
        kl = adapter.kl(new, out_ref)
        per_type = per_type + float(_cfg(cfg, "c_kl")) * kl
    else:
        kl = torch.zeros_like(new.v)
    per_row = per_row + _masked_sum(per_type * p, valid, (1,))
    if bool(_cfg(cfg, "use_moment_heads", False)):
        moment_u, _ = l_u(new, old, targets, adapter, cfg)
        moment_v, _ = l_vvec(new, old, targets, adapter, cfg)
        per_row = per_row + moment_u + moment_v

    ratio = policy_metrics["ratio"]
    weight = p[:, None, :] * nu[:, :, None] * policy_mask.to(p)
    denominator = weight.sum().clamp_min(torch.finfo(weight.dtype).eps)
    log_ratio = torch.log(ratio)
    approx_kl = (weight * (ratio - 1.0 - log_ratio)).sum() / denominator
    clip_frac = (weight * policy_metrics["clipped"].to(weight)).sum() / denominator
    entropy_mean = (p * valid.to(p) * entropy_metrics["entropy"]).sum() / (
        (p * valid.to(p)).sum().clamp_min(torch.finfo(p.dtype).eps)
    )
    wanted = target.v_hat.to(new.v)
    value_weight = p * valid.to(p)
    mean_wanted = (value_weight * wanted).sum() / value_weight.sum().clamp_min(1e-12)
    variance = (value_weight * (wanted - mean_wanted) ** 2).sum()
    residual = (value_weight * (wanted - new.v) ** 2).sum()
    explained = 1.0 - residual / variance.clamp_min(1e-12)
    metrics = {
        "loss": float(per_row.mean().detach()),
        "policy": float(_masked_sum(policy * weight, policy_mask, (1, 2)).mean().detach()),
        "value": float(_masked_sum(value * p, valid, (1,)).mean().detach()),
        "entropy_loss": float(_masked_sum(entropy * p, valid, (1,)).mean().detach()),
        "kl": float((kl * p * valid).sum().detach() / p.sum().clamp_min(1e-12)),
        "approx_kl": float(approx_kl.detach()),
        "clip_frac": float(clip_frac.detach()),
        "entropy": float(entropy_mean.detach()),
        "explained_variance": float(explained.detach()),
        "ratio_max": float(torch.where(policy_mask, ratio, torch.zeros_like(ratio)).max().detach()),
    }
    return per_row.mean(), metrics


COMPOSITES = {"rb_ppo": rb_ppo}
