"""Fixed-policy experiments for branching intensity and size control.

This module deliberately sits outside the production training pipeline.  The
core sampler accepts an optional controller seam, but its default path remains
the fixed-intensity implementation used by training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping

import torch
import yaml
from torch import Tensor

from pbg.core.branching import _sample_branched_labeled
from pbg.core.config import RootConfig, seed_all
from pbg.core.game import CHANCE, PRETERM, TERMINAL, GameBundle
from pbg.core.training import build_game_bundle
from pbg.core.tree import TreeBatch


_STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


@dataclass(frozen=True)
class PublicBucket:
    """Public experiment coordinates for one expandable sampled-tree node."""

    phase: str
    street: int
    street_depth: int
    total_player_depth: int

    @property
    def key(self) -> str:
        return f"{self.phase}:{self.street}:{self.street_depth}"


class BucketLookup:
    """Lookup numeric values from flat ``phase:street:depth`` rules.

    Accepted depth selectors are an integer, an inclusive range such as
    ``2-4``, or ``*``.  Street may be an integer, its poker name, or ``*``.
    Rules are matched by specificity; ``default`` is the final fallback.
    """

    def __init__(self, values: Mapping[str, float] | None, default: float) -> None:
        raw = dict(values or {})
        self.has_explicit_default = "default" in raw
        self.default = float(raw.pop("default", default))
        self.rules: list[tuple[int, str, str, str, float]] = []
        for key, value in raw.items():
            parts = str(key).split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"bucket rule {key!r} must be 'phase:street:depth'"
                )
            phase, street, depth = parts
            if phase not in {"player", "chance", "*"}:
                raise ValueError(f"invalid bucket phase {phase!r}")
            if street not in {"*", *(_STREET_NAMES.values())}:
                try:
                    int(street)
                except ValueError as exc:
                    raise ValueError(f"invalid bucket street {street!r}") from exc
            _depth_matches(depth, 0)
            specificity = int(phase != "*") + int(street != "*") + int(depth != "*")
            self.rules.append((specificity, phase, street, depth, float(value)))
        self.rules.sort(key=lambda item: item[0], reverse=True)

    def get(self, bucket: PublicBucket, fallback: float | None = None) -> float:
        street_name = _STREET_NAMES.get(bucket.street, str(bucket.street))
        for _, phase, street, depth, value in self.rules:
            if phase not in {"*", bucket.phase}:
                continue
            if street not in {"*", street_name, str(bucket.street)}:
                continue
            if _depth_matches(depth, bucket.street_depth):
                return value
        if fallback is not None and not self.has_explicit_default:
            return float(fallback)
        return self.default


def _depth_matches(selector: str, depth: int) -> bool:
    if selector == "*":
        return True
    if "-" in selector:
        lower, upper = selector.split("-", 1)
        if not lower or not upper:
            raise ValueError(f"invalid depth range {selector!r}")
        return int(lower) <= depth <= int(upper)
    return int(selector) == depth


def load_numeric_rules(value: str | Path | None) -> dict[str, float]:
    """Load a flat numeric mapping from JSON/YAML text or a file path."""

    if value is None:
        return {}
    text = str(value)
    path = Path(text)
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(text)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("bucket rules must be a mapping")
    result: dict[str, float] = {}
    for key, item in raw.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"bucket rule {key!r} must be numeric")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"bucket rule {key!r} must be finite and positive")
        result[str(key)] = number
    return result


def gamma_deviance(log_mean: Tensor, target: Tensor) -> Tensor:
    """Return Gamma deviance whose population optimum is the arithmetic mean."""

    if log_mean.shape != target.shape:
        raise ValueError("log_mean and target must have identical shapes")
    if not log_mean.is_floating_point() or not target.is_floating_point():
        raise TypeError("Gamma-deviance arguments must be floating point")
    if bool((target <= 0).any()):
        raise ValueError("Gamma-deviance targets must be positive")
    work = torch.promote_types(log_mean.dtype, target.dtype)
    prediction = log_mean.to(work)
    observed = target.to(work)
    return observed * torch.exp(-prediction) + prediction - torch.log(observed) - 1.0


def update_kappa(kappa: float, target_size: float, realized_size: float, eta: float) -> float:
    """Apply the experimental multiplicative dual update."""

    values = (kappa, target_size, realized_size, eta)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("kappa update arguments must be finite")
    if kappa <= 0.0 or target_size <= 0.0 or realized_size <= 0.0 or eta <= 0.0:
        raise ValueError("kappa update arguments must be positive")
    return kappa * (target_size / realized_size) ** eta


def reject_post_completion_refill() -> None:
    """Reject result-dependent attachment of children to a completed tree."""

    raise RuntimeError(
        "post-completion refill is not an experiment controller: offspring counts "
        "must be chosen before the corresponding child outcomes"
    )


@dataclass
class _PendingNode:
    node: int
    bucket: PublicBucket
    configured_beta: float
    proposed_beta: float
    capped_beta: float
    controller_cap: float
    admitted_beta: float
    predicted_cost: float
    reserve_before: float
    reserve_after_admission: float
    reserve_after_realization: float = 0.0
    randomized_offspring: int = 0
    realized_offspring: int = 0


class FixedPolicyBranchingController:
    """Experiment-only β controller and incremental reserve observer."""

    VARIANTS = {"legacy", "fixed", "cost_only", "constant_kkt"}

    def __init__(
        self,
        *,
        variant: str,
        schedule: Mapping[str, float] | None = None,
        costs: Mapping[str, float] | None = None,
        kappa: float = 1.0,
        soft_budget: float | None = None,
        beta_max: float | None = None,
        verify_reserve: bool = False,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown branching experiment variant {variant!r}")
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be finite and positive")
        if soft_budget is not None and (
            not math.isfinite(soft_budget) or soft_budget <= 0.0
        ):
            raise ValueError("soft_budget must be finite and positive")
        if beta_max is not None and (
            not math.isfinite(beta_max) or beta_max < 1.0
        ):
            raise ValueError("beta_max must be finite and at least one")
        self.variant = variant
        self.schedule = BucketLookup(schedule, 1.0)
        self.costs = BucketLookup(costs, 1.0)
        self.kappa = float(kappa)
        self.soft_budget = None if soft_budget is None else float(soft_budget)
        self.beta_max = float("inf") if beta_max is None else float(beta_max)
        self.verify_reserve = bool(verify_reserve)
        self.records: list[dict[str, Any]] = []
        self.reserve = 0.0
        self._backend: Any = None
        self._buffer: Any = None
        self._street_depth: list[int] = []
        self._kkt_ancestry: list[float] = []
        self._pending: list[_PendingNode] = []

    def start(
        self,
        *,
        backend: Any,
        buffer: Any,
        root_nodes: Tensor,
        root_states: Tensor,
        root_owner: Tensor,
    ) -> None:
        self._backend = backend
        self._buffer = buffer
        self._street_depth = [0] * buffer.node_capacity
        self._kkt_ancestry = [0.0] * buffer.node_capacity
        contexts = self._contexts(root_nodes, root_states, root_owner)
        for node, bucket in zip(root_nodes.tolist(), contexts, strict=True):
            self._kkt_ancestry[node] = 1.0 if bucket.phase == "player" else 0.0
        self.reserve = float(root_nodes.numel()) + sum(
            self.costs.get(bucket) for bucket in contexts
            if bucket.phase in {"player", "chance"}
        )

    def _contexts(
        self, nodes: Tensor, states: Tensor, owners: Tensor
    ) -> list[PublicBucket]:
        node_rows = nodes.to(torch.int64)
        total_depth = self._buffer.depth_player[node_rows].tolist()
        state_store = getattr(self._backend, "states", None)
        street_tensor = getattr(state_store, "street", None)
        if isinstance(street_tensor, Tensor):
            streets = street_tensor[states.to(torch.int64)].tolist()
        else:
            streets = [-1] * nodes.numel()
        result: list[PublicBucket] = []
        for node, owner, street, depth in zip(
            nodes.tolist(), owners.tolist(), streets, total_depth, strict=True
        ):
            if 0 <= owner < int(self._backend.n_players):
                phase = "player"
            elif owner == CHANCE:
                phase = "chance"
            else:
                phase = "leaf"
            result.append(
                PublicBucket(
                    phase=phase,
                    street=int(street),
                    street_depth=int(self._street_depth[node]),
                    total_player_depth=int(depth),
                )
            )
        return result

    def _legal_caps(self, states: Tensor, owners: Tensor) -> list[float]:
        caps = [float("inf")] * states.numel()
        player = ((owners >= 0) & (owners < int(self._backend.n_players))).nonzero().squeeze(1)
        if player.numel():
            _, mask = self._backend.legal_events(states[player])
            counts = mask.sum(1).tolist()
            for position, count in zip(player.tolist(), counts, strict=True):
                caps[position] = float(max(int(count), 1))
        return caps

    def choose_beta(
        self,
        *,
        backend: Any,
        buffer: Any,
        active_nodes: Tensor,
        active_states: Tensor,
        active_owner: Tensor,
        configured_beta: Tensor,
    ) -> Tensor:
        del backend, buffer
        contexts = self._contexts(active_nodes, active_states, active_owner)
        configured = configured_beta.tolist()
        legal_caps = self._legal_caps(active_states, active_owner)
        proposed: list[float] = []
        capped: list[float] = []
        controller_caps: list[float] = []
        costs: list[float] = []
        for node, bucket, legacy, legal_cap in zip(
            active_nodes.tolist(), contexts, configured, legal_caps, strict=True
        ):
            cap = min(
                max(self.schedule.get(bucket, legacy), 1.0),
                legal_cap,
                self.beta_max,
            )
            cost = self.costs.get(bucket)
            costs.append(cost)
            if self.variant == "legacy":
                raw_value = float(legacy)
                value = raw_value
                cap = raw_value
            elif self.variant == "fixed" or bucket.phase != "player":
                raw_value = self.schedule.get(bucket, legacy)
                value = cap
            else:
                path_product = 1.0 / max(
                    float(self._buffer.path_weight[node]), torch.finfo(torch.float32).tiny
                )
                factor = 1.0
                if self.variant == "constant_kkt":
                    factor = self._kkt_ancestry[node] / path_product
                raw_value = math.sqrt(max(self.kappa * factor / cost, 1.0))
                value = min(max(raw_value, 1.0), cap)
            proposed.append(float(raw_value))
            capped.append(float(value))
            controller_caps.append(float(cap))

        pending: list[_PendingNode] = []
        for node, bucket, legacy, proposal, capped_value, cap, cost in zip(
            active_nodes.tolist(),
            contexts,
            configured,
            proposed,
            capped,
            controller_caps,
            costs,
            strict=True,
        ):
            pending.append(
                _PendingNode(
                    node=node,
                    bucket=bucket,
                    configured_beta=float(legacy),
                    proposed_beta=float(proposal),
                    capped_beta=float(capped_value),
                    controller_cap=float(cap),
                    admitted_beta=float(capped_value),
                    predicted_cost=float(cost),
                    reserve_before=float(self.reserve),
                    reserve_after_admission=float(self.reserve),
                )
            )
        self._pending = pending
        return torch.tensor(capped, dtype=torch.float32, device=active_states.device)

    def after_offspring(
        self,
        *,
        active_nodes: Tensor,
        configured_beta: Tensor,
        effective_beta: Tensor,
        offspring: Tensor,
    ) -> Tensor:
        del active_nodes, configured_beta, effective_beta
        if len(self._pending) != offspring.numel():
            raise RuntimeError("experiment controller lost frontier alignment")
        running = self.reserve
        admitted = offspring.clone()
        for index, (pending, count) in enumerate(
            zip(self._pending, offspring.tolist(), strict=True)
        ):
            pending.randomized_offspring = int(count)
            pending.reserve_before = running
            projected = (int(count) - 1) * pending.predicted_cost
            if self.soft_budget is not None and running + projected > self.soft_budget:
                count = 1
                admitted[index] = 1
                pending.admitted_beta = 1.0
            else:
                running += projected
            pending.realized_offspring = int(count)
            pending.reserve_after_admission = running
        return admitted

    def after_children(
        self,
        *,
        backend: Any,
        buffer: Any,
        active_nodes: Tensor,
        active_states: Tensor,
        active_owner: Tensor,
        offspring: Tensor,
        parent_positions: Tensor,
        child_nodes: Tensor,
        next_states: Tensor,
        next_owner: Tensor,
        child_events: Tensor,
    ) -> None:
        del backend
        parent_streets = [pending.bucket.street for pending in self._pending]
        state_store = getattr(self._backend, "states", None)
        street_tensor = getattr(state_store, "street", None)
        if isinstance(street_tensor, Tensor):
            child_streets = street_tensor[next_states.to(torch.int64)].tolist()
        else:
            child_streets = [-1] * child_nodes.numel()
        parent_pos = parent_positions.tolist()
        child_node_rows = child_nodes.tolist()
        child_owner_rows = next_owner.tolist()
        for position, node, owner, child_street in zip(
            parent_pos, child_node_rows, child_owner_rows, child_streets, strict=True
        ):
            parent = self._pending[position]
            if child_street != parent_streets[position]:
                street_depth = 0
            else:
                street_depth = parent.bucket.street_depth + int(
                    parent.bucket.phase == "player"
                )
            self._street_depth[node] = street_depth
            divisor = max(parent.realized_offspring, 1)
            inherited = self._kkt_ancestry[parent.node] / divisor
            self._kkt_ancestry[node] = inherited + int(
                0 <= owner < int(self._backend.n_players)
            )

        child_contexts = self._contexts(child_nodes, next_states, next_owner)
        live_cost_by_parent = [0.0] * len(self._pending)
        for position, bucket in zip(parent_pos, child_contexts, strict=True):
            if bucket.phase in {"player", "chance"}:
                live_cost_by_parent[position] += self.costs.get(bucket)
        running = self.reserve
        for index, pending in enumerate(self._pending):
            running = (
                running
                - pending.predicted_cost
                + pending.realized_offspring
                + live_cost_by_parent[index]
            )
            pending.reserve_after_realization = running
        self.reserve = running
        if self.verify_reserve:
            oracle = self.recomputed_reserve(
                buffer.node_count, child_nodes, next_states, next_owner
            )
            if not math.isclose(self.reserve, oracle, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(
                    f"incremental reserve {self.reserve} differs from oracle {oracle}"
                )
        labels = child_events.tolist()
        positions = parent_positions.tolist()
        for index, pending in enumerate(self._pending):
            sampled = [label for label, position in zip(labels, positions, strict=True) if position == index]
            unique = len(set(sampled))
            count = len(sampled)
            self.records.append(
                {
                    "node": pending.node,
                    "bucket": pending.bucket.key,
                    **asdict(pending.bucket),
                    "configured_beta": pending.configured_beta,
                    "kkt_proposed_beta": pending.proposed_beta,
                    "street_depth_capped_beta": pending.capped_beta,
                    "controller_cap": pending.controller_cap,
                    "cap_hit": pending.capped_beta < pending.proposed_beta,
                    "budget_admitted_beta": pending.admitted_beta,
                    "randomized_offspring": pending.randomized_offspring,
                    "budget_admitted_offspring": pending.realized_offspring,
                    "realized_offspring": pending.realized_offspring,
                    "sampled_public_labels": sampled,
                    "unique_label_fraction": unique / max(count, 1),
                    "duplicate_label_rate": 1.0 - unique / max(count, 1),
                    "predicted_descendant_cost": pending.predicted_cost,
                    "reserve_before_proposal": pending.reserve_before,
                    "reserve_after_admission": pending.reserve_after_admission,
                    "reserve_after_child_realization": pending.reserve_after_realization,
                }
            )
        self._pending = []

    def recomputed_reserve(
        self, stored_nodes: int, frontier_nodes: Tensor, frontier_states: Tensor, frontier_owner: Tensor
    ) -> float:
        """Test-only from-scratch oracle for the incremental reserve."""

        contexts = self._contexts(frontier_nodes, frontier_states, frontier_owner)
        return float(stored_nodes) + sum(
            self.costs.get(bucket)
            for bucket in contexts
            if bucket.phase in {"player", "chance"}
        )

    def finalize(self, tree: TreeBatch) -> None:
        parent = tree.parent.detach().to("cpu", torch.int64).tolist()
        counts = (tree.child_ptr[1:] - tree.child_ptr[:-1]).detach().to("cpu", torch.int64).tolist()
        subtree = [1] * len(parent)
        for node in range(len(parent) - 1, 0, -1):
            subtree[parent[node]] += subtree[node]
        by_node = {record["node"]: record for record in self.records}
        for node, count in enumerate(counts):
            if count and node in by_node:
                by_node[node]["realized_descendant_cost"] = (subtree[node] - 1) / count


def tree_metrics(tree: TreeBatch, node_budget: int) -> dict[str, float]:
    """Return topology and weighted decision-law metrics for one tree."""

    players = tree.depth_p.shape[1]
    decisions = (tree.owner >= 0) & (tree.owner < players)
    leaves = tree.child_ptr[1:] == tree.child_ptr[:-1]
    weights = tree.path_weight[decisions].double()
    weight_sum = weights.sum()
    ess = weight_sum.square() / weights.square().sum().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    nodes = int(tree.parent.numel())
    return {
        "nodes": float(nodes),
        "leaves": float(leaves.sum()),
        "decision_rows": float(decisions.sum()),
        "decision_law_ess": float(ess),
        "overflow_margin_nodes": float(node_budget - nodes),
    }


def _aggregate(values: Iterable[float]) -> dict[str, float]:
    rows = torch.tensor(list(values), dtype=torch.float64)
    if not rows.numel():
        return {"count": 0.0}
    return {
        "count": float(rows.numel()),
        "mean": float(rows.mean()),
        "std": float(rows.std(unbiased=False)),
        "min": float(rows.min()),
        "median": float(rows.quantile(0.5)),
        "p95": float(rows.quantile(0.95)),
        "p99": float(rows.quantile(0.99)),
        "max": float(rows.max()),
    }


def _bucket_predictions(trials: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    split = max(1, len(trials) // 2)
    calibration: dict[str, list[float]] = {}
    for trial in trials[:split]:
        for node in trial.get("nodes", []):
            cost = node.get("realized_descendant_cost")
            if isinstance(cost, (int, float)) and cost > 0:
                calibration.setdefault(node["bucket"], []).append(float(cost))
    return {
        bucket: (
            math.exp(sum(math.log(value) for value in values) / len(values)),
            sum(values) / len(values),
        )
        for bucket, values in calibration.items()
        if values
    }


def annotate_bucket_predictions(trials: list[dict[str, Any]]) -> None:
    """Attach split-calibration forecasts to raw node and tree measurements."""

    predictions = _bucket_predictions(trials)
    for trial in trials:
        for node in trial.get("nodes", []):
            geometric, arithmetic = predictions.get(node["bucket"], (None, None))
            node["geometric_bucket_prediction"] = geometric
            node["arithmetic_bucket_prediction"] = arithmetic
            node["gamma_bucket_prediction"] = arithmetic
        root = next((node for node in trial.get("nodes", []) if node["node"] == 0), None)
        if root is not None:
            offspring = root["realized_offspring"]
            for name in ("geometric", "arithmetic", "gamma"):
                prediction = root.get(f"{name}_bucket_prediction")
                if prediction is not None:
                    predicted_nodes = float(1.0 + offspring * prediction)
                    realized_nodes = float(trial["metrics"]["nodes"])
                    trial["metrics"][f"{name}_predicted_tree_nodes"] = predicted_nodes
                    trial["metrics"][f"{name}_tree_size_error"] = (
                        predicted_nodes - realized_nodes
                    )
                    trial["metrics"][f"{name}_tree_size_relative_error"] = (
                        abs(predicted_nodes - realized_nodes) / realized_nodes
                    )


def _group_nodes(trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        for node in trial.get("nodes", []):
            grouped.setdefault(node["bucket"], []).append(node)
    result: dict[str, Any] = {}
    numeric = (
        "configured_beta",
        "kkt_proposed_beta",
        "street_depth_capped_beta",
        "budget_admitted_beta",
        "randomized_offspring",
        "budget_admitted_offspring",
        "realized_offspring",
        "duplicate_label_rate",
        "realized_descendant_cost",
        "reserve_before_proposal",
        "reserve_after_admission",
        "reserve_after_child_realization",
    )
    for bucket, nodes in grouped.items():
        summary = {
            key: _aggregate(
                float(node[key])
                for node in nodes
                if isinstance(node.get(key), (int, float))
            )
            for key in numeric
        }
        summary["cap_hit_rate"] = sum(bool(node["cap_hit"]) for node in nodes) / len(nodes)
        summary["budget_forced_spine_rate"] = sum(
            node["budget_admitted_beta"] == 1.0
            and node["street_depth_capped_beta"] > 1.0
            for node in nodes
        ) / len(nodes)
        result[bucket] = summary
    return result


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tree metrics and held-out geometric/arithmetic cost forecasts."""

    tree_keys = sorted(
        {key for trial in trials for key, value in trial.get("metrics", {}).items()
         if isinstance(value, (int, float))}
    )
    trees = {
        key: _aggregate(trial["metrics"][key] for trial in trials if key in trial.get("metrics", {}))
        for key in tree_keys
    }
    split = max(1, len(trials) // 2)
    training: dict[str, list[float]] = {}
    held_out: dict[str, list[float]] = {}
    for index, trial in enumerate(trials):
        destination = training if index < split else held_out
        for node in trial.get("nodes", []):
            cost = node.get("realized_descendant_cost")
            if isinstance(cost, (int, float)) and cost > 0:
                destination.setdefault(node["bucket"], []).append(float(cost))
    cost_summary: dict[str, Any] = {}
    for bucket, train_values in training.items():
        if not train_values:
            continue
        arithmetic = sum(train_values) / len(train_values)
        geometric = math.exp(sum(math.log(value) for value in train_values) / len(train_values))
        evaluation = held_out.get(bucket, [])
        record: dict[str, Any] = {
            "training_count": len(train_values),
            "geometric_prediction": geometric,
            "arithmetic_prediction": arithmetic,
            "gamma_deviance_optimum": arithmetic,
            "held_out_count": len(evaluation),
        }
        if evaluation:
            actual_mean = sum(evaluation) / len(evaluation)
            for name, prediction in (("geometric", geometric), ("arithmetic", arithmetic)):
                record[f"{name}_mean_bias"] = prediction - actual_mean
                record[f"{name}_mean_relative_error"] = sum(
                    abs(prediction - value) / value for value in evaluation
                ) / len(evaluation)
                record[f"{name}_underprediction_rate"] = sum(
                    prediction < value for value in evaluation
                ) / len(evaluation)
                record[f"{name}_mean_gamma_deviance"] = sum(
                    value / prediction
                    + math.log(prediction)
                    - math.log(value)
                    - 1.0
                    for value in evaluation
                ) / len(evaluation)
        cost_summary[bucket] = record
    return {
        "trees": trees,
        "street_depth": _group_nodes(trials),
        "cost_estimands": cost_summary,
        "kappa_before": _aggregate(trial["kappa_before"] for trial in trials),
        "kappa_after": _aggregate(trial["kappa_after"] for trial in trials),
    }


def _checkpoint_model_state(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(f"checkpoint {path} has no model state")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "iteration": int(payload.get("iteration", -1)),
        "run_id": payload.get("run_id"),
    }
    return payload["model"], metadata


def run_branching_experiments(
    cfg: RootConfig,
    checkpoint_paths: Iterable[str | Path],
    *,
    variants: Iterable[str],
    trials: int,
    seed: int,
    device: torch.device,
    schedule: Mapping[str, float] | None = None,
    costs: Mapping[str, float] | None = None,
    soft_budget_fraction: float | None = None,
    kappa: float = 1.0,
    kappa_eta: float = 1.0,
    target_size: float | None = None,
    beta_max: float | None = None,
) -> dict[str, Any]:
    """Run read-only fixed-policy branching experiments."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if soft_budget_fraction is not None and not 0.0 < soft_budget_fraction <= 1.0:
        raise ValueError("soft_budget_fraction must be in (0,1]")
    if not math.isfinite(kappa_eta) or kappa_eta <= 0.0:
        raise ValueError("kappa_eta must be finite and positive")
    if target_size is not None and (
        not math.isfinite(target_size) or target_size <= 0.0
    ):
        raise ValueError("target_size must be finite and positive")
    if cfg.training.mode != "branched_rao_blackwell":
        raise ValueError("branching experiments require branched_rao_blackwell config")
    requested_variants = list(variants)
    if not requested_variants:
        raise ValueError("at least one controller variant is required")
    checkpoint_values = list(checkpoint_paths)
    if not checkpoint_values:
        raise ValueError("at least one checkpoint is required")
    bundle = build_game_bundle(cfg, device)
    actual_device = torch.device(bundle.rollout_backend.device)
    bundle.model.to(actual_device).eval()
    bundle.model.requires_grad_(False)
    experiment_rows: list[dict[str, Any]] = []
    for checkpoint_value in checkpoint_values:
        checkpoint = Path(checkpoint_value)
        state, checkpoint_metadata = _checkpoint_model_state(checkpoint)
        bundle.model.load_state_dict(state, strict=True)
        for variant in requested_variants:
            current_kappa = float(kappa)
            rows: list[dict[str, Any]] = []
            for trial_index in range(trials):
                trial_seed = seed + trial_index
                torch_gen, _ = seed_all(trial_seed, 0, actual_device)
                controller = FixedPolicyBranchingController(
                    variant=variant,
                    schedule=schedule,
                    costs=costs,
                    kappa=current_kappa,
                    soft_budget=(
                        None
                        if soft_budget_fraction is None
                        else soft_budget_fraction * cfg.rollout.node_budget
                    ),
                    beta_max=beta_max,
                )
                if actual_device.type == "cuda":
                    torch.cuda.synchronize(actual_device)
                    torch.cuda.reset_peak_memory_stats(actual_device)
                started = time.perf_counter()
                with torch.no_grad():
                    tree, _, _ = _sample_branched_labeled(
                        bundle,
                        cfg.rollout,
                        torch_gen,
                        experiment_controller=controller,
                    )
                if actual_device.type == "cuda":
                    torch.cuda.synchronize(actual_device)
                elapsed = time.perf_counter() - started
                metrics = tree_metrics(tree, cfg.rollout.node_budget)
                metrics["rollout_seconds"] = elapsed
                if actual_device.type == "cuda":
                    metrics["peak_allocated_bytes"] = float(
                        torch.cuda.max_memory_allocated(actual_device)
                    )
                    metrics["peak_reserved_bytes"] = float(
                        torch.cuda.max_memory_reserved(actual_device)
                    )
                before = current_kappa
                if target_size is not None:
                    current_kappa = update_kappa(
                        current_kappa, target_size, metrics["nodes"], kappa_eta
                    )
                rows.append(
                    {
                        "trial": trial_index,
                        "seed": trial_seed,
                        "kappa_before": before,
                        "kappa_after": current_kappa,
                        "metrics": metrics,
                        "target_variance": None,
                        "gradient_proxy_variance": None,
                        "nodes": controller.records,
                        "unavailable_metrics": [
                            "target_variance",
                            "gradient_proxy_variance",
                        ],
                    }
                )
            annotate_bucket_predictions(rows)
            experiment_rows.append(
                {
                    "checkpoint": checkpoint_metadata,
                    "controller": variant,
                    "trials": rows,
                    "summary": summarize_trials(rows),
                }
            )
    environment = {
        "device": str(actual_device),
        "device_name": (
            torch.cuda.get_device_name(actual_device)
            if actual_device.type == "cuda"
            else platform.processor()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    scientific_config = asdict(cfg)
    scientific_config.pop("hardware", None)
    return {
        "schema": "pbg-branching-experiment-v1",
        "environment": environment,
        "scientific_config": scientific_config,
        "config": {
            "node_budget": cfg.rollout.node_budget,
            "beta": cfg.rollout.beta,
            "trials": trials,
            "seed": seed,
            "schedule": dict(schedule or {}),
            "costs": dict(costs or {}),
            "soft_budget_fraction": soft_budget_fraction,
            "initial_kappa": kappa,
            "kappa_eta": kappa_eta,
            "target_size": target_size,
            "beta_max": beta_max,
        },
        "experiments": experiment_rows,
    }
