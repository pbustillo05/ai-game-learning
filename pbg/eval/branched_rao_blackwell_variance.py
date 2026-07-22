"""Measurement core for the checkpoint-1110 branched RB variance run.

The module deliberately keeps optimizer steps out of scope.  It measures one
frozen-gradient estimator, including the production leave-one-out baseline,
and reconstructs the exact ``1/B`` and configured ``1/n`` surfaces.
"""

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import warnings
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import yaml

from pbg.core.branching import BranchingDiagnosticLimits
from pbg.eval.rao_blackwell_variance import (
    checkpoint_sha256,
    logical_seed,
    rademacher_direction,
)


SCHEMA_VERSION = "pbg-branched-rb-variance-v1"


def _conditional(moment: Tensor, mass: Tensor) -> Tensor:
    result = torch.zeros_like(moment)
    valid = mass > 0
    result[valid] = moment[valid] / mass[valid]
    return result


def projected_loss_and_child_partials(
    child_p: Tensor,
    child_m: Tensor,
    old_v: Tensor,
    d_value: Tensor,
    d_logp: Tensor,
    child_mask: Tensor,
    child_count: Tensor,
    *,
    c_v: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return projected per-node loss and its child ``P/M`` derivatives.

    Shapes are ``P,M=[D,E,H]``, ``old_v=[D,H]``,
    ``d_value=[Q,D,H]``, and ``d_logp=[Q,D,E,H]``.  The expression is the
    derivative of the unclipped old-policy ``rb_ppo`` loss at ratio one with
    lambda one.  The returned partials are ``[Q,D,E,H]``.
    """

    if child_p.shape != child_m.shape or child_p.ndim != 3:
        raise ValueError("child P/M must have equal [D,E,H] shapes")
    decisions, edges, hands = child_p.shape
    probes = d_value.shape[0]
    if old_v.shape != (decisions, hands):
        raise ValueError("old_v must be [D,H]")
    if d_value.shape != (probes, decisions, hands):
        raise ValueError("d_value must be [Q,D,H]")
    if d_logp.shape != (probes, decisions, edges, hands):
        raise ValueError("d_logp must be [Q,D,E,H]")
    if child_mask.shape != (decisions, edges) or child_count.shape != (decisions,):
        raise ValueError("child metadata is not aligned")
    if bool((child_count <= 0).any()):
        raise ValueError("every decision must have a child")

    mask = child_mask[:, :, None]
    p = child_p.masked_fill(~mask, 0.0)
    m = child_m.masked_fill(~mask, 0.0)
    count = child_count.to(child_p)[:, None, None]
    p_sum = p.sum(1, keepdim=True)
    m_sum = m.sum(1, keepdim=True)
    continuation = _conditional(m, p)
    sibling_p = p_sum - p
    sibling_m = m_sum - m
    sibling = _conditional(sibling_m, sibling_p)
    multi = (child_count > 1)[:, None, None]
    baseline = torch.where(multi, sibling, old_v[:, None, :])
    advantage = (continuation - baseline).masked_fill(~mask, 0.0)

    dlogp = d_logp.masked_fill(~mask[None], 0.0)
    policy = -(p[None] * advantage[None] * dlogp / count[None]).sum((2, 3))
    parent_p = p_sum[:, 0] / child_count.to(p)[:, None]
    parent_m = m_sum[:, 0] / child_count.to(m)[:, None]
    value = (
        2.0
        * float(c_v)
        * (parent_p[None] * old_v[None] - parent_m[None])
        * d_value
    ).sum(2)
    projected = policy + value

    valid_sibling = (sibling_p > 0) & mask
    safe_sibling_p = torch.where(valid_sibling, sibling_p, torch.ones_like(sibling_p))
    sibling_for_derivative = torch.where(valid_sibling, sibling, torch.zeros_like(sibling))
    scaled = 1.0 / count[None]
    cross_p = torch.where(
        valid_sibling[None],
        -p[None] * dlogp * sibling_for_derivative[None] / safe_sibling_p[None] * scaled,
        torch.zeros_like(dlogp),
    )
    cross_m = torch.where(
        valid_sibling[None],
        p[None] * dlogp / safe_sibling_p[None] * scaled,
        torch.zeros_like(dlogp),
    )
    multi_p = (
        sibling_for_derivative[None] * dlogp * scaled
        + cross_p.sum(2, keepdim=True)
        - cross_p
    )
    multi_m = (
        -dlogp * scaled
        + cross_m.sum(2, keepdim=True)
        - cross_m
    )
    single_p = old_v[None, :, None, :] * dlogp
    single_m = -dlogp
    partial_p = torch.where(multi[None], multi_p, single_p)
    partial_m = torch.where(multi[None], multi_m, single_m)
    value_p = (
        2.0 * float(c_v) * old_v[None] * d_value
        / child_count.to(d_value)[None, :, None]
    )
    value_m = (
        -2.0 * float(c_v) * d_value
        / child_count.to(d_value)[None, :, None]
    )
    partial_p = (partial_p + value_p[:, :, None]).masked_fill(~mask[None], 0.0)
    partial_m = (partial_m + value_m[:, :, None]).masked_fill(~mask[None], 0.0)
    return projected, partial_p, partial_m


def reconstruct_variance(
    *, tree: float, minibatch: float, pair: float, node: float, B: int, n: int
) -> float:
    if B <= 0 or n <= 0:
        raise ValueError("B and n must be positive")
    return (
        float(tree)
        + float(minibatch) / B
        + ((1.0 - 1.0 / B) * float(pair) + float(node) / B) / n
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected {SCHEMA_VERSION} experiment config")
    if raw.get("checkpoint") != 1110:
        raise ValueError("the approved experiment contains only checkpoint 1110")
    coarse = raw.get("beta", {}).get("coarse")
    if coarse != [2.0, 2.25, 2.5]:
        raise ValueError("approved coarse beta grid is [2, 2.25, 2.5]")
    rollout = raw.get("rollout", {})
    if rollout.get("roots_per_gpu") != 1:
        raise ValueError("branched mode requires exactly one rooted tree")
    if int(rollout["branching_upper_cutoff_nodes"]) <= 1:
        raise ValueError("upper cutoff must be greater than one")
    if raw["probes"].get("distribution") != "rademacher":
        raise ValueError("only Rademacher probes are supported")
    for key in ("B", "n"):
        values = raw["report"][key]
        if not values or any(isinstance(v, bool) or int(v) <= 0 for v in values):
            raise ValueError(f"report.{key} must contain positive integers")
    return raw


def manifest_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    scientific = json.loads(json.dumps(config))
    checkpoint = Path(str(config["checkpoint_dir"])) / "ckpt_0001110.pt"
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    encoded = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "config": scientific,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "pbg_git_revision": revision,
        "torch_version": torch.__version__,
    }


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def ensure_manifest(config: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    target = Path(str(config["out_dir"])) / "manifest.json"
    wanted = manifest_payload(config)
    if target.exists():
        current = json.loads(target.read_text(encoding="utf-8"))
        if current.get("fingerprint") != wanted["fingerprint"]:
            raise RuntimeError("existing branched experiment manifest has different scientific identity")
        if not resume:
            raise FileExistsError(f"{target} exists; pass --resume")
        return current
    atomic_json(target, wanted)
    return wanted


def beta_key(beta: float) -> str:
    return f"beta_{float(beta):.3f}".replace(".", "p")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def geometry_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "nodes", "decisions", "leaves", "rollout_seconds",
        "peak_allocated_bytes", "upper_cutoff_activated",
        "upper_cutoff_first_node_count", "upper_cutoff_avoided_children",
    )
    result: dict[str, Any] = {}
    for key in keys:
        values = [float(record[key]) for record in records]
        result[key] = {
            "mean": statistics.fmean(values),
            "p50": _quantile(values, 0.5),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
            "max": max(values),
        }
    return result


def _tree_layout(tree: Any) -> dict[str, Any]:
    parent = tree.parent.detach().to("cpu", torch.int64).tolist()
    count = tree.degree.detach().to("cpu", torch.int64)
    counts = count.tolist()
    nodes = len(parent)
    depth = [0] * nodes
    for node in range(nodes):
        if parent[node] >= 0:
            if parent[node] >= node:
                raise RuntimeError("branched topology is not parent-before-child")
            depth[node] = depth[parent[node]] + 1
    by_depth = [torch.tensor([n for n, d in enumerate(depth) if d == level], dtype=torch.int64) for level in range(max(depth) + 1)]
    begin = tree.child_begin.detach().to("cpu", torch.int64).tolist()
    children = [
        list(range(begin[node], begin[node] + counts[node])) if counts[node] else []
        for node in range(nodes)
    ]
    degree = torch.tensor(counts, dtype=torch.float32, device=tree.parent.device)
    return {"depth": depth, "by_depth": by_depth, "children": children, "degree": degree}


def _node_moments(
    tree: Any, leaf_ids: Tensor, leaf_p: Tensor, leaf_m: Tensor, layout: Mapping[str, Any]
) -> tuple[Tensor, Tensor]:
    nodes = int(tree.parent.numel())
    players, hands = leaf_p.shape[1:]
    node_p = torch.zeros((nodes, players, hands), dtype=torch.float32, device=leaf_p.device)
    node_m = torch.zeros_like(node_p)
    node_p.index_copy_(0, leaf_ids.to(torch.int64), leaf_p)
    node_m.index_copy_(0, leaf_ids.to(torch.int64), leaf_m)
    for host_nodes in reversed(layout["by_depth"][:-1]):
        parents = host_nodes.to(leaf_p.device)
        nonleaf = layout["degree"][parents] > 0
        parents = parents[nonleaf]
        if not parents.numel():
            continue
        selected = parents.detach().cpu().tolist()
        child_lists = [layout["children"][parent] for parent in selected]
        flat = torch.tensor([child for row in child_lists for child in row], dtype=torch.int64, device=leaf_p.device)
        local = torch.repeat_interleave(
            torch.arange(parents.numel(), device=leaf_p.device),
            torch.tensor([len(row) for row in child_lists], device=leaf_p.device),
        )
        reduced_p = torch.zeros((parents.numel(), players, hands), dtype=torch.float32, device=leaf_p.device)
        reduced_m = torch.zeros_like(reduced_p)
        reduced_p.index_add_(0, local, node_p[flat])
        reduced_m.index_add_(0, local, node_m[flat])
        divisor = torch.tensor([len(row) for row in child_lists], dtype=torch.float32, device=leaf_p.device)[:, None, None]
        node_p.index_copy_(0, parents, reduced_p / divisor)
        node_m.index_copy_(0, parents, reduced_m / divisor)
    return node_p, node_m


def _decision_edges(sample: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    tree = sample.topology
    decision_nodes = sample.decision_node
    if decision_nodes is None or sample.action_ptr is None or sample.action_source is None:
        raise RuntimeError("branched direct seal omitted decision/action addresses")
    device = tree.parent.device
    ptr = sample.action_ptr.detach().to("cpu", torch.int64).tolist()
    child_event = sample.action_source["event"].detach().to("cpu", torch.int64).tolist()
    rows = decision_nodes.detach().to("cpu", torch.int64).tolist()
    counts = torch.tensor(
        [ptr[row + 1] - ptr[row] for row in range(len(rows))],
        dtype=torch.int64,
        device=device,
    )
    width = int(counts.max())
    children = torch.zeros((len(rows), width), dtype=torch.int64, device=device)
    events = torch.zeros((len(rows), width), dtype=torch.int64, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for row, node in enumerate(rows):
        count = ptr[row + 1] - ptr[row]
        first = int(tree.child_begin[node])
        children[row, :count] = torch.arange(first, first + count, device=device)
        events[row, :count] = torch.tensor(child_event[ptr[row] : ptr[row + 1]], device=device)
        mask[row, :count] = True
    return children, events, mask, counts


def _model_directional_rows(
    bundle: Any,
    decision_inputs: Any,
    events: Tensor,
    event_mask: Tensor,
    directions: Sequence[Mapping[str, Tensor]],
    maximum_batch_rows: int,
) -> tuple[Tensor, Tensor]:
    """Return ``dV=[Q,D,H]`` and selected ``dlogpi=[Q,D,E,H]``."""

    backend = bundle.rollout_backend
    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = decision_inputs.node_ids.device
    probes = len(directions)
    decisions, edge_width = events.shape
    d_value = torch.empty(
        (probes, decisions, backend.n_hands), dtype=torch.float32, device=device
    )
    d_logp = torch.empty(
        (probes, decisions, edge_width, backend.n_hands),
        dtype=torch.float32,
        device=device,
    )
    parameters = dict(model.named_parameters())
    stacked = {name: torch.stack([direction[name] for direction in directions]) for name in parameters}
    source_rows = torch.arange(decisions, dtype=torch.int64, device=device)
    selected_partitions = decision_inputs.partition_id.to(torch.int64)
    for partition_id in selected_partitions.unique(sorted=True).tolist():
        selected_group = (selected_partitions == partition_id).nonzero(
            as_tuple=False
        ).squeeze(1)
        for selected in selected_group.split(maximum_batch_rows):
            inputs = decision_inputs.gather_partition_inputs(
                int(partition_id),
                decision_inputs.partition_row[source_rows[selected]],
            )
            legal_mask = inputs["legal_action_mask"].to(torch.bool)
            legal_ids = torch.arange(
                backend.e_max, dtype=torch.int32, device=device
            )[None].expand(selected.numel(), -1)
            chosen = events[selected]
            chosen_mask = event_mask[selected]
            safe_column = legal_mask.to(torch.int64).argmax(1, keepdim=True)
            safe_event = legal_ids.gather(1, safe_column).expand_as(chosen)
            safe = torch.where(chosen_mask, chosen, safe_event)

            def call(current: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
                output = backend.functional_policy_forward_partition(
                    model, current, int(partition_id), inputs
                )
                return output.v, bundle.policy_adapter.log_prob(
                    output, safe, legal_ids, legal_mask
                )

            def tangent(direction: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
                return torch.func.jvp(call, (parameters,), (direction,))[1]

            with torch.no_grad():
                dv, dl = torch.vmap(tangent)(stacked)
            d_value[:, selected] = dv.to(torch.float32)
            d_logp[:, selected] = dl.to(torch.float32).masked_fill(~chosen_mask[None, :, :, None], 0.0)
    return d_value, d_logp


def _propagate_functional_coefficients(
    *,
    tree: Any,
    layout: Mapping[str, Any],
    children: Tensor,
    child_mask: Tensor,
    owners: Tensor,
    local_p: Tensor,
    local_m: Tensor,
    functional_weights: Tensor,
    leaf_ids: Tensor,
    players: int,
) -> tuple[Tensor, Tensor]:
    """Propagate local child-summary derivatives to leaf completion rows."""

    functionals, decisions = functional_weights.shape
    nodes = int(tree.parent.numel())
    hands = local_p.shape[-1]
    if local_p.shape[:2] != children.shape or local_m.shape != local_p.shape:
        raise ValueError("local partials are not action aligned")
    direct_p = torch.zeros((functionals, nodes, players, hands), dtype=torch.float32, device=children.device)
    direct_m = torch.zeros_like(direct_p)
    valid = child_mask.nonzero(as_tuple=False)
    decision_row, edge_slot = valid[:, 0], valid[:, 1]
    child = children[decision_row, edge_slot]
    owner = owners[decision_row]
    scale = functional_weights[:, decision_row, None].to(local_p)
    direct_p[:, child, owner] = scale * local_p[decision_row, edge_slot][None]
    direct_m[:, child, owner] = scale * local_m[decision_row, edge_slot][None]
    parent = tree.parent.to(torch.int64)
    degree = layout["degree"]
    for host_nodes in layout["by_depth"][1:]:
        selected = host_nodes.to(children.device)
        parents = parent[selected]
        divisor = degree[parents][None, :, None, None]
        direct_p[:, selected] += direct_p[:, parents] / divisor
        direct_m[:, selected] += direct_m[:, parents] / divisor
    return direct_p[:, leaf_ids.to(torch.int64)], direct_m[:, leaf_ids.to(torch.int64)]


def _slice_rows(values: Mapping[str, Tensor], first: int, stop: int) -> dict[str, Tensor]:
    return {key: value[first:stop].contiguous() for key, value in values.items()}


class BranchedVarianceRunner:
    """One-device checkpoint worker shared by smoke and resumable stages."""

    def __init__(self, experiment: Mapping[str, Any], device: torch.device) -> None:
        from pbg.config import load_config_mapping
        from pbg.core.checkpointing import inspect_checkpoint
        from pbg.games.poker.bundle import build_bundle

        self.experiment = experiment
        self.device = device
        checkpoint = Path(str(experiment["checkpoint_dir"])) / "ckpt_0001110.pt"
        metadata = inspect_checkpoint(checkpoint)
        raw = copy.deepcopy(metadata["config"])
        game = raw["game"]
        if (game["n_players"], float(game["stack_bb"]), game["rank_count"]) != (2, 32.0, 13):
            raise RuntimeError("checkpoint is not heads-up, 32 bb, full-deck NLHE")
        rollout = experiment["rollout"]
        raw["rollout"]["sequential_roots_per_gpu_initial"] = 1
        raw["rollout"]["sequential_roots_per_gpu_minimum"] = 1
        raw["rollout"]["sequential_roots_per_gpu_maximum"] = 1
        raw["rollout"]["storage_page_rows"] = int(
            raw["rollout"].get("storage_page_rows", 256)
        )
        raw["rollout"]["beta"] = {"default": 1.0}
        raw["rollout"]["branching_mode"] = "fixed"
        raw["rollout"].pop("adaptive_kkt", None)
        raw["training"]["mode"] = "branched_rao_blackwell"
        raw["model"]["use_branching_heads"] = True
        raw["training"]["amp"] = False
        raw["training"]["reproducibility"] = "strict"
        raw["training"].setdefault("checkpoint_every", 1)
        raw["completion"]["backend"] = "cuda" if device.type == "cuda" else "cpu"
        raw["completion"]["n_samples"] = int(experiment["oracle"]["n_oracle"])
        raw["loss"]["lam"] = 1.0
        raw["loss"]["c_v"] = 0.0005
        raw["loss"]["c_h"] = 0.0
        raw["loss"]["c_kl"] = 0.0
        raw["loss"]["c_t"] = 1.0
        raw["loss"]["c_c"] = 1.0
        raw["loss"]["value_loss"] = "mse"
        hardware = "configs/hardware/local_3x2080ti.yaml" if device.type == "cuda" else None
        self.cfg = load_config_mapping(raw, hardware_profile=hardware)
        self.cfg.rollout.roots_per_gpu = 1
        self.diagnostic_limits = BranchingDiagnosticLimits(
            upper_nodes=int(rollout["branching_upper_cutoff_nodes"]),
            stop_nodes=rollout.get("branching_stop_cutoff_nodes"),
        )
        if self.cfg.hardware is not None:
            self.cfg.hardware.attention_backend = "chunked_dense"
            self.cfg.hardware.query_chunk_size = min(self.cfg.hardware.query_chunk_size, 16)
        self.bundle = build_bundle(self.cfg, device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        incompatible = self.bundle.model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {
            name for name in self.bundle.model.state_dict()
            if ".branching_heads." in name or name.startswith("branching_heads.")
        }
        missing = set(incompatible.missing_keys)
        if incompatible.unexpected_keys or missing not in (set(), allowed_missing):
            raise RuntimeError(
                "variance checkpoint is incompatible beyond explicitly absent T/C "
                f"heads; missing={sorted(missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        if missing:
            warnings.warn(
                "variance run explicitly initialized T/C heads because its source "
                "checkpoint predates mandatory branched heads",
                RuntimeWarning,
                stacklevel=2,
            )
        del payload
        self.model = self.bundle.model.module if hasattr(self.bundle.model, "module") else self.bundle.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.evaluator = self.bundle.completion_backend._bound_evaluator()
        if str(getattr(self.evaluator.info, "contract_version", "")) != "2":
            raise RuntimeError("complete_linear_moments requires evaluator contract 2")
        self.memory_limit = int(float(experiment["constraints"]["maximum_allocated_gib"]) * 1024**3)
        probe_count = int(experiment["probes"]["parameter_count"])
        self.directions = [
            rademacher_direction(
                self.model,
                logical_seed(int(experiment["logical_seed"]), "branched-parameter-probe", probe),
            )
            for probe in range(probe_count)
        ]

    def memory_gate(self) -> None:
        if self.device.type == "cuda" and torch.cuda.max_memory_allocated(self.device) > self.memory_limit:
            raise MemoryError(
                f"allocated peak {torch.cuda.max_memory_allocated(self.device) / 1024**3:.3f} GiB "
                f"exceeds {self.memory_limit / 1024**3:.3f} GiB"
            )

    def sample_tree(
        self,
        beta: float,
        repeat: int,
        *,
        upper_cutoff: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        from pbg.core.branching import sample_branched_tree

        seed = logical_seed(int(self.experiment["logical_seed"]), "branched-public-tree", 1110, repeat)
        self.bundle.rollout_backend.beta_table = {"default": float(beta)}
        self.cfg.rollout.logical_seed = seed
        self.cfg.training.seed = seed
        diagnostic_limits = BranchingDiagnosticLimits(
            upper_nodes=(
                self.diagnostic_limits.upper_nodes
                if upper_cutoff is None
                else int(upper_cutoff)
            ),
            stop_nodes=self.diagnostic_limits.stop_nodes,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        diagnostics: dict[str, int] = {}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            sample = sample_branched_tree(
                self.bundle,
                self.cfg,
                generator,
                iteration=1110,
                diagnostics=diagnostics,
                diagnostic_limits=diagnostic_limits,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        tree = sample.topology
        degree = tree.degree
        leaf = degree == 0
        layout = _tree_layout(tree)
        weights = torch.ones(tree.nodes, dtype=torch.float64, device=self.device)
        for host_nodes in layout["by_depth"][1:]:
            selected = host_nodes.to(self.device)
            parent = tree.parent[selected].to(torch.int64)
            weights[selected] = weights[parent] / degree[parent]
        geometry = {
            "beta": float(beta),
            "repeat": int(repeat),
            "seed": int(seed),
            "nodes": sample.node_count,
            "decisions": sample.decision_source.rows,
            "leaves": int(leaf.sum()),
            "max_offspring": int(degree.max()),
            "max_player_depth": max(layout["depth"]),
            "leaf_weight_sum": float(weights[leaf].sum()),
            "rollout_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)) if self.device.type == "cuda" else 0,
            "upper_cutoff_activated": int(diagnostics["upper_cutoff_batches"] > 0),
            "upper_cutoff_batches": int(diagnostics["upper_cutoff_batches"]),
            "upper_cutoff_first_node_count": int(diagnostics["upper_cutoff_first_node_count"]),
            "upper_cutoff_avoided_children": int(diagnostics["upper_cutoff_avoided_children"]),
        }
        self.memory_gate()
        return sample, geometry

    def _complete_leaves(
        self, sample: Any, beta: float, repeat: int, n_samples: int
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, float]:
        tree = sample.topology
        leaf_nodes = (tree.leaf_row >= 0).nonzero(as_tuple=False).squeeze(1)
        leaf_ids = leaf_nodes[torch.argsort(tree.leaf_row[leaf_nodes])]
        leaf_reach = sample.leaf_source["log_reach"]
        captured_leaf = sample.leaf_source.columns
        inputs = {
            name: value
            for name, value in captured_leaf.items()
            if name
            not in {
                "node_id",
                "root_id",
                "path_key",
                "log_reach",
                "owner",
                "fold_payoff",
            }
        }
        inputs["range_weights"] = (
            self.bundle.rollout_backend.range_weights_from_log_reach(leaf_reach)
        )
        seeds = torch.tensor(
            [logical_seed(int(self.experiment["logical_seed"]), "branched-oracle", 1110, float(beta), repeat, leaf) for leaf in range(leaf_ids.numel())],
            dtype=torch.int64,
            device=self.device,
        )
        rows = int(self.experiment["oracle"]["leaf_rows_per_batch"])
        p_parts, m_parts = [], []
        minimum_ess = math.inf
        for first in range(0, leaf_ids.numel(), rows):
            stop = min(leaf_ids.numel(), first + rows)
            result = self.evaluator.complete(
                **_slice_rows(inputs, first, stop),
                n_samples=int(n_samples),
                seed=seeds[first:stop].contiguous(),
            )
            p = result.p_hat.to(torch.float32)
            p_parts.append(p)
            m_parts.append(p * result.v_hat.to(torch.float32))
            minimum_ess = min(minimum_ess, float(result.n_eff.min()))
        return leaf_ids, torch.cat(p_parts), torch.cat(m_parts), inputs, seeds, minimum_ess

    def moment_tree(self, beta: float, repeat: int, *, n_samples: int | None = None) -> dict[str, Any]:
        n_samples = int(n_samples or self.experiment["oracle"]["n_oracle"])
        started = time.perf_counter()
        sample, geometry = self.sample_tree(beta, repeat)
        tree = sample.topology
        rollout_done = time.perf_counter()
        decisions = sample.decision_node
        if decisions is None:
            raise RuntimeError("branched direct seal omitted compact decision rows")
        decisions = decisions.to(torch.int64)
        children, events, child_mask, child_count = _decision_edges(sample)
        directions_started = time.perf_counter()
        d_value, d_logp = _model_directional_rows(
            self.bundle,
            sample.decision_inputs,
            events,
            child_mask,
            self.directions,
            int(self.experiment["streaming"]["directional_rows_per_batch"]),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        direction_done = time.perf_counter()
        self.memory_gate()
        leaf_ids, leaf_p, leaf_m, completion_inputs, completion_seeds, minimum_ess = self._complete_leaves(
            sample, beta, repeat, n_samples
        )
        completion_done = time.perf_counter()
        layout = _tree_layout(tree)
        node_p, node_m = _node_moments(tree, leaf_ids, leaf_p, leaf_m, layout)
        owners = sample.decision_aux["owner"].to(torch.int64)
        gather_owner = owners[:, None, None].expand(-1, children.shape[1], node_p.shape[2])
        child_p = node_p[children].gather(2, gather_owner[:, :, None, :]).squeeze(2)
        child_m = node_m[children].gather(2, gather_owner[:, :, None, :]).squeeze(2)
        projected, partial_p, partial_m = projected_loss_and_child_partials(
            child_p,
            child_m,
            sample.decision_source["old_v"],
            d_value,
            d_logp,
            child_mask,
            child_count,
            c_v=0.0005,
        )
        decision_weight = sample.decision_aux["log_weight"].exp().to(torch.float64)
        length = decision_weight.sum()
        gradient = (projected.double() * decision_weight[None]).sum(1)
        minibatch = (
            length * (projected.double().square() * decision_weight[None]).sum(1)
            - gradient.square()
        )
        node_hutchinson = int(self.experiment["probes"]["node_hutchinson_count"])
        pair_coefficients: list[float] = []
        node_coefficients: list[float] = []
        oracle_mean_variance: list[list[float]] = []
        moment_started = time.perf_counter()
        oracle_rows = int(self.experiment["oracle"]["leaf_rows_per_batch"])
        for probe in range(projected.shape[0]):
            z_rows = []
            for node_probe in range(node_hutchinson):
                seed = logical_seed(
                    int(self.experiment["logical_seed"]),
                    "branched-node-probe", probe, node_probe, float(beta), repeat,
                )
                generator = torch.Generator(device="cpu").manual_seed(seed)
                z = torch.randint(0, 2, (decisions.numel(),), generator=generator, dtype=torch.int8)
                z_rows.append(z.to(self.device, torch.float64).mul_(2).sub_(1))
            functional_weights = torch.cat(
                (
                    decision_weight[None],
                    torch.stack(z_rows) * torch.sqrt(length * decision_weight)[None],
                )
            )
            leaf_coeff_p, leaf_coeff_m = _propagate_functional_coefficients(
                tree=tree,
                layout=layout,
                children=children,
                child_mask=child_mask,
                owners=owners,
                local_p=partial_p[probe],
                local_m=partial_m[probe],
                functional_weights=functional_weights,
                leaf_ids=leaf_ids,
                players=int(self.bundle.rollout_backend.n_players),
            )
            world_sum = torch.zeros(1 + node_hutchinson, dtype=torch.float64)
            mean_var_sum = torch.zeros_like(world_sum)
            for first in range(0, leaf_ids.numel(), oracle_rows):
                stop = min(leaf_ids.numel(), first + oracle_rows)
                result = self.evaluator.complete_linear_moments(
                    **_slice_rows(completion_inputs, first, stop),
                    p_coefficient=leaf_coeff_p[:, first:stop].permute(1, 0, 2, 3).contiguous(),
                    m_coefficient=leaf_coeff_m[:, first:stop].permute(1, 0, 2, 3).contiguous(),
                    functional_mask=torch.ones((stop - first, 1 + node_hutchinson), dtype=torch.bool, device=self.device),
                    n_samples=n_samples,
                    seed=completion_seeds[first:stop].contiguous(),
                )
                if not bool(result.functional_valid.all()):
                    raise RuntimeError("linear-moment oracle returned an invalid functional")
                world_sum += result.world_variance.double().sum(0).cpu()
                mean_var_sum += result.mean_variance.double().sum(0).cpu()
            pair_coefficients.append(float(world_sum[0]))
            node_coefficients.append(float(world_sum[1:].mean()))
            oracle_mean_variance.append(mean_var_sum.tolist())
            del leaf_coeff_p, leaf_coeff_m
            self.memory_gate()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        completed = time.perf_counter()
        return {
            "schema_version": SCHEMA_VERSION,
            "success": True,
            "checkpoint": 1110,
            "beta": float(beta),
            "repeat": int(repeat),
            "oracle_samples": n_samples,
            "minimum_effective_sample_size": minimum_ess,
            "geometry": geometry,
            "gradient": gradient.cpu().tolist(),
            "minibatch_coefficient_raw": minibatch.cpu().tolist(),
            "completion_pair_coefficient": pair_coefficients,
            "completion_node_coefficient": node_coefficients,
            "oracle_mean_variance": oracle_mean_variance,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)) if self.device.type == "cuda" else 0,
            "seconds": completed - started,
            "timings": {
                "rollout_seconds": rollout_done - started,
                "directional_seconds": direction_done - directions_started,
                "completion_seconds": completion_done - direction_done,
                "moment_seconds": completed - moment_started,
            },
        }


def component_summary(records: Sequence[Mapping[str, Any]], n_oracle: int) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("at least two moment trees are required")
    probes = len(records[0]["gradient"])
    per_probe = []
    for probe in range(probes):
        gradients = torch.tensor([row["gradient"][probe] for row in records], dtype=torch.float64)
        pair = statistics.fmean(float(row["completion_pair_coefficient"][probe]) for row in records)
        node = statistics.fmean(float(row["completion_node_coefficient"][probe]) for row in records)
        raw_tree = float(gradients.var(unbiased=True))
        raw_mb = statistics.fmean(float(row["minibatch_coefficient_raw"][probe]) for row in records)
        corrected_tree = raw_tree - pair / n_oracle
        corrected_mb = raw_mb - (node - pair) / n_oracle
        per_probe.append(
            {
                "tree_raw": raw_tree,
                "tree_corrected_unclamped": corrected_tree,
                "tree": max(0.0, corrected_tree),
                "minibatch_raw": raw_mb,
                "minibatch_corrected_unclamped": corrected_mb,
                "minibatch": max(0.0, corrected_mb),
                "completion_pair": pair,
                "completion_node": node,
            }
        )
    mean = {
        key: statistics.fmean(row[key] for row in per_probe)
        for key in per_probe[0]
    }
    # Probe-to-probe SE is conservative for the trace estimate and is the
    # adaptive stopping signal. Public-tree resampling uncertainty remains in
    # every probe and is also exposed through the raw records.
    relative_se = {}
    for key in ("tree", "minibatch", "completion_pair", "completion_node"):
        values = [row[key] for row in per_probe]
        standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.inf
        relative_se[key] = standard_error / max(abs(mean[key]), 1e-30)
    return {"trees": len(records), "per_probe": per_probe, "mean": mean, "relative_se": relative_se}


def variance_table(
    components: Mapping[str, float], B_values: Sequence[int], n_values: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for B in B_values:
        for n in n_values:
            rows.append(
                {
                    "B": int(B),
                    "n": int(n),
                    "tree": float(components["tree"]),
                    "minibatch": float(components["minibatch"]) / int(B),
                    "completion": (
                        (1.0 - 1.0 / int(B)) * float(components["completion_pair"])
                        + float(components["completion_node"]) / int(B)
                    ) / int(n),
                    "total": reconstruct_variance(
                        tree=float(components["tree"]),
                        minibatch=float(components["minibatch"]),
                        pair=float(components["completion_pair"]),
                        node=float(components["completion_node"]),
                        B=int(B),
                        n=int(n),
                    ),
                }
            )
    return rows
