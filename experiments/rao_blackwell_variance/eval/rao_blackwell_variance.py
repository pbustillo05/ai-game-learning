"""Rao--Blackwell gradient-variance experiment utilities.

The expensive runner reduces every directional gradient probe to the seven
scalars in eq. (20) of ``experiments/rao_blackwell_variance/docs/sequential.md``.
This module owns the mergeable representation and all subsequent analysis;
none of these routines require CUDA or retain hand-width tensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch import nn
import yaml


SCHEMA_VERSION = "pbg-rb-variance-v1"


def logical_seed(base: int, purpose: str, *address: object) -> int:
    """Derive a stable positive int64 seed without worker-rank dependence."""

    payload = json.dumps(
        [int(base), str(purpose), *address], separators=(",", ":"), sort_keys=True
    ).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value & ((1 << 63) - 1)


def checkpoint_sha256(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_jobs(checkpoints: Sequence[int], superbatches: int) -> tuple[tuple[int, int], ...]:
    if superbatches <= 0:
        raise ValueError("superbatches must be positive")
    return tuple((int(checkpoint), batch) for checkpoint in checkpoints for batch in range(superbatches))


def assigned_jobs(
    checkpoints: Sequence[int], superbatches: int, rank: int, world_size: int
) -> tuple[tuple[int, int], ...]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid worker rank/world size")
    return tuple(job for index, job in enumerate(canonical_jobs(checkpoints, superbatches)) if index % world_size == rank)


def consecutive_blocks(total: int, block_size: int) -> tuple[tuple[int, int], ...]:
    """Return the predetermined disjoint half-open N-blocks."""

    if total <= 0 or block_size <= 0 or total % block_size:
        raise ValueError("total must be a positive multiple of block_size")
    return tuple((first, first + block_size) for first in range(0, total, block_size))


def weighted_ratio_moments(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Equation (9): mean, conditional-world variance, and mean variance."""

    if values.shape != weights.shape or values.ndim == 0:
        raise ValueError("values and weights must have the same non-scalar shape")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("values and weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative")
    total = weights.sum(-1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("each row must have positive weight")
    q = weights / total
    mean = (q * values).sum(-1)
    centered = values - mean[..., None]
    correction = 1.0 - q.square().sum(-1)
    if bool((correction <= 0).any()):
        raise ValueError("weighted variance requires effective sample size above one")
    world = (q * centered.square()).sum(-1) / correction
    mean_variance = (q.square() * centered.square()).sum(-1) / correction
    return mean, world, mean_variance


def rademacher_direction(model: nn.Module, seed: int) -> dict[str, Tensor]:
    """Generate one parameter-space probe lazily from a stable CPU seed."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        draw = torch.randint(0, 2, parameter.shape, generator=generator, dtype=torch.int8)
        result[name] = draw.to(device=parameter.device, dtype=parameter.dtype).mul_(2).sub_(1)
    return result


def parameter_jvp(
    model: nn.Module, forward: Any, seed: int
) -> tuple[Any, Any, dict[str, Tensor]]:
    """Evaluate an exact parameter JVP while treating buffers/inputs as frozen."""

    parameters = dict(model.named_parameters())
    direction = rademacher_direction(model, seed)

    def call(current: Mapping[str, Tensor]) -> Any:
        return torch.func.functional_call(model, current, (), strict=True) if forward is None else forward(current)

    primal, tangent = torch.func.jvp(call, (parameters,), (direction,))
    return primal, tangent, direction


@dataclass
class SufficientStatistics:
    """Mergeable scalar state for one (checkpoint, superbatch, N-block, probe)."""

    trajectories: int = 0
    decisions: int = 0
    sum_mu: float = 0.0
    sum_mu2: float = 0.0
    sum_v_node: float = 0.0
    sum_v_traj: float = 0.0
    sum_e_node: float = 0.0
    sum_e_traj: float = 0.0
    trajectory_means: list[float] = field(default_factory=list)

    def update_trajectory(
        self,
        node_mean: Sequence[float] | Tensor,
        node_world_variance: Sequence[float] | Tensor,
        node_mean_variance: Sequence[float] | Tensor,
        trajectory_mean: float,
        trajectory_world_variance: float,
        trajectory_mean_variance: float,
    ) -> None:
        mu = torch.as_tensor(node_mean, dtype=torch.float64).flatten()
        v = torch.as_tensor(node_world_variance, dtype=torch.float64).flatten()
        e = torch.as_tensor(node_mean_variance, dtype=torch.float64).flatten()
        if not (mu.numel() and mu.shape == v.shape == e.shape):
            raise ValueError("node moment rows must be nonempty and aligned")
        scalars = torch.tensor(
            [trajectory_mean, trajectory_world_variance, trajectory_mean_variance],
            dtype=torch.float64,
        )
        if not bool(torch.isfinite(torch.cat((mu, v, e, scalars))).all()):
            raise ValueError("moment statistics must be finite")
        if bool((v < 0).any()) or bool((e < 0).any()) or trajectory_world_variance < 0 or trajectory_mean_variance < 0:
            raise ValueError("variance statistics must be nonnegative")
        self.trajectories += 1
        self.decisions += int(mu.numel())
        self.sum_mu += float(mu.sum())
        self.sum_mu2 += float(mu.square().sum())
        self.sum_v_node += float(v.sum())
        self.sum_v_traj += float(trajectory_world_variance)
        self.sum_e_node += float(e.sum())
        self.sum_e_traj += float(trajectory_mean_variance)
        self.trajectory_means.append(float(trajectory_mean))

    def merge(self, other: "SufficientStatistics") -> "SufficientStatistics":
        for name in (
            "trajectories", "decisions", "sum_mu", "sum_mu2", "sum_v_node",
            "sum_v_traj", "sum_e_node", "sum_e_traj",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.trajectory_means.extend(other.trajectory_means)
        return self

    def components(self) -> dict[str, float]:
        """Return corrected eqs. (21)--(24) and rollout endpoint."""

        n, length = self.trajectories, self.decisions
        if n < 2 or len(self.trajectory_means) != n:
            raise ValueError("at least two complete trajectory rows are required")
        scale = float(n * n)
        raw_mb = (length * self.sum_mu2 - self.sum_mu * self.sum_mu) / scale
        correction_mb = (length * self.sum_e_node - self.sum_e_traj) / scale
        minibatch = raw_mb - correction_mb
        mean = sum(self.trajectory_means) / n
        sample_variance = sum((value - mean) ** 2 for value in self.trajectory_means) / (n - 1)
        rollout_one = sample_variance - self.sum_e_traj / n
        return {
            "rollout_1": rollout_one,
            "minibatch_N_1_raw": raw_mb,
            "minibatch_noise_correction": correction_mb,
            "minibatch_N_1": minibatch,
            "montecarlo_N_inf": self.sum_v_traj / scale,
            "montecarlo_N_1": length * self.sum_v_node / scale,
        }

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fromdict(cls, raw: Mapping[str, Any]) -> "SufficientStatistics":
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})


def theoretical_variance(
    *, rollout_1: float, minibatch_N_1: float, montecarlo_N_inf: float,
    montecarlo_N_1: float, N: int, B: int, n: int,
) -> float:
    """Reconstruct equation (11) for a candidate geometry."""

    if min(N, B, n) <= 0:
        raise ValueError("N, B, and n must be positive")
    return (
        rollout_1 / N
        + minibatch_N_1 / B
        + ((1.0 - 1.0 / B) * montecarlo_N_inf + montecarlo_N_1 / B) / n
    )


def variance_cube(
    components_by_N: Mapping[int, Mapping[str, float]],
    candidates: Mapping[str, Sequence[int]],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for N in candidates["N"]:
        c = components_by_N[int(N)]
        for B in candidates["B"]:
            for n in candidates["n"]:
                rows.append({
                    "N": int(N), "B": int(B), "n": int(n),
                    "variance": theoretical_variance(
                        rollout_1=float(c["rollout_1"]),
                        minibatch_N_1=float(c["minibatch_N_1"]),
                        montecarlo_N_inf=float(c["montecarlo_N_inf"]),
                        montecarlo_N_1=float(c["montecarlo_N_1"]),
                        N=int(N), B=int(B), n=int(n),
                    ),
                })
    return rows


def cpu_oracle() -> dict[str, float]:
    """Enumerate a tiny shared-completion problem and verify the identities."""

    # Two public trajectories, respectively two and one node. Two equiprobable
    # worlds per trajectory; nodes in trajectory zero are deliberately correlated.
    worlds = (((1.0, 2.0), (3.0, 4.0)), ((-1.0,), (1.0,)))
    node_mu, node_v, traj_mu, traj_v = [], [], [], []
    stats = SufficientStatistics()
    for trajectory in worlds:
        samples = torch.tensor(trajectory, dtype=torch.float64)
        mu = samples.mean(0)
        # These are the exact finite-world law moments, not estimates from a
        # sampled oracle, so use the population variance.
        variance = samples.var(0, unbiased=False)
        sums = samples.sum(1)
        stats.update_trajectory(mu, variance, torch.zeros_like(mu), float(sums.mean()), float(sums.var(unbiased=False)), 0.0)
        node_mu.extend(mu.tolist()); node_v.extend(variance.tolist())
        traj_mu.append(float(sums.mean())); traj_v.append(float(sums.var(unbiased=False)))
    components = stats.components()
    # Directly enumerate all ordered B=2 decision draws and all shared worlds.
    L, N, B = 3, 2, 2
    values = []
    exact_means = []
    nodes = ((0, 0), (0, 1), (1, 0))
    for left in nodes:
        for right in nodes:
            conditional = []
            for w0 in range(2):
                for w1 in range(2):
                    chosen = {0: w0, 1: w1}
                    total = worlds[left[0]][chosen[left[0]]][left[1]] + worlds[right[0]][chosen[right[0]]][right[1]]
                    conditional.append(L / (N * B) * total)
            values.extend(conditional)
            exact_means.append(sum(conditional) / len(conditional))
    direct_completion = sum(
        sum((x - sum(row) / len(row)) ** 2 for x in row) / len(row)
        for row in [values[i:i + 4] for i in range(0, len(values), 4)]
    ) / 9
    predicted_completion = (
        (1 - 1 / B) * components["montecarlo_N_inf"]
        + components["montecarlo_N_1"] / B
    )
    if not math.isclose(direct_completion, predicted_completion, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("shared-completion covariance identity failed")
    return {"direct_completion": direct_completion, "predicted_completion": predicted_completion, **components}


def _model_directional_rows(
    bundle: Any,
    decision_inputs: Any,
    decision_rows: Tensor,
    events: Tensor,
    directions: Sequence[Mapping[str, Tensor]],
    c_v: float,
    maximum_batch_rows: int,
) -> tuple[Tensor, Tensor]:
    """Return current values and eq. (14) for one or more parameter probes."""

    backend = bundle.rollout_backend
    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    device = decision_inputs.node_ids.device
    decision_rows = decision_rows.to(device=device, dtype=torch.int64)
    values = torch.empty(
        (decision_rows.numel(), backend.n_hands),
        dtype=torch.float32,
        device=device,
    )
    if not directions:
        raise ValueError("at least one parameter direction is required")
    directional = torch.empty(
        (len(directions), *values.shape), dtype=torch.float32, device=values.device
    )
    parameters = dict(model.named_parameters())
    stacked_directions = {
        name: torch.stack([direction[name] for direction in directions])
        for name in parameters
    }

    selected_partitions = decision_inputs.partition_id[decision_rows].to(torch.int64)
    for partition_id in selected_partitions.unique(sorted=True).tolist():
        partition_positions = (selected_partitions == partition_id).nonzero(
            as_tuple=False
        ).squeeze(1)
        for selected in partition_positions.split(maximum_batch_rows):
            source_rows = decision_rows[selected]
            inputs = decision_inputs.gather_partition_inputs(
                int(partition_id), decision_inputs.partition_row[source_rows]
            )
            legal_mask = inputs["legal_action_mask"].to(torch.bool)
            legal_ids = torch.arange(
                backend.e_max, dtype=torch.int32, device=device
            )[None].expand(selected.numel(), -1)
            def call(current: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
                output = backend.functional_policy_forward_partition(
                    model, current, int(partition_id), inputs
                )
                logp = bundle.policy_adapter.log_prob(
                    output,
                    events[selected, None],
                    legal_ids,
                    legal_mask,
                )[:, 0]
                return output.v, logp

            # Forward-mode is the only derivative tape needed. Without this
            # guard, requires-grad checkpoint parameters also build a complete
            # reverse-mode graph and exceed the 11 GiB safety envelope.
            def tangent(current_direction: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
                return torch.func.jvp(
                    call, (parameters,), (current_direction,)
                )[1]

            with torch.no_grad():
                value, _ = call(parameters)
                d_value, d_logp = torch.vmap(tangent)(stacked_directions)
            values[selected] = value.detach().to(torch.float32)
            directional[:, selected] = (
                d_logp + 2.0 * float(c_v) * d_value
            ).detach().to(torch.float32)
    return values, directional


def run_variance_job(
    experiment: Mapping[str, Any], checkpoint: int, superbatch: int, device: torch.device
) -> dict[str, Any]:
    """Run one immutable ``(checkpoint, superbatch)`` streamed-moment job."""

    import copy
    import time

    from pbg.config import load_config_mapping
    from pbg.core.checkpointing import inspect_checkpoint
    from pbg.core.rollout import sample_trees_sequential
    from pbg.games.poker.bundle import build_bundle

    checkpoint_path = Path(str(experiment["checkpoint_dir"])) / f"ckpt_{int(checkpoint):07d}.pt"
    identity = checkpoint_sha256(checkpoint_path)
    metadata = inspect_checkpoint(checkpoint_path)
    raw = copy.deepcopy(metadata["config"])
    game = raw.get("game", {})
    if (game.get("n_players"), float(game.get("stack_bb", -1)), game.get("rank_count")) != (2, 32.0, 13):
        raise RuntimeError("checkpoint is not heads-up, full-deck, 32 bb NLHE")
    maximum_n = max(int(value) for value in experiment["candidates"]["N"])
    raw["rollout"]["sequential_roots_per_gpu_initial"] = maximum_n
    raw["rollout"]["sequential_roots_per_gpu_minimum"] = maximum_n
    raw["rollout"]["sequential_roots_per_gpu_maximum"] = maximum_n
    raw["rollout"]["beta"] = None
    raw["rollout"]["branching_mode"] = None
    raw["rollout"].pop("adaptive_kkt", None)
    raw["training"]["mode"] = "rao_blackwell"
    # Older checkpoints inherited this execution cadence from their hardware
    # profile, so it is absent from the scientific payload parsed by the
    # current strict schema. It is inert in this update-free experiment.
    raw["training"].setdefault("checkpoint_every", 1)
    raw["training"]["amp"] = False
    raw["training"]["reproducibility"] = "strict"
    raw["training"]["minibatch_size"] = max(2, int(raw["training"].get("minibatch_size", 2)))
    raw["device"] = device.type
    raw["completion"].pop("backend", None)
    raw["completion"]["n_samples"] = int(experiment["oracle"]["initial_samples"])
    raw["loss"]["lam"] = 1.0
    raw["loss"]["c_v"] = 0.0005
    raw["loss"]["c_h"] = 0.0
    raw["loss"]["c_kl"] = 0.0
    raw["loss"]["value_loss"] = "mse"
    hardware = "configs/hardware/local_3x2080ti.yaml" if device.type == "cuda" else None
    cfg = load_config_mapping(raw, hardware_profile=hardware)
    if cfg.hardware is not None:
        cfg.hardware.attention_backend = "chunked_dense"
        cfg.hardware.query_chunk_size = min(cfg.hardware.query_chunk_size, 16)
    rollout_seed = logical_seed(int(experiment["logical_seed"]), "public-rollout", int(checkpoint), int(superbatch))
    cfg.training.seed = rollout_seed
    setattr(cfg.rollout, "logical_seed", rollout_seed)
    setattr(cfg.rollout, "trajectory_start_index", 0)
    generator = torch.Generator(device=device)
    generator.manual_seed(rollout_seed)
    started = time.perf_counter()
    rollout_started = started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory_limit = int(float(experiment["constraints"]["maximum_allocated_gib"]) * 1024**3)

    def memory_gate() -> None:
        if device.type == "cuda" and torch.cuda.max_memory_allocated(device) > memory_limit:
            raise MemoryError(
                "variance worker exceeded configured allocated-memory ceiling: "
                f"{torch.cuda.max_memory_allocated(device) / 1024**3:.3f} GiB > "
                f"{memory_limit / 1024**3:.3f} GiB"
            )
    bundle = build_bundle(cfg, device)
    # Only model weights belong to this frozen, update-free experiment. Using
    # the training resume loader would also restore every saved CUDA RNG lane,
    # which is both scientifically wrong here and invalid when a multi-GPU
    # checkpoint is inspected through a one-GPU smoke visibility mask.
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle.model.load_state_dict(checkpoint_payload["model"])
    del checkpoint_payload
    model = bundle.model.module if hasattr(bundle.model, "module") else bundle.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    sampled = sample_trees_sequential(bundle, cfg, generator, iteration=int(checkpoint))
    rollout_seconds = time.perf_counter() - rollout_started
    decision_inputs = sampled.decision_inputs
    decision_source = sampled.decision_source
    trajectory_id = decision_source["root_id"].to(torch.int64)
    owner = decision_source["owner"].to(torch.int64)
    events = decision_source["event"].to(torch.int64)
    leaf_source = sampled.leaf_source
    leaf_tree = leaf_source["root_id"].to(torch.int64)
    if not torch.equal(leaf_tree, torch.arange(maximum_n, device=device)):
        raise RuntimeError("sequential rollout must produce one leaf per trajectory")
    leaf_reach = leaf_source["log_reach"]
    n_values = tuple(int(value) for value in experiment["candidates"]["N"])
    probes = int(experiment["probes"]["count"])
    probes_per_chunk = int(experiment["streaming"].get("probes_per_chunk", 1))
    if probes_per_chunk <= 0 or probes % probes_per_chunk:
        raise ValueError("streaming.probes_per_chunk must be a positive divisor of probe count")
    chunk_size = int(experiment["streaming"]["trajectories_per_chunk"])
    functional_bound = int(experiment["streaming"]["maximum_functionals_per_leaf"])
    directional_batch_rows = int(
        experiment["streaming"].get("directional_rows_per_batch", 16)
    )
    if directional_batch_rows <= 0:
        raise ValueError("streaming.directional_rows_per_batch must be positive")
    progress_every = int(experiment["streaming"].get("progress_every_chunks", 8))
    if progress_every <= 0:
        raise ValueError("streaming.progress_every_chunks must be positive")
    chunk_cooldown_seconds = float(
        os.environ.get("PBG_VARIANCE_CHUNK_COOLDOWN_SECONDS", "0")
    )
    if not math.isfinite(chunk_cooldown_seconds) or chunk_cooldown_seconds < 0:
        raise ValueError(
            "PBG_VARIANCE_CHUNK_COOLDOWN_SECONDS must be a finite nonnegative number"
        )
    duty_run_seconds = float(os.environ.get("PBG_VARIANCE_DUTY_RUN_SECONDS", "0"))
    duty_pause_seconds = float(
        os.environ.get("PBG_VARIANCE_DUTY_PAUSE_SECONDS", "0")
    )
    if (
        not math.isfinite(duty_run_seconds)
        or not math.isfinite(duty_pause_seconds)
        or duty_run_seconds < 0
        or duty_pause_seconds < 0
    ):
        raise ValueError("variance duty-cycle seconds must be finite and nonnegative")
    jvp_seconds = 0.0
    moment_seconds = 0.0
    accumulators: dict[tuple[int, int, int], SufficientStatistics] = {}
    for n_value in n_values:
        for block_id, _ in enumerate(consecutive_blocks(maximum_n, n_value)):
            for probe in range(probes):
                accumulators[(n_value, block_id, probe)] = SufficientStatistics()
    evaluator = bundle.completion_backend._bound_evaluator()
    if str(getattr(getattr(evaluator, "info", None), "contract_version", "")) != "2" or not callable(
        getattr(evaluator, "complete_linear_moments", None)
    ):
        raise RuntimeError("Rao--Blackwell variance jobs require evaluator contract v2 linear moments")
    minimum_ess = math.inf
    for first in range(0, maximum_n, chunk_size):
        chunk_started = time.perf_counter()
        stop = min(maximum_n, first + chunk_size)
        trajectory_rows = torch.arange(first, stop, device=device)
        mask = (trajectory_id >= first) & (trajectory_id < stop)
        decision_rows = mask.nonzero(as_tuple=False).squeeze(1)
        local_trajectory = trajectory_id[decision_rows] - first
        counts = torch.bincount(local_trajectory, minlength=stop - first)
        maximum_functionals = int(counts.max().item()) + 1
        if maximum_functionals > functional_bound:
            raise MemoryError(f"trajectory requires {maximum_functionals} functionals, exceeding configured bound {functional_bound}")
        selected_leaf_reach = leaf_reach[first:stop]
        captured_leaf = {
            name: value[first:stop]
            for name, value in leaf_source.columns.items()
        }
        completion_inputs = {
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
        completion_inputs["range_weights"] = (
            bundle.rollout_backend.range_weights_from_log_reach(
                selected_leaf_reach
            )
        )
        seeds = torch.tensor(
            [logical_seed(int(experiment["logical_seed"]), "oracle-completion", int(checkpoint), int(superbatch), int(t)) for t in range(first, stop)],
            dtype=torch.int64,
            device=device,
        )
        for probe_first in range(0, probes, probes_per_chunk):
            probe_ids = tuple(range(probe_first, probe_first + probes_per_chunk))
            directions = [
                rademacher_direction(
                    model,
                    logical_seed(int(experiment["logical_seed"]), "parameter-probe", probe),
                )
                for probe in probe_ids
            ]
            jvp_started = time.perf_counter()
            value, derivatives = _model_directional_rows(
                bundle,
                decision_inputs,
                decision_rows,
                events[decision_rows],
                directions,
                0.0005,
                directional_batch_rows,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            jvp_seconds += time.perf_counter() - jvp_started
            memory_gate()
            for packed_probe, probe in enumerate(probe_ids):
                derivative = derivatives[packed_probe]
                p_coeff = torch.zeros(
                    (stop - first, maximum_functionals, bundle.rollout_backend.n_players, bundle.rollout_backend.n_hands),
                    dtype=torch.float32, device=device,
                )
                m_coeff = torch.zeros_like(p_coeff)
                functional_mask = torch.zeros((stop - first, maximum_functionals), dtype=torch.bool, device=device)
                offsets = torch.zeros(stop - first, dtype=torch.int64, device=device)
                for local_row, source_row in enumerate(decision_rows.tolist()):
                    trajectory = int(local_trajectory[local_row])
                    function = int(offsets[trajectory])
                    player = int(owner[source_row])
                    d = derivative[local_row]
                    p = value[local_row] * d
                    p_coeff[trajectory, function, player] = p
                    m_coeff[trajectory, function, player] = -d
                    functional_mask[trajectory, function] = True
                    offsets[trajectory] += 1
                for trajectory in range(stop - first):
                    total_function = int(counts[trajectory])
                    p_coeff[trajectory, total_function] = p_coeff[trajectory, :total_function].sum(0)
                    m_coeff[trajectory, total_function] = m_coeff[trajectory, :total_function].sum(0)
                    functional_mask[trajectory, total_function] = True
                moment_started = time.perf_counter()
                result = evaluator.complete_linear_moments(
                    **completion_inputs,
                    p_coefficient=p_coeff,
                    m_coefficient=m_coeff,
                    functional_mask=functional_mask,
                    n_samples=int(experiment["oracle"]["initial_samples"]),
                    seed=seeds,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                moment_seconds += time.perf_counter() - moment_started
                memory_gate()
                if not bool(result.functional_valid[functional_mask].all()):
                    raise RuntimeError("oracle returned an invalid functional")
                minimum_ess = min(minimum_ess, float(result.n_eff.min().detach().cpu()))
                for trajectory in range(stop - first):
                    global_trajectory = first + trajectory
                    count = int(counts[trajectory])
                    for n_value in n_values:
                        block_id = global_trajectory // n_value
                        stats = accumulators[(n_value, block_id, probe)]
                        stats.update_trajectory(
                            result.mean[trajectory, :count].detach().cpu(),
                            result.world_variance[trajectory, :count].detach().cpu(),
                            result.mean_variance[trajectory, :count].detach().cpu(),
                            float(result.mean[trajectory, count]),
                            float(result.world_variance[trajectory, count]),
                            float(result.mean_variance[trajectory, count]),
                        )
                del derivative, p_coeff, m_coeff, functional_mask, result
            del directions, value, derivatives
        completed = stop
        chunk_number = first // chunk_size + 1
        chunk_count = math.ceil(maximum_n / chunk_size)
        if chunk_number == 1 or chunk_number % progress_every == 0 or completed == maximum_n:
            elapsed = time.perf_counter() - started
            measured_chunk_work = max(0.0, elapsed - rollout_seconds)
            projected = rollout_seconds + measured_chunk_work * chunk_count / chunk_number
            print(
                f"[variance checkpoint={checkpoint} superbatch={superbatch} "
                f"device={device}] trajectories={completed}/{maximum_n} "
                f"chunks={chunk_number}/{chunk_count} elapsed={elapsed:.1f}s "
                f"job_eta={max(0.0, projected - elapsed):.1f}s "
                f"last_chunk={time.perf_counter() - chunk_started:.1f}s "
                f"jvp_total={jvp_seconds:.1f}s moments_total={moment_seconds:.1f}s",
                flush=True,
            )
        if completed < maximum_n and chunk_cooldown_seconds:
            time.sleep(chunk_cooldown_seconds)
    serial = {
        f"N{n_value}/block{block_id}/probe{probe}": stats.asdict()
        for (n_value, block_id, probe), stats in accumulators.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "checkpoint": int(checkpoint),
        "checkpoint_iteration": int(metadata["iteration"]),
        "checkpoint_sha256": identity,
        "superbatch": int(superbatch),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "precision": "fp32",
        "evaluator_identity": asdict(evaluator.info),
        "oracle_samples": int(experiment["oracle"]["initial_samples"]),
        "chunk_cooldown_seconds": chunk_cooldown_seconds,
        "duty_cycle": {
            "run_seconds": duty_run_seconds,
            "pause_seconds": duty_pause_seconds,
        },
        "minimum_effective_sample_size": minimum_ess,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "seconds": time.perf_counter() - started,
        "timings": {
            "rollout_seconds": rollout_seconds,
            "jvp_seconds": jvp_seconds,
            "moment_seconds": moment_seconds,
        },
        "statistics": serial,
    }


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a mapping")
    required = {"checkpoint_dir", "checkpoints", "logical_seed", "candidates", "probes", "public_sampling", "oracle", "streaming", "smoke", "timing", "constraints", "validation", "analysis", "numerics", "out_dir"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"experiment config is missing {missing}")
    candidates = raw["candidates"]
    if set(candidates) != {"N", "B", "n"} or any(not values for values in candidates.values()):
        raise ValueError("candidates must contain nonempty N, B, and n lists")
    if any(any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values) for values in candidates.values()):
        raise ValueError("candidate values must be positive integers")
    maximum = max(candidates["N"])
    if any(maximum % value for value in candidates["N"]):
        raise ValueError("maximum candidate N must be divisible by every candidate N")
    if raw["probes"].get("distribution") != "rademacher":
        raise ValueError("only Rademacher parameter probes are supported")
    return raw


def manifest_payload(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    scientific = json.loads(json.dumps(config))
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    try:
        import poker_gpu_evaluator as poker
        evaluator_identity = {
            "contract_version": poker.CONTRACT_VERSION,
            "package_version": poker.__version__,
            "build_revision": poker.BUILD_REVISION,
        }
    except (ImportError, AttributeError):
        evaluator_identity = None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": scientific,
        "overrides": dict(overrides),
        "pbg_git_revision": revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "evaluator": evaluator_identity,
        "precision": "fp32",
        "probe_seeds": [
            logical_seed(int(config["logical_seed"]), "parameter-probe", probe)
            for probe in range(int(config["probes"]["count"]))
        ],
    }
    # Worker count and local device selection are recorded execution details,
    # not scientific identity: a one-GPU smoke and three-GPU production run
    # must address the same jobs and share one resumable manifest.
    encoded = json.dumps(
        {"schema_version": SCHEMA_VERSION, "config": scientific},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def ensure_manifest(config: Mapping[str, Any], overrides: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    manifest = manifest_payload(config, overrides)
    path = Path(str(config["out_dir"])) / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            # Migrate manifests produced by the initial implementation, whose
            # fingerprint incorrectly included execution-only overrides.
            if existing.get("schema_version") == SCHEMA_VERSION and existing.get("config") == manifest["config"]:
                history = list(existing.get("invocations", []))
                history.append(dict(overrides))
                manifest["invocations"] = history
                atomic_json(path, manifest)
                return manifest
            raise RuntimeError("output directory contains an incompatible experiment manifest")
        if not resume:
            raise FileExistsError(f"manifest already exists: {path}; pass --resume")
        history = list(existing.get("invocations", []))
        history.append(dict(overrides))
        manifest["invocations"] = history
        atomic_json(path, manifest)
        return manifest
    atomic_json(path, manifest)
    return manifest


__all__ = [
    "SCHEMA_VERSION", "SufficientStatistics", "assigned_jobs", "atomic_json",
    "canonical_jobs", "checkpoint_sha256", "consecutive_blocks", "cpu_oracle",
    "ensure_manifest", "load_experiment_config", "logical_seed", "manifest_payload",
    "theoretical_variance", "variance_cube", "weighted_ratio_moments",
    "parameter_jvp", "rademacher_direction",
]
