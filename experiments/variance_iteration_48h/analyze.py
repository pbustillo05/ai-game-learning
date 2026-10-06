"""Analyze locked experiment artifacts and produce the required report/figures."""

# The report builder uses adjacent strings deliberately to keep Markdown source
# readable without embedding runtime newlines in every paragraph.
# ruff: noqa: ISC004

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from experiments.variance_iteration_48h.protocol import (
    gradient_variance_from_gram,
    logical_seed,
    paired_bootstrap_indices,
    percentile_interval,
)
from experiments.variance_iteration_48h.runner import (
    ExperimentError,
    atomic_json,
    atomic_torch_save,
    load_manifest,
)


def _counts(indices: Tensor, repetitions: int) -> Tensor:
    result = torch.zeros((indices.shape[0], repetitions), dtype=torch.float64)
    result.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))
    return result


def _variance_many(gram: Tensor, weights: Tensor) -> Tensor:
    matrix = gram.double()
    count = weights.sum(1)
    diagonal = weights @ matrix.diagonal()
    squared_mean = torch.einsum("bi,ij,bj->b", weights, matrix, weights) / count
    return (diagonal - squared_mean).clamp_min(0) / (count - 1)


def _build_gram(
    paths: Sequence[Path],
    *,
    device: str,
    group_size: int,
    chunk_elements: int = 131_072,
) -> tuple[Tensor, list[dict[str, object]], dict[str, float]]:
    vectors: list[Tensor] = []
    rows: list[dict[str, object]] = []
    width: int | None = None
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=False)
        gradient = value["gradient"].float().contiguous()
        if width is None:
            width = gradient.numel()
        if gradient.shape != (width,) or not bool(torch.isfinite(gradient).all()):
            raise ExperimentError(f"invalid gradient vector: {path}")
        vectors.append(gradient)
        rows.append(dict(value["metadata"]))
    if width is None:
        raise ExperimentError("gradient inventory is empty")
    matrix = torch.stack(vectors)
    del vectors
    gram = torch.zeros((matrix.shape[0], matrix.shape[0]), dtype=torch.float64)
    for begin in range(0, width, chunk_elements):
        end = min(begin + chunk_elements, width)
        block = matrix[:, begin:end].to(device=device, non_blocking=False)
        gram.add_((block @ block.T).double().cpu())
        del block

    def stable_group(begin: int, end: int) -> tuple[float, Tensor]:
        mean = torch.zeros(width, dtype=torch.float64)
        m2 = torch.zeros((), dtype=torch.float64)
        for count, row in enumerate(range(begin, end), start=1):
            current = matrix[row].double()
            delta = current - mean
            mean.add_(delta / count)
            m2.add_(torch.dot(delta, current - mean))
        return float(m2 / (end - begin - 1)), mean

    if matrix.shape[0] != 2 * group_size:
        raise ExperimentError("combined gradient inventory does not contain two arms")
    sampled_variance, sampled_mean = stable_group(0, group_size)
    rb_variance, rb_mean = stable_group(group_size, 2 * group_size)
    stable = {
        "sampled_variance": sampled_variance,
        "rb_variance": rb_variance,
        "mean_difference_l2": float(torch.linalg.vector_norm(rb_mean - sampled_mean)),
    }
    return gram, rows, stable


def analyze_gradients(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    repetitions = int(manifest["scope"]["gradient_repetitions"])
    draws = int(manifest["bootstrap_draws"])
    summary: dict[str, Any] = {}
    for checkpoint in manifest["checkpoints"]:
        paths = [
            run_dir / "gradient" / checkpoint / f"rep_{repetition:02d}" / f"{arm}.pt"
            for arm in manifest["arms"]
            for repetition in range(repetitions)
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise ExperimentError(
                f"{checkpoint} is missing {len(missing)} gradient artifacts"
            )
        gram, metadata, stable = _build_gram(
            paths, device=device, group_size=repetitions
        )
        sampled_gram = gram[:repetitions, :repetitions]
        rb_gram = gram[repetitions:, repetitions:]
        sampled_gram_variance = float(gradient_variance_from_gram(sampled_gram))
        rb_gram_variance = float(gradient_variance_from_gram(rb_gram))
        sampled_variance = stable["sampled_variance"]
        rb_variance = stable["rb_variance"]
        ratio = rb_variance / sampled_variance
        leave_one_out_ratios: list[float] = []
        for omitted in range(repetitions):
            retained = torch.tensor(
                [index for index in range(repetitions) if index != omitted],
                dtype=torch.int64,
            )
            sampled_retained = sampled_gram.index_select(0, retained).index_select(
                1, retained
            )
            rb_retained = rb_gram.index_select(0, retained).index_select(1, retained)
            leave_one_out_ratios.append(
                float(
                    gradient_variance_from_gram(rb_retained)
                    / gradient_variance_from_gram(sampled_retained)
                )
            )
        indices = paired_bootstrap_indices(
            repetitions,
            draws,
            logical_seed(int(manifest["base_seed"]), "gradient-bootstrap", checkpoint),
        )
        weights = _counts(indices, repetitions)
        sampled_boot = _variance_many(sampled_gram, weights)
        rb_boot = _variance_many(rb_gram, weights)
        sampled_boot.mul_(sampled_variance / sampled_gram_variance)
        rb_boot.mul_(rb_variance / rb_gram_variance)
        ratio_boot = rb_boot / sampled_boot.clamp_min(torch.finfo(torch.float64).tiny)
        ratio_interval = percentile_interval(ratio_boot.tolist())
        cross = gram[:repetitions, repetitions:]
        difference_gram = rb_gram + sampled_gram - cross - cross.T
        mean_difference_l2 = stable["mean_difference_l2"]
        difference_boot = (
            torch.sqrt(
                torch.einsum(
                    "bi,ij,bj->b", weights, difference_gram, weights
                ).clamp_min(0)
            )
            / repetitions
        )
        difference_interval = percentile_interval(difference_boot.tolist())
        artifact = run_dir / "analysis" / f"gradient_gram_{checkpoint}.pt"
        atomic_torch_save(
            artifact,
            {
                "row_order": [
                    {
                        "arm": arm,
                        "repetition": repetition,
                    }
                    for arm in manifest["arms"]
                    for repetition in range(repetitions)
                ],
                "gram": gram,
            },
        )
        summary[checkpoint] = {
            "repetitions": repetitions,
            "sampled_total_variance": sampled_variance,
            "rb1024_total_variance": rb_variance,
            "rb_over_sampled_variance_ratio": ratio,
            "ratio_ci95": list(ratio_interval),
            "paired_leave_one_out_ratio_range": [
                min(leave_one_out_ratios),
                max(leave_one_out_ratios),
            ],
            "minimum_ratio_omits_repetition": int(
                min(
                    range(repetitions),
                    key=leave_one_out_ratios.__getitem__,
                )
            ),
            "mean_gradient_difference_l2": mean_difference_l2,
            "mean_gradient_difference_l2_ci95": list(difference_interval),
            "gram_relative_error": {
                "sampled": abs(sampled_gram_variance - sampled_variance)
                / sampled_variance,
                "rb1024": abs(rb_gram_variance - rb_variance) / rb_variance,
            },
            "sampled_mean_gradient_l2": statistics.mean(
                float(row["gradient_l2"]) for row in metadata[:repetitions]
            ),
            "rb1024_mean_gradient_l2": statistics.mean(
                float(row["gradient_l2"]) for row in metadata[repetitions:]
            ),
            "sampled_mean_seconds": statistics.mean(
                float(row["seconds_including_targets"])
                for row in metadata[:repetitions]
            ),
            "rb1024_mean_seconds": statistics.mean(
                float(row["seconds_including_targets"])
                for row in metadata[repetitions:]
            ),
            "maximum_peak_allocated_bytes": max(
                int(row["peak_allocated_bytes"]) for row in metadata
            ),
            "gram_artifact": str(artifact),
        }
    atomic_json(run_dir / "analysis" / "gradient_summary.json", summary)
    return summary


def _bootstrap_mean_interval(
    values: Sequence[float], draws: int, seed: int
) -> tuple[float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    indices = paired_bootstrap_indices(len(values), draws, seed)
    means = tensor[indices].mean(1)
    return percentile_interval(means.tolist())


def _reference_precision(run_dir: Path, checkpoint: str) -> dict[str, Any]:
    paths = sorted((run_dir / "reference_precision").glob(f"{checkpoint}_m*.json"))
    if len(paths) != 2:
        return {"status": "missing", "files": [str(path) for path in paths]}
    first, second = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    if first["public_contract_sha256"] != second["public_contract_sha256"]:
        raise ExperimentError(
            f"{checkpoint} precision streams do not share one public pool"
        )
    common = sorted(set(first["objectives"]).intersection(second["objectives"]))
    differences = [
        float(second["objectives"][key]) - float(first["objectives"][key])
        for key in common
    ]
    paired_gap_differences: dict[str, float] = {}
    for step in (38, 100):
        sampled_key = f"{checkpoint}/rep00/sampled/step{step}"
        rb_key = f"{checkpoint}/rep00/rb1024/step{step}"
        if all(key in first["objectives"] for key in (sampled_key, rb_key)) and all(
            key in second["objectives"] for key in (sampled_key, rb_key)
        ):
            first_gap = float(first["objectives"][sampled_key]) - float(
                first["objectives"][rb_key]
            )
            second_gap = float(second["objectives"][sampled_key]) - float(
                second["objectives"][rb_key]
            )
            paired_gap_differences[str(step)] = second_gap - first_gap
    return {
        "status": "complete",
        "budgets": [first["completion_samples"], second["completion_samples"]],
        "public_rollouts": first["public_rollouts"],
        "independent_completion_streams": (
            first["completion_seed_mask"] != second["completion_seed_mask"]
        ),
        "same_public_pool": True,
        "candidate_count": len(common),
        "mean_loss_difference": statistics.mean(differences),
        "maximum_absolute_loss_difference": max(map(abs, differences)),
        "rms_loss_difference": math.sqrt(
            statistics.mean(value * value for value in differences)
        ),
        "paired_improvement_gap_difference": paired_gap_differences,
        "maximum_absolute_paired_gap_difference": max(
            map(abs, paired_gap_differences.values())
        ),
        "signed_cache_max_abs_error": max(
            first["signed_cache_max_abs_error"],
            second["signed_cache_max_abs_error"],
        ),
    }


def analyze_iterations(
    run_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    repetitions = int(manifest["scope"]["optimization_repetitions"])
    endpoints = tuple(int(value) for value in manifest["scope"]["endpoints"])
    draws = int(manifest["bootstrap_draws"])
    all_rows: list[dict[str, object]] = []
    summary: dict[str, Any] = {}
    for checkpoint in manifest["checkpoints"]:
        reference_path = run_dir / "reference" / f"{checkpoint}.json"
        if not reference_path.exists():
            raise ExperimentError(f"missing held-out reference: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        theta0 = float(reference["theta0_loss"])
        for repetition in range(repetitions):
            base = run_dir / "optimization" / checkpoint / f"rep_{repetition:02d}"
            for arm in manifest["arms"]:
                metadata = json.loads(
                    (base / f"{arm}.json").read_text(encoding="utf-8")
                )
                for endpoint in endpoints:
                    candidate = f"{checkpoint}/rep{repetition:02d}/{arm}/step{endpoint}"
                    loss = float(reference["objectives"][candidate])
                    all_rows.append(
                        {
                            "checkpoint": checkpoint,
                            "repetition": repetition,
                            "arm": arm,
                            "step": endpoint,
                            "loss": loss,
                            "improvement": theta0 - loss,
                            "elapsed_seconds": float(
                                metadata["endpoint_seconds_including_targets"][
                                    str(endpoint)
                                ]
                            ),
                        }
                    )
        checkpoint_summary: dict[str, Any] = {
            "repetitions": repetitions,
            "heldout_public_rollouts": reference["public_rollouts"],
            "heldout_decision_rows": reference["decision_rows"],
            "heldout_completion_samples": reference["completion_samples"],
            "theta0_loss": theta0,
            "signed_cache_max_abs_error": reference["signed_cache_max_abs_error"],
            "reference_precision": _reference_precision(run_dir, checkpoint),
            "steps": {},
        }
        for endpoint in endpoints:
            sampled = [
                float(row["improvement"])
                for row in all_rows
                if row["checkpoint"] == checkpoint
                and row["arm"] == "sampled"
                and row["step"] == endpoint
            ]
            rb = [
                float(row["improvement"])
                for row in all_rows
                if row["checkpoint"] == checkpoint
                and row["arm"] == "rb1024"
                and row["step"] == endpoint
            ]
            paired = [right - left for left, right in zip(sampled, rb, strict=True)]
            seed_base = int(manifest["base_seed"])
            checkpoint_summary["steps"][str(endpoint)] = {
                "sampled_mean_improvement": statistics.mean(sampled),
                "sampled_ci95": list(
                    _bootstrap_mean_interval(
                        sampled,
                        draws,
                        logical_seed(
                            seed_base,
                            "iteration-bootstrap-sampled",
                            checkpoint,
                            endpoint,
                        ),
                    )
                ),
                "rb1024_mean_improvement": statistics.mean(rb),
                "rb1024_ci95": list(
                    _bootstrap_mean_interval(
                        rb,
                        draws,
                        logical_seed(
                            seed_base,
                            "iteration-bootstrap-rb",
                            checkpoint,
                            endpoint,
                        ),
                    )
                ),
                "paired_rb_minus_sampled": statistics.mean(paired),
                "paired_ci95": list(
                    _bootstrap_mean_interval(
                        paired,
                        draws,
                        logical_seed(
                            seed_base,
                            "iteration-bootstrap-paired",
                            checkpoint,
                            endpoint,
                        ),
                    )
                ),
                "sampled_mean_seconds": statistics.mean(
                    float(row["elapsed_seconds"])
                    for row in all_rows
                    if row["checkpoint"] == checkpoint
                    and row["arm"] == "sampled"
                    and row["step"] == endpoint
                ),
                "rb1024_mean_seconds": statistics.mean(
                    float(row["elapsed_seconds"])
                    for row in all_rows
                    if row["checkpoint"] == checkpoint
                    and row["arm"] == "rb1024"
                    and row["step"] == endpoint
                ),
            }
        summary[checkpoint] = checkpoint_summary
    csv_path = run_dir / "analysis" / "iteration_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    atomic_json(run_dir / "analysis" / "iteration_summary.json", summary)
    return summary, all_rows


def _save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_results(
    run_dir: Path,
    gradient: Mapping[str, Any],
    iteration: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
) -> None:
    labels = list(gradient)
    ratios = [gradient[label]["rb_over_sampled_variance_ratio"] for label in labels]
    intervals = [gradient[label]["ratio_ci95"] for label in labels]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(range(len(labels)), ratios, color="#4C78A8")
    axis.errorbar(
        range(len(labels)),
        ratios,
        yerr=[
            [value - interval[0] for value, interval in zip(ratios, intervals)],
            [interval[1] - value for value, interval in zip(ratios, intervals)],
        ],
        fmt="none",
        color="black",
        capsize=4,
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("RB-1024 / Sampled actor-gradient variance")
    axis.set_yscale("log")
    axis.set_title("Full-gradient variance ratio (paired bootstrap 95% CI)")
    figure.tight_layout()
    _save_figure(figure, run_dir / "figures" / "actor_gradient_variance_ratio")

    colors = {"sampled": "#E45756", "rb1024": "#4C78A8"}
    for horizontal, filename, xlabel in (
        ("step", "heldout_improvement_vs_steps", "Optimizer steps"),
        (
            "elapsed_seconds",
            "heldout_improvement_vs_compute_time",
            "Elapsed GPU seconds (including target construction)",
        ),
    ):
        figure, axes = plt.subplots(1, len(iteration), figsize=(12, 4.3), sharey=True)
        if len(iteration) == 1:
            axes = [axes]
        for axis, checkpoint in zip(axes, iteration, strict=True):
            for arm in ("sampled", "rb1024"):
                selected = [
                    row
                    for row in rows
                    if row["checkpoint"] == checkpoint and row["arm"] == arm
                ]
                endpoints = sorted({int(row["step"]) for row in selected})
                x = []
                y = []
                low = []
                high = []
                for endpoint in endpoints:
                    endpoint_rows = [
                        row for row in selected if int(row["step"]) == endpoint
                    ]
                    x.append(
                        float(endpoint)
                        if horizontal == "step"
                        else statistics.mean(
                            float(row[horizontal]) for row in endpoint_rows
                        )
                    )
                    values = [float(row["improvement"]) for row in endpoint_rows]
                    mean = statistics.mean(values)
                    interval = iteration[checkpoint]["steps"][str(endpoint)][
                        f"{arm}_ci95"
                    ]
                    y.append(mean)
                    low.append(interval[0])
                    high.append(interval[1])
                axis.plot(x, y, marker="o", color=colors[arm], label=arm)
                axis.fill_between(x, low, high, color=colors[arm], alpha=0.18)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(checkpoint)
            axis.set_xlabel(xlabel)
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Held-out actor improvement (larger is better)")
        axes[-1].legend()
        figure.tight_layout()
        _save_figure(figure, run_dir / "figures" / filename)


def _conclusion(estimate: float, interval: Sequence[float]) -> str:
    if interval[0] > 0:
        return "favored RB-1024"
    if interval[1] < 0:
        return "favored Sampled"
    return "was unresolved"


def audit_queue(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    gradient_pairs = sorted((run_dir / "gradient").glob("*/rep_*/pair.json"))
    optimization_pairs = sorted((run_dir / "optimization").glob("*/rep_*/pair.json"))
    expected_gradient = len(manifest["checkpoints"]) * int(
        manifest["scope"]["gradient_repetitions"]
    )
    expected_optimization = len(manifest["checkpoints"]) * int(
        manifest["scope"]["optimization_repetitions"]
    )
    gpu_seconds = 0.0
    peak_bytes = 0
    skipped_updates = 0
    nonfinite_gradient_jobs = 0
    for path in (*gradient_pairs, *optimization_pairs):
        pair = json.loads(path.read_text(encoding="utf-8"))
        for arm in pair["arms"].values():
            gpu_seconds += float(arm["seconds_including_targets"])
            peak_bytes = max(peak_bytes, int(arm["peak_allocated_bytes"]))
            gradient = arm["metrics"].get("gradient", {})
            nonfinite_gradient_jobs += int(not gradient.get("finite", True))
            if pair["experiment"] == "one_iteration_optimization_pair":
                skipped_updates += int(arm["maximum_steps"]) - int(arm["applied_steps"])
    reference_paths = sorted((run_dir / "reference").glob("*.json"))
    precision_paths = sorted((run_dir / "reference_precision").glob("*.json"))
    for path in (*reference_paths, *precision_paths):
        row = json.loads(path.read_text(encoding="utf-8"))
        gpu_seconds += float(row["seconds"])
        peak_bytes = max(peak_bytes, int(row["peak_allocated_bytes"]))
    failed_logs = []
    for path in sorted((run_dir / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in text:
            failed_logs.append(str(path))
    pilot_failure_path = run_dir / "checks" / "reference_pilot_failure.json"
    retained_pilot_failures: list[str] = []
    if pilot_failure_path.exists():
        pilot_failure = json.loads(pilot_failure_path.read_text(encoding="utf-8"))
        retained_pilot_failures = list(pilot_failure.get("failures", ()))
    status = {
        "gradient_pairs": {
            "launched": expected_gradient,
            "completed": len(gradient_pairs),
            "failed": max(0, expected_gradient - len(gradient_pairs)),
        },
        "optimization_pairs": {
            "launched": expected_optimization,
            "completed": len(optimization_pairs),
            "failed": max(0, expected_optimization - len(optimization_pairs)),
        },
        "reference_jobs_completed": len(reference_paths),
        "precision_jobs_completed": len(precision_paths),
        "failed_logs": failed_logs,
        "retained_resolved_pilot_failures": retained_pilot_failures,
        "skipped_optimizer_updates": skipped_updates,
        "nonfinite_gradient_jobs": nonfinite_gradient_jobs,
        "measured_gpu_seconds": gpu_seconds,
        "measured_gpu_hours": gpu_seconds / 3600.0,
        "maximum_peak_allocated_bytes": peak_bytes,
        "analyzed": True,
    }
    if (
        len(gradient_pairs) != expected_gradient
        or len(optimization_pairs) != expected_optimization
        or failed_logs
    ):
        raise ExperimentError("queue audit found missing or failed paired jobs")
    atomic_json(run_dir / "analysis" / "queue_status.json", status)
    return status


def write_results(
    run_dir: Path,
    manifest: Mapping[str, Any],
    gradient: Mapping[str, Any],
    iteration: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    primary = int(manifest["optimization"]["primary_endpoint"])
    secondary = max(int(value) for value in manifest["scope"]["endpoints"])
    lines = [
        "# Gradient variance and one-iteration optimization",
        "",
        "## Result",
        "",
    ]
    for checkpoint in manifest["checkpoints"]:
        grad = gradient[checkpoint]
        primary_row = iteration[checkpoint]["steps"][str(primary)]
        secondary_row = iteration[checkpoint]["steps"][str(secondary)]
        lines.extend(
            [
                f"- **{checkpoint}:** RB/Sampled full actor-gradient variance ratio "
                f"{grad['rb_over_sampled_variance_ratio']:.4g} "
                f"(paired-bootstrap 95% CI {grad['ratio_ci95'][0]:.4g} to "
                f"{grad['ratio_ci95'][1]:.4g}, $R={grad['repetitions']}$).",
                f"- **{checkpoint}, step {primary}:** mean paired held-out improvement "
                f"difference (RB minus Sampled) {primary_row['paired_rb_minus_sampled']:.5g} "
                f"(95% CI {primary_row['paired_ci95'][0]:.5g} to "
                f"{primary_row['paired_ci95'][1]:.5g}, six rollout-batch pairs); "
                f"the comparison {_conclusion(primary_row['paired_rb_minus_sampled'], primary_row['paired_ci95'])}.",
                f"- **{checkpoint}, step {secondary}:** paired difference "
                f"{secondary_row['paired_rb_minus_sampled']:.5g} "
                f"(95% CI {secondary_row['paired_ci95'][0]:.5g} to "
                f"{secondary_row['paired_ci95'][1]:.5g}); "
                f"the comparison {_conclusion(secondary_row['paired_rb_minus_sampled'], secondary_row['paired_ci95'])}.",
            ]
        )
        if grad["ratio_ci95"][0] <= 1.0 <= grad["ratio_ci95"][1]:
            influence = grad["paired_leave_one_out_ratio_range"]
            lines.append(
                f"- **{checkpoint} variance stability (post hoc):** paired leave-one-out "
                f"ratios ranged from {influence[0]:.4g} to {influence[1]:.4g}; "
                f"the minimum omitted repetition "
                f"{grad['minimum_ratio_omits_repetition']}. Treat the variance result "
                "as heavy-tailed and unresolved."
            )
    lines.extend(
        [
            "",
            "Step 38 was predeclared as primary. Intervals bootstrap whole paired "
            "repetition IDs and are conditional on each checkpoint's common held-out pool.",
            "",
            "## Protocol and actual sample sizes",
            "",
            f"The timing-only scope lock retained the default protocol: "
            f"{manifest['scope']['gradient_repetitions']} gradient pairs per checkpoint, "
            f"{manifest['scope']['optimization_repetitions']} optimization pairs per "
            f"checkpoint, and endpoints {manifest['scope']['endpoints']}. The measured "
            f"pre-lock queue forecast was {manifest['scope']['forecast_wall_seconds_excluding_reference_and_analysis'] / 60:.1f} minutes on three RTX 2080 Ti GPUs.",
            "",
            "Every gradient arm used 384 public rollouts, one independently sampled "
            "256-decision logical batch, $\\lambda=1$, stack-normalized advantages, "
            "an effectively unclipped actor-only loss, and no value/entropy/reference "
            "terms. Gradients were captured in FP32 after explicit loss unscaling and "
            "before clipping or Adam. Original FP32 vectors and exact combined Gram "
            "matrices are retained.",
            "",
            "Every optimization arm restored identical within-pair checkpoint weights, "
            "Adam moments and counters, scaler, reference model/state, public rollout, "
            "and minibatch-index schedule. Each arm constructed its own production "
            "targets once and reused them for all optimizer steps. Checkpoint-specific "
            "original hyperparameters were preserved within each pair.",
            "",
            "## Common held-out objective and precision",
            "",
        ]
    )
    for checkpoint in manifest["checkpoints"]:
        item = iteration[checkpoint]
        precision = item["reference_precision"]
        lines.append(
            f"- {checkpoint}: {item['heldout_public_rollouts']} public trajectories, "
            f"{item['heldout_decision_rows']} decision rows, and "
            f"{item['heldout_completion_samples']} completions; signed-moment/direct "
            f"maximum error {item['signed_cache_max_abs_error']:.3g}. Precision check: "
            + (
                f"4096 versus 8192 independent completion streams on "
                f"{precision['public_rollouts']} common public trajectories had RMS loss "
                f"difference {precision['rms_loss_difference']:.3g} and maximum absolute "
                f"difference {precision['maximum_absolute_loss_difference']:.3g}; the "
                f"largest change in an RB-minus-Sampled paired gap was "
                f"{precision['maximum_absolute_paired_gap_difference']:.3g}."
                if precision["status"] == "complete"
                else "not completed."
            )
        )
    lines.extend(
        [
            "",
            "The objective is the posterior-weighted ordinary clipped PPO actor loss "
            "with the checkpoint old policy, baseline, and advantage normalization "
            "fixed across all candidates. It conditions on actor hand before applying "
            "the action ratio. The cached positive/negative-advantage identity was "
            "checked against direct averaging over the same completion worlds, "
            "including both signs.",
            "The production RB completion used only to seal aligned learner rows "
            "was pinned to the validated 1,024-sample setting after the retained "
            "pilot failure; those scaffold action targets were not used by this "
            "objective. All reported reference moments use the independent direct "
            "4,096/8,192-world calculations above.",
            "",
            "## Complete optimization curve",
            "",
            "| Checkpoint | Step | Sampled improvement (95% CI) | RB-1024 improvement (95% CI) | Paired RB-Sampled (95% CI) | Sampled s | RB s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in manifest["checkpoints"]:
        for step, row in iteration[checkpoint]["steps"].items():
            lines.append(
                f"| {checkpoint} | {step} | {row['sampled_mean_improvement']:.5g} "
                f"[{row['sampled_ci95'][0]:.5g}, {row['sampled_ci95'][1]:.5g}] | "
                f"{row['rb1024_mean_improvement']:.5g} "
                f"[{row['rb1024_ci95'][0]:.5g}, {row['rb1024_ci95'][1]:.5g}] | "
                f"{row['paired_rb_minus_sampled']:.5g} "
                f"[{row['paired_ci95'][0]:.5g}, {row['paired_ci95'][1]:.5g}] | "
                f"{row['sampled_mean_seconds']:.2f} | {row['rb1024_mean_seconds']:.2f} |"
            )
    maximum_memory = int(audit["maximum_peak_allocated_bytes"])
    lines.extend(
        [
            "",
            "## Compute, failures, and diagnostics",
            "",
            f"Maximum observed allocation was {maximum_memory / 2**30:.2f} GiB and "
            f"measured aggregate accelerator use was {audit['measured_gpu_hours']:.3f} GPU-hours. "
            "Target construction is included in every elapsed-time endpoint. No "
            "finished paired block was discarded; skipped updates and nonfinite "
            "gradients were zero in the analyzed queue. "
            f"{len(audit['retained_resolved_pilot_failures'])} failed reference-pilot "
            "attempts were retained in `checks/reference_pilot_failure.json` and "
            "resolved before the measured reference queue.",
            "",
            "Training losses are diagnostics only; every comparison above uses the "
            "common held-out objective. With six optimization pairs, its intervals are "
            "preliminary and conditional on one shared held-out pool per checkpoint.",
            "",
            "## Artifacts",
            "",
            "- `analysis/gradient_summary.json` and `analysis/gradient_gram_*.pt`: raw-gradient summaries and recomputable sufficient statistics.",
            "- `analysis/iteration_rows.csv` and `analysis/iteration_summary.json`: candidate losses, improvements, timings, and paired intervals.",
            "- `analysis/queue_status.json`: launched/completed/failed job counts, compute, memory, and skipped-update audit.",
            "- `gradient/`, `optimization/`, `reference/`, and `reference_precision/`: raw atomic job artifacts.",
            "- `configs/`: every resolved source configuration; `manifest.json`: scope, seeds, hashes, and identities.",
            "- `figures/actor_gradient_variance_ratio.{png,svg}`, `figures/heldout_improvement_vs_steps.{png,svg}`, and `figures/heldout_improvement_vs_compute_time.{png,svg}`.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "source /home/pablob/.venvs/poker/bin/activate",
            "export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13",
            'export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"',
            'export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"',
            f"python -m experiments.variance_iteration_48h.runner gradient --run-dir {run_dir} --checkpoint normal10k --repetition 0 --device cuda:0",
            f"python -m experiments.variance_iteration_48h.runner optimize --run-dir {run_dir} --checkpoint normal10k --repetition 0 --device cuda:0",
            f"python -m experiments.variance_iteration_48h.runner reference --run-dir {run_dir} --checkpoint normal10k --device cuda:0",
            f"python -m experiments.variance_iteration_48h.analyze --run-dir {run_dir} --device cuda:0",
            "```",
            "",
        ]
    )
    (run_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_dir: str | Path, *, device: str) -> None:
    directory = Path(run_dir)
    manifest = load_manifest(directory)
    gradient = analyze_gradients(directory, manifest, device=device)
    iteration, rows = analyze_iterations(directory, manifest)
    audit = audit_queue(directory, manifest)
    plot_results(directory, gradient, iteration, rows)
    write_results(directory, manifest, gradient, iteration, audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    analyze(args.run_dir, device=args.device)


if __name__ == "__main__":
    main()
