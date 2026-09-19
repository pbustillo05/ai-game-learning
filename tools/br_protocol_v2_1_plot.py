#!/usr/bin/env python3
"""Render protocol-v2.1 reports without retraining or altering selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROTOCOL = "fixed_target_paired_response_gain_v2_1"
SUFFIX = "v2_1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merged_targets(paths: list[Path], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        document = load(path)
        if document["protocol_version"] != PROTOCOL:
            raise RuntimeError(f"{path}: protocol mismatch")
        for target in document["targets"]:
            target_id = target["target_id"]
            if target_id in result:
                raise RuntimeError(f"duplicate {key} target: {target_id}")
            result[target_id] = target
    return result


def checked_test_report(
    path: Path,
    *,
    split: dict[str, Any],
    target: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    report = load(path)
    expected = {
        "protocol_version": PROTOCOL,
        "split_namespace": split["namespace"],
        "evaluation_seed": split["seed"],
        "trajectory_count": split["trajectories_for_frozen_selection"],
        "target_sha256": target["target_sha256"],
        "response_sha256": selected["response_checkpoint_sha256"],
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise RuntimeError(f"{path}: {name} differs from frozen selection")
    if len(report.get("observations", ())) != report["trajectory_count"]:
        raise RuntimeError(f"{path}: raw observation count mismatch")
    return report


def reserve(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--legacy-report", action="append", default=[])
    parser.add_argument("--promotion", required=True, type=Path, action="append")
    parser.add_argument("--selection", required=True, type=Path, action="append")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="corrected")
    args = parser.parse_args()

    plan = load(args.plan)
    if plan["protocol_version"] != PROTOCOL:
        raise RuntimeError("plan protocol mismatch")
    planned = {target["id"]: target for target in plan["targets"]}
    promotions = merged_targets(args.promotion, "promotion")
    selections = merged_targets(args.selection, "selection")
    if set(promotions) != set(selections):
        raise RuntimeError("promotion and selection target sets differ")
    unknown = set(selections) - set(planned)
    if unknown:
        raise RuntimeError(f"unplanned selected targets: {sorted(unknown)}")
    missing = plan.get("unavailable_original_targets", [])

    points = []
    searches: dict[str, Any] = {}
    test_split = plan["splits"]["test"]
    for target_id, selection in selections.items():
        target = planned[target_id]
        selected = selection["selected"]
        if selected["selection_kind"] == "zero_gain_control":
            report = {
                "mean_gain_bb_per_hand": 0.0,
                "standard_error_bb_per_hand": 0.0,
                "confidence_interval_bb_per_hand": [0.0, 0.0],
                "per_seat_gain_bb_per_hand": [0.0, 0.0],
                "trajectory_count": 0,
                "gpu_hours": 0.0,
                "target_sha256": target["target_sha256"],
                "response_sha256": target["target_sha256"],
            }
            report_path = None
        else:
            iteration = selected["response_iteration"]
            report_path = args.results / (
                f"test_{target_id}_ckpt{iteration:04d}_"
                f"{test_split['trajectories_for_frozen_selection']}traj_{SUFFIX}.json"
            )
            report = checked_test_report(
                report_path,
                split=test_split,
                target=target,
                selected=selected,
            )
        points.append(
            {
                "target_id": target_id,
                "method": target["method"],
                "target_iteration": target["target_iteration"],
                "target_sha256": target["target_sha256"],
                "response_training_semantics": "adaptive_mixed_profile_range_v1",
                "selected_response_iteration": selected["response_iteration"],
                "selected_response_checkpoint_sha256": selected[
                    "response_checkpoint_sha256"
                ],
                "seat_0_gain_bb_per_hand": report["per_seat_gain_bb_per_hand"][0],
                "seat_1_gain_bb_per_hand": report["per_seat_gain_bb_per_hand"][1],
                "mean_gain_bb_per_hand": report["mean_gain_bb_per_hand"],
                "standard_error_bb_per_hand": report[
                    "standard_error_bb_per_hand"
                ],
                "confidence_interval_low_bb_per_hand": report[
                    "confidence_interval_bb_per_hand"
                ][0],
                "confidence_interval_high_bb_per_hand": report[
                    "confidence_interval_bb_per_hand"
                ][1],
                "evaluation_trajectory_count": report["trajectory_count"],
                "gpu_hours": report["gpu_hours"],
                "test_report": None if report_path is None else str(report_path),
            }
        )
        screening = promotions[target_id]["all_screening_candidates"]
        validation = [
            candidate
            for candidate in selection["validation_candidates"]
            if candidate["selection_kind"] == "trained_response"
        ]
        point = points[-1]
        point["screening_trajectory_count"] = len(screening) * plan["splits"][
            "screening"
        ]["trajectories_per_trained_candidate"]
        point["validation_trajectory_count"] = len(validation) * plan["splits"][
            "validation"
        ]["trajectories_per_trained_candidate"]
        point["test_trajectory_count"] = report["trajectory_count"]
        point["total_evaluation_trajectory_count"] = (
            point["screening_trajectory_count"]
            + point["validation_trajectory_count"]
            + point["test_trajectory_count"]
        )
        point["screening_gpu_hours"] = sum(
            row["elapsed_seconds"] for row in screening
        ) / 3600.0
        point["validation_gpu_hours"] = sum(
            row["elapsed_seconds"] for row in validation
        ) / 3600.0
        point["test_gpu_hours"] = report["gpu_hours"]
        point["total_evaluation_gpu_hours"] = (
            point["screening_gpu_hours"]
            + point["validation_gpu_hours"]
            + point["test_gpu_hours"]
        )
        running = 0.0
        best = []
        for candidate in sorted(validation, key=lambda row: row["response_iteration"]):
            running = max(
                running, candidate["validation_mean_gain_bb_per_hand"]
            )
            best.append([candidate["response_iteration"], running])
        searches[target_id] = {
            "screening": screening,
            "validation": validation,
            "validation_best_so_far_including_zero_control": best,
        }

    legacy_points = []
    legacy_specs = args.legacy_report or [
        f"{target_id}={path}"
        for target_id, target in planned.items()
        for path in target.get("legacy_test_reports", [])
    ]
    for specification in legacy_specs:
        if "=" not in specification:
            raise ValueError("legacy report must be TARGET_ID=PATH")
        target_id, raw_path = specification.split("=", 1)
        if target_id not in planned:
            raise ValueError(f"unknown legacy target: {target_id}")
        path = Path(raw_path)
        report = load(path)
        target = planned[target_id]
        if report.get("target_sha256") != target["target_sha256"]:
            raise RuntimeError(f"{path}: legacy target hash mismatch")
        legacy_points.append(
            {
                "target_id": target_id,
                "method": target["method"],
                "target_iteration": target["target_iteration"],
                "mean_gain_bb_per_hand": report["mean_gain_bb_per_hand"],
                "standard_error_bb_per_hand": report["standard_error_bb_per_hand"],
                "trajectory_count": report["trajectory_count"],
                "semantics": report["semantics"],
                "legacy_selection_kind": (
                    "final_response" if path.name.startswith("br_eval_")
                    else "validation_selected_response"
                ),
                "report": str(path),
            }
        )

    points.sort(key=lambda row: (row["method"], row["target_iteration"]))
    legacy_points.sort(key=lambda row: (row["method"], row["target_iteration"]))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    bases = {
        "data": output / f"{args.prefix}_plot_data_{SUFFIX}.json",
        "csv": output / f"{args.prefix}_comparison_table_{SUFFIX}.csv",
        "md": output / f"{args.prefix}_comparison_table_{SUFFIX}.md",
        "main_png": output / f"{args.prefix}_main_{SUFFIX}.png",
        "main_svg": output / f"{args.prefix}_main_{SUFFIX}.svg",
        "main_pdf": output / f"{args.prefix}_main_{SUFFIX}.pdf",
        "search_png": output / f"{args.prefix}_search_{SUFFIX}.png",
        "search_svg": output / f"{args.prefix}_search_{SUFFIX}.svg",
        "search_pdf": output / f"{args.prefix}_search_{SUFFIX}.pdf",
    }
    reserve(list(bases.values()))
    bases["data"].write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL,
                "plan_status": plan["status"],
                "measurement": "paired achieved response gain",
                "unit": "bb/hand",
                "response_training_semantics": "adaptive_mixed_profile_range_v1",
                "points": points,
                "superseded_legacy_points": legacy_points,
                "unavailable_original_targets": missing,
                "search_diagnostics": searches,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fields = list(points[0])
    with bases["csv"].open("x", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)
    header = (
        "| target | method | response seed | selected iteration | seat 0 | seat 1 | mean | SE | 95% CI | trajectories S/V/T | total GPU-hours |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    response_seed = plan.get("candidate_grid", {}).get("response_training_seed", "legacy")
    for point in points:
        rows.append(
            f"| {point['target_iteration']} | {point['method']} | {response_seed} | "
            f"{point['selected_response_iteration']} | {point['seat_0_gain_bb_per_hand']:.6f} | "
            f"{point['seat_1_gain_bb_per_hand']:.6f} | {point['mean_gain_bb_per_hand']:.6f} | "
            f"{point['standard_error_bb_per_hand']:.6f} | "
            f"[{point['confidence_interval_low_bb_per_hand']:.6f}, "
            f"{point['confidence_interval_high_bb_per_hand']:.6f}] | "
            f"{point['screening_trajectory_count']}/"
            f"{point['validation_trajectory_count']}/"
            f"{point['test_trajectory_count']} | "
            f"{point['total_evaluation_gpu_hours']:.4f} |"
        )
    bases["md"].write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

    if legacy_points:
        fig, (ax, legacy_ax) = plt.subplots(
            2, 1, figsize=(8.0, 7.8), sharex=True, constrained_layout=True
        )
    else:
        fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
        legacy_ax = None
    styles = {"Normal": ("#1f77b4", "o"), "RB": ("#d62728", "s")}
    for method in sorted({point["method"] for point in points}):
        values = [point for point in points if point["method"] == method]
        color, marker = styles.get(method, ("#333333", "o"))
        ax.errorbar(
            [point["target_iteration"] for point in values],
            [point["mean_gain_bb_per_hand"] for point in values],
            yerr=[
                [
                    point["mean_gain_bb_per_hand"]
                    - point["confidence_interval_low_bb_per_hand"]
                    for point in values
                ],
                [
                    point["confidence_interval_high_bb_per_hand"]
                    - point["mean_gain_bb_per_hand"]
                    for point in values
                ],
            ],
            linestyle="none",
            marker=marker,
            color=color,
            capsize=3,
            label=method,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    for gap in missing:
        ax.axvline(gap["target_iteration"], color="#777777", linestyle=":", alpha=0.35)
        ax.text(
            gap["target_iteration"], 0.02, "missing grid",
            transform=ax.get_xaxis_transform(), rotation=90,
            ha="center", va="bottom", color="#666666", fontsize=6,
        )
    if legacy_ax is None:
        ax.set_xlabel("Target-training iteration")
    ax.set_ylabel("Mean achieved response gain (bb/hand)")
    ax.set_title("Corrected fixed-target paired evaluation (v2.1)")
    ax.legend()
    if legacy_ax is not None:
        groups = sorted({
            (point["method"], point["legacy_selection_kind"])
            for point in legacy_points
        })
        for method, kind in groups:
            values = [
                point for point in legacy_points
                if point["method"] == method
                and point["legacy_selection_kind"] == kind
            ]
            color, marker = styles.get(method, ("#333333", "o"))
            legacy_ax.errorbar(
                [point["target_iteration"] for point in values],
                [point["mean_gain_bb_per_hand"] for point in values],
                yerr=[1.96 * point["standard_error_bb_per_hand"] for point in values],
                linestyle="none",
                marker=marker,
                markerfacecolor=("none" if kind == "final_response" else color),
                color=color,
                capsize=3,
                label=f"{method} legacy {'final' if kind == 'final_response' else 'selected'}",
            )
        legacy_ax.axhline(0.0, color="black", linewidth=0.8)
        for gap in missing:
            legacy_ax.axvline(
                gap["target_iteration"], color="#777777", linestyle=":", alpha=0.35
            )
        legacy_ax.set_xlabel("Target-training iteration")
        legacy_ax.set_ylabel("Legacy mean gain (bb/hand)")
        legacy_ax.set_title("Superseded legacy protocol—separate, unconnected points")
        legacy_ax.legend()
    screen_count = plan["splits"]["screening"]["trajectories_per_trained_candidate"]
    validation_count = plan["splits"]["validation"][
        "trajectories_per_trained_candidate"
    ]
    test_count = plan["splits"]["test"]["trajectories_for_frozen_selection"]
    fig.text(
        0.5,
        0.005,
        f"Six candidates (50–300): {screen_count:,} screening each; top 3 get "
        f"{validation_count:,} validation each; frozen choice gets {test_count:,} fresh test; "
        "zero control; 95% conditional Monte Carlo CI; saved responses use legacy adaptive training semantics."
        " Dotted guides mark unavailable grids, not observed gains.",
        ha="center",
        fontsize=7,
    )
    for kind in ("png", "svg", "pdf"):
        fig.savefig(bases[f"main_{kind}"], dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(searches),
        1,
        figsize=(8.0, max(4.0, 2.6 * len(searches))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (target_id, values) in zip(
        axes[:, 0], searches.items(), strict=True
    ):
        screening = sorted(values["screening"], key=lambda row: row["response_iteration"])
        ax.plot(
            [row["response_iteration"] for row in screening],
            [row["screening_or_validation_mean_gain_bb_per_hand"] for row in screening],
            marker="o",
            linewidth=1.0,
            label="screening",
        )
        validation = sorted(values["validation"], key=lambda row: row["response_iteration"])
        ax.scatter(
            [row["response_iteration"] for row in validation],
            [row["validation_mean_gain_bb_per_hand"] for row in validation],
            marker="x",
            s=50,
            label="validation",
        )
        best = values["validation_best_so_far_including_zero_control"]
        ax.step(
            [row[0] for row in best],
            [row[1] for row in best],
            where="post",
            linestyle="--",
            linewidth=1.0,
            label="validation best-so-far",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Gain (bb/hand)")
        ax.set_title(target_id)
        ax.legend(fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("Response-training iteration")
    fig.suptitle("Response-search diagnostic—not target convergence")
    for kind in ("png", "svg", "pdf"):
        fig.savefig(bases[f"search_{kind}"], dpi=240)
    plt.close(fig)
    print(bases["data"])


if __name__ == "__main__":
    main()
