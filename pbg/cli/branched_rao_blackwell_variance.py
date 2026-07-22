"""Staged launcher/analyzer for the branched RB variance experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

import torch
from pbg.core.devices import resolve_execution_device, validate_workload_capabilities

from pbg.eval.branched_rao_blackwell_variance import (
    BranchedVarianceRunner,
    SCHEMA_VERSION,
    atomic_json,
    beta_key,
    component_summary,
    ensure_manifest,
    geometry_summary,
    load_config,
    variance_table,
)


def _device(argument: str | None) -> torch.device:
    if argument is None:
        raise ValueError("--device must explicitly select cuda or mps")
    device = resolve_execution_device(argument, workload="variance evaluation")
    validate_workload_capabilities(device, workload="variance evaluation")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _save_pt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _job_path(config: Mapping[str, Any], kind: str, beta: float, repeat: int) -> Path:
    suffix = "json" if kind == "geometry" else "pt"
    return Path(str(config["out_dir"])) / kind / beta_key(beta) / f"{repeat:03d}.{suffix}"


def _load_moments(config: Mapping[str, Any], beta: float, count: int | None = None) -> list[dict[str, Any]]:
    directory = _job_path(config, "moments", beta, 0).parent
    paths = sorted(directory.glob("*.pt"))
    if count is not None:
        wanted = {f"{repeat:03d}.pt" for repeat in range(count)}
        present = {path.name for path in paths}
        missing = sorted(wanted - present)
        if missing:
            raise RuntimeError(f"missing moment jobs for beta={beta}: {missing}")
        paths = [directory / f"{repeat:03d}.pt" for repeat in range(count)]
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def run_smoke(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    smoke = config["smoke"]
    bounded = json.loads(json.dumps(config))
    bounded["rollout"]["branching_upper_cutoff_nodes"] = int(smoke["upper_cutoff_nodes"])
    bounded["oracle"]["n_oracle"] = int(smoke["oracle_samples"])
    bounded["probes"]["parameter_count"] = int(smoke["parameter_probes"])
    bounded["probes"]["node_hutchinson_count"] = int(smoke["node_hutchinson_probes"])
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    runner = BranchedVarianceRunner(bounded, device)
    result = runner.moment_tree(
        float(smoke["beta"]),
        -1,
        n_samples=int(smoke["oracle_samples"]),
    )
    if not all(
        math.isfinite(float(value))
        for key in ("gradient", "minibatch_coefficient_raw", "completion_pair_coefficient", "completion_node_coefficient")
        for value in result[key]
    ):
        raise RuntimeError("smoke produced a nonfinite statistic")
    result["smoke"] = True
    result["allocated_limit_bytes"] = runner.memory_limit
    return result


def _run_geometry_jobs(
    runner: BranchedVarianceRunner,
    config: Mapping[str, Any],
    beta: float,
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    count = int(config["stages"]["geometry_repeats"])
    records = []
    for repeat in range(count):
        path = _job_path(config, "geometry", beta, repeat)
        if path.exists() and resume:
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            sample, record = runner.sample_tree(beta, repeat)
            record["schema_version"] = SCHEMA_VERSION
            record["success"] = True
            atomic_json(path, record)
            del sample
        records.append(record)
        print(
            f"[geometry beta={beta:.3f}] {repeat + 1}/{count} "
            f"nodes={record['nodes']} cutoff={record['upper_cutoff_activated']} "
            f"seconds={record['rollout_seconds']:.1f}",
            flush=True,
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "beta": beta,
        "samples": count,
        "summary": geometry_summary(records),
    }
    atomic_json(_job_path(config, "geometry", beta, 0).parent / "summary.json", summary)
    return records


def _run_moment_jobs(
    runner: BranchedVarianceRunner,
    config: Mapping[str, Any],
    beta: float,
    repeats: list[int],
    *,
    resume: bool,
    manifest_fingerprint: str,
) -> list[dict[str, Any]]:
    records = []
    for repeat in repeats:
        path = _job_path(config, "moments", beta, repeat)
        if path.exists() and resume:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("manifest_fingerprint") != manifest_fingerprint:
                raise RuntimeError(f"moment job has wrong manifest: {path}")
        else:
            payload = runner.moment_tree(beta, repeat)
            payload["manifest_fingerprint"] = manifest_fingerprint
            _save_pt(path, payload)
        records.append(payload)
        print(
            f"[moments beta={beta:.3f}] repeat={repeat} "
            f"nodes={payload['geometry']['nodes']} seconds={payload['seconds']:.1f} "
            f"peak={payload['peak_allocated_bytes'] / 1024**3:.2f}GiB",
            flush=True,
        )
    return records


def _objective(config: Mapping[str, Any], records: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    summary = component_summary(records, int(config["oracle"]["n_oracle"]))
    reference_B = int(config["report"]["selection_reference_B"])
    reference_n = int(config["report"]["selection_reference_n"])
    row = variance_table(summary["mean"], [reference_B], [reference_n])[0]
    return float(row["total"]), summary


def _refinement_candidates(config: Mapping[str, Any], winner: float) -> list[float]:
    beta = config["beta"]
    step = float(beta["refine_step"])
    lower, upper = float(beta["minimum"]), float(beta["maximum"])
    coarse = {float(value) for value in beta["coarse"]}
    return sorted(
        candidate
        for candidate in (winner - step, winner + step)
        if lower <= candidate <= upper and candidate not in coarse
    )


def _timing_gate(
    config: Mapping[str, Any], geometry_records: list[dict[str, Any]], first_moment: Mapping[str, Any]
) -> dict[str, Any]:
    coarse_count = len(config["beta"]["coarse"])
    # Two refinement candidates is the worst case. The final maximum includes
    # its pilot eight, so do not double-count those jobs.
    maximum_moment_jobs = (coarse_count + 2) * int(config["stages"]["pilot_trees_per_beta"]) + (
        int(config["stages"]["maximum_final_trees"]) - int(config["stages"]["pilot_trees_per_beta"])
    )
    moment_seconds = float(first_moment["seconds"])
    geometry_seconds = sum(float(row["rollout_seconds"]) for row in geometry_records)
    # Geometry_records contains one beta; scale by the worst-case five beta candidates.
    projected_seconds = geometry_seconds * 5 + moment_seconds * maximum_moment_jobs
    limit_seconds = float(config["constraints"]["maximum_projected_active_gpu_hours"]) * 3600
    result = {
        "success": projected_seconds <= limit_seconds,
        "first_full_moment_seconds": moment_seconds,
        "maximum_moment_jobs": maximum_moment_jobs,
        "projected_active_gpu_seconds": projected_seconds,
        "projected_active_gpu_hours": projected_seconds / 3600,
        "limit_active_gpu_hours": limit_seconds / 3600,
    }
    atomic_json(Path(str(config["out_dir"])) / "timing_gate.json", result)
    if not result["success"]:
        raise RuntimeError(
            f"projected campaign is {result['projected_active_gpu_hours']:.1f} active GPU-hours, "
            f"above the {result['limit_active_gpu_hours']:.1f}-hour gate"
        )
    return result


def run_experiment(
    config: dict[str, Any], device: torch.device, manifest: Mapping[str, Any], *, resume: bool
) -> None:
    runner = BranchedVarianceRunner(config, device)
    coarse = [float(value) for value in config["beta"]["coarse"]]
    geometry: dict[float, list[dict[str, Any]]] = {}
    for beta in coarse:
        geometry[beta] = _run_geometry_jobs(runner, config, beta, resume=resume)

    pilot_count = int(config["stages"]["pilot_trees_per_beta"])
    pilot_summaries: dict[float, dict[str, Any]] = {}
    for beta_index, beta in enumerate(coarse):
        # Time a high-geometry pilot first; immutable repeat addresses remain unchanged.
        ordered = sorted(
            range(pilot_count),
            key=lambda repeat: float(geometry[beta][repeat]["nodes"]),
            reverse=True,
        )
        records = _run_moment_jobs(
            runner,
            config,
            beta,
            ordered,
            resume=resume,
            manifest_fingerprint=str(manifest["fingerprint"]),
        )
        canonical = _load_moments(config, beta, pilot_count)
        objective, summary = _objective(config, canonical)
        pilot_summaries[beta] = {"objective": objective, "components": summary}
        if beta_index == 0:
            _timing_gate(config, geometry[beta], records[:1][0])
    coarse_winner = min(coarse, key=lambda value: pilot_summaries[value]["objective"])
    refine = _refinement_candidates(config, coarse_winner)
    for beta in refine:
        geometry[beta] = _run_geometry_jobs(runner, config, beta, resume=resume)
        _run_moment_jobs(
            runner,
            config,
            beta,
            list(range(pilot_count)),
            resume=resume,
            manifest_fingerprint=str(manifest["fingerprint"]),
        )
        objective, summary = _objective(config, _load_moments(config, beta, pilot_count))
        pilot_summaries[beta] = {"objective": objective, "components": summary}

    selected = min(pilot_summaries, key=lambda value: pilot_summaries[value]["objective"])
    selection = {
        "success": True,
        "coarse_winner": coarse_winner,
        "refinement_candidates": refine,
        "selected_beta": selected,
        "reference_B": int(config["report"]["selection_reference_B"]),
        "reference_n": int(config["report"]["selection_reference_n"]),
        "pilot": {str(beta): value for beta, value in sorted(pilot_summaries.items())},
    }
    atomic_json(Path(str(config["out_dir"])) / "selection.json", selection)

    target = int(config["stages"]["final_trees"])
    maximum = int(config["stages"]["maximum_final_trees"])
    block = int(config["stages"]["adaptive_block"])
    tolerance = float(config["stages"]["target_component_relative_se"])
    current = pilot_count
    while current < target:
        stop = min(target, current + block)
        _run_moment_jobs(
            runner,
            config,
            selected,
            list(range(current, stop)),
            resume=resume,
            manifest_fingerprint=str(manifest["fingerprint"]),
        )
        current = stop
    while current < maximum:
        records = _load_moments(config, selected, current)
        summary = component_summary(records, int(config["oracle"]["n_oracle"]))
        if max(summary["relative_se"].values()) <= tolerance:
            break
        stop = min(maximum, current + block)
        _run_moment_jobs(
            runner,
            config,
            selected,
            list(range(current, stop)),
            resume=resume,
            manifest_fingerprint=str(manifest["fingerprint"]),
        )
        current = stop

    # The selected-beta large-n check uses a disjoint negative repeat address
    # and is kept separate from the final tree-variance sample.
    check_path = Path(str(config["out_dir"])) / "selected_beta_oracle_check.pt"
    if not (resume and check_path.exists()):
        check = runner.moment_tree(
            selected,
            -2,
            n_samples=int(config["oracle"]["selected_beta_check_samples"]),
        )
        check["manifest_fingerprint"] = manifest["fingerprint"]
        _save_pt(check_path, check)
    analyze(config, manifest)


def analyze(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(str(config["out_dir"]))
    selection_path = out / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("selection.json does not exist")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = float(selection["selected_beta"])
    records = _load_moments(config, selected)
    components = component_summary(records, int(config["oracle"]["n_oracle"]))
    table = variance_table(components["mean"], config["report"]["B"], config["report"]["n"])
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "variance_surface.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    geometry = {}
    for directory in sorted((out / "geometry").glob("beta_*")):
        path = directory / "summary.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            geometry[str(payload["beta"])] = payload
    summary = {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "manifest_fingerprint": manifest["fingerprint"],
        "checkpoint": 1110,
        "selected_beta": selected,
        "moment_trees": len(records),
        "components": components,
        "geometry": geometry,
        "variance_surface": table,
        "recommended_B_rule": "choose the smallest reported B satisfying the deployment variance/time constraint; no sequential minibatch coefficient was reused",
        "K_status": "deferred to a short wall-time/KL training probe after beta and B selection",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(
        out / "inventory.json",
        {
            "success": True,
            "geometry_jobs": len(list((out / "geometry").glob("beta_*/*.json"))),
            "moment_jobs": len(list((out / "moments").glob("beta_*/*.pt"))),
            "selected_beta": selected,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "run", "analyze"))
    parser.add_argument("--device", required=True, choices=("cuda", "mps"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manifest = ensure_manifest(config, resume=args.resume)
    device = _device(args.device)
    if args.phase == "smoke":
        result = run_smoke(config, device)
        result["manifest_fingerprint"] = manifest["fingerprint"]
        atomic_json(Path(str(config["out_dir"])) / "smoke.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    elif args.phase == "run":
        run_experiment(config, device, manifest, resume=args.resume)
    else:
        analyze(config, manifest)


if __name__ == "__main__":
    main()
