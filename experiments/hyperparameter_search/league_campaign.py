"""Paired Normal/RB workload search with league-first policy evaluation.

The v1 worker and process supervisor remain the execution primitives. This v2
protocol selects N/B/steps before loss controls, measures frozen PG noise, and
reserves learned responses for the posterior-clock sensitivity comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import signal
import sys
from pathlib import Path

from experiments.hyperparameter_search.campaign import (
    ESTIMATORS,
    PYTHON,
    ROOT,
    Campaign,
    Job,
    atomic_json,
    campaign_lock,
    combination_candidate,
    digest,
    immutable_json,
    load_profile,
    pool_estimators,
    resolve_gpus,
    score_evaluations,
    source_digest,
)
from experiments.hyperparameter_search.protocol import apply_overrides, sha256_file

PIPELINE = "paired_league_v2"


def validate_profile(path: Path) -> dict:
    profile = load_profile(path)
    if profile.get("pipeline") != PIPELINE:
        raise ValueError("Use the shared league v2 profile")
    if profile["control"].get("posterior_clock", {}).get("exponent", 1) != 1:
        raise ValueError("V2 does not accept an uncalibrated posterior exponent")
    for group in ("workload_arms", "regularizer_arms"):
        arms = profile[group]
        names = [a["name"] for a in arms]
        if len(names) != len(set(names)) or "control" not in names:
            raise ValueError(f"{group} requires unique names and control")
        for arm in arms:
            if any("posterior_clock.exponent" in k for k in arm["overrides"]):
                raise ValueError("No exponent tuning in v2")
            resolved = apply_overrides(profile, arm["overrides"])
            if resolved["control"].get("posterior_clock", {}).get("exponent", 1) != 1:
                raise ValueError("No uncalibrated posterior exponent in v2 arms")
            if "scientific.mode.private_estimator" in arm["overrides"]:
                raise ValueError("Estimator is a paired condition, not an arm override")
    settings = profile["campaign"]
    for stage in ("workload", "regularizer", "confirm", "cold"):
        if (
            not math.isfinite(settings[f"{stage}_seconds"])
            or settings[f"{stage}_seconds"] <= 0
        ):
            raise ValueError("Stage durations must be finite and positive")
        if settings[f"{stage}_hands"] < 4 or settings[f"{stage}_hands"] % 4:
            raise ValueError(
                "League hand counts must be positive seat/button-balanced blocks"
            )
    noise = profile["gradient_noise"]
    if noise["repetitions"] < 2:
        raise ValueError("Noise audit requires independent gradient replications")
    return profile


def league_budget(profile: dict, *, smoke: bool, scale: float) -> dict:
    c = profile["campaign"]
    counts = {
        "workload": min(2, len(profile["workload_arms"]))
        if smoke
        else len(profile["workload_arms"]),
        "regularizer": 2 if smoke else len(profile["regularizer_arms"]),
        "confirm": 3,
        "cold": 2,
    }
    seeds = {
        "workload": 1,
        "regularizer": 1,
        "confirm": 1 if smoke else len(c["confirmation_seeds"]),
        "cold": 1,
    }
    rows = {}
    for stage, count in counts.items():
        policies = count * seeds[stage] * 2
        pairs = policies * (policies - 1) // 2
        rows[stage] = {
            "shared_settings": count,
            "training_seeds": seeds[stage],
            "policies": policies,
            "training_wall_seconds_upper_bound": policies
            * c[f"{stage}_seconds"]
            * scale,
            "league_pairs": pairs,
            "hands_per_pair": 4 if smoke else c[f"{stage}_hands"],
            "league_game_observations": pairs * (4 if smoke else c[f"{stage}_hands"]),
            "anchor_game_observations": policies
            * (4 if smoke else c[f"{stage}_hands"]),
        }
    br = 4 * c["clock_audit_br_seconds"] * scale if c["clock_probe_enabled"] else 0
    return {
        "pipeline": PIPELINE,
        "stages": rows,
        "clock_BR_training_wall_seconds": br,
        "nominal_training_wall_seconds": sum(
            r["training_wall_seconds_upper_bound"] for r in rows.values()
        )
        + br,
        "gradient_probe_jobs": (
            4
            if smoke
            else profile["gradient_noise"]["repetitions"]
            * 2
            * len(profile["gradient_noise"]["roots_per_rank"])
            * len(profile["gradient_noise"]["batches_per_rank"])
        ),
        "excludes": [
            "one-step gradient audit and CUDA analysis",
            "league and selective BR evaluation",
            "initialization",
        ],
        "smoke": smoke,
        "smoke_iteration_limit": 1 if smoke else None,
        "note": "Three GPUs; no routine BR outside the four clock-sensitivity targets. Smoke is iteration-capped, so nominal time ceilings are not an elapsed estimate.",
    }


def ordered_arms(report: dict) -> list[str]:
    if "ordered_arms" in report:
        return report["ordered_arms"]
    return sorted(
        report["arm_scores"],
        key=lambda k: (-report["arm_scores"][k]["mean_bb_per_hand"], k),
    )


def config_signature(profile: dict, overrides: dict) -> str:
    resolved = apply_overrides(profile, overrides)
    return digest({"training": resolved["training"], "control": resolved["control"]})


def confirmation_candidates(
    profile: dict,
    workloads: dict,
    workload_order: list[str],
    regularizers: dict,
    regularizer_order: list[str],
) -> list[dict]:
    base = workloads[workload_order[0]]["overrides"]
    combined = combination_candidate(regularizers, regularizer_order)
    best_changes = regularizers[regularizer_order[0]]["overrides"]
    if combined is not None and regularizer_order[0] != "control":
        best_changes = combined["overrides"]
    proposals = [
        {
            "name": "original_control",
            "overrides": workloads["control"]["overrides"],
            "basis": ["original_control"],
        },
        {"name": "workload_only", "overrides": base, "basis": [workload_order[0]]},
        {
            "name": "workload_plus_loss",
            "overrides": {**base, **best_changes},
            "basis": [
                workload_order[0],
                *(
                    combined["components"]
                    if combined and regularizer_order[0] != "control"
                    else [regularizer_order[0]]
                ),
            ],
        },
    ]
    proposals += [
        {
            "name": "loss_" + name,
            "overrides": {**base, **regularizers[name]["overrides"]},
            "basis": [workload_order[0], name],
        }
        for name in regularizer_order
        if name != "control"
    ]
    proposals += [
        {
            "name": "workload_" + name,
            "overrides": workloads[name]["overrides"],
            "basis": [name],
        }
        for name in workload_order
        if name != "control"
    ]
    selected, seen = [], set()
    for candidate in proposals:
        key = config_signature(profile, candidate["overrides"])
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
        if len(selected) == 3:
            return selected
    raise ValueError("Need three distinct configurations for confirmation")


class LeagueCampaign(Campaign):
    def prepare(self, *, resume: bool) -> None:
        manifest = {
            "pipeline": PIPELINE,
            "profile": self.profile,
            "profile_sha256": digest(self.profile),
            "requested_gpus": self.requested_gpus,
            "duration_scale": self.scale,
            "smoke": self.smoke,
            "source_sha256": source_digest(),
            "estimator_is_selection_variable": False,
        }
        path = self.output / "manifest.json"
        if path.exists() and not resume:
            raise ValueError("Output exists; use --resume")
        immutable_json(path, manifest)
        self.expected_source_digest = manifest["source_sha256"]
        immutable_json(
            self.output / "budget_plan.json",
            league_budget(self.profile, smoke=self.smoke, scale=self.scale),
        )
        for arm in self.profile["workload_arms"]:
            for mode in ESTIMATORS:
                self.worker_spec(
                    "workload",
                    arm["name"],
                    self.profile["campaign"]["screen_seed"],
                    private_estimator=mode,
                )

    def _read_result(self, job: Job) -> dict:
        if not job.name.startswith("noise/"):
            return super()._read_result(job)
        result = json.loads(job.result_path.read_text())
        if result.get("complete") is not True:
            raise ValueError("Incomplete gradient-noise diagnostic")
        if result.get("kind") == "gradient_noise_probe":
            path = Path(result["gradient"])
            if (
                not path.is_absolute()
                or not path.is_file()
                or sha256_file(path) != result["gradient_sha256"]
            ):
                raise ValueError("Gradient artifact identity changed")
        elif result.get("kind") != "gradient_noise_analysis":
            raise ValueError("Unknown diagnostic result kind")
        return result

    def noise_audit(self) -> dict | None:
        n = self.profile["gradient_noise"]
        if not n["enabled"]:
            return None
        groups, jobs = [], []
        for roots in [64] if self.smoke else n["roots_per_rank"]:
            for batch in [16] if self.smoke else n["batches_per_rank"]:
                for mode in ESTIMATORS:
                    label = f"n{roots}_b{batch}_{mode}"
                    paths = []
                    for repetition in range(2 if self.smoke else n["repetitions"]):
                        seed = n["seed_base"] + repetition
                        directory = (
                            self.output / "noise" / "probes" / label / f"seed_{seed}"
                        )
                        overrides = {
                            "run.workload.roots": roots,
                            "run.recovery.minimum_roots": roots,
                            "scientific.schedule.batch_size": batch,
                            "scientific.schedule.steps": 1,
                            "control.entropy.initial_coefficient": 0.0,
                            "scientific.objective.value.coefficient": 0.0,
                            "scientific.objective.reference_kl": None,
                            "run.execution.reference_cache": "off",
                        }
                        spec = {
                            "kind": "self_play",
                            "probe_role": "gradient_noise",
                            "gradient_noise": True,
                            "profile": self.profile,
                            "arm": "gradient_probe",
                            "overrides": overrides,
                            "private_estimator": mode,
                            "seed": seed,
                            "output_dir": str(directory),
                            "budget_seconds": 3600,
                            "max_iterations": 1,
                            "initialization": "model",
                            "smoke": self.smoke,
                        }
                        path = directory / "job.json"
                        immutable_json(path, spec)
                        result = directory / "result.json"
                        paths.append(str(result))
                        jobs.append(
                            (
                                str(directory.relative_to(self.output)),
                                directory,
                                path,
                                result,
                                spec,
                            )
                        )
                    groups.append(
                        {
                            "label": label,
                            "private_estimator": mode,
                            "roots": roots,
                            "batch": batch,
                            "artifacts": paths,
                        }
                    )
        for begin in range(0, len(jobs), len(self.gpus)):
            wave = [
                Job(
                    name,
                    directory,
                    [
                        str(PYTHON),
                        "-m",
                        "experiments.hyperparameter_search.gradient_noise",
                        "--spec",
                        str(path),
                    ],
                    [self.gpus[index]],
                    result,
                    digest(spec),
                )
                for index, (name, directory, path, result, spec) in enumerate(
                    jobs[begin : begin + len(self.gpus)]
                )
            ]
            self.run_jobs(wave)
        manifest = {
            "groups": groups,
            "bootstrap_repetitions": n["bootstrap_repetitions"],
            "seed": n["seed_base"],
            "purpose": "frozen initial PG noise audit; does not infer independent hands or set schedule weights",
        }
        manifest_path = self.output / "noise" / "manifest.json"
        immutable_json(manifest_path, manifest)
        directory = self.output / "noise" / "analysis"
        result = directory / "result.json"
        command = [
            str(PYTHON),
            "-m",
            "experiments.hyperparameter_search.gradient_noise",
            "--analyze",
            str(manifest_path),
            "--output",
            str(result),
            "--device",
            "cuda:0",
        ]
        return self.run_jobs(
            [
                Job(
                    "noise/analysis",
                    directory,
                    command,
                    [self.gpus[0]],
                    result,
                    digest(manifest),
                )
            ]
        )[0]

    def stage(
        self, stage: str, candidates: list[dict], seeds: list[int]
    ) -> tuple[dict, dict]:
        from experiments.hyperparameter_search.league import run_league

        self.arms.update({a["name"]: a for a in candidates})
        order = list(candidates)
        random.Random(
            self.profile["campaign"]["screen_seed"] + sum(map(ord, stage))
        ).shuffle(order)
        targets = {}
        for candidate in order:
            arm = candidate["name"]
            for seed in seeds:
                modes = list(ESTIMATORS)
                if (seed + list(self.arms).index(arm)) % 2:
                    modes.reverse()
                for mode in modes:
                    path, spec = self.worker_spec(
                        stage, arm, seed, private_estimator=mode
                    )
                    result = self.run_jobs([self._worker_job(path, spec, self.gpus)])[0]
                    key = f"{arm}__{mode}__s{seed}"
                    targets[key] = {
                        "arm": arm,
                        "private_estimator": mode,
                        "seed": seed,
                        "snapshot": result["snapshot"],
                        "checkpoint": result["checkpoint"],
                        "training_result": str(path.parent / "result.json"),
                    }
        immutable_json(self.output / stage / "targets.json", targets)
        report = run_league(
            self,
            stage,
            targets,
            4 if self.smoke else self.profile["campaign"][f"{stage}_hands"],
            f"{stage}_league_test_v2",
        )
        report["checkpoint_1110_anchor"] = self.anchor_panel(stage, targets)
        atomic_json(self.output / "league" / stage / "result.json", report)
        return report, targets

    def anchor_panel(self, stage: str, targets: dict) -> dict:
        """Keep a fixed starting-policy regression check separate from selection."""
        anchor_file = self.output / "league" / "checkpoint_1110_anchor.json"
        if stage == "workload":
            first = targets[min(targets)]
            snapshot = (
                Path(first["training_result"]).parent
                / "snapshots"
                / "iteration_00000000.pt"
            )
            immutable_json(
                anchor_file,
                {"snapshot": str(snapshot), "sha256": sha256_file(snapshot)},
            )
        anchor = json.loads(anchor_file.read_text())
        if sha256_file(anchor["snapshot"]) != anchor["sha256"]:
            raise ValueError("Starting-policy anchor changed")
        identities = sorted(targets)
        summaries = {}
        for begin in range(0, len(identities), len(self.gpus)):
            keys = identities[begin : begin + len(self.gpus)]
            jobs = []
            for index, key in enumerate(keys):
                directory = self.output / "anchor" / stage / key
                arguments = {
                    "candidate": targets[key]["snapshot"],
                    "opponent": anchor["snapshot"],
                    "hands": 4
                    if self.smoke
                    else self.profile["campaign"][f"{stage}_hands"],
                    "seed": 918271,
                    "split": f"{stage}_checkpoint_1110_anchor_v2",
                    "device": "cuda:0",
                }
                spec = {
                    "arguments": arguments,
                    "raw_match": True,
                    "anchor_sha256": anchor["sha256"],
                    "candidate_sha256": sha256_file(arguments["candidate"]),
                }
                immutable_json(directory / "job.json", spec)
                command = [
                    str(PYTHON),
                    "-m",
                    "experiments.hyperparameter_search.evaluation",
                    "--raw-match",
                ]
                for key_name, value in arguments.items():
                    command += ["--" + key_name.replace("_", "-"), str(value)]
                result = directory / "result.json"
                command += ["--output", str(result)]
                jobs.append(
                    Job(
                        str(directory.relative_to(self.output)),
                        directory,
                        command,
                        [self.gpus[index]],
                        result,
                        digest(spec),
                        evaluation=True,
                    )
                )
            for key, job, result in zip(keys, jobs, self.run_jobs(jobs), strict=True):
                spec = json.loads((job.directory / "job.json").read_text())
                if (
                    result["candidate_sha256"] != spec["candidate_sha256"]
                    or result["opponent_sha256"] != anchor["sha256"]
                    or result.get("raw_match") is not True
                ):
                    raise ValueError("Starting-policy match identity changed")
                summaries[key] = {
                    "arm": targets[key]["arm"],
                    "private_estimator": targets[key]["private_estimator"],
                    "seed": targets[key]["seed"],
                    "payoff": result["candidate_payoff_vs_fixed_opponent"],
                    "result": str(job.result_path),
                    "result_sha256": digest(result),
                }
        report = {
            "anchor": anchor,
            "matches": summaries,
            "used_for_automatic_selection": False,
            "purpose": "Detect shared regression against checkpoint 1110; relative league strength alone cannot do this",
        }
        atomic_json(self.output / "anchor" / stage / "report.json", report)
        return report

    def clock_audit(self, targets: dict) -> dict | None:
        if not self.profile["campaign"]["clock_probe_enabled"]:
            return None
        from experiments.hyperparameter_search.cross_panel import run_cross_panel

        seed = self.profile["campaign"]["regularizer_seed"]
        scores = {arm: {} for arm in ("control", "posterior_clock_raw")}
        panel = copy.copy(self)
        panel.output = self.output / "clock_audit_panel"
        panel.profile = copy.deepcopy(self.profile)
        panel.profile["campaign"]["cold_seed"] = seed
        for arm, mode_scores in scores.items():
            for mode in ESTIMATORS:
                target = targets[f"{arm}__{mode}__s{seed}"]
                path, _ = self.worker_spec(
                    "clock_audit",
                    arm,
                    seed,
                    private_estimator=mode,
                    restart=0,
                    target=target["checkpoint"],
                )
                parent = path.parent.parent
                target_result = json.loads(Path(target["training_result"]).read_text())
                atomic_json(parent / "target_result.json", target_result)
                responses = []
                for first in range(0, 3, len(self.gpus)):
                    jobs = []
                    for restart in range(first, min(first + len(self.gpus), 3)):
                        spec_path, spec = self.worker_spec(
                            "clock_audit",
                            arm,
                            seed,
                            private_estimator=mode,
                            restart=restart,
                            target=target["checkpoint"],
                        )
                        jobs.append(
                            self._worker_job(
                                spec_path, spec, [self.gpus[restart - first]]
                            )
                        )
                    responses.extend(self.run_jobs(jobs))
                evaluations = []
                for first in range(0, 3, len(self.gpus)):
                    jobs = []
                    for restart in range(first, min(first + len(self.gpus), 3)):
                        directory = parent / f"evaluation_{restart}"
                        arguments = {
                            "candidate": responses[restart]["snapshot"],
                            "opponent": target["snapshot"],
                            "baseline": target["snapshot"],
                            "hands": 4
                            if self.smoke
                            else self.profile["campaign"]["clock_probe_hands"],
                            "seed": seed,
                            "split": "clock_br_selection_v2",
                            "device": "cuda:0",
                        }
                        immutable_json(directory / "job.json", arguments)
                        command = [
                            str(PYTHON),
                            "-m",
                            "experiments.hyperparameter_search.evaluation",
                        ]
                        for key, value in arguments.items():
                            command += ["--" + key.replace("_", "-"), str(value)]
                        result_path = directory / "result.json"
                        command += ["--output", str(result_path)]
                        jobs.append(
                            Job(
                                str(directory.relative_to(self.output)),
                                directory,
                                command,
                                [self.gpus[restart - first]],
                                result_path,
                                digest(arguments),
                                evaluation=True,
                            )
                        )
                    evaluations.extend(self.run_jobs(jobs))
                mode_scores[mode] = score_evaluations(evaluations)
                copied = panel.output / "cold" / arm / f"seed_{seed}" / mode
                immutable_json(copied / "result.json", target_result)
                for restart, response in enumerate(responses):
                    immutable_json(
                        copied / f"response_{restart}" / "result.json", response
                    )
        pooled = {arm: pool_estimators(modes) for arm, modes in scores.items()}
        report = {
            "kind": "clock_sensitivity_br_audit",
            "own_target": pooled,
            "common_attack_panel": run_cross_panel(panel, pooled),
            "used_for_automatic_selection": False,
            "interpretation": "Sensitive clock regression audit; main selection uses league strength, not BR scores",
        }
        atomic_json(self.output / "clock_audit" / "report.json", report)
        return report

    def run(self) -> None:
        self.gpus = resolve_gpus(self.requested_gpus)
        immutable_json(
            self.output / "gpu_assignment.json",
            {"requested": self.requested_gpus, "uuids": self.gpus},
        )
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[sig] = signal.signal(sig, self._signal)
        try:
            c = self.profile["campaign"]
            noise = self.noise_audit()
            workloads = copy.deepcopy(self.profile["workload_arms"])
            if self.smoke:
                workloads = [workloads[0], workloads[-1]]
            workload_report, _ = self.stage("workload", workloads, [c["screen_seed"]])
            workload_map = {a["name"]: a for a in workloads}
            workload_order = ordered_arms(workload_report)
            best_workload = workload_map[workload_order[0]]["overrides"]
            regularizers = copy.deepcopy(self.profile["regularizer_arms"])
            if self.smoke:
                regularizers = [
                    a
                    for a in regularizers
                    if a["name"] in ("control", "posterior_clock_raw")
                ]
            reg_base = {a["name"]: copy.deepcopy(a) for a in regularizers}
            combined = [
                {**a, "overrides": {**best_workload, **a["overrides"]}}
                for a in regularizers
            ]
            reg_report, reg_targets = self.stage(
                "regularizer", combined, [c["regularizer_seed"]]
            )
            audit = self.clock_audit(reg_targets)
            candidates = confirmation_candidates(
                self.profile,
                workload_map,
                workload_order,
                reg_base,
                ordered_arms(reg_report),
            )
            immutable_json(self.output / "confirmation_candidates.json", candidates)
            confirm_report, _ = self.stage(
                "confirm",
                candidates,
                c["confirmation_seeds"][:1] if self.smoke else c["confirmation_seeds"],
            )
            winners = ordered_arms(confirm_report)[:2]
            cold_candidates = [a for a in candidates if a["name"] in winners]
            cold_report, _ = self.stage("cold", cold_candidates, [c["cold_seed"]])
            selected_name = ordered_arms(cold_report)[0]
            selected_arm = next(a for a in candidates if a["name"] == selected_name)
            selected = apply_overrides(self.profile, selected_arm["overrides"])
            training_profile = {
                k: selected[k] for k in ("initialization", "training", "control")
            }
            training_profile.update(
                kind="nlhe_training_profile_v1",
                status="provisional league-selected shared parameters",
                selection={
                    "arm": selected_name,
                    "basis": selected_arm["basis"],
                    "estimator_selected": False,
                    "posterior_multiplier_calibrated": False,
                },
            )
            selected_profile_path = ROOT / self.profile.get(
                "selected_profile_path",
                "configs/profiles/nlhe_hu_32bb_shared_league_selected_v2.json",
            )
            if not self.smoke:
                atomic_json(self.output / "recommended_profile.json", training_profile)
                # JSON is also a valid YAML training profile. Never overwrite a
                # recommendation from a different campaign without explicit review.
                immutable_json(selected_profile_path, training_profile)
            atomic_json(
                self.output / "report.json",
                {
                    "complete": True,
                    "smoke": self.smoke,
                    "pipeline": PIPELINE,
                    "gradient_noise": noise,
                    "workload": workload_report,
                    "regularizer": reg_report,
                    "clock_br_audit": audit,
                    "confirmation": confirm_report,
                    "cold": cold_report,
                    "recommendation": None
                    if self.smoke
                    else {
                        "shared_arm": selected_name,
                        "profile": str(selected_profile_path),
                        "campaign_copy": str(self.output / "recommended_profile.json"),
                        "provisional": True,
                        "limitations": "Population-relative crossplay; strategic cycles and finite-hand/training-seed uncertainty remain. Review clock BR audit separately.",
                    },
                },
            )
        finally:
            for sig, handler in self._old_handlers.items():
                signal.signal(sig, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--duration-scale", type=float, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        gpus = args.gpus.split(",")
        if len(gpus) != 3 and not args.smoke:
            raise ValueError(
                "The registered full campaign requires three GPUs; its N/B values are per rank"
            )
        if not math.isfinite(args.duration_scale) or args.duration_scale <= 0:
            raise ValueError("duration-scale must be finite and positive")
        profile = validate_profile(args.profile)
        with campaign_lock(args.output):
            runner = LeagueCampaign(
                profile, args.output, gpus, scale=args.duration_scale, smoke=args.smoke
            )
            runner.prepare(resume=args.resume)
            if not args.prepare_only:
                runner.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - fail closed at the process-supervision CLI boundary
        print(f"League campaign stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
