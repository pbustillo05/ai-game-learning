#!/usr/bin/env python3
"""Report saved normal/RB 10k training diagnostics; never rerun training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "Normal": "nlhe_hu_32bb_normal_60h_20260812",
    "RB (dualclip5)": "nlhe_hu_32bb_rb_dualclip5_60h_20260814",
}
COLORS = {"Normal": "#2563b4", "RB (dualclip5)": "#d35a23"}


def flatten(value, prefix=""):
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            yield from flatten(item, name + ".")
        elif isinstance(item, (int, float)) or item is None:
            yield name, item


def smooth(values, window=101):
    """Trailing mean, excluding nonfinite samples; shortened initial window."""
    finite = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(finite, values, 0.))]
    counts = np.r_[0, np.cumsum(finite)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return np.divide(sums[end] - sums[start], counts[end] - counts[start],
                     out=np.full(len(values), np.nan), where=counts[end] > counts[start])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/experiments/nlhe_hu_32bb_normal_vs_rb_diagnostics_10k_20260925")
    parser.add_argument("--validate-device", required=True, help="Explicit CUDA device for numerical checks, e.g. cuda:0")
    args = parser.parse_args()
    if not args.validate_device.startswith("cuda:"):
        parser.error("Validation requires an explicit CUDA device")
    import torch
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data, provenance, checks = {}, {}, {}
    all_keys = set()
    for label, name in RUNS.items():
        source = ROOT / "runs/experiments" / name / "metrics.jsonl"
        raw = source.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows = [row for row in rows if 0 <= row["iteration"] < 10000]
        assert [row["iteration"] for row in rows] == list(range(10000)), "Missing, duplicate or unordered iteration"
        assert all(len(row["lanes"]) == 1 for row in rows)
        flat = [dict(flatten(row["lanes"][0])) for row in rows]
        keys = sorted(set().union(*(row.keys() for row in flat)))
        d = {key: np.array([row.get(key) if row.get(key) is not None else np.nan for row in flat], dtype=float) for key in keys}
        all_keys.update(keys)
        d["completed_iterations"] = np.arange(1, len(rows) + 1)
        d["derived.value_contribution"] = .125 * d["objective.value"]
        d["derived.entropy_contribution"] = -.025 * d["objective.entropy"]
        d["derived.reference_contribution"] = .05 * d["objective.reference_kl"]
        d["derived.skipped_updates"] = d["minibatches"] - d["optimizer_steps"]
        d["derived.cumulative_skipped_updates"] = np.cumsum(d["derived.skipped_updates"])
        d["derived.reference_age"] = d["iteration"] - d["reference.reference_iteration"]
        d["derived.presentations_per_node"] = d["statistical_presentations"] / d["player_nodes"]
        for street, family in [("preflop", "pre"), ("postflop", "post_2")]:
            d[f"derived.{street}_actions_per_hand"] = d[f"structure.decision_counts_by_street.{street}"] / d["structure.trajectory_length.count"]
            d[f"derived.{street}_rows_per_call"] = d[f"structure.model_timing_by_family.{family}.rows"] / d[f"structure.model_timing_by_family.{family}.batches"]
        d["derived.model_seconds"] = d["structure.model_timing_by_family.pre.seconds"] + d["structure.model_timing_by_family.post_2.seconds"]
        denom = d["completion_health.leaf_rows"] * d["completion_health.samples_per_leaf"]
        d["derived.effective_fraction"] = np.divide(d["completion_health.posterior_effective_hands"], denom, out=np.full(len(rows), np.nan), where=denom > 0)
        if "gradient.finite" in d:
            d["derived.nonfinite_gradient"] = 1 - d["gradient.finite"]
            d["derived.finite_gradient_clipped"] = np.where(d["gradient.finite"] == 1, (d["gradient.clip_scale_min"] < 1).astype(float), np.nan)
        # Validate every saved row on the explicitly selected idle CUDA device.
        t = lambda key: torch.tensor(d[key], device=args.validate_device, dtype=torch.float64)
        predicted = t("objective.policy") + .125*t("objective.value") - .025*t("objective.entropy") + .05*t("objective.reference_kl")
        residual = (predicted - t("objective.total")).abs().max().item()
        assert residual < 1e-6, residual
        assert torch.equal(t("structure.decision_counts_by_street.preflop") + t("structure.decision_counts_by_street.postflop"), t("player_nodes"))
        assert torch.allclose(t("structure.trajectory_length.mean") * t("structure.trajectory_length.count"), t("player_nodes"))
        for street, family in [("preflop", "pre"), ("postflop", "post_2")]:
            assert torch.equal(t(f"structure.decision_counts_by_street.{street}"), t(f"structure.model_timing_by_family.{family}.rows"))
        assert torch.equal(t("logical_work.optimizer_steps"), t("optimizer_steps").cumsum(0))
        assert torch.equal(t("logical_work.trajectories"), t("structure.trajectory_length.count").cumsum(0))
        checks[label] = {"rows": len(rows), "loss_reconstruction_max_abs_error": residual, "skipped_updates": int(d["derived.skipped_updates"].sum())}
        provenance[label] = {"source": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "identity_first": rows[0]["lanes"][0]["identity"], "identity_last": rows[-1]["lanes"][0]["identity"]}
        data[label] = d
        filename = "normal" if label == "Normal" else "rb"
        with (out / f"{filename}_all_metrics.csv").open("w") as f:
            writer = csv.writer(f)
            columns = sorted(d)
            writer.writerow(columns)
            writer.writerows(zip(*(d[key] for key in columns)))
    torch.cuda.synchronize()
    (out / "validation.json").write_text(json.dumps({"device": args.validate_device, "gpu": torch.cuda.get_device_name(), "checks": checks, "provenance": provenance}, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white"})
    pages = []
    pdf = PdfPages(out / "training_diagnostics_10k.pdf")

    def panel(ax, key, title, unit="", scale="linear", factor=1., raw_only=False):
        for label, d in data.items():
            if key not in d or not np.any(np.isfinite(d[key])):
                continue
            x, y = d["completed_iterations"], d[key]*factor
            y = np.where(np.isfinite(y), y, np.nan)
            ax.plot(x, y, color=COLORS[label], alpha=.7 if raw_only else .19, lw=.6, rasterized=True, label=label if raw_only else None)
            if not raw_only:
                ax.plot(x, smooth(y), color=COLORS[label], lw=1.5, label=label)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-5)
        else:
            ax.set_yscale(scale)
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.set_xlabel("Completed training iterations")
        ax.set_xlim(0, 10000)
        ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000], ["0", "2k", "4k", "6k", "8k", "10k"])
        ax.legend(loc="best", fontsize=8, framealpha=.8)

    def sheet(slug, title, specs, note="Faint: every iteration. Solid: trailing 101-iteration mean. Blue: Normal. Orange: RB (dualclip5).", xlim=None):
        nrows = math.ceil(len(specs)/2)
        fig, axes = plt.subplots(nrows, 2, figsize=(14, nrows*3.1 + .9), squeeze=False)
        for ax, spec in zip(axes.flat, specs):
            panel(ax, *spec)
            if xlim is not None:
                ax.set_xlim(*xlim)
                ticks = np.arange(xlim[0], xlim[1]+1, 100)
                ax.set_xticks(ticks, [str(tick) for tick in ticks])
                ax.axvline(5487, color="#777777", lw=.8, ls="--")
        for ax in list(axes.flat)[len(specs):]:
            ax.set_visible(False)
        height = fig.get_size_inches()[1]
        fig.suptitle(title, fontsize=17, x=.06, ha="left", y=1-.12/height)
        fig.text(.06, 1-.52/height, note, fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, 0, 1, 1-.65/height))
        fig.savefig(out / f"{slug}.png", dpi=150)
        pdf.savefig(fig)
        plt.close(fig)
        pages.append((slug, title, note))

    sheet("01_rollout_actions", "Actions observed in each rollout | 1,536 hands per iteration", [
        ("structure.decision_counts_by_street.preflop", "Preflop actions / model rows", "actions / rollout"),
        ("structure.decision_counts_by_street.postflop", "Postflop actions / model rows", "actions / rollout"),
        ("derived.preflop_actions_per_hand", "Preflop actions per starting hand", "actions / hand"),
        ("derived.postflop_actions_per_hand", "Postflop actions per starting hand", "actions / hand"),
        ("structure.decision_fractions_by_street.postflop", "Postflop share of all actions", "%", "linear", 100),
        ("structure.trajectory_length.mean", "Total actions per starting hand", "actions / hand"),
    ])
    sheet("02_model_calls", "Rollout model calls and GPU inference time", [
        ("structure.model_timing_by_family.pre.batches", "Preflop batched forward calls", "calls / rollout"),
        ("structure.model_timing_by_family.post_2.batches", "Postflop batched forward calls", "calls / rollout"),
        ("structure.model_timing_by_family.pre.seconds", "Preflop model time", "seconds / rollout"),
        ("structure.model_timing_by_family.post_2.seconds", "Postflop model time", "seconds / rollout"),
        ("derived.preflop_rows_per_call", "Preflop batch occupancy", "rows / call"),
        ("derived.postflop_rows_per_call", "Postflop batch occupancy", "rows / call"),
    ], "Rollout forwards only; excludes learner/reference/completion calls. Faint: raw; solid: trailing 101-iteration mean.")
    sheet("03_loss_terms", "All active loss terms | weighted contributions and raw value loss", [
        ("objective.total", "Total objective", "loss"),
        ("objective.policy", "Policy contribution (coefficient 1)", "loss"),
        ("derived.value_contribution", "Value contribution: +0.125 × value", "loss"),
        ("derived.entropy_contribution", "Entropy contribution: −0.025 × entropy", "loss"),
        ("derived.reference_contribution", "Reference KL contribution: +0.05 × KL", "loss"),
        ("objective.value", "Raw clipped value loss", "squared normalized utility"),
    ])
    sheet("04_policy_diagnostics", "Policy diagnostics | preserve the large RB KL spikes", [
        ("objective.entropy", "Policy entropy", "logged entropy"),
        ("objective.clip_fraction", "PPO ratio outside [0.8, 1.2]", "%", "linear", 100),
        ("objective.approximate_kl", "Approximate KL (uncapped ratios; log scale)", "diagnostic", "log"),
        ("objective.reference_kl", "KL to reference policy", "KL"),
        ("objective.negative_advantage_cap_fraction", "RB negative-advantage cap activation", "%", "linear", 100),
        ("objective.negative_advantage_log_ratio_max", "RB maximum negative-advantage log ratio", "log ratio", "linear", 1., True),
    ], "Clipping fractions are learner averages (posterior-weighted for RB), not rollout-action frequencies. Faint: raw; solid: 101-iteration mean.")
    sheet("05_gradient_health", "Gradient health and skipped optimizer updates", [
        ("gradient.norm_max", "RB maximum pre-clip gradient norm (finite values)", "norm", "log", 1., True),
        ("gradient.norm_mean", "RB mean pre-clip gradient norm (finite values)", "norm", "log"),
        ("gradient.clip_scale_min", "RB minimum gradient clip multiplier", "1 = no clipping", "linear", 1., True),
        ("derived.nonfinite_gradient", "RB iterations containing nonfinite gradients", "indicator", "linear", 1., True),
        ("derived.cumulative_skipped_updates", "Cumulative skipped optimizer updates", "updates", "linear", 1., True),
        ("gradient.loss_scale_after", "RB FP16 loss scale after iteration", "scale", "log", 1., True),
    ], "Normal did not log gradient fields. Nonfinite norms omitted; clip multiplier 0 encodes skipped nonfinite gradients, not ordinary clipping.")
    sheet("06_structure_and_work", "Trajectory structure, learner exposure and completion health", [
        ("structure.trajectory_length.p95", "95th-percentile trajectory length", "actions"),
        ("structure.trajectory_length.maximum", "Maximum trajectory length", "actions"),
        ("derived.presentations_per_node", "Learner presentations / observed action", "presentations / action"),
        ("derived.effective_fraction", "RB effective hands / nominal completion samples", "%", "linear", 100),
        ("objective.negative_advantage_influence_max", "RB maximum capped negative-advantage influence", "logged influence", "log", 1., True),
        ("derived.reference_age", "Age of frozen reference", "iterations", "linear", 1., True),
    ], "Reference refresh: Normal 305 iterations; RB 110. Completion logical_samples is cumulative; effective_hands is per rollout.")
    sheet("07_memory", "Saved GPU memory diagnostics", [
        ("memory.allocated_bytes", "Allocated memory at observation", "GiB", "linear", 2**-30),
        ("memory.reserved_bytes", "Reserved memory at observation", "GiB", "linear", 2**-30),
        ("memory.peak_allocated_bytes", "Peak allocated memory", "GiB", "linear", 2**-30),
        ("memory.peak_reserved_bytes", "Peak reserved memory", "GiB", "linear", 2**-30),
    ], "Peak fields are saved allocator high-water marks; restart resets are possible. These are not per-iteration allocation deltas.")

    sheet("08_rb_transition", "Close-up of the RB transition near iteration 5,487", [
        ("objective.approximate_kl", "Uncapped approximate KL", "diagnostic", "log", 1., True),
        ("objective.reference_kl", "KL to frozen reference", "KL", "linear", 1., True),
        ("objective.value", "Raw value loss", "loss", "linear", 1., True),
        ("objective.entropy", "Policy entropy", "logged entropy", "linear", 1., True),
        ("structure.decision_counts_by_street.preflop", "Preflop actions per rollout", "actions", "linear", 1., True),
        ("objective.negative_advantage_cap_fraction", "Negative-advantage ratio cap activation", "%", "linear", 100., True),
    ], "Raw values. Dashed: largest RB approximate-KL spike at completed iteration 5,487. Shared y-limits with full-run views.", xlim=(5300, 5700))
    sheet("09_clipping_frequency", "Gradient clipping and skipped-step frequency", [
        ("derived.finite_gradient_clipped", "RB iterations with any gradient clipping (finite only)", "%", "linear", 100.),
        ("derived.skipped_updates", "Skipped optimizer steps per iteration", "updates", "linear", 1., True),
    ], "RB clipping: at least one gradient clipped; nonfinite-gradient iterations excluded. Solid: trailing 101-iteration percentage.")

    # Every varying saved numeric field has a raw appendix plot, including aliases.
    inventory, varying = [], []
    for key in sorted(all_keys):
        record = {"metric": key}
        changes = False
        for label, d in data.items():
            a = d.get(key, np.full(10000, np.nan))
            finite = a[np.isfinite(a)]
            changes |= len(np.unique(finite)) > 1 or np.any(np.isinf(a))
            record[label] = {"finite_count": len(finite), "nonfinite_count": int((~np.isfinite(a)).sum()), "min": float(finite.min()) if len(finite) else None, "max": float(finite.max()) if len(finite) else None}
        inventory.append(record)
        if changes:
            varying.append(key)
    for start in range(0, len(varying), 8):
        specs = []
        for key in varying[start:start+8]:
            scale = "log" if key in {"objective.approximate_kl", "gradient.norm_mean", "gradient.norm_max", "gradient.loss_scale_before", "gradient.loss_scale_after"} else "linear"
            specs.append((key, key.replace("structure.", "").replace("memory.admitted_phase_required_bytes.", "phase_bytes."), "saved units", scale, 1., True))
        sheet(f"appendix_{start//8+1:02d}", f"Raw saved scalar metrics | appendix {start//8+1}", specs, "All 10,000 samples; no smoothing. Constants/null-only fields are in the inventory. Infinite values: see nonfinite counts.")
    pdf.close()
    (out / "metric_inventory.json").write_text(json.dumps(inventory, indent=2))

    summaries = []
    for label, d in data.items():
        for key, values in d.items():
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            summaries.append({"run": label, "metric": key, "first_500_mean": float(np.mean(values[:500][np.isfinite(values[:500])])), "last_500_mean": float(np.mean(values[-500:][np.isfinite(values[-500:])])), "minimum_finite": float(finite.min()), "maximum_finite": float(finite.max()), "nonfinite_count": int((~np.isfinite(values)).sum())})
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    notes = """Both logs contain all 10,000 iterations with no gaps or duplicate rows. The x-axis is completed iterations = logged zero-based iteration + 1, matching checkpoint names. Each rollout starts 1,536 hands; each learner iteration presents 9,728 rows in 38 minibatches. These are rollout decision counts, not distinct hands reaching a street. Flop/turn/river were combined as postflop (post_2 means the two-player postflop model family, not turn). No per-street loss/entropy, fold/showdown rates, value-clipping fraction, normal gradient norms, or per-step gradient-clipping frequency were saved in these metric logs. The minimum clip multiplier only identifies whether at least one gradient was clipped in an iteration; 0 encodes a skipped nonfinite-gradient step, so those iterations must be considered separately.

Total = policy + 0.125 value − 0.025 entropy + 0.05 reference KL. All 20,000 rows pass this reconstruction within 1e-6 on CUDA. Objectives are learner minibatch averages; RB integrates over posterior private-hand weights. Approximate KL uses the uncapped expression r − 1 − log(r), so enormous values can coexist with a bounded negative-advantage policy term. The RB cap activates when advantage < 0 and ratio > 5; its fraction is across all posterior-weighted examples, not conditional on negative advantages. Entropy is the saved hybrid policy statistic, not a normalized action-count entropy.

Comparison limitations: same initial-model hash, seed 101, roots, learner schedule, loss coefficients and learning rate; RB additionally has a negative-advantage ratio cap of 5, reference refresh every 110 rather than 305 iterations, and weight decay 0.01 rather than 0. The observed differences cannot be attributed solely to Rao–Blackwellization. Raw metrics and appendix retain anomalies; nonfinite gradient norms are not drawn as finite points. Smoothed lines use the trailing 101 iterations and are descriptive, not confidence intervals.

Completion logical_samples is cumulative. RB nominal work is 1,536 × 1,024 = 1,572,864 completion samples per iteration. Effective-hands values describe completion sampling, not model-call counts. Memory peaks can reflect allocator history across a process lifetime. Separate nvidia-smi CSV logs also exist, but lack iteration identifiers and include unrelated GPUs; they are not overlaid on the iteration axis in this report. No training or checkpoint inference was rerun.
"""
    findings = "Final 500-iteration averages: Normal / RB preflop actions per rollout = 4,492 / 5,280; postflop = 1,372 / 451; postflop action share = 23.4% / 7.9%; entropy = 2.257 / 1.675; PPO clip fraction = 1.160% / 0.049%. These ratios describe action mix, not the probability of reaching the flop.\n\nThe main abrupt RB change is near completed iteration 5,487: approximate KL peaks at 9.3568e9; cap activation peaks at 0.514% that same iteration; reference KL peaks at 1.4715 at iteration 5,492. Value loss and preflop action counts step upward and entropy steps downward around this interval. This is an observed coincidence, not a causal diagnosis. The dedicated close-up preserves individual iterations.\n\nRB records 131 iterations containing nonfinite gradients and 134 skipped updates. Normal skips 164 updates but did not save gradient fields. Out of 380,000 attempted updates each, these are 0.0353% (RB) and 0.0432% (Normal). At least one finite gradient was clipped in 3,418 of RB's 9,869 entirely finite-gradient iterations (34.6%); this is not the percentage of individual optimizer steps clipped. The negative-advantage cap activates at least once in 9,540 RB iterations, despite a small posterior-weighted activation fraction."
    readme = "# Normal vs RB training diagnostics through checkpoint 10,000\n\n" + findings + "\n\n" + notes + "\nSources:\n\n" + "\n".join(f"- {label}: `{v['source']}`" for label,v in provenance.items()) + "\n\nOpen `index.html` for the browsable report, or `training_diagnostics_10k.pdf` for all figures. The two `*_all_metrics.csv` files include every scalar and labeled derived columns; `summary.csv` compares the first/last 500 iterations. `validation.json` records source hashes and CUDA checks.\n"
    (out / "README.md").write_text(readme)
    sections = "".join(f'<section id="{slug}"><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p><a href="{slug}.png"><img loading="lazy" src="{slug}.png" alt="{html.escape(title)}"></a></section>' for slug,title,note in pages)
    table = "".join(f'<tr><td>{html.escape(r["metric"])}</td>' + "".join(f'<td>{r[label]["min"]} … {r[label]["max"]}<br><small>{r[label]["finite_count"]} finite; {r[label]["nonfinite_count"]} missing/nonfinite</small></td>' for label in RUNS) + '</tr>' for r in inventory)
    nav = " · ".join(f'<a href="#{slug}">{html.escape(title.split(" | ")[0])}</a>' for slug,title,_ in pages[:9])
    document = '<!doctype html><html><head><meta charset="utf-8"><title>Normal vs RB — 10k training diagnostics</title><style>body{font:16px system-ui;color:#182334;max-width:1450px;margin:32px auto;padding:0 24px;line-height:1.6}img{width:100%}a{color:#245ba5}section{border-top:1px solid #ddd;margin-top:36px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;text-align:left}small{color:#666}nav{padding:16px;background:#f0f5fa}details{margin:20px 0}input{padding:10px;width:60%}</style></head><body><h1>Normal vs RB: training through 10k</h1><p>10,000 rollouts per run · 1,536 hands per rollout · Blue = Normal · Orange = RB (dualclip5)</p><p><a href="training_diagnostics_10k.pdf">All plots (PDF)</a> · <a href="normal_all_metrics.csv">Normal CSV</a> · <a href="rb_all_metrics.csv">RB CSV</a> · <a href="summary.csv">Summary CSV</a> · <a href="validation.json">Provenance and CUDA validation</a></p><nav>' + nav + '</nav><details open><summary>Definitions, missing metrics and comparison limits</summary>' + ''.join('<p>'+html.escape(p)+'</p>' for p in notes.split('\n\n')) + '</details>' + sections + '<h2>Every saved scalar field</h2><p>Constants and duplicate local/global counters are retained here and in the CSV exports.</p><input id="filter" placeholder="Filter metric names"><table><thead><tr><th>Metric</th><th>Normal range</th><th>RB range</th></tr></thead><tbody id="metrics">' + table + '</tbody></table><script>document.getElementById("filter").oninput=function(){for(const row of document.querySelectorAll("#metrics tr")){row.hidden=!row.cells[0].textContent.toLowerCase().includes(this.value.toLowerCase());}};</script></body></html>'
    findings_html = '<h2>Observed changes</h2>' + ''.join('<p>'+html.escape(p)+'</p>' for p in findings.split('\n\n'))
    document = document.replace('<details open>', findings_html + '<details>')
    (out / "index.html").write_text(document)
    print(json.dumps({"output": str(out), "pages": len(pages), "saved_scalar_fields": len(all_keys), "checks": checks}, indent=2))
    for label, d in data.items():
        print(label)
        for key in ["structure.decision_counts_by_street.preflop", "structure.decision_counts_by_street.postflop", "structure.decision_fractions_by_street.postflop", "objective.entropy", "objective.clip_fraction", "objective.approximate_kl", "objective.reference_kl", "objective.negative_advantage_cap_fraction", "derived.nonfinite_gradient", "derived.finite_gradient_clipped"]:
            if key in d:
                a = d[key]
                print(key, "first500", np.nanmean(a[:500]), "last500", np.nanmean(a[-500:]), "max", np.nanmax(a), "max_at", np.nanargmax(a)+1, "nonzero", np.count_nonzero(a > 0))


if __name__ == "__main__":
    main()
