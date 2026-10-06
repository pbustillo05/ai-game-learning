"""Render small exported CUDA summaries; no model computation occurs here."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from experiments.training_diagnosis.probe_policies import OUT


def main():
    data = json.loads((OUT / "policy_probes.json").read_text())
    colors = {"old": "#386cb0", "normal": "#f59f00", "rb": "#c92a2a"}
    names = {"old": "July RB", "normal": "August Normal", "rb": "August RB"}
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for run in colors:
        rows = sorted((r for r in data if r["run"] == run), key=lambda r:r["iteration"])
        x = [r["iteration"] for r in rows]
        axes[0].plot(x, [100*r["root"]["raise"] for r in rows], "o-", label=names[run], color=colors[run], markersize=4)
        axes[1].plot(x, [max(r["root"]["raise_kl_to_uniform"], 1e-8) for r in rows], "o-", color=colors[run], markersize=4)
    axes[0].axhline(100*61/63, color="gray", linestyle=":", label="Entropy-only optimum (96.8%)")
    axes[0].set(ylabel="Unopened SB raise probability (%)", ylim=(-2,102))
    axes[1].set(ylabel="Conditional raise-size KL to uniform", yscale="log")
    for ax in axes:
        ax.set(xlabel="Completed training iterations", xscale="log")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("The collapse occurs early; raise sizing becomes nearly uniform")
    figures.append(("behavior_timeline",fig))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    palette = np.array([[.22,.42,.68],[.25,.72,.43],[.94,.40,.19]])
    for ax,run,iteration in zip(axes,("old","normal","rb"),(1110,10000,10000)):
        row=next(r for r in data if r["run"]==run and r["iteration"]==iteration)
        type_p=np.array(row["root"]["type_probabilities_by_class"]).reshape(13,13,3)
        ax.imshow(type_p@palette,vmin=0,vmax=1)
        ax.set(xticks=range(13),yticks=range(13),xticklabels=list("AKQJT98765432"),yticklabels=list("AKQJT98765432"),title=f"{names[run]}, iteration {iteration:,}\nRaise {100*row['root']['raise']:.2f}%")
        for i in range(13):
            for j in range(13):
                ranks="AKQJT98765432"
                label=ranks[min(i,j)]+ranks[max(i,j)]+("" if i==j else "s" if i<j else "o")
                ax.text(j,i,label,ha="center",va="center",fontsize=6,color="white")
    fig.legend(handles=[Patch(color=palette[i],label=label) for i,label in enumerate(("Fold","Call","Raise"))],loc="outside lower center",ncols=3)
    fig.suptitle("Root action ranges — historical input encoding restored for July")
    figures.append(("root_ranges",fig))

    files=sorted(OUT.glob("continuation_normal_i50_*.json"))
    if files:
        fig, axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for file in files:
            d=json.loads(file.read_text())
            coeff=d["arguments"]["entropy"]
            axes[0].plot([0]+list(range(1,len(d["iterations"])+1)),[100*d["before"]["raise"]]+[100*r["root"]["raise"] for r in d["iterations"]],"o-",label=f"Entropy coefficient {coeff:g}")
            if d["attribution"]:
                norms=d["attribution"]["weighted_gradient_norms"]
        labels=("PPO", "Value", "Reference KL", "Entropy\ncurrent", "Entropy\nold-equivalent")
        vals=[norms["policy"],norms["value"],norms["reference_kl"],norms["entropy"],norms["entropy"]/16]
        axes[1].bar(labels,vals,color=["#386cb0","#777777","#9467bd","#c92a2a","#2ca02c"])
        axes[0].set(xlabel="Short continuation iterations (512 roots, 12 updates each)",ylabel="Unopened SB raise probability (%)",ylim=(0,100))
        axes[0].legend()
        axes[1].set(ylabel="Component gradient norm (first 1,024 learner rows)")
        for ax in axes:ax.grid(alpha=.2,axis="y")
        fig.suptitle("Original target mapping: same checkpoint, optimizer and seed; only entropy changes")
        figures.append(("entropy_ablation",fig))

    rb_files = sorted(OUT.glob("continuation_rb_i50_*.json"))
    if len(rb_files) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for file in rb_files:
            d = json.loads(file.read_text())
            coeff = d["arguments"]["entropy"]
            rows = [d["before"]] + [r["root"] for r in d["iterations"]]
            x = list(range(len(rows)))
            axes[0].plot(x, [100*r["raise"] for r in rows], "o-", label=f"Entropy {coeff:g}")
            axes[1].plot(x, [r["raise_kl_to_uniform"] for r in rows], "o-", label=f"Entropy {coeff:g}")
        axes[0].set(ylabel="Unopened SB raise probability (%)", ylim=(0, 100))
        axes[1].set(ylabel="Conditional raise-size KL to uniform")
        for ax in axes:
            ax.set(xlabel="Short continuation iterations (512 roots, 12 updates each)")
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle("Original target mapping, early RB checkpoint: entropy also changes behavior")
        figures.append(("rb_entropy_ablation", fig))

    original_file = OUT / "ratio_audit_rb_i1100_seed918273.json"
    corrected_file = OUT / "ratio_audit_rb_i1100_seed918273_aligned.json"
    if original_file.exists() and corrected_file.exists():
        original = json.loads(original_file.read_text())["iteration_means"]
        corrected = json.loads(corrected_file.read_text())["iteration_means"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        labels = ["Original mapping", "Corrected mapping"]
        vals = [d["recorded_action_approximate_kl"] for d in (original, corrected)]
        axes[0].bar(labels, vals, color=["#c92a2a", "#2ca02c"], bottom=1e-6)
        axes[0].set(yscale="log", ylabel="Recorded-action approximate KL (log scale)", ylim=(1e-6, 1e10))
        for i, value in enumerate(vals):
            axes[0].text(i, value*2.3, f"{value:,.0f}" if value > 1 else f"{value:.8f}", ha="center")
        vals = [100*d["posterior_mass_recomputed_old_below_floor"] for d in (original, corrected)]
        axes[1].bar(labels, vals, color=["#c92a2a", "#2ca02c"])
        axes[1].set(ylabel="Posterior mass on old action probability below 1e-12 (%)", ylim=(0, 10))
        for i, value in enumerate(vals): axes[1].text(i, value+.2, f"{value:.4f}%", ha="center")
        for ax in axes: ax.grid(alpha=.2, axis="y")
        fig.suptitle("Same RB checkpoint and four updates: only completion-row assignment changes")
        figures.insert(0, ("target_alignment", fig))

    with PdfPages(OUT / "diagnosis_figures.pdf") as pdf:
        for name,fig in figures:
            fig.savefig(OUT / f"{name}.png",dpi=160)
            pdf.savefig(fig)
            plt.close(fig)


if __name__=="__main__":main()
