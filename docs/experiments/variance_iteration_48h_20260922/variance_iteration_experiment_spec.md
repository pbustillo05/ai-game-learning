# Gradient and iteration experiments — strict 48-hour protocol

Deliver analyzed results and figures **within 48 wall-clock hours of receiving this instruction**, including implementation, validation, execution, and analysis. Reuse compatible completed work and schedule the remaining measurements within this deadline.

## 1. Experiments

| Experiment | Starting checkpoints | Arms at each checkpoint | Default size |
|---|---|---|---|
| Raw actor-gradient variance | Normal 10k and RB 10k | Sampled and RB-1024 | 32 independent paired repetitions |
| One-iteration optimization | Normal 10k and RB 10k | Sampled and RB-1024 | Six independent paired rollout batches; up to 100 optimizer steps |

Apply **both estimators to each identical starting checkpoint**. Default training batch: $N=384$ public rollouts and $B=256$ decision records. The iteration study is 24 short runs and at most 2,400 optimizer steps.

The scope comprises these two experiments, their runtime measurements, and the analysis described below.

## 2. Deadline and fixed fallback

- **Hours 0–3:** inspect existing work; perform minimum correctness checks and one representative paired timing pilot. Reuse validated implementation and simple job orchestration.
- **By hour 6:** lock definitions, artifact hashes, seeds, device assignments, repetitions, and the run queue. Report any correctness/implementation blocker immediately; simplify engineering without weakening the measurement.
- **By hour 12:** save the first verified paired measurements and a preliminary result table, explicitly labeled partial.
- **By hour 24:** publish an interim `RESULTS.md` and available figures, then continue the predeclared queue.
- **By hour 40:** finish measurements and close the measurement queue.
- **Hours 40–48:** complete uncertainty estimates, plots, reproducibility information, and cleanup. Deliver by hour 48.

Choose scope from timing and available GPU-hours before inspecting which method wins. Default: 32 gradient repetitions and six iteration repetitions per checkpoint, step 38 primary and step 100 secondary. If forecast completion exceeds hour 40, first cap every iteration run at **38 steps**; if still too large, reduce gradient repetitions to **16 per checkpoint**. Preserve six paired iteration repetitions and both final checkpoints. Record this smaller protocol before the main run.

If the smaller protocol exceeds available resources, report the concrete blocker and deliver verified partial results by the deadline. Retain every finished paired block and report actual sample sizes and interruption reasons.

## 3. Repository and server handling

Work in `/home/pablob/ai-game-learning` on `nova.iems.northwestern.edu` using the existing authorized connection. Read current `AGENTS.md` and relevant nested instructions. Inspect dirty files, current jobs, and exact checkpoint paths before edits. Follow CUDA-only production requirements and use `/home/pablob/.venvs/poker`; verify its shared CUDA toolkit under `lib/python3.10/site-packages/nvidia/cu13` if needed.

Use available GPUs with explicit device assignments. Preserve unrelated jobs and active source versions. Treat `docs/theory/` and historical artifacts as read-only. Use an isolated experiment module/worktree where appropriate. Put new results under `runs/experiments/nlhe_hu_32bb_rb_variance_iteration_48h_<UTC>/` and configs under `configs/experiments/`.

Reuse completed compatible measurements with provenance. Group results by their loss definitions, normalization, precision, and sampling contracts. Handle credentials through the existing secure connection.

## 4. Experiment A: direct raw-gradient variance

At each frozen checkpoint use $\lambda=1$ Monte Carlo targets, an unclipped actor loss, a detached fixed baseline, and identical fixed advantage scaling in both arms. Exclude entropy/reference penalties from the primary actor-gradient quantity. Declare the loss and trajectory/decision normalization. This controlled measurement is distinct from the production PPO iteration below.

For repetition $r$, generate fresh public rollouts and decision indices, then share them between Sampled and RB-1024. Refresh rollout and estimator randomness across repetitions. Preserve each arm's marginal distribution and joint card-compatible posterior. The sampled arm uses the ordinary sampled private-state estimator implemented in the learner.

Measure full actor gradient vectors **before clipping or optimizer stepping**, correctly unscaled if using mixed precision. Estimate

$$
\widehat V=\frac1{R-1}\sum_{r=1}^R\|g_r-\bar g\|^2,
\qquad \bar g=\frac1R\sum_r g_r.
$$

Use stable accumulation and retain enough per-repetition data to recompute variance and paired confidence intervals. Save bounded gradient snapshots or a verified sufficient representation that supports recomputation and bootstrap uncertainty.

Primary output: RB/Sampled actor-variance ratio at each checkpoint with uncertainty. Also save mean-gradient differences, mean norms, timings, and failures. Critic-gradient diagnostics are included only if already supported cheaply. Interpret gradient-mean differences relative to the declared estimand and their sampling uncertainty.

Define any auxiliary moments explicitly: node-square sum $Q=\sum_\nu\|\mu_\nu\|^2$ and squared trajectory sum $\|\sum_\nu\mu_\nu\|^2$. Compute the primary variance result directly from repeated gradient vectors.

## 5. Experiment B: one optimization iteration

At each checkpoint generate six independent training rollout batches, each defining a paired Sampled/RB-1024 comparison. Before each arm restore identical weights, Adam moments and step counters, reference model, scheduler/scaler state, old action probabilities, and caches. Start each repetition from the saved checkpoint. Prefer original training checkpoints; if optimizer state is unavailable, use the same fresh state in both arms and label the study accordingly.

Use identical minibatch-index schedules and the same learning rate, PPO/value coefficients, clipping, regularization, and normalization within each pair. Freeze or independently calibrate common advantage-normalization constants for the controlled comparison; document any departure from production behavior. Production-mode $\lambda$ and loss settings remain fixed across arms.

**Respect target-cache behavior.** If production caches targets for an iteration, cache each arm's own targets once and reuse them. Keep old policy/reference inputs and target-producing baseline quantities frozen as required by the declared learner.

Evaluate steps **0, 10, 20, 38**, and **100** when the timing-selected protocol includes it. Step 38 is primary; verify the saved configuration's optimizer-step count and document any discrepancy. Report the predeclared endpoints and the complete measured curve.

### One common held-out objective

Create an independent held-out public-rollout pool per checkpoint, shared by all candidates and steps. Start with 256 public trajectories and an independently seeded 4,096-completion reference calculation. Check precision on a small subset using independent draws and a doubled completion budget. Increase reference accuracy only within the fixed time cap if necessary. Use separate reference streams and common targets across candidates.

The primary metric is expected ordinary sampled-target clipped PPO actor loss, with fixed old-policy advantage construction and normalization shared by all candidates:

$$
L_{\mathrm{ref}}(\theta)=\mathbb E[-\min\{\rho_\theta A_0,
\operatorname{clip}(\rho_\theta,1-\epsilon,1+\epsilon)A_0\}].
$$

Condition on the actor hand before applying its action ratio, then average with the correct posterior weights. To cache this expected clipped objective efficiently for fixed actor-hand ratio $\rho$, let $c=\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)$ and use

$$
\mathbb E[\min(\rho A,cA)]
=\min(\rho,c)\mathbb E[A_+]+\max(\rho,c)\mathbb E[A_-],
$$

where $A_+=\max(A,0)$ and $A_-=\min(A,0)$. Verify the cache against direct averaging of sampled clipped losses. This computes the declared objective efficiently from cached signed moments.

Report improvement $L_{\mathrm{ref}}(\theta_0)-L_{\mathrm{ref}}(\theta_s)$, so larger is better. Report actor improvement as the primary metric and critic loss separately when supported. Reserve reference data for evaluation of the predeclared settings.

Save training loss, elapsed time including target construction, and available verified KL/clipping/gradient-norm diagnostics. Prioritize the core measurements. Label arm-specific training losses as diagnostics and use the shared held-out objective for comparisons.

## 6. Minimum focused checks

Before scaling verify:

1. Identical checkpoint/optimizer restoration and target reuse in a short paired run.
2. Distinct train/reference/repetition streams and correct within-pair data sharing.
3. Joint private-card compatibility and the documented loss/normalization contract.
4. Gradient measurement occurs before optimizer updates; streaming variance matches direct calculation on a tiny retained sample. Check node-square versus trajectory-square accounting wherever used.
5. Cached common PPO reference loss agrees with direct averaging, including advantages of both signs; policy inputs at each decision contain only information available at that decision.
6. A representative full paired job, bounded resume, and cleanup of its own processes.

Reuse verified tests and add focused checks protecting these measurements. Resolve required-check failures before scaling.

## 7. Statistics and required output

Bootstrap whole paired repetition IDs for gradient ratios and whole paired rollout-batch IDs for iteration improvement, preserving both arms and all time points. Six independent iteration batches determine the replication count and yield preliminary uncertainty. Report intervals conditional on the shared evaluation pool and separately quantify reference sampling/completion uncertainty. If that error prevents distinguishing methods, report the comparison as unresolved.

Count target construction in timing. Plot progress against steps and actual GPU time, and describe the compute comparison as measured progress under the declared settings. Use comparable hardware or separate device-specific timings. Preserve regressions, skipped updates, divergences, and infrastructure failures with reasons.

Deliver by hour 48:

- `RESULTS.md`: actual sample sizes, gradient variance ratios, paired step-38/100 improvements where available, intervals, time/memory use, failures, and concise paper-ready conclusions.
- Three figures with PNG and SVG/PDF plus data: actor-gradient variance ratios; held-out actor improvement versus optimizer steps; the same improvement versus elapsed compute time.
- Manifest/configs/seeds, source/diff and checkpoint identities, raw repetition measurements, focused check outputs, reference-precision evidence, and reproduce/resume commands.

Use simple durable orchestration and atomic outputs. Track launched, completed, failed, and analyzed jobs explicitly. Deliver the available verified measurements and analysis within 48 hours.
