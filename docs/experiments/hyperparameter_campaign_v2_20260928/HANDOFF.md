> Superseded by [the diagnosis-led v3 study](../hyperparameter_loss_search_v3_20260928/README.md): four approximately matched-reuse workload settings and nineteen loss/schedule settings. The v2 execution engine and its validation remain in use.

# Campaign v2 handoff

This is the active v2 protocol. It supersedes the v1 profile and all v1
launch commands. The v2 campaign is a paired shared-parameter Normal/RB study
and has no selected winner until the staged workload, loss, confirmation, cold,
and response measurements finish. Do not launch the old v1 output.

## Active command

From the repository root, the convenience command sets the required venv and
CUDA toolkit environment and resumes the immutable prepared experiment:

```bash
bash experiments/hyperparameter_search/run_shared_league_v2.sh
```

Run it in your usual persistent terminal session. The expanded command is below.

The prepared experiment uses this expanded command:

~~~bash
cd /home/pablob/ai-game-learning
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
python -m experiments.hyperparameter_search.league_campaign \
  --profile configs/profiles/nlhe_hu_32bb_shared_league_v2.yaml \
  --output runs/experiments/nlhe_hu_32bb_rb_shared_league_v2_20260928 \
  --gpus 0,1,2 \
  --resume
~~~

The supervisor checks GPU ownership before every job wave and refuses occupied
GPUs. The multi-day campaign has not been launched in this implementation session.

## Before launch

1. Verify that the v2 profile, resolved manifest, budget plan, and source
   hashes name the v2 output and checkpoint 1110. The old
   `nlhe_hu_32bb_hyperparameter_campaign_v1.yaml` and
   `nlhe_hu_32bb_rb_shared_campaign_20260928` output are historical.
2. Confirm the eight primary workload cells use per-rank
   $(N,B,S)\in\{512,1024\}\times\{128,256\}\times\{24,48\}$ and that
   Normal/RB receive the same shared scientific parameters.
3. Both $N=2048$ bridges are included after capacity checks. They use learner
   pack 6 in both modes; all cells use a 1 GiB reference cache. See
   [VALIDATION.md](VALIDATION.md) for the measured scope and remaining limits.
4. Confirm there is no square-root posterior arm. The raw posterior clock is a
   named scheduling condition; the covariance helper is diagnostic code, not a
   calibration result.
5. Confirm the 16-replicate gradient-noise stage is marked policy-only,
   first-step, frozen-policy measurement. Do not use its timing as a
   learning-speed estimate or as evidence that $S$ produces fresh data.

## Stage order and budgets

After the frozen-gradient diagnostic, run the workload screen: ten settings,
both modes, two hours per condition. Lock the selected workload only
after comparing wall-time progress and actual exposure counters.

Run the ten loss settings at that selected workload, both modes, two hours per
condition. Then run confirmation on three distinct configurations, two fresh
seeds, both modes, three hours per condition. The configurations are the
original control, strongest workload-only setting, and strongest compatible
workload-plus-loss setting. Deduplicate identical configurations by promoting
the next compatible alternate.

Run two cold configurations, both modes, one fresh seed, four hours per
condition. The nominal training-plus-BR total is 136 wall hours for the ten
workload cells and all subsequent stages. This already includes four BR wall
hours in three-GPU waves; gradient-noise and evaluation overhead are
separately ledgered.

The fixed-loss BR regression compares the control against the raw
posterior-clock condition for targets from both modes, with three one-hour
restarts per target. Retain all restart artifacts and report the selected
restart; a weak response is not evidence of a strong target. The launcher also
runs a common four-response by four-target cross-attack panel on fresh held-out deals. Do not replace the fixed BR protocol with
candidate training overrides.

## Evaluation and statistics

Use the same uniform common opponent pool for every candidate, round-robin
match order, balanced deals and seats, shuffled match order, and complete
matrix cycles. Report raw bb/hand. Use 512 hands per screen cell, 2,048 per
confirmation cell, and 8,192 per cold cell. These cycles measure the declared
common pool; they do not solve Nash or provide an exploitability certificate.

Select the shared configuration by the equally weighted Normal/RB mean. Keep
per-mode scores, mode gaps, and all rankings provisional. League confidence intervals use the existing cluster-aware standard-error
estimator over paired logical deals. Gradient variance-ratio intervals bootstrap
paired independent seed batches. Neither treats alternate private hands or
reused learner rows as independent observations. Report uncertainty for the shared mean, mode difference, and
round-robin contrasts. The covariance helper is not a substitute for the
16-replicate frozen-policy gradient measurement.

Track, for every job, wall seconds, roots, fresh player decisions, retained
rows, global presentations, applied optimizer steps, reuse, completion ESS,
posterior breadth, entropy-zero time, reference refresh count/age, skipped or
nonfinite updates, and thermal/memory telemetry. Never call completion ESS or
posterior breadth independent fresh hands without the corresponding declared
estimand.

## Resume and review

Use the same v2 command and GPU topology to resume. Review the manifest,
budget plan, commands, per-condition status, and all source/profile hashes
before accepting a stage. The final provisional shared training profile is also exported to
`configs/profiles/nlhe_hu_32bb_shared_league_selected_v2.json`; the launcher
refuses to overwrite a different existing recommendation. A missing BR or evaluation is an incomplete result,
not a zero attack gain. A condition that changes profile, mode, GPU topology,
or wall-time budget is a new experiment and requires a new reviewed output.

The exact historical workload comparison is in [HISTORY.md](HISTORY.md), and
the variance/clock estimands are in [SCHEDULING.md](SCHEDULING.md).

## Pick up the results in a new conversation

Start with this file and the active output directory:
`runs/experiments/nlhe_hu_32bb_rb_shared_league_v2_20260928`.
The launcher writes a final `report.json` only after every required stage
finishes. `recommended_profile.json` and the matching `configs/profiles/`
export are provisional; no quality winner has been measured at preparation.

Review these artifacts before recommending the next long training run:

- `manifest.json`, `budget_plan.json`, `commands.jsonl`: exact profile,
  initialization hash, live-source hash, GPU assignments and executed commands.
- `noise/analysis/result.json`: first-step gradient variance and mean agreement;
  small ratios or broad intervals are not a posterior-hand calibration.
- `league/{workload,regularizer,confirm,cold}/result.json`: uniform-pool scores,
  per-mode scores, RB-minus-Normal intervals, paired arm contrasts, payoff
  matrices, cycles and frozen population identities. Scores are stage-local.
- `anchor/*/report.json`: raw cross-play against the unchanged checkpoint-1110
  policy. Check whether apparent league improvements are actual regressions
  against this fixed player, especially after cold transfer.
- `clock_audit/report.json` and `clock_audit_panel/`: matched response restarts
  and the held-out common attack panel for decision-clock versus raw breadth.
  This is a sensitivity audit, not a Nash certificate or an automatic override
  of league selection. Investigate disagreements before endorsing that clock.
- `confirmation_candidates.json`: the original control, workload-only and
  compatible workload-plus-loss settings, including deduplication replacements.
- Per-condition `job.json`, `metrics.rank_*.jsonl`, `gpu.rank_*.csv`,
  `latest.json`, `checkpoints/`, and `snapshots/`: exact settings, retained-row
  reuse, elapsed budgets, gradient/clip behavior, entropy-zero crossing,
  posterior versus completion counters, threshold overshoot, reference copies,
  memory/thermal behavior, and intermediate models for additional evaluation.

Fresh training seeds and larger final evaluation budgets reduce selection
noise; two confirmation seeds and one cold seed do not establish a publication
level confidence claim. If the top shared configurations remain close, retain
the intervals and add paired seeds or held-out games rather than declaring the
point-estimate leader definitively superior. The full N/B/S/loss screen is a
registered search range, not a proof of a global optimum.

Earlier literature, historical coefficient mapping and distributed semantics
are in `docs/experiments/hyperparameter_search_20260928/`. V1 implementation
validation remains historical; the active v2 CUDA record is
[VALIDATION.md](VALIDATION.md).
