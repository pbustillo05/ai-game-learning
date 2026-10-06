> Superseded by [the paired league v2 campaign](../hyperparameter_campaign_v2_20260928/README.md). Use its profile and launch command; this v1 protocol is historical.

# Campaign handoff

This search selects shared Normal/RB parameters. The bounded CUDA validation
is complete; see the [validation ledger](README.md) and
[machine-readable evidence](../../../runs/experiments/nlhe_hu_32bb_rb_campaign_validation/validation_summary.json).
The full campaign is **prepared, not started**. Its source/profile/spec hashes
are frozen in `runs/experiments/nlhe_hu_32bb_rb_shared_campaign_20260928`.
Use the run/resume command below. Expected duration is roughly six days:
133 training/BR hours plus measured evaluation overhead, with uncertain policy-
dependent trajectory lengths. No hyperparameter winner has been selected.

## Environment and GPU ownership

~~~bash
cd /home/pablob/ai-game-learning
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi
~~~

Use only idle GPUs. The three-GPU examples require authorization to use all
three and all three must be available. The launcher accepts one/two GPUs,
running BR/evaluation restarts in waves with a longer elapsed budget.
It refuses occupied GPUs and never clears someone else's workload.
These documents launch no long training job. Thermal policy is telemetry-only.

## Prepare and inspect

The named directory was already prepared during this session. This command
revalidates it without a CUDA context or GPU query; omit `--resume` only when
preparing a new output directory:

~~~bash
python -m experiments.hyperparameter_search.campaign \
  --profile configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml \
  --output runs/experiments/nlhe_hu_32bb_rb_shared_campaign_20260928 \
  --gpus 0,1,2 \
  --prepare-only --resume
~~~

Review 'manifest.json', 'budget_plan.json', and both
'screen/control/seed_918273/sampled/job.json' and
'screen/control/seed_918273/rao_blackwell/job.json'.
Thirteen paired settings take 133 nominal training hours on three GPUs,
plus evaluations and initialization. The final common-attack panel adds
16 evaluations of 512 requested paired hands (8,192 hands), with no training. Read the generated budget if the
registered profile changes.

Preparation freezes source hashes. Finish source edits before preparing the
real output. Changes afterward trigger rejection; do not edit manifests to
bypass it. Use a new reviewed output/version when intentionally changing code.

## End-to-end smoke

Use a separate output:

~~~bash
python -m experiments.hyperparameter_search.campaign \
  --profile configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml \
  --output runs/experiments/nlhe_hu_32bb_rb_shared_campaign_smoke_20260928 \
  --gpus 0,1,2 \
  --smoke
~~~

Smoke tests control in both modes, one self-play iteration, one iteration per
BR restart, then four-hand evaluations. It skips confirmation and cold stages.
A decision-clock control smoke does not automatically test posterior-clock
arms; their bounded CUDA tests/specs must also be recorded.

For smoke resume, retain '--smoke' and add '--resume'. Changing mode, GPU list,
profile, duration scale, or source hashes does not define the same campaign.

## Run or resume

After preparation and validation:

~~~bash
python -m experiments.hyperparameter_search.campaign \
  --profile configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml \
  --output runs/experiments/nlhe_hu_32bb_rb_shared_campaign_20260928 \
  --gpus 0,1,2 \
  --resume
~~~

Use an owned persistent terminal/session. A file lock rejects a second
launcher on the same output. 'commands.jsonl' preserves child commands,
working directory, UUIDs and spec hashes. Per-job directories contain specs,
logs, statuses, native checkpoints, external-clock state, charged-time
ledgers, and inference snapshots.

SIGINT/SIGTERM forwards to owned process groups, allowing checkpointing.
The launcher allows a bounded grace period, then may terminate a hung owned
group. Repeat the same command with '--resume': only verified explicit
complete outputs are skipped; unfinished workers resume without resetting
charged training time. The saved topology/UUID assignment must remain valid.
An occupied GPU blocks the next launch; investigate ownership instead of
killing arbitrary processes.

'--duration-scale' defines a separate abbreviated experiment with scaled
budgets. It does not make the abbreviated result a full registered run.
Retain its exact value on resume.

## Standalone Normal/RB training

Use the experiment worker, never the ordinary production CLI. One idle GPU:

~~~bash
CUDA_VISIBLE_DEVICES=0 \
  python -m experiments.hyperparameter_search.worker \
  --profile configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml \
  --mode normal \
  --output runs/experiments/nlhe_hu_32bb_normal_shared_control_standalone \
  --hours 2
~~~

RB on three idle GPUs:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=3 \
  -m experiments.hyperparameter_search.worker \
  --profile configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml \
  --mode rb \
  --output runs/experiments/nlhe_hu_32bb_rb_shared_control_standalone \
  --hours 2
~~~

Standalone uses the control arm. Another candidate requires a registered
job specification or a recorded derived profile. Worker resume still works,
but standalone does not supply the campaign's output lock, manifest-level
source freeze, promotion, or three-restart evaluations. Prefer campaign mode
for comparisons. Repeat the same standalone command to resume; changing
hours changes its specification rather than extending a budget implicitly.

## Interpret artifacts

- 'promotion.json': point ordering, selected shared settings, per-mode scores,
  restart values and uncertainty flags.
- 'combination_candidate.json', when present: deterministic merge of the
  strongest screened change and first nonconflicting next change. Confirmation
  tests control, the strongest single change and this combination without
  increasing the three-setting budget; conflicting paths fall back to the
  original shortlist.
- Per-seed 'paired_score.json': equal-weight mode score and mode gap.
- 'cross_panel/manifest.json': frozen selected response pool and fresh test split.
- 'cross_panel/matrix.json' and 'cross_panel/result.json': all final cross-attacks,
  shared point recommendation, uncertainty and own-response ranking disagreement.
- 'report.json': explicit completion and provisional recommendation.
- 'recommended_profile.json': concrete selected training/control values and
  'estimator_selected: false'.
- 'recommended_training_sampled.json' and
  'recommended_training_rao_blackwell.json': paired standalone profile wrappers
  including initialization, training and external controls. Launch them through
  the experiment worker, not the ordinary production CLI.
- 'gpu.rank_*.csv': utilization, memory, temperature, power, clocks and thermal
  slowdown flags on each device.

Screening and confirmation pool each mode's worst achieved response gain
across three BR restarts. After cold trials, the final cross-panel freezes one
selected attacker per target and evaluates all four attackers against all four
targets on fresh held-out deals. Its mode-pooled point winner becomes the
shared provisional recommendation, with earlier-ranking disagreement recorded.
None of these scores is true exploitability. Control is retained at screening;
later stages do not automatically select it when confidence intervals overlap.
A point winner remains provisional, and opposing per-mode effects should be
reported rather than hidden by their average.

After GPU work, run 'nvidia-smi' to verify owned training/monitor processes
ended and selected devices returned to idle. Leave other users' jobs untouched.

## What the next conversation should do

1. Read this file and README, then the real campaign's `report.json`,
   `promotion.json`, `cross_panel/result.json`, and `commands.jsonl`.
   If incomplete, inspect `status.json` and the failing process log; do not
   treat a missing BR/evaluation as zero attack gain.
2. Compare the shared winner with control **separately in each mode**, then
   inspect paired seeds, own-BR versus common-panel disagreement, entropy
   zero crossings, actual reference copies, skipped/nonfinite updates, and
   progress versus seconds, fresh decisions, roots, and learner presentations.
3. Treat averaged/max-selected uncertainty envelopes as descriptive. The
   learned finite attack pool does not upper-bound exploitability. Confirm
   a surprising winner with longer BR budgets and fresh seeds/deals before
   paper claims or calling parameters optimal.
4. Inspect raw/square-root clock behavior and boundary overshoot. Completion
   sampler ESS in native metrics is **not** the new posterior clock. Exact
   public-history marginals include opponent card-removal weights; terminal
   learner posteriors are a different quantity. Correlation calibration is
   still unmeasured, even though its analysis helper exists.
5. Use `recommended_profile.json` with the experiment worker for either
   mode only after reviewing the provisional recommendation. The two exported
   training profiles include external clock controls; ordinary `pbg.cli.train`
   does not implement these controls. Keep shared scientific parameters.
6. The requested checkpoint1100 was absent; all initial continuation jobs use
   checkpoint1110 with the recorded SHA. Initialization-transfer jobs use its
   recovered reference initializer, not a newly randomized scratch policy.
