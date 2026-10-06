> Superseded by [the diagnosis-led v3 study](../hyperparameter_loss_search_v3_20260928/README.md): four approximately matched-reuse workload settings and nineteen loss/schedule settings. The v2 execution engine and its validation remain in use.

# Shared Normal/RB league campaign v2

This campaign estimates a shared workload and loss profile for sampled
(Normal) and Rao–Blackwell (RB) training. Normal and RB are paired estimator
conditions: they use the same scientific parameters, initialization, seed,
GPU allocation, wall-time budget, and evaluation pool. The campaign selects a
shared mean across the two modes; it does not select an estimator.

The shared standalone control is `configs/profiles/nlhe_hu_32bb_shared_control_v2.yaml`;
the eventual provisional selection is exported to
`configs/profiles/nlhe_hu_32bb_shared_league_selected_v2.json`. The v2 campaign profile is
`configs/profiles/nlhe_hu_32bb_shared_league_v2.yaml`, and the intended output
is `runs/experiments/nlhe_hu_32bb_rb_shared_league_v2_20260928`. The v1
profile and command are superseded. Do not launch
`nlhe_hu_32bb_hyperparameter_campaign_v1.yaml` or its old output directory.

The game and continuation remain HU full-deck NLHE with 32 bb equal stacks,
0.5/1 bb blinds, no rake or ante, sequential rollout, checkpoint 1110, the
historical preflop input adapter, and the exact `/32` value-head migration.
Normalized objective controls remain shared across both modes: value
coefficient 0.016, value clip 0.00625, entropy start 0.0015625, reference KL
0.00625, AdamW learning rate $5\times10^{-5}$, weight decay 0.01, and global
gradient clipping 0.5. These are continuity controls, not a claim of global
optimality.

## Workload factorial

The primary workload screen is an eight-cell joint factorial on per-rank roots
$N$, learner batch $B$, and synchronized optimizer steps $S$:

| $N$/rank | $B$/rank | $S$ | global roots | global batch | global presentations/iteration |
|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 24 | 1,536 | 384 | 9,216 |
| 512 | 128 | 48 | 1,536 | 384 | 18,432 |
| 512 | 256 | 24 | 1,536 | 768 | 18,432 |
| 512 | 256 | 48 | 1,536 | 768 | 36,864 |
| 1,024 | 128 | 24 | 3,072 | 384 | 9,216 |
| 1,024 | 128 | 48 | 3,072 | 384 | 18,432 |
| 1,024 | 256 | 24 | 3,072 | 768 | 18,432 |
| 1,024 | 256 | 48 | 3,072 | 768 | 36,864 |

At fixed $B,S$, changing $N$ measures fresh public exposure against reuse. At
fixed $S$, changing $B$ changes per-update gradient noise and total learner
presentations. At fixed $B$, changing $S$ changes optimizer update count and
reuse. The matched-presentation pair $(B,S)=(128,48)$ versus $(256,24)$ both
has 18,432 global presentations and is the primary batch-versus-update
contrast. Reuse is measured as presentations divided by retained fresh
decision rows; roots alone are not a reuse estimate.

Two additional bridge settings approach the July fresh-data width:

| $N$/rank | $B$/rank | $S$ | global roots | global presentations/iteration |
|---:|---:|---:|---:|---:|
| 2,048 | 128 | 96 | 6,144 | 36,864 |
| 2,048 | 256 | 48 | 6,144 | 36,864 |

Both bridge cells use physical learner packs of six rows, in both modes;
the eight smaller cells use twelve. Logical $B$ and $S$ are unchanged by this
execution setting. All cells use a 1 GiB reference-cache admission budget,
rollout packs of 48 and completion packs of 512. The bridge pack change follows
an actual RB OOM at pack 12; these execution settings and their validation
limits are recorded in [VALIDATION.md](VALIDATION.md). Ten workload settings
are registered. The cache is a bounded admission budget, not an allocation of
1 GiB in every iteration.

All workload cells run for two hours per mode. Compare progress at equal wall
time and report fresh decisions, retained rows, learner presentations,
optimizer steps, reference copies, throughput, and per-cell reuse. Keep the
learning rate fixed when changing $S$; do not turn update-count effects into
learning-rate effects.

## Gradient-noise diagnostic

Before interpreting workload speed, run 16 independent frozen-initializer seed
replicates for each $N/B$ pair in $N\in\{512,1024,2048\}$ and
$B\in\{128,256\}$, in both Normal and RB: 192 probes. Each probe uses one
GPU and measures the local mean gradient. This measures a per-rank estimator;
it does not capture the actual three-GPU all-reduced gradient. Measure the policy-only
gradient on the first learner step after AMP unscaling and before clipping or
Adam. This is a gradient-variance diagnostic, not a learning-speed or
wall-clock exposure estimate; it does not use $S$ as evidence of training
progress. Preserve independent public seeds and report full-vector variance,
mean-gradient agreement, direction, and timing separately. The covariance
helper is available, but it is not an observed statistic until these
replicates run.

The [scheduling note](SCHEDULING.md) defines the estimand, nested public versus
completion randomness, paired bootstrap, and the distinction between
completion ESS and posterior breadth. The v2 profile has no square-root
posterior clock. Use the raw posterior clock only as the explicitly named
clock candidate; no exponent or correlation discount is admitted without a
measured calibration.

## Loss screen and staged selection

After workload selection, test ten registered loss settings at the selected
$N/B/S$ workload, for two hours in each mode. Normal and RB receive identical
shared scientific overrides for every loss setting. The loss screen is paired
and does not reselect workload independently by estimator.

The confirmation stage contains three distinct configurations: the original
control, the strongest workload-only configuration, and the strongest
compatible workload-plus-loss configuration. Run each with two fresh seeds,
both modes, and three hours per condition. If a candidate is duplicated by
the selection rules, fill the slot with the next compatible alternate rather
than running duplicate work.

Cold stage: two selected shared configurations, two modes, one fresh seed, four
hours per condition. This uses the recovered early reference policy from the
historical checkpoint, not random initialization. Treat rankings as provisional
and attach uncertainty conditional on the trained policies. Select by the equally weighted Normal/RB shared mean;
retain mode-specific scores, gaps, and disagreement in the report.

## Evaluation and response tests

The primary evaluation uses one uniform common opponent pool in round-robin
order. Use raw bb/hand, balanced deals and seats, shuffled match order, and
full matrix cycles. A cycle is a measurement block; it is not a Nash solve and
does not claim equilibrium. Use 512 hands per screen cell, 2,048 per
confirmation cell, and 8,192 per cold cell. The larger cold budget is intended
to improve confidence in close shared rankings.

Scores are stage-local because each stage has its own frozen opponent pool;
do not compare their numerical league scores across stages. A separate raw
cross-play panel tests every candidate against the unchanged checkpoint-1110
policy on held-out deals. That fixed anchor detects shared degradation and is
reported separately from automatic league selection.

At the regularizer stage, selective BR tests compare the fixed-loss control against
the raw posterior-clock condition, for targets from both Normal and RB. Use
three one-hour restarts per target in three-GPU waves: four nominal wall
hours, plus initialization and evaluation. BR keeps its fixed response protocol and does
not inherit candidate training overrides. A common cross-attack panel evaluates each of the four selected responses
against all four targets on fresh held-out deals. This exposes response-training
failure and cross-target disagreement; its results are reviewed separately
from automatic league promotion.

A weak best-response restart does not establish that its target is weak. Keep
restart values, selected restart identity, common-pool scores, and cross-panel
disagreement. No response score is true exploitability, and no finite panel
is a Nash certificate.

## Nominal budget

With ten workload cells, the nominal training and response budget is:

| Stage | Calculation | Wall hours |
|---|---:|---:|
| Workload | $10\times2$ modes $\times2$ h | 40 |
| Loss | $10\times2$ modes $\times2$ h | 40 |
| Confirmation | $3\times2$ seeds $\times2$ modes $\times3$ h | 36 |
| Cold | $2\times2$ modes $\times4$ h | 16 |
| BR regression | three one-hour restarts in three-GPU waves | 4 |
| **Total** | | **136** |

The 136 nominal hours already include selective BR training. Gradient-noise
measurements, initialization, league and fixed-anchor evaluation are additional.
There are 378,880 league games and 77,824 anchor games, plus selective BR
evaluation. Allow roughly 6–7 days overall; iteration-boundary budget overshoot
and policy-dependent evaluation speed make this an estimate, not a deadline.

## Reporting requirements

Report raw bb/hand and normalized metrics, paired mode differences, fresh
decisions, retained rows, learner presentations, optimizer updates, actual
reference copies, entropy-zero crossing, temperatures/thermal flags, skipped
updates, and wall-clock throughput. Use common held-out pools and paired
logical-deal confidence intervals.
The league uses the existing cluster-aware standard-error estimator; the
gradient variance ratio uses a paired bootstrap over independent seed batches.
League intervals are conditional on the trained models and do not substitute
for uncertainty across training seeds. A shared mean is a selection rule, not evidence that Normal and RB have identical objectives or
variance. Keep every ranking provisional when uncertainty intervals overlap.

The workload history and exact July/failed-run comparison are in
[HISTORY.md](HISTORY.md). The operational handoff, preparation command, and
resume rules are in [HANDOFF.md](HANDOFF.md).
