# Diagnosis-led shared Normal/RB parameter search, revision 3

**Status checked October 6:** the campaign launched September 28 and stopped
after 192 completed gradient probes because aggregate CUDA analysis ran out
of memory. Long training and evaluation stages have not started. See the
[current status, findings and remaining-time report](STATUS_20261006.md).

This is the active scientific protocol. It supersedes the broad N/B/steps
factorial in the unlaunched v2 campaign. It uses the same validated v2 execution
engine, normalized rewards, initialization, reference cache, league evaluation,
selective response probes, resume behavior and shared Normal/RB selection.

The central question is which objective coefficients and schedules recover
reliable learning on the repaired pipeline. Workload tuning supports that
question. The previous profile still contained ten loss/schedule settings,
but allocating forty screening hours each to workload and loss was the wrong
emphasis. V3 allocates **16 hours to four workload settings and 76 hours to
nineteen loss/schedule settings**: 83% of screening training time addresses the
objective and schedules.

## Evidence and priorities

The authoritative experiment motivation remains the
[failed-run diagnosis](../../../runs/experiments/nlhe_hu_32bb_rb_training_diagnosis_20260925/README.md).
Its target-assignment defect is repaired in production `0a09496`; the failed
curves cannot establish a hyperparameter-only cause. Its unit comparisons are
still important:

| Parameter | Normalized July-continuity control | Failed August runs |
|---|---:|---:|
| Entropy coefficient | 0.0015625 | 0.025, 16 times larger |
| Reference KL coefficient | 0.00625 | 0.05, 8 times larger |
| Value-loss coefficient | 0.016 | 0.125, 7.8125 times larger |
| Value clip | 0.00625, equivalent to 0.2 bb | 0.2, equivalent to 6.4 bb |
| Median presentations/fresh decision | 1.31 | 1.64 Normal, 1.76 RB |

For the actual massive August runs, roots were 1536, statistical batch 256,
and steps 38 on one GPU. July's mature run used 6144 global roots, global batch
256 and 116 steps on two GPUs. Earlier history notes also describe a different
512-root, 16-step pilot; that pilot must not substitute for the massive-run
comparison above.

July kept its pretrained reference fixed throughout 1110 iterations. August
used frequently refreshed references and different initialization. Entropy
rewards individual raise sizes as separate actions, strongly favoring broad
raising when its coefficient dominates. This supports coefficient, decay,
anchor, critic and learning-rate experiments, not a large independent sweep
of reuse.

## A small workload search near fixed reuse

Let $W=3$, $N$ be roots per rank, $B$ the statistical batch per rank, $S$ the
synchronized optimizer updates and $D$ the actual global fresh trained player
decisions in that rollout. Realized mean reuse is

$$E=\frac{WBS}{D}.$$

All four candidates keep $BS/N=6$. At the measured checkpoint-1110 trajectory
length near four decisions per root, this gives about 1.5 presentations per
fresh decision. Roots are not decisions; this is approximate reuse matching,
not a claim that $BS/N$ itself counts epochs.

| Name | N/rank | B/rank | Steps | Global roots | Global presentations |
|---|---:|---:|---:|---:|---:|
| control | 512 | 128 | 24 | 1536 | 9216 |
| n1024_b128_s48 | 1024 | 128 | 48 | 3072 | 18432 |
| n1024_b256_s24 | 1024 | 256 | 24 | 3072 | 18432 |
| n2048_b256_s48 | 2048 | 256 | 48 | 6144 | 36864 |

The middle two settings separate batch size from update count at the same
fresh-root and presentation budgets. The remaining contrast changes fresh-data
width while preserving presentations per root. Each setting gets two hours
in each mode; all share the same coefficients and schedules during this stage.

Policy-dependent trajectory lengths change actual $E$. Compute it from each
iteration's `metrics.global_player_nodes` and the increment in
`clock.global_presentations`, then inspect its distribution and postflop share.
Do not count posterior breadth or completion samples in this denominator. No
hidden per-mode step adjustment is made. A large reuse drift is evidence to
investigate trajectory-length or policy collapse, not silently compensate away.

## Main coefficient and schedule experiments

The selected workload is held fixed across the following **19 settings**,
each trained for two hours in Normal and two hours in RB. Rows are single
changes from the shared normalized control; confirmation separately tests
compatible combinations. The numerical brackets explore neighborhoods around
the historical unit conversion, not claims of literature-derived optima.

| Arm | Changed parameter | Value |
|---|---|---:|
| control | none | shared normalized historical control |
| entropy_zero | initial entropy coefficient | 0 |
| entropy_half | initial entropy coefficient | 0.00078125 |
| entropy_double | initial entropy coefficient | 0.003125 |
| entropy_fast | exposure at exactly zero entropy | 125000 |
| entropy_slow | exposure at exactly zero entropy | 1000000 |
| kl_weak | reference KL coefficient | 0.003125 |
| kl_strong | reference KL coefficient | 0.0125 |
| reference_fast | reference-copy exposure interval | 100000 |
| reference_fixed | reference-copy exposure interval | never refresh |
| learning_rate_slow | AdamW learning rate | 0.000025 |
| learning_rate_fast | AdamW learning rate | 0.0001 |
| critic_weight_quarter | value-loss coefficient | 0.004 |
| critic_weight_4x | value-loss coefficient | 0.064 |
| critic_unclipped | value loss | MSE, coefficient 0.016 |
| critic_clip_wide | value clipping threshold | 0.025, equivalent to 0.8 bb |
| ppo_clip_tight | policy-ratio clip | 0.1 |
| lambda_09 | return/trace lambda | 0.9 |
| posterior_clock_raw | entropy/reference scheduling clock | raw posterior exp(H) |

Unchanged controls: entropy starts at 0.0015625 and reaches exactly zero at
250000 exposure units; reference interval 400000; KL 0.00625; LR 0.00005;
value weight 0.016 and clip 0.00625; PPO clip 0.2; lambda 0.95. Default exposure
is fresh global player decisions. The raw-posterior arm uses the previously
defined current-public posterior breadth for RB and one per Normal decision.
It remains an explicit scheduling hypothesis with no arbitrary exponent or
independence claim. Threshold overshoot is retained in the clock logs.

A further slow finite reference interval beyond the screening exposure would
largely duplicate the fixed-anchor arm, so the bracket is fast/control/fixed.
Entropy's long-decay arm restores the earlier brainstorm and still has an
exact finite zero endpoint. The 19-arm set retains every v2 loss/schedule arm.

Adam betas, epsilon, weight decay and global gradient clip remain at the
historical control; the diagnosis offers stronger failure-specific evidence
for the axes above. Architecture and legal-action definitions stay fixed for
checkpoint compatibility. Splitting action-type and raise-size entropy would
change the objective and belongs in a separate design experiment. RB completion
samples remain 1024, separately from the public-posterior scheduling quantity.
All fixed values are explicit in the profile; they have not been declared
universally optimal.

## Selection and budget

The frozen-gradient diagnostic remains: 192 local one-step probes with paired
seeds across N/B and estimator. It diagnoses variance and mean agreement; it
neither sets a clock multiplier nor claims to capture the all-reduced three-GPU
gradient. See the retained [scheduling analysis](../hyperparameter_campaign_v2_20260928/SCHEDULING.md).

Selection uses balanced randomized cross-play and one uniform frozen opponent
pool per stage, equally weighting Normal and RB for each shared configuration.
Matrices, cycles, paired score intervals, and raw games against unchanged
checkpoint 1110 are retained. Stage-local league scores cannot be compared
numerically across stages without a common anchor. The learned-response audit
is restricted to control versus raw posterior scheduling, with both modes and
a common held-out cross-attack panel.

| Stage | Nominal wall hours using three GPUs |
|---|---:|
| Four workload settings, both modes, 2 h | 16 |
| Nineteen coefficient/schedule settings, both modes, 2 h | 76 |
| Three distinct configurations, two fresh seeds, both modes, 3 h | 36 |
| Two finalists, early-policy initializer, both modes, 4 h | 16 |
| Selective BR training | 4 |
| Total training plus BR | 148 |

Gradient probes, initialization and evaluation are additional. The registered
leagues contain 558592 games and the fixed-anchor panels 80896 games, plus
selective BR evaluation. Allow approximately a week, potentially 7–8 days with
evaluation and iteration-boundary overhead. This is not a time guarantee.

Confirmation retains the original control, workload-only improvement and best
compatible workload-plus-loss combination, deduplicating identical settings.
The full cross-product of workloads and loss settings is not trained. If the
results suggest workload/objective interactions, the next focused experiment
should cross the leading workload and loss settings. The final recommendation
remains provisional, with training-seed and population-relative limitations.

## Launch and resume

```bash
cd /home/pablob/ai-game-learning
bash experiments/hyperparameter_search/run_shared_loss_search_v3.sh
```

Profile: `configs/profiles/nlhe_hu_32bb_shared_loss_search_v3.yaml`.
Output: `runs/experiments/nlhe_hu_32bb_rb_shared_loss_search_v3_20260928`.
The supervisor checks idle GPU ownership before each wave; all three GPUs are
allocated to each training job. Reference cache is 1 GiB; large-root settings
use six-row physical learner packs, identically in both modes. Thermal
telemetry is recorded without cooling pauses.

The initializer remains checkpoint 1110 because checkpoint 1100 is absent;
the legacy input adapter and four `/32` value-head conversions are retained.
The transfer stage starts from its recovered historical reference policy.
The final profile is saved as `recommended_profile.json` in the output and
`configs/profiles/nlhe_hu_32bb_shared_league_selected_v2.json`: that filename
belongs to the unchanged v2 engine, while its contents are selected from this
v3 scientific profile. No optimum has been measured before the full campaign.

In the next conversation, start with this file, `manifest.json`, `report.json`,
`confirmation_candidates.json`, `noise/analysis/result.json`, the stage league
and anchor reports, and `clock_audit/report.json`. All per-condition specs,
checkpoints, intermediate snapshots, per-rank metrics and GPU telemetry remain
available. The detailed artifact map in the previous
[handoff](../hyperparameter_campaign_v2_20260928/HANDOFF.md#pick-up-the-results-in-a-new-conversation)
still applies with the v3 output path.

## Revision validation and current state

At the time of the September 28 validation below, the full v3 campaign was
**prepared, not launched**. For its subsequent launch and failure, see the
[October 6 status report](STATUS_20261006.md). The old v2 shell shorthand
also forwards to v3; its old immutable manifest is preserved as historical.

- All **94 campaign tests passed with CUDA enabled**, including production
  capacity compilation for all 152 workload/loss/mode combinations and checks
  that the scientific controls match between Normal and RB.
- Four additional bounded three-GPU checks exercised combined lower and upper
  coefficient settings in both modes, with the actual new entropy horizons,
  value clipping, learning rate, PPO clipping and lambda controls. Every case
  applied both requested steps with finite gradients and identical rank hashes.
  They are execution checks, not registered quality trials.
- The execution source is unchanged from the completed 114-job v2 integration
  smoke, including league, anchor, response, confirmation, transfer and resume
  checks. Source hash:
  `34ab8614f148cf5e636bea2902df13cd717e0bc694daf16e292c7ce2bc61b662`.
- New measurements and exact validation jobs are under
  `runs/experiments/nlhe_hu_32bb_rb_loss_search_v3_validation/`;
  `validation_summary.json` records their results and the prepared budget.
- All three GPUs returned to their idle display-only memory baseline.

The inherited capacity measurements and cache comparison are in
[the execution validation record](../hyperparameter_campaign_v2_20260928/VALIDATION.md).
No multi-day quality experiment or calibrated posterior multiplier has been
measured during this revision.
