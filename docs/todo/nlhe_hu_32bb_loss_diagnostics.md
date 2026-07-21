# NLHE HU 32 bb loss correction and continuation plan

## Goal and evidence

This plan follows the completed sequential-RB run in
`runs/nlhe_hu_32bb_full_sequential_rb`. It continues the learned policy while
correcting loss units, entropy scheduling, value weighting, KL-reference
cadence, and telemetry before another long run.

The old loss plot is rank-local and accumulated over 115--116 optimizer steps,
not a global per-step mean. Reconstructing the two-rank objective over
iterations 300--1109 gives mean per-step terms of PPO `+0.0425`, value
`+0.0171`, entropy `-0.0418`, and reference KL `+0.0773`. Loss magnitude is not
gradient magnitude, but the auxiliary terms were not small perturbations.

Other warnings were near-universal final-step gradient clipping and extreme
`ratio_max`. PPO loss oscillation alone is not evidence of divergence because
each iteration changes the rollout, behavior policy, opponent, and advantages.

## 1. Normalize poker utility once

For stack size $S$, train on

$$
r'_p = r_p/S.
$$

This bounds utility by $[-1,P-1]$, hence $[-1,1]$ in this heads-up run. Apply
the conversion at the training boundary to fold and showdown rewards alike, so
lambda returns, value targets, old values, and advantages all share the same
units. Keep raw-big-blind payoff for evaluation and human-facing reports.

Do not independently normalize value targets. Do not add advantage whitening
in the first continuation: bounded utility is sufficient for the controlled
comparison, and weighted RB advantage normalization is a separate experiment.

Record the utility convention and scale in checkpoints. Incompatible resumes
must fail rather than silently mix units.

### Continue the same policy safely

Start a new run with checkpoint 1110 as a model initializer, not a resume:

1. retain every trunk and policy parameter exactly;
2. divide both preflop and postflop `value_head.affine` weights and biases by
   32, preserving their predictions in normalized units;
3. initialize a fresh optimizer because old Adam moments use old loss units;
4. initialize the KL reference from the migrated current policy; and
5. record source-checkpoint identity and the deterministic migration.

Test that policy probabilities are bitwise unchanged and migrated values equal
old values divided by 32 within fp32 tolerance.

## 2. Retune value and entropy

The unit-converted continuity value coefficient is
$32(0.0005)=0.016$. It preserves the old component balance approximately; it
is a control, not an assumed optimum. Compare

$$
c_v\in\{0.016,0.05,0.1,0.25,0.5\}.
$$

Select using value RMSE/explained variance, policy/value gradient norms,
gradient clipping, and strategy evaluation. A conventional `0.5` is not
automatically correct without advantage standardization.

Entropy must reach zero after 500 **new training iterations**, equal to about
58,000 optimizer steps at the present geometry. Add an explicit
`entropy_decay_iters: 500`; do not tie decay to the total run horizon or the
37-iteration checkpoint-writing interval. Compare initial

$$
c_h\in\{0.05/32,0.01,0.05\}
$$

with the same linear-to-zero schedule. The `0.05` arm tests the literature-like
coefficient deliberately; it is not unit-equivalent to the old run.

## 3. Measure posterior effective hands

For every sampled sequential-RB decision $d$, obtain its exact-hand posterior
$\widehat P_d(h)$ from `SequentialPosteriorPool`. Require finite nonnegative
entries and positive total mass, then define

$$
q_d(h)=\frac{\widehat P_d(h)}{\sum_{h'}\widehat P_d(h')}
$$

and the entropy effective number of hands

$$
N_{\mathrm{eff}}(d)=\exp\!\left(-\sum_{h:q_d(h)>0}
q_d(h)\log q_d(h)\right).
$$

Thus a point posterior has $N_{\mathrm{eff}}=1$, a uniform posterior on $k$
hands has $N_{\mathrm{eff}}=k$, and always
$1\le N_{\mathrm{eff}}\le1326$ for this game.

Compute $N_{\mathrm{eff}}$ once per posterior-pool row after target
construction. For rank $r$, gradient step $t$, and its partition minibatches
$B_{r,t,m}$, compute

$$
E_{r,t}=\sum_m\sum_{d\in B_{r,t,m}}N_{\mathrm{eff}}(d),\qquad
E_{\mathrm{iter}}=\sum_r\sum_t E_{r,t}.
$$

Count repeated with-replacement samples every time: this metric measures
effective-hand **learner presentations**, matching the requested exposure, not
unique hands. Physical subspan splitting must not change it.

Report at least:

- `posterior_effective_hand_presentations`: rank-local iteration sum;
- `posterior_effective_hand_presentations_rank_sum`: global
  $E_{\mathrm{iter}}$;
- `posterior_effective_hands_per_sample`: global sum divided by global sampled
  statistical rows;
- step-level mean, minimum, and maximum of $E_{r,t}$ after global reduction;
- global sampled statistical rows; and
- a cumulative effective-hand counter since the current KL reference.

Also report a companion sum over each fresh retained public decision exactly
once. It separates newly collected RB information from optimizer reuse.

This is an information-volume proxy, not an independent-sample count: hands
within one posterior share a public state, and decisions can share a deduplicated
trajectory/player posterior. Preserve public-node and optimizer-step counters
alongside it.

## 4. Calibrate the KL reference by exposure

Keep `c_kl=0.2` as the requested primary arm, but measure its gradient norm and
include a lower-coefficient control because NashPG also standardizes advantages.
Reference strength and refresh cadence must be interpreted jointly.

NashPG has one sampled private hand per turn-based transition. Its repository
default exposes nominally 4.096M fresh player nodes and, after four PPO epochs,
16.384M learner presentations before changing reference. The reported paper
experiment uses 40.96M fresh nodes and 163.84M presentations. Valid-mask rows
make the exact effective counts slightly smaller.

Add an optional `kl_ref_effective_hand_presentations` threshold. After an
iteration, globally add $E_{\mathrm{iter}}$. When the threshold is reached,
refresh from the just-updated current policy before the next rollout and reset
the counter. Persist the counter, threshold, and reference identity in every
checkpoint. Log the anchor used for the iteration, threshold overshoot, and
whether a refresh occurred afterward.

First compare thresholds of 16.384M and 163.84M effective-hand presentations.
Because RB hands are correlated, also retain fixed 37- and 138-iteration arms;
do not assume an entropy-effective hand equals an independent NashPG sample.
Report realized iterations, public nodes, gradient steps, and effective hands
per reference period for every arm.

## 5. Repair core telemetry

Make all loss plots use global rank sums divided by optimizer-step count. Add:

- posterior-weighted clip fraction and approximate behavior-policy KL;
- weighted log-ratio median, p90, p99, and maximum;
- weighted advantage mean, standard deviation, RMS, and sign balance;
- normalized value-target RMS, value RMSE, and explained variance;
- raw policy entropy and reference KL, split by street where practical;
- pre-clip gradient norm for every step, not only the last; and
- periodic separate gradient norms for PPO, value, entropy, and reference KL.

## 6. Tests and staged experiments

1. **CPU utility tests:** reward bounds, zero-sum conservation, fold/showdown
   agreement, target scaling, checkpoint metadata, and value-head migration.
2. **Effective-hand unit tests:** point mass gives 1, uniform $k$ gives $k$,
   zeros are ignored, positive rescaling is invariant, malformed zero-mass or
   nonfinite rows fail, and known posteriors match fp64 references.
3. **Aggregation tests:** repeated sampled rows count repeatedly; partition and
   physical-subspan decompositions give the same sum; two-rank reduction is
   exact; checkpoint/resume preserves the cadence counter with no off-by-one
   refresh.
4. **Tiny GPU smoke:** after checking for an idle GPU, verify finite normalized
   targets/gradients, unchanged policy, migrated values, metric bounds, KL reset,
   deterministic continuation, and clean GPU release.
5. **Measurement pilot:** run 10--20 iterations from migrated checkpoint 1110
   with a fixed reference. Measure effective hands before choosing thresholds.
6. **Loss-scale pilot:** branch from the same migrated weights across the value
   and entropy candidates above, holding `c_kl=0.2` and rollout work fixed.
   Include one lower-KL control. Reject configurations dominated by clipping or
   one component's gradient.
7. **Cadence pilot:** compare effective-hand thresholds 16.384M and 163.84M
   against fixed intervals 37 and 138 at equal global public nodes and wall
   time. Confirm the expected KL reset/growth pattern.
8. **Continuation:** select by paired arena and approximate-BR checkpoint
   results, not training loss. Confirm finalists with at least three seeds and
   report uncertainty, throughput, all exposure counters, value quality,
   entropy, behavior KL, clip fraction, and component gradient norms.

No massive continuation should begin until the migration, global metrics, and
effective-hand counter pass the CPU and tiny-GPU gates.
