# Scheduling v2: measured gradient noise, coverage, and shared comparison rules

The v2 study removes the unsupported square-root posterior clock. There is no
chosen entropy exponent, assumed correlation factor, or measured RB acceleration
factor in this note. The useful candidates are explicit clocks with different
units; a variance-calibrated clock becomes eligible only after its calibration
has actually been performed and validated.

This note distinguishes an implemented first-step gradient audit from proposed
follow-up measurements. The audit has not established a calibrated clock:
CUDA numerical tests and the reduced two-seed probe/analysis smoke pass.
The registered 192-probe measurements remain pending until the user launches
the full campaign; see [VALIDATION.md](VALIDATION.md). No authoritative theory
is modified.
Earlier v1 square-root settings and discussion are historical, not an endorsed
v2 default.

## What the existing mathematics does and does not establish

The local decision law weights roots by their realized number of player
decisions and then selects a decision uniformly. It is not an equal-weight
average of per-root mean losses. See
[core_math.tex](../../../docs/theory/core_math.tex), the ideal player-decision
law around lines 168–211, terminal empirical law around 373–393, and RB
pre-terminal law around 600–621.

The ideal RB identity reduces variance by taking conditional expectations.
However, the same document explicitly notes at lines 645–646 that substituting
conditional targets into a loss nonlinear in those targets defines a new
surrogate, rather than necessarily the conditional expectation of the original
loss. The fixed-target response discussion around 810–827 also distinguishes
the pooled-decision PPO surrogate from an unbiased gradient of game payoff.

Consequently “RB has a smaller gradient variance” and “RB learns the same
objective faster” are different claims. Mean-gradient agreement, sampling law,
critic/return approximation, clipping, and normalization must accompany the
variance comparison. Policy-gradient control-variate work similarly treats
variance of the actual gradient estimator, not just dispersion of returns.
[Greensmith, Bartlett and Baxter, JMLR 2004](https://www.jmlr.org/papers/v5/greensmith04a.html)

## Posterior breadth is a coverage quantity

At each fresh decision, use the actor's current-public-history marginal over
exact compatible private combinations. For HU:
$$
p_i(h)\propto
\rho_i(h)\sum_{g:g\cap h=\varnothing}\rho_{-i}(g),
$$
with both hands respecting known board cards. Normalizing actor reach alone
misses opponent-range card-removal information.

Define:
$$
K_i=\exp H(p_i),\qquad
H(p_i)=-\sum_h p_i(h)\log p_i(h).
$$
The raw scheduling proposal counts $K_i$ for RB and one for Normal. A breadth
of 37 thus advances this clock by 37 Normal decision units. This expresses a
coverage-based scheduling hypothesis; it does not assert equivalence to 37
independent trajectory samples.

The exact current-history marginal is distinct from the downstream
completion-conditioned posterior entering a learner target. Preflop target
conversion can irreversibly collapse combinations into 169 classes. A uniform
lift of that downstream class posterior does not recover exact entropy.
The public-history observer's own reconstruction assumptions and blocker
correction must be tested independently.

The implementation point is
[exposure.py](../../../experiments/hyperparameter_search/exposure.py):
the sequential observer runs before action update and next-family compaction.
Its raw breadth and decision counts are diagnostics, not independent-sample
certificates. Completion-world ESS remains separate; increasing completion
samples must not mechanically advance the coverage clock.

The fact that $K_i$ is large does not determine whether per-hand gradient
contributions are similar, anticorrelated, almost zero, or dominated by one
public decision. Replacing $K_i$ by its square root does not solve this:
without a measured model for gradient covariance, that replacement merely
invents another schedule. It is removed from the v2 candidate set.

## Implemented: first-step policy-only audit at a fixed initializer

The executable audit is
[gradient_noise.py](../../../experiments/hyperparameter_search/gradient_noise.py),
registered under `gradient_noise` in
[nlhe_hu_32bb_shared_league_v2.yaml](../../../configs/profiles/nlhe_hu_32bb_shared_league_v2.yaml).
It runs **one CUDA GPU/process per probe, with world size one**. Its $N$ roots
and $B$ statistical learner rows are **per-rank budgets**:

| Setting | Registered values |
|---|---|
| Roots $N$ per probe/rank | 512, 1024, 2048 |
| Learner batch $B$ per probe/rank | 128, 256 |
| Estimators | Normal and RB |
| Independent replicate seeds per setting/mode | 16 |
| Learner steps per probe | Exactly one, including smoke mode |
| Total full-grid probes | $3\times2\times2\times16=192$ |
| Paired bootstrap resamples | 1000 |

Several probes can occupy different GPUs concurrently, but each produces a
**local per-rank gradient**. This is not a measurement of the actual
all-reduced three-GPU learner gradient. At training world size three, matching
per-rank $N,B$ corresponds to three times those aggregate root/row budgets;
cross-rank dependence, collective averaging, and their actual noise are not
measured here. Inferring the variance of a three-rank average as one third of
local variance would require additional independence and weighting assumptions.

Every seed starts a fresh session from the same declared checkpoint model
(or declared reference-model initializer), with the same historical input
adapters and normalized value-head migration. Initialization hashes and dense
parameter layouts must agree across draws. The retained baseline/return
convention is part of the PG estimator; zero critic-loss weight does not
remove the critic's role in constructing targets.

The implementation sets entropy, reference-KL, and value-loss coefficients to
zero after ordinary configuration and smoke resolution. It captures the full
dense actor-objective gradient after removing AMP scale 1024, **before**
gradient clipping, Adam, and weight decay. Parameters used only by value
prediction remain in the recorded layout with their corresponding zero PG
entries. The optimizer then performs one update, which must be finite and
applied exactly once; that updated model is discarded. Thus “fixed policy”
means identical **pre-update initialization across probes**, not that the
optimizer step is disabled or that a persistent model changes between seeds.

The current artifact is the first-step pooled-decision learner estimator,
including its finite completion and learner-row randomness. It is not an
explicit per-root contribution decomposition. One independently seeded public
trajectory batch is the replicate unit. The analysis verifies matching
Normal/RB seed sets and compatible configurations; common seed numbers do not
guarantee identical trajectories, private hands, actions, or completion draws
after the estimator paths diverge.

Implemented CUDA analysis computes full-vector means, trace sample covariance,
mean norms, SNR, cosine and mean-difference/noise diagnostics, and a paired
bootstrap interval for the Normal/RB variance ratio. Each bootstrap draw uses
the same replicate-index multiplicities for both modes. Degenerate draws are
counted and excluded from the ratio interval. Sixteen original batches do not
become a large independent sample merely because 1000 resamples are generated.

Probe outputs include the actual resolved configuration, gradient tensor,
source/specification/initializer/artifact hashes, training duration, optimizer
metrics, and observed posterior breadth. Analysis saves mean vectors and
descriptive JSON results. Numerical tests exist in
[test_hyperparameter_gradient_noise.py](../../../tests/test_hyperparameter_gradient_noise.py);
all 22 tests in that module passed on CUDA during v2 validation.
These are implemented capabilities, not a report of a measured Normal/RB gain.
**No result automatically changes a clock, multiplier, or exposure exponent.**

## Proposed extensions: distributed and component-level calibration

The following measurements are additional work, not capabilities completed by
the first-step audit above.

A distributed calibration should choose a fixed **global** root budget and
global statistical minibatch size, record the corresponding per-rank values
and world size, and capture the actual globally averaged gradient. Match the
normalized initializer, adapters, baseline/return convention, sampling weights,
and reduction rule between modes. Keep the same pre-update policy and optimizer
moments for every replicate; report wall time and total GPU-seconds separately.

Two complementary views would then be useful:

1. **Like-for-like actor estimator.** Extend the implemented policy-only view
   to the declared distributed sampling/reduction law and additional fixed
   policies. At the on-policy starting point, ordinary PPO ratio clipping is
   locally inactive; this does not establish equivalence after further updates.
2. **Operational learner estimator.** Separately capture actor, critic, entropy,
   and KL gradients, and their joint full-loss gradient, at explicitly fixed
   evaluation/rollout parameters. This component decomposition and full-loss
   audit are not currently implemented. They would show how regularizer
   dominance and target nonlinearities affect the gradient seen by the optimizer.

Capture these quantities after AMP unscaling and the declared global mean,
before clipping and Adam. An Adam-preconditioned update-noise comparison would
be another separate measurement requiring matched fixed optimizer moments.
Use a pilot to assess whether substantially more independent batches are needed
for usable uncertainty; 16 replicates is an initial design choice, not evidence
that uncertainty has been resolved.

## Interpretation of the implemented full-vector statistics

For independent replicate gradients $g_{m,r}$, mode $m$, and $R$ replicates:
$$
\bar g_m=\frac1R\sum_{r=1}^{R}g_{m,r},
\qquad
T_m=\frac1{R-1}\sum_{r=1}^{R}\|g_{m,r}-\bar g_m\|_2^2
    =\operatorname{tr}\widehat{\operatorname{Cov}}(g_m).
$$
No dense parameter-by-parameter covariance matrix is needed to estimate its
trace. Compute reductions on CUDA; stream sums if storing all full vectors
is impractical. If using projected gradients, fix and disclose the projection
and distinguish projected-trace estimates from exact full-gradient values.

The local variance ratio is:
$$
\widehat G_{\mathrm{var}}=\frac{T_{\mathrm{Normal}}}{T_{\mathrm{RB}}}.
$$
Only when the estimators target sufficiently similar means and variance scales
approximately inversely with independent data can this be interpreted as a
local equivalent-data gain. Test that scaling at more than one $N$ or $B$.
A near-zero denominator gives an unstable ratio, not evidence of unlimited
sample efficiency.

The current analysis reports point estimates and paired mean-difference/noise
diagnostics; its bootstrap interval is for the variance ratio. It does not
provide formal confidence intervals for mean agreement, cosine, or SNR, nor
an automatic near-zero-signal significance classification. The following
reporting goals distinguish those existing diagnostics from extensions needed
to support stronger conclusions:

- Mean-gradient norms, norm of their difference, and uncertainty in that
  difference. Repeatedly lower variance around a different mean can indicate
  a different surrogate or bias.
- Cosine/directional agreement, with an explicit undefined/low-signal label
  when a mean norm is near zero.
- Gradient signal-to-noise, for example $\|\bar g_m\|^2/T_m$, with finite-replicate
  uncertainty and the upward bias of squared sample means acknowledged.
  $T_m/R$ is the trace contribution of sampling noise to that squared mean;
  do not interpret a noisy or negative debiased estimate as a precise signal.
- As a future extension, per-component/layer/street results alongside the
  implemented full-vector trace. The trace depends on parameter coordinates
  and can hide important directions; these decompositions are not emitted by
  the current audit.
- Time per replicate. A useful local cost comparison is
  $\widehat G_{\mathrm{var}}\,t_{\mathrm{Normal}}/t_{\mathrm{RB}}$ under the same
  inverse-sample-variance assumptions; it is not an observed learning-speed
  result and should be checked against actual training progress.

## Proposed extension: separate trajectory, completion, and row-selection noise

Roots/public trajectories are the independent sampling units. Decisions along
one trajectory, alternate private hands at one public state, and minibatch
reuse of the same retained data are not independent replicates.

The implemented probes redraw a whole independently seeded batch and measure
the resulting combined learner-gradient noise. They do not hold one public
batch fixed, expose independent completion-seed controls, or store root-level
gradient contributions. The following nested experiment is proposed work.

For each frozen public batch, repeat completion work with independent completion
seeds while holding learner-row selection fixed. This nested experiment measures
conditional completion noise. In the outer experiment, collect independently
seeded public batches; optionally repeat learner-row sampling inside each batch
to estimate minibatch-selection noise. Keep seed namespaces distinct and
preserve the actual conditional sampling laws.

A paired Normal/RB comparison can use common exogenous seeds/deals to improve
precision where the coupling preserves each estimator's marginal law. Do not
force identical trajectories by conditioning away the private or action
randomness whose variance is under study. Document where the coupled policies
diverge and use independent root clusters across outer replicates.

The implemented analysis bootstraps whole independent batch-gradient replicates.
A future nested or root-resolved analysis should bootstrap independent batches
or public-root clusters, preserving all their decisions and nested completion
repeats. For paired comparisons, resample
paired clusters jointly. Recompute both numerator and denominator of the
variance ratio within each bootstrap replicate. Do not bootstrap individual
private-hand contributions as if they were independent samples. Report
intervals and unstable-ratio frequency; very few outer clusters do not become
reliable merely by generating many bootstrap resamples. The wider RL evaluation
literature motivates uncertainty-aware reporting, but the cluster structure
here is specific to our trajectory sampler.
[Agarwal et al., NeurIPS 2021](https://agarwl.github.io/rliable/)

Respect the actual pooled-decision objective. For a root cluster $j$ with
gradient-contribution sum $S_j$ and decision count $D_j$, the full-data
estimator is proportional to $\sum_j S_j/\sum_j D_j$, not generally
$\frac1N\sum_j S_j/D_j$. Bootstrap/resampling must recompute that ratio, or
resample the exact retained-row sampler with its weights. Report roots and
decisions separately when policy-dependent trajectory lengths differ.

The existing 'gradient_covariance_diagnostic' in exposure.py measures
$\operatorname{trCov}(\sum_h c_h)/\sum_h\operatorname{trCov}(c_h)$ for repeated,
posterior-weighted per-combination contributions. That diagnoses covariance
among RB components. It is **not** the Normal/RB ratio above and does not,
by itself, turn posterior breadth into independent hands. A metadata tag
identifying that computation is not proof of a calibrated scheduling gain.

## Candidate clocks and what a fair comparison means

| Clock | Counts | Useful interpretation | Main limitation |
|---|---|---|---|
| Fresh global decisions | One per selected fresh public decision, both modes | Matched public-data schedule | Correlated decisions; different trajectory lengths and throughput |
| Raw posterior breadth | $K_i$ per RB decision, one per Normal decision | Explicit coverage hypothesis | Not independent samples; potentially very fast RB schedule |
| Learner presentations | Every global row presentation, including reuse | Matched optimization-data consumption | Repeated rows add no fresh public information |
| Optimizer updates | One synchronized applied update | Matched number of parameter updates | Batch size, learning rate, clipping, and noise can differ |
| Measured variance-equivalent exposure | A separately validated local variance gain | Empirical estimator-precision calibration | Policy/budget/component dependent; not available before measurement |

Apply each rule consistently to both modes; do not optimize a hidden
mode-specific hyperparameter set. Keep entropy horizon and reference interval
independent and require entropy to reach exactly zero. For every clock, log
all other clocks as secondary axes. Do not conflate equal wall-time experiments
with equal decisions or equal optimizer work.

A posterior-clock test with the **same raw numerical horizon** as the decision
control is an intentionally aggressive schedule test. It preserves the user's
meaning that breadth 37 advances 37 units. Label it that way rather than
claiming any performance change is caused only by a superior exposure measure.

Threshold resolution is especially important. If one completed rollout already
crosses the whole entropy horizon or many reference thresholds, the implementation
can only apply the resulting schedule at its actual update boundaries. Record
exposure per iteration, time/updates to entropy zero, reference ages, and
overshoot. Repeatedly copying the same endpoint model cannot simulate inner
optimization against several intermediate references. Reject comparisons whose
intended intervals are unresolved at the chosen rollout granularity, or
explicitly test that boundary-limited aggressive behavior as its own condition.

A separate **rate-matched diagnostic** can distinguish clock shape from its
initial scale: collect an independent frozen-policy calibration batch and
measure each clock's initial exposure rate. Choose horizons to align the
initial expected event time with a preregistered baseline time, then compare
how the schedule changes as posterior breadth evolves. Alternatively match
initial decision exposure rather than wall time, and name that distinction.

This diagnostic is not the original equal-raw-hands hypothesis. Per-mode rate
matching cancels its initial RB speedup; it cannot simultaneously preserve
“37 versus one” and equalize initial event times. A single common rescaling
for both modes preserves their relative acceleration and cannot equalize both
rates. Keep raw-aggressive and rate-matched results separate, freeze calibration
before quality outcomes, report all derived horizons, and never silently
replace the shared primary comparison with separately tuned mode horizons.

If a measured clock is later admitted, publish the fixed policy, root/batch/
completion budgets, weighting, gradients included, mean-agreement tests,
uncertainty, and empirical inverse-data scaling that justify its multiplier.
Recalibrate or mark extrapolation at later policies. No arbitrary exponent or
preselected correlation discount substitutes for those measurements. Until
then, the measured clock is a proposed follow-up, not an implemented justified
default or an established learning-rate correction.

## Practical strength and equilibrium claims

Cross-play against a fixed, balanced opponent league is a valid measure of
practical strength against that league. Balance seats, blinds, deals and
opponent weights; use the same pool and held-out seeds for every candidate.
Include historical policies, meaningful strong anchors and common trained
attackers, not a pool dominated by easily beaten weak agents.

Poker strategies can exhibit cycles: beating one predecessor does not imply
beating another, and a single scalar rating can conceal those matchups.
Inspect the payoff matrix, mode-specific results and robust aggregate metrics.
Nontransitivity is a central issue in population evaluation of zero-sum games.
[Balduzzi et al., ICML 2019](https://arxiv.org/abs/1901.08106)

A trained response's achieved gain is evidence of an exploitable weakness;
a weak or unsuccessful response is not proof of low exploitability.
A common response pool helps expose failures of target-specific response
training. Its finite maximum still does not certify a best response or Nash
equilibrium. Even an equilibrium mixture of the measured league is only an
equilibrium of that restricted empirical game.

Use these evaluations to make a concrete provisional choice of shared
parameters, while reserving claims of general improvement for independent
held-out training seeds/deals and reporting unresolved uncertainty. Neither
coverage clocks, reduced gradient noise, nor a league point winner alone
establishes Nash convergence.
