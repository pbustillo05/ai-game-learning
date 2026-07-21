NOT YET ADDRESSED - TO KEEP

# Branching research specification

> **STATUS — RESEARCH DESIGN AND MEASUREMENT TOOLING ONLY.** None of the
> controllers, schedules, size estimands, or update rules in this note is a
> production training contract. The current sampler remains the default. A
> proposal may enter training only after its experiment gate below passes on
> held-out seeds and checkpoints.

## Problem statement

The `runs/nlhe6_10bb_7h` rank-local iteration logs show that the present
micro-tree experiment is both small and highly dispersed. Across rank-local
iterations, the approximate distribution is:

| statistic | decision rows |
|---|---:|
| mean | 15.3 |
| median | 9 |
| p99 | 107 |
| maximum | 1,303 |
| coefficient of variation | 2.30 |

About 97.2% of rank-local iterations contain fewer than 32 decision rows.
This is a poor design for comparing estimators or filling a training
minibatch: most updates see very little public variation, while rare trees
dominate storage and latency. Increasing the rollout scale should reduce
sampling noise by the law of large numbers, but scale is useful only after the
size distribution is controlled. The eventual rollout and minibatch sizes
should be selected together from measured training-memory profiles so that
VRAM is used efficiently with explicit headroom, rather than from the current
micro-tree defaults.

## Sampling object: one tree, not a forest

A poker rollout has one public initial state. Every public trajectory starts
at that same state and runs the same frozen policy with the same public input.
Replicating trajectories therefore adds children to one sampled labeled tree;
it does not create a statistically meaningful forest. The research object is
one public root per GPU/rank. The controller is evaluated at every expandable
node, with no special root refill rule and no presumption that the root should
receive most of the branching.

One root per GPU is retained only as an engineering boundary for data
placement. A later implementation may execute the first sequential waves on
CPU and migrate once enough model evaluations accumulate to saturate a GPU,
but that optimization is outside these experiments. It should be attempted
only if profiling shows that it repays its added state-transfer complexity.

The sampled topology remains duplicate-preserving. If a node has offspring
count `K`, all `K` child draws remain distinct even when their public labels
coincide, and their outgoing edge weights remain `1/K`. Branching is selected
before those child outcomes are observed.

## Experimental parameterization

### Fixed intensity

`beta_fix` is a table over public phase, street, and the number of prior player
nodes on the current street. Total player depth is recorded as a diagnostic.
No numerical schedule is proposed here. The experiments will measure the
useful schedule and then enforce monotonic candidate schedules only during
offline construction and combined validation.

The root is merely the preflop, street-depth-zero bucket. A plausible measured
schedule may branch less as a street becomes deeper, but values such as
`4, 3, 2, ...` are hypotheses, not defaults. A proposed intensity is capped by
the ordinary legal-event mask and by the experiment's explicit bucket cap.
The harness does not add a poker-specific structural-action-count API. The
number of unique sampled public labels, duplicate-label rate, and saturation
curve are offline diagnostics of effective action diversity; they are not a
runtime posterior-action estimator.

### Size estimands

Let

$$
\widehat C(\mathsf s)
=
\frac{|\mathfrak S_{\mathsf s\preccurlyeq}|-1}{N(\mathsf s)}
$$

be the realized marginal number of descendants below one child of a sampled
node. Two log-output size-head meanings are candidates:

1. The current geometric estimand,
   $S_{\rm geo}(\sigma)=\mathbb E[\log\widehat C\mid\sigma]$, gives
   $e^{S_{\rm geo}}$, the conditional geometric mean.
2. The experimental arithmetic estimand,
   $S_{\rm mean}(\sigma)=\log\mathbb E[\widehat C\mid\sigma]$, gives the
   conditional arithmetic mean while retaining a numerically convenient log
   output. It is fit with Gamma deviance

   $$
   \widehat C e^{-S_{\rm mean}}+S_{\rm mean}-\log\widehat C-1.
   $$

The logarithmic output is useful because costs range from hundreds to
hundreds of thousands. Changing the output to raw count space would create
unnecessarily large activation and gradient scales. Importantly,
$e^{\mathbb E\log\widehat C}$ is not generally conservative: Jensen's
inequality makes it no larger than $\mathbb E\widehat C$, with potentially
severe underprediction for heavy tails. The experiment, rather than an
assumed safety multiplier, decides which estimand is adequate. Capacity
headroom is kept separately to absorb residual forecast error.

### Incremental reserve controller

For stored nodes $\mathfrak S$, live frontier $\mathcal F$, and the chosen
cost prediction $C(\mathsf s)$, maintain

$$
R=|\mathfrak S|+\sum_{\mathsf s\in\mathcal F}C(\mathsf s).
$$

The hot path never rebuilds this sum. Before expanding a frontier node with a
candidate offspring count `K`, test

$$
R_{\rm candidate}=R+(K-1)C(\mathsf s).
$$

Admit the candidate only if it fits the configured soft budget; otherwise use
$\beta=1$ at that node. After materializing children
$\mathsf c_1,\ldots,\mathsf c_K$, update once:

$$
R\leftarrow R-C(\mathsf s)+K+
\sum_{\mathsf c_j\ \mathrm{live}}C(\mathsf c_j).
$$

The `K` term stores every child. Terminal children have no continuation term.
This is a local decision, not a permanent switch: an expensive wave can force
$\beta=1$, then cheap realized public states can lower the reserve and allow
branching at later nodes. A test-only oracle recomputes the frontier sum after
each wave and checks exact agreement; production-like experiment execution
uses only the incremental update.

### Constant-utility KKT controller

The existing KKT approximation balances a variance utility against marginal
tree cost. To study size stabilization without first trusting a noisy variance
head, set the variance proxy to a constant. For a player node $\sigma$, this
gives

$$
\beta_\kappa(\sigma)=
\max\!\left(
1,
\sqrt{
\kappa e^{-S(\sigma)}
\frac{
\sum_{\rho\preccurlyeq\sigma,\ o(\rho)\in\mathcal P}
\left(\prod_{\rho\preccurlyeq\alpha\prec\sigma}N(\alpha)\right)^{-1}
}{
\prod_{\rho\prec\sigma}N(\rho)
}}
\right),
$$

followed by the experimentally selected street/depth cap. This is compared
with the simpler cost-only ablation
$\max(1,\sqrt{\kappa/e^{S(\sigma)}})$.

After tree `k`, update the inverse multiplier multiplicatively:

$$
\kappa_{k+1}=\kappa_k
\left(\frac{M}{|\mathfrak S_k|}\right)^\eta.
$$

Because unconstrained intensity depends on $\sqrt\kappa$, a size ratio `r`
changes it by $r^{\eta/2}$. That square root does not prove stability: the
same multiplicative change may compound through depth. Whole-tree elasticity,
oscillation, and cap hits must therefore be measured for
$\eta\in\{0.25,0.5,1\}$.

## Protected-cohort lower-size refill

Production branching now optionally appends complete root-child subtrees when
the finished public tree is below `branching_min_nodes`. Because this stopping
rule depends on observed subtree sizes, every root target is computed only from
the ordinary pre-refill child prefix. Added branches contribute learning rows
strictly below the root. Final root transition weights are uniform over all
children, while nominal offspring, pre-refill degree, and final degree remain
separate metadata. This supersedes the earlier prohibition on post-rollout
refill with an explicit sampling law and root-estimator correction.

## Harness contract

`pbg-branching-experiment` loads one scientific config and one or more frozen
checkpoints, performs rollout only, and writes JSON. It never constructs an
optimizer, writes a checkpoint, appends training metrics, or changes model
weights. The optional controller is passed directly to the sampler. Omitting
it remains behaviorally identical to the current fixed-intensity rollout;
`legacy` is an instrumented equivalence controller.

The CLI accepts explicit checkpoint paths, seeds, trial count, node capacity,
controller variant, bucket schedule, bucket cost table, soft budget fraction,
initial $\kappa$, update exponent, target size, and global cap. The controller
cost table is expressed in predicted node counts $C=e^S$, never in raw
log-head units. Each trial records:

- phase, street, total player depth, and within-street player depth;
- configured, KKT-proposed, and capped intensities, followed by randomized,
  budget-admitted, and realized offspring counts;
- public child labels, unique-label fraction, and collision rate;
- realized $\widehat C$, plus geometric, arithmetic, and Gamma-optimum bucket
  predictions;
- reserve before proposal, after admission, and after child realization;
- nodes, leaves, decision rows, decision-law ESS, and overflow margin;
- rollout time and peak CUDA allocated/reserved memory;
- target and gradient-proxy variance when the selected rollout path exposes
  them, otherwise an explicit unavailable marker.

The JSON also contains per-street/depth aggregates, checkpoint identity and
hash, configuration, device metadata, and raw trial rows. A typical isolated
measurement is:

```bash
pbg-branching-experiment \
  --config configs/train/nlhe6_10bb.yaml \
  --checkpoint runs/nlhe6_10bb_7h/ckpt_0000250.pt \
  --controller legacy --trials 16 --seed 20260715 \
  --node-capacity 4096 \
  --output benchmarks/results/branching_legacy.json
```

## Experiments

All comparisons use common random seeds. Confidence intervals are paired
bootstrap intervals over seeds, with checkpoints treated as separate strata.
Both rank-local distributions and pooled summaries are retained.

### 1. Size-head estimand

Run $\beta=1$ and existing fixed-$\beta$ rollouts. Split seeds before
fitting bucket statistics. For each public street/depth bucket compare:

- $e^{\operatorname{mean}\log\widehat C}$;
- $\operatorname{mean}\widehat C$;
- the held-out optimum fit by Gamma deviance.

Report signed mean forecast bias, mean relative error, underprediction
frequency, Gamma deviance, and root-predicted versus realized total tree size.
Include deterministic heavy-tail fixtures, for example costs with a rare
$10^6$ outcome, where the geometric estimate is analytically far below the
arithmetic mean.

**Hypothesis S:** the arithmetic estimand reduces held-out underprediction and
absolute total-size bias. **Promotion gate:** its paired 95% intervals for both
quantities must exclude no improvement at every checkpoint, and it must not
worsen median relative error. Otherwise retain the geometric head and treat
budget headroom as the separate control.

### 2. Empirical `beta_fix`

Use checkpoints near iterations 250, 1,000, and 5,750. Establish $\beta=1$
baselines, then pulse one bucket at a time through
$1.25,1.5,2,3,4$, leaving all other buckets at one. Measure marginal nodes,
leaves, decision ESS, label collisions, target variance, gradient-proxy
variance, rollout time, and peak memory.

Construct monotone candidates from measured variance reduction per added node
and per second. Validate the complete candidates on new common seeds because
isolated bucket effects need not compose.

**Hypothesis B:** some non-root buckets provide useful variance reduction at
lower marginal cost than concentrating fanout at the root. **Promotion gate:**
a combined schedule must be Pareto-undominated on decision ESS or an available
variance measure versus both nodes and time at all three checkpoints, have a
paired 95% improvement interval on at least one quality measure, and produce
zero overflows. No guessed schedule is promoted.

### 3. Constant-utility KKT

Compare the selected fixed schedule, cost-only
$\sqrt{\kappa/e^S}$, and the full constant-utility path-factor rule, all with
the same measured caps. Sweep $\eta\in\{0.25,0.5,1\}$, initial $\kappa$,
checkpoint, and target size. Report convergence traces, overshoot, oscillation,
tree-size CV, p99, cap-hit frequency, budget-forced-spine frequency, ESS, time,
and empirical whole-tree elasticity.

**Hypothesis K:** multiplicative dual control moves size toward `M` without
depth-wise oscillation. **Promotion gate:** after a predeclared burn-in, the
paired 95% interval must show lower absolute target error and lower size CV
than fixed $\beta$, with no lower median ESS, no overflow, and no persistent
two-cycle in the signed error. Select $\eta$ only from this sweep.

### 4. Dynamic budget switching

Hold $\kappa$ fixed within each tree. Admit proposals with the incremental
reserve and sweep soft fractions `0.4, 0.5, 0.6, 0.7`. Verify in raw traces
that expensive waves force one child and that later cheap states can restore
branching. Never refill a completed node.

**Hypothesis R:** dynamic admission lowers the p99 and CV without permanently
collapsing later buckets to $\beta=1$. **Promotion gate:** the reserve oracle
must agree after every test update; all realized trees must fit capacity; the
paired 95% interval must show lower p99 or CV than the same controller without
admission; and median decision ESS per second may not decrease. The selected
soft fraction must also leave positive measured CUDA headroom in the held-out
checkpoint runs.

## Test and completion checklist

- Absence of a controller and the instrumented `legacy` controller reproduce
  byte-identical current topology for identical seeds.
- Incremental reserve agrees with a from-scratch oracle after every wave.
- Gamma-deviance optima are analytic for targets from $1$ through $10^6$,
  including a heavy-tailed mixture.
- $\kappa$ updates, street/depth caps, randomized rounding, and controller
  traces are deterministic for a fixed seed.
- Synthetic exact-law tests cover dynamic switching and preserve weighted
  terminal laws.
- Attempts at post-completion refill fail explicitly.
- A tiny poker CUDA rollout runs on a confirmed-idle GPU; `nvidia-smi` is
  checked before and after, with no residual process or unexpected allocation.
- A smoke-result JSON is retained with the note and code changes.

Completion of this research deliverable means the note, the two red theory
annotations, the isolated CLI, tests, and one tiny smoke JSON exist and pass.
It does **not** authorize a production sampler, head, loss, controller, or
training-configuration change. The larger fixed-checkpoint sweeps are a
subsequent, explicitly launched research run.
