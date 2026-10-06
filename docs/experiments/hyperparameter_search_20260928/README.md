# Normal/RB hyperparameter search: research and CUDA pilots

2026-09-28. This investigation establishes a working normalized-reward
continuation and useful throughput improvements. **It does not yet identify
optimal hyperparameters or demonstrate improved poker strength.** All completed
pilots use the repaired production pipeline at `0a09496`, with no experimental
completion permutation. Production code and `docs/theory/` were not changed.

The requested successful-run checkpoint 1100 does not exist. The available
checkpoint **1110** is used: `runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0001110.pt`,
SHA256 `bced881945512f2a4152d3acd88a0a06395a00130cdf68a61aa2652bc1b5aebb`.
Original run artifacts and pre-existing untracked work were preserved.

## The baseline to carry forward

| Setting | Normalized continuity control | Reason |
|---|---:|---|
| Reward | net bb / 32 | HU terminal utility in [-1,1] |
| LR | 5e-5 | successful July control; 1e-4 also smoke-tested |
| Entropy start | .0015625 | .05 / 32; separately test zero from trained checkpoint |
| Entropy end | exactly 0 | required; no positive asymptotic floor |
| Reference KL | .00625 | .2 / 32 |
| Value coefficient | .016 | .0005 × 32 for squared errors |
| Value clip | .00625 | .2 bb / 32 |
| PPO clip / lambda | .2 / .95 | held fixed |
| AdamW decay / grad clip | .01 / .5 | held fixed across Normal and RB |
| Completion samples | 1024 in RB | hold fixed while testing loss coefficients |
| Reference | frozen checkpoint 1110 | isolate coefficients from refresh cadence |

The four final value-affine tensors are divided by 32 **on CUDA**; policy and
trunk tensors are unchanged. The verified July preflop input adapter is applied
to learner, reference, and probes. Distributed functional code templates also
receive it. On three fixed states, policy probabilities were unchanged and value
scaling error was zero in the measured FP32 probes. Fresh AdamW state and loss
scale 1024 are used consistently. This is a continuation from the trained policy,
not exact resumption of July's optimizer/reference or a learning-from-scratch
reproduction. July's old anchor was the recovered iteration 20 initializer; the
new smoke anchor is checkpoint 1110.

The pilot entropy horizon is deliberately only **two iterations** (coefficients
c, c/2,0) to exercise the zero endpoint cheaply. This is **not** a proposed
long-run decay horizon. The larger-workload two-iteration capacity pilot uses
horizon 1. For continuation of an already trained player, zero entropy from the
start remains a serious candidate; this does not require deterministic play.

## Findings that change the search

1. **The production exposure clock does not implement the intended idea.**
   RB accumulates completion-world `n_eff`; Normal counts player decisions.
   With equal-weight completions, one 256-root batch at 1024 samples reports
   262,144 effective completions. The same first RB batch has 1009 decisions,
   native-class/posterior breadth 308,951, and a uniform-within-class hand-posterior lift of
   approximately 589,568 (an upper bound, not measured exact-hand breadth). These are different quantities. Doubling completion
   samples can approximately double today's RB clock without doubling public
   experience. Current entropy scheduling is iteration-based only.
2. **NashPG numbers require both version and unit interpretation.** The latest
   paper gives 10,000 inner rollout updates per reference; current default code
   gives 1000. The older vendored code's entropy .05 differs from paper .1;
   current official defaults now use .1. More consequentially, its code
   standardizes advantages per minibatch; ours does not. Therefore its KL .2
   cannot simply be copied into our stack-normalized loss. See the
   [paper](https://arxiv.org/html/2510.18183v3),
   [defaults](https://raw.githubusercontent.com/ntu-agents/nashpg/main/conf/algorithm/nash_pg.yaml),
   and [loss](https://raw.githubusercontent.com/ntu-agents/nashpg/main/train/core/update_agent.py).
3. **Reference caching is materially better than blindly enlarging packs.**
   In the256-root RB control, a256 MiB reference-cache budget reduced warm
   iteration time from 38.81 s to 27.08 s (30.2% less time, about 1.43× work/s),
   and peak allocated memory from 5.65 GiB to 3.99 GiB. These are short paired-seed
   pilots on identical GPU models, not a hardware-normalized benchmark.
   Pack16 gave only 3.3% lower warm time than pack 12, using 7.16 GiB instead of
   5.65 GiB. Pack24 without reference caching OOMed during the reference forward.
4. **GPU count changes the statistical and gradient budgets.** Batch size is
   per rank, and local mean gradients are SUM-reduced without dividing by world
   size. This preserves relative objective coefficients but changes clipping,
   optimizer moments and effective global batch. See
   [distributed_semantics.md](distributed_semantics.md). Three GPUs are not a
   transparent execution-only change with today's settings.
5. **The historical analyses already identified the unit conversion.** Commit
   `1cde6de` tested the continuity recipe on CUDA. Later stronger defaults in
   `3ec332d` were not selected by a saved controlled quality comparison. The
   successful July run used 6144 global roots and 116 updates/global batch 256,
   unlike 1536 roots and 38 updates in August. See [history.md](history.md).

## What was actually tested

The machine-readable [summary](../../../runs/experiments/nlhe_hu_32bb_rb_hyperparam_20260928_report/summary.json)
and [pilot table](../../../runs/experiments/nlhe_hu_32bb_rb_hyperparam_20260928_report/pilot_table.md)
contain every completed rank and link its raw JSON. Each run also has GPU
telemetry sampled every 2 s. Rows marked incomplete are not passing experiments.
Warm time excludes the first iteration; first RB iterations included lazy
compilation and should not be compared as steady-state timings. Timing excludes
fixed-state policy probes and reports cooldown separately.

The initial paired RB decay and zero-entropy arms each applied18/18 finite
updates. Final root raise probabilities were8.78% and 8.76%, respectively, with
substantial variation across private hands. LR1e-4 also applied18/18 updates,
ending at 10.85% raising and more reference drift. These are behavior diagnostics,
**not strength rankings**. Normal's decay control applied18/18 finite updates,
ending at 6.95% raising. The RB control and its execution-only pack/cache variants
had matching player-node counts in their three iterations; small policy
differences remain from floating-point execution order.

The RB two-GPU CUDA/NCCL pilot applied36 synchronized updates, with batch 128/rank,
256 roots/rank and 12 updates/iteration. Both ranks ended with the same model hash:
`5ba14ab99dc455be05cee7a2de5f863b1ad2766c09fbf6df664e2058cd3de8f5`.
Direct versus functional adapter output differences were below1e-6 on rank 0;
the raw validation contains both ranks. Warm iteration time was 30.82 s, or 64.9
fresh global decisions/s, compared with 38.5/s in the one-GPU cached pilot.
This approximately 1.69× observed throughput is not an exact strong-scaling
claim: synchronized update counts, batch partitioning, policies and SUM
normalization differ.

The pilot telemetry's `sum_exp_entropy` uses the learner's native private domain:
169 classes preflop versus 1326 combination slots postflop. A later pilot field
was initially called `sum_exact_hand_exp_entropy`; **that label is incorrect for
preflop**. It computes the uniform-within-class lift
$H(class)+E[\log(class\ multiplicity)]$, an upper bound on exact-hand entropy.
Actual conditional entropy is $H(hand)=H(class)+H(hand\mid class)$: future board
and action evidence can make the within-class posterior nonuniform. Production
class-sum target conversion loses that conditional distribution. Even a posterior
concentrated on one combination can lift to 4, 6 or 12 combinations if its class
mass is naively spread uniformly. This matters directly to the proposed clock.

The driver now calls this quantity `sum_class_uniform_lift_exp_entropy` and
leaves actual preflop exact-hand breadth unavailable; the aggregator corrects
the older field's interpretation without overwriting raw results. Postflop
breadth is computed from the actual stored combination posterior. Collect true
preflop exact-hand entropy before the lossy target conversion if it is required,
or calibrate the native learning domains separately. These metrics count fresh
decisions, not minibatch resampling. Normal has no posterior; its one sampled
private hand per decision is a sample count, not measured posterior breadth.
None of these counts should be represented as independent samples.

Thermal target read from the cards is 84°C; maximum operating89°C and slowdown91°C.
The driver records hardware/software thermal-slowdown flags and initially used an 82°C
iteration-boundary guard, cooling below75°C before more work. It does not change
fan or power settings. Boundary checking is not a continuous temperature cap.
An earlier Normal attempt stopped after one valid iteration at a conservative
78°C guard; the retried run passed. The failed pack 24 folder/config and
interrupted launch folder/config were removed; concise rejection evidence is
retained under the report directory.

Only two GPUs were used by these experiments at any time. Initially0/1 were
idle. When another user's validation process appeared on GPU 1, our launcher was
stopped and the successful distributed test moved to 0/2. The other process was
not signaled or modified. Three-GPU training remains unlaunched.


The final Normal zero-entropy arm also applied 18/18 updates, ending at 7.17%
root raising. The larger RB capacity pilot used 512 roots, 12 steps, family pack
48, completion pack 512, learner pack 12 and a 256 MiB reference cache. It applied
24/24 updates over two iterations, with warm time 52.09 s, 38.6 fresh decisions/s
and 4.21 GiB peak allocated. Increasing roots doubled work without materially
changing work per second in this small comparison. Multiple execution knobs
changed together, so this is a validated operating point rather than an isolated
family-pack or completion-pack optimum.

Across the ten numerically successful configurations there were 222 optimizer updates
(counting each synchronized distributed update once); all recorded gradients
were finite, and every scheduled update was applied. The maximum recorded-action
approximate KL among those iteration averages was below .0033. This is not a
bound on full-action KL or the largest individual conditional policy change.
The hottest observed pilot sample was 84°C in the Normal zero-entropy arm;
**software thermal slowdown was recorded in 8/69 samples for that Normal run, and in 1/59 samples for the higher-LR RB run (at a 72°C sampled core temperature). Neither reported hardware thermal slowdown.** This brief
peak reaches the card's target temperature, so the current evidence does not
establish the desired sustained thermal margin. Boundary guards missed within-iteration peaks. Those two pilots pass numerical checks but fail the no-throttle operating criterion. A final driver option adds rank-coordinated checks between optimizer updates, pausing at 75°C or a thermal flag until 70°C with no thermal flag. A longer heat soak and phase-aware monitoring are still required before a three-GPU full-power run.

The final guarded **Normal two-GPU** validation completed all 36 synchronized
updates and ended with identical model hashes on both ranks. Direct/functional
adapter error was at most $1.2\times10^{-6}$. Peak sampled temperatures were
71°C and 79°C, with **zero thermal flags in 103 samples per rank**. However,
coordinated cooling consumed 74.1 seconds across 205.6 seconds of measured
iteration time, and warm throughput was only 24.8 global decisions/s. This
conservative guard solves the observed throttling in this short validation at
substantial throughput cost; it is not a final optimized temperature controller.
The third GPU's cooling limits must therefore be part of performance tuning,
and sustained maximum utilization on all three has not been established.

A CUDA counterexample also validates the posterior-label correction: one suited
combination has breadth1, while uniformly lifting its class mass gives breadth4.
See `posterior_label_cuda_validation.json` in the report directory.

Final hygiene: our GPUs 0 and 2 returned to zero utilization and their original
9/19 MiB display-only memory. The other user's process on GPU 1 remained running
and was left untouched. No experimental training or monitor processes remain.

## Calibrate a useful shared clock before selecting refresh periods

Keep four counters with explicit units: unique public decisions (also roots),
exact-hand posterior entropy breadth, completion-world ESS, and optimizer
presentations/updates. Count fresh data globally across ranks once; don't count
reused minibatch rows as fresh data. Posterior breadth is valuable but the many
hidden hands under one public trajectory are correlated. Preflop classes and postflop exact-hand posteriors must not be added as
commensurate coverage without a justified measurement scheme.

At frozen checkpoint 1110, collect independent trajectory batches for Normal and
RB at equal public-root budgets and estimate the covariance of policy gradients
on CUDA. Use enough independent batches (start16–32, extend if uncertainty is
wide), report both variance and mean-gradient agreement, and compare per-GPU
second. A variance ratio at equal trajectory budgets can estimate a *local*
RB equivalent-sample gain, subject to bias and correlation checks. Recalibrate
by street and at later checkpoints; do not declare entropy breadth to be that
gain. Sweep completion samples 256/512/1024/2048 separately to test where further
sampling stops helping. The existing48h variance experiment is relevant but is
pre-existing work and was not changed or run by this investigation.

For the first longer quality runs, the defensible common clock is global fresh
player decisions, with optimizer presentations and roots also matched/reported.
After variance calibration, compare that clock with a measured RB-gain clock.
Use one checkpointed global exposure counter for both reference refresh and
entropy decay; define entropy as $c_H(E)=c_{H,0}\max(0,1-E/E_H)$, which reaches
exact zero. Treat $E_H$ and the reference interval as **independent** parameters:
entropy can disappear while a KL anchor continues to change. Preserve residual
exposure after threshold crossings and record overshoot. These are proposed
schedule changes; they have not been added to production in this pilot.

NashPG's nominal paper reference interval is 40.96 M collected transitions and
160 k optimizer steps; its code default is one tenth of those budgets. July's
1380×116=160080 optimizer steps per anchor is strikingly close to the paper's
update count, whereas August RB used 110×38=4180. This comparison motivates a
long-lived anchor control; it does not establish an exact cross-game cadence.

## Concrete next quality experiment

Use successive stages instead of an enormous cross-product:

1. Continue checkpoint 1110 on the repaired path with normalized continuity
   coefficients. Compare LR `{2.5e-5,5e-5,1e-4}` and entropy `{0, decay from
   .00078125, decay from .0015625}`. Hold anchor, value loss, completion samples
   and matched presentation/fresh-decision budget fixed. Run two paired seeds
   for screening, then at least three independent training seeds for finalists.
2. With LR/entropy narrowed, compare KL `{.003125,.00625,.0125}` and value
   weight `{.008,.016,.032}`, one axis at a time. Include component gradient
   norms and cosine similarities on frozen batches so scalar loss sizes do not
   masquerade as optimization influence. Test policy clip, value clipping,
   lambda, decay and update reuse only after these controls are stable.
3. Compare fixed anchor with reference intervals measured in global fresh
   decisions and the calibrated RB clock. Do not change coefficient and cadence
   together. Calibrate candidate exposure thresholds from observed workload;
   make paper-like160 k and code-like16 k optimizer-step budgets visible as
   comparison points, not inherited defaults.
4. Score policies on held-out, seat-balanced deals versus frozen July1110,
   recovered initializer, and multiple historical/finalist opponents. Report
   paired bb/hand differences with deal-cluster confidence intervals. Add
   equally budgeted learned best responses. A trained BR is evidence about
   exploitable gain, not an exact exploitability certificate. Track later-street
   coverage, hand sensitivity, action-type/conditional-size entropy, value error,
   full-action policy KL, clipping and skipped steps as supporting diagnostics.
5. Promote based on quality gained per wall/GPU-hour, with matched-data views
   alongside it. Short-run self-play payoff is zero-sum and root raise rate is
   not a quality score. Full learning-from-scratch confirmation should start
   from the recovered initializer and use a longer decay horizon, since the
   trained-checkpoint optimum need not transfer to initialization.

For throughput, retain the256 MiB reference cache and conservative learner 12
until larger packs are tested *with caching*. Profile rollout family packs
32/48/64 independently of learner packs12/16/20 and completion packs128/256/512.
Then increase roots 512/1024/1536 while maintaining about 1.3–1.6 learner
presentations per fresh player decision (July's measured control was 1.31).
Measure actual phase peaks: the existing `IterationTelemetry` duplicates one
aggregate peak into multiple phase-labelled fields, so those fields cannot
support independent phase-memory claims. Check memory across longer/deeper
trajectories and sustained heat before choosing a long-run cap.

For three GPUs, first resolve/test global-mean versus SUM semantics, then use
explicit per-rank and global budgets; for example local 128 means global 384.
Don't multiply LR by 3 automatically. Validate NCCL scaling and sustained
thermals on all three only after the user's green light. Two-GPU pilot success
does not establish a three-GPU thermal or memory envelope.

## Research notes and reproduction

- [NashPG paper/code/version audit](nashpg.md).
- [Broader poker PG literature and applicability](poker_pg_literature.md):
  AlphaHoldem, actor-critic/RPG, ACH/NeuRD, AlphaExploitem, additional poker PG
  projects, and explicit CFR/search/NFSP exclusions. This is a broad primary-
  source survey, not a proof that every existing project has been found.
- [Git history and previous CUDA evidence](history.md).
- [Distributed batch, gradient and adapter audit](distributed_semantics.md).
- Driver: `experiments/hyperparameter_search/smoke.py`.
- Aggregator: `python -m experiments.hyperparameter_search.summarize`.

Activate the venv and CUDA toolkit exactly as in `AGENTS.md`; check
`nvidia-smi` and select idle UUIDs. Example single-GPU cached control:

```bash
CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> python -m experiments.hyperparameter_search.smoke \
  --mode rb --tag new_cached_control --reference-cache 268435456 --iterations 3 --thermal-step-guard
```

For the tested single-node two-GPU layout:

```bash
CUDA_VISIBLE_DEVICES=<idle-UUID-1>,<idle-UUID-2> \
  torchrun --standalone --nproc-per-node=2 -m experiments.hyperparameter_search.smoke \
  --mode rb --tag new_distributed2 --reference-cache 268435456 \
  --batch 128 --steps 12 --iterations 3 --thermal-step-guard
```

The driver refuses an occupied GPU or an existing result path. Use the explicit
`--thermal-step-guard` for new experiments; the old boundary-only guard missed
short thermal events. Step-guard pauses are included in iteration wall time. Generated configs
record scientific/execution settings, but **must be used with this driver** for
legacy checkpoint initialization and input adaptation; ordinary training CLI
loading is not equivalent. Experimental saved model state also requires the
adapter and is not a standard export/checkpoint with resumable optimizer/RNG.
