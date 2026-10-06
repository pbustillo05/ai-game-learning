> Superseded by [the paired league v2 campaign](../hyperparameter_campaign_v2_20260928/README.md). Use its profile and launch command; this v1 protocol is historical.

# Shared Normal/RB hyperparameter campaign

This campaign selects **one shared parameter profile** for a paper comparison
between sampled (Normal) and Rao–Blackwell (RB) training. The estimator is a
paired experimental condition, never a candidate to select. Every setting runs
in both modes with the same initialization, seed, GPU allocation, wall-time
budget, and evaluation protocol.

The registered profile is
[configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml](../../../configs/profiles/nlhe_hu_32bb_hyperparameter_campaign_v1.yaml).
Its values are research controls, not a claim that these hyperparameters are
optimal. The exact-combination posterior clock, distributed resume, both-mode
training/BR/evaluation chain, and final common-attack panel have passed bounded
CUDA validation; the multi-day quality experiment is prepared for launch.

## Fixed parameters and search axes

The game is HU cash NLHE, 32 bb equal stacks, full deck, blinds 0.5/1 bb, no
rake/ante, sequential rollout, and no branching. Rewards, advantages, and
critic predictions use stack units: one unit is 32 bb and HU terminal payoff
lies in $[-1,1]$.

| Setting | Shared control |
|---|---:|
| AdamW learning rate | 0.00005 |
| AdamW beta1 / beta2 / epsilon | 0.9 / 0.999 / 0.00000001 |
| Weight decay / global gradient clip | 0.01 / 0.5 |
| Policy ratio clip | 0.2 |
| Value coefficient / value clip | 0.016 / 0.00625 stack |
| Reference KL coefficient | 0.00625 |
| Entropy initial coefficient | 0.0015625 |
| Entropy horizon | 250,000 selected exposure units |
| Reference refresh interval | 400,000 selected exposure units |
| Trace lambda | 0.95 |
| Per-rank learner batch / steps per iteration | 128 / 24 |
| Per-rank roots | 512 |
| RB completion samples | 1,024; absent in Normal |
| Execution packs: rollout / completion / learner | 48 / 512 / 12 |
| Reference output cache | 256 MiB |
| Precision / initial loss scale | FP16 / 1,024 |
| Checkpoint / inference-snapshot cadence | 600 / 1,800 seconds |

Three GPUs give 1,536 roots, global statistical batch 384, and 9,216 global
learner presentations per iteration. Realized player decisions depend on
trajectory lengths, so log presentations per fresh decision rather than
calling 24 updates a constant reuse rate. Physical microbatch size affects
memory without redefining statistical batch size.

The architecture stays fixed: width 128, four attention heads, four preflop
and six postflop layers, two raise components, minimum raise scale 0.02,
class embedding width 32, and folded-line width 8. The profile contains the
complete machine-readable configuration.

| Candidate | Change from control |
|---|---|
| control | None |
| entropy_zero | Entropy starts at zero |
| entropy_slow | Entropy horizon 1,000,000 |
| kl_weak / kl_strong | Reference KL 0.003125 / 0.0125 |
| reference_fast | Reference interval 100,000 |
| reference_fixed | Preserve initial reference throughout trial |
| learning_rate_fast | Learning rate 0.0001 |
| critic_weight_4x | Value coefficient 0.064 |
| critic_unclipped | Unclipped MSE, coefficient 0.016 |
| reuse_half | 12 learner updates per iteration |
| posterior_clock_raw | Exact current-history posterior breadth, exponent 1 |
| posterior_clock_sqrt | Breadth to power 0.5 |

The final two candidates use the clock definitions below. They are
clock hypotheses, not measured RB sample multipliers. Completion count stays
fixed: it is an RB-specific estimator operation, not a common schedule arm.
Entropy horizon and reference interval remain independent parameters.
The search screens individual changes and preregisters one bounded interaction
check. It is not exhaustive over parameter combinations, and its best observed
shared setting is not a proof of global optimality.

## Initialization and invariants

The requested checkpoint 1100 was unavailable. The verified nearby checkpoint
is 'runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0001110.pt', SHA-256
'bced881945512f2a4152d3acd88a0a06395a00130cdf68a61aa2652bc1b5aebb'.
See the [history audit](../hyperparameter_search_20260928/history.md).
Do not label it checkpoint 1100.

Screening/confirmation continue that model with fresh, consistent optimizer
state. Critic output affine weights/biases are divided by 32 exactly once;
the initial KL reference matches the converted model. Historical preflop input
encoding is installed on learner/reference/frozen-opponent lanes and their
separate functional-inference code templates. Matching weight hashes alone
does not establish input-encoding equivalence.

Cold trials use the checkpoint's recovered 'reference_model' initializer.
They test transfer of shared settings to initialization; four hours does not
establish convergence from scratch. Neither stage selects parameters separately
for Normal and RB.

Production distributed training sums local mean gradients. The experiment
worker divides that SUM by world size **before clipping/Adam**, preserving
global-mean objective scale for equal fixed-step local batches. It does not
adjust only a regularizer or automatically scale learning rate with GPU count.
See the [distributed audit](../hyperparameter_search_20260928/distributed_semantics.md).
Resume rejects topology changes. The experiment's bundle loader validates the
rank-zero global manifest and each rank's own execution identity; it does not
require identical physical VRAM capacity across the cards. The observed
approximately 1.8 MiB card-capacity difference motivated this experiment-local
restore path, without relaxing the per-rank identity checks.

## Posterior clocks: precise estimand and distinct units

The decision control increments $E$ by one per fresh global player decision,
excluding minibatch reuse. For the posterior alternatives, the measured
distribution is the actor's **current public-history reach posterior over
exact private combinations, before selecting the next action**. It is not the
learner target posterior conditioned on downstream completion/terminal data.

For normalized current-history posterior $p_i(h)$ at decision $i$:
$$
K_i=\exp\!\left[-\sum_h p_i(h)\log p_i(h)\right].
$$
The raw candidate increments RB exposure by $K_i$ and Normal exposure by one.
The square-root candidate increments RB by $K_i^{0.5}$, Normal still by one:
$E_\gamma=\sum_iK_i^\gamma$ in RB, with $\gamma=1$ or $\gamma=0.5$.

A breadth-37 decision therefore counts as 37 Normal decision units in the raw
candidate, or about 6.08 in the square-root candidate. This is a **scheduling
convention**, not evidence of learning equivalent to 37 or 6.08 independent
decisions. Likewise $\sqrt{1326}\approx36.4$ is not a measured RB acceleration
factor.

Entropy breadth describes posterior dispersion. Private hands share public
trajectories, completion worlds, parameters, and gradient directions.
Exponent 0.5 is a declared correlation-discount heuristic, not an exponent
estimated from covariance data. Normal's increment is a sampled-hand count,
not an estimate of the entropy of an unobserved posterior.

The hook observes resolved reach at the sequential transition, before action
updates and next-family compaction. Actor reach alone is not yet the exact
joint private-hand marginal: the other player's ranges and card collisions
must also be accounted for. For HU, the required marginal is proportional to
$\rho_i(h)\sum_{g:g\cap h=\varnothing}\rho_{-i}(g)$, with both hands compatible
with the known board. Normalizing only $\rho_i$ ignores opponent card-removal
evidence and must not be labeled exact posterior.

The experiment observer implements this blocker correction over the full
combination domain, normalizes the actor marginal, and counts each selected
fresh decision once. CUDA verification of that path belongs in the ledger. Preflop class mass expansion must preserve the exact resolved
reach under its within-class exchangeability assumptions; downstream learner
targets do not share that guarantee. The validation ledger must establish this
path before a posterior-clock training run is promoted.

The observer declares
'heads_up_actor_exact_combo_marginal_given_current_public_history_v1'.
Its records separate 'sum_posterior_exp_entropy',
'sum_sqrt_posterior_exp_entropy', and 'fresh_player_decisions', and explicitly
set 'completion_effective_hands_used=false' and
'is_independent_sample_count=false'. The correlation-discount status remains
unmeasured without repeated posterior-weighted combination gradients; target
advantage dispersion is not accepted as a covariance/ESS substitute.

The distinction matters: the learner's preflop 'ActionTarget.posterior' has
already collapsed 1,326 combinations into 169 classes and can depend on future
board evidence. Exact entropy satisfies
$H(hand)=H(class)+H(hand\mid class)$. A uniform lift of that downstream target
can be wrong: one possible combination has breadth one even when its class
has 4, 6, or 12 combinations. The older pilot's uniform-lift statistic is not
an exact downstream-posterior clock. Current-history reach must not be relabeled
as that unavailable downstream distribution.

Completion-world ESS is also a separate metric. With equal-weight completion
sampling it can equal the sample count, so raising completion samples would
advance the old ESS clock with no additional public data. Neither proposed
posterior clock substitutes completion ESS for combination entropy.

The external worker checkpoints all clocks and implements:
$$
c_H(E)=c_{H,0}\max(0,1-E/E_H).
$$
Entropy reaches exact zero. Reference refresh preserves residual exposure
and records threshold overshoot; copying occurs at real iteration boundaries,
not several fictitious intermediate policy states within one batch. Record
the chosen clock/exponent, raw breadth, discounted breadth, fresh decisions,
presentations, optimizer steps, and refresh count separately.

## Stages and evaluation

For thirteen candidates and three available GPUs:

| Stage | Settings × seeds × modes | Self-play per condition | BR budget per restart | Nominal training elapsed |
|---|---|---:|---:|---:|
| Screen | 13 × 1 × 2 | 2 hours | 0.5 hour | 65 hours |
| Confirm | 3 × 2 × 2 | 3 hours | 1 hour | 48 hours |
| Cold | 2 × 1 × 2 | 4 hours | 1 hour | 20 hours |
| Total | Shared settings throughout | | | 133 hours |

This is about 5.5 days **before evaluations and worker initialization**.
The plan requests 162,816 evaluation hands across own-response evaluations,
plus 8,192 for the default final common-attack panel (171,008 total). Their
runtime may be substantial. It is not a guaranteed 5.5-day end-to-end job.
'budget_plan.json' is authoritative for actual arm/GPU counts and durations.
One/two GPUs run response/evaluation restarts in waves and take longer.

Each self-play job uses all selected GPUs. Three BR jobs then run one per GPU,
followed by three evaluation jobs. BR uses the same fixed RB training protocol
for targets from both modes, without inheriting target-arm hyperparameters.
Two restarts initialize from the target, one randomly; seeds are 4101/4102/4103.
BR training receives the native rank-zero checkpoint, which carries the
production frozen-target metadata; evaluation uses separate inference
snapshots. The evaluator's opponent and baseline are both the frozen target
policy. These artifact formats are not interchangeable.

Screen seed is 918273; confirmation seeds are 20260929/20260930; cold seed is
20261001. Seeds are paired across settings and modes. Each BR restart receives
512/2,048/4,096 requested evaluation hands in screen/confirm/cold respectively.
Held-out namespaces, paired deals, seats, and action-randomness addresses
support candidate-minus-baseline comparisons against the same opponent.

For each target, take the largest achieved response gain among three restarts.
Average that score equally across Normal/RB to rank **shared settings**.
Confirmation uses identical paired seeds. Keep per-mode values, mode gaps,
restart estimates, and uncertainty flags. Achieved response gain is not true
exploitability or a Nash certificate.

Confirmation normally tests control, the strongest screened noncontrol single
change, and one deterministic combination. Starting from the strongest
noncontrol change, scan remaining settings in screen-score order and merge
the first whose dotted override paths do not overlap. Equal paths and
ancestor/descendant replacement paths conflict, so a whole value-loss
replacement cannot silently override a selected nested value coefficient.
The merged candidate is named 'combined_screen_winners' and recorded in
'combination_candidate.json'. It receives the same paired confirmation budget.
If no compatible second change exists, retain the original control-plus-two
single-setting shortlist. This keeps three confirmation settings and the
registered wall budget while testing one plausible interaction.

Later promotion uses point-estimate ordering and flags interval overlap; it
does **not**
automatically choose control when uncertain. Averaged interval endpoints are
descriptive envelopes, not new calibrated confidence intervals. A final
profile is explicitly provisional and applies to both estimators.

A target-specific BR can get stuck in a poor local solution: a weak attack
does not prove a strong target. The final common-attack panel addresses this
by selecting each cold target's strongest of three responses using the already
recorded own-target validation scores, then freezing those four response
snapshots. It evaluates every response against every cold target: two finalist
settings times two modes give four targets and a 4-by-4 matrix.

The panel uses the fresh split 'cross_panel_test_v1', a common seed across
pairs, and 512 requested hands per cell by default. It performs no additional
training. Frozen file/specification hashes prevent changing selected attackers
while resuming. Per-target scores are maximum held-out attack gains over the
common pool; equally averaging the two mode scores selects a shared profile.
The caller uses this provisional panel winner and flags disagreement with the
earlier own-response ordering. No estimator is selected.

This separates response-restart selection from the final test deals and can
detect weaknesses missed by a target's own attackers. It does not make the
finite response pool a best response oracle. The final target settings were
already adaptively selected, and reporting a maximum over test attacks still
has postselection uncertainty; interval maxima and pooled endpoints remain
descriptive, not formally calibrated confidence bounds. The implementation is
'experiments/hyperparameter_search/cross_panel.py'; full CUDA integration
evidence belongs in the validation ledger.

Repeated selection also creates winner's bias. The automation still produces
a concrete provisional best setting rather than refusing to choose, but paper
claims should use a new held-out comparison with seeds/deals not used for
promotion, paired Normal/RB conditions, and the common response pool.
Report uncertain or opposing mode-specific effects instead of treating an
aggregate point winner as established superiority.

## Diagnostics and covariance calibration

Log global roots/decisions, exact current-history breadth by street, completion
ESS, presentations, optimizer steps, reward/value scales, component losses and
gradient norms, clipping/nonfinite updates, policy KL, type/conditional-size
entropy, reference drift, throughput, and GPU telemetry. Self-play payoff,
root raise rate, or reduced entropy alone does not establish poker strength.

For empirical calibration, freeze checkpoint 1110 and collect 16–32 independent
trajectory batches per mode at matched public-root budgets. Measure
mean-gradient agreement and covariance on CUDA with uncertainty across
batches. Separate public-trajectory variation from private-hand/completion
variation; repeated per-combination gradient contributions are needed to
quantify correlation. Advantage dispersion alone is not gradient variance
or an effective sample size.

Compare variance reduction per GPU-second, stratify by street, and repeat at
later checkpoints. These results may justify a measured exposure multiplier;
they do not retroactively establish exponent 0.5. Look for broad posteriors
whose per-hand gradients carry little additional diversity.

## Runtime ownership and validation

The experiment worker owns normalized initialization, historical adapters,
global-mean gradients, external clocks, entropy decay, reference copying,
and resumable time accounting. **Do not use the ordinary production training
CLI for these profiles.** Native iteration scheduling does not implement the
external experiment controls.

Thermal policy is telemetry-only: logs include temperature, utilization,
memory, power, clocks, and hardware/software slowdown flags. The worker does
not modify power/fan settings or enforce a temperature ceiling. Earlier pilots
observed software thermal slowdown in some settings; a sustained heat soak
should be assessed by measured learning and throughput per wall-clock second.
Light throttling is acceptable when it gives the fastest result; there is no
extra temperature-margin objective. High utilization alone is not sufficient.

The launcher resolves UUIDs, refuses occupied selected devices, and never
stops another user's process. Immutable profile/spec/source hashes include
untracked production and experimental source. A file lock serializes campaign
ownership. Resume requires explicit completed outputs and preserves charged
training time; changed source or topology is rejected. SIGINT/SIGTERM forwards
only to owned child process groups for checkpointing.

**Completed engineering validation, 2026-09-28**

Machine-readable evidence is in
[validation_summary.json](../../../runs/experiments/nlhe_hu_32bb_rb_campaign_validation/validation_summary.json).

- **52 focused tests passed; Ruff passed** for all six new pipeline modules.
- Actual three-GPU full profiles completed two iterations each in RB and
  Normal: approximately 62–64 seconds per warm iteration, about 6,000 global
  fresh decisions, 24 synchronized updates, finite gradients, and matching
  final rank model hashes. GPU utilization reached 100%; this short test
  does not establish an optimum after hours of sustained load.
- The exact blocker-corrected current-public posterior observer ran on CUDA,
  including preflop and postflop. Its count matched fresh rollout decisions.
  The three-GPU raw-clock case counted 745 decisions as approximately
  820,891 breadth units, then resumed to 1,620,393 units / 1,481 decisions.
  Zero entropy and the next reference boundary survived interruption.
- The square-root arm completed two CUDA iterations: 481 decisions contributed
  approximately 15,882 discounted units. Normal contributed exactly one unit
  per fresh decision. These are exposure measures, not measured independent
  gradient samples. The repeated-gradient covariance diagnostic is implemented
  and unit-tested, but actual covariance data have **not** been collected.
- [Complete paired smoke report](../../../runs/experiments/nlhe_hu_32bb_rb_shared_campaign_smoke_20260928/report.json):
  both modes on three GPUs, three single-GPU BR restarts per target, six
  paired evaluations. Re-running with `--resume` skipped verified results.
- [Identity evaluation](../../../runs/experiments/nlhe_hu_32bb_rb_campaign_validation/evaluation_identity.json):
  64 paired observations gave exactly zero difference and interval `[0,0]`
  when candidate, opponent and baseline were the same model. It took 41.4 s.
- [Common-panel fixture](../../../runs/experiments/nlhe_hu_32bb_rb_campaign_validation/cross_panel_fixture/cross_panel/result.json):
  all 16 CUDA evaluation cells completed. The two named fixture arms explicitly
  alias the same smoke policies; this verifies orchestration, not strength.
- Distributed resume preserves each GPU's strict execution manifest. GPU2
  reports slightly different total VRAM, so checking only rank0's hardware
  manifest against every rank is invalid. The experiment loader validates the
  common rank0 bundle plus every rank's own native checkpoint and manifest.
- Shared loss coefficients were checked against the July objective: entropy
  and KL divide by32, value clip divides by32, and the squared-error value
  coefficient multiplies by32. This is objective-scale continuity, not
  identical Adam/clipping trajectories in different parameter coordinates.
- All three GPUs returned to idle after validation. No multi-day run was
  started. The full campaign's immutable job specifications are prepared in
  `runs/experiments/nlhe_hu_32bb_rb_shared_campaign_20260928`.

The 133-hour training/BR budget plus a simple extrapolation from the 64-hand
identity check (about10.2 evaluation hours for171,008 observations across three
GPUs) suggests **roughly six days**, with policy-dependent trajectory lengths
and evaluation overhead capable of changing that estimate. It is not a
three-day campaign. The default keeps the two modes paired and the independent
confirmation stages; abbreviating duration is a separate experiment.

Clock granularity matters: raw breadth is already roughly1,102 units per
fresh decision in the small continuation probe. The raw-clock arm can therefore
cross many thresholds in a full rollout. The runner applies one physical
reference copy at the completed iteration boundary and retains all threshold
crossings/overshoot. Interpret this arm as an intentionally aggressive,
batch-limited schedule, not evidence that exactly400,000 independent hands
were learned between every physical copy. `posterior_exposure_global` records
the candidate raw/square-root increment; `control.clock` and the persisted
clock counters identify the schedule actually used.

See [HANDOFF.md](HANDOFF.md) for commands. Source context:
[earlier diagnosis/pilots](../hyperparameter_search_20260928/README.md),
[NashPG](../hyperparameter_search_20260928/nashpg.md),
[broader PG literature](../hyperparameter_search_20260928/poker_pg_literature.md),
and [history](../hyperparameter_search_20260928/history.md). Their pilot results
are historical evidence, not validation of every new campaign control.
