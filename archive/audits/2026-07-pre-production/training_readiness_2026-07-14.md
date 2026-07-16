# Near-term training readiness audit

Audit date: 2026-07-14
Repository: `/home/pablob/ai-game-learning`
Primary mathematical authority: [`../theory/core_math.tex`](../theory/core_math.tex)
Companion authorities: [`../../CONTRACTS.md`](../../CONTRACTS.md) and [`../../NOTATION.md`](../../NOTATION.md)

## Goal

Determine what must be changed before beginning useful training within the next
few hours, without spending time on journal-referee notation issues or
premature benchmark work. The priorities were:

1. **Extremely high:** validate the implemented algorithms against
   `core_math.tex` and identify anything that could change the estimator or the
   first optimization gradient.
2. **High:** determine whether the implementation is sufficiently practical
   and OOM-safe for staged runs on the current shared 3x RTX 2080 Ti server.
3. **Lower:** identify worthwhile multi-GPU, portability, and modularity work
   that should not delay the first synthetic experiments.

The immediate target was a real Kuhn, Leduc, or other synthetic training run.
Poker was included to determine whether it was also safe to start, but a full
6-max production run was not treated as necessary for declaring the synthetic
path ready.

## Work performed and observations

### Static math-to-code trace

The three implemented training modes were traced from rollout through target
construction, minibatch sampling, and loss evaluation:

- ordinary sampled-private terminal training;
- sequential Rao--Blackwell training;
- fixed branched Rao--Blackwell training.

The following important pieces are substantially aligned with
`core_math.tex`:

- Sequential rollout samples one private profile at the root and carries it
  down its path.
- Branched rollout samples a fresh posterior joint profile for every abstract
  child.
- Randomized rounding is `floor(beta) + Bernoulli(frac(beta))`.
- Duplicate public labels remain separate abstract children with independent
  descendants and completion draws.
- Each realized child has uniform edge mass `1/K`.
- The empirical decision law uses stored path mass. For sequential paths this
  reduces to uniform sampling from the pooled decision rows; for a branched
  tree it is the stationary path-weighted law.
- The backward pass propagates posterior mass `P` and posterior-weighted return
  mass `M`, takes uniform child averages, and conditionalizes only when targets
  are emitted.
- Value bootstrapping follows the acting player's decision clock.
- Branched advantages use the leave-one-out sibling baseline, with the old
  value as the single-child fallback.
- Policy loss terms use child posterior mass; value and entropy terms use the
  parent posterior mass.

No additional launch-critical discrepancy was found in these core estimator
steps. Exact end-to-end gradient coverage is strongest at `lambda=1`; the
intermediate-lambda recursion was inspected and appears aligned, but does not
yet have an equally strong closed-form integration test.

### Repository tests

The complete CPU suite was run twice during the audit. The final run was:

```text
/home/pablob/.venvs/poker/bin/python -m pytest -q
198 passed, 8 skipped in 55.57s
```

This establishes that the committed unit and contract tests pass. It did not
establish that rollout inputs and optimization inputs are identical on a real
poker trajectory; that missing integration invariant was the source of the
most important finding below.

Targeted CUDA tests were run on explicitly selected idle GPU 1:

- CPU/CUDA value parity;
- preflop forward/backward;
- postflop forward/backward;
- sibling evaluator output-device integration.

Result:

```text
4 passed in 2.96s
```

GPU 0 was already occupied by another Python process using about 2.9 GiB and
was not touched. GPU 1 was 9 MiB idle before the work and returned to 9 MiB
idle afterward.

### Synthetic training runs

The shipped Kuhn and Leduc presets were run in isolated `/tmp` output
directories so repository run directories remained untouched.

#### Kuhn

Configuration: sequential Rao--Blackwell, 512 trajectories per iteration,
exact completion, 300 iterations.

Observed resource use:

```text
wall time:       186.41 s
maximum RSS:     804,160 KiB
one iteration:   about 0.804 profiled s
decisions:       1,155 in the profiled iteration
leaves:          512
retained target: 148,995 bytes, about 129 bytes/decision
```

Exact exploitability by checkpoint:

| Iteration | Exploitability |
|---:|---:|
| Uniform initialization | 0.916667 |
| 25 | 0.3468 |
| 50 | 0.4837 |
| 75 | 1.1436 |
| 100 | 0.6319 |
| 125 | 0.4277 |
| 150 | 0.8743 |
| 175 | 0.6907 |
| 200 | 0.6819 |
| 225 | 0.7017 |
| 250 | 1.3243 |
| 275 | 1.8330 |
| 300 | 0.29927 |

The run learned, but the trajectory was highly oscillatory. The last
checkpoint happened to be good; it should not be assumed that the last
checkpoint is generally the correct selection.

#### Leduc

Configuration: fixed branched Rao--Blackwell, exact completion, 300
iterations.

Observed resource use:

```text
wall time:       26.84 s
maximum RSS:     808,280 KiB
one iteration:   about 0.112 profiled s
realized nodes:  29 in the profiled iteration
decisions:       14
leaves:          14
retained target: 4,396 bytes
```

Exact exploitability by checkpoint:

| Iteration | Exploitability |
|---:|---:|
| Uniform initialization | 4.74722 |
| 25 | 4.3364 |
| 50 | 4.2543 |
| 75 | 2.7153 |
| 100 | 4.0638 |
| 125 | 3.4967 |
| 150 | 2.9333 |
| 175 | 2.6128 |
| 200 | 3.1696 |
| 225 | 3.5520 |
| 250 | 4.2734 |
| 275 | 3.2326 |
| 300 | 3.2202 |

This also learned a useful policy, but non-monotonically. The best measured
checkpoint was iteration 175 rather than the final checkpoint.

Both shipped presets use one gradient step per rollout iteration. Once the
old/current-policy identity is correct, the ratio is one at that sole step and
PPO clipping is inactive. These presets therefore behave as valid one-step
on-policy actor--critic updates, not as PPO-style repeated reuse of a rollout.

### Poker CUDA launch and environment observations

The first tiny poker benchmark attempts exposed two operational requirements:

1. Invoking the virtualenv Python by absolute path does not automatically put
   the virtualenv's `ninja` executable on `PATH`.
2. The sibling CUDA extension requires `CUDA_HOME`, even when a compatible
   extension already exists in the Torch extension cache.

The working local environment was:

```text
PATH=/home/pablob/.venvs/poker/bin:$PATH
CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1
```

The packaged compiler was `nvcc 13.0.88`. A checked launcher should establish
this environment rather than relying on each operator to rediscover it.

### Tiny and medium poker CUDA smokes

The committed short-stack heads-up benchmark completed after fixing the
environment:

```text
node budget:                 256
completion samples:          8
effective minibatch:         2
gradient steps:              1
realized nodes/decisions:    3 / 2
profiled iteration:          4.345 s
rollout:                     3.376 s
optimization:                0.876 s
rollout peak reserved:       about 42 MiB
optimization peak reserved: about 46 MiB
```

This was only a boundary smoke: it folded early and did not exercise a
postflop training state.

A bounded 6-player smoke used the production-width model with a 512-node
budget, 16 completion samples, effective minibatch 4, and one gradient step:

```text
wall time:                    13.49 s
maximum RSS:                  about 2.16 GiB
realized nodes/decisions:     12 / 11 in the controlled run
profiled iteration:           9.105 s in the earlier medium run
rollout:                      5.249 s
optimization:                 3.753 s
rollout peak reserved:        about 279 MiB
optimization peak reserved:   about 999 MiB
```

The workload was still not representative of a broad postflop tree. It is
useful as a residency and bounded-attention smoke, not as evidence that the
4,096-node production workload has been calibrated.

### Poker old/current-policy identity failure

The medium run initially reported approximately:

```text
ratio_max: 1.30
clip_frac: 0.25
```

This occurred before any optimizer step, when PPO requires the current and old
policies to be the same. AMP was initially suspected because rollout stores
FP32 outputs while training recomputes under autocast. A paired FP32 diagnostic
disproved that hypothesis:

```text
FP32 ratio_max: 1.2507
FP32 clip_frac: 0.25
```

A decision-by-decision comparison showed:

- the root matched;
- the next decision matched;
- the mismatch began exactly at the first decision after a player folded;
- every later preflop decision remained different.

The cause is [`compact_decision_reach`](../../pbg/games/poker/rollout_backend.py),
which retains only live-seat ranges. Optimization reconstructs every folded
range as all `-inf`. The preflop model nevertheless consumes all seat ranges
in class tokens, reach-prior key bias, incidence features, and pooled seat
features. The frozen rollout model therefore sees a different input from the
optimization model.

Using the original un-compacted decision reaches restored identity:

```text
maximum absolute selected-logp difference: 9.5367e-7
maximum absolute value difference:         2.6822e-7
```

With exact reaches, AMP itself was benign:

```text
ratio_min:                  0.999291
ratio_max:                  1.000508
unweighted clip fraction:  0.0
mean absolute log ratio:   0.000158
```

This is a correctness blocker for poker in all three training modes, but it
does not affect the synthetic backend.

### Padded branched-event failure

An FP32 branched poker diagnostic also triggered:

```text
selected_event_ids must select supplied legal events
```

The loss replaces padded child slots with `legal_event_ids[:, 0]`. In poker,
column zero is fold, which is illegal when the actor is not facing a bet. The
padding fallback must instead gather an event whose legal mask is true. This
is independent of the folded-range compaction problem.

### Static memory observations

For the shipped 6-player configuration:

```text
model parameters:              23,775,468
model parameter storage:       90.70 MiB fp32
advertised rollout plan:       291.97 MiB
full 4,096-node decision reach:124.31 MiB
```

Retaining full decision reaches is therefore a practical correctness-first
fix for the current workload. Its worst-case additional retained belief state
is modest relative to the 11 GiB GPU.

The rollout plan is not a hard upper bound. In the branched sampler, offspring
are expanded into repeated state and `[child,P,1326]` reach tensors before the
tree buffer checks remaining capacity. The live implementation can also hold
the old frontier, an active-frontier gather, repeated child reaches, successor
reaches, and player-specific gathers simultaneously. The stated two-frontier
estimate undercounts these transient copies.

The per-iteration frozen model is created with `copy.deepcopy`, adding another
approximately 90.7 MiB. It is not a current OOM blocker, but it is unnecessary
for a synchronous rollout/optimization pipeline once identity is guarded.

### Poker loss-scale observation

The bounded 6-player diagnostics produced initial value losses on the order of
`1.2e3` to `6.9e3`. Poker rewards are in chip/big-blind scale, while the
configuration uses `c_v=0.5` and one global gradient-norm clip. Without
separate gradient diagnostics, the value gradient may dominate the combined
gradient and cause the global clip to suppress the policy update.

The trainer currently discards the norm returned by `clip_grad_norm_`, so this
cannot be diagnosed from normal training metrics.

## Implementation follow-up: launch blockers fixed

Follow-up date: 2026-07-14

The first three extremely-high-priority items and the first two high-priority
items from this audit have now been implemented. KL-regularized NashPG-style
training and synthetic CUDA execution were added at the same time because they
are required for the intended first runs.

### Exact poker decision inputs

Poker now declares `requires_full_decision_reach = True`. The generic
`DecisionReachCache` honors this declaration and retains the exact dense
`[D,P,1326]` rollout reach rows through optimization in normal, sequential-RB,
and branched-RB modes. The old live-seat compaction utility remains available
for future use, but is no longer used for training examples because replacing
folded rows with `-inf` changes preflop model inputs.

A regression test verifies that poker decision-cache construction owns the
original all-seat tensor even when the backend exposes its older compaction
method.

### Legal padded branch events

Inactive child slots no longer use legal-event column zero. The loss now finds
the first `True` legal-mask column per row and gathers its event id before
masking inactive slots out of the objective. A regression covers a row where
column zero is illegal.

### Old/current-policy identity launch gate

The first update of a new launch, and the first update after resume, now checks
every retained decision in bounded physical policy batches before backward:

- selected active-event log probabilities match the rollout cache;
- cached and recomputed values match;
- all active ratios remain inside the PPO clipping interval;
- FP32 uses a `2e-5` tolerance and AMP uses a `2e-3` tolerance.

The check is mode-independent and uses explicit-row gathers rather than
resampling the alias law. Resume performs the check without changing the
deterministic metrics prefix.

The bounded heads-up poker CUDA smoke passed with:

```text
identity ratio range:       0.9994299 to 1.0003335
max absolute logp error:    0.0005703
max absolute value error:   0.0004883
pre-update clipped rows:    0
peak reserved CUDA memory:  44 MiB
```

This is direct evidence that the folded-range mismatch is fixed on that
trajectory. It is not yet a broad-postflop or production-tree benchmark.

### Overflow before expansion

`DeviceTreeBuffer.require_child_capacity` is called before sequential public
steps and, critically, immediately after branched offspring counts are sampled
but before `repeat_interleave`, posterior profile sampling, reach gathers,
state steps, or successor allocations. CPU raises synchronously and CUDA uses
an on-device assertion. The sampled tree is never silently truncated.

### Analytical and measured CUDA memory gates

Training now builds an analytical plan before rollout. It separately records:

- parameters, gradients, and both Adam moments;
- rollout snapshot and optional KL reference model;
- topology and edge storage;
- configured reach storage and four simultaneous transient reach waves;
- exact retained decision reaches;
- policy cache and mode-specific targets.

The structural peak must fit the currently free/configured VRAM limit before
rollout. The first CUDA iteration then profiles every phase. Its byte-complete
peak-reserved measurement includes model activations, allocator behavior,
completion/evaluator workspace, and returned tensors that cannot be inferred
reliably from configuration. Later iterations are allowed only if the measured
peak plus a 25% safety margin fits the hardware profile's absolute safe limit.
The analytical breakdown and measured calibration are persisted in
`hardware.json` and `metrics.jsonl`.

Measured first-iteration calibrations on visible physical GPU 1 were:

| Workload | Measured peak | With 25% margin | Safe limit |
|---|---:|---:|---:|
| Kuhn sequential RB + KL, 64 trajectories | 22 MiB | 27.5 MiB | 8.57 GiB |
| Kuhn normal + KL, 64 trajectories | 22 MiB | 27.5 MiB | 8.57 GiB |
| Leduc branched RB + KL | 48 MiB | 60 MiB | 8.57 GiB |
| Heads-up poker CUDA smoke | 44 MiB | 55 MiB | 8.57 GiB |

These are launch calibrations, not extrapolations for the untested 4,096-node
six-player topology.

### NashPG reference-policy KL

The exact legal-action term

```text
c_kl * KL(current policy || reference policy)
```

is now active in ordinary, sequential-RB, and branched-RB losses. Ordinary
training evaluates it at the sampled actor type. RB training integrates it
against the parent actor-type posterior. Both use the empirical decision law
of their mode. A consistency regression constructs the same finite empirical
law in normal and RB form and obtains the same KL contribution.

`training.kl_ref_interval = K` implements the `core_math.tex` schedule
`ref(k) = K floor(k/K)`: the immutable reference is refreshed from the current
model before rollout at iterations `0, K, 2K, ...`. KL-enabled checkpoints own
both the reference weights and reference iteration. A strict interruption test
shows bitwise continuation with `c_kl=0.2` and a reference update boundary.

The local NashPG paper reports the following experimental settings:

| Setting | Paper value |
|---|---:|
| KL/regularization strength | `0.2` after a sweep |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| PPO epochs | 4 |
| Minibatches per epoch | 4 |
| Discount | 1 |
| GAE lambda | `0.95` |
| PPO clip | `0.2` |
| Entropy coefficient | `0.1` |
| Maximum gradient norm | `0.5` |
| Parallel environments / rollout length | 64 / 64 |
| Kuhn inner updates / outer refinements | 2,500 / 20 |
| Other games inner updates / outer refinements | 10,000 / 25 |

The paper reports stable practical behavior for `alpha >= 0.2` on Kuhn and
Leduc and uses `0.2` thereafter. It explicitly states that practical NashPG
does **not** inherit the formal IMMD convergence guarantee. Its annealing
experiment also diverged as the coefficient approached zero, including when
the learning rate was jointly annealed. We therefore do not schedule `c_kl`
toward zero.

Two GPU smoke configs were added:

- `configs/train/kuhn_nashpg_gpu_smoke.yaml`;
- `configs/train/leduc_nashpg_gpu_smoke.yaml`.

They transfer `c_kl=0.2`, AdamW, `3e-4`, lambda `0.95`, clip `0.2`, entropy
`0.1`, and grad norm `0.5`. They deliberately use four with-replacement reuse
steps and a scaled 25-iteration reference interval. They are **not** exact
reproductions of the paper's 2,500/10,000-update inner loops. Treating four
epochs times four minibatches as 16 with-replacement steps was tested and was
not equivalent: on a two-decision Leduc tree it produced `clip_frac=1.0` and
`ratio_max=1.384`. Four steps produced `clip_frac=0`, `ratio_max=1.038`, and a
nonzero reference KL contribution.

### Synthetic CUDA I/O contract

Synthetic bundle construction now honors the requested CUDA device. Model
parameters, opaque state-handle tensors, reaches, sampling weights and draws,
policy outputs, completion statistics, targets, losses, and optimizer state
remain on that device. Exact and Monte Carlo completion return device-local
tensors.

The intentionally retained Python public-state table is documented as a small
CPU control plane: decoding opaque handles and looking up or creating public
successors may cross to the host. This is not described as GPU computation.
The exception does not permit numeric rollout or learning tensors to escape to
CPU. A CUDA residency regression verifies the model, targets, state refs, and
decision reaches are on CUDA.

### Follow-up validation

The final CPU-isolated repository suite passed after the changes. The CUDA-
bearing test files then ran on explicitly selected idle GPU 1:

```text
74 passed, 1 warning in 5.27s
```

The warning is PyTorch's notice that CUDA synchronization debug mode is a
prototype. GPU 1 returned to its original 9 MiB idle state after every staged
run. The unrelated 2.9 GiB process on GPU 0 was not touched.

## Final readiness output

### Bottom line

**Kuhn and Leduc are ready for staged CUDA/KL runs now. The two discovered
poker correctness blockers are fixed and the bounded poker identity smoke
passes; six-player poker still requires the planned 512/1,024/2,048-node
calibration ladder before a 4,096-node run.**

The core rollout and posterior-value estimator is substantially faithful to
`core_math.tex`. The launch blockers are integration errors in reconstructing
poker model inputs and padded action handling, rather than a different
Rao--Blackwell or branching law.

### Extremely high priority status

#### 1. Preserve exact poker decision inputs — implemented

Immediate safe implementation:

- retain full `[D,P,1326]` decision reaches through optimization;
- do this for normal, sequential RB, and branched RB;
- add old/current-policy identity tests after preflop and postflop folds.

Optimized follow-up:

- retain all-seat reaches preflop;
- retain only live-seat reaches postflop;
- materialize and retain immutable folded-summary inputs at every decision;
- stop depending on mutable backend caches to reproduce a stored example.

#### 2. Select a genuinely legal padding event — implemented

For inactive branched child slots, gather the first `True` entry of
`legal_event_mask` and use its event id. Add a regression with mixed child
counts and a poker row where fold is illegal.

#### 3. Add an old-policy identity launch gate — implemented

Before the first optimizer update, verify with an AMP-aware tolerance that:

- active PPO ratios are approximately one;
- no active row is clipped;
- recomputed values match cached rollout values.

Cover every mode, folds, chance transitions, postflop states, mixed child
counts, FP32, and AMP. This gate would have caught both practical correctness
failures despite the otherwise passing suite.

#### 4. Add the intermediate-lambda exact oracle

Construct a small exact trajectory with zero, one, and multiple downstream
decisions by the same player. Compare closed-form targets and gradients at
`lambda=0`, an intermediate value, and `1`.

This should not prevent tonight's first synthetic run. `lambda=1` has the
strongest existing end-to-end exact-gradient evidence.

### High priority status

#### 1. Reject tree overflow before child expansion — implemented

Sum the realized offspring count and compare it to remaining tree/state
capacity before `repeat_interleave`, profile sampling, reach gathers, state
steps, or successor allocation. Preserve the existing loud overflow behavior;
do not truncate a logical tree.

#### 2. Replace the rollout estimate with a measured byte-complete plan — implemented

Account for:

- the old, active, child, and successor reach tensors;
- state/tree/policy caches;
- compact or full decision inputs;
- trainable parameters, gradients, Adam state, and any frozen copy;
- attention activations and allocator fragmentation;
- evaluator deterministic-reduction workspace and returned tensors;
- completion and backward-frontier targets;
- the configured VRAM reserve.

Use a tiny calibration run to validate the analytical estimate before growing
the node budget.

#### 3. Stage poker rather than jumping to 4,096 nodes

Recommended order:

1. tiny heads-up boundary run;
2. 6-player, 512 nodes, small completion budget and minibatch;
3. 1,024 nodes;
4. 2,048 nodes;
5. 4,096 nodes only after observed peak memory and topology are safe.

Record realized nodes/leaves, every phase peak, completion workspace, gradient
norms, and final GPU state at each stage.

#### 4. Normalize or rebalance the poker value objective

- log pre-clip total, policy, and value gradient norms;
- normalize poker returns/advantages to a documented scale or reduce `c_v`;
- consider separate clipping or optimizers if the value gradient continues to
  dominate.

#### 5. Validate the maximum public-history length

The engine permits a much larger rollout-step bound than the model's default
256 history tokens. Derive a tight legal bound and either increase the model
capacity or reject incompatible configurations before starting rollout.

#### 6. Add a checked local CUDA launcher

The launcher should select an idle visible GPU and establish the shared venv,
packaged CUDA toolkit, `ninja`, and library path. It should print the resolved
device/toolkit and run a pre/post GPU hygiene check.

#### 7. Remove the per-iteration model copy after identity is protected

The synchronous pipeline can use the same parameters under `eval()` and
`no_grad()` for rollout, then switch to `train()` for optimization. This saves
about 90.7 MiB and a redundant copy without changing the frozen rollout policy
because no optimizer step occurs between the phases.

### Tonight's run recommendation

Start with the new Kuhn and Leduc NashPG GPU smoke configs. Keep exact
exploitability at every checkpoint and select the best checkpoint rather than
the last. The 25-iteration reference interval is an intentionally compressed
experimental schedule; compare it with at least one larger interval before
drawing convergence conclusions.

If the intent is to exercise PPO clipping rather than one-step on-policy
actor--critic, increase `gd_steps_per_iter` above one only after choosing an
appropriate learning rate and checking KL/clip behavior.

Do not jump directly to the shipped 4,096-node 6-max NLHE run. The correctness
gates are now resolved, but the broad postflop topology and value-gradient
scale remain uncalibrated. Use the 512/1,024/2,048-node ladder first.

## Lower-priority multi-GPU analysis

- The current manual gradient synchronization performs one `all_reduce` per
  parameter. Replace it with bucketed/flattened reduction, or route forwards
  through DDP properly and use `no_sync()` during physical microbatch
  accumulation.
- Aggregate metrics across ranks or log rank-local topology and completion
  metrics explicitly. Rank zero's tree is not a node-wide summary.
- Document whether minibatch size is per-rank or global and tune learning rate
  accordingly. It is currently per-rank.
- Avoid simultaneous extension compilation on every rank; prebuild once or
  use a coordinated cache.
- Retain independent per-rank trees and RNGs. Do not merge duplicate public
  states across ranks in a way that shares descendants, profiles, or
  completion draws.
- Longer-term rollout/completion pipelining is plausible, but only with
  immutable iteration buffers and an unchanged synchronous rollout policy.

### Multi-GPU mathematical question

`core_math.tex` specifies one sampled branched tree. With multiple GPUs, should
the desired finite-sample objective be:

1. the equal average of independently normalized one-tree objectives, which is
   what the current rank gradient average implements; or
2. one pooled forest decision law, where a rank/tree is weighted by its total
   path-mass-weighted player-decision count?

These are not generally identical finite-sample objectives. Preserve equal
one-tree averaging unless the pooled-forest extension is explicitly intended.

## Lower-priority portability and modularity analysis

- Make a retained decision example own immutable encoded public inputs,
  beliefs, and folded summaries rather than backend state handles and mutable
  caches.
- Replace the hard-coded game-factory mapping with a registry or package entry
  points.
- Port the synthetic rollout backend to arbitrary Torch devices. It currently
  forces CPU even when the trainer selected CUDA.
- Give completion/evaluator backends a formal workspace-estimation method so
  generic training does not infer backend memory from output shapes.
- Record the sibling evaluator version, extension build hash, CUDA toolkit,
  and relevant compiler information in run metadata.
- Add a small CLI for exact synthetic exploitability curves; the implementation
  currently exists only as a Python API.
- Update the audit status documents after the correctness fixes. The prior
  statement that dropping folded full ranges passed a storage gate did not
  test whether the reconstructed policy input was identical.

## Final state

- The requested correctness, overflow, memory, KL, and synthetic CUDA changes
  are implemented in source, tests, contracts, and new smoke configurations.
- Normal, sequential-RB, and branched-RB share the same reference-policy KL
  definition, with mode-appropriate empirical integration.
- Bounded synthetic and poker CUDA runs passed their identity and memory gates.
- GPU 1 returned to its original 9 MiB idle state.
- The unrelated workload on GPU 0 was not touched.
