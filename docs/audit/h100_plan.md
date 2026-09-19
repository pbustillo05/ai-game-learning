# H100 heads-up replica-training implementation and selection plan

Date: 2026-08-04
Status: final proposal; no production or theory change is made by this document.

## Decision

Build one exact learner-replay benchmark and all credible replica executors
before renting the H100:

1. replace lane-major learning with a step-major cohort scheduler;
2. replace the per-lane host-synchronizing Adam loop with one device-vectorized
   replica optimizer;
3. implement three competing cohort executors behind the same learner-loss
   boundary:
   - a low-risk CUDA-stream wave executor using the ordinary lane model;
   - a `vmap(functional_call)` forward/loss followed by ordinary outer
     backward;
   - an explicit replica-native BMM model path, with padded BMM and jagged
     `grouped_mm` alternatives for unequal tails;
4. prepare eager, compiled, and CUDA-Graph-capable forms of every viable
   executor, plus the profiling and selection scripts.

The current serial lane executor remains the correctness oracle. The H100 is
used only to execute, profile, and select among already-implemented paths. No
implementation, debugging by design, or code-generation work is scheduled on
the rental clock. `torch.compile` and CUDA Graphs remain execution competitors,
not substitutes for the scheduler and state-layout work.

Separately, prepare one heads-up-only architecture ablation that removes the
incidence residual and folded-player machinery. This is not mixed into the
execution comparison. It is tested only after an execution winner exists, and
Normal and Rao--Blackwell (RB) must use the same selected architecture for the
eventual scientific A/B comparison.

The winner is the implementation with the highest complete-iteration fresh
player-nodes/s, subject to exact work identity, lane isolation, BF16 numerical
gates, Flash-SDPA dispatch, stable memory, and a minimum 10 GiB device reserve.
Learner presentations/s and model tokens/s diagnose that end-to-end result;
neither replaces it.

## Scope and non-goals

The primary workload is full-deck, 32 bb, two-player sequential Normal or RB
training on one H100. Each replica is an independent experiment with its own:

- FP32 master parameters and dense FP32 gradients;
- Adam/AdamW moments, step count, clipping decision, and loss-scaler state;
- rollout, learner, and completion RNG streams;
- sampled presentations, targets, reference state, metrics, and checkpoint;
- transactional rollback state.

There is no cross-replica gradient reduction, normalization, RNG sharing, or
optimizer coupling. Summing independent scalar losses is allowed only because
each loss depends exclusively on its own parameter lane.

This plan does not attempt to optimize branched training, distributed NCCL
training, MPS, or multi-way poker. The generic cohort scheduler and optimizer
should avoid a gratuitous two-player restriction, so multi-way may inherit
replica execution when shapes match. No implementation time should be spent on
multi-way-specific batching until the heads-up winner is known.

The existing mathematical objective, sampling law, and target construction are
fixed throughout the execution comparison. Removing incidence/folded features
is explicitly a separate model ablation. No file under `docs/theory/` is to be
changed as part of this work.

## What the three idea documents get right

The shared central idea is correct: the H100 should see independent model work
as a visible replica axis rather than receive one tiny learner at a time.
Parameters conceptually have shape `[K, ...]`, activations `[K, B, N, D]`, and
optimizer state `[K, ...]`. Replica-specific linear work is a batch or group of
independent GEMMs, while SDPA can fold the independent replica and presentation
axes into one ordinary attention batch after Q/K/V projection.

The strongest retained ideas are:

- step-major rather than lane-major scheduling;
- a cheap `vmap` feasibility prototype, without nesting `grad_and_value`;
- ordinary outer backward on a vector of independent losses;
- explicit BMM and grouped GEMM as first-class measured competitors;
- one replica-local, vectorized optimizer transition;
- static physical batch-size classes followed by compilation or graph replay;
- measured state lifetimes rather than admission from full-session peak memory;
- strict replay, lane-isolation, failure-atomicity, and checkpoint gates;
- Nsight Systems for scheduling and Nsight Compute for kernel facts;
- deletion of losing execution implementations after their evidence is
  archived.

## Corrections required before using the ideas

The numerical model in `h100_idea_1.md` and `h100_idea_2.md` is illustrative,
not the current paid workload. Its ceilings and target rates must not become
acceptance criteria.

| Item | Idea-document assumption | Repository/H100 workload fact | Consequence |
| --- | --- | --- | --- |
| Width | `d_model=256` | `d_model=128` | GEMM shapes, parameter counts, and saturation point differ. |
| Heads | often 4 at width 256 | 4 at width 128 | The active head width is 32, not 64. |
| Layers | 4--6 common blocks | 4 preflop and 6 postflop, in separate networks | FLOPs must be counted per partition. |
| Model form | ordinary 4x FFN transformer | wide side-input FFNs, incidence contractions, poker heads, public/history construction | Standard-transformer MFU formulas materially miscount work. |
| Master state | BF16 parameters assumed in one estimate | 11,134,238 FP32 master parameters | Persistent bytes per replica are much larger than a BF16-weight estimate. |
| Preflop sequence | 384 | conservative production maximum 536 | Use actual runtime `N`, not 384. |
| Postflop sequence | 2688 | current full-row maximum 2855 | The proposed fixed 2688 shape does not match the model. Keep one full-width `post_2` execution family. Never subdivide it by flop, turn, or river. |
| Pre/post mix | assumed 60/40 | varies by seed, rollout, mode, and whether one counts rollout calls or learner presentations | Weight results from the replay artifact, never a hard-coded mix. |
| Readout | `true_hand_only` assumed available | current model emits all 169/1326 actor rows before the sampled learner gathers one | It is a separate optimization, especially for Normal. |
| Replica learner | implicitly parallel | current production learner is lane-major and serial | Existing K>1 runs do not measure replica learner parallelism. |

For the production model configuration, the two networks contain 11,134,238
parameters total:

| Network | Parameters |
| --- | ---: |
| Preflop | 4,531,771 |
| Postflop | 6,602,467 |
| Total | 11,134,238 |

One FP32 parameter vector is 44,536,952 bytes. Current persistent learner and
reference ownership is approximately 356,295,616 bytes (0.332 GiB) per
replica before small buffers: parameters, dense gradients, two Adam moments,
parameter/moment rollback snapshots, and frozen reference parameters. This is
the appropriate starting point for a measured memory model.

The existing H100 evidence proves BF16 K=1 execution, not an optimized learner.
The retained audit reports approximately 42.13 Normal and 41.12 RB fresh
player-nodes/s with peak reserved memory no greater than 16.38 GiB. The old K=2
timings included a now-deleted duplicate learner pass and cannot estimate the
new design. The current code removed that duplicate pass but still runs every
lane's complete 116-step learner before starting the next lane.

Two current implementation facts are especially important:

1. `DenseAdam.step()` converts the finite check and norm to Python values. That
   forces at least two device-to-host synchronization points per lane per
   optimizer transition, or 232 such points per lane in a 116-step iteration.
2. `FunctionalReplicaInference` uses `index_select` on every stacked parameter
   for an active lane subset. The learner hot path must not copy 11.13M
   parameters merely to form a cohort. Full cohorts should use views, and tail
   cohorts need slice/pointer-based selection or a measured decision to execute
   a masked full cohort.

## ROI ranking

Measurement infrastructure is rank zero because no speed decision is valid
without it. Speed candidates are ordered below by expected full-iteration
return divided by engineering and correctness risk.

| Rank | Candidate | Expected return | Cost/risk | Decision |
| ---: | --- | --- | --- | --- |
| 0 | Exact learner replay, NVTX ranges, dispatch probe, work/FLOP ledger | Enables every valid decision | Low--moderate | Implement first. |
| 1 | Device-vectorized replica Adam, clipping, finite checks, and metric drain | Removes `O(K*steps)` host synchronizations and many tiny launches | Moderate; dense `[K,N]` state already exists | Implement before H100 rental. |
| 2 | Step-major cohort scheduler | Unlocks all replica concurrency while preserving lane sampler order | Moderate orchestration change | Implement before H100 rental. |
| 3 | Explicit replica-native BMM linears plus flattened SDPA | Directly exposes independent equal-shape weights to large batched kernels | High but bounded model/autograd work | Implement before H100 rental. |
| 4 | Vmapped forward/loss plus ordinary outer backward | Can obtain similar batching without duplicating model code | Moderate; Flash backward and autograd layout are uncertain | Implement before H100 rental. |
| 5 | CUDA-stream lane waves | Preserves the exact model/backward path and overlaps underfilled kernels | Low incremental risk after rank 2; still launches K graphs | Implement as the low-risk competitor/control. |
| 6 | Jagged `grouped_mm` versus padded BMM tails | Avoids either small tail cohorts or wasted padded work | Moderate; backward quality/performance uncertain | Implement behind the BMM primitive before rental. |
| 7 | `torch.compile` and CUDA Graph replay variants | Fuses pointwise work and removes residual launch gaps | Moderate; valuable after static cohorts exist | Prepare for every viable executor before rental. |
| 8 | Heads-up-lite model without incidence/folded machinery | Can remove substantial contractions, side features, parameters, and launches | Scientific architecture change | Implement separately before rental; test as a second factor. |
| 9 | `true_hand_only` Normal readout | Avoids full actor policy/value heads after the trunk | Low--moderate; mode-specific | Implement as an isolated switch before rental. |
| 10 | State-lifetime/cache streaming | May admit larger K, especially RB | Moderate; little speed if memory is not binding | Implement bounded last-consumer release where locally provable; measure K effect. |
| 11 | Custom Triton/CUTLASS persistent kernels | Potential gain for a proven bad dominant shape | Very high maintenance | Prepare only if local shape benchmarks expose a concrete target; never author on the rental clock. |

Ranks 1 and 2 are the common learner rewrite. Ranks 3--6 deliberately produce
several complete competitors: explicit batching, transformed batching,
ordinary concurrent dispatch, and two tail strategies. The added local
engineering cost is cheaper than stopping an H100 rental to implement the path
that profiling suggests should have been ready.

If the rank-0 dispatch probe shows that either production family is using math
SDPA, restoring a fused Flash forward and backward immediately becomes the
highest-priority speed task. Replica batching on top of a quadratic materialized
attention path would measure the wrong implementation.

## Implementation architecture

### 1. A frozen learner replay artifact

Add an in-memory/disk-serializable `LearnerReplayV1` containing, for each lane:

- initial model parameters and buffers;
- dense gradients, Adam moments, step counts, scaler, and rollback state;
- sealed sigma/action fields and their owner identities;
- the exact 116 logical steps of sampled indices generated from saved lane RNG
  state;
- family, physical slice boundaries, objective weights, and action spans;
- targets and posterior fields;
- reference outputs, or an immutable reference-cache identity plus its data;
- expected per-step work counters and final lane RNG state.

Artifact construction is outside the timed range. The replay must be usable by
serial, stream-wave, vmap, and explicit-BMM executors without changing
or repacking semantic data. It should support a single-family subset for kernel
diagnosis and a complete mixed 116-step replay for selection.

The replay generator first samples all lanes using their existing generators
in their existing lane-local order. Execution may then transpose the ready work
without changing what any lane sampled. An inactive/tail lane performs no RNG
operation.

### 2. Split loss production from execution without creating another loss

Refactor `LearnerOperation.forward_loss_backward()` into one authoritative
internal operation that returns a scalar loss and detached `LossResult` fields:

```text
authoritative_forward_loss(batch, controls) -> (scaled_loss, metrics)
```

Then provide thin executors:

```text
serial:      loss = authoritative_forward_loss(...); loss.backward()
stream wave: invoke the same function/backward in lane streams
vmap:        losses = vmap(authoritative_forward_loss)(...); losses.sum().backward()
```

There is no replica loss class, replay-specific policy adapter, alternate
posterior reduction, or copied model definition. All loss controls remain
explicit FP32. `losses.sum()` must not be changed to a mean: averaging would
scale every independent gradient by $1/K$.

### 3. Step-major scheduling

Replace:

```text
lane 0: logical steps 0..115
lane 1: logical steps 0..115
...
```

with:

```text
for logical step s in 0..115:
    materialize the already-sampled physical work for every ready lane
    partition it only by the required pre/post_2 model family and physical
    batch-size class
    execute compatible lane cohorts
    finish all backward contributions for step s
    perform one vectorized independent optimizer transition
```

Within a lane, optimizer transition $s$ still precedes forward $s+1$. Across
lanes, the work is independent.

Family counts differ naturally between lane-local pooled samples. Exact row
count equality must not be required for all work, or cohort occupancy will
collapse. Use this policy:

1. split each pre/post family group into fixed full physical chunks, initially
   testing 16, 32, 64, and 128 rows where memory admits them;
2. cohort the full chunks across lanes;
3. place tails into a small set of rounded physical batch-size classes;
4. compare zero-loss padding within a tail class against a smaller cohort;
5. padding never creates a statistical presentation, target, RNG draw, metric
   mass, or optimizer contribution, and useful/physical work is reported
   separately.

This permits a measured amount of padding. A categorical ban on padding is not
appropriate for an absolute-throughput project; the semantic ban is on treating
padding as data. Retain padding only if useful presentations/s improves.

The existing iteration-wide transaction remains the publication boundary.
Any lane/rank failure restores all lane parameters, buffers, moments, versions,
scalers, RNG states, and run states.

### 4. Flat replica parameter and gradient ownership

Preserve one contiguous FP32 parameter row and gradient row per replica:

```text
parameters: [K, N_parameter]
gradients:  [K, N_parameter]
exp_avg:    [K, N_parameter]
exp_avg_sq: [K, N_parameter]
```

Named model parameters are strided views into the flat owner. The vmap and
explicit-BMM paths must accumulate directly into the flat gradient owner or
prove that the cost of one gradient pack is worthwhile. They must not allocate
an unaccounted second full gradient set.

Full cohorts use parameter views directly. Do not `index_select` every named
parameter. For noncontiguous tail lane sets, compare:

- separate contiguous lane runs;
- pointer/group metadata (`grouped_mm` candidate);
- a masked full-K tail;
- an explicit state gather, charged in bytes and time.

The last option is acceptable only if it wins end to end.

### 5. Device-vectorized optimizer

Implement one group transition over `[K,N_parameter]`:

- per-lane finite mask `[K]` entirely on device;
- FP32 squared-norm reduction over dimension 1;
- per-lane norm and clip scale `[K]`;
- per-lane enabled/applied mask;
- per-lane step count and bias correction;
- masked Adam/AdamW moments and parameter updates;
- explicit-zero gradient semantics for every unused parameter;
- one cohort-level metric drain after all steps, rather than `bool()` and
  `float()` per lane/step.

For BF16 training the master parameters, gradients, controls, and moments stay
FP32. Model autocast remains BF16. No optimizer state is silently lowered to
BF16 or FP8.

Begin/commit/rollback snapshots should also use stacked copies rather than a
Python loop over lane optimizers. Distributed SUM, if later used, is still
lane-addressed and never reduces across K.

### 6. CUDA-stream wave executor

This executor uses one bounded stream per active lane wave, not an unbounded
stream per configured replica. It launches the unchanged authoritative model,
loss, and backward for compatible work, records completion events, joins once,
and then calls the vector optimizer.

Its value is diagnostic as well as practical:

- it preserves ordinary Flash-SDPA dispatch and the ordinary autograd graph;
- it requires no transformed operator batching support;
- it reveals whether concurrent independent kernels alone fill the H100;
- it remains a robust competitor if vmap loses Flash backward.

Its weakness is K-fold Python/dispatcher and kernel-launch work. If Nsight
shows CPU submission gaps after the streams are active, vmap/graphs have a
clear target.

### 7. Vmap/outer-backward executor

For one compatible cohort:

```python
losses, metrics = vmap(one_lane_forward_loss)(
    parameter_views,
    buffer_views,
    stacked_batch,
    stacked_controls,
)
losses.sum().backward()
```

Random inputs are materialized before vmap. Model dropout is currently zero;
any future stochastic model operation must use explicitly independent lane RNG.

The first H100 dispatch gate determines whether PyTorch 2.12.1:

- lowers lane-specific `nn.Linear` operations to useful batched GEMMs;
- folds `[K,B,H,N,Dh]` to an ordinary effective SDPA batch;
- retains cuDNN SM90 Flash forward and backward;
- avoids functorch batching fallbacks and parameter copies;
- writes independent gradients to the intended flat rows.

Failure of any one condition rejects vmap as the production executor but does
not reject stream waves or explicit BMM.

### 8. Explicit BMM/grouped-GEMM executor

Build and validate this executor before renting the H100. It is a first-class
competitor, not a contingency to author after vmap profiling. The implementation
may share input construction, loss, output contracts, parameter ownership, and
tests with the canonical model, but the trainable linear/attention execution
must carry an explicit K axis.

A deliberately separate replica-native model implementation is acceptable
here. Avoiding temporary model duplication is less important than arriving at
the H100 with the direct BMM design ready. What may not be duplicated is the
sampling law, policy composition, loss, targets, optimizer semantics, or
checkpoint meaning. If BMM loses, delete the replica-native model after
archiving its patch; if it wins, make it the sole production heads-up model path
and retain the ordinary model only as the narrow correctness oracle.

The direct path needs explicit primitives for:

- replica-specific embedding gather from `[K,V,D]` tables;
- replica LayerNorm: normalize over D, then apply `[K,D]` affine parameters;
- separate-Q/K/V BMM and a second fused-QKV parameter-layout variant;
- attention output and every FFN/head linear via BMM;
- `[K,B,H,N,Dh] -> [K*B,H,N,Dh]` Flash SDPA;
- incidence contractions folded over `K*B` without changing their equations;
- exact lane-wise import/export between canonical and fused parameter layouts.

The fused-QKV variant stores one `[K,3D,D]` weight and one `[K,3D]` bias so it
does not concatenate weights during the timed step. Checkpoint conversion splits
or joins Q/K/V in a fixed order and proves lane-wise output/gradient/update
parity. Separate and fused QKV are both benchmark cells because one fewer launch
can matter at this width, while the fused dW shape may behave differently.

For equal row counts:

```text
X: [K, B*N, Din]
W: [K, Dout, Din]
Y = bmm(X, W.transpose(1,2))
```

After replica-specific Q/K/V projection:

```text
[K,B,H,N,Dh] -> [sum_K B, H, N, Dh] -> one Flash-SDPA call
```

For unequal tails, benchmark jagged `torch.nn.functional.grouped_mm` against
padded BMM. Forward, activation-gradient, weight-gradient, and full learner
must all win; a forward-only microbenchmark cannot select the primitive.

Do not mechanically replace incidence `einsum`s. Profile their generated
kernels and total share. If necessary, batch the same exact contraction across
`K*B`; only then consider a specialized scatter/gather kernel.

### 9. Compilation and CUDA Graphs

Prepare and locally validate eager plus these compile modes for every viable
executor:

- `default`;
- `reduce-overhead`;
- `max-autotune`;
- `max-autotune-no-cudagraphs`.

Record compile/autotune time separately. Reject modes with graph breaks in the
steady learner body or repeated recompilation.

Prepare graph-safe wrappers locally for all viable executors. If Nsight Systems
shows more than 3% unexplained launch gaps, run the already-prepared capture for
each retained `(executor, K-cohort, B-class, family)`
shape. Static input/output buffers are CUDA-resident. The captured region may
include forward, loss, backward, clipping, Adam, and gradient reset. Sampling,
artifact I/O, checkpoint I/O, and iteration transaction setup remain outside.

Do not build a general graph scheduler. The two model families times the small
selected set of common K/B classes is sufficient.

### 10. Heads-up-lite architecture ablation

Implement the current-architecture executors first, then add a heads-up-specific
model under `pbg/games/heads_up/` or a clean, explicit model variant—not
conditionals scattered throughout the current multi-way model. Both are ready
before the rental, while the architecture result remains separate from the
execution result.

For two players there are no folded players at a decision. The ablation may
remove:

- folded-player summaries, pressure, beta factors, and folded seat heads;
- incidence residual projections and contractions;
- the three incidence-derived dynamic side tensors in each layer;
- side-input width that exists only to carry those tensors;
- unused multi-way seat padding and branches.

This is potentially larger than deleting the named incidence modules alone.
Those modules are about 7.3% of preflop parameters and 7.5% of postflop
parameters, while incidence-derived side width also makes the layer FFNs—the
largest parameter groups—substantially wider.

Keep full bidirectional attention, key-side log-reach prior, public/history
tokens, action anchors, and the same policy/value output contract. Normal and
RB must share one model definition. Architecture performance is compared at
K=1 and at the selected K; learning-quality comparison is a later scientific
gate, not inferred from tokens/s.

If this variant wins throughput but has not passed a meaningful learning-quality
study, it remains an experimental model and does not replace the current poker
model. The project goal is a clean Normal-versus-RB experiment, not a claim that
the lite architecture is the strongest poker agent.

## Measurement contract

### Timed boundaries

Use three nested, non-overlapping reported boundaries:

1. `learner_core`: staged inputs ready through completed optimizer transition;
2. `learner_total`: gather/staging through release of cohort temporaries;
3. `iteration_total`: rollout start through learner completion and publication.

CUDA events measure device intervals. Synchronized host wall time measures the
corresponding end-to-end boundary. Artifact load, compile/autotune, profiler
startup, checkpoint serialization, and report writing are excluded from
steady-state throughput and reported separately.

### Exact work metrics

For every result report:

$$
\text{statistical presentations/s}
=\frac{\sum_k \text{nonpadding learner rows}_k}{t_{\text{learner total}}},
$$

$$
\text{replica optimizer transitions/s}
=\frac{\sum_k \text{applied transitions}_k}{t_{\text{learner total}}},
$$

$$
\text{physical model tokens/s}
=\frac{\sum_c B_c N_c}{t_{\text{forward+backward}}},
$$

$$
\text{useful model tokens/s}
=\frac{\sum_c \text{valid nonpadding tokens}_c}{t_{\text{forward+backward}}},
$$

$$
\text{attention token-pairs/s}
=\frac{\sum_c B_c L_c N_c^2}{t_{\text{forward+backward}}}.
$$

Also report padding efficiency, presentations per applied transition, per-lane
transition latency, fresh player-nodes/s, rollout seconds, completion seconds,
target seconds, reference-cache build/hit seconds, and learner share of the
iteration.

For a full configured sequential replay, expected statistical work is
`K * 116 * 256 = K * 29,696` presentations. The report must reconcile this
identity rather than infer work from elapsed time.

### FLOP accounting and MFU

Do not reuse the standard `72ND^2 + 12N^2D` estimate. Build an exact static
algorithmic ledger from the executed model configuration and actual shapes:

- every linear forward, activation-gradient, and weight-gradient GEMM;
- QK and probability-value products per attention layer;
- incidence contractions;
- poker side FFNs and all policy/value heads;
- gather/scatter operations counted as bytes, not invented FLOPs.

For a dense linear with dimensions `(M,K,N)`, count `2MKN` per GEMM and record
forward, dX, and dW separately. Save the ledger beside each artifact so the
reported FLOPs are auditable.

Report:

- algorithmic model TFLOP/s;
- eligible Tensor-Core TFLOP/s;
- estimated dense BF16 MFU using the dense peak for the exact rented H100 SKU;
- padding-adjusted useful MFU;
- shape efficiency versus an isolated benchmark with identical shape, stride,
  dtype, transpose, and primitive.

MFU is a model-based diagnostic. It is not “Tensor Core utilization.”

### Hardware utilization

Use Nsight Systems for the complete three-step schedule and Nsight Compute on a
bounded representative kernel set. Query metric names from the installed tool
version and save the resolved list; do not assume names are identical across
Nsight releases.

Required hardware facts are:

- SM active/throughput percentage;
- Tensor-pipeline active cycles as a percentage of peak active cycles;
- achieved Tensor-Core instruction/FLOP rate and percent of dense SKU peak;
- achieved occupancy and active warps/SM;
- DRAM bytes/s and percent of measured sustainable bandwidth;
- L2 hit rate and L2 bytes;
- issue-stall breakdown for dominant kernels;
- SM/HBM clocks, board power, temperature, throttling, ECC, and MIG mode.

“Tensor active” and “achieved Tensor TFLOP/s” must both be present: a pipeline
can be active frequently while issuing inefficient shapes. `nvidia-smi`
utilization is telemetry only and is never accepted as a performance claim.

### Kernel and scheduling metrics

NVTX ranges cover gather, each model-family forward, reference forward,
loss, backward, finite/norm/clipping, optimizer, reset, staging, and release.
For each range report:

- GPU time and synchronized wall time;
- kernel launches per replica transition and per 1,000 presentations;
- unioned GPU-kernel-active time and active fraction;
- launch gaps over 100 microseconds and over 1 millisecond;
- CPU submission time and device synchronizations;
- GEMM/BMM/grouped-GEMM time split into forward/dX/dW;
- Flash forward/backward time and actual kernel names;
- LayerNorm, incidence, elementwise, concat/zero/copy, reduction, loss, and
  optimizer time;
- H2D, D2H, D2D, and parameter-gather bytes.

The transformed executors must show no per-lane Python model call inside a
purported batched cohort and no functorch batching-fallback warning.

### Memory metrics

At transaction start, cohort entry, after forward, after backward, after
optimizer, after temporary release, and iteration end, record:

- allocated and reserved bytes;
- peak allocated and reserved bytes;
- inactive/non-releasable allocator bytes;
- parameters, buffers, gradients, two moments, rollback snapshots, and frozen
  reference bytes;
- sealed inputs, gathered inputs, targets, reference cache, activations,
  SDPA/GEMM workspace, compiler workspace, and graph-private-pool bytes;
- end-of-step monotonic growth slope.

Fit both modes to:

$$
M(K,B)=M_0+K M_{\mathrm{resident}}+KB M_{\mathrm{row}}+M_{\mathrm{class}},
$$

and publish residuals and held-out prediction error. Use the fit for planning
only if maximum held-out error is at most 15%; otherwise use the measured upper
envelope. Admission requires the configured 10 GiB reserve plus at least 2 GiB
of measured extra headroom, with no OOM and no positive memory-growth trend
across 116 steps.

## Correctness and scientific gates

For K in `{1,2,4}` and both Normal and RB:

1. Inputs, sampled indices, family order, targets, posterior/objective
   mass, reference values, initial state, and final lane RNG state are exact.
2. Changing only lane `j` inputs or parameters leaves every output, loss,
   gradient, moment, transition, and RNG state in `k != j` unchanged.
3. FP32 one-step outputs, losses, gradients, parameters, and moments pass
   `rtol=1e-5, atol=1e-7` against serial.
4. BF16 outputs/losses pass `rtol=5e-3, atol=5e-4`; dense gradient relative L2
   error is at most `5e-4`; gradient cosine similarity is at least `0.999999`;
   and one-step maximum parameter error is at most `1e-6` at learning rate
   `5e-5`.
5. A complete 116-step replay applies every expected transition, keeps sample
   and RNG identity exact, has no nonfinite values, and finishes with per-lane
   parameter relative L2 error at most `2e-3`.
6. An intentionally unused parameter has an explicit zero gradient and the
   same dense Adam/AdamW transition as the independent dense oracle.
7. Injected failures before backward, after backward, and before publication in
   every lane restore parameters, buffers, moments, versions, scaler, RNG, and
   run state exactly.
8. Save/resume after a batched transition reproduces the next samples, losses,
   gradients, and transitions within the same gates.
9. cuDNN SM90 Flash forward and backward remain selected for real preflop and
   postflop batches. Any math-SDPA fallback is a stop condition.

Do not loosen a threshold after seeing a candidate. If an operation cannot
physically meet one, isolate it, derive an independent error bound, and amend
the gate before rerunning all candidates.

The heads-up-lite model cannot be parameter-parity-tested against the current
architecture; it instead needs shape/contract tests, finite training,
loss-response tests, checkpoint tests, and a separate learning-quality
experiment before scientific use.

## Benchmark protocol

Use an exclusive, non-MIG H100. Record the exact SKU, UUID, driver, firmware,
PyTorch/CUDA/cuDNN/Triton versions, evaluator revision, code SHA/dirty patch,
power limit, clocks, persistence mode, ambient contention, and profiler
versions. Confirm the evaluator and extensions are compiled for SM90.

For unprofiled timing:

- warm until compile, autotune, allocator, and graph capture are complete;
- then run at least 100 optimizer transitions and at least 10 seconds per
  repeat;
- collect at least seven repeats per comparison;
- alternate or randomize matched candidate/baseline order in an ABBA-style
  schedule;
- synchronize immediately before and after the timed range;
- report median, minimum, maximum, MAD, coefficient of variation, paired
  speedup, and a bootstrap 95% confidence interval for the paired speedup;
- reject and diagnose a timing set with coefficient of variation above 3%;
- never mix profiler-perturbed timings into the unprofiled result.

All aggregate speedups compare the same K, total presentations, pre/post
mix, B classes, optimizer transitions, and initial replay. K=4 candidate work
must be compared with serial execution of those same four lanes, not K=1.

## Pre-rental exit gate

Do not start the H100 rental until all of the following are true:

- serial, stream-wave, vmap, and padded-BMM executors pass local CUDA FP32
  one-step, 116-step replay, rollback, checkpoint, and lane-isolation tests;
- the explicit BMM path covers every trainable linear in preflop and postflop,
  including forward, dX, and dW, and folds K into the SDPA batch explicitly;
- the grouped-GEMM path is complete behind the same primitive, has shape and
  gradient tests against padded BMM, and is cleanly runtime-gated when the local
  SM75 device cannot execute its SM80+ kernel;
- eager/compiled/graph wrappers, static buffers, warmup, and reset behavior are
  implemented rather than sketched;
- the exact replay artifacts for Normal and RB are generated and hash-checked;
- the one-shot matrix runner has been dry-run in no-GPU/list mode and on the
  local CUDA subset, including resume after an injected timeout;
- every H100 command, profiler invocation, output path, timeout, acceptance
  check, and automatic skip dependency is in the checked-in runbook;
- the rental environment or container is built, version-locked, and preflighted
  as far as possible without SM90;
- no open TODO in the runbook requires source editing on the rental machine.

The local RTX 2080 Ti cannot predict SM90 performance and cannot execute every
modern grouped-GEMM kernel. That is a reason for the Stage-0 H100 smoke, not a
reason to postpone implementation until the rental.

## H100 experiment matrix

Everything below is driven by one checked-in matrix runner. It writes an atomic
result after every cell and can resume by artifact identity. Each cell already
has its config, command, warmup, timeout, expected work, memory ceiling, and
failure policy before the rental starts.

### Stage 0: rental preflight and five-minute smoke

1. Record hardware/software identity and prove exclusive non-MIG ownership.
2. Verify SM90 extensions and evaluator loading.
3. Run one tiny preflop and one tiny full-width postflop forward/backward for
   serial, stream-wave, vmap, padded-BMM, and grouped-GEMM executors.
4. Prove Flash forward/backward kernel selection for each surviving executor.
5. Verify every predeclared Nsight command and result directory.

A failed cell is disabled by the runner with its error artifact retained. The
operator may fix environment/configuration mistakes such as a missing profiler
permission, but does not edit implementation code on the rental.

### Stage 1: correctness and dispatch

For both Normal and RB:

1. K=1 serial oracle for preflop and full-width postflop.
2. K=2 one-step tests for stream-wave, vmap, padded BMM, and grouped GEMM.
3. K=2 complete 116-step replay for each executor.
4. Forward, dX, and dW microbenchmarks for every major linear shape.
5. Padded BMM versus grouped GEMM on saved real tail distributions.

Run eager first, followed by each executor's prebuilt compiled form. Run the
prebuilt CUDA-Graph form only when its eager/compiled parent passes. No speed
candidate advances without correctness, lane isolation, flat-gradient
ownership, and Flash dispatch.

### Stage 2: broad throughput sweep

For serial, stream-wave, vmap, padded BMM, and grouped GEMM, including viable
compiled/graph forms:

1. sweep K in `{1,2,4,8}`;
2. sweep full physical B in `{16,32,64,128}` when admitted;
3. benchmark preflop, full-width postflop, and the exact mixed replay;
4. record unprofiled timing and memory for every cell;
5. retain the fastest three executor/runtime combinations by mixed useful
   presentations/s, provided no family regresses catastrophically.

Initial executor gates remain:

- K=1 regression versus the matched serial path no worse than 3%;
- K=2 aggregate learner gain at least 25%;
- K=4 aggregate learner gain at least 75%;
- Flash forward/backward retained;
- all correctness gates passed.

These thresholds determine production eligibility, not whether raw measurements
are collected. The broad matrix is intentionally collected in one rental so a
surprising SM90 result does not require another engineering/rental cycle.

### Stage 3: finalist plateau and profiling

For the fastest three combinations:

1. sweep K through `{1,2,4,8,16,32}` and K=64 where memory admits;
2. sweep fixed aggregate rows `K*B` in `{256,512,1024,2048,4096}`;
3. sample non-powers-of-two K and nearby B values around each plateau;
4. run bounded Nsight Systems captures on three steady steps;
5. run Nsight Compute on representative forward, dX, dW, Flash forward,
   Flash backward, incidence, and optimizer kernels;
6. compare full-learner kernel performance with the exact-shape isolated
   ceilings gathered in Stage 1.

This stage answers whether explicit BMM actually improves the complete learner,
whether grouped GEMM earns its tail complexity, and whether compilation/graphs
remove real gaps rather than simply moving work between kernels.

### Stage 4: complete Normal/RB selection

For the best settings of each finalist:

1. run at least seven alternating complete 116-step Normal replays;
2. run at least seven alternating complete 116-step RB replays;
3. run the memory scaling and held-out-fit matrix;
4. run full real iterations, including rollout, completion, targets, reference
   work, learner, transaction publication, and release;
5. select mode-specific K/B when resident RB state changes the optimum.

The production scheduler is retained only if the best admitted K improves
complete-replay aggregate learner throughput by at least 50% for both modes and
improves or preserves complete-iteration fresh player-nodes/s.

### Stage 5: architecture and readout factors

Using the top one or two executors, all already implemented:

1. compare current and heads-up-lite models at K=1 and selected K;
2. compare full readout with `true_hand_only` for Normal;
3. keep identical Normal/RB output contracts and training schedules wherever
   the objective requires the same domain;
4. run short finite-training and loss-response sanity tests;
5. reserve a separate learning-quality experiment before adopting the lite
   model for the scientific Normal-versus-RB comparison.

Do not use a faster architecture's tokens/s to claim an executor is faster.
Report architecture, readout, and execution effects as separate factors.

## Retain/reject rules and cleanup

Every implementation item is one commit with:

- commit and parent SHA;
- exact tests and benchmark commands;
- correctness, dispatch, timing, kernel, and memory artifacts;
- raw JSON/Nsight paths and SHA-256 hashes;
- retained/rejected decision and reason;
- exact `git revert <sha>` command.

Simple contained changes need at least 5% paired full-replay gain unless they
remove a correctness/capacity defect. A new executor needs at least 10% gain
over the next-best correct executor. A custom kernel needs at least 10%
credible full-step upside before implementation and must deliver at least 5%
measured full-replay gain to remain.

After selection:

- keep the serial executor as a test/debug oracle;
- keep exactly one production cohort executor;
- delete losing executor code, config switches, tests that only exercise dead
  implementations, and stale CLI paths;
- retain patches and raw evidence under the experiment audit directory;
- do not delete the generic poker model merely because a heads-up-lite
  experiment is faster; that requires its separate scientific adoption gate.

## Proposed commit sequence and likely files

1. **Replay and metrics**
   - `pbg/core/learner.py`, `pbg/core/training.py`, `pbg/benchmark.py`;
   - a focused learner benchmark module/CLI;
   - replay, work-ledger, dispatch, and failure-injection tests.
2. **Full-width capacity correction**
   - make learner admission account for the actually executed 1,326-hand
     postflop width and maximum sequence 2,855;
   - add a test that derived capacity equals runtime sequence geometry.
3. **Vector optimizer**
   - `pbg/core/optimization.py`, `pbg/core/replica.py`;
   - dense-oracle, inactive-lane, rollback, and checkpoint tests.
4. **Step-major scheduler and stream waves**
   - `pbg/core/training.py`, `pbg/core/iteration.py`, `pbg/core/replica.py`;
   - sampler-order, logical-step, tail, and atomic-failure tests.
5. **Vmap outer backward**
   - the same authoritative loss boundary plus one executor module;
   - no replica-specific loss/model copy;
   - dispatch and flat-gradient ownership tests.
6. **Explicit BMM/grouped backend**
   - replica embedding, LayerNorm, linear, attention, FFN, head, and incidence
     primitives;
   - padded equal-shape BMM and jagged grouped-GEMM tails;
   - separate-Q/K/V and persistent fused-QKV parameter-layout variants;
   - flattened ordinary Flash SDPA;
   - exact canonical checkpoint conversion and no duplicated poker loss or
     policy logic.
7. **Compile/graphs for every executor**
   - static wrappers and a bounded graph cache prepared before H100 rental;
   - graph-count, no-recompile, capture-safe optimizer, and replay tests.
8. **Heads-up-lite and readout ablations**
   - preferably `pbg/games/heads_up/` with the existing queued input/output
     contract;
   - separate architecture and learning-quality configs under
     `configs/experiments/`.
9. **One-shot H100 matrix runner**
   - predeclared cells, commands, timeouts, profiler invocations, atomic
     progress, resume identity, and automatic variant disabling;
   - no rental-stage source edits in the runbook.

All experimental configs and outputs follow repository run-folder hygiene and
remain under `configs/experiments/` and `runs/experiments/`.

## Final recommendation

The most likely production winner is:

```text
full-width preflop/postflop fixed-chunk cohorts
  + explicit replica-native BMM
  + flattened Flash SDPA
  + flat device-vectorized Adam
  + compile or a small static CUDA-Graph cache
```

That is the most controllable likely winner, not the conclusion of the paid
test. Vmap/outer-backward, padded BMM, grouped-GEMM tails, and CUDA-stream waves
are all implemented alongside it. The H100, not architectural preference,
chooses among them. The heads-up-lite/no-incidence model has potentially large
additional throughput value, but it answers a different question and must not
contaminate the execution A/B.

This ordering spends local engineering on changes that are independently
useful, arrives at the rental with every credible parallel executor and all
scripts ready, spends the rental clock only on execution and profiling,
collects enough low-level evidence to explain the winner, and leaves the tree
with one production implementation rather than a permanent collection of H100
experiments.
