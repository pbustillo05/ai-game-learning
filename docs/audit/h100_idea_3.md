# Replica-batched H100 learner plan, 2026-08-04

Status: proposal only. No production code, configuration, mathematical
objective, MHA equation, sampling law, or model dimension is changed by this
document.

## Goal

Run the forward, loss, backward, and optimizer work of independent Normal or
Rao--Blackwell replica learners as genuinely batched H100 work. Replicas retain
independent parameters, gradients, Adam state, RNG, samples, reference state,
metrics, and checkpoints. Gradients are never summed between replica lanes.

The primary target is aggregate learner throughput. Replica batching does not
make one replica converge faster unless its training schedule is separately
changed, which is outside this plan.

## Evidence and corrected memory model

The retained H100 K=2 experiment did not test a parallel learner. It ran all
learner work for one lane and then all learner work for the second lane. NVTX
showed zero lane overlap. Replica rollout/reference inference used
`vmap(functional_call)`, while the learner used ordinary lane-local autograd.

An older implementation did use `vmap(grad_and_value(one_lane_loss))`. It was
removed because it forced math SDPA when fused backward lacked functorch
batching support and because it had accumulated a second replica-only loss and
replay implementation. The observed RB parity failure was ultimately localized
to a repeated-index BF16 preflop exact-to-class backward and loss-control dtype
drift, not to an impossibility of replica-batched differentiation. The current
learner keeps preflop RB work in class space and has one authoritative loss
operation, so the old result is not a reason to reject a new bounded prototype.

### Learner-only memory

The model has 11,134,238 FP32 parameters, or 44,536,952 bytes per parameter
vector. Current persistent learner/reference state per replica is approximately:

| State | Bytes per replica |
| --- | ---: |
| Parameters | 44,536,952 |
| Dense gradients | 44,536,952 |
| Two Adam moments | 89,073,904 |
| Parameter and two-moment rollback snapshots | 133,610,856 |
| Frozen reference parameters | 44,536,952 |
| **Total, excluding small buffers** | **356,295,616 (0.332 GiB)** |

Matched 512-root H100 runs increased peak allocation by approximately 0.666 GB
when the physical learner cap increased from 84 to 256 rows. Normal and RB had
nearly identical slopes. The empirical incremental activation estimate is:

$$
M_{\mathrm{active}}(K,B)
\approx K\left(0.332 + 0.00360B\right)\ \mathrm{GiB},
$$

where $B$ is the simultaneously active physical minibatch size per replica.
This is an estimate to validate, not an admission proof.

| $B$ per replica | K=16 | K=32 | K=64 |
| ---: | ---: | ---: | ---: |
| 32 | 7.2 GiB | 14.3 GiB | 28.6 GiB |
| 64 | 9.0 GiB | 18.0 GiB | 36.0 GiB |
| 84 | 10.2 GiB | 20.3 GiB | 40.6 GiB |
| 128 | 12.7 GiB | 25.4 GiB | 50.8 GiB |
| 256 | 20.1 GiB | 40.1 GiB | 80.3 GiB |

The full session has additional rollout, target, completion, reference-cache,
and allocator residency. At 512 roots the measured all-in approximation was:

$$
M_{\mathrm{Normal}}(K,B)
\approx 17.2 + K(0.603 + 0.00360B)\ \mathrm{GiB},
$$

$$
M_{\mathrm{RB}}(K,B)
\approx 17.2 + K(1.643 + 0.00360B)\ \mathrm{GiB}.
$$

The RB increment is predominantly retained lane data/cache state, not an
inherent transformer activation cost. The design should stream or release that
state at its last consumer instead of using full-session peak memory as a limit
on the batched learner. K=32 should be treated as a normal target; K=64 should
be tested at physical $B\in\{32,64,84,128\}$ rather than dismissed in advance.

## Proposed implementation, ordered by expected ROI

### 1. Vmapped forward/loss followed by ordinary backward

Prototype a different transform from the removed learner. Do not nest
`grad_and_value` inside `vmap`. For one compatible lane cohort:

```python
losses, metrics = vmap(one_lane_forward_loss)(
    stacked_parameters,
    stacked_buffers,
    stacked_batch,
)
losses.sum().backward()
```

Each `losses[k]` depends only on `parameters[k]`, so ordinary backward should
produce independent stacked gradients without a gradient transform or
cross-lane reduction. Split the existing authoritative learner boundary into a
loss-producing internal operation and two execution wrappers:

- ordinary lane: loss followed immediately by ordinary backward;
- replica cohort: vmapped losses followed by one ordinary backward.

Do not introduce a replica-specific policy, loss equation, replay pack, or
target conversion. Materialize stochastic inputs before the transform from the
existing lane-local generators.

First determine whether Torch 2.12 folds the hidden replica axis into an
ordinary cuDNN Flash-SDPA batch whose saved graph uses the normal Flash
backward. If it does, vmap may already generate batched GEMMs for lane-specific
linear weights and no handwritten model is required.

### 2. Transpose learner scheduling from lane-major to step-major

Replace:

```text
lane 0: steps 0..S-1
lane 1: steps 0..S-1
...
```

with:

```text
for logical step s:
    obtain the next physical work from every ready lane
    group by input type and compatible physical shape
    execute each compatible replica cohort
    finish each lane's independent optimizer transition
```

Preserve lane-local sampler order. A lane may enter a cohort only when its next
physical presentation has the same input family and tensor contract. Do not pad
inactive lanes merely to make a rectangular K axis. Unequal tails should form a
smaller cohort or a separate size bucket.

The existing iteration-wide transaction remains the publication boundary. A
failure in any lane restores every lane's parameters, moments, versions, RNG,
and run state exactly as today.

### 3. Batch gradient checks and Adam transitions

Use the existing dense `[K,N]` gradient and moment ownership as the physical
optimizer interface. Replace lane-loop finite checks, norms, clipping, and Adam
updates with replica-axis tensor operations:

- finite mask `[K]`;
- FP32 gradient norm `[K]`;
- clip scale `[K]`;
- independent step count and bias correction `[K]`;
- masked parameter/moment update `[K,N]`.

Keep explicit-zero gradient semantics for unused parameters. Metrics must stay
on the device until one cohort-level synchronization. Distributed SUM, when
used, remains lane-addressed and must not reduce across the K axis.

### 4. Use explicit BMM only where vmap dispatch is inadequate

If the first prototype falls back from Flash, produces excessive dispatch, or
does not improve throughput, implement a replica-native execution path for
trainable layers. A replica-specific linear is:

```text
X: [K, ..., Din]
W: [K, Dout, Din]
Y = bmm(X.reshape(K, -1, Din), W.transpose(1, 2)) + bias
```

After replica-specific Q/K/V projections, reshape attention from
`[K,B,H,N,Dh]` to `[K*B,H,N,Dh]` and call ordinary Flash SDPA. Reshape the
result back before the replica-specific output projection. Use the same
principle for MLP and head projections. LayerNorm and elementwise operations
carry an explicit leading K axis; they are not BMMs.

The existing incidence `einsum`s should not be mechanically renamed to `bmm`.
Compare generated kernels first. If needed, batch their existing exact
incidence contraction over `K*B`; a later exact postflop scatter/gather kernel
is a separate candidate.

### 5. Decouple active learner batches from retained replica state

Make K-scale admission depend on simultaneously active minibatches, not on
every replica's complete rollout/target/cache lifetime. In particular:

- gather only the next cohort's learner fields;
- release gathered staging and autograd graphs immediately after backward;
- bound reference outputs per cohort or cache only when measured reuse repays
  their resident bytes;
- release lane target/rollout fields at their declared last consumer;
- report common, per-resident-replica, and per-active-row bytes separately.

Do not hide transfers through CPU. All production staging remains CUDA-local.

### 6. Remove measured launch and allocation amplification

After replica batching is functional, address the operations that remain large
in the batched trace:

- repeated postflop `cat` and zero construction inside every layer;
- repeated BF16 copies/casts;
- small elementwise/reduction launches;
- avoidable host synchronizations in finite/norm/metric handling.

Preallocate or fuse only after an allocation trace identifies the owner and
lifetime. Do not retain a fusion unless it improves both Normal and RB or has a
clear mode-specific dispatch boundary.

### 7. Enable documented board-compatible postflop compaction

As a later memory-enabling implementation optimization, use only compatible
postflop hand rows and scatter outputs back to the declared full domain. The
documented fixed counts are 1,176 flop, 1,128 turn, and 1,081 river instead of
1,326. This must preserve the model equations and output contract. Treat it as
a separate commit because it changes physical sequence geometry and could
affect numerical order.

Removing the incidence residual, layers, heads, or changing model dimensions is
not part of this optimization plan. Such changes require a separate scientific
quality ablation.

## Validation plan

### A. Reproducible benchmark boundary

Create an in-memory learner replay artifact containing the sealed inputs,
sampled indices, targets, objective weights, reference outputs or cache
identity, initial parameters, optimizer state, scaler, and lane RNG states.
Time only the synchronized learner boundary; exclude rollout construction,
checkpoint serialization, profiler startup, and artifact I/O.

For unprofiled timing:

- two warmups and five measured repeats;
- at least 50 optimizer transitions per repeat, or enough repetitions of a
  shorter captured replay to exceed five seconds;
- `torch.cuda.synchronize()` immediately before and after the timed range;
- report median, minimum, maximum, median absolute deviation, and coefficient
  of variation;
- reject timing sets with coefficient of variation above 3% and rerun after
  identifying interference;
- record environment, revisions, CUDA/cuDNN, device clocks/power mode, precision,
  attention backend, K, physical B, family, sequence length, and exact logical
  work.

Run the serial baseline and candidate in alternating order, not all baselines
followed by all candidates.

### B. Correctness gates

For K in `{1,2,4}` and both Normal and RB:

1. **Input identity:** sampled indices, family order, physical row counts,
   objective mass, targets, reference values, and initial lane RNG states must
   be bit-identical to the serial replay.
2. **Lane isolation:** changing only parameters or inputs in lane `j` must
   leave every output, loss, gradient, moment, and parameter transition in all
   `k != j` lanes bit-identical.
3. **FP32 diagnostic parity:** forward fields, scalar losses, dense gradients,
   parameters, and moments must satisfy `rtol=1e-5, atol=1e-7` after one step.
4. **BF16 one-step parity:** every output/loss must satisfy
   `rtol=5e-3, atol=5e-4`; each lane's dense gradient must have relative L2
   error at most `5e-4` and cosine similarity at least `0.999999`; maximum
   parameter discrepancy after the update must be at most `1e-6` for the
   selected learning rate `5e-5`.
5. **Complete replay:** after 116 steps, sampled rows and RNG states remain
   exact, all 116 gradients are finite and applied, no lane version diverges,
   loss-component sequences remain within the BF16 one-step tolerance, and
   final parameter relative L2 error is at most `2e-3` per lane.
6. **Dense-zero semantics:** a deliberately unused parameter receives an
   explicit zero gradient and the same Adam/AdamW transition as the serial
   dense oracle.
7. **Failure atomicity:** inject a failure before backward, after backward, and
   before optimizer publication in each lane. Parameters, buffers, moments,
   versions, scaler, RNG, and run state must restore bit-identically.
8. **Checkpoint identity:** save/resume after a batched step and prove the next
   sampled rows, losses, gradients, and state transitions match an uninterrupted
   batched run under the same tolerances.

Do not relax these thresholds after observing a candidate. If a threshold is
shown to be physically inappropriate, document the specific operation and an
independent numerical error bound before changing it.

### C. Dispatch and kernel gates

For real preflop and postflop batches, retain evidence of:

- cuDNN SM90 Flash forward and backward kernel names;
- no math-SDPA or functorch batching-fallback warning;
- actual Q/K/V shapes including the folded `K*B` batch where applicable;
- GEMM/BMM operation and kernel shapes for replica-specific projections;
- no per-lane Python model call inside a purportedly batched cohort.

Use a bounded three-step Nsight Systems capture with identical NVTX ranges for
serial and candidate. Report:

- learner-range wall time;
- unioned GPU-kernel-active time and active fraction;
- kernel launches per replica optimizer step and per 1,000 presentations;
- count and total duration of launch gaps over 100 microseconds and 1 ms;
- grouped device time for GEMM/BMM, Flash forward, Flash backward, LayerNorm,
  elementwise, concat/copy, reductions, and optimizer kernels;
- H2D and D2H bytes, which must not increase except for explicitly reported
  scalar telemetry.

The earlier 34.1% kernel-active figure is a profiler-perturbed reference, not a
timing baseline. Under the identical capture method, the candidate should reach
at least 55% active fraction at its selected K and reduce launches per replica
optimizer step by at least 30%. Failure of those diagnostic targets does not
override measured throughput, but it requires explaining where the gain came
from or why it was absent.

### D. Throughput metrics and acceptance

Report all of the following; do not use `nvidia-smi` utilization as a speed
claim:

$$
\text{presentations/s} = \frac{\sum_k\text{physical rows}_k}{t_{\mathrm{learner}}},
$$

$$
\text{replica optimizer steps/s} = \frac{K S}{t_{\mathrm{learner}}},
$$

$$
\text{model tokens/s} =
\frac{\sum_{\text{calls}} B_{\text{call}}N_{\text{call}}}
{t_{\mathrm{forward+backward}}}.
$$

Also report median cohort latency, per-replica latency, whole-iteration
player-nodes/s, rollout seconds, learner seconds, and reference-cache build/hit
seconds. Aggregate learner speedup is candidate presentations/s divided by the
matched serial-K presentations/s. It must not be computed from K=1 with a
different total batch, family mixture, root count, or optimizer-step count.

Retain the initial prototype only if:

- K=1 regresses by no more than 3%;
- K=2 improves aggregate learner throughput by at least 25%;
- K=4 improves aggregate learner throughput by at least 75%;
- both Normal and RB pass the one-step correctness gates;
- Flash forward and backward remain selected.

Retain the production scheduler only if the best memory-admitted K improves
aggregate learner throughput by at least 50% in both complete 116-step Normal
and RB replays and improves or preserves full-iteration nodes/s. Report
mode-specific optima if their memory/target behavior selects different K or B.

### E. Memory sweep

For each candidate record synchronized values at cohort entry, after forward,
after backward, after optimizer, and after release:

- `memory_allocated` and `memory_reserved`;
- peak allocated and peak reserved;
- inactive non-releasable/fragmentation bytes where available;
- exact parameter, gradient, optimizer, rollback, reference-model, reference-
  cache, gathered-input, target, activation, and temporary-workspace bytes;
- end-of-step and end-of-iteration allocated bytes.

Run two complementary matrices on the H100:

1. **Per-replica B scaling:** K in `{1,2,4,8,16,32}` with B in
   `{32,64,84,128,256}` when admitted; additionally K=64 with B in
   `{32,64,84,128}`.
2. **Fixed aggregate rows:** choose `K*B` in `{256,512,1024,2048,4096}` and
   vary K to distinguish more replicas from more total model work.

Begin with a 512-root replay so root storage does not conceal learner scaling.
Then validate the selected K/B in a complete iteration with roots per replica
and total roots both stated explicitly. Never extrapolate a K=1 4,096-root
full-session peak as though it were learner activation memory.

Fit measured peak allocation to:

$$
M(K,B)=M_0+K M_{\mathrm{resident}}+KB M_{\mathrm{row}}.
$$

Report fitted coefficients and residuals for Normal and RB. The estimate is
accepted for admission only if maximum prediction error on held-out K/B points
is at most 15%; otherwise use the measured upper envelope. With the configured
10 GiB reserve, require peak allocated memory below the resolved usable-memory
limit, at least 2 GiB additional measured headroom inside that limit, no OOM,
and no monotonic end-of-step memory growth over 116 steps.

## Experiment order and stop conditions

1. K=2 single-family forward/loss plus outer backward, Normal.
2. Prove Flash forward/backward and one-step parity.
3. Repeat for postflop RB.
4. Compare vmap-generated projection kernels with explicit BMM on the same
   tensors; retain only the faster correct path.
5. Implement step-major K=2 and K=4 scheduling.
6. Batch optimizer checks/transitions.
7. Sweep K/B through K=32, then K=64 where admitted.
8. Optimize retained-state/cache lifetime if it, rather than active learner
   tensors, prevents the next K.
9. Only after the batched trace is stable, evaluate concat/copy fusion and
   postflop compaction as separate commits.

Stop and revert a candidate when it loses Flash backward, crosses a correctness
gate, introduces cross-lane state/RNG dependence, exceeds memory admission, or
fails the stated throughput threshold. A failed vmap prototype does not reject
explicit BMM, and a failed explicit BMM prototype does not reject CUDA-stream
lane concurrency as a separate baseline.

## Audit and rollback requirements

Use one commit per numbered implementation item. For every candidate, append:

- commit and parent SHA;
- exact commands/configs;
- environment and profiler versions;
- correctness results;
- timing and memory tables;
- retained/rejected decision and reason;
- paths and SHA-256 hashes for raw JSON, Nsight, and test artifacts;
- the exact `git revert <sha>` command.

Rejected implementations must be reverted from production, with their patches
and evidence retained in the handoff. Do not combine scientific/model changes
with execution optimizations in one commit.
