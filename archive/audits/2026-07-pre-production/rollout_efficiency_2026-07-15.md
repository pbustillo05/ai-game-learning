# Rollout and training efficiency audit — 2026-07-15

## Scope

This audit follows the current working tree through:

- normal and sequential Rao--Blackwell rollout;
- fixed duplicate-preserving branched rollout;
- poker public-state/reach/model preparation;
- preflop/postflop policy routing and physical batch planning;
- completion and reach reconstruction;
- chunked dataset construction and PPO epochs;
- optimizer/reference-model execution; and
- analytical and measured CUDA memory gates.

It focuses on a single RTX 2080 Ti and the six-player 10bb configurations. It
does not propose new estimator mathematics and does not change the sibling
evaluator.

## Executive verdict

The concern is valid.

The current implementation does perform neural forward passes on the GPU and
batches the player decisions that are ready in the **current frontier wave**.
It does **not** maintain persistent preflop and postflop queues, does not wait
across waves to fill a queue, and does not use a forced-drain scheduler. The
word “queue” in existing tests/documentation referred to routing a supplied
batch, not a long-lived rollout work queue.

This is a major performance-architecture gap, not a known estimator-correctness
bug. A measured 128-trajectory rollout made 142 model calls. Pooling the same
realized decision rows into hardware-bounded preflop and street/live-player
postflop buckets required 110 calls, a 22.5% lower bound on call reduction.
An online dependency-aware queue cannot necessarily achieve the full bound, but
the result proves that wave-local scheduling creates avoidable partial batches.

There is a second, independent issue: sequential memory preflight scales several
transient tensors by `n_trajectories * node_budget`, although a sequential
frontier contains at most one row per trajectory. With 1,000 trajectories and a
128-node per-path guard, the current planners report about 9.59 GB for poker
rollout and 31.98 GB for generic structural training. An 11GB 2080 Ti rejects
that configuration even though the corresponding large reach/policy arenas are
not allocated by the sequential hot path. This prevents the large resident root
batch that would make persistent queues most effective.

The current streamed 1,024-trajectory preset avoids the rejection by keeping
only 128 trajectories resident. It rolls, targets, and optimizes each chunk
before starting the next; it does not collect all `N` decision nodes and then
run global epochs.

## Requested design versus current code

| Requirement | Current state | Assessment |
|---|---|---|
| GPU policy/value forwards | Model, inputs, outputs, reaches, and targets stay on CUDA | Implemented |
| Batched rollout inference | All player rows ready in one frontier wave enter one logical call | Implemented, wave-local |
| Persistent preflop queue per GPU | No queue object or pending-work lifecycle exists | Missing |
| Persistent postflop queue per GPU | No persistent queue; one call is partitioned by live count and then street | Missing |
| Fill while trajectories advance, drain at target size | Every wave is drained immediately, including small tails | Missing |
| Drain preflop first, then focus on postflop | Preflop plans run first only within a single call; later preflop work can appear after postflop calls | Partial ordering only |
| Branching compatible with queues | One breadth frontier batches unique parents; no cross-depth ready queues | Partial |
| Use VRAM while leaving learner headroom | Fixed chunk size plus conservative preflight; no autotuning from measured peak | Safe, substantially underfilled |

## Exact current scheduling

### Sequential normal and RB

For `G` resident trajectories, the core holds one current state/reach/profile row
per live trajectory. Every loop iteration advances every active row by exactly
one public transition:

```text
active = current non-leaf rows
players, chances = partition(active)

if players:
    output = poker.policy_forward(all current player rows)
    event = sample one action per row
    next_reach = update acting-player likelihood

if chances:
    event = sample public cards
    next_reach = mask incompatible hands

successor = step every active row
current = successor
```

Trajectories that finish disappear from later waves. A faster trajectory cannot
run multiple decisions while a slower trajectory is still in the same wave;
all model work returned by `policy_forward` is completed before successors are
available.

### Poker routing inside one wave

`plan_policy_batches` sees only the `players` tensor passed above. It computes a
token estimate from street, live-player count, and history length. It gives
preflop family id 0 and postflop family id equal to the live-player count, sorts
each family by token length, and greedily packs under the hardware profile's
token and token-pair caps.

`_policy_forward_direct` then:

- runs one preflop call for all preflop positions in that physical plan; and
- loops over street 1/2/3 and live counts 2..P, making one postflop call for each
  nonempty combination.

The planner can put turn and river rows of the same live count in one plan, but
the direct executor splits them again because their compact combo widths differ.
The implementation therefore behaves like preflop plus `(street, live_count)`
postflop buckets, but those buckets exist only for the duration of one frontier
call.

### Branched rollout

Branched RB owns one sampled tree per rank. Each wave randomized-rounds the
offspring count of every active node. The model is evaluated once per unique
player parent; output is gathered for the children, which have independent
posterior profiles and sampled events.

This avoids redundant model calls for siblings of one parent. It does not fix
narrow frontiers: the root forward is always one row, beta near one can keep
later waves narrow, and ready decisions at different depths are never pooled.

## Physical batch geometry on a 2080 Ti

The local profile sets:

```text
inference: max_tokens=16,384, max_token_pairs=134,217,728
training:  max_tokens= 8,192, max_token_pairs= 67,108,864
```

For the 10bb table, `n_ticks=20` and there are 22 policy events. Sequence length
is approximately:

```text
preflop = 6*169 + 73 scalar fields + state/anchors + history
        ~= 1,095 + history

postflop = live_players * {1176,1128,1081}
           + public/history/anchor fields
```

The resulting maximum physical batch is approximately:

| Model family | Inference | Training |
|---|---:|---:|
| preflop | 14 | 7 |
| postflop, 2 live | 6 | 3 |
| postflop, 3 live | 4 | 2 |
| postflop, 4 live | 3 | 1 |
| postflop, 5 live | 2 | 1 |
| postflop, 6 live | 2 | 1 |

History length can reduce a boundary by one. These small postflop batches are
not automatically evidence of low GPU use: each row contains thousands of
tokens and full attention/incidence compute. The efficiency problem is the
additional calls below these already small safe caps.

An unsharded “postflop queue” is therefore not the best implementation. Padding
a two-live flop row to a six-live sequence wastes most of its attention work.
The logical scheduler should expose two model families to the caller, but the
postflop family should be internally bucketed by at least live-player count and,
with the current compact implementation, street.

## User-scale minibatch arithmetic

Let `N` be retained decision nodes, `B=128`, and `E` be exact PPO epochs. A
single global dataset would use `E * ceil(N/B)` optimizer steps while processing
exactly `E*N` rows (the last logical batch is smaller than 128):

If the stated 5–20 nodes per trajectory counts chance states, roots, and leaves,
then learner `N` is smaller: only player-decision nodes become minibatch rows.
The rollout state/tree capacity must still cover every public node.

| `N` decisions | 1 epoch steps | 2 epoch steps | nominal `K*B/N` |
|---:|---:|---:|---:|
| 5,000 | 40 | 80 | 1.024 / 2.048 |
| 12,500 | 98 | 196 | 1.004 / 2.007 |
| 20,000 | 157 | 314 | 1.005 / 2.010 |

This is a sensible scale. `ppo_epochs` is preferable to choosing
`gd_steps_per_iter` from an expected `N`: the epoch path gives exact
without-replacement coverage, while the legacy path samples with replacement
and can duplicate/omit rows.

### What chunking changes

For chunk decision counts `D_i`, current code performs:

```text
steps = E * sum_i ceil(D_i / B)
optimized rows = E * sum_i D_i
```

It does not perform `E * ceil(sum_i D_i / B)`. Every chunk has its own partial
tail and optimizer steps occur before later chunks are rolled out. At `B=128`
the tail overhead is usually modest; at larger `B` it is visible.

The current 1,024-trajectory normal run realized 9,113 decisions in eight
resident-128 chunks. With `B=512` and one epoch it made 24 optimizer steps;
one global dataset would have made 18. Both process each row once, but the
chunked version has six extra optimizer/gradient-clip tails and cannot shuffle
rows across chunks.

## Measured evidence

### Queue-fragmentation probe

The probe used:

- idle NVIDIA GeForce RTX 2080 Ti;
- PyTorch 2.12.1+cu130;
- `local_3x2080ti` hardware limits;
- the d=64 six-player 10bb normal model;
- 128 resident trajectories, seed 1128; and
- rollout only, after evaluator extension warm-up.

It realized 1,217 public nodes and 1,064 player decisions: 955 preflop and 109
postflop. Peak PyTorch memory was 329,706,496 allocated bytes and 585,105,408
reserved bytes. Rollout took 7.78 seconds; that time includes state transitions,
profile/event sampling, range preparation, evaluator fold summaries, and model
work, so it is not a pure model benchmark.

Actual model-call fill was:

| Family | Rows | Calls | Mean rows/call | Safe max |
|---|---:|---:|---:|---:|
| preflop | 955 | 78 | 12.24 | 14 |
| postflop, 2 live | 6 | 6 | 1.00 | 6 |
| postflop, 3 live | 24 | 16 | 1.50 | 4 |
| postflop, 4 live | 58 | 29 | 2.00 | 3 |
| postflop, 5 live | 21 | 13 | 1.62 | 2 |

Repacking the same realized rows globally by preflop and `(street, live_count)`
under the identical token limits took 110 calls instead of 142. This 22.5%
reduction is an optimistic lower bound because an online scheduler cannot see
future dependent decisions. Persistent queues can approach it by continuing to
advance other ready work before forcing a partial drain.

The result is one topology/policy sample, not a confidence interval. It is
sufficient to demonstrate fragmentation, not to predict an exact speedup.

### Existing realistic streamed training profile

The first profiled iteration in
`runs/nlhe6_10bb_normal_streamed_4h30/metrics.jsonl` used 1,024 logical
trajectories, 128 resident, d=64, `B=512`, one epoch, and reference KL. It
reported:

| Phase | Seconds | Share of profiled phase total |
|---|---:|---:|
| snapshot | 0.014 | <0.1% |
| rollout | 35.705 | 17.8% |
| normal target/backward | 1.953 | 1.0% |
| optimization | 163.081 | 81.2% |
| total | 200.754 | 100% |

It realized 10,467 nodes and 9,113 decisions, made 24 optimizer steps, and
reserved 1,145,044,992 bytes at peak against a 9,199,091,712-byte safe limit.
The analytical structural estimate was 4,117,757,144 bytes. VRAM was therefore
not close to the safe ceiling.

The optimizer phase dominates this configuration because every logical
minibatch is split into the small training batches above, each current-model
batch needs backward, and `c_kl > 0` adds a reference-model forward. Improving
rollout queues remains valuable, but it will not by itself fix the largest
measured phase.

`policy_physical_batches` was reported as exactly 4096 because the backend stores
these records in a `deque(maxlen=4096)`. It is a truncated lower bound for long
iterations, not the real batch count. This observability bug should be fixed
before comparing scheduler implementations.

## Memory allocation and movement audit

### What remains on CUDA

The following stay on one GPU throughout the hot phases:

- tensor public state store and state handles;
- tree topology and path weights;
- dense current reaches and cached old outputs;
- sampled events/profiles and evaluator inputs/outputs;
- completion learning tensors and posterior targets;
- decision reaches and legal-event rows;
- current/reference model execution and optimizer state.

There is no intentional bulk CPU replay buffer or per-minibatch PCIe transfer.
DDP all-reduces gradients. Checkpointing/metrics copy data to host outside the
inner model loop.

### High-volume device allocations/copies

Rollout performs:

- state/reach gathers and successor scatter each wave;
- a dense `exp(log_reach - gauge)` conversion for every policy call;
- preflop combo-to-class scatter-add and class-to-combo output lift;
- postflop compact gather, attention/incidence work, and 1326-row scatter;
- output scatter back to frontier order; and
- list accumulation followed by concatenation of realized old outputs.

Target construction with `reach_storage: recompute` rebuilds reaches, then poker
retains dense `[decision,6,1326]` fp32 rows during optimization. This costs
31,824 bytes per decision before targets/cache metadata. Measured retained bytes
were 31,860 per normal decision and about 58,389 per sequential-RB decision in
the current smokes.

### Host synchronization inside the device path

The code is device-resident but not launch-asynchronous. Examples include:

- `plan_policy_batches`: CUDA `torch.unique(...).tolist()`, row/token
  `.tolist()`, and maximum `.item()` on every logical call;
- `_packed_history`: `lengths.max().item()` per direct model call;
- `_partition_summary`: cache-miss boolean and cache keys copied to CPU;
- public transitions: tensor-to-`bool` branches for folds/showdowns;
- core rollout/branching: tensor-to-`bool` termination, beta, and overflow
  decisions; and
- `ReachCache.rows_for`: `requested.tolist()` plus recursive device scalar
  reads.

Existing `test_aud26_no_host_sync.py` validates the alias table, compact map, and
attention primitive only. It does not exercise this end-to-end scheduler. The
documentation should not claim a host-sync-free rollout.

## Why the 1,000-trajectory plan is rejected today

For sequential recompute mode, actual simultaneous frontier reach is bounded by
`G * P * H`, because each trajectory has one current state. The planners instead
use capacity:

```text
capacity = G * node_budget
dense_reach_bound = capacity * P * H * 4
```

At `G=1000`, `node_budget=128`, `P=6`, and `H=1326`:

- the poker-specific estimate is 9,592,840,004 bytes;
- the generic rollout-phase estimate is 31,976,287,576 bytes; and
- the generic optimization estimate is 19,738,905,960 bytes.

The largest generic terms are four capacity-sized transient reaches
(16,293,888,000 bytes) and a capacity/event/type policy bound
(15,614,976,000 bytes). The sequential sampler does not allocate these dense
capacity-sized objects. Its frontier is at most 1,000 rows and its old-output
cache grows with realized decisions, not every possible event at all 128,000
capacity rows.

The topology and poker state store really are preallocated at 128,000 rows, but
their corresponding estimates are only about 8.2 MB and 53.4 MB. They are not
the reason 1,000 roots are rejected.

For the stated 5–20 decision scale, dense retained reach would be roughly
159–636 MB for 5,000–20,000 decisions. Old sequential rollout log-probability
plus value rows would add roughly 53–212 MB before normal compaction. These are
large but compatible with an 11GB card after accounting for model workspace,
targets, optimizer, evaluator, and a measured safety margin. This is a planning
hypothesis, not approval to bypass preflight: the planner must be corrected and
the sizes staged.

## Bottleneck ranking

1. **Training physical batching and reference forward.** In the realistic
   profile, optimization is 81% of iteration time. Training caps are 1–7 rows
   per physical poker call, and reference KL adds another forward.
2. **Missing persistent rollout queues.** Wave-local partial calls were 22.5%
   above the all-rows lower bound in the measured rollout.
3. **Over-conservative sequential memory preflight.** It blocks the large
   resident trajectory pool that would improve queue fill and use available
   VRAM.
4. **Host synchronization in scheduling/history/cache logic.** Multiple forced
   device-to-host decisions occur per wave/model call.
5. **Python recursive reach replay.** `recompute` saves retained rollout memory
   but reconstructs rows with Python/device scalar traffic and a dense memo per
   request.
6. **Chunk-local optimization.** It adds logical minibatch tails, prevents a
   global shuffle, and starts updating before later chunks are generated.
7. **Incomplete performance counters.** The fixed-length batch-record deque
   hides exact physical call counts and has no queue-fill/forced-drain metrics.

## Recommended design

### 1. Make sequential memory planning mode-aware

Do this before raising trajectory counts.

For sequential modes, independently budget:

- preallocated state/topology capacity;
- current and next reach frontiers using `resident_trajectories`, not node
  capacity;
- old-output cache using a configurable/observed decision bound;
- retained dense decision reach using expected/high-quantile decisions;
- normal or RB target bytes per realized decision;
- completion batches;
- current/snapshot/reference models and optimizer;
- maximum physical inference/training workspace from the hardware packer; and
- allocator/topology safety margin.

Keep the per-trajectory 128-step correctness guard, but do not multiply every
transient by that guard. A global realized-node/decision arena with a per-path
overflow check would make the distinction explicit.

Then stage paired 128, 256, 512, and 1,000-root probes on an idle GPU, using the
same frozen checkpoint and counter-based/per-trajectory seeds. Do not infer
throughput from different random topologies.

### 2. Introduce persistent ready queues

A work item needs at least state handle, reach row, trajectory/tree identity,
private profile or child sampling context, parent/topology metadata, and a stable
RNG identity.

Use:

- one preflop ready queue; and
- postflop queues bucketed by street/live-player count (or a measured padding-
  compatible equivalent).

For the requested preflop-first behavior:

1. process chance rows immediately when no model inference is needed;
2. enqueue every newly ready player row;
3. drain full preflop batches first;
4. allow postflop work to accumulate while any independent preflop work can
   still make progress;
5. after the preflop phase is exhausted, drain postflop buckets by highest
   hardware fill/cost; and
6. force-drain the best partial bucket when no other operation can produce more
   ready rows, so the scheduler cannot deadlock.

An asynchronous schedule changes generator-consumption order. Assign RNG
counters/seeds to trajectories/nodes/abstract children rather than relying on
one global draw stream if reproducibility must not depend on queue drain order.

The current CSR builder assumes parent-major wave appends. Sequential modes can
use a flat trajectory layout. Branched mode will need either parent-order drains
or an edge arena that sorts/builds CSR after sampling.

### 3. Decouple logical dataset size from resident execution

The preferred 1,000-trajectory design is one resident rollout set if corrected
planning and measured peaks allow it. Then construct `N` targets and run global
`B=128`, one- or two-epoch permutations.

If all rows cannot remain resident, choose explicitly between:

- chunked rollout plus a GPU-resident compact dataset followed by global
  epochs;
- measured pinned-host spill with asynchronous transfers; or
- current chunk-local optimization.

Do not call the third option a global epoch. Host spill should be adopted only
if transfer overlap beats the additional VRAM pressure in a real profile.

### 4. Remove host decisions from the steady-state scheduler

Precompute or maintain device metadata for street/live/history token length.
Use tensor histograms/prefix sums to build ready buckets. Replace the Python
partition-summary key dictionary with a device-indexed immutable cache. Make
history padding use a queue/bucket width already known to the scheduler.

Some termination synchronization at a forced drain boundary is acceptable; a
device-to-host branch for every family and call is not.

### 5. Vectorize reach replay or retain checkpoints

For sequential trajectories, store periodic reach checkpoints or use a batched
parent-prefix replay. Avoid Python recursion, per-node scalar reads, and a fresh
whole-node memo for each completion/compaction request. Compare replay time and
bytes against simply storing decision reaches during rollout.

### 6. Add exact performance telemetry

Replace the 4096-record deque as the source of aggregate metrics. Record
unbounded counters/histograms for:

- ready rows and actual model calls by phase/street/live count;
- safe capacity and fill ratio per call;
- forced partial drains and queue wait/age;
- padded tokens and pairs, sum as well as maximum;
- rollout versus optimization current/reference calls;
- scheduler host-sync count/time;
- model CUDA-event time separate from public navigation/evaluator time; and
- actual allocated/reserved peaks by phase.

## Acceptance gates for a queue implementation

Correctness gates:

- identical legal event/reach update semantics;
- one model evaluation per unique branched parent;
- duplicate abstract children and independent child RNG identities preserved;
- exact old-policy/optimization input identity gate passes;
- normal/RB terminal and posterior laws match small exact games; and
- loud per-path/tree capacity failure remains.

Performance gates on the same GPU, checkpoint, seeds, and realized work:

- fewer actual model calls and forced partial drains;
- higher mean fill within every hardware family;
- no increase in padded token-pair work that offsets call reduction;
- lower rollout wall time and no regression in full iteration time;
- measured peak plus 25% margin below the configured safe limit; and
- GPU clean before and after every probe.

## Validation performed for this audit

- Default suite: 214 passed, 10 CUDA-only tests skipped; the sole sandbox Gloo
  rendezvous failure was rerun outside the network sandbox and passed.
- CUDA-visible full suite on GPU 2: 222 passed and one multi-device test skipped.
  Two strict-resume cases failed only at the final exact metrics comparison:
  their checkpoint model tensors were bitwise equal, but the test also compares
  nondeterministic CUDA phase seconds/throughput from two separately profiled
  runs. All CUDA-gated rollout, residency, compact-map, and attention assertions
  passed. The strict-resume test needs to exclude observational timing/memory
  fields when CUDA profiling is active.
- GPU queue probes ran only on idle GPU 2; GPUs 0 and 1 had existing workloads.
- GPU 2 returned to its 19 MiB display/context baseline with 0% utilization
  after the probes.

The audit did not modify rollout/training code. Its recommendations require a
separate implementation and same-workload benchmark before any production
preset is enlarged.
