# Unified dense queued training engine

Status: ambitious throughput proposal; not an implementation contract.

Date: 2026-07-23.

## Executive proposal

Replace the current rollout execution path in all three production modes with
one fixed-address, device-resident queued engine built around dense
per-family `FamilyWork` slabs.

The central physical rule is:

> Put a player occurrence into its final physical execution order once, when
> the occurrence is created, and make every later hot consumer read a direct
> contiguous FIFO prefix. Queue identity is semantic; poker street and padding
> width never create another queue.

For physical family $m$, allocate a structure of arrays with leading capacity
$N_m^{\rm cap}$:

```text
FamilyWork[m]
  hot mutable columns
    state fields              [N_m_cap, ...]
    public history            [N_m_cap, T]
    active range weights      [N_m_cap, P, H]
    logical node/root id       [N_m_cap]
    path key / ordinal         [N_m_cap]
    branching path factors    [N_m_cap, ...]  # branched only

  retained decision columns
    forward-ready input        [N_m_cap, I_m]
    owner and mode metadata    [N_m_cap, ...]
    old model outputs          [N_m_cap, Y_m, ...]

  control
    head
    reserved_tail
    ready_tail
    oldest logical time
```

All tensors are batch-major: `[N_m_cap, ...]`. A dequeue is a view:

```python
rows = slice(head_m, head_m + batch_rows)
state = family.state[rows]
reach = family.range_weights[rows]
model_input = family.input[rows]
```

There is no occurrence-handle queue, dequeue staging tensor, hot page lookup,
active-reach handle, reach gather, or status gather/scatter on the ordinary
player path.

When a parent creates children, the child-materialization path:

1. reserves logical topology and child-indexed edge addresses before any draws;
2. samples the private profile and public event with logical RNG keys;
3. performs the state transition and reach update;
4. classifies the resulting child;
5. appends a player child directly to its destination `FamilyWork` tail;
6. appends a stopped child directly to final leaf storage;
7. sends a chance child into a bounded device chance-closure workspace.

The ultimate poker implementation fuses steps 2--7. The rejection sampler,
event draw, transition, reach update, classification, stable destination
offset calculation, and writes into family/leaf/chance destinations should be
one CUDA pipeline with no host synchronization in the middle. Whether the
fastest realization is one cooperative persistent kernel or a short captured
kernel graph is a benchmark question; the storage and ownership contract
should support both.

The rewrite deliberately spends memory to remove routing, allocator, and
kernel-launch work. For heads-up full-deck poker,

$$
25{,}000\cdot 2\cdot 1326\cdot4
=265{,}200{,}000\ {\rm bytes}
\approx252.9\ {\rm MiB}
$$

for the complete active-reach capacity. Even 50,000 rows cost only about
505.8 MiB. This is small compared with the measured rollout/model workspace
on the target 2080 Ti.

The intended outcome is not merely a faster reach owner. It is removal of the
hot occurrence, queue, active-reach, controller-routing, and mutable
state/history indirection that makes later branched packs more expensive than
earlier packs.

## Throughput rationale

The optimization unit is the complete dequeue-to-enqueue path. Direct family
storage removes hot state, history, reach, occurrence, and controller
indirection together. Fused materialization removes transient profile and
routing traffic. Device-resident chance closure removes host launch and
synchronization boundaries. The layered target engine separately makes target
cost proportional to realized numerical work rather than CPU scheduling over
nodes and actions.

This coordinated layout matters because every hot consumer receives direct
contiguous storage, and every retained field is written in the physical order
required by its next consumer.

## Authority and required contract reconciliation

This proposal preserves the following mathematical and rollout contracts:

- exactly one distinguished rooted tree per GPU in branched mode;
- ordinary root branching and protected-cohort refill semantics;
- nominal and executed child counts;
- one distinct logical child per sampled outgoing occurrence, including
  duplicate public events;
- child and child-indexed edge addresses reserved before their random draws;
- counter-based logical RNG addresses independent of queue order and physical
  packing;
- stable insertion-order FIFO prefixes;
- the unchanged player/chance event laws and reach update;
- no dropped, truncated, replayed, or silently rerolled sequential trajectory;
- whole-attempt-only branched memory recovery;
- exact retained inputs, topology, leaves, targets, and learner law.

The principal intentional conflict is with the current physical contract in
`docs/theory/pseudocode/storage_lifecycle.tex`.

That document requires active-reach capacity to follow live-frontier high
water and explicitly rejects a capacity proportional to total generated work.
This proposal instead preallocates a configured maximum number of player work
rows per family and normally advances a monotone tail without recycling the
processed prefix during the attempt.

Processed rows are logically dead immediately after their children and
retained outputs commit. Their stale bytes are not a reach memo and may never
be read by targets or the learner. Nevertheless, the physical slab capacity
is intentionally proportional to configured queued-work capacity rather than
observed live frontier.

Production acceptance therefore requires an explicit decision to amend that
physical capacity/lifetime rule. Nothing under `docs/theory/` should be
modified without the required approval. Until then, this document is an
experimental proposal, not authority.

The rewrite also supersedes the pseudocode's current occurrence-arena,
active-reach-owner, and queue realization. Those are implementation-contract
changes even where the mathematical semantics above are preserved, so the
authoritative pseudocode must be revised alongside an accepted implementation,
not left describing retired owners.

The proposal forbids street-specific poker partitions and queues. Poker
continues to expose only `pre` and `post_A` semantic families. Token-width
reduction is performed by minimally padding stable FIFO prefixes, as described
below.

## Design goals

The replacement should satisfy all of these goals together:

1. Every ordinary player dequeue reads direct contiguous views.
2. The cost of a work pack is independent of accumulated tree size.
3. No production hot access loops over historical pages or chunks.
4. No player occurrence carries an indirect active-reach handle.
5. No player queue copies handles into dequeue staging.
6. Child routing performs one stable destination calculation per child.
7. Chance closure performs no Python loop per chance generation.
8. Rejection sampling does not allocate a returned profile tensor merely for
   the next kernel to consume it.
9. Family appends can overlap another ready family's model forward.
10. Fixed addresses permit CUDA graph capture and reusable workspaces.
11. Sequential modes use the same engine without paying branched-only work.
12. Rollout slabs are released before targets/optimization so their deliberate
    overcapacity does not reduce learner width.
13. The surviving production implementation has fewer owners and APIs than
    the current implementation.
14. Queue count depends only on declared semantic partitions, never on street
    or a physical padding category.
15. Branched target work is stratified into device-executable dependency
    layers, with no CPU loop over nodes or actions.

## Non-goals

This rewrite does not:

- change the branching intensity law;
- share one sampled private profile among branched siblings;
- merge duplicate public-event children;
- change completion sampling;
- reorder retained target rows by a private quantity;
- introduce CPU rollout state;
- introduce a CPU or automatic device fallback;
- change the PPO/RB losses;
- add a parallel legacy production path;
- make the learner consume mutable rollout slabs;
- split semantic queues by street, board width, token width, or padding bucket;
- allocate any pair-indexed branched field as
  `[decision, max_children, ...]`.

## Physical family definition

Core continues to schedule opaque public partitions supplied by the game.
For poker:

$$
\mathcal M
=\{\mathsf{pre}\}\sqcup\{\mathsf{post}_A:2\le A\le P\}.
$$

Flop, turn, and river with the same live-player count remain in one
`post_A` FIFO. This is important for semantic and implementation modularity:
core never parses poker street.

This is a permanent physical invariant, not merely a first implementation.
The semantic layout has $P$ families: one `pre` family and $P-1$ postflop
families. Crossing it with three postflop streets would produce $1+3(P-1)$
queues: 16 at $P=6$ and 25 at $P=9$. That multiplies small tails, partial
dequeues, capacity estimates, scheduler choices, and CUDA graph variants to
save padding in only a fraction of rollout inference. It becomes less
attractive as player count grows and as the rollout becomes less massive.

The physical planner may inspect the stable head prefix through the
game-owned byte/work oracle. It may select a stable maximum-feasible prefix or
stable consecutive subspans, but it may not reorder or skip an older row.
Padding optimization changes pack boundaries and realized shapes, never queue
identity or queue allocation.

## FamilyWork layout

`FamilyWork` is a schema-generated structure of arrays. It is not one
array-of-structs allocation. Each hot column has a direct batch-major leading
axis and alignment chosen for its kernels.

### Common hot fields

Candidate common fields are:

| Field | Shape | Purpose |
|---|---:|---|
| `node_id` | `[N_m] int32` | stable topology identity; implicit/optional in sequential mode |
| `root_id` | `[N_m] int32` | sequential trajectory identity |
| `path_key` | `[N_m] int64` | logical RNG address component |
| public state columns | `[N_m,...]` | legality, transition, routing |
| dense history | `[N_m,T,...]` | transition and forward-ready public context |
| `range_weight` | `[N_m,P,H] fp32` | active exact reach in sampler-ready linear form |
| `decision_row` | implicit or `[N_m] int32` | stable retained-row address |
| enqueue time | `[N_m]` or run metadata | age scheduling |

The preferred poker implementation stores public state inline in the family
slab instead of storing a handle into another mutable state owner. State is
small beside exact reach. Dense history is also preferred when its exact
schema fits the admitted geometry; shared-history indirection should survive
only if a measured deep/multiplayer case demonstrates that dense copying
reduces end-to-end throughput or violates the hard ceiling.

### Branched-only hot fields

Candidate branched fields are:

| Field | Shape | Purpose |
|---|---:|---|
| `log_weight` or path-mass factor | `[N_m]` | target/controller path mass |
| `log_player_sum` | `[N_m] fp32` | dynamic KKT input |
| `log_ancestor_product` | `[N_m] fp32` | dynamic KKT input |
| queue-cost forecast | `[N_m] fp32` | optional scientific safeguard |
| depth/player clocks | `[N_m,...] int32` | direct child metadata construction |
| path weight | `[N_m] fp32/fp64` | direct topology write |

Fields required only after rollout should be written directly to cold retained
tables and omitted from the hot slab. The final schema follows consumers, not
the current classes.

### Retained decision fields

Every player occurrence is a retained decision in the ordinary training
rollouts. Align its family slot with its stable decision row whenever possible:

```text
family work row q
    ↔ decision/input row q
```

At player-node creation, write the immutable forward-ready natural model
input once. On dequeue, the model reads that direct family prefix. After the
forward, write old outputs to the aligned retained decision row.

Mode-specific data indexed by a sampled pair $(\sigma,a)$ is different: write
it at the row of the realized child node $\sigma a$. A child row may carry a
partition-local reference for variable-width payloads, but the reference is
one-to-one with that child. Do not store pair data in a dense parent row with a
maximum-child dimension. This remains safe when the KKT controller realizes a
large and unpredictable maximum degree.

This permits the model input itself to be direct and contiguous without
reconstructing it from mutable state during every learner presentation.

Retained inputs and old outputs have a longer lifetime than mutable state and
reach. They may occupy separate column groups under one family cursor. At seal,
copy only realized prefixes into exact immutable partition blocks, then
release all overprovisioned family capacity.

## Linear active reach instead of active log reach

The sibling evaluator accepts finite nonnegative `range_weights[P,H]` and is
invariant to positive per-seat scaling. The current adapter converts active
log reach to linear weights before rejection sampling.

The proposed hot representation stores normalized or safely scaled linear
`range_weight` directly:

```text
active rollout owner:      linear range_weight[P,H]
retained model input:      model-ready natural range representation
retained RB leaf:          completion-ready range_weight[P,H]
```

For a player event, the fused transition multiplies the acting player's
linear row by the selected policy likelihood and normalizes it. Other player
rows pass through. At player-node creation, the natural-input capture writes
the model's required log/range representation once. At a leaf, the linear
reach is already evaluator-ready.

Potential advantages:

- eliminate repeated `_linear_reach` conversion before every branched
  rejection-sampler call;
- eliminate leaf log-to-linear conversion;
- make the sampler consume family storage directly;
- avoid storing both active log and active linear reach;
- preserve the exact $H$ domain and scale-invariant evaluator contract.

This changes floating-point execution order, not the mathematical reach law.
It requires strict distributional, target, and gradient tolerance tests,
including zero support, extreme policy probabilities, short decks, and long
trajectories. If numerical evidence rejects linear active reach, retain log
reach in `FamilyWork`; the rest of the architecture does not depend on this
choice.

## Queue semantics

Each family has three monotone cursors:

```text
head_m
ready_tail_m
reserved_tail_m
```

They satisfy:

$$
0\le\mathtt{head}_m
\le\mathtt{readyTail}_m
\le\mathtt{reservedTail}_m
\le N_m^{\rm cap}.
$$

- `[0, head_m)` is processed and logically dead.
- `[head_m, ready_tail_m)` is available to dequeue.
- `[ready_tail_m, reserved_tail_m)` has destination storage reserved but an
  asynchronous producer has not yet committed every column.
- `[reserved_tail_m, N_m_cap)` is unused capacity.

The scheduler computes ready depth as

$$
Q_m^{\rm ready}=\mathtt{readyTail}_m-\mathtt{head}_m.
$$

It never selects pending rows. A producer publishes `ready_tail` only after
all state, history, reach, logical-address, and retained-input writes for the
contiguous append are ordered before the publication.

The dequeue operation:

1. chooses a stable prefix length;
2. returns views of every required family column;
3. advances `head`;
4. never copies payload into staging;
5. never changes an occurrence status field, because no such field exists.

The cursor interval itself proves whether a row is unused, pending, queued, or
processed.

## Capacity policy

### Schema-derived total and semantic-family split

For the initial HU 32 bb benchmark, use a schema-derived calibration plan with

$$
\sum_mN_m^{\rm cap}=25{,}000
$$

with family capacities selected from measured per-family high quantiles plus
guard. If family skew makes 25,000 awkward, 50,000 total exact reach rows still
cost only about 506 MiB in HU.

That number is a calibration point, not an HU-shaped architecture. For any
declared $P$, $H$, and realized-work target, derive the total bytes from the
actual schema, then divide the admitted row budget among only the $P$ semantic
families. The split may use lagged observed family shares and conservative
guards and may change between attempts; addresses and capacities remain fixed
within an admitted attempt. Do not allocate the total capacity independently
to every family, and do not create street capacities. The estimation problem
therefore grows linearly with the declared semantic families and remains
tractable for larger player counts.

Capacity values are physical execution parameters. They do not prescribe a
desired number of sampled decisions.

### Sequential failure

Sequential roots and logical keys are fixed before current draws. A capacity
overflow may not drop, reroll, or replace a trajectory. The capacity planner
must use the existing lagged root-width controller and conservative
per-family evidence. If an admitted sequential iteration nevertheless reaches
a hard family capacity, it releases safely, checkpoints the last completed
iteration, and stops.

### Branched failure

A branched family, node, edge, leaf, or chance-workspace overflow is a
classified whole-attempt memory/admission failure. Release the entire attempt,
perform the existing rank-common controller backoff, choose the disjoint next
attempt address, and retry within the bounded limit.

### Optional future reuse

Do not begin with a circular ring. Monotone addresses are the cleanest and
fastest implementation.

If multiplayer/deep-stack evidence later shows that total-work capacity is
unacceptable, evaluate a second-generation epoch compactor:

- compact only the live suffix when a family reaches a declared epoch
  boundary;
- perform the copy asynchronously into the alternate fixed slab;
- swap only after a stream event;
- preserve FIFO order and stable logical identities.

This must replace monotone capacity if accepted; it must not become a
permanent fallback maze.

## Unified mode behavior

### Normal sequential

- Admit a fixed root set before draws.
- Store the sampled root private profile once per live trajectory or copy its
  tiny `[P]` combo vector into each work row.
- Use offspring count one.
- Materialize the selected action, state transition, reach update, and chance
  closure through the fused sequential pipeline.
- Append the next player occurrence directly to its family.
- Capture terminal reward directly at the leaf.
- Retain only decision-order trajectory metadata; do not build all-occurrence
  sequential topology.

### Sequential Rao--Blackwell

Use the same physical engine as normal sequential, with:

- exact active reach;
- preterminal/terminal completion-ready leaf capture;
- decision-only chronological topology;
- no separate sampled-action table;
- one posterior reference per required trajectory/player after completion.

The sequential kernel receives the root-sampled private profile and does not
invoke the branched rejection sampler at every edge.

### Branched Rao--Blackwell

- Create exactly one root.
- Compute nominal/executed offspring counts with logical keys.
- Reserve every logical child/topology/edge-payload address before its event
  draw.
- Materialize every sibling independently.
- Append player descendants directly into family tails.
- Close chance descendants completely before exposing the next host scheduler
  choice.
- Release the parent's logical mutable ownership after all child destinations
  commit.
- Preserve root refill metadata and final path-mass correction.

The hot family slot is not the topology identity. `node_id` maintains the map
to the cold append-only tree.

## Cold topology and retained outputs

Branched targets still require a compact rooted topology. Keep a fixed-capacity
global topology output with:

```text
parent[N]
child_begin[N]
degree[N]
decision_partition[N]
decision_row[N]
leaf_row[N]
edge_partition[N]         # sentinel at the root and non-player edges
edge_payload_row[N]       # child-indexed reference, when payload is external
nominal_degree[N] when consumed
protected root metadata
```

It is cold during rollout:

- child ID spans are reserved contiguously;
- child rows are committed contiguously;
- every sampled player-edge field is addressed by its child node row;
- parent compute metadata comes from `FamilyWork`, not from topology gathers;
- targets consume it only after rollout.

Scalar edge data such as the sampled event may live directly in child-aligned
topology columns. Variable-width old-policy data may live in an exact
partition-local payload table, but `edge_payload_row[child]` is its sole
address. The payload table has one row per realized player child. There is no
parent-major action tensor and no capacity proportional to the maximum
realized KKT degree. Parent `child_begin` and `degree` already provide the
ragged grouping needed by targets and the learner.

Sequential modes write decision-only trajectory CSR metadata directly and
never instantiate the branched topology owner.

The fixed topology/action/leaf capacities are admitted with `FamilyWork`.
There is no page growth or allocator call during the hot rollout.

## Fused child materialization

The model forward remains a separate family-specific operation. Everything
after the forward and before the next model-ready family rows should be fused
as far as practical.

### Player-parent fused pipeline

For one bounded parent pack:

1. compute nominal/executed offspring;
2. device-scan offspring to obtain stable child offsets;
3. reserve one contiguous topology range and child-aligned edge payload rows;
4. for every logical child:
   - derive its logical RNG keys;
   - sample a collision-conditioned private profile in branched mode, or use
     the trajectory profile in sequential mode;
   - select the model row for the acting private hand;
   - sample the public player event;
   - compute selected old log probability;
   - apply the public-state transition;
   - update the active exact reach;
   - classify terminal/chance/player and player partition;
5. stably compact by destination;
6. reserve destination tail spans;
7. write player, leaf, and chance outputs directly;
8. publish ready tails and failure flags.

The complete parent `PolicyOutput` must not be copied to child shape. The
kernel consumes parent outputs through `parent_position[child]`.

### Chance-parent fused pipeline

Chance parents do not need a model output. For each chance occurrence:

1. reserve its logical children;
2. sample collision-conditioned profiles in branched mode;
3. sample keyed chance events;
4. step public state;
5. update exact reach according to the chance rule;
6. classify results;
7. append player/leaf destinations directly;
8. compact remaining chance children into the next bounded chance frontier;
9. repeat until the chance frontier is empty.

The host should observe only the final per-family ready counts and any failure
flag, not every intermediate chance wave.

## Rejection-sampler fusion

The sibling poker evaluator remains the sole implementation authority for
collision-conditioned profile sampling. Do not copy its rejection sampler
into this repository.

The current contract returns `combo_ids[B,M,P]` and performs a Python-visible
exhaustion check before the caller launches subsequent kernels. The fastest
rollout path needs a lower-level device integration.

Propose a versioned sibling deliverable:

```text
poker profile-sampler device library
  weighted-draw and collision-rejection device primitives
  immutable combo tables
  row-local versioned RNG
  device failure/attempt reporting
  no allocation and no host check
```

The PBG poker rollout extension links against that sibling-owned device
library. This preserves one sampler implementation while allowing a PBG-owned
fused transition/routing kernel to consume accepted combo IDs in registers or
shared memory.

If a linkable device ABI is impractical, the fallback interface is:

```python
sample_joint_profiles_into(
    board,
    range_weights,
    seeds,
    output_combo_ids,
    device_failure,
)
```

followed by a captured transition/route kernel. This is still preferable to a
fresh output allocation plus a per-call host exhaustion check.

Sampling exhaustion sets a device flag with the first failing logical child
address and attempt count. The host checks it once at the declared rollout or
chance-close control boundary. It may never silently drop a lane.

## Stable destination addressing without a CPU synchronization

Correct destination addresses require counts per family, leaf, and chance
continuation. This does not require a CPU round trip.

### Small-pack single-CTA path

For a bounded child pack small enough for one cooperative thread block:

1. each lane materializes and classifies one child;
2. shared-memory histograms count destinations;
3. block-exclusive scans compute stable within-destination positions;
4. one lane reserves each nonempty destination with a device `atomicAdd` on
   `reserved_tail`;
5. `__syncthreads()` publishes the returned bases;
6. lanes write directly to `base[route] + local_position`;
7. one lane publishes the destination's ready extent after a device fence.

Producer packs execute in logical order on one routing stream, so destination
FIFO order remains stable.

### Larger-frontier path

For a chance frontier requiring multiple blocks, use one of:

1. a cooperative persistent kernel with grid-wide scans and barriers; or
2. a fixed captured graph:
   - materialize/classify;
   - stable device scan/radix partition;
   - reserve destination bases;
   - scatter and form the next chance frontier.

The captured graph is the recommended first implementation. It avoids
cooperative-launch occupancy constraints and still removes Python and host
synchronization between stages. A single persistent kernel should replace it
only after a paired benchmark demonstrates a win.

Destination order is part of the physical reproducibility contract. Do not use
nondeterministic cross-block atomic reservation whose order depends on block
scheduling.

## Asynchronous execution

Use at least three logical CUDA streams:

```text
model stream
  family input pack
  current-policy forward

materialization stream
  offspring scan
  rejection sampling
  event/transition/reach update
  chance closure

append/seal stream
  routed family writes when not fused
  cold retained/topology writes
```

Fixed addresses make stream dependencies explicit.

For every asynchronous append:

- reserve a disjoint destination range;
- record the source-ready event;
- retain or `record_stream` every source workspace;
- publish `ready_tail` only after the destination-complete event;
- let the scheduler choose another family from already-ready rows;
- make the model stream wait only when it selects a pending destination.

There is no device-wide synchronization in the ordinary rollout loop.

Overlap is opportunistic. If the newly produced children are the only
remaining work, their append/chance closure is on the critical path. If
another family is ready, its model forward can hide the append.

## Kernel-fusion boundaries

Fusion is useful when it removes a transient tensor, a repeated global-memory
round trip, or a launch-sized operation without extending a large allocation's
lifetime. It is not a goal by itself.

The first production fusion targets should be:

1. **Player child materialization.** After offspring offsets are known, fuse
   logical-key construction, profile sampling when the sibling device ABI
   permits it, event sampling, transition, exact-reach update, classification,
   child-aligned edge writes, and direct destination writes. Keep the model
   forward separate.
2. **Chance closure.** Fuse one bounded chance generation from profile/event
   sampling through next-frontier or final-destination writes. Run successive
   generations device-resident through a captured loop or a measured
   persistent implementation.
3. **Small-pack routing.** Fuse destination histogram, stable local positions,
   tail reservation, and scatter in one CTA. Retain the multi-kernel
   scan/partition path for larger packs.
4. **Leaf capture.** Fuse the final reach normalization/conversion and
   completion-ready leaf write with stopped-child routing.
5. **Layered branched targets.** Within one target-layer kernel, fuse child
   reduction, conditionalization, the parent recurrence, controller
   statistics, and child-row edge-target writes. Completion remains a separate
   bounded service.
6. **Learner elementwise work.** Once rollout and targets are stable, fuse
   gathers, masks, ratio/clamp arithmetic, and reductions around the model
   forward/backward where a custom kernel removes materialized intermediates.

Keep a boundary at a library-scale model forward, a variable-size prefix scan,
a phase-lifetime transition, or an operation whose fusion would retain a
larger workspace. In particular, do not hand-fuse the model with stochastic
tree mutation, do not fuse completion into the tree recurrence, and do not
keep rollout slabs alive to save a seal copy. A persistent kernel must win a
paired end-to-end benchmark; it receives no preference merely for reducing
launch count.

## CUDA graph strategy

Fixed family addresses and bounded workspaces make selective graph capture
possible, but graph capture is an optimization layer over an allocation-free
eager path. It is not part of correctness and must never determine the storage
model. A change to an admitted family-capacity plan invalidates affected
captures; recapture only after the new plan is stable.

Good initial capture units are:

- one model-ready family pack at a common batch quantum and compact token
  width;
- the fixed multi-kernel large-frontier route/append sequence;
- one bounded chance-generation step, replayed until the device frontier is
  empty;
- a fixed-shape learner forward/loss/backward microbatch after target storage
  is immutable.

Capture variants may depend on:

```text
mode
semantic physical family
batch quantum
realized compact token width
fixed/dynamic branching control
```

Street is not a graph or queue identity. The compact-width variant is selected
for the next stable FIFO prefix. Tail rows use a validity mask or a very small
set of common quanta.

Do not capture the whole rollout iteration: scheduler choices, realized child
counts, tree sizes, layer sizes, completion counts, and distributed control
are genuinely dynamic. The flattened target recurrence should initially use
one ordinary custom launch per dependency layer. Its launch count is bounded
by tree height, not node count; bucketing and capturing layer kernels is worth
considering only if profiling later shows launch cost remains material.

The 2080 Ti admission rule is strict: reject a graph if its private pools or
retained workspaces reduce the maximum feasible forward or backward batch.
Those batches are currently memory-limited and dominate the value of a small
launch saving. On H100, recapture the same fixed-address units more
aggressively for pure throughput. In both cases, require identical outputs and
measure end-to-end time; graph capture must not silently select a different
compiled model path or stochastic topology.

## Minimal-padding stable FIFO prefixes

Keep one stable `post_A` FIFO across flop, turn, and river. This remains true
for every player count.

For the next stable prefix $Q$, compute:

$$
b_{\min}(Q)=\min_{q\in Q}|\operatorname{board}(q)|,
\qquad
H_Q=\binom{4R-b_{\min}(Q)}2.
$$

The physical packer compacts only board-legal hand rows needed by that prefix
and pads the pack only to $H_Q$. It does not change canonical retained
$H$-wide targets or outputs.

FIFO order naturally tends to expose later-street suffixes after older
early-street rows drain. A river-only head prefix therefore uses river width
without any street-specific queue or reorder.

The planner begins with the largest FIFO prefix that fits the current
forward-memory budget. It may choose between:

- one large mixed prefix at its minimum required width; and
- shortening that prefix at a public-width transition, then processing the
  following consecutive subspan at its smaller width.

It may not skip an older flop row to process a younger river row. The cost
model must charge the extra model launch and the new tail batch against saved
token pairs. This should make splitting rare on small mixed queues and useful
on the naturally river-heavy suffixes of branched rollouts, where 60--70% of
nodes may already share river width. It requires no per-street capacity guess,
no extra queue tail, and no additional scheduler family.

Required metrics include:

- street composition of every dequeued prefix;
- realized compact hand width;
- padded tokens and token pairs;
- model rows/call;
- tail-batch distribution;
- fraction of postflop rows processed at flop, turn, and river widths.

## Direct leaf capture

The fused router writes stopped children directly into the final
completion-ready leaf schema:

```text
board cards
range weights [P,H]
stratum and payoff fields
logical completion key
```

There is no intermediate live-reach pool row followed by:

```text
gather → clone → leaf append → release
```

The leaf append owns one stable destination range and writes the generated
reach once to its last semantic consumer.

Normal terminal reward is similarly written directly.

## Seal and learner handoff

At rollout completion, each family owns a realized prefix `[0,D_m)`.

The learner already consumes partition-local blocks, so there is no need to
slide all families into one global $N$-row tensor. For each family:

1. allocate exact immutable destination columns of size $D_m$;
2. issue direct contiguous copies from retained family prefixes;
3. validate exact schemas and stable row mappings;
4. release the entire overprovisioned `FamilyWork` slab, including active
   reach, mutable state, history, and unused capacity.

Because sources are already contiguous, seal needs no page routing, row-order
gather, or bounded scratch for ordinary family columns.

Branched protected-root topology may still require its declared reorder or
two-span treatment. That exception must not reintroduce generic routing for
family inputs.

The default is to perform the exact copy. Seal is currently tiny, while
retaining 25,000-row capacity through repeated learner passes would inflate
the optimization base. A zero-copy prefix view is admitted only when measured
end-to-end throughput improves and the larger optimization peak does not
reduce backward width.

## Layer-flattened branched target engine

The branched target builder should be replaced, not tuned around its current
frontier loop. Today `plan_branched_frontier` copies structure to the host and
simulates a slot allocator in Python. `build_branched_rb_targets` then copies
more topology, owner, and action-pointer data to the host; advances a Python
ready-counter tree from each completed leaf; constructs small device tensors
per ready parent; and loops in Python over that parent's sampled actions. Its
live-frontier allocation is memory-efficient, but it turns an embarrassingly
regular tree recurrence into node- and action-granular launch work.

The replacement has two explicit subphases.

### Topology-only flattening

Input is only the sealed CSR tree emitted by hot-work or retained-decision
storage, plus scalar root metadata. No model tensor, reach tensor, completion
value, or game state participates in the schedule.

Compute a root depth for every node:

$$
d(\varnothing)=0,\qquad d(\sigma a)=d(\sigma)+1.
$$

Then create an exact layer-major permutation. Within each layer, retain stable
parent and child-ordinal order, and rebuild `parent`, `child_begin`, and
`degree` under the permutation. This ordering makes every child of a parent
reside in the next layer and keeps sibling spans contiguous. Emit:

```text
FlattenedTopology
  layer_ptr[L + 1]
  source_node[N]
  source_to_flat[N]
  parent[N]
  child_begin[N]
  degree[N]
  target_degree[N]       # protected-root handling; otherwise degree
  decision_partition[N]
  decision_row[N]
  leaf_row[N]
  edge_partition[N]
  edge_payload_row[N]
```

The first implementation may compute the permutation from one bulk CPU copy
of the scalar CSR columns and upload the result once. That is bounded
structural orchestration and immediately removes the current per-node target
loop. It must be deterministic and linear in $N$. If this one transfer and
sort becomes visible on H100, replace it with device depth construction plus a
stable radix sort; the numerical target engine and its schemas do not change.

Do not use a leaf-distance level that destroys contiguous sibling spans unless
it wins a benchmark. Root depth gives the required dependency order directly:
when layers execute from deepest to root, every child row is already complete.

### Exact flattened target tree

Once $N$ is realized, allocate a new exact target tree with exactly $N$ rows:

```text
FlattenedTargetTree
  mass[N, P, H]
  moment[N, P, H]
  value_vector[N, P]       # dynamic KKT only
  cost[N]                  # dynamic KKT only
  delta[N]                 # dynamic KKT only
  decision_value[N, H]     # valid only on player-decision rows
  edge_posterior[N, H]     # pair (sigma,a), stored at child row sigma a
  edge_advantage[N, H]     # pair (sigma,a), stored at child row sigma a
  validity masks
```

Mode-absent columns are omitted by schema. The allocation is exact realized
work, not a guessed maximum and not rollout-capacity overprovisioning. Seed
leaf rows directly from bounded completion batches. Then, for
$\ell=L-2,\ldots,0$, launch one kernel over
`layer_ptr[ell]:layer_ptr[ell+1]`.

Each layer kernel:

1. reads each parent's contiguous child span;
2. reduces child mass, moment, value-vector, cost, and correction state;
3. applies the chance or player recurrence;
4. writes the completed parent row;
5. writes every $(\sigma,a)$ posterior and advantage into the corresponding
   child row $\sigma a$;
6. writes decision-level targets into the parent row.

The kernel is degree-ragged: a warp or CTA strides over the actual child span
and reduces it cooperatively. A large KKT degree increases work for that
parent but never changes an allocation shape. There is no
`[parent, max_degree, ...]` tensor and no Python loop over actions. The root's
protected target degree is a row scalar, not a special alternate tree.

After the root layer completes, compact immutable decision rows and
child-indexed player-edge rows into learner partition blocks. Every final edge
row retains a one-to-one `source_child_node`; CSR grouping, if the learner
needs it, is reconstructed from the already-ragged child spans rather than
from a maximum action width. Release the original topology, completion
outputs, and flattened target tree at their last consumers.

This deliberately exchanges some target-phase memory for regular GPU work.
For $N=20{,}000$ and $H=1326$, `mass` plus `moment` costs about 405 MiB at
$P=2$ and 1.19 GiB at $P=6$ in fp32. Two full child-aligned $H$-wide edge
targets cost another 202 MiB before schema omission or player-edge compaction.
Those bytes are exact, phase-local, and do not overlap the released rollout
slabs. The memory planner must admit them before construction. On the current
2080 Ti, never trade away a larger model forward/backward batch to retain
these buffers across phases; on H100, prefer the full regular layout unless a
measured target-phase bottleneck justifies a more complex exact layer-lifetime
allocator.

## Expected memory

For HU, $P=2$, $H=1326$, and total family capacity 25,000:

| Owner | Approximate payload |
|---|---:|
| active exact reach | 252.9 MiB |
| one 84-row child reach pack | 0.85 MiB |
| double-buffered child/next reach | 1.70 MiB |
| node IDs, keys, controller scalars | low tens of MiB or less |
| dense HU history/state | geometry-dependent, small beside reach/model workspace |

For six players at the same 25,000 total rows, active exact reach is about
758.7 MiB. At 50,000 rows it is about 1.48 GiB.

All estimates must be replaced by schema-derived allocated capacity,
alignment, workspace, and physical high-water accounting before admission.
The slab overlaps rollout inference workspace but is released before targets
and optimization.

## Why this can speed sequential mode

Sequential mode is already much faster, so avoiding regression is the first
gate. The unified engine can still improve it by removing:

- occurrence status transitions;
- handle-ring enqueue/dequeue copies;
- state/reach gathers into family packs;
- Python chance-closure iterations;
- separate rejection/profile interfaces where sequential does not need them;
- repeated small allocations around routing and transition;
- avoidable model-input reconstruction;
- seal routing from out-of-order family chunks.

Sequential mode retains its special advantages:

- one private profile per trajectory;
- offspring one;
- no branched action table;
- decision-only topology;
- no T/C path factors unless explicitly consumed;
- no sibling profile sampling.

The shared engine is a common physical protocol, not a union schema that makes
sequential rows carry branched fields.

## Why this can close the branched gap

The rewrite attacks nearly every non-model term named by the current storage
analysis:

- state/history/reach access becomes direct;
- occurrence routing disappears;
- controller input is direct;
- child materialization is fused;
- private-profile output is consumed without round-trip storage;
- destination routing is one stable pass;
- chance closure stays on device;
- player children land in model-ready order;
- later packs do not scan a larger historical tree;
- known child/topology writes remain contiguous;
- integrity checks become asynchronous device checks or end-of-phase
  validation, not repeated `torch.equal` host boundaries.

It cannot remove the genuine extra private-profile sample per branched edge or
the larger branched target recurrence. It can make those operations batched,
allocation-free, and fused with their immediate consumers.

## Performance targets

Use paired production-path measurements, not microbenchmarks alone.

Initial practical gates:

| Metric | Gate |
|---|---:|
| sequential RB rollout | no regression outside measurement noise |
| sequential RB stretch | at least 300 decisions/s on the fixed benchmark geometry |
| branched rollout first acceptance | at least 180 decisions/s on equivalent realized work |
| branched rollout stretch | 220+ decisions/s |
| branched/sequential per-decision gap | at most 1.5x stretch target |
| undeclared host synchronizations | zero |
| family dequeue payload gathers | zero |
| all-page/all-chunk hot scans | zero |
| OOM retries in accepted baseline | zero |
| branched target host node/action loops | zero |
| branched numerical target launches | one per dependency layer |

The exact current time-to-beat must be re-established from the latest accepted
production path with fixed logical seeds. Historical cross-commit rates are
motivation, not acceptance evidence.

## Instrumentation

Every iteration should report:

- family capacity, realized tail, ready high water, and maximum pending span;
- per-family head/tail depth and age;
- direct dequeued rows and bytes by column;
- child rows by destination;
- stable-route scan and append bytes;
- rejection attempts, exhaustion flags, and time;
- materialization/chance-close CUDA time;
- time hidden under model work and exposed append time;
- model calls, rows, realized token width, padded tokens, and token pairs;
- chance generations and maximum chance frontier;
- topology/edge/leaf capacity and realized counts;
- flattening time, layer count, rows per layer, and maximum/mean degree;
- target rows/s, edge rows/s, target kernel time per layer, and exact
  flattened-tree bytes;
- CUDA graph hits/misses and uncaptured fallbacks during development;
- graph-owned bytes and the batch-width delta versus uncaptured execution;
- peak allocated/reserved/NVML bytes by phase;
- allocator activity after rollout initialization;
- exact work counts, targets, losses, and gradients.

Add an early-pack/late-pack comparison. A core acceptance invariant is that
equal-shape family processing cost does not grow with accumulated topology
extent.

## Correctness tests

### All modes

- identical logical RNG keys under different physical batch widths;
- identical sampled events and public states for fixed keys;
- identical retained natural inputs and old outputs;
- exact FIFO insertion order;
- identical queue count and identity under different street mixtures;
- no street-, board-width-, or token-width-specific physical queue;
- stable-prefix width planning never skips an older row;
- no read from pending or processed family rows;
- direct-prefix alias and stride checks;
- capacity failure before partial publication;
- no CPU numerical target or rollout tensor staging; the one declared bulk
  scalar-topology flattening transfer is allowed;
- clean CUDA teardown.

### Sequential

- no reroll or replacement after admission;
- one private profile per trajectory;
- exact trajectory decision order and leaf association;
- identical normal rewards and sequential-RB completion inputs;
- capacity failure checkpoints and stops.

### Branched

- one distinguished root;
- ordinary randomized root degree;
- duplicate events retained as distinct children;
- sibling logical draws remain independent;
- contiguous child spans and one-to-one child-indexed player-edge rows;
- nominal/executed count correction;
- protected root refill metadata;
- whole-attempt-only capacity recovery;
- exact topology, targets, and controller observations;
- flatten permutation and inverse cover every realized node exactly once;
- every nonroot flattened row has a parent in the preceding layer;
- every parent recurrence reads only completed child rows;
- pair $(\sigma,a)$ outputs are written only at child row $\sigma a$;
- adversarial high KKT degree does not change any dense allocation width;
- layered-kernel output agrees with a small-tree scalar reference.

### Fused sampler

- exact evaluator contract and version reporting;
- row-local RNG invariance to pack order and neighboring rows;
- identical accepted profile distribution to the sibling reference;
- collision and board legality;
- deterministic exhaustion reporting;
- zero-support and nearly degenerate ranges;
- $P=2,\ldots,6$ and $R=4,\ldots,13$;
- no silent rejected lane.

### Numerical reach

- log-versus-linear reference comparisons across long legal trajectories;
- support masks preserved exactly;
- per-seat positive mass;
- posterior/profile distribution agreement;
- completion target agreement;
- finite model inputs;
- loss and gradient agreement within predeclared fp32 tolerance.

## Benchmark matrix

Run on idle CUDA devices with explicit targeting and clean teardown:

1. HU 32 bb, $R=13$, normal.
2. HU 32 bb, $R=13$, sequential RB.
3. HU 32 bb, $R=13$, fixed branched RB at the recorded baseline geometry.
4. HU 32 bb dynamic branching before and after warmup.
5. HU short deck with $R=4$ and $R=8$.
6. Six-player shallow and 128 bb stress configurations.
7. Tiny-tree refill and large-tree drainage cases.
8. Family-skew cases near each configured capacity.
9. Rejection-sampler high-attempt and exhaustion injection.
10. Mixed flop/turn/river FIFO heads at $P=2$ and $P=6$.
11. Deep, bushy, and single-high-degree target topologies with equal node
    counts.

Report per seed:

- rollout seconds/player decision;
- rollout seconds/total node;
- end-to-end decisions/s;
- exact work and model/evaluator rows;
- phase peaks;
- family capacity utilization;
- exposed versus overlapped append time;
- early versus late pack cost;
- width-planner saved token pairs versus added partial launches;
- target flatten time, layer-kernel time, rows/s, launches, and exact bytes;
- captured versus uncaptured batch width and end-to-end throughput;
- target/gradient agreement.

## Proposal summary

The architecture is:

> `FamilyWork` is the sole hot player-occurrence owner; its stable contiguous
> prefixes are the model packs; child kernels route directly into destination
> tails; chance closure is device-resident; queue identity never depends on
> poker street; and branched targets execute on an exact layer-flattened tree
> with edge data stored at child rows.

For HU, the memory price of a 25,000-row exact-reach capacity is modest. For
higher player counts, the schema-derived phase planner—not a multiplication of
street queues—must admit the exact bytes. The potential reward is removal of
the physical distinction that currently makes branched cost grow with its
accumulated tree. Sequential mode receives the same direct, allocation-free
queue engine without branched-only fields or sampling work.

The design should aim for a world in which branching is slower only because it
samples and learns from a larger stochastic object—not because every later
family pack pays to rediscover where its state and reach were stored.

## Implementation phases

Every phase below ends in one production-ready implementation: fixed-seed
correctness, hard-ceiling memory admission, CUDA benchmarks, failure cleanup,
and deletion of the code it supersedes. Temporary evaluation scaffolding does
not survive phase acceptance, and the production tree has no compatibility
switch.

### Phase 0: freeze contracts and measurements

- Record current rollout, completion, target, forward, backward, and optimizer
  timings on the fixed benchmark matrix.
- Record street mixtures inside each semantic FIFO, prefix widths, partial
  batch counts, target layer candidates, degree histograms, and phase peaks.
- Freeze the flattened topology, child-indexed edge, `FamilyWork`, and exact
  memory-accounting schemas.
- Decide the required authoritative capacity/lifetime amendment before the
  monotone `FamilyWork` replacement. Phases 1--3 do not require that rollout
  storage change.

Stable result: unchanged production behavior with enough evidence to detect
cost migration or a reduction in feasible training batch width.

### Phase 1: replace branched target execution first

- Implement the deterministic topology-only flattener using one bulk
  structural host transfer.
- Allocate the exact flattened target tree and seed its leaf rows through the
  existing bounded completion service.
- Implement one custom CUDA target kernel per root-depth layer, including
  player/chance recurrence, KKT statistics, and child-row edge targets.
- Seal the results into the current learner partitions, then delete
  `plan_branched_frontier`, the Python ready/slot allocator, and the per-parent
  and per-action target loops.

Stable result: identical targets, losses, gradients, and controller
observations with target launch count proportional to tree depth rather than
nodes or edges. This is the highest-confidence independent speedup and should
land before the rollout storage rewrite.

### Phase 2: make child-indexed edge storage end-to-end

- Reserve player-edge payloads one-to-one with realized child nodes.
- Move event, selected old log probability, posterior, advantage, and every
  other $(\sigma,a)$ field to the corresponding child row.
- Update retained-decision sealing and learner sampling to use child spans.
  A compact partition CSR pointer may remain as structural grouping, but no
  pair payload lives on a parent row and no allocation uses maximum degree.
- Remove the parent action-range allocator and the Phase 1 conversion adapter.

Stable result: rollout, target, and learner agree on one ragged child-row
identity, including adversarial KKT degrees.

### Phase 3: add minimal-padding FIFO planning

- Keep exactly the semantic `pre` and `post_A` queues.
- For the next FIFO head, compare the largest memory-feasible mixed prefix with
  consecutive width-transition cut points.
- Compact legal hand rows only to the selected prefix width, charge each
  additional launch in the planner, and use validity masks for small tails.
- Accept only an end-to-end win with unchanged FIFO order and no decrease in
  feasible forward/backward batch size.

Stable result: most river-heavy branching benefit without another queue,
capacity estimate, scheduler family, or graph family.

### Phase 4: replace sequential hot rollout ownership

- Implement schema-derived `FamilyWork` slabs, monotone
  head/reserved/ready cursors, exact byte accounting, and async publication.
- Make sequential normal and RB model packs direct FIFO prefix views while
  retaining their current event/profile calls initially.
- Seal exact immutable prefixes and release rollout slabs before targets.
- Remove the sequential occurrence queue, dequeue staging, hot reach/state
  gathers, and superseded schemas after the no-regression gate passes.

Stable result: the simpler modes prove the cursor, lifetime, failure, and seal
protocol without paying for branched-only columns.

### Phase 5: replace branched hot ownership and routing

- Store all live branched state, history, reach, controller scalars, and node
  identity directly in `FamilyWork`.
- Reserve cold CSR child spans and child-indexed edge rows before random draws.
- Route children through the stable single-CTA path or bounded multi-block
  scan path directly into semantic family, leaf, or chance destinations.
- Remove the occurrence arena, active-reach pool, handle queues, paged hot
  topology access, per-destination sorts, and branched compatibility fields.

Stable result: later equal-shape packs cost the same as early packs, one failed
attempt publishes nothing partial, and the production path has no indirect hot
owner.

### Phase 6: fuse stochastic materialization and chance closure

- Add the sibling-owned no-allocation sampler device ABI, or its
  `sample_joint_profiles_into` fallback.
- Fuse player sampling, event selection, transition, reach update,
  classification, child-edge writes, and direct routing at the boundaries
  specified above.
- Keep chance generations device-resident and report exhaustion once at the
  declared control boundary.
- Remove intermediate profile tensors, Python chance-generation loops, and
  redundant state/reach traffic.

Stable result: allocation-free dequeue-to-enqueue execution with exact logical
RNG identities and evaluator-equivalent sampling.

### Phase 7: add selective graphs, overlap, and learner fusion

- Capture only repeated fixed-address units that beat eager execution
  end-to-end.
- Add stream overlap between a ready family forward and independent
  materialization/append work.
- On 2080 Ti, do not retain captures whose memory pools reduce forward or
  backward batch width. On H100, retune graph quanta and concurrency for
  throughput.
- Profile the layered target engine before considering bucketed layer capture;
  never replace it with a whole-tree dynamic graph.
- Fuse learner elementwise/reduction work after rollout and target peaks are
  stable, then retune forward/backward batches.

Stable result: graphs and fusion are removable performance choices over the
same storage contract, with captured bytes, batch-width effects, and
end-to-end wins recorded explicitly.

### Phase 8: deletion and production audit

- Prove every production and evaluation entry point uses the replacement
  owners and child-indexed target path.
- Delete obsolete allocators, schemas, staging conversions, configuration
  fields, tests, and stale implementation documentation.
- Re-run the full CUDA correctness and benchmark matrix on idle devices,
  verify clean teardown, and establish separate accepted 2080 Ti and H100
  tuning profiles without changing semantics.

Stable result: one production architecture, no dormant fallback, semantic
queue cardinality, exact phase lifetimes, and measurements that explain every
remaining branched/sequential throughput gap.
