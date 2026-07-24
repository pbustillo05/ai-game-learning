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

1. reserves logical topology and the parent's contiguous
   `ActionProbability` span before any draws;
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
- child and player-action source spans reserved before their random draws;
- counter-based logical RNG addresses independent of queue order and physical
  packing;
- stable insertion-order FIFO prefixes;
- the unchanged player/chance event laws and reach update;
- no dropped, truncated, replayed, or silently rerolled sequential trajectory;
- whole-attempt-only branched memory recovery;
- exact retained inputs, topology, and leaves.

One principal intentional conflict is with the current physical contract in
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

A second intentional contract change is the preflop RB target and loss domain.
The current pseudocode declares
$X_{\mathsf{pre}}=X_{\mathsf{post}_A}=H$. This proposal instead uses the
model-native preflop quotient $K=R^2$ for every preflop posterior, moment,
value, advantage, and loss row, while postflop remains on
$H=\binom{4R}{2}$. Exact-hand completion output is aggregated to preflop
classes by summing posterior mass and posterior-weighted moment before
conditionalization. This changes the finite-sample preflop estimator and
clipped loss, not only storage. Production acceptance therefore also requires
an explicit amendment to `core_math.tex` and `storage_lifecycle.tex`; until
that approval, the class-native target engine is proposal semantics rather
than authority.

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
16. Branched learner loss execution consumes exactly the realized flattened
    target-child rows and never pads a decision batch to a maximum action
    count.
17. Preflop target propagation and losses use the model-native $K$ hand-class
    quotient; postflop remains exact-hand $H$.

## Non-goals

This rewrite does not:

- change the branching intensity law;
- share one sampled private profile among branched siblings;
- merge duplicate public-event children;
- change completion sampling;
- reorder retained target rows by a private quantity;
- introduce CPU rollout state;
- introduce a CPU or automatic device fallback;
- change the PPO/RB objectives outside the explicit preflop
  exact-hand-to-class quotient;
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

During rollout, selected player-event data uses one exact partition-local
`ActionProbability[m]` source table:

```text
action_ptr[D_m + 1]
event[A_m]
old_log_prob[A_m, Y_m]
```

The rows for one player decision $\sigma$ are contiguous and aligned in order
with its target-eligible child prefix. They are reserved before the draws and
contain no maximum-child dimension. At target flattening, `event` and
`old_log_prob` are copied into the corresponding flattened child rows and the
source table is released. Posterior and advantage are later written into those
same child rows by the layered target kernel.

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

## Fixed failure records and preallocated workspaces

Every admitted attempt owns a fixed-address device failure record:

```text
DeviceFailure
  status
  phase
  failure code
  semantic family / owner
  logical node or child address
  requested extent
  available extent
  sampler attempt count when applicable
```

The first failing lane claims the record atomically. Diagnostics identify the
logical work item, not its physical pack or lane, so changing batch width does
not change the reported failure. Kernels may leave writes only in unpublished
reserved ranges after a failure; no family, topology, `ActionProbability`, or
leaf ready tail advances. An unpublished action reservation is not
learner-visible. The host reads the record once at the declared control
boundary and applies the sequential stop or whole-attempt branched recovery
rule above.
There is no per-lane error allocation or Python-visible check between fused
stages.

All hot transient storage is preallocated from the admitted schema and
physical bounds. This includes operation-specific scan storage, child routing
columns, sampler output when it cannot be consumed in-kernel, stable
destination counts and offsets, ping-pong chance frontiers, completion
microbatches, and learner flattened-child gather/reduction scratch. Workspaces may
be separate when their simultaneous use enables stream overlap; they are not
forced into one universal arena. Allocation, accounting, graph capture, and
failure injection use the same declared capacities. No ordinary rollout,
target, or learner kernel grows a workspace or invokes the allocator.

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
- Reserve every logical child/topology span and its exact
  `ActionProbability` span before the event draws.
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
nominal_degree[N] when consumed
protected root metadata
```

It is cold during rollout:

- child ID spans are reserved contiguously;
- child rows are committed contiguously;
- each player decision's `ActionProbability` source span has the same order as
  its target-eligible child prefix;
- parent compute metadata comes from `FamilyWork`, not from topology gathers;
- targets consume it only after rollout.

`ActionProbability[m]` has one row per realized target-eligible player child,
grouped contiguously by source decision. The source `action_ptr` is a
rollout/seal address only. Topology flattening aligns those rows with the
corresponding child nodes, after which parent `child_begin` and
`target_degree` provide the sole ragged grouping needed by targets and the
learner. There is no parent-major action tensor and no capacity proportional
to the maximum realized KKT degree.

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
3. reserve one contiguous topology range and the exact
   `ActionProbability` span;
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
   `ActionProbability` writes, and direct destination writes. Keep the model
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
one ordinary mixed-domain custom launch per non-leaf root-depth layer. Its
launch count is bounded by tree height, not node count; bucketing, splitting
pre/post specializations, or capturing layer kernels is worth considering only
if profiling later shows an end-to-end win.

The 2080 Ti admission rule is strict: reject a graph if its private pools or
retained workspaces reduce the maximum feasible forward or backward batch.
Those batches are currently memory-limited and dominate the value of a small
launch saving. On H100, recapture the same fixed-address units more
aggressively for pure throughput. In both cases, require identical outputs and
measure end-to-end time; graph capture must not silently select a different
compiled model path or stochastic topology.

A capture is accepted only by a paired end-to-end benchmark against the
allocation-free eager execution of the same unit. The comparison includes
graph-owned allocator pools, retained workspace lifetimes, warmup and
recapture frequency, tail handling, scheduler overlap, and any change in the
largest feasible forward or backward batch. Kernel-time or launch-count wins
alone are insufficient. If a capture does not improve complete-iteration
throughput on a supported hardware profile, omit that capture rather than
retaining an unused optimization path.

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

The reservation and publication are transactional. A stopped child never
enters a player family, chance frontier, generic occurrence owner, or
intermediate reach pool. The router writes every field required by the sibling
completion binding in its final dtype and layout, then publishes the leaf
range only after all columns commit. Completion consumes stable contiguous
leaf subspans directly; its bounded output workspace does not reconstruct or
repack a discarded rollout representation.

Normal terminal reward is similarly written directly into its final leaf row.

## Seal and learner handoff

At rollout completion, each family owns a realized prefix `[0,D_m)`.

The learner's model forwards still consume partition-local input blocks, so
there is no need to slide those inputs into one global $N$-row tensor. The
later layer-flattened target tree stores output targets and source-row
mappings; it does not replace these family-specific model-input schemas. For
each family:

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

## Layered GPU execution of branched targets

The branched target builder should be replaced, not tuned around its current
frontier loop. Today `plan_branched_frontier` copies structure to the host and
simulates a slot allocator in Python. `build_branched_rb_targets` then copies
more topology, owner, and action-pointer data to the host; advances a Python
ready-counter tree from each completed leaf; constructs small device tensors
per ready parent; and loops in Python over that parent's sampled actions.

The replacement contract is explicit:

> After one topology-flattening phase, every numerical branched-target
> recurrence executes on GPU in deepest-to-root dependency-layer order. The
> initial path launches one custom CUDA target kernel per non-leaf root-depth
> layer, never one launch per node or sampled action.

Completion remains a separate bounded service. The host may advance the
root-depth loop and launch its kernels, but it never performs a numerical
recurrence, constructs per-parent tensors, or loops over a parent's actions.

### Topology-only schedule construction

Input is the sealed CSR tree, `ActionProbability` pointers, and scalar
protected-root metadata. Compute a root depth for every node:

$$
d(\varnothing)=0,\qquad d(\sigma a)=d(\sigma)+1.
$$

Create an exact layer-major permutation. Within each layer, retain stable
parent and child-ordinal order, and rebuild `parent`, `child_begin`, and
`degree` under the permutation. Every child of a parent is therefore in the
next root-depth layer and every sibling span remains contiguous. Emit the
global structural columns:

```text
LayeredTargetTree
  layer_ptr[L + 1]
  source_node[N]
  source_to_flat[N]
  parent[N]
  child_begin[N]
  degree[N]
  target_degree[N]          # protected root; otherwise degree
  nominal_degree[N]         # dynamic controller only
  target_partition[N]       # pre or post
  target_row[N]             # row inside that target partition
  model_partition[N]        # pre or post_A learner/model family
  source_decision_row[N]
  leaf_row[N]
```

Target partitions are numerical domains, not rollout queues:

$$
\mathcal X_{\mathsf{pre}}=\mathcal K,\quad |\mathcal K|=K=R^2,
\qquad
\mathcal X_{\mathsf{post}}=\mathcal H,\quad
|\mathcal H|=H=\binom{4R}{2}.
$$

All board-empty nodes, including the flop-dealing chance parent and a
preflop terminal/completion leaf, use `pre`. Flop, turn, and river nodes use
`post`, regardless of live-player count. Thus
$N_{\mathsf{pre}}+N_{\mathsf{post}}=N$. This two-way target partition neither
creates a rollout queue nor changes poker's `pre` and `post_A` rollout
families. `model_partition` remains available because learner forwards still
depend on live-player count postflop. The logical flop chance node remains in
the topology even when chance closure is device-resident, so no player edge
crosses directly from a $K$-wide preflop parent to an $H$-wide postflop child.
All children of one parent therefore share a target partition; stable
filtering into the two target blocks keeps each sibling span contiguous there
as well as in the global node order.

The first implementation may compute the permutation from one bulk CPU copy
of scalar CSR columns and upload it once. That is bounded structural
orchestration. It must be deterministic and linear in $N$. If the transfer and
sort become visible on H100, replace them with device depth construction and a
stable radix sort; the numerical target schemas and layer recurrence do not
change.

### Persistent final-output tree

Allocate two structure-of-arrays blocks under the one logical flattened tree:

```text
TargetBlock[pre]                         TargetBlock[post]
  flat_node[N_pre]                        flat_node[N_post]
  event[N_pre]                            event[N_post]
  old_log_prob[N_pre, K]                  old_log_prob[N_post, H]
  old_value[N_pre, K]                     old_value[N_post, H]
  frozen_trace_variance[N_pre]            frozen_trace_variance[N_post]
  frozen_cost[N_pre]                      frozen_cost[N_post]
  edge_posterior[N_pre, K]                edge_posterior[N_post, H]
  edge_advantage[N_pre, K]                edge_advantage[N_post, H]
  decision_value[N_pre, K]                decision_value[N_post, H]
  trace_variance[N_pre]                   trace_variance[N_post]
  trace_variance_mask[N_pre]              trace_variance_mask[N_post]
  cost_target[N_pre]                      cost_target[N_post]
  edge_valid[N_pre]                       edge_valid[N_post]
  decision_valid[N_pre]                   decision_valid[N_post]
```

Mode-absent columns are omitted by schema. Optional rows are deliberately
dense within each target block: a node reached from a player parent may own
edge outputs even when it is a chance node or leaf; a player node owns
decision outputs even when its incoming edge was chance; and a same-street
player child commonly owns both. Independent validity masks express these
overlaps. Refill-only root children outside `target_degree[0]` have no root
edge outputs.

During flattening, copy each source `ActionProbability[m]` span into the
corresponding target-eligible flattened child rows. This populates `event` and
natural-width `old_log_prob`; copy each player node's natural-width old value
and frozen controller scalars at the same phase. Validate stable
parent/child-ordinal alignment, then release the source topology and
`ActionProbability` stores before allocating the large numerical workspace.
The final posterior, advantage, value, trace, and cost columns remain
uninitialized until their layer kernel writes them.

### Class-native preflop targets

Let $c:\mathcal H\to\mathcal K$ be the ordinary preflop hand-class map and
define the linear aggregation

$$
(Gz)_k=\sum_{h:c(h)=k}z_h.
$$

Whenever exact-hand completion or a postflop child feeds a preflop-domain
node, aggregate posterior mass and posterior-weighted moment separately:

$$
\widehat{\mathbb P}_{p,k}
=\sum_{h:c(h)=k}\widehat{\mathbb P}_{p,h},
\qquad
\widehat M_{p,k}
=\sum_{h:c(h)=k}\widehat M_{p,h}.
$$

Conditionalize only after aggregation. This gives the class conditional

$$
\widehat V_{p,k}
=\operatorname{Cond}(\widehat M_{p,k},\widehat{\mathbb P}_{p,k})
$$

and class-native posterior, advantage, value, and loss rows. The conversion is
one-way during backward propagation:

```text
post <- post    H -> H
pre  <- pre     K -> K
pre  <- post    H -> K     # flop chance boundary
```

There is no `post <- pre` recurrence. A preflop completion leaf likewise
aggregates its evaluator-returned $H$-wide mass and moment directly into its
$K$-wide layer-work row. The aggregation preserves the all-player Monte Carlo
vector because summing moment over classes equals summing it over exact hands.

The learner consequently sums preflop losses over $K$ classes, not $H$ exact
hands mapped repeatedly to the same model row. This removes suit-level Monte
Carlo noise before the nonlinear clipped loss. It is an intentional estimator
change and is subject to the theory reconciliation stated above.

### Adjacent-layer numerical workspace

Final outputs persist for the learner, but mass, moment, Monte Carlo vector,
propagated cost, and $\Delta$ are needed only until the parent layer consumes
them. Allocate an exact phase-local workspace for two adjacent layers:

```text
LayerWork[pre]
  mass[rows, P, K]
  moment[rows, P, K]
  value_vector[rows, P]       # dynamic KKT only
  work_cost[rows]             # dynamic KKT only
  delta[rows]                 # dynamic KKT only

LayerWork[post]
  mass[rows, P, H]
  moment[rows, P, H]
  value_vector[rows, P]       # dynamic KKT only
  work_cost[rows]             # dynamic KKT only
  delta[rows]                 # dynamic KKT only
```

Let $W_{\ell,\mathsf{pre}}$ and $W_{\ell,\mathsf{post}}$ be the two target
widths within root-depth layer $\ell$. The exact workspace byte requirement is
the maximum over adjacent layers:

$$
\max_\ell\left[
  (W_{\ell,\mathsf{pre}}+W_{\ell+1,\mathsf{pre}})S_{\mathsf{pre}}
  +(W_{\ell,\mathsf{post}}+W_{\ell+1,\mathsf{post}})S_{\mathsf{post}}
\right],
$$

where the dominant fp32 row payloads are

$$
S_{\mathsf{pre}}=4(2PK+P+2),\qquad
S_{\mathsf{post}}=4(2PH+P+2)
$$

when dynamic $T/C$ work is present. Fixed branching omits the absent scalar and
vector columns. This workspace replaces full
`mass[N,P,H]`/`moment[N,P,H]` target allocations. Its source and destination
subspans are disjoint for one launch and swap roles after the layer completes.

Leaves may occur at different depths. Seed the leaf positions of a layer
through bounded completion before that layer becomes the next kernel's child
input. A preflop leaf is projected $H\to K$ while seeding; a postflop leaf is
written at $H$. Completion and the layer kernel use events rather than a
device-wide synchronization.

### One CUDA target launch per dependency layer

After seeding the deepest layer, execute
$\ell=L-2,\ldots,0$. One custom CUDA launch covers every internal parent in
`layer_ptr[ell]:layer_ptr[ell+1]`. Each CTA or cooperative warp owns one
parent, reads its contiguous target-child prefix from the layer-$\ell+1$
workspace, and dispatches one of `pre <- pre`, `pre <- post`, or
`post <- post` from the immutable row descriptor.

Within that one launch, the kernel:

1. reduces child posterior mass, moment, Monte Carlo value vector, propagated
   cost, and $\Delta$;
2. writes the completed parent work row in the destination layer span;
3. at a player parent, writes every target child's posterior and advantage
   directly into that child's persistent flattened row;
4. writes the parent's value, trace-variance, and cost targets into its
   persistent flattened row;
5. only then bootstraps the acting-player moment and working cost propagated to
   ancestors.

For $n=\mathtt{targetDegree}(\mathsf s)\ge2$, reduce

$$
\mathbf v_{\rm sum}=\sum_i\widehat{\mathbf V}(\mathsf s\mathsf a_i),
\qquad
q_{\rm sum}=\sum_i\|\widehat{\mathbf V}(\mathsf s\mathsf a_i)\|_2^2
$$

and compute in the same launch

$$
\widehat T^{\rm exec}(\mathsf s)
=\frac{q_{\rm sum}
-n\left\|\mathbf v_{\rm sum}/n\right\|_2^2}{n-1},
\qquad
\overline\Delta(\mathsf s)
=\frac1n\sum_i\Delta(\mathsf s\mathsf a_i).
$$

At a player node, write

$$
\widehat T(\mathsf s)
=\max\{0,\widehat T^{\rm exec}(\mathsf s)-\overline\Delta(\mathsf s)\}
$$

when $n\ge2$ and leave the trace mask false when $n=1$. For correction
propagation use $T^*=\widehat T$ when present and the frozen
$T^{\theta_k}(\sigma)$ otherwise:

$$
\Delta(\mathsf s)
=\frac{T^*(\mathsf s)+\overline\Delta(\mathsf s)}{n}
-\frac{T^*(\mathsf s)}{\widetilde N(\mathsf s)}.
$$

A chance node writes no decision targets and propagates
$\Delta(\mathsf s)=\overline\Delta(\mathsf s)/n$. The refilled root uses only
its protected `target_degree` prefix for posterior, advantage, value, cost,
trace covariance, and $\Delta$.

The kernel is degree-ragged: a warp or CTA strides over the actual child span
and reduces it cooperatively. A large degree increases work for that parent
but never creates `[parent,max_degree,...]` storage. If profiling later proves
that separate pre/post specialized launches beat the single mixed-domain
kernel, a fixed small number of launches per dependency layer is acceptable;
the invariant remains that launch count is proportional to tree depth, never
nodes or actions.

After the root completes, release completion outputs and the entire adjacent-
layer workspace. Retain the flattened structural child spans and persistent
pre/post target blocks through optimization. There is no post-target
action/decision compaction: the learner reads final node rows directly and
releases the flattened tree after the last optimizer step.

For $N=20{,}000$ and $H=1326$, three all-postflop $N\times H$ final fp32
columns for posterior, advantage, and value cost about 303.5 MiB. Preflop rows
replace $H$ with $K$ and use only $K/H$ of those vector bytes, a reduction
factor of $H/K$. The invalid final rows are deliberate; poker's player-node
and player-edge densities make their overhead preferable to another scattered
owner and compaction phase. The memory planner must separately admit
persistent final outputs and the exact maximum-adjacent-layer workspace before
numerical target execution.

## Exact ragged branched learner execution

After target construction, every player decision's sampled
$(\sigma,a)$ rows are its contiguous flattened target-child prefix. The
learner must preserve that ragged domain instead of turning a selected
decision subspan into
`[decision_rows, max_actions_per_decision]`.

For a selected decision-index subspan $J$, build in one preallocated workspace:

```text
local_edge_ptr[|J| + 1]
child_row[E_J]
edge_to_local_decision[E_J]
```

where $E_J$ is the exact sum of the selected nodes' `target_degree` values.
The physical learner planner admits both $|J|$ and $E_J$ and selects the
largest decision subspan whose exact child extent fits. It never introduces an
immutable or transient maximum-action dimension.

The current-policy forward remains decision-major. The policy adapter reads
the appropriate decision output through `edge_to_local_decision` and evaluates
only real child-row events. Posterior-weighted policy terms, ratios, clipping
statistics, and edge contributions are reduced back to decisions with
segmented reductions over `local_edge_ptr`. Preflop rows use the flattened
tree's $K$-wide class targets and postflop rows use its $H$-wide exact-hand
targets. There are no inactive edge slots, invented safe events, or
child-count-dependent copies of the complete model output.

`max_actions_per_decision` is therefore not part of immutable training
storage, workspace admission, or a production loss API. Required physical
bounds are expressed as decision rows and realized flattened child rows per
learner subspan.

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
| branched numerical target launches | one mixed-domain launch per non-leaf dependency layer initially |
| learner maximum-action padding | zero |

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
- topology/action/leaf capacity and realized counts;
- fixed failure status, code, phase, logical address, requested/available
  extent, and operation-specific workspace capacity/high water;
- flattening time, layer count, pre/post rows per layer, maximum adjacent-layer
  workspace bytes, $H\to K$ projection rows/time, and maximum/mean degree;
- target rows/s, action-target rows/s, CUDA launches and kernel time per layer, and
  persistent pre/post flattened-tree bytes;
- learner decision rows, exact flattened child rows, segmented-reduction time,
  and maximum-action padding rows, which must remain zero;
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
- deterministic first-failure records under different physical pack widths;
- no allocator activity after the admitted phase workspaces are initialized;
- stopped children appear only in final completion-ready or normal-reward leaf
  rows, never in an intermediate occurrence or reach owner;
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
- every source `ActionProbability` span matches its decision's
  target-eligible child prefix in event and child-ordinal order;
- nominal/executed count correction;
- protected root refill metadata;
- whole-attempt-only capacity recovery;
- exact topology, targets, and controller observations;
- flatten permutation and inverse cover every realized node exactly once;
- every nonroot flattened row has a parent in the preceding layer;
- every flattened node belongs to exactly one `pre` or `post` target block;
- preflop completion and `pre <- post` boundary rows agree with a reference
  class aggregation of exact-hand mass and moment;
- class aggregation preserves each player's summed moment and therefore the
  all-player Monte Carlo value vector and trace calculation;
- every parent recurrence reads only completed child rows;
- pair $(\sigma,a)$ outputs are written only at child row $\sigma a$;
- final preflop posterior, advantage, value, and loss rows have width $K$;
- final postflop posterior, advantage, value, and loss rows have width $H$;
- numerical target launch count equals the number of non-leaf dependency
  layers for the initial mixed-domain kernel;
- all-node mass, moment, value-vector, cost, and $\Delta$ allocations are
  absent; their workspace high water matches the admitted adjacent-layer plan;
- adversarial high KKT degree does not change any dense allocation width;
- layered-kernel output agrees with a small-tree scalar reference;
- branched learner subspans execute exactly their flattened target-child rows;
- one adversarial high-degree decision changes realized pair work but does not
  create a maximum-action allocation or inactive event evaluation.

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
- target flatten time, pre/post rows per layer, $H\to K$ time, adjacent-layer
  workspace high water, layer-kernel time, rows/s, launches, and exact bytes;
- captured versus uncaptured batch width and end-to-end throughput;
- postflop exact-hand and preflop class-aggregated target/gradient agreement.

## Proposal summary

The architecture is:

> `FamilyWork` is the sole hot player-occurrence owner; its stable contiguous
> prefixes are the model packs; child kernels route directly into destination
> tails; chance closure is device-resident; queue identity never depends on
> poker street; and branched targets execute deepest-to-root on one persistent
> layer-flattened output tree with $K$-wide preflop and $H$-wide postflop
> blocks, child-row action outputs, and an adjacent-layer numerical workspace.

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
- Freeze the flattened topology, `ActionProbability` source, pre/post target
  blocks, adjacent-layer workspace, `FamilyWork`, and exact memory-accounting
  schemas.
- Freeze the device failure record and every operation-specific workspace
  schema, capacity, lifetime, and publication boundary.
- Obtain the required authoritative decision on class-native preflop RB
  targets/losses and the $H\to K$ mass/moment aggregation before implementing
  the target replacement.
- Decide the required authoritative capacity/lifetime amendment before the
  monotone `FamilyWork` replacement. Phases 1--3 do not require that rollout
  storage change.

Stable result: unchanged production behavior with enough evidence to detect
cost migration or a reduction in feasible training batch width.

### Phase 1: replace branched target execution first

- Implement the deterministic topology-only flattener using one bulk
  structural host transfer.
- Allocate the persistent $K$-wide preflop and $H$-wide postflop output blocks,
  populate old action/value rows, and admit the exact maximum-adjacent-layer
  workspace.
- Seed leaf rows through the bounded completion service, aggregating
  exact-hand mass/moment to classes for preflop-domain leaves.
- Implement one mixed-domain custom CUDA target launch per non-leaf
  root-depth layer, including `pre <- pre`, `pre <- post`, `post <- post`,
  player/chance recurrence, $T/C/\Delta$, and direct child/decision target
  writes.
- Make the flattened tree the final learner target owner, then delete
  `plan_branched_frontier`, the Python ready/slot allocator, and the per-parent
  and per-action target loops.

Stable result: postflop targets agree with the prior exact-hand reference;
preflop targets, losses, and gradients agree with an approved class-aggregated
reference; controller observations agree; temporary bytes follow adjacent
layer width rather than $N$; and numerical target launch count equals the
number of non-leaf root-depth layers.

### Phase 2: make flattened action outputs end-to-end

- Use exact partition-local `ActionProbability[m]` source rows during rollout,
  with every decision span aligned to its target-eligible child prefix.
- During flattening, populate event and old log probability into the
  corresponding child rows, validate the mapping, and release
  `ActionProbability` before numerical targets.
- Make learner sampling derive exact action rows only from flattened
  `child_begin` and `target_degree`; no final `action_ptr`, action-table owner,
  target compaction, or maximum-action dimension survives.
- Remove the Phase 1 source-conversion adapter and every superseded
  child/action payload schema.

Stable result: `ActionProbability` is a rollout-to-flatten source only, while
the persistent flattened child row is the sole final action-output and learner
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
- Reserve cold CSR child spans and exact `ActionProbability` spans before
  random draws.
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
  classification, `ActionProbability` writes, and direct routing at the
  boundaries specified above.
- Write stopped children directly into the final completion-binding or normal
  reward leaf schema and publish the leaf span only after every column commits.
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
  stable. Replace the fixed-width branched action expansion with exact
  flattened-child gathers and segmented decision reductions, remove
  `max_actions_per_decision`, then retune forward/backward batches.

Stable result: graphs and fusion are removable performance choices over the
same storage contract, with captured bytes, batch-width effects, and
end-to-end wins recorded explicitly.

### Phase 8: deletion and production audit

- Prove every production and evaluation entry point uses the replacement
  owners, class-native preflop targets, persistent flattened output tree, and
  layered CUDA target path.
- Delete obsolete allocators, schemas, staging conversions, configuration
  fields, tests, and stale implementation documentation.
- Re-run the full CUDA correctness and benchmark matrix on idle devices,
  verify clean teardown, and establish separate accepted 2080 Ti and H100
  tuning profiles without changing semantics.

Stable result: one production architecture, no dormant fallback, semantic
queue cardinality, exact phase lifetimes, and measurements that explain every
remaining branched/sequential throughput gap.
