# `node_budget` allocation audit and retirement proposal — 2026-07-21

## Conclusion

`rollout.node_budget` should be retired from production configuration and from
the runtime storage contract.

It is currently much more than a diagnostic ceiling. It determines the size
of several eagerly allocated CUDA tensors, supplies hard capacities to nominally
chunked structures, bounds scheduler loops, and enters conservative memory
plans for objects that are actually constructed from realized rows. This mixes
four distinct concepts:

1. the exact maximum length of one legal NLHE trajectory;
2. the number of nodes realized by a branching process;
3. the number of rows currently stored or in flight;
4. the number of bytes that can safely coexist in a CUDA phase.

Only the first is a game invariant. The second is a random outcome. The third
should drive append-only allocation. The fourth is the actual safety limit.
A user-supplied node count is not the correct representation of any of them.

The replacement should be page-based, byte-admitted storage that grows with
realized work. Sequential progress should use the exact game-derived path
bound. Branched progress and multiplicity should be governed by byte headroom,
mandatory-continuation reserves, and the branching memory controller rather
than by a global node capacity.

Retiring `node_budget` will remove real waste and substantially improve the
quality of memory forecasting. It will not by itself solve every OOM: model
forward workspaces can dominate total peaks, so dynamically sized family
batches and a closed-loop branching memory governor are still required.

## Why the preallocation exists

The implementation chose one flat row-index domain shared by poker states,
tree nodes, reaches, queues, and caches. A state or node handle is an int32 row
that can be used directly in ordinary tensor indexing. Preallocating the full
domain provided several convenient properties:

- stable handles without reallocation or pointer invalidation;
- direct batched CUDA gather/scatter against ordinary tensors;
- no device allocation decisions in the scheduler hot path;
- simple transactional “does this append exceed capacity?” checks;
- predictable upper-bound accounting before sampling;
- reuse of the backend state table across iterations.

Those are valid implementation goals, but `node_budget` was an expedient way
to obtain them, not a necessary consequence. A paged append-only row domain can
keep stable logical handles, device-local batched access, transactional
admission, and cross-iteration reuse without allocating the unobserved tail.
The current design optimized storage simplicity by turning a random realized
tree size into a user-provided physical capacity; that is the abstraction that
should be removed.

## What is allocated from `node_budget` today

For a rollout width $G$ and configured `node_budget` $M$, bundle construction
sets a single capacity

$$
K = GM.
$$

In branched mode $G=1$. In sequential modes, bundle construction uses the
configured maximum roots per GPU, not necessarily the roots selected for the
current iteration.

### Genuine eager or capacity-shaped allocation

| Component | Current behavior | Consequence |
|---|---|---|
| `PokerTensorStateStore` | Every state/history/fold column is allocated with $K$ rows when the bundle is built. | The backend permanently retains the configured worst-case state table, including while the memory controller selects fewer sequential roots. |
| `DeviceTreeBuffer` | Node tensors have $GM$ rows and edge tensors have $GM-G$ rows on every sampled iteration. | `finalize()` returns slices, so the realized tree continues to retain the complete backing allocations through target construction. |
| Branched `reach_rows` | Allocates dense fp32 `[M,P,X]`. | This is normally the largest direct `node_budget`-shaped core allocation. |
| Branched work queues | Every execution family allocates an int64 ring of length $M$. | Queue storage scales with total possible tree size instead of peak live frontier. |
| Branched control columns | `nominal_count`, `beta_rows`, two KKT path columns, and `queue_cost` each allocate $M$ rows. | Small individually, but all retain the same unnecessary global capacity. |
| `DevicePartitionSummaryCache` | Its validity map is capacity-sized. If evaluator summaries are enabled, each lazily materialized field allocates its complete capacity-shaped table. | A sparse set of street/fold anchors can cause a large dense cache allocation. |
| `ChunkedOccurrenceArena` | Payload chunks are allocated on demand, but the arena still has a hard $GM$ capacity and its default page size depends on $GM$. | Allocation is partly tight, but admission and overflow semantics remain tied to `node_budget`. |

Relevant implementation points are:

- `pbg/games/poker/bundle.py::estimate_rollout_memory` and `build_bundle`;
- `pbg/games/poker/tensor_state.py::PokerTensorStateStore`;
- `pbg/games/poker/partition_cache.py::DevicePartitionSummaryCache`;
- `pbg/core/tree.py::DeviceTreeBuffer`;
- `pbg/core/branching.py::sample_branched_tree`;
- `pbg/core/rollout.py::_sample_trees_sequential`.

### Capacity-sized estimates that are not direct allocations

The memory plans also charge several objects at $K$ rows even though their
production allocation is based on realized work:

- policy log-probability/value caches;
- RB target tables;
- completion workspaces;
- decision reaches;
- some frontier/workspace terms.

This is conservative, but it is not tight. It can reject or throttle useful
work based on memory that is never simultaneously allocated. In particular,
`estimate_rollout_memory()` charges a full all-hand policy cache for every
capacity row even though the sampler accumulates policy rows only for realized
decision nodes.

The all-phase training plan repeats the same pattern by setting
`nodes = roots_per_gpu * node_budget`. Consequently `node_budget` affects both
physical allocation and controller forecasts, but in different and sometimes
contradictory ways.

### Non-allocation uses

`node_budget` additionally controls:

- the sequential leaf logical-address depth table and per-tree length check;
- the branching scheduler's progress limit;
- streamed-value validation that a realized tree did not exceed the configured
  count;
- benchmark identity and reports;
- root-refill threshold validation;
- experiment cutoffs and fault-injection tests;
- strict config parsing and checkpoint scientific identity.

None of these requires a production allocation dial. Exact trajectory bounds,
actual realized shapes, byte limits, or explicitly test-only failure injection
can replace them.

## How loose is it?

### HU 32 bb sequential RB

The checked-in training config uses an exact 43-node path ceiling and up to
3,072 roots/GPU, so bundle construction creates backend capacity for 132,096
states. At iteration 0 the controller selected 2,560 roots and realized about
11,080 nodes per rank:

- backend state capacity utilization: about $11{,}080/132{,}096=8.4\%$;
- current tree-buffer utilization: about $11{,}080/(2{,}560\cdot43)=10.1\%$.

At the final 3,072-root iterations, the run realized roughly 18,500--18,700
nodes, or about 14% of backend/tree capacity.

For the maximum 3,072-root configuration, the rollout bound reports:

| Term | Bound |
|---|---:|
| State store | 26.8 MiB |
| Tree topology | 6.0 MiB |
| Reach-cache/frontier bounds | 93.2 MiB |
| Policy cache bound | 1,320.8 MiB |
| Partition validity map | 0.13 MiB |
| Total rollout bound | 1,447.0 MiB |

The 1.32 GiB policy-cache term is a capacity bound, not an eager table of that
size. Conversely, the state store and tree topology really are allocated at
capacity. Actual rollout-phase peaks were around 5.0--5.3 GiB because model
execution and other resident/transient tensors dominate. Therefore removing
`node_budget` will make storage and forecasts substantially tighter, but it
does not eliminate the need to control model-forward batch shapes.

The configured value 43 is itself not a tunable budget. It is derived from the
game:

$$
N_{\mathrm{player,max}} = SP+3P-S+1,
$$

plus at most three public street transitions and one final terminal or
pre-terminal row. For HU 32 bb this is $39+3+1=43$. Production already has
`max_player_nodes()` and `max_public_history_tokens()` helpers. This exact
bound should remain a game/backend invariant, not appear in YAML or size
storage eagerly.

### HU 32 bb branching validation

The beta-2.3 validation used `node_budget: 12000`. Ignoring optional evaluator
summary fields, direct capacity-shaped core allocations were approximately:

| Term | Capacity-shaped allocation |
|---|---:|
| Poker state store | 2.44 MiB |
| Tree node/edge columns | 0.55 MiB |
| Dense branching reach table `[12000,2,1326]` | 121.40 MiB |
| Two HU family queues | 0.18 MiB |
| Five branching control columns | 0.23 MiB |
| Total | about 124.8 MiB |

The occurrence arena was chunked and therefore is not included as a full eager
allocation. The rollout memory estimator reports a larger 489 MiB bound
because it additionally charges capacity-sized policy and frontier terms that
are not all eagerly materialized. Enabling evaluator-summary cache fields would
raise that bound to about 588 MiB by adding another capacity-shaped term.

Capacity-to-realized-node ratios for observed cases were:

| Realized case | Nodes before/final | Capacity ratio |
|---|---:|---:|
| Tiny tree before refill | 3 | 4,000x |
| Checkpoint 37 after refill | 2,992 | 4.0x |
| Checkpoint 555 after refill | 4,840 | 2.5x |
| Checkpoint 1,110 after refill | 5,435 | 2.2x |
| Ordinary large seed | 4,279 | 2.8x |

The initial 64-row checkpoint smoke OOM occurred during policy-forward work and
disappeared after reducing the family batch to 16 rows; the latter completed at
about 1.1--1.2 GiB peak allocated. This is another reason to keep storage
growth and execution-batch admission as separate controls.

## Other allocation looseness not caused by `node_budget`

Eliminating `node_budget` should not be mistaken for complete tightness.

1. `PolicyCache.old_logp_child` and `NodeTargets` use a dense maximum realized
   child width. One high-degree node, including a refilled root, pads every
   decision row to that width. These should eventually become ragged edge
   columns indexed by a decision pointer.
2. CUDA allocator bins and model workspaces create phase peaks much larger than
   stored topology in current HU training.
3. Lists of per-family/per-wave tensors are concatenated only at finalization.
   They are actual-sized, but may produce fragmentation and a transient source
   plus destination overlap.
4. The rollout memory estimate uses a loose history-step expression in one
   place even when the backend/model has the tighter game-derived history
   bound.

These are independent follow-up targets. They should not be encoded back into
a replacement node count.

## Replacement architecture

### 1. Remove the production node-capacity abstraction

Delete `RolloutConfig.node_budget`. No replacement global node-count setting
should be introduced.

Permitted remaining concepts are:

- a game-derived maximum path length, used only to prove trajectory progress;
- a configurable storage page size, used only for execution efficiency;
- a byte ceiling and byte reserve;
- realized node/edge/decision/leaf counts as observations;
- test-only injected byte limits or page-allocation failures.

`arena_chunk_nodes` should be renamed to something such as
`storage_page_rows`. Page size is an execution choice, not a total capacity or
scientific parameter.

### 2. Introduce a byte-admitted paged SoA store

Tree, state, reach, and branching-control columns should use a common append-only
paged structure-of-arrays store:

- pages are allocated only when an append crosses the current high-water mark;
- a `ByteAccountant` reserves the complete page transaction before CUDA
  allocation;
- an allocation failure does not advance any cursor;
- handles remain stable integer `(page, offset)` encodings or monotone logical
  row ids;
- finalization exposes used rows without retaining a configured-total backing
  allocation; slack is bounded by at most the final page;
- pages can be reused across iterations, but retention is visible to the
  memory ledger and releasable at a phase boundary.

The current `ChunkedOccurrenceArena` already demonstrates transactional page
allocation, but it must lose its fixed total capacity. Its CUDA gather/scatter
implementation also needs a page-count-independent access plan before it can
serve large state/reach fields efficiently.

### 3. Replace `DeviceTreeBuffer`

Use paged node and edge columns with separate append transactions. Child
expansion reserves exactly the proposed node rows and edge rows. CSR
finalization should concatenate or seal only used pages.

This removes:

- full-capacity node and edge tensors;
- `require_child_capacity()` and capacity overflow;
- the backing-storage retention caused by returning views from `finalize()`.

The invariant becomes “every append was byte-admitted,” not “every append was
below an arbitrary node count.”

### 4. Replace the poker state store and partition cache

`PokerTensorStateStore` is the widest refactor because poker code directly
indexes its tensor attributes in many places. It should become a paged state
API supporting grouped gather, scatter, append-copy, and history traversal.
There are roughly 200 direct tensor-indexing sites in the active poker code, so
this should be treated as a deliberate storage-interface migration rather than
hidden behind repeated concatenation.

The partition cache should store only materialized anchor rows. Since state
handles are monotone and an anchor is computed at most once per rollout, a
compact key-to-cache-row map plus paged value columns is sufficient. It should
not allocate every expensive summary field for every possible state handle.

### 5. Make branching reaches and controls paged

Replace these full-length tensors:

- `reach_rows`;
- `nominal_count`;
- `beta_rows`;
- `log_player_sum`;
- `log_ancestor_product`;
- `queue_cost`.

They share tree node ids and can be columns in the node arena. This also avoids
maintaining parallel capacities and makes one node append transaction account
for its complete resident row.

### 6. Size work queues by live frontier

An execution queue contains live handles, not all nodes ever created. Use a
deque of device tensor pages or a growable segmented ring. Recycle empty pages
as the head advances. Its byte reservation should track peak live queue depth.

Sequential queues need at most one live occurrence per admitted trajectory;
that bound comes from the already selected root width, not from path length.

### 7. Use exact and ragged target storage

Remove `configured_node_budget` from `streamed_branched_value()`. Topology
validation and the byte workspace plan already know the realized tree.

Change action-conditioned policy/target tables from `[decisions,max_degree,X]`
to flat edge rows plus a decision pointer. This is not required merely to
delete `node_budget`, but it is required for genuinely tight allocation when a
root refill creates a larger maximum degree than the rest of the tree.

## Safety without a node limit

Removing a node cap requires a stronger safety invariant, not unbounded blind
growth.

### Sequential modes

Sequential execution has a finite game-derived path bound. It should:

1. allocate state/tree pages as rows are realized;
2. keep only root-count-sized working reaches and queues;
3. assert that history length increases and never exceeds
   `max_public_history_tokens()`;
4. let the existing four-phase memory controller choose roots before the
   iteration;
5. stop before draws if the minimum roots plus one storage page cannot fit.

The sequential logical completion key can derive its path from realized depth
or from the exact backend path bound. It does not need a YAML node count.

### Branched mode

Branching has bounded depth but potentially enormous width. Before admitting
offspring, the sampler should reserve:

1. bytes for the proposed child node/edge/state/reach/control rows;
2. bytes for the peak policy-forward batch;
3. a conservative mandatory-continuation reserve for every live child to
   finish with multiplicity one;
4. downstream target/seal bytes predicted for the newly retained decisions
   and edges.

Only multiplicity above one is optional. If the extra children do not fit, the
sampler executes one child and retains nominal-versus-executed counts for the
existing $C/T$ corrections. The reserve for mandatory continuation prevents a
later state where even beta one cannot finish the already-admitted frontier.

The hard reserve should use the exact remaining public path bound. The learned
$C$ head remains useful as a less conservative soft forecast for deciding when
extra branching is worthwhile, but correctness must not depend on its
calibration.

Root refill follows the same rule. Launching another root subtree is allowed
only when its child plus mandatory continuation is byte-admitted. If the
configured lower-size safeguard cannot be supported, training should stop
cleanly before the tree draw rather than relying on a node-capacity exception
or outcome-dependent reroll.

## Branching memory controller

Paged allocation solves storage proportionality. It must be paired with a
closed-loop controller for model and later-phase memory.

### Common byte controller

For every iteration and rank:

1. compute the current operating ceiling from `cudaMemGetInfo`, current
   PyTorch reservation, the hardware reserve, and the configured memory
   fraction;
2. predict rollout, target, seal, and optimization peaks from realized/paged
   storage terms plus calibrated transient envelopes;
3. use the worst rank and limiting phase;
4. contract immediately on unsafe evidence and grow only after a delayed safe
   streak;
5. checkpoint the controller and every effective sampling parameter.

The existing `_memory_operating_ceiling()` should be wired into this process;
it is currently unused.

### Dynamically size family forwards

Policy family batch size changes execution only, not the logical sampling law.
Before each forward, select the largest row count whose calibrated transient
workspace fits current headroom. This is the first response to an inference
workspace threat and should apply in every mode.

### Fixed-beta mode

Keep the configured beta table as a base and control only its excess above one:

$$
\beta^{\mathrm{effective}}_{k,b}
=1+s_k\left(\beta^{\mathrm{base}}_b-1\right).
$$

The checkpointed scale $s_k$ contracts immediately under memory pressure and
grows slowly within configured bounds after sustained safety. This preserves
the relative bucket schedule and leaves beta-one buckets unchanged. Runs using
this behavior should be named memory-governed fixed beta rather than silently
reported as strictly fixed beta.

### Dynamic-beta mode

Keep the scientific KKT controller, but use a memory-safe target

$$
M_k^{\mathrm{effective}}
=\min\left(M_{\mathrm{configured}},M_{\mathrm{memory},k}\right).
$$

Unsafe memory evidence should cause an immediate, worst-rank, no-EMA downward
correction of $\kappa$, followed by a freeze on upward movement. The ordinary
damped update resumes after a safe streak. A separate memory scale may clamp
the KKT proposal,

$$
\beta^{\mathrm{executed}}
=1+s_{\mathrm{memory}}\left(\beta^{\mathrm{KKT}}-1\right),
$$

so safety does not depend on head quality, warmup completion, or the one-tree
lag in the $\kappa$ update. Repeated clamping should feed the fast $\kappa$
correction so that the statistical controller learns to request feasible work.

## Failure semantics

The migration must preserve the existing no-reroll principle.

- A page, forward batch, or downstream phase that cannot be admitted fails
  before the corresponding random draw or mutation.
- If the minimum safe workload cannot fit, write a memory-stop checkpoint and
  terminate cleanly.
- Do not catch a CUDA OOM, lower beta, restore RNG, and sample the iteration
  again. That makes inclusion depend on which sampled trees exhausted memory.
- An unexpected actual CUDA OOM remains a fatal defect in admission and should
  resume only from the last intact checkpoint.

## Configuration and checkpoint migration

`node_budget` currently participates in strict parsing, benchmark identity,
and checkpoint scientific identity. Retirement therefore needs an explicit
schema transition.

1. Remove `node_budget` from new production YAML and `RolloutConfig`.
2. Add exact game-derived `max_public_nodes_per_trajectory` to the backend API,
   not to configuration.
3. Replace node-count experiment cutoffs with byte ceilings. Keep a test-only
   node fault injector only where deterministic overflow tests need it.
4. Remove `configured_node_budget` from value reduction and benchmark workload
   keys.
5. Migrate old checkpoint configs by deleting `rollout.node_budget` during
   explicit legacy normalization. Old model/optimizer/RNG state remains valid;
   the value never entered model tensors.
6. Treat page size, memory reserve, and controller hysteresis as execution
   controls rather than scientific estimator identity.
7. Update the branching-variance harness to report actual nodes, allocated
   pages, admitted bytes, mandatory-continuation reserve, and headroom instead
   of overflow margin relative to a node count.

## Suggested implementation sequence

1. **Baseline instrumentation.** Add per-component allocated/used/page bytes
   and retain common-seed output hashes for all three modes.
2. **Remove semantic uses.** Replace sequential depth/progress checks with the
   exact game bound; remove value-reducer validation and benchmark identity
   dependence.
3. **Page tree and branching columns.** Migrate `DeviceTreeBuffer`, branching
   reaches, KKT columns, and queues to byte-admitted pages.
4. **Page poker states and caches.** Introduce the state accessor API, migrate
   direct indexing, and make partition summaries sparse.
5. **Make targets ragged.** Remove maximum-degree padding from policy and
   target storage.
6. **Wire the common branching memory controller.** Add dynamic family batch
   sizing, fixed-beta scaling, fast dynamic-$\kappa$ contraction, and
   mandatory-continuation admission.
7. **Delete the config field and legacy paths.** Migrate configs/checkpoints,
   remove tests that express production capacity in nodes, and make the source
   inventory clean.

The state-store migration is the largest step because the active poker backend
has many direct tensor-indexing sites. It should not be postponed by retaining
`node_budget` elsewhere: tree/reach/control paging can land first behind a
temporary state-store adapter, followed by complete removal once state paging
is ready.

## Acceptance criteria

The retirement is complete only when:

- `rg "node_budget" pbg configs/train` returns no production uses;
- bundle construction does not receive or derive a total rollout-node
  capacity;
- an empty/tiny branched tree allocates only initial pages plus model/backend
  fixed residents;
- storage bytes follow realized rows within at most one page per live column;
- sequential output remains common-seed identical across root widths and page
  sizes;
- branched output remains common-seed identical whenever memory admission does
  not suppress multiplicity;
- suppressed offspring retain nominal/executed accounting and pass $C/T$
  estimator tests;
- root refill preserves its protected cohort under page-boundary and
  memory-pressure cases;
- no admitted workload can reach an arena/state/queue capacity exception;
- all-rank memory pressure produces one common pre-draw controller decision;
- GPU fault-injection tests prove clean transactional stops without reroll;
- the full checkpoint resume tests cover the schema migration and controller
  state.
