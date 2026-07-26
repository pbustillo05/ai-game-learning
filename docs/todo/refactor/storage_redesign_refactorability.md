# Storage redesign implementation and repository cleanup

Status: implementation plan derived from
[`storage_redesign.md`](storage_redesign.md); it does not redefine that contract.

Related audit: `docs/audit/pbg_redundancy_compactification_audit_20260723.md`.

## 1. Decision

Implement the storage redesign and delete the superseded data path as one project.

The replacement lives under `pbg/core/queued/` until cutover. New and old rollout, storage, topology, target, and
learner representations never cross. The replacement becomes the sole implementation only after all production and
evaluation consumers migrate; then delete the old dependency closure, temporary selection surface, obsolete tests,
configuration, tools, and documentation.

A parallel runtime selector requires an explicit, time-bounded migration exception. Without that approval, comparison
uses test or benchmark harnesses that invoke each implementation independently; production exposes no second engine.

Temporary code growth is allowed only when every addition has a named deletion credit and the projected final tree
stays below budget.

## 2. Cleanup invariant

Every file has a declared role. Every object under `pbg/` is reachable from an accepted production/public root or a
small reviewed dynamic-boundary allowlist.

The final tree has:

- no production module retained only by tests;
- no compatibility alias, fallback path, stale configuration, or runtime implementation selector;
- no duplicate authority for storage, topology, completion binding, targets, learner assembly, or memory admission;
- no wrapper that owns neither a lifetime nor an invariant;
- no stale training, distributed, best-response, probe, inference, variance, or benchmark consumer;
- no undocumented dynamic reachability exception.

Useful test-only or developer-only code belongs under `tests/` or `tools/`, not `pbg/`.

## 3. Baseline and budget

The 2026-07-23 compactification audit counted 24,126 non-comment, nonblank Python lines under `pbg/`; the largest
affected modules were:

| Module | Counted lines |
|---|---:|
| `pbg/core/training.py` | 3,351 |
| `pbg/games/poker/rollout_backend.py` | 2,718 |
| `pbg/core/storage.py` | 1,656 |
| `pbg/core/learner.py` | 1,412 |
| `pbg/core/branching.py` | 1,239 |
| `pbg/core/rollout.py` | 976 |
| `pbg/core/targets.py` | 948 |
| `pbg/core/execution.py` | 917 |

Refresh the baseline before implementation with the audit's token rule. Track:

```text
baseline_total
current_total
temporary_replacement_lines
shared_lines
committed_deletion
projected_final_total
```

$$
\mathtt{projectedFinal}
=\mathtt{baselineTotal}
-\mathtt{committedDeletion}
+\mathtt{temporaryReplacementLines}
+\mathtt{sharedLines}.
$$

The final count must not exceed the audit's 20,750-line ceiling and should be lower. Line count is a guardrail: do not
move production code out of `pbg/`, obscure it, or weaken validation to meet it.

## 4. Migration ledger

Create `docs/audit/storage_redesign_migration_ledger.md` when implementation begins. Assign every relevant symbol or
method chain exactly one disposition:

| Disposition | Meaning | Completion condition |
|---|---|---|
| reuse | representation-independent | one shared definition with accepted callers |
| extract | useful behavior trapped in legacy code | move once and delete the old definition |
| rewrite | required behavior coupled to old ownership | validate the new owner, then delete the chain |
| delete | stale, duplicate, or forbidden | remove code and all references |
| externalize | valid only for tests or tools | move outside `pbg/` |

Each row records:

```text
old symbol or chain
accepted caller
authoritative contract
new owner
disposition
validation evidence
deletion change
status
```

There is no “keep for safety” disposition. Import the existing audit findings instead of repeating its investigation.
Seed the ledger with its duplicate policy/sampling paths, mutable-tree completion adapters, duplicate occurrence and
seal owners, repeated model preparation and evaluator binding, and repeated memory, queue, validation, and
configuration machinery.

## 5. Import firewall

Before implementing the replacement, add an AST boundary test. Code under `pbg/core/queued/` must not import legacy
rollout, storage, topology, target, learner, or training representations.

If a representation-independent function is trapped in a legacy module:

1. extract it into a small shared module;
2. migrate every accepted caller;
3. record the move in the ledger;
4. delete the old definition at cutover.

Do not create a compatibility DTO between implementations.

## 6. Target responsibilities

Exact schemas, indexing, lifetimes, and algorithms come only from
[`storage_redesign.md`](storage_redesign.md). The implementation should converge toward:

```text
pbg/core/
  adaptive_branching.py     fixed/KKT/safeguard mathematics
  queueing.py               semantic FIFOs and stable-prefix selection
  logical_rng.py            logical addresses and counter draws
  distributed_failure.py    rank-common failure agreement
  optimization.py           gradient synchronization and optimizer step
  observability.py          metrics

  queued/
    contracts.py            four-axis schemas and conversion DAG
    slabs.py                fixed SoA allocation, reservation, publication
    stores.py               owners and last-consumer validation
    rollout.py              family scheduling and model -> sample -> commit
    topology.py             CPU Topology -> LayeredTopology planning
    completion.py           Leaf -> evaluator -> LayerWork binding
    targets.py              layered CUDA target graph
    learner.py              aligned and ragged minibatch assembly
    memory.py               phase ledger, admission, failure records
    engine.py               iteration composition
```

Files with inseparable lifetimes may merge. The dependency direction is:

```text
contracts
  -> slabs/stores
  -> rollout/completion/topology
  -> targets
  -> learner
  -> engine
```

Poker-specific model preparation, transition materialization, `sample`, folded statistics, and evaluator adaptation
remain under `pbg/games/poker/`. Core sees generated schemas and opaque hooks, never streets or poker field names.

## 7. Canonical authorities

| Operation | Sole authority |
|---|---|
| mode fields, partitions, widths | immutable four-axis schema |
| queue choice and stable prefix | semantic queue scheduler |
| RNG address and draw | logical RNG |
| bytes, alignment, capacity | field schema and phase ledger |
| reservation and publication | monotone slab transaction |
| model preparation | one `model_input` boundary |
| draws and topology labels | `sample` |
| routing, materialization, retained writes | `commit` |
| showdown and completion | evaluator facade consuming exact `Leaf` |
| rollout structure | CPU `Topology` |
| propagation structure | GPU `LayeredTopology` |
| targets | one layered CUDA graph |
| learner batches | one sequential/branched sampler |
| align, compact, release | `align-and-seal` |

Use classes only for mutable resource lifetimes, invariant-bearing schemas, controller state, or external protocols.
Prefer functions and tensor collections for transformations. Do not create mode class hierarchies or generic field
lookup layers.

## 8. Required data flow

The implementation order is:

```text
semantic family FIFO
  -> model(model_input)
  -> sample(labels, transient descriptors, async topology packet)
  -> commit(saved heads, FamilyWork/Leaf destinations, endpoint patches)

Leaf FIFO
  -> poker-gpu-evaluator
  -> LayerWork in Leaf order

CPU Topology
  -> LayeredTopology and movement plan
  -> align-and-seal
  -> layered CUDA target graph
  -> cleanup and atomic learner publication
  -> minibatches
  -> optimizer
```

Sequential and branched modes share this path. Their generated schemas, sampling laws, row multiplicities, and learner
sampling differ; they do not get separate storage or target engines.

Physical fusion may combine `sample`, descriptor scans, materialization, and `commit`. Fixed-address model packs and
bounded chance closure are additional CUDA-graph capture units; target propagation is one strict layered CUDA graph.
Fusion and graph capture never change logical ownership.

## 9. Storage primitives

Implement only:

```text
SoASchema          immutable fields, dtype, shape, alignment, lifetime
MonotoneSlab       admitted capacity and reserve/fill/publish tails
ExactBlock         sealed realized extent
```

They provide:

- exact aligned byte calculation;
- fixed allocation before admitted work;
- atomic multi-destination `reserve -> fill -> publish`;
- stable addresses and logical order;
- exact stable compaction and reference remapping;
- minimum-logical-address failure;
- explicit release after the last consumer.

Do not add convenience stores beyond the owners in `storage_redesign.md`:
`FamilyWork`, `SigmaHead`, `ActionHead`, `Leaf`, `Topology`, `TopologyChild`, `FoldedRanges`,
`QueueBehaviorMetadata`, two `LayerWork` arenas, `LayeredTopology`, `SigmaTarget`, `ActionTarget`, `LearnerIndex`, and
`DeviceFailure`.

## 10. Memory authority

One coordinator owns:

1. exact bytes for resident schema owners;
2. calibrated bounds for kernels, graphs, and external libraries;
3. admission forecasts for realized statistical work.

Its phases are:

```text
rollout + overlapping completion
align-and-seal
target graph
optimization
```

Admission covers source and destination learner arrays, both `leaf_cap` LayerWork arenas, compaction/permutation
scratch, `LayeredTopology`, graph/library workspace, and a safety guard. Logical sample counts are never reduced to fit
a physical pack; physical work streams through bounded workspaces.

## 11. Implementation phases

### Phase 1 — authority and inventory

- approve required theory, root-convention, and evaluator-ownership changes;
- refresh reachability, line counts, and production roots;
- create the migration ledger and deletion credits;
- install the import firewall and capability matrix.

### Phase 2 — contracts and transactions

- generate every four-axis schema;
- implement exact byte accounting, slabs, transactions, failures, and last-consumer checks;
- test absent fields, dimensions, alignment, atomic publication, and overflow on CUDA.

No model or game integration enters this phase.

### Phase 3 — topology and targets before the easy vertical slice

Use tensor fixtures to test:

- one distinguished root per rollout tree, independent root batching, $\beta=1$ sequential nodes, and ordinary
  branched-root degrees;
- compact `Topology = {trained player nodes} union {N_tilde >= 2}`;
- omitted unary paths and `path_length`;
- duplicate-child identity;
- headless frozen structural reductions;
- CPU layering, carry rows, and output conversion;
- two `leaf_cap` ping-pong arenas;
- `SigmaTarget` and `ActionTarget` from one layered CUDA graph.

This prevents sequential geometry from defining the abstractions.

### Phase 4 — sequential vertical slice

Implement normal sequential poker through:

```text
model -> sample -> commit
Leaf -> LayerWork
Topology -> LayeredTopology
align-and-seal
target graph
randperm learner
optimizer
```

Validate model inputs, labels, transitions, targets, losses, gradients, checkpoint/resume, memory, and throughput.

### Phase 5 — branched RB

Add:

- [`poker_branched_rb_sample.md`](poker_branched_rb_sample.md);
- `ActionHead` spans and weighted learner indices;
- `FoldedRanges` and folded-stat refresh;
- exact evaluator-v2 `Leaf` inputs and completion outputs;
- post-to-pre `LayerWork` reuse;
- fixed/KKT controllers, refill, retry, and failure behavior;
- alias-table sigma sampling and ragged action gathers.

The repository owns rejection sampling. The evaluator owns only `Leaf -> LayerWork` showdown/completion operations.

### Phase 6 — complete axes and consumers

Implement sequential RB, sampled branched, best response, and every declared fixed/KKT combination. Migrate:

- training and distributed/replica training;
- best-response and inference workflows;
- variance runners and probes;
- checkpoints and resume;
- benchmarks and operational tools.

Unsupported combinations fail at schema construction.

### Phase 7 — cutover and deletion

After the complete acceptance matrix passes:

1. make the replacement the sole path;
2. delete the old rollout, storage, branching, topology, target, learner, seal, and memory closure;
3. delete temporary selectors, adapters, configurations, tests, tools, and documentation;
4. move valid test/tool-only remnants out of `pbg/`;
5. run reachability, duplication, configuration, and line-count audits;
6. consolidate redundancy inside the replacement.

Do not leave aliases or deprecated configuration for rollback.

## 12. Validation

Use matched logical seeds where laws agree; compare distributions where the approved law changed.

Validate:

- family FIFO order and pack-subdivision invariance;
- `model -> sample -> commit` transaction atomicity;
- inherited-child plus $N-1$ branched-RB profile sampling, profile propagation, and topology packets;
- exact evaluator `Leaf` rows and `LayerWork` completion epilogue;
- CPU topology flattening, carries, permutations, and frozen-row deletion;
- CUDA-graph targets against `BranchedValue` and sequential references;
- sequential five-array alignment and `randperm` gathering;
- branched weighted sigma sampling and separate ragged action gathers;
- analytical, admitted, allocated, reserved, and device-observed memory;
- overflow, exhaustion, NaN, evaluator, and rank-common failure injection;
- checkpoint/resume and distributed equivalence;
- target and learner throughput on supported CUDA hardware.

Every validation artifact names the ledger entries it closes.

## 13. Cutover gates

Delete the legacy path only when:

- every supported four-axis schema creates only applicable fields;
- poker has only `pre` and `post_A` physical queues;
- all modes use `model -> sample -> commit`;
- `Leaf` and `LayerWork` exactly match evaluator input and output contracts;
- topology contains exactly trained player nodes and nodes with `N_tilde >= 2`;
- all targets use the layered CUDA graph;
- sequential learner rows are five-array aligned;
- branched learner sampling uses decision weights and exact action spans;
- no maximum-action padding or inactive action exists;
- all accepted consumers use the replacement;
- phase ledgers reconcile;
- failure injection and deletion rehearsal pass;
- the migration ledger has no open row;
- final production code is net-negative and below the refreshed 20,750-line ceiling;
- no compatibility surface or unexplained production candidate remains.

## 14. End state

```text
immutable four-axis contract
-> semantic family FIFOs
-> model -> sample -> commit
-> Leaf -> LayerWork
-> CPU Topology -> GPU LayeredTopology
-> align-and-seal
-> layered target CUDA graph
-> aligned/ragged learner
-> optimizer
```

The final repository has one owner per lifetime, one implementation per semantic operation, and no superseded path.
