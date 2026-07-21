# Suggested reading order for `pbg/core`

This guide describes the sole production pipeline after the 2026-07-21
storage-lifecycle restoration. Earlier versions of this guide named the
deleted capacity-shaped tree, reach-cache, padded-target, and minibatch
implementations; those names are historical and are not alternate paths.

The runtime data plane is:

```text
byte-admitted rollout sources
  -> exact immutable RolloutStorage
  -> bounded streamed target construction
  -> immutable TrainingStorage
  -> bounded physical learner subspans
```

Normal, sequential Rao--Blackwell (RB), and branched RB share these ownership
and learner boundaries. They differ only in rollout topology and mathematical
targets.

## Recommended order

### 1. `pbg/core/training.py`: phase orchestration

Start with [`training.py`](../../pbg/core/training.py):

1. `_analytical_memory_plan` derives rollout, seal, target, and optimization
   priors from the same schemas that allocate storage.
2. `_build_iteration_data` dispatches the three rollout modes and makes their
   lifecycle boundaries explicit.
3. `_seal_sequential_rollout_storage` and
   `_seal_branched_rollout_storage` create exact rollout destinations and
   release mutable sources.
4. `_one_iteration` calls the matching target builder, finalizes immutable
   training storage, and passes only that store to the learner.
5. `run_training` owns distributed setup, model/reference/optimizer lifetime,
   memory control, checkpointing, and the iteration loop.

`IterationData` is a phase-local handoff, not a compatibility representation.
The optimization phase sees `TrainingStorage`; it cannot address rollout
states, reaches, topology, leaves, or queues.

### 2. `pbg/core/storage.py`: declarations and owners

Read [`storage.py`](../../pbg/core/storage.py) next:

- `TensorFieldSchema` and `TensorSchema` define dtype, natural/exact width,
  alignment, payload bytes, and allocation.
- `PagedTensorArena`, `PagedRowAccumulator`, and
  `RecyclableTensorPool` provide transactional page growth, stable logical
  ids, exact sealing, final-page slack reporting, and recyclable live slots.
- `RolloutStorage` owns exact topology, natural inputs, retained old outputs,
  ragged actions, and leaves until target construction finishes.
- `TrainingStorage` owns only immutable natural inputs, exact targets, ragged
  action targets, and sampling metadata. Its constructor validates every
  column against the normative mode schema.
- The `rollout_*_schema`, `target_*_schema`, and
  `sequential_posterior_schema` functions are the single declarations used by
  allocation, validation, sealing, and memory planning.

Stable ids are logical integers. No view into a growable allocation is used as
identity.

### 3. `pbg/core/execution.py`: admission, queues, and logical RNG

Read [`execution.py`](../../pbg/core/execution.py) for:

- `ByteAccountant`, whose labeled reservations precede allocation and whose
  grouped transactions fail without partial publication;
- `ChunkedOccurrenceArena` and `DeviceWorkQueue`, which bound live scheduling
  work independently of logical statistical work;
- `TrajectoryAdmissionPlan`, which freezes sequential root work before draws;
- logical RNG keys and draw purposes, which make queue/page/batch changes
  sampling-invariant.

Scientific child selection and physical byte admission are separate. An
already selected branched child wave is admitted completely before posterior
or event draws; a failed admission does not suppress, retry, or reroll it.

### 4. `pbg/core/rollout.py`: normal and sequential RB

Trace `_sample_trees_sequential` in
[`rollout.py`](../../pbg/core/rollout.py). Each live trajectory owns one
overwritten state/reach slot. At a player decision the scheduler captures the
complete immutable natural-width poker input and retained old-policy row. At a
stopped leaf it captures the sole completion reach/input row. Chance nodes do
not survive the sequential seal.

The resulting topology is decision-only CSR:

- `trajectory_ptr[G+1]`;
- `decision_partition[D]` and `decision_row[D]`;
- `leaf_row[G]`.

Changing family batch rows or start-admission width may change physical
packing, but not trajectory addresses or sampled objects.

### 5. `pbg/core/branching.py`: fixed and dynamic branched RB

Read [`branching.py`](../../pbg/core/branching.py) as the alternative rollout
geometry. It maintains one labeled tree rooted at the distinguished root,
recyclable live reaches/states, paged control columns, bounded family queues,
and ragged action rows.

`admit_child_wave` reserves every page needed by the complete already-selected
wave before posterior/action/chance draws or topology mutation. Fixed and
dynamic modes retain duplicate child identity, nominal versus executed count,
KKT intensity, queue-safeguard correction, public $T/C$ predictions, and the
protected ordinary-root cohort under root refill.

### 6. `pbg/core/tree.py` and `pbg/core/topology.py`: source and seal

[`tree.py`](../../pbg/core/tree.py) contains `PagedTreeSource`, the paged
rollout-time topology source, plus the small sealed handoff records.
[`topology.py`](../../pbg/core/topology.py) contains the exact immutable
`SequentialDecisionCSR` and `BranchedTreeSpan` validators. Validation is
vectorized on CUDA and checks row domains, roots, contiguous child spans,
chronology, and player-clock structure without per-node host reads.

The branched sealed topology has six columns: parent, child begin, degree,
decision partition, decision row, and leaf row. Protected-root metadata is
stored separately and never widens the ragged action domain.

### 7. `pbg/core/targets.py`: bounded target construction

[`targets.py`](../../pbg/core/targets.py) is the only target-builder family:

- `build_normal_targets` performs the reverse segmented player-clock
  recurrence and retains one scalar return per decision;
- `build_sequential_rb_targets` streams evaluator completion batches,
  deduplicates exact posterior rows, and retains exact returns plus posterior
  references;
- `plan_branched_frontier` derives the maximum live contraction slots from the
  sealed topology;
- `build_branched_rb_targets` contracts the reusable frontier into exact value,
  ragged posterior/advantage, $T/C$, nominal/executed correction, and protected
  root targets.

No production $Q$ target exists. Finalization releases topology, leaves,
completion results, and frontier scratch before optimization.

### 8. `pbg/core/learner.py`: immutable learner boundary

Finish with [`learner.py`](../../pbg/core/learner.py):

- global stratum-mass normalization and fixed statistical-count allocation;
- index-only statistical draws;
- `PhysicalPlanner` and bounded gather/forward/backward/release subspans;
- fused policy/entropy/reference-KL terms at natural model width;
- optional transactionally admitted reference caches with an uncached
  singleton fallback;
- one flat fp32 gradient SUM buffer and finite-gated optimizer step.

The learner never reconstructs public state or reach paths. Current, old, and
reference outputs stay on each partition's natural width; exact RB posterior,
return, and advantage tensors stay on the exact hand domain.

## Poker-specific companion reading

After the generic path, read:

- [`pbg/games/poker/tensor_state.py`](../../pbg/games/poker/tensor_state.py)
  for recyclable public-state slots and refcounted persistent history;
- [`pbg/games/poker/partition_cache.py`](../../pbg/games/poker/partition_cache.py)
  for sparse folded-summary anchor pages;
- [`pbg/games/poker/partitions.py`](../../pbg/games/poker/partitions.py) for the
  immutable $H\to K$ preflop quotient and partition schemas;
- [`pbg/games/poker/rollout_backend.py`](../../pbg/games/poker/rollout_backend.py)
  for last-consumer input capture and the read-only sibling evaluator boundary.

For $R=13$, preflop natural rows have width $K=169$ and postflop/exact rows
have width $H=1326$. There is one postflop partition family, not one partition
per street.

## Audit checklist

At each phase boundary ask:

1. Which schema declares every live field and its bytes?
2. Which object is its sole owner and its next/last consumer?
3. Was page or destination growth admitted before allocation?
4. Are page/batch/queue choices absent from logical RNG addresses?
5. Did the prior phase's mutable owners actually release?
6. Are natural-width model data and exact-width mathematical data separated?
7. Does optimization retain anything beyond `TrainingStorage`, one bounded
   physical subspan, and an optional admitted reference cache?

The mathematical and lifecycle authorities remain
[`core_math.tex`](../theory/core_math.tex) and the documents in
[`docs/theory/pseudocode/`](../theory/pseudocode/).
