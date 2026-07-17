# Suggested reading order for `pbg/core`

This is a runtime-oriented reading guide, not a file-by-file inventory. It is
optimized for understanding the decisions that determine GPU memory use,
retained state, batch shape, queue depth, and synchronization. The active
pipeline is one implementation shared by normal, sequential Rao--Blackwell
(RB), and branched RB training; Kuhn/Leduc/synthetic code is archived and is
not part of this guide.

## The big picture first

Read the code as this phase pipeline:

```text
configuration
    -> rollout storage + admission plan
    -> device-local family queues
    -> policy evaluation and event expansion
    -> retained topology / reaches / old-policy outputs
    -> completion and streamed backward reduction (RB)
       or compact scalar dataset construction (normal)
    -> stratified minibatches
    -> gradient accumulation, distributed reduction, optimizer step
```

The important lifetime boundary is between rollout and optimization. Rollout
retains enough information to make the estimator exact, then target creation
and completion are arranged so large temporary workspaces do not overlap with
the rollout snapshot unnecessarily.

## Recommended order

### 1. `pbg/core/training.py`: read the orchestration skeleton

Start with [`training.py`](../../pbg/core/training.py), in this order:

1. `SequentialMemoryController` and `_rank_local_config` — how a phase-wise
   memory proof selects one common root width per GPU before any draws.
2. `TrainingStorage` and `_IterationData` — the deliberate retention set crossing the rollout,
   target, and optimization phases.
3. `_analytical_memory_plan`, `_memory_limit`,
   `_validate_memory_preflight`, and `_profiled_call` — the clearest statement
   of what the implementation believes is resident at each phase, including
   parameters, topology, reaches, policy cache, targets, completion workspace,
   optimizer state, gradient synchronization, and reference-model copies.
4. `_build_iteration_data` — the actual phase transition: synchronous read-only
   old-policy rollout, completion microbatches, streamed value reduction, and target/cache
   retention.
5. `_one_iteration` — minibatch reuse, gradient accumulation, distributed
   objective scaling, finite-gradient gating, and optimizer stepping.
6. `run_training` — device setup, model/reference copies, optimizer creation,
   preflight, checkpoint boundaries, and the outer iteration loop.

Do not begin by reading every helper in this file. The portions above explain
the ownership and lifetimes; the logging, checkpoint, metadata, and metric
formatting helpers can be consulted later.

Efficiency questions to answer here:

- Which objects survive from rollout into optimization? In particular, inspect
  `PolicyCache`, `DecisionReachCache`, targets, and the sampled tree.
- Which objects are intentionally deleted before completion/backward workspace
  is allocated?
- Is a phase bounded analytically, calibrated on the first CUDA iteration, or
  merely observed?
- Is each minibatch sampled with replacement, or are rows permuted and reused
  as epochs? The answer differs between normal and RB paths.

### 2. `pbg/core/execution.py`: understand the allocation and queue primitives

Read [`execution.py`](../../pbg/core/execution.py) next. This is the highest
value module for the user's stated interests:

- `ByteAccountant` — labeled reservations made before allocation, with a hard
  limit and high-water accounting.
- `TrajectoryAdmissionPlan` — retained bytes plus in-flight bytes plus a
  per-live-trajectory resident cost. `safe_admit` delays new starts rather
  than changing their logical work or random addresses.
- `ArenaField` and `ChunkedOccurrenceArena` — append-only struct-of-arrays
  storage, optional fields, transactional chunk allocation, integer handles,
  and batched gather/scatter without converting handles to host scalars.
- `DeviceWorkQueue` — bounded insertion-order queue with device storage and a
  host mirror of count/age used only at scheduler boundaries. Read
  `enqueue`, `peek`, `dequeue`, and `mark_done` together with arena status
  transitions (`NEW -> QUEUED -> EVALUATING -> DONE`).
- `stable_partition_masked` — fixed-shape routing by family. Pay attention to
  the device-resident selected counts and the avoidance of dynamic-shape
  synchronization.
- `ReusableStagingSlabs` — reusable bounded staging storage for model-facing
  batches.
- `logical_rng_keys`, `child_path_keys`, and the draw-purpose enum — schedule
  changes must not change stochastic outcomes.

The central design choice is separation of logical work from physical packing:
queue selection and batch boundaries affect throughput, but trajectory IDs and
path-addressed RNG determine the estimator's draws.

The remaining RNG mixing helpers are implementation detail after you understand
the address tuple. The low-level CUDA assertion helper is also lookup material.

### 3. `pbg/core/rollout.py`: trace normal/sequential rollout end to end

Read [`rollout.py`](../../pbg/core/rollout.py) from `_admission_plan` through
`_sample_trees_sequential`, then the public `sample_trees_sequential` and
`sample_trees` wrappers.

Focus on this execution loop:

1. Allocate the `DeviceTreeBuffer`, occurrence arena, queues, root reaches,
   profiles, and working reaches.
2. Construct the admission transaction before private profiles are sampled.
3. Admit only as many trajectories as the byte plan permits.
4. Route frontier nodes into logical families, chance, or stopped leaves.
5. Select a family using full-batch priority, fill ratio, queue age, and stable
   family priority.
6. Dequeue a bounded batch, run actual-cost policy batching, sample events,
   append children, update reach, and mark occurrences done.
7. Continue until all starts are admitted and all queues drain.

Sequential rollout owns one overwritten working reach row per live trajectory.
It commits exact reach rows only at player decisions and stopped leaves, then
releases the working row. There is no selectable replay or all-node reach mode.

The default `family_batch_rows`, `max_active_trajectories`,
`arena_chunk_nodes`, and `node_budget` are not innocuous tuning constants:
they jointly control queue fill, peak temporary model input/output memory,
arena reservations, and the number of scheduler boundaries.

### 4. `pbg/core/branching.py`: read the alternative tree shape and scheduler

Read [`branching.py`](../../pbg/core/branching.py), especially
`sample_branched_tree` and its allocation/routing loop. Compare it with
`_sample_trees_sequential` rather than reading it in isolation.

The key difference is storage geometry: branched RB retains one labeled tree
rooted at the distinguished root, with randomized child multiplicities, while
sequential modes retain one realized path per start. Study where beta is
validated, where child occurrences are allocated, how family queues are shared,
and how policy outputs are indexed back to the current frontier.

Pay special attention to the fact that the root follows the ordinary branching
rule. It is not given special multiplicity. The rest of the code is mostly
branching-specific routing and validation once this distinction is clear.

### 5. `pbg/core/tree.py`: learn the retained topology and cache layouts

Read [`tree.py`](../../pbg/core/tree.py) in this order:

1. `TreeBatch` — flat topology representation (`parent`, `child_ptr`,
   `child_node`, `tree_id`, `state_ref`, `owner`, and edge metadata).
2. `TreeBuilder` — append-only construction and capacity checks.
3. `DeviceTreeBuffer` — preallocated device storage and the conversion from
   mutable rollout storage to the compact retained tree.
4. `PolicyCache` — exactly which old log-probabilities and values are retained
   for later PPO/value targets.
5. `ReachCache` and `DecisionReachCache` — full, compact, and recomputable reach
   representations.
6. `AliasTable` and `build_decision_alias` — decision-mass weighted row
   selection used by RB minibatches.

This module answers “what is stored?” more precisely than the dataclasses do:
topology is flat and index-based, fields are padded to maximum event/type
widths, and caches are aligned by stable decision-node IDs. Read validation
methods after understanding the layouts.

### 6. `pbg/core/values.py`: understand RB target memory and streaming

Read [`values.py`](../../pbg/core/values.py): `NodeTargets`,
`BackwardFrontierMemoryPlan`, `_layout_from_shape`,
`_frontier_memory_plan_from_shape`, `_reduce_streamed_leaf_batches`, and
`streamed_branched_value`.

The core decision is to stream completion leaves into a persistent `P/M`
backward frontier. Ready sibling subtrees are reduced and compacted as soon as
possible, rather than retaining all completion outputs or a full dense
backward workspace. `BackwardFrontierMemoryPlan` makes the intended overlap
explicit: leaf input, targets, frontier, scratch copies, and topology.

Then read `emit`, `conditionalize`, and `bootstrap` to connect the memory layout
to the estimator. The arithmetic is important, but the frontier lifecycle is
the efficiency-critical part.

### 7. Dataset and minibatch materialization

Read [`normal.py`](../../pbg/core/normal.py) first for the normal path:
`NormalDataset`, `build_normal_dataset_from_sampled`, and the gather/sample
functions. It converts sampled trajectories into compact scalar decision rows
and stores only the fields needed by later optimization.

Then read [`minibatch.py`](../../pbg/core/minibatch.py): `MinibatchSample`,
`sample_minibatch`, and `gather_minibatch_rows`. This is where RB target rows,
reach rows, legal-event masks, old policy data, and value data are assembled
into model-ready batches. The `gather_*` functions are particularly useful for
seeing what is copied per optimizer step versus retained for the whole
iteration.

### 8. Loss, distributed scaling, and optimizer boundary

Finish the algorithmic path with:

- [`losses.py`](../../pbg/core/losses.py): `LossSpec`, `ppo`, `rb_ppo`, and
  `finalize_metric_stats`. Read after the batch structures so the tensor names
  have meaning.
- [`distributed.py`](../../pbg/core/distributed.py):
  `global_mass_objective` and `stratum_objective_scale`. These determine how
  unequal rank-local decision mass becomes one global objective.
- [`optimization.py`](../../pbg/core/optimization.py):
  `sync_scaled_gradients` and `finite_gated_optimizer_step`. This is the
  boundary where accumulated gradients are synchronized, checked, clipped,
  and applied.

The important batching distinction is that RB rows are selected according to
decision mass (with alias/stratified selection), whereas normal training uses
the compact empirical decision dataset. The optimizer sees ordinary model
batches in both cases, but the row law and scaling are different.

## Read later or skim initially

These modules matter for a complete implementation audit but do not belong in
the first pass over allocation, retention, batching, or queues:

- [`config.py`](../../pbg/core/config.py): read the dataclasses and validation
  fields early, then defer YAML conversion and validation plumbing.
- [`game.py`](../../pbg/core/game.py): read `GameBundle` and the rollout
  protocol signatures early; defer the protocol boilerplate.
- [`completion.py`](../../pbg/core/completion.py): read the completion result
  shapes and `for_learning` contract before `values.py`; defer generic protocol
  definitions.
- [`checkpointing.py`](../../pbg/core/checkpointing.py): read for resume and
  state-lifetime auditing, not for the training algorithm.
- [`observability.py`](../../pbg/core/observability.py) and
  [`logging_util.py`](../../pbg/core/logging_util.py): use when checking whether
  queue age, synchronization, memory peaks, and completion quality are
  actually measured.
- [`__init__.py`](../../pbg/core/__init__.py): safe to skim last.

## A compact “second pass” checklist

After the first pass, revisit the code with these questions:

1. Before each allocation, what capacity or byte reservation proves it can
   fit? Check arena chunks, staging slabs, admission, completion batches, and
   backward frontier workspace.
2. At every phase boundary, which tensors remain live, and which are deleted?
3. Does each queue operation preserve insertion order, bounded capacity, status
   transitions, and observable age?
4. Which decisions cause a host/device synchronization? In particular, inspect
   fixed-size scheduler count blocks and model batch planning.
5. Does changing queue selection or physical batch boundaries alter RNG keys,
   estimator weights, or decision-mass scaling? It should not.
6. Is the stored representation proportional to possible topology or realized
   work? This distinguishes `full` from compact/recompute reaches and sequential
   from branched trees.
7. Is a “batch size” limiting logical work, physical model tokens, completion
   workspace, or optimizer rows? The core intentionally has separate controls
   for each.

## Suggested supporting material

Once this guide is complete, use the authoritative mathematics and pseudocode
under [`docs/theory`](../theory/) to verify estimator semantics, then consult
[`tests/test_execution_foundations.py`](../../tests/test_execution_foundations.py)
and [`tests/test_p13a_gpu_rollout.py`](../../tests/test_p13a_gpu_rollout.py) as
executable examples of the queue/arena contracts and rollout invariants.
The current audit inventory and readiness evidence are useful orientation, but
the source and theory documents remain the authority for design decisions.
