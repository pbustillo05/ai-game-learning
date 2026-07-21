# Implementation prompt: restore the authoritative storage lifecycle without losing dynamic branching

You are working in `/home/pablob/ai-game-learning`. Treat this as a substantial implementation and validation task, not as an audit-only exercise. Work through the production call graph, implement the migration, run conservative CPU and GPU validation, update the current documentation, and leave a coherent commit. Do not stop after adding abstractions or tests that are disconnected from training.

## Objective

Make the production normal, sequential Rao--Blackwell, and branched Rao--Blackwell training paths obey the memory ownership and storage lifecycle specified in `docs/theory/pseudocode/` and originally motivated by `archive/refactor/implementation_notes.md`.

The central requirement is a one-way data plane:

$$
\text{bounded live rollout state}
\longrightarrow
\text{chunked last-consumer rows}
\longrightarrow
\text{exact sealed rollout storage}
\longrightarrow
\text{streamed targets}
\longrightarrow
\text{immutable training storage}.
$$

Each phase must release its previous owners before the next phase begins, except for the explicitly budgeted source-plus-destination overlap during sealing. Allocation must follow realized rows and byte-admitted page capacity, not a configured total-node capacity.

Preserve the new KKT/dynamic-$\beta$, $T/C$ heads, nominal-versus-executed child correction, and protected-cohort root-refill mathematics. Those additions are valuable and must be integrated into the efficient storage design; reverting the branch to the older fixed-$\beta$ feature set is not an acceptable shortcut.

## Repository and safety rules

1. Read `AGENTS.md` and obey it completely.
2. Inspect `git status` before editing. The current worktree may contain user changes, including ongoing theory and dynamic-branching work. Preserve them and never reset, overwrite, or discard them.
3. Treat `/home/pablob/poker-gpu-evaluator` as read-only. All poker posterior/equity completion must continue through its documented binding in `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`.
4. Use `/home/pablob/.venvs/poker` or `./tools/poker-python`. Never install into system Python.
5. Before CUDA work, run `nvidia-smi`, use only idle GPUs, set device visibility explicitly, stage tiny runs first, and verify that the selected GPUs return to idle afterward. Never deliberately trigger a CUDA OOM.
6. Compile changed TeX only into `/tmp`; do not add generated TeX artifacts.

## Authority and conflict resolution

Read all of the following before changing code.

### Primary lifecycle and implementation authority

- `archive/refactor/implementation_notes.md`: the original compact-storage intent and the distinction between branching edge rows and sequential implicit edges.
- `docs/theory/pseudocode/storage_lifecycle.tex`:
  - exact versus natural model domains and partition contracts;
  - stable row domains and owning objects;
  - normative retained fields;
  - compact topology;
  - write times and phase lifetimes;
  - old-pattern versus new-invariant table;
  - schema-derived byte formulas and four-phase peaks;
  - memory-only sequential root-width control;
  - implementation order and acceptance gates.
- `docs/theory/pseudocode/family_scheduled_rollout.tex`: partitioned queue rollout, logical child reservation, bounded physical materialization, chance closure, exact sealing, last-consumer release, and common failure agreement.
- `docs/theory/pseudocode/backward_targets.tex`: normal, sequential-RB, and branched bounded target builders plus finalization.
- `docs/theory/pseudocode/sequential_memory_control.tex`: root width chosen from rollout, seal, target, and optimization byte feasibility only.
- `docs/theory/pseudocode/distributed_update.tex`: immutable `TrainingStorage`, index-only statistical draws, bounded physical gather/forward/backward/release, optional reference-cache admission, and one flat gradient SUM buffer.

### Mathematical authority

- `docs/theory/core_math.tex` is authoritative for sampling laws, estimators, losses, and the newer dynamic branching work. In particular read:
  - normal and sequential-RB rollout/return laws;
  - branched rollout and the no-private-tilting requirement;
  - current `SampleTree`, including $(N,\widetilde N)$, the queue safeguard, KKT intensity, and lower-size root refill;
  - branched backward moments, $T/C$ targets, protected root target cohort, and the no-stored-$Q$ production rule.
- Read the current worktree version, not only `HEAD`: it contains newer root-refill/protected-cohort mathematics.

### Poker model-input authority

- `docs/theory/MHA_model.md` and `docs/theory/folder_impact.md`.
- The partition contracts in `docs/theory/pseudocode/storage_lifecycle.tex`:
  - $K=R^2$ natural preflop model rows;
  - $H=\binom{4R}{2}$ exact and postflop rows;
  - full-deck widths 169 and 1326;
  - immutable folded-summary dependency anchors only when $A<P$;
  - no postflop street partitions.

### Required documentation reconciliation

The current sources have a real version skew: the storage pseudocode still describes fixed-intensity production v1 and omits retained/transient $T/C$, nominal/executed-count, queue-safeguard, and protected-root-refill fields, while current `core_math.tex` and production code support them. Resolve this explicitly before or alongside implementation:

- keep `core_math.tex` authoritative for the new scientific quantities;
- extend the pseudocode storage schemas and target algorithms with only the fields actually required by the current dynamic-branching equations;
- preserve the lifecycle, natural-width, ragged-edge, bounded-frontier, and release invariants;
- do not use the documentation mismatch to retain the legacy tree/reach/target path;
- do not silently change the estimator to make memory management easier.

## Verified current-state diagnosis

Do not assume the prior refactor commits completed the migration. Verify these facts in the current worktree and use them as the starting defect list.

### The refactor data plane exists but is not the production data plane

Commit history shows the disconnect:

- `165cfb1` is the pre-refactor baseline.
- `04bd089` added `pbg/core/partition.py`, `storage.py`, `topology.py`, `targets.py`, and the phase memory controller. However, production rollout still instantiated `DeviceTreeBuffer`, `DecisionReachCache`, and the legacy `PolicyCache`; production did not call `build_normal_targets`, `build_sequential_rb_targets`, or `build_branched_rb_targets`.
- `86756b8` added the bounded distributed learner, but did not connect the new rollout/target builders.
- `8d94724` added `_seal_iteration_training_storage` as a compatibility conversion from already-built legacy targets into `TrainingStorage`. This happens after completion and backward target construction, so it is not `SealRolloutStorage` as specified by the pseudocode.
- `537700f` optimized the learner built on the post-hoc store.
- `79a354b` and the current uncommitted changes added dynamic KKT branching, $T/C$ heads, queue correction, and root refill on top of the still-legacy branched rollout and `NodeTargets` path.

The result is two architectures in one repository: well-tested new storage/target primitives, and a production path that still constructs large legacy owners before converting them at the end.

### Production phase order is wrong

The current effective RB order is approximately:

1. allocate capacity-shaped poker state/tree/reach/control storage;
2. roll out into `TreeBatch`, `ReachCache`, `PolicyCache`, and `DecisionReachCache`;
3. complete leaves and build padded `NodeTargets` with `streamed_branched_value`;
4. create `_RolloutSealInput` holding the legacy targets and caches;
5. call `_seal_iteration_training_storage` to copy them into `TrainingStorage`;
6. delete the compatibility input and optimize.

The required order is rollout, exact sealing, streamed target construction from sealed topology/leaves, target finalization and topology/leaf release, then optimization. The current order pays for legacy topology/reaches/padded targets before the exact store exists and causes the sealing peak to include the wrong owners.

### Current concrete memory violations

At minimum, address all of these:

- `RolloutConfig.node_budget` still controls production allocation, validation, progress limits, benchmark identity, tests, and memory forecasts.
- `build_bundle()` allocates the poker backend for `maximum_roots_per_gpu * node_budget`, even when the controller selects fewer current roots.
- `PokerTensorStateStore` eagerly allocates every state/history/fold column for that total capacity.
- `DeviceTreeBuffer` eagerly allocates node and edge columns for `roots * node_budget`; `finalize()` returns views that retain the full backing storage.
- Branched rollout eagerly allocates `reach_rows[node_budget,P,H]`.
- Branched `nominal_count`, `beta_rows`, `log_player_sum`, `log_ancestor_product`, and `queue_cost` are all `node_budget`-wide parallel allocations.
- Every branched family queue is sized to total `node_budget`, not live frontier.
- `DevicePartitionSummaryCache` uses a capacity-wide validity vector and lazily creates capacity-wide summary fields although only sparse dependency anchors are real owners.
- Sequential rollout retains lists of full `[D,P,H]` decision reaches and later concatenates them. Poker currently sets `requires_full_decision_reach = True`.
- Branched rollout retains lists of full decision reaches in addition to the all-node reach table.
- Leaf reaches, policy outputs, decision reaches, and several topology fragments are accumulated as tensor lists and concatenated at finalization, causing allocator fragmentation and source-plus-destination overlap outside an admitted seal transaction.
- `PolicyCache.old_logp_child` and legacy `NodeTargets` use `[D,E_max,H]` padded action axes. One high-degree node or refilled root widens every decision row.
- `pbg/core/values.py` grows backward-frontier tensors through repeated `torch.cat`, rather than using a precomputed slot bound and reusable arena.
- `_seal_iteration_training_storage` reconstructs natural inputs after targets from full decision reaches, instead of writing natural partition inputs at decision creation.
- Sealed poker learner inputs retain `state_ref` into the original all-node `PokerTensorStateStore`; `policy_forward_partition()` calls `encode_public()` and summary services through that store during optimization. Thus the all-node public-state owner survives under an integer handle even though `TrainingStorage` itself does not expose the tree.
- Folded summary anchors are state handles into a capacity-shaped cache, not compact references into an immutable sparse anchor store.
- `_analytical_memory_plan()` is a handwritten approximation based on `roots * node_budget`, includes objects whose lifetimes should not overlap, and is not generated by the schemas that allocate the real tensors.
- The sequential controller changes root width, but the persistent backend was already allocated for the configured maximum and the phase model is calibrated against the legacy capacity geometry.
- `torch.cuda.empty_cache()` is used at the rollout/learner boundary to make planning usable. It may remain a diagnostic/allocator action if justified, but it cannot substitute for last-owner release or be required for correctness.

### Existing tests do not prove production integration

- `tests/test_refactor_storage_targets.py` exercises synthetic `RolloutStorage` and target objects directly.
- `tests/test_refactor_gpu_acceptance.py` runs production training, but mostly asserts finite loss and nonzero peak metrics; it explicitly sets `node_budget` and does not prove that the production call graph used the new rollout/target builders.
- `tests/test_phase3_migration.py` proves schema versions and poker widths in isolation, not absence of legacy owners during a real iteration.

Add tests that fail if production silently returns to the compatibility path.

## Required end-state architecture

### 1. One production pipeline, not a post-hoc compatibility conversion

Refactor the production orchestrator so each mode directly produces an exact `RolloutStorage`, then calls the matching builder in `pbg/core/targets.py`, and hands the resulting immutable `TrainingStorage` to `DistributedFixedRolloutUpdate`.

The production call graph must no longer build `_RolloutSealInput`, legacy `NormalDataset`, `PolicyCache`, `DecisionReachCache`, or legacy `NodeTargets` as an intermediate representation. Delete those production dependencies once callers and evaluation tools are migrated. Test-only helpers may remain only when clearly isolated and unable to allocate in training.

The following sequence must be visible in code and telemetry:

1. choose/freeze root work and logical RNG addresses;
2. run partitioned rollout into byte-admitted append-only source arenas;
3. seal exact topology/input/decision/action/leaf destinations while sources coexist under one transaction;
4. release source chunks, queues, mutable public states, and active reaches;
5. build targets in bounded completion/frontier batches from `RolloutStorage`;
6. finalize sampling metadata and release topology, leaves, completion outputs, and target scratch;
7. optimize only from `TrainingStorage` using bounded physical subspans;
8. release `TrainingStorage` and optional reference caches.

### 2. Real schemas must allocate every rollout/training field

Consolidate immutable schemas so the same declaration drives:

- allocation;
- per-page capacity bytes and alignment;
- transactional admission;
- sealing destination extents;
- validation;
- memory forecasts and diagnostics.

Do not maintain a second handwritten field-count model. Include all current mode-specific fields, including dynamic-branching additions, RNG address components, status, root/ordinal, partition row, leaf row, action row, nominal and executed count where needed, log decision weight, $T/C$ transient heads/targets, and protected-root metadata.

Logical row ids must be monotone and stable across pages. A tensor pointer or view into a growable allocation is not an identity.

### 3. Page-based, byte-admitted rollout owners

Remove total-node capacity from production structures. Evolve or replace the existing arenas with a common paged structure-of-arrays design:

- allocate a page only when an append crosses the current page extent;
- reserve the entire page transaction before CUDA allocation;
- roll back without advancing cursors if reservation or allocation fails;
- keep at most final-page slack per column group;
- expose allocated, used, slack, and high-water bytes;
- allow safe reuse across iterations only if retained pages remain explicit in the phase ledger and can be released when the controller needs the bytes.

Migrate:

- tree/topology node columns;
- state/history ownership;
- active/all-node branched reach and controller columns;
- leaf and partition tables;
- sparse summary anchors;
- queues sized by live frontier rather than total generated nodes.

Avoid a facade that pages metadata but still allocates the wide state/reach payload at `node_budget` capacity.

Do not force mutable state slots to share the append-only logical node-id domain. Topology and retained row ids are monotone; live public-state and active-reach handles may point into separately managed recyclable slot/page pools. A completed node must return those slots after its children and retained input/leaf rows commit. If poker history uses persistent sharing, make ownership/refcounts explicit and prove that dead ancestors do not keep payload proportional to all realized nodes alive; otherwise materialize the bounded child history needed by the next consumer before recycling the parent slot.

### 4. Last-consumer natural poker inputs

At every player-node creation, route the node to its immutable partition and append the complete natural learner input then, while the active reach and public state are available.

- Preflop range rows are `[P,K]`, never `[P,H]`.
- Postflop range rows are `[A,H]` for live players only.
- Store the immutable public/history/model input needed to forward that decision. Do not keep a `state_ref` whose consumer requires the original all-node mutable state table during targets or optimization.
- If a compact immutable public-state encoding is shared, give it its own exact decision/anchor-row ownership and schema. It must not keep chance-only or discarded all-node payload alive.
- Folded summaries are sparse immutable dependency-anchor rows. Create/refresh only on fold or street transition, reference compact anchor rows, and allocate none for $A=P$.
- Store exact RB leaf reach and sibling-evaluator completion input only at stopped leaves.
- Release active reach/profile and mutable state immediately after all child/input/leaf consumers commit.

In branched mode, each live occurrence may own one active full reach row, but completed internal nodes may not. The occurrence/topology row should carry a temporary handle into a recyclable reach pool, not an embedded or parallel all-node `[N,P,H]` table.

Normal mode must retain only scalar selected loss data. Sequential RB must retain natural old value/log-probability rows plus one exact return and one deduplicated posterior reference. Branched RB must use ragged action rows and the exact current mathematical target set.

### 5. Exact compact topology and sealing

Sequential modes must seal only decision chronology:

- `trajectoryPtr[G+1]`;
- `decisionPartition[D]`;
- `decisionRow[D]`;
- `leafRow[G]`.

Chance nodes must not survive in sequential topology.

Branched mode must seal one rooted six-column span with contiguous children:

- `parent[N]`;
- `childBegin[N]`;
- `degree[N]`;
- `decisionPartition[N]`;
- `decisionRow[N]`;
- `leafRow[N]`.

Store only the minimal extra root/protected-cohort metadata required by current `core_math.tex`. Player action spans must align one-for-one and in the same order as target-eligible child spans. If the refilled root has final children that are not root target actions, represent that explicitly without reintroducing a dense maximum-action axis or corrupting descendant topology.

Sealing must:

- compute realized extents first;
- admit source capacity plus exact destinations plus bounded scan/copy scratch before allocating destinations;
- allocate each destination column exactly once;
- copy stable page spans without repeated payload-proportional concatenation;
- synchronize and validate topology, row domains, and byte accounting;
- release source pages only after their last copy;
- report source, destination, scratch, overlap peak, slack, and retained bytes.

### 6. Stream targets from sealed storage

Use and extend `pbg/core/targets.py`; do not route production through `pbg/core/values.py::streamed_branched_value` and padded `NodeTargets`.

- Normal: bounded reverse segmented player-clock recurrence; retain one scalar `lambdaReturn` per decision.
- Sequential RB: bounded sibling-evaluator completion microbatches; one posterior row per required `(trajectory, player)`; one `posteriorRef` per decision; exact `[D_m,H]` return; release completion values and running moments immediately.
- Branched RB: precompute the exact maximum live frontier slots from the sealed topology; allocate that arena once; reduce children into parents; release child slots when contracted; retain ragged child posterior/advantage action rows and decision value targets.
- Dynamic branched RB: extend the bounded builder to compute current $T/C$ targets, masks, nominal/executed corrections, and the protected root cohort exactly as current `core_math.tex` specifies. Keep frozen $T/C$ predictions only as transient target-builder inputs if the learner does not need them.
- Derive parent posterior means and other cheap algebraic values at loss gather time.
- Never store a production $Q$ target.
- Never rebuild a discarded reach or public path.

After finalization, `TrainingStorage` must be the only iteration-owned survivor and must contain no topology, leaf completion input, full/all-node state store, reach cache, frontier, queue, or source arena.

### 7. Preserve dynamic KKT and root refill correctly

The storage migration must not weaken the newer branching design.

- Preserve fixed and post-warmup dynamic modes.
- Preserve logical counter-based randomized rounding and separate keys for every child/draw.
- Preserve nominal $\widetilde N$, executed $N$, queue-safeguard correction, KKT controller checkpoint state, $T/C$ head training, and counterfactual tree-size observation.
- Preserve one distinguished root and the ordinary branching rule at its first expansion.
- Preserve the lower-size safeguard's fresh root-child ordinals, protected prefix $N_0(\varnothing)$, pre-refill boundary, final root transition masses, protected root target cohort, and learning data below refill roots.
- Add equality tests against direct current `core_math.tex` formulas for queue suppression and root refill, including root $N_0=1$, final degree greater than $N_0$, nested descendants, and $\lambda=0,1$.

Memory admission and the scientific queue safeguard are different layers:

- the queue safeguard may execute one child only where the current mathematical law explicitly defines $(N,\widetilde N)$ and its corrections;
- allocator/page admission must reserve all bytes for the already selected executed children before their random draws and mutation;
- a failed hard allocation transaction must stop cleanly and collectively, not silently alter $N$, retry the tree, or resample the iteration;
- do not invent a memory-dependent offspring suppression rule without adding and proving its estimator correction in the mathematical authority.

### 8. Retire `node_budget` completely

Implement the valid parts of `docs/audit/node_budget_retirement_20260721.md`, but resolve its suggestions against the current mathematical and lifecycle authority. In particular, do not replace `node_budget` with another global node count and do not make opportunistic memory suppression silently change the branching law.

Required changes:

- remove `RolloutConfig.node_budget` from new production configuration;
- rename `arena_chunk_nodes` to a storage page-size execution control such as `storage_page_rows`;
- use the exact game-derived maximum player/public path bound only for sequential progress assertions and controller cold-start geometry;
- remove `configured_node_budget` from target reduction;
- replace node-count memory formulas with schema/page byte formulas and realized-count observations;
- remove node capacity from bundle construction, benchmark identity, scientific checkpoint identity, production cutoffs, and scheduler progress limits;
- retain explicitly named test-only page/admission fault injection where needed;
- migrate old YAML/checkpoints explicitly and log the schema transition rather than silently reinterpreting the field;
- update evaluation and variance tools to report realized rows/pages/bytes instead of capacity margin.

For branched mode, bounded depth does not imply a practical deterministic width bound. Use byte-admitted pages, a mandatory-completion reserve based on the remaining exact public path bound, calibrated model/completion/target workspaces, and the existing mathematically defined queue/KKT controller. If the minimum legal continuation cannot be admitted, stop cleanly; do not depend on a later capacity exception.

### 9. Four-phase memory control from actual ownership

Rebuild memory forecasting around the real schemas and owners.

- Rollout prediction: live mutable states/frontier, allocated source pages and slack, queues, active reaches/profiles, summary anchors, and bounded inference workspaces.
- Seal prediction: final source-page capacity plus exact destinations plus seal scratch.
- Target prediction: sealed topology/inputs/old outputs/leaves, target destinations, bounded completion workspace, and bounded sequential/branched frontier.
- Optimization prediction: immutable training storage, fixed model/reference/optimizer state, flat gradient buffer, optional admitted reference cache, one index vector, and one physical learner subspan.

Do not add mutually exclusive owners. Account allocated capacity, allocator-reserved high water, backend workspace, and NVML/driver observations separately.

For normal and sequential RB, retain the common per-GPU root-width controller, but make its analytic prior and observations schema-derived. Build the backend for current paged growth rather than the configured maximum total node count. Width remains frozen before draws; unsafe evidence decreases immediately; safe growth remains delayed and capped; all ranks use the minimum safe width.

For branched mode, add a common byte controller that cooperates with, but remains distinct from, the scientific KKT controller. It must bound page growth and physical forward/completion/target workspaces, perform collective-safe stop decisions, and never catch OOM then reroll.

Remove any correctness dependency on `torch.cuda.empty_cache()`. If retained as an optional phase-boundary allocator optimization, measure its effect and prove the same workload succeeds without it under the declared hard bound.

### 10. Learner boundary and retained store

Keep the useful `pbg/core/learner.py` machinery, but verify its input no longer hides legacy owners.

- `TrainingStorage` contains complete immutable natural inputs rather than state handles into a full rollout store.
- Current, old, and cached reference outputs remain on natural model width $Y_m$.
- Exact posteriors/returns/advantages remain on $X_m=H$.
- Statistical draws are index-only.
- Every physical subspan gathers, forwards, backwards, and releases before the next subspan.
- Reference caches are optional and transactionally admitted; the uncached singleton path is the safety baseline.
- One flat fp32 gradient SUM buffer is retained; no DDP averaging or copy-flatten buffer is reintroduced.

## Implementation strategy

Use incremental, testable vertical slices while keeping one final architecture.

1. Add call-graph and ownership tests that expose the current compatibility path.
2. Reconcile the pseudocode schemas with dynamic branching and root refill.
3. Introduce the common byte-admitted page store and migrate compact topology/controller columns.
4. Migrate poker state/public/history access to append/gather/release APIs and sparse summary anchors.
5. Make normal production rollout produce real `RolloutStorage`; wire normal target building and remove its legacy dataset path.
6. Migrate sequential RB, including natural decision inputs, exact leaf completion rows, decision-only CSR, posterior deduplication, and target streaming.
7. Migrate branched RB with ragged actions, bounded frontier, KKT $T/C$, queue correction, and protected root refill.
8. Delete production compatibility conversion and legacy allocation/config surfaces.
9. Rebuild phase prediction/checkpoint migration from schemas.
10. Validate, profile, optimize only measured physical bottlenecks, and update current documentation/audits.

Temporary adapters are acceptable only inside an actively developed vertical slice and must be removed in the same final migration. The final production path may not allocate both representations. Do not declare success while `_seal_iteration_training_storage` still converts legacy `NodeTargets`, or while `TrainingStorage.state_ref` keeps the all-node state store alive.

## Mandatory codebase cleanup and sole-implementation gate

This refactor was intended to replace the old implementation, not sit beside it. Once the direct production pipeline works, perform a deletion-focused cleanup pass. Keeping obsolete code would preserve the exact ambiguity that caused this regression.

### Delete superseded production artifacts

After migrating every real consumer, delete rather than deprecate the old allocation path, including as applicable:

- `DeviceTreeBuffer` and its capacity/finalize-view machinery;
- legacy `PolicyCache`, `ReachCache`, and `DecisionReachCache`;
- legacy `NormalDataset`/normal minibatch storage;
- padded `NodeTargets` and the legacy `streamed_branched_value` frontier;
- `_RolloutSealInput` and `_seal_iteration_training_storage`;
- old tree/reach minibatch gather functions in `pbg/core/minibatch.py`;
- capacity-shaped poker state and partition-summary cache APIs once their paged/live-slot replacements are complete;
- handwritten legacy memory plans and controller adapters based on node capacity;
- retry, replay, compatibility, and configuration migrations that are no longer needed after the supported checkpoint boundary;
- imports, aliases, exports, comments, metrics, and type annotations that keep the old object model conceptually alive.

Do not preserve a production-capable copy in `archive/`. Historical prose and old commits are enough. If a small pure helper remains useful, move it to the new owning module, give it terminology consistent with the authoritative design, and delete the legacy container around it.

### Migrate or delete every consumer

- Production training, benchmarks, probes, variance tools, and evaluators must use the same sole rollout/storage APIs.
- Delete tests whose only purpose is to exercise retired classes or `node_budget` behavior. Replace them with tests of the new invariant instead of mechanically porting obsolete expectations.
- Remove stale YAML fields, CLI flags, documentation examples, audit reading orders, and checkpoint identity fields.
- Do not leave importable dead modules merely because coverage no longer reaches them.
- Any exception requires explicit user approval and must be documented with one owner, one removal condition, and a near-term deadline. A generic "legacy compatibility" exception is unacceptable.

### Prove cleanup, not just migration

Add a final source and import inventory that demonstrates:

- one production rollout entry point per mode over shared scheduling/storage primitives;
- one target-builder family and one learner boundary;
- no duplicate owner/type vocabulary for the same data;
- no imports from retired modules in `pbg/`, tools, or active tests;
- no active configuration or checkpoint field for a retired allocation concept;
- static searches for all retired symbols return either no result or an explicitly historical Markdown reference.

Report deleted files/classes, removed lines, and the final module ownership map. Net line count need not decrease if the correct paged poker state interface requires substantial code, but every superseded implementation and its tests/configuration surface must be gone.

## Tests and proof obligations

### Static and call-graph gates

Add tests or source gates proving that production training does not import, instantiate, or retain:

- `DeviceTreeBuffer`;
- legacy `TreeBatch` in sequential modes;
- `PolicyCache`, `ReachCache`, or `DecisionReachCache`;
- `NormalDataset` as a production rollout store;
- legacy `NodeTargets` or `streamed_branched_value`;
- `_RolloutSealInput` or `_seal_iteration_training_storage`;
- a full `[N,P,H]` or `[D,P,H]` decision reach memo;
- padded `[D,E_max,H]` policy/target tables;
- production `node_budget` or `configured_node_budget`;
- production target/frontier growth through repeated payload `torch.cat`.

Evaluation-only code must be clearly isolated. If an evaluation tool shares production rollout, migrate it rather than preserving a second allocator path.

### Storage and lifecycle tests

Cover:

- transactional page allocation and rollback before mutation;
- exact accounting including alignment, headers, allocator slack, final-page slack, and reusable pages;
- multi-page stable logical ids and device gather/scatter;
- exact multi-page sealing and source/destination/scratch overlap;
- sequential CSR chronological order and one leaf per root;
- branched contiguous child/action spans and root row zero;
- root-refill protected target span versus final topology span;
- sparse folded-summary anchors, refresh rules, and zero $A=P$ summary allocation;
- last-consumer release with alias-aware unique-storage inventories at every phase boundary;
- no hidden all-node state/reach owner after seal and no topology/leaf owner after target finalization.

### Mathematical equality tests

For all three modes, compare randomized small PBGs to direct formulas. Include:

- $\lambda=0$ and $\lambda=1$;
- zero posterior mass;
- single and multiple branched children;
- fixed $\beta$ and dynamic KKT;
- $(N,\widetilde N)$ queue-safeguard correction;
- root refill with protected and added children;
- $T$ present/absent masks and $C$ targets;
- decision sampling/log-mass equality;
- no stored $Q$ target;
- identical sampled objects across queue/page/physical-subspan sizes and different rank ids.

### Poker shape tests

For $R=4$ and $R=13$, and multiple live-player counts, assert through real production rollout, sealing, targets, reference cache/recompute, learner forward, and loss:

- preflop natural model input/output/cache width $K=R^2$ (169 for $R=13$);
- postflop natural model input/output/cache width $H=\binom{4R}{2}$ (1326 for $R=13$);
- sequential/branched RB posterior, return, and advantage exact width $H$ in all partitions;
- impossible hands have zero posterior mass;
- no street-specific postflop partition;
- completion input dtypes/layouts exactly match the sibling API.

### Memory-controller and failure tests

Use synthetic byte ceilings, not actual OOM, to force:

- rollout-, seal-, target-, and optimization-limited cases;
- page admission rollback;
- cold start and no-fit pre-draw stop;
- immediate contraction and delayed/capped growth;
- rank-minimum agreement;
- mandatory-continuation failure in branched mode;
- singleton learner failure;
- collective-safe common stop;
- checkpoint/resume equality;
- no reroll or retry-dependent RNG address;
- cache exclusion from root feasibility and fallback to empty reference cache.

### Production integration tests

Instrument one tiny real iteration in each mode and assert exact owner inventories at boundaries, not only finite loss:

| Boundary | Required live iteration owners |
|---|---|
| end rollout, before seal | source pages, live-to-seal metadata, no completed-node mutable state/reach |
| after seal | exact `RolloutStorage`, no source pages/queues/active reaches/all-node mutable state |
| after targets | exact `TrainingStorage`, no topology/leaves/completion/frontier |
| during learner | `TrainingStorage`, optional admitted reference cache, one bounded physical subspan |
| iteration end | no iteration store or cache |

Record unique tensor storage bytes and allocator allocated/reserved/NVML peaks for each boundary.

## CPU and GPU validation

1. Run focused storage/topology/target/controller tests.
2. Run the full feasible CPU suite and distributed CPU tests.
3. Check `nvidia-smi`; select only idle GPUs.
4. Precompute schema-derived bounds for the tiny configurations.
5. Run a one-GPU normal smoke, then sequential RB, then fixed branched, then dynamic branched with tiny conservative work.
6. Run bounded multi-iteration normal and sequential-RB soaks to verify stable allocator behavior and controller observations.
7. Run tiny fixed/dynamic branched soaks including queue safeguard and root refill without approaching the hard memory ceiling.
8. If two GPUs are idle, run unequal-local-mass distributed normal/sequential tests and a tiny branched common-stop/controller test.
9. Verify final GPU idle state and no lingering processes.

For each mode report:

- realized occurrences, decisions, actions, leaves, and pages;
- source/destination/scratch/slack bytes;
- retained `TrainingStorage` bytes by partition and field;
- rollout, seal, target, and optimization predicted versus allocated/reserved/NVML peaks;
- guard/headroom and limiting phase;
- model/completion/reference rows and physical-tail histogram;
- player decisions or spots per second;
- before/after comparison on identical hardware and sampled-object semantics where possible.

## Documentation, migration, and cleanup

- Update the pseudocode documents so dynamic branching and storage lifecycle no longer contradict one another.
- Update current architecture/training docs to describe the real call graph and owner transitions.
- Add a current audit under `docs/audit/` with exact commit-under-test, commands, hardware, phase bytes, performance, and limitations.
- Mark the 2026-07-17 acceptance notes as historical if their claims are no longer accurate; do not rewrite historical measurements as if they came from the new implementation.
- Update `docs/audit/node_budget_retirement_20260721.md` if implementation decisions differ from its proposal.
- Version and migrate configuration/checkpoint/controller/storage schemas explicitly.
- Remove obsolete configs or migrate them deterministically; do not silently reinterpret `node_budget`.
- Do not commit run checkpoints, large profiler artifacts, or generated TeX files.

## Final acceptance criteria

Do not claim completion unless all of the following are true:

- production directly executes the authoritative rollout, seal, target, and learner objects;
- there is one tensor owner for every retained field;
- `node_budget` is absent from the production configuration and allocation contract;
- allocation follows realized page rows within bounded final-page slack;
- sequential decisions never retain full `[D,P,H]` reaches;
- branched mode has no full `[node_budget,P,H]` reach table or total-capacity queues/control columns;
- poker training inputs no longer retain an all-node state store through `state_ref`;
- player decision inputs are written at creation on natural width;
- leaves own completion input/reach exactly once;
- branched actions and targets are ragged;
- target construction uses a prebounded reusable frontier;
- topology and leaves are gone before optimization;
- no rollout replay, reroll, memory-pressure retry, or stored $Q$ target exists;
- KKT/dynamic-$\beta$, $T/C$, queue correction, and root refill pass direct mathematical equality tests;
- normal, sequential RB, fixed branched RB, and dynamic branched RB pass conservative CUDA smokes;
- phase ledgers reconcile schema bytes with measured owners and show no unexplained monotone growth;
- selected GPUs return to idle after validation.

## Final response and commit

Inspect the full diff and the production call graph before committing. Commit only this migration and its tests/docs with an informative message such as:

```text
refactor(core): restore byte-admitted rollout storage lifecycle
```

End the final response with:

- commit hash;
- concise architecture summary;
- explicit statement that production no longer uses the legacy compatibility pipeline;
- CPU and GPU commands/results;
- GPU ids and final idle confirmation;
- per-phase predicted and measured peaks/headroom;
- retained bytes and 169/1326 shape confirmation;
- before/after throughput where comparable;
- confirmation of no `node_budget`, no all-node/decision reach memo, no padded action axis, no reroll, and no stored $Q$;
- confirmation that fixed and dynamic branching, $T/C$, queue correction, and protected root refill remain mathematically intact;
- any narrowly scoped remaining limitation.
