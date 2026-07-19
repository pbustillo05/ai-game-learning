# Prompt 1 — implement the sealed rollout/target data plane and memory-only root controller

You are working in `/home/pablob/ai-game-learning`. Treat this as one large implementation task sized for roughly a 250k-token context window. Work autonomously through implementation, tests, GPU validation, and a git commit; do not stop after producing a plan or an audit.

## Objective

Implement the new game-neutral rollout, storage, sealing, target-building, and sequential root-width design while reusing the good low-level machinery already present in `pbg/core/`. This is a refactor, not a ground-up rewrite. The resulting data plane must allocate from immutable schemas, retain only last-consumer data, seal chunked rollout data into exact contiguous stores, eliminate reach replay, and choose normal/sequential-RB root counts solely from a conservative GPU-memory budget.

This is phase 1 of a three-phase refactor. Leave a clean, tested commit that phase 2 can use to implement the final distributed learner. Existing learner code may use zero-copy compatibility views over the new immutable `TrainingStorage`, but it must not keep a duplicate legacy rollout/reach/target store alive.

## Read these sources before editing

1. Read the repository `AGENTS.md` completely and obey it. Inspect `git status`; preserve unrelated user changes and never reset or discard them.
2. The implementation authority is:
   - `docs/theory/pseudocode/storage_lifecycle.tex`:
     - authority, v1 scope, canonical exact/model domains, and partition interface at lines 34–97;
     - poker partition, completion, and strict natural-width contracts at lines 99–167;
     - owning objects, append-only chunks, explicit sealing, and normative retained fields at lines 169–276;
     - compact topology and phase lifetimes at lines 278–338;
     - byte formulas and capacity accounting at lines 370–496;
     - memory-only root control and common failure agreement at lines 498–534;
     - acceptance gates and intended module boundaries at lines 599–665.
   - `docs/theory/pseudocode/family_scheduled_rollout.tex`:
     - Algorithm `alg:partitioned-queue-rollout`, lines 41–182;
     - Algorithm `alg:seal-rollout-storage`, lines 184–215;
     - Algorithms `alg:reserve-logical-children`, `alg:materialize-reserved-children`, and `alg:chance-close`, lines 217–319.
   - `docs/theory/pseudocode/sequential_memory_control.tex`:
     - Algorithm `alg:sequential-memory-root-width`, lines 34–104;
     - Algorithm `alg:observe-sequential-memory`, lines 106–127.
   - `docs/theory/pseudocode/backward_targets.tex`:
     - Algorithm `alg:normal-streaming-targets`, lines 36–63;
     - Algorithm `alg:sequential-rb-streaming-targets`, lines 65–109;
     - Algorithm `alg:branched-rb-bounded-targets`, lines 111–168;
     - Algorithm `alg:finalize-training-storage`, lines 170–187.
3. Use `docs/theory/core_math.tex` as the mathematical authority:
   - terminal-rollout decision law and player-clock lambda return, lines 345–412, especially equations `eq:terminal-rollout-index-law` and `eq:sequential-lambda-return`;
   - sequential Rao–Blackwell rollout law, exact posterior domain, completion boundary, and lambda return, lines 553–710, especially `eq:preterminal-rollout-index-law`, `eq:rao-blackwell-lambda-return`, and Algorithm `alg:mc-preterminal`;
   - branched sampled-tree law and no shared-randomness rule, lines 748–876;
   - branched targets and backward moments, equations `eq:branched-targets` and `eq:branched-backward-updates`, lines 878–949 and 1200–1322. Lines 1321–1322 explicitly say fixed-intensity PPO v1 does not store `Q`.
4. For poker model inputs and anchors, read `docs/theory/MHA_model.md` and `docs/theory/folder_impact.md`. For completion, read the sibling repository's read-only `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`. Do not modify that sibling repository and do not implement a second poker equity/posterior evaluator locally.
5. Inspect the current implementation before deciding what to retain. In particular, reuse or carefully evolve:
   - `ByteAccountant`, `ChunkedOccurrenceArena`, `DeviceWorkQueue`, `ReusableStagingSlabs`, and counter-based logical RNG in `pbg/core/execution.py`;
   - the useful topology/alias pieces in `pbg/core/tree.py` and fixed-intensity branch sampling in `pbg/core/branching.py`;
   - the existing family scheduler in `pbg/core/rollout.py`;
   - `MemoryLedger` and the conservative forecasting/checkpoint machinery in `pbg/core/memory_control.py`;
   - `conditionalize`, `bootstrap`, and the bounded backward-frontier ideas in `pbg/core/values.py`;
   - checkpointing and observability infrastructure;
   - poker engine/tensor-state/range/model/partition-summary code in `pbg/games/poker/`.

## Required implementation

### 1. Immutable partition, completion, and storage schemas

Create or consolidate the intended game-neutral boundaries (`pbg/core/partition.py`, `pbg/core/topology.py`, `pbg/core/storage.py`, and `pbg/core/targets.py` are the preferred boundaries unless the existing code gives a compelling simpler placement).

- `PartitionSpec` must expose a stable integer id, diagnostic name, pure router, optional scheduler preorder, canonical ordered exact-state domain `X_m`, natural model-row domain `Y_m`, immutable `phi_m`, natural input/forward boundary, and retained/transient byte contracts. Core code must never parse poker street/family names to infer behavior.
- `CompletionSpec` must declare opaque immutable leaf fields, exact reach row order, keys/seeds, output rows, and bounded workspace. The core owns scheduling/accounting only.
- Define immutable schemas for occurrence, partition-input, decision, action, leaf, sequential-posterior, topology, rollout, and training stores. One schema must generate allocation, exact capacity byte accounting, sealing extents, validation, and diagnostics. Do not keep handwritten approximate field-count formulas disconnected from actual tensors.
- Logical row ids must remain stable across chunks; no pointer into a growable tensor is an identity.
- Count actual allocated capacity, alignment, chunk headers/slack, allocator-reserved high water, and backend workspaces. Keep `ByteAccountant`, but make reservations transactional and schema-derived. A failed reservation must occur before the CUDA allocation it protects.

### 2. Natural poker tensor layouts are non-negotiable

For rank count `R`, use `K = R**2` and `H = choose(4R, 2)` exactly as `storage_lifecycle.tex` specifies.

- Every dense preflop model input and materialized/retained current, old, or reference policy/value output must use width `K`; for `R=13`, this is 169. No staging tensor, store, cache, or compatibility adapter may expand a preflop dense model row to 1326.
- Every dense postflop model input and materialized/retained current, old, or reference policy/value output must use width `H`; for `R=13`, this is 1326. Keeping zeros for board-blocked hands is correct.
- RB posterior, lambda-return, advantage, and loss domains remain canonical exact-hand width `H` preflop and postflop. Preflop loss-time lookup uses the fixed `H -> K` quotient; it is indexing, not a stored `H`-wide model output.
- Normal mode may retain only selected scalar loss slices.
- Preserve poker summary anchors: create/refresh them only when fold dependencies or public street change, reference them rather than inline-copying them, and allocate no neutral folded summary when `A == P`.
- Add hard shape assertions and tests at every store/cache boundary, not only inside the model.

### 3. Partitioned rollout with last-consumer reach ownership

Refactor the current rollout scheduler to implement `alg:partitioned-queue-rollout` without replacing its working queue/RNG primitives.

- Normal and sequential RB have a fixed common per-GPU root set `(rank, local_index)` selected before any draw. Branched RB still samples exactly one rooted tree per GPU and applies the ordinary configured fixed `beta` rule to the root; do not solve adaptive branching in this task.
- Every random operation uses a disjoint logical counter address. Results must be invariant to queue order, physical subspan size, and GPU speed. Include rank in root addresses so ranks do not sample identical objects.
- `SafeAdmit` controls only concurrent live roots/frontier. It cannot change the fixed root count, discard a long path, retry a trajectory, or replace it.
- Preserve full-queue-first scheduling and the declared preorder/load/age partial-drain rule. Physical plans preserve insertion order and may split only into maximum-feasible stable subspans plus an exact tail.
- Reserve all logical children and action spans before physical forwards. A hard transaction failure stops the current run cleanly and preserves the last completed checkpoint; it never reduces multiplicity after its random draw.
- At player-node creation, append the natural partition input and stable `(m,d)` row. At stopped leaves, append terminal reward or exact RB reach plus immutable completion input/key. Release mutable public state and active reach/profile immediately after their last child/leaf/input consumer commits.
- Delete the production dependency on root-cache/path `recompute`, all-node reach memos, and `DecisionReachCache.from_reach_cache`. No target builder or learner may replay a discarded path. Remove the old mode/config once callers have migrated; do not leave a hidden fallback that violates the new lifetime proof.
- Avoid host synchronization in hot rollout paths. Existing no-host-sync tests and execution metrics must continue to pass.

### 4. Explicit, budgeted sealing

Implement `SealRolloutStorage` as an actual phase, not a claim of zero-copy transfer.

- Chunked occurrence/partition/action/leaf arenas are sources. Compute realized counts, transactionally admit simultaneous source capacity + exact destination capacity + bounded scan/copy scratch, and allocate each destination column exactly once.
- Sequential topology is decision-only CSR: `trajectoryPtr[G+1]`, `decisionPartition[D]`, `decisionRow[D]`, and `leafRow[G]`. Preserve chronological order using immutable root id and within-trajectory decision ordinal. Chance nodes do not get rows.
- Branched topology is one six-column int32 rooted tree with contiguous child spans. Player action spans must have exactly the same order as child spans. The root is row zero and receives no special multiplicity.
- Copy stable chunk spans into exact contiguous columns with one reusable bounded scratch span. Do not use repeated `torch.cat`, Python list growth proportional to total tensor payload, or allocator-driven destination growth.
- Validate every pointer, extent, row domain, and byte before releasing sources. Release source chunks only after synchronized last copy. Expose measured source, destination, scratch, and peak bytes.
- Force multi-chunk sealing in tests and prove destination `numel * element_size` equals the schema-derived logical payload while the admission ledger equals allocated capacity plus overlap/scratch.

### 5. Streaming target builders and immutable TrainingStorage

Implement all four algorithms in `backward_targets.tex`, using the existing math helpers where correct.

- Normal: bounded device trajectory batches and reverse segmented player-clock recurrence; retain one scalar `lambdaReturn` per decision and derive value/advantage in the loss.
- Sequential RB: call the sibling evaluator through `CompletionSpec` in bounded leaf microbatches; retain each needed `(trajectory, player)` posterior exactly once; store `posteriorRef[D_m]`; retain one exact-hand `lambdaReturn[D_m,H]`; immediately release completion extras, value rows, and running moments.
- Branched RB: keep it experimental and fixed-intensity, but implement the exact bounded frontier. Determine `F_star` before allocating slots; use contiguous children/actions; retain `valueHat[D_m,H]`, `childPosterior[A_m,H]`, and `advantageHat[A_m,H]`; derive parent posterior means later. Do not allocate or expose `qHat`, `W`, or `S` targets.
- Honor the zero-mass convention and lambda endpoints exactly. Add randomized small-PBG reference tests against direct formulas for `lambda=0,1`, zero posterior mass, single/multiple children, and root multiplicity.
- Finalization builds fp64 `log Z_m` and a nonuniform alias table only for branched strata; sequential strata use an implicit uniform sampler. Release topology, leaf reaches/inputs, completion/frontier arenas, validity flags, temporary log weights, and all scratch. Freeze only aligned natural model inputs/old rows, exact targets/posteriors, and sampling metadata.
- If the current learner needs old names/shapes temporarily, expose lightweight properties/views over the new store. Do not clone full tensors or keep legacy trees/reach caches alive.

### 6. Replace root work targeting with memory-only control

Evolve the existing `PolicyMemoryForecaster`, `SequentialWorkController`, config, checkpoint, and observability code into the two algorithms in `sequential_memory_control.tex`.

- Root width is a common per-GPU `G_k`, not a global count divided unevenly by world size. Introduce unambiguous configuration names for initial/minimum/maximum starts per GPU. Migrate repository configs and checkpoint metadata explicitly; never silently reinterpret an existing global field.
- Remove `target_decisions_per_gpu` and any decision/player-node target from root-width selection. Decision counts may still choose optimizer reuse/step count, but never `G_k`.
- Predict each phase separately: rollout, seal, targets, and optimization on the mandatory no-reference-cache path. Combine exact fixed bytes with lagged one-sided per-root count/capacity/workspace estimates. Include chunk slack and the sealing overlap. The feasible width is the integer monotone-search maximum whose worst phase fits `M_device - M_guard` on every rank; use `all_reduce(MIN)`.
- Cold start uses declared analytic worst-case per-root geometry and conservative singleton workspaces. For NLHE, use the repository's finite maximum player-node bound where helpful, not an OOM probe.
- Freeze width and keys before current draws. Positive surprises widen immediately. Decrease immediately when the safe width falls. Increase only after the configured successful streak and cap the growth fraction. Persist controller version, observations, residual envelope, limiting phase/field, predicted/actual phase peaks, guard, and current width atomically with training checkpoints.
- Remove rollout retry/reroll behavior (`MemoryPressureRetry`, retry nonces, discard-and-backoff loops, or equivalents). A transactional runtime failure records conservative evidence and performs a clean checkpointed stop without retrying the sampled iteration.
- Keep optional reference caches entirely out of the root safety proof; phase 2 will admit them transactionally. The uncached singleton learner path must be feasible.

## Tests and acceptance

Add focused tests rather than only modifying old assertions. At minimum cover:

- schema/accountant equality, reservation rollback, chunk slack, exact sealing, multi-chunk stable order, sequential CSR, branched spans, and no surviving mutable/reach objects;
- RNG invariance under queue/subspan changes and different rank ids;
- no reach replay and no all-node `[N,P,H]` memo in the production path;
- all target equations and deduplication contracts;
- preflop dense widths 169 and postflop dense widths 1326 for `R=13`, with RB exact domain 1326 in both;
- controller cold start, binary search, rank minimum, immediate decrease, delayed/capped increase, checkpoint resume, underestimated geometry, and clean no-reroll failure;
- memory ledgers for rollout, seal, targets, and the uncached optimization baseline.

Run the relevant CPU suite first using the repository environment. Then test on GPUs as required by `AGENTS.md`:

1. Run `nvidia-smi` before CUDA work. If sandboxing blocks it, request escalation. Select only idle GPUs and set device visibility explicitly.
2. Use `/home/pablob/.venvs/poker` or `./tools/poker-python`; never install into system Python. Do not modify the sibling evaluator.
3. Stage tiny one-GPU normal and sequential-RB smoke tests first, with conservative starts and a precomputed memory bound. Then run a tiny fixed-intensity branched smoke test. Only after those are clean, run a two-GPU test if two GPUs are idle.
4. Never intentionally provoke a CUDA OOM. Test rejection/failure paths with synthetic ledgers or tiny configured ceilings before allocation.
5. Record allocated/reserved phase peaks and exact retained bytes. After tests, run `nvidia-smi` again and verify no process or anomalous allocation remains from your work.

Run all directly affected tests and then the full feasible test suite. Compile the pseudocode documents only if you edit them, and send LaTeX artifacts to `/tmp`, not the repository.

## Completion and commit

Before committing:

- inspect the diff for accidental rewrites and unrelated user changes;
- confirm no production code can select reach replay, retry sampled geometry, or expand preflop dense rows to `H`;
- confirm old compatibility views do not duplicate tensor storage;
- update configuration examples and current implementation/audit documentation needed to make the new contracts discoverable;
- report exact CPU/GPU commands, GPU ids, phase peaks, and test results in the commit body or an audit note.

Create one git commit containing only this phase's changes, with a descriptive message such as `refactor(core): add sealed rollout storage and memory width control`. Do not leave the work merely staged. End with the commit hash, a concise summary, tests run, measured GPU memory, and any concrete phase-2 handoff notes.
