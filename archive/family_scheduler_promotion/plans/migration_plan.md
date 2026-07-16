# Queued multi-GPU rollout and learner experimentation plan

Date: 2026-07-15
Status: implementation in progress. Phase 0 and Phase 1 acceptance gates are
closed. The opt-in eager Phase 2 sequential family-FIFO executor now includes
byte-bounded resident admission, calibrated CPU family-prefix packing, a
CUDA-safe fixed-shape pack representation, exact logical queue age, unequal-rank
trajectory prefixes, recompute-minimal live reaches, logically addressed
non-exact completion, direct real-facade coverage of both tensor-native poker
completion branches, a named lower-launch logical RNG, and statistically
controlled benchmark seed grids. Phase 2 is not yet acceptance-complete: a
model-seeded calibrated Leduc probe has fewer model calls and safe memory but a
small rollout regression, the eager CUDA scheduler still has declared
dynamic-shape host synchronizations, the real sibling CUDA extension remains
blocked by the host toolkit, and no admissible poker GPU result exists. See the
newest restart checkpoint below before making changes.

## 0. Restart checkpoint

This section is the authoritative continuity record for restarting in a fresh
context. The design requirements in the rest of this document remain unchanged.

### 0.1 Current phase status

| Phase | Current state | Evidence / remaining gate |
|---|---|---|
| 0 — semantics, baselines, observability | Acceptance complete | Post-change full CPU regression produced 227 passed and 10 skipped; its sole sandbox-denied Gloo case passed when rerun with loopback permission. Global-mass and lossless-counter tests pass. |
| 1 — arena and queue primitives | Acceptance complete for the generic primitive layer | CPU/static tests and a tiny CUDA execution/transactionality test pass on GPU 2. Arena overflow and FIFO overflow are observed before size, accounting, queue depth, or occurrence status changes. GPU 2 returned to its original idle state. |
| 2 — sequential queued rollout | Baseline implementation in progress; substantially advanced but not promoted | `family_fifo_v1` implements persistent family queues, byte-transactional resident admission, eager chance closure, one profile per trajectory, named logical-address private/action/chance/completion draws, normal scalar targets, sequential-RB targets, exact logical queue age, explicit unequal-rank address prefixes, recompute-minimal live reaches, calibrated prefix packing, and a CUDA-safe masked pack representation. Real sibling CPU gates cover collision-free queued profiles, old-policy identity, keyed normal settlement, and mixed pre-terminal/fold-terminal tensor completion. The corrected model-seeded Leduc screen reduces median calls 17→16; paired rollout ratios are 1.059 normal and 1.039 RB, with safe measured memory. Fixed-shape arena routing removed 56 implicit syncs from the diagnostic, but owner/family routing still reports 43–47 per iteration. Remaining: eliminate or explicitly redesign that hot-sync boundary, provide a compatible toolkit for the real sibling CUDA smoke, and obtain admissible poker GPU law/memory/performance evidence. |
| 3 — branched queued rollout | Not started | Must follow the estimator-correct Phase 2 implementation. |
| 4 — mixed-street poker executor | Not started | Current poker execution still re-splits postflop work by street. |
| 5 — target compaction/global-`Z` overlap | Not started | Existing streamed target code remains the baseline. |
| 6 — distributed learner rewrite | Not started | The active learner remains the named rank-local-mean baseline; the semantic global-mass helpers do not yet change training. |
| 7 — compiled/alternative executors | Not started | Do not begin before the eager baseline passes. |
| 8 — integration and documentation | Not started | Executor configuration and final run metadata remain incomplete. |

No admissible performance result has been produced from the queued executor.
One focused synthetic test proves that a larger resident pool reduces model-call
count for identical logically addressed work, but no timed poker or end-to-end
benchmark has been run.

### 0.2 Implemented files and behavior

The following new files are currently untracked in the dirty worktree. They are
real implementation work and must not be discarded when restarting:

- `pbg/core/distributed.py` defines the exact weighted decision numerator,
  global decision-mass objective, rank/family scale `Z[g,m]/Z`, and an explicitly
  named equal-rank diagnostic baseline.
- `pbg/core/observability.py` defines lossless aggregate execution metrics and
  stable pre-queue baseline identities. It replaces bounded diagnostic history
  with additive totals, high-water marks, family/phase strata, and exact batch
  histograms.
- `pbg/core/queued.py` defines the generic Phase 1 foundation: transactional byte
  accounting, mode-selectable chunked SoA occurrence storage, bounded
  device-resident FIFO rings, monotone occurrence states, reusable staging
  slabs, logical-address RNG keys and categorical draws, and a stable
  family-only padded-work packer.
- `pbg/core/rollout.py` now retains `_sample_trees_sequential` as the unchanged
  `wavefront_v1` baseline and adds opt-in `family_fifo_v1`. The queued path owns
  persistent per-family `DeviceFifoRing`s, a mode-minimal occurrence arena,
  stable bounded root admission, eager chance-only closure, logical trajectory
  and path addresses, keyed private/action/chance draws, family priority, and
  queue metrics. It returns the existing `TreeBatch`, `ReachCache`, and
  `PolicyCache` contracts.
- `pbg/core/tree.py` stably restores parent-major edge order at finalization, so
  cross-depth execution still emits valid CSR topology.
- `pbg/core/config.py` exposes `rollout.executor`, `queue_batch_rows`,
  `queue_admission_size`, and `arena_chunk_nodes`; the queued executor is
  currently rejected for branched-RB. Run metadata now records the selected
  rollout executor rather than always naming the baseline.
- `pbg/core/normal.py` dispatches through the selected sequential executor and
  uses logical completion keys for queued normal terminal settlement when the
  backend supports them.
- Synthetic and poker rollout backends now expose family routing and keyed
  sampling boundaries. Poker passes logical profile seeds through the existing
  sibling facade contract; the sibling repository was not modified. Poker
  action, chance-card, and terminal-runout draws are locally keyed.
- `pbg/games/poker/mc.py` adds `PokerGPURejectionSampler.sample_seeded`, a thin
  caller-seed form of the existing facade invocation.
- `tests/test_q00_queued_foundations.py` covers global-mass versus equal-rank
  gradients, exact counter aggregation beyond the old deque bound, transactional
  arena/queue overflow, duplicate scheduling, staging bytes, schedule/placement
  independent RNG, mixed-stage family packing, and static hot-path host-transfer
  checks. It now also has a tiny CUDA arena/queue/RNG/packer execution and
  transactional-overflow test.
- `tests/test_q02_sequential_queued_rollout.py` covers logical trajectory
  invariance under very different queue/admission schedules, model-call
  reduction from a larger resident pool, all reach-storage modes, old-output
  identity, scalar normal targets, all-type sequential-RB targets, loud
  overflow, static scheduler host-transfer checks, and a tiny CUDA RB target
  build.
- `tests/test_p13a_gpu_rollout.py` contains a queued poker CPU smoke that reaches
  postflop through eager chance closure. `tests/test_p5_completion.py` verifies
  that caller-owned logical seeds reach the sibling profile facade unchanged.

Existing dirty files also contain the Phase 0 integration:

- `CONTRACTS.md` and `NOTATION.md` now state the global decision-mass objective,
  flat `SUM` gradient law, unique-root/logical-address requirements, named
  baselines, and occurrence lifecycle.
- `pbg/games/poker/rollout_backend.py` owns `ExecutionMetrics`, records calibrated
  packs and actual model-family invocations, and no longer relies on a bounded
  policy-metric deque.
- `pbg/core/training.py` clears/snapshots those counters and emits the legacy
  rollout/learner implementation identities in run metadata.

The default remains `wavefront_v1`; queued rollout changes behavior only when
`rollout.executor: family_fifo_v1` is selected. Learner gradients remain the
`rank_local_mean_ddp_v1` baseline. In particular, adding
`pbg/core/distributed.py` does not mean the Phase 6 objective has been wired into
optimization.

The Phase 2 implementation is intentionally still an eager baseline. It uses a
fixed logical queue batch row limit and lets the current poker backend perform
its existing safe physical splitting. It does not yet use `FamilyPacker` to
select queue prefixes, does not yet eliminate poker's street re-split or host
scalar planner, and retains a full working reach table during queued rollout
even when the returned cache is `recompute`. Those are explicit remaining
memory/execution gates, not completed optimizations.

### 0.3 Verification already performed

The pre-change full CPU baseline was:

```text
/home/pablob/.venvs/poker/bin/python -m pytest -q
214 passed, 10 skipped, 1 failed in 65.86s
```

The sole failure was
`tests/test_p13b_ddp.py::test_distributed_metric_summaries_allow_rank_local_optional_metrics`:
Gloo could not open its loopback transport inside the filesystem/network
sandbox (`Operation not permitted`). No code assertion failed in that run.

Focused post-change verification passed:

```text
/home/pablob/.venvs/poker/bin/python -m pytest -q tests/test_q00_queued_foundations.py
9 passed in 1.43s

/home/pablob/.venvs/poker/bin/python -m pytest -q \
  tests/test_p0_config.py tests/test_p8_tree.py \
  tests/test_aud26_no_host_sync.py tests/test_p13a_gpu_rollout.py \
  -k 'not cuda'
45 passed, 5 deselected in 4.54s

git diff --check
# clean
```

The Phase 0/1 closure and initial Phase 2 checks performed in this continuation
were:

```text
/home/pablob/.venvs/poker/bin/python -m pytest -q
227 passed, 10 skipped, 1 sandbox transport failure in 67.36s

# Rerun outside the loopback-restricted sandbox:
/home/pablob/.venvs/poker/bin/python -m pytest -q \
  tests/test_p13b_ddp.py::test_distributed_metric_summaries_allow_rank_local_optional_metrics
1 passed in 3.01s

# Phase 1 CUDA primitive test, CUDA_VISIBLE_DEVICES=2:
tests/test_q00_queued_foundations.py::test_cuda_queue_arena_rng_and_packer_execute_transactionally
1 passed in 1.64s

# Phase 2 focused CPU suite before the final small patches:
tests/test_q02_sequential_queued_rollout.py -k 'not cuda'
9 passed, 1 deselected in 4.29s

# Phase 2 tiny CUDA target build, CUDA_VISIBLE_DEVICES=2:
tests/test_q02_sequential_queued_rollout.py::test_queued_sequential_rb_builds_targets_on_cuda
1 passed in 2.05s

# Additional focused regression groups:
26 passed, 1 skipped in 24.21s
29 passed, 2 deselected in 4.45s
22 passed in 2.53s
```

Both CUDA runs were preceded and followed by `nvidia-smi`. GPU 2 was selected
because it had only 19 MiB of display overhead and 0% utilization; afterward it
again had 19 MiB and 0% utilization. GPUs 0 and 1 had other active compute jobs
and were not touched.

A full suite has **not** been rerun after the Phase 2 patches. The latest full
suite evidence therefore closes Phase 0/1 only; the next context must run the
focused Phase 2 set and then a new full regression before promoting Phase 2.

### 0.4 Exact restart sequence

In the next fresh context:

1. Read `AGENTS.md`, this checkpoint, and both pseudocode files completely.
   Reinspect the current worktree; do not assume tracked/untracked state from a
   prior summary.
2. Preserve all unrelated user edits. In particular, do not reset or replace the
   large existing changes in `pbg/core/training.py` or
   `pbg/games/poker/rollout_backend.py`.
3. Inspect the Phase 2 diff and run at least:
   `tests/test_q00_queued_foundations.py`,
   `tests/test_q02_sequential_queued_rollout.py`,
   `tests/test_p0_config.py`, `tests/test_c08_device_rollout.py`,
   `tests/test_p5_completion.py`, `tests/test_p8_rollout.py`, and
   `tests/test_p13a_gpu_rollout.py -k 'not cuda'`. Then run the full CPU suite.
   If Gloo again fails with sandbox transport denial, rerun only that test with
   loopback permission rather than changing code.
4. Review the new poker CPU smoke carefully and strengthen its model-call/family
   assertions if needed; it currently proves topology/chance closure and
   postflop reach, while the synthetic test carries the explicit call-reduction
   assertion.
5. Complete `SafeAdmit`: precompute retained plus maximum in-flight bytes, admit
   by that transaction rather than a fixed row count, and avoid allocating the
   full working reach table in `recompute` mode. Add unequal per-rank calibrated
   quotas and explicit prefix sums for global trajectory addresses; the current
   automatic `rank * local_n_trees` address is correct only for equal quotas.
6. Wire `FamilyPacker`/full thresholds into queue selection using a CUDA-safe
   representation that does not require a dynamic boolean-index size read on
   the host. Preserve the upstream preflop priority and pairwise-incomparable
   postflop layer. Add exact queue age and forced-drain metrics.
7. Add a real sibling-evaluator poker queued smoke, CPU first and then a tiny
   CUDA run after `nvidia-smi`. Verify logical seeds, complete collision-free
   profiles, old-policy identity, no pending chance row, memory peak, and device
   cleanup. The existing fake-facade test only verifies seed plumbing.
8. Add paired/statistical wavefront-versus-queued law gates for normal and
   sequential RB, including normal keyed runouts and non-exact RB completion.
   RB completion seed/address plumbing belongs at the Phase 2/5 boundary and is
   not complete merely because exact synthetic completion passes.
9. Do not begin Phase 3 until the preceding Phase 2 law, real-poker, memory,
   overflow, no-hot-sync, and full-regression gates pass. Once they do, update
   this checkpoint and then begin the one-root branched queue.

Known implementation boundaries to keep explicit during continuation:

- `DeviceFifoRing` maintains device metadata plus a deterministic host-count
  mirror; Phase 2 must not turn that mirror into per-call device scalar reads.
- `FamilyPacker` is family-curried and sees stage only through row cost. Do not
  add street routing to it.
- Semantic validation helpers in `pbg/core/distributed.py` are not learner
  hot-path kernels. Phase 6 must use device-safe assertions/reductions rather
  than copying their eager validation branches blindly into CUDA loops.
- Exact observability APIs exist, but evaluator, completion, collective, queue
  age, and rank-wait counters still need to be attached at their actual Phase
  2/5/6 execution boundaries.
- The queued scheduler currently records queue depth with age zero; do not
  interpret that placeholder as an implemented age metric.
- Keyed poker profile sampling uses the existing evaluator facade's per-row
  `seed` tensor through `sample_seeded`. Custom test doubles without that method
  take a deterministic compatibility fallback; production correctness must be
  validated with the real facade before Phase 2 promotion.
- The sibling poker evaluator remains read-only and every branched posterior draw
  must remain a complete collision-free joint profile.
- `beta` remains opaque and fixed. Memory pressure may delay work but may not
  change rounded offspring counts.

### 0.5 Continuation checkpoint — Phase 2 memory, packing, and real-facade work

This subsection is newer than §0.1--§0.4 and is the first restart source for the
next context. It records work completed after the earlier checkpoint; nothing in
the sibling evaluator repository was modified.

Implemented in this continuation:

- `DeviceTreeBuffer` no longer sorts edges during finalization. Expansion records
  stable per-parent child ordinals and finalization performs an `O(C)` direct CSR
  scatter. This preserves queued cross-depth parent-major topology while
  satisfying the labeled hot-path no-sort/no-host-transfer contract.
- `SequentialAdmissionPlan` reserves retained bytes plus the largest in-flight
  transaction before any profile draw, admits only the largest resident prefix
  that fits, releases resident allowance only at a trajectory's unique stopped
  leaf, and fails before sampling if one start cannot fit. The queued scheduler
  now uses this byte decision instead of treating `queue_admission_size` as an
  unconditional chunk.
- `rollout.queue_memory_bytes` exposes an optional explicit admission limit.
  Backends may provide calibrated `queued_admission_byte_terms`; otherwise the
  core constructs a conservative core-owned structural/in-flight plan. Exact
  reserved-byte high water is emitted through execution metrics.
- Sequential `reach_storage="recompute"` now keeps one transient current reach
  row per trajectory and overwrites it as that trajectory advances; it no longer
  allocates `[n_trees * node_budget, P, X]` working reach storage.
- `rollout.trajectory_rank_counts` supplies explicit unequal-rank quotas and
  prefix sums for collision-free global trajectory addresses, with runtime
  validation that the local quota equals `rollout.n_trees`. The mutually
  exclusive `trajectory_start_index` override is now part of strict config.
- `DeviceFifoRing` maintains exact logical enqueue-time runs alongside its host
  count mirror. Queue oldest age, full-family load/age tie-breaking, forced
  drains, and per-family forced-drain rows are now real metrics rather than age
  zero placeholders.
- `FamilyPacker.pack_masked` returns fixed-shape candidate handles, a feasible
  prefix mask, and device scalar count/width/workspace/fill without dynamic
  boolean indexing. CPU queued scheduling consumes calibrated maximum-feasible
  prefixes and full thresholds. CUDA synthetic execution consumes the masked
  representation under a backend-declared static-safe calibration. Poker CUDA
  deliberately retains its existing safe physical splitter until a masked
  direct executor can consume the device prefix without a host count read.
- A real installed `poker_gpu_evaluator.pbg` CPU queued test now runs two very
  different queue/admission schedules. It verifies identical caller-addressed
  complete profiles, collision freedom, identical public paths, no pending
  chance occurrence, exact old value/log-probability identity after replay, and
  identical logically canonicalized normal actor/event/advantage/value targets,
  including keyed runouts and real sibling showdown settlement.

Verification performed in this continuation:

```text
# Initial focused post-checkpoint baseline:
75 passed, 5 deselected in 28.16s

# Initial full CPU suite before the continuation patches:
237 passed, 12 skipped, 2 failed in 69.75s
# One failure was the known sandbox-only Gloo denial. Approved loopback rerun:
1 passed in 3.00s
# The other was the real DeviceTreeBuffer argsort regression, now repaired.

# CSR/no-sort targeted regression after repair:
23 passed, 2 deselected in 4.38s

# Phase 0/1/2 focused groups during implementation:
51 passed, 3 deselected in 5.40s
48 passed, 4 deselected in 6.78s
49 passed, 4 deselected in 7.03s
14 passed, 1 deselected in 4.57s

# Real sibling evaluator, queued poker CPU schedule/normal-settlement gate:
1 passed in 2.68s

# Final combined non-CUDA closeout group after all continuation patches:
82 passed, 6 deselected in 8.18s

git diff --check
# clean
```

No CUDA command was launched in this continuation: the explicit goal budget was
reached before the required `nvidia-smi`/idle-device selection boundary. The
new CUDA-safe pack primitive and real-sibling CUDA smoke therefore remain
unexecuted. A new full CPU suite also remains required after these patches.

Exact next sequence:

1. Run `git diff --check`, the complete focused Phase 2 group, and then the full
   CPU suite. The repaired labeled-hot-path test must remain green. If Gloo is
   sandbox-denied, rerun only its test with loopback permission as before.
2. Inspect the real-sibling CPU test's canonical decision ordering and keep the
   logical comparison: raw policy/dataset row order legitimately changes with
   cross-depth scheduling and is not itself part of the estimator law.
3. Run `nvidia-smi`, select only an idle GPU, estimate the tiny allocations, and
   execute the Phase 1 primitive CUDA test, the sequential-RB CUDA target test,
   and `test_real_sibling_queued_poker_tiny_cuda`. Recheck the GPU afterward.
4. Add the non-exact sequential-RB completion seed/address boundary and paired or
   statistical wavefront-versus-queued target-law gate. The existing exact
   synthetic completion test is insufficient for this acceptance item.
5. Generalize the masked pack consumer for poker CUDA so device-computed prefix
   length never becomes a host scalar. Do not claim CUDA packer integration from
   the static-safe synthetic path alone, and do not begin Phase 3 yet.
6. Measure actual allocated/reserved peaks and family call/fill behavior against
   the admission plan; retain the 25% safety margin. Only after the full Phase 2
   law, real-poker, no-hot-sync, overflow, memory, CUDA-cleanup, and full-suite
   gates pass should §0.1 promote Phase 2 and Phase 3 begin.

### 0.6 Continuation checkpoint — keyed completion, poker masked CUDA quantum, and measurement

This subsection supersedes §0.5 as the first restart source. Work stopped at the
explicit 350k goal-budget boundary. Phase 2 remains intentionally unpromoted and
Phase 3 has not begun.

Implemented in this continuation:

- `sequential_leaf_logical_rng_keys` derives completion addresses from the
  iteration, collision-free global trajectory prefix, recursive abstract leaf
  path, draw ordinal, and completion-purpose namespace. It reconstructs
  sequential leaf paths without a CUDA scalar read or one device launch per
  stored node. Normal keyed runouts now use the same helper.
- The optional `KeyedMCCompletion` boundary is wired into queued sequential RB.
  `SyntheticMCCompletion`, `PokerGPUCompletion`, and the tensor-native poker
  completion adapter accept one caller-owned int64 stream seed per leaf.
  Synthetic MC derives per-sample keys; poker passes row seeds unchanged to the
  sibling completion facade and seeded full-joint-profile sampler. Completion
  microbatch boundaries no longer affect queued sequential-RB draws.
- A production-shaped non-exact synthetic gate compares queue rows 1 versus 7,
  resident admissions 1 versus 12, completion batches 1 versus 12, and unrelated
  mutable generator seeds. Public paths, completion keys, old values, posterior
  targets, advantages, values, child metadata, and final streamed RB targets
  agree after canonical logical ordering. A separate iteration-builder test
  fails if queued RB attempts to call mutable completion.
- Fixed-seed statistical gates now compare `wavefront_v1` and
  `family_fifo_v1` for both normal and sequential RB over 1,024 Kuhn trajectory
  starts. Empirical decision/posterior distributions and first/second target
  moments remain within the declared fixed tolerances.
- Poker exposes `queued_static_pack_capacity`, a conservative family-only CUDA
  quantum proved against the worst allowed history and, for every `post_A`, the
  largest flop width. The scheduler limits candidates to this host-known
  quantum, consumes `FamilyPacker.pack_masked`, and asserts mask feasibility on
  device; the device-computed selected count is never copied to the host. This
  does not perform the Phase 4 mixed-street physical-forward rewrite: the
  existing poker backend may still physically split by street inside its call.
- Execution counters are now attached at actual completion-batch, compact-target,
  poker joint-profile, poker completion, and sampled-showdown boundaries.
  Tests cover completion rows/batches, target bytes, and real-facade evaluator
  rows/calls.
- Benchmark JSON now records `rollout_executor` globally and per mode; the CLI
  accepts `--rollout-executor`, and baseline comparison rejects an executor
  mismatch. This prevents queued results from being compared accidentally to a
  differently identified wavefront artifact.

Verification in this continuation:

```text
# Baseline before this continuation's patches:
83 passed, 6 deselected in 28.01s
247 passed, 13 skipped, 1 known sandbox-only Gloo denial in 72.64s
approved Gloo-only rerun: 1 passed in 3.00s

# Final pre-metrics focused Phase 2 closeout:
85 passed, 8 deselected in 28.61s

# Full CPU suite after keyed completion/masked packing and before the final
# observability/benchmark-metadata additions:
250 passed, 14 skipped, 1 known sandbox-only Gloo denial in 70.04s
approved Gloo-only rerun: 1 passed in 3.00s

# Focused checks after observability additions:
63 passed, 7 deselected in 10.41s
2 passed in 2.71s

# Statistical wavefront-versus-queued normal/RB law gates:
2 passed in 7.06s

# Benchmark metadata/performance-gate plus queued Phase 2 tests:
21 passed, 1 deselected in 10.15s

# CUDA_VISIBLE_DEVICES=2, after nvidia-smi selected GPU 2 at 19 MiB / 0%:
test_poker_queued_cuda_consumes_masked_static_family_quanta
test_queued_sequential_rb_builds_targets_on_cuda
test_cuda_queue_arena_rng_and_packer_execute_transactionally
3 passed in 2.58s

git diff --check
# clean at the review point before the final documentation update
```

The real sibling CUDA smoke was attempted only after GPU 2 was confirmed idle.
The two core CUDA gates passed, but the sibling smoke cannot load its extension
on this host: the shared virtualenv contains Ninja, while the host has no CUDA
toolkit (`nvcc` is absent and `CUDA_HOME` cannot be set to an installed toolkit).
No sibling source was modified. This is an environment gate, not a test skip to
reinterpret as success. GPU 2 returned to 19 MiB and 0% utilization after every
CUDA run and benchmark.

Two same-work tiny synthetic GPU artifacts were recorded:

- `benchmarks/results/queued_phase2_kuhn_wavefront_gpu2_2026-07-15.json`;
- `benchmarks/results/queued_phase2_kuhn_family_fifo_gpu2_2026-07-15.json`.

Both used GPU 2, 128 trajectory starts, node budget 16, one warmup and three
measured repetitions for normal and sequential RB. The queued and wavefront
rollouts had the same median four model calls/eight distinct model rows on this
degenerate Kuhn probe. Queued normal was slower (0.208 s versus 0.152 s median),
while queued RB was effectively tied/slightly faster (0.600 s versus 0.605 s).
Both had a 23,068,672-byte median rollout reserved peak; queued added only 2,048
bytes to median allocated peak (17,965,056 versus 17,963,008 bytes). The queued
admission planner reported a conservative 289,304-byte structural reservation.
These results validate measurement plumbing and memory cleanup only; they are
not admissible poker performance evidence and do not justify Phase 2 promotion.

Exact next sequence:

1. Run `git diff --check`, the full focused Phase 2 group, and the full CPU suite
   once more because the execution-counter and benchmark-metadata patches were
   added after the latest full-suite run. Rerun only the known Gloo test with
   loopback permission if the sandbox denies it.
2. Re-run the three tiny CUDA gates above after `nvidia-smi`; verify GPU cleanup.
   The real sibling CUDA smoke still requires an installed compatible CUDA
   toolkit/nvcc. Do not modify the read-only sibling repository to work around
   the host environment.
3. Add a tensor-native poker keyed-completion integration gate that exercises
   both pre-terminal sibling completion and public fold-terminal seeded joint
   profiles, preferably with the real CPU facade first. The generic poker facade
   seed plumbing and synthetic end-to-end keyed law are covered, but this mixed
   tensor-native boundary deserves direct coverage.
4. Run a measured Leduc queued-versus-wavefront probe and then a tiny real-poker
   CPU probe. Record call histograms, forced drains, padded work, admission bytes,
   actual allocated/reserved peaks, and the 25% safety margin. Do not claim an
   end-to-end speedup from the Kuhn plumbing probe.
5. Audit the poker CUDA physical planner separately from queue-prefix selection.
   The queue now consumes a masked fixed quantum without a host prefix count,
   but `plan_policy_batches` and `_policy_forward_direct` still contain the
   explicitly deferred host/street splitting owned by Phase 4.
6. Only after the real-poker CUDA/environment gate, final full regression,
   memory margin, no-hot-sync boundary, and measured Phase 2 acceptance evidence
   pass should §0.1 promote Phase 2 and implementation proceed to the one-root
   branched queue in Phase 3.

### 0.7 Continuation checkpoint — direct poker completion, explicit benchmark targeting, and measured overhead

This subsection supersedes §0.6 as the first restart source. Phase 2 remains
unpromoted and Phase 3 has not begun.

Implemented in this continuation:

- A direct real-`poker_gpu_evaluator.pbg` CPU integration gate now sends one
  pre-terminal all-in row and one public fold-terminal row through
  `_TensorPokerGPUCompletion.complete_keyed` in the same batch. It verifies
  bitwise replay under caller-owned logical keys, normalized per-seat posterior
  mass, exact public fold payoff, sample counts, facade metadata, and exact
  evaluator call/row accounting. This closes the mixed tensor-native boundary
  left open in §0.6 without modifying the sibling repository.
- The benchmark API and CLI accept an explicit Torch device and completion
  backend. Artifacts now record `completion_backend` globally and per mode, and
  baseline comparison rejects a backend mismatch just as it rejects a rollout
  executor mismatch. This permits an explicit CPU sibling probe even on a host
  with visible GPUs and makes evaluator-backend identity auditable.
- Four new measured artifacts were recorded:
  `queued_phase2_leduc_wavefront_gpu2_2026-07-15.json`,
  `queued_phase2_leduc_family_fifo_gpu2_2026-07-15.json`,
  `queued_phase2_nlhe_hu_cpu_wavefront_2026-07-15.json`, and
  `queued_phase2_nlhe_hu_cpu_family_fifo_2026-07-15.json`, all under
  `benchmarks/results/`.

Verification performed in this continuation:

```text
# Required focused Phase 2 CPU group:
87 passed, 8 deselected in 34.28s

# Direct mixed tensor-native poker completion plus facade completion tests:
6 passed in 1.92s

# Benchmark gate/config checks after explicit device/backend support:
21 passed in 2.47s

# Final full CPU regression after all code and test changes:
255 passed, 14 skipped, 1 known sandbox-only Gloo denial in 76.57s
approved Gloo-only rerun: 1 passed in 3.04s

# CUDA_VISIBLE_DEVICES=2 after GPU 2 was observed at 19 MiB / 0%:
test_cuda_queue_arena_rng_and_packer_execute_transactionally
test_queued_sequential_rb_builds_targets_on_cuda
test_poker_queued_cuda_consumes_masked_static_family_quanta
3 passed in 2.57s

git diff --check
# clean at the review points in this continuation
```

GPU 2 returned to 19 MiB and 0% utilization after the core CUDA gates, both
sibling-extension attempts, and both Leduc benchmarks. GPUs 0 and 1 retained
other users' compute processes and were not touched.

The Leduc probe used GPU 2, 128 sequential starts, node budget 64, 64 completion
samples, one warmup, and five measured repetitions for normal and sequential RB.
It provides the first non-degenerate measured queue result:

- normal median model calls fell from 10 to 8, but rollout time increased from
  0.310 s to 0.379 s and complete iteration time from 0.348 s to 0.419 s;
- sequential-RB median model calls also fell from 10 to 8, but rollout time
  increased from 0.310 s to 0.376 s and complete iteration time from 1.027 s to
  1.121 s;
- queued allocated rollout peaks were within roughly 1% of wavefront
  (23,277,568 versus 23,048,192 bytes for normal and 23,277,056 versus
  23,399,936 bytes for RB), while every reserved peak was 29,360,128 bytes;
- the queued admission planner reserved 1,441,304 structural/in-flight bytes.

Thus the queued executor reduced calls and stayed comfortably inside the 25%
measured-peak memory margin, but failed the Phase 2 neutral-or-better rollout
time criterion on this workload. Fewer model calls are not sufficient evidence
for promotion; eager scheduler overhead is now a measured optimization target.

The tiny real-poker CPU probe used the real sibling CPU facade, four heads-up 5bb
starts, node budget 64, four completion samples, one warmup, and five measured
repetitions. It records real evaluator calls/rows, pre/post family call
histograms, forced drains, padded tokens/pairs, completion batches/rows, target
bytes, and a 2,989,824-byte queued admission reservation. It is plumbing and
CPU execution evidence only. Wavefront and queued currently use different RNG
mechanisms, their realized decisions and postflop exposure differed, and the
CPU allocator does not expose CUDA allocated/reserved peaks; no relative speed
claim is admissible from these two artifacts.

The real sibling CUDA smoke was rerun conservatively on idle GPU 2. The shared
venv contains Ninja 1.13, but its executable required an explicit venv `PATH`.
With that corrected, extension loading reaches the actual blocker:
`CUDA_HOME` is unset and `nvcc` is absent. The test fails before evaluator work
with `OSError: CUDA_HOME environment variable is not set`. This is an external
host-toolkit gate, not a code pass or a reason to edit the read-only sibling.

Exact next sequence:

1. Do not begin Phase 3. First profile the eager queued scheduler on the Leduc
   artifact with phase-level host/kernel timing and identify why eight calls are
   slower than ten wavefront calls. Prioritize removing Python/device launch
   overhead without introducing a steady-state scalar read or weakening exact
   FIFO/status/accounting laws.
2. Add a statistically controlled same-law comparison harness across multiple
   logical seeds, or extend logical-address RNG to a comparison-only wavefront
   oracle, so executor performance can be conditioned on matched realized work.
   Do not treat the unpaired real-poker CPU artifacts as a speed comparison.
3. When a compatible CUDA toolkit is made available by the host, put the venv
   `bin` directory on `PATH`, rerun the real sibling CUDA smoke on an idle GPU,
   and record actual poker allocated/reserved peaks and cleanup. Do not install
   system CUDA or modify the sibling repository as a workaround.
4. Re-run the Leduc and then real-poker GPU probes after scheduler overhead is
   reduced. Require full call/fill/forced-drain histograms, measured peak plus
   25% margin, no-hot-sync evidence, and neutral-or-better complete rollout time.
5. Only after the real sibling CUDA gate, matched-work law/performance evidence,
   memory margin, and all Phase 2 acceptance criteria pass should §0.1 promote
   Phase 2 and implementation proceed to the one-root branched queue.

### 0.8 Continuation checkpoint — lower-launch logical RNG and controlled queue calibration

This subsection supersedes §0.7 as the first restart source. Phase 2 remains
unpromoted and Phase 3 has not begun.

Implemented in this continuation:

- `ChunkedOccurrenceArena.gather_many` and `scatter_many` share one device
  chunk-index plan across status/address fields. Status transitions reuse the
  same plan for validation and mutation. The queued sequential executor batches
  trajectory/path reads and writes through these APIs, preserving chunked
  allocation, exact status transitions, and the no-host-scalar contract.
- Logical stream generation is explicitly named `counter_hash_v2`. It combines
  `(seed, iteration, global trajectory, abstract path, draw ordinal, purpose)`
  with independent odd counter multipliers and one 63-bit avalanche round.
  Scalar iteration/purpose namespaces are prehashed on the host because they are
  configuration scalars, not device scheduler state. A structured grid gate
  checks uniqueness across iterations, every purpose, multiple draw ordinals,
  trajectories, and paths; all existing schedule/placement and target-law gates
  remain green. Training and benchmark metadata record the RNG version.
- `logical_categorical` remains the fully checked public primitive. Poker and
  synthetic backends now call
  `logical_categorical_from_valid_probability` only after their adapters or
  exact game laws have already produced valid nonnegative rows, avoiding
  duplicate finite/nonnegative/positive-mass reductions in every queued event
  draw. Shape, device, key dtype, and optional mass-shape checks remain.
- Benchmark repetitions now use a recorded common seed grid and a fixed
  scientific iteration. Previously, wavefront reset to the same mutable RNG
  state on every repetition while queued work varied by iteration; that made
  timing dispersion and cross-executor comparisons misleading. Artifacts now
  record training seed, repeat seeds, fixed profile iteration, logical RNG
  version, completion backend, queue batch/admission rows, and arena chunk size.
  Baseline comparison rejects mismatches in these fields.
- The benchmark CLI exposes `--device`, `--completion-backend`,
  `--queue-batch-rows`, `--queue-admission-size`, and
  `--arena-chunk-nodes`, so resource placement and calibrated executor shapes
  are explicit rather than hidden runtime defaults.

Focused and final verification:

```text
# Arena batching and queued sequential CPU suites during implementation:
30 passed, 2 deselected in 10.14s
31 passed, 2 deselected in 9.94s

# Final queued foundations/sequential/poker non-CUDA group:
58 passed, 7 deselected in 12.46s

# Benchmark/config gates after metadata and calibration controls:
22 passed in 2.60s

# Tiny CUDA queue/RNG/target/poker masked-quantum gates on GPU 2:
3 passed in 2.52s

# Final full CPU regression:
258 passed, 14 skipped, 1 known sandbox-only Gloo denial in 74.06s
approved Gloo-only rerun: 1 passed in 3.00s

git diff --check
# clean
```

The RNG microprobe on GPU 2 used 128 rows and 3,000 timed invocations after 100
warmups. The old five-round key path took about 657 microseconds per call; the
first three-round formulation took 295 microseconds; the final one-round
counter formulation took about 125 microseconds in the selection probe.
Replacing comparison/sum categorical selection with batched `searchsorted`
reduced that microprobe from about 237 to 80 microseconds before redundant
backend validation was removed. These are launch-cost diagnostics, not
standalone throughput claims.

The controlled Leduc screening sequence used the fixed repeat seeds
`[104740, 209469, 314198, 418927, 523656]`, GPU 2, node budget 64, one warmup,
five measured repetitions, and both normal and sequential RB:

- At 128 starts, the optimized queue reduced median calls from 10 to 8 in
  normal and 9 to 8 in RB. Paired median raw rollout ratios were 1.061 normal
  and 1.075 RB. Reserved peaks were identical at 29,360,128 bytes; queued
  allocated peaks were within 1% of wavefront.
- At 512 starts with the implicit 256-row default, queue calls increased to 14
  versus wavefront's 10 and paired rollout ratios regressed to about 1.154 and
  1.155. This is direct evidence that the default cap is not a calibration.
- Explicit 512-row batches/admission reduced queued calls to 9 and paired
  rollout ratios to 1.071 normal and 1.075 RB.
- Keeping that 512-row quantum while using 4,096-node arena chunks produced the
  best eager result: median rollout was 1.109 s queued versus 1.088 s wavefront
  in normal and 1.107 s versus 1.071 s in RB. Paired median ratios were 1.013
  normal and 1.034 RB; one RB seed was faster than wavefront. Complete median
  iteration time was 1.173 s versus 1.139 s in normal and 3.912 s versus 3.898 s
  in RB. Calls fell 10→9 in both modes. Reserved peaks were identical at
  39,845,888 bytes; allocated peaks differed by less than 1% (27,380,736 versus
  27,298,304 bytes normal and 27,383,808 versus 27,224,576 bytes RB). The
  planner's queued structural/in-flight high water was 5,765,144 bytes.

The calibrated 512/4,096 result is treated as neutral on this exact synthetic
screen and satisfies the measured-peak plus 25% memory condition. It does not
promote Phase 2 by itself: the family models are tiny, the two RNG algorithms
are statistically controlled rather than pathwise identical, and poker has
very different model/evaluator costs.

Refreshed real-sibling CPU artifacts use the same fixed seed-grid methodology:

- `queued_phase2_nlhe_hu_cpu_wavefront_seedgrid_2026-07-15.json`;
- `queued_phase2_nlhe_hu_cpu_family_fifo_rng_v2_seedgrid_2026-07-15.json`.

They verify real CPU evaluator/completion execution, family call histograms,
forced drains, padded work, target bytes, and the 2,989,824-byte queued
admission plan. Four trajectories remain far too topology-sensitive for a
relative speed claim: individual paired ratios span both large wins and large
losses because the executors implement different statistically valid draw
maps. Keep these as plumbing artifacts only.

GPU 2 was checked before CUDA work and returned after every test/profile/probe
to its original 19 MiB display overhead and 0% utilization. GPUs 0 and 1 had
other compute processes and were not touched. The real sibling CUDA blocker is
unchanged: Ninja is installed and works when the venv `bin` directory is on
`PATH`, but this host has no `nvcc` and `CUDA_HOME` is unset.

Exact next sequence:

1. Preserve `counter_hash_v2` as part of scientific metadata. Any future RNG
   implementation change needs a new version, the structured namespace grid,
   statistical law gates, and new performance artifacts.
2. Do not infer production queue sizes from the implicit `min(n_trees, 256)`
   fallback. Calibrate batch, admission, and arena chunk sizes per game/hardware
   profile, record them, and reject cross-calibration performance comparisons.
3. When a compatible CUDA toolkit is available, put the shared venv `bin` on
   `PATH`, run the real sibling queued CUDA smoke on an idle GPU, then run a
   controlled poker GPU probe with actual allocated/reserved peaks, evaluator
   calls, family fills, forced drains, padded work, and the 25% margin. Do not
   install system CUDA or change the read-only sibling as a workaround.
4. Keep the CPU poker artifacts as boundary evidence only. A larger CPU seed
   grid may diagnose routing, but it cannot replace the required poker CUDA
   execution and cleanup gate.
5. Only after the real sibling CUDA gate and admissible calibrated poker GPU
   result pass, rerun the focused/full suites and promote Phase 2 in §0.1. Then,
   and only then, begin the one-root branched queue in Phase 3.

### 0.9 Continuation checkpoint — truthful CUDA sync accounting and model-seeded comparisons

This subsection supersedes §0.8 as the first restart source. Phase 2 remains
unpromoted and Phase 3 has not begun.

Implemented in this continuation:

- CUDA dynamic-shape `nonzero` boundaries are now timed and counted exactly,
  with source strata for arena compaction, owner partitioning, family routing,
  and poker's deferred physical splits. The earlier AST gate only rejected
  explicit `.cpu()`, `.item()`, and `.tolist()` calls and therefore did not prove
  the pseudocode's no-per-call-synchronization contract.
- Chunked occurrence-arena gather/scatter no longer uses CUDA `nonzero`.
  Fixed-shape per-chunk masks gather rows without dynamic output shapes;
  unique integer occurrence handles scatter by zero-masked modular deltas.
  Occurrence fields are now explicitly integer metadata, duplicate scatter
  handles fail, and the implementation allocates no unaccounted sentinel
  scratch. The tiny CUDA transactional gate and the logical workload identity
  checks pass.
- `benchmark_modes` now seeds the process-global Torch RNG before bundle/model
  construction and resets it afterward for rollout. Previously separate CLI
  processes used a common rollout seed grid but could construct different
  frozen policies, so the cross-executor timing conclusions in §0.8 and all
  earlier executor-pair artifacts are historical diagnostics, not admissible
  comparisons. New metadata records
  `model_initialization_protocol: torch_seed_before_bundle_v1` and the seed;
  executor comparison rejects artifacts without that protocol.
- `pbg-compare-executors` builds a machine-readable statistical report only
  after verifying environment, scientific workload, model initialization,
  mode set, and common repeat seeds. It retains queue calibration and RNG
  metadata, labels the comparison as non-pathwise, and emits deterministic
  paired-ratio percentile-bootstrap intervals. Same-executor regression gates
  continue to reject calibration or RNG-version mismatches.

Corrected controlled artifacts:

- `queued_phase2_leduc512_wavefront_modelseed_v1_gpu2_2026-07-15.json`;
- `queued_phase2_leduc512_family_fifo_rng_v2_q512_c4096_modelseed_v1_gpu2_2026-07-15.json`;
- `queued_phase2_leduc512_q512_c4096_modelseed_v1_executor_comparison_2026-07-15.json`.

They use GPU 2, training seed 11, repeat seeds
`[104740, 209469, 314198, 418927, 523656]`, 512 trajectory starts, node budget
64, one warmup, five repeats, exact completion, and the explicit queued
512/512/4096 calibration. Results:

- normal median calls fall 17→16. Median rollout is 1.115 s queued versus
  1.057 s wavefront; the paired ratio median is 1.059 with bootstrap interval
  `[1.013, 1.066]`. Complete iteration paired ratio median is 1.059 with
  interval `[1.020, 1.115]`;
- sequential RB median calls also fall 17→16. Median rollout is 1.111 s versus
  1.069 s; paired ratio median is 1.039 with interval `[1.030, 1.053]`.
  Complete iteration ratio median is 0.984 with interval `[0.972, 1.004]`;
- queued median allocated peaks are 28,037,120 bytes normal and 27,971,584
  bytes RB, versus 27,886,080 and 27,897,344 wavefront. Reserved peaks are
  37,748,736 versus 39,845,888 normal and identical at 39,845,888 RB. The
  exact queued structural/in-flight plan remains 5,765,144 bytes;
- the first stratified 128-start diagnostic exposed 99 host synchronizations,
  56 from arena chunk compaction. Fixed-shape arena routing removes all 56.
  The corrected 512-start artifacts still report a median 47 synchronizations
  and about 1.7 ms measured synchronization time, from owner and family
  routing. This is a hard acceptance debt even though it is not the dominant
  wall-time cost on Leduc.

Verification so far after these patches:

```text
benchmark/comparison and observability tests: 12 passed
queued foundations/sequential/poker non-CUDA group: 59 passed, 7 deselected
tiny CUDA arena/RB/poker masked-pack gates: 3 passed
full CPU suite: 262 passed, 14 skipped, one known sandbox-only Gloo denial
approved Gloo-only rerun: 1 passed
git diff --check: clean
```

GPU 2 was checked before each CUDA run and returned to 19 MiB display overhead
and 0% utilization after the final artifact. GPUs 0 and 1 retained other users'
jobs and were not touched.

Exact next sequence:

1. Preserve the model-initialization protocol and reject older cross-executor
   artifacts. Regenerate any CPU poker pair before using it statistically; the
   old pair remains boundary-plumbing evidence only.
2. Replace CUDA owner/family dynamic compaction with a device-resident masked
   routing/scheduling representation, or document and test a different
   pseudocode amendment before claiming acceptance. Merely hiding `nonzero`
   behind a helper is not closure. Preserve stable FIFO order, exact queue age,
   family-only keys, and loud transactional overflow.
3. After that representation change, rerun the fixed-shape CUDA gates and the
   model-seeded Leduc pair. Require zero undeclared hot synchronizations and
   neutral-or-better complete rollout time; keep all calibration metadata.
4. The real sibling CUDA smoke and calibrated poker GPU experiment still need a
   compatible host CUDA toolkit. Do not install system CUDA or modify the
   read-only sibling evaluator as a workaround.
5. The current full CPU regression is closed. After any device-scheduler change,
   rerun the focused and full suites. Only after the hot-sync, real-poker CUDA,
   memory, law, cleanup, and regression gates all pass may Phase 2 be promoted
   and the one-root branched Phase 3 begin.

### 0.10 Requested stopping checkpoint — masked routing foundation

Work is intentionally stopped at a representation boundary, not in the middle
of an executor mutation. The existing `family_fifo_v1` path remains runnable
and retains the verification recorded in §0.9.

One additional foundation was completed after §0.9:

- `MaskedStablePartition` and `stable_partition_masked` convert a vector of
  handles and device family ids into one stable fixed-width row per family,
  with valid-prefix masks and device-resident selected counts. The primitive
  uses no boolean indexing, `nonzero`, scalar read, or host transfer. It rejects
  missing/ambiguous family membership and preserves source FIFO order within
  every family.
- CPU gates verify shape, completeness, stable order, valid-prefix structure,
  and loud invalid routing. The primitive was added to the static no-explicit-
  transfer audit. Its tiny CUDA transaction gate passes on GPU 2 (`1 passed in
  1.76s`), and GPU 2 was idle at 19 MiB before the run.

Not implemented at this stopping point:

- a masked FIFO whose host does not need the device valid count;
- masked `NEW→QUEUED→EVALUATING→DONE` transitions;
- device-side full/forced-drain family selection;
- a family model-dispatch mechanism that consumes a device-selected family
  without reading a scalar or executing every family model; and
- integration of the new partition into sequential rollout.

This is the core reason Phase 2 remains open. Stable device partitioning is
straightforward; dynamically choosing and launching one of several large Torch
model families from an eager Python scheduler without a host scalar boundary is
not. The next implementation should prototype masked FIFO/status consumption
first, then decide between fixed-quanta all-family dispatch, a compiled/device
conditional executor, or an explicit pseudocode amendment permitting one
measured scheduler decision transfer. Do not hide the transfer behind a helper
or interpret the fixed-shape partition alone as zero-sync execution.

The latest complete regression remains the §0.9 result (`262 passed, 14
skipped`, plus the approved Gloo-only pass). After the isolated partition
addition, its focused CPU file passes (`15 passed, 1 deselected`) and the CUDA
primitive gate passes. A future continuation should rerun the full suite after
integrating masked queue consumption, not merely after this isolated primitive.

### 0.11 End-of-night handoff — replace the Leduc performance experiment

No further tests or benchmarks should be run in this session. The user correctly
rejected Leduc as the decision workload for queue performance. Keep its artifacts
only as scheduler-overhead and measurement-plumbing diagnostics.

The corrected Leduc comparison used 512 trajectory starts, node budget 64, and
realized only about 2,900 public nodes and 2,100 decisions per repeat. Its one
small synthetic execution family, shallow synchronized paths, and cheap model
let the wavefront executor make only 17 calls; the queue reduced that to 15--16.
The resulting 4--6% rollout regression says that the current eager scheduler has
measurable fixed overhead when there is almost no fragmentation to recover. It
does **not** answer whether queues help the intended workload. Do not spend more
time optimizing against Leduc or cite it as evidence against the queue design.

The next decision experiment must use the six-player full 10bb poker model and
long, heterogeneous sequential trajectories. This workload can vary across
preflop/flop/turn/river, live-player count, betting history, all-in closure, and
fold/showdown termination; 10--20 transition paths and fragmented postflop
families are exactly where cross-depth queues may repay their overhead.

One minimal placement probe was attempted before stopping:

```text
game/config: configs/train/nlhe6_10bb.yaml
model: full configured d_model=96, 4 heads, 2 preflop and 2 postflop layers
mode: normal
starts: 1
node budget: 64
model device: idle GPU 2
requested evaluator backend: cpu
result: failed before rollout sampling
```

The failure is a precise placement boundary, not a queue failure:

```text
ValueError: backend='cpu' requires CPU primary inputs
```

`build_bundle` currently gives the sibling profile sampler the model/rollout
device tensors. The sibling CPU backend refuses those CUDA inputs, while its
CUDA extension cannot be built on this host because `nvcc`/`CUDA_HOME` are
absent. Therefore a full CUDA-model poker comparison cannot currently be run by
merely selecting `completion.backend: cpu`. The failed process exited normally;
no sibling source was changed.

#### Tomorrow's experiment sequence

1. Resolve evaluator placement before changing queue heuristics. Preferred:
   provide a compatible CUDA toolkit and use the real sibling CUDA backend.
   A deliberately named staged-CPU evaluator adapter may be implemented as an
   experimental fallback, but its transfers/synchronizations must be measured
   separately and it must not be confused with the production CUDA result.
2. Use sequential `normal` first, then sequential RB. Do not use branched RB for
   this Phase 2 comparison. Keep the full six-player 10bb model and game rules.
3. Sweep at least 512, 1,024, 2,048, and 4,096 total trajectory starts. More
   important than the total is the simultaneously resident pool: test safe
   resident/admission quanta such as 128, 256, 512, and 1,024 after analytical
   memory preflight. Do not flush the queue between small streamed chunks and
   then call that a large-rollout queue test.
4. Use node budget 128 initially, which covers the intended 10--20-step paths
   with loud headroom. Stage from a tiny smoke to the first 512-start run and
   increase only after measured allocated/reserved peaks retain the required
   25% safety margin.
5. Record path-length and street/live-player histograms in addition to total
   nodes/decisions. For every execution family record model calls, rows/call,
   fill, padded tokens/pairs, forced drains, queue depth/age, evaluator work,
   declared host synchronizations, and phase time. The key question is whether
   queues consolidate late postflop work from different trajectory depths.
6. Construct both executors from the recorded
   `torch_seed_before_bundle_v1` protocol and use the same repeat-seed grid.
   Because their RNG maps are different, treat comparisons as statistical, not
   pathwise; use enough repeats to cover the heavy path-length tail.
7. Compare rollout time first, then complete iteration time. Report speed per
   realized decision and per model/evaluator unit as well as raw time. A queue
   is promotable only if the realistic poker workload is neutral or faster and
   all law, memory, cleanup, and no-undeclared-sync gates pass.

The first task tomorrow is therefore not another synthetic benchmark. It is to
make the real 10bb evaluator/model placement runnable and scientifically
observable, then execute the scale/residency matrix above.

## 1. Objective

Implement and experimentally validate the algorithms in:

- `docs/theory/pseudocode/queued_rollout.tex`;
- `docs/theory/pseudocode/distributed_update.tex`.

The goal is the fastest end-to-end synchronous training iteration for all three
training modes:

- `normal`;
- `rao_blackwell`;
- `branched_rao_blackwell`.

The work must preserve the finite-sample rollout and learner laws. Performance
is judged by complete iteration wall time and learning throughput, not by model
call count in isolation.

This plan is deliberately self-contained so implementation can begin in a fresh
conversation without relying on prior discussion.

## 2. Fixed design decisions and non-goals

### 2.1 Execution families

Poker uses exactly the following logical rollout and learner families:

\[
\mathcal M=
\{\mathsf{pre}\}\sqcup
\{\mathsf{post}_A:2\leq A\leq |\mathcal P|\}.
\]

The priority relation is
\(\mathsf{pre}<\mathsf{post}_A\) for every \(A\). The postflop families are
pairwise incomparable for scheduling.

Street is model-row metadata. It is not part of the baseline queue key,
sampling stratum, or scheduler priority. Mixed flop/turn/river rows belonging to
the same `post_A` family are padded and masked in one physical model call.

Street-separated queues will exist only as an explicit ablation. Cheap
within-family row ordering by predicted work may also be tested without creating
additional queues.

### 2.2 Branching intensity

`beta` is a frozen public black box supplied to rollout. This plan does not:

- derive or optimize `beta`;
- adapt `beta` to memory pressure;
- train variance, size, or cost heads for `beta`;
- compare branching intensities for variance reduction;
- change offspring counts after randomized rounding.

Every branched experiment must use the same fixed black-box `beta` across all
execution alternatives being compared. A separate project will study the
statistical and memory-optimal choice of `beta`.

### 2.3 Posterior samples

At every branched child, the sibling evaluator samples a complete
collision-free joint private profile by its existing rejection sampler. The
profile contains a hand for every player even when only the actor's hand is
consumed by the policy. This is intentional.

There is no actor-marginal or partial-profile sampler in this plan. The joint
profile is consumed transiently for the action or chance draw and then released.

### 2.4 Synchrony and ownership

- The rollout policy is frozen for the entire iteration.
- Normal and sequential RB use parallel trajectory starts, not multiple roots.
- Branched RB samples exactly one labeled rollout tree on each GPU, and each tree
  has exactly one distinguished root.
- The root uses the ordinary black-box `beta` rule.
- Rollout state, queues, RNG addresses, targets, and learner buffers remain owned
  by their GPU/rank.
- The baseline never exchanges rollout nodes or belief tensors across ranks.
- The first required collective is global decision-mass normalization; gradient
  reduction is synchronous before every optimizer step.

No asynchronous actor/learner policy lag, replay across policy versions,
straggler truncation, oversample-and-discard, or rank skipping is allowed.

## 3. Mathematical invariants

For a local decision occurrence `s`, define

\[
\omega_g(s)=
\begin{cases}
1,&\text{normal or sequential RB},\\
\prod_{r\prec s}N_g(r)^{-1},&\text{branched RB}.
\end{cases}
\]

Let

\[
Z_{g,m}=\sum_{s:\kappa(s)=m}\omega_g(s),\qquad
Z=\sum_g\sum_m Z_{g,m}.
\]

The distributed fixed-rollout objective is

\[
L(\theta)=
\sum_g\sum_s\frac{\omega_g(s)}{Z}
\operatorname{ModeLoss}(\theta;s).
\]

The implementation must not replace this with an equal average of locally
normalized rank objectives. If rank `g` samples from

\[
q_{g,m}(s)=\omega_g(s)/Z_{g,m},
\]

then its logical contribution is multiplied by
\(\alpha_{g,m}=Z_{g,m}/Z\). A flat gradient `SUM` collective therefore needs no
division by world size.

Variable physical microbatches must accumulate exact weighted numerator sums.
They may never be averaged independently and then averaged again.

In branched RB:

- all abstract children, including repeated public labels, remain distinct;
- every edge has mass `1 / N(parent)`;
- deterministic frozen-model outputs may be shared;
- joint profiles, actions, chance events, reach evolution, descendants, and
  completion draws may not be shared;
- queue order must not affect RNG because all draws use logical-address keys;
- reassigning a global sequential trajectory index to another GPU must not change
  that trajectory's draws.

## 4. Recommended target architecture

### 4.1 Per-rank rollout executor

Each rank owns one complete rollout executor and one model replica. The executor
contains:

- one bounded FIFO ready queue for every `m in M`;
- a chunked struct-of-arrays occurrence arena;
- integer occurrence handles rather than Python node objects;
- monotone occurrence status `NEW -> QUEUED -> EVALUATING -> DONE`;
- preallocated input, output, child-sampling, chance, and scatter slabs;
- a calibrated family packer;
- schedule-independent RNG address fields;
- exact high-water memory accounting.

The arena stores only fields required by the selected mode. Normal, sequential
RB, and branched RB must not be forced into one maximal retained layout.

The steady-state scheduler should be device-driven or use device-maintained
metadata with no `.item()`, `.tolist()`, `.cpu()`, or per-call host scalar branch.
A host poll at a true forced-drain or termination boundary is acceptable.

### 4.2 Launch policy

The baseline launch rule is:

1. immediately launch any full family batch;
2. among full families, honor the preflop priority and then choose by queued load
   and oldest age;
3. if no family is full, safely admit more sequential trajectory starts;
4. otherwise force-drain the best partial family batch;
5. memory pressure may force progress but may only delay/reorder work, never
   change offspring counts or discard occurrences.

The packer sees only a single family. It chooses the maximum feasible set under
the calibrated batch, padded-work, sampler-workspace, expansion, and VRAM limits.

### 4.3 Mixed-street postflop execution

`post_A` must accept mixed streets in one call. Required implementation work
includes:

- a common padded compact-combo width or a masked full-combo representation;
- per-row compact-to-full maps and compatibility masks;
- a public-history width supplied by the packer;
- masks covering absent compact rows, board-impossible hands, seats, and padding;
- one scatter back to the full actor-hand output contract.

The family packer may use the maximum padded work of its selected mixed rows to
enforce memory safety. It must not create a hidden street queue.

### 4.4 Deterministic forward coalescing

The mandatory optimization is within-call exact deduplication:

- create one representative for each exact frozen-model input;
- evaluate representatives once;
- scatter immutable outputs back to all logical occurrences;
- continue all stochastic child operations independently.

An optional byte-capped snapshot-local forward cache will be implemented as an
ablation. Cache keys must cover every model-relevant public, belief, folded, mask,
and legality input and must verify equality before reuse. Cache reuse never
merges topology.

### 4.5 Completion and backward reduction

The baseline target builder is streaming:

- normal trajectories emit scalar terminal returns and compact scalar decision
  targets;
- sequential-RB leaves are completed in bounded microbatches and immediately
  backward-propagated along their trajectory;
- branched-RB completion batches feed a posterior-mass/return-mass contraction
  frontier;
- once a parent's final child summary arrives, the children emit the parent
  target and are replaced by one summary;
- completed leaf reaches, completion outputs, and child summaries are released
  as soon as their last consumer finishes.

The first implementation may begin completion after local rollout finishes. An
alternative will overlap leaf completion with continued rollout, but only after
the simpler streamed frontier is correct and measured.

### 4.6 Compact resident learner data

After target construction, each rank retains a GPU-resident struct-of-arrays
dataset grouped only by `M`. It contains:

- immutable encoded model input or a valid compact replay/snapshot;
- old rollout outputs required by the loss;
- target tensors and posterior weights;
- child masks/counts for branched rows;
- decision/path weights;
- legal/type masks;
- optional exact cached reference-policy sufficient statistics.

Learner iterations manipulate only device index tensors. They do not rebuild
Python examples or recopy the whole dataset.

### 4.7 Learner sampling and physical execution

The recommended initial learner implements the pseudocode's unbiased
family-stratified sampler. It also exposes an exact weighted-epoch alternative.

For the stratified sampler, choose integer `B[g,m]` under a common predicted
per-rank time budget by minimizing

\[
\sum_m
\frac{(Z_{g,m}/Z)^2\widehat\sigma_{g,m}^2}{B_{g,m}}
\quad\text{subject to}\quad
\sum_m C_{g,m}(B_{g,m})\leq T_g,
\]

where `C[g,m](b)` is a measured nonlinear family execution-cost curve. The
variance term is a frozen lagged estimate of per-decision gradient trace
covariance, or a cheap signal calibrated against it. The classic linear-cost
Neyman rule is a warm start, not the final allocator.

Logical family batches may be split into arbitrary physical calls. Each physical
loss is scaled by its exact logical contribution. Mixed streets stay together
under normal masks and padding.

### 4.8 Multi-GPU reduction

The baseline learner does not rely on standard DDP graph-order assumptions:

1. zero local gradients with `set_to_none=True`;
2. accumulate every physical call under one common AMP loss scale;
3. materialize zeros for locally absent parameters;
4. flatten fp32 gradients in fixed parameter order;
5. execute one `all_reduce(SUM)` per optimizer step;
6. derive the common nonfinite decision from the already-reduced flat buffer,
   without another collective;
7. unscale once, globally clip, and step the optimizer/scaler identically.

The asynchronous global-`Z` reduction should be launched as soon as local path
weights are known and overlapped with local completion, target compaction, alias
construction, and optional reference evaluation. There is no earlier rollout
barrier.

## 5. Relevant open-source precedents

These sources were inspected directly. They are precedents for execution
mechanisms, not evidence that their estimator should be copied wholesale.

### 5.1 Synchronous rollout and buffers

- [rlpyt synchronous runner](https://github.com/astooke/rlpyt/blob/master/rlpyt/runners/sync_rl.py),
  [GPU sampler](https://github.com/astooke/rlpyt/blob/master/rlpyt/samplers/parallel/gpu/sampler.py),
  and [alternating sampler](https://github.com/astooke/rlpyt/blob/master/rlpyt/samplers/parallel/gpu/alternating_sampler.py):
  one sampler/model replica per GPU, rank-local samples, preallocated shared
  buffers, and optional double buffering.
- [Sample Factory inference worker](https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/algo/sampling/inference_worker.py),
  [batched sampler](https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/algo/sampling/batched_sampling.py),
  [learner batcher](https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/algo/learning/batcher.py),
  and [shared buffers](https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/algo/utils/shared_buffers.py):
  persistent request queues, bounded wait-to-fill, indexed writes into
  preallocated slabs, buffer IDs instead of serialized payloads, slice merging,
  and double-buffered runners.
- [TorchRL MultiSyncDataCollector](https://github.com/pytorch/rl/blob/main/torchrl/collectors/_multi_sync.py):
  synchronized worker launch, reusable buffers, per-worker quotas, compilation,
  CUDA-graph options, and optional reduced CUDA synchronization.
- [EnvPool send/recv example](https://github.com/sail-sg/envpool/blob/main/examples/env_step.py):
  ID-addressed asynchronous submission/completion and a large active pool feeding
  smaller physical batches.
- [SEED RL actor](https://github.com/google-research/seed_rl/blob/master/common/actor.py):
  fixed compiled inference batches and bounded FIFO backpressure.

Transferable tricks: rank-local ownership, bounded family queues, buffer pools,
integer IDs, work-conserving fill, optional micro-timeout, worker quotas,
compilation, and exact backpressure.

Rejected from these systems: changing weights during collection, stale-policy
overlap, central cross-rank rollout gathering, partial-worker masks, and
speed-conditioned truncation.

### 5.2 Learner implementations

- [CleanRL multi-GPU PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_multigpu.py):
  one flat gradient buffer and one `all_reduce` rather than a collective per
  parameter.
- [RLlib train operations](https://github.com/ray-project/ray/blob/master/rllib/execution/train_ops.py):
  load a PPO dataset onto the learner once, then use offsets and permutations for
  repeated epochs.
- [TorchRL PPO implementation](https://github.com/pytorch/rl/blob/main/torchrl/trainers/algorithms/ppo.py):
  learner-device lazy tensor storage, flatten-once data, and without-replacement
  PPO epochs.
- [Acme JAX PPO learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/learning.py):
  compiled epoch/minibatch loops and globally reduced normalization statistics.
- [verl data-parallel PPO actor](https://github.com/verl-project/verl/blob/main/verl/workers/actor/dp_actor.py)
  and [sequence-length balancing](https://github.com/verl-project/verl/blob/main/verl/utils/seqlen_balancing.py):
  separation of mathematical minibatches from physical dynamic batches, exact
  contribution scaling, and squared-length work estimates.
- [OpenRLHF experience maker](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py):
  evaluate frozen reference log probabilities once, retain them with experience,
  and reuse them throughout PPO.
- [rl_games dataset and learner](https://github.com/Denys88/rl_games/blob/master/rl_games/common/datasets.py):
  device-resident shuffled datasets, flat gradient reduction, AMP, and reported
  `torch.compile` speedups.

Transferable tricks: resident compact datasets, device-side index schedules,
exact variable-microbatch scaling, one-time reference evaluation, flat/bucketed
fp32 reduction, global weighted statistics, compile/graph experiments, and
skipping disabled diagnostics.

Rejected: rank-local advantage normalization, equal-rank averaging, independent
per-rank KL early stopping, asynchronous rollout replay, and FSDP/ZeRO before
replicated model memory is shown to be a bottleneck.

### 5.3 Batched tree search

- [OpenSpiel AlphaZero evaluator](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/alpha_zero_torch/vpevaluator.cc):
  bounded inference queue, first-row-triggered 1 ms deadline, batch histograms,
  promise/future completion, sharded immutable inference cache, and multi-device
  outstanding-work accounting.
- [KataGo neural evaluator](https://github.com/lightvector/KataGo/blob/master/cpp/neuralnet/nneval.cpp),
  [search node state](https://github.com/lightvector/KataGo/blob/master/cpp/search/searchnode.h),
  and [CUDA backend](https://github.com/lightvector/KataGo/blob/master/cpp/neuralnet/cudabackend.cpp):
  dynamically filled GPU queues, reusable scratch buffers, warmed/cached shape
  plans, monotone node states, immutable inference caches, and masking to mix
  different board shapes in one batch.
- [Minigo search strategy](https://github.com/tensorflow/minigo/blob/master/strategies.py):
  one network call for leaves collected from different tree depths.
- [Mctx search](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/search.py)
  and [tree storage](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/tree.py):
  static struct-of-arrays tree tensors, preallocation, vectorized gather/scatter,
  and per-row PRNG keys.
- [LightZero CTree wrapper](https://github.com/opendilab/LightZero/blob/main/lzero/mcts/tree_search/mcts_ctree.py):
  batched native traversal/backup and compact index-based latent references; it
  also demonstrates the host-sync anti-pattern to avoid through repeated
  `.cpu().numpy()`, `.tolist()`, and `.item()` calls.

Transferable tricks: cross-depth family queues, monotone in-flight state,
struct-of-arrays arenas, exact deterministic output caching, buffer/shape-plan
reuse, mixed-shape masks, and completion reference counts.

Rejected: virtual loss, adaptive leaf selection, transposition-table topology
merging, duplicate-label suppression, and reuse of stochastic branch results.

## 6. Implementation phases

Every phase should be a small reviewable change with tests. Do not combine the
generic scheduler, poker model-shape rewrite, target streaming, and distributed
learner rewrite into one patch.

### Phase 0: freeze semantics, baseline, and observability

Checkpoint status: implemented with focused tests passing; post-change full
regression and approved Gloo rerun remain.

Work:

- translate the two pseudocodes into explicit assertions in `CONTRACTS.md` and
  `NOTATION.md` where necessary;
- record the current wavefront and current learner as named baselines;
- define global decision-mass versus equal-rank objectives in tests;
- replace bounded metric deques as aggregate counters;
- add exact counters for model calls, rows, padded tokens/pairs, queue depth/age,
  forced drains, cache hits, evaluator calls, host synchronizations, completion
  batches, target bytes, collective bytes/time, and rank wait time.

Acceptance:

- existing tests remain green;
- a tiny synthetic test distinguishes equal-rank from global-mass gradients;
- metric counters match hand-computed tiny workloads.

### Phase 1: generic arena and queue primitives

Checkpoint status: generic primitives and CPU/static tests implemented; CUDA
acceptance and rollout integration remain.

Work:

- add chunked struct-of-arrays occurrence storage in `pbg/core/tree.py` or a
  narrowly scoped new core module;
- implement device-resident bounded FIFO rings per `M`;
- implement monotone node status and duplicate-scheduling assertions;
- implement reusable staging slabs and exact byte accounting;
- add schedule-independent key generation for every draw purpose;
- add family packer interfaces that see mixed-stage row costs but only one
  family key.

Acceptance:

- randomized queue schedules and different sequential GPU assignments realize
  identical logical-address draws;
- overflow fails before changing sampled counts;
- no duplicate occurrence executes twice;
- queue operations and pack selection have no steady-state host sync in CUDA
  tests.

### Phase 2: sequential normal and RB queued rollout

Checkpoint status: not started; this is the next implementation phase after the
Phase 1 CUDA gate.

Work:

- replace breadth/wave advancement in `pbg/core/rollout.py` with persistent ready
  queues;
- support dynamic safe admission and calibrated per-rank trajectory quotas;
- carry exactly one joint private profile down each trajectory;
- close chance-only segments and enqueue the next player occurrence;
- retain mode-minimal old outputs and model inputs;
- preserve the current normal terminal settlement and sequential-RB pre-terminal
  cut.

Acceptance:

- exact small-game terminal/public-path laws match the current sampler;
- normal emits only scalar targets;
- sequential RB emits posterior-weighted all-type targets;
- larger resident pools improve fill without exceeding the memory plan.

### Phase 3: branched queued rollout

Work:

- replace breadth waves in `pbg/core/branching.py` with cross-depth family queues;
- preserve exactly one labeled rollout tree rooted at the distinguished root on
  each rank;
- randomized-round black-box `beta` once per node;
- allocate all prescribed abstract children before execution;
- call the sibling evaluator's full joint rejection sampler per child in batched
  form;
- preserve duplicate labels, edge mass, and child-local RNG;
- evaluate each unique frozen-model input once per call and scatter outputs.

Acceptance:

- tree law, offspring counts, path weights, and duplicate handling match exact
  synthetic references;
- queue schedule changes do not change RNG-addressed samples;
- no output cache shares stochastic branch data;
- node-budget failure remains loud and does not resample or truncate.

### Phase 4: poker mixed-street family executor

Work:

- change `PokerRolloutBackend.plan_policy_batches` and direct forward execution so
  one `post_A` physical call may contain all streets;
- represent compact widths, histories, folded summaries, and masks per row;
- eliminate the current plan-then-resplit-by-street behavior;
- keep one postflop network and shared weights for every `post_A` wrapper;
- remove scheduler-path `.tolist()`, `.item()`, and CPU cache-key extraction;
- prewarm safe family/batch execution shapes.

Acceptance:

- mixed-street outputs match separately executed rows within configured numerical
  tolerances;
- old-policy identity checks pass;
- no increase in illegal/board-impossible mass;
- measured padded work and memory remain within the family plan.

### Phase 5: streaming target compaction

Work:

- generalize the existing streamed branched `P/M` frontier to the queue-produced
  topology;
- specialize sequential normal and RB target paths rather than routing them
  through branched structures;
- create compact resident family buffers;
- release leaf, topology, frozen-snapshot, and completion transients at their last
  consumer;
- overlap the asynchronous global-`Z` reduction with this local work.

Acceptance:

- targets match current exact/small-game references;
- retained bytes match mode-specific analytical plans;
- the first collective, not an explicit barrier, is the synchronization point;
- every rank obtains identical `Z` and target-independent optimizer metadata.

### Phase 6: learner rewrite

Work:

- sample or permute only device index tensors;
- implement family-stratified sampling with global-mass scaling;
- implement exact weighted full epochs as an alternative;
- split logical batches into physical family-only calls with exact numerator
  scaling;
- cache reference outputs once when enabled and admitted by the byte plan;
- implement flat fp32 gradient reduction and common AMP/nonfinite handling from
  the reduced buffer;
- synchronize KL stopping, LR schedules, and scaler changes across ranks.

Acceptance:

- a tiny distributed analytical model matches the exact global objective;
- variable physical partitions produce the same gradient;
- absent local families do not deadlock or omit parameter slots;
- reference caching matches recomputation;
- resume preserves model, optimizer, scaler, RNG, cache policy, and world-size
  identity contracts.

### Phase 7: compiled and alternative executors

Implement feature-gated alternatives only after the eager baseline is correct:

- `torch.compile` per family;
- CUDA graphs for a small set of masked batch quanta;
- within-family length/work ordering;
- snapshot-local frozen-forward cache;
- one versus two staging-buffer sets;
- one versus two CUDA streams;
- synchronous versus overlapped gradient buckets;
- optional pinned-host compact dataset spill;
- optional completion-during-rollout pipeline;
- optional tail-only deterministic inference offload.

Each alternative needs an eager fallback and exact numerical/correctness tests.

### Phase 8: integration and documentation

Work:

- update configs with explicit scheduler/executor choices;
- update memory preflight with measured/calibrated workspace terms;
- document logical versus physical batch sizes;
- document per-rank versus global trajectory counts;
- document the chosen learner sampling policy;
- add run metadata containing every alternative flag and calibration table;
- remove superseded wavefront claims only after the queued implementation passes
  end-to-end gates.

## 7. Alternatives that must be testable

The recommended baseline is the first value in each row.

| Axis | Alternatives | Why test it |
|---|---|---|
| Rollout scheduler | family FIFO full-or-work-exhausted; oldest-first; fullest-first; predicted-time; memory-release-aware | Queue heuristics may trade fill against frontier growth |
| Partial flush | immediate on work exhaustion; first-row timeout sweep in microseconds; age threshold | Sparse branched families may benefit from a short wait only if other work is in flight |
| Queue keys | `M` only; experimental `(M, street)` | Directly test the user's family-only hypothesis |
| Within-family order | stable FIFO; token/work sort; greedy padded-work pack | May reduce padding without adding queues |
| Physical sizes | arbitrary dynamic; fixed batch quanta with masks; hybrid full-graph plus eager tail | Graph reuse may beat padding cost |
| Arena | monolithic preallocation; chunked SoA with several chunk sizes | Allocation overhead versus unused capacity |
| Admission | fixed resident trajectories; dynamic `SafeAdmit`; calibrated unequal per-rank quotas | Fill and rank balance under hard memory limits |
| Chance handling | eager `ChanceClose`; persistent batched chance queue | Evaluator/vectorization benefit versus scheduling complexity |
| Forward reuse | within-call dedup; bounded snapshot cache; no dedup | Cache lookup/storage may outweigh saved MHA calls |
| Target timing | post-rollout streamed frontier; completion during rollout | Potential overlap and earlier storage recycling |
| Routing implementation | device metadata/kernels; CPU control baseline; selective C++/CUDA fused queue-pop/expand/scatter | Quantify host-sync and launch overhead |
| Rollout execution | eager; `torch.compile`; CUDA graphs per family/quantum | Kernel/dispatch optimization |
| Buffering | one staging set; two buffer sets | Double buffering may or may not overlap GPU-heavy work |
| CUDA streams | one stream; separate model/evaluator streams | Same-GPU kernels may contend rather than overlap |
| Precision | fp32; fp16+GradScaler on 2080 Ti; bf16 on supported GPUs | Throughput and numerical stability |
| Reach lifecycle | retained decision reaches; recompute; periodic checkpoints | Memory versus replay time |
| Learner storage | GPU-resident compact; pinned-host double buffer; chunk-local baseline | Residency is fastest if it fits |
| Learner sampling | stratified with replacement; exact weighted epochs; current local sampling baseline | Variance, coverage, and objective fidelity |
| Batch allocation | nonlinear integer allocator; linear Neyman; equal logical counts | Cost prediction overhead versus straggler reduction |
| Allocation signal | periodic gradient-variance estimate; calibrated advantage/loss-norm proxy; decision mass only | True per-example gradients may cost more than the allocation saves |
| Reference KL | one-time exact cache; recompute each physical call; compressed cache | Dominant extra forward versus cache bytes |
| Gradient sync | one flat SUM; several large async buckets; controlled DDP/no-sync; current per-parameter reduction | Collective latency and overlap |
| Index schedule | precomputed device indices; generate per step | Python/launch overhead |
| Optional terms | skip disabled computations; always compute diagnostics; sparse diagnostic cadence | Avoid expensive unused entropy/KL/statistics |
| Cross-rank tail | none; deterministic inference-only tail offload | Possible branched straggler relief; communication risk |
| CPU placement | default; pinned worker threads/NUMA affinity | Relevant on shared multi-GPU hosts |

The following are not alternatives and must remain rejected:

- adaptive `beta` in this project;
- partial posterior hand sampling;
- topology transposition merging;
- virtual loss or adaptive MCTS leaf selection;
- queue-order RNG;
- rollout truncation or node dropping;
- sharing completion draws;
- changing weights while rollout is active;
- replay from earlier policy versions.

## 8. Experiment methodology

### 8.1 Shared-GPU hygiene

Before every CUDA experiment:

1. use `/home/pablob/.venvs/poker`;
2. run `nvidia-smi`;
3. select only an idle GPU explicitly;
4. do not touch another user's process;
5. estimate peak bytes before allocation;
6. stage tiny to larger workloads;
7. leave at least the configured VRAM reserve;
8. verify the selected GPU returns to its original idle memory/utilization after
   the process exits.

For multi-GPU runs, verify every selected GPU is idle before launch. Never begin
with the largest workload or rely on recovering from OOM.

### 8.2 Workloads

Use three layers.

#### Layer A: exact synthetic games

- Kuhn-like and Leduc-like games already supported by the repository;
- normal, sequential RB, and branched RB;
- tiny trees whose probabilities, targets, and gradients can be enumerated;
- intentionally repeated public labels and rank-imbalanced decision counts.

Purpose: law, gradient, RNG, cache, and distributed-normalization correctness.

#### Layer B: poker execution probes

- six-player 10bb model at the small validated width;
- controlled batches containing preflop and each `post_A`;
- mixed flop/turn/river rows in each applicable `post_A`;
- rows spanning short/long histories and folded-summary cases;
- frozen checkpoint and fixed pre-generated public/input records.

Purpose: physical batching, padding, memory, compile/graph, cache, reference, and
kernel measurements without rollout-topology noise.

#### Layer C: full synchronous iterations

- existing normal, sequential-RB, and fixed branched-RB GPU smoke/gate configs;
- identical black-box `beta` and completion sample budget across alternatives;
- one GPU first, then two and three identical 2080 Ti GPUs if idle;
- later H100 validation only after local gates pass.

Purpose: actual end-to-end fastest design and multi-rank straggler behavior.
For sequential modes, run both constant-global-work strong scaling and
constant-work-per-GPU weak scaling. Fixed-`beta` branched rollout has one tree per
GPU, so adding GPUs necessarily adds trees: measure it as weak scaling and
aggregate/per-tree throughput, not as constant-work rollout strong scaling.

### 8.3 Same-work comparison

Execution alternatives must be compared on the same scientific work whenever
possible:

- same checkpoint and frozen/reference weights;
- same configuration and completion budget;
- same abstract-address RNG keys;
- same global trajectory addresses or the same one-root tree address per rank;
- same sampled offspring counts and public events;
- same logical learner indices;
- same optimizer state.

If floating-point batching changes a categorical boundary and therefore later
public outcomes, report that the run is statistically paired rather than
pathwise identical. Keep strict deterministic and throughput profiles separate.

### 8.4 Repetitions and reporting

- Warm extensions, compiled graphs, allocator pools, and model kernels before
  timing.
- Exclude compilation/warmup from steady-state timing but report it separately.
- Use at least five measured repetitions for component probes and at least three
  seeds for full iterations.
- Report median, minimum, maximum, and a confidence interval or bootstrap range.
- Record both CUDA-event kernel time and end-to-end wall time.
- Do not select a design from one unusually favorable topology.

### 8.5 Screening order

Do not run a full combinatorial grid initially.

1. Establish the eager family-only queued baseline.
2. Screen one axis at a time with fixed scientific work.
3. Eliminate alternatives that fail correctness or are slower by a material
   margin.
4. Combine the surviving high-value alternatives.
5. Re-test combinations because compile, graphs, caching, and batch size interact.
6. Validate the final two or three candidates end-to-end across all modes and
   GPU counts.

## 9. Required metrics

### 9.1 Rollout

- total wall and CUDA time;
- public nodes, player decisions, stopped leaves, and abstract children;
- model calls and rows per call by `M`;
- safe capacity, actual batch size, fill ratio, padded tokens, and padded token
  pairs;
- queue depth, oldest age, forced drains, and high-water stalls;
- cache lookups/hits/evictions and saved rows;
- full joint posterior sampler calls, rows, rejection work, and time;
- chance, state transition, reach update, folded-summary, pack/scatter, and model
  time;
- host synchronization count/time;
- allocated and reserved peak bytes;
- rank-local completion time and wait until first collective.

### 9.2 Target construction

- completion batches and rows;
- completion effective sample size and estimator error summaries;
- contraction-frontier maximum rows/bytes;
- time from leaf availability to release;
- compact target bytes per decision;
- topology/leaf/reach release points;
- asynchronous global-`Z` overlap and exposed wait.

### 9.3 Learner

- logical rows and physical calls by `M`;
- current and reference model forward/backward times;
- cache bytes and hit/reuse count;
- optimizer steps and effective weighted rows;
- batch allocator prediction error and per-rank predicted/actual step time;
- gradient pack, collective, exposed collective, clipping, and optimizer time;
- rank straggler time;
- compile/graph hit and fallback counts;
- loss, entropy, KL, clip fraction, explained variance, gradient norm, and scaler
  state.

### 9.4 End to end

- complete iteration wall time;
- decisions and effective weighted decisions per second;
- nodes and completion samples per second;
- time share by rollout/completion/target/learner/collective;
- GPU utilization and memory high-water per rank;
- checkpoint/resume identity;
- short learning curves using the same fixed `beta`, data budget, and optimizer
  schedule.

## 10. Correctness and safety gates

No performance result is admissible until all relevant gates pass.

1. Exact normal terminal law on synthetic games.
2. Exact sequential-RB public path and posterior target law.
3. Branched offspring rounding, duplicate preservation, and edge-mass law.
4. Sibling profile/action/completion independence.
5. Unique root assertion for every per-GPU branched tree.
6. Root uses ordinary black-box `beta`.
7. Schedule-independent RNG under randomized queue drains.
8. Mixed-street batched forward matches separate forwards.
9. Frozen old-policy identity at every retained decision.
10. Streaming targets match nonstreamed reference targets.
11. Cached deterministic forward/reference outputs match recomputation.
12. Variable physical partitions produce the same logical loss and gradient.
13. Global-mass distributed gradient matches an analytical single-process
    concatenation.
14. Missing local families contribute explicit zero parameter slots without
    collective mismatch.
15. Global AMP nonfinite decisions and optimizer/scaler state remain identical.
16. Resume reproduces scientific state with the same world size.
17. Hard memory overflow fails before changing the sample law.
18. No hot-path host transfer or scalar synchronization beyond declared
    boundaries.
19. GPU returns cleanly to its pre-run idle state.

## 11. Experiment stages and decision rules

### Stage 1: queue correctness and call reduction

Compare the current wavefront sampler with eager family-only queues on exact and
poker smoke workloads.

Promote only if:

- every law gate passes;
- model calls or partial drains decrease;
- padded work does not increase enough to erase the call reduction;
- rollout wall time improves or remains neutral;
- memory stays within measured peak plus 25% margin.

### Stage 2: mixed-street batching

Compare:

- family-only mixed-street baseline;
- family-only with within-queue work ordering;
- explicit street-separated ablation.

Select by complete rollout and learner time, not padding alone. Scheduling code
complexity and starvation risk are secondary selection criteria when performance
is statistically indistinguishable.

### Stage 3: learner residency and reference reuse

Compare resident compact buffers with current reconstruction/chunking, then test
reference caching.

Promote caching only if saved repeated reference forwards outweigh lookup and
retained bytes without reducing safe physical batch size.

### Stage 4: compilation and graphs

Test eager, `torch.compile`, and family/batch-quantum CUDA graphs independently.
Promote only stable shape plans with an eager tail fallback and acceptable warmup
cost. A graph that requires excessive inert padding is rejected.

### Stage 5: buffering and overlap

Test two staging sets, separate streams, completion-during-rollout, and async
gradient buckets individually. Promote only measured overlap; GPU-heavy MHA and
evaluator kernels may simply contend.

### Stage 6: distributed scaling

Compare one, two, and three GPUs. For normal and sequential RB, use both the same
global trajectory/data budget and weak-scaling budgets. For branched RB, keep
`beta` fixed and use exactly one tree per GPU, so the rollout comparison is weak
scaling; use fixed compact target tables in a separate learner-only probe when a
constant-data multi-GPU scaling curve is needed. Report:

- speedup and parallel efficiency;
- global-`Z` and gradient collective exposure;
- rank decision mass and step time;
- rollout/target straggler tail;
- whether calibrated sequential quotas improve balance.

Tail-only remote deterministic inference is considered only if the independent
rank-local design still has a material exposed tail. It must preserve logical
tree ownership and be tested for communication and numerical equivalence.

### Stage 7: final end-to-end selection

The winning configuration must:

- pass all three modes;
- be fastest or statistically tied on complete iteration time;
- have no learning-quality regression at equal scientific budgets;
- fit every supported hardware profile with the required safety margin;
- have understandable failure modes and maintainable code;
- retain the family-only default unless the street ablation produces a clear,
  repeatable end-to-end win.

## 12. Expected configuration surface

Names may change during implementation, but every experiment must be represented
in configuration and run metadata. Expected fields include:

```yaml
rollout_executor:
  scheduler: family_fifo
  family_key: execution_family
  street_partition: false
  partial_flush: work_exhausted
  flush_timeout_us: 0
  within_family_order: stable
  arena: chunked_soa
  arena_chunk_nodes: null
  staging_buffers: 1
  cuda_streams: 1
  frozen_forward_cache_bytes: 0
  completion_timing: after_rollout
  compile: false
  cuda_graph_batch_quanta: []

learner_executor:
  sampling: stratified_with_replacement
  allocation: nonlinear_cost
  resident_storage: cuda
  reference_cache: exact_if_fit
  gradient_reduction: flat_sum
  compile: false
  cuda_graph_batch_quanta: []
```

The fixed black-box `beta` remains in its existing rollout configuration and is
not an executor alternative.

## 13. Handoff checklist

Before implementation begins in a fresh context:

1. read `AGENTS.md`, `CONTRACTS.md`, `NOTATION.md`, the two pseudocodes, and this
   plan completely;
2. inspect the dirty worktree and preserve unrelated user changes;
3. confirm the current test baseline and audit documents;
4. create an implementation plan matching Phases 0--8;
5. do not start CUDA validation without `nvidia-smi` and an explicitly idle GPU;
6. do not modify the sibling evaluator; treat it as read-only and retain full
   joint posterior sampling;
7. keep `beta` opaque and fixed throughout this work;
8. implement the eager, family-only, estimator-correct baseline before optional
   optimizations;
9. use `apply_patch` for source edits and small reviewable commits/patch groups;
10. record every benchmark command, config, checkpoint, seed, selected GPU,
    memory estimate, timing result, and final GPU state.

For a fresh restart after the checkpoint recorded in §0, do not repeat the
original first coding task: Phase 0 and the generic Phase 1 code already exist.
Close the remaining Phase 0/1 regression, Gloo, and CUDA gates, then start the
sequential normal/RB queue integration described in §0.4 and Phase 2. Branched
queues, mixed-street poker execution, streamed compaction, and learner reduction
still follow only after the preceding layer is correct.
