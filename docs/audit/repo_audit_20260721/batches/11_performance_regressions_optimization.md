# Cross-cutting performance, regression, and optimization audit

## Scope, comparison rule, and bottom line

This pass compares `537700f5b94873c3daa6e6a03ad00cfef4b8e88b`
through audit HEAD `b55df976e80b82c474c86202c90f3b32072af03c`, with
particular attention to `b94941c`, `79a354b`, `1cde6de`, and the storage
migration sequence `a399e30` / `9af6a29` / `759cde6` / `dce75ff`. It also
reconciles batch reports 00--08, all checked-in timing artifacts under `runs/`,
and `docs/todo/optimization.md`. No test or GPU job was run, and no production
file was modified.

The evidence supports this deliberately narrow conclusion:

- **No measured current-HEAD regression can be claimed.** There is no
  same-hardware, same-config, same-statistical-work run after `dce75ff`, and
  the useful historical artifacts either lack a Git revision or explicitly
  identify an older revision.
- The accelerated learner from `537700f`, the KKT primitives from `79a354b`,
  and the protected-root refill from `1cde6de` are **statically preserved**.
  That establishes implementation survival, not equal throughput.
- There are confirmed **static algorithmic regressions/defects** in the new
  sole path: sequential targets now extract CUDA scalars in Python loops; the
  normative compact topology is produced only after a larger whole-tree seal;
  branched child transients scale with logical offspring; and postflop
  admission uses street-specific work while execution always uses flop width.
- Fixed-beta and KKT-warmup branching lack the scientific queue-pressure
  safeguard, while all branched modes lack a complete predictive downstream
  closure admission. Page rollback is valuable but is not an OOM proof.

The categories used below are: **measured regression**, **static algorithmic
regression**, **high-confidence risk**, **unmeasured preservation**,
**optimization opportunity**, and **stale/unsafe idea**.

## Artifact-backed baselines and their limits

### Comparable facts that are actually present

| Evidence | Artifact-backed observation | What it cannot prove |
|---|---|---|
| Full sequential-RB run | `runs/nlhe_hu_32bb_full_sequential_rb/metrics.jsonl` contains 1,110 two-rank iterations. Direct aggregation gives means of 143.340 s profiled iteration, 44.255 s rollout, 7.642 s targets, 91.386 s optimization, 23.371 s learner forward, 42.750 s backward, 14.095 s reference-cache build, 6.953 s gradient SUM, 2.864 s loss, and 0.973 s optimizer update. Mean decisions are 11,188.8, or about 78.06 decisions/s end-to-end. | `run.json` has a run id and config hash but no Git revision. The run predates the restored sole storage path and changes rollout width over its history. It is a budget baseline, not a HEAD comparison. |
| One-GPU 2,048-root target probe | `runs/experiments/nlhe_hu_32bb_sequential_value_vectorized_bench/metrics.jsonl`: 5,051 decisions, 1.054993 s targets (0.209 ms/decision), 18.866 s iteration, 3.927 GB peak allocated and 6.302 GB peak reserved. Hardware is one RTX 2080 Ti, CUDA 13.0, PyTorch 2.12.1, SDPA, CUDA completion. | One iteration; no embedded commit revision. It is consistent with the worktree preceding `b94941c`, but cannot supply a before/after ratio. |
| One-GPU 2,560-root target probe | `runs/experiments/nlhe_hu_32bb_sequential_value_vectorized_bench_2560/metrics.jsonl`: 6,343 decisions, 1.313069 s targets (0.207 ms/decision), 23.718 s iteration, 4.092 GB peak allocated and 6.377 GB peak reserved on the same hardware/software family. | Same one-iteration/revision limitations. |
| Historical optimized learner claim | `docs/audit/nlhe_hu_32bb_optimizer_speed_record_20260717.md` reports 69.427 s for 232 two-rank steps (0.2993 s/step), including 18.828 s forward, 33.873 s backward, 3.196 s synchronization, 8.794 s reference build, and 3.64/3.88 GB allocated/reserved learner peaks. | The cited `runs/experiments/nlhe_hu_32bb_optimizer_final_widecache_232step/` raw directory is absent. Treat these as documentary acceptance values, not independently re-aggregatable measurements. |
| Fixed-beta geometry | `docs/todo/optimization.md:17-104` reports six cutoff-modified trees at median 3,756 nodes/2,308 decisions, median 24.18 s rollout, and about 3.48/6.13 GiB allocated/reserved. A representative tree made 48 postflop model calls for 2,862 rows and 73 evaluator calls for 4,493 rows. | This is an evaluation/diagnostic path with a law-changing upper cutoff and newly initialized T/C heads, not production fixed-beta training. `runs/branched_rao_blackwell_variance_ckpt1110/manifest.json` identifies revision `2429ab6`, not HEAD. |
| Branched variance timing gate | `runs/branched_rao_blackwell_variance_ckpt1110/timing_gate.json` projects 45.336 active GPU-hours against a 24-hour limit and records `success: false`; the first full moment took 1,487.36 s. | This measures an archived-style variance experiment, not any of the five core workflows. It is evidence against treating that experiment as a routine performance gate. |

The normalized-loss smoke/pilot artifacts are also non-comparable: they use
512 roots per GPU, changed objective scaling, and pre-refactor code. The
32-iteration pilot averages 77.985 s/iteration and 2,488 decisions, but that is
not evidence of either a HEAD improvement or regression.

### Measured-regression classification

**None established.** No artifact satisfies all of: current post-`dce75ff`
code, matching Git identity, matching GPU/software, matching model/checkpoint,
matching roots/completion samples, matching learner statistical rows and
physical shapes, and matching mode semantics. Any report that labels one of
the static defects below a *measured* regression would overstate the evidence.

## Named-improvement preservation matrix

| Named improvement | Current evidence | Classification |
|---|---|---|
| `537700f` backward/learner acceleration | Cost-aware positive allocation remains at `pbg/core/learner.py:212-264`; sampled unique-row reference planning at `:1144-1210`; fused policy terms at `:610-665` and `pbg/games/poker/adapter.py:237`; stable direct flat-gradient backward and SUM at `pbg/core/learner.py:1264-1447` and `pbg/core/optimization.py:113-149`; native biased SDPA at `pbg/games/poker/model/primitives.py:181-204`. The diff from `537700f` removes the last optional copy/flatten fallback rather than reintroducing gradient copies. | **Unmeasured preservation**. Equal-work learner timing is required. |
| `b94941c` no-scalar tree validation | Its vectorized `repeat_interleave`, `index_add_`, `bincount`, and tensor comparisons remain in `pbg/core/tree.py:110-175`. | **Unmeasured preservation** locally, but its principle regressed in new target/evaluator/learner loops; see PERF-001 and PERF-005. |
| `79a354b` KKT | `git diff 79a354b..HEAD -- pbg/core/adaptive_branching.py` is empty. `gamma_deviance`, randomized rounding, log-domain intensity, path recurrences, and controller state remain at `pbg/core/adaptive_branching.py:20-209`; wiring is at `pbg/core/branching.py:438-508` and controller observation at `pbg/core/training.py:1027-1064`. | **Unmeasured preservation** with focused static/test survival. No current production-scale KKT throughput artifact. |
| `1cde6de` small-tree safeguard | Ordinary-root protected count, appended independent root expansion, root reweighting, and pre-refill metadata survive at `pbg/core/branching.py:837-905`; protected target degree survives at `pbg/core/targets.py:556-625`; corrected controller observation survives at `pbg/core/training.py:1037-1061`. | **Unmeasured preservation**. Representation changed; tiny GPU tests do not measure cost. |
| Large-tree/OOM safeguards | Grouped page growth/rollback exists at `pbg/core/branching.py:351-380`, but fixed-beta/warmup pressure suppression is disabled at `:474-481`; seal/conversion, queue clones/cats, folded-summary cache, and logical child waves are outside a complete hard envelope. | **High-confidence safety regression/gap**, not a measured OOM rate. |
| Target-value optimization | Old normal target propagation at `537700f:pbg/core/normal.py:123-150` grouped decisions by device-resident owner depth. The current sole builders loop over trajectories/decisions and convert CUDA tensors to Python scalars at `pbg/core/targets.py:169-182` and `:251-312`. `dce75ff` deleted the older device-reduced `pbg/core/values.py`. | **Static algorithmic regression** with unmeasured magnitude. |

## Findings

### PERF-001 — Sequential target construction reintroduces tens of thousands of CUDA scalar extraction points

- Severity: high
- Confidence: confirmed
- Classification: **static algorithmic regression**
- Modes affected: normal, sequential RB, and BR through the sequential-RB target path
- Contract: HIST-PERF-02, STORE-017, STORE-031
- Evidence:
  - `537700f:pbg/core/normal.py:123-150` performs normal player-clock updates by
    device tensors grouped over bounded depth, with no per-decision Python
    extraction.
  - `pbg/core/targets.py:169-182` executes Python `int`/`float` on CUDA leaf,
    pointer, partition, row, owner, and old-value entries.
  - `pbg/core/targets.py:251-312` scans sequential RB decisions three times and
    uses Python `int`/`bool` on CUDA entries, including
    `bool(pool_valid[pool_row])`.
  - `dce75ff` deletes `pbg/core/values.py`; numerical tests at
    `tests/test_refactor_storage_targets.py:274-379` do not gate host syncs.
- Quantification: for $R$ trajectories and $D$ decisions, the normal loop has
  at least $3R+4D$ Python scalar extractions. Sequential RB has at least
  $5R+9D+U$ extractions, where $U$ is the number of retained
  trajectory/player posterior rows. At the artifact-backed 2,048-root/5,051-
  decision geometry this is at least 26,348 extraction points for normal and
  55,699 for sequential RB before final validity checks. At 2,560 roots and
  6,343 decisions, the sequential lower bound is 69,887 plus $U$ if counting
  all pointer passes exactly. These are algorithmic sync opportunities, not a
  measured count of distinct CUDA stalls.
- Consequence: the historical 1.055 s and 1.313 s target baselines are at direct
  risk even though returns remain numerically correct.
- Recommended disposition: implement the authoritative bounded segmented
  player-clock recurrence directly on sealed decision CSR. Do not resurrect
  discarded rollout state or the superseded wide target representation.
- Verification needed: CUDA sync-debug/profiler gate with zero undeclared
  scalar synchronizations; exact numerical equality; then matched 2,048- and
  2,560-root benchmarks against the artifacts above.

### PERF-002 — Whole-tree sealing and conversion create a multi-copy peak outside transactional admission

- Severity: blocker
- Confidence: confirmed
- Classification: **static algorithmic and hard-memory regression**
- Modes affected: all four training estimators; BR inherits sequential sealing
- Contract: STORE-012, STORE-014, STORE-030, STORE-032, HIST-OOM-01
- Evidence:
  - `pbg/core/tree.py:355-389` retains a node schema of $(7+P)\times4$ bytes
    per node and a 20-byte edge row; `PagedTreeSource.finalize()` at
    `:555-587` gathers complete copies, builds a CSR pointer and int64
    destination, allocates three reordered edge arrays, and returns the larger
    `SampledTree`.
  - `pbg/core/training.py:1725-1739` then builds sequential CSR, while
    `:1765-1826` allocates a full sort/permutation and builds
    `BranchedTreeSpan` after the larger tree exists.
  - The normative branched span is only 28 bytes/node when nominal degree is
    present (`pbg/core/topology.py:153-172`).
  - The analytical plan at `pbg/core/training.py:636-662` charges paged source
    and final topology, but `finalize()` itself performs no overlap transaction
    for gathered copies, reorder destination, and conversion scratch.
- Quantification: for a near-tree with $E\simeq N$, paged source payload is
  about 56 bytes/node at $P=2$ and 72 at $P=6$. The returned `SampledTree` is
  about 48 and 64 bytes/node respectively, versus the 28-byte normative
  branched destination. During finalization the still-live pages, gathered
  node/edge copies, returned tree, and int64 reorder destination imply a lower
  bound near 168 bytes/node at $P=2$ and 216 bytes/node at $P=6$, before page
  slack and the later sort/conversion arrays. That is roughly 6.0x and 7.7x the
  final normative topology at the peak. A one-million-node tree therefore has
  at least about 160/206 MiB of topology-only overlap, before reach, decisions,
  completion, or allocator fragmentation.
- Consequence: successful page admission can still OOM at seal; it also adds
  whole-tree gathers and sorting to every iteration.
- Recommended disposition: reserve stable normative row addresses during
  rollout and seal directly into `SequentialDecisionCSR` or
  `BranchedTreeSpan`; delete production `SampledTree` conversion. Admit source
  plus destination plus bounded scratch as one transaction before copy.
- Verification needed: multi-page forced-tight-ceiling tests and allocator-hook
  evidence that no larger legacy tree or whole-tree uncharged scratch exists.

### PERF-003 — Branched transient work scales with logical offspring, with reach alone able to consume GiBs

- Severity: blocker
- Confidence: confirmed
- Classification: **static algorithmic/hard-memory regression**
- Modes affected: fixed-beta branched RB, KKT branched RB
- Contract: STORE-009, STORE-017, STORE-031, HIST-OOM-01
- Evidence:
  - `pbg/core/branching.py:551-610` materializes parent positions, parents,
    states, repeated reach, profiles, repeated model output, events, successor
    states, and updated reach for all logical children in one wave.
  - Page admission occurs at `pbg/core/branching.py:500`, but no physical plan
    bounds the realized `total`; unset `family_batch_rows` can use the full
    queue at `:819-823`.
  - `ReusableStagingSlabs` exists at `pbg/core/execution.py:782-823` but is not
    used by production branching.
- Quantification: `child_reach` and `next_reach` alone require approximately
  $8CPH$ bytes while both fp32 tensors coexist for $C$ children, $P$ players,
  and exact width $H$. With a full deck ($H=1326$), this is 21,216 bytes/child
  for heads-up and 63,648 bytes/child for six-max: 100,000 children consume
  about 1.98 GiB or 5.93 GiB respectively before profiles, model-output
  repetition, states, events, topology, or allocator slack.
- Consequence: beta and queue depth turn logical statistical work directly into
  unbounded transient allocation, defeating the pseudocode's physical-plan
  invariant even when retained pages fit.
- Recommended disposition: reserve all logical child/action addresses and RNG
  keys first, then materialize fixed maximum-feasible physical subspans into
  reusable staging slabs. Release each subspan before the next.
- Verification needed: identical tree/events/targets across 1-row, tail-heavy,
  and maximum physical plans, plus a constant transient high-water mark as
  logical child count grows.

### PERF-004 — Postflop planning undercharges actual execution and preserves a measured late-street opportunity

- Severity: high
- Confidence: confirmed
- Classification: **static algorithmic regression/accounting defect** plus
  **optimization opportunity**; current magnitude unmeasured
- Modes affected: all five workflows at postflop forwards
- Contract: STORE-005, STORE-017, STORE-030, HIST-PERF-01
- Evidence:
  - `pbg/games/poker/rollout_backend.py:1570-1583` gives the planner
    street-specific widths $\binom{D-3}{2}$, $\binom{D-4}{2}$, and
    $\binom{D-5}{2}$.
  - Rollout, learner, and functional execution set flop compact width
    unconditionally at `pbg/games/poker/rollout_backend.py:1861-1872`,
    `:1974-1983`, and `:2065-2073`.
  - `docs/todo/optimization.md:55-87` records a representative fixed-beta tree
    with 97% turn/river decisions but 18.67 billion flop-padded attention pairs.
- Quantification: at $D=52$, compatible hand rows are 1,176 flop, 1,128 turn,
  and 1,081 river. Ignoring small fixed-token terms, actual-versus-planned
  attention-pair work is about 8.69% higher on turn and 18.35% higher on river.
  Conversely, executing the admitted widths saves about 8.0% and 15.5% of
  quadratic hand-token work. The observed 24.8% turn/72.1% river mix implies
  roughly 13% pair-work savings, consistent with the TODO's 11--13% estimate.
- Consequence: the physical memory ceiling is not conservative, and useful
  late-street throughput is left unused.
- Recommended disposition: keep one logical `post_A` partition but execute
  street-homogeneous physical subspans at the exact admitted width. If that is
  not immediately possible, admit all calls at flop width as the safe interim.
- Verification needed: assert admitted tokens/pairs dominate actual MHA shape;
  prove physical-plan invariance; benchmark equal rows separately on flop,
  turn, and river.

### PERF-005 — Avoidable synchronization points multiply across evaluator calls and learner subspans

- Severity: high
- Confidence: confirmed
- Classification: **high-confidence performance risk**
- Modes affected: all five workflows
- Contract: STORE-017, STORE-024, STORE-031, HIST-PERF-01, HIST-PERF-02
- Evidence:
  - Evaluator facade validation uses Python `bool` on CUDA reductions at
    `pbg/games/poker/mc.py:107-135`, `:169-190`, `:449-456`, and `:486-518`.
  - Every sealed learner/reference forward checks CUDA partition predicates in
    Python at `pbg/games/poker/rollout_backend.py:1904-1933`.
  - Every branched learner subspan executes
    `bool((counts <= 0).any())` at `pbg/core/learner.py:786-790`, then derives a
    dynamic scalar extent from `cumulative[-1]` at `:791-798`.
  - Existing no-sync coverage (`tests/test_aud26_no_host_sync.py:56-165`) does
    not exercise these boundaries.
- Quantification: the historical full sequential run averaged 71.8 evaluator
  calls, 176.6 reference-cache build calls, 573.1 physical learner subspans,
  and 935 policy physical batches per iteration. One avoidable sync per repeated
  subspan is therefore not a constant-cost validation. The fixed-beta example
  made 73 evaluator and about 50 policy calls for one tree. These counts are
  workload evidence, not current stall-time measurements.
- Consequence: direct flat-gradient backward can survive structurally while
  repeated validation barriers erase part of its throughput gain.
- Recommended disposition: validate immutable routing/action extents once at
  seal, use async assertions for device invariants, and aggregate evaluator
  validation at declared bounded boundaries.
- Verification needed: sync-debug and profiler tests for profile sampling,
  showdown, completion, cached/uncached partition forwards, and branched loss;
  gate zero undeclared synchronizations.

### PERF-006 — Preflop model rows and BR completion retain multiplicative width/player waste

- Severity: high
- Confidence: confirmed
- Classification: **static algorithmic regression** and **optimization opportunity**
- Modes affected: preflop storage in all workflows; completion in BR
- Contract: STORE-003, STORE-005, STORE-014, STORE-018, STORE-021
- Evidence:
  - Rollout and branching retain model-side rows at global exact width at
    `pbg/core/rollout.py:319-320,398-410,918-932` and
    `pbg/core/branching.py:151-153,215-238,956-997`, then compress preflop
    afterward at `pbg/core/training.py:1896-1914,2030-2040`.
  - BR keeps only selected-seat decisions, but `_seal_sequential_rollout_storage`
    invokes all-player completion at `pbg/core/training.py:1692-1722`;
    `pbg/games/poker/bundle.py:198-292` allocates `[L,P,H]` outputs, while
    `pbg/core/targets.py:248-312` consumes only selected owners.
- Quantification: full-deck natural preflop width is $K=169$ versus exact width
  $H=1326$, a 7.846x width ratio. Every fp32 model-side vector unnecessarily
  widened this way costs 4,628 extra bytes per decision. For BR, `p_hat` and
  `v_hat` alone cost $8LPH$ bytes; selected-owner output needs $8LH$. At
  $P=6,H=1326$, the excess is 53,040 bytes (51.8 KiB) per leaf, or about 506
  MiB at 10,000 leaves, excluding evaluator workspace and quality fields.
- Consequence: avoidable rollout/seal bandwidth and multiway BR completion
  memory scale with dimensions that the downstream consumer does not need.
- Recommended disposition: create preflop model outputs directly at $K$ and
  apply immutable exact-to-model mapping only in exact losses; add a true
  selected-owner sealed completion boundary for BR.
- Verification needed: allocation hooks rejecting preflop model tensors at
  $H$; BR selected-row equality and measured $P=2$ versus $P=6$ peaks.

### PERF-007 — The hard memory ledger omits live queues, folded summaries, and completion/conversion overlaps

- Severity: blocker
- Confidence: confirmed
- Classification: **high-confidence OOM risk**
- Modes affected: all training modes; greatest risk in branched modes
- Contract: STORE-006, STORE-030, STORE-032, HIST-OOM-01
- Evidence:
  - `DeviceWorkQueue` clones enqueue tensors and may concatenate a whole
    selected logical batch without a `ByteAccountant` at
    `pbg/core/execution.py:596-684`; branching approximates queue bytes as
    roughly eight bytes/row at `pbg/core/branching.py:258-262`.
  - Mandatory folded-summary services are charged only when obsolete
    `evaluator_summary_dim > 0` at
    `pbg/games/poker/rollout_backend.py:1723-1750`, although
    `PostflopNet.requires_partition_summary` is true and named folded fields are
    consumed independently (`pbg/games/poker/model/postflop.py:157-160,241-462`).
  - PERF-002 documents unadmitted seal/conversion overlap.
  - Fixed-beta and dynamic warmup cannot use `safe_tree_size` because
    suppression is gated by `dynamic_active` at
    `pbg/core/branching.py:474-481`; configuration nests the threshold inside
    KKT at `pbg/core/config.py:350-371`.
- Current behavior: grouped page transactions roll back cleanly, but they
  account only currently requested owners. They do not prove closure of every
  live frontier plus queue, seal, completion, target, reference, gradient,
  optimizer, driver reserve, and fragmentation envelope.
- Consequence: the advertised ceiling can show headroom while mandatory live
  owners or transient copies trigger allocator OOM. Fixed-beta also lacks the
  mathematically specified pressure correction.
- Recommended disposition: treat complete downstream closure as a release
  blocker. Move scientific pressure control into common branched config and
  preserve nominal/executed correction in fixed mode and KKT warmup. Account
  queues and folded anchors by immutable schemas.
- Verification needed: injected tight ceilings at child admission, queue,
  fold-summary creation, seal, completion, and targets; all must fail before
  allocation/draw or succeed under a measured declared peak.

### PERF-008 — The `537700f` learner survives statically but has no trustworthy current performance gate

- Severity: high
- Confidence: strong static preservation, performance unknown
- Classification: **unmeasured preservation**
- Modes affected: all training workflows
- Contract: HIST-PERF-01, STORE-031, STORE-036
- Evidence:
  - Current symbols are listed in the named-improvement matrix above.
  - The native biased-SDPA CUDA test remains at
    `tests/test_p6_primitives.py:72-118`; sampled-cache update/RNG equality and
    flat-gradient aliasing remain at
    `tests/test_distributed_learner.py:372-461`.
  - `pbg/core/training.py:2353-2361` still calls `torch.cuda.empty_cache()` at
    the rollout-to-learner boundary. This is a global allocator synchronization
    and forfeits cache reuse. The resume-only call at `:2803-2808` is less
    concerning because it releases temporary checkpoint blocks once.
  - `pbg/benchmark.py:28,68-100,123-129,430-461` covers only normal,
    sequential RB, and fixed branched labels; it can silently intersect away
    missing metrics/modes. KKT and BR are absent.
- Consequence: passing correctness microtests does not establish the historical
  0.2993 s/step geometry, and the benchmark can pass incomplete artifacts.
- Recommended disposition: make the benchmark schema strict and five-workflow
  complete; restore an equal-work isolated learner gate before optimizing
  graphs or communication.
- Verification needed: 53/11 (or mathematically equivalent) cost allocation,
  232 two-rank steps, SDPA/AMP, exact cache cadence, current checkpoint schema,
  and phase-specific timings/peaks. Compare against documentary values only in
  a clearly labeled column until raw historical artifacts are restored.

## Optimization triage against `docs/todo/optimization.md`

| Idea | Current disposition | Reason/evidence |
|---|---|---|
| Direct compact storage/seal | **Correctness prerequisite and highest memory ROI** | Reframe old B3/B4 around current paged owners and PERF-002; do not optimize the deleted list/`cat` implementation described by stale line references. |
| Bounded branched child materialization | **Correctness prerequisite** | PERF-003. Logical work must not define transient size. |
| Vectorized/segmented sequential targets | **Immediate high ROI** | PERF-001; the TODO's old “about 0.4 s ceiling” is stale after the scalar-loop migration. |
| Street-aware postflop physical packing | **Measured opportunity, medium effort** | PERF-004; retain one statistical `post_A` partition. |
| Fuse action sampling, selected log-probability, reach update, and retention | **Live optimization opportunity** | Sequential uses separate reach update and selected-logp work at `pbg/core/rollout.py:819-845`; branched repeats it at `pbg/core/branching.py:608-672`. Measure kernel/model calls before/after. |
| Reference-cache streamlining | **Live optimization opportunity** | Artifact-backed 14.095 s mean and 176.6 build calls/iteration. Cache only consumed KL outputs and preserve sampled-row/RNG equality. |
| CUDA graphs / `torch.compile` for rollout and learner | **Promising only after shapes and syncs are fixed** | Current dynamic child waves, street mismatch, scalar validations, and allocator flushes are graph breakers. First establish stable admitted subspans. |
| Overlap gradient communication with backward | **Live, bounded ceiling** | Historical exposed SUM averages 6.953 s. Preserve one SUM and global finite/clip/update semantics; do not alter optimizer law. |
| Remove rollout-to-learner `empty_cache()` | **Live but benchmark carefully** | `pbg/core/training.py:2360`; replace only after the planner distinguishes reusable reserved blocks from live storage. Blind deletion can make `cudaMemGetInfo` understate workspace. |
| Precompute immutable model inputs | **Live after ownership repair** | Folded summaries are currently duplicated/unaccounted (batch 06 POKER-001/002); anchor them exactly once at the authoritative dependency boundary. |
| Fixed-beta upper cutoff as OOM protection | **Stale/unsafe** | Diagnostic cutoff changes offspring and the sampled law. It is valid only for labeled geometry experiments. Do not promote it to production admission. |
| Recreate global node capacities or deleted `values.py` | **Stale/unsafe** | Conflicts with immutable schemas, last-consumer ownership, and bounded streaming. Reimplement the algorithms on normative stores instead. |
| CPU poker transitions to avoid scheduler sync | **Low priority / likely counterproductive without profile evidence** | Historical scheduler control was about 0.009 s and transferred only about 13 KiB/iteration. Moving transitions to CPU would add strict D2H/H2D dependencies. |
| Archived variance experiment optimization | **Out of production scope** | Its checked-in timing gate fails a 24-GPU-hour budget and it duplicates target/rollout logic; archive rather than optimize as a core path. |

## Conservative benchmark and regression-gate matrix

All GPU stages must begin with `nvidia-smi`, select only an idle GPU, use an
explicit CUDA device, record hardware/software/Git/config/checkpoint identities,
and finish with allocator/process-idle verification. Every stage must use the
analytical planner first and abort if the predicted complete envelope is not
below the requested cap. The memory figures below are launch caps, not promises
that current accounting is complete.

| Stage | Purpose and exact workload | Conservative gate |
|---|---|---|
| 0. Static | CUDA sync-debug tests for target/evaluator/learner boundaries; schema/accounting tests; no GPU-sized rollout | No undeclared scalar sync; no allocation. |
| 1. Tiny five-workflow smoke | One GPU; reduced legal deck, $P=2$, $S=8$ bb; 16 then 64 roots for normal/sequential/BR; one naturally tiny fixed tree at $\beta=1$; KKT warmup/active with a tiny safe threshold; one optimizer step and resume | Planner cap 2 GiB allocated and 3 GiB reserved; 16 completion samples initially. Stop on any uncovered owner or non-idle residue. |
| 2. Sequential target scaling | One RTX 2080 Ti, full-deck HU 32 bb, same checkpoint/config/seed and 1,024 completion samples as historical probes; 512 roots first, then 2,048 and 2,560 only after preflight | Final cap 6 GiB allocated / 8 GiB reserved. Compare target phase to 1.054993 s and 1.313069 s; record completion separately and require zero undeclared syncs. Use at least 5 timed repeats after warmup before declaring a ratio. |
| 3. Isolated learner preservation | Two idle RTX 2080 Ti GPUs; SDPA, AMP, current strict checkpoint; exact historical 53/11 or mathematically equivalent allocation; 232 two-rank steps; fixed sampled rows and reference cadence | Cap 5 GiB allocated and 6 GiB reserved per rank. Report median and distribution of gather/reference/forward/loss/backward/SUM/update. Documentary comparator: 69.427 s total and 0.2993 s/step, explicitly labeled non-raw. |
| 4. Seal/topology scaling | Generate deterministic already-realized synthetic topology on one GPU at 10k, 100k, then 1M nodes without model/evaluator work; compare direct normative seal to current path only in a disposable benchmark branch | Cap 1, 3, then 8 GiB according to preflight. Record source, scratch, destination, peak allocated/reserved, seal time, and host RSS. Never advance if measured peak exceeds declared overlap. |
| 5. Branched bounded-materialization gate | One GPU, deterministic fixed logical topology; replay with physical child subspans 1, 8, 64, and maximum admitted size | Start at 128 total children, then 1k and 10k; cap 2/3/6 GiB. Events, logical ids, targets, and RNG continuation must be bit-identical while transient peak stays constant for a fixed physical plan. |
| 6. Fixed-beta and KKT safety | Only after PERF-002/003/007 are fixed: full-deck HU 32 bb, one tree, $\beta=1.25$ then 2.0; fixed, KKT warmup, KKT active; injected pressure and exact closure ceilings | Begin with a 256-node planned envelope and 16 completion samples. Increase to 1k nodes only if complete downstream prediction is covered and below 5 GiB allocated/7 GiB reserved. Verify nominal/executed counts, no reroll, and controller checkpoint/resume. No law-changing diagnostic cutoff. |
| 7. Street packing microbenchmark | Full-deck postflop, identical sealed rows and model weights; flop/turn/river separately; batch sizes 1, 8, 32, 64 within hardware limits | Cap 4 GiB. Record actual tokens, token pairs, model calls, forward/backward time, and peak memory. Require exact masked output/loss equality and admission >= execution. |
| 8. BR selected completion | One GPU, current strict target checkpoint; $P=2$ then $P=6$, same leaves/keys; all-player reference versus selected-owner implementation | Start 64 leaves/16 samples, then 512 leaves/64 samples under 4 GiB. Require exact selected rows and report evaluator calls, workspace, output bytes, and peak. |
| 9. End-to-end production comparison | Only after stages 0--8: current full HU 32 bb sequential geometry and representative fixed/KKT runs on matched hardware | At least 10 warm iterations plus 20 timed iterations; compare normalized decisions, exact-state loss presentations, model rows, token pairs, evaluator samples, and optimizer steps—not wall time alone. Keep measured, documentary, and inferred columns separate. |

## Synthesis priorities

1. Repair hard-envelope correctness: direct normative seal, accounted queues and
   folded anchors, bounded child materialization, and common fixed/KKT pressure
   control.
2. Remove the sequential target and repeated learner/evaluator synchronization
   regressions, then establish equal-work gates.
3. Prove the `537700f`, KKT, and refill implementations on current HEAD with
   strict checkpoint schemas and artifact provenance.
4. Apply street packing, fused rollout policy work, and reference-cache
   streamlining.
5. Consider graphs and communication overlap only after physical shapes and
   ownership lifetimes are stable.

Until those gates exist, the honest performance statement is: the principal
historical algorithms are visible, several new sole-path defects have strong
static evidence, and current end-to-end regression magnitude remains
unmeasured.
