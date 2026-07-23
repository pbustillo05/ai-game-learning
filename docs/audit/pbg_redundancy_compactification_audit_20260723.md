# PBG redundancy and compactification audit

**Date:** 2026-07-23
**Scope:** every Python file under `pbg/`, checked against the accepted NLHE
workflows and the complete design authority in `docs/theory/`
**Change policy:** this document proposes organizational changes plus the
explicitly approved runtime policies in C1 and C2. Estimators, targets, losses,
model outputs, supported modes, and accelerator requirements must not change.
C1's accepted OOM-conditioned resampling bias must be documented rather than
silently treated as the unbiased theory path.

## Executive decision

The current `pbg/` contains **24,126 non-comment, nonblank Python lines** in 59
files. The cleanup should adopt **20,750 lines as a hard acceptance ceiling**,
measured by the same method. That is a net reduction of 3,376 lines, or 14.0%.
A stretch result near 20,500 lines is plausible, but should not be made an
acceptance requirement until the structural changes have CUDA benchmark
evidence.

The 20,750-line target is achievable without removing a supported feature.
This revision retains the intentional OOM retry/reroll policy, so its code is no
longer counted as removable surface.
The expected reductions come from:

| Cleanup family | Expected net reduction |
|---|---:|
| Dead legacy paths, compatibility aliases, and test-only production helpers | 550–750 |
| One authoritative partition scheduler and compact retry-policy plumbing | 100–200 |
| One occurrence owner and one rollout seal boundary | 800–1,100 |
| Canonical schemas, retained fields, input blocks, and memory accounting | 600–800 |
| Shared poker forward/model/evaluator scaffolding without shared weights | 700–1,000 |
| Configuration, validation, checkpoint, and local clone cleanup | 300–500 |
| **Credible total** | **3,050–4,350** |

The line target must not be met by moving implementation out of `pbg/`,
generating opaque code, collapsing readable statements mechanically, or
weakening validation. Any new production module under `pbg/` remains in the
count.

There are four design decisions more important than the line target:

1. The memory-failure retry path intentionally rerolls a different branched
   tree after updating branching controls. This behavior is approved, but
   several theory rules must be amended to describe the accepted bias.
2. Poker rollout scheduling must stop creating street-specific families. A
   dequeued `post_A` pack may nevertheless use the minimum street in that pack
   to select a safe compact token width.
3. Rollout simultaneously owns a paged topology and a separate occurrence
   arena. Consolidate them behind one logical owner while preserving
   access-specific physical paths: direct dense live-state indexing,
   contiguous append slices, and optimized cold multi-page gathers. The old
   generic paged-column hot path must not be restored merely because it is
   shorter.
4. The scheduler seals to `SealedRolloutSample`, after which training performs a
   second seal into `RolloutStorage`. A one-seal replacement is desirable only
   if before/after CUDA measurements show no end-to-end throughput regression.

Those decisions should be resolved before pursuing small stylistic reductions.

## Scope and measurement

### Accepted user cases

An object was treated as necessary when it participates in at least one of:

- the five installed commands: training, benchmark, best-response evaluation,
  inference export, and inference benchmark;
- the declared public runtime/config/evaluation/inference APIs;
- normal sequential, sequential Rao–Blackwell, branched Rao–Blackwell, and the
  shared-response probe;
- CUDA training, distributed training, completion/evaluator work, variance
  runners, and performance tooling;
- the currently supported MPS inference boundary.

NLHE cash poker is the only active game. Archived Kuhn, Leduc, and synthetic
interfaces are not compatibility requirements. Tests and documentation are
evidence about intent, but a test-only reference does not by itself make a
production object a supported user case.

The variance experiments were conservatively treated as supported because the
repository contract names variance runners as production CUDA workflows. One
of them is already stale; that is recorded below rather than used as a reason
to delete the underlying capability.

### Line-count rule

The baseline counts every physical line containing a Python token other than a
comment, newline, indentation token, or end marker. Empty lines and
comment-only lines are excluded. Docstrings are executable string expressions
and therefore remain counted. The baseline was measured from the working tree
on 2026-07-23; unrelated changes outside `pbg/` do not affect it.

The largest files are:

| File | Counted lines |
|---|---:|
| `pbg/core/training.py` | 3,351 |
| `pbg/games/poker/rollout_backend.py` | 2,718 |
| `pbg/core/storage.py` | 1,656 |
| `pbg/core/learner.py` | 1,412 |
| `pbg/core/branching.py` | 1,239 |
| `pbg/core/rollout.py` | 976 |
| `pbg/core/targets.py` | 948 |
| `pbg/core/execution.py` | 917 |

These eight files contain 13,217 lines, or 54.8% of `pbg/`. The first two alone
contain 25.2%. Compactification should reduce responsibilities in these files,
not merely shorten already-small leaf modules.

The tree contains 929 AST class/function definitions when methods and nested
functions are included. The repository reachability audit indexes 426
top-level definitions, reaches 416 from the installed/public roots, and reports
10 review candidates. That analysis is intentionally conservative: protocol
dispatch and broad internal roots make many obsolete methods appear reachable.
The object-level findings below therefore combine call-site searches, accepted
entry-point traces, schema/lifetime tracing, and comparison with the theory.

## Authoritative design constraints used in this audit

The cleanup must preserve these contracts:

- `docs/theory/core_math.tex` owns sampling laws, targets, estimators, and
  losses.
- `docs/theory/pseudocode/storage_lifecycle.tex:77-116` makes
  `PartitionSpec` the opaque core boundary. Core must not parse game-stage
  labels or synthesize subpartitions. Poker has exactly `pre` and `post_A`;
  flop, turn, and river with the same live-player count are one partition.
- `storage_lifecycle.tex:180-210` specifies one chunked append-only
  `OccurrenceArena`, stable logical row addresses, and one explicit exact seal.
- `storage_lifecycle.tex:213-298` gives the normative retained columns. A
  convenience field is not retained merely because it is easy to copy.
- `storage_lifecycle.tex:300-326` specifies decision-only sequential topology
  and contiguous branched child spans. In particular there is no stored
  `childNode`, duplicated child event, or stored edge weight.
- `storage_lifecycle.tex:377-382` requires last-consumer writes and rejects
  replay, all-node reach memos, and algebraically duplicated targets.
- `storage_lifecycle.tex:540-556`,
  `family_scheduled_rollout.tex:136-146`, and
  `sequential_memory_control.tex:123-126` currently prohibit truncating,
  discarding, retrying, or rerolling an already selected iteration to fit
  memory. The repository's intentionally accepted branched-OOM exception now
  supersedes those clauses; the exact theory amendments are listed in C1.
- `family_scheduled_rollout.tex:170-196` currently says a failed rank releases
  the current store, checkpoints the last completed iteration, and terminates
  through one common failure agreement. It needs an explicit branched-OOM
  retry branch while retaining terminal handling for other failures.
- `family_scheduled_rollout.tex:200-232` says the single seal allocates exact
  destinations while source chunks are still charged, copies stable spans
  once, validates, and then releases sources.
- `docs/theory/MHA_model.md:21-25` requires separate preflop and postflop
  weights even where architecture is shared.
- `MHA_model.md:159-171` allows an internal board-compatible postflop
  tokenization, with full canonical-hand output restored. It supports the
  dequeue-time compact-width policy in C2, but does not permit street-specific
  rollout partitions or street-specific retained storage.

## Theory conflicts and approved design decisions

### C1. Branched OOM intentionally restarts with new randomness and controls

**Objects involved**

- `pbg/core/training.py:_one_iteration` (`training.py:2185-2361`)
- `pbg/core/distributed_failure.py:DistributedFailureAgreement.run_retryable`
- `pbg/core/adaptive_branching.py:AdaptiveKKTController.emergency_backoff`
- `pbg/core/adaptive_branching.py:FixedBetaController.emergency_backoff`
- `pbg/core/config.py:RolloutConfig.branching_emergency_backoff`
- `pbg/core/config.py:RolloutConfig.branching_oom_retry_limit`
- the `attempt`/`branching_attempt` plumbing through
  `_build_iteration_data` and `sample_trees_branched`
- `pbg/core/branching.py:335`, where the RNG address becomes
  `tree_address + attempt * world_size`

**Approved behavior**

On a branched memory failure, `_one_iteration` releases every owner from the
failed attempt, clears recoverable allocator state, updates the branching
controls (the applicable $\beta$, $\kappa$, and resulting $N$ policy), advances
the attempt address, and starts over with new randomness. The new attempt is a
different sampled tree conditioned on the earlier attempt failing. The user
explicitly accepts the resulting bias in exchange for safer automatic recovery.

This path must remain narrowly scoped:

- retry only a classified branched memory/admission failure;
- make every rank agree on failure and the next attempt number before any rank
  draws again;
- release the entire failed partial/sealed store before retry;
- update controller state identically on every rank;
- give every attempt a disjoint logical RNG address;
- retain the configured retry limit and terminate cleanly when it is exhausted;
- record failed-attempt cause, old/new controls, attempt address, and memory
  peaks so the accepted selection bias is observable;
- never apply this exception to normal sequential, sequential-RB, correctness,
  evaluator, invalid-data, or collective failures.

The cleanup may consolidate retry records and controller update plumbing, but
must not delete `run_retryable`, retry configuration, emergency backoff, or
attempt-addressed randomness.

**Theory rules that must be changed**

The following clauses currently forbid the approved behavior and should be
amended by the theory owner. This audit does not modify `docs/theory/`:

1. `pseudocode/sequential_memory_control.tex:123-125`: “Never ... reroll, or
   retry an iteration.” Keep this rule for sequential modes and add an explicit
   reference to the branched-OOM exception.
2. `pseudocode/distributed_update.tex:121-123`: the terminal agreement says to
   stop and not reroll. Add a retryable-branched-OOM outcome distinct from the
   terminal-failure outcome, with common-rank agreement before a new draw.
3. `pseudocode/family_scheduled_rollout.tex:85-87`: `SafeAdmit` may not retry or
   replace a trajectory. Preserve this for sequential roots and successful
   admission; state that a failed whole branched attempt may be discarded and
   restarted under the bounded OOM policy.
4. `family_scheduled_rollout.tex:93-96`: any failed hard transaction exits
   without retrying a draw. Split failures into retryable branched memory
   failures and terminal failures.
5. `family_scheduled_rollout.tex:142-145`: the next fixed root may not be
   replaced/retried/rerolled. Preserve this for sequential modes; define the
   single branched tree as replaceable only after the entire failed attempt is
   released.
6. `family_scheduled_rollout.tex:190-192`: any rank failure currently mandates
   checkpoint and termination. Add the synchronized retry transition, followed
   by the existing termination transition on exhaustion or non-memory failure.
7. `pseudocode/storage_lifecycle.tex:550-555`: `SafeAdmit` may not reroll or
   replace selected roots and any failed transaction stops. Add the same
   whole-attempt branched-OOM exception; do not weaken last-consumer or
   transaction accounting.
8. `storage_lifecycle.tex:663-665`: the acceptance gate requires singleton
   failure with no reroll. Split the gate: sequential failure remains
   no-reroll, while branched OOM must prove synchronized release, changed RNG,
   updated controls, bounded retries, and terminal behavior on exhaustion.

`core_math.tex` does not appear to state the runtime OOM policy directly, but
the revised pseudocode should explicitly acknowledge that the realized
branched training sample is selected by this recovery mechanism and may
therefore be biased.

### C2. Street-specific execution families bypass `PartitionSpec`

**Objects involved**

- `pbg/games/poker/rollout_backend.py:PokerRolloutBackend.execution_families`
  (`post_{live}_s{street}`)
- `PokerRolloutBackend.rollout_family_ids`
- `PokerRolloutBackend._parse_execution_family`
- `PokerRolloutBackend.rollout_partition_id`
- street-sensitive family maxima/priorities and related scheduler metrics
- `pbg/core/rollout.py:_family_spec` and `_family_ids`
- the corresponding family-queue logic in `pbg/core/branching.py`

**Conflict and redundancy**

The backend constructs `pre` plus separate flop/turn/river queues for every
live-player count. Core schedules those backend strings and later collapses
them back into the correct learner partition. This is exactly the forbidden
pattern: a parallel physical taxonomy exists beside the immutable
`PartitionSet`, names are parsed for semantic meaning, and the postflop
partition is split by an undeclared public label.

The backend's newer forward code already permits `post_A` to mix all three
streets. Padding every such pack to flop width is safe but wastes substantial
branched work when a dequeued pack contains only turns or rivers.

**Approved replacement**

Create the `PartitionSet` before rollout and make it the queue/scheduler
authority. Route with `PartitionSet.route`, order eligible queues with its
preorder, and use stable integer partition ids. Poker exposes only `pre` and
`post_A`. Delete street-bearing family labels, label parsing, and the later
physical-to-natural remapping.

When a stable prefix $Q$ is dequeued from `post_A`, the poker-owned physical
planner computes

$$
s_{\min}(Q)=\min_{x\in Q}\operatorname{street}(x)
$$

or, equivalently,
$b_{\min}(Q)=\min_{x\in Q}|\operatorname{board}(x)|$. It then uses the compact
hand width

$$
H_Q^{\max}=\binom{4R-b_{\min}(Q)}{2}
$$

for every row in that physical pack. Because the minimum street has the fewest
board cards, this is the maximum compact width needed by any row in the pack:

- any pack containing a flop uses flop width;
- a turn/river-only pack uses turn width;
- an all-river pack uses river width and receives no flop/turn padding.

This is **dequeue-time physical sizing**, not a new family or retained-storage
partition. It must not sort, split, reroute, or delay rows by street. Queue
identity remains only `post_A`; canonical stored outputs and targets remain full
$H$ rows. The poker backend should return the realized pack descriptor/token
cost through `PartitionSpec.byte_work` (or its replacement), so generic core
never parses street. Compute the minimum with one device reduction and fold its
scalar into an already required scheduler-control read; do not add a per-row
device-to-host transfer or synchronization.

**Theory rules that must be changed**

The desired physical optimization is compatible with the optional compact
tokenization in `MHA_model.md:159-171`, but the storage pseudocode currently
forbids it:

1. `pseudocode/storage_lifecycle.tex:107-115` says flop, turn, and river are
   neither packed nor assigned different physical costs by street. Preserve
   the single `post_A` partition and no-street-sorting rules, but allow a
   dequeued stable prefix to declare its maximum required compact width from
   the minimum street.
2. `pseudocode/family_scheduled_rollout.tex:89-92` requires worst-case physical
   planning without an undeclared game label. Make the pack-cost oracle an
   explicit game-owned member of `PartitionSpec.byte_work`; core sees only its
   declared result.
3. `storage_lifecycle.tex:669-671` says the current design intentionally has no
   postflop-street specialization without a controlled experiment. Replace that
   statement after the C2 benchmark gate passes, while continuing to prohibit
   street-specific queues and retained layouts.

The benchmark must compare equal dequeued rows and fixed logical RNG addresses,
report the street composition of each pack, and demonstrate end-to-end gain in
branched mode without a sequential regression outside measurement noise.

**Resolution (2026-07-23): implemented and accepted.** Rollout now constructs
the immutable `PartitionSet` before queue allocation, routes exclusively with
`PartitionSet.route`, keys queues by stable integer partition id, and applies
the declared preorder directly. The poker backend exposes only `pre` and
`post_A`; the street-bearing labels, parser, family router, and later
family-to-partition remap were deleted.

`PartitionSpec.byte_work.realize` now returns an opaque `PhysicalPack`.
For a dequeued poker `post_A` prefix, one device minimum-street reduction
selects flop, turn, or river compact width. The prefix is neither sorted nor
split by street, while canonical captured inputs, outputs, and targets retain
their natural full widths. CUDA coverage asserts the exact mixed-prefix widths
for $R=4$: 78 for a pack containing a flop, 66 for a turn/river-only pack, and
55 for an all-river pack.

The repeatable gate is `tools/benchmark-c2-partition-packs.py`. On the
five-repeat `--smoke` CUDA gate on an RTX 2080 Ti, with identical repeat seeds
and realized decision counts, dequeue compactification improved aggregate
branched end-to-end seconds per decision by **6.59%**. Every paired repeat
improved (0.98%--28.03%). Sequential execution improved by **3.38%**, so the
no-regression gate passed. The separate composition probe observed seven
turn/river-only packs, including three all-river packs. The exact command was:

```bash
python tools/benchmark-c2-partition-packs.py \
  --config configs/train/nlhe6_h100.yaml \
  --hardware-profile local_3x2080ti \
  --smoke --warmup 1 --repeats 5 \
  --branching-upper-cutoff-nodes 256 \
  --max-sequential-regression 0.10
```

### C3. Two occurrence/topology owners retain forbidden edge data

**Objects involved**

- `pbg/core/tree.py:PagedRolloutTopology`
- `PagedRolloutTopology.node_schema`
- `PagedRolloutTopology.edge_schema`
- `pbg/core/execution.py:ChunkedOccurrenceArena`
- `pbg/core/execution.py:ArenaField` and `_OCCURRENCE_FIELDS`
- the unused intended authority
  `pbg/core/storage.py:occurrence_schema`
- `pbg/core/storage.py:PagedTensorArena` and `PagedTensorColumn`
- `pbg/games/poker/tensor_state.py:PokerTensorStateStore`
- scheduler invariants that force topology ids and occurrence handles to remain
  aligned

**Conflict**

Sequential and branched rollout allocate both `PagedRolloutTopology` and
`ChunkedOccurrenceArena`. They duplicate parent/state/owner/tree/child
metadata, status/lifetime information, and row identity. The topology also
retains `edge_parent`, `edge_ordinal`, `child_node`, `child_event`, and
`edge_weight`. The compact branched design explicitly says child rows are a
contiguous span, action order supplies the event association, and the uniform
edge weight is implicit.

`occurrence_schema` describes most of the intended single owner but is not on
the production path. It is a false dead-code candidate: the implementation
drifted away from its own canonical schema.

**What C3 means**

This finding separates the *logical schema* from the *physical allocator*.
`occurrence_schema` is useful as the authoritative list of fields and
lifetimes; it does not imply that rollout should use the existing
`PagedTensorArena`. Today one occurrence is represented in two physical owners,
and scheduler code keeps their row ids aligned. The cleanup goal is one owner,
not “restore the old paged implementation.”

The prior severe throughput regression from paged tensor access makes allocator
choice a measurement question. No implementation should be selected because it
is already generic or removes the most lines.

**What the July 22 performance commits actually establish**

The relevant history does not show that paging itself must be removed. It shows
that a generic page-routing interface must not sit between a hot operation and
a physical span whose location is already known:

1. The immediate prerequisite, commit
   `d95b7d370e20fdbc2d4c959a4b6a3b29e7d7a96d`, separated mutable live
   state/history into a recyclable dense slab. `PagedTensorColumn` delegates
   ordinary indexing directly to that slab, while the arena remains responsible
   for allocation, lifetime, and memory admission. This reduced the accepted
   2,048-start sequential-RB rollout median from the 87.45 s matched baseline
   to 51.65 s, a 40.9% reduction. The reproduced pre-fix run was 90.27 s. The
   expensive behavior was the generic $O(B\,p)$ page-match/merge work on every
   state expression, where $p$ is the physical page width—not the existence of
   paged append-only owners.
2. Commit `5ecfeb490a88d2b6c38ada1e7fc855f4d33fcee3` made a monotone
   append commit use host-known page intersections and direct tensor slices
   rather than routing the reserved ids back through arbitrary-row
   `PagedTensorArena.write`. Its paired family-84 measurement reduced rollout
   from 32.368 s to 32.011 s (1.1%) with exactly equal decisions, model calls,
   padded attention pairs, and memory peak.
3. Commit `cf6bc34a248a34f85f1eda694e358f58eb995d87` made an unavoidable
   multi-page arbitrary gather initialize from its first page and merge only
   later pages. Its paired measurement reduced rollout from 31.953 s to
   31.840 s (0.35%) and peak reservation from 9,055,502,336 to
   9,015,656,448 bytes with equal statistical work.

Therefore C3 is an ownership and schema consolidation, not permission to
replace the measured access paths with one uniform container. One logical owner
may deliberately contain multiple physical access domains:

| Access contract | Required physical path |
|---|---|
| Mutable recyclable live state/history | One admitted dense slab; direct tensor indexing after handles are validated |
| Monotone append-only topology or retained rows | Host-known contiguous page-span slice copies |
| Arbitrary rows within one page | `index_select`/`index_copy_`, never a page-match matrix |
| Unavoidable multi-page arbitrary read | Keep only at existing topology/cold/seal sites; seed from the first page and merge only remaining pages |

`PagedTensorColumn` is not a suitable universal compute abstraction. It may
remain as an ownership-facing facade only where it resolves to the direct slab
or one of the optimized operations above. A C3 cleanup should preferably expose
resolved dense tensors/spans to the hot poker kernels and keep page translation
inside reserve, append, cold gather, and seal operations. It must not recreate:

- a CUDA `[batch_rows, page_rows]` page-match matrix for a known contiguous
  append or a resolved slab;
- a Python page loop for each ordinary live-state field expression;
- gather-modify-write for whole-row replacement;
- zero-initialize-plus-`where` merging of the first gather page;
- device-to-host synchronization to rediscover page intersections already
  known from host cursors and immutable page lengths.

**Required benchmark before choosing**

Prototype the smallest viable versions of these layouts behind the same
rollout operations:

1. **Current dual-owner baseline:** unchanged `PagedRolloutTopology` plus
   `ChunkedOccurrenceArena`, including all three July 22 access optimizations.
2. **Single logical owner over the measured physical paths:** remove duplicate
   fields and row identities, but initially reuse the dense-slab,
   contiguous-span, single-page, and optimized cold-gather kernels above. This
   is the preferred low-risk cleanup candidate.
3. **Single coarse-segment arena:** one schema-declared struct-of-arrays owner
   using large calibrated segments and vectorized contiguous span operations.
   It must preserve the access contracts above and have no Python page loop in
   the hot dequeue/reserve/commit path.
4. **Single generic paged arena:** only if an instrumented prototype proves
   that hot operations resolve to dense tensors or contiguous spans before
   compute. A generic `PagedTensorColumn` implementation that handles every
   indexing expression by page routing is not an eligible candidate.

The prototype may use `occurrence_schema` to guarantee identical fields, but it
must exercise the real reserve, child commit, queue, and seal operations—not a
synthetic append-only microbenchmark.

Measure on idle CUDA GPUs with the current production environment:

- normal and sequential-RB on a realistic heads-up run;
- branched-RB on both a realistic heads-up run and a high-pressure multiplayer
  run (use maintained production configurations; at minimum cover the
  equivalent of $P=2,S=32,R=13$ and $P=6,S=100,R=13$);
- identical fixed logical seeds, roots, branching controls, model, precision,
  physical batch limits, and completion settings;
- warmup followed by at least five measured repetitions;
- end-to-end spots/s, rollout nodes/s, rollout wall time, seal wall time,
  allocator call count, CUDA allocated/reserved peaks, source capacity, seal
  overlap peak, and maximum safely admitted work.
- direct-slab accesses, direct contiguous-span commits, single-page
  index-select/index-copy operations, generic page-match writes, multi-page
  gathers, and bytes/tensors merged through `where`.

Select a replacement only if:

- topology and retained tensors are exactly equal for the same sampled object;
- the hot loop performs zero generic page-match writes and zero multi-page
  gathers for live-state expressions;
- the previously optimized append and gather cases still select their direct
  paths, enforced by focused regression tests rather than inferred only from
  end-to-end timing;
- there is no statistically significant slowdown in rollout or end-to-end
  throughput, no individual realistic case falls below 99% of the current
  baseline, and the storage-attributed portion does not regress;
- peak reserved memory does not regress;
- the candidate removes the duplicated owner and forbidden branched edge
  columns without adding host synchronization.

Among candidates passing those gates, choose the highest end-to-end throughput;
use lower peak memory and then fewer lines as tie-breakers. If memory and
throughput are genuinely non-dominated—for example, materially more admitted
work but a measurable speed loss—present that tradeoff before choosing rather
than hiding it behind the compactification target.

After a winner is measured, logical child reservation should record
`child_begin` and `degree`, action rows should record target-eligible events,
and seal should produce `SequentialDecisionCSR` or `BranchedTreeSpan`. Delete
the duplicated logical owner and its redundant fields in the same migration.
Do not delete a specialized physical access path merely because the new logical
owner can describe it through a generic interface.

### C4. Rollout is sealed twice

**Objects involved**

- `pbg/core/storage.py:SealedRolloutSample`
- `pbg/core/storage.py:SealedDecisionInputs`
- scheduler-local seal/capture code in `pbg/core/rollout.py` and
  `pbg/core/branching.py`
- `pbg/core/training.py:_seal_sequential_rollout_storage`
- `pbg/core/training.py:_seal_branched_rollout_storage`
- `pbg/core/training.py:_natural_partition_set`

**Conflict**

Schedulers copy paged sources into a `SealedRolloutSample`; training then
translates those already sealed blocks into `RolloutStorage`. This creates two
nominal ownership boundaries, two schema translations, duplicated validation,
and ambiguous memory accounting for source plus first destination plus final
destination. The class docstring calls the intermediate a “direct compact-seal
result,” but it is not the immutable object consumed by targets.

**Performance-gated cleanup**

Pass the immutable `PartitionSet` into rollout. Make the scheduler's one seal
return `RolloutStorage` directly, plus a small separate metrics/diagnostics
result if needed. Calculate exact topology/partition/leaf extents once, copy
once, validate once, and release source pages at their last copy.
Delete `SealedRolloutSample` and both training translation seals.

The direct seal must preserve the efficient parts of the existing capture
path. It may not replace batched contiguous copies with Python per-row/page
gathers, `torch.cat` growth, repeated schema reconstruction, extra
synchronization, or an exact destination allocated before its byte transaction
is admitted.

Benchmark C4 independently on the current physical arena before combining it
with the C3 winner. For the same realistic CUDA matrix, fixed sampled objects,
and repetitions defined in C3, compare:

- scheduler source capture time;
- first/intermediate seal time in the baseline;
- training translation-seal time in the baseline;
- direct-seal time in the candidate;
- end-to-end spots/s;
- CUDA allocated/reserved peak during seal;
- bytes copied and number/size distribution of copy operations.

The candidate must produce byte-for-byte equivalent canonical
`RolloutStorage`, stay at or above 98% of baseline throughput in every realistic
case with no statistically significant aggregate slowdown, and not increase
peak reserved memory. A measurable slowdown blocks the migration until the
copy/layout implementation is fixed; line reduction is not sufficient
justification. Repeat the direct-seal comparison after C3 so the combined
layout also passes.

Probe-only output can use a distinct `ProbeRollout` because it is a different
accepted consumer; it must not keep a general training compatibility layer
alive.

### C5. Retained decision schemas duplicate immutable inputs

**Objects involved**

- `pbg/core/storage.py:rollout_decision_schema`
- `legal_event_ids` and `legal_event_mask` decision columns
- `pbg/core/training.py:_seal_sequential_rollout_storage`
- `pbg/core/training.py:_seal_branched_rollout_storage`
- `pbg/core/learner.py:_legal`
- compatibility lookup helpers named `_column` in `targets.py` and
  `learner.py`
- `rollout_policy_source_schema`, `rollout_action_source_schema`,
  `rollout_decision_schema`, and `rollout_action_schema`

**Conflict and redundancy**

The immutable natural poker input already contains `legal_action_mask`, and
the legal event id domain is the fixed ordered `arange(E)`. The seal retains
the mask again per decision and creates an expanded event-id view per decision.
These columns are absent from the normative retained decision tables.

The expanded event-id view also exposes an accounting hazard: any
`numel * element_size` calculation reports logical replicated payload despite
the stride-zero view sharing one storage. More generally, a retained schema
should not contain views whose ownership differs from its declared extent.

The source and destination schemas rename `old_logp` to `old_log_prob` and
`old_v` to `old_value`, forcing translation code. The action source and action
destination schemas repeat almost the same structure. Learner and target
helpers still accept stale names including `logWeight`, `oldV`, `old_v`,
`old_logp`, `legal_ids`, `terminal_reward`, and `actor_type`, even though the
current seal declares canonical names.

**Required cleanup**

- Retain legality once in the immutable partition input.
- Construct the fixed event-id row at the policy adapter boundary.
- Adopt one canonical name for every field from source reservation through
  `TrainingStorage`.
- Use one schema plus a mode/phase field subset where source and destination
  genuinely differ.
- Delete compatibility-name search after the one-time migration.
- Index canonical columns directly so stale schema use fails immediately.

### C6. Several memory models compete with the immutable schema

**Objects involved**

- `pbg/core/branching.py:forecast_workspace`, which calls a learner estimator
  before a no-grad rollout inference forward
- `pbg/games/poker/rollout_backend.py:training_transient_bytes`
- `pbg/core/memory_control.py:PhaseMemoryControl.admit`
- `pbg/core/memory_control.py:PhaseMemoryForecaster`
- `pbg/core/memory_control.py:MemoryLedger`
- `pbg/core/execution.py:ByteAccountant`
- `pbg/games/poker/bundle.py:PokerRolloutMemoryPlan`
- `pbg/games/poker/bundle.py:estimate_rollout_memory`
- `pbg/games/poker/bundle.py:_preflight_cuda_rollout`
- `pbg/core/rollout.py:_admission_plan`, including its generic fallback
- backend `admission_byte_terms`
- `pbg/core/training.py:_analytical_memory_plan`
- `pbg/core/training.py:_value_workspace_budget`
- `pbg/core/training.py:_completion_batch_rows`
- learner physical-shape construction, including the reference-build
  `training_transient_bytes(...) / 2` approximation
- `pbg/core/training.py:_tensor_bytes`
- `pbg/core/memory_control.py:unique_tensor_bytes`
- `PartitionSpec.byte_work`, whose contract is declared but not used as the
  rollout scheduler's authority

**Confirmed false rejection**

`branching.py:forecast_workspace` calls `training_transient_bytes` immediately
before a rollout policy forward executed under `torch.no_grad()`. The estimator
explicitly charges learner activations, gathered loss tensors, forward/backward
memory, and a final $2\times$ envelope. It is not an inference contract.

The observed rejection decomposes exactly as:

| Component | Bytes |
|---|---:|
| Estimated 44-row river learner transient | 11,504,270,976 |
| Existing PyTorch allocator-reserved memory | 2,701,131,776 |
| Reported rejection prediction | **14,205,402,752** |

The same estimator reports 21.96 GB for 84 river rows. A prior real 84-row
river rollout inference completed at 4.58 GB peak allocated and 7.34 GB peak
reserved. No CUDA OOM occurred. The admission result is therefore a known
false positive, not a conservative near miss.

The false rejection changes the statistical result through the approved C1
policy: it backs off branching controls, advances the attempt RNG address,
samples a fresh tree, and may refill a much smaller result. Accurate admission
is therefore both a performance and sampling-quality requirement.

**Why the current model is wrong**

1. **Wrong operation:** rollout inference is charged
   `training_transient_bytes`. A no-grad model call does not retain backward
   activations or loss tensors.
2. **Wrong allocator arithmetic:** `PhaseMemoryControl.admit` computes
   `reserved_now + workspace_bytes`. Allocator-reserved memory is the current
   device footprint containing both live blocks and cached blocks that may
   satisfy the forward. It is neither all free nor all unavailable. Adding a
   complete workspace estimate to it assumes zero cache reuse and double-counts
   capacity already reserved for compatible allocations.
3. **Wrong quantity:** a whole-phase forecast, current allocator state, and one
   operation's transient are combined by `max(...)` despite representing
   different boundaries and units. A phase forecast is not a local workspace
   estimate.
4. **Wrong calibration unit:** the branched controller fits every phase against
   total tree nodes. Local inference memory is determined by operation kind,
   partition, street-aware compact width, rows, padded tokens, attention
   backend, precision, and cache state—not total tree nodes.
5. **Wrong observed peak:** `_completed_phase_observations` supplies
   `phase_*_peak_allocated_bytes` as `actual_peak_bytes`, although the theory and
   admission claim to model allocator-reserved/device high-water memory.
6. **Uncalibrated ratios:** reference-cache construction is approximated as
   one-half of a learner prediction, and completion uses a hand-written
   per-leaf formula plus unexplained budget divisions. These are distinct
   operations and require distinct envelopes.
7. **Independent partial budgets:** `ByteAccountant` defaults to a separate
   rollout limit and accounts only selected owners. The poker preflight,
   generic rollout admission, all-phase analytical plan, learner budget, and
   allocator controller can therefore reject or admit against different
   notions of the same device ceiling.
8. **Insufficient rejection context:** `BranchedMemoryRejection` records phase,
   prediction, ceiling, and old global peaks, but not the rejected attempt's
   current node count, selected family/partition, family rows, street floor,
   padded work, or current allocator snapshot. After C1 restarts, metrics can
   make the fresh tree look like the rejected geometry.

There are separate poker, rollout-admission, and training all-phase estimates.
They reconstruct related tensor costs with different assumptions. For example,
the analytical plan takes maxima over partition schemas and uses global
worst-case widths, while the storage design requires schema-derived,
phase-specific accounting over actual immutable partitions and realized work.
The generic `_admission_plan` fallback estimates legacy all-node topology and
reach costs even though active NLHE supplies a calibrated backend method.

The same unique-storage byte traversal exists in `training._tensor_bytes` and
`memory_control.unique_tensor_bytes`.

**Existing theory already requires the separation**

This is implementation drift rather than a missing design decision:

- `storage_lifecycle.tex:466-469` defines separate calibrated
  $W_{\rm infer}$, $W_{\rm seal}$, $W_{\rm completion}$, and
  $W_{\rm backward}$.
- `storage_lifecycle.tex:471-490` defines $W_{{\rm train},m}$ only for a learner
  decision-index subspan and explicitly lists its input/target gathers,
  current/reference forwards, backward activations, loss reductions, AMP
  temporaries, and backend workspace.
- `storage_lifecycle.tex:500-510` says objects with disjoint lifetimes are not
  added.
- `distributed_update.tex:88-108` treats learner and reference-build
  workspaces as distinct calibrated functions over a small warmed physical
  shape set.
- `storage_lifecycle.tex:511-526` requires allocator-reserved/device high-water
  observations and an explicit guard, not a learner formula reused in rollout.

No theory change is needed for the core C6 correction. The theory's phrase
“admission uses allocator-reserved/NVML bytes” should be implemented as a
measured peak/growth contract; it must not be interpreted as “add all current
reserved bytes to every transient estimate.”

## C6 target: one memory coordinator with three distinct responsibilities

Unification does **not** mean one formula for every phase. It means one
coordinator, one hard device budget, one snapshot type, and one admission
record, with three non-overlapping models.

### 1. Exact resident-owner ledger

The owner ledger accounts only tensors whose lifetime extends across an
operation:

- model, optimizer, gradients, reference model/cache, and collective buffers;
- occurrence/topology/queue/state capacities;
- partition input/decision/action/leaf source capacities;
- exact seal destinations and scratch overlap;
- target/posterior/training-storage destinations.

Schema-generated allocated capacities and explicit non-schema owners feed this
ledger. Aliases count once. Every allocation transaction declares source
owners, new overlapping destinations, scratch, and its release boundary.

`ByteAccountant` should become this exact labeled owner ledger inside the
coordinator. It must stop enforcing an independent default
`execution_memory_bytes` ceiling. The sole hard ceiling comes from the hardware
profile/device budget. If a rollout-owner sub-budget is still useful for
diagnostics, derive it from the phase plan and do not make it a second
authority.

### 2. Operation-specific transient envelopes

Transient memory is indexed by an immutable operation signature, not by tree
nodes:

| Operation kind | Required signature fields |
|---|---|
| `rollout_policy_inference` | partition, minimum street/board count, live players, rows, compact/padded tokens, token pairs, inference dtype, attention backend/query chunk, enabled output heads |
| `reference_build_inference` | partition, rows/tokens, dtype/backend, cache destination/scatter form |
| `learner_forward_backward` | partition, sampled decision rows, action rows, cache mode, dtype/backend, loss/head set |
| `completion` | leaf kind, rows, players, exact width, samples/strata, evaluator version |
| `target_build` | mode, decision/action/leaf rows, exact width |
| `seal_copy` | source spans/pages, destination bytes, scratch shape |
| `optimizer_collective` | parameter/gradient bytes, optimizer, world size |

`training_transient_bytes` must be renamed/replaced as the learner-only
analytic fallback. Add a distinct rollout-inference estimator that excludes
losses, backward activations, optimizer/gradient work, and the learner's
$2\times$ envelope. Do not estimate reference build as half of training; give
it its own signature and calibration.

For every warmed signature, record at the exact operation boundary:

- current and peak `memory_allocated`;
- current and peak `memory_reserved`;
- current and minimum `cudaMemGetInfo.free`;
- active bytes, cached/reserved slack, and inactive-split bytes;
- exact persistent destination growth included by the transaction;
- elapsed time and physical work.

Calibration stores monotone one-sided upper envelopes for:

- $\Delta A^+$: incremental peak live allocation above the boundary;
- $S^+$: peak allocator slack, `peak_reserved - peak_allocated`;
- $\Delta E^+$: non-PyTorch/device growth not explained by allocator reserved
  memory.

Calibration keys must include GPU model, compute capability, CUDA/PyTorch and
evaluator versions, model/config fingerprint, attention backend, query chunk,
precision, and output heads. Mismatched calibration is invalid.

### 3. Between-iteration/attempt geometry forecast

The phase forecaster predicts future resident geometry and selects sequential
root width or initial/next branched controls. It consumes vectors of meaningful
counts—occurrences, decisions by partition, actions, leaves, queue/state high
water, summary rows, and seal destinations—not one `nodes` scalar for every
phase.

It is never passed as the transient workspace for a local policy call.
Surprising completed observations widen it immediately; tightening retains the
lagged window/hysteresis rule. The approved C1 retry uses the rejection record
to update the next attempt, but the phase forecast and operation admission
remain separate.

## Correct local admission arithmetic

At an operation boundary, take one synchronized `MemorySnapshot`:

$$
\begin{aligned}
T&=\text{device total},&
F_0&=\text{device free},&
D_0&=T-F_0,\\
A_0&=\text{PyTorch allocated},&
R_0&=\text{PyTorch reserved},&
E_0&=\max(0,D_0-R_0).
\end{aligned}
$$

Let $L$ be exact new persistent live bytes whose allocation overlaps the
operation. Let $\Delta A^+(q)$, $S^+(q)$, and $\Delta E^+(q)$ be the calibrated
upper envelopes for operation signature $q$. Predict:

$$
\begin{aligned}
\widehat A_{\rm peak}
  &=A_0+L+\Delta A^+(q),\\
\widehat R_{\rm peak}
  &=\max\left\{R_0,\widehat A_{\rm peak}+S^+(q)\right\},\\
\widehat D_{\rm peak}
  &=\max\left\{D_0,E_0+\widehat R_{\rm peak}+\Delta E^+(q)\right\}.
\end{aligned}
$$

Admit only if $\widehat D_{\rm peak}\leq M_{\rm hard}$. The explicit device
guard remains outside this prediction.

The important operation is `max`, not `R_0 + workspace`. Existing reserved
capacity is already part of the current footprint; it can cover later live
allocations. The calibrated slack envelope accounts for fragmentation and
allocator bins without declaring every cached byte either perfectly reusable
or permanently unavailable.

Each transaction must declare whether $L$ is already present in $A_0$ and which
allocations are covered by $\Delta A^+$; admission validation rejects
overlapping/double-counted ownership terms. For CUDA-library/evaluator
allocations outside PyTorch, $\Delta E^+$ and the `cudaMemGetInfo` observation
are mandatory.

Using the reported family-84 run as a calibration point:

- peak allocated is 4.58 GB;
- peak reserved is 7.34 GB;
- the observed peak allocator-slack envelope is about 2.76 GB;
- current reserved at the failing boundary was about 2.70 GB.

For the matching boundary/signature, the reserved prediction is therefore
approximately

$$
\max(2.70,\ 4.58+2.76)=7.34\ {\rm GB}
$$

plus only the explicit residual/guard—not 2.70 GB plus a 21.96 GB learner
estimate. Until a separate 44-row point exists, the monotone 84-row envelope is
a safe conservative bound for that smaller river pack and remains below the
10 GB ceiling.

## Physical planning and cold-start rules

Maintain a small finite set of warmed physical shapes for every operation
signature. Calibration should cover at least rows
`1, 2, 4, 8, 16, 28, 44, 84` where legal, plus exact production tail shapes,
for preflop and flop/turn/river compact widths. Measure both a cold allocator
state and production-like warm/fragmented states, with at least five
repetitions. Store the maximum observed surprise and one-sided residual.

Rules:

- no extrapolation beyond the largest calibrated shape;
- round a requested operation up to the next calibrated envelope when one
  exists;
- otherwise split the stable physical pack into calibrated subspans without
  changing queue order, RNG, topology, or retained rows;
- a transient-size admission rejection is allowed only when even a calibrated
  singleton cannot fit;
- exact resident growth or seal overlap may still reject the attempt when the
  declared transaction cannot fit;
- never call `empty_cache()` per forward as an accounting substitute; measure
  its phase-boundary use separately because it changes both throughput and
  allocator state;
- an actual OOM is a safety failure and C1 recovery event, not a calibration
  probe. It widens/invalidates the envelope before the next attempt.

This means an uncertain 84-row family is physically split rather than causing
an entirely new sampled tree. Once family-84 is calibrated safe, it executes as
one pack.

## Required rejection and retry telemetry

Replace the sparse `BranchedMemoryRejection` payload with one immutable
`MemoryAdmissionRecord`. Every rejected attempt must log, before owners are
released:

- run, iteration, rank, attempt id, logical RNG tree address;
- operation kind and calibration fingerprint/source/sample count;
- current tree node count, decisions, actions, leaves, live state rows, and
  queue depths;
- selected partition/family, selected family-row count, minimum street,
  compact width, padded tokens/token pairs, and proposed child rows;
- exact owner-ledger reservations by label and total;
- $T,F_0,D_0,A_0,R_0,E_0$, active, cached-slack, inactive-split, and current
  peak values;
- $L,\Delta A^+,S^+,\Delta E^+$ and each resulting predicted peak;
- hard ceiling, guard, rejection predicate, and whether this was predicted
  rejection or actual CUDA OOM;
- old and new $\beta/\kappa/N$, cleanup result, next attempt id, and next RNG
  address.

The retry summary must retain each attempt's own geometry. Never report the
fresh attempt's final node count as though it described the rejected tree.

## C6 migration order

1. Add shadow-only operation snapshots around rollout family forwards and
   reproduce the 44/84-row evidence without changing admission decisions.
2. Add `rollout_policy_inference` calibration and the correct local admission
   formula. Remove the branching call to `training_transient_bytes`.
3. Make the physical planner split to calibrated safe subspans before rejecting
   a tree.
4. Introduce `MemoryAdmissionRecord` and preserve rejected-attempt geometry
   through C1 cleanup/retry.
5. Move `ByteAccountant` reservations under the same coordinator/hard ceiling;
   remove the independent default rollout budget and generic active-game
   fallback.
6. Migrate seal, completion, target, learner, reference-build, optimizer, and
   collective operations to separate calibrated envelopes.
7. Replace the single-node branched phase fit with the geometry vector and use
   reserved/device peaks in completed observations.
8. Delete `_analytical_memory_plan`, `PokerRolloutMemoryPlan`,
   `_value_workspace_budget`, duplicated tensor traversal, ratio-based
   reference/completion guesses, and old admission paths once their last
   consumer has migrated.

The temporary shadow instrumentation is measurement-only and must be removed
after the profile is established; it is not a second production admission
path.

## C6 acceptance gates

- The exact 44-row rejection fixture predicts the same terms as the new formula
  and is admitted under the 10 GB ceiling using a valid family-84 envelope.
- Training, reference-build, and rollout-inference envelopes are type-distinct;
  calling a learner envelope from rollout is impossible at the API/type level.
- For every calibrated signature and allocator-state class, observed allocated,
  reserved, and device-used peaks stay below their predicted upper envelopes
  and hard ceiling.
- Cold/warm allocator tests prove cached reserved capacity is not blindly added
  to transient bytes and is not assumed perfectly reusable.
- Oversized uncalibrated packs split without changing fixed-seed outputs.
- Only a singleton or exact owner/seal transaction can cause a predicted
  attempt rejection.
- Injected underprediction widens the precise operation envelope and produces
  the full attempt/node/family telemetry before C1 retries.
- Realistic full runs use the intended near-10-GB ceiling without false
  28–44-row river rejection, while maintaining the configured guard and
  producing no unexplained OOM.
- One final memory report reconciles exact named owners, unattributed active
  bytes, allocator cache/slack, non-PyTorch device use, operation envelopes,
  and phase peaks from the same coordinator.

## Unused-object disposition

This section is the complete high-confidence deletion/promotion set found by
the audit. “Delete” means no accepted user case needs the object after the
named chain is removed. “Promote” means static reachability calls it dead only
because production is using a conflicting parallel implementation.

### Whole obsolete paths

| Object or chain | Evidence | Disposition |
|---|---|---|
| `pbg/games/poker/policy_planning.py` (`PlannedPolicyBatch`, `plan_policy_batches`) | Reached only through the old direct `policy_forward` chain; rollout uses `policy_forward_family_actual_cost` | Delete module |
| `RolloutBackend.plan_policy_batches` and `RolloutBackend.policy_forward` protocol methods | No accepted core caller; describe the obsolete direct path | Delete protocol members |
| `PokerRolloutBackend._planned_policy_batches`, `plan_policy_batches`, `_policy_work_rows`, `_record_policy_model_call`, `policy_forward`, and `_policy_forward_direct` | Self-contained fallback/direct call chain; not used by rollout-family execution | Delete after confirming inference uses the partition preparation boundary |
| Generic `sample_profiles`, `sample_events`, and `sampled_terminal_reward` protocol methods plus poker implementations | Accepted production draws are logical-address keyed; unkeyed methods survive only as fallbacks/tests | Require keyed methods and delete unkeyed chain |
| `core.rollout._sample_profiles_logical` and `_sample_events_logical` fallback branches | The fallback changes scheduler-invariance guarantees and is not needed by NLHE | Make keyed calls direct; update synthetic test doubles |
| `core.completion.ReachView` and `MCCompletion` | Mutable tree/reach completion protocol; targets use immutable sealed completion specs | Delete after `GameBundle.completion_backend` is typed to the sealed boundary |
| `_TensorPokerGPUCompletion.complete`, `complete_keyed`, and their mutable-tree `_complete_impl` path; `bundle._reach_rows` | Current training/evaluation consumes `complete_sealed`/selected sealed rows | Delete mutable completion chain |
| `pbg/games/poker/mc.py:_reach_rows` | No accepted caller; duplicate adapter for the same old reach wrapper | Delete |

The unkeyed methods should not be retained as a generic-game abstraction:
there is only one active game, and counter-addressed draws are part of the
production sampling contract.

### Test-only and documentation-only production objects

| Object | Current references | Disposition |
|---|---|---|
| `cards._HAND_CARDS` and module-level `hand_cards` | Tests/docs only; duplicate `DeckGeometry.combo_cards`/`hand_cards` for a full deck | Move oracle expectations into tests using `DeckGeometry(13)`, then delete |
| module-level `cards.board_conflict_mask` | Tests/docs only; duplicates `DeckGeometry.board_conflict_mask` | Migrate tests and delete |
| `cards.sample_uniform_complement` | Tests/docs only; active sampling belongs to the sibling evaluator/keyed backend | Keep a test-local oracle only if still useful, then delete from `pbg` |
| `rules.PREFLOP` | Documentation only; production uses the numeric street contract without this alias | Delete and update docs/import lists |
| `rules.assemble_showdown_payoff` | Documentation only; production showdown/completion is owned by the sibling evaluator | Delete; do not create a second settlement authority |

Keep `rules.PotStrata`: `PokerRolloutBackend.pot_strata` and
`completion_inputs` use it as the typed public pot-strata boundary supplied to
the sibling evaluator. Only the unused local payoff assembler is redundant.

### Isolated unused aliases and methods

These have no call/reference outside their own definitions unless noted:

- `PartitionSpec.m`, `PartitionSpec.X_m`, and `PartitionSpec.Y_m`;
- `RecyclableTensorPool.slack_bytes`;
- `SequentialDecisionCSR.trajectoryPtr`, `decisionPartition`,
  `decisionRow`, and `leafRow`;
- `PhysicalPlan = PhysicalPlanner`;
- `ChooseSequentialRootWidth` and `ObserveSequentialMemory`;
- `PokerRolloutBackend.state_store = states` and `game = self`;
- `PokerRolloutBackend.rotate_button`, `is_terminal`, and `is_pre_terminal`;
- `PokerRolloutBackend.legal_raise_targets = raise_targets`;
- `PokerPolicyAdapter.compose = probabilities`;
- `PokerTensorStateStore.history_parent`;
- `PagedRolloutTopology.append_roots = add_roots`;
- `pbg.inference._move_public` and `_move_summary`.

`PokerRolloutBackend.owner_of = classify` is referenced by one adapter test,
not by an accepted API. Change that test to call `classify`, then delete the
alias.

`ByteWorkContract.retained_row_bytes` and `transient_pack_bytes` are currently
unreferenced, but they are not deletion candidates: use them, or equivalent
clearly named contract methods, in the unified memory authority. Likewise,
`PartitionSet.scheduler_order` should be wired into rollout rather than
deleted.

### Inference result fields that never contain a value

`Decision.seat_values_bb` and `FullStrategy.seat_values_bb` are always `None`.
`PokerInferenceSession._evaluate_full_device` initializes `seat_values = None`
and never assigns another value. Remove both public fields and the unused tuple
slot unless a real seat-value head is explicitly designed and implemented.
This is stale API surface, not a supported computed result.

`PokerInferenceSession.new_hand` may delegate to `new_hands([button])[0]` after
their validation/state creation is consolidated; both entry points remain
useful and should not be removed.

### Objects to keep or promote despite low static reachability

- `pbg/core/storage.py:occurrence_schema`: keep as the candidate logical field
  authority in C3. Promote it only as a schema; select the physical arena by
  the required CUDA benchmark.
- `pbg/games/poker/model.__getattr__`: keep; it is the deliberate lazy export
  boundary and is dynamically invoked.
- `SealedDecisionInputs.gather_partition_inputs` and
  `PokerRolloutBackend.functional_policy_forward_partition`: retain their
  capability while the variance runner is migrated, then merge them into the
  canonical partition input/forward abstraction described below.
- `BranchingDiagnosticLimits` and cutoff/diagnostic objects: keep; benchmarks,
  variance runners, and tools use them.
- all active configuration dataclasses: low direct name counts are caused by
  reflective construction, not deadness.

### Stale supported variance runner

`experiments/rao_blackwell_variance/eval/branched_rao_blackwell_variance.py`
still accesses:

- `sample.action_ptr`;
- `sample.action_source`;
- `sample.decision_source`.

`SealedRolloutSample` instead exposes partitioned
`decision_inputs.action_ptrs`, `action_sources`, and `decision_sources`.
Therefore the branched variance runner is already incompatible with the
current rollout object.

Do not delete the stateless forward or gather capability based on ordinary
production reachability. In cleanup phase 0, either:

1. migrate the experiment to the canonical `RolloutStorage` partition/action
   API, which is the recommended choice; or
2. obtain an explicit decision that this variance runner is no longer an
   accepted user case.

No general compatibility properties should be added to `SealedRolloutSample`
to mask this mismatch, especially because that intermediate object should
itself disappear.

## Redundant objects and proposed small abstractions

### R1. One measured occurrence-owner API

Current overlap:

- `PagedTensorArena`/`PagedTensorColumn`;
- `ChunkedOccurrenceArena`/`ArenaField`;
- `PagedRolloutTopology`;
- several custom capacity/rollback methods.

Expose one small append-only owner API with:

- `reserve(count)`, `append(columns)`, `write(rows, columns)`, and `release()`;
- transactional growth against `ByteAccountant`;
- stable integer row addresses;
- optional status transitions supplied by a thin occurrence wrapper;
- a sealing iterator over source spans so destinations can be copied without
  concatenation.

`execution.require_capacity` and storage arena growth/rollback already have
near-identical control flow. One implementation should own it, but its physical
layout must be the C3 benchmark winner. A specialized coarse-segment
implementation is preferable to a generic paged abstraction if it is faster.
The immutable schema may describe fields/accounting without controlling every
hot-path access. Do not route individual reads/writes through generic Python
column/page dispatch merely to satisfy this abstraction.

### R2. One partition-row/input block

`PartitionRolloutBlock`, `SealedDecisionInputs`, and `TrainingPartition` all
carry some combination of:

- a sealed input column block;
- partition row identity;
- immutable referenced input pools;
- gather-and-resolve logic;
- row alignment validation.

Use one immutable `PartitionRows` (name not prescriptive) containing the
natural input block, canonical decision/action columns, and input pools. Give
it one bounded `gather_inputs(rows)` method and one
`validate_against(PartitionSpec, mode)` method. Rollout, target construction,
variance evaluation, and learner should consume the same object rather than
re-wrapping it.

### R3. One policy-batch preparation boundary

Poker public reconstruction, slicing, compatibility masks, class aggregation,
live-seat extraction, folded-summary lookup, and legality setup are repeated
across:

- `policy_forward_family`;
- `policy_forward_partition`;
- `functional_policy_forward_partition`;
- inference `_evaluate_full_device`.

Extract an immutable `PreparedPokerBatch` or a small
`prepare_partition_forward` function. It should produce the natural model
arguments and legality for one declared partition. Current-model,
reference-model, stateless-parameter, and inference calls then differ only in
which callable executes the prepared batch.

This abstraction must preserve bounded physical packing and last-consumer
release. It must not create a full-batch cache.

### R4. Shared model scaffolding, independent parameters

`model/preflop.py` and `model/postflop.py` repeat public-token construction,
history encoding, transformer trunk plumbing, output validation, and head
dispatch. Their architectures intentionally differ in private-row
initialization and folded/postflop features, and their parameters must remain
independent.

Extract reusable module classes such as:

- `PublicHistoryEncoder`;
- `PublicTokenEncoder` with pre/post feature adapters;
- `TransformerTrunk`;
- a common output/readout validation scaffold.

Instantiate separate copies inside preflop and postflop models. Do not pass one
shared instance to both. Preserve or explicitly migrate checkpoint key names;
checkpoint compatibility should be a one-time loader migration, not permanent
duplicate modules.

### R5. One sibling-evaluator facade

`poker/mc.py`, `poker/folded.py`, and parts of `bundle.py` repeat device
requirements, board normalization, evaluator contract loading/binding, combo
validation, and result checks. Consolidate those mechanics in one small
accelerator-only sibling-evaluator facade. Completion and folded summaries
remain distinct operations on that facade.

This also makes it clearer that local showdown/card Monte Carlo helpers are not
a second evaluator implementation.

### R6. One seal helper and one completion selection helper

`rollout_backend._seal_columns` and
`training._sealed_dynamic_block` are nearly the same schema validation/copy
operation. `_sealed_dynamic_block` also accepts `schema=None` and infers a
schema from realized columns, although active callers pass a schema. Delete
schema inference and use one exact `seal_columns(schema, spans, rows, ledger)`
helper.

`complete_sealed` and `complete_selected_sealed` duplicate validation,
selection, result assembly, and evaluator dispatch. Make one accept an explicit
row selection; the all-row case passes the full span.

### R7. Share scheduler mechanics, not mode semantics

Sequential and branched schedulers duplicate queue recording, scheduler-control
reads, family choice, physical-subspan loops, leaf completion closures, and
seal metric assembly. Extract:

- a `PartitionQueues` scheduler driven by `PartitionSet`;
- a bounded `execute_reserved_pack` loop;
- one leaf-capture/completion boundary;
- shared queue metrics.

Keep mode-specific root creation, child reservation, and topology rules
explicit. A large inheritance hierarchy would obscure the statistical
difference and is not recommended.

### R8. Canonical validation and configuration helpers

`load_config` and `load_config_mapping` duplicate construction and error-path
wrapping. `TrainingStorage.__post_init__`,
`finalize_training_storage`, and learner input validation reconstruct related
schema facts in three layers. Controller `state_dict`/`from_state_dict` methods
also repeat scalar mapping mechanics.

Recommended small abstractions:

- one `load_config_mapping`, with `load_config(path)` limited to reading and
  delegating;
- a few typed number/path/enum validators used by config dataclasses;
- one schema-bound storage validation call;
- a small explicit checkpoint-field helper for controller scalar state.

Avoid a generic declarative framework or metaclass. The goal is less code and
clearer ownership, not a new internal language.

## Other concrete clone removals

The following local duplicates should be folded into the structural work rather
than assigned standalone frameworks:

- the nested `complete_leaf_rows` implementations in sequential and branched
  training construction;
- `record_queues` in `rollout.py` and `branching.py`;
- target owner/value gather helpers;
- preflop/postflop `_public_tokens` validation and assembly;
- poker backend and adapter public-batch slicing;
- MC and folded board normalization;
- `training._tensor_bytes` versus `memory_control.unique_tensor_bytes`;
- repeated controller scalar `state_dict` parsing;
- schema `require_rows` versus arena `require_capacity`, including rollback;
- `SealedDecisionInputs.gather_partition_inputs` versus
  `TrainingPartition.gather_inputs`.

Each should disappear naturally into R1–R8. Do not create one utility per
two-line duplicate.

## Recommended execution order

The cleanup should be a series of independently reviewable changes. The
cumulative line ceilings below are steering gates, not permission to weaken a
phase's tests.

### Phase 0 — freeze behavior and repair the audit roots

**Line ceiling:** unchanged, 24,126.

1. Record the accepted CLI/public/tool/variance roots in
   `tools/audit_reachability.py`.
2. Add a checked line-count command using the measurement above.
3. Repair the branched variance runner against the current partitioned storage
   API, or obtain an explicit retirement decision.
4. Capture fixed-seed rollout/topology/target/loss outputs for all three
   training modes and the probe.
5. Record the intentional branched-OOM retry policy as an accepted exception
   in the implementation test matrix, including new randomness and updated
   branching controls.
6. Add shadow-only C6 operation snapshots and reproduce the reported 44/84-row
   river evidence without allowing the new observations to affect admission.
7. Record realistic CUDA memory/throughput baselines for C2, C3, and C4 before
   structural changes, using the configurations and repetitions specified in
   those sections. The C3 baseline is current HEAD with the dense live-slab,
   direct contiguous append, and trimmed multi-page gather optimizations from
   `d95b7d3`, `5ecfeb4`, and `cf6bc34`; an older slow paged-column path is not
   an acceptable baseline.

This prevents a dead-code deletion from silently deleting an intended but
currently broken user case.

### Phase 1 — delete isolated dead and compatibility surface

**Cumulative ceiling:** 23,550.

Delete the obsolete direct policy-planning path, mutable completion path,
unkeyed stochastic fallbacks, test-only card/rule helpers, unused aliases,
inference movers, and permanently empty seat-value result fields. Update tests
to exercise canonical objects instead of aliases.

This phase should be mechanically low risk and makes later call graphs honest.

### Phase 2 — establish the unified memory authority

**Cumulative ceiling:** 23,250.

1. Introduce the C6 memory snapshot, operation signature/envelope, exact owner
   ledger, admission record, and one hard device budget.
2. Calibrate `rollout_policy_inference` against the 44/84-row river workload,
   remove the branching call to `training_transient_bytes`, and split unknown
   oversized packs to warmed safe subspans.
3. Preserve rejected-attempt node/family geometry and allocator terms through
   C1 cleanup/retry.
4. Migrate seal, completion, targets, learner, reference build, optimizer, and
   collective work to distinct envelopes.
5. Restrict the phase forecaster to between-iteration/attempt geometry and
   replace its single-node branched fit with meaningful count vectors.
6. Delete the independent poker/generic/all-phase/workspace estimates and the
   separate rollout ceiling after their last consumers migrate.

This phase precedes C2–C4 because their memory/throughput decisions are not
credible under the current false-rejection model. Before merging, pass every C6
acceptance gate, including admission of the known-safe river execution under
the near-10-GB ceiling.

### Phase 3 — align the retry exception and partition scheduler

**Cumulative ceiling:** 22,900.

1. Keep the branched-OOM retry path, but make its scope, common-rank transition,
   controller update, attempt address, observability, and exhaustion behavior
   explicit as specified in C1. Compact duplicated retry bookkeeping only where
   behavior remains identical.
2. Construct and pass one `PartitionSet` at rollout entry.
3. Replace street-specific execution families with `pre`/`post_A` integer
   queues and use `PartitionSet.scheduler_order`.
4. Add the game-owned dequeue-time minimum-street compact-width calculation
   without splitting or reordering `post_A`.
5. Recalibrate the C6 operation signatures for the new dequeue-time
   street-aware physical shapes.

Before merging, prove fixed-seed equality in successful no-OOM runs. Failure
injection must prove that retryable branched OOM releases the old attempt,
updates controls, uses a new disjoint RNG address, and retries synchronously;
exhaustion and every non-retryable failure must still stop cleanly. The C2
before/after benchmark must show the intended branched speedup without a
sequential regression outside measurement noise.

### Phase 4 — benchmark, then choose one logical occurrence owner and one seal

**Cumulative ceiling after migration:** 22,000.

1. Run the C3 baseline and arena prototypes before deleting or redirecting any
   production owner. Temporary benchmark-only prototype lines do not alter the
   final target.
2. Select the physical access domains strictly by the C3 correctness,
   throughput/memory, and synchronization gates. Do not force all fields
   through `PagedTensorArena` or any other universal container.
3. Make `occurrence_schema` (or the measured winner's equivalent immutable
   schema) the logical field authority, consolidate behind one owner that may
   use multiple measured physical access domains, and remove the separate
   node/edge owner.
4. Benchmark the C4 direct seal against the current physical layout, implement
   the vectorized one-seal path, and repeat the benchmark with the C3 winner.
5. Only after both comparisons pass, seal directly to `RolloutStorage` and
   delete `SealedRolloutSample`, both training translation seals, and every
   redundant logical arena/topology implementation. Retain specialized
   dense-slab and contiguous-span kernels selected by the benchmark.

This is the highest-risk and highest-value compactification phase. Preserve
stable logical addresses, child/action ordering, exact source/destination
accounting, and last-consumer release. No realistic case may fall below the
specified throughput floor or regress reserved memory. Do not combine it with
model refactoring.

### Phase 5 — canonical retained schema and poker structure

**Cumulative ceiling:** 21,000.

1. Remove duplicated legal ids/masks and source/destination renaming.
2. Merge the partition input/decision/action block abstractions.
3. Remove stale column aliases.
4. Update the C6 owner schema/ledger atomically with the canonical field
   migration; remove optional schema inference and duplicate tensor traversal.
5. Introduce the common prepared partition-forward boundary.
6. Consolidate sibling-evaluator setup/normalization.
7. Extract shared pre/post architecture modules while instantiating independent
   weights.
8. Migrate checkpoints explicitly and delete old parallel code in the same
   change.

Validate exact tensor storage bytes, not logical `numel` of views. Force
multi-page seals and tail physical subspans.
Benchmark end-to-end sampled spots/second and memory. An abstraction that adds
a retained full-batch intermediate is unacceptable even if it removes source
lines.

### Phase 6 — local validation/config/checkpoint cleanup

**Final ceiling:** **20,750**.

Consolidate config loading, schema validation, controller checkpoint helpers,
queue metrics, and remaining local clones. Re-run reachability and delete only
newly isolated objects whose accepted roots have been checked.

## Acceptance gates

Every production-code phase must pass CUDA validation on an idle GPU, following
the repository environment setup. At minimum:

- all unit/contract tests relevant to the changed layer;
- fixed-seed topology and tensor equality for successful normal, sequential-RB,
  branched-RB, and probe runs;
- direct `core_math.tex` target/loss equality at $\lambda=0$ and $\lambda=1$;
- one/many child, refill, zero posterior mass, and tail-subspan cases;
- sequential failure injection proving no retry, reroll, truncation, or changed
  root;
- branched-OOM injection proving whole-attempt release, synchronized retry,
  updated $\beta/\kappa/N$, a disjoint RNG attempt address, bounded retries, and
  clean stop on exhaustion;
- non-memory failure injection proving that the C1 exception is not entered;
- every C6 gate, especially the exact 44/84-row fixture, operation-type
  separation, reserved-cache `max` arithmetic, calibrated subspan splitting,
  and reconciliation of predicted versus observed allocated/reserved/device
  peaks;
- exact sealed extents and measured source-plus-destination-plus-scratch peak;
- identical partition routing with flop/turn/river mixed inside one `post_A`,
  no street sorting/splitting, and safe minimum-street compact sizing;
- the C2 branched speedup and sequential no-regression benchmark;
- the C3 realistic arena comparison, hot-path access counters, and its 99%
  per-case throughput floor;
- the C4 direct-seal comparison before and after C3, with equivalent
  `RolloutStorage` and no memory regression;
- checkpoint load/resume tests where model scaffolding or controller state
  changes;
- CUDA inference/training performance and reserved-memory comparison, plus MPS
  inference tests for changes that touch its supported boundary;
- a final non-comment/nonblank count no greater than 20,750.

Run `tools/audit_reachability.py` after every phase. Its final review list should
contain only documented dynamic boundaries; a test-only production object
requires an explicit keep decision.

## Expected end state

After the cleanup, the code should have one visible ownership chain:

`PartitionSet` routes immutable public states → one benchmark-selected
occurrence owner, governed by an immutable field schema, and partition tables
reserve logical rows → bounded prepared partition forwards fill those rows →
one measured exact seal returns `RolloutStorage` → targets add only normative
results → learner gathers bounded partition rows.

One memory coordinator should reconcile exact schema/owner capacities,
operation-specific calibrated incremental envelopes, allocator/device
snapshots, and between-attempt geometry forecasts against one hard ceiling.
Current reserved memory is a measured footprint/cache—not a transient to add
again. Poker should contribute routing, immutable inputs, natural forwards,
evaluator completion, and operation signatures without a second scheduler or
memory authority.
Sequential and branched modes should share mechanics where their storage
transactions are identical while keeping their statistical/topological rules
explicit.

That end state removes the accumulated additive layers while preserving every
accepted computation.
