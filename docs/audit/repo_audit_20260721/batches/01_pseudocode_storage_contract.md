# Pseudocode storage and implementation contract

## Scope and authority

This report extracts the implementation contract from every authoritative source
under `docs/theory/pseudocode/*.tex`. It does not assess the Python implementation.
`docs/theory/core_math.tex` remains authoritative for mathematical semantics; the
pseudocode is authoritative for scheduling, ownership, storage layout, phase
lifetimes, bounded work, memory accounting, distributed execution, and failure
behavior. The pseudocode itself states this boundary in
`storage_lifecycle.tex`, section "Authority and exactness boundary," lines 34-49.

`archive/refactor/implementation_notes.md` is historical provenance only. Its
high-level direction—CSR topology, player-node-only data, partitioned inputs,
storing generated reach instead of replaying it, and partitioned gradient
accumulation—appears at lines 10-34. Where it proposes dense or redundant fields
(notably separate advantage, posterior, value, and optional Q arrays at lines
27-33), the later pseudocode's exact schemas and deduplication rules supersede it.

Source abbreviations used below:

- `SL`: `docs/theory/pseudocode/storage_lifecycle.tex`
- `FR`: `docs/theory/pseudocode/family_scheduled_rollout.tex`
- `BT`: `docs/theory/pseudocode/backward_targets.tex`
- `DU`: `docs/theory/pseudocode/distributed_update.tex`
- `SM`: `docs/theory/pseudocode/sequential_memory_control.tex`
- `IN`: `archive/refactor/implementation_notes.md` (provenance, not authority)

## Numbered implementation invariants

### Authority, partitions, and stable row domains

**STORE-001 — Implementation transformations may not change mathematical
arguments.** Scheduling, layout, streaming, and permissible recomputation can
vary, but each production loss must receive the sampling, completion, return,
posterior, clipping, entropy, KL, and branching quantities specified by
`core_math.tex`. The storage design may not change an estimator to save memory.
(SL, "Authority and exactness boundary," lines 34-49; SL, "Old implementation
pattern versus new invariant," lines 381-387.)

**STORE-002 — Partitions are immutable, opaque core contracts.** Core receives a
finite set of immutable `PartitionSpec`s containing stable identity, pure router,
optional preorder, exact loss width $X_m$, natural model width $Y_m$, fixed
$\varphi_m$, model layout/boundary, complete byte/work declarations, safe batch
limits, and completion specification. Core may route, queue, account, sample,
and batch by partition, but may not parse names, inspect game stage, invent
subpartitions, or reorder rows using undeclared game labels. (SL, "Game-neutral
partition interface," lines 75-100; FR, Algorithm
`PartitionedQueueRollout`, lines 53-59 and 84-88.)

**STORE-003 — Exact loss and natural model domains remain distinct.** Model
inputs and outputs live directly on $\mathcal Y_m$; posterior, return, advantage,
and exact loss arrays live on $\mathcal X_m$ and access model rows only through
the immutable map $\varphi_m$. Nonlinear PPO/value terms prohibit aggregating
exact-state targets merely because model rows share a quotient. (SL, lines
51-67; DU, Algorithm `DistributedFixedRolloutUpdate`, lines 74-77; DU,
Algorithm `ModeLoss`, lines 220-248.)

**STORE-004 — Every partition has one canonical ordered exact-state domain.**
The meaning and order of $x$ cannot vary by decision, board, owner, or physical
batch. Impossible states receive zero posterior mass. A game that cannot honor
this must split the partition; core must not repair a decision-local domain.
(SL, lines 69-73.)

**STORE-005 — Poker's partition and width layout is fixed.** Poker has one
preflop partition and one postflop partition per live-player count; flop, turn,
and river with equal live-player count are not separate partitions. Preflop
model-side storage is exactly $K=R^2$ wide, postflop model-side storage is
exactly $H=\binom{4R}{2}$ wide, and RB posterior/target/loss storage remains
canonical exact-hand width $X_m=H$ for both. No staging or cache may expand
preflop model rows to $H$, and board-blocked postflop model rows remain present.
(SL, "Poker application," lines 102-115 and 152-171.)

**STORE-006 — Poker summary anchors are immutable and dependency-scoped.** A
folded-player summary is referenced, not inlined, and a new anchor is created
exactly on a fold or public-street transition. `post_P` has no summary reference
and must not materialize a neutral summary. (SL, lines 130-140.)

**STORE-007 — Completion is an opaque game boundary with a complete byte
contract.** Poker RB completion must use the sibling evaluator's binding API;
core must not locally reimplement posterior, equity, or Monte Carlo completion.
Inputs, outputs, batching, scratch, and backend workspace all count toward hard
admission. (SL, lines 142-150.)

### Ownership, append-only allocation, and sealing

**STORE-008 — Each logical row has one explicit owner and a stable integer
address.** Occurrences, decision rows, action rows, partition-input rows, leaf
rows, sequential posterior rows, compact topology, and final training storage
have the ownership and row domains enumerated in the normative ownership table.
Logical row identities are integers, never pointers into reallocatable tensors.
(SL, "Stable row domains and owning objects," lines 173-205.)

**STORE-009 — Decision and action addresses are assigned before physical
packing.** A player occurrence receives stable $(m,d)$ at creation. In branched
RB, a player edge receives stable $(m,a)$ when all logical children are reserved,
before any model pack. Reservation order aligns each action row with its
contiguous child and cannot be altered by physical splitting. (SL, lines
198-202; FR, Algorithm `ReserveChildren`, lines 231-256.)

**STORE-010 — Growing rollout storage is chunked, append-only, and schema
generated.** Complete immutable schemas define allocation, byte prediction, and
ledger validation. A row append transactionally admits the next chunk's full
capacity, including alignment and every mode-specific field; handwritten field
counts are not authoritative. (SL, lines 204-211 and 426-438.)

**STORE-011 — Child reservation is exact and transactional.** Nominal and
executed child counts are persisted before prefix sums. Exactly the executed
number of occurrences and branched action rows are reserved in stable order.
Hard-memory failure stops the iteration rather than reducing offspring count.
(FR, Algorithm `ReserveChildren`, lines 235-255.)

**STORE-012 — Sealing is a real overlapping-memory phase, not zero-copy.** After
rollout, sealing computes exact realized extents, transactionally admits source
chunk capacity plus all exact destinations plus bounded seal scratch, copies
each stable span once, validates pointers/topology/ledger after synchronization,
and only then releases source chunks. Concatenation, allocator growth, spare
logical destination rows, and claimed zero-copy transfer violate the contract.
(SL, lines 204-211; FR, Algorithm `SealRolloutStorage`, lines 195-228.)

**STORE-013 — Sealed topology is exact and compact.** Sequential topology is
decision-only CSR with one leaf per trajectory and no chance-node rows or
separate sampled-action table. Branched topology is one rooted tree with six
int32 columns, contiguous child spans, implicit $1/\mathtt{degree}$ transition
mass, action spans aligned to target-eligible child prefixes, and exactly two
protected-refill root scalars. No duplicated child index/event/transition table
is retained. (SL, "Compact topology," lines 289-325; FR, Algorithm
`SealRolloutStorage`, lines 207-225.)

**STORE-014 — Final training storage is immutable and minimal.** Finalization
validates domains, spans, references, and exactly-once writes; constructs only
the required partition sampler metadata; releases topology, leaves, completion
and frontier arenas, validity bits, public states, and scratch; and freezes the
remaining aligned blocks. Survivors are limited to immutable model inputs,
natural-width old model rows, exact targets/posteriors, and fixed sampling
metadata. (BT, Algorithm `FinalizeTrainingStorage`, lines 185-201.)

### Last-consumer retention and phase release

**STORE-015 — Generated reach is written once at its last required
destination.** Natural decision input is written when a player node is created;
exact RB leaf reach is written when the stopped leaf is created. A target builder
must neither replay discarded paths nor construct a global/root/all-node reach
memo. (SL, "Write times and lifetime schedule," lines 327-337; FR, Algorithm
`ChanceClose`, lines 302-330; IN, lines 20-25 as provenance.)

**STORE-016 — Mutable occurrence payload dies immediately after its last
child/leaf commit.** Chance-only reach and public state persist only through
descendant materialization. A processed parent releases mutable public state and
active reach/profile after all children commit; a stopped leaf releases them
after writing immutable reward/completion fields. A DONE occurrence retains only
compact topology. (SL, lines 329-335; FR, Algorithm
`PartitionedQueueRollout`, lines 157-167; FR, Algorithm `ChanceClose`, lines
318-329.)

**STORE-017 — Physical forward outputs and staging are subspan-local.** Reserve
logical addresses first, then forward, materialize, and release model output,
sampler scratch, and physical staging for each maximum-feasible physical
subspan. Whole-logical-batch outputs may not remain live while children are
created. (FR, Algorithm `PartitionedQueueRollout`, lines 143-159; SL, lines
369-370.)

**STORE-018 — Completion outputs are projected immediately.** Target builders
retain only posterior/value/moment rows required by their recurrences and scalar
diagnostics. Equity/share/stddev/valid and other completion products are released
within the completion microbatch. (BT, Algorithm `BuildSequentialRBTargets`,
lines 81-100; BT, Algorithm `BuildBranchedRBTargets`, lines 132-139.)

**STORE-019 — Cheap algebraic duplicates are derived, not stored.** Normal and
sequential RB retain one lambda return and derive value target/advantage at loss
time. Sequential completion posterior is stored once per required
trajectory/player and referenced by all its decisions. Branched node posterior
is derived as the mean of child posterior rows. No separate Q target/loss is in
the specified production storage. (SL, lines 220-281 and 371-372; BT, lines
57-59, 101-107, and 170-181.)

**STORE-020 — Branched target working memory is prebounded and contracted.** A
deterministic postorder-compatible schedule is simulated before target work to
compute exact peak live frontier slots $F_\star$. One fixed arena is allocated;
ready parents consume and release all final-child slots, retaining only a
contracted parent slot. List growth, concatenation, and an all-node
$[N,P,X]$ memo are forbidden. (BT, Algorithm `BuildBranchedRBTargets`, lines
128-169.)

**STORE-021 — Every phase has an explicit last-consumer release boundary.**
Rollout releases outputs per pack and then queues/active rows/mutable state/KKT
scratch; seal releases source chunks only after validated copy; targets release
completion results and contracted slots incrementally and then leaves/topology/
control scratch; optimization releases each graph/gather/subspan immediately and
then caches/training storage after the final step. (SL, lifetime table, lines
339-349; DU, lines 176-202.)

**STORE-022 — Release means ownership and reusable memory, not syntax.** The last
owning reference must be gone; allocator-reserved bytes and backend workspaces
are tracked separately. Deleting a Python name is not sufficient evidence of a
memory event. (SL, lines 351-357.)

### Queueing, scheduling, and sampling-law preservation

**STORE-023 — Branched rollout is exactly one rooted tree per GPU.** The unique
root is treated by ordinary randomized rounding, not special multiplicity.
Sequential modes instead have a fixed set of $G_k$ independent roots and one
child below every decision/chance node. (FR, Algorithm
`PartitionedQueueRollout`, lines 65-83 and 101-115; SL, ownership table lines
175-190.)

**STORE-024 — Logical random draws are placement-independent.** Every random
draw has a disjoint counter key based on iteration, stable logical address,
ordinal, and purpose. Queue order, physical pack size, GPU placement/speed, and
scheduling cannot change the sampled object. Physical plans preserve stable
order and may not sort or reroute rows. (FR, lines 78-88; FR, Algorithm
`Materialize`, lines 263-299.)

**STORE-025 — Queue selection follows the declared two-level policy.** Process a
full queue before admitting more sequential roots or forcing a partial drain.
Among full queues, restrict first to a minimal preorder layer, then select by
decreasing load and age. Forced partial draining first restricts to the minimal
nonempty layer, then selects by fill and age. Empty preorder means all
partitions are incomparable. (SL, lines 95-100; FR, lines 93-97 and 118-143.)

**STORE-026 — Queue membership and chance closure are total invariants.** Every
live queued player occurrence belongs to exactly one partition queue; no chance
occurrence remains pending after chance closure; a processed batch transitions
through EVALUATING to DONE. These properties must be asserted, not inferred.
(FR, lines 143-167; FR, Algorithm `ChanceClose`, lines 302-333.)

**STORE-027 — Scientific offspring suppression and hard admission are separate.**
Only the mathematically specified scientific queue safeguard may change executed
branched offspring from nominal randomized rounding, setting a player batch to
one child under its declared conditions. Hard byte admission may never alter
nominal or executed count; it fails the transaction instead. (FR, lines 60-77;
FR, Algorithm `ReserveChildren`, lines 238-246.)

**STORE-028 — Sequential admission controls concurrency, never sample count.**
`ChooseSequentialRootWidth` fixes the common $G_k$ before any current draw.
`SafeAdmit` may only limit how many fixed roots are simultaneously live; it may
not change $G_k$, discard, retry, replace, truncate, or reroll a trajectory.
(FR, lines 81-83 and 124-141; SL, lines 519-547; SM, Algorithm
`ChooseSequentialRootWidth`, lines 65-103.)

**STORE-029 — Root refill preserves a protected statistical cohort.** Branched
lower-size refill appends fresh root children with new logical ordinals and
ordinary recursive expansion, retains the original root degree and pre-refill
node boundary, rewrites final root/path masses using the final degree, excludes
refill-only children from root targets/actions, and leaves descendant decisions
ordinary and trainable. (FR, lines 170-178; FR, Algorithm
`SealRolloutStorage`, lines 213-220; SL, lines 302-319.)

### Memory accounting and controllers

**STORE-030 — Hard memory accounting uses capacity and reserved high-water
bytes.** Logical payload formulas are descriptive, not sufficient for admission.
Accounting includes chunk capacity/alignment/slack/headers, live state frontier,
queue capacity, active profiles, immutable summary anchors, exact destinations,
seal overlap, completion and target arenas, flat gradients, collective
workspace, index buffers, physical gather/activation/reduction workspaces,
allocator reservation, and NVML phase peaks. (SL, "Byte formulas," lines
389-500.)

**STORE-031 — Transient work is bounded independently of statistical work.**
Inference, chance materialization, sealing, completion, target contraction,
reference-cache construction, and learner forward/backward operate in bounded
reusable physical subspans. Statistical counts $B_m$ and logical batch sizes do
not become transient allocation sizes. Different subspans/partitions with
disjoint lifetimes are not summed. (SL, lines 460-481 and 493-499; DU, lines
78-106 and 167-195.)

**STORE-032 — Every phase and cache choice must fit a guarded hard ceiling.**
Peak memory is the maximum of rollout, seal, target, and optimization formulas,
not retained training bytes alone. Every executed training/cache-build subspan
must satisfy $M_{\rm device}-M_{\rm guard}$. The guard may be reduced only after
stable representative high-water measurements; OOM is not a calibration method.
(SL, lines 483-515.)

**STORE-033 — Sequential width prediction is lagged, phase-specific, and
conservative.** The controller uses completed prior iterations, immutable schema
bytes, exact fixed costs, one-sided realized-per-root estimates, calibrated
workspaces, and an all-reduced minimum safe width. Unsafe evidence reduces width
immediately; growth is delayed and capped; state and constraining evidence are
atomically checkpointed. No player-node or throughput target belongs in this
controller. (SL, lines 517-537; SM, Algorithms `ChooseSequentialRootWidth` and
`ObserveSequentialMemory`, lines 34-126.)

**STORE-034 — Uncached reference execution is the mandatory safety fallback.**
Sequential root feasibility excludes optional caches. Optimization first proves
an uncached singleton fits; cache selection transactionally charges final cache
destinations and build/train transients. A cache reservation/build failure
releases the partial cache and falls back to empty cache without changing targets
or draws. (SL, lines 539-547; SM, lines 60-64; DU, lines 121-145.)

### Distributed optimization and failure behavior

**STORE-035 — Distributed loss sampling is normalized once and remains
stratified.** Finalization stores partition-local $\log Z$ and alias laws;
distributed all-gather forms global stratum masses $\alpha_{g,m}$. Each positive
stratum draws an arbitrary positive integer count $B_{g,m}$ from its fixed law,
and each subspan contributes $\alpha_{g,m}/B_{g,m}$. No second importance weight,
universal shape rounding, global sort, or physical-subspan renormalization is
allowed. (DU, lines 58-77, 146-161, 167-195, and 204-210.)

**STORE-036 — Backward uses one direct flat fp32 SUM buffer.** Parameter gradient
attributes are views into a single preallocated buffer in fixed order. Each
physical subspan backpropagates directly into those views; there is no per-step
flatten/copy or per-example parameter-gradient materialization. The global
collective is SUM without division by GPU count, followed by one shared
nonfinite decision, unscale, global clip, and optimizer/loss-scale transition.
(SL, lines 375-377; DU, lines 162-200.)

**STORE-037 — Synchronous parameter ownership removes the rollout-model clone.**
$\theta_k$ stays read-only until every GPU has fixed targets and enters
optimization. An overlapped rollout/optimization pipeline would require a new
snapshot and lifetime proof and is outside this contract. (SL, lines 351-354;
FR, lines 186-191.)

**STORE-038 — Collective order and terminal failure agreement are global
invariants.** Rollout performs no model/queue collectives. Any nonrecoverable hard
transaction failure sets a local flag and stops current work without retrying a
draw; every rank reaches one common-order control agreement before targets or
mass collectives. If any rank failed, all release current stores, atomically
checkpoint the last completed iteration, and terminate. A rank may never skip a
collective entered by healthy peers. (FR, lines 89-92 and 182-191; SL, lines
549-553.)

**STORE-039 — Optional cache failure is the narrowly specified recoverable
exception.** Cache reservation/build failure is not terminal because the
uncached path was first proven safe; it releases partial cache and continues
without changing the rollout or target. Failure of the mandatory uncached
singleton, chunk, seal, completion, or required learner transaction remains a
clean checkpointed stop. (DU, lines 121-145; SL, lines 539-547.)

### Required structural and equivalence tests

**TEST-STORE-001 — Small-PBG estimator equality.** Randomized direct-formula
comparisons must cover all three pseudocode mode families, $\lambda=0$ and 1,
one and many branched children, zero posterior mass, and every production loss.
(SL, "Implementation order and acceptance gates," lines 638-640.)

**TEST-STORE-002 — Physical-plan invariance.** For one fixed sampled index
vector, unsplit and every legal multi-subspan plan—including a one-row tail and
cached versus recomputed reference—must produce the same
$\alpha_m/B_m$-normalized objective and gradient within declared arithmetic
tolerance. Statistical counts remain arbitrary positive integers. (SL, lines
641-648.)

**TEST-STORE-003 — Sealing structure and peak accounting.** Tests must force
multiple source chunks and verify exact destination extents, stable row/topology
order, contiguous child/action alignment, synchronized validation before source
release, and measured peak no greater than the declared
source-plus-destination-plus-scratch transaction. (SL, lines 649-651; FR, lines
199-227.)

**TEST-STORE-004 — Ownership/lifetime assertions.** Tests must establish that
DONE occurrences and finalized stores own no mutable public state, active reach,
queue, occurrence, topology, leaf, completion, or frontier object; every target
is written exactly once; no chance node remains queued; and discarded reach is
never replayed. (FR, lines 160-180; BT, lines 170-200.)

**TEST-STORE-005 — Width-controller and failure semantics.** Cover cold start,
immediate decrease, delayed/capped increase, all-rank minimum, checkpoint resume,
underestimated geometry, missing/inconsistent ledger evidence, and clean
pre-draw or singleton failure with no reroll. Also verify every rank follows the
same collective order on terminal failure. (SL, lines 651-654; SM, lines
68-125.)

**TEST-STORE-006 — Poker width and canonical-order contracts.** Assert all
preflop model inputs/outputs/caches use $K$, all postflop counterparts use $H$,
and RB posterior/target rows use canonical $H$ exact-hand order in both. Assert
`post_P` has no neutral folded-summary allocation and no street-specific
postflop partition exists. (SL, lines 654-661, 130-140, and 152-171.)

**TEST-STORE-007 — Root and RNG invariance.** Fixed seeds must yield identical
logical topology, events, and targets under different queue orders and legal
physical plans. Branched tests must prove one root, ordinary root rounding,
correct nominal/executed counts, protected refill prefix, final mass rewrite,
and no root action row for refill-only children. (FR, lines 65-83 and 170-180;
BT, lines 170-179.)

**TEST-STORE-008 — Phase memory and performance evidence.** Acceptance requires
allocated/reserved high-water measurements for every phase, per-partition
time/physical-byte curves, calibrated tail shapes, and fixed-seed rollout,
target, and optimization timings. Tuning follows measurements of reserved bytes,
occupancy, and end-to-end spots/second, not a universal batch quantum. (SL,
lines 631-648.)

## Internal ambiguities and potential contradictions

### AMB-001 — KKT is both current and “outside v1”

`SL` says fixed or dynamic KKT branching and frozen public $T/C$ heads are
current production requirements (lines 41-48), and `FR` accepts either fixed
$\beta$ or a checkpointed log-space KKT controller (lines 60-64). `DU`, however,
lists “adaptive branching intensity” outside production v1 (lines 46-50). This
can be reconciled only if `DU` means that the optimizer does not update branching
intensity as a learner loss while rollout still consumes an already checkpointed
KKT controller. The wording should say that explicitly; otherwise it can be read
as excluding the KKT production mode.

### AMB-002 — Three pseudocode mode families versus five external modes

`FR` and `DU` enumerate only `normal`, `seqRB`, and `branchRB` (FR lines 49-50;
DU line 45), while branched fixed-$\beta$ and KKT are parameters within
`branchRB` (FR lines 60-64). Best response is absent because these algorithms
describe rollout training and optimization, not evaluation. An implementation
audit must therefore map five externally selectable production modes onto three
storage families rather than mistakenly demand five independent storage paths.
The pseudocode does not itself define the BR storage/lifecycle contract.

### AMB-003 — Refilled-root single-prefix baseline checks final degree

In `BuildBranchedRBTargets`, target sums are explicitly over the protected prefix
$n_s^{\rm tgt}$, but the per-edge baseline selects the single-child formula when
`n=1`, where `n` is introduced as the number of final child slots (BT lines
141-158). At a refilled root it is possible in principle that
$n_s^{\rm tgt}=1<n_s$. Then the leave-one-out denominator over the protected
prefix is zero, yet `n=1` is false. Either the safeguard guarantees the protected
root cohort can never have size one when refill occurs, or the condition should
be $n_s^{\rm tgt}=1$. The present text does not state the guarantee and is
implementation-ambiguous.

### AMB-004 — The scope of “any failed hard transaction” needs its exception

`FR` broadly says any failed hard transaction sets terminal failure (lines
89-92), while `DU` makes optional cache reservation/build failure recoverable by
falling back to the proven-safe empty-cache path (lines 121-145). `STORE-039`
uses the natural narrow interpretation: required statistical work fails
terminally; optional cache construction may fail recoverably. The generic rule
should name that exception.

### AMB-005 — Queue tie-breaking is not fully deterministic

The scheduler orders eligible queues by preorder then `(load, age)` or
`(fill, age)` (SL lines 95-100; FR lines 93-97), but it provides no final
tie-breaker when these tuples are equal. Logical counter keys preserve the
sampled-object law, so this does not justify differing draws, but it can change
execution order, memory high-water behavior, and exact performance traces. A
stable partition-ID tie-break should be specified if bitwise schedule replay is
an acceptance goal.

### AMB-006 — “Complete immutable schema” lacks a single named schema version

The schema must generate allocation, prediction, and ledger validation and must
contain every field (SL lines 426-438), while controller state records a
controller version (SM lines 115-122). The documents do not explicitly require a
schema version/hash in iteration manifests or checkpoints. Without one, resume
validation across schema changes is underspecified even though immutable schema
agreement is normative.

## Provenance-to-authority disposition

The archived note correctly anticipated compact CSR topology, player-decision
and sampled-edge row domains, partitioned model inputs, elimination of reach
replay, and partition-wise gradient accumulation (IN lines 10-25 and 30-34).
The authoritative pseudocode deliberately tightens or supersedes it as follows:

- Dense `max(children_count)` topology is replaced by exact contiguous spans and
  decision-only sequential CSR (`STORE-013`).
- The note's separate advantage, posterior, value, and possible Q arrays are
  replaced by mode-specific minimal fields and loss-time derivations
  (`STORE-019`).
- Its approximate folded-summary description is replaced by immutable
  dependency anchors and zero allocation for `post_P` (`STORE-006`).
- Its broad “compute gradients, accumulate smartly” direction is replaced by
  index-only stratified draws, bounded plan/gather/backward/release subspans, and
  direct flat SUM-gradient views (`STORE-031`, `STORE-035`, `STORE-036`).

Accordingly, the archived note is useful evidence of intent but must never be
used to retain a representation that conflicts with the later exact schemas,
last-consumer lifecycle, or bounded-transaction rules above.
