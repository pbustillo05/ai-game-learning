# Rollout, storage, topology, and queue audit

## Scope and method

This batch reviewed the production paths in:

- `pbg/core/execution.py`
- `pbg/core/storage.py`
- `pbg/core/rollout.py`
- `pbg/core/branching.py`
- `pbg/core/tree.py`
- `pbg/core/topology.py`
- `pbg/core/memory_control.py`
- `pbg/core/completion.py`
- `pbg/core/partition.py`
- `pbg/core/adaptive_branching.py`

It also traced the immediate sealing consumers in `pbg/core/training.py` and
reviewed the directly relevant tests. The baseline contracts are batch 00
(`MATH-*`) and batch 01 (`STORE-*`, `TEST-STORE-*`), with historical regression
IDs from batch 02 where applicable. This was a static, read-only audit: no tests
or GPU jobs were run.

## Executive result

The one-root law, ordinary randomized rounding at the root, independently keyed
sibling sampling, nominal/executed count retention, protected-root refill, KKT
primitives, and the lagged four-phase sequential width controller are all
recognizable and substantially preserved.

The rollout/storage path is nevertheless **not faithful to the authoritative
storage pseudocode**. Its central representation is still a larger legacy
`SampledTree`. Rollout materializes that full node/edge representation and only
later converts it into `SequentialDecisionCSR` or `BranchedTreeSpan`. That is
the exact post-hoc conversion pattern forbidden by the storage authority. The
same path expands preflop model outputs to exact-hand width and compresses them
afterward, infers source schemas from the first batch instead of allocating from
the immutable schemas already present in `storage.py`, and performs branched
child waves without bounded physical subspans.

Queue scheduling and memory admission also have independent correctness gaps:
forced sequential drains ignore the minimum preorder layer, branched scheduling
does not prioritize full queues globally, queue tensors are outside the byte
ledger, and fixed-beta mode cannot configure the scientific queue safeguard.

## Findings

### AUD-RSQ-001 — Production seals a larger legacy tree before building the normative compact topology

- Severity: **blocker**
- Confidence: **confirmed**
- Contract: `STORE-012`, `STORE-013`, `STORE-014`, `STORE-021`, `STORE-030`
- Modes affected: `normal`, `sequential_rb`, branched fixed beta, branched KKT
- Evidence:
  - `pbg/core/tree.py:355-389`, `PagedTreeSource.node_schema` and
    `edge_schema`, retain full node rows (`tree_id`, owner, state, two depth
    arrays, path weight, child count) plus a separate five-column edge table.
  - `pbg/core/tree.py:555-587`, `PagedTreeSource.finalize`, gathers every paged
    node and edge column, allocates full CSR edge tensors, and returns
    `SampledTree`.
  - `pbg/core/rollout.py:879-952` and `pbg/core/branching.py:888-1013` finalize
    and retain this `SampledTree` plus separate policy/leaf/input stores.
  - `pbg/core/training.py:1725-1739` subsequently constructs
    `SequentialDecisionCSR`; `pbg/core/training.py:1765-1826` allocates reorder
    maps and constructs `BranchedTreeSpan` from the already sealed larger tree.
  - `pbg/core/topology.py:17-149` and `153-293` show that the required compact
    representations already exist but are downstream conversion products.
- Current behavior: rollout seals a compatibility representation containing
  chance nodes, state/depth/path fields, explicit child/event/weight arrays, and
  redundant edge ownership. A second phase then sorts and converts it into the
  actual target topology.
- Required behavior: seal directly from stable rollout addresses into
  decision-only sequential CSR or the exact six-int32 branched span, with only
  the two refill scalars and required nominal-count data. No larger legacy
  representation may survive as an intermediate API.
- Consequence: excess source and destination memory, extra all-node gathers and
  sorts, an unproved peak during conversion, and continued architectural
  dependence on a representation explicitly superseded by the pseudocode.
- Recommended disposition: replace `SampledTree` as the production rollout
  handoff; seal the normative topology directly, migrate all target/evaluation
  consumers, then delete or archive the legacy tree path.
- Verification needed: multi-page direct-seal tests proving exact extents,
  stable row order, no chance-node rows in sequential topology, six-column
  branched topology, and absence of any full legacy tree allocation.

### AUD-RSQ-002 — The full-tree seal and post-hoc conversion are outside hard byte admission

- Severity: **blocker**
- Confidence: **confirmed**
- Contract: `STORE-012`, `STORE-030`, `STORE-032`, `HIST-OOM-01`
- Modes affected: all four training storage families in scope
- Evidence:
  - `pbg/core/tree.py:555-587`, `PagedTreeSource.finalize`, allocates full node
    gathers, full edge gathers, `child_ptr`, three reordered edge destinations,
    and a `SampledTree` without a `ByteAccountant` reservation or bounded seal
    scratch.
  - `pbg/core/rollout.py:879-884` and `pbg/core/branching.py:888-909` create the
    destination while paged sources are still owned; neither performs the
    source-plus-destination transaction required by `STORE-012`.
  - `pbg/core/training.py:1765-1808` then allocates `old_order`, `old_to_new`,
    parent, degree, child-begin, decision, and leaf maps proportional to the
    whole branched tree.
  - `pbg/core/training.py:636-662` analytically charges the normative sealed
    topology, not the larger `SampledTree` plus the simultaneous conversion
    tensors actually materialized.
- Current behavior: page growth is admitted, but the largest sealing and
  conversion allocations bypass the page ledger. GPU acceptance compares
  coarse allocator peaks to an analytical plan; it does not make these
  allocations transactional.
- Required behavior: reserve source capacity plus exact compact destinations
  plus bounded scratch before copying, validate after synchronization, then
  release sources. Any mandatory failure must enter terminal failure agreement
  without reroll.
- Consequence: the nominal execution-memory ceiling is not a hard ceiling and
  large trees may OOM precisely at sealing, despite successful child-page
  admission.
- Recommended disposition: fold direct compact sealing into the shared ledger
  and remove all whole-tree conversion scratch.
- Verification needed: force multiple pages under a deliberately tight byte
  ceiling and prove a clean pre-allocation stop or a successful measured peak no
  larger than the declared overlap transaction.

### AUD-RSQ-003 — Rollout schemas are inferred or handwritten instead of generated from the immutable schemas

- Severity: **high**
- Confidence: **confirmed**
- Contract: `STORE-008`, `STORE-010`, `STORE-011`
- Modes affected: all training modes in scope
- Evidence:
  - `pbg/core/storage.py:640-689`, `PagedRowAccumulator`, creates its schema from
    the first realized dictionary of tensors (“schema-on-first-write”).
  - Production constructs this owner at `pbg/core/rollout.py:437-443` and
    `pbg/core/branching.py:272-292` for policy, action, and leaf streams.
  - `pbg/core/execution.py:967-983`, `arena_fields`, assembles a separate
    handwritten occurrence layout used at `pbg/core/rollout.py:333-338` and
    `pbg/core/branching.py:196-205`.
  - `pbg/core/storage.py:1384-1425`, the nominal `occurrence_schema`, is not
    referenced anywhere in production.
  - The authoritative source schemas do exist at
    `pbg/core/storage.py:1448-1684`, but rollout does not instantiate them.
- Current behavior: the first physical batch decides which fields exist. A
  later batch can fail after draws if its fields differ, and pre-admission
  cannot be generated from the exact eventual source schema.
- Required behavior: mode/partition schemas must be immutable before rollout;
  the same schema must generate allocation, byte forecasts, validation, and
  ledger evidence.
- Consequence: schema drift can silently omit fields (notably T/C), while byte
  planning and actual allocation can describe different objects.
- Recommended disposition: delete `PagedRowAccumulator` from production and
  allocate explicit schema-bound arenas per partition/mode before sampling.
- Verification needed: assert schema identity/version before the first draw and
  demonstrate that every source page reservation equals the corresponding
  immutable schema capacity.

### AUD-RSQ-004 — Preflop model rows are expanded to exact-hand width and compressed after rollout

- Severity: **high**
- Confidence: **confirmed**
- Contract: `STORE-003`, `STORE-005`, `STORE-014`, `STORE-031`
- Modes affected: `sequential_rb`, branched fixed beta, branched KKT; normal
  shares the same `x_max` carrier
- Evidence:
  - `pbg/core/rollout.py:319-320`, `398-410`, and `918-932` choose global
    `x_max` and retain all old value/log-probability rows at that width.
  - `pbg/core/branching.py:151-153`, `215-238`, `956-960`, and `989-997` do the
    same for branched reach and policy stores.
  - `pbg/core/training.py:1896-1899` and `1911-1914` explicitly call
    `_compress_preflop_rows`; `pbg/core/training.py:2030-2040` selects one
    representative exact hand per preflop class after the larger row already
    existed.
- Current behavior: model-side preflop rows are retained at $H$ and projected
  back to $K=R^2$ during post-hoc sealing.
- Required behavior: preflop model inputs, outputs, rollout caches, and learner
  caches must live at natural width $K$ from creation; only exact posterior and
  target/loss rows use $H$.
- Consequence: for a full deck, each retained preflop row uses 1,326 values
  where 169 are permitted, inflating rollout, sealing, and transient bytes and
  obscuring exact/model-domain correctness.
- Recommended disposition: reserve stable partition-local decision/action rows
  before forward and write natural-width outputs directly; use immutable
  `exact_to_model` only at exact-loss consumption.
- Verification needed: allocation-hook tests that reject every preflop
  model-side tensor whose trailing width is $H$.

### AUD-RSQ-005 — Branched child materialization is proportional to logical offspring, not bounded physical work

- Severity: **high**
- Confidence: **confirmed**
- Contract: `STORE-009`, `STORE-017`, `STORE-031`
- Modes affected: branched fixed beta, branched KKT
- Evidence:
  - `pbg/core/branching.py:551-585` creates `repeat_interleave`, parent, state,
    reach, path, profile, and indexed model-output tensors for all `total`
    children in one operation.
  - `pbg/core/branching.py:586-646` samples all events, steps all states, updates
    all reach, appends all topology, and allocates all live reach rows as one
    wave.
  - `pbg/core/branching.py:662-681` computes and retains all edge log
    probabilities in the same logical wave.
  - No `PhysicalPlan`, partition `ByteWorkContract.max_rows`, or reusable
    staging subspan participates in `expand`; `ReusableStagingSlabs` at
    `pbg/core/execution.py:782-823` is test-only.
- Current behavior: `family_batch_rows` bounds parent count, but beta multiplies
  it into an unbounded child wave; if the config leaves it unset, the entire
  queue is also one parent batch (`pbg/core/branching.py:819-823`).
- Required behavior: reserve every logical child/action address first, then
  forward/sample/materialize stable maximum-feasible physical subspans and
  release each subspan's outputs and scratch immediately.
- Consequence: increasing beta or queue depth scales transient GPU memory with
  statistical work and can OOM after page admission even when retained storage
  itself fits.
- Recommended disposition: implement partition-supplied physical plans and a
  bounded materialization loop over already-reserved child/action spans.
- Verification needed: fixed logical topology under multiple physical plans,
  including one-row tails, with identical events/targets and a constant
  transient high-water bound.

### AUD-RSQ-006 — Child and action logical addresses are assigned after model work and random draws

- Severity: **high**
- Confidence: **confirmed**
- Contract: `STORE-009`, `STORE-011`, `STORE-024`
- Modes affected: branched fixed beta, branched KKT
- Evidence:
  - `pbg/core/branching.py:500` admits page capacity only; no logical node,
    occurrence, or action rows are reserved there.
  - Profiles, events, successor states, and reach are generated at
    `pbg/core/branching.py:551-610`.
  - Stable topology rows are assigned by `buffer.append_children` only at
    `pbg/core/branching.py:621-623`; occurrence and reach handles follow at
    `637-646`.
  - Action rows are appended still later at `pbg/core/branching.py:672-681`.
- Current behavior: parent-local ordinals provide deterministic RNG keys, so
  observed draws are presently reproducible, but the normative stable row
  addresses do not exist before physical work.
- Required behavior: persist nominal/executed counts, prefix-sum, and reserve
  exactly all occurrence and action row IDs before any physical pack.
- Consequence: physical splitting cannot use the stable output addresses as its
  invariant, and a failure after draws but before row publication is not the
  transaction defined by the pseudocode.
- Recommended disposition: introduce an explicit `ReserveChildren` transaction
  and make materialization write only into its reserved spans.
- Verification needed: failure injection at every allocation boundary proving
  no draw occurs before full logical reservation and no partial rows publish.

### AUD-RSQ-007 — Queue scheduling differs from the declared two-level policy

- Severity: **high**
- Confidence: **confirmed**
- Contract: `STORE-025`, `STORE-024`
- Modes affected: `normal`, `sequential_rb`, branched fixed beta, branched KKT
- Evidence:
  - Sequential full-queue selection correctly restricts to the minimum layer at
    `pbg/core/rollout.py:682-702`.
  - Sequential forced partial drain instead maximizes fill across *all*
    nonempty queues and only uses priority as a later tiebreak at
    `pbg/core/rollout.py:740-752`.
  - Branched drain always restricts to the minimum layer and largest depth at
    `pbg/core/branching.py:797-823`; it never first checks whether any queue in
    any layer is full.
- Current behavior: a fuller downstream sequential queue can beat a partial
  upstream queue during forced drain, and a partial upstream branched queue can
  be processed while a downstream queue is full.
- Required behavior: globally process a full queue before admitting/forcing;
  among full queues restrict to the minimum preorder layer then load/age. For a
  forced partial drain, restrict to the minimum nonempty layer before fill/age.
- Consequence: execution order, occupancy, queue high-water behavior, and
  performance differ from the authoritative schedule. Keyed RNG limits
  scientific-law damage but does not repair ownership or memory behavior.
- Recommended disposition: centralize the scheduler comparator and use it in
  both sequential and branched paths.
- Verification needed: adversarial multi-layer queue tests with upstream
  partial/downstream full and upstream low-fill/downstream high-fill cases.

### AUD-RSQ-008 — Queue storage and queue concatenation bypass byte accounting

- Severity: **high**
- Confidence: **confirmed**
- Contract: `STORE-030`, `STORE-031`, `STORE-032`
- Modes affected: all training modes in scope
- Evidence:
  - `pbg/core/execution.py:596-621`, `DeviceWorkQueue`, owns a Python deque of
    independently allocated tensors and computes `reserved_bytes` from only
    currently live payload plus eight bytes; it has no accountant.
  - `pbg/core/execution.py:660-663` clones every enqueue tensor.
  - `pbg/core/execution.py:671-684`, `peek`, builds a list and may `torch.cat`
    the whole selected logical batch before dequeue.
  - `pbg/core/branching.py:258-262` converts the execution byte limit into a row
    limit by dividing by roughly eight bytes per family, without charging the
    allocations to `ByteAccountant`.
  - Sequential `_admission_plan` samples only current queue payload at
    `pbg/core/rollout.py:211-215`; subsequent clones/cats are not hard
    reservations.
- Current behavior: queue insertion is bounded by a row count but its storage,
  allocator slack, deque fragmentation, and dequeue concatenation are outside
  hard admission.
- Required behavior: queue capacity and all queue/staging transactions must be
  schema-accounted; transient dequeue work must use a bounded reusable slab.
- Consequence: the ledger can report headroom while queue tensors consume more
  memory, and fragmented enqueue batches can create an unbounded concatenation
  transient.
- Recommended disposition: replace deque-owned tensor clones with one admitted
  stable queue arena/ring per partition and bounded prefix staging.
- Verification needed: fragmented enqueue/dequeue tests that measure allocator
  peaks and prove they remain within the declared queue transaction.

### AUD-RSQ-009 — The scientific queue safeguard is unavailable in fixed-beta mode

- Severity: **high**
- Confidence: **confirmed**
- Contract: `MATH-BRANCH-009`, `MATH-FIXED-001`, `STORE-027`, `HIST-OOM-01`
- Modes affected: branched fixed beta
- Evidence:
  - `pbg/core/branching.py:297-307` permits `adaptive_kkt` only for dynamic
    branching.
  - The threshold is read only as `adaptive_cfg.safe_tree_size`, and suppression
    is gated on `dynamic_active` at `pbg/core/branching.py:474-481`.
  - During dynamic warmup, `dynamic_active` is false, so the safeguard is also
    disabled even though rollout is using fixed intensity.
  - `pbg/core/config.py:350-371` likewise exposes `safe_tree_size` only inside
    dynamic KKT configuration.
- Current behavior: fixed-beta and dynamic warmup always execute nominal
  offspring, regardless of the optional one-child queue forecast threshold.
- Required behavior: fixed and KKT geometry both accept the scientific
  $M_{\tt safe}$ rule; suppression remains distinct from hard byte admission and
  retains nominal count.
- Consequence: the fixed production mode lacks one of the requested large-tree
  safeguards and is more exposed to explosive queues; this is a historical
  safeguard regression/gap rather than a mere tuning omission.
- Recommended disposition: move the scientific threshold to the common
  branched configuration and apply it to player batches in both regimes.
- Verification needed: fixed-beta and warmup tests proving batch-local
  suppression, nominal/executed retention, recovery after pressure falls, and
  corrected T/C recurrences.

### AUD-RSQ-010 — Occurrence lifecycle assertions are incomplete and root owners outlive their last consumer

- Severity: **medium**
- Confidence: **confirmed**
- Contract: `STORE-016`, `STORE-021`, `STORE-022`, `STORE-026`, `TEST-STORE-004`
- Modes affected: all training modes in scope
- Evidence:
  - Only queued player occurrences transition `NEW -> QUEUED -> EVALUATING ->
    DONE`; `DeviceWorkQueue` enforces that path at
    `pbg/core/execution.py:653-714`.
  - Chance and stopped-leaf occurrences are routed directly at
    `pbg/core/rollout.py:531-669` and `pbg/core/branching.py:711-795`; they remain
    `NEW` until the arena is released. No total assertion proves every live
    occurrence is queued or fully consumed.
  - Branched non-root reaches/states are released at
    `pbg/core/branching.py:701-708`, but root reach/state are retained until all
    ordinary work and optional refill finish at `880-887`, even when
    `branching_min_nodes is None` and the root has no future consumer.
  - `SampledTree.state_ref` remains part of the sealed compatibility object
    (`pbg/core/tree.py:575-587`) although most state owners have already been
    released and replaced by `-1`.
- Current behavior: important player transitions and backend recycling checks
  exist, but the normative total membership/DONE invariant is not represented
  for chance/leaves, and the compatibility tree retains dead state handles.
- Required behavior: every occurrence kind must have an asserted terminal
  state and own no mutable payload after its last commit; root owners should be
  retained only if lower-size refill is enabled.
- Consequence: lifecycle bugs in chance closure can escape status validation,
  and the interface continues to imply ownership of dead public-state handles.
- Recommended disposition: make terminal status total across occurrence kinds
  and remove `state_ref` from the sealed handoff.
- Verification needed: ownership tests inspecting every occurrence kind and
  backend arena after ordinary rollout, refill rollout, and injected failure.

### AUD-RSQ-011 — Fixed-beta rollout does not require the T/C heads it is required to train

- Severity: **medium**
- Confidence: **strong**
- Contract: `MATH-FIXED-002`, `MATH-BRANCH-008`, `STORE-001`, `STORE-010`
- Modes affected: branched fixed beta and dynamic warmup
- Evidence:
  - `pbg/core/branching.py:683-699` appends `old_t`/`old_c` only if both happen
    to be present; it does not reject a fixed-mode output with neither head.
  - Because `PagedRowAccumulator` infers its schema, the omission becomes the
    sealed schema rather than an immediate contract error.
  - `pbg/core/training.py:1863-1867` rejects only the case where exactly one head
    is present; if both are absent it builds a branched store without them.
  - `pbg/core/training.py:1903-1908` then declares `dynamic_heads=False` based on
    the realized columns.
- Current behavior: the standard poker model may currently emit both heads, but
  core explicitly supports a fixed production path that silently omits their
  rollout values and target/loss fields.
- Required behavior: all branched production rollouts train public T/C from the
  first update, even while fixed beta determines geometry.
- Consequence: a model/config drift can disable required T/C learning without a
  hard failure and leave KKT warmup without trained heads.
- Recommended disposition: make T/C fields unconditional in the branched
  immutable schemas and fail the first model boundary that omits them.
- Verification needed: fixed-beta negative test for missing heads and a positive
  test proving nonzero T/C target/loss construction from iteration zero.

### AUD-RSQ-012 — Legacy/test-only CPU and compatibility objects remain in active core modules

- Severity: **medium**
- Confidence: **confirmed** for reachability; disposition of old completion API
  is **tentative** pending the BR/evaluation audit
- Contract: repository production-only requirement; `STORE-013`, `STORE-014`
- Modes affected: repository hygiene, not estimator output
- Evidence:
  - `pbg/core/tree.py:240-351`, `DiagnosticTreeBuilder`, is referenced only by
    tests and variance diagnostics; it constructs Python lists and CPU tensors.
  - `pbg/core/tree.py:594-680`, `AliasTable` and `build_decision_alias`, are
    referenced only by tests. They retain NumPy-generator and `.numpy()` CPU
    branches (`645-670`) and are not used by production sampling metadata.
  - `pbg/core/execution.py:782-823`, `ReusableStagingSlabs`, is referenced only
    by tests despite the production path needing such bounded slabs.
  - `pbg/core/completion.py:18-26`, `CompletionStats`, has no reference outside
    its own module; `BatchedCompletionStats.row` exists only to construct it.
  - `pbg/core/storage.py:1384-1425`, `occurrence_schema`, is dead while a
    separate handwritten occurrence schema is active.
  - `pbg/core/completion.py:144-177`, `ReachView` and `MCCompletion.complete`,
    feed the older tree-aware completion API used by poker evaluation/tests;
    production training uses `CompletionSpec` plus `complete_sealed` instead.
- Current behavior: active core exposes two tree construction/sampling APIs,
  two completion boundaries, a dead normative occurrence schema, and CPU-only
  alias machinery.
- Required behavior: only objects reachable from the five NVIDIA production
  modes remain active; diagnostics belong under evaluation or `archive/`.
- Consequence: legacy representations continue to look authoritative and make
  accidental CPU or compatibility-path reuse likely.
- Recommended disposition: delete `CompletionStats` and dead aliases; move
  diagnostic builders/alias utilities to evaluation or archive; resolve the old
  `MCCompletion` boundary after the BR audit; make `occurrence_schema` the sole
  implementation rather than deleting the authority-aligned schema.
- Verification needed: symbol-level reachability from all five production entry
  points after migration.

## Preserved contracts and strengths

- **One rooted branched tree and ordinary root rounding:**
  `pbg/core/branching.py:148-149` requires one root; root enters the same queue
  and `offspring_for` path as other player nodes (`438-508`). Fractional rounding
  uses logical keys. Tests cover this at
  `tests/test_c09_labeled_branching.py:253-268`, `334-341`, and
  `tests/test_refactor_storage_targets.py:575-582`.
- **Independent labeled children:** `pbg/core/branching.py:563-601` gives each
  parent-local ordinal a distinct posterior and event key and samples profiles
  per child. Duplicate public events are retained as distinct nodes.
- **Nominal versus executed counts:** `pbg/core/branching.py:472-508` persists
  nominal count before applying dynamic suppression; the count is transferred
  to compact topology at `pbg/core/training.py:1821-1825` and decision storage
  at `1849-1862`.
- **Protected lower-size refill:** `pbg/core/branching.py:837-900` preserves the
  original child prefix/node boundary, recursively expands each added child,
  rewrites final root/path mass, and does not overwrite the original nominal
  root count. Root action filtering at `941-980` excludes refill-only edges.
  This strongly preserves `MATH-BRANCH-010`, `STORE-029`, and
  `HIST-SMALL-01`, subject to the storage-representation findings above.
- **KKT primitives and checkpoint state:** `pbg/core/adaptive_branching.py:52-209`
  preserves randomized rounding, log-domain intensity, ancestor recurrences,
  damped controller update, and versioned state. This agrees with
  `HIST-KKT-01`.
- **Generated leaf reach at the last rollout consumer:** sequential
  `pbg/core/rollout.py:565-610` and branched
  `pbg/core/branching.py:766-787` clone the leaf reach into an immutable leaf
  stream; target code need not replay discarded public paths.
- **Page-level child admission and rollback:**
  `pbg/core/branching.py:351-380` admits tree, occurrence, controller, reach,
  successor-state, and history pages before child sampling and rolls empty page
  growth back on failure. This is valuable but does not cover the unbounded
  transients or seal destinations in findings 002 and 005.
- **Sequential sample-count stability:** `TrajectoryAdmissionPlan.safe_admit`
  (`pbg/core/execution.py:150-217`) delays starts without changing
  `roots_per_gpu`; `_sample_trees_sequential` checks final admission/accounting
  at `pbg/core/rollout.py:885-888`.
- **Lagged four-phase width controller:**
  `pbg/core/memory_control.py:228-579` uses completed observations, immediate
  conservative widening/decrease, delayed capped growth, rank minimum, a hard
  guard, and versioned checkpoint state. Focused tests in
  `tests/test_refactor_memory_width.py` cover the principal controller rules.
- **Completion projection:** `BatchedCompletionStats.for_learning` at
  `pbg/core/completion.py:49-127` drops equity, validity, sample-count, and
  standard-error tensors after retaining scalar diagnostics and learning-only
  posterior/value rows.

## Test coverage assessment

Existing tests strongly cover isolated page transactions/sealing, queue status
transitions, logical RNG primitives, one-root branching, duplicate siblings,
fractional root rounding, protected refill, dynamic safeguard correction,
compact target formulas, natural/exact target widths, and the sequential width
controller.

The suite does **not** establish the following required structural properties:

- direct sealing into the normative compact topology without a `SampledTree`;
- one hard ledger transaction spanning real source pages, real final
  destinations, and bounded scratch;
- absence of preflop $H$-wide model-side rollout tensors;
- physical-plan invariance for branched child materialization;
- the two adversarial queue-policy cases in `AUD-RSQ-007`;
- byte-accounted fragmented queue storage/dequeue staging;
- fixed-beta or dynamic-warmup scientific queue suppression;
- total DONE/membership/lifetime assertions for chance and leaf occurrences;
- unconditional fixed-beta T/C heads and first-update losses;
- fixed-seed identity across different queue orders and all legal physical
  plans (`TEST-STORE-007`).

Tiny GPU acceptance tests report measured phase peaks, but they cannot prove
that unaccounted tree sealing or conversion will remain below the hard ceiling
for production-sized trees. No test makes allocation success conditional on the
same `ByteAccountant` ledger that it later reports.

## Reviewed-object inventory

### Production-reachable and reviewed

- Execution/admission: `ByteAccountant`, `TrajectoryAdmissionPlan`,
  `ChunkedOccurrenceArena`, `DeviceWorkQueue`, `stable_partition_masked`,
  logical RNG/key functions.
- Storage: `TensorFieldSchema`, `TensorSchema`, `SealedTensorBlock`,
  `PagedTensorArena`, `PagedRowAccumulator`, grouped arena admission,
  `RecyclableTensorPool`, rollout/training storage DTOs, all schema factories.
- Rollout/tree: sequential sampler/dispatcher, branched sampler and KKT/refill
  path, `PagedTreeSource`, `SampledTree`, sealed decision/leaf DTOs.
- Normative topology: `SequentialDecisionCSR`, `BranchedTreeSpan`.
- Memory: alias-aware measurement, four-phase forecaster, sequential memory
  controller and checkpoint state.
- Boundaries: `ByteWorkContract`, `CompletionSpec`, `PartitionSpec`,
  `PartitionSet`, `BatchedCompletionStats`.

### Not production-reachable in the four training modes reviewed

- `DiagnosticTreeBuilder`
- `AliasTable` and `build_decision_alias`
- `ReusableStagingSlabs`
- `CompletionStats` and `BatchedCompletionStats.row`
- `occurrence_schema` (dead despite describing the desired schema)
- `BranchingCutoffReached` / `BranchingDiagnosticLimits` are
  benchmark/evaluation-only by design and should not remain in the production
  branching module once surfaces are separated.

### Requires cross-batch disposition

- `ReachView`, `MCCompletion.complete`, and poker's older tree-aware completion
  method are not used by production training but may still be reached by
  evaluation tooling. The BR/evaluation batch should decide whether that is one
  of the five core modes or should move to archive.
- `SampledTree`, `RolloutDecisionRows`, and `StoppedLeafRows` are currently
  production-reachable, but findings 001-004 show they are compatibility owners
  that must disappear from the final production architecture rather than be
  preserved merely because current code imports them.

## Uncertainties and audit boundaries

- The exact mathematical T/C backward recurrences are audited by the target/loss
  batch; this report checked only that rollout retains nominal/executed counts,
  protected metadata, and frozen heads.
- Poker backend state/history page schemas were inspected only through the core
  admission calls. Completeness of backend/evaluator workspace accounting belongs
  to the poker-backend batch.
- `PartitionSpec` itself matches the opaque immutable interface well, but the
  production rollout scheduler still routes through backend family names/values
  rather than consuming the `PartitionSet` directly. The poker partition audit
  should determine whether those families are exactly the immutable partition
  identities or a parallel routing abstraction.
- This pass did not reproduce historical runs or run CUDA profiling. Findings
  002, 004, 005, and 008 are algorithmic memory/performance regressions or risks
  visible directly from allocation shape; their measured magnitude remains to
  be benchmarked.
