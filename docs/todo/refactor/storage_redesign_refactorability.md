# Dual-engine storage redesign and repository compactification plan

Status: implementation plan; not an authoritative production contract.

Date: 2026-07-24.

Related design: `docs/todo/storage_redesign.md`.

Related audit: `docs/audit/pbg_redundancy_compactification_audit_20260723.md`.

## 1. Executive decision

The storage redesign and repository cleanup should be one project.

During implementation, both the current and redesigned engines must be usable. This is a deliberate, temporary
migration period, not a permanent compatibility promise. The two implementations must be isolated behind one
composition-root selector so that completing the migration requires deleting one dependency closure, not removing
mode checks scattered throughout the repository.

Build the replacement under:

```text
pbg/core/queued/
```

Keep the current data path in its existing top-level `pbg/core/*.py` modules while migration is active. Select the
implementation once:

```text
implementation = legacy | queued
```

After selection, rollout, storage, targets, learner data, and poker transition types must never cross between the two
engines. There must be no silent fallback from `queued` to `legacy`.

At cutover:

1. make `queued` satisfy the complete production acceptance matrix;
2. remove the selector and the `legacy` configuration value;
3. delete the entire legacy data-path closure, including adapters, tests, configuration, tools, and documentation;
4. retain only genuinely shared runtime/controller code and `pbg/core/queued/`;
5. run a repository-wide reachability, duplication, configuration, and line-count audit;
6. consolidate any redundancy found inside the new implementation before declaring the project complete.

The final production tree must be smaller than the pre-project tree. Temporary growth is acceptable only when every
new component has a named legacy deletion credit and the projected post-cutover tree remains below the final budget.

## 2. What “100% used” means

Not every repository file should execute in production. Tests, benchmarks, audits, launchers, and documentation are
legitimate non-production assets. The enforceable goal is:

> Every file has an explicit role, and every object under `pbg/` is reachable from an accepted production/public root
> or is covered by a small, reviewed dynamic-boundary allowlist.

Tests or documentation alone are not sufficient reasons to keep an object in `pbg/`. A useful test-only oracle
belongs under `tests/`. A useful developer-only helper belongs under `tools/`.

The final state must have:

- no production module retained only by a test;
- no deprecated alias, fallback path, or unused configuration field;
- no `legacy`, `compat`, `v2`, `new`, or equivalent migration surface;
- no duplicate implementation of the same semantic operation;
- no class whose only purpose is wrapping another object without owning an invariant or lifetime;
- no runtime selector between old and new training engines;
- no stale variance, benchmark, BR, inference, or distributed consumer using an obsolete representation;
- no undocumented dynamic reachability exception.

Static analysis cannot prove semantic usefulness in dynamic Python. The proof therefore combines explicit roots,
static reachability, migration-ledger disposition, production-workflow coverage, call-site searches, schema/lifetime
tracing, and manual review of dynamic boundaries.

## 3. Baseline and budgets

The existing compactification audit measured 24,126 non-comment, nonblank Python lines under `pbg/` in 59 files.
Its eight largest modules contained 54.8% of those lines:

| Module | Counted lines in the audit |
|---|---:|
| `pbg/core/training.py` | 3,351 |
| `pbg/games/poker/rollout_backend.py` | 2,718 |
| `pbg/core/storage.py` | 1,656 |
| `pbg/core/learner.py` | 1,412 |
| `pbg/core/branching.py` | 1,239 |
| `pbg/core/rollout.py` | 976 |
| `pbg/core/targets.py` | 948 |
| `pbg/core/execution.py` | 917 |

Those are exactly the modules most affected by the new ownership design. A clean implementation should remove more
code than a local allocator refactor.

The audit established 20,750 counted lines as a credible hard ceiling without removing supported behavior. This
project adopts that as a maximum, not an aspiration. The expected result should be lower if final-address storage,
one seal, exact ragged actions, one memory authority, and one poker model-preparation boundary eliminate the intended
layers.

At project start, refresh the line baseline because the working tree has evolved. Add a checked line-count tool using
the audit's token-based rule. Track these values after every change:

```text
current_total
temporary_legacy_lines
new_queued_lines
shared_lines
committed_legacy_deletion
projected_final_total
```

The important value during dual implementation is:

$$
\text{projected final}
=\text{current total}
-\text{committed legacy deletion}
+\text{new queued code}
+\text{new shared code}.
$$

Each new module must name the old modules, symbols, or responsibilities it will replace. A change that adds an
abstraction without deleting or consolidating a named responsibility creates deletion debt and is incomplete.

Line count is a guardrail, not a minification contest. Do not satisfy it by moving production code out of `pbg/`,
generating opaque code, weakening validation, compressing readable statements, or hiding duplication behind dynamic
dispatch.

## 4. Temporary dual-engine architecture

### 4.1 One selector, independent internals

The temporary implementation selector belongs at the external composition boundary used by training, benchmarks,
and BR/probe workflows. Resolve the implementation once to a small set of top-level callables. Do not introduce a
large inheritance hierarchy merely to make the two engines look alike.

Conceptually:

```text
resolve_training_implementation(name)
  legacy -> existing production entry points
  queued -> pbg.core.queued entry points
```

If several workflows require resolution, one resolver module may return a small immutable entry-point table. This is
a justified object because it owns the temporary engine boundary. It is deleted at cutover.

The common boundary should contain only external inputs and results:

- immutable run configuration;
- model/bundle builders;
- checkpoint initialization/resume inputs;
- run metrics and checkpoint outputs;
- terminal failure classification.

It must not standardize an intermediate rollout or learner-storage object. A compatibility DTO shared by both
engines would force the redesign back into the old representation and make deletion harder.

### 4.2 Both versions remain genuinely usable

While migration is active:

- `legacy` remains behaviorally unchanged except for deletion of proven-stale surface and carefully reviewed
  extraction of genuinely shared utilities;
- `queued` fails clearly for a not-yet-implemented axis combination and never dispatches that combination to legacy;
- checkpoints record which implementation produced the run;
- parity and performance runs execute the implementations separately with matched configuration and logical seeds;
- no training invocation keeps both engines' tensors resident simultaneously;
- tools and benchmarks take the same temporary selector rather than importing one engine's internals.

“Both usable” means either engine can be intentionally selected. It does not mean every incomplete queued milestone
pretends to support every mode.

### 4.3 Import firewall

Add an AST import-boundary test before writing the new engine. Code under `pbg/core/queued/` must not import legacy
data-path modules or types:

```text
pbg.core.storage
pbg.core.rollout
pbg.core.branching
pbg.core.targets
pbg.core.topology
pbg.core.tree
pbg.core.learner
pbg.core.training
```

Exceptions are not added casually. If the new engine needs a representation-independent function currently trapped
inside one of those modules, extract that function into a clearly shared module, migrate the legacy caller, and
record the move in the migration ledger.

This firewall is what makes final deletion mechanical.

## 5. Migration ledger: copy only by proof

Do not copy old files wholesale. The intent behind copy-pasting is valuable—every old concept should have to justify
itself—but a symbol-level migration ledger enforces that intent more reliably.

Create `docs/audit/storage_redesign_migration_ledger.md` when implementation starts. Every relevant current
top-level symbol and every important method chain receives exactly one disposition:

| Disposition | Meaning | Cutover requirement |
|---|---|---|
| Reuse | Minimal and representation-independent | Remains in a shared module with both real callers |
| Extract | Useful behavior mixed into a legacy module | Move once; update both callers; delete old definition |
| Rewrite | Required semantics coupled to obsolete ownership | New implementation plus validation; old chain deleted |
| Delete | Stale, duplicate, forbidden, or unnecessary | Remove production and compatibility references |
| Externalize | Legitimate test/tool-only behavior | Move out of `pbg/` |

Each ledger row records:

```text
old symbol or chain
accepted production caller
authoritative contract
new owner, if any
disposition
validation evidence
deletion change
status
```

No “keep for safety” disposition exists. Uncertainty means investigate the production roots, not retain an
unbounded fallback.

The existing compactification audit already identifies high-confidence deletions and consolidations, including:

- obsolete direct policy-planning and unkeyed sampling paths;
- mutable-tree completion adapters;
- test-only card/rule production helpers;
- unused aliases and always-`None` inference fields;
- duplicate occurrence owners and double sealing;
- repeated partition/input wrappers;
- duplicate poker model-batch preparation;
- duplicate evaluator normalization/binding;
- duplicate memory traversal and estimation;
- repeated queue metrics, completion selection, validation, and configuration loading.

Import those findings into the ledger rather than performing a second independent audit.

## 6. Final architecture

The exact filenames can evolve, but responsibilities should converge toward:

```text
pbg/core/
  adaptive_branching.py     fixed/KKT/safeguard controller mathematics
  queueing.py               semantic FIFO and stable-prefix selection
  logical_rng.py            logical addresses and counter-keyed draws
  distributed_failure.py    rank-common failure agreement
  optimization.py           gradient synchronization and finite optimizer step
  update.py                 shared distributed update/sampling mechanics
  observability.py          production metrics

  queued/
    contracts.py            four-axis schemas and input/output partitions
    slabs.py                fixed SoA owners, reservation, publication, exact copy
    stores.py               phase owners and last-consumer validation
    topology.py             sequential chronology and branched structural topology
    rollout.py              semantic queue execution and transactional routing
    targets.py              sequential batches and branched layer propagation
    learner.py              exact-span loss construction
    memory.py               phase ledger, admission, and failure records
    engine.py               iteration/training composition
```

This does not require one file per noun. Small modules with inseparable lifetimes may be merged. The goal is a clear
dependency direction:

```text
contracts
  -> slabs/stores
  -> rollout/topology
  -> targets
  -> learner
  -> engine
```

Controllers, queueing, RNG, failure agreement, optimization, and observability are leaf services. They must not import
queued storage.

Poker-specific transition, model preparation, folded dependency, chance closure, and sibling-evaluator code remains
under `pbg/games/poker/`. Core sees generated schemas and opaque hooks, not poker street or field semantics.

## 7. Small shared abstractions, not an object framework

The redesigned core should have one canonical owner for each repeated semantic operation:

| Operation | Canonical authority |
|---|---|
| Mode-dependent fields and widths | One four-axis schema generator |
| Semantic queue choice and stable prefix | One queue scheduler |
| Logical RNG addresses and draws | One logical RNG module |
| SoA byte accounting | The immutable field schema |
| Reservation and publication | One monotone slab transaction protocol |
| Poker model inputs and legality | One prepared natural-forward boundary |
| Child destination classification/commit | One transactional routing boundary |
| Completion/evaluator binding | One sibling-evaluator facade |
| Structural child spans | One append-only tree topology |
| Branched learner actions | One exact ragged span representation |
| Phase memory peak | One phase ledger/admission authority |
| Sealing | One exact seal/compaction implementation |

Keep sequential and branched statistical/topological laws explicit where they genuinely differ. They may share queue
execution, transaction, routing, completion, and metrics mechanics without sharing root creation or child topology.

Use classes only when they own at least one of:

- mutable resource lifetime;
- invariant-bearing immutable schema;
- controller state;
- external protocol boundary.

Prefer functions and plain tensor collections for transformations. Avoid:

- one-method wrapper classes;
- class hierarchies for training modes;
- property aliases for retired names;
- generic `get(name)` layers over known fields;
- optional schema inference;
- “manager”, “context”, or “result” objects that merely forward another object;
- Boolean-flag functions that implement unrelated semantic laws;
- a utility module containing unrelated two-line helpers.

Two functions should merge when they enforce the same invariant and differ only by declared data. Similar syntax alone
is not enough. Conversely, do not keep `normal_*`, `rb_*`, `branched_rb_*`, and `br_branched_rb_*` copies when absent
schema fields or an estimator/game hook can express the difference.

## 8. Implementing the new storage path

### 8.1 Generated contracts

Generate the complete immutable run schema from:

```text
training_regime       self_play | best_response
rollout_geometry      sequential | branched
private_estimator     sampled | rao_blackwell
beta_mode             fixed_beta | KKT_beta       # branched only
```

The generator produces only applicable fields, owners, numerical widths, terminal blocks, and last consumers. It
validates:

- `input_partition` refinement of `output_partition`;
- `output(input_type)` for every model row;
- terminal output routing;
- parent/child recurrence conversions;
- sequential versus branched action ownership;
- required controller and queue metadata;
- dependency references and seal behavior;
- learner-retained columns.

The schema is data, not a class hierarchy and not a runtime union.

The intentional $H\to K$ RB aggregation changes the estimator and nonlinear loss. Its authoritative theory amendment
must be approved before the queued engine becomes production; parity tests must use the amended contract rather than
the legacy exact-$H$ loss.

`beta_mode` applies only to branched geometry. Sampled-branched and branched-best-response sampling/target laws also
require explicit authoritative contracts before those axis combinations become production.

### 8.2 Minimal storage primitives

Start with three concepts:

```text
SoASchema          immutable field dtype, shape, alignment, lifetime
MonotoneSlab       admitted capacity plus reserve/fill/publish tails
ExactBlock         sealed immutable realized extent
```

A row span is a pair of integer addresses, not a Python object graph. `MonotoneSlab` is justified because it owns
device allocations, transaction state, and release. `ExactBlock` may be a small immutable mapping plus schema if a
dedicated class does not add an invariant.

The primitive layer implements:

- exact per-column aligned byte calculation;
- fixed allocation before admitted work;
- `reserve -> fill -> publish`;
- multi-destination atomic publication;
- stable row addresses;
- minimum-logical-address failure reporting;
- stable exact compaction;
- dependency reachability and reference remapping;
- explicit release after the last consumer.

It knows nothing about poker, branching equations, or models.

### 8.3 Phase owners

Build only the owners named in `storage_redesign.md`:

- `FamilyWork[input_type]`;
- branched-only `ActionOccurence[output_type]`;
- branched `TreeTopology`;
- typed `Leaf[output_type]`;
- optional game `DependencyStore`;
- separate queue-behavior metadata;
- exact sequential/flattened target topology;
- `SigmaTarget`, branched `ActionTarget`, and sequential RB posterior;
- two-layer branched `LayerWork`;
- compact `LearnerIndex`;
- one fixed `DeviceFailure`.

Do not create intermediate convenience stores. A producer either writes a bounded transient workspace or the final
address of its next consumer.

### 8.4 Transactional rollout

One family-prefix executor should own the common mechanics:

```text
dequeue stable semantic prefix
prepare physical pack
run natural model forward
reserve logical child/action identities
classify small descriptors
reserve exact destinations
materialize each large payload once
publish all destinations
update queue behavior metadata
```

Sequential and branched geometry supply explicit child/topology planning functions. Poker supplies transition,
chance-closure, dependency, and final-write hooks. Neither creates an alternate scheduler.

The first correct branched-RB version may use a bounded preallocated combo-ID buffer with the current sibling
evaluator API. The hook should already have the final fusion-shaped contract so a new evaluator entry point can later
perform rejection sampling, transition, chance closure, classification, and final writes without changing core
ownership.

### 8.5 Topology and targets

Use one append-only structural topology during branched rollout. Every structural node carries `output_type`.
Flatten once into root-depth order after rollout closes. Allocate target extents from realized counts.

The numerical target engine uses:

- batched reverse-time depth for sequential geometry;
- two exact adjacent layers for branched geometry;
- action posterior rows without duplicated sigma posterior;
- explicit identity or $H\to K$ child/parent conversion;
- launch count proportional to realized depth, not node/action count.

`LayerWork` belongs only to the target phase and is released before sealing and learning. Its exact maximum adjacent
layer allocation remains charged to the target-phase peak; no speculative width such as 30,000 is built into the
design.

### 8.6 Exact ragged learner

Keep current-policy forwards sigma-major. For the largest feasible consecutive decision span, construct exact edge
prefixes and segment indices in one admitted reusable workspace.

Use scans, gather, `repeat_interleave` with an explicit output size, scatter, and segment reduction. Do not recreate a
padded `max_actions_per_decision` tensor or invent safe inactive actions to enable `vmap`.

`vmap`, `torch.compile`, CUDA graphs, Triton, or custom CUDA are optimizations after eager CUDA correctness and
profiles. They do not define retained layout or introduce a fallback matrix.

### 8.7 One memory authority

The new engine has one memory coordinator with three distinct responsibilities:

1. exact bytes for resident schema owners;
2. calibrated bounded envelopes for operations and external libraries;
3. between-iteration/attempt forecasting for admitted geometry.

It computes separate rollout, target, seal, and optimization peaks. Roughly 1 GiB of rollout slabs is acceptable when
the complete phase ceiling fits.

Fixed slabs simplify allocator behavior but do not delete:

- clean pre-draw sequential admission;
- transactional extent checks;
- whole-attempt branched rejection;
- rank-common controller backoff;
- disjoint retry addresses;
- bounded retries;
- failure agreement and guard calibration.

Remove the competing current estimators only after their last callers move.

## 9. Implementation sequence

### Phase 0 — freeze roots, measurements, and deletion commitments

Before adding the new engine:

1. refresh accepted production/public/tool roots in `tools/audit_reachability.py`;
2. refresh the token-count baseline and add the checked line-count command;
3. create the migration ledger and initial legacy deletion manifest;
4. import the existing compactification audit's high-confidence findings;
5. repair or explicitly retire stale accepted workflows before using reachability as deletion evidence;
6. record fixed-seed CUDA outputs for unchanged sampling, topology, targets, losses, and gradients;
7. record throughput, allocated/reserved memory, queue traces, and failure behavior;
8. define the intentional $H\to K$ reference outputs separately from legacy parity;
9. decide which legacy checkpoint contents must remain loadable by queued, and whether that is handled by the final
   canonical loader or a time-bounded offline migration tool.

The current reachability tool already separates production reachability from test/docs/tool evidence. A 2026-07-24
run indexed 481 top-level definitions, reached 471, and produced 10 review candidates. Extend it rather than replacing
it.

### Phase 1 — delete known stale surface

Perform the low-risk deletions already supported by the audit:

- obsolete policy-planning/direct-forward paths;
- unkeyed stochastic fallbacks;
- mutable completion adapters;
- test-only production helpers;
- unused aliases;
- permanently empty result fields;
- stale optional schema inference and compatibility column names where callers are already canonical.

Deleting first makes the legacy call graph more honest and prevents copying dead concepts into the new design.

### Phase 2 — extract the small shared kernel

Split representation-independent code out of mixed legacy modules:

- semantic FIFO and queue selection from `execution.py`;
- logical RNG/key construction from `execution.py`;
- distributed update mechanics from `learner.py`;
- shared loss primitives from storage-specific learner assembly;
- exact device-byte observation from competing memory estimates;
- common controller checkpoint helpers only where they remove real duplication.

Migrate legacy callers immediately. Do not add legacy re-export aliases. This phase creates the importable leaves
allowed by the queued import firewall.

### Phase 3 — add the bounded selector and empty queued package

Add:

- the temporary implementation configuration;
- the single resolver;
- checkpoint/metric implementation identity;
- import-firewall tests;
- a capability declaration for queued milestones;
- the initial projected-final line ledger.

The selector is the only intended migration compatibility surface.

### Phase 4 — implement contracts and slab primitives

Implement generated schemas, exact byte accounting, monotone transactions, exact blocks, dependency remapping, and
failure records. Test every intended four-axis schema and every reservation/publication invariant on CUDA.

No model or game integration enters this phase.

### Phase 5 — force branched structure early

Before the first complete sequential implementation, test:

- one ordinary distinguished root;
- contiguous child and action spans;
- duplicate-child identity;
- root refill target prefixes;
- structural `output_type`;
- frozen-player structural rows without learner rows;
- stable destination routing and publication;
- exact flattening and `LearnerIndex`.

Tensor fixtures are sufficient. This prevents a simple sequential vertical slice from shaping the abstractions around
one-action trajectories.

### Phase 6 — first end-to-end queued vertical slice

Implement normal sequential poker end to end:

```text
roots -> semantic queues -> FamilyWork -> leaves -> targets -> exact seal -> learner -> optimizer
```

This is the earliest point at which both selected implementations can complete real training iterations. Compare
logical draws, model inputs, transitions, targets, loss, gradients, checkpoint/resume, memory, and throughput.

### Phase 7 — branched RB end to end

Add:

- ragged `ActionOccurence`;
- queue-pressure scalar behavior;
- root refill and retry behavior;
- posterior profile sampling;
- final-address poker transitions;
- folded dependencies;
- completion;
- flattened layer targets;
- exact ragged learner work;
- fixed/KKT branching controllers.

Use the bounded accepted-ID evaluator boundary first, then replace it with the versioned fused CUDA evaluator API
without changing core stores.

### Phase 8 — complete the four axes and BR/probe workflows

Add sequential RB, sampled branched, best response, and every declared fixed/KKT combination. A generated capability
matrix must match generated schemas and tests; unsupported combinations cannot be accidentally constructed.

Migrate distributed training, replica workflows, variance runners, probes, benchmarks, and all accepted consumers.

### Phase 9 — queued-default validation period

Make queued the default while retaining explicit `legacy` selection for a short, recorded validation period. During
this period:

- new production runs use queued;
- matched legacy runs remain available for investigation;
- no queued bug is fixed by adding a legacy fallback;
- migration-ledger and projected-final totals must be current;
- every issue is classified as correctness, intentional semantic difference, performance, memory, or observability.

Exit only when all cutover gates pass.

### Phase 10 — deletion cutover

In one deletion-focused change:

1. remove the implementation selector and its config/checkpoint field;
2. remove legacy engine registration;
3. delete legacy storage, rollout, branching, topology, target, learner-assembly, seal, and memory paths;
4. delete legacy-only poker backend methods and wrappers;
5. migrate representation-independent semantic assertions, then delete legacy-only tests, fixtures, benchmarks,
   tools, aliases, and documentation;
6. remove migration-only parity adapters and capability declarations;
7. update imports so accepted production roots reach only queued;
8. delete newly isolated helpers reported by reachability;
9. verify the final line ceiling and projected deletion ledger;
10. rename nothing merely for compatibility—keep the clear `core/queued` ownership boundary.

The migration ledger must have no unresolved row.

### Phase 11 — compact the new implementation

Do not assume clean-room code is automatically clean. Audit it after legacy deletion:

- normalize near-duplicate AST/function reports;
- inspect every public function and class;
- merge operations with identical invariants;
- split Boolean-flag functions that combine different laws;
- delete unused observability and validation added during migration;
- remove one-off migration comments and terminology;
- verify every retained field against its last consumer;
- review the largest modules and functions manually;
- repeat CUDA memory/throughput tests after simplification.

The project is complete after this phase, not merely when queued first trains.

## 10. Dual-engine validation

Use matched artifacts, not a runtime compatibility representation.

For unchanged semantics, run both engines separately with the same:

- configuration;
- model/checkpoint;
- logical root and attempt addresses;
- fixed seeds;
- evaluator version;
- hardware profile.

Compare:

- logical RNG keys and sampled events;
- semantic queue order and stable prefixes;
- model natural inputs;
- public transitions and range normalization;
- topology and target-eligible action spans;
- completion inputs and outputs;
- targets, losses, gradients, and optimizer state;
- controller observations and updates;
- checkpoint/resume behavior;
- allocated, reserved, and device-observed peaks;
- complete-iteration throughput.

Do not require legacy equality for intentional changes such as RB preflop $H\to K$. Test those against small direct
equation oracles, finite checks, invariance properties, and approved theory.

Failure injection must cover:

- sequential overflow without reroll or replacement;
- branched whole-attempt release and synchronized retry;
- updated controller state and disjoint retry address;
- retry exhaustion;
- non-memory terminal failures;
- sampler/evaluator exhaustion;
- unpublished reservation cleanup;
- dependency remap validation.

All production-path changes require CUDA validation on an idle GPU using the repository environment. MPS tests remain
limited to its supported inference boundary.

## 11. Proving the old implementation is removable

Maintain a generated reverse-dependency report rooted at the legacy engine registration. It should contain every
legacy production module and no queued module.

The cutover is ready when:

- queued imports no legacy data-path module;
- all accepted workflows can select queued;
- no accepted workflow requires an old storage or topology object;
- old/new comparisons use only test artifacts;
- the legacy reverse-dependency closure contains only files scheduled for deletion;
- deleting that closure in a local branch leaves imports, configuration parsing, docs links, and tests valid.

Perform a deletion rehearsal before the real cutover. The rehearsal should remove the manifest mechanically and run
import collection, configuration parsing, reachability, and the CUDA acceptance subset. Any surprising reverse
dependency becomes a ledger item; do not add an adapter to make the rehearsal pass.

## 12. Final repository audit

Use the existing `tools/audit_reachability.py` as the base and strengthen its final gate:

- commit the complete accepted root list;
- fail on missing roots;
- fail on high-confidence unreachable objects;
- allow `REVIEW` only for individually documented dynamic boundaries;
- treat test/docs-only production objects as deletion or externalization candidates;
- add method-level or call-site review for large classes that top-level reachability cannot inspect;
- emit a machine-readable inventory used by CI.

Add complementary checks:

```text
production import graph
legacy forbidden-name/import scan
deprecated config and checkpoint-field scan
AST-normalized clone candidate report
non-comment/nonblank line count
top-level symbol and class inventory
largest module/function report
CUDA production-root coverage report
documentation/config/test reference scan
```

Clone detection is evidence, not an automatic refactoring instruction. Review whether candidates share an invariant.
The final audit report classifies every file as:

```text
production runtime
public API
verification/test
benchmark/experiment
developer/operations tooling
documentation
```

Nothing remains unclassified.

## 13. Cutover acceptance gates

The legacy implementation may be deleted only when:

- every requested four-axis configuration generates only applicable fields;
- normal, sequential RB, branched modes, best response, distributed/replica training, probes, and accepted benchmarks
  use queued;
- poker has only `pre` and `post_A` physical queues;
- stable FIFO order and logical draws are independent of physical pack width;
- no large child state/range payload is materialized before its final address is known;
- sequential action/log-probability is sigma-indexed and branched action data uses exact spans;
- every structural branched node carries a valid output domain;
- target launch count is proportional to depth rather than node/action count;
- the learner has no padded maximum-action tensor or invented inactive action;
- phase ledgers reconcile analytical, allocated, reserved, and device-observed peaks;
- roughly 1 GiB rollout profiles fit only when their complete phase ceiling fits;
- sequential and branched failure laws pass injection tests;
- no accepted workflow imports legacy internals;
- the legacy deletion rehearsal succeeds;
- the migration ledger is complete;
- reachability has no unexplained production candidate;
- final production code is net-negative and no greater than the refreshed 20,750-line ceiling;
- performance and memory meet the recorded acceptance floors;
- no compatibility selector, alias, adapter, stale config, or deprecated documentation remains.

## 14. Expected end state

The final training implementation should read as one ownership story:

```text
immutable four-axis contract
-> fixed semantic-family slabs
-> transactional final-address rollout
-> one structural topology
-> exact target-addressed outputs
-> one exact seal
-> exact ragged learner spans
-> optimizer
```

Poker contributes opaque natural inputs, transition/chance hooks, dependencies, and sibling-evaluator operations.
Core contributes queues, addresses, storage transactions, topology, targets, learner assembly, memory admission, and
failure behavior. Controllers and distributed/optimizer services remain small independent leaves.

There should be fewer Python objects because the data itself carries the structure:

- integer row spans instead of wrapper graphs;
- generated schemas instead of mode-specific containers;
- absent columns instead of optional union fields;
- one owner per lifetime;
- one function per semantic operation;
- one explicit game/core boundary;
- no legacy representation waiting behind a fallback.

The redesign is successful only when the new engine works, the old engine is gone, and the repository is visibly
smaller and easier to explain.
