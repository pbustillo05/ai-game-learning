# PBG implementation versus queued-design audit

Date: 2026-07-29

Audited revision: `a5d5fb1dda4d8065c6c3fb49b13676b26974c131`

Post-audit update (2026-07-30): F1 and F2 were resolved by commits
`4616ddb`, `f37bc2f`, and `13f3609`. F3 was completed by commit `9229734`
after the preceding refactor had already unified the objective equation. F4
was completed by commits `35a5a33`, `b3ad5ba`, and `1c8ae2a`. F8 was closed
as an intentional design decision: the explicit
`torch.cuda.empty_cache()` call at the rollout-owner release boundary is
desired. The findings below retain the original audit evidence for historical
context.

Scope: all production Python and CUDA sources under `pbg/`, with particular
attention to `pbg/core/`, `pbg/games/poker/queued/`, and the public composition
in `pbg/runtime.py`. The comparison authority is
`docs/queued/design/`, especially `storage_redesign.md`,
`layered_target_creation.md`, `poker_branched_rb_sample.md`,
`poker_folded_statistics.md`, and `poker_policy_cuda_operator.md`. The audit
also read the implementation and closure records under `docs/queued/` and
`docs/audit/queued/`, Git history through the cutover, the current tests, and
the performance/capacity audits.

The working tree already contained a documentation-only move of
`docs/queued/design/implementation_plan.md` to
`docs/queued/implementation_plan.md` when this audit began. Additional
uncommitted source, test, tool, and issue-note edits appeared while the audit
was in progress. The configuration addendum below inspected the resulting
working-tree versions of the parsers and capacity/controller code. This audit
did not modify or otherwise include any of those pre-existing or concurrent
changes.

## Executive conclusion

The implementation is **not faithful to the strongest “one modular
implementation to rule them all” requirement**.

It has a good shared substrate:

- twelve generated mode schemas;
- one fixed-capacity SoA/slab implementation;
- one generic `RolloutTransaction`;
- one generic bounded-chance primitive;
- one compact topology type and one layer-plan type;
- one custom CUDA layered-target executor;
- one optimizer implementation; and
- one logical-RNG authority.

But that substrate is composed by multiple substantial engines. The public
runtime explicitly dispatches sequential and branched execution. Sequential,
ordinary branched, and replica/distributed branched execution each own a
separate end-to-end orchestration loop. Sequential and branched execution also
have separate topology assemblers, seal/store types, samplers, optimization
loops, and forward/loss/backward bodies. These are not merely different slab
alignments or sampler functions.

The most accurate description is:

> one shared low-level toolkit, used by three production iteration engines

rather than:

> one schema-generated production pipeline for all modes and execution
> dimensions.

The literal pre-rewrite engine appears to have been deleted successfully.
There is no retained `pbg/queued` generic package, old generic core alongside
the new core, CPU production fallback, compatibility engine selector, or
obvious rollback implementation. That is a real success. It does not,
however, establish the stronger claim of zero dead or deprecated production
code. At least three generated storage columns have no production consumer,
the production iteration classes retain profiling-only early-exit branches,
and the current reachability tool is too coarse to detect unused methods or
fields.

Several central acceleration ideas are implemented and have produced large
measured wins, especially device-side publication, compact routing, logical
RNG, and layered CUDA targets. Several others are incomplete:

- stable FIFO prefixes are gathered with advanced indexing rather than passed
  as direct contiguous model views;
- best-response learned and frozen-opponent rows share each input-type FIFO
  and are compacted by role only at the model call instead of being routed to
  role-homogeneous FIFOs;
- the posterior sampler materializes a dense `[G,256,P]` attempt envelope in
  PyTorch instead of using the proposed parent-CTA fused kernel;
- selected-action log-policy computation materializes full legal-action
  probabilities before gathering;
- transition, folded-statistic construction, and chance closure remain
  multi-kernel PyTorch sequences rather than the proposed fusion;
- exact `LayerWork` extents are computed but fixed maximum-capacity slabs are
  allocated anyway; and
- host feedback remains in the rollout scheduling loop.

The storage implementation is structurally close to the design but not
faithful in several important lifetime and ownership details. Most notably,
`LearnerIndex` contains copied `model_input`, `owner`, and sampled-private
payloads even though the design declares it as index metadata only and names
`FamilyWork.retained` as the learner input. This is a second retained payload
owner and a full device copy. The repository's own HU beta-2.4 capacity audit
measured `LearnerIndex.post_2` at 118.637 MiB next to
`FamilyWork.post_2` at 119.591 MiB.

The generic-to-game dependency direction is clean: `pbg/core` does not import
poker. The responsibility boundary is not clean. More than 6,500 lines in
`pbg/games/poker/queued/training.py` and
`pbg/games/poker/queued/branching.py` implement generic scheduling, topology
assembly, phase transitions, completion orchestration, sealing, target
orchestration, learner construction, distributed preflight, optimizer
sequencing, recovery, and release. Poker is therefore not just a collection
of opaque game functions consumed by a generic core.

Finally, the implementation is not short or simple in the sense requested by
the design. The audited core and queued-poker packages contain about 26,600
lines, with individual files of 5,008, 3,075, 2,567, 2,395, 1,770, and 1,549
lines. Size is not itself a defect, but here it correlates with parallel
authorities and duplicated lifecycle code.

### Scorecard

| Question | Verdict | Main reason |
|---|---|---|
| One exact modular pipeline for all twelve modes | **No** | Shared primitives feed distinct sequential, ordinary-branched, and replica-branched engines |
| Sequential is only degree-one branching plus alignment/sampling differences | **No** | Different runtime entry, topology assembly, orchestration, stores, samplers, optimization loop, and loss body |
| Fixed and KKT share an implementation | **Largely yes** | They use a shared controller interface and schema-compatible branched machinery |
| Self-play and best response share core machinery | **Mostly** | Schema/role differences are shared, but compaction and cutoff concerns leak into large orchestration paths |
| Sampled and RB share core machinery | **Partly** | Storage and targets are schema-driven; completion, sampling, and learner equations have sizeable forks |
| Ordinary, replica, and distributed are execution dimensions only | **No** for branched | `ReplicaBranchedPokerIteration` is a separate engine |
| Old pre-rewrite engine deleted | **Yes, with high confidence** | Cutover history, package inventory, imports, and current reachability show no parallel old namespace |
| Zero unused/deprecated production surface | **No** | Concrete unused generated columns; profiling bypasses; static audit cannot inspect methods/fields |
| All proposed acceleration ideas implemented | **No** | Direct views, role-homogeneous BR queues, fusion, posterior CTA, direct selected policy, and exact late allocation remain |
| Storage faithful to `storage_redesign.md` | **Partly** | Main SoA owners exist, but retained-input lifetime, `LearnerIndex`, folded payload, and exact allocation differ |
| `core`/poker dependency boundary | **Yes** | Core has no poker imports |
| `core`/poker responsibility boundary | **No** | Generic iteration-engine work lives in poker modules |
| YAML configuration surface | **No** | It mixes choices with derived internals; 15/16 checked-in files retain a stale preflop input width |
| Simple, short, modular, elegant | **No overall** | Strong primitives, but orchestration and learner duplication dominate the final shape |

## 1. What the design actually requires

The strongest requirements are explicit, not interpretive.

`storage_redesign.md` defines sequential mode as
$N=\widetilde N=1$ at every node and says every mode uses the same generated
schema (`docs/queued/design/storage_redesign.md:8-39`). Its stable
`FamilyWork` prefix is supposed to be a direct model batch
(`storage_redesign.md:89-94`). Every prefix follows the same
`model -> sample -> commit` operation, with commit overlapping the next model
batch (`storage_redesign.md:219-265`).

The clarified execution contract deliberately does not require the completion
overlap permitted by `storage_redesign.md:375-380`. Rollout finishes first,
rollout-only allocations are released, the PyTorch cache is emptied, and
evaluator completion then owns the GPU. This serialization is the intended
memory-admission and execution policy, not a fidelity gap.

In best-response mode the semantic FIFO key is
`(replica, input_type, role)`, where role is learned exploiter or frozen
opponent. Queue identity selects the model weights, so role is not a
per-row `FamilyWork.hot` field. Root role counts are maximally even: for
$N$ roots and $P$ seats, every seat receives
$\lfloor N/P\rfloor$ roots and a uniformly selected set of
$N-P\lfloor N/P\rfloor$ distinct seats receives one additional root.

The moved implementation plan is even more direct:

- one generated path for all twelve modes;
- one authority per operation;
- no legacy implementation or runtime implementation selector;
- a mode may change fields, constants, sampling law, multiplicity, and learner
  sampling, but may not select another storage engine, queue engine, target
  builder, learner, or optimizer; and
- ordinary, rank, replica, and root axes are execution dimensions, not modes.

See `docs/queued/implementation_plan.md:9-84`.

The intended pipeline is:

```text
ModeSchema
  -> semantic FIFO
  -> replica-vmapped inference
  -> sample
  -> commit
  -> Leaf -> LayerWork
  -> one CPU Topology -> one GPU LayerPlan
  -> align-and-seal
  -> one layered target executor
  -> aligned or ragged sampling
  -> one forward/loss/backward operation
  -> one optimizer transition
```

“Aligned or ragged” is an allowed learner-presentation difference. It does not
authorize two optimization state machines or separate objective
implementations.

The focused proposals also set concrete performance contracts:

- branched-RB sampling is a parent-CTA CUDA operator with child-zero reuse and
  no global dense attempt workspace
  (`poker_branched_rb_sample.md:65-162`);
- the canonical policy has a direct selected-action entry point and rollout
  never materializes `[H,event_count]`
  (`poker_policy_cuda_operator.md:5-39`);
- folded statistics write directly into destination `model_input`
  (`poker_folded_statistics.md:5-35`);
- target creation has one kernel template, no Python node/action loop, and no
  inter-layer host synchronization
  (`layered_target_creation.md:5-40`); and
- experiments select one implementation and delete alternatives
  (`docs/queued/design/README.md:39-49`).

These clauses are the standard used below. Passing functional tests alone is
not enough to establish fidelity to them.

## 2. Actual production architecture

### 2.1 Shared substrate that is genuinely successful

The following parts do conform well to the vision.

#### Closed twelve-mode schema

`pbg/core/contracts.py` creates the Cartesian product of training regime,
rollout mode, and private estimator. Fields are generally made present or
absent by schema rather than nullable mode DTOs. Fixed and KKT modes have a
defined physical-compatibility relation. This is exactly the right kind of
schema-driven variation.

#### Fixed generated storage

`SoAField`, `SoASchema`, `MonotoneSlab`, `PublicationTransaction`,
`DeviceFailureState`, `GeneratedOwners`, and the immutable capacity plan form
one coherent allocation vocabulary. Slabs reserve, fill, and publish
monotonically. Publication can cover multiple destinations atomically. There
is no paging, wraparound, recycling, CPU rollout store, or runtime capacity
growth.

#### Generic rollout transaction

`pbg/core/rollout.py:2273-2639` owns one generic
`model -> sample -> transition -> commit` transaction. It uses distinct model
and commit streams and validates descriptor coverage. Both sequential and
branched poker call it.

This is the strongest evidence in favor of the current architecture. The
problem is not that no shared pipeline exists; it is that this shared stage is
embedded inside several larger pipelines.

#### Logical RNG and bounded chance

Logical RNG addresses are explicit and schedule-independent. Sampled and RB
paths use the same address machinery. Sequential and branched poker both use
`BoundedChanceClosure`, so there is not a second chance-loop primitive.

#### Layered targets

`LayeredTargetExecutor` is one generic class
(`pbg/core/targets.py:366-720`). It passes schema constants to one C++/CUDA
entry point. The extension launches layers in order without host
synchronization between them and synchronizes only at the final validation
and publication boundary (`pbg/core/targets.py:635-701`). Sequential target
propagation is a mode flag within this executor, not a separate target
builder.

The current performance audit reports that target propagation fell from
14.670 seconds in the old implementation to a 0.00576-second CUDA interval on
the matched workload. This part of the rewrite achieved both modular and
performance goals.

#### Replica-vmapped inference and one optimizer

The replica module uses stateless/vmapped inference, and dense Adam/AdamW
updates share one optimizer authority. The remaining replica divergence is in
iteration orchestration and learner sequencing, not model-state or optimizer
implementations.

### 2.2 The actual public dispatch

The public runtime contains a direct implementation selector:

```text
TrainingSession.step
  if SequentialModeConfig:
    _run_sequential
      ordinary -> RecoveringSequentialPokerIteration
                    -> SequentialPokerIteration
      replica/distributed -> SequentialPokerIteration
  else:
    _run_branched
      ordinary -> BranchedSampledPokerIteration
                    -> _BranchedSampledPokerAttempt
      replica/distributed -> ReplicaBranchedPokerIteration
```

The branch is at `pbg/runtime.py:797-801`. The two construction paths are at
`pbg/runtime.py:526-598` and `pbg/runtime.py:692-769`.

Calling this “one runtime composition” is true only at the outer
`TrainingSession` façade. The session selects materially different iteration
implementations.

### 2.3 Lifecycle ownership matrix

| Lifecycle responsibility | Sequential | Ordinary branched | Replica/distributed branched |
|---|---|---|---|
| Root publication | `SequentialPokerIteration` | `_BranchedSampledPokerAttempt` | `ReplicaBranchedPokerIteration` |
| Scheduler loop | sequential loop | separate inflight loop | separate per-lane loop |
| Topology assembly | `AsyncTopologyPublisher.seal_deferred_unary` | `BranchedTopologyRecorder` | another use of `BranchedTopologyRecorder` |
| Queue-depth authority | host `depths`, waits every commit | host `depths`, drains up to four inflight commits | host `depths`, waits every commit |
| Completion creation/drain | sequential class | ordinary-branched class | replica-branched class |
| Align/seal | sequential sealed-input loop | branched sealed-input loop | another branched sealed-input loop |
| Target setup | sequential class | ordinary-branched class | replica-branched class |
| Learner-store construction | sequential class | ordinary-branched class | replica-branched class |
| Preflight/optimizer sequencing | sequential class | mostly ordinary direct path | separate replica preflight and actual pass |
| Recovery | `RecoveringSequentialPokerIteration` or runtime wrapper | `BranchedSampledPokerIteration` | runtime replica wrapper |
| Owner release/close | sequential class | ordinary-branched class | replica-branched class |

Only the lower-level transaction, generated storage, layer planner/executor,
and optimizer are common. That is not the requested lifecycle architecture.

## 3. Major findings

### F1 — Resolved (was High): there are multiple production iteration engines

`SequentialPokerIteration.run()` starts at
`pbg/games/poker/queued/training.py:768`. It is a complete 430-line lifecycle
method before its separate learner helper. It constructs queues, roots, hooks,
topology publication, the scheduling loop, topology sealing, completion,
learner sealing, target execution, learner passes, releases, and results.

`_BranchedSampledPokerAttempt.run()` starts at
`pbg/games/poker/queued/branching.py:2918` and implements the same lifecycle
again with different orchestration.

`ReplicaBranchedPokerIteration.run()` starts at
`pbg/games/poker/queued/branching.py:4189` and implements it a third time.
It is not a thin lane wrapper. It republishes roots, builds queues, constructs
transactions and topology recorders, schedules rollout, seals topology,
performs completion, seals learner data, executes targets, performs a
learner-preflight pass, then performs the actual learner pass.

This violates the implementation-plan rule that `n_replica` is an execution
dimension and that modes may not select another engine.

The practical consequence is already visible in Git history. Changes after
the Phase 11 closure had to modify large parts of
`pbg/games/poker/queued/branching.py` to enforce safe tree-profile reuse. Any
future lifecycle change must be audited across ordinary and replica branched
paths separately.

**Required architectural correction:** a single generic `IterationEngine`
must own phase progression, queue scheduling, topology acceptance, completion
draining, sealing, targets, learner invocation, release, and recovery
boundaries. Mode and game behavior should enter through immutable schema data
and a small hook bundle.

### F2 — Resolved (was High): sequential is not implemented as the degree-one specialization

At the storage and target-kernel layers, sequential is close to the
degree-one specialization. At the pipeline level it is not.

Sequential topology is reconstructed by
`AsyncTopologyPublisher.seal_deferred_unary()`
(`pbg/core/rollout.py:1987-2122`). It concatenates a separate trace format,
converts it to Python rows with `trace.tolist()`, and creates unary topology.

Branched topology is assembled by the 600-line poker-specific
`BranchedTopologyRecorder`
(`pbg/games/poker/queued/branching.py:349-964`), with pinned staging slots,
payload acceptance, endpoint patches, CPU materialization, sealing, and
counterfactual accounting.

The learner then splits again:

- `SequentialLearnerBatch`, `SealedSequentialInputs`,
  `AlignedSequentialStore`, and `SequentialBatchSampler`;
- `BranchedLearnerBatch`, `SealedBranchedInputs`,
  `RaggedBranchedStore`, and `BranchedBatchSampler`;
- `SequentialSampledIteration.optimize()`;
- `BranchedSampledIteration.optimize()`; and
- separate sequential, branched-sampled, and branched-RB loss bodies.

Aligned and ragged sampling were explicitly allowed. Separate topology
protocols, optimize loops, and objective bodies were not.

A clean degree-one representation is possible:

- every decision batch has `sigma_index`, `action_ptr`, and `action_index`;
- sequential `action_ptr = arange(B+1)` and every span has length one;
- every topology packet carries degree fields, with sequential values all one;
- the common assembler can optimize unary storage internally without exposing
  another lifecycle;
- trace/cost fields are schema-absent in sequential rows; and
- the same optimization loop handles either batch layout through a small
  gather strategy.

### F3 — Resolved (was High): “one authoritative learner operation” is a
façade over three objective implementations

Resolution: the production learner now has one
`LearnerOperation.forward_loss_backward()` over the common sigma/action-span
schema, with sequential represented by unit spans, and one `LearnerIteration`
optimizer state machine for aligned and ragged presentations. Commit
`9229734` removed the remaining replica/distributed validate-only replay and
routed sequential and branched preparation through one `LearnerTransaction`.
The retired learner objective methods, sampled iteration classes, preflight
optimizer, and `validate_only` path are absent from production. Focused CUDA
tests cover the common objective, one-pass replica execution, and local
transaction rollback.

The text below records the original audited implementation.

`LearnerOperation.forward_loss_backward()` dispatches on rollout mode at
`pbg/core/learner.py:1593-1615`.

The sequential sampled/RB implementation occupies the remainder of the
method. Branched sampled uses
`_forward_branched_loss_backward()` at `pbg/core/learner.py:1832`, and that
method immediately dispatches branched RB to
`_forward_branched_rb_loss_backward()` at `pbg/core/learner.py:2086`.

These paths repeat:

- autocast setup;
- current and reference model execution;
- policy evaluation;
- old/current ratio construction;
- clipping;
- value loss;
- entropy and reference KL;
- objective weighting;
- backward; and
- metric construction.

The branched-only trace/cost terms are valid schema-driven differences. The
three repeated policy/value objective pipelines are not. A single class name
does not create one authority if its body is a mode switch into independent
equation implementations.

Likewise, `pbg/core/training.py` has two optimization state machines:
`SequentialSampledIteration` at line 210 and
`BranchedSampledIteration` at line 411. Both manage logical-step changes,
zeroing, failure agreement, weighted metrics, backward calls, optimizer
steps, and run-state publication.

The class names are also misleading: both “Sampled” classes support RB. This
is migration-era naming debt and makes the architecture harder to understand.

### F4 — Resolved (was High): the core/poker responsibility boundary is misplaced

Resolution: production now constructs one generic `IterationEngine` and one
semantic `PokerOperations` bundle for every sequential, branched, sampled,
RB, replica, and distributed mode. Core exclusively owns phase orchestration,
queue scheduling, topology acceptance and sealing, completion draining,
target-arena padding, align-and-seal, learner stores and samplers, reference
caches, target execution, the learner transaction, recovery, telemetry, and
storage release. Poker supplies only the game operations exposed through the
small `GameOperations` protocol.

The phase-sized `PokerSequentialProgram`, `PokerBranchedProgram`,
`PokerIterationHooks`, and their compatibility surfaces were deleted rather
than wrapped. The former poker `training.py` and `iteration.py` modules are
also absent. Static ownership gates, the full CUDA suite, and the two-GPU NCCL
test passed after comparison with the pre-F4 baseline at `9229734`.

The text below records the original audited implementation.

The import firewall is sound. The current static gate confirms that no module
under `pbg/core` imports `pbg.games.poker`.

The inverse allocation of responsibility is not sound. Poker currently owns:

- full phase orchestration;
- queue construction and host queue-depth scheduling;
- ordinary and replica rollout loops;
- topology assembly and topology staging policy;
- `LayerPlan` invocation;
- completion-FIFO lifecycle;
- completion-to-target arena padding;
- align-and-seal orchestration;
- learner store/sampler construction;
- cost-aware statistical allocation;
- reference-cache admission;
- distributed/replica learner preflight;
- optimizer sequencing;
- overflow recovery wrappers;
- storage release and close; and
- much of generic benchmark telemetry.

These concerns live primarily in:

- `pbg/games/poker/queued/training.py` — 1,549 lines; and
- `pbg/games/poker/queued/branching.py` — 5,008 lines.

Poker-specific ownership should instead be limited to operations such as:

- game geometry and state transition;
- root and joint-profile sampling;
- action legality and chance labels;
- factorized policy composition;
- action-conditioned posterior sampling;
- folded-range creation and folded-statistic materialization;
- poker model input construction;
- evaluator-v2 leaf binding; and
- model architecture.

The generic core should own when those functions execute, how their
destinations are reserved, how topology packets are accepted, and how the
iteration advances and releases storage.

The current boundary therefore passes a dependency-direction test while
failing a responsibility test.

### F5 — Resolved (was High): `LearnerIndex` is a second retained-payload owner

The design declares:

```text
LearnerIndex
  action_begin
  action_count
  N_tgt
  log_sampling_weight
```

and says learner model inputs come from
`FamilyWork.retained.model_input`
(`storage_redesign.md:460-481,557-566`).
These index fields serve the branched learner's compact ragged presentation;
sequential rows are aligned and do not need a substantive standalone index.

The actual schema declares:

```text
LearnerIndex
  model_input
  sampled_private_state   # sampled
  owner
  target_degree           # branched
  action_begin            # branched
  action_count            # branched
  log_sampling_weight     # branched
```

at `pbg/core/contracts.py:352-367`.

`SealedSequentialInputs.seal()` copies the retained fields from
`FamilyWork` into `LearnerIndex` (`pbg/core/learner.py:539-610`).
`SealedBranchedInputs.seal()` gathers and copies them again
(`pbg/core/learner.py:1056-1124`).

This has four consequences:

1. `LearnerIndex` is not an index owner; it is a second payload store.
2. The direct row alignment promised for self-play is replaced by a device
   copy even when no compaction is needed.
3. `FamilyWork` can be released at align-and-seal only because its retained
   payload has been duplicated.
4. The storage design's one-owner-per-lifetime principle is obscured: hot and
   retained fields have different logical lifetimes but share one slab
   allocation, so the implementation creates another owner rather than
   representing those lifetimes directly.

The cost is material. The current HU beta-2.4 capacity audit reports:

| Slab | Allocated MiB |
|---|---:|
| `FamilyWork.post_2` | 119.591 |
| `LearnerIndex.post_2` | 118.637 |

See
`docs/audit/nlhe_hu_32bb_beta24_execution_capacity_audit_20260728.md:230-247`.

**Correction:** represent retained learner payload as its own lifetime-bearing
owner from initial commit, or retain direct views into a payload-only backing
allocation. `LearnerIndex` should exist as metadata-only learner state for the
branched ragged presentation. Sequential mode needs no substantive
`LearnerIndex`; its learner owners are already row-aligned. Best response may
compact trained retained rows when genuinely required; self-play should not
pay that copy.

### F6 — Resolved (was High): exact `LayerWork` sizing is known and ignored

The repository's own capacity audit identifies this as a required fix.

The layer planner computes exact per-replica/per-output arena extents before
completion. Nevertheless, `GeneratedOwners` allocates every `LayerWork`
partition at the manifest's scalar `target_arena_cap`.

On the audited HU beta-2.4 workload:

| `LayerWork` | Current bytes | Exact planned bytes |
|---|---:|---:|
| pre | 304,640,064 | 16,384 |
| post | 2,377,984,064 | 625,027,680 |
| total | 2,682,624,128 | 625,044,064 |

The avoidable allocation is about 2.058 GB, or 1.916 GiB.

See
`docs/audit/nlhe_hu_32bb_beta24_execution_capacity_audit_20260728.md:249-315`.

This undermines three design goals simultaneously:

- exact realized storage at align-and-seal;
- scalable tree capacity; and
- using fixed caps as admission ceilings rather than always allocating the
  ceiling.

The correction does not require dynamic growth. After `build_layer_plan`, the
engine can validate exact extents against immutable caps, allocate each
fixed-address arena once at that exact extent, and retain it unchanged through
targets.

### F7 — Medium-high: queue batches are neither direct nor role-homogeneous

`SemanticFamilyFIFO.stable_prefix()` constructs lane and row index tensors,
then evaluates:

```python
value[lanes[:, None], row]
```

for every `FamilyWork` column (`pbg/core/rollout.py:669-693`).

This is advanced indexing. It materializes gathered tensors; it is not a
direct contiguous view into the SoA column. It also gathers hot columns that
the model does not need into `FamilyBatch.fields`, despite the design saying
the model reads only `model_input`.

For one lane, a contiguous prefix can be a simple slice. For compatible
replica lanes, the physical layout should permit a strided or reshaped batch
view when their heads are aligned; otherwise only the minimal model fields
should be packed into admitted workspace. Calling the current result “direct”
in the class docstring is inaccurate.

This matters because family inference is the dominant measured cost. The
2026-07-28 performance audit attributes 78.58% of rollout-equivalent time to
model inference. Avoidable prefix gathering and unnecessary columns are
therefore on a high-value path.

Best response adds another avoidable transformation. There is one FIFO per
`input_type`, with learned and frozen-opponent rows mixed together.
`compact_best_response_evaluation()` classifies the rows, gathers one compact
batch per role, evaluates the two model-weight sets separately, and scatters
both results back into FIFO order (`pbg/core/learner.py:314-350`).

The intended queue key is `(replica, input_type, role)`. Draining a stable
prefix then presents one semantic batch to one model-weight set without
per-call role classification, gather, or merge. Queue identity also makes the
`role_selector` column in `FamilyWork.hot` redundant. The selected response
player remains root metadata where leaf completion or topology semantics
require it; it is not carried in every hot family row.

### F8 — Resolved: serialized completion and `empty_cache()` are intentional

Resolution: there is no implementation defect and no code change is required.
The explicit `torch.cuda.empty_cache()` call was intentionally added at this
owner-lifecycle boundary so inactive PyTorch cache blocks are returned before
the native evaluator requests its workspace.

All three current lifecycle implementations instead finish rollout and
topology planning, release rollout owners, enter queue drain, synchronize the
device, call `torch.cuda.empty_cache()`, and only then enter completion:

- sequential: `pbg/games/poker/queued/training.py:955-979`;
- ordinary branched:
  `pbg/games/poker/queued/branching.py:3308-3327`; and
- replica branched:
  `pbg/games/poker/queued/branching.py:4397-4416`.

This is the intended execution order. Returning inactive PyTorch cache blocks
before the native evaluator requests workspace is worth more than concurrent
PyTorch/evaluator GPU use. Completion must remain capacity-admitted against
the owners that are still live; `empty_cache()` releases unused cached blocks,
not live tensor storage. No leaf-consumer stream or rollout/completion overlap
is required.

### F9 — Medium-high: proposed high-value poker CUDA operators are still
multi-kernel PyTorch

The current performance audit accurately records that there is no
poker-specific custom CUDA extension for:

- collision-conditioned posterior sampling;
- the policy operator; or
- folded statistics.

The custom extensions under `pbg/core/csrc/` cover only:

- layered targets;
- branching log-mass metadata; and
- alias-table construction.

#### Posterior sampler

`CollisionConditionedProfileSampler.sample_children()` creates:

- a `[G,A,P]` mesh grid with `A=256`;
- logical keys for the full envelope;
- eleven tensor-level binary-search iterations;
- selected hands and cards for every attempt;
- pairwise collision masks for every attempt; and
- first-success selection afterward.

See `pbg/games/poker/queued/rb_sampling.py:76-250`.

This preserves the correct logical first-success law and child-zero reuse, but
it is the opposite physical implementation from the proposed parent-CTA
kernel. The 2026-07-28 audit measured it as the largest identified
poker-specific kernel candidate at 7.40% of rollout-equivalent time and about
46.2 dense attempt rows per executed edge.

#### Selected-action policy

`PokerPolicyOperator.selected_log_pi()` calls
`probabilities()` and gathers afterward
(`pbg/games/poker/queued/policy.py:197-223`). With raw policy shape
`[B,H,...]`, `probabilities()` constructs `[B,H,E]`.

That directly contradicts the focused policy contract's statement that
rollout never materializes `[H,event_count]`. Actor sampling is appropriately
restricted to one acting hand; retained selected-log-policy construction is
not.

The current audit measures this selected-action scope at 4.26% of rollout and
already recommends a direct numerator plus legal-event normalizer.

#### Folded statistics and transition fusion

Folded construction is accelerator-resident and semantically centralized, but
it is a sequence of PyTorch reductions, finite differences, and linear
algebra. Transition, classification, folded-stat refresh, and chance closure
are also Python-orchestrated tensor sequences. There is no evidence of
`torch.compile`, CUDA graphs, or the proposed fused one-CTA commit path in
production.

The design says these operations “should fuse,” so this is an incomplete
performance objective rather than a semantic violation. It still answers the
user's question “did we implement all the core ideas to accelerate training?”
with no.

### F10 — Medium: host feedback still controls rollout progress

The GPU does not read node counts back for reservation and commit, which is a
major improvement. The outer scheduler still requires host-visible topology
feedback to update queue-depth dictionaries.

Sequential waits for deferred feedback after every commit
(`training.py:873-883`).

Ordinary branched permits up to four inflight packets, then synchronizes on
feedback and increments explicit `host_synchronizations`
(`branching.py:3032-3236`). Replica branched waits on every publication
(`branching.py:4338-4359`).

Thus model/commit stream overlap exists inside `RolloutTransaction`, but the
next-family scheduler is not fully device-driven. The measured reduction from
2,494 to 66 host synchronizations is excellent; 66 is not zero, and the
replica path has less inflight behavior than the ordinary path.

### F11 — Medium: generated storage contains unused fields

The following schema fields have no production read or semantic write outside
schema generation and physical-field construction:

- `FamilyWork.reservation_state`;
- `FamilyWork.publication_state`; and
- `QueueMetadata.controller_state`.

Production reservation/publication state actually lives in
`MonotoneSlab._state`. Branch-controller state lives in controller objects and
`RunState.branch_controller_state`.

Repository-wide production references to the three field names occur only in
`pbg/core/contracts.py` and `pbg/core/capacity.py`. Queue initialization fills
`controller_state` with its default, but no controller consumes or updates it.

These are small byte costs but important audit evidence: the generated schema
is not currently “only applicable fields,” and the claim of zero unused
production data is false.

`FamilyWork.folded_range_row` also always allocates width `P`, while the design
specifies references for the `P-A` folded seats. A fixed `P` vector is easier
to index and may be a reasonable deliberate tradeoff, but it is not the exact
schema described.

### F12 — Medium: `FoldedRanges` deliberately diverges from normalized-only
storage

The design says `FoldedRanges` stores normalized `r_pi` only
(`storage_redesign.md:187-194,345-350`).

The actual owner stores:

```text
range
log_mass
```

(`pbg/core/contracts.py:415-420`).

Unlike the unused fields above, `log_mass` is active. Poker writes it on fold
and uses it to restore absolute reach scale when resolving ranges
(`pbg/games/poker/queued/hooks.py:1397-1432,1900-1908` and analogous branched
code).

This may be scientifically necessary to preserve raw folded mass, and the
Phase 6 ledger explicitly mentions raw mass. The problem is governance:
`storage_redesign.md` remains the stated storage authority and was not updated
to describe this second representation component. Either:

- prove `log_mass` is required and amend the storage contract through the
  appropriate approval process; or
- derive the needed scale from another existing retained quantity.

It should not remain an undocumented divergence under a “faithful
implementation” claim.

### F13 — Medium: Leaf fields are semantically faithful but not literally
“exactly” the proposed schema

The opaque field names are sensible:

- `leaf_public_state` is the five-card board;
- `leaf_private_profile` is sampled combo IDs; and
- the evaluator binding maps them directly to `board_cards` and `combo_ids`.

This supports a clean generic core boundary and should not be changed merely
to match poker names in core.

There are, however, extra best-response columns:

- `trained_actor`; and
- `response_cutoff`.

The former corresponds to evaluator `selected_player`; the latter supports
shared-response cutoff behavior outside the basic evaluator-v2 row contract.
This is a justified feature extension, but it means the statement “Leaf stores
exactly the batched evaluator inputs” is no longer literally true. The design
and generated schema should acknowledge the cutoff field explicitly.

### F14 — Medium: prior closure reports overstate what their evidence proves

The Phase 11 report says:

- the repository has one production engine;
- sequential and branched use one engine;
- no production object exists solely for tests; and
- all 441 production definitions are reachable.

The first two claims are contradicted by the runtime and iteration-class
structure above.

The reachability result is useful but much weaker than the prose implies:

- `tools/audit_reachability.py` indexes top-level functions, classes, and
  constants;
- it treats a reachable class as one reachable definition;
- it does not index or prove reachability of individual methods;
- it does not inspect generated schema fields;
- module imports and broad public roots make conservative reachability easy;
  and
- dynamic hazards are reported but cannot prove absence of dead paths.

At the audited HEAD, the tool reports 476 of 476 indexed definitions
reachable, up from the closure report's 441 because executable commits landed
after the audited Phase 11 SHA. That is a good deletion-candidate screen, not
proof of zero dead code or one engine.

The static gate also forbids only the old generic `pbg.queued` namespace. It
does not detect parallel iteration composers under
`pbg.games.poker.queued`, nor should a module-import graph be expected to.

## 4. Storage fidelity, owner by owner

| Designed owner | Actual status | Fidelity judgment |
|---|---|---|
| `FamilyWork.hot` + `.retained` | One input-type slab containing both lifetimes and, in BR, both roles plus a per-row selector | **Partial.** Hot/retained cannot release independently, retained data is copied, and BR role should be queue identity rather than a hot row field |
| `SigmaHead` | Generated by input partition; branched fields conditional | **High.** Matches intended old value/trace/cost ownership |
| `ActionHead` | Input-partitioned sequential, output-partitioned branched | **High.** Matches the core alignment idea |
| sampled `Leaf` | Board/profile/strata/payoff fields with opaque names | **High semantic fidelity.** Direct evaluator mapping |
| RB `Leaf` | Board/ranges/strata/payoff/seed; BR actor/cutoff | **Mostly high.** Extra cutoff is an undocumented extension |
| `Topology` / `TopologyChild` | One final data type, but two assemblers | **Schema high, pipeline low.** Final representation is shared; creation protocol is not |
| `FoldedRanges` | Natural-width range plus scalar `log_mass` | **Partial.** Widths are correct; normalized-only contract is not |
| `QueueMetadata` | Per-input one-row slabs | **Mostly high.** Tail/cost fields match; `controller_state` is dead |
| `LayerWork` | Schema-generated two-arena leading dimension | **Representation high, allocation low.** Exact plan is ignored and maximum capacity is always allocated |
| `SigmaTarget` | Schema-generated, target-published | **High.** Shared executor and correct conditional fields |
| `ActionTarget` | Correct partition axes and posterior applicability | **High.** Shared executor |
| `LearnerIndex` | Full retained payload plus metadata | **Low.** Contradicts metadata-only design and duplicates ownership |

### 4.1 What is faithful

The most important positive storage properties are present:

- fixed addresses after allocation;
- monotone non-wrapping tails;
- immutable declared capacities;
- SoA fields with explicit dtype/shape/alignment;
- device-local reservation and atomic multi-destination publication;
- lane-local failure state and disjoint retry epochs;
- estimator-specific payloads absent when inapplicable;
- natural `K`/`H` partition widths after the recent preflop-width fix;
- sampled scalar output and RB natural-domain output;
- action rows separated from sigma rows;
- two target work parities;
- no physical carry owner in the selected layer planner; and
- explicit last-consumer phase release.

These properties explain why the new engine is substantially more memory
predictable and faster than the deleted implementation.

### 4.2 What is not faithful

The important departures are not naming differences:

1. `FamilyWork` does not preserve a separately releasable retained lifetime.
2. `LearnerIndex` becomes a second retained payload owner.
3. self-play performs a seal copy despite already aligned rows.
4. stable prefixes are gathered rather than direct.
5. maximum-capacity `LayerWork` replaces exact realized allocation.
6. normalized-only folded storage becomes normalized range plus log mass.
7. unused reservation/publication/controller fields are generated.
8. folded references use fixed `P` width rather than folded-seat extent.
9. Leaf cutoff behavior extends the declared evaluator-only payload.
10. best-response hot rows mix model roles in one input-type queue and carry a
    redundant per-row `role_selector`.

The storage is therefore a good implementation of the broad fixed-SoA idea,
but not an exact realization of `storage_redesign.md`.

## 5. Acceleration-idea audit

| Acceleration idea | Status | Evidence / criticism |
|---|---|---|
| Family batching by semantic input type | **Implemented** | One FIFO per declared input partition and full-first/preorder scheduler |
| Stable monotone storage, no repacking during rollout | **Implemented** | Fixed slabs and monotone publication |
| Direct stable-prefix model batch | **Not implemented** | Advanced indexing gathers every column |
| Replica-vmapped inference | **Implemented** | Shared replica inference authority |
| Model/sample and commit streams overlap | **Implemented, bounded** | `RolloutTransaction` owns separate streams; host feedback limits pipeline depth |
| On-device counts/scans/reservation | **Implemented** | Large routing and synchronization win |
| No child-axis policy expansion | **Mostly** | No branching child expansion; selected-action path still expands legal actions over `H` |
| One joint sampled profile per tree | **Implemented** | Shared tree profile behavior and tests |
| Branched-RB child-zero reuse plus `N-1` draws | **Implemented semantically** | Correct law and recent safe-reuse fixes |
| Parent-CTA collision-conditioned sampler | **Not implemented** | Dense PyTorch `[G,A,P]` envelope |
| Action-conditioned chance proposal reuse | **Implemented semantically** | Profile propagation and logical addressing present |
| Bounded chance closure without per-generation sync | **Implemented** | One shared bounded loop; still multi-kernel Python orchestration |
| Sample/classify/scan/materialize/commit fusion | **Not implemented** | Separate PyTorch stages |
| Folded-stat/transition/chance fusion | **Not implemented** | Separate tensor operations |
| Post-rollout serialized completion | **Implemented as intended** | Rollout finishes, inactive cache is returned, then evaluator completion owns the GPU |
| Compact topology and unary compression | **Implemented** | Shared final types and layer plan; separate assemblers remain |
| Exact movement/layer plan | **Implemented logically** | Planner computes exact addresses |
| Exact physical work-arena allocation | **Not implemented** | Maximum `target_arena_cap` allocated |
| One CUDA kernel per target layer | **Implemented** | C++/CUDA loop; no inter-layer host sync |
| H-to-K mass/moment reduction before conditioning | **Implemented** | Target algebra and completion binding |
| Alias sampling for decision mass | **Implemented** | Custom CUDA alias construction and logical sampling |
| Direct selected-action policy operator | **Not implemented** | Full probabilities then gather |
| BF16 model-facing storage/compute with FP32 reductions | **Implemented conditionally** | Explicit model dtype and FP32 policy/target/loss controls |
| Immediate backward in bounded learner workspace | **Implemented** | Per physical batch |
| CUDA graph candidates | **Not implemented** | Design labels this optional |
| No CPU fallback in production | **Implemented** | CUDA/MPS gates and CUDA-only training |

### 5.1 Performance outcome versus architectural completion

The acceleration work should not be dismissed because the architecture is
incomplete. The current matched audit reports:

- 23.9% lower rollout time per retained decision than the July 24 baseline;
- roughly 580 times less generic routing traffic;
- 37.8 times fewer host synchronizations;
- 14.4 times faster child reservation/commit; and
- roughly 2,547 times faster target propagation.

Those are substantial results.

They also reveal where the remaining work is:

- model inference: 78.58% rollout-equivalent share;
- posterior sampler: 7.40%;
- selected-action policy: 4.26%;
- transition: 15.39% as an overlapping/nested scope; and
- folded/live resolution: 1.03% on the measured HU workload.

The next performance work should follow those measurements. The posterior
sampler and direct selected-action operator are justified now. A folded custom
kernel needs a representative multiway profile first.

## 6. Legacy, deprecated, and dead-code audit

### 6.1 Literal old engine

Git history and the current tree support the conclusion that the literal old
engine was removed:

- the Phase 11 cutover replaced the generic core;
- old modules such as the prior adaptive branching, completion, execution,
  loss, tree, and replica engines were deleted;
- the isolated `pbg/queued` replacement namespace no longer exists;
- no current import references `pbg.queued`;
- accepted roots reach the current `pbg.core`; and
- no CPU/auto fallback or runtime compatibility loader remains.

This is stronger than a simple “unreachable” claim: the old files are absent.

### 6.2 Current duplicate logic is not old legacy, but it is still duplication

The sequential/branched/replica implementations were created during the
rewrite, so calling them “legacy code” would be historically inaccurate.
Architecturally, however, they recreate the problem the rewrite intended to
remove: mode- and execution-specific authorities that can drift.

The audit therefore distinguishes:

- **old legacy implementation:** apparently deleted;
- **new parallel implementation:** clearly present; and
- **dead/unused surface:** not zero.

### 6.3 Concrete unused production data

The unused schema fields in F11 are the clearest current dead surface.
Because they are generated into every applicable slab, they are more than
comments or stale names.

### 6.4 Profiling-only bypasses in iteration engines

`benchmark_rollout_only` is a production constructor argument in both the
sequential and ordinary-branched iteration classes. It causes an early return
after rollout and manually walks later phases only to release owners:

- sequential branch: `training.py:772-774,926-954`;
- branched branch:
  `branching.py:2484,2645-2646,3633-3638`; and
- caller: `tools/profile-queued-branching.py`.

This is not an alternate training algorithm, but it is a second lifecycle
through production engine objects. The implementation plan says benchmarks
should run the production path and rejects diagnostic production modes. A
cleaner implementation would expose event instrumentation around the common
engine and let a profiling tool request a stop boundary outside the engine, or
clearly classify this as a non-production diagnostic API.

### 6.5 Naming residue

The `pbg/games/poker/queued/` package is the current implementation, not an
old one, but retaining “queued” after final cutover makes the architecture
look transitional. Names such as `SequentialSampledIteration` and
`BranchedSampledPokerIteration` now cover RB modes and are similarly stale.

These names do not create runtime duplication, but they increase cognitive
load and conceal actual applicability.

### 6.6 Limits of the reachability evidence

The current static audit found 476 indexed definitions and marked all
reachable. Ruff's selected unused-import/undefined-name checks also pass.
These are useful results.

They do not prove zero dead code because the reachability audit:

- indexes a class, not each method;
- does not track protocol methods or reflective calls precisely;
- does not inspect field-level reads;
- conservatively retains anything below a broad root; and
- cannot determine whether a reachable branch is feasible or obsolete.

A serious zero-dead-code gate needs:

- method-level AST/call analysis;
- field-read/write analysis for generated schemas;
- public-root runtime coverage across all twelve mode rows;
- branch coverage for accepted consumers;
- deletion review for zero-hit branches;
- explicit allowlists for reflection/protocol entry points; and
- a rule against duplicated lifecycle owners, which reachability alone cannot
  detect.

## 7. Core versus poker: recommended boundary

### 7.1 Core should own

```text
ModeSchema and execution dimensions
Capacity and owner planning
Phase/lifetime state machine
Semantic FIFO construction and scheduling
RolloutTransaction invocation
Topology packet acceptance and one assembler
Leaf FIFO scheduling
LayerPlan construction and upload
Align/seal mechanics
Layered target execution
Learner batch scheduling
Reference-cache lifecycle
Forward/loss/backward skeleton
Distributed failure/preflight protocol
Optimizer sequencing
Recovery boundary
Metrics timings around generic phases
Release and close
```

### 7.2 Poker should supply

```text
Poker geometry and partition declarations
Root state/profile creation
Model-input encoding
Model operation
Legality and policy composition
Action sampling
Chance sampling
Public transition
Joint/posterior profile sampling
Folded range update and folded statistics
Destination descriptors specific to poker state
Leaf materialization
Evaluator-v2 Leaf -> LayerWork binding
Poker-specific observability values
```

The core does not need to know cards, hands, streets, folds, or poker action
labels. Poker does not need to know how a training iteration sequences
completion, target publication, learner preflight, or final release.

### 7.3 Minimal hook surface

A compact hook bundle could contain pure operations similar to:

```text
make_roots(...)
model_forward(family_view)
sample(mode, family_view, raw_heads, rng)
transition(mode, family_view, sample_output)
topology_payload(sample_output, transition_output)
leaf_payload(transition_output)
completion_binding(mode, layer_work)
target_algebra(mode)
learner_policy(raw_heads, legality, selected_action)
```

The hook should return descriptors and values. It should not own queue loops,
phase transitions, storage release, learner optimization, or recovery.

## 8. Simplicity and elegance assessment

### 8.1 Quantitative shape

The most relevant source groups contain approximately:

- 14,594 lines in `pbg/core` Python/CUDA files;
- 12,054 lines in `pbg/games/poker/queued`; and
- 26,648 lines across the two groups.

Largest files:

| File | Lines |
|---|---:|
| `pbg/games/poker/queued/branching.py` | 5,008 |
| `pbg/core/rollout.py` | 3,075 |
| `pbg/games/poker/queued/hooks.py` | 2,567 |
| `pbg/core/learner.py` | 2,395 |
| `pbg/core/storage.py` | 1,770 |
| `pbg/games/poker/queued/training.py` | 1,549 |
| `pbg/core/topology.py` | 1,226 |
| `pbg/core/capacity.py` | 1,092 |

No target line count is inherently correct. A six-player poker engine with
distributed CUDA execution is genuinely complex. The issue is that the
largest file is a game-specific second engine, and several other largest files
mix multiple ownership domains.

### 8.2 What is elegant

- the three-axis schema generation;
- SoA schema-to-allocation planning;
- monotone multi-destination publication;
- logical RNG addressing;
- the generic transaction boundary;
- separation of sigma and action rows;
- compact topology and selected layer-plan implementation;
- one layered target CUDA authority;
- natural-domain RB targets;
- direct evaluator-v2 bindings; and
- explicit accelerator-only failures.

These are strong foundations worth preserving.

### 8.3 What is not

- a `TrainingSession` façade that dispatches three engines;
- 5,008 lines in a poker branching composer;
- topology assembly split by rollout mode;
- learner objective code repeated three times;
- optimization state machines repeated twice;
- retained payload copied into an object called `LearnerIndex`;
- comments and class names claiming “sole” or “direct” behavior that the code
  does not literally provide;
- benchmark branches embedded in lifecycle methods;
- exact target-arena sizes computed then discarded;
- storage fields generated without consumers; and
- closure reports that equate top-level reachability with architectural
  uniqueness.

The implementation is sophisticated, but it is not yet the simple, short,
modular implementation requested.

## 9. YAML configuration audit

### 9.1 Verdict

The YAML surface is substantially over-specified and is not well designed as
a user interface.

The public parser is commendably closed: unknown keys are rejected and many
mode-inapplicable top-level sections are rejected. That solves misspellings;
it does not solve contradictory authorities. A normal run file currently
contains:

- scientific choices;
- execution choices;
- values exactly derivable from those choices;
- the same value repeated in two or three locations;
- hardware facts that the process can query;
- implementation constants;
- conservative accounting estimates;
- every internal slab capacity;
- every transient pack capacity; and
- raw workspace byte estimates.

Those categories should not have equal status in editable YAML.

The 16 checked-in YAML files have 189 distinct scalar leaf paths when the
profiler-only scenario overlay is included. Every ordinary run repeats a
large `manifest` and `layout`. All 16 explicitly say `distributed: false`,
`attention_backend: sdpa`, `query_chunk_size: 128`,
`hardware.query_chunk_size: 128`, `field_alignment: 64`,
`reference_cache_bytes: 0`, and zero for both `required_external` values.
All also repeat `scientific.root_count` as `manifest.root_count`, game player
count as `layout.player_count`, scientific precision as `layout.model_dtype`,
and formulas already implemented in Python as literal dimensions and byte
counts.

This is not harmless verbosity. An editable derived value creates a second
authority. Some duplicates are checked when parsing, some only when storage is
built, some only when an iteration object is constructed, and some are merely
lower bounds that may be inflated without changing execution. Consequently a
bad file can:

- fail late after CUDA allocation;
- pass admission only because an accounting estimate is too small;
- waste memory because an estimate is too large;
- reduce learner pack size because a token bound is inflated;
- overflow because mutually dependent caps are not jointly realizable; or
- silently carry a field that has no consumer.

The design notes say that caps are fixed run hyperparameters. The
implementation is therefore fairly literal in this narrow respect. The new
requirement exposes a weakness in the originating design: fixed internal
addresses require an immutable *resolved manifest*, but do not require every
manifest entry to be typed by a human. The correct distinction is:

1. a small user specification;
2. a deterministic capacity/layout compiler producing an immutable resolved
   manifest; and
3. runtime observations and high-water telemetry.

Only the first should be authored as the normal run YAML. The second should be
recorded in checkpoints and reports. The third should never be accepted as a
claim from the user.

### 9.2 Concrete failure: 15 checked-in configs retain a stale input width

The redundancy has already caused repository drift.

For $P=2$, $S=32$, and $R=13$, current `NLHEGeometry` derives:

```text
pre     786
post_2  3100
```

Fifteen of the 16 checked-in YAML files instead declare:

```yaml
model_input_shapes:
  pre: [3100]
  post_2: [3100]
```

This includes all current H100/local-pilot YAMLs and the beta-2.4 profiler
fixture. The lone $R=4$, 8 bb smoke file agrees with its current derived
geometry. Commit history shows that the generator was subsequently changed by
`Fix preflop storage widths`; `pbg/h100_pilot.py` now emits the natural
preflop width, but the generated experiment files were not regenerated.

`load_config_mapping()` accepts the stale files because parsing only checks
that the shape map has the expected partition names. The exact natural-shape
check occurs later in the poker iteration, after the config has crossed into
CUDA runtime construction. The profiler fixture even documents the old value
as an intentionally “bloated rollout ABI,” despite current
`direct_model_input("pre", ...)` and `model_context(...)` using the natural
786-wide representation.

These 15 files are legacy configuration artifacts and should be regenerated
or deleted. More importantly, this is the failure mode the schema currently
invites: a formula was fixed in code while serialized copies silently remained
authoritative elsewhere.

### 9.3 Exact quantities the repository already knows how to derive

Let:

$$
D=4R,\qquad H={D\choose2},\qquad K=R^2,
$$

where $D$ is the compact-deck size, $H$ the exact private-hand count, and $K$
the preflop rank-class count. Let:

$$
B=\left\lceil SP+3P-S+1\right\rceil
$$

be the documented maximum player-node count in one trajectory and let
$T_h=B+3$ be the lossless public-history token bound including street
separators.

`NLHEGeometry` already implements all of the following:

$$
\begin{aligned}
W_{\text{public}} &= 8+10P,\\
W_{\text{history}} &= 10T_h,\\
W_{\text{pre}} &= W_{\text{public}}+PK+W_{\text{history}},\\
W_{\text{post-base}} &= W_{\text{public}}+PH+W_{\text{history}}.
\end{aligned}
$$

The postflop folded-summary width is:

$$
F(P,R)=
\begin{cases}
0,&P=2,\\
H+D+D^2+P+PD+2,&P>2,
\end{cases}
$$

so $W_{\text{post}}=W_{\text{post-base}}+F(P,R)$. Thus every
`layout.model_input_shapes` entry is exactly derivable. The iteration code
already rejects a shape that differs from `geometry.model_input_width(...)`.
The model factory likewise derives `max_history_tokens` and explicitly rejects
it as a user model field.

That is the right default. If lossy history truncation is ever desired, expose
an explicitly scientific `history.truncation_tokens` override and validate it
against $T_h$. Do not ask every lossless run to restate the formula as
`model_input_shapes` or a nominal “max history length.”

The model action width is also derived:

$$
k_{\text{act}}=3+m_{\text{raise}}.
$$

`max_policy_partition_sequence_tokens(...)` already computes the per-partition
learner token bounds from $P,S,R,k_{\text{act}}`. With
$n_{\text{ticks}}=2S$ and $C_s=3P+13+(3+n_{\text{ticks}})+n_{\text{ticks}}$:

$$
\begin{aligned}
M_{\text{pre}}&=PK+C_s+1+T_h+k_{\text{act}},\\
M_{\text{post}_A}&=A{D-3\choose2}+C_s+6+T_h+k_{\text{act}}.
\end{aligned}
$$

The current manifest's `model_token_bound` is $\max M_i$ and every checked-in
file sets `model_token_pair_bound` to its square. The H100 config generator
computes these values and then serializes them back into YAML. They are
resolved geometry, not user options.

The following layout values are equally direct:

- `player_count = P`;
- `evaluator_private_width = H` in RB modes and is inapplicable to sampled
  private-state storage;
- `model_dtype` follows `scientific.precision.name`;
- `field_alignment` is an internal allocator/backend constant; and
- `stratum_count` follows the poker/evaluator pot representation, not user
  taste.

For the active equal-stack game, side pots are impossible and the current
poker leaf builder only populates stratum zero, although it materializes a
$P$-wide amount vector. Whether the evaluator-facing resolved width remains
$P$ or is tightened to one is an implementation-contract decision. It is not
a YAML option.

Several workspace formulas already exist too:

- `sequential_family_workspace_row_bytes(geometry, m_raise)`;
- `sampled_branched_chance_frontier_row_bytes(geometry)`;
- `branched_rb_chance_frontier_row_bytes(geometry, max_attempts)`;
- `sampled_branched_topology_packet_row_bytes(geometry)`;
- `rb_conversion_workspace_row_bytes(H,K)`; and
- `rb_conversion_library_workspace_bytes(H,K)`.

The runtime checks some user literals against these formulas as lower bounds.
That is evidence the literal should be removed: the compiler should call the
formula once and store the result in the resolved manifest.

### 9.4 Topology and action equalities

The user's tree/forest objection is correct, with one naming qualification.
The implementation stores structural `Topology` rows and terminal `Leaf` rows
separately. Let $T$ be realized structural rows, $Q$ structural roots, and
$E_{\text{leaf}}$ leaf-target edges. `Topology.validate()` enforces exactly
one incoming edge for every non-root structural row. Therefore:

$$
E=(T-Q)+E_{\text{leaf}}.
$$

When every realized leaf row is a unique terminal endpoint,
$E_{\text{leaf}}=L$, hence:

$$
E=T+L-Q.
$$

If “nodes” means both structural and leaf nodes, $N=T+L$, this is the familiar
forest equality $E=N-Q$. The current fields use `topology_cap` for $T$ only,
so `topology_edge_cap = topology_cap - root_count` omits terminal edges if all
the named caps are meant to be simultaneously attainable.

This is not theoretical. `pbg/h100_pilot.py` generates:

```text
topology_edge_cap = topology_cap - roots
target_arena_cap = topology_cap + leaf_cap
```

The first equation treats leaves as absent while the second treats them as
independently saturable. The public validator merely requires:

```text
topology_edge_cap >= topology_cap - root_count
target_arena_cap >= topology_cap + leaf_cap
```

Those inequalities do not make the three maxima jointly realizable. For
example, the checked-in beta-2.4 profile declares $T_{\max}=36{,}000$,
$L_{\max}=20{,}000$, $Q=1$, and $E_{\max}=35{,}999$. A forest reaching both
node maxima would need $55{,}999$ edges. The independent caps are envelopes
obtained from profiling, not a coherent structural specification.

The clean alternatives are:

- expose one total forest budget and derive the edge bound;
- expose a hard tree-work budget, have a capacity compiler derive compatible
  owner bounds; or
- after topology is known, allocate exact topology/edge/layer extents under
  one admitted ceiling.

Users should not edit `topology_edge_cap` or `target_arena_cap`.

Action rows have an analogous identity. If $\mathcal U$ is the set of trained
topology rows and $N_{\text{tgt}}(s)$ is a trained row's target degree, then:

$$
A=\sum_{s\in\mathcal U}N_{\text{tgt}}(s).
$$

Sequential rows have $N_{\text{tgt}}=1$, so action rows and trained rows are
identical in count and row identity. The implementation explicitly requires
sequential sigma target, action start, and head row to match. Therefore
sequential `action_cap` should be derived exactly from the trained-row
capacity. In branched mode:

$$
U\le A\le \beta_{\max}U,
$$

and output-partition routing determines where those rows land. That routing
may prevent a tight per-partition formula before rollout, but it does not
justify unrelated hand-entered maps. Use one action arena or let the capacity
compiler derive conservative output-partition bounds from the trained-node
and branch envelopes.

### 9.5 `trained_player_cap` and balanced best-response roles

The conclusion is clear: `trained_player_cap` should not be a normal user
field.

For self-play, every player node is learner-owned, so the trained-row envelope
comes directly from the player-node envelope. For best response, only nodes
whose acting owner equals the selected response player are retained. The
current code assigns:

```text
role_selector = logical_root % P
```

in sequential, ordinary branched, and replica-branched root publication. Over
any contiguous logical-root interval, selected-seat root counts differ by at
most one, but the seats receiving the remainder are the deterministic
contiguous cyclic set selected by the logical-base offset.

The intended assignment preserves the exact balance while randomizing only
which seats receive the remainder. For $N$ roots, define:

$$
q=\left\lfloor\frac{N}{P}\right\rfloor,
\qquad
r=N-Pq.
$$

Every seat receives exactly $q$ selected-response roots. Draw $r$ distinct
seats without replacement using the logical RNG and give each one additional
root. Thus selected-seat root counts differ by at most one for every
iteration, including $N<P$, without the imbalance of independent per-root
assignment. The assignment must depend on logical root identity rather than
physical FIFO order so permuting root storage does not change semantics.
Seat labels have no independent training meaning, so no further shuffle is
required merely to decorrelate logical root and seat number. The pipeline
should instead prove equivariance under permutation of physical root rows
while their logical identities are preserved.

However, `ceil(player_node_cap / P)` is not automatically a safe equality if
`player_node_cap` is only an aggregate player-node envelope. Balanced *root
counts* do not prove balanced *player-node counts*: trajectory lengths differ,
and seats can act different numbers of times. The exact safe derivation needs
one additional invariant:

- derive the player-node cap as a $P$-symmetric sum of per-seat occurrence
  envelopes; or
- compute the BR trained cap from the maximally even selected-seat assignment
  and poker's per-seat occurrence bounds.

With that invariant established, a one-$P$th formula is appropriate. Without
it, merely deleting the field and applying `ceil(total/P)` can under-allocate.
The capacity compiler should own this proof. The user should never arbitrate
it.

The present validation is especially weak because it checks only the sums:

```text
sum(trained_player_cap) <= sum(player_node_cap)
sum(action_cap) >= sum(trained_player_cap)
```

It does not enforce per-partition containment. A large `post_2` cap can mask
an impossible `pre` trained cap. The branched beta-2.4 self-play fixture
illustrates the freedom: it declares 6,804 preflop player rows but only 84
preflop trained rows even though self-play has no frozen player. That may match
one observed high-water pattern, but it is not a structural self-play
equality.

### 9.6 Field-by-field disposition

The following classification treats “keep” as “keep in the user
specification,” not merely “the runtime currently reads it.”

#### Mode, game, model, and estimator

| Current field(s) | Disposition | Reason / replacement |
|---|---|---|
| `mode.training_regime`, `rollout_mode`, `private_estimator` | **Keep** | These are the three real mode axes. Prefer nested enums or one validated mode object, but all three distinctions are scientific. |
| `game.players`, `stack_bb`, `rank_count` | **Keep** | They are the active NLHE geometry dials and generate almost every physical shape. |
| `model.d_model`, `n_heads`, `n_layers_pre`, `n_layers_post`, `m_raise`, `s_min`, `class_embedding_dim`, `ffn_hidden`, `folder_beta_rank`, `folder_line_dim`, `epsilon` | **Keep** | These alter architecture or numerical policy. Defaults may shorten ordinary YAML, but resolved scientific config must record them. |
| `model.max_players` | **Default from $P$; expose only for portability** | It changes embeddings and FFNs, so it is not fake. In a fixed-$P$ run it is redundant. Rename to an explicit `seat_capacity` if cross-player-count checkpoint portability is supported. |
| `completion.n_samples` | **Keep in RB only** | It changes posterior/completion work and estimator variance. |
| `completion.strata` | **Delete now** | It is required and checkpointed but never consumed after parsing. Every checked-in value is `"default"`, and the parser does not validate its semantics. Add a real enum only when a second implemented stratifier exists. |

#### Loss, optimizer, schedule, and reference

| Current field(s) | Disposition | Reason / replacement |
|---|---|---|
| `lambda`, `clip_epsilon`, `value_coefficient`, `entropy_coefficient`, `reference_kl_coefficient` | **Keep** | These alter the objective/targets. |
| `value_loss`, `value_clip_epsilon` | **Use a discriminated union** | `value_clip_epsilon` is inapplicable for MSE yet is required in every file. Use `value_loss: {kind: clipped, epsilon: ...}` or `{kind: mse}`. |
| `entropy_schedule`, `entropy_decay_iterations` | **Use a discriminated union** | A constant schedule should not carry `null`; linear decay should own its horizon. |
| `trace_coefficient`, `cost_coefficient`, `trace_epsilon` | **Nest under branched loss** | They are required as zero/no-op fields in every sequential config. The mode schema already knows whether they apply. |
| optimizer name, learning rate, betas, epsilon, gradient clip | **Keep** | These are genuine optimizer choices. |
| `optimizer.weight_decay` | **Conditional** | Keep for AdamW; omit for Adam. `adamw` with zero weight decay is valid, though it may deserve the simpler `adam` spelling if implementations are otherwise identical. |
| `schedule.batch_size` | **Keep** | It is a statistical presentation choice, distinct from physical microbatch/pack size. |
| `schedule.epochs`, `gradient_steps` | **Make mutually exclusive** | When `gradient_steps` is set, `epochs` is ignored by the main fixed-step paths but is still required and serialized. Use `{kind: epochs, count: ...}` or `{kind: steps, count: ...}`. |
| reference interval fields | **Keep one cadence variant** | Exactly one is already required. Represent it without a mandatory `null`, e.g. `refresh: {iterations: 100}` or `{effective_hands: ...}`. In sampled mode the “effective hands” counter intentionally uses node exposure; rename the generic concept if this is retained. |
| `reference.cache_outputs` | **Move to execution and merge with its budget** | It changes performance, not the objective. `off`, `auto`, or an optional byte budget is one authority; a boolean plus `layout.reference_cache_bytes` is two. If the KL coefficient is zero and no other reference consumer exists, the entire reference section should be inapplicable. |

#### Branching and recovery

| Current field(s) | Disposition | Reason / replacement |
|---|---|---|
| `branching.fixed_beta` | **Keep conditionally** | It is the fixed controller intensity and the KKT warmup/chance baseline. Stage-specific player beta is a real controller choice if deliberately supported. |
| `branching.beta_max` | **Keep once** | It is the common degree bound. Sequential derives one. Delete `manifest.beta_max`; the resolved manifest copies this authority. |
| `minimum_scale`, `backoff_factor` | **Keep as recovery policy, with defaults** | They change emergency controller recovery. Nest them under `recovery`, not storage. |
| `minimum_tree_size` | **Keep only if undersized trees are scientifically rejected** | It is a post-rollout validity threshold. It must be cross-validated against target, safe, and hard tree budgets. |
| `safe_tree_size` | **Derive from admitted capacity or expose one hard tree budget** | It suppresses branching based on predicted work but currently has no structural relation to topology, edge, leaf, or memory caps. |
| KKT target/gain/kappa fields | **Keep under KKT only** | They are genuine adaptive-controller parameters. Enforce `minimum_tree_size <= target_tree_size <= safe_tree_size` and controller/capacity compatibility. |
| `cost_safeguard.*` | **Keep under branching, with defaults/profile** | The fields are consumed and can change branch suppression. They are controller calibration, not a top-level universal loss or storage concern. |
| `root_count_minimum` | **Keep as an optional recovery floor** | It is not redundant with root count when overflow recovery is enabled. Default it from policy and name it `recovery.minimum_roots`. |

#### Workload and execution

| Current field(s) | Disposition | Reason / replacement |
|---|---|---|
| `scientific.root_count` and `manifest.root_count` | **Keep once outside the manifest** | Root count is workload/statistical sample size. The parser already does nothing but require equality between the copies. |
| `n_replica` | **Keep once as workload/execution** | It is a real number of independent model lanes, not geometry. |
| `seed`, `reproducibility`, `utility` | **Keep** | They affect stochastic identity, scheduling guarantees, and payoff scale. Their exact placement may be `run` rather than `scientific`. |
| `device` | **Keep explicit** | Accelerator-only policy requires an explicit CUDA target. |
| `iterations`, `checkpoint_every` | **Keep** | They are ordinary run-control choices. A checkpoint interval beyond total iterations can be diagnosed but is not contradictory. |
| `output_dir`, `run_id` | **Simplify** | `run_id` is useful checkpoint identity, but checked-in files repeat it in the directory basename. Default one from the other or from the config stem; permit an explicit override. |
| `distributed` | **Keep as launcher/execution policy or infer from launch context** | It is not scientific. If kept, validate it immediately against process-group/world-size state. |
| `frozen_checkpoint` and `frozen_profile.sha256` | **Keep both in BR** | Path and content identity are not duplicates. One locates bytes; the other binds scientific identity. A single `{path, sha256}` object would make the relationship clearer. |
| `response_initialization` and probe fields | **Keep in BR/probe only** | These are real behavior or diagnostic choices and are already mode-gated. |
| `attention_backend`, `query_chunk_size` | **Use a discriminated union** | `query_chunk_size` is irrelevant for SDPA, yet all files specify it. Use `attention: sdpa` or `attention: {kind: chunked_dense, query_rows: ...}`. |
| `hardware.query_chunk_size` | **Delete** | It duplicates the execution value and is not read as the model setting. |

#### Capacity manifest

| Current field(s) | Disposition | Reason / replacement |
|---|---|---|
| `model_token_bound`, `model_token_pair_bound` | **Derive** | Exact formulas already exist. Under-declaration fails late; over-declaration silently reduces physical batch capacity. |
| `player_node_cap` | **Compile, not hand-author** | Sequential total is bounded by `roots * B`; partition envelopes are poker geometry/planner work. Branched storage should derive from one tree/work budget and admitted memory. |
| `trained_player_cap` | **Derive** | Self-play and BR role geometry determine it; see Section 9.5. |
| `action_cap` | **Derive** | It follows trained rows and target degrees; sequential is exact equality. |
| `folded_range_cap` | **Derive and omit for HU when proven unused** | Fold rows follow fold endpoints/actions. HU has zero folded-summary width and a fold is terminal, yet all files must reserve at least one row because the generic manifest forbids zero. |
| `leaf_cap` | **Derive from root/tree budget** | Sequential has one terminal leaf per realized root; branching leaf extent follows the tree envelope. A user may choose the high-level work budget, not an unrelated leaf maximum. |
| `topology_cap`, `topology_edge_cap` | **Replace with one coherent forest/tree budget** | Their realized counts obey the forest identity above. |
| `target_arena_cap` | **Derive or allocate exactly after planning** | It is currently required to cover `leaf_cap + topology_cap`; making all three editable adds no freedom. |
| `family_pack_cap` | **Optional performance profile input** | A rollout microbatch size can be tunable, but should be named as such and constrained by memory/compiler output. |
| `chance_frontier_cap`, `topology_packet_cap` | **Derive from rollout pack, beta, and packet algorithm** | They are downstream transient concurrency, not independent scientific choices. |
| `completion_pack_cap` | **Optional evaluator microbatch input or auto-tune** | This is a legitimate launch/performance knob, best supplied by a named hardware profile or autotuner. |
| `learner_sigma_pack_cap`, `learner_action_pack_cap` | **Derive from statistical batch and token/pair budgets** | Their current independent values can contradict action/sigma geometry. Sequential should share one physical row cap. |
| movement/library/evaluator workspace bytes | **Derive/query** | Movement follows exact plan sizes; conversion formulas exist; evaluator/library contracts should report their required workspace. These are not user choices. |
| `memory_guard_bytes`, `memory_budget_bytes` | **Keep one memory policy** | Let the user choose a reserve or usable-memory fraction. Compute the budget from the selected device and admitted external work. Do not ask for budget, guard, workspaces, and total memory independently. |
| `required_external.library_bytes`, `.evaluator_bytes` | **Delete from user YAML** | Every checked-in value is zero. Required external workspace is a versioned backend fact; zero currently only disables a redundant lower-bound check. |

#### Hardware and physical layout

| Current field(s) | Disposition | Reason / replacement |
|---|---|---|
| `hardware.total_memory_bytes` | **Observe at runtime** | CUDA reports it. A user literal can only be stale or intentionally understate the device. Use a memory policy to reserve capacity instead. |
| `hardware.memory_budget_bytes` | **Derived from one policy** | See above. Record the resolved value for reproducibility. |
| `training_max_tokens`, `training_max_token_pairs` | **Hardware/profile policy, not geometry** | Per-launch budgets may be tuned, but checked-in H100 files generate them from `learner_pack * derived_bound`, making them a circular restatement of pack size. Keep canonical launch budgets in a named hardware profile and derive partition pack sizes. |
| all `model_input_shapes` | **Derive** | Exact formulas and runtime equality checks already exist. |
| `layout.player_count`, `evaluator_private_width`, `stratum_count` | **Derive** | They are game/evaluator geometry. Make mode-inapplicable fields absent. |
| family/chance/completion/topology row-byte fields | **Derive** | The implementation contains exact or conservative formula functions and validates several as lower bounds. |
| learner sigma/action row-byte fields | **Delete as free estimates** | They are used only in admission accounting and have no implementation-derived lower-bound check. Checked-in values range from 4 KiB to 32 MiB per pack row. A typo can admit an unsafe run or reject a safe one without changing any learner allocation. Admission must calculate actual concurrent tensors/operations. |
| `reference_cache_bytes` | **Merge with cache policy** | Zero means disabled in every checked-in file. When enabled, derive required/resident bytes from exact cached heads or treat one byte budget as the sole execution input. |
| `field_alignment` | **Internal constant** | All files say 64. Expose only to backend developers/tests, never ordinary training YAML. |
| `model_dtype` | **Delete duplicate** | It must follow `precision.name`. A mismatch is nonsensical. |

### 9.7 Cross-field checks that are missing or too late

The schema's closed-key discipline should be preserved, but validation needs
to move from independent type/range checks to construction by derivation.
Today:

- scientific/manifest root counts are equality-checked during parsing;
- controller/manifest `beta_max` equality is checked only when a branched
  iteration is constructed;
- model input shapes are checked only when the poker iteration is
  constructed;
- player count and broad capacity inequalities are checked while building the
  CUDA storage plan;
- declared model token bounds are recomputed and checked in the learner;
- workspace row bytes are often only checked as lower bounds;
- learner row-byte estimates have no corresponding actual-workspace check;
- total memory is only checked not to exceed the physical device;
- `precision.name` and `layout.model_dtype` have separate authorities;
- the two query-chunk fields have separate authorities; and
- `completion.strata` has no runtime consumer.

Important controller relations are also absent. The parser does not establish
that KKT target size lies between minimum and safe tree sizes or that
`safe_tree_size` fits the admitted topology/leaf/edge capacities. Capacity
validation checks aggregate trained/action rows, not the corresponding
partition maps. A generated config can therefore be syntactically valid and
individually range-valid while remaining structurally incoherent.

The solution is not a larger web of equality assertions. Assertions are a
migration safety net. The durable solution is to stop accepting the derived
side of each equality.

### 9.8 Recommended schema architecture

Use three explicit data products:

#### A. User-authored run specification

This contains only algorithm, model, workload, recovery policy, and a small
number of optional performance preferences. A representative shape is:

```yaml
scientific:
  mode:
    training_regime: self_play
    rollout_mode: sequential
    private_estimator: sampled
  game: {players: 2, stack_bb: 32.0, rank_count: 13}
  model:
    d_model: 128
    n_heads: 4
    n_layers: {pre: 4, post: 6}
    raises: {components: 2, minimum_scale: 0.02}
    # Remaining architecture fields use versioned model defaults or are explicit.
  objective:
    policy_clip: 0.2
    value: {kind: clipped, epsilon: 0.2, coefficient: 0.125}
    entropy: {kind: linear_decay, initial: 0.025, iterations: 500}
    reference_kl: {coefficient: 0.05, refresh: {effective_hands: 900000}}
  optimizer:
    kind: adamw
    learning_rate: 5.0e-5
    weight_decay: 0.0
  schedule: {batch_size: 256, steps: 116}
  lambda: 0.95
  utility: stack_normalized

run:
  seed: 104830
  reproducibility: throughput
  workload: {roots: 512, replicas: 4}
  recovery: {minimum_roots: 128}
  execution:
    device: cuda:0
    precision: bf16
    attention: sdpa
    memory: {reserve_bytes: 10737418240}
    microbatch: auto
  iterations: 1
  checkpoint_every: 1
  output_dir: runs/experiments/...
```

RB adds `completion: {samples: ...}`. Fixed branching adds its intensities.
KKT adds only KKT fields. Best response adds one frozen-profile object. No
inactive field is represented by `null` or zero.

Versioned defaults are acceptable for convenience only if the fully resolved
scientific configuration is checkpointed and hashed. Silent default changes
across versions are not acceptable.

#### B. Compiler-produced resolved manifest

The compiler takes the user spec, mode schema, poker hooks, evaluator/library
contracts, selected device, and optional named performance profile. It emits:

- exact geometry and input/token shapes;
- one coherent set of owner capacities;
- exact or proven conservative transient workspaces;
- per-partition physical microbatch limits;
- device-observed memory and the applied reserve;
- phase admission totals; and
- derivation/version identifiers.

This manifest remains immutable for fixed-address storage and is serialized
with the checkpoint. It may be exportable as YAML/JSON for diagnosis, but
should be marked generated and rejected as a normal source config unless an
expert “replay exact manifest” mode validates its hash and all derivations.

#### C. Runtime telemetry

Actual free memory, allocated/reserved bytes, high-water rows, exact topology
counts, pack utilization, overflow demand, and controller backoffs belong in
metrics. They must not feed back as user-declared “hardware facts.”

### 9.9 Profiler YAML is a separate file type

`configs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing.yaml` is not a
public training config. It has a third top-level `profile` object and is
intentionally rejected by `pbg.config.load_config`; only
`tools/profile-queued-branching.py` accepts it. Its scenarios repeat and
override beta, completion samples, safe/minimum tree size, seed, iteration,
safeguard state, output path, and expected result.

That is reasonable as a frozen diagnostic fixture, but placing it beside
normal YAMLs under the same naming convention obscures which schema applies.
Give profiler fixtures a distinct suffix or directory and a top-level schema
version such as `kind: branching_profile_v1`. Scenario overrides should be
compiled into ordinary resolved run specs rather than mutating a base mapping
through another implicit configuration layer.

### 9.10 Configuration acceptance criteria

Add these to the corrected-design gate:

1. A normal user config contains no `layout` and no raw owner/workspace byte
   manifest.
2. Every accepted user field changes scientific behavior, run control,
   recovery policy, or an explicitly documented performance preference.
3. Every derived manifest field has exactly one code derivation and records
   its derivation version.
4. No active schema uses `null` or zero placeholders for inapplicable fields.
5. `completion.strata` is removed until multiple stratifiers exist.
6. Sequential action capacity is derived from trained-row capacity.
7. BR selected-seat roots are maximally even, remainder seats are selected
   without replacement by logical RNG, and trained capacity is proven from
   per-seat occurrence bounds.
8. Topology, leaf, and edge bounds are jointly coherent under
   $E=(T-Q)+E_{\text{leaf}}$.
9. Model input, history, private-domain, token, and token-pair shapes are
   derived from $P,S,R$ and model action width.
10. Precision has one authority; attention chunk size exists only for the
    chunked backend.
11. Runtime queries device memory; the user supplies at most one reserve or
    usable-memory policy.
12. Admission uses implementation-derived concurrent workspace, not arbitrary
    per-row byte estimates.
13. A resolved-manifest round trip is checkpoint-stable and source-YAML
    canonicalization is deterministic.
14. Parser tests prove every remaining field has a production consumer and
    every mode-inapplicable field is rejected.
15. Mutation tests independently perturb every generated field and prove
    either deterministic recomputation or immediate pre-allocation rejection.

## 10. Recommended correction sequence

The order matters. Kernel optimization before unifying lifecycle ownership
would make the current forks harder to remove.

### Step 1 — freeze behavior and define a structural anti-duplication gate

Add a twelve-mode × ordinary/replica/distributed trace fixture that records:

- phase sequence;
- hook calls;
- queue publications;
- topology packets;
- completion calls;
- learner batch descriptors;
- optimizer transitions; and
- releases.

Require the same generic engine class and phase-state machine in every trace.
Allow differences only in schema payloads, hook selection, row multiplicity,
and aligned/ragged batch descriptors.

Static gates should reject:

- another class that owns all phases;
- another topology assembler;
- another optimizer-loop skeleton; and
- poker code calling `owners.enter_phase`, `release_after`, or `close`.

### Step 2 — create one core iteration engine

Move lifecycle logic out of poker. The engine should own one loop over
semantic FIFO cohorts and invoke poker hooks through the existing
`RolloutTransaction`.

Key best-response cohorts by `(replica, input_type, role)` so every stable
prefix uses one model-weight set. Route commits directly to the learned or
frozen-opponent queue and remove per-call split/merge compaction. Do not gather
every hot field into the model batch.

Make `n_replica=1` an ordinary lane count, not a constructor choice.
Distributed execution should wrap failure agreement and optimizer collectives,
not select a different branched iteration.

Delete:

- `SequentialPokerIteration` as an engine;
- `_BranchedSampledPokerAttempt` as an engine;
- `ReplicaBranchedPokerIteration` as an engine; and
- their mode-specific recovery wrappers after recovery moves to the common
  boundary.

Poker result formatting can remain a small adapter over generic iteration
metrics.

### Step 3 — unify topology publication

Define one packet schema carrying degrees, labels, endpoints, trained owner,
family/head/action rows, and path length.

Sequential sends degree-one packets. The assembler may compress unary paths
internally, but it must be the same assembler and acceptance protocol.

Preserve the compact final `Topology`, `TopologyBindings`, and `LayerPlan`;
those are good.

### Step 4 — repair retained-input ownership

Split the physical lifetime of rollout-hot and learner-retained payloads.
Practical options are:

1. two schema-generated backing owners published by one transaction; or
2. a schema object with independently owned field arenas and lifetime groups.

The first is simpler:

```text
FamilyHot[input_type, role]  root/child commit -> queue drain
FamilyRetained[input_type]   learned model commit -> final backward
LearnerIndex[input_type]     branched learner metadata only
```

The role axis is absent in self-play. In best response, queue identity selects
learned versus frozen-opponent inference, so delete `role_selector` from the
hot row schema. Preserve the selected response player once per root through
the root/topology authority needed to route later actors; do not duplicate it
on every family row.

Learned hot and retained rows can share a logical row address. Self-play then
needs no payload copy, and best response publishes only learned rows to the
retained owner. Sequential learning consumes aligned retained owners directly;
branched learning additionally consumes the metadata-only `LearnerIndex`.

Delete the unused reservation/publication columns and either implement or
delete queue `controller_state`.

### Step 5 — unify learner batches and objective skeleton

Use one `DecisionBatch` protocol:

```text
sigma_index
action_ptr
action_index
sigma_fields
action_fields
```

Sequential spans all have length one. Its sampler may still generate a simple
permutation and zero-copy views; it need not build expensive alias metadata.

Write one forward/reference/policy/value/clipping/backward skeleton. Plug in:

- sampled versus RB private reduction;
- optional branched trace/cost losses; and
- objective row weights.

Then use one optimization loop for logical-step boundaries, failure agreement,
metrics, and optimizer steps.

### Step 6 — make stable prefixes truly direct

For one lane, return simple contiguous slices of only the model-required
fields. For replica cohorts, arrange lane/head invariants so aligned prefixes
can be viewed as `[lane,row,...]`; pack only when heads differ and account for
that workspace explicitly.

Do not gather every hot field into the model batch.

Profile the change because model inference dominates rollout.

### Step 7 — allocate exact `LayerWork`

Keep `target_arena_cap` as a hard ceiling. After the one topology/layer planner
returns exact extents:

- validate the extents;
- perform exact phase admission;
- allocate fixed slabs at those extents;
- publish completion rows; and
- retain those exact slabs through target publication.

Use the existing capacity audit's byte calculations as the acceptance oracle.

### Step 8 — preserve a serialized completion boundary

After the common lifecycle exists, retain the explicit post-rollout completion
phase:

```text
finish rollout
  -> release rollout-only allocations
  -> synchronize
  -> empty the PyTorch cache
  -> run evaluator completion
```

Do not overlap PyTorch rollout work with poker-gpu-evaluator completion.
Workspace remains part of immutable capacity admission even though returning
inactive cached blocks improves driver-visible availability.

### Step 9 — implement the measured poker kernels

In measured-priority order:

1. fused parent-CTA collision-conditioned posterior sampler;
2. direct selected-action log-policy formulation;
3. model-level CUDA/PyTorch profiler investigation;
4. only then folded/transition fusion if a multiway workload justifies it.

Keep one clear PyTorch reference implementation in tests, not as a selectable
production fallback.

### Step 10 — compact and rename

After the unified engine is validated:

- rename the final poker package away from `queued`;
- rename generic RB-capable classes that still say `Sampled`;
- remove profiling-only lifecycle branches or move them to an explicit
  diagnostic boundary;
- split files by owner, not by mode;
- rerun method/field-level dead-code analysis; and
- update closure documents to state measured facts rather than “sole” claims
  inferred from reachability.

### Step 11 — replace hand-authored manifests with a capacity compiler

Preserve `CapacityManifest` as the immutable internal contract, but stop
parsing its derived rows, shapes, alignments, and workspace bytes from normal
run YAML.

Implement one compiler over the mode schema, game geometry, selected device,
backend contracts, and a small workload/memory policy. It should emit and hash
the fully resolved manifest, record every derivation version, and serialize
the result in checkpoints and audit output.

During migration, parse old manifests only through an explicitly versioned
legacy-config importer that recomputes every derived field and rejects any
difference. Delete that importer when checked-in configs and checkpoints have
been migrated; do not retain it as a permanent second production schema.

## 11. Acceptance criteria for the corrected design

The refactor should not be declared complete until all of the following are
mechanically demonstrable:

1. `TrainingSession.step()` constructs one iteration-engine type for every
   mode, lane count, and distributed setting.
2. Exactly one source method owns the rollout-to-final-release phase sequence.
3. Exactly one scheduler loop exists.
4. Exactly one topology packet consumer/assembler exists.
5. Sequential packets have degree one and do not invoke a separate assembler.
6. Exactly one learner optimization loop exists.
7. Exactly one policy/value loss skeleton exists; schema-selected terms are
   small functions.
8. Poker modules never advance storage phases or close generated owners.
9. `LearnerIndex` contains metadata only.
10. Self-play retained inputs are not copied at seal.
11. Stable one-lane prefixes are storage-sharing views, proven by data pointer
    and stride tests.
12. `LayerWork` allocated bytes equal exact admitted layer-plan extents.
13. Completion begins only after rollout finishes, rollout-only allocations
    are released, the device is synchronized, and the inactive PyTorch cache
    has been returned.
14. Best-response FIFO identity includes input type and learned/frozen role;
    `FamilyWork.hot` contains no per-row role field.
15. For $N$ best-response roots and $P$ seats, every seat receives
    $\lfloor N/P\rfloor$ roots and exactly $N-P\lfloor N/P\rfloor$ logically
    randomized distinct seats receive one more; permuting physical root rows
    while preserving logical identities leaves results equivariant.
16. Generated schema fields all have a production writer and reader, or are
    deleted.
17. The posterior and selected-policy performance gates meet the thresholds
    already proposed in
    `docs/audit/high_priority_cuda_kernels_20260728.md`.
18. All twelve modes complete an optimizer step on CUDA through the common
    engine.
19. Ordinary and one-lane replica traces are identical up to declared lane
    indexing.
20. Multi-lane fixed/KKT-compatible replicas share inference without a
    replica-specific iteration implementation.
21. Current matched-work performance and memory gates do not regress.
22. A method/field-level audit reports no unexplained production-only dead
    surface.
23. Normal run YAML contains no raw layout, owner-capacity, or workspace-byte
    manifest; one compiler emits the immutable resolved manifest.
24. Every accepted user field has a mode-applicable production consumer, and
    every derived field has one code authority.

## 12. Validation performed for this audit

Read-only/static checks at the audited working tree:

```text
tools/audit_reachability.py --json
  476 indexed definitions
  476 conservatively reachable
  0 deletion candidates

ruff check --select F401,F841,F811,F821 pbg tests tools
  passed

Focused pytest selection
  43 passed, 58 skipped in 5.30s

Full CUDA suite, `CUDA_VISIBLE_DEVICES=1`
  319 passed, 2 skipped in 168.42s
  skips: unavailable MPS; explicit two-CUDA-device distributed gate

Configuration-addendum focused pytest
  25 passed, 7 skipped in 4.85s

Checked-in YAML derivation audit
  16 files inspected
  15 stale preflop model-input widths
  root/player/token/pair/chunk/dtype duplicates otherwise agree

Final full CUDA suite on the working-tree overlay, `CUDA_VISIBLE_DEVICES=1`
  320 passed, 2 skipped in 116.17s
  GPU 1 returned to 6 MiB and 0% utilization

Final F401/F841/F811/F821 lint
  pbg + tests: passed
  pbg + tests + tools: one F401 in the concurrently edited profiler tool
```

The focused selection covered imports, mode contracts, storage transactions,
rollout transactions, layered targets, reachability, and accelerator-only
runtime behavior. In the restricted process environment, PyTorch could not
initialize CUDA, so CUDA-marked cases were skipped. This result is static/CPU
evidence only.

The full suite then ran outside that restricted process environment with the
repository venv and its exact CUDA 13 toolkit paths. GPU 1 was selected
explicitly after all three GPUs were confirmed idle. The suite passed 319
tests; the only skips were the unavailable-MPS test and the test requiring two
visible CUDA devices. Afterward GPU 1 returned to its 6 MiB, 0%-utilization
idle baseline with no compute process. This is the repository-required CUDA
gate for the documentation change; it validates current behavior but does not
alter the structural findings in this report.

After the configuration addendum, focused config, mode-contract, and H100
pilot tests passed. A formula-based pass over every checked-in YAML then found
the 15 stale preflop widths documented in Section 9.2. At that point all three
GPUs were occupied by existing Python workloads, so no competing CUDA job was
launched. When GPU 1 returned to its 6 MiB idle baseline, the final full suite
was run there against the working-tree overlay and passed 320 tests with the
same two environmental skips. GPU 1 returned to the same clean baseline after
the run; the existing jobs on GPUs 0 and 2 were not touched.

The final production/test lint subset is clean. Including `tools/` reports one
unused `BranchedSampledPokerHooks` import in
`tools/profile-queued-branching.py`, which is among the concurrent
working-tree edits and is outside production. This report records that final
overlay state rather than modifying the tool.

The audit also inspected:

- current GPU process state before requesting external CUDA validation;
- exact venv CUDA toolkit paths;
- all current custom CUDA sources;
- current production references to generated owner fields;
- Git deletion history and commits after Phase 11;
- Phase 1–11 closure reports and migration ledger;
- the current branching performance audit; and
- the HU beta-2.4 execution/capacity audit.

No production source was changed by this audit. The only new artifact is this
report.

## Final judgment

The rewrite successfully replaced the old engine with a substantially better
GPU-native foundation. It established strong fixed storage, transactional
publication, logical RNG, compact targets, and accelerator-only behavior.
Those achievements are real and should be preserved.

It did **not** finish the architectural cleanup promised by the originating
design. The mode schema is unified, but the iteration pipeline is not. The
target executor is unified, but topology assembly and learner execution are
not. The dependency firewall is clean, but generic orchestration is still in
poker. Storage is fixed and explicit, but retained payload ownership and
exact arena allocation diverge from the design. Performance is much improved,
but several of the most important proposed kernels and overlaps remain
unimplemented.

The configuration surface repeats the same architectural mistake at a
different level: a strong internal manifest is exposed as dozens of
hand-authored authorities. Shapes, tree relations, trained/action rows,
workspace bytes, and hardware observations should be compiled, not configured.

Accordingly, the answer to “do we have the simple, short, modular and elegant
implementation we asked for?” is:

> **Not yet. We have the right substrate, but it is wrapped in too many
> pipelines.**
