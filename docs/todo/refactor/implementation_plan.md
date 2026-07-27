# Queued-core rewrite and repository cleanup

Status: conditional implementation plan derived from
[`storage_redesign.md`](storage_redesign.md). It becomes executable only after
the authority and external-interface gates in Section 5 close. This plan
cannot approve changes to the three external authorities or the evaluator
ownership boundary.

## 1. Goal

This rewrite is a cleanup. Its value is a training engine that is easier to
understand, smaller in concepts, and easier to change without creating another
mode-specific path.

The final engine has:

- one generated path for all twelve modes;
- one owner per lifetime and one authority per operation;
- shared abstractions for genuinely shared behavior;
- small schema-driven differences where modes differ;
- no legacy implementation, compatibility path, stale option, or runtime
  implementation selector;
- no production object retained only by tests;
- bounded accelerator storage and at least the accepted CUDA performance.

The migration follows a Ship-of-Theseus rule: replace one coherent piece,
validate it, audit it, document it, and continue until none of the old engine
remains.

### 1.1 Refactorability rule

Before introducing or carrying forward an object, ask:

1. What invariant or lifetime does it own?
2. Can it be merged with another object that owns the same thing?
3. Is its variation only a field, shape, constant, or function selected by the
   immutable mode schema?
4. Can it be a pure function instead of a stateful object?
5. Which old object or branch does it replace?
6. Does it make the pipeline easier to explain?

Objects own invariants and lifetimes. Functions transform data. Schemas express
applicability. Do not add a class hierarchy for modes, generic field lookup,
pass-through managers, one-method wrappers, compatibility DTOs, or speculative
extension points.

Prefer direct, readable code and delete duplication. Never trade clarity,
validation, or ownership for a superficial compactness metric.

### 1.2 One path

The axes are:

```text
training_regime   self_play | best_response
rollout_mode      sequential | branched.fixed | branched.KKT
private_estimator sampled | rao_blackwell
```

They generate twelve mode rows. `n_replica`, rank ownership, and `root_count`
are execution dimensions, not new modes.

A mode row may change fields, constants, sampler law, row multiplicity, and
learner sampling. It may not select a different storage engine, queue engine,
target builder, learner, or optimizer.

```text
ModeSchema
  -> semantic family FIFOs
  -> replica-vmapped model inference -> sample -> commit
  -> Leaf -> LayerWork
  -> CPU Topology -> GPU LayerPlan
  -> align-and-seal
  -> layered target executor
  -> aligned or ragged learner sampling
  -> one authoritative forward/loss/backward per learner lane
  -> dense optimizer transition
```

Sequential execution is the
$N=N_{\rm tgt}=\widetilde N=1$ specialization. Fixed and KKT branching differ
through one controller interface. Best response differs through trained versus
frozen ownership. Sampled and RB differ through generated payloads and
completion/target algebra.

## 2. Authority and migration boundary

Use sources in this order:

1. [`docs/theory/core_math.tex`](../../theory/core_math.tex) for sampling,
   estimators, targets, and losses.
2. [`docs/theory/MHA_model.md`](../../theory/MHA_model.md) for the model
   operation and architecture.
3. [`docs/theory/folder_impact.md`](../../theory/folder_impact.md) for folded
   poker inputs and their architectural impact.
4. [`storage_redesign.md`](storage_redesign.md) for the proposed storage,
   execution, ownership, and iteration contract once this refactor folder is
   approved.
5. [`layered_target_creation.md`](layered_target_creation.md) for the sole
   target executor.
6. The focused poker proposals:
   [`poker_branched_rb_sample.md`](poker_branched_rb_sample.md),
   [`poker_folded_statistics.md`](poker_folded_statistics.md),
   and [`poker_policy_cuda_operator.md`](poker_policy_cuda_operator.md).
7. [`features_to_preserve.md`](features_to_preserve.md), plus current
   `pbg/core`, accepted consumers, tests, audits, and git history, for
   functionality that must survive the rewrite.
8. Audit evidence, including:
   [`replica_learner_parity_and_rb_determinism.md`](../../audit/replica_learner_parity_and_rb_determinism.md).

`docs/theory/pseudocode/` records superseded implementation thinking. It may
explain history, but it is not an authority, prerequisite, or regression
oracle for this rewrite.

Do not implement against a known disagreement between this folder and the
three external authorities. Before Phase 1, record every relevant conflict
clause-by-clause, obtain explicit approval for the intended resolution, and
update an external authority only in a separately approved change. Current
code and history establish required functionality and evidence about prior
behavior, not the new internal design.

Useful old-code evidence:

| Concern | Current evidence |
|---|---|
| orchestration and modes | `pbg/core/training.py`, `config.py` |
| storage, queues, sealing | `storage.py`, `execution.py`, `tree.py` |
| rollout and branching | `rollout.py`, `branching.py`, `adaptive_branching.py` |
| topology and targets | `topology.py`, `targets.py`, `completion.py` |
| learner and optimizer | `learner.py`, `optimization.py`, `losses.py` |
| replica divergence | `pbg/core/replica_*.py` |
| poker integration | `pbg/games/poker/` |
| consumers | runtime, CLI, evaluation, inference, benchmarks, and tools |
| evaluator API | `/home/pablob/poker-gpu-evaluator/docs/api.md` (read-only) |

### 2.1 Isolated replacement

The replacement lives under `pbg/queued/` until cutover.

- `pbg/queued/` never imports `pbg/core/`.
- Generic queued core never imports poker, evaluation code, or the sibling
  evaluator.
- Poker implements opaque core hooks.
- Comparison tests may invoke both engines independently; representations
  never cross.
- Production never selects between engines.
- Ownership-coupled engine behavior is rewritten from its contract, not copied
  into a second implementation. A representation-independent utility may move
  once to a neutral final owner after all accepted callers migrate.

Create `docs/audit/storage_redesign_migration_ledger.md` in Phase 1. Every old
symbol or call chain is either:

- `move`: representation-independent behavior is relocated once to its final
  sole owner, with all accepted callers migrated and the old definition
  deleted in the same change;
- `rewrite`: required behavior moves to its new sole owner; or
- `delete`: stale, duplicate, or forbidden behavior disappears with its
  callers, tests, configuration, and documentation.

Record the authority, new owner, phase, validation evidence, and final deletion.
There is no “reuse in place,” “adapter,” or “keep for safety” disposition.
A move may not carry an old representation, compatibility DTO, or parallel
implementation across the firewall.

Every capability in `features_to_preserve.md`, and every reachable capability
discovered later, defaults to preserved behavior. Merging and rewriting are
expected; removal or semantic change requires explicit user approval.

## 3. Mandatory phase protocol

The user may say:

> Implement Phase X from
> `docs/todo/refactor/implementation_plan.md`.

The phase owner reads Sections 1–3 and 5, the recorded prerequisite-closure
evidence, the named phase, that phase's sources, and every completed
`docs/queued/` contract on which the phase depends. It need not load unrelated
later phase descriptions.

For every phase:

1. **Start clean.** Record the clean starting SHA. Do not mix unrelated user
   changes into the phase.
2. **Implement and commit.** The phase owner implements the scope itself, adds
   focused tests and failure injection, reviews the diff for ownership and
   duplication, and commits the candidate.
3. **Validate the commit on CUDA.** The owner spawns an independent validation
   agent and gives it the exact candidate SHA. That agent runs the phase and
   cumulative gates, adversarial cases, determinism checks, and matched-work
   benchmarks. It does not modify production code.
4. **Audit the same commit.** Only after validation passes, the owner spawns a
   different, read-only audit agent for that SHA. It traces behavior to
   authority, checks ownership and memory, searches for parallel paths and
   stale surfaces, and judges whether each abstraction earns its existence.
5. **Fix and repeat.** Findings produce new fix commits, followed by fresh
   validation and audit of the new SHA.
6. **Close and commit.** After both agents pass, the owner closes the ledger
   rows, writes the phase's `docs/queued/` document from verified behavior, and
   creates a phase-closure commit whose parent is the exact validated/audited
   SHA. That commit may contain only reports, ledger evidence, and
   documentation, and receives static/link checks. Any executable,
   configuration, test, or authority/design-contract change restarts
   validation and audit. The next phase starts only from that clean committed
   state.

The CUDA agent uses `/home/pablob/.venvs/poker`, checks `nvidia-smi`, selects
only idle GPUs explicitly, uses the venv CUDA toolkit, and confirms the GPUs
return to their previous idle state. Runtime artifacts and configs follow the
repository naming convention under `runs/experiments/` and
`configs/experiments/`. A short phase report under `docs/audit/` records exact
commands, revisions, environment, seeds, results, memory, and timing.

Use matched logical seeds where approved laws agree. For an intentional law
change, validate the new invariant and distribution rather than demanding
pathwise equality with the old engine.

## 4. Delivery map

| Training regime | Rollout | Estimator | Current behavioral oracle | Delivered |
|---|---|---|---|---|
| self-play | sequential | sampled | production | Phase 5 |
| best response | sequential | sampled | net-new | Phase 6 |
| self-play | sequential | RB | production | Phase 6 |
| best response | sequential | RB | shared-response probe | Phase 6 |
| self-play | branched.fixed | sampled | net-new | Phase 7 |
| best response | branched.fixed | sampled | net-new | Phase 7 |
| self-play | branched.KKT | sampled | net-new | Phase 7 |
| best response | branched.KKT | sampled | net-new | Phase 7 |
| self-play | branched.fixed | RB | production | Phase 8 |
| best response | branched.fixed | RB | net-new | Phase 8 |
| self-play | branched.KKT | RB | production | Phase 8 |
| best response | branched.KKT | RB | net-new | Phase 8 |

All twelve schemas exist in Phase 1. Phase 3 exercises all target recurrences
with fixtures. “Delivered” means full poker integration through an optimizer
step. Production rows require behavior and matched-work regression evidence;
net-new rows require mathematical-reference, distributional, adversarial, and
end-to-end evidence and must never be presented as old-engine parity.

## 5. Prerequisite approval

Before Phase 1, close and record all of the following gates:

- approve the complete refactor folder and its residual preservation baseline;
- reconcile the refactor with `core_math.tex`, `MHA_model.md`, and
  `folder_impact.md`, including root batching, sequential $\beta=1$ geometry,
  branched-RB inherited-profile plus $N-1$ sampling, preflop H-to-K target
  conversion, model execution, folded inputs, and any estimator changes;
- update the repository distinguished-root convention before allowing more
  than one branched rollout tree per GPU;
- resolve the current evaluator ownership requirement before moving posterior
  sampling into this repository, and either use evaluator contract v2 exactly
  or approve and implement a versioned replacement contract before its
  consumer phase;
- preserve whole-attempt transactional overflow recovery: retry at a disjoint
  RNG address after reducing sequential `root_count`, fixed beta, or KKT
  controller scale; completed replicas keep their stores, caps never change,
  and recovery at the minimum control fails the run;
- define the physical-schema compatibility relation that permits fixed and KKT
  replicas in one execution group while retaining each lane's semantic mode
  identity;
- define the source and public configuration of every fixed capacity, including
  how a hardware profile is validated for `(P,S,R)`, mode, estimator,
  `root_count`, replica count, and external workspaces;
- approve the twelve-mode public configuration and the one learner operation
  for ordinary, distributed, and replica execution;
- inventory current checkpoints and run artifacts that must remain usable, and
  choose explicit offline converters. Runtime compatibility loading is not a
  cleanup strategy.

Use
[`nlhe_hu_32bb_family_timing_attribution_20260724.md`](../../audit/nlhe_hu_32bb_family_timing_attribution_20260724.md)
as the initial branching-performance baseline. Reproduce or explicitly
account for drift from its fixed-$\beta=3$ HU 32 bb setup, including seeds
104830, 209559, and 314288, family pack maximum 84, and 1,024 completion
samples per leaf. Its reference observations include 8.575 ms per retained
branched decision versus 3.967 ms sequentially, 51.35 GB of generic routing
metadata, about 16.1 seconds in child reservation plus commit, 2,494 host
synchronizations, and 14.67 seconds of branched target propagation versus
0.040 seconds sequentially.

Before performance claims, freeze explicit pass thresholds for matched logical
work on the required CUDA hardware. Gates must expose rollout time per retained
decision, routing bytes and rows, host synchronization count, child
reservation/commit time, target-propagation time, peak memory, and sequential
regression. Record configs, seeds, environment, logical work, and tolerances;
“accepted,” “substantial,” and “no regression” are not gates until those
thresholds exist.

Also classify the saved H100 patch hunk by hunk as evidence, rewrite
requirement, pre-existing change, or delete. Never apply it wholesale.

## 6. Implementation phases

### Phase 1 — contracts and inventory

**Objective:** establish the closed vocabulary and prevent legacy coupling.

**Read:** proposal `README.md`; `features_to_preserve.md`;
`storage_redesign.md` Sections 1–2; the three external authorities; current
production roots, reachability/compactification audits, and git history.

**Implement:**

- create `pbg/queued/` and its import firewall;
- generate immutable `ModeSchema` rows for all twelve combinations;
- define partitions, conversions, applicable fields, replica-axis rules, and
  minimal opaque game hooks;
- create the migration ledger and accepted production-root inventory;
- inventory every active scientific, execution, capacity, and persistence
  field, plus every checkpoint/run artifact selected for offline conversion;
- verify every reachable production capability and active configuration field
  is represented in `features_to_preserve.md` or explicitly approved for
  change/deletion;
- record the reconciled external-authority clauses and evaluator contract
  version that authorize each later phase;
- freeze matched-work CUDA baselines and numerical, memory, and throughput
  tolerances for every currently delivered mode and accepted consumer;
- add static gates against forbidden imports, legacy mode names in queued core,
  and production roots importing both engines.

The schema is data, not twelve classes.

**Gate:**

- exactly twelve valid schemas, with impossible combinations rejected;
- inapplicable fields are absent rather than nullable;
- representative schemas validate on CUDA;
- queued core cannot reach old core or poker;
- every old production definition/call chain and accepted production root is
  classified, including dynamically reached and test-only retention evidence;
- every preservation row has a planned new owner and regression gate;
- every proposed capacity has a configuration/calibration source and every
  accepted old artifact has a converter-or-delete disposition;
- no unresolved authority or external-contract conflict remains.

**Document:** `docs/queued/mode_contracts.md`.

### Phase 2 — storage, transactions, and memory

**Objective:** implement the only allocation and lifetime mechanisms.

**Read:** storage and memory portions of `storage_redesign.md`;
`features_to_preserve.md`; old storage, execution, memory-control code, tests,
audits, and relevant history as evidence.

**Implement only:**

```text
SoASchema       immutable fields, dtype, shape, alignment, lifetime
MonotoneSlab    fixed-capacity reserve -> fill -> publish
ExactBlock      immutable realized extent
```

Generate all owners from `ModeSchema`. Add replica-local tails and failures,
exact byte accounting, atomic multi-destination transactions, explicit release,
phase admission, and minimum-address `DeviceFailure`. Implement the immutable
capacity manifest and validate it against game geometry, mode, root/replica
dimensions, hardware limits, and bounded evaluator/library workspaces before
allocation.

Do not add paging, recycling, wrapping queues, generic columns, convenience
stores, or post-hoc conversion from legacy storage.

**Gate:**

- all twelve schemas allocate exactly their declared CUDA owners;
- analytical, allocated, reserved, high-water, and observed bytes reconcile;
- injected failures publish either every destination or none;
- addresses never move, wrap, or repeat;
- lane failures do not corrupt completed lanes;
- every owner releases exactly after its last consumer;
- overflow reports the minimum logical address without partial publication;
- capacity manifests reject uncovered geometry and reconcile their configured,
  analytical, allocated, and observed bytes.

**Document:** `docs/queued/storage_and_lifetimes.md`.

### Phase 3 — topology and layered targets

**Objective:** prove the shared structural and mathematical path before poker
integration.

**Read:** `layered_target_creation.md`; branching/target mathematics in
`core_math.tex`; `features_to_preserve.md`; relevant `storage_redesign.md`
sections; old topology/target code, tests, audits, and history as evidence.

**Implement from tensor fixtures:**

- compact CPU `Topology`, topology packets, and `LayerPlan`;
- exact movement plan and two fixed work arenas;
- one CUDA layer executor generated from `ModeSchema`;
- sampled return and RB mass/moment carriers;
- branched trace/cost/delta only where applicable;
- deterministic FP32 post-to-pre H-to-K conversion;
- headless frozen structural rows and atomic target publication.

Benchmark carry rows versus last-consumer slots and direct sibling sum versus
total-minus-child. Select one of each and delete the losers in this phase.

**Gate:**

- agreement with direct mathematical references for all twelve modes;
- exact sequential specialization and ordinary branched-root behavior;
- protected root prefix, unary compression, duplicate children, and unbalanced
  trees;
- deterministic logical-child-order and H-to-K reductions;
- no Python node/action loop or host synchronization between layers;
- exact memory admission and last-consumer release;
- one executor, not mode-specific target builders.

**Document:** `docs/queued/topology_and_targets.md`.

### Phase 4 — queues and `model -> sample -> commit`

**Objective:** implement the generic rollout transaction using synthetic hooks.

**Read:** queue, sampling, and failure portions of `storage_redesign.md`;
`features_to_preserve.md`; logical-RNG mathematics; old rollout/branching code,
tests, audits, and history as evidence.

**Implement:**

- semantic family FIFOs over monotone `FamilyWork`;
- stable full-first/preorder/load-age prefix scheduling;
- compatible replica cohorts without padding;
- schedule-independent logical RNG;
- one `model -> sample -> commit` state machine;
- device counts, scans, reservations, and atomic commit;
- asynchronous topology packets and endpoint patches;
- bounded chance closure and completion FIFO.

**Gate:**

- stable FIFO order and no undeclared queue family;
- no skipped or reordered row;
- subdivision, streams, and lane grouping preserve labels and RNG;
- inactive lanes draw nothing;
- no per-pack host scalar read for counts or reservations;
- `sample` writes no retained destination;
- `commit` publishes all destinations and topology endpoints together or none;
- bounded chance closure has no host synchronization per generation;
- one rollout implementation serves sequential and branched fixtures.

**Document:** `docs/queued/rollout_transactions.md`.

### Phase 5 — self-play sequential sampled poker

**Objective:** complete the smallest end-to-end poker path without creating a
normal-only shortcut.

**Read:** completed Phase 1–4 docs and contracts; sequential sampled portions
of `storage_redesign.md`; `poker_policy_cuda_operator.md`; evaluator showdown
API; old poker and normal-training code as evidence.

**Implement:**

- poker `pre` and `post_A` hooks and direct `model_input`;
- root profiles, categorical labels, chance, and transitions;
- the canonical policy CUDA operator for rollout and learner;
- sampled `Leaf` and evaluator showdown binding;
- sequential topology/targets through the Phase 3 executor;
- aligned five-array `randperm` batches;
- one partition forward/loss/immediate-backward operation;
- transactional overflow recovery that discards the failed attempt, reduces
  `root_count`, and retries at a disjoint RNG address without changing caps;
- a durable high-severity recovery warning containing the iteration, replica,
  failed phase, failed and replacement addresses, old and new `root_count`,
  capacity, observed demand, and a checkpointed/run-summary recovered-overflow
  flag;
- dense FP32 gradients, clipping, selected Adam/AdamW, AMP/scaler behavior,
  checkpoint/resume, and iteration composition;
- every residual preservation-ledger capability assigned to this phase;
- the offline conversion framework and approved sequential-sampled artifacts,
  with source hashes and no runtime compatibility loader. Later phases extend
  the same converter only when their mode state exists.

Do not add a normal target builder, normal store, policy adapter, or
mode-specific learner.

**Gate:**

- complete CUDA iteration and resume;
- matched-seed agreement where the approved law is unchanged;
- policy forward/gradient agreement with a high-precision reference;
- one policy operator across sampling, retained heads, and learner terms;
- exact evaluator rows and sequential alignment;
- dense-gradient Adam and AdamW oracle agreement;
- sequential overflow publishes none of the failed attempt; retries are
  rank-common, use a disjoint address and lower `root_count`, remain permanently
  visible in the run record, and fail only when the minimum control overflows;
- all Phase 5 preservation-ledger gates pass;
- every Phase 5 checkpoint converts once and resumes with exact declared state;
- no rollout-hot owner survives into optimization;
- reconciled memory and no accepted sequential throughput regression.

**Document:** `docs/queued/sequential_sampled_training.md`.

### Phase 6 — all sequential modes

**Objective:** complete the four sequential modes by extending schemas and
hooks, not the engine.

**Read:** RB and best-response mathematics; `features_to_preserve.md`;
`poker_folded_statistics.md`; preflop/dtype sections of the replica audit;
evaluator completion APIs; old RB, probe, folded, and completion code, tests,
audits, and history as evidence.

**Implement:**

- sequential RB completion, mass/moment work, and K-wide preflop targets;
- folded ranges, compact statistics, refresh, and release;
- best-response trained/frozen roles and selected-player completion;
- trained-input compaction and headless frozen rows;
- shared-response probe as an ordinary best-response consumer;
- every residual preservation-ledger capability assigned to this phase;
- extend the sole offline converter to approved sequential-RB and probe
  artifacts;
- sampled best response and RB loss terms through the same learner operation;
- explicit FP32 posterior and loss-control arithmetic.

**Gate:**

- all four sequential modes complete CUDA optimizer steps;
- exact evaluator and selected-owner semantics;
- correct folded refresh and lifetime;
- K-wide preflop heads, targets, posterior, policy reductions, and loss;
- deterministic H-to-K conversion;
- frozen rows own no learner work;
- probe storage is the ordinary best-response sequential-RB schema;
- all Phase 6 preservation-ledger gates pass;
- sampled/RB and self-play/BR share storage, targets, learner, and optimizer.

**Document:** `docs/queued/sequential_modes.md`.

### Phase 7 — branching and sampled modes

**Objective:** add fixed and KKT branching once and deliver the four sampled
branched modes.

**Read:** branching, refill, safeguard, trace, cost, and sampling mathematics;
`features_to_preserve.md`; old controller and branching code, tests, audits,
and history as evidence.

**Implement:**

- ordinary randomized rounding at every node, including the root;
- one fixed/KKT controller interface;
- nominal/executed degrees, safeguard correction, protected root prefix, and
  refill;
- one joint sampled private profile per tree;
- duplicate-child identity;
- ragged action spans, trace/cost targets, decision weights, alias sampling,
  and exact gathers;
- bounded whole-attempt backoff with disjoint RNG and rank agreement.

**Gate:**

- four sampled branched modes complete CUDA optimizer steps;
- fixed and KKT use the same engine;
- root, refill, target-prefix, and descendant-mass behavior is correct;
- sampled siblings share the contracted tree profile;
- duplicate labels keep distinct identities;
- no padded, inactive, or maximum-action representation;
- scheduling invariance;
- retry releases the failed attempt and changes only declared controls/address;
- all other failures terminate; sequential throughput remains accepted.

**Document:** `docs/queued/branched_execution.md`.

### Phase 8 — branched Rao--Blackwell poker

**Objective:** deliver the remaining four modes without adding a second engine.

**Read:** `poker_branched_rb_sample.md`, `poker_folded_statistics.md`,
`poker_policy_cuda_operator.md`, RB branching mathematics, evaluator v2, and
completed Phase 3/4/6/7 docs.

**Implement:**

- inherited child zero plus exactly $N-1$ new posterior profiles;
- collision-conditioned CUDA rejection sampling;
- action-conditioned chance proposals and profile propagation;
- exact live/folded conditioning and chance-card generation;
- branched RB completion, work conversion, targets, decision weights, and
  ragged action rows;
- extend the sole offline converter to approved branched artifacts;
- rejection-tail failure and observability.

Prototype admissible action-conditioned samplers only within this phase. Select
one using complete `sample -> commit` results and delete every loser and
selector before audit.

**Gate:**

- final four modes complete CUDA optimizer steps;
- exact conditioned law for all valid player/rank boundaries;
- correct child-zero reuse and N-1 sampling;
- correct label/profile joint law and folded conditioning;
- fixed searches never select zero-mass hands;
- scheduling invariance and minimum-address exhaustion failure;
- no rollout action-axis expansion or second profile owner;
- one reproducibly selected sampler remains;
- the matched fixed-$\beta=3$ HU 32 bb CUDA benchmark meets the frozen
  thresholds derived from the July 24 timing attribution for rollout time per
  retained decision, routing traffic, host synchronization, child
  reservation/commit, target propagation, peak memory, and sequential
  regression;
- all twelve modes pass the common end-to-end matrix.

**Document:** `docs/queued/branched_rb_sampling.md`.

### Phase 9 — replica inference and distributed learner identity

**Objective:** make ownership and scheduling the only differences among
ordinary, replica, and distributed learning.

**Read:**
[`replica_learner_parity_and_rb_determinism.md`](../../audit/replica_learner_parity_and_rb_determinism.md);
`features_to_preserve.md`; learner/memory storage contract; H100 handoff
evidence; old learner, optimizer, `replica_*.py`, tests, audits, and history.

**Implement:**

- zero-copy parameter and dense-gradient lane binding;
- the same per-lane forward/loss/scale/immediate-backward operation;
- active lane scheduling without padding;
- one functional `vmap` operation for replica-batched rollout and
  reference-model inference over independent lane parameters and state, with
  autograd disabled and no shared-weight substitution;
- compatible grouped non-autograd work;
- one ordered dense replica-axis gradient buffer;
- SUM synchronization, clipping, selected Adam/AdamW, counters, and
  checkpoints;
- lane-local failure and iteration-synchronous versions;
- extend the sole offline converter to approved distributed and replica-lane
  artifacts;
- explicit control, posterior, gradient, and optimizer dtypes.

Do not add functional/`vmap` learner backward, stacked replay, replica
losses/adapters, summed lane graphs, or replica optimizer semantics.

**Gate:**

- ordinary versus `n_replica=1` one-step and complete-run parity;
- prescribed H100 deterministic preflop gates;
- lane counts `{1,2,4,8}` prove complete state isolation;
- inactive lanes do no work and unused parameters receive explicit zero
  gradients;
- replica-`vmap` inference matches independent lane inference for active
  subsets, adds no padding rows or inactive-lane RNG draws, and preserves
  lane-owned outputs and labels;
- dense Adam and AdamW match independent oracles;
- distributed SUM and asymmetric failure agreement pass;
- checkpoints preserve lanes exactly;
- rollout outputs and labels remain lane-owned when weights, loss controls, or
  an approved schema-compatible fixed/KKT controller selection differ;
- all twelve modes retain one learner operation.

**Document:** `docs/queued/learner_and_optimization.md`.

### Phase 10 — consumer migration

**Objective:** move every accepted product to one public replacement boundary.

**Read:** migration ledger and current runtime, training/replica CLI,
best-response evaluation, probes, inference, variance runners, benchmarks,
checkpoints, exports, and operational tools.

**Implement:**

- one public configuration/composition for the three axes;
- ordinary, distributed, and replica training;
- best-response evaluation and probes;
- supported CUDA/MPS inference with clear unsupported-operation failures;
- variance and benchmark consumers;
- checkpoint/resume/export/import;
- close every remaining approved artifact-conversion row and remove all
  runtime compatibility loading;
- strict manifests and observability;
- every residual preservation-ledger capability assigned to consumer
  migration.

Migrate tests to production APIs. Move useful test-only or developer-only code
to `tests/` or `tools/`.

**Gate:**

- every production root reaches the replacement and none reaches old core;
- every accepted consumer has an accelerator gate;
- unsupported MPS work fails before CPU staging;
- artifacts use only the new schema;
- benchmarks use production code;
- no legacy configuration, alias, ignored option, or permissive fallback;
- every residual capability in `features_to_preserve.md` passes through its
  accepted public consumer;
- reachability and similarity audits find no migrated duplicate;
- cumulative twelve-mode, replica, distributed, memory, failure, and
  performance gates pass.

**Document:** `docs/queued/runtime_and_consumers.md`.

### Phase 11 — cutover and deletion

**Objective:** replace the old core with the replacement and delete the old
engine completely.

**Read:** complete migration ledger, queued docs, and final reachability,
configuration, similarity, and consumer inventories.

**Implement atomically:**

1. replace the old `pbg/core/` directory with `pbg/queued/` in one cutover
   commit, preserving version-control history where Git can detect moves;
2. update imports, packaging, and public roots in that same commit;
3. verify that no file from the old core survives except a deliberately
   rewritten final owner already present in the replacement;
4. delete comparison and migration scaffolding;
5. delete superseded tests, configuration, CLI, benchmarks, tools, and docs;
6. delete every prototype, adapter, alias, fallback, and stale checkpoint path;
7. merge replacement modules whose final responsibilities no longer justify a
   boundary;
8. run static, import, reachability, similarity, configuration, and
   documentation audits.

Recovery is version control, not a retained rollback path.

**Gate:**

- every ledger row closes with a deletion;
- no old file or copied implementation body remains;
- every production object is reachable from an accepted root;
- no production object exists only for tests;
- exactly one authority remains for every pipeline operation;
- all modes and consumers pass cumulative CUDA gates;
- distributed and replica gates pass;
- memory reconciles, performance remains accepted, and GPUs return idle;
- queued docs name only final modules and behavior.

**Document:** `docs/queued/core_pipeline.md`. Update earlier queued documents to
final `pbg/core/` paths.

## 7. Final state

The rewrite is complete only when the repository—not merely this plan—has:

```text
one immutable twelve-mode contract
+ one owner per lifetime
+ one implementation per operation
+ one production path per accepted consumer
+ no legacy or compatibility surface
```

`docs/queued/` then describes the verified pipeline and is ready, in a separate
documentation-tree change, to become `docs/pbg/core/`.
