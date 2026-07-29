# PBG design-fidelity remediation order

Date: 2026-07-29

Evidence authority:
[PBG implementation versus queued-design audit](pbg_design_fidelity_audit_20260729.md).

This document recommends the implementation order for correcting every issue
identified by that audit. It does not replace the audit's evidence,
measurements, or acceptance criteria. Finding identifiers `F1` through `F14`
refer to its Section 3. Configuration findings refer to its Section 9, and
legacy/dead-surface findings refer to its Section 6.

The objective is not a gradual accumulation of another framework beside the
current one. The final repository must have:

- one production iteration engine;
- one topology publication and assembly protocol;
- one learner objective and optimization skeleton;
- one schema-generated storage system with one owner for each lifetime;
- poker expressed as game hooks rather than generic orchestration;
- one small user configuration compiled into one immutable resolved manifest;
- no selectable compatibility pipeline, profiling lifecycle, stale config
  schema, unused generated field, or unexplained dead production path; and
- measured completion of the remaining acceleration work.

## 1. Why this order

The work has hard dependencies:

```text
freeze evidence and stop config drift
        |
        v
settle schema/ownership contracts
        |
        +----------------------+
        v                      v
repair retained storage   define core/game hooks
        |                      |
        +----------+-----------+
                   v
       one topology packet/assembler
                   |
                   v
        one core iteration engine
                   |
                   v
       one learner/optimizer skeleton
                   |
                   v
       atomic production cutover/deletion
                   |
          +--------+---------+
          v                  v
 direct views/exact arenas  overlap/device scheduling
          +--------+---------+
                   v
          measured poker kernels
                   |
                   v
   final capacity compiler and YAML migration
                   |
                   v
       naming, dead-code, and closure audit
```

The architectural unification must precede the expensive performance work.
Otherwise every optimization must be implemented and validated across
sequential, ordinary-branched, and replica-branched engines and will make
those engines harder to delete.

The full capacity compiler comes late because capacities, workspaces, and
derivations depend on the final owner graph and common lifecycle. A small
derivation verifier comes first to stop the current configuration drift.

Exact arena allocation precedes completion overlap because the current
global synchronization and `empty_cache()` barrier exist partly to cope with
over-allocation and uncertain workspace. Overlap needs exact admitted memory
and stable lifetimes.

Package renaming and file splitting come last. Moving large duplicated code
before deleting it creates churn without improving architecture.

## 2. Rules for the remediation

### 2.1 Do not ship a fourth production engine

The new common engine may exist behind test-only construction while it is
being built. It must not be exposed as a runtime selector, compatibility
backend, feature flag, or alternate production workflow.

The production cutover must switch `TrainingSession` to the common engine and
delete the displaced sequential/branched/replica engines in the same merge
series. A rollback is a Git rollback, not a permanent engine selector.

### 2.2 Preserve the substrate that already works

Do not rewrite these merely for uniformity:

- generated twelve-mode schemas;
- `MonotoneSlab` and multi-destination publication semantics;
- `RolloutTransaction`;
- logical RNG addressing and first-success semantics;
- compact `Topology`, `TopologyBindings`, and `LayerPlan`;
- the layered-target CUDA executor;
- evaluator-v2 direct bindings;
- natural-domain RB targets;
- accelerator-only failure policy; and
- the dense optimizer implementation.

Refactor their composition and ownership, not their proven semantics.

### 2.3 Every stage has a deletion gate

Adding a common abstraction is not completion. Each stage names the old
authority that must disappear before the stage closes. Temporary adapters are
test scaffolding and must not survive the production cutover.

### 2.4 Behavioral and performance claims require CUDA evidence

Every behavior-changing stage must run on an explicitly selected idle CUDA
device using the repository venv and CUDA toolkit. Multi-GPU distributed gates
must run only when the required devices are idle. Always verify the selected
GPU returns to its clean baseline.

CPU-only tests remain appropriate for parsing, pure topology invariants,
hashing, report generation, and method/field analysis. They are not evidence
for production rollout or training behavior.

### 2.5 Do not silently rewrite mathematical authority

The intended algorithms remain those in `docs/theory/`. Any proposed
mathematical change requires explicit approval before changing those files.
This remediation should normally need only implementation and queued-design
contract updates.

## 3. Stage 0 — establish the executable baseline and issue ledger

### Objective

Make drift visible before moving code, and turn every audit conclusion into a
mechanical test or an explicit design decision.

### Work

1. Create a single remediation ledger with one row for every item in the
   traceability table in Section 18 of this document. Each row owns:

   - affected production symbols;
   - replacement symbol or deletion;
   - required tests;
   - performance/memory gate when applicable;
   - design-document decision;
   - merge stage; and
   - closure evidence.

2. Freeze golden semantic traces for all twelve schema modes:

   - self-play and best response;
   - sequential, fixed branching, and KKT branching;
   - sampled and RB;
   - one lane and multiple lanes; and
   - distributed execution where enough GPUs are available.

3. Trace generic lifecycle events, not poker implementation details:

   - phase transitions;
   - root and child publications;
   - hook calls;
   - topology packets and final topology;
   - leaf/completion calls;
   - learner decision/action descriptors;
   - reference refreshes;
   - optimizer transitions;
   - recovery decisions;
   - owner releases; and
   - logical RNG addresses.

4. Freeze numerical parity fixtures for:

   - selected actions and chance labels;
   - sampled and RB leaf payloads;
   - target values, mass, moment, trace, and cost;
   - PPO ratios and clipping;
   - value loss, entropy, and reference KL;
   - gradients and optimizer versions; and
   - checkpoint/resume continuation.

5. Capture current matched-work CUDA baselines:

   - player nodes and leaves per second;
   - model, transition, posterior, completion, target, and learner time;
   - host synchronizations;
   - phase peak allocated and reserved bytes;
   - owner high-water rows;
   - completion overlap, currently zero; and
   - ordinary/replica/distributed parity.

6. Add structural static gates that will eventually require:

   - one lifecycle owner;
   - one topology assembler;
   - one optimizer loop;
   - no phase advancement from poker;
   - no generated field without a writer and reader; and
   - no production benchmark-only early return.

At this stage, gates that describe the desired architecture may be recorded
as expected failures. They become mandatory at the named later stages.

### Exit gate

- Every audit issue has a ledger row.
- All twelve modes have a reproducible semantic fixture.
- Existing behavior, memory, and performance have a dated CUDA baseline.
- No refactor has begun without a parity oracle.

### Resolves

This stage does not close a production issue. It prevents F1–F14 from being
“fixed” by assertion and directly addresses the evidence weakness in F14.

## 4. Stage 1 — stop configuration drift immediately

### Objective

Make checked-in runs truthful before using them as refactor baselines, without
prematurely freezing today's flawed manifest design into a new compiler.

### Work

1. Extract one pure derivation library for currently undisputed geometry:

   - compact deck size $D=4R$;
   - exact hands $H={D\choose2}$;
   - preflop classes $K=R^2$;
   - player/history bounds;
   - natural model-input widths;
   - action width;
   - model token and token-pair bounds;
   - evaluator private width;
   - player count; and
   - precision-to-storage dtype.

2. Make config validation recompute these values before CUDA storage
   construction. During this transitional stage, old YAML literals may still
   exist, but any disagreement must fail at parse/compile time rather than in
   an iteration.

3. Regenerate or delete the 15 checked-in YAML files that still declare the
   old 3,100-wide preflop representation instead of 786 for
   $P=2,S=32,R=13$.

4. Mark generated manifests with a schema and derivation version. Add a
   deterministic check that checked-in generated configs equal current
   generator output.

5. Give the branching profiler fixture a distinct file kind and location so
   it cannot be mistaken for a public run YAML.

6. Remove the existing unused import and ensure config-generation/profile
   tools pass the same lint gate as production.

7. Add mutation tests for every duplicate that remains temporarily:

   - scientific versus manifest root count;
   - scientific versus manifest beta bound;
   - precision versus model dtype;
   - execution versus hardware query chunk;
   - game versus layout player count; and
   - derived input/token dimensions.

### Deletion gate

- No stale checked-in config remains.
- No parser accepts a known derived mismatch.
- No profiler composite is named or parsed as an ordinary run config.

### Exit gate

All checked-in run specs either compile to current geometry or are explicitly
historical non-runnable artifacts outside the active config tree.

### Resolves

- Section 9.2's 15 stale configurations.
- The most dangerous late-validation cases from Section 9.7.
- The profiler-file ambiguity from Section 9.9.
- The tool-level unused import observed during final audit validation.

This stage deliberately does not yet remove the full editable manifest.

## 5. Stage 2 — settle the storage and extension contracts

### Objective

Resolve every known design/implementation disagreement before encoding new
owner types.

### Work

1. Decide and document the `FoldedRanges.log_mass` representation:

   - prove why normalized `r_pi` alone cannot restore the required raw folded
     reach scale; then retain `log_mass` and amend
     `storage_redesign.md`; or
   - derive the scale from another retained quantity and remove `log_mass`.

   The current undocumented divergence is not allowed to persist.

2. Decide the physical shape of `FamilyWork.folded_range_row`.
   A fixed $P$-wide vector with sentinels may be simpler than a variable
   $P-A$ representation. If retained, document it as a deliberate physical
   simplification and prove that no numerical consumer interprets sentinel
   entries as folded seats.

3. Amend the Leaf contract to acknowledge mode-applicable extensions:

   - `trained_actor` as the best-response selected player; and
   - `response_cutoff` as poker response-cutoff metadata.

   Keep generic field names in core. Do not rename opaque generic fields to
   poker terms merely to imitate evaluator spelling.

4. Specify one lifetime contract for rollout-hot versus learner-retained
   payload. The recommended form is:

   ```text
   FamilyHot[input_type]       rollout -> align/seal
   FamilyRetained[input_type]  model commit -> final backward
   LearnerIndex[input_type]    metadata only
   ```

5. Specify one decision/action relation:

   $$
   A=\sum_{s\in\mathcal U}N_{\mathrm{tgt}}(s),
   $$

   with sequential $N_{\mathrm{tgt}}=1$ and therefore one shared row identity
   for sigma, action, and target.

6. Specify the topology/leaf forest relation:

   $$
   E=(T-Q)+E_{\mathrm{leaf}},
   $$

   and $E=T+L-Q$ whenever each leaf row is a unique endpoint. State clearly
   whether a capacity counts structural nodes only or structural plus leaf
   nodes.

7. Specify BR trained-row derivation. Preserve the already balanced
   `logical_root % P` role assignment, then prove a safe capacity from
   deterministic selected-seat counts and per-seat poker occurrence bounds.
   Do not assume `ceil(total_player_cap/P)` until the cap being divided is
   proven $P$-symmetric.

8. Decide that HU omits folded-summary storage and, once verified against all
   leaf/fold paths, omits the unused folded-range owner rather than requiring
   a one-row phantom slab.

### Exit gate

- Every deliberate divergence from `storage_redesign.md` is either removed or
  explicitly incorporated into the queued-design contract.
- There is no unresolved meaning for owner, row, edge, leaf, trained, or
  action counts.
- The forthcoming storage schema can be generated without a compatibility
  interpretation.

### Resolves

- F11's `folded_range_row` shape divergence.
- F12's undocumented folded raw-mass representation.
- F13's undocumented Leaf extensions.
- Configuration edge/action/trained-cap ambiguity.

## 6. Stage 3 — repair storage ownership and generated fields

### Objective

Create one physical owner per lifetime and remove concrete unused storage
before building the common engine.

### Work

1. Generate separate `FamilyHot` and `FamilyRetained` owners from the mode
   schema. Publish them through one transaction with a shared logical row
   identity.

2. Reduce `LearnerIndex` to metadata:

   - action begin/count;
   - target degree where applicable;
   - sampling/log-weight metadata; and
   - retained-row addressing only when it cannot be implicit.

   It must not own `model_input`, sampled private state, or owner payload.

3. Make self-play learner inputs direct retained views. Best response may
   compact trained rows, but the compact payload must have an explicit
   lifetime-bearing owner and must copy only when the selected rows are
   genuinely noncontiguous.

4. Delete:

   - `FamilyWork.reservation_state`;
   - `FamilyWork.publication_state`; and
   - `QueueMetadata.controller_state`,

   unless a production consumer is implemented first. Existing slab state and
   controller/`RunState` state remain the sole authorities.

5. Generate only mode-applicable fields. In particular:

   - no trace/cost columns in sequential modes;
   - no sampled-private payload in RB;
   - no evaluator private-width storage in sampled modes;
   - no reference cache allocation when disabled;
   - no folded owner for HU after the Stage 2 proof; and
   - no mandatory one-row placeholder for an absent owner.

6. Update analytical byte accounting and lifetime overlap to use the new
   owners. Prove that owner release cannot invalidate a retained learner view.

7. Add field-level writer/reader tests generated from `ModeSchema`.

### Deletion gate

- Old payload-bearing `LearnerIndex` code is deleted.
- Old `SealedSequentialInputs.seal()` and
  `SealedBranchedInputs.seal()` payload-copy behavior is deleted or reduced to
  metadata/necessary BR compaction.
- The three unused generated fields no longer exist.

### Exit gate

- `LearnerIndex` allocated bytes correspond only to metadata.
- Self-play seal performs no retained-payload device copy.
- Every generated field has a production writer and reader in each applicable
  mode.
- CUDA memory measurements show the expected removal of the duplicate
  `FamilyWork`/`LearnerIndex` payload.

### Resolves

- F5 completely.
- F11's concrete dead fields.
- Part of the storage-fidelity and simplicity findings.
- Acceptance criteria 9, 10, and 14 from the audit.

## 7. Stage 4 — define the final core/game interface

### Objective

Make the responsibility boundary enforceable before moving lifecycle code.

### Work

1. Define one small core-owned hook protocol. It should accept opaque generic
   batches and return descriptors/tensors for operations equivalent to:

   - root creation;
   - model-input encoding;
   - model forward;
   - action/chance sampling;
   - transition;
   - topology payload;
   - leaf payload;
   - evaluator completion binding;
   - target algebra; and
   - learner policy composition.

2. Keep poker responsible for:

   - NLHE geometry and partitions;
   - cards, hands, streets, actions, and legality;
   - joint and posterior profile sampling;
   - action/chance sampling;
   - public transition;
   - folded-range/statistic construction;
   - model architecture/input encoding;
   - evaluator-v2 binding; and
   - poker-specific observability values.

3. Move or redesign generic types currently declared in poker:

   - phase/lifecycle results;
   - scheduler descriptors;
   - topology packet/staging policy;
   - learner scheduling inputs;
   - recovery outcomes;
   - replica/distributed preflight descriptors; and
   - generic benchmark metrics.

4. Add static boundary rules:

   - `pbg/core` cannot import poker;
   - poker cannot call `enter_phase`, `release_after`, `close`, optimizer
     sequencing, or distributed preflight;
   - poker hooks cannot own a queue loop; and
   - core cannot inspect cards, hand classes, streets, folds, or poker action
     labels.

5. Keep result formatting as a small poker adapter over generic metrics.

### Exit gate

- A mock non-poker game can satisfy the hook protocol and drive core unit
  tests.
- Poker hooks are callable without constructing an iteration engine.
- Static boundary checks fail on generic orchestration added under poker.

### Resolves

- F4's responsibility failure.
- The core/poker portion of the simplicity finding.
- Acceptance criterion 8.

## 8. Stage 5 — unify topology publication

### Objective

Make sequential the degree-one representation of the same topology protocol,
not a separate trace-to-topology implementation.

### Work

1. Define one core-owned topology packet carrying:

   - parent/path identity;
   - nominal, executed, and target degree;
   - child labels;
   - topology or leaf endpoints;
   - path length;
   - trained owner;
   - family row;
   - sigma row;
   - action begin/count; and
   - input/output partition bindings.

2. Use one reservation, publication, acceptance, endpoint-patch, and sealing
   protocol for every rollout mode and lane count.

3. Encode sequential nodes as ordinary packets with:

   ```text
   nominal_degree = executed_degree = target_degree = 1
   ```

   Unary path compression may remain an internal optimization of the common
   assembler.

4. Preserve compact `Topology`, `TopologyBindings`, `LayerPlan`, and the
   layered CUDA target executor.

5. Validate realized forest identities, unique leaf endpoints, action spans,
   and per-partition row bounds when sealing.

6. Make ordinary and replica paths submit the same packet type. Replica is a
   lane axis, not another recorder.

### Deletion gate

At the production cutover, delete:

- `AsyncTopologyPublisher.seal_deferred_unary()` as an independent topology
  construction authority;
- poker-specific `BranchedTopologyRecorder` lifecycle/staging authority; and
- replica-specific topology assembly.

Reusable low-level packet buffers may remain under generic names in core.

### Exit gate

- One assembler consumes all twelve modes.
- Sequential traces are degree-one packet traces.
- Ordinary one-lane and replica one-lane topologies are identical up to lane
  indexing.
- No poker module materializes or seals generic topology.

### Resolves

- F2's separate topology implementation.
- The topology portion of F1 and F4.
- Topology-capacity structural ambiguity.
- Acceptance criteria 4, 5, 17, and part of 18.

## 9. Stage 6 — implement one core iteration engine

### Objective

Create the sole lifecycle authority.

### Work

1. Implement one core `IterationEngine` state machine owning:

   - phase entry and release;
   - semantic FIFO scheduling;
   - root publication;
   - model cohort selection;
   - `RolloutTransaction` calls;
   - topology packet acceptance;
   - Leaf readiness and completion scheduling;
   - topology/layer-plan sealing;
   - learner sealing;
   - target execution;
   - learner invocation;
   - failure agreement and recovery;
   - metrics; and
   - final close.

2. Make the lane axis universal:

   - one lane is `n_replica=1`;
   - multiple replicas use the same phase state and scheduler;
   - shared inference accepts aligned lane cohorts;
   - distributed execution wraps failure agreement and optimizer collectives;
     and
   - no constructor dispatches to a replica-specific engine.

3. Move sequential root-count recovery and branched controller backoff to one
   recovery boundary. Schema/controller hooks decide the control adjustment;
   the engine decides when a failed attempt is retried or the run fails.

4. Give the common engine explicit lifecycle instrumentation. A profiler may
   observe or request a test-only stop boundary around the engine, but the
   production engine must not contain `benchmark_rollout_only`,
   `benchmark_tree_only`, or a manual phase-walking early return.

5. Keep sampling differences behind hooks:

   - sampled versus RB;
   - fixed versus KKT controller;
   - self-play versus BR role/frozen policy; and
   - aligned versus ragged cohort descriptor.

6. Initially exercise the engine only through the Stage 0 shadow fixtures.
   Do not expose it as a second production selector.

### Exit gate before cutover

- The common engine reproduces all twelve golden traces.
- One-lane and multi-lane execution use the same engine type and method.
- Distributed wrapping changes collectives/failure agreement, not lifecycle.
- Poker contains no phase progression or owner release.

### Resolves

- F1.
- Most of F2 and F4.
- The duplicate recovery and replica orchestration findings.
- The production benchmark-bypass issue in Section 6.4.
- Acceptance criteria 1, 2, 3, 8, 16, 17, and 18.

## 10. Stage 7 — unify learner batches, objective, and optimizer sequencing

### Objective

Remove the remaining mode-specific training pipelines before production
cutover.

### Work

1. Define one `DecisionBatch` protocol:

   ```text
   sigma_index
   action_ptr
   action_index
   sigma_fields
   action_fields
   objective_weight
   ```

2. Represent sequential action spans as length one:

   ```text
   action_ptr = arange(B + 1)
   action_index = arange(B)
   ```

3. Allow small data-access strategies, not different learners:

   - aligned contiguous sequential rows;
   - ragged branched rows;
   - BR compacted rows; and
   - replica lane descriptors.

4. Implement one forward/loss/backward skeleton owning:

   - autocast;
   - current/reference forward;
   - policy composition;
   - old/current log ratio;
   - clipping;
   - value loss;
   - entropy;
   - reference KL;
   - objective weighting;
   - backward; and
   - common metrics.

5. Plug in only the mathematically real differences:

   - sampled versus RB posterior/private-domain reduction;
   - optional branched trace and cost terms; and
   - mode-specific row weights.

6. Implement one optimizer state machine for:

   - logical step boundaries;
   - zero-grad;
   - loss scaling;
   - distributed finite/failure agreement;
   - gradient reduction;
   - optimizer step/version;
   - metric reduction; and
   - run-state publication.

7. Rename generic classes now only when the old names would otherwise leak
   into the new public core. Broad package/file renaming remains late.

### Deletion gate

At cutover, delete:

- separate sequential and branched optimization state machines;
- separate sequential, branched-sampled, and branched-RB objective bodies;
- mode-named learner samplers that own scheduling rather than data access; and
- misleading `*SampledIteration` types that execute RB.

### Exit gate

- One objective skeleton matches every Stage 0 value/gradient fixture.
- One optimizer loop advances all modes and lane counts.
- Sequential and branched differences are visible as schema fields or small
  reduction functions, not method-sized branches.

### Resolves

- F3 completely.
- The learner half of F2.
- Learner/preflight duplication in F1 and F4.
- Acceptance criteria 6 and 7.

## 11. Stage 8 — atomic production cutover and duplicate deletion

### Objective

Make architectural uniqueness true in production before further optimization.

### Work

1. Change `TrainingSession.step()` to construct/invoke only the common core
   engine with a mode schema and poker hook bundle.

2. In the same merge series, delete the displaced authorities:

   - `SequentialPokerIteration` as an engine;
   - `_BranchedSampledPokerAttempt` as an engine;
   - `ReplicaBranchedPokerIteration` as an engine;
   - sequential and branched recovery wrappers;
   - separate topology assemblers;
   - separate learner optimization loops;
   - profiling-only lifecycle branches; and
   - all adapters used only to shadow the new engine.

3. Delete runtime dispatch on sequential versus branched engine type,
   one-lane versus replica engine type, and ordinary versus distributed
   lifecycle.

4. Do not leave aliases, deprecated imports, environment selectors, or a
   “legacy” backend. Migrate all callers and tests in the cutover.

5. Run the full twelve-mode CUDA matrix, one-lane/replica parity,
   checkpoint/resume, recovery, failure atomicity, and distributed gates.

### Exit gate

- A source search finds exactly one rollout-to-final-release owner.
- `TrainingSession` has no mode-to-engine dispatch.
- There is exactly one topology assembler, learner objective skeleton, and
  optimizer loop.
- No deleted engine remains reachable through tools or tests.
- Current behavior, memory, and matched-work performance do not regress beyond
  predeclared tolerances.

### Resolves

- The central “one modular implementation” failure.
- New parallel implementation identified in Section 6.2.
- The strongest parts of F1–F4.
- Acceptance criteria 1–8 and 16–18.

No later performance stage should begin until this gate passes.

## 12. Stage 9 — make storage access and target allocation exact

### Objective

Remove the major avoidable copies and over-allocation on the now-common path.

### Work

1. Make one-lane stable family prefixes simple storage-sharing slices of only
   model-required retained fields.

2. For aligned replica heads, expose a `[lane,row,...]` view with proven
   strides. When heads differ, pack only required model fields into an
   explicitly admitted workspace.

3. Stop gathering every hot `FamilyWork` column into model batches.

4. Prove directness with data pointer, storage offset, stride, and mutation
   alias tests. Do not accept a class name or docstring as evidence.

5. Treat `target_arena_cap` as an admission ceiling, not an allocation size.
   After the common planner returns exact per-lane/per-output extents:

   - validate them against the ceiling;
   - calculate exact phase admission;
   - allocate each fixed-address `LayerWork` slab at its exact extent;
   - publish completion rows;
   - retain the slab through target publication; and
   - release it at the one common lifetime boundary.

6. Replace arbitrary learner sigma/action per-row workspace accounting with
   implementation-derived concurrent tensor/workspace calculations.

7. Re-run the HU beta-2.4 capacity oracle. The exact planned allocation should
   replace the previously avoidable approximately 1.916 GiB.

### Exit gate

- Stable one-lane prefixes are views.
- Replica packing is minimal and its bytes are admitted.
- Self-play retained inputs are not copied at seal.
- `LayerWork` allocated bytes equal exact planned extents.
- Admission is based on actual owners and concurrent operations, not free
  per-row estimates.
- Family inference and phase-memory measurements improve or remain within a
  justified tolerance.

### Resolves

- F6.
- F7.
- The learner workspace-accounting defect from the config audit.
- Acceptance criteria 10, 11, 12, and 19.

## 13. Stage 10 — overlap completion and remove host-driven steady-state
control

### Objective

Implement the designed concurrent lifecycle and make scheduling uniform across
ordinary and replica lanes.

### Work

1. Let a dedicated completion stream consume committed stable Leaf prefixes
   while later families continue rollout.

2. Use explicit stream events for:

   - Leaf publication visibility;
   - evaluator input ownership;
   - completion epilogue completion;
   - topology/layer-plan dependencies; and
   - target consumption.

3. Remove the global `torch.cuda.empty_cache()` and device synchronization
   between rollout and completion. External evaluator/library workspaces must
   be reserved by admission and queried from backend contracts.

4. Allow topology planning and BR compaction to overlap the final completion
   tail where dependencies permit.

5. Move queue depth, packet readiness, and inflight progress to
   accelerator-resident counters/events. Host observation should be limited to
   phase boundaries, exceptional failure/recovery, metrics sampling, and
   unavoidable evaluator contract boundaries.

6. Give ordinary and replica lanes the same inflight packet policy. Remove the
   replica wait-after-every-publication behavior.

7. Preserve deterministic logical RNG and publication ordering independently
   of stream timing.

### Exit gate

- A deterministic CUDA test proves completion begins before final rollout
  commit.
- No production phase transition calls `empty_cache()`.
- Stream/lifetime tests detect premature owner reuse.
- Ordinary and replica host synchronization profiles converge.
- Host synchronizations are eliminated from steady-state family progress or
  each remaining synchronization has a documented external dependency.

### Resolves

- F8.
- F10.
- Remaining scheduler/replica asymmetry.
- Acceptance criterion 13 and the host-control portion of 19.

## 14. Stage 11 — implement and measure the remaining poker acceleration
operators

### Objective

Complete the high-value design ideas after the common path makes each kernel a
single implementation.

### Work in measured order

1. **Collision-conditioned posterior sampler**

   Implement the proposed parent-CTA CUDA operator with:

   - logical first-success semantics;
   - child-zero reuse;
   - bounded rejection;
   - deterministic logical RNG;
   - collision checks without a dense `[G,256,P]` envelope;
   - failure address and attempt telemetry; and
   - one clear PyTorch reference in tests only.

2. **Direct selected-action policy**

   Compute the selected event numerator and legal-event normalizer directly.
   Do not materialize `[B,H,E]` probabilities for retained selected actions.
   Preserve exact gradients and numerical stabilization.

3. **Model-level profiling**

   Re-profile after direct views, exact packing, overlap, and the first two
   kernels. Investigate model fusion, `torch.compile`, CUDA graphs, and
   attention backend behavior using the resulting profile, not the old one.

4. **Transition, folded statistics, and chance closure**

   Profile representative multiway workloads, especially six-player
   configurations. Implement the proposed fused commit/transition/folded
   construction where it closes a measured bottleneck. The closure condition
   is either:

   - a fused production implementation meeting a declared performance gate;
     or
   - measured evidence that the proposal is not beneficial, followed by an
     explicit update to the queued design removing it as a requirement.

   Leaving the idea indefinitely described as “should fuse” is not closure.

5. Keep all reference implementations outside selectable production paths.
   Production selects one operator per function and fails clearly if its
   required CUDA capability is unavailable.

### Exit gate

- Posterior and selected-policy gates from
  [high_priority_cuda_kernels_20260728.md](high_priority_cuda_kernels_20260728.md)
  pass.
- Statistical-law, gradient, rejection-exhaustion, and determinism tests pass.
- Multiway transition/folded work has an implementation or an evidence-backed
  design retirement.
- Updated matched-work profiles identify no remaining unexamined acceleration
  item from the audit.

### Resolves

- F9.
- The remaining “all core acceleration ideas” failure.
- Acceptance criteria 15 and 19.

## 15. Stage 12 — compile the final manifest and replace the YAML surface

### Objective

Replace contradictory hand-authored internals with a small user specification
and one generated immutable execution manifest based on the final architecture.

### Work

1. Define three data products:

   - user-authored scientific/run specification;
   - compiler-produced resolved manifest; and
   - runtime telemetry.

2. Keep only genuine user choices:

   - mode axes;
   - $P,S,R$;
   - model architecture;
   - applicable objective/optimizer/schedule choices;
   - completion samples in RB;
   - fixed/KKT controller choices;
   - BR frozen-profile identity and initialization;
   - seed, reproducibility, utility;
   - roots, replicas, recovery floor;
   - explicit CUDA device;
   - iteration/checkpoint/output control;
   - one memory reserve/usable-fraction policy; and
   - optional documented microbatch preferences.

3. Use discriminated unions so inactive fields are absent:

   - MSE versus clipped value loss;
   - constant versus decaying entropy;
   - epochs versus fixed gradient steps;
   - iteration versus exposure reference refresh;
   - SDPA versus chunked attention;
   - sequential versus fixed versus KKT controller;
   - sampled versus RB completion; and
   - self-play versus BR target profile.

4. Delete `completion.strata` until a second implemented stratifier exists.

5. Derive in one versioned compiler:

   - natural input/history/private shapes;
   - action and model token bounds;
   - player/evaluator/stratum widths;
   - precision dtype and alignment;
   - player, trained, action, folded, leaf, topology, and edge capacities;
   - coherent forest and target-arena bounds;
   - rollout/chance/topology/completion/learner pack limits;
   - exact/conservative workspace bytes from implementations and backend
     contracts;
   - per-partition learner physical caps;
   - actual device memory and applied reserve; and
   - all phase admissions.

6. Enforce structural derivations:

   - sequential action rows equal trained rows;
   - BR trained capacity follows balanced selected roots plus per-seat bounds;
   - topology/leaf/edge bounds satisfy the forest relation;
   - KKT minimum, target, safe, and hard capacity are ordered;
   - per-partition trained/action capacities are coherent, not merely coherent
     in aggregate; and
   - mode-inapplicable owners have zero presence rather than fake one-row
     minima.

7. Remove these user authorities:

   - `layout.model_input_shapes`;
   - layout player/private/stratum widths;
   - layout/model dtype duplication;
   - field alignment;
   - all raw row-byte estimates;
   - total device memory claims;
   - duplicate root/beta/query values;
   - raw owner/edge/action/trained caps;
   - `required_external` zeros;
   - arbitrary library/evaluator/movement workspace;
   - independent target-arena cap; and
   - independent sigma/action learner pack caps.

8. Merge reference caching into one execution policy:

   - off;
   - automatic; or
   - one explicit byte budget.

9. Hash and checkpoint:

   - fully resolved scientific config;
   - resolved manifest;
   - derivation and backend contract versions;
   - observed device identity; and
   - source specification canonical hash.

10. Supply a versioned legacy-config importer only for migration. It must
    recompute every derived value, reject disagreements, and emit the new user
    spec plus resolved manifest. Delete it before final production closure;
    it must not become a second accepted schema.

11. Regenerate all active configs and convert profiler scenarios into
    explicitly typed diagnostic specifications compiled through the same
    authority.

### Exit gate

- Normal YAML contains no `layout` or raw capacity/workspace manifest.
- Every user field has a production consumer in its applicable mode.
- Every derived field has one code authority and version.
- Parser/config mutation tests cover every remaining user option.
- Manifest round trips and checkpoint/resume hashes are stable.
- No legacy importer or old schema remains in production.

### Resolves

- Every configuration issue in audit Section 9.
- The stale-config failure permanently rather than cosmetically.
- Configuration acceptance criteria 1–15.
- Audit acceptance criteria 21 and 22.

## 16. Stage 13 — compact names, files, dead surface, and documentation

### Objective

Finish the cutover rather than leaving migration residue or overstated closure
claims.

### Work

1. Rename `pbg/games/poker/queued/` to the final poker implementation package.
   Update all callers in one migration; leave no import alias.

2. Rename generic RB-capable classes that still say `Sampled`.

3. Split large files by ownership, not mode. A reasonable final organization
   separates:

   - core engine/scheduler;
   - topology protocol/planning;
   - storage owners/lifetimes;
   - learner data access;
   - objective/optimizer;
   - poker state/transition;
   - poker sampling;
   - poker policy;
   - poker folded state;
   - evaluator binding; and
   - model architecture.

   Do not recreate `sequential.py` and `branching.py` orchestration files.

4. Run method-level and field-level dead-code analysis:

   - methods and branches, not just top-level classes;
   - schema writer/readers;
   - twelve-mode runtime coverage;
   - protocol/reflection allowlists;
   - zero-hit accepted branches; and
   - duplicate lifecycle-owner detection.

5. Delete every unexplained zero-hit method, field, branch, compatibility
   name, profiling bypass, stale generator, and obsolete test helper.

6. Reconcile queued design documents with the final implementation:

   - storage ownership and lifetimes;
   - folded raw-mass decision;
   - Leaf extensions;
   - one topology protocol;
   - exact arenas and completion overlap;
   - chosen/retired fusion work; and
   - configuration compiler.

7. Preserve historical audit/closure reports as historical evidence. Publish
   a superseding closure that explicitly corrects the former “one engine” and
   reachability overclaims rather than silently relying on them.

8. Update comments and docstrings that say “sole,” “direct,” or “exactly” only
   when a mechanical test now proves the claim.

### Exit gate

- No package or class name describes a superseded migration phase or wrong
  estimator applicability.
- No production-only dead surface remains unexplained.
- Reachability claims distinguish top-level reachability, method coverage,
  field consumption, and architectural uniqueness.
- Final queued-design documents and implementation agree.

### Resolves

- F14.
- Section 6 naming, profiling, and reachability limitations.
- Remaining F11 dead surface.
- The simplicity/elegance and documentation-governance findings.
- Acceptance criterion 20.

## 17. Stage 14 — final closure and production acceptance

### Objective

Prove the result rather than declaring the migration complete from imports or
unit tests alone.

### Required gates

1. **Architectural uniqueness**

   - one engine type and lifecycle method;
   - one scheduler;
   - one topology assembler;
   - one learner objective skeleton;
   - one optimizer loop;
   - one config compiler;
   - no alternate production implementation.

2. **Mode completeness**

   All twelve mode rows complete a CUDA optimizer step through that engine.
   One-lane, multi-lane, and distributed execution differ only by execution
   dimensions and collectives.

3. **Boundary**

   Core has no poker imports. Poker has no phase, release, recovery-loop,
   optimizer-sequencing, or distributed-preflight ownership.

4. **Storage**

   - one owner per lifetime;
   - metadata-only `LearnerIndex`;
   - no self-play seal copy;
   - direct stable one-lane views;
   - exact `LayerWork` extents;
   - no unused generated fields; and
   - coherent topology/action/trained capacities.

5. **Concurrency**

   Completion overlaps rollout, no global cache-empty barrier remains, stream
   dependencies are deterministic, and steady-state rollout is not
   host-controlled.

6. **Numerics and reproducibility**

   Golden targets, losses, gradients, optimizer versions, RNG identities,
   recovery decisions, checkpoint/resume, and ordinary/replica/distributed
   parity all pass.

7. **Performance**

   - posterior and selected-policy kernel gates pass;
   - family inference no longer pays avoidable all-field gathers;
   - exact arena memory matches the capacity oracle;
   - host synchronization is reduced to documented boundaries;
   - multiway fold/transition work is implemented or formally retired from
     the design with evidence; and
   - matched-work throughput and memory do not regress.

8. **Configuration**

   Normal YAML is small, all internals are compiled, no stale active config or
   old importer remains, and resolved manifests are deterministic and
   checkpointed.

9. **Dead-code closure**

   Method, field, branch, and mode-coverage audits find no unexplained
   production surface. Historical reports are not used as proof of the new
   architecture.

10. **Operational CUDA hygiene**

    The complete suite is run on explicitly selected idle GPUs with the
    required venv/toolkit environment, and every selected GPU returns to its
    clean idle baseline.

### Final deliverable

Publish a closure report that links:

- the original
  [design-fidelity audit](pbg_design_fidelity_audit_20260729.md);
- this remediation order;
- the twelve-mode trace/parity matrix;
- source-level uniqueness checks;
- the method/field dead-code report;
- final resolved-config examples;
- CUDA correctness results;
- memory-capacity comparisons; and
- before/after matched-work performance.

The closure report should state remaining limitations explicitly. “All
top-level definitions are reachable” is not an acceptable substitute for any
gate above.

## 18. Complete issue-to-stage traceability

| Audit issue | Primary stage(s) | Required closure |
|---|---|---|
| F1: three production iteration engines | 6, 8 | One core engine; old engines and recovery wrappers deleted |
| F2: sequential not degree-one specialization | 5, 7, 8 | One topology protocol and one decision/learner representation |
| F3: three objective bodies and two optimizer state machines | 7, 8 | One objective skeleton and optimizer loop |
| F4: generic orchestration in poker | 4, 6, 8 | Enforced hook boundary; poker cannot advance phases |
| F5: payload-bearing `LearnerIndex` | 3 | Metadata-only index; no self-play payload copy |
| F6: maximum-sized `LayerWork` allocation | 9 | Exact planned extents under immutable ceilings |
| F7: advanced-index “direct” prefix | 9 | Storage-sharing slices/strided views or admitted minimal packing |
| F8: no completion/rollout overlap | 10 | Event-driven completion starts before final rollout commit |
| F9: unimplemented posterior/policy/folded fusion | 11 | Kernels or evidence-backed retirement reflected in design |
| F10: host feedback controls progress | 10 | Device/event-driven steady-state scheduling |
| F11: unused fields and folded-row divergence | 2, 3, 13 | Contract settled; unused fields deleted; field audit clean |
| F12: undocumented folded `log_mass` | 2 | Prove and document, or remove |
| F13: undocumented Leaf extensions | 2 | Mode-applicable extension contract |
| F14: overstated closure/reachability | 0, 13, 14 | Method/field/branch evidence and superseding closure |
| New parallel code mistaken for completed cutover | 6, 8 | Atomic cutover with no compatibility selector |
| Profiling-only lifecycle branches | 6, 8, 13 | External instrumentation/test boundary; branches deleted |
| Transitional `queued` and `Sampled` names | 7, 13 | Final ownership-based names with no aliases |
| Weak method/field reachability evidence | 0, 13, 14 | Method/field/branch analysis plus twelve-mode coverage |
| 15 stale model-input widths | 1, 12 | Immediate regeneration; final derivation-only schema |
| Dead `completion.strata` | 12 | Removed until multiple implementations exist |
| Duplicated root/beta/precision/chunk authorities | 1, 12 | Early mismatch rejection; final single user authority |
| Editable model/history/private/token shapes | 1, 12 | One formula library and compiler |
| Independent trained/action caps | 2, 12 | Per-seat BR proof and target-degree derivation |
| Incoherent topology/leaf/edge caps | 2, 5, 12 | Forest identity in planner/compiler |
| Phantom HU folded capacity | 2, 3, 12 | Mode-absent owner after proof |
| Arbitrary learner workspace row bytes | 9, 12 | Actual concurrent workspace accounting |
| User-declared hardware observations/workspaces | 10, 12 | Runtime/backend query plus one memory policy |
| Inapplicable loss/schedule/reference fields | 12 | Discriminated unions; no null/zero placeholders |
| Ambiguous profiler YAML schema | 1, 12, 13 | Distinct typed diagnostic spec compiled by same authority |
| Core files mixing ownership and excessive complexity | 4–8, 13 | Generic responsibilities in core, then ownership-based split |
| Missing direct-view, overlap, and exact-arena acceleration | 9, 10 | Memory/view/concurrency gates |
| Missing measured poker kernels | 11 | Statistical and performance gates |

Every row must be closed. A row may be retired only by an explicit,
evidence-backed design decision, never because a reachability tool does not
report it.

## 19. Recommended merge boundaries

The stages above need not be one giant branch. The safest merge sequence is:

1. baseline fixtures, ledgers, and non-blocking structural checks;
2. config derivation verifier and stale-config regeneration;
3. approved storage/Leaf/folded contract decisions;
4. storage lifetime repair and unused-field deletion;
5. core/game hook protocols and boundary gates;
6. common topology packet/assembler under tests;
7. common engine under tests;
8. common learner/optimizer under tests;
9. atomic runtime cutover plus deletion of all displaced production engines;
10. direct views and exact `LayerWork`;
11. completion overlap and device scheduling;
12. posterior and selected-policy kernels;
13. multiway transition/folded closure;
14. final config compiler, checkpoint migration, and deletion of old schema;
15. package/file rename, dead-code sweep, documentation reconciliation; and
16. final CUDA/performance/closure report.

Each merge must leave production with either the old sole path before Stage 8
or the new sole path after Stage 8. Test-only shadow construction is allowed;
two selectable production engines are not.

## Final recommendation

The critical path is:

> contracts and lifetimes → one topology protocol → one core engine → one
> learner → atomic deletion → exact/concurrent execution → kernels → final
> compiled configuration → dead-code closure

Do not start by renaming packages, adding more validation around editable
manifests, or writing mode-specific kernels. Those actions make the current
shape more permanent.

The remediation is complete only when the repository is easier to describe
than it is today: one engine, one schema-generated storage plan, one topology
and learner path, a small poker hook surface, and a small user configuration.
