# Training, target, learner, distributed, and checkpoint audit

## Scope and result

This batch reviewed the production training path against
`00_math_and_modes_contract.md`, `01_pseudocode_storage_contract.md`, and
`02_historical_regression_contract.md`. The current normal, sequential-RB, fixed
branched-RB, and KKT branched-RB target and loss equations are statically
faithful in the reviewed path. The fixed/KKT distinction correctly affects
rollout intensity, not the downstream `branched_rb` estimator: when the model
emits its public T/C heads, both regimes retain frozen T/C values and train the
same Gamma-deviance targets.

The accelerated learner is also substantially preserved: it uses exact global
stratum masses, arbitrary positive statistical counts, direct subspan backward
into one fp32 flat gradient buffer, a SUM collective, a common nonfinite gate,
and separately bounded optional reference-cache construction. The protected
root prefix, queue-pressure nominal/executed correction, and checkpointed KKT
controller state remain explicit.

There is one distributed-correctness blocker: mandatory rank-local failures do
not enter the required common-order terminal agreement, and the learner starts
its mass collective before it proves mandatory singleton feasibility. There
are also confirmed CUDA host-synchronization and unbounded-host-transient
regression risks in target finalization, plus production-reachable CPU support
and several test-only/compatibility objects contrary to the requested lean GPU
surface.

## Findings

### AUD-TL-001 — Mandatory rank-local failures can strand peers in collectives

- Severity: blocker
- Confidence: confirmed
- Contract: `STORE-038`, `STORE-039`
- Modes affected: normal, sequential RB, branched fixed beta, branched KKT; BR
  where it uses the shared learner
- Evidence:
  - `pbg/core/training.py:2230-2238`, `_one_iteration`, calls rollout, seal, and
    target construction without a rank-common failure envelope.
  - `pbg/core/training.py:2963-3040`, `run_training`, catches only the
    pre-draw sequential `MemoryAdmissionStop`; it has no terminal agreement for
    rollout, seal, completion, target, or learner transaction failure.
  - `pbg/core/learner.py:1112-1123`,
    `DistributedFixedRolloutUpdate.__call__`, starts asynchronous
    `begin_log_mass_exchange()` before proving the mandatory uncached singleton
    plan.
  - `pbg/core/learner.py:1126-1141` performs mandatory reference singleton
    forwards after that collective starts; an exception can leave the
    collective outstanding.
  - `pbg/core/learner.py:1225-1227` later waits for the mass exchange, and
    `pbg/core/learner.py:1384-1401` enters gradient SUM/update collectives.
- Current behavior: a page admission, seal, completion, target, mandatory
  singleton, or required learner failure raises locally. A healthy peer may
  proceed into target/mass/gradient collectives. No common flag causes all ranks
  to release current owners and atomically checkpoint the last completed
  iteration.
- Required behavior: after all rank-local rollout work has stopped, every rank
  must reach one common-order failure agreement before targets or mass
  collectives. Any hard failure must make all ranks release current stores,
  checkpoint the last completed iteration, and terminate. Mandatory singleton
  feasibility must be established before launching the mass exchange; only the
  optional cache transaction may fall back locally under the specified common
  protocol.
- Consequence: a single-rank OOM/admission/backend failure can deadlock a
  multi-GPU job or terminate ranks asymmetrically without the required clean
  checkpoint. This is precisely the failure mode the authoritative pseudocode
  forbids.
- Recommended disposition: fix. Introduce the explicit common-order terminal
  control agreement around rollout/seal/completion/targets and mandatory learner
  admission; reorder learner mass exchange after mandatory feasibility. Add an
  injected asymmetric two-rank failure test for every mandatory phase.
- Verification needed: two-GPU NCCL tests that inject failure on only one rank
  before targets, before mass exchange, and before backward; assert no hang,
  identical terminal disposition, owner release, and last-completed checkpoint.

### AUD-TL-002 — Sequential target builders perform per-decision CUDA scalar synchronizations

- Severity: high
- Confidence: confirmed
- Contract: `STORE-031`, `HIST-PERF-02`; mathematical behavior remains
  consistent with `MATH-LAW-004`, `MATH-NORMAL-003`, and `MATH-SRB-004`
- Modes affected: normal, sequential RB
- Evidence:
  - `pbg/core/targets.py:169-182`, `build_normal_targets`, repeatedly converts
    CUDA `leaf_row`, CSR pointers, partition ids, decision rows, owners, and old
    values to Python scalars inside trajectory/decision loops.
  - `pbg/core/targets.py:251-260`, the sequential-RB extent pass, repeats the
    same topology/owner scalar reads.
  - `pbg/core/targets.py:281-312`, completion/reverse passes, repeat those reads
    and additionally execute `bool(pool_valid[pool_row])` on device state.
  - `tests/test_refactor_storage_targets.py:274-379` verifies return values and
    exact/model width separation, but not device synchronization count or
    equal-work timing.
- Current behavior: the equations are correct, but CUDA execution serializes
  host and device at trajectory/decision granularity. This is the pattern the
  earlier vectorization work removed elsewhere, and the post-refactor path has
  no comparable benchmark.
- Required behavior: target contraction must be bounded and device-resident (or
  use a declared, bounded structural transfer), without per-node or
  per-decision synchronization.
- Consequence: target time can scale with thousands of synchronous launches and
  materially regress the historical target/value optimization despite passing
  numerical tests.
- Recommended disposition: fix and benchmark. Restore a vectorized/segmented
  player-clock reducer for both sequential modes without recreating discarded
  rollout state.
- Verification needed: a CUDA synchronization-count gate and same-workload
  benchmark against the historical 2,048/2,560-root target artifacts named by
  `HIST-PERF-02`.

### AUD-TL-003 — Branched finalization materializes full-tree host metadata and a full Python alias workspace

- Severity: medium
- Confidence: strong
- Contract: `STORE-031`, `STORE-035`
- Modes affected: branched fixed beta, branched KKT
- Evidence:
  - `pbg/core/targets.py:76-98`, `_walker_alias`, copies every decision weight to
    CPU, converts the scaled vector to a Python list, and builds full-size Python
    `small`/`large` work lists before returning the required alias tensors.
  - `pbg/core/targets.py:355-389`, `plan_branched_frontier`, copies full
    parent/degree topology to CPU and allocates Python `ready` state proportional
    to all nodes.
  - `pbg/core/targets.py:445-477`, `build_branched_rb_targets`, makes another
    full topology copy plus partition owner, action-pointer, and nominal-count
    host copies.
  - `pbg/core/targets.py:487-489` allocates Python `node_slot`, `ready`, and
    `free_slots` lists proportional to realized tree/frontier size.
- Current behavior: the retained alias destination is mathematically required,
  and target tensors are bounded by the computed live frontier, but host
  construction introduces duplicate transient state proportional to the entire
  logical tree/decision set. It is not independently bounded as a physical
  subspan and is not included in the CUDA execution metrics.
- Required behavior: bounded target contraction and finalization whose
  transient work does not grow as an undeclared duplicate of logical
  statistical work.
- Consequence: very large fixed/KKT trees can exhaust host memory or incur large
  synchronous D2H and Python costs even when GPU byte admission succeeds.
- Recommended disposition: optimize. Build alias tables on-device or in bounded
  chunks, and replace duplicate full-tree Python metadata with a declared
  structural owner/streaming reducer. Preserve exact alias probabilities and
  stable child identities.
- Verification needed: peak host RSS and D2H-byte measurements over increasing
  fixed trees, plus exact alias/target equality and physical-plan invariance.

### AUD-TL-004 — Branched loss validation synchronizes once per physical subspan

- Severity: medium
- Confidence: confirmed
- Contract: `STORE-031`, `HIST-PERF-01`
- Modes affected: branched fixed beta, branched KKT
- Evidence:
  - `pbg/core/learner.py:786-790`, `mode_loss`, executes
    `bool((counts <= 0).any())` on CUDA for every sampled physical subspan.
  - `pbg/core/learner.py:791-798` then derives a dynamic action extent from
    `cumulative[-1]` for each subspan.
  - The positive-action invariant is already structural and can be validated
    once when `TrainingStorage` is finalized at
    `pbg/core/storage.py:1358-1374`; repeating a host gate in every optimizer
    subspan is not necessary for the mathematical estimator.
- Current behavior: each branched physical subspan can force an implicit
  device-to-host synchronization before its loss kernels.
- Required behavior: validate immutable storage once, then keep subspan gather,
  loss, and backward device-resident and bounded.
- Consequence: the backward optimization survives structurally but can lose
  throughput as the number of physical subspans increases.
- Recommended disposition: fix. Move the invariant to storage finalization or
  use an asynchronous device assertion outside the repeated hot path; use a
  predeclared bounded ragged workspace.
- Verification needed: CUDA profiler/no-host-sync test for branched learner
  subspans and an equal-row before/after learner benchmark.

### AUD-TL-005 — The production training entry point still supports CPU and Gloo execution

- Severity: medium
- Confidence: confirmed
- Contract: repository requirement that production is NVIDIA-GPU-only; GPU
  memory contracts `STORE-030` through `STORE-034` are bypassed on this path
- Modes affected: all shared training modes
- Evidence:
  - `pbg/core/training.py:2546-2553`, `run_training`, explicitly selects CPU for
    a CPU completion backend and silently falls back to CPU when CUDA is absent.
  - `pbg/core/training.py:224-226`, `_hardware_metadata`, accepts CPU and labels
    it `cpu_reference` even under a hardware profile.
  - `pbg/core/training.py:443-461`, `_value_workspace_budget`, invents a 1-GiB
    CPU workspace rather than enforcing the GPU memory contract.
  - `pbg/core/training.py:857-858` and `876-877` bypass CUDA preflight and
    calibrated peak validation on CPU.
  - `pbg/core/training.py:2708-2717` supports multi-process Gloo training.
  - `pbg/core/training.py:2858-2867` creates synthetic CPU memory-controller
    headroom specifically so CPU runs can exercise training geometry.
- Current behavior: CPU is a first-class production selection in the same
  entry point, not a negligible incidental tensor fallback.
- Required behavior: the production launcher should fail early unless it owns
  an NVIDIA CUDA device and the configured CUDA evaluator/backend. Small pure
  unit tests may exercise math on CPU without retaining a CPU production path.
- Consequence: meaningful code and validation branches exist for an unsupported
  execution surface, weakening the guarantee that all production runs receive
  GPU memory admission, NCCL, and CUDA observability.
- Recommended disposition: delete/move to test support. Require CUDA/NCCL in
  `run_training`; keep only compact device-agnostic math helpers where doing so
  adds essentially no code.
- Verification needed: entry-point test that CPU/`auto` fallback is rejected
  before bundle allocation, while pure unit-level target/loss tests remain
  possible.

### AUD-TL-006 — Test-only and compatibility Python objects remain in the production package

- Severity: low
- Confidence: confirmed by repository-wide Python reference search
- Contract: requested production-only surface; related to `STORE-035` and
  `HIST-PERF-01` because parallel estimator helpers can preserve stale designs
- Modes affected: none; these objects are not reached by the five production
  modes
- Evidence:
  - Entire `pbg/core/distributed.py:11-72`: `WeightedDecisionStats`,
    `weighted_decision_stats`, `global_mass_objective`, and
    `stratum_objective_scale` are imported only by
    `tests/test_execution_foundations.py`; production uses the log-mass/alpha
    implementation in `pbg/core/learner.py:91-150` and `1359-1382`.
  - `pbg/core/learner.py:153-209`, `allocate_statistical_counts`, is referenced
    only by `tests/test_distributed_learner.py`; production calls
    `allocate_fixed_statistical_counts` at `pbg/core/training.py:2343-2348`.
  - Entire `pbg/core/logging_util.py:10-41`, `MetricsLogger`, is referenced only
    by `tests/test_p0_config.py`; production writes run/metric JSON directly in
    `pbg/core/training.py:2895-2955` and `3153-3162`.
  - `pbg/core/learner.py:878`, `ModeLoss`; `pbg/core/learner.py:1483`,
    `distributed_fixed_rollout_update`; and `pbg/core/targets.py:747-750`, the
    three `Build*Targets` aliases, have no Python callers.
  - `pbg/core/observability.py:235-250`, `ExecutionMetrics.record_cache`;
    `300-302`, `record_completion`; and `316-317`, `record_rank_wait`, have no
    production callers (only observability tests for the first three).
  - `pbg/core/game.py:114-118`, the protocol member
    `RolloutBackend.sampled_terminal_reward`, is not invoked by production; the
    poker path uses its keyed settlement operation.
  - `pbg/core/checkpointing.py:425-430` retains four compatibility aliases;
    training still imports the underscored aliases rather than the canonical
    functions.
- Current behavior: obsolete alternate estimator/logging APIs and unused
  aliases remain importable, and stale observability fields are emitted as
  permanent zero-valued metrics.
- Required behavior: one production implementation with no parallel legacy or
  test-only package APIs.
- Consequence: larger audit surface, ambiguity about the authoritative
  normalization/allocation path, and continuing compatibility scaffolding.
- Recommended disposition: delete the unused modules/objects and migrate the
  four internal checkpoint imports to canonical names before removing aliases.
  Move any useful test formula or logger fixture under `tests/`.
- Verification needed: import/reachability gate from the production CLI and the
  five mode traces after removal.

### AUD-TL-007 — Fixed-beta T/C training and fatal-failure behavior lack direct regression gates

- Severity: low
- Confidence: confirmed
- Contract: `MATH-FIXED-002`, `STORE-038`, `TEST-STORE-001`, `TEST-STORE-004`,
  `TEST-STORE-005`
- Modes affected: branched fixed beta, all distributed modes for failure
- Evidence:
  - Static fixed-beta T/C preservation runs through
    `pbg/core/branching.py:683-699`, `pbg/core/training.py:1863-1908`,
    `pbg/core/targets.py:508-523` and `604-690`, and
    `pbg/core/learner.py:832-854`.
  - `tests/test_refactor_storage_targets.py:451-568` labels its T/C fixture and
    assertions as dynamic and manually supplies controller columns; it does not
    prove that a real fixed-beta rollout retains/trains T/C from iteration zero.
  - `tests/test_p13b_ddp.py:164-199` verifies SUM and a common nonfinite
    optimizer skip, but no test injects a pre-collective hard failure on one
    rank.
- Current behavior: static wiring supports fixed T/C, but an accidental future
  coupling of `dynamic_heads` to branching mode could regress it unnoticed; the
  missing terminal agreement also has no hang-safe test.
- Required behavior: structural/equivalence tests for both fixed and KKT
  branched regimes and asymmetric hard failure at every collective boundary.
- Consequence: important mode and failure contracts are guarded by inference
  from shared code rather than direct acceptance tests.
- Recommended disposition: add focused tests after repairing `AUD-TL-001`.
- Verification needed: fixed-beta end-to-end target columns and nonzero T/C
  gradient from iteration zero; timeout-bounded two-rank injected failures.

## Confirmed conformance

- Frozen-policy boundary: `_build_iteration_data` keeps the live model in eval
  and under `torch.no_grad()` until targets are sealed
  (`pbg/core/training.py:1081-1108`, `1165-1409`), then optimization begins only
  after the rollout owner is deleted (`pbg/core/training.py:2318-2324`). This
  matches `MATH-LAW-002` and `STORE-037` without a rollout-model clone.
- Normal targets: reverse player-clock returns in
  `pbg/core/targets.py:137-201` implement `MATH-NORMAL-003`; realized actor type,
  frozen log probability/value, PPO/value clipping, entropy, and reference KL
  map directly in `pbg/core/learner.py:711-745`.
- Sequential RB: the leaf posterior is deduplicated per trajectory/player and
  exact state (`pbg/core/targets.py:203-343`); moment recursion uses frozen
  natural values through `phi`; loss weights every exact-state actor/value/
  entropy/KL element by the leaf posterior (`pbg/core/learner.py:760-781`). This
  matches `MATH-SRB-003` through `MATH-SRB-005` and `STORE-003`.
- Branched RB: child posterior weights actor terms and their mean weights parent
  value/entropy/KL terms (`pbg/core/learner.py:782-831`). Target contraction uses
  uniform child reductions, leave-one-out baselines, frozen-value singleton
  fallback, nominal/executed T/C correction, and the protected original root
  prefix (`pbg/core/targets.py:542-648`). This matches `MATH-BRANCH-004` through
  `MATH-BRANCH-010`.
- Fixed and KKT T/C: branched rollout retains model T/C output whenever emitted
  (`pbg/core/branching.py:683-699`); sealing and target generation key off the
  presence of those heads, not `branching_mode` (`pbg/core/training.py:1863-1908`,
  `pbg/core/targets.py:508-523`). Gamma-deviance T/C losses are applied in
  `pbg/core/learner.py:832-854`, including a zero masked T loss for fewer than
  two target children. This is consistent with `MATH-FIXED-002`,
  `MATH-KKT-003`, and `MATH-BRANCH-008`.
- Sampling/gradient normalization: finalization stores uniform sequential or
  log-weight alias laws (`pbg/core/targets.py:101-129`); global fp64 log masses
  form exact alphas (`pbg/core/learner.py:91-150`); each physical subspan uses
  `alpha / B` exactly once (`pbg/core/learner.py:1359-1382`). Gradient attributes
  alias one fp32 flat buffer, all-reduction is SUM, and the common finite result
  controls unscale/clip/update (`pbg/core/optimization.py:19-78`, `113-205`).
  This matches `STORE-035`, `STORE-036`, and `HIST-PERF-01`.
- Reference cache: mandatory uncached planning, pre-drawn sampled rows, separate
  bounded build shapes, transactional fallback, and RNG-preserving reuse remain
  in `pbg/core/learner.py:992-1224`, consistent with `STORE-034`, `STORE-039`,
  and `HIST-PERF-01` apart from the collective ordering in `AUD-TL-001`.
- Checkpoint state: atomic temp-file replacement stores model, optimizer,
  scaler, scientific config, CPU/CUDA and per-rank RNG, reference policy and
  iteration, sequential memory controller, KKT controller, and logical work
  (`pbg/core/checkpointing.py:148-210`). Load validates paired controller schema
  and state and restores per-rank RNG (`pbg/core/checkpointing.py:213-283`).
  Normal cadence gathers all rank RNG states before rank-zero save
  (`pbg/core/training.py:3163-3202`). This preserves the historical KKT and
  resume state under successful iterations.
- Last consumers: sampled rollout temporaries are deleted after exact sealing,
  rollout/completion owners are deleted after target construction, only
  `TrainingStorage` crosses into learning, each physical graph is deleted after
  backward, caches are cleared after the update, and training storage is deleted
  at iteration end (`pbg/core/training.py:1213-1226`, `1301-1324`, `1390-1409`,
  `2318-2324`, `2524`; `pbg/core/learner.py:1379-1383`, `1479`). This is
  consistent with the reviewed last-consumer discipline on successful paths.

## Validation performed

Focused non-GPU tests were run with the repository virtualenv:

```text
./tools/poker-python -m pytest -q \
  tests/test_refactor_storage_targets.py \
  tests/test_distributed_learner.py \
  tests/test_p13b_ddp.py \
  tests/test_adaptive_kkt_branching.py \
  tests/test_utility_normalization.py \
  tests/test_memory_control.py

64 passed in 18.81s
```

No GPU test was run in this batch. The existing CUDA tests include tiny
single-/two-GPU learner and acceptance coverage, but they do not quantify the
target host-sync risks or exercise asymmetric terminal failure agreement.

## Reviewed inventory

Fully reviewed:

- `pbg/core/targets.py`
- `pbg/core/learner.py`
- `pbg/core/losses.py`
- `pbg/core/utility.py`
- `pbg/core/optimization.py`
- `pbg/core/distributed.py`
- `pbg/core/checkpointing.py`
- `pbg/core/observability.py`
- `pbg/core/logging_util.py`
- `pbg/core/game.py`
- `pbg/core/training.py` target/loss/learner orchestration, analytical phase
  planning, frozen-policy boundary, distributed setup/update, controller,
  metrics, and checkpoint/resume sections
- Direct target, learner, DDP, KKT, utility, memory-control, and lifecycle tests
  cited above

Sampled or reviewed only where directly relevant:

- `pbg/core/branching.py` frozen V/T/C capture and release boundary
- `pbg/core/storage.py` training schemas and validation
- `pbg/core/topology.py` protected root and sequential/branched topology
  contracts
- `pbg/games/poker/rollout_backend.py` only for reachability of the generic
  protocol and observability methods

Not reviewed in this batch:

- Poker model internals, policy adapter mathematics, and model architecture
- Evaluator posterior/equity implementation
- Rollout queue ordering and page transactions beyond the interfaces consumed
  here
- BR-specific opponent-probe estimator correctness
- Full configuration/CLI mode reachability
- End-to-end current-versus-historical GPU performance measurements
