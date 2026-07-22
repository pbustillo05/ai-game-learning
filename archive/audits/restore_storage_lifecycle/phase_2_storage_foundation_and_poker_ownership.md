# Phase 2 handoff: storage foundation and poker ownership

Parent phase commit: `fed6b85` (`test(core): define storage migration and
ownership gates`).

Pre-existing unrelated worktree change: the user-edited
`docs/todo/restore_authoritative_storage_lifecycle_20260721.md`. It remains the
master prompt and is intentionally excluded from this phase commit.

## Completed requirements

The common storage foundation is now page- and byte-oriented:

- `PagedTensorArena` replaced the prototype chunk API. A multi-page request is
  admitted as one `ByteAccountant` transaction, allocation failure rolls back
  before page/cursor publication, logical row ids remain stable, and reset can
  either reuse admitted pages or release their reservations.
- Exact sealing owns an explicitly admitted destination and bounded reusable
  scratch block; metrics distinguish logical payload, page slack, source
  pages, destination capacity, and the source/destination/scratch peak.
- `RecyclableTensorPool` adds device-resident free-handle batches and explicit
  last-consumer release. It performs no handle `.cpu()`, `.item()`, or
  `.tolist()` conversion and reuses admitted physical rows without changing
  logical topology identities.
- Branched occurrence schemas now include the current dynamic-controller
  columns required by the reconciled theory. `BranchedTreeSpan` retains the
  protected root degree and pre-refill node boundary as two scalar metadata
  values and validates parent/child ownership with tensor operations.

Poker ownership now follows realized work:

- `PokerTensorStateStore` is a schema-defined paged recyclable owner. Bundle
  construction allocates zero state/history payload; pages are admitted only
  as rows are realized. The old eager `capacity * field` allocations and
  capacity-shaped history table are gone.
- `DevicePartitionSummaryCache` no longer owns a capacity-wide validity bitmap
  or capacity-wide lazy fields. CPU compatibility rollout uses sparse distinct
  anchor pages; all-live rows allocate no folded-summary storage.
- Poker partition schemas retain complete immutable public, legal-action, and
  padded lossless-history tensors plus natural-domain ranges. `state_ref` and
  the old `summary_anchor` field no longer cross the learner boundary.
- Folded summaries seal once per distinct dependency anchor into an exact
  immutable auxiliary pool. Decision rows retain only `folded_summary_ref`,
  and bounded learner gather resolves that reference before model forward.
- All-live postflop evaluator base terms are bounded forward-local work and
  allocate no persistent folded-summary pool.
- The compatibility training integration exercises real rollout, sealing,
  bounded gather, preflop and all-live postflop model forwards without reading
  mutable state from `policy_forward_partition`.

The phase-2 immutable-poker-input migration gate is now a normal passing test.
The four gates owned by phases 3--5 remain strict expected failures.

## Deleted implementations and new ownership boundaries

| Removed implementation | Replacement owner | Evidence |
|---|---|---|
| `ChunkedTensorArena` API and one-page-at-a-time mutation | transactional `PagedTensorArena` | multipage rollback, accounting, gather/scatter, sealing tests |
| eager tensors in `PokerTensorStateStore` | `RecyclableTensorPool` over the immutable `poker_live_state` schema | zero-allocation bundle construction, page/reuse and history tests |
| capacity-wide summary `valid` and `_fields` tensors | sparse page-backed distinct-anchor rows | fold refresh/cache test and zero-allocation all-live test |
| learner `state_ref` lookup and mutable-state model reconstruction | exact immutable partition inputs and `PokerPublicBatch` reconstruction | migration source gate and real compatibility-seal integration |
| per-decision folded-summary anchor handles into mutable state | exact sealed summary pool plus int32 pool reference | sparse pool gather test and poker partition schema tests |

No user-facing feature or evaluator contract was removed. The sibling poker
evaluator remains the sole summary/completion implementation and was not
modified.

## Preservation evidence

- Sealed-storage/learner behavior: exact versus natural partition domains,
  immutable `TrainingStorage`, bounded learner gather, and exact schema byte
  accounting pass their focused suites.
- Evaluator contract v1/v2 behavior: real sibling CPU rollout and a tiny CUDA
  rollout pass with compact-deck geometry and device-local tensors.
- Vectorized CUDA behavior: complete preflop/postflop input construction passes
  under CUDA synchronization debug mode after page gather/scatter stopped using
  dynamic `nonzero` selection.
- Dynamic branching: the occurrence schema carries nominal/executed counts,
  public KKT path/controller inputs, and queue cost; protected-root topology is
  validated without changing current adaptive-KKT tests.
- Poker model widths: preflop remains natural $K=169$ at $R=13$, postflop
  remains exact $H=1326$, and no street-specific postflop partition was added.
- Parent-linked history, evaluator summaries, learner telemetry, cache transfer
  helpers, and the current fixed/dynamic branching tests continue to pass.

## Validation

- `./tools/poker-python -m pytest -q tests/test_refactor_storage_targets.py
  tests/test_distributed_learner.py
  tests/test_storage_lifecycle_migration_gates.py
  tests/test_storage_lifecycle_poker_inputs.py tests/test_p13a_gpu_rollout.py
  tests/test_p13b_integration.py tests/test_adaptive_kkt_branching.py -k 'not
  cuda'` -- `89 passed, 9 deselected, 4 xfailed`.
- After confirming GPU 0 was idle,
  `POKER_GPU=1 CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q
  tests/test_p13a_gpu_rollout.py::test_real_sibling_execution_poker_tiny_cuda
  tests/test_aud26_no_host_sync.py::test_full_preflop_postflop_policy_inputs_request_no_host_sync`
  -- `2 passed` with the expected PyTorch prototype sync-debug warning.
- `nvidia-smi` after the CUDA checks showed GPU 0 back at 6 MiB, 0% compute,
  with only the display process present.
- `git diff --check` -- clean.

An unrelated existing test,
`tests/test_phase3_migration.py::test_checkpoint_v3_and_recent_legacy_metadata_migration`,
still references absent `configs/acceptance/nlhe_hu_normal.yaml`; it fails at the
parent commit before reaching phase-2 code and is not hidden in the focused
result.

## Temporary adapters and next starting conditions

The direct phase order is not yet installed. `_RolloutSealInput`,
`_seal_iteration_training_storage`, legacy `NormalDataset`, tree/reach caches,
and legacy target builders remain production consumers and are still covered
by strict migration gates.

Within that compatibility path, state rows are page-backed but remain live for
the whole rollout because the old tree/history representation is their last
consumer. Phases 3 and 4 must call `release_rows` at real last-consumer
boundaries. CUDA folded summaries are deliberately computed as bounded
forward-local work in the compatibility sampler: growing a sparse cache there
would require an unowned admission boundary or a host synchronization. The
direct sequential/branched rollout must instead write each immutable summary
dependency row at decision creation, making the compatibility exception and
`DevicePartitionSummaryCache` removable.

The old `node_budget` still supplies a maximum-row safety guard and the
transitional analytical memory forecast, but it no longer causes eager poker
state or summary allocation. Phase 5 remains responsible for removing this
configuration/forecast surface after phases 3 and 4 establish sole direct
rollout storage.

Phase 3 starts with complete immutable poker model-input schemas, exact sparse
input-pool resolution, paged/recyclable state primitives, compact sequential
topology, and authoritative normal/sequential target builders available. It
must make normal and sequential-RB production create `RolloutStorage` directly,
stream targets, release live state/history at their final consumers, and delete
their legacy owners rather than adding another conversion layer.
