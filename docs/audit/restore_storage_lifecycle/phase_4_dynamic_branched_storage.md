# Phase 4 handoff: fixed and dynamic branched storage

Parent phase commit: `9af6a29` (`refactor(core): migrate sequential rollout
storage`).

Pre-existing unrelated worktree change: the user-edited
`docs/todo/restore_authoritative_storage_lifecycle_20260721.md`. It remains the
master prompt and is intentionally excluded from this phase commit.

## Completed requirements

Branched production now crosses the same authoritative storage boundary as the
two sequential modes:

1. player decisions capture complete immutable natural partition inputs while
   their public state and active reach are live;
2. active full reaches use a page-backed recyclable pool and are released at
   the final expansion or stopped-leaf consumer rather than residing in an
   all-node `[N,P,H]` tensor;
3. exact sealing creates one contiguous-child `BranchedTreeSpan`, ragged
   per-partition action rows, exact leaf completion rows, and protected-root
   metadata;
4. mutable poker state and summary pages are released before target building;
5. `build_branched_rb_targets` consumes only `RolloutStorage`, uses a
   topology-precomputed reusable frontier arena, and returns immutable
   `TrainingStorage` directly;
6. the old padded target/conversion branch is unreachable for both fixed and
   dynamic branched production.

The bounded builder now carries public value vectors, cost, and queue-suppression
delta in reusable slots. It implements the current nominal-versus-executed
correction, writes masked trace-variance and cost targets, discards frozen
`T/C` predictions at finalization, and stores no production `Q` target. At a
refilled root it waits for the complete final topology but uses only the
ordinary protected prefix for target mass, action rows, `T/C`, and recurrence.

Structural planning performs one explicit bulk device-to-host topology transfer
and then uses CPU structural metadata; it does not read a CUDA scalar per node
or grow frontier payload through `torch.cat`.

## Ownership and migration evidence

| Retired branched behavior | Replacement owner/API | Evidence |
|---|---|---|
| all-node `reach_rows[node_budget,P,H]` | `RecyclableTensorPool` handles on live occurrences | fixed/dynamic production smokes finish with zero live reach rows and released poker pages |
| post-target branched conversion | branched `RolloutStorage` -> `build_branched_rb_targets` -> `TrainingStorage` | real owner-inventory integration test includes branched mode |
| padded target action axis | exact action rows plus `action_ptr` per partition | owner-inventory and protected-root action-span tests |
| growing backward payload frontier | exact `BranchedFrontierPlan.peak_slots` plus reusable mass/moment/value/cost/delta slots | multi-child, zero-mass, dynamic correction, and refill tests |
| mutable state/reach completion dependency | immutable leaf columns plus `complete_sealed` | CPU/GPU sibling-evaluator production smokes |
| fixed-only authoritative target builder | fixed/dynamic builder with transient frozen `T/C`, nominal/executed counts, and protected root degree | direct formula tests and adaptive KKT suite |

The production learner receives only natural model widths, exact posterior and
advantage widths, ragged actions, `T/C` targets, and sampling metadata. The
identity launch gate recomputes every ragged selected action and old value from
the immutable inputs before the first update.

## Preservation evidence

- Fixed and dynamic branching remain explicit; warmup behavior, log-space KKT
  controller updates, beta clipping, rank-addressed logical RNG, queue
  safeguard diagnostics, and counterfactual tree-size observation pass their
  existing tests.
- Duplicate-labeled children and parent-local logical ordinals are unchanged.
- Lower-size root refill retains fresh ordinals, final root transition masses,
  the pre-refill boundary, protected ordinary-root action prefix, and learning
  below added root children.
- Preflop old outputs remain on natural class width after the lossless `H -> K`
  quotient; postflop outputs and exact targets remain on `H`.
- Poker completion still uses the read-only sibling evaluator's immutable
  keyed interface, including compact-deck geometry and utility scaling.
- The existing bounded distributed learner, reference cache, flat gradient
  SUM, fused policy terms, checkpoint/controller state, and diagnostics remain
  downstream of the same `TrainingStorage` boundary.

## Validation

- `./tools/poker-python -m pytest -q tests/test_refactor_storage_targets.py
  tests/test_adaptive_kkt_branching.py tests/test_c09_labeled_branching.py
  tests/test_p13a_gpu_rollout.py tests/test_p13b_integration.py
  tests/test_storage_lifecycle_poker_inputs.py
  tests/test_storage_lifecycle_migration_gates.py tests/test_p11_losses.py
  tests/test_c11_minimal_losses.py -k 'not cuda'` -- `103 passed, 5
  deselected, 4 xfailed`.
- After `nvidia-smi` showed GPU 1 idle, a direct `R=4`, 8 bb, four-completion
  production smoke built exact branched `TrainingStorage` for fixed and
  post-warmup dynamic modes. Both realized one decision, one action, and one
  leaf, and both released the poker state arena.
- Final `nvidia-smi`: GPU 1 at 6 MiB and 0% utilization with no compute
  process. GPU 0's unrelated 446 MiB process was not touched.
- `git diff --check` -- clean at the phase boundary.

## Temporary adapters and phase-5 starting conditions

The family scheduler still assembles transient `DeviceTreeBuffer`/`TreeBatch`
and `PolicyCache` views before exact branched sealing. They no longer reach
completion, target construction, or optimization, but phase 5 must replace the
shared sequential/branched scheduler-local containers with the paged occurrence
and sealed-row APIs before the static deletion gate can pass.

The sequential best-response probe is the only production consumer that still
uses `streamed_branched_value`, `NodeTargets`, and the post-hoc
`_RolloutSealInput` conversion. Historical value/minibatch tests and several
variance/branching diagnostics also import the retired types. Phase 5 must
migrate those consumers to `RolloutStorage`/`TrainingStorage` or a read-only
diagnostics view, then delete `pbg/core/values.py`, the legacy minibatch path,
the shared tree/reach/cache types, `_RolloutSealInput`, and
`_seal_iteration_training_storage` rather than preserving dormant copies.

`node_budget`, capacity-sized work queues/controller columns, the transitional
analytical memory plan, and legacy configuration/checkpoint identity remain.
Phase 5 starts from a direct authoritative target path for all three training
modes and is responsible for making paged scheduling the sole rollout path,
retiring those capacity surfaces, migrating tools/configs/checkpoints, and
turning every static deletion gate into a normal pass.
