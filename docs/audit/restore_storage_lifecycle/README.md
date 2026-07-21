# Authoritative storage-lifecycle restoration index

This is the ordered, reviewable restoration chain from baseline `1cde6de`
(`Pre-refactor fix commit`). No phase was squashed. Each linked handoff records
its parent, ownership changes, preservation evidence, tests, and next starting
condition.

| Phase | Commit | Handoff |
|---|---|---|
| 1. Authority, characterization, deletion plan | `fed6b85` `test(core): define storage migration and ownership gates` | [`phase_1_authority_and_deletion_plan.md`](phase_1_authority_and_deletion_plan.md) |
| 2. Storage foundation and poker ownership | `a399e30` `refactor(core): add byte-admitted rollout storage` | [`phase_2_storage_foundation_and_poker_ownership.md`](phase_2_storage_foundation_and_poker_ownership.md) |
| 3. Normal and sequential-RB production migration | `9af6a29` `refactor(core): migrate sequential rollout storage` | [`phase_3_sequential_rollout_storage.md`](phase_3_sequential_rollout_storage.md) |
| 4. Fixed and dynamic branched production migration | `759cde6` `refactor(core): migrate dynamic branched storage` | [`phase_4_dynamic_branched_storage.md`](phase_4_dynamic_branched_storage.md) |
| 5. Sole-path cleanup, configuration, memory control | `dce75ff` `refactor(core): retire legacy rollout capacity` | [`phase_5_sole_path_and_memory_control.md`](phase_5_sole_path_and_memory_control.md) |
| 6. Full acceptance, profiling, documentation | `d0d246c` `test(core): validate sole memory-safe training pipeline` | [`phase_6_acceptance.md`](phase_6_acceptance.md) |

The chain has one architecture at its tip: byte-admitted paged/recyclable
rollout owners, exact `RolloutStorage`, bounded target construction, immutable
`TrainingStorage`, and bounded learner subspans. The Phase 6 handoff is the
current release verdict; the 2026-07-17 compatibility-path acceptance notes
are historical evidence only.

The user's master implementation prompt remained an unstaged worktree edit and
is not part of any phase commit.
