# Phase 3 handoff: normal and sequential-RB rollout storage

Parent phase commit: `a399e30` (`refactor(core): add byte-admitted rollout
storage`).

Pre-existing unrelated worktree change: the user-edited
`docs/todo/restore_authoritative_storage_lifecycle_20260721.md`. It remains the
master prompt and is intentionally excluded from this phase commit.

## Completed requirements

Normal and sequential Rao--Blackwell production now cross a real pre-target
storage boundary:

1. the family scheduler captures complete immutable natural poker inputs at
   each retained player decision;
2. captured packs enter page-backed input arenas and seal once into exact
   partition blocks, including a permutation copy for out-of-order family
   scheduling;
3. normal scalar old-policy rows or sequential natural old-policy rows seal
   alongside a decision-only `SequentialDecisionCSR` and exact leaf rows;
4. mutable poker state and folded-summary pages are released at the seal
   boundary;
5. `build_normal_targets` or `build_sequential_rb_targets` consumes only
   `RolloutStorage`;
6. the learner receives immutable `TrainingStorage` directly, without the
   post-hoc compatibility converter.

Preflop input ranges seal as `[P,K]`; postflop input ranges seal as `[A,H]`.
No sequential learner input contains `state_ref`. Folded-summary dependencies
are deduplicated across scheduler packs into an exact auxiliary pool and are
resolved only by bounded training gather.

Sequential leaf completion now has an immutable interface containing owner,
exact log reach, logical completion key, fixed fold payoff, board, natural
range/evaluator inputs, and pot-stratum terms. The sibling evaluator produces
the same keyed posterior/value/effective-sample tensors through this interface
without a tree or mutable state store.

The normal `NormalDataset` implementation and `pbg/core/normal.py` were
deleted. Ordinary PPO unit fixtures are now explicitly loss-only and cannot
be mistaken for rollout storage.

## Ownership and deletion evidence

| Retired artifact | Replacement owner/API | Preserved evidence |
|---|---|---|
| `NormalDataset` and its reach/state-ref builder | normal `RolloutStorage` -> `build_normal_targets` | deterministic schedule-invariance test, one-iteration CPU/GPU training, scalar PPO tests |
| `build_normal_dataset_from_sampled` | `_seal_sequential_rollout_storage` plus authoritative target builder | keyed terminal-reward equality across scheduler batch/admission shapes |
| full sequential `[D,P,H]` decision reach in production | scheduler-time `SealedDecisionInputs` with natural partition ranges | immutable forward identity launch gate and complete-input tests |
| tree/mutable-state completion dependency | `complete_sealed` over exact leaf columns | exact equality with `complete_keyed` for mixed showdown/fold leaves |
| repeated wide-column `torch.cat` sealing | `PagedTensorArena.seal(row_order=...)` | multipage permutation and bounded-scratch tests |
| post-target compatibility conversion for normal/sequential production | direct `TrainingStorage` returned by authoritative builders | owner-inventory test observes no dataset/tree/targets/policy/reach owner and zero poker state/summary page bytes |

The sequential posterior pool contains exactly one row per required
`(trajectory, player)` pair. Each decision retains one `posterior_ref` and one
exact `lambda_return`; no duplicated advantage/value target or production
`Q` row is stored.

## Preservation evidence

- Exact/natural partition domains and complete immutable poker inputs pass for
  preflop and postflop, including sparse folded-summary pool resolution.
- Logical RNG invariance across family pack/admission schedules still produces
  identical private profiles, public paths, old outputs, and normal returns.
- Utility normalization remains applied to normal keyed rewards and sealed RB
  completion values.
- Immutable keyed completion remains delegated to the read-only sibling poker
  evaluator and matches the prior keyed call exactly.
- The bounded distributed learner, fused policy terms, reference-cache logic,
  checkpointing, and diagnostics remain the consumer of the resulting
  `TrainingStorage`; three consecutive CPU sequential-RB iterations and one
  CPU normal iteration are finite.
- No branching mathematics, KKT controller, `$T/C$` head, queue correction,
  or root-refill behavior was changed in this phase.

## Validation

- `./tools/poker-python -m pytest -q tests/test_refactor_storage_targets.py
  tests/test_storage_lifecycle_poker_inputs.py tests/test_p13a_gpu_rollout.py
  tests/test_p13b_integration.py tests/test_p11_losses.py
  tests/test_c11_minimal_losses.py tests/test_p5_completion.py
  tests/test_utility_normalization.py
  tests/test_storage_lifecycle_migration_gates.py` -- `81 passed, 4 xfailed`.
  The four strict expected failures belong to phases 4 and 5.
- After `nvidia-smi` showed all three RTX 2080 Ti devices idle, GPU 0 ran
  `test_real_sibling_execution_poker_tiny_cuda` and
  `test_full_preflop_postflop_policy_inputs_request_no_host_sync` -- `2 passed`.
- A direct temporary-output smoke ran one normal and one sequential-RB
  production iteration on GPU 0 with `R=4`, two roots, 8 bb stacks, tiny model
  width, and four completion samples -- both completed successfully.
- Final `nvidia-smi`: GPU 0 at 9 MiB and 0% utilization, with no compute
  process; GPUs 1 and 2 remained idle.
- `git diff --check` -- clean.

The repository's older GPU acceptance test still names three absent
`configs/acceptance/*.yaml` files and therefore fails before constructing a
bundle. The equivalent phase-3 production smokes above used the current
`configs/nlhe6_h100.yaml` plus the checked-in local hardware profile. Phase 5
must migrate that stale test/config surface rather than masking it.

## Temporary adapters and phase-4 starting conditions

`DeviceTreeBuffer`, `TreeBatch`, `PolicyCache`, and `ReachCache` remain shared
with branched rollout and probe/diagnostic consumers. The sequential production
path uses them only as scheduler-local sources and releases their mutable poker
payload at exact sealing; they never reach target construction or the learner.
They cannot be deleted independently while `pbg/core/branching.py` still owns
the same classes. Phase 4 must migrate branched rollout and then delete these
shared containers and their old minibatch/value consumers.

Likewise, `_RolloutSealInput` and `_seal_iteration_training_storage` now serve
only the branched/probe compatibility slice. The direct normal/sequential
branches bypass the converter. Phase 4 must remove the remaining production
consumer, after which phase 5's static sole-path gate can delete both symbols.

Sequential scheduler topology is still temporarily admitted through the
configured legacy node-capacity guard, and parent-linked history keeps state
rows live until exact sealing rather than recycling each transition. These
allocation surfaces are not learner owners, but they are not the final
last-consumer design. The shared paged occurrence/live-history migration must
land with branched phase 4, and phase 5 must remove `node_budget` and the
transitional forecast/controller fields completely.

Phase 4 starts with both sequential training modes already consuming the sole
authoritative target builders, a tested immutable leaf-completion boundary,
exact natural decision input stores, and normal legacy storage deleted. It
must translate fixed/dynamic branched rollout, protected root refill, ragged
actions, and bounded-frontier targets onto these ownership primitives without
regressing the current scientific equations.
