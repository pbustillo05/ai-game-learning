# Phase 6 handoff: full acceptance and release verdict

Parent phase commit: `dce75ff` (`refactor(core): retire legacy rollout
capacity`). Phase 6 is committed as
`test(core): validate sole memory-safe training pipeline`; its hash is recorded
by the final ordered index after this commit exists.

Pre-existing unrelated worktree change: the user-edited
`docs/todo/restore_authoritative_storage_lifecycle_20260721.md`. It remained
unstaged throughout all six phases.

## Release verdict

Accepted. Production normal, sequential Rao--Blackwell (RB), fixed branched
RB, and dynamic branched RB now have one data plane:

```text
byte-admitted paged/recyclable rollout owners
  -> exact RolloutStorage
  -> bounded streamed targets
  -> immutable TrainingStorage
  -> bounded physical learner subspans
```

There is no post-hoc legacy conversion or production-capable parallel
implementation. `TrainingStorage` validation, exact sealing, sparse poker
summary allocation, and the four-phase forecast now use the same normative
mode schemas. The last sealing-time payload `torch.cat` was replaced by one
exact address destination populated by stable spans.

## Deleted implementation inventory

Relative to baseline `1cde6de`, the migration deletes these files:

- `pbg/core/values.py` and its padded `NodeTargets`, growing frontier, and
  `streamed_branched_value` implementation;
- `pbg/core/minibatch.py` and its tree/reach minibatch gathers;
- `pbg/core/normal.py` and its separate `NormalDataset` store;
- retired implementation-only test files
  `test_c10_posterior_moments.py`, `test_c11_minimal_losses.py`,
  `test_p10_values.py`, and `test_p11_losses.py`.

Deleted classes/surfaces inside retained modules include `DeviceTreeBuffer`,
`PolicyCache`, `ReachCache`, `DecisionReachCache`, the full-decision-reach
facade, `_RolloutSealInput`, `_seal_iteration_training_storage`, padded
`old_logp_child`, `configured_node_budget`, and the production
`node_budget`/`arena_chunk_nodes` configuration and checkpoint identities.
Across the six-phase diff from `1cde6de` (excluding the user's todo edit),
8,055 lines were added and 5,835 removed; the additions are the paged poker
owners, direct integrations, schemas, lifecycle tests, and reconciled
authorities rather than a retained duplicate implementation.

No user-facing capability was removed. Normal and both RB trainers, benchmark
and probe APIs, sequential/branched variance CLIs, branching sweep/resume,
checkpoint initialization/resume, MPS inference/export, compact-deck poker,
and the sibling evaluator contract remain active.

## Final ownership map

| Phase | Sole owners | Released at boundary |
|---|---|---|
| Rollout | paged occurrence/topology/control/input/action/leaf sources; recyclable state and active reach; refcounted history; sparse summary anchors; bounded queues | completed-node state/reach immediately; all mutable backend owners and queues after seal |
| Seal | source pages plus exact topology/input/decision/action/leaf destinations and bounded scratch under one transaction | every source page after its last stable-span copy |
| Targets | exact `RolloutStorage`, exact target destinations, bounded completion batches, one preplanned reusable branched frontier | topology, leaves, completion results, frontier, and target scratch after finalization |
| Optimization | immutable `TrainingStorage`, optional admitted reference cache, one index vector, one bounded physical subspan, one flat fp32 gradient SUM buffer | iteration store and cache at iteration end |

Normal stores scalar selected action/value/return data. Sequential RB stores
natural old outputs, exact returns, and deduplicated exact posterior rows.
Branched RB stores ragged old-action and posterior/advantage rows, exact value
targets, nominal/executed counts, and current $T/C$ targets/masks. It never
stores a production $Q$ target.

## CPU acceptance

- Current full feasible suite after the final telemetry/soak diff:
  `CUDA_VISIBLE_DEVICES='' ./tools/poker-python -m pytest -q` -- 454 passed,
  32 platform skips, and three sandbox-only Gloo failures after 105.89 s. All
  failures were local socket creation denied with `Operation not permitted`.
- Distributed rerun outside that socket sandbox:
  `CUDA_VISIBLE_DEVICES='' ./tools/poker-python -m pytest -q
  tests/test_p13b_ddp.py -x` -- 4 passed in 7.99 s. The three failed cases
  from the full run all pass, for 457 distinct passing tests and 32 skips.
- Final schema/owner/shape impact set:
  `tests/test_p13b_integration.py::test_iteration_seals_the_sole_training_owner_and_releases_poker_pages
  tests/test_aud26_no_host_sync.py tests/test_phase3_migration.py` -- 6 passed,
  2 CUDA skips.
- Phase-6 focused lifecycle set -- 102 passed, 9 skips. Phase-5 focused and
  preservation sets were 122 passed/7 skips and 105 passed/11 skips.
- Static gates find no retired production symbols or production
  `node_budget`; payload `torch.cat` is absent from rollout, branching, and
  target construction.

## CUDA acceptance

Hardware: NVIDIA GeForce RTX 2080 Ti, driver 580.173.02, CUDA 13.0. The final
single-device runs used physical GPU 1 only. GPU 0 acquired another user's
426 MiB compute process before the final rerun and was not touched; GPU 2's
desktop processes were also left alone.

- Complete checked-in GPU acceptance on GPU 1:
  `CUDA_VISIBLE_DEVICES=1 ./tools/poker-python -m pytest -q
  tests/test_refactor_gpu_acceptance.py -s -x` -- 9 passed and the two two-GPU
  cases skipped because only one GPU was exposed, in 15.56 s. This includes one
  update in every mode, three-iteration normal/sequential owner-stability
  soaks, a two-iteration fixed root-refill soak, and a two-iteration dynamic
  queue-safeguard soak with observed child suppression. It also proves the
  same tiny workload succeeds when `torch.cuda.empty_cache` is a no-op.
- Full-deck/device/evaluator checks on GPU 1:
  `test_poker_actual_cost_keeps_physical_pack_indices_on_device`,
  `test_real_sibling_execution_poker_tiny_cuda`, and
  `test_cuda_outputs_and_sibling_inputs_remain_on_original_device` -- 3 passed
  in 4.30 s.
- Consolidated multi-GPU acceptance on physical GPUs 1 and 2:
  `CUDA_VISIBLE_DEVICES=1,2 ./tools/poker-python -m pytest -q
  tests/test_distributed_learner.py::test_two_cuda_unequal_strata_exact_sum
  tests/test_refactor_gpu_acceptance.py::test_tiny_two_gpu_common_width_and_collective_update
  tests/test_refactor_gpu_acceptance.py::test_tiny_two_gpu_dynamic_branched_common_controller`
  -- 4 passed in 22.94 s. This covers unequal-mass normal and sequential-RB
  learner updates, common sequential width, and a common dynamic KKT/safeguard
  update.
- Final `nvidia-smi` showed GPUs 0/1/2 at 6/6/16 MiB, all P8 and 0%
  utilization, with no validation process. GPU 0's unrelated 426 MiB process
  ended independently after the tests; it was never touched by this work.

## Tiny-run phase ledger

All byte values below are current Phase-6 schema priors and first-iteration
CUDA allocated peaks. The structural prior deliberately excludes the CUDA
context, allocator cache, and opaque library workspaces; measured allocated,
reserved, and device-used values are reported separately. Branched rollout
and seal priors include the configured 1 GiB hard execution ceiling, so they
are bounds rather than expected allocation.

| Mode | Prior rollout / seal / target / opt (bytes) | Allocated rollout / seal / target / opt (bytes) | One-update reserved peak | Retained training bytes |
|---|---:|---:|---:|---:|
| normal | 249,816 / 329,976 / 551,304 / 674,980 | 9,600,512 / 8,766,976 / 8,739,328 / 18,655,744 | 23,068,672 | 1,860 |
| sequential RB | 250,080 / 368,250 / 632,338 / 747,772 | 18,122,240 / 17,294,848 / 17,295,872 / 18,660,352 | 23,068,672 | 4,028 |
| fixed branched RB | 1,073,883,040 / 1,073,926,296 / 642,516 / 739,308 | 17,740,800 / 17,327,616 / 17,323,008 / 18,712,064 | 23,068,672 | 2,531 |
| dynamic branched RB | 1,073,883,040 / 1,073,926,296 / 642,516 / 739,308 | 17,746,944 / 17,329,664 / 17,325,056 / 18,714,112 | 23,068,672 | 2,531 |

The hardware-profile safe ceiling was 9,199,091,712 bytes. One-update measured
reserved headroom was therefore at least 9,176,023,040 bytes. No run approached
OOM.

The first-iteration exact seal ledgers were:

| Mode | Source capacity | Destination | Scratch | Source pages |
|---|---:|---:|---:|---:|
| normal | 22,016 | 5,496 | 3,896 | 3 |
| sequential RB | 22,280 | 5,562 | 3,962 | 3 |
| fixed/dynamic branched RB | 22,408 | 2,869 | 1,997 | 4 |

Each run used one two-row physical learner subspan and no physical tail. Tiny
final acceptance decision throughput was 2.24/s normal, 10.88/s sequential RB,
4.11/s fixed branched, and 4.81/s dynamic branched. These are cold-start
acceptance measurements, not an equal-semantics performance comparison. The
2026-07-17 throughput records used the retired capacity-shaped rollout path and
larger full-deck workloads, so no honest before/after throughput ratio exists.

## Poker domains and evaluator boundary

The real $R=13$ all-mode owner test now asserts complete production storage:
preflop natural inputs and old RB outputs are width 169; postflop natural
inputs/outputs are width 1326; sequential posterior/return and branched
value/advantage targets are exact width 1326. The compact $R=4$ CUDA acceptance
reported the corresponding 16/120 widths. Impossible-hand, board-blocking,
live/folded-seat, dtype/layout, and original-device tests pass. Posterior,
showdown, and folded-summary computation still delegate to the read-only
sibling evaluator.

## Scientific preservation

Direct tests remain green for fixed/dynamic branching, deterministic KKT rank
behavior and checkpoint state, positive public $T/C$ heads and targets,
nominal/executed queue correction, duplicate-preserving labeled children,
rank-addressed RNG, $\lambda=0,1$, zero posterior mass, bounded frontier
contraction, and protected-root refill including nested descendants. Allocation
admits the complete already-selected child wave before posterior/action draws;
failure rolls every page owner back, performs no draw, and releases poker state,
history, reach, and summary owners. It never changes scientific offspring,
retries, or rerolls an iteration.

## Documentation and limitations

The three changed TeX authorities compile with `latexmk -pdf` into `/tmp`:
`storage_lifecycle.tex` (10 pages), `family_scheduled_rollout.tex`, and
`backward_targets.tex`. Current architecture/training/file-reference docs use
the sole-path vocabulary. The 2026-07-17 acceptance and performance notes are
marked historical rather than having their measurements rewritten.

Narrow limitations:

- allocator-reserved peaks include CUDA context/cache and opaque library
  workspace that are calibrated empirically rather than predicted by tensor
  schemas; correctness and the hard stop do not depend on `empty_cache()`;
- no equal-hardware, equal-sampled-object full-deck throughput baseline exists,
  so only current acceptance throughput is reported;
- the multi-GPU gate used two idle 2080 Ti devices and deliberately did not
  include a large throughput or memory-pressure run.

These are evidence limitations, not alternate implementations or known
correctness failures.
