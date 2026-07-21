# Phase 1 handoff: authority, characterization, and deletion plan

Parent baseline: `1cde6de` (`Pre-refactor fix commit`).

Pre-existing unrelated worktree change: the user-edited
`docs/todo/restore_authoritative_storage_lifecycle_20260721.md`. It is the
master prompt for this goal and is intentionally excluded from this phase
commit.

## Authority reconciliation

`docs/theory/core_math.tex` remains authoritative for the sampled object and
estimators. The implementation pseudocode now treats fixed/dynamic branching,
nominal and executed offspring counts, queue-safeguard correction, public
$T/C$ heads, and protected-cohort root refill as production requirements.

The reconciled storage contract makes these distinctions explicit:

- final topology spans use executed degree $N$ and final root degree $N_+$;
- decision control rows carry frozen $T/C$ predictions and $(\widetilde N,N)$
  only through their last controller/target consumer;
- the bounded branched frontier adds transient vector-value, cost, and
  $\Delta$ rows without creating an all-node posterior/reach memo;
- root metadata is two scalars, $N_0(\varnothing)$ and $|\mathfrak S_0|$;
- the root action span covers only the protected prefix, while refill-subtree
  decisions remain learning rows and final path masses use $N_+$;
- immutable training storage retains only $T/C$ targets, the trace mask, and
  count data actually consumed by training or diagnostics; no $Q$, $\Delta$,
  queue forecast, or KKT path factor survives.

There is no conflict between current mathematics and the lifecycle authority.
The mismatch was missing schema/algorithm detail in the pseudocode.

## Characterized production call graph

The phase-1 production path is still the known invalid compatibility path:

```text
_one_iteration
  -> _build_iteration_data
     -> sample_trees[_sequential]
     -> NormalDataset or streamed_branched_value -> NodeTargets
  -> _seal_iteration_training_storage
     -> post-hoc TrainingStorage (still containing state_ref inputs)
  -> DistributedFixedRolloutUpdate
```

`tests/test_storage_lifecycle_migration_gates.py` encodes the desired direct
call graph, immutable-input boundary, retired-symbol inventory, `node_budget`
removal, and bounded-frontier rule as strict expected failures. Each migration
phase must remove the relevant marker when it makes the gate pass; strict
XPASS prevents silently leaving completed work marked as pending.

## Artifact migration and deletion table

| Retired artifact | Current production consumers | New owner/API | Preserved behavior and proof | Deletion evidence required |
|---|---|---|---|---|
| `DeviceTreeBuffer` / capacity-backed `TreeBatch` | `pbg/core/branching.py`, `pbg/core/rollout.py` | paged occurrence/topology source, then `SequentialDecisionCSR` or `BranchedTreeSpan` | labeled duplicate children and vectorized topology checks in `test_c09_labeled_branching.py`, `test_p8_tree.py`, and new topology tests | no production import/symbol; direct rollout returns sealed topology |
| `PolicyCache` | branching, rollout, minibatch, training | partition decision/action source pages and sealed rows | old-policy identity, natural widths, $T/C$ heads, fused learner calculations | no class/import and no padded `[D,E_max,H]` owner |
| `ReachCache` | branching, rollout, normal, minibatch, training | exact leaf rows plus live recyclable reach slots | evaluator contract and posterior equality tests | no class/import and no all-node leaf/reach replay |
| `DecisionReachCache` | branching, rollout, normal, minibatch, training | natural `PartitionInputTable` written at decision creation | 169/1326 shape tests, impossible-hand behavior, learner forward equality | no class/import, no `[D,P,H]`, no optimizer `state_ref` |
| `NormalDataset` | normal builder and training | normal `RolloutStorage` plus `build_normal_targets` | direct $\lambda$-return equality and scalar normal loss | `pbg/core/normal.py` retired or reduced to non-owning pure helpers |
| `NodeTargets` and `streamed_branched_value` | minibatch, training | `pbg/core/targets.py` bounded builders and ragged action rows | all current value, $T/C$, queue correction, and refill formula tests ported | no production import/symbol; no padded action axis or growing frontier |
| `_RolloutSealInput` and `_seal_iteration_training_storage` | training orchestrator | direct `RolloutStorage -> build_*_targets -> TrainingStorage` | one-iteration owner inventory for all modes | no symbol and call-graph gate passes |
| old tree/reach minibatch gathers | `pbg/core/minibatch.py`, diagnostics | bounded `TrainingStorage` gather and read-only diagnostics view | learner split/cache/one-row-tail tests | no production consumer or duplicate sampler |
| capacity-shaped `PokerTensorStateStore` | poker rollout backend | recyclable live state/history slots plus immutable decision/leaf payloads | engine/history/evaluator and production shape tests | no allocation proportional to realized-total/node capacity |
| `DevicePartitionSummaryCache` capacity validity/fields | poker rollout backend | sparse immutable dependency-anchor pages | fold/street refresh and zero-allocation at $A=P$ | no capacity-wide validity or summary column |
| `_analytical_memory_plan` and capacity controller adapters | training and bundle preflight | schema/page four-phase ledger and observations | controller/failure tests plus allocated/reserved/NVML reconciliation | no handwritten node-count forecast |
| `RolloutConfig.node_budget` / `arena_chunk_nodes` | config, rollout, bundle, tools, tests, YAML | `storage_page_rows`, byte admission, path bound, and scientific controller | config/checkpoint migration and no-reroll tests | static search clean outside historical prose/migration parser |

No retired production implementation will be copied into `archive/`.

## Preservation ledger

| Source change | Behavior to preserve on the sole path | Existing proof to port or retain | Acceptance phase |
|---|---|---|---|
| `04bd089`, `86756b8`, `8d94724` | partitions, exact/natural domains, schemas, compact topology, target builders, immutable storage, bounded distributed learner, sequential controller | `test_refactor_storage_targets.py`, `test_execution_foundations.py`, `test_distributed_learner.py`, `test_memory_control.py`, `test_phase3_migration.py` | 2-5 |
| `537700f` | mass/variance/cost allocation, fused policy math, optional reference caches, stable physical plans/tails, timing, SDPA/query chunking, trained initialization, hardware row/token limits | `test_distributed_learner.py`, `test_aud32_performance_gates.py`, model/initialization tests and phase-6 timing | 3-6 |
| `567d913`, `81f4cdd` | stale-config removal and generated-artifact hygiene | repository inventory and final staged-diff inspection | 5-6 |
| `6fc518c` | MPS inference/export/checkpoint artifact contract | `test_inference_api.py` | 5-6 |
| `818aa17` | duplicate-preserving labeled children, logical identity, sweep/plot/resume tools | `test_c09_labeled_branching.py` and migrated branching-sweep smoke | 4-5 |
| `2429ab6` | evaluator v1 compact-deck layouts, posterior/equity/showdown/fold behavior, CPU/CUDA selection | compact-deck, completion, folded, and GPU integration tests | 2-6 |
| `b94941c` | vectorized CUDA topology validation without per-row host reads | migrated topology validation tests and `test_aud26_no_host_sync.py` | 2-4 |
| `79a354b` | explicit branching modes, KKT controller state/warmup/clipping, $T/C$ heads/losses, nominal/executed correction, no private tilting | `test_adaptive_kkt_branching.py`, model shape tests, new builder equality tests | 4 |
| `3f594c7`, `8573a84` | current variance theory and sequential/branched variance CLIs/calibration/plots/diagnostics | `test_rao_blackwell_variance*.py`, `test_branched_rao_blackwell_variance.py` | 4-6 |
| `1cde6de` | protected root refill, `branching_min_nodes`, utility normalization/conversion, initialization provenance, entropy horizon, optimizer diagnostics, normalized configs | adaptive KKT/refill, utility, config, learner, diagnostics, and performance-gate suites | 2-6 |

The feature/config/CLI inventory at `1cde6de` is the deletion baseline. No
user-facing capability is approved for removal; only internal superseded
owners are scheduled for deletion.

## Phase result

New ownership boundaries: documentation only; no production owner changed in
phase 1. Deleted files/classes: none. Temporary adapters: all legacy artifacts
in the migration table remain and are explicitly gated.

Focused checks:

- `./tools/poker-python -m pytest -q tests/test_storage_lifecycle_migration_gates.py`
  — `5 xfailed` (the five strict, phase-owned migration gates), exit 0.
- `./tools/poker-python -m pytest -q tests/test_refactor_storage_targets.py tests/test_adaptive_kkt_branching.py`
  — `29 passed`.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error` for
  `storage_lifecycle.tex`, `family_scheduled_rollout.tex`, and
  `backward_targets.tex`, with separate output directories below
  `/tmp/storage-lifecycle-phase1/` — all three compiled successfully.
- `git diff --check` — clean.

Next phase starts from this commit and must add the byte-admitted page/live-slot
foundation without changing the production sampled object. It should remove
the immutable-poker-input expected failure only when `state_ref` has genuinely
left the retained input schema; the direct-orchestrator and retired-symbol
gates remain pending until later phases.
