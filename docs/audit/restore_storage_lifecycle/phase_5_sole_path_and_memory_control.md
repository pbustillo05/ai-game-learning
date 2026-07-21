# Phase 5 handoff: sole path, configuration, and memory control

Parent phase commit: `759cde6` (`refactor(core): migrate dynamic branched
storage`).

Pre-existing unrelated worktree change: the user-edited
`docs/todo/restore_authoritative_storage_lifecycle_20260721.md`. It remains the
master prompt and is intentionally excluded from this phase commit.

## Completed requirements

All three production modes now use one ownership sequence:

```text
recyclable live state/reach
  -> paged occurrence/input/action/leaf rows
  -> exact RolloutStorage
  -> bounded target builder
  -> immutable TrainingStorage
  -> bounded learner subspans
```

- Sequential and branched schedulers append payload rows directly to paged
  arenas. Exact sealing is a byte-admitted source/destination/scratch
  transaction; neither scheduler accumulates wide tensor lists for a later
  concatenation.
- Poker public-state rows are recyclable. History is a separate refcounted
  persistent pool, dead ancestor chains are reclaimed recursively, and both
  owners have zero allocated pages after the rollout boundary.
- Decision inputs and folded-summary dependencies are immutable natural-width
  rows captured at their last state/reach consumer. Variance and best-response
  probes use those rows and immutable leaf payloads rather than state handles.
- Branched topology, control, reach, state, history, occurrence, input, action,
  and leaf page growth is charged to one hard byte accountant. Complete child
  waves are admitted transactionally before posterior/action draws or topology
  mutation; rollback leaves every owner unchanged. Cutoff and admission-error
  exits unconditionally release mutable backend owners.
- `node_budget` is gone from production Python, YAML, benchmark identity,
  checkpoint identity, CLI flags, and tests. `storage_page_rows` controls only
  allocation granularity; `execution_memory_bytes` is a byte ceiling, not a
  replacement row limit. Checkpoint schema v5 removes the retired field and
  records that transition without reinterpreting its value.
- The memory prior is generated from real input, topology, decision, action,
  leaf, target, and training schemas. It reports separate rollout, seal,
  target, and optimization fixed/variable peaks. Tests force each phase to be
  the independent sequential-width limiter.
- The active configuration, file reference, poker ownership, and training
  documents now describe the sole paged path. Missing historical
  `configs/acceptance` presets were not resurrected; GPU acceptance constructs
  tiny mode configurations from the current checked-in template.

## Deleted artifacts and replacement owners

| Deleted artifact | Sole replacement | Preserved behavior/evidence |
|---|---|---|
| `pbg/core/values.py`, padded `NodeTargets`, growing value frontier | `pbg/core/targets.py` exact destinations and prebounded branched frontier | direct fixed/dynamic, queue-correction, zero-mass, and protected-root target tests |
| `pbg/core/minibatch.py` tree/reach gathers | `TrainingStorage` index draw and bounded learner gather | distributed learner, cache rollback, physical-tail, and global-mass tests |
| `DeviceTreeBuffer`, legacy cache/reach classes, post-hoc seal input/converter | `PagedTreeSource`, `PagedRowAccumulator`, `SealedDecisionInputs`, direct target builders | sole-path source gates and real all-mode iteration owner tests |
| full decision-reach fallback and compact/expand facade | scheduler-time natural input capture plus exact input pools | complete-input forward identity and 169/1326 domain tests |
| legacy standalone padded PPO/RB-PPO fixture functions and tests | streamed learner loss implementation plus small loss-name/mode registry | learner loss/gradient/reference-cache tests and production iterations |
| policy-conditioned legacy peak forecaster | schema-owned four-phase forecaster and memory-only sequential controller | independent phase limits, surprise contraction, delayed growth, rank-minimum, and checkpoint equality tests |
| obsolete value/minibatch/loss-only tests | lifecycle, target, learner, and integration tests on the production owners | static deletion gates plus focused suites below |

The mutable tree-based completion facade remains only as a direct evaluator
contract used by evaluator-focused tests. Training, probes, variance tools,
benchmarks, and CLIs do not use it to retain rollout state; their production
path is the immutable leaf/decision interface.

## Preservation evidence

- Fixed/dynamic branching, KKT controller resume, public `T/C` heads,
  nominal/executed queue correction, duplicate children, rank-addressed RNG,
  and protected root refill remain on the new target path.
- Utility conversion, initialization provenance/conversion, explicit entropy
  horizon, gradient diagnostics, periodic reference policy, fused attention,
  natural-width reference caches, and flat fp32 gradient SUM remain unchanged
  downstream of `TrainingStorage`.
- Poker completion still delegates compact-deck posterior, showdown, and
  folded-summary work to the read-only sibling evaluator. Immutable completion
  consumes one sealed `range_weights` owner and preserves keyed sampling.
- Variance CLIs, branching sweep/resume, best-response probe, benchmark modes,
  MPS inference/export, and current experiment YAML remain present and use the
  migrated interfaces.

## Validation

- `CUDA_VISIBLE_DEVICES='' ./tools/poker-python -m pytest -q
  tests/test_storage_lifecycle_migration_gates.py
  tests/test_refactor_storage_targets.py tests/test_p8_tree.py
  tests/test_execution_foundations.py tests/test_c09_labeled_branching.py
  tests/test_adaptive_kkt_branching.py tests/test_p13a_gpu_rollout.py
  tests/test_p13b_integration.py tests/test_storage_lifecycle_poker_inputs.py
  tests/test_phase3_migration.py -x` -- `122 passed, 7 skipped` after the
  schema-partition and immutable-completion integration fixes.
- `CUDA_VISIBLE_DEVICES='' ./tools/poker-python -m pytest -q
  tests/test_p0_config.py tests/test_adaptive_work_control.py
  tests/test_refactor_memory_width.py tests/test_br_probe.py
  tests/test_branched_rao_blackwell_variance.py
  tests/test_rao_blackwell_variance.py tests/test_rao_blackwell_variance_cli.py
  tests/test_distributed_learner.py tests/test_inference_api.py
  tests/test_refactor_gpu_acceptance.py tests/test_utility_normalization.py -x`
  -- `105 passed, 11 skipped` (CUDA/MPS unavailable or hidden), with 18
  third-party TorchScript deprecation warnings.
- The real poker 4,000-byte child-wave test reaches one policy forward, fails
  admission before any posterior draw, and leaves state, history, and summary
  allocated bytes at zero.
- Static searches find retired names only inside the source-gate's forbidden
  string set or historical Markdown, and `torch.cat` is absent from rollout,
  branching, and target payload construction.

## Phase-6 starting conditions

There is no temporary production compatibility adapter remaining. Phase 6
starts from the sole direct data plane and must run the full feasible CPU suite,
distributed CPU tests, staged normal/sequential/fixed/dynamic CUDA acceptance,
bounded soaks, memory/throughput measurements, TeX compilation into `/tmp`, and
final idle-GPU verification. It must then add the final acceptance audit and
ordered phase-commit index, mark superseded 2026-07-17 readiness claims as
historical, and reconcile measured phase ledgers with the schema prior.
