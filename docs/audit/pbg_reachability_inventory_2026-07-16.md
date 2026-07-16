# Active `pbg/` reachability inventory — 2026-07-16

This inventory records the evidence used for the post-FIFO cleanup.  It was
written before deleting implementation code.  The authoritative runtime roots
are the `pbg-train` and `pbg-benchmark` entry points in `pyproject.toml`; the
accepted active game is configurable NLHE cash poker.  Imports were checked
across `pbg/` and `tests/`, all YAML files were loaded through the strict
configuration schema, and the pre-cleanup CPU suite produced 223 passes and 14
skips.  Its two failures were sandbox-denied Gloo socket creation in the two
CPU distributed tests.

## Dependency map

```text
pbg-train -> cli.train -> core.config -> games.poker.bundle
                         core.training -> core.rollout/core.branching
                                       -> core queue/execution primitives
                                       -> core values/loss/minibatch/tree
                         games.poker.bundle -> engine/cards/ranges/model
                                            -> tensor rollout backend
                                            -> sibling evaluator PBG API

pbg-benchmark -> cli.benchmark -> benchmark -> the same config, bundle,
                                                rollout, value, and learner path

eval.arena -> poker engine/model/evaluator adapters (offline support tooling)
```

The normal, sequential Rao--Blackwell, and branched Rao--Blackwell modes all
reach the same family-scheduled implementation.  The former wavefront routines
are reached only by `rollout.executor` dispatch, executor comparison tooling,
and their regression tests/configurations.

## Classification

### Active production

- `pbg/core/{config,game,rollout,branching,completion,tree,values,normal,
  minibatch,losses,training}.py`: configuration, rollout/value construction,
  and learning.  `posterior_values.py` contains the current implementation but
  is hidden behind a `values.py` compatibility re-export; these will be
  consolidated under `values.py`.
- `pbg/core/{distributed,observability}.py`: the global-mass/gradient law and
  execution measurements.  `distributed.py` is directly exercised by law
  tests; production's equivalent calculations will be kept aligned rather than
  discarded as test-only code.
- The family scheduler, occurrence arena, stable routing, logical RNG, and
  admission primitives currently in `pbg/core/queued.py`.  They are production
  mechanisms, not migration code, and will move to a neutral module/name.
- Every Python module under `pbg/games/poker/` except empty package markers:
  NLHE rules, compact-deck geometry, ranges, model, rollout backend, completion,
  and the read-only sibling evaluator boundary are all reached by bundle
  construction, training, evaluation, or contract tests.

### Active support/tooling

- `pbg/benchmark.py`, `pbg/cli/benchmark.py`: single-implementation performance
  and mode benchmarking remains useful; executor A/B comparison code does not.
- `pbg/eval/arena.py`: accepted offline poker evaluation surface.
- `pbg/core/logging_util.py` and `pbg/cli/train.py`: metrics and launch surfaces.
- `TreeBuilder` and small CPU fixtures in `tree.py`: deterministic construction
  support for value-law tests, not an alternate rollout implementation.

### Historical and worth archiving

- `pbg/experiments/branching.py`, `pbg/cli/branching_experiment.py`, and the
  experiment-controller hooks in legacy branching: useful record of branching
  intensity investigations, but absent from production training dispatch.
- FIFO migration plans, the executor comparison experiment log/results, and
  wavefront benchmark configurations: useful promotion evidence, but no longer
  operational guidance or supported configuration.
- The already archived Kuhn/Leduc/synthetic snapshot.  It remains outside
  `pbg/`, is not selected by package discovery, and is excluded by pytest's
  `testpaths = ["tests"]`.

### Obsolete and safe to delete

- Wavefront sequential and branched rollout bodies, their dispatch constants,
  `rollout.executor`, trajectory-chunk compatibility, executor-comparison CLI,
  benchmark comparison functions, tests, and supported-documentation claims.
- `pbg/core/policy_adapter.py`: no runtime or test import; poker owns the only
  active adapter/output contracts.
- `pbg/core/values.py` as a re-export shim after moving the implementation into
  that canonical module.
- `RootConfig.train`, `RolloutConfig.n_trees`, and `beta_mode` compatibility
  aliases.  Current checkpoints restore tensor/optimizer/RNG state and do not
  parse their saved configuration back into the active schema, so these aliases
  are not needed for checkpoint resume.
- Fixed-family-quantum packing and its selector.  Accepted production uses the
  actual-cost family splitter; retaining a second physical packing path would
  preserve an unsupported alternate implementation.

### Uncertain pending evidence

- No active `pbg/` module remains uncertain.  Historical reference-paper text
  mentioning retired games is third-party source material and is deliberately
  not rewritten.  Raw run/checkpoint directories are runtime artifacts rather
  than package architecture; only promotion evidence selected for archive will
  be committed.

## Planned archive/delete set

Archive the branching experiment and migration/promotion narrative under
`archive/`.  Delete wavefront and executor comparison implementation, obsolete
configuration presets, comparison-only tests, dead adapter protocol code, and
duplicate API shims.  Rename the surviving queue implementation and controls
to production-neutral execution/family terminology.  All active YAML files
will use one strict NLHE-only schema and removed keys will fail as unknown.
