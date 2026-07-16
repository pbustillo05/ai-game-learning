# Active `pbg/` reachability inventory — 2026-07-16

This inventory was established before deletion and checked again after cleanup.
Evidence came from package imports, `pyproject.toml` entry points, strict loading
of every active YAML file, runtime dispatch, test imports, and the broad CPU
suite. The only active game is configurable NLHE cash poker.

## Dependency map

```text
pbg-train
  -> pbg.config -> core config parser + poker validation
  -> pbg.runtime -> poker bundle + core training
       -> core rollout/branching -> core execution/tree
       -> core completion/values/normal/minibatch/losses/distributed
       -> poker tensor backend/model -> sibling evaluator PBG API

pbg-benchmark -> the same config, bundle, rollout, value, and learner path
pbg.eval.arena -> poker engine/model/evaluator adapters (offline support)
```

`pbg.core` has no static game/evaluation import. The top-level composition
modules own the only core-to-poker connection. Normal, sequential RB, and
branched RB reach the same family scheduler; there is no runtime implementation
selector.

## Classification and disposition

### Active production

- `pbg/config.py` and `pbg/runtime.py`: strict NLHE composition roots.
- `pbg/core/config.py`, `game.py`, `execution.py`, `rollout.py`,
  `branching.py`, `completion.py`, `tree.py`, `values.py`, `normal.py`,
  `minibatch.py`, `losses.py`, `distributed.py`, `observability.py`, and
  `training.py`: the single rollout/value/learner implementation.
- Every nonempty module under `pbg/games/poker/`: NLHE rules, compact deck,
  ranges, model, tensor rollout, completion, and evaluator boundary.

### Active support/tooling

- `pbg/benchmark.py` and its CLI: same-implementation profiling across modes.
- `pbg/eval/arena.py`: offline duplicate-deal evaluation.
- `pbg/core/logging_util.py`, the training CLI, and `tools/poker-python`.
- `TreeBuilder` and CPU fixtures used for deterministic target-law tests.

### Historical and archived

- The branching-controller experiment and CLI.
- Scheduler migration plans, promotion comparisons, raw benchmark JSON, and
  one-off launch scripts under `archive/family_scheduler_promotion/`.
- Pre-production audit generations under `archive/audits/`.
- Kuhn, Leduc, and synthetic implementations under `archive/synthetic_games/`.
  Archived Python is outside package discovery and CI test paths.

### Obsolete and deleted

- Sequential and branched wavefront implementations, executor dispatch,
  comparison CLI/benchmark code, configs, shims, and regression-only tests.
- Fixed-family-quantum packing and its configuration selector.
- `pbg/core/policy_adapter.py`; poker owns the sole active adapter contract.
- The `posterior_values.py`/`values.py` compatibility split and the test-only
  full-completion value reducer; production has one streamed reducer.
- The unused auxiliary value channel, keyed completion protocol, normal rollout
  wrapper, equal-rank diagnostic, and binomial raise-head ablation.
- Game factory/selector machinery and all active retired-game surfaces.
- `RootConfig.train`, `RolloutConfig.n_trees`, `beta_fix`, `beta_mode`,
  `root_states`, FIFO/version names, queue-packing flags, and trajectory chunks.

### Compatibility retained deliberately

Checkpoint resume still normalizes CUDA RNG byte tensors to CPU before calling
the CUDA RNG setter. Relevant NLHE checkpoints restore tensors, optimizer,
scaler, and explicit RNG payloads directly; their saved configuration is
compared as data, not reparsed through YAML. No general legacy-config loader is
therefore retained.

## Uncertain areas

No active `pbg/` module remains unclassified. Third-party reference-paper text
and historical archive content intentionally retain old game and migration
terminology. Existing `runs/` directories are runtime artifacts rather than
source architecture and were not rewritten by this cleanup.
