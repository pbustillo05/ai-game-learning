# Queued-training refactor proposals

This folder designs the replacement before theory or production code changes.
[`storage_redesign.md`](storage_redesign.md) defines the storage and lifecycle contract; focused proposals refine its
game hooks and implementation plan without overriding it:

- [`storage_redesign.md`](storage_redesign.md): fixed-address queued storage and lifecycle.
- [`poker_branched_rb_sample.md`](poker_branched_rb_sample.md): poker branched-RB implementation of `sample`.
- [`poker_folded_statistics.md`](poker_folded_statistics.md): folded-state inputs, outputs, and refresh.
- [`poker_policy_cuda_operator.md`](poker_policy_cuda_operator.md): canonical differentiable policy composition.
- [`storage_redesign_refactorability.md`](storage_redesign_refactorability.md): implementation, cleanup, and cutover.

## Proposed repository boundary

`ai-game-learning` owns:

- `sample`: posterior/profile draws, action/chance labels, and topology packets;
- `commit`: routing, materialization, retained writes, and endpoint patches;
- `FoldedRanges`, folded statistics, policy composition, RNG, and chance-card generation.

`poker-gpu-evaluator` owns the exact `Leaf -> LayerWork` showdown and completion operations involving `hand_rank`.
This proposal supersedes current ownership notes after the required theory and interface changes are approved.

## Configuration and calibration

One common run-level `rollout.beta_max` bounds fixed and adaptive ordinary-node intensity. Ordinary player and chance
degrees share `N_max = ceil(beta_max)`; no second degree-bound hyperparameter exists.
Hardware-selected warp counts and chance-closure variants belong in `configs/hardware/`, with their exact schema
deferred to kernel tuning. Initial tuning loads existing checkpoints, generates short RTX 2080 Ti trajectories for
`P in {2,6}`, and captures their actual work; experimental outputs belong under `runs/experiments/`.

## Deferred

- Standalone prototype versus full queued-storage implementation scope.
- Duplicate-node computation reuse. Duplicate sampled children always keep distinct logical identities and topology
  edges.

The workflow is: finish the design here, approve and update theory/ownership contracts, then implement, test,
benchmark, select hardware tables, and delete superseded production paths.
