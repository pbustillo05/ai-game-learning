# Queued-training refactor proposals

This folder designs the replacement before theory or production code changes. Current focused proposals are:

- [`storage_redesign.md`](storage_redesign.md): fixed-address queued storage and lifecycle.
- [`poker_rejection_chance_closure.md`](poker_rejection_chance_closure.md): branched-RB sampling and closure.
- [`poker_folded_statistics.md`](poker_folded_statistics.md): folded-state inputs, outputs, and refresh.
- [`poker_policy_cuda_operator.md`](poker_policy_cuda_operator.md): canonical differentiable policy composition.
- [`storage_redesign_refactorability.md`](storage_redesign_refactorability.md): migration analysis.

## Proposed repository boundary

`ai-game-learning` owns rollout rejection sampling, `FoldedRanges`, folded-stat construction, policy composition, RNG,
chance-card generation, routing, and materialization. `poker-gpu-evaluator` owns operations involving `hand_rank`,
including showdown and completion. This supersedes current repository and theory ownership notes; those remain
unchanged until the design is approved.

## Configuration and calibration

One common run-level `rollout.beta_max` bounds fixed and adaptive ordinary-node intensity. Ordinary player and chance
degrees share `N_max = ceil(beta_max)`; no second degree-bound hyperparameter exists.
Hardware-selected warp counts and chance-closure variants belong in `configs/hardware/`, with their exact schema
deferred to kernel tuning. Initial tuning loads existing checkpoints, generates short RTX 2080 Ti trajectories for
`P in {2,6}`, and captures their actual work; experimental outputs belong under `runs/experiments/`.

## Deferred

- Standalone prototype versus full queued-storage implementation scope.
- General duplicate-node computation and storage reuse.
- Authoritative theory reconciliation for originating-profile reuse.

The workflow is: finish the design here, approve and update theory/ownership contracts, then implement, test,
benchmark, select hardware tables, and delete superseded production paths.
