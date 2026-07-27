# Queued-training refactor proposals

This folder defines the approved replacement. Prerequisite authority closure
is recorded in
[`queued_refactor_prerequisite_closure_20260726.md`](../../audit/queued_refactor_prerequisite_closure_20260726.md).
Its
external authorities are [`core_math.tex`](../../theory/core_math.tex),
[`MHA_model.md`](../../theory/MHA_model.md), and
[`folder_impact.md`](../../theory/folder_impact.md).
[`storage_redesign.md`](storage_redesign.md) defines the storage and lifecycle contract; focused proposals refine its
game hooks and implementation plan without overriding it:

- [`storage_redesign.md`](storage_redesign.md): fixed-address queued storage and lifecycle.
- [`poker_branched_rb_sample.md`](poker_branched_rb_sample.md): poker branched-RB implementation of `sample`.
- [`poker_folded_statistics.md`](poker_folded_statistics.md): folded-state inputs, outputs, and refresh.
- [`poker_policy_cuda_operator.md`](poker_policy_cuda_operator.md): canonical differentiable policy composition.
- [`layered_target_creation.md`](layered_target_creation.md): sole layered CUDA target path.
- [`features_to_preserve.md`](features_to_preserve.md): residual current
  capabilities not specified by the other proposals.
- [`implementation_plan.md`](implementation_plan.md): implementation, cleanup, and cutover.

Supporting audit evidence:
[`replica_learner_parity_and_rb_determinism.md`](../../audit/replica_learner_parity_and_rb_determinism.md) and
[`nlhe_hu_32bb_family_timing_attribution_20260724.md`](../../audit/nlhe_hu_32bb_family_timing_attribution_20260724.md).

## Proposed repository boundary

`ai-game-learning` owns:

- `sample`: posterior/profile draws, action/chance labels, and topology packets;
- `commit`: routing, materialization, retained writes, and endpoint patches;
- `FoldedRanges`, folded statistics, policy composition, RNG, and chance-card generation.

`poker-gpu-evaluator` owns the exact `Leaf -> LayerWork` showdown and completion operations involving `hand_rank`.
This proposal supersedes current ownership notes after the required
repository-instruction and evaluator-interface changes are approved and
landed.

## Configuration and calibration

One common run-level `rollout.beta_max` bounds fixed and adaptive ordinary-node intensity. Ordinary player and chance
degrees share `N_max = ceil(beta_max)`; no second degree-bound hyperparameter exists.
Capacities are fixed run hyperparameters. Hardware profiles tune launch shapes, not algorithms.
`n_replica` adds independent model lanes with shared schema, capacities, estimator, and schedule; only weights, loss
coefficients, and schema-identical fixed/KKT controller choice and state may differ.
Initial tuning loads existing checkpoints and captures short RTX 2080 Ti trajectories for `P in {2,6}`;
experimental outputs belong under `runs/experiments/`.
Experiments select one implementation; production deletes every alternative.

## Deferred

- Duplicate-node computation reuse. Duplicate sampled children always keep distinct logical identities and topology
  edges.

The design, repository ownership, evaluator boundary, capacity vocabulary, and
matched-work CUDA gates are approved. Implementation follows the phased
protocol in `implementation_plan.md`. `docs/theory/pseudocode/` is historical
design evidence, not an authority or implementation gate for this rewrite.
