# Future probe-only branching

This note records design changes for a possible asymmetric BR estimator. It is
not part of the sequential BR-probe implementation.

## Estimator boundary

- Add a new estimator mode; do not overload `branched_rao_blackwell`.
- Sample exactly one frozen-opponent action and one public chance transition.
- Apply branching only when the current owner is the trajectory's probe seat.
- Treat the root by that same role-aware rule. This differs from the current
  branched-tree convention and therefore requires an explicit theory and
  terminology update before implementation.

## Probe branches

- Decide whether branches represent physical legal event IDs or latent
  fold/call/raise-mode proposals mapped onto the legal raise grid.
- If only a subset is expanded, define a schedule-independent proposal law and
  retain its inclusion probability on every labeled edge.
- Derive target and objective weights for randomized branch inclusion; do not
  treat a sampled subset as an unweighted enumeration.
- Preserve duplicate event labels as distinct sampled children when the
  proposal law can generate duplicates.
- Specify how deterministic BR extraction relates to a stochastic exploration
  head during training.

## Execution changes

- Route work queues by both poker execution family and probe/opponent role.
- Extend byte planning for probe branch frontiers, repeated reaches, edge
  inclusion weights, and multiplied completion leaves.
- Reuse one opponent forward/range update before downstream probe branches
  whenever their public prefix is identical.
- Investigate common-random-number completion across sibling probe actions;
  sharing must not introduce an invalid posterior coupling.
- Keep early termination when the probe folds and selected-player backward
  propagation from the sequential implementation.

## Required gates

- Hand-check a tiny tree against exhaustive probe-action enumeration.
- Verify randomized branch weights against a high-sample sequential oracle.
- Test duplicate labels, asymmetric root ownership, early folds, and DDP
  objective normalization.
- Compare gain variance per logical completion sample and per GPU-hour.
- Require a measured improvement over sequential RB before promoting the mode.
