# Repository audit — 2026-07-21

## Verdict

The repository is **not yet faithful enough to declare the refactor complete**.
Exactly five requested semantic combinations are reachable, and the principal
KKT, refill, direct-backward, SDPA, and cache mechanisms survive. However, two
confirmed implementation blockers remain: production still seals through the
superseded `SampledTree` representation outside one complete hard-memory
transaction (C01), and rank-local mandatory failures can strand distributed
peers (C02). BR additionally has an authority-level certification blocker
(C03) and a confirmed multi-player early-fold defect (C04).

There are **five semantic combinations but only three storage/training labels**.
That is intentional sharing, not a sixth mode or a defect by itself:
`normal`, `sequential_rb`, and `branched_rb` are the three storage/loss
families; fixed versus KKT branching and the BR probe overlay produce the five
externally meaningful combinations. The defect is that no closed semantic
identity is propagated to presets, manifests, acceptance tests, and benchmarks.

No same-hardware, same-workload measurement proves a current-HEAD performance
regression. This audit therefore separates confirmed defects, authority gaps,
static performance risks, measured historical baselines, validation gaps, and
unmeasured optimizations.

## Documents

1. [Scope, contracts, and method](01_scope_contracts_and_method.md)
2. [Executive findings](02_executive_findings.md)
3. [Five-mode traceability](03_five_mode_traceability.md)
4. [Math, model, and estimator fidelity](04_math_model_and_estimator_fidelity.md)
5. [Storage, queue, and distributed fidelity](05_storage_queue_distributed_fidelity.md)
6. [Production reachability and deletion inventory](06_production_reachability_and_deletion_inventory.md)
7. [Historical regression analysis](07_historical_regression_analysis.md)
8. [Performance and optimization](08_performance_and_optimization.md)
9. [Validation matrix](09_validation_matrix.md)
10. [Recommended remediation order](10_recommended_remediation_order.md)

The detailed evidence is preserved unchanged under [batches/](batches/). Batch
[13](batches/13_adversarial_finding_reconciliation.md) is the controlling
finding reconciliation; batch [12](batches/12_validation_evidence.md) is the
controlling execution record.

Audit point: `b55df976e80b82c474c86202c90f3b32072af03c`. The pre-existing
modification to `docs/todo/restore_authoritative_storage_lifecycle_20260721.md`
is user-owned and was neither treated as evidence nor modified. The batch
reports were created as part of this audit. No production code was changed.
