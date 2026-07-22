# Current audit index

Active audit documents describe the sole production NLHE implementation:

- [`reachability_cleanup_20260722.md`](reachability_cleanup_20260722.md)
  records the high-confidence stale-code deletion pass performed from the
  comprehensive audit's production-reachability inventory.
- [`repo_audit_20260721/`](repo_audit_20260721/) is the current comprehensive
  repository audit. Its verdict is **not ready**: two implementation blockers,
  one BR-certification blocker, and the canonical C01--C18 findings remain.

- [`restore_storage_lifecycle/phase_6_acceptance.md`](restore_storage_lifecycle/phase_6_acceptance.md)
  is the current CPU/GPU verdict and links the ordered six-phase restoration
  chain. It is retained as pre-audit migration evidence rather than the current
  repository-wide verdict.
- [`restore_storage_lifecycle/`](restore_storage_lifecycle/) records each
  reviewable phase boundary from authority reconciliation through sole-path
  cleanup and acceptance.

- [`distributed_learner_reference_cache_20260717.md`](distributed_learner_reference_cache_20260717.md)
  is historical evidence for the bounded learner and reference-cache work; its
  compatibility-path measurements are not current rollout acceptance.
- [`sealed_storage_memory_control_20260717.md`](sealed_storage_memory_control_20260717.md)
  is historical evidence for the initially disconnected sealed-storage work.
- [`migration_cleanup_gpu_acceptance_20260717.md`](migration_cleanup_gpu_acceptance_20260717.md)
  is the historical acceptance of the capacity-shaped compatibility pipeline.
- [`release_gate_2026-07-16/`](release_gate_2026-07-16/) — comprehensive
  documentation/math/model/runtime audit and repair record. Current verdict:
  **GO for engineering/estimator correctness**, with strategy quality explicitly
  unclaimed because M-05's approximate-BR probe is owner-deferred.
- [`production_training_readiness_2026-07-16.md`](production_training_readiness_2026-07-16.md)
  is a superseded evidence snapshot from before the comprehensive release gate.
- [`pbg_reachability_inventory_2026-07-16.md`](pbg_reachability_inventory_2026-07-16.md)
  records the import/config/runtime evidence used for the implementation cleanup.

Pre-production findings, synthetic-game validation, migration plans, executor
comparisons, raw promotion artifacts, and experiment commands are historical.
They live under [`../../archive/`](../../archive/) and are not active readiness
criteria. Mathematical contracts remain in [`../theory/`](../theory/), while
runtime and configuration contracts are summarized in
[`../../CONTRACTS.md`](../../CONTRACTS.md).
