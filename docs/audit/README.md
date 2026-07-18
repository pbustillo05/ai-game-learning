# Current audit index

Active audit documents describe the sole production NLHE implementation:

- [`distributed_learner_reference_cache_20260717.md`](distributed_learner_reference_cache_20260717.md)
  records phase-2 bounded learner, direct flat-SUM gradient, reference-cache,
  CPU/distributed, and GPU acceptance evidence.
- [`sealed_storage_memory_control_20260717.md`](sealed_storage_memory_control_20260717.md)
  records the phase-1 immutable storage, streaming-target, and memory-only root
  controller handoff.
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
