# Current audit index

Active audit documents describe the sole production NLHE implementation:

- [`release_gate_2026-07-16/`](release_gate_2026-07-16/) — comprehensive
  documentation/math/model/runtime audit. Current verdict: **NO-GO** pending the
  listed release blockers; this supersedes the older readiness conclusion.
- [`production_training_readiness_2026-07-16.md`](production_training_readiness_2026-07-16.md)
  records the accepted three-mode evidence and remaining deployment work.
- [`pbg_reachability_inventory_2026-07-16.md`](pbg_reachability_inventory_2026-07-16.md)
  records the import/config/runtime evidence used for the implementation cleanup.

Pre-production findings, synthetic-game validation, migration plans, executor
comparisons, raw promotion artifacts, and experiment commands are historical.
They live under [`../../archive/`](../../archive/) and are not active readiness
criteria. Mathematical contracts remain in [`../theory/`](../theory/), while
runtime and configuration contracts are summarized in
[`../../CONTRACTS.md`](../../CONTRACTS.md).
