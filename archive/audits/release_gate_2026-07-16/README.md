# Production release gate audit

Audit date: 2026-07-16

Initial audited revision: `dde65f6d470cd4d890ec1b19293c22e83139cc24`

Validated repair revision: `5c10366ffaa330eeab5af73a6761b50874100246`

## Verdict

**GO for an engineering/estimator-correctness release of the fixed-intensity
trainer.**

All release blockers and all major/moderate findings in this report are resolved
except M-05, which the repository owner explicitly excluded from this repair
scope. Normal, sequential Rao--Blackwell (RB), and branched RB pass the repaired
one- and two-GPU matrix, 25-update checkpoint/resume soaks, and the complete
automated suite. A realistic 6-player checkpoint gate also passes.

This is deliberately not a strategy-quality verdict. The release has no
approximate-best-response or exploitability probe, so it must not be described
as demonstrating strategic improvement, convergence, or a useful poker policy.

## Resolved production contract

- Every legal `rank_count` from 4 through 13 is carried end to end through
  geometry, both models, rollout, evaluator adapters, completion, showdown, and
  folded summaries.
- Lossless public-history capacity is derived from `(P,S)`, cross-checked with
  hardware token limits, and tested at all player/stack boundaries.
- The common reduced-gradient finite decision gates clipping, parameter update,
  and loss-scale transition in FP16, BF16, and FP32.
- Linear entropy decay uses `training.iters`; numeric and cross-field
  configuration validation fails early with dotted paths.
- CUDA library launches require a hardware profile.
- CUDA history input and folded-summary caching stay device resident; the one
  mixed-row scheduler control block is declared and counted.
- Frozen behavior rollout and learner inference use the same configured
  precision, including after checkpoint resume.
- Reference KL is implemented. Optional public moment output heads are
  structural only and have no registered target, loss, or branching controller.
- Cohesive optimization, checkpoint, model-input, cache, and policy-planning
  lifecycles have been extracted from the two oversized modules.

## Final evidence summary

| Gate | Result |
|---|---|
| Complete suite on real CUDA + socket-enabled DDP | 382 passed, 1 version-conditional skip |
| Every `R=4,...,13` and every legal `(P,R)` | passed |
| One GPU, all three modes | finite update + checkpoint in every mode |
| Two GPUs, all three modes | unequal rank mass + finite flat SUM update in every mode |
| Two-rank checkpoint/resume | exact 0--24 history in every mode under a changed compatible profile |
| Realistic 6-player normal gate | 20 finite updates, active reference KL, checkpoint at 20 |
| Peak for realistic gate | 3.91 GiB reserved versus 9.20 GiB calibrated safe limit |
| Final GPU hygiene | test devices returned to 9/19 MiB idle baselines; no owned process remained |

Detailed commands and metrics are in [validation evidence](05_validation_evidence.md).

## Repair commits

| Finding/work | Commit |
|---|---|
| B-04 entropy horizon | `8175b70` |
| B-03 finite-step gate | `1cae1e2` |
| B-05 strict values | `6155287` |
| B-02 history capacity | `666ccf8` |
| M-02 CUDA profile gate | `c6f8eec` |
| B-01 every compact deck | `5c3c5ef` |
| M-01 device-resident inputs/cache | `6659205` |
| M-04 lifecycle extraction | `cd355c0` |
| M-03 active documentation | `3e22e0f` |
| Resume-discovered precision mismatch | `5c10366` |

## Report set

- [Scope and method](01_scope_and_method.md)
- [Findings and repairs](02_findings.md)
- [Mathematics and model traceability](03_math_and_model_traceability.md)
- [Runtime, configuration, and losses](04_runtime_configuration_and_losses.md)
- [Validation evidence](05_validation_evidence.md)
- [File-by-file inventory](06_file_inventory.md)
- [Release checklist](07_release_checklist.md)

The older `production_training_readiness_2026-07-16.md` is retained only as a
superseded evidence snapshot. This report is the current release decision.
