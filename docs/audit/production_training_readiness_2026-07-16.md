# Production training readiness — 2026-07-16

> **Superseded evidence snapshot.** This document records promotion evidence at
> its original revision; it is not the current release verdict. The active gate
> is [`release_gate_2026-07-16/`](release_gate_2026-07-16/).

Status at recorded revision: **one production rollout implementation accepted locally** for normal,
sequential Rao--Blackwell (RB), and branched RB NLHE. The former wavefront and
executor-selection surfaces have been removed. This file is a historical
evidence dashboard; promotion comparisons and their legacy names are archived under
`archive/family_scheduler_promotion/`.

## Fixed production contract

- Poker has only logical execution families `pre` and `post_A`; street remains
  row metadata.
- All modes use bounded family work queues, occurrence handles, logical RNG
  addresses, exact old-policy retention, and the same actual-cost model-call
  splitter.
- Normal and sequential RB use a global start count partitioned across ranks.
  Branched RB creates one distinguished root/tree per rank and applies ordinary
  randomized rounding at the root.
- Completion remains owned by the sibling evaluator. Physical scheduling may
  not change estimator laws, beta, queue membership, RNG, or decision-mass
  scaling.
- The learner uses the exact global decision-mass ratio and one flat fp32
  gradient `SUM`; equal-rank averaging is not supported.

The detailed algorithm is
[`../theory/pseudocode/family_scheduled_rollout.tex`](../theory/pseudocode/family_scheduled_rollout.tex)
and the distributed update is
[`../theory/pseudocode/distributed_update.tex`](../theory/pseudocode/distributed_update.tex).

## Accepted evidence

| Scope | Result |
|---|---|
| CPU and evaluator integration | Queue/arena transactions, all three modes, topology, RNG, old-policy identity, completion, checkpoint, and global-mass laws pass. |
| Trained 6-max normal | At 1,536 starts the promoted scheduler passed the historical three-seed local performance gate: rollout ratio 1.059, 95% interval [1.042, 1.112]; iteration ratio 1.053, [1.039, 1.100]. |
| Trained 6-max sequential RB | At 1,536 starts it passed: rollout ratio 1.016, [0.983, 1.098]; iteration ratio 1.009, [0.988, 1.078], with valid completion and finite gradients. |
| Trained fixed-beta branched RB | Five fixed-root samples passed the topology-aware aggregate wall-time gate: rollout ratio 1.077, [0.972, 1.198]; iteration ratio 0.994, [0.920, 1.080]. |
| Two-rank 20+5 soak/resume | All three modes produced exact 0--24 histories after checkpoint-20 resume, finite reduced gradients, unequal rank work, flat post-warmup execution memory, and clean teardown. |

The historical baseline implementation was retained only long enough to produce
this evidence. Removing it does not alter the accepted estimator or scheduling
contract.

## Cleanup revalidation

The simplified tree was revalidated after removal of the alternate paths:

- The broad CPU suite passed with 212 tests and 12 CUDA/environment skips;
  the socket-enabled DDP suite separately passed all 3 tests.
- Four focused CUDA/evaluator tests passed on GPU 1, including transactional
  queue/arena/RNG behavior and real sibling-evaluator integration.
- One-iteration trained NLHE smokes passed for normal, sequential RB, and
  branched RB through the same scheduler, with finite gradients and checkpoints
  in every mode.
- A two-rank normal smoke on GPUs 1 and 2 exercised unequal local decision
  masses (19 and 24), exact global mass 43, a finite reduced gradient, and a
  successful two-rank load of its checkpoint.
- GPUs 1 and 2 returned to their initial idle footprints of 9 MiB and 19 MiB.

## Current validation requirement

Before CUDA work, inspect `nvidia-smi`; use only idle RTX 2080 Ti devices and
the shared `/home/pablob/.venvs/poker` environment. Stage tiny single-device
tests before two-rank tests, retain the configured reserve, and confirm both
devices return to idle. H100 work is deployment calibration rather than a new
promotion program.

## Remaining work

The highest-value follow-up is the first larger production job plus a real
checkpoint resume under the simplified schema. If a workload regresses, first
measure scheduler/control, model, evaluator, allocator, and learner exposure
separately. Do not add a second executor, street queues, adaptive beta, stale
policy pipelines, or a custom scheduler without evidence that the accepted
path cannot meet the workload.
