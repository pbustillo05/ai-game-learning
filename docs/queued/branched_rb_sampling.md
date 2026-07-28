# Queued branched Rao–Blackwell sampling

Status: Phase 8 candidate contract. Independent exact-revision validation and
read-only audit are required before this status is promoted to validated.

## Closed mode surface

The Phase 7 branched engine now composes Rao–Blackwell private estimation with
both training regimes and both controllers:

| Training regime | Branch controller |
|---|---|
| self play | fixed |
| best response | fixed |
| self play | KKT |
| best response | KKT |

These modes reuse the same family scheduler, transaction, chance closure,
topology recorder, layered target executor, ragged learner, optimizer,
recovery, and checkpoint owners as sampled branching. There is no second
profile, branching, target, or learner engine.

## Posterior child law

For a player or chance node with executed degree $N$, child zero inherits the
parent generating profile and consumes no posterior draw. Exactly $N-1$ later
children receive fresh profiles. Each fresh proposal draws every seat
independently from its action- and board-conditioned marginal, then accepts
the complete profile only when all private cards are distinct.

Conditioning independent seat proposals on collision freedom is exactly the
required joint posterior law. The CUDA sampler:

- masks every board-conflicting hand before constructing the seat CDFs;
- uses logical `(seed, iteration, root, path, attempt, seat, retry)` addresses;
- performs a fixed eleven-step upper-bound search over at most 1,326 hands;
- tests collisions by exact pairwise two-card comparisons across at most six
  seats;
- selects the first successful proposal from a fixed attempt envelope; and
- records attempt count, exhaustion, and the minimum failing logical address
  on device.

Inactive envelope rows and inherited child zero consume no posterior RNG
address. Subdivision and physical scheduling therefore cannot change the
selected proposal.

## Action, fold, and chance conditioning

After an action is selected, only the acting seat's exact-hand log reach is
incremented by that action's per-hand log policy. Folded ranges retain the
conditioned shape and raw log mass in the Phase 6 `FoldedRanges` owner.

Chance proposals sample from these conditioned reaches after masking the
current board. Each generated chance child carries its sampled generating
profile, updated exact reaches, folded references, and logical path through
the existing bounded chance closure. The closure grows its scratch envelope
by generation rather than expanding every generation immediately to the final
6,804-row bound.

## Completion, targets, and learner rows

RB leaves call evaluator contract v2 through the existing `RBCompletionBinding`.
Preflop work remains in the $K=R^2$ class domain and postflop work remains in
the $H=\binom{4R}{2}$ exact-hand domain. The declared post-to-pre conversion is
the only domain conversion.

The common layered executor propagates mass, moment, trace, cost, posterior,
and advantage targets over compact topology. Sibling baselines reuse the
already-computed parent sums, and unary nodes reuse their sole child
aggregate. Decision weights are additive log path masses. The common ragged
store retains only real action spans; no maximum-action learner axis is
created.

The sole offline converter accepts approved branched artifacts by recording
the closed three-axis mode, controller state, retry/logical-work state, and
branched RB estimator identity. Runtime compatibility loading is not used.

## Failure and publication

Rejection exhaustion is a terminal device failure, not a capacity retry. Its
failure record contains phase, input family, minimum logical address,
requested and available attempt extents, and maximum observed attempt count.
The affected lane is blocked before destination publication. The complete
attempt is discarded and no topology, learner row, or partial endpoint is
published.

Capacity overflow remains the only recoverable failure. It follows the Phase
7 bounded controller backoff and disjoint retry-address protocol.

## Asynchronous publication and memory

Topology recording uses a fixed four-slot device/pinned ring. Four family
transactions may be queued before one ordered commit-feedback boundary.
Device-resident realized tree size preserves the exact cost-safeguard cutoff
while feedback is in flight.

Independent model rows are evaluated in at most 64-row CUDA subspans inside
an unchanged family transaction whose admitted and frozen family pack cap is
84. This bounds inference workspace without changing family scheduling,
logical RNG, child order, or commit identity.

Benchmark telemetry reports rollout time, routing bytes, host synchronization,
child reservation/commit CUDA time, target CUDA time, stage attribution, and
phase-resolved allocated/reserved peaks. Target setup wall time remains a
separate diagnostic and is not mislabeled as propagation.

## Frozen CUDA gate

The exact checkpoint-backed HU 32 bb, full-deck, fixed-$\beta=3$,
1,024-completion-sample workload used seeds 104830, 209559, and 314288 on one
RTX 2080 Ti. It retained the checkpoint-declared SDPA execution backend and
family pack cap 84.

The three-seed candidate medians were:

| Metric | Candidate median | Frozen limit |
|---|---:|---:|
| rollout per retained decision | 6.527 ms | 9.433 ms |
| routing traffic | 88,480,920 B | 12,838,000,000 B |
| routing per retained decision | 4,468.7 B | 841,000 B |
| host synchronizations | 66 | 79 |
| child reservation and commit | 1.115 s | 17.71 s |
| target propagation CUDA time | 0.00576 s | 0.500 s |
| peak allocated memory | 4,516,409,344 B | 4,947,018,317 B |

The peak limit is 1.05 times the July reference allocation of
4,711,446,016 bytes, or 4,947,018,316.8 bytes. The benchmark report contained
no violations. All 204 cumulative Phase 1–8 CUDA tests also passed before
candidate freeze.
