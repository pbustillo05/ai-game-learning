# Queued sequential modes

Status: Phase 6 executable contract, validated and audited at
`df25d020284c16f4600ae90603ba3de594d7e98c`.

## Closed mode surface

The common `pbg.core.iteration.IterationEngine` composes all four sequential
modes through `PokerSequentialProgram`, using the same phase lifecycle and
recovery authority as branched, replica, and distributed execution:

| Training regime | Private estimator |
|---|---|
| self play | sampled |
| best response | sampled |
| self play | Rao–Blackwell |
| best response | Rao–Blackwell |

The modes share generated storage, semantic family queues, rollout
transactions, layered targets, learner batches, loss control, and the dense
FP32 optimizer. Mode schemas add only applicable fields. Best response
compacts trained and frozen inference rows before either model call. Frozen
rows retain no learner head, target, reference output, gradient, or optimizer
work.

Preflop RB values, policies, posterior mass, and moments use the
$K=R^2$ class domain. Postflop uses the exact $H=\binom{4R}{2}$ hand domain.
The only post-to-pre conversion is the declared deterministic class sum. It
uses a resident conversion mask and matrix multiplication without a
temporary $K\times H$ product per row.

## Folded-range ownership

`FoldedRanges.pre` owns normalized $K$-wide class shape and
`FoldedRanges.post` owns normalized $H$-wide exact-hand shape. Each row also
owns its raw log mass, so reconstruction preserves both $\log F_k$ and
$F_kp_k$.

For $P>2$, every retained family model input has a compact folded-statistics
suffix containing:

- per-hand $\Lambda_B$;
- signed eigenvalues and eigenvectors for $\beta_B$;
- folder log mass and unnormalized card mass;
- the tight-cluster scalar; and
- a sealed-summary validity marker.

An instant preflop fold is loose. A player that voluntarily invested chips and
later folded is tight. Only tight wings enter the order-three correction.

A fold appends its range and refreshes the destination summary. A street
change refreshes against the new board. Every other transition copies the
summary unchanged. Model forwards read the sealed suffix and never resolve
`FoldedRanges`, recompute finite differences, or run an eigendecomposition.
Align-and-seal copies the complete model input; `FoldedRanges` and hot
references are then released.

Heads-up rows have no folded suffix because a heads-up fold is terminal.

## Completion and selected ownership

Sampled leaves call evaluator-v2 showdown evaluation. Folded seats use
inactive combo id `-1`.

RB leaves call:

- `complete` for self play; or
- `complete_selected` for best response.

Best response therefore owns one selected posterior/value channel rather than
dense all-player output. The CUDA contract test proves selected completion is
the exact corresponding slice of dense completion.

The conversion mask is created once per RB completion binding. Completion is
its exact last consumer, and the binding is deleted before align-and-seal.
The mask is consequently absent from targets, learner, and final backward.

All posterior, moment, conditionalization, PPO controls, losses, gradients,
and optimizer state transitions use explicit FP32 control arithmetic.

## Shared-response probe

The probe is an ordinary best-response sequential-RB consumer. Its scientific
identity includes:

- target checkpoint SHA-256;
- initial-model SHA-256;
- initialization rule (`random` or `target`);
- shared-response semantics version; and
- exact logical trajectory identity.

When the selected response seat folds, rollout creates an immutable synthetic
leaf immediately. Its selected payoff is the negative committed amount; no
later opponent action or showdown contributes.

Evaluation runs mixed and baseline profiles on identical trajectory ids.
Trajectory ids must form consecutive complete $P^2$ blocks, so every
`(selected seat, button)` pair occurs exactly once per block. Reports contain
paired per-seat gains, the approximate NashConv lower bound, standard error,
and a 95% paired Student-$t$ interval. Finite-sample critical values are
computed numerically beyond the tabulated range.

Training checkpoints retain the initialization identity and `RunState`
logical-work counters. Evaluation reports retain trajectory count and a
SHA-256 of the exact trajectory-id vector.

## Offline conversion

The sole offline converter accepts approved sequential sampled, sequential RB,
and shared-response artifacts. For probes it reconciles the configured target
with top-level target metadata, preserves `initial_model_sha256`, validates
the initialization rule, and rejects disagreement. Conversion is
destination-exclusive and records source/result hashes; there is no runtime
compatibility loader.

## Admission and verified boundary

Game-specific admission derives:

- concurrent family bytes: model input/output, transition operands, and
  folded-statistics scratch additively;
- chunked folded finite-difference workspace;
- RB per-row H-to-K posterior/moment workspace;
- the resident $K\times H$ mask;
- the versioned 8 MiB GEMM library bound; and
- 64 KiB CUDA allocation alignment.

At maximum geometry ($P=6$, $S=128$, $R=13$), validation measured:

| Operation | Measured incremental peak | Admitted bound |
|---|---:|---:|
| folded statistics, batch 8 | 25,980,416 B | 29,532,288 B |
| additive sequential family, batch 8 | — | 34,321,280 B |
| H-to-K, fresh process, batch 8 | 9,427,456 B | 9,444,224 B |

The native postflop profiler contained no `_local_scalar_dense` or
`is_nonzero`. Full-deck H-to-K outputs were bit-identical. Weak-reference
instrumentation for HU self-play RB and three-player BR RB proved the
completion binding and CUDA mask dead at align-and-seal and final backward.
