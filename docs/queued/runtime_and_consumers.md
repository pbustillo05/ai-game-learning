# Queued runtime and consumers

Status: Phase 10 accepted contract.

Accepted executable revision:
`944d097e221b90a075c5c813de9f604bfa38dfae`.

This document describes the public product boundary after consumer migration.
It does not perform the Phase 11 directory cutover: legacy source may remain in
the repository until that deletion phase, but no accepted production root may
reach it.

## Public composition

`pbg.config.RootConfig` is the sole public training composition. It contains
one immutable scientific configuration and one explicit execution
configuration. The scientific mode is the Cartesian product of:

- `training_regime`: `self_play` or `best_response`;
- `rollout_mode`: `sequential`, `branched.fixed`, or `branched.KKT`; and
- `private_estimator`: `sampled` or `rao_blackwell`.

Mode applicability is closed by the queued mode/configuration schemas.
Configuration objects reject missing, unknown, legacy, aliased, and
inapplicable nested fields with their dotted path. Hardware profile overlays
and aggregate legacy mode selectors are not runtime inputs. The scientific
mapping is canonical and separate from capacities, device placement,
attention scheduling, paths, iteration count, and checkpoint cadence.

`pbg.runtime.TrainingSession` owns one complete production run. The ordinary,
distributed, and replica entry points all construct that same session:

- `pbg.cli.train` runs one ordinary or distributed configuration;
- `pbg.cli.train_replicas` requires the requested lane count to equal the
  immutable manifest and to exceed one; and
- `pbg.cli.runpod_replica_selection` adds only a synchronized deadline and a
  final strict checkpoint/report around the same session.

There is no runtime engine selector, compatibility adapter, or fallback
composition. Accepted training, evaluation, inference, export, variance, and
benchmark roots reach `pbg.queued` or `pbg.games.poker.queued` and do not reach
`pbg.core`.

## Accelerator boundary

Production devices are explicit `cuda` or `mps`; `cpu`, `auto`, implicit
selection, and fallback are invalid.

Training, distributed training, best-response evaluation, variance
measurement, and training benchmarks require CUDA. Distributed training is
CUDA/NCCL-only. An unavailable requested accelerator is an error.

MPS is supported only for the inference operations implemented without the
evaluator-v2 completion/folded-summary surface. MPS preflop inference remains
accelerator-resident. Folded-state, chance, and postflop operations fail before
CPU staging. `PYTORCH_ENABLE_MPS_FALLBACK=1` is rejected. CUDA supports all
public inference operations.

## Training, ranks, and lanes

Every rank builds the same model state and receives a rank-common broadcast.
Ranks own disjoint logical address ranges and independent explicit RNG state.
Replica lanes own their model, optimizer, scaler, generator, reference state,
controller state, outputs, and labels. Active lanes use one shared generated
storage owner per attempt and the same queued learner operation.

Distributed learners execute a rank-common optimizer-call budget. A rank with
no local rows participates with explicit zero gradients, so every rank enters
the same NCCL SUM sequence. Preflight agreement catches mismatched budgets
before the first reduction. Model, optimizer, and scaler states therefore
remain identical while trajectories and RNG streams remain rank-disjoint.

Recoverable capacity failures discard the unpublished whole attempt, agree the
minimum failure address across ranks, advance to a disjoint retry address, and
apply one rank-common root or branching-controller backoff. An unrecoverable
rank failure is agreed before any continuation. When all ranks remain at the
last completed optimizer boundary, the runtime writes a rank-common failure
checkpoint.

`SIGUSR1` and `SIGTERM` handlers record only a notice. The synchronized
iteration boundary decides whether to checkpoint and stop. Deadline consumers
use the same boundary. The session destroys a process group only after its
last collective consumer.

## Reference cadence and effective work

Iteration-interval reference refresh is evaluated before an applicable
iteration, including iteration zero. Effective-hand cadence is evaluated after
the completed iteration and records the next boundary as `iteration + 1`.
Reference weights, reference iteration, exposure, threshold, overshoot/reset,
and per-lane state are checkpointed and restored.

For Rao--Blackwell modes, the completion binding consumes the evaluator's
actual `Completion.n_eff`. It validates each contribution as finite, positive,
and no greater than the requested sample count, accumulates it on the
accelerator, and publishes the per-lane value through sequential, branched,
and replica results. This actual value controls effective-hand reference
cadence and health reporting. Requested `leaves * n_samples` is a distinct
logical-work count and is never substituted for `n_eff`.

The exact logical-work schema is:

- `trajectories`;
- `completion_samples`;
- `minibatches`; and
- `optimizer_steps`.

Every value is a nonnegative integer and extra or missing fields are invalid.
Metrics publish per-rank/per-lane, per-rank, global-per-lane, and global views.
Trajectories, completion samples, and minibatches add across ranks.
Distributed optimizer steps take the rank maximum for a synchronized
transition before adding across lanes.

## Persistence and artifacts

Runtime loading accepts only the strict queued schemas:

- ordinary checkpoint `pbg-queued-checkpoint-v2`;
- replica checkpoint `pbg-queued-replica-checkpoint-v2`;
- distributed manifest `pbg-queued-distributed-checkpoint-v1`; and
- model-only inference artifact `pbg-queued-inference-v1`.

Ordinary and replica checkpoints preserve model/reference weights, optimizer,
scaler, explicit and ambient RNG, the exact run and logical-work state,
reference cadence, controller/safeguard state, probe identity, scientific
identity, and the gap-free metrics prefix. Saves are atomic. A distributed
manifest is published only after all rank files exist and records the exact
world size, filenames, and content hashes.

Resume rejects scientific drift while allowing execution-only changes. The
existing `metrics.jsonl` must start with the checkpoint's exact gap-free
prefix. A later suffix is backed up and removed before records are appended.
Operational checkpoint discovery accepts only
`checkpoint_????????.pt` and `failure_checkpoint_????????.pt`; rank shards are
not top-level resume candidates.

Inference export accepts an ordinary checkpoint, an explicit replica lane, or
a strict distributed manifest. The artifact records canonical scientific
configuration, source checkpoint identity, model identity, payload identity,
and iteration. Runtime inference never loads a legacy artifact.

Legacy checkpoint, lane, and inference conversion is an explicit offline
operation. Converters accept only approved exact source schemas, validate
derived configuration and mode identity, write atomically to the queued
schema, and produce source/result hashes and a conversion report. Converter
code is not called by runtime loading. Unknown and unselected legacy artifacts
are not runtime inputs.

## Evaluation and inference consumers

Best-response training uses the same `TrainingSession` and requires the
configured target checkpoint content hash. Target initialization may be
independent or use compatible target weights; the resulting initial-model
hash is verified.

`pbg.eval.br_probe.evaluate_br_probe` consumes strict model-only artifacts and
runs on CUDA. It preserves balanced seat/button blocks, paired target/probe
deals, action-path-independent chance addresses, multiplayer early-fold
cutoff, per-seat gain and standard error, and the approximate-NashConv lower
bound. Completion/showdown requirements cannot select MPS or CPU.

`pbg.inference.PokerInferenceSession` owns one strict artifact, one accelerator
model, and stateful public hands. It preserves compact-card parsing, legal
actions, action advancement, private-hand range tables, preflop/postflop
policy/value prediction, singleton use, and bounded batch sessions on the
supported device paths.

## Variance, benchmarks, and diagnostics

The variance consumer restores one frozen model/optimizer/reference/run
boundary, varies only the declared scientific seed stride, runs production
iterations, and accumulates update moments on CUDA. Model snapshots,
statistics, and their overlap with admitted phase memory must fit the immutable
budget. Its report records identities, lane work, update norms, and memory.

The training benchmark constructs a fresh production session for every warmup
and measured repeat. Every repeat begins at the same checkpoint or initial
boundary. Warmups are excluded. A comparison requires identical mode,
scientific/execution configuration, logical RNG, checkpoint, code,
accelerator, required metrics, environment unless explicitly waived, and one
identical before/after logical-work identity across all current and baseline
samples. JSON serialization cannot change that identity. Throughput is based
on measured production player nodes.

Iteration metrics retain:

- scientific, code, accelerator, run, initial-model, rank, lane, and mode
  identity;
- exact rank/lane/global logical work and player-node totals;
- decision, topology, completion, posterior-effective-hand, and reference
  cadence health;
- objective components, gradient finiteness/norm/clipping/scaler state;
- model-family and batch telemetry where requested; and
- configured admission, allocated/reserved, peak, and replica-resident bytes.

Generated topology, completion, target, cache, alias, and learner owners remain
bounded by the immutable manifest and release at their declared final
consumer. The completion effective-hand accumulator is binding-owned and
releases immediately after completion.

## Reachability and remaining cutover

The accepted production roots are training, replica training and selection,
best-response evaluation, inference/export/benchmark, training/replica
benchmarks, and variance. Static import reachability reports no accepted root
that reaches the legacy core, no missing accepted root, and no migrated
implementation-body duplicate.

Phase 10 closes public consumer migration and artifact runtime compatibility.
Phase 11 still owns the atomic directory cutover, deletion of the unreachable
legacy engine and migration scaffolding, package relocation, and final
repository-wide deletion/similarity audit. This document does not authorize
retaining a legacy definition after that cutover.
