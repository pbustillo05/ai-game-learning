# Active implementation contracts

This document describes the production repository after the family-scheduler
promotion. The mathematical authority remains `docs/theory/`; this file fixes
the implementation boundaries that must preserve those laws.

## Scope

The only active game is equal-stack no-limit Hold'em cash poker. A run selects:

- players $P\in\{2,\ldots,6\}$;
- stack depth $S\in[8,128]$ bb in 0.5 bb increments; and
- the highest $R\in\{4,\ldots,13\}$ ranks, with all four suits.

The deck must satisfy $2P+5<4R$. Blinds are 0.5/1 bb, with no ante or rake.
Equal starting stacks make side pots impossible. Kuhn, Leduc, and synthetic
games are historical material under `archive/`; they are not importable,
configurable, benchmarkable, or tested as active games.

## Active package boundary

```text
pbg.config / pbg.runtime       NLHE composition roots
  pbg.core                     game-independent rollout, targets, learner, training
  pbg.games.poker              NLHE rules, model, rollout and evaluator adapters
  pbg.eval                     offline duplicate-deal poker evaluation
pbg.benchmark                  three-mode production profiling
```

`pbg.core` must not import `pbg.games` or `pbg.eval`. The top-level composition
modules may import both core and poker. `pbg.games.poker` may import core but no
other active game or evaluation package. Import-direction tests enforce this.

There is no game factory registry, rollout-executor selector, or compatibility
dispatch. `pbg.config.load_config` validates NLHE, and `pbg.runtime.train`
constructs the NLHE bundle directly.

## Configuration

Configuration loading is recursive and strict. Missing required fields, wrong
types, invalid values, and unknown keys fail with a dotted path. In particular,
retired keys such as `rollout.executor`, `rollout.n_trees`,
`rollout.trajectory_chunk_size`, and `rollout.queue_packing` are errors.

```text
game:
  n_players: int
  stack_bb: float
  rank_count: int
rollout:
  sequential_roots_per_gpu_initial: int
  sequential_roots_per_gpu_minimum: positive int | null
  sequential_roots_per_gpu_maximum: positive int | null
  branching_mode: fixed | dynamic | null
  beta: mapping | null
  family_batch_rows: positive int | null
  max_active_trajectories: positive int | null
  storage_page_rows: positive int | null
  execution_memory_bytes: positive int | null
completion:
  n_samples: positive int | null
  backend: auto | cpu | cuda
training:
  mode: normal | rao_blackwell | branched_rao_blackwell
```

Normal and sequential Rao--Blackwell modes interpret `sequential_starts` as a
global logical count. Distributed launch partitions that count into contiguous,
collision-free rank quotas and requires at least one start per rank. Branched
Rao--Blackwell requires the literal value one and creates one distinguished
root/tree per rank. Only branched mode accepts `beta`; it requires `default`
and all intensities must be finite and at least one. Branched configurations
must also choose `branching_mode: fixed` or `branching_mode: dynamic` explicitly.
Both modes require public T/C heads and positive T/C loss coefficients from the
first update. Fixed mode uses the configured beta schedule for the entire run.
Dynamic mode additionally requires `adaptive_kkt`; it uses the same fixed beta
schedule during warmup, then enables T/C-driven KKT intensities and controller
updates. `adaptive_kkt.enabled` is not a compatibility alias and is rejected.

Hardware profiles contain execution limits only. Checkpoint scientific identity
excludes the hardware profile, so the same scientific run can resume on a
compatible execution envelope.

## Public tree and sampling laws

Every public tree has exactly one distinguished root $\varnothing$. In branched
mode the root obeys the ordinary randomized-rounding rule; it receives no
special multiplicity. In sequential modes only roots branch, producing the
configured independent trajectory starts.

Stored topology is duplicate-preserving CSR. Repeated event labels are distinct
sampled children. Child ordering, empirical edge masses, `path_weight`, owner,
depth, and tree identity are retained without deduplication. Mutable state
handles exist only during rollout and never enter sealed or training storage.
Node, edge, reach, history, and control pages are byte-admitted transactionally
before a child wave mutates topology or consumes a random draw.

Normal, sequential RB, and branched RB use one family-scheduled execution path.
Poker exposes only the logical families `pre` and `post_A`, where $A$ is the
number of active players. Street is row metadata and never an additional queue
key. Each occurrence follows the monotone lifecycle
`NEW -> QUEUED -> EVALUATING -> DONE`.

The scheduler may alter physical execution order only. It must preserve:

- one private profile per sequential trajectory;
- the configured beta and randomized-rounding law;
- family membership and logical insertion order;
- completion and old-policy identity;
- logical RNG addresses; and
- exact decision-mass scaling.

The actual-cost backend may split a logical family block into safe model calls.
There is no fixed-quantum alternative. Admission, arena, and work-queue overflow
must be transactional. All execution allocations are charged to a byte
accountant before mutation.

## Logical randomness

Random draws are addressed by immutable logical coordinates: seed/iteration,
global trajectory or rank tree identity, path key, child ordinal, and draw
purpose. Draw purposes distinguish profile, policy event, chance, branching,
normal settlement, and completion streams. Queue timing, family batch size,
completion microbatching, rank quota placement, and physical model-call order
must not affect addressed samples.

Changing the logical RNG version is a scientific compatibility change and must
be explicit in run and benchmark metadata.

## Poker and evaluator ownership

The CPU engine and tensor backend own public betting state and affine pot
accounting. The sibling `poker_gpu_evaluator.pbg` API exclusively owns joint
private-profile sampling, collision-conditioned posterior/completion draws,
folder partition summaries, and showdown comparison. This repository must not
duplicate those algorithms or modify the sibling repository.

The active compact deck uses the sibling's documented `pbg-v3` mapping. Inputs,
outputs, metadata, dtype, device, shape, and seed behavior are validated at the
boundary. Fold terminal chip rewards are public, but folded-range posterior
sampling still crosses the sibling API.

## Retained rollout state and targets

Rollout stores exact frozen policy likelihoods and values at retained decisions.
The first update of a launch or resume recomputes them and checks identity before
backward. Minibatching never reevaluates the frozen behavior policy.

Each mode seals exact `RolloutStorage` before target construction. Natural poker
decision inputs are captured at their final mutable-state/reach consumer;
completion leaves retain one exact range-weight owner and immutable evaluator
inputs. `build_normal_targets`, `build_sequential_rb_targets`, and
`build_branched_rb_targets` stream bounded work from that store into immutable
`TrainingStorage`, after which topology and leaves are released.

Sequential RB deduplicates posterior rows by trajectory/player. Branched RB
uses ragged action spans, propagates posterior mass $P$ and weighted moment $M$
through a prebounded reusable frontier, and retains current dynamic-$\beta$,
$T/C$, nominal/executed-count, queue-correction, and protected-root semantics.
There is no alternate padded target reducer or stored production $Q$ target.

## Loss and distributed objective

Normal mode uses `ppo`; both RB modes use `rb_ppo`. Policy, clipped value or MSE
value, exact entropy, and optional exact legal-action reference KL terms are
computed only on valid actor types and legal events. A periodic reference model
is immutable between configured refresh iterations and is checkpointed when KL
is enabled.

For decision $s$ on rank $g$, let $\omega_g(s)=1$ in normal/sequential RB and
the stored root-to-node `path_weight` in branched RB. For family $m$ define

$$
Z_{g,m}=\sum_{s\in D_{g,m}}\omega_g(s),\qquad
Z=\sum_{g,m}Z_{g,m}.
$$

The fixed-rollout objective is the single global ratio

$$
L(\theta)=\frac{1}{Z}\sum_{g,m}\sum_{s\in D_{g,m}}
\omega_g(s)\,\ell(\theta;s).
$$

A local family draw therefore samples with probability
$\omega_g(s)/Z_{g,m}$ and scales its sample mean by $Z_{g,m}/Z$. The path mass
must not be multiplied a second time. Ranks accumulate already globally scaled
gradients, zero-fill absent parameter gradients, and execute one flat fp32
`SUM`. Equal-rank averaging and physical-microbatch mean normalization are
forbidden because they change the objective for unequal masses.

All ranks participate in the same finite-gradient decision. AMP unscaling,
clipping, optimizer stepping, and scaler updates occur only after that common
decision.

## Memory, synchronization, and completion

CUDA launch requires a hardware profile with token, token-pair, memory-fraction,
and reserve limits. Rollout, seal, target, and optimization phases report
allocated/reserved peaks. Their priors come from the schemas that allocate the
real pages and exact stores. Sequential width is selected only from four-phase
byte feasibility; branched growth uses a hard execution byte ceiling and stops
cleanly when a mandatory continuation cannot fit. Completion and backward
reduction stream within one fixed rollout; there is no retry, reroll, replay,
or trajectory-chunk objective.

Numeric hot-path tensors remain on their owning device. Declared fixed-size
scheduler control synchronization and transfers are counted separately;
undeclared synchronization is a contract failure. Empty families, single-child
paths, and uneven rank work must complete without collective mismatch.
CUDA history materialization uses the lossless preflighted bound and a batched
device gather. Mutable public states and persistent history records use
recyclable, refcounted slots; dead history ancestry is reclaimed at its final
state consumer. Folded summaries seal only sparse dependency anchors, and
all-live partitions allocate none. Mixed-family physical planning may transfer
one fixed, counted control block; it may not introduce row-dependent host
decisions.

## Checkpoint and resume

Checkpoints atomically contain model, optimizer, scaler, completed iteration,
scientific configuration, explicit per-rank Torch/NumPy/process RNG state, run
identity, and optional reference-policy state. Resume requires:

- the same scientific configuration and world size;
- a checkpoint inside the configured run directory;
- a gap-free metrics prefix with the same run identity; and
- an iteration within the configured schedule.

Current NLHE checkpoints are restored from their tensor/optimizer/RNG payloads.
Their serialized configuration is compared as data rather than passed through
the YAML parser. Checkpoint schema v5 contains one explicit migration that
deletes the retired total-row allocation field without translating its value;
there is no general configuration-alias loader. CUDA RNG byte tensors are
normalized to CPU at the setter boundary; this narrow compatibility handling
is retained for relevant checkpoints.

## Required validation

Changes to production execution must retain tests for all three modes,
configuration rejection, labeled topology, beta/randomized rounding, logical
RNG invariance, old-policy identity, streamed completion, real evaluator
integration, unequal global mass, finite distributed gradients, checkpoint and
resume identity, and memory preflight. CUDA validation starts with idle-device
inspection, uses conservative RTX 2080 Ti workloads, and confirms teardown.

Historical promotion comparisons and retired-game evidence live under
`archive/`; they are not active acceptance surfaces.
