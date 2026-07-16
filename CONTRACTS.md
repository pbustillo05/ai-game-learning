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
  pbg.core                     game-independent rollout, values, loss, training
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
  sequential_starts: int
  beta: mapping | null
  node_budget: int
  reach_storage: full | decisions_only | recompute
  family_batch_rows: positive int | null
  max_active_trajectories: positive int | null
  arena_chunk_nodes: positive int | null
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
and all intensities must be finite and at least one.

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
depth, state references, and tree identity are retained without deduplication.
Node and edge capacity overflow fails loudly before partially mutating topology.

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

RB completion is supplied in bounded batches to the single streamed posterior
value reducer. It propagates posterior mass $P$ and posterior-weighted moment
$M$, conditionalizes only at target emission, uses leave-one-out sibling
baselines, and bootstraps only the acting player's moment. A full-completion
alternate reducer and auxiliary propagation hooks are not active APIs.

Normal mode stores a compact scalar trajectory dataset. Sequential RB uses
single-child topology and completion targets. Branched RB retains labeled
multi-child topology and empirical path masses. `reach_storage` changes storage
and replay only; it must not change target values.

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
and reserve limits. Rollout, completion, value, and optimization phases report
allocated/reserved peaks. Preflight rejects unsafe plans before large
allocations. Completion and backward reduction stream within one fixed rollout;
there is no trajectory-chunk objective.

Numeric hot-path tensors remain on their owning device. Declared fixed-size
scheduler control synchronization and transfers are counted separately;
undeclared synchronization is a contract failure. Empty families, single-child
paths, and uneven rank work must complete without collective mismatch.

## Checkpoint and resume

Checkpoints atomically contain model, optimizer, scaler, completed iteration,
scientific configuration, explicit per-rank Torch/NumPy/process RNG state, run
identity, and optional reference-policy state. Resume requires:

- the same scientific configuration and world size;
- a checkpoint inside the configured run directory;
- a gap-free metrics prefix with the same run identity; and
- an iteration within the configured schedule.

Current NLHE checkpoints are restored from their tensor/optimizer/RNG payloads.
Their serialized configuration is compared as data and is not passed through
the new YAML parser, so removed YAML aliases require no general compatibility
loader. CUDA RNG byte tensors are normalized to CPU at the setter boundary;
this narrow compatibility handling is retained for relevant checkpoints.

## Required validation

Changes to production execution must retain tests for all three modes,
configuration rejection, labeled topology, beta/randomized rounding, logical
RNG invariance, old-policy identity, streamed completion, real evaluator
integration, unequal global mass, finite distributed gradients, checkpoint and
resume identity, and memory preflight. CUDA validation starts with idle-device
inspection, uses conservative RTX 2080 Ti workloads, and confirms teardown.

Historical promotion comparisons and retired-game evidence live under
`archive/`; they are not active acceptance surfaces.
