# Architecture

## Purpose

`pbg` is a research framework for policy-gradient self-play in finite
public-action Bayesian games: games whose actions and chance events are public,
while each player has a hidden type. Synthetic card games provide small exact
oracles. No-limit hold'em provides the large, GPU-oriented target domain.

The current training path is the corrected C08–C11 implementation:
device-resident rollout, duplicate-preserving fixed-intensity branching,
posterior-moment backward propagation, and a minimal posterior-weighted PPO
objective. Older P08–P11 behavior survives only where tests need compatibility
or historical comparison.

## Dependency direction

The repository has three implementation layers:

1. `pbg/core` defines game-neutral protocols and algorithms. It knows about
   public state handles, owners, events, reach tensors, tree topology, posterior
   statistics, and policy/value outputs. It does not know poker rules or
   synthetic-game rules.
2. `pbg/games/synthetic` and `pbg/games/poker` implement the protocols. Each
   game factory returns a `GameBundle(model, rollout_backend, policy_adapter,
   completion_backend)`.
3. `pbg/eval` loads checkpoints through game-owned construction and implements
   read-only evaluation.

`pbg/core/training.py` resolves game factories by import string. This prevents a
static core-to-game dependency and makes package import direction testable.

Poker has one additional boundary. This repository owns public navigation,
action policies, reach likelihoods, pot amounts, and chip-EV assembly. The
sibling `poker_gpu_evaluator.pbg` facade owns all private poker combinatorics:
collision-conditioned joint profiles, folded-card partition summaries,
completion/equity estimates, and fixed-showdown winner shares. Local adapters
validate version, representation, shape, dtype, device, and semantics but do
not reproduce those algorithms.

## Runtime pipeline

One training iteration performs the following phases:

1. Deep-copy the current model into a frozen rollout model.
2. Ask the selected rollout backend for root state handles and initialize
   padded log-reach rows `[tree, player, private_type]`.
3. Build sequential trajectories or one fixed-intensity labeled sampled tree
   per rank. Cache frozen value rows and selected-event log probabilities.
4. Release the frozen model. Complete leaves through the game-owned backend in
   byte-limited microbatches.
5. Feed each completion microbatch into the backward `P/M` reduction frontier.
   Completed siblings emit targets and are replaced by one parent summary.
6. Build a device-resident categorical table over player nodes using sampled-tree path mass.
7. Repeatedly sample aligned decision rows, re-evaluate the current model, and
   optimize the clipped posterior-weighted policy, value, and entropy loss.
8. Append metrics and periodically write an atomic checkpoint containing the
   model, optimizer, scaler, configuration, and all RNG state.

The frozen rollout policy makes PPO ratios well-defined and keeps target
construction outside autograd. Every cache uses the same stable decision-row
order; mismatches fail before optimization.

## Core data model

### Public state and event handles

Backends expose states as opaque device-local `int32` handles. The core never
decodes them. An `EventBatch` contains an `int32 public_id` plus game-owned
payload tensors, such as poker chance cards. This keeps the generic sampler
free from domain branching and allows the poker backend to retain tensor-native
state.

Owner codes are shared: nonnegative integers are players, `CHANCE = -1`,
`TERMINAL = -2`, and `PRETERM = -3`. A pre-terminal state has no further player
decision; completion owns its remaining chance/reward resolution.

### Stored trees

`TreeBatch` is a shared CSR topology container. Sequential modes place a batch
of independent trajectories in it; branched RB places exactly one sampled tree
per rank/GPU in it. Node arrays store parent, logical-tree id, owner, depths,
state handle, and accumulated path weight. CSR arrays
`child_ptr`, `child_node`, `child_event`, and `nu` store outgoing edges. Equal
public event labels are allowed on distinct child edges and are intentionally
not deduplicated.

`DeviceTreeBuffer` preallocates `n_trajectories * node_budget` rows in sequential
modes and exactly `node_budget` rows for the one branched tree. It appends
parent-major child batches with tensor prefix operations. Overflow is a loud
error; it never truncates a trajectory/tree or changes the offspring law.

### Reach and frozen-policy caches

Reach is stored as unnormalized log likelihood by player and private type. The
three storage modes trade memory for replay:

- `full` stores every node.
- `decisions_only` stores player and leaf rows.
- `recompute` stores roots and reconstructs requested rows by replaying cached
  public-edge likelihoods.

`PolicyCache` stores node ids, frozen values, and one all-private-type selected
log-probability row per sampled abstract child. It is aligned with
`NodeTargets`, including duplicate child labels.

### Completion and targets

`BatchedCompletionStats` is the game-neutral leaf boundary. Its learning fields
are posterior mass `p_hat`, conditional private-type values `v_hat`, public
value vectors, squared-return statistic, and effective sample size. Optional
fields preserve evaluator diagnostics without coupling core code to poker.

`NodeTargets` is aligned to player-decision nodes and contains parent posterior,
child posterior, direct advantages, value targets, child labels/masks/counts,
and optional Q or auxiliary tensors.

The previous completion result and its selected reaches are unreachable before
the next completion call. The frontier moves upward until only the root of the
branched tree remains; it never allocates a whole-tree `[N,P,X]` moment arena.

## Design choices

### Integrate private types in the loss

One model forward pass emits policy and value rows for every possible private
type of the acting player. Training sums over those rows using posterior mass
instead of sampling one actor type for the gradient. This is the central
Rao–Blackwellization step.

### Preserve sampled labels, including duplicates

Fixed branching draws independent events. Two draws may produce the same
public action but still represent independent abstract children with independent
descendants and completion draws. Merging them would change the estimator and
its leave-one-branch baseline.

### Propagate linear moments, conditionalize at emission

The backward pass averages posterior mass `P` and posterior-weighted return mass
`M = P V`. It does not directly average conditional values. Ratios `M/P` are
formed only when targets are emitted, with exact zero behavior for empty mass.
This preserves unbiased linear quantities under sampled branching.

### Prefer explicit ownership over fallback implementations

There is no local poker evaluator fallback. A missing, incompatible, or
partially successful sibling operation raises a typed error. This avoids silent
CPU transfers and prevents two implementations of sensitive poker mathematics
from drifting apart.

### Device residency is contractual

The current rollout path is designed to keep topology, public state, reaches,
events, profiles, and evaluator inputs on one device. Hot-path tests scan for
explicit host transfers. CPU remains supported as an oracle and integration
mode; CUDA uses the same tensor state transition path.

### Determinism is explicit

Configuration loading is strict. Torch and NumPy generators are explicit and
rank-local. `training.reproducibility` selects `strict` deterministic kernels,
`seeded` nondeterministic kernels without autotuning, or `throughput` kernels
with cuDNN benchmarking and TF32 enabled. Every profile retains exact seeds and
checkpoints explicit generator plus process RNG state for every DDP rank. DDP
ranks sample locally and synchronize gradients only.

## Current scope and deliberate omissions

The shipped objective is only `rb_ppo`. Reference-policy KL, NeuRD/R-NaD,
learned Neyman allocation, and public moment losses are deferred. `MomentHeads`
and generic auxiliary propagation hooks exist as research extensions but are
not wired into the default training path. Q targets are transient unless a
caller explicitly requests `emit_q=True`.
