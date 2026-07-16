# Architecture

## What the repository implements

`pbg` trains policies for finite public-action Bayesian games. A public state is
shared, each player has a hidden private type, and observed actions update an
unnormalized private-type likelihood. Synthetic games provide small exact
oracles; six-player no-limit hold'em is the GPU target.

There are three implemented training pipelines:

| `training.mode` | Rollout | Target | Loss |
|---|---|---|---|
| `normal` | one sampled private profile and one path per trajectory | sampled scalar lambda return | `ppo` |
| `rao_blackwell` | one sampled profile/path per trajectory, cut at fold/showdown leaves | posterior mass and conditional value over every actor type | `rb_ppo` |
| `branched_rao_blackwell` | one duplicate-preserving sampled tree per rank | posterior-moment reduction with sibling baselines | `rb_ppo` |

Normal and sequential RB can stream a logical trajectory set through resident
chunks. Branched RB always owns one sampled tree per rank and cannot be chunked.

## Package boundaries

The dependency direction is deliberate:

1. `pbg/core` defines opaque state/event protocols, rollout, topology, reach
   caches, completion DTOs, targets, minibatching, losses, and training.
2. `pbg/games/poker` implements those protocols and returns one
   `GameBundle(model, rollout_backend, policy_adapter, completion_backend)`.
3. `pbg/eval` constructs poker models from checkpoints and performs read-only
   duplicate-deal evaluation.

`pbg/core/training.py` imports a selected game factory by string. Core modules
therefore do not statically import the game package.

Poker has another hard boundary. This repository owns public betting state,
public chance navigation, action policy, likelihood reaches, pot strata, and
payoff assembly. `poker_gpu_evaluator.pbg` owns collision-conditioned joint
profiles, folded-card partition summaries, completion/equity, and fixed-profile
winner shares. Local adapters validate the facade version, combo map, tensor
shape, dtype, device, and result semantics. There is no local fallback for that
private-card mathematics.

## One iteration as it actually runs

The non-chunked lifetime is:

```text
trainable model
    -> device-to-device frozen snapshot
    -> rollout: topology + state store + reach + old policy/value cache
    -> normal: settle sampled rewards and build scalar targets, then delete snapshot
       RB: delete snapshot, then completion and posterior target construction
    -> compact iteration data
    -> delete rollout topology/full reach
    -> one or more logical minibatches
       -> one or more hardware-bounded physical forwards/backwards
       -> one optimizer step per logical minibatch
```

Normal rollout settles the carried private profile directly and skips completion.
Sequential/branched RB completes leaves in microbatches of at most 256 rows and
feeds them into the posterior-moment reduction.

When `rollout.trajectory_chunk_size < rollout.n_trees`, `_one_iteration` makes
one frozen snapshot and repeats the following loop:

```text
roll out one resident chunk
construct targets for that chunk
optimize every configured minibatch/epoch for that chunk
release the chunk
continue with the next chunk using the original frozen snapshot
```

This is not a global replay buffer. In particular, all logical trajectories are
not rolled out before optimization starts. `ppo_epochs` shuffles without
replacement inside each resident chunk; it gives exact row coverage per chunk,
not one global permutation of all trajectories. The trainable model changes
between chunks, while rollout probabilities for every chunk still come from the
same snapshot. The snapshot also remains resident during chunk optimization.

The legacy path, used when `ppo_epochs` is absent, performs
`gd_steps_per_iter` with-replacement samples of `minibatch_size` rows.

## Rollout execution model

### Sequential modes

`_sample_trees_sequential` holds one active state, reach row, and private
profile per live trajectory. It advances all active trajectories by one public
transition in a synchronized frontier wave:

1. separate current rows into player and chance states;
2. send all current player rows to one backend `policy_forward` call;
3. sample one action per player row and update the acting player's reach;
4. sample chance events and compatibility-mask reaches;
5. apply all successor transitions and append one child per active row.

The active set shrinks as trajectories finish. Different streets can coexist in
one wave, but a trajectory cannot produce its next decision until the current
wave has completed.

### Branched mode

`_sample_branched_labeled` advances one breadth frontier. Every active node draws
a randomized-rounded offspring count. Policy/value is evaluated once per unique
player parent; policy output is gathered for its abstract children, which draw
independent posterior profiles and events. Duplicate public labels remain
separate edges. The next wave contains all children that were just created.

Both samplers therefore batch the current ready frontier. Neither sampler owns
persistent preflop/postflop ready queues, high/low watermarks, or asynchronous
queue drains across waves. Poker's backend partitions only the rows supplied by
the current call. See the [efficiency audit](audit/rollout_efficiency_2026-07-15.md)
for the consequence.

## Public state and topology

Owner codes are shared: players are nonnegative, `CHANCE = -1`,
`TERMINAL = -2`, and `PRETERM = -3`. A pre-terminal row has no more learned
decisions; completion owns its missing chance/reward resolution.

Backends expose opaque device-local `int32` state handles. `EventBatch` contains
an `int32 public_id [B]` and optional same-device payloads such as dealt cards.
Core code never decodes state handles.

`DeviceTreeBuffer` preallocates:

- `n_trajectories * node_budget` node rows in sequential modes; or
- exactly `node_budget` rows for the one branched tree.

It appends parent-major children into a CSR `TreeBatch`. Nodes retain parent,
tree id, owner, depths, state handle, and path weight; edges retain child node,
public event, and edge mass. Capacity errors are checked before successor
expansion and never truncate a path or tree.

Poker separately preallocates a structure-of-arrays public state store with the
same node capacity. Each successor copies fixed-width public tensors and stores
one parent-linked history record rather than a full history prefix.

## Reach and retained model inputs

Reach is fp32 unnormalized log likelihood `[row, player, private_type]`. Padded
types are `-inf`. The rollout cache modes are:

- `full`: retain every node reach;
- `decisions_only`: retain decisions and leaves; and
- `recompute`: retain roots and reconstruct requested rows by replaying cached
  selected-action log probabilities and chance compatibility.

`recompute` reduces persistent rollout storage, but the current replay in
`ReachCache.rows_for` is a Python recursive traversal with device scalar reads.
It materializes a dense memo arena for each call. It is not a fused GPU replay.

After targets are built, `DecisionReachCache` retains exactly the reaches needed
to reproduce optimization inputs. Games may compact this payload. Poker
currently sets `requires_full_decision_reach = True`, because folded-seat ranges
still affect preflop tokens and reach-derived features. Consequently poker keeps
dense `[decision, P, 1326]` fp32 reaches through optimization; the available
live-seat compactor is intentionally bypassed.

`PolicyCache` is aligned with decision targets. Sequential rows store one
selected-event all-type log-probability vector and one all-type value vector.
Branched rows store one selected-event vector per abstract child. Values and
log-probabilities are fp32 even when model execution uses autocast.

## Completion and target lifetime

`BatchedCompletionStats` carries posterior mass, conditional private-type
values, public values, squared-return statistics, effective sample size, and
optional diagnostics. `for_learning()` drops diagnostics not consumed by the
loss.

RB completion is streamed into `streamed_branched_value`. The reduction tracks
posterior mass `P` and posterior-weighted return mass `M=P*V`; it emits decision
targets once all children of a parent are present, then replaces those children
with one parent summary. A whole-tree `[N,P,X]` moment arena is not retained.

Before optimization, RB data is reduced to targets, the frozen policy cache, a
decision sampling table, state handles, and decision reaches. Normal data is
reduced to state/reach rows plus sampled actor type, action, scalar old outputs,
advantage, and value target.

## Logical and physical minibatches

`training.minibatch_size` is a logical number of decision rows. Poker attention
cannot necessarily execute all of them together. `plan_policy_batches` greedily
packs the logical rows under the hardware profile's padded-token and
padded-token-pair ceilings. The loss is summed over each physical sub-batch and
divided by the logical row count. Gradients accumulate across all physical
sub-batches, then one clip and one Adam/AdamW step occurs.

When `loss.c_kl > 0`, a separately frozen reference model is refreshed every
`training.kl_ref_interval` iterations. Each physical training batch performs an
additional no-grad reference forward and adds exact legal-action
`KL(current || reference)`.

## Device movement and synchronization

Poker topology, states, reaches, events, model inputs/outputs, evaluator inputs,
targets, and minibatches remain on one CUDA device. There is no intended bulk
CPU staging in the rollout/training phases. DDP ranks sample locally and
all-reduce gradients; rank zero copies data to host only for metrics and
checkpoints.

Device residency does not mean the hot path is fully asynchronous. Current
Python scheduling reads CUDA values to choose batch families, padded history
widths, fold branches, partition-cache keys, and termination. The most visible
sync points are `.tolist()`, `.item()`, `.cpu()` and tensor-to-`bool` branches in
the poker planner, history builder, partition cache, and generic rollout loops.
The current no-host-sync tests cover isolated alias/compact-map/attention
primitives, not this end-to-end scheduler.

## Memory safety and observability

Poker construction has a rollout byte estimate and training adds a structural
phase estimate. CUDA runs reserve 2 GiB and cap planned use at 82% on the local
2080 Ti profile. The first iteration records synchronized per-phase time and
allocated/reserved peaks; a 25% margin over measured reserved memory must fit
the configured safe limit.

The sequential estimates currently scale transient reaches and policy cache by
`n_trajectories * node_budget`, even though a sequential frontier has at most one
live row per trajectory and policy caches allocate only realized decisions.
This is safe but can reject useful resident trajectory counts. It is a planner
limitation, not evidence that the corresponding memory is allocated at runtime.

## Deliberate scope

Implemented objectives are scalar `ppo` for normal mode and posterior-weighted
`rb_ppo` for both RB modes, with MSE or clipped value loss, entropy scheduling,
and optional reference KL. Learned branching intensity, NeuRD/R-NaD updates,
public moment losses, and learned completion allocation are not wired into the
training path. Moment heads and auxiliary target hooks exist as research
extensions; Q targets are retained only when a selected loss requests them.
