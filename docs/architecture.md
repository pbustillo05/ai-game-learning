# Architecture

## Scope and composition

`pbg` trains equal-stack configurable NLHE cash-poker policies in three modes:

| `training.mode` | Public rollout | Target | Loss |
|---|---|---|---|
| `normal` | sampled private profile and one path per start | sampled scalar lambda return | `ppo` |
| `rao_blackwell` | sampled profile/path cut at completion leaf | posterior mass and actor-type value | `rb_ppo` |
| `branched_rao_blackwell` | one labeled sampled tree per rank | posterior-moment sibling targets | `rb_ppo` |

`pbg.config` and `pbg.runtime` are the NLHE composition roots. Generic modules
under `pbg.core` never import a game. `pbg.games.poker` implements the core
protocols, and `pbg.eval` owns read-only duplicate-deal evaluation. There is no
game registry or rollout implementation selector.

Poker owns public betting, chance navigation, action policy, likelihood
reaches, pot accounting, and model input. The sibling `poker_gpu_evaluator.pbg`
owns joint private profiles, collision-conditioned completion, folded-card
partition summaries, and showdown shares. Local adapters validate its API and
never implement a fallback hand evaluator.

## One iteration

```text
trainable model
  -> read-only synchronous rollout policy
  -> family-scheduled rollout
       append-only occurrences + natural partition inputs + old policy/value
  -> normal: sampled terminal settlement and compact scalar targets
     RB: streamed evaluator completion into the P/M value frontier
  -> exact seal into immutable TrainingStorage; release topology/reaches
  -> index-only stratified draws by global log mass
  -> bounded gather/forward/backward/release subspans
  -> flat-gradient SUM, common finite gate, clip, optimizer step
```

All completion and target batches belong to one fixed rollout. No trajectory
chunk can introduce a separately normalized objective. `ppo_epochs` shuffles
the complete retained decision set; otherwise `gd_steps_per_iter` draws logical
minibatches with replacement.

## Family execution

`pbg/core/execution.py` owns transactional byte accounting, a chunked occurrence
arena, bounded work queues, stable family routing, staging slabs, and logical
randomness. Sequential and branched samplers both schedule poker's `pre` and
`post_A` families. Different streets may share one `post_A` family block.

The scheduler admits trajectories/offspring only after reserving their bytes.
It repeatedly drains a family block, evaluates its exact current model inputs,
samples addressed events, appends successors, and eagerly closes forced/chance
transitions. The backend may split a logical block by actual padded model cost,
then scatters results to logical order. There is no synchronized wavefront or
fixed-quantum alternative.

Sequential modes carry one active occurrence and private profile per live
trajectory. Branched mode starts one root/tree per rank and stores every
randomized-rounded abstract child, including repeated event labels.

## State, topology, and reach

Owner codes are players $\ge0$, `CHANCE=-1`, `TERMINAL=-2`, and `PRETERM=-3`.
State references are opaque device `int32` handles. `DeviceTreeBuffer` retains
parent-major CSR topology, tree id, owner, depths, state handle, path weight,
child label, and empirical edge mass. Capacity overflow never truncates.

Poker public state is a preallocated structure of arrays with one history event
and a binary-lifting ancestor row per state. Model input gathers chronological
history in one batched device operation at the preflighted lossless width.
Reach is fp32 unnormalized log likelihood
`[row, player, private_type]`; actor actions add selected log likelihood and
board events mask conflicting hands.

There is no selectable reach-storage mode or production reach replay. A
generated decision input and stopped-leaf reach are written once for their last
consumer. The seal boundary creates exact partition destinations; optimization
retains no topology, mutable occurrence, completion row, or all-node reach memo.

Natural old log probabilities and values align with sealed partition rows.
The first update after launch/resume recomputes these quantities and gates
optimization on exact old-policy identity.

## Completion and values

Normal mode carries one sampled joint profile to a terminal reward. RB modes
stream completion batches from the sibling evaluator. Diagnostic completion
fields are summarized and released before learning.

`streamed_branched_value` maintains only a contracted boundary of posterior
mass $P$ and posterior-weighted moment $M=P V$. Once every child of a parent is
available, it emits conditional values, child posterior masses, and
leave-one-out advantages, applies owner-only lambda bootstrap, and replaces the
siblings with their parent summary. The byte plan is checked before allocation.

## Logical and physical learner work

`training.minibatch_size` is statistical decision exposure. Each nonempty
partition receives an arbitrary positive count; physical shapes are separately
bounded beneath hardware token and token-pair ceilings. One exact tail is
allowed and retains the original stratum normalization.

Normal/sequential decisions have mass one; branched decisions use a Walker
sampler built from fixed path weights. Fp64 log masses are exchanged in common
order and each local stratum is scaled by its exact global mass. Ranks SUM one
flat fp32 gradient buffer without DDP averaging.

When reference KL is enabled, an immutable reference model is refreshed on its
configured interval, evaluated without gradients, and checkpointed.

## Device movement and memory

Public state, topology, reaches, events, model/evaluator inputs and outputs,
targets, and learner tensors stay on the owning CUDA device. Folded summaries
use a fixed device-indexed cache and masked evaluator rows. Declared scheduler
control transfers—including the one fixed mixed-row policy-planning block—are
measured; undeclared synchronization is separately counted and forbidden by
acceptance gates.

The hardware profile's precision applies consistently to frozen rollout-policy,
identity-replay, and learner forwards. Completion and posterior/value reduction
remain fp32 outside autocast.

The NLHE bundle preflights rollout storage. Training adds execution, completion,
target, and optimizer plans. Hardware profiles cap attention work, memory
fraction, and reserve. First-iteration calibration reports allocated/reserved
peaks per phase and fails if measured use exceeds the safe plan.

Sequential recompute planning charges live trajectories and realized retained
work rather than a capacity-sized reach frontier for every possible node.

## Reproducibility and lifetime

Logical RNG keys depend on seed/iteration, global trajectory or rank-tree
identity, abstract path, child ordinal, and draw purpose—not queue order or
physical batching. Strict mode additionally configures deterministic Torch
backends.

Atomic checkpoints retain scientific configuration, model, optimizer, scaler,
explicit per-rank RNG state, run id, and optional reference policy. Resume
requires a matching configuration/world size and exact gap-free metrics prefix.
Rollout-only objects are released before the next iteration; post-warmup
reserved memory must remain bounded in soak tests.
