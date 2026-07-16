# Core algorithms

This page describes the executable algorithms in `pbg/core`. Formal definitions
are in [`../NOTATION.md`](../NOTATION.md) and
[`theory/core_math.tex`](theory/core_math.tex).

## Pipeline selection

`training.mode` and `loss.name` select one of three valid pairs:

| Mode | Required loss | `rollout.n_trees` means | Completion |
|---|---|---|---|
| `normal` | `ppo` | trajectories per rank | none |
| `rao_blackwell` | `rb_ppo` | trajectories per rank | required |
| `branched_rao_blackwell` | `rb_ppo` | exactly one sampled tree per rank | required |

`rollout.beta` must be null in both sequential modes. Branched mode requires a
finite fixed table with `default` and values at least one.

## Sequential rollout

`_sample_trees_sequential` starts `G = rollout.n_trees` independent roots. It
draws one collision-free joint private profile per root and carries that profile
unchanged along the trajectory. It also carries one current log-reach tensor per
trajectory.

The sampler is breadth-synchronous across trajectories. During each public-step
wave it:

1. removes terminal and pre-terminal rows from the active frontier;
2. separates player rows from chance rows;
3. evaluates all current player rows through one backend `policy_forward` call;
4. samples one public action using the stipulated acting private type;
5. stores the selected action's log probability for every possible actor type
   and stores every-type value output;
6. updates the acting player's reach likelihood;
7. samples chance events and masks incompatible private types; and
8. appends exactly one successor per active row.

The backend may split the logical player batch into smaller physical model
batches. The core itself does not maintain pending model queues across waves.

Rollout ends only at `TERMINAL` or `PRETERM`. A live frontier after
`max_rollout_steps`, or a child expansion beyond the preallocated capacity,
raises instead of returning a partial sample.

### Normal targets

Normal mode settles each leaf using its carried sampled private profile. Public
folds use the already stored payoff; poker showdowns draw missing board cards
and request evaluator winner shares for that fixed profile.

For every decision, the dataset keeps only the sampled actor type, sampled
event, scalar old log probability, scalar old value, scalar advantage, and
scalar value target, plus state/reach input. Starting from the terminal player
reward, decisions are visited backward on each owner's decision clock:

```text
q_t = current carry for the acting player
target_t = q_t
advantage_t = q_t - old_value_t
carry_actor = lambda * q_t + (1-lambda) * old_value_t
```

No completion posterior or private-type target tensor is constructed.

### Sequential RB targets

Sequential RB uses the same one-child topology but completes every leaf with a
collision-conditioned posterior sample. The generic posterior-moment reduction
then emits an all-actor-type posterior, advantage, and value target at each
decision. With one child there is no sibling baseline, so the frozen value is
the policy baseline.

## Fixed labeled branching

`_sample_branched_labeled` owns one root. At each live node the backend selects a
fixed public-stage intensity `beta >= 1` and draws:

```text
K = floor(beta) + Bernoulli(beta - floor(beta))
```

Every one of the `K` draws becomes a separate abstract child with edge mass
`1/K`. Equal public event labels are not merged: they retain independent private
profile draws, descendants, completion draws, and target slots.

The implementation evaluates policy/value once per unique player parent in the
current frontier, gathers that output for its abstract children, and draws a
fresh posterior joint profile per child. This preserves sibling independence.
Chance nodes also draw independent profiles/events per child.

The sampler is still wave-based. Early frontiers, including the root, can have
too few ready decisions for an efficient model batch. Branching broadens later
frontiers but there is no cross-depth ready queue and no overlap across sampled
trees.

## Reach bookkeeping and replay

For player `p` and private type `x`, reach is the sum of log likelihoods of that
player's observed actions. Chance compatibility sets impossible types to
`-inf`; reaches are never normalized in rollout.

The cache shape is padded `[row, P_max, X_max]`. `full` and `decisions_only`
gather stored rows directly. `recompute` stores root rows and replays each public
edge:

- a player edge adds the selected all-type log-probability row cached at its
  parent; and
- a chance edge calls the game replay hook. Poker derives dealt cards from the
  successor state and reapplies board compatibility.

The current `recompute` implementation is memory-oriented, not throughput-
oriented. `rows_for` converts requested ids into a Python traversal, recursively
reads parent/CSR scalars, and allocates a dense memo tensor for the current call.
Normal and RB target construction eventually materialize all decision reaches.

## Completion microbatches

RB modes require `completion.n_samples`. The default completion row count is:

```text
min(number of leaves, 256, byte-budget-derived limit)
```

For each microbatch the loop reconstructs/gathers leaf reaches, calls the
game-owned completion backend, strips non-learning diagnostics, and yields the
result directly to backward reduction. The previous completion object and its
leaf reaches become unreachable before the next microbatch.

Completion returns at least:

- posterior mass `p_hat[p,x]`;
- conditional value `v_hat[p,x]`;
- unconditional player value vector `vvec_hat[p]`;
- expected squared reward norm `u_hat`; and
- effective sample size `n_eff`.

Synthetic completion may be exact or Monte Carlo. Poker delegates joint
collision conditioning, runouts, and equity to the sibling evaluator. A public
fold still samples the collision-conditioned joint profile so its private-type
posterior is correct even though its public chip reward is already fixed.

## Streamed posterior-moment reduction

Each completed leaf initializes linear moments:

```text
P = p_hat
M = p_hat * v_hat
```

For a parent with `K` abstract children:

```text
P_parent = (1/K) * sum(P_child)
M_parent = (1/K) * sum(M_child)
Cond(M,P) = M/P when P > 0, otherwise 0
```

At a decision owned by player `p`, the emitted value target is the actor row of
`Cond(M_parent, P_parent)`. Child `a` has Q estimate `Cond(M_a, P_a)`. For
`K > 1`, its baseline is the conditional value formed from every sibling except
`a`, computed with total-minus-child sums. For `K = 1`, its baseline is the
frozen value head.

The direct child advantage is:

```text
A_a = Q_a - [lambda * sibling_baseline_a + (1-lambda) * frozen_value]
```

After targets are captured, only the owner's return moment is bootstrapped
before propagation:

```text
M_owner = lambda * M_owner + (1-lambda) * P_owner * frozen_value
```

Chance nodes only reduce child moments. The implementation keeps a frontier of
completed regions whose parents still await siblings; it does not retain one
whole-tree `[N,P,X]` `P/M` arena. Target and frontier allocations are checked
against a byte budget before reduction.

## Decision-row distributions

For RB data, every tree node stores the product of incoming edge masses.
`build_decision_alias` restricts to player nodes and builds a device-local Walker
alias table from that path mass. With-replacement RB minibatches therefore
sample the empirical length-biased decision law. In a sequential tree all path
weights equal one, so this reduces to uniform sampling over pooled decision
rows.

Normal mode samples its flat decision dataset uniformly.

When `training.ppo_epochs` is set, normal/sequential RB bypass with-replacement
sampling. Each resident chunk draws a device `randperm` of its decision rows and
uses every row once per epoch. Branched RB rejects `ppo_epochs` because its
decision law is path-weighted rather than uniform.

## Minibatch alignment and physical splitting

Every target/cache object has one stable decision-row order. Gathering a row
index selects state handle, reach, target, old policy/value, legal events, and
valid private types with the same tensor index. Alignment mismatches raise
before optimization.

A logical minibatch has at most `training.minibatch_size` rows. The game backend
may divide it into hardware-bounded physical batches. For every physical batch:

1. gather the already selected rows;
2. run the current model under configured autocast;
3. optionally run the no-grad KL reference model;
4. sum the loss and backpropagate `loss_sum / logical_row_count`.

After every physical sub-batch has accumulated gradients, DDP gradients are
all-reduced explicitly, unscaled, clipped, and applied in one Adam or AdamW
step. Thus a physical split changes execution shape, not the logical optimizer
step or loss normalization.

## Normal PPO

For the sampled actor type, normal PPO computes:

```text
ratio = exp(log_pi_new(action) - log_pi_old(action))
policy = -min(ratio * advantage, clip(ratio) * advantage)
```

The value term is MSE or PPO-style clipped value error. Entropy is exact over
the complete legal action row. If enabled, the reference term is exact
`c_kl * KL(pi_new || pi_reference)` over legal actions. These scalar terms are
averaged over logical decision rows.

## Posterior-weighted RB-PPO

For abstract child `a` and actor type `x`:

```text
ratio[a,x] = exp(log_pi_new[a,x] - log_pi_old[a,x])
```

The clipped policy element is weighted by `child_p_hat[a,x] / K`; duplicate
children contribute independently. Value, entropy, and optional reference KL
are weighted by `parent_p_hat[x]`. Invalid padded types and padded child slots
are masked.

Per decision row the objective is:

```text
sum[a,x] child_p_hat[a,x]/K * clipped_policy[a,x]
+ sum[x] parent_p_hat[x] * (
      value_loss[x] - entropy_coefficient * entropy[x]
      + c_kl * KL(new || reference)[x]
  )
```

The loss reports policy/value/entropy/reference components, approximate PPO KL,
clip fraction, entropy, explained variance, and maximum active ratio.

## Snapshots, chunks, and epochs

The old-policy snapshot is a deep device copy of the current model in eval mode.
For a non-chunked RB iteration, it is deleted immediately after rollout and
before completion. In normal mode it becomes unreachable when scalar target
construction returns. For a chunked iteration, one snapshot is shared by all
chunks and survives their optimization phases.

Let chunk `i` contain `D_i` decisions, logical batch size `B`, and `E` PPO
epochs. The implementation performs:

```text
optimizer_steps = E * sum_i ceil(D_i / B)
optimized_rows = E * sum_i D_i
```

The final minibatch of each chunk is not padded; `decision_epoch_coverage` is
therefore exactly one per requested epoch. This differs from a single global
dataset, which would take `E * ceil(sum_i D_i / B)` optimizer steps and would
mix rows across chunks.

## Memory planning and measured calibration

Training writes an analytical structural estimate before launch and records
synchronized peak allocated/reserved memory for snapshot, rollout, completion,
backward, compaction, and optimization on the first CUDA iteration. A 25%
margin over the measured reserved peak must fit the hardware safe limit.

The current sequential analytical estimate is intentionally conservative to a
fault: it scales transient reaches and policy-cache bounds with
`n_trajectories * node_budget`. The actual sequential frontier is bounded by
`n_trajectories`, and actual policy storage is bounded by realized decisions.
This distinction is central to the current efficiency audit.
