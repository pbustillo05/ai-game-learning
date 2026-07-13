# Core algorithms

This document explains the algorithms implemented by `pbg/core`. Symbols and
formal statements are defined in [`../NOTATION.md`](../NOTATION.md) and
[`latex/core_math.tex`](latex/core_math.tex).

## Sequential and fixed-intensity rollout

`sample_trees` is the public entry point. `beta_mode: sequential` dispatches to
the one-child sampler. `beta_mode: fixed` dispatches to labeled branching.

At each frontier wave, the backend classifies states into player, chance, and
leaf rows. For player rows it evaluates the frozen model, draws a fresh joint
private profile from the public posterior, draws a public event conditional on
that profile, updates reach likelihoods, and caches the selected action's
all-type log probability. Chance rows similarly draw a fresh posterior profile
before sampling their public event. The child public states and owner codes are
then appended to the device tree buffer.

Fresh profile draws are local to event draws. Sampling one private profile for
an entire tree would correlate branches and violate the intended public-event
mixture.

For fixed intensity `beta >= 1`, the number of abstract children is randomized
rounding:

```text
K = floor(beta) + Bernoulli(beta - floor(beta))
```

Each of the `K` event draws is stored as a separate child, and each outgoing
edge receives mass `nu = 1/K`. The expected terminal distribution of a
uniformly followed child equals the policy-induced terminal distribution.
The backend chooses beta by a public-stage bucket; there is no learned
intensity controller in the current implementation.

Rollout stops only at `TERMINAL` or `PRETERM`. If the backend's declared maximum
number of waves ends with a live frontier, or if the buffer capacity is
exceeded, sampling raises instead of returning a biased partial tree.

## Reach bookkeeping

For each player `p` and private type `x`, reach stores the unnormalized sum of
log action likelihoods contributed by that player's observed actions. Chance
and compatibility updates may mask impossible types. Keeping reach
unnormalized avoids repeated normalization and preserves the numerators needed
by collision-conditioned completion.

The core treats the final tensor shape as `[node, P_max, X_max]`; games mask
padded private-type columns. `ReachCache.rows_for` either gathers stored rows or
replays public edges from roots using cached selected-action likelihoods.

## Empirical decision-node law

Every node has `path_weight`, the product of its incoming sampled-tree edge
masses. `build_decision_alias` restricts to player nodes and builds a Walker
alias table using those weights. Sampling from this table implements the
length-biased law over player decisions. In a sequential tree every path weight
is one, reducing to uniform sampling among stored player decisions after the
usual trajectory-length weighting.

Minibatching samples decision-row indices, then gathers state handles, reaches,
posterior targets, frozen log probabilities, frozen values, legal events, and
valid private-type masks with the same tensor index. No node-id lookup or
reordering occurs in the optimization loop.

## Leaf completion

The completion backend receives leaf node ids and their reach rows and returns:

- `p_hat[p,x]`: posterior mass of private type `x` for player `p`;
- `v_hat[p,x]`: conditional expected payoff given that type;
- `vvec_hat[p]`: unconditional expected payoff for each player;
- `u_hat`: expected squared reward-vector norm;
- `n_eff`: effective sample size.

Synthetic completion samples exact game profiles and terminal continuations
through the synthetic game. Poker completion delegates collision conditioning,
runouts, and showdown equity to the sibling evaluator. Public fold payoffs are
still combined with a collision-conditioned posterior.

## Posterior-moment backward pass

`branched_value` initializes every leaf with two linear tensors:

```text
P = p_hat
M = p_hat * v_hat
```

For an internal node with `K` abstract children, it computes uniform child
means:

```text
P_parent = (1/K) * sum(P_child)
M_parent = (1/K) * sum(M_child)
```

Conditional values are `Cond(M,P) = M/P` for positive mass and exactly zero
otherwise. At a decision owned by player `p`, the emitted value target is the
actor row of `Cond(M_parent, P_parent)`.

For each child `a`, the Q estimate is `Cond(M_a, P_a)`. When there is more than
one child, the action-independent baseline for that child is the conditional
value of all siblings, computed in O(K) total-minus-child form. With one child,
the frozen value head is the baseline. The emitted direct advantage is:

```text
A_a = Q_a - [lambda * sibling_value + (1-lambda) * frozen_value]
```

After emission, only the owner's return moment is bootstrapped before it flows
to the parent:

```text
M_owner <- lambda * M_owner + (1-lambda) * P_owner * frozen_value
```

Capture-before-bootstrap is important: targets at the current decision describe
its continuation before the current value head is mixed in. Chance nodes only
aggregate. Q is retained only when requested.

The implementation processes complete trees in contiguous tree-id chunks under
`max_resident_rows`; a dependency chain is never split. Leaf accumulators must
be fp32 or wider, padded types are zeroed, and configured or realized tree-size
violations raise.

## Posterior-weighted RB-PPO

The current model is compared with cached frozen rollout outputs. For abstract
child `a` and actor type `x`, the ratio is:

```text
rho(a,x) = exp(log pi_new(a|x) - log pi_old(a|x))
```

The policy element uses the usual clipped PPO surrogate with the direct
advantage emitted by the backward pass. Its weight is the child posterior mass
divided by the number of abstract children. Duplicate event labels therefore
contribute independently.

Value loss is squared error between the current actor-type value and the
emitted parent value target. Entropy is computed exactly over the complete
legal event law by the game adapter. Both are weighted by parent posterior
mass. Padded private types and padded child slots are masked explicitly.

The final per-row objective is:

```text
sum_children,sum_types child_p/K * clipped_policy_loss
+ sum_types parent_p * (c_v * value_error^2 - c_h * entropy)
```

The loss reports policy/value/entropy components, approximate KL, clip
fraction, entropy, explained variance, and maximum active ratio. Entropy can be
constant or linearly decayed over configured iterations.

## Checkpointed optimization

Each iteration uses one frozen rollout snapshot for all configured gradient
steps. Adam updates the owning model; optional CUDA autocast and gradient
scaling are used only when CUDA AMP is enabled. Gradients are norm-clipped.

Under DDP, each rank retains the configured number of trees and minibatch rows,
uses a rank-specific seed, constructs targets locally, and participates only in
gradient synchronization. Rank zero writes metrics and checkpoints.

