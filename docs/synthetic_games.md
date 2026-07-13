# Synthetic games

The synthetic package provides configurable finite card games that exercise the
same generic rollout, posterior, and training code as poker while remaining
small enough for exact analysis.

## Configurable game family

`SyntheticDialConfig` controls independent public and private complexity:
player count, deck and suit counts, private hand size, betting rounds, public
cards per round, raise cap, round bet sizes, and ante. Private hands are indexed
colexicographically. Deals are sampled without card collisions.

The game is a capped limit-betting state machine. Its public state table stores
round, board, acting player, live players, commitments, current bet, raises,
checks, and public history in deduplicated structure-of-arrays form. Legal
events are fold, passive/check-call, and raise when the cap permits. Public
chance events deal configured board cards between rounds.

A fold ending is terminal immediately. A closed betting round followed by more
betting is not pre-terminal: chance must deal the next public cards and play
continues. Only the state after final betting is a completion leaf. Showdown
strength is derived from the configurable rank/suit representation, and rewards
are zero-sum chip payoffs.

Reach updates multiply only the acting player's type row by the observed
action probability and mask types inconsistent with public cards. Posterior
profile sampling can enumerate the collision-conditioned distribution when
feasible or use rejection otherwise.

## Presets

`presets.py` constructs games entirely from configuration rather than named
rule branches:

- `kuhn()` is standard two-player, three-card Kuhn poker.
- `leduc()` is two-player, six-card limit Leduc with a public-card betting
  round.
- `dial(public_scale, private_scale, n_players)` grows public-tree and private
  type complexity independently.

The YAML presets in `configs/` duplicate these structures as complete training
configurations.

## Model and policy adapter

`SyntheticNet` embeds an encoded public-state bucket and every private type,
passes the combined representation through a small MLP, and emits categorical
event logits plus an actor-type value. Legal masks are applied to the full event
row.

`SyntheticPolicyAdapter` implements normalized selected-event log probability,
exact categorical entropy, KL, and centered logits. `export_policy_tables`
evaluates a model over an enumerated public tree and returns float64 legal
policy tables for exact evaluation.

## Rollout and completion adapters

`SyntheticRolloutBackend` wraps the scalar mathematical game in the generic
batched protocol. State handles remain CPU tensors, profiles come from the
game's posterior sampler, and beta is chosen from fixed public round/chance
buckets.

`SyntheticCompletion` completes selected leaves by sampling posterior profiles
and terminal continuations, then forms private-type posterior and value
statistics in the generic `BatchedCompletionStats` layout. It is intended for
smoke runs and contract validation rather than large-scale performance.

## Exact analysis

`exact.py` uses CPU float64 arithmetic and hard feasibility guards. It can:

- enumerate the complete public tree;
- enumerate collision-free private profiles;
- compute exact public-node posteriors;
- create uniform policy tables;
- evaluate a policy recursively; and
- compute per-player best responses and total exploitability.

Best-response actions are shared across opponent profiles at the same
information set, so the calculation respects imperfect information instead of
granting the responder access to hidden cards. The exact tools deliberately
raise `FeasibilityError` before profile/tree work exceeds their guard rather
than silently becoming an impractical solver.

These tools serve as oracles for public-tree rules, posterior sampling,
terminal laws, and checkpoint exploitability. They are not in the production
training hot path.

