# Synthetic games

The synthetic package provides configurable finite card games that exercise the
same generic rollout, posterior, and training code as poker while remaining
small enough for exact analysis.

## Configurable game family

`SyntheticDialConfig` is the internal rule description: player count, deck and
suit counts, private hand size, betting rounds, public cards per round, raise
cap, per-round raise-size menus, and ante. Private hands are indexed
colexicographically. Deals are sampled without card collisions.

The game is a capped discrete-raise state machine. Its public state table stores
round, board, acting player, live players, commitments, current bet, raises,
checks, and public history in deduplicated structure-of-arrays form. Legal
events are fold, passive/check-call, and every configured raise increment when
the cap permits. Public
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
- `dial(card_scale, action_scale)` is the only scalable family and has exactly
  two axes. The card axis grows the deck, private type count, and board-chance
  support. The action axis grows streets, the raise cap (depth), and the number
  of raise sizes (width). Player count is fixed at two so it is not a hidden
  third dial.

The anchors are exact: `dial(1, 1) == kuhn()` and
`dial(2, 2) == leduc()`. Valid grid points preserve one shared rules engine and
showdown law; action scales are capped at four poker streets.

The YAML presets in `configs/` duplicate these structures as complete training
configurations.

## Model and policy adapter

Synthetic poker defaults to `SyntheticMHANet`, a scaled version of the binding
poker design. It runs attention over every player/type row plus a lossless
public-history token and discrete action anchors. Key logits receive the
untempered normalized log-reach prior; every layer has exact card-overlap,
identical-type, and own-range incidence channels. Folded-player rows are masked
from attention and integrated through a compact per-card pressure summary. The
discrete raise menu uses one learned anchor per game event rather than NLHE's
continuous tick head. `model.architecture: mlp_baseline` explicitly selects the
former two-layer `SyntheticNet`; it is not the default poker model.

`SyntheticPolicyAdapter` implements normalized selected-event log probability,
exact categorical entropy, KL, and centered logits. `export_policy_tables`
propagates the model's own reach factors down an enumerated public tree before
returning float64 legal policy tables for exact evaluation.

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
