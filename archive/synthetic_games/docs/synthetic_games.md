# Synthetic games

The synthetic package supplies small finite card games that exercise the same
rollout, target, loss, chunking, and checkpoint code as poker while retaining
exact CPU oracles. It is a correctness/research domain, not a model of poker
rollout throughput.

## Game family

`SyntheticDialConfig` defines player count, deck/suit geometry, private hand
size, betting rounds, public cards per round, raise cap, per-round raise menus,
and ante. Private hands are colex-indexed and joint profiles cannot share cards.

`SyntheticPBG` builds one deduplicated public state table containing round,
board, acting player, live mask, commitments, current bet, raise count, checks,
and public history. Legal player events are fold, passive/check-call, and each
configured raise increment allowed by the cap. Chance states deal configured
public cards between rounds.

A one-player fold ending is `TERMINAL`. Final multi-player betting closes at a
`PRETERM` showdown cut. An earlier street closing enters `CHANCE`, deals its
board cards, and continues. Rewards are zero-sum chip payoffs under the
configurable rank/suit showdown rule.

Action reach updates only the acting player's private-type row. Chance reach
updates mask hands that collide with public cards. Joint posterior profiles are
enumerated when feasible and rejection-sampled otherwise.

## Presets and dial

`presets.py` constructs games from the same configuration-driven rules engine:

- `kuhn()` is standard two-player three-card Kuhn;
- `leduc()` is two-player six-card limit Leduc; and
- `dial(card_scale, action_scale)` grows card support on one axis and
  street/raise depth/width on the other.

The exact anchors are `dial(1,1) == kuhn()` and `dial(2,2) == leduc()`. The dial
fixes player count at two and caps the action axis at four streets. YAML presets
serialize complete rule/model/training configurations rather than selecting
named branches in the engine.

## Runtime boundary

`SyntheticRolloutBackend` exposes device-local state handles, reaches, sampled
events, model output, targets, and losses. Unlike poker, public-state lookup and
mutation remain a Python/CPU control plane:

- device handles are converted into Python state-table references;
- owners, legal events, and encodings are assembled from the scalar table; and
- result tensors are placed back on the requested device.

Neural work can therefore run on CUDA, but synthetic rollout is not end-to-end
GPU-native and includes host synchronization/copies around public-state
operations. It should not be used to infer poker scheduler throughput.

The backend batches every player row supplied by the current generic frontier
into one physical model call. It also detects duplicate public state handles and
evaluates each unique state once, scattering output back with an inverse map.
It has no hardware token packer because the finite models are small.

## Models and policy adapter

The default `architecture: mha` builds `SyntheticMHANet`, a scaled discrete
analogue of the poker architecture. It attends over every player/type row plus
lossless public-history and action-anchor tokens. Normalized log reach is an
untempered key bias. Each layer includes card-overlap, identical-type, and
own-range incidence channels. Folded rows are masked from attention and enter
through compact per-card pressure features.

Raise events use learned discrete anchors rather than the NLHE logistic tick
mixture. `architecture: mlp_baseline` selects the older two-layer
`SyntheticNet` explicitly.

`SyntheticPolicyAdapter` provides selected categorical log probability, exact
entropy, exact KL, and sampling. `export_policy_tables` propagates the model's
own reach factors down an enumerated public tree before producing float64 legal
policy tables for exact evaluation.

## Completion

Normal mode has no completion backend and settles the carried private profile
directly.

RB modes select one of two backends:

- `SyntheticExactCompletion` calls the guarded enumerative oracle and returns
  exact posterior/conditional-value tensors; or
- `SyntheticMCCompletion` samples joint posterior profiles and terminal
  continuations, then accumulates posterior counts and conditional rewards.

The MC implementation uses Python loops over leaves, players, and present
private types. It is intended for validation-scale games, not high-throughput
completion benchmarking. Both backends return the generic
`BatchedCompletionStats` layout consumed by the same streamed posterior-moment
reduction as poker.

## Exact analysis

`exact.py` uses CPU float64 arithmetic and explicit feasibility guards. It can
enumerate the public tree and collision-free profiles, compute exact public
posteriors/policy values, export uniform policies, compute information-set best
responses, and report total exploitability.

Best-response actions are shared across opponent profiles at one information
set, so hidden cards are not leaked to the responder. `FeasibilityError` is
raised before profile/tree enumeration exceeds configured limits. These tools
are test/evaluation oracles and are not called in production poker training.

## Construction

`pbg/games/synthetic/bundle.py` validates game and model mappings, enumerates the
reachable public keys to size the public embedding table, constructs MHA or MLP,
places the neural boundary on the requested device, and selects no/exact/MC
completion from the training mode and completion backend.
