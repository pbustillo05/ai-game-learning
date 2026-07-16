# Poker implementation

The poker package implements public no-limit Hold'em navigation, likelihood
reaches, preflop/postflop policy-value networks, tensor rollout, and adapters to
the sibling evaluator. It does not implement private-hand ranking,
collision-conditioned assignment, folded-card partition mathematics, or
showdown comparison locally.

## Run geometry

Each run fixes $(P,S,R)$: 2--6 players, equal 8--128 bb stacks in 0.5 bb
increments, and the highest 4--13 ranks with four suits. The active deck has
$D=4R$ cards and must satisfy $2P+5<D$. Blinds are 0.5/1 bb with no ante or
rake. Equal stacks exclude side pots.

`DeckGeometry` owns $H={D\choose2}$ lexicographic combos, $R^2$ preflop
pair/suited/offsuit classes, compact/global card maps, combo/class maps, board
conflicts, and category statistics. The full-deck case has $D=52$, $H=1326$,
and 169 classes.

Public actions are fold 0, check/call 1, and affine raise-to ticks from 2. The
0.5 bb chip unit gives `n_ticks = 2*stack_bb`. Chance events have disjoint
flop/turn/river labels plus dealt-card payloads.

## Public rules and tensor state

`NLHEEngine` is the append-only CPU rule oracle. It covers blind/button order,
fold/check/call/raise-to, minimum full raises, short-all-in reopening, street
closure, public chance, folds, and showdown cuts. It assembles payoff only from
externally supplied winner shares.

Production uses `PokerRolloutBackend` and `PokerTensorStateStore`. State handles
are device `int32` row indices into a preallocated structure of arrays. Each
successor stores fixed public fields, one history event, and a binary-lifting
ancestor row. Model input gathers complete chronological histories in one
batched device operation. CUDA uses the preflighted lossless `(P,S)` history
bound as a fixed padded width and does not read a realized length on the host.

The backend accepts mixed rows, classifies them into `pre` or `post_A`, builds
family-specific model input, and scatters transition/output rows back to caller
order. Street is never a logical family key.

## Reach and folding

Reach rows are fp32 unnormalized log likelihoods `[B,P,H]`. A selected player
action adds its all-private-type log probability to the actor row. Public cards
mask every conflicting combo. Gauge normalization happens only at model or
evaluator boundaries.

Fold successors retain board-at-fold, public line features, and a partition
dependency key. The sibling's folded-player partition summary lives in a fixed,
memory-planned device-indexed cache until a fold or street change invalidates
it. Hit rows cross the evaluator facade with a false row mask and merge on the
device; no Python cache key or hit decision is required. CPU `RangeState` and
`PokerGame` remain
active support for navigation and duplicate-deal evaluation; tensor training
uses the backend's dense/compact reach objects.

Optimization currently retains every seat's decision reach because folded-seat
features remain part of the exact model input. Compaction may be enabled only
after an immutable summary can reproduce those features identically.

## Evaluator boundary

`pbg/games/poker/mc.py` calls `poker_gpu_evaluator.pbg` for:

- collision-free joint private profiles;
- posterior/completion samples and showdown values;
- deterministic folded-card partition summaries; and
- fixed-profile showdown shares.

The installed `pbg-v3` interface supports the configured compact deck. Local
code validates API version, combo mapping, metadata, dtype, shape, device,
padding, seeds, and numerical validity. Missing or incompatible operations fail
loudly. The sibling contract is documented at
`/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`; this repository treats that
file and repository as read-only.

An RB showdown leaf sends board, live/dealt masks, reaches, and pot strata to
completion. A fold leaf has fixed public chip payoff but still samples the
collision-conditioned private posterior. Normal mode carries one joint profile,
draws its missing board with logical randomness, requests fixed-profile shares,
and assembles the sampled reward.

## Model bundle

`PokerModelBundle` owns:

- `PreflopNet`, operating on $R^2$ classes for every seated player; and
- `PostflopNet`, operating on board-compatible combo rows for live players.

Both emit a common factorized `PolicyOutput` and actor-private-type value rows.
The action policy combines passive/call/raise logits with a discretized logistic
raise mixture and all-in atom. Legal-event masks are applied before sampling,
log probability, entropy, and KL.

Preflop scatters combo reach into class rows, builds private/public/history/action
tokens, applies reach-biased attention and incidence residuals, then lifts actor
class outputs to combo rows. Postflop compacts board-compatible live-player
combos, includes folded summaries and public/history tokens, and scatters output
back to the fixed combo interface.

The hardware profile selects attention backend, query chunk, precision, and
token/token-pair caps. Family cost planning uses actual padded model work and
may split a logical block into multiple safe physical forwards without changing
logical scheduling or output order.

## Construction and memory

`bundle.py` validates $(P,S,R)$ and model configuration, constructs both
networks, configures attention, creates evaluator adapters, preflights rollout
memory, and allocates the tensor backend. Normal mode has no multi-sample
completion backend; both RB modes use tensor-native evaluator completion.

The memory plan includes public state, topology, retained reaches, live
frontiers, policy cache, occurrence arena, queues, staging, completion, target
frontier, and model/optimizer reserve. Sequential `recompute` plans against
live trajectories and retained decisions; branched plans against the one-tree
node budget. CUDA construction rejects a plan that crowds out the configured
reserve before serious allocation.

## Validation

CPU tests cover rule/navigation invariants and model/evaluator boundary errors.
Tensor tests compare public transitions and policy routing. Real sibling tests
cover compact geometry, collision-free profiles, folded posterior, completion,
and showdown. CUDA gates cover residency, logical RNG invariance, all three
modes, memory safety, old-policy identity, and teardown.
