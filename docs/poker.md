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
closure, public chance, folds, and showdown cuts. Exact sampled-showdown payoff
is returned by the sibling evaluator.

Production uses `PokerRolloutBackend` and `PokerTensorStateStore`. State handles
are device `int32` row indices into a recyclable paged structure of arrays.
Each successor receives a separately refcounted history row with binary-lifting
ancestors. Completed state slots and dead history ancestry return to their pools
at the last consumer. Model input gathers complete chronological histories in
one batched device operation. CUDA uses the preflighted lossless `(P,S)` history
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
dependency key. The sibling's folded-player partition summary lives in a sparse
paged dependency store until a fold or street change invalidates
it. Hit rows cross the evaluator facade with a false row mask and merge on the
device; no Python cache key or hit decision is required. CPU `RangeState` and
`PokerGame` remain
active support for navigation and duplicate-deal evaluation; tensor training
uses the backend's dense/compact reach objects.

Optimization currently retains every seat's decision reach because folded-seat
features remain part of the exact model input. Compaction may be enabled only
after an immutable summary can reproduce those features identically.

## Evaluator boundary

`pbg/games/poker/mc.py` binds `poker_gpu_evaluator` by rank count for:

- collision-free joint private profiles;
- posterior/completion samples and showdown values;
- deterministic folded-card partition summaries; and
- fixed-profile showdown payoffs.

The installed contract-v1 interface supports the configured compact deck. Local
code validates contract identity, combo mapping, dtype, shape, device,
padding, seeds, and numerical validity. Missing or incompatible operations fail
loudly. The sibling contract is documented at
`/home/pablob/poker-gpu-evaluator/docs/api.md`; this repository treats that
file and repository as read-only.

An RB showdown leaf sends board, ranges, nested stratum levels, and pot amounts to
completion. A fold leaf has fixed public chip payoff but still samples the
collision-conditioned private posterior. Normal mode carries one joint profile,
draws its missing board with logical randomness, and requests the evaluator's
fixed-profile payoff.

## Model bundle

`PokerModelBundle` owns:

- `PreflopNet`, operating on $R^2$ classes for every seated player; and
- `PostflopNet`, operating on board-compatible combo rows for live players.

Both emit a common factorized `PolicyOutput` and actor-private-type value rows.
The action policy combines passive/call/raise logits with a discretized logistic
raise mixture and all-in atom. Legal-event masks are applied before sampling,
log probability, entropy, and KL.

Preflop model storage and learner forwards remain on $R^2$ class rows (169 for
$R=13$); exact RB posterior/return loops use all $\binom{4R}{2}$ hands (1326 at
full deck) through the fixed quotient. Postflop storage and forwards remain on
the canonical dense hand domain, including board-blocked zero rows. No street-
specific postflop partition or structural-zero packing is used.

The hardware profile selects attention backend, query chunk, precision, and
token/token-pair caps. Family cost planning uses actual padded model work and
may split a logical block into multiple safe physical forwards without changing
logical scheduling or output order.

## Construction and memory

`bundle.py` validates $(P,S,R)$ and model configuration, constructs both
networks, configures attention, creates evaluator adapters, preflights rollout
memory, and allocates the tensor backend. Normal mode has no multi-sample
completion backend; both RB modes use tensor-native evaluator completion.

The four-phase memory plan separately predicts rollout, source/destination seal
overlap, targets, and uncached optimization. Optional reference caches are
admitted only after the uncached singleton path fits. Sequential width is a
common per-GPU root count chosen before draws; there is no retry, reroll, reach
replay, or decision-throughput root target.

## Validation

CPU tests cover rule/navigation invariants and model/evaluator boundary errors.
Tensor tests compare public transitions and policy routing. Real sibling tests
cover compact geometry, collision-free profiles, folded posterior, completion,
and showdown. CUDA gates cover residency, logical RNG invariance, all three
modes, memory safety, old-policy identity, and teardown.
