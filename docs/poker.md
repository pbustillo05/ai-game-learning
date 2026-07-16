# Poker implementation

The poker package implements public no-limit hold'em navigation, range
likelihoods, the preflop/postflop policy-value networks, CUDA rollout, and
adapters to the sibling evaluator. It does not implement private hand ranking,
collision-conditioned assignment, folded-card partition mathematics, or
showdown equity locally.

## Card and action representations

Each run fixes `(P,S,R)`: 2--6 players, equal 8--128 bb stacks in 0.5 bb
increments, and the highest 4--13 ranks in all four suits. The active deck has
`D=4R` cards and keeps global IDs `52-D..51`; `R=4` is therefore AKQJ only.
The deal must satisfy `2P+5 < 4R`.

`DeckGeometry` owns the compact representation: `H=C(D,2)` lexicographic
two-card combos, `R²` pair/suited/offsuit preflop classes, compact/global card
maps, combo/class maps, board conflicts, and exact category statistics. The
standard deck remains the special case `D=52`, `H=1326`, and `R²=169`.

The installed sibling evaluator still exposes only the fixed-width `pbg-v2`
contract. The decision-complete compact `pbg-v3` request is
[POKER_GPU_EVALUATOR_COMPACT_DECK_REQUEST.md](todo/POKER_GPU_EVALUATOR_COMPACT_DECK_REQUEST.md).
Reduced-deck evaluator integration must not emulate poker math locally.

Public action ids are:

- `FOLD = 0`;
- `CHECK_CALL = 1`; and
- `RAISE_OFFSET + tick` for affine raise-to ticks.

Blinds and denomination are fixed at `0.5/1/0.5` bb. `PokerConfig` derives
`n_ticks = 2*stack_bb`; a 10 bb stack gives 20 raise ticks and 22 total policy
events.

### Cumulative five-card category odds

Categories mean that a five-card subset *contains* their minimal essence: a
full house contains trips, two pair, and pair; a straight flush contains a
straight and flush; and high card is always present. For `T=C(4R,5)` and
`q=max(0,R-4)+1[R=13]`, exact numerators are:

| Category | Numerator |
|---|---:|
| high card | `T` |
| pair | `T - 4^5 C(R,5)` |
| two pair | `144(R-2)C(R,2) + 24R(R-1)` |
| trips | `64R C(R-1,2) + 28R(R-1)` |
| straight | `4^5 q` |
| flush | `4C(R,5)` |
| full house | `24R(R-1)` |
| quads | `4R(R-1)` |
| straight flush | `4q` |

Increasing strength follows decreasing probability. Strict crossings reverse
conventional categories; exact ties retain conventional order; impossible
categories are omitted. Strongest-to-weakest feasible orders are:

- `R=4`: quads, full house, trips, two pair, pair, high card.
- `R=5`: straight flush, flush, quads, full house, straight, trips, two pair,
  pair, high card.
- `R=6..7`: straight flush, flush, quads, full house, straight, trips, two pair,
  pair, high card.
- `R=8..11`: straight flush, quads, flush, full house, straight, trips, two
  pair, pair, high card.
- `R=12..13`: conventional order.

At `R=4`, straight and flush are impossible while pair and high card both have
probability one. At `R=5`, every flush is a straight flush. Flush/quads tie at
`R=8` (224 subsets each), and flush/full-house tie at `R=12` (3,168 each).

Chance events use disjoint public-label ranges for flop, turn, and river plus a
padded `cards [B,3]` payload. The tree stores only the compact public label;
reach replay recovers dealt cards from the successor state row.

## CPU oracle and tensor backend

`NLHEEngine` is an append-only CPU reference state machine. It implements blind
posting, button/action order, fold/check/call/raise-to, minimum full raises,
short-all-in reopening, street closing, public chance transitions, folds,
showdown cuts, and nested side-pot strata. It assembles chip payoff from supplied
winner shares and never compares private hands.

Production training uses `PokerRolloutBackend` and
`PokerTensorStateStore`. The state store is a preallocated device
structure-of-arrays. Handles are `int32` row indices and allocation is a monotone
device prefix sum. Each successor copies fixed public fields and stores one
parent-linked history event containing:

- actor, street, and action class;
- amount and pot before the action;
- call/current-bet/min/max-raise values; and
- one history-parent handle.

Model input reconstructs the chronological history by following these links
into padded tensors. This avoids copying a full history prefix into every state,
but the current maximum-history-width read is a CUDA-to-Python synchronization.

`step` supports mixed player and chance rows in one generic batch. It partitions
by the stored phase, applies the two transition kernels, and scatters successors
back into input order.

## Dense reach state

Poker rollout reach is fp32 `[B, P, 1326]` unnormalized log likelihood.
At a player action, the selected public-event log probability for every possible
actor combo is added to that actor's row. At a chance event, every combo
containing a dealt board card is set to `-inf`. Rows are gauge-normalized to
weights only at model/evaluator boundaries.

The older `RangeState`/`FoldedPlayer` classes remain useful for CPU navigation
and range tests. Tensor training uses reach tensors plus fold metadata stored in
the state store. On a fold, the successor records board-at-fold, line features,
and a partition-cache dependency key before changing the live-player layout.

The current optimization input contract retains all seat ranges, including
folded seats. `PokerRolloutBackend.requires_full_decision_reach = True` disables
the otherwise available ragged live-seat `CompactPokerDecisionReach`. Every
retained decision therefore costs `P * 1326 * 4` bytes for reach alone (31,824
bytes at six seats). This is necessary until the model-input contract can
reconstruct every folded contribution from an immutable summary cache.

## Evaluator boundary

`mc.py` lazily imports `poker_gpu_evaluator.pbg` operations:

- `PokerGPURejectionSampler` draws complete collision-free joint profiles;
- `PokerGPUCompletion` estimates collision-conditioned hand posterior and
  showdown values; and
- `PokerGPUShowdownEvaluator` returns exact winner/tie shares for fixed private
  profiles and boards.

`folded.py` maps local fold snapshots to the evaluator's deterministic partition
summary. The returned per-card/per-hand fields are opaque model features.

All calls validate facade version, combo ordering, shape, dtype, device,
padding, seeds, metadata, and numerical validity. Inputs and outputs stay on the
rollout device. A missing or incompatible operation raises; there is no NumPy
or local-poker-math fallback. The binding interface is
[`../archive/POKER_GPU_EVALUATOR_IO.md`](../archive/POKER_GPU_EVALUATOR_IO.md).

### Fold and showdown completion

An RB showdown leaf calls evaluator completion with the state board, live/dealt
masks, dense reaches, and public pot strata. A fold leaf already has a fixed
public chip payoff, but still requests collision-free joint profiles to estimate
the correct private-type posterior.

Normal mode carries one sampled joint profile. At showdown it draws the missing
public board, asks the evaluator for fixed-profile shares, and assembles the
reward; it does not invoke multi-sample completion.

## Model bundle

`PokerModelBundle` owns two networks with one common `PolicyOutput`:

- `PreflopNet` runs on 169 class rows for every seated player; and
- `PostflopNet` runs only board-compatible combo rows for live players.

Both output policy parameters and value for every possible private row of the
acting player. Rollout samples one stipulated actor combo only after the full
law has been computed. Training integrates the emitted rows with scalar or
posterior weights depending on the mode.

### Preflop input and compute

Dense combo likelihood weights are scatter-added into 169 classes per seat.
The network constructs `P * 169` private tokens plus public scalar/history tokens
and action anchors. For six players the base sequence is about 1,095 tokens plus
history. Pre-layer-normalized attention receives the normalized log reach as an
untempered key bias. Incidence residuals add card-overlap/identical-type
structure. The 169 actor outputs are lifted back to 1326 combo rows before the
core caches or samples them.

### Postflop input and compute

Flop/turn/river have 1176/1128/1081 board-compatible combos. The backend groups
rows by street and number of non-folded players, supplies stable compact
gather/scatter maps, and the model attends over:

```text
live_players * compact_combos + public/history tokens + action anchors
```

Only live-player private rows enter attention. Folded players enter through
state metadata and evaluator partition summaries. Outputs are scattered back to
fixed 1326-combo rows for the generic policy/target interface.

The partition summary is cached by a state dependency anchor. Ordinary actions
on one street reuse it; a fold or street transition invalidates it. The cache is
a Python dictionary of device tensors and currently copies cache keys to CPU to
select misses.

### Attention and heads

`MHA` has explicit, checkpointed query-chunk, and FlexAttention paths. The local
hardware profile uses `attention_backend: auto`; it attempts FlexAttention and
falls back to exact query-chunk attention. Learned positive temperatures scale
content scores only; the log-reach key bias remains untempered.

The policy factorization produces passive/call/raise-type logits and a
conditional raise-tick law. `RaiseMixtureHead` is a normalized mixture of
discretized logistic components plus an all-in atom. `PokerPolicyAdapter`
conditions on the supplied legal event row and provides probabilities,
selected-event log probability, sampling, exact entropy, and exact KL. Its
logit-space fallback prevents all legal mass from underflowing when an illegal
type logit dominates.

Detailed equations and token features are in
[Poker player model](theory/MHA_model.md).

## Current rollout batching: frontier packing, not persistent queues

The generic sampler calls `policy_forward` once for the player decisions ready
in the current rollout wave. With a hardware profile, poker then:

1. computes street, live-player count, history length, and estimated token
   length for those rows;
2. assigns preflop to family 0 and postflop to a family keyed by live-player
   count;
3. sorts a family by token length and greedily packs rows under both
   `max_tokens` and `max_token_pairs`;
4. runs preflop plans first, then postflop live-count plans; and
5. inside each direct postflop plan, splits again by street because compact
   combo widths differ.

There are no persistent preflop/postflop queue objects. Rows do not wait across
frontier waves for a full batch; no high-water mark, low-water mark, forced
drain, double buffering, CUDA stream overlap, or cross-chunk queue exists. The
test historically named “shared postflop queue” checks routing within one call,
not a long-lived scheduler.

For the local 2080 Ti profile, the configured ceilings imply approximate
maximum physical batch sizes (history can reduce a boundary by one):

| Family | Sequence length | Inference max | Training max |
|---|---:|---:|---:|
| preflop, 6 seats | about 1.1k | 14 | 7 |
| postflop, 2 live | about 2.3–2.4k | 6 | 3 |
| postflop, 3 live | about 3.3–3.6k | 4 | 2 |
| postflop, 4 live | about 4.4–4.8k | 3 | 1 |
| postflop, 5 live | about 5.5–6.0k | 2 | 1 |
| postflop, 6 live | about 6.6–7.1k | 2 | 1 |

Thus a logical training minibatch of 128 nodes is intentionally accumulated
over many physical forward/backward calls before one optimizer step. A
persistent scheduler can remove partially filled calls, but it cannot make
six-live postflop execute as a 128-row physical batch on this architecture.

## Branching and batching

Branched rollout repeats active parents by offspring count for child-local
profile/action sampling, but evaluates the model only for the unique player
parents in that frontier. Poker applies the same frontier packer. The root call
is necessarily a one-row forward and low-beta trees may stay narrow. Current
branching has no cross-depth ready queue, so it is only partially compatible
with the desired queued execution design.

## Memory allocation and movement

`build_bundle` sizes the poker state store and tree arena from:

```text
resident_trajectories * node_budget
```

where `resident_trajectories` is `trajectory_chunk_size` when present. With
`reach_storage: recompute`, only root reaches persist during rollout, while
current/next frontier reaches and realized old outputs are allocated as needed.
Decision reaches are materialized after target construction and remain resident
during optimization.

All large tensors remain on one CUDA device. The important allocations and
copies are:

- device-to-device frozen rollout/reference model snapshots;
- active-state/reach gathers for each frontier wave;
- dense log-reach to dense weight conversion for every policy call;
- combo-to-class scatter and class-to-combo lift preflop;
- compact gather, attention workspace, and full-width scatter postflop;
- per-wave output scatter back to original frontier order;
- recomputed dense decision reaches; and
- physical minibatch gathers and activation/checkpoint storage.

There is no bulk PCIe staging, but the current scheduler is not host-sync free.
`plan_policy_batches` uses CUDA `.tolist()`/`.item()` reads; history width uses
`.item()`; the partition cache copies keys to CPU; rollout and transition loops
branch on CUDA tensor booleans. These barriers serialize otherwise device-local
work and should be included in any scheduler redesign.

## Memory preflight caveat

The poker rollout estimator counts two capacity-sized dense reach frontiers and
a capacity-sized policy cache. The generic training estimator counts four such
reach arenas. That is a safe branched-tree upper bound, but it is much too large
for sequential rollout: there is at most one active row per trajectory, not one
per possible node.

For 1,000 sequential trajectories with `node_budget: 128`, the current poker
estimate is about 9.59 GB and the generic structural estimate about 31.98 GB,
so an 11GB 2080 Ti rejects the plan. Those numbers do not describe actual
allocations for 1,000 trajectories of 5–20 nodes. This planner mismatch is one
of the main findings in the
[efficiency audit](audit/rollout_efficiency_2026-07-15.md).

## Construction path

`pbg/games/poker/bundle.py` validates the table/model configuration, constructs
one `PokerModelBundle`, configures every attention module from the hardware
profile, builds the policy and evaluator adapters, preallocates the tensor
backend, and selects completion:

- normal mode: no completion backend;
- both RB modes: tensor-native evaluator completion.

The CPU `PokerGame` facade remains an independent oracle for navigation, range,
arena, and evaluator-adapter tests. Production training uses the tensor backend.
