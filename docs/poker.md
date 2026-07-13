# Poker implementation

The poker package implements public no-limit hold'em state, policies, ranges,
GPU rollout, and evaluator adapters. It deliberately does not implement private
hand ranking, collision-conditioned assignment, folded-card partition math, or
completion equity.

## Representations

Cards are integers `0..51`. Unordered two-card combos use a frozen
lexicographic `0..1325` map, and preflop classes use the standard 169 pair,
suited, and offsuit groups. `cards.py` provides inverse maps, class membership,
suit permutations, board-conflict masks, and uniform ordered complement draws.

Before any evaluator operation, the sibling facade's defensive `COMBO_CARDS`
export is checked against the local map. This turns a representation mismatch
into an immediate contract error rather than corrupted hand rows.

## Public NLHE engine

`NLHEEngine` is an append-only CPU reference state machine. It owns:

- blind posting, button rotation, and heads-up/multiway action order;
- fold, check, call, and affine raise-to ticks;
- minimum full raises and the no-reopen rule after short all-ins;
- round closing and flop/turn/river chance transitions;
- terminal fold rewards and pre-terminal showdown states; and
- public contribution levels and nested side-pot strata.

Chance cards enter through `deal`; the engine does not enumerate the enormous
chance event space. A showdown state remains `PRETERM`. The engine constructs
pot amounts and eligibility masks, accepts evaluator-supplied winner shares for
each stratum, and assembles chip payoffs. It never compares private hands.

`PokerRolloutBackend` reproduces these rules in batched Torch operations using
`PokerTensorStateStore`, a preallocated device structure-of-arrays store. The
same transition path supports CPU oracle tests and CUDA rollout. Its state rows
include public betting fields, boards, dealt/live masks, contributions, and
action history. Prefix allocation and copy-append operations avoid Python
objects in the hot path.

Chance events have deterministic colex public labels with disjoint street
offsets and a padded card payload. The backend routes a combined preflop queue
and postflop queues grouped by number of live players, keeping model evaluation
batched across many trees. `beta_fn` maps states to exact configured stage
overrides or the default intensity.

## Range bookkeeping

`RangeState` stores fp32 unnormalized log action likelihoods for every rollout
slot, player, and combo. An action update gathers the probability of the public
event for each possible actor hand. A board update only masks incompatible
combos; it does not renormalize survivors. Rows are gauge-stabilized when their
maximum log weight becomes too negative, preserving relative probabilities.

When a player folds, the range, board, and compact public metadata are cloned
into an immutable `FoldedPlayer` snapshot before the live mask changes. Branch
cloning produces independent reach, live, board, metadata, and stability rows.
On-demand helpers normalize a combo row or aggregate it into the 169 preflop
classes.

These ranges are likelihood inputs. They are not independently normalized
approximations to the full joint posterior; evaluator calls apply card
collisions and folded blockers jointly.

## Evaluator boundary

Three adapters in `mc.py` lazily import stable operations from
`poker_gpu_evaluator.pbg`:

- `PokerGPUCompletion` maps public boards, dealt/contender masks, unnormalized
  reaches, pot strata, and explicit seeds to batched completion statistics.
- `PokerGPURejectionSampler` requests complete collision-free joint profiles
  and reports partial acceptance through `AcceptanceError`.
- `PokerGPUShowdownEvaluator` requests exact winner masks and tie shares for
  fixed private profiles, boards, and eligibility strata.

`folded.py` similarly maps frozen fold snapshots to the deterministic
partition-summary operation. Returned card/hand fields are opaque learning
features; local code validates but does not derive them.

All adapters enforce facade version, combo representation, tensor shape,
dtype, device, padding, result validity, and metadata. Seeds are allocated from
the caller's explicit generator. There is no NumPy conversion, local poker-math
fallback, or hidden device transfer.

The complete binding contract is
[`../POKER_GPU_EVALUATOR_IO.md`](../POKER_GPU_EVALUATOR_IO.md).

## Policy and value model

Poker uses one owning `PokerModelBundle` with two street-specific networks.
Both consume `PokerPublicBatch` fields and expose a common policy/value output.

### Preflop network

`PreflopNet` operates on 169 class rows for each seated player. It embeds public
betting/action metadata and class tokens, applies a pre-layer-normalized
transformer, and emits policy parameters and a value for every actor class.
Empty seats are masked. Class ordering is equivariant, and combo reach can be
lifted/aggregated at the representation boundary.

### Postflop network

`PostflopNet` operates on board-compatible combo rows rather than all 1326
rows. `CompactMap` provides stable padded gather/scatter maps for the ragged
1176/1128/1081 flop/turn/river spaces. Card and combo embeddings, public
tokens, supplied evaluator summary features, optional fold-line features, and
reach priors feed a pre-LN transformer. Outputs are scattered back to the
fixed-width combo representation.

Suit relabeling changes board ids and every combo-indexed tensor consistently;
it is only a representation transform. It does not infer hand availability or
private assignment.

### Attention and heads

`MHA` supports PyTorch scaled-dot-product attention plus an explicit reference
path. Learned positive temperatures scale content scores only; the key-side
reach prior is added without temperature scaling. `FFN`, `LN`, and GELU form
the transformer blocks.

The policy factorization has contextual passive/call/raise-type logits plus a
conditional raise-tick distribution. `RaiseMixtureHead` uses a mixture of
normalized discretized logistic components with a separate all-in atom.
`BinomialTickHead` is an ablation. `ValueHead` emits one scalar per actor private
row. `MomentHeads` exists but is dormant in the current objective.

`PokerPolicyAdapter` composes the contextual type and conditional tick laws
over the engine's complete legal event rows. It computes probabilities,
selected-event log probabilities, sampling, exact entropy, and KL without
sampling a single private row. A stipulated sampled actor combo only selects
which already-computed row controls the actual public draw.

Detailed model notation is in [Poker Player Model](MHA_model.md).

## Bundle and completion integration

`build_bundle` parses the poker game/model configuration, selects CPU/CUDA/auto
residency, constructs one owning model, and connects the tensor rollout backend,
policy adapter, and tensor-native completion backend. The tensor completion
adapter reads backend state and pot tensors directly, including fold and
showdown cases, and calls the sibling facade on the same device.

The CPU `PokerGame` facade remains useful for independent navigation, range,
arena, and adapter tests. Production training uses the tensor backend.

