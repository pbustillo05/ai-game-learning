# P07 — Postflop model and poker public-tree adapter

**Depends on:** P02–P06.
**Files:** `pbg/games/poker/model/postflop.py`,
`pbg/games/poker/model/incidence.py`, `pbg/games/poker/adapter.py`;
`tests/test_p7_postflop.py`, `tests/test_p7_adapter.py`.
**May touch:** `model/heads.py` only to reuse interfaces; append
`IMPLEMENTATION_NOTES.md`.

## Goal

Implement the learned postflop policy/value model and the adapter that navigates
the public poker tree. Poker-private collision laws, partition summaries,
private profiles, completion, and fixed-showdown winners all come from P04/P05
evaluator adapters.

## Postflop model

Compact tokenization remains the default: gather board-compatible combo rows
according to a compatibility map supplied by the representation/evaluator
boundary, batch by street, run the trunk, and scatter actor outputs back to
`[B,1326]`. The model may consume:

- public state: street, board IDs, seats, stacks, commitments, pot, actor, and
  legal-action summary;
- stored per-hand action-likelihood/reach rows;
- learned card/combo embeddings;
- opaque, versioned `PartitionSummaryBatch` fields produced by the evaluator;
- public fold-line metadata.

The model may learn projections, attention, pooling, and elementwise/bilinear
transformations of supplied tensors. It must not enumerate hand pairs to decide
collisions, derive partition ratios, approximate folded-card assignments,
sample a private hand, evaluate a showdown, or compare hand strength.

`incidence.py` is limited to generic tensor scatter/gather/projection helpers
over evaluator-supplied feature fields. Delete the former poker-specific
brute-force collision reference and any test comparing disjoint/intersecting
hand sums.

Suit-consistency augmentation may permute card IDs and model rows, because this
is a representation symmetry. It may not invoke local poker evaluation.

## Poker adapter

```python
class PokerGame:
    # owns public NLHEEngine states, action-likelihood RangeState slots,
    # evaluator-summary cache, evaluator adapters, and model street queues

class PokerPolicyAdapter:
    # composes type and raise-tick heads over a supplied legal event set

class PokerModelBundle:
    # routes preflop/postflop model inference by public street
```

At a player event draw:

1. P05 returns a fresh exact joint profile.
2. Select the acting player's supplied combo/class row from the already computed
   action-head output.
3. Apply the engine-supplied legal-event mask.
4. Compose passive/call/raise-tick probabilities and sample one public action.

The adapter does not sample, repair, reject, or marginalize private cards.
Passive mapping remains contextual: without a bet passive maps to check; facing
a bet passive maps to fold and call maps to check-call. Raise probability is
type probability times conditional tick probability. Entropy and KL are exact
over the composed legal public-event distribution.

At a chance draw, construct `unavailable_cards` from the evaluator-supplied
profile plus current board and call P02's uniform complement sampler. The engine
validates/applies the resulting cards; it canonicalizes a three-card flop to
one unordered public event, matching the source chance rule.

At a fixed showdown, call P05's `PokerGPUShowdownEvaluator` with the final board,
fixed combo IDs, and P02 pot-eligibility strata. Pass returned shares to the
affine payoff assembler. No local winner logic is allowed.

## Tests

- Compact gather/scatter and model forward/autocast tests.
- Opaque evaluator-summary fields reach the intended model projections without
  local recomputation; zero/absent summaries remain supported.
- Suit-consistency representation test.
- Composed event probabilities normalize over supplied legal events; entropy
  and KL match direct numerical calculations.
- Fake profiles select exactly the stipulated actor row; changing unrelated
  private rows does not change that sampled row's event law.
- Chance deals are uniform on the complement of the stipulated profile+board.
- Fixed showdowns cross the fake evaluator facade and use its stipulated shares.
- Static scan rejects local collision enumeration, partition math, private-hand
  sampling/rejection, hand ranking, and direct sibling-internal imports.

## Do not

- No rollout/training logic.
- No full hand-pair matrices or poker-private brute-force references.
- No local fallback when an evaluator operation is unavailable.
