# P05 — Poker-private evaluator adapters

**Depends on:** P00, P02, P03.
**Files:** `pbg/core/completion.py` (generic DTOs/protocol only),
`pbg/games/poker/mc.py`; `tests/test_p5_completion.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

Implement thin adapters to the three production operations frozen in
`POKER_GPU_EVALUATOR_IO.md`:

1. completion/equity coefficients;
2. exact joint private-profile sampling; and
3. exact fixed-showdown winner masks/shares by pot stratum.

There is no local poker reference implementation. The sibling CPU backend is
the oracle and the sibling CUDA backend is the accelerated implementation.

## Generic core

Keep `CompletionStats`, `BatchedCompletionStats`, `MCCompletion`, and
`ReachView` game-generic. Core ferries opaque game payloads and never imports
poker or the sibling package.

## Poker adapters

```python
class PokerGPUCompletion:
    def complete(self, game, tree, node_ids, reach, n_samples, gen) -> BatchedCompletionStats: ...

class PokerGPURejectionSampler:
    def sample(self, reach_rows, board, dealt_mask, n, gen) -> Tensor:
        # int32 [B,n,P], exact whole joint profiles; -1 only for undealt seats

class PokerGPUShowdownEvaluator:
    def evaluate(
        self,
        *,
        board_cards: Tensor,       # final uint8 [B,5]
        combo_ids: Tensor,         # int32 [B,P], -1 inactive
        active_mask: Tensor,       # bool [B,P]
        stratum_players: Tensor,   # bool [B,K,P], locally tracked eligibility
    ) -> ShowdownResult: ...
```

All adapters lazily import only these public facade members:

```python
from poker_gpu_evaluator.pbg import (
    API_VERSION,
    complete,
    evaluate_showdowns,
    sample_joint_profiles,
)
```

Require `API_VERSION == "pbg-v2"`. Never fall back silently. Preserve tensor
device/stream and avoid NumPy conversions. Allocate spot-local seeds from the
explicit generator exactly as the IO contract specifies.

### Completion

Pack engine board/status state, unnormalized reach weights for every dealt seat
(including frozen folded rows), dealt/contender masks, and locally tracked pot
strata/amounts/baselines as required by the contract. Return sibling
statistics field-for-field. Do not sample a
hand or runout, reconstruct equity, rank a hand, or calculate a winner locally.
The supplied PRETERM state is the final chance node before its outcome is
drawn, so the evaluator completion owns that remaining chance outcome and
showdown.

### Joint profiles

Make one batched sibling call and return the complete accepted profile,
including folded seats whose cards still block the deck. Each branch/event draw receives a fresh profile. Partial failure becomes a local
`AcceptanceError` carrying sibling diagnostics. Never synthesize a profile from
independent actor marginals.

This same operation is also the posterior source at public terminal fold-win
leaves: sample the collision-conditioned joint law, reduce samples to per-seat
marginals P̂, and combine them with the already-public exact fold reward. Do
not replace those marginals with independently normalized reach rows.

### Fixed showdowns

Pass the final board, fixed private combos, active mask, and all local
eligibility strata to `evaluate_showdowns`. The sibling ranks each hand once
and returns `winner_mask [B,K]` plus `showdown_share [B,K,P]`. Validate the
result contract, then pass shares to P02's affine payoff assembler. Do not
compare masks/ranks to determine poker winners locally; only shape, eligibility,
share-sum, and chip-conservation validation is allowed.

## Tests

- Fake-facade tests cover exact argument names, shapes, dtypes, devices, board
  padding, seat packing/scattering, seeds, metadata, lazy import, and version
  failure for all three operations.
- Stipulated multi-stratum showdown masks/shares map back to original seats and
  assemble correct chip payoffs. The fake decides winners; the test contains no
  known-hand ranking assertions.
- Rejection returns full disjoint profiles and reports partial failure. The
  adapter does not itself test collision legality beyond trusting/validating the
  sibling result contract.
- Real-sibling CPU/CUDA integration tests run when installed; otherwise only
  those tests are skipped. Mathematical evaluator tests live in the sibling.
- Static scan rejects `evaluate7`, `hand_rank`, category/kicker logic, local SIS,
  local rejection sampling, direct `batch_eval` imports, and local CUDA builds.

## Do not

- No local poker sampling/equity/showdown oracle, even for tests.
- No imports of sibling private/support modules. In particular, do not import
  `batch_eval.cuda` directly; the sibling facade owns backend normalization.
- No local JIT compilation, global RNG, device round trip, or legacy evaluator
  API.
