# P05 — Monte Carlo completion & rejection sampler (`mc.py`, `core/completion.py`)

**Depends on:** P00, P02, P03.
**Files:** `pbg/core/completion.py` (dataclasses + Protocols + backend
registry), `pbg/games/poker/mc.py`; `tests/test_p5_completion.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

The completion stage: at a pre-terminal poker node, integrate out private
hands and the remaining board runout, producing the leaf statistics consumed
by backward value propagation (NOTATION §6). Reference torch implementations
behind Protocols so user CUDA kernels can replace them later.

## `pbg/core/completion.py` (generic)

- `CompletionStats`, batched container `BatchedCompletionStats` (same fields,
  leading [B] dim, plus `node_ids int32 [B]`).
- `MCCompletion` Protocol per PREAMBLE.
- `ReachView`: tiny adapter the caller passes in — `log_r fp32 [B, P, X_max]`,
  `live bool [B, P]`, `board int8 [B, 5]` (game-specific payloads allowed via
  an opaque `extra` field). Core does not interpret poker fields; it just
  ferries them.
- Backend registry: `register_completion_backend(name, factory)`,
  `get_completion_backend(name)`; `"reference"` registered by poker's
  `SISCompletion` at import of `pbg.games.poker.mc`.

## `pbg/games/poker/mc.py`

### Target measure (NOTATION §6)

At a pre-terminal node with board B_now (3–5 cards dealt; possibly fewer than
5), live reach rows 𝔯 and folded players' effects already inside 𝔯 upstream
(P08's job — completion treats 𝔯 as the whole prior):

ℙ(x, runout) ∝ [∏_{p live} 𝔯_p(x_p)] · 1[hands pairwise disjoint, disjoint
from B_now and runout] · Unif(runout | remaining deck)

Rewards from `NLHEEngine.payoffs` at the completed board. Folded players do
not hold sampled hands in the reference implementation; their card-removal
effect is treated as part of 𝔯/Λ upstream. (`# CONTRACT-QUESTION` allowed if
this seems to under-integrate — do not change behavior.)

### `SISCompletion` (reference)

Sequential importance sampling, fully batched over nodes and draws
([B, n_samples] lanes, no Python loop over draws):

1. Order live players by descending range concentration (entropy of
   normalized 𝔯 — compute once per node).
2. For player p in order: sample hand h_p ∝ 𝔯_p(h) masked to cards still
   available in that lane; accumulate importance weight
   w ← w · (masked normalizer / full-board normalizer) — i.e. the proposal is
   the masked-renormalized range, the target factor is the raw masked weight;
   the resulting per-lane weight is Σ_{h available} 𝔯_p(h) / Σ_{h ⊥ B} 𝔯_p(h).
3. Deal the remaining runout uniformly from residual deck.
4. Evaluate payoffs R [lane, P].

Weighted statistics (fp32 accumulators, one pass):
- p_hat[p, x] = Σ w·1[h_p = x] / Σ w  (per live p; zeros elsewhere)
- v_hat[p, x] = Σ w·1[h_p = x]·R_p / Σ w·1[h_p = x]  (0 where no mass)
- vvec_hat[p] = Σ_x p_hat·v_hat  (compute directly as Σ w·R_p / Σ w)
- u_hat = Σ w·‖R‖² / Σ w
- n_eff = (Σ w)² / Σ w²

Optional flag `suit_augment` (default False): also accumulate each sample
under the 4! suit permutations applied to (hands, runout) — permutation of a
valid sample is valid and equally weighted; implement as a weight-shared
scatter, not a re-evaluation, only if `evaluate7` cost dominates; otherwise
re-evaluate. Keep it simple and correct.

### `ReferenceRejectionSampler` (for P09 branching)

```python
class RejectionSampler(Protocol):
    def sample(self, reach_rows, board, n, gen) -> Tensor   # int32 [B, n, P] hand ids; -1 for folded seats
class ReferenceRejectionSampler: ...
```
Propose h_p ∝ 𝔯_p independently per live player (board-masked), accept lanes
with pairwise-disjoint hands; loop with a geometric retry cap
(`max_rounds=64`, raise `AcceptanceError` with the measured rate if exceeded).
Vectorized retries: only resample rejected lanes.

Registries: `register_rejection_backend`, mirroring completion.

## Tests (`test_p5_completion.py`)

- **Exactness on tiny decks:** monkeypath-free route — build a fake
  "engine" fixture with a 6-card deck and 2 live players, 1-card hands is NOT
  possible with the poker types; instead test against brute force on the real
  deck with sharply truncated ranges: give each of 2 live players a 5-hand
  support on a fixed turn board; enumerate all (h1, h2, river) exactly in
  fp64; assert p_hat/v_hat/vvec/u_hat within MC tolerance (n=100k, seeded,
  atol from binomial stderr ×4).
- Importance-weight sanity: with disjoint supports the SIS weights are all
  equal and n_eff == n_samples.
- Rejection sampler: acceptance ratio on uniform 3-player ranges matches the
  closed-form non-collision probability of uniform 2-subsets within stderr;
  samples' marginals match 𝔯 (chi-square).
- Backend registry round-trip; `MCCompletion` runtime-checkable conformance.
- Zero-mass guard: a live player with empty masked range raises loudly.

## Do not

- No use of α_B/β_B/Λ here (upstream's job); no engine stepping beyond
  `payoffs` and residual-deck logic.
- No fp16 accumulation anywhere in this file.
- Do not implement CUDA; reference only.
