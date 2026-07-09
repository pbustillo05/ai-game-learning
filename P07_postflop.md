# P07 — Postflop network, incidence residual, poker game adapter
# (`model/postflop.py`, `model/incidence.py`, `adapter.py`)

**Depends on:** P02–P06.
**Files:** `pbg/games/poker/model/postflop.py`, `model/incidence.py`,
`pbg/games/poker/adapter.py`; `tests/test_p7_incidence.py`,
`test_p7_postflop.py`, `test_p7_adapter.py`.
**May touch:** `model/heads.py` (only to reuse), append `IMPLEMENTATION_NOTES.md`.

## Part A — Incidence residual (`incidence.py`)

Exact identities (self-contained; these are the whole spec). For one player's
per-hand scalars-or-vectors m_g over hands g, and a query hand h = {a, b}:

- Card accumulators: C_v = Σ_{g ∋ v} m_g (v ∈ 0..51); S_h = m_h.
- Exactly: Σ_{g ∩ h = ∅} m_g = Σ_g m_g − C_a − C_b + S_h, and
  Σ_{g ∩ h ≠ ∅, g ≠ h} m_g = C_a + C_b − 2 S_h.

Per layer ℓ and tokenized range player q (live players postflop):

- m^ℓ_{q,g} = r̃_{q,g} · 1[g ⊥ board] · (W_J^S LN x^ℓ_{q,g})   # [rows, d_j]
- C^{(q)}_v = Σ_{g ∋ v} m^ℓ_{q,g};  S^{(q)}_h = m^ℓ_{q,h}
- O^card_{q,h} = Σ_{q' ≠ q} (C^{(q')}_a + C^{(q')}_b − 2 S^{(q')}_h)
- O^=_{q,h}   = Σ_{q' ≠ q} S^{(q')}_h
- O^own_{q,h} = C^{(q)}_a + C^{(q)}_b − 2 S^{(q)}_h
- j^ℓ_{q,h} = W_J^O gelu(W_J^X LN x^ℓ_{q,h} + O^card + W_J^C O^= + W_J^own O^own)

Cost O(rows · d_j) via a [52, d_j] scatter-add per player — no [H, H] tensors
anywhere (assert this by never allocating one). Provide
`incidence_bruteforce(...)` computing the same via explicit hand-pair sums for
the test.

## Part B — Postflop net (`postflop.py`)

**Compact tokenization is the default and only v1 mode.** Per street, rows are
the board-valid combos H_B (1176/1128/1081): build `valid_combo_ids(board)`
[H_B] int32, `combo_to_compact` scatter map; the trunk runs on [B, live·H_B +
n_public + K_act] tokens with fixed shape per street (batch nodes by street —
P08 already queues preflop/postflop separately; postflop queue is further
keyed by street); outputs scattered back to [B, 1326] with zeros on invalid
rows before returning (masked rows get p_hat 0 downstream; no loss on them).

Row init for live player q, hand h={a,b} (from blocker_model, complete):

- card embeddings c_v = rho_rank(v) + sigma_suit(v) (learned tables);
  chi_h = FFN_chi([c_a, c_b, c_a ⊙ c_b]).
- geometry features phi_geo(h) = [1_h (52-hot), sign-split coords
  {(sqrt|λ_j|·⟨b_j, 1_h⟩) into (pos, neg) slots by sign(λ_j)}_{j≤r_β}]
  ∈ R^{52+2 r_max}, zero-padded to r_max = 16 factors. (λ_j, b_j from
  `FoldedStats.beta_factors`; if no folders, zeros.)
- folded per-card pressure: E_k = FFN_fold0([log F_k, line_meta_k]) per folder
  ([K, d_E]); P_v = Σ_k w_k(v) E_k; row gets P_a + P_b.
- x⁰_{q,h} = W_x [ r̃_{q,h}, log(r̃+ε), log r̃′_{q,h}, chi_h, phi_geo,
  P_a + P_b, 1 ] + seat_emb(pos(q)) + status_emb + ζ·1[q = actor],
  where log r̃′ = log r̃ + Λ_B(h) (folder-corrected log-reach; Λ from
  `FoldedStats.lambda_g`, 0 if no folders).

Public tokens as preflop + one token per board card (W_b [c_v, street_dealt]).
Folded-seat summary tokens (no rows): t_k = FFN_fold([log F_k, line_meta_k,
maxpool_v w_k(v) E_k]) — one token per folder, part of the public token set.
Action anchors as P06.

Attention: key_bias = log r̃′ on hand rows (0 elsewhere), untempered (P06 MHA).
Trunk: pre-LN as P06 with FFN side input ψ = [chi_h, raw range feats,
incidence j^ℓ, bilinear units ψ_bil] where
ψ_bil = [O^card ⊙ O^card, O^card ⊙ O^=, {⟨b_j, 1_h⟩²}_{j≤r}, δ-products
{(δ_cᵀ (LN x-proj))}] — implement exactly these four blocks, dims documented;
δ-products use `FoldedStats.delta_c` with a learned [d, 52] projection.

Readout, heads: as blocker_model/P06 — actor summary from anchors; non-actor
live seats via r̃′-weighted pooling of their rows; folded seats via t_k; value
head on actor rows; moment heads (flag) from public+seat summaries only.
Suit-consistency loss hook: `postflop_suit_penalty(net, batch, gen)` — sample
one ρ ∈ S4 per example, permute (board, ranges, folded stats), forward,
inverse-permute row outputs, return mean squared disagreement on (type logits,
tick probs, V). Flag-gated by `loss.use_suit_consistency`; default off.

## Part C — Poker adapter (`adapter.py`)

```python
class PokerGame:            # implements PBGGame over NLHEEngine + RangeState + FoldedStats + samplers
    # owns: engine StateTable; RangeState slots aligned to rollout nodes;
    # FoldedStats cache keyed by (board_key, folder_set_key), refreshed only on
    # street/fold events; per-node street routing for the model queues.
    # sample_profiles -> rejection backend; chance stepping: samples cards
    # uniformly from residual deck GIVEN a sampled profile (used by branching),
    # via engine.deal.
class PokerPolicyAdapter:   # implements PolicyAdapter over PolicyOutput.raw
    # event log-prob composition (frozen):
    #   passive type covers fold AND check_call: when facing no bet, passive→check_call,
    #     fold illegal; when facing a bet, passive→fold and call type→check_call.
    #   log pi(raise tick t | h) = log pi_T(raise) + log tickprobs[t].
    #   Legality: renormalize type probs over legal types (mask+softmax on logits).
    # entropy: EXACT over the composed distribution on legal engine events
    #   (types × ticks); kl likewise, computed from full composed probs.
class PokerModelBundle:     # holds PreflopNet + PostflopNet; routes by street;
    # policy_fn(state_refs) -> PolicyOutput with [B, 1326] (postflop scattered) or
    # [B, 169] preflop rows — X_max handling: preflop rows are lifted to 1326 by
    # class broadcast ONLY at the adapter boundary when the core needs hand-indexed
    # rows; document where each representation lives.
```

`# CONTRACT-QUESTION` sites are expected here if the preflop 169↔1326 lift
interacts awkwardly with NodeTargets' X_max — implement the lift at the
adapter boundary (targets stored hand-indexed everywhere) and flag.

## Tests

- `test_p7_incidence.py` (**T8**): residual O(52) path equals
  `incidence_bruteforce` on random ranges/boards for 2–4 live players
  (fp32 atol 1e-4, fp64 atol 1e-10); no [H,H] allocation (peak-mem guard or
  monkeypatched matmul-shape assert).
- `test_p7_postflop.py`: compact gather/scatter round-trip is the identity on
  valid rows and zero on invalid; forward under fp16 autocast; with zero
  folders, Λ/φ_geo/pressure paths are exact zeros and the net still runs;
  suit penalty is ~0 (< 1e-6) when weights are suit-symmetric by construction
  (initialize suit embeddings equal for the test).
- `test_p7_adapter.py` (**T9**): composed event log-probs normalize over
  legal events for random legality patterns (logsumexp == 0, atol 1e-5);
  entropy matches numerical entropy of the composed distribution; facing-bet
  vs no-bet passive mapping correct; PBGGame runtime-checkable conformance;
  a 200-hand random self-play smoke through PokerGame (uniform-random model
  outputs) conserving chips and hitting pre-terminal states.

## Do not

- No full-1326 postflop tokenization mode; no ragged batching across streets.
- No training logic; no rollout logic (P08).
- No new layer primitives (reuse P06's only).
