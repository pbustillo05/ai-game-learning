# P04 — Folded-player card-removal statistics (`folded.py`)

**Depends on:** P00, P02, P03.
**Files:** `pbg/games/poker/folded.py`; `tests/test_p4_folded.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

Implement the resummed estimator of the folded-range partition function Z(A),
its finite-difference expansion (α_B, β_B, Λ_B), signed factorization, compact
per-folder summaries, and the Monte Carlo validation oracle. All math below is
complete and self-contained; implement it exactly.

## 1. Problem objects

At a node: board B (3–5 cards), K folded players, folder k with raw weights
`f_k` over 1326 hands (from `FoldedPlayer.f_weights`; zero on hands
intersecting `board_at_fold`). Base residual set A₀ = {0..51}∖B, a = |A₀|.
Removal sets are A = A₀∖S for small card sets S (live-hand hypotheses).

Z(A) = Σ over pairwise-disjoint assignments (X_1..X_K), X_k a 2-subset of A, of
∏_k f_k(X_k). Define Φ(R) = log Z({0..51}∖R) for R ⊇ B. Only differences of Φ
across removal sets matter.

## 2. Per-folder sufficient statistics on A₀ (compute once per node)

Zero out f_k on hands meeting B, then on A₀:

- F_k = Σ_X f_k(X)  (scalar; if F_k == 0 raise `DegenerateFolderError`)
- M_k(i,j) = f_k({i,j}) / F_k  for i≠j ∈ A₀; symmetric, zero diagonal ([52,52]
  tensor, zero outside A₀×A₀)
- p_k = M_k @ 1  (card marginals; Σ_i p_k(i) = 2 — assert to 1e-6)
- δ_k = p_k − (2/a)·1_{A₀}
- Σ_k = M_k + diag(p_k) − p_k p_kᵀ   (assert Σ_k @ 1_{A₀} ≈ 0)
- w_k = F_k · p_k  (unnormalized marginals)
- pair profiles v_kl(i) = Σ_j M_k(i,j) M_l(i,j)  (for all folder pairs)

**Tight/loose split:** folder k is *tight* (k ∈ C) iff
`line_meta` says it voluntarily committed chips (n_raises_faced concept: use
`line_meta` field `voluntarily_committed > 0`), else loose (k ∈ U). Also
Σ_tot = Σ_k Σ_k over ALL folders.

## 3. The Φ estimator (evaluate at removal set A = A₀∖S)

Restricted quantities via **removal-set algebra** (never renormalize on A;
derive from A₀ objects). With unordered-set sums:

- F_k(A) = F_k − Σ_{r∈S} w_k(r) + Σ_{{r,r'}⊆S} f_k({r,r'})
- ⟨w_k, w_l⟩_A = ⟨w_k, w_l⟩ − Σ_{r∈S} w_k(r) w_l(r)
- ⟨M_k, M_l⟩_{F,A} = ⟨M_k, M_l⟩_F − 2 Σ_{r∈S} v_kl(r)
                      + Σ_{r,r'∈S} M_k(r,r') M_l(r,r')
  (⟨·,·⟩_F is the FULL ordered double sum Σ_{i,j}; the last term is an ordered
  double sum over S×S, diagonal excluded automatically since M(r,r)=0)
- normalized on A: ⟨p_k, p_l⟩_A = ⟨w_k, w_l⟩_A / (F_k(A) F_l(A));
  ⟨M_k, M_l⟩^norm_{F,A} = (F_k F_l · ⟨M_k, M_l⟩_{F,A}) / (F_k(A) F_l(A))
  — CAREFUL: M_k was normalized by F_k on A₀; re-express in raw weights first:
  define raw Frobenius G_kl(A) = Σ_{i,j∈A} f_k(ij) f_l(ij) via the same
  subtraction pattern with raw profiles; then the A-normalized Frobenius is
  G_kl(A)/(F_k(A) F_l(A)). Implement with raw-weight accumulators throughout
  and divide at the end (this avoids the double-normalization bug).

Then with |A| = a − |S|:

Φ(A) = Σ_k log F_k(A)
     + Σ_{j=1}^{K−1} log[ C(|A|−2j, 2) / C(|A|, 2) ]
     + Σ_{k<l} log[ q_kl(A) / q⁰(|A|) ]
     + Σ_{{c,c'}⊆C} δ_cᵀ (Σ_tot − Σ_c − Σ_{c'}) δ_{c'}

where q_kl(A) = 1 − ⟨p_k,p_l⟩_A + ½·G_kl(A)/(F_k(A)F_l(A)) and
q⁰(n) = 1 − 4/n + 1/C(n,2). If |C| ≤ 1 the last sum is empty. If K == 0,
Φ ≡ 0. Raise if any q_kl(A) ≤ 0 (regime failure — surface loudly).
The triple term uses A₀-based δ/Σ (do NOT re-restrict them per hypothesis;
this is the estimator's stated truncation).

`Phi.phi(removed_sets)` must be batched: input int8 [B, R_max] padded card
lists (all ⊇ B implicit — pass only the extra cards S), output fp32 [B].
Per-eval cost target O(K²|S|²) after the per-node precompute.

## 4. Finite differences and consumables

- α_B(u) = Φ({u}) − Φ(∅) for u ∈ A₀  (Φ(∅) meaning S = ∅)
- β_B(u,v) = Φ({u,v}) − Φ({u}) − Φ({v}) + Φ(∅), u≠v ∈ A₀; β_B(u,u) = 0.
  Store as fp32 [52,52] symmetric, zero outside A₀×A₀.
- Λ_B(g) for hand g={c,d}: α_B(c) + α_B(d) + β_B(c,d) → `lambda_g` [1326]
  (−inf-free; 0 for board-conflicting hands).
- Signed eigendecomposition: eigh(β_B); keep the r ≤ `cfg.r_beta_max` (default
  16) largest-|λ| pairs → `beta_factors: list[(lam_j: float, b_j: fp32[52])]`.
  Signs live in lam_j; never square them away.
- Exposed fields exactly as CONTRACTS P4: `lambda_g, beta_factors, delta_c
  [n_tight,52], sigma_tot_contract` (= Σ_tot − Σ_c − Σ_{c'} contractions:
  expose Σ_tot [52,52] and per-tight Σ_c stacked [n_tight,52,52]), `log_F [K]`,
  `w_k [K,52]`, `line_meta [K,d_line]`.
- `FoldedStats.compute(board, folders, cfg)` does: stats → precompute → all
  1 + a + C(a,2) ≤ 1129 Φ evaluations (batched) → differences → eigh.
  Refresh policy is the caller's job (street/fold events only); this class is
  immutable after compute.

## 5. Monte Carlo oracle (`mc_oracle_diff`)

Ground truth: Z(A) = ∏_k F_k(A) · P_disj(A), estimated by drawing
X_k ~ f_k restricted to A (renormalized for sampling only) independently and
recording the disjoint fraction. Implement:

```python
def mc_oracle_diff(board, folders, hyp_pairs, n, np_gen) -> OracleReport
```
- `hyp_pairs`: list of (S, S′) card-set pairs (from single-hand and two-hand
  hypotheses). Estimate Φ(S) − Φ(S′) with **common random numbers**: one
  shared uniform stream drives inverse-CDF draws for both removal sets (per
  folder, per draw), so the difference estimator's variance collapses.
- Report per pair: MC estimate ± stderr (delta method on log of binomial
  fractions plus exact log ∏ F_k ratios), estimator value, order-2 surrogate
  value Λ/β-based, absolute gaps. Aggregate: max and mean |gap| per regime
  (loose-only, one-tight, ≥2-tight).

## Tests (`test_p4_folded.py`)

- **Small-deck exact check (the critical one):** build a miniature version
  with a 10-card deck and K ∈ {2,3} synthetic ranges by monkeypatching the
  deck-size constant path (make deck size an internal parameter of the math
  functions, default 52, so the test can pass 10). Enumerate Z(A) exactly by
  brute force over all assignments; assert the estimator with |C| ≤ 1 matches
  within the documented truncation (rel. err < 2%) and that finite-difference
  α/β from the estimator match brute finite differences of the estimator's own
  Φ exactly (atol 1e-9 — internal consistency, fp64).
- Identities: Σ p_k = 2; Σ_k @ 1 = 0; ⟨(2/a)1, δ_k⟩ = 0 (atol 1e-8).
- Removal algebra: F_k(A) and G_kl(A) via algebra equal direct recomputation
  on the restricted set for 50 random S (fp64, atol 1e-10).
- Flat reference: with uniform ranges, q_kl == q⁰ and all α_B differences
  across single cards are equal (cardinality-only), so β_B ≈ 0 (atol 1e-8).
- Full-deck oracle run: 3 loose + 1 tight synthetic folders on a random flop;
  order-2 surrogate vs oracle differences: max |gap| < 0.01 log-space on the
  loose-only stratum (seeded n = 200k; document H0 failure prob).
- eigh signs: reconstruct β_B from kept factors; ‖residual‖_F below the
  discarded-eigenvalue bound.

## Do not

- No model/network code, no attention, no incidence residual (P07).
- No caching layers beyond the immutable compute (rollout owns refresh).
- Do not renormalize any range on a hypothesis-dependent set anywhere.
