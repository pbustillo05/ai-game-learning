# P10 — Backward value propagation (`values.py`)

> **Historical prompt — do not launch on the current repository.** C10 replaces
> direct conditional-value averaging with posterior mass/moment propagation.

**Depends on:** P05, P08, P09 (+P01 test bed).
**Files:** `pbg/core/values.py`; `tests/test_p10_values.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

Turn a stored tree + leaf completion statistics + frozen value heads into
per-decision-node training targets (`NodeTargets`). This is the
`Docs/latex/core_math.tex` `BranchedValue` algorithm transcribed in NOTATION §7; implement
it exactly as written below — ordering
matters and has been reviewed; do not rearrange.

## Algorithm (complete, binding)

Inputs: `tree: TreeBatch`, `leaf_stats: BatchedCompletionStats` (one row at
every PRETERM or TERMINAL leaf), `reach` (unused here except for asserts),
frozen head evaluations at
player nodes: `v_head fp32 [Nd, X_max]` (actor-type value V^{θ_k}(x_p, σ)),
optional `vvec_head [Nd, P_max]`, `u_head [Nd]` (present iff moment flag;
`vvec_head` covers every seated player, including folded players, with only
empty/padded seats zero-masked), `lam: float`, `game` (for masks).
The caller prepares terminal fold-win rows before this function: p_hat is the
collision-conditioned marginal from the game's joint-profile posterior
operation (never an independently normalized reach row), while
v_hat[p, x] = R_p for every valid x, vvec = R, and u = ‖R‖² because the fold
reward is public and exact.

Per-node working state, fp32: `p_hat [P_max, X_max]`, `v_all [P_max, X_max]`,
`vvec [P_max]`, `u []`.

```
for nodes in reverse topological order (children of every node processed first):
    if leaf: load stats; continue
    # (1) nu-weighted aggregation over stored children (vectorized gather):
    p_hat(σ)  = Σ_a ν(a|σ) p_hat(σa)
    v_all(σ)  = Σ_a ν(a|σ) v_all(σa)
    vvec(σ)   = Σ_a ν(a|σ) vvec(σa)
    u(σ)      = Σ_a ν(a|σ) u(σa)
    if owner(σ) is player p:
        # (2) CAPTURE targets before any mixing at σ:
        q_hat[σ, a, x]  = v_all(σa)[p, x]   for each stored child a (children's
                          values are post-THEIR-OWN-mixing — that is correct)
        v_hat[σ, x]     = v_all(σ)[p, x]
        emit NodeTargets row: (p_hat(σ)[p, :], q_hat, v_hat, vvec(σ), u(σ), ν row, child events)
        # (3) mix IN PLACE so ancestors see bootstrapped statistics:
        v_all(σ)[p, :] = lam * v_all(σ)[p, :] + (1 − lam) * v_head[σ, :]     # OWNER row only
        if moment heads:
            vvec(σ) = lam * vvec(σ) + (1 − lam) * vvec_head[σ]
            u(σ)    = lam * u(σ)    + (1 − lam) * u_head[σ]
    # chance nodes: aggregation only
```

Notes (binding):
- Non-owner rows v_all[q ≠ p] are NEVER mixed at σ (no head exists for them);
  they propagate as pure ν-mixtures of leaf MC values.
- vvec/u mix at EVERY player node when moment heads are enabled; when
  disabled, they propagate unmixed (λ effectively 1 for them) and NodeTargets
  still carries them (loss ignores them).
- p_hat is never mixed anywhere.
- Implementation: process nodes grouped by depth (reverse), vectorized
  segment-sums over the CSR child ranges; fp32; poker X_max = 1326 — chunk by
  tree if a memory cap `cfg.values_max_rows` (add to TrainConfig? NO — take it
  as a function kwarg `max_resident_rows=2_000_000`) would be exceeded.

```python
def branched_value(tree, leaf_stats, reach, heads: HeadEvals, lam, game,
                   max_resident_rows=2_000_000) -> NodeTargets
@dataclass
class HeadEvals:
    node_ids: Tensor; v: Tensor; vvec: Tensor | None; u: Tensor | None
def branched_value_reference(tree, leaf_stats, reach, heads, lam, game) -> NodeTargets
```

`branched_value_reference`: an INDEPENDENT fp64 implementation used only by
tests — recursive path-enumeration semantics: for each player node σ, owner p,
type x, stored child a,
q_ref(x, σa) = Σ over stored descending paths (σa = ρ_0, ρ_1, …, ρ_T=leaf)
  ∏ ν(edges) · [ λ^{n_p(path)} · leaf v_hat[p, x]
                 + (1−λ) Σ_{j=1..n_p(path)} λ^{j−1} v_head[ρ^{(p)}_j, x] ]
where ρ^{(p)}_1, …, ρ^{(p)}_{n_p} are the p-owned decision nodes strictly
inside the path (NOTATION §8 semantics under the ν-measure). Same for
p_hat/vvec/u analogues with their own mixing rules. Recursion with memo is
fine; clarity over speed.

## Tests (`test_p10_values.py`)

- **T4 (the gate):** on Kuhn and on a `dial()` game with branching
  (beta_fix = 2.5, 200 trees, budget-clamped small), fp64 run of both
  implementations with randomized leaf stats and randomized heads: all five
  target families equal within atol 1e-10. Sweep lam ∈ {0.0, 0.5, 0.95, 1.0}.
- Sequential consistency: on single-path trees, emitted q_hat equals a
  direct implementation of the `Docs/latex/core_math.tex`/NOTATION §8 sequential λ-return
  formula written inline in the test (fp64, atol 1e-12).
- lam = 1 ignores heads entirely (perturb heads; targets unchanged).
- lam = 0: q_hat at σ equals the child's OWN capture-then-mix boundary — for a
  child that is a p-owned decision node, q_hat(x, σa) = v_head[σa, x] exactly;
  assert this on a hand-built 3-level tree.
- Non-owner purity: perturbing v_head at nodes owned by p must not change any
  v_all[q≠p] contribution — verified via vvec-off runs on the reference.
- Fold-win terminal handling: hand-built tree with a coupled private prior;
  supplied collision-conditioned p_hat and exact reward moments produce the
  hand-computed targets. Include a fixture where independent row normalization
  gives a different posterior, so that shortcut is rejected.
- Memory: chunked and unchunked paths agree (set max_resident_rows tiny).

## Do not

- No loss computation, no sampling, no model calls (heads arrive pre-evaluated).
- Do not reorder capture vs mix; do not mix non-owner rows; do not mix p_hat.
- No fp16 anywhere in this file.
