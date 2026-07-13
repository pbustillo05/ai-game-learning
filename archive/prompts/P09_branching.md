# P09 — Branched rollout & variance-guided intensities (`branching.py`)

> **Historical prompt — do not launch on the current repository.** C09 replaces
> its Poisson, deduplicated-tree, budget-clamp, and Neyman contracts.

**Depends on:** P08 (+P01 test bed, P05 samplers).
**Files:** `pbg/core/branching.py`; `tests/test_p9_branching.py`.
**May touch:** `pbg/core/rollout.py` ONLY at the `_draw_children` hook and the
`intensity` argument wiring; append `IMPLEMENTATION_NOTES.md`.

## Goal

Generalize the P08 sampler to spine-Poisson branched trees with empirical
transition weights ν, plus the bucketed Neyman intensity controller.
`beta_mode="sequential"` behavior must remain bit-identical for fixed seeds
(test guards this).

## Math (`Docs/latex/core_math.tex`, SampleTree/Rollout tree; NOTATION §5)

At a non-pre-terminal node σ with intensity β(σ) ∈ [1, ∞):

1. K_σ = 1 + Poisson(β(σ) − 1). (β = 1 ⇒ K = 1 deterministically.)
2. Draw K_σ i.i.d. events from the **public mixture**: for each of the K_σ
   draws independently, sample a fresh private profile x from the posterior at
   σ (`game.sample_profiles(reach_rows, n=K_σ)`), then the event given that
   profile — player node: a ~ π^{θ_k}(· | x_actor, σ) using the policy rows
   already computed for σ; chance node: outcome per the game's profile-
   conditional chance rule (poker: deal from residual deck given x).
   **W1 (binding):** each of the K_σ draws uses its own fresh profile; no
   profile is stored, reused across draws, or reused downstream. Do not
   "optimize" by sampling one profile per node or per tree.
   Poker's implementation is the exact whole-profile sibling API in
   `POKER_GPU_EVALUATOR_IO.md`, not local rejection and not independent actor
   marginals. Given that profile, local poker code only selects the actor's
   action-head row or uniformly samples board cards from its complement.
3. Deduplicate outcomes: children = unique events; ν(e | σ) = count(e)/K_σ.
   Store multiplicity in ν only (the tree has one child per unique event).
4. Node budget: track total stored nodes per tree; when a tree's budget would
   be exceeded, clamp its remaining frontier to K = 1 (spine only) and record
   `budget_clamped=True` in rollout metrics. (Deterministic, documented
   degradation — NOT an error, unlike P08's sequential assert.)

Reach rows: children share the parent's posterior update rule exactly as in
P08 (per-edge updates use the stored per-type policy log-probs / chance
masking); `RangeState.clone_rows`-style copy-on-branch via the game adapter
(synthetic: clone log_r rows).

## Intensity controller (`Docs/latex/core_math.tex`, Variance-Based Intensities; NOTATION §9)

```python
class IntensityController:
    def __init__(self, cfg_rollout, bucket_fn, use_moment_heads: bool): ...
    def beta(self, node_meta, moment_outputs) -> Tensor      # fp32 [B] in [1, beta_max(b)]
    def end_iteration(self, collected_s_by_bucket) -> None   # updates bucket means
```
- `bucket_fn(node_meta) -> str`: game-supplied. Synthetic default: `f"d{depth_bin}"`
  (depth//2). Poker default (adapter provides): `f"{street}_{n_live}_{potbin}"`,
  potbin = digitize(pot/stack, [0.1, 0.3, 1.0]).
- Warmup (iteration k < warmup_iters, OR moment heads disabled/absent):
  β(σ) = beta_fix[b(σ)] (missing bucket key → beta_fix["default"], required).
- Post-warmup: s(σ) = sqrt(relu(U(σ) − ‖Vvec(σ)‖²)), with the norm over the
  full seated-player reward vector (folded seats included; only empty/padded
  seats masked), from the FROZEN rollout
  model's moment heads evaluated at σ (policy_fn already returns them when
  the flag is on); with s̄_prev(b) the previous iteration's mean s over that
  bucket's player nodes:
  β(σ) = clip( beta_avg[b] · s(σ) / s̄_prev(b), 1, beta_max[b] ),
  fallback beta_fix[b] when s̄_prev(b) is missing/zero.
- Chance nodes always use beta_fix[bucket] with bucket_fn applied (games may
  give chance its own buckets); never moment-guided.
- Controller state (bucket means) is part of the training checkpoint: expose
  `state_dict()/load_state_dict()`.

## Tests (`test_p9_branching.py`) — all on P01 synthetic games

- **Sequential regression guard:** with `beta_mode="sequential"`, byte-compare
  the finalized TreeBatch tensors against P08's for 5 fixed seeds.
- **T2 (β≡1 via "fixed" mode):** `beta_fix ≡ 1.0` trees are single paths and
  their terminal-history law matches sequential's (chi-square on 50k trees).
- **T3 (terminal-distribution lemma):** with beta_fix = 3.0 on Kuhn, the
  ν-induced leaf measure averaged over 20k trees matches exact ℙ^π reaching
  pre-terminal histories (fp64 tables from P01; chi-square p > 1e-3). This is
  the single most important test in this prompt.
- Spine survival: every tree has ≥ 1 leaf at maximal game depth reachable
  from the root through stored nodes; K_σ ≥ 1 always.
- ν bookkeeping: duplicated event draws collapse with correct multiplicities
  (drive with a rigged 2-event game and large β; compare to binomial law).
- Budget clamp: tiny budget produces spine-only continuation, flag recorded,
  validate() still passes.
- Controller: warmup returns beta_fix; post-warmup formula verified against a
  hand-computed example (fixed s, s̄, clip active on both ends); missing
  bucket falls back as specified; state_dict round-trip.

## Do not

- No changes to value propagation or losses.
- No moment-head training logic (P11).
- Never estimate s(σ) from rollout returns in this prompt (heads only).
