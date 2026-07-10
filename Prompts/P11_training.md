# P11 — Losses, minibatching, training loop (`losses.py`, `minibatch.py`, `training.py`)

**Depends on:** P08–P10 (+P01 for tests; poker wired in P13).
**Files:** `pbg/core/losses.py`, `pbg/core/minibatch.py`, `pbg/core/training.py`;
`tests/test_p11_losses.py`, `test_p11_training.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## `minibatch.py`

```python
def sample_minibatch(tree, targets: NodeTargets, alias: AliasTable,
                     policy_cache: PolicyCache, B, torch_gen) -> MinibatchSample
@dataclass
class MinibatchSample:
    node_ids: Tensor           # int32 [B] (sampled WITH replacement from alias)
    targets: NodeTargets       # gathered rows [B, ...]
    old_logp_child: Tensor     # fp32 [B, E_max, X_max] from PolicyCache
    old_v: Tensor              # fp32 [B, X_max]
    state_refs: Tensor         # int32 [B] for re-encoding through the current model
    valid_type_mask: Tensor    # bool [B, X_max]
    child_mask: Tensor         # bool [B, E_max]
```
Decision-law sampling per NOTATION §4: alias over player nodes with
path_weight — nothing else. No stratification, no dedup.

## `losses.py`

Primitive functions, each `(new, old, tgt, cfg) -> (loss [B] or [B,X], metrics
dict)`, masked sums only (never mean over padded entries):

- ratio: ρ[b, a, x] = exp(logp_new − old_logp_child), logp_new from
  `adapter.log_prob(out_new, child_event_row)`.
- advantage: Â[b, a, x] = q_hat − old_v (broadcast old_v over a).
- `l_pg`: −ρ Â.
- `l_clip`: −min(ρ Â, clip(ρ, 1±ε) Â).
- `l_v`: c_V (v_new − v_hat)².
- `l_v_clip`: with v_clip = old_v + clip(v_new − old_v, ±ε_V):
  c_V · max((v_new − v_hat)², (v_clip − v_hat)²).
- `l_h`: −c_H(k) · adapter.entropy(out_new); schedule "const" or
  "linear_decay" (c_H · max(0, 1 − k/iters)).
- `l_kl`: c_KL · adapter.kl(out_new, out_ref); out_ref from a frozen ref model
  forwarded on the same minibatch (training.py owns loading θ_{ref(k)},
  ref(k) = ref_period·⌊k/ref_period⌋; term skipped when ref_period is None).
- `l_kl_sample`: c_KL · (ρ log ρ − ρ + 1), using the same rollout-policy
  ratio as `l_pg`/`l_clip`.
- `l_u`, `l_vvec`: c_U (u_new − u_hat)², c_Vvec ‖vvec_new − vvec_hat‖².
  The norm covers every seated player's reward component, including folded
  players; mask only empty/padded seats. This matches the full vector **R** in
  `Docs/latex/core_math.tex`'s V-vector/U heads and covariance-trace identity.

Composite registry `COMPOSITES = {"rb_ppo": rb_ppo}` with (NOTATION §10.3,
binding — three weighting levels):

L = mean_b [ Σ_x Σ_a  p_hat[b,x] ν[b,a] · l_clip[b,a,x]
           + Σ_x      p_hat[b,x] · ( l_v[b,x] + l_h[b,x] + l_kl[b,x] )
           + 1[moment] · ( l_u[b] + l_vvec[b] ) ]

Masks: x over valid_type_mask, a over child_mask. Metrics emitted per step:
each component, approx_kl = Σ p̂ν·(ρ−1−logρ) masked-mean, clip_frac, entropy,
explained variance of v_new vs v_hat, ratio max.

## `training.py`

```python
def train(cfg: RootConfig) -> None      # torchrun entrypoint (also single-process)
```
Iteration k (per rank, world size W):
1. Freeze θ_k. Rollout `n_trees // W` trees via P08/P09 with policy_fn from
   the current model (eval mode, no grad, autocast per cfg); build ReachCache,
   PolicyCache; run completion at PRETERM leaves and collision-conditioned
   posterior reduction plus exact reward moments at TERMINAL fold-win leaves;
   evaluate frozen heads at player
   nodes; `branched_value` → NodeTargets; build decision alias.
2. `gd_steps_per_iter` steps: sample minibatch (local), forward current model
   at `state_refs` (autocast), composite loss, backward through GradScaler,
   clip_grad_norm_(grad_clip), optimizer step. Model wrapped in DDP when
   `WORLD_SIZE > 1` (gradient all-reduce is DDP's; no manual comms).
3. IntensityController.end_iteration with bucket s-stats (all-reduced mean
   across ranks via `dist.all_reduce` when distributed).
4. Rank 0: metrics log; checkpoint every `checkpoint_every` iters:
   `{model, opt, scaler, controller, iter, cfg_dict, torch_rng, np_rng_state}`
   → `out_dir/ckpt_{iter:07d}.pt`; `load_checkpoint(path, ...)` restores all.

Determinism contract: single-rank runs with equal seeds are exactly
reproducible (document the torch flags you set and their cost).
Model/adapters arrive via a small factory:
`build_game_bundle(cfg) -> (game, model, adapter, policy_fn, bucket_fn)` —
implement the synthetic branch here; poker branch raises NotImplementedError
with a pointer to P13 (which fills it in).

## Tests

- `test_p11_losses.py`:
  - Gradient-direction sanity: single decision node, two children, hand-set
    p̂/ν/Â; analytic ∂L/∂logits of l_clip inside the trust region equals the
    autograd gradient (fp64, atol 1e-8); clipping zeroes the gradient outside.
  - Masking: padded types/children contribute exactly zero gradient.
  - **T5 (RB vs naive gradient):** on Kuhn at a fixed random θ: (a) naive
    estimator — sequential rollouts, SAMPLED private state, sampled-action
    l_pg with MC returns (implement the naive baseline inline in the test);
    (b) rb_ppo with ε = ∞ (unclipped ⇒ l_pg semantics), λ = 1, exact
    completion. Both average over 200k decisions: cosine similarity of policy
    gradients > 0.99 and RB gradient variance strictly smaller (report ratio).
    Seeded; document runtime (< ~2 min CPU target).
  - approx_kl ≈ 0 and clip_frac == 0 at θ_new == θ_k.
- `test_p11_training.py`: 30-iteration Kuhn smoke (`kuhn_smoke.yaml` real
  values): loss finite, entropy decays per schedule, checkpoint save→load→
  resume reproduces the next iteration's metrics exactly (single rank);
  **T10-lite**: exploitability(export_policy_tables(model)) after 300 iters
  < 0.5 × exploitability at init (seeded; mark `@pytest.mark.slow`).

## Do not

- No poker imports in core (T12 stays green); poker wiring is P13.
- No NeuRD/R-NaD implementations (registry slots may exist, unimplemented);
  all other source primitives listed above are implemented even when the
  shipped `rb_ppo` composite does not select them.
- No manual gradient all-reduce; no non-DDP distributed paths.
