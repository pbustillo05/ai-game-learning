# P06 — Layer primitives + preflop network (`model/primitives.py`, `model/preflop.py`, `model/heads.py`)

**Depends on:** P00, P02, P03.
**Files:** `pbg/games/poker/model/primitives.py`, `preflop.py`, `heads.py`;
`tests/test_p6_primitives.py`, `test_p6_preflop.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

The shared transformer primitives (with the two non-standard details: per-head
content temperature and un-tempered additive key bias), the policy/value heads
(action-type head + discretized-logistic raise-size mixture), and the
169-class preflop network. Postflop (P07) reuses everything here.

## `primitives.py`

Only these layer types may exist anywhere in the model code; a name subscript
means a private weight set, never a new operation.

```python
class LN(nn.LayerNorm): ...                       # standard, learned affine
def gelu(x): return F.gelu(x)
class FFN(nn.Module):
    # W2 @ gelu(W1 @ x + b1) + b2; hidden = 4*out unless hidden= given;
    # input may be a concat — in_dim explicit.
class MHA(nn.Module):
    """
    forward(q, kv, key_bias, key_mask) -> [Bq, d]
      q [B, Nq, d], kv [B, Nk, d]
      key_bias fp32 [B, Nk]: ADDED to logits of every head UNTEMPERED
      key_mask bool [B, Nk]: False -> -inf
    Per-head logits: (Q W_q)(K W_k)^T / (tau_m * sqrt(d_k)) + key_bias + mask
      tau_m = softplus(theta_m) + 0.5, learned per head, applied to the
      CONTENT term only (key_bias added after division).  # NOTATION §11
    Implementation: compute content scores with SDPA when key_bias is zero
    and mask-only; otherwise materialize logits explicitly (fallback path);
    both paths unit-tested for equality.
    """
```

## `heads.py`

Event-id layout from P02 is frozen: `0=fold, 1=check_call, 2+t=raise tick t`,
ticks t ∈ [0, n_ticks), t = n_ticks−1 ⇔ all-in.

```python
class TypeHead(nn.Module):
    # logits over action types T = {passive(fold/check contextual), call, raise}:
    # per actor row features z_T [B, X, 3] -> type logits; legality handled by
    # the ADAPTER via masks, not here.
class RaiseMixtureHead(nn.Module):
    # per row: mixture weights omega [B,X,M] (softmax), locations mu (sigmoid),
    # scales s (softplus + s_min), all-in atom iota (sigmoid).
def dlogistic_tick(n_ticks, mu, s) -> Tensor
    # [.., n_ticks] probs: discretized logistic on the tick grid u_t=(t+.5)/n_ticks:
    # p_t ∝ CDF((t+1)/n_ticks) − CDF(t/n_ticks), logistic CDF with loc mu scale s,
    # renormalized over [0,1]. Exact normalization (divide by CDF(1)−CDF(0)).
def raise_tick_probs(omega, mu, s, iota, n_ticks) -> Tensor
    # iota on the all-in tick + (1-iota) * mixture of dlogistic_tick; [B,X,n_ticks]
class BinomialTickHead(nn.Module): ...   # flag-gated ablation, same interface
class ValueHead(nn.Module): ...          # FFN -> scalar per actor row: [B, X]
class MomentHeads(nn.Module): ...        # Vvec [B,P] (every seated player, including folded) +
                                         # U [B] (softplus); empty/padded seats zero-masked; built
                                         # from public + seat summaries; used only when flag on
```

The composed per-hand policy over engine events:
π(fold or check_call | h) from the type head (with fold/check sharing the
"passive" type contextually — the ADAPTER maps type→event using legality);
π(raise tick t | h) = π_T(raise | h) · raise_tick_probs[t]. P07's adapter owns
the exact mapping; heads here just emit the pieces. During rollout an actor
combo supplied by the evaluator selects one row before local action sampling.
This module never samples or infers an actor hand.

## `preflop.py`

Tokens (per NOTATION §11 / `Docs/MHA_model.md`, class-level):

- **Class rows**: for every seated player p and class k ∈ [0,169):
  features `[r_class[p,k], log(r_class+eps), n_k/12, class_embedding(k), 1]`.
  `class_embedding` is learned or evaluator-supplied; do not locally derive
  pairwise collision/overlap statistics. Add learned seat embedding
  (button-relative), status embedding
  (live/folded/allin), actor flag ζ if p == actor.
- **Public tokens**: pot, stacks (button-relative order), committed, street
  scalars via one linear W_y + type embeddings; one state token.
- **Action-anchor tokens**: K_act = 3 + M_R learned slots for the actor
  (passive, call, allin_ctx, M_R raise anchors), each + actor seat embedding.

Trunk: pre-LN, L layers:
x̂ = x + MHA(LN x) over the full token set with
key_bias = log r̃_class on class rows (0 elsewhere) and key_mask on empty
seats/pad; then x ← x̂ + FFN([LN x̂, ψ]) where side input ψ for class rows =
[class embedding, raw range features, optional evaluator-summary projection].

Readout (per `Docs/MHA_model.md`): actor summary t_★ from anchor tokens; non-actor
seat summaries by reach-weighted pooling of that seat's class rows; global
g = FFN_read([mean public tokens, t_★, seats...]); per-class features
z_T, z_R from [LN x_row, LN anchors, class emb, g]; heads on top.

```python
class PreflopNet(nn.Module):
    def forward(self, pub: PokerPublicBatch, class_ranges, actor_seat, masks) -> PolicyOutput
    # raw payload = dict(type_logits [B,169,3], omega/mu/s [B,169,M], iota [B,169])
    # v = value head [B,169]; vvec/u present iff cfg flag.
```
`PokerPublicBatch` (define in preflop.py, reused by P07): tensors for street,
board, pot, stacks, committed, to_act, legality summary, n_ticks grid bounds —
document every field with shape.

Config (from `RootConfig.model` dict, parsed by `PokerModelConfig.from_dict`):
`d_model=256, n_heads=8, n_layers_pre=4, m_raise=4, k_act=7, s_min=0.02,
n_buckets_line=...` — defaults given; all overridable.

## Tests

- `test_p6_primitives.py`: MHA fallback path == SDPA path when bias-free
  (atol 1e-5 fp32); key_bias is untempered: doubling all θ_m (temperatures)
  changes content contribution but a pure-bias attention (zero q/k weights)
  is invariant; masked keys receive exactly zero attention.
- Heads: `dlogistic_tick` sums to 1 (atol 1e-6) for random (mu, s) including
  extreme s; monotone location shift moves the argmax tick; all-in atom: iota
  = 1 puts all raise mass on the last tick.
- `test_p6_preflop.py`: forward shape/dtype contract under autocast fp16 and
  fp32; permutation equivariance sanity — permuting the 169 class order of
  inputs permutes per-class outputs identically (run with seat embeddings
  fixed); empty-seat masking: adding a masked seat changes nothing (atol 1e-5
  fp32); gradient flows to every parameter (grad-norm > 0 after a scalar loss
  on random inputs).

## Do not

- No postflop code or locally derived collision/partition channels (P07).
- No adapter/legality logic in heads (P07 owns the adapter).
- No layer types beyond LN/MHA/FFN/affine + the head nonlinearities above.
