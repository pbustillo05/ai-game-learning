# NOTATION.md — Mathematical Specification for Implementation

**Status: implementation companion to `Docs/latex/core_math.tex`.**
`core_math.tex` is the mathematical source of truth. This file maps its symbols
and algorithms to implementation objects; it does not supersede or silently
change the source. §12 records the few typographical normalizations needed to
make the source's displayed equations and pseudocode refer to the same objects.

Notation: symbols as in `core_math.tex`; code names per CONTRACTS §2.

---

## 1. Finite Public-action Bayesian Games (PBG)

A finite PBG is:

- A finite prefix-closed tree 𝒮 of public trajectories with root ∅; terminal set 𝒯
  (maximal elements); each non-terminal σ has finite event set ℰ_σ, children σe.
- Players 𝒫 with owner map o: 𝒮∖𝒯 → 𝒫 ⊔ {c}. |σ|_p = number of p-owned decision
  nodes that are **strict prefixes** of σ (i.e., decisions already taken along σ);
  |σ|_player = Σ_p |σ|_p.
- Private types: finite 𝒳_p per player, joint prior μ on 𝒳 = ∏_p 𝒳_p.
- Chance rules χ(· | x, σ) ∈ Δ(ℰ_σ) at chance nodes (may depend on the full profile x).
- Reward **R** = (R_p): 𝒳 × 𝒯 → ℝ^𝒫.

A strategy profile π assigns π(· | x_p, σ) ∈ Δ(ℰ_σ) at each player node σ with
p = o(σ). It induces ℙ^π on 𝒳 × 𝒯. Player reach r_p^π(σ | x_p) = ∏ π(a | x_p, ρ)
over p-owned edges ρa ≼ σ; chance reach r_c(σ | x) analogous with χ.

**Pre-terminal trajectories.** For a terminal trajectory τ, τ^pre-c is the
public prefix at the final chance node: all player actions have occurred, but
the final public chance outcome has not yet been drawn. Thus the prefix ends
immediately before that chance event (equivalently, immediately after the last
player action when no intervening public event exists). In the source example
`(caccaaaccct)^pre-c = caccaaac`, the trailing `c` identifies that chance
node; it does not mean its outgoing chance outcome has already been sampled.
In poker, once betting
has closed for the final time, rollout stops before drawing the remaining
runout/chance outcome; completion integrates that chance outcome and showdown.

---

## 2. Ideal decision variables

All constructions conditional on X ∼ μ.

- **Player choice.** ℙ(P = p | X) = 𝔼^π(|τ|_p | X) / 𝔼^π(|τ|_player | X).
- **Decision node.** Conditional on the already sampled X,
  ℙ(Σ = σ | P, X) = 1[o(σ)=P] · ℙ^π(σ ≼ τ | X) / 𝔼^π(|τ|_P | X).
  This makes explicit the X-conditioning stated immediately before the source's
  stationary-node display; marginalizing X gives its displayed shorthand.
- **Action.** A ∼ π(· | X_P, Σ).
- **Continuation.** Τ terminal with ℙ(Τ = τ | Σ A, X) = ℙ^π(τ | Σ A, X).

Values: Q(x_p, σa) = 𝔼^π(R_p(x,τ) | x_p, σa), V(x_p, σ) = 𝔼^π(R_p(x,τ) | x_p, σ),
A = Q − V, for p = o(σ). Marginal joint law of (P, Σ, A): a uniformly random player
decision along a ℙ^π-random trajectory, length-biased by |τ|_player — this is what
the empirical laws below approximate.

---

## 3. Rao–Blackwellization (what is integrated, and against what measure)

The model is curried: one forward pass at public node σ outputs
{x_p ↦ π^θ(·|x_p,σ), V^θ(x_p,σ)} for **all** actor types simultaneously. Training
therefore replaces sampled-private-state loss terms by posterior-weighted sums over
x_p, and replaces terminal rewards by conditional expectations given the **public**
pre-terminal trajectory:

- Q̂ target: 𝔼^π(R_p(x,τ) | x_p, τ^pre-c ≼ τ) — the MC completion estimates this.
- Loss: 𝔼(ℒ) = 𝔼( Σ_{x_P} ℒ(x_P, Σ, ·) · ℙ^π(x_P | conditioning) ), variance-reduced
  by the law of total variance.

**Posterior route (signed-off decision).** In the branched pipeline, the posterior
weights P̂(x_p | Σ) used in the loss are the **backward-MC** ones: ν-weighted mixtures,
over the sampled tree below Σ, of the leaf posteriors ℙ^π(x_p | τ^pre-c) estimated by
the completion sampler. All of (P̂, Q̂, V̂, V̂vec, Û) are expectations under the same
tree-conditional measure ℙ^{β,π}(· | 𝐒); unbiasedness of each is supplied jointly by
the terminal-distribution lemma (§5). No forward/cluster-expansion posterior appears
anywhere in production code. Forward-exact posteriors exist only in
`games/synthetic/exact.py` as test oracles (T6).

---

## 4. Empirical decision laws (minibatch sampling)

Given a stored batch of G trees (sequential trajectories are trees with β≡1):

1. Restrict to player-decision nodes. Each node σ carries
   `path_weight(σ) = ∏_{ρe ≼ σ} ν(e|ρ)` (for sequential trees this is 1 for on-path
   nodes). The edge weight is ν(e|ρ), evaluated at its parent ρ, as required by
   the source's branched terminal law.
2. Sample node Σ from the alias table with weights `path_weight`, i.e.
   ℙ(Σ = σ) ∝ ∏_{ρe ≼ σ} ν(e|ρ), restricted to player nodes; set P = o(Σ).
   This implements, in one step, the source's two-stage law (tree index
   length-biased by player-node count, then uniform decision within): the two are
   identical because ν-path-weights of player nodes in tree i sum to |tree_i|_player-
   type mass. For sequential batches it reduces exactly to the source's
   ℙ(I=i) ∝ |Τ_i|_player then uniform-decision rule.
3. For RB losses no action is sampled: the loss sums over child edges a of Σ present
   in the tree, weighted ν(a|Σ) (§10.3). (Sampled-action variants would draw
   A ∼ ν(·|Σ); not used by `rb_ppo`.)

**Convergence statements (filling the source's two red placeholders).**
As G → ∞, the joint law of (Σ, A) under the empirical decision law converges in
total variation to the ideal law of §2 marginalized over X (sequential case), resp.
to the ideal law with Τ replaced by Τ^pre-c (pre-terminal case). Proof sketch: LLN on
numerator and denominator of the length-biased index law plus consistency of the
within-trajectory uniform choice. Needed only for analysis/tests, not implementation.

---

## 5. Branched rollout

**SampleTree(π, β).** Frontier-driven construction of a public tree:

- At non-pre-terminal σ: K_σ ∼ 1 + Poisson(β(σ) − 1), β(σ) ∈ [1, ∞). The
  deterministic “1+” is the spine: every lineage survives; β ≡ 1 gives a single path.
- Children events (E_k)_{k ≤ K_σ} i.i.d. from the **public mixture**:
  𝔼^π(π(·|x_p,σ) | σ) at player nodes, 𝔼^π(χ(·|x,σ) | σ) at chance nodes. In practice:
  draw a fresh private profile x from the posterior at σ (poker: rejection sampler),
  then draw the event given x. **[W1 — Warning, unchanged from source]** Never sample
  one “true” profile per tree and condition the whole tree on it: that correlates
  sibling subtrees and voids the locality/convergence guarantees. Every node's event
  draws use fresh, node-local posterior draws.
- Empirical transition weights ν(e | σ) = #{k : E_k = e} / K_σ; children stored
  deduplicated with their multiplicities in ν.
- Pre-terminal σ: stop; leaf of the stored tree (owner = PRETERM). Completion (§6)
  integrates the remaining chance + showdown.

**Induced measure.** ℙ^{β,π}(τ | 𝐒) = ∏_{σa ≼ τ} ν(a|σ) over stored leaves, extended
to 𝒳 by ℙ^π(x | τ). **Terminal-distribution lemma:**
𝔼_𝐒[ ℙ^{β,π}(· | 𝐒) ] = ℙ^π(·). All target estimators of §7 are conditional
expectations under ℙ^{β,π}(·|𝐒) and are therefore unbiased for their ℙ^π
counterparts (given unbiased leaf statistics and perfect value heads for the
bootstrapped parts).

---

## 6. MC completion at pre-terminal leaves

At a pre-terminal leaf with reach cache 𝔯 (per-player unnormalized posterior weights
over types, board-masked and ⊙-updated along the path), the evaluator completion
operation targets

ℙ^𝔯(x, τ) ∝ ∏_{q ∈ 𝒫 ∪ {c}} r_q(τ | x_q)

over full private profiles x and terminal continuations τ ≽ τ^pre-c, and returns
fp32 statistics (CONTRACTS §4.4):

- P̂(x_p) ≈ ℙ^π(x_p | τ^pre-c)  — per-player type posterior;
- V̂(x_p) ≈ 𝔼(R_p(X,Τ) | X_p = x_p, τ^pre-c) ;
- V̂vec(p) ≈ 𝔼(R_p | τ^pre-c) — computed as Σ_{x_p} P̂(x_p) V̂(x_p), for every
  player reward component, including deterministic folded-player payoffs;
- Û ≈ 𝔼(‖**R**(X,Τ)‖₂² | τ^pre-c).

Reference implementation: sequential importance sampling (poker); synthetic family
can use exact enumeration when small (also the T6 oracle). Suit-symmetry statistics
augmentation (storing the 4! permutations of each draw) is an optional flag, default
off in v1.

Terminal fold-win leaves use the same collision-conditioned posterior law for
P̂; normalizing each player's reach row independently is not equivalent when μ
couples private types. Their payoff is already public, so V̂_p(x_p)=R_p,
V̂vec=**R**, and Û=‖**R**‖₂² exactly once P̂ has been obtained.

---

## 7. BranchedValue — backward propagation

Inputs: stored tree 𝐒 with ν; leaf `CompletionStats`; frozen value heads
V^{θ_k}(x_p, σ) (actor-type value, defined only for p = o(σ)), and, if moment heads
are enabled, **V**^{θ_k}(σ), U^{θ_k}(σ); horizon λ ∈ [0,1].

Per-node state during the pass: `p̂(x_p | σ)` [P,X], `v̂_p(x_p, σ)` [P,X],
`v̂vec(σ)` [P], `û(σ)` scalar.

```
initialize at every PRETERM leaf from CompletionStats
for σ in reverse topological order (children before parents), σ not a leaf:
    # (1) ν-weighted aggregation from children (all four statistics):
    p̂(·|σ)    = Σ_a ν(a|σ) p̂(·|σa)
    v̂_·(·,σ)  = Σ_a ν(a|σ) v̂_·(·,σa)
    v̂vec(σ)   = Σ_a ν(a|σ) v̂vec(σa)
    û(σ)      = Σ_a ν(a|σ) û(σa)
    if o(σ) is a player p:                        # decision node: emit targets
        # (2) capture targets BEFORE any mixing at σ:
        Q̂(x_p, σa) = v̂_p(x_p, σa)   for every stored child σa
        V̂(x_p, σ)  = v̂_p(x_p, σ)                                   # = ν-avg of children
        emit NodeTargets row: (p̂(x_p|σ), Q̂, V̂, v̂vec(σ), û(σ), ν(·|σ))
        # (3) λ-mix in place, so ancestors see bootstrapped statistics:
        v̂_p(x_p, σ)  ← λ v̂_p(x_p, σ) + (1−λ) V^{θ_k}(x_p, σ)        # owner component only
        v̂vec(σ)      ← λ v̂vec(σ) + (1−λ) **V**^{θ_k}(σ)             # if moment heads enabled
        û(σ)         ← λ û(σ)    + (1−λ) U^{θ_k}(σ)                 # if moment heads enabled
    # chance nodes: aggregation only, no mixing, no targets
```

Source-to-code details:

- The child-loop assignment reads `Q̂(x_p, ρa) ← V̂_p(x_p, ρa)` in the source
  with ρ undefined in that scope; ρ is the loop's current σ. Ordering “capture, then mix” is
  explicit; the child values entering step (1)/(2) are the children's **post-mix**
  values, which is exactly what reproduces the sequential λ-return along any path.
- Only the owner's component v̂_p is λ-mixed at a player node σ (there is no
  value head for x_q at a node q doesn't own). Non-owner components v̂_q propagate as
  pure ν-mixtures of leaf MC values. v̂vec and û mix at **every** player node
  (their exponent in the source counts |·|_player). Consequence worth stating: for a
  fixed player p, the target Q̂(x_p, ·) bootstraps at every downstream p-decision,
  matching the sequential λ-return exponent |·|_p.
- Consistency check (T4): at β ≡ 1 and along a single path, the emitted Q̂(x_p, σa)
  equals the sequential λ-return of §8 exactly.

Complexity: one O(|𝐒| · P_max · X_max) pass; poker X_max = 1326 → keep v̂ in fp32 on
GPU, stream by tree if memory-bound.

---

## 8. λ-returns, sequential form (reference semantics)

For a single trajectory Τ, node σa with p = o(σ), and n = number of p-owned decision
nodes ρ with σ ≺ ρ ≼ Τ (equivalently |Τ|_p − |σ|_p − 1 counting conventions per §1;
define via depth_p arrays, not string ops):

Q̂^λ(x_p, σa) = λⁿ · (terminal target) + (1−λ) Σ_{j=1..n} λ^{j−1} V^{θ_k}(x_p, ρ_j)

where ρ_1 ≺ … ≺ ρ_n are those downstream p-decisions and the terminal target is
R_p(x, Τ) (sampled pipeline) or 𝔼(R_p | x_p, Τ^pre-c) (RB pipeline). Weights sum
to 1. This is the source's sequential λ-return with its generic sampled node Σ
specialized to the fixed node σ used on the left-hand side. λ = 1 is pure MC;
λ = 0 is one-step bootstrap. The
implementation is general-λ throughout; shipped default λ = 0.95.

Û and V̂vec use the same backward recursion from the source's `BranchedValue`
algorithm: both bootstrap at every player decision. Consequently their
single-path expansions have the same downstream-player-decision exponent. The
source's earlier `branched-targets` display has a `γ` superscript and an extra
`−1` only on Û; §12 records why those tokens cannot be used together with the
later defining recursion.

---

## 9. Variance-guided intensities (Neyman)

Per-node standard-deviation proxy from the (frozen) public moment heads:

s_k(σ) = sqrt( [ U^{θ_k}(σ) − ‖**V**^{θ_k}(σ)‖₂² ]₊ )

Player nodes are partitioned into buckets b(σ) ∈ ℬ (poker default: street ×
players-live × pot-size bin; synthetic default: depth bin). Controller:

- Warmup (k < k_warm, or moment heads disabled): β_k(σ) = β_fix(b(σ)).
- After warmup: with s̄_{k−1}(b) the mean of s_{k−1} over player nodes of bucket b in
  the previous iteration's trees,
  β_k(σ) = clip( β_avg(b(σ)) · s_k(σ) / s̄_{k−1}(b(σ)), 1, β_max(b(σ)) ),
  falling back to β_fix(b(σ)) when s̄_{k−1}(b(σ)) is 0/undefined.
- Chance nodes: always β_fix(b_chance) (no model, fixed intensity).

Here β_avg is the per-bucket target average intensity (distinct from β_fix),
the declared β_max is applied per bucket, and the zero-denominator fallback is
β_fix(b(σ)). If s is constant on a bucket, every node receives β_avg(b).

---

## 10. Losses

### 10.1 Primitives (per node row; ratios w.r.t. the cached rollout policy θ_k)

ρ^θ(a, x_p) = π^θ(a|x_p,Σ) / π^{θ_k}(a|x_p,Σ);  Â(x_p, a) = Q̂(x_p, Σa) − V^{θ_k}(x_p, Σ).

- ℒ_PG = −ρ Â
- ℒ_clip = −min( ρ Â, clip(ρ, 1−ε, 1+ε) Â )
- ℒ_V = c_V (V^θ − V̂)²;  ℒ_V-clip with V_clip^θ = V^{θ_k} + clip(V^θ − V^{θ_k}, ±ε_V)
- ℒ_H = −c_H · Ent(π^θ(·|x_p,Σ))  (exact over the factored head, per adapter)
- ℒ_KL = c_KL · KL(π^θ ‖ π^{θ_ref(k)});  ref(k) = K_ref ⌊k/K_ref⌋, disabled if None
- ℒ_KL^sample = c_KL · (ρ log ρ − ρ + 1)
- ℒ_U = c_U (U^θ − Û)²;  ℒ_Vvec = c_Vvec ‖**V**^θ − V̂vec‖₂² over the full
  player reward vector (including folded players; excluding padded seats)
  (+ clip variants)
- NeuRD / R-NaD primitives: **out of scope for v1** (registry slot reserved; adapter
  method `centered_logits` may raise NotImplementedError).

### 10.2 What the gradient targets

Composite gradients approximate ∇ of J_p(π^θ_p, stopgrad(π^θ_{−p})) via the policy
gradient theorem for games (source Thm; constant 𝔼|τ|_p absorbed into learning rate).

### 10.3 `rb_ppo` composite (the shipped default)

For a minibatch of sampled decision nodes {Σ_b} with targets from §7:

L(θ) = (1/B) Σ_b [
   Σ_{x} Σ_{a ∈ children(Σ_b)}  P̂_b(x) ν(a|Σ_b) · ℒ_clip(x, a)
 + Σ_{x} P̂_b(x) · ( ℒ_V(x) + ℒ_H(x) + ℒ_KL(x) )
 + 1[moment heads] · ( ℒ_U + ℒ_Vvec )                            ]

Three weighting levels, matching dependency structure: (x, a) terms weighted
P̂·ν; (x) terms weighted P̂; public terms unweighted. Entropy coefficient follows
`entropy_schedule`. All sums over x are masked to valid/board-compatible types.

---

## 11. Poker-specific mappings (pointers, not derivations)

- **Ranges / reach.** `RangeState` may maintain unnormalized per-hand action
  likelihood products for model/training use. Board compatibility is a simple
  mask. Every collision-conditioned posterior, joint private assignment,
  partition ratio, folded-card correction, and sample from those laws is
  produced by `poker_gpu_evaluator.pbg`; none is rederived locally.
- **Partition/folded summaries.** The local `folded.py` is a DTO/adapter boundary
  for versioned evaluator outputs. The learning model may transform supplied
  hand/card-indexed fields, but this repository contains no Φ/α_B/β_B/Λ_B,
  removal-set, or disjoint-assignment implementation.
- **Model.** Two nets (169-class preflop / combo-row postflop), pre-LN trunk,
  key-side reach prior added un-tempered, learned transforms of evaluator-supplied
  partition fields, factored type × DLogisticTick raise head,
  actor-row V head, flag-gated full-player public moment heads. Compact H_B tokenization is the
  default (CONTRACTS §5-P7). Suit-consistency loss flag-gated, default off.
- **Preflop gauge symmetry.** Preflop net is class-native (169); deterministic
  combo-to-class aggregation may live in `cards.py`/`ranges.py` because it is a
  representation map, not a private-state law.
- **Private poker math.** Completion/equity coefficients, exact joint profiles,
  partition summaries, and fixed-showdown per-stratum winner shares use
  `poker_gpu_evaluator.pbg` under `POKER_GPU_EVALUATOR_IO.md`. The sibling CPU
  backend is the reference oracle. Locally, a sampled actor combo merely selects
  one action-head row; a supplied unavailable-card set merely defines the
  complement from which the next board card(s) are sampled uniformly.
- **Settlement.** Public commitments define local pot strata and payoff
  baselines. `evaluate_showdowns` returns winner masks/shares for each local
  eligibility stratum; local code performs only affine chip-payoff assembly.

---

## 12. Source correspondence and typographical normalizations

The implementation follows `core_math.tex`. The following are symbol-level
normalizations required by other equations or pseudocode in that same source;
none changes the represented probability law or loss:

- In `stationary-node-law`, X-conditioning is implicit in the surrounding
  construction and is written explicitly in §2.
- In the two λ-return displays, the fixed left-hand node σ replaces the generic
  sampled-node symbol Σ on the right.
- In the RB advantage definition, `Q̂_f` is read as `Q̂`; no `f`-indexed
  estimator is defined anywhere in the source.
- In `branched-stationary-node-law`, the path product uses ν(e|ρ), matching
  `branched-terminal-law`; ν(e|σ) would repeat the final node in every factor.
- In `BranchedValue`, the undefined loop symbol ρ is the current σ. The source's
  store-then-mix order is preserved exactly, and only the owner component has a
  private-type value head to mix.
- The source's `BranchedValue` recursion and `branched-u-loss` consistently use
  Û^λ and mix Û exactly where V̂vec is mixed. Those later definitions govern the
  implementation; the isolated `Û^γ` and extra `−1` tokens in the preceding
  `branched-targets` display are not a second estimator.
- Bucketed declarations are applied to the controller formula:
  β_max(b(σ)) and the zero-denominator fallback β_fix(b(σ)).
- The policy advantage uses the frozen-rollout baseline V^{θ_k} from the
  source's Primitives paragraph. The later branched-target prose writes V^θ
  while simultaneously declaring θ current; using it would differentiate
  through the target and conflict with the primitive PPO loss.
- The policy-gradient theorem's positive decision-count factor is constant with
  respect to θ for a frozen rollout batch and is absorbed into optimization
  scaling; the minimized policy loss retains the source's negative sign.
