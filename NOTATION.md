# NOTATION.md — Mathematical Specification for Implementation

**Status: implementation companion to `docs/theory/core_math.tex`.**
`core_math.tex` is the mathematical source of truth. This file maps its symbols
and algorithms to implementation objects; it does not supersede or silently
change the source. §12 records the few typographical normalizations needed to
make the source's displayed equations and pseudocode refer to the same objects.
The branching, value-target, and loss rules below supersede the corresponding
P09--P11 companion text: v1 is the fixed-intensity PPO path identified here.

Notation: symbols as in `core_math.tex`; code names per CONTRACTS §2.

---

## 1. Finite Public-action Bayesian Games (PBG)

A finite PBG is:

- A finite prefix-closed tree 𝒮 of public trajectories with root ∅; terminal set 𝒯
  (maximal elements); each non-terminal σ has finite event set ℰ_σ, children σe.
- Players 𝒫 with owner map o: 𝒮∖𝒯 → 𝒫 ⊔ {c}. A player-owned state σ is
  the state immediately before its owner chooses an outgoing action; that action
  is the edge σ→σa. Consistently with `core_math.tex`,
  |σ|_p = #{ρ ≼ σ : o(ρ)=p}, including σ itself when o(σ)=p, and
  |σ|_player = Σ_p |σ|_p.
- Private types: finite 𝒳_p per player, joint prior μ on 𝒳 = ∏_p 𝒳_p.
- Chance rules χ(· | x, σ) ∈ Δ(ℰ_σ) at chance nodes (may depend on the full profile x).
- Reward **R** = (R_p): 𝒳 × 𝒯 → ℝ^𝒫.

A strategy profile π assigns π(· | x_p, σ) ∈ Δ(ℰ_σ) at each player node σ with
p = o(σ). It induces ℙ^π on 𝒳 × 𝒯. Player reach r_p^π(σ | x_p) = ∏ π(a | x_p, ρ)
over p-owned edges ρa ≼ σ; chance reach r_c(σ | x) analogous with χ.

**Pre-terminal trajectories.** For a terminal trajectory τ, τ^pre-c is the
largest public prefix after which no player-owned node remains. All player
actions are complete; continuation consists only of zero or more public-chance
draws and terminal reward resolution. In poker, closed betting with an
incomplete board is pre-terminal and completion owns the runout and showdown.
More generally, a state before a public chance event is not pre-terminal when
another player decision can follow that event.

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

**Posterior route (signed-off decision).** The branched pipeline propagates
backward-MC bucket mass P̂ and posterior-weighted target mass M̂=P̂V̂. These
unnormalized quantities are linear expectations under the tree-conditional
measure and are unbiased when the leaf bucket statistics are unbiased. The
conditional ratio `Cond(M̂,P̂)` is generally biased at finite sample size, but is
consistent for positive-mass buckets and asymptotically unbiased for bounded
rewards. No forward/cluster-expansion posterior appears in production code;
forward-exact posteriors exist only as test oracles.

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

### 4.1 Distributed fixed-rollout normalization

For rank `g` and execution family `m`, let `D[g,m]` be its retained player
decisions and let `omega_g(s)` be one in normal/sequential RB or the stored
root-to-node `path_weight` in branched RB.  Define

`Z[g,m] = sum_{s in D[g,m]} omega_g(s)` and `Z = sum_{g,m} Z[g,m]`.

The fixed-rollout objective is

`L(theta) = sum_{g,m} sum_{s in D[g,m]} omega_g(s) ModeLoss(theta;s) / Z`.

Thus a family-stratified local draw uses
`q[g,m](s)=omega_g(s)/Z[g,m]` and scale
`alpha[g,m]=Z[g,m]/Z`.  The path weight appears in the sampling law and alpha;
it is not multiplied into the sampled loss a second time.  Summing the already
globally scaled rank gradients is exact.  Equal averaging of independently
normalized rank losses is a different objective whenever rank masses differ.

**Convergence statements (filling the source's two red placeholders).**
As G → ∞, the joint law of (Σ, A) under the empirical decision law converges in
total variation to the ideal law of §2 marginalized over X (sequential case), resp.
to the ideal law with Τ replaced by Τ^pre-c (pre-terminal case). Proof sketch: LLN on
numerator and denominator of the length-biased index law plus consistency of the
within-trajectory uniform choice. Needed only for analysis/tests, not implementation.

---

## 5. Branched rollout

**SampleTree(π, β).** Frontier-driven construction of a labeled sampled tree:

- At non-pre-terminal σ,
  `K_σ = floor(β(σ)) + Bernoulli(β(σ) − floor(β(σ)))`.
  Since β ≥ 1, every lineage has a spine; β ≡ 1 gives a single path.
- Children events (E_k)_{k ≤ K_σ} i.i.d. from the **public mixture**:
  𝔼^π(π(·|x_p,σ) | σ) at player nodes, 𝔼^π(χ(·|x,σ) | σ) at chance nodes. In practice:
  draw a fresh private profile x from the posterior at σ (poker: rejection sampler),
  then draw the event given x. **[W1 — Warning, unchanged from source]** Never sample
  one “true” profile per tree and condition the whole tree on it: that correlates
  sibling subtrees and voids the locality/convergence guarantees. Every node's event
  draws use fresh, node-local posterior draws.
- The optimized `training.mode="rao_blackwell"` implementation does not invoke this
  branching construction with `K_σ=1` mechanically. It samples one profile from
  μ at each root and carries it along the single path, exactly as
  `SamplePreTerminal`. The resulting public-trajectory law is the same, without
  repeated posterior sampling.
- Every abstract child is stored separately, even when two children have the
  same public event label. Each outgoing sampled-tree edge has probability
  `1/K_σ`. Equal-labeled nodes may share immutable public-state payloads but
  their descendants and completion samples remain independent.
- Pre-terminal σ: stop; leaf of the stored tree (owner = PRETERM). Completion (§6)
  integrates the remaining chance + showdown.

**Induced measure.** ℙ^{β,π}(τ | 𝐒) follows uniformly selected abstract
children, hence uses the product of `1/K` along a sampled-tree path, and is extended
to 𝒳 by ℙ^π(x | τ). **Terminal-distribution lemma:**
𝔼_𝐒[ ℙ^{β,π}(· | 𝐒) ] = ℙ^π(·). This gives unbiasedness for linear
bucket masses and posterior-weighted moments, not for ratios of their estimators.

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

The production implementation uses the sibling poker evaluator's documented
collision-conditioned Monte Carlo operations. There is no local exact or
suit-augmentation completion alternative.

Terminal fold-win leaves use the same collision-conditioned posterior law for
P̂; normalizing each player's reach row independently is not equivalent when μ
couples private types. Their payoff is already public, so V̂_p(x_p)=R_p,
V̂vec=**R**, and Û=‖**R**‖₂² exactly once P̂ has been obtained.

---

## 7. BranchedValue — backward propagation

Inputs: labeled stored tree 𝐒; leaf `CompletionStats`; frozen actor-type
value heads V^{θ_k}(x_p,σ); horizon λ ∈ [0,1]. Optional public moment output
heads can be instantiated, but this reducer emits no targets for them and no
registered production loss or branching controller consumes them.

Per-node working state is posterior mass `P̂ [P,X]` and posterior-weighted
return mass `M̂ [P,X]`. Conditional values are formed only when emitted.

```
initialize at every PRETERM leaf from CompletionStats
    P̂_p(x,t) = leaf posterior mass
    M̂_p(x,t) = P̂_p(x,t) * leaf conditional reward estimate
for σ in reverse topological order (children before parents), σ not a leaf:
    P̂(·|σ) = (1/K_σ) Σ_a P̂(·|σa)
    M̂(·,σ) = (1/K_σ) Σ_a M̂(·,σa)
    if o(σ) is player p:
        V̂(x,σ) = Cond(M̂_p(x,σ), P̂_p(x|σ))
        for each abstract child a:
            Q̂_a(x) = Cond(M̂_p(x,σa), P̂_p(x|σa))
            if K_σ > 1:
                sibling = Cond(Σ_b M̂_p(x,σb) - M̂_p(x,σa),
                               Σ_b P̂_p(x|σb) - P̂_p(x|σa))
                B̂_a(x) = λ sibling + (1−λ) V^{θ_k}(x,σ)
            else:
                B̂_a(x) = V^{θ_k}(x,σ)
            Â_a(x) = Q̂_a(x) - B̂_a(x)
        emit parent P̂, child P̂, Â, V̂, child labels and masks
        M̂_p(x,σ) ← λ M̂_p(x,σ) + (1−λ)P̂_p(x|σ)V^{θ_k}(x,σ)
    # chance nodes: aggregation only, no mixing, no targets
```

- Leave-one-out totals are computed as total-minus-child, hence linear in the
  number of edges rather than quadratic.
- Q̂ is transient and is stored only when an optional `emit_q=True` consumer asks
  for it. The shipped PPO objective consumes Â directly.
- Branch independence makes the sibling baseline action-independent for the
  target branch. It does not make the conditional ratio finite-sample unbiased.
- No auxiliary estimator channels or unused moment targets are active.

Complexity: one O(|𝐒| · P_max · X_max) pass; poker
$X_{\max}=\binom{4R}{2}\le1326$ → keep P̂/M̂ in fp32 on GPU, stream by tree if
memory-bound.

---

## 8. λ-returns, sequential form (reference semantics)

For a single trajectory Τ, node σa with p = o(σ), and n = number of p-owned decision
nodes ρ with σ ≺ ρ ≼ Τ (equivalently |Τ|_p − |σ|_p under the convention in §1;
define via `depth_p` arrays, not string operations):

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

## 9. Branching intensity modes

The active implementation exposes `training.mode` as `normal`,
`rao_blackwell`, or `branched_rao_blackwell`. Only the branched mode accepts a
`rollout.beta` map. A game-owned batched `beta_fn(states) -> float32[B]` returns
public-stage-dependent values in [1,∞) for poker's preflop/flop/turn/river and
optional chance-transition buckets. Every branched configuration explicitly
selects `rollout.branching_mode: fixed | dynamic`. Both modes instantiate and
train the public trace-variance T and cost C heads from iteration zero. `fixed`
uses `beta_fn` throughout and has no branching controller. `dynamic` uses the
same fixed schedule while `iteration < adaptive_kkt.warmup_iters`; only after
that boundary may rollout geometry and the KKT controller consume T/C.

---

## 10. Losses

### 10.1 Primitives (per node row; ratios w.r.t. the cached rollout policy θ_k)

ρ^θ(a, x_p) = π^θ(a|x_p,Σ) / π^{θ_k}(a|x_p,Σ); Â is emitted directly by §7.

- ℒ_PG = −ρ Â
- ℒ_clip = −min( ρ Â, clip(ρ, 1−ε, 1+ε) Â )
- ℒ_V = c_V (V^θ − V̂)²;  ℒ_V-clip with V_clip^θ = V^{θ_k} + clip(V^θ − V^{θ_k}, ±ε_V)
- ℒ_H = −c_H · Ent(π^θ(·|x_p,Σ))  (exact over the factored head, per adapter)
- Adapter KL and centered-logit operations remain reusable primitives. Exact
  legal-action `KL(new || reference)` and periodic immutable reference
  snapshots are implemented for normal, sequential-RB, and branched-RB modes;
  moment losses and NeuRD/R-NaD remain deferred.

### 10.2 What the gradient targets

Composite gradients approximate ∇ of J_p(π^θ_p, stopgrad(π^θ_{−p})) via the policy
gradient theorem for games (source Thm; constant 𝔼|τ|_p absorbed into learning rate).

### 10.3 `rb_ppo` composite (the shipped default)

For a minibatch of sampled decision nodes {Σ_b} with targets from §7:

L(θ) = (1/B) Σ_b [
   Σ_{a ∈ children(Σ_b)} (1/K_b) Σ_x P̂_b(x | child_a) · ℒ_clip(Â_b(x,a))
 + Σ_x P̂_b(x | parent) · ( ℒ_V(x) + ℒ_H(x) ) ]

Policy terms use child-branch posteriors; value and entropy terms use the parent
posterior. Duplicate public action labels remain separate abstract children.
Entropy follows `entropy_schedule`; all type sums are explicitly masked.

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
- **Model.** Two nets ($R^2$-class preflop / $\binom{4R}{2}$-combo-row
  postflop), pre-LN trunk,
  key-side reach prior added un-tempered, learned transforms of evaluator-supplied
  partition fields, factored type × DLogisticTick raise head,
  actor-row V head. Branched training requires public trace-variance T and cost
  C heads with registered targets and losses in both branching modes; only
  post-warmup dynamic rollout control consumes them. The suit-consistency loss
  is deferred. Compact H_B tokenization is the default (CONTRACTS §5-P7).
- **Preflop gauge symmetry.** Preflop net is class-native ($R^2$ classes; 169
  for $R=13$); deterministic
  combo-to-class aggregation may live in `cards.py`/`ranges.py` because it is a
  representation map, not a private-state law.
- **Private poker math.** Completion/equity coefficients, exact joint profiles,
  partition summaries, and fixed-showdown per-stratum winner shares use
  `poker_gpu_evaluator.pbg` under `archive/POKER_GPU_EVALUATOR_IO.md`. The sibling CPU
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
