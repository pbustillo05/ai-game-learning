# Audit of `core_math.tex`

## Scope and verdict

This audit reviews only `docs/theory/core_math.tex` as it currently stands. It
does not compare the note with `CONTRACTS.md`, `NOTATION.md`, implementation
code, tests, cited papers, or the existing PDF. The previous audit was used
only as a model for the structure of this document and has otherwise been
discarded.

The note has a strong conceptual spine: it isolates a finite public-action
Bayesian game, defines occupancy-weighted decision laws, separates ordinary,
Rao--Blackwellized, and branched rollouts, preserves duplicate children in the
sampled tree, and distinguishes the fixed-intensity PPO implementation target
from more speculative extensions.

It is not yet mathematically ready to serve as a theorem-level or executable
specification. Two central constructions are currently invalid as written:

1. the player-clock lambda-return indexing gives the wrong target even at a
   final player decision and becomes singular in one leave-one-branch formula;
2. the branched continuation law forgets the sampled-tree identity of an edge,
   even though that identity is essential when equal public labels occur in
   distinct branches.

The adaptive-intensity section also mixes random offspring counts with
continuous design variables, and its log-head targets do not identify the
quantities the heads are said to estimate. Finally, the backward pseudocode has
several missing or mismatched objects, so it cannot yet be read as an algorithm
implementing the displayed mathematics.

Severity means:

- **Critical:** invalidates a central estimator, probability law, or theorem.
- **Major:** leaves a required object undefined, makes an algorithm
  non-executable, or materially changes the claimed mathematical meaning.
- **Moderate:** needs a missing hypothesis, proof boundary, or important
  qualification.
- **Minor:** local notation or exposition that does not change the main idea.

Line references are to the current 1,361-line version of `core_math.tex`.

## I. Critical findings

### 1. The player-clock lambda returns have an off-by-one error and are internally inconsistent

**Severity: Critical. Lines 387--399, 633--643, and 864--922.**

At a player node `sigma` owned by `p`, the quantity
`|tau \ sigma|_p` includes the action taken at `sigma`. Therefore, at the last
decision by `p`, it equals one. The sequential lambda-return equation then gives the terminal reward
coefficient `lambda`, not one. In particular, at `lambda = 0` the target at a
last decision is zero. That contradicts the claimed usual player-clock GAE
return in Lemma 4.

The terminal coefficient should be indexed by the number of *future* decisions
by `p`, namely `|tau \ sigma|_p - 1`, with the bootstrap sum indexed
consistently. The same issue appears in the Rao--Blackwellized return and in the
analytic branched targets.

The leave-one-branch expression is more visibly inconsistent. Its second
expectation permits `S = s` through the condition `s <= S`, while its exponent
is `|Sigma \ sigma|_p - 1`. At `S = s`, that exponent is `-1`, producing
`lambda^{-1}`. At the same time, the prefactor uses
`1 + |tau \ sigma|_p`, even though the current decision is already counted in
`|tau \ sigma|_p`.

This needs one explicit clock convention followed through all three settings.
A robust derivation is to define the ordered future `p`-decision nodes after
`sigma a`, write the lambda return as a sum over that sequence, and only then
rewrite it using count notation. The backward recurrence in
`BranchedValue` should be proved equivalent to the corrected formula; it
currently appears closer to the intended recurrence than the displayed
analytic equations.

### 2. The branched joint law loses labeled-edge identity

**Severity: Critical. Lines 764--810, especially 790--803.**

The note correctly insists that two children with the same public event label
remain different sampled-tree nodes, with independent descendants and Monte
Carlo completions. But after sampling the specific edge
`S A`, `eq:branched-continuation-law` conditions the continuation only on the public event
`Sigma A`. If sibling edges have the same event label, that condition mixes
their distinct subtrees. The resulting continuation is not necessarily a
descendant of the sampled edge whose target is being computed.

This also makes the sequence of random choices in Definition 7 ambiguous. A
pre-terminal trajectory is first introduced through a sampled leaf, then
reintroduced after sampling `P`, `S`, and `A`; the length-biased display in
lines 793--797 and the conditional law in lines 799--803 do not define one
unambiguous joint distribution.

Define the law directly on sampled-tree objects. For example: conditional on a
realized labeled tree, sample a leaf by uniform edge mass, sample one player
decision on that exact root-to-leaf path, retain its exact sampled edge, and
then sample the hidden profile and terminal completion conditional on that
leaf's public pre-terminal label. Equivalent factorizations are fine, but every
conditional continuation used by a per-child target must condition on
`S A`, not merely on `Sigma A`. Only after this joint law is fixed should the
node, player, action, and continuation marginals be derived.

## II. Major findings

### 3. The pre-terminal cut is not fully defined

**Severity: Major. Lines 67--69 and 552--583.**

The phrase "largest prefix ending in no more than one node in
`T union chance-nodes`" is not a formal property of a prefix. The following
sentence gives the intended last-player-action construction, but it assumes
that every terminal trajectory contains a player action. The stopping set
`T^{pre-c}` used by `SamplePreTerminal` is never defined.

Define the class of games for which the cut exists, define the map on every
terminal trajectory, and set
`T^{pre-c} := {tau^{pre-c} : tau in T}` (or the intended alternative).
Also state whether different terminal trajectories may share the same cut and
whether a terminal state itself can be pre-terminal.

### 4. Support, positivity, and derivative hypotheses are missing

**Severity: Major. Lines 51--53, 155--199, 253--318, 359--384, 585--611, and all posterior ratios.**

Strategies are allowed on the boundary of a simplex, while later formulas take
logarithms, divide by rollout probabilities, and use KL divergences. The ideal
decision law and all three empirical decision laws also divide by an expected
or realized player-decision count. The convention that a zero-probability
conditional expectation equals zero does not define `log 0`, an importance
ratio with a zero denominator, or an index law with zero total length.

State global hypotheses such as:

- at least one player decision occurs on every sampled trajectory (or give an
  explicit zero-length convention);
- all differentiated rollout, candidate, and reference policies have full
  support on legal actions, or the required absolute-continuity conditions;
- posterior formulas are only asserted on positive-reach public histories;
- the policy-gradient theorem is with respect to a differentiable parameter
  `theta_p`, rather than the currently undefined object `nabla_{pi_p}`.

These hypotheses are also needed to justify the ratio CLTs and concentration
bounds as stated.

### 5. The Rao--Blackwell tower identity drops the private-state condition

**Severity: Major. Lines 613--631.**

`eq:rao-blackwell-q-identity` has `x_p, sigma a` on the left but conditions the outer
expectation on `sigma a` alone on the right. The correct tower identity keeps
the private-state event in both levels, for example

`E[ E[R_p | X_p=x_p, Tau^{pre-c}] | X_p=x_p, sigma a ]`.

As written, the right-hand side averages over the posterior of `X_p`, while the
left-hand side is a value for a fixed type. This is a mathematical error, not
just shorthand. Nearby `eq:sequential-preterminal-estimates` also writes the intended posterior target
as `P(x_p, tau | Tau_i^{pre-c})`; it should be a marginal such as
`P(X_p=x_p | Tau_i^{pre-c})`.

### 6. The unnormalised-moment lemma is undefined and contradicted by the following paragraph

**Severity: Major. Lines 924--929 and 1123--1133.**

Lemma 10 says that "all of the above estimators are unbiased when multiplied by
`1_{x_p}`", but no random indicator event or target expectation is specified.
The estimators are already indexed by a fixed `x_p`. Moreover, the next
paragraph explicitly says that the leave-one-branch baseline is not a
finite-sample unbiased value or advantage estimator.

The valid statement should be limited to linear quantities. For a bucket
`X_p=x_p`, define a mass estimator and a posterior-weighted target-moment
estimator, and prove those two quantities unbiased under the corrected labeled
tree law. Their ratio is generally a biased self-normalized estimator, though
it may be consistent for positive-mass buckets. The same distinction is needed
for every use of `Cond(m,w)` and for the leave-one-branch control variate.

### 7. The analytic branched targets do not represent the conditional lambda-return they claim to estimate

**Severity: Major. Lines 859--929.**

Beyond the clock error in Finding 1, `eq:branched-targets` replaces an expected
sum of future bootstrap terms by a product of an expected count and an expected
value at a sampled future node. Such a factorization works only if that future
node is sampled from the corresponding count-weighted law under the *same
conditioning*. Here the count is conditioned on `x_p`, while the displayed law
of `S` is not. In the `V` expression, even the expected count drops the `x_p`
condition.

Write each target first as one conditional expectation of the complete finite
sum. If an occupancy-sampled-node representation is desired, define its law
conditional on `X_p=x_p` and prove the change of measure. This will also expose
whether the analytic definition and `BranchedValue` recurrence actually match.

### 8. The KKT section mixes three incompatible offspring-count objects

**Severity: Major. Lines 934--1035.**

The rollout definition uses a real intensity `beta(sigma)`, randomized rounding
produces a random integer `N(s)`, and the KKT derivation treats `N(sigma)` as a
deterministic continuous optimization variable. The displayed expectations in
lines 948--973 still contain bare `N`, so they are neither conditional
realized-tree identities nor unconditional formulas. Under randomized
rounding, size terms involve `E[N]=beta`, whereas variance terms generally
involve `E[1/N]`, not `1/beta`.

The fixed-point formula then defines `beta_k(sigma)` using ancestor quantities
`N(alpha)`. Those counts do not exist before the tree is sampled, and
`N(alpha)` is ambiguous when several sampled-tree nodes share public label
`alpha`. Consequently the formula does not define a function on the declared
domain `S \ T` that can be passed to `SampleTree`.

Choose one of two models:

- a deterministic/continuous mean-field relaxation written entirely in terms
  of design variables, clearly labeled as an approximation; or
- an adaptive per-sampled-node intensity whose domain includes the exact
  sampled ancestry, followed by new terminal-law, independence, and cost
  calculations.

### 9. The log-variance and log-size targets do not identify the stated heads

**Severity: Major. Lines 1018--1069 and 1179--1234.**

`W` is defined as the logarithm of a conditional trace covariance, but
`W-hat` is the logarithm of an epsilon-shifted sample covariance. Even if the
unlogged sample covariance is unbiased, its logarithm is not: Jensen's
inequality gives a systematic bias. Line 1051 additionally says the logged
quantity is unbiased for the *unlogged* covariance, which is dimensionally and
mathematically false.

Similarly, squared-error regression on `log(C-hat)` learns a conditional mean
of a log realized-size target, not the logarithm of the expected subtree size
used to define `S`. These may still be useful heuristic targets, but then the
heads and KKT rule must be defined around those heuristic estimands. Otherwise
use estimators and losses designed for the unlogged moments and transform only
at inference, with the resulting bias handled explicitly.

### 10. `PreTerminalMomentMC` gives the public value moment the wrong shape

**Severity: Major. Lines 1071--1093.**

The public vector value introduced in line 940 has one scalar coordinate per
player. In line 1091, however,
`P-hat(-_p) * V-hat(-_p)` remains indexed by every private state `x_p`; it is not
a scalar player coordinate. The required player component is the sum over
private states,

`V-hat_public,p = sum_{x_p} P-hat(x_p) V-hat_p(x_p)`,

or, more directly, the paired sample average of `R_p`. The paired construction
also makes clear which posterior-weighted moments are unbiased before any
ratio is taken.

### 11. The backward and training pseudocode is not executable as written

**Severity: Major. Lines 1135--1293.**

The principal interface failures are:

- `SampleTree` does not return the leaf reach caches later passed as `r` to
  `BranchedValue`, and `r` is undefined in the training algorithm;
- `V_2-hat`, required for the variance target, is never defined or propagated;
- `N(sigma bullet)` applies a sampled-tree count to a public-state expression;
- `C-hat` is sometimes written without an argument, and the overwrite uses
  undefined `C(s)` instead of `C-hat(s)`;
- the target-table assignment contains a stray leading `s`;
- the table does not visibly store the per-child posterior
  `P-hat(x_p | s a)` that the minibatch loss later reads;
- the declared tree count `N` is unused: only one tree is sampled, while the
  convergence lemma requires pooling independent trees with the `Z_m` weights;
- the sampled-tree line has unmatched descriptive structure and does not say
  how multiple trees would be pooled or selected.

Resolve the probability law and estimator definitions before repairing this
pseudocode. Then give every stored field a name and shape, specify leaf
initialization, and type-check each backward assignment against the output
table consumed by the minibatch loss.

## III. Moderate findings

### 12. The named game-solving methods outrun what is established in the note

**Severity: Moderate. Lines 242--318 and 400--495.**

The R-NaD construction assumes only two players, leaves `ref(k)` unspecified,
and supplies no zero-sum or reference-update hypothesis. It should not be
labeled an exact "R-NaD / DeepNash" method within this self-contained note
without those assumptions and a derivation of the synchronous analogue. The
later description of these losses as deferred research extensions is the right
level of confidence and should be reflected in the earlier table.

There is also an internal invariance issue in the NeuRD loss. Because centered
logits sum to zero, the unclipped loss is invariant to adding a constant to all
action values. Componentwise clipping of raw `Q(b)` destroys that invariance,
so a policy update can depend on the arbitrary reward origin. Clip a centered
advantage/regret or state that this is a deliberate non-invariant surrogate.

Finally, `eq:rnad-modified-return` uses uppercase `Tau` inside a definition whose argument
is lowercase `tau`.

### 13. The convergence lemmas need their proof boundary stated

**Severity: Moderate. Lines 371--384, 597--611, and 813--841.**

For a finite game, the almost-sure limits and ratio CLTs are plausible after
the positivity assumptions in Finding 4. The exponential TV bounds and the
branched result additionally rely on bounded offspring, nonvanishing
normalizers, independence across whole trees, and a fixed public intensity.
State these hypotheses in the lemmas and either provide proofs or label the
results as claims to be proved. They do not automatically extend to the
ancestry-adaptive intensity suggested later.

### 14. "Stationary distribution" is the wrong term for the sampled-node law

**Severity: Moderate. Lines 772--810.**

The uniform-edge process on a finite rooted tree is absorbing; its stationary
distributions live on leaves. `eq:branched-stationary-node-law` is a normalized player-node
occupancy measure, not a stationary distribution of that Markov chain. Calling
it an occupancy law will make the derivation and its length normalization much
clearer.

## IV. Sound design choices worth preserving

- Full public histories, correlated initial private types, and hidden-profile-
  dependent public chance are appropriate for the stated game class.
- The ideal decision law correctly exposes the length/occupancy weighting that
  ordinary rollout minibatches approximate.
- Pooling whole rollout trees with weights proportional to their conditional
  expected player-decision counts is the right structure; per-tree normalized
  laws should not simply be averaged uniformly.
- Public posterior-predictive child sampling and independently resampled
  temporary hidden profiles are the right ingredients for sibling independence.
- Duplicate public event labels must remain distinct labeled-tree children.
- Propagating posterior mass together with posterior-weighted reward moments is
  the right way to delay nonlinear conditional ratios.
- The warning that nonlinear losses of conditional-mean targets define a new
  surrogate, rather than the conditional expectation of the original loss, is
  important and should remain.
- Separating fixed-intensity PPO v1 from adaptive-intensity, NeuRD, reference-KL,
  and public-head research extensions is a useful implementation boundary.

## V. Recommended repair order

1. Define the pre-terminal cut, global support assumptions, and parameterized
   derivative convention.
2. Fix the player-clock indexing and prove one lambda-return recurrence before
   duplicating it across sequential, Rao--Blackwellized, and branched forms.
3. Rewrite the branched probability space in a single generative order on exact
   sampled-tree nodes and edges; then rederive the terminal, occupancy, and
   continuation laws.
4. State unbiasedness only for linear bucket masses and weighted moments, and
   separately state bias/consistency properties of ratios and leave-one-branch
   baselines.
5. Decide whether adaptive branching is a deterministic mean-field relaxation
   or a history-adaptive random process, then redo the cost/variance objective
   in that model.
6. Define estimands for the variance and size heads that match their training
   targets.
7. Only after those steps, repair and type-check the pseudocode and prove the
   three empirical-law convergence results under the final sampling scheme.

## Current assessment

The public-game and posterior-predictive branching ideas remain coherent and
worth developing. The ordinary occupancy law and the insistence on labeled,
independent sibling branches are particularly solid foundations. The current
document should nevertheless be treated as a research draft: its lambda-return
formulas and branched continuation law need correction before downstream value
targets, variance guidance, or pseudocode can be relied upon.
