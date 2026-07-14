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

The remaining findings concern the scope of two named game-solving methods and
proof boundaries for the convergence claims. These do not undermine the basic
public-game, stationary-decision-law, or fixed-intensity branching setup.

Severity means:

- **Critical:** invalidates a central estimator, probability law, or theorem.
- **Major:** leaves a required object undefined, makes an algorithm
  non-executable, or materially changes the claimed mathematical meaning.
- **Moderate:** needs a missing hypothesis, proof boundary, or important
  qualification.
- **Minor:** local notation or exposition that does not change the main idea.

Line references are to the current version of `core_math.tex`.

## I. Moderate findings

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

**Severity: Moderate. The two sequential convergence lemmas and
`lem:branched-decision-law-convergence`.**

For a finite game, the almost-sure limits, ratio CLTs, and exponential bounds
in the two sequential lemmas are plausible under the standing regularity
assumptions, but their proof boundary remains unstated. The
rewritten branched lemma now makes the single-tree scaling explicit through
`beta_K = K bar-beta` and asserts only convergence in probability of the
decision marginal. Its proof should still establish the finite-depth law of
large numbers and control its random normalization. None of these statements
automatically extends to an ancestry-adaptive intensity.

## II. Sound design choices worth preserving

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

## III. Recommended repair order

1. Bound the claims made for the named game-solving methods.
2. State and prove the convergence results under the final sampling schemes.

## Current assessment

The public-game and posterior-predictive branching ideas remain coherent and
worth developing. The ordinary occupancy law and the insistence on labeled,
independent sibling branches are particularly solid foundations. The current
document should nevertheless be treated as a research draft: the named-method
and convergence claims need their assumptions and proof boundaries before
those extensions can be relied upon.
