# Deep-dive audit of `core_math.tex`

## Scope and overall verdict

This is now a living audit of `core_math.tex`, restricted to issues deliberately left for later and local inconsistencies that remain after the review pass. It does not audit implementation files, cited papers, or claims against the literature.

The main mathematical spine is now coherent: the ideal decision law matches pooled rollout occupancy; the policy-gradient sampling law is frozen correctly; sequential and branched empirical laws have explicit convergence statements; pre-terminal completion is typed through a fixed public prefix; and branched private-state values propagate posterior-weighted moments.

No unresolved item below challenges the overall PBG, Rao--Blackwellization, or public-branching research program. Items 21--25 are deferred questions about method identification, support assumptions, and exposition. The later checklist contains local notation and algorithm-interface work that should be handled before treating the pseudocode as executable specification.

Severity below means:

- **Critical:** invalidates a definition, theorem, probability law, or central estimator.
- **Major:** changes the mathematical meaning or makes an algorithm ill-defined.
- **Moderate:** a missing hypothesis, unjustified claim, or consequential ambiguity.
- **Minor:** local notation, pseudocode, or editorial cleanup.

## I. Deferred mathematical and methodological issues

### 21. Separate policy-gradient information-state values from counterfactual regret values

**Severity: Major. Lines 233--249 and the NeuRD/R-NaD rows.**

For the policy-gradient theorem, posterior values conditioned on `(x_p,sigma)` and on-policy decision reach are appropriate. CFR-style and regret-dynamics statements usually use opponent-and-chance counterfactual reach, often with an unnormalized counterfactual value. The player's own reach cancels from the normalized Bayesian belief at a fixed information state, but the state-sampling/update weight still matters.

Therefore, before calling the displayed NeuRD loss or its R-NaD combination an exact game-solving method, derive its relationship to counterfactual values and counterfactual state reach in this PBG notation. Otherwise label it a NeuRD-style local update. Centered logits and an all-action value vector are semantically compatible with NeuRD; the missing step is the reach-weighted game-theoretic derivation.

There is also a concrete invariance problem in lines 233--248. Without clipping, `sum_a z(a)=0`, so the NeuRD loss is invariant under adding a common constant to all action values. Componentwise clipping of raw `Q(a)` destroys this payoff-translation invariance: the policy update can depend on an arbitrary choice of reward origin. Clip a centered advantage/regret, or define a translation-equivariant clipping rule, and compare that rule with the intended NeuRD convention.

### 22. Restrict the R-NaD/DeepNash identification to its actual game and update regime

**Severity: Major. Lines 335--373 and 417--430.**

The sign pattern in line 338 is consistent with adding `-eta log(pi_p/pi_ref,p)` for the acting player and the opposite regularizer for the opponent, but the usual R-NaD/DeepNash interpretation requires a two-player zero-sum setting and a prescribed reference-policy update/reset regime. The note assumes only `|P|=2`, not `R_1+R_2=0`, and leaves `ref(k)` unspecified because the original method is asynchronous.

Add the zero-sum hypothesis, define the reference update, and prove the synchronous analogue before labeling the row “R-NaD / DeepNash.” Until then, call it an R-NaD-inspired synchronous surrogate. The six-player poker specialization cannot inherit a two-player R-NaD convergence interpretation.

Six-player no-rake poker is constant-sum, but multiplayer constant-sum is not the two-player minimax setting. State this explicitly rather than using “zero-sum” as if it automatically supplied a scalar game value and two-player saddle-point theory.

### 23. Add support and absolute-continuity hypotheses once, globally

**Severity: Major. Lines 51--53, 205--268, 339--371, and all posterior conditionals.**

The strategy definition permits boundary probabilities, but the note later takes logarithms, divides by rollout probabilities, and evaluates KL divergences. State one of the following:

- all parameterized rollout, candidate, and reference policies have full support on every legal action; or
- every candidate/reference measure is absolutely continuous with respect to the sampling policy on the events used, with explicit conventions for zero terms.

Softmax policies make the first choice natural. Chance zeros and unreachable public histories still require positive-reach restrictions. The sampled NeuRD importance factor and R-NaD log ratio otherwise become undefined.

### 24. Treat “no other-player value head” as an architectural choice, not a theoretical constraint

**Severity: Moderate. Lines 193--201.**

Standard centralized-training/decentralized-execution methods may use privileged full hidden state in a critic even when the actor cannot. The ordinary simulator in this note also knows the sampled full profile. The proposed public vector head is reasonable and may be preferable, but the impossibility language should become a stated architecture/scale choice. Explain whether value heads are actor-observable, public-belief, or privileged critics.

### 25. Clarify which standard formulas are intentionally retained

**Severity: Moderate. Lines 205--270.**

Several nonstandard-looking choices are actually sound and should be documented rather than changed:

- Centering `log pi` uniformly over legal actions gives centered logits because the log-normalizer cancels.
- If `A ~ pi^{theta_k}`, then
  `E[rho log rho - rho + 1] = KL(pi^theta || pi^{theta_k})`; this is the target-to-old KL direction, not the reverse direction often monitored in PPO.
- Uniform-child sampling needs no inverse-`beta` weight for linear public-path expectations when offspring counts are independent of child outcomes and the conditional tree measure is used correctly. The problem is later per-tree normalization and nonlinear ratios, not the branching kernel itself.
- Conditioning training targets on future public events is a valid Rao--Blackwell operation when the entire gradient estimator is conditioned; it does not change the policy's inference-time information.

State these facts as lemmas or remarks. They distinguish deliberate alternatives from mistakes.

## IV. Deliberate conventions that should be retained

- Full public histories are used instead of assuming a lossy Markov state.
- The initial type distribution may be correlated, as required by card exclusion.
- Public chance may depend on the full hidden profile.
- Lowercase coordinates under `\mathbb P^\pi` and `\mathbb E^\pi` are intentional; uppercase notation is reserved for explicitly constructed random variables under unsuperscripted probability.
- The player-decision clock is the intended GAE clock.
- Curried all-private-state policy output is an architectural device enabling conditional summation.
- Target-to-reference KL, uniformly centered logits, and the public vector-value head are intentional choices.
- Public branching resamples temporary hidden profiles independently across siblings to preserve the public posterior-predictive kernel and sibling independence.

## Current assessment

The central constructions are now aligned. The remaining mathematical questions are items 21--25, intentionally deferred. The notation and algorithm-interface cleanup has been completed in `core_math.tex`.
