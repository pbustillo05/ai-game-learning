# C00 — Mathematical and implementation-contract alignment

**Supersedes:** the branching/value/loss portions of `NOTATION.md` and
`CONTRACTS.md` that accompanied P09–P11.
**Files:** `docs/latex/core_math.tex`, `docs/latex/core_math_audit.md`,
`NOTATION.md`, `CONTRACTS.md`.

## Fresh-context bootstrap

This prompt is self-contained; do not paste or rely on earlier prompt text.
Before editing, run `git status --short`, preserve unrelated changes, and read
the four files listed above together with `PREAMBLE.md` and
`IMPLEMENTATION_NOTES.md`. Inspect the current P09–P11 implementation only to
identify terminology that the companion contract must mark as superseded.
Complete the document edits and validation in this session; do not stop after
proposing a plan.

## Goal

Freeze the corrected theory before changing production code. Keep edits small:
the paper continues to describe future extensions, while the implementation
companion clearly distinguishes the fixed-intensity PPO v1 from research ideas.

## Binding corrections

1. **Completion cut.** `tau^pre-c` is the largest prefix after which no
   player-owned node remains. Continuation contains zero or more public-chance
   draws followed by reward resolution. A Leduc state before its board is not
   pre-terminal because another betting round follows.
2. **Offspring law.** At every expanded node,
   `K=floor(beta)+Bernoulli(beta-floor(beta))`. Since `beta>=1`, `K>=1`.
3. **Labeled tree.** Store every abstract sampled child. Equal event labels may
   share immutable state payloads but never topology, descendants, RNG draws,
   or MC completion samples. Each edge has transition probability `1/K`.
4. **Bucket estimators.** Propagate posterior mass `P_hat` and joint target mass
   `M_hat=P_hat*V_hat`. State concisely that P/M are unbiased linear bucket
   estimators, whereas `Cond(M_hat,P_hat)` is generally biased at finite sample
   size, consistent for positive-mass buckets, and asymptotically unbiased for
   bounded rewards as effective samples grow.
5. **Leave one out.** Describe the sibling estimate as an action-independent
   control variate for the target branch, not an unbiased finite-sample value or
   advantage.
6. **Deferred work.** Mark Neyman allocation, public `Vvec/U` heads and losses,
   NeuRD/stored Q targets, and reference-policy KL as research extensions.

## Validation

- `rg` finds no active companion-contract reference to Poisson offspring,
  child deduplication, `IntensityController`, or `q_hat-old_v` advantage.
- Run `git diff --check`.
- Build without touching tracked artifacts:
  `mkdir -p /tmp/core-math-c00 && cd docs/latex && latexmk -pdf
  -halt-on-error -interaction=nonstopmode -outdir=/tmp/core-math-c00
  core_math.tex`.
- Do not alter theorem statements unrelated to these corrections.
