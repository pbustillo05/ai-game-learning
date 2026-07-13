# Deep-dive theory/implementation audit

Audit date: 2026-07-13  
Audited worktree: the current, dirty worktree (including the in-progress sequential-profile rollout changes)  
Primary mathematical authority: `docs/theory/core_math.tex`, with `NOTATION.md` and `CONTRACTS.md` used as implementation companions

## Bottom line

The repository has a coherent implementation of one algorithmic family: posterior-weighted PPO over either one-path pre-terminal rollouts or fixed-intensity labeled public trees. The mass/moment backward pass, duplicate-child handling, path-weighted decision law, and the distinction between carried private profiles in sequential rollout and child-local posterior profiles in branching are substantially aligned with the corrected mathematical companion.

At audit time it did **not** meet the requested training contract. Since then,
`AUD-001` and `AUD-002` have been implemented: all three modes are explicit and
ordinary sampled-private terminal training is available. The remaining open
issues are tracked separately.

- the poker model now consumes lossless public history and implements the binding MHA geometry/incidence/folder paths (`AUD-006`–`AUD-008` resolved);
- dense postflop attention at the advertised batch size is infeasible;
- folded-summary semantics are aligned, but refresh caching and the broader evaluator boundary remain open under `AUD-009` and `AUD-010`;
- the synthetic dial is not a tested spectrum from Kuhn/Leduc to full no-limit poker.

Accordingly, the current code is suitable as a small synthetic/CPU research scaffold and as a partial correctness prototype for fixed branching. It is not yet a production-capable full 6-max trainer in any of the three requested modes.

## Audit packet

- [`findings.md`](findings.md): the 22 currently open findings and their response fields.
- [`resolved_findings.md`](resolved_findings.md): implemented and verified findings, preserving their original identifiers and history.
- [`mode_storage_compute.md`](mode_storage_compute.md): the required compute and lifetime/storage contract for normal, sequential Rao--Blackwell, and branched Rao--Blackwell training.
- [`validation_matrix.md`](validation_matrix.md): capability matrix, tests run, missing gates, and recommended implementation order.
- [`traceability.md`](traceability.md): pseudocode-to-module mapping, evaluator/model conformance, and file-by-file assessment.

## Highest-priority decisions

The issues that should be decided before lower-level optimization work are:

1. `AUD-013`: implement whole-iteration byte-budgeted streaming; for `AUD-005`, pass the local readiness gate and then measure Quest H100 tiers with intentionally staged jobs.
2. `AUD-009` and `AUD-010`: cache folded summaries at their exact invalidation events and decide whether a richer single-call evaluator DTO should replace the correct but redundant per-folder facade calls.
3. `AUD-011`: decide what “full NLHE” means for raise discretization and chip denomination.
4. `AUD-017` and `AUD-018`: define the intended Kuhn/Leduc/dial relationship and whether every poker variant must use the MHA design.

## Positive findings to preserve

- Fixed branching uses randomized rounding `floor(beta) + Bernoulli(frac(beta))` and retains duplicate public labels as distinct abstract children.
- Sequential rollout now samples one root private profile and carries it down its single path, avoiding needless posterior resampling.
- Fixed branching draws fresh posterior profiles per abstract child, preserving the intended sibling independence.
- Decision-node sampling uses stored path mass and therefore implements the length-biased empirical law.
- Backward propagation averages posterior mass and posterior-weighted return mass, conditionalizing only at target emission.
- Owner-only value bootstrapping follows the player-decision clock.
- Policy terms use child posteriors; value and entropy terms use parent posteriors.
- The sibling evaluator boundary is version-checked against `pbg-v2`, and local code does not reimplement showdown ranking or collision-conditioned completion.
- Existing CPU tests pass, and targeted poker parity tests now verify recomputation through all board streets and a fold edge.
