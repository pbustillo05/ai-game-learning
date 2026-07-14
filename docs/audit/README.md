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
- folded summaries use exact street/fold cache invalidation and the evaluator-owned `pbg-folder-v1` model-ready boundary (`AUD-009` and `AUD-010` resolved);
- branched rollout evaluates each unique player parent once, completion drops
  unused diagnostics after enforcing quality, and poker decision storage keeps
  only ragged live-seat reaches after completion (`AUD-012`, `AUD-014`, and
  `AUD-015` resolved);
- branched RB samples exactly one tree per rank/GPU and streams byte-limited
  completion batches through an upward `P/M` frontier; sequential samples are
  consistently described as trajectories (`AUD-013` resolved);
- the synthetic family is now an explicit two-axis algorithmic stress grid with
  exact Kuhn/Leduc anchors; it is not presented as literal convergence to NLHE.

Accordingly, the current code is suitable as a small synthetic/CPU research scaffold and as a partial correctness prototype for fixed branching. It is not yet a production-capable full 6-max trainer in any of the three requested modes.

## Audit packet

- [`findings.md`](findings.md): the 4 currently open findings and their response fields.
- [`resolved_findings.md`](resolved_findings.md): implemented and verified findings, preserving their original identifiers and history.
- [`mode_storage_compute.md`](mode_storage_compute.md): the required compute and lifetime/storage contract for normal, sequential Rao--Blackwell, and branched Rao--Blackwell training.
- [`validation_matrix.md`](validation_matrix.md): capability matrix, tests run, missing gates, and recommended implementation order.
- [`traceability.md`](traceability.md): pseudocode-to-module mapping, evaluator/model conformance, and file-by-file assessment.

## Highest-priority decisions

The issues that should be decided before lower-level optimization work are:

1. For `AUD-005`, pass the local readiness gate and then measure Quest H100
   tiers with intentionally staged jobs, preserving the byte-budgeted
   completion/frontier pipeline resolved in `AUD-013`.
2. Preserve the resolved synthetic/full-poker MHA comparison contract and use
   the `AUD-032` device-local gates during optimization work.

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
