# Open numbered findings register

## Reading guide

Severities:

- **Critical:** a requested training mode is absent, a shipped path fails, or a production configuration is structurally infeasible.
- **Major:** material algorithm/model mismatch, correctness risk, or large unavoidable performance/memory loss.
- **Moderate:** incomplete operational support, dead configuration, documentation drift, or a narrower optimization/test gap.

This file contains only unresolved findings. After a recommendation has been
implemented and verified, move its entire entry to
[`resolved_findings.md`](resolved_findings.md). Preserve its original `AUD-*`
identifier, response, and implementation note; do not renumber remaining
findings or reuse identifiers.

`Response:` fields are for the user's decision or clarification and may be
filled before implementation.

Model recommendations are cost-conscious routing judgments, not claims about
published model pricing or guarantees. This register uses the following
working rubric: `5.5` for mechanical documentation/cleanup, `5.6 Luna` for
contained implementation, `5.6 Terra` for cross-module systems work, and
`5.6 Sol` for mathematically or architecturally high-risk work. Thinking
intensity is `low`, `medium`, or `high`.

## A. Training-mode and core-algorithm findings

## B. Game-family, theory, operations, and validation findings

### AUD-021 — `core_math.tex` lambda-return displays are off by one relative to code

**Severity:** Major specification defect, implementation currently follows the corrected companion  
**Modes:** Normal, sequential RB  
**Evidence:** Equations `sequential-lambda-return` and `rao-blackwell-lambda-return` use terminal exponent `|Tau ∖ sigma|_p`, which includes the current decision under the document's depth convention. The correct player-clock exponent is the number of downstream decisions by the same player, `|Tau|_p - |sigma|_p - 1`. `NOTATION.md` explicitly normalizes this typo, and the backward implementation applies one lambda mix at each downstream owner decision, which is the corrected behavior.

**Consequence:** A future direct transcription of the authoritative TeX would disagree with the current code and standard player-clock GAE, especially visible at the last decision where the terminal reward should have exponent zero.

**Recommendation:** Correct the equations in `core_math.tex` rather than relying on a companion normalization. Add closed-form trajectory tests for zero, one, and multiple downstream same-player decisions at lambda 0, intermediate lambda, and 1.

**Model recommendation:** `5.6 Sol` with `high` thinking — an authoritative mathematical correction should be derived against code and closed-form edge cases, not patched mechanically.

**Response:** Ignore for now; core_math is changing a lot anyways


---

### AUD-022 — Global policy-support assumptions remain unresolved

**Severity:** Major theoretical/robustness issue  
**Modes:** All  
**Evidence:** The existing `core_math_audit.md` already records missing full-support/absolute-continuity hypotheses. Code partly enforces numerical support by clamping probabilities before logs (`eps=1e-12` poker, finite softmax synthetic), but the mathematical strategy space permits zeros and PPO divides current by cached rollout probability. Chance/posterior conditionals also depend on reachable positive-mass events.

**Consequence:** Theorems and loss formulas are not globally well-defined as written, while implementation clamps silently alter exact zero-support semantics.

**Recommendation:** Add one binding support convention to the TeX/contracts: full-support parameterized legal-action policies with a documented probability floor, plus positive-reach restrictions for conditionals. Test that stored actions always have finite old log probability and document the bias introduced by any clamp.

**Model recommendation:** `5.6 Sol` with `high` thinking — this establishes global mathematical assumptions and must reconcile exact theory with numerical implementation bias.

**Response:** Ignore for now, core_math is changing a lot anyways


---

### AUD-024 — Branch and memory budgets are not preflighted

**Severity:** Major  
**Modes:** Branched RB  
**Evidence:** `node_budget` is a hard per-tree capacity multiplier, and overflow is detected only when appending children. There is no expected-node calculation from beta/horizon, high-quantile safety estimate, free-memory query, completion-leaf estimate, or staged dry run. The implementation correctly refuses to truncate, but that only turns budget mismatch into a late failed iteration.

**Consequence:** Fixed beta can explode the tree or allocate massive unused headroom. On a shared GPU server, late capacity failure wastes work and increases OOM risk.

**Recommendation:** Add an analytical and empirical planner: estimate offspring-by-stage tree size, leaf count, target bytes, and model workspace; choose chunk/tree counts under a configured memory fraction; run a tiny calibration before scale-up; log realized branching and peak memory. Keep loud overflow as a final invariant.

**Model recommendation:** `5.6 Sol` with `high` thinking — safe preflight planning combines stochastic tree growth, byte accounting, calibration, and shared-GPU safeguards.

**Response:** Ignore; branching math has been reworked 


---
