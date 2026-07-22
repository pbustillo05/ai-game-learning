# Executive findings

## Verdict

The repository has strong mathematical coverage for the four training
semantics and representative tiny CUDA reachability for three of them, but it
does not yet meet the requested sole-path, NVIDIA-only, storage-lifecycle,
five-mode certification, or performance-gate standard.

## Canonical findings

| ID | Class / severity | Finding | Source finding IDs |
|---|---|---|---|
| C01 | Confirmed defect — blocker | Direct compact seal and a complete hard seal transaction are absent; production first creates `SampledTree` and then converts while both representations overlap. | `AUD-RSQ-001/002`, `AUD-RD-003`, seal part of `HIST-OOM-01`, `PERF-002/007` |
| C02 | Confirmed defect — blocker | Mandatory rank-local failures lack common terminal agreement and can strand peers in collectives. | `AUD-TL-001` |
| C03 | Fixed in d407d8e4f80e6e01b702ccbf707e7659bafc09e7 | BR has no authoritative response class, seat law, objective, stopping/cutoff, or estimator. | `A-01`, `EVAL-001`, authority part of `MODE-007` |
| C04 | Confirmed defect — high | Multi-player BR early-fold settlement is broken; BR also lacks a CUDA acceptance gate. | `TRACE-001`, `EVAL-002`, correctness/test part of `MODE-007` |
| C05 | Confirmed defect — high | Inferred/handwritten rollout schemas, preflop exact-width model carriers, and late logical-address reservation violate immutable natural storage. | `AUD-RSQ-003/004/006` |
| C06 | Confirmed defect — high | Branched child transient materialization scales with logical offspring instead of bounded physical subspans. | `AUD-RSQ-005`, part of `AUD-RSQ-006`, `PERF-003` |
| C07 | Confirmed defect — high | Queue policy/accounting and the common fixed/warmup scientific safeguard differ from authority. | `AUD-RSQ-007/008/009`, fixed/warmup `HIST-OOM-01`, `PERF-007` |
| C08 | Confirmed defect — high | Core training/BR accepts CPU and Gloo instead of requiring NVIDIA CUDA/NCCL. | `MODE-001`, `AUD-TL-005`, CUDA part of `EVAL-002` |
| C09 | Confirmed defect — high | Folded summaries use forbidden neutral `post_P`, duplicate per-pack ownership, and omit mandatory cache bytes from the ledger. | `POKER-001/002`, `AUD-MODEL-002`, `PERF-007` |
| C10 | Confirmed defect — high, plus authority-path gap | Production accepts evaluator v1 although the current sibling package/launcher requires v2; the named binding document is missing. | `POKER-003`; `POKER-008` documents the path gap |
| C11 | Static performance risk — high | Repeated CUDA-to-host controls remain in targets and evaluator/model/learner boundaries; magnitude is unmeasured. | `HIST-PERF-02`, `AUD-TL-002/004`, `POKER-005/006`, `PERF-001/005` |
| C12 | Confirmed defect — high | Performance gating is incomplete and fail-open: three labels, forced fixed branching, omitted BR/KKT, unvalidated mutation, missing-mode/metric tolerance. | `MODE-003/004`, `EVAL-010`, `PERF-008` |
| C13 | Scope defect — high | Installed CPU/MPS arena, inference, list-engine, and range products lie outside the five GPU workflows. | `MODE-002`, `POKER-004`, `EVAL-005/006`, `AUD-RD-001` |
| C14 | Scope defect — high | Checkpoint-specific variance implementations remain active parallel algorithms instead of archived evidence. | `EVAL-008/009`, `AUD-RD-002` |
| C15 | Policy defect — high | Checkpoint/config/model/evaluator/benchmark compatibility paths have no approved, time-bounded migration policy. | `AUD-RD-005`, `EVAL-007/011`, `MODE-008` |
| C16 | Confirmed inefficiency — high | BR completion computes/stages all players although only the selected owner is consumed. | `EVAL-003`, `PERF-006` |
| C17 | Optimization — unmeasured | CUDA `post_A` safely uses flop width on all postflop streets; street-aware physical packing could reduce turn/river work. | revised `POKER-009`; `AUD-MODEL-001` rejected as stated |
| C18 | Readiness defect — high; broken paths medium | Tests/docs preserve stale products and assert desired architecture as current; commands and paths are broken. | `AUD-RD-006/007`; revised `AUD-RD-008` |

## Preserved strengths

- KKT primitives and controller wiring survive with focused tests.
- Protected-root refill, ordinary-prefix targets, and controller exclusion
  survive statically; the end-to-end fixed-refill CUDA soak remains blocked.
- Cost-aware allocation, fused policy terms, sampled-row reference cache,
  biased SDPA, flat fp32 gradients, SUM synchronization, and direct backward
  survive statically and in focused tests.
- Page transactions roll back atomically for the owners they include.
- Tiny CUDA updates completed for sequential RB, fixed branched RB, and KKT;
  a tiny two-GPU exact-mass/SUM learner test passed.

These strengths do not clear the structural blockers or establish equal-work
performance preservation.
