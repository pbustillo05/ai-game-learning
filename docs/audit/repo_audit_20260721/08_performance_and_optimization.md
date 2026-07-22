# Performance and optimization

## Verdict

Performance readiness is fail-open (C12). No current measured regression is
available; the highest priorities are correctness/memory prerequisites, then
removal of static synchronization risks, then equal-work measurement, and only
then optional throughput work.

## Risk and opportunity classification

| Item | Classification | Evidence / next action |
|---|---|---|
| Direct compact seal and complete ledger | Confirmed memory-accounting defect | C01; remove whole-tree source/destination overlap before production-size runs |
| Logical-offspring child waves | Static algorithmic/memory risk | C06; bounded staging with physical-plan invariance |
| Queue/folded/completion closure | Confirmed accounting defect | C07/C09; schema-charge all mandatory live owners |
| Target scalar extraction | High-confidence static performance risk | C11; vectorized bounded segmented targets plus no-host-sync gate |
| Evaluator/model/learner Python CUDA predicates | Static performance risk | Historical workload averaged 71.8 evaluator calls, 176.6 cache builds, 573.1 learner subspans, and 935 policy batches/iteration; profiler/sync-debug required |
| Learner from `537700f` | Unverified preservation | Reproduce exact SDPA/AMP, cache cadence, 53/11-equivalent allocation, and 232 steps |
| Street-aware `post_A` | Unmeasured optimization | C17. At $D=52$, widths are 1,176/1,128/1,081; conservative flop padding is safe. Historical 97% late-street and 18.67-billion-pair data do not prove current gain |
| Preflop natural width | Confirmed structural waste | $H/K=1326/169=7.846$; fix as part of C05 |
| BR selected completion | Confirmed structural waste | C16; at $P=6,H=1326$, `p_hat`/`v_hat` excess is 53,040 bytes per leaf, about 506 MiB at 10,000 leaves |
| Reference-cache streamlining | Artifact-backed opportunity | Historical mean 14.095 s and 176.6 calls/iteration; cache only consumed KL outputs |
| Communication overlap | Bounded opportunity | Historical SUM mean 6.953 s; preserve one SUM and global finite/clip/update law |
| Remove `empty_cache()` | Live optimization, benchmark first | Distinguish reusable reserve from live storage before removal |
| CUDA graphs/compile | Deferred optimization | First stabilize shapes, remove syncs, and bound child waves |

The current worst-width postflop path is conservative. The rejected
`AUD-MODEL-001` claim must not reappear as an admission bug or claimed 11–13%
current speedup.

## Required staged gates

1. Static schema/accounting and CUDA sync-debug gates.
2. Tiny five-semantic CUDA smoke with explicit device and conservative cap.
3. One-GPU 2,048/2,560-root sequential target benchmark against 1.055/1.313 s,
   separating completion from recurrence and using repeated warmed samples.
4. Two-GPU 232-step learner reproduction against the documentary 0.2993
   s/step result, clearly labeling absent raw historical evidence.
5. Deterministic seal scaling and branched physical-plan invariance before
   large trees.
6. Fixed/KKT closure tests with injected pressure, no reroll, and exact
   nominal/executed correction—never the law-changing diagnostic cutoff.
7. Street-specific and selected-BR-completion microbenchmarks.
8. Only then, matched end-to-end full HU 32 bb runs with normalized work units.

Every CUDA stage must preflight complete memory, begin/end with `nvidia-smi`,
select idle GPUs explicitly, and stop rather than OOM. The detailed conservative
matrix is in [batch 11](batches/11_performance_regressions_optimization.md).
